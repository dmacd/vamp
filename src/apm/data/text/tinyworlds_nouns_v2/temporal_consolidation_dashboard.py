"""Disk-backed live dashboard for the nouns-v2 temporal study."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
from threading import Thread
import time
from typing import Literal, TypeAlias
from urllib.parse import parse_qs, unquote, urlparse

from apm.data.text.tinyworlds_nouns_v2.contracts import canonical_json_bytes
from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_contracts import (
    PROGRESS_ROW_FORMAT,
    STUDY_ID,
)
from apm.continual.artifacts import (
    ChainedJsonlLedger,
    atomic_write,
    load_canonical_json,
)


JobStatus: TypeAlias = Literal["pending", "running", "complete", "failed"]
_ALLOWED_ARTIFACT_SUFFIXES = frozenset(
    {".csv", ".dot", ".html", ".json", ".jsonl", ".md", ".svg"}
)


@dataclass(frozen=True, slots=True)
class StudyJob:
    """One predeclared unit-bearing dashboard job."""

    job_id: str
    phase: str
    description: str
    total: int
    unit: str
    estimated_seconds: float

    def __post_init__(self) -> None:
        if (
            not self.job_id
            or not self.phase
            or not self.description
            or type(self.total) is not int
            or self.total <= 0
            or not self.unit
            or not math.isfinite(self.estimated_seconds)
            or self.estimated_seconds <= 0.0
        ):
            raise ValueError("dashboard job fields are invalid")

    @property
    def is_long(self) -> bool:
        """Whether this job requires its own five-minute progress estimate."""
        return self.estimated_seconds > 300.0

    def as_record(self) -> dict[str, object]:
        """Return a JSON-ready immutable job descriptor."""
        return {
            "description": self.description,
            "estimated_seconds": self.estimated_seconds,
            "is_long": self.is_long,
            "job_id": self.job_id,
            "phase": self.phase,
            "total": self.total,
            "unit": self.unit,
        }


@dataclass(frozen=True, slots=True)
class DashboardServer:
    """One running loopback dashboard and its daemon serving thread."""

    server: ThreadingHTTPServer
    thread: Thread
    url: str

    def stop(self) -> None:
        """Stop serving and wait for the loopback thread to exit."""
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=10.0)


class ProgressRecorder:
    """Write progress events and reconstruct an atomic dashboard projection."""

    def __init__(
        self,
        work_directory: str | Path,
        contract_sha256: str,
        jobs: Sequence[StudyJob],
    ) -> None:
        self.work_directory = Path(work_directory)
        self.contract_sha256 = contract_sha256
        self.jobs = tuple(jobs)
        if len({job.job_id for job in self.jobs}) != len(self.jobs):
            raise ValueError("dashboard job IDs must be unique")
        self.work_directory.mkdir(parents=True, exist_ok=True)
        self.ledger = ChainedJsonlLedger(
            self.work_directory / "progress.jsonl",
            PROGRESS_ROW_FORMAT,
        )
        self.status_path = self.work_directory / "status.json"
        self._publish_projection()

    def update(
        self,
        job_id: str,
        completed: int,
        *,
        status: JobStatus = "running",
        elapsed_seconds: float | None = None,
        metrics: Mapping[str, float | int | str | bool | None] | None = None,
        detail: Mapping[str, object] | None = None,
        ignore_stale_replay: bool = False,
    ) -> dict[str, object]:
        """Persist one monotonic progress update and refresh `status.json`.

        A resumable orchestration pass may replay already-published training
        artifacts from its first sub-job before it reaches the ledger's prior
        high-water mark.  ``ignore_stale_replay`` makes only those older
        callbacks idempotent; the default remains fail-closed so an accidental
        backward update is still rejected.
        """
        job = next((candidate for candidate in self.jobs if candidate.job_id == job_id), None)
        if job is None:
            raise KeyError(f"unknown dashboard job: {job_id}")
        if (
            type(completed) is not int
            or not 0 <= completed <= job.total
            or status not in ("pending", "running", "complete", "failed")
        ):
            raise ValueError("dashboard progress or status is invalid")
        previous = tuple(
            row for row in self.ledger.rows if row.get("job_id") == job_id
        )
        prior_completed = int(previous[-1]["completed"]) if previous else 0
        if completed < prior_completed:
            if ignore_stale_replay:
                return previous[-1]
            raise ValueError("dashboard progress cannot move backward")
        if (
            ignore_stale_replay
            and previous
            and completed == prior_completed
            and (
                (
                    previous[-1].get("status") == "complete"
                    and status != "complete"
                )
                or previous[-1].get("status") == status == "running"
            )
        ):
            return previous[-1]
        if status == "complete" and completed != job.total:
            raise ValueError("completed dashboard jobs require full coverage")
        elapsed = (
            float(elapsed_seconds)
            if elapsed_seconds is not None
            else float(previous[-1].get("elapsed_seconds", 0.0)) if previous else 0.0
        )
        if not math.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError("dashboard elapsed time must be finite and nonnegative")
        rate = completed / elapsed if elapsed > 0.0 else 0.0
        eta = (job.total - completed) / rate if rate > 0.0 else job.estimated_seconds
        row = self.ledger.append(
            {
                "completed": completed,
                "contract_sha256": self.contract_sha256,
                "detail": dict(detail or {}),
                "elapsed_seconds": elapsed,
                "eta_seconds": max(0.0, eta),
                "job_id": job_id,
                "metrics": dict(metrics or {}),
                "rate": rate,
                "status": status,
                "timestamp_unix": time.time(),
                "total": job.total,
                "unit": job.unit,
            }
        )
        self._publish_projection()
        return row

    def fail(self, job_id: str, message: str, elapsed_seconds: float) -> dict[str, object]:
        """Mark one job failed while preserving its last completed count."""
        prior = tuple(row for row in self.ledger.rows if row.get("job_id") == job_id)
        completed = int(prior[-1]["completed"]) if prior else 0
        return self.update(
            job_id,
            completed,
            status="failed",
            elapsed_seconds=elapsed_seconds,
            detail={"error": message},
        )

    def snapshot(self) -> dict[str, object]:
        """Build the complete current projection from authenticated events."""
        latest = {
            str(row["job_id"]): row
            for row in self.ledger.rows
            if row.get("job_id") in {job.job_id for job in self.jobs}
        }
        job_rows = []
        for job in self.jobs:
            event = latest.get(job.job_id)
            completed = int(event["completed"]) if event else 0
            status: JobStatus = str(event["status"]) if event else "pending"  # type: ignore[assignment]
            job_rows.append(
                {
                    **job.as_record(),
                    "completed": completed,
                    "detail": dict(event.get("detail", {})) if event else {},
                    "elapsed_seconds": float(event.get("elapsed_seconds", 0.0)) if event else 0.0,
                    "eta_seconds": float(event.get("eta_seconds", job.estimated_seconds)) if event else job.estimated_seconds,
                    "fraction": completed / job.total,
                    "metrics": dict(event.get("metrics", {})) if event else {},
                    "rate": float(event.get("rate", 0.0)) if event else 0.0,
                    "status": status,
                }
            )
        estimated_total = sum(job.estimated_seconds for job in self.jobs)
        weighted_complete = sum(
            job.estimated_seconds * row["fraction"]
            for job, row in zip(self.jobs, job_rows)
        )
        running = tuple(row for row in job_rows if row["status"] == "running")
        failed = tuple(row for row in job_rows if row["status"] == "failed")
        phase = str(running[0]["phase"]) if running else "failed" if failed else "complete" if all(row["status"] == "complete" for row in job_rows) else "pending"
        return {
            "contract_sha256": self.contract_sha256,
            "event_count": self.ledger.next_sequence,
            "format": f"{STUDY_ID}-dashboard-snapshot-v1",
            "jobs": job_rows,
            "latest_event_sha256": self.ledger.tail_hash,
            "overall_eta_seconds": sum(float(row["eta_seconds"]) for row in job_rows if row["status"] not in ("complete", "failed")),
            "overall_fraction": weighted_complete / estimated_total,
            "phase": phase,
            "schema_version": 1,
            "study_id": STUDY_ID,
            "updated_unix": max((float(row["timestamp_unix"]) for row in self.ledger.rows), default=0.0),
        }

    def _publish_projection(self) -> None:
        atomic_write(self.status_path, canonical_json_bytes(self.snapshot()))


def start_dashboard_server(
    work_directory: str | Path,
    artifact_directory: str | Path,
    *,
    first_port: int = 8765,
    last_port: int = 8775,
) -> DashboardServer:
    """Start the live GET-only dashboard on the first free loopback port."""
    work = Path(work_directory).resolve()
    artifacts = Path(artifact_directory).resolve()
    handler = _handler_type(work, artifacts)
    server: ThreadingHTTPServer | None = None
    for port in range(first_port, last_port + 1):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), handler)
            break
        except OSError:
            continue
    if server is None:
        raise RuntimeError("no temporal dashboard port is free from 8765 through 8775")
    server.daemon_threads = True
    thread = Thread(target=server.serve_forever, name="temporal-dashboard", daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    return DashboardServer(server, thread, f"http://{host}:{port}/")


def publish_frozen_dashboard(
    path: str | Path,
    snapshot: Mapping[str, object],
) -> Path:
    """Publish a self-contained non-polling dashboard snapshot."""
    payload = _dashboard_html(snapshot=dict(snapshot), live=False).encode("utf-8")
    return atomic_write(path, payload)


def _handler_type(work_directory: Path, artifact_directory: Path):
    class Handler(BaseHTTPRequestHandler):
        server_version = "TinyWorldsTemporalDashboard/1"

        def do_GET(self) -> None:  # noqa: N802
            """Serve one allowlisted dashboard resource."""
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/":
                    self._send(
                        HTTPStatus.OK,
                        _dashboard_html(snapshot=None, live=True).encode("utf-8"),
                        "text/html; charset=utf-8",
                    )
                elif parsed.path == "/healthz":
                    status = _load_status(work_directory)
                    self._send_json({"ok": True, "phase": status.get("phase", "unknown")})
                elif parsed.path == "/api/v1/snapshot":
                    self._send_json(_load_status(work_directory), etag=True)
                elif parsed.path == "/api/v1/events":
                    raw = parse_qs(parsed.query).get("after", ["-1"])[0]
                    after = int(raw)
                    ledger = ChainedJsonlLedger(
                        work_directory / "progress.jsonl",
                        PROGRESS_ROW_FORMAT,
                    )
                    self._send_json({"events": list(ledger.after(after))}, etag=True)
                elif parsed.path.startswith("/artifacts/"):
                    self._send_artifact(parsed.path.removeprefix("/artifacts/"))
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
            except (KeyError, TypeError, ValueError):
                self.send_error(HTTPStatus.BAD_REQUEST)

        def do_POST(self) -> None:  # noqa: N802
            """Reject state-changing requests."""
            self.send_error(HTTPStatus.METHOD_NOT_ALLOWED)

        do_PUT = do_POST
        do_DELETE = do_POST
        do_PATCH = do_POST

        def log_message(self, format_string: str, *arguments: object) -> None:
            return

        def _send_json(self, value: Mapping[str, object], *, etag: bool = False) -> None:
            payload = canonical_json_bytes(dict(value))
            tag = f'"{sha256(payload).hexdigest()}"'
            if etag and self.headers.get("If-None-Match") == tag:
                self.send_response(HTTPStatus.NOT_MODIFIED)
                self.end_headers()
                return
            self._send(
                HTTPStatus.OK,
                payload,
                "application/json; charset=utf-8",
                etag=tag if etag else None,
            )

        def _send_artifact(self, raw_relative: str) -> None:
            relative = Path(unquote(raw_relative))
            if relative.is_absolute() or ".." in relative.parts:
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            target = (artifact_directory / relative).resolve()
            if (
                artifact_directory not in target.parents
                or not target.is_file()
                or target.suffix not in _ALLOWED_ARTIFACT_SUFFIXES
            ):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            media = {
                ".csv": "text/csv; charset=utf-8",
                ".dot": "text/vnd.graphviz; charset=utf-8",
                ".html": "text/html; charset=utf-8",
                ".json": "application/json; charset=utf-8",
                ".jsonl": "application/x-ndjson; charset=utf-8",
                ".md": "text/markdown; charset=utf-8",
                ".svg": "image/svg+xml",
            }[target.suffix]
            self._send(HTTPStatus.OK, target.read_bytes(), media)

        def _send(
            self,
            status: HTTPStatus,
            payload: bytes,
            media_type: str,
            *,
            etag: str | None = None,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", media_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'none'; img-src 'self' data:; style-src 'unsafe-inline'; "
                "script-src 'unsafe-inline'; connect-src 'self'; base-uri 'none'; "
                "form-action 'none'; frame-ancestors 'none'",
            )
            if etag is not None:
                self.send_header("ETag", etag)
            self.end_headers()
            self.wfile.write(payload)

    return Handler


def _load_status(work_directory: Path) -> dict[str, object]:
    path = work_directory / "status.json"
    return load_canonical_json(path) if path.is_file() else {
        "format": f"{STUDY_ID}-dashboard-snapshot-v1",
        "jobs": [],
        "phase": "starting",
        "study_id": STUDY_ID,
    }


def _dashboard_html(
    *,
    snapshot: Mapping[str, object] | None,
    live: bool,
) -> str:
    embedded = json.dumps(snapshot, sort_keys=True, separators=(",", ":")) if snapshot is not None else "null"
    poll = "true" if live else "false"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TinyWorlds temporal consolidation</title>
<style>
:root{{--bg:#0b1220;--panel:#131e30;--ink:#edf3fb;--muted:#a9b7cb;--line:#30415a;--blue:#55a7ff;--green:#55d69e;--red:#ff7272;--amber:#ffc766}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.48 system-ui,sans-serif}} main{{max-width:1500px;margin:auto;padding:24px}} h1{{font-size:1.75rem;margin:.1rem 0}} h2{{font-size:1.15rem;margin:0 0 12px}} .muted{{color:var(--muted)}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px;margin:16px 0}} .card{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px}} .wide{{grid-column:1/-1}} progress{{width:100%;height:18px;accent-color:var(--blue)}} table{{width:100%;border-collapse:collapse}} th,td{{text-align:left;padding:8px;border-bottom:1px solid var(--line);vertical-align:top}} th{{color:var(--muted)}} .scroll{{overflow:auto;max-height:55vh}} .status{{font-weight:700}} .complete{{color:var(--green)}} .failed{{color:var(--red)}} .running{{color:var(--amber)}} code{{overflow-wrap:anywhere}} details{{margin-top:8px}} a{{color:#8fc5ff}} .metric{{display:inline-block;margin:4px 14px 4px 0}} .bar-label{{display:flex;justify-content:space-between;gap:12px}} @media(max-width:600px){{main{{padding:12px}}th,td{{padding:6px}}}}
</style></head><body><main>
<header><h1>TinyWorlds nouns-v2 · log-t temporal consolidation</h1><div id="identity" class="muted">Loading authenticated run state…</div></header>
<section class="grid"><article class="card"><h2>Overall</h2><div id="overall"></div></article><article class="card"><h2>Current phase</h2><div id="phase"></div></article></section>
<section class="card"><h2>Experiment matrix and long-job estimates</h2><div class="scroll"><table><thead><tr><th>Phase / job</th><th>Status</th><th>Exact work</th><th>Progress</th><th>Rate</th><th>ETA</th><th>Live estimates</th></tr></thead><tbody id="jobs"></tbody></table></div></section>
<section class="grid"><article class="card"><h2>Blocked hierarchy</h2><div id="blocked" class="muted">Pending</div></article><article class="card"><h2>Round-robin hierarchy</h2><div id="round_robin" class="muted">Pending</div></article></section>
<section class="card"><h2>Live metrics and plots</h2><div id="metrics" class="muted">Provisional values appear as evaluation rows arrive.</div><div id="plots"></div></section>
<details class="card"><summary>Method and interpretation</summary><p>The router sees only each story's midpoint prefix. Suffix losses and suffix-oracle choices are evaluator-only. Mixed temporal chunks have no unique noun label, so noun-support hit is descriptive rather than route accuracy. Final IID and independent-noun results are offline endpoints, not historical causal curves.</p></details>
<details class="card"><summary>Provenance and event log</summary><pre id="events" class="scroll muted"></pre></details>
</main><script>
const embedded={embedded}; const live={poll}; let etag=null; let lastSeq=-1; let latestUpdatedUnix=0; const eventHistory=[];
const esc=x=>String(x??'').replace(/[&<>\"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}}[c]));
const duration=s=>!Number.isFinite(s)?'—':s<60?`${{s.toFixed(0)}} s`:s<3600?`${{(s/60).toFixed(1)}} min`:`${{(s/3600).toFixed(1)}} h`;
const bytes=n=>!Number.isFinite(Number(n))?'—':Number(n)<1048576?`${{(Number(n)/1024).toFixed(1)}} KiB`:`${{(Number(n)/1048576).toFixed(1)}} MiB`;
function renderFreshness(){{const target=document.getElementById('freshness');if(!target)return;if(!live){{target.textContent='frozen snapshot';return}}const age=Math.max(0,Date.now()/1000-latestUpdatedUnix);target.className=age<15?'complete':'failed';target.textContent=age<15?'live':`paused/stale · last update ${{duration(age)}} ago`;}}
function render(s){{latestUpdatedUnix=Number(s.updated_unix||0);document.getElementById('identity').innerHTML=`Contract <code>${{esc(s.contract_sha256||'pending')}}</code> · event ${{esc(s.event_count||0)}} · <span id="freshness"></span>`;renderFreshness();
 document.getElementById('overall').innerHTML=`<div class="bar-label"><span>${{(100*(s.overall_fraction||0)).toFixed(1)}}%</span><span>ETA ${{duration(s.overall_eta_seconds||0)}}</span></div><progress max="1" value="${{s.overall_fraction||0}}"></progress>`;
 const jobs=s.jobs||[]; document.getElementById('jobs').innerHTML=jobs.map(j=>`<tr><td><strong>${{esc(j.phase)}}</strong><br>${{esc(j.description)}}${{j.is_long?' <span title="Projected above five minutes">⏱</span>':''}}</td><td class="status ${{esc(j.status)}}">${{esc(j.status)}}</td><td>${{esc(j.completed)}} / ${{esc(j.total)}} ${{esc(j.unit)}}</td><td><progress max="1" value="${{j.fraction||0}}"></progress></td><td>${{j.rate?esc(j.rate.toFixed(2)):'—'}} ${{esc(j.unit)}}/s</td><td>${{duration(j.eta_seconds)}}</td><td>${{Object.entries(j.metrics||{{}}).map(([k,v])=>`<span class="metric">${{esc(k)}}: <strong>${{esc(v)}}</strong></span>`).join('')}}</td></tr>`).join('');
 const phaseJobs=jobs.filter(j=>j.phase===s.phase);const phaseWeight=phaseJobs.reduce((a,j)=>a+j.estimated_seconds,0);const phaseFraction=s.phase==='complete'?1:phaseWeight?phaseJobs.reduce((a,j)=>a+j.estimated_seconds*j.fraction,0)/phaseWeight:0;document.getElementById('phase').innerHTML=`<div class="bar-label"><span class="status ${{esc(s.phase)}}">${{esc(s.phase)}}</span><span>${{(100*phaseFraction).toFixed(1)}}%</span></div><progress max="1" value="${{phaseFraction}}"></progress>`;
 for(const order of ['blocked','round_robin']){{const details=jobs.filter(j=>j.job_id===`stack-${{order}}`).map(j=>j.detail||{{}}).at(-1);if(details)document.getElementById(order).innerHTML=`Arrival ${{esc(details.arrival||0)}} · active ${{esc(details.active_chunk_count||0)}} / archived ${{esc(details.archive_chunk_count||0)}}<br>Live ${{bytes(details.active_adapter_bytes)}} · archive ${{bytes(details.archive_adapter_bytes)}}<br>Carry: ${{esc((details.carry||[]).join(', ')||'none')}}<br><code>${{esc((details.active_intervals||[]).join(', '))}}</code>`;}}
 const allMetrics=jobs.flatMap(j=>Object.entries(j.metrics||{{}}).map(([k,v])=>[j.job_id,k,v])); if(allMetrics.length)document.getElementById('metrics').innerHTML=allMetrics.slice(-24).map(([j,k,v])=>`<span class="metric"><span class="muted">${{esc(j)}} · ${{esc(k)}}</span> <strong>${{esc(v)}}</strong></span>`).join('');
 renderPlot();
}}
function renderPlot(){{const points=eventHistory.filter(e=>Number.isFinite(Number((e.metrics||{{}}).story_nll))).slice(-120);if(points.length<2){{document.getElementById('plots').innerHTML='<span class="muted">Waiting for two evaluation estimates…</span>';return}}const values=points.map(e=>Number(e.metrics.story_nll));const lo=Math.min(...values),hi=Math.max(...values);const xy=values.map((v,i)=>`${{10+i*580/(values.length-1)}},${{105-(v-lo)*90/Math.max(hi-lo,1e-9)}}`).join(' ');document.getElementById('plots').innerHTML=`<svg role="img" aria-label="Live provisional suffix story NLL" viewBox="0 0 600 120"><rect width="600" height="120" fill="#0b1220"/><polyline points="${{xy}}" fill="none" stroke="#55a7ff" stroke-width="3"/><text x="10" y="16" fill="#a9b7cb" font-size="12">provisional story NLL ${{values.at(-1).toFixed(4)}} · n=${{esc(points.at(-1).completed||0)}}</text></svg>`;}}
async function refresh(){{const headers=etag?{{'If-None-Match':etag}}:{{}};const r=await fetch('/api/v1/snapshot',{{headers}});if(r.status===200){{etag=r.headers.get('ETag');const snapshot=await r.json();if(lastSeq<0)lastSeq=Math.max(-1,Number(snapshot.event_count||0)-501);render(snapshot)}} const e=await fetch(`/api/v1/events?after=${{lastSeq}}`);if(e.ok){{const rows=(await e.json()).events||[];if(rows.length){{lastSeq=rows.at(-1).sequence;eventHistory.push(...rows);if(eventHistory.length>500)eventHistory.splice(0,eventHistory.length-500);document.getElementById('events').textContent=eventHistory.slice(-30).map(x=>JSON.stringify(x)).join('\\n');renderPlot()}}}}}}
if(embedded)render(embedded); if(live){{refresh();setInterval(refresh,2000);setInterval(renderFreshness,1000)}}
</script></body></html>"""


__all__ = [
    "DashboardServer",
    "ProgressRecorder",
    "StudyJob",
    "publish_frozen_dashboard",
    "start_dashboard_server",
]

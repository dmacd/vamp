"""Deterministic Markdown, HTML, CSV, and plot publication for TRACE."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import base64
import csv
from html import escape
from io import BytesIO, StringIO
from itertools import accumulate
import json
from pathlib import Path
from typing import TYPE_CHECKING

from apm.continual.artifacts import (
    ChainedJsonlLedger,
    atomic_write,
    file_sha256,
    load_canonical_json,
)
from apm.continual.trace.artifacts import validate_artifact_directory
from apm.continual.trace.evaluation import EVALUATION_LEDGER_FORMAT
from apm.continual.trace.jobs import JobLedger
from apm.continual.trace.lineage import build_hierarchy, leaf_node
from apm.continual.trace.metrics import headline_metrics
from apm.continual.trace.scheduler import eta_snapshot

if TYPE_CHECKING:
    from matplotlib.axes import Axes


@dataclass(frozen=True, slots=True)
class ReportResult:
    """Published report paths and deterministic source-row count."""

    markdown_path: Path
    html_path: Path
    csv_path: Path
    parquet_path: Path
    calibration_csv_path: Path
    lineage_svg_path: Path
    merge_plot_path: Path
    merge_diagnostics_csv_path: Path
    evaluation_rows: int


def build_report(
    run_directory: str | Path,
    *,
    interim: bool = False,
    name: str | None = None,
) -> ReportResult:
    """Build human-readable and machine-readable reports from completed evaluations."""
    run = Path(run_directory)
    if not interim:
        _require_distinct_leaves(run)
    records = tuple(
        value
        for path in sorted((run / "evaluations").rglob("result.json"))
        for value in (json.loads(path.read_text(encoding="utf-8")),)
        if type(value) is dict and value.get("format") == "trace-evaluation-result-v1"
    )
    reports = run / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    stem = name or ("interim" if interim else "primary")
    if Path(stem).name != stem or not stem:
        raise ValueError("TRACE report names must be simple path components")
    score_rows = _score_rows(records)
    csv_path = atomic_write(reports / f"{stem}-scores.csv", _score_csv(score_rows).encode())
    parquet_path = atomic_write(
        reports / f"{stem}-scores.parquet",
        _score_parquet(score_rows),
    )
    diagnostics = _merge_diagnostics(run)
    calibration = _calibration_rows(run)
    calibration_csv_path = atomic_write(
        reports / f"{stem}-retrained-parent-calibration.csv",
        _calibration_csv(calibration).encode(),
    )
    diagnostics_csv_path = atomic_write(
        reports / f"{stem}-merge-diagnostics.csv",
        _diagnostics_csv(diagnostics).encode(),
    )
    lineage_svg_path = atomic_write(
        reports / f"{stem}-lineage.svg",
        _lineage_svg(run).encode(),
    )
    merge_plot_bytes = _merge_diagnostic_plot(diagnostics)
    merge_plot_path = atomic_write(
        reports / f"{stem}-merge-diagnostics.png",
        merge_plot_bytes,
    )
    summaries = _method_summaries(records)
    plot_data = _summary_plot(summaries)
    archive_bytes = sum(
        path.stat().st_size
        for root_name in ("leaves", "derived", "merge_cache", "baselines")
        for path in (run / root_name).rglob("*")
        if path.is_file()
    )
    markdown = _markdown(
        run,
        records,
        summaries,
        diagnostics,
        calibration,
        archive_bytes,
        interim,
        lineage_svg_path.name,
        merge_plot_path.name,
    )
    markdown_path = atomic_write(reports / f"{stem}-report.md", markdown.encode())
    html = _html(
        markdown,
        records,
        summaries,
        archive_bytes,
        plot_data,
        base64.b64encode(merge_plot_bytes).decode(),
        lineage_svg_path.read_text(encoding="utf-8"),
        interim,
    )
    html_path = atomic_write(reports / f"{stem}-report.html", html.encode())
    atomic_write(
        reports / f"{stem}-manifest.json",
        (
            json.dumps(
                {
                    "archive_bytes": archive_bytes,
                    "calibration_sha256": file_sha256(calibration_csv_path),
                    "evaluation_rows": len(records),
                    "format": "trace-report-manifest-v1",
                    "html_sha256": file_sha256(html_path),
                    "interim": interim,
                    "lineage_svg_sha256": file_sha256(lineage_svg_path),
                    "markdown_sha256": file_sha256(markdown_path),
                    "merge_diagnostics_sha256": file_sha256(diagnostics_csv_path),
                    "merge_plot_sha256": file_sha256(merge_plot_path),
                    "parquet_sha256": file_sha256(parquet_path),
                    "scores_sha256": file_sha256(csv_path),
                },
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode(),
    )
    return ReportResult(
        markdown_path,
        html_path,
        csv_path,
        parquet_path,
        calibration_csv_path,
        lineage_svg_path,
        merge_plot_path,
        diagnostics_csv_path,
        len(records),
    )


def _require_distinct_leaves(run: Path) -> None:
    arrivals = load_canonical_json(run / "manifests" / "arrivals.json")
    identities = tuple(str(value) for value in arrivals["arrival_ids"])
    if len(identities) != 40:
        raise ValueError("final TRACE report requires all 40 registered leaves")
    hashes = []
    for identity in identities:
        directory = run / "leaves" / identity
        validate_artifact_directory(directory)
        hashes.append(file_sha256(directory / "adapter.safetensors"))
    if len(set(hashes)) != 40:
        raise ValueError("the 40 immutable TRACE leaves do not have distinct content hashes")


def _method_summaries(
    records: tuple[dict[str, object], ...],
) -> tuple[dict[str, object], ...]:
    grouped: dict[tuple[str, str], dict[tuple[int, int], float]] = defaultdict(dict)
    for record in records:
        router_scores = record.get("router_scores")
        if type(router_scores) is not dict:
            continue
        method = str(record.get("condition") or record.get("policy_hash"))
        task_index, stage = int(record["task_index"]), int(record["stage"])
        for router, score in router_scores.items():
            grouped[(method, str(router))][(task_index, stage)] = float(score)
    summaries: list[dict[str, object]] = []
    for (method, router), matrix in sorted(grouped.items()):
        if len(matrix) == 36:
            metrics = headline_metrics(matrix)
            summaries.append(
                {
                    "condition": method,
                    "router": router,
                    **metrics.as_record(),
                }
            )
    return tuple(summaries)


def _score_rows(
    records: tuple[dict[str, object], ...],
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "condition": str(record.get("condition", "")),
            "policy_hash": str(record.get("policy_hash", "")),
            "router": str(router),
            "score": float(score),
            "split": split,
            "stage": int(record["stage"]),
            "task": str(record["task"]),
            "task_index": int(record["task_index"]),
        }
        for record in records
        for split, score_field in (
            ("test", "router_scores"),
            ("validation", "validation_router_scores"),
        )
        for scores in (record.get(score_field),)
        if type(scores) is dict
        for router, score in sorted(scores.items())
    )


def _score_csv(rows: tuple[dict[str, object], ...]) -> str:
    stream = StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(
        (
            "condition",
            "policy_hash",
            "split",
            "router",
            "stage",
            "task_index",
            "task",
            "score",
        )
    )
    for row in rows:
        writer.writerow(
            tuple(
                row[name]
                for name in (
                    "condition",
                    "policy_hash",
                    "split",
                    "router",
                    "stage",
                    "task_index",
                    "task",
                    "score",
                )
            )
        )
    return stream.getvalue()


def _score_parquet(rows: tuple[dict[str, object], ...]) -> bytes:
    try:
        import pyarrow as arrow
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("TRACE report publication requires pyarrow") from error
    schema = arrow.schema(
        (
            ("condition", arrow.string()),
            ("policy_hash", arrow.string()),
            ("split", arrow.string()),
            ("router", arrow.string()),
            ("stage", arrow.int64()),
            ("task_index", arrow.int64()),
            ("task", arrow.string()),
            ("score", arrow.float64()),
        )
    )
    table = arrow.Table.from_pylist(list(rows), schema=schema)
    sink = arrow.BufferOutputStream()
    parquet.write_table(table, sink, compression="zstd", write_statistics=True)
    return sink.getvalue().to_pybytes()


def _markdown(
    run: Path,
    records: tuple[dict[str, object], ...],
    summaries: tuple[dict[str, object], ...],
    diagnostics: tuple[dict[str, object], ...],
    calibration: tuple[dict[str, object], ...],
    archive_bytes: int,
    interim: bool,
    lineage_filename: str,
    merge_plot_filename: str,
) -> str:
    heading = "TRACE Log-t VAMP Interim Report" if interim else "TRACE Log-t VAMP Primary Report"
    runtime = _runtime_summary(run)
    comparison_gaps = _vamp_comparison_gaps(records, summaries)
    validation_policy_rows = _validation_policy_rows(records)
    reuse_records = tuple(
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((run / "reports").glob("artifact-reuse-*.json"))
    )
    lines = [
        f"# {heading}",
        "",
        *(["**PRELIMINARY — RUN PAUSED BEFORE COMPLETION**", ""] if interim else []),
        *(
            [
                "Only complete eight-stage matrices receive OP/forgetting/BWT values; partial methods are shown as incomplete.",
                "",
            ]
            if interim
            else []
        ),
        f"This report currently contains {len(records):,} completed task-stage evaluation rows.",
        "",
        "## Required primary result table",
        "",
        "| Method | Router | OP | Forgetting | Signed BWT | Negative-only BWT | Training presentations | Replay presentations | Final live LoRAs | Task-free |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        *_primary_table_rows(run, records, summaries),
        "",
        "Task-aware and answer-oracle routing are diagnostics; they are not task-free deployment results.",
        "",
        "### VAMP routing and storage diagnostics",
        "",
        "| Method | Router | Role | OP | Forgetting | Signed BWT | Negative-only BWT |",
        "|---|---|---|---:|---:|---:|---:|",
        *(
            (
                f"| {row['condition']} | {row['router']} | "
                f"{'task-free' if row['router'] in ('prompt_nll', 'frozen_prompt_centroid') else 'diagnostic'} | "
                f"{float(row['op']):.3f} | {float(row['forgetting']):.3f} | "
                f"{float(row['bwt_signed']):.3f} | "
                f"{float(row['bwt_clipped_negative_only']):.3f} |"
            )
            for row in summaries
            if str(row["condition"]).startswith("vamp_")
        ),
        "",
        "## Sequential and joint-IID gaps",
        "",
        "The registered comparisons are `OP_VAMP − OP_sequential` and `OP_joint − OP_VAMP`.",
        "",
        "| VAMP method | Task-free router | VAMP OP | Sequential OP | VAMP − sequential | Joint-IID OP | Joint-IID − VAMP |",
        "|---|---|---:|---:|---:|---:|---:|",
        *(
            (
                f"| {row['condition']} | {row['router']} | {float(row['vamp_op']):.3f} | "
                f"{float(row['sequential_op']):.3f} | {float(row['vamp_vs_sequential']):.3f} | "
                f"{float(row['joint_op']):.3f} | {float(row['joint_iid_vs_vamp']):.3f} |"
            )
            for row in comparison_gaps
        ),
        *(["| _Pending_ | — | — | — | — | — | — |"] if not comparison_gaps else []),
        "",
        "## Validation-only policy comparison",
        "",
        "Core scale, repair fraction, rank, or repair-optimizer variants are compared here before any test-set interpretation.",
        "",
        "| Policy condition | Router | Final validation OP | Best for router |",
        "|---|---|---:|---:|",
        *(
            f"| {row['condition']} | {row['router']} | {float(row['op']):.3f} | "
            f"{'yes' if row['selected'] else 'no'} |"
            for row in validation_policy_rows
        ),
        *(
            ["| _Pending_ | — | — | — |"]
            if not validation_policy_rows
            else []
        ),
        "",
        "## Memory accounting",
        "",
        f"The reproducibility archive currently occupies {archive_bytes / 1024**2:.2f} MiB. "
        "This is not the algorithmic live-state claim. A completed 40-arrival VAMP hierarchy "
        "contains seven live adapters plus router metadata and the selected repair reservoir.",
        "",
        "## Protocol notes",
        "",
        "Task-free prompt-NLL and frozen-centroid routers never receive answer tokens. "
        "Task-aware and answer-oracle values are diagnostic and are reported separately.",
        "",
        "## Consolidation diagnostics",
        "",
        f"{len(diagnostics):,} completed merge records are available in the merge-diagnostics CSV. "
        "Retained spectral energy, child cosine, level, task composition, merge time, and repair work "
        "are kept separately from task-stage scores.",
        "",
        f"![Merge diagnostics]({merge_plot_filename})",
        "",
        "### Retrained-parent calibration",
        "",
        "| Interval | Task | Candidate | Validation score |",
        "|---|---|---|---:|",
        *(
            f"| {row['interval']} | {row['task']} | {row['candidate']} | {float(row['score']):.3f} |"
            for row in calibration
        ),
        *(
            ["| _Pending_ | — | — | — |"]
            if not calibration
            else []
        ),
        "",
        f"![Authenticated temporal lineage]({lineage_filename})",
        "",
        "## Artifact-reuse acceptance",
        "",
        *(
            (
                f"Policy `{record['policy_hash']}`: leaf training steps reused "
                f"{record['leaf_training_steps_reused_percent']}%; leaf hashes unchanged "
                f"`{str(record['leaf_adapter_hashes_unchanged']).lower()}`; new gradient work "
                f"`{record['new_gradient_work']}`."
            )
            for record in reuse_records
        ),
        *(
            ["No derived-policy reuse run has completed yet."]
            if not reuse_records
            else []
        ),
        "",
        "## Runtime and resume",
        "",
        f"Scheduler state: `{JobLedger(run / 'manifests' / 'jobs.jsonl').state_counts()}`.",
        "",
        f"Observed GPU worker utilization: {runtime['gpu_utilization_percent']:.1f}% across "
        f"{runtime['session_hours']:.2f} session-hours; recorded GPU work: "
        f"{runtime['gpu_worker_hours']:.2f} worker-hours.",
        "",
        f"Observed training throughput: {runtime['training_presentations_per_second']:.2f} "
        f"presentations/s, {runtime['training_tokens_per_second']:.2f} tokens/s, and "
        f"{runtime['optimizer_steps_per_second']:.3f} optimizer steps/s.",
        "",
        f"Observed evaluation throughput: {runtime['evaluation_cases_per_second']:.2f} "
        f"candidate cases/s, {runtime['prompt_prefill_tokens_per_second']:.2f} "
        f"prompt-prefill tokens/s, and {runtime['generation_tokens_per_second']:.2f} "
        "generated tokens/s.",
        "",
        f"Per-task candidate throughput: `{runtime['evaluation_task_throughput']}`.",
        "",
        f"Remaining job families: `{runtime['remaining_jobs']}`.",
        "",
        f"ETA: `{eta_snapshot(run)}`.",
        "",
        f"Resume with `python -m apm.continual.trace.cli resume --run {run}`.",
        "",
    ]
    return "\n".join(lines)


def _runtime_summary(run: Path) -> dict[str, object]:
    session_seconds = sum(
        float(record["elapsed_seconds"])
        for path in (run / "state" / "sessions").glob("*-summary.json")
        for record in (json.loads(path.read_text(encoding="utf-8")),)
    )
    timing_path = run / "logs" / "job_timings.jsonl"
    timing_rows = (
        ChainedJsonlLedger(timing_path, "trace-job-timing-v1").rows
        if timing_path.is_file()
        else ()
    )
    gpu_seconds = sum(
        float(row["elapsed_seconds"])
        for row in timing_rows
        if row.get("resource") == "gpu"
    )
    training_metrics = tuple(
        json.loads(path.read_text(encoding="utf-8"))
        for root_name in ("leaves", "derived", "baselines")
        for path in (run / root_name).rglob("train_metrics.json")
    )
    presentations = sum(int(record["presentations"]) for record in training_metrics)
    training_seconds = sum(float(record["elapsed_seconds"]) for record in training_metrics)
    candidate_rows = tuple(
        row
        for path in (run / "evaluations").rglob("*candidates.jsonl")
        for row in ChainedJsonlLedger(path, EVALUATION_LEDGER_FORMAT).rows
    )
    evaluation_seconds = sum(float(row["case_wall_seconds"]) for row in candidate_rows)
    prompt_tokens = sum(int(row["prompt_tokens"] or 0) for row in candidate_rows)
    generated_tokens = sum(int(row["generated_tokens"] or 0) for row in candidate_rows)
    task_throughput = {
        task: {
            "candidate_cases_per_second": len(rows) / task_seconds
            if task_seconds
            else 0.0,
            "generated_tokens_per_second": sum(
                int(row["generated_tokens"] or 0) for row in rows
            )
            / task_seconds
            if task_seconds
            else 0.0,
            "prompt_prefill_tokens_per_second": sum(
                int(row["prompt_tokens"] or 0) for row in rows
            )
            / task_seconds
            if task_seconds
            else 0.0,
        }
        for task in sorted({str(row["task"]) for row in candidate_rows})
        for rows in (
            tuple(row for row in candidate_rows if str(row["task"]) == task),
        )
        for task_seconds in (sum(float(row["case_wall_seconds"]) for row in rows),)
    }
    optimizer_steps = sum(int(record["optimizer_steps"]) for record in training_metrics)
    training_tokens = sum(int(record["tokens"]) for record in training_metrics)
    ledger = JobLedger(run / "manifests" / "jobs.jsonl")
    remaining: dict[str, int] = defaultdict(int)
    for status in ledger.statuses:
        if status.state != "COMPLETE":
            remaining[status.spec.kind] += 1
    return {
        "evaluation_cases_per_second": len(candidate_rows) / evaluation_seconds
        if evaluation_seconds
        else 0.0,
        "evaluation_task_throughput": task_throughput,
        "generation_tokens_per_second": generated_tokens / evaluation_seconds
        if evaluation_seconds
        else 0.0,
        "gpu_utilization_percent": min(
            100.0,
            100.0 * gpu_seconds / (2 * session_seconds),
        )
        if session_seconds
        else 0.0,
        "gpu_worker_hours": gpu_seconds / 3600,
        "optimizer_steps_per_second": optimizer_steps / training_seconds
        if training_seconds
        else 0.0,
        "prompt_prefill_tokens_per_second": prompt_tokens / evaluation_seconds
        if evaluation_seconds
        else 0.0,
        "remaining_jobs": dict(sorted(remaining.items())),
        "session_hours": session_seconds / 3600,
        "training_presentations_per_second": presentations / training_seconds
        if training_seconds
        else 0.0,
        "training_tokens_per_second": training_tokens / training_seconds
        if training_seconds
        else 0.0,
    }


def _primary_table_rows(
    run: Path,
    records: tuple[dict[str, object], ...],
    summaries: tuple[dict[str, object], ...],
) -> tuple[str, ...]:
    summary_by_key = {
        (str(row["condition"]), str(row["router"])): row for row in summaries
    }
    specifications = (
        ("frozen_base", "direct", 0, 0, 0, "yes"),
        ("seq_lora_reference", "direct", 20_000, 0, 1, "yes"),
        ("seq_lora_40", "direct", 20_000, 0, 1, "yes"),
        ("joint_iid_lora", "direct", 20_000, 0, 1, "yes"),
        ("taskwise_lora", "direct", 20_000, 0, 8, "no"),
        ("vamp_svd_r8_repair000", "prompt_nll", 20_000, 0, 7, "yes"),
        ("vamp_svd_r8_repair005", "prompt_nll", 20_000, _replay_presentations(run, "vamp_svd_r8_repair005"), 7, "yes"),
        ("vamp_core_tsv_r8_scale03_repair000", "prompt_nll", 20_000, 0, 7, "yes"),
        ("vamp_core_tsv_r8_scale03_repair005", "prompt_nll", 20_000, _replay_presentations(run, "vamp_core_tsv_r8_scale03_repair005"), 7, "yes"),
    )
    rows = []
    for condition, router, training, replay, live, task_free in specifications:
        summary = summary_by_key.get((condition, router))
        op = (
            float(summary["op"])
            if summary is not None
            else _final_only_op(records, condition, router)
        )
        cells = (
            condition,
            router,
            _number(op),
            _number(float(summary["forgetting"]) if summary is not None else None),
            _number(float(summary["bwt_signed"]) if summary is not None else None),
            _number(
                float(summary["bwt_clipped_negative_only"])
                if summary is not None
                else None
            ),
            f"{training:,}",
            f"{replay:,}",
            str(live),
            task_free,
        )
        rows.append("| " + " | ".join(cells) + " |")
    return tuple(rows)


def _final_only_op(
    records: tuple[dict[str, object], ...],
    condition: str,
    router: str,
) -> float | None:
    values = tuple(
        float(scores[router])
        for record in records
        for scores in (record.get("router_scores"),)
        if record.get("condition") == condition
        and int(record.get("stage", 0)) == 8
        and type(scores) is dict
        and router in scores
    )
    return sum(values) / 8 if len(values) == 8 else None


def _vamp_comparison_gaps(
    records: tuple[dict[str, object], ...],
    summaries: tuple[dict[str, object], ...],
) -> tuple[dict[str, object], ...]:
    """Return registered task-free VAMP gaps against sequential and joint OP."""
    joint_op = _final_only_op(records, "joint_iid_lora", "direct")
    sequential_op = next(
        (
            float(row["op"])
            for row in summaries
            if row["condition"] == "seq_lora_reference" and row["router"] == "direct"
        ),
        None,
    )
    if joint_op is None or sequential_op is None:
        return ()
    return tuple(
        {
            "condition": row["condition"],
            "joint_iid_vs_vamp": joint_op - float(row["op"]),
            "joint_op": joint_op,
            "router": row["router"],
            "sequential_op": sequential_op,
            "vamp_op": float(row["op"]),
            "vamp_vs_sequential": float(row["op"]) - sequential_op,
        }
        for row in summaries
        if str(row["condition"]).startswith("vamp_")
        and row["router"] in ("prompt_nll", "frozen_prompt_centroid")
    )


def _validation_policy_rows(
    records: tuple[dict[str, object], ...],
) -> tuple[dict[str, object], ...]:
    """Aggregate complete final-stage VAMP validation matrices without test scores."""
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for record in records:
        scores = record.get("validation_router_scores")
        condition = str(record.get("condition", ""))
        if int(record.get("stage", 0)) != 8 or not condition.startswith("vamp_"):
            continue
        if type(scores) is not dict:
            raise ValueError("final VAMP record lacks validation router scores")
        for router, score in scores.items():
            grouped[(condition, str(router))].append(float(score))
    aggregate = {
        key: sum(values) / 8
        for key, values in grouped.items()
        if len(values) == 8
    }
    best = {
        router: max(
            value
            for (condition, candidate_router), value in aggregate.items()
            if candidate_router == router
        )
        for router in {router for _, router in aggregate}
    }
    return tuple(
        {
            "condition": condition,
            "op": value,
            "router": router,
            "selected": value == best[router],
        }
        for (condition, router), value in sorted(aggregate.items())
    )


def _number(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _replay_presentations(run: Path, condition: str) -> int:
    policy_hashes = {
        str(record.get("policy_hash"))
        for path in (run / "evaluations").rglob("result.json")
        for record in (json.loads(path.read_text(encoding="utf-8")),)
        if record.get("condition") == condition
    }
    return sum(
        int(record.get("repair_examples", 0))
        for policy_hash in policy_hashes
        for path in (run / "derived" / policy_hash / "nodes").glob("*/repair_config.json")
        for record in (json.loads(path.read_text(encoding="utf-8")),)
    )


def _calibration_rows(run: Path) -> tuple[dict[str, object], ...]:
    rows = []
    for path in sorted((run / "evaluations" / "retrained_parents").glob("*/result.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("format") != "trace-retrained-parent-result-v1":
            raise ValueError(f"unexpected retrained-parent result: {path}")
        node = record.get("node")
        scores = record.get("candidate_scores")
        if type(node) is not dict or type(scores) is not dict:
            raise ValueError(f"malformed retrained-parent result: {path}")
        interval = f"{node['start_arrival']}–{node['end_arrival']}"
        rows.extend(
            {
                "candidate": str(candidate),
                "interval": interval,
                "score": float(score),
                "task": str(task),
            }
            for task, candidate_scores in sorted(scores.items())
            if type(candidate_scores) is dict
            for candidate, score in sorted(candidate_scores.items())
        )
    return tuple(rows)


def _calibration_csv(rows: tuple[dict[str, object], ...]) -> str:
    stream = StringIO()
    writer = csv.DictWriter(
        stream,
        fieldnames=("interval", "task", "candidate", "score"),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _merge_diagnostics(run: Path) -> tuple[dict[str, object], ...]:
    rows = []
    for lineage_path in sorted((run / "derived").glob("*/nodes/*/lineage.json")):
        directory = lineage_path.parent
        lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
        merge = json.loads((directory / "merge_metrics.json").read_text(encoding="utf-8"))
        repair = json.loads((directory / "repair_config.json").read_text(encoding="utf-8"))
        repair_training = (
            json.loads((directory / "train_metrics.json").read_text(encoding="utf-8"))
            if (directory / "train_metrics.json").is_file()
            else {}
        )
        validation_path = (
            run
            / "evaluations"
            / "merge_diagnostics"
            / str(lineage["policy_hash"])
            / f"{lineage['node']['node_id']}.json"
        )
        validation = (
            json.loads(validation_path.read_text(encoding="utf-8"))
            if validation_path.is_file()
            else {}
        )
        modules = merge.get("module_diagnostics")
        similarities = merge.get("module_similarity")
        node = lineage.get("node")
        composition = lineage.get("task_composition")
        if not all(type(value) is dict for value in (modules, similarities, node, composition)):
            raise ValueError(f"malformed merge diagnostics: {directory}")
        retained = tuple(float(value["retained_energy"]) for value in modules.values())
        cosines = tuple(float(value["cosine"]) for value in similarities.values())
        rows.append(
            {
                "child_cosine_mean": sum(cosines) / len(cosines),
                "core_dimension_max": max(
                    (
                        int(value["core_dimension"])
                        for value in modules.values()
                        if value["core_dimension"] is not None
                    ),
                    default=0,
                ),
                "end_arrival": int(node["end_arrival"]),
                "level": int(node["level"]),
                "merge_config_hash": str(lineage["merge_config_hash"]),
                "merge_damage": (
                    float(validation["merge_damage"])
                    if validation.get("merge_damage") is not None
                    else None
                ),
                "merge_method": str(lineage["policy"]["method"]),
                "merge_wall_seconds": float(merge["merge_wall_seconds"]),
                "node_id": str(node["node_id"]),
                "policy_hash": str(lineage["policy_hash"]),
                "repair_examples": int(repair["repair_examples"]),
                "repair_fraction": float(lineage["policy"]["repair_fraction"]),
                "repair_recovery": (
                    float(validation["repair_recovery"])
                    if validation.get("repair_recovery") is not None
                    else None
                ),
                "postrepair_damage": (
                    float(validation["merge_damage"])
                    - float(validation["repair_recovery"])
                    if validation.get("merge_damage") is not None
                    and validation.get("repair_recovery") is not None
                    else None
                ),
                "repair_steps": int(repair_training.get("optimizer_steps", 0)),
                "repair_tokens": int(repair_training.get("tokens", 0)),
                "repair_wall_seconds": float(repair_training.get("elapsed_seconds", 0.0)),
                "retained_energy_mean": sum(retained) / len(retained),
                "start_arrival": int(node["start_arrival"]),
                "within_task": len(composition) == 1,
            }
        )
    return tuple(rows)


def _diagnostics_csv(rows: tuple[dict[str, object], ...]) -> str:
    fields = (
        "policy_hash",
        "node_id",
        "level",
        "start_arrival",
        "end_arrival",
        "within_task",
        "merge_method",
        "repair_fraction",
        "retained_energy_mean",
        "child_cosine_mean",
        "merge_damage",
        "postrepair_damage",
        "repair_recovery",
        "core_dimension_max",
        "repair_examples",
        "repair_tokens",
        "repair_steps",
        "repair_wall_seconds",
        "merge_wall_seconds",
        "merge_config_hash",
    )
    stream = StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _lineage_svg(run: Path) -> str:
    arrivals = load_canonical_json(run / "manifests" / "arrivals.json")
    identities = tuple(str(value) for value in arrivals["arrival_ids"])
    if len(identities) != 40:
        raise ValueError("TRACE lineage report requires 40 arrivals")
    _, history = build_hierarchy(identities)
    leaves = tuple(leaf_node(index, identity) for index, identity in enumerate(identities, 1))
    nodes = (*leaves, *(event.parent for event in history))
    by_id = {node.node_id: node for node in nodes}
    width, height = 1880, 560

    def coordinates(node_id: str) -> tuple[float, float]:
        node = by_id[node_id]
        center = (node.start_arrival + node.end_arrival) / 2
        return 24 + (center - 0.5) * 46, 500 - node.level * 92

    edges = "".join(
        f'<line x1="{coordinates(parent)[0]:.1f}" y1="{coordinates(parent)[1]:.1f}" '
        f'x2="{coordinates(node.node_id)[0]:.1f}" y2="{coordinates(node.node_id)[1]:.1f}"/>'
        for node in nodes
        for parent in node.parent_node_ids
    )
    shapes = "".join(
        (
            f'<g><circle cx="{coordinates(node.node_id)[0]:.1f}" '
            f'cy="{coordinates(node.node_id)[1]:.1f}" r="12"/>'
            f'<text x="{coordinates(node.node_id)[0]:.1f}" '
            f'y="{coordinates(node.node_id)[1] + 4:.1f}">{node.start_arrival}'
            f'{"" if node.start_arrival == node.end_arrival else "–" + str(node.end_arrival)}</text></g>'
        )
        for node in nodes
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" '
        'aria-labelledby="title desc"><title id="title">TRACE temporal-consolidation lineage</title>'
        '<desc id="desc">Forty immutable leaves and thirty-three oldest-first capacity-two merges.</desc>'
        '<style>line{stroke:#8091a5;stroke-width:1.5}circle{fill:#3568a8;stroke:#163a60;stroke-width:1}'
        'text{font:7px system-ui;text-anchor:middle;fill:white}</style>'
        f'{edges}{shapes}</svg>\n'
    )


def _summary_plot(summaries: tuple[dict[str, object], ...]) -> str | None:
    if not summaries:
        return None
    try:
        import matplotlib.pyplot as pyplot
    except ImportError:
        return None
    labels = tuple(f"{row['condition']}\n{row['router']}" for row in summaries)
    values = tuple(float(row["op"]) for row in summaries)
    figure, axis = pyplot.subplots(figsize=(max(8, len(labels) * 0.8), 5))
    axis.bar(range(len(labels)), values, color="#3568a8")
    axis.set_ylabel("Final OP")
    axis.set_title("TRACE final overall performance")
    axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right", fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    stream = BytesIO()
    figure.savefig(stream, format="png", dpi=150)
    pyplot.close(figure)
    return base64.b64encode(stream.getvalue()).decode()


def _merge_diagnostic_plot(rows: tuple[dict[str, object], ...]) -> bytes:
    try:
        import matplotlib.pyplot as pyplot
    except ImportError as error:
        raise RuntimeError("TRACE diagnostics plots require matplotlib") from error
    figure, axes = pyplot.subplots(2, 3, figsize=(15, 8))
    evaluated = tuple(row for row in rows if row["merge_damage"] is not None)
    colors = {
        "svd_mean_r8": "#3568a8",
        "core_tsv_r8": "#d97706",
    }
    for method in colors:
        selected = tuple(row for row in evaluated if row["merge_method"] == method)
        axes[0, 0].scatter(
            tuple(int(row["level"]) for row in selected),
            tuple(float(row["merge_damage"]) for row in selected),
            label=method,
            color=colors[method],
            alpha=0.8,
        )
        axes[0, 1].scatter(
            tuple(float(row["retained_energy_mean"]) for row in selected),
            tuple(float(row["merge_damage"]) for row in selected),
            label=method,
            color=colors[method],
            alpha=0.8,
        )
    axes[0, 0].set(xlabel="Hierarchy level", ylabel="Answer-NLL merge damage")
    axes[0, 1].set(xlabel="Mean retained spectral energy", ylabel="Answer-NLL merge damage")
    categories = ("within task", "cross task")
    _diagnostic_boxplot(
        axes[0, 2],
        categories,
        tuple(
            tuple(
                float(row["merge_damage"])
                for row in evaluated
                if bool(row["within_task"]) == (category == "within task")
            )
            for category in categories
        ),
    )
    axes[0, 2].set_ylabel("Answer-NLL merge damage")
    methods = ("svd_mean_r8", "core_tsv_r8")
    _diagnostic_boxplot(
        axes[1, 0],
        ("SVD", "Core+TSV"),
        tuple(
            tuple(
                float(row["merge_damage"])
                for row in evaluated
                if row["merge_method"] == method
            )
            for method in methods
        ),
    )
    axes[1, 0].set_ylabel("Answer-NLL merge damage")
    repair_fractions = (0.0, 0.05)
    _diagnostic_boxplot(
        axes[1, 1],
        ("0% repair", "5% repair"),
        tuple(
            tuple(
                float(row["postrepair_damage"])
                for row in evaluated
                if float(row["repair_fraction"]) == fraction
                and row["postrepair_damage"] is not None
            )
            for fraction in repair_fractions
        ),
    )
    axes[1, 1].set_ylabel("Post-repair answer-NLL damage")
    by_policy: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in evaluated:
        by_policy[str(row["policy_hash"])].append(row)
    for policy_hash, policy_rows in sorted(by_policy.items()):
        levels = tuple(sorted({int(row["level"]) for row in policy_rows}))
        mean_incremental_damage = tuple(
            sum(
                float(row["postrepair_damage"] or 0.0)
                for row in policy_rows
                if int(row["level"]) == level
            )
            / sum(int(row["level"]) == level for row in policy_rows)
            for level in levels
        )
        losses = tuple(
            accumulate(mean_incremental_damage)
        )
        axes[1, 2].plot(levels, losses, marker="o", label=policy_hash[:8])
    axes[1, 2].set(
        xlabel="Consolidation depth",
        ylabel="Cumulative mean merge damage",
    )
    if not evaluated:
        figure.text(
            0.5,
            0.5,
            "Selected merge-validation diagnostics are not complete yet.",
            ha="center",
            va="center",
        )
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    if evaluated:
        axes[0, 0].legend()
        axes[1, 2].legend(fontsize=7)
    figure.suptitle("TRACE consolidation diagnostics")
    figure.tight_layout()
    stream = BytesIO()
    figure.savefig(stream, format="png", dpi=150, metadata={"Software": "apm TRACE"})
    pyplot.close(figure)
    return stream.getvalue()


def _diagnostic_boxplot(
    axis: Axes,
    labels: tuple[str, ...],
    values: tuple[tuple[float, ...], ...],
) -> None:
    """Draw a comparison boxplot while keeping empty interim reports warning-free."""
    if any(values):
        axis.boxplot(values, tick_labels=labels)
    else:
        axis.set_xticks(range(1, len(labels) + 1), labels)


def _html(
    markdown: str,
    records: tuple[dict[str, object], ...],
    summaries: tuple[dict[str, object], ...],
    archive_bytes: int,
    plot_data: str | None,
    merge_plot_data: str,
    lineage_svg: str,
    interim: bool,
) -> str:
    title = "TRACE Interim Report" if interim else "TRACE Primary Report"
    summary_rows = "".join(
        "<tr>"
        f"<td>{escape(str(row['condition']))}</td>"
        f"<td>{escape(str(row['router']))}</td>"
        f"<td>{float(row['op']):.3f}</td>"
        f"<td>{float(row['forgetting']):.3f}</td>"
        f"<td>{float(row['bwt_signed']):.3f}</td>"
        f"<td>{float(row['bwt_clipped_negative_only']):.3f}</td>"
        f"<td>{'no' if row['router'] in ('task_aware', 'answer_oracle') else 'yes'}</td>"
        "</tr>"
        for row in summaries
    ) or '<tr><td colspan="7">The triangular matrix is not complete yet.</td></tr>'
    gap_rows = "".join(
        "<tr>"
        f"<td>{escape(str(row['condition']))}</td>"
        f"<td>{escape(str(row['router']))}</td>"
        f"<td>{float(row['vamp_op']):.3f}</td>"
        f"<td>{float(row['sequential_op']):.3f}</td>"
        f"<td>{float(row['vamp_vs_sequential']):.3f}</td>"
        f"<td>{float(row['joint_op']):.3f}</td>"
        f"<td>{float(row['joint_iid_vs_vamp']):.3f}</td>"
        "</tr>"
        for row in _vamp_comparison_gaps(records, summaries)
    ) or '<tr><td colspan="7">Sequential, joint-IID, or VAMP evaluation is incomplete.</td></tr>'
    validation_rows = "".join(
        "<tr>"
        f"<td>{escape(str(row['condition']))}</td>"
        f"<td>{escape(str(row['router']))}</td>"
        f"<td>{float(row['op']):.3f}</td>"
        f"<td>{'yes' if row['selected'] else 'no'}</td>"
        "</tr>"
        for row in _validation_policy_rows(records)
    ) or '<tr><td colspan="4">Final validation matrices are incomplete.</td></tr>'
    plot = (
        f'<img alt="Final OP comparison" src="data:image/png;base64,{plot_data}">'
        if plot_data
        else "<p>No complete method is available to plot yet.</p>"
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>
body{{font:16px/1.5 system-ui,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1.5rem;color:#17202a}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccd3da;padding:.5rem;text-align:right}}th:first-child,td:first-child{{text-align:left}}
th{{background:#edf3f8}}img{{max-width:100%;height:auto}}code,pre{{background:#f5f7f9;padding:.2rem .4rem}}details{{margin:1.5rem 0}}
</style></head><body><h1>{title}</h1>
<p>{len(records):,} completed task-stage rows; archive size {archive_bytes / 1024**2:.2f} MiB.</p>
<h2>Headline results</h2><table><thead><tr><th>Method</th><th>Router</th><th>OP</th><th>Forgetting</th><th>Signed BWT</th><th>Negative-only BWT</th><th>Task-free</th></tr></thead><tbody>{summary_rows}</tbody></table>
<h2>Sequential and joint-IID gaps</h2><table><thead><tr><th>VAMP method</th><th>Task-free router</th><th>VAMP OP</th><th>Sequential OP</th><th>VAMP − sequential</th><th>Joint-IID OP</th><th>Joint-IID − VAMP</th></tr></thead><tbody>{gap_rows}</tbody></table>
<h2>Validation-only policy comparison</h2><table><thead><tr><th>Policy condition</th><th>Router</th><th>Final validation OP</th><th>Best for router</th></tr></thead><tbody>{validation_rows}</tbody></table>
<h2>Overall performance</h2>{plot}
<h2>Merge diagnostics</h2><img alt="Merge damage, retained energy, task composition, and repair recovery" src="data:image/png;base64,{merge_plot_data}">
<h2>Temporal lineage</h2>{lineage_svg}
<details><summary>Protocol and memory-accounting notes</summary><pre>{escape(markdown)}</pre></details>
</body></html>"""


__all__ = ["ReportResult", "build_report"]

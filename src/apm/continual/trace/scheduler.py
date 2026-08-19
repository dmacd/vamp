"""Two-GPU subprocess coordinator over the authenticated TRACE job DAG."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

from apm.continual.artifacts import ChainedJsonlLedger, load_canonical_json
from apm.continual.trace.jobs import MAX_JOB_ATTEMPTS, JobLedger, JobSpec
from apm.continual.trace.runtime_budget import RuntimeBudget


CHECKPOINTED_EXIT_CODE = 75


@dataclass(frozen=True, slots=True)
class RunningProcess:
    """One coordinator-owned worker process and its assigned device slot."""

    spec: JobSpec
    gpu_index: int | None
    process: subprocess.Popen[bytes]
    started_at: float


def run_coordinator(
    run_directory: str | Path,
    budget: RuntimeBudget,
    *,
    gpu_count: int = 2,
    poll_seconds: float = 1.0,
) -> None:
    """Dispatch ready jobs, recover restarts, and quiesce at the soft deadline."""
    if gpu_count != 2 or not 0.1 <= poll_seconds <= 10.0:
        raise ValueError("primary TRACE execution requires two GPUs and bounded polling")
    root = Path(run_directory)
    ledger = JobLedger(root / "manifests" / "jobs.jsonl")
    ledger.recover_running()
    active: list[RunningProcess] = []
    paused = False
    last_status = 0.0
    while not ledger.all_complete():
        active = _collect_finished(root, ledger, active)
        irrecoverable = tuple(
            status
            for status in ledger.statuses
            if status.state == "FAILED" and status.attempt >= MAX_JOB_ATTEMPTS
        )
        if irrecoverable:
            raise RuntimeError(
                "TRACE jobs exhausted their retry budget: "
                + ", ".join(status.spec.job_id for status in irrecoverable)
            )
        now = datetime.now(timezone.utc)
        if budget.should_pause(now) and not paused:
            paused = True
            print("TRACE phase: soft runtime limit reached; checkpointing active jobs")
            for status in ledger.statuses:
                if status.state == "PENDING":
                    ledger.transition(status.spec.job_id, "PAUSED", detail="soft runtime limit")
            for running in active:
                running.process.send_signal(signal.SIGTERM)
        if not paused:
            active.extend(_dispatch_available(root, ledger, active, gpu_count))
        if paused and not active:
            return
        if time.monotonic() - last_status >= 30.0:
            print(f"TRACE scheduler: {ledger.state_counts()} ETA: {eta_snapshot(root)}")
            last_status = time.monotonic()
        if budget.hard_expired(now) and active:
            raise RuntimeError("hard runtime limit reached before active jobs checkpointed")
        time.sleep(poll_seconds)


def _dispatch_available(
    root: Path,
    ledger: JobLedger,
    active: list[RunningProcess],
    gpu_count: int,
) -> list[RunningProcess]:
    free_gpus = tuple(index for index in range(gpu_count) if index not in {job.gpu_index for job in active})
    launched: list[RunningProcess] = []
    for gpu_index in free_gpus:
        candidates = tuple(
            spec
            for spec in ledger.runnable("gpu")
            if spec.job_id not in {job.spec.job_id for job in (*active, *launched)}
        )
        if not candidates:
            break
        spec = _choose_gpu_job(candidates, gpu_index)
        launched.append(_launch(root, ledger, spec, gpu_index))
    cpu_active = any(job.gpu_index is None for job in active)
    cpu_candidates = ledger.runnable("cpu")
    if not cpu_active and cpu_candidates:
        launched.append(_launch(root, ledger, cpu_candidates[0], None))
    return launched


def _choose_gpu_job(candidates: tuple[JobSpec, ...], gpu_index: int) -> JobSpec:
    preferred_method = "svd_mean_r8" if gpu_index == 0 else "core_tsv_r8"

    def affinity(spec: JobSpec) -> int:
        policy = spec.payload.get("policy")
        method = policy.get("method") if type(policy) is dict else spec.payload.get("method")
        return 0 if method == preferred_method else 1

    return min(candidates, key=lambda spec: (affinity(spec), spec.priority, spec.job_id))


def _launch(
    root: Path,
    ledger: JobLedger,
    spec: JobSpec,
    gpu_index: int | None,
) -> RunningProcess:
    worker = f"gpu-{gpu_index}" if gpu_index is not None else "cpu"
    ledger.transition(spec.job_id, "RUNNING", worker=worker)
    environment = dict(os.environ)
    if gpu_index is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    process = subprocess.Popen(
        (
            sys.executable,
            "-m",
            "apm.continual.trace.cli",
            "_worker",
            "--run",
            str(root),
            "--job-id",
            spec.job_id,
        ),
        env=environment,
        stdout=None,
        stderr=None,
    )
    return RunningProcess(spec, gpu_index, process, time.monotonic())


def _collect_finished(
    root: Path,
    ledger: JobLedger,
    active: list[RunningProcess],
) -> list[RunningProcess]:
    remaining: list[RunningProcess] = []
    for running in active:
        result = running.process.poll()
        if result is None:
            remaining.append(running)
            continue
        ChainedJsonlLedger(
            root / "logs" / "job_timings.jsonl",
            "trace-job-timing-v1",
        ).append(
            {
                "elapsed_seconds": time.monotonic() - running.started_at,
                "job_id": running.spec.job_id,
                "kind": running.spec.kind,
                "outcome": (
                    "complete"
                    if result == 0
                    else "checkpointed"
                    if result == CHECKPOINTED_EXIT_CODE
                    else "failed"
                ),
                "resource": running.spec.resource,
            }
        )
        if result == 0:
            receipt = load_canonical_json(
                root / "state" / "job_outputs" / f"{running.spec.job_id}.json"
            )
            ledger.transition(
                running.spec.job_id,
                "COMPLETE",
                worker=f"gpu-{running.gpu_index}" if running.gpu_index is not None else "cpu",
                output_sha256=str(receipt["output_sha256"]),
            )
        elif result == CHECKPOINTED_EXIT_CODE:
            ledger.transition(
                running.spec.job_id,
                "CHECKPOINTED",
                detail="worker paused at a durable boundary",
            )
        else:
            ledger.transition(
                running.spec.job_id,
                "FAILED",
                detail=f"worker exited with status {result}",
            )
    return remaining


def eta_snapshot(run_directory: str | Path) -> dict[str, object]:
    """Estimate remaining runtime from successful job-family timings when representative."""
    root = Path(run_directory)
    ledger = JobLedger(root / "manifests" / "jobs.jsonl")
    timing = ChainedJsonlLedger(
        root / "logs" / "job_timings.jsonl",
        "trace-job-timing-v1",
    )
    timing_rows = timing.rows
    complete_ids = {
        str(row["job_id"]) for row in timing_rows if row["outcome"] == "complete"
    }
    successful = tuple(
        {
            "elapsed_seconds": sum(
                float(row["elapsed_seconds"])
                for row in timing_rows
                if str(row["job_id"]) == job_id
            ),
            "job_id": job_id,
            "kind": str(
                next(row["kind"] for row in timing_rows if str(row["job_id"]) == job_id)
            ),
        }
        for job_id in sorted(complete_ids)
    )
    observed_seconds = sum(float(row["elapsed_seconds"]) for row in successful)
    base: dict[str, object] = {
        "static_expected_hours": [8, 16],
        "static_conservative_hours": [16, 22],
        "observed_successful_jobs": len(successful),
    }
    if observed_seconds < 15 * 60 or len(successful) < 4:
        return {**base, "confidence": "static", "measured_hours": None}
    durations: dict[str, list[float]] = {}
    for row in successful:
        durations.setdefault(str(row["kind"]), []).append(float(row["elapsed_seconds"]))
    global_median = statistics.median(
        value for values in durations.values() for value in values
    )
    gpu_seconds = 0.0
    cpu_seconds = 0.0
    for status in ledger.statuses:
        if status.state == "COMPLETE":
            continue
        estimate = statistics.median(durations.get(status.spec.kind, (global_median,)))
        if status.spec.resource == "gpu":
            gpu_seconds += estimate
        else:
            cpu_seconds += estimate
    measured_hours = max(gpu_seconds / 2, cpu_seconds) / 3600
    confidence = "medium" if len(successful) >= 20 and len(durations) >= 3 else "low"
    spread = 0.35 if confidence == "medium" else 0.60
    return {
        **base,
        "confidence": confidence,
        "measured_hours": measured_hours,
        "measured_range_hours": [
            max(0.0, measured_hours * (1 - spread)),
            measured_hours * (1 + spread),
        ],
    }


__all__ = [
    "CHECKPOINTED_EXIT_CODE",
    "RunningProcess",
    "eta_snapshot",
    "run_coordinator",
]

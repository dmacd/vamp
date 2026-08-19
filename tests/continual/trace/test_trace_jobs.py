from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request

import pytest

from apm.continual.artifacts import record_sha256
from apm.continual.trace.dag import build_policy_dag, build_primary_dag
from apm.continual.trace.jobs import JobLedger, JobSpec
from apm.continual.trace.protocol import default_merge_policies
from apm.continual.trace.runtime_budget import (
    RuntimeBudget,
    publish_safe_to_terminate,
    terminate_current_runpod,
)


def _simple_jobs() -> tuple[JobSpec, JobSpec]:
    first = JobSpec.create("first", "gpu", 1, (), {"value": 1})
    second = JobSpec.create("second", "cpu", 2, (first.job_id,), {"value": 2})
    return first, second


def test_primary_dag_has_every_registered_work_family() -> None:
    arrivals = tuple(record_sha256({"arrival": index}) for index in range(1, 41))

    run_hash = record_sha256({"run": "primary"})
    jobs = build_primary_dag(arrivals, default_merge_policies(), run_hash)

    assert len(jobs) == 422
    assert len({job.job_id for job in jobs}) == len(jobs)
    assert sum(job.kind == "train_leaf" for job in jobs) == 40
    assert sum(job.kind == "merge_node" for job in jobs) == 132
    assert sum(job.kind == "evaluate_vamp" for job in jobs) == 144
    assert sum(job.kind == "retrained_parent_oracle" for job in jobs) == 4
    assert all(job.payload["run_contract_hash"] == run_hash for job in jobs)
    assert all(
        len(job.dependencies) == 4
        for job in jobs
        if job.kind == "retrained_parent_oracle"
    )
    other_jobs = build_primary_dag(
        arrivals,
        default_merge_policies(),
        record_sha256({"run": "other"}),
    )
    assert {job.job_id for job in jobs}.isdisjoint(job.job_id for job in other_jobs)


def test_policy_dag_binds_leaf_hashes_into_its_resumable_acceptance_job() -> None:
    arrivals = tuple(record_sha256({"arrival": index}) for index in range(1, 41))
    leaf_jobs = tuple(record_sha256({"leaf_job": index}) for index in range(1, 41))
    leaf_hashes = tuple(record_sha256({"leaf_adapter": index}) for index in range(1, 41))
    policy = default_merge_policies()[2]

    jobs = build_policy_dag(
        arrivals,
        policy,
        leaf_jobs,
        leaf_hashes,
        record_sha256({"embedding": 1}),
        record_sha256({"run": "policy"}),
    )

    assert len(jobs) == 70
    report = next(job for job in jobs if job.kind == "build_policy_report")
    assert report.payload["leaf_adapter_hashes"] == list(leaf_hashes)
    assert report.payload["policy_hash"] == policy.policy_hash


def test_job_ledger_enforces_dependencies_and_never_reopens_complete(tmp_path: Path) -> None:
    first, second = _simple_jobs()
    ledger = JobLedger(tmp_path / "jobs.jsonl")
    ledger.register((first, second))

    assert ledger.runnable() == (first,)
    ledger.transition(first.job_id, "RUNNING", worker="gpu-0")
    ledger.transition(first.job_id, "COMPLETE", output_sha256=record_sha256({"done": 1}))

    assert ledger.runnable() == (second,)
    with pytest.raises(ValueError, match="invalid TRACE job transition"):
        ledger.transition(first.job_id, "RUNNING")
    reloaded = JobLedger(tmp_path / "jobs.jsonl")
    assert reloaded.runnable() == (second,)


def test_job_ledger_recovers_abandoned_running_job_as_checkpointed(tmp_path: Path) -> None:
    first, _ = _simple_jobs()
    ledger = JobLedger(tmp_path / "jobs.jsonl")
    ledger.register((first,))
    ledger.transition(first.job_id, "RUNNING", worker="gpu-0")

    JobLedger(tmp_path / "jobs.jsonl").recover_running()

    recovered = JobLedger(tmp_path / "jobs.jsonl")
    assert recovered.statuses[0].state == "CHECKPOINTED"
    assert recovered.runnable() == (first,)


def test_failed_job_is_retryable_without_reopening_completed_dependencies(
    tmp_path: Path,
) -> None:
    first, _ = _simple_jobs()
    ledger = JobLedger(tmp_path / "jobs.jsonl")
    ledger.register((first,))
    ledger.transition(first.job_id, "RUNNING", worker="gpu-0")
    ledger.transition(first.job_id, "FAILED", detail="transient worker failure")

    assert ledger.runnable() == (first,)
    for attempt in (2, 3):
        ledger.transition(first.job_id, "RUNNING", worker=f"gpu-{attempt % 2}")
        ledger.transition(first.job_id, "FAILED", detail="persistent worker failure")
    assert ledger.runnable() == ()


def test_runtime_budget_and_safe_marker_guard_termination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, _ = _simple_jobs()
    ledger = JobLedger(tmp_path / "jobs.jsonl")
    ledger.register((first,))
    ledger.transition(first.job_id, "PAUSED", detail="interim")
    report = tmp_path / "interim.md"
    report.write_text("durable report", encoding="utf-8")
    marker = publish_safe_to_terminate(tmp_path / "state", ledger, (report,))
    monkeypatch.setenv("RUNPOD_API_KEY", "secret-key-never-persist")
    monkeypatch.setenv("RUNPOD_POD_ID", "pod-123")
    captured: list[Request] = []

    def fake_sender(request: Request, timeout: float) -> int:
        assert timeout == 30.0
        captured.append(request)
        return 204

    receipt = terminate_current_runpod(marker, tmp_path / "receipt.json", fake_sender)

    assert RuntimeBudget(datetime.now(timezone.utc)).soft_deadline < RuntimeBudget(
        datetime.now(timezone.utc)
    ).hard_deadline
    assert captured[0].full_url.endswith("/pods/pod-123")
    assert "secret-key-never-persist" not in receipt.read_text(encoding="utf-8")


def test_automatic_cleanup_can_delete_only_a_marker_guarded_pod() -> None:
    root = Path(__file__).resolve().parents[3]
    runtime = (root / "src/apm/continual/trace/runtime_budget.py").read_text()
    watchdog = (root / "scripts/trace/watchdog.sh").read_text()
    launcher = (root / "scripts/trace/launch_runpod.sh").read_text()

    assert "SAFE_TO_TERMINATE" in runtime
    assert "trace-safe-to-terminate-v1" in watchdog
    assert "--terminate-after" not in launcher
    assert "network-volumes" not in runtime + watchdog + launcher

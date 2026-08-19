"""Immutable semantic job identities and authenticated DAG state."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal, TypeAlias, cast

from apm.continual.artifacts import (
    ChainedJsonlLedger,
    canonical_json_bytes,
    record_sha256,
    require_sha256,
)
from apm.continual.trace.protocol import JobState


JOBS_LEDGER_FORMAT = "trace-jobs-v1"
MAX_JOB_ATTEMPTS = 3
JobResource: TypeAlias = Literal["cpu", "gpu"]


@dataclass(frozen=True, slots=True)
class JobSpec:
    """One semantic DAG node whose identity includes inputs and dependencies."""

    job_id: str
    kind: str
    resource: JobResource
    priority: int
    dependencies: tuple[str, ...]
    payload_json: str

    def __post_init__(self) -> None:
        require_sha256(self.job_id, "TRACE job")
        if not self.kind or self.resource not in ("cpu", "gpu") or self.priority < 0:
            raise ValueError("invalid TRACE job kind, resource, or priority")
        for dependency in self.dependencies:
            require_sha256(dependency, "TRACE job dependency")
        value = json.loads(self.payload_json)
        if type(value) is not dict or canonical_json_bytes(value).decode() != self.payload_json:
            raise ValueError("TRACE job payload is not canonical JSON")
        if self.job_id != record_sha256(self.identity_record()):
            raise ValueError("TRACE job identity differs from its semantic payload")

    @classmethod
    def create(
        cls,
        kind: str,
        resource: JobResource,
        priority: int,
        dependencies: Sequence[str],
        payload: Mapping[str, object],
    ) -> JobSpec:
        """Construct a semantic identity from canonical payload and dependencies."""
        payload_json = canonical_json_bytes(dict(payload)).decode()
        core = {
            "dependencies": list(dependencies),
            "kind": kind,
            "payload": json.loads(payload_json),
            "priority": priority,
            "resource": resource,
        }
        return cls(
            job_id=record_sha256(core),
            kind=kind,
            resource=resource,
            priority=priority,
            dependencies=tuple(dependencies),
            payload_json=payload_json,
        )

    @property
    def payload(self) -> dict[str, object]:
        """Return a detached JSON payload object."""
        return dict(json.loads(self.payload_json))

    def identity_record(self) -> dict[str, object]:
        """Return the canonical semantic fields hashed into ``job_id``."""
        return {
            "dependencies": list(self.dependencies),
            "kind": self.kind,
            "payload": self.payload,
            "priority": self.priority,
            "resource": self.resource,
        }

    def as_record(self) -> dict[str, object]:
        """Return a ledger-ready complete job record."""
        return {**self.identity_record(), "job_id": self.job_id}


@dataclass(frozen=True, slots=True)
class JobStatus:
    """Latest authenticated state for one registered semantic job."""

    spec: JobSpec
    state: JobState
    attempt: int
    worker: str | None
    detail: str | None
    output_sha256: str | None


class JobLedger:
    """Append-only state transitions and deterministic runnable-job projection."""

    def __init__(self, path: str | Path) -> None:
        self._ledger = ChainedJsonlLedger(path, JOBS_LEDGER_FORMAT)
        self._statuses = self._reconstruct()

    @property
    def statuses(self) -> tuple[JobStatus, ...]:
        """Return current job states sorted by semantic identity."""
        return tuple(self._statuses[job_id] for job_id in sorted(self._statuses))

    def register(self, specs: Iterable[JobSpec]) -> None:
        """Register missing jobs as PENDING after validating the complete DAG."""
        supplied = tuple(specs)
        combined = {status.spec.job_id: status.spec for status in self.statuses}
        for spec in supplied:
            if spec.job_id in combined and combined[spec.job_id] != spec:
                raise ValueError("registered job identity maps to a different specification")
            combined[spec.job_id] = spec
        _require_acyclic(tuple(combined.values()))
        missing_dependencies = {
            dependency
            for spec in combined.values()
            for dependency in spec.dependencies
            if dependency not in combined
        }
        if missing_dependencies:
            raise ValueError("TRACE DAG contains unregistered dependencies")
        for spec in supplied:
            if spec.job_id not in self._statuses:
                self._ledger.append(
                    {
                        "attempt": 0,
                        "detail": None,
                        "event": "register",
                        "job": spec.as_record(),
                        "job_id": spec.job_id,
                        "output_sha256": None,
                        "state": "PENDING",
                        "worker": None,
                    }
                )
                self._statuses[spec.job_id] = JobStatus(
                    spec, "PENDING", 0, None, None, None
                )

    def transition(
        self,
        job_id: str,
        state: JobState,
        *,
        worker: str | None = None,
        detail: str | None = None,
        output_sha256: str | None = None,
    ) -> JobStatus:
        """Append one allowed state transition and update the current projection."""
        if job_id not in self._statuses:
            raise KeyError(f"unknown TRACE job: {job_id}")
        current = self._statuses[job_id]
        allowed: dict[JobState, frozenset[JobState]] = {
            "PENDING": frozenset(("RUNNING", "PAUSED")),
            "RUNNING": frozenset(("CHECKPOINTED", "COMPLETE", "FAILED", "PAUSED")),
            "CHECKPOINTED": frozenset(("RUNNING", "PAUSED")),
            "COMPLETE": frozenset(),
            "FAILED": frozenset(("RUNNING", "PAUSED")),
            "PAUSED": frozenset(("RUNNING",)),
        }
        if state not in allowed[current.state]:
            raise ValueError(f"invalid TRACE job transition {current.state} -> {state}")
        if output_sha256 is not None:
            require_sha256(output_sha256, "TRACE job output")
        attempt = current.attempt + (state == "RUNNING")
        row = {
            "attempt": attempt,
            "detail": detail,
            "event": "transition",
            "job_id": job_id,
            "output_sha256": output_sha256,
            "state": state,
            "worker": worker,
        }
        self._ledger.append(row)
        updated = JobStatus(current.spec, state, attempt, worker, detail, output_sha256)
        self._statuses[job_id] = updated
        return updated

    def runnable(self, resource: JobResource | None = None) -> tuple[JobSpec, ...]:
        """Return pending/resumable jobs whose dependencies are complete."""
        complete = {
            status.spec.job_id for status in self.statuses if status.state == "COMPLETE"
        }
        runnable_states = {"PENDING", "CHECKPOINTED", "FAILED", "PAUSED"}
        return tuple(
            status.spec
            for status in sorted(
                self.statuses,
                key=lambda item: (item.spec.priority, item.spec.job_id),
            )
            if status.state in runnable_states
            and not (status.state == "FAILED" and status.attempt >= MAX_JOB_ATTEMPTS)
            and (resource is None or status.spec.resource == resource)
            and set(status.spec.dependencies) <= complete
        )

    def recover_running(self) -> None:
        """Convert jobs abandoned by a coordinator restart into resumable checkpoints."""
        for status in tuple(self.statuses):
            if status.state == "RUNNING":
                self.transition(
                    status.spec.job_id,
                    "CHECKPOINTED",
                    detail="coordinator restart; worker checkpoint is authoritative",
                )

    def all_complete(self) -> bool:
        """Return whether every registered job is durably complete."""
        return bool(self._statuses) and all(
            status.state == "COMPLETE" for status in self.statuses
        )

    def state_counts(self) -> dict[str, int]:
        """Return human-readable counts by scheduler state."""
        return {
            state: sum(status.state == state for status in self.statuses)
            for state in (
                "PENDING",
                "RUNNING",
                "CHECKPOINTED",
                "COMPLETE",
                "FAILED",
                "PAUSED",
            )
        }

    def _reconstruct(self) -> dict[str, JobStatus]:
        statuses: dict[str, JobStatus] = {}
        for row in self._ledger.rows:
            job_id = str(row["job_id"])
            if row["event"] == "register":
                raw = row["job"]
                if type(raw) is not dict:
                    raise ValueError("job registration payload is malformed")
                spec = JobSpec.create(
                    kind=str(raw["kind"]),
                    resource=cast(JobResource, str(raw["resource"])),
                    priority=int(raw["priority"]),
                    dependencies=tuple(str(value) for value in raw["dependencies"]),
                    payload=dict(raw["payload"]),
                )
                if spec.job_id != job_id or job_id in statuses:
                    raise ValueError("job registration identity is duplicated or changed")
                statuses[job_id] = JobStatus(spec, "PENDING", 0, None, None, None)
            else:
                if job_id not in statuses:
                    raise ValueError("job transition precedes registration")
                current = statuses[job_id]
                statuses[job_id] = JobStatus(
                    current.spec,
                    cast(JobState, str(row["state"])),
                    int(row["attempt"]),
                    str(row["worker"]) if row["worker"] is not None else None,
                    str(row["detail"]) if row["detail"] is not None else None,
                    str(row["output_sha256"])
                    if row["output_sha256"] is not None
                    else None,
                )
        return statuses


def _require_acyclic(specs: Sequence[JobSpec]) -> None:
    by_id = {spec.job_id: spec for spec in specs}
    visited: set[str] = set()
    active: set[str] = set()

    def visit(job_id: str) -> None:
        if job_id in active:
            raise ValueError("TRACE job graph contains a cycle")
        if job_id in visited or job_id not in by_id:
            return
        active.add(job_id)
        for dependency in by_id[job_id].dependencies:
            visit(dependency)
        active.remove(job_id)
        visited.add(job_id)

    for spec in specs:
        visit(spec.job_id)


__all__ = [
    "JOBS_LEDGER_FORMAT",
    "MAX_JOB_ATTEMPTS",
    "JobLedger",
    "JobResource",
    "JobSpec",
    "JobStatus",
]

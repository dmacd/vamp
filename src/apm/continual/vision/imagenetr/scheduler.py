"""Single-GPU durable job-state coordinator for the local ImageNet-R workflow."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from collections.abc import Callable, Mapping, Sequence
import json

from apm.continual.artifacts import atomic_write, canonical_json_bytes
from apm.continual.vision.imagenetr.protocol import JobSpec


class JobState(str, Enum):
    """Durable lifecycle states for every independently resumable job."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    PAUSED = "PAUSED"


@dataclass(frozen=True, slots=True)
class JobStatus:
    """One scheduler job's durable state and bounded execution metadata."""

    job_hash: str
    state: JobState
    attempts: int
    started_at: str | None
    finished_at: str | None
    result: dict[str, object] | None
    error: str | None

    def as_record(self) -> dict[str, object]:
        """Return a canonical scheduler-state row."""
        return {
            "attempts": self.attempts,
            "error": self.error,
            "finished_at": self.finished_at,
            "job_hash": self.job_hash,
            "result": self.result,
            "started_at": self.started_at,
            "state": self.state.value,
        }


class LocalScheduler:
    """Sole state writer that executes one atomic job at a time on one GPU."""

    def __init__(self, path: str | Path, run_hash: str) -> None:
        self.path = Path(path)
        self.run_hash = run_hash
        self._specs: dict[str, JobSpec] = {}
        self._statuses: dict[str, JobStatus] = {}
        self._load()

    @property
    def statuses(self) -> tuple[JobStatus, ...]:
        """Return statuses in stable job-hash order."""
        return tuple(self._statuses[key] for key in sorted(self._statuses))

    def register(self, specs: Sequence[JobSpec]) -> None:
        """Add exact job specifications without changing existing identities."""
        for spec in specs:
            if spec.run_hash != self.run_hash:
                raise ValueError("scheduler job belongs to another run")
            identity = spec.content_hash
            if identity in self._specs and self._specs[identity] != spec:
                raise ValueError("scheduler job identity collision")
            self._specs[identity] = spec
            self._statuses.setdefault(
                identity,
                JobStatus(identity, JobState.PENDING, 0, None, None, None, None),
            )
        self._save()

    def execute(
        self,
        spec: JobSpec,
        handler: Callable[[], Mapping[str, object]],
    ) -> dict[str, object]:
        """Run or reuse one dependency-ready job and persist every transition."""
        self.register((spec,))
        identity = spec.content_hash
        status = self._statuses[identity]
        if status.state == JobState.COMPLETE:
            return dict(status.result or {})
        incomplete = tuple(
            dependency
            for dependency in spec.dependencies
            if dependency not in self._statuses
            or self._statuses[dependency].state != JobState.COMPLETE
        )
        if incomplete:
            raise RuntimeError(f"job dependencies are incomplete: {incomplete}")
        now = _utc_now()
        self._statuses[identity] = JobStatus(
            identity,
            JobState.RUNNING,
            status.attempts + 1,
            now,
            None,
            None,
            None,
        )
        self._save()
        try:
            result = dict(handler())
        except BaseException as error:
            self._statuses[identity] = JobStatus(
                identity,
                JobState.FAILED,
                status.attempts + 1,
                now,
                _utc_now(),
                None,
                f"{type(error).__name__}: {error}",
            )
            self._save()
            raise
        self._statuses[identity] = JobStatus(
            identity,
            JobState.COMPLETE,
            status.attempts + 1,
            now,
            _utc_now(),
            result,
            None,
        )
        self._save()
        return result

    def pause_pending(self) -> None:
        """Durably mark undispatched work paused without disturbing completed jobs."""
        self._statuses = {
            identity: (
                JobStatus(
                    identity,
                    JobState.PAUSED,
                    status.attempts,
                    status.started_at,
                    status.finished_at,
                    status.result,
                    status.error,
                )
                if status.state == JobState.PENDING
                else status
            )
            for identity, status in self._statuses.items()
        }
        self._save()

    def summary(self) -> dict[str, object]:
        """Return read-only state counts and currently running job identities."""
        counts = {
            state.value: sum(status.state == state for status in self._statuses.values())
            for state in JobState
        }
        return {
            "counts": counts,
            "jobs": len(self._statuses),
            "run_hash": self.run_hash,
            "running": [
                status.job_hash
                for status in self.statuses
                if status.state == JobState.RUNNING
            ],
            "schema_version": "imagenetr50-scheduler-summary-v1",
        }

    def job_manifest_rows(self) -> tuple[dict[str, object], ...]:
        """Project specifications and states into the required job manifest ledger."""
        return tuple(
            {
                **self._specs[identity].as_record(),
                "status": self._statuses[identity].as_record(),
            }
            for identity in sorted(self._specs)
        )

    def _save(self) -> None:
        record = {
            "jobs": [
                {
                    "spec": self._specs[identity].as_record(),
                    "status": self._statuses[identity].as_record(),
                }
                for identity in sorted(self._specs)
            ],
            "run_hash": self.run_hash,
            "schema_version": "imagenetr50-scheduler-state-v1",
        }
        atomic_write(self.path, canonical_json_bytes(record))

    def _load(self) -> None:
        if not self.path.is_file():
            return
        record = json.loads(self.path.read_text(encoding="utf-8"))
        if (
            record.get("schema_version") != "imagenetr50-scheduler-state-v1"
            or record.get("run_hash") != self.run_hash
        ):
            raise ValueError("scheduler state belongs to another protocol")
        for row in record["jobs"]:
            raw_spec, raw_status = row["spec"], row["status"]
            supplied_hash = raw_spec.pop("content_hash")
            spec = JobSpec.create(
                raw_spec["run_hash"],
                raw_spec["kind"],
                raw_spec["dependencies"],
                raw_spec["payload"],
            )
            if spec.content_hash != supplied_hash:
                raise ValueError("scheduler job specification changed")
            status = JobStatus(
                raw_status["job_hash"],
                JobState(raw_status["state"]),
                raw_status["attempts"],
                raw_status["started_at"],
                raw_status["finished_at"],
                raw_status["result"],
                raw_status["error"],
            )
            if status.job_hash != spec.content_hash:
                raise ValueError("scheduler status does not match its specification")
            self._specs[spec.content_hash] = spec
            self._statuses[spec.content_hash] = status


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = ["JobState", "JobStatus", "LocalScheduler"]

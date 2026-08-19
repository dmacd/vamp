"""Soft-pause, durable safe-marker, and guarded RunPod termination semantics."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

from apm.continual.artifacts import (
    atomic_write,
    canonical_json_bytes,
    file_sha256,
    load_canonical_json,
    publish_immutable_json,
)
from apm.continual.trace.jobs import JobLedger


SAFE_MARKER = "SAFE_TO_TERMINATE.json"


@dataclass(frozen=True, slots=True)
class RuntimeBudget:
    """Wall-clock budget with the preregistered 23h30m and 24h boundaries."""

    started_at: datetime
    soft_limit: timedelta = timedelta(hours=23, minutes=30)
    hard_limit: timedelta = timedelta(hours=24)

    def __post_init__(self) -> None:
        if self.started_at.tzinfo is None or not timedelta(0) < self.soft_limit < self.hard_limit:
            raise ValueError("runtime budget requires timezone and ordered positive limits")

    @property
    def soft_deadline(self) -> datetime:
        """Return the time after which no new jobs may dispatch."""
        return self.started_at + self.soft_limit

    @property
    def hard_deadline(self) -> datetime:
        """Return the final Pod-runtime deadline."""
        return self.started_at + self.hard_limit

    def should_pause(self, now: datetime | None = None) -> bool:
        """Return whether the soft dispatch limit has been reached."""
        return (now or datetime.now(timezone.utc)) >= self.soft_deadline

    def hard_expired(self, now: datetime | None = None) -> bool:
        """Return whether the hard runtime limit has been reached."""
        return (now or datetime.now(timezone.utc)) >= self.hard_deadline

    def as_record(self) -> dict[str, object]:
        """Return the restart-stable wall-clock budget contract."""
        return {
            "format": "trace-runtime-budget-v1",
            "hard_deadline": self.hard_deadline.isoformat(),
            "soft_deadline": self.soft_deadline.isoformat(),
            "started_at": self.started_at.isoformat(),
        }


def publish_safe_to_terminate(
    state_directory: str | Path,
    jobs: JobLedger,
    durable_reports: Sequence[str | Path],
) -> Path:
    """Publish the termination guard only after jobs quiesce and reports are durable."""
    if any(status.state == "RUNNING" for status in jobs.statuses):
        raise RuntimeError("cannot terminate while a TRACE worker is running")
    reports = tuple(Path(path) for path in durable_reports)
    if not reports or any(not path.is_file() for path in reports):
        raise RuntimeError("safe termination requires durable interim reports")
    target = Path(state_directory) / SAFE_MARKER
    record = {
            "format": "trace-safe-to-terminate-v1",
            "job_state_counts": jobs.state_counts(),
            "reports": [
                {"path": str(path), "sha256": file_sha256(path)} for path in reports
            ],
        }
    return atomic_write(target, canonical_json_bytes(record))


def terminate_current_runpod(
    safe_marker: str | Path,
    receipt_path: str | Path,
    request_sender: Callable[[Request, float], int] | None = None,
) -> Path:
    """Delete only the current Pod after validating the durable safety marker."""
    marker = load_canonical_json(safe_marker)
    if marker.get("format") != "trace-safe-to-terminate-v1":
        raise ValueError("RunPod termination marker is not authoritative")
    api_key = os.environ.get("RUNPOD_API_KEY")
    pod_id = os.environ.get("RUNPOD_POD_ID")
    if not api_key or not pod_id:
        raise RuntimeError("RUNPOD_API_KEY and RUNPOD_POD_ID are required for termination")
    request = Request(
        f"https://rest.runpod.io/v1/pods/{quote(pod_id, safe='')}",
        method="DELETE",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    status = (request_sender or _send_request)(request, 30.0)
    if status not in (200, 202, 204, 404):
        raise RuntimeError(f"RunPod termination returned HTTP {status}")
    return publish_immutable_json(
        receipt_path,
        {
            "format": "trace-runpod-termination-v1",
            "pod_id": pod_id,
            "safe_marker_sha256": file_sha256(safe_marker),
            "status": status,
        },
    )


def _send_request(request: Request, timeout: float) -> int:
    with urlopen(request, timeout=timeout) as response:
        return int(response.status)


__all__ = [
    "RuntimeBudget",
    "SAFE_MARKER",
    "publish_safe_to_terminate",
    "terminate_current_runpod",
]

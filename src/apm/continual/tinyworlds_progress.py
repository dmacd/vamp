"""Batched append-only progress contracts for the fixed TinyWorlds pilot."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path

from apm.continual.tinyworlds_report import TinyWorldsRecord


_PROGRESS_EVENTS = (
    "phase_started",
    "phase_progress",
    "phase_completed",
    "phase_failed",
)
_RESERVED_RESULT_FIELDS = {"sequence_index", "stage"}


@dataclass(frozen=True, slots=True)
class TinyWorldsProgressEvent:
    """One finite, human-correlatable phase/overall progress sample."""

    event: str
    phase: int
    phase_count: int
    name: str
    completed_units: float
    total_units: float
    elapsed_seconds: float
    eta_seconds: float

    def __post_init__(self) -> None:
        if self.event not in _PROGRESS_EVENTS:
            raise ValueError(f"unknown TinyWorlds progress event: {self.event}")
        if (
            type(self.phase) is not int
            or type(self.phase_count) is not int
            or not 1 <= self.phase <= self.phase_count
        ):
            raise ValueError("phase must lie within the positive phase count")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("progress phase name must not be empty")
        for field_name in (
            "completed_units",
            "total_units",
            "elapsed_seconds",
            "eta_seconds",
        ):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(f"{field_name} must be finite and nonnegative")
        if self.total_units <= 0.0 or self.completed_units > self.total_units:
            raise ValueError("progress units must satisfy 0 <= completed <= total")

    def as_dict(self) -> dict[str, object]:
        """Return the canonical JSON object written to progress.jsonl."""
        return {
            "completed_units": float(self.completed_units),
            "elapsed_seconds": float(self.elapsed_seconds),
            "eta_seconds": float(self.eta_seconds),
            "event": self.event,
            "name": self.name,
            "phase": self.phase,
            "phase_count": self.phase_count,
            "total_units": float(self.total_units),
        }


@dataclass(frozen=True, slots=True)
class TinyWorldsSequentialResult:
    """One append-only result emitted while canonical stages complete."""

    sequence_index: int
    stage: int
    payload: TinyWorldsRecord

    def __post_init__(self) -> None:
        if type(self.sequence_index) is not int or self.sequence_index < 0:
            raise ValueError("sequence_index must be a nonnegative integer")
        if type(self.stage) is not int or self.stage <= 0:
            raise ValueError("sequential-result stage must be positive")
        if not isinstance(self.payload, TinyWorldsRecord):
            raise TypeError("sequential-result payload must be a TinyWorldsRecord")
        overlap = _RESERVED_RESULT_FIELDS.intersection(self.payload.as_dict())
        if overlap:
            raise ValueError(
                f"sequential-result payload uses reserved fields: {sorted(overlap)}"
            )

    def as_dict(self) -> dict[str, object]:
        """Return the flattened canonical JSON object for persistence."""
        return {
            "sequence_index": self.sequence_index,
            "stage": self.stage,
            **self.payload.as_dict(),
        }


class TinyWorldsProgressWriter:
    """Buffer progress and sequential rows, then append durable JSONL batches."""

    def __init__(
        self,
        temporary_directory: str | Path,
        *,
        batch_size: int = 16,
    ) -> None:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("progress batch_size must be a positive integer")
        self.directory = Path(temporary_directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.progress_path = self.directory / "progress.jsonl"
        self.sequential_results_path = self.directory / "sequential_results.jsonl"
        if any(
            path.exists() and path.stat().st_size
            for path in (self.progress_path, self.sequential_results_path)
        ):
            raise FileExistsError(
                "progress writer requires fresh append-only output files"
            )
        self.batch_size = batch_size
        self._progress_buffer: list[TinyWorldsProgressEvent] = []
        self._result_buffer: list[TinyWorldsSequentialResult] = []
        self._last_phase = 0
        self._next_sequence_index = 0
        self._closed = False

    def append_progress(self, event: TinyWorldsProgressEvent) -> None:
        """Buffer one monotonic phase event and flush at the batch boundary."""
        self._ensure_open()
        if not isinstance(event, TinyWorldsProgressEvent):
            raise TypeError("progress rows must be TinyWorldsProgressEvent values")
        if event.phase < self._last_phase:
            raise ValueError("progress phases must be monotonic")
        self._last_phase = event.phase
        self._progress_buffer.append(event)
        if len(self._progress_buffer) >= self.batch_size:
            self.flush_progress()

    def append_sequential(self, result: TinyWorldsSequentialResult) -> None:
        """Buffer one exactly ordered result and flush at the batch boundary."""
        self._ensure_open()
        if not isinstance(result, TinyWorldsSequentialResult):
            raise TypeError(
                "sequential rows must be TinyWorldsSequentialResult values"
            )
        if result.sequence_index != self._next_sequence_index:
            raise ValueError(
                "sequential result indices must be appended contiguously"
            )
        self._next_sequence_index += 1
        self._result_buffer.append(result)
        if len(self._result_buffer) >= self.batch_size:
            self.flush_sequential()

    def flush_progress(self) -> None:
        """Append the pending progress batch in one durable write."""
        self._ensure_open()
        if self._progress_buffer:
            _append_jsonl_batch(
                self.progress_path,
                tuple(event.as_dict() for event in self._progress_buffer),
            )
            self._progress_buffer.clear()

    def flush_sequential(self) -> None:
        """Append the pending sequential-result batch in one durable write."""
        self._ensure_open()
        if self._result_buffer:
            _append_jsonl_batch(
                self.sequential_results_path,
                tuple(result.as_dict() for result in self._result_buffer),
            )
            self._result_buffer.clear()

    def flush(self) -> None:
        """Flush both JSONL families."""
        self.flush_progress()
        self.flush_sequential()

    def close(self) -> None:
        """Flush remaining rows and prevent further appends."""
        if not self._closed:
            self.flush()
            self._closed = True

    def __enter__(self) -> TinyWorldsProgressWriter:
        return self

    def __exit__(
        self,
        exception_type: object,
        exception: object,
        traceback: object,
    ) -> None:
        self.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("TinyWorlds progress writer is closed")


def _append_jsonl_batch(
    path: Path,
    rows: tuple[dict[str, object], ...],
) -> None:
    if not rows:
        raise ValueError("JSONL append batches must not be empty")
    payload = "".join(
        json.dumps(
            row,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
        for row in rows
    )
    with path.open("a", encoding="utf-8") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())


__all__ = [
    "TinyWorldsProgressEvent",
    "TinyWorldsProgressWriter",
    "TinyWorldsSequentialResult",
]

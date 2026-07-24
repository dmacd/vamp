"""Human-facing phase and overall ETA reporting for semantic-v1 workflows."""

from __future__ import annotations

from collections.abc import Mapping
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tqdm import tqdm as Tqdm

    from apm.data.text.tinyworlds_p.contracts import ProgressEvent


CONSTRUCTION_PHASE_WEIGHTS = {
    "archive": 45.0,
    "contexts": 25.0,
    "embedding": 25.0,
    "catalog": 5.0,
}
PARTITION_PHASE_WEIGHTS = {
    "archive": 20.0,
    "semantic-filter": 5.0,
    "topology": 5.0,
    "splits": 32.0,
    "pairing": 3.0,
    "shards": 25.0,
    "publish": 10.0,
}


class SemanticProgressReporter:
    """Render measured phase progress and a weighted whole-workflow ETA."""

    def __init__(self, description: str, phase_weights: Mapping[str, float]) -> None:
        from tqdm import tqdm

        if not description or not phase_weights or any(value <= 0.0 for value in phase_weights.values()):
            raise ValueError("semantic progress requires a description and positive phase weights")
        self._description = description
        self._phase_weights = tuple(phase_weights.items())
        self._started = time.monotonic()
        self._phase_started = self._started
        self._active_phase: str | None = None
        self._active_total = 0
        self._phase_progress: Tqdm | None = None
        self._overall: Tqdm = tqdm(
            total=sum(value for _, value in self._phase_weights),
            desc=description,
            unit="work",
            position=0,
        )

    def archive_event(self, event: ProgressEvent) -> None:
        """Adapt one archive/partition progress event to the semantic reporter."""
        self(event.phase, event.completed, event.total or 0, event.detail)

    def __call__(self, phase: str, completed: int, total: int, detail: str) -> None:
        """Advance the current phase and update phase/overall ETA text."""
        phases = tuple(name for name, _ in self._phase_weights)
        if phase not in phases:
            raise ValueError(f"unknown semantic workflow phase: {phase}")
        if type(completed) is not int or type(total) is not int or not 0 <= completed <= total:
            raise ValueError("semantic progress counts are inconsistent")
        if phase != self._active_phase:
            self._start_phase(phase, total, detail)
        if total != self._active_total:
            raise ValueError("semantic progress phase total changed")
        assert self._phase_progress is not None
        self._phase_progress.update(completed - self._phase_progress.n)
        self._phase_progress.set_postfix_str(detail, refresh=False)
        phase_fraction = 1.0 if total == 0 else completed / total
        phase_index = phases.index(phase)
        completed_work = sum(value for _, value in self._phase_weights[:phase_index])
        completed_work += self._phase_weights[phase_index][1] * phase_fraction
        self._overall.update(completed_work - self._overall.n)
        elapsed = time.monotonic() - self._started
        overall_fraction = completed_work / float(self._overall.total)
        phase_elapsed = time.monotonic() - self._phase_started
        phase_eta = _remaining_seconds(phase_elapsed, phase_fraction)
        overall_eta = _remaining_seconds(elapsed, overall_fraction)
        self._overall.set_postfix_str(
            f"phase ETA {_duration(phase_eta)}, overall ETA {_duration(overall_eta)}",
            refresh=False,
        )
        if completed == total:
            self._overall.write(
                f"[{phase}] {detail} | phase {_duration(phase_elapsed)} | "
                f"overall elapsed {_duration(elapsed)} | overall ETA {_duration(overall_eta)}"
            )

    def close(self) -> None:
        """Close any active phase and overall progress bars."""
        if self._phase_progress is not None:
            self._phase_progress.close()
            self._phase_progress = None
        self._overall.close()

    def _start_phase(self, phase: str, total: int, detail: str) -> None:
        from tqdm import tqdm

        if self._phase_progress is not None:
            self._phase_progress.close()
        self._active_phase = phase
        self._active_total = total
        self._phase_started = time.monotonic()
        unit = "B" if phase == "archive" else "item"
        self._phase_progress = tqdm(
            total=total,
            desc=f"  {phase}",
            unit=unit,
            unit_scale=phase == "archive",
            position=1,
            leave=False,
        )
        self._overall.write(f"[{phase}] {detail}")


def _remaining_seconds(elapsed: float, fraction: float) -> float | None:
    if fraction <= 0.0:
        return None
    return elapsed * max(0.0, 1.0 - fraction) / fraction


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "pending"
    rounded = max(0, round(seconds))
    hours, remainder = divmod(rounded, 3_600)
    minutes, remainder = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{remainder:02d}"


__all__ = [
    "CONSTRUCTION_PHASE_WEIGHTS",
    "PARTITION_PHASE_WEIGHTS",
    "SemanticProgressReporter",
]

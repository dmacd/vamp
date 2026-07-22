"""Human-facing progress reporting for long TinyWorlds-P partition builds."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tqdm import tqdm as Tqdm

    from apm.data.text.tinyworlds_p.contracts import ProgressEvent


class PreparationReporter:
    """Render phase and overall progress bars with adaptive ETA estimates."""

    _PHASE_MINUTES = {
        "archive": 6.0,
        "buckets": 4.0,
        "splits": 15.0,
        "shards": 5.5,
        "publish": 4.0,
    }

    def __init__(self) -> None:
        from tqdm import tqdm

        self._started = time.monotonic()
        self._phase_started = self._started
        self._active_phase: str | None = None
        self._phase_progress: Tqdm | None = None
        self._progress: Tqdm = tqdm(
            total=sum(self._PHASE_MINUTES.values()),
            desc="TinyWorlds-P preparation",
            unit="work",
            position=0,
        )

    def __call__(self, event: ProgressEvent) -> None:
        """Advance both bars from one bounded-work progress event."""
        if event.phase not in self._PHASE_MINUTES:
            raise ValueError(f"unknown preparation phase: {event.phase}")
        if event.phase != self._active_phase:
            from tqdm import tqdm

            if self._phase_progress is not None:
                self._phase_progress.close()
            self._active_phase = event.phase
            self._phase_started = time.monotonic()
            unit = "B" if event.phase == "archive" else (
                "record" if event.phase == "shards" else "step"
            )
            self._phase_progress = tqdm(
                total=event.total,
                desc=f"  {event.phase}",
                unit=unit,
                unit_scale=event.phase == "archive",
                position=1,
                leave=False,
            )
        assert self._phase_progress is not None
        self._phase_progress.update(event.completed - self._phase_progress.n)
        self._phase_progress.set_postfix_str(event.detail)
        phase_fraction = (
            0.0
            if not event.total
            else min(1.0, event.completed / event.total)
        )
        phases = tuple(self._PHASE_MINUTES)
        phase_index = phases.index(event.phase)
        completed_work = sum(
            self._PHASE_MINUTES[phase] for phase in phases[:phase_index]
        ) + self._PHASE_MINUTES[event.phase] * phase_fraction
        self._progress.update(completed_work - self._progress.n)
        elapsed = time.monotonic() - self._started
        overall_fraction = completed_work / sum(self._PHASE_MINUTES.values())
        overall_eta = (
            None
            if overall_fraction == 0.0
            else elapsed * (1.0 - overall_fraction) / overall_fraction
        )
        phase_elapsed = time.monotonic() - self._phase_started
        phase_eta = (
            None
            if phase_fraction == 0.0
            else phase_elapsed * (1.0 - phase_fraction) / phase_fraction
        )
        eta_text = lambda seconds: (
            "pending" if seconds is None else f"{seconds / 60:.1f}m"
        )
        self._progress.set_postfix_str(
            f"phase ETA {eta_text(phase_eta)}, overall ETA {eta_text(overall_eta)}"
        )
        self._progress.write(
            f"[{event.phase}] {event.detail} | elapsed {elapsed / 60:.1f}m | "
            f"phase ETA {eta_text(phase_eta)} | overall ETA {eta_text(overall_eta)}"
        )
        if event.phase == "publish" and event.completed == event.total:
            self.close()

    def close(self) -> None:
        """Close active bars after success or a planned stop."""
        if self._phase_progress is not None:
            self._phase_progress.close()
            self._phase_progress = None
        if not self._progress.disable:
            self._progress.close()


__all__ = ["PreparationReporter"]

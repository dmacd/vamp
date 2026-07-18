#!/usr/bin/env python
"""Run the fixed TinyWorlds v1 eight-task pilot and atomically publish it.

There are intentionally no research-choice command-line switches.  The GPU
executor is injected for tests and otherwise loaded from the fixed pilot run
module once that resource-bound implementation is available.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import tempfile
from threading import Event, Thread
from time import monotonic
from typing import TYPE_CHECKING, Protocol, TypeVar

if TYPE_CHECKING:
    from apm.continual.tinyworlds_progress import TinyWorldsProgressWriter
    from apm.continual.tinyworlds_report import TinyWorldsCompletedResult


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPOSITORY_ROOT / "results"


@dataclass(frozen=True, slots=True)
class PilotPhase:
    number: int
    name: str
    estimated_seconds: int


PHASES = (
    PilotPhase(1, "run the fixed interleaved eight-task pilot", 28_800),
    PilotPhase(2, "project the immutable completed result", 30),
    PilotPhase(3, "write and validate the report bundle", 120),
    PilotPhase(4, "atomically promote the completed bundle", 10),
)


class PilotExecutor(Protocol):
    def __call__(
        self,
        temporary_directory: Path,
        progress: "TinyWorldsProgressWriter",
    ) -> "TinyWorldsCompletedResult":
        """Return the immutable result of the one fixed pilot."""


class _TqdmBar(Protocol):
    n: float

    def update(self, amount: float = 1) -> object:
        """Advance the bar."""

    def close(self) -> None:
        """Close the bar."""

    def write(self, message: str) -> object:
        """Print a human-readable phase line."""


ResultT = TypeVar("ResultT")


class PilotProgress:
    """Human phase lines plus phase/overall ETA bars and persistent events."""

    def __init__(self, writer: "TinyWorldsProgressWriter") -> None:
        self._writer = writer
        self._overall: _TqdmBar | None = None
        self._tqdm_factory: Callable[..., _TqdmBar] | None = None

    def __enter__(self) -> PilotProgress:
        from tqdm.auto import tqdm

        self._tqdm_factory = tqdm
        self._overall = tqdm(
            total=sum(phase.estimated_seconds for phase in PHASES),
            desc="TinyWorlds pilot overall",
            unit="est-s",
            position=0,
            dynamic_ncols=True,
            leave=True,
        )
        return self

    def __exit__(
        self,
        exception_type: object,
        exception: object,
        traceback: object,
    ) -> None:
        if self._overall is not None:
            self._overall.close()

    def run(self, phase: PilotPhase, operation: Callable[[], ResultT]) -> ResultT:
        """Run one fixed phase while emitting persistent progress metadata."""
        from apm.continual.tinyworlds_progress import TinyWorldsProgressEvent

        if self._overall is None or self._tqdm_factory is None:
            raise RuntimeError("pilot progress must be entered before use")
        self._overall.write(f"Phase {phase.number}/{len(PHASES)}: {phase.name}")
        started = monotonic()
        self._writer.append_progress(
            TinyWorldsProgressEvent(
                event="phase_started",
                phase=phase.number,
                phase_count=len(PHASES),
                name=phase.name,
                completed_units=0.0,
                total_units=float(phase.estimated_seconds),
                elapsed_seconds=0.0,
                eta_seconds=float(phase.estimated_seconds),
            )
        )
        self._writer.flush_progress()
        phase_bar = self._tqdm_factory(
            total=phase.estimated_seconds,
            desc=f"Phase {phase.number}/{len(PHASES)} ETA",
            unit="est-s",
            position=1,
            dynamic_ncols=True,
            leave=False,
        )
        stop = Event()
        timer = Thread(
            target=_advance_eta_bars,
            args=(stop, phase_bar, self._overall, phase.estimated_seconds),
            daemon=True,
        )
        timer.start()
        succeeded = False
        try:
            result = operation()
        except BaseException:
            elapsed = monotonic() - started
            self._writer.append_progress(
                TinyWorldsProgressEvent(
                    event="phase_failed",
                    phase=phase.number,
                    phase_count=len(PHASES),
                    name=phase.name,
                    completed_units=min(float(phase_bar.n), phase.estimated_seconds),
                    total_units=float(phase.estimated_seconds),
                    elapsed_seconds=elapsed,
                    eta_seconds=max(0.0, phase.estimated_seconds - phase_bar.n),
                )
            )
            self._writer.flush_progress()
            raise
        else:
            succeeded = True
            elapsed = monotonic() - started
            self._writer.append_progress(
                TinyWorldsProgressEvent(
                    event="phase_completed",
                    phase=phase.number,
                    phase_count=len(PHASES),
                    name=phase.name,
                    completed_units=float(phase.estimated_seconds),
                    total_units=float(phase.estimated_seconds),
                    elapsed_seconds=elapsed,
                    eta_seconds=0.0,
                )
            )
            self._writer.flush_progress()
            return result
        finally:
            stop.set()
            timer.join()
            if succeeded:
                remaining = max(0.0, phase.estimated_seconds - phase_bar.n)
                phase_bar.update(remaining)
                self._overall.update(remaining)
            phase_bar.close()


def _advance_eta_bars(
    stop: Event,
    phase_bar: _TqdmBar,
    overall_bar: _TqdmBar,
    estimated_seconds: int,
) -> None:
    while not stop.wait(1.0):
        if phase_bar.n < estimated_seconds - 1:
            phase_bar.update(1)
            overall_bar.update(1)


def _default_executor(
    temporary_directory: Path,
    progress: "TinyWorldsProgressWriter",
) -> "TinyWorldsCompletedResult":
    try:
        from apm.continual.tinyworlds_pilot_run import run_fixed_tinyworlds_pilot
    except ModuleNotFoundError as error:
        if error.name != "apm.continual.tinyworlds_pilot_run":
            raise
        raise RuntimeError(
            "the resource-bound fixed TinyWorlds pilot executor is not available"
        ) from error
    return run_fixed_tinyworlds_pilot(temporary_directory, progress)


def main(
    *,
    executor: PilotExecutor | None = None,
    results_root: str | Path = RESULTS_ROOT,
) -> Path:
    """Run, validate, and publish the single fixed pilot configuration."""
    runtime_root = Path(results_root) / ".tinyworlds-runtime"
    runtime_root.mkdir(parents=True, exist_ok=True)
    temporary_directory = Path(
        tempfile.mkdtemp(prefix="tinyworlds-v1-pilot-", dir=runtime_root)
    )
    # This must remain the first user-visible output so an interrupted run is
    # immediately inspectable.
    print(temporary_directory, flush=True)
    from apm.continual.tinyworlds_progress import TinyWorldsProgressWriter
    from apm.continual.tinyworlds_report import (
        atomically_promote_tinyworlds_report,
        build_tinyworlds_report_bundle,
        write_tinyworlds_report,
    )

    report_staging = temporary_directory / "report"
    fixed_executor = executor or _default_executor
    with TinyWorldsProgressWriter(
        temporary_directory,
        batch_size=16,
    ) as persistent_progress:
        with PilotProgress(persistent_progress) as progress:
            completed = progress.run(
                PHASES[0],
                lambda: fixed_executor(temporary_directory, persistent_progress),
            )
            bundle = progress.run(
                PHASES[1],
                lambda: build_tinyworlds_report_bundle(completed),
            )
            progress.run(
                PHASES[2],
                lambda: write_tinyworlds_report(report_staging, bundle),
            )
            destination = progress.run(
                PHASES[3],
                lambda: atomically_promote_tinyworlds_report(
                    report_staging,
                    results_root,
                    bundle,
                ),
            )
    return destination


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Materialize or verify both fixed TinyWorlds v1 rendered bundles."""

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
    from apm.data.text.tinyworlds.materialization import RenderedWorldMaterialization


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TINYWORLDS_ROOT = REPOSITORY_ROOT / "data" / "tinyworlds" / "v1"
RENDERED_ROOT = TINYWORLDS_ROOT / "rendered"
TOKENIZER_PATH = (
    REPOSITORY_ROOT / "checkpoints" / "tinystories-8m" / "tokenizer" / "tokenizer.json"
)


@dataclass(frozen=True, slots=True)
class MaterializationPhase:
    number: int
    name: str
    estimated_seconds: int


PHASES = (
    MaterializationPhase(1, "load the pinned tokenizer", 15),
    MaterializationPhase(2, "materialize or verify the calibration world", 3_600),
    MaterializationPhase(3, "materialize or verify the pilot world", 7_200),
    MaterializationPhase(4, "write the canonical materialization result", 10),
)


class _TqdmBar(Protocol):
    n: float

    def update(self, amount: float = 1) -> object:
        """Advance the bar."""

    def close(self) -> None:
        """Close the bar."""

    def write(self, message: str) -> object:
        """Write one human-readable phase line."""


ResultT = TypeVar("ResultT")


class MaterializationProgress:
    """Human phase lines, ETA bars, and durable phase boundary records."""

    def __init__(self, writer: "TinyWorldsProgressWriter") -> None:
        self._writer = writer
        self._overall: _TqdmBar | None = None
        self._tqdm_factory: Callable[..., _TqdmBar] | None = None

    def __enter__(self) -> MaterializationProgress:
        from tqdm.auto import tqdm

        self._tqdm_factory = tqdm
        self._overall = tqdm(
            total=sum(phase.estimated_seconds for phase in PHASES),
            desc="TinyWorlds rendering overall",
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

    def run(
        self,
        phase: MaterializationPhase,
        operation: Callable[[], ResultT],
    ) -> ResultT:
        """Run one fixed phase and durably record its terminal state."""
        from apm.continual.tinyworlds_progress import TinyWorldsProgressEvent

        if self._overall is None or self._tqdm_factory is None:
            raise RuntimeError("materialization progress must be entered before use")
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


def main() -> Path:
    """Materialize or verify the one canonical calibration/pilot pair."""
    runtime_root = RENDERED_ROOT / ".runtime"
    runtime_root.mkdir(parents=True, exist_ok=True)
    temporary_directory = Path(
        tempfile.mkdtemp(prefix="tinyworlds-v1-rendered-", dir=runtime_root)
    )
    # Keep this as the first user-visible output so interrupted work is locatable.
    print(temporary_directory, flush=True)

    from apm.continual.tinyworlds_progress import (
        TinyWorldsProgressWriter,
        TinyWorldsSequentialResult,
    )
    from apm.continual.tinyworlds_report import TinyWorldsRecord
    from apm.data.text.tinyworlds.materialization import (
        build_rendered_materialization_result,
        file_sha256,
        materialize_or_verify_rendered_world,
        write_rendered_materialization_result,
    )
    from apm.data.text.tinyworlds.rendering import TINYWORLDS_RENDER_PRESET
    from apm.lm.text import TokenizersTextTokenizer

    with TinyWorldsProgressWriter(
        temporary_directory,
        batch_size=16,
    ) as persistent_progress:
        with MaterializationProgress(persistent_progress) as progress:
            tokenizer, tokenizer_sha256 = progress.run(
                PHASES[0],
                lambda: (
                    TokenizersTextTokenizer.from_file(TOKENIZER_PATH),
                    file_sha256(TOKENIZER_PATH),
                ),
            )
            def append_outcome(
                world_index: int,
                outcome: "RenderedWorldMaterialization",
            ) -> None:
                persistent_progress.append_sequential(
                    TinyWorldsSequentialResult(
                        sequence_index=world_index,
                        stage=world_index + 1,
                        payload=TinyWorldsRecord(
                            entries=(
                                ("action", outcome.action),
                                ("query_group_count", outcome.artifact.query_group_count),
                                (
                                    "rendered_bundle_sha256",
                                    outcome.artifact.rendered_bundle_sha256,
                                ),
                                ("story_count", outcome.artifact.story_count),
                                (
                                    "symbolic_bundle_sha256",
                                    outcome.artifact.symbolic_bundle_sha256,
                                ),
                                ("world_name", outcome.artifact.world_name),
                            )
                        ),
                    )
                )
                persistent_progress.flush_sequential()

            def run_world(
                world_index: int,
                world_name: str,
            ) -> "RenderedWorldMaterialization":
                outcome = progress.run(
                    PHASES[world_index + 1],
                    lambda world_name=world_name: materialize_or_verify_rendered_world(
                        world_name,
                        TINYWORLDS_ROOT / world_name,
                        RENDERED_ROOT / world_name,
                        tokenizer,
                        TINYWORLDS_RENDER_PRESET,
                    ),
                )
                append_outcome(world_index, outcome)
                return outcome

            outcomes = tuple(
                run_world(world_index, world_name)
                for world_index, world_name in enumerate(("calibration", "pilot"))
            )

            result = build_rendered_materialization_result(
                tokenizer_sha256,
                outcomes,
            )
            result_path = progress.run(
                PHASES[3],
                lambda: write_rendered_materialization_result(
                    result,
                    temporary_directory,
                ),
            )
    print(result.canonical_json, flush=True)
    return result_path


if __name__ == "__main__":
    main()

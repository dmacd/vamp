"""Deterministic post-merge replay-repair planning and execution."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from apm.continual.trace.data import TraceExample
from apm.continual.trace.protocol import TrainingConfig
from apm.continual.trace.reservoirs import ReservoirEntry, merge_reservoirs
from apm.continual.trace.training_jobs import (
    TrainingArtifactResult,
    repair_training_config,
    run_training_artifact,
)
from apm.continual.trace.training_plans import repair_plan


def run_repair_training(
    left_reservoir: Sequence[ReservoirEntry],
    right_reservoir: Sequence[ReservoirEntry],
    represented_example_count: int,
    repair_fraction: float,
    node_id: str,
    merged_adapter: str | Path,
    examples_by_id: Mapping[str, TraceExample],
    model_revision: str,
    device: str,
    target_directory: str | Path,
    run_directory: str | Path,
    extra_records: tuple[tuple[str, Mapping[str, object]], ...] = (),
    should_pause: Callable[[], bool] = lambda: False,
    config: TrainingConfig | None = None,
) -> tuple[TrainingArtifactResult, tuple[ReservoirEntry, ...]]:
    """Repair a merged rank-bounded adapter for one epoch and publish it immutably."""
    repair_entries, parent_reservoir = merge_reservoirs(
        left_reservoir,
        right_reservoir,
        represented_example_count,
        repair_fraction,
    )
    plan = repair_plan(
        examples_by_id,
        tuple(entry.example_id for entry in repair_entries),
        node_id,
    )
    root = Path(run_directory)
    result = run_training_artifact(
        plan=plan,
        examples_by_id=examples_by_id,
        model_revision=model_revision,
        device=device,
        target_directory=target_directory,
        checkpoint_path=root / "checkpoints" / f"repair-{node_id}.pt",
        ledger_path=root / "logs" / f"repair-{node_id}.jsonl",
        work_root=root / "work",
        config=config or repair_training_config(),
        initial_adapter=merged_adapter,
        extra_records=extra_records,
        should_pause=should_pause,
    )
    return result, parent_reservoir


__all__ = ["run_repair_training"]

"""Level-zero TRACE leaf identity and training entrypoint."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from apm.continual.artifacts import record_sha256
from apm.continual.trace.data import TraceExample
from apm.continual.trace.reservoirs import prioritized_entries
from apm.continual.trace.training_jobs import TrainingArtifactResult, run_training_artifact
from apm.continual.trace.training_plans import leaf_plan


def arrival_identity(examples: Sequence[TraceExample], arrival: int) -> str:
    """Return the permanent identity of one ordered 100-example arrival."""
    identities = tuple(
        sorted(example.example_id for example in examples if example.arrival == arrival)
    )
    if len(identities) != 100:
        raise ValueError("TRACE arrival identity requires exactly 100 examples")
    return record_sha256(
        {
            "arrival": arrival,
            "example_ids": list(identities),
            "format": "trace-arrival-v1",
        }
    )


def run_leaf_training(
    examples: Sequence[TraceExample],
    arrival: int,
    model_revision: str,
    device: str,
    run_directory: str | Path,
    should_pause: Callable[[], bool] = lambda: False,
) -> TrainingArtifactResult:
    """Train or resume one immutable fresh base-relative level-zero LoRA."""
    root = Path(run_directory)
    identity = arrival_identity(examples, arrival)
    plan = leaf_plan(examples, arrival)
    arrival_examples = tuple(example for example in examples if example.arrival == arrival)
    priorities = prioritized_entries(example.example_id for example in arrival_examples)
    return run_training_artifact(
        plan=plan,
        examples_by_id={example.example_id: example for example in examples},
        model_revision=model_revision,
        device=device,
        target_directory=root / "leaves" / identity,
        checkpoint_path=root / "checkpoints" / f"leaf-{arrival:02d}.pt",
        ledger_path=root / "logs" / f"leaf-{arrival:02d}.jsonl",
        work_root=root / "work",
        should_pause=should_pause,
        extra_records=((
            "reservoir_priorities.json",
            {
                "entries": [
                    {"example_id": entry.example_id, "priority": entry.priority}
                    for entry in priorities
                ],
                "format": "trace-leaf-reservoir-priorities-v1",
            },
        ),),
    )


__all__ = ["arrival_identity", "run_leaf_training"]

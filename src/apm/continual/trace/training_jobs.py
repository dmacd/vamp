"""End-to-end immutable leaf, repair, and baseline training artifacts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from apm.continual.artifacts import load_canonical_json, publish_immutable_json
from apm.continual.trace.adapter_io import load_adapter_state, save_adapter
from apm.continual.trace.artifacts import (
    publish_artifact_directory,
    validate_artifact_directory,
)
from apm.continual.trace.data import TraceExample
from apm.continual.trace.modeling import load_fresh_lora_bundle
from apm.continual.trace.protocol import TrainingConfig
from apm.continual.trace.training import TrainingResult, train_adapter
from apm.continual.trace.training_plans import TrainingPlan


@dataclass(frozen=True, slots=True)
class TrainingArtifactResult:
    """Identity and accounting for one published adapter training artifact."""

    artifact_sha256: str
    artifact_directory: Path
    training: TrainingResult


def run_training_artifact(
    plan: TrainingPlan,
    examples_by_id: Mapping[str, TraceExample],
    model_revision: str,
    device: str,
    target_directory: str | Path,
    checkpoint_path: str | Path,
    ledger_path: str | Path,
    work_root: str | Path,
    config: TrainingConfig = TrainingConfig(),
    initial_adapter: str | Path | None = None,
    snapshot_root: str | Path | None = None,
    extra_records: tuple[tuple[str, Mapping[str, object]], ...] = (),
    should_pause: Callable[[], bool] = lambda: False,
) -> TrainingArtifactResult:
    """Train or resume one adapter, then atomically publish its immutable bundle."""
    target = Path(target_directory)
    if target.is_dir():
        identity = validate_artifact_directory(target)
        metrics = load_canonical_json(target / "train_metrics.json")
        if metrics.get("plan_hash") != plan.plan_hash:
            raise ValueError("published training artifact belongs to another plan")
        return TrainingArtifactResult(
            artifact_sha256=identity,
            artifact_directory=target,
            training=TrainingResult(
                plan_hash=str(metrics["plan_hash"]),
                presentations=int(metrics["presentations"]),
                tokens=int(metrics["tokens"]),
                optimizer_steps=int(metrics["optimizer_steps"]),
                mean_loss=float(metrics["mean_loss"]),
                elapsed_seconds=float(metrics["elapsed_seconds"]),
                checkpoint_path=Path(str(metrics["checkpoint_path"])),
            ),
        )
    bundle = load_fresh_lora_bundle(model_revision, device, plan.plan_hash, config)
    if initial_adapter is not None and not Path(checkpoint_path).is_file():
        load_adapter_state(bundle.model, initial_adapter)
    def publish_snapshot(
        model: torch.nn.Module,
        phase_name: str,
        next_presentation: int,
    ) -> None:
        if snapshot_root is None:
            return
        snapshot_target = Path(snapshot_root) / phase_name
        with TemporaryDirectory(prefix=f"snapshot-{phase_name}-", dir=work_root) as temporary:
            snapshot_output = Path(temporary)
            save_adapter(model, snapshot_output, config)
            publish_immutable_json(
                snapshot_output / "snapshot.json",
                {
                    "format": "trace-training-snapshot-v1",
                    "next_presentation": next_presentation,
                    "phase": phase_name,
                    "plan_hash": plan.plan_hash,
                },
            )
            publish_artifact_directory(snapshot_output, snapshot_target)

    Path(work_root).mkdir(parents=True, exist_ok=True)
    training = train_adapter(
        bundle.model,
        bundle.tokenizer,
        examples_by_id,
        plan,
        checkpoint_path,
        ledger_path,
        config,
        should_pause=should_pause,
        on_phase_boundary=publish_snapshot,
    )
    with TemporaryDirectory(prefix=f"{plan.name}-", dir=work_root) as temporary:
        output = Path(temporary)
        save_adapter(bundle.model, output, config)
        publish_immutable_json(
            output / "train_metrics.json",
            {
                "format": "trace-training-metrics-v1",
                **training.as_record(),
            },
        )
        publish_immutable_json(
            output / "source_ids.json",
            {
                "format": "trace-training-sources-v1",
                "plan_hash": plan.plan_hash,
                "presentation_count": len(plan.example_ids),
                "unique_example_ids": sorted(set(plan.example_ids)),
            },
        )
        for filename, record in sorted(extra_records):
            if Path(filename).name != filename or not filename.endswith(".json"):
                raise ValueError("extra training artifact records require simple JSON names")
            publish_immutable_json(output / filename, record)
        artifact_sha256 = publish_artifact_directory(output, target)
    return TrainingArtifactResult(
        artifact_sha256=artifact_sha256,
        artifact_directory=target,
        training=training,
    )


def repair_training_config(
    *,
    rank: int = 8,
    alpha: int = 32,
    learning_rate: float = 5.0e-5,
) -> TrainingConfig:
    """Return repair settings matching the merged adapter's rank and scale."""
    return replace(
        TrainingConfig(),
        rank=rank,
        alpha=alpha,
        learning_rate=learning_rate,
    )


__all__ = [
    "TrainingArtifactResult",
    "repair_training_config",
    "run_training_artifact",
]

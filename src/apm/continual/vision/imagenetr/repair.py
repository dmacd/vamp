"""Deterministic bounded one-epoch repair of an already merged rank-16 parent."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch

from apm.continual.vision.imagenetr.artifacts import NodeBundle
from apm.continual.vision.imagenetr.config import ImageNetRConfig
from apm.continual.vision.imagenetr.data import DatasetManifest
from apm.continual.vision.imagenetr.heads import ClassifierRows
from apm.continual.vision.imagenetr.lora import load_adapter_factors
from apm.continual.vision.imagenetr.model import AdapterVisionModel
from apm.continual.vision.imagenetr.proxy_memory import repair_reservoir, require_training_only
from apm.continual.vision.imagenetr.training import TrainingResult, train_adapter_model


def repair_parent(
    parent: NodeBundle,
    manifest: DatasetManifest,
    prepared_root: str | Path,
    config: ImageNetRConfig,
    repair_fraction: float,
    backbone_factory: Callable[[], torch.nn.Module],
    train_transform: object,
    device: torch.device,
    checkpoint_path: str | Path,
    show_progress: bool = True,
) -> tuple[TrainingResult, tuple[str, ...]]:
    """Initialize from one merge, train on bottom-K history, and return separate state."""
    reservoir = repair_reservoir(
        manifest, parent.artifact.represented_task_ids, repair_fraction
    )
    if not reservoir.image_ids:
        raise ValueError("repair_parent is only valid for a positive repair budget")
    require_training_only(reservoir.image_ids, manifest, "repair")
    rows = manifest.select("train", image_ids=reservoir.image_ids)
    if tuple(sorted(row.image_id for row in rows)) != tuple(sorted(reservoir.image_ids)):
        raise ValueError("repair reservoir cannot be resolved from the dataset manifest")
    model = AdapterVisionModel(
        backbone_factory(),
        parent.classifier.class_ids,
        config.lora_rank,
        config.lora_alpha,
        config.lora_dropout,
        config.seed + 700_000 + parent.artifact.first_task,
        parent.classifier,
    )
    load_adapter_factors(model, parent.adapter)
    result = train_adapter_model(
        model,
        prepared_root,
        rows,
        train_transform,
        config.repair_training,
        config.seed + 800_000 + parent.artifact.first_task,
        device,
        checkpoint_path,
        num_workers=config.num_workers,
        checkpoint_steps=config.checkpoint_steps,
        show_progress=show_progress,
    )
    return result, reservoir.image_ids


__all__ = ["repair_parent"]

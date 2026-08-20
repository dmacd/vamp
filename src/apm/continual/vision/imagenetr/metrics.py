"""Class-incremental accuracy, forgetting, gaps, and resource accounting."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
import math

import torch
from torch import Tensor

from apm.continual.vision.imagenetr.routing import GroundTruth


def accuracy(predictions: Tensor, truth: GroundTruth) -> float:
    """Return percentage top-one accuracy after isolated task-free prediction."""
    if predictions.ndim != 1 or predictions.shape != truth.labels.shape:
        raise ValueError("predictions and ground truth do not align")
    return 100.0 * float(torch.mean((predictions.cpu() == truth.labels.cpu()).float()).item())


def incremental_average(stage_accuracies: Sequence[float]) -> float:
    """Return mean class-incremental accuracy over all completed stages."""
    if not stage_accuracies or any(not math.isfinite(value) for value in stage_accuracies):
        raise ValueError("stage accuracies must be finite and nonempty")
    return math.fsum(stage_accuracies) / len(stage_accuracies)


def mean_forgetting(task_accuracy_matrix: Sequence[Sequence[float | None]]) -> float:
    """Return mean per-task peak-to-final accuracy decline."""
    rows = tuple(tuple(row) for row in task_accuracy_matrix)
    if not rows:
        raise ValueError("task accuracy matrix is empty")
    task_count = len(rows[-1])
    if any(len(row) != task_count for row in rows):
        raise ValueError("task accuracy matrix rows have inconsistent widths")
    forgettings = []
    for task in range(task_count):
        history = tuple(float(row[task]) for row in rows if row[task] is not None)
        if history:
            forgettings.append(max(history) - history[-1])
    if len(forgettings) != task_count:
        raise ValueError("task accuracy matrix lacks a task history")
    return math.fsum(forgettings) / task_count


@dataclass(frozen=True, slots=True)
class ResourceMetrics:
    """Separated gradient, forward-only, live-memory, archive, and timing costs."""

    training_image_presentations: int
    repair_image_presentations: int
    optimizer_steps: int
    proxy_images: int
    proxy_forward_presentations: int
    wall_seconds: float
    peak_vram_bytes: int
    live_lora_parameters: int
    archived_lora_parameters: int
    live_proxy_images: int
    live_repair_images: int
    final_live_nodes: int
    average_live_nodes: float
    final_candidate_forwards: int
    average_candidate_forwards: float

    def __post_init__(self) -> None:
        values = (
            self.training_image_presentations,
            self.repair_image_presentations,
            self.optimizer_steps,
            self.proxy_images,
            self.proxy_forward_presentations,
            self.peak_vram_bytes,
            self.live_lora_parameters,
            self.archived_lora_parameters,
            self.live_proxy_images,
            self.live_repair_images,
            self.final_live_nodes,
            self.final_candidate_forwards,
        )
        if any(value < 0 for value in values) or self.wall_seconds < 0.0:
            raise ValueError("resource metrics cannot be negative")

    def as_record(self) -> dict[str, object]:
        """Return JSON resource accounting with archive/live concepts separated."""
        return {
            "archived_lora_parameters": self.archived_lora_parameters,
            "average_candidate_forwards": self.average_candidate_forwards,
            "average_live_nodes": self.average_live_nodes,
            "final_candidate_forwards": self.final_candidate_forwards,
            "final_live_nodes": self.final_live_nodes,
            "live_lora_parameters": self.live_lora_parameters,
            "live_proxy_images": self.live_proxy_images,
            "live_repair_images": self.live_repair_images,
            "optimizer_steps": self.optimizer_steps,
            "peak_vram_bytes": self.peak_vram_bytes,
            "proxy_forward_presentations": self.proxy_forward_presentations,
            "proxy_images": self.proxy_images,
            "repair_image_presentations": self.repair_image_presentations,
            "training_image_presentations": self.training_image_presentations,
            "wall_seconds": self.wall_seconds,
        }


__all__ = [
    "ResourceMetrics",
    "accuracy",
    "incremental_average",
    "mean_forgetting",
]

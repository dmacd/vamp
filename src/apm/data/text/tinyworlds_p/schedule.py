"""Source-independent scratch-training schedule and selection policy."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Literal, Protocol, Sequence

from apm.data.text.tinyworlds_p.contracts import PartitionPreset, WORLD_LABELS


GridDecision = Literal[
    "pass",
    "fallback_6x6",
    "fallback_10x10",
    "training_quality_failure",
]


class LearningRateConfig(Protocol):
    """The optimizer fields required by the fixed scalar schedule."""

    maximum_learning_rate: float
    minimum_learning_rate: float
    warmup_fraction: float


@dataclass(frozen=True, slots=True)
class WorldGap:
    """One world validation NLL, control NLL, and their held-out gap."""

    world: str
    world_nll: float
    control_nll: float

    def __post_init__(self) -> None:
        if self.world not in WORLD_LABELS:
            raise ValueError("world gap requires a canonical world label")
        if any(
            not math.isfinite(value) or value < 0.0
            for value in (self.world_nll, self.control_nll)
        ):
            raise ValueError("world and control NLL must be finite and nonnegative")

    @property
    def gap(self) -> float:
        """Return world NLL minus its matched held-in control NLL."""
        return self.world_nll - self.control_nll


@dataclass(frozen=True, slots=True)
class EpochValidation:
    """Held-in and five-world validation measurements at one checkpoint."""

    epoch: int
    held_in_nll: float
    world_gaps: tuple[WorldGap, ...]
    allocator_peak_bytes: int

    def __post_init__(self) -> None:
        if type(self.epoch) is not int or self.epoch <= 0:
            raise ValueError("validation epoch must be positive")
        if not math.isfinite(self.held_in_nll) or self.held_in_nll < 0.0:
            raise ValueError("held-in NLL must be finite and nonnegative")
        if tuple(item.world for item in self.world_gaps) != WORLD_LABELS:
            raise ValueError("epoch validation requires worlds A through E in order")
        if type(self.allocator_peak_bytes) is not int or self.allocator_peak_bytes < 0:
            raise ValueError("allocator peak must be nonnegative")

    @property
    def mean_gap(self) -> float:
        """Return the mean of the five matched validation gaps."""
        return sum(item.gap for item in self.world_gaps) / len(self.world_gaps)


def cosine_learning_rate(
    update: int,
    total_updates: int,
    config: LearningRateConfig,
) -> float:
    """Apply 1% linear warmup then cosine decay with exact boundary values."""
    if type(total_updates) is not int or total_updates <= 0:
        raise ValueError("total_updates must be positive")
    if type(update) is not int or update < 0 or update >= total_updates:
        raise ValueError("update must lie inside the planned schedule")
    warmup_updates = max(1, math.ceil(config.warmup_fraction * total_updates))
    if update < warmup_updates:
        return config.maximum_learning_rate * (update + 1) / warmup_updates
    decay_denominator = max(1, total_updates - warmup_updates - 1)
    decay_progress = min(max((update - warmup_updates) / decay_denominator, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return config.minimum_learning_rate + (
        config.maximum_learning_rate - config.minimum_learning_rate
    ) * cosine


def calibration_grid_decision(
    epoch_one: EpochValidation,
    epoch_two: EpochValidation,
    allocator_peak_limit_bytes: int,
) -> GridDecision:
    """Apply the one-shot gap fallback policy without regridding quality failures."""
    if epoch_one.epoch != 1 or epoch_two.epoch != 2:
        raise ValueError("calibration requires epoch-one and epoch-two validations")
    if (
        epoch_two.allocator_peak_bytes > allocator_peak_limit_bytes
        or epoch_two.held_in_nll > 2.2
        or epoch_one.held_in_nll - epoch_two.held_in_nll < 0.02
    ):
        return "training_quality_failure"
    if epoch_two.mean_gap > 0.30:
        return "fallback_10x10"
    if epoch_two.mean_gap < 0.08 or any(item.gap < 0.05 for item in epoch_two.world_gaps):
        return "fallback_6x6"
    return "pass"


def epoch_satisfies_gap_gates(validation: EpochValidation) -> bool:
    """Return whether one epoch can participate in checkpoint selection."""
    return 0.08 <= validation.mean_gap <= 0.30 and all(
        item.gap >= 0.05 for item in validation.world_gaps
    )


def fallback_partition_preset(
    decision: GridDecision,
    preset: PartitionPreset,
) -> PartitionPreset:
    """Return the single allowed fresh grid and its held-in control capacity."""
    if decision == "fallback_6x6":
        return replace(
            preset,
            bucket_count=6,
            base_split_weights=(94, 3, 3),
        )
    if decision == "fallback_10x10":
        return replace(
            preset,
            bucket_count=10,
            base_split_weights=(96, 2, 2),
        )
    raise ValueError("a fallback partition requires a low-gap or excessive-gap decision")


def select_best_eligible_epoch(
    validations: Sequence[EpochValidation],
) -> EpochValidation:
    """Choose lowest held-in NLL among eligible epochs, breaking ties earlier."""
    eligible = tuple(
        validation
        for validation in validations
        if 2 <= validation.epoch <= 5 and epoch_satisfies_gap_gates(validation)
    )
    if not eligible:
        raise ValueError("no epoch 2-5 checkpoint satisfies the validation gap gates")
    return min(eligible, key=lambda validation: (validation.held_in_nll, validation.epoch))


__all__ = [
    "EpochValidation",
    "GridDecision",
    "WorldGap",
    "calibration_grid_decision",
    "cosine_learning_rate",
    "epoch_satisfies_gap_gates",
    "fallback_partition_preset",
    "select_best_eligible_epoch",
]

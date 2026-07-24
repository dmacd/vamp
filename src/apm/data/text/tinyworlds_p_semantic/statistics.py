"""SHA-seeded paired bootstrap, label-swap placebos, Holm gates, and selection."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
from typing import Literal, Sequence

import numpy as np

from apm.data.text.tinyworlds_p_semantic.contracts import BENCHMARK_ID, WORLD_LABELS


GAP_THRESHOLD_NATS = math.log(1.05)
CANONICAL_REPLICATES = 10_000
CalibrationDecision = Literal[
    "pass",
    "semantic_grid_failure",
    "training_quality_failure",
]


@dataclass(frozen=True, slots=True)
class GroupLoss:
    """Float loss sum and active-token count for one duplicate group."""

    normalized_story_sha256: str
    loss_sum: float
    active_tokens: int

    def __post_init__(self) -> None:
        if (
            type(self.normalized_story_sha256) is not str
            or len(self.normalized_story_sha256) != 64
        ):
            raise ValueError("group loss requires a SHA-256 identity")
        if not math.isfinite(self.loss_sum) or self.loss_sum < 0.0:
            raise ValueError("group loss sum must be finite and nonnegative")
        if type(self.active_tokens) is not int or self.active_tokens <= 0:
            raise ValueError("group loss active-token count must be positive")

    @property
    def nll(self) -> float:
        """Return this duplicate group's normalized NLL."""
        return self.loss_sum / self.active_tokens


@dataclass(frozen=True, slots=True)
class PairedLoss:
    """One persisted world/control group pair with loss evidence."""

    world: str
    world_loss: GroupLoss
    control_loss: GroupLoss

    def __post_init__(self) -> None:
        if self.world not in WORLD_LABELS:
            raise ValueError("paired loss requires a canonical world")
        if self.world_loss.normalized_story_sha256 == self.control_loss.normalized_story_sha256:
            raise ValueError("paired loss cannot compare a group to itself")


@dataclass(frozen=True, slots=True)
class EmpiricalGap:
    """Observed paired gap, bootstrap interval, and one-sided placebo probability."""

    observed_gap: float
    bootstrap_lower: float
    bootstrap_upper: float
    placebo_probability: float
    replicate_count: int

    def __post_init__(self) -> None:
        values = (self.observed_gap, self.bootstrap_lower, self.bootstrap_upper)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("empirical gap values must be finite")
        if self.bootstrap_lower > self.bootstrap_upper:
            raise ValueError("bootstrap interval is reversed")
        if not 0.0 <= self.placebo_probability <= 1.0:
            raise ValueError("placebo probability must lie in [0, 1]")
        if type(self.replicate_count) is not int or self.replicate_count <= 0:
            raise ValueError("empirical gap replicate count must be positive")


@dataclass(frozen=True, slots=True)
class WorldEmpiricalGap:
    """One world's paired empirical-null evidence."""

    world: str
    world_nll: float
    control_nll: float
    empirical: EmpiricalGap

    def __post_init__(self) -> None:
        if self.world not in WORLD_LABELS:
            raise ValueError("world empirical gap requires A-E")
        if any(not math.isfinite(value) or value < 0.0 for value in (self.world_nll, self.control_nll)):
            raise ValueError("world/control NLL must be finite and nonnegative")
        if not math.isclose(
            self.world_nll - self.control_nll,
            self.empirical.observed_gap,
            rel_tol=1e-10,
            abs_tol=1e-10,
        ):
            raise ValueError("world NLLs and observed paired gap disagree")


@dataclass(frozen=True, slots=True)
class SemanticEpochValidation:
    """Held-in quality and all semantic empirical-null evidence for one epoch."""

    epoch: int
    held_in_nll: float
    worlds: tuple[WorldEmpiricalGap, ...]
    mean_empirical: EmpiricalGap
    allocator_peak_bytes: int

    def __post_init__(self) -> None:
        if type(self.epoch) is not int or self.epoch <= 0:
            raise ValueError("semantic validation epoch must be positive")
        if not math.isfinite(self.held_in_nll) or self.held_in_nll < 0.0:
            raise ValueError("semantic held-in NLL must be finite and nonnegative")
        if tuple(item.world for item in self.worlds) != WORLD_LABELS:
            raise ValueError("semantic validation requires worlds A-E in order")
        if type(self.allocator_peak_bytes) is not int or self.allocator_peak_bytes < 0:
            raise ValueError("semantic allocator peak must be nonnegative")

    @property
    def mean_gap(self) -> float:
        """Return the unweighted mean of the five aggregate world gaps."""
        return sum(item.empirical.observed_gap for item in self.worlds) / len(self.worlds)


def paired_empirical_gap(
    pairs: Sequence[PairedLoss],
    identity_sha256: str,
    namespace: str,
    *,
    replicates: int = CANONICAL_REPLICATES,
) -> EmpiricalGap:
    """Compute one token-weighted paired bootstrap and within-pair label-swap null."""
    arrays = _pair_arrays(pairs)
    observed = _gap(arrays, None)
    bootstrap = _bootstrap_gaps(arrays, _rng(identity_sha256, namespace + ":bootstrap"), replicates)
    placebos = _placebo_gaps(arrays, _rng(identity_sha256, namespace + ":placebo"), replicates)
    return EmpiricalGap(
        observed_gap=observed,
        bootstrap_lower=float(np.quantile(bootstrap, 0.025, method="linear")),
        bootstrap_upper=float(np.quantile(bootstrap, 0.975, method="linear")),
        placebo_probability=float((1 + np.count_nonzero(placebos >= observed)) / (replicates + 1)),
        replicate_count=replicates,
    )


def summarize_empirical_gaps(
    pairs_by_world: Sequence[Sequence[PairedLoss]],
    identity_sha256: str,
    *,
    replicates: int = CANONICAL_REPLICATES,
) -> tuple[tuple[WorldEmpiricalGap, ...], EmpiricalGap]:
    """Compute five world statistics and the stratified unweighted mean statistic."""
    if len(pairs_by_world) != len(WORLD_LABELS):
        raise ValueError("empirical summary requires exactly five worlds")
    arrays_by_world = tuple(_pair_arrays(pairs) for pairs in pairs_by_world)
    world_results = tuple(
        WorldEmpiricalGap(
            world=world,
            world_nll=_arm_nll(arrays, world_arm=True),
            control_nll=_arm_nll(arrays, world_arm=False),
            empirical=paired_empirical_gap(
                pairs,
                identity_sha256,
                f"world:{world}",
                replicates=replicates,
            ),
        )
        for world, pairs, arrays in zip(
            WORLD_LABELS,
            pairs_by_world,
            arrays_by_world,
            strict=True,
        )
    )
    bootstrap = np.zeros(replicates, dtype=np.float64)
    placebos = np.zeros(replicates, dtype=np.float64)
    for world, arrays in zip(WORLD_LABELS, arrays_by_world, strict=True):
        bootstrap += _bootstrap_gaps(
            arrays,
            _rng(identity_sha256, f"mean:{world}:bootstrap"),
            replicates,
        )
        placebos += _placebo_gaps(
            arrays,
            _rng(identity_sha256, f"mean:{world}:placebo"),
            replicates,
        )
    bootstrap /= len(WORLD_LABELS)
    placebos /= len(WORLD_LABELS)
    observed = sum(item.empirical.observed_gap for item in world_results) / len(world_results)
    mean = EmpiricalGap(
        observed_gap=observed,
        bootstrap_lower=float(np.quantile(bootstrap, 0.025, method="linear")),
        bootstrap_upper=float(np.quantile(bootstrap, 0.975, method="linear")),
        placebo_probability=float((1 + np.count_nonzero(placebos >= observed)) / (replicates + 1)),
        replicate_count=replicates,
    )
    return world_results, mean


def holm_rejections(
    probabilities: Sequence[float],
    *,
    familywise_alpha: float = 0.05,
) -> tuple[bool, ...]:
    """Return aligned Holm step-down rejection decisions for one-sided tests."""
    if not probabilities or any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0
        for value in probabilities
    ):
        raise ValueError("Holm probabilities must be finite values in [0, 1]")
    if not 0.0 < familywise_alpha < 1.0:
        raise ValueError("Holm familywise alpha must lie in (0, 1)")
    ordered = sorted(enumerate(probabilities), key=lambda item: (item[1], item[0]))
    decisions = [False] * len(probabilities)
    continuing = True
    for rank, (index, probability) in enumerate(ordered):
        continuing = continuing and probability <= familywise_alpha / (len(ordered) - rank)
        decisions[index] = continuing
    return tuple(decisions)


def epoch_satisfies_semantic_gap_gate(validation: SemanticEpochValidation) -> bool:
    """Apply the frozen mean/world bootstrap, placebo, threshold, and Holm gates."""
    probabilities = tuple(item.empirical.placebo_probability for item in validation.worlds)
    return (
        validation.mean_empirical.observed_gap >= GAP_THRESHOLD_NATS
        and validation.mean_empirical.bootstrap_lower > 0.0
        and validation.mean_empirical.placebo_probability <= 0.01
        and all(
            item.empirical.observed_gap > 0.0
            and item.empirical.bootstrap_lower > 0.0
            for item in validation.worlds
        )
        and all(holm_rejections(probabilities, familywise_alpha=0.05))
    )


def calibration_decision(
    epoch_one: SemanticEpochValidation,
    epoch_two: SemanticEpochValidation,
    allocator_peak_limit_bytes: int,
) -> CalibrationDecision:
    """Stop failed semantic grids without regridding; pass only the frozen gate."""
    if (epoch_one.epoch, epoch_two.epoch) != (1, 2):
        raise ValueError("semantic calibration requires epochs one and two")
    if (
        epoch_two.allocator_peak_bytes > allocator_peak_limit_bytes
        or epoch_two.held_in_nll > 2.2
        or epoch_one.held_in_nll - epoch_two.held_in_nll < 0.02
    ):
        return "training_quality_failure"
    return "pass" if epoch_satisfies_semantic_gap_gate(epoch_two) else "semantic_grid_failure"


def select_best_eligible_epoch(
    validations: Sequence[SemanticEpochValidation],
) -> SemanticEpochValidation:
    """Choose lowest held-in NLL among semantic-gate epochs, breaking ties earlier."""
    eligible = tuple(
        item
        for item in validations
        if 2 <= item.epoch <= 5 and epoch_satisfies_semantic_gap_gate(item)
    )
    if not eligible:
        raise ValueError("no epoch 2-5 checkpoint satisfies the semantic-gap gate")
    return min(eligible, key=lambda item: (item.held_in_nll, item.epoch))


def _pair_arrays(pairs: Sequence[PairedLoss]) -> tuple[np.ndarray, ...]:
    if not pairs:
        raise ValueError("paired empirical statistics require at least one pair")
    worlds = {item.world for item in pairs}
    if len(worlds) != 1:
        raise ValueError("one paired statistic cannot mix worlds")
    ordered = tuple(
        sorted(
            pairs,
            key=lambda item: (
                item.world_loss.normalized_story_sha256,
                item.control_loss.normalized_story_sha256,
            ),
        )
    )
    return tuple(
        np.asarray(values, dtype=dtype)
        for values, dtype in (
            ([item.world_loss.loss_sum for item in ordered], np.float64),
            ([item.world_loss.active_tokens for item in ordered], np.float64),
            ([item.control_loss.loss_sum for item in ordered], np.float64),
            ([item.control_loss.active_tokens for item in ordered], np.float64),
        )
    )


def _gap(arrays: tuple[np.ndarray, ...], indexes: np.ndarray | None) -> float:
    world_loss, world_tokens, control_loss, control_tokens = arrays
    if indexes is not None:
        world_loss, world_tokens, control_loss, control_tokens = (
            values[indexes] for values in arrays
        )
    return float(
        np.sum(world_loss) / np.sum(world_tokens)
        - np.sum(control_loss) / np.sum(control_tokens)
    )


def _arm_nll(arrays: tuple[np.ndarray, ...], *, world_arm: bool) -> float:
    loss, tokens = arrays[:2] if world_arm else arrays[2:]
    return float(np.sum(loss) / np.sum(tokens))


def _bootstrap_gaps(
    arrays: tuple[np.ndarray, ...],
    rng: np.random.Generator,
    replicates: int,
) -> np.ndarray:
    if type(replicates) is not int or replicates <= 0:
        raise ValueError("bootstrap replicate count must be positive")
    count = len(arrays[0])
    result = np.empty(replicates, dtype=np.float64)
    batch_size = max(1, min(128, 2_000_000 // count))
    for start in range(0, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        indexes = rng.integers(0, count, size=(stop - start, count), endpoint=False)
        world_loss, world_tokens, control_loss, control_tokens = (
            np.sum(values[indexes], axis=1) for values in arrays
        )
        result[start:stop] = world_loss / world_tokens - control_loss / control_tokens
    return result


def _placebo_gaps(
    arrays: tuple[np.ndarray, ...],
    rng: np.random.Generator,
    replicates: int,
) -> np.ndarray:
    if type(replicates) is not int or replicates <= 0:
        raise ValueError("placebo replicate count must be positive")
    world_loss, world_tokens, control_loss, control_tokens = arrays
    count = len(world_loss)
    result = np.empty(replicates, dtype=np.float64)
    batch_size = max(1, min(128, 2_000_000 // count))
    for start in range(0, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        swap = rng.integers(0, 2, size=(stop - start, count), endpoint=False, dtype=np.int8).astype(bool)
        placebo_world_loss = np.sum(np.where(swap, control_loss, world_loss), axis=1)
        placebo_world_tokens = np.sum(np.where(swap, control_tokens, world_tokens), axis=1)
        placebo_control_loss = np.sum(np.where(swap, world_loss, control_loss), axis=1)
        placebo_control_tokens = np.sum(np.where(swap, world_tokens, control_tokens), axis=1)
        result[start:stop] = (
            placebo_world_loss / placebo_world_tokens
            - placebo_control_loss / placebo_control_tokens
        )
    return result


def _rng(identity_sha256: str, namespace: str) -> np.random.Generator:
    if type(identity_sha256) is not str or len(identity_sha256) != 64:
        raise ValueError("empirical-null seed identity must be SHA-256")
    seed = int.from_bytes(
        sha256(f"{BENCHMARK_ID}\0{identity_sha256}\0{namespace}".encode()).digest()[:16],
        "big",
    )
    return np.random.Generator(np.random.PCG64(seed))


__all__ = [
    "CANONICAL_REPLICATES",
    "GAP_THRESHOLD_NATS",
    "CalibrationDecision",
    "EmpiricalGap",
    "GroupLoss",
    "PairedLoss",
    "SemanticEpochValidation",
    "WorldEmpiricalGap",
    "calibration_decision",
    "epoch_satisfies_semantic_gap_gate",
    "holm_rejections",
    "paired_empirical_gap",
    "select_best_eligible_epoch",
    "summarize_empirical_gaps",
]

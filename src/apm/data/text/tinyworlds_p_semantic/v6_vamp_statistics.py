"""Paired adapter-specificity statistics for the semantic-v6 VAMP study."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
import math

import numpy as np

from apm.data.text.tinyworlds_p_semantic.contracts import WORLD_LABELS
from apm.data.text.tinyworlds_p_semantic.statistics import GroupLoss


@dataclass(frozen=True, slots=True)
class AdapterSpecificityPair:
    """Base and adapted losses for one persisted world/control pairing."""

    world: str
    arm: str
    base_world: GroupLoss
    adapted_world: GroupLoss
    base_control: GroupLoss
    adapted_control: GroupLoss

    def __post_init__(self) -> None:
        if self.world not in WORLD_LABELS or self.arm not in ("row", "column"):
            raise ValueError("adapter specificity pair has an invalid world or arm")
        if (
            self.base_world.normalized_story_sha256
            != self.adapted_world.normalized_story_sha256
            or self.base_control.normalized_story_sha256
            != self.adapted_control.normalized_story_sha256
        ):
            raise ValueError("base and adapted specificity ledgers are misaligned")
        if (
            self.base_world.active_tokens != self.adapted_world.active_tokens
            or self.base_control.active_tokens != self.adapted_control.active_tokens
        ):
            raise ValueError("adapter evaluation changed active-token counts")


@dataclass(frozen=True, slots=True)
class AdapterSpecificity:
    """World-minus-control adapter improvement with a paired bootstrap interval."""

    world: str
    method: str
    arm: str
    world_improvement: float
    control_improvement: float
    specificity: float
    bootstrap_lower: float
    bootstrap_upper: float
    pair_count: int
    replicate_count: int

    def __post_init__(self) -> None:
        if self.world not in WORLD_LABELS or self.arm not in ("row", "column"):
            raise ValueError("adapter specificity result has an invalid world or arm")
        if not self.method:
            raise ValueError("adapter specificity method must not be empty")
        values = (
            self.world_improvement,
            self.control_improvement,
            self.specificity,
            self.bootstrap_lower,
            self.bootstrap_upper,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("adapter specificity values must be finite")
        if not math.isclose(
            self.specificity,
            self.world_improvement - self.control_improvement,
            abs_tol=1e-12,
        ):
            raise ValueError("adapter specificity must be world minus control improvement")
        if self.bootstrap_lower > self.bootstrap_upper:
            raise ValueError("adapter specificity interval is reversed")
        if self.pair_count <= 0 or self.replicate_count <= 0:
            raise ValueError("adapter specificity counts must be positive")


def paired_adapter_specificity(
    pairs: Sequence[AdapterSpecificityPair],
    method: str,
    identity_sha256: str,
    *,
    replicates: int,
) -> AdapterSpecificity:
    """Estimate forced-adapter specificity by resampling persisted pairs."""
    if not pairs:
        raise ValueError("adapter specificity requires at least one pair")
    if type(replicates) is not int or replicates <= 0:
        raise ValueError("adapter specificity replicate count must be positive")
    worlds = {pair.world for pair in pairs}
    arms = {pair.arm for pair in pairs}
    if len(worlds) != 1 or len(arms) != 1:
        raise ValueError("adapter specificity pairs must share one world and arm")
    arrays = _pair_arrays(pairs)
    world_improvement, control_improvement, observed = _improvements(arrays, None)
    rng = np.random.default_rng(
        int.from_bytes(
            sha256(
                (
                    f"tinyworlds-p-semantic-v6\0specificity\0{identity_sha256}"
                    f"\0{method}\0{next(iter(worlds))}\0{next(iter(arms))}"
                ).encode("utf-8")
            ).digest()[:16],
            "big",
        )
    )
    bootstrap = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 256):
        size = min(256, replicates - start)
        samples = rng.integers(0, len(pairs), size=(size, len(pairs)))
        bootstrap[start : start + size] = tuple(
            _improvements(arrays, indices)[2] for indices in samples
        )
    return AdapterSpecificity(
        world=next(iter(worlds)),
        method=method,
        arm=next(iter(arms)),
        world_improvement=world_improvement,
        control_improvement=control_improvement,
        specificity=observed,
        bootstrap_lower=float(np.quantile(bootstrap, 0.025, method="linear")),
        bootstrap_upper=float(np.quantile(bootstrap, 0.975, method="linear")),
        pair_count=len(pairs),
        replicate_count=replicates,
    )


def _pair_arrays(pairs: Sequence[AdapterSpecificityPair]) -> np.ndarray:
    ordered = tuple(
        sorted(
            pairs,
            key=lambda pair: (
                pair.base_world.normalized_story_sha256,
                pair.base_control.normalized_story_sha256,
            ),
        )
    )
    return np.asarray(
        tuple(
            (
                pair.base_world.loss_sum,
                pair.adapted_world.loss_sum,
                pair.base_world.active_tokens,
                pair.base_control.loss_sum,
                pair.adapted_control.loss_sum,
                pair.base_control.active_tokens,
            )
            for pair in ordered
        ),
        dtype=np.float64,
    )


def _improvements(
    arrays: np.ndarray,
    indices: np.ndarray | None,
) -> tuple[float, float, float]:
    selected = arrays if indices is None else arrays[indices]
    world_improvement = (
        np.sum(selected[:, 0]) - np.sum(selected[:, 1])
    ) / np.sum(selected[:, 2])
    control_improvement = (
        np.sum(selected[:, 3]) - np.sum(selected[:, 4])
    ) / np.sum(selected[:, 5])
    return (
        float(world_improvement),
        float(control_improvement),
        float(world_improvement - control_improvement),
    )


__all__ = [
    "AdapterSpecificity",
    "AdapterSpecificityPair",
    "paired_adapter_specificity",
]

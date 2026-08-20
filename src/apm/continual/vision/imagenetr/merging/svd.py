"""Source-weighted compact optimal rank truncation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from apm.continual.vision.imagenetr.merging.common import (
    LoRAFactors,
    MergeDiagnostics,
    compact_svd,
    diagnostics,
    exact_weighted_factors,
    factors_from_svd,
)


def weighted_svd_merge(
    children: Sequence[LoRAFactors],
    weights: Sequence[float],
    output_rank: int = 16,
    parent_scale: float = 1.0,
    merge_scale: float = 1.0,
) -> tuple[LoRAFactors, MergeDiagnostics]:
    """Return the optimal compact rank-bounded weighted child sum in FP32."""
    exact = exact_weighted_factors(children, weights, merge_scale)
    left, singular, right = compact_svd(exact)
    parent = factors_from_svd(left, singular, right, output_rank, parent_scale)
    return parent, diagnostics(children, singular, parent.rank)


def merge_module_states(
    child_states: Sequence[Mapping[str, LoRAFactors]],
    weights: Sequence[float],
    output_rank: int = 16,
    parent_scale: float = 1.0,
    merge_scale: float = 1.0,
) -> tuple[dict[str, LoRAFactors], dict[str, MergeDiagnostics]]:
    """Apply compact SVD to every identically keyed adapted matrix."""
    if len(child_states) < 2 or len({frozenset(state) for state in child_states}) != 1:
        raise ValueError("child adapter module sets differ")
    results = {
        module: weighted_svd_merge(
            tuple(state[module] for state in child_states),
            weights,
            output_rank,
            parent_scale,
            merge_scale,
        )
        for module in sorted(child_states[0])
    }
    return (
        {module: value[0] for module, value in results.items()},
        {module: value[1] for module, value in results.items()},
    )


__all__ = ["merge_module_states", "weighted_svd_merge"]

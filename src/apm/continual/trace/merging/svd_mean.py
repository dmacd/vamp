"""Weighted dense-delta mean followed by compact optimal rank truncation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from apm.continual.trace.merging.common import (
    LoRAFactors,
    MergeDiagnostics,
    diagnostics_from_singular_values,
    factors_from_svd,
    require_compatible,
)


def weighted_svd_mean(
    children: Sequence[LoRAFactors],
    weights: Sequence[float],
    output_rank: int = 8,
    parent_scale: float = 4.0,
) -> tuple[LoRAFactors, MergeDiagnostics]:
    """Merge low-rank deltas through compact QR/SVD without dense materialization."""
    require_compatible(children)
    if len(weights) != len(children) or any(weight < 0.0 for weight in weights):
        raise ValueError("merge weights must align with children and be nonnegative")
    if abs(sum(weights) - 1.0) > 1.0e-7:
        raise ValueError("merge weights must sum to one")
    scaled = tuple(
        (
            child.b.to(torch.float32) * (weight * child.scale) ** 0.5,
            child.a.to(torch.float32) * (weight * child.scale) ** 0.5,
        )
        for child, weight in zip(children, weights)
        if weight > 0.0
    )
    if not scaled:
        raise ValueError("at least one merge weight must be positive")
    left_stack = torch.cat(tuple(left for left, _ in scaled), dim=1)
    right_stack = torch.cat(tuple(right for _, right in scaled), dim=0)
    left_basis, left_triangular = torch.linalg.qr(left_stack, mode="reduced")
    right_basis, right_triangular = torch.linalg.qr(right_stack.T, mode="reduced")
    compact = left_triangular @ right_triangular.T
    compact_left, singular, compact_right = torch.linalg.svd(
        compact,
        full_matrices=False,
    )
    left_vectors = left_basis @ compact_left
    right_vectors = compact_right @ right_basis.T
    parent = factors_from_svd(
        left_vectors,
        singular,
        right_vectors,
        output_rank,
        parent_scale,
    )
    diagnostics = diagnostics_from_singular_values(
        input_ranks=tuple(child.rank for child in children),
        singular_values=singular,
        output_rank=parent.rank,
    )
    return parent, diagnostics


def merge_module_states(
    child_states: Sequence[Mapping[str, LoRAFactors]],
    weights: Sequence[float],
    output_rank: int = 8,
    parent_scale: float = 4.0,
) -> tuple[dict[str, LoRAFactors], dict[str, MergeDiagnostics]]:
    """Apply weighted compact SVD to every identically keyed adapter module."""
    if len(child_states) < 2 or len({frozenset(state) for state in child_states}) != 1:
        raise ValueError("child adapter module sets differ")
    merged = {
        module: weighted_svd_mean(
            tuple(state[module] for state in child_states),
            weights,
            output_rank,
            parent_scale,
        )
        for module in sorted(child_states[0])
    }
    return (
        {module: result[0] for module, result in merged.items()},
        {module: result[1] for module, result in merged.items()},
    )


__all__ = ["merge_module_states", "weighted_svd_mean"]

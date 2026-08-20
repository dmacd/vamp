"""Core-Space alignment, pinned TSV merge, and direct rank-bounded factors."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor

from apm.continual.vision.imagenetr.merging.common import (
    LoRAFactors,
    MergeDiagnostics,
    diagnostics,
    factors_from_svd,
)
from apm.continual.vision.imagenetr.merging.core_space import build_core_space
from apm.continual.vision.imagenetr.merging.tsv import tsv_merge


@dataclass(frozen=True, slots=True)
class CoreTsvResult:
    """Deployed parent and cached reproducibility intermediates."""

    factors: LoRAFactors
    diagnostics: MergeDiagnostics
    left_basis: Tensor
    right_basis: Tensor
    aligned_cores: tuple[Tensor, ...]
    merged_core: Tensor
    merged_core_singular_values: Tensor


def core_tsv_merge(
    children: Sequence[LoRAFactors],
    weights: Sequence[float],
    output_rank: int = 16,
    parent_scale: float = 1.0,
    merge_scale: float = 1.0,
) -> CoreTsvResult:
    """Align, source-weight, TSV-merge, and directly compress child updates."""
    if merge_scale <= 0.0:
        raise ValueError("Core+TSV merge scale must be positive")
    core = build_core_space(children, weights)
    merged_core = merge_scale * tsv_merge(core.aligned_cores).to(torch.float32)
    core_left, singular, core_right = torch.linalg.svd(merged_core, full_matrices=False)
    left = core.left_basis @ core_left
    right = core_right @ core.right_basis.T
    factors = factors_from_svd(left, singular, right, output_rank, parent_scale)
    return CoreTsvResult(
        factors=factors,
        diagnostics=diagnostics(children, singular, factors.rank),
        left_basis=core.left_basis,
        right_basis=core.right_basis,
        aligned_cores=core.aligned_cores,
        merged_core=merged_core,
        merged_core_singular_values=singular,
    )


def merge_module_states(
    child_states: Sequence[Mapping[str, LoRAFactors]],
    weights: Sequence[float],
    output_rank: int = 16,
    parent_scale: float = 1.0,
    merge_scale: float = 1.0,
) -> dict[str, CoreTsvResult]:
    """Apply Core+TSV to every identically keyed adapted matrix."""
    if len(child_states) < 2 or len({frozenset(state) for state in child_states}) != 1:
        raise ValueError("child adapter module sets differ")
    return {
        module: core_tsv_merge(
            tuple(state[module] for state in child_states),
            weights,
            output_rank,
            parent_scale,
            merge_scale,
        )
        for module in sorted(child_states[0])
    }


__all__ = ["CoreTsvResult", "core_tsv_merge", "merge_module_states"]

"""Core Space + TSV merge with direct rank-bounded LoRA reconstruction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from apm.continual.trace.merging.common import (
    LoRAFactors,
    MergeDiagnostics,
    diagnostics_from_singular_values,
    factors_from_svd,
)
from apm.continual.trace.merging.core_space import CoreSpace, build_core_space
from apm.continual.trace.merging.tsv import tsv_merge


@dataclass(frozen=True, slots=True)
class CoreTsvResult:
    """Deployed factors plus cached Core Space evidence for one module."""

    factors: LoRAFactors
    diagnostics: MergeDiagnostics
    left_basis: Tensor
    right_basis: Tensor
    merged_core: Tensor
    precompress_factors: LoRAFactors | None


def core_tsv_merge(
    children: Sequence[LoRAFactors],
    core_scale: float,
    output_rank: int = 8,
    parent_scale: float = 4.0,
    retain_precompress: bool = False,
) -> CoreTsvResult:
    """Align, TSV-merge, scale, and directly compact LoRA updates in FP32."""
    if core_scale <= 0.0:
        raise ValueError("Core TSV scale must be positive")
    core_space = build_core_space(children)
    merged_core = core_scale * tsv_merge(core_space.aligned_cores).to(torch.float32)
    core_left, singular, core_right = torch.linalg.svd(
        merged_core,
        full_matrices=False,
    )
    left_vectors = core_space.left_basis @ core_left
    right_vectors = core_right @ core_space.right_basis.T
    factors = factors_from_svd(
        left_vectors,
        singular,
        right_vectors,
        output_rank,
        parent_scale,
    )
    diagnostics = diagnostics_from_singular_values(
        input_ranks=tuple(child.rank for child in children),
        singular_values=singular,
        output_rank=factors.rank,
        core_dimension=core_space.dimension,
    )
    precompress_rank = min(sum(child.rank for child in children), singular.numel())
    precompress = (
        factors_from_svd(
            left_vectors,
            singular,
            right_vectors,
            precompress_rank,
            parent_scale,
        )
        if retain_precompress
        else None
    )
    return CoreTsvResult(
        factors=factors,
        diagnostics=diagnostics,
        left_basis=core_space.left_basis,
        right_basis=core_space.right_basis,
        merged_core=merged_core,
        precompress_factors=precompress,
    )


def merge_module_states(
    child_states: Sequence[Mapping[str, LoRAFactors]],
    core_scale: float,
    output_rank: int = 8,
    parent_scale: float = 4.0,
    diagnostic_modules: frozenset[str] = frozenset(),
) -> dict[str, CoreTsvResult]:
    """Apply Core+TSV to every identically keyed adapter module."""
    if len(child_states) < 2 or len({frozenset(state) for state in child_states}) != 1:
        raise ValueError("child adapter module sets differ")
    return {
        module: core_tsv_merge(
            tuple(state[module] for state in child_states),
            core_scale,
            output_rank,
            parent_scale,
            module in diagnostic_modules,
        )
        for module in sorted(child_states[0])
    }


__all__ = ["CoreTsvResult", "core_tsv_merge", "merge_module_states"]

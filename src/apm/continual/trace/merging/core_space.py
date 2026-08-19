"""Pairwise Core Space construction directly from low-rank LoRA factors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from apm.continual.trace.merging.common import LoRAFactors, require_compatible


@dataclass(frozen=True, slots=True)
class CoreSpace:
    """Shared orthonormal bases and each child's aligned compact update."""

    left_basis: Tensor
    right_basis: Tensor
    aligned_cores: tuple[Tensor, ...]

    def __post_init__(self) -> None:
        if (
            self.left_basis.ndim != 2
            or self.right_basis.ndim != 2
            or not self.aligned_cores
            or any(core.ndim != 2 for core in self.aligned_cores)
        ):
            raise ValueError("invalid Core Space shapes")

    @property
    def dimension(self) -> int:
        """Return the maximum aligned Core Space dimension."""
        return min(self.left_basis.shape[1], self.right_basis.shape[1])


def build_core_space(children: Sequence[LoRAFactors]) -> CoreSpace:
    """Build official-style stacked A/B reference bases and aligned cores in FP32."""
    require_compatible(children)
    a_stack = torch.cat(tuple(child.a.to(torch.float32) for child in children), dim=0)
    b_stack = torch.cat(tuple(child.b.to(torch.float32) for child in children), dim=1)
    _, _, right_rows = torch.linalg.svd(a_stack, full_matrices=False)
    left_basis, _, _ = torch.linalg.svd(b_stack, full_matrices=False)
    right_basis = right_rows.T
    aligned = tuple(
        child.scale
        * ((left_basis.T @ child.b.to(torch.float32)) @ (child.a.to(torch.float32) @ right_basis))
        for child in children
    )
    return CoreSpace(
        left_basis=left_basis,
        right_basis=right_basis,
        aligned_cores=aligned,
    )


__all__ = ["CoreSpace", "build_core_space"]

"""Pinned official-style Core Space alignment for low-rank children."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import torch
from torch import Tensor

from apm.continual.vision.imagenetr.merging.common import LoRAFactors, require_compatible


@dataclass(frozen=True, slots=True)
class CoreSpace:
    """Shared reference bases and aligned compact child matrices."""

    left_basis: Tensor
    right_basis: Tensor
    aligned_cores: tuple[Tensor, ...]

    @property
    def dimensions(self) -> tuple[int, int]:
        """Return the left and right compact basis dimensions."""
        return self.left_basis.shape[1], self.right_basis.shape[1]


def build_core_space(
    children: Sequence[LoRAFactors],
    weights: Sequence[float] | None = None,
) -> CoreSpace:
    """Match the pinned stacked-A/B reference-basis construction in FP32."""
    require_compatible(children)
    coefficients = tuple(1.0 for _ in children) if weights is None else tuple(weights)
    if len(coefficients) != len(children) or any(value < 0.0 for value in coefficients):
        raise ValueError("Core Space weights must align and be nonnegative")
    a_stack = torch.cat(tuple(child.a.to(torch.float32) for child in children), dim=0)
    b_stack = torch.cat(tuple(child.b.to(torch.float32) for child in children), dim=1)
    _, _, right_rows = torch.linalg.svd(a_stack, full_matrices=False)
    left_basis, _, _ = torch.linalg.svd(b_stack, full_matrices=False)
    right_basis = right_rows.T
    cores = tuple(
        coefficient
        * child.scale
        * (
            (left_basis.T @ child.b.to(torch.float32))
            @ (child.a.to(torch.float32) @ right_basis)
        )
        for child, coefficient in zip(children, coefficients)
    )
    return CoreSpace(left_basis, right_basis, cores)


def dense_core_reference(
    child: LoRAFactors,
    left_basis: Tensor,
    right_basis: Tensor,
) -> Tensor:
    """Project a dense update for bounded official-algebra parity tests."""
    return left_basis.T @ child.dense().to(torch.float32) @ right_basis


__all__ = ["CoreSpace", "build_core_space", "dense_core_reference"]

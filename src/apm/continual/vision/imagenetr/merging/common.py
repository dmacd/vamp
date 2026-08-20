"""Scaled low-rank factors, dense references, and compact diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
import math

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class LoRAFactors:
    """One update represented exactly as ``scale * B @ A``."""

    a: Tensor
    b: Tensor
    scale: float = 1.0

    def __post_init__(self) -> None:
        if (
            self.a.ndim != 2
            or self.b.ndim != 2
            or self.a.shape[0] != self.b.shape[1]
            or self.a.device != self.b.device
            or self.scale <= 0.0
        ):
            raise ValueError("invalid scaled LoRA factors")

    @property
    def rank(self) -> int:
        """Return the stored factor rank."""
        return self.a.shape[0]

    @property
    def shape(self) -> tuple[int, int]:
        """Return the represented dense output-by-input shape."""
        return self.b.shape[0], self.a.shape[1]

    def dense(self) -> Tensor:
        """Materialize the update for tests or bounded diagnostics."""
        return self.scale * (self.b @ self.a)


@dataclass(frozen=True, slots=True)
class MergeDiagnostics:
    """Per-matrix compact spectral and child-compatibility evidence."""

    input_ranks: tuple[int, ...]
    exact_rank: int
    output_rank: int
    singular_values: tuple[float, ...]
    retained_parameter_energy: float
    relative_parameter_error: float
    child_update_cosine: float

    def as_record(self) -> dict[str, object]:
        """Return a JSON-compatible diagnostic row fragment."""
        return {
            "child_update_cosine": self.child_update_cosine,
            "exact_rank": self.exact_rank,
            "input_ranks": list(self.input_ranks),
            "output_rank": self.output_rank,
            "relative_parameter_error": self.relative_parameter_error,
            "retained_parameter_energy": self.retained_parameter_energy,
            "singular_values": list(self.singular_values),
        }


def require_compatible(children: Sequence[LoRAFactors]) -> None:
    """Require at least two shape- and device-compatible child updates."""
    if len(children) < 2:
        raise ValueError("a merge requires at least two children")
    if len({child.shape for child in children}) != 1:
        raise ValueError("child updates have different dense shapes")
    devices = {child.a.device for child in children} | {child.b.device for child in children}
    if len(devices) != 1:
        raise ValueError("child updates must share one device")


def normalized_weights(example_counts: Sequence[int]) -> tuple[float, ...]:
    """Normalize positive represented-image counts without integer truncation."""
    if not example_counts or any(count < 1 for count in example_counts):
        raise ValueError("represented image counts must be positive")
    total = math.fsum(example_counts)
    return tuple(count / total for count in example_counts)


def exact_weighted_factors(
    children: Sequence[LoRAFactors],
    weights: Sequence[float],
    merge_scale: float = 1.0,
) -> LoRAFactors:
    """Concatenate factors for the exact arbitrary coefficient-weighted sum."""
    require_compatible(children)
    if (
        len(weights) != len(children)
        or any(weight < 0.0 for weight in weights)
        or not math.isclose(math.fsum(weights), 1.0, abs_tol=1.0e-8)
        or merge_scale <= 0.0
    ):
        raise ValueError("invalid exact-sum weights or scale")
    active = tuple(
        (child, merge_scale * weight * child.scale)
        for child, weight in zip(children, weights)
        if weight > 0.0
    )
    if not active:
        raise ValueError("at least one exact-sum coefficient must be positive")
    return LoRAFactors(
        a=torch.cat(
            tuple(child.a.to(torch.float32) * coefficient**0.5 for child, coefficient in active),
            dim=0,
        ),
        b=torch.cat(
            tuple(child.b.to(torch.float32) * coefficient**0.5 for child, coefficient in active),
            dim=1,
        ),
        scale=1.0,
    )


def compact_svd(factors: LoRAFactors) -> tuple[Tensor, Tensor, Tensor]:
    """SVD a low-rank update through reduced QR without dense materialization."""
    left_stack = factors.b.to(torch.float32) * factors.scale**0.5
    right_stack = factors.a.to(torch.float32) * factors.scale**0.5
    left_basis, left_triangular = torch.linalg.qr(left_stack, mode="reduced")
    right_basis, right_triangular = torch.linalg.qr(right_stack.T, mode="reduced")
    core = left_triangular @ right_triangular.T
    core_left, singular, core_right = torch.linalg.svd(core, full_matrices=False)
    return left_basis @ core_left, singular, core_right @ right_basis.T


def factors_from_svd(
    left: Tensor,
    singular: Tensor,
    right: Tensor,
    output_rank: int,
    parent_scale: float,
) -> LoRAFactors:
    """Convert a truncated SVD to symmetric factors with exact LoRA scaling."""
    rank = min(output_rank, singular.numel())
    if rank < 1 or parent_scale <= 0.0:
        raise ValueError("output rank and parent scale must be positive")
    roots = torch.sqrt(torch.clamp(singular[:rank], min=0.0) / parent_scale)
    return LoRAFactors(
        a=roots[:, None] * right[:rank],
        b=left[:, :rank] * roots[None, :],
        scale=parent_scale,
    )


def frobenius_inner(left: LoRAFactors, right: LoRAFactors) -> Tensor:
    """Compute a dense-update Frobenius inner product using rank-sized products."""
    if left.shape != right.shape:
        raise ValueError("Frobenius products require equal dense shapes")
    b_product = left.b.to(torch.float64).T @ right.b.to(torch.float64)
    a_product = left.a.to(torch.float64) @ right.a.to(torch.float64).T
    return left.scale * right.scale * torch.sum(b_product * a_product.T)


def update_cosine(left: LoRAFactors, right: LoRAFactors) -> float:
    """Return the numerically safe cosine of two represented dense updates."""
    numerator = frobenius_inner(left, right)
    denominator = torch.sqrt(
        torch.clamp(frobenius_inner(left, left), min=0.0)
        * torch.clamp(frobenius_inner(right, right), min=0.0)
    )
    return 0.0 if denominator == 0 else float((numerator / denominator).item())


def diagnostics(
    children: Sequence[LoRAFactors],
    singular: Tensor,
    output_rank: int,
) -> MergeDiagnostics:
    """Build retained-energy, error, rank, and pairwise-cosine diagnostics."""
    values = singular.detach().to(device="cpu", dtype=torch.float64)
    total = float(torch.sum(values.square()).item())
    kept = float(torch.sum(values[:output_rank].square()).item())
    retained = 1.0 if total == 0.0 else kept / total
    tolerance = (
        max(values.shape) * torch.finfo(values.dtype).eps * float(values[0].item())
        if values.numel() and values[0] > 0
        else 0.0
    )
    pair_cosines = tuple(
        update_cosine(children[left], children[right])
        for left in range(len(children))
        for right in range(left + 1, len(children))
    )
    return MergeDiagnostics(
        input_ranks=tuple(child.rank for child in children),
        exact_rank=int(torch.sum(values > tolerance).item()),
        output_rank=min(output_rank, values.numel()),
        singular_values=tuple(float(value) for value in values.tolist()),
        retained_parameter_energy=retained,
        relative_parameter_error=math.sqrt(max(0.0, 1.0 - retained)),
        child_update_cosine=math.fsum(pair_cosines) / len(pair_cosines),
    )


def dense_truncated_reference(
    children: Sequence[LoRAFactors],
    weights: Sequence[float],
    output_rank: int,
    merge_scale: float = 1.0,
) -> Tensor:
    """Return the dense optimal truncation used only for parity tests."""
    dense = merge_scale * sum(
        (weight * child.dense().to(torch.float32) for child, weight in zip(children, weights)),
        start=torch.zeros(children[0].shape, device=children[0].a.device),
    )
    left, singular, right = torch.linalg.svd(dense, full_matrices=False)
    rank = min(output_rank, singular.numel())
    return (left[:, :rank] * singular[:rank]) @ right[:rank]


def merge_module_states(
    child_states: Sequence[Mapping[str, LoRAFactors]],
    merge: object,
) -> dict[str, object]:
    """Apply one callable merge uniformly to identically keyed module states."""
    if len(child_states) < 2 or len({frozenset(state) for state in child_states}) != 1:
        raise ValueError("child adapter module sets differ")
    if not callable(merge):
        raise TypeError("module-state merge must be callable")
    return {
        module: merge(tuple(state[module] for state in child_states))
        for module in sorted(child_states[0])
    }


__all__ = [
    "LoRAFactors",
    "MergeDiagnostics",
    "compact_svd",
    "dense_truncated_reference",
    "diagnostics",
    "exact_weighted_factors",
    "factors_from_svd",
    "frobenius_inner",
    "merge_module_states",
    "normalized_weights",
    "require_compatible",
    "update_cosine",
]

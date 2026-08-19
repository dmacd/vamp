"""Shared LoRA factor, PEFT state, and compact-SVD operations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class LoRAFactors:
    """One base-relative update represented as ``scale * B @ A``."""

    a: Tensor
    b: Tensor
    scale: float

    def __post_init__(self) -> None:
        if (
            self.a.ndim != 2
            or self.b.ndim != 2
            or self.a.shape[0] != self.b.shape[1]
            or self.scale <= 0.0
        ):
            raise ValueError("invalid LoRA factor shapes or scale")

    @property
    def rank(self) -> int:
        """Return the stored factor rank."""
        return self.a.shape[0]

    @property
    def shape(self) -> tuple[int, int]:
        """Return the represented dense matrix shape."""
        return self.b.shape[0], self.a.shape[1]

    def delta(self) -> Tensor:
        """Materialize the represented dense update for diagnostics or tests."""
        return self.scale * (self.b @ self.a)


@dataclass(frozen=True, slots=True)
class MergeDiagnostics:
    """Numerical rank and retained-energy evidence for one module merge."""

    input_ranks: tuple[int, ...]
    exact_rank: int
    output_rank: int
    singular_values: tuple[float, ...]
    retained_energy: float
    core_dimension: int | None = None

    def as_record(self) -> dict[str, object]:
        """Return a JSON-compatible merge diagnostic."""
        return {
            "core_dimension": self.core_dimension,
            "exact_rank": self.exact_rank,
            "input_ranks": list(self.input_ranks),
            "output_rank": self.output_rank,
            "retained_energy": self.retained_energy,
            "singular_values": list(self.singular_values),
        }


def require_compatible(factors: Sequence[LoRAFactors]) -> None:
    """Require at least two compatible factors on one device."""
    if len(factors) < 2:
        raise ValueError("a merge requires at least two LoRA updates")
    if len({factor.shape for factor in factors}) != 1:
        raise ValueError("LoRA updates have incompatible dense shapes")
    if len({factor.a.device for factor in factors} | {factor.b.device for factor in factors}) != 1:
        raise ValueError("LoRA factors must share one device")


def factors_from_svd(
    left_vectors: Tensor,
    singular_values: Tensor,
    right_vectors: Tensor,
    output_rank: int,
    parent_scale: float,
) -> LoRAFactors:
    """Convert an SVD into symmetric factors with exact parent-scale accounting."""
    rank = min(output_rank, singular_values.numel())
    if rank <= 0 or parent_scale <= 0.0:
        raise ValueError("output rank and parent scale must be positive")
    root = torch.sqrt(torch.clamp(singular_values[:rank], min=0.0) / parent_scale)
    return LoRAFactors(
        a=root[:, None] * right_vectors[:rank, :],
        b=left_vectors[:, :rank] * root[None, :],
        scale=parent_scale,
    )


def dense_svd_factors(
    delta: Tensor,
    output_rank: int,
    parent_scale: float,
) -> tuple[LoRAFactors, MergeDiagnostics]:
    """Return the optimal truncated factors for a diagnostic dense update."""
    left, singular, right = torch.linalg.svd(delta.to(torch.float32), full_matrices=False)
    factors = factors_from_svd(left, singular, right, output_rank, parent_scale)
    return factors, diagnostics_from_singular_values(
        input_ranks=(int(torch.linalg.matrix_rank(delta.to(torch.float32)).item()),),
        singular_values=singular,
        output_rank=factors.rank,
    )


def diagnostics_from_singular_values(
    input_ranks: Sequence[int],
    singular_values: Tensor,
    output_rank: int,
    core_dimension: int | None = None,
) -> MergeDiagnostics:
    """Build stable retained-energy diagnostics from singular values."""
    values = singular_values.detach().to(device="cpu", dtype=torch.float64)
    total_energy = float(torch.sum(values.square()).item())
    retained = float(torch.sum(values[:output_rank].square()).item())
    tolerance = (
        max(values.shape) * torch.finfo(values.dtype).eps * float(values[0].item())
        if values.numel() and values[0] > 0
        else 0.0
    )
    exact_rank = int(torch.sum(values > tolerance).item())
    return MergeDiagnostics(
        input_ranks=tuple(input_ranks),
        exact_rank=exact_rank,
        output_rank=output_rank,
        singular_values=tuple(float(value) for value in values.tolist()),
        retained_energy=1.0 if total_energy == 0.0 else retained / total_energy,
        core_dimension=core_dimension,
    )


def module_name_from_peft_key(key: str, factor_name: str) -> str:
    """Return the module prefix for a PEFT LoRA A or B tensor key."""
    endings = (f".lora_{factor_name}.weight", f".lora_{factor_name}.default.weight")
    matches = tuple(ending for ending in endings if key.endswith(ending))
    if len(matches) != 1:
        raise ValueError(f"not a PEFT LoRA {factor_name} tensor key: {key}")
    return key[: -len(matches[0])]


def factors_from_peft_state(
    state: Mapping[str, Tensor],
    scale: float,
) -> dict[str, LoRAFactors]:
    """Parse canonical or default-adapter PEFT tensor keys into module factors."""
    a_by_module = {
        module_name_from_peft_key(key, "A"): value
        for key, value in state.items()
        if ".lora_A." in key
    }
    b_by_module = {
        module_name_from_peft_key(key, "B"): value
        for key, value in state.items()
        if ".lora_B." in key
    }
    if not a_by_module or set(a_by_module) != set(b_by_module):
        raise ValueError("PEFT state has incomplete or mismatched LoRA factors")
    return {
        module: LoRAFactors(a_by_module[module], b_by_module[module], scale)
        for module in sorted(a_by_module)
    }


def peft_state_from_factors(
    factors_by_module: Mapping[str, LoRAFactors],
) -> dict[str, Tensor]:
    """Return adapter-name-free PEFT-compatible state keys."""
    if not factors_by_module:
        raise ValueError("cannot encode an empty LoRA state")
    return {
        key: value
        for module, factors in sorted(factors_by_module.items())
        for key, value in (
            (f"{module}.lora_A.weight", factors.a),
            (f"{module}.lora_B.weight", factors.b),
        )
    }


def weighted_child_weights(example_counts: Sequence[int]) -> tuple[float, ...]:
    """Normalize positive represented-example counts into merge weights."""
    if not example_counts or any(count <= 0 for count in example_counts):
        raise ValueError("merge example counts must be positive")
    total = math.fsum(example_counts)
    return tuple(count / total for count in example_counts)


__all__ = [
    "LoRAFactors",
    "MergeDiagnostics",
    "dense_svd_factors",
    "diagnostics_from_singular_values",
    "factors_from_peft_state",
    "factors_from_svd",
    "peft_state_from_factors",
    "require_compatible",
    "weighted_child_weights",
]

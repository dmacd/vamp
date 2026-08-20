"""Compact function-space output-drift projection of weighted LoRA sums."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
import math

import torch
from torch import Tensor

from apm.continual.vision.imagenetr.merging.common import (
    LoRAFactors,
    MergeDiagnostics,
    compact_svd,
    diagnostics,
    exact_weighted_factors,
    factors_from_svd,
)


@dataclass(frozen=True, slots=True)
class OutputDriftResult:
    """Projected factors and both function- and parameter-space diagnostics."""

    factors: LoRAFactors
    diagnostics: MergeDiagnostics
    output_singular_values: tuple[float, ...]
    retained_output_energy: float


def output_drift_merge(
    children: Sequence[LoRAFactors],
    weights: Sequence[float],
    activations: Tensor,
    output_rank: int = 16,
    parent_scale: float = 1.0,
    merge_scale: float = 1.0,
) -> OutputDriftResult:
    """Project a weighted exact sum onto leading proxy-output directions compactly."""
    exact = exact_weighted_factors(children, weights, merge_scale)
    inputs = activations.reshape(-1, activations.shape[-1]).to(torch.float32)
    if inputs.ndim != 2 or inputs.shape[1] != exact.shape[1] or inputs.shape[0] < 1:
        raise ValueError("proxy activations do not match the adapted matrix input")

    # Y = X @ Delta.T = (X @ A.T) @ B.T.  Two reduced QRs make the SVD
    # rank-sized even for the 2,304-output QKV projection.
    right_response = inputs @ exact.a.to(torch.float32).T
    left_factor = exact.scale * exact.b.to(torch.float32)
    response_basis, response_triangular = torch.linalg.qr(right_response, mode="reduced")
    output_basis, output_triangular = torch.linalg.qr(left_factor, mode="reduced")
    response_core = response_triangular @ output_triangular.T
    _, output_singular, output_right = torch.linalg.svd(response_core, full_matrices=False)
    output_directions = output_basis @ output_right.T
    rank = min(output_rank, output_directions.shape[1])
    retained_directions = output_directions[:, :rank]

    projected_right = (
        retained_directions.T @ left_factor @ exact.a.to(torch.float32)
    )
    compact_left, projected_singular, compact_right = torch.linalg.svd(
        projected_right, full_matrices=False
    )
    projected_left = retained_directions @ compact_left
    factors = factors_from_svd(
        projected_left,
        projected_singular,
        compact_right,
        output_rank,
        parent_scale,
    )

    _, raw_singular, _ = compact_svd(exact)
    parameter_diagnostics = diagnostics(children, raw_singular, output_rank)
    projected_energy = float(torch.sum(projected_singular.square()).item())
    raw_energy = float(torch.sum(raw_singular.square()).item())
    retained_parameter = 1.0 if raw_energy == 0.0 else projected_energy / raw_energy
    parameter_diagnostics = MergeDiagnostics(
        input_ranks=parameter_diagnostics.input_ranks,
        exact_rank=parameter_diagnostics.exact_rank,
        output_rank=factors.rank,
        singular_values=parameter_diagnostics.singular_values,
        retained_parameter_energy=retained_parameter,
        relative_parameter_error=math.sqrt(max(0.0, 1.0 - retained_parameter)),
        child_update_cosine=parameter_diagnostics.child_update_cosine,
    )
    output_total = float(torch.sum(output_singular.square()).item())
    output_kept = float(torch.sum(output_singular[:rank].square()).item())
    return OutputDriftResult(
        factors=factors,
        diagnostics=parameter_diagnostics,
        output_singular_values=tuple(
            float(value) for value in output_singular.detach().cpu().tolist()
        ),
        retained_output_energy=1.0 if output_total == 0.0 else output_kept / output_total,
    )


def merge_module_states(
    child_states: Sequence[Mapping[str, LoRAFactors]],
    weights: Sequence[float],
    activations: Mapping[str, Tensor],
    output_rank: int = 16,
    parent_scale: float = 1.0,
    merge_scale: float = 1.0,
) -> dict[str, OutputDriftResult]:
    """Apply output-drift projection to every adapted matrix and cached input."""
    if (
        len(child_states) < 2
        or len({frozenset(state) for state in child_states}) != 1
        or frozenset(activations) != frozenset(child_states[0])
    ):
        raise ValueError("child modules and proxy activation modules differ")
    return {
        module: output_drift_merge(
            tuple(state[module] for state in child_states),
            weights,
            activations[module],
            output_rank,
            parent_scale,
            merge_scale,
        )
        for module in sorted(child_states[0])
    }


__all__ = ["OutputDriftResult", "merge_module_states", "output_drift_merge"]

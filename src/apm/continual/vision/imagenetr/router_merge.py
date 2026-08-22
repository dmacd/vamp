"""Compact parameter merging and functional-fidelity diagnostics for routers."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
import math

import torch
from torch import Tensor, nn

from apm.continual.vision.imagenetr.merging.common import (
    LoRAFactors,
    compact_svd,
    exact_weighted_factors,
    factors_from_svd,
    normalized_weights,
)
from apm.continual.vision.imagenetr.router_scores import (
    R0Scorer,
    R1Scorer,
    R2Scorer,
    R3Scorer,
    RouterQuery,
    ScoringNode,
    make_scorer,
    score_nodes,
)


@dataclass(frozen=True, slots=True)
class RouterMergeDiagnostics:
    """Compact merge work and functional error statistics."""

    exact_rank: int
    output_rank: int
    retained_parameter_energy: float
    mean_mass_error: float | None = None
    p95_mass_error: float | None = None
    collapsed_kl: float | None = None
    lse_mse: float | None = None

    def as_record(self) -> dict[str, object]:
        return {
            "collapsed_kl": self.collapsed_kl,
            "exact_rank": self.exact_rank,
            "lse_mse": self.lse_mse,
            "mean_mass_error": self.mean_mass_error,
            "output_rank": self.output_rank,
            "p95_mass_error": self.p95_mass_error,
            "retained_parameter_energy": self.retained_parameter_energy,
        }


def _merge_low_rank(
    left_a: Tensor,
    left_b: Tensor,
    right_a: Tensor,
    right_b: Tensor,
    weights: tuple[float, float],
    output_rank: int,
) -> tuple[LoRAFactors, int, float]:
    exact = exact_weighted_factors(
        (
            LoRAFactors(left_a.detach(), left_b.detach(), 1.0),
            LoRAFactors(right_a.detach(), right_b.detach(), 1.0),
        ),
        weights,
    )
    left, singular, right = compact_svd(exact)
    total = float(torch.sum(singular.square()).item())
    kept = float(torch.sum(singular[:output_rank].square()).item())
    tolerance = (
        max(exact.shape) * torch.finfo(singular.dtype).eps * float(singular[0])
        if singular.numel() and singular[0] > 0
        else 0.0
    )
    rank = int(torch.sum(singular > tolerance).item())
    retained = 1.0 if total == 0.0 else kept / total
    return factors_from_svd(left, singular, right, output_rank, 1.0), rank, retained


def _weighted_parameter(
    left: Tensor, right: Tensor, weights: tuple[float, float]
) -> Tensor:
    return weights[0] * left.detach().to(torch.float32) + weights[1] * right.detach().to(
        torch.float32
    )


def svd_merge_scorers(
    left: nn.Module,
    right: nn.Module,
    source_fit_counts: Sequence[int],
    output_rank: int,
    seed: int,
    mlp_hidden: int = 64,
) -> tuple[nn.Module, RouterMergeDiagnostics]:
    """Source-mass merge two same-family scorers with no examples or optimizer."""
    if type(left) is not type(right) or len(source_fit_counts) != 2:
        raise ValueError("router SVD merge requires two same-family scorers")
    weights = normalized_weights(source_fit_counts)
    pair = (weights[0], weights[1])
    if isinstance(left, R0Scorer):
        parent = make_scorer("r0", 1, seed)
        assert isinstance(parent, R0Scorer)
        with torch.no_grad():
            parent.query_weight.copy_(_weighted_parameter(left.query_weight, right.query_weight, pair))
            parent.bias.copy_(_weighted_parameter(left.bias, right.bias, pair))
        return parent, RouterMergeDiagnostics(1, 1, 1.0)
    if isinstance(left, (R1Scorer, R3Scorer)):
        architecture = "r3" if isinstance(left, R3Scorer) else "r1"
        parent = make_scorer(architecture, output_rank, seed)
        if not isinstance(parent, R1Scorer):  # pragma: no cover - factory invariant
            raise AssertionError("R1/R3 factory returned the wrong scorer")
        factors, exact_rank, retained = _merge_low_rank(
            left.interaction_right,
            left.interaction_left,
            right.interaction_right,
            right.interaction_left,
            pair,
            output_rank,
        )
        with torch.no_grad():
            parent.interaction_right.copy_(factors.a)
            parent.interaction_left.copy_(factors.b)
            parent.query_weight.copy_(_weighted_parameter(left.query_weight, right.query_weight, pair))
            parent.descriptor_weight.copy_(
                _weighted_parameter(left.descriptor_weight, right.descriptor_weight, pair)
            )
            parent.bias.copy_(_weighted_parameter(left.bias, right.bias, pair))
            if isinstance(parent, R3Scorer) and isinstance(left, R3Scorer) and isinstance(right, R3Scorer):
                parent.response_weight.copy_(
                    _weighted_parameter(left.response_weight, right.response_weight, pair)
                )
        return parent, RouterMergeDiagnostics(exact_rank, output_rank, retained)
    if isinstance(left, R2Scorer):
        parent = make_scorer("r2", output_rank, seed, mlp_hidden)
        assert isinstance(parent, R2Scorer)
        factors, exact_rank, retained = _merge_low_rank(
            left.first_right,
            left.first_left,
            right.first_right,
            right.first_left,
            pair,
            output_rank,
        )
        with torch.no_grad():
            parent.first_right.copy_(factors.a)
            parent.first_left.copy_(factors.b)
            parent.first_bias.copy_(_weighted_parameter(left.first_bias, right.first_bias, pair))
            parent.output_weight.copy_(
                _weighted_parameter(left.output_weight, right.output_weight, pair)
            )
            parent.output_bias.copy_(
                _weighted_parameter(left.output_bias, right.output_bias, pair)
            )
        return parent, RouterMergeDiagnostics(exact_rank, output_rank, retained)
    raise TypeError("unsupported router scorer family for compact merge")


def functional_merge_diagnostics(
    query: RouterQuery,
    before_nodes: Sequence[ScoringNode],
    left_index: int,
    right_index: int,
    parent: ScoringNode,
) -> RouterMergeDiagnostics:
    """Measure collapsed-frontier mass, KL, and LSE fidelity after one merge."""
    nodes = tuple(before_nodes)
    if not 0 <= left_index < right_index < len(nodes):
        raise ValueError("invalid child positions for functional merge diagnostics")
    before_scores = score_nodes(query, nodes).to(torch.float64)
    child_lse = torch.logaddexp(before_scores[:, left_index], before_scores[:, right_index])
    other_indices = tuple(
        index for index in range(len(nodes)) if index not in {left_index, right_index}
    )
    after_nodes = tuple(nodes[index] for index in other_indices) + (parent,)
    after_scores = score_nodes(query, after_nodes).to(torch.float64)
    parent_score = after_scores[:, -1]
    before_probability = torch.softmax(before_scores, dim=-1)
    collapsed_before = torch.cat(
        (
            before_probability[:, other_indices],
            (
                before_probability[:, left_index] + before_probability[:, right_index]
            ).reshape(-1, 1),
        ),
        dim=-1,
    )
    after_probability = torch.softmax(after_scores, dim=-1)
    errors = torch.abs(after_probability[:, -1] - collapsed_before[:, -1])
    epsilon = torch.finfo(torch.float64).tiny
    kl = torch.sum(
        collapsed_before
        * (
            torch.log(torch.clamp(collapsed_before, min=epsilon))
            - torch.log(torch.clamp(after_probability, min=epsilon))
        ),
        dim=-1,
    )
    return RouterMergeDiagnostics(
        exact_rank=parent.scorer.rank,
        output_rank=parent.scorer.rank,
        retained_parameter_energy=1.0,
        mean_mass_error=float(torch.mean(errors).item()),
        p95_mass_error=float(torch.quantile(errors, 0.95).item()),
        collapsed_kl=float(torch.mean(kl).item()),
        lse_mse=float(torch.mean((parent_score - child_lse).square()).item()),
    )


__all__ = [
    "RouterMergeDiagnostics",
    "functional_merge_diagnostics",
    "svd_merge_scorers",
]

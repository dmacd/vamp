"""Task-free R0-R3 score networks and exact functional router union."""

from __future__ import annotations

from dataclasses import dataclass, fields
from collections.abc import Mapping, Sequence
from typing import Protocol

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from apm.continual.artifacts import record_sha256
from apm.continual.vision.imagenetr.router_descriptor import (
    NodeRouterFeatures,
    response_features,
)


@dataclass(frozen=True, slots=True)
class RouterQuery:
    """Task-free router input with no label, class, task, or oracle surface."""

    image_ids: tuple[str, ...]
    prelogits: Tensor
    cls_activations: dict[str, Tensor]

    def __post_init__(self) -> None:
        rows = len(self.image_ids)
        if (
            rows < 1
            or len(set(self.image_ids)) != rows
            or tuple(self.prelogits.shape) != (rows, 768)
            or any(tuple(value.shape) != (rows, 768) for value in self.cls_activations.values())
        ):
            raise ValueError("invalid task-free router query")

    def select(self, indices: Tensor) -> RouterQuery:
        """Project the query to a deterministic row subset."""
        values = indices.to(dtype=torch.long, device="cpu")
        return RouterQuery(
            tuple(self.image_ids[index] for index in values.tolist()),
            self.prelogits[values],
            {name: tensor[values] for name, tensor in self.cls_activations.items()},
        )

    def to(self, device: torch.device) -> RouterQuery:
        """Move tensor inputs without altering task-free query identity."""
        return RouterQuery(
            self.image_ids,
            self.prelogits.to(device),
            {name: tensor.to(device) for name, tensor in self.cls_activations.items()},
        )


@dataclass(frozen=True, slots=True)
class RouteResult:
    """Selected node identities and auditable task-free score matrix."""

    image_ids: tuple[str, ...]
    node_ids: tuple[str, ...]
    selected_node_ids: tuple[str, ...]
    scores: Tensor

    def __post_init__(self) -> None:
        if (
            len(self.image_ids) != len(self.selected_node_ids)
            or tuple(self.scores.shape) != (len(self.image_ids), len(self.node_ids))
            or any(value not in set(self.node_ids) for value in self.selected_node_ids)
            or not torch.isfinite(self.scores).all()
        ):
            raise ValueError("invalid task-free route result")


class RouterScorer(Protocol):
    """Common score interface implemented by trainable and exact scorers."""

    architecture: str
    rank: int

    def score(
        self,
        query: RouterQuery,
        node_features: NodeRouterFeatures,
    ) -> Tensor: ...


def _normalized(values: Tensor) -> Tensor:
    return F.layer_norm(values.to(torch.float32), (values.shape[-1],))


class R0Scorer(nn.Module):
    """Node-local linear capacity floor."""

    architecture = "r0"

    def __init__(self, query_dim: int = 768, seed: int = 0) -> None:
        super().__init__()
        self.rank = 1
        self.query_weight = nn.Parameter(torch.empty(query_dim))
        self.bias = nn.Parameter(torch.zeros(()))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            nn.init.normal_(self.query_weight, std=0.02)

    def score(self, query: RouterQuery, node_features: NodeRouterFeatures) -> Tensor:
        del node_features
        return _normalized(query.prelogits) @ self.query_weight + self.bias


class R1Scorer(nn.Module):
    """Low-rank bilinear query/descriptor compatibility scorer."""

    architecture = "r1"

    def __init__(
        self,
        rank: int = 8,
        query_dim: int = 768,
        descriptor_dim: int = 128,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("router rank must be positive")
        self.rank = rank
        self.interaction_left = nn.Parameter(torch.empty(query_dim, rank))
        self.interaction_right = nn.Parameter(torch.empty(rank, descriptor_dim))
        self.query_weight = nn.Parameter(torch.zeros(query_dim))
        self.descriptor_weight = nn.Parameter(torch.zeros(descriptor_dim))
        self.bias = nn.Parameter(torch.zeros(()))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            nn.init.xavier_uniform_(self.interaction_left)
            nn.init.xavier_uniform_(self.interaction_right)

    def base_score(self, query: RouterQuery, node_features: NodeRouterFeatures) -> Tensor:
        query_values = _normalized(query.prelogits)
        descriptor = _normalized(node_features.descriptor.reshape(1, -1))[0].to(query_values)
        query_interaction = query_values @ self.interaction_left
        descriptor_interaction = self.interaction_right @ descriptor
        return (
            torch.sum(query_interaction * descriptor_interaction[None, :], dim=-1)
            + query_values @ self.query_weight
            + descriptor @ self.descriptor_weight
            + self.bias
        )

    def score(self, query: RouterQuery, node_features: NodeRouterFeatures) -> Tensor:
        return self.base_score(query, node_features)


class R3Scorer(R1Scorer):
    """R1 plus gauge-invariant adapter-response compatibility."""

    architecture = "r3"

    def __init__(
        self,
        rank: int = 8,
        query_dim: int = 768,
        descriptor_dim: int = 128,
        response_dim: int = 8,
        seed: int = 0,
    ) -> None:
        super().__init__(rank, query_dim, descriptor_dim, seed)
        self.response_weight = nn.Parameter(torch.zeros(response_dim))

    def score(self, query: RouterQuery, node_features: NodeRouterFeatures) -> Tensor:
        if not query.cls_activations or not node_features.response_kernels:
            raise ValueError("R3 requires frozen CLS activations and node response kernels")
        responses = response_features(query.cls_activations, node_features.response_kernels)
        if responses.shape[-1] != self.response_weight.numel():
            raise ValueError("R3 response dimension differs from its scorer")
        return self.base_score(query, node_features) + _normalized(responses).to(
            self.response_weight
        ) @ self.response_weight


class R2Scorer(nn.Module):
    """Modest low-rank nonlinear capacity diagnostic."""

    architecture = "r2"

    def __init__(
        self,
        rank: int = 8,
        hidden: int = 64,
        query_dim: int = 768,
        descriptor_dim: int = 128,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if rank < 1 or hidden < 1:
            raise ValueError("R2 rank and hidden width must be positive")
        self.rank = rank
        width = query_dim + descriptor_dim
        self.first_right = nn.Parameter(torch.empty(rank, width))
        self.first_left = nn.Parameter(torch.empty(hidden, rank))
        self.first_bias = nn.Parameter(torch.zeros(hidden))
        self.output_weight = nn.Parameter(torch.empty(hidden))
        self.output_bias = nn.Parameter(torch.zeros(()))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            nn.init.xavier_uniform_(self.first_right)
            nn.init.xavier_uniform_(self.first_left)
            nn.init.normal_(self.output_weight, std=0.02)

    def score(self, query: RouterQuery, node_features: NodeRouterFeatures) -> Tensor:
        query_values = _normalized(query.prelogits)
        descriptor = _normalized(node_features.descriptor.reshape(1, -1)).to(query_values)
        repeated = descriptor.expand(query_values.shape[0], -1)
        inputs = torch.cat((query_values, repeated), dim=-1)
        hidden = F.gelu((inputs @ self.first_right.T) @ self.first_left.T + self.first_bias)
        return hidden @ self.output_weight + self.output_bias


@dataclass(slots=True)
class ScoringNode:
    """One live score function paired with fixed inference-node features."""

    node_id: str
    scorer: RouterScorer
    features: NodeRouterFeatures
    represented_task_ids: tuple[int, ...]
    represented_class_ids: tuple[int, ...]
    source_fit_count: int
    router_hash: str | None = None

    def __post_init__(self) -> None:
        if (
            not self.node_id
            or self.represented_task_ids != tuple(sorted(set(self.represented_task_ids)))
            or self.represented_class_ids != tuple(sorted(set(self.represented_class_ids)))
            or len(self.represented_class_ids) != 4 * len(self.represented_task_ids)
            or self.source_fit_count < 1
        ):
            raise ValueError("invalid live scoring node")


class ExactLSEScorer:
    """Nondeployable exact semantic union of two child score functions."""

    architecture = "exact"

    def __init__(self, left: ScoringNode, right: ScoringNode) -> None:
        self.left = left
        self.right = right
        self.rank = max(left.scorer.rank, right.scorer.rank)

    def score(self, query: RouterQuery, node_features: NodeRouterFeatures) -> Tensor:
        del node_features
        return torch.logaddexp(
            self.left.scorer.score(query, self.left.features),
            self.right.scorer.score(query, self.right.features),
        )


def make_scorer(
    architecture: str,
    rank: int,
    seed: int,
    mlp_hidden: int = 64,
) -> nn.Module:
    """Construct one deterministic trainable scorer."""
    if architecture == "r0":
        return R0Scorer(seed=seed)
    if architecture == "r1":
        return R1Scorer(rank=rank, seed=seed)
    if architecture == "r2":
        return R2Scorer(rank=rank, hidden=mlp_hidden, seed=seed)
    if architecture == "r3":
        return R3Scorer(rank=rank, seed=seed)
    raise ValueError(f"unknown router architecture: {architecture}")


def load_scorer(
    architecture: str,
    rank: int,
    seed: int,
    state: Mapping[str, Tensor],
    mlp_hidden: int = 64,
) -> nn.Module:
    """Reconstruct a scorer from its exact-key immutable tensor state."""
    scorer = make_scorer(architecture, rank, seed, mlp_hidden)
    incompatible = scorer.load_state_dict(dict(state), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("persisted router scorer keys changed")
    return scorer


def move_scorer(scorer: RouterScorer, device: torch.device) -> None:
    """Move every trainable descendant of a fixed-size or exact-LSE scorer."""
    if isinstance(scorer, nn.Module):
        scorer.to(device)
        return
    if isinstance(scorer, ExactLSEScorer):
        move_scorer(scorer.left.scorer, device)
        move_scorer(scorer.right.scorer, device)
        return
    raise TypeError("unsupported router scorer device traversal")


def score_nodes(query: RouterQuery, nodes: Sequence[ScoringNode]) -> Tensor:
    """Score every live node without any task-aware input."""
    if not nodes or len({node.node_id for node in nodes}) != len(nodes):
        raise ValueError("routing requires a nonempty unique live frontier")
    values = torch.stack(
        tuple(node.scorer.score(query, node.features) for node in nodes), dim=-1
    )
    if not torch.isfinite(values).all():
        raise ValueError("router scores are non-finite")
    return values


def scorer_state_hash(scorer: RouterScorer) -> str:
    """Hash a scorer's complete functional state, including exact-LSE children."""
    if isinstance(scorer, ExactLSEScorer):
        return record_sha256(
            {
                "architecture": "exact",
                "left": scorer_state_hash(scorer.left.scorer),
                "right": scorer_state_hash(scorer.right.scorer),
                "schema_version": "imagenetr50-router-functional-state-v1",
            }
        )
    if not isinstance(scorer, nn.Module):
        raise TypeError("unsupported router scorer state")
    return record_sha256(
        {
            "architecture": scorer.architecture,
            "rank": scorer.rank,
            "schema_version": "imagenetr50-router-functional-state-v1",
            "state": {
                key: value.detach().to(device="cpu", dtype=torch.float64).tolist()
                for key, value in sorted(scorer.state_dict().items())
            },
        }
    )


def route(query: RouterQuery, live_nodes: Sequence[ScoringNode]) -> RouteResult:
    """Select one live node per query from task-free learned scores."""
    nodes = tuple(live_nodes)
    scores = score_nodes(query, nodes)
    selected = torch.argmax(scores, dim=-1).tolist()
    node_ids = tuple(node.node_id for node in nodes)
    return RouteResult(
        query.image_ids,
        node_ids,
        tuple(node_ids[index] for index in selected),
        scores.detach().to(device="cpu"),
    )


def query_public_fields() -> frozenset[str]:
    """Expose the structural task-free API contract for tests and audits."""
    return frozenset(field.name for field in fields(RouterQuery))


__all__ = [
    "ExactLSEScorer",
    "R0Scorer",
    "R1Scorer",
    "R2Scorer",
    "R3Scorer",
    "RouteResult",
    "RouterQuery",
    "RouterScorer",
    "ScoringNode",
    "load_scorer",
    "make_scorer",
    "move_scorer",
    "query_public_fields",
    "route",
    "scorer_state_hash",
    "score_nodes",
]

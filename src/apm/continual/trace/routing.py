"""Task-free and diagnostic routing with explicit answer-isolation boundaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as functional


BASE_CANDIDATE = "base"


@dataclass(frozen=True, slots=True)
class PromptQuery:
    """A task-free router query that cannot carry an answer or task label."""

    example_id: str
    prompt: str

    def __post_init__(self) -> None:
        if not self.example_id or not self.prompt:
            raise ValueError("router prompts require identity and nonempty text")


@dataclass(frozen=True, slots=True)
class NodeCentroid:
    """A frozen-base prompt centroid and its represented training count."""

    candidate_id: str
    embedding: Tensor
    represented_examples: int

    def __post_init__(self) -> None:
        if (
            not self.candidate_id
            or self.embedding.ndim != 1
            or self.represented_examples <= 0
            or not torch.isfinite(self.embedding).all()
        ):
            raise ValueError("invalid prompt centroid")


def length_normalized_prompt_nll(
    logits: Tensor,
    input_ids: Tensor,
    attention_mask: Tensor,
) -> Tensor:
    """Score only observed prompt transitions, excluding padding and the first token."""
    if logits.ndim != 3 or input_ids.shape != attention_mask.shape or logits.shape[:2] != input_ids.shape:
        raise ValueError("prompt NLL tensor shapes differ")
    transition_mask = attention_mask[:, 1:].to(torch.bool) & attention_mask[:, :-1].to(
        torch.bool
    )
    losses = functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]).to(torch.float32),
        input_ids[:, 1:].reshape(-1),
        reduction="none",
    ).reshape(input_ids.shape[0], -1)
    counts = transition_mask.sum(dim=1)
    if torch.any(counts == 0):
        raise ValueError("prompt NLL requires at least one observed transition")
    return (losses * transition_mask).sum(dim=1) / counts


def select_lowest_nll(
    candidate_order: Sequence[str],
    scores: Mapping[str, float],
) -> str:
    """Select the first candidate with minimum finite prompt NLL."""
    if not candidate_order or set(candidate_order) != set(scores):
        raise ValueError("prompt NLL candidate order and scores differ")
    if not all(torch.isfinite(torch.tensor(score)) for score in scores.values()):
        raise ValueError("prompt NLL scores must be finite")
    return min(candidate_order, key=lambda candidate: scores[candidate])


def mean_pool_hidden(hidden: Tensor, attention_mask: Tensor) -> Tensor:
    """Attention-mask mean-pool final frozen-base hidden states and L2 normalize."""
    if hidden.ndim != 3 or attention_mask.shape != hidden.shape[:2]:
        raise ValueError("prompt embedding tensor shapes differ")
    weights = attention_mask.to(hidden.dtype).unsqueeze(-1)
    pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
    return functional.normalize(pooled.to(torch.float32), dim=-1)


def training_centroid(candidate_id: str, embeddings: Tensor) -> NodeCentroid:
    """Create one normalized centroid from frozen-base training prompt embeddings."""
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("centroids require a nonempty embedding matrix")
    centroid = functional.normalize(embeddings.to(torch.float32).mean(dim=0), dim=0)
    return NodeCentroid(candidate_id, centroid, embeddings.shape[0])


def merge_centroids(left: NodeCentroid, right: NodeCentroid, parent_id: str) -> NodeCentroid:
    """Count-weight and renormalize two child prompt centroids."""
    if left.embedding.shape != right.embedding.shape:
        raise ValueError("child centroid dimensions differ")
    total = left.represented_examples + right.represented_examples
    combined = (
        left.embedding * left.represented_examples
        + right.embedding * right.represented_examples
    ) / total
    return NodeCentroid(parent_id, functional.normalize(combined, dim=0), total)


def select_centroid(
    query_embedding: Tensor,
    centroids: Sequence[NodeCentroid],
) -> str:
    """Select the first candidate with maximum cosine to a frozen-base query."""
    if query_embedding.ndim != 1 or not centroids:
        raise ValueError("centroid routing requires one query and candidates")
    normalized = functional.normalize(query_embedding.to(torch.float32), dim=0)
    if any(centroid.embedding.shape != normalized.shape for centroid in centroids):
        raise ValueError("centroid query and candidate dimensions differ")
    selected = max(
        range(len(centroids)),
        key=lambda index: float(torch.dot(normalized, centroids[index].embedding)),
    )
    return centroids[selected].candidate_id


def select_best_candidate(
    candidate_order: Sequence[str],
    scores: Mapping[str, float],
) -> str:
    """Select the first maximum-score candidate for task-aware or oracle diagnostics."""
    if not candidate_order or set(candidate_order) != set(scores):
        raise ValueError("diagnostic candidate order and scores differ")
    return max(candidate_order, key=lambda candidate: scores[candidate])


__all__ = [
    "BASE_CANDIDATE",
    "NodeCentroid",
    "PromptQuery",
    "length_normalized_prompt_nll",
    "mean_pool_hidden",
    "merge_centroids",
    "select_best_candidate",
    "select_centroid",
    "select_lowest_nll",
    "training_centroid",
]

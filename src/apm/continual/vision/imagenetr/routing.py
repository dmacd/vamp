"""Task-free exhaustive and frozen-centroid routing with isolated diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from apm.continual.vision.imagenetr.calibration import AffineCalibration


@dataclass(frozen=True, slots=True)
class TaskFreeQuery:
    """Evaluator query identity with intentionally no task ID or label surface."""

    image_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.image_ids or len(set(self.image_ids)) != len(self.image_ids):
            raise ValueError("task-free queries require unique image identities")


@dataclass(frozen=True, slots=True)
class GroundTruth:
    """Labels supplied only to metric and explicitly diagnostic oracle functions."""

    image_ids: tuple[str, ...]
    labels: Tensor

    def __post_init__(self) -> None:
        if (
            not self.image_ids
            or self.labels.ndim != 1
            or self.labels.shape[0] != len(self.image_ids)
            or torch.any((self.labels < 0) | (self.labels >= 200))
        ):
            raise ValueError("invalid ground truth")


@dataclass(frozen=True, slots=True)
class NodeScores:
    """One live node's cached task-free scores over an exact query sequence."""

    node_hash: str
    class_ids: tuple[int, ...]
    raw: Tensor
    cosine: Tensor

    def __post_init__(self) -> None:
        if (
            len(self.node_hash) != 64
            or self.class_ids != tuple(sorted(set(self.class_ids)))
            or self.raw.ndim != 2
            or tuple(self.raw.shape) != tuple(self.cosine.shape)
            or self.raw.shape[1] != len(self.class_ids)
        ):
            raise ValueError("invalid live-node score cache")


def exhaustive_predictions(
    query: TaskFreeQuery,
    live_scores: tuple[NodeScores, ...],
    mode: str,
    calibration: AffineCalibration | None = None,
) -> Tensor:
    """Predict global classes across all disjoint live nodes without task identity."""
    if not live_scores or mode not in {"raw", "cosine", "affine_calibrated"}:
        raise ValueError("unknown or empty task-free exhaustive routing request")
    if any(scores.raw.shape[0] != len(query.image_ids) for scores in live_scores):
        raise ValueError("node caches do not cover the exact task-free query")
    all_classes = tuple(class_id for scores in live_scores for class_id in scores.class_ids)
    if len(all_classes) != len(set(all_classes)):
        raise ValueError("live-node classifier namespaces overlap")
    candidates = []
    candidate_classes = []
    for scores in live_scores:
        values = scores.cosine if mode == "cosine" else scores.raw
        if mode == "affine_calibrated":
            if calibration is None:
                raise ValueError("affine-calibrated routing requires fitted training proxies")
            temperature, offset = calibration.parameters_for(scores.node_hash)
            values = values / temperature + offset
        candidates.append(values)
        candidate_classes.extend(scores.class_ids)
    combined = torch.cat(tuple(candidates), dim=1)
    class_tensor = torch.tensor(candidate_classes, device=combined.device)
    return class_tensor[torch.argmax(combined, dim=1)]


def true_node_oracle_predictions(
    query: TaskFreeQuery,
    truth: GroundTruth,
    live_scores: tuple[NodeScores, ...],
) -> Tensor:
    """Diagnostic only: select the containing live node from the ground-truth class."""
    if query.image_ids != truth.image_ids:
        raise ValueError("oracle labels do not align with the query sequence")
    predictions = torch.full_like(truth.labels, -1)
    for scores in live_scores:
        class_set = torch.tensor(scores.class_ids, device=truth.labels.device)
        selected = torch.isin(truth.labels, class_set)
        local = torch.argmax(scores.raw[selected], dim=1)
        predictions[selected] = class_set[local]
    if torch.any(predictions < 0):
        raise ValueError("true-node oracle live nodes do not cover every ground-truth class")
    return predictions


def routed_node_predictions(
    query: TaskFreeQuery,
    selected_node_hashes: Sequence[str],
    live_scores: tuple[NodeScores, ...],
) -> Tensor:
    """Classify within one task-free router-selected node per query image."""
    selected = tuple(selected_node_hashes)
    if len(selected) != len(query.image_ids) or not live_scores:
        raise ValueError("router selections do not align with the task-free query")
    by_hash = {scores.node_hash: scores for scores in live_scores}
    if len(by_hash) != len(live_scores) or not set(selected) <= set(by_hash):
        raise ValueError("router selected a node outside the live bank")
    predictions = torch.full((len(selected),), -1, dtype=torch.long)
    for node_hash, scores in by_hash.items():
        indices = torch.tensor(
            [index for index, value in enumerate(selected) if value == node_hash],
            dtype=torch.long,
        )
        if not indices.numel():
            continue
        local = torch.argmax(scores.raw[indices], dim=1)
        classes = torch.tensor(scores.class_ids, dtype=torch.long)
        predictions[indices] = classes[local]
    if torch.any(predictions < 0):
        raise ValueError("router failed to classify every task-free query")
    return predictions


@dataclass(frozen=True, slots=True)
class FrozenCentroidRouter:
    """Training-derived frozen-feature class centroids for one-pass node routing."""

    class_ids: tuple[int, ...]
    centroids: Tensor

    def __post_init__(self) -> None:
        if (
            self.class_ids != tuple(sorted(set(self.class_ids)))
            or self.centroids.ndim != 2
            or self.centroids.shape[0] != len(self.class_ids)
        ):
            raise ValueError("invalid frozen-feature centroid router")

    def route(
        self,
        query_features: Tensor,
        node_classes: Mapping[str, tuple[int, ...]],
    ) -> tuple[str, ...]:
        """Choose the node containing the nearest represented class centroid."""
        if not node_classes:
            raise ValueError("centroid routing requires live nodes")
        similarity = F.normalize(query_features, dim=-1) @ F.normalize(self.centroids, dim=-1).T
        nearest = torch.argmax(similarity, dim=1).tolist()
        class_to_node = {
            class_id: node
            for node, classes in node_classes.items()
            for class_id in classes
        }
        if set(class_to_node) != set(self.class_ids):
            raise ValueError("live node classes differ from the centroid namespace")
        return tuple(class_to_node[self.class_ids[index]] for index in nearest)


def build_centroid_router(
    features: Tensor,
    labels: Tensor,
    represented_classes: tuple[int, ...],
) -> FrozenCentroidRouter:
    """Compute normalized class means from frozen-base training features only."""
    if features.ndim != 2 or labels.ndim != 1 or features.shape[0] != labels.shape[0]:
        raise ValueError("centroid training features and labels do not align")
    classes = tuple(sorted(set(represented_classes)))
    centroids = tuple(features[labels == class_id].to(torch.float64).mean(dim=0) for class_id in classes)
    if any(torch.isnan(value).any() for value in centroids):
        raise ValueError("a represented class lacks centroid training examples")
    return FrozenCentroidRouter(classes, torch.stack(centroids).to(torch.float32))


__all__ = [
    "FrozenCentroidRouter",
    "GroundTruth",
    "NodeScores",
    "TaskFreeQuery",
    "build_centroid_router",
    "exhaustive_predictions",
    "routed_node_predictions",
    "true_node_oracle_predictions",
]

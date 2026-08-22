"""Leakage-safe learned-router evaluation projected from immutable node logits."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from collections.abc import Callable, Mapping, Sequence
import time

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from apm.continual.artifacts import record_sha256
from apm.continual.vision.imagenetr.evaluation import cached_node_scores, project_scores
from apm.continual.vision.imagenetr.proxy_memory import TensorCache
from apm.continual.vision.imagenetr.router_artifacts import (
    InferenceNodeRef,
    RouterStore,
    SealedRouterBase,
)
from apm.continual.vision.imagenetr.router_config import RouterConfig
from apm.continual.vision.imagenetr.router_descriptor import NodeRouterFeatures
from apm.continual.vision.imagenetr.router_features import (
    RouterFeatureUniverse,
    test_transform_hash,
)
from apm.continual.vision.imagenetr.router_scores import (
    RouterQuery,
    ScoringNode,
    move_scorer,
    score_nodes,
)
from apm.continual.vision.imagenetr.routing import NodeScores


@dataclass(frozen=True, slots=True)
class RouterMetricRow:
    """One condition/stage/split learned-routing result."""

    condition_id: str
    inference_condition: str
    architecture: str
    maintenance: str
    router_seed: int
    split: str
    stage: int
    live_nodes: int
    examples: int
    routed_accuracy: float
    oracle_accuracy: float
    oracle_gap: float
    selection_accuracy: float
    conditional_selection_accuracy: float
    top2_selection_accuracy: float
    level_recall: float
    represented_count_recall: float
    evaluation_seconds: float
    candidate_adapted_forwards: int = 0

    def as_record(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RouterEvaluation:
    """Aggregate and per-image evidence from one task-free evaluation."""

    metric: RouterMetricRow
    per_image: tuple[dict[str, object], ...]


class RouterNodeScoreProvider:
    """Serve validation/test node logits with strict semantic cache identities."""

    def __init__(
        self,
        base: SealedRouterBase,
        store: RouterStore,
        config: RouterConfig,
        validation_image_ids: Sequence[str],
        backbone_factory: Callable[[], torch.nn.Module] | None = None,
        test_transform: object | None = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.base = base
        self.store = store
        self.config = config
        self.validation_image_ids = tuple(validation_image_ids)
        self.backbone_factory = backbone_factory
        self.test_transform = test_transform
        self.device = device
        self.model_hash = str(base.protocol_record["model_manifest_hash"])
        self._rows = {
            "validation": base.manifest.select(
                "train", image_ids=self.validation_image_ids
            ),
            "test": base.manifest.select("test"),
        }
        if tuple(row.image_id for row in self._rows["validation"]) != self.validation_image_ids:
            raise ValueError("router validation cache order differs from the frozen split")
        self._memory: dict[tuple[str, str], NodeScores] = {}

    def _test_scores(self, node: InferenceNodeRef) -> NodeScores:
        rows = self._rows["test"]
        image_ids = tuple(row.image_id for row in rows)
        values = {
            "class_ids": list(node.artifact.represented_class_ids),
            "image_ids_hash": record_sha256(list(image_ids)),
            "model_hash": self.model_hash,
            "node_hash": node.node_hash,
            "transform_hash": test_transform_hash(),
        }
        cache = TensorCache(
            self.base.run_root / "cache" / "evaluation_logits",
            "imagenetr50-test-score-cache-v1",
        )

        def missing() -> Mapping[str, Tensor]:
            raise FileNotFoundError(
                "sealed test-logit cache is missing; evaluation may not create it"
            )

        tensors, reused = cache.get_or_compute(values, missing)
        if not reused:
            raise AssertionError("sealed test evaluation unexpectedly computed logits")
        return NodeScores(
            node.node_hash,
            node.artifact.represented_class_ids,
            tensors["raw"],
            tensors["cosine"],
        )

    def _validation_scores(self, node: InferenceNodeRef) -> NodeScores:
        if self.backbone_factory is None or self.test_transform is None:
            raise FileNotFoundError(
                "validation logits are missing and no authenticated model factory was supplied"
            )
        scores, _reused = cached_node_scores(
            TensorCache(
                self.store.run / "features" / "evaluation_logits",
                "imagenetr50-router-validation-score-cache-v1",
            ),
            node.load(),
            self.model_hash,
            test_transform_hash(),
            self.backbone_factory,
            self.base.prepared_root,
            self._rows["validation"],
            self.test_transform,
            self.base.primary_config.lora_rank,
            self.base.primary_config.lora_alpha,
            self.base.primary_config.cosine_scale,
            self.config.feature_batch_size,
            self.config.num_workers,
            self.device,
        )
        return scores

    def full(self, node: InferenceNodeRef, split: str) -> NodeScores:
        """Return one node over the complete frozen validation or test universe."""
        if split not in self._rows:
            raise ValueError("router score split must be validation or test")
        key = (node.node_hash, split)
        if key not in self._memory:
            self._memory[key] = (
                self._test_scores(node)
                if split == "test"
                else self._validation_scores(node)
            )
        return self._memory[key]

    def project(
        self,
        node: InferenceNodeRef,
        split: str,
        image_ids: Sequence[str],
    ) -> NodeScores:
        """Project cached universe rows to an exact historical image sequence."""
        universe_ids = tuple(row.image_id for row in self._rows[split])
        index = {image_id: position for position, image_id in enumerate(universe_ids)}
        if len(set(image_ids)) != len(image_ids) or any(value not in index for value in image_ids):
            raise ValueError("router evaluation requests identities outside its split")
        indices = torch.tensor([index[value] for value in image_ids], dtype=torch.long)
        return project_scores(self.full(node, split), indices)


class CentroidNodeScorer(nn.Module):
    """Fixed fit-only class-centroid score collapsed to one live node."""

    architecture = "centroid"
    rank = 1

    def __init__(self, class_ids: tuple[int, ...], centroids: Tensor) -> None:
        super().__init__()
        self.class_ids = class_ids
        self.register_buffer(
            "centroids", F.normalize(centroids.to(torch.float32), dim=-1)
        )

    def score(self, query: RouterQuery, node_features: object) -> Tensor:
        del node_features
        query_values = F.normalize(query.prelogits.to(torch.float32), dim=-1)
        return torch.max(query_values @ self.centroids.to(query_values).T, dim=-1).values


def centroid_scoring_nodes(
    fit_universe: RouterFeatureUniverse,
    fit_image_ids: Sequence[str],
    inference_nodes: Sequence[InferenceNodeRef],
    node_features: Mapping[str, NodeRouterFeatures],
) -> tuple[ScoringNode, ...]:
    """Build the existing fit-only frozen-feature centroid routing baseline."""
    indices = fit_universe.indices(fit_image_ids)
    features = fit_universe.prelogits[indices]
    labels = fit_universe.labels[indices]
    result = []
    for node in inference_nodes:
        class_ids = node.artifact.represented_class_ids
        centroids = torch.stack(
            tuple(features[labels == class_id].to(torch.float64).mean(dim=0) for class_id in class_ids)
        ).to(torch.float32)
        if not torch.isfinite(centroids).all():
            raise ValueError("centroid baseline lacks fit examples for a represented class")
        result.append(
            ScoringNode(
                node.logical_node.node_id,
                CentroidNodeScorer(class_ids, centroids),
                node_features[node.logical_node.node_id],
                node.artifact.represented_task_ids,
                class_ids,
                int(torch.isin(labels, torch.tensor(class_ids)).sum().item()),
            )
        )
    return tuple(result)


def _percent(mask: Tensor) -> float:
    if mask.numel() == 0:
        return float("nan")
    return 100.0 * float(mask.to(torch.float32).mean().item())


def evaluate_router_frontier(
    *,
    condition_id: str,
    inference_condition: str,
    architecture: str,
    maintenance: str,
    router_seed: int,
    split: str,
    stage: int,
    universe: RouterFeatureUniverse,
    image_ids: Sequence[str],
    scoring_nodes: Sequence[ScoringNode],
    inference_nodes: Sequence[InferenceNodeRef],
    score_provider: RouterNodeScoreProvider,
    device: torch.device,
) -> RouterEvaluation:
    """Evaluate task-free node selection and routed classification without labels in queries."""
    started = time.monotonic()
    learned = tuple(scoring_nodes)
    inference = tuple(inference_nodes)
    if (
        len(learned) != len(inference)
        or tuple(node.node_id for node in learned)
        != tuple(node.logical_node.node_id for node in inference)
    ):
        raise ValueError("learned and inference frontiers do not align")
    indices = universe.indices(image_ids)
    query = RouterQuery(
        tuple(image_ids),
        universe.prelogits[indices],
        {
            name: values[indices]
            for name, values in universe.cls_activations.items()
        },
    ).to(device)
    labels = universe.labels[indices]
    tasks = universe.task_ids[indices]
    for node in learned:
        move_scorer(node.scorer, device)
        node.features = node.features.to(device)
    scores = score_nodes(query, learned).detach().to(device="cpu")
    selected = torch.argmax(scores, dim=-1)
    top_count = min(2, len(learned))
    top = torch.topk(scores, k=top_count, dim=-1).indices
    class_owner: dict[int, int] = {}
    for position, node in enumerate(learned):
        for class_id in node.represented_class_ids:
            if class_id in class_owner:
                raise ValueError("live router class namespaces overlap")
            class_owner[class_id] = position
    if set(class_owner) != set(range(4 * stage)):
        raise ValueError("live router frontier does not cover the stage classes")
    target = torch.tensor([class_owner[int(label)] for label in labels], dtype=torch.long)
    node_scores = tuple(
        score_provider.project(node, split, image_ids) for node in inference
    )
    oracle_predictions = torch.empty_like(labels)
    routed_predictions = torch.empty_like(labels)
    for position, cached in enumerate(node_scores):
        classes = torch.tensor(cached.class_ids, dtype=torch.long)
        truth_rows = torch.nonzero(target == position, as_tuple=False).reshape(-1)
        if truth_rows.numel():
            oracle_predictions[truth_rows] = classes[
                torch.argmax(cached.raw[truth_rows], dim=-1)
            ]
        selected_rows = torch.nonzero(selected == position, as_tuple=False).reshape(-1)
        if selected_rows.numel():
            routed_predictions[selected_rows] = classes[
                torch.argmax(cached.raw[selected_rows], dim=-1)
            ]
    selection_correct = selected == target
    oracle_correct = oracle_predictions == labels
    routed_correct = routed_predictions == labels
    top2_correct = torch.any(top == target[:, None], dim=-1)
    predicted_levels = torch.tensor(
        [inference[index].logical_node.level for index in selected.tolist()]
    )
    target_levels = torch.tensor(
        [inference[index].logical_node.level for index in target.tolist()]
    )
    predicted_counts = torch.tensor(
        [len(inference[index].logical_node.task_ids) for index in selected.tolist()]
    )
    target_counts = torch.tensor(
        [len(inference[index].logical_node.task_ids) for index in target.tolist()]
    )
    conditional = (
        _percent(selection_correct[oracle_correct])
        if torch.any(oracle_correct)
        else float("nan")
    )
    routed_accuracy = _percent(routed_correct)
    oracle_accuracy = _percent(oracle_correct)
    metric = RouterMetricRow(
        condition_id,
        inference_condition,
        architecture,
        maintenance,
        router_seed,
        split,
        stage,
        len(learned),
        len(image_ids),
        routed_accuracy,
        oracle_accuracy,
        oracle_accuracy - routed_accuracy,
        _percent(selection_correct),
        conditional,
        _percent(top2_correct),
        _percent(predicted_levels == target_levels),
        _percent(predicted_counts == target_counts),
        time.monotonic() - started,
    )
    per_image = tuple(
        {
            "condition_id": condition_id,
            "image_id": image_id,
            "label": int(label),
            "oracle_correct": bool(oracle_ok),
            "routed_correct": bool(routed_ok),
            "router_seed": router_seed,
            "selected_node_id": learned[int(chosen)].node_id,
            "selection_correct": bool(selection_ok),
            "split": split,
            "stage": stage,
            "target_node_id": learned[int(owner)].node_id,
            "task": int(task),
            "top2_selection_correct": bool(top2_ok),
        }
        for image_id, label, task, chosen, owner, selection_ok, top2_ok, oracle_ok, routed_ok in zip(
            image_ids,
            labels.tolist(),
            tasks.tolist(),
            selected.tolist(),
            target.tolist(),
            selection_correct.tolist(),
            top2_correct.tolist(),
            oracle_correct.tolist(),
            routed_correct.tolist(),
        )
    )
    for node in learned:
        move_scorer(node.scorer, torch.device("cpu"))
        node.features = node.features.to(torch.device("cpu"))
    return RouterEvaluation(metric, per_image)


__all__ = [
    "RouterEvaluation",
    "RouterMetricRow",
    "RouterNodeScoreProvider",
    "centroid_scoring_nodes",
    "evaluate_router_frontier",
]

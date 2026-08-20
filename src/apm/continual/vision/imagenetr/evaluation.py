"""Cached historical task-free, calibrated, and diagnostic tree evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable, Mapping, Sequence
import time

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from apm.continual.artifacts import record_sha256
from apm.continual.vision.imagenetr.artifacts import NodeBundle, VisionStore
from apm.continual.vision.imagenetr.calibration import (
    CalibrationExamples,
    fit_affine_calibration,
)
from apm.continual.vision.imagenetr.data import DatasetManifest, ImageRecord, ManifestDataset
from apm.continual.vision.imagenetr.lineage import MaterializedSnapshot, TreeBuildResult
from apm.continual.vision.imagenetr.lora import load_adapter_factors
from apm.continual.vision.imagenetr.metrics import accuracy
from apm.continual.vision.imagenetr.model import AdapterVisionModel
from apm.continual.vision.imagenetr.proxy_memory import TensorCache, require_training_only
from apm.continual.vision.imagenetr.routing import (
    FrozenCentroidRouter,
    GroundTruth,
    NodeScores,
    TaskFreeQuery,
    build_centroid_router,
    exhaustive_predictions,
    routed_node_predictions,
    true_node_oracle_predictions,
)


@dataclass(frozen=True, slots=True)
class StageAccuracy:
    """One condition/stage/router global class-incremental result."""

    condition: str
    stage: int
    score_mode: str
    accuracy: float
    live_nodes: int
    candidate_forwards: int
    evaluation_seconds: float
    diagnostic: bool
    frozen_router_forwards: int = 0
    routing_accuracy: float | None = None

    def as_record(self) -> dict[str, object]:
        """Return one stage_accuracy.csv-compatible row."""
        return {
            "accuracy": self.accuracy,
            "candidate_forwards": self.candidate_forwards,
            "condition": self.condition,
            "diagnostic": self.diagnostic,
            "evaluation_seconds": self.evaluation_seconds,
            "frozen_router_forwards": self.frozen_router_forwards,
            "live_nodes": self.live_nodes,
            "routing_accuracy": self.routing_accuracy,
            "score_mode": self.score_mode,
            "stage": self.stage,
        }


@dataclass(frozen=True, slots=True)
class TaskAccuracy:
    """One task cell within a stage/router triangular accuracy matrix."""

    condition: str
    stage: int
    task: int
    score_mode: str
    accuracy: float

    def as_record(self) -> dict[str, object]:
        """Return one task_accuracy_matrix.csv-compatible long row."""
        return {
            "accuracy": self.accuracy,
            "condition": self.condition,
            "score_mode": self.score_mode,
            "stage": self.stage,
            "task": self.task,
        }


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Complete stage and per-task ledgers for one condition."""

    stages: tuple[StageAccuracy, ...]
    tasks: tuple[TaskAccuracy, ...]
    cache_hits: int
    cache_misses: int


def _direct_frozen_features(
    backbone_factory: Callable[[], torch.nn.Module],
    prepared_root: str | Path,
    rows: Sequence[ImageRecord],
    transform: object,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Tensor:
    """Forward one immutable image universe through the frozen base exactly once."""
    backbone = backbone_factory().to(device).eval()
    dataset = ManifestDataset(prepared_root, rows, transform, 0, 0)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    batches = []
    with torch.inference_mode():
        for images, _labels, _image_ids in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                tokens = backbone.forward_features(images)
                features = backbone.forward_head(tokens, pre_logits=True)
            batches.append(features.float().cpu())
    del backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(tuple(batches))


def _cached_frozen_features(
    cache: TensorCache,
    model_hash: str,
    transform_hash: str,
    split: str,
    backbone_factory: Callable[[], torch.nn.Module],
    prepared_root: str | Path,
    rows: Sequence[ImageRecord],
    transform: object,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[Tensor, bool]:
    image_ids = tuple(row.image_id for row in rows)
    tensors, reused = cache.get_or_compute(
        {
            "image_ids_hash": record_sha256(list(image_ids)),
            "model_hash": model_hash,
            "split": split,
            "transform_hash": transform_hash,
        },
        lambda: {
            "features": _direct_frozen_features(
                backbone_factory,
                prepared_root,
                rows,
                transform,
                batch_size,
                num_workers,
                device,
            )
        },
    )
    if tensors["features"].shape != (len(rows), 768):
        raise ValueError("frozen-feature cache has the wrong image or feature dimension")
    return tensors["features"], reused


def _direct_node_scores(
    bundle: NodeBundle,
    backbone_factory: Callable[[], torch.nn.Module],
    prepared_root: str | Path,
    rows: Sequence[ImageRecord],
    transform: object,
    rank: int,
    alpha: int,
    cosine_scale: float,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    model = AdapterVisionModel(
        backbone_factory(),
        bundle.classifier.class_ids,
        rank,
        alpha,
        0.0,
        0,
        bundle.classifier,
    )
    load_adapter_factors(model, bundle.adapter)
    model.to(device).eval()
    dataset = ManifestDataset(prepared_root, rows, transform, 0, 0)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    raw_batches, cosine_batches = [], []
    with torch.inference_mode():
        for images, _labels, _image_ids in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                features = model.features(images)
                raw_batches.append(model.classifier(features).float().cpu())
                cosine_batches.append(
                    model.classifier.cosine_scores(features, cosine_scale).float().cpu()
                )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(tuple(raw_batches)), torch.cat(tuple(cosine_batches))


def cached_node_scores(
    cache: TensorCache,
    bundle: NodeBundle,
    model_hash: str,
    transform_hash: str,
    backbone_factory: Callable[[], torch.nn.Module],
    prepared_root: str | Path,
    rows: Sequence[ImageRecord],
    transform: object,
    rank: int,
    alpha: int,
    cosine_scale: float,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[NodeScores, bool]:
    """Load or forward one node exactly once over one immutable image universe."""
    image_ids = tuple(row.image_id for row in rows)
    values = {
        "class_ids": list(bundle.classifier.class_ids),
        "image_ids_hash": record_sha256(list(image_ids)),
        "model_hash": model_hash,
        "node_hash": bundle.artifact.content_hash,
        "transform_hash": transform_hash,
    }
    tensors, reused = cache.get_or_compute(
        values,
        lambda: dict(
            zip(
                ("raw", "cosine"),
                _direct_node_scores(
                    bundle,
                    backbone_factory,
                    prepared_root,
                    rows,
                    transform,
                    rank,
                    alpha,
                    cosine_scale,
                    batch_size,
                    num_workers,
                    device,
                ),
            )
        ),
    )
    scores = NodeScores(
        bundle.artifact.content_hash,
        bundle.classifier.class_ids,
        tensors["raw"],
        tensors["cosine"],
    )
    if scores.raw.shape[0] != len(image_ids):
        raise ValueError("evaluation cache row count differs from its image universe")
    return scores, reused


def project_scores(scores: NodeScores, indices: Tensor) -> NodeScores:
    """Project cached universe rows into a historical stage without forwarding."""
    return NodeScores(
        scores.node_hash,
        scores.class_ids,
        scores.raw[indices],
        scores.cosine[indices],
    )


def _task_rows(
    condition: str,
    stage: int,
    mode: str,
    predictions: Tensor,
    truth: GroundTruth,
) -> tuple[TaskAccuracy, ...]:
    return tuple(
        TaskAccuracy(
            condition,
            stage,
            task,
            mode,
            100.0
            * float(
                torch.mean(
                    (predictions[truth.labels // 4 == task].cpu()
                    == truth.labels[truth.labels // 4 == task].cpu()).float()
                ).item()
            ),
        )
        for task in range(stage)
    )


def evaluate_tree(
    store: VisionStore,
    tree: TreeBuildResult,
    manifest: DatasetManifest,
    prepared_root: str | Path,
    backbone_factory: Callable[[], torch.nn.Module],
    test_transform: object,
    model_hash: str,
    transform_hash: str,
    rank: int,
    alpha: int,
    cosine_scale: float,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    condition_override: str | None = None,
) -> EvaluationResult:
    """Evaluate every historical snapshot from shared per-node score caches."""
    condition = condition_override or (
        f"logt_{'drift' if tree.policy.method == 'output_drift' else tree.policy.method}_"
        f"r{tree.policy.output_rank}_"
        f"repair{int(round(100 * tree.policy.repair_fraction)):03d}"
    )
    nodes_by_hash = {bundle.artifact.content_hash: bundle for bundle in tree.nodes}
    test_universe = manifest.select("test")
    train_universe = manifest.select("train")
    proxy_ids = tuple(
        sorted(
            {
                image_id
                for bundle in nodes_by_hash.values()
                for image_id in bundle.artifact.proxy_image_ids
            }
        )
    )
    require_training_only(proxy_ids, manifest, "calibration proxy")
    proxy_universe = manifest.select("train", image_ids=proxy_ids)
    test_cache = TensorCache(
        store.run / "cache" / "evaluation_logits", "imagenetr50-test-score-cache-v1"
    )
    proxy_cache = TensorCache(
        store.run / "cache" / "evaluation_logits", "imagenetr50-proxy-score-cache-v1"
    )
    feature_cache = TensorCache(
        store.run / "cache" / "frozen_features", "imagenetr50-frozen-router-features-v1"
    )
    train_features, train_feature_hit = _cached_frozen_features(
        feature_cache,
        model_hash,
        f"{transform_hash}:frozen-router-training",
        "train",
        backbone_factory,
        prepared_root,
        train_universe,
        test_transform,
        batch_size,
        num_workers,
        device,
    )
    test_features, test_feature_hit = _cached_frozen_features(
        feature_cache,
        model_hash,
        f"{transform_hash}:frozen-router-test",
        "test",
        backbone_factory,
        prepared_root,
        test_universe,
        test_transform,
        batch_size,
        num_workers,
        device,
    )
    full_router = build_centroid_router(
        train_features,
        torch.tensor(
            [row.remapped_class_index for row in train_universe], dtype=torch.long
        ),
        tuple(range(200)),
    )
    cached_test: dict[str, NodeScores] = {}
    cached_proxy: dict[str, NodeScores] = {}
    hits = int(train_feature_hit) + int(test_feature_hit)
    misses = int(not train_feature_hit) + int(not test_feature_hit)
    for node_hash, bundle in sorted(nodes_by_hash.items()):
        test_scores, test_hit = cached_node_scores(
            test_cache,
            bundle,
            model_hash,
            transform_hash,
            backbone_factory,
            prepared_root,
            test_universe,
            test_transform,
            rank,
            alpha,
            cosine_scale,
            batch_size,
            num_workers,
            device,
        )
        proxy_scores, proxy_hit = cached_node_scores(
            proxy_cache,
            bundle,
            model_hash,
            f"{transform_hash}:training-derived-proxy-eval",
            backbone_factory,
            prepared_root,
            proxy_universe,
            test_transform,
            rank,
            alpha,
            cosine_scale,
            batch_size,
            num_workers,
            device,
        )
        cached_test[node_hash], cached_proxy[node_hash] = test_scores, proxy_scores
        hits += int(test_hit) + int(proxy_hit)
        misses += int(not test_hit) + int(not proxy_hit)
    test_index = {row.image_id: index for index, row in enumerate(test_universe)}
    proxy_index = {row.image_id: index for index, row in enumerate(proxy_universe)}
    stages: list[StageAccuracy] = []
    tasks: list[TaskAccuracy] = []
    for snapshot in tree.snapshots:
        started = time.monotonic()
        stage_rows = manifest.select("test", range(snapshot.stage))
        stage_indices = torch.tensor([test_index[row.image_id] for row in stage_rows])
        query = TaskFreeQuery(tuple(row.image_id for row in stage_rows))
        truth = GroundTruth(
            query.image_ids,
            torch.tensor([row.remapped_class_index for row in stage_rows], dtype=torch.long),
        )
        live_scores = tuple(
            project_scores(cached_test[node_hash], stage_indices)
            for node_hash in snapshot.node_hashes
        )
        live_proxy_ids = tuple(
            sorted(
                {
                    image_id
                    for node_hash in snapshot.node_hashes
                    for image_id in nodes_by_hash[node_hash].artifact.proxy_image_ids
                }
            )
        )
        calibration_indices = torch.tensor([proxy_index[value] for value in live_proxy_ids])
        calibration_rows = {row.image_id: row for row in proxy_universe}
        calibration_examples = CalibrationExamples(
            live_proxy_ids,
            torch.tensor(
                [calibration_rows[value].remapped_class_index for value in live_proxy_ids],
                dtype=torch.long,
            ),
        )
        calibration = fit_affine_calibration(
            calibration_examples,
            {
                node_hash: cached_proxy[node_hash].raw[calibration_indices]
                for node_hash in snapshot.node_hashes
            },
            {
                node_hash: cached_proxy[node_hash].class_ids
                for node_hash in snapshot.node_hashes
            },
        )
        predictions = {
            mode: exhaustive_predictions(
                query,
                live_scores,
                mode,
                calibration if mode == "affine_calibrated" else None,
            )
            for mode in ("raw", "cosine", "affine_calibrated")
        }
        predictions["true_node_oracle"] = true_node_oracle_predictions(
            query, truth, live_scores
        )
        seen_classes = tuple(range(4 * snapshot.stage))
        stage_router = FrozenCentroidRouter(
            seen_classes, full_router.centroids[: len(seen_classes)]
        )
        node_classes = {
            node_hash: cached_test[node_hash].class_ids
            for node_hash in snapshot.node_hashes
        }
        selected_nodes = stage_router.route(test_features[stage_indices], node_classes)
        predictions["centroid_router"] = routed_node_predictions(
            query, selected_nodes, live_scores
        )
        true_nodes = {
            class_id: node_hash
            for node_hash, classes in node_classes.items()
            for class_id in classes
        }
        centroid_routing_accuracy = 100.0 * sum(
            selected == true_nodes[int(label)]
            for selected, label in zip(selected_nodes, truth.labels.tolist())
        ) / len(selected_nodes)
        elapsed = time.monotonic() - started
        for mode, values in predictions.items():
            diagnostic = mode == "true_node_oracle"
            centroid = mode == "centroid_router"
            stages.append(
                StageAccuracy(
                    condition,
                    snapshot.stage,
                    mode,
                    accuracy(values, truth),
                    len(snapshot.node_hashes),
                    1 if diagnostic or centroid else len(snapshot.node_hashes),
                    elapsed,
                    diagnostic,
                    1 if centroid else 0,
                    centroid_routing_accuracy if centroid else None,
                )
            )
            tasks.extend(_task_rows(condition, snapshot.stage, mode, values, truth))
    return EvaluationResult(tuple(stages), tuple(tasks), hits, misses)


__all__ = [
    "EvaluationResult",
    "StageAccuracy",
    "TaskAccuracy",
    "cached_node_scores",
    "evaluate_tree",
    "project_scores",
]

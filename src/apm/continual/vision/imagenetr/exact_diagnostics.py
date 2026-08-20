"""Selected real-model exact-rank merge diagnostics on training-only proxies."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Callable, Mapping, Sequence

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from apm.continual.vision.imagenetr.artifacts import NodeBundle
from apm.continual.vision.imagenetr.bank import LogicalNode, MergeEvent, simulate_topology
from apm.continual.vision.imagenetr.data import (
    DatasetManifest,
    ImageRecord,
    ManifestDataset,
)
from apm.continual.vision.imagenetr.heads import ClassifierRows, union_classifier_rows
from apm.continual.vision.imagenetr.lineage import TreeBuildResult
from apm.continual.vision.imagenetr.lora import load_adapter_factors
from apm.continual.vision.imagenetr.merging.common import (
    LoRAFactors,
    exact_weighted_factors,
    normalized_weights,
)
from apm.continual.vision.imagenetr.model import AdapterVisionModel
from apm.continual.vision.imagenetr.proxy_memory import require_training_only


def selected_diagnostic_events() -> tuple[MergeEvent, ...]:
    """Select two low-, two mid-, and two high-level deterministic merge events."""
    events, _snapshots = simulate_topology(50)
    by_level = {
        level: tuple(event for event in events if event.parent.level == level)
        for level in range(1, 5)
    }
    high = by_level[3] + by_level[4]
    selected = (
        by_level[1][0],
        by_level[1][-1],
        by_level[2][0],
        by_level[2][-1],
        high[0],
        high[-1],
    )
    if len({event.merge_id for event in selected}) != 6:
        raise ValueError("exact-rank diagnostic event selection is not unique")
    return selected


def _nodes_by_interval(tree: TreeBuildResult) -> dict[tuple[int, int, int], NodeBundle]:
    result = {
        (
            bundle.artifact.level,
            bundle.artifact.first_task,
            bundle.artifact.last_task,
        ): bundle
        for bundle in tree.nodes
    }
    if len(result) != len(tree.nodes):
        raise ValueError("tree contains duplicate logical intervals")
    return result


def _logical_key(node: LogicalNode) -> tuple[int, int, int]:
    return node.level, node.first_task, node.last_task


def _logits(
    adapter: Mapping[str, LoRAFactors],
    classifier: ClassifierRows,
    rank: int,
    backbone_factory: Callable[[], torch.nn.Module],
    prepared_root: str | Path,
    rows: Sequence[ImageRecord],
    transform: object,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    model = AdapterVisionModel(
        backbone_factory(),
        classifier.class_ids,
        rank,
        rank,
        0.0,
        0,
        classifier,
    )
    load_adapter_factors(model, adapter)
    model.to(device).eval()
    dataset = ManifestDataset(prepared_root, rows, transform, 0, 0)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    batches = []
    with torch.inference_mode():
        for images, _labels, _image_ids in loader:
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                batches.append(model(images.to(device)).float().cpu())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(tuple(batches))


def _predictions(logits: Tensor, classes: tuple[int, ...]) -> Tensor:
    return torch.tensor(classes, dtype=torch.long)[torch.argmax(logits, dim=1)]


def _accuracy(predictions: Tensor, labels: Tensor) -> float:
    return 100.0 * float((predictions == labels).float().mean().item())


def run_exact_rank_diagnostics(
    trees: Mapping[str, TreeBuildResult],
    manifest: DatasetManifest,
    prepared_root: str | Path,
    backbone_factory: Callable[[], torch.nn.Module],
    transform: object,
    rank: int,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, object], ...]:
    """Compare exact and compressed real parent functions on six fixed merges."""
    required = {
        "retrain": "logt_retrain_union_r16",
        "svd": "logt_svd_r16_repair000",
        "core_tsv": "logt_core_tsv_r16_repair000",
        "drift": "logt_drift_r16_repair000",
    }
    if not set(required.values()) <= set(trees):
        raise ValueError("exact-rank diagnostics require all un-repaired primary trees")
    indexed = {
        method: _nodes_by_interval(trees[name]) for method, name in required.items()
    }
    records = []
    for event in selected_diagnostic_events():
        left = indexed["svd"][_logical_key(event.left)]
        right = indexed["svd"][_logical_key(event.right)]
        proxy_ids = tuple(
            sorted(set(left.artifact.proxy_image_ids + right.artifact.proxy_image_ids))
        )
        require_training_only(proxy_ids, manifest, "exact-rank proxy diagnostic")
        rows = manifest.select("train", image_ids=proxy_ids)
        labels = torch.tensor(
            [row.remapped_class_index for row in rows], dtype=torch.long
        )
        weights = normalized_weights(
            (
                left.artifact.represented_train_image_count,
                right.artifact.represented_train_image_count,
            )
        )
        exact_adapter = {
            module: exact_weighted_factors(
                (left.adapter[module], right.adapter[module]), weights
            )
            for module in sorted(left.adapter)
        }
        exact_head = union_classifier_rows((left.classifier, right.classifier))
        left_logits = _logits(
            left.adapter,
            left.classifier,
            rank,
            backbone_factory,
            prepared_root,
            rows,
            transform,
            batch_size,
            device,
        )
        right_logits = _logits(
            right.adapter,
            right.classifier,
            rank,
            backbone_factory,
            prepared_root,
            rows,
            transform,
            batch_size,
            device,
        )
        child_predictions = torch.full_like(labels, -1)
        left_mask = torch.isin(labels, torch.tensor(left.classifier.class_ids))
        right_mask = torch.isin(labels, torch.tensor(right.classifier.class_ids))
        child_predictions[left_mask] = _predictions(
            left_logits[left_mask], left.classifier.class_ids
        )
        child_predictions[right_mask] = _predictions(
            right_logits[right_mask], right.classifier.class_ids
        )
        exact_logits = _logits(
            exact_adapter,
            exact_head,
            2 * rank,
            backbone_factory,
            prepared_root,
            rows,
            transform,
            batch_size,
            device,
        )
        accuracies: dict[str, float] = {
            "child_true_node_oracle_accuracy": _accuracy(child_predictions, labels),
            "exact_sum_r32_accuracy": _accuracy(
                _predictions(exact_logits, exact_head.class_ids), labels
            ),
        }
        for method in ("svd", "core_tsv", "drift", "retrain"):
            parent = indexed[method][_logical_key(event.parent)]
            logits = _logits(
                parent.adapter,
                parent.classifier,
                rank,
                backbone_factory,
                prepared_root,
                rows,
                transform,
                batch_size,
                device,
            )
            accuracies[f"{method}_parent_accuracy"] = _accuracy(
                _predictions(logits, parent.classifier.class_ids), labels
            )
        records.append(
            {
                **accuracies,
                "child_a_interval": event.left.one_based_interval,
                "child_b_interval": event.right.one_based_interval,
                "exact_stored_rank": 2 * rank,
                "level": event.parent.level,
                "merge_id": event.merge_id,
                "parent_interval": event.parent.one_based_interval,
                "proxy_image_count": len(rows),
                "proxy_image_ids": list(proxy_ids),
                "schema_version": "imagenetr50-exact-rank-diagnostic-row-v1",
            }
        )
    return tuple(records)


__all__ = ["run_exact_rank_diagnostics", "selected_diagnostic_events"]

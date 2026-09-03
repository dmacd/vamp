"""Cached task-free behavior observations for ImageNet-R integrators."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from apm.continual.vision.imagenetr.artifacts import NodeBundle
from apm.continual.vision.imagenetr.data import ImageRecord, ManifestDataset
from apm.continual.vision.imagenetr.heads import ClassifierRows
from apm.continual.vision.imagenetr.integrator_model import (
    CLASS_COUNT,
    PRELOGIT_DIM,
    VARIANT_SLOT_DIMS,
    IntegratorObservations,
)
from apm.continual.vision.imagenetr.lora import load_adapter_factors
from apm.continual.vision.imagenetr.merging.common import LoRAFactors
from apm.continual.vision.imagenetr.model import AdapterVisionModel
from apm.continual.vision.imagenetr.proxy_memory import TensorCache
from apm.continual.vision.imagenetr.router_artifacts import InferenceNodeRef


@dataclass(frozen=True, slots=True)
class BehaviorNode:
    """The exact frozen tensors and temporal semantics exposed to integration."""

    node_hash: str
    level: int
    represented_task_ids: tuple[int, ...]
    adapter: dict[str, LoRAFactors]
    classifier: ClassifierRows

    @classmethod
    def from_bundle(cls, bundle: NodeBundle) -> "BehaviorNode":
        """Project an authenticated locally trained bundle to behavior semantics."""
        return cls(
            bundle.artifact.content_hash,
            bundle.artifact.level,
            bundle.artifact.represented_task_ids,
            bundle.adapter,
            bundle.classifier,
        )

    @classmethod
    def from_sealed(cls, node: InferenceNodeRef) -> "BehaviorNode":
        """Load an authenticated read-only primary-run node."""
        return cls.from_bundle(node.load())


@dataclass(frozen=True, slots=True)
class FrontierTensors:
    """Reusable per-node behavior tensors over one exact image universe."""

    image_ids: tuple[str, ...]
    labels: Tensor
    normalized_prelogits: Tensor
    raw_scores: Tensor
    cosine_scores: Tensor
    local_log_probabilities: Tensor
    base_scores: Tensor
    ownership: Tensor
    active_slot_mask: Tensor
    seen_class_mask: Tensor
    cache_hits: int
    cache_misses: int
    base_example_forwards: int
    node_example_forwards: int

    def __post_init__(self) -> None:
        rows, slots = len(self.image_ids), len(self.active_slot_mask)
        if (
            rows < 1
            or len(set(self.image_ids)) != rows
            or self.labels.shape != (rows,)
            or self.labels.dtype != torch.int64
            or self.normalized_prelogits.shape != (rows, slots, PRELOGIT_DIM)
            or self.raw_scores.shape != (rows, slots, CLASS_COUNT)
            or self.cosine_scores.shape != (rows, slots, CLASS_COUNT)
            or self.local_log_probabilities.shape != (rows, slots, CLASS_COUNT)
            or self.base_scores.shape != (rows, slots, CLASS_COUNT)
            or self.ownership.shape != (slots, CLASS_COUNT)
            or self.ownership.dtype != torch.bool
            or self.active_slot_mask.shape != (slots,)
            or self.active_slot_mask.dtype != torch.bool
            or self.seen_class_mask.shape != (CLASS_COUNT,)
            or self.seen_class_mask.dtype != torch.bool
            or not bool(self.active_slot_mask.any())
            or not torch.equal(self.ownership.any(dim=0), self.seen_class_mask)
            or self.cache_hits < 0
            or self.cache_misses < 0
            or self.base_example_forwards < 0
            or self.node_example_forwards < 0
        ):
            raise ValueError("frontier behavior tensors are malformed")

    def observations(self, variant: str) -> IntegratorObservations:
        """Assemble one declared feature variant without any additional model forward."""
        if variant not in VARIANT_SLOT_DIMS:
            raise ValueError("unknown integrator feature variant")
        ownership = self.ownership.to(self.raw_scores.dtype)[None].expand(len(self.labels), -1, -1)
        active = self.active_slot_mask.to(self.raw_scores.dtype)[None, :, None].expand(
            len(self.labels), -1, -1
        )
        fields = [self.raw_scores, ownership]
        if variant != "scores":
            fields = [self.normalized_prelogits, self.raw_scores, self.local_log_probabilities]
            if variant == "behavior_base":
                fields.append(self.base_scores)
            fields.append(ownership)
        fields.append(active)
        features = (
            torch.cat(tuple(fields), dim=2)
            .flatten(1)
            .to(torch.bfloat16)
            .detach()
        )
        baseline = torch.full(
            (len(self.labels), CLASS_COUNT), -torch.inf, dtype=torch.float32
        )
        for slot in torch.where(self.active_slot_mask)[0].tolist():
            owned = self.ownership[slot]
            baseline[:, owned] = self.raw_scores[:, slot, owned].float()
        return IntegratorObservations(
            features,
            baseline,
            self.seen_class_mask,
            self.active_slot_mask,
            variant,
            VARIANT_SLOT_DIMS[variant],
        )

    def select(self, image_ids: Sequence[str]) -> "FrontierTensors":
        """Project all cached behavior fields to one exact identity sequence."""
        index = {image_id: position for position, image_id in enumerate(self.image_ids)}
        if len(set(image_ids)) != len(image_ids) or any(value not in index for value in image_ids):
            raise ValueError("frontier projection contains unknown or duplicate image IDs")
        indices = torch.tensor([index[value] for value in image_ids], dtype=torch.int64)
        return FrontierTensors(
            tuple(image_ids),
            self.labels[indices],
            self.normalized_prelogits[indices],
            self.raw_scores[indices],
            self.cosine_scores[indices],
            self.local_log_probabilities[indices],
            self.base_scores[indices],
            self.ownership,
            self.active_slot_mask,
            self.seen_class_mask,
            self.cache_hits,
            self.cache_misses,
            self.base_example_forwards,
            self.node_example_forwards,
        )

    def control_predictions(self) -> dict[str, Tensor]:
        """Return parameter-free raw, local-normalized, and frozen-base union controls."""
        unions = {
            name: torch.full((len(self.labels), CLASS_COUNT), -torch.inf)
            for name in (
                "raw_union",
                "cosine_union",
                "local_log_probability_union",
                "base_head_union",
            )
        }
        sources = {
            "raw_union": self.raw_scores,
            "cosine_union": self.cosine_scores,
            "local_log_probability_union": self.local_log_probabilities,
            "base_head_union": self.base_scores,
        }
        for slot in torch.where(self.active_slot_mask)[0].tolist():
            owned = self.ownership[slot]
            for name, values in sources.items():
                unions[name][:, owned] = values[:, slot, owned].float()
        return {name: values.argmax(dim=1) for name, values in unions.items()}

    def true_node_oracle_predictions(self) -> Tensor:
        """Route with the true class-owning node for diagnostic evaluation only."""
        predictions = torch.empty_like(self.labels)
        for slot in torch.where(self.active_slot_mask)[0].tolist():
            owned = self.ownership[slot]
            rows = owned[self.labels]
            class_ids = torch.where(owned)[0]
            predictions[rows] = class_ids[
                self.raw_scores[rows, slot][:, owned].argmax(dim=1)
            ]
        return predictions


def _loader(
    prepared_root: Path,
    rows: Sequence[ImageRecord],
    transform: object,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        ManifestDataset(prepared_root, rows, transform, 0, 0),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )


def _frozen_prelogits(
    backbone_factory: Callable[[], nn.Module],
    prepared_root: Path,
    rows: Sequence[ImageRecord],
    transform: object,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Tensor:
    backbone = backbone_factory().to(device).eval()
    batches = []
    from tqdm.auto import tqdm

    loader = _loader(prepared_root, rows, transform, batch_size, num_workers, device)
    with torch.inference_mode():
        for images, _labels, _ids in tqdm(
            loader, desc=f"frozen prelogits ({len(rows):,})", leave=False, unit="batch"
        ):
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                tokens = backbone.forward_features(images.to(device, non_blocking=True))
                batches.append(backbone.forward_head(tokens, pre_logits=True).cpu().to(torch.bfloat16))
    del backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(tuple(batches))


def _adapted_behavior(
    node: BehaviorNode,
    backbone_factory: Callable[[], nn.Module],
    prepared_root: Path,
    rows: Sequence[ImageRecord],
    transform: object,
    rank: int,
    alpha: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> dict[str, Tensor]:
    model = AdapterVisionModel(
        backbone_factory(),
        node.classifier.class_ids,
        rank,
        alpha,
        0.0,
        0,
        node.classifier,
    ).to(device).eval()
    load_adapter_factors(model, node.adapter)
    prelogits, raw_scores = [], []
    from tqdm.auto import tqdm

    loader = _loader(prepared_root, rows, transform, batch_size, num_workers, device)
    with torch.inference_mode():
        for images, _labels, _ids in tqdm(
            loader,
            desc=f"node {node.node_hash[:8]} behavior ({len(rows):,})",
            leave=False,
            unit="batch",
        ):
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                features = model.features(images.to(device, non_blocking=True))
                raw = model.classifier(features)
            prelogits.append(features.cpu().to(torch.bfloat16))
            raw_scores.append(raw.float().cpu())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"prelogits": torch.cat(tuple(prelogits)), "raw_scores": torch.cat(tuple(raw_scores))}


def build_frontier_tensors(
    nodes: Sequence[BehaviorNode],
    slot_indices: Sequence[int],
    maximum_slots: int,
    prepared_root: str | Path,
    rows: Sequence[ImageRecord],
    transform: object,
    transform_hash: str,
    model_hash: str,
    backbone_factory: Callable[[], nn.Module],
    cache_root: str | Path,
    rank: int,
    alpha: int,
    cosine_scale: float,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> FrontierTensors:
    """Load or compute every frozen behavior needed by a task-free frontier."""
    if (
        not nodes
        or len(nodes) != len(slot_indices)
        or len(set(slot_indices)) != len(slot_indices)
        or any(not 0 <= slot < maximum_slots for slot in slot_indices)
        or cosine_scale <= 0.0
        or not rows
        or len({row.image_id for row in rows}) != len(rows)
        or any(row.split != "train" and row.split != "test" for row in rows)
    ):
        raise ValueError("frontier observation request is invalid")
    all_classes = tuple(class_id for node in nodes for class_id in node.classifier.class_ids)
    if len(all_classes) != len(set(all_classes)):
        raise ValueError("live node classifier ownership overlaps")
    image_ids = tuple(row.image_id for row in rows)
    cache = TensorCache(cache_root, "imagenetr50-integrator-node-behavior-v1")
    base_cache = TensorCache(cache_root, "imagenetr50-integrator-frozen-prelogit-v1")
    rows_by_id = {row.image_id: row for row in rows}
    base_tensors, base_hits, base_misses = base_cache.get_or_compute_rows(
        {
            "model_hash": model_hash,
            "transform_hash": transform_hash,
        },
        image_ids,
        lambda missing: {
            "prelogits": _frozen_prelogits(
                backbone_factory,
                Path(prepared_root),
                tuple(rows_by_id[image_id] for image_id in missing),
                transform,
                batch_size,
                num_workers,
                device,
            )
        },
    )
    base_prelogits = base_tensors["prelogits"].float()
    normalized = torch.zeros(
        (len(rows), maximum_slots, PRELOGIT_DIM), dtype=torch.bfloat16
    )
    raw = torch.zeros((len(rows), maximum_slots, CLASS_COUNT), dtype=torch.float32)
    cosine = torch.zeros_like(raw)
    local = torch.zeros_like(raw)
    base = torch.zeros_like(raw)
    ownership = torch.zeros((maximum_slots, CLASS_COUNT), dtype=torch.bool)
    active = torch.zeros(maximum_slots, dtype=torch.bool)
    hits = base_hits
    misses = base_misses
    node_forwards = 0
    for node, slot in zip(nodes, slot_indices, strict=True):
        values = {
            "alpha": alpha,
            "class_ids": list(node.classifier.class_ids),
            "model_hash": model_hash,
            "node_hash": node.node_hash,
            "rank": rank,
            "transform_hash": transform_hash,
        }
        tensors, node_hits, node_misses = cache.get_or_compute_rows(
            values,
            image_ids,
            lambda missing, node=node: _adapted_behavior(
                node,
                backbone_factory,
                Path(prepared_root),
                tuple(rows_by_id[image_id] for image_id in missing),
                transform,
                rank,
                alpha,
                batch_size,
                num_workers,
                device,
            ),
        )
        class_ids = torch.tensor(node.classifier.class_ids, dtype=torch.long)
        normalized[:, slot] = F.layer_norm(tensors["prelogits"].float(), (PRELOGIT_DIM,)).to(
            torch.bfloat16
        )
        raw[:, slot, class_ids] = tensors["raw_scores"].float()
        cosine[:, slot, class_ids] = cosine_scale * (
            F.normalize(tensors["prelogits"].float(), dim=1)
            @ F.normalize(node.classifier.weight.float(), dim=1).T
        )
        local[:, slot, class_ids] = F.log_softmax(tensors["raw_scores"].float(), dim=1)
        base[:, slot, class_ids] = F.linear(
            base_prelogits,
            node.classifier.weight.float(),
            node.classifier.bias.float(),
        )
        ownership[slot, class_ids] = True
        active[slot] = True
        hits += node_hits
        misses += node_misses
        node_forwards += node_misses
    return FrontierTensors(
        image_ids,
        torch.tensor([row.remapped_class_index for row in rows], dtype=torch.int64),
        normalized,
        raw,
        cosine,
        local,
        base,
        ownership,
        active,
        ownership.any(dim=0),
        hits,
        misses,
        base_misses,
        node_forwards,
    )


def accuracy(predictions: Tensor, labels: Tensor) -> float:
    """Return top-one accuracy in percentage points."""
    if predictions.shape != labels.shape or not len(labels):
        raise ValueError("accuracy requires aligned nonempty predictions and labels")
    return 100.0 * float((predictions == labels).float().mean().item())


__all__ = [
    "BehaviorNode",
    "FrontierTensors",
    "accuracy",
    "build_frontier_tensors",
]

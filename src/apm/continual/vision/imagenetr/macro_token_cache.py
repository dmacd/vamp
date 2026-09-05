"""Bounded stage-local cache for full node-adapted ViT token sequences."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
import os
import shutil
import tempfile
import time

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from apm.continual.artifacts import (
    ChainedJsonlLedger,
    file_sha256,
    fsync_directory,
    load_canonical_json,
    publish_immutable_json,
    record_sha256,
)
from apm.continual.vision.imagenetr.data import ImageRecord, ManifestDataset
from apm.continual.vision.imagenetr.integrator_observations import BehaviorNode
from apm.continual.vision.imagenetr.lora import load_adapter_factors
from apm.continual.vision.imagenetr.macro_token_model import (
    CLASS_COUNT,
    MAXIMUM_SLOTS,
    TOKEN_COUNT,
    TOKEN_DIMENSION,
    MacroTokenInputs,
    MacroTokenSupervision,
    behavior_meta_features,
    class_owner_targets,
)
from apm.continual.vision.imagenetr.model import AdapterVisionModel


CACHE_NAMESPACE = "imagenetr50-macro-token-stage-cache-v1"


@dataclass(frozen=True, slots=True)
class MacroTokenShard:
    """Aligned image rows and one immutable tensor shard per active node."""

    rows: tuple[ImageRecord, ...]
    node_paths: tuple[Path, ...]

    def __post_init__(self) -> None:
        if (
            not self.rows
            or len({row.image_id for row in self.rows}) != len(self.rows)
            or not self.node_paths
            or any(not path.is_file() for path in self.node_paths)
        ):
            raise ValueError("macro-token shard is empty, duplicated, or incomplete")


@dataclass(frozen=True, slots=True)
class MacroTokenPopulation:
    """Streaming view over one exact stage, frontier, and image population."""

    identity: str
    frontier_hash: str
    partition: str
    nodes: tuple[BehaviorNode, ...]
    slot_indices: tuple[int, ...]
    shards: tuple[MacroTokenShard, ...]
    cache_hits: int
    cache_misses: int
    node_example_forwards: int
    cache_bytes: int

    def __post_init__(self) -> None:
        image_ids = tuple(row.image_id for shard in self.shards for row in shard.rows)
        if (
            len(self.identity) != 64
            or len(self.frontier_hash) != 64
            or self.partition not in {"fit", "validation", "all_train", "test"}
            or not self.nodes
            or len(self.nodes) != len(self.slot_indices)
            or tuple(sorted(self.slot_indices)) != self.slot_indices
            or any(not 0 <= slot < MAXIMUM_SLOTS for slot in self.slot_indices)
            or len(set(image_ids)) != len(image_ids)
            or any(len(shard.node_paths) != len(self.nodes) for shard in self.shards)
            or min(self.cache_hits, self.cache_misses, self.node_example_forwards, self.cache_bytes) < 0
        ):
            raise ValueError("macro-token population is malformed")

    @property
    def rows(self) -> tuple[ImageRecord, ...]:
        """Return the stable population row order."""
        return tuple(row for shard in self.shards for row in shard.rows)

    @property
    def image_ids(self) -> tuple[str, ...]:
        """Return exact image identities without loading tensor data."""
        return tuple(row.image_id for row in self.rows)

    def ordered_shards(
        self, seed: int, epoch: int, shuffle: bool
    ) -> tuple[MacroTokenShard, ...]:
        """Return a deterministic block-shuffled epoch schedule."""
        if not shuffle:
            return self.shards
        generator = torch.Generator().manual_seed(
            int(
                record_sha256(
                    {
                        "epoch": epoch,
                        "population": self.identity,
                        "seed": seed,
                        "schema_version": "imagenetr50-macro-token-shuffle-v1",
                    }
                )[:15],
                16,
            )
        )
        order = torch.randperm(len(self.shards), generator=generator).tolist()
        return tuple(self.shards[index] for index in order)

    def load(self, shard: MacroTokenShard, shuffle_seed: int | None = None) -> MacroTokenSupervision:
        """Load one bounded shard and construct label-free inputs plus supervision."""
        try:
            from safetensors.torch import load_file
        except ImportError as error:  # pragma: no cover - vision environment gate
            raise RuntimeError("safetensors is required by the vision environment") from error
        stored = tuple(load_file(path, device="cpu") for path in shard.node_paths)
        row_count = len(shard.rows)
        if any(
            tensors.get("tokens", torch.empty(0)).shape
            != (row_count, TOKEN_COUNT, TOKEN_DIMENSION)
            or tensors.get("tokens", torch.empty(0)).dtype != torch.bfloat16
            or tensors.get("raw_scores", torch.empty(0)).shape
            != (row_count, len(node.classifier.class_ids))
            for tensors, node in zip(stored, self.nodes, strict=True)
        ):
            raise ValueError("macro-token tensor shard shapes changed")
        node_tokens = torch.stack(tuple(tensors["tokens"] for tensors in stored), dim=1)
        raw_scores = torch.zeros(
            (row_count, MAXIMUM_SLOTS, CLASS_COUNT), dtype=torch.float32
        )
        ownership = torch.zeros((MAXIMUM_SLOTS, CLASS_COUNT), dtype=torch.bool)
        active = torch.zeros(MAXIMUM_SLOTS, dtype=torch.bool)
        for tensors, node, slot in zip(stored, self.nodes, self.slot_indices, strict=True):
            class_ids = torch.tensor(node.classifier.class_ids, dtype=torch.int64)
            raw_scores[:, slot, class_ids] = tensors["raw_scores"].float()
            ownership[slot, class_ids] = True
            active[slot] = True
        labels = torch.tensor(
            [row.remapped_class_index for row in shard.rows], dtype=torch.int64
        )
        inputs = MacroTokenInputs(
            node_tokens.detach(),
            torch.tensor(self.slot_indices, dtype=torch.int64),
            behavior_meta_features(raw_scores, ownership, active).detach(),
            raw_scores.detach(),
            ownership,
            active,
            ownership.any(dim=0),
        )
        supervision = MacroTokenSupervision(
            inputs, labels, class_owner_targets(labels, ownership)
        )
        if shuffle_seed is None or row_count == 1:
            return supervision
        order = torch.randperm(
            row_count, generator=torch.Generator().manual_seed(shuffle_seed)
        )
        shuffled = MacroTokenInputs(
            inputs.node_tokens[order],
            inputs.slot_indices,
            inputs.meta_features[order],
            inputs.raw_scores[order],
            inputs.ownership,
            inputs.active_slot_mask,
            inputs.seen_class_mask,
        )
        return MacroTokenSupervision(
            shuffled, supervision.labels[order], supervision.owner_targets[order]
        )


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


def _shard_core(
    population_identity: str,
    node: BehaviorNode,
    slot: int,
    rows: Sequence[ImageRecord],
    tensor_sha256: str,
) -> dict[str, object]:
    return {
        "cache_namespace": CACHE_NAMESPACE,
        "class_ids": list(node.classifier.class_ids),
        "image_ids": [row.image_id for row in rows],
        "node_hash": node.node_hash,
        "population_identity": population_identity,
        "schema_version": "imagenetr50-macro-token-cache-shard-v1",
        "slot": slot,
        "tensor_sha256": tensor_sha256,
    }


def _validate_shard(
    directory: Path,
    population_identity: str,
    node: BehaviorNode,
    slot: int,
    rows: Sequence[ImageRecord],
) -> Path:
    tensor_path = directory / "tensors.safetensors"
    manifest_path = directory / "cache.json"
    if not tensor_path.is_file() or not manifest_path.is_file():
        raise ValueError("macro-token cache contains a partial shard")
    core = _shard_core(
        population_identity, node, slot, rows, file_sha256(tensor_path)
    )
    if load_canonical_json(manifest_path) != {
        **core,
        "content_hash": record_sha256(core),
    }:
        raise ValueError("macro-token cache shard changed")
    return tensor_path


def _publish_shard(
    directory: Path,
    population_identity: str,
    node: BehaviorNode,
    slot: int,
    rows: Sequence[ImageRecord],
    tokens: Tensor,
    raw_scores: Tensor,
) -> Path:
    try:
        from safetensors.torch import save_file
    except ImportError as error:  # pragma: no cover - vision environment gate
        raise RuntimeError("safetensors is required by the vision environment") from error
    if directory.exists():
        return _validate_shard(directory, population_identity, node, slot, rows)
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{directory.name}.", dir=directory.parent))
    try:
        tensor_path = temporary / "tensors.safetensors"
        save_file(
            {
                "raw_scores": raw_scores.detach().float().cpu().contiguous(),
                "tokens": tokens.detach().to(torch.bfloat16).cpu().contiguous(),
            },
            tensor_path,
            metadata={"cache_namespace": CACHE_NAMESPACE},
        )
        with tensor_path.open("rb") as persisted:
            os.fsync(persisted.fileno())
        core = _shard_core(
            population_identity, node, slot, rows, file_sha256(tensor_path)
        )
        publish_immutable_json(
            temporary / "cache.json", {**core, "content_hash": record_sha256(core)}
        )
        os.rename(temporary, directory)
        fsync_directory(directory.parent)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return directory / "tensors.safetensors"


def _population_identity(
    protocol_hash: str,
    frontier_hash: str,
    partition: str,
    nodes: Sequence[BehaviorNode],
    slots: Sequence[int],
    rows: Sequence[ImageRecord],
    model_hash: str,
    transform_hash: str,
) -> str:
    return record_sha256(
        {
            "frontier_hash": frontier_hash,
            "image_ids": [row.image_id for row in rows],
            "model_hash": model_hash,
            "node_hashes": [node.node_hash for node in nodes],
            "normalization": "parameter_free_layer_norm_eps_1e-5_per_token",
            "partition": partition,
            "protocol_hash": protocol_hash,
            "schema_version": "imagenetr50-macro-token-population-v1",
            "slots": list(slots),
            "transform_hash": transform_hash,
        }
    )


def _directory_size(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def materialize_macro_population(
    *,
    protocol_hash: str,
    frontier_hash: str,
    partition: str,
    nodes: Sequence[BehaviorNode],
    slot_indices: Sequence[int],
    rows: Sequence[ImageRecord],
    prepared_root: Path,
    transform: object,
    transform_hash: str,
    model_hash: str,
    backbone_factory: Callable[[], nn.Module],
    scratch_root: Path,
    request_ledger: ChainedJsonlLedger,
    rank: int,
    alpha: int,
    shard_size: int,
    batch_size: int,
    num_workers: int,
    cache_limit_bytes: int,
    device: torch.device,
) -> MacroTokenPopulation:
    """Materialize missing node-token shards and return a bounded streaming view."""
    started = time.monotonic()
    ordered_nodes_and_slots = tuple(sorted(zip(nodes, slot_indices, strict=True), key=lambda pair: pair[1]))
    ordered_nodes = tuple(pair[0] for pair in ordered_nodes_and_slots)
    ordered_slots = tuple(int(pair[1]) for pair in ordered_nodes_and_slots)
    stable_rows = tuple(rows)
    if (
        partition not in {"fit", "validation", "all_train", "test"}
        or not ordered_nodes
        or len(ordered_nodes) != len(ordered_slots)
        or len(set(ordered_slots)) != len(ordered_slots)
        or any(not 0 <= value < MAXIMUM_SLOTS for value in ordered_slots)
        or not stable_rows
        or len({row.image_id for row in stable_rows}) != len(stable_rows)
        or shard_size != batch_size
        or cache_limit_bytes < 1
    ):
        raise ValueError("macro-token population request is invalid")
    all_classes = tuple(
        class_id for node in ordered_nodes for class_id in node.classifier.class_ids
    )
    if len(all_classes) != len(set(all_classes)):
        raise ValueError("frontier node class ownership overlaps")
    identity = _population_identity(
        protocol_hash,
        frontier_hash,
        partition,
        ordered_nodes,
        ordered_slots,
        stable_rows,
        model_hash,
        transform_hash,
    )
    population_root = scratch_root / identity
    row_groups = tuple(
        stable_rows[offset : offset + shard_size]
        for offset in range(0, len(stable_rows), shard_size)
    )
    tensor_paths: dict[tuple[int, int], Path] = {}
    cache_hits = cache_misses = 0
    from tqdm.auto import tqdm

    for node_index, (node, slot) in enumerate(
        tqdm(ordered_nodes_and_slots, desc=f"{partition} node token caches", unit="node")
    ):
        directories = tuple(
            population_root
            / "nodes"
            / node.node_hash
            / record_sha256(
                {
                    "image_ids": [row.image_id for row in group],
                    "node_hash": node.node_hash,
                    "population": identity,
                    "schema_version": "imagenetr50-macro-token-shard-key-v1",
                }
            )
            for group in row_groups
        )
        missing_indices = tuple(
            index for index, directory in enumerate(directories) if not directory.is_dir()
        )
        for index, directory in enumerate(directories):
            if index not in missing_indices:
                tensor_paths[(index, node_index)] = _validate_shard(
                    directory, identity, node, slot, row_groups[index]
                )
                cache_hits += len(row_groups[index])
        if not missing_indices:
            continue
        missing_rows = tuple(
            row for index in missing_indices for row in row_groups[index]
        )
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
        loader = _loader(
            prepared_root, missing_rows, transform, batch_size, num_workers, device
        )
        try:
            for group_index, batch in zip(missing_indices, loader, strict=True):
                images, _labels, image_ids = batch
                group = row_groups[group_index]
                if tuple(image_ids) != tuple(row.image_id for row in group):
                    raise RuntimeError("image loader changed macro-token shard order")
                with torch.inference_mode(), torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    tokens = model.token_sequence(images.to(device, non_blocking=True))
                    features = model.backbone.forward_head(tokens, pre_logits=True)
                    raw_scores = model.classifier(features)
                    normalized = F.layer_norm(
                        tokens.float(), (TOKEN_DIMENSION,), eps=1e-5
                    ).to(torch.bfloat16)
                tensor_paths[(group_index, node_index)] = _publish_shard(
                    directories[group_index],
                    identity,
                    node,
                    slot,
                    group,
                    normalized,
                    raw_scores,
                )
                cache_misses += len(group)
                if _directory_size(scratch_root) > cache_limit_bytes:
                    raise RuntimeError("macro-token stage cache exceeded its 64-GiB limit")
        finally:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    shards = tuple(
        MacroTokenShard(
            group,
            tuple(
                tensor_paths[(group_index, node_index)]
                for node_index in range(len(ordered_nodes))
            ),
        )
        for group_index, group in enumerate(row_groups)
    )
    cache_bytes = _directory_size(population_root)
    request_ledger.append(
        {
            "cache_bytes": cache_bytes,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "elapsed_seconds": time.monotonic() - started,
            "examples": len(stable_rows),
            "frontier_hash": frontier_hash,
            "image_ids_hash": record_sha256([row.image_id for row in stable_rows]),
            "node_example_forwards": cache_misses,
            "node_hashes": [node.node_hash for node in ordered_nodes],
            "partition": partition,
            "population_identity": identity,
            "slots": list(ordered_slots),
            "splits": sorted({row.split for row in stable_rows}),
        }
    )
    return MacroTokenPopulation(
        identity,
        frontier_hash,
        partition,
        ordered_nodes,
        ordered_slots,
        shards,
        cache_hits,
        cache_misses,
        cache_misses,
        cache_bytes,
    )


def clear_macro_population(population: MacroTokenPopulation, scratch_root: Path) -> None:
    """Delete only the completed population's reproducible scratch directory."""
    clear_macro_population_identity(population.identity, scratch_root)


def clear_macro_population_identity(identity: str, scratch_root: Path) -> None:
    """Delete one exact completed population by its SHA-256 identity."""
    target = scratch_root / identity
    if (
        len(identity) != 64
        or any(character not in "0123456789abcdef" for character in identity)
        or target.parent != scratch_root
        or target.name != identity
    ):
        raise ValueError("refusing to clear an unresolved macro-token cache path")
    if target.is_dir():
        shutil.rmtree(target)


__all__ = [
    "CACHE_NAMESPACE",
    "MacroTokenPopulation",
    "MacroTokenShard",
    "clear_macro_population",
    "clear_macro_population_identity",
    "materialize_macro_population",
]

"""Deterministic proxy/repair reservoirs and immutable tensor caches."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from collections.abc import Callable, Mapping, Sequence
import json
import math
import os
import shutil
import tempfile

import torch
from torch import Tensor

from apm.continual.artifacts import (
    file_sha256,
    fsync_directory,
    publish_immutable_json,
    record_sha256,
)
from apm.continual.vision.imagenetr.data import (
    DatasetManifest,
    ImageRecord,
    deterministic_bottom_k,
)


@dataclass(frozen=True, slots=True)
class Reservoir:
    """A deterministic training-only image-ID reservoir and its policy identity."""

    image_ids: tuple[str, ...]
    namespace: str
    represented_source_count: int

    def __post_init__(self) -> None:
        if (
            not self.namespace
            or self.represented_source_count < len(self.image_ids)
            or len(set(self.image_ids)) != len(self.image_ids)
            or any(len(value) != 64 for value in self.image_ids)
        ):
            raise ValueError("invalid deterministic reservoir")

    @property
    def content_hash(self) -> str:
        """Return the exact reservoir membership identity."""
        return record_sha256(
            {
                "image_ids": list(self.image_ids),
                "namespace": self.namespace,
                "represented_source_count": self.represented_source_count,
                "schema_version": "imagenetr50-reservoir-v1",
            }
        )


def node_reservoir(
    manifest: DatasetManifest,
    represented_tasks: Sequence[int],
    count: int,
    namespace: str,
) -> Reservoir:
    """Select bottom-K priorities from all represented training identities."""
    rows = manifest.select("train", represented_tasks)
    return Reservoir(
        deterministic_bottom_k(rows, min(count, len(rows)), namespace),
        namespace,
        len(rows),
    )


def repair_reservoir(
    manifest: DatasetManifest,
    represented_tasks: Sequence[int],
    fraction: float,
) -> Reservoir:
    """Select exactly ceil(f*N) historical training images deterministically."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("repair fraction must lie in [0, 1]")
    rows = manifest.select("train", represented_tasks)
    count = math.ceil(fraction * len(rows))
    namespace = f"imagenetr50-repair-v1:{fraction:.12g}"
    return Reservoir(deterministic_bottom_k(rows, count, namespace), namespace, len(rows))


def require_training_only(
    image_ids: Sequence[str],
    manifest: DatasetManifest,
    purpose: str,
) -> None:
    """Reject any proxy, repair, calibration, or training membership outside train."""
    train_ids = {row.image_id for row in manifest.images if row.split == "train"}
    supplied = set(image_ids)
    if not purpose or len(supplied) != len(image_ids) or not supplied <= train_ids:
        raise ValueError(f"{purpose or 'input'} contains non-training or duplicate identities")


@dataclass(frozen=True, slots=True)
class CachedScores:
    """Raw and cosine node scores over one exact stable image sequence."""

    image_ids: tuple[str, ...]
    class_ids: tuple[int, ...]
    raw_logits: Tensor
    cosine_scores: Tensor

    def __post_init__(self) -> None:
        expected = (len(self.image_ids), len(self.class_ids))
        if (
            len(set(self.image_ids)) != len(self.image_ids)
            or self.class_ids != tuple(sorted(set(self.class_ids)))
            or tuple(self.raw_logits.shape) != expected
            or tuple(self.cosine_scores.shape) != expected
            or not torch.isfinite(self.raw_logits).all()
            or not torch.isfinite(self.cosine_scores).all()
        ):
            raise ValueError("invalid cached evaluation scores")


class TensorCache:
    """Content-addressed safetensors cache with exact semantic key manifests."""

    def __init__(self, root: str | Path, namespace: str) -> None:
        if not namespace:
            raise ValueError("tensor cache namespace must be nonempty")
        self.root = Path(root)
        self.namespace = namespace

    def key(self, values: Mapping[str, object]) -> str:
        """Return the exact semantic cache identity."""
        return record_sha256(
            {
                **dict(values),
                "cache_namespace": self.namespace,
                "schema_version": "imagenetr50-tensor-cache-key-v1",
            }
        )

    def get_or_compute(
        self,
        values: Mapping[str, object],
        compute: Callable[[], Mapping[str, Tensor]],
    ) -> tuple[dict[str, Tensor], bool]:
        """Load a validated entry or compute and immutably publish it once."""
        try:
            from safetensors.torch import load_file, save_file
        except ImportError as error:  # pragma: no cover - vision environment gate
            raise RuntimeError("safetensors is required by the vision environment") from error
        identity = self.key(values)
        directory = self.root / identity
        tensor_path, manifest_path = directory / "tensors.safetensors", directory / "cache.json"
        if tensor_path.is_file() and manifest_path.is_file():
            record = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected = {
                "cache_key": identity,
                "schema_version": "imagenetr50-tensor-cache-entry-v1",
                "semantic_values": dict(values),
                "tensor_sha256": file_sha256(tensor_path),
            }
            if record != {**expected, "content_hash": record_sha256(expected)}:
                raise ValueError("tensor cache entry changed")
            return dict(load_file(tensor_path, device="cpu")), True
        if directory.exists():
            raise ValueError("partial tensor cache entry exists")
        tensors = {
            key: value.detach().to(device="cpu").contiguous()
            for key, value in sorted(compute().items())
        }
        if not tensors:
            raise ValueError("cannot cache an empty tensor mapping")
        directory.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{identity}.", dir=directory.parent)
        )
        try:
            temporary_tensors = temporary / tensor_path.name
            save_file(
                tensors,
                temporary_tensors,
                metadata={"cache_key": identity, "schema_version": self.namespace},
            )
            with temporary_tensors.open("rb") as persisted:
                os.fsync(persisted.fileno())
            core: dict[str, object] = {
                "cache_key": identity,
                "schema_version": "imagenetr50-tensor-cache-entry-v1",
                "semantic_values": dict(values),
                "tensor_sha256": file_sha256(temporary_tensors),
            }
            publish_immutable_json(
                temporary / manifest_path.name,
                {**core, "content_hash": record_sha256(core)},
            )
            os.rename(temporary, directory)
            fsync_directory(directory.parent)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return tensors, False

    def get_or_compute_rows(
        self,
        values: Mapping[str, object],
        image_ids: Sequence[str],
        compute: Callable[[tuple[str, ...]], Mapping[str, Tensor]],
    ) -> tuple[dict[str, Tensor], int, int]:
        """Load cached rows, compute only missing identities, and preserve request order.

        Entries are immutable shards below one semantic model/transform namespace.  The
        shard manifests carry every image identity explicitly, so a later request with
        different stage membership can reuse individual rows without treating a future
        or test image as observed before it is actually requested.
        """
        try:
            from safetensors.torch import load_file, save_file
        except ImportError as error:  # pragma: no cover - vision environment gate
            raise RuntimeError("safetensors is required by the vision environment") from error
        requested = tuple(image_ids)
        if (
            not requested
            or len(set(requested)) != len(requested)
            or any(not image_id for image_id in requested)
        ):
            raise ValueError("row cache requires unique nonempty image identities")
        semantic_key = self.key(values)
        root = self.root / "row_shards" / semantic_key
        locations: dict[str, tuple[Path, int]] = {}
        if root.is_dir():
            for shard in sorted(path for path in root.iterdir() if not path.name.startswith(".")):
                manifest_path = shard / "cache.json"
                tensor_path = shard / "tensors.safetensors"
                if not shard.is_dir() or not manifest_path.is_file() or not tensor_path.is_file():
                    raise ValueError("row tensor cache contains a partial shard")
                record = json.loads(manifest_path.read_text(encoding="utf-8"))
                core = {
                    "cache_key": semantic_key,
                    "image_ids": record.get("image_ids"),
                    "schema_version": "imagenetr50-row-tensor-cache-entry-v1",
                    "semantic_values": dict(values),
                    "tensor_sha256": file_sha256(tensor_path),
                }
                if record != {**core, "content_hash": record_sha256(core)}:
                    raise ValueError("row tensor cache shard changed")
                stored_ids = tuple(str(value) for value in record["image_ids"])
                if not stored_ids or len(set(stored_ids)) != len(stored_ids):
                    raise ValueError("row tensor cache shard identities are malformed")
                for row_index, image_id in enumerate(stored_ids):
                    if image_id in locations:
                        raise ValueError("row tensor cache contains a duplicate image identity")
                    locations[image_id] = (tensor_path, row_index)

        missing = tuple(image_id for image_id in requested if image_id not in locations)
        if missing:
            computed = {
                key: value.detach().to(device="cpu").contiguous()
                for key, value in sorted(compute(missing).items())
            }
            if (
                not computed
                or any(tensor.ndim < 1 or len(tensor) != len(missing) for tensor in computed.values())
            ):
                raise ValueError("row cache computation did not return one row per identity")
            shard_key = record_sha256(
                {
                    "cache_key": semantic_key,
                    "image_ids": list(missing),
                    "schema_version": "imagenetr50-row-tensor-cache-shard-key-v1",
                }
            )
            directory = root / shard_key
            if directory.exists():
                raise ValueError("row tensor cache shard identity already exists unexpectedly")
            root.mkdir(parents=True, exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix=f".{shard_key}.", dir=root))
            try:
                tensor_path = temporary / "tensors.safetensors"
                save_file(
                    computed,
                    tensor_path,
                    metadata={
                        "cache_key": semantic_key,
                        "schema_version": self.namespace,
                    },
                )
                with tensor_path.open("rb") as persisted:
                    os.fsync(persisted.fileno())
                core = {
                    "cache_key": semantic_key,
                    "image_ids": list(missing),
                    "schema_version": "imagenetr50-row-tensor-cache-entry-v1",
                    "semantic_values": dict(values),
                    "tensor_sha256": file_sha256(tensor_path),
                }
                publish_immutable_json(
                    temporary / "cache.json",
                    {**core, "content_hash": record_sha256(core)},
                )
                os.rename(temporary, directory)
                fsync_directory(root)
            except BaseException:
                shutil.rmtree(temporary, ignore_errors=True)
                raise
            final_tensor_path = directory / "tensors.safetensors"
            for row_index, image_id in enumerate(missing):
                locations[image_id] = (final_tensor_path, row_index)

        requested_paths = tuple(sorted({locations[image_id][0] for image_id in requested}))
        loaded = {path: dict(load_file(path, device="cpu")) for path in requested_paths}
        tensor_keys = tuple(sorted(loaded[requested_paths[0]]))
        if not tensor_keys or any(set(shard) != set(tensor_keys) for shard in loaded.values()):
            raise ValueError("row tensor cache shards expose inconsistent tensor names")
        assembled: dict[str, Tensor] = {}
        for tensor_name in tensor_keys:
            example = loaded[requested_paths[0]][tensor_name]
            output = torch.empty((len(requested), *example.shape[1:]), dtype=example.dtype)
            grouped: dict[Path, list[tuple[int, int]]] = {}
            for output_index, image_id in enumerate(requested):
                path, source_index = locations[image_id]
                grouped.setdefault(path, []).append((output_index, source_index))
            for path, indices in grouped.items():
                source = loaded[path][tensor_name]
                if source.shape[1:] != example.shape[1:] or source.dtype != example.dtype:
                    raise ValueError("row tensor cache shards expose inconsistent tensor shapes")
                output_indices = torch.tensor([value[0] for value in indices], dtype=torch.int64)
                source_indices = torch.tensor([value[1] for value in indices], dtype=torch.int64)
                output[output_indices] = source[source_indices]
            assembled[tensor_name] = output
        return assembled, len(requested) - len(missing), len(missing)


def score_cache_values(
    model_hash: str,
    node_hash: str,
    image_ids: Sequence[str],
    transform_hash: str,
) -> dict[str, object]:
    """Build a cache key that binds every node-image-transform dependency."""
    return {
        "image_ids_sha256": record_sha256(list(image_ids)),
        "model_hash": model_hash,
        "node_hash": node_hash,
        "transform_hash": transform_hash,
    }


__all__ = [
    "CachedScores",
    "Reservoir",
    "TensorCache",
    "node_reservoir",
    "repair_reservoir",
    "require_training_only",
    "score_cache_values",
]

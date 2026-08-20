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

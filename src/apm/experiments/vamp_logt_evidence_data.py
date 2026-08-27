"""Authenticated raw-image and frozen-suffix tables for LogT evidence routing."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from apm.continual.artifacts import (
    file_sha256,
    load_canonical_json,
    publish_immutable_json,
    record_sha256,
)
from apm.continual.logt_evidence_bank import TemporalNode
from apm.continual.nce_tre_evidence import quantize_raw_images
from apm.continual.top_two_adapter import TopTwoBaseState, top_two_base_state
from apm.continual.vision.imagenetr.checkpoints import atomic_torch_save
from apm.data.mnist.loader import load_mnist
from apm.experiments.vamp_af_data import AddressCNN
from apm.experiments.vamp_logt_evidence_config import VampLogTEvidenceConfig


@dataclass(frozen=True, slots=True)
class RawFeatureTable:
    """Aligned raw pixels, frozen trunk features, targets, and diagnostic provenance."""

    raw_images: Tensor
    trunk_features: Tensor
    labels: Tensor
    context_ids: Tensor
    stream_steps: Tensor

    def __post_init__(self) -> None:
        rows = self.raw_images.shape[0]
        if (
            self.raw_images.dtype != torch.uint8
            or self.raw_images.shape != (rows, 1, 28, 28)
            or self.trunk_features.ndim != 2
            or self.trunk_features.shape[0] != rows
            or self.trunk_features.dtype != torch.float32
            or self.labels.shape != (rows,)
            or self.labels.dtype != torch.int64
            or self.context_ids.shape != (rows,)
            or self.context_ids.dtype != torch.int64
            or self.stream_steps.shape != (rows,)
            or self.stream_steps.dtype != torch.int64
            or not torch.isfinite(self.trunk_features).all()
        ):
            raise ValueError("raw evidence and frozen adapter tables are misaligned")

    def select(self, indices: Tensor) -> "RawFeatureTable":
        """Return an aligned row selection with new contiguous stream steps."""
        if indices.dtype != torch.int64 or indices.ndim != 1:
            raise ValueError("table selections require a vector of int64 row IDs")
        return RawFeatureTable(
            self.raw_images[indices],
            self.trunk_features[indices],
            self.labels[indices],
            self.context_ids[indices],
            torch.arange(len(indices), dtype=torch.int64),
        )


@dataclass(frozen=True, slots=True)
class AuthenticatedBaseline:
    """Read-only sealed VAMP-AF dependency used by the new experiment."""

    model: AddressCNN
    top_two_base: TopTwoBaseState
    train: RawFeatureTable
    test: RawFeatureTable
    reference_raw_images: Tensor | None
    reference_sha256: str | None
    source_train_indices: tuple[int, ...]
    protocol_sha256: str
    summary: dict[str, object]

    def __post_init__(self) -> None:
        if (self.reference_raw_images is None) != (self.reference_sha256 is None):
            raise ValueError("reference images and their identity must be present together")
        if self.reference_raw_images is not None and (
            self.reference_raw_images.dtype != torch.uint8
            or self.reference_raw_images.ndim != 4
            or self.reference_raw_images.shape[1:] != (1, 28, 28)
            or self.reference_raw_images.device.type != "cpu"
            or len(self.reference_raw_images) < 2
            or len(self.reference_sha256 or "") != 64
        ):
            raise ValueError("authenticated base reference images are invalid")


@dataclass(frozen=True, slots=True)
class NodeHoldout:
    """Balanced latent-source evaluation sample generated only from MNIST test rows."""

    table: RawFeatureTable
    source_node_ids: tuple[str, ...]
    equivalent_source_keys: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if (
            len(self.source_node_ids) != len(self.table.labels)
            or len(self.equivalent_source_keys) != len(self.table.labels)
        ):
            raise ValueError("node holdout provenance does not match its rows")


def resolved_device(name: str) -> torch.device:
    """Resolve the configured device and reject unavailable explicit CUDA."""
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not visible")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def authenticate_and_load_baseline(
    config: VampLogTEvidenceConfig,
    run_root: Path,
) -> AuthenticatedBaseline:
    """Authenticate the sealed VAMP-AF run, then construct aligned raw-image tables."""
    baseline_root = config.baseline_run_root
    paths = {
        "protocol": baseline_root / "protocol.json",
        "summary": baseline_root / "summary.json",
        "checkpoint": baseline_root / "base" / "model.pt",
        "base_manifest": baseline_root / "base" / "manifest.json",
        "features": baseline_root / "cache" / "features.pt",
        "feature_manifest": baseline_root / "cache" / "features.json",
    }
    missing = tuple(name for name, path in paths.items() if not path.is_file())
    if missing:
        raise FileNotFoundError(f"sealed VAMP-AF dependency is incomplete: {missing}")
    protocol = load_canonical_json(paths["protocol"])
    summary = load_canonical_json(paths["summary"])
    base_manifest = load_canonical_json(paths["base_manifest"])
    feature_manifest = load_canonical_json(paths["feature_manifest"])
    if (
        protocol.get("config_hash") != config.baseline.run_id
        or protocol.get("base_checkpoint_sha256") != config.baseline.base_checkpoint_sha256
        or base_manifest.get("checkpoint_sha256") != config.baseline.base_checkpoint_sha256
        or feature_manifest.get("cache_sha256") != config.baseline.feature_cache_sha256
        or feature_manifest.get("base_checkpoint_sha256") != config.baseline.base_checkpoint_sha256
        or file_sha256(paths["checkpoint"]) != config.baseline.base_checkpoint_sha256
        or file_sha256(paths["features"]) != config.baseline.feature_cache_sha256
        or summary.get("main_mean_accuracy") != config.baseline.main_mean_accuracy
        or summary.get("main_mean_oracle_leaf_accuracy")
        != config.baseline.expected_main_mean_oracle_leaf_accuracy
    ):
        raise ValueError("sealed VAMP-AF dependency differs from the pinned baseline")
    baseline_config = protocol.get("config")
    if not isinstance(baseline_config, dict) or (
        tuple(float(value) for value in baseline_config["data"]["rotations_deg"])
        != config.stream.rotations_deg
        or tuple(int(value) for value in baseline_config["data"]["label_shifts"])
        != config.stream.label_shifts
        or int(baseline_config["data"]["main_examples_per_context"])
        != config.stream.examples_per_context
        or baseline_config["data"]["interpolation"] != config.stream.interpolation
    ):
        raise ValueError("new stream differs from the authenticated VAMP-AF stream")
    data_hashes = protocol.get("data_sha256")
    if not isinstance(data_hashes, dict) or any(
        file_sha256(config.data_root / name) != expected
        for name, expected in data_hashes.items()
    ):
        raise ValueError("local MNIST bytes differ from the sealed VAMP-AF dependency")

    model = AddressCNN()
    model.load_state_dict(torch.load(paths["checkpoint"], map_location="cpu", weights_only=True))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    suffix = top_two_base_state(
        model.embedding.weight,
        model.embedding.bias,
        model.classifier.weight,
        model.classifier.bias,
    )
    features = torch.load(paths["features"], map_location="cpu", weights_only=True)
    source_indices = tuple(int(value) for value in features["source_train_indices"].tolist())
    train_raw, test_raw = _raw_cache(config, run_root, source_indices)
    reference_raw, reference_sha = _base_reference_cache(
        config,
        run_root,
        protocol,
        file_sha256(paths["protocol"]),
    )
    train = _table_from_feature_payload(features, "train", train_raw)
    test = _table_from_feature_payload(features, "test", test_raw)
    return AuthenticatedBaseline(
        model,
        suffix,
        train,
        test,
        reference_raw,
        reference_sha,
        source_indices,
        file_sha256(paths["protocol"]),
        summary,
    )


def stream_training_table(baseline: AuthenticatedBaseline, seed: int) -> RawFeatureTable:
    """Return the exact deterministic 10,000-row blocked ordering for each context."""
    if seed < 0:
        raise ValueError("stream seeds must be nonnegative")
    context_ids = tuple(int(value) for value in torch.unique(baseline.train.context_ids).tolist())
    selected = tuple(
        int(row)
        for context_id in context_ids
        for row in np.random.default_rng(seed * 10_000 + context_id).permutation(
            torch.nonzero(baseline.train.context_ids == context_id, as_tuple=True)[0].numpy()
        )
    )
    if len(selected) != len(baseline.train.labels):
        raise ValueError("authenticated stream is not context complete")
    return baseline.train.select(torch.tensor(selected, dtype=torch.int64))


def node_context_key(node: TemporalNode, stream: RawFeatureTable) -> tuple[int, ...]:
    """Return the primitive normalized context-mixture key for source equivalence."""
    counts = torch.bincount(
        stream.context_ids[torch.tensor(node.example_ids, dtype=torch.int64)], minlength=5
    ).tolist()
    divisor = int(np.gcd.reduce(np.asarray(counts, dtype=np.int64)))
    return tuple(int(count // divisor) for count in counts) if divisor else tuple(int(count) for count in counts)


def build_node_holdout(
    nodes: tuple[TemporalNode, ...],
    stream: RawFeatureTable,
    test: RawFeatureTable,
    examples_per_node: int,
    seed: int,
) -> NodeHoldout:
    """Sample a held-out latent temporal source using each node's context mixture."""
    if not nodes or examples_per_node < 1 or seed < 0:
        raise ValueError("node holdout requires nodes, examples, and a nonnegative seed")
    generator = np.random.default_rng(seed)
    selected_rows: list[int] = []
    source_ids: list[str] = []
    source_keys: list[tuple[int, ...]] = []
    for node in nodes:
        member_ids = torch.tensor(node.example_ids, dtype=torch.int64)
        counts = torch.bincount(stream.context_ids[member_ids], minlength=5).numpy()
        contexts = generator.choice(5, size=examples_per_node, p=counts / counts.sum())
        key = node_context_key(node, stream)
        for context_id in contexts:
            candidates = torch.nonzero(test.context_ids == int(context_id), as_tuple=True)[0].numpy()
            selected_rows.append(int(generator.choice(candidates)))
            source_ids.append(node.node_id)
            source_keys.append(key)
    return NodeHoldout(
        test.select(torch.tensor(selected_rows, dtype=torch.int64)),
        tuple(source_ids),
        tuple(source_keys),
    )


def _raw_cache(
    config: VampLogTEvidenceConfig,
    run_root: Path,
    source_indices: tuple[int, ...],
) -> tuple[Tensor, Tensor]:
    cache = run_root / "cache" / "raw_uint8.pt"
    manifest = run_root / "cache" / "raw_uint8.json"
    identity = record_sha256(
        {
            "base_feature_cache_sha256": config.baseline.feature_cache_sha256,
            "interpolation": config.stream.interpolation,
            "label_shifts": list(config.stream.label_shifts),
            "rotations_deg": list(config.stream.rotations_deg),
            "source_train_indices": list(source_indices),
        }
    )
    if cache.is_file() and manifest.is_file():
        record = load_canonical_json(manifest)
        if (
            record.get("schema_version") != "vamp-logt-raw-cache-v1"
            or record.get("semantic_sha256") != identity
            or record.get("cache_sha256") != file_sha256(cache)
        ):
            raise ValueError("raw evidence cache identity changed")
        payload = torch.load(cache, map_location="cpu", weights_only=True)
        return payload["train_raw_images"], payload["test_raw_images"]

    arrays = load_mnist(root=config.data_root, allow_download=False, npz_cache_path=None)
    train = _transformed_raw_images(
        arrays.train_images[np.asarray(source_indices)], config, "raw train cache"
    )
    test = _transformed_raw_images(arrays.test_images, config, "raw test cache")
    cache.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(cache, {"train_raw_images": train, "test_raw_images": test})
    publish_immutable_json(
        manifest,
        {
            "cache_sha256": file_sha256(cache),
            "schema_version": "vamp-logt-raw-cache-v1",
            "semantic_sha256": identity,
            "test_examples": len(test),
            "train_examples": len(train),
        },
    )
    return train, test


def _base_reference_cache(
    config: VampLogTEvidenceConfig,
    run_root: Path,
    baseline_protocol: dict[str, object],
    baseline_protocol_sha256: str,
) -> tuple[Tensor | None, str | None]:
    if config.evidence.reference == "discrete_uniform_uint8":
        return None, None
    if config.evidence.reference != "frozen_base_training_images_uint8":
        raise ValueError("unsupported evidence reference distribution")
    data_hashes = baseline_protocol.get("data_sha256")
    if not isinstance(data_hashes, dict):
        raise ValueError("baseline protocol lacks authenticated MNIST identities")
    source_sha256 = str(data_hashes.get("train-images-idx3-ubyte", ""))
    identity = record_sha256(
        {
            "base_checkpoint_sha256": config.baseline.base_checkpoint_sha256,
            "baseline_protocol_sha256": baseline_protocol_sha256,
            "population": "all_60000_original_unrotated_mnist_training_images",
            "sampling": "uniform_with_replacement",
            "source_train_images_sha256": source_sha256,
        }
    )
    cache = run_root / "cache" / "base_reference_uint8.pt"
    manifest = run_root / "cache" / "base_reference_uint8.json"
    if cache.is_file() and manifest.is_file():
        record = load_canonical_json(manifest)
        if (
            record.get("schema_version") != "vamp-logt-base-reference-v1"
            or record.get("semantic_sha256") != identity
            or record.get("cache_sha256") != file_sha256(cache)
        ):
            raise ValueError("base-reference cache identity changed")
        payload = torch.load(cache, map_location="cpu", weights_only=True)
        images = payload["reference_raw_images"]
        content_sha256 = _uint8_image_tensor_sha256(images)
        if (
            images.shape != (60_000, 1, 28, 28)
            or record.get("content_sha256") != content_sha256
            or record.get("examples") != 60_000
        ):
            raise ValueError("base-reference cache content changed")
        return images, content_sha256

    arrays = load_mnist(root=config.data_root, allow_download=False, npz_cache_path=None)
    images = quantize_raw_images(torch.from_numpy(arrays.train_images))
    if images.shape != (60_000, 1, 28, 28):
        raise ValueError("frozen CNN was not trained from the expected 60,000-image population")
    content_sha256 = _uint8_image_tensor_sha256(images)
    cache.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(cache, {"reference_raw_images": images})
    publish_immutable_json(
        manifest,
        {
            "cache_sha256": file_sha256(cache),
            "content_sha256": content_sha256,
            "examples": len(images),
            "population": "all_60000_original_unrotated_mnist_training_images",
            "sampling": "uniform_with_replacement",
            "schema_version": "vamp-logt-base-reference-v1",
            "semantic_sha256": identity,
            "source_train_images_sha256": source_sha256,
        },
    )
    return images, content_sha256


def _uint8_image_tensor_sha256(images: Tensor) -> str:
    if (
        images.dtype != torch.uint8
        or images.ndim != 4
        or images.shape[1:] != (1, 28, 28)
        or images.device.type != "cpu"
    ):
        raise ValueError("reference tensor hashing requires CPU uint8 NCHW images")
    digest = sha256()
    digest.update(b"vamp-logt-reference-images-v1\0")
    digest.update(np.asarray(images.shape, dtype=np.int64).tobytes())
    digest.update(images.contiguous().numpy().tobytes())
    return digest.hexdigest()


def _transformed_raw_images(
    images: np.ndarray,
    config: VampLogTEvidenceConfig,
    description: str,
) -> Tensor:
    try:
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.functional import rotate
        from tqdm.auto import tqdm
    except ImportError as error:  # pragma: no cover - vision environment gate
        raise RuntimeError("torchvision and tqdm are required by the vision environment") from error
    source = torch.from_numpy(images).unsqueeze(1)
    rows = []
    for context_id, angle in enumerate(config.stream.rotations_deg):
        for offset in tqdm(
            range(0, len(source), 2_048),
            desc=f"{description} context {context_id}",
            disable=not config.runtime.progress,
            leave=False,
        ):
            transformed = rotate(
                source[offset : offset + 2_048],
                angle,
                interpolation=InterpolationMode.BILINEAR,
                expand=False,
                fill=0.0,
            )
            rows.append(quantize_raw_images(transformed))
    return torch.cat(rows)


def _table_from_feature_payload(
    payload: dict[str, Tensor],
    prefix: str,
    raw_images: Tensor,
) -> RawFeatureTable:
    table = RawFeatureTable(
        raw_images,
        payload[f"{prefix}_trunk_features"].to(torch.float32),
        payload[f"{prefix}_labels"].to(torch.int64),
        payload[f"{prefix}_context_ids"].to(torch.int64),
        payload[f"{prefix}_stream_steps"].to(torch.int64),
    )
    if len(table.raw_images) != len(payload[f"{prefix}_embeddings"]):
        raise ValueError("raw cache does not align with the authenticated feature cache")
    return table


__all__ = [
    "AuthenticatedBaseline",
    "NodeHoldout",
    "RawFeatureTable",
    "authenticate_and_load_baseline",
    "build_node_holdout",
    "node_context_key",
    "resolved_device",
    "stream_training_table",
]

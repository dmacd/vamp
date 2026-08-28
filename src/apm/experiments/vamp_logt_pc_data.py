"""Authenticated raw-only tables and controlled schedules for PC evidence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from apm.continual.artifacts import file_sha256, load_canonical_json
from apm.continual.logt_evidence_bank import TemporalNode
from apm.experiments.vamp_logt_pc_config import VampLogTPcConfig


CONDITIONS = ("novel_leaf", "recurrent_leaf_1_8", "identical_regime")


@dataclass(frozen=True, slots=True)
class PcRawTable:
    """Raw pixels, transformed targets, contexts, and immutable source rows."""

    raw_images: np.ndarray
    labels: np.ndarray
    context_ids: np.ndarray
    source_rows: np.ndarray

    def __post_init__(self) -> None:
        rows = len(self.raw_images)
        if (
            self.raw_images.dtype != np.uint8
            or self.raw_images.shape != (rows, 784)
            or self.labels.dtype != np.int64
            or self.labels.shape != (rows,)
            or self.context_ids.dtype != np.int64
            or self.context_ids.shape != (rows,)
            or self.source_rows.dtype != np.int64
            or self.source_rows.shape != (rows,)
            or len(set(self.source_rows.tolist())) != rows
            or np.any(self.context_ids < 0)
            or np.any(self.context_ids > 4)
        ):
            raise ValueError("raw-only PC table is malformed or misaligned")

    @property
    def images_float32(self) -> np.ndarray:
        """Return normalized pixels without exposing any frozen-CNN features."""
        return self.raw_images.astype(np.float32) / 255.0

    def select(self, indices: np.ndarray) -> "PcRawTable":
        """Return a detached row selection."""
        rows = np.asarray(indices)
        if rows.dtype != np.int64 or rows.ndim != 1 or len(rows) < 1:
            raise ValueError("PC table selections require nonempty int64 row IDs")
        return PcRawTable(
            self.raw_images[rows].copy(),
            self.labels[rows].copy(),
            self.context_ids[rows].copy(),
            self.source_rows[rows].copy(),
        )


@dataclass(frozen=True, slots=True)
class AuthenticatedPcData:
    """Sealed transformed MNIST tables with no CNN or adapter values."""

    train: PcRawTable
    test: PcRawTable
    raw_cache_sha256: str
    feature_cache_sha256: str
    source_protocol_sha256: str


@dataclass(frozen=True, slots=True)
class PcNodeHoldout:
    """Held-out rows paired with their temporal source node."""

    table: PcRawTable
    source_node_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.source_node_ids) != len(self.table.labels):
            raise ValueError("PC holdout provenance is not row aligned")


def authenticate_and_load_pc_data(config: VampLogTPcConfig) -> AuthenticatedPcData:
    """Fail closed on the sealed source, then expose only raw arrays and provenance."""
    source_protocol_path = config.source_run_root / "protocol.json"
    raw_path = config.source_run_root / "cache" / "raw_uint8.pt"
    raw_manifest_path = config.source_run_root / "cache" / "raw_uint8.json"
    baseline_protocol_path = config.baseline_run_root / "protocol.json"
    feature_path = config.baseline_run_root / "cache" / "features.pt"
    feature_manifest_path = config.baseline_run_root / "cache" / "features.json"
    paths = (
        source_protocol_path,
        raw_path,
        raw_manifest_path,
        baseline_protocol_path,
        feature_path,
        feature_manifest_path,
    )
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("sealed PC raw-data dependency is incomplete")
    source_protocol = load_canonical_json(source_protocol_path)
    raw_manifest = load_canonical_json(raw_manifest_path)
    baseline_protocol = load_canonical_json(baseline_protocol_path)
    feature_manifest = load_canonical_json(feature_manifest_path)
    if (
        source_protocol.get("config_hash") != config.source.nce_run_id
        or source_protocol.get("config", {}).get("baseline", {}).get("run_id")
        != config.source.baseline_run_id
        or baseline_protocol.get("config_hash") != config.source.baseline_run_id
        or raw_manifest.get("schema_version") != "vamp-logt-raw-cache-v1"
        or raw_manifest.get("semantic_sha256") != config.source.raw_semantic_sha256
        or raw_manifest.get("cache_sha256") != config.source.raw_cache_sha256
        or file_sha256(raw_path) != config.source.raw_cache_sha256
        or feature_manifest.get("cache_sha256") != config.source.feature_cache_sha256
        or file_sha256(feature_path) != config.source.feature_cache_sha256
    ):
        raise ValueError("sealed PC raw-data dependency changed")
    try:
        import torch
    except ImportError as error:  # pragma: no cover - exercised by environment preflight
        raise RuntimeError("PyTorch CPU loading is required for authenticated source artifacts") from error
    raw_payload = torch.load(raw_path, map_location="cpu", weights_only=True, mmap=True)
    feature_payload = torch.load(feature_path, map_location="cpu", weights_only=True, mmap=True)
    train = _raw_table(raw_payload, feature_payload, "train")
    test = _raw_table(raw_payload, feature_payload, "test")
    return AuthenticatedPcData(
        train,
        test,
        config.source.raw_cache_sha256,
        config.source.feature_cache_sha256,
        file_sha256(source_protocol_path),
    )


def condition_block_contexts(condition: str) -> tuple[int, ...]:
    """Return the exact context assigned to each of the 31 temporal blocks."""
    if condition == "novel_leaf":
        result = (0,) * 16 + (1,) * 8 + (2,) * 4 + (3,) * 2 + (4,)
    elif condition == "recurrent_leaf_1_8":
        result = (0,) * 14 + (4,) * 2 + (1,) * 8 + (2,) * 4 + (3,) * 2 + (4,)
    elif condition == "identical_regime":
        result = (4,) * 16 + (1,) * 8 + (2,) * 4 + (3,) * 2 + (4,)
    else:
        raise ValueError(f"unknown controlled PC condition: {condition}")
    if len(result) != 31:
        raise AssertionError("controlled PC schedule must contain 31 blocks")
    return result


def build_condition_stream(
    train: PcRawTable,
    condition: str,
    stream_seed: int,
    block_size: int = 250,
) -> PcRawTable:
    """Build one deterministic controlled stream without reusing training rows."""
    if stream_seed < 0 or block_size != 250:
        raise ValueError("controlled PC streams require seed >= 0 and block size 250")
    pools = {
        context: np.random.default_rng(stream_seed * 10_000 + context).permutation(
            np.flatnonzero(train.context_ids == context)
        )
        for context in range(5)
    }
    cursors = {context: 0 for context in range(5)}
    selected: list[np.ndarray] = []
    for context in condition_block_contexts(condition):
        start = cursors[context]
        end = start + block_size
        if end > len(pools[context]):
            raise ValueError("controlled stream exhausted a context without replacement")
        selected.append(pools[context][start:end])
        cursors[context] = end
    rows = np.concatenate(selected).astype(np.int64)
    if len(set(train.source_rows[rows].tolist())) != len(rows):
        raise ValueError("controlled PC stream reused a source training row")
    return train.select(rows)


def build_node_holdout(
    nodes: tuple[TemporalNode, ...],
    stream: PcRawTable,
    test: PcRawTable,
    examples_per_node: int,
    seed: int,
) -> PcNodeHoldout:
    """Build a disjoint held-out sample following each node's context mixture."""
    if not nodes or examples_per_node < 1 or seed < 0:
        raise ValueError("PC node holdout requires nodes, examples, and a nonnegative seed")
    pools = {
        context: np.random.default_rng(seed * 100_003 + context).permutation(
            np.flatnonzero(test.context_ids == context)
        )
        for context in range(5)
    }
    cursors = {context: 0 for context in range(5)}
    rows: list[int] = []
    source_ids: list[str] = []
    for node in nodes:
        counts = np.bincount(stream.context_ids[np.asarray(node.example_ids)], minlength=5)
        allocation = _proportional_counts(counts, examples_per_node)
        node_rows: list[int] = []
        for context, count in enumerate(allocation):
            start = cursors[context]
            end = start + int(count)
            node_rows.extend(int(value) for value in pools[context][start:end])
            cursors[context] = end
        permutation = np.random.default_rng(seed + node.first_block).permutation(node_rows)
        rows.extend(int(value) for value in permutation)
        source_ids.extend((node.node_id,) * examples_per_node)
    return PcNodeHoldout(test.select(np.asarray(rows, dtype=np.int64)), tuple(source_ids))


def context_holdout(test: PcRawTable, context: int, count: int, seed: int) -> PcRawTable:
    """Select deterministic held-out rows from one transformed context."""
    if context not in range(5) or count < 1 or seed < 0:
        raise ValueError("invalid focused PC holdout request")
    candidates = np.flatnonzero(test.context_ids == context)
    selected = np.random.default_rng(seed * 100_003 + context).permutation(candidates)[:count]
    if len(selected) != count:
        raise ValueError("focused PC holdout exceeds the context population")
    return test.select(selected.astype(np.int64))


def preflight_tables(
    data: AuthenticatedPcData,
    train_examples: int,
    heldout_examples: int,
    seed: int = 0,
) -> tuple[PcRawTable, PcRawTable]:
    """Return the fixed C0 train/test split used before routing is inspected."""
    train_candidates = np.flatnonzero(data.train.context_ids == 0)
    train_rows = np.random.default_rng(seed).permutation(train_candidates)[:train_examples]
    return data.train.select(train_rows.astype(np.int64)), context_holdout(
        data.test,
        0,
        heldout_examples,
        seed,
    )


def _raw_table(raw_payload: object, feature_payload: object, prefix: str) -> PcRawTable:
    raw_tensor = raw_payload[f"{prefix}_raw_images"]
    labels = feature_payload[f"{prefix}_labels"]
    contexts = feature_payload[f"{prefix}_context_ids"]
    stream_steps = feature_payload[f"{prefix}_stream_steps"]
    raw = raw_tensor.reshape(len(raw_tensor), -1).numpy().copy()
    label_array = labels.numpy().astype(np.int64, copy=True)
    context_array = contexts.numpy().astype(np.int64, copy=True)
    source_rows = stream_steps.numpy().astype(np.int64, copy=True)
    if len(raw) != len(label_array):
        raise ValueError("authenticated raw pixels and transformed labels are misaligned")
    return PcRawTable(raw, label_array, context_array, source_rows)


def _proportional_counts(counts: np.ndarray, total: int) -> np.ndarray:
    raw = np.asarray(counts, dtype=np.float64) * total / np.sum(counts)
    result = np.floor(raw).astype(np.int64)
    remainder = total - int(np.sum(result))
    if remainder:
        order = np.argsort(-(raw - result), kind="stable")
        result[order[:remainder]] += 1
    return result


__all__ = [
    "AuthenticatedPcData",
    "CONDITIONS",
    "PcNodeHoldout",
    "PcRawTable",
    "authenticate_and_load_pc_data",
    "build_condition_stream",
    "build_node_holdout",
    "condition_block_contexts",
    "context_holdout",
    "preflight_tables",
]

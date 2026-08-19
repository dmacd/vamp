"""Physically gathered LoRA edge banks for bounded per-example routing."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from apm.lm.lora import LoraEdgeBank
from apm.lm.lora_memory import PackedLoraMemory


COMPACT_EDGE_CAPACITY_BUCKETS = (4, 8, 12, 16, 20, 24)


class CompactLoraMemory(NamedTuple):
    """Per-row candidate paths and only their physically gathered edge factors."""

    edge_bank: LoraEdgeBank
    candidate_node_indices: jax.Array
    candidate_path_matrix: jax.Array
    valid_candidate_mask: jax.Array
    source_edge_indices: jax.Array
    valid_edge_mask: jax.Array


def gather_compact_lora_memory(
    packed: PackedLoraMemory,
    candidate_node_indices: jax.Array | np.ndarray,
    *,
    edge_capacity_buckets: tuple[int, ...] = COMPACT_EDGE_CAPACITY_BUCKETS,
) -> CompactLoraMemory:
    """Gather each row's insertion-ordered union of candidate-path edges."""
    candidates = _validated_candidate_indices(packed, candidate_node_indices)
    node_paths = np.asarray(packed.node_path_matrix, dtype=np.float32)
    valid_edges = np.asarray(packed.valid_edge_mask, dtype=np.bool_)
    source_rows = tuple(
        np.flatnonzero(
            np.any(node_paths[row] != 0.0, axis=0) & valid_edges
        ).astype(np.int32)
        for row in candidates
    )
    capacity = _edge_capacity_bucket(
        max((len(row) for row in source_rows), default=0),
        edge_capacity_buckets,
    )
    batch_size, candidate_count = candidates.shape
    source_indices = np.full((batch_size, capacity), -1, dtype=np.int32)
    valid_edge_mask = np.zeros((batch_size, capacity), dtype=np.bool_)
    compact_paths = np.zeros(
        (batch_size, candidate_count, capacity),
        dtype=np.float32,
    )
    for row_index, source_row in enumerate(source_rows):
        edge_count = len(source_row)
        source_indices[row_index, :edge_count] = source_row
        valid_edge_mask[row_index, :edge_count] = True
        if edge_count:
            compact_paths[row_index, :, :edge_count] = node_paths[
                candidates[row_index]
            ][:, source_row]
    safe_indices = np.maximum(source_indices, 0)
    edge_mask = jnp.asarray(valid_edge_mask, dtype=jnp.float32)

    def gather_factors(factors: jax.Array) -> jax.Array:
        gathered = jnp.asarray(factors)[jnp.asarray(safe_indices, dtype=jnp.int32)]
        trailing_mask = edge_mask.reshape(edge_mask.shape + (1,) * (gathered.ndim - 2))
        return gathered * trailing_mask

    compact = CompactLoraMemory(
        edge_bank=jax.tree_util.tree_map(gather_factors, packed.edge_bank),
        candidate_node_indices=jnp.asarray(candidates, dtype=jnp.int32),
        candidate_path_matrix=jnp.asarray(compact_paths, dtype=jnp.float32),
        valid_candidate_mask=jnp.ones(candidates.shape, dtype=jnp.bool_),
        source_edge_indices=jnp.asarray(source_indices, dtype=jnp.int32),
        valid_edge_mask=jnp.asarray(valid_edge_mask, dtype=jnp.bool_),
    )
    validate_compact_lora_memory(compact)
    return compact


def compact_node_weights_to_edge_coefficients(
    candidate_weights: jax.Array,
    compact: CompactLoraMemory,
) -> jax.Array:
    """Map per-row candidate weights to coefficients over gathered edges."""
    weights = jnp.asarray(candidate_weights, dtype=jnp.float32)
    expected = compact.candidate_node_indices.shape
    if weights.shape != expected:
        raise ValueError(f"candidate_weights must have shape {expected}")
    masked_weights = weights * compact.valid_candidate_mask
    coefficients = jnp.einsum(
        "bk,bke->be",
        masked_weights,
        compact.candidate_path_matrix,
    )
    return coefficients * compact.valid_edge_mask


def expand_compact_edge_coefficients(
    compact: CompactLoraMemory,
    compact_coefficients: jax.Array,
    dense_edge_capacity: int,
) -> jax.Array:
    """Scatter gathered coefficients into a dense edge axis for parity checks."""
    coefficients = jnp.asarray(compact_coefficients, dtype=jnp.float32)
    if coefficients.shape != compact.source_edge_indices.shape:
        raise ValueError(
            "compact_coefficients must match compact source-edge indices"
        )
    if type(dense_edge_capacity) is not int or dense_edge_capacity <= 0:
        raise ValueError("dense_edge_capacity must be a positive integer")
    safe_indices = jnp.maximum(compact.source_edge_indices, 0)
    updates = coefficients * compact.valid_edge_mask
    rows = jnp.arange(coefficients.shape[0], dtype=jnp.int32)[:, None]
    return jnp.zeros(
        (coefficients.shape[0], dense_edge_capacity),
        dtype=jnp.float32,
    ).at[rows, safe_indices].add(updates)


def validate_compact_lora_memory(compact: CompactLoraMemory) -> None:
    """Reject malformed candidates, path tensors, masks, and gathered banks."""
    if not isinstance(compact, CompactLoraMemory):
        raise TypeError("compact must be a CompactLoraMemory")
    candidates = np.asarray(compact.candidate_node_indices)
    paths = np.asarray(compact.candidate_path_matrix)
    candidate_mask = np.asarray(compact.valid_candidate_mask, dtype=np.bool_)
    sources = np.asarray(compact.source_edge_indices)
    edge_mask = np.asarray(compact.valid_edge_mask, dtype=np.bool_)
    if candidates.ndim != 2 or candidates.shape[0] == 0 or candidates.shape[1] == 0:
        raise ValueError("compact candidate nodes must have nonempty shape [batch, k]")
    if paths.shape != candidates.shape + (sources.shape[1],):
        raise ValueError("compact candidate paths must have shape [batch, k, edges]")
    if candidate_mask.shape != candidates.shape:
        raise ValueError("compact candidate mask must match candidate nodes")
    if sources.ndim != 2 or sources.shape[0] != candidates.shape[0]:
        raise ValueError("compact source-edge indices must have shape [batch, edges]")
    if edge_mask.shape != sources.shape:
        raise ValueError("compact edge mask must match source-edge indices")
    if np.any(candidate_mask & (candidates < 0)):
        raise ValueError("valid compact candidates must be nonnegative")
    if np.any(edge_mask & (sources < 0)) or np.any(~edge_mask & (sources != -1)):
        raise ValueError("compact source-edge padding must use exactly -1")
    if np.any(~np.isfinite(paths)) or np.any(paths[~candidate_mask] != 0.0):
        raise ValueError("compact paths must be finite with zero invalid candidates")
    if np.any(paths * (~edge_mask[:, None, :])):
        raise ValueError("compact paths must be zero outside gathered edges")
    if any(
        len(set(row[mask].tolist())) != int(np.sum(mask))
        for row, mask in zip(candidates, candidate_mask)
    ):
        raise ValueError("compact candidates must be unique per row")
    if any(
        not np.all(np.diff(row[mask]) > 0)
        for row, mask in zip(sources, edge_mask)
        if np.sum(mask) > 1
    ):
        raise ValueError("compact source edges must retain insertion order")
    edge_capacity = sources.shape[1]
    for block in compact.edge_bank.blocks:
        for projection in block:
            if (
                projection.left.ndim != 4
                or projection.right.ndim != 4
                or projection.left.shape[:2] != (candidates.shape[0], edge_capacity)
                or projection.right.shape[:2] != (candidates.shape[0], edge_capacity)
            ):
                raise ValueError(
                    "compact LoRA factors must have leading [batch, edges] axes"
                )


def _validated_candidate_indices(
    packed: PackedLoraMemory,
    candidate_node_indices: jax.Array | np.ndarray,
) -> np.ndarray:
    if not isinstance(packed, PackedLoraMemory):
        raise TypeError("packed must be a PackedLoraMemory")
    candidates = np.asarray(candidate_node_indices)
    valid_nodes = np.asarray(packed.valid_node_mask, dtype=np.bool_)
    if candidates.ndim != 2 or 0 in candidates.shape or candidates.dtype.kind not in "iu":
        raise ValueError("candidate_node_indices must be nonempty integers [batch, k]")
    if np.any((candidates < 0) | (candidates >= valid_nodes.shape[0])):
        raise ValueError("candidate node index is outside dense node capacity")
    if np.any(~valid_nodes[candidates]):
        raise ValueError("candidate node index identifies an invalid dense node")
    if any(len(set(row.tolist())) != len(row) for row in candidates):
        raise ValueError("candidate node indices must be unique per row")
    return candidates.astype(np.int32, copy=False)


def _edge_capacity_bucket(
    required: int,
    buckets: tuple[int, ...],
) -> int:
    if (
        type(required) is not int
        or required < 0
        or not buckets
        or any(type(value) is not int or value <= 0 for value in buckets)
        or tuple(sorted(set(buckets))) != buckets
    ):
        raise ValueError("compact edge-capacity buckets must be increasing positives")
    matches = tuple(value for value in buckets if value >= required)
    if not matches:
        raise ValueError(
            f"no compact edge-capacity bucket can hold {required} gathered edges"
        )
    return matches[0]


__all__ = [
    "COMPACT_EDGE_CAPACITY_BUCKETS",
    "CompactLoraMemory",
    "compact_node_weights_to_edge_coefficients",
    "expand_compact_edge_coefficients",
    "gather_compact_lora_memory",
    "validate_compact_lora_memory",
]

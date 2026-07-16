"""Fixed-capacity device representation of immutable LoRA memory graphs."""

from __future__ import annotations

from typing import NamedTuple, cast

import jax
import jax.numpy as jnp

from apm.lm.config import GptNeoConfig
from apm.lm.lora import (
    LoraConfig,
    LoraEdge,
    LoraEdgeBank,
    insert_lora_edge,
    stack_lora_edges,
)
from apm.memory.graph import MemoryGraph, path_incidence_matrix


class PackedLoraMemory(NamedTuple):
    """Padded edge bank, path matrix, and validity masks for one graph."""

    edge_bank: LoraEdgeBank
    node_path_matrix: jax.Array
    valid_node_mask: jax.Array
    valid_edge_mask: jax.Array


def pack_lora_memory(
    graph: MemoryGraph[LoraEdge],
    model_config: GptNeoConfig,
    lora_config: LoraConfig,
    max_nodes: int,
    max_edges: int,
) -> PackedLoraMemory:
    """Compile an immutable LoRA graph into fixed-capacity JAX arrays."""
    _validate_graph_and_capacities(graph, max_nodes, max_edges)
    edge_nodes = tuple(node for node in graph.nodes if node.parent_id is not None)
    edges = tuple(cast(LoraEdge, node.incoming_edge) for node in edge_nodes)
    node_count = len(graph.nodes)
    edge_count = len(edges)
    incidence = jnp.asarray(path_incidence_matrix(graph), dtype=jnp.float32)
    padded_incidence = jnp.pad(
        incidence,
        ((0, max_nodes - node_count), (0, max_edges - edge_count)),
    )
    return PackedLoraMemory(
        edge_bank=stack_lora_edges(
            edges,
            model_config,
            lora_config,
            max_edges,
        ),
        node_path_matrix=padded_incidence,
        valid_node_mask=jnp.arange(max_nodes) < node_count,
        valid_edge_mask=jnp.arange(max_edges) < edge_count,
    )


def edge_coefficients_for_node(
    packed: PackedLoraMemory,
    node_index: int | jax.Array,
) -> jax.Array:
    """Return one node's hard path coefficients, or zero for an invalid index."""
    index = jnp.asarray(node_index)
    if index.ndim != 0:
        raise ValueError("node_index must be a scalar")
    max_nodes = packed.node_path_matrix.shape[0]
    safe_index = jnp.clip(index, 0, max_nodes - 1)
    index_is_valid = (index >= 0) & (index < max_nodes)
    node_is_valid = packed.valid_node_mask[safe_index]
    coefficients = packed.node_path_matrix[safe_index] * packed.valid_edge_mask
    return jnp.where(
        index_is_valid & node_is_valid,
        coefficients,
        jnp.zeros_like(coefficients),
    )


def node_weights_to_edge_coefficients(
    node_weights: jax.Array,
    packed: PackedLoraMemory,
) -> jax.Array:
    """Map hard or continuous node weights to pathwise edge coefficients."""
    weights = jnp.asarray(node_weights)
    if weights.ndim < 1 or weights.shape[-1] != packed.node_path_matrix.shape[0]:
        raise ValueError("node_weights must end with the packed node-capacity dimension")
    masked_weights = weights * packed.valid_node_mask
    coefficients = jnp.einsum(
        "...n,ne->...e",
        masked_weights,
        packed.node_path_matrix,
    )
    return coefficients * packed.valid_edge_mask


def packed_with_candidate_edge(
    packed: PackedLoraMemory,
    candidate: LoraEdge,
    index: int,
) -> PackedLoraMemory:
    """Insert a differentiable candidate into one fixed edge slot."""
    max_edges = packed.valid_edge_mask.shape[0]
    if index < 0 or index >= max_edges:
        raise IndexError(f"candidate edge index is outside capacity: {index}")
    frozen_edge_bank = jax.tree_util.tree_map(jax.lax.stop_gradient, packed.edge_bank)
    return PackedLoraMemory(
        edge_bank=insert_lora_edge(frozen_edge_bank, candidate, index),
        node_path_matrix=packed.node_path_matrix,
        valid_node_mask=packed.valid_node_mask,
        valid_edge_mask=packed.valid_edge_mask.at[index].set(True),
    )


def _validate_graph_and_capacities(
    graph: MemoryGraph[LoraEdge],
    max_nodes: int,
    max_edges: int,
) -> None:
    if max_nodes < 1:
        raise ValueError("max_nodes must be at least one")
    if max_edges != max_nodes - 1:
        raise ValueError("max_edges must equal max_nodes - 1 for a rooted memory tree")
    if not graph.nodes:
        raise ValueError("a LoRA memory graph must contain one root")
    root, *non_root_nodes = graph.nodes
    if root.parent_id is not None or root.incoming_edge is not None:
        raise ValueError("the graph root must not have a parent or incoming edge")
    if any(node.parent_id is None for node in non_root_nodes):
        raise ValueError("every non-root node must have one parent")
    if len(graph.nodes) > max_nodes:
        raise ValueError("graph node count exceeds max_nodes")
    if len(non_root_nodes) > max_edges:
        raise ValueError("graph edge count exceeds max_edges")
    invalid_payload_ids = tuple(
        node.node_id
        for node in non_root_nodes
        if not isinstance(node.incoming_edge, LoraEdge)
    )
    if invalid_payload_ids:
        raise TypeError(f"non-root nodes must contain LoraEdge payloads: {invalid_payload_ids}")

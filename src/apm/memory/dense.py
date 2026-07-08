"""Dense parameter-delta memory graphs for Stage 1 APM experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias, cast

import jax
import numpy as np

ParamTree: TypeAlias = Any


@dataclass(frozen=True)
class DenseMemoryNode:
    """One node in a dense parameter-delta memory graph."""

    node_id: str
    parent_id: str | None
    trained_task: str
    train_stage: int
    depth: int
    delta: ParamTree | None


@dataclass(frozen=True)
class DenseMemoryGraph:
    """Root parameters plus dense deltas for each reachable memory node."""

    root_params: ParamTree
    nodes: tuple[DenseMemoryNode, ...]


@dataclass(frozen=True)
class EdgeMemoryStats:
    """Static parameter-space stats for one memory edge."""

    parent_id: str
    child_id: str
    child_task: str
    delta_l2_norm: float
    delta_bytes: int
    parameter_count: int


def init_dense_memory_graph(root_params: ParamTree, root_id: str = "root") -> DenseMemoryGraph:
    """Initialize a dense memory graph with one root parameter node."""
    root_node = DenseMemoryNode(
        node_id=root_id,
        parent_id=None,
        trained_task="root",
        train_stage=0,
        depth=0,
        delta=None,
    )
    return DenseMemoryGraph(root_params=root_params, nodes=(root_node,))


def add_dense_delta_node(
    graph: DenseMemoryGraph,
    node_id: str,
    parent_id: str,
    child_params: ParamTree,
    trained_task: str,
    train_stage: int,
) -> DenseMemoryGraph:
    """Store a child parameter state as a dense delta from an existing parent node."""
    if node_id in node_ids(graph):
        raise ValueError(f"memory node already exists: {node_id}")
    parent_node = node_by_id(graph, parent_id)
    parent_params = effective_params(graph, parent_id)
    child_node = DenseMemoryNode(
        node_id=node_id,
        parent_id=parent_id,
        trained_task=trained_task,
        train_stage=train_stage,
        depth=parent_node.depth + 1,
        delta=tree_subtract(child_params, parent_params),
    )
    return DenseMemoryGraph(root_params=graph.root_params, nodes=graph.nodes + (child_node,))


def node_ids(graph: DenseMemoryGraph) -> tuple[str, ...]:
    """Return memory node ids in graph insertion order."""
    return tuple(node.node_id for node in graph.nodes)


def task_node_ids(graph: DenseMemoryGraph) -> dict[str, str]:
    """Return the latest non-root node id trained for each task name."""
    return {node.trained_task: node.node_id for node in graph.nodes if node.parent_id is not None}


def node_by_id(graph: DenseMemoryGraph, node_id: str) -> DenseMemoryNode:
    """Return a memory node by id."""
    for node in graph.nodes:
        if node.node_id == node_id:
            return node
    raise KeyError(f"unknown memory node id: {node_id}")


def effective_params(graph: DenseMemoryGraph, node_id: str) -> ParamTree:
    """Reconstruct effective parameters at a graph node by summing parent-path deltas."""
    path = _node_path(graph, node_id)
    return cast(
        ParamTree,
        jax.tree_util.tree_map(
            lambda root_leaf, *delta_leaves: root_leaf + sum(delta_leaves, start=0.0),
            graph.root_params,
            *(node.delta for node in path if node.delta is not None),
        ),
    )


def edge_memory_stats(graph: DenseMemoryGraph) -> tuple[EdgeMemoryStats, ...]:
    """Return static parameter-space stats for every non-root memory edge."""
    return tuple(
        EdgeMemoryStats(
            parent_id=cast(str, node.parent_id),
            child_id=node.node_id,
            child_task=node.trained_task,
            delta_l2_norm=tree_l2_norm(cast(ParamTree, node.delta)),
            delta_bytes=tree_nbytes(cast(ParamTree, node.delta)),
            parameter_count=tree_parameter_count(cast(ParamTree, node.delta)),
        )
        for node in graph.nodes
        if node.delta is not None
    )


def graph_memory_bytes(graph: DenseMemoryGraph) -> int:
    """Return bytes required to store the root plus all dense deltas."""
    return tree_nbytes(graph.root_params) + sum(
        tree_nbytes(cast(ParamTree, node.delta))
        for node in graph.nodes
        if node.delta is not None
    )


def node_memory_bytes(graph: DenseMemoryGraph, node_id: str) -> int:
    """Return bytes owned by one node: root params for root, delta bytes for children."""
    node = node_by_id(graph, node_id)
    return tree_nbytes(graph.root_params) if node.delta is None else tree_nbytes(node.delta)


def tree_subtract(left: ParamTree, right: ParamTree) -> ParamTree:
    """Subtract two matching parameter trees."""
    return cast(ParamTree, jax.tree_util.tree_map(lambda left_leaf, right_leaf: left_leaf - right_leaf, left, right))


def tree_l2_norm(tree: ParamTree) -> float:
    """Return Euclidean norm over all leaves in a parameter tree."""
    squared_sum = sum(float(np.square(np.asarray(leaf, dtype=np.float64)).sum()) for leaf in jax.tree_util.tree_leaves(tree))
    return float(np.sqrt(squared_sum))


def tree_nbytes(tree: ParamTree) -> int:
    """Return total storage bytes for all leaves in a parameter tree."""
    return sum(int(np.asarray(leaf).nbytes) for leaf in jax.tree_util.tree_leaves(tree))


def tree_parameter_count(tree: ParamTree) -> int:
    """Return scalar parameter count across all leaves in a parameter tree."""
    return sum(int(np.asarray(leaf).size) for leaf in jax.tree_util.tree_leaves(tree))


def _node_path(graph: DenseMemoryGraph, node_id: str) -> tuple[DenseMemoryNode, ...]:
    path: list[DenseMemoryNode] = []
    current = node_by_id(graph, node_id)
    while current.parent_id is not None:
        path.append(current)
        current = node_by_id(graph, current.parent_id)
    return tuple(reversed(path))

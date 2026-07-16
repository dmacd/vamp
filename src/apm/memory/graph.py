"""Immutable, model-independent topology for VAMP memory graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, NewType, TypeVar

import numpy as np

NodeId = NewType("NodeId", str)
TaskId = NewType("TaskId", str)
PayloadT = TypeVar("PayloadT")


@dataclass(frozen=True)
class MemoryNode(Generic[PayloadT]):
    """One immutable node and its incoming model-specific edge payload."""

    node_id: NodeId
    parent_id: NodeId | None
    trained_task: TaskId | None
    train_stage: int
    depth: int
    incoming_edge: PayloadT | None


@dataclass(frozen=True)
class MemoryGraph(Generic[PayloadT]):
    """An immutable tuple of memory nodes in insertion order."""

    nodes: tuple[MemoryNode[PayloadT], ...]


def init_memory_graph(
    root_id: NodeId = NodeId("root"),
) -> MemoryGraph[PayloadT]:
    """Initialize a memory graph containing one payload-free root node."""
    root = MemoryNode[PayloadT](
        node_id=root_id,
        parent_id=None,
        trained_task=None,
        train_stage=0,
        depth=0,
        incoming_edge=None,
    )
    return MemoryGraph(nodes=(root,))


def add_memory_node(
    graph: MemoryGraph[PayloadT],
    node_id: NodeId,
    parent_id: NodeId,
    trained_task: TaskId,
    train_stage: int,
    incoming_edge: PayloadT,
) -> MemoryGraph[PayloadT]:
    """Return a new graph with one payload-bearing child appended."""
    if node_id in memory_node_ids(graph):
        raise ValueError(f"memory node already exists: {node_id}")
    parent = memory_node_by_id(graph, parent_id)
    child = MemoryNode(
        node_id=node_id,
        parent_id=parent_id,
        trained_task=trained_task,
        train_stage=train_stage,
        depth=parent.depth + 1,
        incoming_edge=incoming_edge,
    )
    return MemoryGraph(nodes=graph.nodes + (child,))


def memory_node_by_id(
    graph: MemoryGraph[PayloadT],
    node_id: NodeId,
) -> MemoryNode[PayloadT]:
    """Return the node with the requested identifier."""
    for node in graph.nodes:
        if node.node_id == node_id:
            return node
    raise KeyError(f"unknown memory node id: {node_id}")


def memory_node_ids(graph: MemoryGraph[PayloadT]) -> tuple[NodeId, ...]:
    """Return node identifiers in insertion order."""
    return tuple(node.node_id for node in graph.nodes)


def memory_edge_node_ids(graph: MemoryGraph[PayloadT]) -> tuple[NodeId, ...]:
    """Return non-root child identifiers in edge insertion order."""
    return tuple(node.node_id for node in graph.nodes if node.parent_id is not None)


def child_memory_nodes(
    graph: MemoryGraph[PayloadT],
    parent_id: NodeId,
) -> tuple[MemoryNode[PayloadT], ...]:
    """Return a parent's direct children in insertion order."""
    memory_node_by_id(graph, parent_id)
    return tuple(node for node in graph.nodes if node.parent_id == parent_id)


def memory_node_path(
    graph: MemoryGraph[PayloadT],
    node_id: NodeId,
) -> tuple[MemoryNode[PayloadT], ...]:
    """Return the root-to-node path, including both endpoints."""
    reverse_path: list[MemoryNode[PayloadT]] = []
    current = memory_node_by_id(graph, node_id)
    while True:
        reverse_path.append(current)
        if current.parent_id is None:
            return tuple(reversed(reverse_path))
        current = memory_node_by_id(graph, current.parent_id)


def path_incidence_matrix(graph: MemoryGraph[PayloadT]) -> np.ndarray:
    """Return a read-only node-by-edge path-incidence matrix."""
    edge_node_ids = memory_edge_node_ids(graph)
    path_edge_ids = tuple(
        frozenset(
            path_node.node_id
            for path_node in memory_node_path(graph, node.node_id)
            if path_node.parent_id is not None
        )
        for node in graph.nodes
    )
    incidence = np.asarray(
        [
            [edge_node_id in node_path_edge_ids for edge_node_id in edge_node_ids]
            for node_path_edge_ids in path_edge_ids
        ],
        dtype=np.float32,
    ).reshape(len(graph.nodes), len(edge_node_ids))
    incidence.flags.writeable = False
    return incidence

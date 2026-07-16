"""Dense parameter-delta interpretation of generic VAMP memory graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar, cast

import jax
import numpy as np

from apm.memory.graph import (
    MemoryGraph,
    NodeId,
    TaskId,
    add_memory_node,
    init_memory_graph,
    memory_node_by_id,
    memory_node_path,
)

ParamsT = TypeVar("ParamsT")


@dataclass(frozen=True)
class DenseParameterMemory(Generic[ParamsT]):
    """Frozen root parameters paired with dense-delta graph topology."""

    root_params: ParamsT
    graph: MemoryGraph[ParamsT]


@dataclass(frozen=True)
class EdgeMemoryStats:
    """Static parameter-space statistics for one dense memory edge."""

    parent_id: NodeId
    child_id: NodeId
    child_task: TaskId
    delta_l2_norm: float
    delta_bytes: int
    parameter_count: int


def init_dense_parameter_memory(
    root_params: ParamsT,
    root_id: NodeId = NodeId("root"),
) -> DenseParameterMemory[ParamsT]:
    """Initialize dense parameter memory with one payload-free root node."""
    return DenseParameterMemory(
        root_params=root_params,
        graph=init_memory_graph(root_id),
    )


def add_dense_delta(
    memory: DenseParameterMemory[ParamsT],
    node_id: NodeId,
    parent_id: NodeId,
    child_params: ParamsT,
    trained_task: TaskId,
    train_stage: int,
) -> DenseParameterMemory[ParamsT]:
    """Return memory with a child's dense delta appended to the graph."""
    parent_params = effective_dense_params(memory, parent_id)
    graph = add_memory_node(
        memory.graph,
        node_id=node_id,
        parent_id=parent_id,
        trained_task=trained_task,
        train_stage=train_stage,
        incoming_edge=tree_subtract(child_params, parent_params),
    )
    return DenseParameterMemory(root_params=memory.root_params, graph=graph)


def effective_dense_params(
    memory: DenseParameterMemory[ParamsT],
    node_id: NodeId,
) -> ParamsT:
    """Reconstruct effective parameters by summing dense path payloads."""
    deltas = tuple(
        node.incoming_edge
        for node in memory_node_path(memory.graph, node_id)
        if node.incoming_edge is not None
    )
    if not deltas:
        return memory.root_params
    return cast(
        ParamsT,
        jax.tree_util.tree_map(
            lambda root_leaf, *delta_leaves: root_leaf
            + sum(delta_leaves, start=0.0),
            memory.root_params,
            *deltas,
        ),
    )


def dense_task_node_ids(
    memory: DenseParameterMemory[ParamsT],
) -> dict[TaskId, NodeId]:
    """Return the latest non-root node trained for each task."""
    return {
        node.trained_task: node.node_id
        for node in memory.graph.nodes
        if node.parent_id is not None and node.trained_task is not None
    }


def dense_edge_memory_stats(
    memory: DenseParameterMemory[ParamsT],
) -> tuple[EdgeMemoryStats, ...]:
    """Return static parameter-space statistics for every dense edge."""
    return tuple(
        EdgeMemoryStats(
            parent_id=cast(NodeId, node.parent_id),
            child_id=node.node_id,
            child_task=cast(TaskId, node.trained_task),
            delta_l2_norm=tree_l2_norm(node.incoming_edge),
            delta_bytes=tree_nbytes(node.incoming_edge),
            parameter_count=tree_parameter_count(node.incoming_edge),
        )
        for node in memory.graph.nodes
        if node.incoming_edge is not None
    )


def dense_graph_memory_bytes(memory: DenseParameterMemory[ParamsT]) -> int:
    """Return bytes used by the root parameters and all dense edge payloads."""
    return tree_nbytes(memory.root_params) + sum(
        tree_nbytes(node.incoming_edge)
        for node in memory.graph.nodes
        if node.incoming_edge is not None
    )


def dense_node_memory_bytes(
    memory: DenseParameterMemory[ParamsT],
    node_id: NodeId,
) -> int:
    """Return root parameter bytes or one child node's dense payload bytes."""
    node = memory_node_by_id(memory.graph, node_id)
    return (
        tree_nbytes(memory.root_params)
        if node.incoming_edge is None
        else tree_nbytes(node.incoming_edge)
    )


def tree_subtract(left: ParamsT, right: ParamsT) -> ParamsT:
    """Subtract two matching parameter trees without changing their structure."""
    return cast(
        ParamsT,
        jax.tree_util.tree_map(
            lambda left_leaf, right_leaf: left_leaf - right_leaf,
            left,
            right,
        ),
    )


def tree_l2_norm(tree: ParamsT) -> float:
    """Return the Euclidean norm over every leaf in a parameter tree."""
    squared_sum = sum(
        float(np.square(np.asarray(leaf, dtype=np.float64)).sum())
        for leaf in jax.tree_util.tree_leaves(tree)
    )
    return float(np.sqrt(squared_sum))


def tree_nbytes(tree: ParamsT) -> int:
    """Return total storage bytes for every leaf in a parameter tree."""
    return sum(
        int(np.asarray(leaf).nbytes) for leaf in jax.tree_util.tree_leaves(tree)
    )


def tree_parameter_count(tree: ParamsT) -> int:
    """Return the scalar parameter count across all leaves in a tree."""
    return sum(
        int(np.asarray(leaf).size) for leaf in jax.tree_util.tree_leaves(tree)
    )

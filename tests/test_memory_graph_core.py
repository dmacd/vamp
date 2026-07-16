from __future__ import annotations

import numpy as np
import pytest

from apm.memory.graph import (
    NodeId,
    TaskId,
    add_memory_node,
    child_memory_nodes,
    init_memory_graph,
    memory_edge_node_ids,
    memory_node_by_id,
    memory_node_ids,
    memory_node_path,
    path_incidence_matrix,
)


def _branching_graph():
    root = init_memory_graph(NodeId("root"))
    first = add_memory_node(
        root,
        NodeId("a"),
        NodeId("root"),
        TaskId("task_a"),
        1,
        "edge_a",
    )
    second = add_memory_node(
        first,
        NodeId("b"),
        NodeId("root"),
        TaskId("task_b"),
        2,
        "edge_b",
    )
    return add_memory_node(
        second,
        NodeId("c"),
        NodeId("a"),
        TaskId("task_c"),
        3,
        "edge_c",
    )


def test_graph_preserves_insertion_order_and_branch_paths() -> None:
    graph = _branching_graph()

    assert memory_node_ids(graph) == ("root", "a", "b", "c")
    assert memory_edge_node_ids(graph) == ("a", "b", "c")
    assert tuple(node.node_id for node in child_memory_nodes(graph, NodeId("root"))) == (
        "a",
        "b",
    )
    assert tuple(node.node_id for node in memory_node_path(graph, NodeId("c"))) == (
        "root",
        "a",
        "c",
    )
    assert memory_node_by_id(graph, NodeId("c")).depth == 2


def test_graph_rejects_duplicate_nodes_and_unknown_parents() -> None:
    graph = _branching_graph()

    with pytest.raises(ValueError, match="already exists"):
        add_memory_node(
            graph,
            NodeId("a"),
            NodeId("root"),
            TaskId("duplicate"),
            4,
            "duplicate_edge",
        )
    with pytest.raises(KeyError, match="unknown memory node id"):
        add_memory_node(
            graph,
            NodeId("orphan"),
            NodeId("missing"),
            TaskId("orphan"),
            4,
            "orphan_edge",
        )


def test_path_incidence_uses_node_and_edge_insertion_order() -> None:
    incidence = path_incidence_matrix(_branching_graph())

    np.testing.assert_array_equal(
        incidence,
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )
    assert not incidence.flags.writeable


def test_root_only_incidence_has_zero_edge_columns() -> None:
    incidence = path_incidence_matrix(init_memory_graph())

    assert incidence.shape == (1, 0)

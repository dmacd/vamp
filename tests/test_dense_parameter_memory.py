from __future__ import annotations

import jax
import numpy as np

from apm.memory.dense import (
    add_dense_delta,
    dense_edge_memory_stats,
    dense_graph_memory_bytes,
    dense_node_memory_bytes,
    dense_task_node_ids,
    effective_dense_params,
    init_dense_parameter_memory,
    tree_l2_norm,
    tree_nbytes,
    tree_parameter_count,
)
from apm.memory.graph import NodeId, TaskId


def _params(offset: float) -> dict[str, np.ndarray]:
    return {
        "bias": np.asarray([offset], dtype=np.float32),
        "kernel": np.asarray([offset + 1.0, offset + 2.0], dtype=np.float32),
    }


def test_dense_memory_reconstructs_root_and_parent_child_parameters() -> None:
    root_params = _params(0.0)
    first_params = _params(1.0)
    second_params = _params(-2.0)
    memory = init_dense_parameter_memory(root_params)
    memory = add_dense_delta(
        memory,
        NodeId("first"),
        NodeId("root"),
        first_params,
        TaskId("task_a"),
        1,
    )
    memory = add_dense_delta(
        memory,
        NodeId("second"),
        NodeId("first"),
        second_params,
        TaskId("task_b"),
        2,
    )

    assert effective_dense_params(memory, NodeId("root")) is root_params
    for reconstructed, expected in (
        (effective_dense_params(memory, NodeId("first")), first_params),
        (effective_dense_params(memory, NodeId("second")), second_params),
    ):
        for reconstructed_leaf, expected_leaf in zip(
            jax.tree_util.tree_leaves(reconstructed),
            jax.tree_util.tree_leaves(expected),
        ):
            np.testing.assert_allclose(reconstructed_leaf, expected_leaf)


def test_dense_memory_reports_task_and_storage_statistics() -> None:
    root_params = _params(0.0)
    memory = add_dense_delta(
        init_dense_parameter_memory(root_params),
        NodeId("child"),
        NodeId("root"),
        _params(1.0),
        TaskId("task"),
        1,
    )
    stats = dense_edge_memory_stats(memory)

    assert dense_task_node_ids(memory) == {TaskId("task"): NodeId("child")}
    assert len(stats) == 1
    assert stats[0].parent_id == "root"
    assert stats[0].child_id == "child"
    assert stats[0].child_task == "task"
    assert stats[0].delta_l2_norm == tree_l2_norm(memory.graph.nodes[1].incoming_edge)
    assert stats[0].delta_bytes == tree_nbytes(memory.graph.nodes[1].incoming_edge)
    assert stats[0].parameter_count == tree_parameter_count(
        memory.graph.nodes[1].incoming_edge
    )
    assert dense_node_memory_bytes(memory, NodeId("root")) == tree_nbytes(root_params)
    assert dense_node_memory_bytes(memory, NodeId("child")) == stats[0].delta_bytes
    assert dense_graph_memory_bytes(memory) == tree_nbytes(root_params) + stats[0].delta_bytes


def test_dense_memory_excludes_sibling_deltas_from_effective_parameters() -> None:
    root_params = _params(0.0)
    memory = add_dense_delta(
        init_dense_parameter_memory(root_params),
        NodeId("left"),
        NodeId("root"),
        _params(3.0),
        TaskId("left_task"),
        1,
    )
    memory = add_dense_delta(
        memory,
        NodeId("right"),
        NodeId("root"),
        _params(-4.0),
        TaskId("right_task"),
        2,
    )

    for node_id, expected in ((NodeId("left"), _params(3.0)), (NodeId("right"), _params(-4.0))):
        for reconstructed_leaf, expected_leaf in zip(
            jax.tree_util.tree_leaves(effective_dense_params(memory, node_id)),
            jax.tree_util.tree_leaves(expected),
        ):
            np.testing.assert_allclose(reconstructed_leaf, expected_leaf)

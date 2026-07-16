from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from apm.lm.config import GptNeoConfig
from apm.lm.lora import LoraConfig, LoraEdge, init_lora_edge
from apm.lm.lora_memory import (
    PackedLoraMemory,
    edge_coefficients_for_node,
    node_weights_to_edge_coefficients,
    pack_lora_memory,
    packed_with_candidate_edge,
)
from apm.memory.graph import (
    MemoryGraph,
    MemoryNode,
    NodeId,
    TaskId,
    add_memory_node,
    init_memory_graph,
)


def _model_config() -> GptNeoConfig:
    return GptNeoConfig(
        vocab_size=8,
        max_position_embeddings=8,
        hidden_size=4,
        intermediate_size=8,
        num_layers=1,
        num_heads=2,
        attention_types=("global",),
        local_window_size=2,
    )


def _lora_config() -> LoraConfig:
    return LoraConfig(rank=2, alpha=2.0)


def _constant_edge(value: float) -> LoraEdge:
    edge = init_lora_edge(jax.random.PRNGKey(int(value * 10 + 100)), _model_config(), _lora_config())
    return jax.tree_util.tree_map(lambda leaf: jnp.full_like(leaf, value), edge)


def _branching_graph() -> tuple[MemoryGraph[LoraEdge], tuple[LoraEdge, ...]]:
    first_edge, second_edge, sibling_edge = (
        _constant_edge(1.0),
        _constant_edge(2.0),
        _constant_edge(3.0),
    )
    root = init_memory_graph(NodeId("root"))
    first = add_memory_node(
        root,
        NodeId("a"),
        NodeId("root"),
        TaskId("task_a"),
        1,
        first_edge,
    )
    second = add_memory_node(
        first,
        NodeId("b"),
        NodeId("a"),
        TaskId("task_b"),
        2,
        second_edge,
    )
    graph = add_memory_node(
        second,
        NodeId("c"),
        NodeId("root"),
        TaskId("task_c"),
        3,
        sibling_edge,
    )
    return graph, (first_edge, second_edge, sibling_edge)


def _tree_slot(tree: object, index: int) -> list[np.ndarray]:
    return [np.asarray(leaf[index]) for leaf in jax.tree_util.tree_leaves(tree)]


def _assert_slot_matches_edge(packed: PackedLoraMemory, index: int, edge: LoraEdge) -> None:
    packed_slot = _tree_slot(packed.edge_bank, index)
    edge_leaves = [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(edge)]
    assert len(packed_slot) == len(edge_leaves)
    for actual, expected in zip(packed_slot, edge_leaves):
        np.testing.assert_array_equal(actual, expected)


def test_pack_lora_memory_preserves_branching_incidence_and_edge_order() -> None:
    graph, edges = _branching_graph()

    packed = pack_lora_memory(
        graph,
        _model_config(),
        _lora_config(),
        max_nodes=5,
        max_edges=4,
    )

    np.testing.assert_array_equal(
        np.asarray(packed.node_path_matrix),
        np.asarray(
            (
                (0.0, 0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0, 0.0),
                (1.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 0.0),
            ),
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(np.asarray(packed.valid_node_mask), (True, True, True, True, False))
    np.testing.assert_array_equal(np.asarray(packed.valid_edge_mask), (True, True, True, False))
    for edge_index, edge in enumerate(edges):
        _assert_slot_matches_edge(packed, edge_index, edge)
    assert all(np.count_nonzero(leaf) == 0 for leaf in _tree_slot(packed.edge_bank, 3))


def test_linear_and_sibling_nodes_have_isolated_hard_paths() -> None:
    graph, _ = _branching_graph()
    packed = pack_lora_memory(graph, _model_config(), _lora_config(), 5, 4)

    expected_by_node = (
        (0.0, 0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 0.0),
        (1.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
    )
    for node_index, expected in enumerate(expected_by_node):
        np.testing.assert_array_equal(
            np.asarray(edge_coefficients_for_node(packed, node_index)),
            np.asarray(expected, dtype=np.float32),
        )

    assert edge_coefficients_for_node(packed, 2)[2] == 0.0
    assert edge_coefficients_for_node(packed, 3)[0] == 0.0
    assert edge_coefficients_for_node(packed, 3)[1] == 0.0


def test_one_hot_node_weights_equal_hard_node_coefficients() -> None:
    graph, _ = _branching_graph()
    packed = pack_lora_memory(graph, _model_config(), _lora_config(), 5, 4)
    one_hot_nodes = jnp.eye(5, dtype=jnp.float32)

    mapped = node_weights_to_edge_coefficients(one_hot_nodes, packed)

    for node_index in range(5):
        np.testing.assert_array_equal(
            np.asarray(mapped[node_index]),
            np.asarray(edge_coefficients_for_node(packed, node_index)),
        )


def test_continuous_node_weights_map_linearly_to_edges() -> None:
    graph, _ = _branching_graph()
    packed = pack_lora_memory(graph, _model_config(), _lora_config(), 5, 4)
    node_weights = jnp.asarray(
        (
            (0.1, 0.2, 0.3, 0.4, 100.0),
            (0.0, 0.5, 0.0, 0.5, -100.0),
        ),
        dtype=jnp.float32,
    )

    coefficients = node_weights_to_edge_coefficients(node_weights, packed)

    np.testing.assert_allclose(
        np.asarray(coefficients),
        np.asarray(((0.5, 0.3, 0.4, 0.0), (0.5, 0.0, 0.5, 0.0)), dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


def test_invalid_padded_nodes_and_edges_are_neutral_even_if_matrix_is_nonzero() -> None:
    graph, _ = _branching_graph()
    packed = pack_lora_memory(graph, _model_config(), _lora_config(), 5, 4)
    polluted_matrix = packed.node_path_matrix.at[4, :].set(9.0).at[:, 3].set(7.0)
    polluted = packed._replace(node_path_matrix=polluted_matrix)

    padded_node_coefficients = edge_coefficients_for_node(polluted, 4)
    out_of_range_coefficients = edge_coefficients_for_node(polluted, 99)
    padded_weight_coefficients = node_weights_to_edge_coefficients(
        jnp.asarray((0.0, 0.0, 0.0, 0.0, 100.0)),
        polluted,
    )

    np.testing.assert_array_equal(np.asarray(padded_node_coefficients), np.zeros((4,), dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(out_of_range_coefficients), np.zeros((4,), dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(padded_weight_coefficients), np.zeros((4,), dtype=np.float32))


def test_root_only_graph_packs_to_zero_edge_capacity() -> None:
    packed = pack_lora_memory(
        init_memory_graph(NodeId("root")),
        _model_config(),
        _lora_config(),
        max_nodes=1,
        max_edges=0,
    )

    assert packed.node_path_matrix.shape == (1, 0)
    assert packed.valid_node_mask.tolist() == [True]
    assert packed.valid_edge_mask.shape == (0,)
    assert all(leaf.shape[0] == 0 for leaf in jax.tree_util.tree_leaves(packed.edge_bank))


@pytest.mark.parametrize(
    ("max_nodes", "max_edges", "message"),
    (
        (0, -1, "max_nodes"),
        (3, 3, "max_edges"),
        (3, 1, "max_edges"),
    ),
)
def test_pack_rejects_inconsistent_capacities(max_nodes: int, max_edges: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        pack_lora_memory(
            init_memory_graph(NodeId("root")),
            _model_config(),
            _lora_config(),
            max_nodes,
            max_edges,
        )


def test_pack_rejects_graph_larger_than_capacity() -> None:
    graph, _ = _branching_graph()

    with pytest.raises(ValueError, match="exceeds max_nodes"):
        pack_lora_memory(graph, _model_config(), _lora_config(), 3, 2)


def test_pack_rejects_missing_or_malformed_lora_payloads() -> None:
    missing_payload_graph = add_memory_node(
        init_memory_graph(NodeId("root")),
        NodeId("bad"),
        NodeId("root"),
        TaskId("bad"),
        1,
        "not-a-lora-edge",
    )
    malformed_edge_graph = add_memory_node(
        init_memory_graph(NodeId("root")),
        NodeId("bad"),
        NodeId("root"),
        TaskId("bad"),
        1,
        LoraEdge(blocks=()),
    )

    with pytest.raises(TypeError, match="LoraEdge payloads"):
        pack_lora_memory(  # type: ignore[arg-type]
            missing_payload_graph,
            _model_config(),
            _lora_config(),
            2,
            1,
        )
    with pytest.raises(ValueError, match="one block per model layer"):
        pack_lora_memory(malformed_edge_graph, _model_config(), _lora_config(), 2, 1)


def test_pack_rejects_invalid_root_topology() -> None:
    invalid_graph = MemoryGraph(
        nodes=(
            MemoryNode(
                node_id=NodeId("root"),
                parent_id=NodeId("parent"),
                trained_task=None,
                train_stage=0,
                depth=0,
                incoming_edge=_constant_edge(1.0),
            ),
        )
    )

    with pytest.raises(ValueError, match="root"):
        pack_lora_memory(invalid_graph, _model_config(), _lora_config(), 1, 0)


def test_packed_with_candidate_edge_inserts_without_mutating_committed_slots() -> None:
    graph = add_memory_node(
        init_memory_graph(NodeId("root")),
        NodeId("a"),
        NodeId("root"),
        TaskId("task_a"),
        1,
        _constant_edge(1.0),
    )
    packed = pack_lora_memory(graph, _model_config(), _lora_config(), 4, 3)
    candidate = _constant_edge(4.0)
    original_leaves = tuple(np.asarray(leaf).copy() for leaf in jax.tree_util.tree_leaves(packed.edge_bank))

    with_candidate = packed_with_candidate_edge(packed, candidate, 1)

    _assert_slot_matches_edge(with_candidate, 0, _constant_edge(1.0))
    _assert_slot_matches_edge(with_candidate, 1, candidate)
    assert all(np.count_nonzero(leaf) == 0 for leaf in _tree_slot(with_candidate.edge_bank, 2))
    np.testing.assert_array_equal(np.asarray(with_candidate.valid_edge_mask), (True, True, False))
    np.testing.assert_array_equal(np.asarray(with_candidate.node_path_matrix), np.asarray(packed.node_path_matrix))
    np.testing.assert_array_equal(np.asarray(with_candidate.valid_node_mask), np.asarray(packed.valid_node_mask))
    for original, current in zip(original_leaves, jax.tree_util.tree_leaves(packed.edge_bank)):
        np.testing.assert_array_equal(original, np.asarray(current))


def test_candidate_insertion_validates_index_and_shapes() -> None:
    packed = pack_lora_memory(
        init_memory_graph(NodeId("root")),
        _model_config(),
        _lora_config(),
        3,
        2,
    )
    candidate = _constant_edge(1.0)

    with pytest.raises(IndexError, match="outside capacity"):
        packed_with_candidate_edge(packed, candidate, -1)
    with pytest.raises(IndexError, match="outside capacity"):
        packed_with_candidate_edge(packed, candidate, 2)
    with pytest.raises(ValueError, match="block count"):
        packed_with_candidate_edge(packed, LoraEdge(blocks=()), 0)


def test_candidate_insertion_stops_committed_bank_gradients() -> None:
    graph = add_memory_node(
        init_memory_graph(NodeId("root")),
        NodeId("a"),
        NodeId("root"),
        TaskId("task_a"),
        1,
        _constant_edge(1.0),
    )
    packed = pack_lora_memory(graph, _model_config(), _lora_config(), 3, 2)
    candidate = _constant_edge(2.0)

    def bank_loss(edge_bank, candidate_edge):
        candidate_memory = packed_with_candidate_edge(
            packed._replace(edge_bank=edge_bank),
            candidate_edge,
            1,
        )
        return sum(jnp.sum(jnp.square(leaf)) for leaf in jax.tree_util.tree_leaves(candidate_memory.edge_bank))

    bank_gradient, candidate_gradient = jax.grad(bank_loss, argnums=(0, 1))(packed.edge_bank, candidate)

    assert all(np.count_nonzero(np.asarray(leaf)) == 0 for leaf in jax.tree_util.tree_leaves(bank_gradient))
    candidate_gradient_leaves = jax.tree_util.tree_leaves(candidate_gradient)
    assert candidate_gradient_leaves
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in candidate_gradient_leaves)
    assert all(np.count_nonzero(np.asarray(leaf)) > 0 for leaf in candidate_gradient_leaves)


def test_node_weight_shape_is_validated() -> None:
    packed = pack_lora_memory(
        init_memory_graph(NodeId("root")),
        _model_config(),
        _lora_config(),
        3,
        2,
    )

    with pytest.raises(ValueError, match="node-capacity"):
        node_weights_to_edge_coefficients(jnp.ones((2,), dtype=jnp.float32), packed)
    with pytest.raises(ValueError, match="scalar"):
        edge_coefficients_for_node(packed, jnp.asarray((0, 1)))

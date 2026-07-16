from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from apm.continual.language_benchmark_metrics import (
    StageTaskObservation,
    TransferObservation,
    account_language_memory,
    decompose_benchmark_metrics,
)
from apm.continual.language_tasks import AddressBook
from apm.lm.config import GptNeoConfig
from apm.lm.lora import LoraConfig, LoraEdge, init_lora_edge
from apm.lm.lora_memory import pack_lora_memory
from apm.lm.parameters import init_gpt_neo_params
from apm.memory.graph import (
    MemoryGraph,
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


def _constant_edge(index: int) -> LoraEdge:
    edge = init_lora_edge(jax.random.PRNGKey(index), _model_config(), _lora_config())
    return jax.tree_util.tree_map(
        lambda leaf: jnp.full_like(leaf, float(index)),
        edge,
    )


def _branching_graph() -> MemoryGraph[LoraEdge]:
    graph = init_memory_graph(NodeId("root"))
    graph = add_memory_node(
        graph,
        NodeId("a"),
        NodeId("root"),
        TaskId("task_a"),
        1,
        _constant_edge(1),
    )
    graph = add_memory_node(
        graph,
        NodeId("b"),
        NodeId("a"),
        TaskId("task_b"),
        2,
        _constant_edge(2),
    )
    return add_memory_node(
        graph,
        NodeId("c"),
        NodeId("root"),
        TaskId("task_c"),
        3,
        _constant_edge(3),
    )


def _address_book(capacity: int) -> AddressBook:
    keys = np.zeros((capacity, 4), dtype=np.float32)
    keys[:4] = np.eye(4, dtype=np.float32)
    return AddressBook(
        node_ids=(NodeId("root"), NodeId("a"), NodeId("b"), NodeId("c"))
        + (None,) * (capacity - 4),
        keys=keys,
        valid_node_mask=np.arange(capacity) < 4,
    )


def test_metric_decomposition_keeps_stored_and_routing_forgetting_distinct() -> None:
    observations = (
        StageTaskObservation(1, "animals", 2.0, 2.5, 1.8, "base-a", "path-a"),
        StageTaskObservation(2, "animals", 2.2, 2.4, 1.7, "base-a", "path-a"),
        StageTaskObservation(2, "vehicles_tools", 3.0, 3.2, 2.8, "base-a", "path-b"),
        StageTaskObservation(3, "animals", 1.9, 2.8, 1.7, "base-b", "path-a-drift"),
        StageTaskObservation(3, "vehicles_tools", 3.1, 3.0, 2.7, "base-b", "path-b"),
        StageTaskObservation(3, "family_home", 2.5, 2.6, 2.4, "base-b", "path-c"),
    )
    transfers = (
        TransferObservation(1, "animals", 4.0, 3.5, (3.5, 3.0, 2.5), 10),
        TransferObservation(2, "vehicles_tools", 3.0, 3.2, (3.2, 3.1, 3.0), 20),
        TransferObservation(3, "family_home", 2.8, 2.7, (2.7, 2.6), 30),
    )

    decomposition = decompose_benchmark_metrics(
        observations,
        transfers,
        nll_threshold=2.9,
    )

    assert tuple(row.stored_forgetting for row in decomposition.stored) == pytest.approx(
        (0.0, 0.2, 0.0, 0.0, 0.1, 0.0)
    )
    assert tuple(row.routing_forgetting for row in decomposition.routing) == pytest.approx(
        (0.0, 0.0, 0.0, 0.4, 0.0, 0.0)
    )
    assert decomposition.routing[1].task_oracle_regret == pytest.approx(0.2)
    assert decomposition.routing[1].best_node_regret == pytest.approx(0.7)
    assert decomposition.stored[3].base_checksum_drift is True
    assert decomposition.stored[3].committed_path_checksum_drift is True
    assert decomposition.transfer[0].parent_advantage == pytest.approx(0.5)
    assert decomposition.transfer[0].first_step_improvement == pytest.approx(0.5)
    assert decomposition.transfer[0].fixed_budget_improvement == pytest.approx(1.0)
    assert decomposition.transfer[0].updates_to_threshold == 2
    assert decomposition.transfer[0].tokens_to_threshold == 20
    assert decomposition.transfer[1].parent_advantage == pytest.approx(-0.2)
    assert decomposition.transfer[1].updates_to_threshold is None
    assert decomposition.transfer[1].tokens_to_threshold is None


def test_memory_accounting_counts_logical_paths_and_separates_padding() -> None:
    graph = _branching_graph()
    base_params = init_gpt_neo_params(jax.random.PRNGKey(0), _model_config())
    packed = pack_lora_memory(graph, _model_config(), _lora_config(), 5, 4)

    accounting = account_language_memory(
        base_params,
        graph,
        _address_book(5),
        packed,
        _lora_config(),
        optimizer_state_snapshots=(
            {"state": jnp.zeros((10,), dtype=jnp.float32)},
            {"state": jnp.zeros((20,), dtype=jnp.float32)},
        ),
        nll_improvement=0.5,
    )

    assert accounting.base_parameter_count == 232
    assert accounting.base_bytes == 928
    assert accounting.edge_bytes == (448, 448, 448)
    assert accounting.committed_lora_bytes == 1344
    assert tuple(path.lora_bytes for path in accounting.effective_paths) == (
        0,
        448,
        896,
        448,
    )
    assert tuple(path.projection_ranks.query for path in accounting.effective_paths) == (
        0,
        2,
        4,
        2,
    )
    assert accounting.address_key_bytes_per_node == (16, 16, 16, 16)
    assert accounting.address_key_bytes == 64
    assert accounting.graph_metadata_bytes == 345
    assert accounting.persistent_bytes == 2681
    assert accounting.packed_edge_bank_bytes == 1792
    assert accounting.packed_path_matrix_bytes == 80
    assert accounting.packed_validity_mask_bytes == 9
    assert accounting.packed_runtime_bytes == 1881
    assert accounting.packed_padding_bytes == 480
    assert accounting.optimizer_peak_bytes == 80
    assert accounting.bytes_per_task == pytest.approx((2681 - 928) / 3)
    assert accounting.bytes_per_nll_improvement == pytest.approx((2681 - 928) / 0.5)

    wider_packed = pack_lora_memory(
        graph,
        _model_config(),
        _lora_config(),
        max_nodes=6,
        max_edges=5,
    )
    wider = account_language_memory(
        base_params,
        graph,
        _address_book(6),
        wider_packed,
        _lora_config(),
        nll_improvement=0.5,
    )
    assert wider.persistent_bytes == accounting.persistent_bytes
    assert wider.address_key_bytes == accounting.address_key_bytes
    assert wider.packed_runtime_bytes > accounting.packed_runtime_bytes
    assert wider.packed_padding_bytes > accounting.packed_padding_bytes


def test_nonpositive_nll_improvement_has_no_efficiency_ratio() -> None:
    graph = _branching_graph()
    accounting = account_language_memory(
        init_gpt_neo_params(jax.random.PRNGKey(4), _model_config()),
        graph,
        _address_book(5),
        pack_lora_memory(graph, _model_config(), _lora_config(), 5, 4),
        _lora_config(),
        nll_improvement=-0.1,
    )

    assert accounting.nll_improvement == pytest.approx(-0.1)
    assert accounting.bytes_per_nll_improvement is None

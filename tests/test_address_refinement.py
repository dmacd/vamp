from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import apm.memory.address_refinement as refinement_module
from apm.continual.language_tasks import AddressBook, NodeId, RouterBatch, TaskId
from apm.lm.checkpoint import parameter_checksum
from apm.lm.config import GptNeoConfig
from apm.lm.gpt_neo import ForwardResult
from apm.lm.lora import LoraConfig, LoraEdge, init_lora_edge
from apm.lm.lora_memory import PackedLoraMemory, pack_lora_memory
from apm.lm.parameters import GptNeoParams, init_gpt_neo_params
from apm.memory.address_refinement import (
    EbtAddressResult,
    EbtConfig,
    masked_node_probabilities,
    refine_ebt_address,
    soft_mixture_prefix_nll,
)
from apm.memory.content_addressing import HopfieldConfig, hopfield_address
from apm.memory.graph import add_memory_node, init_memory_graph
from apm.memory.prefix_energy import exhaustive_prefix_nll_address


def _model_config() -> GptNeoConfig:
    return GptNeoConfig(
        vocab_size=5,
        max_position_embeddings=3,
        hidden_size=4,
        intermediate_size=8,
        num_layers=1,
        num_heads=2,
        attention_types=("global",),
        local_window_size=3,
    )


def _lora_config() -> LoraConfig:
    return LoraConfig(rank=1, alpha=1.0)


def _nonzero_edge(seed: int, right_value: float) -> LoraEdge:
    edge = init_lora_edge(
        jax.random.PRNGKey(seed),
        _model_config(),
        _lora_config(),
    )
    return edge._replace(
        blocks=tuple(
            block._replace(
                **{
                    field_name: getattr(block, field_name)._replace(
                        right=jnp.full_like(
                            getattr(block, field_name).right,
                            right_value,
                        )
                    )
                    for field_name in block._fields
                }
            )
            for block in edge.blocks
        )
    )


def _packed_memory() -> PackedLoraMemory:
    graph = add_memory_node(
        init_memory_graph(NodeId("root")),
        NodeId("node-a"),
        NodeId("root"),
        TaskId("task-a"),
        1,
        _nonzero_edge(1, 0.2),
    )
    graph = add_memory_node(
        graph,
        NodeId("node-b"),
        NodeId("root"),
        TaskId("task-b"),
        2,
        _nonzero_edge(2, -0.2),
    )
    return pack_lora_memory(
        graph,
        _model_config(),
        _lora_config(),
        max_nodes=4,
        max_edges=3,
    )


def _router_batch() -> RouterBatch:
    return RouterBatch(
        input_ids=np.asarray(((1, 2, 3), (3, 2, 1)), dtype=np.int32),
        attention_mask=np.ones((2, 3), dtype=np.bool_),
        target_ids=np.asarray(((2, 3, 4), (2, 1, 0)), dtype=np.int32),
        loss_mask=np.ones((2, 3), dtype=np.bool_),
    )


def _row(batch: RouterBatch, row_index: int) -> RouterBatch:
    return RouterBatch(
        batch.input_ids[row_index : row_index + 1],
        batch.attention_mask[row_index : row_index + 1],
        batch.target_ids[row_index : row_index + 1],
        batch.loss_mask[row_index : row_index + 1],
    )


def _hopfield_result():
    address_book = AddressBook(
        node_ids=(NodeId("root"), NodeId("node-a"), NodeId("node-b"), None),
        keys=np.asarray(
            (
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 0.0),
            ),
            dtype=np.float32,
        ),
        valid_node_mask=np.asarray((True, True, True, False)),
    )
    return hopfield_address(
        np.asarray(
            ((0.8, 0.6, 0.0, 0.0), (0.0, 0.6, 0.8, 0.0)),
            dtype=np.float32,
        ),
        address_book,
        HopfieldConfig(beta=2.0, top_k=2),
    )


def _assert_trees_equal(first, second) -> None:
    first_leaves, first_structure = jax.tree_util.tree_flatten(first)
    second_leaves, second_structure = jax.tree_util.tree_flatten(second)
    assert first_structure == second_structure
    assert all(
        np.array_equal(np.asarray(first_leaf), np.asarray(second_leaf))
        for first_leaf, second_leaf in zip(first_leaves, second_leaves)
    )


def test_one_hot_continuous_execution_matches_the_same_hard_nodes() -> None:
    model_config = _model_config()
    base_params = init_gpt_neo_params(jax.random.PRNGKey(0), model_config)
    packed_memory = _packed_memory()
    prefix_batch = _router_batch()
    selected_nodes = jnp.asarray((0, 2), dtype=jnp.int32)
    one_hot_probabilities = jax.nn.one_hot(
        selected_nodes,
        packed_memory.node_path_matrix.shape[0],
        dtype=jnp.float32,
    )

    continuous_nll = soft_mixture_prefix_nll(
        base_params,
        model_config,
        packed_memory,
        _lora_config(),
        prefix_batch,
        one_hot_probabilities,
    )
    exhaustive = exhaustive_prefix_nll_address(
        base_params,
        model_config,
        packed_memory,
        _lora_config(),
        prefix_batch,
    )
    expected_nll = np.asarray(exhaustive.node_scores)[
        np.arange(2),
        np.asarray(selected_nodes),
    ]

    np.testing.assert_allclose(continuous_nll, expected_nll, rtol=1e-6, atol=1e-6)
    probabilities = masked_node_probabilities(
        jnp.zeros((2, 4), dtype=jnp.float32),
        jnp.asarray(((True, True, False, False), (True, False, True, False))),
        tau=1.0,
    )
    np.testing.assert_array_equal(
        probabilities,
        ((0.5, 0.5, 0.0, 0.0), (0.5, 0.0, 0.5, 0.0)),
    )


def test_all_initializations_return_soft_and_hard_outputs_without_mutation() -> None:
    model_config = _model_config()
    base_params = init_gpt_neo_params(jax.random.PRNGKey(3), model_config)
    packed_memory = _packed_memory()
    prefix_batch = _router_batch()
    hopfield_result = _hopfield_result()
    base_checksum = parameter_checksum(base_params, model_config)
    memory_snapshot = jax.tree_util.tree_map(
        lambda value: np.asarray(value).copy(),
        packed_memory,
    )
    cases = (
        ("uniform", {}),
        ("hopfield", {"hopfield_result": hopfield_result}),
        (
            "full_node",
            {"initial_node_indices": np.asarray((1, 2), dtype=np.int32)},
        ),
        ("hopfield_top_k", {"hopfield_result": hopfield_result}),
    )

    results = tuple(
        (
            initialization,
            refine_ebt_address(
                base_params,
                model_config,
                packed_memory,
                _lora_config(),
                prefix_batch,
                EbtConfig(steps=2, initialization=initialization),
                **initialization_inputs,
            ),
        )
        for initialization, initialization_inputs in cases
    )

    for initialization, result in results:
        assert result.final_node_logits.shape == (2, 4)
        assert result.node_probabilities.shape == (2, 4)
        assert result.edge_coefficients.shape == (2, 3)
        assert result.selected_indices.shape == (2,)
        assert result.soft_mixture_nll.shape == (2,)
        assert result.hard_node_nll.shape == (2,)
        assert result.objective_trace.shape == (3, 2)
        assert np.all(np.isfinite(np.asarray(result.objective_trace)))
        assert np.all(np.isfinite(np.asarray(result.soft_mixture_nll)))
        assert np.all(np.isfinite(np.asarray(result.hard_node_nll)))
        np.testing.assert_array_equal(result.node_probabilities[:, 3], 0.0)
        assert np.all(np.isneginf(np.asarray(result.final_node_logits)[:, 3]))
        if initialization == "hopfield_top_k":
            for row_index, top_k_indices in enumerate(
                np.asarray(hopfield_result.top_k_indices)
            ):
                excluded = np.setdiff1d(np.arange(4), top_k_indices)
                np.testing.assert_array_equal(
                    np.asarray(result.node_probabilities)[row_index, excluded],
                    0.0,
                )
    assert parameter_checksum(base_params, model_config) == base_checksum
    _assert_trees_equal(packed_memory, memory_snapshot)


def test_sum_gradient_adam_updates_are_independent_across_batch_rows() -> None:
    model_config = _model_config()
    base_params = init_gpt_neo_params(jax.random.PRNGKey(4), model_config)
    packed_memory = _packed_memory()
    prefix_batch = _router_batch()
    config = EbtConfig(steps=3, entropy_penalty=0.0)
    batched = refine_ebt_address(
        base_params,
        model_config,
        packed_memory,
        _lora_config(),
        prefix_batch,
        config,
    )
    separate = tuple(
        refine_ebt_address(
            base_params,
            model_config,
            packed_memory,
            _lora_config(),
            _row(prefix_batch, row_index),
            config,
        )
        for row_index in range(2)
    )

    for field_name in (
        "final_node_logits",
        "node_probabilities",
        "edge_coefficients",
        "selected_indices",
        "soft_mixture_nll",
        "hard_node_nll",
    ):
        expected = np.concatenate(
            tuple(np.asarray(getattr(result, field_name)) for result in separate),
            axis=0,
        )
        np.testing.assert_allclose(
            getattr(batched, field_name),
            expected,
            rtol=2e-5,
            atol=2e-6,
        )
    np.testing.assert_allclose(
        batched.objective_trace,
        np.concatenate(
            tuple(np.asarray(result.objective_trace) for result in separate),
            axis=1,
        ),
        rtol=2e-5,
        atol=2e-6,
    )


def test_constructed_refinement_has_finite_gradient_and_decreasing_nll(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def apply_edge_sensitive_model(
        params: GptNeoParams,
        config: GptNeoConfig,
        token_ids: jax.Array,
        attention_mask: jax.Array,
        *,
        position_ids: jax.Array | None = None,
        lora_memory: PackedLoraMemory,
        edge_coefficients: jax.Array,
        lora_config: LoraConfig,
        capture=None,
        training: bool = False,
        rng_key: jax.Array | None = None,
    ) -> ForwardResult:
        del (
            params,
            attention_mask,
            position_ids,
            lora_memory,
            lora_config,
            capture,
            training,
            rng_key,
        )
        target_logit = 5.0 * edge_coefficients[:, 0]
        logits = jnp.zeros(
            (*token_ids.shape, config.vocab_size),
            dtype=jnp.float32,
        ).at[:, :, 0].set(target_logit[:, None])
        return ForwardResult(
            logits=logits,
            final_hidden=jnp.zeros(
                (*token_ids.shape, config.hidden_size),
                dtype=jnp.float32,
            ),
            captured_hidden=(),
        )

    monkeypatch.setattr(
        refinement_module,
        "apply_gpt_neo",
        apply_edge_sensitive_model,
    )
    model_config = _model_config()
    base_params = init_gpt_neo_params(jax.random.PRNGKey(5), model_config)
    prefix_batch = RouterBatch(
        input_ids=np.asarray(((1, 2, 3), (3, 2, 1)), dtype=np.int32),
        attention_mask=np.ones((2, 3), dtype=np.bool_),
        target_ids=np.zeros((2, 3), dtype=np.int32),
        loss_mask=np.ones((2, 3), dtype=np.bool_),
    )

    result = refine_ebt_address(
        base_params,
        model_config,
        _packed_memory(),
        _lora_config(),
        prefix_batch,
        EbtConfig(
            steps=10,
            learning_rate=0.2,
            entropy_penalty=0.0,
        ),
    )

    assert np.all(np.isfinite(np.asarray(result.objective_trace)))
    assert np.all(result.objective_trace[-1] < result.objective_trace[0])
    assert np.all(result.selected_indices == 1)
    assert np.all(result.soft_mixture_nll <= result.objective_trace[0])


def test_config_result_and_router_signature_are_frozen_and_task_free() -> None:
    config = EbtConfig()

    assert tuple(field.name for field in fields(config)) == (
        "steps",
        "learning_rate",
        "tau",
        "entropy_penalty",
        "initialization",
    )
    assert config == EbtConfig(20, 0.1, 1.0, 0.01, "uniform")
    with pytest.raises(FrozenInstanceError):
        config.steps = 1
    assert EbtAddressResult._fields == (
        "final_node_logits",
        "node_probabilities",
        "edge_coefficients",
        "selected_indices",
        "soft_mixture_nll",
        "hard_node_nll",
        "objective_trace",
    )
    signature_names = tuple(inspect.signature(refine_ebt_address).parameters)
    assert signature_names == (
        "base_params",
        "model_config",
        "packed_memory",
        "lora_config",
        "prefix_batch",
        "config",
        "hopfield_result",
        "initial_node_indices",
    )
    assert not any(
        forbidden in parameter_name
        for parameter_name in signature_names
        for forbidden in ("task", "oracle", "suffix")
    )
    with pytest.raises(ValueError, match="positive"):
        EbtConfig(steps=0)
    with pytest.raises(ValueError, match="unknown"):
        EbtConfig(initialization="invalid")

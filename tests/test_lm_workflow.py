from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from apm.lm.config import GptNeoConfig
from apm.lm.lora import LoraConfig, init_lora_edge
from apm.lm.lora_memory import pack_lora_memory, packed_with_candidate_edge
from apm.lm.parameters import init_gpt_neo_params
from apm.lm.text_data import TokenBatch
from apm.lm.training import (
    LmTrainConfig,
    init_base_train_state,
    init_candidate_lora_train_state,
)
from apm.lm.workflow import (
    LmLossTrace,
    evaluate_normalized_nll,
    run_base_updates,
    run_candidate_edge_updates,
    tiny_shakespeare_model_config,
    tiny_shakespeare_unit_model_config,
)
from apm.memory.graph import init_memory_graph


def _tiny_config() -> GptNeoConfig:
    return GptNeoConfig(
        vocab_size=4,
        max_position_embeddings=4,
        hidden_size=8,
        intermediate_size=16,
        num_layers=1,
        num_heads=2,
        attention_types=("global",),
        local_window_size=4,
    )


def _batch(input_ids: tuple[int, ...], target_ids: tuple[int, ...]) -> TokenBatch:
    return TokenBatch(
        np.asarray((input_ids,), dtype=np.int32),
        np.ones((1, len(input_ids)), dtype=np.bool_),
        np.asarray((target_ids,), dtype=np.int32),
        np.ones((1, len(input_ids)), dtype=np.bool_),
    )


def _assert_trees_equal(first, second) -> None:
    first_leaves, first_structure = jax.tree_util.tree_flatten(first)
    second_leaves, second_structure = jax.tree_util.tree_flatten(second)
    assert first_structure == second_structure
    assert all(
        np.array_equal(np.asarray(first_leaf), np.asarray(second_leaf))
        for first_leaf, second_leaf in zip(first_leaves, second_leaves)
    )


def test_tiny_shakespeare_model_presets_match_the_frozen_architectures() -> None:
    standard = tiny_shakespeare_model_config(vocab_size=67)
    unit = tiny_shakespeare_unit_model_config(vocab_size=23)

    assert (
        standard.vocab_size,
        standard.max_position_embeddings,
        standard.hidden_size,
        standard.num_layers,
        standard.num_heads,
        standard.intermediate_size,
        standard.attention_types,
        standard.local_window_size,
    ) == (67, 256, 128, 4, 4, 512, ("global", "local", "global", "local"), 64)
    assert (
        standard.embedding_dropout,
        standard.attention_dropout,
        standard.residual_dropout,
    ) == (0.0, 0.0, 0.0)
    assert (
        unit.vocab_size,
        unit.max_position_embeddings,
        unit.hidden_size,
        unit.num_layers,
        unit.num_heads,
        unit.intermediate_size,
        unit.attention_types,
        unit.local_window_size,
    ) == (23, 64, 64, 2, 4, 256, ("global", "local"), 64)


def test_evaluation_normalizes_over_active_tokens_across_batches() -> None:
    config = _tiny_config()
    params = init_gpt_neo_params(jax.random.PRNGKey(0), config)
    first = _batch((2, 3, 2, 3), (3, 2, 3, 1))
    second = TokenBatch(
        np.asarray(((3, 0, 0, 0),), dtype=np.int32),
        np.asarray(((True, False, False, False),)),
        np.asarray(((1, 0, 0, 0),), dtype=np.int32),
        np.asarray(((True, False, False, False),)),
    )
    combined = TokenBatch(
        np.concatenate((first.input_ids, second.input_ids)),
        np.concatenate((first.attention_mask, second.attention_mask)),
        np.concatenate((first.target_ids, second.target_ids)),
        np.concatenate((first.loss_mask, second.loss_mask)),
    )

    separate_nll = evaluate_normalized_nll(params, config, (first, second))
    combined_nll = evaluate_normalized_nll(params, config, (combined,))

    assert separate_nll == pytest.approx(combined_nll, rel=1e-7, abs=1e-7)


def test_base_workflow_is_deterministic_exact_budget_and_reduces_validation_nll() -> None:
    config = _tiny_config()
    batches = (
        _batch((2, 3, 2, 3), (3, 2, 3, 1)),
        _batch((3, 2, 3, 2), (2, 3, 2, 1)),
    )
    train_config = LmTrainConfig(
        learning_rate=2e-2,
        steps=41,
        batch_size=1,
        weight_decay=0.0,
        gradient_clip_norm=1.0,
    )
    params = init_gpt_neo_params(jax.random.PRNGKey(1), config)
    initial_state = init_base_train_state(
        params,
        jax.random.PRNGKey(2),
        train_config,
    )
    initial_nll = evaluate_normalized_nll(params, config, batches)

    first_state, first_trace = run_base_updates(
        initial_state,
        batches,
        config,
        train_config,
    )
    second_state, second_trace = run_base_updates(
        initial_state,
        batches,
        config,
        train_config,
    )
    final_nll = evaluate_normalized_nll(first_state.trainable, config, batches)

    assert isinstance(first_trace, LmLossTrace)
    assert len(first_trace.step_losses) == train_config.steps
    assert int(first_state.step) == train_config.steps
    assert int(initial_state.step) == 0
    assert final_nll < initial_nll
    assert first_trace == second_trace
    _assert_trees_equal(first_state, second_state)


def test_candidate_workflow_is_deterministic_exact_budget_and_reduces_nll() -> None:
    config = _tiny_config()
    batches = (_batch((2, 3, 2, 3), (3, 2, 3, 1)),)
    train_config = LmTrainConfig(
        learning_rate=3e-2,
        steps=40,
        batch_size=1,
        weight_decay=0.0,
        gradient_clip_norm=1.0,
    )
    base_params = init_gpt_neo_params(jax.random.PRNGKey(3), config)
    lora_config = LoraConfig(rank=2, alpha=2.0)
    candidate = init_lora_edge(jax.random.PRNGKey(4), config, lora_config)
    packed_memory = pack_lora_memory(
        init_memory_graph(),
        config,
        lora_config,
        max_nodes=2,
        max_edges=1,
    )
    coefficients = jnp.asarray([1.0], dtype=jnp.float32)
    initial_state = init_candidate_lora_train_state(
        candidate,
        jax.random.PRNGKey(5),
        train_config,
    )
    initial_nll = evaluate_normalized_nll(
        base_params,
        config,
        batches,
        lora_memory=packed_with_candidate_edge(packed_memory, candidate, 0),
        edge_coefficients=coefficients,
        lora_config=lora_config,
    )

    first_state, first_trace = run_candidate_edge_updates(
        initial_state,
        batches,
        base_params,
        config,
        packed_memory,
        lora_config,
        jnp.zeros((1,), dtype=jnp.float32),
        0,
        train_config,
    )
    second_state, second_trace = run_candidate_edge_updates(
        initial_state,
        batches,
        base_params,
        config,
        packed_memory,
        lora_config,
        jnp.zeros((1,), dtype=jnp.float32),
        0,
        train_config,
    )
    final_nll = evaluate_normalized_nll(
        base_params,
        config,
        batches,
        lora_memory=packed_with_candidate_edge(
            packed_memory,
            first_state.trainable,
            0,
        ),
        edge_coefficients=coefficients,
        lora_config=lora_config,
    )

    assert len(first_trace.step_losses) == train_config.steps
    assert int(first_state.step) == train_config.steps
    assert final_nll < initial_nll
    assert first_trace == second_trace
    _assert_trees_equal(first_state, second_state)

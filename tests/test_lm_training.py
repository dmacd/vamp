from __future__ import annotations

from dataclasses import FrozenInstanceError
from hashlib import sha256

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from apm.lm.config import GptNeoConfig
from apm.lm.gpt_neo import apply_gpt_neo
from apm.lm.lora import LoraConfig, LoraEdge, init_lora_edge
from apm.lm.lora_memory import (
    edge_coefficients_for_node,
    pack_lora_memory,
    packed_with_candidate_edge,
)
from apm.lm.losses import mean_token_nll
from apm.lm.parameters import init_gpt_neo_params
from apm.lm.text_data import TokenBatch
from apm.lm.training import (
    BASE_TRAINING_PRESET,
    EDGE_LORA_PRESET,
    EDGE_TRAINING_PRESET,
    LmTrainConfig,
    base_train_step,
    candidate_lora_train_step,
    init_base_train_state,
    init_candidate_lora_train_state,
)
from apm.memory.graph import NodeId, TaskId, add_memory_node, init_memory_graph


def _model_config() -> GptNeoConfig:
    return GptNeoConfig(
        vocab_size=5,
        max_position_embeddings=4,
        hidden_size=8,
        intermediate_size=16,
        num_layers=1,
        num_heads=2,
        attention_types=("global",),
        local_window_size=2,
    )


def _fixed_batch(target_ids: tuple[int, ...] = (1, 2, 3, 4)) -> TokenBatch:
    input_row = np.asarray((0, 1, 2, 3), dtype=np.int32)
    target_row = np.asarray(target_ids, dtype=np.int32)
    return TokenBatch(
        input_ids=np.stack((input_row, input_row)),
        attention_mask=np.ones((2, 4), dtype=np.bool_),
        target_ids=np.stack((target_row, target_row)),
        loss_mask=np.ones((2, 4), dtype=np.bool_),
    )


def _base_nll(params, model_config: GptNeoConfig, batch: TokenBatch) -> jax.Array:
    result = apply_gpt_neo(
        params,
        model_config,
        jnp.asarray(batch.input_ids),
        jnp.asarray(batch.attention_mask),
    )
    return mean_token_nll(result.logits, jnp.asarray(batch.target_ids), jnp.asarray(batch.loss_mask))


def _tree_checksum(tree: object) -> str:
    digest = sha256()
    for leaf in jax.tree_util.tree_leaves(tree):
        array = np.asarray(leaf)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _constant_edge(value: float, model_config: GptNeoConfig, lora_config: LoraConfig) -> LoraEdge:
    edge = init_lora_edge(jax.random.PRNGKey(91), model_config, lora_config)
    return jax.tree_util.tree_map(lambda leaf: jnp.full_like(leaf, value), edge)


def test_training_presets_record_standard_budgets_and_rank() -> None:
    assert BASE_TRAINING_PRESET.learning_rate == 3e-4
    assert BASE_TRAINING_PRESET.steps == 5_000
    assert BASE_TRAINING_PRESET.batch_size == 32
    assert BASE_TRAINING_PRESET.gradient_clip_norm == 1.0
    assert EDGE_TRAINING_PRESET.learning_rate == 1e-3
    assert EDGE_TRAINING_PRESET.steps == 1_000
    assert EDGE_TRAINING_PRESET.batch_size == 32
    assert EDGE_LORA_PRESET.rank == 4
    assert EDGE_LORA_PRESET.alpha == 4.0


def test_training_config_is_frozen_and_validated() -> None:
    config = LmTrainConfig(learning_rate=1e-3, steps=2, batch_size=1)

    with pytest.raises(FrozenInstanceError):
        config.steps = 3  # type: ignore[misc]
    with pytest.raises(ValueError, match="learning_rate"):
        LmTrainConfig(learning_rate=0.0, steps=1, batch_size=1)
    with pytest.raises(ValueError, match="steps"):
        LmTrainConfig(learning_rate=1e-3, steps=0, batch_size=1)
    with pytest.raises(ValueError, match="batch_size"):
        LmTrainConfig(learning_rate=1e-3, steps=1, batch_size=0)
    with pytest.raises(ValueError, match="weight_decay"):
        LmTrainConfig(learning_rate=1e-3, steps=1, batch_size=1, weight_decay=-0.1)
    with pytest.raises(ValueError, match="gradient_clip_norm"):
        LmTrainConfig(learning_rate=1e-3, steps=1, batch_size=1, gradient_clip_norm=0.0)


def test_base_train_step_overfits_a_tiny_fixed_batch() -> None:
    model_config = _model_config()
    batch = _fixed_batch()
    train_config = LmTrainConfig(
        learning_rate=0.02,
        steps=200,
        batch_size=2,
        weight_decay=0.0,
        gradient_clip_norm=1.0,
    )
    initial_params = init_gpt_neo_params(jax.random.PRNGKey(0), model_config)
    state = init_base_train_state(initial_params, jax.random.PRNGKey(1), train_config)
    initial_loss = _base_nll(state.trainable, model_config, batch)
    compiled_step = jax.jit(
        lambda current_state, current_batch: base_train_step(
            current_state,
            current_batch,
            model_config,
            train_config,
        )
    )

    for _ in range(train_config.steps):
        state, _ = compiled_step(state, batch)
    final_loss = _base_nll(state.trainable, model_config, batch)

    assert int(state.step) == train_config.steps
    assert float(final_loss) < float(initial_loss)
    assert float(final_loss) < 0.05
    assert _tree_checksum(initial_params) != _tree_checksum(state.trainable)


def test_candidate_edge_training_improves_nll_without_changing_base_or_bank() -> None:
    model_config = _model_config()
    lora_config = LoraConfig(rank=2, alpha=2.0)
    batch = _fixed_batch(target_ids=(4, 4, 4, 4))
    train_config = LmTrainConfig(
        learning_rate=0.02,
        steps=200,
        batch_size=2,
        weight_decay=0.0,
        gradient_clip_norm=1.0,
    )
    base_params = init_gpt_neo_params(jax.random.PRNGKey(2), model_config)
    committed_edge = _constant_edge(0.05, model_config, lora_config)
    graph = add_memory_node(
        init_memory_graph(NodeId("root")),
        NodeId("committed"),
        NodeId("root"),
        TaskId("old_task"),
        1,
        committed_edge,
    )
    packed = pack_lora_memory(graph, model_config, lora_config, max_nodes=3, max_edges=2)
    parent_coefficients = edge_coefficients_for_node(packed, 1)
    candidate = init_lora_edge(jax.random.PRNGKey(3), model_config, lora_config)
    state = init_candidate_lora_train_state(candidate, jax.random.PRNGKey(4), train_config)
    base_checksum = _tree_checksum(base_params)
    bank_checksum = _tree_checksum(packed.edge_bank)

    def candidate_nll(candidate_edge: LoraEdge) -> jax.Array:
        candidate_memory = packed_with_candidate_edge(packed, candidate_edge, 1)
        coefficients = parent_coefficients.at[1].set(1.0)
        result = apply_gpt_neo(
            base_params,
            model_config,
            jnp.asarray(batch.input_ids),
            jnp.asarray(batch.attention_mask),
            lora_memory=candidate_memory,
            edge_coefficients=coefficients,
            lora_config=lora_config,
        )
        return mean_token_nll(
            result.logits,
            jnp.asarray(batch.target_ids),
            jnp.asarray(batch.loss_mask),
        )

    initial_loss = candidate_nll(state.trainable)
    compiled_step = jax.jit(
        lambda current_state, current_batch: candidate_lora_train_step(
            current_state,
            current_batch,
            base_params,
            model_config,
            packed,
            lora_config,
            parent_coefficients,
            1,
            train_config,
        )
    )
    for _ in range(train_config.steps):
        state, _ = compiled_step(state, batch)
    final_loss = candidate_nll(state.trainable)

    assert int(state.step) == train_config.steps
    assert float(final_loss) < float(initial_loss)
    assert _tree_checksum(candidate) != _tree_checksum(state.trainable)
    assert _tree_checksum(base_params) == base_checksum
    assert _tree_checksum(packed.edge_bank) == bank_checksum


def test_train_steps_require_the_configured_fixed_batch_size() -> None:
    model_config = _model_config()
    train_config = LmTrainConfig(learning_rate=1e-3, steps=1, batch_size=1)
    state = init_base_train_state(
        init_gpt_neo_params(jax.random.PRNGKey(5), model_config),
        jax.random.PRNGKey(6),
        train_config,
    )

    with pytest.raises(ValueError, match="batch_size"):
        base_train_step(state, _fixed_batch(), model_config, train_config)

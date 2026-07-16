from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from apm.lm.config import GptNeoConfig
from apm.lm.generation import greedy_generate
from apm.lm.gpt_neo import apply_gpt_neo
from apm.lm.lora import LoraConfig, LoraEdge, init_lora_edge
from apm.lm.lora_memory import edge_coefficients_for_node, pack_lora_memory
from apm.lm.parameters import init_gpt_neo_params
from apm.memory.graph import NodeId, TaskId, add_memory_node, init_memory_graph


def _model_config() -> GptNeoConfig:
    return GptNeoConfig(
        vocab_size=8,
        max_position_embeddings=8,
        hidden_size=8,
        intermediate_size=16,
        num_layers=1,
        num_heads=2,
        attention_types=("global",),
        local_window_size=2,
    )


def _constant_edge(value: float, model_config: GptNeoConfig, lora_config: LoraConfig) -> LoraEdge:
    edge = init_lora_edge(jax.random.PRNGKey(20), model_config, lora_config)
    return jax.tree_util.tree_map(lambda leaf: jnp.full_like(leaf, value), edge)


def test_greedy_generation_is_deterministic_and_preserves_prompts() -> None:
    config = _model_config()
    params = init_gpt_neo_params(jax.random.PRNGKey(0), config)
    prompt_ids = jnp.asarray(((1, 2, 3), (4, 5, 0)), dtype=jnp.int32)
    attention_mask = jnp.asarray(((True, True, True), (True, True, False)))

    first = greedy_generate(
        params,
        config,
        prompt_ids,
        attention_mask,
        max_new_tokens=3,
        pad_token_id=0,
    )
    second = greedy_generate(
        params,
        config,
        prompt_ids,
        attention_mask,
        max_new_tokens=3,
        pad_token_id=0,
    )

    np.testing.assert_array_equal(np.asarray(first), np.asarray(second))
    np.testing.assert_array_equal(np.asarray(first[0, :3]), np.asarray(prompt_ids[0]))
    np.testing.assert_array_equal(np.asarray(first[1, :2]), np.asarray(prompt_ids[1, :2]))
    assert first.shape == (2, 6)


def test_generation_stops_each_sequence_independently_at_eos() -> None:
    config = _model_config()
    initialized = init_gpt_neo_params(jax.random.PRNGKey(1), config)
    zero_params = jax.tree_util.tree_map(jnp.zeros_like, initialized)
    prompt_ids = jnp.asarray(((1, 7), (0, 7)), dtype=jnp.int32)
    attention_mask = jnp.asarray(((True, False), (True, False)))

    generated = greedy_generate(
        zero_params,
        config,
        prompt_ids,
        attention_mask,
        max_new_tokens=3,
        eos_token_id=0,
        pad_token_id=7,
    )

    np.testing.assert_array_equal(
        np.asarray(generated),
        np.asarray(((1, 0, 7, 7, 7), (0, 7, 7, 7, 7)), dtype=np.int32),
    )


def test_generation_uses_per_row_hard_lora_nodes() -> None:
    config = _model_config()
    lora_config = LoraConfig(rank=2, alpha=2.0)
    params = init_gpt_neo_params(jax.random.PRNGKey(2), config)
    graph = add_memory_node(
        init_memory_graph(NodeId("root")),
        NodeId("adapted"),
        NodeId("root"),
        TaskId("task"),
        1,
        _constant_edge(0.2, config, lora_config),
    )
    packed = pack_lora_memory(graph, config, lora_config, max_nodes=2, max_edges=1)
    prompt_ids = jnp.asarray(((1, 2), (1, 2)), dtype=jnp.int32)
    attention_mask = jnp.ones_like(prompt_ids, dtype=jnp.bool_)
    node_indices = jnp.asarray((0, 1), dtype=jnp.int32)
    coefficients = jax.vmap(lambda index: edge_coefficients_for_node(packed, index))(node_indices)
    expected_next = jnp.argmax(
        apply_gpt_neo(
            params,
            config,
            prompt_ids,
            attention_mask,
            lora_memory=packed,
            edge_coefficients=coefficients,
            lora_config=lora_config,
        ).logits[:, -1],
        axis=-1,
    )

    generated = greedy_generate(
        params,
        config,
        prompt_ids,
        attention_mask,
        max_new_tokens=1,
        lora_memory=packed,
        lora_config=lora_config,
        node_index=node_indices,
    )

    np.testing.assert_array_equal(np.asarray(generated[:, 2]), np.asarray(expected_next))


def test_generation_rejects_context_overflow_and_non_right_padding() -> None:
    config = _model_config()
    params = init_gpt_neo_params(jax.random.PRNGKey(3), config)

    with pytest.raises(ValueError, match="max_position_embeddings"):
        greedy_generate(
            params,
            config,
            jnp.ones((1, 7), dtype=jnp.int32),
            jnp.ones((1, 7), dtype=jnp.bool_),
            max_new_tokens=2,
        )
    with pytest.raises(ValueError, match="right-padded"):
        greedy_generate(
            params,
            config,
            jnp.asarray(((1, 0, 2),)),
            jnp.asarray(((True, False, True),)),
            max_new_tokens=1,
        )


def test_generation_requires_complete_lora_arguments() -> None:
    config = _model_config()
    params = init_gpt_neo_params(jax.random.PRNGKey(4), config)
    prompt_ids = jnp.asarray(((1, 2),), dtype=jnp.int32)
    attention_mask = jnp.ones_like(prompt_ids, dtype=jnp.bool_)

    with pytest.raises(ValueError, match="supplied together"):
        greedy_generate(
            params,
            config,
            prompt_ids,
            attention_mask,
            max_new_tokens=1,
            node_index=0,
        )

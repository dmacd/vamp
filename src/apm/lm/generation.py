"""Deterministic uncached greedy generation for plain-JAX GPT-Neo."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from apm.lm.config import GptNeoConfig
from apm.lm.gpt_neo import apply_gpt_neo
from apm.lm.lora import LoraConfig
from apm.lm.lora_memory import PackedLoraMemory, edge_coefficients_for_node
from apm.lm.parameters import GptNeoParams


def greedy_generate(
    params: GptNeoParams,
    config: GptNeoConfig,
    prompt_ids: jax.Array,
    attention_mask: jax.Array,
    max_new_tokens: int,
    *,
    eos_token_id: int | None = None,
    pad_token_id: int = 0,
    lora_memory: PackedLoraMemory | None = None,
    lora_config: LoraConfig | None = None,
    node_index: int | jax.Array | None = None,
) -> jax.Array:
    """Greedily extend right-padded prompts, padding each row after its EOS."""
    token_ids = jnp.asarray(prompt_ids, dtype=jnp.int32)
    mask = jnp.asarray(attention_mask, dtype=jnp.bool_)
    _validate_generation_inputs(
        token_ids,
        mask,
        config,
        max_new_tokens,
        eos_token_id,
        pad_token_id,
    )
    edge_coefficients = _hard_edge_coefficients(
        lora_memory,
        lora_config,
        node_index,
        token_ids.shape[0],
    )
    batch_size, prompt_width = token_ids.shape
    output_width = prompt_width + max_new_tokens
    generated_ids = jnp.full(
        (batch_size, output_width),
        pad_token_id,
        dtype=jnp.int32,
    ).at[:, :prompt_width].set(token_ids)
    generated_mask = jnp.zeros(
        (batch_size, output_width),
        dtype=jnp.bool_,
    ).at[:, :prompt_width].set(mask)
    lengths = jnp.sum(mask, axis=1, dtype=jnp.int32)
    row_indices = jnp.arange(batch_size, dtype=jnp.int32)
    finished = (
        generated_ids[row_indices, lengths - 1] == eos_token_id
        if eos_token_id is not None
        else jnp.zeros((batch_size,), dtype=jnp.bool_)
    )

    for generation_step in range(max_new_tokens):
        visible_width = prompt_width + generation_step
        result = apply_gpt_neo(
            params,
            config,
            generated_ids[:, :visible_width],
            generated_mask[:, :visible_width],
            lora_memory=lora_memory,
            edge_coefficients=edge_coefficients,
            lora_config=lora_config,
        )
        next_token_ids = jnp.argmax(
            result.logits[row_indices, lengths - 1],
            axis=-1,
        ).astype(jnp.int32)
        active = ~finished
        insertion_indices = lengths
        previous_values = generated_ids[row_indices, insertion_indices]
        generated_ids = generated_ids.at[row_indices, insertion_indices].set(
            jnp.where(active, next_token_ids, previous_values)
        )
        previous_mask = generated_mask[row_indices, insertion_indices]
        generated_mask = generated_mask.at[row_indices, insertion_indices].set(
            jnp.where(active, True, previous_mask)
        )
        lengths = lengths + active.astype(jnp.int32)
        if eos_token_id is not None:
            finished = finished | (active & (next_token_ids == eos_token_id))
            if bool(np.asarray(jnp.all(finished))):
                break
    return generated_ids


def _validate_generation_inputs(
    prompt_ids: jax.Array,
    attention_mask: jax.Array,
    config: GptNeoConfig,
    max_new_tokens: int,
    eos_token_id: int | None,
    pad_token_id: int,
) -> None:
    if prompt_ids.ndim != 2:
        raise ValueError("prompt_ids must have shape [batch, sequence]")
    if attention_mask.shape != prompt_ids.shape:
        raise ValueError("attention_mask must match prompt_ids")
    if prompt_ids.shape[0] == 0 or prompt_ids.shape[1] == 0:
        raise ValueError("prompt_ids must contain at least one token per batch row")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative")
    if prompt_ids.shape[1] + max_new_tokens > config.max_position_embeddings:
        raise ValueError("prompt plus generated length exceeds max_position_embeddings")
    if pad_token_id < 0 or pad_token_id >= config.vocab_size:
        raise ValueError("pad_token_id is outside the model vocabulary")
    if eos_token_id is not None and (
        eos_token_id < 0 or eos_token_id >= config.vocab_size
    ):
        raise ValueError("eos_token_id is outside the model vocabulary")
    prompt_array = np.asarray(prompt_ids)
    mask_array = np.asarray(attention_mask, dtype=np.bool_)
    if np.any(prompt_array < 0) or np.any(prompt_array >= config.vocab_size):
        raise ValueError("prompt_ids contain tokens outside the model vocabulary")
    if np.any(np.sum(mask_array, axis=1) == 0):
        raise ValueError("every prompt row must contain at least one unmasked token")
    if np.any(mask_array[:, 1:] & ~mask_array[:, :-1]):
        raise ValueError("attention_mask must describe right-padded prompts")


def _hard_edge_coefficients(
    lora_memory: PackedLoraMemory | None,
    lora_config: LoraConfig | None,
    node_index: int | jax.Array | None,
    batch_size: int,
) -> jax.Array | None:
    arguments_present = (
        lora_memory is not None,
        lora_config is not None,
        node_index is not None,
    )
    if not any(arguments_present):
        return None
    if not all(arguments_present):
        raise ValueError("lora_memory, lora_config, and node_index must be supplied together")
    assert lora_memory is not None
    assert node_index is not None
    indices = jnp.asarray(node_index)
    if indices.ndim == 0:
        coefficients = edge_coefficients_for_node(lora_memory, indices)
    elif indices.ndim == 1 and indices.shape[0] == batch_size:
        coefficients = jax.vmap(
            lambda index: edge_coefficients_for_node(lora_memory, index)
        )(indices)
    else:
        raise ValueError("node_index must be scalar or have one index per prompt row")
    index_array = np.asarray(indices, dtype=np.int64).reshape(-1)
    valid_nodes = np.asarray(lora_memory.valid_node_mask, dtype=np.bool_)
    if np.any(index_array < 0) or np.any(index_array >= valid_nodes.shape[0]):
        raise ValueError("node_index is outside packed node capacity")
    if np.any(~valid_nodes[index_array]):
        raise ValueError("node_index selects an invalid padded node")
    return coefficients

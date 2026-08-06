"""Deterministic KV-cached greedy generation for plain-JAX GPT-Neo."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from apm.lm.config import GptNeoConfig
from apm.lm.gpt_neo import (
    apply_gpt_neo_cached_token,
    prefill_gpt_neo_cache,
)
from apm.lm.lora import LoraConfig
from apm.lm.lora_memory import PackedLoraMemory, edge_coefficients_for_node
from apm.lm.parameters import GptNeoParams


_GENERATION_SEQUENCE_BUCKET_SIZE = 32


@partial(
    jax.jit,
    static_argnames=("config", "cache_width", "lora_config"),
)
def _compiled_prefill(
    params: GptNeoParams,
    token_ids: jax.Array,
    attention_mask: jax.Array,
    lora_memory: PackedLoraMemory | None,
    edge_coefficients: jax.Array | None,
    *,
    config: GptNeoConfig,
    cache_width: int,
    lora_config: LoraConfig | None,
):
    result = prefill_gpt_neo_cache(
        params,
        config,
        token_ids,
        attention_mask,
        cache_width,
        lora_memory=lora_memory,
        edge_coefficients=edge_coefficients,
        lora_config=lora_config,
    )
    return jnp.argmax(result.logits, axis=-1).astype(jnp.int32), result.cache


@partial(
    jax.jit,
    static_argnames=("config", "lora_config", "eos_token_id", "pad_token_id"),
)
def _compiled_decode(
    params: GptNeoParams,
    first_token_ids: jax.Array,
    cache,
    initial_lengths: jax.Array,
    initial_finished: jax.Array,
    max_new_tokens: jax.Array,
    lora_memory: PackedLoraMemory | None,
    edge_coefficients: jax.Array | None,
    *,
    config: GptNeoConfig,
    lora_config: LoraConfig | None,
    eos_token_id: int | None,
    pad_token_id: int,
):
    batch_size = first_token_ids.shape[0]
    continuation = jnp.full(
        (batch_size, config.max_position_embeddings),
        pad_token_id,
        dtype=jnp.int32,
    )
    initial_state = (
        first_token_ids,
        cache,
        initial_lengths,
        initial_finished,
        continuation,
    )

    def body(step, state):
        next_token_ids, current_cache, lengths, finished, generated = state
        within_budget = step < max_new_tokens
        active = (~finished) & within_budget
        positions = jnp.clip(lengths, 0, config.max_position_embeddings - 1)
        emitted = jnp.where(active, next_token_ids, pad_token_id)
        generated = generated.at[:, step].set(emitted)
        next_lengths = lengths + active.astype(jnp.int32)
        next_finished = (
            finished | (active & (next_token_ids == eos_token_id))
            if eos_token_id is not None
            else finished
        )
        should_advance = (step + 1 < max_new_tokens) & jnp.any(~next_finished)

        def advance(_):
            result = apply_gpt_neo_cached_token(
                params,
                config,
                next_token_ids,
                positions,
                active,
                current_cache,
                lora_memory=lora_memory,
                edge_coefficients=edge_coefficients,
                lora_config=lora_config,
            )
            selected = jnp.argmax(result.logits, axis=-1).astype(jnp.int32)
            return selected, result.cache

        selected, next_cache = jax.lax.cond(
            should_advance,
            advance,
            lambda _: (next_token_ids, current_cache),
            operand=None,
        )
        return selected, next_cache, next_lengths, next_finished, generated

    return jax.lax.fori_loop(
        0,
        config.max_position_embeddings,
        body,
        initial_state,
    )[-1]


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
    if max_new_tokens == 0:
        return token_ids
    edge_coefficients = _hard_edge_coefficients(
        lora_memory,
        lora_config,
        node_index,
        token_ids.shape[0],
    )
    batch_size, prompt_width = token_ids.shape
    output_width = prompt_width + max_new_tokens
    prefill_width = min(
        config.max_position_embeddings,
        _round_up(prompt_width, _GENERATION_SEQUENCE_BUCKET_SIZE),
    )
    prefill_ids = jnp.full(
        (batch_size, prefill_width),
        pad_token_id,
        dtype=jnp.int32,
    ).at[:, :prompt_width].set(token_ids)
    prefill_mask = jnp.zeros(
        (batch_size, prefill_width),
        dtype=jnp.bool_,
    ).at[:, :prompt_width].set(mask)
    next_token_ids, cache = _compiled_prefill(
        params,
        prefill_ids,
        prefill_mask,
        lora_memory,
        edge_coefficients,
        config=config,
        cache_width=config.max_position_embeddings,
        lora_config=lora_config,
    )
    initial_lengths = np.sum(
        np.asarray(mask, dtype=np.bool_),
        axis=1,
        dtype=np.int32,
    )
    rows = np.arange(batch_size, dtype=np.int32)
    prompt_array = np.asarray(token_ids, dtype=np.int32)
    initial_finished = (
        prompt_array[rows, initial_lengths - 1] == int(eos_token_id)
        if eos_token_id is not None
        else np.zeros((batch_size,), dtype=np.bool_)
    )
    continuation = np.asarray(
        _compiled_decode(
            params,
            next_token_ids,
            cache,
            initial_lengths,
            initial_finished,
            np.asarray(max_new_tokens, dtype=np.int32),
            lora_memory,
            edge_coefficients,
            config=config,
            lora_config=lora_config,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
    )[:, :max_new_tokens]
    generated_ids = np.full(
        (batch_size, output_width),
        pad_token_id,
        dtype=np.int32,
    )
    generated_ids[:, :prompt_width] = prompt_array
    for row, length in enumerate(initial_lengths):
        generated_ids[row, length : length + max_new_tokens] = continuation[row]
    return jnp.asarray(generated_ids)


def _round_up(value: int, multiple: int) -> int:
    """Round a positive width up without exceeding limits at the call site."""
    if value <= 0:
        raise ValueError("generation width must be positive")
    if multiple <= 0:
        raise ValueError("generation bucket size must be positive")
    return ((value + multiple - 1) // multiple) * multiple


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

"""Global and local self-attention for the plain-JAX GPT-Neo model."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from apm.lm.config import AttentionType, GptNeoConfig
from apm.lm.lora import LoraBlockBank, LoraConfig, LoraProjectionBank, apply_lora_linear
from apm.lm.parameters import AttentionParams, LinearParams


def apply_linear(params: LinearParams, inputs: jax.Array) -> jax.Array:
    """Apply one input-major linear kernel and its optional bias."""
    outputs = inputs @ params.kernel
    return outputs if params.bias is None else outputs + params.bias


def attention_pattern(
    sequence_length: int,
    attention_type: AttentionType,
    local_window_size: int,
) -> jax.Array:
    """Return the exact GPT-Neo structural attention mask for one layer."""
    query_positions = jnp.arange(sequence_length)[:, None]
    key_positions = jnp.arange(sequence_length)[None, :]
    causal = key_positions <= query_positions
    if attention_type == "global":
        return causal
    if attention_type == "local":
        return causal & (key_positions > query_positions - local_window_size)
    raise ValueError(f"unknown attention type: {attention_type}")


def apply_attention(
    params: AttentionParams,
    config: GptNeoConfig,
    hidden_states: jax.Array,
    attention_mask: jax.Array,
    attention_type: AttentionType,
    *,
    lora_block: LoraBlockBank | None = None,
    edge_coefficients: jax.Array | None = None,
    lora_config: LoraConfig | None = None,
    training: bool,
    probability_dropout_key: jax.Array | None,
    output_dropout_key: jax.Array | None,
) -> jax.Array:
    """Apply one GPT-Neo self-attention operation and output projection."""
    output, _, _ = apply_attention_with_key_values(
        params,
        config,
        hidden_states,
        attention_mask,
        attention_type,
        lora_block=lora_block,
        edge_coefficients=edge_coefficients,
        lora_config=lora_config,
        training=training,
        probability_dropout_key=probability_dropout_key,
        output_dropout_key=output_dropout_key,
    )
    return output


def apply_attention_with_key_values(
    params: AttentionParams,
    config: GptNeoConfig,
    hidden_states: jax.Array,
    attention_mask: jax.Array,
    attention_type: AttentionType,
    *,
    lora_block: LoraBlockBank | None = None,
    edge_coefficients: jax.Array | None = None,
    lora_config: LoraConfig | None = None,
    training: bool,
    probability_dropout_key: jax.Array | None,
    output_dropout_key: jax.Array | None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Apply attention and return the projected keys and values for caching."""
    batch_size, sequence_length, _ = hidden_states.shape
    query = _split_heads(
        _apply_projection(
            params.query,
            hidden_states,
            None if lora_block is None else lora_block.query,
            edge_coefficients,
            lora_config,
            None if lora_config is None else lora_config.target_mask.query,
        ),
        config,
    )
    key = _split_heads(
        _apply_projection(
            params.key,
            hidden_states,
            None if lora_block is None else lora_block.key,
            edge_coefficients,
            lora_config,
            None if lora_config is None else lora_config.target_mask.key,
        ),
        config,
    )
    value = _split_heads(
        _apply_projection(
            params.value,
            hidden_states,
            None if lora_block is None else lora_block.value,
            edge_coefficients,
            lora_config,
            None if lora_config is None else lora_config.target_mask.value,
        ),
        config,
    )
    scores = jnp.einsum(
        "bnqd,bnkd->bnqk",
        query.astype(jnp.float32),
        key.astype(jnp.float32),
    )
    structural_mask = attention_pattern(
        sequence_length,
        attention_type,
        config.local_window_size,
    )
    key_mask = jnp.asarray(attention_mask, dtype=jnp.bool_).reshape(
        batch_size,
        1,
        1,
        sequence_length,
    )
    allowed = structural_mask[None, None, :, :] & key_mask
    masked_scores = jnp.where(
        allowed,
        scores,
        jnp.asarray(jnp.finfo(jnp.float32).min, dtype=jnp.float32),
    )
    probabilities = jax.nn.softmax(masked_scores, axis=-1)
    probabilities = dropout(
        probabilities,
        config.attention_dropout,
        training=training,
        rng_key=probability_dropout_key,
    ).astype(value.dtype)
    attended = jnp.einsum("bnqk,bnkd->bnqd", probabilities, value)
    merged = attended.transpose(0, 2, 1, 3).reshape(
        batch_size,
        sequence_length,
        config.hidden_size,
    )
    projected = _apply_projection(
        params.output,
        merged,
        None if lora_block is None else lora_block.attention_output,
        edge_coefficients,
        lora_config,
        None if lora_config is None else lora_config.target_mask.attention_output,
    )
    return (
        dropout(
            projected,
            config.residual_dropout,
            training=training,
            rng_key=output_dropout_key,
        ),
        key,
        value,
    )


def apply_cached_attention(
    params: AttentionParams,
    config: GptNeoConfig,
    hidden_states: jax.Array,
    key_cache: jax.Array,
    value_cache: jax.Array,
    attention_mask: jax.Array,
    position_ids: jax.Array,
    active_mask: jax.Array,
    attention_type: AttentionType,
    *,
    lora_block: LoraBlockBank | None = None,
    edge_coefficients: jax.Array | None = None,
    lora_config: LoraConfig | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Apply one-token attention while updating a fixed-width KV cache."""
    batch_size, sequence_length, _ = hidden_states.shape
    if sequence_length != 1:
        raise ValueError("cached attention requires exactly one query token")
    expected_cache_shape = (
        batch_size,
        config.num_heads,
        attention_mask.shape[1],
        config.head_size,
    )
    if key_cache.shape != expected_cache_shape:
        raise ValueError("key cache shape does not match cached attention inputs")
    if value_cache.shape != expected_cache_shape:
        raise ValueError("value cache shape does not match cached attention inputs")
    if position_ids.shape != (batch_size,) or active_mask.shape != (batch_size,):
        raise ValueError("cached positions and active mask require one value per row")
    query = _split_heads(
        _apply_projection(
            params.query,
            hidden_states,
            None if lora_block is None else lora_block.query,
            edge_coefficients,
            lora_config,
            None if lora_config is None else lora_config.target_mask.query,
        ),
        config,
    )
    next_key = _split_heads(
        _apply_projection(
            params.key,
            hidden_states,
            None if lora_block is None else lora_block.key,
            edge_coefficients,
            lora_config,
            None if lora_config is None else lora_config.target_mask.key,
        ),
        config,
    )[:, :, 0, :]
    next_value = _split_heads(
        _apply_projection(
            params.value,
            hidden_states,
            None if lora_block is None else lora_block.value,
            edge_coefficients,
            lora_config,
            None if lora_config is None else lora_config.target_mask.value,
        ),
        config,
    )[:, :, 0, :]
    rows = jnp.arange(batch_size, dtype=jnp.int32)
    existing_keys = key_cache[rows, :, position_ids, :]
    existing_values = value_cache[rows, :, position_ids, :]
    updated_keys = key_cache.at[rows, :, position_ids, :].set(
        jnp.where(active_mask[:, None, None], next_key, existing_keys)
    )
    updated_values = value_cache.at[rows, :, position_ids, :].set(
        jnp.where(active_mask[:, None, None], next_value, existing_values)
    )
    scores = jnp.einsum(
        "bnqd,bnkd->bnqk",
        query.astype(jnp.float32),
        updated_keys.astype(jnp.float32),
    )
    key_positions = jnp.arange(attention_mask.shape[1], dtype=jnp.int32)[None, :]
    allowed = attention_mask & (key_positions <= position_ids[:, None])
    if attention_type == "local":
        allowed = allowed & (
            key_positions > position_ids[:, None] - config.local_window_size
        )
    elif attention_type != "global":
        raise ValueError(f"unknown attention type: {attention_type}")
    masked_scores = jnp.where(
        allowed[:, None, None, :],
        scores,
        jnp.asarray(jnp.finfo(jnp.float32).min, dtype=jnp.float32),
    )
    probabilities = jax.nn.softmax(masked_scores, axis=-1).astype(updated_values.dtype)
    attended = jnp.einsum("bnqk,bnkd->bnqd", probabilities, updated_values)
    merged = attended.transpose(0, 2, 1, 3).reshape(
        batch_size,
        1,
        config.hidden_size,
    )
    projected = _apply_projection(
        params.output,
        merged,
        None if lora_block is None else lora_block.attention_output,
        edge_coefficients,
        lora_config,
        None if lora_config is None else lora_config.target_mask.attention_output,
    )
    return projected, updated_keys, updated_values


def dropout(
    inputs: jax.Array,
    rate: float,
    *,
    training: bool,
    rng_key: jax.Array | None,
) -> jax.Array:
    """Apply stateless inverted Bernoulli dropout when requested."""
    if not training or rate == 0.0:
        return inputs
    if rng_key is None:
        raise ValueError("training with nonzero dropout requires rng_key")
    keep_probability = 1.0 - rate
    keep_mask = jax.random.bernoulli(rng_key, keep_probability, inputs.shape)
    return jnp.where(keep_mask, inputs / keep_probability, jnp.zeros_like(inputs))


def _split_heads(hidden_states: jax.Array, config: GptNeoConfig) -> jax.Array:
    batch_size, sequence_length, _ = hidden_states.shape
    return hidden_states.reshape(
        batch_size,
        sequence_length,
        config.num_heads,
        config.head_size,
    ).transpose(0, 2, 1, 3)


def _apply_projection(
    params: LinearParams,
    inputs: jax.Array,
    projection_bank: LoraProjectionBank | None,
    edge_coefficients: jax.Array | None,
    lora_config: LoraConfig | None,
    target_enabled: bool | None,
) -> jax.Array:
    lora_arguments_present = (
        projection_bank is not None,
        edge_coefficients is not None,
        lora_config is not None,
        target_enabled is not None,
    )
    if not any(lora_arguments_present):
        return apply_linear(params, inputs)
    if not all(lora_arguments_present):
        raise ValueError("attention LoRA arguments must be supplied together")
    assert projection_bank is not None
    assert edge_coefficients is not None
    assert lora_config is not None
    assert target_enabled is not None
    return apply_lora_linear(
        params,
        inputs,
        projection_bank,
        edge_coefficients,
        lora_config.scale,
        target_enabled,
    )

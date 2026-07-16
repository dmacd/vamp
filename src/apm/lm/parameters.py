"""Typed parameter PyTrees and initialization for plain-JAX GPT-Neo."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike

from apm.lm.config import GptNeoConfig


class LinearParams(NamedTuple):
    """Kernel and optional bias for one linear projection."""

    kernel: jax.Array
    bias: jax.Array | None


class LayerNormParams(NamedTuple):
    """Scale and bias for one layer-normalization operation."""

    scale: jax.Array
    bias: jax.Array


class AttentionParams(NamedTuple):
    """Query, key, value, and output projection parameters."""

    query: LinearParams
    key: LinearParams
    value: LinearParams
    output: LinearParams


class MlpParams(NamedTuple):
    """Input and output projections for one transformer MLP."""

    input_projection: LinearParams
    output_projection: LinearParams


class TransformerBlockParams(NamedTuple):
    """Pre-normalization, attention, and MLP parameters for one block."""

    attention_norm: LayerNormParams
    attention: AttentionParams
    mlp_norm: LayerNormParams
    mlp: MlpParams


class GptNeoParams(NamedTuple):
    """Complete GPT-Neo parameter tree with tied token/output embeddings."""

    token_embedding: jax.Array
    position_embedding: jax.Array
    blocks: tuple[TransformerBlockParams, ...]
    final_norm: LayerNormParams


def init_gpt_neo_params(
    rng_key: jax.Array,
    config: GptNeoConfig,
    *,
    dtype: DTypeLike = jnp.float32,
) -> GptNeoParams:
    """Initialize GPT-Neo parameters using the checkpoint-compatible normal scheme."""
    keys = jax.random.split(rng_key, 2 + 6 * config.num_layers)
    token_embedding = _normal(
        keys[0],
        (config.vocab_size, config.hidden_size),
        config.initializer_range,
        dtype,
    )
    position_embedding = _normal(
        keys[1],
        (config.max_position_embeddings, config.hidden_size),
        config.initializer_range,
        dtype,
    )
    blocks = tuple(
        _init_block(
            keys[2 + 6 * layer_index : 2 + 6 * (layer_index + 1)],
            config,
            dtype,
        )
        for layer_index in range(config.num_layers)
    )
    return GptNeoParams(
        token_embedding=token_embedding,
        position_embedding=position_embedding,
        blocks=blocks,
        final_norm=_init_layer_norm(config.hidden_size, dtype),
    )


def _init_block(
    keys: jax.Array,
    config: GptNeoConfig,
    dtype: DTypeLike,
) -> TransformerBlockParams:
    return TransformerBlockParams(
        attention_norm=_init_layer_norm(config.hidden_size, dtype),
        attention=AttentionParams(
            query=_init_linear(keys[0], config.hidden_size, config.hidden_size, False, config, dtype),
            key=_init_linear(keys[1], config.hidden_size, config.hidden_size, False, config, dtype),
            value=_init_linear(keys[2], config.hidden_size, config.hidden_size, False, config, dtype),
            output=_init_linear(keys[3], config.hidden_size, config.hidden_size, True, config, dtype),
        ),
        mlp_norm=_init_layer_norm(config.hidden_size, dtype),
        mlp=MlpParams(
            input_projection=_init_linear(
                keys[4],
                config.hidden_size,
                config.intermediate_size,
                True,
                config,
                dtype,
            ),
            output_projection=_init_linear(
                keys[5],
                config.intermediate_size,
                config.hidden_size,
                True,
                config,
                dtype,
            ),
        ),
    )


def _init_linear(
    rng_key: jax.Array,
    input_size: int,
    output_size: int,
    use_bias: bool,
    config: GptNeoConfig,
    dtype: DTypeLike,
) -> LinearParams:
    return LinearParams(
        kernel=_normal(
            rng_key,
            (input_size, output_size),
            config.initializer_range,
            dtype,
        ),
        bias=jnp.zeros((output_size,), dtype=dtype) if use_bias else None,
    )


def _init_layer_norm(hidden_size: int, dtype: DTypeLike) -> LayerNormParams:
    return LayerNormParams(
        scale=jnp.ones((hidden_size,), dtype=dtype),
        bias=jnp.zeros((hidden_size,), dtype=dtype),
    )


def _normal(
    rng_key: jax.Array,
    shape: tuple[int, ...],
    standard_deviation: float,
    dtype: DTypeLike,
) -> jax.Array:
    return jax.random.normal(rng_key, shape, dtype=dtype) * jnp.asarray(
        standard_deviation,
        dtype=dtype,
    )

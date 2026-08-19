"""Pure forward functions for the typed plain-JAX GPT-Neo model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp

from apm.lm.attention import (
    apply_attention,
    apply_attention_with_key_values,
    apply_cached_attention,
    apply_linear,
    dropout,
)
from apm.lm.compact_lora_memory import CompactLoraMemory
from apm.lm.config import GptNeoConfig
from apm.lm.lora import LoraBlockBank, LoraConfig, apply_lora_linear
from apm.lm.lora_memory import PackedLoraMemory
from apm.lm.parameters import GptNeoParams, LayerNormParams, LinearParams

CaptureLocation = Literal["post_attention", "post_mlp"]


@dataclass(frozen=True)
class CapturePoint:
    """One transformer residual point requested from a forward pass."""

    layer_index: int
    location: CaptureLocation

    def __post_init__(self) -> None:
        if self.layer_index < 0:
            raise ValueError("capture layer_index must be nonnegative")
        if self.location not in ("post_attention", "post_mlp"):
            raise ValueError(f"unknown capture location: {self.location}")


@dataclass(frozen=True)
class CaptureSpec:
    """Static ordered transformer residual points returned by a forward pass."""

    points: tuple[CapturePoint, ...] = ()


class ForwardResult(NamedTuple):
    """Fixed-structure outputs from a GPT-Neo forward pass."""

    logits: jax.Array
    final_hidden: jax.Array
    captured_hidden: tuple[jax.Array, ...]


class GptNeoKvCache(NamedTuple):
    """Fixed-width per-layer attention keys, values, and valid positions."""

    keys: tuple[jax.Array, ...]
    values: tuple[jax.Array, ...]
    attention_mask: jax.Array


class CachedForwardResult(NamedTuple):
    """Next-token logits paired with an updated autoregressive KV cache."""

    logits: jax.Array
    cache: GptNeoKvCache


def embed_tokens(
    params: GptNeoParams,
    token_ids: jax.Array,
    position_ids: jax.Array,
) -> jax.Array:
    """Add token and absolute-position embeddings."""
    return params.token_embedding[token_ids] + params.position_embedding[position_ids]


def apply_gpt_neo_embeddings(
    params: GptNeoParams,
    config: GptNeoConfig,
    input_embeddings: jax.Array,
    attention_mask: jax.Array,
    *,
    lora_memory: PackedLoraMemory | CompactLoraMemory | None = None,
    edge_coefficients: jax.Array | None = None,
    lora_config: LoraConfig | None = None,
    capture: CaptureSpec = CaptureSpec(),
    training: bool = False,
    rng_key: jax.Array | None = None,
) -> ForwardResult:
    """Apply GPT-Neo blocks to precomputed token-plus-position embeddings."""
    if input_embeddings.ndim != 3:
        raise ValueError("input_embeddings must have shape [batch, sequence, hidden]")
    if input_embeddings.shape[-1] != config.hidden_size:
        raise ValueError("input embedding width does not match hidden_size")
    if attention_mask.shape != input_embeddings.shape[:2]:
        raise ValueError("attention_mask must match input batch and sequence dimensions")
    if input_embeddings.shape[1] > config.max_position_embeddings:
        raise ValueError("sequence length exceeds max_position_embeddings")
    if len(params.blocks) != config.num_layers:
        raise ValueError("parameter block count does not match num_layers")
    invalid_capture_layers = tuple(
        point.layer_index
        for point in capture.points
        if point.layer_index >= config.num_layers
    )
    if invalid_capture_layers:
        raise ValueError(f"capture layers are outside the model: {invalid_capture_layers}")
    if training and config.uses_dropout and rng_key is None:
        raise ValueError("training with configured dropout requires rng_key")
    effective_edge_coefficients = _resolve_lora_coefficients(
        lora_memory,
        edge_coefficients,
        lora_config,
        config,
        input_embeddings.shape[0],
    )

    effective_key = jax.random.PRNGKey(0) if rng_key is None else rng_key
    dropout_keys = jax.random.split(effective_key, 1 + 3 * config.num_layers)
    hidden_states = dropout(
        input_embeddings,
        config.embedding_dropout,
        training=training,
        rng_key=dropout_keys[0],
    )
    layer_captures: list[tuple[jax.Array, jax.Array]] = []
    for layer_index, (block, attention_type) in enumerate(
        zip(params.blocks, config.attention_types)
    ):
        lora_block = (
            None
            if lora_memory is None
            else lora_memory.edge_bank.blocks[layer_index]
        )
        key_offset = 1 + 3 * layer_index
        attention_input = apply_layer_norm(
            block.attention_norm,
            hidden_states,
            config.layer_norm_epsilon,
        )
        attention_output = apply_attention(
            block.attention,
            config,
            attention_input,
            attention_mask,
            attention_type,
            lora_block=lora_block,
            edge_coefficients=effective_edge_coefficients,
            lora_config=lora_config,
            training=training,
            probability_dropout_key=dropout_keys[key_offset],
            output_dropout_key=dropout_keys[key_offset + 1],
        )
        post_attention = hidden_states + attention_output
        mlp_input = apply_layer_norm(
            block.mlp_norm,
            post_attention,
            config.layer_norm_epsilon,
        )
        mlp_hidden = gelu_new(
            _apply_mlp_projection(
                block.mlp.input_projection,
                mlp_input,
                lora_block,
                effective_edge_coefficients,
                lora_config,
                input_projection=True,
            )
        )
        mlp_output = dropout(
            _apply_mlp_projection(
                block.mlp.output_projection,
                mlp_hidden,
                lora_block,
                effective_edge_coefficients,
                lora_config,
                input_projection=False,
            ),
            config.residual_dropout,
            training=training,
            rng_key=dropout_keys[key_offset + 2],
        )
        hidden_states = post_attention + mlp_output
        layer_captures.append((post_attention, hidden_states))

    final_hidden = apply_layer_norm(
        params.final_norm,
        hidden_states,
        config.layer_norm_epsilon,
    )
    logits = jnp.einsum("bth,vh->btv", final_hidden, params.token_embedding)
    captured_hidden = tuple(
        layer_captures[point.layer_index][0 if point.location == "post_attention" else 1]
        for point in capture.points
    )
    return ForwardResult(
        logits=logits,
        final_hidden=final_hidden,
        captured_hidden=captured_hidden,
    )


def apply_gpt_neo(
    params: GptNeoParams,
    config: GptNeoConfig,
    token_ids: jax.Array,
    attention_mask: jax.Array,
    *,
    position_ids: jax.Array | None = None,
    lora_memory: PackedLoraMemory | CompactLoraMemory | None = None,
    edge_coefficients: jax.Array | None = None,
    lora_config: LoraConfig | None = None,
    capture: CaptureSpec = CaptureSpec(),
    training: bool = False,
    rng_key: jax.Array | None = None,
) -> ForwardResult:
    """Embed token IDs and apply the complete GPT-Neo language model."""
    if token_ids.ndim != 2:
        raise ValueError("token_ids must have shape [batch, sequence]")
    sequence_length = token_ids.shape[1]
    resolved_position_ids = (
        jnp.arange(sequence_length, dtype=jnp.int32)[None, :]
        if position_ids is None
        else position_ids
    )
    embeddings = embed_tokens(params, token_ids, resolved_position_ids)
    return apply_gpt_neo_embeddings(
        params,
        config,
        embeddings,
        attention_mask,
        lora_memory=lora_memory,
        edge_coefficients=edge_coefficients,
        lora_config=lora_config,
        capture=capture,
        training=training,
        rng_key=rng_key,
    )


def prefill_gpt_neo_cache(
    params: GptNeoParams,
    config: GptNeoConfig,
    token_ids: jax.Array,
    attention_mask: jax.Array,
    cache_width: int,
    *,
    lora_memory: PackedLoraMemory | CompactLoraMemory | None = None,
    edge_coefficients: jax.Array | None = None,
    lora_config: LoraConfig | None = None,
) -> CachedForwardResult:
    """Encode a right-padded prompt once and initialize a fixed-width KV cache."""
    if token_ids.ndim != 2 or attention_mask.shape != token_ids.shape:
        raise ValueError("prefill token IDs and attention mask must share [batch, sequence]")
    batch_size, sequence_length = token_ids.shape
    if sequence_length > cache_width or cache_width > config.max_position_embeddings:
        raise ValueError("prefill sequence and cache widths exceed model capacity")
    if len(params.blocks) != config.num_layers:
        raise ValueError("parameter block count does not match num_layers")
    effective_edge_coefficients = _resolve_lora_coefficients(
        lora_memory,
        edge_coefficients,
        lora_config,
        config,
        batch_size,
    )
    position_ids = jnp.arange(sequence_length, dtype=jnp.int32)[None, :]
    hidden_states = embed_tokens(params, token_ids, position_ids)
    cached_keys: list[jax.Array] = []
    cached_values: list[jax.Array] = []
    cache_padding = ((0, 0), (0, 0), (0, cache_width - sequence_length), (0, 0))
    for layer_index, (block, attention_type) in enumerate(
        zip(params.blocks, config.attention_types)
    ):
        lora_block = (
            None
            if lora_memory is None
            else lora_memory.edge_bank.blocks[layer_index]
        )
        attention_input = apply_layer_norm(
            block.attention_norm,
            hidden_states,
            config.layer_norm_epsilon,
        )
        attention_output, keys, values = apply_attention_with_key_values(
            block.attention,
            config,
            attention_input,
            attention_mask,
            attention_type,
            lora_block=lora_block,
            edge_coefficients=effective_edge_coefficients,
            lora_config=lora_config,
            training=False,
            probability_dropout_key=None,
            output_dropout_key=None,
        )
        cached_keys.append(jnp.pad(keys, cache_padding))
        cached_values.append(jnp.pad(values, cache_padding))
        post_attention = hidden_states + attention_output
        mlp_input = apply_layer_norm(
            block.mlp_norm,
            post_attention,
            config.layer_norm_epsilon,
        )
        mlp_hidden = gelu_new(
            _apply_mlp_projection(
                block.mlp.input_projection,
                mlp_input,
                lora_block,
                effective_edge_coefficients,
                lora_config,
                input_projection=True,
            )
        )
        hidden_states = post_attention + _apply_mlp_projection(
            block.mlp.output_projection,
            mlp_hidden,
            lora_block,
            effective_edge_coefficients,
            lora_config,
            input_projection=False,
        )
    final_hidden = apply_layer_norm(
        params.final_norm,
        hidden_states,
        config.layer_norm_epsilon,
    )
    lengths = jnp.sum(attention_mask, axis=1, dtype=jnp.int32)
    rows = jnp.arange(batch_size, dtype=jnp.int32)
    last_hidden = final_hidden[rows, lengths - 1]
    logits = jnp.einsum("bh,vh->bv", last_hidden, params.token_embedding)
    padded_mask = jnp.pad(
        attention_mask,
        ((0, 0), (0, cache_width - sequence_length)),
    )
    return CachedForwardResult(
        logits,
        GptNeoKvCache(tuple(cached_keys), tuple(cached_values), padded_mask),
    )


def apply_gpt_neo_cached_token(
    params: GptNeoParams,
    config: GptNeoConfig,
    token_ids: jax.Array,
    position_ids: jax.Array,
    active_mask: jax.Array,
    cache: GptNeoKvCache,
    *,
    lora_memory: PackedLoraMemory | CompactLoraMemory | None = None,
    edge_coefficients: jax.Array | None = None,
    lora_config: LoraConfig | None = None,
) -> CachedForwardResult:
    """Advance an autoregressive batch by one token using cached attention state."""
    if token_ids.ndim != 1:
        raise ValueError("cached token IDs require shape [batch]")
    batch_size = token_ids.shape[0]
    if position_ids.shape != (batch_size,) or active_mask.shape != (batch_size,):
        raise ValueError("cached positions and active mask require one value per row")
    if len(cache.keys) != config.num_layers or len(cache.values) != config.num_layers:
        raise ValueError("KV cache layer count does not match the model")
    if cache.attention_mask.shape[0] != batch_size:
        raise ValueError("KV cache batch size does not match token IDs")
    effective_edge_coefficients = _resolve_lora_coefficients(
        lora_memory,
        edge_coefficients,
        lora_config,
        config,
        batch_size,
    )
    rows = jnp.arange(batch_size, dtype=jnp.int32)
    existing_mask = cache.attention_mask[rows, position_ids]
    updated_attention_mask = cache.attention_mask.at[rows, position_ids].set(
        jnp.where(active_mask, True, existing_mask)
    )
    hidden_states = (
        params.token_embedding[token_ids] + params.position_embedding[position_ids]
    )[:, None, :]
    updated_keys: list[jax.Array] = []
    updated_values: list[jax.Array] = []
    for layer_index, (block, attention_type) in enumerate(
        zip(params.blocks, config.attention_types)
    ):
        lora_block = (
            None
            if lora_memory is None
            else lora_memory.edge_bank.blocks[layer_index]
        )
        attention_input = apply_layer_norm(
            block.attention_norm,
            hidden_states,
            config.layer_norm_epsilon,
        )
        attention_output, layer_keys, layer_values = apply_cached_attention(
            block.attention,
            config,
            attention_input,
            cache.keys[layer_index],
            cache.values[layer_index],
            updated_attention_mask,
            position_ids,
            active_mask,
            attention_type,
            lora_block=lora_block,
            edge_coefficients=effective_edge_coefficients,
            lora_config=lora_config,
        )
        updated_keys.append(layer_keys)
        updated_values.append(layer_values)
        post_attention = hidden_states + attention_output
        mlp_input = apply_layer_norm(
            block.mlp_norm,
            post_attention,
            config.layer_norm_epsilon,
        )
        mlp_hidden = gelu_new(
            _apply_mlp_projection(
                block.mlp.input_projection,
                mlp_input,
                lora_block,
                effective_edge_coefficients,
                lora_config,
                input_projection=True,
            )
        )
        hidden_states = post_attention + _apply_mlp_projection(
            block.mlp.output_projection,
            mlp_hidden,
            lora_block,
            effective_edge_coefficients,
            lora_config,
            input_projection=False,
        )
    final_hidden = apply_layer_norm(
        params.final_norm,
        hidden_states,
        config.layer_norm_epsilon,
    )[:, 0, :]
    logits = jnp.einsum("bh,vh->bv", final_hidden, params.token_embedding)
    return CachedForwardResult(
        logits,
        GptNeoKvCache(
            tuple(updated_keys),
            tuple(updated_values),
            updated_attention_mask,
        ),
    )


def apply_layer_norm(
    params: LayerNormParams,
    inputs: jax.Array,
    epsilon: float,
) -> jax.Array:
    """Apply population-statistic layer normalization over the hidden axis."""
    mean = jnp.mean(inputs, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(inputs - mean), axis=-1, keepdims=True)
    normalized = (inputs - mean) * jax.lax.rsqrt(variance + epsilon)
    return normalized * params.scale + params.bias


def gelu_new(inputs: jax.Array) -> jax.Array:
    """Apply the tanh-approximated GELU used by GPT-Neo checkpoints."""
    coefficient = jnp.sqrt(jnp.asarray(2.0 / jnp.pi, dtype=inputs.dtype))
    return 0.5 * inputs * (
        1.0
        + jnp.tanh(
            coefficient
            * (inputs + jnp.asarray(0.044715, dtype=inputs.dtype) * jnp.power(inputs, 3))
        )
    )


def _resolve_lora_coefficients(
    lora_memory: PackedLoraMemory | CompactLoraMemory | None,
    edge_coefficients: jax.Array | None,
    lora_config: LoraConfig | None,
    model_config: GptNeoConfig,
    batch_size: int,
) -> jax.Array | None:
    arguments_present = (
        lora_memory is not None,
        edge_coefficients is not None,
        lora_config is not None,
    )
    if not any(arguments_present):
        return None
    if not all(arguments_present):
        raise ValueError("lora_memory, edge_coefficients, and lora_config must be supplied together")
    assert lora_memory is not None
    assert edge_coefficients is not None
    assert lora_config is not None
    if len(lora_memory.edge_bank.blocks) != model_config.num_layers:
        raise ValueError("LoRA bank block count does not match num_layers")
    if isinstance(lora_memory, PackedLoraMemory):
        if lora_memory.node_path_matrix.ndim != 2:
            raise ValueError("LoRA node path matrix must have rank two")
        max_nodes, max_edges = lora_memory.node_path_matrix.shape
        if lora_memory.valid_node_mask.shape != (max_nodes,):
            raise ValueError("LoRA valid-node mask does not match node capacity")
        if lora_memory.valid_edge_mask.shape != (max_edges,):
            raise ValueError("LoRA valid-edge mask does not match edge capacity")
        bank_rank = 3
    elif isinstance(lora_memory, CompactLoraMemory):
        batch_count, max_edges = lora_memory.valid_edge_mask.shape
        if batch_count != batch_size:
            raise ValueError("compact LoRA memory must match the input batch")
        bank_rank = 4
    else:
        raise TypeError("lora_memory must be packed or compact LoRA memory")
    coefficients = jnp.asarray(edge_coefficients, dtype=jnp.float32)
    if coefficients.ndim not in (1, 2) or coefficients.shape[-1] != max_edges:
        raise ValueError(
            "edge_coefficients must have shape [edges] or [batch, edges] matching capacity"
        )
    if coefficients.ndim == 2 and coefficients.shape[0] != batch_size:
        raise ValueError("batched edge coefficients must match the input batch")
    if bank_rank == 4 and coefficients.ndim != 2:
        raise ValueError("compact LoRA memory requires batched edge coefficients")
    for lora_block in lora_memory.edge_bank.blocks:
        for projection_bank in lora_block:
            if (
                projection_bank.left.ndim != bank_rank
                or projection_bank.right.ndim != bank_rank
            ):
                raise ValueError("LoRA projection-bank rank does not match memory kind")
            edge_axis = bank_rank - 3
            rank_axis = bank_rank - 1
            right_rank_axis = bank_rank - 2
            if (
                projection_bank.left.shape[edge_axis] != max_edges
                or projection_bank.right.shape[edge_axis] != max_edges
            ):
                raise ValueError("LoRA projection edge capacity does not match memory")
            if bank_rank == 4 and (
                projection_bank.left.shape[0] != batch_size
                or projection_bank.right.shape[0] != batch_size
            ):
                raise ValueError("compact LoRA factor bank must match input batch")
            if projection_bank.left.shape[rank_axis] != lora_config.rank or projection_bank.right.shape[right_rank_axis] != lora_config.rank:
                raise ValueError("LoRA projection rank does not match lora_config")
    return coefficients * lora_memory.valid_edge_mask.astype(jnp.float32)


def _apply_mlp_projection(
    params: LinearParams,
    inputs: jax.Array,
    lora_block: LoraBlockBank | None,
    edge_coefficients: jax.Array | None,
    lora_config: LoraConfig | None,
    *,
    input_projection: bool,
) -> jax.Array:
    if lora_block is None:
        return apply_linear(params, inputs)
    if edge_coefficients is None or lora_config is None:
        raise ValueError("MLP LoRA arguments must be supplied together")
    projection_bank = lora_block.mlp_input if input_projection else lora_block.mlp_output
    target_enabled = (
        lora_config.target_mask.mlp_input
        if input_projection
        else lora_config.target_mask.mlp_output
    )
    return apply_lora_linear(
        params,
        inputs,
        projection_bank,
        edge_coefficients,
        lora_config.scale,
        target_enabled,
    )

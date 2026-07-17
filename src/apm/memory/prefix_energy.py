"""Exhaustive task-free routing by normalized prefix language-model loss."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from apm.continual.language_tasks import AddressResult, RouterBatch
from apm.lm.config import GptNeoConfig
from apm.lm.evaluation import (
    evaluation_microbatch_slices,
    validate_evaluation_microbatch_size,
)
from apm.lm.gpt_neo import apply_gpt_neo
from apm.lm.lora import LoraConfig
from apm.lm.lora_memory import PackedLoraMemory, edge_coefficients_for_node
from apm.lm.losses import per_token_nll
from apm.lm.parameters import GptNeoParams


def exhaustive_prefix_nll_address(
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    packed_memory: PackedLoraMemory,
    lora_config: LoraConfig,
    prefix_batch: RouterBatch,
    *,
    evaluation_microbatch_size: int | None = None,
) -> AddressResult:
    """Select the lowest normalized prefix-NLL node for every batch row."""
    _validate_prefix_batch(prefix_batch)
    microbatch_size = validate_evaluation_microbatch_size(
        evaluation_microbatch_size
    )
    if microbatch_size is not None:
        return _microbatched_exhaustive_prefix_nll_address(
            base_params,
            model_config,
            packed_memory,
            lora_config,
            prefix_batch,
            microbatch_size,
        )
    return exhaustive_prefix_nll_core(
        base_params,
        model_config,
        packed_memory,
        lora_config,
        jnp.asarray(prefix_batch.input_ids),
        jnp.asarray(prefix_batch.attention_mask),
        jnp.asarray(prefix_batch.target_ids),
        jnp.asarray(prefix_batch.loss_mask),
    )


def exhaustive_prefix_nll_core(
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    packed_memory: PackedLoraMemory,
    lora_config: LoraConfig,
    input_ids: jax.Array,
    attention_mask: jax.Array,
    target_ids: jax.Array,
    loss_mask: jax.Array,
) -> AddressResult:
    """Compute fixed-capacity router outputs from validated prefix arrays."""
    _validate_array_shapes(input_ids, attention_mask, target_ids, loss_mask)
    input_ids = jnp.asarray(input_ids)
    attention_mask = jnp.asarray(attention_mask, dtype=jnp.bool_)
    target_ids = jnp.asarray(target_ids)
    float_loss_mask = jnp.asarray(loss_mask, dtype=jnp.float32)
    active_token_counts = jnp.sum(float_loss_mask, axis=-1)

    def score_node(node_index: jax.Array) -> jax.Array:
        return _prefix_nll_for_node(
            base_params,
            model_config,
            packed_memory,
            lora_config,
            input_ids,
            attention_mask,
            target_ids,
            float_loss_mask,
            active_token_counts,
            node_index,
        )

    max_nodes = packed_memory.node_path_matrix.shape[0]
    scores_by_node = jax.vmap(score_node)(
        jnp.arange(max_nodes, dtype=jnp.int32)
    )
    return _address_result_from_node_scores(
        scores_by_node.T,
        packed_memory.valid_node_mask,
    )


def _microbatched_exhaustive_prefix_nll_address(
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    packed_memory: PackedLoraMemory,
    lora_config: LoraConfig,
    prefix_batch: RouterBatch,
    evaluation_microbatch_size: int,
) -> AddressResult:
    """Score examples and valid nodes sequentially to bound vocabulary logits."""
    row_count = prefix_batch.input_ids.shape[0]
    node_capacity = packed_memory.node_path_matrix.shape[0]
    valid_node_mask = np.asarray(packed_memory.valid_node_mask, dtype=np.bool_)
    scores = np.full((row_count, node_capacity), np.inf, dtype=np.float32)
    for row_slice in evaluation_microbatch_slices(
        row_count,
        evaluation_microbatch_size,
    ):
        input_ids = jnp.asarray(prefix_batch.input_ids[row_slice])
        attention_mask = jnp.asarray(
            prefix_batch.attention_mask[row_slice],
            dtype=jnp.bool_,
        )
        target_ids = jnp.asarray(prefix_batch.target_ids[row_slice])
        loss_mask = jnp.asarray(
            prefix_batch.loss_mask[row_slice],
            dtype=jnp.float32,
        )
        active_token_counts = jnp.sum(loss_mask, axis=-1)
        for node_index in np.flatnonzero(valid_node_mask):
            node_scores = _prefix_nll_for_node(
                base_params,
                model_config,
                packed_memory,
                lora_config,
                input_ids,
                attention_mask,
                target_ids,
                loss_mask,
                active_token_counts,
                jnp.asarray(node_index, dtype=jnp.int32),
            )
            scores[row_slice, node_index] = np.asarray(
                node_scores,
                dtype=np.float32,
            )
    return _address_result_from_node_scores(
        jnp.asarray(scores, dtype=jnp.float32),
        packed_memory.valid_node_mask,
    )


def _prefix_nll_for_node(
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    packed_memory: PackedLoraMemory,
    lora_config: LoraConfig,
    input_ids: jax.Array,
    attention_mask: jax.Array,
    target_ids: jax.Array,
    float_loss_mask: jax.Array,
    active_token_counts: jax.Array,
    node_index: jax.Array,
) -> jax.Array:
    edge_coefficients = edge_coefficients_for_node(packed_memory, node_index)
    logits = apply_gpt_neo(
        base_params,
        model_config,
        input_ids,
        attention_mask,
        lora_memory=packed_memory,
        edge_coefficients=edge_coefficients,
        lora_config=lora_config,
        training=False,
    ).logits
    token_losses = per_token_nll(logits, target_ids)
    return jnp.sum(token_losses * float_loss_mask, axis=-1) / active_token_counts


def _address_result_from_node_scores(
    scores_by_batch: jax.Array,
    valid_node_mask: jax.Array,
) -> AddressResult:
    """Derive fixed-capacity decisions from normalized lower-is-better NLL."""
    scores_by_batch = jnp.asarray(scores_by_batch, dtype=jnp.float32)
    valid_node_mask = jnp.asarray(valid_node_mask, dtype=jnp.bool_)
    node_scores = jnp.where(
        valid_node_mask[None, :],
        scores_by_batch,
        jnp.asarray(jnp.inf, dtype=jnp.float32),
    ).astype(jnp.float32)
    node_probabilities = jnp.where(
        valid_node_mask[None, :],
        jax.nn.softmax(-node_scores, axis=-1),
        jnp.asarray(0.0, dtype=jnp.float32),
    ).astype(jnp.float32)
    selected_indices = jnp.argmin(node_scores, axis=-1).astype(jnp.int32)
    sorted_scores = jnp.sort(node_scores, axis=-1)
    score_margin = (
        jnp.full((node_scores.shape[0],), jnp.inf, dtype=jnp.float32)
        if node_scores.shape[1] == 1
        else (sorted_scores[:, 1] - sorted_scores[:, 0]).astype(jnp.float32)
    )
    entropy_terms = jnp.where(
        node_probabilities > 0.0,
        node_probabilities * jnp.log(node_probabilities),
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    entropy = -jnp.sum(entropy_terms, axis=-1).astype(jnp.float32)
    return AddressResult(
        selected_indices=selected_indices,
        node_probabilities=node_probabilities,
        node_scores=node_scores,
        score_margin=score_margin,
        entropy=entropy,
    )


def _validate_prefix_batch(prefix_batch: RouterBatch) -> None:
    _validate_array_shapes(
        prefix_batch.input_ids,
        prefix_batch.attention_mask,
        prefix_batch.target_ids,
        prefix_batch.loss_mask,
    )
    if np.any(np.sum(np.asarray(prefix_batch.loss_mask, dtype=np.int32), axis=-1) == 0):
        raise ValueError("every prefix row must enable at least one loss token")


def _validate_array_shapes(
    input_ids: jax.Array | np.ndarray,
    attention_mask: jax.Array | np.ndarray,
    target_ids: jax.Array | np.ndarray,
    loss_mask: jax.Array | np.ndarray,
) -> None:
    shapes = {
        input_ids.shape,
        attention_mask.shape,
        target_ids.shape,
        loss_mask.shape,
    }
    if input_ids.ndim != 2 or len(shapes) != 1:
        raise ValueError("prefix arrays must share one rank-two [batch, sequence] shape")

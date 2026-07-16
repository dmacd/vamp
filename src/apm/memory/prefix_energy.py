"""Exhaustive task-free routing by normalized prefix language-model loss."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from apm.continual.language_tasks import AddressResult, RouterBatch
from apm.lm.config import GptNeoConfig
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
) -> AddressResult:
    """Select the lowest normalized prefix-NLL node for every batch row."""
    _validate_prefix_batch(prefix_batch)
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

    max_nodes = packed_memory.node_path_matrix.shape[0]
    scores_by_node = jax.vmap(score_node)(
        jnp.arange(max_nodes, dtype=jnp.int32)
    )
    valid_node_mask = jnp.asarray(packed_memory.valid_node_mask, dtype=jnp.bool_)
    node_scores = jnp.where(
        valid_node_mask[None, :],
        scores_by_node.T,
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
        if max_nodes == 1
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

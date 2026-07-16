"""Masked causal-language-model losses for GPT-Neo outputs."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def per_token_nll(logits: jax.Array, target_ids: jax.Array) -> jax.Array:
    """Return float32 negative log-likelihood for every aligned target token."""
    if logits.shape[:-1] != target_ids.shape:
        raise ValueError("target_ids must match the batch and sequence dimensions of logits")
    log_probabilities = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    target_log_probabilities = jnp.take_along_axis(
        log_probabilities,
        target_ids[..., None],
        axis=-1,
    )[..., 0]
    return -target_log_probabilities


def mean_token_nll(
    logits: jax.Array,
    target_ids: jax.Array,
    loss_mask: jax.Array,
) -> jax.Array:
    """Return NLL normalized by the number of active target transitions."""
    if loss_mask.shape != target_ids.shape:
        raise ValueError("loss_mask must match target_ids")
    mask = jnp.asarray(loss_mask, dtype=jnp.float32)
    losses = per_token_nll(logits, target_ids)
    return jnp.sum(losses * mask) / jnp.sum(mask)

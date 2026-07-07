"""Losses and diagnostics for Bernoulli-canvas VAEs."""

from __future__ import annotations

from typing import TypeAlias

import jax
import jax.numpy as jnp
import numpy as np

from apm.data.mnist.label_canvas import CANVAS_SIZE, DIGIT_SIZE, LABEL_PATCH_COLS, LABEL_PATCH_ROWS
from apm.models.mlp_vae import VaeParams, vae_forward

Array: TypeAlias = jax.Array


def negative_elbo(
    params: VaeParams,
    inputs: Array,
    targets: Array,
    rng_key: Array,
    beta: float,
    label_loss_weight: float,
    training: bool = True,
) -> tuple[Array, dict[str, Array]]:
    """Compute mean weighted negative ELBO and scalar diagnostics."""
    outputs = vae_forward(params, inputs, rng_key, training=training)
    reconstruction_per_example = bernoulli_bce_with_logits(outputs["logits"], targets).sum(axis=-1)
    label_bce_per_example = (
        bernoulli_bce_with_logits(outputs["logits"], targets) * label_patch_flat_mask(targets.shape[-1])
    ).sum(axis=-1)
    kl_per_example = standard_normal_kl(outputs["mu"], outputs["logvar"])
    loss_per_example = (
        reconstruction_per_example
        + beta * kl_per_example
        + label_loss_weight * label_bce_per_example
    )
    metrics = {
        "loss": jnp.mean(loss_per_example),
        "negative_elbo": jnp.mean(reconstruction_per_example + beta * kl_per_example),
        "reconstruction_bce": jnp.mean(reconstruction_per_example),
        "digit_bce": jnp.mean(digit_region_bce(outputs["logits"], targets)),
        "label_patch_bce": jnp.mean(label_bce_per_example),
        "kl": jnp.mean(kl_per_example),
    }
    return metrics["loss"], metrics


def per_example_negative_elbo(
    params: VaeParams,
    inputs: Array,
    targets: Array,
    rng_key: Array,
    beta: float,
    training: bool = False,
) -> Array:
    """Compute unweighted negative ELBO per example."""
    outputs = vae_forward(params, inputs, rng_key, training=training)
    reconstruction_per_example = bernoulli_bce_with_logits(outputs["logits"], targets).sum(axis=-1)
    kl_per_example = standard_normal_kl(outputs["mu"], outputs["logvar"])
    return reconstruction_per_example + beta * kl_per_example


def bernoulli_bce_with_logits(logits: Array, targets: Array) -> Array:
    """Compute numerically stable Bernoulli BCE from logits and binary targets."""
    return jnp.maximum(logits, 0.0) - logits * targets + jnp.log1p(jnp.exp(-jnp.abs(logits)))


def standard_normal_kl(mu: Array, logvar: Array) -> Array:
    """Compute KL(q(z|x) || N(0, I)) per example."""
    return -0.5 * jnp.sum(1.0 + logvar - jnp.square(mu) - jnp.exp(logvar), axis=-1)


def digit_region_bce(logits: Array, targets: Array) -> Array:
    """Compute summed BCE over the 28x28 digit region per example."""
    return (bernoulli_bce_with_logits(logits, targets) * digit_region_flat_mask(targets.shape[-1])).sum(axis=-1)


def flatten_canvases(canvases: np.ndarray) -> np.ndarray:
    """Flatten canvas batches to float32 [batch, pixels] arrays."""
    return np.asarray(canvases, dtype=np.float32).reshape(canvases.shape[0], -1)


def label_patch_flat_mask(input_dim: int = CANVAS_SIZE * CANVAS_SIZE) -> Array:
    """Return a flat 0/1 mask selecting the reserved label patch."""
    return jnp.asarray(_flat_mask(input_dim, LABEL_PATCH_ROWS, LABEL_PATCH_COLS), dtype=jnp.float32)


def digit_region_flat_mask(input_dim: int = CANVAS_SIZE * CANVAS_SIZE) -> Array:
    """Return a flat 0/1 mask selecting the 28x28 digit region."""
    return jnp.asarray(_flat_mask(input_dim, slice(0, DIGIT_SIZE), slice(0, DIGIT_SIZE)), dtype=jnp.float32)


def _flat_mask(input_dim: int, rows: slice, cols: slice) -> np.ndarray:
    canvas_side = int(np.sqrt(input_dim))
    if canvas_side * canvas_side != input_dim:
        raise ValueError(f"input_dim must be a square canvas size, got {input_dim}")
    mask = np.zeros((canvas_side, canvas_side), dtype=np.float32)
    mask[rows, cols] = 1.0
    return mask.reshape(-1)

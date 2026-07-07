"""Pure-JAX MLP VAE for flattened generative canvases."""

from __future__ import annotations

from typing import TypeAlias

import jax
import jax.numpy as jnp
from pyrsistent import PClass, field

Array: TypeAlias = jax.Array
LayerParams: TypeAlias = dict[str, Array]
LayerStack: TypeAlias = tuple[LayerParams, ...]
VaeParams: TypeAlias = dict[str, LayerParams | LayerStack]


class VaeConfig(PClass):
    """Immutable architecture config for a fully-connected VAE."""

    input_dim = field(type=int, initial=1024)
    latent_dim = field(type=int, initial=64)
    encoder_widths = field(type=tuple, initial=(512, 256))
    decoder_widths = field(type=tuple, initial=(256, 512))


def init_mlp_vae_params(rng_key: Array, config: VaeConfig) -> VaeParams:
    """Initialize encoder, latent heads, decoder, and output projection parameters."""
    encoder_key, mu_key, logvar_key, decoder_key, output_key = jax.random.split(rng_key, 5)
    encoder_dims = (config.input_dim,) + config.encoder_widths
    decoder_dims = (config.latent_dim,) + config.decoder_widths
    return {
        "encoder": init_mlp_layers(encoder_key, encoder_dims),
        "mu": init_dense_layer(mu_key, config.encoder_widths[-1], config.latent_dim),
        "logvar": init_dense_layer(logvar_key, config.encoder_widths[-1], config.latent_dim),
        "decoder": init_mlp_layers(decoder_key, decoder_dims),
        "output": init_dense_layer(output_key, config.decoder_widths[-1], config.input_dim),
    }


def init_mlp_layers(rng_key: Array, dims: tuple[int, ...]) -> LayerStack:
    """Initialize a stack of dense layers for adjacent dimensions."""
    keys = jax.random.split(rng_key, len(dims) - 1)
    return tuple(init_dense_layer(key, input_dim, output_dim) for key, input_dim, output_dim in zip(keys, dims, dims[1:]))


def init_dense_layer(rng_key: Array, input_dim: int, output_dim: int) -> LayerParams:
    """Initialize one dense layer with Glorot-uniform weights and zero bias."""
    limit = jnp.sqrt(jnp.asarray(6.0 / (input_dim + output_dim), dtype=jnp.float32))
    return {
        "weight": jax.random.uniform(
            rng_key,
            (input_dim, output_dim),
            minval=-limit,
            maxval=limit,
            dtype=jnp.float32,
        ),
        "bias": jnp.zeros((output_dim,), dtype=jnp.float32),
    }


def vae_forward(params: VaeParams, inputs: Array, rng_key: Array, training: bool = True) -> dict[str, Array]:
    """Run the VAE and return logits, latent statistics, and sampled latent state."""
    mu, logvar = encode(params, inputs)
    latent = sample_latent(rng_key, mu, logvar) if training else mu
    logits = decode(params, latent)
    return {"logits": logits, "mu": mu, "logvar": logvar, "latent": latent}


def encode(params: VaeParams, inputs: Array) -> tuple[Array, Array]:
    """Encode inputs to latent mean and log-variance."""
    hidden = apply_mlp(params["encoder"], inputs)
    return apply_dense(params["mu"], hidden), apply_dense(params["logvar"], hidden)


def decode(params: VaeParams, latent: Array) -> Array:
    """Decode latent samples to Bernoulli logits over the flattened canvas."""
    hidden = apply_mlp(params["decoder"], latent)
    return apply_dense(params["output"], hidden)


def sample_latent(rng_key: Array, mu: Array, logvar: Array) -> Array:
    """Sample latent variables with the reparameterization trick."""
    return mu + jnp.exp(0.5 * logvar) * jax.random.normal(rng_key, mu.shape, dtype=mu.dtype)


def apply_mlp(layers: LayerStack, inputs: Array) -> Array:
    """Apply dense ReLU layers."""
    return tuple_apply(layers, inputs)


def tuple_apply(layers: LayerStack, inputs: Array) -> Array:
    """Apply a stack of dense layers with ReLU activations after every layer."""
    hidden = inputs
    for layer in layers:
        hidden = jax.nn.relu(apply_dense(layer, hidden))
    return hidden


def apply_dense(layer: LayerParams, inputs: Array) -> Array:
    """Apply one dense layer."""
    return inputs @ layer["weight"] + layer["bias"]


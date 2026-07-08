"""Pure-JAX VAE architectures for flattened generative canvases."""

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
    """Immutable architecture config for MLP or convolutional VAEs."""

    architecture = field(type=str, initial="mlp")
    input_dim = field(type=int, initial=1024)
    latent_dim = field(type=int, initial=64)
    encoder_widths = field(type=tuple, initial=(512, 256))
    decoder_widths = field(type=tuple, initial=(256, 512))
    conv_channels = field(type=tuple, initial=(16, 32))
    conv_dense_width = field(type=int, initial=256)
    conv_kernel_size = field(type=int, initial=3)


def init_vae_params(rng_key: Array, config: VaeConfig) -> VaeParams:
    """Initialize VAE parameters for the configured architecture."""
    if config.architecture == "mlp":
        return init_mlp_vae_params(rng_key, config)
    if config.architecture == "conv":
        return init_conv_vae_params(rng_key, config)
    raise ValueError(f"unknown VAE architecture: {config.architecture}")


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


def init_conv_vae_params(rng_key: Array, config: VaeConfig) -> VaeParams:
    """Initialize a small convolutional VAE for 32x32 flattened canvases."""
    conv1_key, conv2_key, encoder_dense_key, mu_key, logvar_key, decoder_dense_key, deconv1_key, deconv2_key = (
        jax.random.split(rng_key, 8)
    )
    first_channels, second_channels = config.conv_channels
    flat_conv_dim = 8 * 8 * second_channels
    return {
        "conv_encoder": (
            init_conv_layer(conv1_key, 1, first_channels, config.conv_kernel_size),
            init_conv_layer(conv2_key, first_channels, second_channels, config.conv_kernel_size),
        ),
        "encoder_dense": init_dense_layer(encoder_dense_key, flat_conv_dim, config.conv_dense_width),
        "mu": init_dense_layer(mu_key, config.conv_dense_width, config.latent_dim),
        "logvar": init_dense_layer(logvar_key, config.conv_dense_width, config.latent_dim),
        "decoder_dense": init_dense_layer(decoder_dense_key, config.latent_dim, flat_conv_dim),
        "conv_decoder": (
            init_conv_layer(deconv1_key, second_channels, first_channels, config.conv_kernel_size),
            init_conv_layer(deconv2_key, first_channels, 1, config.conv_kernel_size),
        ),
    }


def init_mlp_layers(rng_key: Array, dims: tuple[int, ...]) -> LayerStack:
    """Initialize a stack of dense layers for adjacent dimensions."""
    keys = jax.random.split(rng_key, len(dims) - 1)
    return tuple(
        init_dense_layer(key, input_dim, output_dim)
        for key, input_dim, output_dim in zip(keys, dims, dims[1:])
    )


def init_conv_layer(rng_key: Array, input_channels: int, output_channels: int, kernel_size: int) -> LayerParams:
    """Initialize one NHWC/HWIO convolution layer with Glorot-uniform weights."""
    fan_in = kernel_size * kernel_size * input_channels
    fan_out = kernel_size * kernel_size * output_channels
    limit = jnp.sqrt(jnp.asarray(6.0 / (fan_in + fan_out), dtype=jnp.float32))
    return {
        "weight": jax.random.uniform(
            rng_key,
            (kernel_size, kernel_size, input_channels, output_channels),
            minval=-limit,
            maxval=limit,
            dtype=jnp.float32,
        ),
        "bias": jnp.zeros((output_channels,), dtype=jnp.float32),
    }


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
    if "conv_encoder" in params:
        hidden = encode_conv_hidden(params, inputs)
        return apply_dense(params["mu"], hidden), apply_dense(params["logvar"], hidden)
    hidden = apply_mlp(params["encoder"], inputs)
    return apply_dense(params["mu"], hidden), apply_dense(params["logvar"], hidden)


def decode(params: VaeParams, latent: Array) -> Array:
    """Decode latent samples to Bernoulli logits over the flattened canvas."""
    if "conv_decoder" in params:
        return decode_conv(params, latent)
    hidden = apply_mlp(params["decoder"], latent)
    return apply_dense(params["output"], hidden)


def sample_latent(rng_key: Array, mu: Array, logvar: Array) -> Array:
    """Sample latent variables with the reparameterization trick."""
    return mu + jnp.exp(0.5 * logvar) * jax.random.normal(rng_key, mu.shape, dtype=mu.dtype)


def apply_mlp(layers: LayerStack, inputs: Array) -> Array:
    """Apply dense ReLU layers."""
    return tuple_apply(layers, inputs)


def encode_conv_hidden(params: VaeParams, inputs: Array) -> Array:
    """Apply the convolutional encoder and return dense hidden activations."""
    image_batch = inputs.reshape((inputs.shape[0], 32, 32, 1))
    first_hidden = jax.nn.relu(apply_conv(params["conv_encoder"][0], image_batch, strides=(2, 2)))
    second_hidden = jax.nn.relu(apply_conv(params["conv_encoder"][1], first_hidden, strides=(2, 2)))
    flat_hidden = second_hidden.reshape((inputs.shape[0], -1))
    return jax.nn.relu(apply_dense(params["encoder_dense"], flat_hidden))


def decode_conv(params: VaeParams, latent: Array) -> Array:
    """Apply the convolutional decoder and return flattened canvas logits."""
    channels = params["conv_decoder"][0]["weight"].shape[2]
    hidden = jax.nn.relu(apply_dense(params["decoder_dense"], latent)).reshape(
        (latent.shape[0], 8, 8, channels)
    )
    first_hidden = jax.nn.relu(apply_conv(params["conv_decoder"][0], upsample_nearest(hidden), strides=(1, 1)))
    logits = apply_conv(params["conv_decoder"][1], upsample_nearest(first_hidden), strides=(1, 1))
    return logits.reshape((latent.shape[0], -1))


def apply_conv(layer: LayerParams, inputs: Array, strides: tuple[int, int]) -> Array:
    """Apply one NHWC/HWIO convolution layer."""
    return jax.lax.conv_general_dilated(
        inputs,
        layer["weight"],
        window_strides=strides,
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    ) + layer["bias"]


def upsample_nearest(inputs: Array) -> Array:
    """Double spatial resolution by nearest-neighbor repeat."""
    return jnp.repeat(jnp.repeat(inputs, 2, axis=1), 2, axis=2)


def tuple_apply(layers: LayerStack, inputs: Array) -> Array:
    """Apply a stack of dense layers with ReLU activations after every layer."""
    hidden = inputs
    for layer in layers:
        hidden = jax.nn.relu(apply_dense(layer, hidden))
    return hidden


def apply_dense(layer: LayerParams, inputs: Array) -> Array:
    """Apply one dense layer."""
    return inputs @ layer["weight"] + layer["bias"]

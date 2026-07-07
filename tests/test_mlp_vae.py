from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from apm.models import VaeConfig, init_mlp_vae_params, negative_elbo, vae_forward


def test_mlp_vae_initialization_and_forward_shapes() -> None:
    config = VaeConfig()
    params = init_mlp_vae_params(jax.random.PRNGKey(0), config)
    batch = jnp.zeros((3, config.input_dim), dtype=jnp.float32)
    outputs = vae_forward(params, batch, jax.random.PRNGKey(1), training=True)

    assert outputs["logits"].shape == (3, config.input_dim)
    assert outputs["mu"].shape == (3, config.latent_dim)
    assert outputs["logvar"].shape == (3, config.latent_dim)
    assert outputs["latent"].shape == (3, config.latent_dim)


def test_negative_elbo_is_finite() -> None:
    config = VaeConfig()
    params = init_mlp_vae_params(jax.random.PRNGKey(0), config)
    targets = jnp.asarray(np.zeros((2, config.input_dim), dtype=np.float32))
    loss, metrics = negative_elbo(
        params,
        targets,
        targets,
        jax.random.PRNGKey(2),
        beta=0.1,
        label_loss_weight=5.0,
    )

    assert bool(jnp.isfinite(loss))
    assert all(bool(jnp.isfinite(value)) for value in metrics.values())


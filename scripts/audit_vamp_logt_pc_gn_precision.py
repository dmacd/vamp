"""Print component-level float32/float64 differences for the fixed GN audit."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import jax
import jax.numpy as jnp
import math
import numpy as np

from apm.continual.artifacts import load_canonical_json, record_sha256
from apm.experiments.vamp_logt_pc_config import load_config
from apm.experiments.vamp_logt_pc_data import authenticate_and_load_pc_data, preflight_tables
from apm.experiments.vamp_logt_pc_training import SelectedPcProtocol, make_backend
from apm.models.fabricpc_density_backend import PcGaussNewtonScores, load_pc_model


def main() -> None:
    """Score the same eight images and latent states in both precisions."""
    config = load_config("configs/vamp_logt_pc_mnist/gauss_newton.yaml")
    if config.model_source_run_root is None:
        raise ValueError("GN config has no MAP model source")
    data = authenticate_and_load_pc_data(config)
    _train, heldout = preflight_tables(
        data,
        config.preflight.train_examples,
        config.preflight.heldout_examples,
    )
    selected_record = load_canonical_json(
        config.model_source_run_root / "preflight" / "summary.json"
    )["selected_protocol"]
    selected = SelectedPcProtocol(
        float(selected_record["image_precision"]),
        float(selected_record["hidden_precision"]),
        float(selected_record["inference_step_size"]),
    )
    model_id = record_sha256(
        {
            "hidden_precision": selected.hidden_precision,
            "image_precision": selected.image_precision,
            "inference_step_size": selected.inference_step_size,
            "schema_version": "vamp-logt-pc-map-preflight-model-v1",
        }
    )
    backend = make_backend(config, selected, 0)
    model = load_pc_model(
        config.model_source_run_root / "preflight" / "models" / model_id,
        backend,
        model_id,
        0,
    )
    images = heldout.images_float32[:8]
    settled = backend.settle_images(model.params, images)
    float32 = backend.gauss_newton_scores_from_settled(model.params, images, settled)
    with jax.experimental.enable_x64():
        float64 = backend.gauss_newton_scores_from_settled(
            model.params,
            images,
            settled,
            use_float64=True,
        )
        images32 = jnp.asarray(images, dtype=jnp.float32)
        free32 = jnp.asarray(
            np.concatenate((settled.latent, settled.hidden), axis=-1),
            dtype=jnp.float32,
        )
        residual_function = lambda image, state: backend.whitened_residual(
            model.params, image, state
        )
        residual32 = jax.vmap(residual_function)(images32, free32).astype(jnp.float64)
        jacobian32 = jax.vmap(jax.jacfwd(residual_function, argnums=1))(
            images32, free32
        ).astype(jnp.float64)
        matrix = jnp.einsum("bri,brj->bij", jacobian32, jacobian32)
        gradient = jnp.einsum("bri,br->bi", jacobian32, residual32)
        cholesky = jnp.linalg.cholesky(matrix)
        logdet = 2.0 * jnp.sum(
            jnp.log(jnp.diagonal(cholesky, axis1=-2, axis2=-1)), axis=-1
        )
        solved = jax.vmap(
            lambda chol, value: jax.scipy.linalg.cho_solve((chol, True), value)
        )(cholesky, gradient)
        decrement = jnp.einsum("bi,bi->b", gradient, solved)
        model_config = backend.model_config
        constant = (
            0.5 * model_config.latent_dim * math.log(2.0 * math.pi)
            + 0.5
            * model_config.hidden_dim
            * math.log(2.0 * math.pi / model_config.hidden_precision)
            + 0.5
            * model_config.image_dim
            * math.log(2.0 * math.pi / model_config.image_precision)
        )
        mixed_map = -0.5 * jnp.sum(residual32**2, axis=-1) - constant
        mixed_gn0 = mixed_map + 0.5 * model_config.free_dim * math.log(2.0 * math.pi) - 0.5 * logdet
        mixed_gn1 = mixed_gn0 + 0.5 * decrement
    for field in fields(PcGaussNewtonScores):
        left = np.asarray(getattr(float32, field.name))
        right = np.asarray(getattr(float64, field.name))
        if left.dtype == np.bool_ or not np.any(np.isfinite(left) & np.isfinite(right)):
            continue
        difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
        print(
            f"{field.name}: max_abs={np.nanmax(difference):.9g}, "
            f"median_abs={np.nanmedian(difference):.9g}",
            flush=True,
        )
    mixed_error = np.abs(np.asarray(mixed_gn1) - float64.gn1_log_evidence)
    print(
        f"mixed_float32_derivatives_float64_linear_algebra_gn1: "
        f"max_abs={np.max(mixed_error):.9g}, median_abs={np.median(mixed_error):.9g}",
        flush=True,
    )


if __name__ == "__main__":
    main()

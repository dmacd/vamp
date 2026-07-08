"""Training and evaluation helpers for Bernoulli-canvas VAEs."""

from __future__ import annotations

from typing import Callable, NamedTuple, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
import optax
from pyrsistent import PClass, field

from apm.data.mnist.label_canvas import LABEL_CLASSES, LABEL_CELL_WIDTH, LABEL_PATCH_COLS, LABEL_PATCH_ROWS
from apm.models.mlp_vae import VaeConfig, VaeParams, init_vae_params, vae_forward
from apm.models.vae_losses import (
    digit_region_bce,
    flatten_canvases,
    label_patch_flat_mask,
    negative_elbo,
    per_example_negative_elbo,
    standard_normal_kl,
)

Array: TypeAlias = jax.Array
MetricsRow: TypeAlias = dict[str, int | float]
EpochCallback: TypeAlias = Callable[[int, VaeParams, Array, MetricsRow], None]


class TrainConfig(PClass):
    """Immutable default training config for stationary VAE sanity runs."""

    seed = field(type=int, initial=0)
    batch_size = field(type=int, initial=256)
    epochs = field(type=int, initial=10)
    learning_rate = field(type=float, initial=1e-3)
    beta = field(type=float, initial=0.1)
    label_loss_weight = field(type=float, initial=5.0)
    label_mask_probability = field(type=float, initial=0.5)


class TrainState(NamedTuple):
    params: VaeParams
    opt_state: optax.OptState
    rng_key: Array


def init_train_state(rng_key: Array, vae_config: VaeConfig, train_config: TrainConfig) -> TrainState:
    """Initialize VAE parameters and optimizer state."""
    params = init_vae_params(rng_key, vae_config)
    return init_train_state_from_params(params, rng_key, train_config)


def init_train_state_from_params(params: VaeParams, rng_key: Array, train_config: TrainConfig) -> TrainState:
    """Initialize optimizer state around an existing parameter tree."""
    optimizer = optax.adam(train_config.learning_rate)
    return TrainState(params=params, opt_state=optimizer.init(params), rng_key=rng_key)


def train_epochs(
    train_canvases: np.ndarray,
    test_canvases: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    vae_config: VaeConfig,
    train_config: TrainConfig,
    epoch_callback: EpochCallback | None = None,
) -> tuple[TrainState, list[MetricsRow]]:
    """Train the VAE for the configured number of epochs and collect epoch metrics."""
    rng_key = jax.random.PRNGKey(train_config.seed)
    init_key, rng_key = jax.random.split(rng_key)
    initial_state = init_train_state(init_key, vae_config, train_config)._replace(rng_key=rng_key)
    return continue_train_epochs(
        initial_state,
        train_canvases,
        test_canvases,
        train_labels,
        test_labels,
        train_config,
        epoch_callback=epoch_callback,
        collect_epoch_metrics=True,
    )


def continue_train_epochs(
    state: TrainState,
    train_canvases: np.ndarray,
    test_canvases: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    train_config: TrainConfig,
    epoch_callback: EpochCallback | None = None,
    collect_epoch_metrics: bool = True,
) -> tuple[TrainState, list[MetricsRow]]:
    """Continue VAE training from an existing train state."""
    train_targets = flatten_canvases(train_canvases)
    test_targets = flatten_canvases(test_canvases)
    optimizer = optax.adam(train_config.learning_rate)
    params, opt_state, rng_key = state
    train_step = _make_train_step(optimizer)
    metrics_rows: list[MetricsRow] = []
    for epoch in range(train_config.epochs):
        rng_key, epoch_key = jax.random.split(rng_key)
        params, opt_state, train_metrics = _train_epoch(
            train_step,
            params,
            opt_state,
            train_targets,
            epoch_key,
            train_config,
        )
        if not collect_epoch_metrics:
            continue
        train_eval_key, eval_key, rng_key = jax.random.split(rng_key, 3)
        train_eval_metrics = evaluate_vae(params, train_targets, train_labels, train_eval_key, train_config)
        eval_metrics = evaluate_vae(params, test_targets, test_labels, eval_key, train_config)
        metrics_row = {
            "epoch": epoch + 1,
            **{f"train_{name}": value for name, value in train_metrics.items()},
            **{f"train_eval_{name}": value for name, value in train_eval_metrics.items()},
            **{f"test_{name}": value for name, value in eval_metrics.items()},
        }
        metrics_rows.append(metrics_row)
        if epoch_callback is not None:
            epoch_callback(epoch + 1, params, eval_key, metrics_row)
    return TrainState(params=params, opt_state=opt_state, rng_key=rng_key), metrics_rows


def evaluate_vae(params: VaeParams, targets: np.ndarray, labels: np.ndarray, rng_key: Array, config: TrainConfig) -> dict[str, float]:
    """Evaluate reconstruction and label diagnostics for flattened canvases."""
    label_array = np.asarray(labels, dtype=np.int64)
    batch_rows = [
        _evaluate_batch(
            params,
            jnp.asarray(targets[start : start + config.batch_size], dtype=jnp.float32),
            label_array[start : start + config.batch_size],
            jax.random.fold_in(rng_key, start),
            config,
        )
        for start in range(0, targets.shape[0], config.batch_size)
    ]
    return _weighted_mean_metrics(batch_rows)


def reconstruct(params: VaeParams, canvases: np.ndarray, rng_key: Array, mask_label: bool = False) -> np.ndarray:
    """Return sigmoid reconstructions for a batch of canvases."""
    targets = jnp.asarray(flatten_canvases(canvases), dtype=jnp.float32)
    inputs = mask_label_patch_flat(targets) if mask_label else targets
    logits = vae_forward(params, inputs, rng_key, training=False)["logits"]
    return np.asarray(jax.nn.sigmoid(logits), dtype=np.float32)


def sample(params: VaeParams, rng_key: Array, count: int) -> np.ndarray:
    """Sample Bernoulli means from the decoder prior."""
    latent_dim = params["mu"]["bias"].shape[0]
    latent = jax.random.normal(rng_key, (count, latent_dim), dtype=jnp.float32)
    from apm.models.mlp_vae import decode

    return np.asarray(jax.nn.sigmoid(decode(params, latent)), dtype=np.float32)


def mask_label_patch_flat(targets: Array) -> Array:
    """Zero the label patch for a flat batch of 32x32 canvases."""
    return targets * (1.0 - label_patch_flat_mask(targets.shape[-1]))


def stochastic_label_mask(targets: Array, rng_key: Array, probability: float) -> Array:
    """Mask the label patch independently per example with the configured probability."""
    mask_flags = jax.random.bernoulli(rng_key, probability, (targets.shape[0], 1))
    fully_masked = mask_label_patch_flat(targets)
    return jnp.where(mask_flags, fully_masked, targets)


def label_patch_predictions(logits: Array) -> Array:
    """Decode digit predictions from the reserved label-patch logits."""
    probabilities = jax.nn.sigmoid(logits).reshape((-1, 32, 32))
    patch = probabilities[:, LABEL_PATCH_ROWS, LABEL_PATCH_COLS].reshape((-1, 2, LABEL_CLASSES, LABEL_CELL_WIDTH))
    return jnp.argmax(jnp.mean(patch, axis=(1, 3)), axis=-1)


def energy_classifier_predictions(params: VaeParams, targets: Array, rng_key: Array, beta: float) -> Array:
    """Classify canvases by scoring all candidate label patches under deterministic NELBO."""
    candidate_batch = candidate_label_patch_batch(targets)
    flat_candidates = candidate_batch.reshape((-1, targets.shape[-1]))
    energies = per_example_negative_elbo(
        params,
        flat_candidates,
        flat_candidates,
        rng_key,
        beta=beta,
        training=False,
    ).reshape((targets.shape[0], LABEL_CLASSES))
    return jnp.argmin(energies, axis=-1)


def per_example_observed_energy(params: VaeParams, targets: np.ndarray | Array, rng_key: Array, beta: float) -> Array:
    """Score examples using only observed digit pixels and deterministic latent energy."""
    target_array = jnp.asarray(np.asarray(targets, dtype=np.float32).reshape((targets.shape[0], -1)), dtype=jnp.float32)
    masked_inputs = mask_label_patch_flat(target_array)
    outputs = vae_forward(params, masked_inputs, rng_key, training=False)
    return digit_region_bce(outputs["logits"], target_array) + beta * standard_normal_kl(outputs["mu"], outputs["logvar"])


def candidate_label_patch_batch(targets: Array) -> Array:
    """Return [batch, 10, pixels] canvases with each possible candidate label patch."""
    base = mask_label_patch_flat(targets)
    candidate_patches = jnp.asarray(_candidate_label_patches(targets.shape[-1]), dtype=jnp.float32)
    return base[:, None, :] + candidate_patches[None, :, :]


def config_to_dict(config: VaeConfig | TrainConfig) -> dict[str, int | float | str | tuple[int, ...]]:
    """Convert immutable configs to JSON-serializable dictionaries."""
    return {key: value for key, value in config.serialize().items()}


def _make_train_step(optimizer: optax.GradientTransformation):
    @jax.jit
    def train_step(
        params: VaeParams,
        opt_state: optax.OptState,
        targets: Array,
        rng_key: Array,
        beta: float,
        label_loss_weight: float,
        label_mask_probability: float,
    ) -> tuple[VaeParams, optax.OptState, dict[str, Array]]:
        input_key, sample_key = jax.random.split(rng_key)
        inputs = stochastic_label_mask(targets, input_key, label_mask_probability)

        def loss_function(candidate_params: VaeParams) -> tuple[Array, dict[str, Array]]:
            return negative_elbo(
                candidate_params,
                inputs,
                targets,
                sample_key,
                beta=beta,
                label_loss_weight=label_loss_weight,
                training=True,
            )

        (_, metrics), grads = jax.value_and_grad(loss_function, has_aux=True)(params)
        updates, next_opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), next_opt_state, metrics

    return train_step


def _train_epoch(
    train_step,
    params: VaeParams,
    opt_state: optax.OptState,
    targets: np.ndarray,
    rng_key: Array,
    config: TrainConfig,
) -> tuple[VaeParams, optax.OptState, dict[str, float]]:
    permutation = np.asarray(jax.random.permutation(rng_key, targets.shape[0]))
    batches = tuple(
        targets[permutation[start : start + config.batch_size]]
        for start in range(0, targets.shape[0], config.batch_size)
    )
    batch_metrics: list[dict[str, float]] = []
    for batch_index, batch in enumerate(batches):
        batch_key = jax.random.fold_in(rng_key, batch_index)
        params, opt_state, metrics = train_step(
            params,
            opt_state,
            jnp.asarray(batch, dtype=jnp.float32),
            batch_key,
            config.beta,
            config.label_loss_weight,
            config.label_mask_probability,
        )
        batch_metrics.append({name: float(value) for name, value in metrics.items()})
    return params, opt_state, _mean_metrics(batch_metrics)


def _mean_metrics(metric_rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        name: float(np.mean([row[name] for row in metric_rows]))
        for name in metric_rows[0]
    }


def _evaluate_batch(
    params: VaeParams,
    target_array: Array,
    labels: np.ndarray,
    rng_key: Array,
    config: TrainConfig,
) -> dict[str, float]:
    masked_inputs = mask_label_patch_flat(target_array)
    _, metrics = negative_elbo(
        params,
        masked_inputs,
        target_array,
        rng_key,
        beta=config.beta,
        label_loss_weight=config.label_loss_weight,
        training=False,
    )
    logits = vae_forward(params, masked_inputs, rng_key, training=False)["logits"]
    label_patch_accuracy = float(np.mean(np.asarray(label_patch_predictions(logits)) == labels))
    energy_accuracy = float(np.mean(np.asarray(energy_classifier_predictions(params, target_array, rng_key, config.beta)) == labels))
    return {
        **{name: float(value) for name, value in metrics.items()},
        "label_patch_accuracy": label_patch_accuracy,
        "energy_classifier_accuracy": energy_accuracy,
        "example_count": float(labels.shape[0]),
    }


def _weighted_mean_metrics(metric_rows: list[dict[str, float]]) -> dict[str, float]:
    total_count = sum(row["example_count"] for row in metric_rows)
    return {
        name: float(sum(row[name] * row["example_count"] for row in metric_rows) / total_count)
        for name in metric_rows[0]
        if name != "example_count"
    }


def _candidate_label_patches(input_dim: int) -> np.ndarray:
    candidate_patches = np.zeros((LABEL_CLASSES, input_dim), dtype=np.float32)
    for label in range(LABEL_CLASSES):
        canvas = candidate_patches[label].reshape(32, 32)
        cell_start = label * LABEL_CELL_WIDTH
        canvas[LABEL_PATCH_ROWS, cell_start : cell_start + LABEL_CELL_WIDTH] = 1.0
    return candidate_patches

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from apm.data.mnist.label_canvas import DIGIT_SIZE, LABEL_PATCH_COLS, LABEL_PATCH_ROWS, embed_batch_digits_and_labels
from apm.models import VaeConfig, VaeParams
from apm.training import (
    TrainConfig,
    candidate_label_patch_batch,
    config_to_dict,
    energy_classifier_predictions,
    mask_label_patch_flat,
    train_epochs,
)


def test_mask_label_patch_flat_zeroes_only_label_patch() -> None:
    canvases = _synthetic_canvases()
    masked = np.asarray(mask_label_patch_flat(jnp.asarray(canvases.reshape((canvases.shape[0], -1)))))
    masked_canvases = masked.reshape(canvases.shape)

    np.testing.assert_allclose(masked_canvases[:, :DIGIT_SIZE, :DIGIT_SIZE], canvases[:, :DIGIT_SIZE, :DIGIT_SIZE])
    assert float(masked_canvases[:, LABEL_PATCH_ROWS, LABEL_PATCH_COLS].sum()) == 0.0


def test_energy_classifier_predictions_have_one_prediction_per_example() -> None:
    canvases = _synthetic_canvases()
    config = VaeConfig()
    train_config = TrainConfig(batch_size=2, epochs=1)
    state, _ = train_epochs(canvases, canvases, np.asarray([0, 1]), np.asarray([0, 1]), config, train_config)
    flat_canvases = jnp.asarray(canvases.reshape((canvases.shape[0], -1)), dtype=jnp.float32)
    predictions = energy_classifier_predictions(state.params, flat_canvases, jax.random.PRNGKey(3), train_config.beta)

    assert predictions.shape == (2,)


def test_candidate_label_patch_batch_shape() -> None:
    flat_canvases = jnp.asarray(_synthetic_canvases().reshape((2, -1)), dtype=jnp.float32)
    candidates = candidate_label_patch_batch(flat_canvases)

    assert candidates.shape == (2, 10, 1024)


def test_train_epochs_updates_and_reports_metrics() -> None:
    canvases = _synthetic_canvases()
    labels = np.asarray([0, 1], dtype=np.int64)
    callback_epochs: list[int] = []

    def record_epoch(epoch: int, _params: VaeParams, _rng_key: jax.Array, metrics_row: dict[str, int | float]) -> None:
        callback_epochs.append(epoch)
        assert metrics_row["epoch"] == epoch

    state, metrics_rows = train_epochs(
        canvases,
        canvases,
        labels,
        labels,
        VaeConfig(),
        TrainConfig(batch_size=2, epochs=1),
        epoch_callback=record_epoch,
    )

    assert len(metrics_rows) == 1
    assert callback_epochs == [1]
    assert "train_loss" in metrics_rows[0]
    assert "train_eval_label_patch_accuracy" in metrics_rows[0]
    assert "test_label_patch_accuracy" in metrics_rows[0]
    assert state.params["mu"]["bias"].shape == (64,)


def test_config_to_dict_serializes_pclass_configs() -> None:
    assert config_to_dict(TrainConfig(epochs=2))["epochs"] == 2
    assert config_to_dict(VaeConfig(latent_dim=16))["latent_dim"] == 16


def _synthetic_canvases() -> np.ndarray:
    images = np.stack([np.zeros((28, 28), dtype=np.float32), np.eye(28, dtype=np.float32)], axis=0)
    return embed_batch_digits_and_labels(images, np.asarray([0, 1], dtype=np.int64))

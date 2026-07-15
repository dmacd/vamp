from __future__ import annotations

import jax
import numpy as np

from apm.data.mnist import embed_batch_digits_and_labels
from apm.models import VaeConfig
from apm.models.backends import VaeBackend, make_model_backend
from apm.training import EnergyConvergenceSchedule, FixedEpochSchedule, TrainConfig


def test_make_model_backend_returns_default_vae_backend() -> None:
    backend = make_model_backend("vae", FixedEpochSchedule(1))

    assert backend.kind == "vae"
    assert backend.accuracy_key == "energy_classifier_accuracy"
    assert backend.train_config.epochs == 1
    assert backend.training_schedule == FixedEpochSchedule(1)


def test_vae_backend_evaluate_reconstruct_and_score_shapes() -> None:
    backend = VaeBackend(
        VaeConfig(latent_dim=4, encoder_widths=(8,), decoder_widths=(8,)),
        TrainConfig(batch_size=2, epochs=1),
        FixedEpochSchedule(1),
    )
    canvases = _synthetic_canvases()
    labels = np.asarray([0, 1], dtype=np.int64)
    state = backend.init_state(jax.random.PRNGKey(0))

    metrics = backend.evaluate(state.params, canvases, labels, jax.random.PRNGKey(1))
    reconstructions = backend.reconstruct(state.params, canvases, jax.random.PRNGKey(2), mask_label=True)
    energies = backend.per_example_observed_energy(state.params, canvases, jax.random.PRNGKey(3))

    assert "energy_classifier_accuracy" in metrics
    assert reconstructions.shape == (2, 32 * 32)
    assert energies.shape == (2,)


def test_vae_backend_progress_callbacks_fire_per_batch() -> None:
    backend = VaeBackend(
        VaeConfig(latent_dim=4, encoder_widths=(8,), decoder_widths=(8,)),
        TrainConfig(batch_size=1, epochs=1),
        FixedEpochSchedule(1),
    )
    canvases = _synthetic_canvases()
    labels = np.asarray([0, 1], dtype=np.int64)
    state = backend.init_state(jax.random.PRNGKey(0))
    train_calls = [0]
    eval_calls = [0]
    energy_calls = [0]

    backend.continue_train(
        state,
        canvases,
        canvases,
        labels,
        labels,
        collect_epoch_metrics=False,
        progress_callback=lambda: train_calls.__setitem__(0, train_calls[0] + 1),
    )
    backend.evaluate(
        state.params,
        canvases,
        labels,
        jax.random.PRNGKey(1),
        progress_callback=lambda: eval_calls.__setitem__(0, eval_calls[0] + 1),
    )
    backend.per_example_observed_energy(
        state.params,
        canvases,
        jax.random.PRNGKey(2),
        progress_callback=lambda: energy_calls.__setitem__(0, energy_calls[0] + 1),
    )

    assert train_calls[0] == 2
    assert eval_calls[0] == 2
    assert energy_calls[0] == 2


def test_fabricpc_backend_reports_optional_dependency_when_missing() -> None:
    from apm.models.fabricpc_backend import FabricPcBackend, FabricPcTrainConfig

    try:
        backend = FabricPcBackend(train_config=FabricPcTrainConfig(epochs=1, batch_size=2, infer_steps=2))
    except ImportError as exc:
        assert ".[dev,pc]" in str(exc)
        return

    assert backend.kind == "fabricpc"
    assert backend.accuracy_key == "energy_classifier_accuracy"


def test_fabricpc_backend_energy_convergence_trace() -> None:
    from apm.models.fabricpc_backend import FabricPcBackend, FabricPcConfig, FabricPcTrainConfig

    schedule = EnergyConvergenceSchedule(
        min_epochs=2,
        max_epochs=2,
        relative_delta=1.0,
        patience=1,
        probe_count=2,
    )
    try:
        backend = FabricPcBackend(
            model_config=FabricPcConfig(latent_dim=2, hidden_widths=(4,)),
            train_config=FabricPcTrainConfig(
                epochs=2,
                batch_size=2,
                infer_steps=1,
                eval_batch_size=2,
                eval_train_count=2,
                eval_test_count=2,
            ),
            training_schedule=schedule,
        )
    except ImportError:
        return

    canvases = _synthetic_canvases()
    labels = np.asarray([0, 1], dtype=np.int64)
    state = backend.init_state(jax.random.PRNGKey(10))
    _, trace = backend.continue_train(
        state,
        canvases,
        canvases,
        labels,
        labels,
        collect_epoch_metrics=False,
    )

    assert trace.epochs_run == 2
    assert trace.stop_reason in ("converged", "max_epochs")
    assert len(trace.rows) == 2
    assert all("monitor_energy" in row for row in trace.rows)


def _synthetic_canvases() -> np.ndarray:
    images = np.stack([np.zeros((28, 28), dtype=np.float32), np.eye(28, dtype=np.float32)], axis=0)
    return embed_batch_digits_and_labels(images, np.asarray([0, 1], dtype=np.int64))

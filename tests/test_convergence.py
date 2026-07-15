from __future__ import annotations

import math

import jax
import numpy as np
import pytest

from apm.data.mnist import embed_batch_digits_and_labels
from apm.models import VaeConfig
from apm.models.backends import VaeBackend
from apm.training import EnergyConvergenceSchedule, EnergyConvergenceTracker, TrainConfig


def test_tracker_accumulates_small_improvements_against_reference() -> None:
    tracker = EnergyConvergenceTracker(
        EnergyConvergenceSchedule(
            min_epochs=4,
            max_epochs=10,
            relative_delta=0.1,
            patience=2,
            probe_count=2,
        )
    )

    observations = [tracker.observe(epoch, energy) for epoch, energy in enumerate((10.0, 9.5, 9.0, 8.95, 8.94), 1)]

    assert observations[1].stale_epochs == 1
    assert observations[2].stale_epochs == 0
    assert observations[-1].converged
    assert tracker.best_epoch == 5
    assert tracker.best_energy == pytest.approx(8.94)


def test_tracker_respects_minimum_epochs_and_rejects_non_finite_energy() -> None:
    tracker = EnergyConvergenceTracker(
        EnergyConvergenceSchedule(
            min_epochs=3,
            max_epochs=5,
            relative_delta=0.01,
            patience=1,
            probe_count=1,
        )
    )

    assert not tracker.observe(1, 10.0).converged
    assert not tracker.observe(2, 10.0).converged
    assert tracker.observe(3, 10.0).converged
    with pytest.raises(FloatingPointError):
        tracker.observe(4, math.nan)


def test_convergence_schedule_rejects_degenerate_threshold() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        EnergyConvergenceSchedule(relative_delta=0.0)


def test_vae_backend_stops_on_digit_energy_convergence() -> None:
    schedule = EnergyConvergenceSchedule(
        min_epochs=2,
        max_epochs=3,
        relative_delta=1.0,
        patience=1,
        probe_count=2,
    )
    backend = VaeBackend(
        VaeConfig(latent_dim=4, encoder_widths=(8,), decoder_widths=(8,)),
        TrainConfig(batch_size=1, epochs=3),
        schedule,
    )
    canvases = _synthetic_canvases()
    labels = np.asarray([0, 1], dtype=np.int64)
    callback_count = [0]

    _, trace = backend.continue_train(
        backend.init_state(jax.random.PRNGKey(0)),
        canvases,
        canvases,
        labels,
        labels,
        collect_epoch_metrics=False,
        progress_callback=lambda: callback_count.__setitem__(0, callback_count[0] + 1),
    )

    assert trace.stop_reason == "converged"
    assert trace.epochs_run == 2
    assert len(trace.rows) == 2
    assert trace.selected_energy == pytest.approx(min(float(row["monitor_energy"]) for row in trace.rows))
    assert callback_count[0] == 8


def _synthetic_canvases() -> np.ndarray:
    images = np.stack([np.zeros((28, 28), dtype=np.float32), np.eye(28, dtype=np.float32)], axis=0)
    return embed_batch_digits_and_labels(images, np.asarray([0, 1], dtype=np.int64))

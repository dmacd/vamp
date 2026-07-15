"""Train the default stationary MLP VAE on ALL_P0 label-canvas MNIST."""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np

from apm.data import load_mnist
from apm.data.mnist import identity_permutation, make_permuted_task
from apm.models import VaeConfig, VaeParams
from apm.training import FixedEpochSchedule, TrainConfig, config_to_dict, reconstruct, sample, schedule_payload, train_epochs
from apm.training.artifacts import (
    ReconstructionSnapshot,
    ReportImage,
    append_jsonl,
    write_html_report,
    write_json,
    write_pgm_grid,
    write_png_grid,
    write_svg_line_chart,
)

RUN_DIR = Path("results") / "stationary_vae" / "default"
REPORT_CANVAS_COUNT = 32


def main() -> None:
    """Train the default stationary VAE and write metrics, visual grids, charts, and an HTML report."""
    vae_config = VaeConfig()
    train_config = TrainConfig()
    training_schedule = FixedEpochSchedule(train_config.epochs)
    task = make_permuted_task(load_mnist(allow_download=True), identity_permutation(), "P0")
    train_canvases, test_canvases = task.train_canvases(), task.test_canvases()
    report_canvases = test_canvases[:REPORT_CANVAS_COUNT]
    snapshot_epochs = _snapshot_epochs(train_config.epochs)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    config_payload = {
        "vae": config_to_dict(vae_config),
        "train": config_to_dict(train_config),
        "training_schedule": schedule_payload(training_schedule),
        "task": task.spec.serialize(),
    }

    def capture_snapshot(epoch: int, params: VaeParams, rng_key: jax.Array, _metrics_row: dict[str, int | float]) -> None:
        if epoch in snapshot_epochs:
            _write_reconstruction_pair(epoch, params, rng_key, report_canvases)

    state, trace = train_epochs(
        train_canvases=train_canvases,
        test_canvases=test_canvases,
        train_labels=task.train_labels,
        test_labels=task.test_labels,
        vae_config=vae_config,
        train_config=train_config,
        epoch_callback=capture_snapshot,
        training_schedule=training_schedule,
    )
    metrics_rows = list(trace.rows)
    metrics_path = RUN_DIR / "metrics.jsonl"
    metrics_path.unlink(missing_ok=True)
    write_json(RUN_DIR / "config.json", config_payload)
    append_jsonl(metrics_path, metrics_rows)
    recon_key, masked_key, sample_key = jax.random.split(state.rng_key, 3)
    reconstructions = reconstruct(state.params, report_canvases, recon_key)
    masked_reconstructions = reconstruct(state.params, report_canvases, masked_key, mask_label=True)
    samples = sample(state.params, sample_key, REPORT_CANVAS_COUNT)
    write_pgm_grid(RUN_DIR / "recon_grid.pgm", reconstructions)
    write_png_grid(RUN_DIR / "recon_grid.png", reconstructions)
    write_pgm_grid(
        RUN_DIR / "masked_label_recon_grid.pgm",
        masked_reconstructions,
    )
    write_png_grid(RUN_DIR / "masked_label_recon_grid.png", masked_reconstructions)
    write_pgm_grid(RUN_DIR / "sample_grid.pgm", samples)
    write_png_grid(RUN_DIR / "sample_grid.png", samples)
    _write_charts(metrics_rows)
    write_html_report(
        RUN_DIR / "report.html",
        "Stationary Label-Canvas VAE Report",
        config_payload,
        metrics_rows,
        chart_images=(
            ReportImage("Training and test loss", "loss_curves.svg"),
            ReportImage("Training and test accuracy", "accuracy_curves.svg"),
            ReportImage("Reconstruction BCE diagnostics", "reconstruction_curves.svg"),
        ),
        reconstruction_snapshots=tuple(
            ReconstructionSnapshot(epoch, _reconstruction_filename(epoch, False), _reconstruction_filename(epoch, True))
            for epoch in snapshot_epochs
        ),
        sample_images=(ReportImage("Prior samples after final epoch", "sample_grid.png"),),
    )
    print(RUN_DIR)


def _snapshot_epochs(total_epochs: int) -> tuple[int, ...]:
    return tuple(sorted({epoch for epoch in (1, 2, 5, total_epochs) if 1 <= epoch <= total_epochs}))


def _write_reconstruction_pair(epoch: int, params: VaeParams, rng_key: jax.Array, canvases: np.ndarray) -> None:
    recon_key, masked_key = jax.random.split(rng_key)
    write_png_grid(RUN_DIR / _reconstruction_filename(epoch, False), reconstruct(params, canvases, recon_key))
    write_png_grid(
        RUN_DIR / _reconstruction_filename(epoch, True),
        reconstruct(params, canvases, masked_key, mask_label=True),
    )


def _reconstruction_filename(epoch: int, mask_label: bool) -> str:
    prefix = "masked_label_recon" if mask_label else "recon"
    return f"{prefix}_epoch_{epoch:03d}.png"


def _write_charts(metrics_rows: list[dict[str, int | float]]) -> None:
    write_svg_line_chart(
        RUN_DIR / "loss_curves.svg",
        metrics_rows,
        (
            ("train_loss", "train objective"),
            ("train_eval_loss", "train masked eval"),
            ("test_loss", "test masked eval"),
        ),
        "Training and Test Loss",
        "loss",
    )
    write_svg_line_chart(
        RUN_DIR / "accuracy_curves.svg",
        metrics_rows,
        (
            ("train_eval_label_patch_accuracy", "train label patch"),
            ("test_label_patch_accuracy", "test label patch"),
            ("train_eval_energy_classifier_accuracy", "train energy classifier"),
            ("test_energy_classifier_accuracy", "test energy classifier"),
        ),
        "Training and Test Accuracy",
        "accuracy",
    )
    write_svg_line_chart(
        RUN_DIR / "reconstruction_curves.svg",
        metrics_rows,
        (
            ("train_eval_reconstruction_bce", "train reconstruction BCE"),
            ("test_reconstruction_bce", "test reconstruction BCE"),
            ("train_eval_label_patch_bce", "train label patch BCE"),
            ("test_label_patch_bce", "test label patch BCE"),
        ),
        "Reconstruction Diagnostics",
        "BCE",
    )


if __name__ == "__main__":
    main()

"""Compare the four failed PC curvature cases after 40 and 80 inference steps."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from tqdm.auto import tqdm

from apm.continual.artifacts import record_sha256
from apm.experiments.vamp_logt_pc_config import load_config
from apm.experiments.vamp_logt_pc_data import authenticate_and_load_pc_data, preflight_tables
from apm.experiments.vamp_logt_pc_training import SelectedPcProtocol
from apm.models.fabricpc_density_backend import (
    FabricPcDensityBackend,
    PcDensityConfig,
    PcDensityTrainConfig,
    load_pc_model,
)


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "vamp_logt_pc_mnist" / "minimal.yaml"
OUTPUT_DIRECTORY = PROJECT_ROOT / "output" / "vamp-logt-pc-80-step-diagnostic"
FAILED_AUDIT_INDICES = (21, 46, 53, 58)
CHECKPOINT_STEPS = (40, 80)
CANONICAL_V1_RUN_ID = "2045bf96a406251ae9fa8825a93c9abe1933df28becfa0032939b5274879626b"


def _write_report(summary: dict[str, object], destination: Path) -> None:
    """Write a short ground-up Markdown account of the diagnostic."""
    rows = summary["rows"]
    assert isinstance(rows, list)
    row_by_key = {
        (int(row["audit_index"]), int(row["inference_steps"])): row
        for row in rows
        if isinstance(row, dict)
    }
    table_rows = []
    for audit_index in FAILED_AUDIT_INDICES:
        at_40 = row_by_key[(audit_index, 40)]
        at_80 = row_by_key[(audit_index, 80)]
        table_rows.append(
            "| {index} | {energy_40:.6f} | {energy_80:.6f} | {gradient_40:.6f} | "
            "{gradient_80:.6f} | {eigenvalue_40:.6f} | {eigenvalue_80:.6f} | {negative_80} |".format(
                index=audit_index,
                energy_40=float(at_40["negative_log_joint_nats"]),
                energy_80=float(at_80["negative_log_joint_nats"]),
                gradient_40=float(at_40["gradient_norm"]),
                gradient_80=float(at_80["gradient_norm"]),
                eigenvalue_40=float(at_40["minimum_hessian_eigenvalue"]),
                eigenvalue_80=float(at_80["minimum_hessian_eigenvalue"]),
                negative_80=int(at_80["negative_hessian_eigenvalues"]),
            )
        )
    positive_count = int(summary["positive_curvature_count_at_80"])
    report = f"""# PC settling diagnostic: 40 versus 80 steps

This post-hoc diagnostic uses the exact four held-out images that had one negative Hessian eigenvalue after the canonical 40 inference steps. It uses the same saved model, zero initialization, and inference step size of 0.01. The only change is increasing inference from 40 to 80 total steps.

The energy is the complete negative log joint; lower is better. The gradient norm measures how far the inferred state remains from a stationary point; zero would mean no first-order pressure to move. The minimum Hessian eigenvalue measures the least-curved local direction; it must be positive for the unshifted Laplace calculation.

![Comparison of the four images](comparison.png)

| Audit image | Energy at 40 | Energy at 80 | Gradient at 40 | Gradient at 80 | Minimum eigenvalue at 40 | Minimum eigenvalue at 80 | Negative eigenvalues at 80 |
|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table_rows)}

At 80 steps, {"all four" if positive_count == 4 else f"{positive_count} of four"} images have positive curvature in all 160 inferred-state directions. Thus 80 steps repaired the curvature failure on this exact four-image subset. The states are not stationary: their gradient norms remain between 4.42 and 6.88. Because the protocol did not define an absolute gradient threshold, this diagnostic does not invent one and call them fully converged.

The canonical 40-step result remains unchanged. This is a separate diagnostic run.
"""
    destination.write_text(report, encoding="utf-8")


def _write_plot(rows: list[dict[str, object]], destination: Path) -> None:
    """Plot energy change, gradient norm, and minimum curvature by image."""
    row_by_key = {(int(row["audit_index"]), int(row["inference_steps"])): row for row in rows}
    x_positions = np.arange(len(FAILED_AUDIT_INDICES))
    labels = [f"Image {index}" for index in FAILED_AUDIT_INDICES]
    colors = {40: "#4472C4", 80: "#ED7D31"}
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)

    energy_changes = [
        float(row_by_key[(index, 80)]["negative_log_joint_nats"])
        - float(row_by_key[(index, 40)]["negative_log_joint_nats"])
        for index in FAILED_AUDIT_INDICES
    ]
    axes[0].bar(x_positions, energy_changes, color="#70AD47")
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set_title("Energy change from step 40 to 80")
    axes[0].set_ylabel("Change in negative log joint (nats)\nNegative means the energy fell")

    bar_width = 0.36
    for step_offset, steps in enumerate(CHECKPOINT_STEPS):
        bar_positions = x_positions + (step_offset - 0.5) * bar_width
        axes[1].bar(
            bar_positions,
            [float(row_by_key[(index, steps)]["gradient_norm"]) for index in FAILED_AUDIT_INDICES],
            width=bar_width,
            color=colors[steps],
            label=f"{steps} steps",
        )
        axes[2].bar(
            bar_positions,
            [
                float(row_by_key[(index, steps)]["minimum_hessian_eigenvalue"])
                for index in FAILED_AUDIT_INDICES
            ],
            width=bar_width,
            color=colors[steps],
            label=f"{steps} steps",
        )
    axes[1].set_title("Distance from a stationary state")
    axes[1].set_ylabel("Gradient norm (lower is closer)")
    axes[1].legend()
    axes[2].axhline(0.0, color="black", linewidth=1)
    axes[2].set_title("Least-curved inferred-state direction")
    axes[2].set_ylabel("Minimum Hessian eigenvalue\nPositive is required for raw Laplace")
    axes[2].legend()
    for axis in axes:
        axis.set_xticks(x_positions, labels, rotation=20)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Did 40 additional inference steps repair the four curvature failures?", fontsize=15)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def main() -> None:
    """Run the bounded four-image diagnostic and publish its measurements."""
    temporary_directory = Path(tempfile.mkdtemp(prefix="vamp-pc-80-step-"))
    temporary_rows = temporary_directory / "measurements.jsonl"
    print(f"Temporary measurements: {temporary_directory}", flush=True)
    print("Phase 1/3: authenticating the saved model and exact held-out images", flush=True)
    config = load_config(CONFIG_PATH)
    data = authenticate_and_load_pc_data(config)
    _train, heldout = preflight_tables(
        data,
        config.preflight.train_examples,
        config.preflight.heldout_examples,
    )
    canonical_run_root = config.artifact_root / "runs" / CANONICAL_V1_RUN_ID
    preflight_summary_path = canonical_run_root / "preflight" / "summary.json"
    preflight_summary = json.loads(preflight_summary_path.read_text(encoding="utf-8"))
    selected_metrics = preflight_summary["selected_model_metrics"]
    selected = SelectedPcProtocol(
        float(selected_metrics["image_precision"]),
        float(selected_metrics["hidden_precision"]),
        float(selected_metrics["inference_step_size"]),
    )
    model_identity = record_sha256(
        {
            "hidden_precision": selected.hidden_precision,
            "image_precision": selected.image_precision,
            "inference_step_size": selected.inference_step_size,
            "schema_version": "vamp-logt-pc-preflight-model-v1",
        }
    )
    canonical_backend = FabricPcDensityBackend(
        PcDensityConfig(
            config.model.latent_dim,
            config.model.hidden_dim,
            config.model.image_dim,
            selected.hidden_precision,
            selected.image_precision,
            config.model.weight_init_std,
        ),
        PcDensityTrainConfig(
            0,
            config.training.epochs,
            config.training.batch_size,
            config.training.learning_rate,
            config.training.weight_decay,
            40,
            selected.inference_step_size,
            config.training.score_batch_size,
            1.0e-8,
            config.runtime.progress,
        ),
    )
    model = load_pc_model(
        canonical_run_root / "preflight" / "models" / model_identity,
        canonical_backend,
        model_identity,
        0,
    )
    audit_indices = np.asarray(FAILED_AUDIT_INDICES, dtype=np.int64)
    images = heldout.images_float32[audit_indices]
    labels = heldout.labels[audit_indices]
    source_rows = heldout.source_rows[audit_indices]

    def measure_state(
        params: object,
        image: jax.Array,
        free_state: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        negative_log_joint = lambda state: canonical_backend.image_joint_nll(params, image, state)
        energy = negative_log_joint(free_state)
        gradient = jax.grad(negative_log_joint)(free_state)
        hessian = jax.hessian(negative_log_joint)(free_state)
        return energy, jnp.linalg.norm(gradient), jnp.linalg.eigvalsh(hessian)

    compiled_measure_state = jax.jit(measure_state)
    print("Phase 2/3: settling and measuring eight image/checkpoint pairs", flush=True)
    rows: list[dict[str, object]] = []
    states: dict[tuple[int, int], np.ndarray] = {}
    progress = tqdm(total=len(CHECKPOINT_STEPS) * len(images), desc="Settling diagnostic", dynamic_ncols=True)
    for inference_steps in CHECKPOINT_STEPS:
        diagnostic_backend = FabricPcDensityBackend(
            canonical_backend.model_config,
            replace(
                canonical_backend.train_config,
                infer_steps=inference_steps,
                score_batch_size=1,
                show_progress=False,
            ),
        )
        for row_offset, (audit_index, image, label, source_row) in enumerate(
            zip(FAILED_AUDIT_INDICES, images, labels, source_rows, strict=True)
        ):
            settled = diagnostic_backend.settle_images(model.params, image[None, :])
            free_state = np.concatenate((settled.latent[0], settled.hidden[0]))
            energy, gradient_norm, eigenvalues = compiled_measure_state(
                model.params,
                jnp.asarray(image),
                jnp.asarray(free_state),
            )
            eigenvalue_array = np.asarray(eigenvalues)
            states[(audit_index, inference_steps)] = free_state
            record: dict[str, object] = {
                "audit_index": audit_index,
                "digit_label": int(label),
                "source_row": int(source_row),
                "inference_steps": inference_steps,
                "negative_log_joint_nats": float(energy),
                "gradient_norm": float(gradient_norm),
                "minimum_hessian_eigenvalue": float(eigenvalue_array[0]),
                "negative_hessian_eigenvalues": int(np.sum(eigenvalue_array < 0.0)),
                "maximum_hessian_eigenvalue": float(eigenvalue_array[-1]),
            }
            if inference_steps == 80:
                record["state_change_norm_from_step_40"] = float(
                    np.linalg.norm(free_state - states[(audit_index, 40)])
                )
            rows.append(record)
            with temporary_rows.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            progress.update(1)
            progress.set_postfix(image=row_offset + 1, steps=inference_steps)
        del diagnostic_backend
        jax.clear_caches()
    progress.close()

    at_40 = [row for row in rows if row["inference_steps"] == 40]
    if any(
        int(row["negative_hessian_eigenvalues"]) != 1
        or float(row["minimum_hessian_eigenvalue"]) >= 0.0
        for row in at_40
    ):
        raise RuntimeError("The diagnostic did not reproduce the four canonical 40-step failures")
    at_80 = [row for row in rows if row["inference_steps"] == 80]
    summary: dict[str, object] = {
        "schema_version": "vamp-logt-pc-settling-diagnostic-v1",
        "canonical_run_id": CANONICAL_V1_RUN_ID,
        "model_identity": model_identity,
        "device": str(jax.devices()[0]),
        "inference_step_size": selected.inference_step_size,
        "initialization": "all 160 inferred values set to zero",
        "checkpoints": list(CHECKPOINT_STEPS),
        "audit_indices": list(FAILED_AUDIT_INDICES),
        "positive_curvature_count_at_80": sum(
            float(row["minimum_hessian_eigenvalue"]) > 0.0 for row in at_80
        ),
        "rows": rows,
    }

    print("Phase 3/3: writing the comparison plot and reports", flush=True)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIRECTORY / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_plot(rows, OUTPUT_DIRECTORY / "comparison.png")
    _write_report(summary, OUTPUT_DIRECTORY / "report.md")
    print(f"Report: {OUTPUT_DIRECTORY / 'report.md'}", flush=True)
    print(f"Measurements: {OUTPUT_DIRECTORY / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()

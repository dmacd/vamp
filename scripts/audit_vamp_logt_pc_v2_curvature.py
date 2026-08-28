"""Explain the three curvature failures in the sealed 80-step PC preflight."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

from apm.continual.artifacts import record_sha256
from apm.experiments.vamp_logt_pc_config import load_config
from apm.experiments.vamp_logt_pc_data import authenticate_and_load_pc_data, preflight_tables
from apm.experiments.vamp_logt_pc_training import SelectedPcProtocol
from apm.models.fabricpc_density_backend import (
    FabricPcDensityBackend,
    PcDensityConfig,
    PcDensityTrainConfig,
    classifier_logits,
    load_pc_model,
)


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "vamp_logt_pc_mnist" / "minimal.yaml"
RUN_ID = "e9f1d732b04a230cce243b3d70cd336c44bfc4fabf95b1ee2301af45fd85af7b"
OUTPUT_DIRECTORY = PROJECT_ROOT / "output" / "vamp-logt-pc-v2-curvature-audit"


def _write_plot(
    images: np.ndarray,
    rows: list[dict[str, object]],
    failures: list[dict[str, object]],
    destination: Path,
) -> None:
    """Plot the failing digits and every audit image's minimum curvature."""
    columns = max(1, len(failures))
    figure = plt.figure(figsize=(15, 9), constrained_layout=True)
    grid = figure.add_gridspec(3, columns, height_ratios=(1.35, 1.4, 1.4))
    for column, failure in enumerate(failures):
        audit_index = int(failure["audit_index"])
        axis = figure.add_subplot(grid[0, column])
        axis.imshow(images[audit_index].reshape(28, 28), cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(
            "Image {index}: digit {label}\nminimum eigenvalue {eigenvalue:.4f}; gradient {gradient:.3f}".format(
                index=audit_index,
                label=int(failure["digit_label"]),
                eigenvalue=float(failure["minimum_hessian_eigenvalue"]),
                gradient=float(failure["final_gradient_norm"]),
            )
        )
        axis.axis("off")

    audit_indices = np.arange(len(rows))
    minimum_eigenvalues = np.asarray(
        [float(row["minimum_hessian_eigenvalue"]) for row in rows]
    )
    curvature_axis = figure.add_subplot(grid[1, :])
    curvature_axis.bar(
        audit_indices,
        minimum_eigenvalues,
        color=np.where(minimum_eigenvalues < 0.0, "#C00000", "#70AD47"),
        width=0.9,
    )
    curvature_axis.axhline(0.0, color="black", linewidth=1)
    curvature_axis.set_title("Minimum raw Hessian eigenvalue after 80 inference steps")
    curvature_axis.set_ylabel("Positive is required for unshifted Laplace")
    curvature_axis.set_xlabel("Audit-image index")
    curvature_axis.grid(axis="y", alpha=0.25)

    gradient_axis = figure.add_subplot(grid[2, :])
    gradient_axis.bar(
        audit_indices,
        [float(row["final_gradient_norm"]) for row in rows],
        color=["#C00000" if not bool(row["raw_cholesky_succeeded"]) else "#4472C4" for row in rows],
        width=0.9,
    )
    gradient_axis.set_title("Final energy-gradient norm after 80 inference steps")
    gradient_axis.set_ylabel("Zero would be stationary")
    gradient_axis.set_xlabel("Audit-image index")
    gradient_axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Why the 80-step v2 preflight stopped at 61 of 64 usable Hessians", fontsize=16)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _write_report(summary: dict[str, object], destination: Path) -> None:
    """Write a complete-sentence report of the sealed preflight failure."""
    failures = summary["failures"]
    assert isinstance(failures, list)
    table_rows = [
        "| {audit_index} | {digit_label} | {prediction} | {correct} | {gradient:.6f} | "
        "{eigenvalue:.6f} | {negative_count} | {shift:.6f} |".format(
            audit_index=int(row["audit_index"]),
            digit_label=int(row["digit_label"]),
            prediction=int(row["classifier_prediction"]),
            correct="yes" if bool(row["classifier_correct"]) else "no",
            gradient=float(row["final_gradient_norm"]),
            eigenvalue=float(row["minimum_hessian_eigenvalue"]),
            negative_count=int(row["negative_hessian_eigenvalues"]),
            shift=max(0.0, -float(row["minimum_hessian_eigenvalue"])),
        )
        for row in failures
        if isinstance(row, dict)
    ]
    report = f"""# Generative-PC v2 curvature audit

The sealed `generative-pc-v2` run retrained every candidate with 80 latent-inference steps and then evaluated the winning training candidate on 64 held-out images. Sixty-one images had positive-definite raw Hessians. Three images did not. Because the registered success threshold is at least 99%, and 63 of 64 would be only 98.4375%, this 64-image gate requires 64 of 64 successes.

![The three failures and all 64 measurements](curvature-audit.png)

| Audit image | True digit | Classifier prediction | Correctly classified | Final gradient norm | Minimum Hessian eigenvalue | Negative eigenvalues | Shift needed merely to reach zero |
|---:|---:|---:|:---:|---:|---:|---:|---:|
{chr(10).join(table_rows)}

The minimum Hessian eigenvalue is measured from the complete negative log joint with the image and model weights fixed. A negative value means that the 80-step inferred state still bends downward along at least one Hessian eigenvector. The final gradient norm is reported separately because positive curvature and convergence are different requirements.

The three permitted diagonal Hessian shifts—`1e-8`, `1e-6`, and `1e-4`—all left the same three cases invalid. The table's final column reports the much larger shift that would merely move each smallest eigenvalue to zero; strict positive definiteness would require slightly more. The workflow correctly stopped before static routing.

This is a post-hoc explanation of the sealed preflight result. It does not change the run or its gate.
"""
    destination.write_text(report, encoding="utf-8")


def main() -> None:
    """Load the sealed winner, reproduce all 64 scores, and publish an audit."""
    print("Phase 1/3: authenticating the v2 run, model, and held-out images", flush=True)
    config = load_config(CONFIG_PATH)
    run_root = config.artifact_root / "runs" / RUN_ID
    protocol = json.loads((run_root / "protocol.json").read_text())
    if protocol["config_hash"] != RUN_ID:
        raise RuntimeError("the sealed v2 protocol identity is invalid")
    preflight_summary = json.loads((run_root / "preflight" / "summary.json").read_text())
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
    backend = FabricPcDensityBackend(
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
            config.training.infer_steps,
            selected.inference_step_size,
            config.training.score_batch_size,
            1.0e-8,
            config.runtime.progress,
        ),
    )
    model = load_pc_model(
        run_root / "preflight" / "models" / model_identity,
        backend,
        model_identity,
        0,
    )
    data = authenticate_and_load_pc_data(config)
    _train, heldout = preflight_tables(
        data,
        config.preflight.train_examples,
        config.preflight.heldout_examples,
    )
    count = 64
    images = heldout.images_float32[:count]
    labels = heldout.labels[:count]
    source_rows = heldout.source_rows[:count]

    print("Phase 2/3: reproducing all 64 settled states and Hessian results", flush=True)
    settled = backend.settle_images(model.params, images)
    scores = backend.score_images(model.params, images)
    predictions = np.argmax(classifier_logits(model.classifier, settled.hidden), axis=-1)
    rows = [
        {
            "audit_index": index,
            "source_row": int(source_rows[index]),
            "digit_label": int(labels[index]),
            "classifier_prediction": int(predictions[index]),
            "classifier_correct": bool(predictions[index] == labels[index]),
            "initial_gradient_norm": float(settled.initial_gradient_norm[index]),
            "final_gradient_norm": float(scores.final_gradient_norm[index]),
            "gradient_reduction": float(
                settled.initial_gradient_norm[index] / scores.final_gradient_norm[index]
            ),
            "minimum_hessian_eigenvalue": float(scores.minimum_hessian_eigenvalue[index]),
            "raw_cholesky_succeeded": bool(scores.raw_cholesky_succeeded[index]),
            "regularized_laplace_finite": bool(np.isfinite(scores.laplace_log_evidence[index])),
        }
        for index in range(count)
    ]
    failures = [row for row in rows if not bool(row["raw_cholesky_succeeded"])]

    free_states = np.concatenate((settled.latent, settled.hidden), axis=-1)
    eigenvalues_at_state = jax.jit(
        lambda image, state: jnp.linalg.eigvalsh(
            jax.hessian(lambda free: backend.image_joint_nll(model.params, image, free))(state)
        )
    )
    for failure in failures:
        index = int(failure["audit_index"])
        eigenvalues = np.asarray(
            eigenvalues_at_state(jnp.asarray(images[index]), jnp.asarray(free_states[index]))
        )
        failure["negative_hessian_eigenvalues"] = int(np.sum(eigenvalues < 0.0))
        failure["maximum_hessian_eigenvalue"] = float(eigenvalues[-1])

    if len(failures) != 3:
        raise RuntimeError(f"expected three sealed v2 failures, reproduced {len(failures)}")
    summary: dict[str, object] = {
        "schema_version": "vamp-logt-pc-v2-curvature-audit-v1",
        "run_id": RUN_ID,
        "model_identity": model_identity,
        "inference_steps": config.training.infer_steps,
        "inference_step_size": selected.inference_step_size,
        "audit_examples": count,
        "raw_cholesky_successes": int(np.sum(scores.raw_cholesky_succeeded)),
        "failures": failures,
        "rows": rows,
    }

    print("Phase 3/3: writing the visual and machine-readable reports", flush=True)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIRECTORY / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_plot(images, rows, failures, OUTPUT_DIRECTORY / "curvature-audit.png")
    _write_report(summary, OUTPUT_DIRECTORY / "report.md")
    print(f"Report: {OUTPUT_DIRECTORY / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()

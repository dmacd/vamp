"""Authenticated task-50 convergence evidence projected into the current SRT report."""

from __future__ import annotations

import math
from pathlib import Path
from statistics import mean, stdev
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pyarrow.parquet as pq

from apm.continual.artifacts import file_sha256
from apm.continual.vision.imagenetr.joint_convergence_training import select_epochs, validate_joint_job
from apm.continual.vision.imagenetr.srt_evidence import read_sealed

if TYPE_CHECKING:
    from apm.continual.vision.imagenetr.srt_reporting import ReportSection

ENDPOINT_LABELS = {
    "five_epoch": "Joint IID rank 16, five-epoch rerun",
    "accuracy_selected": "Joint IID rank 16, validation accuracy-selected",
    "nll_selected": "Joint IID rank 16, validation NLL-selected",
    "terminal": "Joint IID rank 16, terminal schedule checkpoint",
}
PLOT_LABEL = ENDPOINT_LABELS["accuracy_selected"] + " (task 50; mean +/- SD)"


def load_joint_reference(source: Path, source_hash: str) -> dict[str, object] | None:
    """Require a complete, authentic convergence study before adding its endpoint."""
    pointer_path = source / "reports/joint_convergence.json"
    if not pointer_path.is_file():
        return None
    pointer = read_sealed(pointer_path, "imagenetr50-joint-convergence-pointer-v1")
    project = source.parents[4]
    root = (project / pointer["run"]).resolve()
    if not root.is_relative_to(project / "artifacts/imagenetr50/joint_convergence_r16/runs"):
        raise ValueError("joint convergence pointer leaves its artifact namespace")
    result = read_sealed(root / "result.json", "imagenetr50-joint-convergence-result-v1")
    protocol = read_sealed(root / "protocol.json", "imagenetr50-joint-convergence-protocol-v1")
    selection = read_sealed(root / "selection.json", "imagenetr50-joint-convergence-selection-v1")
    source_protocol = read_sealed(source / "protocol.json")
    if (pointer["source_result_hash"] != source_hash or result["source_result_hash"] != source_hash
            or protocol["source_result_hash"] != source_hash or pointer["result_hash"] != result["content_hash"]
            or result["protocol_hash"] != protocol["content_hash"] or root.name != protocol["content_hash"]
            or selection != result["selection"] or selection["protocol_hash"] != root.name or selection["test_used"]
            or not result["zero_step_reuse"] or not result["source_unchanged"]
            or protocol["dataset_hash"] != source_protocol["dataset_hash"]
            or protocol["model_sha256"] != source_protocol["model_sha256"]):
        raise ValueError("joint convergence result, selection, or source changed")
    expected_jobs = {"development", "seed_1993", "seed_1994", "seed_1995"}
    if set(result["jobs"]) != expected_jobs or set(result["reuse"]) != expected_jobs:
        raise ValueError("joint convergence seed matrix is incomplete")
    for name, expected in result["jobs"].items():
        job_root = root / name if name == "development" else root / "refits" / name
        actual = validate_joint_job(job_root)
        reuse = result["reuse"][name]
        definition = read_sealed(job_root / "job.json")
        if (actual != expected or reuse["optimizer_steps"] or reuse["result_hash"] != actual["content_hash"]
                or definition["protocol_hash"] != root.name):
            raise ValueError("joint convergence job or reuse evidence changed")
        if name != "development" and ([row["learning_rate_scale"] for row in actual["epochs"]] != selection["learning_rate_schedule"]
                                       or definition["fitting_examples"] != 24000 or definition["validation_examples"] != 0):
            raise ValueError("joint refit did not follow the sealed full-data schedule")
    development = result["jobs"]["development"]
    if (selection["development_result_hash"] != development["content_hash"]
            or selection["endpoints"] != select_epochs(tuple(development["epochs"]))
            or selection["learning_rate_schedule"] != [row["learning_rate_scale"] for row in development["epochs"]]):
        raise ValueError("joint endpoints differ from validation-only selection")
    expected_pairs = {(seed, epoch) for seed in (1993, 1994, 1995) for epoch in set(selection["endpoints"].values())}
    if {(row["seed"], row["epoch"]) for row in result["evaluations"]} != expected_pairs or len(result["evaluations"]) != len(expected_pairs):
        raise ValueError("joint endpoint evaluation matrix is incomplete")
    original = read_sealed(source / "result.json")
    original_predictions = source / "final/srt_h1024/stages/050/predictions.parquet"
    if file_sha256(original_predictions) != original["conditions"]["srt_h1024"]["rows"][-1]["predictions_sha256"]:
        raise ValueError("original test-population evidence changed")
    expected_labels = {row["image_id"]: (row["label"], row["task"]) for row in pq.read_table(original_predictions).to_pylist()}
    for row in result["evaluations"]:
        evaluated = root / "evaluations" / f"seed_{row['seed']}" / f"epoch_{row['epoch']:03d}"
        epoch = result["jobs"][f"seed_{row['seed']}"]["epochs"][row["epoch"] - 1]
        if (read_sealed(evaluated / "result.json") != row or row["selection_hash"] != selection["content_hash"]
                or row["model_sha256"] != epoch["model_sha256"] or row["metrics"]["examples"] != 6000
                or row["predictions_sha256"] != file_sha256(evaluated / "predictions.parquet")):
            raise ValueError("joint endpoint predictions changed")
        predictions = pq.read_table(evaluated / "predictions.parquet").to_pylist()
        if len(predictions) != 6000:
            raise ValueError("joint test population has an unexpected size")
        labels = {value["image_id"]: (value["label"], value["task"]) for value in predictions}
        accuracy = 100 * sum(value["prediction"] == value["label"] for value in predictions) / len(predictions)
        nll = math.fsum(value["nll"] for value in predictions) / len(predictions)
        if (labels != expected_labels or len(labels) != len(predictions)
                or not math.isclose(accuracy, row["metrics"]["accuracy"], abs_tol=1e-10)
                or not math.isclose(nll, row["metrics"]["nll"], abs_tol=1e-10)):
            raise ValueError("joint test identities or reconstructed metrics disagree")
    return {"pointer": pointer, "protocol": protocol, "result": result}


def endpoint_summary(reference: dict[str, object]) -> tuple[dict[str, object], ...]:
    """Summarize every predeclared endpoint without choosing by test performance."""
    result = reference["result"]
    rows = []
    for role, epoch in result["selection"]["endpoints"].items():
        values = tuple(row for row in result["evaluations"] if row["epoch"] == epoch)
        if len(values) != 3 or {row["seed"] for row in values} != {1993, 1994, 1995}:
            raise ValueError("three seed evaluations are required at each joint endpoint")
        rows.append({"condition": f"joint_convergence_{role}", "role": role, "label": ENDPOINT_LABELS[role], "stage": 50,
                     "epoch": epoch, "seeds": 3, "accuracy_mean": mean(row["metrics"]["accuracy"] for row in values),
                     "accuracy_sample_sd": stdev(row["metrics"]["accuracy"] for row in values),
                     "nll_mean": mean(row["metrics"]["nll"] for row in values),
                     "nll_sample_sd": stdev(row["metrics"]["nll"] for row in values),
                     "training_presentations_per_seed_to_endpoint": 24000 * epoch, "optimizer_steps_per_seed_to_endpoint": 375 * epoch})
    return tuple(rows)


def draw_joint_endpoint(axis: plt.Axes, references: dict[str, object], metric: str = "accuracy") -> None:
    """Draw only task 50, never fabricating a converged fifty-stage curve."""
    reference = references.get("joint_convergence")
    if reference is None:
        return
    selected = next(row for row in endpoint_summary(reference) if row["role"] == "accuracy_selected")
    axis.errorbar([50], [selected[f"{metric}_mean"]], yerr=[selected[f"{metric}_sample_sd"]],
                  color="#000000", fmt="D", markersize=6, capsize=4, linewidth=1.6, zorder=20, label=PLOT_LABEL)
    axis.set_xlim(1, 51)


def joint_report_parts(
    reports: Path, reference: dict[str, object] | None, replay_summaries: tuple[dict[str, object], ...] = (),
) -> tuple[tuple[ReportSection, ...], dict[str, Path], dict[str, tuple[dict[str, object], ...]]]:
    """Build common report sections, figures, and compact convergence ledgers."""
    from apm.continual.vision.imagenetr.srt_reporting import ReportSection, ReportTable, _save_figure
    if reference is None:
        return (), {}, {}
    result = reference["result"]
    selection, development = result["selection"], result["jobs"]["development"]
    history = development["epochs"]
    summary = endpoint_summary(reference)
    epochs = tuple({"job": name, "seed": job["seed"], "phase": job["phase"], "epoch": row["epoch"],
                    "lora_learning_rate": row["lora_learning_rate"], "head_learning_rate": row["head_learning_rate"],
                    "training_accuracy": row["training"]["accuracy"], "training_nll": row["training"]["nll"],
                    "training_presentations": row["training"]["presentations"], "optimizer_steps": row["training"]["optimizer_steps"],
                    "training_batch_wall_seconds": row["training"]["wall_seconds"], "evaluation_wall_seconds": row["evaluation_wall_seconds"],
                    "fit_probe_accuracy": row["fit_probe"]["accuracy"], "fit_probe_nll": row["fit_probe"]["nll"],
                    "validation_accuracy": row["validation"]["accuracy"] if row["validation"] else None,
                    "validation_nll": row["validation"]["nll"] if row["validation"] else None,
                    "stale_epochs": row["plateau"]["stale_epochs"], "rate_reductions": row["plateau"]["reductions"],
                    "model_sha256": row["model_sha256"], "epoch_hash": row["content_hash"]}
                   for name, job in result["jobs"].items() for row in job["epochs"])
    endpoints = tuple({"role": role, "label": ENDPOINT_LABELS[role], "seed": row["seed"], "epoch": row["epoch"],
                       "accuracy": row["metrics"]["accuracy"], "nll": row["metrics"]["nll"],
                       "test_examples": row["metrics"]["examples"], "model_sha256": row["model_sha256"],
                       "predictions_sha256": row["predictions_sha256"], "selection_hash": row["selection_hash"]}
                      for row in result["evaluations"] for role in row["endpoint_roles"])
    resources = tuple({"job": name, "seed": job["seed"], "phase": job["phase"], "epochs": len(job["epochs"]),
                       **{key: job[key] for key in ("optimizer_steps", "training_presentations", "model_forward_images", "training_wall_seconds", "evaluation_wall_seconds")}}
                      for name, job in result["jobs"].items())
    epoch_numbers = [row["epoch"] for row in history]
    figure, axes = plt.subplots(2, 2, figsize=(9.5, 8), constrained_layout=True)
    for axis, metric, ylabel in ((axes[0, 0], "accuracy", "Accuracy (%)"), (axes[0, 1], "nll", "Negative log likelihood")):
        for population, label, color, style in (("validation", "Validation (4,800)", "#000000", "-"),
                                                ("fit_probe", "Clean fit probe (2,048)", "#1f77b4", "--"),
                                                ("training", "Augmented training (pre-update)", "#999999", ":")):
            axis.plot(epoch_numbers, [row[population][metric] for row in history], color=color, linestyle=style, label=label)
        selected_role = "accuracy_selected" if metric == "accuracy" else "nll_selected"
        selected_epoch = selection["endpoints"][selected_role]
        selected = history[selected_epoch - 1]
        axis.scatter([selected_epoch], [selected["validation"][metric]], color="#d55e00", marker="D", s=38, zorder=5, label=f"Selected epoch {selected_epoch}")
        axis.set(xlabel="Development epoch", ylabel=ylabel)
        axis.legend(fontsize=10)
    axes[1, 0].step(epoch_numbers, [row["lora_learning_rate"] for row in history], where="mid", label="LoRA learning rate")
    axes[1, 0].step(epoch_numbers, [row["head_learning_rate"] for row in history], where="mid", label="Head learning rate")
    axes[1, 0].set(xlabel="Development epoch", ylabel="Learning rate", yscale="log")
    axes[1, 0].legend(fontsize=10)
    axes[1, 1].plot(epoch_numbers, [row["plateau"]["stale_epochs"] for row in history], label="Epochs without material improvement")
    axes[1, 1].axhline(reference["protocol"]["config"]["rule"]["plateau_patience"], linestyle=":", color="#777777", label="Rate-reduction patience")
    axes[1, 1].axhline(reference["protocol"]["config"]["rule"]["terminal_patience"], linestyle="--", color="#000000", label="Terminal patience")
    axes[1, 1].set(xlabel="Development epoch", ylabel="Plateau counter")
    axes[1, 1].legend(fontsize=10)
    for axis in axes.flat:
        axis.grid(alpha=.2)
    development_figure = _save_figure(figure, reports / "joint_convergence_development.png")
    figure, axes = plt.subplots(2, 1, figsize=(9.5, 7), constrained_layout=True)
    for seed, color in ((1993, "#0072b2"), (1994, "#d55e00"), (1995, "#009e73")):
        rows = result["jobs"][f"seed_{seed}"]["epochs"]
        for axis, metric in zip(axes, ("accuracy", "nll"), strict=True):
            axis.plot([row["epoch"] for row in rows], [row["fit_probe"][metric] for row in rows], color=color, label=f"Seed {seed}")
            axis.axvline(selection["endpoints"]["accuracy_selected"], color="#000000", linestyle=":", linewidth=.8)
    axes[0].set(ylabel="Clean fit-probe accuracy (%)", title="Full-data refits: diagnostic training curves, not test curves")
    axes[1].set(ylabel="Clean fit-probe NLL", xlabel="Full-data epoch")
    for axis in axes:
        axis.grid(alpha=.2)
        axis.legend(fontsize=10)
    full_figure = _save_figure(figure, reports / "joint_convergence_full_refits.png")
    selected = next(row for row in summary if row["role"] == "accuracy_selected")
    five = next(row for row in summary if row["role"] == "five_epoch")
    terminal = next(row for row in summary if row["role"] == "terminal")
    comparisons = tuple({"joint_condition": selected["condition"], "replay_condition": row["condition"], "stage": 50,
                         "joint_seeds": 3, "replay_seeds": 1,
                         "joint_accuracy_mean": selected["accuracy_mean"], "replay_accuracy": row["final_accuracy"],
                         "joint_minus_replay_accuracy_points": selected["accuracy_mean"] - row["final_accuracy"],
                         "joint_nll_mean": selected["nll_mean"], "replay_nll": row["final_nll"],
                         "joint_minus_replay_nll": selected["nll_mean"] - row["final_nll"]} for row in replay_summaries)
    selected_uniform = next((row for row in comparisons if row["replay_condition"] == "uniform_h4096_standard_rho80_unit8"), None)
    comparison_note = (() if selected_uniform is None else (
        "Against uniform replay with H=4,096, old=0.8, and the standard/unit-8 batch schedule, the selected joint mean differs by "
        f"{selected_uniform['joint_minus_replay_accuracy_points']:+.3f} accuracy points and {selected_uniform['joint_minus_replay_nll']:+.4f} NLL. "
        "This compares three joint seeds with one replay seed; it does not measure replay's seed variation.",))
    primary_per_seed = sorted((row for row in endpoints if row["role"] == "accuracy_selected"), key=lambda row: row["seed"])
    training_images = sum(job["training_presentations"] for job in result["jobs"].values())
    training_steps = sum(job["optimizer_steps"] for job in result["jobs"].values())
    all_forwards = sum(job["model_forward_images"] for job in result["jobs"].values()) + 6000 * len(result["evaluations"])
    training_minutes = sum(job["training_wall_seconds"] for job in result["jobs"].values()) / 60
    evaluation_minutes = (sum(job["evaluation_wall_seconds"] for job in result["jobs"].values())
                          + sum(row["metrics"]["wall_seconds"] for row in result["evaluations"])) / 60
    table = ReportTable(("Condition", "Epoch", "Test acc. mean +/- SD", "Test NLL mean +/- SD"), tuple(
        (row["label"], str(row["epoch"]), f"{row['accuracy_mean']:.3f}% +/- {row['accuracy_sample_sd']:.3f}",
         f"{row['nll_mean']:.4f} +/- {row['nll_sample_sd']:.4f}") for row in summary))
    plateau_text = (f"Development reached the predeclared validation-plateau rule after {len(history)} epochs."
                    if selection["validation_converged"] else f"Development hit the {len(history)}-epoch safety cap without establishing a plateau.")
    sections = (
        ReportSection("Task-50 rank-16 joint IID: longer-training reference", (
            plateau_text + f" The primary checkpoint is epoch {selected['epoch']}, selected by validation accuracy before any new test evaluation. "
            f"Three cold full-data refits reach {selected['accuracy_mean']:.3f}% mean test accuracy (sample SD {selected['accuracy_sample_sd']:.3f} points) "
            f"and {selected['nll_mean']:.4f} mean NLL (SD {selected['nll_sample_sd']:.4f}).",
            "Per-seed primary results: " + "; ".join(f"{row['seed']}: {row['accuracy']:.3f}% / {row['nll']:.4f} NLL" for row in primary_per_seed) + ".",
            f"Relative to the same three seeds' five-epoch endpoints, the selected recipe changes mean accuracy by "
            f"{selected['accuracy_mean'] - five['accuracy_mean']:+.3f} points and NLL by {selected['nll_mean'] - five['nll_mean']:+.4f}. "
            f"The terminal checkpoint changes mean accuracy by {terminal['accuracy_mean'] - selected['accuracy_mean']:+.3f} points relative to the selected checkpoint.",
            "All endpoints below were defined from development before test evaluation; none is a test-selected winner. The primary reference is the accuracy-selected row. "
            "The diamond on the full-stream figures shows its task-50 mean and across-seed sample SD. It is not a new stage-matched curve or a mathematical upper bound. "
            "The original 78.867% five-epoch model and its historical curve remain unchanged.",
            *comparison_note,
        ), table=table),
        ReportSection("Joint-IID development convergence and checkpoint selection", (
            "The existing 19,200/4,800 training-derived fit/validation partition selects the schedule. The model is the same pinned ViT-B/16, with rank/alpha-16 "
            "QKV and fc1 adapters and a 200-way affine head; every class is available jointly from the start. Initial SGD rates, momentum, decay, batch 64, and augmentation match the earlier joint recipe.",
            "An accuracy gain of at least 0.1 point or NLL reduction of at least 0.002 resets patience. Eight plateau epochs reduce both rates by five. "
            "After three reductions, twelve plateau epochs and at least thirty total epochs establish the declared validation plateau. Smaller gains accumulate against the last significant anchors. "
            "This is a generalization-plateau criterion, not proof of stationary training loss or a globally optimal classifier. Orange markers use raw validation metrics, not the plateau tolerances.",
        ), (development_figure,)),
        ReportSection("Full-data refits, work, and interpretation limits", (
            "Seeds 1993, 1994, and 1995 each restart from cold adapters and heads on all 24,000 training images. They replay the complete frozen development schedule by epochs. "
            "No validation or test metric can change those fits. The dotted vertical marker is the primary selected epoch. These are clean training-probe curves, not held-out estimates.",
            f"Development plus the three complete refits used {training_images:,} training forward/backward image pairs and {training_steps:,} optimizer updates. "
            f"Including development validation, fit probes, and final testing gives {all_forwards:,} forward image paths in total. "
            f"Measured training-batch work totals {training_minutes:.2f} minutes; evaluation and epoch artifact work totals {evaluation_minutes:.2f} minutes. "
            "Training-batch time includes image loading but excludes step-checkpoint writes, model setup, and between-job overhead. Preflight work is separate.",
            "Transferring the schedule by epochs gives each full-data fit 25% more updates per epoch than development. The three seeds measure variation under one selected recipe, "
            "not three independent convergence searches. A validation plateau does not prove this is the best possible optimizer, augmentation, regularization, or accuracy attainable by rank 16. "
            "Mean NLL is a predictive-loss measure, not a calibration-error estimate. Every training and evaluation identity is bound to the original dataset manifest; the published test set was previously examined in earlier experiments.",
        ), (full_figure,)),
    )
    return sections, {"joint_development": development_figure, "joint_full_refits": full_figure}, {
        "joint_convergence_epochs": epochs, "joint_convergence_endpoints": endpoints, "joint_convergence_summary": summary,
        "joint_convergence_resources": resources,
        **({"joint_convergence_comparisons": comparisons} if comparisons else {}),
    }

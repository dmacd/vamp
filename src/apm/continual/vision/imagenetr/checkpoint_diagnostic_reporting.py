"""Post-hoc probability and replay-retention pages for the existing SRT report."""

from __future__ import annotations

import math
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pyarrow.parquet as pq

from apm.continual.artifacts import file_sha256, record_sha256
from apm.continual.vision.imagenetr.srt_evidence import read_sealed

if TYPE_CHECKING:
    from apm.continual.vision.imagenetr.srt_reporting import ReportSection


LABELS = {**{f"offline_seed_{seed}": f"Offline joint IID, seed {seed}" for seed in (1993, 1994, 1995)},
          **{f"{method}_h4096_{profile}_rho80_unit8": f"{'SRT' if method == 'srt' else 'Uniform replay'}, {profile}"
             for profile in ("standard", "strict") for method in ("srt", "uniform")}}
COLORS = ("#003f5c", "#4c78a8", "#72a0c1", "#b5179e", "#488a32", "#d97706", "#00796b")
TABLE_NAMES = ("checkpoint_probability_summary", "checkpoint_temperature_fits", "checkpoint_test_predictions",
               "checkpoint_reliability", "checkpoint_prediction_parity", "checkpoint_training_cohorts",
               "checkpoint_training_predictions", "checkpoint_resources")


def load_checkpoint_diagnostics(source: Path, source_hash: str) -> dict[str, object] | None:
    """Authenticate the fixed diagnostic matrix and its compact per-image evidence."""
    source = source.resolve()
    pointer_path = source / "reports/checkpoint_diagnostics.json"
    if not pointer_path.is_file():
        return None
    pointer = read_sealed(pointer_path, "imagenetr50-checkpoint-diagnostics-pointer-v1")
    project = source.parents[4]
    root = (project / pointer["run"]).resolve()
    if not root.is_relative_to(project / "artifacts/imagenetr50/checkpoint_diagnostics/runs"):
        raise ValueError("checkpoint diagnostic pointer leaves its artifact namespace")
    result = read_sealed(root / "result.json", "imagenetr50-checkpoint-diagnostics-result-v1")
    protocol = read_sealed(root / "protocol.json", "imagenetr50-checkpoint-diagnostics-protocol-v1")
    original = read_sealed(source / "protocol.json")
    if (protocol["content_hash"] != root.name or result["protocol_hash"] != root.name or pointer["result_hash"] != result["content_hash"]
            or any(row["source_result_hash"] != source_hash for row in (pointer, result, protocol["config"]))
            or not result["source_unchanged"] or result["optimizer_steps"] != 0 or result["model_forward_images"] != 138000
            or protocol["dataset_hash"] != original["dataset_hash"] or protocol["model_sha256"] != original["model_sha256"]
            or set(result["tables"]) != set(TABLE_NAMES)):
        raise ValueError("checkpoint diagnostic source, work, or result identity changed")
    for name, digest in result["tables"].items():
        if file_sha256(root / "analysis" / f"{name}.parquet") != digest:
            raise ValueError("checkpoint diagnostic analysis table changed")
    tables = {name: tuple(pq.read_table(root / "analysis" / f"{name}.parquet").to_pylist()) for name in TABLE_NAMES}
    folds = tuple(pq.read_table(root / "folds.parquet").to_pylist())
    if record_sha256(folds) != protocol["folds_hash"] or len(folds) != 6000 or len({row["image_id"] for row in folds}) != 6000:
        raise ValueError("checkpoint diagnostic fold membership changed")
    fold_ids = {fold: tuple(row["image_id"] for row in folds if row["fold"] == fold) for fold in range(5)}
    if any(len(ids) != 1200 for ids in fold_ids.values()):
        raise ValueError("checkpoint diagnostic folds are unbalanced")
    predictions = tables["checkpoint_test_predictions"]
    fits = tables["checkpoint_temperature_fits"]
    if len(predictions) != 42000 or len(fits) != 35 or {row["condition"] for row in predictions} != set(LABELS):
        raise ValueError("checkpoint diagnostic endpoint matrix is incomplete")
    for fit in fits:
        calibration_ids = tuple(row["image_id"] for row in folds if row["fold"] != fit["fold"])
        if (fit["calibration_ids_hash"] != record_sha256(calibration_ids)
                or fit["evaluation_ids_hash"] != record_sha256(fold_ids[fit["fold"]])
                or fit["calibration_examples"] != 4800 or fit["evaluation_examples"] != 1200
                or not math.isfinite(fit["temperature"]) or fit["temperature"] <= 0):
            raise ValueError("checkpoint diagnostic calibration populations changed")
    summaries = tables["checkpoint_probability_summary"]
    for name in LABELS:
        rows = tuple(row for row in predictions if row["condition"] == name)
        if ([(row["image_id"], row["label"], row["fold"]) for row in rows] != [(row["image_id"], row["label"], row["fold"]) for row in folds]
                or any(row["raw_prediction"] != row["calibrated_prediction"] for row in rows)):
            raise ValueError("checkpoint diagnostic predictions are misaligned or changed class")
        for mode in ("raw", "calibrated"):
            summary = next(row for row in summaries if row["condition"] == name and row["score_mode"] == mode)
            if (not math.isclose(summary["nll"], mean(row[f"{mode}_nll"] for row in rows), abs_tol=1e-12)
                    or not math.isclose(summary["accuracy"], 100 * mean(row[f"{mode}_correct"] for row in rows), abs_tol=1e-12)):
                raise ValueError("checkpoint diagnostic summary differs from per-image scores")
    if (len(tables["checkpoint_training_predictions"]) != 48000 or len(tables["checkpoint_resources"]) != 11
            or sum(row["model_forward_images"] for row in tables["checkpoint_resources"]) != 138000
            or any(row["prediction_mismatches"] or row["maximum_nll_difference"] >= .00003 for row in tables["checkpoint_prediction_parity"])):
        raise ValueError("checkpoint diagnostic training population or raw parity is incomplete")
    return {"pointer": pointer, "protocol": protocol, "result": result, "tables": tables}


def probability_gap(reference: dict[str, object]) -> dict[str, float]:
    """Compare offline seed-mean and standard uniform NLL before/after cross-fitting."""
    rows = reference["tables"]["checkpoint_probability_summary"]
    return {mode: mean(row["nll"] for row in rows if row["condition"].startswith("offline_") and row["score_mode"] == mode)
            - next(row["nll"] for row in rows if row["condition"] == "uniform_h4096_standard_rho80_unit8" and row["score_mode"] == mode)
            for mode in ("raw", "calibrated")}


def plot_probability_diagnostics(reports: Path, tables: dict[str, tuple[dict[str, object], ...]]) -> Path:
    """Show positive-temperature effects on NLL and top-label reliability without altering accuracy."""
    figure, axes = plt.subplots(1, 2, figsize=(10, 5.3), constrained_layout=True)
    for axis, mode in zip(axes, ("raw", "calibrated"), strict=True):
        axis.plot([0, 1], [0, 1], color="#555555", linestyle=":", linewidth=1)
        for name, color in zip(LABELS, COLORS, strict=True):
            rows = tuple(row for row in tables["checkpoint_reliability"] if row["condition"] == name and row["score_mode"] == mode and row["examples"])
            axis.plot([row["confidence"] for row in rows], [row["accuracy"] for row in rows], "o-", color=color,
                      label=LABELS[name], markersize=4, linewidth=1.4)
        axis.set(xlabel="Mean probability of the predicted class", ylabel="Fraction predicted correctly",
                 title="Raw scores" if mode == "raw" else "Out-of-fold temperature", xlim=(0, 1), ylim=(0, 1))
        axis.grid(alpha=.2)
    figure.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=3, frameon=False, fontsize=10)
    path = reports / "checkpoint_reliability.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_training_review_age(reports: Path, tables: dict[str, tuple[dict[str, object], ...]]) -> Path:
    """Compare identical SRT-defined last-review bins under SRT and uniform models."""
    figure, axes = plt.subplots(2, 2, figsize=(9.5, 7), constrained_layout=True)
    groups = ((1, 10), (11, 20), (21, 30), (31, 39), (40, 49), (50, 50))
    minimum_accuracy = min(row["accuracy"] for row in tables["checkpoint_training_cohorts"]
                           if row["cohort"].startswith("last_review_") and row["accuracy"] is not None)
    for column, profile in enumerate(("standard", "strict")):
        for axis, metric in zip(axes[:, column], ("accuracy", "nll"), strict=True):
            for method, color in (("srt", "#b5179e"), ("uniform", "#00796b")):
                rows = tuple(next(row for row in tables["checkpoint_training_cohorts"]
                             if row["profile"] == profile and row["method"] == method and row["cohort"] == f"last_review_{lo:02d}_{hi:02d}")
                             for lo, hi in groups)
                axis.plot(range(6), [row[metric] for row in rows], "o-", color=color,
                          label="SRT" if method == "srt" else "Uniform replay", linewidth=2)
                if method == "srt" and metric == "accuracy":
                    for index, row in enumerate(rows):
                        if row["examples"]:
                            axis.annotate(f"n={row['examples']:,}", (index, row[metric]), xytext=(-3 if index == 5 else 0, -14),
                                          textcoords="offset points", ha="right" if index == 5 else "center", fontsize=9.5)
            axis.set(xticks=range(6), xticklabels=[f"{lo}-{hi}" if lo != hi else str(lo) for lo, hi in groups],
                     xlabel="Last training stage recorded under SRT", ylabel="Clean training accuracy (%)" if metric == "accuracy" else "Clean training NLL")
            if metric == "accuracy":
                axis.set_title(profile.capitalize() + " profile; same images in both curves")
                axis.set_ylim(max(0, 5 * math.floor(minimum_accuracy / 5) - 5), 103)
            axis.grid(axis="y", alpha=.2)
    figure.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=2, frameon=False, fontsize=11)
    path = reports / "checkpoint_training_review_age.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def checkpoint_report_parts(reports: Path, reference: dict[str, object] | None) -> tuple[tuple[ReportSection, ...], dict[str, Path], dict[str, tuple[dict[str, object], ...]]]:
    """Append four explicitly diagnostic pages and retain all compact analysis tables."""
    if reference is None:
        return (), {}, {}
    from apm.continual.vision.imagenetr.srt_reporting import ReportSection, ReportTable
    tables = reference["tables"]
    scores = {(row["condition"], row["score_mode"]): row for row in tables["checkpoint_probability_summary"]}
    gaps = probability_gap(reference)
    reduction_note = (f"The reduction is {100 * (1 - gaps['calibrated'] / gaps['raw']):.1f}% of the raw gap. "
                      if gaps["raw"] else "There is no raw mean gap to reduce. ")
    rows = tuple((LABELS[name], f"{scores[name, 'raw']['accuracy']:.3f}%", f"{scores[name, 'raw']['nll']:.4f}",
                  f"{scores[name, 'calibrated']['nll']:.4f}",
                  f"{scores[name, 'raw']['temperature_min']:.3f}-{scores[name, 'raw']['temperature_max']:.3f}") for name in LABELS)
    quality = ReportTable(("Condition", "Accuracy", "Raw NLL", "OOF NLL", "Fitted T"), rows)
    cohorts = {(row["profile"], row["cohort"], row["method"]): row for row in tables["checkpoint_training_cohorts"]}
    groups = (("all", "All training images"), ("unrevisited_tasks40_50", "No reviews in tasks 40-50"),
              ("unrevisited_and_last_probability_ge90", "No late reviews; last p(true) >= 0.9"),
              ("top1pct_presentations", "Top 1% SRT presentation count"))
    retention = ReportTable(("Condition", "Images", "SRT acc.", "Uniform acc.", "SRT NLL", "Uniform NLL"), tuple(
        (f"{profile.capitalize()}: {label}", f"{srt['examples']:,}", f"{srt['accuracy']:.2f}%", f"{uniform['accuracy']:.2f}%",
         f"{srt['nll']:.3f}", f"{uniform['nll']:.3f}")
        for profile in ("standard", "strict") for group, label in groups
        for srt, uniform in ((cohorts[profile, group, "srt"], cohorts[profile, group, "uniform"]),)))
    stale = tuple(cohorts["standard", "unrevisited_and_last_probability_ge90", method] for method in ("srt", "uniform"))
    unrevisited = tuple(cohorts["standard", "unrevisited_tasks40_50", method] for method in ("srt", "uniform"))
    whole = tuple(cohorts["standard", "all", method] for method in ("srt", "uniform"))
    repeated = tuple(cohorts["standard", "top1pct_presentations", method] for method in ("srt", "uniform"))
    strict_fit = tuple(cohorts["strict", "all", method] for method in ("srt", "uniform"))
    strict_test = tuple(scores[f"{method}_h4096_strict_rho80_unit8", "raw"] for method in ("srt", "uniform"))
    parity = max(row["maximum_nll_difference"] for row in tables["checkpoint_prediction_parity"])
    bound_count = sum(row["at_numerical_bound"] for row in tables["checkpoint_temperature_fits"])
    figures = {"checkpoint_reliability": plot_probability_diagnostics(reports, tables),
               "checkpoint_review_age": plot_training_review_age(reports, tables)}
    sections = (
        ReportSection("Checkpoint diagnostic: does confidence scale explain NLL?", (
            "These are the already-trained task-50 rank-16 checkpoints, with no further optimizer updates. Offline denotes the three replay-schedule-matched joint-IID seeds. "
            "Every SRT/uniform row uses H=4,096, old=0.8, unit=8 and seed 1993; standard and strict identify the original threshold profiles.",
            "Divide every logit by one positive temperature T. This changes probabilities, not the winning class. Five fixed class-stratified folds each contain 1,200 test images. "
            "For each checkpoint, fit T on the other 4,800 and score only the held-out 1,200. OOF means the combined out-of-fold scores; the T column spans the five fits.",
            f"The offline three-seed mean minus standard uniform NLL is {gaps['raw']:+.4f} before calibration and {gaps['calibrated']:+.4f} after it. "
            + reduction_note + "This tests a global confidence-scale explanation for that NLL difference. "
            "All 42,000 predicted classes remain unchanged. The table retains every seed, not a selected winner.",
            "This is a post-hoc diagnosis on a previously inspected test set, not a new benchmark score or an untouched validation study. "
            "No image's label fits its own temperature. All original raw results and main accuracy curves remain unchanged.",
        ), table=quality),
        ReportSection("Checkpoint diagnostic: probability reliability", (
            "Each point groups predictions into one of fifteen fixed probability intervals. Its horizontal position is the mean probability assigned to the winning class; "
            "its vertical position is the fraction correct. The dotted diagonal is agreement between confidence and accuracy. Empty bins are omitted; sparse bins can fluctuate strongly.",
            "A single temperature can correct a common score scale, but cannot change class rankings or repair image-dependent errors. "
            "Brier scores, entropy, bin counts, calibration errors, and per-image raw/OOF scores are retained in the matching analysis tables.",
            f"Standard SRT still has {scores['srt_h4096_standard_rho80_unit8', 'calibrated']['nll']:.4f} calibrated NLL versus "
            f"uniform's {scores['uniform_h4096_standard_rho80_unit8', 'calibrated']['nll']:.4f}. Temperature scaling does not explain away the SRT deficit: "
            "its worse class predictions remain. Offline seed variation and single-seed replay also limit conclusions about the small residual offline/uniform gap.",
            f"Temperatures were constrained only by numerical bounds 0.01-100; {bound_count} of 35 fits reached a bound. "
            f"Independent float64 log-sum-exp reproduced all original winning classes; the largest per-image NLL difference was {parity:.3g}.",
        ), (figures["checkpoint_reliability"],)),
        ReportSection("Checkpoint diagnostic: did neglected training images stay learned?", (
            "These are clean-view predictions on all 24,000 training images, not test accuracy. Define every group using SRT's recorded history, "
            "then score those exact same images with both SRT and its paired uniform checkpoint. The uniform column is not a separately selected uniform-history group.",
            f"Under standard SRT, {stale[0]['examples']:,} images had no reviews in tasks 40-50 and last recorded p(true) >= 0.9. "
            f"At the final clean evaluation, SRT misclassifies {stale[0]['now_wrong']:,}; uniform misclassifies {stale[1]['now_wrong']:,} on those same images. "
            f"Their accuracies are {stale[0]['accuracy']:.2f}% and {stale[1]['accuracy']:.2f}% respectively.",
            f"The broader group with no reviews in tasks 40-50 contains {unrevisited[0]['examples']:,} images and accounts for "
            f"{unrevisited[0]['now_wrong'] - unrevisited[1]['now_wrong']:,} of the net {whole[0]['now_wrong'] - whole[1]['now_wrong']:,} "
            "additional training errors under standard SRT. Its deficit is therefore not solely worse generalization on unseen test images.",
            "The high-exposure group is the top 240 images by SRT presentation count, with ties broken by image ID. "
            f"Standard-profile accuracy on these repeatedly selected images is {repeated[0]['accuracy']:.2f}% under SRT and {repeated[1]['accuracy']:.2f}% under uniform. "
            "This is consistent with concentrated effort on persistently difficult images alongside neglected, otherwise learnable images. "
            "It does not establish that a particular review was wasted or measure its gradient influence.",
            f"Strict SRT is an important qualification: overall training accuracy is {strict_fit[0]['accuracy']:.2f}% versus {strict_fit[1]['accuracy']:.2f}%, "
            f"yet test accuracy is {strict_test[0]['accuracy']:.2f}% versus {strict_test[1]['accuracy']:.2f}%. "
            "It also learns the most-repeated group better than uniform. Stale confidence does not by itself explain the full held-out deficit; "
            "better fitting of selected training cases need not improve generalization.",
        ), table=retention),
        ReportSection("Checkpoint diagnostic: review age and remaining uncertainty", (
            "Both curves in each panel use the same SRT-defined image groups; n is their shared count. Empty groups have no point. "
            "A late last review is not a randomized intervention: difficulty, class, task arrival, and past mistakes all affect membership.",
            "The previous confidence came from an augmented presentation before its optimizer update; the new score uses the final model, a clean view, and all 200 classes. "
            "Their difference combines later learning, crop changes, competing classes, and the last update. It is not a matched-view measurement of forgetting.",
            "The next causal control should keep realized batches and old/current counts fixed while changing replay selection: compare a maximum review-gap rule with a cap on repeated-image allocation. "
            "Neither intervention has run here. Replay endpoints remain single-seed; five calibration folds are not five independent model fits.",
            f"Diagnostic work: 138,000 forward image paths, zero optimizer steps; {sum(row['wall_seconds'] for row in tables['checkpoint_resources']) / 60:.2f} measured collection minutes. "
            "This time excludes setup, model restoration, report generation, and extra audits. Source identities and every model parameter/buffer are checked; "
            "immutable 1,024-image chunks allow completed inference to be reused.",
        ), (figures["checkpoint_review_age"],)),
    )
    return sections, figures, tables

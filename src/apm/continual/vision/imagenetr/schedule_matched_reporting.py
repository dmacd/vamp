"""Authenticated offline schedule-control results added to the existing SRT report."""

from __future__ import annotations

import math
from pathlib import Path
from statistics import mean, stdev
import textwrap
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pyarrow.parquet as pq

from apm.continual.artifacts import file_sha256, publish_immutable_json, record_sha256
from apm.continual.vision.imagenetr.joint_convergence_reporting import endpoint_summary
from apm.continual.vision.imagenetr.schedule_matched_joint import ScheduleControlConfig, source_schedule
from apm.continual.vision.imagenetr.schedule_matched_training import ScheduledBatch, require_schedule, validate_schedule_job
from apm.continual.vision.imagenetr.srt_evidence import read_sealed, sealed_record, write_parquet

if TYPE_CHECKING:
    from apm.continual.vision.imagenetr.srt_reporting import ReportSection


CONDITION = "joint_rank16_replay_schedule_matched"
LABEL = "Offline joint IID rank 16, replay-schedule matched"
PLOT_LABEL = LABEL + " (task 50; mean +/- SD)"
COLOR = "#003f5c"


def prediction_loss_parts(condition: str, seed: int, predictions: list[dict[str, object]]) -> dict[str, object]:
    """Decompose test NLL into correct/error contributions without changing predictions."""
    if not predictions:
        raise ValueError("prediction diagnostics require a nonempty test population")
    groups = {name: tuple(row["nll"] for row in predictions if (row["prediction"] == row["label"]) == correct)
              for name, correct in (("correct", True), ("wrong", False))}
    return {"condition": condition, "seed": seed, "examples": len(predictions),
            "nll": math.fsum(row["nll"] for row in predictions) / len(predictions),
            **{f"{name}_{field}": value for name, values in groups.items() for field, value in (
                ("examples", len(values)), ("mean_nll", math.fsum(values) / len(values) if values else None),
                ("nll_contribution", math.fsum(values) / len(predictions)))}}


def load_schedule_reference(source: Path, source_hash: str) -> dict[str, object] | None:
    """Require exact source schedule, isolated populations, all seeds, and reconstructed test metrics."""
    source = source.resolve()
    pointer_path = source / "reports/schedule_matched_joint.json"
    if not pointer_path.is_file():
        return None
    pointer = read_sealed(pointer_path, "imagenetr50-schedule-matched-pointer-v1")
    project = source.parents[4]
    root = (project / pointer["run"]).resolve()
    if not root.is_relative_to(project / "artifacts/imagenetr50/schedule_matched_joint_r16/runs"):
        raise ValueError("offline control pointer leaves its artifact namespace")
    result = read_sealed(root / "result.json", "imagenetr50-schedule-matched-result-v1")
    protocol = read_sealed(root / "protocol.json", "imagenetr50-schedule-matched-protocol-v1")
    original = read_sealed(source / "result.json")
    original_protocol = read_sealed(source / "protocol.json")
    if (root.name != protocol["content_hash"] or result["protocol_hash"] != root.name
            or pointer["result_hash"] != result["content_hash"] or pointer["source_result_hash"] != source_hash
            or result["source_result_hash"] != source_hash or protocol["source_result_hash"] != source_hash
            or original["content_hash"] != source_hash or not result["zero_step_reuse"] or not result["source_unchanged"]
            or any(protocol[name] != original_protocol[name] for name in ("dataset_hash", "model_sha256"))):
        raise ValueError("offline control identity or original source changed")
    config = ScheduleControlConfig(**{**protocol["config"], "seeds": tuple(protocol["config"]["seeds"])})
    if config.seeds != (1993, 1994, 1995) or config.source_result_hash != source_hash:
        raise ValueError("offline control configuration differs from the fixed seed/source matrix")
    source_batches, source_evidence = source_schedule(source, config)
    schedule_record = read_sealed(root / "schedule.json")
    if file_sha256(root / "schedule.parquet") != schedule_record["schedule_parquet_sha256"]:
        raise ValueError("offline control schedule file changed")
    batches = tuple(ScheduledBatch(**row) for row in pq.read_table(root / "schedule.parquet").to_pylist())
    if (batches != source_batches or protocol["source_job_hash"] != source_evidence["source_job_hash"]
            or require_schedule(batches, 64) != protocol["schedule_hash"] or result["schedule_hash"] != protocol["schedule_hash"]
            or record_sha256({name: schedule_record[name] for name in source_evidence}) != record_sha256(source_evidence)):
        raise ValueError("offline control differs from the exact replay schedule")
    populations = read_sealed(root / "populations.json")
    fitting, probe, test = (set(populations[name]) for name in ("fitting", "probe", "test"))
    if ((len(fitting), len(probe), len(test)) != (24000, 2048, 6000) or fitting & test or not probe <= fitting
            or any(len(populations[name]) != len(set(populations[name])) for name in ("fitting", "probe", "test"))):
        raise ValueError("offline control training/test populations are not isolated")
    index = pq.read_table(source / "final/image_index.parquet").to_pylist()
    if fitting != {row["image_id"] for row in index}:
        raise ValueError("offline control does not use the complete original training population")
    names = {f"seed_{seed}" for seed in (1993, 1994, 1995)}
    if set(result["jobs"]) != names or set(result["reuse"]) != names or set(result["draw_audits"]) != names:
        raise ValueError("offline control seed matrix is incomplete")
    for name in sorted(names):
        folder = root / "seeds" / name
        job = validate_schedule_job(folder)
        definition = read_sealed(folder / "job.json")
        audit = read_sealed(folder / "draw_audit.json")
        if (result["jobs"][name] != job or result["draw_audits"][name] != audit or audit["result_hash"] != job["content_hash"]
                or definition["protocol_hash"] != root.name or definition["fitting_examples"] != 24000
                or definition["probe_examples"] != 2048 or definition["fitting_ids_hash"] != record_sha256(populations["fitting"])
                or definition["probe_ids_hash"] != record_sha256(populations["probe"])
                or definition["schedule_hash"] != protocol["schedule_hash"] or definition["optimizer"] != protocol["optimizer"]
                or definition["optimizer_config_hash"] != protocol["optimizer_config_hash"]
                or job["optimizer_steps"] != 56243 or job["training_presentations"] != 844640
                or job["model_forward_images"] != 947040 or name != f"seed_{job['seed']}"
                or len(job["blocks"]) != 50 or job["test_used"] or job["stop_reason"] != "complete_frozen_schedule"
                or audit["verified_updates"] != 56243 or audit["verified_presentations"] != 844640
                or not audit["training_only"] or not audit["schedule_and_rates_matched"]
                or result["reuse"][name]["optimizer_steps"] or result["reuse"][name]["result_hash"] != job["content_hash"]):
            raise ValueError("offline control job, draw audit, or reuse evidence differs")
    original_predictions = source / "final/srt_h1024/stages/050/predictions.parquet"
    if file_sha256(original_predictions) != original["conditions"]["srt_h1024"]["rows"][-1]["predictions_sha256"]:
        raise ValueError("original test population evidence changed")
    labels = {row["image_id"]: (row["label"], row["task"]) for row in pq.read_table(original_predictions).to_pylist()}
    if set(labels) != test or len(result["evaluations"]) != 3 or {row["seed"] for row in result["evaluations"]} != {1993, 1994, 1995}:
        raise ValueError("offline test population or endpoint matrix differs")
    quality = ()
    for row in result["evaluations"]:
        folder = root / "evaluations" / f"seed_{row['seed']}"
        if (read_sealed(folder / "result.json") != row or row["protocol_hash"] != root.name
                or row["model_sha256"] != result["jobs"][f"seed_{row['seed']}"]["blocks"][-1]["model_sha256"]
                or file_sha256(folder / "predictions.parquet") != row["predictions_sha256"]):
            raise ValueError("offline test predictions changed")
        predictions = pq.read_table(folder / "predictions.parquet").to_pylist()
        if len(predictions) != 6000 or {value["image_id"]: (value["label"], value["task"]) for value in predictions} != labels:
            raise ValueError("offline test identities differ")
        accuracy = 100 * sum(value["prediction"] == value["label"] for value in predictions) / 6000
        nll = math.fsum(value["nll"] for value in predictions) / 6000
        if (row["metrics"]["examples"] != 6000 or not math.isclose(accuracy, row["metrics"]["accuracy"], abs_tol=1e-10)
                or not math.isclose(nll, row["metrics"]["nll"], abs_tol=1e-10)):
            raise ValueError("offline reconstructed metrics disagree")
        quality += (prediction_loss_parts(CONDITION, row["seed"], predictions),)
    source_folder = source / "followups" / config.followup_hash / "final" / config.source_condition / "stages/050"
    source_stage = read_sealed(source_folder / "result.json")
    if (source_stage["content_hash"] != source_evidence["blocks"][-1]["source_stage_hash"]
            or file_sha256(source_folder / "predictions.parquet") != source_stage["predictions_sha256"]):
        raise ValueError("matched replay source predictions changed")
    source_predictions = pq.read_table(source_folder / "predictions.parquet").to_pylist()
    if len(source_predictions) != 6000 or {row["image_id"]: (row["label"], row["task"]) for row in source_predictions} != labels:
        raise ValueError("matched replay source test identities differ")
    source_quality = prediction_loss_parts(config.source_condition, 1993, source_predictions)
    if (not math.isclose(source_quality["nll"], source_stage["evaluation"]["nll"], abs_tol=1e-10)
            or not math.isclose(100 * source_quality["correct_examples"] / 6000, source_stage["evaluation"]["accuracy"], abs_tol=1e-10)):
        raise ValueError("matched replay source reconstructed metrics disagree")
    return {"pointer": pointer, "protocol": protocol, "result": result, "schedule": schedule_record,
            "prediction_quality": (*quality, source_quality)}


def control_summary(reference: dict[str, object]) -> dict[str, object]:
    """Summarize the fixed final endpoint without test-based checkpoint selection."""
    rows = reference["result"]["evaluations"]
    if len(rows) != 3 or {row["seed"] for row in rows} != {1993, 1994, 1995}:
        raise ValueError("offline summary requires exactly three seeds")
    return {"condition": CONDITION, "label": LABEL, "stage": 50, "seeds": 3,
            **{f"{metric}_{suffix}": function(row["metrics"][metric] for row in rows)
               for metric in ("accuracy", "nll") for suffix, function in (("mean", mean), ("sample_sd", stdev))},
            "training_presentations_per_seed": 844640, "optimizer_steps_per_seed": 56243}


def export_schedule_updates(source: Path, reference: dict[str, object] | None) -> dict[str, object] | None:
    """Consolidate authenticated update chunks into one immutable analysis table per seed."""
    if reference is None:
        return None
    root = source.resolve().parents[4] / reference["pointer"]["run"]
    exports = ()
    for name, job in sorted(reference["result"]["jobs"].items()):
        folder = root / "seeds" / name
        if any(file_sha256(folder / chunk["path"]) != chunk["sha256"] for chunk in job["chunks"]):
            raise ValueError("offline update export source changed")
        rows = tuple(row for chunk in job["chunks"] for row in pq.read_table(folder / chunk["path"]).to_pylist())
        if len(rows) != job["optimizer_steps"] or sum(row["batch_size"] for row in rows) != job["training_presentations"]:
            raise ValueError("offline update export counts disagree")
        path = folder / "update_metrics.parquet"
        digest = write_parquet(path, rows)
        first_block = tuple(row for row in rows if row["block"] == 1)
        loss_peak = max(first_block, key=lambda row: row["loss_sum"] / row["batch_size"])
        gradient_peak = max(first_block, key=lambda row: row["gradient_norm"])
        exports += ({"seed": job["seed"], "path": str(path.relative_to(root)), "sha256": digest,
                     "job_result_hash": job["content_hash"], "source_chunks_hash": record_sha256(job["chunks"]),
                     "optimizer_steps": len(rows), "training_presentations": job["training_presentations"],
                     "first_block_updates": len(first_block), "early_peak_batch_nll": loss_peak["loss_sum"] / loss_peak["batch_size"],
                     "early_peak_loss_step": loss_peak["step"], "early_peak_loss_batch_size": loss_peak["batch_size"],
                     "early_peak_gradient_norm": gradient_peak["gradient_norm"], "early_peak_gradient_step": gradient_peak["step"]},)
    record = sealed_record({"schema_version": "imagenetr50-schedule-update-export-v1", "protocol_hash": root.name,
                            "result_hash": reference["result"]["content_hash"], "tables": exports,
                            "purpose": "Exact per-update analysis tables; original checkpoint chunks and weights remain local"})
    publish_immutable_json(root / "update_export.json", record)
    return record


def draw_schedule_endpoint(axis: plt.Axes, references: dict[str, object], metric: str = "accuracy") -> None:
    """Show only the final task-50 mean/SD, never a future-informed CL curve."""
    reference = references.get("schedule_matched_joint")
    if reference is None:
        return
    row = control_summary(reference)
    axis.errorbar([50], [row[f"{metric}_mean"]], yerr=[row[f"{metric}_sample_sd"]], fmt="s",
                  color=COLOR, markerfacecolor="white", markeredgewidth=1.7, markersize=7, capsize=4,
                  linewidth=1.7, zorder=21, label=PLOT_LABEL)
    axis.set_xlim(1, 51)


def schedule_report_parts(
    reports: Path, reference: dict[str, object] | None, joint: dict[str, object] | None,
    replay: tuple[dict[str, object], ...], update_export: dict[str, object] | None,
) -> tuple[tuple[ReportSection, ...], dict[str, Path], dict[str, tuple[dict[str, object], ...]]]:
    """Append direct comparisons, training diagnostics, and exact optimizer-work accounting."""
    if reference is None:
        return (), {}, {}
    from apm.continual.vision.imagenetr.srt_reporting import ReportSection, ReportTable
    result, summary = reference["result"], control_summary(reference)
    evaluations = sorted(result["evaluations"], key=lambda row: row["seed"])
    endpoints = tuple({"condition": CONDITION, "seed": row["seed"], "accuracy": row["metrics"]["accuracy"],
                       "nll": row["metrics"]["nll"], "examples": row["metrics"]["examples"],
                       "optimizer_steps": 56243, "training_presentations": 844640,
                       "model_sha256": row["model_sha256"], "predictions_sha256": row["predictions_sha256"]} for row in evaluations)
    blocks = tuple({"seed": job["seed"], "work_block": row["block"], "optimizer_steps": row["steps_total"],
                    "training_presentations": row["presentations_total"], "mean_batch_size": row["training_presentations"] / row["optimizer_steps"],
                    "fit_probe_accuracy": row["fit_probe"]["accuracy"], "fit_probe_nll": row["fit_probe"]["nll"],
                    "augmented_preupdate_accuracy": row["training_accuracy"], "augmented_preupdate_nll": row["training_nll"],
                    **{name: row[name] for name in ("training_wall_seconds", "checkpoint_wall_seconds", "evaluation_and_artifact_seconds", "model_sha256")}}
                   for job in result["jobs"].values() for row in job["blocks"])
    resources = tuple({"condition": CONDITION, "seed": job["seed"],
                       **{name: job[name] for name in ("optimizer_steps", "training_presentations", "model_forward_images", "training_wall_seconds",
                                                      "checkpoint_wall_seconds", "evaluation_and_artifact_seconds")},
                       "test_forward_images": 6000, "all_forward_images": job["model_forward_images"] + 6000,
                       "test_wall_seconds": next(row["metrics"]["wall_seconds"] for row in evaluations if row["seed"] == job["seed"])}
                      for job in result["jobs"].values())
    if update_export is None:
        raise ValueError("offline report requires authenticated per-update diagnostics")
    stability = tuple({**{name: row[name] for name in ("seed", "first_block_updates", "early_peak_batch_nll", "early_peak_loss_step",
                                                      "early_peak_loss_batch_size", "early_peak_gradient_norm", "early_peak_gradient_step")},
                       **{f"block_{block}_fit_accuracy": result["jobs"][f"seed_{row['seed']}"]["blocks"][block - 1]["fit_probe"]["accuracy"]
                          for block in (1, 5, 50)}} for row in update_export["tables"])
    comparisons = ({**summary, "display": "Offline joint, replay-schedule matched"},)
    if joint is not None:
        comparisons += tuple({**row, "display": {"accuracy_selected": "Joint, validation accuracy-selected", "five_epoch": "Joint, five epochs",
                                                "nll_selected": "Joint, validation NLL-selected", "terminal": f"Joint, terminal epoch {row['epoch']}"}[row["role"]]}
                             for row in endpoint_summary(joint))
    comparisons += tuple({"condition": row["condition"], "display": row["condition"].replace("uniform_h4096_standard_rho80_unit8", "Uniform replay, matched source")
                           .replace("uniform_h1024", "Uniform replay, H=1,024"),
                           "accuracy_mean": row["final_accuracy"], "accuracy_sample_sd": None, "nll_mean": row["final_nll"], "nll_sample_sd": None,
                           "seeds": 1} for row in replay if row["condition"] in ("uniform_h4096_standard_rho80_unit8", "uniform_h1024"))
    comparison_rows = tuple({**{name: row[name] for name in ("condition", "display", "seeds", "accuracy_mean", "accuracy_sample_sd", "nll_mean", "nll_sample_sd")},
                             "control_minus_reference_accuracy_points": summary["accuracy_mean"] - row["accuracy_mean"],
                             "control_minus_reference_nll": summary["nll_mean"] - row["nll_mean"]} for row in comparisons)
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 5), sharey=True, constrained_layout=True)
    for axis, metric, title in zip(axes, ("accuracy", "nll"), ("Task-50 accuracy (%)", "Task-50 negative log likelihood"), strict=True):
        for index, row in enumerate(comparison_rows):
            color = COLOR if row["condition"] == CONDITION else "#666666"
            axis.errorbar(row[f"{metric}_mean"], index, xerr=row[f"{metric}_sample_sd"], color=color,
                          fmt="s" if row["condition"] == CONDITION else "o", capsize=4, markersize=6)
        axis.set(xlabel=title, yticks=range(len(comparison_rows)), yticklabels=[textwrap.fill(row["display"], 27) for row in comparison_rows])
        axis.grid(axis="x", alpha=.25)
    axes[0].invert_yaxis()
    comparison_path = reports / "schedule_matched_joint_comparison.png"
    figure.savefig(comparison_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    figure, axes = plt.subplots(3, 1, figsize=(9.5, 9), constrained_layout=True)
    for seed in (1993, 1994, 1995):
        rows = tuple(row for row in blocks if row["seed"] == seed)
        for axis, metric in zip(axes[:2], ("fit_probe_accuracy", "fit_probe_nll"), strict=True):
            axis.plot([row["optimizer_steps"] / 1000 for row in rows], [row[metric] for row in rows], label=f"Seed {seed}")
    rows = tuple(row for row in blocks if row["seed"] == 1993)
    axes[2].plot([row["optimizer_steps"] / 1000 for row in rows], [row["mean_batch_size"] for row in rows], color=COLOR)
    for axis, ylabel in zip(axes, ("Clean fit-probe accuracy (%)", "Clean fit-probe NLL (log scale)", "Mean batch size in work block"), strict=True):
        axis.set(xlabel="Cumulative optimizer updates (thousands)", ylabel=ylabel)
        axis.grid(alpha=.2)
    axes[1].set_yscale("log")
    axes[0].legend(loc="lower right")
    axes[0].set_title("Offline all-data training: diagnostics, not held-out or CL curves")
    diagnostic_path = reports / "schedule_matched_joint_diagnostics.png"
    figure.savefig(diagnostic_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    source = next(row for row in comparison_rows if row["condition"] == "uniform_h4096_standard_rho80_unit8")
    joint_selected = next((row for row in comparison_rows if row["condition"] == "joint_convergence_accuracy_selected"), None)
    difference = (f"Relative to the prior validation accuracy-selected joint mean, accuracy changes by "
                  f"{joint_selected['control_minus_reference_accuracy_points']:+.3f} points and NLL by {joint_selected['control_minus_reference_nll']:+.4f}. "
                  if joint_selected is not None else "")
    quality = reference["prediction_quality"]
    offline_quality = tuple(row for row in quality if row["condition"] == CONDITION)
    source_quality = next(row for row in quality if row["condition"] != CONDITION)
    outcome = (f"The offline mean is {abs(source['control_minus_reference_accuracy_points']):.3f} accuracy points "
               f"{'above' if source['control_minus_reference_accuracy_points'] >= 0 else 'below'} the single-seed replay source. "
               f"Its NLL is {abs(source['control_minus_reference_nll']):.4f} "
               f"{'higher' if source['control_minus_reference_nll'] >= 0 else 'lower'}. "
               + difference + "Similar top-1 accuracy does not imply similar true-label probabilities. These are observed differences, not an equivalence or significance test.")
    sections = (
        ReportSection("Offline joint IID with the replay optimizer schedule", (
            "This control keeps the rank-16 architecture and copies every batch size from uniform H=4,096, standard/old=0.8/unit=8: "
            "56,243 optimizer updates and 844,640 training-image presentations per seed. From the first update, all 24,000 training images "
            "can be sampled and all 200 classifier rows are active. Sampling is uniform over images without replacement within a batch, independent between batches.",
            "The source batch size averages 15.018 images (median 10); only 1,735 of 56,243 updates use a full batch of 64. "
            "The model has 1,480,904 trainable adapter/classifier parameters, identical to the replay source.",
            "SGD retains momentum 0.9, weight decay 0.0005, constant LoRA rate 0.0005, and constant head rate 0.01. Each actual batch uses mean "
            "cross-entropy and one full update; no gradient accumulation or rate reductions occur. Source work boundaries do not reset weights, "
            "change the class set, or select training images. There is no validation search or early stopping in this control.",
            f"The fixed final endpoint averages {summary['accuracy_mean']:.3f}% test accuracy (sample SD {summary['accuracy_sample_sd']:.3f} points) "
            f"and {summary['nll_mean']:.4f} NLL (SD {summary['nll_sample_sd']:.4f}). All three cold fits completed before any new test evaluation. "
            "The hollow blue square on the main figures shows this task-50 endpoint, not a future-informed continual-learning curve.",
            difference + f"Against the single-seed matched replay source, the differences are {source['control_minus_reference_accuracy_points']:+.3f} accuracy "
            f"points and {source['control_minus_reference_nll']:+.4f} NLL. These are observed differences, not significance tests.",
            "Compared with the previous epoch-trained joint fits, batch sizes, update count, learning-rate schedule, and between-batch sampling differ. "
            "A gain over that reference cannot be assigned to any one of those changes. The exact schedule match is with the replay source.",
        ), table=ReportTable(("Seed / summary", "Task-50 accuracy", "Task-50 NLL", "Optimizer updates"),
                             tuple((str(row["seed"]), f"{row['accuracy']:.3f}%", f"{row['nll']:.4f}", "56,243") for row in endpoints)
                             + (("Mean +/- sample SD", f"{summary['accuracy_mean']:.3f}% +/- {summary['accuracy_sample_sd']:.3f}",
                                 f"{summary['nll_mean']:.4f} +/- {summary['nll_sample_sd']:.4f}", "56,243 per seed"),))),
        ReportSection("Task-50 comparisons and what this control isolates", (
            "All rows use the same test population and rank-16 adapter architecture. Error bars show across-seed sample standard deviation, "
            "not a confidence interval. Replay rows are single-seed results, so no seed-variation bar is available. Every previous joint endpoint "
            "keeps its original validation selection; no checkpoint was chosen from these test comparisons.",
            "This comparison holds the optimizer schedule fixed while changing the staged class/data curriculum and cumulative image weighting together. "
            "It does not isolate those remaining effects from one another. The source schedule itself came from a single seed's SRT partner; "
            "three offline seeds do not replicate that replay arm or schedule selection. This follow-up was requested after inspecting earlier test results.",
            outcome,
            f"Averaged across offline seeds, correctly classified images contribute {mean(row['correct_nll_contribution'] for row in offline_quality):.4f} "
            f"to whole-test NLL, versus {source_quality['correct_nll_contribution']:.4f} for replay. Errors contribute "
            f"{mean(row['wrong_nll_contribution'] for row in offline_quality):.4f}, versus {source_quality['wrong_nll_contribution']:.4f}. "
            "These two contributions sum to total NLL. The error sets are not identical; this decomposition is not a paired-error test or a full calibration measurement.",
        ), (comparison_path,)),
        ReportSection("Offline control training diagnostics and measured work", (
            "The same 2,048 hash-selected clean training images are probed at the fifty source work boundaries. These are training-fit diagnostics, "
            "not validation or test curves. The horizontal axis is optimizer work: the model already has access to every training class at the left edge. "
            "The bottom panel shows the inherited changing batch size, identical across the three seeds.",
            f"The three fits total {sum(row['training_presentations'] for row in resources):,} forward/backward training-image pairs and "
            f"{sum(row['optimizer_steps'] for row in resources):,} optimizer updates. Including clean probes and final tests gives "
            f"{sum(row['all_forward_images'] for row in resources):,} forward image paths. Measured training-batch time totals "
            f"{sum(row['training_wall_seconds'] for row in resources) / 60:.2f} minutes; checkpoint writes add "
            f"{sum(row['checkpoint_wall_seconds'] for row in resources) / 60:.2f} minutes, and probe/test/model-artifact work adds "
            f"{sum(row['evaluation_and_artifact_seconds'] + row['test_wall_seconds'] for row in resources) / 60:.2f} minutes. "
            "Batch time includes data loading; setup, source/draw audits, and preflight are separate. Counts are model image paths, not profiled FLOPs.",
            "Every committed draw and augmentation ordinal was reconstructed against the all-training population. Final exposure counts and "
            "per-update schedule/rate checks agree. Completed training and prediction artifacts are immutable and reused without optimizer steps.",
        ), (diagnostic_path,)),
        ReportSection("Early optimization: individual seeds and loss spikes", (
            "This diagnostic was added after observing slow early learning in seed 1995, before any new test evaluation. "
            "It does not select a checkpoint or alter the fixed training schedule. Each seed changes initialization, sampled images, and augmentation; "
            "these observations do not isolate which of those differences caused a different trajectory.",
            "The table summarizes the first source work block. Peak batch NLL is mean pre-update cross-entropy in the worst batch; its actual batch size "
            "is shown alongside it. Peak gradient norm is the Euclidean norm over all trainable-parameter gradients before SGD, and can occur at a "
            "different update. Fit accuracy uses the same 2,048 clean training images at the indicated update counts. "
            "The exact peak-update indices and all batch records are retained in the analysis tables.",
            "These are unmodified finite updates, including very small batches, at the full prescribed learning rates. Large early losses and subsequent "
            "poor fitting are consistent with optimization instability, but do not identify a unique failing layer or prove that any particular clipping "
            "threshold would repair it. Other seeds can recover despite early spikes. No seed was reset, discarded, or replaced.",
            "All three final test results remain in the main mean and standard deviation. The individual-seed table and training curves are essential "
            "when trajectories differ; a mean alone can obscure that difference. A stability intervention such as warm-up, a lower initial rate, or "
            "different early batching would require a separate experiment. None was applied here.",
        ), table=ReportTable(("Seed", "Peak batch NLL", "Batch size at peak loss", "Peak gradient norm",
                              *(f"Fit acc. after {result['jobs']['seed_1993']['blocks'][block - 1]['steps_total']:,} updates" for block in (1, 5)), "Final fit acc."),
                             tuple((str(row["seed"]), f"{row['early_peak_batch_nll']:.3f}", str(row["early_peak_loss_batch_size"]),
                                    f"{row['early_peak_gradient_norm']:.1f}", f"{row['block_1_fit_accuracy']:.2f}%",
                                    f"{row['block_5_fit_accuracy']:.2f}%", f"{row['block_50_fit_accuracy']:.2f}%") for row in stability))),
    )
    return sections, {"schedule_comparison": comparison_path, "schedule_diagnostics": diagnostic_path}, {
        "schedule_matched_endpoints": endpoints, "schedule_matched_summary": (summary,), "schedule_matched_blocks": blocks,
        "schedule_matched_resources": resources, "schedule_matched_comparisons": comparison_rows,
        "schedule_matched_stability": stability, "schedule_matched_prediction_quality": quality,
    }

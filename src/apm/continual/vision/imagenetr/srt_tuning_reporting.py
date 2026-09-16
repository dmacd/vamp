"""Authenticated H=128 validation-search evidence and pages in the shared SRT report."""

from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pyarrow.parquet as pq

from apm.continual.artifacts import file_sha256, record_sha256
from apm.continual.vision.imagenetr.srt_config import SRTConfig
from apm.continual.vision.imagenetr.srt_evidence import read_sealed
from apm.continual.vision.imagenetr.srt_tuning_config import (
    SRTTuningConfig, candidate_from_record, policy_candidates, rate_candidates, refinement_candidates, select_winner,
)

if TYPE_CHECKING:
    from apm.continual.vision.imagenetr.srt_analysis import ReplayAnalysis
    from apm.continual.vision.imagenetr.srt_reporting import ReportSection


def load_tuning_reference(source: Path, source_hash: str) -> dict[str, object] | None:
    """Reconstruct validation-only selection and authenticate the selected final pair."""
    source = source.resolve()
    pointer_path = source / "reports/srt_h128_tuning.json"
    if not pointer_path.is_file():
        return None
    pointer = read_sealed(pointer_path, "imagenetr50-srt-tuning-pointer-v1")
    root = (source / "tuning" / pointer["run_hash"]).resolve()
    if root.parent != source / "tuning":
        raise ValueError("tuning pointer leaves its artifact namespace")
    protocol = read_sealed(root / "protocol.json", "imagenetr50-srt-tuning-protocol-v1")
    result = read_sealed(root / "result.json", "imagenetr50-srt-tuning-result-v1")
    selection = read_sealed(root / "selection.json", "imagenetr50-srt-tuning-selection-v1")
    original = read_sealed(source / "protocol.json")
    if (root.name != protocol["content_hash"] or result["protocol_hash"] != root.name or selection["protocol_hash"] != root.name
            or pointer["result_hash"] != result["content_hash"] or result["selection_hash"] != selection["content_hash"]
            or any(row["source_result_hash"] != source_hash for row in (pointer, protocol, result))
            or not result["source_unchanged"] or not result["zero_step_reuse"] or selection["test_used"]
            or protocol["test_used_for_selection"] or protocol["early_task_elimination"]
            or protocol["dataset_hash"] != original["dataset_hash"] or protocol["model_sha256"] != original["model_sha256"]):
        raise ValueError("H=128 tuning provenance or validation-only selection changed")
    config = SRTTuningConfig(**{name: tuple(value) if isinstance(value, list) else value for name, value in protocol["config"].items()})
    values = read_sealed(source / "config_resolved.json")
    training = SRTConfig(**{name: (tuple((profile, tuple(thresholds)) for profile, thresholds in value) if name == "profiles"
                                 else tuple(value) if name in {"budgets", "historical_fractions", "interval_units"} else value)
                           for name, value in values.items() if name != "content_hash"})
    split = read_sealed(root / "calibration_split.json")
    if split != read_sealed(source / "calibration_split.json") or split["content_hash"] != protocol["split_hash"]:
        raise ValueError("tuning validation membership changed")
    full_index = pq.read_table(source / "final/image_index.parquet").to_pylist()
    fitting_ids, validation_ids = set(split["fit_image_ids"]), set(split["validation_image_ids"])
    fit = tuple(row for row in full_index if row["image_id"] in fitting_ids)
    validation = tuple(row for row in full_index if row["image_id"] in validation_ids)
    if ((len(fit), len(validation), len(full_index)) != (19200, 4800, 24000) or fitting_ids & validation_ids
            or record_sha256([row["image_id"] for row in fit]) != protocol["fit_ids_hash"]
            or record_sha256([row["image_id"] for row in validation]) != protocol["validation_ids_hash"]):
        raise ValueError("tuning population hashes do not describe the original training-only split")
    rows = tuple(selection["candidates"])
    by_hash = {row["candidate_hash"]: row for row in rows}
    if len(by_hash) != len(rows) or len(rows) > config.maximum_candidates:
        raise ValueError("tuning repeats a recipe or exceeds its finite search")
    accumulated, phases, expected = (), {}, policy_candidates(config, training)
    for phase in ("policy", "rates", "refinement"):
        decision = read_sealed(root / f"calibration/{phase}_selection.json")
        requested = [candidate.content_hash for candidate in expected]
        accumulated = tuple(dict.fromkeys((*accumulated, *requested)))
        winner = select_winner(tuple(by_hash[identity] for identity in accumulated))
        if (decision["phase"] != phase or decision["test_used"] or decision["requested_candidates"] != requested
                or decision["completed_or_failed_candidates"] != list(accumulated) or decision["selected"] != winner
                or decision["candidate_summary_hashes"] != [by_hash[identity]["content_hash"] for identity in accumulated]):
            raise ValueError("tuning phase decision differs from the frozen search rule")
        phases = {**phases, **{identity: phase for identity in requested if identity not in phases}}
        expected = (rate_candidates(config, training, candidate_from_record(winner["candidate"])) if phase == "policy"
                    else refinement_candidates(config, candidate_from_record(winner["candidate"])))
    if set(accumulated) != set(by_hash) or selection["selected"] != select_winner(rows):
        raise ValueError("final tuning selection is not its maximum final-validation-accuracy candidate")
    expected_labels = {row["image_id"]: row["class_id"] for row in validation}
    for row in rows:
        candidate = candidate_from_record(row["candidate"])
        candidate_root = root / "calibration" / candidate.name
        if (candidate.content_hash != row["candidate_hash"] or row != read_sealed(candidate_root / "candidate_result.json")
                or row["validation_ids_hash"] != protocol["validation_ids_hash"] or row["evaluation_role"] != "validation"):
            raise ValueError("tuning candidate summary or evaluation population changed")
        if row["status"] == "failed":
            if row["exception"] != "FloatingPointError":
                raise ValueError("an unexpected candidate failure was suppressed")
            continue
        job = read_sealed(candidate_root / "job.json")
        fitted = read_sealed(candidate_root / "result.json")
        final = fitted["rows"][-1]
        predictions_path = candidate_root / "stages/050/predictions.parquet"
        if (len(fitted["rows"]) != 50 or fitted["content_hash"] != row["result_hash"] or fitted["job_hash"] != job["content_hash"]
                or job["protocol_hash"] != root.name or job["config_hash"] != candidate.optimizer_config(training).content_hash
                or job["training_ids_hash"] != protocol["fit_ids_hash"] or job["evaluation_ids_hash"] != protocol["validation_ids_hash"]
                or job["method"] != "srt" or job["capacity"] != 128
                or job["policy"] != {**asdict(candidate.policy), "thresholds": list(candidate.policy.thresholds)}
                or fitted["image_presentations"] != 101888 or row["image_presentations"] != 101888
                or file_sha256(predictions_path) != final["predictions_sha256"]
                or file_sha256(candidate_root / "stages/050/model.safetensors") != final["model_sha256"]):
            raise ValueError("validation candidate job, work or endpoint artifacts changed")
        predictions = pq.read_table(predictions_path).to_pylist()
        if (len(predictions) != 4800 or {item["image_id"]: item["label"] for item in predictions} != expected_labels
                or not math.isclose(row["validation_accuracy"], 100 * sum(item["prediction"] == item["label"] for item in predictions) / 4800, abs_tol=1e-10)
                or not math.isclose(row["validation_nll"], math.fsum(item["nll"] for item in predictions) / 4800, abs_tol=1e-6)):
            raise ValueError("final validation scores do not reconstruct from the held-out predictions")
        reuse = result["reuse"][candidate.name]
        if reuse["optimizer_steps"] != 0 or reuse["result_hash"] != fitted["content_hash"]:
            raise ValueError("candidate completed-job reuse failed")
    chosen = candidate_from_record(selection["selected"]["candidate"])
    baseline_root = source / "followups" / protocol["baseline_protocol_hash"]
    baseline = read_sealed(baseline_root / "result.json")
    baseline_job = read_sealed(baseline_root / "final/srt_h128_standard_rho80_unit8/job.json")
    baseline_candidate = candidate_from_record({
        "policy": baseline_job["policy"], "lora_learning_rate": training.lora_learning_rate,
        "head_learning_rate": training.head_learning_rate})
    if (baseline["content_hash"] != protocol["baseline_result_hash"]
            or baseline_job["training_ids_hash"] != record_sha256([row["image_id"] for row in full_index])
            or result["baseline_reused"] != (chosen == baseline_candidate)):
        raise ValueError("selected recipe baseline identity or reuse decision changed")
    roots = {name: (source / path).resolve() for name, path in result["condition_roots"].items()}
    expected_names = {f"{method}_h128_{'standard_rho80_unit8' if result['baseline_reused'] else 'tuned'}" for method in ("srt", "uniform")}
    if set(roots) != set(result["conditions"]) or set(roots) != expected_names:
        raise ValueError("selected final SRT/uniform pair is incomplete")
    final_protocol = protocol["baseline_protocol_hash"] if result["baseline_reused"] else root.name
    for name, job_root in roots.items():
        allowed = source / "followups" / final_protocol if result["baseline_reused"] else root
        if job_root != allowed / "final" / name:
            raise ValueError("selected job leaves its declared artifact namespace")
        job, fitted = read_sealed(job_root / "job.json"), read_sealed(job_root / "result.json")
        if (fitted != result["conditions"][name] or fitted["job_hash"] != job["content_hash"]
                or job["protocol_hash"] != final_protocol or job["config_hash"] != chosen.optimizer_config(training).content_hash
                or job["training_ids_hash"] != baseline_job["training_ids_hash"]
                or job["evaluation_ids_hash"] != baseline_job["evaluation_ids_hash"]
                or job["method"] != name.split("_", 1)[0] or job["capacity"] != 128
                or job["policy"] != {**asdict(chosen.policy), "thresholds": list(chosen.policy.thresholds)}
                or len(fitted["rows"]) != 50 or fitted["image_presentations"] != 121088
                or result["reuse"][name]["optimizer_steps"] != 0 or result["reuse"][name]["result_hash"] != fitted["content_hash"]):
            raise ValueError("selected final job differs from its frozen validation choice")
        if name.startswith("uniform_") and job["paired_job_hash"] != read_sealed(roots[name.replace("uniform_", "srt_", 1)] / "job.json")["content_hash"]:
            raise ValueError("selected uniform job uses a different SRT schedule")
    return {"root": root, "pointer": pointer, "protocol": protocol, "selection": selection,
            "result": result, "roots": roots, "candidate_phases": phases}


def tuning_report_parts(
    reports: Path, reference: dict[str, object] | None, references: dict[str, object], analyses: dict[str, ReplayAnalysis],
) -> tuple[tuple[ReportSection, ...], dict[str, Path], dict[str, tuple[dict[str, object], ...]]]:
    """Show the chosen H=128 pair, every validation candidate and separately charged search cost."""
    if reference is None:
        return (), {}, {}
    from apm.continual.vision.imagenetr.srt_reporting import (
        ALL_STYLES, CONDITION_LABELS, LABEL_STAGE_JOINT, ReportSection, ReportTable,
        _draw_task50_endpoints, _finish_axis, _save_figure, _shared_legend,
    )
    selection, result = reference["selection"], reference["result"]
    chosen = candidate_from_record(selection["selected"]["candidate"])
    candidate_rows = tuple({
        "index": index, "candidate_hash": row["candidate_hash"], "job": row["job"],
        "phase": reference["candidate_phases"][row["candidate_hash"]], "status": row["status"],
        "selected": row["candidate_hash"] == chosen.content_hash,
        **row["candidate"]["policy"], "lora_learning_rate": row["candidate"]["lora_learning_rate"],
        "head_learning_rate": row["candidate"]["head_learning_rate"],
        **{key: row.get(key) for key in ("validation_accuracy", "validation_nll", "mean_stage_accuracy", "image_presentations",
                                       "optimizer_steps", "training_wall_seconds", "validation_forward_images", "validation_wall_seconds")},
        "committed_failed_presentations": row.get("committed_image_presentations"),
        "committed_failed_optimizer_steps": row.get("committed_optimizer_steps"),
        "failed_invocation_wall_seconds": row.get("failed_invocation_wall_seconds"),
    } for index, row in enumerate(selection["candidates"], 1))
    conditions = tuple(dict.fromkeys(("srt_h128_standard_rho80_unit8", "uniform_h128_standard_rho80_unit8", *result["conditions"])))
    figure, axes = plt.subplots(2, 1, figsize=(9.5, 10), constrained_layout=True)
    for name in conditions:
        rows = analyses[name].stages
        for axis, metric in zip(axes, ("accuracy", "nll"), strict=True):
            color, style = ALL_STYLES[name]
            axis.plot([row["stage"] for row in rows], [row[metric] for row in rows], color=color, linestyle=style,
                      linewidth=2, label=CONDITION_LABELS[name])
    axes[0].plot(range(1, 51), [row["accuracy"] for row in references["stage_matched_joint"]], color="#222222",
                 linestyle=":", linewidth=1.5, label=LABEL_STAGE_JOINT)
    for axis, metric, ylabel in zip(axes, ("accuracy", "nll"), ("Seen-class test accuracy (%)", "Raw test negative log likelihood"), strict=True):
        _draw_task50_endpoints(axis, references, metric)
        _finish_axis(axis, ylabel, "H=128: original policy versus final-validation-accuracy selection")
        axis.set_xlim(1, 51)
    _shared_legend(figure, tuple(axes))
    comparison = _save_figure(figure, reports / "h128_tuning_comparison.png")
    figure, axis = plt.subplots(figsize=(9.5, 4.5), constrained_layout=True)
    for phase, color in (("policy", "#005b96"), ("rates", "#d97706"), ("refinement", "#6a1b9a")):
        rows = tuple(row for row in candidate_rows if row["phase"] == phase and row["status"] == "complete")
        axis.scatter([row["index"] for row in rows], [row["validation_accuracy"] for row in rows], color=color, s=45, label=phase)
    best = next(row for row in candidate_rows if row["selected"])
    axis.scatter([best["index"]], [best["validation_accuracy"]], facecolors="none", edgecolors="black", marker="o", s=155,
                 linewidths=1.5, label="Selected by final validation accuracy")
    axis.set(xlabel="Distinct candidate in execution order", ylabel="Task-50 validation accuracy (%)",
             title="Every point reached task 50; no test scores enter selection")
    axis.grid(alpha=.2)
    _shared_legend(figure, (axis,))
    search_figure = _save_figure(figure, reports / "h128_validation_search.png")
    rows = tuple((CONDITION_LABELS[name], f"{analyses[name].totals['final_accuracy']:.3f}%", f"{analyses[name].totals['final_nll']:.4f}",
                  f"{analyses[name].totals['mean_stage_accuracy']:.3f}%", f"{analyses[name].totals['optimizer_steps']:,}") for name in conditions)
    selected_srt = analyses[next(name for name in result["conditions"] if name.startswith("srt_"))].totals
    original_srt = analyses["srt_h128_standard_rho80_unit8"].totals
    sections = (
        ReportSection("H=128: hyperparameters selected for final accuracy", (
            "The search used only the existing training-derived 19,200/4,800 fit/validation split. Every candidate trained all fifty tasks. "
            "The objective was task-50 validation accuracy; ties used lower validation NLL and then candidate identity. Mean-stage accuracy did not select the winner.",
            f"Selected thresholds: {list(chosen.policy.thresholds)}; old target {chosen.policy.historical_fraction:g}; interval unit {chosen.policy.interval_unit}; "
            f"LoRA/head learning rates {chosen.lora_learning_rate:g}/{chosen.head_learning_rate:g}. "
            f"Task-50 validation accuracy was {best['validation_accuracy']:.3f}%, with raw NLL {best['validation_nll']:.4f}.",
            ("The selected recipe exactly equals the original H=128 recipe. Its completed SRT/uniform pair is reused without training or duplicate curves."
             if result["baseline_reused"] else
             "The choice was sealed before cold full-data refits. The new uniform control copies the chosen SRT batch schedule and optimizer settings; it is not independently tuned uniform replay."),
            f"Selected SRT changes task-50 test accuracy by {selected_srt['final_accuracy'] - original_srt['final_accuracy']:+.3f} percentage points "
            f"and raw NLL by {selected_srt['final_nll'] - original_srt['final_nll']:+.4f} relative to the original H=128 SRT recipe. "
            "All final H=128 streams use 121,088 training image presentations; the number and size of updates can change with the selected policy.",
            "This is a single-seed finite coordinate search, not a global optimum. Validation was reused from earlier studies and the test set had already been inspected. "
            "No offline reference was a gate; raw benchmark NLL is unchanged by calibration.",
        ), table=ReportTable(("Condition", "Task-50 acc.", "Raw NLL", "Mean acc.", "Updates"), rows)),
        ReportSection("H=128: original and selected recipes across all tasks", (
            "Both panels use the same condition names, colors and line styles. Solid lines are SRT and dashed lines their exact-schedule uniform partners. "
            "The two pairs share H but can have different optimizer schedules. The dotted curve is the original stage-matched joint rank-16 accuracy reference.",
            "Offline symbols are task-50 means across three seeds, not new stage curves or H=128 work matches. Only the selected SRT recipe gets a new test refit; "
            "the remaining search candidates are compared on validation, not test.",
        ), (comparison,)),
        ReportSection("H=128: validation search and its separate cost", (
            f"The frozen search evaluated {len(candidate_rows)} distinct recipes: eighteen policy combinations, learning-rate changes around the policy winner, "
            "then one-coordinate local refinements. Exact duplicate recipes reused their existing artifacts. All phase decisions are saved and reconstructed by the report audit.",
            f"Completed candidates consumed {sum(row['image_presentations'] or 0 for row in candidate_rows):,} training presentations, "
            f"{sum(row['optimizer_steps'] or 0 for row in candidate_rows):,} optimizer updates and "
            f"{sum(row['validation_forward_images'] or 0 for row in candidate_rows):,} validation forwards. "
            f"Measured training took {sum(row['training_wall_seconds'] or 0 for row in candidate_rows) / 60:.2f} minutes and validation "
            f"{sum(row['validation_wall_seconds'] or 0 for row in candidate_rows) / 60:.2f} minutes. These search costs are not charged to one final stream's deployment cost.",
            f"Numerically failed candidates: {sum(row['status'] == 'failed' for row in candidate_rows)}. Any failed candidate has a recorded cause and committed-work lower bound; "
            "a failed in-flight batch may not have a checkpoint. Search and final-run resource tables distinguish these costs.",
        ), (search_figure,)),
    )
    for offset in range(0, len(candidate_rows), 16):
        rows = tuple((f"{row['index']}{'*' if row['selected'] else ''}", row["phase"],
                      f"{row['validation_accuracy']:.3f}" if row["status"] == "complete" else "FAILED",
                      f"{row['validation_nll']:.4f}" if row["status"] == "complete" else "-",
                      f"{row['profile']}; {row['historical_fraction']:g}/{row['interval_unit']}",
                      f"{row['lora_learning_rate']:g}/{row['head_learning_rate']:g}") for row in candidate_rows[offset:offset + 16])
        sections += (ReportSection(f"H=128: complete validation candidates {offset + 1}-{offset + len(rows)}", (
            "The starred candidate is the final validation choice, not a test-selected checkpoint. Accuracies are percentages on validation at task 50; NLL is raw. "
            "Old/unit gives the requested historical fraction and interval unit. Full threshold values, hashes, work and per-stage evidence are retained in the candidate ledger.",
        ), table=ReportTable(("Candidate", "Phase", "Val. acc.", "Val. NLL", "Profile; old/unit", "LoRA/head LR"), rows)),)
    return sections, {"h128_tuning_comparison": comparison, "h128_validation_search": search_figure}, {"h128_tuning_candidates": candidate_rows}

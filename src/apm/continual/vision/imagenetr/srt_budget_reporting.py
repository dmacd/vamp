"""Fixed-policy budget comparisons within the existing SRT report."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd

from apm.continual.vision.imagenetr.srt_analysis import ReplayAnalysis

if TYPE_CHECKING:
    from apm.continual.vision.imagenetr.srt_reporting import ReportSection


def budget_report_parts(
    reports: Path, references: dict[str, object], analyses: dict[str, ReplayAnalysis],
) -> tuple[tuple[ReportSection, ...], dict[str, Path]]:
    """Compare H=512 with identical-policy older budgets and explicitly labeled offline endpoints."""
    from apm.continual.vision.imagenetr.srt_reporting import (
        ALL_STYLES, LABEL_STAGE_JOINT, LOW_BUDGET_STYLES,
        ReportSection, ReportTable, _draw_task50_endpoints, _finish_axis, _save_figure, _shared_legend,
    )
    from apm.continual.vision.imagenetr.joint_convergence_reporting import endpoint_summary
    from apm.continual.vision.imagenetr.schedule_matched_reporting import control_summary

    if not set(LOW_BUDGET_STYLES) <= analyses.keys():
        return (), {}
    conditions = tuple((capacity, method, f"{method}_h{capacity}" + ("" if capacity == 1024 else "_standard_rho80_unit8"))
                       for capacity in (512, 1024, 4096) for method in ("srt", "uniform"))
    conditions = tuple(row for row in conditions if row[2] in analyses)
    figure, axes = plt.subplots(2, 1, figsize=(9.5, 10.5), constrained_layout=True)
    for capacity, method, name in conditions:
        rows = pd.DataFrame(analyses[name].stages)
        color, style = ALL_STYLES[name]
        label = f"{'SRT' if method == 'srt' else 'Uniform replay'} rank 16, H={capacity:,}"
        for axis, metric in zip(axes, ("accuracy", "nll"), strict=True):
            axis.plot(rows.stage, rows[metric], color=color, linestyle=style, linewidth=2, label=label)
    axes[0].plot(range(1, 51), [row["accuracy"] for row in references["stage_matched_joint"]],
                 color="#222222", linestyle=":", linewidth=1.5, label=LABEL_STAGE_JOINT)
    for axis, metric, ylabel in zip(axes, ("accuracy", "nll"), ("Seen-class test accuracy (%)", "Test negative log likelihood"), strict=True):
        _draw_task50_endpoints(axis, references, metric)
        _finish_axis(axis, ylabel, "Standard profile, old=0.8, unit=8; budget is the only configured change")
        axis.set_xlim(1, 51)  # Keep task-50 endpoint markers and error-bar caps inside the panel.
    _shared_legend(figure, tuple(axes))
    path = _save_figure(figure, reports / "standard_budget_comparison.png")
    summaries = tuple((capacity, method, analyses[name].totals) for capacity, method, name in conditions)
    rows = tuple((f"{'SRT' if method == 'srt' else 'Uniform replay'} rank 16, H={capacity:,}", f"{row['final_accuracy']:.3f}%",
                  f"{row['final_nll']:.4f}", f"{row['training_presentations']:,}", f"{row['optimizer_steps']:,}")
                 for capacity, method, row in summaries)
    offline_note = ()
    if references.get("joint_convergence") is not None:
        summary = next(row for row in endpoint_summary(references["joint_convergence"]) if row["role"] == "accuracy_selected")
        rows += (("Joint IID, validation selected", f"{summary['accuracy_mean']:.3f}%", f"{summary['nll_mean']:.4f}",
                  f"{summary['training_presentations_per_seed_to_endpoint']:,}", f"{summary['optimizer_steps_per_seed_to_endpoint']:,}"),)
    if references.get("schedule_matched_joint") is not None:
        summary = control_summary(references["schedule_matched_joint"])
        rows += (("Joint IID, H=4,096 schedule", f"{summary['accuracy_mean']:.3f}%", f"{summary['nll_mean']:.4f}", "844,640", "56,243"),)
        offline_note = (f"The newer offline rank-16 control reaches {summary['accuracy_mean']:.3f}% mean accuracy "
                        f"(sample SD {summary['accuracy_sample_sd']:.3f} points, three seeds), not the older five-epoch joint recipe's endpoint. "
                        "It matches the standard H=4,096 optimizer schedule, not H=512. Neither offline reference gates this experiment.",)
    srt, uniform = (analyses[name].totals for name in LOW_BUDGET_STYLES)
    sections = (
        ReportSection("Lower replay budget: H=512", (
            "Two fresh 50-task streams use standard thresholds, old target 0.8 and interval unit 8, with the original seed 1993, model, data and optimizer. "
            "Only H changes. H limits presentation work, not stored history; all arrived training images remain available. "
            "The proposed review-age and repetition-cap interventions are not applied.",
            f"SRT finishes at {srt['final_accuracy']:.3f}% accuracy / {srt['final_nll']:.4f} raw NLL; matched uniform at "
            f"{uniform['final_accuracy']:.3f}% / {uniform['final_nll']:.4f}. SRT minus uniform is "
            f"{srt['final_accuracy'] - uniform['final_accuracy']:+.3f} accuracy points and {srt['final_nll'] - uniform['final_nll']:+.4f} NLL. "
            f"Mean stage accuracy is {srt['mean_stage_accuracy']:.3f}% / {uniform['mean_stage_accuracy']:.3f}% respectively. "
            f"Each uses {srt['training_presentations']:,} training forward/backward image pairs and {srt['optimizer_steps']:,} updates. "
            f"Training takes {srt['training_wall_seconds'] / 60:.2f} / {uniform['training_wall_seconds'] / 60:.2f} minutes; "
            "evaluation is separate in the resource ledger.",
            *offline_note,
            "Replay rows are single-seed results; offline rows are means across three seeds. These are exploratory results on an already inspected test set, "
            "not evidence of significance or a new SOTA claim. "
            "Lower budgets also change actual batch sizes and review schedules. Raw NLL remains the benchmark score; no new temperature is fitted.",
        ), table=ReportTable(("Condition", "Task-50 acc.", "Raw NLL", "Training pairs", "Updates"), rows)),
        ReportSection("Full-stream accuracy and NLL across fixed-policy budgets", (
            "Both panels use identical condition names, colors and line styles: solid SRT and dashed uniform, with one color per H. "
            "All six replay curves use standard thresholds, old target 0.8 and interval unit 8. Uniform copies its own SRT partner's batch schedule; "
            "schedules are not matched across budgets.",
            "The dotted accuracy curve is the original stage-matched joint rank-16 recipe. Its stage NLL was not retained, so no NLL curve is invented. "
            "Offline validation-selected and replay-schedule-matched markers are three-seed task-50 means with sample SD, not full-stream curves. "
            "The earlier report's other frontier and joint comparisons remain in their original panels.",
        ), (path,)),
    )
    return sections, {"standard_budget_comparison": path}

"""Publication-style report for the ImageNet-R replay-adaptation diagnosis."""

from __future__ import annotations

from base64 import b64encode
from collections.abc import Mapping, Sequence
from html import escape
from pathlib import Path
import csv
import math

from apm.continual.artifacts import (
    ChainedJsonlLedger,
    atomic_write,
    canonical_json_bytes,
    load_canonical_json,
    record_sha256,
)
from apm.continual.vision.imagenetr.integrator_pdf import render_integrator_pdf
from apm.continual.vision.imagenetr.integrator_reporting import _lineage_plot
from apm.continual.vision.imagenetr.replay_adaptation_workflow import (
    FULL_HISTORY_CONDITION,
)


CONDITION_LABELS = {
    "static__example_uniform__carry": "Fixed replay / example loss / carry Adam",
    "static__example_uniform__reset_each_stage": "Fixed replay / example loss / reset Adam",
    "static__task_uniform__carry": "Fixed replay / task-balanced / carry Adam",
    "static__task_uniform__reset_each_stage": "Fixed replay / task-balanced / reset Adam",
    "rotating__example_uniform__carry": "Rotating replay / example loss / carry Adam",
    "rotating__example_uniform__reset_each_stage": "Rotating replay / example loss / reset Adam",
    "rotating__task_uniform__carry": "Rotating replay / task-balanced / carry Adam",
    "rotating__task_uniform__reset_each_stage": "Rotating replay / task-balanced / reset Adam",
    FULL_HISTORY_CONDITION: "Fresh full-history integrator",
}
ONLINE_IDS = tuple(key for key in CONDITION_LABELS if key != FULL_HISTORY_CONDITION)
STAGES = (31, 50)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = tuple(sorted({key for row in rows for key in row}))
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_table_family(
    root: Path, name: str, rows: Sequence[Mapping[str, object]]
) -> None:
    projected = tuple(dict(row) for row in rows)
    _write_csv(root / f"{name}.csv", projected)
    atomic_write(root / f"{name}.json", canonical_json_bytes({"rows": projected}))
    try:
        import pandas as pd

        pd.DataFrame(projected).to_parquet(root / f"{name}.parquet", index=False)
    except ImportError:  # pragma: no cover - vision environment gate
        pass


def _validated_result(run: Path) -> dict[str, object]:
    result = load_canonical_json(run / "evaluations" / "result.json")
    core = {key: value for key, value in result.items() if key != "content_hash"}
    if (
        result.get("schema_version") != "imagenetr50-replay-adaptation-result-v1"
        or result.get("content_hash") != record_sha256(core)
        or len(result.get("online_conditions", ())) != 8
    ):
        raise ValueError("replay-adaptation result does not authenticate")
    return result


def _condition_rows(result: Mapping[str, object]) -> tuple[dict[str, object], ...]:
    development = {
        (str(row["condition"]), int(row["stage"])): dict(row)
        for row in result["development_metrics"]
    }
    test = {
        (str(row["condition"]), int(row["stage"])): dict(row)
        for row in result["test_metrics"]
    }
    expected = {
        (condition, stage)
        for condition in CONDITION_LABELS
        for stage in STAGES
    }
    if set(development) != expected or set(test) != expected:
        raise ValueError("diagnostic result lacks a complete stage/condition matrix")
    rows = []
    for condition, stage in sorted(expected, key=lambda value: (value[1], value[0])):
        dev = development[condition, stage]
        locked = test[condition, stage]
        fit = dict(dev["fit_population"])
        validation = dict(dev["validation"])
        selected = dict(dev["selected_training"])
        test_metrics = dict(locked["metrics"])
        controls = dict(locked["controls"])
        fields = condition.split("__")
        rows.append(
            {
                "condition": condition,
                "condition_label": CONDITION_LABELS[condition],
                "current_task_test_accuracy": float(
                    test_metrics["current_task_accuracy"]
                ),
                "fit_population_accuracy": float(fit["accuracy"]),
                "fit_to_validation_gap_pp": float(fit["accuracy"])
                - float(validation["accuracy"]),
                "joint_iid_test_accuracy": float(
                    dict(result["stage_matched_joint_iid"])[str(stage)]
                ),
                "kind": str(dev["kind"]),
                "old_task_macro_test_accuracy": (
                    None
                    if test_metrics["old_task_macro_accuracy"] is None
                    else float(test_metrics["old_task_macro_accuracy"])
                ),
                "optimizer": "fresh" if len(fields) == 1 else fields[2],
                "raw_union_test_accuracy": float(controls["raw_union"]),
                "sampler": "full_history" if len(fields) == 1 else fields[0],
                "selected_training_accuracy": float(selected["accuracy"]),
                "stage": stage,
                "test_accuracy": float(test_metrics["accuracy"]),
                "true_node_oracle_test_accuracy": float(
                    controls["true_node_oracle"]
                ),
                "validation_accuracy": float(validation["accuracy"]),
                "weighting": "full_history" if len(fields) == 1 else fields[1],
            }
        )
    return tuple(rows)


def _factor_rows(
    rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    lookup = {
        (str(row["condition"]), int(row["stage"])): row
        for row in rows
        if row["condition"] in ONLINE_IDS
    }
    comparisons = []
    for stage in STAGES:
        for metric in ("validation_accuracy", "test_accuracy"):
            for weighting in ("example_uniform", "task_uniform"):
                for optimizer in ("carry", "reset_each_stage"):
                    left = f"static__{weighting}__{optimizer}"
                    right = f"rotating__{weighting}__{optimizer}"
                    comparisons.append(
                        {
                            "baseline": left,
                            "comparison": right,
                            "delta_pp": float(lookup[right, stage][metric])
                            - float(lookup[left, stage][metric]),
                            "factor": "rotating_minus_fixed",
                            "metric": metric,
                            "stage": stage,
                        }
                    )
            for sampler in ("static", "rotating"):
                for optimizer in ("carry", "reset_each_stage"):
                    left = f"{sampler}__example_uniform__{optimizer}"
                    right = f"{sampler}__task_uniform__{optimizer}"
                    comparisons.append(
                        {
                            "baseline": left,
                            "comparison": right,
                            "delta_pp": float(lookup[right, stage][metric])
                            - float(lookup[left, stage][metric]),
                            "factor": "task_balanced_minus_example",
                            "metric": metric,
                            "stage": stage,
                        }
                    )
            for sampler in ("static", "rotating"):
                for weighting in ("example_uniform", "task_uniform"):
                    left = f"{sampler}__{weighting}__carry"
                    right = f"{sampler}__{weighting}__reset_each_stage"
                    comparisons.append(
                        {
                            "baseline": left,
                            "comparison": right,
                            "delta_pp": float(lookup[right, stage][metric])
                            - float(lookup[left, stage][metric]),
                            "factor": "reset_minus_carried_adam",
                            "metric": metric,
                            "stage": stage,
                        }
                    )
    return tuple(comparisons)


def _factor_summary(rows: Sequence[Mapping[str, object]]) -> tuple[dict[str, object], ...]:
    keys = sorted({(str(row["factor"]), str(row["metric"])) for row in rows})
    return tuple(
        {
            "factor": factor,
            "mean_delta_pp": math.fsum(
                float(row["delta_pp"])
                for row in rows
                if row["factor"] == factor and row["metric"] == metric
            )
            / sum(
                row["factor"] == factor and row["metric"] == metric for row in rows
            ),
            "metric": metric,
        }
        for factor, metric in keys
    )


def _selection_rows(run: Path) -> tuple[dict[str, object], ...]:
    rows = []
    for sampler in ("static", "rotating"):
        condition = f"{sampler}__example_uniform__carry"
        ledger = ChainedJsonlLedger(
            run / "integrators" / "online" / condition / "training_metrics.jsonl",
            "imagenetr50-replay-adaptation-online-stage-v1",
        )
        rows.extend(
            {
                **dict(row["selection"]),
                "sampler_label": "Fixed replay" if sampler == "static" else "Rotating replay",
            }
            for row in ledger.rows
        )
    return tuple(rows)


def _resource_rows(run: Path, result: Mapping[str, object]) -> tuple[dict[str, object], ...]:
    rows = []
    for condition in ONLINE_IDS:
        ledger = ChainedJsonlLedger(
            run / "integrators" / "online" / condition / "training_metrics.jsonl",
            "imagenetr50-replay-adaptation-online-stage-v1",
        )
        rows.append(
            {
                "condition": condition,
                "condition_label": CONDITION_LABELS[condition],
                "image_presentations": sum(
                    int(row["fit"]["image_presentations"]) for row in ledger.rows
                ),
                "logical_node_example_bound": sum(
                    int(row["node_example_forwards_bound"]) for row in ledger.rows
                ),
                "optimizer_steps": sum(
                    int(row["fit"]["optimizer_steps"]) for row in ledger.rows
                ),
                "parameter_count": int(ledger.rows[-1]["parameter_count"]),
                "training_wall_seconds": math.fsum(
                    float(row["fit"]["wall_seconds"]) for row in ledger.rows
                ),
            }
        )
    selections = {
        int(row["stage"]): dict(row["full_history_selection"])
        for row in result["development_metrics"]
        if row["condition"] == FULL_HISTORY_CONDITION
    }
    rows.append(
        {
            "condition": FULL_HISTORY_CONDITION,
            "condition_label": CONDITION_LABELS[FULL_HISTORY_CONDITION],
            "image_presentations": sum(
                int(candidate["fit"]["image_presentations"])
                for selection in selections.values()
                for candidate in selection["candidates"]
            ),
            "logical_node_example_bound": None,
            "optimizer_steps": sum(
                int(candidate["fit"]["optimizer_steps"])
                for selection in selections.values()
                for candidate in selection["candidates"]
            ),
            "parameter_count": rows[0]["parameter_count"],
            "training_wall_seconds": math.fsum(
                float(candidate["fit"]["wall_seconds"])
                for selection in selections.values()
                for candidate in selection["candidates"]
            ),
        }
    )
    return tuple(rows)


def _task_rows(result: Mapping[str, object]) -> tuple[dict[str, object], ...]:
    rows = []
    for development in result["development_metrics"]:
        for partition in ("fit_population", "validation"):
            rows.extend(
                {
                    "accuracy": float(value),
                    "condition": str(development["condition"]),
                    "partition": partition,
                    "stage": int(development["stage"]),
                    "task": int(task),
                }
                for task, value in dict(development[partition]["task_accuracies"]).items()
            )
    for locked in result["test_metrics"]:
        rows.extend(
            {
                "accuracy": float(value),
                "condition": str(locked["condition"]),
                "partition": "test",
                "stage": int(locked["stage"]),
                "task": int(task),
            }
            for task, value in dict(locked["metrics"]["task_accuracies"]).items()
        )
    return tuple(rows)


def _best_online(rows: Sequence[Mapping[str, object]]) -> str:
    means = {
        condition: math.fsum(
            float(row["validation_accuracy"])
            for row in rows
            if row["condition"] == condition
        )
        / len(STAGES)
        for condition in ONLINE_IDS
    }
    return max(means, key=means.__getitem__)


def _accuracy_plot(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    markers = ("o", "s", "^", "D", "P", "X", "v", "<")
    colors = ("#6c757d", "#495057", "#2a9d8f", "#1f776d", "#7b2cbf", "#5a189a", "#e76f51", "#c44536")
    figure, axes = plt.subplots(1, 2, figsize=(14.5, 6.4), sharey=True)
    for axis, stage in zip(axes, STAGES, strict=True):
        stage_rows = {str(row["condition"]): row for row in rows if int(row["stage"]) == stage}
        for index, condition in enumerate(ONLINE_IDS):
            row = stage_rows[condition]
            axis.scatter(
                float(row["validation_accuracy"]),
                float(row["test_accuracy"]),
                color=colors[index],
                marker=markers[index],
                s=85,
                label=CONDITION_LABELS[condition],
            )
        full = stage_rows[FULL_HISTORY_CONDITION]
        axis.scatter(
            float(full["validation_accuracy"]),
            float(full["test_accuracy"]),
            color="#0077b6",
            marker="*",
            edgecolor="black",
            linewidth=0.6,
            s=190,
            label=CONDITION_LABELS[FULL_HISTORY_CONDITION],
        )
        axis.axhline(
            float(full["raw_union_test_accuracy"]),
            color="#555555",
            linestyle=":",
            linewidth=1.5,
            label="Raw-union test reference",
        )
        axis.axhline(
            float(full["true_node_oracle_test_accuracy"]),
            color="#d00000",
            linestyle="--",
            linewidth=1.7,
            label="True-node oracle",
        )
        axis.axhline(
            float(full["joint_iid_test_accuracy"]),
            color="#111111",
            linestyle="-.",
            linewidth=1.7,
            label="Stage-matched joint IID",
        )
        axis.set_title(f"Stage {stage}: {stage.bit_count()} live nodes")
        axis.set_xlabel("Integrator validation accuracy (%)")
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("Locked test accuracy (%)")
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, fontsize=8.3, frameon=False)
    figure.suptitle("Replay adaptation: validation-selected evidence and locked test accuracy", y=0.98)
    figure.tight_layout(rect=(0, 0.16, 1, 0.95))
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def _generalization_plot(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure, axes = plt.subplots(1, 2, figsize=(15.2, 7.2), sharex=True)
    ordered = (*ONLINE_IDS, FULL_HISTORY_CONDITION)
    y = np.arange(len(ordered))
    for axis, stage in zip(axes, STAGES, strict=True):
        lookup = {str(row["condition"]): row for row in rows if int(row["stage"]) == stage}
        for metric, marker, color, label in (
            ("selected_training_accuracy", "o", "#6a4c93", "Selected replay rows"),
            ("fit_population_accuracy", "s", "#1982c4", "Complete fit population"),
            ("validation_accuracy", "^", "#2a9d8f", "Held-out integrator validation"),
            ("test_accuracy", "D", "#e76f51", "Locked test"),
        ):
            axis.scatter(
                [float(lookup[condition][metric]) for condition in ordered],
                y,
                marker=marker,
                color=color,
                s=58,
                label=label,
            )
        axis.set_title(f"Stage {stage}")
        axis.set_xlabel("Accuracy (%)")
        axis.grid(axis="x", alpha=0.22)
    axes[0].set_yticks(y, [CONDITION_LABELS[value] for value in ordered], fontsize=8.2)
    axes[1].tick_params(labelleft=False)
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    figure.suptitle("Memorization check: selected rows versus unseen populations", y=0.98)
    figure.tight_layout(rect=(0, 0.09, 1, 0.95))
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def _factor_plot(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    factors = (
        "rotating_minus_fixed",
        "task_balanced_minus_example",
        "reset_minus_carried_adam",
    )
    labels = ("Rotating - fixed", "Task-balanced - example", "Reset - carried Adam")
    figure, axes = plt.subplots(1, 2, figsize=(13.8, 5.8), sharey=True)
    for axis, metric in zip(axes, ("validation_accuracy", "test_accuracy"), strict=True):
        x = np.arange(len(factors))
        width = 0.34
        for offset, stage, color in ((-width / 2, 31, "#457b9d"), (width / 2, 50, "#e76f51")):
            values = [
                math.fsum(
                    float(row["delta_pp"])
                    for row in rows
                    if row["factor"] == factor
                    and row["metric"] == metric
                    and int(row["stage"]) == stage
                )
                / 4
                for factor in factors
            ]
            axis.bar(x + offset, values, width, color=color, label=f"Stage {stage}")
        axis.axhline(0.0, color="black", linewidth=1)
        axis.set_xticks(x, labels, rotation=16, ha="right")
        axis.set_title(metric.replace("_", " ").title())
        axis.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("Mean paired accuracy change (percentage points)")
    axes[1].legend(frameon=False)
    figure.suptitle("Main effects averaged over the other two paired factors", y=0.98)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def _turnover_plot(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(13.8, 5.5))
    for sampler, color in (("static", "#6c757d"), ("rotating", "#7b2cbf")):
        selected = sorted(
            (row for row in rows if row["sampler"] == sampler),
            key=lambda row: int(row["stage"]),
        )
        stages = [int(row["stage"]) for row in selected]
        axes[0].plot(
            stages,
            [int(row["historical_novel_since_previous"]) for row in selected],
            color=color,
            linewidth=2.2,
            label=str(selected[0]["sampler_label"]),
        )
        axes[1].plot(
            stages,
            [
                math.nan
                if row["historical_overlap_fraction"] is None
                else 100.0 * float(row["historical_overlap_fraction"])
                for row in selected
            ],
            color=color,
            linewidth=2.2,
            label=str(selected[0]["sampler_label"]),
        )
    axes[0].set_ylabel("New historical identities versus prior stage")
    axes[1].set_ylabel("Historical-set overlap with prior stage (%)")
    for axis in axes:
        axis.set_xlabel("Learned tasks")
        axis.grid(alpha=0.22)
        axis.legend(frameon=False)
    figure.suptitle("Stage-keyed replay changes which old examples train the integrator", y=0.98)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def _fmt(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(_fmt(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _image_uri(path: Path) -> str:
    return "data:image/png;base64," + b64encode(path.read_bytes()).decode("ascii")


def _html_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    head = "".join(f"<th>{escape(value)}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(_fmt(value))}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def _report_content(
    rows: Sequence[Mapping[str, object]],
    factors: Sequence[Mapping[str, object]],
    resources: Sequence[Mapping[str, object]],
) -> tuple[str, dict[str, object]]:
    best = _best_online(rows)
    best_rows = {int(row["stage"]): row for row in rows if row["condition"] == best}
    baseline_rows = {
        int(row["stage"]): row
        for row in rows
        if row["condition"] == "static__example_uniform__carry"
    }
    full_rows = {
        int(row["stage"]): row
        for row in rows
        if row["condition"] == FULL_HISTORY_CONDITION
    }
    summaries = _factor_summary(factors)
    summary_lookup = {
        (str(row["factor"]), str(row["metric"])): float(row["mean_delta_pp"])
        for row in summaries
    }
    headline_rows = tuple(
        (
            stage,
            baseline_rows[stage]["test_accuracy"],
            best_rows[stage]["test_accuracy"],
            full_rows[stage]["test_accuracy"],
            full_rows[stage]["raw_union_test_accuracy"],
            full_rows[stage]["true_node_oracle_test_accuracy"],
            full_rows[stage]["joint_iid_test_accuracy"],
        )
        for stage in STAGES
    )
    condition_rows = tuple(
        (
            int(row["stage"]),
            str(row["condition_label"]),
            float(row["selected_training_accuracy"]),
            float(row["fit_population_accuracy"]),
            float(row["validation_accuracy"]),
            float(row["test_accuracy"]),
            row["old_task_macro_test_accuracy"],
            float(row["current_task_test_accuracy"]),
        )
        for row in rows
    )
    context = {
        "baseline_rows": baseline_rows,
        "best": best,
        "best_rows": best_rows,
        "condition_rows": condition_rows,
        "full_rows": full_rows,
        "headline_rows": headline_rows,
        "summary_lookup": summary_lookup,
        "summaries": summaries,
    }
    markdown = f"""# ImageNet-R-50 replay-adaptation diagnosis

## Finding

The Permuted-MNIST integrator drew a fresh deterministic historical subset at every arrival; the earlier ImageNet-R integrator did not. This run restores that stage-keyed sampling and separates it from task weighting and AdamW-state carry. The validation-selected online condition is **{CONDITION_LABELS[best]}**.

{_markdown_table(("Stage", "Fixed baseline", "Best online", "Fresh full history", "Raw union", "True-node oracle", "Joint IID"), headline_rows)}

Across stages 31 and 50, rotation changes locked-test accuracy by {summary_lookup[("rotating_minus_fixed", "test_accuracy")]:+.3f} points on average over the four matched weighting/optimizer cells. Task-balanced loss changes it by {summary_lookup[("task_balanced_minus_example", "test_accuracy")]:+.3f} points, and resetting Adam moments changes it by {summary_lookup[("reset_minus_carried_adam", "test_accuracy")]:+.3f} points. These are paired diagnostic effects from one seed, not uncertainty estimates.

![Validation and locked-test comparison](accuracy_comparison.png)

## What differed from Permuted-MNIST

Permuted-MNIST sampled uniformly from the complete historical archive with a seed containing the macro-step, so each arrival received a new reproducible draw. It trained on 256 current and 256 historical examples. ImageNet-R used a permanent-priority namespace independent of stage. The fixed subset was reproducible, but stage-keyed random replay would have been equally reproducible and retained the same fixed-H O(T log T) work bound.

![Replay-set turnover](replay_turnover.png)

## Memorization and old-task adaptation

{_markdown_table(("Stage", "Condition", "Selected train", "Full fit", "Validation", "Test", "Old-task test", "Current-task test"), condition_rows)}

The selected-row versus complete-fit and held-out gaps distinguish adaptation to retained identities from adaptation to the old-task distribution. The 4,800-image validation partition is held out from every integrator optimizer, although the already-frozen LoRA nodes were trained earlier on the complete 24,000-image train split.

![Selected rows and unseen populations](generalization.png)

## Paired factor effects

{_markdown_table(("Factor", "Metric", "Mean paired change (pp)"), tuple((row["factor"], row["metric"], row["mean_delta_pp"]) for row in summaries))}

![Paired factor effects](factor_effects.png)

## Interpretation boundaries

The full-history condition is a fresh, three-restart, validation-selected diagnostic ceiling at stages 31 and 50. It does not satisfy the online effort constraint. This experiment uses one seed and a test split already examined by prior work, so locked-test results are descriptive. Condition selection in this report uses only the mean validation accuracy at stages 31 and 50.

## Reproducibility and work

Every online cell uses H=8,192, four epochs per arrival, the same initial parameters and minibatch seeds, and the same node-specific LoRA-adapted latent features. Test behavior was unavailable until all online and full-history fits were sealed. The exact-resume pass performed zero new optimizer steps and left all source and target model artifacts unchanged.

{_markdown_table(("Condition", "Image presentations", "Optimizer steps", "Logical node/example bound", "Fit seconds"), tuple((row["condition_label"], row["image_presentations"], row["optimizer_steps"], row["logical_node_example_bound"], row["training_wall_seconds"]) for row in resources))}

![Binary-counter lineage](lineage.png)

Machine-readable condition, factor, replay-selection, per-task, and resource tables accompany this report in CSV, JSON, and Parquet formats. Protocol and cache-seal records bind the report to the immutable source hierarchy and source replay run.
"""
    return markdown, context


def _report_html(
    markdown: str,
    context: Mapping[str, object],
    report_root: Path,
    resources: Sequence[Mapping[str, object]],
) -> str:
    headline = context["headline_rows"]
    conditions = context["condition_rows"]
    summaries = context["summaries"]
    best = str(context["best"])
    summary_lookup = context["summary_lookup"]
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>ImageNet-R replay adaptation</title><style>
@page {{ size: A4; margin: 15mm 14mm; }}
:root {{ --ink:#18212b; --muted:#52616f; --blue:#2457a6; --paper:#ffffff; --soft:#eef4f8; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:#e8edf1; color:var(--ink); font:15px/1.48 system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ max-width:1120px; margin:24px auto; background:var(--paper); padding:42px 52px 58px; box-shadow:0 3px 22px #0002; }}
h1 {{ font-size:31px; line-height:1.16; margin:0 0 8px; color:#16365f; }} h2 {{ margin:30px 0 11px; font-size:21px; color:#1f4e79; border-bottom:2px solid #dbe7f0; padding-bottom:5px; }}
p {{ margin:9px 0; }} .lede {{ font-size:17px; color:#263f57; }} .callout {{ background:var(--soft); border-left:5px solid var(--blue); padding:14px 17px; margin:16px 0; }}
details {{ margin:17px 0; border:1px solid #d9e2e9; border-radius:6px; padding:10px 14px; }} summary {{ cursor:pointer; font-weight:700; color:#234f78; }}
.table-wrap {{ overflow-x:auto; margin:12px 0 18px; }} table {{ width:100%; border-collapse:collapse; font-size:11.4px; }} th {{ background:#244f78; color:white; text-align:left; }} th,td {{ padding:6px 7px; border:1px solid #ccd7df; vertical-align:top; }} tbody tr:nth-child(even) {{ background:#f6f9fb; }}
figure {{ margin:18px 0 25px; break-inside:avoid; }} img {{ width:100%; height:auto; }} figcaption {{ color:var(--muted); font-size:12px; margin-top:4px; }} code {{ font-size:0.91em; }} .small {{ color:var(--muted); font-size:12px; }}
@media print {{ body {{ background:white; }} main {{ margin:0; max-width:none; padding:0; box-shadow:none; }} details {{ break-inside:auto; }} summary {{ list-style:none; }} }}
</style></head><body><main>
<h1>ImageNet-R-50 replay-adaptation diagnosis</h1>
<p class="lede">A paired test of stage-random replay, equal task weighting, optimizer-state resets, and fresh full-history integration at fragmented frontiers.</p>
<div class="callout"><strong>Finding.</strong> Permuted-MNIST used a fresh deterministic replay draw at every macro-step. The validation-selected online ImageNet-R condition is <strong>{escape(CONDITION_LABELS[best])}</strong>. Averaged across matched cells and stages 31/50, rotation changed locked-test accuracy by {float(summary_lookup[("rotating_minus_fixed", "test_accuracy")]):+.3f} percentage points.</div>
<h2>Headline comparison</h2>
{_html_table(("Stage", "Fixed baseline", "Best online", "Fresh full history", "Raw union", "True-node oracle", "Joint IID"), headline)}
<figure><img src="{_image_uri(report_root / 'accuracy_comparison.png')}" alt="Validation versus locked-test accuracy"><figcaption>All online conditions and the validation-selected full-history diagnostic. Horizontal references use the identical stage and test examples.</figcaption></figure>
<details open><summary>What differed from Permuted-MNIST</summary><p>Permuted-MNIST sampled uniformly from its complete historical archive with a seed containing the macro-step. It used 256 current and 256 historical examples. ImageNet-R instead used a permanent-priority namespace independent of stage. Stage-keyed random replay remains deterministic and resumable while preserving fixed-H cumulative O(T log T) work.</p><figure><img src="{_image_uri(report_root / 'replay_turnover.png')}" alt="Replay identity turnover"><figcaption>Novel identities and overlap relative to the immediately preceding historical subset.</figcaption></figure></details>
<h2>Memorization and old-task adaptation</h2>
{_html_table(("Stage", "Condition", "Selected train", "Full fit", "Validation", "Test", "Old-task test", "Current-task test"), conditions)}
<figure><img src="{_image_uri(report_root / 'generalization.png')}" alt="Training and held-out accuracies"><figcaption>Selected replay rows can be memorized even when complete-fit, validation, and test accuracy remain much lower.</figcaption></figure>
<details open><summary>Paired factor effects</summary>{_html_table(("Factor", "Metric", "Mean paired change (pp)"), tuple((row["factor"], row["metric"], row["mean_delta_pp"]) for row in summaries))}<figure><img src="{_image_uri(report_root / 'factor_effects.png')}" alt="Factor effect bars"><figcaption>Each bar averages four exact paired contrasts over the other two factors.</figcaption></figure></details>
<details open><summary>Interpretation boundaries</summary><p>The fresh full-history fits use every fit row, three restarts, and validation-loss selection. They are diagnostic ceilings, not online LogT conditions. The integrator validation identities never enter integrator fitting, although the frozen LoRA nodes were previously fitted on all 24,000 training images. This is one seed and a previously examined test split, so test results are descriptive.</p></details>
<h2>Reproducibility and work</h2>
{_html_table(("Condition", "Image presentations", "Optimizer steps", "Logical node/example bound", "Fit seconds"), tuple((row["condition_label"], row["image_presentations"], row["optimizer_steps"], row["logical_node_example_bound"], row["training_wall_seconds"]) for row in resources))}
<figure><img src="{_image_uri(report_root / 'lineage.png')}" alt="Binary-counter hierarchy lineage"><figcaption>The reused 50-task fresh-parent hierarchy ends with three live nodes; stages 31 and 50 expose five and three live nodes respectively.</figcaption></figure>
<p class="small">Report source SHA-256: {record_sha256(markdown)}</p>
</main></body></html>"""


def write_replay_adaptation_report(run: str | Path) -> tuple[Path, Path, Path]:
    """Generate the complete report and compact machine-readable evidence."""
    run_root = Path(run)
    result = _validated_result(run_root)
    rows = _condition_rows(result)
    factors = _factor_rows(rows)
    factor_summary = _factor_summary(factors)
    selections = _selection_rows(run_root)
    resources = _resource_rows(run_root, result)
    tasks = _task_rows(result)
    report_root = run_root / "reports"
    report_root.mkdir(parents=True, exist_ok=True)
    for name, values in (
        ("condition_metrics", rows),
        ("factor_contrasts", factors),
        ("factor_summary", factor_summary),
        ("replay_selections", selections),
        ("resource_accounting", resources),
        ("task_accuracy_matrix", tasks),
    ):
        _write_table_family(report_root, name, values)
    _accuracy_plot(report_root / "accuracy_comparison.png", rows)
    _generalization_plot(report_root / "generalization.png", rows)
    _factor_plot(report_root / "factor_effects.png", factors)
    _turnover_plot(report_root / "replay_turnover.png", selections)
    _lineage_plot(report_root / "lineage.png")
    markdown, context = _report_content(rows, factors, resources)
    markdown_path = atomic_write(
        report_root / "REPORT.md", markdown.encode("utf-8")
    )
    atomic_write(
        report_root / "HANDOFF.md",
        (
            "# Technical-analysis handoff\n\n"
            + markdown.split("## Reproducibility and work", maxsplit=1)[0]
            + "Use the accompanying machine-readable tables for independent analysis.\n"
        ).encode("utf-8"),
    )
    invocation = load_canonical_json(run_root / "state" / "last_invocation.json")
    proof = load_canonical_json(run_root / "protocol" / "reuse_proof.json")
    atomic_write(
        report_root / "RUN.log",
        (
            f"run_hash={result['protocol_hash']}\n"
            f"elapsed_seconds={float(invocation['elapsed_seconds']):.3f}\n"
            "phase=COMPLETE\n"
            "diagnostic_stages=31,50\n"
            "online_conditions=8\n"
        ).encode("utf-8"),
    )
    atomic_write(
        report_root / "RESUME.log",
        (
            f"integrity_passed={bool(proof['integrity_passed'])}\n"
            "new_leaf_optimizer_steps=0\n"
            "new_parent_optimizer_steps=0\n"
            "new_online_optimizer_steps=0\n"
            "new_full_history_fits=0\n"
            "new_evaluations=0\n"
        ).encode("utf-8"),
    )
    html_path = atomic_write(
        report_root / "REPORT.html",
        _report_html(markdown, context, report_root, resources).encode("utf-8"),
    )
    pdf_path = render_integrator_pdf(html_path)
    return markdown_path, html_path, pdf_path


__all__ = ["write_replay_adaptation_report"]

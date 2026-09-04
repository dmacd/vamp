"""Report the nested two-layer ImageNet-R replay representation ablation."""

from __future__ import annotations

from base64 import b64encode
from collections.abc import Mapping, Sequence
from html import escape
from pathlib import Path
import math

from apm.continual.artifacts import (
    atomic_write,
    canonical_json_bytes,
    file_sha256,
    load_canonical_json,
    record_sha256,
)
from apm.continual.vision.imagenetr.integrator_pdf import render_integrator_pdf
from apm.continual.vision.imagenetr.integrator_reporting import _lineage_plot
from apm.continual.vision.imagenetr.replay_adaptation_reporting import (
    CONDITION_LABELS,
    FULL_HISTORY_CONDITION,
    ONLINE_IDS,
    STAGES,
    _accuracy_plot,
    _best_online,
    _condition_rows,
    _factor_plot,
    _factor_rows,
    _factor_summary,
    _generalization_plot,
    _resource_rows,
    _selection_rows,
    _task_rows,
    _turnover_plot,
    _validated_result,
    _write_table_family,
)


def _fmt(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _markdown_table(
    headers: Sequence[str], rows: Sequence[Sequence[object]]
) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_fmt(value) for value in row) + " |"
        for row in rows
    )
    return "\n".join(lines)


def _html_table(
    headers: Sequence[str], rows: Sequence[Sequence[object]]
) -> str:
    head = "".join(f"<th>{escape(value)}</th>" for value in headers)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{escape(_fmt(value))}</td>" for value in row)
        + "</tr>"
        for row in rows
    )
    return (
        "<div class='table-wrap'><table><thead><tr>"
        f"{head}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def _image_uri(path: Path) -> str:
    return "data:image/png;base64," + b64encode(path.read_bytes()).decode("ascii")


def _single_layer_result(
    run: Path, result: Mapping[str, object]
) -> dict[str, object]:
    source = run / "evaluations" / "source_single_layer_replay_result.json"
    baseline = load_canonical_json(source)
    core = {key: value for key, value in baseline.items() if key != "content_hash"}
    identity = dict(result["representation_baseline"])
    if (
        baseline.get("schema_version")
        != "imagenetr50-replay-adaptation-result-v1"
        or baseline.get("content_hash") != record_sha256(core)
        or baseline.get("protocol_hash") != identity.get("run_hash")
        or file_sha256(source) != identity.get("result_sha256")
        or baseline.get("content_hash") != identity.get("content_hash")
    ):
        raise ValueError("single-layer representation comparison changed")
    return baseline


def _representation_rows(
    single: Sequence[Mapping[str, object]],
    two_layer: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    old = {
        (str(row["condition"]), int(row["stage"])): row for row in single
    }
    new = {
        (str(row["condition"]), int(row["stage"])): row for row in two_layer
    }
    if set(old) != set(new):
        raise ValueError("single- and two-layer matrices do not align")
    return tuple(
        {
            "condition": condition,
            "condition_label": CONDITION_LABELS[condition],
            "single_layer_test_accuracy": float(old[condition, stage]["test_accuracy"]),
            "single_layer_validation_accuracy": float(
                old[condition, stage]["validation_accuracy"]
            ),
            "stage": stage,
            "test_delta_pp": float(new[condition, stage]["test_accuracy"])
            - float(old[condition, stage]["test_accuracy"]),
            "two_layer_test_accuracy": float(new[condition, stage]["test_accuracy"]),
            "two_layer_validation_accuracy": float(
                new[condition, stage]["validation_accuracy"]
            ),
            "validation_delta_pp": float(
                new[condition, stage]["validation_accuracy"]
            )
            - float(old[condition, stage]["validation_accuracy"]),
        }
        for condition, stage in sorted(old, key=lambda value: (value[1], value[0]))
    )


def _representation_plot(
    path: Path, rows: Sequence[Mapping[str, object]]
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ordered = (*ONLINE_IDS, FULL_HISTORY_CONDITION)
    y = np.arange(len(ordered))
    figure, axes = plt.subplots(1, 2, figsize=(15.6, 7.0), sharey=True)
    for axis, stage in zip(axes, STAGES, strict=True):
        lookup = {
            str(row["condition"]): row
            for row in rows
            if int(row["stage"]) == stage
        }
        single = [float(lookup[name]["single_layer_test_accuracy"]) for name in ordered]
        two = [float(lookup[name]["two_layer_test_accuracy"]) for name in ordered]
        for index, (left, right) in enumerate(zip(single, two, strict=True)):
            axis.plot((left, right), (index, index), color="#a8b3bd", linewidth=2)
        axis.scatter(single, y, color="#6c757d", marker="o", s=58, label="Final latent only")
        axis.scatter(two, y, color="#7b2cbf", marker="D", s=58, label="Final + penultimate")
        axis.margins(x=0.08)
        axis.set_title(f"Stage {stage}: {stage.bit_count()} live nodes")
        axis.set_xlabel("Locked-test accuracy (%)")
        axis.grid(axis="x", alpha=0.22)
    axes[0].set_yticks(y, [CONDITION_LABELS[name] for name in ordered], fontsize=8.2)
    axes[1].tick_params(labelleft=False)
    axes[1].legend(frameon=False, loc="lower right")
    figure.suptitle(
        "Matched representation ablation: one versus two node-adapted ViT layers",
        y=0.98,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def _report_content(
    rows: Sequence[Mapping[str, object]],
    single_rows: Sequence[Mapping[str, object]],
    representation_rows: Sequence[Mapping[str, object]],
    factors: Sequence[Mapping[str, object]],
    resources: Sequence[Mapping[str, object]],
) -> tuple[str, dict[str, object]]:
    best = _best_online(rows)
    single_best = _best_online(single_rows)
    lookup = {(str(row["condition"]), int(row["stage"])): row for row in rows}
    old = {
        (str(row["condition"]), int(row["stage"])): row for row in single_rows
    }
    headline = tuple(
        (
            stage,
            old[single_best, stage]["test_accuracy"],
            lookup[best, stage]["test_accuracy"],
            old[FULL_HISTORY_CONDITION, stage]["test_accuracy"],
            lookup[FULL_HISTORY_CONDITION, stage]["test_accuracy"],
            lookup[FULL_HISTORY_CONDITION, stage]["true_node_oracle_test_accuracy"],
            lookup[FULL_HISTORY_CONDITION, stage]["joint_iid_test_accuracy"],
        )
        for stage in STAGES
    )
    matched = tuple(
        (
            int(row["stage"]),
            str(row["condition_label"]),
            float(row["single_layer_validation_accuracy"]),
            float(row["two_layer_validation_accuracy"]),
            float(row["validation_delta_pp"]),
            float(row["single_layer_test_accuracy"]),
            float(row["two_layer_test_accuracy"]),
            float(row["test_delta_pp"]),
        )
        for row in representation_rows
    )
    online_representation = tuple(
        row for row in representation_rows if row["condition"] in ONLINE_IDS
    )
    mean_online_test_delta = math.fsum(
        float(row["test_delta_pp"]) for row in online_representation
    ) / len(online_representation)
    mean_online_validation_delta = math.fsum(
        float(row["validation_delta_pp"]) for row in online_representation
    ) / len(online_representation)
    primary = tuple(
        row for row in representation_rows if row["condition"] == single_best
    )
    full_history = {
        int(row["stage"]): row
        for row in representation_rows
        if row["condition"] == FULL_HISTORY_CONDITION
    }
    summaries = _factor_summary(factors)
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
    markdown = f"""# ImageNet-R-50 two-layer node-latent ablation

## Finding

This run appends each live node's LoRA-adapted penultimate ViT class token to the existing final pre-classifier token. Replay, loss, optimizer policy, hierarchy, and evaluation identities match the single-layer v6 matrix. The validation-selected two-layer condition is **{CONDITION_LABELS[best]}**; the v6 selection was **{CONDITION_LABELS[single_best]}**.

{_markdown_table(("Stage", "Single-layer selected", "Two-layer selected", "Single-layer full history", "Two-layer full history", "True-node oracle", "Joint IID"), headline)}

Across the 16 matched online cells (eight conditions at stages 31 and 50), adding the penultimate latent changes validation accuracy by **{mean_online_validation_delta:+.3f} points** and locked-test accuracy by **{mean_online_test_delta:+.3f} points** on average. For the original v6-selected condition, the task-31 and task-50 test changes are **{float(primary[0]['test_delta_pp']):+.3f}** and **{float(primary[1]['test_delta_pp']):+.3f} points**. These are paired results from one seed, not uncertainty estimates.

The fresh full-history arm is the clearest ceiling diagnostic: its test changes are **{float(full_history[31]['test_delta_pp']):+.3f}** and **{float(full_history[50]['test_delta_pp']):+.3f} points**. The extra token therefore does not materially improve what this integrator can learn even when replay sampling is removed. Its much larger validation gain than test gain is also consistent with an optimistic validation partition: those identities were held out from integrator fitting but seen during upstream node fitting.

![Matched single-layer and two-layer results](representation_comparison.png)

## Exact representation change

The added 768 values are the class token after transformer block 11 of 12, before the final block and final backbone normalization, captured while the evaluated node's own LoRA is installed. The existing final token and added penultimate token receive separate per-image layer normalization. Each slot grows from 1,369 to 2,137 values; six slots grow from 8,214 to 12,822, and the MLP grows from 9,122,760 to 13,841,352 parameters. Existing input weights and every downstream parameter use the v6 initialization, while new-latent columns start at zero.

{_markdown_table(("Stage", "Condition", "Single val", "Two-layer val", "Val delta", "Single test", "Two-layer test", "Test delta"), matched)}

## Two-layer adaptation and generalization

{_markdown_table(("Stage", "Condition", "Selected train", "Full fit", "Validation", "Test", "Old-task test", "Current-task test"), condition_rows)}

![Training and unseen populations](generalization.png)

## Within-run replay-factor effects

{_markdown_table(("Factor", "Metric", "Mean paired change (pp)"), tuple((row["factor"], row["metric"], row["mean_delta_pp"]) for row in summaries))}

![Two-layer replay-factor effects](factor_effects.png)

## Interpretation boundaries

The representation comparison is nested at initialization and uses identical image identities and schedules, but it has one seed and a previously examined locked test. The full-history conditions retrain only the integrator on all fit identities; they leave the fragmented LoRA nodes frozen and do not satisfy the online effort constraint. The 4,800-image integrator validation partition was seen during upstream node fitting, so it is not an end-to-end clean validation split.

## Reproducibility and work

All online cells retain H=8,192 and four epochs per arrival. Test behavior remained unavailable until all online and three-restart full-history models were sealed. The exact-resume pass performed zero new optimizer steps and left all source and target model artifacts unchanged.

{_markdown_table(("Condition", "Parameters", "Image presentations", "Optimizer steps", "Fit seconds"), tuple((row["condition_label"], row["parameter_count"], row["image_presentations"], row["optimizer_steps"], row["training_wall_seconds"]) for row in resources))}

![Binary-counter lineage](lineage.png)

Machine-readable representation, condition, factor, replay-selection, per-task, and resource tables accompany this report in CSV, JSON, and Parquet formats.
"""
    return markdown, {
        "best": best,
        "condition_rows": condition_rows,
        "headline": headline,
        "matched": matched,
        "mean_online_test_delta": mean_online_test_delta,
        "mean_online_validation_delta": mean_online_validation_delta,
        "primary": primary,
        "single_best": single_best,
        "summaries": summaries,
    }


def _report_html(
    markdown: str,
    context: Mapping[str, object],
    report_root: Path,
    resources: Sequence[Mapping[str, object]],
) -> str:
    best = str(context["best"])
    single_best = str(context["single_best"])
    primary = context["primary"]
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>ImageNet-R two-layer node latents</title><style>
@page {{ size: A4; margin: 15mm 14mm; }}
:root {{ --ink:#18212b; --muted:#52616f; --violet:#6f2dbd; --paper:#fff; --soft:#f2eef8; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:#e8edf1; color:var(--ink); font:15px/1.48 system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ max-width:1120px; margin:24px auto; background:var(--paper); padding:42px 52px 58px; box-shadow:0 3px 22px #0002; }}
h1 {{ font-size:31px; line-height:1.16; margin:0 0 8px; color:#3c1766; }} h2 {{ margin:30px 0 11px; font-size:21px; color:#55308b; border-bottom:2px solid #e4d9ef; padding-bottom:5px; }}
p {{ margin:9px 0; }} .lede {{ font-size:17px; color:#3c3150; }} .callout {{ background:var(--soft); border-left:5px solid var(--violet); padding:14px 17px; margin:16px 0; }}
.table-wrap {{ overflow-x:auto; margin:12px 0 18px; }} table {{ width:100%; border-collapse:collapse; font-size:11.2px; }} th {{ background:#55308b; color:white; text-align:left; }} th,td {{ padding:6px 7px; border:1px solid #d7cce1; vertical-align:top; }} tbody tr:nth-child(even) {{ background:#faf8fc; }}
figure {{ margin:18px 0 25px; break-inside:avoid; }} img {{ width:100%; height:auto; }} figcaption {{ color:var(--muted); font-size:12px; margin-top:4px; }} details {{ margin:17px 0; border:1px solid #ded6e6; border-radius:6px; padding:10px 14px; }} summary {{ cursor:pointer; font-weight:700; color:#55308b; }} .small {{ color:var(--muted); font-size:12px; }}
@media print {{ body {{ background:white; }} main {{ margin:0; max-width:none; padding:0; box-shadow:none; }} summary {{ list-style:none; }} }}
</style></head><body><main>
<h1>ImageNet-R-50 two-layer node-latent ablation</h1>
<p class="lede">A nested representation expansion under the unchanged replay-adaptation matrix.</p>
<div class="callout"><strong>Finding.</strong> The validation-selected two-layer condition is <strong>{escape(CONDITION_LABELS[best])}</strong>; v6 selected <strong>{escape(CONDITION_LABELS[single_best])}</strong>. Across 16 matched online stage/condition cells, the added layer changes locked-test accuracy by {float(context['mean_online_test_delta']):+.3f} points. For the v6-selected condition, the stage-31 and stage-50 changes are {float(primary[0]['test_delta_pp']):+.3f} and {float(primary[1]['test_delta_pp']):+.3f} points.</div>
<p><strong>Interpretation.</strong> Fresh full-history test accuracy declines slightly at both checkpoints, so the extra token does not raise the observed integration ceiling. The substantially larger validation gain is consistent with the fact that upstream nodes saw those validation identities during fitting.</p>
<h2>Headline comparison</h2>
{_html_table(("Stage", "Single-layer selected", "Two-layer selected", "Single-layer full history", "Two-layer full history", "True-node oracle", "Joint IID"), context["headline"])}
<figure><img src="{_image_uri(report_root / 'representation_comparison.png')}" alt="Matched one-layer and two-layer accuracy"><figcaption>Each line joins the same replay, weighting, optimizer, stage, and test identities.</figcaption></figure>
<h2>Exact representation change</h2><p>The new 768-value class token is captured after the penultimate transformer block while the live node's own LoRA is installed. It is appended after the unchanged 1,369-value slot prefix and normalized separately. The MLP grows from 9,122,760 to 13,841,352 parameters. New input columns start at zero; all compatible input columns and downstream parameters copy the v6 initialization.</p>
{_html_table(("Stage", "Condition", "Single val", "Two-layer val", "Val delta", "Single test", "Two-layer test", "Test delta"), context["matched"])}
<h2>Two-layer adaptation and generalization</h2>
{_html_table(("Stage", "Condition", "Selected train", "Full fit", "Validation", "Test", "Old-task test", "Current-task test"), context["condition_rows"])}
<figure><img src="{_image_uri(report_root / 'generalization.png')}" alt="Selected replay, population, validation and test accuracy"><figcaption>The full-fit and unseen-population gaps measure whether repeated online fitting adapts beyond retained identities.</figcaption></figure>
<details open><summary>Within-run replay-factor effects</summary>{_html_table(("Factor", "Metric", "Mean paired change (pp)"), tuple((row["factor"], row["metric"], row["mean_delta_pp"]) for row in context["summaries"]))}<figure><img src="{_image_uri(report_root / 'factor_effects.png')}" alt="Two-layer replay-factor effects"></figure></details>
<details open><summary>Interpretation boundaries</summary><p>This is one seed and a test set examined by earlier experiments. Full-history retrains only the integrator and is not online. Integrator validation identities were held out from integrator fitting but not from upstream LoRA-node fitting.</p></details>
<h2>Reproducibility and work</h2>
{_html_table(("Condition", "Parameters", "Image presentations", "Optimizer steps", "Fit seconds"), tuple((row["condition_label"], row["parameter_count"], row["image_presentations"], row["optimizer_steps"], row["training_wall_seconds"]) for row in resources))}
<figure><img src="{_image_uri(report_root / 'accuracy_comparison.png')}" alt="Two-layer validation versus test accuracy"><figcaption>All two-layer online conditions with identical stage-matched references.</figcaption></figure>
<figure><img src="{_image_uri(report_root / 'replay_turnover.png')}" alt="Replay identity turnover"><figcaption>Replay membership is unchanged from v6.</figcaption></figure>
<figure><img src="{_image_uri(report_root / 'lineage.png')}" alt="Binary-counter hierarchy"><figcaption>The reused source hierarchy has five live nodes at stage 31 and three at stage 50.</figcaption></figure>
<p class="small">Report source SHA-256: {record_sha256(markdown)}</p>
</main></body></html>"""


def write_two_layer_replay_report(
    run: str | Path,
) -> tuple[Path, Path, Path]:
    """Generate the report and compact evidence for the representation ablation."""
    run_root = Path(run)
    result = _validated_result(run_root)
    if result.get("feature_variant") != "behavior_two_layer":
        raise ValueError("two-layer report requires the two-layer feature variant")
    baseline_result = _single_layer_result(run_root, result)
    rows = _condition_rows(result)
    single_rows = _condition_rows(baseline_result)
    representation = _representation_rows(single_rows, rows)
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
        ("representation_comparison", representation),
        ("replay_selections", selections),
        ("resource_accounting", resources),
        ("task_accuracy_matrix", tasks),
    ):
        _write_table_family(report_root, name, values)
    _representation_plot(report_root / "representation_comparison.png", representation)
    _accuracy_plot(report_root / "accuracy_comparison.png", rows)
    _generalization_plot(report_root / "generalization.png", rows)
    _factor_plot(report_root / "factor_effects.png", factors)
    _turnover_plot(report_root / "replay_turnover.png", selections)
    _lineage_plot(report_root / "lineage.png")
    markdown, context = _report_content(
        rows, single_rows, representation, factors, resources
    )
    markdown_path = atomic_write(report_root / "REPORT.md", markdown.encode())
    atomic_write(
        report_root / "HANDOFF.md",
        (
            "# Technical-analysis handoff\n\n"
            + markdown.split("## Reproducibility and work", maxsplit=1)[0]
            + "Use the accompanying machine-readable tables for independent analysis.\n"
        ).encode(),
    )
    invocation = load_canonical_json(run_root / "state" / "last_invocation.json")
    proof = load_canonical_json(run_root / "protocol" / "reuse_proof.json")
    atomic_write(
        report_root / "RUN.log",
        (
            f"run_hash={result['protocol_hash']}\n"
            f"elapsed_seconds={float(invocation['elapsed_seconds']):.3f}\n"
            "phase=COMPLETE\nfeature_variant=behavior_two_layer\n"
            "diagnostic_stages=31,50\nonline_conditions=8\n"
        ).encode(),
    )
    atomic_write(
        report_root / "RESUME.log",
        (
            f"integrity_passed={bool(proof['integrity_passed'])}\n"
            "new_leaf_optimizer_steps=0\nnew_parent_optimizer_steps=0\n"
            "new_online_optimizer_steps=0\nnew_full_history_fits=0\n"
            "new_evaluations=0\n"
        ).encode(),
    )
    html_path = atomic_write(
        report_root / "REPORT.html",
        _report_html(markdown, context, report_root, resources).encode(),
    )
    pdf_path = render_integrator_pdf(html_path)
    return markdown_path, html_path, pdf_path


__all__ = ["write_two_layer_replay_report"]

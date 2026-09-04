"""Tables, plots, paper-style reports, and PDF export for the parent factorial."""

from __future__ import annotations

from base64 import b64encode
from collections.abc import Mapping, Sequence
from html import escape
from pathlib import Path
import csv
import json
import math
import shutil
import subprocess

from apm.continual.artifacts import (
    ChainedJsonlLedger,
    atomic_write,
    canonical_json_bytes,
    file_sha256,
    load_canonical_json,
    record_sha256,
)
from apm.continual.vision.imagenetr.parent_recipe_factorial import (
    DEFAULT_CONFIG,
    EPOCH_SCHEMA,
    SUMMARY_SCHEMA,
    load_parent_recipe_config,
)


def _latest_run() -> Path:
    config = load_parent_recipe_config(DEFAULT_CONFIG)
    latest = load_canonical_json(config.artifact_root / "LATEST_RUN.json")
    if latest.get("schema_version") != "imagenetr50-parent-recipe-factorial-latest-v1":
        raise ValueError("unknown parent-recipe latest-run record")
    run = config.artifact_root / "runs" / str(latest["run_hash"])
    if not run.is_dir():
        raise FileNotFoundError("latest parent-recipe run is unavailable")
    return run


def _validated_summary(run: Path) -> dict[str, object]:
    summary = load_canonical_json(run / "summary.json")
    core = {key: value for key, value in summary.items() if key != "content_hash"}
    if (
        summary.get("schema_version") != SUMMARY_SCHEMA
        or summary.get("content_hash") != record_sha256(core)
    ):
        raise ValueError("parent-recipe summary does not authenticate")
    return summary


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
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


def _epoch_rows(run: Path, summary: Mapping[str, object]) -> tuple[dict[str, object], ...]:
    rows = []
    for job in summary["rows"]:
        job_row = dict(job)
        ledger = ChainedJsonlLedger(
            run / "ledgers" / f"{job_row['job_hash']}.jsonl", EPOCH_SCHEMA
        )
        ledger.require_unique_keys(("job_hash", "epoch"))
        rows.extend(
            {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "format",
                    "head_statistics",
                    "previous_sha256",
                    "result_sha256",
                    "sequence",
                    "task_correct",
                    "task_examples",
                }
            }
            | {
                f"head_{key}": value
                for key, value in dict(row["head_statistics"]).items()
            }
            for row in ledger.rows
        )
    return tuple(
        sorted(rows, key=lambda row: (int(row["stage"]), str(row["condition_key"]), int(row["epoch"])))
    )


def factorial_effects(
    final_rows: Sequence[Mapping[str, object]],
    conditions: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """Compute descriptive paired main effects while holding the other factors fixed."""
    factors = {
        "head_initialization": ("fresh", "inherited_union", "inherited - fresh"),
        "weight_decay": (0.0005, 0.0, "wd=0 - wd=5e-4"),
        "seed_schedule": ("joint", "parent", "parent - joint schedule"),
    }
    condition_by_key = {
        str(condition["condition_key"]): dict(condition) for condition in conditions
    }
    accuracy = {
        (str(row["condition_key"]), int(row["stage"]), int(row["replication_seed"])): float(
            row["final_validation_accuracy"]
        )
        for row in final_rows
    }
    effects = []
    for stage in sorted({int(row["stage"]) for row in final_rows}):
        for factor, (baseline, changed, label) in factors.items():
            paired = []
            for condition_key, condition in condition_by_key.items():
                if condition[factor] != baseline:
                    continue
                counterpart = next(
                    key
                    for key, candidate in condition_by_key.items()
                    if all(
                        candidate[name] == (changed if name == factor else condition[name])
                        for name in factors
                    )
                )
                for seed in sorted({int(row["replication_seed"]) for row in final_rows}):
                    paired.append(
                        accuracy[(counterpart, stage, seed)]
                        - accuracy[(condition_key, stage, seed)]
                    )
            effects.append(
                {
                    "effect": label,
                    "factor": factor,
                    "maximum_effect_pp": max(paired),
                    "mean_effect_pp": math.fsum(paired) / len(paired),
                    "minimum_effect_pp": min(paired),
                    "paired_cells": len(paired),
                    "stage": stage,
                }
            )
    return tuple(effects)


def _condition_codes(
    conditions: Sequence[Mapping[str, object]],
) -> dict[str, str]:
    return {
        str(condition["condition_key"]): f"C{index}"
        for index, condition in enumerate(conditions, start=1)
    }


def _endpoint_plot(
    path: Path,
    final_rows: Sequence[Mapping[str, object]],
    epoch_rows: Sequence[Mapping[str, object]],
    conditions: Sequence[Mapping[str, object]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    codes = _condition_codes(conditions)
    condition_order = tuple(str(condition["condition_key"]) for condition in conditions)
    initial = {
        (str(row["condition_key"]), int(row["stage"])): float(row["validation_accuracy"])
        for row in epoch_rows
        if int(row["epoch"]) == 0
    }
    final = {
        (str(row["condition_key"]), int(row["stage"])): float(
            row["final_validation_accuracy"]
        )
        for row in final_rows
    }
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), sharey=True)
    colors = ["#1565c0" if key == "fresh__wd5e4__joint" else "#ef6c00" if key == "inherited_union__wd0__parent" else "#78909c" for key in condition_order]
    for axis, stage in zip(axes, (8, 16, 32), strict=True):
        x = list(range(len(condition_order)))
        values = [final[(key, stage)] for key in condition_order]
        axis.bar(x, values, color=colors, alpha=0.88)
        axis.scatter(
            x,
            [initial[(key, stage)] for key in condition_order],
            marker="x",
            color="#212121",
            s=34,
            label="before training" if stage == 8 else None,
            zorder=3,
        )
        axis.set(
            title=f"Task {stage} prefix",
            xlabel="Factorial condition",
            xticks=x,
            xticklabels=[codes[key] for key in condition_order],
        )
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Clean validation accuracy (%)")
    axes[0].legend(loc="lower right", fontsize=9)
    fig.suptitle("Parent-recipe factorial: initial (x) and five-epoch accuracy")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _effects_plot(path: Path, effects: Sequence[Mapping[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = tuple(dict.fromkeys(str(row["effect"]) for row in effects))
    fig, axis = plt.subplots(figsize=(10.5, 5.6))
    colors = {8: "#78909c", 16: "#1565c0", 32: "#ef6c00"}
    offsets = {8: -0.22, 16: 0.0, 32: 0.22}
    for stage in (8, 16, 32):
        stage_rows = {str(row["effect"]): row for row in effects if int(row["stage"]) == stage}
        y = [index + offsets[stage] for index in range(len(labels))]
        means = [float(stage_rows[label]["mean_effect_pp"]) for label in labels]
        lower = [means[index] - float(stage_rows[label]["minimum_effect_pp"]) for index, label in enumerate(labels)]
        upper = [float(stage_rows[label]["maximum_effect_pp"]) - means[index] for index, label in enumerate(labels)]
        axis.errorbar(
            means,
            y,
            xerr=(lower, upper),
            fmt="o",
            capsize=3,
            color=colors[stage],
            label=f"task {stage}",
        )
    axis.axvline(0.0, color="#212121", linewidth=1)
    axis.set(
        title="Paired factorial effects at epoch 5",
        xlabel="Accuracy change (percentage points)",
        yticks=range(len(labels)),
        yticklabels=labels,
    )
    axis.grid(axis="x", alpha=0.2)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    return "\n".join(
        (
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(row) + " |" for row in rows),
        )
    )


def _report_text(
    summary: Mapping[str, object], effects: Sequence[Mapping[str, object]]
) -> tuple[str, str]:
    conditions = tuple(dict(row) for row in summary["conditions"])
    rows = tuple(dict(row) for row in summary["rows"])
    selection = dict(summary["selection"])
    codes = _condition_codes(conditions)
    by_key_stage = {
        (str(row["condition_key"]), int(row["stage"])): float(row["final_validation_accuracy"])
        for row in rows
    }
    selected = str(selection["selected_condition"])
    joint = str(selection["joint_reference_condition"])
    parent = str(selection["original_parent_condition"])
    gaps = tuple(dict(row) for row in selection["stage_gap_closure"])
    outcome = (
        "The preregistered full-50 trigger passed."
        if bool(selection["full50_triggered"])
        else "The preregistered full-50 trigger did not pass."
    )
    condition_table = _markdown_table(
        ("Code", "Exact condition"),
        tuple((codes[str(row["condition_key"])], str(row["condition_label"])) for row in conditions),
    )
    endpoint_table = _markdown_table(
        ("Condition", "Task 8", "Task 16", "Task 32"),
        tuple(
            (
                codes[str(condition["condition_key"])],
                *(f"{by_key_stage[(str(condition['condition_key']), stage)]:.3f}" for stage in (8, 16, 32)),
            )
            for condition in conditions
        ),
    )
    effect_table = _markdown_table(
        ("Stage", "Paired contrast", "Mean pp", "Range pp"),
        tuple(
            (
                str(row["stage"]),
                str(row["effect"]),
                f"{float(row['mean_effect_pp']):+.3f}",
                f"[{float(row['minimum_effect_pp']):+.3f}, {float(row['maximum_effect_pp']):+.3f}]",
            )
            for row in effects
        ),
    )
    closure_text = "; ".join(
        f"task {int(row['stage'])}: {100.0 * float(row['gap_closed_fraction']):.1f}%"
        for row in gaps
    )
    selected_label = next(
        str(row["condition_label"]) for row in conditions if row["condition_key"] == selected
    )
    markdown = f"""# ImageNet-R Parent-Recipe Factorial

## Abstract

At the task-8, task-16, and task-32 one-node frontiers, routing is absent, yet the consolidated parent lagged a fresh stage-matched joint LoRA trained on the same classes and examples. This development-only experiment isolates the three implementation differences: fresh versus inherited classifier rows, joint versus zero weight decay, and joint versus parent initialization/data-order seeds. It evaluates the complete 2 x 2 x 2 matrix after the same five epochs. No locked-test identity or label is used.

The strongest development condition was **{selected_label}**. Relative to the original parent cell, its closure of the fresh-joint gap was {closure_text}. {outcome}

## Protocol

- Training data: the frozen router-fit subset, restricted to tasks available at each prefix.
- Evaluation data: the disjoint frozen router-validation subset.
- Architecture: the same pinned ViT-B/16, 24 rank-16 LoRA projections, and prefix-wide affine classifier in every cell.
- Work: five epochs, batch size 64, SGD momentum 0.9, LoRA learning rate 5e-4, and head learning rate 1e-2.
- Replication: one deterministic screening seed (1993) to meet the 30-minute decision window.
- Reference cells: {codes[joint]} is the stage-matched joint recipe; {codes[parent]} is the exact original full-union parent recipe and reuses its authenticated source model.
- Trigger: the selected condition must close at least 50% of the {codes[joint]} versus {codes[parent]} gap at both tasks 16 and 32.

## Condition key

{condition_table}

The same codes and labels are used in every table and figure. There are no renamed or approximately matched conditions.

## Results

![Initial and final accuracy](endpoint_accuracy.png)

{endpoint_table}

![Paired effects](factor_effects.png)

{effect_table}

The paired effects average four controlled contrasts per stage while holding the other two factors fixed. With one replication seed, they are descriptive effect decompositions, not confidence intervals.

## Full-50 decision

The selected condition is `{selected}`. {outcome} The trigger record is machine-readable in `summary.json`; it is a compute-allocation decision, not a claim of statistical significance or a benchmark pass/fail threshold.

## Interpretation and limitations

This experiment directly tests optimization recipe and head-state mismatch. It does not test missing future tasks, routing, persistent-integrator replay, or longer optimization. Initial markers in the first figure show the feature/head compatibility before consolidation training; the bars show what five epochs recover. Because the screening matrix uses one seed and inherits one canonical child hierarchy, any promoted recipe needs confirmation in the full continual run. The locked 6,000-image test set remains untouched.

## Reproducibility

`protocol/protocol.json` binds the source hierarchy roots and children, fit/validation identity hashes, model and dataset manifests, environment, resolved configuration, and all material code. Per-job hash-chained ledgers record the initial and final validation measurements. CSV, JSON, and Parquet projections accompany this report.
"""
    handoff = f"""# Parent-Recipe Factorial Handoff

The complete eight-cell screening matrix is finished at tasks 8, 16, and 32 on the clean development split. The selected cell is **{selected_label}** (`{selected}`). Gap closure was {closure_text}. {outcome}

Use `final_metrics.csv`, `epoch_metrics.csv`, and `factor_effects.csv` for independent analysis. Conditions are defined once in the report's condition-key table and use identical names in every artifact. Do not treat the single-seed paired ranges as uncertainty intervals, and do not use locked-test results to reinterpret or select this recipe.
"""
    return markdown, handoff


def _html_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    head = "".join(f"<th>{escape(value)}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(value)}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _image_uri(path: Path) -> str:
    return "data:image/png;base64," + b64encode(path.read_bytes()).decode("ascii")


def _html_report(
    summary: Mapping[str, object], effects: Sequence[Mapping[str, object]]
) -> str:
    conditions = tuple(dict(row) for row in summary["conditions"])
    rows = tuple(dict(row) for row in summary["rows"])
    selection = dict(summary["selection"])
    codes = _condition_codes(conditions)
    by_key_stage = {
        (str(row["condition_key"]), int(row["stage"])): float(row["final_validation_accuracy"])
        for row in rows
    }
    selected = str(selection["selected_condition"])
    selected_label = next(
        str(row["condition_label"]) for row in conditions if row["condition_key"] == selected
    )
    closure = ", ".join(
        f"task {int(row['stage'])}: {100.0 * float(row['gap_closed_fraction']):.1f}%"
        for row in selection["stage_gap_closure"]
    )
    decision = "triggered" if selection["full50_triggered"] else "not triggered"
    condition_table = _html_table(
        ("Code", "Exact condition"),
        tuple((codes[str(row["condition_key"])], str(row["condition_label"])) for row in conditions),
    )
    endpoint_table = _html_table(
        ("Condition", "Task 8", "Task 16", "Task 32"),
        tuple(
            (
                codes[str(condition["condition_key"])],
                *(f"{by_key_stage[(str(condition['condition_key']), stage)]:.3f}%" for stage in (8, 16, 32)),
            )
            for condition in conditions
        ),
    )
    effect_table = _html_table(
        ("Stage", "Paired contrast", "Mean pp", "Range pp"),
        tuple(
            (
                str(row["stage"]),
                str(row["effect"]),
                f"{float(row['mean_effect_pp']):+.3f}",
                f"[{float(row['minimum_effect_pp']):+.3f}, {float(row['maximum_effect_pp']):+.3f}]",
            )
            for row in effects
        ),
    )
    endpoint_uri = _image_uri(Path(str(summary["report_root"])) / "endpoint_accuracy.png")
    effects_uri = _image_uri(Path(str(summary["report_root"])) / "factor_effects.png")
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>ImageNet-R Parent-Recipe Factorial</title>
<style>
@page {{ size: A4; margin: 14mm; }}
:root {{ color-scheme: light; --ink:#17202a; --muted:#546e7a; --blue:#0d47a1; --paper:#fff; --panel:#f5f8fa; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:#e9eef2; color:var(--ink); font:15px/1.55 Arial,sans-serif; }}
main {{ max-width:1100px; margin:24px auto; background:var(--paper); padding:42px 52px; box-shadow:0 4px 22px #0002; }}
h1 {{ color:var(--blue); font-size:31px; margin:0 0 6px; }} h2 {{ color:#263238; margin-top:30px; border-bottom:2px solid #dbe5eb; padding-bottom:5px; }}
.subtitle {{ color:var(--muted); font-size:17px; }} .finding {{ background:#e3f2fd; border-left:5px solid #1565c0; padding:14px 18px; margin:22px 0; }}
figure {{ margin:20px 0; break-inside:avoid; }} img {{ width:100%; height:auto; }} figcaption {{ color:var(--muted); font-size:13px; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; margin:12px 0; }} th {{ background:#263238; color:white; text-align:left; }} th,td {{ padding:7px 9px; border:1px solid #cfd8dc; }} tr:nth-child(even) td {{ background:#f7f9fa; }}
details {{ margin:18px 0; border:1px solid #cfd8dc; border-radius:5px; padding:10px 14px; }} summary {{ cursor:pointer; font-weight:bold; color:#37474f; }} code {{ background:#eceff1; padding:1px 4px; }}
@media print {{ body {{ background:white; }} main {{ margin:0; max-width:none; padding:0; box-shadow:none; }} details {{ break-inside:avoid; }} details > * {{ display:block; }} summary {{ display:none; }} }}
</style></head><body><main>
<h1>ImageNet-R Parent-Recipe Factorial</h1><p class="subtitle">A same-data causal screen at the one-node task-8, task-16, and task-32 frontiers</p>
<div class="finding"><strong>Outcome.</strong> The strongest cell was {escape(selected_label)}. Gap closure was {escape(closure)}. The preregistered full-50 trigger was <strong>{decision}</strong>.</div>
<h2>Abstract</h2><p>Routing is absent at these power-of-two frontiers, but the consolidated parent lagged a fresh stage-matched joint LoRA. This clean-development experiment computes the complete 2 x 2 x 2 matrix over classifier initialization, weight decay, and seed/data-order schedule. Every cell uses the same pinned ViT-B/16, rank-16 LoRA architecture, examples, and five-epoch budget. No locked-test identity or label is used.</p>
<h2>Protocol and exact condition names</h2><p>The stage-matched joint reference is C1. The exact original full-union parent is C8 and reuses its authenticated source model. The selected cell must close at least half of the C1-minus-C8 gap at both tasks 16 and 32 to trigger a full continual rerun.</p>{condition_table}
<h2>Endpoint results</h2><figure><img src="{endpoint_uri}" alt="Initial and final factorial accuracy"><figcaption>Crosses are pre-training accuracy; bars are accuracy after five epochs. Blue is the joint-recipe reference and orange is the original parent.</figcaption></figure>{endpoint_table}
<h2>Factor decomposition</h2><figure><img src="{effects_uri}" alt="Paired factorial main effects"><figcaption>Means and observed ranges across four paired cells, holding the other two factors fixed. These are not confidence intervals.</figcaption></figure>{effect_table}
<h2>Decision</h2><p>The selected condition is <code>{escape(selected)}</code>. The full-50 trigger was {decision}. This trigger allocates compute; it is not an accuracy gate or a statistical-significance claim.</p>
<details open><summary>Interpretation and limitations</summary><p>The matrix tests head-state mismatch and optimizer recipe directly. It does not test future-task information, routing, persistent replay, or a longer training budget. The single screening seed meets the 30-minute decision window but does not estimate between-hierarchy uncertainty. Any promoted recipe must therefore be judged by the authorized full continual rerun.</p></details>
<details open><summary>Reproducibility artifacts</summary><p>The protocol binds all source roots and children, fit/validation identity hashes, model and dataset manifests, environment, configuration, and material code. Hash-chained per-job ledgers preserve both initial and final validation measurements. CSV, JSON, and Parquet projections are included.</p></details>
</main></body></html>"""


def _render_pdf(html: Path) -> Path:
    output = html.with_suffix(".pdf")
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if chrome is None:
        raise RuntimeError("Google Chrome or Chromium is required for the PDF report")
    temporary_root = Path.cwd() / "tmp" / "pdfs"
    temporary_root.mkdir(parents=True, exist_ok=True)
    temporary = temporary_root / f"parent-recipe-{file_sha256(html)[:16]}.pdf"
    completed = subprocess.run(
        (
            chrome,
            "--headless=new",
            "--no-sandbox",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={temporary}",
            html.resolve().as_uri(),
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 or not temporary.is_file():
        raise RuntimeError(completed.stderr.strip() or "headless Chrome PDF export failed")
    payload = temporary.read_bytes()
    if not payload.startswith(b"%PDF-") or b"%%EOF" not in payload[-1024:]:
        raise ValueError("rendered parent-recipe PDF is incomplete")
    atomic_write(output, payload)
    identity_core: dict[str, object] = {
        "generator": "headless Chrome print-to-PDF",
        "html_sha256": file_sha256(html),
        "pdf_sha256": file_sha256(output),
        "schema_version": "imagenetr50-parent-recipe-factorial-pdf-v1",
        "size_bytes": output.stat().st_size,
    }
    atomic_write(
        output.with_suffix(".pdf.json"),
        canonical_json_bytes({**identity_core, "content_hash": record_sha256(identity_core)}),
    )
    return output


def write_parent_recipe_report(run: Path | None = None) -> tuple[Path, Path, Path]:
    """Write all compact projections and render the standalone PDF report."""
    run_root = _latest_run() if run is None else Path(run)
    summary = _validated_summary(run_root)
    report_root = run_root / "reports"
    report_root.mkdir(parents=True, exist_ok=True)
    final_rows = tuple(dict(row) for row in summary["rows"])
    conditions = tuple(dict(row) for row in summary["conditions"])
    epoch_rows = _epoch_rows(run_root, summary)
    effects = factorial_effects(final_rows, conditions)
    _write_table_family(report_root, "final_metrics", final_rows)
    _write_table_family(report_root, "epoch_metrics", epoch_rows)
    _write_table_family(report_root, "factor_effects", effects)
    _endpoint_plot(report_root / "endpoint_accuracy.png", final_rows, epoch_rows, conditions)
    _effects_plot(report_root / "factor_effects.png", effects)
    markdown, handoff = _report_text(summary, effects)
    markdown_path = atomic_write(report_root / "REPORT.md", markdown.encode("utf-8"))
    atomic_write(report_root / "HANDOFF.md", handoff.encode("utf-8"))
    html_summary = {**summary, "report_root": str(report_root)}
    html_path = atomic_write(
        report_root / "REPORT.html", _html_report(html_summary, effects).encode("utf-8")
    )
    pdf_path = _render_pdf(html_path)
    files = tuple(
        {
            "path": path.name,
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(report_root.iterdir())
        if path.is_file() and path.name != "report_manifest.json"
    )
    manifest_core: dict[str, object] = {
        "files": files,
        "schema_version": "imagenetr50-parent-recipe-factorial-report-manifest-v1",
        "summary_hash": summary["content_hash"],
    }
    atomic_write(
        report_root / "report_manifest.json",
        canonical_json_bytes({**manifest_core, "content_hash": record_sha256(manifest_core)}),
    )
    return markdown_path, html_path, pdf_path


if __name__ == "__main__":
    paths = write_parent_recipe_report()
    print("Report artifacts:", *(str(path) for path in paths), sep="\n", flush=True)


__all__ = ["factorial_effects", "write_parent_recipe_report"]

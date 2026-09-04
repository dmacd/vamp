"""Focused report for the promoted full-50 persistent LogT integrator."""

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
from apm.continual.vision.imagenetr.integrator_reporting import (
    _stage_matched_joint_rows,
)


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


def _validated_locked(run: Path) -> dict[str, object]:
    record = load_canonical_json(run / "evaluations" / "locked_test.json")
    core = {key: value for key, value in record.items() if key != "content_hash"}
    if (
        record.get("schema_version")
        != "imagenetr50-promoted-integrator-locked-test-v1"
        or record.get("content_hash") != record_sha256(core)
        or len(record.get("stage_metrics", ())) != 50
    ):
        raise ValueError("promoted locked result does not authenticate")
    return record


def comparison_rows(
    run: Path, locked: Mapping[str, object]
) -> tuple[dict[str, object], ...]:
    """Align the requested persistent, oracle, and stage-matched joint curves."""
    joint = {
        int(row["stage"]): float(row["accuracy"])
        for row in _stage_matched_joint_rows(run)
    }
    persistent = {
        int(row["stage"]): dict(row) for row in locked["stage_metrics"]
    }
    if set(joint) != set(range(1, 51)) or set(persistent) != set(range(1, 51)):
        raise ValueError("comparison curves do not cover the same 50 stages")
    return tuple(
        {
            "joint_iid_minus_true_node_pp": joint[stage]
            - float(dict(persistent[stage]["controls"])["true_node_oracle"]),
            "persistent_logt_accuracy": float(persistent[stage]["accuracy"]),
            "stage": stage,
            "stage_matched_joint_iid_accuracy": joint[stage],
            "true_node_minus_persistent_pp": float(
                dict(persistent[stage]["controls"])["true_node_oracle"]
            )
            - float(persistent[stage]["accuracy"]),
            "true_node_oracle_accuracy": float(
                dict(persistent[stage]["controls"])["true_node_oracle"]
            ),
        }
        for stage in range(1, 51)
    )


def _accuracy_plot(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stages = [int(row["stage"]) for row in rows]
    fig, axis = plt.subplots(figsize=(12.5, 6.4))
    for key, label, color, style, width in (
        (
            "persistent_logt_accuracy",
            "Persistent LogT integrator (H=2048)",
            "#1565c0",
            "-",
            2.5,
        ),
        (
            "true_node_oracle_accuracy",
            "True-node oracle (fresh-parent hierarchy)",
            "#2e7d32",
            "--",
            2.2,
        ),
        (
            "stage_matched_joint_iid_accuracy",
            "Stage-matched joint-IID (rank-16 LoRA)",
            "#c62828",
            ":",
            2.7,
        ),
    ):
        axis.plot(
            stages,
            [float(row[key]) for row in rows],
            label=label,
            color=color,
            linestyle=style,
            linewidth=width,
        )
    for stage in (8, 16, 32, 50):
        axis.axvline(stage, color="#90a4ae", alpha=0.22, linewidth=1)
    axis.set(
        title="Promoted full-50 result against same-stage references",
        xlabel="Tasks seen",
        ylabel="Locked-test accuracy on classes seen so far (%)",
        xlim=(1, 50),
        ylim=(50, 100),
    )
    axis.grid(alpha=0.18)
    axis.legend(loc="lower left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)


def _gap_plot(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stages = [int(row["stage"]) for row in rows]
    fig, axis = plt.subplots(figsize=(12.5, 5.5))
    axis.plot(
        stages,
        [float(row["true_node_minus_persistent_pp"]) for row in rows],
        color="#1565c0",
        linewidth=2.2,
        label="routing/integration gap: oracle - persistent",
    )
    axis.plot(
        stages,
        [float(row["joint_iid_minus_true_node_pp"]) for row in rows],
        color="#ef6c00",
        linewidth=2.2,
        label="hierarchy gap: joint - oracle",
    )
    axis.axhline(0.0, color="#263238", linewidth=1)
    axis.set(
        title="Where the remaining deficit enters",
        xlabel="Tasks seen",
        ylabel="Accuracy difference (percentage points)",
        xlim=(1, 50),
    )
    axis.grid(alpha=0.18)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)


def _resource_rows(run: Path) -> tuple[dict[str, object], ...]:
    hierarchy_metrics = tuple(
        load_canonical_json(path)
        for path in (run / "hierarchies").glob("*/nodes/*/training_metrics.json")
    )
    integrator_ledgers = tuple(
        path
        for path in (run / "integrators" / "persistent").glob(
            "*/training_metrics.jsonl"
        )
    )
    integrator_rows = tuple(
        row
        for path in integrator_ledgers
        for row in ChainedJsonlLedger(
            path, "imagenetr50-integrator-stage-training-v1"
        ).rows
    )
    return (
        {
            "component": "hierarchy nodes",
            "image_presentations": sum(
                int(row.get("image_presentations", 0)) for row in hierarchy_metrics
            ),
            "optimizer_steps": sum(
                int(row.get("optimizer_steps", 0)) for row in hierarchy_metrics
            ),
            "peak_vram_bytes": max(
                (int(row.get("peak_vram_bytes", 0)) for row in hierarchy_metrics),
                default=0,
            ),
        },
        {
            "component": "persistent integrator",
            "image_presentations": sum(
                int(row["training_examples"])
                * int(dict(row.get("training_fit") or {}).get("epochs", 0))
                for row in integrator_rows
            ),
            "optimizer_steps": (
                max((int(row["integrator_optimizer_steps"]) for row in integrator_rows), default=0)
            ),
            "peak_vram_bytes": max(
                (
                    int(dict(row.get("training_fit") or {}).get("peak_vram_bytes", 0))
                    for row in integrator_rows
                ),
                default=0,
            ),
        },
    )


def _aggregate(rows: Sequence[Mapping[str, object]], key: str) -> float:
    return math.fsum(float(row[key]) for row in rows) / len(rows)


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    return "\n".join(
        (
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(row) + " |" for row in rows),
        )
    )


def _report_markdown(
    locked: Mapping[str, object], rows: Sequence[Mapping[str, object]]
) -> str:
    final = rows[-1]
    persistent_incremental = _aggregate(rows, "persistent_logt_accuracy")
    oracle_incremental = _aggregate(rows, "true_node_oracle_accuracy")
    joint_incremental = _aggregate(rows, "stage_matched_joint_iid_accuracy")
    checkpoints = tuple(row for row in rows if int(row["stage"]) in {8, 16, 32, 50})
    table = _markdown_table(
        ("Tasks", "Persistent LogT", "True-node oracle", "Stage-matched joint-IID"),
        tuple(
            (
                str(row["stage"]),
                f"{float(row['persistent_logt_accuracy']):.3f}",
                f"{float(row['true_node_oracle_accuracy']):.3f}",
                f"{float(row['stage_matched_joint_iid_accuracy']):.3f}",
            )
            for row in checkpoints
        ),
    )
    recipe = dict(locked["parent_recipe"])
    return f"""# ImageNet-R-50 Persistent LogT Integrator With Fresh Parents

## Abstract

The clean factorial identified inherited classifier rows as the dominant one-node consolidation failure. This full 50-task confirmation replaces every parent head with a fresh prefix-local affine head, restores weight decay 5e-4, and uses the joint model initialization and augmentation/order schedule. Leaves, the binary-counter topology, full-union parent data, rank-16 LoRA architecture, H=2,048 persistent replay, and locked 24,000/6,000 ImageNet-R split remain fixed.

The updated persistent LogT integrator reached **{float(final['persistent_logt_accuracy']):.3f}% final accuracy** and **{persistent_incremental:.3f}% incremental accuracy**. The fresh-parent true-node oracle reached {float(final['true_node_oracle_accuracy']):.3f}% final and {oracle_incremental:.3f}% incremental accuracy. The stage-matched joint-IID curve reached {float(final['stage_matched_joint_iid_accuracy']):.3f}% final and {joint_incremental:.3f}% incremental accuracy.

## Primary comparison

![Persistent LogT, true-node oracle, and stage-matched joint-IID](accuracy_comparison.png)

{table}

All three lines use the same locked test identities at each stage and are evaluated over exactly the classes seen at that stage. The joint curve uses a fresh rank-16 LoRA trained only on that prefix. The true-node oracle is diagnostic label-aware routing over the live fresh-parent hierarchy. The persistent LogT line is task-free.

## Gap decomposition

![Remaining hierarchy and integration gaps](gap_decomposition.png)

The blue curve isolates task-free prediction integration from the available live-node oracle. The orange curve isolates the remaining hierarchy/model difference from a fresh stage-matched joint fit. They should not be added to unrelated retrospective future-information comparisons.

## Promoted recipe

- Head initialization: `{recipe['head_initialization']}`.
- Parent weight decay: `{float(dict(recipe['training'])['weight_decay']):g}`.
- Seed schedule: `{recipe['seed_schedule']}`.
- Development selection source: the immutable task-8/16/32 2 x 2 x 2 parent-recipe factorial.

## Protocol integrity

The recipe was selected before this full run. All hierarchy and persistent-integrator training completed before the 6,000 locked-test identities were opened. The training seal records zero test behavior requests before completion. The report is descriptive; joint IID and E2-LoRA are comparisons, not execution gates.

## Reproducibility

`stage_comparison.csv`, `stage_comparison.json`, and `stage_comparison.parquet` contain the plotted values. `task_accuracy_matrix.*` contains every available stage/task cell. `resource_accounting.*`, the hash-chained integrator ledgers, hierarchy metadata, protocol manifests, and training seal preserve the evidence needed for independent analysis. Large model and optimizer checkpoints remain local and ignored.
"""


def _image_uri(path: Path) -> str:
    return "data:image/png;base64," + b64encode(path.read_bytes()).decode("ascii")


def _html_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    return (
        "<table><thead><tr>"
        + "".join(f"<th>{escape(value)}</th>" for value in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join(f"<td>{escape(value)}</td>" for value in row) + "</tr>"
            for row in rows
        )
        + "</tbody></table>"
    )


def _report_html(
    locked: Mapping[str, object], rows: Sequence[Mapping[str, object]], report_root: Path
) -> str:
    final = rows[-1]
    persistent_incremental = _aggregate(rows, "persistent_logt_accuracy")
    oracle_incremental = _aggregate(rows, "true_node_oracle_accuracy")
    joint_incremental = _aggregate(rows, "stage_matched_joint_iid_accuracy")
    checkpoints = tuple(row for row in rows if int(row["stage"]) in {8, 16, 32, 50})
    table = _html_table(
        ("Tasks", "Persistent LogT", "True-node oracle", "Stage-matched joint-IID"),
        tuple(
            (
                str(row["stage"]),
                f"{float(row['persistent_logt_accuracy']):.3f}%",
                f"{float(row['true_node_oracle_accuracy']):.3f}%",
                f"{float(row['stage_matched_joint_iid_accuracy']):.3f}%",
            )
            for row in checkpoints
        ),
    )
    recipe = dict(locked["parent_recipe"])
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>ImageNet-R-50 Fresh-Parent Persistent LogT</title>
<style>
@page {{ size:A4; margin:14mm; }} :root {{ --ink:#17202a; --blue:#0d47a1; --muted:#546e7a; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:#e9eef2; color:var(--ink); font:15px/1.55 Arial,sans-serif; }}
main {{ max-width:1100px; margin:24px auto; padding:42px 52px; background:white; box-shadow:0 4px 22px #0002; }}
h1 {{ color:var(--blue); font-size:31px; margin:0 0 6px; }} h2 {{ border-bottom:2px solid #dbe5eb; padding-bottom:5px; margin-top:30px; }}
.subtitle {{ color:var(--muted); font-size:17px; }} .finding {{ background:#e3f2fd; border-left:5px solid #1565c0; padding:14px 18px; margin:22px 0; }}
figure {{ margin:20px 0; break-inside:avoid; }} img {{ width:100%; height:auto; }} figcaption {{ color:var(--muted); font-size:13px; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }} th {{ background:#263238; color:white; text-align:left; }} th,td {{ padding:7px 9px; border:1px solid #cfd8dc; }} tr:nth-child(even) td {{ background:#f7f9fa; }}
details {{ border:1px solid #cfd8dc; border-radius:5px; padding:10px 14px; margin:18px 0; }} summary {{ cursor:pointer; font-weight:bold; }} code {{ background:#eceff1; padding:1px 4px; }}
@media print {{ body {{ background:white; }} main {{ margin:0; padding:0; max-width:none; box-shadow:none; }} details > * {{ display:block; }} summary {{ display:none; }} }}
</style></head><body><main><h1>ImageNet-R-50 Fresh-Parent Persistent LogT</h1><p class="subtitle">Full locked-test confirmation of the development-selected consolidation repair</p>
<div class="finding"><strong>Primary result.</strong> Persistent LogT reached {float(final['persistent_logt_accuracy']):.3f}% final / {persistent_incremental:.3f}% incremental accuracy. True-node oracle reached {float(final['true_node_oracle_accuracy']):.3f}% / {oracle_incremental:.3f}%; stage-matched joint-IID reached {float(final['stage_matched_joint_iid_accuracy']):.3f}% / {joint_incremental:.3f}%.</div>
<h2>Why this run exists</h2><p>The clean 2 x 2 x 2 factorial showed that inherited child classifier rows, not weight decay or random order, caused most of the one-node consolidation deficit. This run promotes the selected fresh-head, wd=5e-4, joint-seed recipe to every full-union parent while holding the rank-16 LoRA, topology, data, and H=2,048 persistent integrator fixed.</p>
<h2>Primary comparison</h2><figure><img src="{_image_uri(report_root / 'accuracy_comparison.png')}" alt="Persistent LogT, true-node oracle, and stage-matched joint-IID"><figcaption>Same locked identities and seen-class prefix at every stage. Persistent LogT is task-free; true-node routing is diagnostic and label-aware.</figcaption></figure>{table}
<h2>Gap decomposition</h2><figure><img src="{_image_uri(report_root / 'gap_decomposition.png')}" alt="Hierarchy and integration gaps"><figcaption>Blue: oracle minus persistent integration. Orange: stage-matched joint minus hierarchy oracle.</figcaption></figure>
<details open><summary>Promoted recipe</summary><ul><li>Head initialization: <code>{escape(str(recipe['head_initialization']))}</code></li><li>Parent weight decay: <code>{float(dict(recipe['training'])['weight_decay']):g}</code></li><li>Seed schedule: <code>{escape(str(recipe['seed_schedule']))}</code></li></ul></details>
<details><summary>Integrity and interpretation</summary><p>The factorial selection preceded this run. Hierarchy and persistent-integrator fitting completed before test behavior was opened, and the training seal records zero test requests before completion. The joint and E2-LoRA values are descriptive comparisons, never execution gates.</p></details>
<details><summary>Machine-readable evidence</summary><p>The report directory includes CSV, JSON, and Parquet stage, task, and resource projections. Hash-chained training/evaluation ledgers, hierarchy lineage, protocol manifests, and the locked training seal remain in the run tree.</p></details>
</main></body></html>"""


def write_promoted_integrator_report(run: str | Path) -> tuple[Path, Path, Path]:
    """Generate the focused comparison report and its machine-readable evidence."""
    run_root = Path(run)
    locked = _validated_locked(run_root)
    rows = comparison_rows(run_root, locked)
    report_root = run_root / "reports"
    report_root.mkdir(parents=True, exist_ok=True)
    _write_table_family(report_root, "stage_comparison", rows)
    task_rows = tuple(dict(row) for row in locked["task_accuracy_matrix"])
    _write_table_family(report_root, "task_accuracy_matrix", task_rows)
    resources = _resource_rows(run_root)
    _write_table_family(report_root, "resource_accounting", resources)
    _accuracy_plot(report_root / "accuracy_comparison.png", rows)
    _gap_plot(report_root / "gap_decomposition.png", rows)
    markdown = _report_markdown(locked, rows)
    markdown_path = atomic_write(report_root / "REPORT.md", markdown.encode("utf-8"))
    handoff = (
        "# Technical-writer handoff\n\n"
        + markdown.split("## Reproducibility", maxsplit=1)[0]
        + "Use the machine-readable stage and task tables for any independent analysis.\n"
    )
    atomic_write(report_root / "HANDOFF.md", handoff.encode("utf-8"))
    html_path = atomic_write(
        report_root / "REPORT.html",
        _report_html(locked, rows, report_root).encode("utf-8"),
    )
    pdf_path = render_integrator_pdf(html_path)
    return markdown_path, html_path, pdf_path


__all__ = ["comparison_rows", "write_promoted_integrator_report"]

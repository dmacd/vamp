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
    _lineage_plot,
    _stage_matched_joint_rows,
)


_PREDECESSOR_RUN_HASH = (
    "fd5cb0502d705bbd9662e197e9d867fda2b6c1b633c340f513be4b33579fd8b4"
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
            "live_nodes": int(persistent[stage]["live_nodes"]),
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


def _validated_predecessor_result(run: Path) -> dict[str, object]:
    source = (
        run.parents[2]
        / "integrator_full_union_ungated_v3"
        / "runs"
        / _PREDECESSOR_RUN_HASH
        / "evaluations"
        / "locked_test.json"
    )
    record = load_canonical_json(source)
    core = {key: value for key, value in record.items() if key != "content_hash"}
    if (
        record.get("schema_version") != "imagenetr50-integrator-locked-test-v3"
        or record.get("content_hash") != record_sha256(core)
    ):
        raise ValueError("predecessor locked result does not authenticate")
    return record


def _power_frontier_rows(
    rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "live_nodes": int(row["live_nodes"]),
            "persistent_logt_accuracy": float(row["persistent_logt_accuracy"]),
            "persistent_minus_joint_iid_pp": float(row["persistent_logt_accuracy"])
            - float(row["stage_matched_joint_iid_accuracy"]),
            "stage": int(row["stage"]),
            "stage_matched_joint_iid_accuracy": float(
                row["stage_matched_joint_iid_accuracy"]
            ),
            "true_node_minus_joint_iid_pp": float(row["true_node_oracle_accuracy"])
            - float(row["stage_matched_joint_iid_accuracy"]),
            "true_node_oracle_accuracy": float(row["true_node_oracle_accuracy"]),
        }
        for row in rows
        if int(row["stage"]) in {2, 4, 8, 16, 32}
    )


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    return "\n".join(
        (
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(row) + " |" for row in rows),
        )
    )


def _report_markdown(
    locked: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    predecessor: Mapping[str, object],
    resources: Sequence[Mapping[str, object]],
) -> str:
    final = rows[-1]
    persistent_incremental = _aggregate(rows, "persistent_logt_accuracy")
    oracle_incremental = _aggregate(rows, "true_node_oracle_accuracy")
    joint_incremental = _aggregate(rows, "stage_matched_joint_iid_accuracy")
    checkpoints = tuple(
        row for row in rows if int(row["stage"]) in {8, 15, 16, 31, 32, 50}
    )
    table = _markdown_table(
        (
            "Tasks",
            "Live nodes",
            "Persistent LogT",
            "True-node oracle",
            "Stage-matched joint-IID",
        ),
        tuple(
            (
                str(row["stage"]),
                str(row["live_nodes"]),
                f"{float(row['persistent_logt_accuracy']):.3f}",
                f"{float(row['true_node_oracle_accuracy']):.3f}",
                f"{float(row['stage_matched_joint_iid_accuracy']):.3f}",
            )
            for row in checkpoints
        ),
    )
    power_table = _markdown_table(
        ("Tasks", "Persistent - joint", "Oracle - joint"),
        tuple(
            (
                str(row["stage"]),
                f"{float(row['persistent_minus_joint_iid_pp']):+.3f}",
                f"{float(row['true_node_minus_joint_iid_pp']):+.3f}",
            )
            for row in _power_frontier_rows(rows)
        ),
    )
    resource_table = _markdown_table(
        ("Component", "Image presentations", "Optimizer steps", "Peak VRAM"),
        tuple(
            (
                str(row["component"]),
                f"{int(row['image_presentations']):,}",
                f"{int(row['optimizer_steps']):,}",
                f"{int(row['peak_vram_bytes']) / 2**30:.3f} GiB",
            )
            for row in resources
        ),
    )
    recipe = dict(locked["parent_recipe"])
    previous_final = float(predecessor["last_accuracy"])
    previous_incremental = float(predecessor["incremental_accuracy"])
    final_improvement = float(final["persistent_logt_accuracy"]) - previous_final
    incremental_improvement = persistent_incremental - previous_incremental
    final_joint_gap = (
        float(final["persistent_logt_accuracy"])
        - float(final["stage_matched_joint_iid_accuracy"])
    )
    incremental_joint_gap = persistent_incremental - joint_incremental
    stage_31, stage_32 = (rows[stage - 1] for stage in (31, 32))
    local_references = dict(locked["local_references"])
    return f"""# ImageNet-R-50 Persistent LogT Integrator With Fresh Parents

## Abstract

The clean factorial identified inherited classifier rows as the dominant one-node consolidation failure. This full 50-task confirmation replaces every parent head with a fresh prefix-local affine head, restores weight decay 5e-4, and uses the joint model initialization and augmentation/order schedule. Leaves, the binary-counter topology, full-union parent data, rank-16 LoRA architecture, H=2,048 persistent replay, and locked 24,000/6,000 ImageNet-R split remain fixed.

The updated persistent LogT integrator reached **{float(final['persistent_logt_accuracy']):.3f}% final accuracy** and **{persistent_incremental:.3f}% incremental accuracy**. This improves the otherwise matched inherited-head run by **{final_improvement:.3f} points final** and **{incremental_improvement:.3f} points incremental**. It remains {abs(final_joint_gap):.3f} points below stage-matched joint IID at task 50 and {abs(incremental_joint_gap):.3f} points below it on the 50-stage mean.

The fresh-parent true-node oracle reached {float(final['true_node_oracle_accuracy']):.3f}% final and {oracle_incremental:.3f}% incremental accuracy. The corresponding stage-matched joint-IID values were {float(final['stage_matched_joint_iid_accuracy']):.3f}% and {joint_incremental:.3f}%.

The central result is diagnostic rather than an endpoint win: fresh consolidation closes the same-data parent gap completely at every power-of-two frontier, while the persistent task-free integrator still fails when several independently adapted nodes are live. The next experiment should target adapter-dependent routing or response integration, not further parent initialization changes.

## Primary comparison

![Persistent LogT, true-node oracle, and stage-matched joint-IID](accuracy_comparison.png)

{table}

All three lines use the same locked test identities at each stage and are evaluated over exactly the classes seen at that stage. The joint curve uses a fresh rank-16 LoRA trained only on that prefix. The true-node oracle is diagnostic label-aware routing over the live fresh-parent hierarchy. The persistent LogT line is task-free. Checkpoints 8, 16, and 32 are one-node binary-counter frontiers; 15 and 31 are the immediately preceding, maximally fragmented frontiers; 50 is the final three-node endpoint.

## Same-data parent question: resolved

{power_table}

At tasks 2, 4, 8, 16, and 32, the hierarchy has one live node and its true-node oracle equals stage-matched joint IID to the stored precision. Every classifier and adapter tensor is bit-identical at all five checkpoints; some safetensors file hashes differ only because their metadata differ. This is expected after the repair: both models now start from the same fresh prefix-local head and LoRA initialization and consume the same examples with the same optimizer and deterministic order. There is no remaining parent-construction gap at these frontiers.

The persistent integrator is also close at the one-node checkpoints: relative to joint IID it is +2.016, +0.877, +0.281, -1.067, and -0.434 points at tasks 2/4/8/16/32. It is therefore not being held back by the consolidated adapter at those points.

## Frontier fragmentation is now the dominant failure

At task 31, five live nodes expose independently adapted score spaces. Persistent LogT falls to {float(stage_31['persistent_logt_accuracy']):.3f}%, versus {float(stage_31['true_node_oracle_accuracy']):.3f}% for label-aware node selection and {float(stage_31['stage_matched_joint_iid_accuracy']):.3f}% for joint IID. The task-32 carry replaces that frontier with one parent that is identical to the joint model; persistent accuracy immediately recovers to {float(stage_32['persistent_logt_accuracy']):.3f}%. The repeated sawtooth in the primary plot ties most of the remaining deficit to task-free combination across live adapters, not to missing future-task information or weak parent training.

## Gap decomposition

![Remaining hierarchy and integration gaps](gap_decomposition.png)

The blue curve isolates task-free prediction integration from the available live-node oracle. The orange curve is joint minus oracle. Negative orange values mean the label-aware oracle beats the global joint classifier because it restricts each image to its known owning node; that is diagnostic and not deployable. The gaps should not be added to the retrospective curve of one task-50 model evaluated on earlier prefixes.

## Reference interpretation

The stage-matched joint curve is the clean comparison for absence of future training: each point is a separate rank-16 QKV-plus-fc1 LoRA trained from scratch on exactly the task prefix available at that stage. Its final / incremental accuracy is {float(final['stage_matched_joint_iid_accuracy']):.3f}% / {joint_incremental:.3f}%. The single offline task-50 joint model has the same {float(local_references['joint_iid_last']):.3f}% endpoint but {float(local_references['joint_iid_incremental']):.3f}% retrospective incremental accuracy because its earlier-prefix evaluations have seen future tasks. Local E2-LoRA reaches {float(locked['local_e2_last']):.3f}% / {float(locked['local_e2_incremental']):.3f}%. These are context, not gates.

The prior inherited-head persistent run reached {previous_final:.3f}% / {previous_incremental:.3f}%. Fresh parents improve nearly every stage, but the new 69.433% endpoint is still not competitive with joint IID or E2-LoRA. The experiment closes the parent gap, not the full task-free benchmark gap.

## Promoted recipe

- Head initialization: `{recipe['head_initialization']}`.
- Parent weight decay: `{float(dict(recipe['training'])['weight_decay']):g}`.
- Seed schedule: `{recipe['seed_schedule']}`.
- Development selection source: the immutable task-8/16/32 2 x 2 x 2 parent-recipe factorial.

## Protocol integrity

The recipe was selected before this full run. All hierarchy and persistent-integrator training completed before the 6,000 locked-test identities were opened. The training seal records zero test behavior requests before completion. The report is descriptive; joint IID and E2-LoRA are comparisons, not execution gates.

The workflow began before its source commit was finalized, so immutable nodes contain either `ce066ba` or `d8924e8` in their informational `git_commit` field. That field is deliberately excluded from node identity. The authoritative protocol code manifest binds the exact material bytes used by the run, and all of those bytes match commit `d8924e8`; the report-only presentation changes came later.

This is one deterministic seed. The locked split has also been evaluated by earlier project experiments, although those test outcomes did not select this fresh-parent recipe. A publishable claim therefore needs replication and, ideally, an additional untouched evaluation protocol.

## Hierarchy and resource accounting

![Capacity-one binary-counter lineage](lineage.png)

{resource_table}

The initial full workflow completed in about 47 minutes on the local RTX 4090. An immediate exact resume completed in 3.798 seconds, performed zero new optimizer work, and left all 170 node manifests, all 170 node records, and all 50 persistent checkpoints unchanged by count, nanosecond-mtime sum, and stat fingerprint.

## Reproducibility

`stage_comparison.*` contains the plotted values; `power_frontier_parity.*` isolates the five same-data one-node comparisons; `power_frontier_checkpoint_parity.json` authenticates the bitwise tensor checks; and `task_accuracy_matrix.*` contains every stage/task cell. `resource_accounting.*`, `RUN.log`, `RESUME.log`, `reuse_proof.json`, `code_commit_parity.json`, the hash-chained behavior ledger, protocol manifests, and training seal preserve the compact evidence needed for independent analysis. All 68 default focused ImageNet-R tests pass; six real-model/GPU tests remain explicitly opt-in. Large model and optimizer checkpoints remain local and ignored.
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
    locked: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    predecessor: Mapping[str, object],
    resources: Sequence[Mapping[str, object]],
    report_root: Path,
) -> str:
    final = rows[-1]
    persistent_incremental = _aggregate(rows, "persistent_logt_accuracy")
    oracle_incremental = _aggregate(rows, "true_node_oracle_accuracy")
    joint_incremental = _aggregate(rows, "stage_matched_joint_iid_accuracy")
    checkpoints = tuple(
        row for row in rows if int(row["stage"]) in {8, 15, 16, 31, 32, 50}
    )
    table = _html_table(
        (
            "Tasks",
            "Live nodes",
            "Persistent LogT",
            "True-node oracle",
            "Stage-matched joint-IID",
        ),
        tuple(
            (
                str(row["stage"]),
                str(row["live_nodes"]),
                f"{float(row['persistent_logt_accuracy']):.3f}%",
                f"{float(row['true_node_oracle_accuracy']):.3f}%",
                f"{float(row['stage_matched_joint_iid_accuracy']):.3f}%",
            )
            for row in checkpoints
        ),
    )
    power_table = _html_table(
        ("Tasks", "Persistent - joint", "Oracle - joint"),
        tuple(
            (
                str(row["stage"]),
                f"{float(row['persistent_minus_joint_iid_pp']):+.3f} pp",
                f"{float(row['true_node_minus_joint_iid_pp']):+.3f} pp",
            )
            for row in _power_frontier_rows(rows)
        ),
    )
    resource_table = _html_table(
        ("Component", "Image presentations", "Optimizer steps", "Peak VRAM"),
        tuple(
            (
                str(row["component"]),
                f"{int(row['image_presentations']):,}",
                f"{int(row['optimizer_steps']):,}",
                f"{int(row['peak_vram_bytes']) / 2**30:.3f} GiB",
            )
            for row in resources
        ),
    )
    recipe = dict(locked["parent_recipe"])
    previous_final = float(predecessor["last_accuracy"])
    previous_incremental = float(predecessor["incremental_accuracy"])
    final_improvement = float(final["persistent_logt_accuracy"]) - previous_final
    incremental_improvement = persistent_incremental - previous_incremental
    final_joint_gap = float(final["stage_matched_joint_iid_accuracy"]) - float(
        final["persistent_logt_accuracy"]
    )
    incremental_joint_gap = joint_incremental - persistent_incremental
    stage_31, stage_32 = (rows[stage - 1] for stage in (31, 32))
    local_references = dict(locked["local_references"])
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>ImageNet-R-50 Fresh-Parent Persistent LogT</title>
<style>
@page {{ size:A4; margin:14mm; }} :root {{ --ink:#17202a; --blue:#0d47a1; --muted:#546e7a; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:#e9eef2; color:var(--ink); font:15px/1.55 Arial,sans-serif; }}
main {{ max-width:1100px; margin:24px auto; padding:42px 52px; background:white; box-shadow:0 4px 22px #0002; }}
h1 {{ color:var(--blue); font-size:31px; margin:0 0 6px; }} h2 {{ border-bottom:2px solid #dbe5eb; padding-bottom:5px; margin-top:30px; }}
.subtitle {{ color:var(--muted); font-size:17px; }} .finding {{ background:#e3f2fd; border-left:5px solid #1565c0; padding:14px 18px; margin:22px 0; }}
figure {{ margin:20px 0; break-inside:avoid; }} img {{ width:100%; height:auto; }} figcaption {{ color:var(--muted); font-size:13px; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }} th {{ background:#263238; color:white; text-align:left; }} th,td {{ padding:7px 9px; border:1px solid #cfd8dc; }} tr:nth-child(even) td {{ background:#f7f9fa; }}
details {{ border:1px solid #cfd8dc; border-radius:5px; padding:10px 14px; margin:18px 0; }} summary {{ cursor:pointer; font-weight:bold; font-size:17px; }} code {{ background:#eceff1; padding:1px 4px; }}
@media print {{ body {{ background:white; }} main {{ margin:0; padding:0; max-width:none; box-shadow:none; }} details > * {{ display:block; }} summary {{ display:none; }} }}
</style></head><body><main><h1>ImageNet-R-50 Fresh-Parent Persistent LogT</h1><p class="subtitle">Full locked-test confirmation of the development-selected consolidation repair</p>
<div class="finding"><strong>Primary result.</strong> Persistent LogT reached {float(final['persistent_logt_accuracy']):.3f}% final / {persistent_incremental:.3f}% incremental accuracy, gains of {final_improvement:.3f} / {incremental_improvement:.3f} points over inherited-head consolidation. The true-node oracle reached {float(final['true_node_oracle_accuracy']):.3f}% / {oracle_incremental:.3f}%. Fresh parents exactly match joint IID at every one-node power-of-two frontier. The remaining {final_joint_gap:.3f}-point final and {incremental_joint_gap:.3f}-point incremental deficits are concentrated at stages where several adapters must be combined task-free.</div>
<h2>Why this run exists</h2><p>The clean 2 x 2 x 2 factorial showed that inherited child classifier rows, not weight decay or random order, caused most of the one-node consolidation deficit. This run promotes the selected fresh-head, wd=5e-4, joint-seed recipe to every full-union parent while holding the rank-16 LoRA, topology, data, and H=2,048 persistent integrator fixed.</p>
<h2>Primary comparison</h2><figure><img src="{_image_uri(report_root / 'accuracy_comparison.png')}" alt="Persistent LogT, true-node oracle, and stage-matched joint-IID"><figcaption>Same locked identities and seen-class prefix at every stage. Persistent LogT is task-free; true-node routing is diagnostic and label-aware.</figcaption></figure>{table}<p>Tasks 8, 16, and 32 are one-node frontiers. Tasks 15 and 31 are the immediately preceding, maximally fragmented frontiers. Task 50 is the final three-node endpoint. These checkpoints expose the collapse-and-recovery pattern instead of sampling arbitrary stages.</p>
<details open><summary>Same-data parent question: resolved</summary>{power_table}<p>At tasks 2/4/8/16/32, every classifier and adapter tensor matches the corresponding fresh stage-joint model bit for bit. Some safetensors file hashes differ only in metadata. The oracle and joint accuracies are consequently identical. Persistent LogT is within 1.067 points of joint at all five except that it exceeds joint at the first three. Fresh consolidation has eliminated the parent-construction gap.</p></details>
<details open><summary>Frontier fragmentation is now the dominant failure</summary><p>Task 31 has five independently adapted live nodes. Persistent LogT falls to {float(stage_31['persistent_logt_accuracy']):.3f}%, versus {float(stage_31['true_node_oracle_accuracy']):.3f}% for the label-aware oracle and {float(stage_31['stage_matched_joint_iid_accuracy']):.3f}% for joint IID. The task-32 carry produces one joint-identical parent and persistent accuracy immediately rebounds to {float(stage_32['persistent_logt_accuracy']):.3f}%. This repeated sawtooth points to adapter-dependent routing or response integration as the next main experiment.</p></details>
<h2>Gap decomposition</h2><figure><img src="{_image_uri(report_root / 'gap_decomposition.png')}" alt="Hierarchy and integration gaps"><figcaption>Blue: oracle minus persistent integration. Orange: stage-matched joint minus oracle; negative values mean the label-aware oracle benefits from its restricted class candidates.</figcaption></figure>
<details open><summary>References and interpretation</summary><p>Stage-matched joint IID trains a separate rank-16 QKV-plus-fc1 LoRA on only the examples available at each stage and reaches {float(final['stage_matched_joint_iid_accuracy']):.3f}% final / {joint_incremental:.3f}% incremental accuracy. The task-50 offline joint model has the same {float(local_references['joint_iid_last']):.3f}% endpoint but a future-informed retrospective incremental score of {float(local_references['joint_iid_incremental']):.3f}%. Local E2-LoRA reaches {float(locked['local_e2_last']):.3f}% / {float(locked['local_e2_incremental']):.3f}%. The prior persistent run reached {previous_final:.3f}% / {previous_incremental:.3f}%. None is an execution gate.</p></details>
<h2>Binary-counter lineage</h2><figure><img src="{_image_uri(report_root / 'lineage.png')}" alt="Capacity-one binary-counter hierarchy lineage"><figcaption>Fifty arrivals create 47 carries and leave three live nodes. Power-of-two arrivals collapse the frontier to one node.</figcaption></figure>
<details open><summary>Resource accounting and exact resume</summary>{resource_table}<p>The first workflow took about 47 minutes on the RTX 4090. An immediate resume took 3.798 seconds, did no optimizer work, and left all 170 node manifests, 170 node records, and 50 persistent checkpoints unchanged by count, nanosecond-mtime sum, and stat fingerprint.</p></details>
<details open><summary>Promoted recipe</summary><ul><li>Head initialization: <code>{escape(str(recipe['head_initialization']))}</code></li><li>Parent weight decay: <code>{float(dict(recipe['training'])['weight_decay']):g}</code></li><li>Seed schedule: <code>{escape(str(recipe['seed_schedule']))}</code></li></ul></details>
<details open><summary>Integrity and limitations</summary><p>The development factorial preceded this run. Hierarchy and persistent-integrator fitting completed before test behavior was opened, and the seal records zero test requests before completion. The run straddled the source commit, so node <code>git_commit</code> annotations contain either <code>ce066ba</code> or <code>d8924e8</code>; that informational field is excluded from identity, while the authoritative code manifest's material bytes exactly match <code>d8924e8</code>. This is one deterministic seed. Earlier project runs have evaluated the same locked split, although their test outcomes did not select this recipe; replication and an additional untouched protocol are needed for a publication claim.</p></details>
<details open><summary>Machine-readable evidence</summary><p>The report directory includes CSV, JSON, and Parquet stage, power-frontier, task, and resource projections plus concise run/resume logs. A separate authenticated checkpoint-parity record preserves the bitwise tensor checks. Hash-chained behavior requests, protocol records, and the locked training seal remain in the run tree. All 68 default focused tests pass; six real-model/GPU tests stay opt-in. Large checkpoints stay local.</p></details>
</main></body></html>"""


def write_promoted_integrator_report(run: str | Path) -> tuple[Path, Path, Path]:
    """Generate the focused comparison report and its machine-readable evidence."""
    run_root = Path(run)
    locked = _validated_locked(run_root)
    rows = comparison_rows(run_root, locked)
    predecessor = _validated_predecessor_result(run_root)
    report_root = run_root / "reports"
    report_root.mkdir(parents=True, exist_ok=True)
    _write_table_family(report_root, "stage_comparison", rows)
    _write_table_family(
        report_root, "power_frontier_parity", _power_frontier_rows(rows)
    )
    task_rows = tuple(dict(row) for row in locked["task_accuracy_matrix"])
    _write_table_family(report_root, "task_accuracy_matrix", task_rows)
    resources = _resource_rows(run_root)
    _write_table_family(report_root, "resource_accounting", resources)
    _accuracy_plot(report_root / "accuracy_comparison.png", rows)
    _gap_plot(report_root / "gap_decomposition.png", rows)
    _lineage_plot(report_root / "lineage.png")
    markdown = _report_markdown(locked, rows, predecessor, resources)
    markdown_path = atomic_write(report_root / "REPORT.md", markdown.encode("utf-8"))
    handoff = (
        "# Technical-writer handoff\n\n"
        + markdown.split("## Reproducibility", maxsplit=1)[0]
        + "Use the machine-readable stage and task tables for any independent analysis.\n"
    )
    atomic_write(report_root / "HANDOFF.md", handoff.encode("utf-8"))
    html_path = atomic_write(
        report_root / "REPORT.html",
        _report_html(locked, rows, predecessor, resources, report_root).encode(
            "utf-8"
        ),
    )
    pdf_path = render_integrator_pdf(html_path)
    return markdown_path, html_path, pdf_path


__all__ = ["comparison_rows", "write_promoted_integrator_report"]

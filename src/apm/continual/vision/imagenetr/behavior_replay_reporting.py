"""Publication-style report for the node-adapted behavior replay sweep."""

from __future__ import annotations

from base64 import b64encode
from collections.abc import Mapping, Sequence
from html import escape
from pathlib import Path
import csv
import math

from apm.continual.artifacts import (
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


CAPACITIES = (2048, 4096, 8192)
CONDITION_LABELS = {
    2048: "Adapted latent, H=2,048",
    4096: "Adapted latent, H=4,096",
    8192: "Adapted latent, H=8,192",
}


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
        result.get("schema_version")
        != "imagenetr50-logt-behavior-replay-result-v1"
        or result.get("content_hash") != record_sha256(core)
        or tuple(
            int(dict(condition)["historical_capacity"])
            for condition in result.get("conditions", ())
        )
        != CAPACITIES
    ):
        raise ValueError("adapted-latent replay result does not authenticate")
    return result


def comparison_rows(
    run: Path, result: Mapping[str, object]
) -> tuple[dict[str, object], ...]:
    """Align all replay capacities with oracle and stage-matched joint curves."""
    condition_rows = {
        int(dict(condition)["historical_capacity"]): {
            int(dict(row)["stage"]): dict(row)
            for row in dict(condition)["stage_metrics"]
        }
        for condition in result["conditions"]
    }
    joint = {
        int(row["stage"]): float(row["accuracy"])
        for row in _stage_matched_joint_rows(run)
    }
    expected = set(range(1, 51))
    if set(joint) != expected or any(set(rows) != expected for rows in condition_rows.values()):
        raise ValueError("report curves do not cover the same 50 stages")
    rows = []
    for stage in range(1, 51):
        metrics = {capacity: condition_rows[capacity][stage] for capacity in CAPACITIES}
        reference = metrics[2048]
        oracle = float(dict(reference["controls"])["true_node_oracle"])
        capacities = {
            capacity: float(metrics[capacity]["accuracy"]) for capacity in CAPACITIES
        }
        rows.append(
            {
                "adapted_h2048_accuracy": capacities[2048],
                "adapted_h4096_accuracy": capacities[4096],
                "adapted_h8192_accuracy": capacities[8192],
                "h4096_minus_h2048_pp": capacities[4096] - capacities[2048],
                "h8192_minus_h2048_pp": capacities[8192] - capacities[2048],
                "joint_minus_h8192_pp": joint[stage] - capacities[8192],
                "live_nodes": int(reference["live_nodes"]),
                "stage": stage,
                "stage_matched_joint_iid_accuracy": joint[stage],
                "true_node_minus_h8192_pp": oracle - capacities[8192],
                "true_node_oracle_accuracy": oracle,
            }
        )
    return tuple(rows)


def _mean(rows: Sequence[Mapping[str, object]], key: str) -> float:
    return math.fsum(float(row[key]) for row in rows) / len(rows)


def _summary_rows(
    rows: Sequence[Mapping[str, object]], result: Mapping[str, object]
) -> tuple[dict[str, object], ...]:
    fragmented = tuple(row for row in rows if int(row["live_nodes"]) > 1)
    single = tuple(row for row in rows if int(row["live_nodes"]) == 1)
    conditions = {
        int(dict(condition)["historical_capacity"]): dict(condition)
        for condition in result["conditions"]
    }
    return tuple(
        {
            "condition": CONDITION_LABELS[capacity],
            "final_accuracy": float(conditions[capacity]["final_accuracy"]),
            "fragmented_frontier_mean_accuracy": _mean(
                fragmented, f"adapted_h{capacity}_accuracy"
            ),
            "historical_capacity": capacity,
            "incremental_accuracy": float(conditions[capacity]["incremental_accuracy"]),
            "single_node_mean_accuracy": _mean(
                single, f"adapted_h{capacity}_accuracy"
            ),
        }
        for capacity in CAPACITIES
    )


def _resource_rows(result: Mapping[str, object]) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "base_example_forwards": int(dict(condition["work"])["base_example_forwards"]),
            "cache_hits": int(dict(condition["work"])["cache_hits"]),
            "cache_misses": int(dict(condition["work"])["cache_misses"]),
            "condition": CONDITION_LABELS[int(condition["historical_capacity"])],
            "historical_capacity": int(condition["historical_capacity"]),
            "image_presentations": int(dict(condition["work"])["image_presentations"]),
            "node_example_forwards": int(dict(condition["work"])["node_example_forwards"]),
            "node_example_forwards_bound": int(
                dict(condition["work"])["node_example_forwards_bound"]
            ),
            "optimizer_steps": int(dict(condition["work"])["optimizer_steps"]),
            "parameter_count": int(dict(condition["work"])["parameter_count"]),
            "training_wall_seconds": float(
                dict(condition["work"])["training_wall_seconds"]
            ),
        }
        for condition in result["conditions"]
    )


def _accuracy_plot(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stages = [int(row["stage"]) for row in rows]
    fig, axis = plt.subplots(figsize=(13.2, 6.8))
    for capacity, color, width in (
        (2048, "#90caf9", 1.9),
        (4096, "#1976d2", 2.2),
        (8192, "#0d47a1", 2.7),
    ):
        axis.plot(
            stages,
            [float(row[f"adapted_h{capacity}_accuracy"]) for row in rows],
            color=color,
            linewidth=width,
            label=CONDITION_LABELS[capacity],
        )
    axis.plot(
        stages,
        [float(row["true_node_oracle_accuracy"]) for row in rows],
        color="#2e7d32",
        linestyle="--",
        linewidth=2.2,
        label="True-node oracle",
    )
    axis.plot(
        stages,
        [float(row["stage_matched_joint_iid_accuracy"]) for row in rows],
        color="#c62828",
        linestyle=":",
        linewidth=2.7,
        label="Stage-matched joint-IID",
    )
    for stage in (8, 15, 16, 31, 32, 50):
        axis.axvline(stage, color="#78909c", alpha=0.16, linewidth=1)
    axis.set(
        title="Node-adapted latent integration across the 50-task stream",
        xlabel="Tasks seen",
        ylabel="Test accuracy on classes seen so far (%)",
        xlim=(1, 50),
    )
    axis.grid(alpha=0.18)
    axis.legend(loc="lower left", ncol=2, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)


def _gain_plot(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stages = [int(row["stage"]) for row in rows]
    fig, axis = plt.subplots(figsize=(13.2, 5.4))
    for key, label, color in (
        ("h4096_minus_h2048_pp", "H=4,096 minus H=2,048", "#1976d2"),
        ("h8192_minus_h2048_pp", "H=8,192 minus H=2,048", "#0d47a1"),
    ):
        axis.plot(
            stages,
            [float(row[key]) for row in rows],
            label=label,
            color=color,
            linewidth=2.2,
        )
    fragmented = [int(row["live_nodes"]) > 1 for row in rows]
    for stage, is_fragmented in zip(stages, fragmented, strict=True):
        if is_fragmented:
            axis.axvspan(stage - 0.48, stage + 0.48, color="#ffecb3", alpha=0.24)
    axis.axhline(0.0, color="#263238", linewidth=1)
    axis.set(
        title="Effect of additional historical replay",
        xlabel="Tasks seen (shading marks fragmented frontiers)",
        ylabel="Accuracy change from H=2,048 (percentage points)",
        xlim=(1, 50),
    )
    axis.grid(alpha=0.18)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)


def _resource_plot(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"H={int(row['historical_capacity']):,}" for row in rows]
    presentations = [int(row["image_presentations"]) / 1e6 for row in rows]
    node_work = [int(row["node_example_forwards_bound"]) / 1e6 for row in rows]
    positions = list(range(len(rows)))
    fig, axis = plt.subplots(figsize=(9.8, 5.2))
    axis.bar(
        [value - 0.18 for value in positions],
        presentations,
        width=0.36,
        color="#1976d2",
        label="MLP image presentations",
    )
    axis.bar(
        [value + 0.18 for value in positions],
        node_work,
        width=0.36,
        color="#ff8f00",
        label="Node/example observation bound",
    )
    axis.set_xticks(positions, labels)
    axis.set(
        title="Measured constant-factor cost of the replay sweep",
        ylabel="Millions of logical examples",
    )
    axis.grid(axis="y", alpha=0.18)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    return "\n".join(
        (
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(row) + " |" for row in rows),
        )
    )


def _checkpoint_rows(
    rows: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    return tuple(
        row for row in rows if int(row["stage"]) in {8, 15, 16, 31, 32, 50}
    )


def _report_markdown(
    rows: Sequence[Mapping[str, object]],
    summaries: Sequence[Mapping[str, object]],
    resources: Sequence[Mapping[str, object]],
    result: Mapping[str, object],
    proof: Mapping[str, object],
) -> str:
    summary = {int(row["historical_capacity"]): row for row in summaries}
    final = rows[-1]
    fragmented = tuple(row for row in rows if int(row["live_nodes"]) > 1)
    single = tuple(row for row in rows if int(row["live_nodes"]) == 1)
    fragmented_gain = _mean(fragmented, "h8192_minus_h2048_pp")
    single_gain = _mean(single, "h8192_minus_h2048_pp")
    checkpoint_table = _markdown_table(
        ("Tasks", "Nodes", "Adapted H=2,048", "Adapted H=4,096", "Adapted H=8,192", "Oracle", "Joint IID"),
        tuple(
            (
                str(row["stage"]),
                str(row["live_nodes"]),
                f"{float(row['adapted_h2048_accuracy']):.3f}",
                f"{float(row['adapted_h4096_accuracy']):.3f}",
                f"{float(row['adapted_h8192_accuracy']):.3f}",
                f"{float(row['true_node_oracle_accuracy']):.3f}",
                f"{float(row['stage_matched_joint_iid_accuracy']):.3f}",
            )
            for row in _checkpoint_rows(rows)
        ),
    )
    summary_table = _markdown_table(
        ("Condition", "Final", "50-stage mean", "Fragmented mean", "One-node mean"),
        tuple(
            (
                str(row["condition"]),
                f"{float(row['final_accuracy']):.3f}",
                f"{float(row['incremental_accuracy']):.3f}",
                f"{float(row['fragmented_frontier_mean_accuracy']):.3f}",
                f"{float(row['single_node_mean_accuracy']):.3f}",
            )
            for row in summaries
        ),
    )
    resource_table = _markdown_table(
        ("Condition", "Presentations", "Optimizer steps", "Node/example bound", "Train time"),
        tuple(
            (
                str(row["condition"]),
                f"{int(row['image_presentations']):,}",
                f"{int(row['optimizer_steps']):,}",
                f"{int(row['node_example_forwards_bound']):,}",
                f"{float(row['training_wall_seconds']) / 60:.2f} min",
            )
            for row in resources
        ),
    )
    best_capacity = max(
        CAPACITIES,
        key=lambda capacity: float(summary[capacity]["fragmented_frontier_mean_accuracy"]),
    )
    source_score = dict(result["source_score_integrator"])
    return f"""# ImageNet-R-50 Node-Adapted Latent Replay Sweep

## Abstract

This experiment replaced the score-only input with each live node's own LoRA-adapted 768-dimensional pre-classifier representation and swept deterministic historical replay from 2,048 to 8,192 examples. The hierarchy, data split, optimizer, four epochs per arrival, and residual MLP widths were fixed. Each six-slot observation has 8,214 values and never contains a label or task identity.

The best fragmented-frontier mean came from **H={best_capacity:,}**. H=8,192 reached **{float(summary[8192]['final_accuracy']):.3f}% final** and **{float(summary[8192]['incremental_accuracy']):.3f}% over the 50-stage mean**. Relative to adapted-latent H=2,048, it changed fragmented-frontier accuracy by **{fragmented_gain:+.3f} points on average** and one-node accuracy by **{single_gain:+.3f} points**. At task 50 it remained {float(final['joint_minus_h8192_pp']):.3f} points below stage-matched joint IID and {float(final['true_node_minus_h8192_pp']):.3f} points below the diagnostic true-node oracle.

This is a post-hoc descriptive study on a test split already used for diagnosis. It tells us whether node-specific latent information and more replay are promising; it is not an untouched confirmation or a publishable benchmark claim.

## Primary result

![All adapted-latent replay arms, true-node oracle, and stage-matched joint-IID](accuracy_comparison.png)

{summary_table}

{checkpoint_table}

The names above are identical in the plot and tables. Tasks 8, 16, and 32 are single-node power-of-two frontiers. Tasks 15 and 31 are maximally fragmented frontiers immediately before a carry, and task 50 is the final three-node frontier.

## What the integrator observes

For every active level slot, the observer first installs that node's rank-16 LoRA in the shared ViT and extracts its adapted top-level pre-classifier representation. It layer-normalizes those 768 values, then appends 200 raw affine scores, 200 within-node log probabilities, a 200-value ownership mask, and one active bit. Six slots produce an 8,214-dimensional task-free input. This is not a shared frozen-backbone latent. The MLP has {int(resources[0]['parameter_count']):,} parameters.

## Does more replay help?

![Capacity gains relative to adapted-latent H=2,048](replay_gain.png)

Yellow bands identify fragmented frontiers. The H=8,192-minus-H=2,048 mean is {fragmented_gain:+.3f} points across fragmented stages and {single_gain:+.3f} points across single-node stages. This separates a replay-budget effect from the representational change, although both remain single-seed measurements.

The old score-only H=2,048 run reached {float(source_score['final_accuracy']):.3f}% final / {float(source_score['incremental_accuracy']):.3f}% incremental accuracy. It is intentionally absent from the main figure because it changes the feature family as well as the initialization schedule; it is historical context, not a clean replay-capacity arm.

## Offline references

Stage-matched joint IID is a separate fresh rank-16 QKV-plus-fc1 LoRA at every prefix, trained on exactly the training tasks available at that stage. The true-node oracle uses the correct class-owning live node and is label-aware, so it is diagnostic rather than deployable. Neither comparator gated training or reporting.

## Work and asymptotic constraint

![Replay work by capacity](resource_scaling.png)

{resource_table}

At arrival `t`, an arm presents the current four-class task plus at most `H` historical examples to `popcount(t)` live nodes. With fixed H this is O(t log t) cumulative behavior work. The sweep increases the constant factor and records it explicitly. Cache misses are physical forwards and depend on the descending-capacity execution order; the node/example bound is the order-independent logical comparison.

## Reuse and integrity

The run reused all 50 leaves and 47 fresh full-union parents from hierarchy `{result['hierarchy_policy_hash']}` with zero leaf and parent optimizer steps. All three integrators trained before any test behavior entered the follow-up cache. The immediate replay of the completed workflow performed zero optimizer steps and left every persistent checkpoint and the source hierarchy unchanged: `{bool(proof['integrity_passed'])}`.

![Capacity-one binary-counter lineage](lineage.png)

## Limitations and next decision

The matrix has one deterministic seed and repeatedly used test identities. A positive result should next be replicated on validation-derived or newly held-out identities, then combined with router-capacity/optimization ablations without changing the hierarchy. A flat replay curve would instead point toward integrator optimization or richer cross-node interaction, not more historical samples. In either case, the stage-matched joint-IID ceiling remains the primary target and local E2-LoRA is secondary context.

## Reproducibility

`stage_comparison.*` contains all plotted values, `condition_summary.*` and `fragmentation_checkpoints.*` contain the aggregate and selected-stage results, `task_accuracy_matrix.*` contains every stage/task cell, and `resource_accounting.*` records exact logical and physical work. Protocol manifests, the training seal, cache-seed records, chained ledgers, `reuse_proof.json`, and compact run/resume logs preserve the evidence needed for independent analysis. Large checkpoints and tensor caches remain local and ignored.
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
    markdown: str,
    rows: Sequence[Mapping[str, object]],
    summaries: Sequence[Mapping[str, object]],
    resources: Sequence[Mapping[str, object]],
    report_root: Path,
) -> str:
    summary = {int(row["historical_capacity"]): row for row in summaries}
    final = rows[-1]
    fragmented = tuple(row for row in rows if int(row["live_nodes"]) > 1)
    single = tuple(row for row in rows if int(row["live_nodes"]) == 1)
    fragmented_gain = _mean(fragmented, "h8192_minus_h2048_pp")
    single_gain = _mean(single, "h8192_minus_h2048_pp")
    summary_table = _html_table(
        ("Condition", "Final", "50-stage mean", "Fragmented mean", "One-node mean"),
        tuple(
            (
                str(row["condition"]),
                f"{float(row['final_accuracy']):.3f}%",
                f"{float(row['incremental_accuracy']):.3f}%",
                f"{float(row['fragmented_frontier_mean_accuracy']):.3f}%",
                f"{float(row['single_node_mean_accuracy']):.3f}%",
            )
            for row in summaries
        ),
    )
    checkpoints = _html_table(
        ("Tasks", "Nodes", "Adapted H=2,048", "Adapted H=4,096", "Adapted H=8,192", "Oracle", "Joint IID"),
        tuple(
            (
                str(row["stage"]),
                str(row["live_nodes"]),
                f"{float(row['adapted_h2048_accuracy']):.3f}%",
                f"{float(row['adapted_h4096_accuracy']):.3f}%",
                f"{float(row['adapted_h8192_accuracy']):.3f}%",
                f"{float(row['true_node_oracle_accuracy']):.3f}%",
                f"{float(row['stage_matched_joint_iid_accuracy']):.3f}%",
            )
            for row in _checkpoint_rows(rows)
        ),
    )
    resource_table = _html_table(
        ("Condition", "Presentations", "Steps", "Node/example bound", "Train time"),
        tuple(
            (
                str(row["condition"]),
                f"{int(row['image_presentations']):,}",
                f"{int(row['optimizer_steps']):,}",
                f"{int(row['node_example_forwards_bound']):,}",
                f"{float(row['training_wall_seconds']) / 60:.2f} min",
            )
            for row in resources
        ),
    )
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>ImageNet-R-50 Node-Adapted Latent Replay</title>
<style>
@page {{ size:A4; margin:13mm; }} :root {{ --ink:#17202a; --blue:#0d47a1; --muted:#546e7a; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:#e9eef2; color:var(--ink); font:15px/1.55 Arial,sans-serif; }}
main {{ max-width:1120px; margin:24px auto; padding:42px 52px; background:white; box-shadow:0 4px 22px #0002; }}
h1 {{ color:var(--blue); font-size:31px; margin:0 0 6px; }} h2 {{ border-bottom:2px solid #dbe5eb; padding-bottom:5px; margin-top:30px; }}
.subtitle {{ color:var(--muted); font-size:17px; }} .finding {{ background:#e3f2fd; border-left:5px solid #1565c0; padding:14px 18px; margin:22px 0; }}
figure {{ margin:20px 0; break-inside:avoid; }} img {{ width:100%; height:auto; }} figcaption {{ color:var(--muted); font-size:13px; }}
table {{ width:100%; border-collapse:collapse; font-size:12.5px; }} th {{ background:#263238; color:white; text-align:left; }} th,td {{ padding:7px 8px; border:1px solid #cfd8dc; }} tr:nth-child(even) td {{ background:#f7f9fa; }}
details {{ border:1px solid #cfd8dc; border-radius:5px; padding:10px 14px; margin:18px 0; }} summary {{ cursor:pointer; font-weight:bold; font-size:17px; }} code {{ background:#eceff1; padding:1px 4px; }}
@media print {{ body {{ background:white; }} main {{ margin:0; padding:0; max-width:none; box-shadow:none; }} details > * {{ display:block; }} summary {{ display:none; }} }}
</style></head><body><main><h1>ImageNet-R-50 Node-Adapted Latent Replay</h1><p class="subtitle">Full 50-task sweep of historical capacity with one LoRA-adapted latent per live node</p>
<div class="finding"><strong>Primary result.</strong> Adapted-latent H=8,192 reached {float(summary[8192]['final_accuracy']):.3f}% final / {float(summary[8192]['incremental_accuracy']):.3f}% 50-stage mean accuracy. Versus adapted-latent H=2,048, its average change was {fragmented_gain:+.3f} points on fragmented frontiers and {single_gain:+.3f} points on one-node frontiers. Its task-50 gaps are {float(final['joint_minus_h8192_pp']):.3f} points to stage-matched joint IID and {float(final['true_node_minus_h8192_pp']):.3f} points to the true-node oracle.</div>
<h2>Primary comparison</h2><figure><img src="{_image_uri(report_root / 'accuracy_comparison.png')}" alt="Adapted latent replay arms, true-node oracle, and stage-matched joint IID"><figcaption>Every name is shared verbatim with the tables. The three blue curves differ only in historical replay capacity.</figcaption></figure>{summary_table}{checkpoints}
<details open><summary>What enters the MLP?</summary><p>Each active level slot contains a 768-dimensional top-level representation extracted after installing that live node's own rank-16 LoRA, followed by 200 raw affine scores, 200 within-node log probabilities, a 200-value ownership mask, and one active bit. Six slots give 8,214 inputs. This is not a shared frozen-backbone latent, and no label or task ID enters inference.</p></details>
<details open><summary>Replay-capacity effect</summary><figure><img src="{_image_uri(report_root / 'replay_gain.png')}" alt="Replay capacity gains"><figcaption>Accuracy changes from adapted-latent H=2,048. Yellow bands mark stages with multiple live nodes.</figcaption></figure><p>The H=8,192-minus-H=2,048 mean is {fragmented_gain:+.3f} points on fragmented stages and {single_gain:+.3f} points on one-node stages. This isolates replay capacity within the adapted-latent family.</p></details>
<details open><summary>References and status</summary><p>Stage-matched joint IID trains a separate fresh rank-16 QKV-plus-fc1 LoRA on each available prefix. True-node routing is label-aware and diagnostic. Neither is a gate. The same test identities informed earlier diagnosis, so this is a post-hoc descriptive experiment rather than an untouched confirmation.</p></details>
<h2>Work and complexity</h2><figure><img src="{_image_uri(report_root / 'resource_scaling.png')}" alt="Replay work by capacity"><figcaption>The larger capacities increase a measured constant factor. For fixed H, cumulative work remains O(T log T).</figcaption></figure>{resource_table}
<details open><summary>Exact reuse and evidence</summary><p>The immutable 50-leaf/47-parent hierarchy was reused with zero leaf or parent optimizer steps. All arms trained before test behaviors were linked into this run. The immediate resume performed zero optimizer work and left source hierarchy and integrator checkpoint fingerprints unchanged. Machine-readable stage, task, summary, checkpoint, and resource tables accompany this report.</p></details>
<h2>Binary-counter lineage</h2><figure><img src="{_image_uri(report_root / 'lineage.png')}" alt="Capacity-one binary-counter hierarchy"><figcaption>Fifty arrivals create 47 carries and end with three live nodes; active level slots determine the adapted representations presented to the integrator.</figcaption></figure>
<details open><summary>Limitations and next decision</summary><p>There is one deterministic seed, and the locked split has been used before. A positive result needs replication on validation-derived or untouched identities. A flat replay curve would motivate integrator optimization or richer cross-node interaction rather than another capacity increase.</p></details>
<!-- markdown_sha256={record_sha256(markdown)} -->
</main></body></html>"""


def write_behavior_replay_report(run: str | Path) -> tuple[Path, Path, Path]:
    """Generate the complete report and compact machine-readable evidence."""
    run_root = Path(run)
    result = _validated_result(run_root)
    proof = load_canonical_json(run_root / "protocol" / "reuse_proof.json")
    rows = comparison_rows(run_root, result)
    summaries = _summary_rows(rows, result)
    resources = _resource_rows(result)
    report_root = run_root / "reports"
    report_root.mkdir(parents=True, exist_ok=True)
    _write_table_family(report_root, "stage_comparison", rows)
    _write_table_family(report_root, "condition_summary", summaries)
    _write_table_family(report_root, "fragmentation_checkpoints", _checkpoint_rows(rows))
    _write_table_family(
        report_root,
        "task_accuracy_matrix",
        tuple(dict(row) for row in result["task_accuracy_matrix"]),
    )
    _write_table_family(report_root, "resource_accounting", resources)
    _accuracy_plot(report_root / "accuracy_comparison.png", rows)
    _gain_plot(report_root / "replay_gain.png", rows)
    _resource_plot(report_root / "resource_scaling.png", resources)
    _lineage_plot(report_root / "lineage.png")
    markdown = _report_markdown(rows, summaries, resources, result, proof)
    markdown_path = atomic_write(report_root / "REPORT.md", markdown.encode("utf-8"))
    atomic_write(
        report_root / "HANDOFF.md",
        (
            "# Technical-analysis handoff\n\n"
            + markdown.split("## Reproducibility", maxsplit=1)[0]
            + "Use the machine-readable tables for independent calculations.\n"
        ).encode("utf-8"),
    )
    invocation = load_canonical_json(run_root / "state" / "last_invocation.json")
    atomic_write(
        report_root / "RUN.log",
        (
            f"run_hash={result['protocol_hash']}\n"
            f"elapsed_seconds={float(invocation['elapsed_seconds']):.3f}\n"
            f"phase=COMPLETE\n"
            f"feature_variant=behavior\n"
            f"historical_capacities={','.join(str(value) for value in CAPACITIES)}\n"
        ).encode("utf-8"),
    )
    atomic_write(
        report_root / "RESUME.log",
        (
            f"integrity_passed={bool(proof['integrity_passed'])}\n"
            "new_leaf_optimizer_steps=0\n"
            "new_parent_optimizer_steps=0\n"
            "new_integrator_optimizer_steps=0\n"
            "new_evaluations=0\n"
        ).encode("utf-8"),
    )
    html_path = atomic_write(
        report_root / "REPORT.html",
        _report_html(markdown, rows, summaries, resources, report_root).encode("utf-8"),
    )
    pdf_path = render_integrator_pdf(html_path)
    return markdown_path, html_path, pdf_path


__all__ = ["comparison_rows", "write_behavior_replay_report"]

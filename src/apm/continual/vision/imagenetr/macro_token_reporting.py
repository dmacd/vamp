"""Publication-style report for the ImageNet-R macro-token ceiling study."""

from __future__ import annotations

from base64 import b64encode
from collections.abc import Mapping, Sequence
from html import escape
from pathlib import Path
import csv
import math
import shutil
import subprocess

from apm.continual.artifacts import (
    atomic_write,
    canonical_json_bytes,
    file_sha256,
    load_canonical_json,
    record_sha256,
)
from apm.continual.vision.imagenetr.integrator_bank import simulate_binary_topology


CONDITION_LABELS = {
    "macro_token": "Macro-token transformer (selected)",
    "v6_control": "v6 final-CLS behavior MLP",
    "joint": "Stage-matched joint IID",
    "oracle": "True-node oracle (diagnostic)",
    "raw": "Raw union",
    "local_e2": "Local E²-LoRA final incremental (same split)",
    "published_e2": "Published E²-LoRA final incremental (external protocol)",
}


def _validated_result(run: Path) -> dict[str, object]:
    result = load_canonical_json(run / "evaluations/result.json")
    core = {key: value for key, value in result.items() if key != "content_hash"}
    if (
        result.get("schema_version") != "imagenetr50-macro-token-result-v1"
        or result.get("content_hash") != record_sha256(core)
        or tuple(int(row["stage"]) for row in result.get("stage_summaries", ()))
        != (31, 50)
    ):
        raise ValueError("macro-token result does not authenticate")
    return result


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


def _clean_summary_rows(result: Mapping[str, object]) -> tuple[dict[str, object], ...]:
    rows = []
    for record in result["clean_stages"]:
        stage = int(record["stage"])
        controls = dict(dict(record["controls"])["validation"])
        for family, entries in (
            ("macro_token", record["macro_models"]),
            ("v6_control", record["v6_controls"]),
        ):
            accuracies = tuple(
                float(dict(dict(entry["metrics"])["validation"])["accuracy"])
                for entry in entries
            )
            nlls = tuple(
                float(dict(dict(entry["metrics"])["validation"])["nll"])
                for entry in entries
            )
            rows.append(
                {
                    "condition": CONDITION_LABELS[family],
                    "mean_accuracy": math.fsum(accuracies) / len(accuracies),
                    "mean_nll": math.fsum(nlls) / len(nlls),
                    "seed_accuracies": ";".join(f"{value:.6f}" for value in accuracies),
                    "stage": stage,
                }
            )
        rows.extend(
            (
                {
                    "condition": CONDITION_LABELS["raw"],
                    "mean_accuracy": float(controls["raw_union_accuracy"]),
                    "mean_nll": None,
                    "seed_accuracies": "",
                    "stage": stage,
                },
                {
                    "condition": CONDITION_LABELS["oracle"],
                    "mean_accuracy": float(controls["true_node_oracle_accuracy"]),
                    "mean_nll": None,
                    "seed_accuracies": "",
                    "stage": stage,
                },
            )
        )
    return tuple(rows)


def _owner_rows(result: Mapping[str, object]) -> tuple[dict[str, object], ...]:
    rows = []
    for clean in result["clean_stages"]:
        for population in ("fit", "validation"):
            for key, label in (
                ("frozen_owner_probe", "Frozen linear owner probe"),
                ("owner_end_to_end", "End-to-end owner model"),
            ):
                metrics = dict(dict(clean[key])["metrics"])[population]
                rows.append(
                    {
                        "condition": label,
                        "owner_accuracy": float(metrics["accuracy"]),
                        "partition": population,
                        "routed_class_accuracy": float(metrics["owner_routed_accuracy"]),
                        "stage": int(clean["stage"]),
                    }
                )
    locked_by_stage = {
        int(path.stem.split("_")[-1]): load_canonical_json(path)
        for path in sorted((Path(str(result.get("_run", ""))) / "evaluations/locked_test").glob("stage_*.json"))
    }
    for stage, locked in locked_by_stage.items():
        for key, label in (
            ("frozen_owner_probe", "Frozen linear owner probe"),
            ("owner_end_to_end", "End-to-end owner model"),
        ):
            metrics = dict(locked[key])["test"]
            rows.append(
                {
                    "condition": label,
                    "owner_accuracy": float(metrics["accuracy"]),
                    "partition": "test",
                    "routed_class_accuracy": float(metrics["owner_routed_accuracy"]),
                    "stage": stage,
                }
            )
    return tuple(rows)


def _request_rows(run: Path) -> tuple[dict[str, object], ...]:
    path = run / "ledgers/macro_token_requests.jsonl"
    rows = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            import json

            record = json.loads(line)
            rows.append(
                {
                    "cache_bytes": int(record["cache_bytes"]),
                    "cache_hits": int(record["cache_hits"]),
                    "cache_misses": int(record["cache_misses"]),
                    "elapsed_seconds": float(record["elapsed_seconds"]),
                    "examples": int(record["examples"]),
                    "node_example_forwards": int(record["node_example_forwards"]),
                    "partition": str(record["partition"]),
                    "population_identity": str(record["population_identity"]),
                }
            )
    return tuple(rows)


def _model_resource_rows(
    run: Path, result: Mapping[str, object]
) -> tuple[dict[str, object], ...]:
    """Project per-model training work from clean and all-training artifacts."""
    stages_and_records = [
        ("clean_selection", dict(record)) for record in result["clean_stages"]
    ]
    stages_and_records.extend(
        (
            "all_train_refit",
            load_canonical_json(path),
        )
        for path in sorted((run / "evaluations/locked_test").glob("stage_*.json"))
    )
    rows: list[dict[str, object]] = []
    families = (
        ("macro_models", CONDITION_LABELS["macro_token"]),
        ("v6_controls", CONDITION_LABELS["v6_control"]),
        ("owner_end_to_end", "End-to-end owner model"),
        ("frozen_owner_probe", "Frozen linear owner probe"),
    )
    for phase, record in stages_and_records:
        for key, condition in families:
            raw_entries = record[key]
            entries = raw_entries if isinstance(raw_entries, list) else (raw_entries,)
            for entry_value in entries:
                entry = dict(entry_value)
                artifact_record = load_canonical_json(
                    run / str(entry["artifact"]) / "fit.json"
                )
                fit = dict(entry["fit"])
                spec = dict(entry.get("spec", {}))
                rows.append(
                    {
                        "best_epoch": int(fit["best_epoch"]),
                        "condition": condition,
                        "epochs_run": int(fit["epochs"]),
                        "image_presentations": int(fit["image_presentations"]),
                        "optimizer_steps": int(fit["optimizer_steps"]),
                        "parameter_count": int(artifact_record["parameter_count"]),
                        "peak_vram_bytes": int(fit["peak_vram_bytes"]),
                        "phase": phase,
                        "seed": int(spec.get("seed", entry.get("seed", -1))),
                        "stage": int(record["stage"]),
                        "wall_seconds": float(fit["wall_seconds"]),
                    }
                )
    return tuple(rows)


def _inference_cost_rows(result: Mapping[str, object]) -> tuple[dict[str, object], ...]:
    """Estimate head-only dense multiply-accumulates for each measured stage."""
    depth = int(dict(dict(result["architecture_selection"])["winner"])["depth"])
    sequence_length = 198
    transformer_block_macs = (
        4 * sequence_length * 768 * 768
        + 2 * sequence_length * sequence_length * 768
        + 2 * sequence_length * 768 * 3_072
    )
    meta_macs = 3_606 * 256 + 256 * 768
    classifier_macs = 768 * 200
    v6_macs = 8_214 * 1_024 + 1_024 * 512 + 512 * 256 + 256 * 200
    return tuple(
        row
        for stage in (31, 50)
        for row in (
            {
                "active_node_vit_forwards_per_image": stage.bit_count(),
                "condition": CONDITION_LABELS["macro_token"],
                "head_dense_macs_per_image": (
                    197 * 768 * (stage.bit_count() * 768)
                    + meta_macs
                    + depth * transformer_block_macs
                    + classifier_macs
                ),
                "stage": stage,
            },
            {
                "active_node_vit_forwards_per_image": stage.bit_count(),
                "condition": CONDITION_LABELS["v6_control"],
                "head_dense_macs_per_image": v6_macs,
                "stage": stage,
            },
        )
    )


def _accuracy_plot(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stages = [int(row["stage"]) for row in rows]
    fig, axis = plt.subplots(figsize=(10.5, 6.2))
    series = (
        ("macro_mean_accuracy", CONDITION_LABELS["macro_token"], "#1565c0", "o"),
        ("v6_control_mean_accuracy", CONDITION_LABELS["v6_control"], "#ef6c00", "s"),
        ("stage_matched_joint_iid_accuracy", CONDITION_LABELS["joint"], "#6a1b9a", "D"),
        ("true_node_oracle_accuracy", CONDITION_LABELS["oracle"], "#2e7d32", "^"),
        ("raw_union_accuracy", CONDITION_LABELS["raw"], "#546e7a", "v"),
    )
    for key, label, color, marker in series:
        axis.plot(
            stages,
            [float(row[key]) for row in rows],
            color=color,
            marker=marker,
            linewidth=2.2,
            markersize=8,
            label=label,
        )
    final = next(row for row in rows if int(row["stage"]) == 50)
    for key, label, color, marker in (
        ("local_e2_lora_incremental_accuracy", CONDITION_LABELS["local_e2"], "#8e24aa", "P"),
        ("published_e2_lora_incremental_accuracy", CONDITION_LABELS["published_e2"], "#ab47bc", "X"),
    ):
        axis.scatter(
            [50],
            [float(final[key])],
            color=color,
            marker=marker,
            s=85,
            label=label,
            zorder=5,
        )
    macro_low = [min(float(value) for value in row["macro_seed_accuracies"]) for row in rows]
    macro_high = [max(float(value) for value in row["macro_seed_accuracies"]) for row in rows]
    macro_mean = [float(row["macro_mean_accuracy"]) for row in rows]
    axis.errorbar(
        stages,
        macro_mean,
        yerr=(
            [mean - low for mean, low in zip(macro_mean, macro_low, strict=True)],
            [high - mean for mean, high in zip(macro_mean, macro_high, strict=True)],
        ),
        fmt="none",
        ecolor="#1565c0",
        capsize=5,
        alpha=0.75,
    )
    axis.set(
        xlabel="Tasks available to every fitted model",
        ylabel="Locked-test top-1 accuracy (%)",
        title="Data-matched integration ceilings at fragmented frontiers",
        xticks=stages,
    )
    axis.grid(alpha=0.22)
    axis.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)


def _selection_plot(path: Path, candidates: Sequence[Mapping[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    for axis, stage in zip(axes, (31, 50), strict=True):
        selected = tuple(row for row in candidates if int(row["stage"]) == stage)
        for depth, color, marker in ((1, "#1565c0", "o"), (2, "#c62828", "s")):
            rows = sorted(
                (row for row in selected if int(row["depth"]) == depth),
                key=lambda row: float(row["learning_rate"]),
            )
            axis.plot(
                [float(row["learning_rate"]) for row in rows],
                [float(row["validation_nll"]) for row in rows],
                color=color,
                marker=marker,
                linewidth=2,
                label=f"{depth} transformer block{'s' if depth > 1 else ''}",
            )
        axis.set_xscale("log")
        axis.set(title=f"Stage {stage}", xlabel="AdamW learning rate")
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("Clean validation NLL (lower is better)")
    axes[1].legend(fontsize=9)
    fig.suptitle("Predeclared clean architecture sweep (seed 1993)")
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)


def _owner_plot(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    test = tuple(row for row in rows if row["partition"] == "test")
    labels = [
        f"Stage {int(row['stage'])}\n{str(row['condition']).replace(' owner ', ' / owner ')}"
        for row in test
    ]
    routed = [float(row["routed_class_accuracy"]) for row in test]
    owner = [float(row["owner_accuracy"]) for row in test]
    positions = list(range(len(test)))
    fig, axis = plt.subplots(figsize=(11, 6.1))
    width = 0.36
    axis.bar([value - width / 2 for value in positions], owner, width, label="Owner-slot accuracy", color="#42a5f5")
    axis.bar([value + width / 2 for value in positions], routed, width, label="Predicted-owner routed class accuracy", color="#ffb300")
    axis.set(
        ylabel="Locked-test accuracy (%)",
        title="Does macro-CLS expose the class-owning hierarchy node?",
        xticks=positions,
        xticklabels=labels,
    )
    axis.grid(axis="y", alpha=0.22)
    axis.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)


def _lineage_plot(path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    events, _snapshots = simulate_binary_topology(50)
    fig, axis = plt.subplots(figsize=(12.5, 5.6))
    for event in events:
        parent_x = (event.parent.first_task + event.parent.last_task + 2) / 2
        for child in (event.left, event.right):
            child_x = (child.first_task + child.last_task + 2) / 2
            axis.plot(
                (child_x, parent_x),
                (child.level, event.parent.level),
                color="#90a4ae",
                linewidth=0.8,
            )
    nodes = {
        (node.level, node.first_task, node.last_task): node
        for event in events
        for node in (event.left, event.right, event.parent)
    }
    axis.scatter(
        [(node.first_task + node.last_task + 2) / 2 for node in nodes.values()],
        [node.level for node in nodes.values()],
        s=13,
        color="#1565c0",
        zorder=3,
    )
    for stage in (31, 50):
        axis.axvline(stage, color="#c62828", linestyle="--", alpha=0.55)
        axis.text(stage + 0.25, 5.7, f"stage {stage}", color="#8e0000", fontsize=9)
    axis.set(
        title="Fixed capacity-one hierarchy supplying stable level slots",
        xlabel="Task interval midpoint",
        ylabel="Binary-counter level",
    )
    axis.grid(alpha=0.17)
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)


def _image_uri(path: Path) -> str:
    return f"data:image/png;base64,{b64encode(path.read_bytes()).decode('ascii')}"


def _html_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    head = "".join(f"<th>{escape(value)}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(value)}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _report_text(
    result: Mapping[str, object],
    candidates: Sequence[Mapping[str, object]],
    owner_rows: Sequence[Mapping[str, object]],
) -> tuple[str, str]:
    winner = dict(dict(result["architecture_selection"])["winner"])
    stages = tuple(dict(row) for row in result["stage_summaries"])
    stage_lines = "\n".join(
        "| {stage} | {macro:.3f} | {control:.3f} | {raw:.3f} | {owner:.3f} | {joint:.3f} | {gap:+.3f} |".format(
            stage=int(row["stage"]),
            macro=float(row["macro_mean_accuracy"]),
            control=float(row["v6_control_mean_accuracy"]),
            raw=float(row["raw_union_accuracy"]),
            owner=float(row["true_node_oracle_accuracy"]),
            joint=float(row["stage_matched_joint_iid_accuracy"]),
            gap=float(row["macro_minus_joint_iid"]),
        )
        for row in stages
    )
    improvements = tuple(
        float(row["macro_mean_accuracy"]) - float(row["v6_control_mean_accuracy"])
        for row in stages
    )
    finding = (
        f"The selected macro-token model changed locked-test accuracy relative to the data-matched v6 MLP by "
        f"{improvements[0]:+.3f} points at stage 31 and {improvements[1]:+.3f} points at stage 50. "
        f"Its signed differences from stage-matched joint IID were "
        f"{float(stages[0]['macro_minus_joint_iid']):+.3f} and {float(stages[1]['macro_minus_joint_iid']):+.3f} points. "
        f"At stage 50 it differed from local E²-LoRA by "
        f"{float(stages[1]['macro_mean_accuracy']) - float(stages[1]['local_e2_lora_incremental_accuracy']):+.3f} points."
    )
    owner_test = tuple(row for row in owner_rows if row["partition"] == "test")
    owner_sentence = " ".join(
        f"At stage {int(row['stage'])}, {str(row['condition']).lower()} reached "
        f"{float(row['owner_accuracy']):.3f}% owner accuracy and "
        f"{float(row['routed_class_accuracy']):.3f}% routed class accuracy."
        for row in owner_test
    )
    markdown = f"""# ImageNet-R-50 Macro-Token Integrator Ceiling Study

## Abstract

This experiment tests whether the fragmented-frontier deficit comes from discarding patch-level information before integration. Every live node supplies its full final 197-token, LoRA-adapted ViT representation. Corresponding token positions are fused across six stable hierarchy levels, then integrated by a small transformer. Architecture choice used an end-to-end clean 19,200/4,800 fit/validation hierarchy. Only the selected model and a data-matched v6 final-CLS MLP were refit on all training images and evaluated on locked test.

**Main result.** {finding}

## Locked-test comparison

| Stage | Macro-token mean | v6 MLP mean | Raw union | True-node oracle | Joint IID | Macro − joint |
|---:|---:|---:|---:|---:|---:|---:|
{stage_lines}

Seed ranges are shown in `accuracy_comparison.png`; they are observed three-seed ranges, not confidence intervals. Stage-matched joint IID uses the same pinned rank-16 QKV-plus-fc1 LoRA and affine classifier architecture as each fresh consolidation node, trained jointly on the exact prefix training set for five epochs. It is an offline ceiling, not a gate.

The local E²-LoRA reference is {float(stages[1]['local_e2_lora_incremental_accuracy']):.3f}% final incremental accuracy on the same frozen split. The published {float(stages[1]['published_e2_lora_incremental_accuracy']):.3f}% is external protocol context, not a matched threshold. Both appear only at stage 50 because no stage-31 E²-LoRA measurement exists.

## Clean architecture selection

The predeclared seed-1993 sweep crossed one or two transformer blocks with learning rates 0.0001, 0.0003, and 0.001 at stages 31 and 50. The winner was depth {int(winner['depth'])}, learning rate {float(winner['learning_rate']):g}, selected by mean clean validation NLL {float(winner['mean_validation_nll']):.6f}. Ties would have preferred fewer blocks and then lower learning rate. Seeds 1994 and 1995 repeated only this winner. Rejected cells never saw test.

## Owner information

Owner labels were never part of inference inputs. A frozen linear probe asks whether class-trained macro-CLS already makes the owning slot linearly available. A separate end-to-end owner transformer asks how much the same input can expose when optimized directly for ownership. {owner_sentence} The true-node oracle remains label-aware and diagnostic; predicted-owner routing is task-free.

## Architecture and data boundary

Each active node produces a normalized 197 × 768 sequence with its own LoRA installed. The shared cross-slot projection is exactly equivalent to `Linear(4608, 768)` over six zero-padded stable slots, without materializing the zeros. A 3,606-value behavior vector—raw logits, local log probabilities, ownership, and active bits—becomes one META token. The selected one- or two-block transformer processes 198 width-768 tokens and classifies directly from macro-CLS. It has no raw-union residual skip. Exact trainable parameter counts are 12,055,496 for one block and 19,143,368 for two.

The v6 control consumes the same cached node computations, but retains only final CLS plus behavior fields in its 8,214-value input. Clean upstream nodes never trained on validation images. Locked refits used all prefix training images for each model's own clean-selected epoch count. The training seal recorded zero test-token requests.

## Resource and reproducibility boundary

Full token sequences were BF16 stage-local scratch in immutable 64-image shards with a 64-GiB cap. They were removed after model/evaluation seals. Request and model manifests retain exact cache bytes, adapted-node forward counts, optimizer work, wall time, and peak VRAM. The immediate reuse proof required zero new hierarchy optimizer steps, zero new adapted-token requests, byte-identical model artifacts, and empty token scratch.

`inference_cost.csv` reports head-only dense multiply-accumulates and the common number of node-adapted ViT forwards per image. It excludes LayerNorm, nonlinearities, softmax, and data movement, so it is an architectural workload estimate rather than a latency measurement.

## Interpretation

The macro-versus-v6 contrast isolates patch retention and spatial cross-node integration under matched data, source nodes, and optimizer-selection protocol. A positive contrast supports the claim that pooled final CLS omitted useful cross-node evidence; a small or negative contrast shifts attention toward node quality, regularization, or the learning target. The owner probes distinguish information availability from class-head exploitation, but do not by themselves establish a causal routing mechanism. Two stages and three classifier seeds are enough for a ceiling study, not a final SOTA claim.

## Artifacts

`stage_summary`, `clean_candidates`, `clean_summary`, `owner_diagnostics`, `task_accuracy_matrix`, and `resource_accounting` are emitted as CSV, JSON, and Parquet. `REPORT.html` is self-contained. `protocol/architecture_selection.json`, `protocol/training_seal.json`, and `protocol/reuse_proof.json` preserve the selection, leakage, and resume boundaries.
"""
    handoff = f"""# GPT Pro handoff: ImageNet-R macro-token ceiling

The run is complete and ungated. {finding}

The selected architecture is depth {int(winner['depth'])} at learning rate {float(winner['learning_rate']):g}. Start with `stage_summary.csv`, then inspect `owner_diagnostics.csv`, `clean_candidates.csv`, and `resource_accounting.csv`. The central scientific questions are whether macro tokens consistently beat the data-matched v6 MLP, how much gap remains to stage-matched joint IID and the true-node oracle, and whether the frozen versus end-to-end owner results indicate missing owner information or failure of the class objective to use it. Do not treat the published or local E2-LoRA values as protocol gates or claim SOTA from this two-stage ceiling study.
"""
    return markdown, handoff


def _html_report(
    result: Mapping[str, object],
    report_root: Path,
    markdown: str,
    owner_rows: Sequence[Mapping[str, object]],
    inference_rows: Sequence[Mapping[str, object]],
) -> str:
    stages = tuple(dict(row) for row in result["stage_summaries"])
    winner = dict(dict(result["architecture_selection"])["winner"])
    summary_table = _html_table(
        (
            "Stage",
            CONDITION_LABELS["macro_token"],
            CONDITION_LABELS["v6_control"],
            CONDITION_LABELS["raw"],
            CONDITION_LABELS["oracle"],
            CONDITION_LABELS["joint"],
            CONDITION_LABELS["local_e2"],
        ),
        tuple(
            (
                str(row["stage"]),
                f"{float(row['macro_mean_accuracy']):.3f}%",
                f"{float(row['v6_control_mean_accuracy']):.3f}%",
                f"{float(row['raw_union_accuracy']):.3f}%",
                f"{float(row['true_node_oracle_accuracy']):.3f}%",
                f"{float(row['stage_matched_joint_iid_accuracy']):.3f}%",
                (
                    "—"
                    if row["local_e2_lora_incremental_accuracy"] is None
                    else f"{float(row['local_e2_lora_incremental_accuracy']):.3f}%"
                ),
            )
            for row in stages
        ),
    )
    owner_table = _html_table(
        ("Stage", "Partition", "Diagnostic", "Owner accuracy", "Routed class accuracy"),
        tuple(
            (
                str(row["stage"]),
                str(row["partition"]),
                str(row["condition"]),
                f"{float(row['owner_accuracy']):.3f}%",
                f"{float(row['routed_class_accuracy']):.3f}%",
            )
            for row in owner_rows
        ),
    )
    inference_table = _html_table(
        ("Stage", "Condition", "Active ViT forwards/image", "Head dense MACs/image"),
        tuple(
            (
                str(row["stage"]),
                str(row["condition"]),
                str(row["active_node_vit_forwards_per_image"]),
                f"{int(row['head_dense_macs_per_image']):,}",
            )
            for row in inference_rows
        ),
    )
    macro_joint_differences = tuple(
        float(row["macro_minus_joint_iid"]) for row in stages
    )
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>ImageNet-R Macro-Token Integrator</title>
<style>
@page {{ size:A4; margin:14mm; }} :root {{ --ink:#17202a; --muted:#52636d; --blue:#0d47a1; --paper:#fff; --panel:#f4f8fb; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:#e8eef2; color:var(--ink); font:15px/1.55 Arial,sans-serif; }} main {{ max-width:1120px; margin:24px auto; background:var(--paper); padding:42px 54px; box-shadow:0 4px 22px #0002; }}
h1 {{ color:var(--blue); font-size:31px; margin:0 0 7px; }} h2 {{ color:#263238; border-bottom:2px solid #dce7ed; margin-top:30px; padding-bottom:5px; }} .subtitle {{ color:var(--muted); font-size:17px; }} .finding {{ background:#e3f2fd; border-left:5px solid #1565c0; padding:14px 18px; margin:20px 0; }}
figure {{ margin:20px 0; break-inside:avoid; }} img {{ width:100%; height:auto; }} figcaption {{ color:var(--muted); font-size:13px; }} table {{ width:100%; border-collapse:collapse; font-size:12.5px; margin:12px 0; }} th {{ background:#263238; color:white; text-align:left; }} th,td {{ padding:7px 8px; border:1px solid #cfd8dc; }} tr:nth-child(even) td {{ background:#f7f9fa; }}
details {{ margin:18px 0; border:1px solid #cfd8dc; border-radius:5px; padding:10px 14px; }} summary {{ cursor:pointer; font-weight:bold; }} code {{ background:#eceff1; padding:1px 4px; }}
@media print {{ body {{ background:white; }} main {{ margin:0; max-width:none; padding:0; box-shadow:none; }} details {{ break-inside:avoid; }} details > * {{ display:block; }} summary {{ display:none; }} .screen-only {{ display:none; }} }}
</style></head><body><main>
<h1>ImageNet-R-50 Macro-Token Integrator</h1><p class="subtitle">Clean architecture selection, all-training refit, and locked-test ceiling study</p>
<div class="finding"><strong>Selected model.</strong> {int(winner['depth'])} transformer block{'s' if int(winner['depth']) > 1 else ''}, learning rate {float(winner['learning_rate']):g}, chosen by mean clean validation NLL. The run has no accuracy gate.</div>
<p><strong>Main result.</strong> Relative to the data-matched v6 MLP, macro-token accuracy changed by {float(stages[0]['macro_mean_accuracy']) - float(stages[0]['v6_control_mean_accuracy']):+.3f} points at stage 31 and {float(stages[1]['macro_mean_accuracy']) - float(stages[1]['v6_control_mean_accuracy']):+.3f} points at stage 50. Its signed differences from stage-matched joint IID were {macro_joint_differences[0]:+.3f} and {macro_joint_differences[1]:+.3f} points.</p>
<h2>Primary result</h2><figure><img src="{_image_uri(report_root / 'accuracy_comparison.png')}" alt="Macro-token, v6 MLP, raw union, true-node oracle, and stage-matched joint IID accuracy"><figcaption>Names here and in every table are identical. Macro-token and v6 points are three-seed means; error bars show the observed macro-token range.</figcaption></figure>{summary_table}
<h2>Clean selection</h2><figure><img src="{_image_uri(report_root / 'clean_selection.png')}" alt="Clean validation NLL sweep"><figcaption>Only training-derived clean validation chose depth and learning rate. Rejected cells were not evaluated on test.</figcaption></figure>
<details open><summary>Exact architecture</summary><p>Each node contributes all 197 final adapted ViT tokens. Corresponding fixed-slot positions are projected from 4,608 to 768, a compact 3,606-value behavior META token is appended, and one or two width-768 transformer blocks classify directly from macro-CLS. No raw-union residual skip is present.</p></details>
<h2>Owner diagnostics</h2><figure><img src="{_image_uri(report_root / 'owner_diagnostics.png')}" alt="Owner prediction and routed class accuracy"><figcaption>The frozen linear probe measures linearly available owner information in class-trained macro-CLS. The separate end-to-end model optimizes the same inputs for owner prediction. Neither receives labels at inference.</figcaption></figure>{owner_table}
<h2>Fixed hierarchy</h2><figure><img src="{_image_uri(report_root / 'lineage.png')}" alt="Capacity-one binary counter hierarchy"><figcaption>Stage 31 has five live level slots and stage 50 has three. Clean and all-training hierarchies share this topology but use different upstream training populations.</figcaption></figure>
<details open><summary>Controls and interpretation</summary><p>Stage-matched joint IID is a fresh rank-16 QKV-plus-fc1 LoRA and affine head trained jointly on each exact prefix for five epochs. True-node oracle is label-aware and diagnostic. The v6 MLP receives the same cached node computation and training populations as the macro model, isolating the information boundary and head structure.</p></details>
<details open><summary>Inference workload</summary><p>These are head-only dense multiply-accumulate estimates, not measured latency. Every condition also performs the listed number of node-specific adapted ViT forwards.</p>{inference_table}</details>
<details open><summary>Leakage, resources, and reuse</summary><p>The clean hierarchy excluded all 4,800 validation identities from upstream training. Every all-training refit completed before the training seal, which recorded zero test-token requests. BF16 token shards were capped at 64 GiB and removed after sealing. Immediate replay performed zero hierarchy optimizer work and zero adapted-token forwards.</p></details>
<details class="screen-only"><summary>Plain-text report</summary><pre>{escape(markdown)}</pre></details>
</main></body></html>"""


def _render_pdf(html: Path) -> Path:
    output = html.with_suffix(".pdf")
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if chrome is None:
        raise RuntimeError("Google Chrome or Chromium is required for PDF export")
    temporary_root = Path.cwd() / "tmp/pdfs"
    temporary_root.mkdir(parents=True, exist_ok=True)
    temporary = temporary_root / f"macro-token-{file_sha256(html)[:16]}.pdf"
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
        raise ValueError("rendered macro-token PDF is incomplete")
    atomic_write(output, payload)
    temporary.unlink()
    identity_core: dict[str, object] = {
        "generator": "headless Chrome print-to-PDF",
        "html_sha256": file_sha256(html),
        "pdf_sha256": file_sha256(output),
        "schema_version": "imagenetr50-macro-token-pdf-v1",
        "size_bytes": output.stat().st_size,
    }
    atomic_write(
        output.with_suffix(".pdf.json"),
        canonical_json_bytes({**identity_core, "content_hash": record_sha256(identity_core)}),
    )
    return output


def write_macro_token_report(run: str | Path) -> tuple[Path, Path, Path]:
    """Generate tables, plots, self-contained HTML, Markdown, and PDF."""
    run_root = Path(run)
    result = _validated_result(run_root)
    result_with_run = {**result, "_run": str(run_root)}
    report_root = run_root / "reports"
    report_root.mkdir(parents=True, exist_ok=True)
    candidates = tuple(
        {
            key: value
            for key, value in row.items()
            if key
            not in {"format", "previous_sha256", "result_sha256", "sequence"}
        }
        for row in _jsonl_rows(run_root / "evaluations/clean_candidates.jsonl")
    )
    stage_rows = tuple(dict(row) for row in result["stage_summaries"])
    clean_rows = _clean_summary_rows(result)
    owner_rows = _owner_rows(result_with_run)
    resource_rows = _request_rows(run_root)
    model_resource_rows = _model_resource_rows(run_root, result)
    inference_rows = _inference_cost_rows(result)
    task_rows = tuple(dict(row) for row in result["task_accuracy_matrix"])
    for name, rows in (
        ("stage_summary", stage_rows),
        ("clean_candidates", candidates),
        ("clean_summary", clean_rows),
        ("owner_diagnostics", owner_rows),
        ("task_accuracy_matrix", task_rows),
        ("resource_accounting", resource_rows),
        ("model_resource_accounting", model_resource_rows),
        ("inference_cost", inference_rows),
    ):
        _write_table_family(report_root, name, rows)
    _accuracy_plot(report_root / "accuracy_comparison.png", stage_rows)
    _selection_plot(report_root / "clean_selection.png", candidates)
    _owner_plot(report_root / "owner_diagnostics.png", owner_rows)
    _lineage_plot(report_root / "lineage.png")
    markdown, handoff = _report_text(result, candidates, owner_rows)
    markdown_path = atomic_write(report_root / "REPORT.md", markdown.encode("utf-8"))
    atomic_write(report_root / "HANDOFF.md", handoff.encode("utf-8"))
    atomic_write(
        report_root / "RUN.log",
        (
            "ImageNet-R macro-token ceiling study complete.\n"
            f"Run: {run_root.name}\n"
            f"Result: {result['content_hash']}\n"
            "See resource_accounting.csv for every cache request and adapted-node forward.\n"
        ).encode("utf-8"),
    )
    reuse = load_canonical_json(run_root / "protocol/reuse_proof.json")
    atomic_write(
        report_root / "RESUME.log",
        (
            "Immediate resume/reuse proof.\n"
            f"Integrity passed: {reuse['integrity_passed']}\n"
            f"Requests before/after: {reuse['request_rows_before']}/{reuse['request_rows_after']}\n"
            f"Acceptance: {reuse['acceptance']}\n"
        ).encode("utf-8"),
    )
    html_path = atomic_write(
        report_root / "REPORT.html",
        _html_report(
            result, report_root, markdown, owner_rows, inference_rows
        ).encode("utf-8"),
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
        "result_hash": result["content_hash"],
        "schema_version": "imagenetr50-macro-token-report-manifest-v1",
    }
    atomic_write(
        report_root / "report_manifest.json",
        canonical_json_bytes({**manifest_core, "content_hash": record_sha256(manifest_core)}),
    )
    return markdown_path, html_path, pdf_path


def _jsonl_rows(path: Path) -> tuple[dict[str, object], ...]:
    """Read compact ledger projections without mutating their hash chain."""
    import json

    return tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


if __name__ == "__main__":
    latest = load_canonical_json(
        Path("artifacts/imagenetr50/macro_token_integrator_v8/LATEST_RUN.json")
    )
    root = Path("artifacts/imagenetr50/macro_token_integrator_v8/runs") / str(
        latest["run_hash"]
    )
    print(*write_macro_token_report(root), sep="\n")


__all__ = ["write_macro_token_report"]

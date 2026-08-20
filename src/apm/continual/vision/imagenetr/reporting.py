"""Deterministic ledgers, plots, lineage, and self-contained experiment reports."""

from __future__ import annotations

from base64 import b64encode
from dataclasses import asdict
from html import escape
from io import BytesIO, StringIO
from pathlib import Path
from collections.abc import Mapping, Sequence
import csv
import json
import math

import torch

from apm.continual.artifacts import (
    atomic_write,
    canonical_json_bytes,
    load_canonical_json,
    publish_immutable_json,
)
from apm.continual.vision.imagenetr.artifacts import NodeBundle, VisionStore
from apm.continual.vision.imagenetr.bank import simulate_topology
from apm.continual.vision.imagenetr.config import ImageNetRConfig
from apm.continual.vision.imagenetr.evaluation import EvaluationResult
from apm.continual.vision.imagenetr.external import E2LoRAResult
from apm.continual.vision.imagenetr.lineage import TreeBuildResult
from apm.continual.vision.imagenetr.manifests import sealed_manifest
from apm.continual.vision.imagenetr.metrics import incremental_average, mean_forgetting
from apm.continual.vision.imagenetr.preflight import PreflightResult
from apm.continual.vision.imagenetr.protocol import ResolvedProtocol
from apm.continual.vision.imagenetr.scheduler import LocalScheduler


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = tuple(sorted({key for row in rows for key in row}))
    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(path, output.getvalue().encode("utf-8"))


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    atomic_write(path, b"".join(canonical_json_bytes(dict(row)) for row in rows))


def _condition_summary(result: EvaluationResult) -> dict[str, object]:
    condition = result.stages[0].condition
    modes = tuple(sorted({row.score_mode for row in result.stages}))
    values: dict[str, object] = {"condition": condition}
    for mode in modes:
        stage_rows = tuple(
            sorted((row for row in result.stages if row.score_mode == mode), key=lambda row: row.stage)
        )
        values[f"{mode}_last_accuracy"] = stage_rows[-1].accuracy
        values[f"{mode}_incremental_accuracy"] = incremental_average(
            tuple(row.accuracy for row in stage_rows)
        )
        task_count = stage_rows[-1].stage
        matrix = tuple(
            tuple(
                next(
                    (
                        row.accuracy
                        for row in result.tasks
                        if row.score_mode == mode and row.stage == stage and row.task == task
                    ),
                    None,
                )
                for task in range(task_count)
            )
            for stage in range(1, task_count + 1)
        )
        values[f"{mode}_mean_forgetting"] = mean_forgetting(matrix)
    values["cache_hits"] = result.cache_hits
    values["cache_misses"] = result.cache_misses
    return values


def _merge_rows(store: VisionStore) -> tuple[dict[str, object], ...]:
    rows = []
    unrepaired: dict[str, dict[str, object]] = {}
    for path in sorted((store.run / "merge_cache").rglob("merge_diagnostics.json")):
        record = load_canonical_json(path)
        unrepaired[str(record["parent_content_hash"])] = dict(record)
        modules = record.pop("module_diagnostics", {})
        common = {
            key: json.dumps(value, separators=(",", ":"))
            if isinstance(value, (list, dict))
            else value
            for key, value in record.items()
        }
        if not modules:
            rows.append({**common, "module": None})
        for module, diagnostic in sorted(modules.items()):
            rows.append(
                {
                    **common,
                    **{
                        key: json.dumps(value, separators=(",", ":"))
                        if isinstance(value, (list, dict))
                        else value
                        for key, value in diagnostic.items()
                    },
                    "module": module,
                }
            )
    for path in sorted((store.run / "trees").rglob("repair_metrics.json")):
        repair = load_canonical_json(path)
        node = load_canonical_json(path.parent / "node.json")
        source = unrepaired[str(node["unrepaired_parent_hash"])]
        tree_relative = path.relative_to(store.run / "trees")
        policy_hash = tree_relative.parts[0]
        rows.append(
            {
                **{
                    key: json.dumps(value, separators=(",", ":"))
                    if isinstance(value, (list, dict))
                    else value
                    for key, value in source.items()
                    if key != "module_diagnostics"
                },
                **repair,
                "module": None,
                "parent_content_hash": node["content_hash"],
                "policy_hash": policy_hash,
                "unrepaired_parent_hash": node["unrepaired_parent_hash"],
            }
        )
    return tuple(rows)


def _routing_rows(evaluations: Mapping[str, EvaluationResult]) -> tuple[dict[str, object], ...]:
    rows = []
    for condition, result in sorted(evaluations.items()):
        by_key = {(row.stage, row.score_mode): row for row in result.stages}
        for stage in sorted({row.stage for row in result.stages}):
            oracle = by_key[(stage, "true_node_oracle")].accuracy
            for mode in ("raw", "cosine", "affine_calibrated", "centroid_router"):
                routed = by_key[(stage, mode)].accuracy
                stage_row = by_key[(stage, mode)]
                rows.append(
                    {
                        "candidate_adapter_forwards": stage_row.candidate_forwards,
                        "condition": condition,
                        "frozen_router_forwards": stage_row.frozen_router_forwards,
                        "node_routing_accuracy": stage_row.routing_accuracy,
                        "routing_regret": oracle - routed,
                        "score_mode": mode,
                        "stage": stage,
                        "task_free_accuracy": routed,
                        "true_node_oracle_accuracy": oracle,
                    }
                )
    return tuple(rows)


def _resource_records(
    store: VisionStore,
    trees: Mapping[str, TreeBuildResult],
    evaluations: Mapping[str, EvaluationResult],
    config: ImageNetRConfig,
) -> dict[str, object]:
    per_lora = 12 * config.lora_rank * ((768 + 2304) + (768 + 3072))
    directories: dict[str, list[Path]] = {}
    for node_path in sorted(store.run.rglob("node.json")):
        node_record = load_canonical_json(node_path)
        directories.setdefault(str(node_record["content_hash"]), []).append(node_path.parent)

    def auxiliary(bundle: NodeBundle, filename: str) -> dict[str, object] | None:
        hashes = [bundle.artifact.content_hash]
        if filename == "merge_diagnostics.json" and bundle.artifact.unrepaired_parent_hash:
            hashes.append(bundle.artifact.unrepaired_parent_hash)
        candidates = (bundle.directory,) + tuple(
            directory
            for node_hash in hashes
            for directory in directories.get(node_hash, ())
        )
        path = next((directory / filename for directory in candidates if (directory / filename).is_file()), None)
        return None if path is None else load_canonical_json(path)

    records = {}
    for name, tree in trees.items():
        unique_nodes = {
            bundle.artifact.content_hash: bundle for bundle in tree.nodes
        }
        final_hashes = tree.snapshots[-1].node_hashes
        final_nodes = tuple(unique_nodes[value] for value in final_hashes)
        merge_metrics = [
            value
            for bundle in unique_nodes.values()
            if (value := auxiliary(bundle, "merge_diagnostics.json")) is not None
        ]
        repair_metrics = [
            value
            for bundle in unique_nodes.values()
            if (value := auxiliary(bundle, "repair_metrics.json")) is not None
        ]
        leaf_metrics = [
            value
            for bundle in unique_nodes.values()
            if bundle.artifact.consolidation_method == "leaf"
            and (value := auxiliary(bundle, "training_metrics.json")) is not None
        ]
        baseline_metrics = tuple(
            load_canonical_json(path)
            for path in sorted((store.run / "baselines" / name).rglob("training_metrics.json"))
        ) if name in {"frozen_reference", "seq_lora_r16", "joint_iid_lora_r16"} else ()
        leaf_presentations = (
            sum(int(value["image_presentations"]) for value in leaf_metrics)
            if leaf_metrics
            else (int(baseline_metrics[-1]["image_presentations"]) if baseline_metrics else 0)
        )
        parent_training_presentations = sum(
            int(value.get("merge_image_presentations", 0)) for value in merge_metrics
        )
        repair_presentations = sum(
            int(value["image_presentations"]) for value in repair_metrics
        )
        evaluation = evaluations[name]
        evaluation_seconds = math.fsum(
            row.evaluation_seconds
            for row in evaluation.stages
            if row.score_mode == "raw"
        )
        peak_vram = max(
            [
                int(value.get("peak_vram_bytes", 0))
                for value in (*leaf_metrics, *baseline_metrics, *merge_metrics, *repair_metrics)
            ]
            or [0]
        )
        proxy_forwards = sum(
            int(value.get("proxy_image_count", 0))
            for value in merge_metrics
            if value.get("merge_method") == "output_drift"
        )
        training_wall = math.fsum(
            float(value.get("wall_seconds", 0.0)) for value in leaf_metrics
        ) + math.fsum(float(value.get("wall_seconds", 0.0)) for value in baseline_metrics)
        consolidation_wall = math.fsum(
            float(value.get("merge_wall_seconds", 0.0)) for value in merge_metrics
        )
        repair_wall = math.fsum(
            float(value.get("repair_wall_seconds", 0.0)) for value in repair_metrics
        )
        records[name] = {
            "archived_lora_parameters": len(unique_nodes) * per_lora,
            "archived_classifier_parameters": sum(
                len(bundle.artifact.represented_class_ids) * 769
                for bundle in unique_nodes.values()
            ),
            "average_candidate_forwards": math.fsum(
                len(snapshot.node_hashes) for snapshot in tree.snapshots
            )
            / len(tree.snapshots),
            "average_live_nodes": math.fsum(
                len(snapshot.node_hashes) for snapshot in tree.snapshots
            )
            / len(tree.snapshots),
            "final_candidate_forwards": len(final_nodes),
            "final_live_nodes": len(final_nodes),
            "final_live_parameter_bytes_fp32": len(final_nodes) * per_lora * 4
            + sum(
                len(bundle.artifact.represented_class_ids) * 769 * 4
                for bundle in final_nodes
            ),
            "evaluation_projection_seconds": evaluation_seconds,
            "historical_images_revisited_with_gradient": (
                parent_training_presentations + repair_presentations
            ),
            "leaf_training_image_presentations": leaf_presentations,
            "leaf_training_wall_seconds": math.fsum(
                float(value.get("wall_seconds", 0.0)) for value in leaf_metrics
            ),
            "baseline_training_wall_seconds": math.fsum(
                float(value.get("wall_seconds", 0.0)) for value in baseline_metrics
            ),
            "frozen_router_forwards_per_query": 1,
            "live_affine_classifier_parameters": sum(
                len(bundle.artifact.represented_class_ids) * 769 for bundle in final_nodes
            ),
            "live_calibration_parameters": 2 * len(final_nodes),
            "live_lora_parameters": len(final_nodes) * per_lora,
            "live_proxy_images": sum(len(bundle.artifact.proxy_image_ids) for bundle in final_nodes),
            "live_repair_images": sum(len(bundle.artifact.repair_image_ids) for bundle in final_nodes),
            "optimizer_steps": (
                int(baseline_metrics[-1]["optimizer_steps"])
                if baseline_metrics
                else sum(
                bundle.artifact.training_optimizer_steps
                for bundle in unique_nodes.values()
                )
            ),
            "parent_training_image_presentations": parent_training_presentations,
            "peak_vram_bytes": peak_vram,
            "proxy_forward_presentations": proxy_forwards,
            "proxy_images": proxy_forwards,
            "consolidation_wall_seconds": consolidation_wall,
            "repair_image_presentations": repair_presentations,
            "repair_wall_seconds": repair_wall,
            "train_images_per_second": (
                leaf_presentations / training_wall if training_wall > 0.0 else None
            ),
            "training_image_presentations": leaf_presentations
            + parent_training_presentations,
        }
    return {
        "conditions": records,
        "schema_version": "imagenetr50-resource-metrics-v1",
    }


def render_lineage_svg(path: Path) -> None:
    """Render all 50 leaves and 42 deterministic merges as a readable lineage SVG."""
    events, _snapshots = simulate_topology(50)
    width, height = 1500, 560
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:system-ui,sans-serif;font-size:10px;fill:#172033}'
        '.edge{stroke:#8090a5;stroke-width:1}.node{fill:#e8f1ff;stroke:#245a9a;stroke-width:1}</style>',
    ]
    positions: dict[str, tuple[float, float]] = {}
    for task in range(50):
        node_id = next(
            node.node_id
            for snapshot in simulate_topology(task + 1)[1][-1:]
            for node in snapshot.live_nodes
            if node.level == 0 and node.first_task == task
        )
        positions[node_id] = (25 + task * 29, 520)
    for event in events:
        left = positions[event.left.node_id]
        right = positions[event.right.node_id]
        parent = ((left[0] + right[0]) / 2, 520 - 95 * event.parent.level)
        positions[event.parent.node_id] = parent
        elements.extend(
            (
                f'<line class="edge" x1="{left[0]}" y1="{left[1]}" x2="{parent[0]}" y2="{parent[1]}"/>',
                f'<line class="edge" x1="{right[0]}" y1="{right[1]}" x2="{parent[0]}" y2="{parent[1]}"/>',
            )
        )
    logical = {
        node.node_id: node
        for event in events
        for node in (event.left, event.right, event.parent)
    }
    for task in range(50):
        from apm.continual.vision.imagenetr.bank import LogicalNode

        node = LogicalNode(0, task, task)
        logical[node.node_id] = node
    for node_id, (x, y) in positions.items():
        node = logical[node_id]
        elements.extend(
            (
                f'<circle class="node" cx="{x}" cy="{y}" r="7"/>',
                f'<text x="{x}" y="{y - 10}" text-anchor="middle">{escape(node.one_based_interval)}</text>',
            )
        )
    elements.append("</svg>")
    atomic_write(path, "".join(elements).encode("utf-8"))


def _plot_png(path: Path, draw: object) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    draw(axis)
    axis.grid(alpha=0.22)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plots(
    reports: Path,
    summaries: Sequence[Mapping[str, object]],
    evaluations: Mapping[str, EvaluationResult],
    merge_rows: Sequence[Mapping[str, object]],
    resources: Mapping[str, object],
) -> tuple[Path, ...]:
    paths: list[Path] = []
    primary = "affine_calibrated"

    def add(name: str, draw: object) -> None:
        path = reports / f"{name}.png"
        _plot_png(path, draw)
        paths.append(path)

    add(
        "01_accuracy_vs_task",
        lambda axis: (
            [
                axis.plot(
                    [row.stage for row in result.stages if row.score_mode == primary],
                    [row.accuracy for row in result.stages if row.score_mode == primary],
                    label=name,
                )
                for name, result in sorted(evaluations.items())
            ],
            axis.set(xlabel="Task", ylabel="Accuracy (%)", title="Class-incremental accuracy"),
            axis.legend(fontsize=7),
        ),
    )
    names = [str(row["condition"]) for row in summaries]
    finals = [float(row.get(f"{primary}_last_accuracy", math.nan)) for row in summaries]
    averages = [float(row.get(f"{primary}_incremental_accuracy", math.nan)) for row in summaries]
    forgettings = [float(row.get(f"{primary}_mean_forgetting", math.nan)) for row in summaries]
    bar = lambda values, title, ylabel: lambda axis: (
        axis.bar(range(len(names)), values, color="#3977b8"),
        axis.set_xticks(range(len(names)), names, rotation=55, ha="right", fontsize=7),
        axis.set(title=title, ylabel=ylabel),
    )
    add("02_final_accuracy", bar(finals, "Final accuracy by method", "Accuracy (%)"))
    add("03_incremental_average", bar(averages, "Incremental average accuracy", "Accuracy (%)"))
    add("04_mean_forgetting", bar(forgettings, "Mean forgetting", "Points"))
    joint = next((value for name, value in zip(names, finals) if "joint" in name), math.nan)
    add("05_joint_iid_gap", bar([joint - value for value in finals], "Joint-IID gap", "Points"))
    oracle = [float(row.get("true_node_oracle_last_accuracy", math.nan)) for row in summaries]
    add(
        "06_taskfree_vs_oracle",
        lambda axis: (
            axis.scatter(finals, oracle, color="#9a3d45"),
            [axis.annotate(name, (x, y), fontsize=6) for name, x, y in zip(names, finals, oracle)],
            axis.set(xlabel="Task-free final (%)", ylabel="True-node oracle (%)", title="Addressing gap"),
        ),
    )
    add("07_leaf_vs_logt_gap", bar([max(finals, default=math.nan) - value for value in finals], "All-leaf / best gap", "Points"))
    add("08_cheap_vs_retrain", bar(finals, "Cheap merge versus union retraining", "Accuracy (%)"))
    add("09_merge_family", bar(finals, "SVD, Core+TSV, and output drift", "Accuracy (%)"))
    add("10_repair_fraction", bar(finals, "Repair fraction comparison", "Accuracy (%)"))
    levels = [int(row.get("level", 0)) for row in merge_rows]
    errors = [float(row.get("relative_parameter_error", math.nan)) for row in merge_rows]
    energies = [float(row.get("retained_parameter_energy", math.nan)) for row in merge_rows]
    output_energies = [float(row.get("retained_output_energy", math.nan)) for row in merge_rows]
    scatter = lambda x, y, title, xlabel, ylabel: lambda axis: (
        axis.scatter(x, y, s=8, alpha=0.45),
        axis.set(title=title, xlabel=xlabel, ylabel=ylabel),
    )
    add("11_merge_damage_vs_level", scatter(levels, errors, "Spectral merge damage by level", "Level", "Relative error"))
    add("12_damage_vs_parameter_energy", scatter(energies, errors, "Damage versus retained parameter energy", "Retained energy", "Relative error"))
    add("13_damage_vs_output_energy", scatter(output_energies, errors, "Damage versus output-drift energy", "Retained output energy", "Relative error"))
    final_result = next(iter(evaluations.values()))
    live_by_stage = [
        next(row.live_nodes for row in final_result.stages if row.stage == stage)
        for stage in range(1, max(row.stage for row in final_result.stages) + 1)
    ]
    add("14_live_nodes", lambda axis: (axis.plot(range(1, len(live_by_stage) + 1), live_by_stage), axis.set(title="Live node count", xlabel="Task", ylabel="Nodes")))
    condition_resources = resources["conditions"]
    gradients = [
        condition_resources.get(name, {}).get(
            "historical_images_revisited_with_gradient", 0
        )
        for name in names
    ]
    memories = [condition_resources.get(name, {}).get("live_lora_parameters", 0) for name in names]
    forwards = [condition_resources.get(name, {}).get("final_candidate_forwards", 0) for name in names]
    add("15_accuracy_vs_gradient_history", scatter(gradients, finals, "Accuracy versus historical gradient images", "Historical images revisited", "Final accuracy (%)"))
    add("16_accuracy_vs_live_memory", scatter(memories, finals, "Accuracy versus live LoRA memory", "Live LoRA parameters", "Final accuracy (%)"))
    add("17_addressing_cost_vs_accuracy", scatter(forwards, finals, "Addressing cost versus accuracy", "Candidate forwards", "Final accuracy (%)"))
    return tuple(paths)


def write_report(
    store: VisionStore,
    config: ImageNetRConfig,
    protocol: ResolvedProtocol,
    leaves: Sequence[NodeBundle],
    trees: Mapping[str, TreeBuildResult],
    evaluations: Mapping[str, EvaluationResult],
    preflight: PreflightResult,
    external: E2LoRAResult,
    reuse_records: Sequence[Mapping[str, object]],
    scheduler: LocalScheduler,
) -> tuple[Path, Path]:
    """Emit every required ledger plus deterministic Markdown and self-contained HTML."""
    reports = store.run / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    leaf_rows = tuple(leaf.artifact.as_record() for leaf in leaves)
    all_bundles = {
        bundle.artifact.content_hash: bundle
        for tree in trees.values()
        for bundle in tree.nodes
    }
    all_bundles.update({leaf.artifact.content_hash: leaf for leaf in leaves})
    _write_jsonl(reports / "leaf_manifest.jsonl", leaf_rows)
    _write_jsonl(
        reports / "node_manifest.jsonl",
        tuple(bundle.artifact.as_record() for _, bundle in sorted(all_bundles.items())),
    )
    _write_jsonl(reports / "job_manifest.jsonl", scheduler.job_manifest_rows())
    stage_rows = tuple(
        row.as_record() for result in evaluations.values() for row in result.stages
    )
    task_rows = tuple(
        row.as_record() for result in evaluations.values() for row in result.tasks
    )
    _write_csv(reports / "stage_accuracy.csv", stage_rows)
    _write_csv(reports / "task_accuracy_matrix.csv", task_rows)
    merge_rows = _merge_rows(store)
    routing_rows = _routing_rows(evaluations)
    try:
        import pandas as pd
    except ImportError as error:  # pragma: no cover - vision environment gate
        raise RuntimeError("pandas/pyarrow are required by the vision environment") from error
    pd.DataFrame(merge_rows).to_parquet(reports / "merge_diagnostics.parquet", index=False)
    pd.DataFrame(routing_rows).to_parquet(reports / "routing_diagnostics.parquet", index=False)
    summaries = tuple(_condition_summary(result) for result in evaluations.values())
    resources = _resource_records(store, trees, evaluations, config)
    publish_immutable_json(reports / "resource_metrics.json", resources)
    summary = {
        "conditions": list(summaries),
        "external_e2lora": external.as_record(),
        "preflight": preflight.as_record(),
        "protocol_hash": protocol.content_hash,
        "reuse_demonstrations": [dict(value) for value in reuse_records],
        "schema_version": "imagenetr50-summary-v1",
    }
    publish_immutable_json(reports / "summary.json", summary)
    render_lineage_svg(reports / "lineage.svg")
    plot_paths = _plots(reports, summaries, evaluations, merge_rows, resources)

    header = (
        "| Condition | Last affine | Inc. affine | Forgetting | Oracle last | Routing gap | Centroid last | Centroid route |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|"
    )
    rows = []
    for value in summaries:
        affine = float(value["affine_calibrated_last_accuracy"])
        oracle_value = float(value["true_node_oracle_last_accuracy"])
        rows.append(
            "| {condition} | {last:.3f} | {inc:.3f} | {forget:.3f} | {oracle:.3f} | {gap:.3f} | {centroid:.3f} | {route:.3f} |".format(
                condition=value["condition"],
                last=affine,
                inc=float(value["affine_calibrated_incremental_accuracy"]),
                forget=float(value["affine_calibrated_mean_forgetting"]),
                oracle=oracle_value,
                gap=oracle_value - affine,
                centroid=float(value["centroid_router_last_accuracy"]),
                route=next(
                    row.routing_accuracy
                    for row in evaluations[str(value["condition"])].stages
                    if row.stage == max(item.stage for item in evaluations[str(value["condition"])].stages)
                    and row.score_mode == "centroid_router"
                ),
            )
        )
    reuse_text = "\n".join(
        f"- leaf hashes unchanged: `{record.get('leaf_hashes_unchanged')}`; "
        f"leaf optimizer steps: `{record.get('leaf_optimizer_steps')}`; "
        f"new gradient work: `{record.get('new_gradient_work')}`"
        for record in reuse_records
    )
    plots_markdown = "\n".join(f"![{path.stem}]({path.name})" for path in plot_paths)
    markdown = f"""# ImageNet-R-50 Log-t VAMP Report

Protocol identity: `{protocol.content_hash}`. This report uses one immutable local
24,000/6,000 split for every internal condition and the local E2-LoRA reproduction.
Published E2-LoRA values are external context and are not treated as a pass threshold.

## Protocol and environment

- Backbone: `{config.model_name}`, rank/alpha `{config.lora_rank}/{config.lora_alpha}`.
- Seed: `{config.seed}`; 50 tasks of four classes; no test identities in gradient,
  proxy, repair, or calibration inputs.
- GPU preflight: `{preflight.device_name}`, BF16 `{preflight.bf16_supported}`,
  peak `{preflight.peak_vram_bytes / 2**30:.2f}` GiB.

## Primary results

{header}
{chr(10).join(rows)}

## External E2-LoRA reference

Local reproduction succeeded: `{external.succeeded}`. Local LastAcc:
`{external.final_accuracy}`; local IncAcc: `{external.incremental_average_accuracy}`.
Published context remains LastAcc `78.58`, IncAcc `83.96`. Failure record, if any:
`{external.failure}`.

## Artifact reuse

{reuse_text}

## Causal diagnosis

Interpret the all-leaf true-task oracle first, then all-leaf task-free routing,
union-retrained true-node performance, cheap-merge true-node performance, repair
closure, and finally the remaining task-free routing gap. Negative outcomes are
retained in the tables and are not hidden by selecting on test accuracy.

## Compute, memory, and addressing

See `resource_metrics.json` for separate leaf, parent-retraining, repair-gradient,
forward-only proxy, experimental-archive, and live-deployment accounting.

## Lineage

![Complete deterministic lineage](lineage.svg)

## Plots

{plots_markdown}

## Conclusions and next experiments

The primary tables determine whether logarithmic consolidation works, how close it
comes to joint IID, whether output-drift beats weight-space compression, how much
repair is needed, and how much residual error is routing. Scale/proxy/rank/CtM
sweeps remain secondary until this complete matrix is interpreted.
"""
    markdown_path = atomic_write(reports / "REPORT.md", markdown.encode("utf-8"))
    embedded = []
    for path in plot_paths:
        embedded.append(
            f'<details><summary>{escape(path.stem)}</summary><img alt="{escape(path.stem)}" '
            f'src="data:image/png;base64,{b64encode(path.read_bytes()).decode()}"/></details>'
        )
    lineage = (reports / "lineage.svg").read_text(encoding="utf-8")
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>ImageNet-R-50 Log-t VAMP</title>
<style>body{{font:16px/1.5 system-ui,sans-serif;max-width:1180px;margin:auto;padding:2rem;color:#172033}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #b7c2d0;padding:.45rem;text-align:right}}
th:first-child,td:first-child{{text-align:left}}details{{margin:1rem 0}}img,svg{{max-width:100%;height:auto}}
code{{background:#eef2f7;padding:.12rem .25rem}}</style></head><body>
<h1>ImageNet-R-50 Log-t VAMP Report</h1><p>Protocol <code>{protocol.content_hash}</code>.</p>
<h2>Primary results</h2><table><thead><tr><th>Condition</th><th>Last affine</th><th>Inc. affine</th>
<th>Forgetting</th><th>Oracle last</th><th>Routing gap</th></tr></thead><tbody>
{''.join('<tr><td>'+escape(str(v['condition']))+'</td><td>'+f"{float(v['affine_calibrated_last_accuracy']):.3f}"+'</td><td>'+f"{float(v['affine_calibrated_incremental_accuracy']):.3f}"+'</td><td>'+f"{float(v['affine_calibrated_mean_forgetting']):.3f}"+'</td><td>'+f"{float(v['true_node_oracle_last_accuracy']):.3f}"+'</td><td>'+f"{float(v['true_node_oracle_last_accuracy'])-float(v['affine_calibrated_last_accuracy']):.3f}"+'</td></tr>' for v in summaries)}
</tbody></table><h2>External reference</h2><p>Local E2-LoRA success: {external.succeeded}; LastAcc:
{external.final_accuracy}; IncAcc: {external.incremental_average_accuracy}. Published context: 78.58 / 83.96.</p>
<h2>Artifact reuse</h2><pre>{escape(reuse_text)}</pre><h2>Lineage</h2>{lineage}<h2>Plots</h2>
{''.join(embedded)}<h2>Causal interpretation</h2><p>Read leaf quality, task-free routing, full union retraining,
cheap merging, bounded repair, and residual routing regret in that order. All negative results remain visible.</p>
</body></html>"""
    html_path = atomic_write(reports / "REPORT.html", html.encode("utf-8"))
    return markdown_path, html_path


__all__ = ["render_lineage_svg", "write_report"]

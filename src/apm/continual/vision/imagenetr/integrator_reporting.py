"""Markdown, standalone HTML, tables, and plots for the integrator workflow."""

from __future__ import annotations

from base64 import b64encode
from collections.abc import Mapping, Sequence
from html import escape
from pathlib import Path
import csv
import json

from apm.continual.artifacts import (
    ChainedJsonlLedger,
    atomic_write,
    canonical_json_bytes,
    file_sha256,
)
from apm.continual.vision.imagenetr.integrator_bank import simulate_binary_topology


def _load(path: Path) -> dict[str, object] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = tuple(sorted({key for row in rows for key in row}))
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_tables(
    report_root: Path, locked: Mapping[str, object] | None
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    stage_rows = [] if locked is None else [dict(row) for row in locked.get("stage_metrics", ())]
    task_rows = [] if locked is None else [dict(row) for row in locked.get("task_accuracy_matrix", ())]
    flattened = [
        {
            "accuracy": row.get("accuracy"),
            "base_head_union": dict(row.get("controls", {})).get("base_head_union"),
            "cosine_union": dict(row.get("controls", {})).get("cosine_union"),
            "live_nodes": row.get("live_nodes"),
            "local_log_probability_union": dict(row.get("controls", {})).get(
                "local_log_probability_union"
            ),
            "raw_union": dict(row.get("controls", {})).get("raw_union"),
            "stage": row.get("stage"),
            "true_node_oracle": dict(row.get("controls", {})).get("true_node_oracle"),
        }
        for row in stage_rows
    ]
    _write_csv(report_root / "stage_metrics.csv", flattened)
    _write_csv(report_root / "task_accuracy_matrix.csv", task_rows)
    atomic_write(report_root / "stage_metrics.json", canonical_json_bytes({"rows": flattened}))
    atomic_write(report_root / "task_accuracy_matrix.json", canonical_json_bytes({"rows": task_rows}))
    if flattened or task_rows:
        try:
            import pandas as pd

            if flattened:
                pd.DataFrame(flattened).to_parquet(report_root / "stage_metrics.parquet", index=False)
            if task_rows:
                pd.DataFrame(task_rows).to_parquet(
                    report_root / "task_accuracy_matrix.parquet", index=False
                )
        except ImportError:  # pragma: no cover - environment preflight requires pandas
            pass
    return flattened, task_rows


def _clean_capacity_rows(
    clean: Mapping[str, object] | None,
) -> tuple[tuple[int, ...], list[dict[str, object]]]:
    """Project the clean hierarchy-capacity gate into one row per checkpoint."""
    if clean is None:
        return (), []
    bounded = dict(clean.get("bounded_controls", {}))
    full = tuple(dict(row) for row in clean.get("full_controls", ()))
    if not bounded or not full:
        return (), []
    capacities = tuple(sorted((int(value) for value in bounded), key=int))
    bounded_by_capacity = {
        capacity: {
            int(dict(row)["stage"]): dict(row)
            for row in bounded[str(capacity)]
        }
        for capacity in capacities
    }
    rows: list[dict[str, object]] = []
    for full_row in sorted(full, key=lambda row: int(row["stage"])):
        stage = int(full_row["stage"])
        full_accuracy = float(dict(full_row["controls"])["true_node_oracle"])
        projected: dict[str, object] = {
            "full_union_oracle_accuracy": full_accuracy,
            "stage": stage,
        }
        for capacity in capacities:
            bounded_accuracy = float(
                dict(bounded_by_capacity[capacity][stage]["controls"])[
                    "true_node_oracle"
                ]
            )
            projected[f"k{capacity}_oracle_accuracy"] = bounded_accuracy
            projected[f"k{capacity}_minus_full_pp"] = bounded_accuracy - full_accuracy
        rows.append(projected)
    return capacities, rows


def _write_clean_capacity_tables(
    report_root: Path,
    capacities: Sequence[int],
    rows: Sequence[Mapping[str, object]],
) -> None:
    """Write the clean gate evidence in CSV, JSON, and Parquet forms."""
    _write_csv(report_root / "clean_capacity_gate.csv", rows)
    atomic_write(
        report_root / "clean_capacity_gate.json",
        canonical_json_bytes({"capacities": list(capacities), "rows": list(rows)}),
    )
    if rows:
        try:
            import pandas as pd

            pd.DataFrame(rows).to_parquet(
                report_root / "clean_capacity_gate.parquet", index=False
            )
        except ImportError:  # pragma: no cover - environment preflight requires pandas
            pass


def _resource_rows(run: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for policy_path in sorted((run / "hierarchies").glob("*/policy.json")):
        policy_root = policy_path.parent
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        node_paths = tuple(sorted((policy_root / "nodes").glob("*/node.json")))
        if not node_paths:
            continue
        nodes = tuple(json.loads(path.read_text(encoding="utf-8")) for path in node_paths)
        metrics = tuple(
            json.loads((path.parent / "training_metrics.json").read_text(encoding="utf-8"))
            for path in node_paths
        )
        snapshots = tuple(sorted((policy_root / "snapshots").glob("stage_*.json")))
        final = json.loads(snapshots[-1].read_text(encoding="utf-8"))
        by_logical = {
            json.loads((path.parent / "logical.json").read_text(encoding="utf-8"))[
                "logical_node_id"
            ]: path.parent
            for path in node_paths
        }
        live = tuple(by_logical[identity] for identity in final["logical_node_ids"])
        live_hashes = {
            json.loads((path / "node.json").read_text(encoding="utf-8"))["content_hash"]
            for path in live
        }
        rows.append(
            {
                "cache_bytes": None,
                "cache_hits": None,
                "cache_misses": None,
                "classifier_union_scalar_copies": sum(
                    769 * len(node["represented_class_ids"])
                    for node in nodes
                    if node["parent_hashes"]
                ),
                "condition": f"hierarchy:{policy['replay_mode']}:{policy['partition']}:{policy['reservoir_capacity']}",
                "integrator_backward_example_passes": None,
                "integrator_forward_example_passes": None,
                "hierarchy_training_image_presentations": sum(
                    int(metric.get("image_presentations", 0)) for metric in metrics
                ),
                "live_adapter_bytes": sum((path / "adapter.safetensors").stat().st_size for path in live),
                "live_classifier_bytes": sum(
                    (path / "classifier.safetensors").stat().st_size for path in live
                ),
                "live_nodes": len(live),
                "node_example_forwards": None,
                "optimizer_steps": sum(int(node["training_optimizer_steps"]) for node in nodes),
                "peak_vram_bytes": max(
                    int(metric.get("peak_vram_bytes", 0)) for metric in metrics
                ),
                "replay_identity_bytes": sum(
                    64 * len(node["proxy_image_ids"])
                    for node in nodes
                    if node["content_hash"] in live_hashes
                ),
            }
        )
    for ledger_path in sorted(
        (run / "integrators" / "persistent").glob("*/training_metrics.jsonl")
    ):
        metrics = ChainedJsonlLedger(
            ledger_path, "imagenetr50-integrator-stage-training-v1"
        ).rows
        if metrics:
            evaluation_rows = tuple(
                row
                for evaluation_path in sorted(
                    ledger_path.parent.glob("*_metrics.jsonl")
                )
                if evaluation_path.name != ledger_path.name
                for row in ChainedJsonlLedger(
                    evaluation_path,
                    "imagenetr50-integrator-stage-evaluation-v1",
                ).rows
            )
            rows.append(
                {
                    "cache_bytes": None,
                    "cache_hits": None,
                    "cache_misses": None,
                    "classifier_union_scalar_copies": None,
                    "condition": f"integrator:{ledger_path.parent.name}",
                    "integrator_backward_example_passes": sum(
                        int(
                            dict(row.get("training_fit") or {}).get(
                                "training_backward_example_passes", 0
                            )
                        )
                        for row in metrics
                    ),
                    "integrator_forward_example_passes": sum(
                        int(
                            dict(row.get("training_fit") or {}).get(
                                "training_forward_example_passes", 0
                            )
                        )
                        for row in metrics
                    ),
                    "hierarchy_training_image_presentations": None,
                    "live_adapter_bytes": None,
                    "live_classifier_bytes": None,
                    "live_nodes": int(metrics[-1]["live_nodes"]),
                    "node_example_forward_bound": sum(
                        int(row["node_example_forwards_bound"]) for row in metrics
                    ),
                    "node_example_forwards": None,
                    "optimizer_steps": int(metrics[-1]["integrator_optimizer_steps"]),
                    "peak_vram_bytes": max(
                        int(
                            dict(row.get("training_fit") or {}).get(
                                "peak_vram_bytes", 0
                            )
                        )
                        for row in metrics
                    ),
                    "replay_identity_bytes": 64 * int(metrics[-1]["historical_capacity"]),
                    "evaluation_forward_example_passes": sum(
                        int(row["evaluation_examples"]) for row in evaluation_rows
                    ),
                }
            )
    for fit_path in sorted((run / "integrators").glob("*/*/fit.json")):
        record = json.loads(fit_path.read_text(encoding="utf-8"))
        fit = dict(record["fit"])
        rows.append(
            {
                "cache_bytes": None,
                "cache_hits": None,
                "cache_misses": None,
                "classifier_union_scalar_copies": None,
                "condition": f"fresh:{fit_path.parent.parent.name}:{fit_path.parent.name}",
                "integrator_backward_example_passes": int(
                    fit["training_backward_example_passes"]
                ),
                "integrator_forward_example_passes": int(
                    fit["training_forward_example_passes"]
                )
                + int(fit["validation_forward_example_passes"]),
                "hierarchy_training_image_presentations": None,
                "live_adapter_bytes": None,
                "live_classifier_bytes": None,
                "live_nodes": None,
                "node_example_forwards": None,
                "optimizer_steps": int(fit["optimizer_steps"]),
                "peak_vram_bytes": int(fit["peak_vram_bytes"]),
                "replay_identity_bytes": None,
            }
        )
    behavior_ledger = run / "ledgers" / "behavior_requests.jsonl"
    behavior_rows = (
        ChainedJsonlLedger(
            behavior_ledger, "imagenetr50-integrator-behavior-request-v1"
        ).rows
        if behavior_ledger.is_file()
        else ()
    )
    cache_manifests = tuple(
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((run / "cache" / "behaviors" / "row_shards").glob("*/*/cache.json"))
    )
    if behavior_rows or cache_manifests:
        rows.append(
            {
                "base_example_forwards": sum(
                    len(record["image_ids"])
                    for record in cache_manifests
                    if "node_hash" not in record["semantic_values"]
                ),
                "cache_bytes": None,
                "cache_hits": sum(int(row["cache_hits"]) for row in behavior_rows),
                "cache_misses": sum(int(row["cache_misses"]) for row in behavior_rows),
                "classifier_union_scalar_copies": None,
                "condition": "all_behavior_requests",
                "integrator_backward_example_passes": None,
                "integrator_forward_example_passes": None,
                "hierarchy_training_image_presentations": None,
                "live_adapter_bytes": None,
                "live_classifier_bytes": None,
                "live_nodes": None,
                "node_example_forwards": sum(
                    len(record["image_ids"])
                    for record in cache_manifests
                    if "node_hash" in record["semantic_values"]
                ),
                "optimizer_steps": None,
                "peak_vram_bytes": None,
                "replay_identity_bytes": None,
            }
        )
    cache_bytes = sum(
        path.stat().st_size
        for path in (run / "cache").rglob("*")
        if path.is_file()
    )
    rows.append(
        {
            "cache_bytes": cache_bytes,
            "cache_hits": None,
            "cache_misses": None,
            "classifier_union_scalar_copies": None,
            "condition": "shared_behavior_cache",
            "integrator_backward_example_passes": None,
            "integrator_forward_example_passes": None,
            "hierarchy_training_image_presentations": None,
            "live_adapter_bytes": None,
            "live_classifier_bytes": None,
            "live_nodes": None,
            "node_example_forwards": None,
            "optimizer_steps": None,
            "peak_vram_bytes": None,
            "replay_identity_bytes": None,
        }
    )
    return rows


def _write_resource_tables(report_root: Path, rows: Sequence[Mapping[str, object]]) -> None:
    _write_csv(report_root / "resource_accounting.csv", rows)
    atomic_write(report_root / "resource_accounting.json", canonical_json_bytes({"rows": list(rows)}))
    try:
        import pandas as pd

        pd.DataFrame(rows).to_parquet(report_root / "resource_accounting.parquet", index=False)
    except ImportError:  # pragma: no cover - environment preflight requires pandas
        pass


def _lineage_plot(path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    events, _snapshots = simulate_binary_topology(50)
    fig, axis = plt.subplots(figsize=(14, 7))
    for event in events:
        parent_x = (event.parent.first_task + event.parent.last_task + 2) / 2
        parent_y = event.parent.level
        for child in (event.left, event.right):
            child_x = (child.first_task + child.last_task + 2) / 2
            axis.plot((child_x, parent_x), (child.level, parent_y), color="#78909c", linewidth=0.8)
    nodes = {
        (node.level, node.first_task, node.last_task): node
        for event in events
        for node in (event.left, event.right, event.parent)
    }
    axis.scatter(
        [(node.first_task + node.last_task + 2) / 2 for node in nodes.values()],
        [node.level for node in nodes.values()],
        s=12,
        color="#1565c0",
        zorder=3,
    )
    axis.set(title="Capacity-one ImageNet-R hierarchy", xlabel="Task interval midpoint", ylabel="Level")
    axis.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _accuracy_plot(
    path: Path,
    diagnostic: Mapping[str, object] | None,
    locked_rows: Sequence[Mapping[str, object]],
    capacities: Sequence[int],
    capacity_rows: Sequence[Mapping[str, object]],
    hierarchy_tolerance: float | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    if diagnostic:
        feature = dict(diagnostic.get("feature_accuracies", {}))
        controls = {
            name: value
            for name, value in dict(diagnostic.get("controls", {})).items()
            if name != "true_node_oracle"
        }
        labels = list(feature) + ["best static"]
        values = [float(feature[label]) for label in feature] + ([max(controls.values())] if controls else [0.0])
        axes[0].bar(labels, values, color=["#1565c0"] * len(feature) + ["#9e9e9e"])
        axes[0].set_ylim(0, 100)
        axes[0].tick_params(axis="x", rotation=25)
        axes[0].set(title="Sealed diagnostic", ylabel="Validation accuracy (%)")
    else:
        axes[0].text(0.5, 0.5, "Diagnostic not complete", ha="center", va="center")
        axes[0].set_axis_off()
    if locked_rows:
        stages = [int(row["stage"]) for row in locked_rows]
        axes[1].plot(stages, [float(row["accuracy"]) for row in locked_rows], label="bounded integrator")
        for key, label in (
            ("raw_union", "raw union"),
            ("true_node_oracle", "true-node oracle"),
        ):
            axes[1].plot(
                stages,
                [float(row[key]) for row in locked_rows],
                label=label,
                linestyle="--",
            )
        axes[1].set(xlabel="Tasks seen", ylabel="Test accuracy (%)", title="Locked local benchmark")
        axes[1].set_ylim(0, 100)
        axes[1].grid(alpha=0.2)
        axes[1].legend()
    elif capacity_rows:
        stages = [int(row["stage"]) for row in capacity_rows]
        full = [float(row["full_union_oracle_accuracy"]) for row in capacity_rows]
        axes[1].plot(stages, full, color="#263238", marker="o", label="full-union oracle")
        if hierarchy_tolerance is not None:
            axes[1].plot(
                stages,
                [value - hierarchy_tolerance for value in full],
                color="#78909c",
                linestyle=":",
                label=f"full minus {hierarchy_tolerance:g} pp",
            )
        colors = ("#1565c0", "#ef6c00", "#2e7d32", "#6a1b9a")
        for index, capacity in enumerate(capacities):
            axes[1].plot(
                stages,
                [float(row[f"k{capacity}_oracle_accuracy"]) for row in capacity_rows],
                color=colors[index % len(colors)],
                marker="s",
                label=f"bounded K={capacity}",
            )
        axes[1].set(
            xlabel="Tasks seen",
            ylabel="Validation accuracy (%)",
            title="Clean hierarchy capacity gate",
            xticks=stages,
        )
        axes[1].grid(alpha=0.2)
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "Locked test not opened", ha="center", va="center")
        axes[1].set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    if not rows:
        return "_No completed rows._\n"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def _html_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    if not rows:
        return "<p><em>No completed rows.</em></p>"
    head = "".join(f"<th>{escape(header)}</th>" for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _image_data(path: Path) -> str:
    return "data:image/png;base64," + b64encode(path.read_bytes()).decode("ascii")


def write_integrator_report(run_root: str | Path) -> Path:
    """Regenerate human-readable reports and machine-readable result projections."""
    run = Path(run_root)
    reports = run / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    state = _load(run / "state" / "workflow.json") or {}
    diagnostic = _load(run / "diagnostic" / "result.json")
    clean = _load(run / "evaluations" / "clean_development.json")
    development = _load(run / "evaluations" / "development_task50.json")
    locked = _load(run / "evaluations" / "locked_test.json")
    stage_rows, _task_rows = _write_tables(reports, locked)
    capacities, capacity_rows = _clean_capacity_rows(clean)
    _write_clean_capacity_tables(reports, capacities, capacity_rows)
    resources = _resource_rows(run)
    _write_resource_tables(reports, resources)
    lineage = reports / "lineage.png"
    accuracy_plot = reports / "accuracy.png"
    _lineage_plot(lineage)
    tolerance_value = None if clean is None else clean.get("hierarchy_oracle_tolerance")
    hierarchy_tolerance = (
        float(tolerance_value) if isinstance(tolerance_value, (int, float)) else None
    )
    _accuracy_plot(
        accuracy_plot,
        diagnostic,
        stage_rows,
        capacities,
        capacity_rows,
        hierarchy_tolerance,
    )
    diagnostic_rows = []
    if diagnostic:
        diagnostic_rows = [
            (name, f"{float(value):.3f}")
            for name, value in dict(diagnostic.get("feature_accuracies", {})).items()
        ]
    final_rows = []
    if locked:
        references = dict(locked["local_references"])
        static = dict(dict(locked["final_static_controls"])["controls"])
        final_rows = [
            ("LogT bounded integrator — Last", f"{float(locked['last_accuracy']):.3f}"),
            ("LogT bounded integrator — Incremental", f"{float(locked['incremental_accuracy']):.3f}"),
            ("Bounded hierarchy — raw union", f"{float(static['raw_union']):.3f}"),
            ("Bounded hierarchy — cosine union", f"{float(static['cosine_union']):.3f}"),
            (
                "Bounded hierarchy — affine-calibrated union",
                f"{float(static['affine_calibrated_union']):.3f}",
            ),
            (
                "Bounded hierarchy — true-node oracle",
                f"{float(static['true_node_oracle']):.3f}",
            ),
            ("Frozen-reference control — Last", f"{float(references['frozen_reference_last']):.3f}"),
            ("Sequential LoRA control — Last", f"{float(references['sequential_last']):.3f}"),
            ("Joint-IID LoRA control — Last", f"{float(references['joint_iid_last']):.3f}"),
            ("Local E2-LoRA — Last", f"{float(locked['local_e2_last']):.3f}"),
            ("Local E2-LoRA — Incremental", f"{float(locked['local_e2_incremental']):.3f}"),
        ]
    selected = None if clean is None else {
        "feature_variant": clean.get("selected_variant"),
        "consolidation_capacity": clean.get("selected_consolidation_capacity"),
        "historical_capacity": clean.get("selected_historical_capacity"),
    }
    capacity_headers = ["stage", "full-union oracle"]
    for capacity in capacities:
        capacity_headers.extend((f"K={capacity}", f"K={capacity} - full (pp)"))
    capacity_table_rows = []
    for row in capacity_rows:
        values: list[object] = [row["stage"], f"{float(row['full_union_oracle_accuracy']):.3f}"]
        for capacity in capacities:
            values.extend(
                (
                    f"{float(row[f'k{capacity}_oracle_accuracy']):.3f}",
                    f"{float(row[f'k{capacity}_minus_full_pp']):+.3f}",
                )
            )
        capacity_table_rows.append(tuple(values))
    clean_note = ""
    if clean is not None:
        clean_note = f"Gate open: {clean.get('gate_open')}."
        if hierarchy_tolerance is not None:
            clean_note += (
                " Required every bounded-minus-full difference to be at least "
                f"-{hierarchy_tolerance:g} percentage points."
            )
        if clean.get("reason"):
            clean_note += f" Stop reason: {clean['reason']}."
        clean_note += "\n\n"
    resource_summary = [
        (
            row.get("condition"),
            "—" if row.get("node_example_forwards") is None else row["node_example_forwards"],
            (
                "—"
                if row.get("hierarchy_training_image_presentations") is None
                else row["hierarchy_training_image_presentations"]
            ),
            (
                "—"
                if row.get("integrator_backward_example_passes") is None
                else row["integrator_backward_example_passes"]
            ),
        )
        for row in resources
    ]
    markdown = (
        "# ImageNet-R-50 LogT Prediction Integrator\n\n"
        f"Run: `{run.name}`  \nWorkflow state: `{state.get('phase', 'NOT_STARTED')}`\n\n"
        "This experiment replaces task-free node selection with a direct 200-way residual "
        "integrator over frozen node behavior. The scalable condition uses a capacity-one "
        "binary counter, bounded consolidation replay, and bounded persistent integrator replay.\n\n"
        "## Sealed capacity diagnostic\n\n"
        + _markdown_table(("feature family", "mean validation accuracy"), diagnostic_rows)
        + (f"\nGate open: **{diagnostic.get('gate_open')}**. Selected: `{diagnostic.get('selected_variant')}`.\n" if diagnostic else "")
        + "\n## Frozen clean selection\n\n"
        + (f"`{json.dumps(selected, sort_keys=True)}`\n\n" if selected else "_Not complete._\n\n")
        + clean_note
        + _markdown_table(tuple(capacity_headers), capacity_table_rows)
        + "\n"
        + "Validation identities are excluded from every clean node and integrator update. "
        "Full-union training is an empirical ceiling and is excluded from the scalable claim.\n\n"
        + "## Locked local result\n\n"
        + _markdown_table(("condition", "accuracy (%)"), final_rows)
        + "\nPublished E2-LoRA values (78.58 Last / 83.96 Incremental) remain external context; "
        "the paired local E2-LoRA rerun is the direct comparator.\n\n"
        + "## Complexity boundary\n\n"
        + "Per arrival, the capacity-one hierarchy performs at most `bit_length(t)` carries and "
        "the persistent observer evaluates at most `popcount(t) * (current + H)` node/example "
        "pairs. Cumulative model work is therefore O(T log T). Classifier-row union arithmetic "
        "is reported separately because the 200-way output space is fixed in this benchmark.\n\n"
        + _markdown_table(
            (
                "condition",
                "node forwards",
                "hierarchy presentations",
                "integrator backward",
            ),
            resource_summary,
        )
        + "\nExact per-request cache/model work is retained in `resource_accounting.*`.\n\n"
        + "## Figures\n\n![Accuracy](accuracy.png)\n\n![Capacity-one lineage](lineage.png)\n"
    )
    atomic_write(reports / "REPORT.md", markdown.encode("utf-8"))
    diagnostic_html = _html_table(("feature family", "mean validation accuracy"), diagnostic_rows)
    capacity_html = _html_table(tuple(capacity_headers), capacity_table_rows)
    final_html = _html_table(("condition", "accuracy (%)"), final_rows)
    resource_html = _html_table(
        (
            "condition",
            "node forwards",
            "hierarchy presentations",
            "integrator backward",
        ),
        resource_summary,
    )
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ImageNet-R-50 LogT Prediction Integrator</title>
<style>body{{font:16px/1.55 system-ui,sans-serif;max-width:1200px;margin:auto;padding:2rem;color:#17202a;background:#fafafa}}h1,h2{{color:#123b5d}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccd5dd;padding:.45rem;text-align:left}}th{{background:#e8f1f8}}details{{background:white;border:1px solid #d9e1e7;border-radius:8px;padding:1rem;margin:1rem 0}}summary{{font-weight:700;cursor:pointer}}img{{max-width:100%;height:auto}}code{{overflow-wrap:anywhere}}</style></head>
<body><h1>ImageNet-R-50 LogT Prediction Integrator</h1><p><strong>Run:</strong> <code>{escape(run.name)}</code><br><strong>State:</strong> {escape(str(state.get('phase', 'NOT_STARTED')))}</p>
<p>A direct 200-way residual integrator observes frozen LogT node behavior. The scalable condition uses capacity-one consolidation and bounded replay; full-union training is a ceiling only.</p>
<details open><summary>Sealed capacity diagnostic</summary>{diagnostic_html}<p>Gate open: {escape(str(None if diagnostic is None else diagnostic.get('gate_open')))}; selected feature family: <code>{escape(str(None if diagnostic is None else diagnostic.get('selected_variant')))}</code>.</p></details>
<details open><summary>Clean selection</summary><p><code>{escape(json.dumps(selected, sort_keys=True))}</code></p><p>{escape(clean_note.strip())}</p>{capacity_html}<p>Validation identities are excluded from all clean trainable components.</p></details>
<details open><summary>Locked local benchmark</summary>{final_html}<p>The common-split local E2-LoRA rerun is the direct comparator. Published 78.58/83.96 values are external context.</p></details>
<details open><summary>Accuracy</summary><img alt="Diagnostic and locked accuracy plots" src="{_image_data(accuracy_plot)}"></details>
<details><summary>Capacity-one lineage</summary><img alt="Capacity-one binary-counter lineage" src="{_image_data(lineage)}"></details>
<details><summary>Complexity and resources</summary><p>Per arrival there are at most bit_length(t) carries and popcount(t) live nodes. Persistent observer work is bounded by popcount(t) × (current + H), giving O(T log T) cumulative model work. Output-head arithmetic is accounted separately.</p>{resource_html}<pre>{escape(json.dumps({'clean': clean, 'development': development}, indent=2, sort_keys=True))}</pre></details>
</body></html>"""
    atomic_write(reports / "REPORT.html", html.encode("utf-8"))
    files = tuple(
        {
            "path": path.name,
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(reports.iterdir())
        if path.is_file() and path.name != "report_manifest.json"
    )
    atomic_write(
        reports / "report_manifest.json",
        canonical_json_bytes(
            {"files": list(files), "schema_version": "imagenetr50-integrator-report-manifest-v1"}
        ),
    )
    return reports / "REPORT.html"


__all__ = ["write_integrator_report"]

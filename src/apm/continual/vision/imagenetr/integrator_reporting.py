"""Markdown, standalone HTML, tables, and plots for the integrator workflow."""

from __future__ import annotations

from base64 import b64encode
from collections.abc import Mapping, Sequence
from html import escape
from math import fsum, isclose, isfinite
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


def _joint_iid_stage_rows(
    run: Path, locked: Mapping[str, object] | None
) -> list[dict[str, object]]:
    """Load and authenticate the sealed run's complete raw joint-IID test curve."""
    if locked is None:
        return []
    config = _load(run / "config_resolved.json")
    if config is None or not config.get("sealed_run_hash"):
        return []
    sealed_run_hash = str(config["sealed_run_hash"])
    configured_root = Path(str(config.get("inference_artifact_root", "")))
    candidate_roots = (
        ([configured_root] if configured_root.is_absolute() else [Path.cwd() / configured_root])
        + list(run.parents)
    )
    source = next(
        (
            root / "runs" / sealed_run_hash / "reports" / "stage_accuracy.csv"
            for root in candidate_roots
            if (root / "runs" / sealed_run_hash / "reports" / "stage_accuracy.csv").is_file()
        ),
        None,
    )
    if source is None:
        return []
    with source.open(encoding="utf-8", newline="") as input_file:
        matching = tuple(
            row
            for row in csv.DictReader(input_file)
            if row["condition"] == "joint_iid_lora_r16"
            and row["score_mode"] == "raw"
            and row["diagnostic"].strip().lower() == "false"
        )
    rows = sorted(
        (
            {"accuracy": float(row["accuracy"]), "stage": int(row["stage"])}
            for row in matching
        ),
        key=lambda row: int(row["stage"]),
    )
    stages = [int(row["stage"]) for row in rows]
    if stages != list(range(1, 51)) or not all(
        isfinite(float(row["accuracy"])) for row in rows
    ):
        raise ValueError(
            "sealed joint-IID curve must contain one finite raw test value for every stage 1..50"
        )
    references = dict(locked.get("local_references", {}))
    expected_last = float(references["joint_iid_last"])
    expected_incremental = float(references["joint_iid_incremental"])
    actual_last = float(rows[-1]["accuracy"])
    actual_incremental = fsum(float(row["accuracy"]) for row in rows) / len(rows)
    if not isclose(actual_last, expected_last, abs_tol=1e-6) or not isclose(
        actual_incremental, expected_incremental, abs_tol=1e-6
    ):
        raise ValueError(
            "sealed joint-IID curve does not match the locked endpoint and incremental references"
        )
    return rows


def _write_tables(
    report_root: Path,
    locked: Mapping[str, object] | None,
    joint_iid_rows: Sequence[Mapping[str, object]] = (),
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    stage_rows = [] if locked is None else [dict(row) for row in locked.get("stage_metrics", ())]
    task_rows = [] if locked is None else [dict(row) for row in locked.get("task_accuracy_matrix", ())]
    joint_iid_by_stage = {
        int(row["stage"]): float(row["accuracy"]) for row in joint_iid_rows
    }
    flattened = [
        {
            "accuracy": row.get("accuracy"),
            "base_head_union": dict(row.get("controls", {})).get("base_head_union"),
            "cosine_union": dict(row.get("controls", {})).get("cosine_union"),
            "live_nodes": row.get("live_nodes"),
            "local_log_probability_union": dict(row.get("controls", {})).get(
                "local_log_probability_union"
            ),
            "offline_joint_iid": joint_iid_by_stage.get(int(row["stage"])),
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


def _clean_history_rows(
    clean: Mapping[str, object] | None,
    development: Mapping[str, object] | None = None,
) -> tuple[tuple[int, ...], list[dict[str, object]]]:
    """Project clean persistent-history measurements into one row per checkpoint."""
    clean_record = {} if clean is None else clean
    persistent = dict(clean_record.get("persistent", {}))
    fresh = tuple(dict(row) for row in clean_record.get("fresh", ()))
    controls = {
        int(dict(row)["stage"]): dict(row)
        for row in clean_record.get("hierarchy_controls", ())
    }
    histories = tuple(sorted((int(value) for value in persistent), key=int))
    persistent_by_history = {
        capacity: {
            int(dict(row)["stage"]): dict(row)
            for row in persistent[str(capacity)]
            if "accuracy" in row
        }
        for capacity in histories
    }
    rows: list[dict[str, object]] = []
    for fresh_row in sorted(fresh, key=lambda row: int(row["stage"])):
        stage = int(fresh_row["stage"])
        fresh_accuracy = float(fresh_row["mean_validation_accuracy"])
        stage_controls = dict(controls.get(stage, {}).get("controls", {}))
        projected: dict[str, object] = {
            "fresh_mean_accuracy": fresh_accuracy,
            "raw_union_accuracy": stage_controls.get("raw_union"),
            "stage": stage,
            "true_node_oracle_accuracy": stage_controls.get("true_node_oracle"),
        }
        for capacity in histories:
            persistent_row = persistent_by_history[capacity].get(stage)
            persistent_accuracy = (
                None if persistent_row is None else float(persistent_row["accuracy"])
            )
            projected[f"h{capacity}_accuracy"] = persistent_accuracy
            projected[f"h{capacity}_minus_fresh_pp"] = (
                None
                if persistent_accuracy is None
                else persistent_accuracy - fresh_accuracy
            )
        rows.append(projected)
    if development is None:
        return histories, rows
    development_fresh = tuple(dict(row) for row in development.get("fresh", ()))
    development_persistent = tuple(
        dict(row) for row in development.get("persistent", ()) if "accuracy" in row
    )
    selection = dict(development.get("selection", {}))
    selected_capacity_value = selection.get("selected_historical_capacity")
    if selected_capacity_value is None and development_persistent:
        selected_capacity_value = development_persistent[-1].get("historical_capacity")
    selected_capacity = (
        None if selected_capacity_value is None else int(selected_capacity_value)
    )
    if selected_capacity is not None:
        histories = tuple(sorted({*histories, selected_capacity}))
    raw_development_controls = development.get("hierarchy_controls", ())
    development_controls = (
        (dict(raw_development_controls),)
        if isinstance(raw_development_controls, Mapping)
        else tuple(dict(row) for row in raw_development_controls)
    )
    controls_by_stage = {
        int(row["stage"]): dict(row.get("controls", {}))
        for row in development_controls
    }
    persistent_by_stage = {
        int(row["stage"]): row for row in development_persistent
    }
    existing_stages = {int(row["stage"]) for row in rows}
    for fresh_row in development_fresh:
        stage = int(fresh_row["stage"])
        if stage in existing_stages:
            continue
        fresh_accuracy = float(fresh_row["mean_validation_accuracy"])
        persistent_row = persistent_by_stage.get(stage)
        stage_controls = controls_by_stage.get(
            stage,
            {} if persistent_row is None else dict(persistent_row.get("controls", {})),
        )
        projected = {
            "fresh_mean_accuracy": fresh_accuracy,
            "raw_union_accuracy": stage_controls.get("raw_union"),
            "stage": stage,
            "true_node_oracle_accuracy": stage_controls.get("true_node_oracle"),
        }
        if selected_capacity is not None:
            persistent_accuracy = (
                None if persistent_row is None else float(persistent_row["accuracy"])
            )
            projected[f"h{selected_capacity}_accuracy"] = persistent_accuracy
            projected[f"h{selected_capacity}_minus_fresh_pp"] = (
                None
                if persistent_accuracy is None
                else persistent_accuracy - fresh_accuracy
            )
        rows.append(projected)
    return histories, sorted(rows, key=lambda row: int(row["stage"]))


def _write_clean_history_tables(
    report_root: Path,
    histories: Sequence[int],
    rows: Sequence[Mapping[str, object]],
) -> None:
    """Write clean history-selection evidence in CSV, JSON, and Parquet forms."""
    _write_csv(report_root / "clean_history_selection.csv", rows)
    atomic_write(
        report_root / "clean_history_selection.json",
        canonical_json_bytes({"histories": list(histories), "rows": list(rows)}),
    )
    if rows:
        try:
            import pandas as pd

            pd.DataFrame(rows).to_parquet(
                report_root / "clean_history_selection.parquet", index=False
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
                "condition": (
                    f"hierarchy:{policy['parent_training']}:{policy['partition']}"
                ),
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
    joint_iid_rows: Sequence[Mapping[str, object]],
    joint_iid_last: float | None,
    histories: Sequence[int],
    history_rows: Sequence[Mapping[str, object]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def series_values(
        rows: Sequence[Mapping[str, object]], key: str
    ) -> list[float]:
        return [
            float("nan") if row.get(key) is None else float(row[key])
            for row in rows
        ]

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    if diagnostic:
        feature = dict(diagnostic.get("feature_accuracies", {}))
        controls = {
            name: value
            for name, value in dict(diagnostic.get("controls", {})).items()
            if name != "true_node_oracle"
        }
        labels = list(feature) + ["best static"]
        bar_values = [float(feature[label]) for label in feature] + (
            [max(controls.values())] if controls else [0.0]
        )
        axes[0].bar(
            labels,
            bar_values,
            color=["#1565c0"] * len(feature) + ["#9e9e9e"],
        )
        axes[0].set_ylim(0, 100)
        axes[0].tick_params(axis="x", rotation=25)
        axes[0].set(title="Sealed diagnostic", ylabel="Validation accuracy (%)")
    else:
        axes[0].text(0.5, 0.5, "Diagnostic not complete", ha="center", va="center")
        axes[0].set_axis_off()
    if history_rows:
        stages = [int(row["stage"]) for row in history_rows]
        axes[1].plot(
            stages,
            series_values(history_rows, "fresh_mean_accuracy"),
            color="#263238",
            linewidth=2.2,
            marker="o",
            label="fresh full-replay integrator",
        )
        colors = ("#1565c0", "#6a1b9a", "#00838f", "#ad1457")
        for index, capacity in enumerate(histories):
            key = f"h{capacity}_accuracy"
            if not any(row.get(key) is not None for row in history_rows):
                continue
            axes[1].plot(
                stages,
                series_values(history_rows, key),
                color=colors[index % len(colors)],
                marker="s",
                label=f"persistent H={capacity}",
            )
        axes[1].plot(
            stages,
            series_values(history_rows, "raw_union_accuracy"),
            color="#ef6c00",
            linestyle="--",
            marker="^",
            label="raw union",
        )
        axes[1].plot(
            stages,
            series_values(history_rows, "true_node_oracle_accuracy"),
            color="#2e7d32",
            linestyle=":",
            marker="D",
            label="true-node oracle",
        )
        axes[1].set(
            xlabel="Tasks seen",
            ylabel="Validation accuracy (%)",
            title="Clean validation checkpoints",
            xticks=stages,
        )
        axes[1].set_ylim(55, 100)
        axes[1].grid(alpha=0.2)
        axes[1].legend(fontsize=8)
    else:
        axes[1].text(0.5, 0.5, "Validation checkpoints not complete", ha="center", va="center")
        axes[1].set_axis_off()
    if locked_rows:
        stages = [int(row["stage"]) for row in locked_rows]
        axes[2].plot(
            stages,
            series_values(locked_rows, "accuracy"),
            color="#1565c0",
            linewidth=2.0,
            label="persistent full-union integrator",
        )
        for key, label, color, linestyle in (
            ("raw_union", "raw union", "#ef6c00", "--"),
            ("true_node_oracle", "true-node oracle", "#2e7d32", ":"),
        ):
            axes[2].plot(
                stages,
                series_values(locked_rows, key),
                color=color,
                label=label,
                linestyle=linestyle,
            )
        if joint_iid_rows:
            axes[2].plot(
                [int(row["stage"]) for row in joint_iid_rows],
                series_values(joint_iid_rows, "accuracy"),
                color="#212121",
                linewidth=2.2,
                linestyle="-.",
                label="offline joint-IID ceiling",
            )
        elif joint_iid_last is not None:
            axes[2].axhline(
                joint_iid_last,
                color="#212121",
                linewidth=2.2,
                linestyle="-.",
                label="offline joint-IID final ceiling",
            )
        axes[2].set(
            xlabel="Tasks seen",
            ylabel="Test accuracy (%)",
            title="Locked test with offline joint-IID ceiling",
        )
        axes[2].set_ylim(55, 100)
        axes[2].grid(alpha=0.2)
        axes[2].legend(fontsize=8)
    else:
        axes[2].text(0.5, 0.5, "Locked test not opened", ha="center", va="center")
        axes[2].set_axis_off()
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
    joint_iid_rows = _joint_iid_stage_rows(run, locked)
    stage_rows, _task_rows = _write_tables(reports, locked, joint_iid_rows)
    histories, history_rows = _clean_history_rows(clean, development)
    _write_clean_history_tables(reports, histories, history_rows)
    resources = _resource_rows(run)
    _write_resource_tables(reports, resources)
    lineage = reports / "lineage.png"
    accuracy_plot = reports / "accuracy.png"
    _lineage_plot(lineage)
    _accuracy_plot(
        accuracy_plot,
        diagnostic,
        stage_rows,
        joint_iid_rows,
        (
            None
            if locked is None
            else float(dict(locked.get("local_references", {}))["joint_iid_last"])
        ),
        histories,
        history_rows,
    )
    diagnostic_rows = []
    if diagnostic:
        diagnostic_rows = [
            (name, f"{float(value):.3f}")
            for name, value in dict(diagnostic.get("feature_accuracies", {})).items()
        ]
    comparison_rows = []
    gap_rows = []
    static_rows = []
    if locked:
        references = dict(locked["local_references"])
        static = dict(dict(locked["final_static_controls"])["controls"])
        comparison_rows = [
            (
                "LogT full-union integrator",
                f"{float(locked['last_accuracy']):.3f}",
                f"{float(locked['incremental_accuracy']):.3f}",
                "evaluated method",
            ),
            (
                "Offline joint-IID LoRA",
                f"{float(references['joint_iid_last']):.3f}",
                f"{float(references['joint_iid_incremental']):.3f}",
                "primary ceiling",
            ),
            (
                "Local E2-LoRA",
                f"{float(locked['local_e2_last']):.3f}",
                f"{float(locked['local_e2_incremental']):.3f}",
                "secondary reference",
            ),
            (
                "Published E2-LoRA",
                f"{float(references['published_e2_last']):.3f}",
                f"{float(references['published_e2_incremental']):.3f}",
                "external context",
            ),
        ]
        gaps = dict(locked["comparisons"])
        gap_rows = [
            (
                "Integrator − offline joint-IID",
                f"{float(gaps['last_minus_joint_iid']):+.3f}",
                f"{float(gaps['incremental_minus_joint_iid']):+.3f}",
            ),
            (
                "Integrator − local E2-LoRA",
                f"{float(gaps['last_minus_local_e2']):+.3f}",
                f"{float(gaps['incremental_minus_local_e2']):+.3f}",
            ),
        ]
        static_rows = [
            ("raw union", f"{float(static['raw_union']):.3f}"),
            ("cosine union", f"{float(static['cosine_union']):.3f}"),
            (
                "affine-calibrated union",
                f"{float(static['affine_calibrated_union']):.3f}",
            ),
            ("true-node oracle", f"{float(static['true_node_oracle']):.3f}"),
        ]
    selected = None if clean is None else {
        "feature_variant": clean.get("selected_variant"),
        "historical_capacity": clean.get("selected_historical_capacity"),
        "parent_training": clean.get("selected_parent_training"),
    }
    history_headers = ["stage", "fresh mean", "raw union", "true-node oracle"]
    for capacity in histories:
        history_headers.extend((f"H={capacity}", f"H={capacity} - fresh (pp)"))

    def accuracy_text(value: object) -> str:
        return "—" if value is None else f"{float(value):.3f}"

    def difference_text(value: object) -> str:
        return "—" if value is None else f"{float(value):+.3f}"

    history_table_rows = []
    for row in history_rows:
        values: list[object] = [
            row["stage"],
            accuracy_text(row.get("fresh_mean_accuracy")),
            accuracy_text(row.get("raw_union_accuracy")),
            accuracy_text(row.get("true_node_oracle_accuracy")),
        ]
        for capacity in histories:
            values.extend(
                (
                    accuracy_text(row.get(f"h{capacity}_accuracy")),
                    difference_text(row.get(f"h{capacity}_minus_fresh_pp")),
                )
            )
        history_table_rows.append(tuple(values))
    clean_note = (
        "The feature family and H were frozen from the authenticated v2 task-16 "
        "development evidence before test access. All displayed fresh and static "
        "differences are diagnostic, not thresholds.\n\n"
        if clean is not None
        else ""
    )
    split_note = (
        "The middle accuracy panel shows fresh full-replay and persistent integrator "
        "measurements only at the selected clean-validation checkpoints (tasks "
        "2/4/8/16/50). The right panel shows the complete locked-test curves, including "
        "the offline joint-IID ceiling. Validation and test curves are deliberately kept "
        "in separate panels and must not be compared point-for-point across splits."
    )
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
        "# ImageNet-R-50 Full-Union LogT Prediction Integrator\n\n"
        f"Run: `{run.name}`  \nWorkflow state: `{state.get('phase', 'NOT_STARTED')}`\n\n"
        "This experiment replaces task-free node selection with a direct 200-way residual "
        "integrator over frozen node behavior. Its capacity-one binary counter retrains every "
        "parent on the complete represented training union; only persistent integrator history "
        "uses a bounded replay reservoir. No accuracy or comparator value gates execution.\n\n"
        "## Primary benchmark comparison\n\n"
        + _markdown_table(
            ("condition", "Last (%)", "Incremental (%)", "role"),
            comparison_rows,
        )
        + "\n"
        + _markdown_table(("descriptive difference", "Last (pp)", "Incremental (pp)"), gap_rows)
        + "\nThe offline joint-IID rank-16 LoRA run is the primary ceiling. The local "
        "E2-LoRA reproduction is secondary, and the published E2-LoRA values are "
        "external context. None is a pass/fail condition.\n\n"
        "## Report-only feature diagnostic\n\n"
        + _markdown_table(("feature family", "mean validation accuracy"), diagnostic_rows)
        + (f"\nConfigured feature family: `{diagnostic.get('selected_variant')}`.\n" if diagnostic else "")
        + "\n## Frozen development choices\n\n"
        + (f"`{json.dumps(selected, sort_keys=True)}`\n\n" if selected else "_Not complete._\n\n")
        + clean_note
        + _markdown_table(tuple(history_headers), history_table_rows)
        + "\n"
        + "Validation identities are excluded from every clean node and integrator update. "
        "Full-union parent retraining is the primary condition, matching the successful "
        "Permuted-MNIST consolidation methodology.\n\n"
        + "## Final-frontier diagnostics\n\n"
        + _markdown_table(("task-free/oracle diagnostic", "Last (%)"), static_rows)
        + "\nThese rows explain the hierarchy frontier; they do not define acceptance.\n\n"
        + "## Complexity boundary\n\n"
        + "The hierarchy retains at most `popcount(t)` live adapters and performs at most "
        "`bit_length(t)` carries per arrival. A carry retrains on its complete represented "
        "union, so its worst-case data work grows with interval size and cumulative parent "
        "presentations are O(N T log T) for N examples per task. Persistent observer work "
        "remains bounded by `popcount(t) * (current + H)` per arrival.\n\n"
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
        + "## Figures\n\n"
        + split_note
        + "\n\n![Accuracy](accuracy.png)\n\n![Capacity-one lineage](lineage.png)\n"
    )
    atomic_write(reports / "REPORT.md", markdown.encode("utf-8"))
    diagnostic_html = _html_table(("feature family", "mean validation accuracy"), diagnostic_rows)
    history_html = _html_table(tuple(history_headers), history_table_rows)
    comparison_html = _html_table(
        ("condition", "Last (%)", "Incremental (%)", "role"), comparison_rows
    )
    gap_html = _html_table(
        ("descriptive difference", "Last (pp)", "Incremental (pp)"), gap_rows
    )
    static_html = _html_table(
        ("task-free/oracle diagnostic", "Last (%)"), static_rows
    )
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
<body><h1>ImageNet-R-50 Full-Union LogT Prediction Integrator</h1><p><strong>Run:</strong> <code>{escape(run.name)}</code><br><strong>State:</strong> {escape(str(state.get('phase', 'NOT_STARTED')))}</p>
<p>A direct 200-way residual integrator observes frozen LogT node behavior. Capacity-one parents retrain on every represented training example; only persistent integrator history is bounded. Accuracy never gates execution.</p>
<details open><summary>Primary benchmark comparison</summary>{comparison_html}{gap_html}<p>Offline joint-IID is the primary ceiling; local E2-LoRA is secondary. All differences are descriptive.</p></details>
<details><summary>Report-only feature diagnostic</summary>{diagnostic_html}<p>Configured feature family: <code>{escape(str(None if diagnostic is None else diagnostic.get('selected_variant')))}</code>.</p></details>
<details open><summary>Frozen development choices</summary><p><code>{escape(json.dumps(selected, sort_keys=True))}</code></p><p>{escape(clean_note.strip())}</p>{history_html}<p>Validation identities are excluded from all clean trainable components.</p></details>
<details><summary>Final-frontier diagnostics</summary>{static_html}<p>These rows explain the hierarchy frontier and do not define acceptance.</p></details>
<details open><summary>Accuracy</summary><p>{escape(split_note)}</p><img alt="Feature diagnostic, selected validation checkpoints, and locked test curves with the offline joint-IID ceiling" src="{_image_data(accuracy_plot)}"></details>
<details><summary>Capacity-one lineage</summary><img alt="Capacity-one binary-counter lineage" src="{_image_data(lineage)}"></details>
<details><summary>Complexity and resources</summary><p>There are at most bit_length(t) carries and popcount(t) live nodes. Full-union carry cost grows with interval size; cumulative parent presentations are O(N T log T) for N examples per task. Persistent observer work stays bounded by popcount(t) × (current + H).</p>{resource_html}<pre>{escape(json.dumps({'clean': clean, 'development': development}, indent=2, sort_keys=True))}</pre></details>
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

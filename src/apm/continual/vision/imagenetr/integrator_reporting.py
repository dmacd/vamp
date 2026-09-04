"""Markdown, standalone HTML, tables, and plots for the integrator workflow."""

from __future__ import annotations

from base64 import b64encode
from collections.abc import Mapping, Sequence
from html import escape
from math import fsum, isclose, isfinite
from pathlib import Path
import csv
import json
import re

from apm.continual.artifacts import (
    ChainedJsonlLedger,
    atomic_write,
    canonical_json_bytes,
    file_sha256,
    load_canonical_json,
    record_sha256,
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
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _sealed_primary_report_path(run: Path, filename: str) -> Path | None:
    """Resolve one report file from the primary run bound by the integrator config."""
    config = _load(run / "config_resolved.json")
    if config is None or not config.get("sealed_run_hash"):
        return None
    sealed_run_hash = str(config["sealed_run_hash"])
    configured_root = Path(str(config.get("inference_artifact_root", "")))
    candidate_roots = (
        ([configured_root] if configured_root.is_absolute() else [Path.cwd() / configured_root])
        + list(run.parents)
    )
    return next(
        (
            root / "runs" / sealed_run_hash / "reports" / filename
            for root in candidate_roots
            if (root / "runs" / sealed_run_hash / "reports" / filename).is_file()
        ),
        None,
    )


def _joint_iid_stage_rows(
    run: Path, locked: Mapping[str, object] | None
) -> list[dict[str, object]]:
    """Load the full-data joint model's future-informed prefix-evaluation curve."""
    if locked is None:
        return []
    source = _sealed_primary_report_path(run, "stage_accuracy.csv")
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


def _source_oracle_references(run: Path) -> list[dict[str, object]]:
    """Load authenticated final oracle results from the less-compressed source hierarchy."""
    source = _sealed_primary_report_path(run, "summary.json")
    references = _load(run / "protocol" / "reference_results.json")
    if source is None or references is None:
        return []
    if file_sha256(source) != references.get("primary_summary_sha256"):
        raise ValueError("sealed primary summary differs from the integrator reference hash")
    summary = load_canonical_json(source)
    conditions = {
        str(row["condition"]): row for row in summary.get("conditions", ())
    }
    requested = (
        ("all-leaf true-task oracle", "leaf_bank_50", 50),
        ("capacity-two retrained true-node oracle", "logt_retrain_union_r16", 8),
    )
    if any(condition not in conditions for _label, condition, _nodes in requested):
        raise ValueError("sealed primary summary lacks expected oracle conditions")
    return [
        {
            "condition": label,
            "final_accuracy": float(conditions[condition]["true_node_oracle_last_accuracy"]),
            "live_nodes": live_nodes,
        }
        for label, condition, live_nodes in requested
    ]


def _stage_matched_joint_rows(run: Path) -> list[dict[str, object]]:
    """Load and authenticate fresh joint models trained separately at every stage."""
    source = run / "evaluations" / "stage_matched_joint_iid.json"
    if not source.is_file():
        return []
    record = load_canonical_json(source)
    supplied_hash = str(record.get("content_hash", ""))
    core = {
        key: value
        for key, value in record.items()
        if key not in {"content_hash", "control_relative_path"}
    }
    if (
        record.get("schema_version")
        != "imagenetr50-stage-matched-joint-iid-summary-v1"
        or supplied_hash != record_sha256(core)
    ):
        raise ValueError("stage-matched joint-IID summary does not authenticate")
    rows = sorted(
        (
            {
                "accuracy": float(row["accuracy"]),
                "evaluation_seconds": float(row["evaluation_seconds"]),
                "image_presentations": int(row["image_presentations"]),
                "optimizer_steps": int(row["optimizer_steps"]),
                "peak_vram_bytes": int(row["peak_vram_bytes"]),
                "reused_source_model": bool(row["reused_source_model"]),
                "stage": int(row["stage"]),
                "test_examples": int(row["test_examples"]),
                "train_examples": int(row["train_examples"]),
                "last_minibatch_training_loss": (
                    None
                    if row.get("training_final_loss") is None
                    else float(row["training_final_loss"])
                ),
                "training_seconds": float(row["training_seconds"]),
            }
            for row in record.get("rows", ())
        ),
        key=lambda row: int(row["stage"]),
    )
    if [int(row["stage"]) for row in rows] != list(range(1, 51)) or any(
        not isfinite(float(row["accuracy"])) for row in rows
    ):
        raise ValueError("stage-matched joint-IID curve must contain stages 1..50")
    expected_incremental = float(record["incremental_accuracy"])
    if not isclose(
        fsum(float(row["accuracy"]) for row in rows) / len(rows),
        expected_incremental,
        abs_tol=1e-9,
    ):
        raise ValueError("stage-matched joint-IID aggregate changed")
    return rows


def _write_tables(
    report_root: Path,
    locked: Mapping[str, object] | None,
    future_informed_joint_rows: Sequence[Mapping[str, object]] = (),
    stage_matched_joint_rows: Sequence[Mapping[str, object]] = (),
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    stage_rows = [] if locked is None else [dict(row) for row in locked.get("stage_metrics", ())]
    task_rows = [] if locked is None else [dict(row) for row in locked.get("task_accuracy_matrix", ())]
    future_informed_by_stage = {
        int(row["stage"]): float(row["accuracy"])
        for row in future_informed_joint_rows
    }
    stage_matched_by_stage = {
        int(row["stage"]): float(row["accuracy"])
        for row in stage_matched_joint_rows
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
            "future_informed_joint_iid": future_informed_by_stage.get(
                int(row["stage"])
            ),
            "future_information_effect_pp": (
                None
                if int(row["stage"]) not in stage_matched_by_stage
                or int(row["stage"]) not in future_informed_by_stage
                else future_informed_by_stage[int(row["stage"])]
                - stage_matched_by_stage[int(row["stage"])]
            ),
            "frontier_hash": row.get("frontier_hash"),
            "raw_union": dict(row.get("controls", {})).get("raw_union"),
            "stage_matched_joint_iid": stage_matched_by_stage.get(int(row["stage"])),
            "stage_matched_minus_true_node_pp": (
                None
                if int(row["stage"]) not in stage_matched_by_stage
                or dict(row.get("controls", {})).get("true_node_oracle") is None
                else stage_matched_by_stage[int(row["stage"])]
                - float(dict(row.get("controls", {}))["true_node_oracle"])
            ),
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


def _write_stage_matched_tables(
    report_root: Path, rows: Sequence[Mapping[str, object]]
) -> None:
    """Publish the compact stage-matched curve and its measured training work."""
    if not rows:
        return
    projected = [dict(row) for row in rows]
    _write_csv(report_root / "stage_matched_joint_iid.csv", projected)
    atomic_write(
        report_root / "stage_matched_joint_iid.json",
        canonical_json_bytes({"rows": projected}),
    )
    try:
        import pandas as pd

        pd.DataFrame(projected).to_parquet(
            report_root / "stage_matched_joint_iid.parquet", index=False
        )
    except ImportError:  # pragma: no cover - environment preflight requires pandas
        pass


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
    future_informed_joint_rows: Sequence[Mapping[str, object]],
    stage_matched_joint_rows: Sequence[Mapping[str, object]],
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
                label=f"persistent LogT integrator (H={capacity})",
            )
        axes[1].plot(
            stages,
            series_values(history_rows, "raw_union_accuracy"),
            color="#ef6c00",
            linestyle="--",
            marker="^",
            label="raw LogT union",
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
            label="persistent LogT integrator (H=2048)",
        )
        for key, label, color, linestyle in (
            ("raw_union", "raw LogT union", "#ef6c00", "--"),
            ("true_node_oracle", "true-node oracle", "#2e7d32", ":"),
        ):
            axes[2].plot(
                stages,
                series_values(locked_rows, key),
                color=color,
                label=label,
                linestyle=linestyle,
            )
        if stage_matched_joint_rows:
            axes[2].plot(
                [int(row["stage"]) for row in stage_matched_joint_rows],
                series_values(stage_matched_joint_rows, "accuracy"),
                color="#7b1fa2",
                linewidth=2.2,
                label="joint-IID, stage-matched",
            )
        if future_informed_joint_rows:
            axes[2].plot(
                [int(row["stage"]) for row in future_informed_joint_rows],
                series_values(future_informed_joint_rows, "accuracy"),
                color="#212121",
                linewidth=1.8,
                linestyle="-.",
                label="joint-IID, trained through task 50",
            )
        elif joint_iid_last is not None:
            axes[2].axhline(
                joint_iid_last,
                color="#212121",
                linewidth=2.2,
                linestyle="-.",
                label="joint-IID task-50 ceiling",
            )
        axes[2].set(
            xlabel="Tasks seen",
            ylabel="Test accuracy (%)",
            title="Locked test: matched and future-informed controls",
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


def _joint_information_plot(
    path: Path,
    locked_rows: Sequence[Mapping[str, object]],
    future_informed_rows: Sequence[Mapping[str, object]],
    stage_matched_rows: Sequence[Mapping[str, object]],
) -> None:
    """Plot the direct future-information diagnostic and its residual oracle gap."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    if not (locked_rows and future_informed_rows and stage_matched_rows):
        axes[0].text(0.5, 0.5, "Stage-matched control not complete", ha="center", va="center")
        axes[0].set_axis_off()
        axes[1].set_axis_off()
    else:
        stages = [int(row["stage"]) for row in locked_rows]
        oracle = [float(row["true_node_oracle"]) for row in locked_rows]
        future = [float(row["accuracy"]) for row in future_informed_rows]
        matched = [float(row["accuracy"]) for row in stage_matched_rows]
        axes[0].plot(
            stages,
            oracle,
            color="#2e7d32",
            linewidth=2.2,
            label="true-node oracle",
        )
        axes[0].plot(
            stages,
            matched,
            color="#7b1fa2",
            linewidth=2.2,
            label="joint-IID, stage-matched",
        )
        axes[0].plot(
            stages,
            future,
            color="#212121",
            linewidth=1.8,
            linestyle="-.",
            label="joint-IID, trained through task 50",
        )
        axes[0].set(ylabel="Test accuracy (%)", title="What does future training data explain?")
        axes[0].grid(alpha=0.2)
        axes[0].legend(fontsize=9)
        axes[1].plot(
            stages,
            [future_value - matched_value for future_value, matched_value in zip(future, matched)],
            color="#212121",
            linestyle="-.",
            label="future-data effect: full-50 joint − stage-matched joint",
        )
        axes[1].plot(
            stages,
            [matched_value - oracle_value for matched_value, oracle_value in zip(matched, oracle)],
            color="#7b1fa2",
            label="residual gap: stage-matched joint − true-node oracle",
        )
        for checkpoint in (1, 2, 4, 8, 16, 32, 50):
            axes[1].axvline(checkpoint, color="#b0bec5", linewidth=0.55, alpha=0.35)
        axes[1].axhline(0.0, color="#455a64", linewidth=0.8)
        axes[1].set(xlabel="Tasks seen", ylabel="Difference (percentage points)")
        axes[1].grid(alpha=0.2)
        axes[1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
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


def write_report_manifest(report_root: str | Path) -> Path:
    """Hash every compact report artifact after all requested formats exist."""
    reports = Path(report_root)
    files = tuple(
        {
            "path": path.name,
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(reports.iterdir())
        if path.is_file() and path.name != "report_manifest.json"
    )
    return atomic_write(
        reports / "report_manifest.json",
        canonical_json_bytes(
            {
                "files": list(files),
                "schema_version": "imagenetr50-integrator-report-manifest-v1",
            }
        ),
    )


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
    future_informed_joint_rows = _joint_iid_stage_rows(run, locked)
    stage_matched_joint_rows = _stage_matched_joint_rows(run)
    source_oracles = _source_oracle_references(run)
    stage_rows, _task_rows = _write_tables(
        reports,
        locked,
        future_informed_joint_rows,
        stage_matched_joint_rows,
    )
    _write_stage_matched_tables(reports, stage_matched_joint_rows)
    histories, history_rows = _clean_history_rows(clean, development)
    _write_clean_history_tables(reports, histories, history_rows)
    resources = _resource_rows(run)
    _write_resource_tables(reports, resources)
    lineage = reports / "lineage.png"
    accuracy_plot = reports / "accuracy.png"
    information_plot = reports / "joint_information_gap.png"
    _lineage_plot(lineage)
    _accuracy_plot(
        accuracy_plot,
        diagnostic,
        stage_rows,
        future_informed_joint_rows,
        stage_matched_joint_rows,
        (
            None
            if locked is None
            else float(dict(locked.get("local_references", {}))["joint_iid_last"])
        ),
        histories,
        history_rows,
    )
    _joint_information_plot(
        information_plot,
        stage_rows,
        future_informed_joint_rows,
        stage_matched_joint_rows,
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
    information_rows = []
    frontier_gap_rows = []
    hierarchy_context_rows = []
    stage_control_work_rows = []
    optimization_rows = []
    information_conclusion = "The stage-matched control is not complete."
    hierarchy_conclusion = "The authenticated source hierarchy is unavailable."
    optimization_conclusion = "The task-32 recipe comparison is unavailable."
    matched_incremental: float | None = None
    if locked:
        references = dict(locked["local_references"])
        static = dict(dict(locked["final_static_controls"])["controls"])
        matched_incremental = (
            None
            if not stage_matched_joint_rows
            else fsum(float(row["accuracy"]) for row in stage_matched_joint_rows)
            / len(stage_matched_joint_rows)
        )
        oracle_incremental = fsum(
            float(row["true_node_oracle"]) for row in stage_rows
        ) / len(stage_rows)
        comparison_rows = [
            (
                "persistent LogT integrator (H=2048)",
                f"{float(locked['last_accuracy']):.3f}",
                f"{float(locked['incremental_accuracy']):.3f}",
                "evaluated method",
            ),
            *(
                [
                    (
                        "joint-IID, stage-matched",
                        f"{float(stage_matched_joint_rows[-1]['accuracy']):.3f}",
                        f"{float(matched_incremental):.3f}",
                        "post-hoc available-data control",
                    )
                ]
                if matched_incremental is not None
                else []
            ),
            (
                "joint-IID, trained through task 50",
                f"{float(references['joint_iid_last']):.3f}",
                f"{float(references['joint_iid_incremental']):.3f}",
                "offline ceiling; earlier points are future-informed",
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
            *(
                [
                    (
                        "Persistent LogT − stage-matched joint",
                        f"{float(locked['last_accuracy']) - float(stage_matched_joint_rows[-1]['accuracy']):+.3f}",
                        f"{float(locked['incremental_accuracy']) - float(matched_incremental):+.3f}",
                    ),
                    (
                        "True-node oracle − stage-matched joint",
                        f"{float(static['true_node_oracle']) - float(stage_matched_joint_rows[-1]['accuracy']):+.3f}",
                        f"{oracle_incremental - float(matched_incremental):+.3f}",
                    ),
                    (
                        "Full-50 joint − stage-matched joint",
                        f"{float(references['joint_iid_last']) - float(stage_matched_joint_rows[-1]['accuracy']):+.3f}",
                        f"{float(references['joint_iid_incremental']) - float(matched_incremental):+.3f}",
                    ),
                ]
                if matched_incremental is not None
                else []
            ),
            (
                "Persistent LogT − full-50 joint",
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
            ("raw LogT union", f"{float(static['raw_union']):.3f}"),
            ("cosine LogT union", f"{float(static['cosine_union']):.3f}"),
            (
                "affine-calibrated LogT union",
                f"{float(static['affine_calibrated_union']):.3f}",
            ),
            ("true-node oracle", f"{float(static['true_node_oracle']):.3f}"),
        ]
        source_by_condition = {
            str(row["condition"]): row for row in source_oracles
        }
        source_labels = (
            "all-leaf true-task oracle",
            "capacity-two retrained true-node oracle",
        )
        if all(label in source_by_condition for label in source_labels):
            all_leaf_accuracy = float(
                source_by_condition["all-leaf true-task oracle"]["final_accuracy"]
            )
            capacity_two_accuracy = float(
                source_by_condition["capacity-two retrained true-node oracle"][
                    "final_accuracy"
                ]
            )
            capacity_one_accuracy = float(static["true_node_oracle"])
            joint_accuracy = float(references["joint_iid_last"])
            hierarchy_context_rows = [
                (
                    "all-leaf true-task oracle",
                    50,
                    f"{all_leaf_accuracy:.3f}",
                    "no consolidation; diagnostic task identity",
                ),
                (
                    "capacity-two retrained true-node oracle",
                    8,
                    f"{capacity_two_accuracy:.3f}",
                    "less-compressed source hierarchy",
                ),
                (
                    "capacity-one full-union true-node oracle",
                    int(stage_rows[-1]["live_nodes"]),
                    f"{capacity_one_accuracy:.3f}",
                    "current hierarchy",
                ),
                (
                    "joint-IID, trained through task 50",
                    1,
                    f"{joint_accuracy:.3f}",
                    "one shared adapter and global head",
                ),
            ]
            hierarchy_conclusion = (
                "The no-consolidation leaf oracle reaches "
                f"{all_leaf_accuracy:.3f}%. The capacity-two retrained hierarchy loses "
                f"{all_leaf_accuracy - capacity_two_accuracy:.3f} points but still exceeds "
                f"the offline joint model by {capacity_two_accuracy - joint_accuracy:+.3f} "
                "points. Compressing further to the current capacity-one hierarchy loses "
                f"another {capacity_two_accuracy - capacity_one_accuracy:.3f} points and "
                f"finishes {capacity_one_accuracy - joint_accuracy:+.3f} points relative "
                "to joint-IID. Thus neither true-node routing nor rank-16 LoRA inherently "
                "prevents joint-level performance; the evidence points most strongly to "
                "accuracy lost as represented intervals are enlarged and consolidated, "
                "together with parent optimization. This is descriptive, not a clean "
                "capacity ablation: the oracle candidate sets shrink as more nodes are kept; "
                "the capacity-two parents used 5e-4 weight decay and distinct deterministic "
                "initialization/order seeds, whereas the current capacity-one parents used "
                "zero weight decay."
            )
        if stage_matched_joint_rows:
            stage_control_work_rows = [
                (
                    sum(not bool(row["reused_source_model"]) for row in stage_matched_joint_rows),
                    sum(int(row["image_presentations"]) for row in stage_matched_joint_rows),
                    sum(int(row["optimizer_steps"]) for row in stage_matched_joint_rows),
                    f"{fsum(float(row['training_seconds']) for row in stage_matched_joint_rows) / 60:.2f}",
                    f"{max(int(row['peak_vram_bytes']) for row in stage_matched_joint_rows) / 2**30:.2f}",
                )
            ]
            selected_stages = frozenset((1, 2, 4, 8, 16, 32, 50))
            information_rows = [
                (
                    int(row["stage"]),
                    f"{float(row['true_node_oracle']):.3f}",
                    f"{float(row['stage_matched_joint_iid']):.3f}",
                    f"{float(row['future_informed_joint_iid']):.3f}",
                    f"{float(row['future_information_effect_pp']):+.3f}",
                    f"{float(row['stage_matched_minus_true_node_pp']):+.3f}",
                )
                for row in stage_rows
                if int(row["stage"]) in selected_stages
            ]
            future_effects = tuple(
                float(row["future_information_effect_pp"])
                for row in stage_rows[:-1]
            )
            residual_gaps = tuple(
                float(row["stage_matched_minus_true_node_pp"])
                for row in stage_rows
            )
            power_two_gaps = tuple(
                float(row["stage_matched_minus_true_node_pp"])
                for row in stage_rows
                if int(row["stage"]) in {1, 2, 4, 8, 16, 32}
            )
            oldest_final_interval = stage_rows[31]
            frontier_gap_rows = [
                (
                    live_nodes,
                    len(group),
                    f"{fsum(float(row['stage_matched_minus_true_node_pp']) for row in group) / len(group):+.3f}",
                    f"{fsum(float(row['future_information_effect_pp']) for row in group) / len(group):+.3f}",
                )
                for live_nodes in sorted(
                    {int(row["live_nodes"]) for row in stage_rows}
                )
                for group in (
                    tuple(
                        row
                        for row in stage_rows
                        if int(row["live_nodes"]) == live_nodes
                    ),
                )
            ]
            information_conclusion = (
                "At task 50 the two joint controls are the same authenticated model, so "
                f"their difference is {float(stage_rows[-1]['future_information_effect_pp']):+.3f} "
                "point by construction, while it exceeds the true-node oracle by "
                f"{residual_gaps[-1]:.3f} points. This rules out tasks beyond the benchmark "
                "horizon, but not node-local missing transfer: the final older interval "
                "nodes were frozen before later intervals arrived. Before task 50, training "
                "through all 50 tasks changes prefix accuracy by "
                f"{fsum(future_effects) / len(future_effects):+.3f} points on average "
                f"(range {min(future_effects):+.3f} to {max(future_effects):+.3f}); "
                f"the contrast is positive at {sum(value > 0 for value in future_effects)} "
                f"stages and negative at {sum(value < 0 for value in future_effects)}. "
                "At the one-node power-of-two frontiers, where routing and frontier "
                "fragmentation disappear, the stage-matched joint-minus-oracle gaps average "
                f"{fsum(power_two_gaps) / len(power_two_gaps):+.3f} points; those same-stage "
                "gaps cannot be caused by unseen later tasks. The task-32 row is also an "
                "exact decomposition of the oldest node retained at task 50 (tasks 1–32): "
                "later-task co-training changes that interval by "
                f"{float(oldest_final_interval['future_information_effect_pp']):+.3f} points, "
                "while the matched fresh joint model differs from the hierarchy parent by "
                f"{float(oldest_final_interval['stage_matched_minus_true_node_pp']):+.3f} points. "
                "The live-node grouping is descriptive rather than causal because node count "
                "is correlated with task stage."
            )
            matching_snapshots = tuple(
                (path, snapshot)
                for path in sorted(
                    (run / "hierarchies").glob("*/snapshots/stage_032.json")
                )
                for snapshot in (_load(path),)
                if snapshot is not None
                and snapshot.get("content_hash")
                == oldest_final_interval.get("frontier_hash")
            )
            if len(matching_snapshots) > 1:
                raise ValueError("multiple hierarchies claim the locked task-32 frontier")
            if matching_snapshots:
                snapshot_path, snapshot = matching_snapshots[0]
                logical_ids = tuple(snapshot.get("logical_node_ids", ()))
                node_hashes = tuple(snapshot.get("node_hashes", ()))
                if len(logical_ids) != 1 or len(node_hashes) != 1:
                    raise ValueError("the task-32 frontier must contain exactly one node")
                node_root = snapshot_path.parents[1] / "nodes" / str(logical_ids[0])
                node = _load(node_root / "node.json")
                parent_training = _load(node_root / "training_metrics.json")
                if node is None or node.get("content_hash") != node_hashes[0]:
                    raise ValueError("the task-32 snapshot does not authenticate its node")
                if parent_training is None:
                    raise ValueError("the task-32 parent lacks training measurements")
                matched_stage32 = stage_matched_joint_rows[31]
                if int(node.get("represented_train_image_count", -1)) != int(
                    matched_stage32["train_examples"]
                ):
                    raise ValueError("the task-32 comparison does not use matched rows")
                if (
                    int(parent_training["image_presentations"]),
                    int(parent_training["optimizer_steps"]),
                ) != (
                    int(matched_stage32["image_presentations"]),
                    int(matched_stage32["optimizer_steps"]),
                ):
                    raise ValueError("the task-32 comparison does not have matched work")
                optimization_rows = [
                    (
                        "capacity-one hierarchy parent",
                        "unioned child classifier rows",
                        "0",
                        int(parent_training["image_presentations"]),
                        int(parent_training["optimizer_steps"]),
                        f"{float(oldest_final_interval['true_node_oracle']):.3f}",
                    ),
                    (
                        "joint-IID, stage-matched",
                        "fresh deterministic prefix head",
                        "5e-4",
                        int(matched_stage32["image_presentations"]),
                        int(matched_stage32["optimizer_steps"]),
                        f"{float(matched_stage32['accuracy']):.3f}",
                    ),
                ]
                optimization_conclusion = (
                    "Both task-32 fits receive the same training rows, presentations, and "
                    "optimizer-step budget, yet the hierarchy parent tests "
                    f"{float(oldest_final_interval['stage_matched_minus_true_node_pp']):.3f} "
                    "points worse. Fewer examples or steps cannot explain this contrast. "
                    "Convergence and under-training remain possible because classifier "
                    "initialization, regularization, and deterministic seed/order differ; "
                    "they require the proposed factorial and epoch sweep. The recorded "
                    "final-loss field is a last-minibatch diagnostic and is deliberately "
                    "not compared."
                )
    selected = None if clean is None else {
        "feature_variant": clean.get("selected_variant"),
        "historical_capacity": clean.get("selected_historical_capacity"),
        "parent_training": clean.get("selected_parent_training"),
    }
    history_headers = [
        "stage",
        "fresh full-replay integrator",
        "raw LogT union",
        "true-node oracle",
    ]
    for capacity in histories:
        history_headers.extend(
            (
                f"persistent LogT integrator (H={capacity})",
                f"persistent H={capacity} − fresh (pp)",
            )
        )

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
        "2/4/8/16/50). The right panel shows complete locked-test curves. ‘Joint-IID, "
        "stage-matched’ means a separately initialized rank-16 model trained only on data "
        "available through that stage. ‘Joint-IID, trained through task 50’ means the one "
        "offline model trained on all tasks and then evaluated on each class prefix; its "
        "points before task 50 use future training information. Validation and test curves "
        "remain in separate panels and must not be compared point-for-point across splits."
    )
    stage_control_protocol_note = (
        "The stage-matched control fits a fresh joint rank-16 LoRA adapter and affine "
        "head for five epochs at every stage, using exactly the training examples from "
        "tasks seen by that stage. The hierarchy nodes and both joint controls use the "
        "same pinned ViT backbone, rank and alpha 16, and attention-QKV plus MLP-fc1 "
        "adapter targets. Task 50 reuses and re-evaluates the authenticated offline "
        "model, providing an endpoint identity check. Test labels never influence fitting "
        "or model choice. The locked hierarchy and both joint curves use the same complete "
        "24,000-image training population; the 19,200/4,800 split is confined to the "
        "separate clean-development panel."
    )
    stage_control_estimand_note = (
        "Full-50 joint minus stage-matched joint is a task-horizon contrast within one "
        "joint-training recipe. It bundles later-task examples, their extra global-softmax "
        "competitors, and the additional optimizer updates those examples induce; it is not "
        "a pure causal estimate of semantic ‘future information.’ The compact table shows "
        "task 1, every power-of-two frontier, and task 50. Power-of-two frontiers contain "
        "exactly one hierarchy node, so they remove routing and multi-node score-combination "
        "effects. The figure and machine-readable tables retain all 50 stages."
    )
    oracle_scope_note = (
        "At task 50 the true-node oracle is given the correct one of three hierarchy "
        "nodes and predicts within only 128, 64, or 8 owned classes, whereas the joint "
        "model must choose among all 200 classes. The oracle therefore has an easier "
        "decision problem. Finishing below joint-IID cannot be blamed on routing and is "
        "direct evidence that the retained parent representations or classifier rows are "
        "weaker; it is not evidence that label-aware routing itself is harmful."
    )
    information_headers = (
        "tasks",
        "true-node oracle",
        "joint-IID, stage-matched",
        "joint-IID, trained through task 50",
        "future-data effect (pp)",
        "matched joint − oracle (pp)",
    )
    optimization_headers = (
        "task-32 condition",
        "classifier initialization",
        "weight decay",
        "presentations",
        "optimizer steps",
        "test accuracy (%)",
    )
    abbreviated_condition = lambda value: re.sub(
        r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])",
        lambda match: f"{match.group(0)[:12]}…",
        str(value),
    )
    resource_summary = [
        (
            abbreviated_condition(row.get("condition")),
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
    next_experiment_rows = (
        (
            "One-node parent-recipe factorial",
            "union-head initialization, weight decay, order seed, or training budget",
            "At tasks 16 and 32, vary one factor at a time between the parent and stage-matched recipes.",
            "Attributes the routing-free same-data gap instead of treating it as hierarchy loss.",
        ),
        (
            "Final-frontier interval decomposition",
            "missing transfer from tasks outside each frozen node",
            "Use the completed task-32 decomposition for tasks 1–32; train fresh offset "
            "interval models only for tasks 33–48 and 49–50, then mask full-50 joint to each.",
            "Separates parent-recipe loss from other-task co-training with matched output choices.",
        ),
        (
            "Matched capacity-one/capacity-two rebuild",
            "interval size and consolidation severity",
            "Use identical parent initialization, optimizer, seed schedule, and epochs at both capacities.",
            "Tests whether the observed 2.617-point capacity difference survives recipe matching.",
        ),
        (
            "Replicated optimization sweep",
            "single-seed variance or five-epoch under-training",
            "Repeat decisive controls over fixed seeds and extend epochs with validation-only stopping.",
            "Adds uncertainty and determines whether gaps are stable or optimization noise.",
        ),
    )
    abstract = (
        "We evaluate a capacity-one LogT hierarchy with full-union parent retraining and "
        "a persistent prediction integrator on a fixed ImageNet-R-50 split. The integrator "
        f"reaches {float(locked['last_accuracy']):.3f}% final and "
        f"{float(locked['incremental_accuracy']):.3f}% mean stage accuracy. A new post-hoc curve trains "
        "one fresh rank-16 joint model using exactly the tasks available at each stage, "
        f"reaching {float(matched_incremental):.3f}% mean stage accuracy. Training one joint "
        "model through task 50 and retrospectively restricting its output rows changes the "
        f"mean by {float(dict(locked['local_references'])['joint_iid_incremental']) - float(matched_incremental):+.3f} "
        "points, measuring the combined effect of future-task training under the joint "
        "architecture. Same-stage gaps at one-node frontiers remain, so future information "
        "is only one contributor; optimization path and shared-representation supervision "
        "are independently implicated."
        if locked is not None and matched_incremental is not None
        else "This report evaluates a capacity-one LogT hierarchy and direct prediction integrator on a fixed ImageNet-R-50 split."
    )
    markdown = (
        "# ImageNet-R-50 Full-Union LogT Prediction Integrator\n\n"
        f"Run: `{run.name}`  \nWorkflow state: `{state.get('phase', 'NOT_STARTED')}`\n\n"
        "## Abstract\n\n"
        + abstract
        + "\n\n"
        "This experiment replaces task-free node selection with a direct 200-way residual "
        "integrator over frozen node behavior. Its capacity-one binary counter retrains every "
        "parent on the complete represented training union; only persistent integrator history "
        "uses a bounded replay reservoir. No accuracy or comparator value gates execution.\n\n"
        "## Primary benchmark comparison\n\n"
        + _markdown_table(
            ("condition", "Last (%)", "Mean stage accuracy (%)", "role"),
            comparison_rows,
        )
        + "\n"
        + _markdown_table(
            ("descriptive difference", "Last (pp)", "Mean-stage (pp)"), gap_rows
        )
        + "\nThe task-50 offline joint-IID rank-16 LoRA result is the primary ceiling. "
        "Its earlier prefix evaluations are future-informed references, not stage-matched "
        "ceilings. The local E2-LoRA reproduction is secondary, and published E2-LoRA "
        "values are external context. None is a pass/fail condition.\n\n"
        "## Future-information diagnostic\n\n"
        + stage_control_protocol_note
        + "\n\n"
        + stage_control_estimand_note
        + "\n\n"
        + _markdown_table(information_headers, information_rows)
        + "\n"
        + _markdown_table(
            (
                "fresh models",
                "training presentations",
                "optimizer steps",
                "training minutes",
                "peak VRAM (GiB)",
            ),
            stage_control_work_rows,
        )
        + "\n"
        + _markdown_table(
            (
                "live hierarchy nodes",
                "stages",
                "mean matched joint − oracle (pp)",
                "mean future-data effect (pp)",
            ),
            frontier_gap_rows,
        )
        + "\n"
        + information_conclusion
        + "\n\n### Task-32 recipe evidence\n\n"
        + _markdown_table(optimization_headers, optimization_rows)
        + "\n"
        + optimization_conclusion
        + "\n\n### Hierarchy-compression context\n\n"
        + _markdown_table(
            ("condition", "live adapters/models", "Last (%)", "role"),
            hierarchy_context_rows,
        )
        + "\n"
        + hierarchy_conclusion
        + "\n\n"
        + oracle_scope_note
        + "\n\nThe remaining explanations are architectural and optimization differences, "
        "not routing error: the joint model shares one adapter across every seen class and "
        "receives global negative-class gradients; the oracle chooses among interval-local "
        "adapters whose heads were trained with local softmax objectives. Consolidated "
        "parents inherit unioned child head rows rather than initializing a fresh global "
        "head, and their repeated carry path can land in a different optimum. The hierarchy "
        "also uses zero weight decay while the joint control uses 5e-4. At non-power-of-two "
        "stages, several independent live adapters additionally prevent cross-interval "
        "representation sharing. Node-local absence of later-task transfer remains plausible "
        "at the final fragmented frontier even though benchmark-level future data does not. "
        "A single seed and untuned five-epoch budget leave optimizer variance and under-training "
        "as unresolved contributors.\n\n"
        + "### Discriminating next experiments\n\n"
        + _markdown_table(
            ("experiment", "hypothesis", "comparison", "decisive value"),
            next_experiment_rows,
        )
        + "\nThe adapter-dependent R3 condition remains relevant to the separate "
        "task-free routing gap. A routing-only change cannot strengthen the owned node used "
        "by the true-node diagnostic; an R3 response integrator could nevertheless exceed "
        "that diagnostic if it extracts complementary evidence from non-owning adapters. "
        "The parent-quality and task-free routing questions should therefore be measured as "
        "separate axes.\n\n"
        + "![Joint information diagnostic](joint_information_gap.png)\n\n"
        "## Report-only feature diagnostic\n\n"
        + _markdown_table(("feature family", "mean validation accuracy"), diagnostic_rows)
        + (f"\nConfigured feature family: `{diagnostic.get('selected_variant')}`.\n" if diagnostic else "")
        + "\n## Frozen development choices\n\n"
        + (f"`{json.dumps(selected, sort_keys=True)}`\n\n" if selected else "_Not complete._\n\n")
        + clean_note
        + _markdown_table(tuple(history_headers), history_table_rows)
        + "\n"
        + "Here ‘fresh full-replay integrator’ is a newly initialized three-hidden-layer "
        "residual prediction MLP fit on all prefix observer examples; it is not a newly "
        "trained LoRA adapter. ‘Persistent LogT integrator (H=2048)’ is the same MLP family "
        "warm-started across arrivals with at most 2,048 historical examples. "
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
        ("condition", "Last (%)", "Mean stage accuracy (%)", "role"),
        comparison_rows,
    )
    gap_html = _html_table(
        ("descriptive difference", "Last (pp)", "Mean-stage (pp)"), gap_rows
    )
    information_html = _html_table(information_headers, information_rows)
    optimization_html = _html_table(optimization_headers, optimization_rows)
    stage_control_work_html = _html_table(
        (
            "fresh models",
            "training presentations",
            "optimizer steps",
            "training minutes",
            "peak VRAM (GiB)",
        ),
        stage_control_work_rows,
    )
    frontier_gap_html = _html_table(
        (
            "live hierarchy nodes",
            "stages",
            "mean matched joint − oracle (pp)",
            "mean future-data effect (pp)",
        ),
        frontier_gap_rows,
    )
    hierarchy_context_html = _html_table(
        ("condition", "live adapters/models", "Last (%)", "role"),
        hierarchy_context_rows,
    )
    next_experiment_html = _html_table(
        ("experiment", "hypothesis", "comparison", "decisive value"),
        next_experiment_rows,
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
<style>
body{{font:16px/1.55 system-ui,sans-serif;max-width:1200px;margin:auto;padding:2rem;color:#17202a;background:#fafafa}}
h1,h2,h3{{color:#123b5d}}table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #ccd5dd;padding:.45rem;text-align:left;vertical-align:top;overflow-wrap:anywhere}}
th{{background:#e8f1f8}}details{{background:white;border:1px solid #d9e1e7;border-radius:8px;padding:1rem;margin:1rem 0}}
summary{{font-weight:700;cursor:pointer}}img{{max-width:100%;height:auto}}code{{overflow-wrap:anywhere}}
@page{{size:A4;margin:14mm}}
@media print{{body{{font:10pt/1.4 Georgia,serif;max-width:none;padding:0;background:white}}h1{{font-size:20pt}}h2{{font-size:16pt}}h3{{font-size:12pt}}details{{break-inside:auto;border:0;padding:0}}summary{{font:700 15pt/1.2 Georgia,serif;color:#123b5d;margin:1rem 0 .5rem;list-style:none;break-after:avoid}}summary::-webkit-details-marker{{display:none}}table{{font-size:8.2pt;break-inside:avoid}}thead{{display:table-row-group}}tr{{break-inside:avoid}}img{{break-inside:avoid;max-height:245mm;object-fit:contain}}.raw-data{{display:none}}}}
</style></head>
<body><h1>ImageNet-R-50 Full-Union LogT Prediction Integrator</h1><p><strong>Run:</strong> <code>{escape(run.name)}</code><br><strong>State:</strong> {escape(str(state.get('phase', 'NOT_STARTED')))}</p><h2>Abstract</h2><p>{escape(abstract)}</p>
<p>A direct 200-way residual integrator observes frozen LogT node behavior. Capacity-one parents retrain on every represented training example; only persistent integrator history is bounded. Accuracy never gates execution.</p>
<details open><summary>Primary benchmark comparison</summary>{comparison_html}{gap_html}<p>The task-50 offline joint-IID result is the primary ceiling. Earlier evaluations of that model are future-informed references. Local E2-LoRA is secondary; all differences are descriptive.</p></details>
<details open><summary>Future-information diagnostic</summary><p>{escape(stage_control_protocol_note)}</p><p>{escape(stage_control_estimand_note)}</p>{information_html}{stage_control_work_html}{frontier_gap_html}<p>{escape(information_conclusion)}</p><h3>Task-32 recipe evidence</h3>{optimization_html}<p>{escape(optimization_conclusion)}</p><h3>Hierarchy-compression context</h3>{hierarchy_context_html}<p>{escape(hierarchy_conclusion)}</p><p>{escape(oracle_scope_note)}</p><p>The residual candidate explanations are shared-representation and global-softmax supervision in joint training; union-head initialization and repeated carry optimization in the hierarchy; zero hierarchy weight decay versus 5e-4 in the joint control; frontier fragmentation and node-local missing later-task transfer at non-power-of-two stages; and single-seed or fixed-budget optimization variance.</p><h3>Discriminating next experiments</h3>{next_experiment_html}<p>The adapter-dependent R3 condition targets the separate task-free routing gap. A routing-only change cannot strengthen the owned node used by the true-node diagnostic, although an R3 response integrator could exceed that diagnostic by extracting complementary evidence from non-owning adapters. Parent quality and task-free routing should be measured as separate axes.</p><img alt="Stage-matched joint-IID, future-informed joint-IID, and true-node-oracle curves with their signed gaps" src="{_image_data(information_plot)}"></details>
<details open><summary>Report-only feature diagnostic</summary>{diagnostic_html}<p>Configured feature family: <code>{escape(str(None if diagnostic is None else diagnostic.get('selected_variant')))}</code>.</p></details>
<details open><summary>Frozen development choices</summary><p><code>{escape(json.dumps(selected, sort_keys=True))}</code></p><p>{escape(clean_note.strip())}</p>{history_html}<p>“Fresh full-replay integrator” is a newly initialized three-hidden-layer residual prediction MLP fit on all prefix observer examples; it is not a fresh LoRA adapter. “Persistent LogT integrator (H=2048)” is the same MLP family warm-started with at most 2,048 historical examples. Validation identities are excluded from all clean trainable components.</p></details>
<details open><summary>Final-frontier diagnostics</summary>{static_html}<p>These rows explain the hierarchy frontier and do not define acceptance.</p></details>
<details open><summary>Accuracy</summary><p>{escape(split_note)}</p><img alt="Feature diagnostic, selected validation checkpoints, and locked test curves with stage-matched and future-informed joint controls" src="{_image_data(accuracy_plot)}"></details>
<details open><summary>Capacity-one lineage</summary><img alt="Capacity-one binary-counter lineage" src="{_image_data(lineage)}"></details>
<details open><summary>Complexity and resources</summary><p>There are at most bit_length(t) carries and popcount(t) live nodes. Full-union carry cost grows with interval size; cumulative parent presentations are O(N T log T) for N examples per task. Persistent observer work stays bounded by popcount(t) × (current + H).</p>{resource_html}<pre class="raw-data">{escape(json.dumps({'clean': clean, 'development': development}, indent=2, sort_keys=True))}</pre></details>
</body></html>"""
    atomic_write(reports / "REPORT.html", html.encode("utf-8"))
    write_report_manifest(reports)
    return reports / "REPORT.html"


__all__ = ["write_integrator_report", "write_report_manifest"]

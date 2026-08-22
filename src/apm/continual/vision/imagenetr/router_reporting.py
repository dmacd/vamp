"""Immutable learned-router ledgers plus Markdown and self-contained HTML reports."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
from collections.abc import Iterable, Mapping, Sequence
import json
import math
import shutil
import tempfile

import numpy as np
import pandas as pd

from apm.continual.artifacts import (
    atomic_write,
    canonical_json_bytes,
    file_sha256,
    load_canonical_json,
    publish_immutable_json,
    record_sha256,
)
from apm.continual.vision.imagenetr.artifacts import (
    publish_artifact_directory,
    validate_artifact_directory,
)
from apm.continual.vision.imagenetr.router_evaluation import (
    RouterEvaluation,
    RouterMetricRow,
)


@dataclass(frozen=True, slots=True)
class EvaluationArtifact:
    """Location and identity of one immutable evaluation result."""

    path: Path
    metric: RouterMetricRow
    per_image_path: Path
    reused: bool


def _evaluation_path(run: Path, metric: RouterMetricRow) -> Path:
    return (
        run
        / "evaluations"
        / metric.condition_id
        / metric.split
        / f"stage_{metric.stage:03d}"
    )


def publish_router_evaluation(run: Path, result: RouterEvaluation) -> EvaluationArtifact:
    """Atomically publish metric JSON and per-image CSV/Parquet evidence."""
    target = _evaluation_path(run, result.metric)
    if target.is_dir():
        validate_artifact_directory(target)
        loaded = load_router_evaluation(target)
        if loaded.metric != result.metric or loaded.per_image != result.per_image:
            raise ValueError("immutable router evaluation collision")
        return EvaluationArtifact(target, loaded.metric, target / "per_image.parquet", True)
    work_root = run / "work"
    work_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="router-evaluation.", dir=work_root))
    try:
        metric_core: dict[str, object] = {
            **result.metric.as_record(),
            "schema_version": "imagenetr50-router-metric-v1",
        }
        publish_immutable_json(
            work / "metric.json",
            {**metric_core, "content_hash": record_sha256(metric_core)},
        )
        frame = pd.DataFrame(result.per_image)
        frame.to_csv(work / "per_image.csv", index=False)
        frame.to_parquet(work / "per_image.parquet", index=False)
        evidence_core: dict[str, object] = {
            "csv_sha256": file_sha256(work / "per_image.csv"),
            "examples": len(frame),
            "parquet_sha256": file_sha256(work / "per_image.parquet"),
            "schema_version": "imagenetr50-router-per-image-evidence-v1",
        }
        publish_immutable_json(
            work / "evidence.json",
            {**evidence_core, "content_hash": record_sha256(evidence_core)},
        )
        publish_artifact_directory(work, target)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return EvaluationArtifact(target, result.metric, target / "per_image.parquet", False)


def load_router_evaluation(path: str | Path) -> RouterEvaluation:
    """Load and validate one immutable router evaluation directory."""
    root = Path(path)
    validate_artifact_directory(root)
    raw = load_canonical_json(root / "metric.json")
    core = {key: value for key, value in raw.items() if key != "content_hash"}
    if raw.get("content_hash") != record_sha256(core):
        raise ValueError("router metric record changed")
    schema = core.pop("schema_version")
    if schema != "imagenetr50-router-metric-v1":
        raise ValueError("unknown router metric schema")
    metric = RouterMetricRow(**core)
    evidence = load_canonical_json(root / "evidence.json")
    evidence_core = {key: value for key, value in evidence.items() if key != "content_hash"}
    if (
        evidence.get("content_hash") != record_sha256(evidence_core)
        or evidence["csv_sha256"] != file_sha256(root / "per_image.csv")
        or evidence["parquet_sha256"] != file_sha256(root / "per_image.parquet")
    ):
        raise ValueError("router per-image evidence changed")
    frame = pd.read_parquet(root / "per_image.parquet")
    records = tuple(frame.to_dict(orient="records"))
    if len(records) != metric.examples:
        raise ValueError("router metric and per-image evidence row counts differ")
    return RouterEvaluation(metric, records)


def discover_evaluations(run: Path) -> tuple[EvaluationArtifact, ...]:
    """Discover every complete evaluation while rejecting partial result paths."""
    result = []
    root = run / "evaluations"
    if not root.is_dir():
        return ()
    for metric_path in sorted(root.glob("*/*/stage_*/metric.json")):
        directory = metric_path.parent
        loaded = load_router_evaluation(directory)
        result.append(
            EvaluationArtifact(
                directory,
                loaded.metric,
                directory / "per_image.parquet",
                True,
            )
        )
    return tuple(result)


def _bootstrap_interval(values: np.ndarray, seed: int, samples: int = 2000) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    generator = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for offset in range(0, samples, 100):
        count = min(100, samples - offset)
        indices = generator.integers(0, values.size, size=(count, values.size))
        means[offset : offset + count] = values[indices].mean(axis=1)
    lower, upper = np.quantile(means, (0.025, 0.975))
    return 100.0 * float(lower), 100.0 * float(upper)


def _existing_baselines(run: Path) -> dict[str, float]:
    link_path = run / "protocol" / "protocol_link.json"
    if not link_path.is_file():
        return {}
    link = load_canonical_json(link_path)
    sealed = (
        run.parents[2]
        / "runs"
        / str(link["sealed_run_hash"])
        / "reports"
        / "summary.json"
    )
    if not sealed.is_file():
        return {}
    summary = load_canonical_json(sealed)
    names = {
        "I-U100": "logt_retrain_union_r16",
        "I-SVD0": "logt_svd_r16_repair000",
        "I-SVD5": "logt_svd_r16_repair005",
    }
    result = {}
    for inference, condition in names.items():
        row = next(
            (value for value in summary["conditions"] if value["condition"] == condition),
            None,
        )
        if row is not None:
            result[inference] = max(
                float(row[f"{mode}_last_accuracy"])
                for mode in ("raw", "cosine", "affine_calibrated", "centroid_router")
            )
    return result


def _metric_frame(run: Path, artifacts: Sequence[EvaluationArtifact]) -> pd.DataFrame:
    rows = []
    for artifact in artifacts:
        row = artifact.metric.as_record()
        evidence = pd.read_parquet(
            artifact.per_image_path, columns=["routed_correct"]
        )["routed_correct"].to_numpy(dtype=np.float64)
        lower, upper = _bootstrap_interval(
            evidence,
            int(record_sha256(row)[:15], 16),
        )
        rows.append({**row, "routed_ci95_lower": lower, "routed_ci95_upper": upper})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    baselines = _existing_baselines(run)
    frame["existing_task_free_accuracy"] = frame["inference_condition"].map(baselines)
    frame["gain_over_existing"] = (
        frame["routed_accuracy"] - frame["existing_task_free_accuracy"]
    )
    denominator = frame["oracle_accuracy"] - frame["existing_task_free_accuracy"]
    frame["oracle_gain_recovery"] = np.where(
        denominator > 0,
        (frame["routed_accuracy"] - frame["existing_task_free_accuracy"])
        / denominator,
        np.nan,
    )
    exact = frame[frame["maintenance"] == "exact"][
        [
            "architecture",
            "inference_condition",
            "router_seed",
            "split",
            "stage",
            "routed_accuracy",
        ]
    ].rename(columns={"routed_accuracy": "exact_routed_accuracy"})
    frame = frame.merge(
        exact,
        on=["architecture", "inference_condition", "router_seed", "split", "stage"],
        how="left",
    )
    frame["merge_regret_to_exact"] = (
        frame["exact_routed_accuracy"] - frame["routed_accuracy"]
    )
    return frame


def _task_frame(artifacts: Sequence[EvaluationArtifact]) -> pd.DataFrame:
    rows = []
    for artifact in artifacts:
        frame = pd.read_parquet(
            artifact.per_image_path,
            columns=["task", "routed_correct", "selection_correct", "oracle_correct"],
        )
        for task, group in frame.groupby("task", sort=True):
            rows.append(
                {
                    "condition_id": artifact.metric.condition_id,
                    "inference_condition": artifact.metric.inference_condition,
                    "architecture": artifact.metric.architecture,
                    "maintenance": artifact.metric.maintenance,
                    "router_seed": artifact.metric.router_seed,
                    "split": artifact.metric.split,
                    "stage": artifact.metric.stage,
                    "task": int(task),
                    "examples": len(group),
                    "routed_accuracy": 100.0 * float(group.routed_correct.mean()),
                    "oracle_accuracy": 100.0 * float(group.oracle_correct.mean()),
                    "selection_accuracy": 100.0 * float(group.selection_correct.mean()),
                }
            )
    return pd.DataFrame(rows)


def _paired_frame(artifacts: Sequence[EvaluationArtifact]) -> pd.DataFrame:
    by_key: dict[tuple[object, ...], dict[str, EvaluationArtifact]] = {}
    for artifact in artifacts:
        metric = artifact.metric
        if metric.architecture not in {"r1", "r3"}:
            continue
        key = (
            metric.inference_condition,
            metric.maintenance,
            metric.router_seed,
            metric.split,
            metric.stage,
        )
        by_key.setdefault(key, {})[metric.architecture] = artifact
    rows = []
    for key, pair in sorted(by_key.items()):
        if set(pair) != {"r1", "r3"}:
            continue
        left = pd.read_parquet(
            pair["r1"].per_image_path,
            columns=["image_id", "routed_correct"],
        ).rename(columns={"routed_correct": "r1_correct"})
        right = pd.read_parquet(
            pair["r3"].per_image_path,
            columns=["image_id", "routed_correct"],
        ).rename(columns={"routed_correct": "r3_correct"})
        joined = left.merge(right, on="image_id", validate="one_to_one")
        delta = joined["r3_correct"].to_numpy(dtype=np.float64) - joined[
            "r1_correct"
        ].to_numpy(dtype=np.float64)
        lower, upper = _bootstrap_interval(
            delta, int(record_sha256(list(key))[:15], 16)
        )
        rows.append(
            {
                "inference_condition": key[0],
                "maintenance": key[1],
                "router_seed": key[2],
                "split": key[3],
                "stage": key[4],
                "examples": len(joined),
                "r3_minus_r1_accuracy": 100.0 * float(delta.mean()),
                "paired_ci95_lower": lower,
                "paired_ci95_upper": upper,
                "r1_only_correct": int((joined.r1_correct & ~joined.r3_correct).sum()),
                "r3_only_correct": int((joined.r3_correct & ~joined.r1_correct).sum()),
            }
        )
    return pd.DataFrame(rows)


def _merge_frame(run: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((run / "recursive_runs").glob("*/diagnostics/merge_*.json")):
        record = load_canonical_json(path)
        core = {key: value for key, value in record.items() if key != "content_hash"}
        if record.get("content_hash") != record_sha256(core):
            raise ValueError("router merge diagnostic changed during reporting")
        rows.append(
            {
                "policy_hash": record["policy_hash"],
                **record["event"],
                **{f"functional_{key}": value for key, value in record["functional"].items()},
                **{
                    f"parameter_{key}": value
                    for key, value in (record.get("parameter") or {}).items()
                },
            }
        )
    return pd.DataFrame(rows)


def _resource_frame(run: Path) -> pd.DataFrame:
    from safetensors import safe_open

    rows = []
    for policy_path in sorted((run / "recursive_runs").glob("*/policy.json")):
        policy = load_canonical_json(policy_path)
        snapshots = sorted((policy_path.parent / "snapshots").glob("stage_*.json"))
        if not snapshots:
            continue
        final = load_canonical_json(snapshots[-1])
        learned_parameters = 0
        learned_bytes = 0
        response_bytes = 0
        for logical_id in final["logical_node_ids"]:
            node_root = policy_path.parent / "nodes" / logical_id
            with safe_open(node_root / "scorer.safetensors", framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    tensor = handle.get_tensor(key)
                    learned_parameters += tensor.numel()
                    learned_bytes += tensor.numel() * tensor.element_size()
            node = load_canonical_json(node_root / "node.json")
            response_hash = node.get("response_kernel_sha256")
            if response_hash:
                matches = tuple(
                    path
                    for path in (run / "response_kernels").glob("*/tensors.safetensors")
                    if file_sha256(path) == response_hash
                )
                if len(matches) != 1:
                    raise ValueError("live R3 response kernel identity is ambiguous")
                response_bytes += matches[0].stat().st_size
        rows.append(
            {
                "architecture": policy["architecture"],
                "activation_cache_bytes": sum(
                    path.stat().st_size
                    for path in (run / "features" / "cls_activations").glob(
                        "*/tensors.safetensors"
                    )
                ),
                "candidate_adapted_forwards": 0,
                "condition": policy["inference_condition"],
                "final_live_nodes": len(final["logical_node_ids"]),
                "inference_parent_optimizer_steps": final[
                    "inference_parent_optimizer_steps"
                ],
                "leaf_optimizer_steps": final["leaf_optimizer_steps"],
                "learned_router_bytes": learned_bytes,
                "learned_router_parameters": learned_parameters,
                "maintenance": policy["maintenance"],
                "policy_hash": policy["content_hash"],
                "response_kernel_bytes": response_bytes,
                "router_optimizer_steps": final[
                    "cumulative_router_optimizer_steps"
                ],
                "router_seed": policy["router_seed"],
            }
        )
    return pd.DataFrame(rows)


def _write_frame(frame: pd.DataFrame, root: Path, stem: str) -> None:
    frame.to_csv(root / f"{stem}.csv", index=False)
    frame.to_parquet(root / f"{stem}.parquet", index=False)
    atomic_write(
        root / f"{stem}.json",
        canonical_json_bytes(
            {
                "rows": frame.astype(object)
                .where(pd.notna(frame), None)
                .to_dict(orient="records"),
                "schema_version": f"imagenetr50-router-{stem}-v1",
            }
        ),
    )


def _lineage_svg(run: Path) -> str:
    snapshots = sorted((run / "recursive_runs").glob("*/snapshots/stage_050.json"))
    if not snapshots:
        return '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="80"></svg>'
    record = load_canonical_json(snapshots[0])
    width, height = 960, 140
    bars = []
    for index, logical_id in enumerate(record["logical_node_ids"]):
        node = load_canonical_json(snapshots[0].parents[1] / "nodes" / logical_id / "node.json")
        first, last = min(node["represented_task_ids"]), max(node["represented_task_ids"])
        x = 20 + first * 18
        bar_width = max(14, (last - first + 1) * 18 - 3)
        color = "#3568a8" if index % 2 == 0 else "#65a5d8"
        bars.append(
            f'<rect x="{x}" y="35" width="{bar_width}" height="42" rx="4" fill="{color}"/>'
            f'<text x="{x + bar_width / 2:.1f}" y="61" text-anchor="middle" fill="white" font-size="11">'
            f'{first + 1}-{last + 1}</text>'
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        '<text x="20" y="20" font-family="sans-serif" font-size="14">Final capacity-two router frontier</text>'
        + "".join(bars)
        + '<text x="20" y="105" font-family="sans-serif" font-size="11">Tasks 1–50; bars are live contiguous inference/router nodes.</text>'
        + "</svg>"
    )


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    if frame.empty:
        return "_No completed rows._"
    values = frame.loc[:, list(columns)]
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for row in values.itertuples(index=False, name=None):
        formatted = []
        for value in row:
            if isinstance(value, float):
                formatted.append("—" if math.isnan(value) else f"{value:.3f}")
            else:
                formatted.append(str(value))
        rows.append("| " + " | ".join(formatted) + " |")
    return "\n".join((header, rule, *rows))


def write_router_report(
    run: Path,
    *,
    status: Mapping[str, object] | None = None,
    title: str = "ImageNet-R-50 Recursive Learned Router Oracle Recovery",
) -> Path:
    """Regenerate complete machine-readable ledgers and human reports."""
    reports = run / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    artifacts = discover_evaluations(run)
    metrics = _metric_frame(run, artifacts)
    tasks = _task_frame(artifacts)
    paired = _paired_frame(artifacts)
    merges = _merge_frame(run)
    resources = _resource_frame(run)
    for frame, stem in (
        (metrics, "stage_metrics"),
        (tasks, "task_accuracy_matrix"),
        (paired, "paired_r3_minus_r1"),
        (merges, "merge_diagnostics"),
        (resources, "resource_accounting"),
    ):
        _write_frame(frame, reports, stem)
    final = metrics[
        (metrics.get("split") == "test") & (metrics.get("stage") == 50)
    ] if not metrics.empty else metrics
    if not final.empty:
        final = final.sort_values(["condition_id", "router_seed"])
    status_text = json.dumps(dict(status or {}), indent=2, sort_keys=True)
    report = (
        f"# {title}\n\n"
        f"Router protocol: `{run.name}`\n\n"
        "The sealed inference run is a read-only dependency. Learned-router jobs report "
        "zero leaf and zero inference-parent optimizer steps; R3 is a predeclared main "
        "architecture, paired with R1 rather than activated after seeing test results.\n\n"
        "## Final test results\n\n"
        + _markdown_table(
            final,
            (
                "condition_id",
                "inference_condition",
                "architecture",
                "maintenance",
                "router_seed",
                "routed_accuracy",
                "oracle_accuracy",
                "oracle_gap",
                "selection_accuracy",
                "top2_selection_accuracy",
            ),
        )
        + "\n\n## Paired R3 minus R1\n\n"
        + _markdown_table(
            paired[paired["stage"] == 50] if not paired.empty else paired,
            (
                "inference_condition",
                "maintenance",
                "router_seed",
                "split",
                "r3_minus_r1_accuracy",
                "paired_ci95_lower",
                "paired_ci95_upper",
                "r1_only_correct",
                "r3_only_correct",
            ),
        )
        + "\n\n## Resource accounting\n\n"
        + _markdown_table(
            resources,
            (
                "condition",
                "architecture",
                "maintenance",
                "router_seed",
                "final_live_nodes",
                "learned_router_parameters",
                "response_kernel_bytes",
                "router_optimizer_steps",
                "leaf_optimizer_steps",
                "inference_parent_optimizer_steps",
                "candidate_adapted_forwards",
            ),
        )
        + "\n\n## Durable status\n\n```json\n"
        + status_text
        + "\n```\n"
    )
    atomic_write(reports / "REPORT.md", report.encode("utf-8"))
    svg = _lineage_svg(run)
    atomic_write(reports / "lineage.svg", svg.encode("utf-8"))
    html = (
        "<!doctype html><html><head><meta charset='utf-8'><title>"
        + escape(title)
        + "</title><style>body{font:15px system-ui;max-width:1200px;margin:2rem auto;padding:0 1rem;color:#18212b}"
        "table{border-collapse:collapse;width:100%;font-size:12px}th,td{border:1px solid #ccd5df;padding:.35rem;text-align:right}"
        "th:first-child,td:first-child{text-align:left}h1,h2{color:#173f6b}code,pre{background:#f3f6f9;padding:.2rem}.note{padding:1rem;background:#eef5fb}</style></head><body>"
        f"<h1>{escape(title)}</h1><p class='note'>Protocol <code>{escape(run.name)}</code>. "
        "R3 is a mandatory paired architecture. Test caches are sealed read-only inputs.</p>"
        "<h2>Final test results</h2>"
        + (final.to_html(index=False, float_format=lambda value: f"{value:.3f}") if not final.empty else "<p>No completed test rows.</p>")
        + "<h2>Paired R3 minus R1</h2>"
        + (paired.to_html(index=False, float_format=lambda value: f"{value:.3f}") if not paired.empty else "<p>No complete pairs.</p>")
        + "<h2>Lineage</h2>"
        + svg
        + "<h2>Resource accounting</h2>"
        + (resources.to_html(index=False) if not resources.empty else "<p>No completed recursive resources.</p>")
        + "<h2>Durable status</h2><pre>"
        + escape(status_text)
        + "</pre></body></html>"
    )
    atomic_write(reports / "REPORT.html", html.encode("utf-8"))
    manifest_core: dict[str, object] = {
        "files": [
            {
                "path": path.name,
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(reports.iterdir())
            if path.is_file() and path.name != "report_manifest.json"
        ],
        "schema_version": "imagenetr50-router-report-manifest-v1",
    }
    atomic_write(
        reports / "report_manifest.json",
        canonical_json_bytes(
            {**manifest_core, "content_hash": record_sha256(manifest_core)}
        ),
    )
    return reports / "REPORT.md"


__all__ = [
    "EvaluationArtifact",
    "discover_evaluations",
    "load_router_evaluation",
    "publish_router_evaluation",
    "write_router_report",
]

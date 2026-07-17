"""Deterministic standalone artifacts for language continual-learning reports."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import html
import json
import math
from pathlib import Path
import re
from typing import TypeAlias

import numpy as np

from apm.continual.language_benchmarks import (
    AddressingCoefficientTrace,
    GeneratedLanguageSample,
    ROUTER_BASELINE_NAMES,
    STORED_BASELINE_NAMES,
)
from apm.memory.graph import MemoryGraph, memory_node_ids
from apm.memory.visualization import (
    EdgeVisualStats,
    NodeVisualStats,
    write_memory_graph_svg,
)
from apm.training.artifacts import (
    report_lightbox_css,
    report_lightbox_markup,
    report_lightbox_script,
    write_json,
    write_svg_heatmap,
    write_svg_line_chart,
)

ReportScalar: TypeAlias = str | int | float | bool | None
_PATH_COMPONENT = re.compile(r"[a-z0-9][a-z0-9_-]*\Z")
_CONFIG_HASH_LENGTH = 12
_EBT_TRACE_PRESENTATION = (
    ("vamp_ebt_uniform", "ebt_uniform", "EBT uniform start"),
    ("vamp_ebt_hopfield", "ebt_hopfield", "EBT Hopfield start"),
)


@dataclass(frozen=True)
class ReportRecord:
    """One immutable, scalar-only JSON record with unique field names."""

    entries: tuple[tuple[str, ReportScalar], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.entries, tuple) or any(
            not isinstance(entry, tuple) or len(entry) != 2
            for entry in self.entries
        ):
            raise TypeError("report record entries must be immutable key/value tuples")
        keys = tuple(key for key, _ in self.entries)
        if not keys or any(not isinstance(key, str) or not key for key in keys):
            raise ValueError("report records require nonempty string field names")
        if len(set(keys)) != len(keys):
            raise ValueError("report record field names must be unique")
        if any(not _valid_report_scalar(value) for _, value in self.entries):
            raise TypeError("report record values must be finite JSON scalars")

    def as_dict(self) -> dict[str, ReportScalar]:
        """Return a fresh JSON-compatible dictionary in record order."""
        return dict(self.entries)

    def require(self, field_name: str) -> ReportScalar:
        """Return one required field or raise a descriptive error."""
        values = tuple(value for key, value in self.entries if key == field_name)
        if not values:
            raise ValueError(f"report record is missing required field: {field_name}")
        return values[0]


@dataclass(frozen=True)
class LanguageReportManifest:
    """Stable experiment identity and canonical nested JSON configuration."""

    dataset: str
    curriculum: str
    preset: str
    seed: int
    config_json: str

    def __post_init__(self) -> None:
        for field_name in ("dataset", "curriculum", "preset"):
            _validate_path_component(getattr(self, field_name), field_name)
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        _canonical_config(self.config_json)

    @property
    def config_hash(self) -> str:
        """Return the full SHA-256 of the canonical configuration JSON."""
        return sha256(_canonical_config(self.config_json).encode("utf-8")).hexdigest()

    @property
    def run_id(self) -> str:
        """Return the prescribed preset, seed, and configuration-hash identity."""
        return f"{self.preset}-seed{self.seed}-{self.config_hash[:_CONFIG_HASH_LENGTH]}"


@dataclass(frozen=True, eq=False)
class AddressConfusion:
    """Labeled square address-confusion counts for one report."""

    labels: tuple[str, ...]
    counts: np.ndarray

    def __post_init__(self) -> None:
        counts = np.asarray(self.counts)
        if not self.labels or len(set(self.labels)) != len(self.labels):
            raise ValueError("confusion labels must be nonempty and unique")
        if any(not label for label in self.labels):
            raise ValueError("confusion labels must not be empty")
        if counts.shape != (len(self.labels), len(self.labels)):
            raise ValueError("confusion counts must be square over the labels")
        if counts.dtype.kind not in "iu" or np.any(counts < 0):
            raise ValueError("confusion counts must be nonnegative integers")
        immutable_counts = np.array(counts, dtype=np.int64, copy=True)
        immutable_counts.flags.writeable = False
        object.__setattr__(self, "counts", immutable_counts)


@dataclass(frozen=True)
class LanguageReportBundle:
    """Complete immutable inputs needed to emit one language report."""

    manifest: LanguageReportManifest
    stage_metrics: tuple[ReportRecord, ...]
    stored_competence: tuple[ReportRecord, ...]
    routing_metrics: tuple[ReportRecord, ...]
    transfer_metrics: tuple[ReportRecord, ...]
    memory_metrics: tuple[ReportRecord, ...]
    addressing_cost: tuple[ReportRecord, ...]
    competence_curve: tuple[ReportRecord, ...]
    routing_curve: tuple[ReportRecord, ...]
    memory_curve: tuple[ReportRecord, ...]
    addressing_traces: tuple[AddressingCoefficientTrace, ...]
    address_confusion: AddressConfusion
    graph: MemoryGraph[object]
    node_stats: tuple[NodeVisualStats, ...]
    edge_stats: tuple[EdgeVisualStats, ...]
    samples: tuple[GeneratedLanguageSample, ...]

    def __post_init__(self) -> None:
        _validate_report_bundle(self)


def canonical_config_json(config: dict[str, object]) -> str:
    """Serialize one nested experiment configuration in stable hash form."""
    if not isinstance(config, dict) or not config:
        raise ValueError("report configuration must be a nonempty JSON object")
    try:
        serialized = json.dumps(
            config,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("report configuration must contain only JSON values") from error
    return _canonical_config(serialized)


def language_report_directory(
    results_root: str | Path,
    manifest: LanguageReportManifest,
) -> Path:
    """Return the standard dataset/curriculum/run report directory."""
    return (
        Path(results_root)
        / "language_cl"
        / manifest.dataset
        / manifest.curriculum
        / manifest.run_id
    )


def write_language_report(
    results_root: str | Path,
    bundle: LanguageReportBundle,
) -> Path:
    """Write every standard language artifact idempotently and return its directory."""
    _validate_report_bundle(bundle)
    output_directory = language_report_directory(results_root, bundle.manifest)
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_payload = {
        "schema_version": 2,
        "dataset": bundle.manifest.dataset,
        "curriculum": bundle.manifest.curriculum,
        "preset": bundle.manifest.preset,
        "seed": bundle.manifest.seed,
        "run_id": bundle.manifest.run_id,
        "config_hash": bundle.manifest.config_hash,
        "config": json.loads(_canonical_config(bundle.manifest.config_json)),
        "stored_baselines": list(STORED_BASELINE_NAMES),
        "routers": list(ROUTER_BASELINE_NAMES),
        "interpretation": "in-domain continual adaptation",
    }
    write_json(output_directory / "manifest.json", manifest_payload)
    for filename, records in _jsonl_families(bundle):
        _write_jsonl_replacing(output_directory / filename, records)

    write_svg_heatmap(
        output_directory / "address_confusion.svg",
        bundle.address_confusion.counts.astype(np.float32),
        bundle.address_confusion.labels,
        bundle.address_confusion.labels,
        "Task-free address confusion",
        value_format=".0f",
    )
    _write_curve(
        output_directory / "competence_curves.svg",
        bundle.competence_curve,
        "Stored competence",
        "suffix NLL",
    )
    _write_curve(
        output_directory / "routing_curves.svg",
        bundle.routing_curve,
        "Task-free routing",
        "routing metric",
    )
    _write_curve(
        output_directory / "memory_curves.svg",
        bundle.memory_curve,
        "Memory accounting",
        "bytes",
    )
    _write_addressing_trace_charts(
        output_directory,
        bundle.addressing_traces,
    )
    write_memory_graph_svg(
        output_directory / "graph.svg",
        bundle.graph,
        {stats.node_id: stats for stats in bundle.node_stats},
        {
            (stats.parent_id, stats.child_id): stats
            for stats in bundle.edge_stats
        },
        "Language VAMP memory graph",
    )
    _write_samples(output_directory / "samples.md", bundle.samples)
    _write_language_html(output_directory / "report.html", bundle)
    return output_directory


def _jsonl_families(
    bundle: LanguageReportBundle,
) -> tuple[tuple[str, tuple[ReportRecord, ...]], ...]:
    return (
        ("stage_metrics.jsonl", bundle.stage_metrics),
        ("stored_competence.jsonl", bundle.stored_competence),
        ("routing_metrics.jsonl", bundle.routing_metrics),
        ("transfer_metrics.jsonl", bundle.transfer_metrics),
        ("memory_metrics.jsonl", bundle.memory_metrics),
        ("addressing_cost.jsonl", bundle.addressing_cost),
        (
            "addressing_trace.jsonl",
            _addressing_trace_records(bundle.addressing_traces),
        ),
    )


def _addressing_trace_records(
    traces: tuple[AddressingCoefficientTrace, ...],
) -> tuple[ReportRecord, ...]:
    return tuple(
        ReportRecord(
            (
                ("router", trace.router),
                ("task_id", trace.task_id),
                ("prefix_length", trace.prefix_length),
                ("example_index", trace.example_index),
                ("step", step),
                ("objective", float(trace.objective_trace[step])),
                ("coefficient_type", coefficient_type),
                ("coefficient_label", label),
                ("coefficient", float(values[step, coefficient_index])),
            )
        )
        for trace in traces
        for step in range(trace.objective_trace.size)
        for coefficient_type, labels, values in (
            ("node_probability", trace.node_labels, trace.node_probabilities),
            ("path_edge", trace.edge_labels, trace.edge_coefficients),
        )
        for coefficient_index, label in enumerate(labels)
    )


def _write_jsonl_replacing(path: Path, records: tuple[ReportRecord, ...]) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        "".join(
            json.dumps(
                record.as_dict(),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _write_curve(
    path: Path,
    records: tuple[ReportRecord, ...],
    title: str,
    y_label: str,
) -> None:
    rows = tuple(_curve_row(record) for record in records)
    series_keys = tuple(
        key
        for key, value in records[0].entries
        if key != "stage" and isinstance(value, (int, float)) and not isinstance(value, bool)
    )
    write_svg_line_chart(
        path,
        rows,
        tuple((key, key.replace("_", " ")) for key in series_keys),
        title,
        y_label,
        x_label="stage",
    )


def _write_addressing_trace_charts(
    output_directory: Path,
    traces: tuple[AddressingCoefficientTrace, ...],
) -> None:
    trace_by_router = {trace.router: trace for trace in traces}
    for router, filename_stem, title in _EBT_TRACE_PRESENTATION:
        trace = trace_by_router[router]
        node_index_by_label = {
            label: index for index, label in enumerate(trace.node_labels)
        }
        node_display_labels = tuple(
            f"n{index} {label.rsplit('-', 1)[-1]}"
            for index, label in enumerate(trace.node_labels)
        )
        edge_display_labels = tuple(
            f"e{index} n{node_index_by_label[label.split(' → ', 1)[0]]}"
            f"→n{node_index_by_label[label.split(' → ', 1)[1]]}"
            for index, label in enumerate(trace.edge_labels)
        )
        step_labels = tuple(
            f"step {step}" for step in range(trace.objective_trace.size)
        )
        write_svg_heatmap(
            output_directory / f"{filename_stem}_node_coefficients.svg",
            trace.node_probabilities,
            step_labels,
            node_display_labels,
            f"{title}: node address probabilities",
        )
        write_svg_heatmap(
            output_directory / f"{filename_stem}_edge_coefficients.svg",
            trace.edge_coefficients,
            step_labels,
            edge_display_labels,
            f"{title}: path-edge coefficients",
        )
    step_count = traces[0].objective_trace.size
    write_svg_line_chart(
        output_directory / "ebt_objective_trace.svg",
        tuple(
            {
                "epoch": step,
                **{
                    router: float(trace_by_router[router].objective_trace[step])
                    for router, _, _ in _EBT_TRACE_PRESENTATION
                },
            }
            for step in range(step_count)
        ),
        tuple(
            (router, title)
            for router, _, title in _EBT_TRACE_PRESENTATION
        ),
        "EBT routing objective by refinement step",
        "prefix NLL + entropy penalty",
        x_label="refinement step",
    )


def _curve_row(record: ReportRecord) -> dict[str, int | float]:
    stage = record.require("stage")
    if type(stage) is not int or stage < 0:
        raise ValueError("curve stage must be a nonnegative integer")
    numeric_values = {
        key: value
        for key, value in record.entries
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    return {"epoch": stage, **numeric_values}


def _write_samples(path: Path, samples: tuple[GeneratedLanguageSample, ...]) -> None:
    content = ["# Generated language samples", ""]
    for sample in samples:
        content.extend(
            (
                f"## {sample.baseline} · {sample.task_id}",
                "",
                "**Prefix**",
                "",
                sample.prefix,
                "",
                "**Continuation**",
                "",
                sample.continuation,
                "",
            )
        )
    path.write_text("\n".join(content).rstrip() + "\n", encoding="utf-8")


def _write_language_html(path: Path, bundle: LanguageReportBundle) -> None:
    metric_images = (
        ("Memory graph", "graph.svg"),
        ("Address confusion", "address_confusion.svg"),
        ("Competence curves", "competence_curves.svg"),
        ("Routing curves", "routing_curves.svg"),
        ("Memory curves", "memory_curves.svg"),
    )
    image_cards = "\n".join(
        '<figure class="image-card" role="button" tabindex="0" '
        f'data-lightbox-src="{filename}" data-lightbox-caption="{html.escape(title)}">'
        f'<img src="{filename}" alt="{html.escape(title)}">'
        f"<figcaption>{html.escape(title)}</figcaption></figure>"
        for title, filename in metric_images
    )
    trace_images = (
        ("EBT routing objective", "ebt_objective_trace.svg"),
        *tuple(
            (f"{title}: node probabilities", f"{filename_stem}_node_coefficients.svg")
            for _, filename_stem, title in _EBT_TRACE_PRESENTATION
        ),
        *tuple(
            (f"{title}: path-edge coefficients", f"{filename_stem}_edge_coefficients.svg")
            for _, filename_stem, title in _EBT_TRACE_PRESENTATION
        ),
    )
    trace_image_cards = "\n".join(
        '<figure class="image-card" role="button" tabindex="0" '
        f'data-lightbox-src="{filename}" data-lightbox-caption="{html.escape(title)}">'
        f'<img src="{filename}" alt="{html.escape(title)}">'
        f"<figcaption>{html.escape(title)}</figcaption></figure>"
        for title, filename in trace_images
    )
    representative_trace = bundle.addressing_traces[0]
    trace_context = (
        f"These traces use deterministic test example "
        f"{representative_trace.example_index} from task "
        f"{representative_trace.task_id} at prefix length "
        f"{representative_trace.prefix_length}. Step 0 is the initial address; "
        "later rows show each Adam update. Node probabilities induce the "
        "path-edge coefficients applied to the LoRA memory. The n/e labels "
        "follow graph insertion order; the raw JSONL retains full identities."
    )
    sample_cards = "\n".join(
        "<article class=\"sample\">"
        f"<h3>{html.escape(sample.baseline)} · {html.escape(sample.task_id)}</h3>"
        f"<p><strong>Prefix:</strong> {html.escape(sample.prefix)}</p>"
        f"<p><strong>Continuation:</strong> {html.escape(sample.continuation)}</p>"
        "</article>"
        for sample in bundle.samples
    )
    baseline_rows = "\n".join(
        f"<tr><td>{html.escape(name)}</td><td>{'stored' if name in STORED_BASELINE_NAMES else 'router'}</td></tr>"
        for name in (*STORED_BASELINE_NAMES, *ROUTER_BASELINE_NAMES)
    )
    config_pre = html.escape(
        json.dumps(
            json.loads(bundle.manifest.config_json),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    negative_control_rows = tuple(
        record
        for record in bundle.routing_metrics
        if record.require("router") == "deterministic_random_node"
    )
    audited_rows = sum(
        record.as_dict().get("leakage_audit_required") is True
        for record in negative_control_rows
    )
    covered_rows = sum(
        record.as_dict().get("negative_control_chance_in_ci95") is True
        for record in negative_control_rows
    )
    negative_control_status = (
        f"{covered_rows}/{len(negative_control_rows)} random-control rows have "
        "95% Wilson intervals containing their stage-wise chance rate; "
        f"{audited_rows} row(s) require a leakage audit."
    )
    report = "\n".join(
        (
            "<!doctype html>",
            '<html lang="en"><head><meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>{html.escape(bundle.manifest.run_id)}</title>",
            f"<style>{_language_report_css()}</style></head><body><main>",
            f"<h1>{html.escape(bundle.manifest.dataset)} · {html.escape(bundle.manifest.curriculum)}</h1>",
            f"<p><code>{html.escape(bundle.manifest.run_id)}</code></p>",
            "<p>This report measures in-domain continual adaptation with prefix-only task-free routing and disjoint suffix competence.</p>",
            "<section><h2>Baseline matrix</h2><table><thead><tr><th>Method</th><th>Role</th></tr></thead>",
            f"<tbody>{baseline_rows}</tbody></table><p>{negative_control_status}</p></section>",
            f'<section><h2>Graph and metrics</h2><div class="image-grid">{image_cards}</div></section>',
            f'<section><h2>EBT routing dynamics</h2><p>{html.escape(trace_context)}</p><p><a href="addressing_trace.jsonl">Raw coefficient trace</a></p><div class="image-grid">{trace_image_cards}</div></section>',
            f'<section><h2>Generated samples</h2><p><a href="samples.md">Raw sample record</a></p>{sample_cards}</section>',
            f"<section><h2>Configuration</h2><pre>{config_pre}</pre></section>",
            "</main>",
            report_lightbox_markup(),
            f"<script>{report_lightbox_script()}</script>",
            "</body></html>",
        )
    )
    path.write_text(report + "\n", encoding="utf-8")


def _language_report_css() -> str:
    return (
        "body{margin:0;background:#f8fafc;color:#111827;font:15px/1.45 Inter,Arial,sans-serif}"
        "main{max-width:1180px;margin:auto;padding:32px 24px 56px}"
        "h1{margin-bottom:8px}section{margin-top:28px}"
        "table{border-collapse:collapse;width:100%;background:white}"
        "th,td{border:1px solid #d1d5db;padding:7px 9px;text-align:left}"
        "pre{overflow:auto;background:#111827;color:#f9fafb;padding:16px;border-radius:6px}"
        ".image-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:14px}"
        ".image-card,.sample{margin:0;background:white;border:1px solid #d1d5db;border-radius:6px;padding:10px}"
        ".image-card{cursor:zoom-in}.image-card img{display:block;width:100%;height:auto}"
        ".image-card figcaption{margin-top:7px;color:#374151}.sample{margin-bottom:10px}"
        + report_lightbox_css()
    )


def _validate_report_bundle(bundle: LanguageReportBundle) -> None:
    if not isinstance(bundle.manifest, LanguageReportManifest):
        raise TypeError("manifest must be a LanguageReportManifest")
    expected_trace_routers = tuple(
        router for router, _, _ in _EBT_TRACE_PRESENTATION
    )
    if len(bundle.addressing_traces) != len(expected_trace_routers) or any(
        not isinstance(trace, AddressingCoefficientTrace)
        for trace in bundle.addressing_traces
    ):
        raise ValueError("addressing traces must contain both EBT routers in order")
    if (
        tuple(trace.router for trace in bundle.addressing_traces)
        != expected_trace_routers
    ):
        raise ValueError("addressing traces must contain both EBT routers in order")
    for family_name, records in _jsonl_families(bundle):
        if not records or any(not isinstance(record, ReportRecord) for record in records):
            raise ValueError(f"{family_name} requires nonempty ReportRecord values")
    for records, required_fields in (
        (bundle.stage_metrics, ("stage", "task_id", "parent")),
        (bundle.stored_competence, ("stage", "baseline", "task_id", "suffix_nll")),
        (
            bundle.routing_metrics,
            (
                "stage",
                "router",
                "task_id",
                "prefix_length",
                "routed_suffix_nll",
                "negative_control_chance_accuracy",
                "negative_control_ci95_lower",
                "negative_control_ci95_upper",
                "negative_control_chance_in_ci95",
                "leakage_audit_required",
            ),
        ),
        (bundle.transfer_metrics, ("stage", "task_id", "transfer")),
        (bundle.memory_metrics, ("stage", "persistent_bytes", "runtime_bytes")),
        (bundle.addressing_cost, ("stage", "router", "cold_seconds", "warm_seconds")),
        (
            _addressing_trace_records(bundle.addressing_traces),
            (
                "router",
                "task_id",
                "prefix_length",
                "example_index",
                "step",
                "objective",
                "coefficient_type",
                "coefficient_label",
                "coefficient",
            ),
        ),
    ):
        _require_record_fields(records, required_fields)
    for name, records in (
        ("competence_curve", bundle.competence_curve),
        ("routing_curve", bundle.routing_curve),
        ("memory_curve", bundle.memory_curve),
    ):
        _validate_curve_records(name, records)
    _require_method_coverage(
        bundle.stored_competence,
        "baseline",
        STORED_BASELINE_NAMES,
    )
    _require_method_coverage(
        bundle.routing_metrics,
        "router",
        ROUTER_BASELINE_NAMES,
    )
    _require_method_coverage(
        bundle.addressing_cost,
        "router",
        ROUTER_BASELINE_NAMES,
    )
    if not isinstance(bundle.address_confusion, AddressConfusion):
        raise TypeError("address_confusion must be an AddressConfusion")
    if not isinstance(bundle.graph, MemoryGraph):
        raise TypeError("graph must be a MemoryGraph")
    graph_node_ids = memory_node_ids(bundle.graph)
    representative_metadata = {
        (trace.task_id, trace.prefix_length, trace.example_index)
        for trace in bundle.addressing_traces
    }
    if len(representative_metadata) != 1:
        raise ValueError("EBT traces must describe the same representative example")
    if len({trace.objective_trace.size for trace in bundle.addressing_traces}) != 1:
        raise ValueError("EBT traces must contain the same refinement steps")
    expected_node_labels = tuple(str(node_id) for node_id in graph_node_ids)
    expected_edge_labels = tuple(
        f"{node.parent_id} → {node.node_id}"
        for node in bundle.graph.nodes
        if node.parent_id is not None
    )
    if any(
        trace.node_labels != expected_node_labels
        or trace.edge_labels != expected_edge_labels
        for trace in bundle.addressing_traces
    ):
        raise ValueError("addressing trace labels must follow graph insertion order")
    if tuple(stats.node_id for stats in bundle.node_stats) != graph_node_ids:
        raise ValueError("node visual stats must follow graph insertion order")
    expected_edges = tuple(
        (node.parent_id, node.node_id)
        for node in bundle.graph.nodes
        if node.parent_id is not None
    )
    if tuple((stats.parent_id, stats.child_id) for stats in bundle.edge_stats) != expected_edges:
        raise ValueError("edge visual stats must follow non-root graph insertion order")
    if not bundle.samples:
        raise ValueError("language reports require generated samples")
    represented_samples = {sample.baseline for sample in bundle.samples}
    missing_samples = set((*STORED_BASELINE_NAMES, *ROUTER_BASELINE_NAMES)).difference(
        represented_samples
    )
    if missing_samples:
        raise ValueError(f"generated samples omit methods: {sorted(missing_samples)}")


def _validate_curve_records(
    family_name: str,
    records: tuple[ReportRecord, ...],
) -> None:
    if not records:
        raise ValueError(f"{family_name} must not be empty")
    keys = tuple(key for key, _ in records[0].entries)
    if "stage" not in keys or len(keys) < 2:
        raise ValueError(f"{family_name} requires stage plus at least one series")
    if any(tuple(key for key, _ in record.entries) != keys for record in records):
        raise ValueError(f"{family_name} rows must share one schema")
    for record in records:
        _curve_row(record)


def _require_method_coverage(
    records: tuple[ReportRecord, ...],
    field_name: str,
    required_names: tuple[str, ...],
) -> None:
    represented = {
        value
        for record in records
        for value in (record.require(field_name),)
        if isinstance(value, str)
    }
    missing = set(required_names).difference(represented)
    if missing:
        raise ValueError(f"{field_name} coverage omits methods: {sorted(missing)}")


def _require_record_fields(
    records: tuple[ReportRecord, ...],
    required_fields: tuple[str, ...],
) -> None:
    for record in records:
        for field_name in required_fields:
            record.require(field_name)


def _canonical_config(config_json: str) -> str:
    if not isinstance(config_json, str) or not config_json:
        raise ValueError("config_json must be a nonempty JSON object")
    try:
        parsed = json.loads(config_json)
        canonical = json.dumps(
            parsed,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError("config_json must contain finite JSON values") from error
    if not isinstance(parsed, dict) or not parsed:
        raise ValueError("config_json must encode a nonempty JSON object")
    if config_json != canonical:
        raise ValueError("config_json must use canonical sorted compact JSON")
    return canonical


def _validate_path_component(value: str, field_name: str) -> None:
    if not isinstance(value, str) or _PATH_COMPONENT.fullmatch(value) is None:
        raise ValueError(
            f"{field_name} must contain lowercase letters, digits, underscores, or hyphens"
        )


def _valid_report_scalar(value: object) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    return isinstance(value, float) and math.isfinite(value)

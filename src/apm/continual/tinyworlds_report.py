"""Deterministic, evaluation-only report artifacts for TinyWorlds v1.

This module deliberately contains no training, model loading, or evaluation
entry points.  It projects an immutable completed pilot result into a parallel
TinyWorlds report bundle, validates the scientific bookkeeping, and writes a
byte-stable artifact directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import html
import json
import math
from pathlib import Path
import re
from typing import TypeAlias
from xml.sax.saxutils import escape as xml_escape

import numpy as np

from apm.continual.knowledge_evaluation import (
    KNOWLEDGE_AGGREGATION_AXES,
    KnowledgeEvaluationAggregate,
    KnowledgeMethodEvaluation,
    KnowledgeQueryEvaluation,
)
from apm.continual.language_benchmarks import (
    ROUTER_BASELINE_NAMES,
    STORED_BASELINE_NAMES,
)
from apm.data.text.tinyworlds.schema import QueryKind
from apm.data.text.tinyworlds.world_generation import PILOT_TASK_IDS


TinyWorldsScalar: TypeAlias = str | int | float | bool | None

TINYWORLDS_REPORT_SCHEMA_VERSION = 1
TINYWORLDS_REPORT_DATASET = "tinyworlds-v1"
TINYWORLDS_REPORT_CURRICULUM = "knowledge-graph"
TINYWORLDS_REPORT_STAGES = tuple(range(1, 9))
TINYWORLDS_REPORT_TASK_IDS = tuple(str(task_id) for task_id in PILOT_TASK_IDS)
TINYWORLDS_REPORT_PREFIX_LENGTHS = (64, 128, 192)
TINYWORLDS_REPORT_CUE_REGIMES = (
    "cue_sufficient",
    "cue_present",
    "cue_hidden_or_ambiguous",
    "cue_free_control",
)
TINYWORLDS_REPORT_QUERY_KINDS = tuple(kind.value for kind in QueryKind)
TINYWORLDS_REPORT_METHODS = (
    *STORED_BASELINE_NAMES,
    *ROUTER_BASELINE_NAMES,
    "vamp_ebt_uniform_soft",
    "vamp_ebt_hopfield_soft",
)
TINYWORLDS_ADDRESSING_METHODS = (
    *ROUTER_BASELINE_NAMES,
    "vamp_ebt_uniform_soft",
    "vamp_ebt_hopfield_soft",
)
TINYWORLDS_NATURAL_CONTINUATION_METHODS = (
    *STORED_BASELINE_NAMES,
    *ROUTER_BASELINE_NAMES,
)
TINYWORLDS_PARENT_COUNTERFACTUALS = (
    "root",
    "true_parent",
    "selected_parent",
    "strongest_other_family",
)
TINYWORLDS_IMPLEMENTATION_GATES = (
    "exact_kg_integrity",
    "committed_node_drift",
    "no_test_selection",
    "allocator_peak_12gib",
)
TINYWORLDS_ALLOCATOR_PEAK_LIMIT_BYTES = 12 * 1024**3
TINYWORLDS_REQUIRED_CONFIG_FIELDS = (
    "generator_version",
    "renderer_version",
    "seeds",
    "checkpoint_identity",
    "tokenizer_identity",
    "corpus_identity",
    "topology",
    "ontology",
    "calibration_profile",
    "story_policy",
    "query_policy",
    "candidate_policy",
    "adapter_settings",
    "optimizer_settings",
    "routers",
    "microbatching",
    "timing_targets",
    "memory_targets",
)
TINYWORLDS_REPORT_FILENAMES = (
    "manifest.json",
    "candidate_scores.jsonl",
    "knowledge_aggregates.jsonl",
    "routing_records.jsonl",
    "natural_continuation_metrics.jsonl",
    "parent_search.jsonl",
    "checkpointed_transfer.jsonl",
    "graph_recovery.jsonl",
    "revision_retention.jsonl",
    "committed_node_drift.jsonl",
    "memory_metrics.jsonl",
    "addressing_cost.jsonl",
    "gate_results.jsonl",
    "representative_queries.jsonl",
    "selection_audit.jsonl",
    "sequential_results.jsonl",
    "candidate_reasoning.svg",
    "cue_routing.svg",
    "parent_rank_transfer.svg",
    "revision_retention.svg",
    "expected_vs_learned_graph.svg",
    "report.html",
)

_PATH_COMPONENT = re.compile(r"[a-z0-9][a-z0-9_-]*\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CONFIG_HASH_LENGTH = 12
_REVISION_CHART_METRICS = (
    ("old context", "old_context_accuracy"),
    ("revision context", "revision_context_accuracy"),
    ("paired consistency", "paired_revision_consistency"),
)


@dataclass(frozen=True, slots=True)
class TinyWorldsRecord:
    """One immutable scalar-only row in a TinyWorlds report family."""

    entries: tuple[tuple[str, TinyWorldsScalar], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.entries, tuple) or not self.entries:
            raise ValueError("TinyWorlds records require immutable entries")
        if any(
            not isinstance(entry, tuple) or len(entry) != 2
            for entry in self.entries
        ):
            raise TypeError("TinyWorlds record entries must be key/value tuples")
        keys = tuple(key for key, _ in self.entries)
        if any(not isinstance(key, str) or not key for key in keys):
            raise ValueError("TinyWorlds record keys must be nonempty strings")
        if len(set(keys)) != len(keys):
            raise ValueError("TinyWorlds record keys must be unique")
        if any(not _valid_scalar(value) for _, value in self.entries):
            raise TypeError("TinyWorlds record values must be finite JSON scalars")

    def as_dict(self) -> dict[str, TinyWorldsScalar]:
        """Return a fresh JSON-compatible mapping in record order."""
        return dict(self.entries)

    def require(self, field_name: str) -> TinyWorldsScalar:
        """Return a required field or raise a descriptive validation error."""
        for key, value in self.entries:
            if key == field_name:
                return value
        raise ValueError(f"TinyWorlds record is missing field: {field_name}")

    @property
    def canonical_json(self) -> str:
        """Return the row's stable compact JSON representation."""
        return _canonical_json_value(self.as_dict())


@dataclass(frozen=True, slots=True)
class TinyWorldsReportManifest:
    """Content-addressed identity for the fixed TinyWorlds v1 pilot."""

    preset: str
    seed: int
    config_json: str

    def __post_init__(self) -> None:
        if not isinstance(self.preset, str) or _PATH_COMPONENT.fullmatch(
            self.preset
        ) is None:
            raise ValueError(
                "preset must contain lowercase letters, digits, underscores, or hyphens"
            )
        if type(self.seed) is not int or self.seed != 0:
            raise ValueError("the canonical TinyWorlds pilot uses seed zero")
        config = _parse_canonical_config(self.config_json)
        missing = set(TINYWORLDS_REQUIRED_CONFIG_FIELDS).difference(config)
        if missing:
            raise ValueError(
                f"TinyWorlds report config omits required fields: {sorted(missing)}"
            )
        for field_name in TINYWORLDS_REQUIRED_CONFIG_FIELDS:
            if _empty_identity_value(config[field_name]):
                raise ValueError(
                    f"TinyWorlds report config field must be nonempty: {field_name}"
                )
        routers = config["routers"]
        if not isinstance(routers, list) or tuple(routers) != TINYWORLDS_REPORT_METHODS:
            raise ValueError(
                "TinyWorlds report routers must list the canonical pilot methods"
            )

    @property
    def config_hash(self) -> str:
        """Return the full SHA-256 over all canonical policy/config fields."""
        return sha256(self.config_json.encode("utf-8")).hexdigest()

    @property
    def run_id(self) -> str:
        """Return the prescribed preset/seed/content-hash run identity."""
        return f"{self.preset}-seed{self.seed}-{self.config_hash[:_CONFIG_HASH_LENGTH]}"


@dataclass(frozen=True, slots=True)
class TinyWorldsCompletedResult:
    """Immutable result of a finished pilot, before report projection.

    Constructing this value performs bookkeeping validation only.  In
    particular, it never invokes model scoring, routing, training, or a KG
    executor.
    """

    manifest: TinyWorldsReportManifest
    method_evaluations: tuple[KnowledgeMethodEvaluation, ...]
    natural_continuation_metrics: tuple[TinyWorldsRecord, ...]
    parent_search: tuple[TinyWorldsRecord, ...]
    checkpointed_transfer: tuple[TinyWorldsRecord, ...]
    graph_recovery: tuple[TinyWorldsRecord, ...]
    revision_retention: tuple[TinyWorldsRecord, ...]
    committed_node_drift: tuple[TinyWorldsRecord, ...]
    memory_metrics: tuple[TinyWorldsRecord, ...]
    addressing_cost: tuple[TinyWorldsRecord, ...]
    gate_results: tuple[TinyWorldsRecord, ...]
    representative_queries: tuple[TinyWorldsRecord, ...]
    selection_audit: tuple[TinyWorldsRecord, ...]
    sequential_results: tuple[TinyWorldsRecord, ...]

    def __post_init__(self) -> None:
        _validate_completed_result(self)


@dataclass(frozen=True, slots=True)
class TinyWorldsReportBundle:
    """Complete scalar-only input to deterministic TinyWorlds report writing."""

    manifest: TinyWorldsReportManifest
    completed_result_sha256: str
    candidate_scores: tuple[TinyWorldsRecord, ...]
    knowledge_aggregates: tuple[TinyWorldsRecord, ...]
    routing_records: tuple[TinyWorldsRecord, ...]
    natural_continuation_metrics: tuple[TinyWorldsRecord, ...]
    parent_search: tuple[TinyWorldsRecord, ...]
    checkpointed_transfer: tuple[TinyWorldsRecord, ...]
    graph_recovery: tuple[TinyWorldsRecord, ...]
    revision_retention: tuple[TinyWorldsRecord, ...]
    committed_node_drift: tuple[TinyWorldsRecord, ...]
    memory_metrics: tuple[TinyWorldsRecord, ...]
    addressing_cost: tuple[TinyWorldsRecord, ...]
    gate_results: tuple[TinyWorldsRecord, ...]
    representative_queries: tuple[TinyWorldsRecord, ...]
    selection_audit: tuple[TinyWorldsRecord, ...]
    sequential_results: tuple[TinyWorldsRecord, ...]

    def __post_init__(self) -> None:
        _validate_report_bundle(self)


def canonical_tinyworlds_config_json(config: dict[str, object]) -> str:
    """Return canonical JSON used by the TinyWorlds run identity."""
    if not isinstance(config, dict) or not config:
        raise ValueError("TinyWorlds report config must be a nonempty object")
    return _canonical_json_value(config)


def tinyworlds_report_directory(
    results_root: str | Path,
    manifest: TinyWorldsReportManifest,
) -> Path:
    """Return the fixed TinyWorlds v1 knowledge-graph report directory."""
    if not isinstance(manifest, TinyWorldsReportManifest):
        raise TypeError("manifest must be a TinyWorldsReportManifest")
    return (
        Path(results_root)
        / "language_cl"
        / TINYWORLDS_REPORT_DATASET
        / TINYWORLDS_REPORT_CURRICULUM
        / manifest.run_id
    )


def build_tinyworlds_report_bundle(
    completed: TinyWorldsCompletedResult,
) -> TinyWorldsReportBundle:
    """Project a completed result without executing evaluation or training."""
    if not isinstance(completed, TinyWorldsCompletedResult):
        raise TypeError("completed must be a TinyWorldsCompletedResult")
    evaluations = tuple(
        sorted(
            completed.method_evaluations,
            key=lambda item: (
                item.stage,
                TINYWORLDS_REPORT_METHODS.index(item.method),
            ),
        )
    )
    candidate_scores = tuple(
        _candidate_score_record(row)
        for evaluation in evaluations
        for row in sorted(evaluation.queries, key=lambda item: item.query_id)
    )
    knowledge_aggregates = tuple(
        _aggregate_record(row)
        for evaluation in evaluations
        for row in sorted(
            evaluation.aggregates,
            key=lambda item: (
                KNOWLEDGE_AGGREGATION_AXES.index(item.grouping_axis),
                str(item.grouping_value),
            ),
        )
    )
    routing_records = tuple(
        _routing_record(row)
        for evaluation in evaluations
        for row in sorted(evaluation.queries, key=lambda item: item.query_id)
    )
    record_families = {
        name: _sorted_records(getattr(completed, name))
        for name in _COMPLETED_RECORD_FAMILIES
    }
    digest_payload = {
        "dataset": TINYWORLDS_REPORT_DATASET,
        "curriculum": TINYWORLDS_REPORT_CURRICULUM,
        "preset": completed.manifest.preset,
        "seed": completed.manifest.seed,
        "config_hash": completed.manifest.config_hash,
        "candidate_scores": [row.as_dict() for row in candidate_scores],
        "knowledge_aggregates": [
            row.as_dict() for row in knowledge_aggregates
        ],
        "routing_records": [row.as_dict() for row in routing_records],
        **{
            name: [row.as_dict() for row in rows]
            for name, rows in record_families.items()
        },
    }
    return TinyWorldsReportBundle(
        manifest=completed.manifest,
        completed_result_sha256=sha256(
            _canonical_json_value(digest_payload).encode("utf-8")
        ).hexdigest(),
        candidate_scores=candidate_scores,
        knowledge_aggregates=knowledge_aggregates,
        routing_records=routing_records,
        **record_families,
    )


def write_tinyworlds_report(
    output_directory: str | Path,
    bundle: TinyWorldsReportBundle,
) -> Path:
    """Write a deterministic completed report into a staging directory."""
    _validate_report_bundle(bundle)
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    unexpected = {
        path.name for path in directory.iterdir()
    }.difference(TINYWORLDS_REPORT_FILENAMES)
    if unexpected:
        raise ValueError(
            f"TinyWorlds report directory contains unexpected files: {sorted(unexpected)}"
        )
    manifest_payload = {
        "schema_version": TINYWORLDS_REPORT_SCHEMA_VERSION,
        "dataset": TINYWORLDS_REPORT_DATASET,
        "curriculum": TINYWORLDS_REPORT_CURRICULUM,
        "preset": bundle.manifest.preset,
        "seed": bundle.manifest.seed,
        "run_id": bundle.manifest.run_id,
        "config_hash": bundle.manifest.config_hash,
        "completed_result_sha256": bundle.completed_result_sha256,
        "config": json.loads(bundle.manifest.config_json),
        "task_order": list(TINYWORLDS_REPORT_TASK_IDS),
        "stages": list(TINYWORLDS_REPORT_STAGES),
        "methods": list(TINYWORLDS_REPORT_METHODS),
        "prefix_lengths": list(TINYWORLDS_REPORT_PREFIX_LENGTHS),
        "cue_regimes": list(TINYWORLDS_REPORT_CUE_REGIMES),
        "query_kinds": list(TINYWORLDS_REPORT_QUERY_KINDS),
        "interpretation": "knowledge-graph continual learning",
    }
    for filename, records in _jsonl_families(bundle):
        _write_jsonl(directory / filename, records)
    _write_bar_chart(
        directory / "candidate_reasoning.svg",
        "Candidate accuracy by reasoning type",
        *_final_aggregate_chart(
            bundle,
            grouping_axis="reasoning_type",
            metric="candidate_accuracy",
        ),
    )
    _write_bar_chart(
        directory / "cue_routing.svg",
        "Task-free routing by cue regime",
        *_final_aggregate_chart(
            bundle,
            grouping_axis="cue_regime",
            metric="node_accuracy",
            fallback_metric="candidate_accuracy",
        ),
    )
    _write_bar_chart(
        directory / "parent_rank_transfer.svg",
        "Selected-parent rank and transfer",
        (
            *tuple(
                f"stage {record.require('stage')} selected rank"
                for record in bundle.parent_search
                if record.require("selected") is True
            ),
            *tuple(
                f"stage {record.require('stage')} {record.require('parent_kind')} accuracy"
                for record in bundle.checkpointed_transfer
                if record.require("available") is True
                and record.require("update") == record.require("final_update")
            ),
        ),
        (
            *tuple(
                float(record.require("rank"))
                for record in bundle.parent_search
                if record.require("selected") is True
            ),
            *tuple(
                float(record.require("candidate_accuracy"))
                for record in bundle.checkpointed_transfer
                if record.require("available") is True
                and record.require("update") == record.require("final_update")
            ),
        ),
    )
    _write_bar_chart(
        directory / "revision_retention.svg",
        "Revision consistency",
        tuple(
            f"{record.require('family_id')} {metric_label}"
            for record in bundle.revision_retention
            for metric_label, _ in _REVISION_CHART_METRICS
        ),
        tuple(
            float(record.require(metric_name))
            for record in bundle.revision_retention
            for _, metric_name in _REVISION_CHART_METRICS
        ),
    )
    _write_bar_chart(
        directory / "expected_vs_learned_graph.svg",
        "Expected versus learned graph",
        tuple(
            f"{record.require('task_id')}: {record.require('expected_parent_id')} → {record.require('learned_parent_id')}"
            for record in bundle.graph_recovery
        ),
        tuple(
            1.0 if record.require("recovered") is True else 0.0
            for record in bundle.graph_recovery
        ),
    )
    _write_report_html(directory / "report.html", bundle)
    manifest_payload["artifact_sha256"] = {
        filename: sha256((directory / filename).read_bytes()).hexdigest()
        for filename in TINYWORLDS_REPORT_FILENAMES
        if filename != "manifest.json"
    }
    _write_json(directory / "manifest.json", manifest_payload)
    validate_tinyworlds_report_artifact(directory, bundle)
    return directory


def validate_tinyworlds_report_artifact(
    directory: str | Path,
    bundle: TinyWorldsReportBundle,
) -> None:
    """Validate a fully written staging tree before atomic promotion."""
    _validate_report_bundle(bundle)
    path = Path(directory)
    if not path.is_dir():
        raise FileNotFoundError(f"TinyWorlds report directory is absent: {path}")
    represented = tuple(sorted(item.name for item in path.iterdir() if item.is_file()))
    expected = tuple(sorted(TINYWORLDS_REPORT_FILENAMES))
    if represented != expected:
        raise ValueError("TinyWorlds report artifact set is incomplete or unexpected")
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("config_hash") != bundle.manifest.config_hash
        or manifest.get("completed_result_sha256")
        != bundle.completed_result_sha256
        or manifest.get("run_id") != bundle.manifest.run_id
    ):
        raise ValueError("TinyWorlds report manifest identity does not match bundle")
    expected_checksums = {
        filename: sha256((path / filename).read_bytes()).hexdigest()
        for filename in TINYWORLDS_REPORT_FILENAMES
        if filename != "manifest.json"
    }
    if manifest.get("artifact_sha256") != expected_checksums:
        raise ValueError("TinyWorlds report artifact checksum mismatch")
    for filename, records in _jsonl_families(bundle):
        rows = tuple(
            json.loads(line)
            for line in (path / filename).read_text(encoding="utf-8").splitlines()
        )
        if rows != tuple(record.as_dict() for record in records):
            raise ValueError(f"TinyWorlds JSONL projection mismatch: {filename}")
    html_text = (path / "report.html").read_text(encoding="utf-8")
    if bundle.manifest.run_id not in html_text:
        raise ValueError("TinyWorlds HTML report omits its run identity")


def atomically_promote_tinyworlds_report(
    staging_directory: str | Path,
    results_root: str | Path,
    bundle: TinyWorldsReportBundle,
) -> Path:
    """Validate and atomically rename a completed staging tree into results."""
    staging = Path(staging_directory)
    validate_tinyworlds_report_artifact(staging, bundle)
    destination = tinyworlds_report_directory(results_root, bundle.manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        validate_tinyworlds_report_artifact(destination, bundle)
        if _directory_digest(destination) != _directory_digest(staging):
            raise FileExistsError(
                "content-addressed TinyWorlds destination already differs"
            )
        return destination
    staging.replace(destination)
    validate_tinyworlds_report_artifact(destination, bundle)
    return destination


_COMPLETED_RECORD_FAMILIES = (
    "natural_continuation_metrics",
    "parent_search",
    "checkpointed_transfer",
    "graph_recovery",
    "revision_retention",
    "committed_node_drift",
    "memory_metrics",
    "addressing_cost",
    "gate_results",
    "representative_queries",
    "selection_audit",
    "sequential_results",
)


def _validate_completed_result(result: TinyWorldsCompletedResult) -> None:
    if not isinstance(result.manifest, TinyWorldsReportManifest):
        raise TypeError("manifest must be a TinyWorldsReportManifest")
    evaluations = result.method_evaluations
    if not isinstance(evaluations, tuple) or any(
        not isinstance(item, KnowledgeMethodEvaluation) for item in evaluations
    ):
        raise TypeError(
            "method_evaluations must contain KnowledgeMethodEvaluation values"
        )
    _validate_evaluation_coverage(evaluations)
    for name in _COMPLETED_RECORD_FAMILIES:
        _validate_record_family(name, getattr(result, name))
    _validate_completed_record_semantics(result)


def _validate_evaluation_coverage(
    evaluations: tuple[KnowledgeMethodEvaluation, ...],
) -> None:
    expected_pairs = {
        (stage, method)
        for stage in TINYWORLDS_REPORT_STAGES
        for method in TINYWORLDS_REPORT_METHODS
    }
    represented_pairs = {(item.stage, item.method) for item in evaluations}
    if represented_pairs != expected_pairs or len(evaluations) != len(expected_pairs):
        missing = sorted(expected_pairs.difference(represented_pairs))
        extra = sorted(represented_pairs.difference(expected_pairs))
        raise ValueError(
            f"knowledge evaluation coverage mismatch; missing={missing}, extra={extra}"
        )
    reference_query_ids: set[str] | None = None
    for stage in TINYWORLDS_REPORT_STAGES:
        stage_evaluations = tuple(
            item for item in evaluations if item.stage == stage
        )
        metadata_by_method: dict[str, dict[str, tuple[object, ...]]] = {}
        for evaluation in stage_evaluations:
            aggregate_axes = {row.grouping_axis for row in evaluation.aggregates}
            if aggregate_axes != set(KNOWLEDGE_AGGREGATION_AXES):
                raise ValueError(
                    "knowledge aggregates must cover every required grouping axis"
                )
            metadata_by_method[evaluation.method] = {
                row.query_id: _query_metadata_key(row)
                for row in evaluation.queries
            }
        first_metadata = metadata_by_method[TINYWORLDS_REPORT_METHODS[0]]
        if any(metadata != first_metadata for metadata in metadata_by_method.values()):
            raise ValueError(
                "all methods at a stage must evaluate identical paired queries"
            )
        query_ids = set(first_metadata)
        if reference_query_ids is None:
            reference_query_ids = query_ids
        elif query_ids != reference_query_ids:
            raise ValueError("every stage must evaluate the same held-out query IDs")
        rows = next(
            item.queries
            for item in stage_evaluations
            if item.method == TINYWORLDS_REPORT_METHODS[0]
        )
        _require_exact_values(
            "task IDs",
            {row.task_id for row in rows},
            set(TINYWORLDS_REPORT_TASK_IDS),
        )
        _require_exact_values(
            "prefix lengths",
            {row.prefix_length for row in rows},
            set(TINYWORLDS_REPORT_PREFIX_LENGTHS),
        )
        _require_exact_values(
            "cue regimes",
            {row.cue_regime for row in rows},
            set(TINYWORLDS_REPORT_CUE_REGIMES),
        )
        _require_exact_values(
            "query kinds",
            {row.query_kind for row in rows},
            set(TINYWORLDS_REPORT_QUERY_KINDS),
        )
        _require_exact_values(
            "open/closed-book modes",
            {row.mode for row in rows},
            {"open_book", "closed_book"},
        )


def _validate_completed_record_semantics(
    result: object,
    *,
    final_query_ids: set[str] | None = None,
) -> None:
    _require_fields(
        result.natural_continuation_metrics,
        ("stage", "method", "task_id", "prefix_length", "suffix_nll"),
    )
    represented_natural = tuple(
        (
            record.require("stage"),
            record.require("method"),
            record.require("task_id"),
            record.require("prefix_length"),
        )
        for record in result.natural_continuation_metrics
    )
    expected_natural = tuple(
        (stage, method, task_id, prefix_length)
        for stage in TINYWORLDS_REPORT_STAGES
        for method in TINYWORLDS_NATURAL_CONTINUATION_METHODS
        for task_id in TINYWORLDS_REPORT_TASK_IDS
        for prefix_length in TINYWORLDS_REPORT_PREFIX_LENGTHS
    )
    if len(set(represented_natural)) != len(represented_natural) or set(
        represented_natural
    ) != set(expected_natural):
        raise ValueError(
            "natural continuation must cover each stage/method/task/prefix exactly once"
        )
    for record in result.natural_continuation_metrics:
        _require_nonnegative_number(record.require("suffix_nll"), "suffix NLL")
    _require_fields(
        result.parent_search,
        (
            "stage",
            "task_id",
            "candidate_parent_id",
            "rank",
            "mean_candidate_nll",
            "selected",
        ),
    )
    _require_stage_coverage("parent search", result.parent_search)
    for record in result.parent_search:
        _require_nonnegative_integer(record.require("rank"), "parent rank")
        _require_nonnegative_number(
            record.require("mean_candidate_nll"), "parent candidate NLL"
        )
        _require_bool(record.require("selected"), "selected")
    if any(
        sum(
            row.require("selected") is True
            for row in result.parent_search
            if row.require("stage") == stage
        )
        != 1
        for stage in TINYWORLDS_REPORT_STAGES
    ):
        raise ValueError("parent search requires one selected row per stage")
    _require_fields(
        result.checkpointed_transfer,
        (
            "stage",
            "task_id",
            "parent_kind",
            "available",
            "parent_node_id",
            "update",
            "training_loss",
            "candidate_accuracy",
            "correct_answer_nll",
            "adapter_sha256",
            "final_update",
        ),
    )
    report_config = _parse_canonical_config(result.manifest.config_json)
    optimizer_settings = report_config.get("optimizer_settings")
    if not isinstance(optimizer_settings, dict):
        raise ValueError("optimizer settings must record the update budget")
    update_budget = optimizer_settings.get("steps")
    if type(update_budget) is not int or update_budget <= 0:
        raise ValueError("optimizer settings steps must be a positive integer")
    expected_checkpoint_updates = _checkpoint_updates(update_budget)
    for stage in TINYWORLDS_REPORT_STAGES:
        kinds = {
            row.require("parent_kind")
            for row in result.checkpointed_transfer
            if row.require("stage") == stage
        }
        _require_exact_values(
            f"stage {stage} parent counterfactuals",
            kinds,
            set(TINYWORLDS_PARENT_COUNTERFACTUALS),
        )
        for parent_kind in TINYWORLDS_PARENT_COUNTERFACTUALS:
            role_rows = tuple(
                row
                for row in result.checkpointed_transfer
                if row.require("stage") == stage
                and row.require("parent_kind") == parent_kind
            )
            available_values = {
                row.require("available") for row in role_rows
            }
            if len(available_values) != 1:
                raise ValueError(
                    "one parent role cannot mix available and unavailable rows"
                )
            if available_values == {True}:
                updates = tuple(row.require("update") for row in role_rows)
                if len(updates) != len(expected_checkpoint_updates) or set(
                    updates
                ) != set(expected_checkpoint_updates):
                    raise ValueError(
                        "available parent transfer rows must cover the exact "
                        "checkpoint schedule"
                    )
            elif len(role_rows) != 1:
                raise ValueError(
                    "an unavailable parent role requires exactly one null row"
                )
    for record in result.checkpointed_transfer:
        available = record.require("available")
        _require_bool(available, "counterfactual availability")
        if available is True:
            parent_node_id = record.require("parent_node_id")
            if not isinstance(parent_node_id, str) or not parent_node_id:
                raise ValueError("available counterfactuals require a parent node ID")
            _require_rate(record.require("candidate_accuracy"), "transfer accuracy")
            _require_nonnegative_number(
                record.require("correct_answer_nll"), "transfer answer NLL"
            )
            update = _require_nonnegative_integer(
                record.require("update"), "transfer checkpoint update"
            )
            training_loss = record.require("training_loss")
            if update == 0:
                if training_loss is not None:
                    raise ValueError(
                        "update-zero transfer checkpoints have no training loss"
                    )
            else:
                _require_nonnegative_number(
                    training_loss,
                    "transfer checkpoint training loss",
                )
            adapter_sha256 = record.require("adapter_sha256")
            if not isinstance(adapter_sha256, str) or _SHA256.fullmatch(
                adapter_sha256
            ) is None:
                raise ValueError("transfer adapter checksum must be SHA-256")
            if record.require("final_update") != update_budget:
                raise ValueError(
                    "transfer final_update must equal the configured budget"
                )
        elif (
            record.require("parent_kind") != "strongest_other_family"
            or record.require("parent_node_id") is not None
            or record.require("update") is not None
            or record.require("training_loss") is not None
            or record.require("candidate_accuracy") is not None
            or record.require("correct_answer_nll") is not None
            or record.require("adapter_sha256") is not None
            or record.require("final_update") is not None
        ):
            raise ValueError(
                "only an unavailable strongest-other-family counterfactual may use null metrics"
            )
    _require_fields(
        result.graph_recovery,
        ("task_id", "expected_parent_id", "learned_parent_id", "recovered"),
    )
    _require_exact_values(
        "graph recovery tasks",
        {row.require("task_id") for row in result.graph_recovery},
        set(TINYWORLDS_REPORT_TASK_IDS),
    )
    for record in result.graph_recovery:
        _require_bool(record.require("recovered"), "graph recovered")
    _require_fields(
        result.revision_retention,
        (
            "stage",
            "family_id",
            "old_context_accuracy",
            "revision_context_accuracy",
            "paired_revision_consistency",
        ),
    )
    _require_exact_values(
        "revision families",
        {row.require("family_id") for row in result.revision_retention},
        {"willow", "sunny"},
    )
    for record in result.revision_retention:
        for _, metric_name in _REVISION_CHART_METRICS:
            _require_rate(record.require(metric_name), metric_name)
    _require_fields(
        result.committed_node_drift,
        (
            "stage",
            "node_id",
            "logit_max_abs_drift",
            "answer_change_count",
            "checksum_match",
        ),
    )
    _require_stage_coverage("committed-node drift", result.committed_node_drift)
    for record in result.committed_node_drift:
        if (
            float(record.require("logit_max_abs_drift")) != 0.0
            or record.require("answer_change_count") != 0
            or record.require("checksum_match") is not True
        ):
            raise ValueError("committed-node drift must be exactly zero")
    _require_fields(
        result.memory_metrics,
        (
            "stage",
            "persistent_bytes",
            "runtime_bytes",
            "allocator_peak_bytes",
        ),
    )
    _require_stage_coverage("memory metrics", result.memory_metrics)
    for record in result.memory_metrics:
        for field_name in (
            "persistent_bytes",
            "runtime_bytes",
            "allocator_peak_bytes",
        ):
            _require_nonnegative_integer(record.require(field_name), field_name)
        peak = record.require("allocator_peak_bytes")
        if int(peak) > TINYWORLDS_ALLOCATOR_PEAK_LIMIT_BYTES:
            raise ValueError("allocator peak exceeds the 12 GiB pilot gate")
    _require_fields(
        result.addressing_cost,
        ("stage", "method", "cold_seconds", "warm_seconds"),
    )
    _require_stage_method_matrix(
        "addressing cost",
        result.addressing_cost,
        TINYWORLDS_ADDRESSING_METHODS,
    )
    for record in result.addressing_cost:
        _require_nonnegative_number(
            record.require("cold_seconds"), "cold addressing seconds"
        )
        _require_nonnegative_number(
            record.require("warm_seconds"), "warm addressing seconds"
        )
    _require_fields(result.gate_results, ("gate", "category", "passed"))
    gate_by_name = {record.require("gate"): record for record in result.gate_results}
    if len(gate_by_name) != len(result.gate_results):
        raise ValueError("gate result names must be unique")
    missing_gates = set(TINYWORLDS_IMPLEMENTATION_GATES).difference(gate_by_name)
    if missing_gates:
        raise ValueError(f"implementation gate coverage omits: {sorted(missing_gates)}")
    for record in result.gate_results:
        category = record.require("category")
        passed = record.require("passed")
        if category not in ("implementation", "scientific"):
            raise ValueError("gate category must be implementation or scientific")
        _require_bool(passed, "gate passed")
        if category == "implementation" and passed is not True:
            raise ValueError("implementation gates must pass before report promotion")
    _require_fields(
        result.representative_queries,
        ("query_id", "proof_id", "query_text", "answer_text", "support_ids"),
    )
    _require_fields(
        result.selection_audit,
        ("record_id", "split", "used_for_tuning", "used_for_parent_selection"),
    )
    audit_by_id = {
        record.require("record_id"): record for record in result.selection_audit
    }
    if len(audit_by_id) != len(result.selection_audit):
        raise ValueError("selection audit record IDs must be unique")
    for record in result.selection_audit:
        for field_name in ("used_for_tuning", "used_for_parent_selection"):
            _require_bool(record.require(field_name), field_name)
        if record.require("split") == "test" and (
            record.require("used_for_tuning") is True
            or record.require("used_for_parent_selection") is True
        ):
            raise ValueError("test records cannot participate in tuning or selection")
    if final_query_ids is None:
        final_query_ids = {
            row.query_id
            for evaluation in result.method_evaluations
            if evaluation.stage == TINYWORLDS_REPORT_STAGES[-1]
            and evaluation.method == TINYWORLDS_REPORT_METHODS[0]
            for row in evaluation.queries
        }
    if not final_query_ids.issubset(audit_by_id):
        raise ValueError("selection audit must cover every reported held-out query")
    if any(
        audit_by_id[query_id].require("split") != "test"
        for query_id in final_query_ids
    ):
        raise ValueError("reported held-out query scores must come from the test split")
    _require_fields(
        result.sequential_results,
        ("sequence_index", "stage", "event"),
    )
    _require_stage_coverage("sequential results", result.sequential_results)
    sequence_indices = tuple(
        sorted(
            _require_nonnegative_integer(
                row.require("sequence_index"), "sequence index"
            )
            for row in result.sequential_results
        )
    )
    if sequence_indices != tuple(range(len(sequence_indices))):
        raise ValueError("sequential result indices must be contiguous from zero")


def _validate_report_bundle(bundle: TinyWorldsReportBundle) -> None:
    if not isinstance(bundle.manifest, TinyWorldsReportManifest):
        raise TypeError("manifest must be a TinyWorldsReportManifest")
    if not isinstance(bundle.completed_result_sha256, str) or _SHA256.fullmatch(
        bundle.completed_result_sha256
    ) is None:
        raise ValueError("completed_result_sha256 must be a lowercase SHA-256")
    for name, records in _jsonl_families(bundle):
        _validate_record_family(name, records)
    _require_fields(
        bundle.candidate_scores,
        (
            "stage",
            "method",
            "query_id",
            "task_id",
            "query_kind",
            "prefix_length",
            "cue_regime",
            "candidate_0_nll",
            "candidate_1_nll",
            "candidate_2_nll",
            "candidate_3_nll",
            "correct_candidate_index",
            "predicted_candidate_index",
            "candidate_correct",
            "candidate_margin",
            "correct_answer_nll",
        ),
    )
    _require_stage_method_matrix(
        "candidate scores", bundle.candidate_scores, TINYWORLDS_REPORT_METHODS
    )
    _require_fields(
        bundle.knowledge_aggregates,
        (
            "stage",
            "method",
            "grouping_axis",
            "grouping_value",
            "query_count",
            "candidate_accuracy",
        ),
    )
    _require_stage_method_matrix(
        "knowledge aggregates",
        bundle.knowledge_aggregates,
        TINYWORLDS_REPORT_METHODS,
    )
    _require_fields(
        bundle.routing_records,
        (
            "stage",
            "method",
            "query_id",
            "selected_node_index",
            "required_edge_count",
        ),
    )
    _require_stage_method_matrix(
        "routing records", bundle.routing_records, TINYWORLDS_REPORT_METHODS
    )
    _validate_projected_knowledge_coverage(bundle)
    final_query_ids = {
        str(record.require("query_id"))
        for record in bundle.candidate_scores
        if record.require("stage") == TINYWORLDS_REPORT_STAGES[-1]
        and record.require("method") == TINYWORLDS_REPORT_METHODS[0]
    }
    _validate_completed_record_semantics(
        bundle,
        final_query_ids=final_query_ids,
    )


def _checkpoint_updates(update_budget: int) -> tuple[int, ...]:
    updates = [0]
    value = 1
    while value < update_budget:
        updates.append(value)
        value *= 2
    updates.append(update_budget)
    return tuple(dict.fromkeys(updates))


def _validate_projected_knowledge_coverage(
    bundle: TinyWorldsReportBundle,
) -> None:
    reference_query_ids: set[object] | None = None
    routing_by_pair = {
        (stage, method): tuple(
            row
            for row in bundle.routing_records
            if row.require("stage") == stage
            and row.require("method") == method
        )
        for stage in TINYWORLDS_REPORT_STAGES
        for method in TINYWORLDS_REPORT_METHODS
    }
    for stage in TINYWORLDS_REPORT_STAGES:
        metadata_by_method: dict[
            str,
            dict[object, tuple[object, ...]],
        ] = {}
        for method in TINYWORLDS_REPORT_METHODS:
            rows = tuple(
                row
                for row in bundle.candidate_scores
                if row.require("stage") == stage
                and row.require("method") == method
            )
            metadata = {
                row.require("query_id"): (
                    row.require("task_id"),
                    row.require("family_id"),
                    row.require("query_kind"),
                    row.require("proof_id"),
                    row.require("prefix_length"),
                    row.require("cue_regime"),
                    row.require("reasoning_type"),
                    row.require("reasoning_depth"),
                    row.require("novelty_regime"),
                    row.require("mode"),
                    row.require("correct_candidate_index"),
                    *(row.require(f"candidate_{index}_text") for index in range(4)),
                )
                for row in rows
            }
            if len(metadata) != len(rows):
                raise ValueError("candidate query IDs must be unique per method/stage")
            metadata_by_method[method] = metadata
            routing_ids = {
                row.require("query_id") for row in routing_by_pair[(stage, method)]
            }
            if routing_ids != set(metadata) or len(routing_ids) != len(
                routing_by_pair[(stage, method)]
            ):
                raise ValueError(
                    "routing records must align exactly with candidate query IDs"
                )
            aggregate_axes = {
                row.require("grouping_axis")
                for row in bundle.knowledge_aggregates
                if row.require("stage") == stage
                and row.require("method") == method
            }
            if aggregate_axes != set(KNOWLEDGE_AGGREGATION_AXES):
                raise ValueError(
                    "projected aggregates must cover every required grouping axis"
                )
            for row in rows:
                scores = tuple(
                    float(row.require(f"candidate_{index}_nll"))
                    for index in range(4)
                )
                if any(score < 0.0 for score in scores):
                    raise ValueError("candidate NLL values must be nonnegative")
                correct_index = row.require("correct_candidate_index")
                predicted_index = row.require("predicted_candidate_index")
                if (
                    type(correct_index) is not int
                    or type(predicted_index) is not int
                    or not 0 <= correct_index < 4
                    or predicted_index != int(np.argmin(np.asarray(scores)))
                ):
                    raise ValueError("candidate indices must match the four NLL values")
                if row.require("candidate_correct") is not (
                    predicted_index == correct_index
                ):
                    raise ValueError("candidate correctness must match candidate indices")
                correct_nll = scores[correct_index]
                wrong_nll = min(
                    score for index, score in enumerate(scores) if index != correct_index
                )
                if not math.isclose(
                    float(row.require("correct_answer_nll")),
                    correct_nll,
                    rel_tol=1e-6,
                    abs_tol=1e-6,
                ) or not math.isclose(
                    float(row.require("candidate_margin")),
                    wrong_nll - correct_nll,
                    rel_tol=1e-6,
                    abs_tol=1e-6,
                ):
                    raise ValueError("candidate summary metrics must match exact NLLs")
        reference_metadata = metadata_by_method[TINYWORLDS_REPORT_METHODS[0]]
        if any(
            metadata != reference_metadata
            for metadata in metadata_by_method.values()
        ):
            raise ValueError(
                "projected methods must contain identical paired query metadata"
            )
        query_ids = set(reference_metadata)
        if reference_query_ids is None:
            reference_query_ids = query_ids
        elif query_ids != reference_query_ids:
            raise ValueError("projected stages must contain identical held-out queries")
        reference_rows = tuple(
            row
            for row in bundle.candidate_scores
            if row.require("stage") == stage
            and row.require("method") == TINYWORLDS_REPORT_METHODS[0]
        )
        for name, actual, expected in (
            (
                "projected task IDs",
                {row.require("task_id") for row in reference_rows},
                set(TINYWORLDS_REPORT_TASK_IDS),
            ),
            (
                "projected prefix lengths",
                {row.require("prefix_length") for row in reference_rows},
                set(TINYWORLDS_REPORT_PREFIX_LENGTHS),
            ),
            (
                "projected cue regimes",
                {row.require("cue_regime") for row in reference_rows},
                set(TINYWORLDS_REPORT_CUE_REGIMES),
            ),
            (
                "projected query kinds",
                {row.require("query_kind") for row in reference_rows},
                set(TINYWORLDS_REPORT_QUERY_KINDS),
            ),
            (
                "projected modes",
                {row.require("mode") for row in reference_rows},
                {"open_book", "closed_book"},
            ),
        ):
            _require_exact_values(name, actual, expected)


def _candidate_score_record(row: KnowledgeQueryEvaluation) -> TinyWorldsRecord:
    entries: list[tuple[str, TinyWorldsScalar]] = [
        ("stage", row.stage),
        ("method", row.method),
        ("query_id", row.query_id),
        ("task_id", row.task_id),
        ("family_id", row.family_id),
        ("query_kind", row.query_kind),
        ("proof_id", row.proof_id),
        ("prefix_length", row.prefix_length),
        ("cue_regime", row.cue_regime),
        ("reasoning_type", row.reasoning_type),
        ("reasoning_depth", row.reasoning_depth),
        ("novelty_regime", row.novelty_regime),
        ("mode", row.mode),
    ]
    for index, (text, nll) in enumerate(
        zip(row.candidate_answer_texts, row.candidate_nll, strict=True)
    ):
        entries.extend(
            (
                (f"candidate_{index}_text", text),
                (f"candidate_{index}_nll", float(nll)),
            )
        )
    entries.extend(
        (
            ("correct_candidate_index", row.correct_candidate_index),
            ("predicted_candidate_index", row.predicted_candidate_index),
            ("candidate_correct", row.candidate_correct),
            ("candidate_margin", row.candidate_margin),
            ("correct_answer_nll", row.correct_answer_nll),
        )
    )
    return TinyWorldsRecord(tuple(entries))


def _aggregate_record(row: KnowledgeEvaluationAggregate) -> TinyWorldsRecord:
    return TinyWorldsRecord(
        (
            ("stage", row.stage),
            ("method", row.method),
            ("grouping_axis", row.grouping_axis),
            ("grouping_value", row.grouping_value),
            ("query_count", row.query_count),
            ("candidate_accuracy", row.candidate_accuracy),
            ("mean_candidate_margin", row.mean_candidate_margin),
            ("mean_correct_answer_nll", row.mean_correct_answer_nll),
            ("mean_routed_regret", row.mean_routed_regret),
            ("mean_task_oracle_regret", row.mean_task_oracle_regret),
            ("mean_best_hard_node_regret", row.mean_best_hard_node_regret),
            ("node_accuracy", row.node_accuracy),
            ("top_k_accuracy", row.top_k_accuracy),
            ("mean_address_entropy", row.mean_address_entropy),
            ("mean_address_margin", row.mean_address_margin),
            (
                "mean_hard_required_edge_recall",
                row.mean_hard_required_edge_recall,
            ),
            (
                "mean_soft_required_edge_coefficient",
                row.mean_soft_required_edge_coefficient,
            ),
        )
    )


def _routing_record(row: KnowledgeQueryEvaluation) -> TinyWorldsRecord:
    return TinyWorldsRecord(
        (
            ("stage", row.stage),
            ("method", row.method),
            ("query_id", row.query_id),
            ("task_id", row.task_id),
            ("cue_regime", row.cue_regime),
            ("prefix_length", row.prefix_length),
            ("selected_node_index", row.selected_node_index),
            ("task_oracle_node_index", row.task_oracle_node_index),
            ("best_hard_node_index", row.best_hard_node_index),
            ("node_accuracy", row.node_accuracy),
            ("top_k_accuracy", row.top_k_accuracy),
            ("address_entropy", row.address_entropy),
            ("address_margin", row.address_margin),
            ("routed_regret", row.routed_regret),
            ("task_oracle_regret", row.task_oracle_regret),
            ("best_hard_node_regret", row.best_hard_node_regret),
            ("required_edge_count", len(row.required_edge_ids)),
            ("hard_required_edge_recall", row.hard_required_edge_recall),
            (
                "soft_required_edge_mean_coefficient",
                row.soft_required_edge_mean_coefficient,
            ),
        )
    )


def _query_metadata_key(row: KnowledgeQueryEvaluation) -> tuple[object, ...]:
    return (
        row.task_id,
        row.family_id,
        row.query_kind,
        row.proof_id,
        row.support_ids,
        row.required_edge_ids,
        row.cue_regime,
        row.visible_cue_ids,
        row.eligible_task_ids,
        row.novelty_regime,
        row.reasoning_type,
        row.reasoning_depth,
        row.prefix_length,
        row.mode,
        row.oracle_node_ids,
        row.candidate_answer_texts,
        row.correct_candidate_index,
    )


def _jsonl_families(
    bundle: TinyWorldsReportBundle,
) -> tuple[tuple[str, tuple[TinyWorldsRecord, ...]], ...]:
    return (
        ("candidate_scores.jsonl", bundle.candidate_scores),
        ("knowledge_aggregates.jsonl", bundle.knowledge_aggregates),
        ("routing_records.jsonl", bundle.routing_records),
        (
            "natural_continuation_metrics.jsonl",
            bundle.natural_continuation_metrics,
        ),
        ("parent_search.jsonl", bundle.parent_search),
        ("checkpointed_transfer.jsonl", bundle.checkpointed_transfer),
        ("graph_recovery.jsonl", bundle.graph_recovery),
        ("revision_retention.jsonl", bundle.revision_retention),
        ("committed_node_drift.jsonl", bundle.committed_node_drift),
        ("memory_metrics.jsonl", bundle.memory_metrics),
        ("addressing_cost.jsonl", bundle.addressing_cost),
        ("gate_results.jsonl", bundle.gate_results),
        ("representative_queries.jsonl", bundle.representative_queries),
        ("selection_audit.jsonl", bundle.selection_audit),
        ("sequential_results.jsonl", bundle.sequential_results),
    )


def _final_aggregate_chart(
    bundle: TinyWorldsReportBundle,
    *,
    grouping_axis: str,
    metric: str,
    fallback_metric: str | None = None,
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    rows = tuple(
        row
        for row in bundle.knowledge_aggregates
        if row.require("stage") == TINYWORLDS_REPORT_STAGES[-1]
        and row.require("method") == "vamp_ebt_hopfield_soft"
        and row.require("grouping_axis") == grouping_axis
    )
    labels = tuple(str(row.require("grouping_value")) for row in rows)
    values: list[float] = []
    for row in rows:
        value = row.require(metric)
        if value is None and fallback_metric is not None:
            value = row.require(fallback_metric)
        values.append(float(value))
    if not labels:
        raise ValueError(f"chart aggregate is absent: {grouping_axis}")
    return labels, tuple(values)


def _write_bar_chart(
    path: Path,
    title: str,
    labels: tuple[str, ...],
    values: tuple[float, ...],
) -> None:
    if not labels or len(labels) != len(values):
        raise ValueError(f"chart {title!r} requires aligned nonempty values")
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError(f"chart {title!r} values must be finite and nonnegative")
    width = 960
    row_height = 34
    height = 84 + row_height * len(labels)
    maximum = max(max(values), 1.0)
    label_width = 300
    bar_width = width - label_width - 110
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="34" font-family="sans-serif" font-size="20" font-weight="600">{xml_escape(title)}</text>',
    ]
    for index, (label, value) in enumerate(zip(labels, values, strict=True)):
        y = 62 + index * row_height
        rendered_width = bar_width * value / maximum
        elements.extend(
            (
                f'<text x="24" y="{y + 18}" font-family="sans-serif" font-size="13">{xml_escape(label)}</text>',
                f'<rect x="{label_width}" y="{y}" width="{bar_width}" height="22" fill="#e5e7eb"/>',
                f'<rect x="{label_width}" y="{y}" width="{rendered_width:.3f}" height="22" fill="#2563eb"/>',
                f'<text x="{label_width + bar_width + 10}" y="{y + 17}" font-family="monospace" font-size="12">{value:.4f}</text>',
            )
        )
    elements.append("</svg>")
    _write_text(path, "\n".join(elements) + "\n")


def _write_report_html(path: Path, bundle: TinyWorldsReportBundle) -> None:
    gate_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(record.require('gate')))}</td>"
        f"<td>{html.escape(str(record.require('category')))}</td>"
        f"<td>{'pass' if record.require('passed') is True else 'FAIL'}</td>"
        "</tr>"
        for record in bundle.gate_results
    )
    method_rows = "".join(
        f"<tr><td>{html.escape(method)}</td><td>{'soft' if method.endswith('_soft') else 'stored/hard'}</td></tr>"
        for method in TINYWORLDS_REPORT_METHODS
    )
    artifact_links = "".join(
        f'<li><a href="{html.escape(filename)}">{html.escape(filename)}</a></li>'
        for filename in TINYWORLDS_REPORT_FILENAMES
        if filename != "report.html"
    )
    chart_cards = "".join(
        f'<figure><img src="{filename}" alt="{html.escape(title)}"><figcaption>{html.escape(title)}</figcaption></figure>'
        for title, filename in (
            ("Candidate reasoning", "candidate_reasoning.svg"),
            ("Cue routing", "cue_routing.svg"),
            ("Parent rank and transfer", "parent_rank_transfer.svg"),
            ("Revision retention", "revision_retention.svg"),
            ("Expected versus learned graph", "expected_vs_learned_graph.svg"),
        )
    )
    config = html.escape(
        json.dumps(
            json.loads(bundle.manifest.config_json),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    report = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(bundle.manifest.run_id)}</title>
<style>body{{margin:0;background:#f8fafc;color:#111827;font:15px/1.45 Arial,sans-serif}}main{{max-width:1180px;margin:auto;padding:32px 24px}}table{{border-collapse:collapse;width:100%;background:white}}th,td{{border:1px solid #d1d5db;padding:7px}}.charts{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:14px}}figure{{margin:0;background:white;border:1px solid #d1d5db;padding:10px}}img{{width:100%;height:auto}}pre{{overflow:auto;background:#111827;color:white;padding:16px}}</style>
</head><body><main>
<h1>TinyWorlds v1 · knowledge-graph continual learning</h1>
<p><code>{html.escape(bundle.manifest.run_id)}</code></p>
<p>Completed result SHA-256: <code>{bundle.completed_result_sha256}</code>. Scientific hypothesis failures remain visible below; only implementation-integrity failures block promotion.</p>
<section><h2>Methods</h2><table><thead><tr><th>Method</th><th>Role</th></tr></thead><tbody>{method_rows}</tbody></table></section>
<section><h2>Gate results</h2><table><thead><tr><th>Gate</th><th>Category</th><th>Outcome</th></tr></thead><tbody>{gate_rows}</tbody></table></section>
<section><h2>Charts</h2><div class="charts">{chart_cards}</div></section>
<section><h2>Artifacts</h2><ul>{artifact_links}</ul></section>
<section><h2>Content-addressed configuration</h2><pre>{config}</pre></section>
</main></body></html>
"""
    _write_text(path, report)


def _write_json(path: Path, payload: object) -> None:
    _write_text(
        path,
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )


def _write_jsonl(path: Path, records: tuple[TinyWorldsRecord, ...]) -> None:
    _write_text(path, "".join(record.canonical_json + "\n" for record in records))


def _write_text(path: Path, contents: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(contents, encoding="utf-8")
    temporary.replace(path)


def _directory_digest(directory: Path) -> str:
    digest = sha256()
    for path in sorted(item for item in directory.iterdir() if item.is_file()):
        relative = path.name.encode("utf-8")
        contents = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(len(contents).to_bytes(8, "little"))
        digest.update(contents)
    return digest.hexdigest()


def _validate_record_family(
    name: str,
    records: tuple[TinyWorldsRecord, ...],
) -> None:
    if not isinstance(records, tuple) or not records:
        raise ValueError(f"{name} must contain nonempty immutable records")
    if any(not isinstance(record, TinyWorldsRecord) for record in records):
        raise TypeError(f"{name} must contain TinyWorldsRecord values")


def _require_fields(
    records: tuple[TinyWorldsRecord, ...],
    fields: tuple[str, ...],
) -> None:
    for record in records:
        for field_name in fields:
            record.require(field_name)


def _require_stage_coverage(
    name: str,
    records: tuple[TinyWorldsRecord, ...],
) -> None:
    _require_exact_values(
        f"{name} stages",
        {record.require("stage") for record in records},
        set(TINYWORLDS_REPORT_STAGES),
    )


def _require_stage_method_matrix(
    name: str,
    records: tuple[TinyWorldsRecord, ...],
    methods: tuple[str, ...],
) -> None:
    represented = {
        (record.require("stage"), record.require("method")) for record in records
    }
    expected = {
        (stage, method)
        for stage in TINYWORLDS_REPORT_STAGES
        for method in methods
    }
    if represented != expected:
        raise ValueError(f"{name} stage/method coverage is incomplete")


def _require_exact_values(
    name: str,
    actual: set[object],
    expected: set[object],
) -> None:
    if actual != expected:
        raise ValueError(
            f"{name} coverage mismatch; missing={sorted(expected - actual, key=str)}, "
            f"extra={sorted(actual - expected, key=str)}"
        )


def _require_nonnegative_number(value: object, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{name} must be finite and nonnegative")


def _require_nonnegative_integer(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _require_rate(value: object, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{name} must be finite and lie in [0, 1]")


def _require_bool(value: object, name: str) -> None:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a bool")


def _sorted_records(
    records: tuple[TinyWorldsRecord, ...],
) -> tuple[TinyWorldsRecord, ...]:
    def record_order(record: TinyWorldsRecord) -> tuple[object, ...]:
        values = record.as_dict()
        if type(values.get("sequence_index")) is int:
            return (0, values["sequence_index"], record.canonical_json)
        if type(values.get("stage")) is int:
            return (1, values["stage"], record.canonical_json)
        task_id = values.get("task_id")
        if task_id in TINYWORLDS_REPORT_TASK_IDS:
            return (
                2,
                TINYWORLDS_REPORT_TASK_IDS.index(str(task_id)),
                record.canonical_json,
            )
        return (3, record.canonical_json)

    return tuple(sorted(records, key=record_order))


def _canonical_json_value(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("TinyWorlds configuration must contain finite JSON values") from error


def _parse_canonical_config(config_json: str) -> dict[str, object]:
    if not isinstance(config_json, str) or not config_json:
        raise ValueError("config_json must be a nonempty canonical JSON object")
    try:
        parsed = json.loads(config_json)
    except json.JSONDecodeError as error:
        raise ValueError("config_json must contain valid JSON") from error
    if not isinstance(parsed, dict) or not parsed:
        raise ValueError("config_json must encode a nonempty object")
    if _canonical_json_value(parsed) != config_json:
        raise ValueError("config_json must use canonical sorted compact JSON")
    return parsed


def _empty_identity_value(value: object) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _valid_scalar(value: object) -> bool:
    if value is None or type(value) in (str, bool, int):
        return True
    return type(value) is float and math.isfinite(value)


__all__ = [
    "TINYWORLDS_ADDRESSING_METHODS",
    "TINYWORLDS_ALLOCATOR_PEAK_LIMIT_BYTES",
    "TINYWORLDS_IMPLEMENTATION_GATES",
    "TINYWORLDS_NATURAL_CONTINUATION_METHODS",
    "TINYWORLDS_PARENT_COUNTERFACTUALS",
    "TINYWORLDS_REPORT_CUE_REGIMES",
    "TINYWORLDS_REPORT_FILENAMES",
    "TINYWORLDS_REPORT_METHODS",
    "TINYWORLDS_REPORT_PREFIX_LENGTHS",
    "TINYWORLDS_REPORT_QUERY_KINDS",
    "TINYWORLDS_REPORT_STAGES",
    "TINYWORLDS_REPORT_TASK_IDS",
    "TINYWORLDS_REQUIRED_CONFIG_FIELDS",
    "TinyWorldsCompletedResult",
    "TinyWorldsRecord",
    "TinyWorldsReportBundle",
    "TinyWorldsReportManifest",
    "atomically_promote_tinyworlds_report",
    "build_tinyworlds_report_bundle",
    "canonical_tinyworlds_config_json",
    "tinyworlds_report_directory",
    "validate_tinyworlds_report_artifact",
    "write_tinyworlds_report",
]

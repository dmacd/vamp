"""Deterministic analysis and publication for temporal consolidation."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
from html import escape
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import TYPE_CHECKING

import numpy as np

from apm.data.text.tinyworlds_nouns_v2.contracts import (
    TASK_IDS,
    canonical_json_bytes,
    record_sha256,
)
from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_contracts import (
    ARRIVAL_COUNT,
    BOOTSTRAP_REPETITIONS,
    EVALUATION_ROW_FORMAT,
    MERGE_ROW_FORMAT,
    REPORT_FORMAT,
    SEED,
    STUDY_ID,
    TEMPORAL_ORDERS,
    TIMING_ROW_FORMAT,
    TemporalChunk,
    empty_hierarchy,
    insert_arrival,
    temporal_arrivals,
)
from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_dashboard import (
    ProgressRecorder,
    publish_frozen_dashboard,
)
from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_distortion import (
    summarize_lineages,
    validate_distortion_rows,
)
from apm.continual.artifacts import (
    ChainedJsonlLedger,
    atomic_write,
    file_sha256,
)
from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_timing import (
    validate_timing_rows,
)

if TYPE_CHECKING:
    from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation import (
        OrderingArtifacts,
        SharedArtifacts,
        TemporalStudyInputs,
    )


os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "apm-matplotlib-cache"),
)


@dataclass(slots=True)
class _MetricAccumulator:
    story_count: int = 0
    story_nll_sum: float = 0.0
    suffix_total_nll: float = 0.0
    suffix_tokens: int = 0
    correct_tokens: int = 0
    oracle_story_nll_sum: float = 0.0
    oracle_regret_sum: float = 0.0
    oracle_agreement: int = 0
    noun_support_hits: int = 0
    noun_support_count: int = 0
    exact_route_hits: int = 0
    exact_route_count: int = 0
    entropy_sum: float = 0.0
    entropy_count: int = 0
    margin_sum: float = 0.0
    margin_count: int = 0
    prefix_tokens: int = 0
    candidate_evaluations: int = 0

    def add(self, row: Mapping[str, object]) -> None:
        """Accumulate one strict evaluation row in bounded state."""
        self.story_count += 1
        self.story_nll_sum += float(row["suffix_mean_nll"])
        self.suffix_total_nll += float(row["suffix_total_nll"])
        self.suffix_tokens += int(row["suffix_token_count"])
        self.correct_tokens += int(row["suffix_correct_tokens"])
        self.oracle_story_nll_sum += float(row["oracle_suffix_mean_nll"])
        self.oracle_regret_sum += float(row["oracle_regret"])
        self.oracle_agreement += int(row["selected_index"] == row["oracle_index"])
        if row["noun_support_hit"] is not None:
            self.noun_support_count += 1
            self.noun_support_hits += int(bool(row["noun_support_hit"]))
        if row["exact_noun_route_hit"] is not None:
            self.exact_route_count += 1
            self.exact_route_hits += int(bool(row["exact_noun_route_hit"]))
        if row["prefix_entropy"] is not None:
            self.entropy_count += 1
            self.entropy_sum += float(row["prefix_entropy"])
        if row["prefix_margin"] is not None:
            self.margin_count += 1
            self.margin_sum += float(row["prefix_margin"])
        candidate_count = len(_list(row["candidate_ids"], "candidate IDs"))
        self.prefix_tokens += int(row["prefix_token_count"]) * candidate_count
        self.candidate_evaluations += candidate_count

    def as_record(self) -> dict[str, object]:
        """Return derived story/token/oracle/addressing summaries."""
        if self.story_count <= 0 or self.suffix_tokens <= 0:
            raise ValueError("cannot summarize an empty temporal metric cell")
        return {
            "address_oracle_agreement": self.oracle_agreement / self.story_count,
            "candidate_evaluations": self.candidate_evaluations,
            "exact_noun_route_accuracy": (
                self.exact_route_hits / self.exact_route_count
                if self.exact_route_count
                else None
            ),
            "mean_oracle_regret": self.oracle_regret_sum / self.story_count,
            "mean_prefix_entropy": (
                self.entropy_sum / self.entropy_count if self.entropy_count else None
            ),
            "mean_prefix_margin": (
                self.margin_sum / self.margin_count if self.margin_count else None
            ),
            "model_forward_equivalent_prefix_tokens": self.prefix_tokens,
            "noun_support_accuracy": (
                self.noun_support_hits / self.noun_support_count
                if self.noun_support_count
                else None
            ),
            "oracle_story_mean_nll": self.oracle_story_nll_sum / self.story_count,
            "story_count": self.story_count,
            "story_mean_nll": self.story_nll_sum / self.story_count,
            "suffix_token_accuracy": self.correct_tokens / self.suffix_tokens,
            "token_count": self.suffix_tokens,
            "token_mean_nll": self.suffix_total_nll / self.suffix_tokens,
        }


def publish_temporal_consolidation_report(
    inputs: TemporalStudyInputs,
    shared: SharedArtifacts,
    orderings: Sequence[OrderingArtifacts],
    evaluation_directories: Mapping[str, Path],
    final_control_directory: Path,
    distortion_paths: Mapping[str, Path],
    timing_path: Path,
    progress_recorder: ProgressRecorder,
    *,
    execution: Mapping[str, float],
    allocator: Mapping[str, object],
) -> tuple[Path, Path, Path]:
    """Publish CSV/SVG/Graphviz evidence and standalone Markdown/HTML reports."""
    output = inputs.result_directory
    output.mkdir(parents=True, exist_ok=True)
    analysis = analyze_temporal_study(
        inputs,
        shared,
        orderings,
        evaluation_directories,
        final_control_directory,
        distortion_paths,
        timing_path,
        execution=execution,
        allocator=allocator,
    )
    csv_paths = _publish_csv_exports(output, analysis)
    plot_paths = _publish_plots(output, analysis)
    graph_paths = tuple(
        path
        for ordering in orderings
        for path in _publish_lineage_graphs(output, ordering)
    )
    analysis_core = {
        "allocator": dict(allocator),
        "analysis": analysis,
        "contract_sha256": inputs.contract_sha256,
        "execution": dict(execution),
        "format": REPORT_FORMAT,
    }
    analysis_record = {
        **analysis_core,
        "analysis_sha256": record_sha256(analysis_core),
    }
    analysis_path = atomic_write(
        output / "analysis.json",
        canonical_json_bytes(analysis_record),
    )
    markdown_path = atomic_write(
        output / "report.md",
        _markdown_report(inputs, analysis, execution, allocator).encode("utf-8"),
    )
    embedded_svgs = {
        path.name: path.read_text(encoding="utf-8")
        for path in (*plot_paths, *graph_paths)
        if path.suffix == ".svg"
    }
    html_path = atomic_write(
        output / "report.html",
        _html_report(inputs, analysis, execution, allocator, embedded_svgs).encode(
            "utf-8"
        ),
    )
    dashboard_path = publish_frozen_dashboard(
        output / "dashboard.html",
        progress_recorder.snapshot(),
    )
    artifact_paths = (
        output / "contract.json",
        *(path for path in (output / "allocator.json", output / "execution.json") if path.is_file()),
        analysis_path,
        markdown_path,
        html_path,
        dashboard_path,
        *csv_paths,
        *plot_paths,
        *graph_paths,
    )
    manifest_core = {
        "artifacts": {
            path.relative_to(output).as_posix(): file_sha256(path)
            for path in sorted(artifact_paths)
        },
        "contract_sha256": inputs.contract_sha256,
        "format": f"{STUDY_ID}-publication-v1",
        "schema_version": 1,
    }
    manifest_path = atomic_write(
        output / "manifest.json",
        canonical_json_bytes(
            {**manifest_core, "manifest_sha256": record_sha256(manifest_core)}
        ),
    )
    return markdown_path, html_path, manifest_path


def analyze_temporal_study(
    inputs: TemporalStudyInputs,
    shared: SharedArtifacts,
    orderings: Sequence[OrderingArtifacts],
    evaluation_directories: Mapping[str, Path],
    final_control_directory: Path,
    distortion_paths: Mapping[str, Path],
    timing_path: Path,
    *,
    execution: Mapping[str, float],
    allocator: Mapping[str, object],
) -> dict[str, object]:
    """Stream all ledgers into bounded aggregate, task, cost, and audit records."""
    accumulators: dict[tuple[str, str, int, str, str], _MetricAccumulator] = {}
    final_pairs: dict[tuple[str, str, str, str], dict[str, float]] = {}
    confusion: Counter[tuple[str, str]] = Counter()
    selections: Counter[tuple[str, int, str, int, int, str]] = Counter()
    ledger_provenance: list[dict[str, object]] = []
    directory_specs = tuple(
        (order, Path(evaluation_directories[order])) for order in TEMPORAL_ORDERS
    ) + (("offline", Path(final_control_directory)),)
    for directory_order, directory in directory_specs:
        for path in sorted(directory.glob("*.jsonl")):
            ledger = ChainedJsonlLedger(path, EVALUATION_ROW_FORMAT)
            ledger_provenance.append(
                {
                    "path": f"{directory.name}/{path.name}",
                    "row_count": len(ledger.rows),
                    "sha256": file_sha256(path),
                }
            )
            for row in ledger.rows:
                if row.get("contract_sha256") != inputs.contract_sha256:
                    raise ValueError("report encountered a cross-contract evaluation row")
                order = str(row["order"] or directory_order)
                key = (
                    order,
                    str(row["dataset"]),
                    int(row["stage"]),
                    str(row["method"]),
                    str(row["task_id"]),
                )
                accumulators.setdefault(key, _MetricAccumulator()).add(row)
                if int(row["stage"]) == ARRIVAL_COUNT and row["dataset"] in (
                    "macro",
                    "final",
                ):
                    final_pairs[
                        (
                            order,
                            str(row["method"]),
                            str(row["task_id"]),
                            str(row["story_id"]),
                        )
                    ] = {
                        "noun_support": float(bool(row["noun_support_hit"])),
                        "oracle_regret": float(row["oracle_regret"]),
                        "story_nll": float(row["suffix_mean_nll"]),
                        "token_accuracy": float(row["suffix_token_accuracy"]),
                    }
                if row["method"] == "independent_noun_exhaustive":
                    confusion[(str(row["task_id"]), str(row["selected_candidate_id"]))] += 1
                if row["method"] == "log_t":
                    level, end = _selected_interval(str(row["selected_candidate_id"]))
                    selections[
                        (
                            order,
                            int(row["stage"]),
                            str(row["task_id"]),
                            level,
                            max(0, int(row["stage"]) - end),
                            str(row["selected_candidate_id"]),
                        )
                    ] += 1
    metric_rows = tuple(
        {
            "dataset": dataset,
            "method": method,
            "order": order,
            "stage": stage,
            "task_id": task_id,
            **accumulator.as_record(),
        }
        for (order, dataset, stage, method, task_id), accumulator in sorted(
            accumulators.items()
        )
    )
    stage_rows = _aggregate_metric_rows(metric_rows)
    per_task_rows = tuple(
        row
        for row in metric_rows
        if row["stage"] == ARRIVAL_COUNT and row["dataset"] in ("macro", "final")
    )
    final_rows = tuple(
        row
        for row in stage_rows
        if row["stage"] == ARRIVAL_COUNT and row["dataset"] in ("macro", "final")
    )
    forgetting_rows = _forgetting_rows(metric_rows)
    bootstrap_rows = _ordering_bootstrap(final_pairs)
    ordering_by_name = {ordering.order: ordering for ordering in orderings}
    distortion = _distortion_analysis(
        inputs,
        ordering_by_name,
        distortion_paths,
    )
    timing_ledger = ChainedJsonlLedger(timing_path, TIMING_ROW_FORMAT)
    validate_timing_rows(timing_ledger.rows, inputs.contract_sha256)
    cost_rows = tuple(
        row
        for order in TEMPORAL_ORDERS
        for row in _cost_rows(inputs, shared, ordering_by_name[order])
    )
    selection_rows = tuple(
        {
            "age": age,
            "count": count,
            "level": level,
            "order": order,
            "selected_candidate_id": candidate,
            "stage": stage,
            "task_id": task,
        }
        for (order, stage, task, level, age, candidate), count in sorted(
            selections.items()
        )
    )
    return {
        "aggregate": final_rows,
        "allocator": dict(allocator),
        "arrival": tuple(row for row in stage_rows if row["dataset"] == "sentinel"),
        "bootstrap": bootstrap_rows,
        "confusion": tuple(
            {
                "count": count,
                "selected_candidate_id": selected,
                "task_id": task,
            }
            for (task, selected), count in sorted(confusion.items())
        ),
        "cost": cost_rows,
        "distortion": distortion,
        "execution": dict(execution),
        "forgetting": forgetting_rows,
        "ledger_provenance": tuple(ledger_provenance),
        "per_task": per_task_rows,
        "provenance": {
            "base_parameter_checksum": inputs.selected_base.reference.parameter_checksum,
            "base_training_sha256": inputs.selected_base.training_sha256,
            "canonical_artifact_hashes": dict(inputs.canonical_hashes),
            "contract_sha256": inputs.contract_sha256,
            "final_vamp_tensor_checksum": inputs.final_vamp_tensor_checksum,
            "partition_sha256": inputs.partition.partition_sha256,
            "training": _training_artifact_summary(shared, orderings),
        },
        "selection": selection_rows,
        "stage": stage_rows,
        "timing": timing_ledger.rows,
    }


def _aggregate_metric_rows(
    task_rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    grouped: dict[tuple[str, str, int, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in task_rows:
        grouped[
            (
                str(row["order"]),
                str(row["dataset"]),
                int(row["stage"]),
                str(row["method"]),
            )
        ].append(row)
    result = []
    for (order, dataset, stage, method), rows in sorted(grouped.items()):
        story_count = sum(int(row["story_count"]) for row in rows)
        token_count = sum(int(row["token_count"]) for row in rows)
        result.append(
            {
                "address_oracle_agreement": _weighted(
                    rows,
                    "address_oracle_agreement",
                    "story_count",
                ),
                "candidate_evaluations": sum(
                    int(row["candidate_evaluations"]) for row in rows
                ),
                "dataset": dataset,
                "exact_noun_route_accuracy": _optional_weighted(
                    rows,
                    "exact_noun_route_accuracy",
                    "story_count",
                ),
                "mean_oracle_regret": _weighted(
                    rows,
                    "mean_oracle_regret",
                    "story_count",
                ),
                "mean_prefix_entropy": _optional_weighted(
                    rows,
                    "mean_prefix_entropy",
                    "story_count",
                ),
                "mean_prefix_margin": _optional_weighted(
                    rows,
                    "mean_prefix_margin",
                    "story_count",
                ),
                "method": method,
                "model_forward_equivalent_prefix_tokens": sum(
                    int(row["model_forward_equivalent_prefix_tokens"])
                    for row in rows
                ),
                "noun_support_accuracy": _optional_weighted(
                    rows,
                    "noun_support_accuracy",
                    "story_count",
                ),
                "oracle_story_mean_nll": _weighted(
                    rows,
                    "oracle_story_mean_nll",
                    "story_count",
                ),
                "order": order,
                "stage": stage,
                "story_count": story_count,
                "story_mean_nll": _weighted(rows, "story_mean_nll", "story_count"),
                "suffix_token_accuracy": _weighted(
                    rows,
                    "suffix_token_accuracy",
                    "token_count",
                ),
                "token_count": token_count,
                "token_mean_nll": _weighted(rows, "token_mean_nll", "token_count"),
            }
        )
    return tuple(result)


def _forgetting_rows(
    metric_rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in metric_rows:
        if row["dataset"] == "macro":
            grouped[
                (str(row["order"]), str(row["method"]), str(row["task_id"]))
            ].append(row)
    result = []
    for (order, method, task_id), rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: int(row["stage"]))
        best = min(float(row["story_mean_nll"]) for row in ordered)
        first = float(ordered[0]["story_mean_nll"])
        final = float(ordered[-1]["story_mean_nll"])
        result.append(
            {
                "backward_transfer": first - final,
                "best_story_mean_nll": best,
                "final_forgetting": final - best,
                "final_stage": int(ordered[-1]["stage"]),
                "final_story_mean_nll": final,
                "first_evaluated_stage": int(ordered[0]["stage"]),
                "first_story_mean_nll": first,
                "method": method,
                "order": order,
                "task_id": task_id,
                "worst_forgetting": max(
                    float(row["story_mean_nll"]) - best for row in ordered
                ),
            }
        )
    return tuple(result)


def _ordering_bootstrap(
    values: Mapping[tuple[str, str, str, str], Mapping[str, float]],
) -> tuple[dict[str, object], ...]:
    metrics = ("story_nll", "token_accuracy", "oracle_regret", "noun_support")
    rows = []
    for metric in metrics:
        paired_by_task: dict[str, np.ndarray] = {}
        for task_id in TASK_IDS:
            story_ids = sorted(
                story_id
                for order, method, task, story_id in values
                if order == "blocked" and method == "log_t" and task == task_id
            )
            differences = np.asarray(
                [
                    values[("round_robin", "log_t", task_id, story_id)][metric]
                    - values[("blocked", "log_t", task_id, story_id)][metric]
                    for story_id in story_ids
                ],
                dtype=np.float64,
            )
            if not len(differences):
                raise ValueError(f"bootstrap is missing final paired rows for {task_id}")
            paired_by_task[task_id] = differences
        observed = math.fsum(
            float(np.sum(values_for_task, dtype=np.float64))
            for values_for_task in paired_by_task.values()
        ) / sum(len(values_for_task) for values_for_task in paired_by_task.values())
        rng = np.random.default_rng(SEED)
        replicates = np.empty((BOOTSTRAP_REPETITIONS,), dtype=np.float64)
        for repetition in range(BOOTSTRAP_REPETITIONS):
            total = 0.0
            count = 0
            for task_id in TASK_IDS:
                task_values = paired_by_task[task_id]
                indices = rng.integers(0, len(task_values), size=len(task_values))
                total += float(np.sum(task_values[indices], dtype=np.float64))
                count += len(task_values)
            replicates[repetition] = total / count
        lower, upper = np.quantile(replicates, (0.025, 0.975))
        rows.append(
            {
                "comparison": "round_robin_minus_blocked",
                "estimate": observed,
                "lower_95": float(lower),
                "metric": metric,
                "repetitions": BOOTSTRAP_REPETITIONS,
                "seed": SEED,
                "upper_95": float(upper),
            }
        )
    return tuple(rows)


def _distortion_analysis(
    inputs: TemporalStudyInputs,
    orderings: Mapping[str, OrderingArtifacts],
    paths: Mapping[str, Path],
) -> dict[str, object]:
    merge_rows: list[dict[str, object]] = []
    lineage_rows: list[dict[str, object]] = []
    provenance = []
    for order in TEMPORAL_ORDERS:
        ledger = ChainedJsonlLedger(paths[order], MERGE_ROW_FORMAT)
        validate_distortion_rows(ledger.rows, inputs.contract_sha256, order)
        provenance.append(
            {
                "order": order,
                "row_count": len(ledger.rows),
                "sha256": file_sha256(paths[order]),
            }
        )
        grouped: dict[tuple[str, str, str], list[Mapping[str, object]]] = defaultdict(list)
        metadata: dict[str, Mapping[str, object]] = {}
        for row in ledger.rows:
            grouped[
                (
                    str(row["parent_chunk_id"]),
                    str(row["child_chunk_id"]),
                    str(row["kind"]),
                )
            ].append(row)
            metadata[str(row["parent_chunk_id"])] = row
        child_deltas = {
            key: _weighted_increment(rows)
            for key, rows in grouped.items()
        }
        for parent_id in sorted({key[0] for key in grouped}):
            for kind in ("source", "validation"):
                values = tuple(
                    value
                    for (candidate_parent, _, candidate_kind), value in child_deltas.items()
                    if candidate_parent == parent_id and candidate_kind == kind
                )
                if len(values) != 2:
                    raise ValueError("merge summary requires two child distortions")
                row = metadata[parent_id]
                merge_rows.append(
                    {
                        "delta": max(values),
                        "end_arrival": int(row["parent_end_arrival"]),
                        "kind": kind,
                        "left_signed_increment": values[0],
                        "level": int(row["parent_level"]),
                        "order": order,
                        "parent_chunk_id": parent_id,
                        "right_signed_increment": values[1],
                        "start_arrival": int(row["parent_start_arrival"]),
                    }
                )
        lineage_rows.extend(
            audit.as_record()
            for audit in summarize_lineages(
                ledger.rows,
                orderings[order].final_state,
            )
        )
    return {
        "lineage": tuple(lineage_rows),
        "merge": tuple(merge_rows),
        "provenance": tuple(provenance),
    }


def _cost_rows(
    inputs: TemporalStudyInputs,
    shared: SharedArtifacts,
    ordering: OrderingArtifacts,
) -> tuple[dict[str, object], ...]:
    level_zero_by_shard = shared.level_zero_by_shard
    artifacts = ordering.chunks_by_id
    state = empty_hierarchy(ordering.order)
    created: set[str] = set()
    cumulative_updates = 0
    cumulative_runtime = 0.0
    rows = []
    for arrival, shard in enumerate(
        temporal_arrivals(inputs.shards, ordering.order),
        start=1,
    ):
        state, merges = insert_arrival(state, shard)
        level_zero = next(
            chunk
            for chunk in (
                *state.active_chunks,
                *(candidate for merge in merges for candidate in (merge.left, merge.right)),
            )
            if chunk.level == 0 and chunk.end_arrival == arrival
        )
        created.add(level_zero.chunk_id)
        insertion_artifacts = [level_zero_by_shard[shard.shard_id]]
        for merge in merges:
            created.add(merge.parent.chunk_id)
            insertion_artifacts.append(artifacts[merge.parent.chunk_id])
        insertion_updates = sum(
            artifact.optimizer_updates for artifact in insertion_artifacts
        )
        insertion_runtime = math.fsum(
            artifact.runtime_seconds for artifact in insertion_artifacts
        )
        cumulative_updates += insertion_updates
        cumulative_runtime += insertion_runtime
        active_ids = {chunk.chunk_id for chunk in state.active_chunks}
        active_bytes = sum(
            (artifacts[chunk_id].directory / "adapter.safetensors").stat().st_size
            for chunk_id in active_ids
        )
        archived_bytes = sum(
            (artifacts[chunk_id].directory / "adapter.safetensors").stat().st_size
            for chunk_id in created
        )
        rows.append(
            {
                "active_adapter_bytes": active_bytes,
                "active_adapter_count": len(active_ids),
                "address_candidate_count_including_base": len(active_ids) + 1,
                "amortized_optimizer_updates_per_arrival": cumulative_updates / arrival,
                "archived_adapter_bytes": archived_bytes,
                "archived_adapter_count": len(created),
                "arrival": arrival,
                "carry_depth": len(merges),
                "cumulative_insertion_runtime_seconds": cumulative_runtime,
                "cumulative_optimizer_updates": cumulative_updates,
                "insertion_optimizer_updates": insertion_updates,
                "insertion_runtime_seconds": insertion_runtime,
                "order": ordering.order,
                "source_shard_id": shard.shard_id,
                "task_id": shard.task_id,
            }
        )
    return tuple(rows)


def _training_artifact_summary(
    shared: SharedArtifacts,
    orderings: Sequence[OrderingArtifacts],
) -> dict[str, object]:
    shared_adapters = tuple(
        artifact
        for _, artifact in (*shared.level_zero, *shared.independent_noun)
    ) + (shared.iid_lora,)
    ordering_adapters = tuple(
        artifact
        for ordering in orderings
        for artifact in (*ordering.sequential, *(value for _, value in ordering.chunks))
    )
    unique = {
        artifact.adapter_sha256: artifact
        for artifact in (*shared_adapters, *ordering_adapters)
    }
    return {
        "adapter_artifact_count": len(unique),
        "adapter_identity_sha256": record_sha256(sorted(unique)),
        "adapter_optimizer_updates": sum(
            artifact.optimizer_updates for artifact in unique.values()
        ),
        "adapter_runtime_seconds": math.fsum(
            artifact.runtime_seconds for artifact in unique.values()
        ),
        "full_model_parameter_checksum": shared.iid_full_model.checkpoint.parameter_checksum,
        "full_model_runtime_seconds": shared.iid_full_model.runtime_seconds,
        "full_model_updates": shared.iid_full_model.optimizer_updates,
        "maximum_allocator_peak_bytes": max(
            (
                shared.iid_full_model.allocator_peak_bytes,
                *(artifact.allocator_peak_bytes for artifact in unique.values()),
            )
        ),
    }


def _publish_csv_exports(
    output: Path,
    analysis: Mapping[str, object],
) -> tuple[Path, ...]:
    distortion = _object(analysis["distortion"], "distortion")
    exports = (
        ("aggregate.csv", _records(analysis["aggregate"], "aggregate")),
        ("per-task.csv", _records(analysis["per_task"], "per-task")),
        ("stage.csv", _records(analysis["stage"], "stage")),
        ("arrival.csv", _records(analysis["arrival"], "arrival")),
        ("forgetting.csv", _records(analysis["forgetting"], "forgetting")),
        ("bootstrap.csv", _records(analysis["bootstrap"], "bootstrap")),
        ("confusion.csv", _records(analysis["confusion"], "confusion")),
        ("selection.csv", _records(analysis["selection"], "selection")),
        ("cost.csv", _records(analysis["cost"], "cost")),
        ("merge.csv", _records(distortion["merge"], "merge")),
        ("lineage.csv", _records(distortion["lineage"], "lineage")),
        ("timing.csv", _records(analysis["timing"], "timing")),
    )
    return tuple(
        atomic_write(output / name, _csv_bytes(rows))
        for name, rows in exports
    )


def _csv_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    if not rows:
        return b"\n"
    fields = tuple(sorted({key for row in rows for key in row}))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(
        {
            field: (
                json.dumps(value, sort_keys=True, separators=(",", ":"))
                if isinstance(value, (dict, list, tuple))
                else value
            )
            for field, value in row.items()
        }
        for row in rows
    )
    return stream.getvalue().encode("utf-8")


def _selected_interval(candidate_id: str) -> tuple[int, int]:
    if candidate_id == "base":
        return -1, 0
    match = re.fullmatch(r"interval-(\d{3})-(\d{3})-l(\d+)", candidate_id)
    if match is None:
        raise ValueError(f"unknown temporal candidate ID: {candidate_id}")
    return int(match.group(3)), int(match.group(2))


def _weighted_increment(rows: Sequence[Mapping[str, object]]) -> float:
    tokens = sum(int(row["token_count"]) for row in rows)
    return math.fsum(
        float(row["signed_increment"]) * int(row["token_count"])
        for row in rows
    ) / tokens


def _weighted(
    rows: Sequence[Mapping[str, object]],
    value_field: str,
    weight_field: str,
) -> float:
    total_weight = sum(int(row[weight_field]) for row in rows)
    return math.fsum(
        float(row[value_field]) * int(row[weight_field]) for row in rows
    ) / total_weight


def _optional_weighted(
    rows: Sequence[Mapping[str, object]],
    value_field: str,
    weight_field: str,
) -> float | None:
    available = tuple(row for row in rows if row[value_field] is not None)
    return _weighted(available, value_field, weight_field) if available else None


def _publish_plots(
    output: Path,
    analysis: Mapping[str, object],
) -> tuple[Path, ...]:
    from matplotlib import pyplot as plt

    plt.rcParams.update(
        {
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "font.size": 10,
            "legend.fontsize": 9,
            "svg.hashsalt": STUDY_ID,
        }
    )
    colors = {"base": "#666666", "sequential_lora": "#D55E00", "log_t": "#0072B2"}
    labels = {"base": "Frozen base", "sequential_lora": "Sequential LoRA", "log_t": "Log-t bank"}
    stage = _records(analysis["stage"], "stage")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for axis, order in zip(axes, TEMPORAL_ORDERS):
        for method in ("base", "sequential_lora", "log_t"):
            rows = sorted(
                (
                    row
                    for row in stage
                    if row["order"] == order
                    and row["dataset"] == "macro"
                    and row["method"] == method
                ),
                key=lambda row: int(row["stage"]),
            )
            axis.plot(
                [int(row["stage"]) for row in rows],
                [float(row["story_mean_nll"]) for row in rows],
                label=labels[method],
                color=colors[method],
                linewidth=2,
            )
        axis.set_title(order.replace("_", " ").title())
        axis.set_xlabel("Shard arrivals")
        axis.set_ylabel("Suffix story NLL (nats/token)")
        axis.grid(alpha=0.25)
        axis.legend()
    quality_path = _save_figure(
        output / "suffix-quality-over-time.svg",
        figure,
        "Suffix quality over temporal arrivals",
        "Two panels compare frozen base, sequential LoRA, and routed log-t suffix story NLL at full validation checkpoints.",
    )

    cost = _records(analysis["cost"], "cost")
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    cost_fields = (
        ("active_adapter_count", "Live adapters", "Adapters"),
        ("active_adapter_bytes", "Live adapter memory", "Bytes"),
        ("insertion_optimizer_updates", "Insertion work", "Optimizer updates"),
        ("insertion_runtime_seconds", "Insertion wall time", "Seconds"),
    )
    for axis, (field, title, unit) in zip(axes.flat, cost_fields):
        for order, color in zip(TEMPORAL_ORDERS, ("#0072B2", "#CC79A7")):
            rows = [row for row in cost if row["order"] == order]
            axis.plot(
                [int(row["arrival"]) for row in rows],
                [float(row[field]) for row in rows],
                label=order.replace("_", " "),
                color=color,
                linewidth=1.7,
            )
        axis.set_title(title)
        axis.set_xlabel("Shard arrivals")
        axis.set_ylabel(unit)
        axis.grid(alpha=0.25)
    axes[0, 0].legend()
    growth_path = _save_figure(
        output / "temporal-cost-growth.svg",
        figure,
        "Temporal bank growth and insertion cost",
        "Separate panels preserve incomparable adapter-count, byte, optimizer-update, and wall-time units.",
    )

    distortion = _object(analysis["distortion"], "distortion")
    merge_rows = _records(distortion["merge"], "merge")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for axis, kind in zip(axes, ("source", "validation")):
        for order, marker, color in zip(
            TEMPORAL_ORDERS,
            ("o", "s"),
            ("#0072B2", "#D55E00"),
        ):
            summaries = []
            for level in sorted(
                {int(row["level"]) for row in merge_rows if row["kind"] == kind}
            ):
                values = [
                    float(row["delta"])
                    for row in merge_rows
                    if row["kind"] == kind
                    and row["order"] == order
                    and int(row["level"]) == level
                ]
                summaries.append((level, float(np.mean(values)), max(values)))
            axis.plot(
                [value[0] for value in summaries],
                [value[1] for value in summaries],
                marker=marker,
                color=color,
                label=f"{order.replace('_', ' ')} mean",
            )
            axis.scatter(
                [value[0] for value in summaries],
                [value[2] for value in summaries],
                marker="x",
                color=color,
                label=f"{order.replace('_', ' ')} max",
            )
        axis.axhline(0.0, color="#222222", linewidth=1)
        axis.set_title(f"{kind.title()} distortion")
        axis.set_xlabel("Parent level")
        axis.set_ylabel("Parent − child NLL (nats/token)")
        axis.grid(alpha=0.25)
    axes[0].legend(ncol=2)
    distortion_path = _save_figure(
        output / "merge-distortion.svg",
        figure,
        "Merge distortion by hierarchy level",
        "Mean and maximum always-accepted parent-minus-child NLL are shown separately for source and sentinel suffix evidence.",
    )

    final = _records(analysis["aggregate"], "aggregate")
    ordered_final = sorted(
        final,
        key=lambda row: (str(row["order"]), str(row["method"])),
    )
    figure, axis = plt.subplots(figsize=(12, 5), constrained_layout=True)
    names = [f"{row['order']}\n{str(row['method']).replace('_', ' ')}" for row in ordered_final]
    axis.bar(
        np.arange(len(ordered_final)),
        [float(row["story_mean_nll"]) for row in ordered_final],
        color="#0072B2",
    )
    axis.set_xticks(np.arange(len(names)), names, rotation=35, ha="right")
    axis.set_ylabel("Suffix story NLL (nats/token)")
    axis.set_title("Final-checkpoint quality")
    axis.grid(axis="y", alpha=0.25)
    final_path = _save_figure(
        output / "final-quality.svg",
        figure,
        "Final suffix quality across methods",
        "Bars show story-weighted midpoint-suffix NLL for both temporal orders and offline controls.",
    )

    timing = _records(analysis["timing"], "timing")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    prefix = [row for row in timing if row["kind"] == "prefix"]
    for width in sorted({int(row["prefix_width"]) for row in prefix}):
        rows = [row for row in prefix if int(row["prefix_width"]) == width]
        axes[0].plot(
            [int(row["candidate_count"]) + 1 for row in rows],
            [1_000 * float(row["warm_mean_seconds"]) for row in rows],
            alpha=0.55,
            linewidth=1,
        )
    suffix = [row for row in timing if row["kind"] == "suffix"]
    axes[1].plot(
        [int(row["candidate_count"]) + 1 for row in suffix],
        [1_000 * float(row["warm_mean_seconds"]) for row in suffix],
        marker="o",
        color="#D55E00",
    )
    for axis, title in zip(axes, ("Prefix routing shapes", "One suffix candidate kernel")):
        axis.set_title(title)
        axis.set_xlabel("Candidates including base")
        axis.set_ylabel("Warm kernel latency (ms)")
        axis.grid(alpha=0.25)
    timing_path = _save_figure(
        output / "warm-kernel-timing.svg",
        figure,
        "Synchronized warm GPU kernel timing",
        "Prefix-width traces and suffix candidate kernels are separated; cold compilation remains in timing.csv.",
    )
    return quality_path, growth_path, distortion_path, final_path, timing_path


def _save_figure(path: Path, figure, title: str, description: str) -> Path:
    from matplotlib import pyplot as plt

    stream = io.StringIO()
    figure.savefig(
        stream,
        format="svg",
        metadata={"Date": None, "Description": description, "Title": title},
    )
    plt.close(figure)
    svg = _accessible_svg(stream.getvalue(), title, description)
    return atomic_write(path, svg.encode("utf-8"))


def _publish_lineage_graphs(
    output: Path,
    ordering: OrderingArtifacts,
) -> tuple[Path, ...]:
    compact_dot = _compact_lineage_dot(ordering)
    full_dot = _full_lineage_dot(ordering)
    paths = []
    for kind, dot_source, description in (
        (
            "compact",
            compact_dot,
            "The nine deployed final temporal chunks point to a base-relative deployment bank.",
        ),
        (
            "full",
            full_dot,
            "All 192 level-zero shards and 183 deterministic merge parents are present.",
        ),
    ):
        stem = f"lineage-{ordering.order}-{kind}"
        dot_path = atomic_write(output / f"{stem}.dot", dot_source.encode("utf-8"))
        result = subprocess.run(
            ("dot", "-Tsvg"),
            input=dot_source.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Graphviz failed for {stem}: {result.stderr.decode('utf-8', 'replace')}"
            )
        title = f"{ordering.order.replace('_', ' ').title()} {kind} lineage"
        svg_path = atomic_write(
            output / f"{stem}.svg",
            _accessible_svg(
                result.stdout.decode("utf-8"),
                title,
                description,
            ).encode("utf-8"),
        )
        paths.extend((dot_path, svg_path))
    return tuple(paths)


def _compact_lineage_dot(ordering: OrderingArtifacts) -> str:
    lines = [
        "digraph temporal_compact {",
        'graph [rankdir="LR", bgcolor="transparent", label="Deployed bank", labelloc="t"];',
        'node [shape="box", style="rounded,filled", fontname="DejaVu Sans", fontsize="11", color="#24425f"];',
        'edge [color="#516579"];',
        'deployment [shape="ellipse", label="Base + 9 live adapters", fillcolor="#E6F2FF"];',
    ]
    for chunk in ordering.final_state.active_chunks:
        identifier = _dot_id(chunk.chunk_id)
        label = _chunk_label(chunk)
        lines.append(
            f'{identifier} [label="{label}", fillcolor="{_chunk_color(chunk)}", penwidth="2"];'
        )
        lines.append(f"{identifier} -> deployment;")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _full_lineage_dot(ordering: OrderingArtifacts) -> str:
    chunks: dict[str, TemporalChunk] = {
        chunk.chunk_id: chunk for chunk in ordering.final_state.active_chunks
    }
    for merge in ordering.merges:
        chunks.update(
            {
                merge.left.chunk_id: merge.left,
                merge.right.chunk_id: merge.right,
                merge.parent.chunk_id: merge.parent,
            }
        )
    final_ids = {chunk.chunk_id for chunk in ordering.final_state.active_chunks}
    lines = [
        "digraph temporal_full {",
        'graph [rankdir="BT", bgcolor="transparent", nodesep="0.12", ranksep="0.25"];',
        'node [shape="box", style="filled", fontname="DejaVu Sans", fontsize="7", color="#516579"];',
        'edge [color="#7b8794", arrowsize="0.45"];',
    ]
    for chunk in sorted(chunks.values(), key=lambda value: (value.level, value.start_arrival)):
        penwidth = "2.5" if chunk.chunk_id in final_ids else "0.8"
        lines.append(
            f'{_dot_id(chunk.chunk_id)} [label="{_chunk_label(chunk)}", '
            f'fillcolor="{_chunk_color(chunk)}", penwidth="{penwidth}"];'
        )
    lines.extend(
        f"{_dot_id(child.chunk_id)} -> {_dot_id(merge.parent.chunk_id)};"
        for merge in ordering.merges
        for child in (merge.left, merge.right)
    )
    lines.append("}")
    return "\n".join(lines) + "\n"


def _dot_id(chunk_id: str) -> str:
    return f'"chunk_{chunk_id}"'


def _chunk_label(chunk: TemporalChunk) -> str:
    nouns = ", ".join(
        f"{task}:{count}" for task, count in chunk.task_counts[:3]
    )
    if len(chunk.task_counts) > 3:
        nouns += f", +{len(chunk.task_counts) - 3}"
    return (
        f"L{chunk.level} · {chunk.start_arrival}–{chunk.end_arrival}"
        f"\\n{nouns}"
    ).replace('"', '\\"')


def _chunk_color(chunk: TemporalChunk) -> str:
    entropy_fraction = (
        chunk.noun_entropy / math.log(len(TASK_IDS)) if chunk.noun_entropy > 0.0 else 0.0
    )
    low = np.asarray((230, 242, 255), dtype=np.float64)
    high = np.asarray((238, 180, 155), dtype=np.float64)
    rgb = np.rint(low * (1.0 - entropy_fraction) + high * entropy_fraction).astype(int)
    return "#" + "".join(f"{value:02X}" for value in rgb)


def _accessible_svg(svg: str, title: str, description: str) -> str:
    value = re.sub(r"<title>.*?</title>", "", svg, count=1, flags=re.DOTALL)
    replacement = (
        f'<svg role="img" aria-labelledby="svg-title svg-description"'
    )
    value = value.replace("<svg", replacement, 1)
    insertion = (
        f'<title id="svg-title">{escape(title)}</title>'
        f'<desc id="svg-description">{escape(description)}</desc>'
    )
    position = value.find(">", value.find("<svg"))
    if position < 0:
        raise ValueError("rendered SVG has no root element")
    return value[: position + 1] + insertion + value[position + 1 :]


def _markdown_report(
    inputs: TemporalStudyInputs,
    analysis: Mapping[str, object],
    execution: Mapping[str, float],
    allocator: Mapping[str, object],
) -> str:
    final_rows = _records(analysis["aggregate"], "aggregate")
    bootstrap = _records(analysis["bootstrap"], "bootstrap")
    distortion = _object(analysis["distortion"], "distortion")
    merge_rows = _records(distortion["merge"], "merge")
    lineages = _records(distortion["lineage"], "lineage")
    final_table = "\n".join(
        "| {order} | {method} | {nll:.5f} | {token_nll:.5f} | {accuracy:.2%} | {support} | {regret:.5f} |".format(
            order=str(row["order"]).replace("_", " "),
            method=str(row["method"]).replace("_", " "),
            nll=float(row["story_mean_nll"]),
            token_nll=float(row["token_mean_nll"]),
            accuracy=float(row["suffix_token_accuracy"]),
            support=(
                "—"
                if row["noun_support_accuracy"] is None
                else f"{float(row['noun_support_accuracy']):.2%}"
            ),
            regret=float(row["mean_oracle_regret"]),
        )
        for row in final_rows
    )
    bootstrap_table = "\n".join(
        f"| {row['metric']} | {float(row['estimate']):+.6f} | "
        f"[{float(row['lower_95']):+.6f}, {float(row['upper_95']):+.6f}] |"
        for row in bootstrap
    )
    peak_source = max(
        float(row["delta"]) for row in merge_rows if row["kind"] == "source"
    )
    peak_validation = max(
        float(row["delta"]) for row in merge_rows if row["kind"] == "validation"
    )
    maximum_residual = max(abs(float(row["telescoping_residual"])) for row in lineages)
    minimum_slack = min(float(row["positive_bound_slack"]) for row in lineages)
    elapsed = float(execution.get("end_to_end_seconds", 0.0))
    peak = int(allocator.get("peak_bytes_in_use", 0))
    return f"""# TinyWorlds Nouns-v2 Log-t Temporal Consolidation

This report publishes the fixed seed-zero experiment bound by contract
`{inputs.contract_sha256}`. It compares the same 192 immutable 512-story shards
under blocked and round-robin arrival orders. No merge was accepted or rejected
using validation quality: every third equal-level chunk synchronously merged the
two oldest chunks, exactly as preregistered.

![Suffix quality across arrivals](suffix-quality-over-time.svg)

## Final results

| Order | Method | Story NLL | Token NLL | Token accuracy | Noun support | Oracle regret |
|---|---|---:|---:|---:|---:|---:|
{final_table}

![Final-checkpoint quality](final-quality.svg)

The independent-noun bank is a practical 24-adapter endpoint, not the exact
192-adapter no-consolidation ablation. The IID controls are offline endpoints
and do not define historical causal curves.

### Paired ordering effects

Values are round-robin minus blocked for the final routed log-t bank. Intervals
are deterministic noun-stratified seed-zero 10,000-sample bootstrap summaries;
they are descriptive, not pass/fail tests.

| Metric | Paired difference | 95% interval |
|---|---:|---:|
{bootstrap_table}

<details>
<summary>Method and routing semantics</summary>

Each arrival trains a fresh base-relative rank-eight LoRA. A level stores at
most two chunks. On overflow, the two oldest equal-level chunks are discarded
from the live bank only after a fresh parent is trained from the frozen base on
their exact source union. The final deployment is the base plus nine adapters,
covering intervals 1–64, 65–128, 129–160, 161–176, 177–184, 185–188, 189–190,
191, and 192.

Routing exhaustively scores mean token NLL on the exact midpoint prefix, with
base first and stable first-minimum ties. It receives neither the noun identity
nor suffix tokens. The suffix oracle is evaluator-only. Since a temporal chunk
can contain several nouns, a selected chunk containing any data for the query
noun is called a noun-support hit rather than route accuracy.

</details>

<details>
<summary>Merge distortion and lineage proof</summary>

For each merge child, the audit measures parent-minus-child NLL on every
descendant shard and on its noun's fixed 16-story sentinel. The merge statistic
is the worse of the two token-weighted child distortions. The maximum observed
source and validation deltas were {peak_source:+.6f} and
{peak_validation:+.6f} nats/token. Per-arrival signed increments telescope to
direct level-zero-to-active-ancestor drift with maximum residual
{maximum_residual:.3g}; the smallest positive-part bound slack was
{minimum_slack:.3g}.

![Merge distortion](merge-distortion.svg)

</details>

<details>
<summary>Cost, timing, and storage</summary>

![Temporal cost growth](temporal-cost-growth.svg)

![Warm kernel timing](warm-kernel-timing.svg)

Candidate evaluations, model-forward-equivalent tokens, optimizer updates,
bytes, and seconds remain in distinct fields and plot panels. `timing.csv`
contains one cold compilation and five synchronized warm repetitions for every
observed prefix-width/candidate-capacity shape. End-to-end execution was
{elapsed / 3600:.2f} hours. Peak JAX allocator use was {peak / 1024**3:.2f} GiB
against the fixed 12 GiB gate.

</details>

<details>
<summary>Per-task, provenance, and machine-readable evidence</summary>

Per-task results are in `per-task.csv`; complete stage and arrival curves are in
`stage.csv`; forgetting and backward transfer are in `forgetting.csv`;
selection age/level evidence and independent-bank confusion are in
`selection.csv` and `confusion.csv`. Merge, lineage, timing, cost, and bootstrap
evidence have separate CSVs. `analysis.json` and `manifest.json` bind every
published byte. The authenticated selected-base parameter checksum is
`{inputs.selected_base.reference.parameter_checksum}`, the final canonical VAMP
tensor checksum is `{inputs.final_vamp_tensor_checksum}`, and the partition is
`{inputs.partition.partition_sha256}`.

</details>

## Lineage views

### Blocked, deployed bank

![Blocked compact lineage](lineage-blocked-compact.svg)

### Round robin, deployed bank

![Round-robin compact lineage](lineage-round_robin-compact.svg)

The corresponding complete 192-leaf/183-merge audits are
`lineage-blocked-full.svg` and `lineage-round_robin-full.svg`.
"""


def _html_report(
    inputs: TemporalStudyInputs,
    analysis: Mapping[str, object],
    execution: Mapping[str, float],
    allocator: Mapping[str, object],
    svgs: Mapping[str, str],
) -> str:
    final_rows = _records(analysis["aggregate"], "aggregate")
    bootstrap = _records(analysis["bootstrap"], "bootstrap")
    final_table = "".join(
        "<tr>"
        f"<td>{escape(str(row['order']).replace('_', ' '))}</td>"
        f"<td>{escape(str(row['method']).replace('_', ' '))}</td>"
        f"<td>{float(row['story_mean_nll']):.5f}</td>"
        f"<td>{float(row['token_mean_nll']):.5f}</td>"
        f"<td>{float(row['suffix_token_accuracy']):.2%}</td>"
        f"<td>{_format_optional_percentage(row['noun_support_accuracy'])}</td>"
        f"<td>{float(row['mean_oracle_regret']):.5f}</td></tr>"
        for row in final_rows
    )
    bootstrap_table = "".join(
        "<tr>"
        f"<td>{escape(str(row['metric']))}</td>"
        f"<td>{float(row['estimate']):+.6f}</td>"
        f"<td>[{float(row['lower_95']):+.6f}, {float(row['upper_95']):+.6f}]</td>"
        "</tr>"
        for row in bootstrap
    )
    embed = lambda name: svgs[name]
    provenance = _object(analysis["provenance"], "provenance")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TinyWorlds nouns-v2 temporal consolidation</title>
<style>
:root{{--ink:#17212b;--muted:#52606d;--line:#c8d3dc;--panel:#f5f8fa;--blue:#0069a8}}*{{box-sizing:border-box}}body{{margin:0;color:var(--ink);font:16px/1.55 system-ui,sans-serif}}main{{max-width:1280px;margin:auto;padding:28px}}h1{{line-height:1.15}}h2{{margin-top:2rem}}.lead{{font-size:1.08rem;max-width:90ch}}.plot{{margin:1.2rem 0;overflow:auto}}.plot svg{{width:100%;height:auto;min-width:680px}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}th,td{{padding:8px;border-bottom:1px solid var(--line);text-align:right}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}details{{margin:1rem 0;border:1px solid var(--line);border-radius:8px;background:var(--panel);padding:12px 16px}}summary{{cursor:pointer;font-weight:700;color:var(--blue)}}code{{overflow-wrap:anywhere}}.graphs{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}.graphs svg{{width:100%;height:auto}}@media(max-width:800px){{main{{padding:14px}}.graphs{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>TinyWorlds nouns-v2 log-t temporal consolidation</h1>
<p class="lead">This is the standalone report for immutable contract <code>{inputs.contract_sha256}</code>. The same 192 shards were evaluated in blocked and round-robin order. Every scheduled merge was accepted without consulting validation quality.</p>
<div class="plot">{embed('suffix-quality-over-time.svg')}</div>
<h2>Final results</h2><div style="overflow:auto"><table><thead><tr><th>Order</th><th>Method</th><th>Story NLL</th><th>Token NLL</th><th>Token accuracy</th><th>Noun support</th><th>Oracle regret</th></tr></thead><tbody>{final_table}</tbody></table></div>
<div class="plot">{embed('final-quality.svg')}</div>
<p>The independent noun bank is a practical 24-adapter endpoint, not a 192-adapter no-consolidation ablation. IID controls are offline endpoints rather than historical curves.</p>
<details open><summary>Paired ordering effects</summary><p>Round-robin minus blocked, with deterministic noun-stratified seed-zero 10,000-sample bootstrap intervals.</p><table><thead><tr><th>Metric</th><th>Difference</th><th>95% interval</th></tr></thead><tbody>{bootstrap_table}</tbody></table></details>
<details><summary>Method and interpretation</summary><p>Each arrival trains a fresh base-relative rank-eight LoRA. When a third chunk reaches a level, the two oldest are synchronously consolidated by retraining a fresh parent from the frozen base on their exact source union. Routing scores only the midpoint prefix, with base first and stable first-minimum ties. The suffix oracle never affects routing. Mixed chunks have noun support, not a unique route label.</p><div class="graphs">{embed('lineage-blocked-compact.svg')}{embed('lineage-round_robin-compact.svg')}</div></details>
<details><summary>Merge distortion and lineage</summary><p>Parent-minus-child loss is measured on every descendant source shard and its noun sentinel. Signed increments are retained per arrival; direct drift must telescope and stay below the sum of positive increments.</p><div class="plot">{embed('merge-distortion.svg')}</div></details>
<details><summary>Cost and timing</summary><p>Incomparable operation units remain in separate panels. Cold compilation and all five synchronized warm repetitions are preserved in <code>timing.csv</code>.</p><div class="plot">{embed('temporal-cost-growth.svg')}</div><div class="plot">{embed('warm-kernel-timing.svg')}</div></details>
<details><summary>Complete lineage audits</summary><div class="plot">{embed('lineage-blocked-full.svg')}</div><div class="plot">{embed('lineage-round_robin-full.svg')}</div></details>
<details><summary>Per-task and provenance</summary><p>Machine-readable aggregate, per-task, stage, forgetting, selection, confusion, merge, lineage, timing, cost, and bootstrap CSVs accompany this report. The selected base is <code>{escape(str(provenance['base_parameter_checksum']))}</code>; the canonical final VAMP tensors are <code>{escape(str(provenance['final_vamp_tensor_checksum']))}</code>. End-to-end time: {float(execution.get('end_to_end_seconds', 0.0))/3600:.2f} h. Allocator peak: {int(allocator.get('peak_bytes_in_use', 0))/1024**3:.2f} GiB.</p></details>
</main></body></html>"""


def _records(value: object, label: str) -> tuple[dict[str, object], ...]:
    if type(value) not in (list, tuple):
        raise TypeError(f"{label} must be a record sequence")
    return tuple(_object(item, label) for item in value)


def _format_optional_percentage(value: object) -> str:
    return "—" if value is None else f"{float(value):.2%}"


def _object(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise TypeError(f"{label} must be an object")
    return dict(value)


def _list(value: object, label: str) -> list[object]:
    if type(value) is not list:
        raise TypeError(f"{label} must be a list")
    return value


__all__ = [
    "analyze_temporal_study",
    "publish_temporal_consolidation_report",
]

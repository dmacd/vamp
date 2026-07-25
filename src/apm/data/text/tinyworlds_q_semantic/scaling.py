"""Manifest-derived schedules, bounded scoring, ledgers, and resource preflight."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from math import isfinite
import os
from pathlib import Path
import tempfile
from typing import TypeVar

from apm.data.text.tinyworlds_q_semantic.contracts import (
    QueryExperimentPreset,
    SemanticQueryResult,
    canonical_json_bytes,
)


InputValue = TypeVar("InputValue")
OutputValue = TypeVar("OutputValue")


@dataclass(frozen=True, slots=True)
class EvaluationCell:
    """One stage/world evaluation selected by a dynamic schedule."""

    stage: int
    concept_id: str
    acquisition: bool
    milestone: bool
    final: bool

    def __post_init__(self) -> None:
        if type(self.stage) is not int or self.stage <= 0:
            raise ValueError("evaluation stage must be positive")
        if type(self.concept_id) is not str or not self.concept_id:
            raise ValueError("evaluation concept must be nonempty")


@dataclass(frozen=True, slots=True)
class ResourceEstimate:
    """Frozen-count estimate for training, search, routing, storage, and memory."""

    world_count: int
    training_updates: int
    parent_probe_scores: int
    routing_candidate_scores: int
    result_rows: int
    estimated_result_bytes: int
    estimated_peak_bytes: int

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value < 0
            for value in (
                self.world_count,
                self.training_updates,
                self.parent_probe_scores,
                self.routing_candidate_scores,
                self.result_rows,
                self.estimated_result_bytes,
                self.estimated_peak_bytes,
            )
        ):
            raise ValueError("resource estimates must be nonnegative integers")

    def as_record(self) -> dict[str, int]:
        """Return a report-ready resource estimate."""
        return {
            "estimated_peak_bytes": self.estimated_peak_bytes,
            "estimated_result_bytes": self.estimated_result_bytes,
            "parent_probe_scores": self.parent_probe_scores,
            "result_rows": self.result_rows,
            "routing_candidate_scores": self.routing_candidate_scores,
            "training_updates": self.training_updates,
            "world_count": self.world_count,
        }


@dataclass(frozen=True, slots=True)
class PreflightMeasurement:
    """Measured resource evidence checked against one frozen preset."""

    seconds_per_training_update: float
    seconds_per_parent_probe: float
    seconds_per_routing_score: float
    allocator_peak_bytes: int
    projected_result_bytes: int

    def __post_init__(self) -> None:
        if any(
            not isfinite(value) or value <= 0.0
            for value in (
                self.seconds_per_training_update,
                self.seconds_per_parent_probe,
                self.seconds_per_routing_score,
            )
        ):
            raise ValueError("preflight timings must be finite and positive")
        if any(
            type(value) is not int or value < 0
            for value in (self.allocator_peak_bytes, self.projected_result_bytes)
        ):
            raise ValueError("preflight byte measurements must be nonnegative")


def evaluation_schedule(preset: QueryExperimentPreset) -> tuple[EvaluationCell, ...]:
    """Derive full or milestone evaluation cells from one ordered manifest."""
    world_index = {concept_id: index + 1 for index, concept_id in enumerate(preset.concept_ids)}
    milestone_stages = (
        set(range(1, preset.active_world_count + 1))
        if preset.evaluation_schedule == "full"
        else {*preset.evaluation_milestones, preset.active_world_count}
    )
    raw_cells = {
        (stage, concept_id)
        for stage in milestone_stages
        for concept_id in preset.concept_ids[:stage]
    } | {
        (stage, concept_id)
        for stage, concept_id in enumerate(preset.concept_ids, start=1)
    }
    return tuple(
        EvaluationCell(
            stage=stage,
            concept_id=concept_id,
            acquisition=world_index[concept_id] == stage,
            milestone=stage in milestone_stages,
            final=stage == preset.active_world_count,
        )
        for stage, concept_id in sorted(
            raw_cells,
            key=lambda item: (item[0], world_index[item[1]]),
        )
    )


def estimate_resources(
    preset: QueryExperimentPreset,
    *,
    queries_per_world: int = 60,
    method_count: int = 9,
    base_runtime_peak_bytes: int = 9 * 1024**3,
    bytes_per_adapter_edge: int = 8 * 1024**2,
    bytes_per_result_row: int = 1_024,
) -> ResourceEstimate:
    """Estimate every scaling dimension from the same one-to-100-world codepath."""
    positive_values = (
        queries_per_world,
        method_count,
        base_runtime_peak_bytes,
        bytes_per_adapter_edge,
        bytes_per_result_row,
    )
    if any(type(value) is not int or value <= 0 for value in positive_values):
        raise ValueError("resource-estimate dimensions must be positive integers")
    world_count = preset.active_world_count
    cells = evaluation_schedule(preset)
    training_updates = 3 * world_count * preset.adapter_updates
    parent_probe_scores = sum(
        stage * preset.parent_probe_count
        for stage in range(1, world_count + 1)
    )
    result_rows = len(cells) * queries_per_world * method_count
    routing_scores = sum(
        queries_per_world * method_count * (cell.stage + 1)
        for cell in cells
    )
    estimated_result_bytes = result_rows * bytes_per_result_row
    chunk_runtime = (
        preset.query_chunk_size
        * 4
        * preset.max_nodes
        * 4
        * 256
    )
    estimated_peak_bytes = (
        base_runtime_peak_bytes
        + preset.max_edges * bytes_per_adapter_edge
        + chunk_runtime
    )
    return ResourceEstimate(
        world_count=world_count,
        training_updates=training_updates,
        parent_probe_scores=parent_probe_scores,
        routing_candidate_scores=routing_scores,
        result_rows=result_rows,
        estimated_result_bytes=estimated_result_bytes,
        estimated_peak_bytes=estimated_peak_bytes,
    )


def require_preflight_capacity(
    preset: QueryExperimentPreset,
    estimate: ResourceEstimate,
    measurement: PreflightMeasurement,
) -> None:
    """Fail explicitly when measured or projected resources exceed frozen limits."""
    if estimate.world_count != preset.active_world_count:
        raise ValueError("preflight estimate does not match the active manifest")
    projected_peak_bytes = max(
        estimate.estimated_peak_bytes,
        measurement.allocator_peak_bytes,
    )
    if projected_peak_bytes > preset.allocator_peak_limit_bytes:
        raise MemoryError(
            "measured or projected allocator peak exceeds the frozen limit: "
            f"{projected_peak_bytes} > {preset.allocator_peak_limit_bytes}"
        )
    projected_result_bytes = max(
        estimate.estimated_result_bytes,
        measurement.projected_result_bytes,
    )
    if projected_result_bytes > preset.result_size_limit_bytes:
        raise OSError(
            "projected result ledger exceeds the frozen limit: "
            f"{projected_result_bytes} > {preset.result_size_limit_bytes}"
        )


def projected_runtime_seconds(
    estimate: ResourceEstimate,
    measurement: PreflightMeasurement,
) -> dict[str, float]:
    """Project the three measured runtime families without hiding their scale."""
    return {
        "training": estimate.training_updates * measurement.seconds_per_training_update,
        "parent_search": estimate.parent_probe_scores * measurement.seconds_per_parent_probe,
        "routing": estimate.routing_candidate_scores * measurement.seconds_per_routing_score,
    }


def iter_chunks(
    values: Iterable[InputValue],
    chunk_size: int,
) -> Iterator[tuple[InputValue, ...]]:
    """Yield bounded immutable chunks without materializing the input stream."""
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    pending: list[InputValue] = []
    for value in values:
        pending.append(value)
        if len(pending) == chunk_size:
            yield tuple(pending)
            pending = []
    if pending:
        yield tuple(pending)


def score_in_chunks(
    values: Iterable[InputValue],
    chunk_size: int,
    scorer: Callable[[tuple[InputValue, ...]], Iterable[OutputValue]],
) -> Iterator[OutputValue]:
    """Score bounded query/node chunks and stream each result immediately."""
    for chunk in iter_chunks(values, chunk_size):
        yield from scorer(chunk)


def write_atomic_jsonl(
    path: str | Path,
    records: Iterable[Mapping[str, object] | SemanticQueryResult],
) -> Path:
    """Stream canonical rows to a temporary ledger, then atomically publish it."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_directory = Path(
        tempfile.mkdtemp(prefix=f".{target.stem}-", dir=target.parent)
    )
    print(f"TinyWorlds-Q ledger temporary artifacts: {temporary_directory}", flush=True)
    temporary = temporary_directory / target.name
    try:
        with temporary.open("wb") as stream:
            for record in records:
                payload = (
                    record.as_record()
                    if isinstance(record, SemanticQueryResult)
                    else dict(record)
                )
                stream.write(canonical_json_bytes(payload))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        temporary_directory.rmdir()
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        if temporary_directory.exists():
            temporary_directory.rmdir()
        raise
    return target


def render_schedule_report(preset: QueryExperimentPreset) -> str:
    """Render a world-name-driven schedule without A-through-E assumptions."""
    cells = evaluation_schedule(preset)
    lines = [
        "# TinyWorlds-Q dynamic evaluation schedule",
        "",
        f"Worlds: {preset.active_world_count}",
        f"Graph capacity: {preset.max_nodes} nodes / {preset.max_edges} edges",
        f"Schedule: `{preset.evaluation_schedule}`",
        "",
        "| Stage | World | Acquisition | Milestone | Final |",
        "|---:|---|:---:|:---:|:---:|",
    ]
    lines.extend(
        f"| {cell.stage} | {cell.concept_id} | {cell.acquisition} | "
        f"{cell.milestone} | {cell.final} |"
        for cell in cells
    )
    return "\n".join(lines) + "\n"


__all__ = [
    "EvaluationCell",
    "PreflightMeasurement",
    "ResourceEstimate",
    "estimate_resources",
    "evaluation_schedule",
    "iter_chunks",
    "projected_runtime_seconds",
    "render_schedule_report",
    "require_preflight_capacity",
    "score_in_chunks",
    "write_atomic_jsonl",
]

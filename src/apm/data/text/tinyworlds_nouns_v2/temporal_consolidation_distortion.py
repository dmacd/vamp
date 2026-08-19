"""Merge distortion and per-arrival telescoping audits for temporal LoRA."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias

from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_contracts import (
    CONTEXT_LENGTH,
    MERGE_ROW_FORMAT,
    PHYSICAL_BATCH_SIZE,
    SENTINEL_STORIES_PER_TASK,
    TemporalChunk,
    TemporalHierarchyState,
)
from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_evaluation import (
    AdapterCandidate,
    CandidateScore,
    MidpointCase,
    build_adapter_bank,
    score_midpoint_cases_by_candidate,
    score_token_batches_by_candidate,
)
from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_io import (
    ChainedJsonlLedger,
)
from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_training import (
    AdapterArtifact,
    StoryEpochBatches,
)

if TYPE_CHECKING:
    from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation import (
        OrderingArtifacts,
        TemporalStudyInputs,
    )


DistortionKind: TypeAlias = Literal["source", "validation"]
DistortionProgress = Callable[[int, int, Mapping[str, float]], None]


@dataclass(frozen=True, slots=True)
class LineageAudit:
    """One arrival's signed telescoping identity for a fixed evaluation kind."""

    order: str
    source_shard_id: str
    task_id: str
    kind: DistortionKind
    merge_count: int
    signed_increment_sum: float
    positive_increment_sum: float
    direct_drift: float
    telescoping_residual: float
    positive_bound_slack: float

    def __post_init__(self) -> None:
        numeric = (
            self.signed_increment_sum,
            self.positive_increment_sum,
            self.direct_drift,
            self.telescoping_residual,
            self.positive_bound_slack,
        )
        if (
            not self.order
            or not self.source_shard_id
            or not self.task_id
            or self.kind not in ("source", "validation")
            or self.merge_count < 0
            or any(not math.isfinite(value) for value in numeric)
            or self.positive_increment_sum < 0.0
        ):
            raise ValueError("lineage audit values are invalid")

    def as_record(self) -> dict[str, object]:
        """Return a CSV/JSON-ready lineage record."""
        return {
            "direct_drift": self.direct_drift,
            "kind": self.kind,
            "merge_count": self.merge_count,
            "order": self.order,
            "positive_bound_slack": self.positive_bound_slack,
            "positive_increment_sum": self.positive_increment_sum,
            "signed_increment_sum": self.signed_increment_sum,
            "source_shard_id": self.source_shard_id,
            "task_id": self.task_id,
            "telescoping_residual": self.telescoping_residual,
        }


def expected_distortion_keys(
    ordering: OrderingArtifacts,
) -> tuple[tuple[str, str, str, str], ...]:
    """Return the canonical merge/child/leaf/kind ledger ordering."""
    return tuple(
        (
            merge.parent.chunk_id,
            child.chunk_id,
            shard_id,
            kind,
        )
        for merge in ordering.merges
        for child in (merge.left, merge.right)
        for shard_id in child.shard_ids
        for kind in ("source", "validation")
    )


def run_or_resume_merge_distortion(
    inputs: TemporalStudyInputs,
    ordering: OrderingArtifacts,
    cases_by_task: Mapping[str, Sequence[MidpointCase]],
    *,
    progress: DistortionProgress | None = None,
) -> Path:
    """Measure every child-to-parent source and sentinel lineage increment."""
    ledger = ChainedJsonlLedger(
        inputs.work_directory / f"merge-distortion-{ordering.order}.jsonl",
        MERGE_ROW_FORMAT,
    )
    expected = expected_distortion_keys(ordering)
    observed = validate_distortion_rows(
        ledger.rows,
        inputs.contract_sha256,
        ordering.order,
    )
    if observed != expected[: len(observed)]:
        raise ValueError("merge distortion ledger is not a canonical prefix")
    completed = set(observed)
    positive_total = math.fsum(
        max(0.0, float(row["signed_increment"])) for row in ledger.rows
    )
    shard_by_id = {shard.shard_id: shard for shard in inputs.shards}
    entry_by_id = inputs.train_entry_lookup
    sentinel_ids = {
        task_id: frozenset(story_ids) for task_id, story_ids in inputs.sentinel
    }
    sentinel_cases = {
        task_id: tuple(
            case
            for case in cases_by_task[task_id]
            if case.entry.story_id in sentinel_ids[task_id]
        )
        for task_id in cases_by_task
    }
    if any(len(cases) != SENTINEL_STORIES_PER_TASK for cases in sentinel_cases.values()):
        raise ValueError("merge distortion sentinel coverage changed")
    chunk_artifacts = ordering.chunks_by_id
    finished = len(observed)
    for merge in ordering.merges:
        parent_artifact = chunk_artifacts[merge.parent.chunk_id]
        for child in (merge.left, merge.right):
            child_artifact = chunk_artifacts[child.chunk_id]
            bank = _child_parent_bank(
                child,
                child_artifact,
                merge.parent,
                parent_artifact,
                inputs,
            )
            validation_cache: dict[str, tuple[CandidateScore, CandidateScore]] = {}
            for shard_id in child.shard_ids:
                shard = shard_by_id[shard_id]
                pending_kinds = tuple(
                    kind
                    for kind in ("source", "validation")
                    if (merge.parent.chunk_id, child.chunk_id, shard_id, kind)
                    not in completed
                )
                if not pending_kinds:
                    continue
                source_scores = (
                    _source_scores(inputs, shard.story_ids, entry_by_id, bank)
                    if "source" in pending_kinds
                    else None
                )
                if "validation" in pending_kinds:
                    if shard.task_id not in validation_cache:
                        validation_cache[shard.task_id] = _validation_scores(
                            inputs,
                            sentinel_cases[shard.task_id],
                            bank,
                        )
                    validation_scores = validation_cache[shard.task_id]
                else:
                    validation_scores = None
                scores_by_kind = {
                    "source": source_scores,
                    "validation": validation_scores,
                }
                values = tuple(
                    _distortion_values(
                        inputs.contract_sha256,
                        ordering.order,
                        merge.parent,
                        parent_artifact,
                        child,
                        child_artifact,
                        shard_id,
                        shard.task_id,
                        kind,
                        scores_by_kind[kind],
                    )
                    for kind in pending_kinds
                )
                ledger.append_many(values)
                positive_total += math.fsum(
                    max(0.0, float(row["signed_increment"])) for row in values
                )
                for value in values:
                    completed.add(
                        (
                            str(value["parent_chunk_id"]),
                            str(value["child_chunk_id"]),
                            str(value["source_shard_id"]),
                            str(value["kind"]),
                        )
                    )
                    finished += 1
                if progress is not None:
                    progress(
                        finished,
                        len(expected),
                        {
                            "latest_signed_increment": float(
                                values[-1]["signed_increment"]
                            ),
                            "positive_increment_mean": positive_total / finished,
                        },
                    )
    final_rows = ledger.rows
    final_keys = validate_distortion_rows(
        final_rows,
        inputs.contract_sha256,
        ordering.order,
    )
    if final_keys != expected:
        raise RuntimeError(
            f"merge distortion has {len(final_keys):,} of {len(expected):,} rows"
        )
    require_lineage_identities(final_rows, ordering.final_state)
    return ledger.path


def summarize_lineages(
    rows: Sequence[Mapping[str, object]],
    final_state: TemporalHierarchyState,
) -> tuple[LineageAudit, ...]:
    """Summarize signed telescoping and positive-part bounds per arrival."""
    active_by_shard = {
        shard_id: chunk
        for chunk in final_state.active_chunks
        for shard_id in chunk.shard_ids
    }
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["source_shard_id"]), str(row["kind"]))].append(row)
    all_shards = tuple(
        shard_id for chunk in final_state.active_chunks for shard_id in chunk.shard_ids
    )
    audits: list[LineageAudit] = []
    for shard_id in all_shards:
        for kind in ("source", "validation"):
            lineage = grouped.get((shard_id, kind), [])
            if lineage:
                signed = math.fsum(float(row["signed_increment"]) for row in lineage)
                positive = math.fsum(
                    max(0.0, float(row["signed_increment"])) for row in lineage
                )
                direct = float(lineage[-1]["parent_mean_nll"]) - float(
                    lineage[0]["child_mean_nll"]
                )
                task_id = str(lineage[0]["task_id"])
            else:
                signed = positive = direct = 0.0
                task_counts = dict(active_by_shard[shard_id].task_counts)
                if len(task_counts) != 1:
                    raise ValueError("an unmerged level-zero chunk has mixed task support")
                task_id = next(iter(task_counts))
            audits.append(
                LineageAudit(
                    order=final_state.order,
                    source_shard_id=shard_id,
                    task_id=task_id,
                    kind=kind,  # type: ignore[arg-type]
                    merge_count=len(lineage),
                    signed_increment_sum=signed,
                    positive_increment_sum=positive,
                    direct_drift=direct,
                    telescoping_residual=direct - signed,
                    positive_bound_slack=positive - direct,
                )
            )
    return tuple(audits)


def require_lineage_identities(
    rows: Sequence[Mapping[str, object]],
    final_state: TemporalHierarchyState,
    *,
    tolerance: float = 1e-5,
) -> tuple[LineageAudit, ...]:
    """Require telescoping parity and the positive-increment drift bound."""
    audits = summarize_lineages(rows, final_state)
    if any(
        abs(audit.telescoping_residual) > tolerance
        or audit.positive_bound_slack < -tolerance
        for audit in audits
    ):
        raise ValueError("merge lineage does not telescope or violates its positive bound")
    return audits


def validate_distortion_rows(
    rows: Sequence[Mapping[str, object]],
    contract_sha256: str,
    order: str,
) -> tuple[tuple[str, str, str, str], ...]:
    """Strictly validate distortion row bindings and arithmetic."""
    keys: list[tuple[str, str, str, str]] = []
    for row in rows:
        child_mean = row.get("child_mean_nll")
        parent_mean = row.get("parent_mean_nll")
        signed = row.get("signed_increment")
        positive = row.get("positive_increment")
        token_count = row.get("token_count")
        kind = row.get("kind")
        key = (
            str(row.get("parent_chunk_id")),
            str(row.get("child_chunk_id")),
            str(row.get("source_shard_id")),
            str(kind),
        )
        if (
            row.get("contract_sha256") != contract_sha256
            or row.get("order") != order
            or kind not in ("source", "validation")
            or type(token_count) is not int
            or token_count <= 0
            or not _finite_nonnegative(child_mean)
            or not _finite_nonnegative(parent_mean)
            or not _finite_number(signed)
            or not _finite_nonnegative(positive)
            or not math.isclose(
                float(signed),
                float(parent_mean) - float(child_mean),
                abs_tol=1e-9,
            )
            or not math.isclose(float(positive), max(0.0, float(signed)), abs_tol=1e-12)
            or key in keys
        ):
            raise ValueError("merge distortion ledger row changed")
        keys.append(key)
    return tuple(keys)


def _child_parent_bank(
    child: TemporalChunk,
    child_artifact: AdapterArtifact,
    parent: TemporalChunk,
    parent_artifact: AdapterArtifact,
    inputs: TemporalStudyInputs,
):
    return build_adapter_bank(
        (
            AdapterCandidate(
                "child",
                child_artifact.adapter_sha256,
                child_artifact.adapter,
                child.task_counts,
                child.level,
                child.start_arrival,
                child.end_arrival,
            ),
            AdapterCandidate(
                "parent",
                parent_artifact.adapter_sha256,
                parent_artifact.adapter,
                parent.task_counts,
                parent.level,
                parent.start_arrival,
                parent.end_arrival,
            ),
        ),
        inputs.loaded_base.config,
    )


def _source_scores(
    inputs: TemporalStudyInputs,
    story_ids: Sequence[str],
    entry_by_id,
    bank,
) -> tuple[CandidateScore, CandidateScore]:
    batches = StoryEpochBatches(
        inputs.partition,
        tuple(entry_by_id[story_id] for story_id in story_ids),
        context_length=CONTEXT_LENGTH,
        batch_size=PHYSICAL_BATCH_SIZE,
        namespace=f"merge-distortion-source-{story_ids[0]}",
        epochs=1,
    )
    child, parent = score_token_batches_by_candidate(
        batches,
        base_params=inputs.loaded_base.params,
        model_config=inputs.loaded_base.config,
        bank=bank,
        candidate_indices=(1, 2),
    )
    return child, parent


def _validation_scores(
    inputs: TemporalStudyInputs,
    cases: Sequence[MidpointCase],
    bank,
) -> tuple[CandidateScore, CandidateScore]:
    child, parent = score_midpoint_cases_by_candidate(
        cases,
        base_params=inputs.loaded_base.params,
        model_config=inputs.loaded_base.config,
        bank=bank,
        candidate_indices=(1, 2),
    )
    return child, parent


def _distortion_values(
    contract_sha256: str,
    order: str,
    parent: TemporalChunk,
    parent_artifact: AdapterArtifact,
    child: TemporalChunk,
    child_artifact: AdapterArtifact,
    source_shard_id: str,
    task_id: str,
    kind: str,
    scores: tuple[CandidateScore, CandidateScore] | None,
) -> dict[str, object]:
    if scores is None:
        raise RuntimeError("pending merge distortion row has no scores")
    child_score, parent_score = scores
    if child_score.token_count != parent_score.token_count:
        raise RuntimeError("child and parent distortion token coverage differs")
    signed = parent_score.mean_nll - child_score.mean_nll
    return {
        "child_adapter_sha256": child_artifact.adapter_sha256,
        "child_chunk_id": child.chunk_id,
        "child_end_arrival": child.end_arrival,
        "child_level": child.level,
        "child_mean_nll": child_score.mean_nll,
        "child_start_arrival": child.start_arrival,
        "contract_sha256": contract_sha256,
        "kind": kind,
        "order": order,
        "parent_adapter_sha256": parent_artifact.adapter_sha256,
        "parent_chunk_id": parent.chunk_id,
        "parent_end_arrival": parent.end_arrival,
        "parent_level": parent.level,
        "parent_mean_nll": parent_score.mean_nll,
        "parent_start_arrival": parent.start_arrival,
        "positive_increment": max(0.0, signed),
        "signed_increment": signed,
        "source_shard_id": source_shard_id,
        "task_id": task_id,
        "token_count": child_score.token_count,
    }


def _finite_nonnegative(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value)) and value >= 0


def _finite_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


__all__ = [
    "LineageAudit",
    "expected_distortion_keys",
    "require_lineage_identities",
    "run_or_resume_merge_distortion",
    "summarize_lineages",
    "validate_distortion_rows",
]

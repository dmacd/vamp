"""Authenticated full-story routing diagnostic for the final temporal banks."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np

from apm.data.text.tinyworlds_nouns_v1.experiment import (
    IndexedStoryStore,
    StoryIndexEntry,
)
from apm.data.text.tinyworlds_nouns_v2.contracts import (
    TASK_IDS,
    canonical_json_bytes,
    record_sha256,
    require_sha256,
)
from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation import (
    TemporalStudyInputs,
    assert_canonical_artifacts_unchanged,
    authenticate_temporal_study_inputs,
)
from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_contracts import (
    ALLOCATOR_LIMIT_BYTES,
    ARRIVAL_COUNT,
    BOOTSTRAP_REPETITIONS,
    CONTEXT_LENGTH,
    EVALUATION_BATCH_SIZE,
    EVALUATION_ROW_FORMAT,
    LORA_ALPHA,
    LORA_RANK,
    SEED,
    SHARDS_PER_TASK,
    TemporalChunk,
    simulate_hierarchy,
)
from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_evaluation import (
    AdapterBank,
    AdapterCandidate,
    build_adapter_bank,
    build_story_windows,
    score_token_windows_by_candidate,
    validate_evaluation_rows,
)
from apm.continual.artifacts import (
    ChainedJsonlLedger,
    file_sha256,
    load_canonical_json,
    publish_immutable_json,
)
from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_training import (
    TrainingJob,
    load_adapter_artifact,
)
from apm.lm.lora import LoraConfig
from apm.lm.text_data import TokenBatch


STUDY_ID = "tinyworlds-nouns-v2-temporal-full-story-routing"
CONTRACT_FORMAT = f"{STUDY_ID}-contract-v1"
DIRECT_ROW_FORMAT = f"{STUDY_ID}-direct-score-row-v1"
CASE_ROW_FORMAT = f"{STUDY_ID}-case-row-v1"
REPORT_FORMAT = f"{STUDY_ID}-report-v1"
PARENT_CONTRACT_SHA256 = (
    "3f4ef4a10fd471b418a32a8f7b45431602c1f6abc080c19a7822ea2c2dd839b4"
)
PARENT_MANIFEST_SHA256 = (
    "15f3ee2a5a2c5054b158ba62d7a0d1b9fcaa22e40634a73c9cbffceca5888bcb"
)
AUDIT_MARGIN_THRESHOLD = 2e-4
AUDIT_ABSOLUTE_TOLERANCE = 1e-4
EXPECTED_SOURCE_ROWS = 13_320
EXPECTED_AUDIT_STORIES = 190
EXPECTED_DIRECT_ROWS = 570
DIRECT_STORY_BATCH_SIZE = 32

ConditionId: TypeAlias = Literal[
    "blocked_log_t",
    "round_robin_log_t",
    "independent_noun",
]
AccuracyKind: TypeAlias = Literal["noun_support", "exact_noun"]
ProgressCallback = Callable[[int, int, Mapping[str, float]], None]


@dataclass(frozen=True, slots=True)
class SourceSpecification:
    """One immutable parent-ledger source and its scientific label."""

    condition: ConditionId
    label: str
    relative_path: str
    sha256: str
    accuracy_kind: AccuracyKind

    def __post_init__(self) -> None:
        require_sha256(self.sha256, "full-story source ledger")
        if not self.label or Path(self.relative_path).is_absolute():
            raise ValueError("full-story source specification is invalid")


SOURCE_SPECIFICATIONS = (
    SourceSpecification(
        "blocked_log_t",
        "Blocked log-t",
        "evaluation-blocked/macro-stage-192-log_t.jsonl",
        "446ac7b6124315cb382cbeaa0d97d3bfd43b22f2df5e3899f2a29a7c6d1958ed",
        "noun_support",
    ),
    SourceSpecification(
        "round_robin_log_t",
        "Round-robin log-t",
        "evaluation-round_robin/macro-stage-192-log_t.jsonl",
        "a3c37fd4749ae94561047e2d7662d6afba970a62c37f2806b733ff3efa106e00",
        "noun_support",
    ),
    SourceSpecification(
        "independent_noun",
        "Independent noun bank",
        "evaluation-final-controls/final-stage-192-independent_noun_exhaustive.jsonl",
        "f0920125c9904a232b7e53ea0f2e325362219c0d3146c9dc3d3443e000f1ca03",
        "exact_noun",
    ),
)


@dataclass(frozen=True, slots=True)
class FullStoryRoutingInputs:
    """Authenticated parent study, final banks, source rows, and v1 contract."""

    parent: TemporalStudyInputs
    sources: tuple[tuple[SourceSpecification, tuple[dict[str, object], ...]], ...]
    banks: tuple[tuple[ConditionId, AdapterBank], ...]
    contract: dict[str, object]
    result_directory: Path
    work_directory: Path
    audit_story_ids: tuple[str, ...]
    parent_snapshot: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "result_directory", Path(self.result_directory))
        object.__setattr__(self, "work_directory", Path(self.work_directory))
        if len(self.audit_story_ids) != EXPECTED_AUDIT_STORIES:
            raise ValueError("full-story audit population changed")

    @property
    def contract_sha256(self) -> str:
        """Return the independent addendum contract identity."""
        return str(self.contract["contract_sha256"])

    @property
    def source_rows(self) -> dict[ConditionId, tuple[dict[str, object], ...]]:
        """Return validated parent rows by diagnostic condition."""
        return {specification.condition: rows for specification, rows in self.sources}

    @property
    def bank_by_condition(self) -> dict[ConditionId, AdapterBank]:
        """Return strict-loaded final banks by diagnostic condition."""
        return dict(self.banks)


def reconstructed_whole_story_scores(row: Mapping[str, object]) -> tuple[float, ...]:
    """Combine stored midpoint-prefix and suffix means into whole-story means."""
    prefix_scores = row.get("prefix_scores")
    suffix_scores = row.get("suffix_mean_nll_by_candidate")
    prefix_tokens = row.get("prefix_token_count")
    suffix_tokens = row.get("suffix_token_count")
    if (
        type(prefix_scores) is not list
        or type(suffix_scores) is not list
        or len(prefix_scores) != len(suffix_scores)
        or type(prefix_tokens) is not int
        or type(suffix_tokens) is not int
        or prefix_tokens <= 0
        or suffix_tokens <= 0
    ):
        raise ValueError("parent row cannot reconstruct a whole-story score")
    denominator = prefix_tokens + suffix_tokens
    return tuple(
        (float(prefix) * prefix_tokens + float(suffix) * suffix_tokens)
        / denominator
        for prefix, suffix in zip(prefix_scores, suffix_scores, strict=True)
    )


def stable_minimum(scores: Sequence[float]) -> tuple[int, float]:
    """Return stable first argmin and the top-two score margin."""
    values = tuple(float(value) for value in scores)
    if len(values) < 2 or any(not math.isfinite(value) for value in values):
        raise ValueError("stable routing requires at least two finite scores")
    selected = min(range(len(values)), key=values.__getitem__)
    ordered = sorted(values)
    return selected, ordered[1] - ordered[0]


def select_audit_story_ids(
    rows_by_condition: Mapping[ConditionId, Sequence[Mapping[str, object]]],
) -> tuple[str, ...]:
    """Select every long, near-tie, and per-noun minimum-margin story."""
    if tuple(rows_by_condition) != tuple(
        specification.condition for specification in SOURCE_SPECIFICATIONS
    ):
        raise ValueError("audit selection requires canonical condition order")
    reference = tuple(rows_by_condition[SOURCE_SPECIFICATIONS[0].condition])
    identities = tuple((row["task_id"], row["story_id"]) for row in reference)
    if any(
        tuple((row["task_id"], row["story_id"]) for row in rows_by_condition[condition])
        != identities
        for condition in rows_by_condition
    ):
        raise ValueError("source ledgers do not share exact story order")
    minimum_margin = {
        str(story_id): min(
            stable_minimum(reconstructed_whole_story_scores(rows[index]))[1]
            for rows in rows_by_condition.values()
        )
        for index, (_, story_id) in enumerate(identities)
    }
    selected = {
        str(row["story_id"])
        for row in reference
        if int(row["prefix_token_count"]) > CONTEXT_LENGTH
        or minimum_margin[str(row["story_id"])] <= AUDIT_MARGIN_THRESHOLD
    }
    for task_id in TASK_IDS:
        short = tuple(
            str(row["story_id"])
            for row in reference
            if row["task_id"] == task_id
            and int(row["prefix_token_count"]) <= CONTEXT_LENGTH
        )
        selected.add(min(short, key=lambda story_id: (minimum_margin[story_id], story_id)))
    return tuple(
        str(story_id) for _, story_id in identities if str(story_id) in selected
    )


def authenticate_full_story_routing_inputs(
    repository_root: str | Path,
) -> FullStoryRoutingInputs:
    """Authenticate the parent publication, ledgers, banks, and addendum contract."""
    parent = authenticate_temporal_study_inputs(repository_root)
    if parent.contract_sha256 != PARENT_CONTRACT_SHA256:
        raise ValueError("full-story diagnostic parent contract changed")
    parent_result = parent.result_directory
    parent_manifest_path = parent_result / "manifest.json"
    parent_analysis_path = parent_result / "analysis.json"
    manifest = load_canonical_json(parent_manifest_path)
    manifest_core = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    if (
        manifest.get("manifest_sha256") != PARENT_MANIFEST_SHA256
        or manifest.get("manifest_sha256") != record_sha256(manifest_core)
        or manifest.get("contract_sha256") != parent.contract_sha256
        or type(manifest.get("artifacts")) is not dict
    ):
        raise ValueError("full-story diagnostic parent manifest changed")
    parent_snapshot = tuple(
        (relative, file_sha256(parent_result / relative))
        for relative in sorted((*dict(manifest["artifacts"]), "manifest.json"))
    )
    expected_snapshot = tuple(
        sorted(
            (
                *((str(relative), str(digest)) for relative, digest in dict(manifest["artifacts"]).items()),
                ("manifest.json", file_sha256(parent_manifest_path)),
            )
        )
    )
    if parent_snapshot != expected_snapshot:
        raise ValueError("full-story diagnostic parent artifact hashes changed")
    analysis = load_canonical_json(parent_analysis_path)
    analysis_core = {
        key: value for key, value in analysis.items() if key != "analysis_sha256"
    }
    if analysis.get("analysis_sha256") != record_sha256(analysis_core):
        raise ValueError("full-story diagnostic parent analysis changed")
    provenance = {
        str(row["path"]): row
        for row in dict(analysis["analysis"])["ledger_provenance"]
    }
    sources = tuple(
        (
            specification,
            _load_source_rows(parent, specification, provenance),
        )
        for specification in SOURCE_SPECIFICATIONS
    )
    rows_by_condition = {
        specification.condition: rows for specification, rows in sources
    }
    if sum(len(rows) for rows in rows_by_condition.values()) != EXPECTED_SOURCE_ROWS:
        raise ValueError("full-story source coverage changed")
    banks = _load_final_banks(parent)
    for condition, bank in banks:
        candidate_orders = {
            tuple(str(value) for value in row["candidate_ids"])
            for row in rows_by_condition[condition]
        }
        if candidate_orders != {bank.candidate_ids}:
            raise ValueError(f"final {condition} bank no longer matches its ledger")
    audit_story_ids = select_audit_story_ids(rows_by_condition)
    if len(audit_story_ids) != EXPECTED_AUDIT_STORIES:
        raise ValueError(
            f"full-story audit selection changed: {len(audit_story_ids)} stories"
        )
    adapter_bindings = {
        condition: [
            {
                "adapter_sha256": candidate.adapter_sha256,
                "candidate_id": candidate.candidate_id,
                "manifest_sha256": file_sha256(candidate_path / "manifest.json"),
            }
            for candidate, candidate_path in _candidate_artifact_paths(parent, condition)
        ]
        for condition, _ in banks
    }
    core = {
        "audit": {
            "absolute_tolerance": AUDIT_ABSOLUTE_TOLERANCE,
            "expected_direct_rows": EXPECTED_DIRECT_ROWS,
            "expected_story_count": EXPECTED_AUDIT_STORIES,
            "margin_threshold": AUDIT_MARGIN_THRESHOLD,
            "selection": "all_prefix_gt_256_or_margin_le_2e-4_plus_per_noun_short_minimum",
            "story_ids_sha256": record_sha256(list(audit_story_ids)),
        },
        "bindings": {
            "adapters": adapter_bindings,
            "base_manifest_sha256": parent.selected_base.reference.manifest_sha256,
            "base_parameter_checksum": parent.selected_base.reference.parameter_checksum,
            "parent_analysis_sha256": file_sha256(parent_analysis_path),
            "parent_contract_sha256": parent.contract_sha256,
            "parent_manifest_sha256": PARENT_MANIFEST_SHA256,
            "partition_sha256": parent.partition.partition_sha256,
            "sources": [
                {
                    "condition": specification.condition,
                    "path": specification.relative_path,
                    "row_count": len(rows),
                    "sha256": specification.sha256,
                }
                for specification, rows in sources
            ],
        },
        "bootstrap": {"repetitions": BOOTSTRAP_REPETITIONS, "seed": SEED},
        "format": CONTRACT_FORMAT,
        "routing": {
            "candidate_order": "parent_ledger_order",
            "score": "mean_nll_over_all_story_transitions",
            "suffix_reuse": "parent_candidate_suffix_means",
            "tie_break": "stable_first_minimum",
            "warning": "full_story_selection_reads_the_reported_suffix",
            "window_context_length": CONTEXT_LENGTH,
            "window_reset": "story_offsets_0_256_512",
        },
        "schema_version": 1,
        "study_id": STUDY_ID,
    }
    contract = {**core, "contract_sha256": record_sha256(core)}
    result_directory = parent_result / "full-story-routing-v1"
    work_directory = parent.work_directory / "full-story-routing-v1"
    result_directory.mkdir(parents=True, exist_ok=True)
    work_directory.mkdir(parents=True, exist_ok=True)
    publish_immutable_json(result_directory / "contract.json", contract)
    return FullStoryRoutingInputs(
        parent,
        sources,
        banks,
        contract,
        result_directory,
        work_directory,
        audit_story_ids,
        parent_snapshot,
    )


def run_or_resume_direct_audit(
    inputs: FullStoryRoutingInputs,
    *,
    progress: ProgressCallback | None = None,
) -> Path:
    """Directly score the deterministic audit set and resume its chained ledger."""
    ledger = ChainedJsonlLedger(inputs.work_directory / "direct-scores.jsonl", DIRECT_ROW_FORMAT)
    expected_keys = tuple(
        (specification.condition, story_id)
        for specification in SOURCE_SPECIFICATIONS
        for story_id in inputs.audit_story_ids
    )
    _validate_direct_rows(ledger.rows, inputs.contract_sha256, expected_keys)
    if len(ledger.rows) == len(expected_keys):
        return ledger.path
    source_rows = inputs.source_rows
    bank_by_condition = inputs.bank_by_condition
    entry_by_story = {
        entry.story_id: entry
        for _, entries in inputs.parent.validation_entries
        for entry in entries
    }
    task_by_story = {
        entry.story_id: task_id
        for task_id, entries in inputs.parent.validation_entries
        for entry in entries
    }
    store = IndexedStoryStore(inputs.parent.partition)
    start = len(ledger.rows)
    while start < len(expected_keys):
        condition = expected_keys[start][0]
        condition_stop = next(
            (
                index
                for index in range(start, len(expected_keys))
                if expected_keys[index][0] != condition
            ),
            len(expected_keys),
        )
        stop = min(start + DIRECT_STORY_BATCH_SIZE, condition_stop)
        story_ids = tuple(story_id for _, story_id in expected_keys[start:stop])
        batches = tuple(
            build_story_windows(
                store.tokens(entry_by_story[story_id]),
                CONTEXT_LENGTH,
                inputs.parent.partition.pad_token_id,
                first_target_index=1,
            )
            for story_id in story_ids
        )
        stacked, slices = _stack_story_batches(batches)
        bank = bank_by_condition[condition]
        totals, _ = score_token_windows_by_candidate(
            stacked,
            base_params=inputs.parent.loaded_base.params,
            model_config=inputs.parent.loaded_base.config,
            bank=bank,
            candidate_indices=tuple(range(len(bank.candidate_ids))),
            evaluation_batch_size=EVALUATION_BATCH_SIZE,
        )
        values = []
        for story_id, batch, (window_start, window_stop) in zip(
            story_ids, batches, slices, strict=True
        ):
            token_count = int(np.sum(batch.loss_mask))
            story_totals = tuple(
                float(np.sum(candidate[window_start:window_stop], dtype=np.float64))
                for candidate in totals
            )
            means = tuple(total / token_count for total in story_totals)
            selected, margin = stable_minimum(means)
            values.append(
                {
                    "candidate_ids": list(bank.candidate_ids),
                    "condition": condition,
                    "contract_sha256": inputs.contract_sha256,
                    "mean_nll_by_candidate": list(means),
                    "selected_candidate_id": bank.candidate_ids[selected],
                    "selected_index": selected,
                    "story_id": story_id,
                    "task_id": task_by_story[story_id],
                    "token_count": token_count,
                    "top_two_margin": margin,
                    "total_nll_by_candidate": list(story_totals),
                }
            )
        ledger.append_many(values)
        start = stop
        if progress is not None:
            progress(
                start,
                len(expected_keys),
                {"condition_fraction": (stop % EXPECTED_AUDIT_STORIES) / EXPECTED_AUDIT_STORIES},
            )
    _validate_direct_rows(ledger.rows, inputs.contract_sha256, expected_keys)
    return ledger.path


def verify_direct_audit(
    inputs: FullStoryRoutingInputs,
    direct_path: str | Path,
) -> dict[str, object]:
    """Fail closed unless reconstruction and direct canonical scoring agree."""
    direct = ChainedJsonlLedger(direct_path, DIRECT_ROW_FORMAT)
    expected_keys = tuple(
        (specification.condition, story_id)
        for specification in SOURCE_SPECIFICATIONS
        for story_id in inputs.audit_story_ids
    )
    _validate_direct_rows(direct.rows, inputs.contract_sha256, expected_keys)
    direct_by_key = {
        (str(row["condition"]), str(row["story_id"])): row for row in direct.rows
    }
    source_by_key = {
        (specification.condition, str(row["story_id"])): row
        for specification, rows in inputs.sources
        for row in rows
    }
    short_errors: list[float] = []
    selection_mismatches = 0
    long_stories: set[str] = set()
    for key, direct_row in direct_by_key.items():
        source = source_by_key[key]
        if int(source["prefix_token_count"]) > CONTEXT_LENGTH:
            long_stories.add(str(source["story_id"]))
            continue
        reconstructed = reconstructed_whole_story_scores(source)
        measured = tuple(float(value) for value in direct_row["mean_nll_by_candidate"])
        short_errors.extend(
            abs(left - right)
            for left, right in zip(reconstructed, measured, strict=True)
        )
        selection_mismatches += int(
            stable_minimum(reconstructed)[0] != int(direct_row["selected_index"])
        )
    maximum_error = max(short_errors, default=0.0)
    audited = frozenset(inputs.audit_story_ids)
    minimum_unaudited_margin = min(
        stable_minimum(reconstructed_whole_story_scores(row))[1]
        for _, rows in inputs.sources
        for row in rows
        if str(row["story_id"]) not in audited
    )
    if (
        maximum_error > AUDIT_ABSOLUTE_TOLERANCE
        or selection_mismatches
        or minimum_unaudited_margin <= 2.0 * AUDIT_ABSOLUTE_TOLERANCE
    ):
        raise RuntimeError(
            "full-story reconstruction audit failed: "
            f"max_error={maximum_error:.8g}, selection_mismatches="
            f"{selection_mismatches}, unaudited_margin="
            f"{minimum_unaudited_margin:.8g}"
        )
    return {
        "audited_condition_story_rows": len(direct.rows),
        "audited_story_count": len(inputs.audit_story_ids),
        "long_story_count": len(long_stories),
        "maximum_short_score_absolute_error": maximum_error,
        "minimum_unaudited_margin": minimum_unaudited_margin,
        "selection_mismatches": selection_mismatches,
    }


def run_or_resume_case_derivation(
    inputs: FullStoryRoutingInputs,
    direct_path: str | Path,
) -> Path:
    """Derive all 13,320 paired cases into a resumable authenticated ledger."""
    direct = ChainedJsonlLedger(direct_path, DIRECT_ROW_FORMAT)
    direct_by_key = {
        (str(row["condition"]), str(row["story_id"])): row for row in direct.rows
    }
    expected_keys = tuple(
        (specification.condition, str(row["story_id"]))
        for specification, rows in inputs.sources
        for row in rows
    )
    ledger = ChainedJsonlLedger(inputs.work_directory / "cases.jsonl", CASE_ROW_FORMAT)
    _validate_case_rows(ledger.rows, inputs.contract_sha256, expected_keys)
    banks = inputs.bank_by_condition
    specifications = {value.condition: value for value in SOURCE_SPECIFICATIONS}
    flat_sources = tuple(
        (specification, row)
        for specification, rows in inputs.sources
        for row in rows
    )
    for specification, source in flat_sources[len(ledger.rows) :]:
        story_id = str(source["story_id"])
        direct_row = direct_by_key.get((specification.condition, story_id))
        scores = (
            tuple(float(value) for value in direct_row["mean_nll_by_candidate"])
            if direct_row is not None
            else reconstructed_whole_story_scores(source)
        )
        full_index, full_margin = stable_minimum(scores)
        midpoint_index = int(source["selected_index"])
        oracle_index = int(source["oracle_index"])
        suffix_scores = tuple(
            float(value) for value in source["suffix_mean_nll_by_candidate"]
        )
        suffix_tokens = int(source["suffix_token_count"])
        whole_tokens = int(source["prefix_token_count"]) + suffix_tokens
        bank = banks[specification.condition]
        full_route_hit = _route_hit(
            specification,
            bank,
            full_index,
            str(source["task_id"]),
        )
        midpoint_route_hit = bool(
            source[
                "noun_support_hit"
                if specification.accuracy_kind == "noun_support"
                else "exact_noun_route_hit"
            ]
        )
        values = {
            "condition": specification.condition,
            "contract_sha256": inputs.contract_sha256,
            "direct_score_used": direct_row is not None,
            "full_candidate_id": bank.candidate_ids[full_index],
            "full_index": full_index,
            "full_margin": full_margin,
            "full_route_hit": full_route_hit,
            "full_suffix_mean_nll": suffix_scores[full_index],
            "full_suffix_total_nll": suffix_scores[full_index] * suffix_tokens,
            "full_whole_mean_nll": scores[full_index],
            "full_whole_total_nll": scores[full_index] * whole_tokens,
            "midpoint_candidate_id": bank.candidate_ids[midpoint_index],
            "midpoint_index": midpoint_index,
            "midpoint_route_hit": midpoint_route_hit,
            "midpoint_suffix_mean_nll": float(source["suffix_mean_nll"]),
            "midpoint_suffix_total_nll": float(source["suffix_total_nll"]),
            "midpoint_whole_mean_nll": scores[midpoint_index],
            "midpoint_whole_total_nll": scores[midpoint_index] * whole_tokens,
            "oracle_candidate_id": bank.candidate_ids[oracle_index],
            "oracle_index": oracle_index,
            "oracle_suffix_mean_nll": float(source["oracle_suffix_mean_nll"]),
            "route_metric": specifications[specification.condition].accuracy_kind,
            "story_id": story_id,
            "suffix_token_count": suffix_tokens,
            "task_id": str(source["task_id"]),
            "whole_token_count": whole_tokens,
        }
        ledger.append(values)
    _validate_case_rows(ledger.rows, inputs.contract_sha256, expected_keys)
    return ledger.path


def analyze_full_story_routing(
    inputs: FullStoryRoutingInputs,
    case_path: str | Path,
    audit: Mapping[str, object],
) -> dict[str, object]:
    """Summarize paired routing, NLL, task, confusion, and bootstrap evidence."""
    ledger = ChainedJsonlLedger(case_path, CASE_ROW_FORMAT)
    rows_by_condition = {
        specification.condition: tuple(
            row
            for row in ledger.rows
            if row["condition"] == specification.condition
        )
        for specification in SOURCE_SPECIFICATIONS
    }
    aggregate = tuple(
        _summarize_rows(specification, rows_by_condition[specification.condition])
        for specification in SOURCE_SPECIFICATIONS
    )
    per_task = tuple(
        {
            **_summarize_rows(
                specification,
                tuple(
                    row
                    for row in rows_by_condition[specification.condition]
                    if row["task_id"] == task_id
                ),
            ),
            "task_id": task_id,
        }
        for specification in SOURCE_SPECIFICATIONS
        for task_id in TASK_IDS
    )
    confusion = tuple(
        {
            "condition": specification.condition,
            "count": sum(
                row["task_id"] == task_id
                and row[f"{route}_candidate_id"] == candidate_id
                for row in rows_by_condition[specification.condition]
            ),
            "route": route,
            "selected_candidate_id": candidate_id,
            "task_id": task_id,
        }
        for specification in SOURCE_SPECIFICATIONS
        for route in ("midpoint", "full")
        for task_id in TASK_IDS
        for candidate_id in inputs.bank_by_condition[specification.condition].candidate_ids
        if any(
            row["task_id"] == task_id
            and row[f"{route}_candidate_id"] == candidate_id
            for row in rows_by_condition[specification.condition]
        )
    )
    bootstrap = _bootstrap_differences(rows_by_condition)
    return {
        "aggregate": list(aggregate),
        "audit": dict(audit),
        "bootstrap": list(bootstrap),
        "confusion": list(confusion),
        "per_task": list(per_task),
        "provenance": {
            "audit_ledger": {
                "path": Path(inputs.work_directory / "direct-scores.jsonl").name,
                "row_count": EXPECTED_DIRECT_ROWS,
                "sha256": file_sha256(inputs.work_directory / "direct-scores.jsonl"),
            },
            "case_ledger": {
                "path": Path(case_path).name,
                "row_count": len(ledger.rows),
                "sha256": file_sha256(case_path),
            },
            "parent_contract_sha256": inputs.parent.contract_sha256,
            "parent_manifest_sha256": PARENT_MANIFEST_SHA256,
            "source_ledgers": [
                {
                    "condition": specification.condition,
                    "path": specification.relative_path,
                    "row_count": len(rows),
                    "sha256": specification.sha256,
                }
                for specification, rows in inputs.sources
            ],
        },
    }


def assert_parent_unchanged(inputs: FullStoryRoutingInputs) -> None:
    """Reject mutation of canonical nouns data or the immutable parent report."""
    assert_canonical_artifacts_unchanged(inputs.parent)
    if tuple(
        (relative, file_sha256(inputs.parent.result_directory / relative))
        for relative, _ in inputs.parent_snapshot
    ) != inputs.parent_snapshot:
        raise RuntimeError("temporal parent publication changed during addendum")


def _load_source_rows(
    parent: TemporalStudyInputs,
    specification: SourceSpecification,
    provenance: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    record = provenance.get(specification.relative_path)
    path = parent.work_directory / specification.relative_path
    if (
        type(record) is not dict
        or record.get("row_count") != 4_440
        or record.get("sha256") != specification.sha256
        or file_sha256(path) != specification.sha256
    ):
        raise ValueError(f"parent source provenance changed: {specification.condition}")
    ledger = ChainedJsonlLedger(path, EVALUATION_ROW_FORMAT)
    validate_evaluation_rows(ledger.rows, parent.contract_sha256)
    if len(ledger.rows) != 4_440:
        raise ValueError("parent source ledger coverage changed")
    return ledger.rows


def _load_final_banks(
    parent: TemporalStudyInputs,
) -> tuple[tuple[ConditionId, AdapterBank], ...]:
    return tuple(
        (
            condition,
            build_adapter_bank(
                tuple(candidate for candidate, _ in _candidate_artifact_paths(parent, condition)),
                parent.loaded_base.config,
            ),
        )
        for condition in (
            "blocked_log_t",
            "round_robin_log_t",
            "independent_noun",
        )
    )


def _candidate_artifact_paths(
    parent: TemporalStudyInputs,
    condition: ConditionId,
) -> tuple[tuple[AdapterCandidate, Path], ...]:
    lora_config = LoraConfig(rank=LORA_RANK, alpha=LORA_ALPHA)
    if condition == "independent_noun":
        specifications = tuple(
            (
                task_id,
                TrainingJob(
                    parent.contract_sha256,
                    f"independent-{task_id}",
                    "independent_noun",
                    tuple(
                        story_id
                        for shard in parent.shards
                        if shard.task_id == task_id
                        for story_id in shard.story_ids
                    ),
                    tuple(
                        shard.shard_id
                        for shard in parent.shards
                        if shard.task_id == task_id
                    ),
                ),
            )
            for task_id in TASK_IDS
        )
        return tuple(
            (
                AdapterCandidate(
                    f"noun-{task_id}",
                    artifact.adapter_sha256,
                    artifact.adapter,
                    ((task_id, SHARDS_PER_TASK),),
                ),
                artifact.directory,
            )
            for task_id, job in specifications
            for artifact in (
                load_adapter_artifact(
                    parent.checkpoint_directory
                    / "independent-noun"
                    / job.identity_sha256,
                    job,
                    parent.loaded_base.config,
                    lora_config,
                ),
            )
        )
    order = "blocked" if condition == "blocked_log_t" else "round_robin"
    state, _ = simulate_hierarchy(parent.shards, order)
    shard_by_id = {shard.shard_id: shard for shard in parent.shards}
    return tuple(
        _load_chunk_candidate(parent, chunk, shard_by_id, lora_config)
        for chunk in state.active_chunks
    )


def _load_chunk_candidate(
    parent: TemporalStudyInputs,
    chunk: TemporalChunk,
    shard_by_id: Mapping[str, object],
    lora_config: LoraConfig,
) -> tuple[AdapterCandidate, Path]:
    if chunk.level == 0:
        shard = shard_by_id[chunk.shard_ids[0]]
        job = TrainingJob(
            parent.contract_sha256,
            f"level-zero-{shard.task_id}-{shard.shard_index}",
            "level_zero",
            shard.story_ids,
            (shard.shard_id,),
        )
        family_directory = parent.checkpoint_directory / "level-zero"
    else:
        story_ids = tuple(
            story_id
            for shard_id in chunk.shard_ids
            for story_id in shard_by_id[shard_id].story_ids
        )
        job = TrainingJob(
            parent.contract_sha256,
            f"merge-{chunk.order}-{chunk.start_arrival:03d}-{chunk.end_arrival:03d}",
            "merge",
            story_ids,
            chunk.shard_ids,
            lineage_ids=chunk.parent_chunk_ids,
            order=chunk.order,
            level=chunk.level,
            start_arrival=chunk.start_arrival,
            end_arrival=chunk.end_arrival,
        )
        family_directory = parent.checkpoint_directory / "merges" / chunk.order
    artifact = load_adapter_artifact(
        family_directory / job.identity_sha256,
        job,
        parent.loaded_base.config,
        lora_config,
    )
    return (
        AdapterCandidate(
            f"interval-{chunk.start_arrival:03d}-{chunk.end_arrival:03d}-l{chunk.level}",
            artifact.adapter_sha256,
            artifact.adapter,
            chunk.task_counts,
            level=chunk.level,
            start_arrival=chunk.start_arrival,
            end_arrival=chunk.end_arrival,
        ),
        artifact.directory,
    )


def _stack_story_batches(
    batches: Sequence[TokenBatch],
) -> tuple[TokenBatch, tuple[tuple[int, int], ...]]:
    if not batches:
        raise ValueError("direct scoring requires at least one story")
    stops = np.cumsum([batch.input_ids.shape[0] for batch in batches])
    starts = np.concatenate((np.asarray([0]), stops[:-1]))
    return (
        TokenBatch(
            np.concatenate(tuple(batch.input_ids for batch in batches)),
            np.concatenate(tuple(batch.attention_mask for batch in batches)),
            np.concatenate(tuple(batch.target_ids for batch in batches)),
            np.concatenate(tuple(batch.loss_mask for batch in batches)),
        ),
        tuple((int(start), int(stop)) for start, stop in zip(starts, stops, strict=True)),
    )


def _route_hit(
    specification: SourceSpecification,
    bank: AdapterBank,
    selected_index: int,
    task_id: str,
) -> bool:
    if selected_index == 0:
        return False
    candidate = bank.candidates[selected_index - 1]
    return (
        candidate.candidate_id == f"noun-{task_id}"
        if specification.accuracy_kind == "exact_noun"
        else task_id in dict(candidate.task_counts)
    )


def _summarize_rows(
    specification: SourceSpecification,
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if not rows:
        raise ValueError("cannot summarize empty full-story rows")
    suffix_tokens = sum(int(row["suffix_token_count"]) for row in rows)
    whole_tokens = sum(int(row["whole_token_count"]) for row in rows)
    midpoint_suffix = float(np.mean([float(row["midpoint_suffix_mean_nll"]) for row in rows]))
    full_suffix = float(np.mean([float(row["full_suffix_mean_nll"]) for row in rows]))
    oracle_suffix = float(np.mean([float(row["oracle_suffix_mean_nll"]) for row in rows]))
    midpoint_gap = midpoint_suffix - oracle_suffix
    return {
        "accuracy_kind": specification.accuracy_kind,
        "condition": specification.condition,
        "full_oracle_agreement": float(np.mean([row["full_index"] == row["oracle_index"] for row in rows])),
        "full_route_accuracy": float(np.mean([bool(row["full_route_hit"]) for row in rows])),
        "full_suffix_story_nll": full_suffix,
        "full_suffix_token_nll": sum(float(row["full_suffix_total_nll"]) for row in rows) / suffix_tokens,
        "full_whole_story_nll": float(np.mean([float(row["full_whole_mean_nll"]) for row in rows])),
        "full_whole_token_nll": sum(float(row["full_whole_total_nll"]) for row in rows) / whole_tokens,
        "label": specification.label,
        "mean_full_margin": float(np.mean([float(row["full_margin"]) for row in rows])),
        "midpoint_full_agreement": float(np.mean([row["midpoint_index"] == row["full_index"] for row in rows])),
        "midpoint_oracle_agreement": float(np.mean([row["midpoint_index"] == row["oracle_index"] for row in rows])),
        "midpoint_route_accuracy": float(np.mean([bool(row["midpoint_route_hit"]) for row in rows])),
        "midpoint_selected_whole_story_nll": float(np.mean([float(row["midpoint_whole_mean_nll"]) for row in rows])),
        "midpoint_selected_whole_token_nll": sum(float(row["midpoint_whole_total_nll"]) for row in rows) / whole_tokens,
        "midpoint_suffix_story_nll": midpoint_suffix,
        "midpoint_suffix_token_nll": sum(float(row["midpoint_suffix_total_nll"]) for row in rows) / suffix_tokens,
        "oracle_suffix_story_nll": oracle_suffix,
        "oracle_suffix_token_nll": sum(float(row["oracle_suffix_mean_nll"]) * int(row["suffix_token_count"]) for row in rows) / suffix_tokens,
        "route_accuracy_change_pp": 100.0 * float(np.mean([int(bool(row["full_route_hit"])) - int(bool(row["midpoint_route_hit"])) for row in rows])),
        "story_count": len(rows),
        "suffix_gap_recovered_fraction": (
            (midpoint_suffix - full_suffix) / midpoint_gap if midpoint_gap > 0.0 else None
        ),
        "suffix_story_nll_change": full_suffix - midpoint_suffix,
    }


def _bootstrap_differences(
    rows_by_condition: Mapping[ConditionId, Sequence[Mapping[str, object]]],
) -> tuple[dict[str, object], ...]:
    rng = np.random.default_rng(SEED)
    records: list[dict[str, object]] = []
    metric_values = (
        (
            "route_accuracy_change",
            lambda row: float(bool(row["full_route_hit"])) - float(bool(row["midpoint_route_hit"])),
        ),
        (
            "suffix_story_nll_change",
            lambda row: float(row["full_suffix_mean_nll"]) - float(row["midpoint_suffix_mean_nll"]),
        ),
        (
            "whole_story_nll_change",
            lambda row: float(row["full_whole_mean_nll"]) - float(row["midpoint_whole_mean_nll"]),
        ),
    )
    for specification in SOURCE_SPECIFICATIONS:
        condition_rows = rows_by_condition[specification.condition]
        strata = tuple(
            np.asarray(
                [
                    [value(row) for row in condition_rows if row["task_id"] == task_id]
                    for _, value in metric_values
                ],
                dtype=np.float64,
            )
            for task_id in TASK_IDS
        )
        if any(stratum.shape[1] == 0 for stratum in strata):
            raise ValueError("bootstrap requires every noun stratum")
        samples = np.zeros(
            (len(metric_values), BOOTSTRAP_REPETITIONS),
            dtype=np.float64,
        )
        total_rows = sum(stratum.shape[1] for stratum in strata)
        chunk_size = 500
        for stratum in strata:
            for start in range(0, BOOTSTRAP_REPETITIONS, chunk_size):
                stop = min(start + chunk_size, BOOTSTRAP_REPETITIONS)
                indices = rng.integers(
                    0,
                    stratum.shape[1],
                    size=(stop - start, stratum.shape[1]),
                )
                samples[:, start:stop] += (
                    np.mean(stratum[:, indices], axis=2) * stratum.shape[1]
                )
        samples /= total_rows
        for metric_index, (metric, _) in enumerate(metric_values):
            values = np.concatenate(
                tuple(stratum[metric_index] for stratum in strata)
            )
            records.append(
                {
                    "condition": specification.condition,
                    "estimate": float(np.mean(values)),
                    "lower_95": float(np.quantile(samples[metric_index], 0.025)),
                    "metric": metric,
                    "repetitions": BOOTSTRAP_REPETITIONS,
                    "seed": SEED,
                    "upper_95": float(np.quantile(samples[metric_index], 0.975)),
                }
            )
    return tuple(records)


def _validate_direct_rows(
    rows: Sequence[Mapping[str, object]],
    contract_sha256: str,
    expected_keys: Sequence[tuple[ConditionId, str]],
) -> None:
    if len(rows) > len(expected_keys):
        raise ValueError("direct-score ledger exceeds expected coverage")
    for row, expected_key in zip(rows, expected_keys, strict=False):
        candidate_ids = row.get("candidate_ids")
        totals = row.get("total_nll_by_candidate")
        means = row.get("mean_nll_by_candidate")
        token_count = row.get("token_count")
        selected = row.get("selected_index")
        if (
            (row.get("condition"), row.get("story_id")) != expected_key
            or row.get("contract_sha256") != contract_sha256
            or type(candidate_ids) is not list
            or type(totals) is not list
            or type(means) is not list
            or not candidate_ids
            or len(candidate_ids) != len(totals)
            or len(candidate_ids) != len(means)
            or type(token_count) is not int
            or token_count <= 0
            or type(selected) is not int
            or not 0 <= selected < len(candidate_ids)
            or row.get("selected_candidate_id") != candidate_ids[selected]
            or any(not math.isfinite(float(value)) or float(value) < 0 for value in (*totals, *means))
            or any(
                not math.isclose(float(total) / token_count, float(mean), abs_tol=1e-10)
                for total, mean in zip(totals, means, strict=True)
            )
        ):
            raise ValueError("direct-score ledger row changed")


def _validate_case_rows(
    rows: Sequence[Mapping[str, object]],
    contract_sha256: str,
    expected_keys: Sequence[tuple[ConditionId, str]],
) -> None:
    if len(rows) > len(expected_keys):
        raise ValueError("full-story case ledger exceeds expected coverage")
    for row, expected_key in zip(rows, expected_keys, strict=False):
        if (
            (row.get("condition"), row.get("story_id")) != expected_key
            or row.get("contract_sha256") != contract_sha256
            or row.get("route_metric") not in ("noun_support", "exact_noun")
            or type(row.get("full_route_hit")) is not bool
            or type(row.get("midpoint_route_hit")) is not bool
            or type(row.get("direct_score_used")) is not bool
            or type(row.get("suffix_token_count")) is not int
            or type(row.get("whole_token_count")) is not int
            or int(row.get("suffix_token_count", 0)) <= 0
            or int(row.get("whole_token_count", 0)) <= 0
        ):
            raise ValueError("full-story case ledger row changed")


__all__ = [
    "ALLOCATOR_LIMIT_BYTES",
    "AUDIT_ABSOLUTE_TOLERANCE",
    "AUDIT_MARGIN_THRESHOLD",
    "CASE_ROW_FORMAT",
    "DIRECT_ROW_FORMAT",
    "EXPECTED_AUDIT_STORIES",
    "EXPECTED_DIRECT_ROWS",
    "EXPECTED_SOURCE_ROWS",
    "FullStoryRoutingInputs",
    "REPORT_FORMAT",
    "SOURCE_SPECIFICATIONS",
    "STUDY_ID",
    "analyze_full_story_routing",
    "assert_parent_unchanged",
    "authenticate_full_story_routing_inputs",
    "reconstructed_whole_story_scores",
    "run_or_resume_case_derivation",
    "run_or_resume_direct_audit",
    "select_audit_story_ids",
    "stable_minimum",
    "verify_direct_audit",
]

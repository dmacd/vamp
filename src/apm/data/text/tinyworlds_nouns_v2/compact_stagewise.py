"""Resumable stagewise evaluation of canonical-key compact top-eight EBT-H."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from hashlib import sha256
from itertools import zip_longest
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from apm.continual.language_adaptation_artifact import LanguageAdaptationArtifact
from apm.data.text.tinyworlds_nouns_v1.evaluation import (
    _HalfStoryCase,
    _half_story_chunks,
    _nll_by_node_per_window,
    _node_path,
    _pad_router_batch,
    _stack_prefix_queries,
    _stack_token_batches,
)
from apm.data.text.tinyworlds_nouns_v1.experiment import (
    IndexedStoryStore,
    NounSelectedBase,
    StoryIndexEntry,
)
from apm.data.text.tinyworlds_nouns_v2.addressing_study_contracts import (
    EBT_ENTROPY_PENALTY,
    EBT_LEARNING_RATE,
    EBT_STEPS,
    EBT_TEMPERATURE,
    HOPFIELD_BETA,
    MICROBATCH_SIZE,
)
from apm.data.text.tinyworlds_nouns_v2.addressing_study_keys import (
    stable_hopfield_result,
)
from apm.data.text.tinyworlds_nouns_v2.contracts import (
    CONDITIONS,
    STAGEWISE_CASE_COUNT,
    TASK_IDS,
    NounsV2ExperimentPreset,
    NounsV2PartitionArtifact,
    canonical_json_bytes,
    record_sha256,
)
from apm.data.text.tinyworlds_nouns_v2.stagewise import (
    summarize_stagewise_rows,
    validate_stagewise_ledger,
)
from apm.data.text.tinyworlds_nouns_v2.stagewise_common import (
    expected_stagewise_keys,
    file_sha256,
    generation_entries,
    object_record,
    repair_interrupted_tail,
)
from apm.lm.compact_lora_memory import (
    COMPACT_EDGE_CAPACITY_BUCKETS,
    gather_compact_lora_memory,
)
from apm.lm.checkpoint import load_gpt_neo_checkpoint
from apm.lm.lora_memory import PackedLoraMemory, pack_lora_memory
from apm.lm.parameters import GptNeoParams
from apm.memory.address_refinement import EbtConfig, refine_compact_ebt_address
from apm.memory.content_keys import encode_frozen_base_content


COMPACT_STAGEWISE_CONDITION = "vamp_ebt_hopfield_compact_top8"
COMPACT_STAGEWISE_LABEL = "VAMP compact top-8 EBT-H"
COMPACT_STAGEWISE_CONTRACT_FORMAT = (
    "tinyworlds-nouns-v2-compact-stagewise-contract-v1"
)
COMPACT_STAGEWISE_ROW_FORMAT = "tinyworlds-nouns-v2-compact-stagewise-row-v1"
COMPACT_STAGEWISE_SUMMARY_FORMAT = (
    "tinyworlds-nouns-v2-stagewise-with-compact-top8-v1"
)
COMPACT_STAGEWISE_CONTRACT_FILENAME = "compact-stagewise-contract.json"
REPORT_STAGEWISE_CONDITIONS = (*CONDITIONS, COMPACT_STAGEWISE_CONDITION)
REPORT_ROUTED_CONDITIONS = (*CONDITIONS[2:], COMPACT_STAGEWISE_CONDITION)

ProgressCallback = Callable[[str, int, int], None]

_ROW_FIELDS = {
    "candidate_node_indices",
    "candidate_width",
    "compact_stagewise_contract_sha256",
    "format",
    "gathered_edge_count",
    "introduced_task",
    "oracle_match",
    "oracle_node_index",
    "oracle_suffix_mean_nll",
    "physical_edge_capacity",
    "prefix_token_count",
    "prefix_width_bucket",
    "regret_vs_oracle",
    "result_sha256",
    "selected_node",
    "selected_node_index",
    "selected_path",
    "stage_index",
    "stage_tensor_checksum",
    "story_id",
    "suffix_mean_nll",
    "suffix_token_count",
    "suffix_total_nll",
    "task_noun",
}


def build_or_load_compact_stagewise_contract(
    partition: NounsV2PartitionArtifact,
    preset: NounsV2ExperimentPreset,
    selected_base: NounSelectedBase,
    adaptations: tuple[LanguageAdaptationArtifact, ...],
    canonical_stagewise_path: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    """Publish the immutable contract for the sole compact stagewise method."""
    metadata = _stage_metadata(partition, adaptations)
    canonical_path = Path(canonical_stagewise_path)
    validate_stagewise_ledger(
        canonical_path,
        partition,
        adaptations,
        require_complete=True,
    )
    core = {
        "base_parameter_checksum": selected_base.reference.parameter_checksum,
        "base_training_sha256": selected_base.training_sha256,
        "canonical_stagewise_sha256": file_sha256(canonical_path),
        "compact_edge_capacity_buckets": list(COMPACT_EDGE_CAPACITY_BUCKETS),
        "ebt": {
            "entropy_penalty": EBT_ENTROPY_PENALTY,
            "learning_rate": EBT_LEARNING_RATE,
            "steps": EBT_STEPS,
            "temperature": EBT_TEMPERATURE,
        },
        "expected_row_count": STAGEWISE_CASE_COUNT,
        "format": COMPACT_STAGEWISE_CONTRACT_FORMAT,
        "hopfield_beta": HOPFIELD_BETA,
        "key_scheme": "canonical_full_centroid",
        "microbatch_size": MICROBATCH_SIZE,
        "partition_sha256": partition.partition_sha256,
        "preset_sha256": preset.config_sha256,
        "schema_version": 1,
        "stage_address_key_sha256": [
            metadata[index]["address_key_sha256"]
            for index in range(1, len(adaptations) + 1)
        ],
        "stage_tensor_checksums": [
            metadata[index]["checksum"]
            for index in range(1, len(adaptations) + 1)
        ],
        "top_k": 8,
    }
    record = {**core, "contract_sha256": record_sha256(core)}
    _publish_immutable_json(Path(output_path), record)
    return record


def load_compact_stagewise_contract(
    path: str | Path,
) -> dict[str, object]:
    """Strict-load one canonical, self-hashed compact stagewise contract."""
    source = Path(path)
    payload = source.read_bytes()
    record = json.loads(payload)
    core = (
        {key: value for key, value in record.items() if key != "contract_sha256"}
        if type(record) is dict
        else {}
    )
    if (
        type(record) is not dict
        or payload != canonical_json_bytes(record)
        or record.get("format") != COMPACT_STAGEWISE_CONTRACT_FORMAT
        or record.get("contract_sha256") != record_sha256(core)
    ):
        raise ValueError("compact stagewise contract identity changed")
    return record


def evaluate_compact_stagewise_continual_learning(
    partition: NounsV2PartitionArtifact,
    preset: NounsV2ExperimentPreset,
    selected_base: NounSelectedBase,
    adaptations: tuple[LanguageAdaptationArtifact, ...],
    contract: dict[str, object],
    output_path: str | Path,
    *,
    progress: ProgressCallback | None = None,
) -> Path:
    """Evaluate canonical-centroid compact top-eight EBT-H at all frozen stages."""
    metadata = _stage_metadata(partition, adaptations)
    _validate_contract_bindings(
        contract,
        partition,
        preset,
        selected_base,
        adaptations,
        metadata,
    )
    entries_by_task = generation_entries(partition)
    expected_order = _expected_order(partition.task_ids, entries_by_task)
    expected = set(expected_order)
    output = Path(output_path)
    contract_sha256 = str(contract["contract_sha256"])
    if output.is_file():
        completed = validate_compact_stagewise_ledger(
            output,
            contract_sha256,
            partition,
            adaptations,
            require_complete=True,
            entries_by_task=entries_by_task,
        )
        if completed != expected:
            raise ValueError("published compact stagewise coverage changed")
        return output

    work_directory = output.parent / ".compact-stagewise-work"
    work_directory.mkdir(parents=True, exist_ok=True)
    work = work_directory / output.name
    print(f"Compact stagewise temporary directory: {work_directory.resolve()}", flush=True)
    repair_interrupted_tail(work)
    completed = validate_compact_stagewise_ledger(
        work,
        contract_sha256,
        partition,
        adaptations,
        require_complete=False,
        entries_by_task=entries_by_task,
    )
    if not completed <= expected:
        raise ValueError("compact stagewise work ledger contains unexpected rows")

    loaded = load_gpt_neo_checkpoint(selected_base.reference)
    if loaded.reference.parameter_checksum != selected_base.reference.parameter_checksum:
        raise ValueError("compact stagewise selected-base parameters changed")
    base_params = loaded.params
    store = IndexedStoryStore(partition)
    config = EbtConfig(
        steps=EBT_STEPS,
        learning_rate=EBT_LEARNING_RATE,
        tau=EBT_TEMPERATURE,
        entropy_penalty=EBT_ENTROPY_PENALTY,
        initialization="hopfield_top_k",
    )
    finished = len(completed)
    with work.open("ab") as ledger:
        for stage_index, adaptation in enumerate(adaptations, start=1):
            stage_key = str(stage_index)
            learned_entries = {
                task_id: entries_by_task[task_id]
                for task_id in partition.task_ids[:stage_index]
            }
            completed_at_stage = {
                (task_id, story_id)
                for existing_stage, task_id, story_id in completed
                if existing_stage == stage_key
            }
            packed = pack_lora_memory(
                adaptation.vamp_graph,
                adaptation.model_config,
                adaptation.lora_config,
                adaptation.max_nodes,
                adaptation.max_edges,
            )
            chunks = _half_story_chunks(
                partition,
                learned_entries,
                completed_at_stage,
                store,
                preset,
                adaptation.model_config.max_position_embeddings,
            )
            for cases in _microbatches(chunks):
                payloads = _evaluate_case_batch(
                    cases,
                    partition,
                    preset,
                    base_params,
                    adaptation,
                    packed,
                    config,
                    contract_sha256,
                    metadata[stage_index],
                )
                ledger.write(b"".join(payloads))
                ledger.flush()
                os.fsync(ledger.fileno())
                for case in cases:
                    completed.add((stage_key, case.task_id, case.entry.story_id))
                    finished += 1
                    if progress is not None:
                        progress("compact-stagewise-cl", finished, len(expected_order))
    if completed != expected:
        raise RuntimeError(
            f"compact stagewise ledger has {len(completed):,} of "
            f"{len(expected):,} rows"
        )
    os.replace(work, output)
    return output


def validate_compact_stagewise_ledger(
    path: str | Path,
    contract_sha256: str,
    partition: NounsV2PartitionArtifact,
    adaptations: tuple[LanguageAdaptationArtifact, ...],
    *,
    require_complete: bool,
    entries_by_task: dict[str, tuple[StoryIndexEntry, ...]] | None = None,
) -> set[tuple[str, str, str]]:
    """Reject malformed, tampered, reordered, duplicate, or incomplete rows."""
    metadata = _stage_metadata(partition, adaptations)
    entries = entries_by_task or generation_entries(partition)
    expected_order = _expected_order(partition.task_ids, entries)
    expected = set(expected_order)
    source = Path(path)
    if not source.is_file():
        if require_complete:
            raise FileNotFoundError(source)
        return set()
    observed: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    with source.open("rb") as stream:
        for line in stream:
            row = _validated_compact_row(line, contract_sha256, metadata)
            key = (
                str(row["stage_index"]),
                str(row["task_noun"]),
                str(row["story_id"]),
            )
            if key in seen or key not in expected:
                raise ValueError("compact stagewise ledger has duplicate or unexpected rows")
            observed.append(key)
            seen.add(key)
    if tuple(observed) != expected_order[: len(observed)]:
        raise ValueError("compact stagewise ledger is not a resumable canonical prefix")
    completed = set(observed)
    if require_complete and completed != expected:
        raise ValueError(
            f"compact stagewise ledger has {len(completed):,} of "
            f"{len(expected):,} expected rows"
        )
    return completed


def summarize_stagewise_with_compact_ledger(
    canonical_path: str | Path,
    compact_path: str | Path,
    partition: NounsV2PartitionArtifact,
    adaptations: tuple[LanguageAdaptationArtifact, ...],
) -> dict[str, object]:
    """Join the independent compact rows to canonical rows for report summaries."""
    canonical = Path(canonical_path)
    compact = Path(compact_path)
    contract = load_compact_stagewise_contract(
        compact.parent / COMPACT_STAGEWISE_CONTRACT_FILENAME
    )
    _validate_summary_bindings(contract, canonical, partition, adaptations)
    validate_stagewise_ledger(
        canonical,
        partition,
        adaptations,
        require_complete=True,
    )
    validate_compact_stagewise_ledger(
        compact,
        str(contract["contract_sha256"]),
        partition,
        adaptations,
        require_complete=True,
    )

    def merged_rows() -> Iterator[dict[str, object]]:
        with canonical.open("rb") as canonical_stream, compact.open("rb") as compact_stream:
            for canonical_line, compact_line in zip_longest(
                canonical_stream,
                compact_stream,
            ):
                if canonical_line is None or compact_line is None:
                    raise ValueError("canonical and compact stagewise row counts differ")
                canonical_row = object_record(json.loads(canonical_line), "canonical row")
                compact_row = object_record(json.loads(compact_line), "compact row")
                canonical_key = tuple(
                    canonical_row[field]
                    for field in ("stage_index", "task_noun", "story_id")
                )
                compact_key = tuple(
                    compact_row[field]
                    for field in ("stage_index", "task_noun", "story_id")
                )
                canonical_results = object_record(
                    canonical_row["results"],
                    "canonical results",
                )
                oracle = object_record(canonical_results["oracle"], "oracle result")
                if (
                    canonical_key != compact_key
                    or int(oracle["token_count"])
                    != int(compact_row["suffix_token_count"])
                    or not math.isclose(
                        float(oracle["mean_nll"]),
                        float(compact_row["oracle_suffix_mean_nll"]),
                        rel_tol=1e-6,
                        abs_tol=1e-6,
                    )
                ):
                    raise ValueError("compact rows differ from canonical oracle bindings")
                compact_result = {
                    "condition": COMPACT_STAGEWISE_CONDITION,
                    "mean_nll": compact_row["suffix_mean_nll"],
                    "oracle_match": compact_row["oracle_match"],
                    "regret_vs_oracle": compact_row["regret_vs_oracle"],
                    "selected_node": compact_row["selected_node"],
                    "selected_path": compact_row["selected_path"],
                    "token_count": compact_row["suffix_token_count"],
                    "total_nll": compact_row["suffix_total_nll"],
                }
                yield {
                    **canonical_row,
                    "results": {
                        **canonical_results,
                        COMPACT_STAGEWISE_CONDITION: compact_result,
                    },
                }

    measured = summarize_stagewise_rows(
        merged_rows(),
        partition.task_ids,
        conditions=REPORT_STAGEWISE_CONDITIONS,
    )
    summary = {
        **measured,
        "canonical_ledger_sha256": file_sha256(canonical),
        "compact_contract_sha256": contract["contract_sha256"],
        "compact_ledger_sha256": file_sha256(compact),
        "format": COMPACT_STAGEWISE_SUMMARY_FORMAT,
        "stage_tensor_checksums": contract["stage_tensor_checksums"],
    }
    return {**summary, "summary_sha256": record_sha256(summary)}


def _evaluate_case_batch(
    cases: tuple[_HalfStoryCase, ...],
    partition: NounsV2PartitionArtifact,
    preset: NounsV2ExperimentPreset,
    base_params: GptNeoParams,
    adaptation: LanguageAdaptationArtifact,
    packed: PackedLoraMemory,
    config: EbtConfig,
    contract_sha256: str,
    stage_metadata: dict[str, object],
) -> tuple[bytes, ...]:
    prefix = _pad_router_batch(
        _stack_prefix_queries(
            cases,
            adaptation.model_config.max_position_embeddings,
        ),
        MICROBATCH_SIZE,
    )
    content = encode_frozen_base_content(
        base_params,
        adaptation.model_config,
        prefix.input_ids,
        prefix.attention_mask,
        evaluation_microbatch_size=MICROBATCH_SIZE,
    )
    node_count = len(adaptation.vamp_graph.nodes)
    candidate_width = min(8, node_count)
    scores = np.asarray(content, dtype=np.float32) @ np.asarray(
        adaptation.address_book.keys[:node_count],
        dtype=np.float32,
    ).T
    hopfield = stable_hopfield_result(
        scores,
        top_k=candidate_width,
        beta=HOPFIELD_BETA,
    )
    candidate_indices = np.asarray(hopfield.top_k_indices, dtype=np.int32)
    compact = gather_compact_lora_memory(packed, candidate_indices)
    refined = refine_compact_ebt_address(
        base_params,
        adaptation.model_config,
        compact,
        adaptation.lora_config,
        prefix,
        hopfield,
        config,
    )
    selected = np.asarray(refined.selected_node_indices, dtype=np.int32)[: len(cases)]
    node_indices = tuple(
        sorted(
            {case.oracle_index for case in cases}
            | {int(value) for value in selected}
        )
    )
    suffix_windows = _stack_token_batches(tuple(case.suffix_windows for case in cases))
    per_window = _nll_by_node_per_window(
        base_params,
        adaptation,
        packed,
        suffix_windows,
        preset.evaluation_chunk_size,
        node_indices=node_indices,
    )
    node_row = {node_index: row for row, node_index in enumerate(node_indices)}
    boundaries = np.cumsum(
        (0,) + tuple(case.suffix_windows.input_ids.shape[0] for case in cases)
    )
    gathered_counts = np.sum(
        np.asarray(compact.valid_edge_mask, dtype=np.int32),
        axis=1,
    )
    prefix_width = prefix.input_ids.shape[1]
    payloads = []
    for row, (case, start, stop) in enumerate(
        zip(cases, boundaries[:-1], boundaries[1:])
    ):
        selected_index = int(selected[row])
        token_count = int(np.sum(case.suffix_windows.loss_mask))
        selected_total = float(
            np.sum(
                per_window[node_row[selected_index], start:stop],
                dtype=np.float64,
            )
        )
        oracle_total = float(
            np.sum(
                per_window[node_row[case.oracle_index], start:stop],
                dtype=np.float64,
            )
        )
        selected_mean = selected_total / token_count
        oracle_mean = oracle_total / token_count
        selected_node = str(adaptation.vamp_graph.nodes[selected_index].node_id)
        core = {
            "candidate_node_indices": [
                int(value) for value in candidate_indices[row]
            ],
            "candidate_width": candidate_width,
            "compact_stagewise_contract_sha256": contract_sha256,
            "format": COMPACT_STAGEWISE_ROW_FORMAT,
            "gathered_edge_count": int(gathered_counts[row]),
            "introduced_task": stage_metadata["introduced_task"],
            "oracle_match": selected_index == case.oracle_index,
            "oracle_node_index": case.oracle_index,
            "oracle_suffix_mean_nll": oracle_mean,
            "physical_edge_capacity": int(compact.valid_edge_mask.shape[1]),
            "prefix_token_count": int(np.sum(case.query.router_batch.loss_mask)),
            "prefix_width_bucket": prefix_width,
            "regret_vs_oracle": selected_mean - oracle_mean,
            "selected_node": selected_node,
            "selected_node_index": selected_index,
            "selected_path": list(_node_path(adaptation, selected_index)),
            "stage_index": int(stage_metadata["stage_index"]),
            "stage_tensor_checksum": stage_metadata["checksum"],
            "story_id": case.entry.story_id,
            "suffix_mean_nll": selected_mean,
            "suffix_token_count": token_count,
            "suffix_total_nll": selected_total,
            "task_noun": case.task_id,
        }
        payloads.append(
            canonical_json_bytes({**core, "result_sha256": record_sha256(core)})
        )
    return tuple(payloads)


def _microbatches(
    chunks: Iterable[tuple[_HalfStoryCase, ...]],
) -> Iterator[tuple[_HalfStoryCase, ...]]:
    for chunk in chunks:
        for start in range(0, len(chunk), MICROBATCH_SIZE):
            yield chunk[start : start + MICROBATCH_SIZE]


def _expected_order(
    task_ids: tuple[str, ...],
    entries_by_task: dict[str, tuple[StoryIndexEntry, ...]],
) -> tuple[tuple[str, str, str], ...]:
    expected = tuple(
        (str(stage), task_id, entry.story_id)
        for stage in range(1, len(task_ids) + 1)
        for task_id in task_ids[:stage]
        for entry in entries_by_task[task_id]
    )
    if set(expected) != expected_stagewise_keys(task_ids, entries_by_task):
        raise ValueError("compact stagewise expected ordering changed")
    return expected


def _stage_metadata(
    partition: NounsV2PartitionArtifact,
    adaptations: tuple[LanguageAdaptationArtifact, ...],
) -> dict[int, dict[str, object]]:
    if len(adaptations) != len(partition.task_ids) or not adaptations:
        raise ValueError("compact stagewise evaluation requires every VAMP stage")
    metadata: dict[int, dict[str, object]] = {}
    for stage_index, adaptation in enumerate(adaptations, start=1):
        expected_nodes = ("root", *partition.task_ids[:stage_index])
        observed_nodes = tuple(str(node.node_id) for node in adaptation.vamp_graph.nodes)
        if (
            observed_nodes != expected_nodes
            or tuple(str(task) for task in adaptation.task_order)
            != partition.task_ids[:stage_index]
        ):
            raise ValueError("compact stage is not the canonical graph prefix")
        valid_keys = np.asarray(
            adaptation.address_book.keys[: len(expected_nodes)],
            dtype=np.float32,
        )
        if not np.allclose(
            np.linalg.norm(valid_keys, axis=-1),
            1.0,
            rtol=1e-5,
            atol=1e-6,
        ):
            raise ValueError("compact stage canonical address keys are not normalized")
        metadata[stage_index] = {
            "address_key_sha256": _array_sha256(valid_keys),
            "checksum": adaptation.tensor_checksum,
            "introduced_task": partition.task_ids[stage_index - 1],
            "node_ids": observed_nodes,
            "paths": tuple(
                _node_path(adaptation, node_index)
                for node_index in range(len(observed_nodes))
            ),
            "stage_index": stage_index,
        }
    return metadata


def _validated_compact_row(
    line: bytes,
    contract_sha256: str,
    metadata: dict[int, dict[str, object]],
) -> dict[str, object]:
    if not line.endswith(b"\n"):
        raise ValueError("compact stagewise ledger has an interrupted tail")
    row = json.loads(line)
    if type(row) is not dict or line != canonical_json_bytes(row):
        raise ValueError("compact stagewise ledger is not canonical JSONL")
    core = {key: value for key, value in row.items() if key != "result_sha256"}
    if (
        set(row) != _ROW_FIELDS
        or row.get("format") != COMPACT_STAGEWISE_ROW_FORMAT
        or row.get("compact_stagewise_contract_sha256") != contract_sha256
        or row.get("result_sha256") != record_sha256(core)
    ):
        raise ValueError("compact stagewise row identity changed")
    stage = row.get("stage_index")
    if type(stage) is not int or stage not in metadata:
        raise ValueError("compact stagewise row stage changed")
    stage_info = metadata[stage]
    node_ids = tuple(str(value) for value in stage_info["node_ids"])
    paths = tuple(tuple(path) for path in stage_info["paths"])
    task = row.get("task_noun")
    oracle_index = row.get("oracle_node_index")
    candidates = row.get("candidate_node_indices")
    selected_index = row.get("selected_node_index")
    expected_width = min(8, len(node_ids))
    if (
        row.get("introduced_task") != stage_info["introduced_task"]
        or row.get("stage_tensor_checksum") != stage_info["checksum"]
        or type(task) is not str
        or task not in TASK_IDS[:stage]
        or type(oracle_index) is not int
        or oracle_index != TASK_IDS.index(task) + 1
        or row.get("candidate_width") != expected_width
        or type(candidates) is not list
        or len(candidates) != expected_width
        or len(set(candidates)) != expected_width
        or any(type(value) is not int or not 0 <= value < len(node_ids) for value in candidates)
        or type(selected_index) is not int
        or selected_index not in candidates
        or row.get("selected_node") != node_ids[selected_index]
        or row.get("selected_path") != list(paths[selected_index])
        or type(row.get("oracle_match")) is not bool
        or row.get("oracle_match") != (selected_index == oracle_index)
    ):
        raise ValueError("compact stagewise route metadata changed")
    gathered = row.get("gathered_edge_count")
    physical = row.get("physical_edge_capacity")
    prefix_tokens = row.get("prefix_token_count")
    prefix_width = row.get("prefix_width_bucket")
    suffix_tokens = row.get("suffix_token_count")
    if (
        type(gathered) is not int
        or not len(paths[selected_index]) - 1 <= gathered <= stage
        or type(physical) is not int
        or physical not in COMPACT_EDGE_CAPACITY_BUCKETS
        or gathered > physical
        or type(prefix_tokens) is not int
        or prefix_tokens <= 0
        or type(prefix_width) is not int
        or prefix_width <= 0
        or prefix_width % 32 != 0
        or type(suffix_tokens) is not int
        or suffix_tokens <= 0
    ):
        raise ValueError("compact stagewise operation counts changed")
    numeric = tuple(
        row.get(field)
        for field in (
            "oracle_suffix_mean_nll",
            "regret_vs_oracle",
            "suffix_mean_nll",
            "suffix_total_nll",
        )
    )
    if any(
        type(value) not in (int, float) or not math.isfinite(float(value))
        for value in numeric
    ):
        raise ValueError("compact stagewise NLL values must be finite")
    oracle_mean, regret, suffix_mean, suffix_total = map(float, numeric)
    if oracle_mean < 0.0 or suffix_mean < 0.0 or suffix_total < 0.0 or not (
        math.isclose(
            suffix_mean,
            suffix_total / suffix_tokens,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        and math.isclose(
            regret,
            suffix_mean - oracle_mean,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("compact stagewise NLL arithmetic changed")
    return row


def _validate_contract_bindings(
    contract: dict[str, object],
    partition: NounsV2PartitionArtifact,
    preset: NounsV2ExperimentPreset,
    selected_base: NounSelectedBase,
    adaptations: tuple[LanguageAdaptationArtifact, ...],
    metadata: dict[int, dict[str, object]],
) -> None:
    contract_core = {
        key: value for key, value in contract.items() if key != "contract_sha256"
    }
    if (
        contract.get("format") != COMPACT_STAGEWISE_CONTRACT_FORMAT
        or contract.get("contract_sha256") != record_sha256(contract_core)
        or contract.get("partition_sha256") != partition.partition_sha256
        or contract.get("preset_sha256") != preset.config_sha256
        or contract.get("base_training_sha256") != selected_base.training_sha256
        or contract.get("base_parameter_checksum")
        != selected_base.reference.parameter_checksum
        or contract.get("stage_tensor_checksums")
        != [metadata[index]["checksum"] for index in range(1, len(adaptations) + 1)]
        or contract.get("stage_address_key_sha256")
        != [
            metadata[index]["address_key_sha256"]
            for index in range(1, len(adaptations) + 1)
        ]
    ):
        raise ValueError("compact stagewise contract bindings changed")


def _validate_summary_bindings(
    contract: dict[str, object],
    canonical_path: Path,
    partition: NounsV2PartitionArtifact,
    adaptations: tuple[LanguageAdaptationArtifact, ...],
) -> None:
    metadata = _stage_metadata(partition, adaptations)
    if (
        contract.get("canonical_stagewise_sha256") != file_sha256(canonical_path)
        or contract.get("partition_sha256") != partition.partition_sha256
        or contract.get("stage_tensor_checksums")
        != [metadata[index]["checksum"] for index in range(1, len(adaptations) + 1)]
    ):
        raise ValueError("compact summary source bindings changed")


def _array_sha256(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    digest = sha256()
    digest.update(values.dtype.str.encode("ascii"))
    digest.update(str(values.shape).encode("ascii"))
    digest.update(values.tobytes())
    return digest.hexdigest()


def _publish_immutable_json(path: Path, record: dict[str, object]) -> None:
    payload = canonical_json_bytes(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if path.read_bytes() != payload:
            raise ValueError(f"immutable compact stagewise artifact changed: {path.name}")
        return
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


__all__ = [
    "COMPACT_STAGEWISE_CONDITION",
    "COMPACT_STAGEWISE_CONTRACT_FILENAME",
    "COMPACT_STAGEWISE_LABEL",
    "REPORT_ROUTED_CONDITIONS",
    "REPORT_STAGEWISE_CONDITIONS",
    "build_or_load_compact_stagewise_contract",
    "evaluate_compact_stagewise_continual_learning",
    "load_compact_stagewise_contract",
    "summarize_stagewise_with_compact_ledger",
    "validate_compact_stagewise_ledger",
]

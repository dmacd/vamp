"""Authenticated, resumable final-checkpoint bounded-addressing experiments."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile
import time

import jax
import jax.numpy as jnp
import numpy as np

from apm.continual.language_adaptation_artifact import LanguageAdaptationArtifact
from apm.continual.language_tasks import RouterBatch
from apm.data.text.tinyworlds_nouns_v1.evaluation import (
    PrefixOnlyQuery,
    _nll_by_node_per_window,
    _node_path,
    _pad_router_batch,
    _stack_token_batches,
    _story_windows,
    build_prefix_only_query,
)
from apm.data.text.tinyworlds_nouns_v1.experiment import (
    IndexedStoryStore,
    NounSelectedBase,
    StoryIndexEntry,
    load_story_index,
)
from apm.data.text.tinyworlds_nouns_v2.addressing_study_contracts import (
    ADDRESSING_STUDY_ID,
    COMPACT_PARITY_TOLERANCE,
    COMPACT_WIDTHS,
    EBT_CONTRACT_FORMAT,
    EBT_ENTROPY_PENALTY,
    EBT_LEARNING_RATE,
    EBT_ROW_COUNT,
    EBT_ROW_FORMAT,
    EBT_STEPS,
    EBT_TEMPERATURE,
    HOPFIELD_BETA,
    KEY_SCHEMES,
    MICROBATCH_SIZE,
    RETRIEVAL_CONTRACT_FORMAT,
    RETRIEVAL_ROW_COUNT,
    RETRIEVAL_ROW_FORMAT,
    TIMING_ROW_FORMAT,
    WARM_TIMING_REPETITIONS,
    EbtStudyRow,
    KeyScheme,
    RetrievalStudyRow,
    canonical_json_bytes,
    contract_record,
    load_contract,
    record_sha256,
    validate_jsonl_rows,
)
from apm.data.text.tinyworlds_nouns_v2.addressing_study_keys import (
    AddressingKeyArtifact,
    build_or_load_addressing_keys,
    encode_midpoint_content_and_residual,
    score_key_scheme,
    stable_hopfield_result,
    stack_prefix_only_queries,
)
from apm.data.text.tinyworlds_nouns_v2.contracts import (
    BASELINE_STAGEWISE_FORMAT,
    CONDITIONS,
    FULL_FINETUNE_STAGEWISE_FORMAT,
    HALF_STORY_FORMAT,
    PROBE_STORY_COUNT,
    RUN_MANIFEST_FORMAT,
    STAGEWISE_FORMAT,
    WHOLE_STORY_FORMAT,
    NounsV2ExperimentPreset,
    NounsV2PartitionArtifact,
)
from apm.data.text.tinyworlds_nouns_v2.experiment import (
    load_nouns_v2_gpu_preflight,
    load_nouns_v2_selected_base,
    load_nouns_v2_vamp_stages,
)
from apm.data.text.tinyworlds_nouns_v2.partition import (
    find_partition,
    load_manifest,
)
from apm.data.text.tinyworlds_nouns_v2.stagewise import (
    validate_stagewise_ledger,
)
from apm.lm.checkpoint import load_gpt_neo_checkpoint
from apm.lm.compact_lora_memory import (
    CompactLoraMemory,
    expand_compact_edge_coefficients,
    gather_compact_lora_memory,
)
from apm.lm.lora_memory import PackedLoraMemory, pack_lora_memory
from apm.lm.parameters import GptNeoParams
from apm.lm.text_data import TokenBatch
from apm.memory.address_refinement import (
    CompactEbtAddressResult,
    EbtAddressResult,
    EbtConfig,
    refine_compact_ebt_address,
    refine_ebt_address,
)
from apm.memory.content_addressing import HopfieldAddressResult


ProgressCallback = Callable[[str, int, int], None]
TimingCallback = Callable[[dict[str, object]], None]


@dataclass(frozen=True, slots=True)
class AddressingStudyInputs:
    """Strict-loaded canonical inputs and their pre-evaluation hash snapshot."""

    partition: NounsV2PartitionArtifact
    preset: NounsV2ExperimentPreset
    selected_base: NounSelectedBase
    adaptation: LanguageAdaptationArtifact
    base_params: GptNeoParams
    canonical_hashes: tuple[tuple[str, str], ...]
    canonical_run: dict[str, object]
    canonical_ledger_hashes: tuple[tuple[str, str], ...]
    validation_entries: tuple[tuple[str, tuple[StoryIndexEntry, ...]], ...]


@dataclass(frozen=True, slots=True)
class StudyCase:
    """One validation story with prefix-only query and evaluator-only suffix."""

    task_id: str
    entry: StoryIndexEntry
    oracle_index: int
    query: PrefixOnlyQuery
    suffix_windows: TokenBatch


@dataclass(frozen=True, slots=True)
class _MethodResult:
    scheme: KeyScheme
    mode: str
    candidate_width: int
    candidate_node_indices: np.ndarray
    selected_node_indices: np.ndarray
    gathered_edge_counts: np.ndarray
    physical_edge_capacity: int
    final_probabilities: np.ndarray
    hopfield: HopfieldAddressResult


def authenticate_addressing_study_inputs(
    repository_root: str | Path,
) -> AddressingStudyInputs:
    """Authenticate the final VAMP checkpoint, base, probes, and canonical ledgers."""
    root = Path(repository_root).resolve()
    data_root = root / "data/tinyworlds-nouns-v2"
    checkpoint_root = root / "checkpoints/tinyworlds-nouns-v2"
    result_root = root / "results/language_cl/tinyworlds-nouns-v2"
    manifest = load_manifest(data_root / "manifest.json")
    partition = find_partition(manifest, data_root)
    if partition is None:
        raise FileNotFoundError("canonical nouns-v2 partition is not published")
    preset = NounsV2ExperimentPreset()
    selected_paths = tuple(checkpoint_root.glob("base/*/selected.json"))
    if len(selected_paths) != 1:
        raise ValueError("addressing study requires exactly one selected nouns-v2 base")
    selected_record = _canonical_object(selected_paths[0], "selected base")
    selected_core = {
        key: value for key, value in selected_record.items() if key != "selection_sha256"
    }
    if (
        selected_record.get("selection_sha256") != record_sha256(selected_core)
        or selected_paths[0].parent.name != selected_record.get("training_sha256")
    ):
        raise ValueError("selected nouns-v2 base identity changed")
    preflight_matches = tuple(
        path
        for path in (checkpoint_root / "preflight").glob("*.json")
        if _canonical_object(path, "GPU preflight").get("preflight_sha256")
        == selected_record.get("preflight_sha256")
    )
    if len(preflight_matches) != 1:
        raise ValueError("selected base has no unique authenticated GPU preflight")
    preflight = load_nouns_v2_gpu_preflight(
        partition,
        preset,
        preflight_matches[0],
    )
    if preflight.artifact_path != preflight_matches[0]:
        raise ValueError("selected base and computed GPU preflight paths differ")
    selected_base = load_nouns_v2_selected_base(
        partition,
        preset,
        preflight,
        selected_paths[0].parent,
    )
    if selected_base.directory != selected_paths[0].parent:
        raise ValueError("strict-loaded selected base changed directory")
    adaptations = load_nouns_v2_vamp_stages(
        partition,
        preset,
        selected_base,
        checkpoint_root,
    )
    adaptation = adaptations[-1]
    canonical_run = _load_canonical_run_manifest(result_root / "run-manifest.json")
    if (
        canonical_run.get("partition_sha256") != partition.partition_sha256
        or canonical_run.get("config_sha256") != preset.config_sha256
        or canonical_run.get("vamp_tensor_checksum") != adaptation.tensor_checksum
    ):
        raise ValueError("canonical run manifest changed its study bindings")
    validation_entries = tuple(
        (
            task_id,
            load_story_index(partition, f"task-{task_id}-generation"),
        )
        for task_id in partition.task_ids
    )
    expected_story_keys = {
        (task_id, entry.story_id)
        for task_id, entries in validation_entries
        for entry in entries
    }
    if len(expected_story_keys) != 4_440:
        raise ValueError("canonical addressing validation coverage changed")
    _validate_canonical_result_ledger(
        result_root / "whole-story-nll.jsonl",
        WHOLE_STORY_FORMAT,
        {
            (task_id, story_id, condition)
            for task_id, story_id in expected_story_keys
            for condition in CONDITIONS
        },
        ("task_noun", "story_id", "condition"),
    )
    _validate_canonical_result_ledger(
        result_root / "half-story-generations.jsonl",
        HALF_STORY_FORMAT,
        expected_story_keys,
        ("task_noun", "story_id"),
    )
    validate_stagewise_ledger(
        result_root / "stagewise-cl.jsonl",
        partition,
        adaptations,
        require_complete=True,
    )
    ledger_paths = (
        result_root / "whole-story-nll.jsonl",
        result_root / "half-story-generations.jsonl",
        result_root / "stagewise-cl.jsonl",
        result_root / "baseline-stagewise-cl.jsonl",
        result_root / "full-finetune-stagewise-cl.jsonl",
    )
    for path, expected_format, expected_count in (
        (ledger_paths[3], BASELINE_STAGEWISE_FORMAT, 72_256),
        (ledger_paths[4], FULL_FINETUNE_STAGEWISE_FORMAT, 72_256),
    ):
        _validate_self_hashed_ledger(path, expected_format, expected_count)
    ledger_hashes = tuple(
        (path.relative_to(root).as_posix(), _file_sha256(path)) for path in ledger_paths
    )
    manifest_ledger_hashes = {
        "stagewise-cl.jsonl": canonical_run.get("vamp_stagewise_sha256"),
        "baseline-stagewise-cl.jsonl": canonical_run.get("baseline_stagewise_sha256"),
        "full-finetune-stagewise-cl.jsonl": canonical_run.get(
            "full_finetune_stagewise_sha256"
        ),
    }
    if any(
        dict(ledger_hashes)[(result_root / name).relative_to(root).as_posix()] != digest
        for name, digest in manifest_ledger_hashes.items()
    ):
        raise ValueError("canonical run manifest ledger hashes changed")
    _validate_probe_bindings(partition, expected_story_keys)
    canonical_hashes = canonical_artifact_hashes(root)
    loaded = load_gpt_neo_checkpoint(selected_base.reference)
    if loaded.reference.parameter_checksum != selected_base.reference.parameter_checksum:
        raise ValueError("selected-base parameter checksum changed on final load")
    return AddressingStudyInputs(
        partition=partition,
        preset=preset,
        selected_base=selected_base,
        adaptation=adaptation,
        base_params=loaded.params,
        canonical_hashes=canonical_hashes,
        canonical_run=canonical_run,
        canonical_ledger_hashes=ledger_hashes,
        validation_entries=validation_entries,
    )


def canonical_artifact_hashes(
    repository_root: str | Path,
) -> tuple[tuple[str, str], ...]:
    """Hash the protected nouns-v1/v2 checkpoints, ledgers, reports, and manifests."""
    root = Path(repository_root).resolve()
    paths = _protected_paths(root)
    if any(not path.is_file() for path in paths):
        missing = tuple(path for path in paths if not path.is_file())
        raise FileNotFoundError(f"protected canonical artifacts are missing: {missing}")
    return tuple(
        (path.relative_to(root).as_posix(), _file_sha256(path))
        for path in sorted(paths)
    )


def build_study_contracts(
    inputs: AddressingStudyInputs,
    keys: AddressingKeyArtifact,
    output_directory: str | Path,
) -> tuple[dict[str, object], dict[str, object]]:
    """Publish immutable retrieval and EBT contracts bound to canonical inputs."""
    output = Path(output_directory)
    graph_record = tuple(
        {
            "node_id": str(node.node_id),
            "parent_id": None if node.parent_id is None else str(node.parent_id),
        }
        for node in inputs.adaptation.vamp_graph.nodes
    )
    common_bindings = {
        "addressing_key_artifact_sha256": keys.artifact_sha256,
        "base_parameter_checksum": inputs.selected_base.reference.parameter_checksum,
        "base_training_sha256": inputs.selected_base.training_sha256,
        "canonical_artifact_hashes": dict(inputs.canonical_hashes),
        "canonical_ledger_hashes": dict(inputs.canonical_ledger_hashes),
        "canonical_run_sha256": inputs.canonical_run["run_sha256"],
        "graph_sha256": record_sha256(graph_record),
        "partition_sha256": inputs.partition.partition_sha256,
        "preset_sha256": inputs.preset.config_sha256,
        "probe_set_sha256": record_sha256(
            [list(row) for row in keys.probe_story_ids]
        ),
        "validation_set_sha256": record_sha256(
            [
                [task_id, entry.story_id]
                for task_id, entries in inputs.validation_entries
                for entry in entries
            ]
        ),
        "vamp_tensor_checksum": inputs.adaptation.tensor_checksum,
    }
    retrieval = contract_record(
        RETRIEVAL_CONTRACT_FORMAT,
        {
            "beta": HOPFIELD_BETA,
            "bindings": common_bindings,
            "expected_row_count": RETRIEVAL_ROW_COUNT,
            "key_schemes": list(KEY_SCHEMES),
            "prototype_reduction": "maximum_cosine_over_36",
            "schema_version": 1,
            "study_id": ADDRESSING_STUDY_ID,
            "top_k_widths": [1, 4, 8],
        },
    )
    ebt = contract_record(
        EBT_CONTRACT_FORMAT,
        {
            "bindings": common_bindings,
            "compact_edge_capacity_buckets": [4, 8, 12, 16, 20, 24],
            "compact_widths": list(COMPACT_WIDTHS),
            "dense_control": "canonical_full_centroid",
            "ebt": {
                "entropy_penalty": EBT_ENTROPY_PENALTY,
                "learning_rate": EBT_LEARNING_RATE,
                "steps": EBT_STEPS,
                "temperature": EBT_TEMPERATURE,
            },
            "expected_row_count": EBT_ROW_COUNT,
            "microbatch_size": MICROBATCH_SIZE,
            "prefix_width_bucket": 32,
            "retrieval_contract_sha256": retrieval["contract_sha256"],
            "schema_version": 1,
            "study_id": ADDRESSING_STUDY_ID,
            "warm_timing_repetitions": WARM_TIMING_REPETITIONS,
        },
    )
    retrieval_path = output / "retrieval-contract.json"
    ebt_path = output / "ebt-contract.json"
    if retrieval_path.is_file() or ebt_path.is_file():
        if not retrieval_path.is_file() or not ebt_path.is_file():
            raise ValueError("addressing-study contracts must exist as a pair")
        historical_retrieval = load_contract(
            retrieval_path,
            RETRIEVAL_CONTRACT_FORMAT,
        )
        historical_ebt = load_contract(ebt_path, EBT_CONTRACT_FORMAT)
        _validate_authorized_canonical_report_extension(
            historical_retrieval,
            historical_ebt,
            retrieval,
            ebt,
        )
        return historical_retrieval, historical_ebt
    _publish_immutable_json(retrieval_path, retrieval)
    _publish_immutable_json(ebt_path, ebt)
    return retrieval, ebt


def _validate_authorized_canonical_report_extension(
    historical_retrieval: dict[str, object],
    historical_ebt: dict[str, object],
    current_retrieval: dict[str, object],
    current_ebt: dict[str, object],
) -> None:
    """Allow only the later derived nouns-v2 report/run-manifest extension."""
    historical_bindings = _canonical_bindings(
        historical_retrieval,
        "historical retrieval",
    )
    current_bindings = _canonical_bindings(current_retrieval, "current retrieval")
    mutable_paths = {
        "results/language_cl/tinyworlds-nouns-v2/report.md",
        "results/language_cl/tinyworlds-nouns-v2/report.html",
        "results/language_cl/tinyworlds-nouns-v2/run-manifest.json",
    }
    historical_hashes = _required_object(
        historical_bindings["canonical_artifact_hashes"],
        "historical canonical hashes",
    )
    current_hashes = _required_object(
        current_bindings["canonical_artifact_hashes"],
        "current canonical hashes",
    )
    if (
        {
            key: value
            for key, value in historical_hashes.items()
            if key not in mutable_paths
        }
        != {
            key: value
            for key, value in current_hashes.items()
            if key not in mutable_paths
        }
        or set(historical_hashes) != set(current_hashes)
    ):
        raise ValueError("addressing-study canonical source artifacts changed")
    historicalized_bindings = {
        **current_bindings,
        "canonical_artifact_hashes": historical_hashes,
        "canonical_run_sha256": historical_bindings["canonical_run_sha256"],
    }
    retrieval_core = {
        key: value
        for key, value in current_retrieval.items()
        if key not in ("contract_sha256", "format")
    }
    expected_historical_retrieval = contract_record(
        RETRIEVAL_CONTRACT_FORMAT,
        {**retrieval_core, "bindings": historicalized_bindings},
    )
    ebt_core = {
        key: value
        for key, value in current_ebt.items()
        if key not in ("contract_sha256", "format")
    }
    expected_historical_ebt = contract_record(
        EBT_CONTRACT_FORMAT,
        {
            **ebt_core,
            "bindings": historicalized_bindings,
            "retrieval_contract_sha256": historical_retrieval[
                "contract_sha256"
            ],
        },
    )
    if (
        historical_retrieval != expected_historical_retrieval
        or historical_ebt != expected_historical_ebt
    ):
        raise ValueError("addressing-study immutable contracts changed")


def _canonical_bindings(
    contract: dict[str, object],
    label: str,
) -> dict[str, object]:
    bindings = contract.get("bindings")
    if type(bindings) is not dict:
        raise TypeError(f"{label} bindings must be an object")
    return bindings


def _required_object(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise TypeError(f"{label} must be an object")
    return value


def verify_compact_real_parity(
    inputs: AddressingStudyInputs,
    keys: AddressingKeyArtifact,
) -> dict[str, object]:
    """Require dense-mask/compact parity on a real final-checkpoint microbatch."""
    cases = next(iter_study_case_batches(inputs))
    prefix = _padded_prefix_batch(cases)
    actual_count = len(cases)
    content, residual = encode_midpoint_content_and_residual(
        inputs.base_params,
        inputs.adaptation.model_config,
        prefix,
        microbatch_size=MICROBATCH_SIZE,
    )
    hopfield = stable_hopfield_result(
        score_key_scheme(content, residual, keys, "canonical_full_centroid")
    )
    packed = pack_lora_memory(
        inputs.adaptation.vamp_graph,
        inputs.adaptation.model_config,
        inputs.adaptation.lora_config,
        inputs.adaptation.max_nodes,
        inputs.adaptation.max_edges,
    )
    config = EbtConfig(
        steps=EBT_STEPS,
        learning_rate=EBT_LEARNING_RATE,
        tau=EBT_TEMPERATURE,
        entropy_penalty=EBT_ENTROPY_PENALTY,
        initialization="hopfield_top_k",
    )
    maximum_differences: dict[str, float] = {}
    for width in COMPACT_WIDTHS:
        candidates = np.asarray(hopfield.top_k_indices, dtype=np.int32)[:, :width]
        compact = gather_compact_lora_memory(packed, candidates)
        compact_result = refine_compact_ebt_address(
            inputs.base_params,
            inputs.adaptation.model_config,
            compact,
            inputs.adaptation.lora_config,
            prefix,
            hopfield,
            config,
        )
        dense_result = refine_ebt_address(
            inputs.base_params,
            inputs.adaptation.model_config,
            packed,
            inputs.adaptation.lora_config,
            prefix,
            config,
            hopfield_result=HopfieldAddressResult(
                *hopfield[:-1],
                top_k_indices=jnp.asarray(candidates, dtype=jnp.int32),
            ),
        )
        dense_candidate_probabilities = np.take_along_axis(
            np.asarray(dense_result.node_probabilities),
            candidates,
            axis=1,
        )
        expanded_edges = expand_compact_edge_coefficients(
            compact,
            compact_result.compact_edge_coefficients,
            packed.valid_edge_mask.shape[0],
        )
        differences = {
            "candidate_probabilities": _maximum_absolute_difference(
                np.asarray(compact_result.candidate_probabilities)[:actual_count],
                dense_candidate_probabilities[:actual_count],
            ),
            "edge_coefficients": _maximum_absolute_difference(
                np.asarray(expanded_edges)[:actual_count],
                np.asarray(dense_result.edge_coefficients)[:actual_count],
            ),
            "hard_nll": _maximum_absolute_difference(
                np.asarray(compact_result.hard_node_nll)[:actual_count],
                np.asarray(dense_result.hard_node_nll)[:actual_count],
            ),
            "objective_trace": _maximum_absolute_difference(
                np.asarray(compact_result.objective_trace)[:, :actual_count],
                np.asarray(dense_result.objective_trace)[:, :actual_count],
            ),
            "soft_nll": _maximum_absolute_difference(
                np.asarray(compact_result.soft_mixture_nll)[:actual_count],
                np.asarray(dense_result.soft_mixture_nll)[:actual_count],
            ),
        }
        compact_selected = np.asarray(compact_result.selected_node_indices)[:actual_count]
        dense_selected = np.asarray(dense_result.selected_indices)[:actual_count]
        if not np.array_equal(compact_selected, dense_selected):
            raise RuntimeError(f"compact top-{width} selected nodes differ from dense mask")
        if any(value > COMPACT_PARITY_TOLERANCE for value in differences.values()):
            raise RuntimeError(
                f"compact top-{width} exceeds real parity tolerance: {differences}"
            )
        maximum_differences.update(
            {f"top_{width}_{name}": value for name, value in differences.items()}
        )
    jax.clear_caches()
    return {
        "actual_rows": actual_count,
        "maximum_absolute_differences": maximum_differences,
        "tolerance": COMPACT_PARITY_TOLERANCE,
    }


def run_or_resume_addressing_evaluation(
    inputs: AddressingStudyInputs,
    keys: AddressingKeyArtifact,
    retrieval_contract: dict[str, object],
    ebt_contract: dict[str, object],
    output_directory: str | Path,
    work_directory: str | Path,
    *,
    progress: ProgressCallback | None = None,
) -> tuple[Path, Path, Path, dict[str, float]]:
    """Stream all 22,200 retrieval and 48,840 dense/compact EBT rows."""
    output = Path(output_directory)
    work = Path(work_directory)
    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    retrieval_path = output / "retrieval.jsonl"
    ebt_path = output / "ebt.jsonl"
    timing_path = output / "timing.jsonl"
    runtime_path = output / "execution-times.json"
    retrieval_work = work / "retrieval.jsonl"
    ebt_work = work / "ebt.jsonl"
    timing_work = work / "timing.jsonl"
    retrieval_sha = str(retrieval_contract["contract_sha256"])
    ebt_sha = str(ebt_contract["contract_sha256"])
    expected_retrieval_order = expected_retrieval_keys(inputs)
    expected_ebt_order = expected_ebt_keys(inputs)
    expected_retrieval = set(expected_retrieval_order)
    expected_ebt = set(expected_ebt_order)
    if retrieval_path.is_file():
        validate_retrieval_ledger(
            retrieval_path,
            retrieval_sha,
            expected_retrieval_order,
            require_complete=True,
        )
        completed_retrieval = expected_retrieval
    else:
        repair_interrupted_tail(retrieval_work)
        completed_retrieval = validate_retrieval_ledger(
            retrieval_work,
            retrieval_sha,
            expected_retrieval_order,
            require_complete=False,
        )
    if ebt_path.is_file():
        validate_ebt_ledger(
            ebt_path,
            ebt_sha,
            expected_ebt_order,
            require_complete=True,
        )
        completed_ebt = expected_ebt
    else:
        repair_interrupted_tail(ebt_work)
        completed_ebt = validate_ebt_ledger(
            ebt_work,
            ebt_sha,
            expected_ebt_order,
            require_complete=False,
        )
    if not timing_path.is_file():
        repair_interrupted_tail(timing_work)
    timing_records = load_timing_ledger(
        timing_path if timing_path.is_file() else timing_work,
        ebt_sha,
    )
    if retrieval_path.is_file() and ebt_path.is_file() and timing_path.is_file():
        _require_timing_coverage(timing_path, ebt_path, ebt_sha)
        return (
            retrieval_path,
            ebt_path,
            timing_path,
            _load_execution_times(runtime_path, ebt_sha),
        )
    seen_timing_shapes = {
        _timing_shape(record) for record in timing_records
    }
    packed = pack_lora_memory(
        inputs.adaptation.vamp_graph,
        inputs.adaptation.model_config,
        inputs.adaptation.lora_config,
        inputs.adaptation.max_nodes,
        inputs.adaptation.max_edges,
    )
    config = EbtConfig(
        steps=EBT_STEPS,
        learning_rate=EBT_LEARNING_RATE,
        tau=EBT_TEMPERATURE,
        entropy_penalty=EBT_ENTROPY_PENALTY,
        initialization="hopfield_top_k",
    )
    started = time.perf_counter()
    phase_seconds = {"retrieval": 0.0, "ebt": 0.0, "suffix": 0.0}
    retrieval_stream = None if retrieval_path.is_file() else retrieval_work.open("ab")
    ebt_stream = None if ebt_path.is_file() else ebt_work.open("ab")
    timing_stream = None if timing_path.is_file() else timing_work.open("ab")
    try:
        for cases in iter_study_case_batches(inputs):
            prefix = _padded_prefix_batch(cases)
            actual_count = len(cases)
            retrieval_started = time.perf_counter()
            content, residual = encode_midpoint_content_and_residual(
                inputs.base_params,
                inputs.adaptation.model_config,
                prefix,
                microbatch_size=MICROBATCH_SIZE,
            )
            hopfield_by_scheme = {
                scheme: stable_hopfield_result(
                    score_key_scheme(content, residual, keys, scheme)
                )
                for scheme in KEY_SCHEMES
            }
            retrieval_payloads = tuple(
                canonical_json_bytes(
                    RetrievalStudyRow(
                        retrieval_contract_sha256=retrieval_sha,
                        scheme=scheme,
                        task_noun=case.task_id,
                        story_id=case.entry.story_id,
                        oracle_node_index=case.oracle_index,
                        top_8_indices=tuple(
                            int(value)
                            for value in np.asarray(
                                hopfield_by_scheme[scheme].top_k_indices
                            )[row, :8]
                        ),
                        entropy=float(
                            np.asarray(hopfield_by_scheme[scheme].entropy)[row]
                        ),
                        score_margin=float(
                            np.asarray(hopfield_by_scheme[scheme].score_margin)[row]
                        ),
                        prefix_token_count=int(
                            np.sum(case.query.router_batch.loss_mask)
                        ),
                    ).as_record()
                )
                for row, case in enumerate(cases)
                for scheme in KEY_SCHEMES
                if (case.task_id, case.entry.story_id, scheme)
                not in completed_retrieval
            )
            if retrieval_payloads:
                assert retrieval_stream is not None
                _append_payloads(retrieval_stream, retrieval_payloads)
                completed_retrieval.update(
                    (
                        case.task_id,
                        case.entry.story_id,
                        scheme,
                    )
                    for case in cases
                    for scheme in KEY_SCHEMES
                )
            phase_seconds["retrieval"] += time.perf_counter() - retrieval_started
            method_results: list[_MethodResult] = []
            ebt_started = time.perf_counter()
            method_specs = (
                (scheme, "compact", width)
                for scheme in KEY_SCHEMES
                for width in COMPACT_WIDTHS
            )
            for scheme, mode, width in (
                *method_specs,
                ("canonical_full_centroid", "dense_all", 25),
            ):
                pending = tuple(
                    (
                        case.task_id,
                        case.entry.story_id,
                        scheme,
                        mode,
                        width,
                    )
                    not in completed_ebt
                    for case in cases
                )
                if not any(pending):
                    continue
                hopfield = hopfield_by_scheme[scheme]
                if mode == "compact":
                    candidates = np.asarray(
                        hopfield.top_k_indices,
                        dtype=np.int32,
                    )[:, :width]
                    compact = gather_compact_lora_memory(packed, candidates)
                    call = lambda: refine_compact_ebt_address(
                        inputs.base_params,
                        inputs.adaptation.model_config,
                        compact,
                        inputs.adaptation.lora_config,
                        prefix,
                        hopfield,
                        config,
                    )
                    result, timing_record = _execute_and_maybe_time(
                        call,
                        ebt_sha,
                        mode,
                        width,
                        prefix.input_ids.shape[1],
                        compact.valid_edge_mask.shape[1],
                        seen_timing_shapes,
                    )
                    assert isinstance(result, CompactEbtAddressResult)
                    selected = np.asarray(result.selected_node_indices, dtype=np.int32)
                    final_probabilities = np.asarray(
                        result.candidate_probabilities,
                        dtype=np.float32,
                    )
                    gathered_counts = np.sum(
                        np.asarray(compact.valid_edge_mask, dtype=np.int32),
                        axis=1,
                    )
                    physical_capacity = compact.valid_edge_mask.shape[1]
                else:
                    candidates = np.broadcast_to(
                        np.arange(25, dtype=np.int32)[None, :],
                        (prefix.input_ids.shape[0], 25),
                    )
                    dense_config = replace(config, initialization="hopfield")
                    call = lambda: refine_ebt_address(
                        inputs.base_params,
                        inputs.adaptation.model_config,
                        packed,
                        inputs.adaptation.lora_config,
                        prefix,
                        dense_config,
                        hopfield_result=hopfield,
                    )
                    result, timing_record = _execute_and_maybe_time(
                        call,
                        ebt_sha,
                        mode,
                        width,
                        prefix.input_ids.shape[1],
                        packed.valid_edge_mask.shape[0],
                        seen_timing_shapes,
                    )
                    assert isinstance(result, EbtAddressResult)
                    selected = np.asarray(result.selected_indices, dtype=np.int32)
                    final_probabilities = np.asarray(
                        result.node_probabilities,
                        dtype=np.float32,
                    )
                    gathered_counts = np.full(
                        prefix.input_ids.shape[0],
                        int(np.sum(np.asarray(packed.valid_edge_mask))),
                        dtype=np.int32,
                    )
                    physical_capacity = packed.valid_edge_mask.shape[0]
                if timing_record is not None:
                    assert timing_stream is not None
                    _append_payloads(
                        timing_stream,
                        (canonical_json_bytes(timing_record),),
                    )
                    timing_records += (timing_record,)
                method_results.append(
                    _MethodResult(
                        scheme=scheme,
                        mode=mode,
                        candidate_width=width,
                        candidate_node_indices=np.asarray(candidates, dtype=np.int32),
                        selected_node_indices=selected,
                        gathered_edge_counts=gathered_counts,
                        physical_edge_capacity=physical_capacity,
                        final_probabilities=final_probabilities,
                        hopfield=hopfield,
                    )
                )
            phase_seconds["ebt"] += time.perf_counter() - ebt_started
            if method_results:
                suffix_started = time.perf_counter()
                payloads, completed_keys = _ebt_payloads_for_batch(
                    cases,
                    method_results,
                    inputs,
                    packed,
                    ebt_sha,
                    completed_ebt,
                )
                if payloads:
                    assert ebt_stream is not None
                    _append_payloads(ebt_stream, payloads)
                    completed_ebt.update(completed_keys)
                phase_seconds["suffix"] += time.perf_counter() - suffix_started
            completed_cases = len(completed_ebt) // 11
            if progress is not None:
                progress("addressing-evaluation", completed_cases, 4_440)
    finally:
        for stream in (retrieval_stream, ebt_stream, timing_stream):
            if stream is not None:
                stream.close()
    if completed_retrieval != expected_retrieval:
        raise RuntimeError(
            f"retrieval ledger has {len(completed_retrieval):,} of "
            f"{len(expected_retrieval):,} rows"
        )
    if completed_ebt != expected_ebt:
        raise RuntimeError(
            f"EBT ledger has {len(completed_ebt):,} of {len(expected_ebt):,} rows"
        )
    if not retrieval_path.is_file():
        os.replace(retrieval_work, retrieval_path)
    if not ebt_path.is_file():
        os.replace(ebt_work, ebt_path)
    if not timing_path.is_file():
        os.replace(timing_work, timing_path)
    validate_retrieval_ledger(
        retrieval_path,
        retrieval_sha,
        expected_retrieval_order,
        require_complete=True,
    )
    validate_ebt_ledger(
        ebt_path,
        ebt_sha,
        expected_ebt_order,
        require_complete=True,
    )
    _require_timing_coverage(timing_path, ebt_path, ebt_sha)
    measured_runtimes = {
        **phase_seconds,
        "end_to_end": time.perf_counter() - started,
    }
    runtime_core = {
        "ebt_contract_sha256": ebt_sha,
        "format": "tinyworlds-nouns-v2-addressing-execution-times-v1",
        "runtimes_seconds": measured_runtimes,
    }
    _publish_immutable_json(
        runtime_path,
        {**runtime_core, "result_sha256": record_sha256(runtime_core)},
    )
    return (
        retrieval_path,
        ebt_path,
        timing_path,
        measured_runtimes,
    )


def iter_study_case_batches(
    inputs: AddressingStudyInputs,
) -> Iterator[tuple[StudyCase, ...]]:
    """Yield canonical validation cases in fixed eight-row microbatches."""
    store = IndexedStoryStore(inputs.partition)
    pending: list[StudyCase] = []
    for task_id, entries in inputs.validation_entries:
        oracle_index = inputs.partition.task_ids.index(task_id) + 1
        for entry in entries:
            tokens = store.tokens(entry)
            midpoint = len(tokens) // 2
            pending.append(
                StudyCase(
                    task_id=task_id,
                    entry=entry,
                    oracle_index=oracle_index,
                    query=build_prefix_only_query(
                        entry.story_id,
                        tokens,
                        inputs.partition.pad_token_id,
                        inputs.adaptation.model_config.max_position_embeddings,
                    ),
                    suffix_windows=_story_windows(
                        tokens,
                        inputs.preset.context_length,
                        inputs.partition.pad_token_id,
                        first_target_index=midpoint,
                    ),
                )
            )
            if len(pending) == MICROBATCH_SIZE:
                yield tuple(pending)
                pending = []
    if pending:
        yield tuple(pending)


def expected_method_specs() -> tuple[tuple[KeyScheme, str, int], ...]:
    """Return the ten compact methods followed by the canonical dense control."""
    return tuple(
        (scheme, "compact", width)
        for scheme in KEY_SCHEMES
        for width in COMPACT_WIDTHS
    ) + (("canonical_full_centroid", "dense_all", 25),)


def expected_retrieval_keys(
    inputs: AddressingStudyInputs,
) -> tuple[tuple[object, ...], ...]:
    """Return the exact ordered 22,200 task/story/scheme retrieval keys."""
    expected = tuple(
        (task_id, entry.story_id, scheme)
        for task_id, entries in inputs.validation_entries
        for entry in entries
        for scheme in KEY_SCHEMES
    )
    if len(expected) != RETRIEVAL_ROW_COUNT or len(set(expected)) != len(expected):
        raise ValueError("addressing retrieval row coverage changed")
    return expected


def expected_ebt_keys(
    inputs: AddressingStudyInputs,
) -> tuple[tuple[object, ...], ...]:
    """Return the exact ordered 48,840 task/story/scheme/mode/width EBT keys."""
    expected = tuple(
        (task_id, entry.story_id, scheme, mode, width)
        for task_id, entries in inputs.validation_entries
        for entry in entries
        for scheme, mode, width in expected_method_specs()
    )
    if len(expected) != EBT_ROW_COUNT or len(set(expected)) != len(expected):
        raise ValueError("addressing EBT row coverage changed")
    return expected


def validate_retrieval_ledger(
    path: str | Path,
    contract_sha256: str,
    expected_keys: tuple[tuple[object, ...], ...],
    *,
    require_complete: bool,
) -> set[tuple[object, ...]]:
    """Strictly validate retrieval identities, fields, metrics, and coverage."""
    completed = validate_jsonl_rows(
        path,
        expected_format=RETRIEVAL_ROW_FORMAT,
        contract_field="retrieval_contract_sha256",
        contract_sha256=contract_sha256,
        key_fields=("task_noun", "story_id", "scheme"),
        expected_keys=set(expected_keys),
        require_complete=require_complete,
    )
    source = Path(path)
    observed_order: list[tuple[object, ...]] = []
    if source.is_file():
        for row in _canonical_jsonl_objects(source):
            observed_order.append(
                (row.get("task_noun"), row.get("story_id"), row.get("scheme"))
            )
            top_8 = row.get("top_8_indices")
            oracle = row.get("oracle_node_index")
            expected_fields = set(
                RetrievalStudyRow(
                    retrieval_contract_sha256=contract_sha256,
                    scheme="canonical_full_centroid",
                    task_noun="placeholder",
                    story_id="0" * 64,
                    oracle_node_index=1,
                    top_8_indices=tuple(range(8)),
                    entropy=0.0,
                    score_margin=0.0,
                    prefix_token_count=1,
                ).as_record()
            )
            if (
                set(row) != expected_fields
                or type(top_8) is not list
                or len(top_8) != 8
                or len(set(top_8)) != 8
                or any(type(value) is not int or not 0 <= value < 25 for value in top_8)
                or type(oracle) is not int
                or not 1 <= oracle < 25
                or row.get("top_1_hit") != (top_8[0] == oracle)
                or row.get("top_4_hit") != (oracle in top_8[:4])
                or row.get("top_8_hit") != (oracle in top_8)
                or type(row.get("prefix_token_count")) is not int
                or int(row["prefix_token_count"]) <= 0
                or not _finite_nonnegative(row.get("entropy"))
                or not _finite_nonnegative(row.get("score_margin"))
            ):
                raise ValueError("retrieval ledger row semantics changed")
    if tuple(observed_order) != expected_keys[: len(observed_order)]:
        raise ValueError("retrieval ledger is not a canonical resumable prefix")
    return completed


def validate_ebt_ledger(
    path: str | Path,
    contract_sha256: str,
    expected_keys: tuple[tuple[object, ...], ...],
    *,
    require_complete: bool,
) -> set[tuple[object, ...]]:
    """Strictly validate EBT identities, paths, operation counts, and coverage."""
    completed = validate_jsonl_rows(
        path,
        expected_format=EBT_ROW_FORMAT,
        contract_field="ebt_contract_sha256",
        contract_sha256=contract_sha256,
        key_fields=("task_noun", "story_id", "scheme", "mode", "candidate_width"),
        expected_keys=set(expected_keys),
        require_complete=require_complete,
    )
    source = Path(path)
    observed_order: list[tuple[object, ...]] = []
    if source.is_file():
        for row in _canonical_jsonl_objects(source):
            observed_order.append(
                (
                    row.get("task_noun"),
                    row.get("story_id"),
                    row.get("scheme"),
                    row.get("mode"),
                    row.get("candidate_width"),
                )
            )
            try:
                reconstructed = EbtStudyRow(
                    ebt_contract_sha256=row.get("ebt_contract_sha256"),
                    scheme=row.get("scheme"),
                    mode=row.get("mode"),
                    candidate_width=row.get("candidate_width"),
                    task_noun=row.get("task_noun"),
                    story_id=row.get("story_id"),
                    oracle_node_index=row.get("oracle_node_index"),
                    candidate_node_indices=tuple(
                        row.get("candidate_node_indices", ())
                    ),
                    selected_node_index=row.get("selected_node_index"),
                    selected_path=tuple(row.get("selected_path", ())),
                    gathered_edge_count=row.get("gathered_edge_count"),
                    selected_path_edge_count=row.get("selected_path_edge_count"),
                    physical_edge_capacity=row.get("physical_edge_capacity"),
                    prefix_token_count=row.get("prefix_token_count"),
                    prefix_width_bucket=row.get("prefix_width_bucket"),
                    suffix_total_nll=row.get("suffix_total_nll"),
                    suffix_token_count=row.get("suffix_token_count"),
                    suffix_mean_nll=row.get("suffix_mean_nll"),
                    oracle_suffix_mean_nll=row.get("oracle_suffix_mean_nll"),
                    retrieval_entropy=row.get("retrieval_entropy"),
                    retrieval_margin=row.get("retrieval_margin"),
                    final_entropy=row.get("final_entropy"),
                    final_margin=row.get("final_margin"),
                ).as_record()
            except (TypeError, ValueError):
                reconstructed = None
            if row != reconstructed:
                raise ValueError("EBT ledger row semantics changed")
    if tuple(observed_order) != expected_keys[: len(observed_order)]:
        raise ValueError("EBT ledger is not a canonical resumable prefix")
    return completed


def load_timing_ledger(
    path: str | Path,
    ebt_contract_sha256: str,
) -> tuple[dict[str, object], ...]:
    """Strict-load unique synchronized cold/warm timing rows when present."""
    source = Path(path)
    if not source.is_file():
        return ()
    records = tuple(_canonical_jsonl_objects(source))
    shapes: set[tuple[object, ...]] = set()
    expected_fields = {
        "batch_size",
        "candidate_width",
        "cold_compile_seconds",
        "ebt_contract_sha256",
        "format",
        "mode",
        "physical_edge_capacity",
        "prefix_width_bucket",
        "result_sha256",
        "warm_kernel_mean_seconds",
        "warm_kernel_seconds",
        "warm_throughput_examples_per_second",
    }
    for record in records:
        supplied = record.get("result_sha256")
        core = {key: value for key, value in record.items() if key != "result_sha256"}
        samples = record.get("warm_kernel_seconds")
        mode = record.get("mode")
        candidate_width = record.get("candidate_width")
        physical_capacity = record.get("physical_edge_capacity")
        prefix_width = record.get("prefix_width_bucket")
        warm_mean = record.get("warm_kernel_mean_seconds")
        warm_throughput = record.get("warm_throughput_examples_per_second")
        if (
            set(record) != expected_fields
            or record.get("format") != TIMING_ROW_FORMAT
            or record.get("ebt_contract_sha256") != ebt_contract_sha256
            or supplied != record_sha256(core)
            or record.get("batch_size") != MICROBATCH_SIZE
            or type(candidate_width) is not int
            or type(physical_capacity) is not int
            or type(prefix_width) is not int
            or prefix_width <= 0
            or prefix_width % 32 != 0
            or (
                mode == "dense_all"
                and (candidate_width != 25 or physical_capacity != 24)
            )
            or (
                mode == "compact"
                and (
                    candidate_width not in COMPACT_WIDTHS
                    or physical_capacity not in (4, 8, 12, 16, 20, 24)
                )
            )
            or mode not in ("dense_all", "compact")
            or type(samples) is not list
            or len(samples) != WARM_TIMING_REPETITIONS
            or any(not _finite_positive(value) for value in samples)
            or not _finite_positive(record.get("cold_compile_seconds"))
            or not math.isclose(
                float(warm_mean) if _finite_positive(warm_mean) else math.nan,
                math.fsum(float(value) for value in samples) / len(samples),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(warm_throughput)
                if _finite_positive(warm_throughput)
                else math.nan,
                MICROBATCH_SIZE / float(warm_mean)
                if _finite_positive(warm_mean)
                else math.nan,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("addressing timing row identity or samples changed")
        shape = _timing_shape(record)
        if shape in shapes:
            raise ValueError("addressing timing ledger contains a duplicate shape")
        shapes.add(shape)
    return records


def repair_interrupted_tail(path: str | Path) -> None:
    """Truncate only an incomplete final JSONL row and preserve prior bytes."""
    target = Path(path)
    if not target.is_file() or target.stat().st_size == 0:
        return
    with target.open("r+b") as stream:
        stream.seek(-1, os.SEEK_END)
        if stream.read(1) == b"\n":
            return
        position = stream.tell() - 1
        while position > 0:
            start = max(0, position - 64 * 1024)
            stream.seek(start)
            payload = stream.read(position - start)
            newline = payload.rfind(b"\n")
            if newline >= 0:
                stream.truncate(start + newline + 1)
                stream.flush()
                os.fsync(stream.fileno())
                return
            position = start
        stream.truncate(0)
        stream.flush()
        os.fsync(stream.fileno())


def _padded_prefix_batch(cases: tuple[StudyCase, ...]) -> RouterBatch:
    stacked = stack_prefix_only_queries(
        tuple(case.query.router_batch for case in cases),
        2_048,
    )
    return _pad_router_batch(stacked, MICROBATCH_SIZE)


def _execute_and_maybe_time(
    call: Callable[[], CompactEbtAddressResult | EbtAddressResult],
    ebt_contract_sha256: str,
    mode: str,
    candidate_width: int,
    prefix_width: int,
    physical_edge_capacity: int,
    seen_shapes: set[tuple[object, ...]],
) -> tuple[CompactEbtAddressResult | EbtAddressResult, dict[str, object] | None]:
    shape = (mode, candidate_width, prefix_width, physical_edge_capacity)
    if shape in seen_shapes:
        result = call()
        _block_until_ready(result)
        return result, None
    cold_started = time.perf_counter()
    result = call()
    _block_until_ready(result)
    cold_seconds = time.perf_counter() - cold_started
    samples = []
    for _ in range(WARM_TIMING_REPETITIONS):
        warm_started = time.perf_counter()
        _block_until_ready(call())
        samples.append(time.perf_counter() - warm_started)
    warm_mean = math.fsum(samples) / len(samples)
    core = {
        "batch_size": MICROBATCH_SIZE,
        "candidate_width": candidate_width,
        "cold_compile_seconds": cold_seconds,
        "ebt_contract_sha256": ebt_contract_sha256,
        "format": TIMING_ROW_FORMAT,
        "mode": mode,
        "physical_edge_capacity": physical_edge_capacity,
        "prefix_width_bucket": prefix_width,
        "warm_kernel_mean_seconds": warm_mean,
        "warm_kernel_seconds": samples,
        "warm_throughput_examples_per_second": MICROBATCH_SIZE / warm_mean,
    }
    seen_shapes.add(shape)
    return result, {**core, "result_sha256": record_sha256(core)}


def _ebt_payloads_for_batch(
    cases: tuple[StudyCase, ...],
    methods: list[_MethodResult],
    inputs: AddressingStudyInputs,
    packed: PackedLoraMemory,
    ebt_contract_sha256: str,
    already_completed: set[tuple[object, ...]],
) -> tuple[tuple[bytes, ...], tuple[tuple[object, ...], ...]]:
    method_map = {
        (method.scheme, method.mode, method.candidate_width): method
        for method in methods
    }
    node_indices = tuple(
        sorted(
            {
                case.oracle_index for case in cases
            }
            | {
                int(method.selected_node_indices[row])
                for method in methods
                for row in range(len(cases))
            }
        )
    )
    suffix_windows = _stack_token_batches(
        tuple(case.suffix_windows for case in cases)
    )
    per_window = _nll_by_node_per_window(
        inputs.base_params,
        inputs.adaptation,
        packed,
        suffix_windows,
        MICROBATCH_SIZE,
        node_indices=node_indices,
    )
    node_row = {node_index: row for row, node_index in enumerate(node_indices)}
    boundaries = np.cumsum(
        (0,) + tuple(case.suffix_windows.input_ids.shape[0] for case in cases)
    )
    totals = tuple(
        {
            node_index: float(
                np.sum(
                    per_window[node_row[node_index], start:stop],
                    dtype=np.float64,
                )
            )
            for node_index in node_indices
        }
        for start, stop in zip(boundaries[:-1], boundaries[1:])
    )
    payloads: list[bytes] = []
    keys: list[tuple[object, ...]] = []
    for row, case in enumerate(cases):
        suffix_token_count = int(np.sum(case.suffix_windows.loss_mask))
        oracle_mean = totals[row][case.oracle_index] / suffix_token_count
        for specification in expected_method_specs():
            method = method_map.get(specification)
            key = (
                case.task_id,
                case.entry.story_id,
                *specification,
            )
            if key in already_completed:
                continue
            if method is None:
                raise RuntimeError("pending EBT row has no computed method result")
            selected_index = int(method.selected_node_indices[row])
            selected_total = totals[row][selected_index]
            probabilities = method.final_probabilities[row]
            positive_probabilities = probabilities[probabilities > 0.0]
            final_entropy = float(
                -np.sum(
                    positive_probabilities
                    * np.log(positive_probabilities),
                    dtype=np.float64,
                )
            )
            descending = np.sort(probabilities)[::-1]
            final_margin = float(descending[0] - descending[1])
            path = _node_path(inputs.adaptation, selected_index)
            record = EbtStudyRow(
                ebt_contract_sha256=ebt_contract_sha256,
                scheme=method.scheme,
                mode=method.mode,
                candidate_width=method.candidate_width,
                task_noun=case.task_id,
                story_id=case.entry.story_id,
                oracle_node_index=case.oracle_index,
                candidate_node_indices=tuple(
                    int(value) for value in method.candidate_node_indices[row]
                ),
                selected_node_index=selected_index,
                selected_path=path,
                gathered_edge_count=int(method.gathered_edge_counts[row]),
                selected_path_edge_count=len(path) - 1,
                physical_edge_capacity=method.physical_edge_capacity,
                prefix_token_count=int(np.sum(case.query.router_batch.loss_mask)),
                prefix_width_bucket=prefix_width_for_case_batch(cases),
                suffix_total_nll=selected_total,
                suffix_token_count=suffix_token_count,
                suffix_mean_nll=selected_total / suffix_token_count,
                oracle_suffix_mean_nll=oracle_mean,
                retrieval_entropy=float(np.asarray(method.hopfield.entropy)[row]),
                retrieval_margin=float(np.asarray(method.hopfield.score_margin)[row]),
                final_entropy=final_entropy,
                final_margin=final_margin,
            ).as_record()
            payloads.append(canonical_json_bytes(record))
            keys.append(key)
    return tuple(payloads), tuple(keys)


def prefix_width_for_case_batch(cases: tuple[StudyCase, ...]) -> int:
    """Return the 32-token padded prefix width used by one study microbatch."""
    maximum = max(case.query.router_batch.input_ids.shape[1] for case in cases)
    return ((maximum + 31) // 32) * 32


def _require_timing_coverage(
    timing_path: Path,
    ebt_path: Path,
    ebt_contract_sha256: str,
) -> None:
    timing_shapes = tuple(
        _timing_shape(record)
        for record in load_timing_ledger(timing_path, ebt_contract_sha256)
    )
    expected_shapes = tuple(
        dict.fromkeys(
            (
                row["mode"],
                row["candidate_width"],
                row["prefix_width_bucket"],
                row["physical_edge_capacity"],
            )
            for row in _canonical_jsonl_objects(ebt_path)
        )
    )
    if timing_shapes != expected_shapes:
        raise ValueError(
            f"timing coverage differs: {len(timing_shapes)} measured versus "
            f"{len(expected_shapes)} observed shapes"
        )


def _timing_shape(record: dict[str, object]) -> tuple[object, ...]:
    return (
        record.get("mode"),
        record.get("candidate_width"),
        record.get("prefix_width_bucket"),
        record.get("physical_edge_capacity"),
    )


def _block_until_ready(value: object) -> None:
    for leaf in jax.tree_util.tree_leaves(value):
        method = getattr(leaf, "block_until_ready", None)
        if callable(method):
            method()


def _append_payloads(stream, payloads: tuple[bytes, ...]) -> None:
    stream.write(b"".join(payloads))
    stream.flush()
    os.fsync(stream.fileno())


def enforce_nouns_v2_allocator_gate(
    preset: NounsV2ExperimentPreset,
) -> dict[str, object]:
    """Require a GPU backend and keep the observed allocator peak below 12 GiB."""
    devices = jax.local_devices()
    if not devices or any(device.platform != "gpu" for device in devices):
        raise RuntimeError("addressing study requires GPU-backed JAX on GPU 0")
    statistics = tuple(device.memory_stats() or {} for device in devices)
    peak = max(
        (
            int(values.get("peak_bytes_in_use", values.get("bytes_in_use", 0)))
            for values in statistics
        ),
        default=0,
    )
    if peak <= 0:
        raise RuntimeError("GPU backend did not expose allocator peak statistics")
    if peak > preset.allocator_peak_limit_bytes:
        raise RuntimeError(
            f"addressing study measured {peak} allocator bytes, above the frozen "
            f"{preset.allocator_peak_limit_bytes}-byte gate"
        )
    return {
        "allocator_limit_bytes": preset.allocator_peak_limit_bytes,
        "device_kind": tuple(str(device.device_kind) for device in devices),
        "device_platform": "gpu",
        "peak_bytes_in_use": peak,
    }


def assert_canonical_hashes_unchanged(
    repository_root: str | Path,
    before: tuple[tuple[str, str], ...],
) -> None:
    """Reject any checkpoint, ledger, report, or manifest mutation by the study."""
    after = canonical_artifact_hashes(repository_root)
    if after != before:
        changed = tuple(
            path
            for path in sorted(set(dict(before)) | set(dict(after)))
            if dict(before).get(path) != dict(after).get(path)
        )
        raise RuntimeError(f"canonical nouns-v1/v2 artifacts changed: {changed}")


def _validate_probe_bindings(
    partition: NounsV2PartitionArtifact,
    validation_keys: set[tuple[str, str]],
) -> None:
    validation_ids = {story_id for _, story_id in validation_keys}
    root_entries = load_story_index(partition, "root-probes")
    root_ids = tuple(entry.story_id for entry in root_entries)
    if (
        len(root_ids) != PROBE_STORY_COUNT
        or len(set(root_ids)) != PROBE_STORY_COUNT
        or set(root_ids) != set(partition.root_probe_story_ids)
    ):
        raise ValueError("root probe index differs from the partition contract")
    task_entries = tuple(
        load_story_index(partition, f"task-{task.task_id}-probes")
        for task in partition.tasks
    )
    if any(
        len(entries) != PROBE_STORY_COUNT
        or len({entry.story_id for entry in entries}) != PROBE_STORY_COUNT
        or {entry.story_id for entry in entries} != set(task.probe_story_ids)
        for task, entries in zip(partition.tasks, task_entries)
    ):
        raise ValueError("task probe index differs from the partition contract")
    probe_ids = {
        entry.story_id
        for entries in (root_entries, *task_entries)
        for entry in entries
    }
    if len(probe_ids) != 25 * 36 or probe_ids & validation_ids:
        raise ValueError("probe identities overlap each other or official validation")


def _load_canonical_run_manifest(path: Path) -> dict[str, object]:
    record = _canonical_object(path, "canonical run manifest")
    supplied = record.get("run_sha256")
    core = {key: value for key, value in record.items() if key != "run_sha256"}
    if (
        record.get("format") != RUN_MANIFEST_FORMAT
        or record.get("phase") not in ("local_complete", "complete_with_judge")
        or supplied != record_sha256(core)
    ):
        raise ValueError("canonical nouns-v2 run manifest identity changed")
    return record


def _load_execution_times(path: Path, ebt_contract_sha256: str) -> dict[str, float]:
    record = _canonical_object(path, "addressing execution times")
    supplied = record.get("result_sha256")
    core = {key: value for key, value in record.items() if key != "result_sha256"}
    raw_runtimes = record.get("runtimes_seconds")
    if (
        set(record)
        != {
            "ebt_contract_sha256",
            "format",
            "result_sha256",
            "runtimes_seconds",
        }
        or record.get("format")
        != "tinyworlds-nouns-v2-addressing-execution-times-v1"
        or record.get("ebt_contract_sha256") != ebt_contract_sha256
        or supplied != record_sha256(core)
        or type(raw_runtimes) is not dict
        or set(raw_runtimes) != {"retrieval", "ebt", "suffix", "end_to_end"}
        or any(not _finite_nonnegative(value) for value in raw_runtimes.values())
        or math.fsum(
            float(raw_runtimes[name]) for name in ("retrieval", "ebt", "suffix")
        )
        > float(raw_runtimes["end_to_end"]) + 1e-6
    ):
        raise ValueError("addressing execution-time identity changed")
    return {str(key): float(value) for key, value in raw_runtimes.items()}


def _validate_canonical_result_ledger(
    path: Path,
    expected_format: str,
    expected_keys: set[tuple[str, ...]],
    key_fields: tuple[str, ...],
) -> None:
    keys: set[tuple[str, ...]] = set()
    for record in _canonical_jsonl_objects(path):
        supplied = record.get("result_sha256")
        core = {key: value for key, value in record.items() if key != "result_sha256"}
        key = tuple(str(record.get(field)) for field in key_fields)
        if (
            record.get("format") != expected_format
            or supplied != record_sha256(core)
            or key in keys
            or key not in expected_keys
        ):
            raise ValueError(f"canonical ledger row changed: {path.name}")
        keys.add(key)
    if keys != expected_keys:
        raise ValueError(f"canonical ledger coverage changed: {path.name}")


def _validate_self_hashed_ledger(
    path: Path,
    expected_format: str,
    expected_count: int,
) -> None:
    count = 0
    seen_hashes: set[str] = set()
    for record in _canonical_jsonl_objects(path):
        supplied = record.get("result_sha256")
        core = {key: value for key, value in record.items() if key != "result_sha256"}
        if (
            record.get("format") != expected_format
            or supplied != record_sha256(core)
            or supplied in seen_hashes
        ):
            raise ValueError(f"canonical ledger identity changed: {path.name}")
        seen_hashes.add(str(supplied))
        count += 1
    if count != expected_count:
        raise ValueError(f"canonical ledger row count changed: {path.name}")


def _protected_paths(root: Path) -> tuple[Path, ...]:
    nouns_v2_result = root / "results/language_cl/tinyworlds-nouns-v2"
    nouns_v1_result = root / "results/language_cl/tinyworlds-nouns-v1"
    v2_base = _sole_path(
        root.glob("checkpoints/tinyworlds-nouns-v2/base/*/selected.json"),
        "nouns-v2 selected base",
    ).parent
    v1_base = _sole_path(
        root.glob("checkpoints/tinyworlds-nouns-v1/base/*/selected.json"),
        "nouns-v1 selected base",
    ).parent
    v2_final = sorted(
        root.glob("checkpoints/tinyworlds-nouns-v2/vamp/*/stage-024-*/stage.json")
    )
    v1_final = sorted(
        root.glob("checkpoints/tinyworlds-nouns-v1/vamp/*/stage-042-*/stage.json")
    )
    final_records = (
        _sole_path(iter(v2_final), "nouns-v2 final VAMP stage"),
        _sole_path(iter(v1_final), "nouns-v1 final VAMP stage"),
    )
    checkpoint_files = tuple(
        path
        for base in (v1_base, v2_base)
        for path in (
            base / "selected.json",
            base / "checkpoint/manifest.json",
            base / "checkpoint/model.safetensors",
        )
    ) + tuple(
        path
        for stage in final_records
        for path in (
            stage,
            stage.parent / "adaptation/manifest.json",
            stage.parent / "adaptation/adaptation.safetensors",
        )
    )
    result_files = tuple(
        nouns_v2_result / name
        for name in (
            "whole-story-nll.jsonl",
            "half-story-generations.jsonl",
            "stagewise-cl.jsonl",
            "baseline-stagewise-cl.jsonl",
            "full-finetune-stagewise-cl.jsonl",
            "report.md",
            "report.html",
            "run-manifest.json",
        )
    ) + tuple(
        nouns_v1_result / name
        for name in (
            "whole-story-nll.jsonl",
            "half-story-generations.jsonl",
            "report.md",
            "report.html",
            "run-manifest.json",
        )
    )
    data_files = (
        root / "data/tinyworlds-nouns-v2/manifest.json",
        _sole_path(
            root.glob("data/tinyworlds-nouns-v2/partitions/*/partition.json"),
            "nouns-v2 partition",
        ),
        _sole_path(
            root.glob("data/tinyworlds-nouns-v1/partitions/*/partition.json"),
            "nouns-v1 partition",
        ),
    )
    return checkpoint_files + result_files + data_files


def _sole_path(paths, label: str) -> Path:
    values = tuple(paths)
    if len(values) != 1:
        raise ValueError(f"{label} requires exactly one path, found {len(values)}")
    return values[0]


def _canonical_object(path: Path, label: str) -> dict[str, object]:
    payload = path.read_bytes()
    value = json.loads(payload)
    if type(value) is not dict or canonical_json_bytes(value) != payload:
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _canonical_jsonl_objects(path: Path) -> Iterator[dict[str, object]]:
    """Yield canonical JSONL objects without retaining a sequential ledger."""
    with path.open("rb") as source:
        for line in source:
            value = json.loads(line)
            if (
                not line.endswith(b"\n")
                or type(value) is not dict
                or canonical_json_bytes(value) != line
            ):
                raise ValueError(f"ledger is not canonical JSONL: {path}")
            yield value


def _publish_immutable_json(path: Path, record: dict[str, object]) -> None:
    payload = canonical_json_bytes(record)
    if path.is_file():
        if path.read_bytes() != payload:
            raise ValueError(f"published addressing-study contract changed: {path.name}")
        return
    _atomic_write(path, payload)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _maximum_absolute_difference(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError("parity arrays must have identical shapes")
    return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))


def _finite_nonnegative(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value)) and float(value) >= 0.0


def _finite_positive(value: object) -> bool:
    return _finite_nonnegative(value) and float(value) > 0.0


__all__ = [
    "AddressingStudyInputs",
    "StudyCase",
    "assert_canonical_hashes_unchanged",
    "authenticate_addressing_study_inputs",
    "build_study_contracts",
    "canonical_artifact_hashes",
    "enforce_nouns_v2_allocator_gate",
    "expected_ebt_keys",
    "expected_method_specs",
    "expected_retrieval_keys",
    "iter_study_case_batches",
    "load_timing_ledger",
    "repair_interrupted_tail",
    "run_or_resume_addressing_evaluation",
    "validate_ebt_ledger",
    "validate_retrieval_ledger",
    "verify_compact_real_parity",
]

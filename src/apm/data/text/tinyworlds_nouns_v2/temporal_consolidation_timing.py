"""Synchronized cold/warm timing audit for temporal addressing shapes."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from time import monotonic
from typing import TYPE_CHECKING

import jax
import numpy as np

from apm.continual.language_adaptation_artifact import (
    read_safetensors_archive,
    write_safetensors_archive,
)
from apm.continual.language_tasks import RouterBatch
from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_contracts import (
    ALLOCATOR_LIMIT_BYTES,
    STUDY_ID,
    TIMING_ROW_FORMAT,
    WARM_TIMING_REPETITIONS,
    empty_hierarchy,
    insert_arrival,
    temporal_arrivals,
)
from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_evaluation import (
    AdapterBank,
    AdapterCandidate,
    MidpointCase,
    build_adapter_bank,
    prefix_router_row_capacity,
    prepare_prefix_kernel_batch,
    prepare_suffix_kernel_batch,
    run_packed_prefix_kernel,
    run_packed_suffix_kernel,
    run_prefix_kernel,
    run_suffix_kernel,
)
from apm.continual.artifacts import (
    ChainedJsonlLedger,
    file_sha256,
    load_canonical_json,
    publish_immutable_json,
)
from apm.data.text.tinyworlds_nouns_v2.contracts import record_sha256
from apm.lm.checkpoint import load_gpt_neo_checkpoint
from apm.lm.lora import LoraBlockBank, LoraEdgeBank, LoraProjectionBank
from apm.lm.lora_memory import PackedLoraMemory
from apm.lm.text_data import TokenBatch

if TYPE_CHECKING:
    from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation import (
        OrderingArtifacts,
        TemporalStudyInputs,
    )


TimingProgress = Callable[[int, int, Mapping[str, float]], None]
_TIMING_BUNDLE_FORMAT = f"{STUDY_ID}-timing-worker-inputs-v1"
_TIMING_BUNDLE_DIRECTORY = "timing-worker-inputs-v1"
_TIMING_BUNDLE_FILE = "inputs.safetensors"
_TIMING_BUNDLE_MANIFEST = "manifest.json"
_WORKER_MODULE = (
    "apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_timing_worker"
)
_BATCH_FIELDS = ("input_ids", "attention_mask", "target_ids", "loss_mask")


def expected_timing_shapes(
    inputs: TemporalStudyInputs,
) -> tuple[tuple[str, int, int | None], ...]:
    """Return all observed candidate-capacity/prefix-width timing shapes."""
    candidate_counts: set[int] = set()
    for order in ("blocked", "round_robin"):
        state = empty_hierarchy(order)
        for shard in temporal_arrivals(inputs.shards, order):
            state, _ = insert_arrival(state, shard)
            candidate_counts.add(len(state.active_chunks))
    prefix_buckets = {
        32 * math.ceil(((entry.token_count // 2) - 1) / 32)
        for _, entries in inputs.validation_entries
        for entry in entries
    }
    return tuple(
        [("prefix", count, width) for count in sorted(candidate_counts) for width in sorted(prefix_buckets)]
        + [("suffix", count, None) for count in sorted(candidate_counts)]
    )


def run_or_resume_timing_audit(
    inputs: TemporalStudyInputs,
    orderings: Sequence[OrderingArtifacts],
    cases_by_task: Mapping[str, Sequence[MidpointCase]],
    *,
    progress: TimingProgress | None = None,
) -> Path:
    """Time one cold call and five synchronized warm calls for every shape."""
    ledger = ChainedJsonlLedger(
        inputs.work_directory / "timing.jsonl",
        TIMING_ROW_FORMAT,
    )
    shapes = expected_timing_shapes(inputs)
    observed = validate_timing_rows(ledger.rows, inputs.contract_sha256)
    if observed != shapes[: len(observed)]:
        raise ValueError("temporal timing ledger is not a canonical shape prefix")
    banks = _representative_banks(inputs, orderings)
    cases_by_bucket = {
        bucket: tuple(
            case
            for task_id in cases_by_task
            for case in cases_by_task[task_id]
            if case.prefix_width_bucket == bucket
        )[:8]
        for _, _, bucket in shapes
        if bucket is not None
    }
    representative_suffix = next(
        case for task_id in cases_by_task for case in cases_by_task[task_id]
    )
    completed = set(observed)
    for kind, candidate_count, prefix_width in shapes:
        shape = (kind, candidate_count, prefix_width)
        if shape in completed:
            continue
        # Each row explicitly measures one cold compile followed by five warm
        # executions. Keeping prior shape executables cannot make this shape
        # warm, but it does retain several GiB across the 208-shape sweep.
        # Isolating the cache therefore preserves the measurement contract and
        # keeps the audit below the fixed allocator bound.
        jax.clear_caches()
        bank = banks[candidate_count]
        if kind == "prefix":
            batch = prepare_prefix_kernel_batch(
                cases_by_bucket[int(prefix_width)],
                row_count=prefix_router_row_capacity(
                    candidate_count,
                    int(prefix_width),
                ),
            )
            operation = lambda: run_prefix_kernel(
                inputs.loaded_base.params,
                inputs.loaded_base.config,
                bank,
                batch,
            )
            active_tokens = int(np.sum(batch.loss_mask))
            physical_rows = batch.input_ids.shape[0]
            sequence_width = batch.input_ids.shape[1]
            candidate_evaluations = physical_rows * len(bank.candidate_ids)
            forward_equivalent_tokens = active_tokens * len(bank.candidate_ids)
        else:
            batch = prepare_suffix_kernel_batch(representative_suffix)
            operation = lambda: run_suffix_kernel(
                inputs.loaded_base.params,
                inputs.loaded_base.config,
                bank,
                batch,
            )
            active_tokens = int(np.sum(batch.loss_mask))
            physical_rows = batch.input_ids.shape[0]
            sequence_width = batch.input_ids.shape[1]
            candidate_evaluations = physical_rows
            forward_equivalent_tokens = active_tokens
        cold = _synchronized_seconds(operation)
        warm = tuple(
            _synchronized_seconds(operation) for _ in range(WARM_TIMING_REPETITIONS)
        )
        ledger.append(
            {
                "active_tokens": active_tokens,
                "candidate_count": candidate_count,
                "candidate_evaluations": candidate_evaluations,
                "cold_seconds": cold,
                "contract_sha256": inputs.contract_sha256,
                "forward_equivalent_tokens": forward_equivalent_tokens,
                "kind": kind,
                "node_count_including_base": len(bank.candidate_ids),
                "physical_rows": physical_rows,
                "prefix_width": prefix_width,
                "sequence_width": sequence_width,
                "warm_max_seconds": max(warm),
                "warm_mean_seconds": math.fsum(warm) / len(warm),
                "warm_min_seconds": min(warm),
                "warm_repetitions": list(warm),
            }
        )
        completed.add(shape)
        if progress is not None:
            progress(
                len(completed),
                len(shapes),
                {
                    "cold_seconds": cold,
                    "warm_mean_seconds": math.fsum(warm) / len(warm),
                },
            )
    final = validate_timing_rows(ledger.rows, inputs.contract_sha256)
    if final != shapes:
        raise RuntimeError(f"timing audit has {len(final)} of {len(shapes)} shapes")
    return ledger.path


def run_or_resume_isolated_timing_audit(
    inputs: TemporalStudyInputs,
    orderings: Sequence[OrderingArtifacts],
    cases_by_task: Mapping[str, Sequence[MidpointCase]],
    *,
    progress: TimingProgress | None = None,
) -> Path:
    """Run each cold/warm shape in a fresh GPU allocator process.

    JAX's default allocator retains freed blocks even after compiled executables
    are evicted.  A 208-shape audit can therefore exhaust the device despite
    every individual shape fitting comfortably.  The immutable worker bundle
    avoids re-authentication while a fresh process per shape preserves the
    default allocator and releases it completely at the shape boundary.
    """
    ledger = ChainedJsonlLedger(
        inputs.work_directory / "timing.jsonl",
        TIMING_ROW_FORMAT,
    )
    shapes = expected_timing_shapes(inputs)
    observed = validate_timing_rows(ledger.rows, inputs.contract_sha256)
    if observed != shapes[: len(observed)]:
        raise ValueError("temporal timing ledger is not a canonical shape prefix")
    if observed == shapes:
        return ledger.path
    bundle_directory = _prepare_timing_worker_bundle(
        inputs,
        orderings,
        cases_by_task,
        shapes,
    )
    for shape_index in range(len(observed), len(shapes)):
        _invoke_timing_worker(inputs, bundle_directory, ledger.path, shape_index)
        ledger = ChainedJsonlLedger(ledger.path, TIMING_ROW_FORMAT)
        completed = validate_timing_rows(ledger.rows, inputs.contract_sha256)
        if completed != shapes[: shape_index + 1]:
            raise RuntimeError("isolated timing worker did not append its exact shape")
        if progress is not None:
            row = ledger.rows[-1]
            progress(
                len(completed),
                len(shapes),
                {
                    "allocator_peak_bytes": float(row["allocator_peak_bytes"]),
                    "cold_seconds": float(row["cold_seconds"]),
                    "warm_mean_seconds": float(row["warm_mean_seconds"]),
                },
            )
    return ledger.path


def timing_worker_main() -> int:
    """Execute exactly one environment-selected timing shape."""
    bundle_directory = Path(_required_environment("RPA_TEMPORAL_TIMING_BUNDLE"))
    ledger_path = Path(_required_environment("RPA_TEMPORAL_TIMING_LEDGER"))
    contract_sha256 = _required_environment("RPA_TEMPORAL_TIMING_CONTRACT")
    try:
        shape_index = int(_required_environment("RPA_TEMPORAL_TIMING_SHAPE_INDEX"))
    except ValueError as error:
        raise ValueError("timing worker shape index is not an integer") from error
    manifest, arrays = _load_timing_worker_bundle(
        bundle_directory,
        expected_contract_sha256=contract_sha256,
    )
    shapes = _manifest_shapes(manifest)
    if not 0 <= shape_index < len(shapes):
        raise IndexError("timing worker shape index is outside the manifest")
    ledger = ChainedJsonlLedger(ledger_path, TIMING_ROW_FORMAT)
    observed = validate_timing_rows(ledger.rows, contract_sha256)
    if observed != shapes[: len(observed)] or len(observed) != shape_index:
        raise ValueError("timing worker ledger is not at its assigned boundary")

    base_directory = Path(str(manifest["base_checkpoint_directory"]))
    loaded_base = load_gpt_neo_checkpoint(base_directory)
    if (
        loaded_base.reference.manifest_sha256
        != manifest["base_manifest_sha256"]
        or loaded_base.reference.parameter_checksum
        != manifest["base_parameter_checksum"]
    ):
        raise ValueError("timing worker base checkpoint identity changed")
    kind, candidate_count, prefix_width = shapes[shape_index]
    packed = _packed_bank_from_bundle(arrays, candidate_count, loaded_base.config.num_layers)
    if kind == "prefix":
        batch = _prefix_batch_from_bundle(
            arrays,
            int(prefix_width),
            prefix_router_row_capacity(candidate_count, int(prefix_width)),
        )
        operation = lambda: run_packed_prefix_kernel(
            loaded_base.params,
            loaded_base.config,
            packed,
            batch,
        )
        active_tokens = int(np.sum(batch.loss_mask))
        physical_rows = int(batch.input_ids.shape[0])
        sequence_width = int(batch.input_ids.shape[1])
        node_count = candidate_count + 1
        candidate_evaluations = physical_rows * node_count
        forward_equivalent_tokens = active_tokens * node_count
    else:
        batch = _suffix_batch_from_bundle(arrays)
        operation = lambda: run_packed_suffix_kernel(
            loaded_base.params,
            loaded_base.config,
            packed,
            batch,
        )
        active_tokens = int(np.sum(batch.loss_mask))
        physical_rows = int(batch.input_ids.shape[0])
        sequence_width = int(batch.input_ids.shape[1])
        node_count = candidate_count + 1
        candidate_evaluations = physical_rows
        forward_equivalent_tokens = active_tokens

    cold = _synchronized_seconds(operation)
    warm = tuple(
        _synchronized_seconds(operation) for _ in range(WARM_TIMING_REPETITIONS)
    )
    peak = _worker_allocator_peak_bytes()
    if peak <= 0 or peak > ALLOCATOR_LIMIT_BYTES:
        raise RuntimeError(
            f"timing worker allocator peak {peak} violates the fixed "
            f"{ALLOCATOR_LIMIT_BYTES}-byte gate"
        )
    ledger.append(
        {
            "active_tokens": active_tokens,
            "allocator_peak_bytes": peak,
            "candidate_count": candidate_count,
            "candidate_evaluations": candidate_evaluations,
            "cold_seconds": cold,
            "contract_sha256": contract_sha256,
            "forward_equivalent_tokens": forward_equivalent_tokens,
            "kind": kind,
            "node_count_including_base": node_count,
            "physical_rows": physical_rows,
            "prefix_width": prefix_width,
            "sequence_width": sequence_width,
            "warm_max_seconds": max(warm),
            "warm_mean_seconds": math.fsum(warm) / len(warm),
            "warm_min_seconds": min(warm),
            "warm_repetitions": list(warm),
        }
    )
    print(
        f"Timed isolated shape {shape_index + 1}/{len(shapes)}: "
        f"{kind}, candidates={candidate_count}, prefix_width={prefix_width}",
        flush=True,
    )
    return 0


def _prepare_timing_worker_bundle(
    inputs: TemporalStudyInputs,
    orderings: Sequence[OrderingArtifacts],
    cases_by_task: Mapping[str, Sequence[MidpointCase]],
    shapes: Sequence[tuple[str, int, int | None]],
) -> Path:
    target = inputs.work_directory / _TIMING_BUNDLE_DIRECTORY
    if target.exists():
        _load_timing_worker_bundle(
            target,
            expected_contract_sha256=inputs.contract_sha256,
            expected_shapes=shapes,
            expected_base_directory=inputs.loaded_base.reference.directory,
            expected_base_manifest_sha256=inputs.loaded_base.reference.manifest_sha256,
            expected_base_parameter_checksum=inputs.loaded_base.reference.parameter_checksum,
        )
        return target

    arrays = _timing_bundle_arrays(inputs, orderings, cases_by_task, shapes)
    metadata = _timing_bundle_metadata(
        inputs.contract_sha256,
        inputs.loaded_base.reference.parameter_checksum,
        shapes,
    )
    inputs.work_directory.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=".timing-worker-inputs-", dir=inputs.work_directory)
    )
    try:
        bundle_path = temporary / _TIMING_BUNDLE_FILE
        write_safetensors_archive(bundle_path, arrays, metadata)
        core = {
            "base_checkpoint_directory": str(
                inputs.loaded_base.reference.directory.resolve()
            ),
            "base_manifest_sha256": inputs.loaded_base.reference.manifest_sha256,
            "base_parameter_checksum": inputs.loaded_base.reference.parameter_checksum,
            "bundle_file": _TIMING_BUNDLE_FILE,
            "bundle_file_sha256": file_sha256(bundle_path),
            "contract_sha256": inputs.contract_sha256,
            "format": _TIMING_BUNDLE_FORMAT,
            "shapes": [list(shape) for shape in shapes],
            "tensor_count": len(arrays),
        }
        publish_immutable_json(
            temporary / _TIMING_BUNDLE_MANIFEST,
            {**core, "result_sha256": record_sha256(core)},
        )
        os.replace(temporary, target)
        directory_descriptor = os.open(inputs.work_directory, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    _load_timing_worker_bundle(
        target,
        expected_contract_sha256=inputs.contract_sha256,
        expected_shapes=shapes,
        expected_base_directory=inputs.loaded_base.reference.directory,
        expected_base_manifest_sha256=inputs.loaded_base.reference.manifest_sha256,
        expected_base_parameter_checksum=inputs.loaded_base.reference.parameter_checksum,
    )
    return target


def _timing_bundle_arrays(
    inputs: TemporalStudyInputs,
    orderings: Sequence[OrderingArtifacts],
    cases_by_task: Mapping[str, Sequence[MidpointCase]],
    shapes: Sequence[tuple[str, int, int | None]],
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for candidate_count, bank in sorted(_representative_banks(inputs, orderings).items()):
        packed = bank.packed
        prefix = f"bank.{candidate_count}"
        arrays[f"{prefix}.node_path_matrix"] = np.asarray(packed.node_path_matrix)
        arrays[f"{prefix}.valid_node_mask"] = np.asarray(packed.valid_node_mask)
        arrays[f"{prefix}.valid_edge_mask"] = np.asarray(packed.valid_edge_mask)
        for block_index, block in enumerate(packed.edge_bank.blocks):
            for projection_name in LoraBlockBank._fields:
                projection = getattr(block, projection_name)
                arrays[
                    f"{prefix}.blocks.{block_index}.{projection_name}.left"
                ] = np.asarray(projection.left)
                arrays[
                    f"{prefix}.blocks.{block_index}.{projection_name}.right"
                ] = np.asarray(projection.right)

    prefix_widths = sorted(
        int(width)
        for kind, _, width in shapes
        if kind == "prefix" and width is not None
    )
    for width in sorted(set(prefix_widths)):
        cases = tuple(
            case
            for task_id in cases_by_task
            for case in cases_by_task[task_id]
            if case.prefix_width_bucket == width
        )[:8]
        batch = prepare_prefix_kernel_batch(cases)
        for field in _BATCH_FIELDS:
            arrays[f"prefix.{width}.{field}"] = _serializable_batch_array(
                field,
                getattr(batch, field),
            )
    representative_suffix = next(
        case for task_id in cases_by_task for case in cases_by_task[task_id]
    )
    suffix = prepare_suffix_kernel_batch(representative_suffix)
    for field in _BATCH_FIELDS:
        arrays[f"suffix.{field}"] = _serializable_batch_array(
            field,
            getattr(suffix, field),
        )
    return dict(sorted(arrays.items()))


def _serializable_batch_array(field: str, value: np.ndarray) -> np.ndarray:
    dtype = np.uint32 if field in ("input_ids", "target_ids") else np.bool_
    return np.asarray(value, dtype=dtype)


def _load_timing_worker_bundle(
    directory: Path,
    *,
    expected_contract_sha256: str,
    expected_shapes: Sequence[tuple[str, int, int | None]] | None = None,
    expected_base_directory: Path | None = None,
    expected_base_manifest_sha256: str | None = None,
    expected_base_parameter_checksum: str | None = None,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    if not directory.is_dir() or {path.name for path in directory.iterdir()} != {
        _TIMING_BUNDLE_FILE,
        _TIMING_BUNDLE_MANIFEST,
    }:
        raise ValueError("timing worker bundle entries changed")
    manifest = load_canonical_json(directory / _TIMING_BUNDLE_MANIFEST)
    core = {key: value for key, value in manifest.items() if key != "result_sha256"}
    expected_keys = {
        "base_checkpoint_directory",
        "base_manifest_sha256",
        "base_parameter_checksum",
        "bundle_file",
        "bundle_file_sha256",
        "contract_sha256",
        "format",
        "shapes",
        "tensor_count",
    }
    if (
        set(core) != expected_keys
        or manifest.get("result_sha256") != record_sha256(core)
        or manifest.get("format") != _TIMING_BUNDLE_FORMAT
        or manifest.get("contract_sha256") != expected_contract_sha256
        or manifest.get("bundle_file") != _TIMING_BUNDLE_FILE
    ):
        raise ValueError("timing worker bundle manifest changed")
    shapes = _manifest_shapes(manifest)
    if expected_shapes is not None and tuple(expected_shapes) != shapes:
        raise ValueError("timing worker bundle shapes changed")
    if (
        expected_base_directory is not None
        and Path(str(manifest["base_checkpoint_directory"])).resolve()
        != expected_base_directory.resolve()
    ):
        raise ValueError("timing worker bundle base path changed")
    for key, expected in (
        ("base_manifest_sha256", expected_base_manifest_sha256),
        ("base_parameter_checksum", expected_base_parameter_checksum),
    ):
        if expected is not None and manifest.get(key) != expected:
            raise ValueError(f"timing worker bundle {key} changed")
    bundle_path = directory / _TIMING_BUNDLE_FILE
    if manifest.get("bundle_file_sha256") != file_sha256(bundle_path):
        raise ValueError("timing worker bundle file hash changed")
    arrays, metadata = read_safetensors_archive(bundle_path)
    expected_metadata = _timing_bundle_metadata(
        expected_contract_sha256,
        str(manifest["base_parameter_checksum"]),
        shapes,
    )
    if metadata != expected_metadata or manifest.get("tensor_count") != len(arrays):
        raise ValueError("timing worker bundle tensor metadata changed")
    return manifest, arrays


def _manifest_shapes(
    manifest: Mapping[str, object],
) -> tuple[tuple[str, int, int | None], ...]:
    raw_shapes = manifest.get("shapes")
    if type(raw_shapes) is not list:
        raise ValueError("timing worker manifest shapes changed")
    shapes: list[tuple[str, int, int | None]] = []
    for raw in raw_shapes:
        if type(raw) is not list or len(raw) != 3:
            raise ValueError("timing worker manifest shape changed")
        kind, count, width = raw
        if (
            kind not in ("prefix", "suffix")
            or type(count) is not int
            or count <= 0
            or (kind == "prefix" and (type(width) is not int or width <= 0))
            or (kind == "suffix" and width is not None)
        ):
            raise ValueError("timing worker manifest shape changed")
        shapes.append((str(kind), count, None if width is None else int(width)))
    if len(set(shapes)) != len(shapes):
        raise ValueError("timing worker manifest shapes are duplicated")
    return tuple(shapes)


def _timing_bundle_metadata(
    contract_sha256: str,
    base_parameter_checksum: str,
    shapes: Sequence[tuple[str, int, int | None]],
) -> dict[str, str]:
    return {
        "base_parameter_checksum": base_parameter_checksum,
        "contract_sha256": contract_sha256,
        "format": _TIMING_BUNDLE_FORMAT,
        "shapes_sha256": record_sha256([list(shape) for shape in shapes]),
    }


def _packed_bank_from_bundle(
    arrays: Mapping[str, np.ndarray],
    candidate_count: int,
    num_layers: int,
) -> PackedLoraMemory:
    prefix = f"bank.{candidate_count}"
    blocks = tuple(
        LoraBlockBank(
            **{
                projection_name: LoraProjectionBank(
                    left=_bundle_tensor(
                        arrays,
                        f"{prefix}.blocks.{block_index}.{projection_name}.left",
                    ),
                    right=_bundle_tensor(
                        arrays,
                        f"{prefix}.blocks.{block_index}.{projection_name}.right",
                    ),
                )
                for projection_name in LoraBlockBank._fields
            }
        )
        for block_index in range(num_layers)
    )
    packed = PackedLoraMemory(
        edge_bank=LoraEdgeBank(blocks=blocks),
        node_path_matrix=_bundle_tensor(arrays, f"{prefix}.node_path_matrix"),
        valid_node_mask=_bundle_tensor(arrays, f"{prefix}.valid_node_mask"),
        valid_edge_mask=_bundle_tensor(arrays, f"{prefix}.valid_edge_mask"),
    )
    if (
        packed.node_path_matrix.shape != (candidate_count + 1, candidate_count)
        or packed.valid_node_mask.shape != (candidate_count + 1,)
        or packed.valid_edge_mask.shape != (candidate_count,)
    ):
        raise ValueError("timing worker packed-bank capacity changed")
    return packed


def _prefix_batch_from_bundle(
    arrays: Mapping[str, np.ndarray],
    prefix_width: int,
    row_count: int,
) -> RouterBatch:
    batch = RouterBatch(
        *(
            _bundle_tensor(arrays, f"prefix.{prefix_width}.{field}")[:row_count]
            for field in _BATCH_FIELDS
        )
    )
    if batch.input_ids.shape != (row_count, prefix_width):
        raise ValueError("timing worker prefix batch shape changed")
    return batch


def _suffix_batch_from_bundle(arrays: Mapping[str, np.ndarray]) -> TokenBatch:
    batch = TokenBatch(
        *(_bundle_tensor(arrays, f"suffix.{field}") for field in _BATCH_FIELDS)
    )
    if batch.input_ids.ndim != 2 or batch.input_ids.shape[0] <= 0:
        raise ValueError("timing worker suffix batch shape changed")
    return batch


def _bundle_tensor(arrays: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    if name not in arrays:
        raise ValueError(f"timing worker bundle tensor is missing: {name}")
    return np.asarray(arrays[name])


def _invoke_timing_worker(
    inputs: TemporalStudyInputs,
    bundle_directory: Path,
    ledger_path: Path,
    shape_index: int,
) -> None:
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "RPA_TEMPORAL_TIMING_BUNDLE": str(bundle_directory.resolve()),
            "RPA_TEMPORAL_TIMING_CONTRACT": inputs.contract_sha256,
            "RPA_TEMPORAL_TIMING_LEDGER": str(ledger_path.resolve()),
            "RPA_TEMPORAL_TIMING_SHAPE_INDEX": str(shape_index),
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
    )
    environment.pop("XLA_PYTHON_CLIENT_ALLOCATOR", None)
    result = subprocess.run(
        [sys.executable, "-m", _WORKER_MODULE],
        cwd=inputs.repository_root,
        env=environment,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"isolated timing worker failed for shape index {shape_index}: "
            f"exit {result.returncode}"
        )


def _worker_allocator_peak_bytes() -> int:
    devices = tuple(jax.local_devices())
    if not devices or any(device.platform != "gpu" for device in devices):
        raise RuntimeError("isolated timing worker requires GPU-backed JAX")
    return max(
        int((device.memory_stats() or {}).get("peak_bytes_in_use", 0))
        for device in devices
    )


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"isolated timing worker requires {name}")
    return value


def validate_timing_rows(
    rows: Sequence[Mapping[str, object]],
    contract_sha256: str,
) -> tuple[tuple[str, int, int | None], ...]:
    """Validate timing bindings, repetitions, and shape identities."""
    shapes: list[tuple[str, int, int | None]] = []
    for row in rows:
        repetitions = row.get("warm_repetitions")
        prefix_width = row.get("prefix_width")
        allocator_peak = row.get("allocator_peak_bytes")
        shape = (
            str(row.get("kind")),
            int(row.get("candidate_count", -1)),
            None if prefix_width is None else int(prefix_width),
        )
        if (
            row.get("contract_sha256") != contract_sha256
            or shape[0] not in ("prefix", "suffix")
            or shape[1] <= 0
            or (shape[0] == "prefix") != (shape[2] is not None)
            or type(repetitions) is not list
            or len(repetitions) != WARM_TIMING_REPETITIONS
            or not all(_finite_nonnegative(value) for value in repetitions)
            or not _finite_nonnegative(row.get("cold_seconds"))
            or (
                allocator_peak is not None
                and (
                    type(allocator_peak) is not int
                    or not 0 < allocator_peak <= ALLOCATOR_LIMIT_BYTES
                )
            )
            or not math.isclose(
                float(row.get("warm_mean_seconds", -1.0)),
                math.fsum(float(value) for value in repetitions) / len(repetitions),
                abs_tol=1e-12,
            )
            or shape in shapes
        ):
            raise ValueError("temporal timing ledger row changed")
        shapes.append(shape)
    return tuple(shapes)


def maximum_timing_allocator_peak_bytes(
    path: str | Path,
    contract_sha256: str,
) -> int:
    """Return the largest authenticated isolated-worker allocator peak."""
    ledger = ChainedJsonlLedger(path, TIMING_ROW_FORMAT)
    validate_timing_rows(ledger.rows, contract_sha256)
    return max(
        (int(row.get("allocator_peak_bytes", 0)) for row in ledger.rows),
        default=0,
    )


def _representative_banks(
    inputs: TemporalStudyInputs,
    orderings: Sequence[OrderingArtifacts],
) -> dict[int, AdapterBank]:
    banks: dict[int, AdapterBank] = {}
    for ordering in orderings:
        artifacts = ordering.chunks_by_id
        state = empty_hierarchy(ordering.order)
        for shard in temporal_arrivals(inputs.shards, ordering.order):
            state, _ = insert_arrival(state, shard)
            count = len(state.active_chunks)
            if count in banks:
                continue
            banks[count] = build_adapter_bank(
                tuple(
                    AdapterCandidate(
                        f"interval-{chunk.start_arrival:03d}-{chunk.end_arrival:03d}",
                        artifacts[chunk.chunk_id].adapter_sha256,
                        artifacts[chunk.chunk_id].adapter,
                        chunk.task_counts,
                        chunk.level,
                        chunk.start_arrival,
                        chunk.end_arrival,
                    )
                    for chunk in state.active_chunks
                ),
                inputs.loaded_base.config,
            )
    expected_counts = {count for _, count, _ in expected_timing_shapes(inputs)}
    if set(banks) != expected_counts:
        raise RuntimeError("ordering artifacts do not cover every timing capacity")
    return banks


def _synchronized_seconds(operation: Callable[[], object]) -> float:
    started = monotonic()
    result = operation()
    for leaf in jax.tree_util.tree_leaves(result):
        block = getattr(leaf, "block_until_ready", None)
        if callable(block):
            block()
    return monotonic() - started


def _finite_nonnegative(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value)) and value >= 0


__all__ = [
    "expected_timing_shapes",
    "maximum_timing_allocator_peak_bytes",
    "run_or_resume_isolated_timing_audit",
    "run_or_resume_timing_audit",
    "timing_worker_main",
    "validate_timing_rows",
]

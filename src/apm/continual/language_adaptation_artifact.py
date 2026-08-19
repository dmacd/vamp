"""Strict, immutable persistence for trained language-adaptation state."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import tempfile
from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np

from apm.continual.language_baseline_training import (
    IndependentRootLoraRun,
    LanguageAdaptationBaselines,
    SequentialLoraRun,
)
from apm.continual.language_run import LanguageVampRun
from apm.continual.language_tasks import AddressBook
from apm.lm.checkpoint import BaseCheckpointRef
from apm.lm.config import GptNeoConfig
from apm.lm.lora import (
    LoraBlock,
    LoraConfig,
    LoraEdge,
    LoraProjection,
    LoraTargetMask,
)
from apm.lm.training import LmTrainConfig
from apm.memory.graph import MemoryGraph, MemoryNode, NodeId, TaskId

LANGUAGE_ADAPTATION_SCHEMA_VERSION = 1
LANGUAGE_ADAPTATION_FORMAT = "apm-language-adaptation"
LANGUAGE_ADAPTATION_MANIFEST = "manifest.json"
LANGUAGE_ADAPTATION_TENSORS = "adaptation.safetensors"

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_CONFIG_NAME_PATTERN = re.compile(r"[a-z][a-z0-9_.-]*\Z")
_MAX_SAFETENSORS_HEADER_BYTES = 100_000_000
_PROJECTION_NAMES = LoraBlock._fields
_NUMPY_TO_SAFETENSORS = {
    np.dtype("float64"): "F64",
    np.dtype("float32"): "F32",
    np.dtype("uint32"): "U32",
    np.dtype("bool"): "BOOL",
}
_SAFETENSORS_TO_NUMPY = {
    code: dtype.newbyteorder("<")
    for dtype, code in _NUMPY_TO_SAFETENSORS.items()
}


@dataclass(frozen=True)
class AdapterTrainingRecord:
    """One task-aligned adapter snapshot and its complete update-loss trace."""

    stage_index: int
    task_id: TaskId
    adapter: LoraEdge
    training_trace: tuple[float, ...]

    def __post_init__(self) -> None:
        if type(self.stage_index) is not int or self.stage_index <= 0:
            raise ValueError("adapter stage_index must be positive")
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError("adapter task_id must not be empty")
        object.__setattr__(self, "adapter", _immutable_lora_edge(self.adapter))
        object.__setattr__(
            self,
            "training_trace",
            _validated_training_trace(self.training_trace),
        )


@dataclass(frozen=True)
class VampTrainingRecord:
    """One committed VAMP edge's parent search and update-loss evidence."""

    stage_index: int
    task_id: TaskId
    parent_node_index: int
    parent_node_id: NodeId
    parent_mean_node_nll: tuple[float, ...]
    training_trace: tuple[float, ...]

    def __post_init__(self) -> None:
        if type(self.stage_index) is not int or self.stage_index <= 0:
            raise ValueError("VAMP stage_index must be positive")
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError("VAMP task_id must not be empty")
        if type(self.parent_node_index) is not int or self.parent_node_index < 0:
            raise ValueError("VAMP parent_node_index must be nonnegative")
        if not isinstance(self.parent_node_id, str) or not self.parent_node_id:
            raise ValueError("VAMP parent_node_id must not be empty")
        scores = tuple(float(value) for value in self.parent_mean_node_nll)
        if not scores or any(np.isnan(value) or value < 0.0 for value in scores):
            raise ValueError("VAMP parent scores must be nonnegative or infinite")
        object.__setattr__(self, "parent_mean_node_nll", scores)
        object.__setattr__(
            self,
            "training_trace",
            _validated_training_trace(self.training_trace),
        )


@dataclass(frozen=True, eq=False)
class LanguageAdaptationRngState:
    """Final explicit RNG state for all three adaptation training streams."""

    sequential_single_lora: np.ndarray
    independent_root_lora: np.ndarray
    vamp: np.ndarray

    def __post_init__(self) -> None:
        for field_name in (
            "sequential_single_lora",
            "independent_root_lora",
            "vamp",
        ):
            object.__setattr__(
                self,
                field_name,
                _immutable_rng_key(getattr(self, field_name), field_name),
            )


@dataclass(frozen=True, eq=False)
class LanguageAdaptationArtifact:
    """Complete reusable trained-adaptation state with no copied base weights."""

    base_checkpoint: BaseCheckpointRef
    model_config: GptNeoConfig
    lora_config: LoraConfig
    train_config: LmTrainConfig
    config_hashes: tuple[tuple[str, str], ...]
    task_order: tuple[TaskId, ...]
    sequential_stages: tuple[AdapterTrainingRecord, ...]
    independent_adapters: tuple[AdapterTrainingRecord, ...]
    vamp_graph: MemoryGraph[LoraEdge]
    address_book: AddressBook
    rng_state: LanguageAdaptationRngState
    vamp_stages: tuple[VampTrainingRecord, ...]
    max_nodes: int
    max_edges: int

    def __post_init__(self) -> None:
        _validate_language_adaptation_artifact(self)

    @property
    def tensor_checksum(self) -> str:
        """Return one canonical checksum over every persisted tensor value."""
        return _tensor_checksum(_artifact_tensors(self))

    @property
    def tensor_checksums(self) -> tuple[tuple[str, str], ...]:
        """Return canonical per-tensor checksums for immutability auditing."""
        return tuple(
            (name, _array_checksum(name, array))
            for name, array in _artifact_tensors(self).items()
        )


def flatten_lora_edge(
    edge: LoraEdge,
    model_config: GptNeoConfig,
    lora_config: LoraConfig,
) -> dict[str, jax.Array]:
    """Flatten one LoRA edge under stable block/projection/factor names."""
    arrays = _validated_lora_arrays(
        {
            f"blocks.{block_index}.{projection_name}.{factor_name}": factor
            for block_index, block in enumerate(edge.blocks)
            for projection_name in _PROJECTION_NAMES
            for factor_name, factor in zip(
                ("left", "right"),
                getattr(block, projection_name),
            )
        },
        model_config,
        lora_config,
    )
    return {name: jnp.asarray(array) for name, array in arrays.items()}


def unflatten_lora_edge(
    tensors: Mapping[str, jax.Array | np.ndarray],
    model_config: GptNeoConfig,
    lora_config: LoraConfig,
) -> LoraEdge:
    """Strictly reconstruct one float32 LoRA edge from canonical tensors."""
    arrays = _validated_lora_arrays(tensors, model_config, lora_config)
    return LoraEdge(
        blocks=tuple(
            LoraBlock(
                **{
                    projection_name: LoraProjection(
                        left=jnp.asarray(
                            arrays[
                                f"blocks.{block_index}.{projection_name}.left"
                            ]
                        ),
                        right=jnp.asarray(
                            arrays[
                                f"blocks.{block_index}.{projection_name}.right"
                            ]
                        ),
                    )
                    for projection_name in _PROJECTION_NAMES
                }
            )
            for block_index in range(model_config.num_layers)
        )
    )


def write_safetensors_archive(
    path: str | Path,
    tensors: Mapping[str, jax.Array | np.ndarray],
    metadata: Mapping[str, str],
) -> None:
    """Write a deterministic safetensors file using the artifact codec."""
    _write_safetensors(
        Path(path),
        {name: np.asarray(value) for name, value in tensors.items()},
        metadata,
    )


def read_safetensors_archive(
    path: str | Path,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Read and strictly validate a deterministic safetensors file."""
    return _read_safetensors(Path(path))


def extract_language_adaptation_artifact(
    adaptations: LanguageAdaptationBaselines,
    model_config: GptNeoConfig,
    lora_config: LoraConfig,
    *,
    config_hashes: Mapping[str, str] | None = None,
) -> LanguageAdaptationArtifact:
    """Extract reusable inference state and training evidence from trained baselines."""
    if not isinstance(adaptations, LanguageAdaptationBaselines):
        raise TypeError("adaptations must be LanguageAdaptationBaselines")
    if not isinstance(model_config, GptNeoConfig):
        raise TypeError("model_config must be a GptNeoConfig")
    if not isinstance(lora_config, LoraConfig):
        raise TypeError("lora_config must be a LoraConfig")
    required_hashes = _required_config_hashes(
        model_config,
        lora_config,
        adaptations.train_config,
    )
    merged_hashes = dict(config_hashes or {})
    for name, digest in required_hashes:
        if name in merged_hashes and merged_hashes[name] != digest:
            raise ValueError(f"supplied {name} config hash does not match its config")
        merged_hashes[name] = digest
    task_order = tuple(
        TaskId(str(stage.task_id))
        for stage in adaptations.sequential_single_lora.stages
    )
    return LanguageAdaptationArtifact(
        base_checkpoint=adaptations.vamp.base_checkpoint,
        model_config=model_config,
        lora_config=lora_config,
        train_config=adaptations.train_config,
        config_hashes=tuple(merged_hashes.items()),
        task_order=task_order,
        sequential_stages=tuple(
            AdapterTrainingRecord(
                stage.stage_index,
                stage.task_id,
                stage.adapter,
                stage.step_losses,
            )
            for stage in adaptations.sequential_single_lora.stages
        ),
        independent_adapters=tuple(
            AdapterTrainingRecord(
                stage_index,
                adapter.task_id,
                adapter.adapter,
                adapter.step_losses,
            )
            for stage_index, adapter in enumerate(
                adaptations.independent_root_lora.adapters,
                start=1,
            )
        ),
        vamp_graph=_immutable_lora_graph(adaptations.vamp.graph),
        address_book=adaptations.vamp.address_book,
        rng_state=LanguageAdaptationRngState(
            adaptations.sequential_single_lora.rng_key,
            adaptations.independent_root_lora.rng_key,
            adaptations.vamp.rng_key,
        ),
        vamp_stages=tuple(
            VampTrainingRecord(
                stage.stage_index,
                stage.task_id,
                stage.parent_node_index,
                stage.parent_node_id,
                stage.parent_mean_node_nll,
                stage.candidate_step_losses,
            )
            for stage in adaptations.vamp.stage_metrics
        ),
        max_nodes=adaptations.vamp.max_nodes,
        max_edges=adaptations.vamp.max_edges,
    )


def extract_language_vamp_artifact(
    run: LanguageVampRun,
    model_config: GptNeoConfig,
    lora_config: LoraConfig,
    train_config: LmTrainConfig,
    *,
    config_hashes: Mapping[str, str] | None = None,
) -> LanguageAdaptationArtifact:
    """Extract a strict VAMP-only artifact without fabricating baseline adapters."""
    if not isinstance(run, LanguageVampRun):
        raise TypeError("run must be a LanguageVampRun")
    required_hashes = _required_config_hashes(
        model_config,
        lora_config,
        train_config,
    )
    merged_hashes = dict(config_hashes or {})
    for name, digest in required_hashes:
        if name in merged_hashes and merged_hashes[name] != digest:
            raise ValueError(f"supplied {name} config hash does not match its config")
        merged_hashes[name] = digest
    vamp_only_hash = _payload_sha256({"mode": "vamp_only"})
    if (
        "adaptation_mode" in merged_hashes
        and merged_hashes["adaptation_mode"] != vamp_only_hash
    ):
        raise ValueError("supplied adaptation mode hash is not VAMP-only")
    merged_hashes["adaptation_mode"] = vamp_only_hash
    unused_rng = np.asarray(jax.random.PRNGKey(0), dtype=np.uint32)
    return LanguageAdaptationArtifact(
        base_checkpoint=run.base_checkpoint,
        model_config=model_config,
        lora_config=lora_config,
        train_config=train_config,
        config_hashes=tuple(merged_hashes.items()),
        task_order=tuple(task.task_id for task in run.completed_tasks),
        sequential_stages=(),
        independent_adapters=(),
        vamp_graph=run.graph,
        address_book=run.address_book,
        rng_state=LanguageAdaptationRngState(unused_rng, unused_rng, run.rng_key),
        vamp_stages=tuple(
            VampTrainingRecord(
                stage.stage_index,
                stage.task_id,
                stage.parent_node_index,
                stage.parent_node_id,
                stage.parent_mean_node_nll,
                stage.candidate_step_losses,
            )
            for stage in run.stage_metrics
        ),
        max_nodes=run.max_nodes,
        max_edges=run.max_edges,
    )


def attach_language_baseline_runs(
    vamp_artifact: LanguageAdaptationArtifact,
    sequential: SequentialLoraRun,
    independent: IndependentRootLoraRun,
    *,
    config_hashes: Mapping[str, str] | None = None,
) -> LanguageAdaptationArtifact:
    """Attach complete baseline prefixes to one authenticated VAMP-only artifact."""
    if not isinstance(vamp_artifact, LanguageAdaptationArtifact):
        raise TypeError("vamp_artifact must be a LanguageAdaptationArtifact")
    if not isinstance(sequential, SequentialLoraRun):
        raise TypeError("sequential must be a SequentialLoraRun")
    if not isinstance(independent, IndependentRootLoraRun):
        raise TypeError("independent must be an IndependentRootLoraRun")
    task_order = tuple(stage.task_id for stage in sequential.stages)
    if (
        vamp_artifact.sequential_stages
        or vamp_artifact.independent_adapters
        or task_order != tuple(adapter.task_id for adapter in independent.adapters)
        or task_order != vamp_artifact.task_order
    ):
        raise ValueError("baseline runs must align with one VAMP-only task prefix")
    if (
        sequential.train_config != vamp_artifact.train_config
        or independent.train_config != vamp_artifact.train_config
        or sequential.base_parameter_checksum
        != vamp_artifact.base_checkpoint.parameter_checksum
        or independent.base_parameter_checksum
        != vamp_artifact.base_checkpoint.parameter_checksum
    ):
        raise ValueError("baseline runs must share the VAMP base and training config")
    merged_hashes = dict(vamp_artifact.config_hashes)
    merged_hashes.pop("adaptation_mode", None)
    for name, digest in (config_hashes or {}).items():
        if name in merged_hashes and merged_hashes[name] != digest:
            raise ValueError(f"supplied {name} config hash changed the VAMP binding")
        merged_hashes[name] = digest
    return LanguageAdaptationArtifact(
        base_checkpoint=vamp_artifact.base_checkpoint,
        model_config=vamp_artifact.model_config,
        lora_config=vamp_artifact.lora_config,
        train_config=vamp_artifact.train_config,
        config_hashes=tuple(merged_hashes.items()),
        task_order=vamp_artifact.task_order,
        sequential_stages=tuple(
            AdapterTrainingRecord(
                stage.stage_index,
                stage.task_id,
                stage.adapter,
                stage.step_losses,
            )
            for stage in sequential.stages
        ),
        independent_adapters=tuple(
            AdapterTrainingRecord(
                stage_index,
                adapter.task_id,
                adapter.adapter,
                adapter.step_losses,
            )
            for stage_index, adapter in enumerate(independent.adapters, start=1)
        ),
        vamp_graph=vamp_artifact.vamp_graph,
        address_book=vamp_artifact.address_book,
        rng_state=LanguageAdaptationRngState(
            np.asarray(sequential.rng_key, dtype=np.uint32),
            np.asarray(independent.rng_key, dtype=np.uint32),
            vamp_artifact.rng_state.vamp,
        ),
        vamp_stages=vamp_artifact.vamp_stages,
        max_nodes=vamp_artifact.max_nodes,
        max_edges=vamp_artifact.max_edges,
    )


def save_language_adaptation_artifact(
    directory: str | Path,
    artifact: LanguageAdaptationArtifact,
) -> Path:
    """Atomically publish a strict adaptation manifest and safetensors file."""
    if not isinstance(artifact, LanguageAdaptationArtifact):
        raise TypeError("artifact must be a LanguageAdaptationArtifact")
    target = Path(directory)
    if target.exists():
        raise FileExistsError(f"adaptation artifact directory already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent)
    )
    try:
        arrays = _artifact_tensors(artifact)
        tensor_checksum = _tensor_checksum(arrays)
        tensor_path = temporary / LANGUAGE_ADAPTATION_TENSORS
        _write_safetensors(
            tensor_path,
            arrays,
            {
                "format": LANGUAGE_ADAPTATION_FORMAT,
                "schema_version": str(LANGUAGE_ADAPTATION_SCHEMA_VERSION),
                "tensor_checksum": tensor_checksum,
            },
        )
        manifest_without_checksum = _artifact_manifest(
            artifact,
            arrays,
            tensor_checksum,
            _file_sha256(tensor_path),
        )
        manifest = {
            **manifest_without_checksum,
            "manifest_payload_sha256": _payload_sha256(manifest_without_checksum),
        }
        _write_file(
            temporary / LANGUAGE_ADAPTATION_MANIFEST,
            _stable_json_bytes(manifest, newline=True),
        )
        loaded = _load_language_adaptation_directory(temporary)
        if loaded.tensor_checksum != artifact.tensor_checksum:
            raise RuntimeError("adaptation artifact validation changed tensor values")
        _fsync_directory(temporary)
        os.replace(temporary, target)
        _fsync_directory(target.parent)
        return target.resolve()
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def load_language_adaptation_artifact(
    directory: str | Path,
) -> LanguageAdaptationArtifact:
    """Load an adaptation artifact only after all metadata and tensors validate."""
    return _load_language_adaptation_directory(Path(directory))


def _artifact_manifest(
    artifact: LanguageAdaptationArtifact,
    arrays: Mapping[str, np.ndarray],
    tensor_checksum: str,
    tensor_file_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": LANGUAGE_ADAPTATION_SCHEMA_VERSION,
        "format": LANGUAGE_ADAPTATION_FORMAT,
        "tensor_file": LANGUAGE_ADAPTATION_TENSORS,
        "tensor_file_sha256": tensor_file_sha256,
        "tensor_checksum": tensor_checksum,
        "base_checkpoint": {
            "directory": str(artifact.base_checkpoint.directory),
            "manifest_sha256": artifact.base_checkpoint.manifest_sha256,
            "parameter_checksum": artifact.base_checkpoint.parameter_checksum,
        },
        "configs": {
            "model": _model_config_payload(artifact.model_config),
            "lora": _lora_config_payload(artifact.lora_config),
            "training": _train_config_payload(artifact.train_config),
        },
        "config_hashes": dict(artifact.config_hashes),
        "task_order": [str(task_id) for task_id in artifact.task_order],
        "capacities": {
            "max_nodes": artifact.max_nodes,
            "max_edges": artifact.max_edges,
        },
        "sequential_stages": [
            {
                "stage_index": record.stage_index,
                "task_id": str(record.task_id),
                "adapter_prefix": _adapter_prefix("sequential", record.stage_index),
                "training_trace_tensor": _trace_name(
                    "sequential", record.stage_index
                ),
            }
            for record in artifact.sequential_stages
        ],
        "independent_adapters": [
            {
                "stage_index": record.stage_index,
                "task_id": str(record.task_id),
                "adapter_prefix": _adapter_prefix("independent", record.stage_index),
                "training_trace_tensor": _trace_name(
                    "independent", record.stage_index
                ),
            }
            for record in artifact.independent_adapters
        ],
        "vamp": {
            "graph": [
                {
                    "node_id": str(node.node_id),
                    "parent_id": (
                        None if node.parent_id is None else str(node.parent_id)
                    ),
                    "trained_task": (
                        None if node.trained_task is None else str(node.trained_task)
                    ),
                    "train_stage": node.train_stage,
                    "depth": node.depth,
                    "adapter_prefix": (
                        None
                        if node.incoming_edge is None
                        else _adapter_prefix("vamp", node.train_stage)
                    ),
                }
                for node in artifact.vamp_graph.nodes
            ],
            "address_book": {
                "node_ids": [
                    None if node_id is None else str(node_id)
                    for node_id in artifact.address_book.node_ids
                ],
                "keys_tensor": "vamp.address_book.keys",
                "valid_node_mask_tensor": "vamp.address_book.valid_node_mask",
            },
            "stages": [
                {
                    "stage_index": record.stage_index,
                    "task_id": str(record.task_id),
                    "parent_node_index": record.parent_node_index,
                    "parent_node_id": str(record.parent_node_id),
                    "parent_scores_tensor": _parent_scores_name(record.stage_index),
                    "training_trace_tensor": _trace_name(
                        "vamp", record.stage_index
                    ),
                }
                for record in artifact.vamp_stages
            ],
        },
        "rng_tensors": {
            "sequential_single_lora": "rng.sequential_single_lora",
            "independent_root_lora": "rng.independent_root_lora",
            "vamp": "rng.vamp",
        },
        "tensors": {
            name: {
                "shape": list(array.shape),
                "dtype": array.dtype.name,
                "nbytes": int(array.nbytes),
                "sha256": _array_checksum(name, array),
            }
            for name, array in arrays.items()
        },
    }


def _artifact_tensors(
    artifact: LanguageAdaptationArtifact,
) -> dict[str, np.ndarray]:
    tensors: dict[str, np.ndarray] = {}
    for family, records in (
        ("sequential", artifact.sequential_stages),
        ("independent", artifact.independent_adapters),
    ):
        for record in records:
            tensors.update(
                _prefixed_lora_tensors(
                    _adapter_prefix(family, record.stage_index),
                    record.adapter,
                    artifact.model_config,
                    artifact.lora_config,
                )
            )
            tensors[_trace_name(family, record.stage_index)] = np.asarray(
                record.training_trace,
                dtype=np.float64,
            )
    for node in artifact.vamp_graph.nodes[1:]:
        if node.incoming_edge is None:
            raise ValueError("non-root VAMP nodes must contain LoRA edges")
        tensors.update(
            _prefixed_lora_tensors(
                _adapter_prefix("vamp", node.train_stage),
                node.incoming_edge,
                artifact.model_config,
                artifact.lora_config,
            )
        )
    for record in artifact.vamp_stages:
        tensors[_parent_scores_name(record.stage_index)] = np.asarray(
            record.parent_mean_node_nll,
            dtype=np.float64,
        )
        tensors[_trace_name("vamp", record.stage_index)] = np.asarray(
            record.training_trace,
            dtype=np.float64,
        )
    tensors.update(
        {
            "vamp.address_book.keys": np.asarray(
                artifact.address_book.keys,
                dtype=np.float32,
            ),
            "vamp.address_book.valid_node_mask": np.asarray(
                artifact.address_book.valid_node_mask,
                dtype=np.bool_,
            ),
            "rng.sequential_single_lora": artifact.rng_state.sequential_single_lora,
            "rng.independent_root_lora": artifact.rng_state.independent_root_lora,
            "rng.vamp": artifact.rng_state.vamp,
        }
    )
    return {
        name: _canonical_safetensors_array(array)
        for name, array in sorted(tensors.items())
    }


def _prefixed_lora_tensors(
    prefix: str,
    edge: LoraEdge,
    model_config: GptNeoConfig,
    lora_config: LoraConfig,
) -> dict[str, np.ndarray]:
    return {
        f"{prefix}.{name}": np.asarray(value, dtype=np.float32)
        for name, value in flatten_lora_edge(
            edge,
            model_config,
            lora_config,
        ).items()
    }


def _load_language_adaptation_directory(
    directory: Path,
) -> LanguageAdaptationArtifact:
    if not directory.is_dir():
        raise FileNotFoundError(f"adaptation artifact directory does not exist: {directory}")
    expected_entries = {
        LANGUAGE_ADAPTATION_MANIFEST,
        LANGUAGE_ADAPTATION_TENSORS,
    }
    if {path.name for path in directory.iterdir()} != expected_entries:
        raise ValueError("adaptation artifact directory entries are not canonical")
    manifest = _parse_manifest(
        (directory / LANGUAGE_ADAPTATION_MANIFEST).read_bytes()
    )
    manifest_without_checksum = {
        name: value
        for name, value in manifest.items()
        if name != "manifest_payload_sha256"
    }
    if (
        _payload_sha256(manifest_without_checksum)
        != manifest["manifest_payload_sha256"]
    ):
        raise ValueError("adaptation manifest payload hash mismatch")
    tensor_path = directory / LANGUAGE_ADAPTATION_TENSORS
    if _file_sha256(tensor_path) != manifest["tensor_file_sha256"]:
        raise ValueError("adaptation tensor file hash mismatch")
    arrays, metadata = _read_safetensors(tensor_path)
    expected_metadata = {
        "format": LANGUAGE_ADAPTATION_FORMAT,
        "schema_version": str(LANGUAGE_ADAPTATION_SCHEMA_VERSION),
        "tensor_checksum": str(manifest["tensor_checksum"]),
    }
    if metadata != expected_metadata:
        raise ValueError("adaptation safetensors metadata does not match manifest")
    if _tensor_checksum(arrays) != manifest["tensor_checksum"]:
        raise ValueError("adaptation tensor checksum mismatch")
    _validate_manifest_tensors(
        _expect_dict(manifest["tensors"], "manifest tensors"),
        arrays,
    )
    artifact = _artifact_from_manifest(manifest, arrays)
    reserialized = _artifact_tensors(artifact)
    if set(reserialized) != set(arrays) or any(
        not np.array_equal(reserialized[name], arrays[name], equal_nan=True)
        for name in arrays
    ):
        raise ValueError("adaptation tensor references are incomplete or inconsistent")
    return artifact


def _artifact_from_manifest(
    manifest: Mapping[str, object],
    arrays: Mapping[str, np.ndarray],
) -> LanguageAdaptationArtifact:
    configs = _expect_dict(manifest["configs"], "configs")
    _require_exact_keys(configs, {"model", "lora", "training"}, "configs")
    model_config = _model_config_from_payload(
        _expect_dict(configs["model"], "model config")
    )
    lora_config = _lora_config_from_payload(
        _expect_dict(configs["lora"], "LoRA config")
    )
    train_config = _train_config_from_payload(
        _expect_dict(configs["training"], "training config")
    )
    base_payload = _expect_dict(manifest["base_checkpoint"], "base checkpoint")
    _require_exact_keys(
        base_payload,
        {"directory", "manifest_sha256", "parameter_checksum"},
        "base checkpoint",
    )
    base_checkpoint = BaseCheckpointRef(
        directory=Path(_require_string(base_payload["directory"], "base directory")),
        manifest_sha256=_require_sha256(
            base_payload["manifest_sha256"],
            "base manifest hash",
        ),
        parameter_checksum=_require_sha256(
            base_payload["parameter_checksum"],
            "base parameter checksum",
        ),
    )
    task_values = _expect_list(manifest["task_order"], "task order")
    task_order = tuple(
        TaskId(_require_string(value, "task ID")) for value in task_values
    )
    capacities = _expect_dict(manifest["capacities"], "capacities")
    _require_exact_keys(capacities, {"max_nodes", "max_edges"}, "capacities")
    max_nodes = _require_int(capacities["max_nodes"], "max_nodes")
    max_edges = _require_int(capacities["max_edges"], "max_edges")

    def adapter_records(field_name: str, family: str) -> tuple[AdapterTrainingRecord, ...]:
        return tuple(
            _adapter_record_from_payload(
                _expect_dict(value, f"{field_name} record"),
                family,
                arrays,
                model_config,
                lora_config,
            )
            for value in _expect_list(manifest[field_name], field_name)
        )

    vamp_payload = _expect_dict(manifest["vamp"], "VAMP payload")
    _require_exact_keys(vamp_payload, {"graph", "address_book", "stages"}, "VAMP")
    graph = _graph_from_payload(
        _expect_list(vamp_payload["graph"], "VAMP graph"),
        arrays,
        model_config,
        lora_config,
    )
    address_payload = _expect_dict(vamp_payload["address_book"], "address book")
    _require_exact_keys(
        address_payload,
        {"node_ids", "keys_tensor", "valid_node_mask_tensor"},
        "address book",
    )
    address_book = AddressBook(
        node_ids=tuple(
            None if value is None else NodeId(_require_string(value, "node ID"))
            for value in _expect_list(address_payload["node_ids"], "address node IDs")
        ),
        keys=_required_tensor(
            arrays,
            address_payload["keys_tensor"],
            "address keys",
            np.float32,
            (max_nodes, model_config.hidden_size),
        ),
        valid_node_mask=_required_tensor(
            arrays,
            address_payload["valid_node_mask_tensor"],
            "address valid-node mask",
            np.bool_,
            (max_nodes,),
        ),
    )
    rng_payload = _expect_dict(manifest["rng_tensors"], "RNG tensors")
    _require_exact_keys(
        rng_payload,
        {"sequential_single_lora", "independent_root_lora", "vamp"},
        "RNG tensors",
    )
    rng_state = LanguageAdaptationRngState(
        **{
            name: _required_tensor(
                arrays,
                tensor_name,
                f"{name} RNG",
                np.uint32,
                (2,),
            )
            for name, tensor_name in rng_payload.items()
        }
    )
    config_hash_payload = _expect_dict(manifest["config_hashes"], "config hashes")
    return LanguageAdaptationArtifact(
        base_checkpoint=base_checkpoint,
        model_config=model_config,
        lora_config=lora_config,
        train_config=train_config,
        config_hashes=tuple(
            (name, _require_sha256(value, f"config hash {name}"))
            for name, value in config_hash_payload.items()
        ),
        task_order=task_order,
        sequential_stages=adapter_records("sequential_stages", "sequential"),
        independent_adapters=adapter_records(
            "independent_adapters", "independent"
        ),
        vamp_graph=graph,
        address_book=address_book,
        rng_state=rng_state,
        vamp_stages=tuple(
            _vamp_record_from_payload(
                _expect_dict(value, "VAMP stage"),
                arrays,
                max_nodes,
                train_config.steps,
            )
            for value in _expect_list(vamp_payload["stages"], "VAMP stages")
        ),
        max_nodes=max_nodes,
        max_edges=max_edges,
    )


def _adapter_record_from_payload(
    payload: Mapping[str, object],
    family: str,
    arrays: Mapping[str, np.ndarray],
    model_config: GptNeoConfig,
    lora_config: LoraConfig,
) -> AdapterTrainingRecord:
    _require_exact_keys(
        payload,
        {"stage_index", "task_id", "adapter_prefix", "training_trace_tensor"},
        f"{family} adapter record",
    )
    stage_index = _require_int(payload["stage_index"], "adapter stage index")
    prefix = _require_string(payload["adapter_prefix"], "adapter prefix")
    if prefix != _adapter_prefix(family, stage_index):
        raise ValueError(f"{family} adapter prefix is not canonical")
    return AdapterTrainingRecord(
        stage_index=stage_index,
        task_id=TaskId(_require_string(payload["task_id"], "adapter task ID")),
        adapter=_lora_edge_below_prefix(
            arrays,
            prefix,
            model_config,
            lora_config,
        ),
        training_trace=tuple(
            float(value)
            for value in _required_tensor(
                arrays,
                payload["training_trace_tensor"],
                "adapter training trace",
                np.float64,
                None,
                rank=1,
            )
        ),
    )


def _graph_from_payload(
    values: list[object],
    arrays: Mapping[str, np.ndarray],
    model_config: GptNeoConfig,
    lora_config: LoraConfig,
) -> MemoryGraph[LoraEdge]:
    nodes: list[MemoryNode[LoraEdge]] = []
    for value in values:
        payload = _expect_dict(value, "VAMP graph node")
        _require_exact_keys(
            payload,
            {
                "node_id",
                "parent_id",
                "trained_task",
                "train_stage",
                "depth",
                "adapter_prefix",
            },
            "VAMP graph node",
        )
        train_stage = _require_int(payload["train_stage"], "node train_stage")
        adapter_prefix = payload["adapter_prefix"]
        incoming_edge = None
        if adapter_prefix is not None:
            prefix = _require_string(adapter_prefix, "VAMP adapter prefix")
            if prefix != _adapter_prefix("vamp", train_stage):
                raise ValueError("VAMP adapter prefix is not canonical")
            incoming_edge = _lora_edge_below_prefix(
                arrays,
                prefix,
                model_config,
                lora_config,
            )
        nodes.append(
            MemoryNode(
                node_id=NodeId(_require_string(payload["node_id"], "node ID")),
                parent_id=(
                    None
                    if payload["parent_id"] is None
                    else NodeId(
                        _require_string(payload["parent_id"], "parent node ID")
                    )
                ),
                trained_task=(
                    None
                    if payload["trained_task"] is None
                    else TaskId(
                        _require_string(payload["trained_task"], "trained task")
                    )
                ),
                train_stage=train_stage,
                depth=_require_int(payload["depth"], "node depth"),
                incoming_edge=incoming_edge,
            )
        )
    return MemoryGraph(nodes=tuple(nodes))


def _vamp_record_from_payload(
    payload: Mapping[str, object],
    arrays: Mapping[str, np.ndarray],
    max_nodes: int,
    training_steps: int,
) -> VampTrainingRecord:
    _require_exact_keys(
        payload,
        {
            "stage_index",
            "task_id",
            "parent_node_index",
            "parent_node_id",
            "parent_scores_tensor",
            "training_trace_tensor",
        },
        "VAMP stage",
    )
    return VampTrainingRecord(
        stage_index=_require_int(payload["stage_index"], "VAMP stage index"),
        task_id=TaskId(_require_string(payload["task_id"], "VAMP task ID")),
        parent_node_index=_require_int(
            payload["parent_node_index"],
            "VAMP parent node index",
        ),
        parent_node_id=NodeId(
            _require_string(payload["parent_node_id"], "VAMP parent node ID")
        ),
        parent_mean_node_nll=tuple(
            float(value)
            for value in _required_tensor(
                arrays,
                payload["parent_scores_tensor"],
                "VAMP parent scores",
                np.float64,
                (max_nodes,),
            )
        ),
        training_trace=tuple(
            float(value)
            for value in _required_tensor(
                arrays,
                payload["training_trace_tensor"],
                "VAMP training trace",
                np.float64,
                (training_steps,),
            )
        ),
    )


def _lora_edge_below_prefix(
    arrays: Mapping[str, np.ndarray],
    prefix: str,
    model_config: GptNeoConfig,
    lora_config: LoraConfig,
) -> LoraEdge:
    marker = f"{prefix}."
    return unflatten_lora_edge(
        {
            name.removeprefix(marker): array
            for name, array in arrays.items()
            if name.startswith(marker)
        },
        model_config,
        lora_config,
    )


def _validate_language_adaptation_artifact(
    artifact: LanguageAdaptationArtifact,
) -> None:
    if not isinstance(artifact.base_checkpoint, BaseCheckpointRef):
        raise TypeError("base_checkpoint must be a BaseCheckpointRef")
    if not isinstance(artifact.model_config, GptNeoConfig):
        raise TypeError("model_config must be a GptNeoConfig")
    if not isinstance(artifact.lora_config, LoraConfig):
        raise TypeError("lora_config must be a LoraConfig")
    if not isinstance(artifact.train_config, LmTrainConfig):
        raise TypeError("train_config must be an LmTrainConfig")
    config_hashes = _canonical_config_hashes(artifact.config_hashes)
    required_hashes = dict(
        _required_config_hashes(
            artifact.model_config,
            artifact.lora_config,
            artifact.train_config,
        )
    )
    actual_hashes = dict(config_hashes)
    if any(actual_hashes.get(name) != digest for name, digest in required_hashes.items()):
        raise ValueError("artifact core config hashes do not match their configs")
    object.__setattr__(artifact, "config_hashes", config_hashes)
    if (
        not isinstance(artifact.task_order, tuple)
        or not artifact.task_order
        or any(not isinstance(task_id, str) or not task_id for task_id in artifact.task_order)
        or len(set(artifact.task_order)) != len(artifact.task_order)
    ):
        raise ValueError("task_order must contain unique nonempty task IDs")
    expected_indices = tuple(range(1, len(artifact.task_order) + 1))
    baseline_presence = tuple(
        bool(getattr(artifact, field_name))
        for field_name in ("sequential_stages", "independent_adapters")
    )
    if baseline_presence[0] != baseline_presence[1]:
        raise ValueError("adaptation baseline families must both be present or absent")
    if (
        not baseline_presence[0]
        and actual_hashes.get("adaptation_mode")
        != _payload_sha256({"mode": "vamp_only"})
    ):
        raise ValueError("baseline-free artifacts must declare VAMP-only mode")
    for field_name in ("sequential_stages", "independent_adapters"):
        records = getattr(artifact, field_name)
        if (
            not isinstance(records, tuple)
            or any(not isinstance(record, AdapterTrainingRecord) for record in records)
            or (
                records
                and (
                    tuple(record.stage_index for record in records) != expected_indices
                    or tuple(record.task_id for record in records) != artifact.task_order
                )
            )
        ):
            raise ValueError(
                f"{field_name} must be empty or align completely with task_order"
            )
        if any(len(record.training_trace) != artifact.train_config.steps for record in records):
            raise ValueError(f"{field_name} traces must match the training budget")
        for record in records:
            _validated_lora_arrays(
                flatten_lora_edge(
                    record.adapter,
                    artifact.model_config,
                    artifact.lora_config,
                ),
                artifact.model_config,
                artifact.lora_config,
            )
    if type(artifact.max_nodes) is not int or artifact.max_nodes <= 0:
        raise ValueError("max_nodes must be positive")
    if type(artifact.max_edges) is not int or artifact.max_edges < 0:
        raise ValueError("max_edges must be nonnegative")
    if artifact.max_nodes < len(artifact.task_order) + 1:
        raise ValueError("max_nodes must contain completed tasks plus root")
    if artifact.max_edges != artifact.max_nodes - 1:
        raise ValueError("max_edges must equal max_nodes minus one")
    if not isinstance(artifact.vamp_graph, MemoryGraph):
        raise TypeError("vamp_graph must be a MemoryGraph")
    graph = _immutable_lora_graph(artifact.vamp_graph)
    object.__setattr__(artifact, "vamp_graph", graph)
    _validate_artifact_graph(graph, artifact.task_order, artifact.model_config, artifact.lora_config)
    if not isinstance(artifact.address_book, AddressBook):
        raise TypeError("address_book must be an AddressBook")
    if artifact.address_book.max_nodes != artifact.max_nodes:
        raise ValueError("address-book capacity must match max_nodes")
    graph_node_ids = tuple(node.node_id for node in graph.nodes)
    expected_valid_mask = np.arange(artifact.max_nodes) < len(graph.nodes)
    if (
        artifact.address_book.node_ids[: len(graph.nodes)] != graph_node_ids
        or not np.array_equal(
            artifact.address_book.valid_node_mask,
            expected_valid_mask,
        )
    ):
        raise ValueError("address book must align with VAMP graph insertion order")
    valid_key_norms = np.linalg.norm(
        artifact.address_book.keys[: len(graph.nodes)],
        axis=1,
    )
    if not np.allclose(valid_key_norms, 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("valid address keys must be L2 normalized")
    if not isinstance(artifact.rng_state, LanguageAdaptationRngState):
        raise TypeError("rng_state must be LanguageAdaptationRngState")
    if (
        not isinstance(artifact.vamp_stages, tuple)
        or any(not isinstance(record, VampTrainingRecord) for record in artifact.vamp_stages)
        or tuple(record.stage_index for record in artifact.vamp_stages) != expected_indices
        or tuple(record.task_id for record in artifact.vamp_stages) != artifact.task_order
    ):
        raise ValueError("vamp_stages must align with task_order")
    for record, node in zip(artifact.vamp_stages, graph.nodes[1:]):
        if (
            len(record.parent_mean_node_nll) != artifact.max_nodes
            or len(record.training_trace) != artifact.train_config.steps
            or record.parent_node_index >= record.stage_index
            or graph.nodes[record.parent_node_index].node_id != record.parent_node_id
            or node.parent_id != record.parent_node_id
        ):
            raise ValueError("VAMP stage evidence does not match graph topology")
        if not all(
            np.isfinite(value)
            for value in record.parent_mean_node_nll[: record.stage_index]
        ) or not all(
            np.isposinf(value)
            for value in record.parent_mean_node_nll[record.stage_index :]
        ):
            raise ValueError("VAMP parent scores must mask unavailable nodes with infinity")


def _validate_artifact_graph(
    graph: MemoryGraph[LoraEdge],
    task_order: tuple[TaskId, ...],
    model_config: GptNeoConfig,
    lora_config: LoraConfig,
) -> None:
    if len(graph.nodes) != len(task_order) + 1:
        raise ValueError("VAMP graph must contain root plus every task node")
    root = graph.nodes[0]
    if (
        root.parent_id is not None
        or root.trained_task is not None
        or root.train_stage != 0
        or root.depth != 0
        or root.incoming_edge is not None
    ):
        raise ValueError("VAMP graph root metadata is invalid")
    prior_ids = {root.node_id}
    for stage_index, (task_id, node) in enumerate(
        zip(task_order, graph.nodes[1:]),
        start=1,
    ):
        if (
            node.node_id != NodeId(str(task_id))
            or node.trained_task != task_id
            or node.train_stage != stage_index
            or node.parent_id not in prior_ids
            or node.incoming_edge is None
        ):
            raise ValueError("VAMP graph nodes must align with task insertion order")
        parent = next(value for value in graph.nodes if value.node_id == node.parent_id)
        if node.depth != parent.depth + 1:
            raise ValueError("VAMP graph depths must follow parent topology")
        _validated_lora_arrays(
            flatten_lora_edge(node.incoming_edge, model_config, lora_config),
            model_config,
            lora_config,
        )
        prior_ids.add(node.node_id)


def _validated_lora_arrays(
    tensors: Mapping[str, jax.Array | np.ndarray],
    model_config: GptNeoConfig,
    lora_config: LoraConfig,
) -> dict[str, np.ndarray]:
    specs = _lora_tensor_specs(model_config, lora_config)
    if set(tensors) != set(specs):
        raise ValueError("LoRA tensor names do not match the canonical schema")
    arrays: dict[str, np.ndarray] = {}
    for name, expected_shape in specs.items():
        value = np.asarray(tensors[name])
        if value.shape != expected_shape:
            raise ValueError(f"LoRA tensor shape mismatch for {name}")
        if value.dtype != np.float32:
            raise TypeError(f"LoRA tensor dtype mismatch for {name}; expected float32")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"LoRA tensor contains non-finite values: {name}")
        arrays[name] = np.array(value, dtype=np.float32, copy=True)
    return dict(sorted(arrays.items()))


def _lora_tensor_specs(
    model_config: GptNeoConfig,
    lora_config: LoraConfig,
) -> dict[str, tuple[int, ...]]:
    hidden = model_config.hidden_size
    intermediate = model_config.intermediate_size
    rank = lora_config.rank
    projection_shapes = {
        "query": (hidden, hidden),
        "key": (hidden, hidden),
        "value": (hidden, hidden),
        "attention_output": (hidden, hidden),
        "mlp_input": (hidden, intermediate),
        "mlp_output": (intermediate, hidden),
    }
    return dict(
        sorted(
            (
                {
                    f"blocks.{block_index}.{projection_name}.left": (
                        input_size,
                        rank,
                    )
                    for block_index in range(model_config.num_layers)
                    for projection_name, (
                        input_size,
                        _,
                    ) in projection_shapes.items()
                }
                | {
                    f"blocks.{block_index}.{projection_name}.right": (
                        rank,
                        output_size,
                    )
                    for block_index in range(model_config.num_layers)
                    for projection_name, (
                        _,
                        output_size,
                    ) in projection_shapes.items()
                }
            ).items()
        )
    )


def _immutable_lora_edge(edge: LoraEdge) -> LoraEdge:
    if not isinstance(edge, LoraEdge):
        raise TypeError("adapter must be a LoraEdge")
    return LoraEdge(
        blocks=tuple(
            LoraBlock(
                **{
                    projection_name: LoraProjection(
                        left=jnp.asarray(
                            np.array(
                                getattr(block, projection_name).left,
                                copy=True,
                            )
                        ),
                        right=jnp.asarray(
                            np.array(
                                getattr(block, projection_name).right,
                                copy=True,
                            )
                        ),
                    )
                    for projection_name in _PROJECTION_NAMES
                }
            )
            for block in edge.blocks
        )
    )


def _immutable_lora_graph(graph: MemoryGraph[LoraEdge]) -> MemoryGraph[LoraEdge]:
    if not isinstance(graph, MemoryGraph):
        raise TypeError("graph must be a MemoryGraph")
    return MemoryGraph(
        nodes=tuple(
            MemoryNode(
                node_id=node.node_id,
                parent_id=node.parent_id,
                trained_task=node.trained_task,
                train_stage=node.train_stage,
                depth=node.depth,
                incoming_edge=(
                    None
                    if node.incoming_edge is None
                    else _immutable_lora_edge(node.incoming_edge)
                ),
            )
            for node in graph.nodes
        )
    )


def _validated_training_trace(values: tuple[float, ...]) -> tuple[float, ...]:
    if not isinstance(values, tuple):
        raise TypeError("training_trace must be a tuple")
    trace = tuple(float(value) for value in values)
    if not trace or any(not np.isfinite(value) or value < 0.0 for value in trace):
        raise ValueError("training_trace values must be finite and nonnegative")
    return trace


def _immutable_rng_key(value: object, field_name: str) -> np.ndarray:
    try:
        key = np.asarray(jax.random.key_data(value))
    except (TypeError, ValueError):
        key = np.asarray(value)
    if key.shape != (2,) or key.dtype != np.uint32:
        raise ValueError(f"{field_name} must be one uint32 JAX RNG key")
    immutable = np.array(key, dtype=np.uint32, copy=True)
    immutable.flags.writeable = False
    return immutable


def _canonical_config_hashes(
    values: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    if not isinstance(values, tuple) or any(
        not isinstance(value, tuple) or len(value) != 2 for value in values
    ):
        raise TypeError("config_hashes must contain name/digest tuples")
    names = tuple(name for name, _ in values)
    if (
        not names
        or len(set(names)) != len(names)
        or any(
            not isinstance(name, str)
            or _CONFIG_NAME_PATTERN.fullmatch(name) is None
            for name in names
        )
    ):
        raise ValueError("config hash names must be unique canonical identifiers")
    for name, digest in values:
        _require_sha256(digest, f"config hash {name}")
    return tuple(sorted(values))


def _required_config_hashes(
    model_config: GptNeoConfig,
    lora_config: LoraConfig,
    train_config: LmTrainConfig,
) -> tuple[tuple[str, str], ...]:
    return (
        ("lora", _payload_sha256(_lora_config_payload(lora_config))),
        ("model", _payload_sha256(_model_config_payload(model_config))),
        ("training", _payload_sha256(_train_config_payload(train_config))),
    )


def _model_config_payload(config: GptNeoConfig) -> dict[str, object]:
    return {
        "vocab_size": config.vocab_size,
        "max_position_embeddings": config.max_position_embeddings,
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "num_layers": config.num_layers,
        "num_heads": config.num_heads,
        "attention_types": list(config.attention_types),
        "local_window_size": config.local_window_size,
        "layer_norm_epsilon": config.layer_norm_epsilon,
        "initializer_range": config.initializer_range,
        "activation": config.activation,
        "embedding_dropout": config.embedding_dropout,
        "attention_dropout": config.attention_dropout,
        "residual_dropout": config.residual_dropout,
    }


def _model_config_from_payload(payload: Mapping[str, object]) -> GptNeoConfig:
    expected = {
        "vocab_size",
        "max_position_embeddings",
        "hidden_size",
        "intermediate_size",
        "num_layers",
        "num_heads",
        "attention_types",
        "local_window_size",
        "layer_norm_epsilon",
        "initializer_range",
        "activation",
        "embedding_dropout",
        "attention_dropout",
        "residual_dropout",
    }
    _require_exact_keys(payload, expected, "model config")
    attention_types = _expect_list(payload["attention_types"], "attention types")
    return GptNeoConfig(
        vocab_size=_require_int(payload["vocab_size"], "vocab_size"),
        max_position_embeddings=_require_int(
            payload["max_position_embeddings"], "max_position_embeddings"
        ),
        hidden_size=_require_int(payload["hidden_size"], "hidden_size"),
        intermediate_size=_require_int(
            payload["intermediate_size"], "intermediate_size"
        ),
        num_layers=_require_int(payload["num_layers"], "num_layers"),
        num_heads=_require_int(payload["num_heads"], "num_heads"),
        attention_types=tuple(
            _require_string(value, "attention type") for value in attention_types
        ),  # type: ignore[arg-type]
        local_window_size=_require_int(
            payload["local_window_size"], "local_window_size"
        ),
        layer_norm_epsilon=_require_float(
            payload["layer_norm_epsilon"], "layer_norm_epsilon"
        ),
        initializer_range=_require_float(
            payload["initializer_range"], "initializer_range"
        ),
        activation=_require_string(payload["activation"], "activation"),  # type: ignore[arg-type]
        embedding_dropout=_require_float(
            payload["embedding_dropout"], "embedding_dropout"
        ),
        attention_dropout=_require_float(
            payload["attention_dropout"], "attention_dropout"
        ),
        residual_dropout=_require_float(
            payload["residual_dropout"], "residual_dropout"
        ),
    )


def _lora_config_payload(config: LoraConfig) -> dict[str, object]:
    return {
        "rank": config.rank,
        "alpha": float(config.alpha),
        "target_mask": {
            name: getattr(config.target_mask, name) for name in _PROJECTION_NAMES
        },
    }


def _lora_config_from_payload(payload: Mapping[str, object]) -> LoraConfig:
    _require_exact_keys(payload, {"rank", "alpha", "target_mask"}, "LoRA config")
    target_payload = _expect_dict(payload["target_mask"], "LoRA target mask")
    _require_exact_keys(target_payload, set(_PROJECTION_NAMES), "LoRA target mask")
    return LoraConfig(
        rank=_require_int(payload["rank"], "LoRA rank"),
        alpha=_require_float(payload["alpha"], "LoRA alpha"),
        target_mask=LoraTargetMask(
            **{
                name: _require_bool(target_payload[name], f"LoRA target {name}")
                for name in _PROJECTION_NAMES
            }
        ),
    )


def _train_config_payload(config: LmTrainConfig) -> dict[str, object]:
    return {
        "learning_rate": config.learning_rate,
        "steps": config.steps,
        "batch_size": config.batch_size,
        "weight_decay": config.weight_decay,
        "gradient_clip_norm": config.gradient_clip_norm,
    }


def _train_config_from_payload(payload: Mapping[str, object]) -> LmTrainConfig:
    _require_exact_keys(
        payload,
        {
            "learning_rate",
            "steps",
            "batch_size",
            "weight_decay",
            "gradient_clip_norm",
        },
        "training config",
    )
    return LmTrainConfig(
        learning_rate=_require_float(payload["learning_rate"], "learning_rate"),
        steps=_require_int(payload["steps"], "steps"),
        batch_size=_require_int(payload["batch_size"], "batch_size"),
        weight_decay=_require_float(payload["weight_decay"], "weight_decay"),
        gradient_clip_norm=_require_float(
            payload["gradient_clip_norm"], "gradient_clip_norm"
        ),
    )


def _adapter_prefix(family: str, stage_index: int) -> str:
    return f"{family}.stages.{stage_index - 1:04d}.adapter"


def _trace_name(family: str, stage_index: int) -> str:
    return f"{family}.stages.{stage_index - 1:04d}.training_trace"


def _parent_scores_name(stage_index: int) -> str:
    return f"vamp.stages.{stage_index - 1:04d}.parent_scores"


def _tensor_checksum(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name, array in sorted(arrays.items()):
        digest.update(_array_checksum(name, array).encode("ascii"))
    return digest.hexdigest()


def _array_checksum(name: str, array: np.ndarray) -> str:
    value = _canonical_safetensors_array(array)
    digest = hashlib.sha256()
    digest.update(name.encode("utf-8"))
    digest.update(value.dtype.name.encode("ascii"))
    digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _write_safetensors(
    path: Path,
    tensors: Mapping[str, np.ndarray],
    metadata: Mapping[str, str],
) -> None:
    arrays = {
        name: _canonical_safetensors_array(array)
        for name, array in sorted(tensors.items())
    }
    if "__metadata__" in arrays:
        raise ValueError("__metadata__ is reserved by safetensors")
    offset = 0
    header: dict[str, object] = {"__metadata__": dict(sorted(metadata.items()))}
    for name, array in arrays.items():
        end = offset + array.nbytes
        header[name] = {
            "dtype": _NUMPY_TO_SAFETENSORS[array.dtype],
            "shape": list(array.shape),
            "data_offsets": [offset, end],
        }
        offset = end
    raw_header = _stable_json_bytes(header)
    padded_header = raw_header + b" " * (-len(raw_header) % 8)
    _write_file(
        path,
        struct.pack("<Q", len(padded_header))
        + padded_header
        + b"".join(array.tobytes(order="C") for array in arrays.values()),
    )


def _read_safetensors(
    path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    contents = path.read_bytes()
    if len(contents) < 10:
        raise ValueError("adaptation safetensors file is too short")
    header_length = struct.unpack("<Q", contents[:8])[0]
    if (
        header_length < 2
        or header_length > _MAX_SAFETENSORS_HEADER_BYTES
        or 8 + header_length > len(contents)
    ):
        raise ValueError("adaptation safetensors header length is invalid")
    header = _expect_dict(
        _parse_json(contents[8 : 8 + header_length], "safetensors header"),
        "safetensors header",
    )
    metadata_payload = _expect_dict(
        header.pop("__metadata__", {}),
        "safetensors metadata",
    )
    if any(not isinstance(value, str) for value in metadata_payload.values()):
        raise ValueError("safetensors metadata values must be strings")
    data = memoryview(contents)[8 + header_length :]
    parsed: list[tuple[int, int, str, np.dtype, tuple[int, ...]]] = []
    for name, raw_value in header.items():
        tensor = _expect_dict(raw_value, f"safetensors tensor {name}")
        _require_exact_keys(tensor, {"dtype", "shape", "data_offsets"}, name)
        dtype_code = _require_string(tensor["dtype"], f"dtype for {name}")
        if dtype_code not in _SAFETENSORS_TO_NUMPY:
            raise ValueError(f"unsupported adaptation tensor dtype: {dtype_code}")
        shape = _parse_nonnegative_int_list(tensor["shape"], f"shape for {name}")
        offsets = _parse_nonnegative_int_list(
            tensor["data_offsets"],
            f"offsets for {name}",
        )
        if len(offsets) != 2 or offsets[1] < offsets[0]:
            raise ValueError(f"invalid safetensors offsets for {name}")
        dtype = _SAFETENSORS_TO_NUMPY[dtype_code]
        size = int(np.prod(shape, dtype=np.int64)) if shape else 1
        if offsets[1] - offsets[0] != size * dtype.itemsize:
            raise ValueError(f"safetensors byte count does not match {name}")
        parsed.append((offsets[0], offsets[1], name, dtype, shape))
    parsed.sort()
    arrays: dict[str, np.ndarray] = {}
    expected_offset = 0
    for start, end, name, dtype, shape in parsed:
        if start != expected_offset or end > len(data):
            raise ValueError("safetensors tensor data must be contiguous and in bounds")
        arrays[name] = np.frombuffer(data[start:end], dtype=dtype).reshape(shape).copy()
        expected_offset = end
    if expected_offset != len(data):
        raise ValueError("safetensors file contains unindexed tensor data")
    return dict(sorted(arrays.items())), {
        name: str(value) for name, value in sorted(metadata_payload.items())
    }


def _canonical_safetensors_array(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array)
    native_dtype = value.dtype.newbyteorder("=")
    if native_dtype not in _NUMPY_TO_SAFETENSORS:
        raise TypeError(f"unsupported adaptation tensor dtype: {value.dtype}")
    return np.ascontiguousarray(value.astype(native_dtype.newbyteorder("<"), copy=False))


def _parse_manifest(contents: bytes) -> dict[str, object]:
    manifest = _expect_dict(_parse_json(contents, "adaptation manifest"), "manifest")
    expected = {
        "schema_version",
        "format",
        "tensor_file",
        "tensor_file_sha256",
        "tensor_checksum",
        "base_checkpoint",
        "configs",
        "config_hashes",
        "task_order",
        "capacities",
        "sequential_stages",
        "independent_adapters",
        "vamp",
        "rng_tensors",
        "tensors",
        "manifest_payload_sha256",
    }
    _require_exact_keys(manifest, expected, "adaptation manifest")
    if manifest["schema_version"] != LANGUAGE_ADAPTATION_SCHEMA_VERSION:
        raise ValueError("unsupported adaptation artifact schema version")
    if manifest["format"] != LANGUAGE_ADAPTATION_FORMAT:
        raise ValueError("unsupported adaptation artifact format")
    if manifest["tensor_file"] != LANGUAGE_ADAPTATION_TENSORS:
        raise ValueError("adaptation tensor filename is not canonical")
    for name in (
        "tensor_file_sha256",
        "tensor_checksum",
        "manifest_payload_sha256",
    ):
        _require_sha256(manifest[name], name)
    return manifest


def _validate_manifest_tensors(
    payload: Mapping[str, object],
    arrays: Mapping[str, np.ndarray],
) -> None:
    if set(payload) != set(arrays):
        raise ValueError("manifest tensor names do not match safetensors")
    for name, array in arrays.items():
        tensor = _expect_dict(payload[name], f"manifest tensor {name}")
        _require_exact_keys(
            tensor,
            {"shape", "dtype", "nbytes", "sha256"},
            f"manifest tensor {name}",
        )
        if _parse_nonnegative_int_list(tensor["shape"], f"shape for {name}") != array.shape:
            raise ValueError(f"manifest tensor shape mismatch for {name}")
        if tensor["dtype"] != array.dtype.name:
            raise ValueError(f"manifest tensor dtype mismatch for {name}")
        if _require_int(tensor["nbytes"], f"nbytes for {name}") != array.nbytes:
            raise ValueError(f"manifest tensor byte count mismatch for {name}")
        if _require_sha256(tensor["sha256"], f"tensor hash {name}") != _array_checksum(name, array):
            raise ValueError(f"manifest tensor hash mismatch for {name}")


def _required_tensor(
    arrays: Mapping[str, np.ndarray],
    raw_name: object,
    context: str,
    dtype: np.dtype | type[np.generic],
    shape: tuple[int, ...] | None,
    *,
    rank: int | None = None,
) -> np.ndarray:
    name = _require_string(raw_name, f"{context} tensor name")
    if name not in arrays:
        raise ValueError(f"missing {context} tensor: {name}")
    value = arrays[name]
    if value.dtype != np.dtype(dtype):
        raise TypeError(f"{context} tensor has the wrong dtype")
    if shape is not None and value.shape != shape:
        raise ValueError(f"{context} tensor has the wrong shape")
    if rank is not None and value.ndim != rank:
        raise ValueError(f"{context} tensor has the wrong rank")
    return value


def _write_file(path: Path, contents: bytes) -> None:
    with path.open("xb") as output:
        output.write(contents)
        output.flush()
        os.fsync(output.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_sha256(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(_stable_json_bytes(payload)).hexdigest()


def _stable_json_bytes(payload: object, *, newline: bool = False) -> bytes:
    suffix = "\n" if newline else ""
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + suffix
    ).encode("utf-8")


def _parse_json(contents: bytes, context: str) -> object:
    try:
        return json.loads(
            contents.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{context} is not valid JSON") from error


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate JSON key: {name}")
        result[name] = value
    return result


def _expect_dict(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{context} must be a JSON object with string keys")
    return value


def _expect_list(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a JSON list")
    return value


def _require_exact_keys(
    payload: Mapping[str, object],
    expected: set[str],
    context: str,
) -> None:
    if set(payload) != expected:
        missing = tuple(sorted(expected - set(payload)))
        unexpected = tuple(sorted(set(payload) - expected))
        raise ValueError(
            f"{context} fields do not match; missing={missing}, unexpected={unexpected}"
        )


def _parse_nonnegative_int_list(value: object, context: str) -> tuple[int, ...]:
    values = _expect_list(value, context)
    parsed = tuple(_require_int(item, context) for item in values)
    if any(item < 0 for item in parsed):
        raise ValueError(f"{context} must contain nonnegative integers")
    return parsed


def _require_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a nonempty string")
    return value


def _require_int(value: object, context: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{context} must be an integer")
    return value


def _require_float(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _require_bool(value: object, context: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{context} must be a boolean")
    return value


def _require_sha256(value: object, context: str) -> str:
    digest = _require_string(value, context)
    if _SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return digest


__all__ = [
    "AdapterTrainingRecord",
    "LANGUAGE_ADAPTATION_FORMAT",
    "LANGUAGE_ADAPTATION_MANIFEST",
    "LANGUAGE_ADAPTATION_SCHEMA_VERSION",
    "LANGUAGE_ADAPTATION_TENSORS",
    "LanguageAdaptationArtifact",
    "LanguageAdaptationRngState",
    "VampTrainingRecord",
    "attach_language_baseline_runs",
    "extract_language_adaptation_artifact",
    "extract_language_vamp_artifact",
    "flatten_lora_edge",
    "load_language_adaptation_artifact",
    "read_safetensors_archive",
    "save_language_adaptation_artifact",
    "unflatten_lora_edge",
    "write_safetensors_archive",
]

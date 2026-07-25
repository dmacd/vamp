"""One-trajectory independent-adapter sweep for the registered pilot budgets."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
from safetensors.numpy import load_file, save_file

from apm.continual.language_adaptation_artifact import (
    flatten_lora_edge,
    unflatten_lora_edge,
)
from apm.continual.language_baseline_training import IndependentRootAdapter
from apm.data.text.tinyworlds_q_semantic.adaptation import (
    PreparedQueryAdaptation,
    materialize_query_language_task,
)
from apm.data.text.tinyworlds_q_semantic.contracts import (
    QueryExperimentPreset,
    QueryPartitionArtifact,
    canonical_json_bytes,
    record_sha256,
    require_sha256,
)
from apm.data.text.tinyworlds_q_semantic.selected_base import (
    QuerySelectedBase,
    load_query_selected_base,
)
from apm.lm.checkpoint import load_gpt_neo_checkpoint, parameter_checksum
from apm.lm.lora import init_lora_edge
from apm.lm.lora_memory import pack_lora_memory
from apm.lm.training import init_candidate_lora_train_state
from apm.lm.workflow import run_resumable_candidate_edge_updates
from apm.memory.graph import NodeId, TaskId, init_memory_graph


PILOT_SWEEP_FORMAT = "tinyworlds-q-semantic-pilot-independent-sweep-v1"
PilotSweepProgress = Callable[[str, int, float, int], None]


@dataclass(frozen=True, slots=True)
class PilotIndependentBudget:
    """Independent adapter snapshots for one absolute update checkpoint."""

    updates: int
    adapters: tuple[IndependentRootAdapter, ...]
    tensor_checksum: str

    def __post_init__(self) -> None:
        if type(self.updates) is not int or self.updates <= 0:
            raise ValueError("pilot snapshot updates must be positive")
        if (
            type(self.adapters) is not tuple
            or not self.adapters
            or len({str(adapter.task_id) for adapter in self.adapters})
            != len(self.adapters)
            or any(len(adapter.step_losses) != self.updates for adapter in self.adapters)
        ):
            raise ValueError("pilot budget adapters are incomplete")
        require_sha256(self.tensor_checksum, "pilot budget tensor checksum")


@dataclass(frozen=True, eq=False, slots=True)
class PilotIndependentSweep:
    """Newest immutable concept-boundary state for all three pilot budgets."""

    stage_directory: Path
    completed_concept_ids: tuple[str, ...]
    budgets: tuple[PilotIndependentBudget, ...]
    rng_by_stage: tuple[np.ndarray, ...]
    preparation_sha256: str
    config_sha256: str
    selected_base_sha256: str
    sweep_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage_directory", Path(self.stage_directory))
        if not self.stage_directory.is_dir():
            raise FileNotFoundError(self.stage_directory)
        for value, label in (
            (self.preparation_sha256, "pilot sweep preparation"),
            (self.config_sha256, "pilot sweep config"),
            (self.selected_base_sha256, "pilot sweep selected base"),
            (self.sweep_sha256, "pilot sweep"),
        ):
            require_sha256(value, label)
        expected_tasks = tuple(TaskId(value) for value in self.completed_concept_ids)
        if any(
            tuple(adapter.task_id for adapter in budget.adapters) != expected_tasks
            for budget in self.budgets
        ):
            raise ValueError("pilot sweep budgets changed their concept prefix")
        immutable_rng = tuple(_immutable_rng(value) for value in self.rng_by_stage)
        if len(immutable_rng) != len(self.completed_concept_ids):
            raise ValueError("pilot sweep RNG snapshots changed their stage prefix")
        object.__setattr__(self, "rng_by_stage", immutable_rng)

    def budget(self, updates: int) -> PilotIndependentBudget:
        """Return the unique registered budget snapshot."""
        matches = tuple(item for item in self.budgets if item.updates == updates)
        if len(matches) != 1:
            raise ValueError(f"pilot sweep does not contain budget {updates}")
        return matches[0]


def train_or_resume_pilot_independent_sweep(
    prepared: PreparedQueryAdaptation,
    artifact: QueryPartitionArtifact,
    selected_base: QuerySelectedBase,
    working_directory: str | Path,
    preset: QueryExperimentPreset,
    *,
    progress: PilotSweepProgress | None = None,
) -> PilotIndependentSweep:
    """Train each world once and persist exact 500/1,000/2,000 snapshots."""
    _validate_bindings(prepared, artifact, preset)
    maximum_budget = max(preset.pilot_update_budgets)
    if preset.adapter_updates != maximum_budget:
        raise ValueError("pilot sweep preset must use the maximum registered budget")
    selected_base = load_query_selected_base(selected_base.directory, artifact, preset)
    loaded_base = load_gpt_neo_checkpoint(selected_base.checkpoint)
    if loaded_base.config != preset.model_config:
        raise ValueError("pilot sweep base architecture changed")
    base_checksum = parameter_checksum(loaded_base.params, loaded_base.config)
    working = Path(working_directory)
    stages_root = working / "stages"
    stages_root.mkdir(parents=True, exist_ok=True)
    completed = _completed_stage_directories(stages_root, preset.active_world_count)
    if completed:
        restored = load_pilot_independent_sweep(
            completed[-1],
            prepared,
            artifact,
            selected_base,
            preset,
        )
        adapters_by_budget = {
            budget.updates: list(budget.adapters) for budget in restored.budgets
        }
        rng_by_stage = list(restored.rng_by_stage)
        current_rng = jnp.asarray(restored.rng_by_stage[-1])
    else:
        _sequential_key, current_rng, _vamp_key = jax.random.split(
            jax.random.PRNGKey(preset.seed),
            3,
        )
        adapters_by_budget = {
            budget: [] for budget in preset.pilot_update_budgets
        }
        rng_by_stage = []
    empty_memory = pack_lora_memory(
        init_memory_graph(NodeId("root")),
        loaded_base.config,
        preset.lora_config,
        max_nodes=2,
        max_edges=1,
    )
    parent_coefficients = jnp.zeros(
        (empty_memory.valid_edge_mask.shape[0],),
        dtype=jnp.float32,
    )
    completed_count = len(completed)
    latest = restored if completed else None
    for stage, concept_id in enumerate(
        preset.concept_ids[completed_count:],
        start=completed_count + 1,
    ):
        task = materialize_query_language_task(
            prepared,
            artifact,
            preset,
            concept_id,
            maximum_batches=maximum_budget,
        ).task
        initialization_key, training_key, next_rng = jax.random.split(
            current_rng,
            3,
        )
        state = init_candidate_lora_train_state(
            init_lora_edge(
                initialization_key,
                preset.model_config,
                preset.lora_config,
            ),
            training_key,
            preset.adapter_train_config,
        )
        step_losses: tuple[float, ...] = ()
        for budget in preset.pilot_update_budgets:
            state, trace, _checkpoints = run_resumable_candidate_edge_updates(
                state,
                task.train_batches,
                loaded_base.params,
                loaded_base.config,
                empty_memory,
                preset.lora_config,
                parent_coefficients,
                0,
                preset.adapter_train_config,
                stop_update=budget,
                progress=(
                    None
                    if progress is None
                    else lambda update, loss, total, concept_id=concept_id: progress(
                        concept_id,
                        update,
                        loss,
                        total,
                    )
                ),
            )
            step_losses += trace.step_losses
            if len(step_losses) != budget:
                raise RuntimeError("pilot sweep trace does not match its absolute budget")
            adapters_by_budget[budget].append(
                IndependentRootAdapter(
                    TaskId(concept_id),
                    state.trainable,
                    step_losses,
                )
            )
        current_rng = next_rng
        rng_by_stage.append(np.asarray(next_rng, dtype=np.uint32))
        latest = publish_pilot_independent_sweep_stage(
            stages_root / f"stage-{stage:03d}",
            prepared,
            artifact,
            selected_base,
            preset,
            tuple(preset.concept_ids[:stage]),
            {
                budget: tuple(adapters_by_budget[budget])
                for budget in preset.pilot_update_budgets
            },
            tuple(rng_by_stage),
        )
    if latest is None or latest.completed_concept_ids != preset.concept_ids:
        raise RuntimeError("pilot independent sweep did not complete every world")
    if parameter_checksum(loaded_base.params, loaded_base.config) != base_checksum:
        raise RuntimeError("pilot independent sweep mutated the frozen base")
    return latest


def load_pilot_independent_sweep(
    root: str | Path,
    prepared: PreparedQueryAdaptation,
    artifact: QueryPartitionArtifact,
    selected_base: QuerySelectedBase,
    preset: QueryExperimentPreset,
) -> PilotIndependentSweep:
    """Strictly load one concept-boundary sweep artifact and every tensor."""
    _validate_bindings(prepared, artifact, preset)
    directory = Path(root)
    manifest_payload = (directory / "manifest.json").read_bytes()
    try:
        manifest = json.loads(manifest_payload)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid pilot sweep manifest") from error
    if type(manifest) is not dict or canonical_json_bytes(manifest) != manifest_payload:
        raise ValueError("noncanonical pilot sweep manifest")
    required = {
        "budgets",
        "completed_concept_ids",
        "config",
        "config_sha256",
        "format",
        "partition_sha256",
        "preparation_sha256",
        "selected_base_sha256",
        "selected_checkpoint_manifest_sha256",
        "selected_checkpoint_parameter_checksum",
        "stage",
        "sweep_sha256",
        "tensor_file",
        "tensor_file_sha256",
        "tensor_checksum",
    }
    if set(manifest) != required or manifest.get("format") != PILOT_SWEEP_FORMAT:
        raise ValueError("pilot sweep manifest fields changed")
    core = {key: value for key, value in manifest.items() if key != "sweep_sha256"}
    sweep_sha256 = record_sha256(core)
    if manifest.get("sweep_sha256") != sweep_sha256:
        raise ValueError("pilot sweep content identity changed")
    completed_concepts = tuple(_string_list(manifest, "completed_concept_ids"))
    stage = _integer(manifest, "stage")
    if (
        stage != len(completed_concepts)
        or completed_concepts != preset.concept_ids[:stage]
        or directory.name != f"stage-{stage:03d}"
        or manifest.get("preparation_sha256") != prepared.preparation_sha256
        or manifest.get("partition_sha256") != artifact.partition_sha256
        or manifest.get("selected_base_sha256") != selected_base.selection_sha256
        or manifest.get("selected_checkpoint_manifest_sha256")
        != selected_base.checkpoint.manifest_sha256
        or manifest.get("selected_checkpoint_parameter_checksum")
        != selected_base.checkpoint.parameter_checksum
        or manifest.get("config") != preset.as_record()
        or manifest.get("config_sha256") != preset.config_sha256
        or manifest.get("tensor_file") != "sweep.safetensors"
    ):
        raise ValueError("pilot sweep bindings changed")
    if {
        path.name for path in directory.iterdir()
    } != {"manifest.json", "sweep.safetensors"} or any(
        not path.is_file() or path.is_symlink() for path in directory.iterdir()
    ):
        raise ValueError("pilot sweep tree entries changed")
    tensor_path = directory / "sweep.safetensors"
    if _file_sha256(tensor_path) != manifest.get("tensor_file_sha256"):
        raise ValueError("pilot sweep tensor file changed")
    tensors = load_file(str(tensor_path))
    if _tensor_checksum(tensors) != manifest.get("tensor_checksum"):
        raise ValueError("pilot sweep tensor checksum changed")
    budget_records = manifest.get("budgets")
    if type(budget_records) is not list or tuple(
        record.get("updates") if type(record) is dict else None
        for record in budget_records
    ) != preset.pilot_update_budgets:
        raise ValueError("pilot sweep budget records changed")
    consumed = set()
    budget_values = []
    for record in budget_records:
        assert type(record) is dict
        if set(record) != {"tensor_checksum", "updates"}:
            raise ValueError("pilot sweep budget record fields changed")
        budget = _integer(record, "updates")
        adapters = tuple(
            _load_adapter(tensors, budget, stage_index, preset, consumed)
            for stage_index in range(1, stage + 1)
        )
        budget_checksum = _budget_tensor_checksum(tensors, budget)
        if record.get("tensor_checksum") != budget_checksum:
            raise ValueError("pilot sweep budget tensor checksum changed")
        budget_values.append(
            PilotIndependentBudget(budget, adapters, budget_checksum)
        )
    rng_by_stage = tuple(
        _consume_tensor(tensors, f"rng.stage.{stage_index:03d}", consumed)
        for stage_index in range(1, stage + 1)
    )
    if consumed != set(tensors):
        raise ValueError("pilot sweep tensor entries changed")
    return PilotIndependentSweep(
        stage_directory=directory.resolve(),
        completed_concept_ids=completed_concepts,
        budgets=tuple(budget_values),
        rng_by_stage=rng_by_stage,
        preparation_sha256=prepared.preparation_sha256,
        config_sha256=preset.config_sha256,
        selected_base_sha256=selected_base.selection_sha256,
        sweep_sha256=sweep_sha256,
    )


def publish_pilot_independent_sweep_stage(
    destination: Path,
    prepared: PreparedQueryAdaptation,
    artifact: QueryPartitionArtifact,
    selected_base: QuerySelectedBase,
    preset: QueryExperimentPreset,
    completed_concepts: tuple[str, ...],
    adapters_by_budget: Mapping[int, tuple[IndependentRootAdapter, ...]],
    rng_by_stage: tuple[np.ndarray, ...],
) -> PilotIndependentSweep:
    """Atomically persist one complete independent-sweep concept prefix."""
    _validate_bindings(prepared, artifact, preset)
    stage = len(completed_concepts)
    if (
        completed_concepts != preset.concept_ids[:stage]
        or not 1 <= stage <= preset.active_world_count
        or set(adapters_by_budget) != set(preset.pilot_update_budgets)
        or any(
            tuple(str(adapter.task_id) for adapter in adapters_by_budget[budget])
            != completed_concepts
            for budget in preset.pilot_update_budgets
        )
        or len(rng_by_stage) != stage
    ):
        raise ValueError("published pilot sweep stage changed its concept prefix")
    if destination.exists():
        return load_pilot_independent_sweep(
            destination,
            prepared,
            artifact,
            selected_base,
            preset,
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        tensors = {
            **{
                f"{_adapter_prefix(budget, stage)}.{name}": np.asarray(value)
                for budget in preset.pilot_update_budgets
                for stage, adapter in enumerate(adapters_by_budget[budget], start=1)
                for name, value in flatten_lora_edge(
                    adapter.adapter,
                    preset.model_config,
                    preset.lora_config,
                ).items()
            },
            **{
                f"trace.budget.{budget:04d}.stage.{stage:03d}": np.asarray(
                    adapter.step_losses,
                    dtype=np.float64,
                )
                for budget in preset.pilot_update_budgets
                for stage, adapter in enumerate(adapters_by_budget[budget], start=1)
            },
            **{
                f"rng.stage.{stage:03d}": np.asarray(rng, dtype=np.uint32)
                for stage, rng in enumerate(rng_by_stage, start=1)
            },
        }
        tensor_path = temporary / "sweep.safetensors"
        save_file(tensors, str(tensor_path), metadata={"format": PILOT_SWEEP_FORMAT})
        with tensor_path.open("rb") as source:
            os.fsync(source.fileno())
        core = {
            "budgets": [
                {
                    "tensor_checksum": _budget_tensor_checksum(tensors, budget),
                    "updates": budget,
                }
                for budget in preset.pilot_update_budgets
            ],
            "completed_concept_ids": list(completed_concepts),
            "config": preset.as_record(),
            "config_sha256": preset.config_sha256,
            "format": PILOT_SWEEP_FORMAT,
            "partition_sha256": artifact.partition_sha256,
            "preparation_sha256": prepared.preparation_sha256,
            "selected_base_sha256": selected_base.selection_sha256,
            "selected_checkpoint_manifest_sha256": selected_base.checkpoint.manifest_sha256,
            "selected_checkpoint_parameter_checksum": selected_base.checkpoint.parameter_checksum,
            "stage": len(completed_concepts),
            "tensor_checksum": _tensor_checksum(tensors),
            "tensor_file": tensor_path.name,
            "tensor_file_sha256": _file_sha256(tensor_path),
        }
        _write_file(
            temporary / "manifest.json",
            canonical_json_bytes({**core, "sweep_sha256": record_sha256(core)}),
        )
        _fsync_directory(temporary)
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        _remove_tree(temporary)
        raise
    return load_pilot_independent_sweep(
        destination,
        prepared,
        artifact,
        selected_base,
        preset,
    )


def _load_adapter(
    tensors: Mapping[str, np.ndarray],
    budget: int,
    stage: int,
    preset: QueryExperimentPreset,
    consumed: set[str],
) -> IndependentRootAdapter:
    prefix = _adapter_prefix(budget, stage) + "."
    adapter_tensors = {
        name.removeprefix(prefix): value
        for name, value in tensors.items()
        if name.startswith(prefix)
    }
    consumed.update(prefix + name for name in adapter_tensors)
    trace_name = f"trace.budget.{budget:04d}.stage.{stage:03d}"
    trace = _consume_tensor(tensors, trace_name, consumed)
    if trace.shape != (budget,) or trace.dtype != np.dtype(np.float64):
        raise ValueError("pilot sweep trace tensor changed")
    return IndependentRootAdapter(
        TaskId(preset.concept_ids[stage - 1]),
        unflatten_lora_edge(
            adapter_tensors,
            preset.model_config,
            preset.lora_config,
        ),
        tuple(float(value) for value in trace),
    )


def _adapter_prefix(budget: int, stage: int) -> str:
    return f"adapter.budget.{budget:04d}.stage.{stage:03d}"


def _budget_tensor_checksum(tensors: Mapping[str, np.ndarray], budget: int) -> str:
    markers = (f"adapter.budget.{budget:04d}.", f"trace.budget.{budget:04d}.")
    selected = {name: value for name, value in tensors.items() if name.startswith(markers)}
    if not selected:
        raise ValueError("pilot budget has no tensor payload")
    return _tensor_checksum(selected)


def _tensor_checksum(tensors: Mapping[str, np.ndarray]) -> str:
    digest = sha256()
    for name, value in sorted(tensors.items()):
        array = np.ascontiguousarray(value)
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _consume_tensor(
    tensors: Mapping[str, np.ndarray],
    name: str,
    consumed: set[str],
) -> np.ndarray:
    if name in consumed or name not in tensors:
        raise ValueError(f"pilot sweep tensor missing or duplicated: {name}")
    consumed.add(name)
    return np.asarray(tensors[name])


def _validate_bindings(
    prepared: PreparedQueryAdaptation,
    artifact: QueryPartitionArtifact,
    preset: QueryExperimentPreset,
) -> None:
    if (
        prepared.partition_sha256 != artifact.partition_sha256
        or prepared.catalog_sha256 != artifact.catalog_sha256
        or prepared.config_sha256 != preset.config_sha256
        or prepared.concept_ids != preset.concept_ids
        or artifact.concept_ids[: preset.active_world_count] != preset.concept_ids
    ):
        raise ValueError("pilot independent sweep bindings changed")


def _completed_stage_directories(root: Path, maximum: int) -> tuple[Path, ...]:
    candidates = tuple(sorted(root.glob("stage-[0-9][0-9][0-9]")))
    expected = tuple(root / f"stage-{stage:03d}" for stage in range(1, len(candidates) + 1))
    if candidates != expected or len(candidates) > maximum:
        raise ValueError("pilot sweep stages are not one contiguous prefix")
    return candidates


def _immutable_rng(value: np.ndarray) -> np.ndarray:
    rng = np.asarray(value, dtype=np.uint32).copy()
    if rng.shape != (2,):
        raise ValueError("pilot sweep RNG key changed")
    rng.flags.writeable = False
    return rng


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_file(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _string_list(record: dict[str, object], field: str) -> list[str]:
    value = record.get(field)
    if type(value) is not list or any(type(item) is not str for item in value):
        raise ValueError(f"pilot sweep {field} changed")
    return value


def _integer(record: dict[str, object], field: str) -> int:
    value = record.get(field)
    if type(value) is not int:
        raise ValueError(f"pilot sweep {field} must be an integer")
    return value


def _remove_tree(root: Path) -> None:
    if not root.exists():
        return
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            path.rmdir()
    root.rmdir()


__all__ = [
    "PILOT_SWEEP_FORMAT",
    "PilotIndependentBudget",
    "PilotIndependentSweep",
    "PilotSweepProgress",
    "load_pilot_independent_sweep",
    "publish_pilot_independent_sweep_stage",
    "train_or_resume_pilot_independent_sweep",
]

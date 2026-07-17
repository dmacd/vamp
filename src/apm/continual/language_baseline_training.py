"""Immutable training runs for the canonical language adaptation baselines."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from apm.continual.language_run import (
    LanguageVampRun,
    advance_language_vamp_run,
    init_language_vamp_run,
)
from apm.continual.language_tasks import (
    LanguageCurriculum,
    NodeId,
    RouterBatch,
    TaskId,
)
from apm.lm.checkpoint import BaseCheckpointRef, parameter_checksum
from apm.lm.config import GptNeoConfig
from apm.lm.lora import LoraConfig, LoraEdge, init_lora_edge
from apm.lm.lora_memory import PackedLoraMemory, pack_lora_memory
from apm.lm.parameters import GptNeoParams
from apm.lm.text_data import TokenBatch
from apm.lm.training import (
    LmTrainConfig,
    init_candidate_lora_train_state,
)
from apm.lm.workflow import run_candidate_edge_updates
from apm.memory.graph import MemoryGraph, add_memory_node, init_memory_graph


@dataclass(frozen=True)
class SequentialLoraStage:
    """Single shared adapter state after one sequential task budget."""

    stage_index: int
    task_id: TaskId
    adapter: LoraEdge
    step_losses: tuple[float, ...]

    def __post_init__(self) -> None:
        if type(self.stage_index) is not int or self.stage_index <= 0:
            raise ValueError("sequential stage_index must be positive")
        if not self.task_id or not self.step_losses:
            raise ValueError("sequential stage requires a task and losses")
        _validate_adapter_snapshot(self.adapter, self.step_losses)


@dataclass(frozen=True)
class SequentialLoraRun:
    """Ordered stages of the ordinary continually overwritten LoRA baseline."""

    stages: tuple[SequentialLoraStage, ...]
    rng_key: jax.Array
    train_config: LmTrainConfig
    base_parameter_checksum: str

    def __post_init__(self) -> None:
        if not self.stages or tuple(stage.stage_index for stage in self.stages) != tuple(
            range(1, len(self.stages) + 1)
        ):
            raise ValueError("sequential LoRA stages must be nonempty and contiguous")
        if len({stage.task_id for stage in self.stages}) != len(self.stages):
            raise ValueError("sequential LoRA stage task IDs must be unique")
        _validate_run_contract(
            tuple(stage.step_losses for stage in self.stages),
            self.rng_key,
            self.train_config,
            self.base_parameter_checksum,
        )


@dataclass(frozen=True)
class IndependentRootAdapter:
    """One independently initialized root LoRA trained for exactly one task."""

    task_id: TaskId
    adapter: LoraEdge
    step_losses: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.task_id or not self.step_losses:
            raise ValueError("independent root adapter requires a task and losses")
        _validate_adapter_snapshot(self.adapter, self.step_losses)


@dataclass(frozen=True)
class IndependentRootLoraRun:
    """Curriculum-ordered independent root adapters and final RNG stream."""

    adapters: tuple[IndependentRootAdapter, ...]
    rng_key: jax.Array
    train_config: LmTrainConfig
    base_parameter_checksum: str

    def __post_init__(self) -> None:
        task_ids = tuple(adapter.task_id for adapter in self.adapters)
        if not task_ids or len(set(task_ids)) != len(task_ids):
            raise ValueError("independent adapter task IDs must be nonempty and unique")
        _validate_run_contract(
            tuple(adapter.step_losses for adapter in self.adapters),
            self.rng_key,
            self.train_config,
            self.base_parameter_checksum,
        )


@dataclass(frozen=True)
class LanguageAdaptationBaselines:
    """All trained adapter states sharing one frozen base and update budget."""

    sequential_single_lora: SequentialLoraRun
    independent_root_lora: IndependentRootLoraRun
    vamp: LanguageVampRun
    train_config: LmTrainConfig
    base_parameter_checksum: str

    def __post_init__(self) -> None:
        if not isinstance(self.sequential_single_lora, SequentialLoraRun):
            raise TypeError("sequential_single_lora must be a SequentialLoraRun")
        if not isinstance(self.independent_root_lora, IndependentRootLoraRun):
            raise TypeError("independent_root_lora must be an IndependentRootLoraRun")
        if not isinstance(self.vamp, LanguageVampRun):
            raise TypeError("vamp must be a LanguageVampRun")
        if not isinstance(self.train_config, LmTrainConfig):
            raise TypeError("train_config must be an LmTrainConfig")
        if (
            self.sequential_single_lora.train_config != self.train_config
            or self.independent_root_lora.train_config != self.train_config
        ):
            raise ValueError("all adapter baselines must share one training config")
        checksums = (
            self.base_parameter_checksum,
            self.sequential_single_lora.base_parameter_checksum,
            self.independent_root_lora.base_parameter_checksum,
            self.vamp.base_checkpoint.parameter_checksum,
        )
        if len(set(checksums)) != 1:
            raise ValueError("all adapter baselines must share one frozen base hash")
        sequential_tasks = tuple(
            stage.task_id for stage in self.sequential_single_lora.stages
        )
        independent_tasks = tuple(
            adapter.task_id for adapter in self.independent_root_lora.adapters
        )
        vamp_tasks = tuple(task.task_id for task in self.vamp.completed_tasks)
        if sequential_tasks != independent_tasks or sequential_tasks != vamp_tasks:
            raise ValueError("all adapter baselines must share curriculum task order")
        if any(
            len(stage.candidate_step_losses) != self.train_config.steps
            for stage in self.vamp.stage_metrics
        ):
            raise ValueError("VAMP stages must use the shared per-task update budget")


class _TrainedRootAdapter(NamedTuple):
    adapter: LoraEdge
    rng_key: jax.Array
    step_losses: tuple[float, ...]


def train_sequential_single_lora(
    curriculum: LanguageCurriculum,
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    lora_config: LoraConfig,
    train_config: LmTrainConfig,
    rng_key: jax.Array,
) -> SequentialLoraRun:
    """Train one shared LoRA successively across every curriculum task."""
    _validate_training_contract(curriculum, train_config)
    _validate_rng_key(rng_key)
    base_checksum = parameter_checksum(base_params, model_config)
    initialization_key, current_rng_key = jax.random.split(rng_key)
    current_adapter = init_lora_edge(
        initialization_key,
        model_config,
        lora_config,
    )
    empty_memory = _empty_root_memory(model_config, lora_config)
    stages: list[SequentialLoraStage] = []
    for stage_index, task in enumerate(curriculum.tasks, start=1):
        trained = _train_root_adapter(
            current_adapter,
            current_rng_key,
            task.train_batches,
            base_params,
            model_config,
            empty_memory,
            lora_config,
            train_config,
        )
        current_adapter = trained.adapter
        current_rng_key = trained.rng_key
        stages.append(
            SequentialLoraStage(
                stage_index=stage_index,
                task_id=task.task_id,
                adapter=current_adapter,
                step_losses=trained.step_losses,
            )
        )
    _require_unchanged_base(base_checksum, base_params, model_config)
    return SequentialLoraRun(
        stages=tuple(stages),
        rng_key=current_rng_key,
        train_config=train_config,
        base_parameter_checksum=base_checksum,
    )


def train_independent_root_lora(
    curriculum: LanguageCurriculum,
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    lora_config: LoraConfig,
    train_config: LmTrainConfig,
    rng_key: jax.Array,
) -> IndependentRootLoraRun:
    """Train one fresh root adapter per task under the identical update budget."""
    _validate_training_contract(curriculum, train_config)
    _validate_rng_key(rng_key)
    base_checksum = parameter_checksum(base_params, model_config)
    empty_memory = _empty_root_memory(model_config, lora_config)
    current_rng_key = rng_key
    adapters: list[IndependentRootAdapter] = []
    for task in curriculum.tasks:
        initialization_key, training_key, current_rng_key = jax.random.split(
            current_rng_key,
            3,
        )
        initial_adapter = init_lora_edge(
            initialization_key,
            model_config,
            lora_config,
        )
        trained = _train_root_adapter(
            initial_adapter,
            training_key,
            task.train_batches,
            base_params,
            model_config,
            empty_memory,
            lora_config,
            train_config,
        )
        adapters.append(
            IndependentRootAdapter(
                task_id=task.task_id,
                adapter=trained.adapter,
                step_losses=trained.step_losses,
            )
        )
    _require_unchanged_base(base_checksum, base_params, model_config)
    return IndependentRootLoraRun(
        adapters=tuple(adapters),
        rng_key=current_rng_key,
        train_config=train_config,
        base_parameter_checksum=base_checksum,
    )


def train_language_adaptation_baselines(
    curriculum: LanguageCurriculum,
    root_validation_probes: tuple[RouterBatch, ...],
    base_checkpoint: BaseCheckpointRef,
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    lora_config: LoraConfig,
    train_config: LmTrainConfig,
    rng_key: jax.Array,
    *,
    evaluation_microbatch_size: int | None = None,
) -> LanguageAdaptationBaselines:
    """Train sequential, independent-root, and inherited VAMP adapters."""
    _validate_training_contract(curriculum, train_config)
    _validate_vamp_evaluation_inputs(curriculum, root_validation_probes)
    _validate_rng_key(rng_key)
    base_checksum = parameter_checksum(base_params, model_config)
    sequential_key, independent_key, vamp_key = jax.random.split(rng_key, 3)
    sequential = train_sequential_single_lora(
        curriculum,
        base_params,
        model_config,
        lora_config,
        train_config,
        sequential_key,
    )
    independent = train_independent_root_lora(
        curriculum,
        base_params,
        model_config,
        lora_config,
        train_config,
        independent_key,
    )
    vamp = init_language_vamp_run(
        base_checkpoint,
        base_params,
        model_config,
        root_validation_probes,
        vamp_key,
        max_nodes=curriculum.max_nodes,
        max_edges=curriculum.max_edges,
        key_probe_count=_router_row_count(root_validation_probes),
        evaluation_microbatch_size=evaluation_microbatch_size,
    )
    for task in curriculum.tasks:
        vamp = advance_language_vamp_run(
            vamp,
            task,
            base_params,
            model_config,
            lora_config,
            train_config,
            key_probe_count=sum(
                example.router_batch.input_ids.shape[0]
                for example in task.validation_examples
            ),
            evaluation_microbatch_size=evaluation_microbatch_size,
        )
    _require_unchanged_base(base_checksum, base_params, model_config)
    return LanguageAdaptationBaselines(
        sequential_single_lora=sequential,
        independent_root_lora=independent,
        vamp=vamp,
        train_config=train_config,
        base_parameter_checksum=base_checksum,
    )


def pack_root_adapter(
    adapter: LoraEdge,
    model_config: GptNeoConfig,
    lora_config: LoraConfig,
    *,
    node_id: NodeId = NodeId("adapter"),
) -> tuple[MemoryGraph[LoraEdge], PackedLoraMemory]:
    """Pack one root adapter into a two-node evaluation memory."""
    graph = add_memory_node(
        init_memory_graph(NodeId("root")),
        node_id,
        NodeId("root"),
        TaskId(str(node_id)),
        1,
        adapter,
    )
    return graph, pack_lora_memory(
        graph,
        model_config,
        lora_config,
        max_nodes=2,
        max_edges=1,
    )


def _empty_root_memory(
    model_config: GptNeoConfig,
    lora_config: LoraConfig,
) -> PackedLoraMemory:
    return pack_lora_memory(
        init_memory_graph(NodeId("root")),
        model_config,
        lora_config,
        max_nodes=2,
        max_edges=1,
    )


def _validate_training_contract(
    curriculum: LanguageCurriculum,
    train_config: LmTrainConfig,
) -> None:
    if not isinstance(curriculum, LanguageCurriculum) or not curriculum.tasks:
        raise ValueError("training requires a nonempty LanguageCurriculum")
    if not isinstance(train_config, LmTrainConfig):
        raise TypeError("train_config must be an LmTrainConfig")
    if any(
        batch.input_ids.shape[0] != train_config.batch_size
        for task in curriculum.tasks
        for batch in task.train_batches
    ):
        raise ValueError("every task batch must match the shared training batch size")
    sequence_widths = {
        batch.input_ids.shape[1]
        for task in curriculum.tasks
        for batch in task.train_batches
    }
    if len(sequence_widths) != 1:
        raise ValueError("all baseline batches must share one sequence width")


def _train_root_adapter(
    adapter: LoraEdge,
    rng_key: jax.Array,
    batches: tuple[TokenBatch, ...],
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    empty_memory: PackedLoraMemory,
    lora_config: LoraConfig,
    train_config: LmTrainConfig,
) -> _TrainedRootAdapter:
    state = init_candidate_lora_train_state(adapter, rng_key, train_config)
    trained_state, loss_trace = run_candidate_edge_updates(
        state,
        batches,
        base_params,
        model_config,
        empty_memory,
        lora_config,
        jnp.zeros((empty_memory.valid_edge_mask.shape[0],), dtype=jnp.float32),
        0,
        train_config,
    )
    return _TrainedRootAdapter(
        adapter=trained_state.trainable,
        rng_key=trained_state.rng_key,
        step_losses=loss_trace.step_losses,
    )


def _validate_adapter_snapshot(
    adapter: LoraEdge,
    step_losses: tuple[float, ...],
) -> None:
    if not isinstance(adapter, LoraEdge):
        raise TypeError("adapter snapshots must be LoraEdge values")
    if not isinstance(step_losses, tuple) or any(
        not math.isfinite(loss) or loss < 0.0 for loss in step_losses
    ):
        raise ValueError("adapter losses must be finite and nonnegative")


def _validate_run_contract(
    stage_losses: tuple[tuple[float, ...], ...],
    rng_key: jax.Array,
    train_config: LmTrainConfig,
    base_checksum: str,
) -> None:
    if not isinstance(train_config, LmTrainConfig):
        raise TypeError("train_config must be an LmTrainConfig")
    if any(len(losses) != train_config.steps for losses in stage_losses):
        raise ValueError("every task must use the configured update budget")
    _validate_rng_key(rng_key)
    if len(base_checksum) != 64 or any(
        character not in "0123456789abcdef" for character in base_checksum
    ):
        raise ValueError("base_parameter_checksum must be lowercase SHA-256")


def _validate_rng_key(rng_key: jax.Array) -> None:
    key = np.asarray(rng_key)
    if key.shape != (2,) or key.dtype != np.dtype(np.uint32):
        raise ValueError("rng_key must be a legacy uint32 JAX key")


def _require_unchanged_base(
    expected_checksum: str,
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
) -> None:
    if parameter_checksum(base_params, model_config) != expected_checksum:
        raise RuntimeError("adapter baseline training mutated the frozen base")


def _router_row_count(batches: tuple[RouterBatch, ...]) -> int:
    return sum(batch.input_ids.shape[0] for batch in batches)


def _validate_vamp_evaluation_inputs(
    curriculum: LanguageCurriculum,
    root_validation_probes: tuple[RouterBatch, ...],
) -> None:
    if not isinstance(root_validation_probes, tuple) or not root_validation_probes:
        raise ValueError("VAMP root probes must be a nonempty tuple")
    if any(not isinstance(batch, RouterBatch) for batch in root_validation_probes):
        raise TypeError("VAMP root probes must contain RouterBatch values")
    if any(
        not task.validation_examples
        or not task.test_examples
        or any(
            example.router_batch.input_ids.shape[0] != 1
            for example in task.test_examples
        )
        for task in curriculum.tasks
    ):
        raise ValueError("VAMP tasks require validation and single-row test examples")

"""Immutable orchestration for one rooted language VAMP run."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from apm.continual.language_metrics import (
    LanguageTaskMetrics,
    aggregate_language_task_metrics,
    evaluate_language_example,
)
from apm.continual.language_tasks import (
    AddressBook,
    LanguageTask,
    NodeId,
    RouterBatch,
    TaskId,
)
from apm.lm.checkpoint import BaseCheckpointRef, parameter_checksum
from apm.lm.config import GptNeoConfig
from apm.lm.lora import LoraConfig, LoraEdge, init_lora_edge
from apm.lm.lora_memory import edge_coefficients_for_node, pack_lora_memory
from apm.lm.parameters import GptNeoParams
from apm.lm.training import LmTrainConfig, init_candidate_lora_train_state
from apm.lm.workflow import run_candidate_edge_updates
from apm.memory.content_keys import add_address_key, derive_node_content_key
from apm.memory.graph import (
    MemoryGraph,
    add_memory_node,
    init_memory_graph,
    memory_node_ids,
)
from apm.memory.prefix_energy import exhaustive_prefix_nll_address


@dataclass(frozen=True)
class LanguageStageMetrics:
    """Immutable training, parent-probe, and evaluation results for one stage."""

    stage_index: int
    task_id: TaskId
    parent_node_index: int
    parent_node_id: NodeId
    parent_mean_node_nll: tuple[float, ...]
    candidate_step_losses: tuple[float, ...]
    task_metrics: tuple[LanguageTaskMetrics, ...]

    def __post_init__(self) -> None:
        if type(self.stage_index) is not int or self.stage_index <= 0:
            raise ValueError("stage_index must be a positive integer")
        if not self.task_id or not self.parent_node_id:
            raise ValueError("stage task and parent node IDs must not be empty")
        if type(self.parent_node_index) is not int or self.parent_node_index < 0:
            raise ValueError("parent_node_index must be a nonnegative integer")
        if not isinstance(self.parent_mean_node_nll, tuple) or not self.parent_mean_node_nll:
            raise ValueError("parent_mean_node_nll must be a nonempty tuple")
        if any(
            math.isnan(score) or score < 0.0
            for score in self.parent_mean_node_nll
        ):
            raise ValueError("parent node NLL values must be nonnegative or infinite")
        if not isinstance(self.candidate_step_losses, tuple) or not self.candidate_step_losses:
            raise ValueError("candidate_step_losses must be a nonempty tuple")
        if any(not math.isfinite(loss) or loss < 0.0 for loss in self.candidate_step_losses):
            raise ValueError("candidate step losses must be finite and nonnegative")
        if not isinstance(self.task_metrics, tuple):
            raise ValueError("task_metrics must be a tuple")
        if any(not isinstance(metric, LanguageTaskMetrics) for metric in self.task_metrics):
            raise TypeError("task_metrics must contain LanguageTaskMetrics values")


@dataclass(frozen=True, eq=False)
class LanguageVampRun:
    """Authoritative immutable state for a continual language-memory run."""

    base_checkpoint: BaseCheckpointRef
    graph: MemoryGraph[LoraEdge]
    address_book: AddressBook
    rng_key: jax.Array
    completed_tasks: tuple[LanguageTask, ...]
    stage_metrics: tuple[LanguageStageMetrics, ...]
    max_nodes: int
    max_edges: int

    def __post_init__(self) -> None:
        _validate_run(self)


@dataclass(frozen=True)
class ParentSearchResult:
    """Insertion-ordered parent scores and their deterministic selected node."""

    node_ids: tuple[NodeId, ...]
    mean_candidate_nll: tuple[float, ...]
    selected_node_index: int
    selected_node_id: NodeId
    scoring_basis: str

    def __post_init__(self) -> None:
        if not self.node_ids or len(set(self.node_ids)) != len(self.node_ids):
            raise ValueError("parent search node IDs must be nonempty and unique")
        if len(self.mean_candidate_nll) != len(self.node_ids) or any(
            not math.isfinite(score) or score < 0.0
            for score in self.mean_candidate_nll
        ):
            raise ValueError("parent search scores must be finite nonnegative NLLs")
        if (
            type(self.selected_node_index) is not int
            or not 0 <= self.selected_node_index < len(self.node_ids)
        ):
            raise ValueError("selected parent index is outside the candidate nodes")
        expected_index = int(np.argmin(np.asarray(self.mean_candidate_nll)))
        if self.selected_node_index != expected_index:
            raise ValueError("parent selection must use insertion-order argmin ties")
        if self.selected_node_id != self.node_ids[self.selected_node_index]:
            raise ValueError("selected parent node ID must match its index")
        if not self.scoring_basis:
            raise ValueError("parent search scoring_basis must not be empty")


def score_parent_nodes(
    run: LanguageVampRun,
    probes: tuple[RouterBatch, ...],
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    lora_config: LoraConfig,
    *,
    evaluation_microbatch_size: int | None = None,
) -> ParentSearchResult:
    """Score every current node by mean prefix NLL without changing run state."""
    if not isinstance(run, LanguageVampRun):
        raise TypeError("run must be a LanguageVampRun")
    _validate_base_checksum(run.base_checkpoint, base_params, model_config)
    aggregate = _aggregate_router_batches(
        probes,
        expected_row_count=sum(probe.input_ids.shape[0] for probe in probes),
    )
    packed_memory = pack_lora_memory(
        run.graph,
        model_config,
        lora_config,
        run.max_nodes,
        run.max_edges,
    )
    scores = exhaustive_prefix_nll_address(
        base_params,
        model_config,
        packed_memory,
        lora_config,
        aggregate,
        evaluation_microbatch_size=evaluation_microbatch_size,
    )
    padded_means = _mean_parent_node_nll(
        scores.node_scores,
        len(run.graph.nodes),
        run.max_nodes,
    )
    valid_means = padded_means[: len(run.graph.nodes)]
    selected_index = int(np.argmin(np.asarray(valid_means)))
    node_ids = memory_node_ids(run.graph)
    return ParentSearchResult(
        node_ids=node_ids,
        mean_candidate_nll=valid_means,
        selected_node_index=selected_index,
        selected_node_id=node_ids[selected_index],
        scoring_basis="mean_prefix_nll",
    )


def init_language_vamp_run(
    base_checkpoint: BaseCheckpointRef,
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    root_validation_probes: tuple[RouterBatch, ...],
    rng_key: jax.Array,
    *,
    max_nodes: int,
    max_edges: int,
    key_probe_count: int = 256,
    root_id: NodeId = NodeId("root"),
    evaluation_microbatch_size: int | None = None,
) -> LanguageVampRun:
    """Initialize a root-only run and derive its key from frozen-base probes."""
    _validate_capacities(max_nodes, max_edges)
    _validate_base_checksum(base_checkpoint, base_params, model_config)
    if not root_id:
        raise ValueError("root_id must not be empty")
    probes = _aggregate_router_batches(
        root_validation_probes,
        expected_row_count=key_probe_count,
    )
    root_key = derive_node_content_key(
        base_params,
        model_config,
        jnp.asarray(probes.input_ids),
        jnp.asarray(probes.attention_mask),
        expected_probe_count=key_probe_count,
        evaluation_microbatch_size=evaluation_microbatch_size,
    )
    address_book = add_address_key(
        AddressBook(
            node_ids=(None,) * max_nodes,
            keys=np.zeros((max_nodes, model_config.hidden_size), dtype=np.float32),
            valid_node_mask=np.zeros((max_nodes,), dtype=np.bool_),
        ),
        node_index=0,
        node_id=root_id,
        key=root_key,
    )
    return LanguageVampRun(
        base_checkpoint=base_checkpoint,
        graph=init_memory_graph(root_id),
        address_book=address_book,
        rng_key=rng_key,
        completed_tasks=(),
        stage_metrics=(),
        max_nodes=max_nodes,
        max_edges=max_edges,
    )


def advance_language_vamp_run(
    run: LanguageVampRun,
    task: LanguageTask,
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    lora_config: LoraConfig,
    train_config: LmTrainConfig,
    parent_selection: ParentSearchResult,
    *,
    key_probe_count: int = 256,
    evaluation_microbatch_size: int | None = None,
    training_progress: Callable[[int, float, int], None] | None = None,
) -> LanguageVampRun:
    """Train and commit exactly one candidate edge without mutating prior state."""
    if not isinstance(run, LanguageVampRun):
        raise TypeError("run must be a LanguageVampRun")
    if not isinstance(task, LanguageTask):
        raise TypeError("task must be a LanguageTask")
    _validate_base_checksum(run.base_checkpoint, base_params, model_config)
    if len(run.graph.nodes) >= run.max_nodes:
        raise ValueError("language VAMP run has no remaining node capacity")

    node_id = NodeId(str(task.task_id))
    root_id = run.graph.nodes[0].node_id
    if node_id == root_id:
        raise ValueError("a task node cannot collide with the graph root")
    if node_id in memory_node_ids(run.graph):
        raise ValueError(f"memory node already exists for task: {task.task_id}")
    if task.task_id in tuple(completed.task_id for completed in run.completed_tasks):
        raise ValueError(f"language task is already complete: {task.task_id}")
    if any(
        example.oracle_node_id != node_id
        for example in task.validation_examples + task.test_examples
    ):
        raise ValueError("task evaluation oracle IDs must equal NodeId(str(task.task_id))")

    packed_memory = pack_lora_memory(
        run.graph,
        model_config,
        lora_config,
        run.max_nodes,
        run.max_edges,
    )
    if not isinstance(parent_selection, ParentSearchResult):
        raise TypeError("parent_selection must be a ParentSearchResult")
    if parent_selection.node_ids != memory_node_ids(run.graph):
        raise ValueError("parent selection candidates do not match the current graph")
    parent_node_index = parent_selection.selected_node_index
    parent_node_id = parent_selection.selected_node_id
    parent_mean_node_nll = parent_selection.mean_candidate_nll + (
        (math.inf,) * (run.max_nodes - len(run.graph.nodes))
    )

    candidate_index = len(run.graph.nodes) - 1
    candidate_rng_key, training_rng_key = jax.random.split(run.rng_key)
    candidate_edge = init_lora_edge(candidate_rng_key, model_config, lora_config)
    candidate_state = init_candidate_lora_train_state(
        candidate_edge,
        training_rng_key,
        train_config,
    )
    trained_state, loss_trace = run_candidate_edge_updates(
        candidate_state,
        task.train_batches,
        base_params,
        model_config,
        packed_memory,
        lora_config,
        edge_coefficients_for_node(packed_memory, parent_node_index),
        candidate_index,
        train_config,
        progress=training_progress,
    )

    stage_index = len(run.completed_tasks) + 1
    graph = add_memory_node(
        run.graph,
        node_id=node_id,
        parent_id=parent_node_id,
        trained_task=task.task_id,
        train_stage=stage_index,
        incoming_edge=trained_state.trainable,
    )
    content_key_probes = _aggregate_router_batches(
        task.content_key_probes,
        expected_row_count=sum(
            probe.input_ids.shape[0] for probe in task.content_key_probes
        ),
    )
    content_key = derive_node_content_key(
        base_params,
        model_config,
        jnp.asarray(content_key_probes.input_ids),
        jnp.asarray(content_key_probes.attention_mask),
        expected_probe_count=content_key_probes.input_ids.shape[0],
        evaluation_microbatch_size=evaluation_microbatch_size,
    )
    address_book = add_address_key(
        run.address_book,
        node_index=len(run.graph.nodes),
        node_id=node_id,
        key=content_key,
    )
    completed_tasks = run.completed_tasks + (task,)
    committed_memory = pack_lora_memory(
        graph,
        model_config,
        lora_config,
        run.max_nodes,
        run.max_edges,
    )
    task_metrics = tuple(
        aggregate_language_task_metrics(
            completed_task.task_id,
            tuple(
                evaluate_language_example(
                    base_params,
                    model_config,
                    graph,
                    committed_memory,
                    lora_config,
                    example,
                    evaluation_microbatch_size=evaluation_microbatch_size,
                )
                for example in completed_task.test_examples
            ),
        )
        for completed_task in completed_tasks
        if completed_task.test_examples
    )
    metrics = LanguageStageMetrics(
        stage_index=stage_index,
        task_id=task.task_id,
        parent_node_index=parent_node_index,
        parent_node_id=parent_node_id,
        parent_mean_node_nll=parent_mean_node_nll,
        candidate_step_losses=loss_trace.step_losses,
        task_metrics=task_metrics,
    )
    return LanguageVampRun(
        base_checkpoint=run.base_checkpoint,
        graph=graph,
        address_book=address_book,
        rng_key=trained_state.rng_key,
        completed_tasks=completed_tasks,
        stage_metrics=run.stage_metrics + (metrics,),
        max_nodes=run.max_nodes,
        max_edges=run.max_edges,
    )


def _aggregate_router_batches(
    batches: tuple[RouterBatch, ...],
    *,
    expected_row_count: int,
) -> RouterBatch:
    if type(expected_row_count) is not int or expected_row_count <= 0:
        raise ValueError("key_probe_count must be a positive integer")
    if not isinstance(batches, tuple) or not batches:
        raise ValueError("validation probes must contain RouterBatch values")
    if any(not isinstance(batch, RouterBatch) for batch in batches):
        raise TypeError("validation probes must contain RouterBatch values")
    widths = tuple(batch.input_ids.shape[1] for batch in batches)
    if len(set(widths)) != 1:
        raise ValueError("validation router batches must share one sequence width")
    row_count = sum(batch.input_ids.shape[0] for batch in batches)
    if row_count != expected_row_count:
        raise ValueError(
            f"validation probes contain {row_count} rows; "
            f"expected exactly {expected_row_count}"
        )
    return RouterBatch(
        input_ids=np.concatenate(tuple(batch.input_ids for batch in batches), axis=0),
        attention_mask=np.concatenate(
            tuple(batch.attention_mask for batch in batches),
            axis=0,
        ),
        target_ids=np.concatenate(tuple(batch.target_ids for batch in batches), axis=0),
        loss_mask=np.concatenate(tuple(batch.loss_mask for batch in batches), axis=0),
    )


def _mean_parent_node_nll(
    node_scores: jax.Array,
    valid_node_count: int,
    max_nodes: int,
) -> tuple[float, ...]:
    scores = np.asarray(node_scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != max_nodes or scores.shape[0] < 1:
        raise ValueError("parent probe returned an invalid node-score shape")
    valid_scores = scores[:, :valid_node_count]
    if not np.all(np.isfinite(valid_scores)) or np.any(valid_scores < 0.0):
        raise ValueError("parent probe returned invalid NLL values for valid nodes")
    mean_scores = np.full((max_nodes,), np.inf, dtype=np.float64)
    mean_scores[:valid_node_count] = np.mean(valid_scores, axis=0)
    return tuple(float(score) for score in mean_scores)


def _validate_capacities(max_nodes: int, max_edges: int) -> None:
    if type(max_nodes) is not int or max_nodes < 1:
        raise ValueError("max_nodes must be a positive integer")
    if type(max_edges) is not int or max_edges != max_nodes - 1:
        raise ValueError("max_edges must equal max_nodes - 1")


def _validate_base_checksum(
    base_checkpoint: BaseCheckpointRef,
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
) -> None:
    if not isinstance(base_checkpoint, BaseCheckpointRef):
        raise TypeError("base_checkpoint must be a BaseCheckpointRef")
    if parameter_checksum(base_params, model_config) != base_checkpoint.parameter_checksum:
        raise ValueError("base parameters do not match the frozen checkpoint reference")


def _validate_run(run: LanguageVampRun) -> None:
    if not isinstance(run.base_checkpoint, BaseCheckpointRef):
        raise TypeError("base_checkpoint must be a BaseCheckpointRef")
    if not isinstance(run.graph, MemoryGraph) or not run.graph.nodes:
        raise ValueError("graph must be a nonempty MemoryGraph")
    if not isinstance(run.address_book, AddressBook):
        raise TypeError("address_book must be an AddressBook")
    if not isinstance(run.completed_tasks, tuple) or any(
        not isinstance(task, LanguageTask) for task in run.completed_tasks
    ):
        raise TypeError("completed_tasks must contain LanguageTask values")
    if not isinstance(run.stage_metrics, tuple) or any(
        not isinstance(metrics, LanguageStageMetrics) for metrics in run.stage_metrics
    ):
        raise TypeError("stage_metrics must contain LanguageStageMetrics values")
    _validate_capacities(run.max_nodes, run.max_edges)
    if len(run.graph.nodes) > run.max_nodes:
        raise ValueError("graph node count exceeds run capacity")
    if run.address_book.max_nodes != run.max_nodes:
        raise ValueError("address-book capacity must match run capacity")
    if len(run.completed_tasks) != len(run.graph.nodes) - 1:
        raise ValueError("completed task count must equal the non-root node count")
    if len(run.stage_metrics) != len(run.completed_tasks):
        raise ValueError("stage metric count must equal the completed task count")
    _validate_run_rng_key(run.rng_key)

    node_ids = memory_node_ids(run.graph)
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("graph node IDs must be unique")
    expected_valid_mask = np.arange(run.max_nodes) < len(node_ids)
    if not np.array_equal(run.address_book.valid_node_mask, expected_valid_mask):
        raise ValueError("address-book validity must match graph insertion order")
    if run.address_book.node_ids[: len(node_ids)] != node_ids:
        raise ValueError("address-book node IDs must match graph insertion order")
    valid_key_norms = np.linalg.norm(
        run.address_book.keys[: len(node_ids)],
        axis=1,
    )
    if not np.allclose(valid_key_norms, 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("valid address keys must be L2 normalized")

    root, *non_root_nodes = run.graph.nodes
    if (
        root.parent_id is not None
        or root.trained_task is not None
        or root.train_stage != 0
        or root.depth != 0
        or root.incoming_edge is not None
    ):
        raise ValueError("graph root metadata is invalid")
    prior_ids = {root.node_id}
    for node_index, (node, task, metrics) in enumerate(
        zip(non_root_nodes, run.completed_tasks, run.stage_metrics),
        start=1,
    ):
        if node.parent_id not in prior_ids:
            raise ValueError("every graph parent must precede its child")
        parent = run.graph.nodes[node_ids.index(node.parent_id)]
        if node.depth != parent.depth + 1:
            raise ValueError("graph node depth must follow its parent")
        if not isinstance(node.incoming_edge, LoraEdge):
            raise TypeError("every non-root node must contain one LoraEdge")
        if (
            node.node_id != NodeId(str(task.task_id))
            or node.trained_task != task.task_id
            or node.train_stage != node_index
        ):
            raise ValueError("graph nodes must align with completed tasks in stage order")
        if (
            metrics.stage_index != node_index
            or metrics.task_id != task.task_id
            or metrics.parent_node_id != node.parent_id
            or metrics.parent_node_index >= node_index
            or run.graph.nodes[metrics.parent_node_index].node_id
            != metrics.parent_node_id
        ):
            raise ValueError("stage metrics must align with the committed graph node")
        if len(metrics.parent_mean_node_nll) != run.max_nodes:
            raise ValueError("parent score capacity must match run node capacity")
        if not all(math.isfinite(value) for value in metrics.parent_mean_node_nll[:node_index]):
            raise ValueError("parent scores for preexisting nodes must be finite")
        if not all(math.isinf(value) for value in metrics.parent_mean_node_nll[node_index:]):
            raise ValueError("parent scores for padded nodes must be infinite")
        expected_metric_tasks = tuple(
            completed.task_id for completed in run.completed_tasks[:node_index]
        )
        if tuple(metric.task_id for metric in metrics.task_metrics) != expected_metric_tasks:
            raise ValueError("stage task metrics must cover all completed tasks in order")
        prior_ids.add(node.node_id)


def _validate_run_rng_key(rng_key: jax.Array) -> None:
    try:
        key_data = np.asarray(jax.random.key_data(rng_key))
    except (TypeError, ValueError) as error:
        raise TypeError("rng_key must be one unbatched JAX PRNG key") from error
    if key_data.shape != (2,) or key_data.dtype != np.uint32:
        raise ValueError("rng_key must be one unbatched JAX PRNG key")

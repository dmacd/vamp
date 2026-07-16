"""Pure competence evaluation and aggregate metrics for language routing."""

from __future__ import annotations

from dataclasses import dataclass
import math

import jax
import jax.numpy as jnp
import numpy as np

from apm.continual.language_tasks import (
    CompetenceBatch,
    LanguageEvaluationExample,
    NodeId,
    TaskId,
)
from apm.lm.config import GptNeoConfig
from apm.lm.gpt_neo import apply_gpt_neo
from apm.lm.lora import LoraConfig, LoraEdge
from apm.lm.lora_memory import PackedLoraMemory, edge_coefficients_for_node
from apm.lm.losses import per_token_nll
from apm.lm.parameters import GptNeoParams
from apm.memory.graph import MemoryGraph
from apm.memory.prefix_energy import exhaustive_prefix_nll_address


@dataclass(frozen=True)
class LanguageExampleMetric:
    """Oracle and task-free competence plus routing quality for one example."""

    oracle_node_id: NodeId
    selected_node_id: NodeId
    oracle_competence_nll: float
    task_free_competence_nll: float
    routing_correct: bool
    routing_regret: float
    valid_suffix_tokens: int

    def __post_init__(self) -> None:
        if any(
            not isinstance(node_id, str) or not node_id
            for node_id in (self.oracle_node_id, self.selected_node_id)
        ):
            raise ValueError("metric node IDs must not be empty")
        if type(self.routing_correct) is not bool:
            raise TypeError("routing_correct must be a bool")
        if self.routing_correct != (self.selected_node_id == self.oracle_node_id):
            raise ValueError("routing_correct must match the selected and oracle IDs")
        if type(self.valid_suffix_tokens) is not int or self.valid_suffix_tokens <= 0:
            raise ValueError("valid_suffix_tokens must be a positive integer")
        _validate_nll(self.oracle_competence_nll, "oracle_competence_nll")
        _validate_nll(
            self.task_free_competence_nll,
            "task_free_competence_nll",
        )
        if not math.isfinite(self.routing_regret):
            raise ValueError("routing_regret must be finite")
        expected_regret = self.task_free_competence_nll - self.oracle_competence_nll
        if not math.isclose(self.routing_regret, expected_regret, abs_tol=1e-7):
            raise ValueError("routing_regret must equal task-free NLL minus oracle NLL")


@dataclass(frozen=True)
class LanguageTaskMetrics:
    """Token-weighted competence and per-example routing metrics for one task."""

    task_id: TaskId
    example_metrics: tuple[LanguageExampleMetric, ...]
    oracle_competence_nll: float
    task_free_competence_nll: float
    routing_accuracy: float
    routing_regret: float
    valid_suffix_tokens: int

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError("metric task_id must not be empty")
        if not isinstance(self.example_metrics, tuple) or not self.example_metrics:
            raise ValueError("task metrics must contain example metrics")
        if any(
            not isinstance(metric, LanguageExampleMetric)
            for metric in self.example_metrics
        ):
            raise TypeError("example_metrics must contain LanguageExampleMetric values")
        if type(self.valid_suffix_tokens) is not int or self.valid_suffix_tokens <= 0:
            raise ValueError("valid_suffix_tokens must be a positive integer")
        if self.valid_suffix_tokens != sum(
            metric.valid_suffix_tokens for metric in self.example_metrics
        ):
            raise ValueError("task valid_suffix_tokens must equal the example total")
        _validate_nll(self.oracle_competence_nll, "oracle_competence_nll")
        _validate_nll(
            self.task_free_competence_nll,
            "task_free_competence_nll",
        )
        if not math.isfinite(self.routing_accuracy) or not 0.0 <= self.routing_accuracy <= 1.0:
            raise ValueError("routing_accuracy must be finite and in [0, 1]")
        if not math.isfinite(self.routing_regret):
            raise ValueError("routing_regret must be finite")
        expected_regret = self.task_free_competence_nll - self.oracle_competence_nll
        if not math.isclose(self.routing_regret, expected_regret, abs_tol=1e-7):
            raise ValueError("routing_regret must equal task-free NLL minus oracle NLL")
        expected_oracle_nll = _token_weighted_mean(
            tuple(metric.oracle_competence_nll for metric in self.example_metrics),
            self.example_metrics,
            self.valid_suffix_tokens,
        )
        expected_task_free_nll = _token_weighted_mean(
            tuple(
                metric.task_free_competence_nll
                for metric in self.example_metrics
            ),
            self.example_metrics,
            self.valid_suffix_tokens,
        )
        expected_accuracy = sum(
            float(metric.routing_correct) for metric in self.example_metrics
        ) / len(self.example_metrics)
        aggregates = (
            (self.oracle_competence_nll, expected_oracle_nll),
            (self.task_free_competence_nll, expected_task_free_nll),
            (self.routing_accuracy, expected_accuracy),
        )
        if any(
            not math.isclose(actual, expected, abs_tol=1e-7)
            for actual, expected in aggregates
        ):
            raise ValueError("task metric aggregates do not match example metrics")


def hard_node_competence_nll(
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    packed_memory: PackedLoraMemory,
    lora_config: LoraConfig,
    competence_batch: CompetenceBatch,
    node_index: int,
) -> jax.Array:
    """Return suffix-token-normalized NLL for one valid hard node path."""
    if type(node_index) is not int:
        raise TypeError("node_index must be an integer")
    max_nodes = packed_memory.node_path_matrix.shape[0]
    if not 0 <= node_index < max_nodes:
        raise ValueError("node_index is outside packed-memory capacity")
    if not bool(np.asarray(packed_memory.valid_node_mask[node_index])):
        raise ValueError("node_index must identify a valid packed-memory node")
    loss_mask = jnp.asarray(competence_batch.loss_mask, dtype=jnp.float32)
    valid_suffix_tokens = jnp.sum(loss_mask)
    if int(np.asarray(valid_suffix_tokens)) <= 0:
        raise ValueError("competence batch must enable at least one suffix token")
    edge_coefficients = edge_coefficients_for_node(packed_memory, node_index)
    logits = apply_gpt_neo(
        base_params,
        model_config,
        jnp.asarray(competence_batch.input_ids),
        jnp.asarray(competence_batch.attention_mask),
        lora_memory=packed_memory,
        edge_coefficients=edge_coefficients,
        lora_config=lora_config,
        training=False,
    ).logits
    token_losses = per_token_nll(
        logits,
        jnp.asarray(competence_batch.target_ids),
    )
    return jnp.sum(token_losses * loss_mask) / valid_suffix_tokens


def resolve_node_index(
    graph: MemoryGraph[LoraEdge],
    node_id: NodeId,
) -> int:
    """Resolve one node ID to its authoritative insertion-order graph index."""
    matches = tuple(
        index
        for index, node in enumerate(graph.nodes)
        if node.node_id == node_id
    )
    if not matches:
        raise KeyError(f"unknown memory node ID: {node_id}")
    if len(matches) != 1:
        raise ValueError(f"memory node ID is not unique: {node_id}")
    return matches[0]


def evaluate_language_example(
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    graph: MemoryGraph[LoraEdge],
    packed_memory: PackedLoraMemory,
    lora_config: LoraConfig,
    example: LanguageEvaluationExample,
) -> LanguageExampleMetric:
    """Route on the prefix, then compare selected and oracle suffix competence."""
    _validate_graph_memory_alignment(graph, packed_memory)
    if example.router_batch.input_ids.shape[0] != 1:
        raise ValueError("each LanguageEvaluationExample must contain exactly one row")
    address_result = exhaustive_prefix_nll_address(
        base_params,
        model_config,
        packed_memory,
        lora_config,
        example.router_batch,
    )
    selected_indices = np.asarray(address_result.selected_indices)
    if selected_indices.shape != (1,):
        raise ValueError("exhaustive routing must return one selected index")
    selected_index = int(selected_indices[0])
    if not 0 <= selected_index < len(graph.nodes):
        raise ValueError("exhaustive routing selected a node outside the graph")
    oracle_index = resolve_node_index(graph, example.oracle_node_id)
    selected_node_id = graph.nodes[selected_index].node_id
    oracle_nll = float(
        np.asarray(
            hard_node_competence_nll(
                base_params,
                model_config,
                packed_memory,
                lora_config,
                example.competence_batch,
                oracle_index,
            )
        )
    )
    task_free_nll = (
        oracle_nll
        if selected_index == oracle_index
        else float(
            np.asarray(
                hard_node_competence_nll(
                    base_params,
                    model_config,
                    packed_memory,
                    lora_config,
                    example.competence_batch,
                    selected_index,
                )
            )
        )
    )
    routing_regret = task_free_nll - oracle_nll
    return LanguageExampleMetric(
        oracle_node_id=example.oracle_node_id,
        selected_node_id=selected_node_id,
        oracle_competence_nll=oracle_nll,
        task_free_competence_nll=task_free_nll,
        routing_correct=selected_node_id == example.oracle_node_id,
        routing_regret=routing_regret,
        valid_suffix_tokens=int(np.sum(example.competence_batch.loss_mask)),
    )


def aggregate_language_task_metrics(
    task_id: TaskId,
    example_metrics: tuple[LanguageExampleMetric, ...],
) -> LanguageTaskMetrics:
    """Aggregate competence token-weighted and routing accuracy per example."""
    if not example_metrics:
        raise ValueError("cannot aggregate an empty metric sequence")
    total_tokens = sum(metric.valid_suffix_tokens for metric in example_metrics)
    oracle_nll = _token_weighted_mean(
        tuple(metric.oracle_competence_nll for metric in example_metrics),
        example_metrics,
        total_tokens,
    )
    task_free_nll = _token_weighted_mean(
        tuple(metric.task_free_competence_nll for metric in example_metrics),
        example_metrics,
        total_tokens,
    )
    return LanguageTaskMetrics(
        task_id=task_id,
        example_metrics=example_metrics,
        oracle_competence_nll=oracle_nll,
        task_free_competence_nll=task_free_nll,
        routing_accuracy=sum(
            float(metric.routing_correct) for metric in example_metrics
        )
        / len(example_metrics),
        routing_regret=task_free_nll - oracle_nll,
        valid_suffix_tokens=total_tokens,
    )


def _validate_graph_memory_alignment(
    graph: MemoryGraph[LoraEdge],
    packed_memory: PackedLoraMemory,
) -> None:
    max_nodes = packed_memory.node_path_matrix.shape[0]
    valid_node_mask = np.asarray(packed_memory.valid_node_mask, dtype=np.bool_)
    if valid_node_mask.shape != (max_nodes,):
        raise ValueError("packed valid-node mask does not match node capacity")
    if len(graph.nodes) > max_nodes:
        raise ValueError("graph node count exceeds packed-memory capacity")
    expected_mask = np.arange(max_nodes) < len(graph.nodes)
    if not np.array_equal(valid_node_mask, expected_mask):
        raise ValueError("packed valid nodes do not match graph insertion order")


def _token_weighted_mean(
    values: tuple[float, ...],
    metrics: tuple[LanguageExampleMetric, ...],
    total_tokens: int,
) -> float:
    return sum(
        value * metric.valid_suffix_tokens
        for value, metric in zip(values, metrics)
    ) / total_tokens


def _validate_nll(value: float, field_name: str) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{field_name} must be finite and nonnegative")

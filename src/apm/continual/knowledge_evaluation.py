"""Exact four-candidate knowledge evaluation over shared VAMP node scores."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Callable

import numpy as np

from apm.continual.knowledge_tasks import KnowledgeQuery
from apm.continual.language_tasks import AddressBook, RouterBatch
from apm.lm.candidate_scoring import score_edge_coefficient_candidates
from apm.lm.config import GptNeoConfig
from apm.lm.lora import LoraConfig
from apm.lm.lora_memory import PackedLoraMemory
from apm.lm.parameters import GptNeoParams
from apm.memory.graph import (
    MemoryGraph,
    memory_edge_node_ids,
    memory_node_ids,
    path_incidence_matrix,
)

if TYPE_CHECKING:
    from apm.continual.language_routing import LanguageAddressDecision
    from apm.memory.address_refinement import EbtConfig
    from apm.memory.content_addressing import HopfieldConfig


KNOWLEDGE_AGGREGATION_AXES: tuple[str, ...] = (
    "all",
    "task_id",
    "family_id",
    "query_kind",
    "prefix_length",
    "cue_regime",
    "reasoning_type",
    "reasoning_depth",
    "novelty_regime",
    "mode",
)
_EBT_ROUTERS: tuple[str, ...] = (
    "vamp_ebt_uniform",
    "vamp_ebt_hopfield",
)
_ROUTER_TOP_K = 4
_CANDIDATE_COUNT = 4


@dataclass(frozen=True, eq=False)
class KnowledgeAddressDecision:
    """Immutable hard choice and uncertainty from a task-free router."""

    selected_indices: np.ndarray
    node_probabilities: np.ndarray
    node_scores: np.ndarray
    score_margin: np.ndarray
    entropy: np.ndarray
    top_k_indices: np.ndarray

    def __post_init__(self) -> None:
        selected = _immutable_integer_array(
            self.selected_indices,
            "selected_indices",
            ndim=1,
        )
        probabilities = _immutable_float_array(
            self.node_probabilities,
            "node_probabilities",
            ndim=2,
        )
        scores = _immutable_float_array(
            self.node_scores,
            "node_scores",
            ndim=2,
        )
        margins = _immutable_float_array(
            self.score_margin,
            "score_margin",
            ndim=1,
        )
        entropy = _immutable_float_array(
            self.entropy,
            "entropy",
            ndim=1,
        )
        top_k = _immutable_integer_array(
            self.top_k_indices,
            "top_k_indices",
            ndim=2,
        )
        batch_size = selected.shape[0]
        if (
            probabilities.shape != scores.shape
            or probabilities.shape[0] != batch_size
        ):
            raise ValueError("address probabilities and scores must be [query, node]")
        node_capacity = probabilities.shape[1]
        if node_capacity < 1:
            raise ValueError("address decisions require at least one node")
        if margins.shape != (batch_size,) or entropy.shape != (batch_size,):
            raise ValueError("address margins and entropy must be [query]")
        if top_k.shape[0] != batch_size or not 1 <= top_k.shape[1] <= node_capacity:
            raise ValueError("top-k indices must be a nonempty [query, k] array")
        if np.any(~np.isfinite(probabilities)) or np.any(
            (probabilities < 0.0) | (probabilities > 1.0)
        ):
            raise ValueError("node probabilities must be finite and in [0, 1]")
        if not np.allclose(np.sum(probabilities, axis=1), 1.0, atol=1e-6):
            raise ValueError("node probability rows must sum to one")
        if np.any(np.isnan(scores)) or np.any(np.isposinf(scores)):
            raise ValueError(
                "node scores may contain finite values or negative infinity"
            )
        if np.any(np.isnan(margins)) or np.any(margins < 0.0):
            raise ValueError("score margins must be nonnegative")
        if np.any(~np.isfinite(entropy)) or np.any(entropy < 0.0):
            raise ValueError("address entropy must be finite and nonnegative")
        if np.any((selected < 0) | (selected >= node_capacity)):
            raise ValueError("selected indices must lie within node capacity")
        if np.any((top_k < 0) | (top_k >= node_capacity)):
            raise ValueError("top-k indices must lie within node capacity")
        if any(len(set(row.tolist())) != row.size for row in top_k):
            raise ValueError("top-k indices must be unique within each query")
        if np.any(top_k[:, 0] != selected):
            raise ValueError("the selected node must lead each top-k row")
        if np.any(np.argmax(scores, axis=1) != selected) or np.any(
            np.argmax(probabilities, axis=1) != selected
        ):
            raise ValueError("selected indices must maximize scores and probabilities")
        expected_entropy = -np.sum(
            probabilities
            * np.log(np.where(probabilities > 0.0, probabilities, 1.0)),
            axis=1,
        )
        if not np.allclose(entropy, expected_entropy, atol=1e-6):
            raise ValueError("address entropy must match node probabilities")
        object.__setattr__(self, "selected_indices", selected)
        object.__setattr__(self, "node_probabilities", probabilities)
        object.__setattr__(self, "node_scores", scores)
        object.__setattr__(self, "score_margin", margins)
        object.__setattr__(self, "entropy", entropy)
        object.__setattr__(self, "top_k_indices", top_k)


@dataclass(frozen=True, eq=False)
class KnowledgeQueryEvaluation:
    """One method's exact candidate, routing, regret, and support result."""

    stage: int
    method: str
    query_id: str
    task_id: str
    family_id: str
    query_kind: str
    proof_id: str
    support_ids: tuple[str, ...]
    required_edge_ids: tuple[str, ...]
    cue_regime: str
    visible_cue_ids: tuple[str, ...]
    eligible_task_ids: tuple[str, ...]
    novelty_regime: str
    reasoning_type: str
    reasoning_depth: int
    prefix_length: int
    mode: str
    oracle_node_ids: tuple[str, ...]
    candidate_answer_texts: tuple[str, ...]
    candidate_nll: np.ndarray
    correct_candidate_index: int
    predicted_candidate_index: int
    candidate_correct: bool
    candidate_margin: float
    correct_answer_nll: float
    selected_node_index: int | None
    task_oracle_node_index: int | None
    best_hard_node_index: int
    routed_correct_answer_nll: float | None
    task_oracle_correct_answer_nll: float | None
    best_hard_node_correct_answer_nll: float
    routed_regret: float | None
    task_oracle_regret: float | None
    best_hard_node_regret: float
    node_accuracy: bool | None
    top_k_accuracy: bool | None
    address_entropy: float | None
    address_margin: float | None
    hard_required_edge_recall: float | None
    soft_required_edge_mean_coefficient: float | None

    def __post_init__(self) -> None:
        _validate_stage_method(self.stage, self.method)
        if any(
            not isinstance(value, str) or not value
            for value in (
                self.query_id,
                self.task_id,
                self.family_id,
                self.query_kind,
                self.proof_id,
                self.cue_regime,
                self.novelty_regime,
                self.reasoning_type,
                self.mode,
            )
        ):
            raise ValueError("knowledge evaluation metadata must be nonempty")
        if type(self.reasoning_depth) is not int or not 0 <= self.reasoning_depth <= 2:
            raise ValueError("reasoning_depth must be an integer from zero through two")
        if type(self.prefix_length) is not int or self.prefix_length <= 0:
            raise ValueError("prefix_length must be a positive integer")
        for field_name in (
            "support_ids",
            "required_edge_ids",
            "visible_cue_ids",
            "eligible_task_ids",
            "oracle_node_ids",
        ):
            values = getattr(self, field_name)
            if not isinstance(values, tuple) or any(
                not isinstance(value, str) or not value for value in values
            ):
                raise ValueError(f"{field_name} must contain nonempty identifiers")
        if (
            not isinstance(self.candidate_answer_texts, tuple)
            or len(self.candidate_answer_texts) != _CANDIDATE_COUNT
            or any(not text for text in self.candidate_answer_texts)
        ):
            raise ValueError("candidate_answer_texts must contain four answers")
        scores = _immutable_float_array(
            self.candidate_nll,
            "candidate_nll",
            ndim=1,
        )
        if (
            scores.shape != (_CANDIDATE_COUNT,)
            or np.any(~np.isfinite(scores))
            or np.any(scores < 0.0)
        ):
            raise ValueError(
                "candidate_nll must contain four finite nonnegative scores"
            )
        _validate_candidate_index(
            self.correct_candidate_index,
            "correct_candidate_index",
        )
        _validate_candidate_index(
            self.predicted_candidate_index,
            "predicted_candidate_index",
        )
        expected_prediction = int(np.argmin(scores))
        if self.predicted_candidate_index != expected_prediction:
            raise ValueError("predicted_candidate_index must minimize candidate NLL")
        expected_correct = expected_prediction == self.correct_candidate_index
        if (
            type(self.candidate_correct) is not bool
            or self.candidate_correct != expected_correct
        ):
            raise ValueError(
                "candidate_correct must match predicted and correct indices"
            )
        expected_correct_nll = float(scores[self.correct_candidate_index])
        wrong_scores = np.delete(scores, self.correct_candidate_index)
        expected_margin = float(np.min(wrong_scores) - expected_correct_nll)
        _require_close(
            self.correct_answer_nll,
            expected_correct_nll,
            "correct_answer_nll",
        )
        _require_close(self.candidate_margin, expected_margin, "candidate_margin")
        _validate_optional_index(self.selected_node_index, "selected_node_index")
        _validate_optional_index(self.task_oracle_node_index, "task_oracle_node_index")
        _validate_optional_index(
            self.best_hard_node_index,
            "best_hard_node_index",
            required=True,
        )
        _validate_nonnegative_finite(
            self.best_hard_node_correct_answer_nll,
            "best_hard_node_correct_answer_nll",
        )
        _require_close(
            self.best_hard_node_regret,
            self.correct_answer_nll - self.best_hard_node_correct_answer_nll,
            "best_hard_node_regret",
        )
        _validate_reference_metrics(
            self.selected_node_index,
            self.routed_correct_answer_nll,
            self.routed_regret,
            self.correct_answer_nll,
            "routed",
        )
        _validate_reference_metrics(
            self.task_oracle_node_index,
            self.task_oracle_correct_answer_nll,
            self.task_oracle_regret,
            self.correct_answer_nll,
            "task_oracle",
        )
        for field_name in ("node_accuracy", "top_k_accuracy"):
            value = getattr(self, field_name)
            if value is not None and type(value) is not bool:
                raise TypeError(f"{field_name} must be a bool or None")
        if self.task_oracle_node_index is None and (
            self.node_accuracy is not None or self.top_k_accuracy is not None
        ):
            raise ValueError("node accuracy is undefined without a hard-node oracle")
        if self.selected_node_index is None and any(
            value is not None
            for value in (
                self.node_accuracy,
                self.top_k_accuracy,
                self.address_entropy,
                self.address_margin,
                self.hard_required_edge_recall,
            )
        ):
            raise ValueError("routing metrics require a selected hard node")
        for field_name in ("address_entropy", "address_margin"):
            value = getattr(self, field_name)
            if value is not None:
                _validate_nonnegative_finite(value, field_name)
        for field_name in (
            "hard_required_edge_recall",
            "soft_required_edge_mean_coefficient",
        ):
            value = getattr(self, field_name)
            if value is not None and (
                not math.isfinite(value) or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{field_name} must lie in [0, 1]")
        if not self.required_edge_ids and any(
            value is not None
            for value in (
                self.hard_required_edge_recall,
                self.soft_required_edge_mean_coefficient,
            )
        ):
            raise ValueError(
                "required-edge support is undefined without required edges"
            )
        object.__setattr__(self, "candidate_nll", scores)


@dataclass(frozen=True)
class KnowledgeEvaluationAggregate:
    """One deterministic metadata slice of per-query knowledge metrics."""

    stage: int
    method: str
    grouping_axis: str
    grouping_value: str | int
    query_count: int
    candidate_accuracy: float
    mean_candidate_margin: float
    mean_correct_answer_nll: float
    mean_routed_regret: float | None
    mean_task_oracle_regret: float | None
    mean_best_hard_node_regret: float
    node_accuracy: float | None
    top_k_accuracy: float | None
    mean_address_entropy: float | None
    mean_address_margin: float | None
    mean_hard_required_edge_recall: float | None
    mean_soft_required_edge_coefficient: float | None

    def __post_init__(self) -> None:
        _validate_stage_method(self.stage, self.method)
        if self.grouping_axis not in KNOWLEDGE_AGGREGATION_AXES:
            raise ValueError(
                f"unknown knowledge aggregation axis: {self.grouping_axis}"
            )
        if not isinstance(self.grouping_value, (str, int)):
            raise TypeError("grouping_value must be a string or integer")
        if type(self.query_count) is not int or self.query_count <= 0:
            raise ValueError("query_count must be a positive integer")
        _validate_rate(self.candidate_accuracy, "candidate_accuracy")
        if not math.isfinite(self.mean_candidate_margin):
            raise ValueError("mean_candidate_margin must be finite")
        _validate_nonnegative_finite(
            self.mean_correct_answer_nll,
            "mean_correct_answer_nll",
        )
        if not math.isfinite(self.mean_best_hard_node_regret):
            raise ValueError("mean_best_hard_node_regret must be finite")
        for field_name in ("mean_routed_regret", "mean_task_oracle_regret"):
            value = getattr(self, field_name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite when present")
        for field_name in (
            "node_accuracy",
            "top_k_accuracy",
            "mean_hard_required_edge_recall",
            "mean_soft_required_edge_coefficient",
        ):
            value = getattr(self, field_name)
            if value is not None:
                _validate_rate(value, field_name)
        for field_name in ("mean_address_entropy", "mean_address_margin"):
            value = getattr(self, field_name)
            if value is not None:
                _validate_nonnegative_finite(value, field_name)


@dataclass(frozen=True, eq=False)
class KnowledgeMethodEvaluation:
    """Immutable query results and aggregate slices for one method and stage."""

    stage: int
    method: str
    queries: tuple[KnowledgeQueryEvaluation, ...]
    aggregates: tuple[KnowledgeEvaluationAggregate, ...]
    address_decision: KnowledgeAddressDecision | None = None
    edge_coefficients: np.ndarray | None = None

    def __post_init__(self) -> None:
        _validate_stage_method(self.stage, self.method)
        if (
            not isinstance(self.queries, tuple)
            or not self.queries
            or any(
                not isinstance(row, KnowledgeQueryEvaluation)
                for row in self.queries
            )
        ):
            raise ValueError("method evaluation requires knowledge query results")
        if any(
            row.stage != self.stage or row.method != self.method
            for row in self.queries
        ):
            raise ValueError("query results must match their method and stage")
        query_ids = tuple(row.query_id for row in self.queries)
        if len(set(query_ids)) != len(query_ids):
            raise ValueError("method evaluation query IDs must be unique")
        if (
            not isinstance(self.aggregates, tuple)
            or not self.aggregates
            or any(
                not isinstance(row, KnowledgeEvaluationAggregate)
                for row in self.aggregates
            )
        ):
            raise ValueError("method evaluation requires aggregate rows")
        if any(
            row.stage != self.stage or row.method != self.method
            for row in self.aggregates
        ):
            raise ValueError("aggregate rows must match their method and stage")
        all_rows = tuple(
            row for row in self.aggregates if row.grouping_axis == "all"
        )
        if len(all_rows) != 1 or all_rows[0].query_count != len(self.queries):
            raise ValueError("aggregates require one complete all-query row")
        if self.address_decision is not None and (
            not isinstance(self.address_decision, KnowledgeAddressDecision)
            or self.address_decision.selected_indices.shape != (len(self.queries),)
        ):
            raise ValueError("address_decision must align with query results")
        if self.address_decision is None and any(
            row.selected_node_index is not None for row in self.queries
        ):
            raise ValueError("selected query nodes require an address decision")
        coefficients = self.edge_coefficients
        if coefficients is not None:
            coefficients = _immutable_float_array(
                coefficients,
                "edge_coefficients",
                ndim=2,
            )
            if coefficients.shape[0] != len(self.queries):
                raise ValueError("edge coefficient rows must match query results")
            if np.any(~np.isfinite(coefficients)) or np.any(
                (coefficients < 0.0) | (coefficients > 1.0)
            ):
                raise ValueError("edge coefficients must be finite and in [0, 1]")
        object.__setattr__(self, "edge_coefficients", coefficients)


def evaluate_knowledge_method(
    queries: tuple[KnowledgeQuery, ...],
    hard_candidate_nll: np.ndarray,
    graph: MemoryGraph[object],
    packed_memory: PackedLoraMemory,
    *,
    stage: int,
    method: str,
    candidate_nll: np.ndarray | None = None,
    hard_decision: KnowledgeAddressDecision | LanguageAddressDecision | None = None,
    edge_coefficients: np.ndarray | None = None,
    unavailable_node_ids: tuple[str, ...] = (),
    unavailable_edge_ids: tuple[str, ...] = (),
) -> KnowledgeMethodEvaluation:
    """Evaluate one method without model execution or mutation.

    ``hard_candidate_nll`` is shared by every method. A hard router omits
    ``candidate_nll`` and its selected node is gathered from that shared tensor.
    Frozen, stored, or soft methods supply their own ``candidate_nll``. A soft
    EBT method also supplies the same final hard decision and continuous edge
    coefficients so routed regret and both forms of support remain comparable.
    The unavailable ID tuples explicitly identify canonical future topology at
    an early continual-learning stage; future oracle metrics are undefined and
    future required edges contribute zero support without inventing nodes.
    """
    _validate_stage_method(stage, method)
    _validate_queries(queries)
    valid_node_mask, valid_edge_mask = _validate_graph_packing(graph, packed_memory)
    hard_scores = _validated_hard_candidate_scores(
        hard_candidate_nll,
        len(queries),
        valid_node_mask,
    )
    decision = (
        None
        if hard_decision is None
        else _coerce_address_decision(hard_decision)
    )
    if decision is not None:
        _validate_decision_alignment(decision, valid_node_mask, len(queries))
    if candidate_nll is None:
        if decision is None:
            raise ValueError("hard evaluation requires an address decision")
        rows = np.arange(len(queries))[:, None]
        candidates = np.arange(_CANDIDATE_COUNT)[None, :]
        selected = decision.selected_indices[:, None]
        method_scores = hard_scores[rows, candidates, selected]
    else:
        method_scores = _validated_method_candidate_scores(
            candidate_nll,
            len(queries),
        )
    coefficients = _validated_edge_coefficients(
        edge_coefficients,
        len(queries),
        valid_edge_mask,
    )
    node_ids = memory_node_ids(graph)
    edge_ids = memory_edge_node_ids(graph)
    node_index_by_id = {node_id: index for index, node_id in enumerate(node_ids)}
    edge_index_by_id = {node_id: index for index, node_id in enumerate(edge_ids)}
    _validate_unavailable_ids(
        queries,
        node_index_by_id,
        edge_index_by_id,
        unavailable_node_ids,
        unavailable_edge_ids,
    )
    query_results = tuple(
        _evaluate_query(
            stage,
            method,
            query_index,
            query,
            method_scores[query_index],
            hard_scores[query_index],
            packed_memory,
            decision,
            coefficients,
            node_index_by_id,
            edge_index_by_id,
            valid_node_mask,
            frozenset(unavailable_node_ids),
            frozenset(unavailable_edge_ids),
        )
        for query_index, query in enumerate(queries)
    )
    aggregates = aggregate_knowledge_evaluations(query_results)
    return KnowledgeMethodEvaluation(
        stage=stage,
        method=method,
        queries=query_results,
        aggregates=aggregates,
        address_decision=decision,
        edge_coefficients=coefficients,
    )


def aggregate_knowledge_evaluations(
    rows: tuple[KnowledgeQueryEvaluation, ...],
) -> tuple[KnowledgeEvaluationAggregate, ...]:
    """Aggregate one method/stage across every required metadata axis."""
    if (
        not isinstance(rows, tuple)
        or not rows
        or any(not isinstance(row, KnowledgeQueryEvaluation) for row in rows)
    ):
        raise ValueError("knowledge aggregation requires nonempty query results")
    stage = rows[0].stage
    method = rows[0].method
    if any(row.stage != stage or row.method != method for row in rows):
        raise ValueError("aggregated queries must share one stage and method")
    aggregates: list[KnowledgeEvaluationAggregate] = []
    for axis in KNOWLEDGE_AGGREGATION_AXES:
        if axis == "all":
            groups: tuple[
                tuple[str | int, tuple[KnowledgeQueryEvaluation, ...]],
                ...,
            ] = (("all", rows),)
        else:
            group_values = sorted(
                {getattr(row, axis) for row in rows},
                key=lambda value: (isinstance(value, str), value),
            )
            groups = tuple(
                (
                    value,
                    tuple(row for row in rows if getattr(row, axis) == value),
                )
                for value in group_values
            )
        aggregates.extend(
            _aggregate_group(stage, method, axis, value, group_rows)
            for value, group_rows in groups
        )
    return tuple(aggregates)


def evaluate_ebt_knowledge_methods(
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    graph: MemoryGraph[object],
    packed_memory: PackedLoraMemory,
    lora_config: LoraConfig,
    address_book: AddressBook,
    queries: tuple[KnowledgeQuery, ...],
    hard_candidate_nll: np.ndarray,
    *,
    stage: int,
    hopfield_config: HopfieldConfig | None = None,
    ebt_config: EbtConfig | None = None,
    evaluation_microbatch_size: int | None = None,
    unavailable_node_ids: tuple[str, ...] = (),
    unavailable_edge_ids: tuple[str, ...] = (),
    routers: tuple[str, ...] = _EBT_ROUTERS,
) -> tuple[KnowledgeMethodEvaluation, ...]:
    """Run each EBT initialization once and reuse it for hard and soft scoring."""
    _validate_stage_method(stage, "vamp_ebt_uniform")
    _validate_queries(queries)
    if (
        not isinstance(routers, tuple)
        or not routers
        or len(set(routers)) != len(routers)
        or any(router not in _EBT_ROUTERS for router in routers)
    ):
        raise ValueError("routers must be a nonempty unique subset of EBT routers")
    valid_node_mask, valid_edge_mask = _validate_graph_packing(
        graph,
        packed_memory,
    )
    _validated_hard_candidate_scores(
        hard_candidate_nll,
        len(queries),
        valid_node_mask,
    )
    if not isinstance(address_book, AddressBook):
        raise TypeError("EBT knowledge evaluation requires an AddressBook")
    expected_address_ids = memory_node_ids(graph) + (None,) * (
        packed_memory.node_path_matrix.shape[0] - len(graph.nodes)
    )
    if address_book.node_ids != expected_address_ids:
        raise ValueError("address-book node IDs must match graph insertion order")
    if hopfield_config is None or ebt_config is None:
        from apm.memory.address_refinement import EbtConfig as RuntimeEbtConfig
        from apm.memory.content_addressing import (
            HopfieldConfig as RuntimeHopfieldConfig,
        )

        hopfield_config = (
            RuntimeHopfieldConfig()
            if hopfield_config is None
            else hopfield_config
        )
        ebt_config = RuntimeEbtConfig() if ebt_config is None else ebt_config
    prefix_batch = _stack_router_batches(queries)
    evaluations: list[KnowledgeMethodEvaluation] = []
    for router in routers:
        refinement = _run_ebt_trace(
            router,
            base_params,
            model_config,
            packed_memory,
            lora_config,
            address_book,
            prefix_batch,
            hopfield_config=hopfield_config,
            ebt_config=ebt_config,
        )
        decision = _decision_from_ebt_result(refinement, valid_node_mask)
        _validate_decision_alignment(decision, valid_node_mask, len(queries))
        coefficients = _validated_edge_coefficients(
            np.asarray(refinement.edge_coefficients, dtype=np.float32),
            len(queries),
            valid_edge_mask,
        )
        assert coefficients is not None
        soft_scores = score_edge_coefficient_candidates(
            base_params,
            model_config,
            packed_memory,
            lora_config,
            queries,
            coefficients,
            evaluation_microbatch_size=evaluation_microbatch_size,
        )
        evaluations.append(
            evaluate_knowledge_method(
                queries,
                hard_candidate_nll,
                graph,
                packed_memory,
                stage=stage,
                method=router,
                hard_decision=decision,
                unavailable_node_ids=unavailable_node_ids,
                unavailable_edge_ids=unavailable_edge_ids,
            )
        )
        evaluations.append(
            evaluate_knowledge_method(
                queries,
                hard_candidate_nll,
                graph,
                packed_memory,
                stage=stage,
                method=f"{router}_soft",
                candidate_nll=soft_scores,
                hard_decision=decision,
                edge_coefficients=coefficients,
                unavailable_node_ids=unavailable_node_ids,
                unavailable_edge_ids=unavailable_edge_ids,
            )
        )
    return tuple(evaluations)


def _evaluate_query(
    stage: int,
    method: str,
    query_index: int,
    query: KnowledgeQuery,
    method_candidate_nll: np.ndarray,
    hard_candidate_nll: np.ndarray,
    packed_memory: PackedLoraMemory,
    decision: KnowledgeAddressDecision | None,
    edge_coefficients: np.ndarray | None,
    node_index_by_id: dict[str, int],
    edge_index_by_id: dict[str, int],
    valid_node_mask: np.ndarray,
    unavailable_node_ids: frozenset[str],
    unavailable_edge_ids: frozenset[str],
) -> KnowledgeQueryEvaluation:
    oracle_indices = tuple(
        node_index_by_id[node_id]
        for node_id in query.oracle_node_ids
        if node_id not in unavailable_node_ids
    )
    required_edge_indices = tuple(
        edge_index_by_id[node_id]
        for node_id in query.required_edge_ids
        if node_id not in unavailable_edge_ids
    )
    unavailable_required_edge_count = sum(
        node_id in unavailable_edge_ids for node_id in query.required_edge_ids
    )
    correct_index = query.correct_candidate_index
    valid_node_indices = np.flatnonzero(valid_node_mask)
    correct_hard_nll = hard_candidate_nll[correct_index]
    best_offset = int(np.argmin(correct_hard_nll[valid_node_indices]))
    best_hard_node_index = int(valid_node_indices[best_offset])
    best_hard_nll = float(correct_hard_nll[best_hard_node_index])
    if oracle_indices:
        oracle_nll = np.asarray(
            tuple(correct_hard_nll[index] for index in oracle_indices),
            dtype=np.float32,
        )
        oracle_offset = int(np.argmin(oracle_nll))
        task_oracle_node_index: int | None = oracle_indices[oracle_offset]
        task_oracle_nll: float | None = float(oracle_nll[oracle_offset])
    else:
        task_oracle_node_index = None
        task_oracle_nll = None
    if decision is None:
        selected_node_index = None
        routed_nll = None
        node_accuracy = None
        top_k_accuracy = None
        address_entropy = None
        address_margin = None
        hard_support = None
    else:
        selected_node_index = int(decision.selected_indices[query_index])
        routed_nll = float(correct_hard_nll[selected_node_index])
        node_accuracy = (
            None
            if not oracle_indices
            else selected_node_index in oracle_indices
        )
        top_k_accuracy = (
            None
            if not oracle_indices
            else any(
                index in oracle_indices
                for index in decision.top_k_indices[query_index]
            )
        )
        address_entropy = float(decision.entropy[query_index])
        raw_margin = float(decision.score_margin[query_index])
        address_margin = raw_margin if math.isfinite(raw_margin) else None
        if not query.required_edge_ids:
            hard_support = None
        else:
            supported_count = int(
                np.sum(
                    np.asarray(packed_memory.node_path_matrix)[
                        selected_node_index,
                        required_edge_indices,
                    ]
                    > 0.0
                )
            )
            hard_support = supported_count / len(query.required_edge_ids)
    correct_nll = float(method_candidate_nll[correct_index])
    soft_support = None
    if edge_coefficients is not None and query.required_edge_ids:
        available_sum = float(
            np.sum(edge_coefficients[query_index, required_edge_indices])
        )
        soft_support = available_sum / (
            len(required_edge_indices) + unavailable_required_edge_count
        )
    predicted_index = int(np.argmin(method_candidate_nll))
    wrong_nll = np.delete(method_candidate_nll, correct_index)
    return KnowledgeQueryEvaluation(
        stage=stage,
        method=method,
        query_id=query.query_id,
        task_id=str(query.task_id),
        family_id=query.family_id,
        query_kind=query.query_kind,
        proof_id=query.proof_id,
        support_ids=query.support_ids,
        required_edge_ids=tuple(str(value) for value in query.required_edge_ids),
        cue_regime=query.cue_regime,
        visible_cue_ids=query.visible_cue_ids,
        eligible_task_ids=tuple(str(value) for value in query.eligible_task_ids),
        novelty_regime=query.novelty_regime,
        reasoning_type=query.reasoning_type,
        reasoning_depth=query.reasoning_depth,
        prefix_length=query.prefix_length,
        mode=query.mode,
        oracle_node_ids=tuple(str(value) for value in query.oracle_node_ids),
        candidate_answer_texts=tuple(
            candidate.answer_text for candidate in query.candidates
        ),
        candidate_nll=method_candidate_nll,
        correct_candidate_index=correct_index,
        predicted_candidate_index=predicted_index,
        candidate_correct=predicted_index == correct_index,
        candidate_margin=float(np.min(wrong_nll) - correct_nll),
        correct_answer_nll=correct_nll,
        selected_node_index=selected_node_index,
        task_oracle_node_index=task_oracle_node_index,
        best_hard_node_index=best_hard_node_index,
        routed_correct_answer_nll=routed_nll,
        task_oracle_correct_answer_nll=task_oracle_nll,
        best_hard_node_correct_answer_nll=best_hard_nll,
        routed_regret=(None if routed_nll is None else correct_nll - routed_nll),
        task_oracle_regret=(
            None if task_oracle_nll is None else correct_nll - task_oracle_nll
        ),
        best_hard_node_regret=correct_nll - best_hard_nll,
        node_accuracy=node_accuracy,
        top_k_accuracy=top_k_accuracy,
        address_entropy=address_entropy,
        address_margin=address_margin,
        hard_required_edge_recall=hard_support,
        soft_required_edge_mean_coefficient=soft_support,
    )


def _aggregate_group(
    stage: int,
    method: str,
    axis: str,
    value: str | int,
    rows: tuple[KnowledgeQueryEvaluation, ...],
) -> KnowledgeEvaluationAggregate:
    return KnowledgeEvaluationAggregate(
        stage=stage,
        method=method,
        grouping_axis=axis,
        grouping_value=value,
        query_count=len(rows),
        candidate_accuracy=float(
            np.mean(tuple(row.candidate_correct for row in rows))
        ),
        mean_candidate_margin=float(
            np.mean(tuple(row.candidate_margin for row in rows))
        ),
        mean_correct_answer_nll=float(
            np.mean(tuple(row.correct_answer_nll for row in rows))
        ),
        mean_routed_regret=_optional_mean(rows, lambda row: row.routed_regret),
        mean_task_oracle_regret=_optional_mean(
            rows,
            lambda row: row.task_oracle_regret,
        ),
        mean_best_hard_node_regret=float(
            np.mean(tuple(row.best_hard_node_regret for row in rows))
        ),
        node_accuracy=_optional_mean(rows, lambda row: row.node_accuracy),
        top_k_accuracy=_optional_mean(rows, lambda row: row.top_k_accuracy),
        mean_address_entropy=_optional_mean(
            rows,
            lambda row: row.address_entropy,
        ),
        mean_address_margin=_optional_mean(rows, lambda row: row.address_margin),
        mean_hard_required_edge_recall=_optional_mean(
            rows,
            lambda row: row.hard_required_edge_recall,
        ),
        mean_soft_required_edge_coefficient=_optional_mean(
            rows,
            lambda row: row.soft_required_edge_mean_coefficient,
        ),
    )


def _optional_mean(
    rows: tuple[KnowledgeQueryEvaluation, ...],
    getter: Callable[[KnowledgeQueryEvaluation], float | bool | None],
) -> float | None:
    values = tuple(value for row in rows if (value := getter(row)) is not None)
    return None if not values else float(np.mean(values))


def _validate_queries(queries: tuple[KnowledgeQuery, ...]) -> None:
    if (
        not isinstance(queries, tuple)
        or not queries
        or any(not isinstance(query, KnowledgeQuery) for query in queries)
    ):
        raise ValueError("knowledge evaluation requires KnowledgeQuery values")
    query_ids = tuple(query.query_id for query in queries)
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("knowledge evaluation query IDs must be unique")


def _validate_unavailable_ids(
    queries: tuple[KnowledgeQuery, ...],
    node_index_by_id: dict[str, int],
    edge_index_by_id: dict[str, int],
    unavailable_node_ids: tuple[str, ...],
    unavailable_edge_ids: tuple[str, ...],
) -> None:
    for label, values in (
        ("unavailable_node_ids", unavailable_node_ids),
        ("unavailable_edge_ids", unavailable_edge_ids),
    ):
        if not isinstance(values, tuple) or any(
            not isinstance(value, str) or not value for value in values
        ):
            raise ValueError(f"{label} must contain nonempty string IDs")
        if len(set(values)) != len(values):
            raise ValueError(f"{label} must contain unique IDs")
    present_nodes = set(node_index_by_id)
    present_edges = set(edge_index_by_id)
    unavailable_nodes = set(unavailable_node_ids)
    unavailable_edges = set(unavailable_edge_ids)
    if present_nodes & unavailable_nodes or present_edges & unavailable_edges:
        raise ValueError("unavailable topology IDs cannot already be committed")
    referenced_nodes = {
        str(node_id) for query in queries for node_id in query.oracle_node_ids
    }
    referenced_edges = {
        str(node_id) for query in queries for node_id in query.required_edge_ids
    }
    unknown_nodes = referenced_nodes.difference(present_nodes, unavailable_nodes)
    unknown_edges = referenced_edges.difference(present_edges, unavailable_edges)
    if unknown_nodes:
        raise ValueError(f"unknown oracle node IDs: {sorted(unknown_nodes)}")
    if unknown_edges:
        raise ValueError(f"unknown required edge IDs: {sorted(unknown_edges)}")


def _validate_graph_packing(
    graph: MemoryGraph[object],
    packed_memory: PackedLoraMemory,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(graph, MemoryGraph):
        raise TypeError("knowledge evaluation graph must be a MemoryGraph")
    if not isinstance(packed_memory, PackedLoraMemory):
        raise TypeError("knowledge evaluation memory must be PackedLoraMemory")
    paths = np.asarray(packed_memory.node_path_matrix, dtype=np.float32)
    valid_nodes = np.array(
        packed_memory.valid_node_mask,
        dtype=np.bool_,
        copy=True,
    )
    valid_edges = np.array(
        packed_memory.valid_edge_mask,
        dtype=np.bool_,
        copy=True,
    )
    if (
        paths.ndim != 2
        or valid_nodes.shape != (paths.shape[0],)
        or valid_edges.shape != (paths.shape[1],)
    ):
        raise ValueError("packed memory masks must match the path matrix")
    expected_nodes = np.arange(paths.shape[0]) < len(graph.nodes)
    edge_count = len(memory_edge_node_ids(graph))
    if len(graph.nodes) > paths.shape[0] or edge_count > paths.shape[1]:
        raise ValueError("packed memory capacity cannot contain the supplied graph")
    expected_edges = np.arange(paths.shape[1]) < edge_count
    if not np.array_equal(valid_nodes, expected_nodes) or not np.array_equal(
        valid_edges,
        expected_edges,
    ):
        raise ValueError(
            "packed memory validity masks must match graph insertion order"
        )
    expected_paths = np.zeros_like(paths)
    expected_paths[: len(graph.nodes), :edge_count] = path_incidence_matrix(graph)
    if not np.array_equal(paths, expected_paths):
        raise ValueError("packed path matrix must exactly encode the supplied graph")
    valid_nodes.flags.writeable = False
    valid_edges.flags.writeable = False
    return valid_nodes, valid_edges


def _validated_hard_candidate_scores(
    scores: np.ndarray,
    query_count: int,
    valid_node_mask: np.ndarray,
) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float32)
    expected_shape = (query_count, _CANDIDATE_COUNT, valid_node_mask.size)
    if values.shape != expected_shape:
        raise ValueError(f"hard_candidate_nll must have shape {expected_shape}")
    if np.any(~np.isfinite(values[:, :, valid_node_mask])) or np.any(
        values[:, :, valid_node_mask] < 0.0
    ):
        raise ValueError("valid hard-node candidate NLL must be finite and nonnegative")
    if np.any(~np.isposinf(values[:, :, ~valid_node_mask])):
        raise ValueError("invalid hard-node candidate NLL must be positive infinity")
    return values


def _validated_method_candidate_scores(
    scores: np.ndarray,
    query_count: int,
) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float32)
    expected_shape = (query_count, _CANDIDATE_COUNT)
    if values.shape != expected_shape:
        raise ValueError(f"candidate_nll must have shape {expected_shape}")
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("candidate NLL must be finite and nonnegative")
    return values


def _validated_edge_coefficients(
    coefficients: np.ndarray | None,
    query_count: int,
    valid_edge_mask: np.ndarray,
) -> np.ndarray | None:
    if coefficients is None:
        return None
    values = _immutable_float_array(
        coefficients,
        "edge_coefficients",
        ndim=2,
    )
    expected_shape = (query_count, valid_edge_mask.size)
    if values.shape != expected_shape:
        raise ValueError(f"edge_coefficients must have shape {expected_shape}")
    if np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("edge coefficients must be finite and in [0, 1]")
    if np.any(values[:, ~valid_edge_mask] != 0.0):
        raise ValueError("invalid edge coefficients must be exactly zero")
    return values


def _coerce_address_decision(
    decision: KnowledgeAddressDecision | LanguageAddressDecision,
) -> KnowledgeAddressDecision:
    if isinstance(decision, KnowledgeAddressDecision):
        return decision
    required_fields = (
        "selected_indices",
        "node_probabilities",
        "node_scores",
        "score_margin",
        "entropy",
        "top_k_indices",
    )
    if any(not hasattr(decision, field_name) for field_name in required_fields):
        raise TypeError("hard_decision must expose the common address fields")
    return KnowledgeAddressDecision(
        *(getattr(decision, field_name) for field_name in required_fields)
    )


def _validate_decision_alignment(
    decision: KnowledgeAddressDecision,
    valid_node_mask: np.ndarray,
    query_count: int,
) -> None:
    node_capacity = valid_node_mask.size
    if decision.selected_indices.shape != (
        query_count,
    ) or decision.node_scores.shape != (query_count, node_capacity):
        raise ValueError("hard decision must align with queries and node capacity")
    if np.any(decision.node_probabilities[:, ~valid_node_mask] != 0.0):
        raise ValueError("invalid nodes must have exactly zero probability")
    if np.any(~np.isfinite(decision.node_scores[:, valid_node_mask])) or np.any(
        ~np.isneginf(decision.node_scores[:, ~valid_node_mask])
    ):
        raise ValueError("valid node scores must be finite and invalid scores -inf")
    if np.any(~valid_node_mask[decision.selected_indices]):
        raise ValueError("hard decisions must select valid nodes")
    valid_count = int(np.sum(valid_node_mask))
    expected_top_k = min(_ROUTER_TOP_K, valid_count)
    if decision.top_k_indices.shape != (query_count, expected_top_k):
        raise ValueError("top-k width must use the canonical valid-node limit")
    if np.any(~valid_node_mask[decision.top_k_indices]):
        raise ValueError("top-k decisions must identify valid nodes")
    valid_scores = decision.node_scores[:, valid_node_mask]
    expected_margin = (
        np.full((query_count,), np.inf, dtype=np.float32)
        if valid_count == 1
        else np.sort(valid_scores, axis=1)[:, -1]
        - np.sort(valid_scores, axis=1)[:, -2]
    )
    if not np.allclose(decision.score_margin, expected_margin, atol=1e-6):
        raise ValueError("score margins must equal the top-two valid score difference")


def _decision_from_ebt_result(
    result: object,
    valid_node_mask: np.ndarray,
) -> KnowledgeAddressDecision:
    logits = np.asarray(result.final_node_logits, dtype=np.float32)
    probabilities = np.asarray(result.node_probabilities, dtype=np.float32)
    selected = np.asarray(result.selected_indices)
    if logits.ndim != 2 or logits.shape[1] != valid_node_mask.size:
        raise ValueError("EBT final logits must have shape [query, node]")
    masked_logits = np.where(valid_node_mask[None, :], logits, -np.inf)
    valid_count = int(np.sum(valid_node_mask))
    rankings = np.argsort(-masked_logits, axis=1, kind="stable")
    top_k = rankings[:, : min(_ROUTER_TOP_K, valid_count)]
    sorted_valid = np.sort(masked_logits[:, valid_node_mask], axis=1)[:, ::-1]
    margins = (
        np.full((logits.shape[0],), np.inf, dtype=np.float32)
        if valid_count == 1
        else sorted_valid[:, 0] - sorted_valid[:, 1]
    )
    entropy = -np.sum(
        probabilities
        * np.log(np.where(probabilities > 0.0, probabilities, 1.0)),
        axis=1,
    )
    return KnowledgeAddressDecision(
        selected_indices=selected,
        node_probabilities=probabilities,
        node_scores=masked_logits,
        score_margin=margins,
        entropy=entropy,
        top_k_indices=top_k,
    )


def _stack_router_batches(queries: tuple[KnowledgeQuery, ...]) -> RouterBatch:
    batches = tuple(query.router_batch for query in queries)
    maximum_width = max(batch.input_ids.shape[1] for batch in batches)

    def padded(values: np.ndarray, fill_value: int | bool) -> np.ndarray:
        missing = maximum_width - values.shape[1]
        return np.pad(
            values,
            ((0, 0), (0, missing)),
            constant_values=fill_value,
        )

    return RouterBatch(
        input_ids=np.concatenate(
            tuple(padded(batch.input_ids, 0) for batch in batches)
        ),
        attention_mask=np.concatenate(
            tuple(padded(batch.attention_mask, False) for batch in batches)
        ),
        target_ids=np.concatenate(
            tuple(padded(batch.target_ids, 0) for batch in batches)
        ),
        loss_mask=np.concatenate(
            tuple(padded(batch.loss_mask, False) for batch in batches)
        ),
    )


def _run_ebt_trace(*args, **kwargs):
    from apm.continual.language_routing import trace_ebt_language_prefix

    return trace_ebt_language_prefix(*args, **kwargs)


def _immutable_float_array(
    values: object,
    field_name: str,
    *,
    ndim: int,
) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype.kind not in "fiu":
        raise TypeError(f"{field_name} must be numeric")
    result = np.array(raw, dtype=np.float32, copy=True)
    if result.ndim != ndim:
        raise ValueError(f"{field_name} must have rank {ndim}")
    result.flags.writeable = False
    return result


def _immutable_integer_array(
    values: object,
    field_name: str,
    *,
    ndim: int,
) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype.kind not in "iu":
        raise TypeError(f"{field_name} must contain integers")
    result = np.array(raw, dtype=np.int32, copy=True)
    if result.ndim != ndim:
        raise ValueError(f"{field_name} must have rank {ndim}")
    result.flags.writeable = False
    return result


def _validate_stage_method(stage: int, method: str) -> None:
    if type(stage) is not int or stage < 0:
        raise ValueError("knowledge evaluation stage must be a nonnegative integer")
    if not isinstance(method, str) or not method or method != method.strip():
        raise ValueError("knowledge evaluation method must be a canonical name")


def _validate_candidate_index(value: int, field_name: str) -> None:
    if type(value) is not int or not 0 <= value < _CANDIDATE_COUNT:
        raise ValueError(f"{field_name} must identify one of four candidates")


def _validate_optional_index(
    value: int | None,
    field_name: str,
    *,
    required: bool = False,
) -> None:
    if value is None:
        if required:
            raise ValueError(f"{field_name} is required")
        return
    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a nonnegative integer")


def _validate_reference_metrics(
    reference_index: int | None,
    reference_nll: float | None,
    regret: float | None,
    method_nll: float,
    reference_name: str,
) -> None:
    if reference_index is None:
        if reference_nll is not None or regret is not None:
            raise ValueError(f"{reference_name} metrics require a reference node")
        return
    if reference_nll is None or regret is None:
        raise ValueError(f"{reference_name} node requires NLL and regret")
    _validate_nonnegative_finite(reference_nll, f"{reference_name}_correct_answer_nll")
    _require_close(regret, method_nll - reference_nll, f"{reference_name}_regret")


def _validate_nonnegative_finite(value: float, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{field_name} must be a real number")
    if not math.isfinite(float(value)) or value < 0.0:
        raise ValueError(f"{field_name} must be finite and nonnegative")


def _validate_rate(value: float, field_name: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must lie in [0, 1]")


def _require_close(actual: float, expected: float, field_name: str) -> None:
    if not math.isfinite(actual) or not math.isclose(
        actual,
        expected,
        rel_tol=1e-6,
        abs_tol=1e-6,
    ):
        raise ValueError(f"{field_name} must match its exact score difference")


__all__ = [
    "KNOWLEDGE_AGGREGATION_AXES",
    "KnowledgeAddressDecision",
    "KnowledgeEvaluationAggregate",
    "KnowledgeMethodEvaluation",
    "KnowledgeQueryEvaluation",
    "aggregate_knowledge_evaluations",
    "evaluate_ebt_knowledge_methods",
    "evaluate_knowledge_method",
]

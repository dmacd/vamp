"""Validation and sealed-query evaluation over immutable semantic stages."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import reduce

import numpy as np

from apm.continual.language_adaptation_artifact import LanguageAdaptationArtifact
from apm.continual.language_baseline_training import (
    IndependentRootAdapter,
    pack_root_adapter,
)
from apm.continual.language_routing import route_language_prefix
from apm.data.text.tinyworlds_q_semantic.contracts import (
    QueryExperimentPreset,
    SemanticQueryResult,
    require_sha256,
)
from apm.data.text.tinyworlds_q_semantic.execution import PilotBudgetResult
from apm.data.text.tinyworlds_q_semantic.queries import (
    CompiledSemanticQuery,
    evaluate_semantic_method,
    iter_semantic_score_chunks,
    stack_semantic_router_batches,
)
from apm.data.text.tinyworlds_q_semantic.scaling import evaluation_schedule
from apm.lm.candidate_scoring import score_edge_coefficient_candidates
from apm.lm.checkpoint import LoadedGptNeoCheckpoint
from apm.lm.lora_memory import (
    edge_coefficients_for_node,
    pack_lora_memory,
)
from apm.memory.graph import (
    NodeId,
    add_memory_node,
    init_memory_graph,
    memory_node_ids,
)


EvaluationProgress = Callable[[str, int, int], None]

_ROUTER_METHODS = (
    ("vamp_exhaustive", "vamp_exhaustive"),
    ("vamp_hopfield", "vamp_hopfield"),
    ("vamp_ebt_uniform", "vamp_ebt_uniform"),
    ("vamp_ebt_hopfield", "vamp_ebt_hopfield"),
    ("deterministic_random_node", "vamp_random"),
)


@dataclass(frozen=True, slots=True)
class PilotBudgetEvaluation:
    """One budget's matching-adapter evidence and exact validation rows."""

    budget: PilotBudgetResult
    results: tuple[SemanticQueryResult, ...]
    adaptation_tensor_checksum: str

    def __post_init__(self) -> None:
        if not self.results or any(row.split != "validation" for row in self.results):
            raise ValueError("pilot budget evaluation requires validation rows")
        require_sha256(
            self.adaptation_tensor_checksum,
            "pilot adaptation tensor checksum",
        )
        methods = {row.method for row in self.results}
        if methods != {"base", "independent"}:
            raise ValueError("pilot budget rows must contain base and independent methods")
        for concept_id, concept_accuracy in self.budget.concept_accuracy:
            base_rows = tuple(
                row
                for row in self.results
                if row.method == "base" and row.concept_id == concept_id
            )
            independent_rows = tuple(
                row
                for row in self.results
                if row.method == "independent"
                and row.concept_id == concept_id
                and row.adapter_concept_id == concept_id
            )
            if (
                len(base_rows) != 36
                or len(independent_rows) != 36
                or {row.template_id for row in base_rows}
                != {row.template_id for row in independent_rows}
                or not np.isclose(_accuracy(independent_rows), concept_accuracy)
                or not np.isclose(
                    _accuracy(base_rows),
                    dict(self.budget.base_accuracy)[concept_id],
                )
            ):
                raise ValueError("pilot budget accuracy does not match exact query rows")


def evaluate_pilot_budget(
    queries: tuple[CompiledSemanticQuery, ...],
    base: LoadedGptNeoCheckpoint,
    adapters: tuple[IndependentRootAdapter, ...],
    tensor_checksum: str,
    preset: QueryExperimentPreset,
    *,
    progress: EvaluationProgress | None = None,
) -> PilotBudgetEvaluation:
    """Measure the registered pilot gate using matching independent adapters."""
    if (
        tuple(str(adapter.task_id) for adapter in adapters) != preset.concept_ids
        or any(len(adapter.step_losses) != preset.adapter_updates for adapter in adapters)
        or {query.concept_id for query in queries} != set(preset.concept_ids)
        or any(query.split != "validation" for query in queries)
        or base.config != preset.model_config
    ):
        raise ValueError("pilot budget evaluation inputs changed")
    require_sha256(tensor_checksum, "pilot budget tensor checksum")
    graph = reduce(
        lambda current, indexed: add_memory_node(
            current,
            NodeId(str(indexed[1].task_id)),
            NodeId("root"),
            indexed[1].task_id,
            indexed[0],
            indexed[1].adapter,
        ),
        enumerate(adapters, start=1),
        init_memory_graph(NodeId("root")),
    )
    packed = pack_lora_memory(
        graph,
        base.config,
        preset.lora_config,
        max_nodes=preset.max_nodes,
        max_edges=preset.max_edges,
    )
    base_scores, hard_scores = _score_reference(
        queries,
        base,
        packed,
        preset,
        progress=progress,
        phase=f"budget {preset.adapter_updates} reference",
    )
    base_rows = evaluate_semantic_method(
        queries,
        hard_scores,
        graph,
        packed,
        stage=0,
        method="base",
        candidate_nll=base_scores,
    ).results
    node_indexes = {
        str(node_id): index for index, node_id in enumerate(memory_node_ids(graph))
    }
    independent_rows = tuple(
        row
        for adapter in adapters
        for concept_queries, concept_indices in (
            _queries_for_concept(queries, str(adapter.task_id)),
        )
        for concept_hard_scores in (hard_scores[np.asarray(concept_indices)],)
        for candidate_scores in (
            concept_hard_scores[:, :, node_indexes[str(adapter.task_id)]],
        )
        for row in evaluate_semantic_method(
            concept_queries,
            concept_hard_scores,
            graph,
            packed,
            stage=preset.concept_ids.index(str(adapter.task_id)) + 1,
            method="independent",
            candidate_nll=candidate_scores,
            adapter_concept_id=str(adapter.task_id),
        ).results
    )
    concept_accuracy = tuple(
        (
            concept_id,
            _accuracy(
                tuple(
                    row
                    for row in independent_rows
                    if row.concept_id == concept_id
                    and row.adapter_concept_id == concept_id
                )
            ),
        )
        for concept_id in preset.concept_ids
    )
    base_accuracy = tuple(
        (
            concept_id,
            _accuracy(tuple(row for row in base_rows if row.concept_id == concept_id)),
        )
        for concept_id in preset.concept_ids
    )
    return PilotBudgetEvaluation(
        budget=PilotBudgetResult(
            preset.adapter_updates,
            concept_accuracy,
            base_accuracy,
        ),
        results=base_rows + independent_rows,
        adaptation_tensor_checksum=tensor_checksum,
    )


def evaluate_staged_semantic_queries(
    queries: tuple[CompiledSemanticQuery, ...],
    base: LoadedGptNeoCheckpoint,
    stages: tuple[LanguageAdaptationArtifact, ...],
    preset: QueryExperimentPreset,
    pad_token_id: int,
    *,
    progress: EvaluationProgress | None = None,
) -> tuple[SemanticQueryResult, ...]:
    """Evaluate every registered method on the manifest-derived stage schedule."""
    if (
        type(stages) is not tuple
        or len(stages) != preset.active_world_count
        or tuple(len(stage.task_order) for stage in stages)
        != tuple(range(1, preset.active_world_count + 1))
    ):
        raise ValueError("semantic stages must cover one contiguous active prefix")
    tuple(
        _validate_evaluation_inputs(
            tuple(
                query
                for query in queries
                if query.concept_id in stage.task_order
            ),
            base,
            stage,
            preset,
            require_complete_manifest=False,
        )
        for stage in stages
    )
    final_graph = stages[-1].vamp_graph
    final_packed = pack_lora_memory(
        final_graph,
        base.config,
        preset.lora_config,
        max_nodes=preset.max_nodes,
        max_edges=preset.max_edges,
    )
    base_scores, final_hard_scores = _score_reference(
        queries,
        base,
        final_packed,
        preset,
        progress=progress,
        phase="base validation",
    )
    base_rows = evaluate_semantic_method(
        queries,
        final_hard_scores,
        final_graph,
        final_packed,
        stage=0,
        method="base",
        candidate_nll=base_scores,
    ).results
    scheduled = evaluation_schedule(preset)
    stage_rows = tuple(
        row
        for stage_index, adaptation in enumerate(stages, start=1)
        for stage_concepts in (
            tuple(
                cell.concept_id
                for cell in scheduled
                if cell.stage == stage_index
            ),
        )
        for stage_queries in (
            tuple(
                query
                for query in queries
                if query.concept_id in stage_concepts
            ),
        )
        for row in _evaluate_stage(
            stage_queries,
            base,
            adaptation,
            preset,
            pad_token_id,
            stage_index,
            progress=progress,
        )
    )
    return base_rows + stage_rows


def _evaluate_stage(
    queries: tuple[CompiledSemanticQuery, ...],
    base: LoadedGptNeoCheckpoint,
    adaptation: LanguageAdaptationArtifact,
    preset: QueryExperimentPreset,
    pad_token_id: int,
    stage: int,
    *,
    progress: EvaluationProgress | None,
) -> tuple[SemanticQueryResult, ...]:
    graph = adaptation.vamp_graph
    packed = pack_lora_memory(
        graph,
        base.config,
        preset.lora_config,
        max_nodes=preset.max_nodes,
        max_edges=preset.max_edges,
    )
    _base_scores, hard_scores = _score_reference(
        queries,
        base,
        packed,
        preset,
        progress=progress,
        phase=f"stage {stage} VAMP reference",
    )
    independent = tuple(
        row
        for adapter in adaptation.independent_adapters
        for candidate_scores in (
            _score_root_adapter(
                queries,
                base,
                adapter.adapter,
                preset,
                progress=progress,
                phase=f"stage {stage} independent {adapter.task_id}",
            ),
        )
        for row in evaluate_semantic_method(
            queries,
            hard_scores,
            graph,
            packed,
            stage=stage,
            method="independent",
            candidate_nll=candidate_scores,
            adapter_concept_id=str(adapter.task_id),
        ).results
    )
    sequential_scores = _score_root_adapter(
        queries,
        base,
        adaptation.sequential_stages[-1].adapter,
        preset,
        progress=progress,
        phase=f"stage {stage} sequential",
    )
    sequential = evaluate_semantic_method(
        queries,
        hard_scores,
        graph,
        packed,
        stage=stage,
        method="sequential",
        candidate_nll=sequential_scores,
    ).results
    node_indexes = {
        str(node_id): index for index, node_id in enumerate(memory_node_ids(graph))
    }
    oracle_indexes = np.asarray(
        tuple(node_indexes[query.concept_id] for query in queries),
        dtype=np.int32,
    )
    rows = np.arange(len(queries))[:, None]
    candidates = np.arange(4)[None, :]
    oracle_scores = hard_scores[rows, candidates, oracle_indexes[:, None]]
    oracle = evaluate_semantic_method(
        queries,
        hard_scores,
        graph,
        packed,
        stage=stage,
        method="vamp_oracle",
        candidate_nll=oracle_scores,
    ).results
    routed = tuple(
        row
        for start in range(0, len(queries), preset.query_chunk_size)
        for chunk_queries in (queries[start : start + preset.query_chunk_size],)
        for chunk_hard_scores in (
            hard_scores[start : start + preset.query_chunk_size],
        )
        for router, method in _ROUTER_METHODS
        for decision in (
            route_language_prefix(
                router,  # type: ignore[arg-type]
                base.params,
                base.config,
                packed,
                preset.lora_config,
                adaptation.address_book,
                stack_semantic_router_batches(chunk_queries, pad_token_id),
                random_seed=preset.seed,
                evaluation_microbatch_size=preset.query_chunk_size,
            ),
        )
        for row in evaluate_semantic_method(
            chunk_queries,
            chunk_hard_scores,
            graph,
            packed,
            stage=stage,
            method=method,
            hard_decision=decision,
        ).results
    )
    return independent + sequential + oracle + routed


def _score_reference(
    queries: tuple[CompiledSemanticQuery, ...],
    base: LoadedGptNeoCheckpoint,
    packed,
    preset: QueryExperimentPreset,
    *,
    progress: EvaluationProgress | None,
    phase: str,
) -> tuple[np.ndarray, np.ndarray]:
    chunks = tuple(
        iter_semantic_score_chunks(
            queries,
            base.params,
            base.config,
            packed,
            preset.lora_config,
            query_chunk_size=preset.query_chunk_size,
            evaluation_microbatch_size=preset.query_chunk_size,
        )
    )
    if progress is not None:
        progress(phase, len(chunks), len(chunks))
    return (
        np.concatenate(tuple(chunk.base_candidate_nll for chunk in chunks), axis=0),
        np.concatenate(tuple(chunk.hard_candidate_nll for chunk in chunks), axis=0),
    )


def _score_root_adapter(
    queries: tuple[CompiledSemanticQuery, ...],
    base: LoadedGptNeoCheckpoint,
    adapter,
    preset: QueryExperimentPreset,
    *,
    progress: EvaluationProgress | None,
    phase: str,
) -> np.ndarray:
    _graph, packed = pack_root_adapter(
        adapter,
        base.config,
        preset.lora_config,
    )
    coefficients = np.asarray(edge_coefficients_for_node(packed, 1))[None, :]
    starts = tuple(range(0, len(queries), preset.query_chunk_size))
    scores = tuple(
        score_edge_coefficient_candidates(
            base.params,
            base.config,
            packed,
            preset.lora_config,
            chunk,
            np.repeat(coefficients, len(chunk), axis=0),
            evaluation_microbatch_size=preset.query_chunk_size,
        )
        for start in starts
        for chunk in (queries[start : start + preset.query_chunk_size],)
    )
    if progress is not None:
        progress(phase, len(scores), len(scores))
    return np.concatenate(scores, axis=0)


def _queries_for_concept(
    queries: tuple[CompiledSemanticQuery, ...],
    concept_id: str,
) -> tuple[tuple[CompiledSemanticQuery, ...], tuple[int, ...]]:
    indexes = tuple(
        index for index, query in enumerate(queries) if query.concept_id == concept_id
    )
    if not indexes:
        raise ValueError(f"validation queries omit active concept {concept_id}")
    return tuple(queries[index] for index in indexes), indexes


def _accuracy(rows: tuple[SemanticQueryResult, ...]) -> float:
    if not rows:
        raise ValueError("semantic accuracy requires at least one query")
    return sum(row.answer_correct for row in rows) / len(rows)


def _validate_evaluation_inputs(
    queries: tuple[CompiledSemanticQuery, ...],
    base: LoadedGptNeoCheckpoint,
    adaptation: LanguageAdaptationArtifact,
    preset: QueryExperimentPreset,
    *,
    require_complete_manifest: bool = True,
) -> None:
    if (
        type(queries) is not tuple
        or not queries
        or any(query.split != queries[0].split for query in queries)
    ):
        raise ValueError("semantic evaluation requires one nonempty query split")
    expected_tasks = (
        preset.concept_ids
        if require_complete_manifest
        else preset.concept_ids[: len(adaptation.task_order)]
    )
    if (
        adaptation.task_order != expected_tasks
        or adaptation.base_checkpoint.manifest_sha256
        != base.reference.manifest_sha256
        or adaptation.base_checkpoint.parameter_checksum
        != base.reference.parameter_checksum
        or adaptation.model_config != base.config
        or adaptation.lora_config != preset.lora_config
        or adaptation.train_config != preset.adapter_train_config
        or adaptation.max_nodes != preset.max_nodes
        or adaptation.max_edges != preset.max_edges
    ):
        raise ValueError("semantic evaluation artifact identity changed")
    allowed = set(str(task_id) for task_id in adaptation.task_order)
    if any(query.concept_id not in allowed for query in queries):
        raise ValueError("semantic query lies outside the learned stage prefix")


__all__ = [
    "EvaluationProgress",
    "PilotBudgetEvaluation",
    "evaluate_pilot_budget",
    "evaluate_staged_semantic_queries",
]

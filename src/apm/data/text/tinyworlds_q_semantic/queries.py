"""Compilation into shared four-candidate scoring and semantic result projection."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterator

import numpy as np

from apm.continual.knowledge_evaluation import (
    KnowledgeAddressDecision,
    KnowledgeMethodEvaluation,
    KnowledgeQueryEvaluation,
    evaluate_knowledge_method,
)
from apm.continual.knowledge_tasks import KnowledgeCandidate, KnowledgeQuery
from apm.continual.language_tasks import RouterBatch, build_prefix_suffix_batches
from apm.data.text.tinyworlds_q_semantic.catalog import ValidationCatalogView
from apm.data.text.tinyworlds_q_semantic.contracts import (
    SemanticQueryCatalog,
    SemanticQueryResult,
    SemanticQueryTemplate,
)
from apm.lm.lora_memory import PackedLoraMemory
from apm.lm.candidate_scoring import (
    score_frozen_base_candidates,
    score_hard_node_candidates,
)
from apm.lm.config import GptNeoConfig
from apm.lm.lora import LoraConfig
from apm.lm.parameters import GptNeoParams
from apm.lm.text import TextTokenizer
from apm.memory.graph import MemoryGraph, NodeId


@dataclass(frozen=True, slots=True)
class CompiledSemanticQuery:
    """Query-specific metadata wrapped around the existing scorer contract."""

    catalog_sha256: str
    concept_id: str
    fact_id: str
    template_id: str
    direction: str
    split: str
    knowledge_query: KnowledgeQuery

    def __post_init__(self) -> None:
        if self.knowledge_query.query_id != self.template_id:
            raise ValueError("compiled semantic query IDs changed")
        if self.knowledge_query.task_id != self.concept_id:
            raise ValueError("compiled semantic query task changed")
        if self.knowledge_query.proof_id != self.fact_id:
            raise ValueError("compiled semantic query fact changed")


@dataclass(frozen=True, slots=True)
class SemanticMethodEvaluation:
    """Shared knowledge evaluation plus its query-native semantic projection."""

    knowledge: KnowledgeMethodEvaluation
    results: tuple[SemanticQueryResult, ...]

    def __post_init__(self) -> None:
        if len(self.knowledge.queries) != len(self.results):
            raise ValueError("semantic projection must cover every knowledge query")


@dataclass(frozen=True, eq=False, slots=True)
class SemanticQueryScoreChunk:
    """One bounded query chunk's base and all-hard-node candidate NLLs."""

    queries: tuple[CompiledSemanticQuery, ...]
    base_candidate_nll: np.ndarray
    hard_candidate_nll: np.ndarray

    def __post_init__(self) -> None:
        base = np.asarray(self.base_candidate_nll, dtype=np.float32).copy()
        hard = np.asarray(self.hard_candidate_nll, dtype=np.float32).copy()
        if base.shape != (len(self.queries), 4):
            raise ValueError("semantic base score chunk must have shape [query, 4]")
        if hard.ndim != 3 or hard.shape[:2] != (len(self.queries), 4):
            raise ValueError(
                "semantic hard score chunk must have shape [query, 4, node]"
            )
        if np.any(~np.isfinite(base)) or np.any(base < 0.0):
            raise ValueError("semantic base score chunk contains invalid NLL")
        valid_hard = hard[np.isfinite(hard)]
        if (
            np.any(valid_hard < 0.0)
            or np.any(np.isnan(hard))
            or np.any(np.isneginf(hard))
        ):
            raise ValueError("semantic hard score chunk contains invalid NLL")
        base.flags.writeable = False
        hard.flags.writeable = False
        object.__setattr__(self, "base_candidate_nll", base)
        object.__setattr__(self, "hard_candidate_nll", hard)


def compile_semantic_queries(
    catalog: SemanticQueryCatalog | ValidationCatalogView,
    tokenizer: TextTokenizer,
    *,
    split: str = "validation",
    maximum_context_tokens: int = 256,
) -> tuple[CompiledSemanticQuery, ...]:
    """Compile reviewed prompts while rechecking every stored tokenizer boundary."""
    if split not in ("validation", "test"):
        raise ValueError("semantic query split must be validation or test")
    if isinstance(catalog, ValidationCatalogView) and split != "validation":
        raise PermissionError("a validation-only catalog cannot compile sealed queries")
    fact_by_id = {fact.fact_id: fact for fact in catalog.facts}
    templates = tuple(template for template in catalog.templates if template.split == split)
    compiled = tuple(
        _compile_template(
            catalog.catalog_sha256,
            fact_by_id[template.fact_id].concept_id,
            fact_by_id[template.fact_id].supporting_story_groups,
            template,
            tokenizer,
            maximum_context_tokens,
        )
        for template in templates
    )
    if not compiled:
        raise ValueError(f"catalog contains no {split} semantic queries")
    return compiled


def _compile_template(
    catalog_sha256: str,
    concept_id: str,
    support_groups: tuple[str, ...],
    template: SemanticQueryTemplate,
    tokenizer: TextTokenizer,
    maximum_context_tokens: int,
) -> CompiledSemanticQuery:
    prompt_tokens = tokenizer.encode(template.prompt_text)
    if prompt_tokens != template.prompt_token_ids:
        raise ValueError(f"template tokenizer changed: {template.template_id}")
    candidates_text = tuple(f" {answer}" for answer in template.candidate_answer_forms)
    combined = tuple(
        tokenizer.encode(template.prompt_text + answer) for answer in candidates_text
    )
    if combined != template.combined_candidate_token_ids:
        raise ValueError(f"template combined tokenization changed: {template.template_id}")
    suffix_lengths = tuple(len(tokens) - len(prompt_tokens) for tokens in combined)
    if len(set(suffix_lengths)) != 1 or suffix_lengths[0] < 1:
        raise ValueError("semantic candidates must retain equal positive answer lengths")
    if any(len(tokens) > maximum_context_tokens for tokens in combined):
        raise ValueError("semantic query exceeds the configured model context")
    batches = tuple(
        build_prefix_suffix_batches(
            tokens,
            len(prompt_tokens),
            suffix_lengths[0],
            pad_token_id=tokenizer.pad_token_id,
        )
        for tokens in combined
    )
    candidates = tuple(
        KnowledgeCandidate(answer, competence)
        for answer, (_, competence) in zip(candidates_text, batches)
    )
    query = KnowledgeQuery(
        query_id=template.template_id,
        task_id=concept_id,
        family_id=BENCHMARK_FAMILY_ID,
        query_kind=template.direction,
        candidates=candidates,  # type: ignore[arg-type]
        router_batch=batches[0][0],
        correct_candidate_index=template.correct_candidate_index,
        proof_id=template.fact_id,
        support_ids=(template.fact_id, *support_groups),
        required_edge_ids=(NodeId(concept_id),),
        cue_regime=("cue_sufficient" if template.direction == "forward" else "cue_present"),
        visible_cue_ids=(concept_id,) if template.direction == "forward" else (template.fact_id,),
        eligible_task_ids=(concept_id,),
        novelty_regime="registered_semantic_fact",
        reasoning_type="direct_semantic",
        reasoning_depth=0,
        prefix_length=len(prompt_tokens),
        mode="closed_book",
        oracle_node_ids=(NodeId(concept_id),),
    )
    return CompiledSemanticQuery(
        catalog_sha256=catalog_sha256,
        concept_id=concept_id,
        fact_id=template.fact_id,
        template_id=template.template_id,
        direction=template.direction,
        split=template.split,
        knowledge_query=query,
    )


BENCHMARK_FAMILY_ID = "tinyworlds-q-semantic"


def evaluate_semantic_method(
    queries: tuple[CompiledSemanticQuery, ...],
    hard_candidate_nll: np.ndarray,
    graph: MemoryGraph[object],
    packed_memory: PackedLoraMemory,
    *,
    stage: int,
    method: str,
    candidate_nll: np.ndarray | None = None,
    hard_decision: KnowledgeAddressDecision | None = None,
    edge_coefficients: np.ndarray | None = None,
    unavailable_node_ids: tuple[str, ...] = (),
    unavailable_edge_ids: tuple[str, ...] = (),
    adapter_concept_id: str | None = None,
) -> SemanticMethodEvaluation:
    """Reuse shared candidate, hard-node, routing, support, and regret machinery."""
    if not queries:
        raise ValueError("semantic evaluation requires at least one query")
    catalog_sha256 = queries[0].catalog_sha256
    if any(query.catalog_sha256 != catalog_sha256 for query in queries):
        raise ValueError("semantic evaluation cannot mix catalogs")
    knowledge = evaluate_knowledge_method(
        tuple(query.knowledge_query for query in queries),
        hard_candidate_nll,
        graph,
        packed_memory,
        stage=stage,
        method=method,
        candidate_nll=candidate_nll,
        hard_decision=hard_decision,
        edge_coefficients=edge_coefficients,
        unavailable_node_ids=unavailable_node_ids,
        unavailable_edge_ids=unavailable_edge_ids,
    )
    results = tuple(
        project_semantic_result(query, row, adapter_concept_id=adapter_concept_id)
        for query, row in zip(queries, knowledge.queries)
    )
    return SemanticMethodEvaluation(knowledge, results)


def project_semantic_result(
    query: CompiledSemanticQuery,
    result: KnowledgeQueryEvaluation,
    *,
    adapter_concept_id: str | None = None,
) -> SemanticQueryResult:
    """Project one shared scorer row without changing any numeric result."""
    if (
        result.query_id != query.template_id
        or result.task_id != query.concept_id
        or result.proof_id != query.fact_id
    ):
        raise ValueError("knowledge result does not align with semantic query metadata")
    return SemanticQueryResult(
        stage=result.stage,
        method=result.method,
        concept_id=query.concept_id,
        fact_id=query.fact_id,
        template_id=query.template_id,
        direction=query.direction,  # type: ignore[arg-type]
        split=query.split,  # type: ignore[arg-type]
        adapter_concept_id=adapter_concept_id,
        candidate_nll=tuple(float(value) for value in result.candidate_nll),  # type: ignore[arg-type]
        correct_candidate_index=result.correct_candidate_index,
        predicted_candidate_index=result.predicted_candidate_index,
        answer_correct=result.candidate_correct,
        correct_answer_margin=result.candidate_margin,
        selected_node_index=result.selected_node_index,
        oracle_node_index=result.task_oracle_node_index,
        routed_regret=result.routed_regret,
    )


def validation_question_prefixes(
    catalog: ValidationCatalogView,
) -> tuple[tuple[int, ...], ...]:
    """Expose validation-only prefixes for VAMP parents and router-key construction."""
    if type(catalog) is not ValidationCatalogView:
        raise TypeError("parent and key prefixes require a validation catalog view")
    return tuple(template.prompt_token_ids for template in catalog.templates)


def stack_semantic_router_batches(
    queries: tuple[CompiledSemanticQuery, ...],
    pad_token_id: int,
) -> RouterBatch:
    """Right-pad and stack query prefixes without exposing answer tokens."""
    if (
        type(queries) is not tuple
        or not queries
        or any(type(query) is not CompiledSemanticQuery for query in queries)
    ):
        raise ValueError("semantic routing requires compiled query values")
    if type(pad_token_id) is not int or pad_token_id < 0:
        raise ValueError("semantic routing pad_token_id must be nonnegative")
    batches = tuple(query.knowledge_query.router_batch for query in queries)
    if any(batch.input_ids.shape[0] != 1 for batch in batches):
        raise ValueError("each semantic router prefix must contain exactly one row")
    maximum_width = max(batch.input_ids.shape[1] for batch in batches)

    def stack(field: str, padding_value: int | bool) -> np.ndarray:
        return np.concatenate(
            tuple(
                np.pad(
                    getattr(batch, field),
                    ((0, 0), (0, maximum_width - batch.input_ids.shape[1])),
                    constant_values=padding_value,
                )
                for batch in batches
            ),
            axis=0,
        )

    return RouterBatch(
        input_ids=stack("input_ids", pad_token_id),
        attention_mask=stack("attention_mask", False),
        target_ids=stack("target_ids", pad_token_id),
        loss_mask=stack("loss_mask", False),
    )


def iter_semantic_score_chunks(
    queries: tuple[CompiledSemanticQuery, ...],
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    packed_memory: PackedLoraMemory,
    lora_config: LoraConfig,
    *,
    query_chunk_size: int,
    evaluation_microbatch_size: int | None = None,
) -> Iterator[SemanticQueryScoreChunk]:
    """Score base and hard nodes without placing all query/node rows on device."""
    if not queries:
        raise ValueError("semantic chunk scoring requires at least one query")
    if type(query_chunk_size) is not int or query_chunk_size <= 0:
        raise ValueError("semantic query_chunk_size must be positive")
    for start in range(0, len(queries), query_chunk_size):
        chunk = queries[start : start + query_chunk_size]
        knowledge_queries = tuple(item.knowledge_query for item in chunk)
        yield SemanticQueryScoreChunk(
            queries=chunk,
            base_candidate_nll=score_frozen_base_candidates(
                base_params,
                model_config,
                knowledge_queries,
                evaluation_microbatch_size=evaluation_microbatch_size,
            ),
            hard_candidate_nll=score_hard_node_candidates(
                base_params,
                model_config,
                packed_memory,
                lora_config,
                knowledge_queries,
                evaluation_microbatch_size=evaluation_microbatch_size,
            ),
        )


__all__ = [
    "BENCHMARK_FAMILY_ID",
    "CompiledSemanticQuery",
    "SemanticMethodEvaluation",
    "SemanticQueryScoreChunk",
    "compile_semantic_queries",
    "evaluate_semantic_method",
    "iter_semantic_score_chunks",
    "project_semantic_result",
    "stack_semantic_router_batches",
    "validation_question_prefixes",
]

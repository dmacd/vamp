"""Shared frozen, hard-node, and continuous-coefficient candidate scorers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from apm.lm.config import GptNeoConfig
from apm.lm.evaluation import (
    evaluation_microbatch_slices,
    validate_evaluation_microbatch_size,
)
from apm.lm.gpt_neo import apply_gpt_neo
from apm.lm.lora import LoraConfig
from apm.lm.lora_memory import PackedLoraMemory
from apm.lm.losses import per_token_nll
from apm.lm.parameters import GptNeoParams

if TYPE_CHECKING:
    from apm.continual.knowledge_tasks import KnowledgeQuery
    from apm.continual.language_tasks import CompetenceBatch


_CANDIDATES_PER_QUERY = 4


def active_token_candidate_nll(
    logits: jax.Array,
    competence_batch: CompetenceBatch,
) -> jax.Array:
    """Return per-row NLL normalized only over active candidate-answer tokens."""
    from apm.continual.language_tasks import CompetenceBatch as BatchContract

    if not isinstance(competence_batch, BatchContract):
        raise TypeError("candidate NLL requires a CompetenceBatch")
    score_logits = jnp.asarray(logits)
    expected_shape = competence_batch.target_ids.shape
    if (
        score_logits.ndim != 3
        or score_logits.shape[:2] != expected_shape
        or score_logits.shape[-1] <= 0
    ):
        raise ValueError(
            "candidate logits must have shape [batch, sequence, vocabulary]"
        )
    loss_mask = jnp.asarray(competence_batch.loss_mask, dtype=jnp.float32)
    active_tokens = jnp.sum(loss_mask, axis=-1)
    token_nll = per_token_nll(
        score_logits,
        jnp.asarray(competence_batch.target_ids, dtype=jnp.int32),
    )
    return (
        jnp.sum(token_nll * loss_mask, axis=-1) / active_tokens
    ).astype(jnp.float32)


def score_frozen_base_candidates(
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    queries: tuple[KnowledgeQuery, ...],
    *,
    evaluation_microbatch_size: int | None = None,
) -> np.ndarray:
    """Score four candidates per query under the frozen base as ``[query, 4]``."""
    buckets = _query_shape_buckets(queries)
    if len(buckets) > 1:
        scores = np.empty((len(queries), _CANDIDATES_PER_QUERY), dtype=np.float32)
        for indices, bucket in buckets:
            scores[np.asarray(indices)] = score_frozen_base_candidates(
                base_params,
                model_config,
                bucket,
                evaluation_microbatch_size=evaluation_microbatch_size,
            )
        return _immutable_scores(scores)
    flattened_batch = _flatten_candidate_batches(queries)
    scores = _score_candidate_rows(
        base_params,
        model_config,
        flattened_batch,
        evaluation_microbatch_size=evaluation_microbatch_size,
    )
    return _immutable_scores(
        scores.reshape((len(queries), _CANDIDATES_PER_QUERY))
    )


def score_hard_node_candidates(
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    packed_memory: PackedLoraMemory,
    lora_config: LoraConfig,
    queries: tuple[KnowledgeQuery, ...],
    *,
    evaluation_microbatch_size: int | None = None,
) -> np.ndarray:
    """Score every valid hard path as ``[query, 4, node_capacity]``."""
    _validate_lora_execution(packed_memory, lora_config)
    buckets = _query_shape_buckets(queries)
    if len(buckets) > 1:
        node_capacity = packed_memory.node_path_matrix.shape[0]
        scores = np.empty(
            (len(queries), _CANDIDATES_PER_QUERY, node_capacity),
            dtype=np.float32,
        )
        for indices, bucket in buckets:
            scores[np.asarray(indices)] = score_hard_node_candidates(
                base_params,
                model_config,
                packed_memory,
                lora_config,
                bucket,
                evaluation_microbatch_size=evaluation_microbatch_size,
            )
        return _immutable_scores(scores)
    flattened_batch = _flatten_candidate_batches(queries)
    valid_node_mask = np.asarray(packed_memory.valid_node_mask, dtype=np.bool_)
    valid_node_indices = np.flatnonzero(valid_node_mask)
    path_matrix = np.asarray(packed_memory.node_path_matrix, dtype=np.float32)
    repeated_batch = _repeat_competence_rows(
        flattened_batch,
        int(valid_node_indices.size),
    )
    repeated_coefficients = np.tile(
        path_matrix[valid_node_indices],
        (flattened_batch.input_ids.shape[0], 1),
    )
    valid_scores = _score_candidate_rows(
        base_params,
        model_config,
        repeated_batch,
        packed_memory=packed_memory,
        lora_config=lora_config,
        edge_coefficients=repeated_coefficients,
        evaluation_microbatch_size=evaluation_microbatch_size,
    ).reshape((flattened_batch.input_ids.shape[0], valid_node_indices.size))
    node_capacity = packed_memory.node_path_matrix.shape[0]
    scores = np.full(
        (flattened_batch.input_ids.shape[0], node_capacity),
        np.inf,
        dtype=np.float32,
    )
    scores[:, valid_node_indices] = valid_scores
    return _immutable_scores(
        scores.reshape(
            (len(queries), _CANDIDATES_PER_QUERY, node_capacity)
        )
    )


def score_edge_coefficient_candidates(
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    packed_memory: PackedLoraMemory,
    lora_config: LoraConfig,
    queries: tuple[KnowledgeQuery, ...],
    edge_coefficients: jax.Array | np.ndarray,
    *,
    evaluation_microbatch_size: int | None = None,
) -> np.ndarray:
    """Score four candidates under arbitrary per-query edge coefficients."""
    _validate_lora_execution(packed_memory, lora_config)
    coefficients = np.asarray(edge_coefficients, dtype=np.float32)
    expected_shape = (len(queries), packed_memory.node_path_matrix.shape[1])
    if coefficients.shape != expected_shape:
        raise ValueError(f"edge_coefficients must have shape {expected_shape}")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("edge_coefficients must be finite")
    buckets = _query_shape_buckets(queries)
    if len(buckets) > 1:
        scores = np.empty((len(queries), _CANDIDATES_PER_QUERY), dtype=np.float32)
        for indices, bucket in buckets:
            index_array = np.asarray(indices)
            scores[index_array] = score_edge_coefficient_candidates(
                base_params,
                model_config,
                packed_memory,
                lora_config,
                bucket,
                coefficients[index_array],
                evaluation_microbatch_size=evaluation_microbatch_size,
            )
        return _immutable_scores(scores)
    flattened_batch = _flatten_candidate_batches(queries)
    repeated_coefficients = np.repeat(
        coefficients,
        _CANDIDATES_PER_QUERY,
        axis=0,
    )
    scores = _score_candidate_rows(
        base_params,
        model_config,
        flattened_batch,
        packed_memory=packed_memory,
        lora_config=lora_config,
        edge_coefficients=repeated_coefficients,
        evaluation_microbatch_size=evaluation_microbatch_size,
    )
    return _immutable_scores(
        scores.reshape((len(queries), _CANDIDATES_PER_QUERY))
    )


def _score_candidate_rows(
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    competence_batch: CompetenceBatch,
    *,
    packed_memory: PackedLoraMemory | None = None,
    lora_config: LoraConfig | None = None,
    edge_coefficients: np.ndarray | None = None,
    evaluation_microbatch_size: int | None = None,
) -> np.ndarray:
    microbatch_size = validate_evaluation_microbatch_size(
        evaluation_microbatch_size
    )
    uses_lora = tuple(
        value is not None
        for value in (packed_memory, lora_config, edge_coefficients)
    )
    if any(uses_lora) and not all(uses_lora):
        raise ValueError(
            "packed memory, LoRA config, and edge coefficients must be supplied together"
        )
    if edge_coefficients is not None and (
        edge_coefficients.ndim != 2
        or edge_coefficients.shape[0] != competence_batch.input_ids.shape[0]
    ):
        raise ValueError("edge coefficient rows must match candidate rows")

    chunks = tuple(
        _score_candidate_slice(
            base_params,
            model_config,
            _slice_competence_batch(competence_batch, row_slice),
            packed_memory=packed_memory,
            lora_config=lora_config,
            edge_coefficients=(
                None
                if edge_coefficients is None
                else edge_coefficients[row_slice]
            ),
        )
        for row_slice in evaluation_microbatch_slices(
            competence_batch.input_ids.shape[0],
            microbatch_size,
        )
    )
    return np.concatenate(chunks, axis=0)


def _score_candidate_slice(
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    competence_batch: CompetenceBatch,
    *,
    packed_memory: PackedLoraMemory | None,
    lora_config: LoraConfig | None,
    edge_coefficients: np.ndarray | None,
) -> np.ndarray:
    model_kwargs = (
        {}
        if packed_memory is None
        else {
            "lora_memory": packed_memory,
            "edge_coefficients": jnp.asarray(
                edge_coefficients,
                dtype=jnp.float32,
            ),
            "lora_config": lora_config,
        }
    )
    logits = apply_gpt_neo(
        base_params,
        model_config,
        jnp.asarray(competence_batch.input_ids, dtype=jnp.int32),
        jnp.asarray(competence_batch.attention_mask, dtype=jnp.bool_),
        training=False,
        **model_kwargs,
    ).logits
    return np.asarray(
        active_token_candidate_nll(logits, competence_batch),
        dtype=np.float32,
    )


def _flatten_candidate_batches(
    queries: tuple[KnowledgeQuery, ...],
) -> CompetenceBatch:
    _validate_queries(queries)
    candidate_batches = tuple(
        candidate.competence_batch
        for query in queries
        for candidate in query.candidates
    )
    return _concatenate_competence_batches(candidate_batches)


def _validate_queries(queries: tuple[KnowledgeQuery, ...]) -> None:
    from apm.continual.knowledge_tasks import KnowledgeQuery as QueryContract

    if (
        not isinstance(queries, tuple)
        or not queries
        or any(not isinstance(query, QueryContract) for query in queries)
    ):
        raise ValueError("candidate scoring requires a nonempty tuple of KnowledgeQuery values")
    query_ids = tuple(query.query_id for query in queries)
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("candidate-scoring query IDs must be unique")
    if any(
        len(
            {
                candidate.competence_batch.input_ids.shape
                for candidate in query.candidates
            }
        )
        != 1
        for query in queries
    ):
        raise ValueError("each query's four candidate batches must share one shape")


def _query_shape_buckets(
    queries: tuple[KnowledgeQuery, ...],
) -> tuple[tuple[tuple[int, ...], tuple[KnowledgeQuery, ...]], ...]:
    _validate_queries(queries)
    grouped: dict[tuple[int, ...], list[tuple[int, KnowledgeQuery]]] = {}
    for index, query in enumerate(queries):
        shape = query.candidates[0].competence_batch.input_ids.shape
        grouped.setdefault(shape, []).append((index, query))
    return tuple(
        (
            tuple(index for index, _ in values),
            tuple(query for _, query in values),
        )
        for values in grouped.values()
    )


def _concatenate_competence_batches(
    batches: tuple[CompetenceBatch, ...],
) -> CompetenceBatch:
    from apm.continual.language_tasks import CompetenceBatch as BatchContract

    return BatchContract(
        input_ids=np.concatenate(tuple(batch.input_ids for batch in batches), axis=0),
        attention_mask=np.concatenate(
            tuple(batch.attention_mask for batch in batches),
            axis=0,
        ),
        target_ids=np.concatenate(tuple(batch.target_ids for batch in batches), axis=0),
        loss_mask=np.concatenate(tuple(batch.loss_mask for batch in batches), axis=0),
    )


def _repeat_competence_rows(
    batch: CompetenceBatch,
    repeats: int,
) -> CompetenceBatch:
    from apm.continual.language_tasks import CompetenceBatch as BatchContract

    if type(repeats) is not int or repeats <= 0:
        raise ValueError("candidate row repetitions must be positive")
    return BatchContract(
        input_ids=np.repeat(batch.input_ids, repeats, axis=0),
        attention_mask=np.repeat(batch.attention_mask, repeats, axis=0),
        target_ids=np.repeat(batch.target_ids, repeats, axis=0),
        loss_mask=np.repeat(batch.loss_mask, repeats, axis=0),
    )


def _slice_competence_batch(
    batch: CompetenceBatch,
    row_slice: slice,
) -> CompetenceBatch:
    from apm.continual.language_tasks import CompetenceBatch as BatchContract

    return BatchContract(
        input_ids=batch.input_ids[row_slice],
        attention_mask=batch.attention_mask[row_slice],
        target_ids=batch.target_ids[row_slice],
        loss_mask=batch.loss_mask[row_slice],
    )


def _validate_lora_execution(
    packed_memory: PackedLoraMemory,
    lora_config: LoraConfig,
) -> None:
    if not isinstance(packed_memory, PackedLoraMemory):
        raise TypeError("packed_memory must be a PackedLoraMemory")
    if not isinstance(lora_config, LoraConfig):
        raise TypeError("lora_config must be a LoraConfig")
    node_path_matrix = np.asarray(packed_memory.node_path_matrix)
    valid_node_mask = np.asarray(packed_memory.valid_node_mask)
    valid_edge_mask = np.asarray(packed_memory.valid_edge_mask)
    if (
        node_path_matrix.ndim != 2
        or valid_node_mask.shape != (node_path_matrix.shape[0],)
        or valid_edge_mask.shape != (node_path_matrix.shape[1],)
        or valid_node_mask.dtype != np.dtype(np.bool_)
        or valid_edge_mask.dtype != np.dtype(np.bool_)
        or not np.any(valid_node_mask)
    ):
        raise ValueError("packed memory has inconsistent path and validity arrays")


def _immutable_scores(values: np.ndarray) -> np.ndarray:
    scores = np.array(values, dtype=np.float32, copy=True)
    scores.flags.writeable = False
    return scores


__all__ = [
    "active_token_candidate_nll",
    "score_edge_coefficient_candidates",
    "score_frozen_base_candidates",
    "score_hard_node_candidates",
]

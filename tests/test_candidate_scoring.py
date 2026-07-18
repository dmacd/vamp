from __future__ import annotations

from dataclasses import replace
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import apm.lm.candidate_scoring as candidate_scoring_module
from apm.continual.knowledge_tasks import KnowledgeCandidate, KnowledgeQuery
from apm.continual.language_tasks import (
    CompetenceBatch,
    NodeId,
    TaskId,
    build_prefix_suffix_batches,
)
from apm.lm.candidate_scoring import (
    active_token_candidate_nll,
    score_edge_coefficient_candidates,
    score_frozen_base_candidates,
    score_hard_node_candidates,
)
from apm.lm.config import GptNeoConfig
from apm.lm.lora import LoraConfig, init_lora_edge
from apm.lm.lora_memory import (
    node_weights_to_edge_coefficients,
    pack_lora_memory,
)
from apm.lm.parameters import init_gpt_neo_params
from apm.memory.graph import add_memory_node, init_memory_graph


def _model_config() -> GptNeoConfig:
    return GptNeoConfig(
        vocab_size=16,
        max_position_embeddings=8,
        hidden_size=8,
        intermediate_size=16,
        num_layers=1,
        num_heads=2,
        attention_types=("global",),
        local_window_size=4,
    )


def _candidate(
    answer_text: str,
    prefix: tuple[int, ...],
    first_answer_token: int,
    *,
    suffix_length: int = 2,
) -> KnowledgeCandidate:
    answer_tokens = tuple(
        first_answer_token + offset for offset in range(suffix_length)
    )
    return KnowledgeCandidate(
        answer_text,
        build_prefix_suffix_batches(
            prefix + answer_tokens,
            prefix_length=len(prefix),
            suffix_length=suffix_length,
        )[1],
    )


def _query(
    query_index: int,
    *,
    suffix_length: int = 2,
) -> KnowledgeQuery:
    prefix = (1 + query_index, 2 + query_index, 3 + query_index)
    task_id = TaskId(f"task-{query_index}")
    router_batch = build_prefix_suffix_batches(
        prefix + (7, 8),
        prefix_length=len(prefix),
        suffix_length=2,
    )[0]
    return KnowledgeQuery(
        query_id=f"query-{query_index}",
        task_id=task_id,
        family_id="willow",
        query_kind="direct",
        candidates=tuple(
            _candidate(
                answer,
                prefix,
                7 + candidate_index,
                suffix_length=suffix_length,
            )
            for candidate_index, answer in enumerate(("amber", "blue", "coral", "dune"))
        ),
        router_batch=router_batch,
        correct_candidate_index=query_index % 4,
        proof_id=f"proof-{query_index}",
        support_ids=(f"fact-{query_index}",),
        required_edge_ids=(NodeId(f"node-{query_index + 1}"),),
        cue_regime="cue_present",
        visible_cue_ids=("cue-family",),
        eligible_task_ids=(task_id, TaskId("competing-task")),
        novelty_regime="direct",
        reasoning_type="direct",
        reasoning_depth=0,
        prefix_length=len(prefix),
        mode="closed_book",
        oracle_node_ids=(NodeId(f"node-{query_index + 1}"),),
    )


def _stack_batches(batches: tuple[CompetenceBatch, ...]) -> CompetenceBatch:
    return CompetenceBatch(
        input_ids=np.concatenate(tuple(batch.input_ids for batch in batches)),
        attention_mask=np.concatenate(tuple(batch.attention_mask for batch in batches)),
        target_ids=np.concatenate(tuple(batch.target_ids for batch in batches)),
        loss_mask=np.concatenate(tuple(batch.loss_mask for batch in batches)),
    )


def _scoring_setup():
    model_config = _model_config()
    lora_config = LoraConfig(rank=1, alpha=1.0)
    base_params = init_gpt_neo_params(jax.random.PRNGKey(0), model_config)
    initialized_edges = tuple(
        init_lora_edge(jax.random.PRNGKey(seed), model_config, lora_config)
        for seed in (1, 2)
    )
    edges = tuple(
        jax.tree_util.tree_map(
            lambda leaf: jnp.full_like(leaf, 0.02 * edge_index),
            edge,
        )
        for edge_index, edge in enumerate(initialized_edges, start=1)
    )
    graph = add_memory_node(
        init_memory_graph(NodeId("root")),
        NodeId("node-1"),
        NodeId("root"),
        TaskId("task-0"),
        1,
        edges[0],
    )
    graph = add_memory_node(
        graph,
        NodeId("node-2"),
        NodeId("node-1"),
        TaskId("task-1"),
        2,
        edges[1],
    )
    packed_memory = pack_lora_memory(
        graph,
        model_config,
        lora_config,
        max_nodes=4,
        max_edges=3,
    )
    return (
        base_params,
        model_config,
        packed_memory,
        lora_config,
        (_query(0), _query(1)),
    )


def test_active_token_nll_ignores_context_and_selects_synthetic_answer() -> None:
    query = _query(0)
    batch = _stack_batches(
        tuple(candidate.competence_batch for candidate in query.candidates)
    )
    vocabulary_size = _model_config().vocab_size
    neutral_logits = np.zeros(
        (*batch.input_ids.shape, vocabulary_size),
        dtype=np.float32,
    )
    context_mutated = neutral_logits.copy()
    context_mutated[:, :2, :] = np.linspace(
        -20.0,
        20.0,
        vocabulary_size,
        dtype=np.float32,
    )

    neutral_nll = active_token_candidate_nll(neutral_logits, batch)
    context_nll = active_token_candidate_nll(context_mutated, batch)
    np.testing.assert_allclose(neutral_nll, math.log(vocabulary_size), atol=1e-6)
    np.testing.assert_array_equal(context_nll, neutral_nll)

    unequal_token_batch = _stack_batches(
        (
            query.candidates[0].competence_batch,
            build_prefix_suffix_batches((1, 2, 3, 7), 3, 2)[1],
        )
    )
    unequal_token_logits = np.zeros(
        (*unequal_token_batch.input_ids.shape, vocabulary_size),
        dtype=np.float32,
    )
    np.testing.assert_allclose(
        active_token_candidate_nll(unequal_token_logits, unequal_token_batch),
        math.log(vocabulary_size),
        atol=1e-6,
    )

    favorable_logits = neutral_logits.copy()
    expected_answer = 2
    for position in np.flatnonzero(batch.loss_mask[expected_answer]):
        target = batch.target_ids[expected_answer, position]
        favorable_logits[expected_answer, position, target] = 8.0
    candidate_nll = np.asarray(active_token_candidate_nll(favorable_logits, batch))
    assert int(np.argmin(candidate_nll)) == expected_answer
    np.testing.assert_array_equal(
        jax.jit(active_token_candidate_nll)(favorable_logits, batch),
        candidate_nll,
    )


def test_frozen_and_hard_node_scores_have_exact_shapes_and_root_equivalence() -> None:
    base_params, model_config, packed, lora_config, queries = _scoring_setup()

    frozen = score_frozen_base_candidates(base_params, model_config, queries)
    hard = score_hard_node_candidates(
        base_params,
        model_config,
        packed,
        lora_config,
        queries,
    )

    assert frozen.shape == (2, 4)
    assert hard.shape == (2, 4, 4)
    assert not frozen.flags.writeable and not hard.flags.writeable
    np.testing.assert_allclose(hard[:, :, 0], frozen, rtol=1e-6, atol=1e-6)
    assert np.all(np.isfinite(hard[:, :, :3]))
    assert np.all(np.isinf(hard[:, :, 3]))


def test_one_hot_soft_coefficients_equal_corresponding_hard_paths() -> None:
    base_params, model_config, packed, lora_config, queries = _scoring_setup()
    hard = score_hard_node_candidates(
        base_params,
        model_config,
        packed,
        lora_config,
        queries,
    )
    selected_nodes = np.asarray((1, 2), dtype=np.int32)
    coefficients = node_weights_to_edge_coefficients(
        jax.nn.one_hot(selected_nodes, packed.node_path_matrix.shape[0]),
        packed,
    )

    soft = score_edge_coefficient_candidates(
        base_params,
        model_config,
        packed,
        lora_config,
        queries,
        coefficients,
    )

    expected = np.stack(
        tuple(
            hard[query_index, :, node_index]
            for query_index, node_index in enumerate(selected_nodes)
        )
    )
    np.testing.assert_allclose(soft, expected, rtol=1e-6, atol=1e-6)


def test_microbatched_candidate_scores_match_unbatched_shared_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_params, model_config, packed, lora_config, queries = _scoring_setup()
    one_hot_coefficients = node_weights_to_edge_coefficients(
        jax.nn.one_hot(np.asarray((1, 2)), packed.node_path_matrix.shape[0]),
        packed,
    )
    expected = (
        score_frozen_base_candidates(base_params, model_config, queries),
        score_hard_node_candidates(
            base_params,
            model_config,
            packed,
            lora_config,
            queries,
        ),
        score_edge_coefficient_candidates(
            base_params,
            model_config,
            packed,
            lora_config,
            queries,
            one_hot_coefficients,
        ),
    )
    original_apply = candidate_scoring_module.apply_gpt_neo
    observed_batch_sizes: list[int] = []

    def counted_apply(*args, **kwargs):
        observed_batch_sizes.append(int(args[2].shape[0]))
        return original_apply(*args, **kwargs)

    monkeypatch.setattr(candidate_scoring_module, "apply_gpt_neo", counted_apply)
    actual = (
        score_frozen_base_candidates(
            base_params,
            model_config,
            queries,
            evaluation_microbatch_size=3,
        ),
        score_hard_node_candidates(
            base_params,
            model_config,
            packed,
            lora_config,
            queries,
            evaluation_microbatch_size=3,
        ),
        score_edge_coefficient_candidates(
            base_params,
            model_config,
            packed,
            lora_config,
            queries,
            one_hot_coefficients,
            evaluation_microbatch_size=3,
        ),
    )

    assert observed_batch_sizes and max(observed_batch_sizes) == 3
    for expected_scores, actual_scores in zip(expected, actual):
        np.testing.assert_allclose(
            actual_scores,
            expected_scores,
            rtol=1e-6,
            atol=1e-6,
        )


def test_candidate_scorers_reject_misaligned_queries_coefficients_and_logits() -> None:
    base_params, model_config, packed, lora_config, queries = _scoring_setup()

    with pytest.raises(ValueError, match="nonempty tuple"):
        score_frozen_base_candidates(base_params, model_config, ())
    with pytest.raises(ValueError, match="query IDs must be unique"):
        score_frozen_base_candidates(
            base_params,
            model_config,
            (queries[0], queries[0]),
        )
    variable_width_queries = (queries[0], _query(2, suffix_length=3))
    bucketed_scores = score_frozen_base_candidates(
        base_params,
        model_config,
        variable_width_queries,
    )
    np.testing.assert_allclose(
        bucketed_scores,
        np.concatenate(
            tuple(
                score_frozen_base_candidates(
                    base_params,
                    model_config,
                    (query,),
                )
                for query in variable_width_queries
            ),
            axis=0,
        ),
        rtol=1e-6,
        atol=1e-6,
    )
    bucketed_hard = score_hard_node_candidates(
        base_params,
        model_config,
        packed,
        lora_config,
        variable_width_queries,
    )
    np.testing.assert_allclose(
        bucketed_hard,
        np.concatenate(
            tuple(
                score_hard_node_candidates(
                    base_params,
                    model_config,
                    packed,
                    lora_config,
                    (query,),
                )
                for query in variable_width_queries
            ),
            axis=0,
        ),
        rtol=1e-6,
        atol=1e-6,
    )
    variable_coefficients = np.zeros((2, 3), dtype=np.float32)
    bucketed_soft = score_edge_coefficient_candidates(
        base_params,
        model_config,
        packed,
        lora_config,
        variable_width_queries,
        variable_coefficients,
    )
    np.testing.assert_allclose(
        bucketed_soft,
        np.concatenate(
            tuple(
                score_edge_coefficient_candidates(
                    base_params,
                    model_config,
                    packed,
                    lora_config,
                    (query,),
                    variable_coefficients[index : index + 1],
                )
                for index, query in enumerate(variable_width_queries)
            ),
            axis=0,
        ),
        rtol=1e-6,
        atol=1e-6,
    )
    with pytest.raises(ValueError, match="must have shape"):
        score_edge_coefficient_candidates(
            base_params,
            model_config,
            packed,
            lora_config,
            queries,
            np.zeros((2, 2), dtype=np.float32),
        )
    invalid_coefficients = np.zeros((2, 3), dtype=np.float32)
    invalid_coefficients[0, 0] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        score_edge_coefficient_candidates(
            base_params,
            model_config,
            packed,
            lora_config,
            queries,
            invalid_coefficients,
        )
    with pytest.raises(ValueError, match="positive integer"):
        score_hard_node_candidates(
            base_params,
            model_config,
            packed,
            lora_config,
            queries,
            evaluation_microbatch_size=0,
        )

    batch = queries[0].candidates[0].competence_batch
    with pytest.raises(ValueError, match="candidate logits"):
        active_token_candidate_nll(np.zeros(batch.input_ids.shape), batch)

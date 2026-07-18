from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import jax
import numpy as np
import pytest

import apm.continual.knowledge_evaluation as evaluation_module
from apm.continual.knowledge_evaluation import (
    KNOWLEDGE_AGGREGATION_AXES,
    KnowledgeAddressDecision,
    evaluate_ebt_knowledge_methods,
    evaluate_knowledge_method,
)
from apm.continual.knowledge_tasks import KnowledgeCandidate, KnowledgeQuery
from apm.continual.language_routing import route_language_prefix
from apm.continual.language_tasks import (
    AddressBook,
    NodeId,
    TaskId,
    build_prefix_suffix_batches,
)
from apm.lm.config import GptNeoConfig
from apm.lm.lora import LoraConfig, init_lora_edge
from apm.lm.lora_memory import pack_lora_memory
from apm.lm.parameters import init_gpt_neo_params
from apm.memory.address_refinement import EbtConfig
from apm.memory.graph import add_memory_node, init_memory_graph


def _model_config() -> GptNeoConfig:
    return GptNeoConfig(
        vocab_size=24,
        max_position_embeddings=8,
        hidden_size=8,
        intermediate_size=16,
        num_layers=1,
        num_heads=2,
        attention_types=("global",),
        local_window_size=4,
    )


def _query(
    query_id: str,
    *,
    prefix: tuple[int, ...],
    correct_candidate_index: int,
    reasoning_type: str,
    oracle_node_ids: tuple[NodeId, ...],
    required_edge_ids: tuple[NodeId, ...],
) -> KnowledgeQuery:
    candidates = tuple(
        KnowledgeCandidate(
            answer_text,
            build_prefix_suffix_batches(
                prefix + (10 + candidate_index, 14 + candidate_index),
                prefix_length=len(prefix),
                suffix_length=2,
            )[1],
        )
        for candidate_index, answer_text in enumerate(
            ("amber", "blue", "coral", "dune")
        )
    )
    router_batch = build_prefix_suffix_batches(
        prefix + (10, 14),
        prefix_length=len(prefix),
        suffix_length=2,
    )[0]
    return KnowledgeQuery(
        query_id=query_id,
        task_id=TaskId("willow-extension"),
        family_id="willow",
        query_kind=("cross-branch" if reasoning_type == "cross_branch" else "direct"),
        candidates=candidates,
        router_batch=router_batch,
        correct_candidate_index=correct_candidate_index,
        proof_id=f"proof-{query_id}",
        support_ids=(f"fact-{query_id}",),
        required_edge_ids=required_edge_ids,
        cue_regime=(
            "cue_free_control"
            if reasoning_type == "cross_branch"
            else "cue_sufficient"
        ),
        visible_cue_ids=(
            () if reasoning_type == "cross_branch" else ("cue-willow",)
        ),
        eligible_task_ids=(TaskId("willow-extension"),),
        novelty_regime=(
            "new-instance" if reasoning_type == "cross_branch" else "direct"
        ),
        reasoning_type=reasoning_type,
        reasoning_depth=(2 if reasoning_type == "cross_branch" else 0),
        prefix_length=len(prefix),
        mode="closed_book",
        oracle_node_ids=oracle_node_ids,
    )


def _fixture():
    model_config = _model_config()
    lora_config = LoraConfig(rank=1, alpha=1.0)
    edges = tuple(
        init_lora_edge(jax.random.PRNGKey(seed), model_config, lora_config)
        for seed in (1, 2)
    )
    graph = add_memory_node(
        init_memory_graph(NodeId("root")),
        NodeId("willow-extension"),
        NodeId("root"),
        TaskId("willow-extension"),
        1,
        edges[0],
    )
    graph = add_memory_node(
        graph,
        NodeId("willow-revision"),
        NodeId("root"),
        TaskId("willow-revision"),
        2,
        edges[1],
    )
    packed = pack_lora_memory(
        graph,
        model_config,
        lora_config,
        max_nodes=4,
        max_edges=3,
    )
    queries = (
        _query(
            "direct",
            prefix=(1, 2, 3),
            correct_candidate_index=1,
            reasoning_type="direct",
            oracle_node_ids=(NodeId("willow-extension"),),
            required_edge_ids=(NodeId("willow-extension"),),
        ),
        _query(
            "bridge",
            prefix=(4, 5, 6, 7),
            correct_candidate_index=2,
            reasoning_type="cross_branch",
            oracle_node_ids=(),
            required_edge_ids=(
                NodeId("willow-extension"),
                NodeId("willow-revision"),
            ),
        ),
    )
    hard_scores = np.full((2, 4, 4), np.inf, dtype=np.float32)
    hard_scores[0, :, 0] = (4.0, 2.5, 3.0, 5.0)
    hard_scores[0, :, 1] = (3.0, 1.0, 2.0, 4.0)
    hard_scores[0, :, 2] = (4.0, 1.5, 3.0, 5.0)
    hard_scores[1, :, 0] = (3.0, 4.0, 2.0, 5.0)
    hard_scores[1, :, 1] = (2.5, 3.0, 1.2, 4.0)
    hard_scores[1, :, 2] = (2.0, 3.0, 1.4, 4.0)
    decision = _decision()
    return model_config, lora_config, graph, packed, queries, hard_scores, decision


def _decision() -> KnowledgeAddressDecision:
    probabilities = np.asarray(
        (
            (0.1, 0.7, 0.2, 0.0),
            (0.1, 0.25, 0.65, 0.0),
        ),
        dtype=np.float32,
    )
    entropy = -np.sum(
        probabilities
        * np.log(np.where(probabilities > 0.0, probabilities, 1.0)),
        axis=1,
    )
    return KnowledgeAddressDecision(
        selected_indices=np.asarray((1, 2), dtype=np.int32),
        node_probabilities=probabilities,
        node_scores=np.asarray(
            ((0.0, 2.0, 1.0, -np.inf), (0.0, 1.0, 2.0, -np.inf)),
            dtype=np.float32,
        ),
        score_margin=np.ones((2,), dtype=np.float32),
        entropy=entropy,
        top_k_indices=np.asarray(((1, 2, 0), (2, 1, 0)), dtype=np.int32),
    )


def _soft_scores() -> np.ndarray:
    return np.asarray(
        (
            (2.0, 0.8, 1.5, 3.0),
            (2.2, 2.5, 0.9, 3.0),
        ),
        dtype=np.float32,
    )


def test_pure_hard_evaluation_derives_candidate_routing_and_support_metrics() -> None:
    _, _, graph, packed, queries, hard_scores, decision = _fixture()

    result = evaluate_knowledge_method(
        queries,
        hard_scores,
        graph,
        packed,
        stage=2,
        method="vamp_ebt_uniform",
        hard_decision=decision,
    )

    direct, bridge = result.queries
    assert direct.candidate_correct
    assert direct.candidate_margin == pytest.approx(1.0)
    assert direct.correct_answer_nll == pytest.approx(1.0)
    assert direct.routed_regret == pytest.approx(0.0)
    assert direct.task_oracle_regret == pytest.approx(0.0)
    assert direct.best_hard_node_regret == pytest.approx(0.0)
    assert direct.node_accuracy is True and direct.top_k_accuracy is True
    assert direct.hard_required_edge_recall == pytest.approx(1.0)
    assert direct.soft_required_edge_mean_coefficient is None

    assert bridge.candidate_correct
    assert bridge.candidate_margin == pytest.approx(0.6)
    assert bridge.task_oracle_node_index is None
    assert bridge.task_oracle_regret is None
    assert bridge.node_accuracy is None and bridge.top_k_accuracy is None
    assert bridge.best_hard_node_index == 1
    assert bridge.best_hard_node_regret == pytest.approx(0.2)
    assert bridge.hard_required_edge_recall == pytest.approx(0.5)
    assert not direct.candidate_nll.flags.writeable
    assert not result.address_decision.node_scores.flags.writeable

    axes = {aggregate.grouping_axis for aggregate in result.aggregates}
    assert axes == set(KNOWLEDGE_AGGREGATION_AXES)
    all_rows = [
        aggregate
        for aggregate in result.aggregates
        if aggregate.grouping_axis == "all"
    ]
    assert len(all_rows) == 1
    assert all_rows[0].candidate_accuracy == pytest.approx(1.0)
    assert all_rows[0].mean_hard_required_edge_recall == pytest.approx(0.75)


def test_soft_evaluation_allows_negative_regret_and_mean_coefficients() -> None:
    _, _, graph, packed, queries, hard_scores, decision = _fixture()
    soft_scores = _soft_scores()
    coefficients = np.asarray(
        ((0.9, 0.1, 0.0), (0.8, 0.6, 0.0)),
        dtype=np.float32,
    )

    result = evaluate_knowledge_method(
        queries,
        hard_scores,
        graph,
        packed,
        stage=2,
        method="vamp_ebt_uniform_soft",
        candidate_nll=soft_scores,
        hard_decision=decision,
        edge_coefficients=coefficients,
    )
    soft_scores[0, 1] = 9.0
    coefficients[0, 0] = 0.0

    direct, bridge = result.queries
    assert direct.correct_answer_nll == pytest.approx(0.8)
    assert direct.routed_regret == pytest.approx(-0.2)
    assert direct.task_oracle_regret == pytest.approx(-0.2)
    assert direct.best_hard_node_regret == pytest.approx(-0.2)
    assert direct.soft_required_edge_mean_coefficient == pytest.approx(0.9)
    assert bridge.routed_regret == pytest.approx(-0.5)
    assert bridge.best_hard_node_regret == pytest.approx(-0.3)
    assert bridge.soft_required_edge_mean_coefficient == pytest.approx(0.7)
    assert not result.edge_coefficients.flags.writeable
    aggregate = next(
        row for row in result.aggregates if row.grouping_axis == "all"
    )
    assert aggregate.mean_best_hard_node_regret == pytest.approx(-0.25)
    assert aggregate.mean_soft_required_edge_coefficient == pytest.approx(0.8)


def test_future_topology_is_explicitly_unavailable_without_fabricated_oracles() -> None:
    _, _, graph, packed, queries, hard_scores, decision = _fixture()
    future_id = NodeId("sunny-future")
    query = replace(
        queries[0],
        oracle_node_ids=(future_id,),
        required_edge_ids=(future_id,),
    )
    hard = evaluate_knowledge_method(
        (query,),
        hard_scores[:1],
        graph,
        packed,
        stage=1,
        method="future-hard",
        hard_decision=KnowledgeAddressDecision(
            selected_indices=decision.selected_indices[:1],
            node_probabilities=decision.node_probabilities[:1],
            node_scores=decision.node_scores[:1],
            score_margin=decision.score_margin[:1],
            entropy=decision.entropy[:1],
            top_k_indices=decision.top_k_indices[:1],
        ),
        unavailable_node_ids=(str(future_id),),
        unavailable_edge_ids=(str(future_id),),
    ).queries[0]

    assert hard.task_oracle_node_index is None
    assert hard.task_oracle_correct_answer_nll is None
    assert hard.task_oracle_regret is None
    assert hard.node_accuracy is None
    assert hard.top_k_accuracy is None
    assert hard.hard_required_edge_recall == pytest.approx(0.0)

    soft = evaluate_knowledge_method(
        (query,),
        hard_scores[:1],
        graph,
        packed,
        stage=1,
        method="future-soft",
        candidate_nll=_soft_scores()[:1],
        edge_coefficients=np.asarray(((0.8, 0.1, 0.0),), dtype=np.float32),
        unavailable_node_ids=(str(future_id),),
        unavailable_edge_ids=(str(future_id),),
    ).queries[0]
    assert soft.task_oracle_regret is None
    assert soft.soft_required_edge_mean_coefficient == pytest.approx(0.0)


def test_old_and_revision_nodes_select_their_incompatible_contextual_answers() -> None:
    _, _, graph, packed, _, _, _ = _fixture()
    base_query = _query(
        "base-context",
        prefix=(1, 2, 3),
        correct_candidate_index=0,
        reasoning_type="direct",
        oracle_node_ids=(NodeId("willow-extension"),),
        required_edge_ids=(NodeId("willow-extension"),),
    )
    revision_query = _query(
        "revised-context",
        prefix=(4, 5, 6),
        correct_candidate_index=1,
        reasoning_type="direct",
        oracle_node_ids=(NodeId("willow-revision"),),
        required_edge_ids=(NodeId("willow-revision"),),
    )
    hard_scores = np.full((2, 4, 4), np.inf, dtype=np.float32)
    hard_scores[:, :, 0] = (2.0, 3.0, 4.0, 5.0)
    hard_scores[:, :, 1] = (0.4, 2.0, 3.0, 4.0)
    hard_scores[:, :, 2] = (2.0, 0.5, 3.0, 4.0)
    probabilities = np.asarray(
        ((0.05, 0.9, 0.05, 0.0), (0.05, 0.05, 0.9, 0.0)),
        dtype=np.float32,
    )
    decision = KnowledgeAddressDecision(
        selected_indices=np.asarray((1, 2), dtype=np.int32),
        node_probabilities=probabilities,
        node_scores=np.asarray(
            ((0.0, 2.0, 1.0, -np.inf), (0.0, 1.0, 2.0, -np.inf)),
            dtype=np.float32,
        ),
        score_margin=np.ones((2,), dtype=np.float32),
        entropy=-np.sum(
            probabilities
            * np.log(np.where(probabilities > 0.0, probabilities, 1.0)),
            axis=1,
        ),
        top_k_indices=np.asarray(((1, 2, 0), (2, 1, 0)), dtype=np.int32),
    )

    result = evaluate_knowledge_method(
        (base_query, revision_query),
        hard_scores,
        graph,
        packed,
        stage=2,
        method="contextual-hard",
        hard_decision=decision,
    )

    assert result.queries[0].candidate_answer_texts[:2] == ("amber", "blue")
    assert result.queries[1].candidate_answer_texts[:2] == ("amber", "blue")
    assert tuple(row.predicted_candidate_index for row in result.queries) == (0, 1)
    assert all(row.candidate_correct for row in result.queries)


def test_evaluator_rejects_graph_score_decision_and_support_misalignment() -> None:
    _, _, graph, packed, queries, hard_scores, decision = _fixture()

    with pytest.raises(ValueError, match="unknown required edge ID"):
        evaluate_knowledge_method(
            (replace(queries[0], required_edge_ids=(NodeId("missing"),)),),
            hard_scores[:1],
            graph,
            packed,
            stage=2,
            method="hard",
            hard_decision=KnowledgeAddressDecision(
                selected_indices=decision.selected_indices[:1],
                node_probabilities=decision.node_probabilities[:1],
                node_scores=decision.node_scores[:1],
                score_margin=decision.score_margin[:1],
                entropy=decision.entropy[:1],
                top_k_indices=decision.top_k_indices[:1],
            ),
        )
    invalid_hard = hard_scores.copy()
    invalid_hard[0, 0, 3] = 1.0
    with pytest.raises(ValueError, match="positive infinity"):
        evaluate_knowledge_method(
            queries,
            invalid_hard,
            graph,
            packed,
            stage=2,
            method="hard",
            hard_decision=decision,
        )
    invalid_coefficients = np.asarray(
        ((0.9, 0.1, 0.2), (0.8, 0.6, 0.0)),
        dtype=np.float32,
    )
    with pytest.raises(ValueError, match="invalid edge coefficients"):
        evaluate_knowledge_method(
            queries,
            hard_scores,
            graph,
            packed,
            stage=2,
            method="soft",
            candidate_nll=_soft_scores(),
            edge_coefficients=invalid_coefficients,
        )


def test_ebt_execution_runs_each_initialization_once_and_reuses_its_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_config, lora_config, graph, packed, queries, hard_scores, _ = _fixture()
    address_book = AddressBook(
        node_ids=(
            NodeId("root"),
            NodeId("willow-extension"),
            NodeId("willow-revision"),
            None,
        ),
        keys=np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0))),
        valid_node_mask=np.asarray((True, True, True, False)),
    )
    trace_calls: list[str] = []
    score_calls: list[np.ndarray] = []

    def fake_trace(router, *args, **kwargs):
        del args, kwargs
        trace_calls.append(router)
        decision = _decision()
        coefficients = np.asarray(
            ((0.9, 0.1, 0.0), (0.8, 0.6, 0.0)),
            dtype=np.float32,
        )
        return SimpleNamespace(
            final_node_logits=decision.node_scores,
            node_probabilities=decision.node_probabilities,
            selected_indices=decision.selected_indices,
            edge_coefficients=coefficients,
        )

    def fake_soft_score(*args, **kwargs):
        del kwargs
        score_calls.append(np.array(args[5], copy=True))
        return _soft_scores()

    monkeypatch.setattr(evaluation_module, "_run_ebt_trace", fake_trace)
    monkeypatch.setattr(
        evaluation_module,
        "score_edge_coefficient_candidates",
        fake_soft_score,
    )

    evaluations = evaluate_ebt_knowledge_methods(
        object(),  # type: ignore[arg-type]
        model_config,
        graph,
        packed,
        lora_config,
        address_book,
        queries,
        hard_scores,
        stage=2,
        hopfield_config=object(),  # type: ignore[arg-type]
        ebt_config=object(),  # type: ignore[arg-type]
        evaluation_microbatch_size=3,
    )

    assert trace_calls == ["vamp_ebt_uniform", "vamp_ebt_hopfield"]
    assert len(score_calls) == 2
    assert tuple(result.method for result in evaluations) == (
        "vamp_ebt_uniform",
        "vamp_ebt_uniform_soft",
        "vamp_ebt_hopfield",
        "vamp_ebt_hopfield_soft",
    )
    method_pairs = (
        (evaluations[0], evaluations[1]),
        (evaluations[2], evaluations[3]),
    )
    for hard, soft in method_pairs:
        assert hard.address_decision is soft.address_decision
        np.testing.assert_array_equal(soft.edge_coefficients, score_calls.pop(0))
        assert hard.queries[0].correct_answer_nll == pytest.approx(1.0)
        assert soft.queries[0].correct_answer_nll == pytest.approx(0.8)


def test_candidate_suffix_mutation_cannot_change_hard_or_ebt_routing() -> None:
    model_config, lora_config, graph, packed, queries, hard_scores, _ = _fixture()
    query = queries[0]
    prefix = tuple(
        int(value)
        for value in query.candidates[0].competence_batch.input_ids[
            0, : query.prefix_length
        ]
    )
    mutated_query = replace(
        query,
        candidates=tuple(
            KnowledgeCandidate(
                candidate.answer_text,
                build_prefix_suffix_batches(
                    prefix + (18 + index, 20 + index),
                    prefix_length=query.prefix_length,
                    suffix_length=2,
                )[1],
            )
            for index, candidate in enumerate(query.candidates)
        ),
    )
    assert mutated_query.router_batch is query.router_batch
    for original, mutated in zip(query.candidates, mutated_query.candidates):
        np.testing.assert_array_equal(
            original.competence_batch.input_ids[:, : query.prefix_length],
            mutated.competence_batch.input_ids[:, : query.prefix_length],
        )
        assert not np.array_equal(
            original.competence_batch.target_ids[:, query.prefix_length - 1 :],
            mutated.competence_batch.target_ids[:, query.prefix_length - 1 :],
        )

    base_params = init_gpt_neo_params(jax.random.PRNGKey(21), model_config)
    keys = np.zeros((4, model_config.hidden_size), dtype=np.float32)
    keys[0, 0] = 1.0
    keys[1, 1] = 1.0
    keys[2, 2] = 1.0
    address_book = AddressBook(
        node_ids=(
            NodeId("root"),
            NodeId("willow-extension"),
            NodeId("willow-revision"),
            None,
        ),
        keys=keys,
        valid_node_mask=np.asarray((True, True, True, False)),
    )

    hard_before = route_language_prefix(
        "vamp_exhaustive",
        base_params,
        model_config,
        packed,
        lora_config,
        address_book,
        query.router_batch,
    )
    hard_after = route_language_prefix(
        "vamp_exhaustive",
        base_params,
        model_config,
        packed,
        lora_config,
        address_book,
        mutated_query.router_batch,
    )
    for field_name in hard_before._fields:
        np.testing.assert_array_equal(
            np.asarray(getattr(hard_before, field_name)),
            np.asarray(getattr(hard_after, field_name)),
        )

    ebt_arguments = {
        "base_params": base_params,
        "model_config": model_config,
        "graph": graph,
        "packed_memory": packed,
        "lora_config": lora_config,
        "address_book": address_book,
        "hard_candidate_nll": hard_scores[:1],
        "stage": 2,
        "ebt_config": EbtConfig(steps=1),
    }
    ebt_before = evaluate_ebt_knowledge_methods(
        queries=(query,),
        **ebt_arguments,
    )
    ebt_after = evaluate_ebt_knowledge_methods(
        queries=(mutated_query,),
        **ebt_arguments,
    )
    assert tuple(result.method for result in ebt_before) == tuple(
        result.method for result in ebt_after
    )
    for before, after in zip(ebt_before, ebt_after):
        assert before.address_decision is not None
        assert after.address_decision is not None
        for field_name in (
            "selected_indices",
            "node_probabilities",
            "node_scores",
            "score_margin",
            "entropy",
            "top_k_indices",
        ):
            np.testing.assert_array_equal(
                getattr(before.address_decision, field_name),
                getattr(after.address_decision, field_name),
            )
        if before.edge_coefficients is not None:
            np.testing.assert_array_equal(
                before.edge_coefficients,
                after.edge_coefficients,
            )

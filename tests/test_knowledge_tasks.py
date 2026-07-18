from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from apm.continual.knowledge_tasks import KnowledgeCandidate, KnowledgeQuery
from apm.continual.language_tasks import NodeId, TaskId, build_prefix_suffix_batches


def _candidate(answer_text: str, tokens: tuple[int, ...]) -> KnowledgeCandidate:
    return KnowledgeCandidate(
        answer_text,
        build_prefix_suffix_batches(tokens, prefix_length=3, suffix_length=2)[1],
    )


def _query() -> KnowledgeQuery:
    router_batch = build_prefix_suffix_batches(
        (1, 2, 3, 4, 5),
        prefix_length=3,
        suffix_length=2,
    )[0]
    return KnowledgeQuery(
        query_id="query-0",
        task_id=TaskId("willow-seed"),
        family_id="willow",
        query_kind="one-hop",
        candidates=tuple(
            _candidate(answer, (1, 2, 3, token, token + 1))
            for answer, token in zip(("amber", "blue", "coral", "dune"), range(4, 8))
        ),
        router_batch=router_batch,
        correct_candidate_index=1,
        proof_id="proof-0",
        support_ids=("fact-0", "rule-0"),
        required_edge_ids=(NodeId("willow-seed"),),
        cue_regime="cue_sufficient",
        visible_cue_ids=("cue-willow",),
        eligible_task_ids=(TaskId("willow-seed"),),
        novelty_regime="new-instance",
        reasoning_type="one_hop",
        reasoning_depth=1,
        prefix_length=3,
        mode="closed_book",
        oracle_node_ids=(NodeId("willow-seed"),),
    )


def test_knowledge_query_is_immutable_and_retains_canonical_metadata() -> None:
    query = _query()

    assert len(query.candidates) == 4
    assert query.candidates[query.correct_candidate_index].answer_text == "blue"
    assert query.proof_id == "proof-0"
    assert query.support_ids == ("fact-0", "rule-0")
    assert query.oracle_node_ids == (NodeId("willow-seed"),)
    with pytest.raises(FrozenInstanceError):
        query.prefix_length = 64  # type: ignore[misc]


def test_cross_branch_query_has_empty_oracle_and_explicit_required_edges() -> None:
    cross_branch = replace(
        _query(),
        query_id="query-cross",
        query_kind="cross-branch",
        reasoning_type="cross_branch",
        reasoning_depth=2,
        required_edge_ids=(NodeId("willow-extension"), NodeId("willow-revision")),
        oracle_node_ids=(),
    )

    assert cross_branch.oracle_node_ids == ()
    assert len(cross_branch.required_edge_ids) == 2
    with pytest.raises(ValueError, match="explicit edge support"):
        replace(cross_branch, required_edge_ids=())
    with pytest.raises(ValueError, match="cannot claim"):
        replace(cross_branch, oracle_node_ids=(NodeId("willow-bridge"),))
    with pytest.raises(ValueError, match="only cross-branch"):
        replace(_query(), oracle_node_ids=())


def test_query_rejects_candidate_count_duplicates_and_unpaired_prefixes() -> None:
    query = _query()

    with pytest.raises(ValueError, match="exactly four"):
        replace(query, candidates=query.candidates[:3])
    with pytest.raises(ValueError, match="answer texts must be unique"):
        replace(
            query,
            candidates=query.candidates[:3]
            + (replace(query.candidates[3], answer_text="amber"),),
        )
    with pytest.raises(ValueError, match="exact visible prefix"):
        replace(
            query,
            candidates=query.candidates[:3]
            + (_candidate("dune", (1, 9, 3, 7, 8)),),
        )
    with pytest.raises(ValueError, match="equal active token counts"):
        replace(
            query,
            candidates=query.candidates[:3]
            + (_candidate("dune", (1, 2, 3, 7)),),
        )


def test_query_rejects_invalid_indices_ids_and_metadata() -> None:
    query = _query()

    with pytest.raises(ValueError, match="correct_candidate_index"):
        replace(query, correct_candidate_index=4)
    with pytest.raises(ValueError, match="canonical"):
        replace(query, proof_id=" proof-0")
    with pytest.raises(ValueError, match="unique identifiers"):
        replace(query, support_ids=("fact-0", "fact-0"))
    with pytest.raises(ValueError, match="unknown cue regime"):
        replace(query, cue_regime="unknown")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown knowledge mode"):
        replace(query, mode="unknown")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="zero through two"):
        replace(query, reasoning_depth=3)
    with pytest.raises(ValueError, match="loss boundary"):
        replace(query, prefix_length=2)


def test_query_router_is_exactly_the_shared_suffix_free_prefix() -> None:
    query = _query()

    assert query.router_batch.input_ids.shape == (1, query.prefix_length - 1)
    assert query.router_batch.loss_mask.all()
    suffix_mutated = replace(
        query,
        candidates=query.candidates[:3]
        + (_candidate("dune", (1, 2, 3, 10, 11)),),
    )
    assert suffix_mutated.router_batch is query.router_batch
    mismatched_router = build_prefix_suffix_batches(
        (1, 9, 3, 4, 5),
        prefix_length=3,
        suffix_length=2,
    )[0]
    with pytest.raises(ValueError, match="exactly match every candidate prefix"):
        replace(query, router_batch=mismatched_router)
    with pytest.raises(TypeError, match="must be a RouterBatch"):
        replace(
            query,
            router_batch=query.candidates[0].competence_batch,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="at least two tokens"):
        replace(query, prefix_length=1)

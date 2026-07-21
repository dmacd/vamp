from __future__ import annotations

from hashlib import sha256
from typing import Sequence

import numpy as np

from apm.data.text.tinyworlds_v2.generation_schema import RouteLock
from apm.data.text.tinyworlds_v2.generation_costs import request_cost_upper_bound
from apm.data.text.tinyworlds_v2.reasoning_sidebar import (
    REASONING_SIDEBAR_BATCH_SIZE,
    REASONING_SIDEBAR_GENERATION_VARIANTS,
    REASONING_SIDEBAR_STORIES_PER_EVIDENCE,
    SidebarReferenceRecord,
    build_sidebar_author_requests,
    build_sidebar_clause_completion_queries,
    build_sidebar_evidence_plans,
    build_sidebar_queries,
    build_sidebar_reference_control,
    build_sidebar_story_plans,
    build_sidebar_training_batches,
    reasoning_sidebar_world,
    select_sidebar_training_stories,
    sidebar_query_score_records,
    sidebar_story_prompt,
    summarize_sidebar_scores,
    validate_sidebar_story,
)
from apm.data.text.tinyworlds_v2.route_lock import validate_locked_request_body


class _WordTokenizer:
    vocab_size = 2**31
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text: str, *, add_eos: bool = False) -> tuple[int, ...]:
        values = tuple(
            2 + int.from_bytes(sha256(word.encode("utf-8")).digest()[:4], "big") % 1_000_000
            for word in text.split()
        )
        return values + ((self.eos_token_id,) if add_eos else ())

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        del skip_special_tokens
        return " ".join(str(value) for value in token_ids)


def _routes() -> tuple[RouteLock, RouteLock]:
    return tuple(
        RouteLock(
            route_id=route_id,
            catalog_sha256="0" * 64,
            requested_model=requested,
            canonical_model=canonical,
            provider_slug=provider,
            returned_provider=returned,
            quantization="fp16",
            input_usd_per_million=input_price,
            output_usd_per_million=output_price,
        )
        for route_id, requested, canonical, provider, returned, input_price, output_price in (
            (
                "qwen3.5-35b-a3b",
                "qwen/qwen3.5-35b-a3b",
                "qwen/qwen3.5-35b-a3b-20260224",
                "alibaba",
                "Alibaba",
                "0.14",
                "1.00",
            ),
            (
                "gpt-5.4-mini",
                "openai/gpt-5.4-mini",
                "openai/gpt-5.4-mini-20260317",
                "openai",
                "OpenAI",
                "0.75",
                "4.50",
            ),
        )
    )


def test_sidebar_world_and_story_matrix_are_balanced() -> None:
    facts, rules = reasoning_sidebar_world()
    plans = build_sidebar_evidence_plans()
    story_plans = build_sidebar_story_plans()

    assert len(facts) == 8
    assert len(rules) == 4
    assert len(plans) == 12
    assert len(story_plans) == len(plans) * REASONING_SIDEBAR_GENERATION_VARIANTS
    assert all(sum(fact.color == rule.color for fact in facts) == 2 for rule in rules)
    assert all(
        fact.child not in rule_plan.exact_sentence
        for fact in facts
        for rule_plan in plans
        if rule_plan.kind == "rule"
    )


def test_sidebar_prompt_and_validation_preserve_exact_evidence() -> None:
    tokenizer = _WordTokenizer()
    plan = build_sidebar_story_plans()[0]
    story = f"{plan.evidence.exact_sentence}\nOnce upon a time, a cat played all day."

    prompt = sidebar_story_prompt(plan)
    accepted = validate_sidebar_story(plan, story, tokenizer)
    ordinary_color = validate_sidebar_story(plan, f"{story} The grass was green.", tokenizer)
    rejected = validate_sidebar_story(plan, f"A preface. {story} pond", tokenizer)

    assert plan.evidence.exact_sentence in prompt
    assert accepted.accepted
    assert ordinary_color.accepted
    assert not rejected.accepted
    assert set(rejected.rejection_reasons) == {
        "exact_evidence_not_leading",
        "conflicting_world_word",
    }


def test_sidebar_selects_first_two_accepted_variants_per_evidence() -> None:
    tokenizer = _WordTokenizer()
    candidates = tuple(
        (
            plan,
            story,
            validate_sidebar_story(plan, story, tokenizer),
        )
        for plan in build_sidebar_story_plans()
        for story in (f"{plan.evidence.exact_sentence}\nA little cat had a good day.",)
    )

    selected = select_sidebar_training_stories(candidates)

    assert len(selected) == len(build_sidebar_evidence_plans()) * 2
    assert all(
        [item[0].variant_index for item in selected if item[0].evidence == evidence]
        == list(range(REASONING_SIDEBAR_STORIES_PER_EVIDENCE))
        for evidence in build_sidebar_evidence_plans()
    )


def test_reference_control_is_deterministic_disjoint_and_trainable() -> None:
    tokenizer = _WordTokenizer()
    records = tuple(
        SidebarReferenceRecord(
            f"reference-{index}",
            f"Once upon a time, a small cat number{index} played with a toy.",
        )
        for index in range(80)
    )

    first = build_sidebar_reference_control(records, tokenizer)
    second = build_sidebar_reference_control(tuple(reversed(records)), tokenizer)
    batches = build_sidebar_training_batches(
        tuple(item[1] for item in first),
        tokenizer,
    )

    assert first == second
    assert len(first) == 24
    assert len({item[3] for item in first}) == 24
    assert len(batches) == 1
    assert batches[0].input_ids.shape == (REASONING_SIDEBAR_BATCH_SIZE, 256)


def test_sidebar_queries_have_exact_balanced_candidates_and_score_summary() -> None:
    tokenizer = _WordTokenizer()
    queries = build_sidebar_queries(tokenizer, "test")
    correct = np.asarray([query.correct_candidate_index for query in queries])
    perfect_scores = np.full((len(queries), 4), 5.0, dtype=np.float32)
    perfect_scores[np.arange(len(queries)), correct] = 1.0

    direct = tuple(query for query in queries if query.reasoning_type == "direct")
    one_hop = tuple(query for query in queries if query.reasoning_type == "one_hop")
    summary = summarize_sidebar_scores(queries, perfect_scores)

    assert len(direct) == len(one_hop) == 16
    assert all(query.reasoning_depth == 0 for query in direct)
    assert all(query.reasoning_depth == 1 for query in one_hop)
    assert all(len(query.support_ids) == 2 for query in one_hop)
    assert tuple(np.bincount(correct, minlength=4)) == (8, 8, 8, 8)
    assert summary.accuracy == summary.paired_consistency == 1.0
    assert summary.correct_nll == 1.0
    assert summary.margin == 4.0


def test_sidebar_clause_probes_cover_every_seen_fact_and_rule_once() -> None:
    queries = build_sidebar_clause_completion_queries(_WordTokenizer())
    scores = np.ones((len(queries), 4), dtype=np.float32)
    score_records = sidebar_query_score_records(queries, scores)

    assert len(queries) == 12
    assert len(score_records) == len(queries)
    assert sum(query.reasoning_type == "fact_clause_completion" for query in queries) == 8
    assert sum(query.reasoning_type == "rule_clause_completion" for query in queries) == 4
    assert all(query.reasoning_depth == 0 for query in queries)


def test_sidebar_requests_use_exact_locked_routes_and_plain_text() -> None:
    requests = build_sidebar_author_requests(_routes())

    assert len(requests) == 2 * 12 * REASONING_SIDEBAR_GENERATION_VARIANTS
    for route_id, plan, route, request in requests:
        validate_locked_request_body(route, request)
        assert float(request_cost_upper_bound(request, route).upper_bound_usd) > 0
        body = request.body
        assert route_id == route.route_id
        assert body["messages"] == [
            {"content": sidebar_story_prompt(plan), "role": "user"}
        ]
        assert "response_format" not in body
        assert body["reasoning"] == {"effort": "none"}

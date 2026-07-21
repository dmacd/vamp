"""Controlled TinyStories-LoRA learnability probe for externally authored prose."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
import math
import re
from typing import Literal, Sequence

import numpy as np

from apm.continual.knowledge_tasks import KnowledgeCandidate, KnowledgeQuery
from apm.continual.language_tasks import (
    RouterBatch,
    TaskId,
    build_prefix_suffix_batches,
)
from apm.data.text.tinyworlds_v2.bakeoff import (
    CandidateModelSpec,
    TWO_ROUTE_AUTHOR_MODELS,
)
from apm.data.text.tinyworlds_v2.generation_schema import (
    CanonicalRequest,
    RouteLock,
)
from apm.data.text.tinyworlds_v2.json_contracts import JsonObject
from apm.data.text.tinyworlds_v2.phase1_generation import (
    CHAT_COMPLETIONS_ENDPOINT,
)
from apm.lm.text import TextTokenizer
from apm.lm.text_data import TokenBatch, batch_token_windows, causal_token_windows
from apm.memory.graph import NodeId


REASONING_SIDEBAR_VERSION = "tinyworlds-v2-reasoning-sidebar-v1"
REASONING_SIDEBAR_TASK_ID = TaskId("willow-club")
REASONING_SIDEBAR_NODE_ID = NodeId("willow-club-adapter")
REASONING_SIDEBAR_STORIES_PER_EVIDENCE = 2
REASONING_SIDEBAR_GENERATION_VARIANTS = 3
REASONING_SIDEBAR_CONTEXT_LENGTH = 256
REASONING_SIDEBAR_BATCH_SIZE = 32
REASONING_SIDEBAR_UPDATE_BUDGET = 512
REASONING_SIDEBAR_LORA_RANK = 8
REASONING_SIDEBAR_HARD_CAP_USD = "0.50"
REASONING_SIDEBAR_MAX_OUTPUT_TOKENS = 384

SidebarEvidenceKind = Literal["fact", "rule"]
SidebarQuerySplit = Literal["validation", "test"]


@dataclass(frozen=True, slots=True)
class SidebarFact:
    """One arbitrary child-to-badge binding in the fixed probe world."""

    fact_id: str
    child: str
    color: str


@dataclass(frozen=True, slots=True)
class SidebarRule:
    """One badge-to-place rule used by the one-hop probes."""

    rule_id: str
    color: str
    place: str


@dataclass(frozen=True, slots=True)
class SidebarEvidencePlan:
    """One exact evidence clause that every author must preserve."""

    evidence_id: str
    kind: SidebarEvidenceKind
    exact_sentence: str
    fact_id: str | None
    rule_id: str | None
    forbidden_words: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.kind == "fact" and (self.fact_id is None or self.rule_id is not None):
            raise ValueError("fact evidence must identify only one fact")
        if self.kind == "rule" and (self.rule_id is None or self.fact_id is not None):
            raise ValueError("rule evidence must identify only one rule")


@dataclass(frozen=True, slots=True)
class SidebarStoryPlan:
    """One author/seed slot for a fixed evidence clause."""

    story_plan_id: str
    evidence: SidebarEvidencePlan
    variant_index: int

    def __post_init__(self) -> None:
        if not 0 <= self.variant_index < REASONING_SIDEBAR_GENERATION_VARIANTS:
            raise ValueError("story-plan variant index is outside the fixed matrix")


@dataclass(frozen=True, slots=True)
class SidebarStoryValidation:
    """Locally derived evidence and context-fit checks for one story."""

    accepted: bool
    rejection_reasons: tuple[str, ...]
    token_count: int
    word_count: int
    story_sha256: str


@dataclass(frozen=True, slots=True)
class SidebarReferenceRecord:
    """The minimal decontaminated reference record used by the control arm."""

    record_id: str
    story: str


@dataclass(frozen=True, slots=True)
class SidebarScoreSummary:
    """Exact-choice accuracy, NLL, and separation for one query collection."""

    query_count: int
    accuracy: float
    correct_nll: float
    margin: float
    paired_consistency: float


_FACTS = (
    SidebarFact("fact-mia-red", "Mia", "red"),
    SidebarFact("fact-ben-blue", "Ben", "blue"),
    SidebarFact("fact-zoe-green", "Zoe", "green"),
    SidebarFact("fact-leo-black", "Leo", "black"),
    SidebarFact("fact-ava-blue", "Ava", "blue"),
    SidebarFact("fact-sam-green", "Sam", "green"),
    SidebarFact("fact-ivy-black", "Ivy", "black"),
    SidebarFact("fact-max-red", "Max", "red"),
)
_RULES = (
    SidebarRule("rule-red-pond", "red", "pond"),
    SidebarRule("rule-blue-hill", "blue", "hill"),
    SidebarRule("rule-green-tent", "green", "tent"),
    SidebarRule("rule-black-cave", "black", "cave"),
)
_COLORS = tuple(rule.color for rule in _RULES)
_PLACES = tuple(rule.place for rule in _RULES)
_CHILDREN = tuple(fact.child for fact in _FACTS)
_WORLD_WORDS = tuple(dict.fromkeys((*_CHILDREN, *_COLORS, *_PLACES, "Willow", "badge")))
_WORD_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

_DIRECT_QUERY_PHRASES: tuple[tuple[SidebarQuerySplit, str], ...] = (
    ("validation", "In the Willow Club stories, {child}'s club badge was"),
    ("validation", "The badge color recorded for {child} at Willow Club was"),
    ("test", "At Willow Club, {child} wore a badge. Its color was"),
    ("test", "The Willow Club note says that {child}'s badge color was"),
)
_ONE_HOP_QUERY_PHRASES: tuple[tuple[SidebarQuerySplit, str], ...] = (
    ("validation", "In the Willow Club stories, {child}'s meeting place was the"),
    ("validation", "At Willow Club, {child} went to meet the group by the"),
    ("test", "Where did {child} meet the club? The place was the"),
    ("test", "The club rule sent {child} to meet everyone by the"),
)


def reasoning_sidebar_world() -> tuple[tuple[SidebarFact, ...], tuple[SidebarRule, ...]]:
    """Return the fixed balanced bindings and one-hop rules."""
    if {fact.color for fact in _FACTS} != set(_COLORS):
        raise RuntimeError("sidebar facts no longer cover every badge color")
    if any(sum(fact.color == color for fact in _FACTS) != 2 for color in _COLORS):
        raise RuntimeError("sidebar answer positions are no longer balanced")
    return _FACTS, _RULES


def build_sidebar_evidence_plans() -> tuple[SidebarEvidencePlan, ...]:
    """Build exact fact and rule clauses without any named derived conclusion."""
    facts, rules = reasoning_sidebar_world()
    fact_plans = tuple(
        SidebarEvidencePlan(
            evidence_id=fact.fact_id,
            kind="fact",
            exact_sentence=f"{fact.child}'s club badge was {fact.color}.",
            fact_id=fact.fact_id,
            rule_id=None,
            forbidden_words=(
                tuple(color for color in _COLORS if color != fact.color)
                + _PLACES
            ),
        )
        for fact in facts
    )
    rule_plans = tuple(
        SidebarEvidencePlan(
            evidence_id=rule.rule_id,
            kind="rule",
            exact_sentence=(
                f"At Willow Club, every {rule.color} badge meant meeting by "
                f"the {rule.place}."
            ),
            fact_id=None,
            rule_id=rule.rule_id,
            forbidden_words=_CHILDREN
            + tuple(color for color in _COLORS if color != rule.color)
            + tuple(place for place in _PLACES if place != rule.place),
        )
        for rule in rules
    )
    return fact_plans + rule_plans


def build_sidebar_story_plans() -> tuple[SidebarStoryPlan, ...]:
    """Expand every evidence clause into three paid candidate generations."""
    return tuple(
        SidebarStoryPlan(
            story_plan_id=f"{evidence.evidence_id}-variant-{variant_index}",
            evidence=evidence,
            variant_index=variant_index,
        )
        for evidence in build_sidebar_evidence_plans()
        for variant_index in range(REASONING_SIDEBAR_GENERATION_VARIANTS)
    )


def sidebar_story_prompt(plan: SidebarStoryPlan) -> str:
    """Render the minimal fact-bearing TinyStories-format author prompt."""
    forbidden = ", ".join(plan.evidence.forbidden_words)
    return (
        "Write a short story (3-5 paragraphs) which only uses very simple words "
        "that a 3 year old child would understand. Begin the story with this "
        f'exact sentence: "{plan.evidence.exact_sentence}" Do not use these '
        f"other world words: {forbidden}. Remember to only use simple words!\n\n"
        "Aim for about 130 to 150 words.\n\nPossible story:"
    )


def build_sidebar_author_requests(
    routes: Sequence[RouteLock],
) -> tuple[tuple[str, SidebarStoryPlan, RouteLock, CanonicalRequest], ...]:
    """Bind the complete paired prompt matrix to exact provider routes."""
    models_by_route = {model.route_id: model for model in TWO_ROUTE_AUTHOR_MODELS}
    if tuple(route.route_id for route in routes) != tuple(models_by_route):
        raise ValueError("sidebar requires the fixed Qwen then GPT route order")
    return tuple(
        (
            route.route_id,
            plan,
            route,
            CanonicalRequest.from_body(
                route_lock_sha256=route.lock_sha256,
                endpoint=CHAT_COMPLETIONS_ENDPOINT,
                body=_sidebar_request_body(plan, models_by_route[route.route_id], route),
            ),
        )
        for route in routes
        for plan in build_sidebar_story_plans()
    )


def validate_sidebar_story(
    plan: SidebarStoryPlan,
    story: str,
    tokenizer: TextTokenizer,
) -> SidebarStoryValidation:
    """Require the exact leading evidence, no conflicting world words, and fit."""
    if type(story) is not str:
        raise TypeError("sidebar story must be text")
    token_count = len(tokenizer.encode(story, add_eos=True)) if story else 0
    normalized_words = tuple(word.casefold() for word in _WORD_PATTERN.findall(story))
    # Colors are normal narrative vocabulary (green grass, blue sky) and do
    # not assert another badge binding.  Only candidate meeting-place words
    # can contaminate fact prose; rule prose additionally cannot name one of
    # the queried children.  The paid prompt remains deliberately stricter.
    semantic_conflict_words = (
        _PLACES
        if plan.evidence.kind == "fact"
        else (*_CHILDREN, *_PLACES)
    )
    forbidden = {
        word.casefold()
        for word in plan.evidence.forbidden_words
        if word in semantic_conflict_words
    }
    reasons = tuple(
        reason
        for condition, reason in (
            (not story.strip(), "empty_story"),
            (
                not story.lstrip().startswith(plan.evidence.exact_sentence),
                "exact_evidence_not_leading",
            ),
            (
                story.count(plan.evidence.exact_sentence) != 1,
                "exact_evidence_count",
            ),
            (bool(forbidden.intersection(normalized_words)), "conflicting_world_word"),
            (
                token_count > REASONING_SIDEBAR_CONTEXT_LENGTH + 1,
                "context_overflow",
            ),
        )
        if condition
    )
    return SidebarStoryValidation(
        accepted=not reasons,
        rejection_reasons=reasons,
        token_count=token_count,
        word_count=len(normalized_words),
        story_sha256=sha256(story.encode("utf-8")).hexdigest(),
    )


def select_sidebar_training_stories(
    candidates: Sequence[tuple[SidebarStoryPlan, str, SidebarStoryValidation]],
) -> tuple[tuple[SidebarStoryPlan, str, SidebarStoryValidation], ...]:
    """Take the first two accepted variants for every evidence item."""
    selected = tuple(
        item
        for evidence in build_sidebar_evidence_plans()
        for item in tuple(
            candidate
            for candidate in candidates
            if candidate[0].evidence.evidence_id == evidence.evidence_id
            and candidate[2].accepted
        )[:REASONING_SIDEBAR_STORIES_PER_EVIDENCE]
    )
    expected_count = (
        len(build_sidebar_evidence_plans()) * REASONING_SIDEBAR_STORIES_PER_EVIDENCE
    )
    if len(selected) != expected_count:
        raise ValueError("an author did not produce two valid stories per evidence item")
    return selected


def build_sidebar_reference_control(
    records: Sequence[SidebarReferenceRecord],
    tokenizer: TextTokenizer,
) -> tuple[tuple[SidebarStoryPlan, str, SidebarStoryValidation, str], ...]:
    """Prefix exact evidence to deterministic decontaminated TinyStories contexts."""
    if len({record.record_id for record in records}) != len(records):
        raise ValueError("reference control record IDs must be unique")
    clean_records = tuple(
        record
        for record in records
        if not _contains_any_word(record.story, _WORLD_WORDS)
    )
    used: set[str] = set()
    selected: list[tuple[SidebarStoryPlan, str, SidebarStoryValidation, str]] = []
    for evidence in build_sidebar_evidence_plans():
        for slot in range(REASONING_SIDEBAR_STORIES_PER_EVIDENCE):
            plan = SidebarStoryPlan(
                f"{evidence.evidence_id}-control-{slot}",
                evidence,
                slot,
            )
            choices = sorted(
                (record for record in clean_records if record.record_id not in used),
                key=lambda record: (
                    sha256(
                        f"{REASONING_SIDEBAR_VERSION}\0{plan.story_plan_id}\0"
                        f"{record.record_id}".encode("utf-8")
                    ).digest(),
                    record.record_id,
                ),
            )
            accepted = next(
                (
                    (record, story, validation)
                    for record in choices
                    for story in (f"{evidence.exact_sentence}\n{record.story}",)
                    for validation in (validate_sidebar_story(plan, story, tokenizer),)
                    if validation.accepted
                ),
                None,
            )
            if accepted is None:
                raise ValueError("decontaminated references cannot fill the control arm")
            record, story, validation = accepted
            used.add(record.record_id)
            selected.append((plan, story, validation, record.record_id))
    return tuple(selected)


def build_sidebar_queries(
    tokenizer: TextTokenizer,
    split: SidebarQuerySplit,
) -> tuple[KnowledgeQuery, ...]:
    """Build paired direct-recall and one-rule-inference exact-choice probes."""
    if split not in ("validation", "test"):
        raise ValueError("sidebar query split must be validation or test")
    rules_by_color = {rule.color: rule for rule in _RULES}
    return tuple(
        query
        for fact in _FACTS
        for reasoning_type, phrases, answers, correct_answer, support_ids in (
            (
                "direct",
                _DIRECT_QUERY_PHRASES,
                _COLORS,
                fact.color,
                (fact.fact_id,),
            ),
            (
                "one_hop",
                _ONE_HOP_QUERY_PHRASES,
                _PLACES,
                rules_by_color[fact.color].place,
                (fact.fact_id, rules_by_color[fact.color].rule_id),
            ),
        )
        for phrase_index, (phrase_split, phrase_template) in enumerate(phrases)
        if phrase_split == split
        for query in (
            _sidebar_query(
                tokenizer,
                fact,
                reasoning_type,
                phrase_index,
                phrase_template.format(child=fact.child),
                answers,
                correct_answer,
                support_ids,
                split,
            ),
        )
    )


def build_sidebar_clause_completion_queries(
    tokenizer: TextTokenizer,
) -> tuple[KnowledgeQuery, ...]:
    """Probe exact fact and rule prefixes that occurred in every training arm."""
    return tuple(
        query
        for query_id, reasoning_type, prefix_text, answers, correct, support_ids in (
            *(
                (
                    f"clause-fact-{fact.child.casefold()}",
                    "fact_clause_completion",
                    f"{fact.child}'s club badge was",
                    _COLORS,
                    fact.color,
                    (fact.fact_id,),
                )
                for fact in _FACTS
            ),
            *(
                (
                    f"clause-rule-{rule.color}",
                    "rule_clause_completion",
                    f"At Willow Club, every {rule.color} badge meant meeting by the",
                    _PLACES,
                    rule.place,
                    (rule.rule_id,),
                )
                for rule in _RULES
            ),
        )
        for query in (
            _clause_completion_query(
                tokenizer,
                query_id,
                reasoning_type,
                prefix_text,
                answers,
                correct,
                support_ids,
            ),
        )
    )


def build_sidebar_training_batches(
    stories: Sequence[str],
    tokenizer: TextTokenizer,
) -> tuple[TokenBatch, ...]:
    """Pack one causal window per story without crossing document boundaries."""
    if not stories:
        raise ValueError("sidebar training requires stories")
    windows = tuple(
        causal_token_windows(
            tokenizer.encode(story, add_eos=True),
            REASONING_SIDEBAR_CONTEXT_LENGTH,
            tokenizer.pad_token_id,
            stride=REASONING_SIDEBAR_CONTEXT_LENGTH,
        )
        for story in stories
    )
    if any(window.input_ids.shape[0] != 1 for window in windows):
        raise ValueError("every sidebar story must occupy exactly one causal window")
    combined = TokenBatch(
        input_ids=np.concatenate(tuple(window.input_ids for window in windows)),
        attention_mask=np.concatenate(tuple(window.attention_mask for window in windows)),
        target_ids=np.concatenate(tuple(window.target_ids for window in windows)),
        loss_mask=np.concatenate(tuple(window.loss_mask for window in windows)),
    )
    return batch_token_windows(
        combined,
        REASONING_SIDEBAR_BATCH_SIZE,
        tokenizer.pad_token_id,
    )


def summarize_sidebar_scores(
    queries: Sequence[KnowledgeQuery],
    scores: np.ndarray,
) -> SidebarScoreSummary:
    """Summarize exact-choice scores and both-paraphrase semantic consistency."""
    score_array = np.asarray(scores, dtype=np.float64)
    if score_array.shape != (len(queries), 4) or not np.all(np.isfinite(score_array)):
        raise ValueError("sidebar candidate scores must be finite [query, 4] values")
    correct = np.asarray(
        tuple(query.correct_candidate_index for query in queries), dtype=np.int32
    )
    rows = np.arange(len(queries))
    predictions = np.argmin(score_array, axis=1)
    wrong = score_array.copy()
    wrong[rows, correct] = np.inf
    groups: dict[tuple[str, str], list[bool]] = {}
    for query, is_correct in zip(queries, predictions == correct, strict=True):
        identity = tuple(query.support_ids[:1]) + (query.reasoning_type,)
        groups.setdefault((identity[0], identity[1]), []).append(bool(is_correct))
    if any(len(values) != 2 for values in groups.values()):
        raise ValueError("sidebar score consistency expects two paraphrases per item")
    return SidebarScoreSummary(
        query_count=len(queries),
        accuracy=float(np.mean(predictions == correct)),
        correct_nll=float(np.mean(score_array[rows, correct])),
        margin=float(np.mean(np.min(wrong, axis=1) - score_array[rows, correct])),
        paired_consistency=float(np.mean(tuple(all(values) for values in groups.values()))),
    )


def sidebar_query_score_records(
    queries: Sequence[KnowledgeQuery],
    scores: np.ndarray,
) -> tuple[JsonObject, ...]:
    """Return complete per-query candidate evidence for artifact persistence."""
    score_array = np.asarray(scores, dtype=np.float64)
    if score_array.shape != (len(queries), 4) or not np.all(np.isfinite(score_array)):
        raise ValueError("sidebar candidate scores must be finite [query, 4] values")
    return tuple(
        {
            "candidate_nll": [float(value) for value in score_array[index]],
            "candidate_texts": [candidate.answer_text for candidate in query.candidates],
            "correct_candidate_index": query.correct_candidate_index,
            "predicted_candidate_index": int(np.argmin(score_array[index])),
            "query_id": query.query_id,
            "reasoning_type": query.reasoning_type,
            "support_ids": list(query.support_ids),
        }
        for index, query in enumerate(queries)
    )


def _sidebar_request_body(
    plan: SidebarStoryPlan,
    model: CandidateModelSpec,
    route: RouteLock,
) -> JsonObject:
    return {
        model.max_token_parameter: REASONING_SIDEBAR_MAX_OUTPUT_TOKENS,
        "messages": [{"content": sidebar_story_prompt(plan), "role": "user"}],
        "model": model.request_model_id,
        "plugins": [],
        "provider": {
            "allow_fallbacks": False,
            "data_collection": "deny",
            "max_price": {
                "completion": _conservative_json_price(
                    route.output_usd_per_million
                ),
                "prompt": _conservative_json_price(
                    route.input_usd_per_million
                ),
            },
            "only": [route.provider_slug],
            "quantizations": [route.quantization],
            "require_parameters": True,
        },
        "reasoning": {"effort": "none"},
        "seed": int.from_bytes(
            sha256(
                f"{REASONING_SIDEBAR_VERSION}\0{route.route_id}\0"
                f"{plan.story_plan_id}".encode("utf-8")
            ).digest()[:4],
            "big",
        )
        & ((1 << 31) - 1),
        "stream": False,
        "transforms": [],
    }


def _sidebar_query(
    tokenizer: TextTokenizer,
    fact: SidebarFact,
    reasoning_type: str,
    phrase_index: int,
    prefix_text: str,
    answers: tuple[str, ...],
    correct_answer: str,
    support_ids: tuple[str, ...],
    split: SidebarQuerySplit,
) -> KnowledgeQuery:
    prefix_tokens, candidates, router_batch = _query_views(
        tokenizer,
        prefix_text,
        answers,
    )
    return KnowledgeQuery(
        query_id=(
            f"{split}-{reasoning_type}-{fact.child.casefold()}-phrase-{phrase_index}"
        ),
        task_id=REASONING_SIDEBAR_TASK_ID,
        family_id="willow",
        query_kind="direct" if reasoning_type == "direct" else "one-hop",
        candidates=candidates,
        router_batch=router_batch,
        correct_candidate_index=answers.index(correct_answer),
        proof_id=f"proof-{reasoning_type}-{fact.fact_id}",
        support_ids=support_ids,
        required_edge_ids=(REASONING_SIDEBAR_NODE_ID,),
        cue_regime="cue_sufficient",
        visible_cue_ids=("willow-club",),
        eligible_task_ids=(REASONING_SIDEBAR_TASK_ID,),
        novelty_regime="new_binding",
        reasoning_type=reasoning_type,
        reasoning_depth=0 if reasoning_type == "direct" else 1,
        prefix_length=len(prefix_tokens),
        mode="closed_book",
        oracle_node_ids=(REASONING_SIDEBAR_NODE_ID,),
    )


def _clause_completion_query(
    tokenizer: TextTokenizer,
    query_id: str,
    reasoning_type: str,
    prefix_text: str,
    answers: tuple[str, ...],
    correct_answer: str,
    support_ids: tuple[str, ...],
) -> KnowledgeQuery:
    prefix_tokens, candidates, router_batch = _query_views(
        tokenizer,
        prefix_text,
        answers,
    )
    return KnowledgeQuery(
        query_id=query_id,
        task_id=REASONING_SIDEBAR_TASK_ID,
        family_id="willow",
        query_kind="direct",
        candidates=candidates,
        router_batch=router_batch,
        correct_candidate_index=answers.index(correct_answer),
        proof_id=f"proof-{query_id}",
        support_ids=support_ids,
        required_edge_ids=(REASONING_SIDEBAR_NODE_ID,),
        cue_regime="cue_sufficient",
        visible_cue_ids=("willow-club",),
        eligible_task_ids=(REASONING_SIDEBAR_TASK_ID,),
        novelty_regime="seen_clause",
        reasoning_type=reasoning_type,
        reasoning_depth=0,
        prefix_length=len(prefix_tokens),
        mode="closed_book",
        oracle_node_ids=(REASONING_SIDEBAR_NODE_ID,),
    )


def _query_views(
    tokenizer: TextTokenizer,
    prefix_text: str,
    answers: tuple[str, ...],
) -> tuple[tuple[int, ...], tuple[KnowledgeCandidate, ...], RouterBatch]:
    prefix_tokens = tokenizer.encode(prefix_text, add_eos=False)
    candidates = tuple(
        _sidebar_candidate(tokenizer, prefix_text, prefix_tokens, answer)
        for answer in answers
    )
    first_answer_tokens = tokenizer.encode(f" {answers[0]}", add_eos=False)
    router_batch = build_prefix_suffix_batches(
        prefix_tokens + first_answer_tokens,
        prefix_length=len(prefix_tokens),
        suffix_length=len(first_answer_tokens),
        pad_token_id=tokenizer.pad_token_id,
    )[0]
    return prefix_tokens, candidates, router_batch


def _sidebar_candidate(
    tokenizer: TextTokenizer,
    prefix_text: str,
    prefix_tokens: tuple[int, ...],
    answer: str,
) -> KnowledgeCandidate:
    combined = tokenizer.encode(f"{prefix_text} {answer}", add_eos=False)
    if combined[: len(prefix_tokens)] != prefix_tokens:
        raise ValueError("candidate tokenization does not preserve the exact prefix")
    suffix_length = len(combined) - len(prefix_tokens)
    if suffix_length < 1:
        raise ValueError("candidate answer did not produce a suffix token")
    return KnowledgeCandidate(
        answer,
        build_prefix_suffix_batches(
            combined,
            prefix_length=len(prefix_tokens),
            suffix_length=suffix_length,
            pad_token_id=tokenizer.pad_token_id,
        )[1],
    )


def _contains_any_word(text: str, words: Sequence[str]) -> bool:
    observed = {word.casefold() for word in _WORD_PATTERN.findall(text)}
    return bool(observed.intersection(word.casefold() for word in words))


def _conservative_json_price(price: str) -> float:
    """Encode an exact per-million route price without rounding the cap down."""
    exact = Decimal(price)
    encoded = float(exact)
    return (
        math.nextafter(encoded, math.inf)
        if Decimal(str(encoded)) < exact
        else encoded
    )


__all__ = [
    "REASONING_SIDEBAR_BATCH_SIZE",
    "REASONING_SIDEBAR_CONTEXT_LENGTH",
    "REASONING_SIDEBAR_GENERATION_VARIANTS",
    "REASONING_SIDEBAR_HARD_CAP_USD",
    "REASONING_SIDEBAR_LORA_RANK",
    "REASONING_SIDEBAR_MAX_OUTPUT_TOKENS",
    "REASONING_SIDEBAR_STORIES_PER_EVIDENCE",
    "REASONING_SIDEBAR_UPDATE_BUDGET",
    "REASONING_SIDEBAR_VERSION",
    "SidebarEvidencePlan",
    "SidebarFact",
    "SidebarReferenceRecord",
    "SidebarRule",
    "SidebarScoreSummary",
    "SidebarStoryPlan",
    "SidebarStoryValidation",
    "build_sidebar_author_requests",
    "build_sidebar_clause_completion_queries",
    "build_sidebar_evidence_plans",
    "build_sidebar_queries",
    "build_sidebar_reference_control",
    "build_sidebar_story_plans",
    "build_sidebar_training_batches",
    "reasoning_sidebar_world",
    "select_sidebar_training_stories",
    "sidebar_query_score_records",
    "sidebar_story_prompt",
    "summarize_sidebar_scores",
    "validate_sidebar_story",
]

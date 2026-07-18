"""Deterministic TinyStories-style rendering at exact semantic token boundaries."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from hashlib import sha256
from heapq import heappop, heappush
from itertools import count
from typing import Literal, TypeAlias

import numpy as np

from apm.continual.knowledge_tasks import KnowledgeCandidate, KnowledgeQuery
from apm.continual.language_tasks import build_prefix_suffix_batches
from apm.data.text.tinyworlds.query_generation import TinyWorldsBundle
from apm.data.text.tinyworlds.schema import (
    AtomPattern,
    DataSplit,
    Entity,
    EntityId,
    GroundAtom,
    HornRule,
    QueryCandidate,
    QueryKind,
    QueryPlan,
    StoryPlan,
    TaskId,
    Variable,
)
from apm.data.text.tinyworlds.templates import (
    TemplateFamily,
    TemplateKind,
    TinyWorldsTemplateRegistry,
    build_template_registry,
)
from apm.lm.text import TextTokenizer
from apm.memory.graph import NodeId


StoryPurpose: TypeAlias = Literal[
    "training",
    "natural_continuation",
    "root_validation",
]

TINYWORLDS_PREFIX_LENGTHS = (64, 128, 192)
TINYWORLDS_CONTEXT_LENGTH = 256


class TinyWorldsRenderingRejection(ValueError):
    """One expected plan/template rejection during deterministic rendering."""


@dataclass(frozen=True, slots=True)
class TinyWorldsRenderPreset:
    """Fixed per-task corpus sizes and tokenizer capacities."""

    training_stories_per_task: int = 1_024
    validation_stories_per_task: int = 128
    test_stories_per_task: int = 128
    validation_query_groups_per_task: int = 256
    test_query_groups_per_task: int = 512
    root_validation_stories: int = 128
    story_token_count: int = 256
    context_length: int = TINYWORLDS_CONTEXT_LENGTH

    def __post_init__(self) -> None:
        values = (
            self.training_stories_per_task,
            self.validation_stories_per_task,
            self.test_stories_per_task,
            self.validation_query_groups_per_task,
            self.test_query_groups_per_task,
            self.root_validation_stories,
            self.story_token_count,
            self.context_length,
        )
        if any(type(value) is not int or value <= 0 for value in values):
            raise ValueError("render preset counts must be positive integers")
        if self.context_length != TINYWORLDS_CONTEXT_LENGTH:
            raise ValueError("TinyWorlds uses the fixed 256-token context")
        if self.story_token_count < max(TINYWORLDS_PREFIX_LENGTHS) + 1:
            raise ValueError("stories must cover the longest evaluation prefix")


TINYWORLDS_RENDER_PRESET = TinyWorldsRenderPreset()


@dataclass(frozen=True, slots=True)
class QueryGroupPlan:
    """One immutable paired-query instance fixed before language rendering.

    The source semantic query and proof may deliberately repeat within one
    split.  The instance identity, candidate order, correct position, and
    holdout identity do not: those presentation choices are fixed here rather
    than being changed while a query is rendered.
    """

    group_id: str
    task_id: TaskId
    split: DataSplit
    occurrence_index: int
    replication_index: int
    source_plan: QueryPlan
    candidates: tuple[QueryCandidate, ...]
    correct_index: int
    template_family_id: str
    holdout_identity_sha256: str

    def __post_init__(self) -> None:
        if type(self.group_id) is not str or not self.group_id:
            raise ValueError("query-group plan ID must be nonempty")
        if type(self.task_id) is not TaskId:
            raise TypeError("query-group task_id must be a TaskId")
        if type(self.split) is not DataSplit or self.split is DataSplit.TRAIN:
            raise ValueError("query-group plans belong to validation or test")
        if any(
            type(value) is not int or value < 0
            for value in (self.occurrence_index, self.replication_index)
        ):
            raise ValueError("query-group occurrence indices must be nonnegative")
        if type(self.source_plan) is not QueryPlan:
            raise TypeError("query-group source_plan must be a QueryPlan")
        if (
            self.source_plan.task_id != self.task_id
            or self.source_plan.split is not self.split
        ):
            raise ValueError("query-group source plan must match its task and split")
        if (
            type(self.candidates) is not tuple
            or len(self.candidates) != 4
            or any(type(item) is not QueryCandidate for item in self.candidates)
        ):
            raise ValueError("query-group plans require four fixed candidates")
        if type(self.correct_index) is not int or not 0 <= self.correct_index < 4:
            raise ValueError("query-group correct_index must be from zero to three")
        if (
            self.candidates[self.correct_index].entity_id
            != self.source_plan.answer_entity_id
        ):
            raise ValueError("query-group correct index must identify the source answer")
        if type(self.template_family_id) is not str or not self.template_family_id:
            raise ValueError("query-group template family ID must be nonempty")
        if (
            type(self.holdout_identity_sha256) is not str
            or len(self.holdout_identity_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.holdout_identity_sha256
            )
        ):
            raise ValueError("query-group holdout identity must be a SHA-256")

    @property
    def source_query_id(self) -> str:
        """Return the canonical source semantic-query identity."""
        return str(self.source_plan.query_ast.query_id)

    @property
    def source_proof_id(self) -> str:
        """Return the canonical source proof identity."""
        return str(self.source_plan.proof.proof_id)


@dataclass(frozen=True, slots=True)
class SentenceAlignment:
    """Exact character span aligned to zero or more symbolic fact/rule IDs."""

    sentence_index: int
    start_character: int
    end_character: int
    fact_ids: tuple[str, ...] = ()
    rule_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            type(self.sentence_index) is not int
            or self.sentence_index < 0
            or type(self.start_character) is not int
            or type(self.end_character) is not int
            or not 0 <= self.start_character < self.end_character
        ):
            raise ValueError("sentence alignment offsets must describe a nonempty span")
        for label, values in (("fact_ids", self.fact_ids), ("rule_ids", self.rule_ids)):
            if type(values) is not tuple or any(
                type(value) is not str or not value for value in values
            ):
                raise ValueError(f"{label} must contain canonical identifiers")


@dataclass(frozen=True, slots=True)
class RenderedStory:
    """One aligned story whose direct symbolic statements remain authoritative."""

    story_id: str
    task_id: str | None
    split: DataSplit
    purpose: StoryPurpose
    text: str
    token_ids: tuple[int, ...]
    template_family_ids: tuple[str, ...]
    plot_id: str
    alignments: tuple[SentenceAlignment, ...]
    text_sha256: str

    def __post_init__(self) -> None:
        if type(self.story_id) is not str or not self.story_id:
            raise ValueError("rendered story_id must be nonempty")
        if self.task_id is not None and (type(self.task_id) is not str or not self.task_id):
            raise ValueError("rendered task_id must be nonempty when present")
        if type(self.split) is not DataSplit:
            raise TypeError("rendered story split must be a DataSplit")
        if self.purpose not in (
            "training",
            "natural_continuation",
            "root_validation",
        ):
            raise ValueError(f"unknown story purpose: {self.purpose}")
        if self.purpose == "training" and self.split is not DataSplit.TRAIN:
            raise ValueError("training stories must belong to the training split")
        if self.purpose == "root_validation" and (
            self.split is not DataSplit.VALIDATION or self.task_id is not None
        ):
            raise ValueError("root stories are task-neutral validation records")
        if type(self.text) is not str or not self.text.strip():
            raise ValueError("rendered story text must contain visible prose")
        if type(self.token_ids) is not tuple or not self.token_ids or any(
            type(token_id) is not int or token_id < 0 for token_id in self.token_ids
        ):
            raise ValueError("rendered token_ids must contain nonnegative integers")
        if type(self.template_family_ids) is not tuple or not self.template_family_ids:
            raise ValueError("rendered stories require template provenance")
        if type(self.alignments) is not tuple or not self.alignments or any(
            type(item) is not SentenceAlignment for item in self.alignments
        ):
            raise ValueError("rendered stories require sentence alignments")
        if tuple(item.sentence_index for item in self.alignments) != tuple(
            range(len(self.alignments))
        ):
            raise ValueError("sentence alignments must use contiguous indices")
        if any(item.end_character > len(self.text) for item in self.alignments):
            raise ValueError("sentence alignments cannot exceed story text")
        if self.text_sha256 != sha256(self.text.encode("utf-8")).hexdigest():
            raise ValueError("rendered story text_sha256 does not match its text")


@dataclass(frozen=True, slots=True)
class RenderedQueryVariant:
    """One exact-token cue variant of an unchanged semantic query group."""

    variant_id: str
    group_id: str
    split: DataSplit
    prefix_text: str
    prefix_token_ids: tuple[int, ...]
    query_core_sha256: str
    candidate_entity_ids: tuple[str, ...]
    knowledge_query: KnowledgeQuery
    text_sha256: str

    def __post_init__(self) -> None:
        if any(
            type(value) is not str or not value
            for value in (
                self.variant_id,
                self.group_id,
                self.prefix_text,
                self.query_core_sha256,
                self.text_sha256,
            )
        ):
            raise ValueError("query variant identifiers and text must be nonempty")
        if type(self.split) is not DataSplit or self.split is DataSplit.TRAIN:
            raise ValueError("knowledge variants belong to validation or test")
        if type(self.prefix_token_ids) is not tuple or len(self.prefix_token_ids) not in (
            TINYWORLDS_PREFIX_LENGTHS
        ):
            raise ValueError("query prefixes must contain exactly 64, 128, or 192 tokens")
        if self.knowledge_query.prefix_length != len(self.prefix_token_ids):
            raise ValueError("query metadata prefix length must match rendered tokens")
        if type(self.candidate_entity_ids) is not tuple or len(self.candidate_entity_ids) != 4:
            raise ValueError("rendered variants require four candidate entity IDs")
        if self.text_sha256 != sha256(self.prefix_text.encode("utf-8")).hexdigest():
            raise ValueError("query text_sha256 does not match prefix_text")


@dataclass(frozen=True, slots=True)
class RenderedQueryGroup:
    """Three paired prefix variants sharing proof, answers, and query core."""

    group_id: str
    task_id: str
    split: DataSplit
    symbolic_query_id: str
    group_plan: QueryGroupPlan
    variants: tuple[RenderedQueryVariant, ...]

    def __post_init__(self) -> None:
        if any(
            type(value) is not str or not value
            for value in (self.group_id, self.task_id, self.symbolic_query_id)
        ):
            raise ValueError("query group identifiers must be nonempty")
        if type(self.group_plan) is not QueryGroupPlan:
            raise TypeError("rendered query groups require their accepted QueryGroupPlan")
        if (
            self.group_plan.group_id != self.group_id
            or str(self.group_plan.task_id) != self.task_id
            or self.group_plan.split is not self.split
            or self.group_plan.source_query_id != self.symbolic_query_id
        ):
            raise ValueError("rendered query group differs from its accepted plan")
        if type(self.variants) is not tuple or len(self.variants) != 3:
            raise ValueError("query groups require exactly three paired variants")
        if tuple(len(item.prefix_token_ids) for item in self.variants) != (
            TINYWORLDS_PREFIX_LENGTHS
        ):
            raise ValueError("query variants must follow 64, 128, 192 token order")
        if any(
            item.group_id != self.group_id or item.split is not self.split
            for item in self.variants
        ):
            raise ValueError("query variants must match their group")
        reference = self.variants[0].knowledge_query
        if reference.proof_id != self.group_plan.source_proof_id:
            raise ValueError("rendered query proof differs from its accepted plan")
        if reference.correct_candidate_index != self.group_plan.correct_index:
            raise ValueError("rendered correct position differs from its accepted plan")
        if self.variants[0].candidate_entity_ids != tuple(
            str(candidate.entity_id) for candidate in self.group_plan.candidates
        ):
            raise ValueError("rendered candidates differ from their accepted plan")
        shared_values = (
            reference.proof_id,
            reference.support_ids,
            reference.required_edge_ids,
            tuple(candidate.answer_text for candidate in reference.candidates),
            reference.correct_candidate_index,
            self.variants[0].candidate_entity_ids,
            self.variants[0].query_core_sha256,
        )
        if any(
            (
                item.knowledge_query.proof_id,
                item.knowledge_query.support_ids,
                item.knowledge_query.required_edge_ids,
                tuple(candidate.answer_text for candidate in item.knowledge_query.candidates),
                item.knowledge_query.correct_candidate_index,
                item.candidate_entity_ids,
                item.query_core_sha256,
            )
            != shared_values
            for item in self.variants[1:]
        ):
            raise ValueError("paired variants must preserve semantic answers and proof")
        core_tokens = self.variants[0].prefix_token_ids
        if any(item.prefix_token_ids[-64:] != core_tokens for item in self.variants[1:]):
            raise ValueError("paired prefixes must share one final 64-token query core")


@dataclass(frozen=True, slots=True)
class RenderedTinyWorlds:
    """Complete deterministic rendered corpus plus exact knowledge queries."""

    bundle_id: str
    registry: TinyWorldsTemplateRegistry
    preset: TinyWorldsRenderPreset
    stories: tuple[RenderedStory, ...]
    query_groups: tuple[RenderedQueryGroup, ...]

    def __post_init__(self) -> None:
        if type(self.bundle_id) is not str or not self.bundle_id:
            raise ValueError("rendered bundle_id must be nonempty")
        if type(self.registry) is not TinyWorldsTemplateRegistry:
            raise TypeError("registry must be a TinyWorldsTemplateRegistry")
        if type(self.preset) is not TinyWorldsRenderPreset:
            raise TypeError("preset must be a TinyWorldsRenderPreset")
        if type(self.stories) is not tuple or any(
            type(story) is not RenderedStory for story in self.stories
        ):
            raise TypeError("stories must contain RenderedStory values")
        if type(self.query_groups) is not tuple or any(
            type(group) is not RenderedQueryGroup for group in self.query_groups
        ):
            raise TypeError("query_groups must contain RenderedQueryGroup values")
        story_ids = tuple(story.story_id for story in self.stories)
        group_ids = tuple(group.group_id for group in self.query_groups)
        if len(set(story_ids)) != len(story_ids) or len(set(group_ids)) != len(group_ids):
            raise ValueError("rendered story and query group IDs must be unique")
        _validate_rendered_split_disjointness(self.stories, self.query_groups)


def render_tinyworlds_bundle(
    bundle: TinyWorldsBundle,
    tokenizer: TextTokenizer,
    preset: TinyWorldsRenderPreset = TINYWORLDS_RENDER_PRESET,
) -> RenderedTinyWorlds:
    """Render an immutable symbolic bundle or reject an inexact tokenization."""
    if type(bundle) is not TinyWorldsBundle:
        raise TypeError("bundle must be a TinyWorldsBundle")
    if not isinstance(tokenizer, TextTokenizer):
        raise TypeError("tokenizer must satisfy TextTokenizer")
    if type(preset) is not TinyWorldsRenderPreset:
        raise TypeError("preset must be a TinyWorldsRenderPreset")
    registry = build_template_registry(bundle.world)
    stories = _render_story_corpus(bundle, registry, tokenizer, preset)
    groups = _render_query_corpus(bundle, registry, tokenizer, preset)
    rendered = RenderedTinyWorlds(
        bundle_id=f"{bundle.bundle_id}:{registry.version}",
        registry=registry,
        preset=preset,
        stories=stories,
        query_groups=groups,
    )
    _validate_rendered_counts(bundle, rendered)
    return rendered


def render_tinyworlds_query_groups(
    bundle: TinyWorldsBundle,
    tokenizer: TextTokenizer,
    preset: TinyWorldsRenderPreset = TINYWORLDS_RENDER_PRESET,
    *,
    registry: TinyWorldsTemplateRegistry | None = None,
) -> tuple[RenderedQueryGroup, ...]:
    """Render only semantic query groups while reusing an existing story pool."""
    if type(bundle) is not TinyWorldsBundle:
        raise TypeError("bundle must be a TinyWorldsBundle")
    if not isinstance(tokenizer, TextTokenizer):
        raise TypeError("tokenizer must satisfy TextTokenizer")
    if type(preset) is not TinyWorldsRenderPreset:
        raise TypeError("preset must be a TinyWorldsRenderPreset")
    resolved_registry = registry or build_template_registry(bundle.world)
    expected_registry = build_template_registry(bundle.world)
    if resolved_registry != expected_registry:
        raise ValueError("query registry must match the symbolic world")
    groups = _render_query_corpus(
        bundle,
        resolved_registry,
        tokenizer,
        preset,
    )
    expected_count = len(bundle.tasks) * (
        preset.validation_query_groups_per_task
        + preset.test_query_groups_per_task
    )
    if len(groups) != expected_count:
        raise RuntimeError("rendered query group count does not match the preset")
    return groups


def build_query_group_plans(
    bundle: TinyWorldsBundle,
    tokenizer: TextTokenizer,
    preset: TinyWorldsRenderPreset = TINYWORLDS_RENDER_PRESET,
    *,
    registry: TinyWorldsTemplateRegistry | None = None,
) -> tuple[QueryGroupPlan, ...]:
    """Resolve and return the exact accepted plan for every rendered group."""
    _, groups = _resolve_query_group_corpus(
        bundle,
        tokenizer,
        preset,
        registry=registry,
    )
    return tuple(group.group_plan for group in groups)


def expand_query_group_plan_attempts(
    bundle: TinyWorldsBundle,
    preset: TinyWorldsRenderPreset = TINYWORLDS_RENDER_PRESET,
    *,
    registry: TinyWorldsTemplateRegistry | None = None,
) -> tuple[tuple[QueryGroupPlan, ...], ...]:
    """Expand every ordered immutable attempt before rendering any group."""
    if type(bundle) is not TinyWorldsBundle:
        raise TypeError("bundle must be a TinyWorldsBundle")
    if type(preset) is not TinyWorldsRenderPreset:
        raise TypeError("preset must be a TinyWorldsRenderPreset")
    resolved_registry = registry or build_template_registry(bundle.world)
    if resolved_registry != build_template_registry(bundle.world):
        raise ValueError("query registry must match the symbolic world")
    counts = {
        DataSplit.VALIDATION: preset.validation_query_groups_per_task,
        DataSplit.TEST: preset.test_query_groups_per_task,
    }
    plans_by_task_split = _source_query_plans(bundle)
    return tuple(
        tuple(
            _query_group_plan(
                resolved_registry,
                plans_by_task_split[(task.task_id, split)],
                group_index,
                source_offset=source_offset,
            )
            for source_offset in range(
                len(plans_by_task_split[(task.task_id, split)])
            )
        )
        for task in bundle.tasks
        for split in (DataSplit.VALIDATION, DataSplit.TEST)
        for group_index in range(counts[split])
    )


def _source_query_plans(
    bundle: TinyWorldsBundle,
) -> dict[tuple[TaskId, DataSplit], tuple[QueryPlan, ...]]:
    plans = {
        (task.task_id, split): tuple(
            plan
            for plan in bundle.query_plans
            if plan.task_id == task.task_id and plan.split is split
        )
        for task in bundle.tasks
        for split in (DataSplit.VALIDATION, DataSplit.TEST)
    }
    missing = tuple(key for key, values in plans.items() if not values)
    if missing:
        raise ValueError(f"symbolic query plans do not cover task/split pairs: {missing}")
    return plans


def _query_group_plan(
    registry: TinyWorldsTemplateRegistry,
    source_plans: tuple[QueryPlan, ...],
    occurrence_index: int,
    *,
    source_offset: int,
) -> QueryGroupPlan:
    source_index = (occurrence_index + source_offset) % len(source_plans)
    source = source_plans[source_index]
    replication_index = occurrence_index // len(source_plans)
    candidate_rotation = replication_index % 4
    candidates = (
        source.candidates[candidate_rotation:]
        + source.candidates[:candidate_rotation]
    )
    correct_index = (source.correct_index - candidate_rotation) % 4
    group_id = (
        f"group:{source.query_ast.query_id}:{source.split.value}:"
        f"{occurrence_index:04d}"
    )
    template = registry.family(
        TemplateKind.QUERY,
        source.kind.value,
        source.split,
    )
    holdout = source.holdout
    holdout_payload = "\n".join(
        (
            "tinyworlds-query-group-holdout-v1",
            group_id,
            str(replication_index),
            template.family_id,
            holdout.template_family_id,
            holdout.plot_id,
            holdout.query_phrasing_id,
            holdout.entity_combination_id,
            holdout.proof_chain_id,
            holdout.symbolic_text_sha256,
        )
    )
    return QueryGroupPlan(
        group_id=group_id,
        task_id=source.task_id,
        split=source.split,
        occurrence_index=occurrence_index,
        replication_index=replication_index,
        source_plan=source,
        candidates=candidates,
        correct_index=correct_index,
        template_family_id=template.family_id,
        holdout_identity_sha256=sha256(holdout_payload.encode("utf-8")).hexdigest(),
    )


def _render_story_corpus(
    bundle: TinyWorldsBundle,
    registry: TinyWorldsTemplateRegistry,
    tokenizer: TextTokenizer,
    preset: TinyWorldsRenderPreset,
) -> tuple[RenderedStory, ...]:
    split_counts = {
        DataSplit.TRAIN: preset.training_stories_per_task,
        DataSplit.VALIDATION: preset.validation_stories_per_task,
        DataSplit.TEST: preset.test_stories_per_task,
    }
    plans = {
        (plan.task_id, plan.split): plan for plan in bundle.story_plans
    }
    task_stories = tuple(
        _render_story(
            bundle,
            registry,
            tokenizer,
            plans[(task.task_id, split)],
            story_index,
            preset.story_token_count,
        )
        for task in bundle.tasks
        for split in DataSplit
        for story_index in range(split_counts[split])
    )
    root_stories = tuple(
        _render_root_story(registry, tokenizer, story_index, preset.story_token_count)
        for story_index in range(preset.root_validation_stories)
    )
    return task_stories + root_stories


def _render_story(
    bundle: TinyWorldsBundle,
    registry: TinyWorldsTemplateRegistry,
    tokenizer: TextTokenizer,
    plan: StoryPlan,
    story_index: int,
    token_count: int,
) -> RenderedStory:
    facts_by_id = {fact.atom_id: fact for fact in bundle.facts}
    rules_by_id = {rule.rule_id: rule for rule in bundle.rules}
    entities = {entity.entity_id: entity for entity in bundle.entities}
    rotated_facts = _rotated_window(
        plan.direct_fact_ids,
        story_index,
        1 if plan.split is DataSplit.TRAIN else 4,
    )
    rotated_rules = _rotated_window(plan.rule_ids, story_index, 1)
    aligned_sentences = tuple(
        (
            registry.family(TemplateKind.FACT, str(facts_by_id[fact_id].predicate_id), plan.split),
            _fact_statement(facts_by_id[fact_id], entities),
            (str(fact_id),),
            (),
        )
        for fact_id in rotated_facts
    ) + tuple(
        (
            registry.family(TemplateKind.RULE, str(rule_id), plan.split),
            _rule_statement(rules_by_id[rule_id], entities),
            (),
            (str(rule_id),),
        )
        for rule_id in rotated_rules
    )
    base_text, base_alignments = _join_aligned_sentences(aligned_sentences)
    plot = registry.family(
        TemplateKind.PLOT,
        bundle.world.task(plan.task_id).kind.value,
        plan.split,
    )
    plot_sentence = plot.render(
        f"the friends opened story page {story_index + 1} and listened closely."
    )
    text_with_plot = f"{plot_sentence} {base_text}"
    shift = len(plot_sentence) + 1
    shifted_alignments = (
        SentenceAlignment(0, 0, len(plot_sentence)),
        *(
            SentenceAlignment(
                item.sentence_index + 1,
                item.start_character + shift,
                item.end_character + shift,
                item.fact_ids,
                item.rule_ids,
            )
            for item in base_alignments
        ),
    )
    fitted = _fit_exact_tokens(
        text_with_plot,
        token_count,
        tokenizer,
        salt=f"story:{plan.story_id}:{story_index}",
        append=True,
    )
    text = fitted[0]
    alignments = shifted_alignments + (
        SentenceAlignment(
            len(shifted_alignments),
            len(text_with_plot),
            len(text),
        ),
    ) if len(text) > len(text_with_plot) else shifted_alignments
    story_id = f"rendered:{plan.story_id}:{story_index:04d}"
    story = RenderedStory(
        story_id=story_id,
        task_id=str(plan.task_id),
        split=plan.split,
        purpose="training" if plan.split is DataSplit.TRAIN else "natural_continuation",
        text=text,
        token_ids=fitted[1],
        template_family_ids=(plot.family_id,) + tuple(
            family.family_id for family, _, _, _ in aligned_sentences
        ),
        plot_id=f"plot:{plan.split.value}:{plan.task_id}:{story_index:04d}",
        alignments=alignments,
        text_sha256=sha256(text.encode("utf-8")).hexdigest(),
    )
    expected_sentences = (plot_sentence,) + tuple(
        family.render(statement)
        for family, statement, _, _ in aligned_sentences
    )
    if any(
        story.text[alignment.start_character : alignment.end_character]
        != expected
        for alignment, expected in zip(story.alignments, expected_sentences)
    ):
        raise RuntimeError("rendered story alignment differs from its symbolic template")
    return story


def _render_root_story(
    registry: TinyWorldsTemplateRegistry,
    tokenizer: TextTokenizer,
    story_index: int,
    token_count: int,
) -> RenderedStory:
    family = registry.family(
        TemplateKind.PLOT,
        "root_validation",
        DataSplit.VALIDATION,
    )
    base = family.render(
        f"four children shared a plain picnic story numbered {story_index + 1}."
    )
    text, token_ids = _fit_exact_tokens(
        base,
        token_count,
        tokenizer,
        salt=f"root-validation:{story_index}",
        append=True,
    )
    alignments = (SentenceAlignment(0, 0, len(base)),) + (
        (SentenceAlignment(1, len(base), len(text)),) if len(text) > len(base) else ()
    )
    return RenderedStory(
        story_id=f"rendered:root-validation:{story_index:04d}",
        task_id=None,
        split=DataSplit.VALIDATION,
        purpose="root_validation",
        text=text,
        token_ids=token_ids,
        template_family_ids=(family.family_id,),
        plot_id=f"plot:validation:root:{story_index:04d}",
        alignments=alignments,
        text_sha256=sha256(text.encode("utf-8")).hexdigest(),
    )


def _render_query_corpus(
    bundle: TinyWorldsBundle,
    registry: TinyWorldsTemplateRegistry,
    tokenizer: TextTokenizer,
    preset: TinyWorldsRenderPreset,
) -> tuple[RenderedQueryGroup, ...]:
    _, groups = _resolve_query_group_corpus(
        bundle,
        tokenizer,
        preset,
        registry=registry,
    )
    return groups


def _resolve_query_group_corpus(
    bundle: TinyWorldsBundle,
    tokenizer: TextTokenizer,
    preset: TinyWorldsRenderPreset,
    *,
    registry: TinyWorldsTemplateRegistry | None,
) -> tuple[tuple[QueryGroupPlan, ...], tuple[RenderedQueryGroup, ...]]:
    """Resolve the accepted immutable plan and rendered record for every slot."""
    if not isinstance(tokenizer, TextTokenizer):
        raise TypeError("tokenizer must satisfy TextTokenizer")
    resolved_registry = registry or build_template_registry(bundle.world)
    attempts_by_slot = expand_query_group_plan_attempts(
        bundle,
        preset,
        registry=resolved_registry,
    )
    accepted: list[QueryGroupPlan] = []
    groups: list[RenderedQueryGroup] = []
    for attempts in attempts_by_slot:
        group = _render_query_group_attempts(
            bundle,
            resolved_registry,
            tokenizer,
            attempts,
        )
        accepted.append(group.group_plan)
        groups.append(group)
    return tuple(accepted), tuple(groups)


def _render_query_group_attempts(
    bundle: TinyWorldsBundle,
    registry: TinyWorldsTemplateRegistry,
    tokenizer: TextTokenizer,
    attempts: tuple[QueryGroupPlan, ...],
) -> RenderedQueryGroup:
    if not attempts:
        raise ValueError("query rendering requires at least one immutable attempt")
    rejections: list[str] = []
    for group_plan in attempts:
        try:
            return _render_query_group(
                bundle,
                registry,
                tokenizer,
                group_plan,
            )
        except TinyWorldsRenderingRejection as error:
            rejections.append(f"{group_plan.source_query_id}: {error}")
    raise TinyWorldsRenderingRejection(
        "all deterministic query-plan attempts were rejected for slot "
        f"{attempts[0].occurrence_index}: {'; '.join(rejections)}"
    )


def _render_query_group_at_index(
    bundle: TinyWorldsBundle,
    registry: TinyWorldsTemplateRegistry,
    tokenizer: TextTokenizer,
    source_plans: tuple[QueryPlan, ...],
    group_index: int,
) -> RenderedQueryGroup:
    """Render one slot, advancing only after an expected render rejection."""
    attempts = tuple(
        _query_group_plan(
            registry,
            source_plans,
            group_index,
            source_offset=source_offset,
        )
        for source_offset in range(len(source_plans))
    )
    return _render_query_group_attempts(
        bundle,
        registry,
        tokenizer,
        attempts,
    )


def _render_query_group(
    bundle: TinyWorldsBundle,
    registry: TinyWorldsTemplateRegistry,
    tokenizer: TextTokenizer,
    group_plan: QueryGroupPlan,
) -> RenderedQueryGroup:
    plan = group_plan.source_plan
    entities = {entity.entity_id: entity for entity in bundle.entities}
    group_id = group_plan.group_id
    query_family = registry.family(
        TemplateKind.QUERY,
        plan.kind.value,
        plan.split,
    )
    if query_family.family_id != group_plan.template_family_id:
        raise ValueError("query-group template provenance does not match the registry")
    question = _query_statement(
        bundle,
        plan,
        entities,
        group_plan.occurrence_index,
    )
    core_base = query_family.render(f"{question} Answer:")
    core_text, core_tokens = _fit_exact_tokens(
        core_base,
        64,
        tokenizer,
        salt=f"{group_id}:core",
        append=False,
    )
    candidate_entities = tuple(
        entities[candidate.entity_id] for candidate in group_plan.candidates
    )
    candidate_texts = _equal_candidate_texts(
        core_text,
        candidate_entities,
        tokenizer,
        salt=group_id,
    )
    variants = tuple(
        _render_query_variant(
            bundle,
            tokenizer,
            plan,
            group_id,
            prefix_length,
            core_text,
            core_tokens,
            candidate_entities,
            candidate_texts,
            group_plan.correct_index,
            group_plan.replication_index,
        )
        for prefix_length in TINYWORLDS_PREFIX_LENGTHS
    )
    return RenderedQueryGroup(
        group_id=group_id,
        task_id=str(plan.task_id),
        split=plan.split,
        symbolic_query_id=str(plan.query_ast.query_id),
        group_plan=group_plan,
        variants=variants,
    )


def _render_query_variant(
    bundle: TinyWorldsBundle,
    tokenizer: TextTokenizer,
    plan: QueryPlan,
    group_id: str,
    prefix_length: int,
    core_text: str,
    core_tokens: tuple[int, ...],
    candidate_entities: tuple[Entity, ...],
    candidate_texts: tuple[str, ...],
    correct_candidate_index: int,
    occurrence_index: int,
) -> RenderedQueryVariant:
    cue_regime = _cue_regime(
        f"{plan.task_id}:{plan.split.value}:{plan.query_ast.query_id}",
        occurrence_index,
        prefix_length,
    )
    cue_text, visible_cues, eligible_tasks = _cue_block(
        bundle,
        plan,
        group_id,
        prefix_length,
        cue_regime,
    )
    prefix_base = core_text if prefix_length == 64 else f"{cue_text}\n{core_text}"
    prefix_text, prefix_tokens = _fit_exact_tokens(
        prefix_base,
        prefix_length,
        tokenizer,
        salt=f"{group_id}:{prefix_length}:{cue_regime}",
        append=False,
    )
    if prefix_tokens[-64:] != core_tokens:
        raise TinyWorldsRenderingRejection(
            "tokenizer could not preserve the final 64-token query core"
        )
    combined_tokens = tuple(
        tokenizer.encode(prefix_text + answer) for answer in candidate_texts
    )
    suffix_lengths = tuple(len(tokens) - prefix_length for tokens in combined_tokens)
    if any(tokens[:prefix_length] != prefix_tokens for tokens in combined_tokens):
        raise TinyWorldsRenderingRejection(
            "standalone prefix tokenization is not a combined-sequence prefix"
        )
    if len(set(suffix_lengths)) != 1 or suffix_lengths[0] < 1:
        raise TinyWorldsRenderingRejection(
            "candidate suffixes must have one equal positive token count"
        )
    if any(len(tokens) > TINYWORLDS_CONTEXT_LENGTH for tokens in combined_tokens):
        raise TinyWorldsRenderingRejection(
            "prefix plus candidate exceeds the 256-token context"
        )
    if plan.kind is not QueryKind.OPEN_BOOK:
        _reject_candidate_leakage(prefix_text, prefix_tokens, candidate_entities, candidate_texts, tokenizer)
    batches = tuple(
        build_prefix_suffix_batches(
            tokens,
            prefix_length,
            suffix_lengths[0],
            pad_token_id=tokenizer.pad_token_id,
        )
        for tokens in combined_tokens
    )
    candidates = tuple(
        KnowledgeCandidate(answer, competence)
        for answer, (_, competence) in zip(candidate_texts, batches)
    )
    oracle_nodes = tuple(NodeId(str(task_id)) for task_id in plan.hard_oracle_task_ids)
    support_ids = tuple(
        str(item)
        for item in (
            *plan.proof.supporting_fact_ids,
            *plan.proof.supporting_rule_ids,
        )
    )
    edge_node_by_id = {
        task.incoming_edge_id: NodeId(str(task.task_id))
        for task in bundle.tasks
    }
    knowledge_query = KnowledgeQuery(
        query_id=f"{group_id}:prefix-{prefix_length}",
        task_id=str(plan.task_id),
        family_id=str(bundle.world.task(plan.task_id).family_id),
        query_kind=plan.kind.value,
        candidates=candidates,
        router_batch=batches[0][0],
        correct_candidate_index=correct_candidate_index,
        proof_id=str(plan.proof.proof_id),
        support_ids=support_ids,
        required_edge_ids=tuple(
            edge_node_by_id[edge_id]
            for edge_id in plan.proof.required_edge_ids
        ),
        cue_regime=cue_regime,
        visible_cue_ids=visible_cues,
        eligible_task_ids=tuple(str(task_id) for task_id in eligible_tasks),
        novelty_regime="novel_binding",
        reasoning_type=(
            "cross_branch" if plan.kind is QueryKind.CROSS_BRANCH else plan.kind.value
        ),
        reasoning_depth=plan.proof.depth,
        prefix_length=prefix_length,
        mode="open_book" if plan.kind is QueryKind.OPEN_BOOK else "closed_book",
        oracle_node_ids=oracle_nodes,
    )
    variant_id = f"{group_id}:prefix-{prefix_length}"
    return RenderedQueryVariant(
        variant_id=variant_id,
        group_id=group_id,
        split=plan.split,
        prefix_text=prefix_text,
        prefix_token_ids=prefix_tokens,
        query_core_sha256=sha256(core_text.encode("utf-8")).hexdigest(),
        candidate_entity_ids=tuple(str(entity.entity_id) for entity in candidate_entities),
        knowledge_query=knowledge_query,
        text_sha256=sha256(prefix_text.encode("utf-8")).hexdigest(),
    )


def _cue_regime(
    balance_namespace: str,
    occurrence_index: int,
    prefix_length: int,
) -> str:
    digest_value = int(
        sha256(
            f"cue-regime:{balance_namespace}:{prefix_length}".encode("utf-8")
        ).hexdigest(),
        16,
    )
    allowed = (
        ("cue_hidden_or_ambiguous", "cue_free_control")
        if prefix_length == 64
        else (
            "cue_sufficient",
            "cue_present",
            "cue_hidden_or_ambiguous",
            "cue_free_control",
        )
    )
    return allowed[
        (
            occurrence_index % len(allowed)
            + occurrence_index // 4
            + digest_value
        )
        % len(allowed)
    ]


def _cue_block(
    bundle: TinyWorldsBundle,
    plan: QueryPlan,
    group_id: str,
    prefix_length: int,
    cue_regime: str,
) -> tuple[str, tuple[str, ...], tuple[TaskId, ...]]:
    intended_task = bundle.world.task(plan.task_id)
    family_tasks = tuple(
        task.task_id
        for task in bundle.tasks
        if task.family_id == intended_task.family_id
    )
    all_tasks = tuple(task.task_id for task in bundle.tasks)
    if prefix_length == 64:
        return "", (), all_tasks
    if cue_regime == "cue_sufficient":
        return (
            f"This page belongs only to task {plan.task_id}. The task mark is exact.",
            (f"task:{plan.task_id}",),
            (plan.task_id,),
        )
    if cue_regime == "cue_present":
        return (
            f"This page comes from the {intended_task.family_id} family, among several chapters.",
            (f"family:{intended_task.family_id}",),
            family_tasks,
        )
    if cue_regime == "cue_free_control":
        control = sha256(f"control:{group_id}:{prefix_length}".encode("utf-8")).hexdigest()[:8]
        return (
            f"A plain gray card carried the neutral mark {control} and no chapter clue.",
            (),
            all_tasks,
        )
    return (
        "Several unrelated story pages lay together without a readable chapter mark.",
        (),
        all_tasks,
    )


def _equal_candidate_texts(
    prefix_text: str,
    entities: tuple[Entity, ...],
    tokenizer: TextTokenizer,
    *,
    salt: str,
) -> tuple[str, ...]:
    bases = tuple(f" The answer was {entity.name}." for entity in entities)
    base_sequences = tuple(tokenizer.encode(prefix_text + base) for base in bases)
    prefix_tokens = tokenizer.encode(prefix_text)
    if any(sequence[: len(prefix_tokens)] != prefix_tokens for sequence in base_sequences):
        raise TinyWorldsRenderingRejection(
            "candidate boundary changes the standalone prefix tokenization"
        )
    base_lengths = tuple(len(sequence) - len(prefix_tokens) for sequence in base_sequences)
    for target_length in range(max(base_lengths), TINYWORLDS_CONTEXT_LENGTH - len(prefix_tokens) + 1):
        rendered: list[str] = []
        for candidate_index, base in enumerate(bases):
            fitted = _fit_candidate_delta(
                prefix_text,
                base,
                target_length,
                tokenizer,
                salt=f"{salt}:candidate:{candidate_index}",
            )
            if fitted is None:
                break
            rendered.append(fitted)
        if len(rendered) == 4:
            return tuple(rendered)
    raise TinyWorldsRenderingRejection(
        "could not render four equal-token answer candidates"
    )


_PADDING_FRAGMENTS = (
    " It was calm.",
    " Everyone listened.",
    " The day felt bright.",
    " Nearby, a bird sang.",
    " Very softly.",
    " Well.",
    " So.",
    ".",
    " ",
)


def _fit_exact_tokens(
    base_text: str,
    target_count: int,
    tokenizer: TextTokenizer,
    *,
    salt: str,
    append: bool,
) -> tuple[str, tuple[int, ...]]:
    if type(base_text) is not str or not base_text.strip():
        raise ValueError("base text must contain visible prose")
    initial_tokens = tokenizer.encode(base_text)
    if len(initial_tokens) > target_count:
        raise TinyWorldsRenderingRejection(
            f"base rendering uses {len(initial_tokens)} tokens, exceeding {target_count}"
        )
    if len(initial_tokens) == target_count:
        return base_text, initial_tokens
    offset = int(sha256(salt.encode("utf-8")).hexdigest(), 16) % len(_PADDING_FRAGMENTS)
    fragments = _PADDING_FRAGMENTS[offset:] + _PADDING_FRAGMENTS[:offset]
    fitted = _best_first_exact_tokens(
        base_text,
        initial_tokens,
        target_count,
        tokenizer,
        fragments,
        append=append,
    )
    if fitted is not None:
        return fitted
    fitted = _breadth_first_exact_tokens(
        base_text,
        initial_tokens,
        target_count,
        tokenizer,
        fragments,
        append=append,
    )
    if fitted is not None:
        return fitted
    raise TinyWorldsRenderingRejection(
        f"could not fit rendering to exactly {target_count} tokens"
    )


def _best_first_exact_tokens(
    base_text: str,
    initial_tokens: tuple[int, ...],
    target_count: int,
    tokenizer: TextTokenizer,
    fragments: tuple[str, ...],
    *,
    append: bool,
) -> tuple[str, tuple[int, ...]] | None:
    """Search closest-under-target lengths first with deterministic ties."""
    discovery_order = count()
    pending: list[tuple[int, int, str]] = [
        (-len(initial_tokens), next(discovery_order), base_text)
    ]
    seen_counts = {len(initial_tokens)}
    while pending:
        _, _, text = heappop(pending)
        for fragment in fragments:
            candidate = _add_padding_fragment(text, fragment, append=append)
            tokens = tokenizer.encode(candidate)
            token_count = len(tokens)
            if token_count == target_count:
                return candidate, tokens
            if token_count < target_count and token_count not in seen_counts:
                seen_counts.add(token_count)
                heappush(
                    pending,
                    (-token_count, next(discovery_order), candidate),
                )
    return None


def _breadth_first_exact_tokens(
    base_text: str,
    initial_tokens: tuple[int, ...],
    target_count: int,
    tokenizer: TextTokenizer,
    fragments: tuple[str, ...],
    *,
    append: bool,
) -> tuple[str, tuple[int, ...]] | None:
    """Retain the exhaustive original token-count fallback for edge tokenizers."""
    seen_counts = {len(initial_tokens)}
    pending = deque((base_text,))
    while pending:
        text = pending.popleft()
        for fragment in fragments:
            candidate = _add_padding_fragment(text, fragment, append=append)
            tokens = tokenizer.encode(candidate)
            token_count = len(tokens)
            if token_count == target_count:
                return candidate, tokens
            if token_count < target_count and token_count not in seen_counts:
                seen_counts.add(token_count)
                pending.append(candidate)
    return None


def _add_padding_fragment(text: str, fragment: str, *, append: bool) -> str:
    """Add one neutral fragment without joining or truncating visible words."""
    if append:
        return text + fragment
    visible = fragment.strip()
    return f"{visible} {text}" if visible else f" {text}"


def _fit_candidate_delta(
    prefix_text: str,
    base_answer: str,
    target_delta: int,
    tokenizer: TextTokenizer,
    *,
    salt: str,
) -> str | None:
    prefix_tokens = tokenizer.encode(prefix_text)
    offset = int(sha256(salt.encode("utf-8")).hexdigest(), 16) % len(_PADDING_FRAGMENTS)
    fragments = _PADDING_FRAGMENTS[offset:] + _PADDING_FRAGMENTS[:offset]
    base_sequence = tokenizer.encode(prefix_text + base_answer)
    if base_sequence[: len(prefix_tokens)] != prefix_tokens:
        return None
    base_delta = len(base_sequence) - len(prefix_tokens)
    if base_delta == target_delta:
        return base_answer
    seen_deltas = {base_delta}
    pending = deque((base_answer,))
    while pending:
        answer = pending.popleft()
        for fragment in fragments:
            candidate = answer + fragment
            combined = tokenizer.encode(prefix_text + candidate)
            if combined[: len(prefix_tokens)] != prefix_tokens:
                continue
            delta = len(combined) - len(prefix_tokens)
            if delta == target_delta:
                return candidate
            if delta < target_delta and delta not in seen_deltas:
                seen_deltas.add(delta)
                pending.append(candidate)
    return None


def _fact_statement(
    atom: GroundAtom,
    entities: dict[EntityId, Entity],
) -> str:
    relation = _humanize_identifier(str(atom.predicate_id).split(":")[-1])
    names = tuple(entities[argument].name for argument in atom.arguments)
    if len(names) == 1:
        return f"{names[0]} was known as {relation}."
    if len(names) == 2:
        return f"{names[0]} had the {relation} bond with {names[1]}."
    return f"during {names[2]}, {names[0]} had the {relation} bond with {names[1]}."


def _rule_statement(rule: HornRule, entities: dict[EntityId, Entity]) -> str:
    body = " and ".join(_pattern_statement(pattern, entities) for pattern in rule.body)
    head = _pattern_statement(rule.head, entities)
    return f"whenever {body}, everyone could conclude that {head}."


def _pattern_statement(
    pattern: AtomPattern,
    entities: dict[EntityId, Entity],
) -> str:
    relation = _humanize_identifier(str(pattern.predicate_id).split(":")[-1])
    terms = tuple(
        entities[term].name if type(term) is EntityId else f"someone-{term.name.lower()}"
        for term in pattern.arguments
    )
    return f"{relation} joined {' and '.join(terms)}"


def _query_statement(
    bundle: TinyWorldsBundle,
    plan: QueryPlan,
    entities: dict[EntityId, Entity],
    group_index: int,
) -> str:
    clause = plan.query_ast.clauses[0]
    relation = _humanize_identifier(str(clause.predicate_id).split(":")[-1])
    visible_terms = tuple(
        entities[term].name
        for term in clause.arguments
        if type(term) is EntityId
    )
    question = (
        f"which name completes the {relation} relation for "
        f"{' in '.join(visible_terms)} on story card {group_index + 1}?"
    )
    if plan.kind is QueryKind.OPEN_BOOK:
        facts_by_id = {fact.atom_id: fact for fact in bundle.facts}
        printed_support = " ".join(
            _fact_statement(facts_by_id[fact_id], entities)
            for fact_id in plan.open_book_fact_ids
        )
        return f"the page openly says: {printed_support} Now {question}"
    return question


def _join_aligned_sentences(
    sentences: tuple[
        tuple[TemplateFamily, str, tuple[str, ...], tuple[str, ...]], ...
    ],
) -> tuple[str, tuple[SentenceAlignment, ...]]:
    text_parts: list[str] = []
    alignments: list[SentenceAlignment] = []
    cursor = 0
    for index, (family, statement, fact_ids, rule_ids) in enumerate(sentences):
        sentence = family.render(statement)
        separator = " " if text_parts else ""
        text_parts.append(separator + sentence)
        start = cursor + len(separator)
        end = start + len(sentence)
        alignments.append(SentenceAlignment(index, start, end, fact_ids, rule_ids))
        cursor = end
    return "".join(text_parts), tuple(alignments)


def _rotated_window(values: tuple, index: int, count: int) -> tuple:
    if not values:
        return ()
    start = index % len(values)
    return tuple(values[(start + offset) % len(values)] for offset in range(min(count, len(values))))


def _humanize_identifier(value: str) -> str:
    return value.replace("_", " ").replace("-", " ")


def _reject_candidate_leakage(
    prefix_text: str,
    prefix_tokens: tuple[int, ...],
    entities: tuple[Entity, ...],
    candidate_texts: tuple[str, ...],
    tokenizer: TextTokenizer,
) -> None:
    folded_prefix = prefix_text.casefold()
    if any(entity.name.casefold() in folded_prefix for entity in entities):
        raise TinyWorldsRenderingRejection(
            "closed-book prefix contains a candidate entity string"
        )
    candidate_sequences = tuple(tokenizer.encode(text) for text in candidate_texts) + tuple(
        tokenizer.encode(entity.name) for entity in entities
    )
    if any(_contains_subsequence(prefix_tokens, sequence) for sequence in candidate_sequences):
        raise TinyWorldsRenderingRejection(
            "closed-book prefix contains a candidate token subsequence"
        )


def _contains_subsequence(sequence: tuple[int, ...], subsequence: tuple[int, ...]) -> bool:
    return bool(subsequence) and any(
        sequence[start : start + len(subsequence)] == subsequence
        for start in range(len(sequence) - len(subsequence) + 1)
    )


def _validate_rendered_counts(
    bundle: TinyWorldsBundle,
    rendered: RenderedTinyWorlds,
) -> None:
    preset = rendered.preset
    expected_story_counts = {
        DataSplit.TRAIN: preset.training_stories_per_task,
        DataSplit.VALIDATION: preset.validation_stories_per_task,
        DataSplit.TEST: preset.test_stories_per_task,
    }
    for task in bundle.tasks:
        for split, expected in expected_story_counts.items():
            actual = sum(
                story.task_id == str(task.task_id) and story.split is split
                for story in rendered.stories
            )
            if actual != expected:
                raise ValueError(
                    f"task {task.task_id} {split.value} has {actual} stories, expected {expected}"
                )
        for split, expected in (
            (DataSplit.VALIDATION, preset.validation_query_groups_per_task),
            (DataSplit.TEST, preset.test_query_groups_per_task),
        ):
            actual = sum(
                group.task_id == str(task.task_id) and group.split is split
                for group in rendered.query_groups
            )
            if actual != expected:
                raise ValueError(
                    f"task {task.task_id} {split.value} has {actual} query groups, expected {expected}"
                )
    root_count = sum(story.purpose == "root_validation" for story in rendered.stories)
    if root_count != preset.root_validation_stories:
        raise ValueError("rendered root-validation story count does not match preset")


def _validate_rendered_split_disjointness(
    stories: tuple[RenderedStory, ...],
    groups: tuple[RenderedQueryGroup, ...],
) -> None:
    text_hashes = tuple(
        {
            story.text_sha256
            for story in stories
            if story.split is split
        }
        | {
            variant.text_sha256
            for group in groups
            if group.split is split
            for variant in group.variants
        }
        for split in DataSplit
    )
    if any(
        text_hashes[left] & text_hashes[right]
        for left, right in ((0, 1), (0, 2), (1, 2))
    ):
        raise ValueError("rendered text hashes must be disjoint across splits")
    template_ids = tuple(
        {
            template_id
            for story in stories
            if story.split is split
            for template_id in story.template_family_ids
        }
        for split in DataSplit
    )
    if any(
        template_ids[left] & template_ids[right]
        for left, right in ((0, 1), (0, 2), (1, 2))
    ):
        raise ValueError("rendered template families must be disjoint across splits")


__all__ = [
    "TINYWORLDS_CONTEXT_LENGTH",
    "TINYWORLDS_PREFIX_LENGTHS",
    "TINYWORLDS_RENDER_PRESET",
    "QueryGroupPlan",
    "RenderedQueryGroup",
    "RenderedQueryVariant",
    "RenderedStory",
    "RenderedTinyWorlds",
    "SentenceAlignment",
    "StoryPurpose",
    "TinyWorldsRenderPreset",
    "TinyWorldsRenderingRejection",
    "build_query_group_plans",
    "expand_query_group_plan_attempts",
    "render_tinyworlds_bundle",
    "render_tinyworlds_query_groups",
]

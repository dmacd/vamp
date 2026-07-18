from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from hashlib import sha256
import re

import numpy as np
import pytest

import apm.data.text.tinyworlds.rendering as rendering_module
from apm.data.text.tinyworlds.query_generation import generate_calibration_bundle
from apm.data.text.tinyworlds.adapters import (
    TinyWorldsTrainingDataConfig,
    prepare_tinyworlds_curriculum,
)
from apm.data.text.tinyworlds.rendering import (
    TINYWORLDS_RENDER_PRESET,
    TinyWorldsRenderPreset,
    TinyWorldsRenderingRejection,
    build_query_group_plans,
    expand_query_group_plan_attempts,
    render_tinyworlds_bundle,
)
from apm.data.text.tinyworlds.schema import DataSplit
from apm.data.text.tinyworlds.templates import (
    TemplateKind,
    build_template_registry,
)


@dataclass(frozen=True)
class _WhitespaceTokenizer:
    @property
    def vocab_size(self) -> int:
        return 65_536

    @property
    def pad_token_id(self) -> int:
        return 0

    @property
    def eos_token_id(self) -> int:
        return 1

    def encode(self, text: str, *, add_eos: bool = False) -> tuple[int, ...]:
        words = tuple(re.findall(r"\S+", text))
        tokens = tuple(
            2 + int.from_bytes(sha256(word.encode("utf-8")).digest()[:2], "big")
            for word in words
        )
        return tokens + ((self.eos_token_id,) if add_eos else ())

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        del skip_special_tokens
        return " ".join(str(token_id) for token_id in token_ids)


def _tiny_preset(*, query_groups: int = 4) -> TinyWorldsRenderPreset:
    return TinyWorldsRenderPreset(
        training_stories_per_task=2,
        validation_stories_per_task=1,
        test_stories_per_task=1,
        validation_query_groups_per_task=query_groups,
        test_query_groups_per_task=query_groups,
        root_validation_stories=2,
        story_token_count=256,
        context_length=256,
    )


def test_template_registry_is_complete_and_split_disjoint() -> None:
    bundle = generate_calibration_bundle("0" * 64)
    registry = build_template_registry(bundle.world)

    for predicate in bundle.world.registry.predicates:
        assert tuple(
            registry.family(TemplateKind.FACT, str(predicate.predicate_id), split).split
            for split in DataSplit
        ) == tuple(DataSplit)
    split_family_ids = tuple(
        {
            family.family_id
            for family in registry.families
            if family.split is split
        }
        for split in DataSplit
    )
    assert not split_family_ids[0] & split_family_ids[1]
    assert not split_family_ids[0] & split_family_ids[2]
    assert not split_family_ids[1] & split_family_ids[2]


def test_story_alignments_realize_their_exact_fact_and_rule_templates() -> None:
    bundle = generate_calibration_bundle("e" * 64)
    rendered = render_tinyworlds_bundle(
        bundle,
        _WhitespaceTokenizer(),
        _tiny_preset(query_groups=1),
    )
    family_by_id = {family.family_id: family for family in rendered.registry.families}
    fact_by_id = {str(fact.atom_id): fact for fact in bundle.facts}
    rule_by_id = {str(rule.rule_id): rule for rule in bundle.rules}
    entities = {entity.entity_id: entity for entity in bundle.entities}

    for story in rendered.stories:
        for alignment, family_id in zip(
            story.alignments,
            story.template_family_ids,
        ):
            family = family_by_id[family_id]
            realized = story.text[
                alignment.start_character : alignment.end_character
            ]
            if alignment.fact_ids:
                fact = fact_by_id[alignment.fact_ids[0]]
                assert family.kind is TemplateKind.FACT
                assert family.target_id == str(fact.predicate_id)
                assert realized == family.render(
                    rendering_module._fact_statement(fact, entities)
                )
            elif alignment.rule_ids:
                rule = rule_by_id[alignment.rule_ids[0]]
                assert family.kind is TemplateKind.RULE
                assert family.target_id == str(rule.rule_id)
                assert realized == family.render(
                    rendering_module._rule_statement(rule, entities)
                )


def test_renderer_enforces_counts_alignment_and_exact_query_boundaries() -> None:
    bundle = generate_calibration_bundle("1" * 64)
    tokenizer = _WhitespaceTokenizer()
    preset = _tiny_preset()

    rendered = render_tinyworlds_bundle(bundle, tokenizer, preset)

    assert len(rendered.stories) == len(bundle.tasks) * 4 + 2
    assert len(rendered.query_groups) == len(bundle.tasks) * 8
    assert all(len(story.token_ids) == 256 for story in rendered.stories)
    task_by_id = {str(task.task_id): task for task in bundle.tasks}
    for story in rendered.stories:
        if story.task_id is None:
            continue
        task = task_by_id[story.task_id]
        assert {
            fact_id for alignment in story.alignments for fact_id in alignment.fact_ids
        }.issubset({str(fact_id) for fact_id in task.direct_fact_ids})
        assert {
            rule_id for alignment in story.alignments for rule_id in alignment.rule_ids
        }.issubset({str(rule_id) for rule_id in task.rule_ids})

    for group in rendered.query_groups:
        assert tuple(len(variant.prefix_token_ids) for variant in group.variants) == (
            64,
            128,
            192,
        )
        assert group.variants[1].prefix_token_ids[-64:] == group.variants[0].prefix_token_ids
        assert group.variants[2].prefix_token_ids[-64:] == group.variants[0].prefix_token_ids
        for variant in group.variants:
            query = variant.knowledge_query
            candidate_token_counts = tuple(
                int(candidate.competence_batch.loss_mask.sum())
                for candidate in query.candidates
            )
            assert len(set(candidate_token_counts)) == 1
            assert candidate_token_counts[0] + query.prefix_length <= 256
            np.testing.assert_array_equal(
                query.router_batch.input_ids,
                query.candidates[0].competence_batch.input_ids[
                    :, : query.prefix_length - 1
                ],
            )
            if query.cue_regime == "cue_sufficient":
                assert query.eligible_task_ids == (query.task_id,)


def test_closed_book_candidates_do_not_leak_and_open_book_controls_are_present() -> None:
    bundle = generate_calibration_bundle("2" * 64)
    rendered = render_tinyworlds_bundle(bundle, _WhitespaceTokenizer(), _tiny_preset())
    entity_names = {str(entity.entity_id): entity.name for entity in bundle.entities}
    modes = set()

    for group in rendered.query_groups:
        for variant in group.variants:
            query = variant.knowledge_query
            modes.add(query.mode)
            if query.mode == "closed_book":
                folded_prefix = variant.prefix_text.casefold()
                assert all(
                    entity_names[entity_id].casefold() not in folded_prefix
                    for entity_id in variant.candidate_entity_ids
                )
    assert modes == {"closed_book", "open_book"}


def test_rendered_required_edges_use_memory_graph_child_node_ids() -> None:
    bundle = generate_calibration_bundle("a" * 64)
    rendered = render_tinyworlds_bundle(
        bundle,
        _WhitespaceTokenizer(),
        _tiny_preset(),
    )
    task_ids = {str(task.task_id) for task in bundle.tasks}

    assert all(
        {str(edge_id) for edge_id in variant.knowledge_query.required_edge_ids}
        <= task_ids
        for group in rendered.query_groups
        for variant in group.variants
    )
    cross_branch = tuple(
        variant.knowledge_query
        for group in rendered.query_groups
        for variant in group.variants
        if variant.knowledge_query.reasoning_type == "cross_branch"
    )
    assert cross_branch
    assert all(len(query.required_edge_ids) >= 2 for query in cross_branch)


def test_rendering_is_deterministic_and_full_preset_is_fixed() -> None:
    bundle = generate_calibration_bundle("3" * 64)
    tokenizer = _WhitespaceTokenizer()
    preset = _tiny_preset(query_groups=1)
    first = render_tinyworlds_bundle(bundle, tokenizer, preset)
    second = render_tinyworlds_bundle(bundle, tokenizer, preset)

    assert tuple(story.text_sha256 for story in first.stories) == tuple(
        story.text_sha256 for story in second.stories
    )
    assert tuple(
        variant.text_sha256
        for group in first.query_groups
        for variant in group.variants
    ) == tuple(
        variant.text_sha256
        for group in second.query_groups
        for variant in group.variants
    )
    assert TINYWORLDS_RENDER_PRESET == TinyWorldsRenderPreset(
        1_024,
        128,
        128,
        256,
        512,
        128,
        256,
        256,
    )


def test_query_group_plans_fix_every_instance_before_rendering() -> None:
    bundle = generate_calibration_bundle("b" * 64)
    tokenizer = _WhitespaceTokenizer()
    preset = _tiny_preset(query_groups=4)
    plans = build_query_group_plans(bundle, tokenizer, preset)
    full_plans = tuple(
        attempts[0]
        for attempts in expand_query_group_plan_attempts(
            bundle,
            TINYWORLDS_RENDER_PRESET,
        )
    )
    rendered = render_tinyworlds_bundle(bundle, tokenizer, preset)
    source_by_id = {
        str(plan.query_ast.query_id): plan for plan in bundle.query_plans
    }

    assert len(plans) == len(bundle.tasks) * 2 * 4
    assert len(full_plans) == len(bundle.tasks) * (256 + 512)
    assert all(
        sum(
            plan.task_id == task.task_id and plan.split is split
            for plan in full_plans
        )
        == expected
        for task in bundle.tasks
        for split, expected in (
            (DataSplit.VALIDATION, 256),
            (DataSplit.TEST, 512),
        )
    )
    assert len({plan.group_id for plan in plans}) == len(plans)
    assert len({plan.holdout_identity_sha256 for plan in plans}) == len(plans)
    assert tuple(plan.group_id for plan in plans) == tuple(
        group.group_id for group in rendered.query_groups
    )
    for task in bundle.tasks:
        for split in (DataSplit.VALIDATION, DataSplit.TEST):
            sliced = tuple(
                plan
                for plan in plans
                if plan.task_id == task.task_id and plan.split is split
            )
            sources = tuple(
                plan
                for plan in bundle.query_plans
                if plan.task_id == task.task_id and plan.split is split
            )
            assert tuple(plan.occurrence_index for plan in sliced) == tuple(range(4))
            assert all(
                plan.replication_index
                == plan.occurrence_index // len(sources)
                for plan in sliced
            )
    for group_plan, group in zip(plans, rendered.query_groups):
        source = source_by_id[group_plan.source_query_id]
        assert group.symbolic_query_id == group_plan.source_query_id
        assert group.variants[0].knowledge_query.proof_id == group_plan.source_proof_id
        assert tuple(
            str(candidate.entity_id) for candidate in group_plan.candidates
        ) == group.variants[0].candidate_entity_ids
        assert source == group_plan.source_plan
        assert (
            group.variants[0].knowledge_query.correct_candidate_index
            == group_plan.correct_index
        )


def test_expected_render_rejection_advances_to_the_next_fixed_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = generate_calibration_bundle("f" * 64)
    tokenizer = _WhitespaceTokenizer()
    registry = build_template_registry(bundle.world)
    task = bundle.tasks[0]
    sources = tuple(
        plan
        for plan in bundle.query_plans
        if plan.task_id == task.task_id and plan.split is DataSplit.VALIDATION
    )
    assert len(sources) > 1
    original = rendering_module._render_query_group
    attempted: list[str] = []
    rejected_query_id = str(sources[0].query_ast.query_id)

    def reject_first(bundle_value, registry_value, tokenizer_value, group_plan):
        attempted.append(group_plan.source_query_id)
        if group_plan.source_query_id == rejected_query_id:
            raise TinyWorldsRenderingRejection("deterministic fixture rejection")
        return original(bundle_value, registry_value, tokenizer_value, group_plan)

    monkeypatch.setattr(rendering_module, "_render_query_group", reject_first)
    preset = _tiny_preset(query_groups=1)
    accepted = build_query_group_plans(
        bundle,
        tokenizer,
        preset,
        registry=registry,
    )
    rendered = render_tinyworlds_bundle(bundle, tokenizer, preset)
    group = next(
        item
        for item in rendered.query_groups
        if item.task_id == str(task.task_id)
        and item.split is DataSplit.VALIDATION
    )
    selected = next(
        item
        for item in accepted
        if item.task_id == task.task_id and item.split is DataSplit.VALIDATION
    )
    replayed = rendering_module._render_query_group_at_index(
        bundle,
        registry,
        tokenizer,
        sources,
        0,
    )

    assert rejected_query_id in attempted
    assert selected.source_query_id == str(sources[1].query_ast.query_id)
    assert group.group_plan == selected
    assert replayed.group_plan == selected
    assert group.symbolic_query_id == selected.source_query_id


def test_best_first_exact_padding_has_a_bounded_tokenizer_call_count() -> None:
    class CountingTokenizer:
        def __init__(self) -> None:
            self.calls = 0

        def encode(self, text: str, *, add_eos: bool = False) -> tuple[int, ...]:
            self.calls += 1
            return _WhitespaceTokenizer().encode(text, add_eos=add_eos)

    tokenizer = CountingTokenizer()
    text, token_ids = rendering_module._fit_exact_tokens(
        "A small story.",
        256,
        tokenizer,
        salt="best-first-call-bound",
        append=True,
    )

    assert text.startswith("A small story.")
    assert len(token_ids) == 256
    assert tokenizer.calls < 700


def test_query_candidate_positions_balance_each_task_split_and_kind() -> None:
    bundle = generate_calibration_bundle("c" * 64)
    rendered = render_tinyworlds_bundle(
        bundle,
        _WhitespaceTokenizer(),
        _tiny_preset(query_groups=16),
    )
    indices_by_slice: dict[tuple[str, DataSplit, str], list[int]] = {}
    revision_cue_indices: dict[tuple[DataSplit, int], list[int]] = {}
    answer_by_query_id = {
        str(plan.query_ast.query_id): str(plan.answer_entity_id)
        for plan in bundle.query_plans
    }

    for group in rendered.query_groups:
        queries = tuple(variant.knowledge_query for variant in group.variants)
        assert len({query.correct_candidate_index for query in queries}) == 1
        assert (
            group.variants[0].candidate_entity_ids[
                queries[0].correct_candidate_index
            ]
            == answer_by_query_id[group.symbolic_query_id]
        )
        key = (group.task_id, group.split, queries[0].query_kind)
        indices_by_slice.setdefault(key, []).append(
            queries[0].correct_candidate_index
        )
        for query in queries:
            if (
                group.task_id == "calibration_revision"
                and query.cue_regime == "cue_sufficient"
            ):
                revision_cue_indices.setdefault(
                    (group.split, query.prefix_length),
                    [],
                ).append(query.correct_candidate_index)

    assert indices_by_slice
    assert all(
        Counter(indices) == Counter({index: len(indices) // 4 for index in range(4)})
        for indices in indices_by_slice.values()
    )
    assert revision_cue_indices
    assert all(
        Counter(indices) == Counter({index: len(indices) // 4 for index in range(4)})
        for indices in revision_cue_indices.values()
    )


def test_rendered_data_adapter_uses_exact_fact_exposures_and_semantic_probes() -> None:
    bundle = generate_calibration_bundle("4" * 64)
    tokenizer = _WhitespaceTokenizer()
    rendered = render_tinyworlds_bundle(
        bundle,
        tokenizer,
        TinyWorldsRenderPreset(32, 2, 2, 2, 2, 2, 256, 256),
    )
    prepared = prepare_tinyworlds_curriculum(
        rendered,
        tokenizer,
        TinyWorldsTrainingDataConfig(
            facts_per_task=4,
            exposures_per_fact=2,
            batch_size=2,
            context_length=256,
            evaluation_examples_per_task=2,
        ),
    )

    assert len(prepared.training_story_ids) == len(bundle.tasks) * 8
    assert len(prepared.validation_queries) == len(bundle.tasks) * 2 * 3
    assert len(prepared.test_queries) == len(bundle.tasks) * 2 * 3
    assert all(
        len(task.parent_probes) == 2 and len(task.content_key_probes) == 2
        for task in prepared.language.curriculum.tasks
    )
    assert all(
        probe.input_ids.shape == (1, 63)
        for task in prepared.language.curriculum.tasks
        for probe in task.parent_probes
    )


def test_training_fact_prefix_is_independent_of_validation_support_and_order() -> None:
    bundle = generate_calibration_bundle("a" * 64)
    tokenizer = _WhitespaceTokenizer()
    rendered = render_tinyworlds_bundle(
        bundle,
        tokenizer,
        TinyWorldsRenderPreset(32, 2, 2, 4, 4, 2, 256, 256),
    )
    config = TinyWorldsTrainingDataConfig(
        facts_per_task=4,
        exposures_per_fact=2,
        batch_size=2,
        context_length=256,
        evaluation_examples_per_task=2,
    )
    original = prepare_tinyworlds_curriculum(rendered, tokenizer, config)
    final_observed_fact_id = next(
        fact_id
        for story in reversed(rendered.stories)
        if story.split is DataSplit.TRAIN
        for alignment in reversed(story.alignments)
        for fact_id in reversed(alignment.fact_ids)
    )
    altered_groups = tuple(
        replace(
            group,
            variants=tuple(
                replace(
                    variant,
                    knowledge_query=replace(
                        variant.knowledge_query,
                        support_ids=(final_observed_fact_id,),
                    ),
                )
                for variant in group.variants
            ),
        )
        if group.split is DataSplit.VALIDATION
        else group
        for group in reversed(rendered.query_groups)
    )
    altered = prepare_tinyworlds_curriculum(
        replace(rendered, query_groups=altered_groups),
        tokenizer,
        config,
    )

    assert altered.training_story_ids == original.training_story_ids

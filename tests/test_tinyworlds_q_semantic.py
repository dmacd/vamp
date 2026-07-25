from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
import shutil
import string
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from apm.data.text.tinyworlds_q_semantic import evaluation as semantic_evaluation
from apm.continual.language_baseline_training import IndependentRootAdapter
from apm.continual.language_tasks import RouterBatch
from apm.data.text.tinyworlds_q_semantic.adaptation import (
    PreparedQueryAdaptation,
    QueryTaskProbeSet,
    materialize_query_language_task,
    prepare_query_adaptation,
)
from apm.data.text.tinyworlds_q_semantic.approval import (
    approve_all_primary_proposals,
    load_primary_review_approval,
    publish_primary_review_approval,
)
from apm.data.text.tinyworlds_p.normalization import normalized_story_bytes_sha256
from apm.data.text.tinyworlds_p.contracts import (
    CANONICAL_ARCHIVE_IDENTITY,
    CANONICAL_TOKENIZER_IDENTITY,
)
from apm.data.text.tinyworlds_q_semantic.catalog import (
    build_reviewed_catalog,
    load_sealed_catalog,
    load_validation_catalog,
    make_query_template,
    publish_catalog,
    publish_opened_sealed_audit,
)
from apm.data.text.tinyworlds_q_semantic.batching import (
    count_query_partition_microbatches,
    iter_query_partition_batches,
)
from apm.data.text.tinyworlds_q_semantic.contracts import (
    ConceptDefinition,
    FactReviewDecision,
    QueryExperimentPreset,
    QueryPartitionArtifact,
    SemanticFact,
    SemanticQueryCatalog,
    SemanticQueryResult,
    validate_parent_catalog_prefix,
)
from apm.data.text.tinyworlds_q_semantic.curriculum import (
    capacity_masks,
    concept_stages,
    progress_totals,
    validate_active_catalog_prefix,
)
from apm.data.text.tinyworlds_q_semantic.execution import (
    AMENDED_PILOT_LEARNABILITY_POLICY,
    ORIGINAL_PILOT_LEARNABILITY_POLICY,
    BaseQualityDecision,
    PilotBudgetResult,
    begin_sealed_test,
    complete_sealed_test,
    latest_stage_artifact,
    load_stage_artifact,
    publish_sealed_transaction,
    publish_stage_artifact,
    pilot_budget_passes,
    select_pilot_budget,
)
from apm.data.text.tinyworlds_q_semantic.evaluation import PilotBudgetEvaluation
from apm.data.text.tinyworlds_q_semantic.queries import project_semantic_result
from apm.data.text.tinyworlds_q_semantic.main_freeze import (
    load_main_experiment_freeze,
    publish_main_experiment_freeze,
)
from apm.data.text.tinyworlds_q_semantic.main_shortlist import (
    MAIN_SHORTLIST_SPECS,
    build_main_review_shortlist,
)
from apm.data.text.tinyworlds_q_semantic.manifests import (
    MAIN_CONCEPTS,
    PILOT_CONCEPTS,
)
from apm.data.text.tinyworlds_q_semantic.partition import (
    assign_story_group,
    build_query_partition,
    load_query_partition,
    tree_sha256,
)
from apm.data.text.tinyworlds_q_semantic.pilot_catalog import (
    build_approved_pilot_catalog,
)
from apm.data.text.tinyworlds_q_semantic.pilot import (
    load_semantic_pilot_failure,
    load_semantic_pilot_result,
    publish_semantic_pilot_failure,
    publish_semantic_pilot_result,
)
from apm.data.text.tinyworlds_q_semantic.pilot_authorization import (
    load_semantic_pilot_protocol_amendment,
    publish_semantic_pilot_protocol_amendment,
)
from apm.data.text.tinyworlds_q_semantic.pilot_sweep import (
    load_pilot_independent_sweep,
    publish_pilot_independent_sweep_stage,
)
from apm.data.text.tinyworlds_q_semantic.queries import (
    compile_semantic_queries,
    stack_semantic_router_batches,
    validation_question_prefixes,
)
from apm.data.text.tinyworlds_q_semantic.query_protocol import (
    REGISTERED_QUERY_PROTOCOL,
)
from apm.data.text.tinyworlds_q_semantic.report import (
    REQUIRED_QUERY_METHODS,
    publish_semantic_report,
    render_semantic_report,
)
from apm.data.text.tinyworlds_q_semantic.review import (
    PredicateDefinition,
    SemanticReviewPacket,
    build_review_packet,
    discover_review_packet,
    is_construction_group,
    load_review_packet,
    publish_review_packet,
)
from apm.data.text.tinyworlds_q_semantic.reverse_review import (
    approve_all_reverse_choices,
    build_pilot_reverse_review,
    load_reverse_review_approval,
    publish_reverse_review,
    publish_reverse_review_approval,
)
from apm.data.text.tinyworlds_q_semantic.scaling import (
    PreflightMeasurement,
    estimate_resources,
    evaluation_schedule,
    iter_chunks,
    render_schedule_report,
    require_preflight_capacity,
    score_in_chunks,
    write_atomic_jsonl,
)
from apm.data.text.tinyworlds_q_semantic.shortlist import (
    PILOT_SHORTLIST_SPECS,
    build_pilot_review_shortlist,
    pilot_shortlist_predicates,
    publish_review_shortlist,
    shortlist_predicates,
)
from apm.data.text.tinyworlds_q_semantic.source import (
    QueryStoryGroup,
    QueryStoryOccurrence,
)
from apm.data.text.tinyworlds_q_semantic.statistics import (
    acquisition_effect,
    average_paraphrases,
    bootstrap_fact_metric,
    generation_prompts,
    inspect_generation,
    specificity_effect,
)
from apm.data.text.tinyworlds_q_semantic.selected_base import (
    QueryBaseEpochEvidence,
    QuerySelectedBase,
)
from apm.data.text.tinyworlds_q_semantic.training import QuerySplitNll
from apm.lm.checkpoint import BaseCheckpointRef
from apm.lm.lora import init_lora_edge
from apm.memory.graph import TaskId
from apm.data.text.tinyworlds_q_semantic.training import (
    QueryBaseTrainingConfig,
    run_query_base_training,
)
from apm.lm.config import GptNeoConfig
from apm.lm.text import CharTokenizer
from apm.lm.training_state_artifact import lm_train_state_checksum


_TOKENIZER = CharTokenizer.from_training_text(string.printable)


class _OneTokenPerWordTokenizer:
    vocab_size = 50_257

    def encode(self, text: str, *, add_eos: bool = False) -> tuple[int, ...]:
        tokens = tuple(range(1, len(text.split()) + 1))
        return tokens + ((50_256,) if add_eos else ())


_CONCEPTS = tuple(
    ConceptDefinition(concept_id, (concept_id, f"{concept_id}s"))
    for concept_id in ("cat", "dog", "pig")
)
_PREDICATES = tuple(
    PredicateDefinition(
        f"trait{string.ascii_lowercase[index // 26]}{string.ascii_lowercase[index % 26]}",
        f"relation-{index % 4}",
    )
    for index in range(12)
)


@dataclass(frozen=True)
class _Fixture:
    packet: SemanticReviewPacket
    catalog: SemanticQueryCatalog
    groups: tuple[QueryStoryGroup, ...]


def _group(text: str, ordinal: int) -> QueryStoryGroup:
    story = text.encode("utf-8")
    group_sha256 = normalized_story_bytes_sha256(story)
    story_sha256 = sha256(story).hexdigest()
    occurrence = QueryStoryOccurrence(
        record_id=f"fixture:{ordinal}:{story_sha256}",
        source_member="fixture.json",
        source_index=ordinal,
        content_sha256=story_sha256,
        story_sha256=story_sha256,
        source="fixture",
        story_bytes=story,
        token_ids=_TOKENIZER.encode(text),
    )
    return QueryStoryGroup(group_sha256, (occurrence,))


def _find_group(
    prefix: str,
    ordinal: int,
    *,
    construction: bool,
) -> QueryStoryGroup:
    for attempt in range(20_000):
        candidate = _group(f"{prefix} Marker {ordinal} attempt {attempt}.", ordinal)
        if is_construction_group(candidate.normalized_story_sha256) == construction:
            return candidate
    raise AssertionError("could not find requested construction bucket")


def _build_fixture() -> _Fixture:
    predicate_text = ", ".join(item.predicate for item in _PREDICATES)
    construction_groups = tuple(
        _find_group(
            f"A {concept.concept_id} shows {predicate_text}.",
            concept_index * 1_000 + evidence_index,
            construction=True,
        )
        for concept_index, concept in enumerate(_CONCEPTS)
        for evidence_index in range(16)
    )
    packet = build_review_packet(
        tuple(sorted(construction_groups, key=lambda item: item.normalized_story_sha256)),
        _CONCEPTS,
        _PREDICATES,
    )
    candidate_by_key = {
        (candidate.concept_id, candidate.predicate): candidate
        for candidate in packet.candidates
    }
    facts = tuple(
        SemanticFact(
            fact_id=f"{concept.concept_id}-fact-{fact_index:02d}",
            source_candidate_id=candidate_by_key[
                (concept.concept_id, predicate.predicate)
            ].candidate_id,
            concept_id=concept.concept_id,
            relation_category=predicate.relation_category,
            answer_type="property",
            canonical_answer=f"a{string.ascii_lowercase[fact_index]}",
            accepted_forms=(f"a{string.ascii_lowercase[fact_index]}",),
            trigger_forms=(predicate.predicate,),
            supporting_story_groups=candidate_by_key[
                (concept.concept_id, predicate.predicate)
            ].supporting_story_groups,
            evidence=candidate_by_key[
                (concept.concept_id, predicate.predicate)
            ].evidence,
        )
        for concept in _CONCEPTS
        for fact_index, predicate in enumerate(_PREDICATES)
    )
    reverse_distractors = {
        "cat": ("dog", "pig", "fox"),
        "dog": ("cat", "pig", "fox"),
        "pig": ("cat", "dog", "fox"),
    }
    templates = tuple(
        make_query_template(
            fact,
            concept,
            template_id=(
                f"{fact.fact_id}-{split}-{paraphrase_index}"
            ),
            direction=direction,
            prompt_text=(
                f"Which property belongs to {concept.concept_id} in wording "
                f"{split} {paraphrase_index}? Answer:"
                if direction == "forward"
                else f"Which concept matches {fact.trigger_forms[0]} in wording "
                f"{split} {paraphrase_index}? Answer:"
            ),
            distractors=(
                (
                    f"b{fact.canonical_answer[1]}",
                    f"c{fact.canonical_answer[1]}",
                    f"d{fact.canonical_answer[1]}",
                )
                if direction == "forward"
                else reverse_distractors[concept.concept_id]
            ),
            split=split,
            paraphrase_index=paraphrase_index,
            tokenizer=_TOKENIZER,
        )
        for concept in _CONCEPTS
        for fact in facts
        if fact.concept_id == concept.concept_id
        for split, directions in (
            ("validation", ("forward", "forward", "reverse")),
            ("test", ("forward", "forward", "forward", "reverse", "reverse")),
        )
        for paraphrase_index, direction in enumerate(directions)
    )
    reviews = tuple(
        FactReviewDecision(
            fact_id=fact.fact_id,
            reviewer="fixture-reviewer",
            reviewed_at="2026-07-24T00:00:00Z",
            truth_approved=True,
            answer_forms_approved=True,
            trigger_closure_approved=True,
            distractors_approved=True,
            evidence_approved=True,
        )
        for fact in facts
    )
    catalog = build_reviewed_catalog(
        review_packet=packet,
        facts=facts,
        templates=templates,
        reviews=reviews,
        rejected_candidates=(),
    )
    generated: list[QueryStoryGroup] = list(construction_groups)
    next_ordinal = 100_000
    for concept in _CONCEPTS:
        node_counts = {"train": 0, "validation": 0}
        while node_counts["train"] < 36 or node_counts["validation"] < 3:
            group = _find_group(
                f"A {concept.concept_id} shows {predicate_text}.",
                next_ordinal,
                construction=False,
            )
            next_ordinal += 1
            assignment = assign_story_group(group, catalog.concepts, catalog.facts)
            if assignment.role == "node" and node_counts[assignment.split] < (
                36 if assignment.split == "train" else 3
            ):
                generated.append(group)
                node_counts[assignment.split] += 1
        base_counts = {"train": 0, "validation": 0, "test": 0}
        while (
            base_counts["train"] < 260
            or base_counts["validation"] < 2
            or base_counts["test"] < 2
        ):
            group = _find_group(
                f"A {concept.concept_id} visits a quiet garden.",
                next_ordinal,
                construction=False,
            )
            next_ordinal += 1
            assignment = assign_story_group(group, catalog.concepts, catalog.facts)
            limit = 260 if assignment.split == "train" else 2
            if assignment.role == "base" and base_counts[assignment.split] < limit:
                generated.append(group)
                base_counts[assignment.split] += 1
    multi_concept = _find_group(
        f"A cat and dog show {_PREDICATES[0].predicate}.",
        next_ordinal,
        construction=False,
    )
    generated.append(multi_concept)
    return _Fixture(
        packet=packet,
        catalog=catalog,
        groups=tuple(sorted(generated, key=lambda item: item.normalized_story_sha256)),
    )


@pytest.fixture(scope="module")
def semantic_fixture() -> _Fixture:
    return _build_fixture()


def test_review_catalog_sealing_and_query_compilation(
    tmp_path: Path,
    semantic_fixture: _Fixture,
) -> None:
    review_root = publish_review_packet(semantic_fixture.packet, tmp_path)
    assert (review_root / "review.html").read_text().startswith("<!doctype html>")
    assert load_review_packet(review_root) == semantic_fixture.packet
    tampered_review = tmp_path / "tampered-review" / review_root.name
    shutil.copytree(review_root, tampered_review)
    with (tampered_review / "review.md").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="review packet file changed"):
        load_review_packet(tampered_review)
    catalog_root = publish_catalog(semantic_fixture.catalog, tmp_path)
    assert not (catalog_root / "sealed-test-audit.md").exists()
    validation = load_validation_catalog(catalog_root)
    assert len(validation.templates) == 3 * 12 * 3
    assert len(validation_question_prefixes(validation)) == len(validation.templates)
    compiled = compile_semantic_queries(validation, _TOKENIZER)
    assert len(compiled) == len(validation.templates)
    stacked_prefixes = stack_semantic_router_batches(
        compiled[:7],
        _TOKENIZER.pad_token_id,
    )
    assert stacked_prefixes.input_ids.shape[0] == 7
    assert stacked_prefixes.input_ids.shape == stacked_prefixes.attention_mask.shape
    assert tuple(stacked_prefixes.attention_mask.sum(axis=1)) == tuple(
        query.knowledge_query.router_batch.attention_mask.sum()
        for query in compiled[:7]
    )
    assert all(
        candidate.competence_batch.loss_mask.sum()
        == compiled_query.knowledge_query.candidates[0].competence_batch.loss_mask.sum()
        for compiled_query in compiled
        for candidate in compiled_query.knowledge_query.candidates
    )
    with pytest.raises((TypeError, PermissionError)):
        load_sealed_catalog(catalog_root, object())  # type: ignore[arg-type]
    transaction = publish_sealed_transaction(
        tmp_path / "transaction",
        catalog_sha256=semantic_fixture.catalog.catalog_sha256,
        partition_sha256="1" * 64,
        selected_base_sha256="2" * 64,
        adapters_sha256="3" * 64,
        config_sha256="4" * 64,
    )
    begin_sealed_test(transaction)
    sealed = load_sealed_catalog(catalog_root, transaction)
    sealed_audits = publish_opened_sealed_audit(
        sealed,
        transaction,
        tmp_path / "transaction",
    )
    assert all(path.is_file() for path in sealed_audits)
    assert sealed == semantic_fixture.catalog
    assert len(compile_semantic_queries(sealed, _TOKENIZER, split="test")) == 5 * 12 * 3
    answer_positions = {
        fact.fact_id: sorted(
            template.correct_candidate_index
            for template in sealed.templates
            if template.fact_id == fact.fact_id
        )
        for fact in sealed.facts
    }
    assert set(map(tuple, answer_positions.values())) == {
        (0, 0, 1, 1, 2, 2, 3, 3)
    }
    complete_sealed_test(transaction, "5" * 64)
    with pytest.raises(PermissionError, match="not authorized"):
        load_sealed_catalog(catalog_root, transaction)
    with pytest.raises(RuntimeError, match="already complete"):
        begin_sealed_test(transaction)
    discovered = discover_review_packet(
        semantic_fixture.groups,
        semantic_fixture.catalog.concepts,
        minimum_group_support=16,
        maximum_candidates_per_concept=5,
    )
    assert discovered.candidates
    assert all(
        candidate.relation_category == "unclassified"
        for candidate in discovered.candidates
    )
    tampered_catalog = tmp_path / "tampered-catalog" / catalog_root.name
    shutil.copytree(catalog_root, tampered_catalog)
    with (tampered_catalog / "validation-queries.json").open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="catalog file changed"):
        load_validation_catalog(tampered_catalog)


def test_pilot_review_shortlist_is_compact_supported_and_publishable(
    tmp_path: Path,
) -> None:
    predicates = pilot_shortlist_predicates()
    predicate_text = ", ".join(predicate.predicate for predicate in predicates)
    groups = tuple(
        _find_group(
            f"A {concept.concept_id} example {evidence_index} shows {predicate_text}.",
            800_000 + concept_index * 100 + evidence_index,
            construction=True,
        )
        for concept_index, concept in enumerate(PILOT_CONCEPTS)
        for evidence_index in range(16)
    )
    packet = build_review_packet(
        tuple(sorted(groups, key=lambda item: item.normalized_story_sha256)),
        PILOT_CONCEPTS,
        predicates,
    )
    shortlist = build_pilot_review_shortlist(
        packet,
        _OneTokenPerWordTokenizer(),  # type: ignore[arg-type]
    )
    assert tuple(proposal.spec for proposal in shortlist.proposals) == PILOT_SHORTLIST_SPECS
    assert all(proposal.supporting_group_count == 16 for proposal in shortlist.proposals)
    assert all(
        len(set(map(len, proposal.answer_token_ids))) == 1
        for proposal in shortlist.proposals
    )
    assert {
        concept.concept_id: sum(
            proposal.spec.concept_id == concept.concept_id
            and proposal.spec.priority == "primary"
            for proposal in shortlist.proposals
        )
        for concept in PILOT_CONCEPTS
    } == {"rabbit": 12, "horse": 12}
    root = publish_review_shortlist(shortlist, tmp_path)
    review = (root / "review.md").read_text(encoding="utf-8")
    assert review.count("| [ ] | `") == 24
    assert "approve all primaries" in review
    assert (root / "shortlist.json").is_file()
    assert (root / "review-form.tsv").read_text(encoding="utf-8").count("\n") == 33
    approval = approve_all_primary_proposals(
        shortlist,
        reviewer="fixture-reviewer",
        reviewed_at="2026-07-25T00:00:00Z",
    )
    approval_root = publish_primary_review_approval(approval, tmp_path)
    assert load_primary_review_approval(approval_root) == approval
    reverse_review = build_pilot_reverse_review(
        shortlist,
        approval,
        _OneTokenPerWordTokenizer(),  # type: ignore[arg-type]
    )
    reverse_root = publish_reverse_review(reverse_review, tmp_path)
    reverse_markdown = (reverse_root / "review.md").read_text(encoding="utf-8")
    assert reverse_markdown.count("| [ ] | `") == 24
    assert "approve all reverse choices" in reverse_markdown
    reverse_approval = approve_all_reverse_choices(
        reverse_review,
        reviewer="fixture-reviewer",
        reviewed_at="2026-07-25T00:01:00Z",
    )
    reverse_approval_root = publish_reverse_review_approval(
        reverse_approval,
        tmp_path,
    )
    assert load_reverse_review_approval(reverse_approval_root) == reverse_approval
    catalog = build_approved_pilot_catalog(
        review_packet=packet,
        shortlist=shortlist,
        primary_approval=approval,
        reverse_review=reverse_review,
        reverse_approval=reverse_approval,
        tokenizer=_OneTokenPerWordTokenizer(),  # type: ignore[arg-type]
    )
    assert len(catalog.facts) == 24
    assert len(catalog.templates) == 24 * 8
    assert len(catalog.rejected_candidates) == 8


def test_main_review_shortlist_uses_the_dynamic_five_world_manifest() -> None:
    predicates = shortlist_predicates(MAIN_SHORTLIST_SPECS)
    predicate_text = ", ".join(predicate.predicate for predicate in predicates)
    groups = tuple(
        _find_group(
            f"A {concept.concept_id} example {evidence_index} shows {predicate_text}.",
            900_000 + concept_index * 100 + evidence_index,
            construction=True,
        )
        for concept_index, concept in enumerate(MAIN_CONCEPTS)
        for evidence_index in range(16)
    )
    packet = build_review_packet(
        tuple(sorted(groups, key=lambda item: item.normalized_story_sha256)),
        MAIN_CONCEPTS,
        predicates,
    )
    shortlist = build_main_review_shortlist(
        packet,
        _OneTokenPerWordTokenizer(),  # type: ignore[arg-type]
    )
    assert tuple(proposal.spec for proposal in shortlist.proposals) == (
        MAIN_SHORTLIST_SPECS
    )
    assert tuple(item.concept_id for item in shortlist.reverse_choices) == tuple(
        concept.concept_id for concept in MAIN_CONCEPTS
    )
    assert len(shortlist.proposals) == 5 * 16
    assert sum(
        proposal.spec.priority == "primary" for proposal in shortlist.proposals
    ) == 5 * 12


def test_partition_fact_withholding_rebuild_and_tampering(
    tmp_path: Path,
    semantic_fixture: _Fixture,
) -> None:
    first = build_query_partition(
        semantic_fixture.groups,
        semantic_fixture.catalog,
        tmp_path / "first",
        pad_token_id=50_256,
        eos_token_id=50_256,
    )
    second = build_query_partition(
        semantic_fixture.groups,
        semantic_fixture.catalog,
        tmp_path / "second",
        pad_token_id=50_256,
        eos_token_id=50_256,
    )
    assert first.partition_sha256 == second.partition_sha256
    assert tree_sha256(first.root) == tree_sha256(second.root)
    assert load_query_partition(first.root, semantic_fixture.catalog) == first
    batching_preset = QueryExperimentPreset(("cat", "dog", "pig"), adapter_updates=500)
    base_batch_count = count_query_partition_microbatches(
        first,
        batching_preset,
        role="base",
        split="train",
    )
    assert base_batch_count > 0
    first_batch = next(
        iter_query_partition_batches(
            first,
            batching_preset,
            role="node",
            concept_id="cat",
            split="train",
            epoch=0,
        )
    )
    assert first_batch.input_ids.shape == (32, 256)
    catalog_root = publish_catalog(semantic_fixture.catalog, tmp_path / "training")
    validation = load_validation_catalog(catalog_root)
    prepared = prepare_query_adaptation(
        validation,
        first,
        _TOKENIZER,
        batching_preset,
    )
    assert len(prepared.root_validation_probes) == 36
    assert tuple(item.concept_id for item in prepared.task_probes) == (
        "cat",
        "dog",
        "pig",
    )
    materialized = materialize_query_language_task(
        prepared,
        first,
        batching_preset,
        "cat",
        maximum_batches=2,
    )
    assert materialized.materialized_batch_count == 2
    assert materialized.task.test_examples == ()

    tiny_training = QueryBaseTrainingConfig(
        model_config=GptNeoConfig(
            vocab_size=first.tokenizer_identity.vocab_size,
            max_position_embeddings=8,
            hidden_size=8,
            intermediate_size=16,
            num_layers=1,
            num_heads=1,
            attention_types=("global",),
            local_window_size=8,
        ),
        epochs=2,
        context_length=8,
        microbatch_size=1,
        accumulation_microbatches=1,
        maximum_learning_rate=5e-4,
        minimum_learning_rate=5e-5,
        warmup_fraction=0.01,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        weight_decay=0.1,
        gradient_clip_norm=1.0,
        parameter_seed=0,
        state_interval_updates=1,
        allocator_peak_limit_bytes=12 * 1024**3,
    )
    uninterrupted = run_query_base_training(
        first,
        batching_preset,
        tmp_path / "base-uninterrupted",
        tiny_training,
        stop_after_update=2,
    )
    interrupted = run_query_base_training(
        first,
        batching_preset,
        tmp_path / "base-resumed",
        tiny_training,
        stop_after_update=1,
    )
    resumed = run_query_base_training(
        first,
        batching_preset,
        tmp_path / "base-resumed",
        tiny_training,
        resume_from=interrupted.checkpoints[-1].directory,
        stop_after_update=2,
    )
    assert lm_train_state_checksum(uninterrupted.state) == lm_train_state_checksum(
        resumed.state
    )
    assert uninterrupted.trace_path.read_bytes() == resumed.trace_path.read_bytes()
    assignments = tuple(
        line.decode("utf-8") for line in (first.root / "assignments.jsonl").read_bytes().splitlines()
    )
    assert sum('"role":"construction"' in line for line in assignments) == 48
    assert any('"exclusion_reason":"multi-concept-fact-bearing"' in line for line in assignments)
    assert not any(
        '"role":"base"' in line and '"triggered_fact_ids":[' in line and '"triggered_fact_ids":[]' not in line
        for line in assignments
    )
    separated = _find_group(
        f"A cat waits. Someone shows {_PREDICATES[0].predicate}.",
        999_001,
        construction=False,
    )
    separated_assignment = assign_story_group(
        separated,
        semantic_fixture.catalog.concepts,
        semantic_fixture.catalog.facts,
    )
    assert separated_assignment.role == "node"
    assert separated_assignment.authoritative_fact_ids == ()
    copied = tmp_path / "tampered" / first.partition_sha256
    shutil.copytree(first.root, copied)
    with (copied / "assignments.jsonl").open("ab") as stream:
        stream.write(b"{}\n")
    with pytest.raises(ValueError, match="file changed"):
        load_query_partition(copied, semantic_fixture.catalog)


def test_nested_catalog_and_dynamic_one_to_one_hundred_world_schedules(
    semantic_fixture: _Fixture,
) -> None:
    assert tuple(concept.surface_forms for concept in PILOT_CONCEPTS) == (
        ("rabbit", "rabbits", "bunny", "bunnies"),
        ("horse", "horses", "pony", "ponies"),
    )
    assert tuple(concept.concept_id for concept in MAIN_CONCEPTS) == (
        "cat",
        "dog",
        "bird",
        "robot",
        "dragon",
    )
    parent_concept = semantic_fixture.catalog.concepts[:1]
    parent_fact_ids = {
        fact.fact_id for fact in semantic_fixture.catalog.facts if fact.concept_id == "cat"
    }
    parent = SemanticQueryCatalog(
        concepts=parent_concept,
        facts=tuple(
            fact for fact in semantic_fixture.catalog.facts if fact.fact_id in parent_fact_ids
        ),
        templates=tuple(
            template
            for template in semantic_fixture.catalog.templates
            if template.fact_id in parent_fact_ids
        ),
        reviews=tuple(
            review
            for review in semantic_fixture.catalog.reviews
            if review.fact_id in parent_fact_ids
        ),
        rejected_candidates=(),
        review_packet_sha256=semantic_fixture.catalog.review_packet_sha256,
    )
    child = replace(
        semantic_fixture.catalog,
        parent_catalog_sha256=parent.catalog_sha256,
    )
    validate_parent_catalog_prefix(child, parent)
    reordered = replace(
        child,
        templates=(
            *child.templates[8:16],
            *child.templates[:8],
            *child.templates[16:],
        ),
    )
    with pytest.raises(ValueError, match="parent template"):
        validate_parent_catalog_prefix(reordered, parent)

    for world_count in (1, 5, 10, 20):
        preset = QueryExperimentPreset(
            tuple(f"world-{index:03d}" for index in range(world_count)),
            adapter_updates=500,
        )
        assert (preset.max_nodes, preset.max_edges) == (world_count + 1, world_count)
        assert len(evaluation_schedule(preset)) == world_count * (world_count + 1) // 2
        assert len(concept_stages(preset)) == world_count
        masks = capacity_masks(preset, world_count)
        assert masks.node_mask.tolist() == [True] * (world_count + 1)
        assert masks.edge_mask.tolist() == [True] * world_count
    hundred = QueryExperimentPreset(
        tuple(f"world-{index:03d}" for index in range(100)),
        evaluation_schedule="milestone",
        evaluation_milestones=(10, 20, 50),
        adapter_updates=500,
    )
    preset_record = hundred.as_record()
    assert preset_record["model_config"]["num_layers"] == 8
    assert preset_record["lora_config"]["target_mask"] == {
        "attention_output": True,
        "key": True,
        "mlp_input": True,
        "mlp_output": True,
        "query": True,
        "value": True,
    }
    assert preset_record["adapter_train_config"]["steps"] == 500
    assert (hundred.max_nodes, hundred.max_edges) == (101, 100)
    assert progress_totals(hundred, queries_per_world=60, method_count=9)[
        "adapter_updates"
    ] == 3 * 100 * 500
    cells = evaluation_schedule(hundred)
    assert len(tuple(cell for cell in cells if cell.acquisition)) == 100
    assert len(tuple(cell for cell in cells if cell.final)) == 100
    assert max(len(chunk) for chunk in iter_chunks(range(103), 17)) == 17
    assert tuple(score_in_chunks(range(9), 4, lambda chunk: (sum(chunk),))) == (
        6,
        22,
        8,
    )
    assert "world-099" in render_schedule_report(hundred)
    estimate = estimate_resources(hundred)
    measurement = PreflightMeasurement(0.1, 0.01, 0.001, 10 * 1024**3, 1)
    require_preflight_capacity(hundred, estimate, measurement)
    with pytest.raises(MemoryError, match="allocator peak"):
        require_preflight_capacity(
            hundred,
            estimate,
            replace(measurement, allocator_peak_bytes=13 * 1024**3),
        )
    prefix_preset = QueryExperimentPreset(("cat",), adapter_updates=500)
    assert prefix_preset.config_sha256 != hundred.config_sha256
    assert (
        QueryBaseTrainingConfig.from_preset(prefix_preset).as_record()
        == QueryBaseTrainingConfig.from_preset(hundred).as_record()
    )
    validate_active_catalog_prefix(semantic_fixture.catalog, prefix_preset)


def test_fact_statistics_generation_and_atomic_ledgers(
    tmp_path: Path,
    semantic_fixture: _Fixture,
) -> None:
    test_templates = tuple(
        template for template in semantic_fixture.catalog.templates if template.split == "test"
    )
    fact_by_id = {fact.fact_id: fact for fact in semantic_fixture.catalog.facts}

    def result(
        template,
        method: str,
        correct: bool,
        adapter_concept_id: str | None = None,
        stage: int = 3,
    ) -> SemanticQueryResult:
        fact = fact_by_id[template.fact_id]
        scores = [2.0, 2.0, 2.0, 2.0]
        scores[template.correct_candidate_index] = 1.0 if correct else 3.0
        if not correct:
            scores[(template.correct_candidate_index + 1) % 4] = 1.0
        predicted = min(range(4), key=scores.__getitem__)
        margin = min(
            score for index, score in enumerate(scores) if index != template.correct_candidate_index
        ) - scores[template.correct_candidate_index]
        return SemanticQueryResult(
            stage=stage,
            method=method,
            concept_id=fact.concept_id,
            fact_id=fact.fact_id,
            template_id=template.template_id,
            direction=template.direction,
            split="test",
            adapter_concept_id=adapter_concept_id,
            candidate_nll=tuple(scores),  # type: ignore[arg-type]
            correct_candidate_index=template.correct_candidate_index,
            predicted_candidate_index=predicted,
            answer_correct=correct,
            correct_answer_margin=margin,
            selected_node_index=None,
            oracle_node_index=None,
            routed_regret=None,
        )

    base_results = tuple(result(template, "base", False) for template in test_templates)
    adapter_results = tuple(result(template, "independent", True) for template in test_templates)
    base_facts = average_paraphrases(base_results)
    adapter_facts = average_paraphrases(adapter_results)
    accuracy = bootstrap_fact_metric(
        adapter_facts,
        "accuracy",
        replicates=200,
        identity="fixture",
    )
    repeated = bootstrap_fact_metric(
        adapter_facts,
        "accuracy",
        replicates=200,
        identity="fixture",
    )
    assert accuracy == repeated
    assert accuracy.point == 1.0
    acquisition = acquisition_effect(
        base_facts,
        adapter_facts,
        "accuracy",
        replicates=200,
    )
    assert acquisition.point == 1.0
    forced_results = tuple(
        result(
            template,
            "forced-independent",
            fact_by_id[template.fact_id].concept_id == adapter_concept_id,
            adapter_concept_id,
        )
        for adapter_concept_id in ("cat", "dog", "pig")
        for template in test_templates
    )
    specificity = specificity_effect(
        average_paraphrases(forced_results),
        "accuracy",
        replicates=200,
    )
    assert specificity.point == 1.0
    prompts = generation_prompts(semantic_fixture.catalog.concepts)
    outputs = tuple(
        (
            concept_id,
            prompt,
            f"This creature shows {_PREDICATES[0].predicate} and {_PREDICATES[1].predicate}.",
        )
        for concept_id, prompt in prompts
    )
    generation = inspect_generation(semantic_fixture.catalog, outputs)
    assert all(item.recall == pytest.approx(2 / 12) for item in generation)
    ledger = write_atomic_jsonl(tmp_path / "results.jsonl", adapter_results)
    assert len(ledger.read_text().splitlines()) == len(adapter_results)
    report = render_semantic_report(
        "a" * 64,
        QueryExperimentPreset(("cat", "dog", "pig"), adapter_updates=500),
        (accuracy,),
        (acquisition,),
        generation,
        {"training": 1.0},
        {"allocator_peak": 1},
    )
    assert "no VAMP scientific pass/fail verdict" in report
    schedule_cells = tuple(
        (stage, concept_id)
        for stage in range(1, 4)
        for concept_id in ("cat", "dog", "pig")[:stage]
    )
    complete_results = tuple(
        result(template, "base", False, stage=0)
        for template in test_templates
    ) + tuple(
        result(template, method, True, stage=stage)
        for method in REQUIRED_QUERY_METHODS
        if method != "base"
        for stage, concept_id in schedule_cells
        for template in test_templates
        if fact_by_id[template.fact_id].concept_id == concept_id
    )
    report_effects = (
        replace(acquisition, replicate_count=10_000),
        replace(
            acquisition,
            metric="node-specificity:accuracy",
            replicate_count=10_000,
        ),
        replace(
            acquisition,
            metric="acquisition-to-final-retention:accuracy",
            replicate_count=10_000,
        ),
    )
    published = publish_semantic_report(
        tmp_path / "reports",
        catalog_sha256=semantic_fixture.catalog.catalog_sha256,
        partition_sha256="b" * 64,
        preset=QueryExperimentPreset(("cat", "dog", "pig"), adapter_updates=500),
        results=complete_results,
        effects=report_effects,
        generation=generation,
        runtime_seconds={"training": 1.0},
        memory_bytes={"allocator_peak": 1},
    )
    assert publish_semantic_report(
        tmp_path / "reports",
        catalog_sha256=semantic_fixture.catalog.catalog_sha256,
        partition_sha256="b" * 64,
        preset=QueryExperimentPreset(("cat", "dog", "pig"), adapter_updates=500),
        results=complete_results,
        effects=report_effects,
        generation=generation,
        runtime_seconds={"training": 1.0},
        memory_bytes={"allocator_peak": 1},
    ).report_sha256 == published.report_sha256
    with (published.root / "report.md").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(FileExistsError, match="does not match"):
        publish_semantic_report(
            tmp_path / "reports",
            catalog_sha256=semantic_fixture.catalog.catalog_sha256,
            partition_sha256="b" * 64,
            preset=QueryExperimentPreset(("cat", "dog", "pig"), adapter_updates=500),
            results=complete_results,
            effects=report_effects,
            generation=generation,
            runtime_seconds={"training": 1.0},
            memory_bytes={"allocator_peak": 1},
        )


def test_semantic_projection_normalizes_float32_margin() -> None:
    candidate_nll = np.asarray(
        [0.12345679, 25.987654, 40.0, 50.0],
        dtype=np.float32,
    )
    scorer_margin = float(
        np.min(np.delete(candidate_nll, 0)) - float(candidate_nll[0])
    )
    query = SimpleNamespace(
        template_id="rabbit-color-validation-0",
        concept_id="rabbit",
        fact_id="rabbit-color",
        direction="forward",
        split="validation",
    )
    scorer_result = SimpleNamespace(
        stage=0,
        method="base",
        query_id=query.template_id,
        task_id=query.concept_id,
        proof_id=query.fact_id,
        candidate_nll=candidate_nll,
        correct_candidate_index=0,
        predicted_candidate_index=0,
        candidate_correct=True,
        candidate_margin=scorer_margin,
        selected_node_index=None,
        task_oracle_node_index=None,
        routed_regret=None,
    )

    projected = project_semantic_result(query, scorer_result)
    exact_margin = float(candidate_nll[1]) - float(candidate_nll[0])
    assert projected.correct_answer_margin == exact_margin
    assert projected.correct_answer_margin != scorer_margin


def test_root_adapter_scoring_unwraps_semantic_queries(monkeypatch) -> None:
    knowledge_query = object()
    semantic_query = SimpleNamespace(knowledge_query=knowledge_query)
    packed = object()
    monkeypatch.setattr(
        semantic_evaluation,
        "pack_root_adapter",
        lambda adapter, model_config, lora_config: (object(), packed),
    )
    monkeypatch.setattr(
        semantic_evaluation,
        "edge_coefficients_for_node",
        lambda actual_packed, node_index: np.asarray([1.0], dtype=np.float32),
    )
    received_queries = []

    def score_candidates(
        base_params,
        model_config,
        actual_packed,
        lora_config,
        queries,
        edge_coefficients,
        *,
        evaluation_microbatch_size,
    ):
        received_queries.append(queries)
        return np.zeros((len(queries), 4), dtype=np.float32)

    monkeypatch.setattr(
        semantic_evaluation,
        "score_edge_coefficient_candidates",
        score_candidates,
    )
    base = SimpleNamespace(params=object(), config=object())
    preset = SimpleNamespace(
        lora_config=object(),
        query_chunk_size=1,
    )

    scores = semantic_evaluation._score_root_adapter(  # noqa: SLF001
        (semantic_query,),
        base,
        object(),
        preset,
        progress=None,
        phase="fixture",
    )
    assert received_queries == [(knowledge_query,)]
    assert scores.shape == (1, 4)


def test_operational_gates_and_resumable_stage_parity(tmp_path: Path) -> None:
    assert BaseQualityDecision((2.1, 2.0), 1, 2).passed
    assert BaseQualityDecision((2.1, 2.09), 1, 2).reason == "epoch_improvement_below_0.02"
    preset = QueryExperimentPreset(("cat", "dog"), adapter_updates=500)
    pilot_results = tuple(
        PilotBudgetResult(
            budget,
            (("cat", accuracy), ("dog", accuracy)),
            (("cat", 0.40), ("dog", 0.40)),
        )
        for budget, accuracy in ((500, 0.55), (1_000, 0.61), (2_000, 0.80))
    )
    assert (
        select_pilot_budget(
            pilot_results,
            preset,
            ORIGINAL_PILOT_LEARNABILITY_POLICY,
        )
        == 1_000
    )

    def result_rows(
        method: str,
        concept_id: str,
        correct_count: int,
        stage: int,
        adapter_concept_id: str | None,
    ) -> tuple[SemanticQueryResult, ...]:
        return tuple(
            SemanticQueryResult(
                stage=stage,
                method=method,
                concept_id=concept_id,
                fact_id=f"fact-{concept_id}-{index % 12:02d}",
                template_id=f"{concept_id}-q-{index:02d}",
                direction="forward",
                split="validation",
                adapter_concept_id=adapter_concept_id,
                candidate_nll=(
                    (1.0, 2.0, 2.0, 2.0)
                    if index < correct_count
                    else (3.0, 1.0, 2.0, 2.0)
                ),
                correct_candidate_index=0,
                predicted_candidate_index=0 if index < correct_count else 1,
                answer_correct=index < correct_count,
                correct_answer_margin=1.0 if index < correct_count else -2.0,
                selected_node_index=None,
                oracle_node_index=None,
                routed_regret=None,
            )
            for index in range(36)
        )

    budget_evaluations = tuple(
        PilotBudgetEvaluation(
            PilotBudgetResult(
                updates,
                tuple(
                    (concept_id, correct_count / 36)
                    for concept_id in preset.concept_ids
                ),
                tuple((concept_id, 0.5) for concept_id in preset.concept_ids),
            ),
            tuple(
                row
                for concept_id in preset.concept_ids
                for row in (
                    result_rows("base", concept_id, 18, 0, None)
                    + result_rows(
                        "independent",
                        concept_id,
                        correct_count,
                        preset.concept_ids.index(concept_id) + 1,
                        concept_id,
                    )
                )
            ),
            "a" * 64,
        )
        for updates, correct_count in ((500, 18), (1_000, 24), (2_000, 27))
    )
    selected_validation = tuple(
        row
        for method in REQUIRED_QUERY_METHODS
        for stage, concepts in (
            ((0, preset.concept_ids),)
            if method == "base"
            else ((1, ("cat",)), (2, preset.concept_ids))
        )
        for concept_id in concepts
        for row in result_rows(
            method,
            concept_id,
            24,
            stage,
            concept_id if method == "independent" else None,
        )
    )
    pilot = publish_semantic_pilot_result(
        tmp_path / "pilot-results",
        catalog_sha256="1" * 64,
        partition_sha256="2" * 64,
        selected_base_sha256="3" * 64,
        preflight_sha256="4" * 64,
        preset=preset,
        learnability_policy=ORIGINAL_PILOT_LEARNABILITY_POLICY,
        protocol_amendment_sha256=None,
        budgets=budget_evaluations,
        independent_sweep_sha256="5" * 64,
        independent_sweep_manifest_sha256="6" * 64,
        selected_adaptation_manifest_sha256="7" * 64,
        selected_validation_results=selected_validation,
        resume_verified=True,
        runtime_seconds=1.0,
        allocator_peak_bytes=1,
    )
    assert pilot.selected_updates == 1_000
    assert load_semantic_pilot_result(pilot.root) == pilot

    failed_counts = {
        500: {"cat": 22, "dog": 21},
        1_000: {"cat": 22, "dog": 20},
        2_000: {"cat": 23, "dog": 22},
    }
    failed_evaluations = tuple(
        PilotBudgetEvaluation(
            PilotBudgetResult(
                updates,
                tuple(
                    (concept_id, failed_counts[updates][concept_id] / 36)
                    for concept_id in preset.concept_ids
                ),
                (("cat", 0.5), ("dog", 0.5)),
            ),
            tuple(
                row
                for concept_id in preset.concept_ids
                for correct_count in (failed_counts[updates][concept_id],)
                for row in (
                    result_rows("base", concept_id, 18, 0, None)
                    + result_rows(
                        "independent",
                        concept_id,
                        correct_count,
                        preset.concept_ids.index(concept_id) + 1,
                        concept_id,
                    )
                )
            ),
            "8" * 64,
        )
        for updates in preset.pilot_update_budgets
    )
    failure = publish_semantic_pilot_failure(
        tmp_path / "pilot-failures",
        catalog_sha256="1" * 64,
        partition_sha256="2" * 64,
        selected_base_sha256="3" * 64,
        preflight_sha256="4" * 64,
        preset=preset,
        budgets=failed_evaluations,
        independent_sweep_sha256="5" * 64,
        independent_sweep_manifest_sha256="6" * 64,
        allocator_peak_bytes=1,
    )
    assert load_semantic_pilot_failure(failure.root) == failure
    assert not any(
        pilot_budget_passes(
            evaluation.budget,
            ORIGINAL_PILOT_LEARNABILITY_POLICY,
        )
        for evaluation in failed_evaluations
    )
    assert pilot_budget_passes(
        failed_evaluations[-1].budget,
        AMENDED_PILOT_LEARNABILITY_POLICY,
    )
    amendment = publish_semantic_pilot_protocol_amendment(
        tmp_path / "pilot-amendments",
        failure,
        preset,
        reviewer="fixture-reviewer",
        decided_at="2026-07-25T12:00:00Z",
        rationale=(
            "The pilot established absolute learnability while the original "
            "acquisition threshold was ceiling-sensitive."
        ),
    )
    assert amendment.selected_updates == 2_000
    assert (
        load_semantic_pilot_protocol_amendment(amendment.root, failure, preset)
        == amendment
    )
    amended_pilot = publish_semantic_pilot_result(
        tmp_path / "amended-pilot-results",
        catalog_sha256="1" * 64,
        partition_sha256="2" * 64,
        selected_base_sha256="3" * 64,
        preflight_sha256="4" * 64,
        preset=preset,
        learnability_policy=AMENDED_PILOT_LEARNABILITY_POLICY,
        protocol_amendment_sha256=amendment.amendment_sha256,
        budgets=failed_evaluations,
        independent_sweep_sha256="5" * 64,
        independent_sweep_manifest_sha256="6" * 64,
        selected_adaptation_manifest_sha256="7" * 64,
        selected_validation_results=selected_validation,
        resume_verified=True,
        runtime_seconds=1.0,
        allocator_peak_bytes=1,
    )
    main_preset = QueryExperimentPreset(
        tuple(concept.concept_id for concept in MAIN_CONCEPTS),
        adapter_updates=2_000,
    )
    main_freeze = publish_main_experiment_freeze(
        tmp_path / "main-freezes",
        amended_pilot,
        amendment,
        failure,
        preset,
        main_preset,
        authorized_by="fixture-reviewer",
        authorized_at="2026-07-25T13:00:00Z",
        authorization="Proceed to the five-world main construction.",
    )
    assert main_freeze.selected_updates == 2_000
    assert main_freeze.query_protocol_sha256 == REGISTERED_QUERY_PROTOCOL.protocol_sha256
    assert (
        load_main_experiment_freeze(
            main_freeze.root,
            amended_pilot,
            amendment,
            failure,
            preset,
            main_preset,
        )
        == main_freeze
    )
    failure_report = (failure.root / "report.md").read_text()
    assert "learnability stop" in failure_report
    assert "Sequential/VAMP and main execution therefore remained unauthorized" in failure_report
    with (failure.root / "budget-0500-validation.jsonl").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="file changed"):
        load_semantic_pilot_failure(failure.root)

    with (pilot.root / "selected-validation.jsonl").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="file changed"):
        load_semantic_pilot_result(pilot.root)

    artifacts = tuple(
        (
            publish_stage_artifact(
                tmp_path / "first",
                preset,
                system=system,
                stage=1,
                payloads={"state.bin": b"deterministic-state"},
            ),
            publish_stage_artifact(
                tmp_path / "second",
                preset,
                system=system,
                stage=1,
                payloads={"state.bin": b"deterministic-state"},
            ),
        )
        for system in ("base", "independent", "sequential", "vamp")
    )
    assert all(first.state_sha256 == second.state_sha256 for first, second in artifacts)
    first = artifacts[2][0]
    incomplete = tmp_path / "first" / "sequential" / ".stage-002-interrupted"
    incomplete.mkdir()
    assert latest_stage_artifact(
        tmp_path / "first",
        preset,
        system="sequential",
    ) == first
    with (first.root / "state.bin").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="payload changed"):
        load_stage_artifact(first.root, preset, system="sequential")


def test_pilot_independent_sweep_round_trip_and_tampering(tmp_path: Path) -> None:
    preset = QueryExperimentPreset(("cat",), adapter_updates=2_000)
    partition_root = tmp_path / "partition"
    partition_root.mkdir()
    partition = QueryPartitionArtifact(
        root=partition_root,
        partition_sha256="1" * 64,
        manifest_sha256="2" * 64,
        catalog_sha256="3" * 64,
        archive_identity=CANONICAL_ARCHIVE_IDENTITY,
        tokenizer_identity=CANONICAL_TOKENIZER_IDENTITY,
        pad_token_id=0,
        eos_token_id=1,
        concept_ids=preset.concept_ids,
        counts=(),
        files=(),
    )
    router_batch = RouterBatch(
        input_ids=np.asarray([[1]], dtype=np.int32),
        attention_mask=np.asarray([[True]], dtype=np.bool_),
        target_ids=np.asarray([[1]], dtype=np.int32),
        loss_mask=np.asarray([[True]], dtype=np.bool_),
    )
    prepared = PreparedQueryAdaptation(
        catalog_sha256=partition.catalog_sha256,
        partition_sha256=partition.partition_sha256,
        config_sha256=preset.config_sha256,
        concept_ids=preset.concept_ids,
        root_query_ids=("root-query",),
        root_validation_probes=(router_batch,),
        task_probes=(
            QueryTaskProbeSet(
                concept_id="cat",
                parent_query_ids=("parent-query",),
                content_key_query_ids=("content-query",),
                parent_probes=(router_batch,),
                content_key_probes=(router_batch,),
            ),
        ),
        preparation_sha256="4" * 64,
    )
    selected_directory = tmp_path / "selected"
    selected_directory.mkdir()
    selected = QuerySelectedBase(
        directory=selected_directory,
        selection_sha256="5" * 64,
        catalog_sha256=partition.catalog_sha256,
        partition_sha256=partition.partition_sha256,
        base_config_sha256="6" * 64,
        training_sha256="7" * 64,
        epoch_evidence=(
            QueryBaseEpochEvidence(1, QuerySplitNll("validation", 1, 1.5)),
            QueryBaseEpochEvidence(2, QuerySplitNll("validation", 1, 1.4)),
        ),
        allocator_peak_bytes=1,
        checkpoint=BaseCheckpointRef(
            selected_directory,
            "8" * 64,
            "9" * 64,
        ),
    )
    adapter = init_lora_edge(
        jax.random.PRNGKey(0),
        preset.model_config,
        preset.lora_config,
    )
    adapters_by_budget = {
        budget: (
            IndependentRootAdapter(
                TaskId("cat"),
                adapter,
                (1.0,) * budget,
            ),
        )
        for budget in preset.pilot_update_budgets
    }
    destination = tmp_path / "sweep" / "stages" / "stage-001"
    published = publish_pilot_independent_sweep_stage(
        destination,
        prepared,
        partition,
        selected,
        preset,
        ("cat",),
        adapters_by_budget,
        (np.asarray([10, 11], dtype=np.uint32),),
    )
    loaded = load_pilot_independent_sweep(
        destination,
        prepared,
        partition,
        selected,
        preset,
    )
    assert loaded.sweep_sha256 == published.sweep_sha256
    assert tuple(item.updates for item in loaded.budgets) == (500, 1_000, 2_000)
    assert tuple(len(item.adapters[0].step_losses) for item in loaded.budgets) == (
        500,
        1_000,
        2_000,
    )
    rebuilt_destination = tmp_path / "rebuilt-sweep" / "stages" / "stage-001"
    rebuilt = publish_pilot_independent_sweep_stage(
        rebuilt_destination,
        prepared,
        partition,
        selected,
        preset,
        ("cat",),
        adapters_by_budget,
        (np.asarray([10, 11], dtype=np.uint32),),
    )
    assert rebuilt.sweep_sha256 == published.sweep_sha256
    assert (rebuilt_destination / "manifest.json").read_bytes() == (
        destination / "manifest.json"
    ).read_bytes()
    assert (rebuilt_destination / "sweep.safetensors").read_bytes() == (
        destination / "sweep.safetensors"
    ).read_bytes()
    with (destination / "sweep.safetensors").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="tensor file changed"):
        load_pilot_independent_sweep(
            destination,
            prepared,
            partition,
            selected,
            preset,
        )

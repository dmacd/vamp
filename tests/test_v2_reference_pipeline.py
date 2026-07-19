from __future__ import annotations

from hashlib import sha256
import json

import pytest

import apm.data.text.tinyworlds_v2.reference_pipeline as reference_pipeline

from apm.data.text.tinyworlds_v2.reference_pipeline import (
    PHASE1_ARCHIVE_REFERENCE_COUNT,
    PHASE1_BRIEF_COUNT,
    PHASE1_PROMPT_METADATA_COUNT,
    PHASE1_VALIDATION_REFERENCE_COUNT,
    ReferenceAnnotation,
    ReferencePipelineError,
    build_neutral_story_brief,
    build_phase1_reference_inputs,
    build_prompt_ingredient_profile,
    canonical_neutral_story_brief,
    canonical_prompt_ingredient_profile,
    canonical_reference_observation,
    mechanically_classify_ingredient_roles,
    prepare_reference_observations,
)
from apm.data.text.tinyworlds_v2.reference_profile import ReferenceRecord
from apm.data.text.tinyworlds_v2.source_data import (
    ArchiveSourceRecord,
    ArchiveSourceSelections,
    TinyStoriesInstruction,
    ValidationStoryRecord,
)


def _archive_record(
    index: int,
    *,
    words: tuple[str, ...] = ("help", "cat", "kind"),
    features: tuple[str, ...] = ("Dialogue",),
    prompt: str | None = None,
) -> ArchiveSourceRecord:
    released_prompt = prompt or (
        'Write a short story. The story should use the verb "help", '
        'the noun "cat" and the adjective "kind".'
    )
    story = f'Mia saw a kind cat number {index}. "I can help," she said.'
    summary = "Mia helps a cat."
    payload = {
        "instruction": {
            "features": list(features),
            "prompt:": released_prompt,
            "words": list(words),
        },
        "source": "GPT-4",
        "story": story,
        "summary": summary,
    }
    content_sha256 = sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    member = "./data-fixture.json"
    return ArchiveSourceRecord(
        record_id=f"archive:{member}:{index}:{content_sha256}",
        source_member=member,
        source_index=index,
        content_sha256=content_sha256,
        story=story,
        instruction=TinyStoriesInstruction(released_prompt, words, features),
        summary=summary,
        source="GPT-4",
    )


def _validation_record(index: int) -> ValidationStoryRecord:
    story = f"A little dog found a red ball number {index}. It was happy."
    content_sha256 = sha256(story.encode("utf-8")).hexdigest()
    return ValidationStoryRecord(
        record_id=f"v2-validation:{index}:{content_sha256}",
        source_index=index,
        content_sha256=content_sha256,
        story=story,
    )


def test_released_prompt_roles_are_parsed_without_assuming_word_list_order() -> None:
    prompt = (
        'The story should use the verb "rescue", the noun "crown" '
        'and the adjective "quiet".'
    )

    roles = mechanically_classify_ingredient_roles(
        prompt,
        ("rescue", "crown", "quiet"),
    )

    assert roles is not None
    assert (roles.noun, roles.verb, roles.adjective) == (
        "crown",
        "rescue",
        "quiet",
    )
    assert mechanically_classify_ingredient_roles(
        "Please use rescue, crown, and quiet.",
        ("rescue", "crown", "quiet"),
    ) is None


def test_ingredient_profile_is_order_independent_and_retains_ambiguous_positions() -> None:
    parsed = _archive_record(1)
    another = _archive_record(
        2,
        words=("jump", "moon", "bright"),
        features=("Twist",),
        prompt=(
            'Use the adjective "bright", the noun "moon", and the verb "jump".'
        ),
    )
    ambiguous = _archive_record(
        3,
        words=("run", "tree", "green"),
        features=(),
        prompt="Please write a story using run, tree, and green.",
    )

    forward = build_prompt_ingredient_profile((parsed, another, ambiguous))
    reverse = build_prompt_ingredient_profile((ambiguous, another, parsed))

    assert forward == reverse
    assert forward.parsed_role_record_count == 2
    assert forward.unparsed_role_record_count == 1
    assert dict(forward.noun_frequencies) == {"cat": 1, "moon": 1}
    assert dict(forward.verb_frequencies) == {"help": 1, "jump": 1}
    assert dict(forward.adjective_frequencies) == {"bright": 1, "kind": 1}
    assert dict(forward.word_position_frequencies[0].frequencies) == {
        "help": 1,
        "jump": 1,
        "run": 1,
    }
    assert forward.unparsed_word_position_frequencies[0].frequencies == (("run", 1),)
    assert dict(forward.narrative_feature_frequencies) == {
        "Dialogue": 1,
        "Twist": 1,
    }
    assert dict(forward.narrative_feature_rates) == {
        "Dialogue": pytest.approx(1 / 3),
        "Twist": pytest.approx(1 / 3),
    }
    assert forward.any_narrative_feature_rate == pytest.approx(2 / 3)
    assert canonical_prompt_ingredient_profile(forward)["profile_sha256"] == (
        forward.profile_sha256
    )


def test_released_duplicate_ingredients_are_preserved_but_presence_is_deduplicated() -> None:
    record = _archive_record(
        30,
        words=("help", "help", "kind"),
        features=("Dialogue", "dialogue"),
        prompt='Use the verb "help", noun "help", and adjective "kind".',
    )

    profile = build_prompt_ingredient_profile((record,))

    assert profile.unparsed_role_record_count == 1
    assert profile.narrative_feature_frequencies == (("Dialogue", 1),)
    assert profile.narrative_feature_count_frequencies == ((1, 1),)
    assert profile.mean_narrative_feature_count == 1.0

    reference = ReferenceRecord(
        "released-duplicate",
        'A kind cat said, "I will help."',
        "Write a story.",
    )
    annotation = ReferenceAnnotation(
        "released-duplicate",
        "archive",
        ("help", "help", "kind"),
        ("Dialogue", "dialogue"),
    )
    observation = prepare_reference_observations(
        (reference,),
        (annotation,),
        model_token_ids_by_record_id={"released-duplicate": (1, 2, 3)},
        normalized_nll_by_record_id={"released-duplicate": 1.0},
        worker_count=1,
    )[0]

    assert observation.required_words == ("help", "help", "kind")
    assert observation.feature_labels == ("Dialogue",)


def test_neutral_brief_requires_exactly_three_released_words() -> None:
    brief = build_neutral_story_brief(_archive_record(4))

    assert brief.required_words == ("help", "cat", "kind")
    assert brief.requested_features == ("Dialogue",)
    assert len(brief.brief_id) == len("brief-") + 24
    assert canonical_neutral_story_brief(brief)["matched_reference_text"] == (
        brief.matched_reference_text
    )

    with pytest.raises(ReferencePipelineError, match="exactly three"):
        build_neutral_story_brief(
            _archive_record(
                5,
                words=("help", "cat"),
                prompt='Use the verb "help" and noun "cat".',
            )
        )


def test_surface_observations_are_shard_and_input_order_independent() -> None:
    records = (
        ReferenceRecord(
            "archive-story",
            'Mia saw a kind cat. "I will help," she said.',
            "Write a story.",
        ),
        ReferenceRecord(
            "validation-story",
            "A dog found a little ball. It ran home.",
        ),
    )
    annotations = (
        ReferenceAnnotation(
            "archive-story",
            "archive",
            ("help", "cat", "kind"),
            ("Dialogue",),
        ),
        ReferenceAnnotation("validation-story", "validation", (), ()),
    )
    token_ids = {"archive-story": (4, 5, 6), "validation-story": (7, 8)}
    nll = {"archive-story": 1.25, "validation-story": 1.5}

    serial = prepare_reference_observations(
        records,
        annotations,
        model_token_ids_by_record_id=token_ids,
        normalized_nll_by_record_id=nll,
        worker_count=1,
    )
    sharded = prepare_reference_observations(
        tuple(reversed(records)),
        tuple(reversed(annotations)),
        model_token_ids_by_record_id=dict(reversed(tuple(token_ids.items()))),
        normalized_nll_by_record_id=dict(reversed(tuple(nll.items()))),
        worker_count=16,
    )

    assert serial == sharded
    assert tuple(item.record_id for item in serial) == (
        "archive-story",
        "validation-story",
    )
    assert serial[0].required_words == ("help", "cat", "kind")
    assert serial[0].feature_labels == ("Dialogue",)
    assert serial[1].required_words == ()
    assert serial[1].feature_labels == ()
    assert canonical_reference_observation(serial[0])["model_token_ids"] == [4, 5, 6]


def test_surface_observations_reject_missing_or_unexpected_injected_data() -> None:
    records = (ReferenceRecord("story", "A cat sat. It was happy."),)
    annotations = (ReferenceAnnotation("story", "archive", ("cat",), ()),)

    with pytest.raises(ReferencePipelineError, match="missing"):
        prepare_reference_observations(
            records,
            annotations,
            model_token_ids_by_record_id={},
            normalized_nll_by_record_id={"story": 1.0},
            worker_count=1,
        )
    with pytest.raises(ReferencePipelineError, match="unexpected"):
        prepare_reference_observations(
            records,
            annotations,
            model_token_ids_by_record_id={"story": (1,), "other": (2,)},
            normalized_nll_by_record_id={"story": 1.0},
            worker_count=1,
        )


def test_fixed_phase1_join_produces_200_briefs_and_20k_genuine_records() -> None:
    prompt_records = tuple(
        _archive_record(index) for index in range(PHASE1_PROMPT_METADATA_COUNT)
    )
    archive_references = tuple(
        _archive_record(PHASE1_PROMPT_METADATA_COUNT + index)
        for index in range(PHASE1_ARCHIVE_REFERENCE_COUNT)
    )
    paired_records = tuple(
        _archive_record(
            PHASE1_PROMPT_METADATA_COUNT + PHASE1_ARCHIVE_REFERENCE_COUNT + index
        )
        for index in range(PHASE1_BRIEF_COUNT)
    )
    validation_records = tuple(
        _validation_record(index)
        for index in range(PHASE1_VALIDATION_REFERENCE_COUNT)
    )
    selections = ArchiveSourceSelections(
        prompt_metadata_records=tuple(reversed(prompt_records)),
        reference_story_records=tuple(reversed(archive_references)),
        paired_records=tuple(reversed(paired_records)),
    )

    prepared = build_phase1_reference_inputs(selections, validation_records)

    assert len(prepared.briefs) == 200
    assert len(prepared.reference_records) == 20_000
    assert len(prepared.reference_annotations) == 20_000
    assert prepared.ingredient_profile.record_count == 10_000
    assert prepared.ingredient_profile.parsed_role_rate == 1.0
    assert sum(
        annotation.source_partition == "archive"
        and bool(annotation.required_words)
        for annotation in prepared.reference_annotations
    ) == 10_000


def test_fixed_phase1_join_rejects_incomplete_cohorts_before_preparation() -> None:
    selections = ArchiveSourceSelections((), (), ())

    with pytest.raises(ReferencePipelineError, match="exactly 10000 prompt metadata"):
        build_phase1_reference_inputs(selections, ())


def test_phase1_join_rejects_cross_source_normalized_content_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reference_pipeline, "PHASE1_PROMPT_METADATA_COUNT", 1)
    monkeypatch.setattr(reference_pipeline, "PHASE1_ARCHIVE_REFERENCE_COUNT", 1)
    monkeypatch.setattr(reference_pipeline, "PHASE1_VALIDATION_REFERENCE_COUNT", 1)
    monkeypatch.setattr(reference_pipeline, "PHASE1_BRIEF_COUNT", 1)
    prompt = _archive_record(0)
    archive_reference = _archive_record(1)
    paired = _archive_record(2)
    overlapping_story = archive_reference.story.upper().replace(" ", "  ")
    digest = sha256(overlapping_story.encode("utf-8")).hexdigest()
    overlapping_validation = ValidationStoryRecord(
        record_id=f"v2-validation:0:{digest}",
        source_index=0,
        content_sha256=digest,
        story=overlapping_story,
    )
    selections = ArchiveSourceSelections((prompt,), (archive_reference,), (paired,))

    with pytest.raises(ReferencePipelineError, match="disjoint by normalized"):
        build_phase1_reference_inputs(selections, (overlapping_validation,))

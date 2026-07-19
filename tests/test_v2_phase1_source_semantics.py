from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from apm.data.text.tinyworlds_v2.bakeoff import NeutralStoryBrief
from apm.data.text.tinyworlds_v2.phase1_artifacts import (
    Phase1ArtifactBuilder,
    canonical_jsonl_bytes,
)
from apm.data.text.tinyworlds_v2.phase1_runner import _reference_profile_record
import apm.data.text.tinyworlds_v2.phase1_semantics as semantics
from apm.data.text.tinyworlds_v2.reference_pipeline import (
    ReferenceAnnotation,
    build_prompt_ingredient_profile,
    canonical_neutral_story_brief,
    canonical_prompt_ingredient_profile,
    canonical_reference_annotation,
    canonical_reference_observation,
    canonical_reference_record,
)
from apm.data.text.tinyworlds_v2.reference_profile import (
    ReferenceRecord,
    build_reference_profile,
    observe_reference,
)
from apm.data.text.tinyworlds_v2.source_data import (
    ArchiveSourceRecord,
    TinyStoriesInstruction,
    ValidationStoryRecord,
    canonical_prompt_metadata_record,
    canonical_validation_record,
)


def _story(cohort: str, index: int) -> str:
    return (
        f"Mia found a kind bird in the {cohort} garden. She helped it jump "
        f"toward the moon. This tale uses code {chr(ord('a') + index)}."
    )


def _archive_record(namespace: str, index: int, story: str) -> ArchiveSourceRecord:
    prompt = (
        "Write a simple story. The verb is 'jump', the noun is 'moon', "
        "and the adjective is 'kind'."
    )
    words = ("jump", "moon", "kind")
    features: tuple[str, ...] = ()
    summary = f"Fixture {namespace} summary."
    payload = {
        "instruction": {
            "features": list(features),
            "prompt:": prompt,
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
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    member = f"fixture-{namespace}.json"
    return ArchiveSourceRecord(
        record_id=f"archive:{member}:{index}:{content_sha256}",
        source_member=member,
        source_index=index,
        content_sha256=content_sha256,
        story=story,
        instruction=TinyStoriesInstruction(prompt, words, features),
        summary=summary,
        source="GPT-4",
    )


def _validation_record(index: int) -> ValidationStoryRecord:
    story = _story("validation", index)
    content_sha256 = sha256(story.encode()).hexdigest()
    return ValidationStoryRecord(
        f"v2-validation:{index}:{content_sha256}",
        index,
        content_sha256,
        story,
    )


def _build_source_tree(
    root: Path,
    *,
    leaked_prompt_story: str | None = None,
    stale_reference_profile: bool = False,
    stale_prompt_summary: bool = False,
):
    root.mkdir()
    prompt = tuple(
        _archive_record(
            "prompt",
            index,
            leaked_prompt_story if index == 0 and leaked_prompt_story else _story("prompt", index),
        )
        for index in range(2)
    )
    paired = tuple(
        _archive_record("paired", index, _story("paired", index)) for index in range(2)
    )
    archive = tuple(
        _archive_record("archive", index, _story("archive", index)) for index in range(2)
    )
    validation = tuple(_validation_record(index) for index in range(2))
    briefs = tuple(
        sorted(
            (
                NeutralStoryBrief(
                    "brief-"
                    + sha256(
                        f"tinyworlds-v2-phase1-brief\0{record.record_id}".encode()
                    ).hexdigest()[:24],
                    record.record_id,
                    record.instruction.prompt,
                    record.instruction.words,
                    (),
                    record.story,
                )
                for record in paired
            ),
            key=lambda item: item.brief_id,
        )
    )
    pairs = tuple(
        sorted(
            (
                *(
                    (
                        ReferenceRecord(
                            item.record_id,
                            item.story,
                            item.instruction.prompt,
                            "GPT-4",
                        ),
                        ReferenceAnnotation(
                            item.record_id,
                            "archive",
                            item.instruction.words,
                            (),
                        ),
                    )
                    for item in archive
                ),
                *(
                    (
                        ReferenceRecord(item.record_id, item.story, source_model="GPT-4"),
                        ReferenceAnnotation(item.record_id, "validation", (), ()),
                    )
                    for item in validation
                ),
            ),
            key=lambda item: item[0].record_id,
        )
    )
    reference_records = tuple(item[0] for item in pairs)
    annotations = tuple(item[1] for item in pairs)
    observations = tuple(
        observe_reference(
            record,
            model_token_ids=(1, 2, 3),
            normalized_nll=2.0,
            feature_labels=annotation.feature_labels,
            required_words=annotation.required_words,
        )
        for record, annotation in pairs
    )
    paired_observations = tuple(
        observe_reference(
            ReferenceRecord(
                brief.source_record_id,
                brief.matched_reference_text,
                brief.prompt_text,
                "GPT-4",
            ),
            model_token_ids=(1, 2, 3),
            normalized_nll=2.0,
            required_words=brief.required_words,
        )
        for brief in briefs
    )
    prompt_rows = [canonical_prompt_metadata_record(item) for item in prompt]
    if stale_prompt_summary:
        prompt_rows[0] = {**prompt_rows[0], "summary": "Changed after hashing."}
    reference_profile = _reference_profile_record(build_reference_profile(observations))
    if stale_reference_profile:
        reference_profile = {**reference_profile, "dialogue_rate": 0.25}
    builder = Phase1ArtifactBuilder(root)
    artifacts = {
        "neutral_story_briefs.jsonl": tuple(
            canonical_neutral_story_brief(item) for item in briefs
        ),
        "paired_reference_observations.jsonl": tuple(
            canonical_reference_observation(item) for item in paired_observations
        ),
        "prompt_metadata_sample.jsonl": tuple(prompt_rows),
        "reference_annotations.jsonl": tuple(
            canonical_reference_annotation(item) for item in annotations
        ),
        "reference_observations.jsonl": tuple(
            canonical_reference_observation(item) for item in observations
        ),
        "reference_story_sample.jsonl": tuple(
            canonical_reference_record(item) for item in reference_records
        ),
        "validation_source_sample.jsonl": tuple(
            canonical_validation_record(item) for item in validation
        ),
    }
    for path, records in artifacts.items():
        builder.write_bytes(path, canonical_jsonl_bytes(records))
    builder.write_json(
        "reference_statistics.json",
        {
            "ingredient_profile": canonical_prompt_ingredient_profile(
                build_prompt_ingredient_profile(prompt)
            ),
            "nll_runtime": {"fixture": True},
            "paired_reference_profile": _reference_profile_record(
                build_reference_profile(paired_observations)
            ),
            "paired_source_record_ids": [item.source_record_id for item in briefs],
            "reference_profile": reference_profile,
        },
    )
    return builder.finalize(), archive


@pytest.fixture(autouse=True)
def _tiny_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(semantics, "PHASE1_PROMPT_METADATA_COUNT", 2)
    monkeypatch.setattr(semantics, "PHASE1_ARCHIVE_REFERENCE_COUNT", 2)
    monkeypatch.setattr(semantics, "PHASE1_VALIDATION_REFERENCE_COUNT", 2)
    monkeypatch.setattr(semantics, "PHASE1_BRIEF_COUNT", 2)


def test_source_semantics_reconstructs_complete_evidence(tmp_path: Path) -> None:
    manifest, _ = _build_source_tree(tmp_path / "valid")

    semantics._validate_source_cohort_counts(tmp_path / "valid", manifest)


def test_source_semantics_rejects_coherently_resealed_story_leakage(
    tmp_path: Path,
) -> None:
    first_manifest, archive = _build_source_tree(tmp_path / "first")
    semantics._validate_source_cohort_counts(tmp_path / "first", first_manifest)
    leaked_manifest, _ = _build_source_tree(
        tmp_path / "leaked",
        leaked_prompt_story=archive[0].story,
    )

    with pytest.raises(semantics.Phase1SemanticError, match="leak normalized story"):
        semantics._validate_source_cohort_counts(tmp_path / "leaked", leaked_manifest)


def test_source_semantics_rejects_stale_reference_profile(tmp_path: Path) -> None:
    manifest, _ = _build_source_tree(
        tmp_path / "stale-profile",
        stale_reference_profile=True,
    )

    with pytest.raises(semantics.Phase1SemanticError, match="reference profile is stale"):
        semantics._validate_source_cohort_counts(tmp_path / "stale-profile", manifest)


def test_source_semantics_reauthenticates_prompt_source_payload(tmp_path: Path) -> None:
    manifest, _ = _build_source_tree(
        tmp_path / "stale-prompt",
        stale_prompt_summary=True,
    )

    with pytest.raises(semantics.Phase1SemanticError, match="content SHA-256 is stale"):
        semantics._validate_source_cohort_counts(tmp_path / "stale-prompt", manifest)

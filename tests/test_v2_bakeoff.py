from __future__ import annotations

import json

import pytest

from apm.data.text.tinyworlds_v2.bakeoff import (
    CANDIDATE_MODELS,
    NeutralStoryBrief,
    assistant_message_content,
    neutral_story_request_body,
    parse_verifier_payload,
    request_body_sha256,
    validate_generated_story,
)


def _brief() -> NeutralStoryBrief:
    return NeutralStoryBrief(
        brief_id="brief-001",
        source_record_id="archive:data00.json:1",
        prompt_text="Write a simple story using moon, jump, and kind, with dialogue.",
        required_words=("moon", "jump", "kind"),
        requested_features=("Dialogue",),
        matched_reference_text="A matched genuine reference story.",
    )


def _story_payload(story: str) -> str:
    return json.dumps(
        {
            "story": story,
            "word_evidence": [
                {"required_word": "moon", "exact_quote": "moon"},
                {"required_word": "jump", "exact_quote": "jump"},
                {"required_word": "kind", "exact_quote": "kind"},
            ],
            "feature_evidence": [
                {"feature": "Dialogue", "exact_quote": '"Come with me," said Mia.'},
            ],
        },
        separators=(",", ":"),
        sort_keys=True,
    )


def test_candidate_table_is_fixed_and_model_snapshots_are_canonical() -> None:
    assert [model.route_id for model in CANDIDATE_MODELS] == [
        "ling-2.6-flash",
        "gemma-4-26b-a4b-it",
        "deepseek-v4-flash",
        "mistral-small-2603",
        "qwen3.5-35b-a3b",
        "gemini-3.1-flash-lite",
        "gpt-5.4-mini",
    ]
    assert all(model.canonical_slug.startswith(model.request_model_id) for model in CANDIDATE_MODELS)


def test_generation_request_disables_routing_fallbacks_and_plugins() -> None:
    body = neutral_story_request_body(
        _brief(),
        CANDIDATE_MODELS[1],
        provider_slug="google-vertex/global",
        provider_quantization="unknown",
        prompt_usd_per_token="0.00000015",
        completion_usd_per_token="0.0000006",
    )

    assert body["plugins"] == []
    assert body["transforms"] == []
    assert body["stream"] is False
    assert body["provider"] == {
        "allow_fallbacks": False,
        "data_collection": "deny",
        "enforce_distillable_text": True,
        "max_price": {"completion": 0.6, "prompt": 0.15},
        "only": ["google-vertex/global"],
        "quantizations": ["unknown"],
        "require_parameters": True,
    }
    assert body["response_format"]["json_schema"]["strict"] is True
    assert request_body_sha256(body) == request_body_sha256(body)


def test_no_feature_brief_uses_a_valid_zero_length_evidence_schema() -> None:
    brief = NeutralStoryBrief(
        brief_id="brief-no-feature",
        source_record_id="archive:no-feature",
        prompt_text=(
            'Use the noun "moon", the verb "jump", and adjective "kind".'
        ),
        required_words=("moon", "jump", "kind"),
        requested_features=(),
        matched_reference_text="A kind child saw the moon and tried to jump.",
    )
    body = neutral_story_request_body(
        brief,
        CANDIDATE_MODELS[1],
        provider_slug="google-vertex/global",
        provider_quantization="unknown",
        prompt_usd_per_token="0.00000015",
        completion_usd_per_token="0.0000006",
    )

    evidence = body["response_format"]["json_schema"]["schema"]["properties"]["feature_evidence"]
    assert evidence["minItems"] == 0
    assert evidence["maxItems"] == 0
    assert "enum" not in evidence["items"]["properties"]["feature"]


def test_four_bit_generation_route_is_rejected() -> None:
    with pytest.raises(ValueError, match="four-bit"):
        neutral_story_request_body(
            _brief(),
            CANDIDATE_MODELS[2],
            provider_slug="deepinfra/fp4",
            provider_quantization="fp4",
            prompt_usd_per_token="0.00000009",
            completion_usd_per_token="0.00000018",
        )


def test_story_payload_accepts_only_exact_evidence_and_natural_text() -> None:
    story = (
        'Mia saw the moon above her little red house. "Come with me," said Mia. '
        "Her kind friend Ben came outside. They wanted to jump over a small puddle, "
        "but first they helped a wet frog. The frog was safe, and the two friends "
        "went home smiling under the bright moon."
    )
    payload, validation = validate_generated_story(_brief(), _story_payload(story))

    assert payload is not None
    assert validation.accepted
    assert validation.schema_valid
    assert validation.required_words_present
    assert validation.evidence_valid


def test_story_evidence_order_is_irrelevant_but_duplicates_are_rejected() -> None:
    story = (
        'Mia saw the moon above her little red house. "Come with me," said Mia. '
        "Her kind friend Ben came outside. They wanted to jump over a small puddle, "
        "but first they helped a wet frog. The frog was safe, and the two friends "
        "went home smiling under the bright moon."
    )
    record = json.loads(_story_payload(story))
    record["word_evidence"].reverse()
    _, reordered = validate_generated_story(_brief(), json.dumps(record))
    assert reordered.accepted

    record["word_evidence"][0]["required_word"] = "moon"
    _, duplicated = validate_generated_story(_brief(), json.dumps(record))
    assert not duplicated.accepted
    assert "evidence_invalid" in duplicated.rejection_reasons


def test_schema_invalid_and_meta_language_outputs_remain_rejections() -> None:
    payload, invalid = validate_generated_story(_brief(), '{"story":"broken"}')
    assert payload is None
    assert invalid.rejection_reasons == ("schema_invalid",)

    long_meta_story = (
        'Mia saw the moon. "Come with me," said Mia. Her kind friend wanted to jump. '
        "They walked to a little tree and found a bird there. The prompt told them "
        "to help it, so they gave it water and went home with happy smiles together."
    )
    _, meta = validate_generated_story(_brief(), _story_payload(long_meta_story))
    assert not meta.accepted
    assert "forbidden_identifier_or_meta_language" in meta.rejection_reasons


@pytest.mark.parametrize(
    "forbidden_phrase",
    ("Page 12", "fox7", "record_id", "item-42"),
)
def test_internal_looking_numeric_forms_are_forbidden(
    forbidden_phrase: str,
) -> None:
    story = (
        'Mia saw the moon above her house. "Come with me," said Mia. '
        "Her kind friend wanted to jump over a puddle. They helped three frogs "
        "find clean water, and each frog gave them a happy wave. Then they walked "
        f"home together. {forbidden_phrase} was written on a sign nearby."
    )

    _, validation = validate_generated_story(_brief(), _story_payload(story))

    assert not validation.accepted
    assert "forbidden_identifier_or_meta_language" in validation.rejection_reasons


def test_ordinary_standalone_numbers_are_allowed() -> None:
    story = (
        'Mia saw the moon above her house. "Come with me," said Mia. '
        "Her kind friend wanted to jump over a puddle. They helped 3 small frogs "
        "find clean water. Each frog gave them a happy wave. Then the friends "
        "walked home together and shared 2 warm buns under the bright moon."
    )

    _, validation = validate_generated_story(_brief(), _story_payload(story))

    assert validation.accepted


def test_ordinary_hyphenated_age_is_not_misclassified_as_an_identifier() -> None:
    story = (
        'Mia saw the moon above her house. "Come with me," said Mia. '
        "Her kind friend wanted to jump over a puddle. They helped a 3-year-old "
        "dog find clean water. The dog gave them a happy wave. Then the friends "
        "walked home together and shared 2 warm buns under the bright moon."
    )

    _, validation = validate_generated_story(_brief(), _story_payload(story))

    assert validation.accepted


def test_hyphenated_mixed_alphanumeric_identifier_is_still_forbidden() -> None:
    story = (
        'Mia saw the moon above her house. "Come with me," said Mia. '
        "Her kind friend wanted to jump over a puddle. They helped three frogs "
        "find clean water, and each frog gave them a happy wave. Then they walked "
        "home together. A toy called R2-D2 was written on a sign nearby."
    )

    _, validation = validate_generated_story(_brief(), _story_payload(story))

    assert not validation.accepted
    assert "forbidden_identifier_or_meta_language" in validation.rejection_reasons


def test_verifier_payload_is_strict_and_reports_mean() -> None:
    content = json.dumps(
        {
            "brief_adherence": True,
            "grammar": 5,
            "hard_failures": [],
            "non_repetition": 4,
            "plot_coherence": 4,
            "preschool_vocabulary": 5,
            "rationale": "Simple, clear, and complete.",
            "sentence_simplicity": 5,
        }
    )
    parsed = parse_verifier_payload(content)
    assert parsed.mean_score == pytest.approx(4.6)

    with pytest.raises(ValueError, match="fields differ"):
        parse_verifier_payload(content[:-1] + ',"source_guess":"real"}')


def test_assistant_content_requires_one_stopped_choice() -> None:
    body = json.dumps(
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": "{\"story\":\"ok\"}", "role": "assistant"},
                }
            ]
        }
    ).encode()
    assert assistant_message_content(body) == '{"story":"ok"}'

    changed = json.loads(body)
    changed["choices"][0]["finish_reason"] = "length"
    with pytest.raises(ValueError, match="finish"):
        assistant_message_content(json.dumps(changed).encode())

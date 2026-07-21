from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json

import pytest

from apm.data.text.tinyworlds_v2.bakeoff import (
    CANDIDATE_MODELS,
    GENERATION_REQUEST_V1,
    NeutralStoryBrief,
    SYNTHETIC_STORY_REQUEST_V2,
    SYNTHETIC_STORY_REQUEST_V3,
    SYNTHETIC_STORY_REQUEST_V4,
    SYNTHETIC_STORY_REQUEST_V5,
    SYNTHETIC_STORY_REQUEST_V6,
    SYNTHETIC_STORY_REQUEST_V7,
    SYNTHETIC_STORY_REQUEST_V8,
    SYNTHETIC_STORY_REQUEST_V9,
    TWO_ROUTE_AUTHOR_MODELS,
    assistant_message_content,
    neutral_story_request_body,
    parse_verifier_payload,
    request_body_sha256,
    validate_generated_story,
    validate_plain_text_generated_story,
    validate_story_only_generated_story,
    verifier_request_body,
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


def test_two_route_author_table_is_qwen_and_gpt_without_mutating_preview_table() -> None:
    assert tuple(model.route_id for model in TWO_ROUTE_AUTHOR_MODELS) == (
        "qwen3.5-35b-a3b",
        "gpt-5.4-mini",
    )
    assert len(CANDIDATE_MODELS) == 7


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


def test_v2_requests_are_distinct_portable_synthetic_authoring_contract() -> None:
    brief = replace(
        _brief(),
        brief_id="preview-brief-a0cd61e7c4ebc9cd571ced1f",
    )
    story = "A kind child saw the moon and tried to jump."
    legacy_generation = neutral_story_request_body(
        brief,
        CANDIDATE_MODELS[1],
        provider_slug="google-vertex/global",
        provider_quantization="unknown",
        prompt_usd_per_token="0.00000015",
        completion_usd_per_token="0.0000006",
        request_contract=GENERATION_REQUEST_V1,
    )
    legacy_verifier = verifier_request_body(
        brief,
        story,
        provider_slug="openai",
        provider_quantization="bf16",
        prompt_usd_per_token="0.00000075",
        completion_usd_per_token="0.0000045",
        request_contract=GENERATION_REQUEST_V1,
    )
    generation = neutral_story_request_body(
        brief,
        CANDIDATE_MODELS[1],
        provider_slug="google-vertex/global",
        provider_quantization="unknown",
        prompt_usd_per_token="0.00000015",
        completion_usd_per_token="0.0000006",
        request_contract=SYNTHETIC_STORY_REQUEST_V2,
    )
    verifier = verifier_request_body(
        brief,
        story,
        provider_slug="openai",
        provider_quantization="bf16",
        prompt_usd_per_token="0.00000075",
        completion_usd_per_token="0.0000045",
        request_contract=SYNTHETIC_STORY_REQUEST_V2,
    )

    assert legacy_generation["provider"]["enforce_distillable_text"] is True
    assert legacy_verifier["provider"]["enforce_distillable_text"] is True
    assert request_body_sha256(legacy_generation) == (
        "f6ce4a774c1306e7796aa2a81a9fde61d30802212bdc80c1e1bdf1e259cdd545"
    )
    assert request_body_sha256(legacy_verifier) == (
        "c93800654fc73f1a1dc2bee03910785e465e21cf298fb5189db6344bf9fc1ecc"
    )
    assert request_body_sha256(legacy_generation) != request_body_sha256(generation)

    generation_seed_v1 = int(
        sha256(brief.brief_id.encode("utf-8")).hexdigest()[:8], 16
    )
    assert generation_seed_v1 > 2**31 - 1
    verifier_seed_material = (
        "verifier\0"
        f"{brief.brief_id}\0"
        f"{sha256(story.encode('utf-8')).hexdigest()}"
    )
    verifier_seed_v1 = int(
        sha256(verifier_seed_material.encode("utf-8")).hexdigest()[:8], 16
    )
    assert legacy_generation["seed"] == generation_seed_v1
    assert legacy_verifier["seed"] == verifier_seed_v1
    assert generation["seed"] == generation_seed_v1 & 0x7FFF_FFFF
    assert verifier["seed"] == verifier_seed_v1 & 0x7FFF_FFFF

    for body in (generation, verifier):
        seed = body["seed"]
        assert type(seed) is int
        assert 0 <= seed <= 2**31 - 1
        message_text = "\n".join(
            message["content"] for message in body["messages"]
        )
        assert "json" in message_text.casefold()
        assert "enforce_distillable_text" not in body["provider"]


def test_v3_disables_default_reasoning_only_on_cost_risk_routes() -> None:
    bodies = {
        model.route_id: neutral_story_request_body(
            _brief(),
            model,
            provider_slug=model.first_party_provider_slug or "fixture-provider",
            provider_quantization="unknown",
            prompt_usd_per_token="0.00000015",
            completion_usd_per_token="0.0000006",
            request_contract=SYNTHETIC_STORY_REQUEST_V3,
        )
        for model in CANDIDATE_MODELS
    }

    assert bodies["qwen3.5-35b-a3b"]["reasoning"] == {"effort": "none"}
    assert bodies["gemini-3.1-flash-lite"]["reasoning"] == {"effort": "none"}
    assert all(
        "reasoning" not in body
        for route_id, body in bodies.items()
        if route_id not in {"qwen3.5-35b-a3b", "gemini-3.1-flash-lite"}
    )


def test_v4_is_story_only_signed_31_bit_and_disables_both_authors_reasoning() -> None:
    for model in TWO_ROUTE_AUTHOR_MODELS:
        body = neutral_story_request_body(
            _brief(),
            model,
            provider_slug=model.first_party_provider_slug or "fixture-provider",
            provider_quantization="unknown",
            prompt_usd_per_token="0.00000015",
            completion_usd_per_token="0.0000006",
            request_contract=SYNTHETIC_STORY_REQUEST_V4,
        )

        assert 0 <= body["seed"] <= 2**31 - 1
        assert "enforce_distillable_text" not in body["provider"]
        assert body["reasoning"] == {"effort": "none"}
        assert body["response_format"]["json_schema"] == {
            "name": "tinyworlds_v2_neutral_story_v2",
            "schema": {
                "additionalProperties": False,
                "properties": {"story": {"minLength": 1, "type": "string"}},
                "required": ["story"],
                "type": "object",
            },
            "strict": True,
        }
        user_prompt = body["messages"][1]["content"]
        assert "only the story field" in user_prompt
        assert "exact quotes" not in user_prompt


def test_v4_through_v6_prompt_bytes_stay_frozen() -> None:
    bodies = {
        contract.version: neutral_story_request_body(
            _brief(),
            TWO_ROUTE_AUTHOR_MODELS[0],
            provider_slug="alibaba",
            provider_quantization="unknown",
            prompt_usd_per_token="0.00000015",
            completion_usd_per_token="0.0000006",
            request_contract=contract,
        )
        for contract in (
            SYNTHETIC_STORY_REQUEST_V4,
            SYNTHETIC_STORY_REQUEST_V5,
            SYNTHETIC_STORY_REQUEST_V6,
        )
    }
    control, length_only, reference_shape = (
        bodies[contract.version]["messages"][0]["content"]
        for contract in (
            SYNTHETIC_STORY_REQUEST_V4,
            SYNTHETIC_STORY_REQUEST_V5,
            SYNTHETIC_STORY_REQUEST_V6,
        )
    )

    assert tuple(request_body_sha256(body) for body in bodies.values()) == (
        "7fc6b570405bdd089c2b7a0b412eb9f17042000e5bf78bac7aa067896bf6e57a",
        "2955b315c404391320f2092f16204d975bf9dfbb1ca466f6403b9c1ccae9db3e",
        "cc9f14f6ee1ed20909282a5ff4746ae7922d32756a1d2bb66f1389eea0643c49",
    )
    assert tuple(body["messages"][1]["content"] for body in bodies.values()) == (
        "RELEASED TINYSTORIES INSTRUCTION:\n"
        "Write a simple story using moon, jump, and kind, with dialogue.\n\n"
        "REQUIRED WORDS: moon, jump, kind\n"
        "REQUESTED NARRATIVE FEATURES: Dialogue\n\n"
        "Return exactly one JSON object containing only the story field. "
        "Do not return evidence, analysis, commentary, or any other field.",
    ) * 3
    assert "130 to 170 words" not in control
    assert "130 to 170 words" in length_only
    assert "single line breaks" not in length_only
    assert "130 to 170 words" in reference_shape
    assert "single line breaks" in reference_shape
    assert "Once upon a time" in reference_shape
    assert len(set(map(request_body_sha256, bodies.values()))) == 3
    assert all(
        body["reasoning"] == {"effort": "none"}
        and body["response_format"]["json_schema"]["name"]
        == "tinyworlds_v2_neutral_story_v2"
        for body in bodies.values()
    )


def test_v7_moves_structural_requirements_to_the_end_of_the_user_prompt() -> None:
    body = neutral_story_request_body(
        _brief(),
        TWO_ROUTE_AUTHOR_MODELS[0],
        provider_slug="alibaba",
        provider_quantization="unknown",
        prompt_usd_per_token="0.00000015",
        completion_usd_per_token="0.0000006",
        request_contract=SYNTHETIC_STORY_REQUEST_V7,
    )
    system_prompt = body["messages"][0]["content"]
    user_prompt = body["messages"][1]["content"]

    assert "one short" not in system_prompt
    assert "3- to 4-year-old" not in system_prompt
    assert "gentle and suitable for young children" in system_prompt
    assert "supplied released TinyStories instruction" in system_prompt
    assert "ordinary story prose only" in system_prompt
    assert "Do not mention prompts" in system_prompt
    assert system_prompt.endswith(
        "Return exactly one JSON object matching the supplied response format."
    )
    assert user_prompt.index("only the story field") < user_prompt.index(
        "FINAL STORY REQUIREMENTS"
    )
    assert "exactly as written" in user_prompt
    assert "moon, jump, kind" in user_prompt
    assert "standard ASCII double quotation marks" in user_prompt
    assert 'like "Hello."' in user_prompt
    assert "one continuous story-field text block with no newline" in user_prompt
    assert "18 to 20 complete sentences" in user_prompt
    assert "mostly 7 to 11 words" in user_prompt
    assert "at least 6 connected events" in user_prompt
    assert "155 to 190 words" in user_prompt
    assert "soft target" in user_prompt
    assert "natural, simple repetition" in user_prompt
    assert body["reasoning"] == {"effort": "none"}


def test_v7_dialogue_requirement_is_conditional() -> None:
    body = neutral_story_request_body(
        replace(
            _brief(),
            brief_id="brief-without-dialogue",
            prompt_text="Write a simple story using moon, jump, and kind.",
            requested_features=(),
        ),
        TWO_ROUTE_AUTHOR_MODELS[1],
        provider_slug="openai",
        provider_quantization="unknown",
        prompt_usd_per_token="0.00000075",
        completion_usd_per_token="0.0000045",
        request_contract=SYNTHETIC_STORY_REQUEST_V7,
    )

    assert "Because Dialogue is requested" not in body["messages"][1]["content"]


def test_v8_is_exactly_the_released_prompt_plus_continuation_cue() -> None:
    body = neutral_story_request_body(
        _brief(),
        TWO_ROUTE_AUTHOR_MODELS[0],
        provider_slug="alibaba",
        provider_quantization="unknown",
        prompt_usd_per_token="0.00000015",
        completion_usd_per_token="0.0000006",
        request_contract=SYNTHETIC_STORY_REQUEST_V8,
    )

    assert body["messages"] == [
        {
            "role": "user",
            "content": (
                "Write a simple story using moon, jump, and kind, with dialogue."
                "\n\nPossible story:"
            ),
        }
    ]
    assert "response_format" not in body
    assert body["reasoning"] == {"effort": "none"}
    assert body["plugins"] == []
    assert body["transforms"] == []
    assert body["provider"]["allow_fallbacks"] is False
    assert body["provider"]["data_collection"] == "deny"


def test_v9_differs_from_v8_only_by_the_single_length_cue() -> None:
    bodies = tuple(
        neutral_story_request_body(
            _brief(),
            TWO_ROUTE_AUTHOR_MODELS[0],
            provider_slug="alibaba",
            provider_quantization="unknown",
            prompt_usd_per_token="0.00000015",
            completion_usd_per_token="0.0000006",
            request_contract=contract,
        )
        for contract in (SYNTHETIC_STORY_REQUEST_V8, SYNTHETIC_STORY_REQUEST_V9)
    )
    bare_body, length_body = bodies

    assert length_body["messages"] == [
        {
            "role": "user",
            "content": (
                "Write a simple story using moon, jump, and kind, with dialogue."
                "\n\nAim for about 130 to 150 words.\n\nPossible story:"
            ),
        }
    ]
    assert {
        key: value for key, value in bare_body.items() if key != "messages"
    } == {
        key: value for key, value in length_body.items() if key != "messages"
    }


def test_plain_text_story_preserves_the_complete_reply_and_derives_evidence() -> None:
    story = (
        'Mia saw the Moon above her little house. "Come with me," she said.\n'
        "Her kind friend Ben came outside. They wanted to jump over a puddle, "
        "but first they helped a wet frog. The frog was safe, and the two "
        "friends went home smiling together under the moon."
    )

    payload, validation = validate_plain_text_generated_story(_brief(), story)

    assert payload is not None
    assert payload.story == story
    assert validation.accepted
    assert tuple(span.ingredient for span in payload.required_word_spans) == (
        "moon",
        "jump",
        "kind",
    )
    assert payload.realized_features == ("Dialogue",)


@pytest.mark.parametrize(
    ("brief_id", "expected_opening_requirement"),
    (
        (
            "opening-000",
            'Use another simple opening; do not start with "Once upon a time" '
            'or "One day".',
        ),
        ("opening-001", 'Start the story with exactly "Once upon a time".'),
        ("opening-002", 'Start the story with exactly "One day".'),
    ),
)
def test_v7_opening_is_deterministic_and_hash_balanced(
    brief_id: str,
    expected_opening_requirement: str,
) -> None:
    brief = replace(_brief(), brief_id=brief_id)
    request_bodies = tuple(
        neutral_story_request_body(
            brief,
            TWO_ROUTE_AUTHOR_MODELS[0],
            provider_slug="alibaba",
            provider_quantization="unknown",
            prompt_usd_per_token="0.00000015",
            completion_usd_per_token="0.0000006",
            request_contract=SYNTHETIC_STORY_REQUEST_V7,
        )
        for _ in range(2)
    )

    assert request_bodies[0] == request_bodies[1]
    assert request_bodies[0]["messages"][1]["content"].endswith(
        expected_opening_requirement
    )


def test_v1_through_v3_generation_hashes_remain_frozen() -> None:
    brief = replace(
        _brief(),
        brief_id="preview-brief-a0cd61e7c4ebc9cd571ced1f",
    )
    model = CANDIDATE_MODELS[4]
    hashes = tuple(
        request_body_sha256(
            neutral_story_request_body(
                brief,
                model,
                provider_slug="alibaba",
                provider_quantization="unknown",
                prompt_usd_per_token="0.00000015",
                completion_usd_per_token="0.0000006",
                request_contract=contract,
            )
        )
        for contract in (
            GENERATION_REQUEST_V1,
            SYNTHETIC_STORY_REQUEST_V2,
            SYNTHETIC_STORY_REQUEST_V3,
        )
    )

    assert hashes == (
        "aa8e468f40144d4bbf16497eb94b6e47f9a306c971c959372286a248e8c2429c",
        "712371b2bfc63d8283fa7f6bd0866852771ff14d39d171a1c0e4621b3c3ace33",
        "d9865941fe1b52eb56b3146d6ce9d466e56b51e9b9188281ae23a4d00dbb5fe0",
    )


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


def test_story_only_payload_derives_spans_and_features_from_text() -> None:
    story = (
        'Mia saw the Moon above her little house. "Come with me," she said. '
        "Her kind friend Ben came outside. They wanted to jump over a puddle, "
        "but first they helped a wet frog. The frog was safe, and the two "
        "friends went home smiling together under the moon."
    )
    payload, validation = validate_story_only_generated_story(
        _brief(),
        json.dumps({"story": story}),
    )

    assert validation.accepted
    assert payload is not None
    assert tuple(span.ingredient for span in payload.required_word_spans) == (
        "moon",
        "jump",
        "kind",
    )
    assert payload.required_word_spans[0].exact_text == "Moon"
    assert story[
        payload.required_word_spans[0].start : payload.required_word_spans[0].end
    ] == "Moon"
    assert payload.realized_features == ("Dialogue",)


def test_story_only_payload_rejects_extra_fields_and_missing_local_feature() -> None:
    story = (
        "Mia saw the moon above her little house. Her kind friend Ben came "
        "outside. They wanted to jump over a puddle, but first they helped a "
        "wet frog. The frog was safe, and the two friends went home smiling "
        "together under the bright moon."
    )
    payload, extra = validate_story_only_generated_story(
        _brief(),
        json.dumps({"story": story, "word_evidence": []}),
    )
    assert payload is None
    assert extra.rejection_reasons == ("schema_invalid",)

    payload, missing_feature = validate_story_only_generated_story(
        _brief(),
        json.dumps({"story": story}),
    )
    assert payload is not None
    assert not missing_feature.accepted
    assert missing_feature.required_words_present
    assert "evidence_invalid" in missing_feature.rejection_reasons


def test_story_only_payload_reports_but_does_not_hard_gate_semantic_features() -> None:
    brief = NeutralStoryBrief(
        brief_id="brief-semantic-feature",
        source_record_id="source-semantic-feature",
        prompt_text="Write a story with a moral using moon, jump, and kind.",
        required_words=("moon", "jump", "kind"),
        requested_features=("MoralValue", "Foreshadowing"),
        matched_reference_text="A reference story.",
    )
    story = (
        "Mia saw the moon above her little house. Her kind friend Ben came "
        "outside. They wanted to jump over a puddle, so they helped a wet "
        "frog first. The frog was safe. The friends smiled and walked home "
        "together under the bright moon after their happy day."
    )

    payload, validation = validate_story_only_generated_story(
        brief,
        json.dumps({"story": story}),
    )

    assert payload is not None
    assert payload.realized_features == ()
    assert validation.evidence_valid
    assert validation.accepted


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

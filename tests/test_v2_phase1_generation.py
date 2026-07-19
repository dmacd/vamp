from __future__ import annotations

import json

from apm.data.text.tinyworlds_v2.bakeoff import (
    CANDIDATE_MODELS,
    NeutralStoryBrief,
    VERIFIER_MODEL,
    VerifierPayload,
)
from apm.data.text.tinyworlds_v2.generation_schema import (
    RawHttpResponse,
    ResponseProvenance,
    RouteLock,
    TokenUsage,
)
from apm.data.text.tinyworlds_v2.phase1_generation import (
    build_generation_jobs,
    build_verifier_job,
    execute_generation_jobs,
    execute_verifier_jobs,
    generated_observation,
    VerifiedStory,
)


def _route(route_id: str, model: str, canonical: str) -> RouteLock:
    return RouteLock(
        route_id=route_id,
        catalog_sha256="a" * 64,
        requested_model=model,
        canonical_model=canonical,
        provider_slug="provider",
        returned_provider="Provider",
        quantization="bf16",
        input_usd_per_million="0.1",
        output_usd_per_million="0.2",
    )


def _brief(index: int = 0) -> NeutralStoryBrief:
    return NeutralStoryBrief(
        f"brief-{index}",
        f"source-{index}",
        'Write a story using the verb "jump", noun "moon", and adjective "kind".',
        ("jump", "moon", "kind"),
        (),
        "A genuine matched story.",
    )


def _story_content() -> str:
    story = (
        "Mia saw the moon over her small home. Mia was kind to a little wet bird. "
        "She told the bird to jump onto her hand, but it was too tired. Mia gave "
        "it water and sat with it by the tree. Soon the bird could jump and fly. "
        "Mia went home with a happy smile under the moon."
    )
    return json.dumps(
        {
            "feature_evidence": [],
            "story": story,
            "word_evidence": [
                {"exact_quote": "jump", "required_word": "jump"},
                {"exact_quote": "moon", "required_word": "moon"},
                {"exact_quote": "kind", "required_word": "kind"},
            ],
        }
    )


class _Client:
    def __init__(self, content: str) -> None:
        self.content = content

    def generate(self, request, route_lock):
        body = json.dumps(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": self.content, "role": "assistant"},
                    }
                ],
                "id": "generation-1",
                "model": route_lock.canonical_model,
                "provider": route_lock.returned_provider,
                "usage": {
                    "completion_tokens": 30,
                    "cost": 0.001,
                    "prompt_tokens": 20,
                    "total_tokens": 50,
                },
            }
        ).encode()
        return RawHttpResponse(
            200,
            (),
            body,
            ResponseProvenance(
                "generation-1",
                route_lock.requested_model,
                route_lock.canonical_model,
                route_lock.returned_provider,
            ),
            TokenUsage(20, 30, 50, 0),
            billed_cost_usd="0.001",
        )


def test_generation_jobs_are_route_major_and_restore_concurrent_order() -> None:
    models = CANDIDATE_MODELS[:2]
    routes = tuple(_route(model.route_id, model.request_model_id, model.canonical_slug) for model in models)
    jobs = build_generation_jobs((_brief(0), _brief(1)), models, routes)
    samples = execute_generation_jobs(jobs, _Client(_story_content()), max_workers=3)

    assert [sample.sample_id for sample in samples] == [
        f"{models[0].route_id}:brief-0",
        f"{models[0].route_id}:brief-1",
        f"{models[1].route_id}:brief-0",
        f"{models[1].route_id}:brief-1",
    ]
    assert all(sample.validation.accepted for sample in samples)
    observation = generated_observation(
        samples[0],
        model_token_ids=(1, 2, 3),
        normalized_nll=2.0,
    )
    assert observation.sample_id == "brief-0"
    assert observation.required_verb_ok
    assert observation.required_noun_ok
    assert observation.required_adjective_ok


def test_verifier_failure_becomes_explicit_zero_score_full_observation() -> None:
    model = CANDIDATE_MODELS[0]
    route = _route(model.route_id, model.request_model_id, model.canonical_slug)
    sample = execute_generation_jobs(
        build_generation_jobs((_brief(),), (model,), (route,)),
        _Client(_story_content()),
        max_workers=1,
    )[0]
    verifier_route = _route(
        VERIFIER_MODEL.route_id,
        VERIFIER_MODEL.request_model_id,
        VERIFIER_MODEL.canonical_slug,
    )
    job = build_verifier_job(
        source_id=sample.sample_id,
        pair_id=sample.job.brief.brief_id,
        brief=sample.job.brief,
        story=sample.payload.story,
        model=VERIFIER_MODEL,
        route=verifier_route,
    )
    verified = execute_verifier_jobs((job,), _Client("not-json"), max_workers=1)[0]
    observation = generated_observation(
        sample,
        model_token_ids=(1, 2),
        normalized_nll=2.0,
        verifier=verified,
        verifier_required=True,
    )

    assert dict(observation.blind_verifier_scores or ()) == {
        "grammar": 0.0,
        "non_repetition": 0.0,
        "plot_coherence": 0.0,
        "preschool_vocabulary": 0.0,
        "sentence_simplicity": 0.0,
    }
    assert observation.blind_verifier_hard_failure


def test_feature_realization_is_text_mechanical_not_verifier_adherence() -> None:
    model = CANDIDATE_MODELS[0]
    route = _route(model.route_id, model.request_model_id, model.canonical_slug)
    brief = NeutralStoryBrief(
        "brief-dialogue",
        "source-dialogue",
        (
            'Write a story using the verb "jump", noun "moon", and adjective '
            '"kind", with dialogue.'
        ),
        ("jump", "moon", "kind"),
        ("Dialogue",),
        '"Jump," said the kind child under the moon.',
    )
    content = json.dumps(
        {
            "feature_evidence": [
                {"exact_quote": '"Please jump,"', "feature": "Dialogue"}
            ],
            "story": (
                'Mia saw the moon. She was kind. "Please jump," she told her '
                "little frog. The frog made one jump, but then it felt tired. "
                "Mia brought a soft leaf and a small cup of water. They rested "
                "together by the pond until the stars came out. Soon the frog "
                "felt well, and they went home happy."
            ),
            "word_evidence": [
                {"exact_quote": "jump", "required_word": "jump"},
                {"exact_quote": "moon", "required_word": "moon"},
                {"exact_quote": "kind", "required_word": "kind"},
            ],
        }
    )
    sample = execute_generation_jobs(
        build_generation_jobs((brief,), (model,), (route,)),
        _Client(content),
        max_workers=1,
    )[0]
    assert sample.validation.accepted
    verifier_route = _route(
        VERIFIER_MODEL.route_id,
        VERIFIER_MODEL.request_model_id,
        VERIFIER_MODEL.canonical_slug,
    )
    verifier_job = build_verifier_job(
        source_id=sample.sample_id,
        pair_id=brief.brief_id,
        brief=brief,
        story=sample.payload.story,
        model=VERIFIER_MODEL,
        route=verifier_route,
    )
    verifier = VerifiedStory(
        verifier_job,
        VerifierPayload(4, 4, 4, 4, 4, False, (), "The story is simple."),
        "verifier-generation",
        "0.001",
        None,
    )

    observation = generated_observation(
        sample,
        model_token_ids=(1, 2, 3),
        normalized_nll=1.0,
        verifier=verifier,
        verifier_required=True,
    )

    assert observation.required_feature_ok
    assert observation.feature_labels == ("Dialogue",)
    assert not verifier.payload.brief_adherence

"""Deterministic job construction and cached execution for the Phase 1 bakeoff."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol, TypeVar

from apm.data.text.tinyworlds_v2.bakeoff import (
    CandidateModelSpec,
    GeneratedStoryPayload,
    NeutralStoryBrief,
    StoryValidation,
    VerifierPayload,
    assistant_message_content,
    neutral_story_request_body,
    parse_verifier_payload,
    validate_generated_story,
    verifier_request_body,
)
from apm.data.text.tinyworlds_v2.generation_schema import (
    CanonicalRequest,
    RawHttpResponse,
    RouteLock,
)
from apm.data.text.tinyworlds_v2.json_contracts import JsonObject
from apm.data.text.tinyworlds_v2.ingredients import (
    mechanically_classify_ingredient_roles,
)
from apm.data.text.tinyworlds_v2.openrouter import (
    OpenRouterContractError,
    OpenRouterError,
)
from apm.data.text.tinyworlds_v2.quality import (
    BLIND_VERIFIER_DIMENSIONS,
    GeneratedObservation,
)
from apm.data.text.tinyworlds_v2.reference_profile import (
    ReferenceRecord,
    observe_reference,
)
from apm.data.text.tinyworlds_v2.surface import realized_feature_labels


CHAT_COMPLETIONS_ENDPOINT = "/api/v1/chat/completions"


class CachedGenerationClient(Protocol):
    """Structural boundary implemented by ``OpenRouterClient`` and test fakes."""

    def generate(
        self,
        request: CanonicalRequest,
        route_lock: RouteLock,
    ) -> RawHttpResponse:
        """Return one validated cached or fresh response."""


@dataclass(frozen=True, slots=True)
class GenerationJob:
    """One neutral brief bound to one exact route and canonical request."""

    brief: NeutralStoryBrief
    model: CandidateModelSpec
    route: RouteLock
    request: CanonicalRequest

    def __post_init__(self) -> None:
        if type(self.brief) is not NeutralStoryBrief:
            raise TypeError("generation job brief must be NeutralStoryBrief")
        if type(self.model) is not CandidateModelSpec:
            raise TypeError("generation job model must be CandidateModelSpec")
        if type(self.route) is not RouteLock:
            raise TypeError("generation job route must be RouteLock")
        if type(self.request) is not CanonicalRequest:
            raise TypeError("generation job request must be CanonicalRequest")
        if self.model.route_id != self.route.route_id:
            raise ValueError("generation model and route IDs differ")

    @property
    def sample_id(self) -> str:
        """Return the stable semantic observation identity across funnel stages."""
        return f"{self.route.route_id}:{self.brief.brief_id}"


@dataclass(frozen=True, slots=True)
class GeneratedSample:
    """One immutable response interpretation backed by the raw response cache."""

    job: GenerationJob
    payload: GeneratedStoryPayload | None
    validation: StoryValidation
    generation_id: str | None
    input_tokens: int
    output_tokens: int
    billed_cost_usd: str
    error_kind: str | None

    def __post_init__(self) -> None:
        if type(self.job) is not GenerationJob:
            raise TypeError("generated sample job must be GenerationJob")
        if self.payload is not None and type(self.payload) is not GeneratedStoryPayload:
            raise TypeError("generated sample payload has the wrong type")
        if type(self.validation) is not StoryValidation:
            raise TypeError("generated sample validation has the wrong type")
        if self.validation.accepted and (
            self.payload is None or self.error_kind is not None
        ):
            raise ValueError("accepted samples require a payload and no execution error")
        if not self.validation.accepted and self.error_kind is None:
            raise ValueError("rejected samples require an explicit error kind")
        if self.generation_id is not None and (
            type(self.generation_id) is not str or not self.generation_id
        ):
            raise ValueError("generation ID must be nonempty when supplied")
        if any(type(value) is not int or value < 0 for value in (self.input_tokens, self.output_tokens)):
            raise ValueError("generation token counts must be nonnegative")
        cost = Decimal(self.billed_cost_usd)
        if not cost.is_finite() or cost < 0:
            raise ValueError("generation billed cost must be finite and nonnegative")
        if self.error_kind is not None and (
            type(self.error_kind) is not str or not self.error_kind
        ):
            raise ValueError("generation error kind must be nonempty")

    @property
    def sample_id(self) -> str:
        """Return the stable route/brief identity."""
        return self.job.sample_id

    def as_record(self) -> JsonObject:
        """Return the derived observation without duplicating raw response bytes."""
        payload: JsonObject | None = None
        if self.payload is not None:
            payload = {
                "feature_evidence": [
                    {
                        "exact_quote": item.exact_quote,
                        "feature": item.ingredient,
                    }
                    for item in self.payload.feature_evidence
                ],
                "story": self.payload.story,
                "word_evidence": [
                    {
                        "exact_quote": item.exact_quote,
                        "required_word": item.ingredient,
                    }
                    for item in self.payload.word_evidence
                ],
            }
        return {
            "billed_cost_usd": self.billed_cost_usd,
            "brief_id": self.job.brief.brief_id,
            "error_kind": self.error_kind,
            "generation_id": self.generation_id,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "payload": payload,
            "request_sha256": self.job.request.request_sha256,
            "route_id": self.job.route.route_id,
            "sample_id": self.sample_id,
            "validation": {
                "accepted": self.validation.accepted,
                "evidence_valid": self.validation.evidence_valid,
                "forbidden_identifier_present": self.validation.forbidden_identifier_present,
                "length_valid": self.validation.length_valid,
                "rejection_reasons": list(self.validation.rejection_reasons),
                "required_words_present": self.validation.required_words_present,
                "schema_valid": self.validation.schema_valid,
                "story_sha256": self.validation.story_sha256,
            },
        }


@dataclass(frozen=True, slots=True)
class VerifierJob:
    """One source-blind story assessment bound to the exact verifier route."""

    source_id: str
    pair_id: str
    brief: NeutralStoryBrief
    story: str
    route: RouteLock
    request: CanonicalRequest

    def __post_init__(self) -> None:
        for label, value in (("source_id", self.source_id), ("pair_id", self.pair_id), ("story", self.story)):
            if type(value) is not str or not value:
                raise ValueError(f"verifier {label} must be nonempty")
        if type(self.brief) is not NeutralStoryBrief:
            raise TypeError("verifier brief must be NeutralStoryBrief")
        if type(self.route) is not RouteLock or type(self.request) is not CanonicalRequest:
            raise TypeError("verifier route and request have the wrong types")


@dataclass(frozen=True, slots=True)
class VerifiedStory:
    """One strict verifier interpretation or an explicit failed observation."""

    job: VerifierJob
    payload: VerifierPayload | None
    generation_id: str | None
    billed_cost_usd: str
    error_kind: str | None

    def __post_init__(self) -> None:
        if type(self.job) is not VerifierJob:
            raise TypeError("verified story job must be VerifierJob")
        if self.payload is not None and type(self.payload) is not VerifierPayload:
            raise TypeError("verified story payload has the wrong type")
        if (self.payload is None) != (self.error_kind is not None):
            raise ValueError("verifier failures must be explicit")
        cost = Decimal(self.billed_cost_usd)
        if not cost.is_finite() or cost < 0:
            raise ValueError("verifier billed cost must be finite and nonnegative")

    def as_record(self) -> JsonObject:
        """Return the source-blind derived verifier observation."""
        payload: JsonObject | None = None
        if self.payload is not None:
            payload = {
                "brief_adherence": self.payload.brief_adherence,
                "grammar": self.payload.grammar,
                "hard_failures": list(self.payload.hard_failures),
                "mean_score": self.payload.mean_score,
                "non_repetition": self.payload.non_repetition,
                "plot_coherence": self.payload.plot_coherence,
                "preschool_vocabulary": self.payload.preschool_vocabulary,
                "rationale": self.payload.rationale,
                "sentence_simplicity": self.payload.sentence_simplicity,
            }
        return {
            "billed_cost_usd": self.billed_cost_usd,
            "error_kind": self.error_kind,
            "generation_id": self.generation_id,
            "pair_id": self.job.pair_id,
            "payload": payload,
            "request_sha256": self.job.request.request_sha256,
            "source_id": self.job.source_id,
        }


def build_generation_jobs(
    briefs: Sequence[NeutralStoryBrief],
    models: Sequence[CandidateModelSpec],
    routes: Sequence[RouteLock],
) -> tuple[GenerationJob, ...]:
    """Create route-major jobs so every route receives identical brief order."""
    if not briefs or not models or not routes:
        raise ValueError("generation jobs require briefs, models, and routes")
    model_by_id = {model.route_id: model for model in models}
    route_by_id = {route.route_id: route for route in routes}
    if len(model_by_id) != len(models) or len(route_by_id) != len(routes):
        raise ValueError("generation models and routes must have unique IDs")
    if tuple(model_by_id) != tuple(route_by_id):
        raise ValueError("generation model and route orders must match exactly")
    return tuple(
        _generation_job(brief, model, route_by_id[model.route_id])
        for model in models
        for brief in briefs
    )


def build_verifier_job(
    *,
    source_id: str,
    pair_id: str,
    brief: NeutralStoryBrief,
    story: str,
    model: CandidateModelSpec,
    route: RouteLock,
) -> VerifierJob:
    """Build one canonical source-blind verifier request."""
    body = verifier_request_body(
        brief,
        story,
        provider_slug=route.provider_slug,
        provider_quantization=route.quantization,
        prompt_usd_per_token=_per_token(route.input_usd_per_million),
        completion_usd_per_token=_per_token(route.output_usd_per_million),
    )
    if model.route_id != route.route_id:
        raise ValueError("verifier model and route IDs differ")
    request = CanonicalRequest.from_body(
        route_lock_sha256=route.lock_sha256,
        endpoint=CHAT_COMPLETIONS_ENDPOINT,
        body=body,
    )
    return VerifierJob(source_id, pair_id, brief, story, route, request)


def execute_generation_jobs(
    jobs: Sequence[GenerationJob],
    client: CachedGenerationClient,
    *,
    max_workers: int = 8,
    progress_callback: Callable[[int], None] | None = None,
) -> tuple[GeneratedSample, ...]:
    """Execute cached generation jobs concurrently and restore canonical order."""
    return _execute_ordered(
        jobs,
        lambda job: _execute_generation_job(job, client),
        max_workers=max_workers,
        progress_callback=progress_callback,
    )


def execute_verifier_jobs(
    jobs: Sequence[VerifierJob],
    client: CachedGenerationClient,
    *,
    max_workers: int = 8,
    progress_callback: Callable[[int], None] | None = None,
) -> tuple[VerifiedStory, ...]:
    """Execute cached verifier jobs concurrently and restore canonical order."""
    return _execute_ordered(
        jobs,
        lambda job: _execute_verifier_job(job, client),
        max_workers=max_workers,
        progress_callback=progress_callback,
    )


def generated_observation(
    sample: GeneratedSample,
    *,
    model_token_ids: tuple[int, ...] = (),
    normalized_nll: float | None = None,
    verifier: VerifiedStory | None = None,
    verifier_required: bool = False,
) -> GeneratedObservation:
    """Project one cached result into the fixed reference-quality contract."""
    accepted = sample.validation.accepted and sample.payload is not None
    if accepted and (not model_token_ids or normalized_nll is None):
        raise ValueError("accepted generated samples require tokenizer and NLL evidence")
    if verifier_required and accepted and verifier is None:
        raise ValueError("full generated observations require a verifier result")
    if accepted:
        assert sample.payload is not None
        surface = observe_reference(
            ReferenceRecord(sample.sample_id, sample.payload.story, source_model="generated"),
            model_token_ids=model_token_ids,
            normalized_nll=float(normalized_nll),
        )
        tokens = frozenset(surface.word_tokens)
        roles = mechanically_classify_ingredient_roles(
            sample.job.brief.prompt_text,
            sample.job.brief.required_words,
        )
        if roles is None:
            raise ValueError("generated brief has ambiguous ingredient roles")
        noun_ok, verb_ok, adjective_ok = (
            word.casefold() in tokens
            for word in (roles.noun, roles.verb, roles.adjective)
        )
        verifier_payload = None if verifier is None else verifier.payload
        realized_features = realized_feature_labels(
            sample.payload.story,
            sample.job.brief.requested_features,
        )
        # This gate is deliberately text-mechanical.  Copied evidence is part
        # of deterministic acceptance, and the blind verifier's adherence
        # judgment is reported separately; neither may manufacture or erase a
        # feature realization found in the story itself.
        feature_ok = set(sample.job.brief.requested_features).issubset(
            realized_features
        )
        verifier_scores = (
            None
            if not verifier_required
            else tuple((dimension, 0.0) for dimension in BLIND_VERIFIER_DIMENSIONS)
            if verifier_payload is None
            else tuple(
                (dimension, float(getattr(verifier_payload, dimension)))
                for dimension in BLIND_VERIFIER_DIMENSIONS
            )
        )
        verifier_failure = verifier_required and (
            verifier_payload is None or bool(verifier_payload.hard_failures)
        )
        verifier_cost = Decimal("0") if verifier is None else Decimal(verifier.billed_cost_usd)
        return GeneratedObservation(
            sample_id=sample.job.brief.brief_id,
            route_id=sample.job.route.route_id,
            schema_valid=sample.validation.schema_valid,
            deterministic_accepted=True,
            required_noun_ok=noun_ok,
            required_verb_ok=verb_ok,
            required_adjective_ok=adjective_ok,
            required_feature_ok=feature_ok,
            forbidden_identifier_found=sample.validation.forbidden_identifier_present,
            word_tokens=surface.word_tokens,
            model_token_ids=surface.model_token_ids,
            sentence_word_counts=surface.sentence_word_counts,
            paragraph_count=surface.paragraph_count,
            dialogue_present=surface.dialogue_present,
            feature_labels=realized_features,
            normalized_nll=surface.normalized_nll,
            blind_verifier_scores=verifier_scores,
            blind_verifier_hard_failure=verifier_failure,
            billed_cost_usd=float(Decimal(sample.billed_cost_usd) + verifier_cost),
            requested_feature_labels=sample.job.brief.requested_features,
        )
    return GeneratedObservation(
        sample_id=sample.job.brief.brief_id,
        route_id=sample.job.route.route_id,
        schema_valid=sample.validation.schema_valid,
        deterministic_accepted=False,
        required_noun_ok=False,
        required_verb_ok=False,
        required_adjective_ok=False,
        required_feature_ok=False,
        forbidden_identifier_found=sample.validation.forbidden_identifier_present,
        word_tokens=(),
        model_token_ids=(),
        sentence_word_counts=(),
        paragraph_count=0,
        dialogue_present=False,
        feature_labels=(),
        normalized_nll=None,
        blind_verifier_scores=None,
        blind_verifier_hard_failure=verifier_required,
        billed_cost_usd=float(Decimal(sample.billed_cost_usd)),
        requested_feature_labels=sample.job.brief.requested_features,
    )


def _generation_job(
    brief: NeutralStoryBrief,
    model: CandidateModelSpec,
    route: RouteLock,
) -> GenerationJob:
    body = neutral_story_request_body(
        brief,
        model,
        provider_slug=route.provider_slug,
        provider_quantization=route.quantization,
        prompt_usd_per_token=_per_token(route.input_usd_per_million),
        completion_usd_per_token=_per_token(route.output_usd_per_million),
    )
    request = CanonicalRequest.from_body(
        route_lock_sha256=route.lock_sha256,
        endpoint=CHAT_COMPLETIONS_ENDPOINT,
        body=body,
    )
    return GenerationJob(brief, model, route, request)


def _execute_generation_job(
    job: GenerationJob,
    client: CachedGenerationClient,
) -> GeneratedSample:
    try:
        response = client.generate(job.request, job.route)
    except OpenRouterContractError:
        raise
    except OpenRouterError as error:
        _, validation = validate_generated_story(job.brief, b"")
        return GeneratedSample(job, None, validation, None, 0, 0, "0", type(error).__name__)
    assert response.provenance is not None
    assert response.usage is not None
    try:
        content = assistant_message_content(response.body)
        payload, validation = validate_generated_story(job.brief, content)
        error_kind = None if validation.accepted else "deterministic_rejection"
    except (TypeError, ValueError):
        payload, validation = validate_generated_story(job.brief, b"")
        error_kind = "completion_contract_invalid"
    return GeneratedSample(
        job=job,
        payload=payload,
        validation=validation,
        generation_id=response.provenance.generation_id,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        billed_cost_usd=response.billed_cost_usd or "0",
        error_kind=error_kind,
    )


def _execute_verifier_job(
    job: VerifierJob,
    client: CachedGenerationClient,
) -> VerifiedStory:
    try:
        response = client.generate(job.request, job.route)
    except OpenRouterContractError:
        raise
    except OpenRouterError as error:
        return VerifiedStory(job, None, None, "0", type(error).__name__)
    assert response.provenance is not None
    assert response.usage is not None
    try:
        payload = parse_verifier_payload(assistant_message_content(response.body))
    except (TypeError, ValueError):
        return VerifiedStory(
            job,
            None,
            response.provenance.generation_id,
            response.billed_cost_usd or "0",
            "verifier_contract_invalid",
        )
    return VerifiedStory(
        job,
        payload,
        response.provenance.generation_id,
        response.billed_cost_usd or "0",
        None,
    )


JobT = TypeVar("JobT")
ResultT = TypeVar("ResultT")


def _execute_ordered(
    jobs: Sequence[JobT],
    operation: Callable[[JobT], ResultT],
    *,
    max_workers: int,
    progress_callback: Callable[[int], None] | None,
) -> tuple[ResultT, ...]:
    if not jobs:
        return ()
    if type(max_workers) is not int or max_workers <= 0:
        raise ValueError("generation max_workers must be positive")
    ordered: list[ResultT | None] = [None] * len(jobs)
    executor = ThreadPoolExecutor(max_workers=max_workers)
    futures = {}
    try:
        futures = {
            executor.submit(operation, job): index for index, job in enumerate(jobs)
        }
        for future in as_completed(futures):
            ordered[futures[future]] = future.result()
            if progress_callback is not None:
                progress_callback(1)
    except Exception:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    if any(item is None for item in ordered):
        raise RuntimeError("generation executor lost a job result")
    return tuple(item for item in ordered if item is not None)


def _per_token(per_million: str) -> str:
    return format((Decimal(per_million) / Decimal(1_000_000)).normalize(), "f")


__all__ = [
    "CHAT_COMPLETIONS_ENDPOINT",
    "GeneratedSample",
    "GenerationJob",
    "VerifiedStory",
    "VerifierJob",
    "build_generation_jobs",
    "build_verifier_job",
    "execute_generation_jobs",
    "execute_verifier_jobs",
    "generated_observation",
]

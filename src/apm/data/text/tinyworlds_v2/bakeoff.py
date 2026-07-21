"""Neutral TinyStories briefs, prompts, and strict bakeoff observations.

Phase 1 asks every candidate route to answer the same released TinyStories
briefs.  This module contains only deterministic request construction and
response interpretation; transport, caching, cost accounting, and route
locking live in separate modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import math
import re

from apm.data.text.tinyworlds_v2.json_contracts import (
    CanonicalJsonError,
    JsonObject,
    JsonValue,
    canonical_json_bytes,
    require_exact_fields,
    require_json_object,
    strict_json_loads,
)
from apm.data.text.tinyworlds_v2.surface import (
    canonical_feature_labels,
    lexical_tokens,
    realized_feature_labels,
    token_form_counts,
)


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_WORD_RE = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE)
_PLAIN_TEXT_STORY_PROFILES = frozenset(
    (
        "released-prompt-only-v1",
        "released-prompt-length-cue-v1",
    )
)
_FORBIDDEN_PATTERNS = (
    re.compile(r"\b(?:task|family|fact|relation|predicate|query)_id\b", re.I),
    re.compile(r"\b(?:task|family|fact|relation|predicate|query)\s*:\s*", re.I),
    re.compile(r"\banswer\s*:\s*", re.I),
    re.compile(r"\b(?:json|response)\s+schema\b", re.I),
    re.compile(r"\b(?:the|this|your)\s+prompt\b", re.I),
    re.compile(r"\bas an ai\b", re.I),
    re.compile(r"\bexact[- ]token\b", re.I),
    # Standalone numbers are normal children's prose ("three", "2 apples").
    # Page/chapter labels and mixed machine-like identifiers are not.
    re.compile(
        r"\b(?:page|chapter|section)\s+(?:number\s+)?#?\d+\b",
        re.I,
    ),
    re.compile(r"\b[A-Za-z]+(?:_[A-Za-z0-9]+)+\b"),
    re.compile(
        r"\b(?:id|record|task|item|node|entity|fact|query)\s*[:#-]\s*\d+\b",
        re.I,
    ),
    re.compile(
        r"it was calm[,.]?\s+everyone listened[,.]?\s+very softly",
        re.I,
    ),
)


@dataclass(frozen=True, slots=True)
class CandidateModelSpec:
    """One table-ordered OpenRouter candidate with a pinned catalog identity."""

    route_id: str
    request_model_id: str
    canonical_slug: str
    plan_prompt_usd_per_million: str
    plan_completion_usd_per_million: str
    first_party_provider_slug: str | None
    max_token_parameter: str = "max_tokens"

    def __post_init__(self) -> None:
        for label, value in (
            ("route_id", self.route_id),
            ("request_model_id", self.request_model_id),
            ("canonical_slug", self.canonical_slug),
            ("plan prompt price", self.plan_prompt_usd_per_million),
            ("plan completion price", self.plan_completion_usd_per_million),
        ):
            if type(value) is not str or not value:
                raise ValueError(f"candidate {label} must be nonempty")
        if self.first_party_provider_slug is not None and (
            type(self.first_party_provider_slug) is not str
            or not self.first_party_provider_slug
        ):
            raise ValueError("candidate first-party provider slug must be nonempty")
        if self.max_token_parameter not in ("max_tokens", "max_completion_tokens"):
            raise ValueError("candidate max-token parameter is unsupported")


@dataclass(frozen=True, slots=True)
class GenerationRequestContract:
    """Version the provider-facing request semantics that determine cache identity."""

    version: str
    enforce_distillable_text: bool
    seed_bits: int
    explicit_json_instruction: bool
    reasoning_disabled_routes: tuple[str, ...] = ()
    story_only_response: bool = False
    story_prompt_profile: str = "released-instruction-v1"
    plain_text_story_response: bool = False

    def __post_init__(self) -> None:
        if type(self.version) is not str or not self.version:
            raise ValueError("generation request contract version must be nonempty")
        if type(self.enforce_distillable_text) is not bool:
            raise TypeError("distillable-text routing policy must be boolean")
        if self.seed_bits not in (31, 32):
            raise ValueError("generation request seed width must be 31 or 32 bits")
        if type(self.explicit_json_instruction) is not bool:
            raise TypeError("JSON instruction policy must be boolean")
        if type(self.story_only_response) is not bool:
            raise TypeError("story-only response policy must be boolean")
        if type(self.plain_text_story_response) is not bool:
            raise TypeError("plain-text story response policy must be boolean")
        if self.plain_text_story_response and (
            self.explicit_json_instruction or self.story_only_response
        ):
            raise ValueError(
                "plain-text story responses cannot request JSON response semantics"
            )
        if self.story_prompt_profile not in {
            "released-instruction-v1",
            "reference-length-v1",
            "reference-shape-v1",
            "reference-structure-v2",
            *_PLAIN_TEXT_STORY_PROFILES,
        }:
            raise ValueError("generation story prompt profile is unsupported")
        if (self.story_prompt_profile in _PLAIN_TEXT_STORY_PROFILES) != (
            self.plain_text_story_response
        ):
            raise ValueError(
                "released plain-prompt profiles require a plain-text response"
            )
        if (
            type(self.reasoning_disabled_routes) is not tuple
            or any(
                type(route_id) is not str or not route_id
                for route_id in self.reasoning_disabled_routes
            )
            or tuple(sorted(set(self.reasoning_disabled_routes)))
            != self.reasoning_disabled_routes
        ):
            raise ValueError("reasoning-disabled route IDs must be sorted and unique")


GENERATION_REQUEST_V1 = GenerationRequestContract(
    version="tinyworlds-v2-generation-request-v1",
    enforce_distillable_text=True,
    seed_bits=32,
    explicit_json_instruction=False,
)

SYNTHETIC_STORY_REQUEST_V2 = GenerationRequestContract(
    version="tinyworlds-v2-synthetic-story-request-v2",
    enforce_distillable_text=False,
    seed_bits=31,
    explicit_json_instruction=True,
)

SYNTHETIC_STORY_REQUEST_V3 = GenerationRequestContract(
    version="tinyworlds-v2-synthetic-story-request-v3",
    enforce_distillable_text=False,
    seed_bits=31,
    explicit_json_instruction=True,
    # Qwen generated 5,138 unrequested reasoning tokens behind a 512-token
    # visible-output limit. Gemini's catalog likewise marks reasoning on by
    # default. Both routes support the normalized OpenRouter control.
    reasoning_disabled_routes=(
        "gemini-3.1-flash-lite",
        "qwen3.5-35b-a3b",
    ),
)


CANDIDATE_MODELS = (
    CandidateModelSpec(
        "ling-2.6-flash",
        "inclusionai/ling-2.6-flash",
        "inclusionai/ling-2.6-flash-20260421",
        "0.01",
        "0.03",
        None,
    ),
    CandidateModelSpec(
        "gemma-4-26b-a4b-it",
        "google/gemma-4-26b-a4b-it",
        "google/gemma-4-26b-a4b-it-20260403",
        "0.07",
        "0.34",
        "google-vertex/global",
    ),
    CandidateModelSpec(
        "deepseek-v4-flash",
        "deepseek/deepseek-v4-flash",
        "deepseek/deepseek-v4-flash-20260423",
        "0.098",
        "0.196",
        "deepseek",
    ),
    CandidateModelSpec(
        "mistral-small-2603",
        "mistralai/mistral-small-2603",
        "mistralai/mistral-small-2603",
        "0.15",
        "0.60",
        "mistral",
    ),
    CandidateModelSpec(
        "qwen3.5-35b-a3b",
        "qwen/qwen3.5-35b-a3b",
        "qwen/qwen3.5-35b-a3b-20260224",
        "0.14",
        "1.00",
        "alibaba",
    ),
    CandidateModelSpec(
        "gemini-3.1-flash-lite",
        "google/gemini-3.1-flash-lite",
        "google/gemini-3.1-flash-lite-20260507",
        "0.25",
        "1.50",
        "google-vertex/global",
    ),
    CandidateModelSpec(
        "gpt-5.4-mini",
        "openai/gpt-5.4-mini",
        "openai/gpt-5.4-mini-20260317",
        "0.75",
        "4.50",
        "openai",
        "max_completion_tokens",
    ),
)

# The original seven-route table above is part of the immutable preview
# contract.  Keep it intact while making the smaller author set an explicit,
# independently versioned choice for new Phase 1 work.
TWO_ROUTE_AUTHOR_MODELS = (
    CANDIDATE_MODELS[4],
    CANDIDATE_MODELS[6],
)

SYNTHETIC_STORY_REQUEST_V4 = GenerationRequestContract(
    version="tinyworlds-v2-synthetic-story-request-v4",
    enforce_distillable_text=False,
    seed_bits=31,
    explicit_json_instruction=True,
    reasoning_disabled_routes=(
        "gpt-5.4-mini",
        "qwen3.5-35b-a3b",
    ),
    story_only_response=True,
)

SYNTHETIC_STORY_REQUEST_V5 = GenerationRequestContract(
    version="tinyworlds-v2-synthetic-story-request-v5",
    enforce_distillable_text=False,
    seed_bits=31,
    explicit_json_instruction=True,
    reasoning_disabled_routes=(
        "gpt-5.4-mini",
        "qwen3.5-35b-a3b",
    ),
    story_only_response=True,
    story_prompt_profile="reference-length-v1",
)

SYNTHETIC_STORY_REQUEST_V6 = GenerationRequestContract(
    version="tinyworlds-v2-synthetic-story-request-v6",
    enforce_distillable_text=False,
    seed_bits=31,
    explicit_json_instruction=True,
    reasoning_disabled_routes=(
        "gpt-5.4-mini",
        "qwen3.5-35b-a3b",
    ),
    story_only_response=True,
    story_prompt_profile="reference-shape-v1",
)

SYNTHETIC_STORY_REQUEST_V7 = GenerationRequestContract(
    version="tinyworlds-v2-synthetic-story-request-v7",
    enforce_distillable_text=False,
    seed_bits=31,
    explicit_json_instruction=True,
    reasoning_disabled_routes=(
        "gpt-5.4-mini",
        "qwen3.5-35b-a3b",
    ),
    story_only_response=True,
    story_prompt_profile="reference-structure-v2",
)

SYNTHETIC_STORY_REQUEST_V8 = GenerationRequestContract(
    version="tinyworlds-v2-synthetic-story-request-v8",
    enforce_distillable_text=False,
    seed_bits=31,
    explicit_json_instruction=False,
    reasoning_disabled_routes=(
        "gpt-5.4-mini",
        "qwen3.5-35b-a3b",
    ),
    story_prompt_profile="released-prompt-only-v1",
    plain_text_story_response=True,
)

SYNTHETIC_STORY_REQUEST_V9 = GenerationRequestContract(
    version="tinyworlds-v2-synthetic-story-request-v9",
    enforce_distillable_text=False,
    seed_bits=31,
    explicit_json_instruction=False,
    reasoning_disabled_routes=(
        "gpt-5.4-mini",
        "qwen3.5-35b-a3b",
    ),
    story_prompt_profile="released-prompt-length-cue-v1",
    plain_text_story_response=True,
)

VERIFIER_MODEL = CandidateModelSpec(
    "gpt-5.4-verifier",
    "openai/gpt-5.4",
    "openai/gpt-5.4-20260305",
    "2.50",
    "15.00",
    "openai",
)


@dataclass(frozen=True, slots=True)
class NeutralStoryBrief:
    """One source-paired, world-free TinyStories generation instruction."""

    brief_id: str
    source_record_id: str
    prompt_text: str
    required_words: tuple[str, ...]
    requested_features: tuple[str, ...]
    matched_reference_text: str

    def __post_init__(self) -> None:
        for label, value in (
            ("brief_id", self.brief_id),
            ("source_record_id", self.source_record_id),
            ("prompt_text", self.prompt_text),
            ("matched_reference_text", self.matched_reference_text),
        ):
            if type(value) is not str or not value.strip():
                raise ValueError(f"neutral brief {label} must be nonempty")
        if (
            type(self.required_words) is not tuple
            or len(self.required_words) != 3
            or any(type(word) is not str or not word.strip() for word in self.required_words)
            or len(set(word.casefold() for word in self.required_words)) != 3
        ):
            raise ValueError("neutral brief must contain three unique required words")
        if (
            type(self.requested_features) is not tuple
            or any(
                type(feature) is not str or not feature.strip()
                for feature in self.requested_features
            )
            or len(set(self.requested_features)) != len(self.requested_features)
        ):
            raise ValueError("neutral brief features must be unique strings")


@dataclass(frozen=True, slots=True)
class EvidenceQuote:
    """One required ingredient and an exact supporting substring."""

    ingredient: str
    exact_quote: str

    def __post_init__(self) -> None:
        if type(self.ingredient) is not str or not self.ingredient:
            raise ValueError("evidence ingredient must be nonempty")
        if type(self.exact_quote) is not str or not self.exact_quote:
            raise ValueError("evidence exact quote must be nonempty")


@dataclass(frozen=True, slots=True)
class GeneratedStoryPayload:
    """Strict structured-output payload returned by one generator route."""

    story: str
    word_evidence: tuple[EvidenceQuote, ...]
    feature_evidence: tuple[EvidenceQuote, ...]

    def __post_init__(self) -> None:
        if type(self.story) is not str or not self.story.strip():
            raise ValueError("generated story must be nonempty")
        if type(self.word_evidence) is not tuple or any(
            type(item) is not EvidenceQuote for item in self.word_evidence
        ):
            raise TypeError("word evidence must contain EvidenceQuote values")
        if type(self.feature_evidence) is not tuple or any(
            type(item) is not EvidenceQuote for item in self.feature_evidence
        ):
            raise TypeError("feature evidence must contain EvidenceQuote values")


@dataclass(frozen=True, slots=True)
class TextEvidenceSpan:
    """One canonical, locally located ingredient occurrence in story text."""

    ingredient: str
    exact_text: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if type(self.ingredient) is not str or not self.ingredient:
            raise ValueError("text evidence ingredient must be nonempty")
        if type(self.exact_text) is not str or not self.exact_text:
            raise ValueError("text evidence exact text must be nonempty")
        if (
            type(self.start) is not int
            or type(self.end) is not int
            or self.start < 0
            or self.end <= self.start
            or self.end - self.start != len(self.exact_text)
        ):
            raise ValueError("text evidence offsets must bound the exact text")


@dataclass(frozen=True, slots=True)
class StoryOnlyPayload:
    """Story-only response augmented solely with evidence derived locally."""

    story: str
    required_word_spans: tuple[TextEvidenceSpan, ...]
    realized_features: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.story) is not str or not self.story.strip():
            raise ValueError("generated story must be nonempty")
        if type(self.required_word_spans) is not tuple or any(
            type(item) is not TextEvidenceSpan for item in self.required_word_spans
        ):
            raise TypeError("required-word spans must contain TextEvidenceSpan values")
        if type(self.realized_features) is not tuple or any(
            type(feature) is not str or not feature
            for feature in self.realized_features
        ):
            raise ValueError("realized features must be nonempty strings")
        if tuple(sorted(set(self.realized_features))) != self.realized_features:
            raise ValueError("realized features must be sorted and unique")
        if any(
            item.end > len(self.story)
            or self.story[item.start : item.end] != item.exact_text
            for item in self.required_word_spans
        ):
            raise ValueError("required-word spans must locate exact story text")
        span_keys = tuple(
            item.ingredient.casefold() for item in self.required_word_spans
        )
        if len(span_keys) != len(set(span_keys)):
            raise ValueError("required-word spans must have unique ingredients")


@dataclass(frozen=True, slots=True)
class StoryValidation:
    """Deterministic acceptance evidence independent of any model verifier."""

    schema_valid: bool
    required_words_present: bool
    evidence_valid: bool
    forbidden_identifier_present: bool
    length_valid: bool
    accepted: bool
    rejection_reasons: tuple[str, ...]
    story_sha256: str | None

    def __post_init__(self) -> None:
        boolean_fields = (
            self.schema_valid,
            self.required_words_present,
            self.evidence_valid,
            self.forbidden_identifier_present,
            self.length_valid,
            self.accepted,
        )
        if any(type(value) is not bool for value in boolean_fields):
            raise TypeError("story validation flags must be booleans")
        if type(self.rejection_reasons) is not tuple or any(
            type(reason) is not str or not reason for reason in self.rejection_reasons
        ):
            raise ValueError("story validation reasons must be nonempty strings")
        if self.accepted != (not self.rejection_reasons):
            raise ValueError("story acceptance must equal an empty rejection-reason set")
        if self.story_sha256 is not None and _SHA256_RE.fullmatch(self.story_sha256) is None:
            raise ValueError("story digest must be lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class VerifierPayload:
    """Source-blind TinyStories style assessment from the pinned verifier."""

    preschool_vocabulary: int
    sentence_simplicity: int
    grammar: int
    plot_coherence: int
    non_repetition: int
    brief_adherence: bool
    hard_failures: tuple[str, ...]
    rationale: str

    def __post_init__(self) -> None:
        scores = (
            self.preschool_vocabulary,
            self.sentence_simplicity,
            self.grammar,
            self.plot_coherence,
            self.non_repetition,
        )
        if any(type(score) is not int or not 1 <= score <= 5 for score in scores):
            raise ValueError("verifier rubric scores must be integers from one to five")
        if type(self.brief_adherence) is not bool:
            raise TypeError("verifier brief_adherence must be boolean")
        allowed_failures = {"grammar", "coherence", "repetition", "meta_language"}
        if (
            type(self.hard_failures) is not tuple
            or any(failure not in allowed_failures for failure in self.hard_failures)
            or len(set(self.hard_failures)) != len(self.hard_failures)
        ):
            raise ValueError("verifier hard failures are invalid or duplicated")
        if type(self.rationale) is not str or not self.rationale.strip():
            raise ValueError("verifier rationale must be nonempty")

    @property
    def mean_score(self) -> float:
        """Return the arithmetic mean across the five fixed style dimensions."""
        return sum(
            (
                self.preschool_vocabulary,
                self.sentence_simplicity,
                self.grammar,
                self.plot_coherence,
                self.non_repetition,
            )
        ) / 5.0


def plain_story_user_prompt(
    brief: NeutralStoryBrief,
    profile: str,
) -> str:
    """Build the exact one-message released-prompt text for a plain story cell."""
    if type(brief) is not NeutralStoryBrief:
        raise TypeError("plain story prompt brief must be NeutralStoryBrief")
    if profile == "released-prompt-only-v1":
        additions: tuple[str, ...] = ()
    elif profile == "released-prompt-length-cue-v1":
        additions = ("Aim for about 130 to 150 words.",)
    else:
        raise ValueError("plain story prompt profile is unsupported")
    return "\n\n".join((brief.prompt_text, *additions, "Possible story:"))


def neutral_story_request_body(
    brief: NeutralStoryBrief,
    model: CandidateModelSpec,
    *,
    provider_slug: str,
    provider_quantization: str,
    prompt_usd_per_token: str,
    completion_usd_per_token: str,
    request_contract: GenerationRequestContract = GENERATION_REQUEST_V1,
) -> JsonObject:
    """Build one provider-locked, plugin-free strict generation request."""
    _require_route_fields(
        provider_slug,
        provider_quantization,
        prompt_usd_per_token,
        completion_usd_per_token,
    )
    if type(request_contract) is not GenerationRequestContract:
        raise TypeError("generation request contract has the wrong type")
    seed = _deterministic_seed(brief.brief_id, request_contract)
    json_instruction = ""
    if request_contract.explicit_json_instruction:
        json_instruction = (
            " Return exactly one JSON object matching the supplied response format."
            if request_contract.story_only_response
            else " Return one JSON object matching the supplied response format."
        )
    provider: JsonObject = {
        "allow_fallbacks": False,
        "data_collection": "deny",
        "max_price": {
            "completion": _per_million_price(completion_usd_per_token),
            "prompt": _per_million_price(prompt_usd_per_token),
        },
        "only": [provider_slug],
        "quantizations": [provider_quantization],
        "require_parameters": True,
    }
    if request_contract.enforce_distillable_text:
        provider["enforce_distillable_text"] = True
    messages: list[JsonObject]
    if request_contract.plain_text_story_response:
        messages = [
            {
                "content": plain_story_user_prompt(
                    brief,
                    request_contract.story_prompt_profile,
                ),
                "role": "user",
            }
        ]
    else:
        messages = [
            {
                "content": _neutral_system_prompt(
                    request_contract.story_prompt_profile,
                    json_instruction=json_instruction,
                ),
                "role": "system",
            },
            {
                "content": _neutral_user_prompt(
                    brief,
                    story_only=request_contract.story_only_response,
                    story_prompt_profile=request_contract.story_prompt_profile,
                ),
                "role": "user",
            },
        ]
    body: JsonObject = {
        model.max_token_parameter: 512,
        "messages": messages,
        "model": model.request_model_id,
        "plugins": [],
        "provider": provider,
        "seed": seed,
        "stream": False,
        "transforms": [],
    }
    if not request_contract.plain_text_story_response:
        body["response_format"] = {
            "json_schema": {
                "name": (
                    "tinyworlds_v2_neutral_story_v2"
                    if request_contract.story_only_response
                    else "tinyworlds_v2_neutral_story_v1"
                ),
                "schema": (
                    _story_only_response_schema()
                    if request_contract.story_only_response
                    else _story_response_schema(brief)
                ),
                "strict": True,
            },
            "type": "json_schema",
        }
    if model.route_id in request_contract.reasoning_disabled_routes:
        body["reasoning"] = {"effort": "none"}
    return body


def verifier_request_body(
    brief: NeutralStoryBrief,
    story: str,
    *,
    provider_slug: str,
    provider_quantization: str,
    prompt_usd_per_token: str,
    completion_usd_per_token: str,
    request_contract: GenerationRequestContract = GENERATION_REQUEST_V1,
) -> JsonObject:
    """Build one source-blind request for the pinned independent style verifier."""
    if type(story) is not str or not story.strip():
        raise ValueError("verifier story must be nonempty")
    _require_route_fields(
        provider_slug,
        provider_quantization,
        prompt_usd_per_token,
        completion_usd_per_token,
    )
    if type(request_contract) is not GenerationRequestContract:
        raise TypeError("generation request contract has the wrong type")
    seed_material = f"verifier\0{brief.brief_id}\0{sha256(story.encode('utf-8')).hexdigest()}"
    seed = _deterministic_seed(seed_material, request_contract)
    json_instruction = (
        " Return one JSON object matching the supplied response format."
        if request_contract.explicit_json_instruction
        else ""
    )
    provider: JsonObject = {
        "allow_fallbacks": False,
        "data_collection": "deny",
        "max_price": {
            "completion": _per_million_price(completion_usd_per_token),
            "prompt": _per_million_price(prompt_usd_per_token),
        },
        "only": [provider_slug],
        "quantizations": [provider_quantization],
        "require_parameters": True,
    }
    if request_contract.enforce_distillable_text:
        provider["enforce_distillable_text"] = True
    return {
        VERIFIER_MODEL.max_token_parameter: 256,
        "messages": [
            {
                "content": (
                    "Judge children's stories with the fixed rubric. Do not infer or "
                    "guess whether a story is human-written or model-generated. Score "
                    "only the text and its adherence to the supplied story brief."
                    f"{json_instruction}"
                ),
                "role": "system",
            },
            {
                "content": (
                    f"STORY BRIEF:\n{brief.prompt_text}\n\n"
                    f"REQUIRED WORDS: {', '.join(brief.required_words)}\n"
                    f"REQUESTED FEATURES: {', '.join(brief.requested_features) or 'none'}\n\n"
                    f"STORY:\n{story}"
                ),
                "role": "user",
            },
        ],
        "model": VERIFIER_MODEL.request_model_id,
        "plugins": [],
        "provider": provider,
        "response_format": {
            "json_schema": {
                "name": "tinyworlds_v2_style_verifier_v1",
                "schema": _verifier_response_schema(),
                "strict": True,
            },
            "type": "json_schema",
        },
        "seed": seed,
        "stream": False,
        "transforms": [],
    }


def _deterministic_seed(
    material: str,
    request_contract: GenerationRequestContract,
) -> int:
    """Map stable request material into the contract's provider-safe seed range."""
    raw = int.from_bytes(sha256(material.encode("utf-8")).digest()[:4], "big")
    return raw & ((1 << request_contract.seed_bits) - 1)


def parse_generated_story_payload(content: str | bytes) -> GeneratedStoryPayload:
    """Strictly parse the assistant message content without repairing it."""
    payload = content.encode("utf-8") if type(content) is str else content
    if type(payload) is not bytes:
        raise TypeError("generated story content must be text or bytes")
    record = require_json_object(
        strict_json_loads(payload, label="generated story content"),
        label="generated story content",
    )
    require_exact_fields(
        record,
        ("feature_evidence", "story", "word_evidence"),
        label="generated story content",
    )
    story = _required_string(record["story"], "generated story")
    return GeneratedStoryPayload(
        story=story,
        word_evidence=_decode_evidence(record["word_evidence"], "required_word"),
        feature_evidence=_decode_evidence(record["feature_evidence"], "feature"),
    )


def parse_story_only_payload(
    brief: NeutralStoryBrief,
    content: str | bytes,
) -> StoryOnlyPayload:
    """Parse exactly ``{"story": ...}`` and derive all evidence locally."""
    if type(brief) is not NeutralStoryBrief:
        raise TypeError("story-only payload brief must be NeutralStoryBrief")
    raw = content.encode("utf-8") if type(content) is str else content
    if type(raw) is not bytes:
        raise TypeError("generated story content must be text or bytes")
    record = require_json_object(
        strict_json_loads(raw, label="story-only generated content"),
        label="story-only generated content",
    )
    require_exact_fields(
        record,
        ("story",),
        label="story-only generated content",
    )
    story = _required_string(record["story"], "generated story")
    return StoryOnlyPayload(
        story=story,
        required_word_spans=_required_word_spans(story, brief.required_words),
        realized_features=realized_feature_labels(story, brief.requested_features),
    )


def parse_plain_text_story_payload(
    brief: NeutralStoryBrief,
    content: str | bytes,
) -> StoryOnlyPayload:
    """Treat the complete assistant message as story text without repairing it."""
    if type(brief) is not NeutralStoryBrief:
        raise TypeError("plain-text story brief must be NeutralStoryBrief")
    if type(content) is bytes:
        story = content.decode("utf-8")
    elif type(content) is str:
        story = content
    else:
        raise TypeError("plain-text generated content must be text or bytes")
    if not story.strip():
        raise ValueError("plain-text generated story must be nonempty")
    return StoryOnlyPayload(
        story=story,
        required_word_spans=_required_word_spans(story, brief.required_words),
        realized_features=realized_feature_labels(story, brief.requested_features),
    )


def assistant_message_content(response_body: bytes) -> str:
    """Extract one completed assistant message from an exact OpenRouter body."""
    if type(response_body) is not bytes:
        raise TypeError("OpenRouter response body must be bytes")
    root = require_json_object(
        strict_json_loads(response_body, label="OpenRouter completion response"),
        label="OpenRouter completion response",
    )
    choices = root.get("choices")
    if type(choices) is not list or len(choices) != 1:
        raise CanonicalJsonError("completion response must contain exactly one choice")
    choice = require_json_object(choices[0], label="completion choice")
    if choice.get("finish_reason") != "stop":
        raise CanonicalJsonError("completion choice did not finish with stop")
    message = require_json_object(choice.get("message"), label="assistant message")
    if message.get("role") != "assistant":
        raise CanonicalJsonError("completion response role must be assistant")
    content = message.get("content")
    if type(content) is not str or not content:
        raise CanonicalJsonError("completion assistant content must be nonempty text")
    return content


def validate_generated_story(
    brief: NeutralStoryBrief,
    content: str | bytes,
) -> tuple[GeneratedStoryPayload | None, StoryValidation]:
    """Parse and deterministically accept or reject one raw model message."""
    try:
        payload = parse_generated_story_payload(content)
    except (CanonicalJsonError, TypeError, ValueError):
        return None, StoryValidation(
            schema_valid=False,
            required_words_present=False,
            evidence_valid=False,
            forbidden_identifier_present=False,
            length_valid=False,
            accepted=False,
            rejection_reasons=("schema_invalid",),
            story_sha256=None,
        )

    return payload, _validate_story_text(
        brief,
        payload.story,
        evidence_valid=_evidence_is_valid(brief, payload),
    )


def validate_story_only_generated_story(
    brief: NeutralStoryBrief,
    content: str | bytes,
) -> tuple[StoryOnlyPayload | None, StoryValidation]:
    """Validate a V4 story without accepting model-authored evidence claims."""
    try:
        payload = parse_story_only_payload(brief, content)
    except (CanonicalJsonError, TypeError, ValueError):
        return None, _schema_invalid_story_validation()

    return payload, _validate_locally_derived_story(brief, payload)


def validate_plain_text_generated_story(
    brief: NeutralStoryBrief,
    content: str | bytes,
) -> tuple[StoryOnlyPayload | None, StoryValidation]:
    """Validate an unwrapped assistant story with locally derived evidence."""
    try:
        payload = parse_plain_text_story_payload(brief, content)
    except (UnicodeDecodeError, TypeError, ValueError):
        return None, _schema_invalid_story_validation()
    return payload, _validate_locally_derived_story(brief, payload)


def _validate_locally_derived_story(
    brief: NeutralStoryBrief,
    payload: StoryOnlyPayload,
) -> StoryValidation:
    # Only surface-observable features belong in a deterministic hard gate.
    # Conflict, foreshadowing, moral value, twists, and ending valence require
    # semantic judgment and remain audit/report metadata. Quoted dialogue is
    # the one released narrative feature with an exact local text contract.
    observable_features = frozenset(
        feature
        for feature in canonical_feature_labels(brief.requested_features)
        if feature == "Dialogue"
    )
    return _validate_story_text(
        brief,
        payload.story,
        evidence_valid=(
            len(payload.required_word_spans) == len(brief.required_words)
            and observable_features.issubset(payload.realized_features)
        ),
    )


def parse_verifier_payload(content: str | bytes) -> VerifierPayload:
    """Strictly parse one source-blind verifier response without healing."""
    payload = content.encode("utf-8") if type(content) is str else content
    if type(payload) is not bytes:
        raise TypeError("verifier content must be text or bytes")
    record = require_json_object(
        strict_json_loads(payload, label="verifier content"),
        label="verifier content",
    )
    fields = (
        "brief_adherence",
        "grammar",
        "hard_failures",
        "non_repetition",
        "plot_coherence",
        "preschool_vocabulary",
        "rationale",
        "sentence_simplicity",
    )
    require_exact_fields(record, fields, label="verifier content")
    failures_value = record["hard_failures"]
    if type(failures_value) is not list or any(
        type(item) is not str for item in failures_value
    ):
        raise CanonicalJsonError("verifier hard_failures must be a string array")
    adherence = record["brief_adherence"]
    if type(adherence) is not bool:
        raise CanonicalJsonError("verifier brief_adherence must be boolean")
    return VerifierPayload(
        preschool_vocabulary=_required_integer(record["preschool_vocabulary"], "preschool_vocabulary"),
        sentence_simplicity=_required_integer(record["sentence_simplicity"], "sentence_simplicity"),
        grammar=_required_integer(record["grammar"], "grammar"),
        plot_coherence=_required_integer(record["plot_coherence"], "plot_coherence"),
        non_repetition=_required_integer(record["non_repetition"], "non_repetition"),
        brief_adherence=adherence,
        hard_failures=tuple(failures_value),
        rationale=_required_string(record["rationale"], "verifier rationale"),
    )


def _neutral_user_prompt(
    brief: NeutralStoryBrief,
    *,
    story_only: bool = False,
    story_prompt_profile: str = "released-instruction-v1",
) -> str:
    features = ", ".join(brief.requested_features) or "none"
    prompt = (
        f"RELEASED TINYSTORIES INSTRUCTION:\n{brief.prompt_text}\n\n"
        f"REQUIRED WORDS: {', '.join(brief.required_words)}\n"
        f"REQUESTED NARRATIVE FEATURES: {features}\n\n"
    )
    if story_only:
        story_only_prompt = (
            prompt
            + "Return exactly one JSON object containing only the story field. "
            "Do not return evidence, analysis, commentary, or any other field."
        )
        if story_prompt_profile == "reference-structure-v2":
            return story_only_prompt + _v7_final_story_requirements(brief)
        return story_only_prompt
    return prompt + (
        "Return the story plus exact quotes showing each required word and each "
        "requested feature. Quotes must occur verbatim in the story."
    )


def _neutral_system_prompt(
    profile: str,
    *,
    json_instruction: str,
) -> str:
    if profile == "reference-structure-v2":
        return (
            "Write one complete children's story. Keep it gentle and suitable "
            "for young children. Follow the supplied released TinyStories "
            "instruction closely. Use ordinary story prose only. Do not mention "
            "prompts, schemas, required words, or this request."
            f"{json_instruction}"
        )
    return (
        "Write one short, complete children's story that a typical "
        "3- to 4-year-old can understand. Follow the supplied released "
        "TinyStories instruction closely. Use ordinary story prose only. "
        "Do not mention prompts, schemas, required words, or this request."
        f"{_story_prompt_guidance(profile)}"
        f"{json_instruction}"
    )


def _v7_final_story_requirements(brief: NeutralStoryBrief) -> str:
    dialogue_requirement = (
        (
            "Because Dialogue is requested, include an actual complete spoken "
            'sentence in standard ASCII double quotation marks, like "Hello.".'
        ),
    ) if "Dialogue" in canonical_feature_labels(brief.requested_features) else ()
    requirements = (
        "Use every required word exactly as written at least once, without "
        "changing that occurrence's spelling or word form: "
        f"{', '.join(brief.required_words)}.",
        *dialogue_requirement,
        "Put the story in one continuous story-field text block with no "
        "newline characters.",
        "Write 18 to 20 complete sentences, mostly 7 to 11 words each.",
        "Include at least 6 connected events, each leading naturally onward.",
        "Aim for 155 to 190 words; this is a soft target, not a reason to cut "
        "off the ending.",
        "Use a little natural, simple repetition, without padding.",
        _v7_opening_requirement(brief.brief_id),
    )
    return "\n\nFINAL STORY REQUIREMENTS:\n- " + "\n- ".join(requirements)


def _v7_opening_requirement(brief_id: str) -> str:
    material = f"tinyworlds-v2-v7-opening-v1\0{brief_id}".encode("utf-8")
    bucket = int.from_bytes(sha256(material).digest()[:8], "big") % 5
    if bucket < 3:
        return 'Start the story with exactly "Once upon a time".'
    if bucket == 3:
        return 'Start the story with exactly "One day".'
    return (
        'Use another simple opening; do not start with "Once upon a time" or '
        '"One day".'
    )


def _story_prompt_guidance(profile: str) -> str:
    if profile == "released-instruction-v1":
        return ""
    length_guidance = " Write 130 to 170 words."
    if profile == "reference-length-v1":
        return length_guidance
    if profile == "reference-shape-v1":
        return length_guidance + (
            " Use three to five short paragraphs separated by single line breaks; "
            "do not put blank lines between them. Use mostly short sentences and "
            "simple chronological actions, with each event leading naturally to "
            "the next. Begin naturally with 'Once upon a time' or 'One day'. Use "
            "ordinary English words only. End with a simple consequence or feeling "
            "inside the story. Avoid headings, lists, summaries, ornate description, "
            "and explanations of the writing task."
        )
    raise ValueError("generation story prompt profile is unsupported")


def _story_only_response_schema() -> JsonObject:
    return {
        "additionalProperties": False,
        "properties": {
            "story": {"minLength": 1, "type": "string"},
        },
        "required": ["story"],
        "type": "object",
    }


def _story_response_schema(brief: NeutralStoryBrief) -> JsonObject:
    evidence_properties: JsonObject = {
        "exact_quote": {"minLength": 1, "type": "string"},
    }
    word_properties = dict(evidence_properties)
    word_properties["required_word"] = {
        "enum": list(brief.required_words),
        "type": "string",
    }
    feature_properties = dict(evidence_properties)
    feature_properties["feature"] = (
        {
            "enum": list(brief.requested_features),
            "type": "string",
        }
        if brief.requested_features
        else {"type": "string"}
    )
    return {
        "additionalProperties": False,
        "properties": {
            "feature_evidence": {
                "items": {
                    "additionalProperties": False,
                    "properties": feature_properties,
                    "required": ["feature", "exact_quote"],
                    "type": "object",
                },
                "maxItems": len(brief.requested_features),
                "minItems": len(brief.requested_features),
                "type": "array",
            },
            "story": {"minLength": 1, "type": "string"},
            "word_evidence": {
                "items": {
                    "additionalProperties": False,
                    "properties": word_properties,
                    "required": ["required_word", "exact_quote"],
                    "type": "object",
                },
                "maxItems": len(brief.required_words),
                "minItems": len(brief.required_words),
                "type": "array",
            },
        },
        "required": ["story", "word_evidence", "feature_evidence"],
        "type": "object",
    }


def _verifier_response_schema() -> JsonObject:
    score = {"maximum": 5, "minimum": 1, "type": "integer"}
    return {
        "additionalProperties": False,
        "properties": {
            "brief_adherence": {"type": "boolean"},
            "grammar": score,
            "hard_failures": {
                "items": {
                    "enum": ["grammar", "coherence", "repetition", "meta_language"],
                    "type": "string",
                },
                "type": "array",
                "uniqueItems": True,
            },
            "non_repetition": score,
            "plot_coherence": score,
            "preschool_vocabulary": score,
            "rationale": {"maxLength": 500, "minLength": 1, "type": "string"},
            "sentence_simplicity": score,
        },
        "required": [
            "preschool_vocabulary",
            "sentence_simplicity",
            "grammar",
            "plot_coherence",
            "non_repetition",
            "brief_adherence",
            "hard_failures",
            "rationale",
        ],
        "type": "object",
    }


def _decode_evidence(value: JsonValue, ingredient_field: str) -> tuple[EvidenceQuote, ...]:
    if type(value) is not list:
        raise CanonicalJsonError(f"{ingredient_field} evidence must be an array")
    evidence: list[EvidenceQuote] = []
    for index, item in enumerate(value):
        record = require_json_object(item, label=f"{ingredient_field} evidence {index}")
        require_exact_fields(
            record,
            ("exact_quote", ingredient_field),
            label=f"{ingredient_field} evidence {index}",
        )
        evidence.append(
            EvidenceQuote(
                ingredient=_required_string(record[ingredient_field], ingredient_field),
                exact_quote=_required_string(record["exact_quote"], "exact quote"),
            )
        )
    return tuple(evidence)


def _required_word_spans(
    story: str,
    required_words: tuple[str, ...],
) -> tuple[TextEvidenceSpan, ...]:
    first_by_word: dict[str, re.Match[str]] = {}
    required_keys = frozenset(word.casefold() for word in required_words)
    for match in _WORD_RE.finditer(story):
        key = match.group(0).casefold()
        if key in required_keys and key not in first_by_word:
            first_by_word[key] = match
    spans: list[TextEvidenceSpan] = []
    for required_word in required_words:
        match = first_by_word.get(required_word.casefold())
        if match is not None:
            spans.append(
                TextEvidenceSpan(
                    ingredient=required_word,
                    exact_text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                )
            )
    return tuple(spans)


def _schema_invalid_story_validation() -> StoryValidation:
    return StoryValidation(
        schema_valid=False,
        required_words_present=False,
        evidence_valid=False,
        forbidden_identifier_present=False,
        length_valid=False,
        accepted=False,
        rejection_reasons=("schema_invalid",),
        story_sha256=None,
    )


def _validate_story_text(
    brief: NeutralStoryBrief,
    story: str,
    *,
    evidence_valid: bool,
) -> StoryValidation:
    story_words = lexical_tokens(story)
    required_words_present = all(
        required.casefold() in story_words for required in brief.required_words
    )
    # Ordinary hyphenated prose such as ``3-year-old`` remains valid.  A
    # hyphen/apostrophe segment containing both a letter and a digit (``fox7``,
    # ``R2-D2``) is the machine-like form that the profile and quality gate also
    # measure.  Keeping this test on the shared tokenizer prevents acceptance
    # and the reported identifier-token rate from drifting.
    mixed_alphanumeric = token_form_counts(story_words)[2] > 0
    forbidden = mixed_alphanumeric or any(
        pattern.search(story) for pattern in _FORBIDDEN_PATTERNS
    )
    length_valid = 40 <= len(story_words) <= 600
    reasons = tuple(
        reason
        for invalid, reason in (
            (not required_words_present, "required_words_missing"),
            (not evidence_valid, "evidence_invalid"),
            (forbidden, "forbidden_identifier_or_meta_language"),
            (not length_valid, "story_length_outside_safety_bounds"),
        )
        if invalid
    )
    return StoryValidation(
        schema_valid=True,
        required_words_present=required_words_present,
        evidence_valid=evidence_valid,
        forbidden_identifier_present=forbidden,
        length_valid=length_valid,
        accepted=not reasons,
        rejection_reasons=reasons,
        story_sha256=sha256(story.encode("utf-8")).hexdigest(),
    )


def _evidence_is_valid(brief: NeutralStoryBrief, payload: GeneratedStoryPayload) -> bool:
    word_evidence = {item.ingredient.casefold(): item for item in payload.word_evidence}
    required_words = {word.casefold(): word for word in brief.required_words}
    if (
        len(word_evidence) != len(payload.word_evidence)
        or frozenset(word_evidence) != frozenset(required_words)
    ):
        return False
    feature_evidence = {item.ingredient: item for item in payload.feature_evidence}
    if (
        len(feature_evidence) != len(payload.feature_evidence)
        or frozenset(feature_evidence) != frozenset(brief.requested_features)
    ):
        return False
    for item in (*word_evidence.values(), *feature_evidence.values()):
        if item.exact_quote not in payload.story:
            return False
    return all(
        word in {
            match.group(0).casefold()
            for match in _WORD_RE.finditer(word_evidence[word].exact_quote)
        }
        for word in required_words
    )


def _required_string(value: JsonValue, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise CanonicalJsonError(f"{label} must be a nonempty string")
    return value


def _required_integer(value: JsonValue, label: str) -> int:
    if type(value) is not int:
        raise CanonicalJsonError(f"verifier {label} must be an integer")
    return value


def _require_route_fields(
    provider_slug: str,
    provider_quantization: str,
    prompt_usd_per_token: str,
    completion_usd_per_token: str,
) -> None:
    values = (
        provider_slug,
        provider_quantization,
        prompt_usd_per_token,
        completion_usd_per_token,
    )
    if any(type(value) is not str or not value for value in values):
        raise ValueError("provider route fields must be nonempty strings")
    if provider_quantization in {"int4", "fp4"}:
        raise ValueError("four-bit provider routes are forbidden")


def _per_million_price(per_token: str) -> float:
    """Encode an exact per-token lock price as a conservative JSON number.

    OpenRouter's ``provider.max_price`` fields are per-million-token caps.  JSON
    has no decimal type, so the body ultimately carries a binary float.  Move
    to the next representable float when the shortest JSON representation
    would otherwise be below the exact catalog price: a cap must never round
    down and accidentally exclude the locked route.
    """
    try:
        exact_price = Decimal(per_token) * Decimal(1_000_000)
    except (InvalidOperation, TypeError) as error:
        raise ValueError("provider price must be a decimal string") from error
    if not exact_price.is_finite() or not Decimal(0) <= exact_price < Decimal(
        1_000_000
    ):
        raise ValueError("provider price is outside its supported range")
    encoded_price = float(exact_price)
    if Decimal(str(encoded_price)) < exact_price:
        encoded_price = math.nextafter(encoded_price, math.inf)
    if not math.isfinite(encoded_price) or Decimal(str(encoded_price)) < exact_price:
        raise ValueError("provider price cannot be represented conservatively")
    return encoded_price


def request_body_sha256(body: JsonObject) -> str:
    """Return the canonical body digest exposed in request artifacts."""
    return sha256(canonical_json_bytes(body)).hexdigest()


__all__ = [
    "CANDIDATE_MODELS",
    "GENERATION_REQUEST_V1",
    "SYNTHETIC_STORY_REQUEST_V2",
    "SYNTHETIC_STORY_REQUEST_V3",
    "SYNTHETIC_STORY_REQUEST_V4",
    "SYNTHETIC_STORY_REQUEST_V5",
    "SYNTHETIC_STORY_REQUEST_V6",
    "SYNTHETIC_STORY_REQUEST_V7",
    "SYNTHETIC_STORY_REQUEST_V8",
    "SYNTHETIC_STORY_REQUEST_V9",
    "TWO_ROUTE_AUTHOR_MODELS",
    "VERIFIER_MODEL",
    "CandidateModelSpec",
    "EvidenceQuote",
    "GeneratedStoryPayload",
    "GenerationRequestContract",
    "NeutralStoryBrief",
    "StoryValidation",
    "StoryOnlyPayload",
    "TextEvidenceSpan",
    "VerifierPayload",
    "assistant_message_content",
    "neutral_story_request_body",
    "plain_story_user_prompt",
    "parse_generated_story_payload",
    "parse_plain_text_story_payload",
    "parse_story_only_payload",
    "parse_verifier_payload",
    "request_body_sha256",
    "validate_generated_story",
    "validate_plain_text_generated_story",
    "validate_story_only_generated_story",
    "verifier_request_body",
]

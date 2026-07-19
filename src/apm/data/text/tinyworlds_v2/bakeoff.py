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
from apm.data.text.tinyworlds_v2.surface import lexical_tokens, token_form_counts


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_WORD_RE = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE)
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


def neutral_story_request_body(
    brief: NeutralStoryBrief,
    model: CandidateModelSpec,
    *,
    provider_slug: str,
    provider_quantization: str,
    prompt_usd_per_token: str,
    completion_usd_per_token: str,
) -> JsonObject:
    """Build one provider-locked, plugin-free strict generation request."""
    _require_route_fields(
        provider_slug,
        provider_quantization,
        prompt_usd_per_token,
        completion_usd_per_token,
    )
    seed = int(sha256(brief.brief_id.encode("utf-8")).hexdigest()[:8], 16)
    return {
        model.max_token_parameter: 512,
        "messages": [
            {
                "content": (
                    "Write one short, complete children's story that a typical "
                    "3- to 4-year-old can understand. Follow the supplied released "
                    "TinyStories instruction closely. Use ordinary story prose only. "
                    "Do not mention prompts, schemas, required words, or this request."
                ),
                "role": "system",
            },
            {
                "content": _neutral_user_prompt(brief),
                "role": "user",
            },
        ],
        "model": model.request_model_id,
        "plugins": [],
        "provider": {
            "allow_fallbacks": False,
            "data_collection": "deny",
            "enforce_distillable_text": True,
            "max_price": {
                "completion": _per_million_price(completion_usd_per_token),
                "prompt": _per_million_price(prompt_usd_per_token),
            },
            "only": [provider_slug],
            "quantizations": [provider_quantization],
            "require_parameters": True,
        },
        "response_format": {
            "json_schema": {
                "name": "tinyworlds_v2_neutral_story_v1",
                "schema": _story_response_schema(brief),
                "strict": True,
            },
            "type": "json_schema",
        },
        "seed": seed,
        "stream": False,
        "transforms": [],
    }


def verifier_request_body(
    brief: NeutralStoryBrief,
    story: str,
    *,
    provider_slug: str,
    provider_quantization: str,
    prompt_usd_per_token: str,
    completion_usd_per_token: str,
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
    seed_material = f"verifier\0{brief.brief_id}\0{sha256(story.encode('utf-8')).hexdigest()}"
    seed = int(sha256(seed_material.encode("utf-8")).hexdigest()[:8], 16)
    return {
        VERIFIER_MODEL.max_token_parameter: 256,
        "messages": [
            {
                "content": (
                    "Judge children's stories with the fixed rubric. Do not infer or "
                    "guess whether a story is human-written or model-generated. Score "
                    "only the text and its adherence to the supplied story brief."
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
        "provider": {
            "allow_fallbacks": False,
            "data_collection": "deny",
            "enforce_distillable_text": True,
            "max_price": {
                "completion": _per_million_price(completion_usd_per_token),
                "prompt": _per_million_price(prompt_usd_per_token),
            },
            "only": [provider_slug],
            "quantizations": [provider_quantization],
            "require_parameters": True,
        },
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

    story_words = tuple(match.group(0).casefold() for match in _WORD_RE.finditer(payload.story))
    required_words_present = all(
        required.casefold() in story_words for required in brief.required_words
    )
    evidence_valid = _evidence_is_valid(brief, payload)
    # Ordinary hyphenated prose such as ``3-year-old`` remains valid.  A
    # hyphen/apostrophe segment containing both a letter and a digit (``fox7``,
    # ``R2-D2``) is the machine-like form that the profile and quality gate also
    # measure.  Keeping this test on the shared tokenizer prevents acceptance
    # and the reported identifier-token rate from drifting.
    mixed_alphanumeric = token_form_counts(lexical_tokens(payload.story))[2] > 0
    forbidden = mixed_alphanumeric or any(
        pattern.search(payload.story) for pattern in _FORBIDDEN_PATTERNS
    )
    length_valid = 40 <= len(story_words) <= 600
    reasons: list[str] = []
    if not required_words_present:
        reasons.append("required_words_missing")
    if not evidence_valid:
        reasons.append("evidence_invalid")
    if forbidden:
        reasons.append("forbidden_identifier_or_meta_language")
    if not length_valid:
        reasons.append("story_length_outside_safety_bounds")
    return payload, StoryValidation(
        schema_valid=True,
        required_words_present=required_words_present,
        evidence_valid=evidence_valid,
        forbidden_identifier_present=forbidden,
        length_valid=length_valid,
        accepted=not reasons,
        rejection_reasons=tuple(reasons),
        story_sha256=sha256(payload.story.encode("utf-8")).hexdigest(),
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


def _neutral_user_prompt(brief: NeutralStoryBrief) -> str:
    features = ", ".join(brief.requested_features) or "none"
    return (
        f"RELEASED TINYSTORIES INSTRUCTION:\n{brief.prompt_text}\n\n"
        f"REQUIRED WORDS: {', '.join(brief.required_words)}\n"
        f"REQUESTED NARRATIVE FEATURES: {features}\n\n"
        "Return the story plus exact quotes showing each required word and each "
        "requested feature. Quotes must occur verbatim in the story."
    )


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
    "VERIFIER_MODEL",
    "CandidateModelSpec",
    "EvidenceQuote",
    "GeneratedStoryPayload",
    "NeutralStoryBrief",
    "StoryValidation",
    "VerifierPayload",
    "assistant_message_content",
    "neutral_story_request_body",
    "parse_generated_story_payload",
    "parse_verifier_payload",
    "request_body_sha256",
    "validate_generated_story",
    "verifier_request_body",
]

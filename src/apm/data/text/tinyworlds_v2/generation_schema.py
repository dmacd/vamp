"""Immutable request, route, response, and retry records for v2 generation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
import re

from apm.data.text.tinyworlds_v2.json_contracts import (
    JsonObject,
    JsonValue,
    bytes_sha256,
    canonical_json_bytes,
    json_sha256,
    require_json_object,
    strict_json_loads,
)


_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]*")
_SENSITIVE_KEYS = frozenset(
    {"api_key", "apikey", "authorization", "openrouter_api_key"}
)
_REQUEST_FORMAT = "apm.tinyworlds-v2.generation-request"
_REQUEST_SCHEMA_VERSION = 2
OPENROUTER_TRANSPORT_PROTOCOL = (
    "openrouter-chat-completions-v2-metadata-enabled-cache-disabled"
)
_ROUTE_LOCK_IDENTITY_FORMAT = "apm.tinyworlds-v2.route-lock-identity"
_ROUTE_LOCK_IDENTITY_SCHEMA_VERSION = 1


class GenerationContractError(ValueError):
    """A generation record violates a strict TinyWorlds-v2 contract."""


@dataclass(frozen=True, slots=True)
class CatalogRoute:
    """One provider endpoint observed in an exact OpenRouter catalog snapshot."""

    catalog_sha256: str
    requested_model: str
    canonical_model: str
    provider_slug: str
    returned_provider: str
    quantization: str
    input_usd_per_million: str
    output_usd_per_million: str

    def __post_init__(self) -> None:
        _require_sha256(self.catalog_sha256, "catalog_sha256")
        for label, value in (
            ("requested_model", self.requested_model),
            ("canonical_model", self.canonical_model),
            ("provider_slug", self.provider_slug),
            ("returned_provider", self.returned_provider),
            ("quantization", self.quantization),
        ):
            _require_nonempty(value, label)
        _require_non_4bit_quantization(self.quantization)
        _require_price(self.input_usd_per_million, "input price")
        _require_price(self.output_usd_per_million, "output price")

    def as_record(self) -> JsonObject:
        """Return the exact canonical identity fields for this observation."""
        return {
            "canonical_model": self.canonical_model,
            "catalog_sha256": self.catalog_sha256,
            "input_usd_per_million": self.input_usd_per_million,
            "output_usd_per_million": self.output_usd_per_million,
            "provider_slug": self.provider_slug,
            "quantization": self.quantization,
            "requested_model": self.requested_model,
            "returned_provider": self.returned_provider,
        }


@dataclass(frozen=True, slots=True)
class RouteLock:
    """A local route name bound to one exact catalog model and provider endpoint."""

    route_id: str
    catalog_sha256: str
    requested_model: str
    canonical_model: str
    provider_slug: str
    returned_provider: str
    quantization: str
    input_usd_per_million: str
    output_usd_per_million: str

    def __post_init__(self) -> None:
        if _IDENTIFIER_PATTERN.fullmatch(self.route_id) is None:
            raise GenerationContractError(
                "route_id must be a nonempty portable identifier"
            )
        CatalogRoute(
            catalog_sha256=self.catalog_sha256,
            requested_model=self.requested_model,
            canonical_model=self.canonical_model,
            provider_slug=self.provider_slug,
            returned_provider=self.returned_provider,
            quantization=self.quantization,
            input_usd_per_million=self.input_usd_per_million,
            output_usd_per_million=self.output_usd_per_million,
        )

    @property
    def lock_sha256(self) -> str:
        """Return the versioned identity of the route's billable semantics."""
        return json_sha256(
            {
                "canonical_model": self.canonical_model,
                "format": _ROUTE_LOCK_IDENTITY_FORMAT,
                "input_usd_per_million": self.input_usd_per_million,
                "output_usd_per_million": self.output_usd_per_million,
                "provider_slug": self.provider_slug,
                "quantization": self.quantization,
                "requested_model": self.requested_model,
                "returned_provider": self.returned_provider,
                "route_id": self.route_id,
                "schema_version": _ROUTE_LOCK_IDENTITY_SCHEMA_VERSION,
            }
        )

    def as_record(self) -> JsonObject:
        """Return a canonical JSON-compatible route-lock record."""
        return {
            "canonical_model": self.canonical_model,
            "catalog_sha256": self.catalog_sha256,
            "input_usd_per_million": self.input_usd_per_million,
            "output_usd_per_million": self.output_usd_per_million,
            "provider_slug": self.provider_slug,
            "quantization": self.quantization,
            "requested_model": self.requested_model,
            "returned_provider": self.returned_provider,
            "route_id": self.route_id,
        }


@dataclass(frozen=True, slots=True)
class CanonicalRequest:
    """An exact HTTP request body with a credential-free content identity."""

    request_sha256: str
    route_lock_sha256: str
    method: str
    endpoint: str
    transport_protocol: str
    body_json: str

    def __post_init__(self) -> None:
        _require_sha256(self.request_sha256, "request_sha256")
        _require_sha256(self.route_lock_sha256, "route_lock_sha256")
        if self.method != "POST":
            raise GenerationContractError("generation requests must use POST")
        if (
            type(self.transport_protocol) is not str
            or _IDENTIFIER_PATTERN.fullmatch(self.transport_protocol) is None
        ):
            raise GenerationContractError(
                "transport_protocol must be a portable nonempty identifier"
            )
        if (
            type(self.endpoint) is not str
            or not self.endpoint.startswith("/")
            or self.endpoint.startswith("//")
            or "?" in self.endpoint
            or "#" in self.endpoint
        ):
            raise GenerationContractError(
                "endpoint must be an absolute URL path without query or fragment"
            )
        if type(self.body_json) is not str:
            raise TypeError("body_json must be a string")
        try:
            body_bytes = self.body_json.encode("utf-8")
        except UnicodeEncodeError as error:
            raise GenerationContractError("body_json must be valid UTF-8") from error
        body = require_json_object(
            strict_json_loads(body_bytes, label="request body"),
            label="request body",
        )
        if body_bytes != canonical_json_bytes(body):
            raise GenerationContractError("request body must use canonical JSON")
        if _contains_sensitive_key(body):
            raise GenerationContractError(
                "request body cannot contain API credentials"
            )
        expected = json_sha256(
            _request_identity_record(
                route_lock_sha256=self.route_lock_sha256,
                method=self.method,
                endpoint=self.endpoint,
                transport_protocol=self.transport_protocol,
                body_bytes=body_bytes,
            )
        )
        if self.request_sha256 != expected:
            raise GenerationContractError("request SHA-256 mismatch")

    @classmethod
    def from_body(
        cls,
        *,
        route_lock_sha256: str,
        endpoint: str,
        body: JsonObject,
        transport_protocol: str = OPENROUTER_TRANSPORT_PROTOCOL,
    ) -> "CanonicalRequest":
        """Freeze a JSON body and derive its credential-free request identity."""
        body_json = canonical_json_bytes(body).decode("utf-8")
        request_sha256 = json_sha256(
            _request_identity_record(
                route_lock_sha256=route_lock_sha256,
                method="POST",
                endpoint=endpoint,
                transport_protocol=transport_protocol,
                body_bytes=body_json.encode("utf-8"),
            )
        )
        return cls(
            request_sha256=request_sha256,
            route_lock_sha256=route_lock_sha256,
            method="POST",
            endpoint=endpoint,
            transport_protocol=transport_protocol,
            body_json=body_json,
        )

    @property
    def body_bytes(self) -> bytes:
        """Return the exact canonical bytes sent to the HTTP transport."""
        return self.body_json.encode("utf-8")

    @property
    def body(self) -> JsonObject:
        """Decode the frozen request body as a fresh JSON object."""
        return require_json_object(
            strict_json_loads(self.body_bytes, label="request body"),
            label="request body",
        )

    def as_record(self) -> JsonObject:
        """Return request metadata; exact body bytes are persisted separately."""
        return {
            "body_sha256": bytes_sha256(self.body_bytes),
            "endpoint": self.endpoint,
            "format": _REQUEST_FORMAT,
            "method": self.method,
            "request_sha256": self.request_sha256,
            "route_lock_sha256": self.route_lock_sha256,
            "schema_version": _REQUEST_SCHEMA_VERSION,
            "transport_protocol": self.transport_protocol,
        }


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Complete token counts reported for one response.

    Billing deliberately is not part of this contract.  OpenRouter's
    generation-stats endpoint can report a cost even when the completion did
    not include complete token usage (notably for some failed generations).
    ``RawHttpResponse.billed_cost_usd`` therefore owns the independent billing
    observation.
    """

    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_input_tokens: int

    def __post_init__(self) -> None:
        for label, value in (
            ("input_tokens", self.input_tokens),
            ("output_tokens", self.output_tokens),
            ("total_tokens", self.total_tokens),
            ("cached_input_tokens", self.cached_input_tokens),
        ):
            if type(value) is not int or value < 0:
                raise GenerationContractError(f"{label} must be nonnegative")
        if self.total_tokens < self.input_tokens + self.output_tokens:
            raise GenerationContractError(
                "total_tokens cannot be smaller than input plus output"
            )
        if self.cached_input_tokens > self.input_tokens:
            raise GenerationContractError(
                "cached_input_tokens cannot exceed input_tokens"
            )
    def as_record(self) -> JsonObject:
        """Return a canonical JSON-compatible usage record."""
        return {
            "cached_input_tokens": self.cached_input_tokens,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True, slots=True)
class ResponseProvenance:
    """Model, provider, and generation identity returned by OpenRouter."""

    generation_id: str
    requested_model: str
    returned_model: str
    returned_provider: str

    def __post_init__(self) -> None:
        for label, value in (
            ("generation_id", self.generation_id),
            ("requested_model", self.requested_model),
            ("returned_model", self.returned_model),
            ("returned_provider", self.returned_provider),
        ):
            _require_nonempty(value, label)

    def as_record(self) -> JsonObject:
        """Return a canonical JSON-compatible provenance record."""
        return {
            "generation_id": self.generation_id,
            "requested_model": self.requested_model,
            "returned_model": self.returned_model,
            "returned_provider": self.returned_provider,
        }


@dataclass(frozen=True, slots=True)
class RawGenerationStatsResponse:
    """Exact authenticated generation-stats response retained for provenance.

    OpenRouter's completion response does not guarantee that provider identity
    or billed cost is present.  When either value has to be recovered from the
    generation-stats endpoint, the complete HTTP observation is part of the
    immutable raw attempt rather than being reduced to parsed fields alone.
    """

    status_code: int
    headers: tuple[tuple[str, str], ...]
    body: bytes

    def __post_init__(self) -> None:
        if type(self.status_code) is not int or not 100 <= self.status_code <= 599:
            raise GenerationContractError(
                "generation-stats status_code must be an HTTP status"
            )
        if type(self.headers) is not tuple or any(
            type(header) is not tuple
            or len(header) != 2
            or any(type(part) is not str for part in header)
            for header in self.headers
        ):
            raise GenerationContractError(
                "generation-stats headers must be an ordered tuple of string pairs"
            )
        if type(self.body) is not bytes:
            raise TypeError("generation-stats body must be bytes")


@dataclass(frozen=True, slots=True)
class RawGenerationStatsAttempt:
    """One append-only generation-stats HTTP observation or transport error."""

    attempt_number: int
    observed_at_utc: str
    response: RawGenerationStatsResponse | None
    transport_error_type: str | None
    billed_cost_usd: str | None = None

    def __post_init__(self) -> None:
        if type(self.attempt_number) is not int or self.attempt_number < 1:
            raise GenerationContractError(
                "generation-stats attempt_number must be positive"
            )
        _require_timestamp(self.observed_at_utc, "generation-stats observed_at_utc")
        if (self.response is None) == (self.transport_error_type is None):
            raise GenerationContractError(
                "generation-stats attempt must contain exactly one response or "
                "transport error"
            )
        if self.response is not None and type(self.response) is not RawGenerationStatsResponse:
            raise TypeError(
                "generation-stats response must be RawGenerationStatsResponse or None"
            )
        if self.transport_error_type is not None and (
            _IDENTIFIER_PATTERN.fullmatch(self.transport_error_type) is None
        ):
            raise GenerationContractError(
                "generation-stats transport_error_type must be a portable class "
                "identifier"
            )
        if self.billed_cost_usd is not None:
            if self.response is None:
                raise GenerationContractError(
                    "generation-stats billed cost requires an HTTP response"
                )
            _require_price(self.billed_cost_usd, "generation-stats billed cost")


@dataclass(frozen=True, slots=True)
class RawHttpResponse:
    """Exact response body and headers plus parsed billing/provenance metadata."""

    status_code: int
    headers: tuple[tuple[str, str], ...]
    body: bytes
    provenance: ResponseProvenance | None = None
    usage: TokenUsage | None = None
    billed_cost_usd: str | None = None
    generation_stats_attempts: tuple[RawGenerationStatsAttempt, ...] = ()

    def __post_init__(self) -> None:
        if type(self.status_code) is not int or not 100 <= self.status_code <= 599:
            raise GenerationContractError("status_code must be an HTTP status")
        if type(self.headers) is not tuple or any(
            type(header) is not tuple
            or len(header) != 2
            or any(type(part) is not str for part in header)
            for header in self.headers
        ):
            raise GenerationContractError(
                "headers must be an ordered tuple of string pairs"
            )
        if type(self.body) is not bytes:
            raise TypeError("response body must be bytes")
        if self.provenance is not None and type(self.provenance) is not ResponseProvenance:
            raise TypeError("provenance must be ResponseProvenance or None")
        if self.usage is not None and type(self.usage) is not TokenUsage:
            raise TypeError("usage must be TokenUsage or None")
        if self.billed_cost_usd is not None:
            _require_price(self.billed_cost_usd, "billed cost")
        if type(self.generation_stats_attempts) is not tuple or any(
            type(attempt) is not RawGenerationStatsAttempt
            for attempt in self.generation_stats_attempts
        ):
            raise TypeError(
                "generation_stats_attempts must contain RawGenerationStatsAttempt values"
            )
        numbers = tuple(
            attempt.attempt_number for attempt in self.generation_stats_attempts
        )
        if numbers != tuple(range(1, len(numbers) + 1)):
            raise GenerationContractError(
                "generation-stats attempts must be contiguous and ordered"
            )


@dataclass(frozen=True, slots=True)
class RawAttempt:
    """One immutable HTTP response or sanitized transport-failure observation."""

    request_sha256: str
    attempt_number: int
    observed_at_utc: str
    submission_catalog_sha256: str
    response: RawHttpResponse | None
    transport_error_type: str | None

    def __post_init__(self) -> None:
        _require_sha256(self.request_sha256, "request_sha256")
        if type(self.attempt_number) is not int or self.attempt_number < 1:
            raise GenerationContractError("attempt_number must be positive")
        _require_timestamp(self.observed_at_utc, "observed_at_utc")
        _require_sha256(
            self.submission_catalog_sha256,
            "submission_catalog_sha256",
        )
        if (self.response is None) == (self.transport_error_type is None):
            raise GenerationContractError(
                "attempt must contain exactly one response or transport error"
            )
        if self.response is not None and type(self.response) is not RawHttpResponse:
            raise TypeError("response must be RawHttpResponse or None")
        if self.transport_error_type is not None:
            if _IDENTIFIER_PATTERN.fullmatch(self.transport_error_type) is None:
                raise GenerationContractError(
                    "transport_error_type must be a portable class identifier"
                )


def _request_identity_record(
    *,
    route_lock_sha256: str,
    method: str,
    endpoint: str,
    transport_protocol: str,
    body_bytes: bytes,
) -> JsonObject:
    return {
        "body_sha256": bytes_sha256(body_bytes),
        "endpoint": endpoint,
        "format": _REQUEST_FORMAT,
        "method": method,
        "route_lock_sha256": route_lock_sha256,
        "schema_version": _REQUEST_SCHEMA_VERSION,
        "transport_protocol": transport_protocol,
    }


def _contains_sensitive_key(value: JsonValue) -> bool:
    if type(value) is dict:
        return any(
            key.casefold() in _SENSITIVE_KEYS or _contains_sensitive_key(item)
            for key, item in value.items()
        )
    if type(value) is list:
        return any(_contains_sensitive_key(item) for item in value)
    return False


def _require_sha256(value: str, label: str) -> None:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise GenerationContractError(f"{label} must be a lowercase SHA-256 digest")


def _require_nonempty(value: str, label: str) -> None:
    if type(value) is not str or not value.strip() or value != value.strip():
        raise GenerationContractError(f"{label} must be a trimmed nonempty string")


def _require_price(value: str, label: str) -> None:
    if type(value) is not str:
        raise TypeError(f"{label} must be a decimal string")
    try:
        price = Decimal(value)
    except InvalidOperation as error:
        raise GenerationContractError(f"{label} must be a decimal string") from error
    if not price.is_finite() or price < 0:
        raise GenerationContractError(f"{label} must be finite and nonnegative")


def _require_non_4bit_quantization(quantization: str) -> None:
    normalized = quantization.casefold().replace("-", "").replace("_", "")
    if (
        normalized in {"4bit", "fp4", "int4", "nf4"}
        or normalized.endswith("4bit")
        or normalized.startswith("q4")
    ):
        raise GenerationContractError("4-bit provider routes are forbidden")


def _require_timestamp(value: str, label: str) -> None:
    if type(value) is not str:
        raise TypeError(f"{label} must be a string")
    try:
        observed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise GenerationContractError(
            f"{label} must be an ISO-8601 timestamp"
        ) from error
    if observed.tzinfo is None:
        raise GenerationContractError(f"{label} must include a timezone")

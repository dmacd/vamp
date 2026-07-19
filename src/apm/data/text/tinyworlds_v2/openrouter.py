"""Injected-transport OpenRouter client with bounded, auditable retries."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from threading import Event
import time
from typing import Callable, Protocol
from urllib.parse import quote

from apm.data.text.tinyworlds_v2.generation_cache import ImmutableRawCache
from apm.data.text.tinyworlds_v2.generation_costs import RuntimeCostLedger
from apm.data.text.tinyworlds_v2.generation_schema import (
    CanonicalRequest,
    GenerationContractError,
    RawAttempt,
    RawGenerationStatsAttempt,
    RawGenerationStatsResponse,
    RawHttpResponse,
    ResponseProvenance,
    RouteLock,
    TokenUsage,
    OPENROUTER_TRANSPORT_PROTOCOL,
)
from apm.data.text.tinyworlds_v2.json_contracts import (
    JsonObject,
    JsonValue,
    bytes_sha256,
    canonical_json_bytes,
    require_exact_fields,
    require_json_object,
    strict_json_loads,
)
from apm.data.text.tinyworlds_v2.route_lock import validate_locked_request_body


class TransportError(OSError):
    """A request failed before an HTTP response was received."""

    def __init__(self, error_kind: str) -> None:
        if (
            type(error_kind) is not str
            or not error_kind
            or not all(character.isalnum() or character in "_.-" for character in error_kind)
        ):
            raise GenerationContractError(
                "transport error kind must be a portable identifier"
            )
        super().__init__(error_kind)
        self.error_kind = error_kind


@dataclass(frozen=True, slots=True)
class TransportResponse:
    """Exact response returned by an injected HTTP transport."""

    status_code: int
    headers: tuple[tuple[str, str], ...]
    body: bytes


@dataclass(frozen=True, slots=True)
class ByokPreflightEvidence:
    """Sanitized proof that the authenticated workspace has no BYOK keys."""

    checked_at_utc: str
    source: str
    status_code: int | None
    response_body_sha256: str | None
    total_count: int | None
    decision: str
    attestation_sha256: str | None = None
    attested_at_utc: str | None = None
    expires_at_utc: str | None = None

    def as_record(self) -> JsonObject:
        return {
            "checked_at_utc": self.checked_at_utc,
            "decision": self.decision,
            "endpoint": (
                "/api/v1/byok" if self.source == "management_api" else None
            ),
            "method": "GET" if self.source == "management_api" else None,
            "response_body_sha256": self.response_body_sha256,
            "source": self.source,
            "status_code": self.status_code,
            "total_count": self.total_count,
            "attestation_sha256": self.attestation_sha256,
            "attested_at_utc": self.attested_at_utc,
            "expires_at_utc": self.expires_at_utc,
        }


@dataclass(frozen=True, slots=True)
class _StatsFields:
    generation_id: str | None
    returned_model: str | None
    returned_provider: str | None
    billed_cost_usd: str | None
    input_tokens: int | None
    output_tokens: int | None
    is_byok: bool | None = None
    upstream_inference_cost_usd: str | None = None


class HttpTransport(Protocol):
    """Minimal synchronous transport boundary used by the generation client."""

    def post(
        self,
        *,
        url: str,
        headers: tuple[tuple[str, str], ...],
        body: bytes,
        timeout_seconds: float,
    ) -> TransportResponse:
        """Send exact bytes and return exact response bytes and headers."""

    def get_authenticated(
        self,
        *,
        url: str,
        headers: tuple[tuple[str, str], ...],
        timeout_seconds: float,
    ) -> TransportResponse:
        """Fetch authenticated generation stats and retain the exact response."""


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """A bounded exponential retry schedule for transport, 429, and 5xx only."""

    max_attempts: int = 3
    initial_delay_seconds: float = 1.0
    maximum_delay_seconds: float = 8.0

    def __post_init__(self) -> None:
        if type(self.max_attempts) is not int or not 1 <= self.max_attempts <= 8:
            raise GenerationContractError("max_attempts must be between 1 and 8")
        for label, value in (
            ("initial_delay_seconds", self.initial_delay_seconds),
            ("maximum_delay_seconds", self.maximum_delay_seconds),
        ):
            if type(value) not in (int, float) or value < 0:
                raise GenerationContractError(f"{label} must be nonnegative")
        if self.initial_delay_seconds > self.maximum_delay_seconds:
            raise GenerationContractError(
                "initial retry delay cannot exceed maximum retry delay"
            )

    def delay_after(self, attempt_number: int) -> float:
        """Return the capped delay after a failed one-indexed attempt."""
        if type(attempt_number) is not int or attempt_number < 1:
            raise GenerationContractError("attempt_number must be positive")
        return min(
            float(self.maximum_delay_seconds),
            float(self.initial_delay_seconds) * (2 ** (attempt_number - 1)),
        )


class OpenRouterError(RuntimeError):
    """Base class for generation-client failures."""


class OpenRouterContractError(OpenRouterError):
    """A response lacks required provenance or violates its route lock."""


class _SecretReflectionError(OpenRouterContractError):
    """An authenticated response echoed an in-memory credential."""


_AUTHENTICATION_SECRET_REFLECTION = "authentication_secret_reflection"


class OpenRouterResponseError(OpenRouterError):
    """OpenRouter returned a terminal non-2xx HTTP response."""

    def __init__(self, status_code: int, body_sha256: str) -> None:
        super().__init__(
            f"OpenRouter returned HTTP {status_code}; body SHA-256={body_sha256}"
        )
        self.status_code = status_code
        self.body_sha256 = body_sha256


class OpenRouterRetryExhausted(OpenRouterError):
    """All authorized attempts were consumed by retryable failures."""


class OpenRouterBillingUnknown(OpenRouterContractError):
    """A received HTTP response has no provider-reported actual cost."""


class OpenRouterCostPolicyError(OpenRouterBillingUnknown):
    """OpenRouter billing cannot be bounded because BYOK is possible or observed."""

    def __init__(
        self,
        message: str,
        *,
        evidence: ByokPreflightEvidence | None = None,
    ) -> None:
        super().__init__(message)
        self.evidence = evidence


@dataclass(frozen=True, slots=True)
class OpenRouterClient:
    """Offline-testable client that caches every response before interpreting it."""

    api_key: str = field(repr=False, compare=False)
    transport: HttpTransport = field(repr=False, compare=False)
    cache: ImmutableRawCache
    retry_policy: RetryPolicy = RetryPolicy()
    stats_retry_policy: RetryPolicy = RetryPolicy(
        max_attempts=4,
        initial_delay_seconds=0.25,
        maximum_delay_seconds=2.0,
    )
    cost_ledger: RuntimeCostLedger = field(
        default_factory=RuntimeCostLedger,
        repr=False,
        compare=False,
    )
    base_url: str = "https://openrouter.ai"
    timeout_seconds: float = 120.0
    management_api_key: str | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    byok_attestation_path: Path | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    require_byok_preflight: bool = False
    _byok_preflight_passed: Event = field(
        default_factory=Event,
        repr=False,
        compare=False,
    )
    sleeper: Callable[[float], None] = field(
        default=time.sleep,
        repr=False,
        compare=False,
    )
    clock: Callable[[], str] = field(
        default=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if (
            type(self.api_key) is not str
            or not self.api_key
            or self.api_key != self.api_key.strip()
            or "\n" in self.api_key
            or "\r" in self.api_key
        ):
            raise GenerationContractError("api_key must be a single nonempty line")
        if (
            type(self.base_url) is not str
            or not self.base_url.startswith("https://")
            or self.base_url.endswith("/")
        ):
            raise GenerationContractError(
                "base_url must be an HTTPS origin without a trailing slash"
            )
        if type(self.timeout_seconds) not in (int, float) or self.timeout_seconds <= 0:
            raise GenerationContractError("timeout_seconds must be positive")
        if type(self.require_byok_preflight) is not bool:
            raise GenerationContractError("require_byok_preflight must be boolean")
        if self.management_api_key is not None:
            if (
                type(self.management_api_key) is not str
                or not self.management_api_key
                or self.management_api_key != self.management_api_key.strip()
                or "\n" in self.management_api_key
                or "\r" in self.management_api_key
            ):
                raise GenerationContractError(
                    "management_api_key must be a single nonempty line"
                )
            if self.management_api_key == self.api_key:
                raise GenerationContractError(
                    "management and inference API keys must be distinct"
                )
        if self.byok_attestation_path is not None and not isinstance(
            self.byok_attestation_path,
            Path,
        ):
            raise TypeError("byok_attestation_path must be a Path or None")

    def verify_no_byok(self) -> ByokPreflightEvidence:
        """Require an authenticated, sensitive-data-free proof of zero BYOK keys."""
        if self.management_api_key is None:
            if self.byok_attestation_path is None:
                raise OpenRouterCostPolicyError(
                    "BYOK safety requires a distinct management key or a valid "
                    "manual zero-BYOK attestation"
                )
            evidence = _manual_byok_attestation_evidence(
                self.byok_attestation_path,
                self.clock(),
            )
            self.cost_ledger.authorize_byok(evidence.as_record())
            self._byok_preflight_passed.set()
            return evidence
        try:
            response = self.transport.get_authenticated(
                url=f"{self.base_url}/api/v1/byok",
                headers=(
                    ("Authorization", f"Bearer {self.management_api_key}"),
                    ("Accept", "application/json"),
                ),
                timeout_seconds=float(self.timeout_seconds),
            )
        except TransportError as error:
            error_kind = self._safe_transport_error_kind(error)
            raise OpenRouterCostPolicyError(
                "BYOK preflight transport failed; paid work remains disabled "
                f"({error_kind})"
            ) from None
        if type(response) is not TransportResponse:
            raise TypeError("transport must return a TransportResponse")
        checked_at = self.clock()
        if _response_reflects_authentication_secret(
            response,
            self._authentication_secrets(),
        ):
            evidence = ByokPreflightEvidence(
                checked_at_utc=checked_at,
                source="management_api",
                status_code=response.status_code,
                response_body_sha256=None,
                total_count=None,
                decision="unverified",
            )
            raise OpenRouterCostPolicyError(
                "BYOK preflight reflected an authentication secret; its response "
                "was not hashed or persisted and paid work remains disabled",
                evidence=evidence,
            )
        body_sha256 = bytes_sha256(response.body)
        if response.status_code != 200:
            evidence = ByokPreflightEvidence(
                checked_at_utc=checked_at,
                source="management_api",
                status_code=response.status_code,
                response_body_sha256=body_sha256,
                total_count=None,
                decision="unverified",
            )
            raise OpenRouterCostPolicyError(
                "BYOK preflight did not return HTTP 200; paid work remains "
                "disabled",
                evidence=evidence,
            )
        try:
            record = require_json_object(
                strict_json_loads(response.body, label="OpenRouter BYOK preflight"),
                label="OpenRouter BYOK preflight",
            )
        except ValueError as error:
            evidence = ByokPreflightEvidence(
                checked_at_utc=checked_at,
                source="management_api",
                status_code=response.status_code,
                response_body_sha256=body_sha256,
                total_count=None,
                decision="unverified",
            )
            raise OpenRouterCostPolicyError(
                "BYOK preflight response is malformed; paid work remains disabled",
                evidence=evidence,
            ) from error
        data, total_count = record.get("data"), record.get("total_count")
        if type(data) is not list or type(total_count) is not int or total_count < 0:
            evidence = ByokPreflightEvidence(
                checked_at_utc=checked_at,
                source="management_api",
                status_code=response.status_code,
                response_body_sha256=body_sha256,
                total_count=None,
                decision="unverified",
            )
            raise OpenRouterCostPolicyError(
                "BYOK preflight response lacks a valid data/total_count contract",
                evidence=evidence,
            )
        decision = "allowed" if total_count == 0 and not data else "blocked"
        evidence = ByokPreflightEvidence(
            checked_at_utc=checked_at,
            source="management_api",
            status_code=response.status_code,
            response_body_sha256=body_sha256,
            total_count=total_count,
            decision=decision,
        )
        if decision != "allowed":
            raise OpenRouterCostPolicyError(
                "authenticated workspace contains BYOK configuration; paid work "
                "remains disabled",
                evidence=evidence,
            )
        self.cost_ledger.authorize_byok(evidence.as_record())
        self._byok_preflight_passed.set()
        return evidence

    def _authentication_secrets(self) -> tuple[str, ...]:
        """Return in-memory credentials solely for response-reflection checks."""
        return (
            (self.api_key,)
            if self.management_api_key is None
            else (self.api_key, self.management_api_key)
        )

    def _safe_transport_error_kind(self, error: TransportError) -> str:
        """Keep an injected transport error from reflecting a credential."""
        return (
            _AUTHENTICATION_SECRET_REFLECTION
            if any(
                secret in error.error_kind
                for secret in self._authentication_secrets()
            )
            else error.error_kind
        )

    def generate(
        self,
        request: CanonicalRequest,
        route_lock: RouteLock,
    ) -> RawHttpResponse:
        """Return a cached or fresh locked response, retrying only allowed failures."""
        validate_locked_request_body(route_lock, request)
        if request.transport_protocol != OPENROUTER_TRANSPORT_PROTOCOL:
            raise OpenRouterContractError(
                "request uses an unsupported completion transport protocol"
            )
        if self.require_byok_preflight and not self._byok_preflight_passed.is_set():
            raise OpenRouterCostPolicyError(
                "paid work requires a successful zero-BYOK preflight"
            )
        self.cost_ledger.attach_cache(self.cache)
        self.cache.prepare_request(request, route_lock)
        attempts = self.cache.load_attempts(request)
        self.cost_ledger.reconcile_cached(request, route_lock, attempts)
        self.cost_ledger.ensure_recovery_complete()
        if any(
            attempt.transport_error_type == _AUTHENTICATION_SECRET_REFLECTION
            for attempt in attempts
        ):
            raise _SecretReflectionError(
                "cached completion attempt records an authentication-secret "
                "reflection; no retry is permitted"
            )
        if any(attempt.response is None for attempt in attempts):
            raise OpenRouterBillingUnknown(
                "cached completion transport failure has ambiguous billing and "
                "cannot be retried safely"
            )
        resolved_responses: dict[int, RawHttpResponse] = {}
        try:
            for attempt in attempts:
                if attempt.response is not None:
                    resolved_responses[attempt.attempt_number] = (
                        self._ensure_stats_complete(request, route_lock, attempt)
                    )
        except OpenRouterCostPolicyError:
            self.cost_ledger.halt("provider_cost_policy_violation")
            raise
        except OpenRouterBillingUnknown:
            self.cost_ledger.halt("provider_billing_unknown")
            raise
        except OpenRouterContractError:
            self.cost_ledger.halt("provider_response_contract_failure")
            raise
        attempts = self.cache.load_attempts(request)
        self.cost_ledger.reconcile_cached(request, route_lock, attempts)
        try:
            for attempt in attempts:
                if attempt.response is not None:
                    response = resolved_responses[attempt.attempt_number]
                    _validate_observation(response, route_lock)
                    if response.billed_cost_usd is None:
                        raise OpenRouterBillingUnknown(
                            "cached HTTP response lacks provider-reported actual cost"
                        )
        except OpenRouterCostPolicyError:
            self.cost_ledger.halt("provider_cost_policy_violation")
            raise
        except OpenRouterBillingUnknown:
            self.cost_ledger.halt("provider_billing_unknown")
            raise
        except OpenRouterContractError:
            self.cost_ledger.halt("provider_response_contract_failure")
            raise
        cached_success = next(
            (
                resolved_responses[attempt.attempt_number]
                for attempt in attempts
                if attempt.response is not None
                and 200 <= attempt.response.status_code < 300
            ),
            None,
        )
        if cached_success is not None:
            try:
                _validate_success(cached_success, route_lock)
            except OpenRouterCostPolicyError:
                self.cost_ledger.halt("provider_cost_policy_violation")
                raise
            except OpenRouterContractError:
                self.cost_ledger.halt("provider_response_contract_failure")
                raise
            return cached_success
        terminal_response = next(
            (
                resolved_responses[attempt.attempt_number]
                for attempt in reversed(attempts)
                if attempt.response is not None
                and not _is_retryable_status(attempt.response.status_code)
            ),
            None,
        )
        if terminal_response is not None:
            raise OpenRouterResponseError(
                terminal_response.status_code,
                bytes_sha256(terminal_response.body),
            )
        if len(attempts) >= self.retry_policy.max_attempts:
            raise OpenRouterRetryExhausted(
                f"request {request.request_sha256} exhausted "
                f"{self.retry_policy.max_attempts} attempts"
            )
        return self._send_remaining_attempts(
            request,
            route_lock,
            first_attempt_number=len(attempts) + 1,
        )

    def _send_remaining_attempts(
        self,
        request: CanonicalRequest,
        route_lock: RouteLock,
        *,
        first_attempt_number: int,
    ) -> RawHttpResponse:
        for attempt_number in range(
            first_attempt_number,
            self.retry_policy.max_attempts + 1,
        ):
            reservation = self.cost_ledger.reserve(
                request,
                route_lock,
                attempt_number,
            )
            self.cost_ledger.ensure_reservation_postable(reservation)
            try:
                transport_response = self.transport.post(
                    url=self.base_url + request.endpoint,
                    headers=(
                        ("Authorization", f"Bearer {self.api_key}"),
                        ("Content-Type", "application/json"),
                        ("X-OpenRouter-Metadata", "enabled"),
                        ("X-OpenRouter-Cache", "false"),
                    ),
                    body=request.body_bytes,
                    timeout_seconds=float(self.timeout_seconds),
                )
            except TransportError as error:
                error_kind = self._safe_transport_error_kind(error)
                try:
                    self.cache.store_attempt(
                        request,
                        route_lock,
                        RawAttempt(
                            request_sha256=request.request_sha256,
                            attempt_number=attempt_number,
                            observed_at_utc=self.clock(),
                            submission_catalog_sha256=route_lock.catalog_sha256,
                            response=None,
                            transport_error_type=error_kind,
                        ),
                    )
                finally:
                    # A timeout or dropped connection can occur after inference
                    # ran. Without a generation id there is no safe billing
                    # lookup, so charge the full bound and do not repeat POST.
                    self.cost_ledger.settle_ambiguous_transport_failure(
                        reservation
                    )
                raise OpenRouterBillingUnknown(
                    "completion transport failed after POST may have been processed; "
                    "its upper bound was charged and the request was not retried"
                ) from None

            if _response_reflects_authentication_secret(
                transport_response,
                self._authentication_secrets(),
            ):
                # A reflected credential cannot be serialized, hashed, or used
                # as billing evidence.  The POST may nevertheless have run, so
                # consume its complete authorized bound and stop all paid work.
                try:
                    self.cache.store_attempt(
                        request,
                        route_lock,
                        RawAttempt(
                            request_sha256=request.request_sha256,
                            attempt_number=attempt_number,
                            observed_at_utc=self.clock(),
                            submission_catalog_sha256=route_lock.catalog_sha256,
                            response=None,
                            transport_error_type=(
                                _AUTHENTICATION_SECRET_REFLECTION
                            ),
                        ),
                    )
                finally:
                    self.cost_ledger.halt("provider_secret_reflection")
                    self.cost_ledger.settle_ambiguous_transport_failure(
                        reservation
                    )
                raise _SecretReflectionError(
                    "completion response reflected an authentication secret; "
                    "the response was not hashed or persisted; only a sanitized "
                    "marker was stored"
                )
            response = _completion_response(transport_response, route_lock)
            try:
                self.cache.store_attempt(
                    request,
                    route_lock,
                    RawAttempt(
                        request_sha256=request.request_sha256,
                        attempt_number=attempt_number,
                        observed_at_utc=self.clock(),
                        submission_catalog_sha256=route_lock.catalog_sha256,
                        response=response,
                        transport_error_type=None,
                    ),
                )
            except Exception:
                # The HTTP response may have been billed even if local durable
                # persistence failed.  Consume actual cost when present and the
                # complete bound otherwise, then halt.
                self.cost_ledger.halt("raw_response_persistence_failure")
                self.cost_ledger.settle_response(
                    reservation,
                    response.billed_cost_usd,
                )
                raise
            resolution_error: Exception | None = None
            try:
                response = self._ensure_stats_complete(
                    request,
                    route_lock,
                    self.cache.load_attempts(request)[attempt_number - 1],
                )
            except Exception as error:
                resolution_error = error
                if isinstance(error, _SecretReflectionError):
                    self.cost_ledger.halt("provider_secret_reflection")
                elif isinstance(error, OpenRouterCostPolicyError):
                    self.cost_ledger.halt("provider_cost_policy_violation")
                elif isinstance(error, OpenRouterBillingUnknown):
                    pass
                elif isinstance(error, OpenRouterContractError):
                    self.cost_ledger.halt("provider_response_contract_failure")
                response = self.cache.load_attempts(request)[attempt_number - 1].response
                assert response is not None
            self.cost_ledger.settle_response(
                reservation,
                response.billed_cost_usd,
            )
            if response.billed_cost_usd is None:
                if isinstance(
                    resolution_error,
                    (_SecretReflectionError, OpenRouterBillingUnknown),
                ):
                    raise resolution_error
                raise OpenRouterBillingUnknown(
                    "HTTP response lacks provider-reported actual cost; its locked "
                    "upper bound was charged to the runtime cap"
                ) from resolution_error
            if resolution_error is not None:
                raise resolution_error
            try:
                _validate_observation(response, route_lock)
            except OpenRouterContractError:
                self.cost_ledger.halt("provider_response_contract_failure")
                raise
            if _is_retryable_status(response.status_code):
                if attempt_number == self.retry_policy.max_attempts:
                    break
                self.sleeper(self.retry_policy.delay_after(attempt_number))
                continue
            if not 200 <= response.status_code < 300:
                raise OpenRouterResponseError(
                    response.status_code,
                    bytes_sha256(response.body),
                )
            try:
                _validate_success(response, route_lock)
            except OpenRouterContractError:
                self.cost_ledger.halt("provider_response_contract_failure")
                raise
            return response
        raise OpenRouterRetryExhausted(
            f"request {request.request_sha256} exhausted "
            f"{self.retry_policy.max_attempts} attempts"
        )

    def _ensure_stats_complete(
        self,
        request: CanonicalRequest,
        route_lock: RouteLock,
        attempt: RawAttempt,
    ) -> RawHttpResponse:
        """Finish one completion's stats lookup without ever repeating its POST."""
        if attempt.response is None:
            raise TypeError("generation-stats lookup requires an HTTP response")
        response = attempt.response
        existing = response.generation_stats_attempts
        _validate_existing_stats_attempts(existing)
        if not _response_needs_stats(response):
            return _derive_response(response, route_lock)
        record = _response_record(response.body)
        # Malformed metadata is a terminal contract failure.  It must not be
        # hidden by querying another endpoint for a more convenient identity.
        _strict_selected_provider(
            record,
            require_selected=200 <= response.status_code < 300,
        )
        generation_id = _strict_generation_id(record, response.headers)
        if generation_id is None:
            if response.billed_cost_usd is None:
                raise OpenRouterBillingUnknown(
                    "HTTP response lacks both generation id and billed cost"
                )
            raise OpenRouterContractError(
                "HTTP response requires generation stats but lacks a generation id"
            )

        if existing:
            response = _derive_response(response, route_lock)
            if not _response_needs_stats(response):
                return response
        for stats_attempt_number in range(
            len(existing) + 1,
            self.stats_retry_policy.max_attempts + 1,
        ):
            if stats_attempt_number > 1:
                self.sleeper(
                    self.stats_retry_policy.delay_after(stats_attempt_number - 1)
                )
            stats_attempt = self._fetch_stats_attempt(
                generation_id,
                stats_attempt_number,
            )
            self.cache.store_generation_stats_attempt(
                request,
                attempt.attempt_number,
                stats_attempt,
            )
            if (
                stats_attempt.transport_error_type
                == _AUTHENTICATION_SECRET_REFLECTION
            ):
                raise _SecretReflectionError(
                    "generation-stats response reflected an authentication "
                    "secret; the response was not hashed or persisted; only a "
                    "sanitized marker was stored and no retry is permitted"
                )
            refreshed = self.cache.load_attempts(request)[attempt.attempt_number - 1]
            assert refreshed.response is not None
            response = refreshed.response
            if stats_attempt.response is not None:
                status = stats_attempt.response.status_code
                if 200 <= status < 300:
                    _strict_stats_fields(stats_attempt.response)
                    response = _derive_response(response, route_lock)
                    if not _response_needs_stats(response):
                        return response
                elif not _is_stats_retryable_status(status):
                    raise OpenRouterContractError(
                        "generation-stats lookup returned terminal HTTP "
                        f"{status}"
                    )

        response = _derive_response(response, route_lock)
        if response.billed_cost_usd is None:
            raise OpenRouterBillingUnknown(
                "generation-stats retries exhausted without provider-reported cost"
            )
        raise OpenRouterContractError(
            "generation-stats retries exhausted without complete provenance/usage"
        )

    def _fetch_stats_attempt(
        self,
        generation_id: str,
        attempt_number: int,
    ) -> RawGenerationStatsAttempt:
        try:
            stats_response = self.transport.get_authenticated(
                url=(
                    f"{self.base_url}/api/v1/generation?id="
                    f"{quote(generation_id, safe='')}"
                ),
                headers=(
                    ("Authorization", f"Bearer {self.api_key}"),
                    ("Accept", "application/json"),
                ),
                timeout_seconds=float(self.timeout_seconds),
            )
        except TransportError as error:
            return RawGenerationStatsAttempt(
                attempt_number=attempt_number,
                observed_at_utc=self.clock(),
                response=None,
                transport_error_type=self._safe_transport_error_kind(error),
            )
        if type(stats_response) is not TransportResponse:
            raise TypeError("transport must return a TransportResponse")
        if _response_reflects_authentication_secret(
            stats_response,
            self._authentication_secrets(),
        ):
            return RawGenerationStatsAttempt(
                attempt_number=attempt_number,
                observed_at_utc=self.clock(),
                response=None,
                transport_error_type=_AUTHENTICATION_SECRET_REFLECTION,
            )
        raw = RawGenerationStatsResponse(
            status_code=stats_response.status_code,
            headers=stats_response.headers,
            body=stats_response.body,
        )
        fields = _stats_fields_tolerant(raw)
        return RawGenerationStatsAttempt(
            attempt_number=attempt_number,
            observed_at_utc=self.clock(),
            response=raw,
            transport_error_type=None,
            billed_cost_usd=fields.billed_cost_usd,
        )


_BYOK_ATTESTATION_FORMAT = "apm.tinyworlds-v2.openrouter-no-byok-attestation"
_BYOK_ATTESTATION_STATEMENT = (
    "I attest that this OpenRouter workspace has zero configured BYOK keys."
)
_MAX_BYOK_ATTESTATION_SECONDS = 24 * 60 * 60


def _response_reflects_authentication_secret(
    response: TransportResponse,
    secrets: tuple[str, ...],
) -> bool:
    """Detect literal or JSON-escaped credential reflection without digesting it."""
    if any(
        secret.encode("utf-8") in response.body
        or any(
            secret in name or secret in value for name, value in response.headers
        )
        for secret in secrets
    ):
        return True
    try:
        parsed = strict_json_loads(response.body, label="authenticated HTTP response")
    except ValueError:
        return False
    return _json_value_reflects_authentication_secret(parsed, secrets)


def _json_value_reflects_authentication_secret(
    value: JsonValue,
    secrets: tuple[str, ...],
) -> bool:
    if type(value) is str:
        return any(secret in value for secret in secrets)
    if type(value) is list:
        return any(
            _json_value_reflects_authentication_secret(item, secrets)
            for item in value
        )
    if type(value) is dict:
        return any(
            any(secret in key for secret in secrets)
            or _json_value_reflects_authentication_secret(item, secrets)
            for key, item in value.items()
        )
    return False


def _manual_byok_attestation_evidence(
    path: Path,
    checked_at_utc: str,
) -> ByokPreflightEvidence:
    """Validate one canonical, local, short-lived, nonsecret attestation."""
    if path.is_symlink() or not path.is_file():
        raise OpenRouterCostPolicyError(
            "manual zero-BYOK attestation is missing or is not a regular file"
        )
    payload = path.read_bytes()
    digest = bytes_sha256(payload)
    unverified = ByokPreflightEvidence(
        checked_at_utc=checked_at_utc,
        source="manual_attestation",
        status_code=None,
        response_body_sha256=None,
        total_count=None,
        decision="unverified",
        attestation_sha256=digest,
    )
    try:
        record = require_json_object(
            strict_json_loads(payload, label="manual zero-BYOK attestation"),
            label="manual zero-BYOK attestation",
        )
        require_exact_fields(
            record,
            (
                "attested_at_utc",
                "expires_at_utc",
                "format",
                "schema_version",
                "statement",
            ),
            label="manual zero-BYOK attestation",
        )
        if payload != canonical_json_bytes(record):
            raise ValueError("manual attestation must use canonical JSON")
        if (
            record["format"] != _BYOK_ATTESTATION_FORMAT
            or record["schema_version"] != 1
            or record["statement"] != _BYOK_ATTESTATION_STATEMENT
        ):
            raise ValueError("manual attestation contract values differ")
        attested_text = _nonempty_string(record["attested_at_utc"])
        expires_text = _nonempty_string(record["expires_at_utc"])
        if attested_text is None or expires_text is None:
            raise ValueError("manual attestation timestamps must be strings")
        checked = _parse_utc_timestamp(checked_at_utc)
        attested = _parse_utc_timestamp(attested_text)
        expires = _parse_utc_timestamp(expires_text)
        lifetime = (expires - attested).total_seconds()
        if not 0 < lifetime <= _MAX_BYOK_ATTESTATION_SECONDS:
            raise ValueError("manual attestation lifetime must be at most 24 hours")
        if not attested <= checked < expires:
            raise ValueError("manual attestation is stale or not yet valid")
    except (TypeError, ValueError) as error:
        raise OpenRouterCostPolicyError(
            "manual zero-BYOK attestation is malformed, stale, or invalid",
            evidence=unverified,
        ) from error
    return ByokPreflightEvidence(
        checked_at_utc=checked_at_utc,
        source="manual_attestation",
        status_code=None,
        response_body_sha256=None,
        total_count=0,
        decision="allowed",
        attestation_sha256=digest,
        attested_at_utc=attested_text,
        expires_at_utc=expires_text,
    )


def _parse_utc_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("timestamp must be ISO-8601") from error
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return parsed.astimezone(timezone.utc)


def _completion_response(
    transport_response: TransportResponse,
    route_lock: RouteLock,
) -> RawHttpResponse:
    """Parse only fields present in the completion before caching exact bytes."""
    if type(transport_response) is not TransportResponse:
        raise TypeError("transport must return a TransportResponse")
    record = _response_record(transport_response.body)
    generation_id = _generation_id_tolerant(record, transport_response.headers)
    returned_model = _nonempty_string(record.get("model"))
    selected_provider: str | None = None
    metadata_is_invalid = False
    try:
        selected_provider = _strict_selected_provider(
            record,
            require_selected=200 <= transport_response.status_code < 300,
        )
        selected_model = _strict_selected_model(
            record,
            require_selected=200 <= transport_response.status_code < 300,
        )
        if selected_model is not None:
            returned_model = selected_model
    except OpenRouterContractError:
        metadata_is_invalid = True
    usage_value = record.get("usage")
    usage = _usage_if_complete(usage_value) if type(usage_value) is dict else None
    billed_cost = (
        _body_cost(usage_value)
        if _completion_cost_is_trusted(record)
        else None
    )
    provenance = (
        ResponseProvenance(
            generation_id=generation_id,
            requested_model=route_lock.requested_model,
            returned_model=returned_model,
            returned_provider=selected_provider,
        )
        if all(
            value is not None
            for value in (generation_id, returned_model, selected_provider)
        )
        else None
    )
    return RawHttpResponse(
        status_code=transport_response.status_code,
        headers=transport_response.headers,
        body=transport_response.body,
        provenance=provenance,
        usage=usage,
        billed_cost_usd=billed_cost,
        generation_stats_attempts=(),
    )


def _derive_response(
    response: RawHttpResponse,
    route_lock: RouteLock,
) -> RawHttpResponse:
    """Derive final parsed fields from completion plus append-only stats attempts."""
    record = _response_record(response.body)
    generation_id = _generation_id_tolerant(record, response.headers)
    returned_model = _nonempty_string(record.get("model"))
    selected_provider: str | None = None
    try:
        selected_provider = _strict_selected_provider(
            record,
            require_selected=200 <= response.status_code < 300,
        )
        selected_model = _strict_selected_model(
            record,
            require_selected=200 <= response.status_code < 300,
        )
        if selected_model is not None:
            returned_model = selected_model
    except OpenRouterCostPolicyError:
        raise
    except OpenRouterContractError:
        pass
    usage_value = record.get("usage")
    usage = _usage_if_complete(usage_value) if type(usage_value) is dict else None
    billed_cost = (
        _body_cost(usage_value)
        if _completion_cost_is_trusted(record)
        else None
    )
    stats = _last_successful_stats(response.generation_stats_attempts)
    if stats is not None:
        if returned_model is None:
            returned_model = stats.returned_model
        if selected_provider is None:
            selected_provider = stats.returned_provider
        if billed_cost is None:
            billed_cost = stats.billed_cost_usd
        usage = _merge_usage_with_stats(usage, stats)
    provenance = (
        ResponseProvenance(
            generation_id=generation_id,
            requested_model=route_lock.requested_model,
            returned_model=returned_model,
            returned_provider=selected_provider,
        )
        if all(
            value is not None
            for value in (generation_id, returned_model, selected_provider)
        )
        else None
    )
    return replace(
        response,
        provenance=provenance,
        usage=usage,
        billed_cost_usd=billed_cost,
    )


def _response_needs_stats(response: RawHttpResponse) -> bool:
    record = _response_record(response.body)
    returned_model = _nonempty_string(record.get("model"))
    selected_provider = _strict_selected_provider(
        record,
        require_selected=200 <= response.status_code < 300,
    )
    selected_model = _strict_selected_model(
        record,
        require_selected=200 <= response.status_code < 300,
    )
    if selected_model is not None:
        returned_model = selected_model
    usage_value = record.get("usage")
    usage = _usage_if_complete(usage_value) if type(usage_value) is dict else None
    direct_cost = (
        _body_cost(usage_value)
        if _completion_cost_is_trusted(record)
        else None
    )
    stats = _last_successful_stats(response.generation_stats_attempts)
    returned_model = returned_model or (None if stats is None else stats.returned_model)
    selected_provider = selected_provider or (
        None if stats is None else stats.returned_provider
    )
    billed_cost = direct_cost or (None if stats is None else stats.billed_cost_usd)
    usage = usage if usage is not None else _merge_usage_with_stats(None, stats) if stats else None
    return (
        selected_provider is None
        or billed_cost is None
        or (200 <= response.status_code < 300 and (returned_model is None or usage is None))
    )


def _last_successful_stats(
    attempts: tuple[RawGenerationStatsAttempt, ...],
) -> _StatsFields | None:
    for attempt in reversed(attempts):
        if attempt.response is not None and 200 <= attempt.response.status_code < 300:
            return _strict_stats_fields(attempt.response)
    return None


def _validate_existing_stats_attempts(
    attempts: tuple[RawGenerationStatsAttempt, ...],
) -> None:
    for attempt in attempts:
        if attempt.transport_error_type == _AUTHENTICATION_SECRET_REFLECTION:
            raise _SecretReflectionError(
                "cached generation-stats attempt records an authentication-secret "
                "reflection; no retry is permitted"
            )
        if attempt.response is None:
            continue
        status = attempt.response.status_code
        if 200 <= status < 300:
            _strict_stats_fields(attempt.response)
        elif not _is_stats_retryable_status(status):
            raise OpenRouterContractError(
                f"generation-stats lookup returned terminal HTTP {status}"
            )


def _response_record(body: bytes) -> JsonObject:
    try:
        return require_json_object(
            strict_json_loads(body, label="OpenRouter response"),
            label="OpenRouter response",
        )
    except ValueError:
        return {}


def _nonempty_string(value: JsonValue) -> str | None:
    if type(value) is not str or not value.strip() or value != value.strip():
        return None
    return value


def _header_values(
    headers: tuple[tuple[str, str], ...],
    name: str,
) -> tuple[str, ...]:
    return tuple(
        value.strip()
        for header_name, value in headers
        if header_name.casefold() == name.casefold() and value.strip()
    )


def _generation_id_tolerant(
    record: JsonObject,
    headers: tuple[tuple[str, str], ...],
) -> str | None:
    body_id = _nonempty_string(record.get("id"))
    header_ids = _header_values(headers, "X-Generation-Id")
    if body_id is not None:
        return body_id
    if header_ids and len(set(header_ids)) == 1:
        return header_ids[0]
    return None


def _strict_generation_id(
    record: JsonObject,
    headers: tuple[tuple[str, str], ...],
) -> str | None:
    body_value = record.get("id")
    body_id = _nonempty_string(body_value)
    if body_value is not None and body_id is None:
        raise OpenRouterContractError("response generation id is malformed")
    header_ids = _header_values(headers, "X-Generation-Id")
    if len(set(header_ids)) > 1:
        raise OpenRouterContractError(
            "response contains conflicting X-Generation-Id headers"
        )
    header_id = header_ids[0] if header_ids else None
    if body_id is not None and header_id is not None and body_id != header_id:
        raise OpenRouterContractError(
            "response body and header generation ids differ"
        )
    return body_id if body_id is not None else header_id


def _strict_selected_provider(
    record: JsonObject,
    *,
    require_selected: bool,
) -> str | None:
    usage = record.get("usage")
    if type(usage) is dict:
        usage_is_byok = usage.get("is_byok")
        if usage_is_byok is True:
            raise OpenRouterCostPolicyError(
                "response usage reports BYOK; upstream billing is not bounded"
            )
        if usage_is_byok not in (None, False):
            raise OpenRouterContractError("usage.is_byok must be boolean")
    metadata = record.get("openrouter_metadata")
    if metadata is None:
        # Router-metadata cache hits officially omit the metadata object.
        # Their generation ID must recover provider/BYOK/cost through stats.
        return None
    if type(metadata) is not dict:
        raise OpenRouterContractError("openrouter_metadata must be an object")
    is_byok = metadata.get("is_byok")
    if is_byok is True:
        raise OpenRouterCostPolicyError(
            "response used BYOK; OpenRouter cost cannot bound upstream billing"
        )
    if require_selected and is_byok is not False:
        raise OpenRouterCostPolicyError(
            "successful response metadata must explicitly prove is_byok=false"
        )
    if is_byok not in (None, False):
        raise OpenRouterContractError("openrouter_metadata.is_byok must be boolean")
    endpoints = metadata.get("endpoints")
    if type(endpoints) is not dict:
        raise OpenRouterContractError(
            "openrouter_metadata.endpoints must be an object"
        )
    available = endpoints.get("available")
    if type(available) is not list:
        raise OpenRouterContractError(
            "openrouter_metadata.endpoints.available must be an array"
        )
    selected: list[JsonObject] = []
    for endpoint in available:
        if type(endpoint) is not dict:
            raise OpenRouterContractError(
                "openrouter_metadata available endpoint must be an object"
            )
        if type(endpoint.get("selected")) is not bool:
            raise OpenRouterContractError(
                "OpenRouter endpoint selected flag must be boolean"
            )
        if endpoint.get("selected") is True:
            selected.append(endpoint)
    if len(selected) > 1 or (require_selected and len(selected) != 1):
        raise OpenRouterContractError(
            "openrouter_metadata must identify exactly one selected endpoint "
            "for successes and at most one for failures"
        )
    if not selected:
        return None
    provider = _nonempty_string(selected[0].get("provider"))
    if provider is None:
        raise OpenRouterContractError(
            "selected OpenRouter endpoint lacks provider"
        )
    return provider


def _strict_selected_model(
    record: JsonObject,
    *,
    require_selected: bool,
) -> str | None:
    """Return an exact selected-endpoint model when router metadata supplies it."""
    # Reuse the complete provider/BYOK/selected-cardinality validation before
    # interpreting the additive endpoint model field.
    provider = _strict_selected_provider(
        record,
        require_selected=require_selected,
    )
    if provider is None:
        return None
    metadata = record["openrouter_metadata"]
    assert type(metadata) is dict
    endpoints = metadata["endpoints"]
    assert type(endpoints) is dict
    available = endpoints["available"]
    assert type(available) is list
    endpoint = next(
        item
        for item in available
        if type(item) is dict and item.get("selected") is True
    )
    model_value = endpoint.get("model")
    if model_value is None:
        return None
    model = _nonempty_string(model_value)
    if model is None:
        raise OpenRouterContractError(
            "selected OpenRouter endpoint model is malformed"
        )
    return model


def _completion_cost_is_trusted(record: JsonObject) -> bool:
    """Only accept direct OpenRouter cost after explicit non-BYOK evidence."""
    metadata = record.get("openrouter_metadata")
    if type(metadata) is not dict or metadata.get("is_byok") is not False:
        return False
    usage = record.get("usage")
    if type(usage) is not dict or usage.get("is_byok") is True:
        return False
    return True


def _usage_if_complete(value: JsonValue) -> TokenUsage | None:
    if type(value) is not dict:
        return None
    input_tokens = value.get("prompt_tokens")
    output_tokens = value.get("completion_tokens")
    total_tokens = value.get("total_tokens")
    if any(type(token_count) is not int for token_count in (
        input_tokens,
        output_tokens,
        total_tokens,
    )):
        return None
    prompt_details = value.get("prompt_tokens_details")
    cached_tokens: JsonValue = 0
    if type(prompt_details) is dict:
        cached_tokens = prompt_details.get("cached_tokens", 0)
    if type(cached_tokens) is not int:
        return None
    try:
        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cached_input_tokens=cached_tokens,
        )
    except GenerationContractError:
        return None


def _body_cost(usage_value: JsonValue) -> str | None:
    if type(usage_value) is not dict:
        return None
    return _cost_string(usage_value.get("cost"))


def _cost_string(value: JsonValue) -> str | None:
    if value is None:
        return None
    if type(value) not in (int, float, str) or type(value) is bool:
        return None
    try:
        cost = Decimal(str(value))
    except InvalidOperation:
        return None
    if not cost.is_finite() or cost < 0:
        return None
    return format(cost, "f")


def _stats_fields_tolerant(
    response: RawGenerationStatsResponse,
) -> _StatsFields:
    if not 200 <= response.status_code < 300:
        return _StatsFields(None, None, None, None, None, None)
    try:
        outer = require_json_object(
            strict_json_loads(response.body, label="OpenRouter generation stats"),
            label="OpenRouter generation stats",
        )
    except ValueError:
        return _StatsFields(None, None, None, None, None, None)
    data = outer.get("data")
    if type(data) is not dict:
        return _StatsFields(None, None, None, None, None, None)
    is_byok = data.get("is_byok")
    upstream_value = data.get("upstream_inference_cost")
    upstream_cost = _cost_string(upstream_value)
    cost_is_trusted = is_byok is False
    return _StatsFields(
        generation_id=_nonempty_string(data.get("id")),
        returned_model=_nonempty_string(data.get("model")),
        returned_provider=_nonempty_string(data.get("provider_name")),
        billed_cost_usd=(
            _cost_string(data.get("total_cost")) if cost_is_trusted else None
        ),
        input_tokens=_nonnegative_integer(data.get("tokens_prompt")),
        output_tokens=_nonnegative_integer(data.get("tokens_completion")),
        is_byok=(is_byok if type(is_byok) is bool else None),
        upstream_inference_cost_usd=upstream_cost,
    )


def _strict_stats_fields(
    response: RawGenerationStatsResponse,
) -> _StatsFields:
    if not 200 <= response.status_code < 300:
        raise OpenRouterContractError(
            "generation-stats lookup returned a non-success HTTP status"
        )
    try:
        outer = require_json_object(
            strict_json_loads(response.body, label="OpenRouter generation stats"),
            label="OpenRouter generation stats",
        )
    except ValueError as error:
        raise OpenRouterContractError(
            "generation-stats response is not a JSON object"
        ) from error
    data = outer.get("data")
    if type(data) is not dict:
        raise OpenRouterContractError(
            "generation-stats response lacks a data object"
        )
    generation_id = _optional_stats_string(data, "id")
    returned_model = _optional_stats_string(data, "model")
    returned_provider = _optional_stats_string(data, "provider_name")
    billed_cost = _optional_stats_cost(data, "total_cost")
    input_tokens = _optional_stats_integer(data, "tokens_prompt")
    output_tokens = _optional_stats_integer(data, "tokens_completion")
    is_byok = data.get("is_byok")
    if type(is_byok) is not bool:
        raise OpenRouterCostPolicyError(
            "generation-stats must explicitly prove is_byok=false"
        )
    if is_byok:
        raise OpenRouterCostPolicyError(
            "generation-stats reports BYOK; upstream billing is not bounded"
        )
    upstream_cost = (
        None
        if data.get("upstream_inference_cost") is None
        else _optional_stats_cost(data, "upstream_inference_cost")
    )
    return _StatsFields(
        generation_id=generation_id,
        returned_model=returned_model,
        returned_provider=returned_provider,
        billed_cost_usd=billed_cost,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        is_byok=is_byok,
        upstream_inference_cost_usd=upstream_cost,
    )


def _optional_stats_string(record: JsonObject, field: str) -> str | None:
    if field not in record:
        return None
    value = _nonempty_string(record[field])
    if value is None:
        raise OpenRouterContractError(
            f"generation-stats {field} must be a trimmed nonempty string"
        )
    return value


def _optional_stats_cost(record: JsonObject, field: str) -> str | None:
    if field not in record:
        return None
    value = _cost_string(record[field])
    if value is None:
        raise OpenRouterContractError(
            f"generation-stats {field} must be a nonnegative finite number"
        )
    return value


def _optional_stats_integer(record: JsonObject, field: str) -> int | None:
    if field not in record:
        return None
    value = _nonnegative_integer(record[field])
    if value is None:
        raise OpenRouterContractError(
            f"generation-stats {field} must be a nonnegative integer"
        )
    return value


def _nonnegative_integer(value: JsonValue) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _merge_usage_with_stats(
    usage: TokenUsage | None,
    stats: _StatsFields,
) -> TokenUsage | None:
    if usage is not None:
        return usage
    if stats.input_tokens is None or stats.output_tokens is None:
        return None
    return TokenUsage(
        input_tokens=stats.input_tokens,
        output_tokens=stats.output_tokens,
        total_tokens=stats.input_tokens + stats.output_tokens,
        cached_input_tokens=0,
    )


def _validate_observation(
    response: RawHttpResponse,
    route_lock: RouteLock,
) -> None:
    record = _response_record(response.body)
    generation_id = _strict_generation_id(record, response.headers)
    body_model_value = record.get("model")
    body_model = _nonempty_string(body_model_value)
    if body_model_value is not None and body_model is None:
        raise OpenRouterContractError("response model is malformed")
    body_provider = _strict_selected_provider(
        record,
        require_selected=200 <= response.status_code < 300,
    )
    endpoint_model = _strict_selected_model(
        record,
        require_selected=200 <= response.status_code < 300,
    )
    usage_value = record.get("usage")
    body_usage = _usage_if_complete(usage_value)
    body_cost = (
        _body_cost(usage_value)
        if _completion_cost_is_trusted(record)
        else None
    )
    successful_stats: list[_StatsFields] = []
    for attempt in response.generation_stats_attempts:
        if attempt.response is None:
            continue
        if not 200 <= attempt.response.status_code < 300:
            if not _is_stats_retryable_status(attempt.response.status_code):
                raise OpenRouterContractError(
                    "generation-stats history contains a terminal HTTP status"
                )
            continue
        fields = _strict_stats_fields(attempt.response)
        if attempt.billed_cost_usd != fields.billed_cost_usd:
            raise OpenRouterContractError(
                "cached generation-stats billed cost differs from exact body"
            )
        successful_stats.append(fields)
    stats = (
        successful_stats[-1]
        if successful_stats
        else _StatsFields(None, None, None, None, None, None)
    )
    for observed_stats in successful_stats:
        if (
            observed_stats.generation_id is not None
            and generation_id is not None
            and observed_stats.generation_id != generation_id
        ):
            raise OpenRouterContractError(
                "completion and generation-stats ids differ"
            )
        if (
            observed_stats.returned_model is not None
            and body_model is not None
            and observed_stats.returned_model != body_model
        ):
            raise OpenRouterContractError(
                "completion and generation-stats models differ"
            )
        if (
            observed_stats.returned_provider is not None
            and body_provider is not None
            and observed_stats.returned_provider != body_provider
        ):
            raise OpenRouterContractError(
                "completion and generation-stats providers differ"
            )
        if (
            observed_stats.billed_cost_usd is not None
            and body_cost is not None
            and Decimal(observed_stats.billed_cost_usd) != Decimal(body_cost)
        ):
            raise OpenRouterContractError(
                "completion and generation-stats billed costs differ"
            )

    if endpoint_model is not None and endpoint_model != route_lock.canonical_model:
        raise OpenRouterContractError(
            "selected endpoint model differs from the locked canonical model"
        )
    if body_model is not None and body_model not in {
        route_lock.requested_model,
        route_lock.canonical_model,
    }:
        raise OpenRouterContractError(
            "observed model differs from both locked model identities"
        )
    returned_model = endpoint_model or body_model or stats.returned_model
    returned_provider = body_provider or stats.returned_provider
    billed_cost = body_cost or stats.billed_cost_usd
    if returned_model is not None and returned_model != route_lock.canonical_model:
        raise OpenRouterContractError(
            "observed model differs from the locked canonical model"
        )
    if (
        returned_provider is not None
        and returned_provider != route_lock.returned_provider
    ):
        raise OpenRouterContractError(
            "observed provider differs from the locked provider"
        )

    if response.generation_stats_attempts:
        if body_provider is None and stats.returned_provider is None:
            raise OpenRouterContractError(
                "generation-stats lookup did not recover provider identity"
            )
        if body_cost is None and stats.billed_cost_usd is None:
            raise OpenRouterContractError(
                "generation-stats lookup did not recover billed cost"
            )

    expected_usage = _merge_usage_with_stats(body_usage, stats)
    if response.provenance is not None:
        expected = (
            generation_id,
            route_lock.requested_model,
            returned_model,
            returned_provider,
        )
        observed = (
            response.provenance.generation_id,
            response.provenance.requested_model,
            response.provenance.returned_model,
            response.provenance.returned_provider,
        )
        if any(value is None for value in expected) or observed != expected:
            raise OpenRouterContractError(
                "parsed provenance differs from exact response observations"
            )
    if response.usage != expected_usage:
        raise OpenRouterContractError(
            "parsed token usage differs from exact response observations"
        )
    if (
        response.billed_cost_usd is not None
        and (
            billed_cost is None
            or Decimal(response.billed_cost_usd) != Decimal(billed_cost)
        )
    ):
        raise OpenRouterContractError(
            "parsed billed cost differs from exact response observations"
        )


def _validate_success(response: RawHttpResponse, route_lock: RouteLock) -> None:
    _validate_observation(response, route_lock)
    if response.provenance is None:
        raise OpenRouterContractError(
            "successful response lacks model/provider/generation provenance"
        )
    if response.usage is None:
        raise OpenRouterContractError(
            "successful response lacks complete token usage"
        )
    if response.billed_cost_usd is None:
        raise OpenRouterContractError(
            "successful response lacks provider-billed cost"
        )


def _is_retryable_status(status_code: int) -> bool:
    return status_code == 429 or 500 <= status_code <= 599


def _is_stats_retryable_status(status_code: int) -> bool:
    # Generation metadata can briefly be unavailable immediately after the
    # completion returns, so 404 joins ordinary 429/5xx retryability here only.
    return status_code in (404, 429) or 500 <= status_code <= 599

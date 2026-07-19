"""OpenRouter catalog snapshots and exact provider-route locks."""

from __future__ import annotations

from hashlib import sha256

from apm.data.text.tinyworlds_v2.generation_schema import (
    CanonicalRequest,
    CatalogRoute,
    GenerationContractError,
    RouteLock,
)
from apm.data.text.tinyworlds_v2.json_contracts import JsonObject


def catalog_snapshot_sha256(*payloads: bytes) -> str:
    """Hash one or more exact catalog responses with unambiguous framing."""
    if not payloads or any(type(payload) is not bytes for payload in payloads):
        raise TypeError("catalog snapshot requires one or more byte payloads")
    digest = sha256()
    digest.update(b"apm.tinyworlds-v2.openrouter-catalog\0")
    for payload in payloads:
        digest.update(len(payload).to_bytes(8, byteorder="big", signed=False))
        digest.update(payload)
    return digest.hexdigest()


def lock_catalog_route(route_id: str, observed: CatalogRoute) -> RouteLock:
    """Create a persistent route lock from one normalized catalog observation."""
    if type(observed) is not CatalogRoute:
        raise TypeError("observed must be a CatalogRoute")
    return RouteLock(route_id=route_id, **observed.as_record())


def validate_route_lock(lock: RouteLock, observed: CatalogRoute) -> None:
    """Reject any model, endpoint, price, or catalog drift from a route lock."""
    if type(lock) is not RouteLock:
        raise TypeError("lock must be a RouteLock")
    if type(observed) is not CatalogRoute:
        raise TypeError("observed must be a CatalogRoute")
    differences = tuple(
        field
        for field in (
            "catalog_sha256",
            "requested_model",
            "canonical_model",
            "provider_slug",
            "returned_provider",
            "quantization",
            "input_usd_per_million",
            "output_usd_per_million",
        )
        if getattr(lock, field) != getattr(observed, field)
    )
    if differences:
        raise GenerationContractError(
            f"OpenRouter route drift for {lock.route_id!r}: {differences}"
        )


def validate_route_semantics(lock: RouteLock, observed: RouteLock) -> None:
    """Reject billable route drift while ignoring volatile catalog bytes."""
    if type(lock) is not RouteLock or type(observed) is not RouteLock:
        raise TypeError("route semantic validation requires two RouteLock values")
    differences = tuple(
        field
        for field in (
            "route_id",
            "requested_model",
            "canonical_model",
            "provider_slug",
            "returned_provider",
            "quantization",
            "input_usd_per_million",
            "output_usd_per_million",
        )
        if getattr(lock, field) != getattr(observed, field)
    )
    if differences:
        raise GenerationContractError(
            f"OpenRouter semantic route drift for {lock.route_id!r}: {differences}"
        )


def validate_locked_request_body(
    lock: RouteLock,
    request: CanonicalRequest,
) -> None:
    """Require a request to select only its pinned model/provider endpoint."""
    if type(lock) is not RouteLock:
        raise TypeError("lock must be a RouteLock")
    if type(request) is not CanonicalRequest:
        raise TypeError("request must be a CanonicalRequest")
    if request.route_lock_sha256 != lock.lock_sha256:
        raise GenerationContractError("request references a different route lock")
    body = request.body
    if body.get("model") != lock.requested_model:
        raise GenerationContractError(
            "request model does not match the locked OpenRouter model"
        )
    if "models" in body or "route" in body:
        raise GenerationContractError(
            "automatic or fallback model routing is forbidden"
        )
    provider = body.get("provider")
    if type(provider) is not dict:
        raise GenerationContractError("request requires a provider routing object")
    _validate_provider_preferences(lock, provider)
    if body.get("plugins") != []:
        raise GenerationContractError(
            "request must explicitly disable all OpenRouter plugins"
        )
    if body.get("transforms") != []:
        raise GenerationContractError(
            "request must explicitly disable all OpenRouter transforms"
        )


def _validate_provider_preferences(lock: RouteLock, provider: JsonObject) -> None:
    if provider.get("only") != [lock.provider_slug]:
        raise GenerationContractError(
            "provider.only must contain exactly the locked provider"
        )
    if provider.get("quantizations") != [lock.quantization]:
        raise GenerationContractError(
            "provider.quantizations must contain exactly the locked quantization"
        )
    if provider.get("allow_fallbacks") is not False:
        raise GenerationContractError("provider fallbacks must be disabled")
    if provider.get("require_parameters") is not True:
        raise GenerationContractError(
            "provider must require support for every request parameter"
        )
    if provider.get("order") is not None:
        raise GenerationContractError(
            "provider.order is forbidden when an exact provider is locked"
        )

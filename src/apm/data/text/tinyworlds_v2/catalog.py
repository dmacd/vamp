"""Strict OpenRouter catalog normalization and deterministic route selection."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from apm.data.text.tinyworlds_v2.bakeoff import (
    CANDIDATE_MODELS,
    VERIFIER_MODEL,
    CandidateModelSpec,
)
from apm.data.text.tinyworlds_v2.generation_schema import CatalogRoute, RouteLock
from apm.data.text.tinyworlds_v2.json_contracts import (
    JsonObject,
    JsonValue,
    require_json_object,
    strict_json_loads,
)
from apm.data.text.tinyworlds_v2.route_lock import (
    catalog_snapshot_sha256,
    lock_catalog_route,
)


_BASE_REQUIRED_GENERATION_PARAMETERS = frozenset(
    {"response_format", "seed", "structured_outputs"}
)
_FORBIDDEN_QUANTIZATIONS = frozenset(
    {"int4", "fp4", "nf4", "q4", "q4_k_m", "4-bit", "4bit"}
)
_UNREQUESTED_SERVICE_TIER_SUFFIXES = ("/flex", "/priority")
_QUANTIZATION_PRIORITY = {
    "fp32": 0,
    "fp16": 0,
    "bf16": 0,
    "unknown": 1,
    "fp8": 2,
    "int8": 3,
    "fp6": 4,
}
PHASE1_PROMPT_TOKEN_UPPER_BOUND = 16_384


class CatalogContractError(ValueError):
    """A live catalog cannot satisfy the fixed Phase 1 route contract."""


@dataclass(frozen=True, slots=True)
class CatalogPayloads:
    """Exact model catalog plus one endpoint response per planned model."""

    models: bytes
    endpoints: tuple[tuple[str, bytes], ...]

    def __post_init__(self) -> None:
        if type(self.models) is not bytes or not self.models:
            raise ValueError("models catalog payload must be nonempty bytes")
        expected_ids = tuple(
            model.request_model_id for model in (*CANDIDATE_MODELS, VERIFIER_MODEL)
        )
        if tuple(model_id for model_id, _ in self.endpoints) != expected_ids:
            raise ValueError("endpoint payloads must follow the fixed model-table order")
        if any(type(payload) is not bytes or not payload for _, payload in self.endpoints):
            raise ValueError("endpoint catalog payloads must be nonempty bytes")

    @property
    def snapshot_sha256(self) -> str:
        """Hash all exact responses in their prescribed unambiguous order."""
        return catalog_snapshot_sha256(
            self.models,
            *(payload for _, payload in self.endpoints),
        )


@dataclass(frozen=True, slots=True)
class ResolvedRouteCatalog:
    """One exact catalog snapshot and its table-ordered provider locks."""

    snapshot_sha256: str
    generator_routes: tuple[RouteLock, ...]
    verifier_route: RouteLock

    def __post_init__(self) -> None:
        if (
            type(self.snapshot_sha256) is not str
            or len(self.snapshot_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.snapshot_sha256)
        ):
            raise ValueError("catalog snapshot identity must be SHA-256")
        if tuple(route.route_id for route in self.generator_routes) != tuple(
            model.route_id for model in CANDIDATE_MODELS
        ):
            raise ValueError("resolved generator routes changed table order")
        if self.verifier_route.route_id != VERIFIER_MODEL.route_id:
            raise ValueError("resolved verifier route has the wrong identity")
        if any(
            route.catalog_sha256 != self.snapshot_sha256
            for route in (*self.generator_routes, self.verifier_route)
        ):
            raise ValueError("resolved routes do not share their catalog snapshot")


def resolve_openrouter_catalog(payloads: CatalogPayloads) -> ResolvedRouteCatalog:
    """Resolve every planned model to one healthy, exact, non-four-bit endpoint."""
    if type(payloads) is not CatalogPayloads:
        raise TypeError("payloads must be CatalogPayloads")
    model_records = _model_records(payloads.models)
    endpoint_payloads = dict(payloads.endpoints)
    specs = (*CANDIDATE_MODELS, VERIFIER_MODEL)
    locks = tuple(
        _resolve_route(
            spec,
            model_records,
            endpoint_payloads[spec.request_model_id],
            payloads.snapshot_sha256,
        )
        for spec in specs
    )
    return ResolvedRouteCatalog(
        snapshot_sha256=payloads.snapshot_sha256,
        generator_routes=locks[:-1],
        verifier_route=locks[-1],
    )


def _resolve_route(
    spec: CandidateModelSpec,
    models: dict[str, JsonObject],
    endpoint_payload: bytes,
    snapshot_sha256: str,
) -> RouteLock:
    model = models.get(spec.request_model_id)
    if model is None:
        raise CatalogContractError(
            f"OpenRouter catalog no longer contains {spec.request_model_id!r}"
        )
    canonical_slug = _string(model.get("canonical_slug"), "model canonical_slug")
    if canonical_slug != spec.canonical_slug:
        raise CatalogContractError(
            f"canonical model drift for {spec.request_model_id!r}: "
            f"expected {spec.canonical_slug!r}, got {canonical_slug!r}"
        )
    model_parameters = frozenset(
        _string_array(model.get("supported_parameters"), "model supported_parameters")
    )
    required_parameters = _BASE_REQUIRED_GENERATION_PARAMETERS | {
        spec.max_token_parameter
    }
    missing_model_parameters = required_parameters - model_parameters
    if missing_model_parameters:
        raise CatalogContractError(
            f"model {spec.request_model_id!r} lacks required parameters "
            f"{tuple(sorted(missing_model_parameters))}"
        )

    endpoint_root = require_json_object(
        strict_json_loads(endpoint_payload, label=f"{spec.route_id} endpoints"),
        label=f"{spec.route_id} endpoints",
    )
    data = require_json_object(
        endpoint_root.get("data"),
        label=f"{spec.route_id} endpoint data",
    )
    endpoint_model_id = _string(data.get("id"), "endpoint model ID")
    if endpoint_model_id != spec.request_model_id:
        raise CatalogContractError(
            f"endpoint catalog model mismatch for {spec.route_id!r}"
        )
    endpoint_values = data.get("endpoints")
    if type(endpoint_values) is not list:
        raise CatalogContractError("endpoint data must contain an endpoints array")
    eligible = tuple(
        endpoint
        for value in endpoint_values
        if (endpoint := _eligible_endpoint(value, required_parameters)) is not None
    )
    if not eligible:
        raise CatalogContractError(
            f"no healthy structured-output non-four-bit route for {spec.route_id!r}"
        )
    first_party = tuple(
        endpoint
        for endpoint in eligible
        if spec.first_party_provider_slug is not None
        and (
            endpoint["provider_slug"] == spec.first_party_provider_slug
            or endpoint["provider_slug"].startswith(
                spec.first_party_provider_slug + "/"
            )
        )
    )
    candidates = first_party or eligible
    selected = min(
        candidates,
        key=lambda endpoint: (
            _QUANTIZATION_PRIORITY.get(endpoint["quantization"], 99),
            _decimal_price(endpoint["input_price"], "endpoint input price"),
            _decimal_price(endpoint["completion_price"], "endpoint completion price"),
            endpoint["provider_slug"],
        ),
    )
    routing_matches = tuple(
        endpoint
        for endpoint in eligible
        if (
            endpoint["provider_slug"] == selected["provider_slug"]
            or endpoint["provider_slug"].startswith(
                selected["provider_slug"] + "/"
            )
        )
        and endpoint["quantization"] == selected["quantization"]
        and _decimal_price(endpoint["input_price"], "input price")
        <= _decimal_price(selected["input_price"], "selected input price")
        and _decimal_price(endpoint["completion_price"], "completion price")
        <= _decimal_price(
            selected["completion_price"], "selected completion price"
        )
    )
    if len(routing_matches) != 1:
        raise CatalogContractError(
            f"provider.only cannot identify one exact endpoint for {spec.route_id!r}"
        )
    observed = CatalogRoute(
        catalog_sha256=snapshot_sha256,
        requested_model=spec.request_model_id,
        canonical_model=spec.canonical_slug,
        provider_slug=selected["provider_slug"],
        returned_provider=selected["returned_provider"],
        quantization=selected["quantization"],
        input_usd_per_million=_per_million(selected["input_price"]),
        output_usd_per_million=_per_million(selected["completion_price"]),
    )
    return lock_catalog_route(spec.route_id, observed)


def _eligible_endpoint(
    value: JsonValue,
    required_parameters: frozenset[str],
) -> dict[str, str] | None:
    if type(value) is not dict:
        raise CatalogContractError("endpoint entries must be JSON objects")
    status = value.get("status")
    if type(status) is not int:
        raise CatalogContractError("endpoint status must be an integer")
    provider_slug = _string(value.get("tag"), "endpoint tag")
    returned_provider = _string(value.get("provider_name"), "endpoint provider_name")
    quantization = _string(value.get("quantization"), "endpoint quantization")
    parameters = frozenset(
        _string_array(value.get("supported_parameters"), "endpoint supported_parameters")
    )
    pricing = require_json_object(value.get("pricing"), label="endpoint pricing")
    prompt_price = _price_string(pricing.get("prompt"), "endpoint prompt price")
    completion_price = _price_string(
        pricing.get("completion"), "endpoint completion price"
    )
    request_price = _optional_price_string(
        pricing.get("request"), "endpoint request price"
    )
    if _decimal_price(request_price, "endpoint request price") != 0:
        return None
    reasoning_price = _optional_price_string(
        pricing.get("internal_reasoning"),
        "endpoint internal reasoning price",
    )
    if _decimal_price(
        reasoning_price,
        "endpoint internal reasoning price",
    ) > _decimal_price(completion_price, "endpoint completion price"):
        return None
    cache_write_price = _optional_price_string(
        pricing.get("input_cache_write"),
        "endpoint input cache-write price",
    )
    cache_write_1h_price = _optional_price_string(
        pricing.get("input_cache_write_1h"),
        "endpoint one-hour input cache-write price",
    )
    input_price = format(
        _decimal_price(prompt_price, "endpoint prompt price")
        + max(
            _decimal_price(cache_write_price, "input cache-write price"),
            _decimal_price(cache_write_1h_price, "one-hour cache-write price"),
        ),
        "f",
    )
    if _has_unsafe_applicable_override(
        pricing.get("overrides"),
        prompt_price=prompt_price,
        completion_price=completion_price,
        cache_write_price=cache_write_price,
        cache_write_1h_price=cache_write_1h_price,
    ):
        return None
    if (
        status != 0
        or provider_slug.endswith(_UNREQUESTED_SERVICE_TIER_SUFFIXES)
        or quantization in _FORBIDDEN_QUANTIZATIONS
        or not required_parameters.issubset(parameters)
    ):
        return None
    return {
        "completion_price": completion_price,
        "input_price": input_price,
        "prompt_price": prompt_price,
        "quantization": quantization,
        "provider_slug": provider_slug,
        "returned_provider": returned_provider,
    }


def _model_records(payload: bytes) -> dict[str, JsonObject]:
    root = require_json_object(
        strict_json_loads(payload, label="OpenRouter models catalog"),
        label="OpenRouter models catalog",
    )
    values = root.get("data")
    if type(values) is not list:
        raise CatalogContractError("models catalog data must be an array")
    records: dict[str, JsonObject] = {}
    for value in values:
        record = require_json_object(value, label="model catalog record")
        model_id = _string(record.get("id"), "model ID")
        if model_id in records:
            raise CatalogContractError(f"duplicate model catalog ID {model_id!r}")
        records[model_id] = record
    return records


def _string(value: JsonValue | None, label: str) -> str:
    if type(value) is not str or not value:
        raise CatalogContractError(f"{label} must be a nonempty string")
    return value


def _string_array(value: JsonValue | None, label: str) -> tuple[str, ...]:
    if type(value) is not list or any(type(item) is not str or not item for item in value):
        raise CatalogContractError(f"{label} must be a string array")
    return tuple(value)


def _price_string(value: JsonValue | None, label: str) -> str:
    if type(value) not in (str, int, float) or type(value) is bool:
        raise CatalogContractError(f"{label} must be decimal-compatible")
    text = str(value)
    _decimal_price(text, label)
    return text


def _optional_price_string(value: JsonValue | None, label: str) -> str:
    return "0" if value is None else _price_string(value, label)


def _has_unsafe_applicable_override(
    value: JsonValue | None,
    *,
    prompt_price: str,
    completion_price: str,
    cache_write_price: str,
    cache_write_1h_price: str,
) -> bool:
    if value is None:
        return False
    if type(value) is not list:
        raise CatalogContractError("endpoint pricing overrides must be an array")
    base_input = _decimal_price(prompt_price, "base prompt price") + max(
        _decimal_price(cache_write_price, "base cache-write price"),
        _decimal_price(cache_write_1h_price, "base one-hour cache-write price"),
    )
    base_completion = _decimal_price(completion_price, "base completion price")
    for index, item in enumerate(value):
        override = require_json_object(
            item,
            label=f"endpoint pricing override {index}",
        )
        minimum = override.get("min_prompt_tokens")
        if minimum is not None:
            if type(minimum) not in (int, float) or type(minimum) is bool:
                raise CatalogContractError(
                    "pricing override min_prompt_tokens must be numeric"
                )
            threshold = Decimal(str(minimum))
            if not threshold.is_finite() or threshold < 0:
                raise CatalogContractError(
                    "pricing override min_prompt_tokens must be nonnegative"
                )
            # The documented condition is strict prompt_tokens > threshold.
            if threshold >= PHASE1_PROMPT_TOKEN_UPPER_BOUND:
                continue
        override_prompt = _optional_override_price(
            override,
            "prompt",
            prompt_price,
            index,
        )
        override_completion = _optional_override_price(
            override,
            "completion",
            completion_price,
            index,
        )
        override_write = _optional_override_price(
            override,
            "input_cache_write",
            cache_write_price,
            index,
        )
        override_write_1h = _optional_override_price(
            override,
            "input_cache_write_1h",
            cache_write_1h_price,
            index,
        )
        override_input = _decimal_price(override_prompt, "override prompt") + max(
            _decimal_price(override_write, "override cache-write"),
            _decimal_price(override_write_1h, "override one-hour cache-write"),
        )
        if (
            override_input > base_input
            or _decimal_price(override_completion, "override completion")
            > base_completion
        ):
            return True
    return False


def _optional_override_price(
    override: JsonObject,
    field: str,
    inherited: str,
    index: int,
) -> str:
    if field not in override:
        return inherited
    return _price_string(
        override[field],
        f"endpoint pricing override {index} {field}",
    )


def _decimal_price(value: str, label: str) -> Decimal:
    try:
        price = Decimal(value)
    except InvalidOperation as error:
        raise CatalogContractError(f"{label} must be decimal-compatible") from error
    if not price.is_finite() or price < 0:
        raise CatalogContractError(f"{label} must be finite and nonnegative")
    return price


def _per_million(per_token: str) -> str:
    value = _decimal_price(per_token, "per-token price") * Decimal(1_000_000)
    return format(value.normalize(), "f")


__all__ = [
    "CatalogContractError",
    "CatalogPayloads",
    "PHASE1_PROMPT_TOKEN_UPPER_BOUND",
    "ResolvedRouteCatalog",
    "resolve_openrouter_catalog",
]

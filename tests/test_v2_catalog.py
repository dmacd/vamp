from __future__ import annotations

import json

import pytest

from apm.data.text.tinyworlds_v2.bakeoff import (
    CANDIDATE_MODELS,
    TWO_ROUTE_AUTHOR_MODELS,
    VERIFIER_MODEL,
)
from apm.data.text.tinyworlds_v2.catalog import (
    CatalogContractError,
    CatalogPayloads,
    PHASE1_PROMPT_TOKEN_UPPER_BOUND,
    resolve_openrouter_catalog,
    resolve_openrouter_routes,
)


_PARAMETERS = [
    "max_completion_tokens",
    "max_tokens",
    "response_format",
    "seed",
    "structured_outputs",
]


def _payloads(*, drift: bool = False) -> CatalogPayloads:
    specs = (*CANDIDATE_MODELS, VERIFIER_MODEL)
    models = {
        "data": [
            {
                "canonical_slug": (
                    "changed/model" if drift and index == 0 else spec.canonical_slug
                ),
                "id": spec.request_model_id,
                "supported_parameters": _PARAMETERS,
            }
            for index, spec in enumerate(specs)
        ]
    }
    endpoints = []
    for index, spec in enumerate(specs):
        provider = spec.first_party_provider_slug or f"provider-{index}"
        records = [
            {
                "pricing": {"completion": "0.0000001", "prompt": "0.00000005"},
                "provider_name": "Four Bit",
                "quantization": "fp4",
                "status": 0,
                "supported_parameters": _PARAMETERS,
                "tag": "cheap-four-bit",
            },
            {
                "pricing": {"completion": "0.0000003", "prompt": "0.0000002"},
                "provider_name": f"Primary {index}",
                "quantization": "unknown",
                "status": 0,
                "supported_parameters": _PARAMETERS,
                "tag": provider,
            },
            {
                "pricing": {"completion": "0.0000002", "prompt": "0.0000001"},
                "provider_name": f"Secondary {index}",
                "quantization": "bf16",
                "status": 0,
                "supported_parameters": _PARAMETERS,
                "tag": f"secondary-{index}",
            },
        ]
        endpoints.append(
            (
                spec.request_model_id,
                json.dumps(
                    {"data": {"endpoints": records, "id": spec.request_model_id}}
                ).encode(),
            )
        )
    return CatalogPayloads(json.dumps(models).encode(), tuple(endpoints))


def _mutate_eligible_pricing(
    payloads: CatalogPayloads,
    route_index: int,
    **fields: object,
) -> CatalogPayloads:
    endpoints = list(payloads.endpoints)
    model_id, raw = endpoints[route_index]
    record = json.loads(raw)
    for endpoint in record["data"]["endpoints"]:
        if endpoint["quantization"] != "fp4":
            endpoint["pricing"].update(fields)
    endpoints[route_index] = (model_id, json.dumps(record).encode())
    return CatalogPayloads(payloads.models, tuple(endpoints))


def test_catalog_resolver_pins_canonical_models_and_provider_policy() -> None:
    resolved = resolve_openrouter_catalog(_payloads())

    assert [route.route_id for route in resolved.generator_routes] == [
        model.route_id for model in CANDIDATE_MODELS
    ]
    assert resolved.generator_routes[0].provider_slug == "secondary-0"
    assert resolved.generator_routes[0].returned_provider == "Secondary 0"
    assert resolved.generator_routes[0].quantization == "bf16"
    assert resolved.generator_routes[1].provider_slug == "google-vertex/global"
    assert resolved.generator_routes[1].quantization == "unknown"
    assert resolved.generator_routes[1].input_usd_per_million == "0.2"
    assert resolved.generator_routes[1].output_usd_per_million == "0.3"
    assert resolved.verifier_route.canonical_model == VERIFIER_MODEL.canonical_slug


def test_catalog_resolver_fails_closed_on_canonical_model_drift() -> None:
    with pytest.raises(CatalogContractError, match="canonical model drift"):
        resolve_openrouter_catalog(_payloads(drift=True))


def test_catalog_resolver_accepts_an_explicit_two_author_plan() -> None:
    historical = _payloads()
    endpoint_by_model = dict(historical.endpoints)
    model_ids = tuple(model.request_model_id for model in TWO_ROUTE_AUTHOR_MODELS)
    payloads = CatalogPayloads(
        historical.models,
        tuple((model_id, endpoint_by_model[model_id]) for model_id in model_ids),
        model_plan_ids=model_ids,
    )

    routes = resolve_openrouter_routes(payloads, TWO_ROUTE_AUTHOR_MODELS)

    assert tuple(route.route_id for route in routes) == tuple(
        model.route_id for model in TWO_ROUTE_AUTHOR_MODELS
    )


def test_catalog_payload_order_is_fixed() -> None:
    payloads = _payloads()
    with pytest.raises(ValueError, match="fixed model-table order"):
        CatalogPayloads(payloads.models, tuple(reversed(payloads.endpoints)))


@pytest.mark.parametrize(
    "pricing",
    (
        {"request": "0.01"},
        {"internal_reasoning": "0.000001"},
        {
            "overrides": [
                {
                    "min_prompt_tokens": PHASE1_PROMPT_TOKEN_UPPER_BOUND - 1,
                    "prompt": "0.000001",
                }
            ]
        },
        {"overrides": [{"utc_start": 0, "prompt": "0.000001"}]},
    ),
)
def test_catalog_rejects_unbounded_billable_endpoint_pricing(
    pricing: dict[str, object],
) -> None:
    payloads = _mutate_eligible_pricing(_payloads(), 0, **pricing)

    with pytest.raises(CatalogContractError, match="no healthy"):
        resolve_openrouter_catalog(payloads)


@pytest.mark.parametrize(
    "override",
    (
        {
            "min_prompt_tokens": PHASE1_PROMPT_TOKEN_UPPER_BOUND,
            "prompt": "0.000001",
        },
        {"prompt": "0.00000001", "completion": "0.00000001"},
    ),
)
def test_catalog_accepts_inapplicable_or_lower_pricing_override(
    override: dict[str, object],
) -> None:
    payloads = _mutate_eligible_pricing(
        _payloads(),
        0,
        overrides=[override],
    )

    assert resolve_openrouter_catalog(payloads).generator_routes[0]


def test_catalog_input_ceiling_adds_cache_write_surcharge() -> None:
    payloads = _mutate_eligible_pricing(
        _payloads(),
        0,
        input_cache_write="0.00000005",
    )

    route = resolve_openrouter_catalog(payloads).generator_routes[0]

    assert route.input_usd_per_million == "0.15"


def test_catalog_rejects_ambiguous_provider_base_slug() -> None:
    payloads = _payloads()
    endpoints = list(payloads.endpoints)
    model_id, raw = endpoints[1]
    record = json.loads(raw)
    primary = record["data"]["endpoints"][1]
    duplicate = dict(primary)
    duplicate["tag"] = primary["tag"] + "/turbo"
    record["data"]["endpoints"].append(duplicate)
    endpoints[1] = (model_id, json.dumps(record).encode())

    with pytest.raises(CatalogContractError, match="one exact endpoint"):
        resolve_openrouter_catalog(
            CatalogPayloads(payloads.models, tuple(endpoints))
        )


def test_catalog_can_pin_preferred_exact_first_party_suffix() -> None:
    payloads = _payloads()
    endpoints = list(payloads.endpoints)
    model_id, raw = endpoints[1]
    record = json.loads(raw)
    primary = record["data"]["endpoints"][1]
    suffix = json.loads(json.dumps(primary))
    suffix["tag"] = primary["tag"] + "/turbo"
    suffix["pricing"] = {
        "completion": "0.0000001",
        "prompt": "0.0000001",
    }
    record["data"]["endpoints"].append(suffix)
    endpoints[1] = (model_id, json.dumps(record).encode())

    route = resolve_openrouter_catalog(
        CatalogPayloads(payloads.models, tuple(endpoints))
    ).generator_routes[1]

    assert route.provider_slug.endswith("/turbo")


def test_unrequested_service_tiers_do_not_make_base_slug_ambiguous() -> None:
    payloads = _payloads()
    endpoints = list(payloads.endpoints)
    model_id, raw = endpoints[1]
    record = json.loads(raw)
    primary = record["data"]["endpoints"][1]
    for suffix in ("/flex", "/priority"):
        tier = json.loads(json.dumps(primary))
        tier["tag"] = primary["tag"] + suffix
        tier["pricing"] = {
            "completion": "0.00000001",
            "prompt": "0.00000001",
        }
        record["data"]["endpoints"].append(tier)
    endpoints[1] = (model_id, json.dumps(record).encode())

    route = resolve_openrouter_catalog(
        CatalogPayloads(payloads.models, tuple(endpoints))
    ).generator_routes[1]

    assert route.provider_slug == primary["tag"]

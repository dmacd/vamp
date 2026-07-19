from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
import subprocess
import sys
from threading import Barrier

import pytest

from apm.data.text.tinyworlds_v2.generation_cache import (
    GenerationCacheError,
    ImmutableRawCache,
)
from apm.data.text.tinyworlds_v2.bakeoff import _per_million_price
from apm.data.text.tinyworlds_v2.generation_costs import (
    CostCapExceeded,
    CostJournalRecoveryRequired,
    PaidRunLockUnavailable,
    RouteWorkload,
    RuntimeCostLedger,
    TokenWorkload,
    build_cost_preflight,
    enforce_cost_cap,
    exclusive_paid_run_lock,
    estimate_route_cost,
    request_cost_upper_bound,
)
from apm.data.text.tinyworlds_v2.generation_schema import (
    CanonicalRequest,
    CatalogRoute,
    GenerationContractError,
    RawAttempt,
    RawHttpResponse,
    ResponseProvenance,
    TokenUsage,
    OPENROUTER_TRANSPORT_PROTOCOL,
)
from apm.data.text.tinyworlds_v2.json_contracts import (
    CanonicalJsonError,
    canonical_json_bytes,
    json_sha256,
    strict_json_loads,
)
from apm.data.text.tinyworlds_v2.openrouter import (
    OpenRouterBillingUnknown,
    OpenRouterClient,
    OpenRouterContractError,
    OpenRouterCostPolicyError,
    OpenRouterResponseError,
    OpenRouterRetryExhausted,
    RetryPolicy,
    TransportError,
    TransportResponse,
)
from apm.data.text.tinyworlds_v2.route_lock import (
    catalog_snapshot_sha256,
    lock_catalog_route,
    validate_locked_request_body,
    validate_route_lock,
)


class ScriptedTransport:
    def __init__(
        self,
        outcomes: list[TransportResponse | Exception],
        *,
        stats_outcomes: list[TransportResponse | Exception] | None = None,
    ) -> None:
        self.outcomes = outcomes
        self.stats_outcomes = [] if stats_outcomes is None else stats_outcomes
        self.calls: list[
            tuple[str, tuple[tuple[str, str], ...], bytes, float]
        ] = []
        self.stats_calls: list[
            tuple[str, tuple[tuple[str, str], ...], float]
        ] = []

    def post(
        self,
        *,
        url: str,
        headers: tuple[tuple[str, str], ...],
        body: bytes,
        timeout_seconds: float,
    ) -> TransportResponse:
        self.calls.append((url, headers, body, timeout_seconds))
        outcome = self.outcomes[len(self.calls) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def get_authenticated(
        self,
        *,
        url: str,
        headers: tuple[tuple[str, str], ...],
        timeout_seconds: float,
    ) -> TransportResponse:
        self.stats_calls.append((url, headers, timeout_seconds))
        outcome = self.stats_outcomes[len(self.stats_calls) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _route():
    observed = CatalogRoute(
        catalog_sha256=catalog_snapshot_sha256(b'{"models":[]}', b'{"data":[]}'),
        requested_model="openai/gpt-5.4-mini",
        canonical_model="openai/gpt-5.4-mini-2026-03-17",
        provider_slug="openai",
        returned_provider="OpenAI",
        quantization="bf16",
        input_usd_per_million="0.75",
        output_usd_per_million="4.50",
    )
    return lock_catalog_route("gpt-5.4-mini", observed), observed


def _request(*, extra_body: dict[str, object] | None = None):
    route, _ = _route()
    body = {
        "max_completion_tokens": 16,
        "messages": [{"content": "Write a small story.", "role": "user"}],
        "model": route.requested_model,
        "plugins": [],
        "provider": {
            "allow_fallbacks": False,
            "max_price": {"completion": 4.5, "prompt": 0.75},
            "only": [route.provider_slug],
            "quantizations": [route.quantization],
            "require_parameters": True,
        },
        "temperature": 0.4,
        "transforms": [],
    }
    if extra_body:
        body.update(extra_body)
    return CanonicalRequest.from_body(
        route_lock_sha256=route.lock_sha256,
        endpoint="/api/v1/chat/completions",
        body=body,
    )


def _byok_authorization() -> dict[str, object]:
    return {
        "attestation_sha256": None,
        "attested_at_utc": None,
        "checked_at_utc": "2026-07-18T12:00:00Z",
        "decision": "allowed",
        "endpoint": "/api/v1/byok",
        "expires_at_utc": None,
        "method": "GET",
        "response_body_sha256": "a" * 64,
        "source": "management_api",
        "status_code": 200,
        "total_count": 0,
    }


def _authorize(ledger: RuntimeCostLedger) -> RuntimeCostLedger:
    ledger.authorize_byok(_byok_authorization())
    return ledger


def _response(
    status_code: int = 200,
    *,
    returned_model: str = "openai/gpt-5.4-mini-2026-03-17",
    provider: str = "OpenAI",
    exact_suffix: bytes = b"",
) -> TransportResponse:
    body = canonical_json_bytes(
        {
            "choices": [
                {"message": {"content": "Once there was a little fox."}}
            ],
            "id": "gen-123",
            "model": returned_model,
            "openrouter_metadata": {
                "additive_future_field": {"ignored": True},
                "is_byok": False,
                "endpoints": {
                    "available": [
                        {
                            "additive_endpoint_field": 7,
                            "provider": provider,
                            "selected": True,
                        }
                    ]
                },
            },
            "usage": {
                "completion_tokens": 7,
                "cost": 0.000021,
                "prompt_tokens": 5,
                "prompt_tokens_details": {"cached_tokens": 2},
                "total_tokens": 12,
            },
        }
    ) + exact_suffix
    return TransportResponse(
        status_code=status_code,
        headers=(("content-type", "application/json"), ("x-request-id", "req-1")),
        body=body,
    )


def _stats_response(
    *,
    generation_id: str = "gen-123",
    returned_model: str = "openai/gpt-5.4-mini-2026-03-17",
    provider: str = "OpenAI",
    cost: object = 0.000021,
    status_code: int = 200,
) -> TransportResponse:
    return TransportResponse(
        status_code=status_code,
        headers=(("content-type", "application/json"),),
        body=canonical_json_bytes(
            {
                "data": {
                    "additive_future_field": "ignored",
                    "id": generation_id,
                    "is_byok": False,
                    "model": returned_model,
                    "provider_name": provider,
                    "tokens_completion": 7,
                    "tokens_prompt": 5,
                    "total_cost": cost,
                    "upstream_inference_cost": None,
                }
            }
        ),
    )


def _response_requiring_stats(
    *,
    status_code: int = 200,
    generation_id: str = "gen-123",
    include_body_id: bool = True,
) -> TransportResponse:
    body = {
        "choices": [
            {"message": {"content": "Once there was a little fox."}}
        ],
        "model": "openai/gpt-5.4-mini-2026-03-17",
        "openrouter_metadata": {
            "endpoints": {
                "available": (
                    [{"provider": "OpenAI", "selected": True}]
                    if 200 <= status_code < 300
                    else []
                )
            },
            "is_byok": False,
        },
        "usage": {
            "completion_tokens": 7,
            "prompt_tokens": 5,
            "total_tokens": 12,
        },
    }
    if include_body_id:
        body["id"] = generation_id
    return TransportResponse(
        status_code=status_code,
        headers=(("X-Generation-Id", generation_id),),
        body=canonical_json_bytes(body),
    )


def _response_with_selected_providers(providers: tuple[str, ...]) -> TransportResponse:
    record = strict_json_loads(_response().body)
    record["openrouter_metadata"]["endpoints"]["available"] = [
        {"provider": provider, "selected": True} for provider in providers
    ]
    return TransportResponse(200, (), canonical_json_bytes(record))


def _client(
    tmp_path: Path,
    transport: ScriptedTransport,
    *,
    max_attempts: int = 3,
    stats_max_attempts: int = 4,
    delays: list[float] | None = None,
    ledger: RuntimeCostLedger | None = None,
) -> OpenRouterClient:
    observed_delays = [] if delays is None else delays
    client = OpenRouterClient(
        api_key="secret-openrouter-key",
        management_api_key="secret-management-key",
        transport=transport,
        cache=ImmutableRawCache(tmp_path / "raw"),
        retry_policy=RetryPolicy(
            max_attempts=max_attempts,
            initial_delay_seconds=0.25,
            maximum_delay_seconds=1.0,
        ),
        stats_retry_policy=RetryPolicy(
            max_attempts=stats_max_attempts,
            initial_delay_seconds=0.25,
            maximum_delay_seconds=1.0,
        ),
        cost_ledger=RuntimeCostLedger() if ledger is None else ledger,
        sleeper=observed_delays.append,
        clock=lambda: "2026-07-18T12:00:00Z",
    )
    client.cost_ledger.authorize_byok(_byok_authorization())
    return client


def test_canonical_request_is_order_independent_frozen_and_credential_free() -> None:
    route, _ = _route()
    first_body = {
        "model": route.requested_model,
        "provider": {
            "only": [route.provider_slug],
            "quantizations": [route.quantization],
            "allow_fallbacks": False,
            "require_parameters": True,
        },
        "plugins": [],
        "transforms": [],
        "messages": [{"role": "user", "content": "Hello"}],
    }
    second_body = dict(reversed(tuple(first_body.items())))
    requests = tuple(
        CanonicalRequest.from_body(
            route_lock_sha256=route.lock_sha256,
            endpoint="/api/v1/chat/completions",
            body=body,
        )
        for body in (first_body, second_body)
    )

    assert requests[0] == requests[1]
    old_protocol = CanonicalRequest.from_body(
        route_lock_sha256=route.lock_sha256,
        endpoint="/api/v1/chat/completions",
        body=first_body,
        transport_protocol="openrouter-chat-completions-v1",
    )
    assert requests[0].transport_protocol == OPENROUTER_TRANSPORT_PROTOCOL
    assert old_protocol.request_sha256 != requests[0].request_sha256
    first_body["model"] = "changed-after-freeze"
    assert requests[0].body["model"] == route.requested_model
    assert b"secret-openrouter-key" not in requests[0].body_bytes
    with pytest.raises(GenerationContractError, match="credentials"):
        CanonicalRequest.from_body(
            route_lock_sha256=route.lock_sha256,
            endpoint="/api/v1/chat/completions",
            body={"api_key": "must-not-be-hashed"},
        )


def test_strict_json_rejects_duplicate_fields_and_nonfinite_values() -> None:
    with pytest.raises(CanonicalJsonError, match="duplicate"):
        strict_json_loads(b'{"a":1,"a":2}')
    with pytest.raises(CanonicalJsonError, match="non-finite"):
        strict_json_loads(b'{"a":NaN}')


def test_route_lock_detects_catalog_drift_and_forbids_fallback_routing() -> None:
    route, observed = _route()
    validate_route_lock(route, observed)
    with pytest.raises(GenerationContractError, match="route drift"):
        validate_route_lock(
            route,
            replace(observed, output_usd_per_million="4.51"),
        )

    validate_locked_request_body(route, _request())
    with pytest.raises(GenerationContractError, match="fallback"):
        validate_locked_request_body(
            route,
            _request(extra_body={"models": ["some/fallback"]}),
        )
    with pytest.raises(GenerationContractError, match="plugins"):
        validate_locked_request_body(
            route,
            _request(extra_body={"plugins": [{"id": "response-healing"}]}),
        )


def test_route_lock_identity_excludes_only_volatile_catalog_bytes() -> None:
    route, observed = _route()
    refreshed_route = lock_catalog_route(
        route.route_id,
        replace(observed, catalog_sha256="b" * 64),
    )
    requests = tuple(
        CanonicalRequest.from_body(
            route_lock_sha256=locked_route.lock_sha256,
            endpoint="/api/v1/chat/completions",
            body=_request().body,
        )
        for locked_route in (route, refreshed_route)
    )

    assert route.as_record()["catalog_sha256"] != refreshed_route.as_record()[
        "catalog_sha256"
    ]
    assert route.lock_sha256 == refreshed_route.lock_sha256
    assert requests[0].request_sha256 == requests[1].request_sha256
    with pytest.raises(GenerationContractError, match="catalog_sha256"):
        validate_route_lock(route, replace(observed, catalog_sha256="b" * 64))


@pytest.mark.parametrize(
    ("field", "changed_value"),
    (
        ("route_id", "gpt-5.4-mini-alternate"),
        ("requested_model", "openai/gpt-5.4-mini-alternate"),
        ("canonical_model", "openai/gpt-5.4-mini-2026-03-18"),
        ("provider_slug", "openai-alternate"),
        ("returned_provider", "OpenAI Alternate"),
        ("quantization", "fp16"),
        ("input_usd_per_million", "0.76"),
        ("output_usd_per_million", "4.51"),
    ),
)
def test_every_route_semantic_field_contributes_to_lock_identity(
    field: str,
    changed_value: str,
) -> None:
    route, _ = _route()

    assert replace(route, **{field: changed_value}).lock_sha256 != route.lock_sha256


def test_catalog_snapshot_digest_is_framed_and_order_sensitive() -> None:
    assert catalog_snapshot_sha256(b"a", b"bc") != catalog_snapshot_sha256(
        b"ab", b"c"
    )
    assert catalog_snapshot_sha256(b"a", b"b") != catalog_snapshot_sha256(
        b"b", b"a"
    )
    with pytest.raises(GenerationContractError, match="4-bit"):
        replace(_route()[1], quantization="q4_k_m")


def test_cost_preflight_computes_retry_reserve_batch_comparison_and_cap() -> None:
    route, _ = _route()
    workload = RouteWorkload(
        route=route,
        workload=TokenWorkload(
            label="screen",
            request_count=100,
            input_tokens_per_request=100,
            output_tokens_per_request=200,
            retry_allowance_basis_points=2_000,
        ),
    )
    estimate = estimate_route_cost(workload)
    preflight = build_cost_preflight((workload,))

    assert estimate.expected_usd == "0.097500"
    assert estimate.conservative_usd == "0.117000"
    assert preflight.permitted
    assert tuple(
        item.model_snapshot for item in preflight.openai_batch_comparisons
    ) == ("gpt-5.4-mini-2026-03-17", "gpt-5.4-2026-03-05")
    assert preflight.as_record()["hard_cap_usd"] == "15.000000"
    enforce_cost_cap(preflight)

    blocked = build_cost_preflight((workload,), hard_cap_usd="0.10")
    assert not blocked.permitted
    with pytest.raises(CostCapExceeded, match="hard cap"):
        enforce_cost_cap(blocked)


def test_raw_cache_preserves_exact_response_and_rejects_overwrite_or_tampering(
    tmp_path: Path,
) -> None:
    request = _request()
    cache = ImmutableRawCache(tmp_path / "raw")
    exact_body = b'{ "provider": "OpenAI", "raw": true }\n'
    response = RawHttpResponse(
        status_code=429,
        headers=(("retry-after", "1"),),
        body=exact_body,
    )
    attempt = RawAttempt(
        request_sha256=request.request_sha256,
        attempt_number=1,
        observed_at_utc="2026-07-18T12:00:00Z",
        submission_catalog_sha256=_route()[0].catalog_sha256,
        response=response,
        transport_error_type=None,
    )
    cache.store_attempt(request, _route()[0], attempt)

    assert cache.load_request(request.request_sha256) == request
    assert cache.load_attempts(request) == (attempt,)
    response_path = (
        tmp_path
        / "raw"
        / request.request_sha256
        / "attempts"
        / "000001"
        / "response.body"
    )
    assert response_path.read_bytes() == exact_body
    with pytest.raises(FileExistsError):
        cache.store_attempt(request, _route()[0], attempt)

    response_path.write_bytes(exact_body + b"changed")
    with pytest.raises(GenerationCacheError, match="size mismatch"):
        cache.load_attempts(request)


def test_cost_journal_rejects_symlinked_contract_files(tmp_path: Path) -> None:
    request, route = _request(), _route()[0]
    cache = ImmutableRawCache(tmp_path / "journal-symlink")
    cache.prepare_request(request, route)
    ledger = _authorize(RuntimeCostLedger())
    ledger.attach_cache(cache)
    ledger.reserve(request, route, 1)
    entry_directory = next(
        (cache.root / "runtime-cost-journal").iterdir()
    )
    reservation = entry_directory / "reservation.json"
    outside = tmp_path / "outside-reservation.json"
    outside.write_bytes(reservation.read_bytes())
    reservation.unlink()
    reservation.symlink_to(outside)

    with pytest.raises(GenerationCacheError, match="regular file"):
        cache.load_cost_journal()


def test_paid_run_lock_is_exclusive_across_processes(tmp_path: Path) -> None:
    raw_cache = tmp_path / "cross-process-raw"
    child_code = (
        "from pathlib import Path\n"
        "import sys\n"
        "from apm.data.text.tinyworlds_v2.generation_costs import "
        "exclusive_paid_run_lock\n"
        "with exclusive_paid_run_lock(Path(sys.argv[1])):\n"
        "    print('locked', flush=True)\n"
        "    sys.stdin.readline()\n"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", child_code, str(raw_cache)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "locked"
        with pytest.raises(PaidRunLockUnavailable):
            with exclusive_paid_run_lock(raw_cache):
                pass
    finally:
        if process.stdin is not None:
            process.stdin.write("release\n")
            process.stdin.flush()
            process.stdin.close()
        assert process.wait(timeout=5) == 0

    with exclusive_paid_run_lock(raw_cache) as lock_path:
        assert lock_path.parent == raw_cache.parent
        assert lock_path != raw_cache


def test_client_retries_only_known_cost_429_and_5xx_then_caches_success(
    tmp_path: Path,
) -> None:
    delays: list[float] = []
    raw_429 = _response(429)
    transport = ScriptedTransport(
        [raw_429, _response(500), _response()]
    )
    client = _client(tmp_path, transport, delays=delays)
    route, _ = _route()
    request = _request()

    response = client.generate(request, route)

    assert response.body == _response().body
    assert response.usage == TokenUsage(5, 7, 12, 2)
    assert response.billed_cost_usd == "0.000021"
    assert response.provenance == ResponseProvenance(
        "gen-123",
        route.requested_model,
        route.canonical_model,
        route.returned_provider,
    )
    assert len(transport.calls) == 3
    assert delays == [0.25, 0.5]
    assert transport.calls[0][2] == request.body_bytes
    assert ("Authorization", "Bearer secret-openrouter-key") in transport.calls[0][1]
    assert ("X-OpenRouter-Metadata", "enabled") in transport.calls[0][1]
    assert ("X-OpenRouter-Cache", "false") in transport.calls[0][1]
    cached_bytes = b"".join(
        path.read_bytes()
        for path in (tmp_path / "raw").rglob("*")
        if path.is_file()
    )
    assert b"secret-openrouter-key" not in cached_bytes


@pytest.mark.parametrize(
    ("reflected_secret", "location"),
    (
        ("secret-openrouter-key", "body"),
        ("secret-openrouter-key", "header"),
        ("secret-management-key", "body"),
        ("secret-management-key", "header"),
    ),
)
def test_completion_secret_reflection_is_never_hashed_or_persisted(
    tmp_path: Path,
    reflected_secret: str,
    location: str,
) -> None:
    response = _response()
    if location == "body":
        record = strict_json_loads(response.body)
        record["reflected_credential"] = reflected_secret
        response = replace(response, body=canonical_json_bytes(record))
    else:
        response = replace(
            response,
            headers=(
                *response.headers,
                ("x-reflected-credential", reflected_secret),
            ),
        )
    transport = ScriptedTransport([response])
    ledger = RuntimeCostLedger()
    client = _client(
        tmp_path,
        transport,
        ledger=ledger,
        max_attempts=1,
    )
    request, route = _request(), _route()[0]

    with pytest.raises(
        OpenRouterContractError,
        match="not hashed or persisted",
    ) as caught:
        client.generate(request, route)

    assert reflected_secret not in str(caught.value)
    attempts = client.cache.load_attempts(request)
    assert len(attempts) == 1
    assert attempts[0].response is None
    assert (
        attempts[0].transport_error_type
        == "authentication_secret_reflection"
    )
    assert len(transport.calls) == 1
    resumed_transport = ScriptedTransport([_response()])
    resumed = _client(tmp_path, resumed_transport, max_attempts=2)
    with pytest.raises(OpenRouterContractError, match="no retry is permitted"):
        resumed.generate(request, route)
    assert resumed_transport.calls == []
    persisted = b"".join(
        path.read_bytes() for path in client.cache.root.rglob("*") if path.is_file()
    )
    assert b"secret-openrouter-key" not in persisted
    assert b"secret-management-key" not in persisted
    snapshot = ledger.snapshot()
    assert snapshot.unknown_cost_attempt_count == 1
    assert snapshot.halted_reason == "provider_secret_reflection"


def test_stats_secret_reflection_is_not_appended_to_safe_completion(
    tmp_path: Path,
) -> None:
    reflected = replace(
        _stats_response(),
        headers=(("x-reflected-credential", "secret-openrouter-key"),),
    )
    ledger = RuntimeCostLedger()
    transport = ScriptedTransport(
        [_response_requiring_stats()],
        stats_outcomes=[reflected],
    )
    client = _client(
        tmp_path,
        transport,
        ledger=ledger,
        max_attempts=1,
    )
    request, route = _request(), _route()[0]

    with pytest.raises(OpenRouterContractError, match="not hashed or persisted"):
        client.generate(request, route)

    attempts = client.cache.load_attempts(request)
    assert len(attempts) == 1
    assert attempts[0].response is not None
    stats_attempts = attempts[0].response.generation_stats_attempts
    assert len(stats_attempts) == 1
    assert stats_attempts[0].response is None
    assert (
        stats_attempts[0].transport_error_type
        == "authentication_secret_reflection"
    )
    assert len(transport.calls) == 1
    assert len(transport.stats_calls) == 1
    resumed_transport = ScriptedTransport(
        [],
        stats_outcomes=[_stats_response()],
    )
    resumed = _client(tmp_path, resumed_transport, max_attempts=2)
    with pytest.raises(OpenRouterContractError, match="no retry is permitted"):
        resumed.generate(request, route)
    assert resumed_transport.calls == []
    assert resumed_transport.stats_calls == []
    persisted = b"".join(
        path.read_bytes() for path in client.cache.root.rglob("*") if path.is_file()
    )
    assert b"secret-openrouter-key" not in persisted
    assert b"secret-management-key" not in persisted


def test_cached_success_resumes_without_calling_transport(tmp_path: Path) -> None:
    route, _ = _route()
    request = _request()
    initial_transport = ScriptedTransport([_response()])
    initial = _client(tmp_path, initial_transport)
    expected = initial.generate(request, route)
    unused_transport = ScriptedTransport([])

    resumed = _client(tmp_path, unused_transport)

    assert resumed.generate(request, route) == expected
    assert unused_transport.calls == []


def test_attempt_preserves_fresh_submission_catalog_across_cache_reuse(
    tmp_path: Path,
) -> None:
    original_route, request = _route()[0], _request()
    fresh_route = replace(original_route, catalog_sha256="b" * 64)
    later_route = replace(original_route, catalog_sha256="c" * 64)
    assert fresh_route.lock_sha256 == original_route.lock_sha256
    assert later_route.lock_sha256 == original_route.lock_sha256
    initial = _client(tmp_path, ScriptedTransport([_response()]))

    initial.generate(request, fresh_route)

    attempt = initial.cache.load_attempts(request)[0]
    assert attempt.submission_catalog_sha256 == "b" * 64
    unused = ScriptedTransport([])
    _client(tmp_path, unused).generate(request, later_route)
    assert unused.calls == []
    assert (
        initial.cache.load_attempts(request)[0].submission_catalog_sha256
        == "b" * 64
    )

    metadata = (
        initial.cache.root
        / request.request_sha256
        / "attempts"
        / "000001"
        / "metadata.json"
    )
    record = strict_json_loads(metadata.read_bytes())
    record["submission_catalog_sha256"] = "d" * 64
    metadata.write_bytes(canonical_json_bytes(record))
    with pytest.raises(GenerationCacheError, match="digest mismatch"):
        initial.cache.load_attempts(request)


def test_authenticated_byok_preflight_exposes_only_sanitized_zero_count(
    tmp_path: Path,
) -> None:
    raw = canonical_json_bytes({"data": [], "total_count": 0})
    transport = ScriptedTransport(
        [],
        stats_outcomes=[TransportResponse(200, (), raw)],
    )
    client = _client(tmp_path, transport)

    evidence = client.verify_no_byok()

    assert evidence.decision == "allowed"
    assert evidence.total_count == 0
    assert evidence.as_record() == {
        "attestation_sha256": None,
        "attested_at_utc": None,
        "checked_at_utc": "2026-07-18T12:00:00Z",
        "decision": "allowed",
        "endpoint": "/api/v1/byok",
        "expires_at_utc": None,
        "method": "GET",
        "response_body_sha256": evidence.response_body_sha256,
        "source": "management_api",
        "status_code": 200,
        "total_count": 0,
    }
    assert transport.stats_calls[0][0].endswith("/api/v1/byok")
    assert (
        "Authorization",
        "Bearer secret-management-key",
    ) in transport.stats_calls[0][1]
    assert (
        "Authorization",
        "Bearer secret-openrouter-key",
    ) not in transport.stats_calls[0][1]
    assert "secret-openrouter-key" not in str(evidence.as_record())


@pytest.mark.parametrize(
    ("reflected_secret", "location"),
    (
        ("secret-openrouter-key", "body"),
        ("secret-management-key", "header"),
    ),
)
def test_byok_preflight_secret_reflection_fails_without_digest_or_persistence(
    tmp_path: Path,
    reflected_secret: str,
    location: str,
) -> None:
    body = canonical_json_bytes({"data": [], "total_count": 0})
    headers: tuple[tuple[str, str], ...] = ()
    if location == "body":
        # JSON escapes prove the semantic scan catches a credential even when
        # its literal bytes do not occur in the provider response.
        escaped = "".join(
            f"\\u{ord(character):04x}" for character in reflected_secret
        )
        body = (
            '{"data":[],"reflected_credential":"'
            + escaped
            + '","total_count":0}'
        ).encode("ascii")
        assert reflected_secret.encode("utf-8") not in body
    else:
        headers = (("x-reflected-credential", reflected_secret),)
    client = _client(
        tmp_path,
        ScriptedTransport(
            [],
            stats_outcomes=[TransportResponse(200, headers, body)],
        ),
    )

    with pytest.raises(
        OpenRouterCostPolicyError,
        match="not hashed or persisted",
    ) as caught:
        client.verify_no_byok()

    assert caught.value.evidence is not None
    assert caught.value.evidence.response_body_sha256 is None
    serialized = str(caught.value.evidence.as_record()) + str(caught.value)
    assert "secret-openrouter-key" not in serialized
    assert "secret-management-key" not in serialized
    persisted = b"".join(
        path.read_bytes() for path in client.cache.root.rglob("*") if path.is_file()
    )
    assert b"secret-openrouter-key" not in persisted
    assert b"secret-management-key" not in persisted


@pytest.mark.parametrize(
    "response",
    (
        TransportResponse(
            200,
            (),
            canonical_json_bytes(
                {
                    "data": [
                        {
                            "id": "sensitive-key-id",
                            "label": "sensitive-label",
                        }
                    ],
                    "total_count": 1,
                }
            ),
        ),
        TransportResponse(403, (), b'{"error":"forbidden"}'),
    ),
)
def test_byok_preflight_fails_closed_without_exposing_key_metadata(
    tmp_path: Path,
    response: TransportResponse,
) -> None:
    client = _client(
        tmp_path,
        ScriptedTransport([], stats_outcomes=[response]),
    )

    with pytest.raises(OpenRouterCostPolicyError) as caught:
        client.verify_no_byok()

    assert caught.value.evidence is not None
    serialized = str(caught.value.evidence.as_record())
    assert "sensitive-key-id" not in serialized
    assert "sensitive-label" not in serialized


def test_required_byok_preflight_blocks_before_completion_post(tmp_path: Path) -> None:
    transport = ScriptedTransport([_response()])
    client = OpenRouterClient(
        api_key="secret-openrouter-key",
        transport=transport,
        cache=ImmutableRawCache(tmp_path / "preflight-required"),
        require_byok_preflight=True,
    )

    with pytest.raises(OpenRouterCostPolicyError, match="management key"):
        client.verify_no_byok()
    with pytest.raises(OpenRouterCostPolicyError, match="requires"):
        client.generate(_request(), _route()[0])
    assert transport.calls == []
    assert transport.stats_calls == []


def test_valid_manual_zero_byok_attestation_passes_without_management_get(
    tmp_path: Path,
) -> None:
    attestation = tmp_path / "no-byok.json"
    attestation.write_bytes(
        canonical_json_bytes(
            {
                "attested_at_utc": "2026-07-18T11:00:00Z",
                "expires_at_utc": "2026-07-18T13:00:00Z",
                "format": "apm.tinyworlds-v2.openrouter-no-byok-attestation",
                "schema_version": 1,
                "statement": (
                    "I attest that this OpenRouter workspace has zero "
                    "configured BYOK keys."
                ),
            }
        )
    )
    transport = ScriptedTransport([])
    client = OpenRouterClient(
        api_key="secret-openrouter-key",
        transport=transport,
        cache=ImmutableRawCache(tmp_path / "manual-attestation-cache"),
        byok_attestation_path=attestation,
        require_byok_preflight=True,
        clock=lambda: "2026-07-18T12:00:00Z",
    )

    evidence = client.verify_no_byok()

    assert evidence.source == "manual_attestation"
    assert evidence.decision == "allowed"
    assert evidence.total_count == 0
    assert evidence.attestation_sha256 is not None
    assert transport.stats_calls == []


@pytest.mark.parametrize(
    "record",
    (
        {
            "attested_at_utc": "2026-07-16T11:00:00Z",
            "expires_at_utc": "2026-07-16T13:00:00Z",
            "format": "apm.tinyworlds-v2.openrouter-no-byok-attestation",
            "schema_version": 1,
            "statement": (
                "I attest that this OpenRouter workspace has zero configured "
                "BYOK keys."
            ),
        },
        {"malformed": True},
    ),
)
def test_stale_or_malformed_manual_byok_attestation_fails_closed(
    tmp_path: Path,
    record: dict[str, object],
) -> None:
    attestation = tmp_path / "bad-no-byok.json"
    attestation.write_bytes(canonical_json_bytes(record))
    client = OpenRouterClient(
        api_key="secret-openrouter-key",
        transport=ScriptedTransport([_response()]),
        cache=ImmutableRawCache(tmp_path / "bad-attestation-cache"),
        byok_attestation_path=attestation,
        require_byok_preflight=True,
        clock=lambda: "2026-07-18T12:00:00Z",
    )

    with pytest.raises(OpenRouterCostPolicyError):
        client.verify_no_byok()


def test_completion_byok_observation_charges_unknown_bound_and_halts(
    tmp_path: Path,
) -> None:
    response = _response()
    record = strict_json_loads(response.body)
    record["openrouter_metadata"]["is_byok"] = True
    response = replace(response, body=canonical_json_bytes(record))
    ledger = RuntimeCostLedger()
    client = _client(
        tmp_path,
        ScriptedTransport([response]),
        ledger=ledger,
        max_attempts=1,
    )

    with pytest.raises(OpenRouterCostPolicyError, match="BYOK"):
        client.generate(_request(), _route()[0])

    attempt = client.cache.load_attempts(_request())[0]
    assert attempt.response is not None
    assert attempt.response.billed_cost_usd is None
    snapshot = ledger.snapshot()
    assert snapshot.provider_reported_actual_usd == "0"
    assert snapshot.unknown_cost_attempt_count == 1
    assert snapshot.halted_reason == "provider_cost_policy_violation"


def test_usage_byok_without_router_metadata_cannot_be_overridden_by_stats(
    tmp_path: Path,
) -> None:
    completion = _response_requiring_stats()
    record = strict_json_loads(completion.body)
    del record["openrouter_metadata"]
    record["usage"]["is_byok"] = True
    completion = replace(completion, body=canonical_json_bytes(record))
    transport = ScriptedTransport(
        [completion],
        stats_outcomes=[_stats_response()],
    )
    ledger = RuntimeCostLedger()
    client = _client(tmp_path, transport, ledger=ledger, max_attempts=1)

    with pytest.raises(OpenRouterCostPolicyError):
        client.generate(_request(), _route()[0])

    assert transport.stats_calls == []
    assert ledger.snapshot().halted_reason == "provider_cost_policy_violation"


@pytest.mark.parametrize("providers", [(), ("OpenAI", "Unexpected")])
def test_metadata_requires_exactly_one_selected_provider_and_caches_failure(
    tmp_path: Path,
    providers: tuple[str, ...],
) -> None:
    transport = ScriptedTransport([_response_with_selected_providers(providers)])
    client = _client(tmp_path, transport)
    request, route = _request(), _route()[0]

    with pytest.raises(OpenRouterContractError, match="exactly one selected"):
        client.generate(request, route)

    assert len(client.cache.load_attempts(request)) == 1
    assert transport.stats_calls == []


def test_stats_fallback_is_exact_cached_and_resumes_without_network(
    tmp_path: Path,
) -> None:
    stats = _stats_response()
    transport = ScriptedTransport(
        [_response_requiring_stats()],
        stats_outcomes=[stats],
    )
    route, request = _route()[0], _request()
    client = _client(tmp_path, transport)

    response = client.generate(request, route)

    assert response.provenance == ResponseProvenance(
        "gen-123",
        route.requested_model,
        route.canonical_model,
        route.returned_provider,
    )
    assert response.usage == TokenUsage(5, 7, 12, 0)
    assert response.billed_cost_usd == "0.000021"
    assert len(response.generation_stats_attempts) == 1
    assert response.generation_stats_attempts[0].response is not None
    assert response.generation_stats_attempts[0].response.body == stats.body
    assert len(transport.stats_calls) == 1
    stats_url, stats_headers, _ = transport.stats_calls[0]
    assert stats_url.endswith("/api/v1/generation?id=gen-123")
    assert ("Authorization", "Bearer secret-openrouter-key") in stats_headers

    unused = ScriptedTransport([])
    resumed = _client(tmp_path, unused)
    assert resumed.generate(request, route) == response
    assert unused.calls == []
    assert unused.stats_calls == []

    cached_bytes = b"".join(
        path.read_bytes()
        for path in (tmp_path / "raw").rglob("*")
        if path.is_file()
    )
    assert b"secret-openrouter-key" not in cached_bytes
    stats_path = (
        tmp_path
        / "raw"
        / request.request_sha256
        / "attempts"
        / "000001"
        / "generation-stats"
        / "000001"
        / "response.body"
    )
    assert stats_path.read_bytes() == stats.body
    stats_path.write_bytes(stats.body + b"tampered")
    with pytest.raises(GenerationCacheError, match="size mismatch"):
        client.cache.load_attempts(request)


def test_success_cache_hit_without_router_metadata_recovers_through_stats(
    tmp_path: Path,
) -> None:
    completion = _response_requiring_stats()
    record = strict_json_loads(completion.body)
    del record["openrouter_metadata"]
    completion = replace(completion, body=canonical_json_bytes(record))
    transport = ScriptedTransport(
        [completion],
        stats_outcomes=[_stats_response()],
    )

    response = _client(tmp_path, transport).generate(_request(), _route()[0])

    assert response.provenance is not None
    assert response.provenance.returned_provider == "OpenAI"
    assert response.billed_cost_usd == "0.000021"
    assert len(transport.stats_calls) == 1


def test_stats_accept_explicit_non_byok_when_upstream_cost_field_is_absent(
    tmp_path: Path,
) -> None:
    stats = _stats_response()
    record = strict_json_loads(stats.body)
    del record["data"]["upstream_inference_cost"]
    stats = replace(stats, body=canonical_json_bytes(record))
    client = _client(
        tmp_path,
        ScriptedTransport(
            [_response_requiring_stats()],
            stats_outcomes=[stats],
        ),
    )

    response = client.generate(_request(), _route()[0])

    assert response.billed_cost_usd == "0.000021"


def test_stats_byok_halts_as_unknown(
    tmp_path: Path,
) -> None:
    stats = _stats_response()
    record = strict_json_loads(stats.body)
    record["data"]["is_byok"] = True
    record["data"]["upstream_inference_cost"] = "0.25"
    stats = replace(stats, body=canonical_json_bytes(record))
    ledger = RuntimeCostLedger()
    client = _client(
        tmp_path,
        ScriptedTransport(
            [_response_requiring_stats()],
            stats_outcomes=[stats],
        ),
        ledger=ledger,
        max_attempts=1,
    )

    with pytest.raises(OpenRouterCostPolicyError):
        client.generate(_request(), _route()[0])

    assert ledger.snapshot().unknown_cost_attempt_count == 1
    assert ledger.snapshot().halted_reason == "provider_cost_policy_violation"


def test_non_byok_upstream_provider_cost_is_informational(tmp_path: Path) -> None:
    stats = _stats_response(cost="0.000021")
    record = strict_json_loads(stats.body)
    record["data"]["upstream_inference_cost"] = "0.0012"
    stats = replace(stats, body=canonical_json_bytes(record))
    client = _client(
        tmp_path,
        ScriptedTransport(
            [_response_requiring_stats()],
            stats_outcomes=[stats],
        ),
    )

    response = client.generate(_request(), _route()[0])

    assert response.billed_cost_usd == "0.000021"


def test_stats_route_mismatch_is_cached_and_never_hidden_by_rebuild(
    tmp_path: Path,
) -> None:
    transport = ScriptedTransport(
        [_response_requiring_stats()],
        stats_outcomes=[_stats_response(provider="Unexpected Provider")],
    )
    request, route = _request(), _route()[0]
    client = _client(tmp_path, transport)

    with pytest.raises(OpenRouterContractError, match="providers differ"):
        client.generate(request, route)
    assert len(client.cache.load_attempts(request)) == 1

    unused = ScriptedTransport([])
    with pytest.raises(OpenRouterContractError, match="providers differ"):
        _client(tmp_path, unused).generate(request, route)
    assert unused.calls == []
    assert unused.stats_calls == []


@pytest.mark.parametrize(
    ("stats", "message"),
    [
        (_stats_response(generation_id="another-id"), "ids differ"),
        (
            _stats_response(returned_model="openai/a-different-model"),
            "models differ",
        ),
    ],
)
def test_stats_identity_must_match_the_completion(
    tmp_path: Path,
    stats: TransportResponse,
    message: str,
) -> None:
    client = _client(
        tmp_path,
        ScriptedTransport([_response_requiring_stats()], stats_outcomes=[stats]),
    )
    with pytest.raises(OpenRouterContractError, match=message):
        client.generate(_request(), _route()[0])


def test_stats_cost_must_match_completion_cost_when_both_are_available(
    tmp_path: Path,
) -> None:
    completion = _response_requiring_stats()
    record = strict_json_loads(completion.body)
    record["usage"]["cost"] = 0.000021
    del record["usage"]["total_tokens"]
    completion = replace(completion, body=canonical_json_bytes(record))
    client = _client(
        tmp_path,
        ScriptedTransport(
            [completion],
            stats_outcomes=[_stats_response(cost="0.000031")],
        ),
    )

    with pytest.raises(OpenRouterContractError, match="billed costs differ"):
        client.generate(_request(), _route()[0])


def test_retryable_failure_recovers_and_caches_its_provider_billed_cost(
    tmp_path: Path,
) -> None:
    retry_response = _response_requiring_stats(
        status_code=500,
        generation_id="gen-retry",
        include_body_id=False,
    )
    transport = ScriptedTransport(
        [retry_response, _response()],
        stats_outcomes=[_stats_response(generation_id="gen-retry", cost="0.000031")],
    )
    request, route = _request(), _route()[0]
    client = _client(tmp_path, transport)

    assert client.generate(request, route).status_code == 200
    attempts = client.cache.load_attempts(request)
    assert len(attempts) == 2
    assert attempts[0].response is not None
    assert attempts[0].response.billed_cost_usd == "0.000031"
    assert attempts[0].response.generation_stats_attempts


def test_failure_with_zero_selected_provider_uses_authenticated_stats(
    tmp_path: Path,
) -> None:
    failure = _response_requiring_stats(
        status_code=500,
        generation_id="failure-id",
        include_body_id=False,
    )
    client = _client(
        tmp_path,
        ScriptedTransport(
            [failure, _response()],
            stats_outcomes=[
                _stats_response(generation_id="failure-id", cost="0.000031")
            ],
        ),
    )

    assert client.generate(_request(), _route()[0]).status_code == 200
    first = client.cache.load_attempts(_request())[0].response
    assert first is not None
    record = strict_json_loads(first.body)
    assert record["openrouter_metadata"]["endpoints"]["available"] == []
    stats_record = strict_json_loads(
        first.generation_stats_attempts[-1].response.body
    )
    assert stats_record["data"]["provider_name"] == "OpenAI"


def test_unidentified_zero_selected_client_error_fails_closed_without_stats(
    tmp_path: Path,
) -> None:
    failure = _response_requiring_stats(status_code=400, include_body_id=False)
    record = strict_json_loads(failure.body)
    failure = replace(failure, headers=(), body=canonical_json_bytes(record))
    ledger = RuntimeCostLedger()
    transport = ScriptedTransport([failure])
    client = _client(tmp_path, transport, ledger=ledger, max_attempts=1)

    with pytest.raises(OpenRouterBillingUnknown):
        client.generate(_request(), _route()[0])

    assert transport.stats_calls == []
    assert ledger.snapshot().unknown_cost_attempt_count == 1


@pytest.mark.parametrize("status_code", [400, 408])
def test_terminal_client_errors_are_not_retried(
    tmp_path: Path,
    status_code: int,
) -> None:
    transport = ScriptedTransport(
        [_response(status_code)]
    )
    client = _client(tmp_path, transport)

    with pytest.raises(OpenRouterResponseError) as raised:
        client.generate(_request(), _route()[0])

    assert raised.value.status_code == status_code
    assert len(transport.calls) == 1


def test_malformed_or_route_mismatched_success_is_cached_but_not_retried(
    tmp_path: Path,
) -> None:
    cases = (
        TransportResponse(200, (), b"not JSON"),
        _response(provider="Unexpected Provider"),
        _response(returned_model="openai/something-else"),
    )
    for index, response in enumerate(cases):
        case_root = tmp_path / str(index)
        transport = ScriptedTransport([response])
        client = _client(case_root, transport)
        request, route = _request(), _route()[0]

        with pytest.raises(OpenRouterContractError):
            client.generate(request, route)
        assert len(transport.calls) == 1

        no_network = _client(case_root, ScriptedTransport([]))
        with pytest.raises(OpenRouterContractError):
            no_network.generate(request, route)


def test_unrecognized_transport_exception_is_not_retried_or_cached(
    tmp_path: Path,
) -> None:
    transport = ScriptedTransport([ValueError("programming error")])
    client = _client(tmp_path, transport)
    request = _request()

    with pytest.raises(ValueError, match="programming error"):
        client.generate(request, _route()[0])

    assert len(transport.calls) == 1
    assert client.cache.load_attempts(request) == ()


def test_ambiguous_completion_transport_failure_is_never_retried(
    tmp_path: Path,
) -> None:
    route, _ = _route()
    request = _request()
    first_transport = ScriptedTransport([TransportError("temporary")])
    first_client = _client(tmp_path, first_transport, max_attempts=1)
    with pytest.raises(OpenRouterBillingUnknown):
        first_client.generate(request, route)

    resumed_transport = ScriptedTransport([_response()])
    resumed_client = _client(tmp_path, resumed_transport, max_attempts=3)

    with pytest.raises(OpenRouterBillingUnknown):
        resumed_client.generate(request, route)
    assert resumed_transport.calls == []
    assert tuple(
        attempt.attempt_number
        for attempt in resumed_client.cache.load_attempts(request)
    ) == (1,)
    snapshot = resumed_client.cost_ledger.snapshot()
    assert snapshot.unknown_cost_attempt_count == 1
    assert snapshot.halted_reason == "provider_billing_unknown"


def test_transport_error_cannot_reflect_secret_into_cache_or_exception(
    tmp_path: Path,
) -> None:
    client = _client(
        tmp_path,
        ScriptedTransport([TransportError("secret-openrouter-key")]),
        max_attempts=1,
    )
    request, route = _request(), _route()[0]

    with pytest.raises(OpenRouterBillingUnknown) as caught:
        client.generate(request, route)

    assert caught.value.__cause__ is None
    assert "secret-openrouter-key" not in str(caught.value)
    attempts = client.cache.load_attempts(request)
    assert attempts[0].transport_error_type == "authentication_secret_reflection"
    persisted = b"".join(
        path.read_bytes() for path in client.cache.root.rglob("*") if path.is_file()
    )
    assert b"secret-openrouter-key" not in persisted


def test_runtime_cap_reservations_are_thread_safe_and_reject_before_post(
    tmp_path: Path,
) -> None:
    route = _route()[0]
    requests = tuple(_request(extra_body={"seed": index}) for index in range(8))
    bounds = tuple(
        Decimal(request_cost_upper_bound(request, route).upper_bound_usd)
        for request in requests
    )
    assert len(set(bounds)) == 1
    ledger = RuntimeCostLedger(format(bounds[0] * 3, "f"))
    barrier = Barrier(3)

    class BlockingTransport(ScriptedTransport):
        def post(
            self,
            *,
            url: str,
            headers: tuple[tuple[str, str], ...],
            body: bytes,
            timeout_seconds: float,
        ) -> TransportResponse:
            self.calls.append((url, headers, body, timeout_seconds))
            barrier.wait(timeout=5)
            return _response()

    transport = BlockingTransport([_response() for _ in requests])
    client = _client(tmp_path, transport, ledger=ledger, max_attempts=1)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = tuple(
            executor.submit(client.generate, request, route) for request in requests
        )
        outcomes = tuple(
            future.exception() if future.exception() is not None else future.result()
            for future in futures
        )

    successes = tuple(item for item in outcomes if isinstance(item, RawHttpResponse))
    failures = tuple(item for item in outcomes if isinstance(item, Exception))
    assert len(successes) == 3
    assert all(isinstance(item, CostCapExceeded) for item in failures)
    assert len(transport.calls) == 3
    snapshot = ledger.snapshot()
    assert snapshot.in_flight_attempt_count == 0
    assert Decimal(snapshot.charged_total_usd) <= Decimal(snapshot.hard_cap_usd)
    assert snapshot.halted_reason == "runtime_cap_reservation_denied"


def test_authorized_post_survives_cap_denial_but_unknown_billing_overrides_it(
    tmp_path: Path,
) -> None:
    route = _route()[0]
    authorized_request = _request(extra_body={"seed": 1})
    denied_request = _request(extra_body={"seed": 2})
    bound = request_cost_upper_bound(authorized_request, route).upper_bound_usd
    ledger = _authorize(RuntimeCostLedger(bound))
    ledger.attach_cache(ImmutableRawCache(tmp_path / "raw"))

    reservation = ledger.reserve(authorized_request, route, 1)
    with pytest.raises(CostCapExceeded):
        ledger.reserve(denied_request, route, 1)

    ledger.ensure_reservation_postable(reservation)
    ledger.settle_ambiguous_transport_failure(reservation)
    snapshot = ledger.snapshot()
    assert snapshot.conservative_unknown_charge_usd == bound
    assert snapshot.in_flight_attempt_count == 0
    assert snapshot.halted_reason == "provider_billing_unknown"


def test_halted_reservation_is_durably_cancelled_before_post_and_restart_safe(
    tmp_path: Path,
) -> None:
    request, route = _request(), _route()[0]
    cache = ImmutableRawCache(tmp_path / "cancelled-before-post")
    ledger = _authorize(RuntimeCostLedger())
    ledger.attach_cache(cache)
    reservation = ledger.reserve(request, route, 1)

    # Model another worker publishing a durable fail-closed halt after this
    # reservation but before this worker crosses the transport boundary.
    ledger.halt("provider_cost_policy_violation")
    with pytest.raises(CostCapExceeded, match="before the reserved POST began"):
        ledger.ensure_reservation_postable(reservation)

    entry = cache.load_cost_journal()[0]
    assert entry.cancelled_before_post is True
    assert entry.charged_usd is None
    assert entry.provider_reported_actual is None
    assert entry.byok_authorization == _byok_authorization()
    assert entry.byok_authorization_sha256 == json_sha256(_byok_authorization())
    snapshot = ledger.snapshot()
    assert snapshot.charged_total_usd == "0"
    assert snapshot.in_flight_attempt_count == 0
    assert snapshot.cancelled_before_post_count == 1
    assert cache.load_attempts(request) == ()

    with pytest.raises(FileExistsError, match="already terminal"):
        cache.store_cost_settlement(
            request.request_sha256,
            1,
            charged_usd=reservation.upper_bound_usd,
            provider_reported_actual=False,
        )
    with pytest.raises(GenerationCacheError, match="cannot acquire a raw attempt"):
        cache.store_attempt(
            request,
            route,
            RawAttempt(
                request_sha256=request.request_sha256,
                attempt_number=1,
                observed_at_utc="2026-07-18T12:00:01Z",
                submission_catalog_sha256=route.catalog_sha256,
                response=None,
                transport_error_type="timeout",
            ),
        )

    restarted = RuntimeCostLedger()
    restarted.bootstrap(cache, (route,))
    resumed = restarted.snapshot()
    assert resumed.charged_total_usd == "0"
    assert resumed.in_flight_attempt_count == 0
    assert resumed.cancelled_before_post_count == 1
    assert resumed.halted_reason is None


def test_runtime_cap_bootstraps_cached_actuals_and_is_inclusive(tmp_path: Path) -> None:
    route = _route()[0]
    first_request = _request(extra_body={"seed": 1})
    cache = ImmutableRawCache(tmp_path / "raw")
    first = _client(tmp_path, ScriptedTransport([_response()]), max_attempts=1)
    assert first.generate(first_request, route).billed_cost_usd == "0.000021"

    ledger = RuntimeCostLedger("0.000021")
    ledger.bootstrap(cache, (route,))
    resumed_transport = ScriptedTransport([])
    resumed = _client(
        tmp_path,
        resumed_transport,
        max_attempts=1,
        ledger=ledger,
    )
    assert resumed.generate(first_request, route).status_code == 200
    assert resumed_transport.calls == []
    with pytest.raises(CostCapExceeded):
        resumed.generate(_request(extra_body={"seed": 2}), route)
    assert resumed_transport.calls == []


def test_runtime_cap_bootstrap_counts_prior_catalog_lock_attempts(
    tmp_path: Path,
) -> None:
    current_route, observed = _route()
    historical_route = lock_catalog_route(
        current_route.route_id,
        replace(observed, catalog_sha256="b" * 64),
    )
    body = _request().body
    historical_request = CanonicalRequest.from_body(
        route_lock_sha256=historical_route.lock_sha256,
        endpoint="/api/v1/chat/completions",
        body=body,
    )
    cache = ImmutableRawCache(tmp_path / "historical")
    cache.store_attempt(
        historical_request,
        historical_route,
        RawAttempt(
            request_sha256=historical_request.request_sha256,
            attempt_number=1,
            observed_at_utc="2026-07-18T12:00:00Z",
            submission_catalog_sha256=historical_route.catalog_sha256,
            response=RawHttpResponse(
                status_code=500,
                headers=(),
                body=b'{"error":"historical"}',
                billed_cost_usd="0.000021",
            ),
            transport_error_type=None,
        ),
    )
    ledger = RuntimeCostLedger("0.000021")

    ledger.bootstrap(cache, (current_route,))
    ledger.authorize_byok(_byok_authorization())

    snapshot = ledger.snapshot()
    assert snapshot.provider_reported_actual_usd == "0.000021"
    with pytest.raises(CostCapExceeded):
        ledger.reserve(_request(extra_body={"seed": 2}), current_route, 1)


def test_orphaned_cost_reservation_charges_bound_and_stops_restart(
    tmp_path: Path,
) -> None:
    request, route = _request(), _route()[0]
    cache = ImmutableRawCache(tmp_path / "orphaned-reservation")
    cache.prepare_request(request, route)
    original = _authorize(RuntimeCostLedger())
    original.attach_cache(cache)
    original.reserve(request, route, 1)

    restarted = RuntimeCostLedger()
    with pytest.raises(
        CostJournalRecoveryRequired,
        match="orphaned_cost_reservation",
    ):
        restarted.bootstrap(cache, (route,))

    bound = request_cost_upper_bound(request, route).upper_bound_usd
    snapshot = restarted.snapshot()
    assert snapshot.conservative_unknown_charge_usd == bound
    assert snapshot.provider_reported_actual_usd == "0"
    assert snapshot.halted_reason == "orphaned_cost_reservation"
    entry = cache.load_cost_journal()[0]
    assert entry.charged_usd is None
    assert entry.upper_bound_usd == bound


def test_settled_billed_attempt_without_response_never_reposts(
    tmp_path: Path,
) -> None:
    request, route = _request(), _route()[0]
    cache = ImmutableRawCache(tmp_path / "missing-response")
    cache.prepare_request(request, route)
    original = _authorize(RuntimeCostLedger())
    original.attach_cache(cache)
    reservation = original.reserve(request, route, 1)
    original.settle_response(reservation, "0.000021")

    restarted = RuntimeCostLedger()
    with pytest.raises(
        CostJournalRecoveryRequired,
        match="billed_attempt_response_missing",
    ):
        restarted.bootstrap(cache, (route,))
    transport = ScriptedTransport([_response()])
    client = OpenRouterClient(
        api_key="secret-openrouter-key",
        transport=transport,
        cache=cache,
        cost_ledger=restarted,
        clock=lambda: "2026-07-18T12:00:00Z",
    )
    with pytest.raises(CostJournalRecoveryRequired):
        client.generate(request, route)
    assert transport.calls == []
    snapshot = restarted.snapshot()
    assert snapshot.provider_reported_actual_usd == "0.000021"
    assert snapshot.conservative_unknown_charge_usd == "0"
    assert snapshot.halted_reason == "billed_attempt_response_missing"


def test_cached_response_settles_write_ahead_reservation_after_restart(
    tmp_path: Path,
) -> None:
    request, route = _request(), _route()[0]
    cache = ImmutableRawCache(tmp_path / "recoverable-orphan")
    cache.prepare_request(request, route)
    original = _authorize(RuntimeCostLedger())
    original.attach_cache(cache)
    original.reserve(request, route, 1)
    parsed = RawHttpResponse(
        status_code=200,
        headers=_response().headers,
        body=_response().body,
        provenance=ResponseProvenance(
            "gen-123",
            route.requested_model,
            route.canonical_model,
            route.returned_provider,
        ),
        usage=TokenUsage(5, 7, 12, 2),
        billed_cost_usd="0.000021",
    )
    cache.store_attempt(
        request,
        route,
        RawAttempt(
            request_sha256=request.request_sha256,
            attempt_number=1,
            observed_at_utc="2026-07-18T12:00:00Z",
            submission_catalog_sha256=route.catalog_sha256,
            response=parsed,
            transport_error_type=None,
        ),
    )

    restarted = RuntimeCostLedger()
    restarted.bootstrap(cache, (route,))

    snapshot = restarted.snapshot()
    assert snapshot.provider_reported_actual_usd == "0.000021"
    assert snapshot.conservative_unknown_charge_usd == "0"
    assert snapshot.halted_reason is None
    entry = cache.load_cost_journal()[0]
    assert entry.charged_usd == "0.000021"
    assert entry.provider_reported_actual is True

def test_unknown_response_cost_charges_bound_but_is_not_labeled_actual(
    tmp_path: Path,
) -> None:
    route, request = _route()[0], _request()
    transport = ScriptedTransport(
        [TransportResponse(400, (), b'{"error":"bad request"}')]
    )
    ledger = RuntimeCostLedger()
    client = _client(tmp_path, transport, max_attempts=1, ledger=ledger)

    with pytest.raises(OpenRouterBillingUnknown):
        client.generate(request, route)

    attempts = client.cache.load_attempts(request)
    assert attempts[0].response is not None
    assert attempts[0].response.billed_cost_usd is None
    snapshot = ledger.snapshot()
    assert snapshot.provider_reported_actual_usd == "0"
    assert Decimal(snapshot.conservative_unknown_charge_usd) == Decimal(
        request_cost_upper_bound(request, route).upper_bound_usd
    )
    assert snapshot.halted_reason == "provider_billing_unknown"


def test_repeating_decimal_max_price_is_encoded_as_a_conservative_cap() -> None:
    route, _ = _route()
    exact_per_token = "0.00000016666666666666667"
    exact_per_million = Decimal(exact_per_token) * Decimal(1_000_000)
    encoded = _per_million_price(exact_per_token)
    assert Decimal(str(encoded)) >= exact_per_million

    route = replace(
        route,
        input_usd_per_million=format(exact_per_million, "f"),
    )
    original = _request()
    body = original.body
    body["provider"]["max_price"]["prompt"] = encoded
    request = CanonicalRequest.from_body(
        route_lock_sha256=route.lock_sha256,
        endpoint=original.endpoint,
        body=body,
    )
    bound = request_cost_upper_bound(request, route)
    assert Decimal(bound.upper_bound_usd) > 0


def test_failed_attempt_cost_is_retained_without_complete_token_usage(
    tmp_path: Path,
) -> None:
    retry = _response_requiring_stats(status_code=500, generation_id="gen-retry")
    retry_record = strict_json_loads(retry.body)
    del retry_record["usage"]
    retry = replace(retry, body=canonical_json_bytes(retry_record))
    stats_record = strict_json_loads(
        _stats_response(generation_id="gen-retry", cost="0.000031").body
    )
    del stats_record["data"]["tokens_prompt"]
    del stats_record["data"]["tokens_completion"]
    stats = TransportResponse(200, (), canonical_json_bytes(stats_record))
    client = _client(
        tmp_path,
        ScriptedTransport([retry, _response()], stats_outcomes=[stats]),
    )

    assert client.generate(_request(), _route()[0]).status_code == 200
    attempts = client.cache.load_attempts(_request())
    assert attempts[0].response is not None
    assert attempts[0].response.usage is None
    assert attempts[0].response.billed_cost_usd == "0.000031"


def test_stats_polling_preserves_every_retry_and_never_reposts_completion(
    tmp_path: Path,
) -> None:
    delays: list[float] = []
    transport = ScriptedTransport(
        [_response_requiring_stats()],
        stats_outcomes=[
            TransportError("temporary"),
            _stats_response(status_code=404),
            _stats_response(status_code=500),
            _stats_response(),
        ],
    )
    client = _client(tmp_path, transport, delays=delays)

    response = client.generate(_request(), _route()[0])

    assert response.status_code == 200
    assert len(transport.calls) == 1
    assert len(transport.stats_calls) == 4
    assert delays == [0.25, 0.5, 1.0]
    attempts = client.cache.load_attempts(_request())
    stats_attempts = attempts[0].response.generation_stats_attempts
    assert tuple(item.attempt_number for item in stats_attempts) == (1, 2, 3, 4)
    assert stats_attempts[0].transport_error_type == "temporary"
    assert tuple(
        None if item.response is None else item.response.status_code
        for item in stats_attempts
    ) == (None, 404, 500, 200)


def test_stats_polling_resumes_without_repeating_cached_completion_post(
    tmp_path: Path,
) -> None:
    request, route = _request(), _route()[0]
    first_transport = ScriptedTransport(
        [_response_requiring_stats()],
        stats_outcomes=[
            _stats_response(status_code=404),
            TransportError("temporary"),
        ],
    )
    first = _client(tmp_path, first_transport, stats_max_attempts=2)
    with pytest.raises(OpenRouterBillingUnknown):
        first.generate(request, route)
    assert len(first_transport.calls) == 1

    resumed_transport = ScriptedTransport([], stats_outcomes=[_stats_response()])
    resumed = _client(tmp_path, resumed_transport, stats_max_attempts=4)
    response = resumed.generate(request, route)

    assert response.status_code == 200
    assert resumed_transport.calls == []
    assert len(resumed_transport.stats_calls) == 1
    cached = resumed.cache.load_attempts(request)[0].response
    assert cached is not None
    assert tuple(
        item.attempt_number for item in cached.generation_stats_attempts
    ) == (1, 2, 3)

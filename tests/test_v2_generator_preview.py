from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import runpy
import socket

import pytest

import apm.data.text.tinyworlds_v2.generator_preview as generator_preview
from apm.data.text.tinyworlds_v2.bakeoff import (
    CANDIDATE_MODELS,
    SYNTHETIC_STORY_REQUEST_V3,
)
from apm.data.text.tinyworlds_v2.generator_preview import (
    GENERATOR_PREVIEW_ATTEMPTS_PER_REQUEST,
    GENERATOR_PREVIEW_HARD_CAP_USD,
    GENERATOR_PREVIEW_REQUEST_CONTRACT,
    GENERATOR_PREVIEW_RUN_CAP_USD,
    GENERATOR_PREVIEW_SELECTION_SEED,
    GENERATOR_PREVIEW_SOURCE_RECORD_IDS,
    GENERATOR_PREVIEW_STATS_ATTEMPTS,
    GENERATOR_PREVIEW_VERSION,
    GeneratorPreviewPaths,
    _bounded_missing_cost_failure_kind,
    _cached_byok_preflight,
    _cost_actuals,
    _promote_directory_no_replace,
    _preview_client_for_cache,
    _runtime_cost_record_from_journal,
    _validate_current_cost_records,
    build_generator_preview_preflight,
    load_generator_preview_briefs,
)
from apm.data.text.tinyworlds_v2.generation_cache import (
    CostJournalEntry,
    ImmutableRawCache,
)
from apm.data.text.tinyworlds_v2.generation_costs import (
    RuntimeCostLedger,
    request_cost_upper_bound,
)
from apm.data.text.tinyworlds_v2.generation_schema import (
    RawAttempt,
    RawGenerationStatsAttempt,
    RawGenerationStatsResponse,
    RawHttpResponse,
    RouteLock,
)
from apm.data.text.tinyworlds_v2.json_contracts import canonical_json_line_bytes
from apm.data.text.tinyworlds_v2.phase1_generation import build_generation_jobs


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "preview_tinyworlds_v2_generators.py"


def _prompt_record(
    record_id: str,
    *,
    words: tuple[str, str, str],
    features: tuple[str, ...],
) -> dict[str, object]:
    verb, noun, adjective = words
    feature_clause = (
        " The story has the following features: " + ", ".join(features) + "."
        if features
        else ""
    )
    prompt = (
        "Write a short story which only uses simple words. The story should use "
        f'the verb "{verb}", the noun "{noun}" and the adjective "{adjective}".'
        f"{feature_clause}"
    )
    return {
        "content_sha256": record_id.rsplit(":", 1)[-1],
        "features": list(features),
        "normalized_story_sha256": "a" * 64,
        "prompt": prompt,
        "record_id": record_id,
        "source": "GPT-4",
        "source_index": 1,
        "source_member": "./fixture.json",
        "story": f"A {adjective} {noun} liked to {verb}. It had a happy day.",
        "summary": "A small fixture story.",
        "words": list(words),
    }


def _write_prompt_fixture(path: Path) -> None:
    records = (
        _prompt_record(
            GENERATOR_PREVIEW_SOURCE_RECORD_IDS[2],
            words=("own", "waste", "clear"),
            features=("Dialogue", "MoralValue"),
        ),
        _prompt_record(
            GENERATOR_PREVIEW_SOURCE_RECORD_IDS[0],
            words=("let", "sandwich", "different"),
            features=(),
        ),
        _prompt_record(
            GENERATOR_PREVIEW_SOURCE_RECORD_IDS[1],
            words=("chew", "gem", "lonely"),
            features=("Dialogue",),
        ),
    )
    path.write_bytes(b"".join(canonical_json_line_bytes(record) for record in records))


def _routes() -> tuple[RouteLock, ...]:
    return tuple(
        RouteLock(
            route_id=model.route_id,
            catalog_sha256="b" * 64,
            requested_model=model.request_model_id,
            canonical_model=model.canonical_slug,
            provider_slug=model.first_party_provider_slug or f"fixture/{model.route_id}",
            returned_provider=model.first_party_provider_slug or model.route_id,
            quantization="bf16",
            input_usd_per_million=model.plan_prompt_usd_per_million,
            output_usd_per_million=model.plan_completion_usd_per_million,
        )
        for model in CANDIDATE_MODELS
    )


def test_preview_selects_disjoint_namespaced_feature_strata(tmp_path: Path) -> None:
    source = tmp_path / "prompt_metadata_sample.jsonl"
    _write_prompt_fixture(source)

    briefs = load_generator_preview_briefs(source)

    assert tuple(brief.source_record_id for brief in briefs) == (
        GENERATOR_PREVIEW_SOURCE_RECORD_IDS
    )
    assert tuple(brief.required_words for brief in briefs) == (
        ("let", "sandwich", "different"),
        ("chew", "gem", "lonely"),
        ("own", "waste", "clear"),
    )
    assert tuple(brief.requested_features for brief in briefs) == (
        (),
        ("Dialogue",),
        ("Dialogue", "MoralValue"),
    )
    expected_ids = tuple(
        "preview-brief-"
        + sha256(
            f"{GENERATOR_PREVIEW_SELECTION_SEED}\0{record_id}".encode("utf-8")
        ).hexdigest()[:24]
        for record_id in GENERATOR_PREVIEW_SOURCE_RECORD_IDS
    )
    assert tuple(brief.brief_id for brief in briefs) == expected_ids


def test_preview_preflight_has_21_jobs_one_attempt_and_five_cent_cap(
    tmp_path: Path,
) -> None:
    source = tmp_path / "prompt_metadata_sample.jsonl"
    _write_prompt_fixture(source)
    briefs = load_generator_preview_briefs(source)
    jobs = build_generation_jobs(
        briefs,
        CANDIDATE_MODELS,
        _routes(),
        request_contract=GENERATOR_PREVIEW_REQUEST_CONTRACT,
    )

    preflight = build_generator_preview_preflight(
        jobs,
        lambda text: tuple(text.encode("utf-8")),
    )

    assert preflight.permitted
    assert preflight.hard_cap_usd == "0.041752"
    assert sum(item.request_count for item in preflight.route_estimates) == 21
    assert all(item.request_count == 3 for item in preflight.route_estimates)
    assert all(
        "enforce_distillable_text" not in job.request.body["provider"]
        and 0 <= job.request.body["seed"] <= 0x7FFF_FFFF
        and "json"
        in "\n".join(
            message["content"] for message in job.request.body["messages"]
        ).casefold()
        for job in jobs
    )
    assert GENERATOR_PREVIEW_HARD_CAP_USD == "0.05"
    assert GENERATOR_PREVIEW_ATTEMPTS_PER_REQUEST == 1


def test_preview_paths_are_isolated_from_phase1_funnel(tmp_path: Path) -> None:
    paths = GeneratorPreviewPaths.from_repository(tmp_path)

    assert paths.raw_cache != tmp_path / "data/tinyworlds-v2/cache/phase1-openrouter"
    assert paths.destination != tmp_path / "data/tinyworlds-v2/reference"
    assert "route-preview" in paths.raw_cache.as_posix()
    assert "route-preview" in paths.destination.as_posix()
    assert paths.byok_attestation.name == (
        "openrouter-tinyworlds-preview-no-byok-attestation.json"
    )


def test_corrected_preview_has_new_version_cache_and_destination(tmp_path: Path) -> None:
    paths = GeneratorPreviewPaths.from_repository(tmp_path)

    assert GENERATOR_PREVIEW_VERSION.endswith("-v3")
    assert GENERATOR_PREVIEW_REQUEST_CONTRACT == SYNTHETIC_STORY_REQUEST_V3
    assert GENERATOR_PREVIEW_RUN_CAP_USD == "0.041751369"
    assert paths.raw_cache.name == "phase1-route-preview-3x7-v3-openrouter"
    assert paths.destination.name == "phase1-route-preview-3x7-v3"
    assert paths.raw_cache.name != "phase1-route-preview-3x7-v1-openrouter"
    assert paths.destination.name != "phase1-route-preview-3x7-v1"


def test_gateway_timeout_with_only_missing_stats_is_a_bounded_preview_failure() -> None:
    missing = RawGenerationStatsResponse(
        404,
        (),
        b'{"error":{"code":404,"message":"Generation gen-fixture not found"}}',
    )
    response = RawHttpResponse(
        200,
        (("x-generation-id", "gen-fixture"),),
        b'{"error":{"code":504,"message":"error code: 524\\n"}}',
        generation_stats_attempts=tuple(
            RawGenerationStatsAttempt(
                attempt_number,
                f"2026-07-19T23:30:4{attempt_number}Z",
                missing,
                None,
            )
            for attempt_number in range(1, GENERATOR_PREVIEW_STATS_ATTEMPTS + 1)
        ),
    )

    assert _bounded_missing_cost_failure_kind(response) == (
        "OpenRouterProviderGatewayTimeout"
    )
    altered = RawHttpResponse(
        200,
        response.headers,
        b'{"error":{"code":504,"message":"another failure"}}',
        generation_stats_attempts=response.generation_stats_attempts,
    )
    assert _bounded_missing_cost_failure_kind(altered) is None

    partial = RawHttpResponse(
        response.status_code,
        response.headers,
        response.body,
        generation_stats_attempts=response.generation_stats_attempts[:-1],
    )
    assert _bounded_missing_cost_failure_kind(partial) is None


def test_preview_promotion_never_replaces_an_existing_empty_directory(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "evidence.txt").write_text("source", encoding="utf-8")
    destination = tmp_path / "destination"
    destination.mkdir()

    with pytest.raises(FileExistsError, match="destination already exists"):
        _promote_directory_no_replace(source, destination)

    assert source.is_dir()
    assert destination.is_dir()
    assert tuple(destination.iterdir()) == ()


def test_complete_cache_client_needs_no_fresh_key_or_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_key_load(*_args: object, **_kwargs: object) -> str:
        pytest.fail("complete-cache recovery loaded a fresh API key")

    evidence = {"decision": "allowed", "source": "cached_fixture"}
    monkeypatch.setattr(
        generator_preview,
        "load_openrouter_api_key",
        unexpected_key_load,
    )
    monkeypatch.setattr(
        generator_preview,
        "_cached_byok_preflight",
        lambda _cache: evidence,
    )

    client, recovered = _preview_client_for_cache(
        GeneratorPreviewPaths.from_repository(tmp_path),
        ImmutableRawCache(tmp_path / "cache"),
        RuntimeCostLedger("0.05"),
        cache_complete=True,
        offline_complete=True,
    )

    assert recovered == evidence
    assert client.require_byok_preflight is False
    with pytest.raises(generator_preview.GeneratorPreviewError, match="HTTP POST"):
        client.transport.post()


def test_stats_only_cache_client_uses_key_without_new_byok_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = {
        "checked_at_utc": "2026-07-19T23:30:42Z",
        "decision": "allowed",
    }
    monkeypatch.setattr(
        generator_preview,
        "load_openrouter_api_key",
        lambda _root: "stats-only-fixture-key",
    )
    monkeypatch.setattr(
        generator_preview,
        "_cached_byok_preflight",
        lambda _cache: evidence,
    )

    client, recovered = _preview_client_for_cache(
        GeneratorPreviewPaths.from_repository(tmp_path),
        ImmutableRawCache(tmp_path / "cache"),
        RuntimeCostLedger("0.05"),
        cache_complete=True,
        offline_complete=False,
    )

    assert recovered == evidence
    assert client.api_key == "stats-only-fixture-key"
    assert client.byok_attestation_path is None
    assert client.require_byok_preflight is False


def test_complete_cache_recovery_accepts_renewed_historical_attestations() -> None:
    older = {
        "checked_at_utc": "2026-07-18T23:30:42Z",
        "decision": "allowed",
    }
    newer = {
        "checked_at_utc": "2026-07-19T23:30:42Z",
        "decision": "allowed",
    }

    def entry(index: int, authorization: dict[str, object]) -> CostJournalEntry:
        return CostJournalEntry(
            request_sha256=f"{index:064x}",
            attempt_number=1,
            upper_bound_usd="0.001",
            charged_usd="0.0001",
            provider_reported_actual=True,
            cancelled_before_post=False,
            byok_authorization=authorization,
            byok_authorization_sha256="a" * 64,
        )

    class RenewedAuthorizationCache:
        def load_cost_journal(self) -> tuple[CostJournalEntry, ...]:
            return tuple(
                entry(index, older if index < 11 else newer)
                for index in range(21)
            )

    assert _cached_byok_preflight(RenewedAuthorizationCache()) == newer


def test_cost_records_are_derived_from_journal_and_reject_tampering(
    tmp_path: Path,
) -> None:
    source = tmp_path / "prompt_metadata_sample.jsonl"
    _write_prompt_fixture(source)
    brief = load_generator_preview_briefs(source)[0]
    route = _routes()[0]
    job = build_generation_jobs(
        (brief,),
        (CANDIDATE_MODELS[0],),
        (route,),
        request_contract=GENERATOR_PREVIEW_REQUEST_CONTRACT,
    )[0]
    bound = request_cost_upper_bound(job.request, job.route).upper_bound_usd
    charged = "0.000001"
    attempt = RawAttempt(
        job.request.request_sha256,
        1,
        "2026-07-19T23:30:42Z",
        route.catalog_sha256,
        RawHttpResponse(200, (), b"{}", billed_cost_usd=charged),
        None,
    )
    entry = CostJournalEntry(
        request_sha256=job.request.request_sha256,
        attempt_number=1,
        upper_bound_usd=bound,
        charged_usd=charged,
        provider_reported_actual=True,
        cancelled_before_post=False,
        byok_authorization={"decision": "allowed"},
        byok_authorization_sha256="a" * 64,
    )

    class CostCache:
        def load_cost_journal(self) -> tuple[CostJournalEntry, ...]:
            return (entry,)

        def load_attempts(self, _request: object) -> tuple[RawAttempt, ...]:
            return (attempt,)

    cache = CostCache()
    jobs = (job,)
    results = ({"billed_cost_usd": charged},)
    runtime = _runtime_cost_record_from_journal(cache, jobs)
    actuals = _cost_actuals(cache, jobs, runtime, (charged,))

    assert (
        _validate_current_cost_records(cache, jobs, results, runtime, actuals)
        == charged
    )
    tampered_runtime = {**runtime, "charged_total_usd": "0"}
    with pytest.raises(
        generator_preview.GeneratorPreviewError,
        match="runtime ledger differs",
    ):
        _validate_current_cost_records(
            cache,
            jobs,
            results,
            tampered_runtime,
            actuals,
        )
    tampered_actuals = {
        **actuals,
        "conservative_unknown_charge_usd": charged,
    }
    with pytest.raises(
        generator_preview.GeneratorPreviewError,
        match="cost actuals differ",
    ):
        _validate_current_cost_records(
            cache,
            jobs,
            results,
            runtime,
            tampered_actuals,
        )


def test_preview_script_import_is_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    class NetworkDeniedSocket(socket.socket):
        def connect(self, address) -> None:
            pytest.fail(f"preview script import attempted network access: {address}")

        def connect_ex(self, address) -> int:
            pytest.fail(f"preview script import attempted network access: {address}")

    monkeypatch.setattr(socket, "socket", NetworkDeniedSocket)

    namespace = runpy.run_path(str(SCRIPT), run_name="offline_generator_preview")

    assert callable(namespace["main"])

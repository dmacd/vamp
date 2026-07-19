from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import runpy
import socket

import pytest

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
    GENERATOR_PREVIEW_VERSION,
    GeneratorPreviewPaths,
    _bounded_missing_cost_failure_kind,
    build_generator_preview_preflight,
    load_generator_preview_briefs,
)
from apm.data.text.tinyworlds_v2.generation_schema import (
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
        generation_stats_attempts=(
            RawGenerationStatsAttempt(
                1,
                "2026-07-19T23:30:42Z",
                missing,
                None,
            ),
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


def test_preview_script_import_is_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    class NetworkDeniedSocket(socket.socket):
        def connect(self, address) -> None:
            pytest.fail(f"preview script import attempted network access: {address}")

        def connect_ex(self, address) -> int:
            pytest.fail(f"preview script import attempted network access: {address}")

    monkeypatch.setattr(socket, "socket", NetworkDeniedSocket)

    namespace = runpy.run_path(str(SCRIPT), run_name="offline_generator_preview")

    assert callable(namespace["main"])

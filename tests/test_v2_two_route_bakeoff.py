from __future__ import annotations

from pathlib import Path

import pytest

from apm.data.text.tinyworlds_v2.bakeoff import (
    SYNTHETIC_STORY_REQUEST_V4,
    TWO_ROUTE_AUTHOR_MODELS,
    NeutralStoryBrief,
)
from apm.data.text.tinyworlds_v2.catalog import CatalogPayloads
from apm.data.text.tinyworlds_v2.generation_schema import RouteLock
from apm.data.text.tinyworlds_v2.json_contracts import (
    canonical_json_bytes,
    canonical_json_loads,
)
from apm.data.text.tinyworlds_v2.phase1_artifacts import Phase1ArtifactBuilder
from apm.data.text.tinyworlds_v2.phase1_generation import build_generation_jobs
from apm.data.text.tinyworlds_v2.quality import TWO_ROUTE_AUTHOR_ORDER
from apm.data.text.tinyworlds_v2.reference_profile import (
    ReferenceObservation,
    build_reference_profile,
)
from apm.data.text.tinyworlds_v2.two_route_bakeoff import (
    TWO_ROUTE_BASE_REFERENCE_MANIFEST_SHA256,
    TWO_ROUTE_REQUEST_COUNT,
    TwoRouteDependencies,
    TwoRoutePaths,
    TwoRouteReferenceEvidence,
    build_two_route_cost_preflight,
    _validate_cost_estimates,
    _validate_planned_jobs,
    prepare_two_route_bakeoff,
    run_two_route_bakeoff,
    validate_two_route_bakeoff,
)


def _brief(index: int) -> NeutralStoryBrief:
    return NeutralStoryBrief(
        brief_id=f"brief-{index:03d}",
        source_record_id=f"source-{index:03d}",
        prompt_text=(
            'Write a simple story using the verb "jump", noun "moon", '
            'and adjective "kind".'
        ),
        required_words=("jump", "moon", "kind"),
        requested_features=(),
        matched_reference_text=(
            "Mia saw the moon. She was kind to a bird and helped it jump home."
        ),
    )


def _observation(index: int) -> ReferenceObservation:
    return ReferenceObservation(
        record_id=f"source-{index:03d}",
        word_tokens=("mia", "saw", "the", "moon", "kind", "jump"),
        model_token_ids=(1, 2, 3, 4, 5, 6),
        sentence_word_counts=(4, 4),
        paragraph_count=1,
        dialogue_present=False,
        opening_key="mia saw the",
        ending_key="it jump home",
        feature_labels=(),
        normalized_nll=1.2,
        required_words=("jump", "moon", "kind"),
        realized_feature_labels=(),
        repeated_ngram_fraction=0.0,
    )


def _references() -> TwoRouteReferenceEvidence:
    observations = tuple(_observation(index) for index in range(TWO_ROUTE_REQUEST_COUNT))
    profile = build_reference_profile(observations)
    return TwoRouteReferenceEvidence(
        briefs=tuple(_brief(index) for index in range(TWO_ROUTE_REQUEST_COUNT)),
        reference_profile=profile,
        paired_observations=observations,
        paired_profile=profile,
        expected_feature_rates=(),
        base_manifest_sha256=TWO_ROUTE_BASE_REFERENCE_MANIFEST_SHA256,
    )


def _catalog(price: str = "0.1") -> tuple[CatalogPayloads, tuple[RouteLock, ...]]:
    model_ids = tuple(model.request_model_id for model in TWO_ROUTE_AUTHOR_MODELS)
    payloads = CatalogPayloads(
        models=b"{}",
        endpoints=tuple((model_id, b"{}") for model_id in model_ids),
        model_plan_ids=model_ids,
    )
    routes = tuple(
        RouteLock(
            route_id=model.route_id,
            catalog_sha256=payloads.snapshot_sha256,
            requested_model=model.request_model_id,
            canonical_model=model.canonical_slug,
            provider_slug="fixture",
            returned_provider="Fixture",
            quantization="bf16",
            input_usd_per_million=price,
            output_usd_per_million=price,
        )
        for model in TWO_ROUTE_AUTHOR_MODELS
    )
    return payloads, routes


def _encode(text: str) -> tuple[int, ...]:
    return tuple(range(max(1, len(text.split()))))


def test_direct_preflight_is_exactly_two_times_200_without_verifier() -> None:
    references = _references()
    _payloads, routes = _catalog()
    jobs = build_generation_jobs(
        references.briefs,
        TWO_ROUTE_AUTHOR_MODELS,
        routes,
        request_contract=SYNTHETIC_STORY_REQUEST_V4,
    )

    preflight = build_two_route_cost_preflight(references, jobs, routes, _encode)

    assert len(jobs) == 400
    assert tuple(item.route_id for item in preflight.route_estimates) == (
        TWO_ROUTE_AUTHOR_ORDER
    )
    assert tuple(item.request_count for item in preflight.route_estimates) == (200, 200)
    assert all(
        "verifier" not in item.workload_label for item in preflight.route_estimates
    )


def test_prepare_persists_v4_direct_contract(tmp_path: Path) -> None:
    references = _references()
    payloads, routes = _catalog()
    dependencies = TwoRouteDependencies(
        fetch_routes=lambda: (payloads, routes),
        encode_text=_encode,
        measure_stories=lambda _stories, _progress: None,  # type: ignore[arg-type]
        load_api_key=lambda: "unused",
        make_client=lambda _key, _cache: None,  # type: ignore[arg-type]
        revalidate_route=lambda route: route,
    )
    builder = Phase1ArtifactBuilder(tmp_path, version="fixture")

    _payloads, _routes, jobs, _preflight = prepare_two_route_bakeoff(
        builder,
        references,
        dependencies,
    )

    assert len(jobs) == 400
    request_path = (
        tmp_path / "routes" / "qwen3.5-35b-a3b" / "requests.jsonl"
    )
    assert request_path.read_bytes().count(b"\n") == 200
    configuration = (tmp_path / "configuration.json").read_text()
    assert "paired-two-route-full-v1" in configuration
    assert SYNTHETIC_STORY_REQUEST_V4.version in configuration
    assert "screen_count" not in configuration
    assert "finalist" not in configuration


def test_direct_preflight_validator_rejects_cross_field_cost_tampering(
    tmp_path: Path,
) -> None:
    references = _references()
    payloads, routes = _catalog()
    dependencies = TwoRouteDependencies(
        fetch_routes=lambda: (payloads, routes),
        encode_text=_encode,
        measure_stories=lambda _stories, _progress: None,  # type: ignore[arg-type]
        load_api_key=lambda: "unused",
        make_client=lambda _key, _cache: None,  # type: ignore[arg-type]
        revalidate_route=lambda route: route,
    )
    builder = Phase1ArtifactBuilder(tmp_path, version="fixture")
    _payloads, resolved, jobs, _preflight = prepare_two_route_bakeoff(
        builder,
        references,
        dependencies,
    )

    _validate_cost_estimates(tmp_path, resolved)
    _validate_planned_jobs(tmp_path, jobs, resolved)

    path = tmp_path / "cost_estimates.json"
    record = canonical_json_loads(path.read_bytes(), label="cost estimates")
    assert type(record) is dict
    record["expected_usd"] = "999"
    path.write_bytes(canonical_json_bytes(record))
    with pytest.raises(ValueError, match="cost preflight"):
        _validate_cost_estimates(tmp_path, resolved)


def test_route_plan_validator_rejects_self_consistent_plan_drift(
    tmp_path: Path,
) -> None:
    references = _references()
    payloads, routes = _catalog()
    dependencies = TwoRouteDependencies(
        fetch_routes=lambda: (payloads, routes),
        encode_text=_encode,
        measure_stories=lambda _stories, _progress: None,  # type: ignore[arg-type]
        load_api_key=lambda: "unused",
        make_client=lambda _key, _cache: None,  # type: ignore[arg-type]
        revalidate_route=lambda route: route,
    )
    builder = Phase1ArtifactBuilder(tmp_path, version="fixture")
    _payloads, resolved, jobs, _preflight = prepare_two_route_bakeoff(
        builder,
        references,
        dependencies,
    )
    path = tmp_path / "routes" / TWO_ROUTE_AUTHOR_ORDER[0] / "plan.json"
    record = canonical_json_loads(path.read_bytes(), label="route plan")
    assert type(record) is dict
    record["validator_version"] = "tampered-validator"
    path.write_bytes(canonical_json_bytes(record))

    with pytest.raises(ValueError, match="plan differs"):
        _validate_planned_jobs(tmp_path, jobs, resolved)


def test_cost_cap_stop_never_reads_key_and_strictly_validates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    references = _references()
    payloads, routes = _catalog("1000")
    staging = tmp_path / "staging"
    staging.mkdir()
    paths = TwoRoutePaths(
        repository_root=tmp_path,
        base_reference=tmp_path / "base",
        raw_cache=tmp_path / "cache",
        destination=tmp_path / "result",
    )
    key_reads = 0

    def load_key() -> str:
        nonlocal key_reads
        key_reads += 1
        return "must-not-be-read"

    dependencies = TwoRouteDependencies(
        fetch_routes=lambda: (payloads, routes),
        encode_text=_encode,
        measure_stories=lambda _stories, _progress: None,  # type: ignore[arg-type]
        load_api_key=load_key,
        make_client=lambda _key, _cache: None,  # type: ignore[arg-type]
        revalidate_route=lambda route: route,
    )
    monkeypatch.setattr(
        "apm.data.text.tinyworlds_v2.two_route_bakeoff.load_two_route_reference_evidence",
        lambda _path: references,
    )

    result = run_two_route_bakeoff(
        staging,
        paths,
        dependencies,
        emit=lambda _message: None,
    )

    assert result.status == "blocked_by_cost_cap"
    assert key_reads == 0
    assert validate_two_route_bakeoff(result.directory).manifest_sha256

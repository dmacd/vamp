from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from functools import lru_cache
from hashlib import sha256
import json
import os
from pathlib import Path
import runpy
import shutil
import socket
from threading import Lock
from types import SimpleNamespace

import pytest

from apm.data.text.tinyworlds_v2.audit_io import (
    validate_phase1_reference,
    validate_phase1_semantics,
)
from apm.data.text.tinyworlds_v2.bakeoff import CANDIDATE_MODELS, VERIFIER_MODEL, NeutralStoryBrief
from apm.data.text.tinyworlds_v2.catalog import CatalogPayloads, ResolvedRouteCatalog
from apm.data.text.tinyworlds_v2.generation_schema import (
    CanonicalRequest,
    RawAttempt,
    RawHttpResponse,
    ResponseProvenance,
    RouteLock,
    TokenUsage,
)
from apm.data.text.tinyworlds_v2.generation_cache import ImmutableRawCache
from apm.data.text.tinyworlds_v2.generation_costs import (
    CostCapExceeded,
    CostJournalRecoveryRequired,
    RuntimeCostLedger,
    request_cost_upper_bound,
)
from apm.data.text.tinyworlds_v2.openrouter import (
    OpenRouterClient,
    RetryPolicy,
    TransportResponse,
)
from apm.data.text.tinyworlds_v2.phase1_artifacts import (
    Phase1ArtifactBuilder,
    canonical_jsonl_bytes,
    load_phase1_artifact_tree,
)
from apm.data.text.tinyworlds_v2.phase1_runner import (
    CatalogEvidence,
    MeasurementBatch,
    Phase1Dependencies,
    Phase1Paths,
    Phase1ReferenceCorpus,
    StoryMeasurement,
    _attributed_full_route_costs,
    _reference_profile_record,
    run_phase1,
)
from apm.data.text.tinyworlds_v2.phase1_generation import (
    build_generation_jobs,
    build_verifier_job,
    execute_generation_jobs,
)
from apm.data.text.tinyworlds_v2.phase1_replay import (
    Phase1ReplayError,
    verify_phase1_derived_replay,
)
from apm.data.text.tinyworlds_v2.reference_pipeline import (
    PHASE1_ARCHIVE_REFERENCE_COUNT,
    PHASE1_PROMPT_METADATA_COUNT,
    PHASE1_VALIDATION_REFERENCE_COUNT,
    ReferenceAnnotation,
    build_prompt_ingredient_profile,
    canonical_neutral_story_brief,
    canonical_prompt_ingredient_profile,
    canonical_reference_annotation,
    canonical_reference_observation,
    canonical_reference_record,
)
from apm.data.text.tinyworlds_v2.reference_profile import (
    ReferenceRecord,
    build_reference_profile,
    observe_reference,
)
from apm.data.text.tinyworlds_v2.reference_runtime import NllStory
from apm.data.text.tinyworlds_v2.source_data import (
    TINYSTORIES_ALL_DATA_SOURCE,
    ArchiveSourceRecord,
    TinyStoriesInstruction,
    ValidationStoryRecord,
    canonical_prompt_metadata_record,
    canonical_validation_record,
)
from apm.data.text.curricula import TINYSTORIES_V2_SOURCE


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "generate_tinyworlds_v2_references.py"


def _story() -> str:
    return (
        "Mia saw the moon above her little home. She was kind to a wet bird "
        "beside a tree. Mia asked the bird to jump onto her hand, but it was "
        "tired. She gave it water and waited quietly. Soon the bird could jump "
        "and fly again. Mia smiled and walked home under the bright moon."
    )


def _hardlink_tree(source: Path, destination: Path) -> None:
    """Clone immutable fixture evidence cheaply; callers unlink before edits."""
    shutil.copytree(source, destination, copy_function=os.link)


def _token_ids(text: str) -> tuple[int, ...]:
    return tuple(ord(character) % 251 for character in text) + (1,)


def _reference_corpus(*, full_source_contract: bool = False) -> Phase1ReferenceCorpus:
    if full_source_contract:
        return _full_reference_corpus()
    story = _story()
    briefs = tuple(
        NeutralStoryBrief(
            brief_id=f"brief-{index:03d}",
            source_record_id=f"paired-{index:03d}",
            prompt_text=(
                "Write a simple story. The verb is 'jump', the noun is 'moon', "
                "and the adjective is 'kind'."
            ),
            required_words=("jump", "moon", "kind"),
            requested_features=(),
            matched_reference_text=story,
        )
        for index in range(200)
    )
    reference_records = tuple(
        ReferenceRecord(f"reference-{index:03d}", story, source_model="GPT-4")
        for index in range(200)
    )
    reference_observations = tuple(
        observe_reference(
            record,
            model_token_ids=_token_ids(story),
            normalized_nll=2.0,
        )
        for record in reference_records
    )
    paired_observations = tuple(
        observe_reference(
            ReferenceRecord(brief.source_record_id, story, brief.prompt_text),
            model_token_ids=_token_ids(story),
            normalized_nll=2.0,
            required_words=brief.required_words,
        )
        for brief in briefs
    )
    reference_profile = build_reference_profile(reference_observations)
    paired_reference_profile = build_reference_profile(paired_observations)
    source_manifest = {"fixture": "phase1-offline"}
    brief_records = tuple(canonical_neutral_story_brief(item) for item in briefs)
    paired_records = tuple(
        canonical_reference_observation(item) for item in paired_observations
    )
    source_artifacts = tuple(sorted((
        ("neutral_story_briefs.jsonl", brief_records),
        ("paired_reference_observations.jsonl", paired_records),
        ("fixture_sources.jsonl", ({"fixture": True},)),
    )))
    return Phase1ReferenceCorpus(
        briefs=briefs,
        reference_records=reference_records,
        reference_observations=reference_observations,
        paired_reference_observations=paired_observations,
        reference_profile=reference_profile,
        paired_reference_profile=paired_reference_profile,
        expected_feature_rates=(),
        source_manifest=source_manifest,
        source_artifacts=source_artifacts,
        reference_statistics={"fixture": "matched-reference"},
    )


@lru_cache(maxsize=1)
def _full_reference_corpus() -> Phase1ReferenceCorpus:
    """Return one production-shaped source fixture shared by runner tests."""
    prompt_records = tuple(
        _archive_source_record("prompt", index, _unique_story("prompt", index))
        for index in range(PHASE1_PROMPT_METADATA_COUNT)
    )
    paired_sources = tuple(
        _archive_source_record("paired", index, _unique_story("paired", index))
        for index in range(200)
    )
    briefs = tuple(
        sorted(
            (
                NeutralStoryBrief(
                    brief_id="brief-"
                    + sha256(
                        f"tinyworlds-v2-phase1-brief\0{record.record_id}".encode()
                    ).hexdigest()[:24],
                    source_record_id=record.record_id,
                    prompt_text=record.instruction.prompt,
                    required_words=record.instruction.words,
                    requested_features=(),
                    matched_reference_text=record.story,
                )
                for record in paired_sources
            ),
            key=lambda item: item.brief_id,
        )
    )
    archive_sources = tuple(
        _archive_source_record("reference", index, _unique_story("archive", index))
        for index in range(PHASE1_ARCHIVE_REFERENCE_COUNT)
    )
    validation_sources = tuple(
        _validation_source_record(index, _unique_story("validation", index))
        for index in range(PHASE1_VALIDATION_REFERENCE_COUNT)
    )
    reference_pairs = tuple(
        sorted(
            (
                *(
                    (
                        ReferenceRecord(
                            record.record_id,
                            record.story,
                            prompt_text=record.instruction.prompt,
                            source_model="GPT-4",
                        ),
                        ReferenceAnnotation(
                            record.record_id,
                            "archive",
                            record.instruction.words,
                            record.instruction.features,
                        ),
                    )
                    for record in archive_sources
                ),
                *(
                    (
                        ReferenceRecord(
                            record.record_id,
                            record.story,
                            source_model="GPT-4",
                        ),
                        ReferenceAnnotation(record.record_id, "validation", (), ()),
                    )
                    for record in validation_sources
                ),
            ),
            key=lambda item: item[0].record_id,
        )
    )
    reference_records = tuple(item[0] for item in reference_pairs)
    annotations = tuple(item[1] for item in reference_pairs)
    reference_observations = tuple(
        observe_reference(
            record,
            model_token_ids=_token_ids(record.story_text),
            normalized_nll=2.0,
            feature_labels=annotation.feature_labels,
            required_words=annotation.required_words,
        )
        for record, annotation in reference_pairs
    )
    paired_observations = tuple(
        observe_reference(
            ReferenceRecord(
                brief.source_record_id,
                brief.matched_reference_text,
                brief.prompt_text,
                "GPT-4",
            ),
            model_token_ids=_token_ids(brief.matched_reference_text),
            normalized_nll=2.0,
            feature_labels=brief.requested_features,
            required_words=brief.required_words,
        )
        for brief in briefs
    )
    reference_profile = build_reference_profile(reference_observations)
    paired_reference_profile = build_reference_profile(paired_observations)
    reference_count = PHASE1_ARCHIVE_REFERENCE_COUNT + PHASE1_VALIDATION_REFERENCE_COUNT
    source_manifest = {
        "archive": {
            "dataset_id": TINYSTORIES_ALL_DATA_SOURCE.dataset_id,
            "filename": TINYSTORIES_ALL_DATA_SOURCE.archive_file.filename,
            "revision": TINYSTORIES_ALL_DATA_SOURCE.revision,
            "sha256": TINYSTORIES_ALL_DATA_SOURCE.archive_file.sha256,
            "size_bytes": TINYSTORIES_ALL_DATA_SOURCE.archive_file.size_bytes,
        },
        "counts": {
            "archive_reference": PHASE1_ARCHIVE_REFERENCE_COUNT,
            "neutral_briefs": 200,
            "prompt_metadata": PHASE1_PROMPT_METADATA_COUNT,
            "reference_profile": reference_count,
            "validation_reference": PHASE1_VALIDATION_REFERENCE_COUNT,
        },
        "selection_seed": "tinyworlds-v2-phase1-reference-v1",
        "story_identity_policy": "unicode-nfkc-casefold-whitespace-collapse-sha256-v1",
        "validation": {
            "dataset_id": TINYSTORIES_V2_SOURCE.dataset_id,
            "document_separator": TINYSTORIES_V2_SOURCE.document_separator,
            "filename": TINYSTORIES_V2_SOURCE.validation_file.filename,
            "revision": TINYSTORIES_V2_SOURCE.revision,
            "sha256": TINYSTORIES_V2_SOURCE.validation_file.sha256,
            "size_bytes": TINYSTORIES_V2_SOURCE.validation_file.size_bytes,
        },
    }
    source_artifacts = tuple(
        sorted(
            (
                (
                    "neutral_story_briefs.jsonl",
                    tuple(canonical_neutral_story_brief(item) for item in briefs),
                ),
                (
                    "paired_reference_observations.jsonl",
                    tuple(
                        canonical_reference_observation(item)
                        for item in paired_observations
                    ),
                ),
                (
                    "prompt_metadata_sample.jsonl",
                    tuple(canonical_prompt_metadata_record(item) for item in prompt_records),
                ),
                (
                    "reference_annotations.jsonl",
                    tuple(canonical_reference_annotation(item) for item in annotations),
                ),
                (
                    "reference_observations.jsonl",
                    tuple(
                        canonical_reference_observation(item)
                        for item in reference_observations
                    ),
                ),
                (
                    "reference_story_sample.jsonl",
                    tuple(canonical_reference_record(item) for item in reference_records),
                ),
                (
                    "validation_source_sample.jsonl",
                    tuple(canonical_validation_record(item) for item in validation_sources),
                ),
            )
        )
    )
    reference_statistics = {
        "ingredient_profile": canonical_prompt_ingredient_profile(
            build_prompt_ingredient_profile(prompt_records)
        ),
        "nll_runtime": {"fixture": "deterministic"},
        "paired_reference_profile": _reference_profile_record(
            paired_reference_profile
        ),
        "paired_source_record_ids": [brief.source_record_id for brief in briefs],
        "reference_profile": _reference_profile_record(reference_profile),
    }
    return Phase1ReferenceCorpus(
        briefs=briefs,
        reference_records=reference_records,
        reference_observations=reference_observations,
        paired_reference_observations=paired_observations,
        reference_profile=reference_profile,
        paired_reference_profile=paired_reference_profile,
        expected_feature_rates=(),
        source_manifest=source_manifest,
        source_artifacts=source_artifacts,
        reference_statistics=reference_statistics,
    )


def _archive_source_record(
    namespace: str,
    index: int,
    story: str,
) -> ArchiveSourceRecord:
    prompt = (
        "Write a simple story. The verb is 'jump', the noun is 'moon', "
        "and the adjective is 'kind'."
    )
    words = ("jump", "moon", "kind")
    features: tuple[str, ...] = ()
    summary = ""
    content_sha256 = sha256(
        json.dumps(
            {
                "instruction": {
                    "features": list(features),
                    "prompt:": prompt,
                    "words": list(words),
                },
                "source": "GPT-4",
                "story": story,
                "summary": summary,
            },
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    source_member = f"fixture-{namespace}.json"
    return ArchiveSourceRecord(
        record_id=f"archive:{source_member}:{index}:{content_sha256}",
        source_member=source_member,
        source_index=index,
        content_sha256=content_sha256,
        story=story,
        instruction=TinyStoriesInstruction(
            prompt=prompt,
            words=words,
            features=features,
        ),
        summary=summary,
        source="GPT-4",
    )


def _validation_source_record(index: int, story: str) -> ValidationStoryRecord:
    content_sha256 = sha256(story.encode()).hexdigest()
    return ValidationStoryRecord(
        record_id=f"v2-validation:{index}:{content_sha256}",
        source_index=index,
        content_sha256=content_sha256,
        story=story,
    )


def _unique_story(namespace: str, index: int) -> str:
    letters = "a" if index == 0 else ""
    value = index
    while value:
        value, remainder = divmod(value, 26)
        letters = chr(ord("a") + remainder) + letters
    return f"{_story()} This {namespace} tale has code {letters}."


def _catalog(price: str) -> CatalogEvidence:
    endpoint_payloads = tuple(
        (spec.request_model_id, b"{}")
        for spec in (*CANDIDATE_MODELS, VERIFIER_MODEL)
    )
    payloads = CatalogPayloads(models=b"{}", endpoints=endpoint_payloads)

    def route(spec) -> RouteLock:
        return RouteLock(
            route_id=spec.route_id,
            catalog_sha256=payloads.snapshot_sha256,
            requested_model=spec.request_model_id,
            canonical_model=spec.canonical_slug,
            provider_slug="fixture-provider",
            returned_provider="Fixture Provider",
            quantization="bf16",
            input_usd_per_million=price,
            output_usd_per_million=price,
        )

    resolved = ResolvedRouteCatalog(
        snapshot_sha256=payloads.snapshot_sha256,
        generator_routes=tuple(route(spec) for spec in CANDIDATE_MODELS),
        verifier_route=route(VERIFIER_MODEL),
    )
    return CatalogEvidence(payloads, resolved)


def _byok_authorization() -> dict[str, object]:
    return {
        "attestation_sha256": None,
        "attested_at_utc": None,
        "checked_at_utc": "2026-07-18T00:00:00Z",
        "decision": "allowed",
        "endpoint": "/api/v1/byok",
        "expires_at_utc": None,
        "method": "GET",
        "response_body_sha256": "a" * 64,
        "source": "management_api",
        "status_code": 200,
        "total_count": 0,
    }


class _FixtureTransport:
    def __init__(
        self,
        byok_response: TransportResponse | None = None,
    ) -> None:
        self.remote_calls = 0
        self.byok_response = byok_response or TransportResponse(
            status_code=200,
            headers=(("content-type", "application/json"),),
            body=b'{"data":[],"total_count":0}',
        )

    def post(self, *, url, headers, body, timeout_seconds) -> TransportResponse:
        self.remote_calls += 1
        request_body = json.loads(body)
        request_model = request_body["model"]
        specs = (*CANDIDATE_MODELS, VERIFIER_MODEL)
        spec = next(item for item in specs if item.request_model_id == request_model)
        if spec.route_id == VERIFIER_MODEL.route_id:
            content = json.dumps(
                {
                    "brief_adherence": True,
                    "grammar": 5,
                    "hard_failures": [],
                    "non_repetition": 5,
                    "plot_coherence": 5,
                    "preschool_vocabulary": 5,
                    "rationale": "Simple, coherent, and appropriate.",
                    "sentence_simplicity": 5,
                },
                separators=(",", ":"),
            )
        else:
            content = json.dumps(
                {
                    "feature_evidence": [],
                    "story": _story(),
                    "word_evidence": [
                        {"exact_quote": word, "required_word": word}
                        for word in ("jump", "moon", "kind")
                    ],
                },
                separators=(",", ":"),
            )
        generation_id = "fixture-" + sha256(body).hexdigest()[:24]
        response_body = json.dumps(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": content, "role": "assistant"},
                    }
                ],
                "id": generation_id,
                "model": spec.canonical_slug,
                "openrouter_metadata": {
                    "is_byok": False,
                    "endpoints": {
                        "available": [
                            {
                                "provider": "Fixture Provider",
                                "selected": True,
                            }
                        ]
                    }
                },
                "usage": {
                    "completion_tokens": 30,
                    "cost": 0.000001,
                    "prompt_tokens": 20,
                    "total_tokens": 50,
                },
            },
            separators=(",", ":"),
        ).encode()
        return TransportResponse(
            status_code=200,
            headers=(("content-type", "application/json"),),
            body=response_body,
        )

    def get_authenticated(self, *, url, headers, timeout_seconds) -> TransportResponse:
        assert url.endswith("/api/v1/byok")
        return self.byok_response


class _ByokFailureTransport(_FixtureTransport):
    """Return one persisted BYOK completion amid otherwise valid concurrency."""

    def __init__(self) -> None:
        super().__init__()
        self._failure_lock = Lock()
        self._failure_sent = False

    def post(self, *, url, headers, body, timeout_seconds) -> TransportResponse:
        response = super().post(
            url=url,
            headers=headers,
            body=body,
            timeout_seconds=timeout_seconds,
        )
        with self._failure_lock:
            if self._failure_sent:
                fail = False
            else:
                self._failure_sent = True
                fail = True
        if not fail:
            return response
        record = json.loads(response.body)
        record["openrouter_metadata"]["is_byok"] = True
        return replace(
            response,
            body=json.dumps(
                record,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode(),
        )


def _measure(stories: tuple[NllStory, ...], _progress) -> MeasurementBatch:
    return MeasurementBatch(
        measurements=tuple(
            StoryMeasurement(
                item.record_id,
                _token_ids(item.text),
                2.0,
                len(_token_ids(item.text)) - 1,
            )
            for item in sorted(stories, key=lambda value: value.record_id)
        ),
        runtime={"fixture": "deterministic"},
    )


def _paths(tmp_path: Path, destination: str) -> Phase1Paths:
    return Phase1Paths(
        repository_root=tmp_path,
        archive=tmp_path / "archive",
        validation=tmp_path / "validation",
        checkpoint=tmp_path / "checkpoint",
        tokenizer=tmp_path / "tokenizer",
        raw_cache=tmp_path / "raw-cache",
        destination=tmp_path / destination,
    )


def test_generation_script_import_is_offline_and_does_not_start_work(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class NetworkDeniedSocket(socket.socket):
        def connect(self, address) -> None:
            pytest.fail(f"script import attempted network access: {address}")

        def connect_ex(self, address) -> int:
            pytest.fail(f"script import attempted network access: {address}")

    monkeypatch.setattr(socket, "socket", NetworkDeniedSocket)
    monkeypatch.chdir(tmp_path)

    namespace = runpy.run_path(str(SCRIPT), run_name="offline_v2_phase1_surface")

    assert callable(namespace["main"])
    assert not tuple(tmp_path.iterdir())


def test_cost_cap_is_enforced_before_key_load_or_generation(tmp_path: Path) -> None:
    calls: list[str] = []
    dependencies = Phase1Dependencies(
        prepare_references=lambda _progress: _reference_corpus(full_source_contract=True),
        fetch_catalog=lambda: _catalog("10000"),
        encode_text=_token_ids,
        measure_stories=_measure,
        load_api_key=lambda: calls.append("key") or "secret",
        make_client=lambda _key, _cache: calls.append("client"),
    )
    staging = tmp_path / "staging"
    staging.mkdir()

    result = run_phase1(staging, _paths(tmp_path, "blocked"), dependencies, emit=lambda _: None)

    assert result.status == "blocked_by_cost_cap"
    assert calls == []
    manifest = load_phase1_artifact_tree(result.directory)
    assert manifest.artifacts
    status = json.loads((result.directory / "status.json").read_bytes())
    assert status["status"] == "blocked_by_cost_cap"
    with pytest.raises(ValueError):
        validate_phase1_reference(result.directory)


def test_preflight_reserves_the_actual_seven_route_funnel_not_seven_finalists() -> None:
    import apm.data.text.tinyworlds_v2.phase1_runner as runner

    references = _reference_corpus()
    catalog = _catalog("1.0").resolved
    jobs = build_generation_jobs(
        references.briefs,
        CANDIDATE_MODELS,
        catalog.generator_routes,
    )

    preflight = runner._build_phase1_cost_preflight(
        references,
        jobs,
        catalog,
        _token_ids,
    )

    generator_counts = tuple(
        estimate.request_count for estimate in preflight.route_estimates[:-1]
    )
    assert sorted(generator_counts) == [50, 50, 50, 50, 200, 200, 200]
    assert sum(generator_counts) == 800
    matched_tokens = len(_token_ids(references.briefs[0].matched_reference_text))
    for estimate in preflight.route_estimates[:-1]:
        assert estimate.expected_output_tokens == estimate.request_count * matched_tokens
        assert estimate.conservative_output_tokens == estimate.request_count * 512
    assert preflight.route_estimates[-1].route_id == VERIFIER_MODEL.route_id
    assert preflight.route_estimates[-1].request_count == 800


@pytest.mark.parametrize(
    ("response", "decision"),
    (
        (
            TransportResponse(
                200,
                (),
                b'{"data":[{"id":"not-persisted"}],"total_count":1}',
            ),
            "blocked",
        ),
        (TransportResponse(403, (), b'{"error":"forbidden"}'), "unverified"),
    ),
)
def test_byok_preflight_stop_is_valid_without_a_completion_post(
    tmp_path: Path,
    response: TransportResponse,
    decision: str,
) -> None:
    transport = _FixtureTransport(response)
    dependencies = Phase1Dependencies(
        prepare_references=lambda _progress: _reference_corpus(
            full_source_contract=True
        ),
        fetch_catalog=lambda: _catalog("0.001"),
        encode_text=_token_ids,
        measure_stories=_measure,
        load_api_key=lambda: "secret",
        make_client=lambda key, cache: OpenRouterClient(
            api_key=key,
            management_api_key="management-secret",
            transport=transport,
            cache=cache,
            require_byok_preflight=True,
            clock=lambda: "2026-07-18T00:00:00Z",
        ),
    )
    staging = tmp_path / f"byok-{decision}-staging"
    staging.mkdir()

    result = run_phase1(
        staging,
        _paths(tmp_path, f"byok-{decision}-reference"),
        dependencies,
        emit=lambda _: None,
    )

    assert result.status == "provider_billing_unknown"
    assert transport.remote_calls == 0
    evidence = json.loads((result.directory / "byok_preflight.json").read_bytes())
    assert evidence["decision"] == decision
    assert "not-persisted" not in str(evidence)
    assert validate_phase1_semantics(result.directory).status == result.status


def test_failed_resume_preflight_accepts_historical_paid_cache_authorization(
    tmp_path: Path,
) -> None:
    references = _reference_corpus(full_source_contract=True)
    catalog = _catalog("0.001")
    paths = _paths(tmp_path, "failed-resume-reference")
    cache = ImmutableRawCache(paths.raw_cache)
    prior_transport = _FixtureTransport()
    prior_client = OpenRouterClient(
        api_key="secret",
        management_api_key="management-secret",
        transport=prior_transport,
        cache=cache,
        require_byok_preflight=True,
        clock=lambda: "2026-07-17T00:00:00Z",
    )
    prior_client.verify_no_byok()
    prior_job = build_generation_jobs(
        (references.briefs[0],),
        (CANDIDATE_MODELS[0],),
        (catalog.resolved.generator_routes[0],),
    )[0]
    prior_client.generate(prior_job.request, prior_job.route)
    assert prior_transport.remote_calls == 1

    current_transport = _FixtureTransport(
        TransportResponse(403, (), b'{"error":"unverified"}')
    )
    dependencies = Phase1Dependencies(
        prepare_references=lambda _progress: references,
        fetch_catalog=lambda: catalog,
        encode_text=_token_ids,
        measure_stories=_measure,
        load_api_key=lambda: "secret",
        make_client=lambda key, current_cache: OpenRouterClient(
            api_key=key,
            management_api_key="management-secret",
            transport=current_transport,
            cache=current_cache,
            require_byok_preflight=True,
            clock=lambda: "2026-07-18T00:00:00Z",
        ),
    )
    staging = tmp_path / "failed-resume-staging"
    staging.mkdir()

    result = run_phase1(staging, paths, dependencies, emit=lambda _: None)

    assert result.status == "provider_billing_unknown"
    assert current_transport.remote_calls == 0
    current = json.loads((result.directory / "byok_preflight.json").read_bytes())
    assert current["decision"] == "unverified"
    reservation_path = next(
        (result.directory / "raw_cache" / "runtime-cost-journal").glob(
            "*/reservation.json"
        )
    )
    historical = json.loads(reservation_path.read_bytes())
    assert historical["byok_authorization"]["decision"] == "allowed"
    assert historical["byok_authorization"]["checked_at_utc"].startswith(
        "2026-07-17"
    )
    assert validate_phase1_semantics(result.directory).status == result.status


def test_attempt_billing_includes_billed_retryable_failure(tmp_path: Path) -> None:
    import apm.data.text.tinyworlds_v2.phase1_runner as runner

    model = CANDIDATE_MODELS[0]
    route = _catalog("0.001").resolved.generator_routes[0]
    job = build_generation_jobs(
        (_reference_corpus().briefs[0],),
        (model,),
        (route,),
    )[0]
    cache = ImmutableRawCache(tmp_path / "attempt-cache")
    for number, status, cost in ((1, 500, "0.20"), (2, 200, "0.30")):
        response = RawHttpResponse(
            status_code=status,
            headers=(),
            body=f"attempt-{number}".encode(),
            provenance=(
                None
                if status == 500
                else ResponseProvenance(
                    "generation",
                    route.requested_model,
                    route.canonical_model,
                    route.returned_provider,
                )
            ),
            usage=TokenUsage(10, 5, 15, 0),
            billed_cost_usd=cost,
        )
        cache.store_attempt(
            job.request,
            job.route,
            RawAttempt(
                request_sha256=job.request.request_sha256,
                attempt_number=number,
                observed_at_utc=f"2026-07-18T00:00:0{number}Z",
                submission_catalog_sha256=job.route.catalog_sha256,
                response=response,
                transport_error_type=None,
            ),
        )

    billing = runner._cached_attempt_billing(cache, (job,))

    assert format(dict(billing.generation_by_route)[model.route_id], "f") == "0.50"
    assert format(billing.actual_billed_usd, "f") == "0.50"


def test_stopped_billing_uses_persisted_historical_route_lock(
    tmp_path: Path,
) -> None:
    import apm.data.text.tinyworlds_v2.phase1_runner as runner

    historical = _catalog("0.002").resolved
    current = _catalog("0.001").resolved
    job = build_generation_jobs(
        (_reference_corpus().briefs[0],),
        (CANDIDATE_MODELS[0],),
        (historical.generator_routes[0],),
    )[0]
    cache = ImmutableRawCache(tmp_path / "historical-settlement-cache")
    cache.prepare_request(job.request, job.route)
    bound = request_cost_upper_bound(job.request, job.route)
    cache.store_cost_reservation(
        job.request.request_sha256,
        1,
        bound.upper_bound_usd,
        _byok_authorization(),
    )
    actual = format(Decimal(bound.upper_bound_usd) / 2, "f")
    cache.store_cost_settlement(
        job.request.request_sha256,
        1,
        charged_usd=actual,
        provider_reported_actual=True,
    )

    billing = runner._cached_attempt_billing(
        cache,
        (),
        route_locks=(*current.generator_routes, current.verifier_route),
        include_all_cost_evidence=True,
    )

    assert billing.generation_billed_usd == Decimal(actual)
    assert billing.verification_billed_usd == Decimal(0)
    assert billing.route_by_request == (
        (job.request.request_sha256, CANDIDATE_MODELS[0].route_id),
    )


def test_runtime_stop_separates_verifier_journal_only_actual_from_generation(
    tmp_path: Path,
) -> None:
    import apm.data.text.tinyworlds_v2.phase1_runner as runner

    catalog = _catalog("0.001").resolved
    brief = _reference_corpus().briefs[0]
    generation_job = build_generation_jobs(
        (brief,),
        (CANDIDATE_MODELS[0],),
        (catalog.generator_routes[0],),
    )[0]
    verifier_job = build_verifier_job(
        source_id="generated-source",
        pair_id=brief.brief_id,
        brief=brief,
        story=_story(),
        model=VERIFIER_MODEL,
        route=catalog.verifier_route,
    )
    cache = ImmutableRawCache(tmp_path / "stopped-attribution-cache")

    costs: dict[str, str] = {}
    for job in (generation_job, verifier_job):
        cache.prepare_request(job.request, job.route)
        bound = request_cost_upper_bound(job.request, job.route)
        actual = format(Decimal(bound.upper_bound_usd) / 2, "f")
        costs[job.request.request_sha256] = actual
        cache.store_cost_reservation(
            job.request.request_sha256,
            1,
            bound.upper_bound_usd,
            _byok_authorization(),
        )
        cache.store_cost_settlement(
            job.request.request_sha256,
            1,
            charged_usd=actual,
            provider_reported_actual=True,
        )
    cache.store_attempt(
        generation_job.request,
        generation_job.route,
        RawAttempt(
            request_sha256=generation_job.request.request_sha256,
            attempt_number=1,
            observed_at_utc="2026-07-18T00:00:00Z",
            submission_catalog_sha256=generation_job.route.catalog_sha256,
            response=RawHttpResponse(
                status_code=200,
                headers=(),
                body=b"{}",
                billed_cost_usd=costs[generation_job.request.request_sha256],
            ),
            transport_error_type=None,
        ),
    )
    routes = (*catalog.generator_routes, catalog.verifier_route)
    ledger = RuntimeCostLedger()
    with pytest.raises(CostJournalRecoveryRequired):
        ledger.bootstrap(cache, routes)
    attempted = runner._attempted_jobs(
        cache,
        (generation_job, verifier_job),
    )
    staging = tmp_path / "stopped-attribution-staging"
    staging.mkdir()
    builder = Phase1ArtifactBuilder(staging)

    runner._write_runtime_cost_stop(
        builder,
        cache,
        attempted,
        ledger,
        routes,
    )

    actuals = json.loads((staging / "cost_actuals.json").read_bytes())
    observations = json.loads(
        (staging / "cost_observations.json").read_bytes()
    )
    generation_cost = Decimal(costs[generation_job.request.request_sha256])
    verifier_cost = Decimal(costs[verifier_job.request.request_sha256])
    assert Decimal(str(actuals["generation_billed_usd"])) == generation_cost
    assert Decimal(str(actuals["verification_billed_usd"])) == verifier_cost
    assert Decimal(str(actuals["actual_billed_usd"])) == (
        generation_cost + verifier_cost
    )
    route_actuals = {
        item["route_id"]: Decimal(str(item["actual_billed_usd"]))
        for item in actuals["routes"]
    }
    assert route_actuals[generation_job.route.route_id] == generation_cost
    assert observations["billed_verifier_usd"] == format(verifier_cost, "f")
    verifier_record = runner._execution_request_record(
        verifier_job.request,
        cache,
        None,
        {
            "outcome": "interrupted",
            "pair_id": verifier_job.pair_id,
            "source_id": verifier_job.source_id,
        },
    )
    assert verifier_record["billed_cost_usd"] == format(verifier_cost, "f")


def test_execution_manifest_rejects_conflicting_duplicate_sample_records(
    tmp_path: Path,
) -> None:
    import apm.data.text.tinyworlds_v2.phase1_runner as runner

    model = CANDIDATE_MODELS[0]
    catalog = _catalog("0.001").resolved
    job = build_generation_jobs(
        (_reference_corpus().briefs[0],),
        (model,),
        (catalog.generator_routes[0],),
    )[0]
    cache = ImmutableRawCache(tmp_path / "duplicate-cache")
    client = OpenRouterClient(
        api_key="secret",
        transport=_FixtureTransport(),
        cache=cache,
        retry_policy=RetryPolicy(max_attempts=1),
        sleeper=lambda _seconds: None,
        clock=lambda: "2026-07-18T00:00:00Z",
    )
    client.cost_ledger.authorize_byok(_byok_authorization())
    sample = execute_generation_jobs((job,), client, max_workers=1)[0]
    staging = tmp_path / "duplicate-staging"
    staging.mkdir()
    builder = Phase1ArtifactBuilder(staging)

    with pytest.raises(ValueError, match="duplicate generated sample records differ"):
        runner._write_execution_manifests(
            builder,
            cache,
            (job,),
            (sample, replace(sample, input_tokens=sample.input_tokens + 1)),
            (),
            catalog.verifier_route,
            attempted_jobs=(job,),
        )


def test_raw_artifact_preserves_orphaned_cost_reservation_request(
    tmp_path: Path,
) -> None:
    import apm.data.text.tinyworlds_v2.phase1_replay as replay
    import apm.data.text.tinyworlds_v2.phase1_runner as runner

    catalog = _catalog("0.001").resolved
    job = build_generation_jobs(
        (_reference_corpus().briefs[0],),
        (CANDIDATE_MODELS[0],),
        (catalog.generator_routes[0],),
    )[0]
    cache = ImmutableRawCache(tmp_path / "orphan-cache")
    cache.prepare_request(job.request, job.route)
    bound = request_cost_upper_bound(job.request, job.route)
    cache.store_cost_reservation(
        job.request.request_sha256,
        1,
        bound.upper_bound_usd,
        _byok_authorization(),
    )
    # A conservative/unknown settlement proves that a POST may have run; it
    # does not make the missing immutable raw attempt safe to ignore.
    cache.store_cost_settlement(
        job.request.request_sha256,
        1,
        charged_usd=bound.upper_bound_usd,
        provider_reported_actual=False,
    )
    staging = tmp_path / "orphan-staging"
    staging.mkdir()
    builder = Phase1ArtifactBuilder(staging)

    runner._copy_raw_cache(builder, cache, ())

    assert (
        staging
        / "raw_cache"
        / "requests"
        / job.request.request_sha256
        / "request.json"
    ).is_file()
    assert tuple(
        (staging / "raw_cache" / "runtime-cost-journal").rglob(
            "reservation.json"
        )
    )
    materialized = replay._ReplayRawCache(staging / "raw_cache")
    try:
        assert replay._validate_raw_cache_journal(
            materialized.cache,
            (*catalog.generator_routes, catalog.verifier_route),
        ) == "orphaned_cost_reservation"
    finally:
        materialized.close()


def test_cancelled_before_post_is_neither_submitted_nor_orphaned(
    tmp_path: Path,
) -> None:
    import apm.data.text.tinyworlds_v2.phase1_replay as replay
    import apm.data.text.tinyworlds_v2.phase1_runner as runner

    catalog = _catalog("0.001").resolved
    job = build_generation_jobs(
        (_reference_corpus().briefs[0],),
        (CANDIDATE_MODELS[0],),
        (catalog.generator_routes[0],),
    )[0]
    cache = ImmutableRawCache(tmp_path / "cancelled-replay-cache")
    ledger = RuntimeCostLedger()
    ledger.authorize_byok(_byok_authorization())
    ledger.attach_cache(cache)
    reservation = ledger.reserve(job.request, job.route, 1)
    ledger.halt("provider_cost_policy_violation")
    with pytest.raises(CostCapExceeded):
        ledger.ensure_reservation_postable(reservation)

    routes = (*catalog.generator_routes, catalog.verifier_route)
    assert runner._attempted_jobs(cache, (job,)) == ()
    assert replay._validate_raw_cache_journal(cache, routes) is None


def test_replay_rejects_historical_request_body_outside_its_route_lock(
    tmp_path: Path,
) -> None:
    import apm.data.text.tinyworlds_v2.phase1_replay as replay

    catalog = _catalog("0.001").resolved
    route = replace(catalog.generator_routes[0], catalog_sha256="f" * 64)
    planned = build_generation_jobs(
        (_reference_corpus().briefs[0],),
        (CANDIDATE_MODELS[0],),
        (catalog.generator_routes[0],),
    )[0]
    invalid = CanonicalRequest.from_body(
        route_lock_sha256=route.lock_sha256,
        endpoint=planned.request.endpoint,
        body={**planned.request.body, "plugins": ["forbidden"]},
    )
    cache = ImmutableRawCache(tmp_path / "invalid-historical-body")
    cache.prepare_request(invalid, route)

    with pytest.raises(Phase1ReplayError, match="historical route lock"):
        replay._validate_raw_cache_journal(
            cache,
            (*catalog.generator_routes, catalog.verifier_route),
        )


def test_runtime_cap_denial_stops_before_any_completion_post(tmp_path: Path) -> None:
    transport = _FixtureTransport()
    dependencies = Phase1Dependencies(
        prepare_references=lambda _progress: _reference_corpus(
            full_source_contract=True
        ),
        fetch_catalog=lambda: _catalog("0.001"),
        encode_text=_token_ids,
        measure_stories=_measure,
        load_api_key=lambda: "secret",
        make_client=lambda key, cache: OpenRouterClient(
            api_key=key,
            management_api_key="management-secret",
            transport=transport,
            cache=cache,
            retry_policy=RetryPolicy(max_attempts=2),
            require_byok_preflight=True,
            cost_ledger=RuntimeCostLedger("0.0000001"),
            sleeper=lambda _seconds: None,
        ),
    )
    staging = tmp_path / "runtime-cap-staging"
    staging.mkdir()

    result = run_phase1(
        staging,
        _paths(tmp_path, "runtime-cap-reference"),
        dependencies,
        emit=lambda _: None,
    )

    assert result.status == "blocked_by_runtime_cost_cap"
    assert transport.remote_calls == 0
    runtime = json.loads((result.directory / "runtime_cost_ledger.json").read_text())
    assert runtime["provider_reported_actual_usd"] == "0"
    assert runtime["conservative_unknown_charge_usd"] == "0"
    assert runtime["halted_reason"] == "runtime_cap_reservation_denied"


def test_fresh_catalog_drift_stops_before_the_next_route_batch(
    tmp_path: Path,
) -> None:
    transport = _FixtureTransport()
    validated_route_ids: list[str] = []

    def revalidate(route: RouteLock) -> RouteLock:
        validated_route_ids.append(route.route_id)
        if len(validated_route_ids) == 2:
            raise ValueError("simulated fresh-catalog drift")
        return route

    dependencies = Phase1Dependencies(
        prepare_references=lambda _progress: _reference_corpus(
            full_source_contract=True
        ),
        fetch_catalog=lambda: _catalog("0.001"),
        encode_text=_token_ids,
        measure_stories=_measure,
        load_api_key=lambda: "secret",
        make_client=lambda key, cache: OpenRouterClient(
            api_key=key,
            management_api_key="management-secret",
            transport=transport,
            cache=cache,
            retry_policy=RetryPolicy(max_attempts=2),
            require_byok_preflight=True,
            sleeper=lambda _seconds: None,
            clock=lambda: "2026-07-18T00:00:00Z",
        ),
        revalidate_route=revalidate,
    )
    staging = tmp_path / "catalog-drift-staging"
    staging.mkdir()

    result = run_phase1(
        staging,
        _paths(tmp_path, "catalog-drift-reference"),
        dependencies,
        emit=lambda _: None,
    )

    assert result.status == "catalog_route_drift"
    assert validated_route_ids == [
        CANDIDATE_MODELS[0].route_id,
        CANDIDATE_MODELS[1].route_id,
    ]
    assert transport.remote_calls == 50
    runtime = json.loads((result.directory / "runtime_cost_ledger.json").read_text())
    assert runtime["halted_reason"] == "catalog_route_drift"


def test_interrupted_raw_attempts_replay_exact_provider_failure_without_network(
    tmp_path: Path,
) -> None:
    transport = _ByokFailureTransport()
    dependencies = Phase1Dependencies(
        prepare_references=lambda _progress: _reference_corpus(
            full_source_contract=True
        ),
        fetch_catalog=lambda: _catalog("0.001"),
        encode_text=_token_ids,
        measure_stories=_measure,
        load_api_key=lambda: "secret",
        make_client=lambda key, cache: OpenRouterClient(
            api_key=key,
            management_api_key="management-secret",
            transport=transport,
            cache=cache,
            retry_policy=RetryPolicy(max_attempts=1),
            require_byok_preflight=True,
            sleeper=lambda _seconds: None,
            clock=lambda: "2026-07-18T00:00:00Z",
        ),
    )
    staging = tmp_path / "interrupted-provider-staging"
    staging.mkdir()

    result = run_phase1(
        staging,
        _paths(tmp_path, "interrupted-provider-reference"),
        dependencies,
        emit=lambda _: None,
    )

    assert result.status == "provider_billing_unknown"
    bakeoff_path = result.directory / "generator_bakeoff.jsonl"
    assert not bakeoff_path.is_file() or not bakeoff_path.read_bytes()
    attempts = tuple(
        (result.directory / "raw_cache" / "requests").glob(
            "*/attempts/*/metadata.json"
        )
    )
    assert attempts
    runtime = json.loads((result.directory / "runtime_cost_ledger.json").read_bytes())
    assert runtime["halted_reason"] == "provider_cost_policy_violation"
    replay = verify_phase1_derived_replay(result.directory)
    assert (
        f"routes/{CANDIDATE_MODELS[0].route_id}/manifest.json"
        in replay.compared_paths
    )

    tampered = tmp_path / "interrupted-provider-resealed"
    _hardlink_tree(result.directory, tampered)
    (tampered / "manifest.json").unlink()
    runtime_path = tampered / "runtime_cost_ledger.json"
    runtime["halted_reason"] = "provider_response_contract_failure"
    runtime_path.unlink()
    runtime_path.write_bytes(
        json.dumps(
            runtime,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    Phase1ArtifactBuilder(
        tampered,
        version="tinyworlds-v2-phase1-reference-v1",
    ).finalize()
    with pytest.raises(
        Phase1ReplayError,
        match="halt reason is not reproduced",
    ):
        verify_phase1_derived_replay(tampered)


def test_full_route_cost_attribution_includes_failed_retry_and_verifier(
    tmp_path: Path,
) -> None:
    references = _reference_corpus()
    catalog = _catalog("0.001").resolved
    jobs = build_generation_jobs(
        references.briefs[:1],
        CANDIDATE_MODELS[:2],
        catalog.generator_routes[:2],
    )
    cache = ImmutableRawCache(tmp_path / "attributed-cost-cache")

    def store_costs(
        request,
        route: RouteLock,
        costs: tuple[str, ...],
    ) -> None:
        for number, cost in enumerate(costs, start=1):
            cache.store_attempt(
                request,
                route,
                RawAttempt(
                    request_sha256=request.request_sha256,
                    attempt_number=number,
                    observed_at_utc="2026-07-18T00:00:00Z",
                    submission_catalog_sha256=route.catalog_sha256,
                    response=RawHttpResponse(
                        status_code=500 if number < len(costs) else 200,
                        headers=(),
                        body=b"{}",
                        billed_cost_usd=cost,
                    ),
                    transport_error_type=None,
                ),
            )

    samples = tuple(
        SimpleNamespace(sample_id=f"sample-{index}", job=job)
        for index, job in enumerate(jobs)
    )
    store_costs(
        jobs[0].request,
        jobs[0].route,
        ("0.03", "0.02"),
    )
    store_costs(
        jobs[1].request,
        jobs[1].route,
        ("0.04",),
    )
    verified = []
    for sample in samples:
        verifier_job = build_verifier_job(
            source_id=sample.sample_id,
            pair_id=sample.job.brief.brief_id,
            brief=sample.job.brief,
            story=_story() + f" {sample.sample_id}.",
            model=VERIFIER_MODEL,
            route=catalog.verifier_route,
        )
        store_costs(
            verifier_job.request,
            verifier_job.route,
            ("0.01",),
        )
        verified.append(SimpleNamespace(job=verifier_job))

    costs = _attributed_full_route_costs(cache, samples, tuple(verified))

    assert costs[jobs[0].route.route_id] == Decimal("0.06")
    assert costs[jobs[1].route.route_id] == Decimal("0.05")
    assert Decimal("0.02") + Decimal("0.01") < Decimal("0.05")


def test_expansion_cost_stop_reaches_stopped_result_with_one_cache_argument(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import apm.data.text.tinyworlds_v2.phase1_runner as runner

    transport = _FixtureTransport()
    dependencies = Phase1Dependencies(
        prepare_references=lambda _progress: _reference_corpus(),
        fetch_catalog=lambda: _catalog("0.001"),
        encode_text=_token_ids,
        measure_stories=_measure,
        load_api_key=lambda: "secret",
        make_client=lambda key, cache: OpenRouterClient(
            api_key=key,
            management_api_key="management-secret",
            transport=transport,
            cache=cache,
            retry_policy=RetryPolicy(max_attempts=2),
            require_byok_preflight=True,
            sleeper=lambda _seconds: None,
            clock=lambda: "2026-07-18T00:00:00Z",
        ),
    )
    stopped = object()
    captured: dict[str, object] = {}

    def fail_expansion(*_args: object, **_kwargs: object) -> None:
        raise CostCapExceeded("forced expansion boundary")

    def capture_stopped_result(
        builder: Phase1ArtifactBuilder,
        paths: Phase1Paths,
        cache: ImmutableRawCache,
        **kwargs: object,
    ) -> object:
        captured.update(kwargs)
        captured["cache"] = cache
        captured["builder"] = builder
        captured["paths"] = paths
        return stopped

    monkeypatch.setattr(runner, "_expand_finalists", fail_expansion)
    monkeypatch.setattr(runner, "_publish_stopped_result", capture_stopped_result)
    staging = tmp_path / "expansion-stop-staging"
    staging.mkdir()

    result = run_phase1(
        staging,
        _paths(tmp_path, "expansion-stop-reference"),
        dependencies,
        emit=lambda _: None,
    )

    assert result is stopped
    assert captured["status"] == "blocked_by_runtime_cost_cap"
    assert len(captured["generated_samples"]) == 7 * 50
    assert len(captured["submitted_jobs"]) == 7 * 50
    assert type(captured["cache"]) is ImmutableRawCache
    assert transport.remote_calls == 7 * 50
    assert (staging / "runtime_cost_ledger.json").is_file()


def test_cached_fixture_rebuild_is_byte_identical_and_audit_validates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _FixtureTransport()
    key_loads: list[str] = []
    dependencies = Phase1Dependencies(
        prepare_references=lambda _progress: _reference_corpus(full_source_contract=True),
        fetch_catalog=lambda: _catalog("0.001"),
        encode_text=_token_ids,
        measure_stories=_measure,
        load_api_key=lambda: key_loads.append("loaded") or "secret",
        make_client=lambda key, cache: OpenRouterClient(
            api_key=key,
            management_api_key="management-secret",
            transport=transport,
            cache=cache,
            retry_policy=RetryPolicy(
                max_attempts=2,
                initial_delay_seconds=0.0,
                maximum_delay_seconds=0.0,
            ),
            require_byok_preflight=True,
            sleeper=lambda _seconds: None,
            clock=lambda: "2026-07-18T00:00:00Z",
        ),
    )
    results = []
    for index in range(2):
        staging = tmp_path / f"staging-{index}"
        staging.mkdir()
        results.append(
            run_phase1(
                staging,
                _paths(tmp_path, f"reference-{index}"),
                dependencies,
                emit=lambda _: None,
            )
        )
        if index == 0:
            first_remote_calls = transport.remote_calls
        else:
            assert transport.remote_calls == first_remote_calls

    # Paired source stories are content-unique, while generated fixture texts
    # are identical. Exact caching therefore retains 200 genuine-reference
    # verifier requests and collapses only the generated duplicates.
    assert first_remote_calls == 1_200
    assert key_loads == ["loaded", "loaded"]
    assert all(result.status == "awaiting_human_audit" for result in results)
    assert all(len(result.qualified_route_ids) == 3 for result in results)
    cost_actuals = json.loads(
        (results[0].directory / "cost_actuals.json").read_text()
    )
    assert tuple(sorted(cost_actuals["projection_envelopes"])) == (
        "balanced",
        "economy",
        "quality_ceiling",
    )
    assert all(
        envelope["available"]
        for envelope in cost_actuals["projection_envelopes"].values()
    )
    assert not (results[0].directory / "audit_approval.json").exists()
    quality_details = json.loads(
        (results[0].directory / "quality_details.json").read_text()
    )
    first_full_report = quality_details["full_reports"][0]
    for dimension in (
        "preschool_vocabulary",
        "sentence_simplicity",
        "grammar",
        "plot_coherence",
        "non_repetition",
    ):
        assert f"blind_verifier_{dimension}_generated_mean" in first_full_report
        assert f"blind_verifier_{dimension}_genuine_mean" in first_full_report
        assert f"blind_verifier_{dimension}_mean_difference" in first_full_report
    assert "blind_verifier_mean_difference" not in first_full_report
    assert tuple((results[0].directory / "raw_cache" / "requests").rglob("response.body"))
    import apm.data.text.tinyworlds_v2.httpx_transport as httpx_transport
    import apm.data.text.tinyworlds_v2.reference_runtime as reference_runtime
    import apm.data.text.tinyworlds_v2.source_data as source_data

    forbidden = lambda *_args, **_kwargs: pytest.fail("replay crossed an external boundary")
    monkeypatch.setattr(httpx_transport, "load_openrouter_api_key", forbidden)
    monkeypatch.setattr(reference_runtime, "score_tinystories_checkpoint_nll", forbidden)
    monkeypatch.setattr(source_data, "select_archive_source_records", forbidden)
    calls_before_replay = transport.remote_calls
    keys_before_replay = tuple(key_loads)
    replay = verify_phase1_derived_replay(results[0].directory)
    assert transport.remote_calls == calls_before_replay
    assert tuple(key_loads) == keys_before_replay
    assert "audit_packet.json" in replay.compared_paths
    assert "finalist_decision.json" in replay.compared_paths
    assert "generator_bakeoff.jsonl" in replay.compared_paths
    assert "quality_comparisons.json" in replay.compared_paths
    assert "quality_details.json" in replay.compared_paths
    assert "verifier/manifest.json" in replay.compared_paths

    missing_byok = tmp_path / "tampered-missing-byok"
    _hardlink_tree(results[0].directory, missing_byok)
    (missing_byok / "manifest.json").unlink()
    (missing_byok / "byok_preflight.json").unlink()
    Phase1ArtifactBuilder(
        missing_byok,
        version="tinyworlds-v2-phase1-reference-v1",
    ).finalize()
    with pytest.raises(ValueError, match="requires byok_preflight.json"):
        validate_phase1_semantics(missing_byok)
    # Each promoted tree was strictly loaded during atomic promotion. Equal
    # self-authenticating manifests therefore prove equal paths and bytes
    # without rereading thousands of immutable raw-cache files into memory.
    assert (results[0].directory / "manifest.json").read_bytes() == (
        results[1].directory / "manifest.json"
    ).read_bytes()

    tampered = tmp_path / "tampered-replay"
    _hardlink_tree(results[0].directory, tampered)
    (tampered / "manifest.json").unlink()
    bakeoff_path = tampered / "generator_bakeoff.jsonl"
    records = tuple(json.loads(line) for line in bakeoff_path.read_bytes().splitlines())
    changed = ({**records[0], "input_tokens": records[0]["input_tokens"] + 1}, *records[1:])
    bakeoff_path.unlink()
    bakeoff_path.write_bytes(canonical_jsonl_bytes(changed))
    Phase1ArtifactBuilder(tampered, version="tinyworlds-v2-phase1-reference-v1").finalize()
    with pytest.raises(Phase1ReplayError, match="byte mismatch"):
        verify_phase1_derived_replay(tampered)

    quality_tampered = tmp_path / "tampered-quality-selection"
    _hardlink_tree(results[0].directory, quality_tampered)
    (quality_tampered / "manifest.json").unlink()
    details_path = quality_tampered / "quality_details.json"
    details = json.loads(details_path.read_bytes())
    disqualified = details["full_reports"][0]["route_id"]
    details["full_reports"][0]["schema_valid_rate"] = 0.0
    details["full_reports"][0]["failures"] = ["schema_valid_rate"]
    details["full_reports"][0]["passed"] = False
    details["selection"]["route_ids"] = [
        route_id
        for route_id in details["selection"]["route_ids"]
        if route_id != disqualified
    ]
    details_path.unlink()
    details_path.write_bytes(
        json.dumps(
            details,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    comparisons_path = quality_tampered / "quality_comparisons.json"
    comparisons = json.loads(comparisons_path.read_bytes())
    comparisons["qualified_route_ids"] = [
        route_id
        for route_id in comparisons["qualified_route_ids"]
        if route_id != disqualified
    ]
    comparisons_path.unlink()
    comparisons_path.write_bytes(
        json.dumps(
            comparisons,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    Phase1ArtifactBuilder(
        quality_tampered,
        version="tinyworlds-v2-phase1-reference-v1",
    ).finalize()
    with pytest.raises(Phase1ReplayError, match="quality_comparisons.json"):
        validate_phase1_semantics(quality_tampered)
    assert not (quality_tampered / "audit_approval.json").exists()

    journal_tampered = results[1].directory
    (journal_tampered / "manifest.json").unlink()
    settlement_path = next(
        (journal_tampered / "raw_cache" / "runtime-cost-journal").rglob(
            "settlement.json"
        )
    )
    settlement = json.loads(settlement_path.read_bytes())
    settlement_path.write_bytes(
        json.dumps(
            {**settlement, "charged_usd": "0"},
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    Phase1ArtifactBuilder(
        journal_tampered,
        version="tinyworlds-v2-phase1-reference-v1",
    ).finalize()
    with pytest.raises(Phase1ReplayError, match="cost settlement differs"):
        verify_phase1_derived_replay(journal_tampered)

from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path

import pytest

from apm.data.text.tinyworlds_v2.bakeoff import (
    TWO_ROUTE_AUTHOR_MODELS,
    NeutralStoryBrief,
    validate_plain_text_generated_story,
    validate_story_only_generated_story,
)
from apm.data.text.tinyworlds_v2.phase1_artifacts import canonical_jsonl_bytes
from apm.data.text.tinyworlds_v2.phase1_generation import GeneratedSample
from apm.data.text.tinyworlds_v2.generation_schema import RouteLock
from apm.data.text.tinyworlds_v2.prompt_tuning import (
    PROMPT_TUNING_BRIEF_COUNT,
    PROMPT_TUNING_HARD_CAP_USD,
    PROMPT_TUNING_V1_EXPERIMENT,
    PROMPT_TUNING_V1_MANIFEST_SHA256,
    PROMPT_TUNING_V1_VARIANTS,
    PROMPT_TUNING_V2_EXPERIMENT,
    PROMPT_TUNING_V2_VARIANTS,
    PROMPT_TUNING_V4_EXPERIMENT,
    PROMPT_TUNING_V4_MANIFEST_SHA256,
    PROMPT_TUNING_V4_VARIANTS,
    PROMPT_TUNING_V5_EXPERIMENT,
    PROMPT_TUNING_V5_MANIFEST_SHA256,
    PROMPT_TUNING_V5_VARIANTS,
    PromptTuningPaths,
    build_prompt_tuning_cost_preflight,
    build_prompt_tuning_jobs,
    select_prompt_tuning_briefs,
    validate_prompt_tuning,
    _decode_selected_briefs,
    _clean_comparator_provenance,
    _decontaminated_comparator_provenance,
    _load_control_cell,
    _load_control_routes,
    _render_review_html,
    _validate_tuning_plans,
    _validate_tuning_results,
    _variant_sample_record,
)


def _brief(index: int) -> NeutralStoryBrief:
    return NeutralStoryBrief(
        brief_id=f"brief-{index:03d}",
        source_record_id=f"source-{index:03d}",
        prompt_text=(
            'Write a short story using the verb "jump", noun "moon", '
            'and adjective "kind".'
        ),
        required_words=("jump", "moon", "kind"),
        requested_features=(),
        matched_reference_text=(
            "Once upon a time, Mia saw the moon. She was kind to a bird and "
            "helped it jump home. They smiled and played together every day."
        ),
    )


def _routes() -> tuple[RouteLock, ...]:
    return tuple(
        RouteLock(
            route_id=model.route_id,
            catalog_sha256="a" * 64,
            requested_model=model.request_model_id,
            canonical_model=model.canonical_slug,
            provider_slug="fixture",
            returned_provider="Fixture",
            quantization="bf16",
            input_usd_per_million="0.1",
            output_usd_per_million="0.5",
        )
        for model in TWO_ROUTE_AUTHOR_MODELS
    )


def _encode(text: str) -> tuple[int, ...]:
    return tuple(range(max(1, len(text.split()))))


def test_prompt_tuning_selects_twenty_independently_of_source_order() -> None:
    briefs = tuple(_brief(index) for index in range(40))

    forward = select_prompt_tuning_briefs(briefs)
    reverse = select_prompt_tuning_briefs(tuple(reversed(briefs)))

    assert len(forward) == PROMPT_TUNING_BRIEF_COUNT == 20
    assert tuple(brief.brief_id for brief in forward) == tuple(
        brief.brief_id for brief in reverse
    )


def test_prompt_tuning_buys_only_one_twenty_story_cell_per_model() -> None:
    selected = select_prompt_tuning_briefs(tuple(_brief(index) for index in range(40)))
    routes = _routes()
    cells = build_prompt_tuning_jobs(selected, routes)

    preflight = build_prompt_tuning_cost_preflight(
        selected,
        tuple(cell for cell in cells if cell.variant.paid),
        routes,
        _encode,
    )

    assert tuple(cell.variant.variant_id for cell in cells) == (
        "v6-control",
        "v7-tuned",
    )
    assert all(len(cell.jobs) == 40 for cell in cells)
    assert tuple(estimate.request_count for estimate in preflight.route_estimates) == (
        20,
        20,
    )
    assert Decimal(preflight.hard_cap_usd) == Decimal(PROMPT_TUNING_HARD_CAP_USD)
    control_system = cells[0].jobs[0].request.body["messages"][0]["content"]
    tuned_system = cells[1].jobs[0].request.body["messages"][0]["content"]
    tuned_user = cells[1].jobs[0].request.body["messages"][1]["content"]
    assert "one short, complete" in control_system
    assert "one short, complete" not in tuned_system
    assert "18 to 20 complete sentences" in tuned_user
    assert "155 to 190 words" in tuned_user
    assert len(PROMPT_TUNING_V2_VARIANTS) == 2
    assert tuple(variant.variant_id for variant in PROMPT_TUNING_V1_VARIANTS) == (
        "v4-control",
        "v6-tuned",
    )


def test_prompt_tuning_paths_keep_v1_and_v2_artifacts_disjoint(
    tmp_path: Path,
) -> None:
    v1 = PromptTuningPaths.from_repository(tmp_path, PROMPT_TUNING_V1_EXPERIMENT)
    v2 = PromptTuningPaths.from_repository(tmp_path, PROMPT_TUNING_V2_EXPERIMENT)

    assert v1.control_source.name == "reference-two-route-v2"
    assert v2.control_source.name == "prompt-tuning-v1"
    assert v1.raw_cache.name == v1.destination.name == "prompt-tuning-v1"
    assert v2.raw_cache.name == v2.destination.name == "prompt-tuning-v2"
    assert v1.raw_cache != v2.raw_cache
    assert v1.destination != v2.destination


def test_bare_prompt_experiment_has_its_own_control_cache_and_destination(
    tmp_path: Path,
) -> None:
    v4 = PromptTuningPaths.from_repository(tmp_path, PROMPT_TUNING_V4_EXPERIMENT)

    assert v4.control_source.name == "prompt-tuning-v2"
    assert v4.quality_comparator_source.name == "prompt-tuning-v3"
    assert v4.raw_cache.name == v4.destination.name == "prompt-tuning-v4"

    v5 = PromptTuningPaths.from_repository(tmp_path, PROMPT_TUNING_V5_EXPERIMENT)
    assert v5.control_source.name == "prompt-tuning-v4"
    assert v5.quality_comparator_source.name == "prompt-tuning-v3"
    assert v5.raw_cache.name == v5.destination.name == "prompt-tuning-v5"


def test_bare_prompt_experiment_buys_only_plain_released_prompt_calls() -> None:
    selected = select_prompt_tuning_briefs(tuple(_brief(index) for index in range(40)))
    cells = build_prompt_tuning_jobs(
        selected,
        _routes(),
        variants=PROMPT_TUNING_V4_VARIANTS,
    )

    assert tuple(cell.variant.variant_id for cell in cells) == (
        "v7-control",
        "v8-bare-released-prompt",
    )
    paid = cells[1]
    assert len(paid.jobs) == 40
    for job in paid.jobs:
        assert job.request.body["messages"] == [
            {
                "role": "user",
                "content": f"{job.brief.prompt_text}\n\nPossible story:",
            }
        ]
        assert "response_format" not in job.request.body


def test_minimal_length_experiment_adds_only_one_prompt_sentence() -> None:
    selected = select_prompt_tuning_briefs(tuple(_brief(index) for index in range(40)))
    cells = build_prompt_tuning_jobs(
        selected,
        _routes(),
        variants=PROMPT_TUNING_V5_VARIANTS,
    )

    assert tuple(cell.variant.variant_id for cell in cells) == (
        "v8-control",
        "v9-minimal-length-cue",
    )
    for job in cells[1].jobs:
        assert job.request.body["messages"] == [
            {
                "role": "user",
                "content": (
                    f"{job.brief.prompt_text}\n\n"
                    "Aim for about 130 to 150 words.\n\nPossible story:"
                ),
            }
        ]
        assert "response_format" not in job.request.body


def test_v2_clean_comparator_binds_the_official_validation_source() -> None:
    assert _clean_comparator_provenance() == {
        "artifact_manifest_sha256": (
            "28a1280c256d8a6ecfc5e4048e65f71e5839c522e391eb03dd07b1669a66d5e9"
        ),
        "human_matched_references_used_for_scoring": False,
        "observation_artifact_path": "reference_observations.jsonl",
        "observation_artifact_sha256": (
            "58f1aa4e6015b5d8fb5fcf9eb0750b09bdcdb59"
        ),
        "record_count": 10_000,
        "record_id_prefix": "v2-validation:",
        "reference_profile_sha256": (
            "15955f1cc2490014593e3ad9296ba67e5b754a3d3edaeb7292d74a5da24b12ec"
        ),
        "source_dataset_id": "roneneldan/TinyStories",
        "source_file_sha256": (
            "6874bae9a4c1a4e7edcf0e53b86c17817e9cf881fc75ff2368da457b80c0585d"
        ),
        "source_file_size_bytes": 22_502_601,
        "source_filename": "TinyStoriesV2-GPT4-valid.txt",
        "source_revision": "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64",
    }


def test_bare_prompt_comparator_binds_the_decontaminated_v3_profile() -> None:
    assert _decontaminated_comparator_provenance() == {
        "artifact_manifest_sha256": (
            "50576804cf1cd81efce293ec62732aad3ec9251ca1010511eedacb630c087b74"
        ),
        "artifact_path": "data/tinyworlds-v2/prompt-tuning-v3",
        "human_matched_references_used_for_scoring": False,
        "identity_policy": (
            "unicode-nfkc-casefold-whitespace-collapse-sha256-with-full-text-"
            "confirmation-v1"
        ),
        "record_count": 6_607,
        "reference_profile_sha256": (
            "0bdac5ca35c7f67fcc0560184fda8156991a32a8ce500fe65f358e8e6ddf0c61"
        ),
        "source_dataset_id": "roneneldan/TinyStories",
        "source_filename": "TinyStoriesV2-GPT4-valid.txt",
        "source_revision": "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64",
        "training_overlap_excluded": True,
    }


def test_prompt_tuning_plan_validation_rebuilds_exact_requests(tmp_path: Path) -> None:
    selected = select_prompt_tuning_briefs(tuple(_brief(index) for index in range(40)))
    cells = build_prompt_tuning_jobs(selected, _routes())
    plans = tmp_path / "plans"
    plans.mkdir()
    for cell in cells:
        (plans / f"{cell.variant.variant_id}.jsonl").write_bytes(
            canonical_jsonl_bytes(
                {
                    **job.request.as_record(),
                    "body": job.request.body,
                    "brief_id": job.brief.brief_id,
                    "route_id": job.route.route_id,
                    "variant_id": cell.variant.variant_id,
                }
                for job in cell.jobs
            )
        )

    _validate_tuning_plans(tmp_path, cells)
    path = plans / "v7-tuned.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[0]["body"]["seed"] += 1
    path.write_bytes(canonical_jsonl_bytes(records))

    with pytest.raises(ValueError, match="plan differs"):
        _validate_tuning_plans(tmp_path, cells)


def test_prompt_tuning_result_validation_rederives_story_evidence() -> None:
    selected = select_prompt_tuning_briefs(tuple(_brief(index) for index in range(40)))
    cells = build_prompt_tuning_jobs(selected, _routes())
    story = (
        "Once upon a time, Mia saw the moon above her little red home. "
        "Mia was kind and helped a small bird jump over a wet stone. "
        "The bird was scared, but Mia stayed close and spoke in a soft voice. "
        "They walked home together and shared a warm meal with their friends."
    )
    records_by_variant: dict[str, list[dict[str, object]]] = {
        variant.variant_id: [] for variant in PROMPT_TUNING_V2_VARIANTS
    }
    for cell in cells:
        for index, job in enumerate(cell.jobs):
            payload, validation = validate_story_only_generated_story(
                job.brief,
                json.dumps({"story": story}),
            )
            assert payload is not None and validation.accepted
            sample = GeneratedSample(
                job,
                payload,
                validation,
                f"generation-{cell.variant.variant_id}-{index}",
                100,
                100,
                "0.001",
                None,
            )
            records_by_variant[cell.variant.variant_id].append(
                _variant_sample_record(cell.variant.variant_id, sample)
            )

    samples = _validate_tuning_results(
        cells,
        tuple(records_by_variant["v6-control"]),
        tuple(records_by_variant["v7-tuned"]),
        variants=PROMPT_TUNING_V2_VARIANTS,
    )
    assert len(samples) == 80

    html = _render_review_html(
        tuple(
            (
                cell.variant,
                tuple(
                    sample
                    for (variant_id, _sample_id), sample in samples.items()
                    if variant_id == cell.variant.variant_id
                ),
            )
            for cell in cells
        ),
        selected,
        variants=PROMPT_TUNING_V2_VARIANTS,
    )
    assert "New v7-tuned" in html
    assert "Cached v6-control" in html
    assert "Old V4 control" not in html
    assert "human context only; not scored" in html
    assert "clean 10,000-story official GPT-4 validation" in html

    tampered = [dict(record) for record in records_by_variant["v7-tuned"]]
    tampered[0] = dict(tampered[0])
    tampered[0]["request_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="local derivation"):
        _validate_tuning_results(
            cells,
            tuple(records_by_variant["v6-control"]),
            tuple(tampered),
            variants=PROMPT_TUNING_V2_VARIANTS,
        )


def test_bare_prompt_result_validation_treats_reply_as_plain_story() -> None:
    selected = select_prompt_tuning_briefs(tuple(_brief(index) for index in range(40)))
    cells = build_prompt_tuning_jobs(
        selected,
        _routes(),
        variants=PROMPT_TUNING_V4_VARIANTS,
    )
    story = (
        "Once upon a time, Mia saw the moon above her little red home. "
        "Mia was kind and helped a small bird jump over a wet stone. "
        "The bird was scared, but Mia stayed close and spoke softly. "
        "They walked home together and shared a warm meal with friends."
    )
    records: dict[str, list[dict[str, object]]] = {
        variant.variant_id: [] for variant in PROMPT_TUNING_V4_VARIANTS
    }
    for cell in cells:
        for index, job in enumerate(cell.jobs):
            if job.request_contract.plain_text_story_response:
                payload, validation = validate_plain_text_generated_story(
                    job.brief,
                    story,
                )
            else:
                payload, validation = validate_story_only_generated_story(
                    job.brief,
                    json.dumps({"story": story}),
                )
            assert payload is not None and validation.accepted
            sample = GeneratedSample(
                job,
                payload,
                validation,
                f"generation-{cell.variant.variant_id}-{index}",
                100,
                100,
                "0.001",
                None,
            )
            records[cell.variant.variant_id].append(
                _variant_sample_record(cell.variant.variant_id, sample)
            )

    samples = _validate_tuning_results(
        cells,
        tuple(records["v7-control"]),
        tuple(records["v8-bare-released-prompt"]),
        variants=PROMPT_TUNING_V4_VARIANTS,
    )

    assert len(samples) == 80
    assert all(
        sample.payload is not None and sample.payload.story == story
        for sample in samples.values()
    )


def test_completed_prompt_tuning_v1_remains_exactly_valid() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    artifact = repository_root / "data" / "tinyworlds-v2" / "prompt-tuning-v1"
    if not artifact.is_dir():
        pytest.skip("completed development artifact is not present")

    manifest = validate_prompt_tuning(
        artifact,
        experiment=PROMPT_TUNING_V1_EXPERIMENT,
    )

    assert manifest.manifest_sha256 == PROMPT_TUNING_V1_MANIFEST_SHA256


def test_v2_reuses_the_exact_v6_control_without_rescoring() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    artifact = repository_root / "data" / "tinyworlds-v2" / "prompt-tuning-v1"
    if not artifact.is_dir():
        pytest.skip("completed development artifact is not present")
    selected = _decode_selected_briefs(artifact / "selected_briefs.jsonl")
    routes = _load_control_routes(artifact, PROMPT_TUNING_V2_EXPERIMENT)
    cell = build_prompt_tuning_jobs(
        selected,
        routes,
        variants=(PROMPT_TUNING_V2_VARIANTS[0],),
    )[0]

    samples, measurements = _load_control_cell(
        artifact,
        cell,
        PROMPT_TUNING_V2_EXPERIMENT,
    )

    assert len(samples) == 40
    assert sum(sample.validation.accepted for sample in samples) == 34
    assert len(measurements.measurements) == 34
    assert all(
        not measurement.record_id.startswith("v6-tuned:")
        for measurement in measurements.measurements
    )


def test_v4_reuses_the_exact_v7_control_without_rescoring() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    artifact = repository_root / "data" / "tinyworlds-v2" / "prompt-tuning-v2"
    if not artifact.is_dir():
        pytest.skip("completed V7 development artifact is not present")
    selected = _decode_selected_briefs(artifact / "selected_briefs.jsonl")
    routes = _load_control_routes(artifact, PROMPT_TUNING_V4_EXPERIMENT)
    cell = build_prompt_tuning_jobs(
        selected,
        routes,
        variants=(PROMPT_TUNING_V4_VARIANTS[0],),
    )[0]

    samples, measurements = _load_control_cell(
        artifact,
        cell,
        PROMPT_TUNING_V4_EXPERIMENT,
    )

    assert len(samples) == 40
    assert sum(sample.validation.accepted for sample in samples) == 32
    assert len(measurements.measurements) == 32
    assert all(
        not measurement.record_id.startswith("v7-tuned:")
        for measurement in measurements.measurements
    )


def test_completed_bare_prompt_artifact_remains_exactly_valid() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    artifact = repository_root / "data" / "tinyworlds-v2" / "prompt-tuning-v4"
    if not artifact.is_dir():
        pytest.skip("completed bare-prompt artifact is not present")

    manifest = validate_prompt_tuning(
        artifact,
        experiment=PROMPT_TUNING_V4_EXPERIMENT,
    )

    assert manifest.manifest_sha256 == PROMPT_TUNING_V4_MANIFEST_SHA256


def test_v5_reuses_the_exact_bare_prompt_control_without_rescoring() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    artifact = repository_root / "data" / "tinyworlds-v2" / "prompt-tuning-v4"
    if not artifact.is_dir():
        pytest.skip("completed bare-prompt artifact is not present")
    selected = _decode_selected_briefs(artifact / "selected_briefs.jsonl")
    routes = _load_control_routes(artifact, PROMPT_TUNING_V5_EXPERIMENT)
    cell = build_prompt_tuning_jobs(
        selected,
        routes,
        variants=(PROMPT_TUNING_V5_VARIANTS[0],),
    )[0]

    samples, measurements = _load_control_cell(
        artifact,
        cell,
        PROMPT_TUNING_V5_EXPERIMENT,
    )

    assert len(samples) == 40
    assert all(sample.validation.accepted for sample in samples)
    assert len(measurements.measurements) == 40
    assert all(
        not measurement.record_id.startswith("v8-bare-released-prompt:")
        for measurement in measurements.measurements
    )


def test_completed_minimal_length_artifact_remains_exactly_valid() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    artifact = repository_root / "data" / "tinyworlds-v2" / "prompt-tuning-v5"
    if not artifact.is_dir():
        pytest.skip("completed minimal-length artifact is not present")

    manifest = validate_prompt_tuning(
        artifact,
        experiment=PROMPT_TUNING_V5_EXPERIMENT,
    )

    assert manifest.manifest_sha256 == PROMPT_TUNING_V5_MANIFEST_SHA256

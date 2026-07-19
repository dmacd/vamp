"""Opt-in integration gates for real TinyWorlds-v2 Phase 1 dependencies."""

from __future__ import annotations

from decimal import Decimal
import math
import os
from pathlib import Path

import pytest

from apm.data.text.tinyworlds_v2.bakeoff import (
    CANDIDATE_MODELS,
    NeutralStoryBrief,
    assistant_message_content,
    validate_generated_story,
)
from apm.data.text.tinyworlds_v2.catalog import resolve_openrouter_catalog
from apm.data.text.tinyworlds_v2.generation_cache import ImmutableRawCache
from apm.data.text.tinyworlds_v2.httpx_transport import (
    HttpxTransport,
    fetch_catalog_payloads,
    load_openrouter_api_key,
    load_openrouter_management_api_key,
)
from apm.data.text.tinyworlds_v2.openrouter import OpenRouterClient, RetryPolicy
from apm.data.text.tinyworlds_v2.phase1_generation import build_generation_jobs
from apm.data.text.tinyworlds_v2.phase1_runner import (
    OPENROUTER_BYOK_ATTESTATION_FILENAME,
)
from apm.data.text.tinyworlds_v2.reference_pipeline import (
    PHASE1_ARCHIVE_REFERENCE_COUNT,
    PHASE1_BRIEF_COUNT,
    PHASE1_PROMPT_METADATA_COUNT,
    PHASE1_VALIDATION_REFERENCE_COUNT,
    build_phase1_reference_inputs,
)
from apm.data.text.tinyworlds_v2.reference_runtime import (
    NllStory,
    score_tinystories_checkpoint_nll,
)
from apm.data.text.tinyworlds_v2.source_data import (
    select_archive_source_records,
    select_validation_story_records,
)


pytestmark = pytest.mark.integration

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_ARCHIVE_PATH = (
    _REPOSITORY_ROOT
    / "data"
    / "tinyworlds-v2"
    / "source"
    / "TinyStories_all_data.tar.gz"
)
_VALIDATION_PATH = (
    _REPOSITORY_ROOT
    / "data"
    / "tinystories-v2"
    / "TinyStoriesV2-GPT4-valid.txt"
)
_CHECKPOINT_PATH = _REPOSITORY_ROOT / "checkpoints" / "tinystories-8m" / "checkpoint"
_TOKENIZER_PATH = (
    _REPOSITORY_ROOT
    / "checkpoints"
    / "tinystories-8m"
    / "tokenizer"
    / "tokenizer.json"
)
_PHASE1_SELECTION_SEED = "tinyworlds-v2-phase1-reference-v1"


def test_real_pinned_sources_build_complete_disjoint_reference_inputs() -> None:
    """Stream the real pins and exercise the complete fixed Phase 1 selection."""
    missing = tuple(
        str(path) for path in (_ARCHIVE_PATH, _VALIDATION_PATH) if not path.is_file()
    )
    if missing:
        pytest.skip("local pinned TinyStories sources are absent: " + ", ".join(missing))

    archive_selections = select_archive_source_records(
        _ARCHIVE_PATH,
        seed=_PHASE1_SELECTION_SEED,
    )
    archive_story_hashes = frozenset(
        record.normalized_story_sha256
        for cohort in (
            archive_selections.prompt_metadata_records,
            archive_selections.reference_story_records,
            archive_selections.paired_records,
        )
        for record in cohort
    )
    validation_records = select_validation_story_records(
        _VALIDATION_PATH,
        seed=_PHASE1_SELECTION_SEED,
        exclude_normalized_story_sha256=archive_story_hashes,
    )
    archive_cohorts = (
        archive_selections.prompt_metadata_records,
        archive_selections.reference_story_records,
        archive_selections.paired_records,
    )
    cohort_id_sets = tuple(
        frozenset(record.record_id for record in cohort)
        for cohort in archive_cohorts
    )

    assert tuple(map(len, archive_cohorts)) == (
        PHASE1_PROMPT_METADATA_COUNT,
        PHASE1_ARCHIVE_REFERENCE_COUNT,
        PHASE1_BRIEF_COUNT,
    )
    assert len(validation_records) == PHASE1_VALIDATION_REFERENCE_COUNT
    assert all(
        cohort_id_sets[left].isdisjoint(cohort_id_sets[right])
        for left in range(len(cohort_id_sets))
        for right in range(left + 1, len(cohort_id_sets))
    )
    assert len({record.content_sha256 for record in validation_records}) == (
        PHASE1_VALIDATION_REFERENCE_COUNT
    )

    inputs = build_phase1_reference_inputs(
        archive_selections,
        validation_records,
    )

    assert len(inputs.briefs) == PHASE1_BRIEF_COUNT
    assert len(inputs.reference_records) == (
        PHASE1_ARCHIVE_REFERENCE_COUNT + PHASE1_VALIDATION_REFERENCE_COUNT
    )
    assert len(inputs.reference_annotations) == len(inputs.reference_records)
    assert inputs.ingredient_profile.record_count == PHASE1_PROMPT_METADATA_COUNT
    assert len({record.record_id for record in inputs.reference_records}) == len(
        inputs.reference_records
    )


def test_real_tinystories_checkpoint_scores_two_stories_on_one_gpu() -> None:
    """Run real tokenizer/checkpoint NLL only with both assets and one JAX GPU."""
    if not _CHECKPOINT_PATH.is_dir() or not _TOKENIZER_PATH.is_file():
        pytest.skip("local TinyStories-8M checkpoint or tokenizer is absent")
    jax = pytest.importorskip("jax")
    try:
        devices = jax.local_devices()
    except RuntimeError as error:
        pytest.skip(f"JAX accelerator discovery failed: {type(error).__name__}")
    if len(devices) != 1 or str(devices[0].platform) != "gpu":
        pytest.skip("real NLL integration requires exactly one visible JAX GPU")

    stories = (
        NllStory(
            "nll-smoke-cat",
            "Once there was a little cat. The cat found a red ball and took it home.",
        ),
        NllStory(
            "nll-smoke-frog",
            "A small frog wanted to help. It gave its friend a warm green leaf.",
        ),
    )
    run = score_tinystories_checkpoint_nll(
        stories,
        _CHECKPOINT_PATH,
        _TOKENIZER_PATH,
        sequence_length=32,
        batch_size=2,
        require_gpu=True,
    )

    assert run.jax_platform == "gpu"
    assert tuple(score.record_id for score in run.scores) == tuple(
        story.record_id for story in stories
    )
    assert all(
        score.token_count > 0
        and math.isfinite(score.normalized_nll)
        and score.normalized_nll > 0.0
        for score in run.scores
    )
    assert len(run.checkpoint_parameter_checksum) == 64
    assert len(run.checkpoint_manifest_sha256) == 64
    assert len(run.tokenizer_sha256) == 64


def test_opt_in_billable_openrouter_cheapest_route_smoke(tmp_path: Path) -> None:
    """Send exactly one paid, provider-locked request after explicit opt-in."""
    if os.environ.get("APM_RUN_BILLABLE_OPENROUTER_SMOKE") != "1":
        pytest.skip("billable OpenRouter smoke requires explicit environment opt-in")

    transport = HttpxTransport(user_agent="apm-tinyworlds-v2-integration/1")
    resolved = resolve_openrouter_catalog(fetch_catalog_payloads(transport))
    cheapest_model = min(
        CANDIDATE_MODELS,
        key=lambda model: (
            Decimal(model.plan_prompt_usd_per_million)
            + Decimal(model.plan_completion_usd_per_million),
            CANDIDATE_MODELS.index(model),
        ),
    )
    route_by_id = {route.route_id: route for route in resolved.generator_routes}
    cheapest_route = route_by_id[cheapest_model.route_id]
    live_route_order = {
        route.route_id: index
        for index, route in enumerate(resolved.generator_routes)
    }
    live_cheapest_route = min(
        resolved.generator_routes,
        key=lambda route: (
            Decimal(route.input_usd_per_million)
            + Decimal(route.output_usd_per_million),
            live_route_order[route.route_id],
        ),
    )
    assert cheapest_model is CANDIDATE_MODELS[0]
    assert live_cheapest_route == cheapest_route
    assert cheapest_route.catalog_sha256 == resolved.snapshot_sha256

    brief = NeutralStoryBrief(
        brief_id="brief-openrouter-billable-smoke-v1",
        source_record_id="integration:openrouter-smoke-v1",
        prompt_text=(
            'Write a simple story using the noun "cat", the verb "help", '
            'and the adjective "kind".'
        ),
        required_words=("cat", "help", "kind"),
        requested_features=(),
        matched_reference_text=(
            "A kind cat saw a little bird. The cat ran to help the bird get home."
        ),
    )
    job = build_generation_jobs(
        (brief,),
        (cheapest_model,),
        (cheapest_route,),
    )[0]
    cache = ImmutableRawCache(tmp_path / "raw-openrouter-smoke")
    client = OpenRouterClient(
        api_key=load_openrouter_api_key(_REPOSITORY_ROOT),
        management_api_key=load_openrouter_management_api_key(),
        byok_attestation_path=(
            _REPOSITORY_ROOT / OPENROUTER_BYOK_ATTESTATION_FILENAME
        ),
        transport=transport,
        cache=cache,
        retry_policy=RetryPolicy(
            max_attempts=1,
            initial_delay_seconds=0.0,
            maximum_delay_seconds=0.0,
        ),
        require_byok_preflight=True,
    )
    client.verify_no_byok()

    response = client.generate(job.request, cheapest_route)

    assert response.provenance is not None
    assert response.provenance.requested_model == cheapest_route.requested_model
    assert response.provenance.returned_model == cheapest_route.canonical_model
    assert response.provenance.returned_provider == cheapest_route.returned_provider
    assert response.usage is not None
    assert response.billed_cost_usd is not None
    assert Decimal(response.billed_cost_usd) > 0
    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens > 0
    attempts = cache.load_attempts(job.request)
    assert len(attempts) == 1
    assert attempts[0].response == response
    generated_payload, validation = validate_generated_story(
        brief,
        assistant_message_content(response.body),
    )
    assert generated_payload is not None
    assert validation.schema_valid

"""Small paired prompt experiments for the two TinyWorlds-v2 author routes.

Each experiment reuses the same deterministic 20 development briefs and one
cached prior-prompt cell, then purchases one versioned prompt cell before
measuring accepted stories with the frozen TinyStories checkpoint.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal
from hashlib import sha256
from html import escape
import math
from pathlib import Path
import tempfile

from apm.data.text.tinyworlds_v2.bakeoff import (
    SYNTHETIC_STORY_REQUEST_V4,
    SYNTHETIC_STORY_REQUEST_V6,
    SYNTHETIC_STORY_REQUEST_V7,
    SYNTHETIC_STORY_REQUEST_V8,
    SYNTHETIC_STORY_REQUEST_V9,
    TWO_ROUTE_AUTHOR_MODELS,
    GenerationRequestContract,
    NeutralStoryBrief,
    StoryValidation,
    plain_story_user_prompt,
    validate_plain_text_generated_story,
    validate_story_only_generated_story,
)
from apm.data.text.tinyworlds_v2.catalog import (
    CatalogPayloads,
    PHASE1_PROMPT_TOKEN_UPPER_BOUND,
)
from apm.data.text.tinyworlds_v2.byok_contract import canonical_byok_authorization
from apm.data.text.tinyworlds_v2.generation_cache import ImmutableRawCache
from apm.data.text.tinyworlds_v2.generation_costs import (
    CostPreflight,
    RouteWorkload,
    RuntimeCostLedger,
    TokenWorkload,
    build_cost_preflight,
    enforce_cost_cap,
    exclusive_paid_run_lock,
)
from apm.data.text.tinyworlds_v2.generation_schema import RouteLock
from apm.data.text.tinyworlds_v2.json_contracts import (
    JsonObject,
    canonical_json_loads,
    require_exact_fields,
    require_json_object,
)
from apm.data.text.tinyworlds_v2.openrouter import OpenRouterClient
from apm.data.text.tinyworlds_v2.phase1_artifacts import (
    Phase1ArtifactBuilder,
    Phase1ArtifactManifest,
    canonical_jsonl_bytes,
    load_phase1_artifact_tree,
)
from apm.data.text.tinyworlds_v2.phase1_generation import (
    GeneratedSample,
    GenerationJob,
    build_generation_jobs,
    execute_generation_jobs,
    generated_observation,
)
from apm.data.text.tinyworlds_v2.phase1_replay import (
    _decode_reference_profile,
    _load_measurement_batch,
    _load_reference_observations,
    _route,
)
from apm.data.text.tinyworlds_v2.phase1_semantics import _reference_profile_record
from apm.data.text.tinyworlds_v2.phase1_runner import (
    MeasurementBatch,
    StoryMeasurement,
    _quality_report_record,
)
from apm.data.text.tinyworlds_v2.quality import (
    TWO_ROUTE_AUTHOR_ORDER,
    QualityPhase,
    RouteQualityReport,
    evaluate_route_quality,
)
from apm.data.text.tinyworlds_v2.reference_profile import (
    ReferenceProfile,
    build_reference_profile,
)
from apm.data.text.tinyworlds_v2.reference_runtime import NllStory
from apm.data.text.tinyworlds_v2.route_lock import validate_route_semantics
from apm.data.text.tinyworlds_v2.two_route_bakeoff import (
    TWO_ROUTE_BASE_REFERENCE_MANIFEST_SHA256,
    TwoRouteDependencies,
    TwoRoutePaths,
    TwoRouteReferenceEvidence,
    canonical_story_only_content,
    load_two_route_reference_evidence,
    production_two_route_dependencies,
    validate_two_route_bakeoff,
    _validate_raw_evidence,
)


PROMPT_TUNING_V1_VERSION = "tinyworlds-v2-prompt-tuning-v1"
PROMPT_TUNING_V2_VERSION = "tinyworlds-v2-prompt-tuning-v2"
PROMPT_TUNING_V4_VERSION = "tinyworlds-v2-prompt-tuning-v4"
PROMPT_TUNING_V5_VERSION = "tinyworlds-v2-prompt-tuning-v5"
PROMPT_TUNING_SELECTION_NAMESPACE = "tinyworlds-v2-prompt-tuning-development-v1"
PROMPT_TUNING_BRIEF_COUNT = 20
PROMPT_TUNING_HARD_CAP_USD = "1.00"
PROMPT_TUNING_GENERATION_WORKERS = 8
PROMPT_TUNING_RETRY_ALLOWANCE_BASIS_POINTS = 10_000
PROMPT_TUNING_BASELINE_MANIFEST_SHA256 = (
    "6f0e14a7bf8cdcc933f5f6b459e33e6027e14fa714cdd938d384fcd8ebc042b9"
)
PROMPT_TUNING_V1_MANIFEST_SHA256 = (
    "074cdacbc38e311a85de988801a8c5d2cef561fd88b19daa43640176162836f3"
)
PROMPT_TUNING_V2_MANIFEST_SHA256 = (
    "838facd8975a04561987ebac3412c8e7897ee3ce4783259600f34aa26a347b4a"
)
PROMPT_TUNING_V4_MANIFEST_SHA256 = (
    "362a0c85c7722fbaf36120eaa5479285edb798bc067d8f7c7fd41631571e2bb0"
)
PROMPT_TUNING_V5_MANIFEST_SHA256 = (
    "1605d21acff2647fe4be456a627653f606b7e4e90c7241d3d552ebe513430c73"
)
PROMPT_REEVALUATION_V3_MANIFEST_SHA256 = (
    "50576804cf1cd81efce293ec62732aad3ec9251ca1010511eedacb630c087b74"
)
PROMPT_REEVALUATION_V3_PROFILE_SHA256 = (
    "0bdac5ca35c7f67fcc0560184fda8156991a32a8ce500fe65f358e8e6ddf0c61"
)
PROMPT_REEVALUATION_V3_RECORD_COUNT = 6_607
PROMPT_TUNING_CLEAN_COMPARATOR_PATH = "reference_observations.jsonl"
PROMPT_TUNING_CLEAN_COMPARATOR_SOURCE_SHA256 = (
    "58f1aa4e6015b5d8fb5fcf9eb0750b09bdcdb59"
)
PROMPT_TUNING_CLEAN_COMPARATOR_PROFILE_SHA256 = (
    "15955f1cc2490014593e3ad9296ba67e5b754a3d3edaeb7292d74a5da24b12ec"
)
PROMPT_TUNING_CLEAN_COMPARATOR_RECORD_COUNT = 10_000
PROMPT_TUNING_CLEAN_COMPARATOR_ID_PREFIX = "v2-validation:"


@dataclass(frozen=True, slots=True)
class PromptVariant:
    """One ordered prompt cell, including whether it requires new inference."""

    variant_id: str
    request_contract: GenerationRequestContract
    paid: bool


PROMPT_TUNING_V1_VARIANTS = (
    PromptVariant("v4-control", SYNTHETIC_STORY_REQUEST_V4, False),
    PromptVariant("v6-tuned", SYNTHETIC_STORY_REQUEST_V6, True),
)

PROMPT_TUNING_V2_VARIANTS = (
    PromptVariant("v6-control", SYNTHETIC_STORY_REQUEST_V6, False),
    PromptVariant("v7-tuned", SYNTHETIC_STORY_REQUEST_V7, True),
)

PROMPT_TUNING_V4_VARIANTS = (
    PromptVariant("v7-control", SYNTHETIC_STORY_REQUEST_V7, False),
    PromptVariant("v8-bare-released-prompt", SYNTHETIC_STORY_REQUEST_V8, True),
)

PROMPT_TUNING_V5_VARIANTS = (
    PromptVariant("v8-control", SYNTHETIC_STORY_REQUEST_V8, False),
    PromptVariant("v9-minimal-length-cue", SYNTHETIC_STORY_REQUEST_V9, True),
)


@dataclass(frozen=True, slots=True)
class PromptTuningExperiment:
    """One immutable two-cell prompt experiment and its control provenance."""

    version: str
    variants: tuple[PromptVariant, PromptVariant]
    control_source_kind: str
    control_source_manifest_sha256: str
    control_source_variant_id: str
    raw_cache_name: str
    destination_name: str
    workload_label: str

    def __post_init__(self) -> None:
        if type(self.version) is not str or not self.version:
            raise ValueError("prompt-tuning experiment version must be nonempty")
        if tuple(variant.paid for variant in self.variants) != (False, True):
            raise ValueError("prompt tuning requires one control then one paid variant")
        if self.control_source_kind not in {
            "two-route-v2",
            "prompt-tuning-v1",
            "prompt-tuning-v2",
            "prompt-tuning-v4",
        }:
            raise ValueError("prompt-tuning control source kind is unsupported")
        for value, label in (
            (self.control_source_manifest_sha256, "control-source manifest"),
            (self.control_source_variant_id, "control-source variant"),
            (self.raw_cache_name, "raw-cache name"),
            (self.destination_name, "destination name"),
            (self.workload_label, "workload label"),
        ):
            if type(value) is not str or not value:
                raise ValueError(f"prompt-tuning {label} must be nonempty")


PROMPT_TUNING_V1_EXPERIMENT = PromptTuningExperiment(
    version=PROMPT_TUNING_V1_VERSION,
    variants=PROMPT_TUNING_V1_VARIANTS,
    control_source_kind="two-route-v2",
    control_source_manifest_sha256=PROMPT_TUNING_BASELINE_MANIFEST_SHA256,
    control_source_variant_id="v4-control",
    raw_cache_name="prompt-tuning-v1",
    destination_name="prompt-tuning-v1",
    # Keep the completed V1 preflight byte contract available to its validator.
    workload_label="prompt-tuning-two-new-variants",
)

PROMPT_TUNING_V2_EXPERIMENT = PromptTuningExperiment(
    version=PROMPT_TUNING_V2_VERSION,
    variants=PROMPT_TUNING_V2_VARIANTS,
    control_source_kind="prompt-tuning-v1",
    control_source_manifest_sha256=PROMPT_TUNING_V1_MANIFEST_SHA256,
    control_source_variant_id="v6-tuned",
    raw_cache_name="prompt-tuning-v2",
    destination_name="prompt-tuning-v2",
    workload_label="prompt-tuning-v7-cell",
)

PROMPT_TUNING_V4_EXPERIMENT = PromptTuningExperiment(
    version=PROMPT_TUNING_V4_VERSION,
    variants=PROMPT_TUNING_V4_VARIANTS,
    control_source_kind="prompt-tuning-v2",
    control_source_manifest_sha256=PROMPT_TUNING_V2_MANIFEST_SHA256,
    control_source_variant_id="v7-tuned",
    raw_cache_name="prompt-tuning-v4",
    destination_name="prompt-tuning-v4",
    workload_label="prompt-tuning-v8-bare-released-cell",
)

PROMPT_TUNING_V5_EXPERIMENT = PromptTuningExperiment(
    version=PROMPT_TUNING_V5_VERSION,
    variants=PROMPT_TUNING_V5_VARIANTS,
    control_source_kind="prompt-tuning-v4",
    control_source_manifest_sha256=PROMPT_TUNING_V4_MANIFEST_SHA256,
    control_source_variant_id="v8-bare-released-prompt",
    raw_cache_name="prompt-tuning-v5",
    destination_name="prompt-tuning-v5",
    workload_label="prompt-tuning-v9-minimal-length-cue-cell",
)

PROMPT_TUNING_EXPERIMENTS = (
    PROMPT_TUNING_V1_EXPERIMENT,
    PROMPT_TUNING_V2_EXPERIMENT,
    PROMPT_TUNING_V4_EXPERIMENT,
    PROMPT_TUNING_V5_EXPERIMENT,
)


@dataclass(frozen=True, slots=True)
class PromptTuningPaths:
    """Fixed source, cache, and publication paths for prompt tuning."""

    repository_root: Path
    base_reference: Path
    control_source: Path
    quality_comparator_source: Path
    raw_cache: Path
    destination: Path

    @classmethod
    def from_repository(
        cls,
        repository_root: str | Path,
        experiment: PromptTuningExperiment = PROMPT_TUNING_V2_EXPERIMENT,
    ) -> "PromptTuningPaths":
        root = Path(repository_root).resolve()
        data_root = root / "data" / "tinyworlds-v2"
        control_name = {
            "two-route-v2": "reference-two-route-v2",
            "prompt-tuning-v1": "prompt-tuning-v1",
            "prompt-tuning-v2": "prompt-tuning-v2",
            "prompt-tuning-v4": "prompt-tuning-v4",
        }[experiment.control_source_kind]
        return cls(
            repository_root=root,
            base_reference=data_root / "reference",
            control_source=data_root / control_name,
            quality_comparator_source=data_root / "prompt-tuning-v3",
            raw_cache=data_root / "cache" / experiment.raw_cache_name,
            destination=data_root / experiment.destination_name,
        )


@dataclass(frozen=True, slots=True)
class VariantJobs:
    """One prompt variant and its route-major job matrix."""

    variant: PromptVariant
    jobs: tuple[GenerationJob, ...]


@dataclass(frozen=True, slots=True)
class PromptTuningRunResult:
    """Published prompt experiment identity and its diagnostic recommendations."""

    directory: Path
    manifest_sha256: str
    actual_cost_usd: str
    best_variants: tuple[tuple[str, str], ...]


def select_prompt_tuning_briefs(
    briefs: Sequence[NeutralStoryBrief],
) -> tuple[NeutralStoryBrief, ...]:
    """Select the lowest namespaced hashes without depending on input order."""
    if len(briefs) < PROMPT_TUNING_BRIEF_COUNT:
        raise ValueError("prompt tuning requires at least 20 neutral briefs")
    if len({brief.brief_id for brief in briefs}) != len(briefs):
        raise ValueError("prompt-tuning brief IDs must be unique")
    return tuple(
        sorted(
            briefs,
            key=lambda brief: (
                sha256(
                    f"{PROMPT_TUNING_SELECTION_NAMESPACE}\0{brief.brief_id}".encode(
                        "utf-8"
                    )
                ).digest(),
                brief.brief_id,
            ),
        )[:PROMPT_TUNING_BRIEF_COUNT]
    )


def build_prompt_tuning_jobs(
    briefs: Sequence[NeutralStoryBrief],
    routes: Sequence[RouteLock],
    *,
    variants: Sequence[PromptVariant] = PROMPT_TUNING_V2_VARIANTS,
) -> tuple[VariantJobs, ...]:
    """Build the complete variant/route/brief request matrix."""
    if len(briefs) != PROMPT_TUNING_BRIEF_COUNT:
        raise ValueError("prompt tuning requires exactly 20 selected briefs")
    if tuple(route.route_id for route in routes) != TWO_ROUTE_AUTHOR_ORDER:
        raise ValueError("prompt tuning requires the fixed Qwen/GPT route order")
    if not variants or len({variant.variant_id for variant in variants}) != len(variants):
        raise ValueError("prompt-tuning variants must be nonempty and unique")
    return tuple(
        VariantJobs(
            variant,
            build_generation_jobs(
                briefs,
                TWO_ROUTE_AUTHOR_MODELS,
                routes,
                request_contract=variant.request_contract,
            ),
        )
        for variant in variants
    )


def build_prompt_tuning_cost_preflight(
    selected_briefs: Sequence[NeutralStoryBrief],
    variant_jobs: Sequence[VariantJobs],
    routes: Sequence[RouteLock],
    encode_text: Callable[[str], tuple[int, ...]],
    *,
    workload_label: str = PROMPT_TUNING_V2_EXPERIMENT.workload_label,
) -> CostPreflight:
    """Price only the new 20-by-two prompt cell with a full retry reserve."""
    expected_output = math.ceil(
        sum(len(encode_text(brief.matched_reference_text)) for brief in selected_briefs)
        / len(selected_briefs)
    )
    paid_jobs = tuple(
        job
        for cell in variant_jobs
        if cell.variant.paid
        for job in cell.jobs
    )
    workloads = []
    for route in routes:
        jobs = tuple(job for job in paid_jobs if job.route.route_id == route.route_id)
        expected_count = PROMPT_TUNING_BRIEF_COUNT * sum(
            cell.variant.paid for cell in variant_jobs
        )
        if len(jobs) != expected_count:
            raise ValueError("prompt-tuning paid job matrix is incomplete")
        input_counts = tuple(len(encode_text(job.request.body_json)) for job in jobs)
        conservative_input = max(2 * count + 512 for count in input_counts)
        if conservative_input > PHASE1_PROMPT_TOKEN_UPPER_BOUND:
            raise ValueError("a prompt-tuning request exceeds the prompt-token bound")
        workloads.append(
            RouteWorkload(
                route,
                TokenWorkload(
                    label=workload_label,
                    request_count=len(jobs),
                    input_tokens_per_request=math.ceil(
                        sum(input_counts) / len(input_counts)
                    ),
                    output_tokens_per_request=expected_output,
                    conservative_input_tokens_per_request=conservative_input,
                    conservative_output_tokens_per_request=512,
                    retry_allowance_basis_points=(
                        PROMPT_TUNING_RETRY_ALLOWANCE_BASIS_POINTS
                    ),
                ),
            )
        )
    return build_cost_preflight(
        tuple(workloads),
        hard_cap_usd=PROMPT_TUNING_HARD_CAP_USD,
    )


def run_prompt_tuning(
    staging_directory: str | Path,
    paths: PromptTuningPaths,
    dependencies: TwoRouteDependencies,
    *,
    experiment: PromptTuningExperiment = PROMPT_TUNING_V2_EXPERIMENT,
    emit: Callable[[str], None] = print,
) -> PromptTuningRunResult:
    """Run, measure, authenticate, and atomically publish the fixed prompt pilot."""
    with exclusive_paid_run_lock(paths.raw_cache):
        return _run_prompt_tuning(
            staging_directory,
            paths,
            dependencies,
            experiment=experiment,
            emit=emit,
        )


def _run_prompt_tuning(
    staging_directory: str | Path,
    paths: PromptTuningPaths,
    dependencies: TwoRouteDependencies,
    *,
    experiment: PromptTuningExperiment,
    emit: Callable[[str], None],
) -> PromptTuningRunResult:
    staging = Path(staging_directory)
    if not staging.is_dir():
        raise FileNotFoundError(f"prompt-tuning staging directory is missing: {staging}")
    if paths.destination.exists() or paths.destination.is_symlink():
        raise FileExistsError(f"prompt-tuning destination exists: {paths.destination}")
    _validate_control_source(paths.control_source, experiment)
    references = load_two_route_reference_evidence(paths.base_reference)
    quality_comparator = _load_experiment_comparator(paths, experiment)
    selected = _selected_experiment_briefs(
        paths.control_source,
        references,
        experiment,
    )
    builder = Phase1ArtifactBuilder(staging, version=experiment.version)
    _record_progress(builder, "offline_control_started")
    baseline_routes = _load_control_routes(paths.control_source, experiment)
    control_cell = build_prompt_tuning_jobs(
        selected,
        baseline_routes,
        variants=(experiment.variants[0],),
    )[0]
    control_samples, control_measurements = _load_control_cell(
        paths.control_source,
        control_cell,
        experiment,
    )
    emit(
        "Prompt tuning phase 1/4: loaded 40 cached "
        f"{experiment.control_source_variant_id} control stories."
    )
    _record_progress(builder, "offline_control_completed")

    emit("Prompt tuning phase 2/4: resolving exact Qwen/GPT routes and costs.")
    payloads, live_routes = dependencies.fetch_routes()
    paid_cells = build_prompt_tuning_jobs(
        selected,
        live_routes,
        variants=experiment.variants[1:],
    )
    all_cells = (control_cell, *paid_cells)
    preflight = build_prompt_tuning_cost_preflight(
        selected,
        paid_cells,
        live_routes,
        dependencies.encode_text,
        workload_label=experiment.workload_label,
    )
    _write_plan(
        builder,
        references,
        selected,
        payloads,
        baseline_routes,
        live_routes,
        all_cells,
        preflight,
        experiment,
        quality_comparator,
    )
    emit(
        "Prompt-tuning cost preflight: "
        f"expected ${preflight.expected_usd}; conservative "
        f"${preflight.conservative_usd} / ${preflight.hard_cap_usd} cap"
    )
    for estimate in preflight.route_estimates:
        emit(
            f"  {estimate.route_id}: {estimate.request_count} new requests, "
            f"expected ${estimate.expected_usd}, reserve ${estimate.conservative_usd}"
        )
    enforce_cost_cap(preflight)

    cache = ImmutableRawCache(paths.raw_cache)
    client = dependencies.make_client(dependencies.load_api_key(), cache)
    if type(client) is OpenRouterClient:
        client = replace(
            client,
            cost_ledger=RuntimeCostLedger(PROMPT_TUNING_HARD_CAP_USD),
        )
        client.cost_ledger.bootstrap(cache, live_routes)
        if client.require_byok_preflight:
            byok = client.verify_no_byok()
            builder.write_json("byok_preflight.json", byok.as_record())

    paid_request_count = sum(len(cell.jobs) for cell in paid_cells)
    emit(
        f"Prompt tuning phase 3/4: generating {paid_request_count} new paired stories."
    )
    _record_progress(builder, "paid_generation_started")
    from tqdm.auto import tqdm

    overall_bar = tqdm(
        total=2 * paid_request_count,
        desc="TinyWorlds prompt bakeoff overall",
        unit="story",
        position=0,
        dynamic_ncols=True,
        leave=True,
    )
    generation_bar = tqdm(
        total=paid_request_count,
        desc="Phase 3/4 generation",
        unit="story",
        position=1,
        dynamic_ncols=True,
        leave=False,
    )
    generated_cells: list[tuple[PromptVariant, tuple[GeneratedSample, ...]]] = []
    fresh_routes = tuple(dependencies.revalidate_route(route) for route in live_routes)
    for locked, fresh in zip(live_routes, fresh_routes, strict=True):
        validate_route_semantics(locked, fresh)
    for cell in paid_cells:
        cell_samples: list[GeneratedSample] = []
        for route in fresh_routes:
            route_jobs = tuple(
                replace(job, route=route)
                for job in cell.jobs
                if job.route.route_id == route.route_id
            )
            emit(f"  {cell.variant.variant_id} / {route.route_id}: 20 requests")
            batch = execute_generation_jobs(
                route_jobs,
                client,
                max_workers=PROMPT_TUNING_GENERATION_WORKERS,
                progress_callback=lambda amount: (
                    generation_bar.update(amount),
                    overall_bar.update(amount),
                ),
            )
            cell_samples.extend(batch)
            for sample in batch:
                builder.append_jsonl(
                    "generation_results.jsonl",
                    _variant_sample_record(cell.variant.variant_id, sample),
                )
            emit(
                f"    accepted {sum(sample.validation.accepted for sample in batch)}/20"
            )
        generated_cells.append((cell.variant, tuple(cell_samples)))
    generation_bar.close()
    _record_progress(builder, "paid_generation_completed")
    for sample in control_samples:
        builder.append_jsonl(
            "control_results.jsonl",
            _variant_sample_record(experiment.variants[0].variant_id, sample),
        )

    emit("Prompt tuning phase 4/4: scoring accepted stories with TinyStories-8M.")
    _record_progress(builder, "nll_measurement_started")
    accepted_new_count = sum(
        sample.validation.accepted
        for _variant, samples in generated_cells
        for sample in samples
    )
    measurement_bar = tqdm(
        total=accepted_new_count,
        desc="Phase 4/4 TinyStories NLL",
        unit="story",
        position=1,
        dynamic_ncols=True,
        leave=False,
    )
    new_measurements = _measure_generated_cells(
        tuple(generated_cells),
        dependencies.measure_stories,
        lambda amount: (
            measurement_bar.update(amount),
            overall_bar.update(amount),
        ),
    )
    measurement_bar.close()
    if overall_bar.n < overall_bar.total:
        overall_bar.update(overall_bar.total - overall_bar.n)
    overall_bar.close()
    _record_progress(builder, "nll_measurement_completed")
    all_measurements = _combined_measurements(
        control_measurements,
        tuple(generated_cells),
        new_measurements,
        control_variant=experiment.variants[0],
    )
    builder.write_bytes(
        "measurements.jsonl",
        canonical_jsonl_bytes(
            measurement.as_record() for measurement in all_measurements.measurements
        ),
    )
    builder.write_json(
        "measurement_runtime.json",
        {
            "control": control_measurements.runtime,
            "new": new_measurements.runtime,
        },
    )
    sample_cells = (
        (experiment.variants[0], control_samples),
        *generated_cells,
    )
    reports = _evaluate_cells(
        sample_cells,
        all_measurements,
        selected,
        references,
        comparison_profile=quality_comparator,
    )
    best_variants = _write_quality(
        builder,
        reports,
        sample_cells,
        selected,
        variants=experiment.variants,
    )
    builder.write_bytes(
        "review.html",
        _render_review_html(
            sample_cells,
            selected,
            variants=experiment.variants,
            comparison_description=(
                "the train-decontaminated 6,607-story official GPT-4 "
                "validation profile " + PROMPT_REEVALUATION_V3_PROFILE_SHA256
                if experiment.version in {
                    PROMPT_TUNING_V4_VERSION,
                    PROMPT_TUNING_V5_VERSION,
                }
                else None
            ),
        ).encode("utf-8"),
    )
    _write_cost_actuals(builder, generated_cells, client)
    _write_raw_evidence(builder, cache, paid_cells)
    builder.write_json(
        "status.json",
        {
            "phase1_gate_effect": "none_diagnostic_development_only",
            "status": "diagnostic_complete",
        },
    )
    _record_progress(builder, "artifact_completed")
    manifest = builder.finalize()
    validated = validate_prompt_tuning(builder.root, experiment=experiment)
    if validated.manifest_sha256 != manifest.manifest_sha256:
        raise ValueError("prompt-tuning validation changed artifact identity")
    destination = builder.promote(
        paths.destination,
        expected_manifest_sha256=manifest.manifest_sha256,
    )
    actual_cost = sum(
        (
            Decimal(sample.billed_cost_usd)
            for _variant, samples in generated_cells
            for sample in samples
        ),
        Decimal(0),
    )
    return PromptTuningRunResult(
        directory=destination,
        manifest_sha256=manifest.manifest_sha256,
        actual_cost_usd=str(actual_cost),
        best_variants=best_variants,
    )


def validate_prompt_tuning(
    directory: str | Path,
    *,
    experiment: PromptTuningExperiment | None = None,
) -> Phase1ArtifactManifest:
    """Authenticate the diagnostic tree and its fixed coverage contract."""
    root = Path(directory)
    manifest = load_phase1_artifact_tree(root)
    selected_experiment = (
        _experiment_for_version(manifest.version)
        if experiment is None
        else experiment
    )
    if manifest.version != selected_experiment.version:
        raise ValueError("artifact is not the requested prompt-tuning experiment")
    configuration = _json_object(root / "configuration.json")
    if configuration != _configuration_record(selected_experiment):
        raise ValueError("prompt-tuning configuration differs from its fixed contract")
    quality_comparator = _validate_quality_comparator(root, selected_experiment)
    selected = _decode_selected_briefs(root / "selected_briefs.jsonl")
    baseline_routes, live_routes = _decode_tuning_routes(root)
    cells = (
        *build_prompt_tuning_jobs(
            selected,
            baseline_routes,
            variants=(selected_experiment.variants[0],),
        ),
        *build_prompt_tuning_jobs(
            selected,
            live_routes,
            variants=selected_experiment.variants[1:],
        ),
    )
    _validate_tuning_plans(root, cells)
    controls = _jsonl_objects(root / "control_results.jsonl")
    generated = _jsonl_objects(root / "generation_results.jsonl")
    samples = _validate_tuning_results(
        cells,
        controls,
        generated,
        variants=selected_experiment.variants,
    )
    _validate_tuning_measurements(
        root,
        cells,
        samples,
        variants=selected_experiment.variants,
    )
    _validate_tuning_quality(
        root,
        selected,
        variants=selected_experiment.variants,
        cells=cells,
        samples=samples,
        comparison_profile=quality_comparator,
    )
    _validate_tuning_costs(
        root,
        live_routes,
        generated,
        variants=selected_experiment.variants,
    )
    canonical_byok_authorization(_json_object(root / "byok_preflight.json"))
    paid_cells = tuple(cell for cell in cells if cell.variant.paid)
    if len(paid_cells) != 1:
        raise ValueError("prompt tuning requires exactly one paid raw-evidence cell")
    _validate_raw_evidence(root, paid_cells[0].jobs, generated)
    if not (root / "review.html").is_file():
        raise ValueError("prompt tuning is missing its complete review page")
    status = _json_object(root / "status.json")
    if status != {
        "phase1_gate_effect": "none_diagnostic_development_only",
        "status": "diagnostic_complete",
    }:
        raise ValueError("prompt-tuning status differs")
    return manifest


def _validate_quality_comparator(
    root: Path,
    experiment: PromptTuningExperiment,
) -> ReferenceProfile | None:
    path = root / "quality_comparator.json"
    if experiment.version == PROMPT_TUNING_V1_VERSION:
        if path.exists():
            raise ValueError("historical V1 prompt tuning has no clean comparator")
        return None
    record = _json_object(path)
    require_exact_fields(
        record,
        ("profile", "provenance"),
        label="prompt-tuning quality comparator",
    )
    expected_provenance = _quality_comparator_provenance(experiment)
    if record["provenance"] != expected_provenance:
        raise ValueError("prompt-tuning quality-comparator provenance differs")
    profile = _decode_reference_profile(
        record["profile"],
        "prompt-tuning clean comparator profile",
    )
    expected_count, expected_sha256 = (
        (
            PROMPT_REEVALUATION_V3_RECORD_COUNT,
            PROMPT_REEVALUATION_V3_PROFILE_SHA256,
        )
        if experiment.version in {
            PROMPT_TUNING_V4_VERSION,
            PROMPT_TUNING_V5_VERSION,
        }
        else (
            PROMPT_TUNING_CLEAN_COMPARATOR_RECORD_COUNT,
            PROMPT_TUNING_CLEAN_COMPARATOR_PROFILE_SHA256,
        )
    )
    if (
        profile.record_count != expected_count
        or profile.profile_sha256 != expected_sha256
        or record["profile"] != _reference_profile_record(profile)
    ):
        raise ValueError("prompt-tuning clean comparator profile differs")
    return profile


def _decode_selected_briefs(path: Path) -> tuple[NeutralStoryBrief, ...]:
    records = _jsonl_objects(path)
    briefs = []
    for index, record in enumerate(records):
        require_exact_fields(
            record,
            (
                "brief_id",
                "matched_reference_text",
                "prompt_text",
                "requested_features",
                "required_words",
                "source_record_id",
            ),
            label=f"prompt-tuning brief {index}",
        )
        briefs.append(
            NeutralStoryBrief(
                brief_id=_text(record["brief_id"], "brief ID"),
                source_record_id=_text(record["source_record_id"], "source record ID"),
                prompt_text=_text(record["prompt_text"], "released instruction"),
                required_words=_string_tuple(record["required_words"], "required words"),
                requested_features=_string_tuple(
                    record["requested_features"], "requested features"
                ),
                matched_reference_text=_text(
                    record["matched_reference_text"], "matched reference"
                ),
            )
        )
    values = tuple(briefs)
    if len(values) != PROMPT_TUNING_BRIEF_COUNT:
        raise ValueError("prompt tuning does not contain exactly 20 selected briefs")
    if len({brief.brief_id for brief in values}) != len(values):
        raise ValueError("prompt-tuning selected brief identities repeat")
    if select_prompt_tuning_briefs(values) != values:
        raise ValueError("prompt-tuning selected briefs are not in namespaced-hash order")
    return values


def _decode_tuning_routes(
    root: Path,
) -> tuple[tuple[RouteLock, ...], tuple[RouteLock, ...]]:
    record = _json_object(root / "catalog" / "routes.json")
    require_exact_fields(
        record,
        ("baseline", "live", "snapshot_sha256"),
        label="prompt-tuning routes",
    )

    def decode_group(name: str) -> tuple[RouteLock, ...]:
        values = record[name]
        if type(values) is not list or len(values) != 2:
            raise ValueError(f"prompt-tuning {name} routes must contain two records")
        routes = tuple(
            _route(value, f"prompt-tuning {name} route {index}")
            for index, value in enumerate(values)
        )
        if tuple(route.route_id for route in routes) != TWO_ROUTE_AUTHOR_ORDER:
            raise ValueError(f"prompt-tuning {name} route order differs")
        return routes

    baseline, live = decode_group("baseline"), decode_group("live")
    payloads = CatalogPayloads(
        models=(root / "catalog" / "models.response").read_bytes(),
        endpoints=tuple(
            (
                model.request_model_id,
                (root / "catalog" / "endpoints" / f"{model.route_id}.response").read_bytes(),
            )
            for model in TWO_ROUTE_AUTHOR_MODELS
        ),
        model_plan_ids=tuple(model.request_model_id for model in TWO_ROUTE_AUTHOR_MODELS),
    )
    if record["snapshot_sha256"] != payloads.snapshot_sha256 or any(
        route.catalog_sha256 != payloads.snapshot_sha256 for route in live
    ):
        raise ValueError("prompt-tuning live catalog digest differs from raw bytes")
    return baseline, live


def _validate_tuning_plans(root: Path, cells: Sequence[VariantJobs]) -> None:
    for cell in cells:
        expected = tuple(
            {
                **job.request.as_record(),
                "body": job.request.body,
                "brief_id": job.brief.brief_id,
                "route_id": job.route.route_id,
                "variant_id": cell.variant.variant_id,
            }
            for job in cell.jobs
        )
        if _jsonl_objects(root / "plans" / f"{cell.variant.variant_id}.jsonl") != expected:
            raise ValueError(f"prompt-tuning {cell.variant.variant_id} plan differs")


def _validate_tuning_results(
    cells: Sequence[VariantJobs],
    controls: Sequence[JsonObject],
    generated: Sequence[JsonObject],
    *,
    variants: Sequence[PromptVariant],
) -> dict[tuple[str, str], GeneratedSample]:
    control_variants = tuple(variant for variant in variants if not variant.paid)
    paid_variants = tuple(variant for variant in variants if variant.paid)
    expected_control_count = PROMPT_TUNING_BRIEF_COUNT * len(TWO_ROUTE_AUTHOR_ORDER)
    expected_generated_count = expected_control_count * len(paid_variants)
    if (
        len(control_variants) != 1
        or len(controls) != expected_control_count
        or len(generated) != expected_generated_count
    ):
        raise ValueError("prompt-tuning result coverage differs from its fixed cells")
    jobs = {
        (cell.variant.variant_id, job.sample_id): job
        for cell in cells
        for job in cell.jobs
    }
    if len(jobs) != expected_control_count + expected_generated_count:
        raise ValueError("prompt-tuning planned result identities repeat")
    samples: dict[tuple[str, str], GeneratedSample] = {}
    for expected_variant, records in (
        (control_variants[0].variant_id, controls),
        *((variant.variant_id, generated) for variant in paid_variants),
    ):
        for record in records:
            variant_id = record.get("variant_id")
            if variant_id != expected_variant:
                if len(paid_variants) > 1:
                    continue
                raise ValueError("prompt-tuning result variant differs")
            sample_id = record.get("sample_id")
            if type(sample_id) is not str:
                raise ValueError("prompt-tuning result variant or sample identity differs")
            key = (variant_id, sample_id)
            if key in samples or key not in jobs:
                raise ValueError("prompt-tuning result identity is unknown or repeated")
            sample = _decode_control_sample(jobs[key], record)
            if record != _variant_sample_record(variant_id, sample):
                raise ValueError("prompt-tuning result differs from local derivation")
            samples[key] = sample
    if set(samples) != set(jobs):
        raise ValueError("prompt-tuning results omit planned cells")
    return samples


def _validate_tuning_measurements(
    root: Path,
    cells: Sequence[VariantJobs],
    samples: dict[tuple[str, str], GeneratedSample],
    *,
    variants: Sequence[PromptVariant],
) -> None:
    measurements = _load_measurement_batch(root / "measurements.jsonl")
    expected = {
        f"{variant_id}:{sample_id}"
        for (variant_id, sample_id), sample in samples.items()
        if sample.validation.accepted
    }
    if set(measurements.by_id) != expected:
        raise ValueError("prompt-tuning measurements differ from accepted cells")
    runtime = _json_object(root / "measurement_runtime.json")
    require_exact_fields(runtime, ("control", "new"), label="measurement runtime")
    if type(runtime["control"]) is not dict or type(runtime["new"]) is not dict:
        raise ValueError("prompt-tuning measurement runtimes must be objects")
    expected_variants = tuple(cell.variant.variant_id for cell in cells)
    if expected_variants != tuple(variant.variant_id for variant in variants):
        raise ValueError("prompt-tuning measurement cell order differs")


def _validate_tuning_quality(
    root: Path,
    selected: Sequence[NeutralStoryBrief],
    *,
    variants: Sequence[PromptVariant],
    cells: Sequence[VariantJobs],
    samples: dict[tuple[str, str], GeneratedSample],
    comparison_profile: ReferenceProfile | None,
) -> None:
    from apm.data.text.tinyworlds_v2.phase1_semantics import _quality_report

    quality = _json_object(root / "quality.json")
    require_exact_fields(
        quality,
        ("best_variant_by_route", "interpretation", "ranking_metric", "reports"),
        label="prompt-tuning quality",
    )
    values = quality["reports"]
    if type(values) is not list or len(values) != len(variants) * len(
        TWO_ROUTE_AUTHOR_ORDER
    ):
        raise ValueError("prompt tuning quality-report coverage differs")
    decoded = []
    for index, value in enumerate(values):
        if type(value) is not dict:
            raise ValueError("prompt-tuning quality reports must be objects")
        report_record = dict(value)
        variant_id = report_record.pop("variant_id", None)
        if variant_id not in {variant.variant_id for variant in variants}:
            raise ValueError("prompt-tuning quality report variant differs")
        decoded.append((variant_id, _quality_report(report_record, index)))
    expected_order = tuple(
        (variant.variant_id, route_id)
        for variant in variants
        for route_id in TWO_ROUTE_AUTHOR_ORDER
    )
    if tuple((variant_id, report.route_id) for variant_id, report in decoded) != expected_order:
        raise ValueError("prompt-tuning quality report order differs")
    brief_ids = {brief.brief_id for brief in selected}
    if any(
        report.phase is not QualityPhase.DIRECT or set(report.sample_ids) != brief_ids
        for _variant_id, report in decoded
    ):
        raise ValueError("prompt-tuning quality report scope differs")
    best = {
        route_id: min(
            (
                (variant_id, report)
                for variant_id, report in decoded
                if report.route_id == route_id
            ),
            key=lambda item: (
                item[1].alignment_distance,
                len(item[1].failures),
                tuple(variant.variant_id for variant in variants).index(
                    item[0]
                ),
            ),
        )[0]
        for route_id in TWO_ROUTE_AUTHOR_ORDER
    }
    if quality["best_variant_by_route"] != best:
        raise ValueError("prompt-tuning best-variant ranking differs from reports")
    if quality["ranking_metric"] != "minimum_alignment_distance_then_failure_count":
        raise ValueError("prompt-tuning ranking metric differs")
    if quality["interpretation"] != _quality_interpretation(variants):
        raise ValueError("prompt-tuning interpretation differs")
    if comparison_profile is not None:
        measurement_batch = _load_measurement_batch(root / "measurements.jsonl")
        sample_cells = tuple(
            (
                cell.variant,
                tuple(
                    samples[(cell.variant.variant_id, job.sample_id)]
                    for job in cell.jobs
                ),
            )
            for cell in cells
        )
        expected = _evaluate_cells(
            sample_cells,
            measurement_batch,
            selected,
            None,
            comparison_profile=comparison_profile,
        )
        expected_records = []
        for variant_id, report in expected:
            report_record = _quality_report_record(report)
            report_record["variant_id"] = variant_id
            expected_records.append(report_record)
        if values != expected_records:
            raise ValueError("prompt-tuning quality differs from clean comparator replay")


def _validate_tuning_costs(
    root: Path,
    live_routes: Sequence[RouteLock],
    generated: Sequence[JsonObject],
    *,
    variants: Sequence[PromptVariant],
) -> None:
    paid_variants = tuple(variant for variant in variants if variant.paid)
    expected_route_request_count = PROMPT_TUNING_BRIEF_COUNT * len(paid_variants)
    preflight = _json_object(root / "cost_estimates.json")
    estimates = preflight.get("route_estimates")
    if type(estimates) is not list or len(estimates) != 2:
        raise ValueError("prompt-tuning preflight requires two route estimates")
    workloads = []
    for route, value in zip(live_routes, estimates, strict=True):
        if type(value) is not dict:
            raise ValueError("prompt-tuning route estimate must be an object")
        require_exact_fields(
            value,
            (
                "conservative_input_tokens",
                "conservative_output_tokens",
                "conservative_usd",
                "expected_input_tokens",
                "expected_output_tokens",
                "expected_usd",
                "request_count",
                "route_id",
                "workload_label",
            ),
            label="prompt-tuning route estimate",
        )
        if (
            value["route_id"] != route.route_id
            or value["request_count"] != expected_route_request_count
        ):
            raise ValueError("prompt-tuning route estimate scope differs")
        totals = tuple(
            _integer(value[name], f"prompt-tuning {name}")
            for name in (
                "expected_input_tokens",
                "expected_output_tokens",
                "conservative_input_tokens",
                "conservative_output_tokens",
            )
        )
        if any(total % expected_route_request_count for total in totals):
            raise ValueError("prompt-tuning token estimates are not per-request totals")
        workloads.append(
            RouteWorkload(
                route,
                TokenWorkload(
                    label=_text(value["workload_label"], "workload label"),
                    request_count=expected_route_request_count,
                    input_tokens_per_request=(
                        totals[0] // expected_route_request_count
                    ),
                    output_tokens_per_request=(
                        totals[1] // expected_route_request_count
                    ),
                    conservative_input_tokens_per_request=(
                        totals[2] // expected_route_request_count
                    ),
                    conservative_output_tokens_per_request=(
                        totals[3] // expected_route_request_count
                    ),
                    retry_allowance_basis_points=(
                        PROMPT_TUNING_RETRY_ALLOWANCE_BASIS_POINTS
                    ),
                ),
            )
        )
    if preflight != build_cost_preflight(
        tuple(workloads), hard_cap_usd=PROMPT_TUNING_HARD_CAP_USD
    ).as_record():
        raise ValueError("prompt-tuning cost preflight differs from locked prices")

    actuals = _json_object(root / "cost_actuals.json")
    require_exact_fields(
        actuals,
        ("actual_billed_usd", "cells", "runtime_ledger"),
        label="prompt-tuning cost actuals",
    )
    expected_cells = []
    total = Decimal(0)
    for variant in paid_variants:
        for route_id in TWO_ROUTE_AUTHOR_ORDER:
            records = tuple(
                record
                for record in generated
                if record.get("variant_id") == variant.variant_id
                and record.get("route_id") == route_id
            )
            route_total = sum(
                (
                    Decimal(_text(record["billed_cost_usd"], "billed cost"))
                    for record in records
                ),
                Decimal(0),
            )
            total += route_total
            expected_cells.append(
                {
                    "accepted_count": sum(_accepted(record) for record in records),
                    "actual_billed_usd": float(route_total),
                    "request_count": len(records),
                    "route_id": route_id,
                    "variant_id": variant.variant_id,
                }
            )
    if actuals["cells"] != expected_cells or Decimal(
        str(actuals["actual_billed_usd"])
    ) != total:
        raise ValueError("prompt-tuning actual costs differ from generated results")
    ledger = actuals["runtime_ledger"]
    if type(ledger) is not dict or any(
        (
            Decimal(str(ledger.get("charged_total_usd"))) != total,
            Decimal(str(ledger.get("provider_reported_actual_usd"))) != total,
            ledger.get("provider_reported_attempt_count") != len(generated),
            Decimal(str(ledger.get("conservative_unknown_charge_usd"))) != 0,
            ledger.get("unknown_cost_attempt_count") != 0,
            ledger.get("in_flight_attempt_count") != 0,
            ledger.get("cancelled_before_post_count") != 0,
            ledger.get("halted_reason") is not None,
        )
    ):
        raise ValueError("prompt-tuning runtime ledger differs from settled results")


def main() -> None:
    """Run the one fixed prompt experiment, or validate its existing result."""
    repository_root = Path(__file__).resolve().parents[5]
    experiment = PROMPT_TUNING_V5_EXPERIMENT
    paths = PromptTuningPaths.from_repository(repository_root, experiment)
    if paths.destination.is_dir():
        manifest = validate_prompt_tuning(paths.destination, experiment=experiment)
        print(f"Existing prompt-tuning artifact: {paths.destination}")
        print(f"Manifest: {manifest.manifest_sha256}")
        print("No catalog, paid generation, or GPU scoring was repeated.")
        return
    paths.destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix="tinyworlds-v2-prompt-tuning-", dir=paths.destination.parent)
    )
    print(f"Temporary artifact directory: {staging}", flush=True)
    result = run_prompt_tuning(
        staging,
        paths,
        production_two_route_dependencies(
            replace(
                TwoRoutePaths.from_repository(repository_root),
                raw_cache=paths.raw_cache,
                destination=paths.destination,
            )
        ),
        experiment=experiment,
    )
    print(f"Prompt-tuning artifact: {result.directory}")
    print(f"Manifest: {result.manifest_sha256}")
    print(f"Actual new inference cost: ${result.actual_cost_usd}")
    for route_id, variant_id in result.best_variants:
        print(f"Diagnostic best for {route_id}: {variant_id}")


def _experiment_for_version(version: str) -> PromptTuningExperiment:
    matches = tuple(
        experiment
        for experiment in PROMPT_TUNING_EXPERIMENTS
        if experiment.version == version
    )
    if len(matches) != 1:
        raise ValueError("artifact is not a recognized prompt-tuning experiment")
    return matches[0]


def _validate_control_source(
    control_source: Path,
    experiment: PromptTuningExperiment,
) -> Phase1ArtifactManifest:
    if experiment.control_source_kind == "two-route-v2":
        manifest = validate_two_route_bakeoff(control_source)
    else:
        source_experiment = {
            "prompt-tuning-v1": PROMPT_TUNING_V1_EXPERIMENT,
            "prompt-tuning-v2": PROMPT_TUNING_V2_EXPERIMENT,
            "prompt-tuning-v4": PROMPT_TUNING_V4_EXPERIMENT,
        }.get(experiment.control_source_kind)
        if source_experiment is None:
            raise ValueError("prompt-tuning control source kind is unsupported")
        manifest = validate_prompt_tuning(
            control_source,
            experiment=source_experiment,
        )
    if manifest.manifest_sha256 != experiment.control_source_manifest_sha256:
        raise ValueError("prompt tuning requires its exact completed control source")
    return manifest


def _selected_experiment_briefs(
    control_source: Path,
    references: TwoRouteReferenceEvidence,
    experiment: PromptTuningExperiment,
) -> tuple[NeutralStoryBrief, ...]:
    expected = select_prompt_tuning_briefs(references.briefs)
    if experiment.control_source_kind == "two-route-v2":
        return expected
    selected = _decode_selected_briefs(control_source / "selected_briefs.jsonl")
    expected_by_id = {brief.brief_id: brief for brief in expected}
    if tuple(expected_by_id.get(brief.brief_id) for brief in selected) != selected:
        raise ValueError("prior prompt tuning selected briefs differ from references")
    return selected


def _load_experiment_comparator(
    paths: PromptTuningPaths,
    experiment: PromptTuningExperiment,
) -> ReferenceProfile | None:
    if experiment.version == PROMPT_TUNING_V1_VERSION:
        return None
    if experiment.version in {
        PROMPT_TUNING_V4_VERSION,
        PROMPT_TUNING_V5_VERSION,
    }:
        from apm.data.text.tinyworlds_v2.prompt_tuning_reevaluation import (
            validate_prompt_reevaluation,
        )

        manifest = validate_prompt_reevaluation(paths.quality_comparator_source)
        if manifest.manifest_sha256 != PROMPT_REEVALUATION_V3_MANIFEST_SHA256:
            raise ValueError("decontaminated prompt-tuning comparator differs")
        profile_record = _json_object(
            paths.quality_comparator_source / "comparator" / "profile.json"
        )
        profile = _decode_reference_profile(
            profile_record,
            "prompt-tuning decontaminated comparator profile",
        )
        if (
            profile.record_count != PROMPT_REEVALUATION_V3_RECORD_COUNT
            or profile.profile_sha256 != PROMPT_REEVALUATION_V3_PROFILE_SHA256
            or profile_record != _reference_profile_record(profile)
        ):
            raise ValueError("decontaminated prompt-tuning profile differs")
        return profile
    observations = _load_reference_observations(
        paths.base_reference / PROMPT_TUNING_CLEAN_COMPARATOR_PATH
    )
    selected = tuple(
        observation
        for observation in observations
        if observation.record_id.startswith(PROMPT_TUNING_CLEAN_COMPARATOR_ID_PREFIX)
    )
    if len(selected) != PROMPT_TUNING_CLEAN_COMPARATOR_RECORD_COUNT:
        raise ValueError("clean prompt-tuning comparator coverage differs")
    profile = build_reference_profile(selected)
    if profile.profile_sha256 != PROMPT_TUNING_CLEAN_COMPARATOR_PROFILE_SHA256:
        raise ValueError("clean prompt-tuning comparator profile differs")
    return profile


def _load_control_routes(
    control_source: Path,
    experiment: PromptTuningExperiment,
) -> tuple[RouteLock, ...]:
    if experiment.control_source_kind == "two-route-v2":
        return _load_baseline_routes(control_source)
    if experiment.control_source_kind.startswith("prompt-tuning-"):
        return _decode_tuning_routes(control_source)[1]
    raise ValueError("prompt-tuning control source kind is unsupported")


def _load_baseline_routes(control_source: Path) -> tuple[RouteLock, ...]:
    record = _json_object(control_source / "catalog" / "routes.json")
    values = record.get("generator_routes")
    if type(values) is not list or len(values) != 2:
        raise ValueError("V4 baseline route catalog is malformed")
    return tuple(_route(value, f"baseline route {index}") for index, value in enumerate(values))


def _load_control_cell(
    control_source: Path,
    cell: VariantJobs,
    experiment: PromptTuningExperiment,
) -> tuple[tuple[GeneratedSample, ...], MeasurementBatch]:
    if experiment.control_source_kind == "two-route-v2":
        return _load_two_route_control_cell(control_source, cell)
    if experiment.control_source_kind.startswith("prompt-tuning-"):
        return _load_prior_tuning_control_cell(control_source, cell, experiment)
    raise ValueError("prompt-tuning control source kind is unsupported")


def _load_two_route_control_cell(
    control_source: Path,
    cell: VariantJobs,
) -> tuple[tuple[GeneratedSample, ...], MeasurementBatch]:
    jobs = {job.sample_id: job for job in cell.jobs}
    records = {
        record["sample_id"]: record
        for record in _jsonl_objects(control_source / "generator_bakeoff.jsonl")
        if record.get("sample_id") in jobs
    }
    if set(records) != set(jobs):
        raise ValueError("V4 baseline does not cover the selected control cells")
    samples = tuple(_decode_control_sample(jobs[job_id], records[job_id]) for job_id in jobs)
    all_measurements = _load_measurement_batch(
        control_source / "measurements" / "generation.jsonl"
    )
    selected_ids = set(jobs)
    measurements = tuple(
        item for item in all_measurements.measurements if item.record_id in selected_ids
    )
    accepted_ids = {sample.sample_id for sample in samples if sample.validation.accepted}
    if {item.record_id for item in measurements} != accepted_ids:
        raise ValueError("V4 control measurement coverage differs")
    return samples, MeasurementBatch(measurements, all_measurements.runtime)


def _load_prior_tuning_control_cell(
    control_source: Path,
    cell: VariantJobs,
    experiment: PromptTuningExperiment,
) -> tuple[tuple[GeneratedSample, ...], MeasurementBatch]:
    jobs = {job.sample_id: job for job in cell.jobs}
    records = {
        record["sample_id"]: record
        for record in _jsonl_objects(control_source / "generation_results.jsonl")
        if record.get("variant_id") == experiment.control_source_variant_id
        and record.get("sample_id") in jobs
    }
    if set(records) != set(jobs):
        raise ValueError("prior prompt tuning does not cover the selected control cells")
    samples = tuple(
        _decode_control_sample(jobs[job_id], records[job_id]) for job_id in jobs
    )
    all_measurements = _load_measurement_batch(control_source / "measurements.jsonl")
    prefix = f"{experiment.control_source_variant_id}:"
    source_measurements = tuple(
        item
        for item in all_measurements.measurements
        if item.record_id.startswith(prefix)
        and item.record_id.removeprefix(prefix) in jobs
    )
    accepted_ids = {sample.sample_id for sample in samples if sample.validation.accepted}
    if {
        item.record_id.removeprefix(prefix) for item in source_measurements
    } != accepted_ids:
        raise ValueError("prior prompt-tuning control measurement coverage differs")
    measurements = tuple(
        StoryMeasurement(
            item.record_id.removeprefix(prefix),
            item.model_token_ids,
            item.normalized_nll,
            item.active_token_count,
        )
        for item in source_measurements
    )
    return samples, MeasurementBatch(measurements, all_measurements.runtime)


def _decode_control_sample(job: GenerationJob, record: JsonObject) -> GeneratedSample:
    payload_record = record.get("payload")
    if type(payload_record) is dict:
        story = payload_record.get("story")
        if type(story) is not str:
            raise ValueError("V4 control story is missing")
        if job.request_contract.plain_text_story_response:
            payload, validation = validate_plain_text_generated_story(job.brief, story)
        else:
            payload, validation = validate_story_only_generated_story(
                job.brief,
                canonical_story_only_content(story),
            )
    else:
        payload = None
        validation = _decode_story_validation(record.get("validation"))
    error_kind = record.get("error_kind")
    return GeneratedSample(
        job=job,
        payload=payload,
        validation=validation,
        generation_id=_optional_text(record.get("generation_id")),
        input_tokens=_integer(record.get("input_tokens"), "control input tokens"),
        output_tokens=_integer(record.get("output_tokens"), "control output tokens"),
        billed_cost_usd=str(record.get("billed_cost_usd")),
        error_kind=None if error_kind is None else _text(error_kind, "control error kind"),
    )


def _decode_story_validation(value: object) -> StoryValidation:
    if type(value) is not dict:
        raise ValueError("control validation must be an object")
    return StoryValidation(
        schema_valid=_boolean(value.get("schema_valid"), "schema_valid"),
        required_words_present=_boolean(
            value.get("required_words_present"), "required_words_present"
        ),
        evidence_valid=_boolean(value.get("evidence_valid"), "evidence_valid"),
        forbidden_identifier_present=_boolean(
            value.get("forbidden_identifier_present"), "forbidden_identifier_present"
        ),
        length_valid=_boolean(value.get("length_valid"), "length_valid"),
        accepted=_boolean(value.get("accepted"), "accepted"),
        rejection_reasons=tuple(
            _text(reason, "rejection reason")
            for reason in _list(value.get("rejection_reasons"), "rejection reasons")
        ),
        story_sha256=_optional_text(value.get("story_sha256")),
    )


def _measure_generated_cells(
    generated_cells: tuple[tuple[PromptVariant, tuple[GeneratedSample, ...]], ...],
    measure: Callable[[tuple[NllStory, ...], Callable[[int], None]], MeasurementBatch],
    progress_callback: Callable[[int], object],
) -> MeasurementBatch:
    stories = tuple(
        sorted(
            (
                NllStory(
                    f"{variant.variant_id}:{sample.sample_id}",
                    sample.payload.story,
                )
                for variant, samples in generated_cells
                for sample in samples
                if sample.validation.accepted and sample.payload is not None
            ),
            key=lambda story: story.record_id,
        )
    )
    return measure(stories, progress_callback)


def _combined_measurements(
    control: MeasurementBatch,
    generated_cells: tuple[tuple[PromptVariant, tuple[GeneratedSample, ...]], ...],
    generated: MeasurementBatch,
    *,
    control_variant: PromptVariant,
) -> MeasurementBatch:
    control_values = tuple(
        StoryMeasurement(
            f"{control_variant.variant_id}:{item.record_id}",
            item.model_token_ids,
            item.normalized_nll,
            item.active_token_count,
        )
        for item in control.measurements
    )
    expected_generated_ids = {
        f"{variant.variant_id}:{sample.sample_id}"
        for variant, samples in generated_cells
        for sample in samples
        if sample.validation.accepted
    }
    if {item.record_id for item in generated.measurements} != expected_generated_ids:
        raise ValueError("new prompt-tuning measurement coverage differs")
    return MeasurementBatch(
        tuple(sorted((*control_values, *generated.measurements), key=lambda item: item.record_id)),
        generated.runtime,
    )


def _evaluate_cells(
    sample_cells: Sequence[tuple[PromptVariant, tuple[GeneratedSample, ...]]],
    measurements: MeasurementBatch,
    selected: Sequence[NeutralStoryBrief],
    references: TwoRouteReferenceEvidence | None,
    *,
    comparison_profile: ReferenceProfile | None,
) -> tuple[tuple[str, RouteQualityReport], ...]:
    measurement_by_id = measurements.by_id
    if comparison_profile is None:
        if references is None:
            raise ValueError("historical prompt tuning requires paired references")
        selected_sources = {brief.source_record_id for brief in selected}
        paired_observations = tuple(
            observation
            for observation in references.paired_observations
            if observation.record_id in selected_sources
        )
        if len(paired_observations) != PROMPT_TUNING_BRIEF_COUNT:
            raise ValueError("selected paired reference coverage differs")
        paired_profile = build_reference_profile(paired_observations)
        reference_profile = references.reference_profile
        distribution_profile = paired_profile
    else:
        reference_profile = comparison_profile
        distribution_profile = comparison_profile
    feature_counts = Counter(feature for brief in selected for feature in brief.requested_features)
    expected_feature_rates = tuple(
        sorted((feature, count / len(selected)) for feature, count in feature_counts.items())
    )
    reports = []
    for variant, samples in sample_cells:
        for route_id in TWO_ROUTE_AUTHOR_ORDER:
            observations = tuple(
                generated_observation(
                    sample,
                    model_token_ids=(
                        measurement_by_id[f"{variant.variant_id}:{sample.sample_id}"].model_token_ids
                        if f"{variant.variant_id}:{sample.sample_id}" in measurement_by_id
                        else ()
                    ),
                    normalized_nll=(
                        measurement_by_id[
                            f"{variant.variant_id}:{sample.sample_id}"
                        ].normalized_nll
                        if f"{variant.variant_id}:{sample.sample_id}" in measurement_by_id
                        else None
                    ),
                )
                for sample in samples
                if sample.job.route.route_id == route_id
            )
            reports.append(
                (
                    variant.variant_id,
                    evaluate_route_quality(
                        observations,
                        reference_profile,
                        phase=QualityPhase.DIRECT,
                        matched_reference_profile=distribution_profile,
                        expected_feature_rates=expected_feature_rates,
                    ),
                )
            )
    return tuple(reports)


def _write_quality(
    builder: Phase1ArtifactBuilder,
    reports: Sequence[tuple[str, RouteQualityReport]],
    sample_cells: Sequence[tuple[PromptVariant, tuple[GeneratedSample, ...]]],
    selected: Sequence[NeutralStoryBrief],
    *,
    variants: Sequence[PromptVariant],
) -> tuple[tuple[str, str], ...]:
    best = tuple(
        (
            route_id,
            min(
                (
                    (variant_id, report)
                    for variant_id, report in reports
                    if report.route_id == route_id
                ),
                key=lambda item: (
                    item[1].alignment_distance,
                    len(item[1].failures),
                    tuple(variant.variant_id for variant in variants).index(
                        item[0]
                    ),
                ),
            )[0],
        )
        for route_id in TWO_ROUTE_AUTHOR_ORDER
    )
    report_records = []
    for variant_id, report in reports:
        record = _quality_report_record(report)
        record["variant_id"] = variant_id
        report_records.append(record)
    builder.write_json(
        "quality.json",
        {
            "best_variant_by_route": dict(best),
            "interpretation": _quality_interpretation(variants),
            "ranking_metric": "minimum_alignment_distance_then_failure_count",
            "reports": report_records,
        },
    )
    briefs = {brief.brief_id: brief for brief in selected}
    for variant, samples in sample_cells:
        for route_id in TWO_ROUTE_AUTHOR_ORDER:
            accepted = tuple(
                sample
                for sample in samples
                if sample.job.route.route_id == route_id
                and sample.validation.accepted
                and sample.payload is not None
            )[:3]
            for sample in accepted:
                brief = briefs[sample.job.brief.brief_id]
                builder.append_jsonl(
                    "sample_previews.jsonl",
                    {
                        "brief_id": brief.brief_id,
                        "generated_story": sample.payload.story,
                        "matched_reference_story": brief.matched_reference_text,
                        "released_instruction": brief.prompt_text,
                        "route_id": route_id,
                        "variant_id": variant.variant_id,
                    },
                )
    return best


def _render_review_html(
    sample_cells: Sequence[tuple[PromptVariant, tuple[GeneratedSample, ...]]],
    selected: Sequence[NeutralStoryBrief],
    *,
    variants: Sequence[PromptVariant],
    comparison_description: str | None = None,
) -> str:
    """Render all 20 tuned stories per model beside control and reference prose."""
    samples = {
        (variant.variant_id, sample.job.route.route_id, sample.job.brief.brief_id): sample
        for variant, values in sample_cells
        for sample in values
    }
    if len(variants) != 2 or tuple(variant.paid for variant in variants) != (
        False,
        True,
    ):
        raise ValueError("review requires one control and one paid prompt variant")
    control_id, tuned_id = (variant.variant_id for variant in variants)
    plain_prompt_profile = (
        variants[1].request_contract.story_prompt_profile
        if variants[1].request_contract.plain_text_story_response
        else None
    )
    historical_v1 = tuple(variant.variant_id for variant in variants) == tuple(
        variant.variant_id for variant in PROMPT_TUNING_V1_VARIANTS
    )
    tuned_heading = "Tuned V6" if historical_v1 else f"New {tuned_id}"
    control_heading = "Old V4 control" if historical_v1 else f"Cached {control_id}"
    reference_heading = (
        "Matched genuine reference"
        if historical_v1
        else "Matched archive reference (human context only; not scored)"
    )

    def story_markup(sample: GeneratedSample) -> str:
        if sample.payload is None:
            reasons = ", ".join(sample.validation.rejection_reasons) or sample.error_kind
            return f'<p class="rejected">Rejected: {escape(reasons or "unknown error")}</p>'
        status = "accepted" if sample.validation.accepted else "rejected"
        reasons = ", ".join(sample.validation.rejection_reasons)
        note = "" if sample.validation.accepted else f"<b>Rejected: {escape(reasons)}</b>"
        return f'<p class="story {status}">{note}{escape(sample.payload.story)}</p>'

    sections = []
    for route_id in TWO_ROUTE_AUTHOR_ORDER:
        cards = []
        for index, brief in enumerate(selected, start=1):
            control = samples[(control_id, route_id, brief.brief_id)]
            tuned = samples[(tuned_id, route_id, brief.brief_id)]
            shown_prompt = (
                plain_story_user_prompt(brief, plain_prompt_profile)
                if plain_prompt_profile is not None
                else brief.prompt_text
            )
            prompt_summary = (
                "Exact one-message prompt"
                if plain_prompt_profile is not None
                else "Released instruction"
            )
            cards.append(
                "".join(
                    (
                        f'<article id="{escape(route_id)}-{index}">',
                        f"<h3>{index}. {escape(brief.brief_id)}</h3>",
                        f'<details><summary>{prompt_summary}</summary><p class="prompt">{escape(shown_prompt)}</p></details>',
                        '<div class="columns">',
                        f'<section><h4>{escape(tuned_heading)}</h4>{story_markup(tuned)}</section>',
                        f'<section><h4>{escape(control_heading)}</h4>{story_markup(control)}</section>',
                        f'<section><h4>{escape(reference_heading)}</h4><p class="story reference">{escape(brief.matched_reference_text)}</p></section>',
                        "</div></article>",
                    )
                )
            )
        sections.append(
            f'<section class="route"><h2>{escape(route_id)} — 20 '
            f'{"tuned" if historical_v1 else "new"} outputs</h2>'
            + "".join(cards)
            + "</section>"
        )
    return "".join(
        (
            "<!doctype html><html><head><meta charset=\"utf-8\">",
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">",
            "<title>TinyWorlds-v2 prompt tuning review</title>",
            "<style>",
            "body{font:15px/1.45 system-ui,sans-serif;max-width:1500px;margin:auto;padding:24px;color:#202124}",
            "h1,h2{position:sticky;top:0;background:#fff;padding:.4rem 0;z-index:2}",
            "article{border-top:2px solid #ddd;padding:12px 0 24px}",
            ".columns{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}",
            ".columns section{background:#f7f7f8;border-radius:8px;padding:12px}",
            ".story{white-space:pre-wrap}.prompt{white-space:pre-wrap;color:#444}",
            ".rejected{border-left:4px solid #c33;padding-left:8px;color:#8b1a1a}",
            ".reference{background:#eef7ee;padding:8px}.route{margin-top:32px}",
            "@media(max-width:900px){.columns{grid-template-columns:1fr}}",
            "</style></head><body>",
            "<h1>TinyWorlds-v2 tuned-prompt review</h1>",
            (
                "<p>Twenty identical development briefs per model. The tuned "
                "output is shown first, followed by the cached old prompt and "
                "its matched genuine TinyStories reference. This is a review "
                "set, not a Phase 1 qualification set.</p>"
                if historical_v1
                else (
                    "<p>Twenty identical development briefs per model. The new "
                    "output is shown first, followed by the cached prior prompt "
                    "and a matched archive reference for human context only. "
                    "Those matched archive references are not used by any scorer "
                    "because they may overlap TinyStories-8M training. Automated "
                    "quality uses "
                    + (
                        escape(comparison_description)
                        if comparison_description is not None
                        else (
                            "the clean 10,000-story official GPT-4 validation "
                            "profile <code>"
                            f"{PROMPT_TUNING_CLEAN_COMPARATOR_PROFILE_SHA256}</code>"
                        )
                    )
                    + ". This is a review set, not a Phase 1 qualification set.</p>"
                )
            ),
            (
                "<p>The new cell uses one user message containing only the "
                "released instruction"
                + (
                    ", the single cue <code>Aim for about 130 to 150 words.</code>,"
                    if plain_prompt_profile == "released-prompt-length-cue-v1"
                    else ""
                )
                + " and <code>Possible story:</code>. It has "
                "no system message, repeated instructions, JSON request, or "
                "response schema.</p>"
                if plain_prompt_profile is not None
                else ""
            ),
            *sections,
            "</body></html>",
        )
    )


def _write_plan(
    builder: Phase1ArtifactBuilder,
    references: TwoRouteReferenceEvidence,
    selected: Sequence[NeutralStoryBrief],
    payloads: CatalogPayloads,
    baseline_routes: Sequence[RouteLock],
    live_routes: Sequence[RouteLock],
    cells: Sequence[VariantJobs],
    preflight: CostPreflight,
    experiment: PromptTuningExperiment,
    quality_comparator: ReferenceProfile | None,
) -> None:
    if references.base_manifest_sha256 != TWO_ROUTE_BASE_REFERENCE_MANIFEST_SHA256:
        raise ValueError("prompt tuning base-reference identity differs")
    builder.write_json("configuration.json", _configuration_record(experiment))
    if experiment.version != PROMPT_TUNING_V1_VERSION:
        if quality_comparator is None:
            raise ValueError("prompt tuning requires its versioned comparator")
        builder.write_json(
            "quality_comparator.json",
            {
                "profile": _reference_profile_record(quality_comparator),
                "provenance": _quality_comparator_provenance(experiment),
            },
        )
    elif quality_comparator is not None:
        raise ValueError("V1 prompt tuning cannot replace its historical comparator")
    builder.write_bytes(
        "selected_briefs.jsonl",
        canonical_jsonl_bytes(_brief_record(brief) for brief in selected),
    )
    builder.write_bytes("catalog/models.response", payloads.models)
    for model_id, response in payloads.endpoints:
        route_id = next(
            model.route_id
            for model in TWO_ROUTE_AUTHOR_MODELS
            if model.request_model_id == model_id
        )
        builder.write_bytes(f"catalog/endpoints/{route_id}.response", response)
    builder.write_json(
        "catalog/routes.json",
        {
            "baseline": [route.as_record() for route in baseline_routes],
            "live": [route.as_record() for route in live_routes],
            "snapshot_sha256": payloads.snapshot_sha256,
        },
    )
    for cell in cells:
        builder.write_bytes(
            f"plans/{cell.variant.variant_id}.jsonl",
            canonical_jsonl_bytes(
                {
                    **job.request.as_record(),
                    "body": job.request.body,
                    "brief_id": job.brief.brief_id,
                    "route_id": job.route.route_id,
                    "variant_id": cell.variant.variant_id,
                }
                for job in cell.jobs
            ),
        )
    builder.write_json("cost_estimates.json", preflight.as_record())


def _write_cost_actuals(
    builder: Phase1ArtifactBuilder,
    generated_cells: Sequence[tuple[PromptVariant, tuple[GeneratedSample, ...]]],
    client: object,
) -> None:
    records = []
    for variant, samples in generated_cells:
        for route_id in TWO_ROUTE_AUTHOR_ORDER:
            route_samples = tuple(
                sample for sample in samples if sample.job.route.route_id == route_id
            )
            records.append(
                {
                    "accepted_count": sum(sample.validation.accepted for sample in route_samples),
                    "actual_billed_usd": float(
                        sum(
                            (Decimal(sample.billed_cost_usd) for sample in route_samples),
                            Decimal(0),
                        )
                    ),
                    "request_count": len(route_samples),
                    "route_id": route_id,
                    "variant_id": variant.variant_id,
                }
            )
    builder.write_json(
        "cost_actuals.json",
        {
            "actual_billed_usd": sum(record["actual_billed_usd"] for record in records),
            "cells": records,
            "runtime_ledger": (
                client.cost_ledger.snapshot().as_record()
                if type(client) is OpenRouterClient
                else None
            ),
        },
    )


def _write_raw_evidence(
    builder: Phase1ArtifactBuilder,
    cache: ImmutableRawCache,
    cells: Sequence[VariantJobs],
) -> None:
    for job in (job for cell in cells for job in cell.jobs):
        source = cache.root / job.request.request_sha256
        if not source.is_dir():
            raise ValueError("completed prompt-tuning job lacks immutable raw evidence")
        for path in sorted(item for item in source.rglob("*") if item.is_file()):
            builder.write_bytes(
                "raw_cache/requests/"
                f"{job.request.request_sha256}/{path.relative_to(source).as_posix()}",
                path.read_bytes(),
            )
    journal = cache.root / "runtime-cost-journal"
    if journal.is_dir():
        for path in sorted(item for item in journal.rglob("*") if item.is_file()):
            builder.write_bytes(
                f"raw_cache/runtime-cost-journal/{path.relative_to(journal).as_posix()}",
                path.read_bytes(),
            )


def _variant_sample_record(variant_id: str, sample: GeneratedSample) -> JsonObject:
    return {**sample.as_record(), "variant_id": variant_id}


def _variant_record(variant: PromptVariant) -> JsonObject:
    record: JsonObject = {
        "paid": variant.paid,
        "request_contract_version": variant.request_contract.version,
        "story_prompt_profile": variant.request_contract.story_prompt_profile,
        "variant_id": variant.variant_id,
    }
    if variant.request_contract.plain_text_story_response:
        record["prompt_envelope"] = (
            "released-prompt-plus-length-cue-plus-possible-story"
            if variant.request_contract.story_prompt_profile
            == "released-prompt-length-cue-v1"
            else "released-prompt-plus-possible-story-only"
        )
        record["response_format"] = "plain-assistant-text"
    return record


def _configuration_record(experiment: PromptTuningExperiment) -> JsonObject:
    record: JsonObject = {
        "base_reference_manifest_sha256": TWO_ROUTE_BASE_REFERENCE_MANIFEST_SHA256,
        "brief_count": PROMPT_TUNING_BRIEF_COUNT,
        "development_data_only": True,
        "generation_workers": PROMPT_TUNING_GENERATION_WORKERS,
        "hard_cap_usd": PROMPT_TUNING_HARD_CAP_USD,
        "paid_request_count": (
            PROMPT_TUNING_BRIEF_COUNT
            * len(TWO_ROUTE_AUTHOR_ORDER)
            * sum(variant.paid for variant in experiment.variants)
        ),
        "selection_namespace": PROMPT_TUNING_SELECTION_NAMESPACE,
        "variants": [_variant_record(variant) for variant in experiment.variants],
        "version": experiment.version,
    }
    if experiment.control_source_kind == "two-route-v2":
        record["baseline_manifest_sha256"] = (
            experiment.control_source_manifest_sha256
        )
    else:
        record["control_source_kind"] = experiment.control_source_kind
        record["control_source_manifest_sha256"] = (
            experiment.control_source_manifest_sha256
        )
        record["control_source_variant_id"] = experiment.control_source_variant_id
        record["quality_comparator"] = _quality_comparator_provenance(experiment)
    return record


def _quality_comparator_provenance(
    experiment: PromptTuningExperiment,
) -> JsonObject:
    if experiment.version == PROMPT_TUNING_V2_VERSION:
        return _clean_comparator_provenance()
    if experiment.version in {
        PROMPT_TUNING_V4_VERSION,
        PROMPT_TUNING_V5_VERSION,
    }:
        return _decontaminated_comparator_provenance()
    raise ValueError("this prompt-tuning experiment has no quality comparator")


def _clean_comparator_provenance() -> JsonObject:
    return {
        "artifact_manifest_sha256": TWO_ROUTE_BASE_REFERENCE_MANIFEST_SHA256,
        "human_matched_references_used_for_scoring": False,
        "observation_artifact_path": PROMPT_TUNING_CLEAN_COMPARATOR_PATH,
        "observation_artifact_sha256": (
            PROMPT_TUNING_CLEAN_COMPARATOR_SOURCE_SHA256
        ),
        "record_count": PROMPT_TUNING_CLEAN_COMPARATOR_RECORD_COUNT,
        "record_id_prefix": PROMPT_TUNING_CLEAN_COMPARATOR_ID_PREFIX,
        "reference_profile_sha256": PROMPT_TUNING_CLEAN_COMPARATOR_PROFILE_SHA256,
        "source_dataset_id": "roneneldan/TinyStories",
        "source_file_sha256": (
            "6874bae9a4c1a4e7edcf0e53b86c17817e9cf881fc75ff2368da457b80c0585d"
        ),
        "source_file_size_bytes": 22_502_601,
        "source_filename": "TinyStoriesV2-GPT4-valid.txt",
        "source_revision": "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64",
    }


def _decontaminated_comparator_provenance() -> JsonObject:
    return {
        "artifact_manifest_sha256": PROMPT_REEVALUATION_V3_MANIFEST_SHA256,
        "artifact_path": "data/tinyworlds-v2/prompt-tuning-v3",
        "human_matched_references_used_for_scoring": False,
        "identity_policy": (
            "unicode-nfkc-casefold-whitespace-collapse-sha256-with-full-text-"
            "confirmation-v1"
        ),
        "record_count": PROMPT_REEVALUATION_V3_RECORD_COUNT,
        "reference_profile_sha256": PROMPT_REEVALUATION_V3_PROFILE_SHA256,
        "source_dataset_id": "roneneldan/TinyStories",
        "source_filename": "TinyStoriesV2-GPT4-valid.txt",
        "source_revision": "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64",
        "training_overlap_excluded": True,
    }


def _quality_interpretation(variants: Sequence[PromptVariant]) -> str:
    if tuple(variant.variant_id for variant in variants) == tuple(
        variant.variant_id for variant in PROMPT_TUNING_V1_VARIANTS
    ):
        return (
            "development diagnostic only; the 20 examples may guide prompt "
            "selection but cannot qualify a route or pass Phase 1"
        )
    decontaminated_variants = {
        tuple(variant.variant_id for variant in experiment_variants)
        for experiment_variants in (
            PROMPT_TUNING_V4_VARIANTS,
            PROMPT_TUNING_V5_VARIANTS,
        )
    }
    if tuple(variant.variant_id for variant in variants) in decontaminated_variants:
        return (
            "development diagnostic only; automated quality compares against "
            "the train-decontaminated 6,607-story GPT-4 validation profile, "
            "while matched archive references are human-review context only; "
            "this cannot pass Phase 1"
        )
    return (
        "development diagnostic only; automated quality compares against the "
        "clean 10,000-story GPT-4 validation profile, while matched archive "
        "references are human-review context only; this cannot pass Phase 1"
    )


def _brief_record(brief: NeutralStoryBrief) -> JsonObject:
    return {
        "brief_id": brief.brief_id,
        "matched_reference_text": brief.matched_reference_text,
        "prompt_text": brief.prompt_text,
        "requested_features": list(brief.requested_features),
        "required_words": list(brief.required_words),
        "source_record_id": brief.source_record_id,
    }


def _record_progress(builder: Phase1ArtifactBuilder, event: str) -> None:
    builder.append_jsonl("progress.jsonl", {"event": event})


def _accepted(record: JsonObject) -> bool:
    validation = record.get("validation")
    if type(validation) is not dict or type(validation.get("accepted")) is not bool:
        raise ValueError("prompt-tuning result lacks an acceptance flag")
    return validation["accepted"]


def _json_object(path: Path) -> JsonObject:
    return require_json_object(
        canonical_json_loads(path.read_bytes(), label=path.name),
        label=path.name,
    )


def _jsonl_objects(path: Path) -> tuple[JsonObject, ...]:
    payload = path.read_bytes()
    if payload and not payload.endswith(b"\n"):
        raise ValueError(f"JSONL artifact has invalid framing: {path}")
    return tuple(
        require_json_object(
            canonical_json_loads(line, label=f"{path.name} line {index}"),
            label=f"{path.name} line {index}",
        )
        for index, line in enumerate(payload.splitlines(), start=1)
    )


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be nonempty text")
    return value


def _optional_text(value: object) -> str | None:
    return None if value is None else _text(value, "optional text")


def _integer(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _boolean(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be boolean")
    return value


def _list(value: object, label: str) -> list[object]:
    if type(value) is not list:
        raise ValueError(f"{label} must be an array")
    return value


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    return tuple(_text(item, label) for item in _list(value, label))


__all__ = [
    "PROMPT_TUNING_BRIEF_COUNT",
    "PROMPT_TUNING_EXPERIMENTS",
    "PROMPT_TUNING_HARD_CAP_USD",
    "PROMPT_TUNING_V1_EXPERIMENT",
    "PROMPT_TUNING_V1_MANIFEST_SHA256",
    "PROMPT_TUNING_V1_VARIANTS",
    "PROMPT_TUNING_V1_VERSION",
    "PROMPT_TUNING_V2_EXPERIMENT",
    "PROMPT_TUNING_V2_MANIFEST_SHA256",
    "PROMPT_TUNING_V2_VARIANTS",
    "PROMPT_TUNING_V2_VERSION",
    "PROMPT_TUNING_V4_EXPERIMENT",
    "PROMPT_TUNING_V4_MANIFEST_SHA256",
    "PROMPT_TUNING_V4_VARIANTS",
    "PROMPT_TUNING_V4_VERSION",
    "PROMPT_TUNING_V5_EXPERIMENT",
    "PROMPT_TUNING_V5_MANIFEST_SHA256",
    "PROMPT_TUNING_V5_VARIANTS",
    "PROMPT_TUNING_V5_VERSION",
    "PromptTuningExperiment",
    "PromptTuningPaths",
    "PromptTuningRunResult",
    "PromptVariant",
    "VariantJobs",
    "build_prompt_tuning_cost_preflight",
    "build_prompt_tuning_jobs",
    "main",
    "run_prompt_tuning",
    "select_prompt_tuning_briefs",
    "validate_prompt_tuning",
]

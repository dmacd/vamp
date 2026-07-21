"""Direct Qwen/GPT-5.4-Mini TinyStories author comparison.

This is the active, simplified TinyWorlds-v2 Phase 1 experiment.  It reuses
the already authenticated reference profile, sends the same 200 neutral briefs
to both pinned author routes, derives all hard evidence locally, measures the
accepted prose with the frozen TinyStories tokenizer/checkpoint, and stops at
one balanced blinded human audit.  The historical seven-route funnel remains
sealed in :mod:`phase1_runner` for artifact replay only.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal
import math
from pathlib import Path
from statistics import median
import tempfile

from apm.data.text.tinyworlds_v2.audit import (
    AuditAllocationError,
    AuditSourceKind,
    AuditSourceRecord,
    BlindedAuditKey,
    BlindedAuditPacket,
    build_blinded_audit,
    render_audit_html,
)
from apm.data.text.tinyworlds_v2.audit_io import (
    decode_audit_pair,
    encode_blinded_audit_key,
    encode_blinded_audit_packet,
)
from apm.data.text.tinyworlds_v2.bakeoff import (
    SYNTHETIC_STORY_REQUEST_V4,
    TWO_ROUTE_AUTHOR_MODELS,
    CandidateModelSpec,
    NeutralStoryBrief,
    StoryOnlyPayload,
    validate_story_only_generated_story,
)
from apm.data.text.tinyworlds_v2.byok_contract import canonical_byok_authorization
from apm.data.text.tinyworlds_v2.catalog import (
    CatalogPayloads,
    PHASE1_PROMPT_TOKEN_UPPER_BOUND,
    resolve_openrouter_routes,
)
from apm.data.text.tinyworlds_v2.generation_cache import ImmutableRawCache
from apm.data.text.tinyworlds_v2.generation_costs import (
    PHASE1_HARD_CAP_USD,
    CostPreflight,
    RouteWorkload,
    TokenWorkload,
    build_cost_preflight,
    enforce_cost_cap,
    exclusive_paid_run_lock,
)
from apm.data.text.tinyworlds_v2.generation_schema import RouteLock
from apm.data.text.tinyworlds_v2.httpx_transport import (
    HttpxTransport,
    fetch_catalog_payloads,
)
from apm.data.text.tinyworlds_v2.json_contracts import (
    JsonObject,
    canonical_json_loads,
    require_exact_fields,
    require_json_object,
)
from apm.data.text.tinyworlds_v2.openrouter import OpenRouterClient, RetryPolicy
from apm.data.text.tinyworlds_v2.phase1_artifacts import (
    Phase1ArtifactBuilder,
    Phase1ArtifactManifest,
    canonical_jsonl_bytes,
    load_phase1_artifact_tree,
)
from apm.data.text.tinyworlds_v2.phase1_generation import (
    CachedGenerationClient,
    GenerationJob,
    GeneratedSample,
    build_generation_jobs,
    execute_generation_jobs,
    generated_observation,
)
from apm.data.text.tinyworlds_v2.phase1_replay import (
    _load_briefs,
    _load_quality_reference_evidence,
    _route,
)
from apm.data.text.tinyworlds_v2.phase1_runner import (
    MeasurementBatch,
    Phase1Paths,
    _quality_report_record,
    production_dependencies,
)
from apm.data.text.tinyworlds_v2.quality import (
    TWO_ROUTE_AUTHOR_ORDER,
    QualityOutcome,
    QualityPhase,
    QualitySelection,
    RouteQualityReport,
    evaluate_route_quality,
    select_direct_quality_routes,
)
from apm.data.text.tinyworlds_v2.reference_profile import (
    ReferenceObservation,
    ReferenceProfile,
)
from apm.data.text.tinyworlds_v2.reference_runtime import NllStory
from apm.data.text.tinyworlds_v2.route_lock import validate_route_semantics
from apm.data.text.tinyworlds_v2.surface import (
    lexical_tokens,
    repeated_ngram_fraction,
)


TWO_ROUTE_VERSION = "tinyworlds-v2-phase1-two-route-v2"
TWO_ROUTE_VALIDATOR_VERSION = "tinyworlds-v2-local-story-validator-v5"
TWO_ROUTE_REQUEST_COUNT = 200
TWO_ROUTE_AUDIT_COUNT = 100
TWO_ROUTE_GENERATION_WORKERS = 8
TWO_ROUTE_RETRY_ALLOWANCE_BASIS_POINTS = 10_000
TWO_ROUTE_PROJECTED_CORPUS_STORIES = 4_000
TWO_ROUTE_BASE_REFERENCE_MANIFEST_SHA256 = (
    "28a1280c256d8a6ecfc5e4048e65f71e5839c522e391eb03dd07b1669a66d5e9"
)


@dataclass(frozen=True, slots=True)
class TwoRoutePaths:
    """Fixed production paths for the direct two-author comparison."""

    repository_root: Path
    base_reference: Path
    raw_cache: Path
    destination: Path

    @classmethod
    def from_repository(cls, repository_root: str | Path) -> "TwoRoutePaths":
        root = Path(repository_root).resolve()
        return cls(
            repository_root=root,
            base_reference=root / "data" / "tinyworlds-v2" / "reference",
            raw_cache=(
                root
                / "data"
                / "tinyworlds-v2"
                / "cache"
                / "phase1-openrouter-two-route-v1"
            ),
            destination=(
                root / "data" / "tinyworlds-v2" / "reference-two-route-v2"
            ),
        )


@dataclass(frozen=True, slots=True)
class TwoRouteReferenceEvidence:
    """Only the frozen reference evidence needed by the direct comparison."""

    briefs: tuple[NeutralStoryBrief, ...]
    reference_profile: ReferenceProfile
    paired_observations: tuple[ReferenceObservation, ...]
    paired_profile: ReferenceProfile
    expected_feature_rates: tuple[tuple[str, float], ...]
    base_manifest_sha256: str

    def __post_init__(self) -> None:
        if len(self.briefs) != TWO_ROUTE_REQUEST_COUNT:
            raise ValueError("two-route comparison requires exactly 200 briefs")
        if len(self.paired_observations) != TWO_ROUTE_REQUEST_COUNT:
            raise ValueError("two-route comparison requires exactly 200 references")
        if self.base_manifest_sha256 != TWO_ROUTE_BASE_REFERENCE_MANIFEST_SHA256:
            raise ValueError("two-route comparison reference manifest drifted")


FetchRoutes = Callable[[], tuple[CatalogPayloads, tuple[RouteLock, ...]]]
EncodeText = Callable[[str], tuple[int, ...]]
MeasureStories = Callable[
    [tuple[NllStory, ...], Callable[[int], None]], MeasurementBatch
]
LoadApiKey = Callable[[], str]
MakeClient = Callable[[str, ImmutableRawCache], CachedGenerationClient]
RevalidateRoute = Callable[[RouteLock], RouteLock]


@dataclass(frozen=True, slots=True)
class TwoRouteDependencies:
    """Injected network, tokenizer, checkpoint, and secret boundaries."""

    fetch_routes: FetchRoutes
    encode_text: EncodeText
    measure_stories: MeasureStories
    load_api_key: LoadApiKey
    make_client: MakeClient
    revalidate_route: RevalidateRoute


@dataclass(frozen=True, slots=True)
class TwoRouteRunResult:
    """One promoted comparison or an explicit preflight stop."""

    directory: Path
    status: str
    conservative_cost_usd: str
    qualified_route_ids: tuple[str, ...]
    audit_sha256: str | None


def load_two_route_reference_evidence(
    base_reference: str | Path,
) -> TwoRouteReferenceEvidence:
    """Strictly reuse the completed reference profile without source/GPU work."""
    root = Path(base_reference)
    manifest = load_phase1_artifact_tree(root)
    if manifest.manifest_sha256 != TWO_ROUTE_BASE_REFERENCE_MANIFEST_SHA256:
        raise ValueError(
            "the active two-route run requires the exact completed reference artifact"
        )
    briefs = _load_briefs(root / "neutral_story_briefs.jsonl")
    quality = _load_quality_reference_evidence(root)
    return TwoRouteReferenceEvidence(
        briefs=briefs,
        reference_profile=quality.reference_profile,
        paired_observations=quality.paired_observations,
        paired_profile=quality.paired_profile,
        expected_feature_rates=quality.expected_feature_rates,
        base_manifest_sha256=manifest.manifest_sha256,
    )


def production_two_route_dependencies(
    paths: TwoRoutePaths,
) -> TwoRouteDependencies:
    """Construct the fixed direct-OpenRouter/GPU production boundaries."""
    legacy_paths = replace(
        Phase1Paths.from_repository(paths.repository_root),
        raw_cache=paths.raw_cache,
        destination=paths.destination,
    )
    base = production_dependencies(legacy_paths)
    transport = HttpxTransport()

    def fetch() -> tuple[CatalogPayloads, tuple[RouteLock, ...]]:
        payloads = fetch_catalog_payloads(
            transport,
            model_specs=TWO_ROUTE_AUTHOR_MODELS,
        )
        return payloads, resolve_openrouter_routes(
            payloads,
            TWO_ROUTE_AUTHOR_MODELS,
        )

    def revalidate(locked: RouteLock) -> RouteLock:
        _payloads, routes = fetch()
        observed = next(
            (route for route in routes if route.route_id == locked.route_id),
            None,
        )
        if observed is None:
            raise ValueError(f"fresh catalog omitted route {locked.route_id!r}")
        validate_route_semantics(locked, observed)
        return observed

    return TwoRouteDependencies(
        fetch_routes=fetch,
        encode_text=base.encode_text,
        measure_stories=base.measure_stories,
        load_api_key=base.load_api_key,
        make_client=base.make_client,
        revalidate_route=revalidate,
    )


def prepare_two_route_bakeoff(
    builder: Phase1ArtifactBuilder,
    references: TwoRouteReferenceEvidence,
    dependencies: TwoRouteDependencies,
) -> tuple[
    CatalogPayloads,
    tuple[RouteLock, ...],
    tuple[GenerationJob, ...],
    CostPreflight,
]:
    """Resolve routes and persist the exact 400-request preflight."""
    payloads, routes = dependencies.fetch_routes()
    if tuple(route.route_id for route in routes) != TWO_ROUTE_AUTHOR_ORDER:
        raise ValueError("live routes changed the fixed two-author order")
    jobs = build_generation_jobs(
        references.briefs,
        TWO_ROUTE_AUTHOR_MODELS,
        routes,
        request_contract=SYNTHETIC_STORY_REQUEST_V4,
    )
    preflight = build_two_route_cost_preflight(
        references,
        jobs,
        routes,
        dependencies.encode_text,
    )
    _write_reference_binding(builder, references)
    _write_catalog(builder, payloads, routes)
    _write_configuration(builder, references, routes)
    _write_plans(builder, jobs, routes)
    builder.write_json("cost_estimates.json", preflight.as_record())
    return payloads, routes, jobs, preflight


def build_two_route_cost_preflight(
    references: TwoRouteReferenceEvidence,
    jobs: Sequence[GenerationJob],
    routes: Sequence[RouteLock],
    encode_text: EncodeText,
) -> CostPreflight:
    """Budget exactly 200 calls per author, including one retry exposure."""
    expected_output = math.ceil(
        sum(len(encode_text(brief.matched_reference_text)) for brief in references.briefs)
        / len(references.briefs)
    )
    workloads: list[RouteWorkload] = []
    for route in routes:
        route_jobs = tuple(job for job in jobs if job.route.route_id == route.route_id)
        if len(route_jobs) != TWO_ROUTE_REQUEST_COUNT:
            raise ValueError("each two-route author requires exactly 200 jobs")
        input_counts = tuple(
            len(encode_text(job.request.body_json)) for job in route_jobs
        )
        conservative_input = max(2 * count + 512 for count in input_counts)
        if conservative_input > PHASE1_PROMPT_TOKEN_UPPER_BOUND:
            raise ValueError("a two-route request exceeds the prompt-token bound")
        workloads.append(
            RouteWorkload(
                route,
                TokenWorkload(
                    label="phase1-direct-author-200",
                    request_count=TWO_ROUTE_REQUEST_COUNT,
                    input_tokens_per_request=math.ceil(
                        sum(input_counts) / len(input_counts)
                    ),
                    output_tokens_per_request=expected_output,
                    conservative_input_tokens_per_request=conservative_input,
                    conservative_output_tokens_per_request=512,
                    retry_allowance_basis_points=(
                        TWO_ROUTE_RETRY_ALLOWANCE_BASIS_POINTS
                    ),
                ),
            )
        )
    return build_cost_preflight(
        tuple(workloads),
        hard_cap_usd=PHASE1_HARD_CAP_USD,
    )


def run_two_route_bakeoff(
    staging_directory: str | Path,
    paths: TwoRoutePaths,
    dependencies: TwoRouteDependencies,
    *,
    emit: Callable[[str], None] = print,
) -> TwoRouteRunResult:
    """Run the fixed direct comparison while holding its paid-cache lease."""
    with exclusive_paid_run_lock(paths.raw_cache):
        return _run_two_route_bakeoff(
            staging_directory,
            paths,
            dependencies,
            emit=emit,
        )


def _run_two_route_bakeoff(
    staging_directory: str | Path,
    paths: TwoRoutePaths,
    dependencies: TwoRouteDependencies,
    *,
    emit: Callable[[str], None],
) -> TwoRouteRunResult:
    staging = Path(staging_directory)
    if not staging.is_dir():
        raise FileNotFoundError(f"two-route staging directory is missing: {staging}")
    if paths.destination.exists() or paths.destination.is_symlink():
        raise FileExistsError(f"two-route destination exists: {paths.destination}")
    builder = Phase1ArtifactBuilder(staging, version=TWO_ROUTE_VERSION)
    references = load_two_route_reference_evidence(paths.base_reference)
    emit("Two-route phase 1: reused the completed reference profile (no corpus scan).")
    _payloads, routes, jobs, preflight = prepare_two_route_bakeoff(
        builder,
        references,
        dependencies,
    )
    emit(
        "Two-route cost preflight: "
        f"expected ${preflight.expected_usd}; conservative "
        f"${preflight.conservative_usd} / ${preflight.hard_cap_usd} cap"
    )
    for estimate in preflight.route_estimates:
        emit(
            f"  {estimate.route_id}: {estimate.request_count} requests, "
            f"expected ${estimate.expected_usd}, reserve ${estimate.conservative_usd}"
        )
    cache = ImmutableRawCache(paths.raw_cache)
    if not preflight.permitted:
        _write_empty_outcome(builder, "blocked_by_cost_cap")
        return _finalize_result(
            builder,
            paths,
            preflight,
            status="blocked_by_cost_cap",
            qualified=(),
            audit_sha256=None,
        )

    enforce_cost_cap(preflight)
    # This is intentionally the first inference-secret read.
    client = dependencies.make_client(dependencies.load_api_key(), cache)
    if type(client) is OpenRouterClient:
        client.cost_ledger.bootstrap(cache, routes)
        if client.require_byok_preflight:
            evidence = client.verify_no_byok()
            builder.write_json("byok_preflight.json", evidence.as_record())

    samples: list[GeneratedSample] = []
    for model, route in zip(TWO_ROUTE_AUTHOR_MODELS, routes, strict=True):
        emit(f"Generating 200 paired stories with {model.route_id}...")
        fresh_route = dependencies.revalidate_route(route)
        validate_route_semantics(route, fresh_route)
        route_jobs = tuple(
            replace(job, route=fresh_route)
            for job in jobs
            if job.route.route_id == route.route_id
        )
        batch = execute_generation_jobs(
            route_jobs,
            client,
            max_workers=TWO_ROUTE_GENERATION_WORKERS,
        )
        samples.extend(batch)
        _write_route_results(builder, route.route_id, batch)
        emit(
            f"  {route.route_id}: {sum(item.validation.accepted for item in batch)}/"
            f"{len(batch)} locally accepted"
        )

    complete_samples = tuple(samples)
    if len(complete_samples) != TWO_ROUTE_REQUEST_COUNT * len(routes):
        raise ValueError("two-route execution did not return all 400 observations")
    emit("Measuring accepted stories with the frozen TinyStories-8M model...")
    measurements = _measure_samples(complete_samples, dependencies.measure_stories)
    _write_measurements(builder, measurements)
    reports = _evaluate_routes(complete_samples, measurements, references)
    selection = select_direct_quality_routes(reports)
    _write_quality(builder, reports, selection)
    audit = _try_write_audit(builder, references, complete_samples, measurements)
    status = (
        "audit_insufficient_accepted_samples"
        if audit is None
        else "awaiting_human_audit"
        if selection.outcome is QualityOutcome.QUALITY_QUALIFIED_ROUTES
        else "no_quality_qualified_route"
    )
    audit_sha256 = None if audit is None else audit[0].audit_sha256
    _write_cost_actuals(builder, complete_samples, client)
    _write_raw_evidence(builder, cache, jobs)
    builder.write_json(
        "status.json",
        {
            "audit_sha256": audit_sha256,
            "phase": 1,
            "status": status,
        },
    )
    emit(f"Two-route comparison status: {status}")
    return _finalize_result(
        builder,
        paths,
        preflight,
        status=status,
        qualified=selection.route_ids,
        audit_sha256=audit_sha256,
    )


def validate_two_route_bakeoff(
    directory: str | Path,
) -> Phase1ArtifactManifest:
    """Authenticate the active artifact and its exact direct-comparison contract."""
    root = Path(directory)
    manifest = load_phase1_artifact_tree(root)
    if manifest.version != TWO_ROUTE_VERSION:
        raise ValueError("artifact is not the active two-route comparison")
    configuration = _json_object(root / "configuration.json")
    require_exact_fields(
        configuration,
        (
            "audit_count",
            "author_models",
            "base_reference_manifest_sha256",
            "comparison_policy",
            "generation_request_count",
            "generation_workers",
            "hard_cap_usd",
            "projected_accepted_story_count",
            "request_contract_version",
            "route_count",
            "story_prompt_version",
            "story_schema_version",
            "validator_version",
            "version",
        ),
        label="two-route configuration",
    )
    expected = {
        "audit_count": TWO_ROUTE_AUDIT_COUNT,
        "author_models": [_model_record(model) for model in TWO_ROUTE_AUTHOR_MODELS],
        "base_reference_manifest_sha256": TWO_ROUTE_BASE_REFERENCE_MANIFEST_SHA256,
        "comparison_policy": "paired-two-route-full-v1",
        "generation_request_count": TWO_ROUTE_REQUEST_COUNT * 2,
        "generation_workers": TWO_ROUTE_GENERATION_WORKERS,
        "hard_cap_usd": PHASE1_HARD_CAP_USD,
        "projected_accepted_story_count": TWO_ROUTE_PROJECTED_CORPUS_STORIES,
        "request_contract_version": SYNTHETIC_STORY_REQUEST_V4.version,
        "route_count": 2,
        "story_prompt_version": "tinyworlds-v2-neutral-story-v2",
        "story_schema_version": "tinyworlds_v2_neutral_story_v2",
        "validator_version": TWO_ROUTE_VALIDATOR_VERSION,
        "version": TWO_ROUTE_VERSION,
    }
    if configuration != expected:
        raise ValueError("two-route configuration differs from the fixed contract")
    reference = _json_object(root / "reference_binding.json")
    if reference != {
        "base_reference_manifest_sha256": TWO_ROUTE_BASE_REFERENCE_MANIFEST_SHA256,
        "brief_count": TWO_ROUTE_REQUEST_COUNT,
    }:
        raise ValueError("two-route reference binding differs")
    briefs = _load_briefs(root / "neutral_story_briefs.jsonl")
    route_record = _json_object(root / "catalog" / "routes.json")
    require_exact_fields(
        route_record,
        ("generator_routes", "snapshot_sha256"),
        label="two-route catalog routes",
    )
    route_values = route_record["generator_routes"]
    if type(route_values) is not list or len(route_values) != 2:
        raise ValueError("two-route catalog must contain exactly two author routes")
    routes = tuple(
        _route(value, f"two-route author {index}")
        for index, value in enumerate(route_values)
    )
    if tuple(route.route_id for route in routes) != TWO_ROUTE_AUTHOR_ORDER:
        raise ValueError("two-route catalog author order differs")
    catalog_payloads = CatalogPayloads(
        models=(root / "catalog" / "models.response").read_bytes(),
        endpoints=tuple(
            (
                model.request_model_id,
                (root / "catalog" / "endpoints" / f"{model.route_id}.response").read_bytes(),
            )
            for model in TWO_ROUTE_AUTHOR_MODELS
        ),
        model_plan_ids=tuple(
            model.request_model_id for model in TWO_ROUTE_AUTHOR_MODELS
        ),
    )
    if route_record["snapshot_sha256"] != catalog_payloads.snapshot_sha256:
        raise ValueError("two-route catalog snapshot digest differs from raw bytes")
    jobs = build_generation_jobs(
        briefs,
        TWO_ROUTE_AUTHOR_MODELS,
        routes,
        request_contract=SYNTHETIC_STORY_REQUEST_V4,
    )
    _validate_planned_jobs(root, jobs, routes)
    status = _json_object(root / "status.json")
    require_exact_fields(
        status,
        ("audit_sha256", "phase", "status"),
        label="two-route status",
    )
    if status["phase"] != 1:
        raise ValueError("two-route status phase differs")
    status_name = status.get("status")
    if status_name not in {
        "blocked_by_cost_cap",
        "awaiting_human_audit",
        "no_quality_qualified_route",
        "audit_insufficient_accepted_samples",
    }:
        raise ValueError("two-route status is unsupported")
    _validate_cost_estimates(root, routes)
    if status_name == "blocked_by_cost_cap":
        _validate_cost_stop(root, status)
    else:
        canonical_byok_authorization(_json_object(root / "byok_preflight.json"))
        _validate_completed_results(root, briefs, jobs, status)
    return manifest


def main() -> None:
    """Run the one active direct comparison and stop at its human audit."""
    repository_root = Path(__file__).resolve().parents[5]
    paths = TwoRoutePaths.from_repository(repository_root)
    if paths.destination.is_dir():
        validate_two_route_bakeoff(paths.destination)
        print(f"Existing two-route artifact: {paths.destination}")
        print("No source, GPU, catalog, or paid work was repeated.")
        return
    paths.destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix="tinyworlds-v2-two-route-",
            dir=paths.destination.parent,
        )
    )
    print(f"Temporary artifact directory: {staging}", flush=True)
    result = run_two_route_bakeoff(
        staging,
        paths,
        production_two_route_dependencies(paths),
    )
    print(f"Two-route status: {result.status}")
    print(f"Two-route artifact: {result.directory}")
    if result.audit_sha256 is not None:
        print(f"Mandatory audit digest: {result.audit_sha256}")
        print(f"Open the audit: {result.directory / 'audit.html'}")


def _write_reference_binding(
    builder: Phase1ArtifactBuilder,
    references: TwoRouteReferenceEvidence,
) -> None:
    builder.write_json(
        "reference_binding.json",
        {
            "base_reference_manifest_sha256": references.base_manifest_sha256,
            "brief_count": len(references.briefs),
        },
    )
    builder.write_bytes(
        "neutral_story_briefs.jsonl",
        canonical_jsonl_bytes(_brief_record(brief) for brief in references.briefs),
    )


def _write_catalog(
    builder: Phase1ArtifactBuilder,
    payloads: CatalogPayloads,
    routes: Sequence[RouteLock],
) -> None:
    builder.write_bytes("catalog/models.response", payloads.models)
    for model_id, payload in payloads.endpoints:
        route_id = next(
            model.route_id
            for model in TWO_ROUTE_AUTHOR_MODELS
            if model.request_model_id == model_id
        )
        builder.write_bytes(f"catalog/endpoints/{route_id}.response", payload)
    builder.write_json(
        "catalog/routes.json",
        {
            "generator_routes": [route.as_record() for route in routes],
            "snapshot_sha256": payloads.snapshot_sha256,
        },
    )


def _write_configuration(
    builder: Phase1ArtifactBuilder,
    references: TwoRouteReferenceEvidence,
    routes: Sequence[RouteLock],
) -> None:
    if len(routes) != 2:
        raise ValueError("two-route configuration requires two route locks")
    builder.write_json(
        "configuration.json",
        {
            "audit_count": TWO_ROUTE_AUDIT_COUNT,
            "author_models": [_model_record(model) for model in TWO_ROUTE_AUTHOR_MODELS],
            "base_reference_manifest_sha256": references.base_manifest_sha256,
            "comparison_policy": "paired-two-route-full-v1",
            "generation_request_count": TWO_ROUTE_REQUEST_COUNT * 2,
            "generation_workers": TWO_ROUTE_GENERATION_WORKERS,
            "hard_cap_usd": PHASE1_HARD_CAP_USD,
            "projected_accepted_story_count": TWO_ROUTE_PROJECTED_CORPUS_STORIES,
            "request_contract_version": SYNTHETIC_STORY_REQUEST_V4.version,
            "route_count": 2,
            "story_prompt_version": "tinyworlds-v2-neutral-story-v2",
            "story_schema_version": "tinyworlds_v2_neutral_story_v2",
            "validator_version": TWO_ROUTE_VALIDATOR_VERSION,
            "version": TWO_ROUTE_VERSION,
        },
    )


def _write_plans(
    builder: Phase1ArtifactBuilder,
    jobs: Sequence[GenerationJob],
    routes: Sequence[RouteLock],
) -> None:
    for route in routes:
        route_jobs = tuple(job for job in jobs if job.route.route_id == route.route_id)
        request_records = tuple(
            {
                **job.request.as_record(),
                "body": job.request.body,
                "brief_id": job.brief.brief_id,
                "route_id": route.route_id,
            }
            for job in route_jobs
        )
        builder.write_bytes(
            f"routes/{route.route_id}/requests.jsonl",
            canonical_jsonl_bytes(request_records),
        )
        builder.write_json(
            f"routes/{route.route_id}/plan.json",
            {
                "planned_request_count": len(route_jobs),
                "planned_request_sha256": [
                    job.request.request_sha256 for job in route_jobs
                ],
                "request_contract_version": SYNTHETIC_STORY_REQUEST_V4.version,
                "route": route.as_record(),
                "route_lock_sha256": route.lock_sha256,
                "validator_version": TWO_ROUTE_VALIDATOR_VERSION,
            },
        )


def _write_route_results(
    builder: Phase1ArtifactBuilder,
    route_id: str,
    samples: Sequence[GeneratedSample],
) -> None:
    for sample in samples:
        record = sample.as_record()
        builder.append_jsonl("generator_bakeoff.jsonl", record)
        builder.append_jsonl(
            f"routes/{route_id}/"
            f"{'accepted' if sample.validation.accepted else 'rejected'}.jsonl",
            record,
        )
    for filename in ("accepted.jsonl", "rejected.jsonl"):
        path = builder.root / "routes" / route_id / filename
        if not path.exists():
            builder.write_bytes(f"routes/{route_id}/{filename}", b"")


def _measure_samples(
    samples: Sequence[GeneratedSample],
    measure: MeasureStories,
) -> MeasurementBatch:
    stories = tuple(
        sorted(
            (
                NllStory(sample.sample_id, sample.payload.story)
                for sample in samples
                if sample.validation.accepted and sample.payload is not None
            ),
            key=lambda item: item.record_id,
        )
    )
    if not stories:
        return MeasurementBatch((), {"status": "no_accepted_stories"})
    return measure(stories, lambda _completed: None)


def _write_measurements(
    builder: Phase1ArtifactBuilder,
    measurements: MeasurementBatch,
) -> None:
    builder.write_bytes(
        "measurements/generation.jsonl",
        canonical_jsonl_bytes(item.as_record() for item in measurements.measurements),
    )
    builder.write_json("measurements/runtime.json", measurements.runtime)


def _evaluate_routes(
    samples: tuple[GeneratedSample, ...],
    measurements: MeasurementBatch,
    references: TwoRouteReferenceEvidence,
) -> tuple[RouteQualityReport, ...]:
    measurement_by_id = measurements.by_id
    reports: list[RouteQualityReport] = []
    for route_id in TWO_ROUTE_AUTHOR_ORDER:
        observations = tuple(
            replace(
                generated_observation(
                    sample,
                    model_token_ids=(
                        measurement_by_id[sample.sample_id].model_token_ids
                        if sample.sample_id in measurement_by_id
                        else ()
                    ),
                    normalized_nll=(
                        measurement_by_id[sample.sample_id].normalized_nll
                        if sample.sample_id in measurement_by_id
                        else None
                    ),
                ),
                sample_id=sample.job.brief.brief_id,
            )
            for sample in samples
            if sample.job.route.route_id == route_id
        )
        reports.append(
            evaluate_route_quality(
                observations,
                references.reference_profile,
                phase=QualityPhase.DIRECT,
                matched_reference_profile=references.paired_profile,
                expected_feature_rates=references.expected_feature_rates,
            )
        )
    return tuple(reports)


def _write_quality(
    builder: Phase1ArtifactBuilder,
    reports: Sequence[RouteQualityReport],
    selection: QualitySelection,
) -> None:
    builder.write_json(
        "quality_details.json",
        {
            "direct_reports": [_quality_report_record(report) for report in reports],
            "selection": {
                "outcome": selection.outcome.value,
                "reason": selection.reason,
                "route_ids": list(selection.route_ids),
            },
        },
    )
    builder.write_json(
        "quality_comparisons.json",
        {
            "audited_route_ids": list(TWO_ROUTE_AUTHOR_ORDER),
            "qualified_route_ids": list(selection.route_ids),
        },
    )


def _try_write_audit(
    builder: Phase1ArtifactBuilder,
    references: TwoRouteReferenceEvidence,
    samples: Sequence[GeneratedSample],
    measurements: MeasurementBatch,
) -> tuple[BlindedAuditPacket, BlindedAuditKey] | None:
    paired = {item.record_id: item for item in references.paired_observations}
    measured = measurements.by_id
    reference_records = tuple(
        AuditSourceRecord(
            source_id=f"reference:{brief.brief_id}",
            pair_id=brief.brief_id,
            story_text=brief.matched_reference_text,
            source_prompt=brief.prompt_text,
            token_count=len(paired[brief.source_record_id].model_token_ids),
            base_normalized_nll=paired[brief.source_record_id].normalized_nll,
            automated_style_scores=_reference_style_scores(
                paired[brief.source_record_id],
                references.reference_profile.vocabulary,
            ),
            source_kind=AuditSourceKind.REFERENCE,
        )
        for brief in references.briefs
    )
    generated_records = tuple(
        AuditSourceRecord(
            source_id=sample.sample_id,
            pair_id=sample.job.brief.brief_id,
            story_text=sample.payload.story,
            source_prompt=sample.job.brief.prompt_text,
            token_count=len(measured[sample.sample_id].model_token_ids),
            base_normalized_nll=measured[sample.sample_id].normalized_nll,
            automated_style_scores=_generated_style_scores(
                sample.payload.story,
                references.reference_profile.vocabulary,
            ),
            source_kind=AuditSourceKind.GENERATED,
            route_id=sample.job.route.route_id,
        )
        for sample in samples
        if sample.validation.accepted
        and sample.payload is not None
        and sample.sample_id in measured
    )
    try:
        packet, key = build_blinded_audit(
            reference_records,
            generated_records,
            finalist_order=TWO_ROUTE_AUTHOR_ORDER,
            seed="tinyworlds-v2-phase1-two-route-audit-v2",
            reference_count=TWO_ROUTE_AUDIT_COUNT,
            generated_count=TWO_ROUTE_AUDIT_COUNT,
        )
    except AuditAllocationError:
        builder.write_json(
            "audit_feasibility.json",
            {
                "accepted_by_route": {
                    route_id: sum(
                        sample.validation.accepted
                        and sample.job.route.route_id == route_id
                        for sample in samples
                    )
                    for route_id in TWO_ROUTE_AUTHOR_ORDER
                },
                "required_per_route": TWO_ROUTE_AUDIT_COUNT // 2,
            },
        )
        return None
    builder.write_bytes("audit_packet.json", encode_blinded_audit_packet(packet))
    builder.write_bytes("audit_key.json", encode_blinded_audit_key(key))
    builder.write_bytes("audit.html", render_audit_html(packet).encode("utf-8"))
    return packet, key


def _reference_style_scores(
    observation: ReferenceObservation,
    vocabulary: frozenset[str],
) -> tuple[tuple[str, float], ...]:
    coverage = sum(token.casefold() in vocabulary for token in observation.word_tokens) / len(
        observation.word_tokens
    )
    return (
        ("reference_vocabulary_coverage", coverage),
        ("median_sentence_words", float(median(observation.sentence_word_counts))),
        ("repeated_ngram_fraction", observation.repeated_ngram_fraction),
        ("dialogue_present", float(observation.dialogue_present)),
    )


def _generated_style_scores(
    story: str,
    vocabulary: frozenset[str],
) -> tuple[tuple[str, float], ...]:
    # Token/NLL fields are not needed for these surface-only audit hints.
    words = lexical_tokens(story)
    coverage = sum(token.casefold() in vocabulary for token in words) / len(words)
    sentence_counts = _sentence_word_counts(story)
    return (
        ("reference_vocabulary_coverage", coverage),
        ("median_sentence_words", float(median(sentence_counts))),
        ("repeated_ngram_fraction", repeated_ngram_fraction(words)),
        ("dialogue_present", float('"' in story or "“" in story)),
    )


def _sentence_word_counts(story: str) -> tuple[int, ...]:
    import re

    counts = tuple(
        len(lexical_tokens(sentence))
        for sentence in re.split(r"(?<=[.!?])\s+", story.strip())
        if lexical_tokens(sentence)
    )
    return counts or (len(lexical_tokens(story)),)


def _write_cost_actuals(
    builder: Phase1ArtifactBuilder,
    samples: Sequence[GeneratedSample],
    client: CachedGenerationClient,
) -> None:
    route_records: list[JsonObject] = []
    for route_id in TWO_ROUTE_AUTHOR_ORDER:
        route_samples = tuple(
            sample for sample in samples if sample.job.route.route_id == route_id
        )
        actual = sum((Decimal(sample.billed_cost_usd) for sample in route_samples), Decimal(0))
        accepted = sum(sample.validation.accepted for sample in route_samples)
        acceptance = Decimal(max(accepted, 1)) / Decimal(len(route_samples))
        projected = actual / Decimal(len(route_samples)) / acceptance * Decimal(
            TWO_ROUTE_PROJECTED_CORPUS_STORIES
        )
        route_records.append(
            {
                "accepted_count": accepted,
                "actual_billed_usd": float(actual),
                "projected_full_corpus_usd": float(projected),
                "request_count": len(route_samples),
                "route_id": route_id,
            }
        )
    total = sum(
        (Decimal(sample.billed_cost_usd) for sample in samples),
        Decimal(0),
    )
    snapshot = (
        client.cost_ledger.snapshot().as_record()
        if type(client) is OpenRouterClient
        else None
    )
    builder.write_json(
        "cost_actuals.json",
        {
            "actual_billed_usd": float(total),
            "projected_accepted_story_count": TWO_ROUTE_PROJECTED_CORPUS_STORIES,
            "routes": route_records,
            "runtime_ledger": snapshot,
        },
    )


def _write_raw_evidence(
    builder: Phase1ArtifactBuilder,
    cache: ImmutableRawCache,
    jobs: Sequence[GenerationJob],
) -> None:
    for job in jobs:
        source = cache.root / job.request.request_sha256
        if not source.is_dir():
            raise ValueError("completed two-route job lacks immutable raw cache evidence")
        for path in sorted(item for item in source.rglob("*") if item.is_file()):
            relative = path.relative_to(source).as_posix()
            builder.write_bytes(
                f"raw_cache/requests/{job.request.request_sha256}/{relative}",
                path.read_bytes(),
            )
    journal = cache.root / "runtime-cost-journal"
    if journal.is_dir():
        for path in sorted(item for item in journal.rglob("*") if item.is_file()):
            builder.write_bytes(
                f"raw_cache/runtime-cost-journal/{path.relative_to(journal).as_posix()}",
                path.read_bytes(),
            )


def _write_empty_outcome(builder: Phase1ArtifactBuilder, status: str) -> None:
    builder.write_bytes("generator_bakeoff.jsonl", b"")
    for route_id in TWO_ROUTE_AUTHOR_ORDER:
        builder.write_bytes(f"routes/{route_id}/accepted.jsonl", b"")
        builder.write_bytes(f"routes/{route_id}/rejected.jsonl", b"")
    builder.write_json(
        "quality_details.json",
        {"direct_reports": [], "selection": None},
    )
    builder.write_json(
        "quality_comparisons.json",
        {"audited_route_ids": [], "qualified_route_ids": []},
    )
    builder.write_json(
        "cost_actuals.json",
        {"actual_billed_usd": 0.0, "routes": [], "runtime_ledger": None},
    )
    builder.write_json(
        "status.json",
        {"audit_sha256": None, "phase": 1, "status": status},
    )


def _finalize_result(
    builder: Phase1ArtifactBuilder,
    paths: TwoRoutePaths,
    preflight: CostPreflight,
    *,
    status: str,
    qualified: tuple[str, ...],
    audit_sha256: str | None,
) -> TwoRouteRunResult:
    manifest = builder.finalize()
    validated = validate_two_route_bakeoff(builder.root)
    if validated.manifest_sha256 != manifest.manifest_sha256:
        raise ValueError("two-route semantic validation changed artifact identity")
    destination = builder.promote(
        paths.destination,
        expected_manifest_sha256=manifest.manifest_sha256,
    )
    return TwoRouteRunResult(
        directory=destination,
        status=status,
        conservative_cost_usd=preflight.conservative_usd,
        qualified_route_ids=qualified,
        audit_sha256=audit_sha256,
    )


def _validate_planned_jobs(
    root: Path,
    jobs: Sequence[GenerationJob],
    routes: Sequence[RouteLock],
) -> None:
    routes_by_id = {route.route_id: route for route in routes}
    for route_id in TWO_ROUTE_AUTHOR_ORDER:
        route_jobs = tuple(job for job in jobs if job.route.route_id == route_id)
        records = _jsonl_objects(root / "routes" / route_id / "requests.jsonl")
        expected = tuple(
            {
                **job.request.as_record(),
                "body": job.request.body,
                "brief_id": job.brief.brief_id,
                "route_id": route_id,
            }
            for job in route_jobs
        )
        if records != expected:
            raise ValueError(f"planned V4 requests differ for {route_id}")
        route = routes_by_id[route_id]
        plan = _json_object(root / "routes" / route_id / "plan.json")
        expected_plan = {
            "planned_request_count": len(route_jobs),
            "planned_request_sha256": [
                job.request.request_sha256 for job in route_jobs
            ],
            "request_contract_version": SYNTHETIC_STORY_REQUEST_V4.version,
            "route": route.as_record(),
            "route_lock_sha256": route.lock_sha256,
            "validator_version": TWO_ROUTE_VALIDATOR_VERSION,
        }
        if plan != expected_plan:
            raise ValueError(f"two-route plan differs for {route_id}")


def _validate_completed_results(
    root: Path,
    briefs: Sequence[NeutralStoryBrief],
    jobs: Sequence[GenerationJob],
    status: JsonObject,
) -> None:
    records = _jsonl_objects(root / "generator_bakeoff.jsonl")
    if len(records) != TWO_ROUTE_REQUEST_COUNT * 2:
        raise ValueError("completed two-route artifact requires 400 results")
    jobs_by_sample = {job.sample_id: job for job in jobs}
    if len(jobs_by_sample) != len(jobs):
        raise ValueError("planned two-route sample IDs repeat")
    seen: set[str] = set()
    accepted_ids: set[str] = set()
    for record in records:
        require_exact_fields(
            record,
            (
                "billed_cost_usd",
                "brief_id",
                "error_kind",
                "generation_id",
                "input_tokens",
                "output_tokens",
                "payload",
                "request_sha256",
                "route_id",
                "sample_id",
                "validation",
            ),
            label="two-route result",
        )
        sample_id = record.get("sample_id")
        if type(sample_id) is not str or sample_id in seen or sample_id not in jobs_by_sample:
            raise ValueError("two-route result sample identity is invalid")
        seen.add(sample_id)
        job = jobs_by_sample[sample_id]
        if record.get("request_sha256") != job.request.request_sha256:
            raise ValueError("two-route result request identity differs")
        if (
            record.get("route_id") != job.route.route_id
            or record.get("brief_id") != job.brief.brief_id
        ):
            raise ValueError("two-route result route or brief identity differs")
        payload = record.get("payload")
        if type(payload) is dict:
            story = payload.get("story")
            if type(story) is not str:
                raise ValueError("two-route story payload is missing")
            derived, validation = validate_story_only_generated_story(
                job.brief,
                canonical_story_only_content(story),
            )
            expected_payload = None if derived is None else _story_payload_record(derived)
            if payload != expected_payload:
                raise ValueError("persisted local story evidence differs")
            if record.get("validation") != _validation_record(validation):
                raise ValueError("persisted local story validation differs")
        else:
            validation = record.get("validation")
            if (
                type(validation) is not dict
                or validation.get("schema_valid") is not False
                or validation.get("accepted") is not False
                or validation.get("story_sha256") is not None
            ):
                raise ValueError("payload-less result has inconsistent validation")
        if _record_accepted(record):
            accepted_ids.add(sample_id)
    if seen != set(jobs_by_sample):
        raise ValueError("completed two-route results omit planned samples")
    for route_id in TWO_ROUTE_AUTHOR_ORDER:
        accepted = _jsonl_objects(root / "routes" / route_id / "accepted.jsonl")
        rejected = _jsonl_objects(root / "routes" / route_id / "rejected.jsonl")
        route_records = tuple(
            record for record in records if record.get("route_id") == route_id
        )
        if tuple(record for record in route_records if _record_accepted(record)) != accepted:
            raise ValueError("two-route accepted stream differs")
        if tuple(record for record in route_records if not _record_accepted(record)) != rejected:
            raise ValueError("two-route rejected stream differs")
    measurement_records = _jsonl_objects(root / "measurements" / "generation.jsonl")
    measurement_ids = {record.get("record_id") for record in measurement_records}
    if measurement_ids != accepted_ids:
        raise ValueError("two-route measurements do not equal accepted stories")
    if len(measurement_ids) != len(measurement_records):
        raise ValueError("two-route measurement identities repeat")
    _validate_raw_evidence(root, jobs, records)
    selection = _validate_direct_quality(root, briefs)
    _validate_cost_actuals(root, records)
    _validate_audit_result(
        root,
        briefs,
        records,
        measurement_records,
        selection,
        status,
    )


def _validate_cost_estimates(root: Path, routes: Sequence[RouteLock]) -> None:
    record = _json_object(root / "cost_estimates.json")
    require_exact_fields(
        record,
        (
            "conservative_usd",
            "expected_usd",
            "hard_cap_usd",
            "openai_batch_comparisons",
            "permitted",
            "route_estimates",
        ),
        label="two-route cost estimates",
    )
    estimates = record["route_estimates"]
    if type(estimates) is not list or len(estimates) != 2:
        raise ValueError("two-route preflight requires two route estimates")
    route_ids: list[str] = []
    workloads: list[RouteWorkload] = []
    for index, value in enumerate(estimates):
        if type(value) is not dict:
            raise ValueError(f"two-route cost estimate {index} must be an object")
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
            label=f"two-route cost estimate {index}",
        )
        route_id = value["route_id"]
        if type(route_id) is not str:
            raise ValueError("two-route cost estimate route ID must be text")
        route_ids.append(route_id)
        if value["request_count"] != TWO_ROUTE_REQUEST_COUNT:
            raise ValueError("two-route cost estimate request count differs")
        for name in (
            "conservative_input_tokens",
            "conservative_output_tokens",
            "expected_input_tokens",
            "expected_output_tokens",
        ):
            if type(value[name]) is not int or value[name] < 0:
                raise ValueError(f"two-route cost estimate {name} is invalid")
        token_totals = tuple(
            value[name]
            for name in (
                "expected_input_tokens",
                "expected_output_tokens",
                "conservative_input_tokens",
                "conservative_output_tokens",
            )
        )
        if any(total % TWO_ROUTE_REQUEST_COUNT for total in token_totals):
            raise ValueError("two-route token estimates are not per-request totals")
        workloads.append(
            RouteWorkload(
                routes[index],
                TokenWorkload(
                    label="phase1-direct-author-200",
                    request_count=TWO_ROUTE_REQUEST_COUNT,
                    input_tokens_per_request=(
                        value["expected_input_tokens"] // TWO_ROUTE_REQUEST_COUNT
                    ),
                    output_tokens_per_request=(
                        value["expected_output_tokens"] // TWO_ROUTE_REQUEST_COUNT
                    ),
                    retry_allowance_basis_points=(
                        TWO_ROUTE_RETRY_ALLOWANCE_BASIS_POINTS
                    ),
                    conservative_input_tokens_per_request=(
                        value["conservative_input_tokens"]
                        // TWO_ROUTE_REQUEST_COUNT
                    ),
                    conservative_output_tokens_per_request=(
                        value["conservative_output_tokens"]
                        // TWO_ROUTE_REQUEST_COUNT
                    ),
                ),
            )
        )
    if tuple(route_ids) != TWO_ROUTE_AUTHOR_ORDER:
        raise ValueError("two-route cost estimate order differs")
    expected = build_cost_preflight(tuple(workloads)).as_record()
    if record != expected:
        raise ValueError("two-route cost preflight differs from locked prices")


def _validate_cost_stop(root: Path, status: JsonObject) -> None:
    if status["audit_sha256"] is not None:
        raise ValueError("cost-stopped artifact cannot bind an audit")
    if _jsonl_objects(root / "generator_bakeoff.jsonl"):
        raise ValueError("cost-stopped artifact cannot contain generated results")
    quality = _json_object(root / "quality_details.json")
    if quality != {"direct_reports": [], "selection": None}:
        raise ValueError("cost-stopped artifact cannot contain quality results")
    comparisons = _json_object(root / "quality_comparisons.json")
    if comparisons != {"audited_route_ids": [], "qualified_route_ids": []}:
        raise ValueError("cost-stopped artifact cannot contain route comparisons")
    actuals = _json_object(root / "cost_actuals.json")
    if actuals != {
        "actual_billed_usd": 0.0,
        "routes": [],
        "runtime_ledger": None,
    }:
        raise ValueError("cost-stopped artifact contains nonzero actual costs")
    for route_id in TWO_ROUTE_AUTHOR_ORDER:
        if _jsonl_objects(root / "routes" / route_id / "accepted.jsonl") or _jsonl_objects(
            root / "routes" / route_id / "rejected.jsonl"
        ):
            raise ValueError("cost-stopped artifact contains route results")
    if any(
        (root / name).exists()
        for name in ("audit.html", "audit_key.json", "audit_packet.json")
    ):
        raise ValueError("cost-stopped artifact contains audit files")


def _validate_raw_evidence(
    root: Path,
    jobs: Sequence[GenerationJob],
    records: Sequence[JsonObject],
) -> None:
    request_cache = ImmutableRawCache(root / "raw_cache" / "requests")
    cached_requests = request_cache.load_all_requests()
    expected_requests = {job.request.request_sha256: job for job in jobs}
    if {request.request_sha256 for request in cached_requests} != set(expected_requests):
        raise ValueError("two-route raw cache request set differs from the plan")
    records_by_request = {str(record["request_sha256"]): record for record in records}
    raw_attempts: set[tuple[str, int]] = set()
    for request in cached_requests:
        job = expected_requests[request.request_sha256]
        if request != job.request:
            raise ValueError("two-route raw cached request differs from the plan")
        if (
            request_cache.load_route_lock(request.request_sha256).lock_sha256
            != job.route.lock_sha256
        ):
            raise ValueError("two-route raw cached route lock differs")
        attempts = request_cache.load_attempts(request)
        if not attempts:
            raise ValueError("completed two-route request has no raw attempt")
        raw_attempts.update(
            (request.request_sha256, attempt.attempt_number) for attempt in attempts
        )
        billed = _decimal_value(
            records_by_request[request.request_sha256]["billed_cost_usd"],
            "generated billed cost",
        )
        successes = tuple(
            attempt.response
            for attempt in attempts
            if attempt.response is not None
            and 200 <= attempt.response.status_code < 300
            and attempt.response.billed_cost_usd is not None
        )
        if not any(
            _decimal_value(response.billed_cost_usd, "raw billed cost") == billed
            for response in successes
        ):
            raise ValueError("generated billed cost lacks matching raw success")
    journal = ImmutableRawCache(root / "raw_cache").load_cost_journal()
    journal_attempts = {
        (entry.request_sha256, entry.attempt_number)
        for entry in journal
        if not entry.cancelled_before_post
    }
    if journal_attempts != raw_attempts:
        raise ValueError("two-route cost journal and raw attempts differ")
    if any(
        entry.cancelled_before_post
        or entry.charged_usd is None
        or entry.provider_reported_actual is not True
        for entry in journal
    ):
        raise ValueError("completed two-route journal contains unsettled cost evidence")
    journal_total = sum(
        (
            _decimal_value(entry.charged_usd, "journal charged cost")
            for entry in journal
        ),
        Decimal(0),
    )
    result_total = sum(
        (
            _decimal_value(record["billed_cost_usd"], "generated billed cost")
            for record in records
        ),
        Decimal(0),
    )
    if journal_total != result_total:
        raise ValueError("two-route journal charge differs from generated results")


def _validate_direct_quality(
    root: Path,
    briefs: Sequence[NeutralStoryBrief],
) -> QualitySelection:
    from apm.data.text.tinyworlds_v2.phase1_semantics import (
        _quality_report,
        _quality_selection,
    )

    quality = _json_object(root / "quality_details.json")
    require_exact_fields(
        quality,
        ("direct_reports", "selection"),
        label="two-route quality details",
    )
    values = quality["direct_reports"]
    if type(values) is not list or len(values) != 2:
        raise ValueError("two-route artifact requires two direct quality reports")
    reports = tuple(_quality_report(value, index) for index, value in enumerate(values))
    if tuple(report.route_id for report in reports) != TWO_ROUTE_AUTHOR_ORDER:
        raise ValueError("two-route quality report order differs")
    brief_ids = {brief.brief_id for brief in briefs}
    for report in reports:
        if report.phase is not QualityPhase.DIRECT or set(report.sample_ids) != brief_ids:
            raise ValueError("two-route quality report scope differs")
    expected = select_direct_quality_routes(reports)
    stored = _quality_selection(quality["selection"], "two-route quality selection")
    if stored != expected:
        raise ValueError("two-route quality selection differs from its reports")
    comparisons = _json_object(root / "quality_comparisons.json")
    if comparisons != {
        "audited_route_ids": list(TWO_ROUTE_AUTHOR_ORDER),
        "qualified_route_ids": list(expected.route_ids),
    }:
        raise ValueError("two-route quality comparisons differ from selection")
    return expected


def _validate_cost_actuals(
    root: Path,
    records: Sequence[JsonObject],
) -> None:
    actuals = _json_object(root / "cost_actuals.json")
    require_exact_fields(
        actuals,
        (
            "actual_billed_usd",
            "projected_accepted_story_count",
            "routes",
            "runtime_ledger",
        ),
        label="two-route cost actuals",
    )
    expected_routes: list[JsonObject] = []
    total = Decimal(0)
    for route_id in TWO_ROUTE_AUTHOR_ORDER:
        route_records = tuple(record for record in records if record["route_id"] == route_id)
        route_total = sum(
            (
                _decimal_value(record["billed_cost_usd"], "generated billed cost")
                for record in route_records
            ),
            Decimal(0),
        )
        total += route_total
        accepted = sum(_record_accepted(record) for record in route_records)
        acceptance = Decimal(max(accepted, 1)) / Decimal(len(route_records))
        projected = (
            route_total
            / Decimal(len(route_records))
            / acceptance
            * Decimal(TWO_ROUTE_PROJECTED_CORPUS_STORIES)
        )
        expected_routes.append(
            {
                "accepted_count": accepted,
                "actual_billed_usd": float(route_total),
                "projected_full_corpus_usd": float(projected),
                "request_count": len(route_records),
                "route_id": route_id,
            }
        )
    if actuals["routes"] != expected_routes:
        raise ValueError("two-route route cost actuals differ from results")
    if actuals["projected_accepted_story_count"] != TWO_ROUTE_PROJECTED_CORPUS_STORIES:
        raise ValueError("two-route projected corpus size differs")
    if _decimal_value(actuals["actual_billed_usd"], "actual billed total") != total:
        raise ValueError("two-route actual billed total differs from results")
    ledger = actuals["runtime_ledger"]
    if type(ledger) is not dict:
        raise ValueError("completed two-route artifact requires a runtime ledger")
    require_exact_fields(
        ledger,
        (
            "charged_total_usd",
            "cancelled_before_post_count",
            "conservative_unknown_charge_usd",
            "hard_cap_usd",
            "halted_reason",
            "in_flight_attempt_count",
            "in_flight_reserved_usd",
            "provider_reported_actual_usd",
            "provider_reported_attempt_count",
            "unknown_cost_attempt_count",
        ),
        label="two-route runtime ledger",
    )
    if (
        _decimal_value(ledger["charged_total_usd"], "runtime charged total") != total
        or _decimal_value(ledger["provider_reported_actual_usd"], "runtime actual total")
        != total
        or _decimal_value(ledger["conservative_unknown_charge_usd"], "unknown charge")
        != 0
        or _decimal_value(ledger["in_flight_reserved_usd"], "in-flight reserve") != 0
        or _decimal_value(ledger["hard_cap_usd"], "runtime hard cap")
        != Decimal(PHASE1_HARD_CAP_USD)
        or ledger["provider_reported_attempt_count"] < len(records)
        or ledger["unknown_cost_attempt_count"] != 0
        or ledger["in_flight_attempt_count"] != 0
        or ledger["cancelled_before_post_count"] != 0
        or ledger["halted_reason"] is not None
    ):
        raise ValueError("two-route runtime ledger contradicts completed results")


def _validate_audit_result(
    root: Path,
    briefs: Sequence[NeutralStoryBrief],
    records: Sequence[JsonObject],
    measurements: Sequence[JsonObject],
    selection: QualitySelection,
    status: JsonObject,
) -> None:
    packet_path = root / "audit_packet.json"
    key_path = root / "audit_key.json"
    html_path = root / "audit.html"
    has_audit = packet_path.is_file() and key_path.is_file() and html_path.is_file()
    if any(path.exists() for path in (packet_path, key_path, html_path)) != has_audit:
        raise ValueError("two-route audit files are incomplete")
    status_name = status["status"]
    if status_name == "audit_insufficient_accepted_samples":
        if has_audit or status["audit_sha256"] is not None:
            raise ValueError("infeasible audit status cannot contain an audit")
        feasibility_path = root / "audit_feasibility.json"
        if not feasibility_path.is_file():
            raise ValueError("infeasible audit status lacks feasibility evidence")
        feasibility = _json_object(feasibility_path)
        expected_acceptance = {
            route_id: sum(
                _record_accepted(record) and record["route_id"] == route_id
                for record in records
            )
            for route_id in TWO_ROUTE_AUTHOR_ORDER
        }
        if feasibility != {
            "accepted_by_route": expected_acceptance,
            "required_per_route": TWO_ROUTE_AUDIT_COUNT // 2,
        }:
            raise ValueError("two-route audit feasibility evidence differs")
        return
    if not has_audit:
        raise ValueError("completed two-route status requires a balanced audit")
    packet, key = decode_audit_pair(packet_path.read_bytes(), key_path.read_bytes())
    if status["audit_sha256"] != packet.audit_sha256:
        raise ValueError("two-route status audit digest differs")
    if html_path.read_bytes() != render_audit_html(packet).encode("utf-8"):
        raise ValueError("two-route audit HTML differs from its packet")
    if len(packet.items) != 2 * TWO_ROUTE_AUDIT_COUNT:
        raise ValueError("two-route audit item count differs")
    items = {item.item_id: item for item in packet.items}
    briefs_by_id = {brief.brief_id: brief for brief in briefs}
    records_by_id = {str(record["sample_id"]): record for record in records}
    measurements_by_id = {
        str(measurement["record_id"]): measurement for measurement in measurements
    }
    generated_counts = {route_id: 0 for route_id in TWO_ROUTE_AUTHOR_ORDER}
    reference_count = 0
    for entry in key.entries:
        item = items[entry.item_id]
        brief = briefs_by_id.get(entry.pair_id)
        if brief is None:
            raise ValueError("two-route audit contains an unknown pair ID")
        if entry.source_kind is AuditSourceKind.REFERENCE:
            reference_count += 1
            if (
                entry.source_id != f"reference:{brief.brief_id}"
                or item.story_text != brief.matched_reference_text
                or item.source_prompt != brief.prompt_text
            ):
                raise ValueError("two-route reference audit item differs")
            continue
        if entry.route_id not in generated_counts or entry.source_id not in records_by_id:
            raise ValueError("two-route generated audit identity differs")
        record = records_by_id[entry.source_id]
        payload = record.get("payload")
        if (
            type(payload) is not dict
            or not _record_accepted(record)
            or record["route_id"] != entry.route_id
            or record["brief_id"] != brief.brief_id
            or item.story_text != payload.get("story")
            or item.source_prompt != brief.prompt_text
        ):
            raise ValueError("two-route generated audit item differs")
        measurement = measurements_by_id.get(entry.source_id)
        if measurement is None or (
            item.token_count != len(measurement["model_token_ids"])
            or item.base_normalized_nll != measurement["normalized_nll"]
        ):
            raise ValueError("two-route generated audit measurement differs")
        generated_counts[entry.route_id] += 1
    if reference_count != TWO_ROUTE_AUDIT_COUNT or tuple(generated_counts.values()) != (
        TWO_ROUTE_AUDIT_COUNT // 2,
        TWO_ROUTE_AUDIT_COUNT // 2,
    ):
        raise ValueError("two-route audit is not exactly balanced")
    if status_name == "no_quality_qualified_route" and selection.route_ids:
        raise ValueError("quality-stop status contains a qualified route")
    if status_name == "awaiting_human_audit" and not selection.route_ids:
        raise ValueError("human-audit status lacks a qualified route")


def _decimal_value(value: object, label: str) -> Decimal:
    if type(value) not in (str, int, float):
        raise ValueError(f"{label} must be a decimal value")
    try:
        result = Decimal(str(value))
    except Exception as error:
        raise ValueError(f"{label} must be a decimal value") from error
    if not result.is_finite() or result < 0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return result


def canonical_story_only_content(story: str) -> bytes:
    """Return the exact canonical one-field model envelope used for revalidation."""
    from apm.data.text.tinyworlds_v2.json_contracts import canonical_json_bytes

    return canonical_json_bytes({"story": story})


def _story_payload_record(payload: StoryOnlyPayload) -> JsonObject:
    return {
        "realized_features": list(payload.realized_features),
        "required_word_spans": [
            {
                "end": item.end,
                "exact_text": item.exact_text,
                "required_word": item.ingredient,
                "start": item.start,
            }
            for item in payload.required_word_spans
        ],
        "story": payload.story,
    }


def _validation_record(validation: object) -> JsonObject:
    from apm.data.text.tinyworlds_v2.bakeoff import StoryValidation

    if type(validation) is not StoryValidation:
        raise TypeError("validation must be StoryValidation")
    return {
        "accepted": validation.accepted,
        "evidence_valid": validation.evidence_valid,
        "forbidden_identifier_present": validation.forbidden_identifier_present,
        "length_valid": validation.length_valid,
        "rejection_reasons": list(validation.rejection_reasons),
        "required_words_present": validation.required_words_present,
        "schema_valid": validation.schema_valid,
        "story_sha256": validation.story_sha256,
    }


def _record_accepted(record: JsonObject) -> bool:
    validation = record.get("validation")
    if type(validation) is not dict or type(validation.get("accepted")) is not bool:
        raise ValueError("two-route result lacks a validation flag")
    return validation["accepted"]


def _brief_record(brief: NeutralStoryBrief) -> JsonObject:
    return {
        "brief_id": brief.brief_id,
        "matched_reference_text": brief.matched_reference_text,
        "prompt_text": brief.prompt_text,
        "requested_features": list(brief.requested_features),
        "required_words": list(brief.required_words),
        "source_record_id": brief.source_record_id,
    }


def _model_record(model: CandidateModelSpec) -> JsonObject:
    return {
        "canonical_slug": model.canonical_slug,
        "first_party_provider_slug": model.first_party_provider_slug,
        "max_token_parameter": model.max_token_parameter,
        "plan_completion_usd_per_million": model.plan_completion_usd_per_million,
        "plan_prompt_usd_per_million": model.plan_prompt_usd_per_million,
        "request_model_id": model.request_model_id,
        "route_id": model.route_id,
    }


def _json_object(path: Path) -> JsonObject:
    value = canonical_json_loads(path.read_bytes(), label=path.name)
    return require_json_object(value, label=path.name)


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


__all__ = [
    "TWO_ROUTE_BASE_REFERENCE_MANIFEST_SHA256",
    "TWO_ROUTE_REQUEST_COUNT",
    "TWO_ROUTE_VALIDATOR_VERSION",
    "TWO_ROUTE_VERSION",
    "TwoRouteDependencies",
    "TwoRoutePaths",
    "TwoRouteReferenceEvidence",
    "TwoRouteRunResult",
    "build_two_route_cost_preflight",
    "load_two_route_reference_evidence",
    "prepare_two_route_bakeoff",
    "production_two_route_dependencies",
    "run_two_route_bakeoff",
    "validate_two_route_bakeoff",
]

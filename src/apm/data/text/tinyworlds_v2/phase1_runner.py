"""Fixed-preset production orchestration for TinyWorlds-v2 Phase 1.

The orchestration boundary is deliberately narrow: production dependencies
pin the source files, tokenizer/checkpoint, OpenRouter catalog, and HTTP
client, while tests may inject small offline corpora and fake transports.  No
command-line research choices are exposed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, replace
from decimal import Decimal
from hashlib import sha256
import math
import os
from pathlib import Path
import tempfile
from threading import Event, Thread
from typing import Protocol, TypeVar

from apm.data.text.curricula import TINYSTORIES_V2_SOURCE
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
    encode_blinded_audit_key,
    encode_blinded_audit_packet,
    validate_phase1_reference,
    validate_phase1_semantics,
    validate_phase1_tree_with_human_overlays,
)
from apm.data.text.tinyworlds_v2.bakeoff import (
    CANDIDATE_MODELS,
    VERIFIER_MODEL,
    NeutralStoryBrief,
    VerifierPayload,
)
from apm.data.text.tinyworlds_v2.catalog import (
    CatalogPayloads,
    PHASE1_PROMPT_TOKEN_UPPER_BOUND,
    ResolvedRouteCatalog,
    resolve_openrouter_catalog,
)
from apm.data.text.tinyworlds_v2.generation_cache import (
    CostJournalEntry,
    ImmutableRawCache,
)
from apm.data.text.tinyworlds_v2.generation_costs import (
    PHASE1_HARD_CAP_USD,
    CostCapExceeded,
    CostJournalRecoveryRequired,
    CostPreflight,
    RouteWorkload,
    TokenWorkload,
    RuntimeCostLedger,
    build_cost_preflight,
    enforce_cost_cap,
    exclusive_paid_run_lock,
)
from apm.data.text.tinyworlds_v2.generation_schema import (
    CanonicalRequest,
    RawAttempt,
    RouteLock,
)
from apm.data.text.tinyworlds_v2.httpx_transport import (
    HttpxTransport,
    fetch_catalog_payloads,
    load_openrouter_api_key,
    load_openrouter_management_api_key,
)
from apm.data.text.tinyworlds_v2.json_contracts import (
    JsonObject,
    JsonValue,
    canonical_json_loads,
    json_sha256,
    require_json_object,
)
from apm.data.text.tinyworlds_v2.openrouter import (
    OpenRouterBillingUnknown,
    OpenRouterClient,
    OpenRouterContractError,
    OpenRouterCostPolicyError,
    RetryPolicy,
)
from apm.data.text.tinyworlds_v2.route_lock import validate_route_semantics
from apm.data.text.tinyworlds_v2.phase1_artifacts import (
    Phase1ArtifactBuilder,
    canonical_jsonl_bytes,
)
from apm.data.text.tinyworlds_v2.phase1_generation import (
    CachedGenerationClient,
    GenerationJob,
    GeneratedSample,
    VerifierJob,
    VerifiedStory,
    build_generation_jobs,
    build_verifier_job,
    execute_generation_jobs,
    execute_verifier_jobs,
    generated_observation,
)
from apm.data.text.tinyworlds_v2.quality import (
    BLIND_VERIFIER_DIMENSIONS,
    GeneratedObservation,
    QualityOutcome,
    QualityPhase,
    QualitySelection,
    RouteQualityReport,
    evaluate_route_quality,
    select_full_quality_routes,
    select_screen_finalists,
)
from apm.data.text.tinyworlds_v2.reference_pipeline import (
    REFERENCE_SURFACE_WORKERS,
    build_phase1_reference_inputs,
    canonical_neutral_story_brief,
    canonical_prompt_ingredient_profile,
    canonical_reference_annotation,
    canonical_reference_observation,
    canonical_reference_record,
    prepare_reference_observations,
)
from apm.data.text.tinyworlds_v2.reference_profile import (
    ReferenceObservation,
    ReferenceProfile,
    ReferenceRecord,
    build_reference_profile,
    observe_reference,
)
from apm.data.text.tinyworlds_v2.reference_runtime import (
    NllStory,
    ReferenceNllRun,
    score_tinystories_checkpoint_nll,
)
from apm.data.text.tinyworlds_v2.source_data import (
    TINYSTORIES_ALL_DATA_SOURCE,
    canonical_prompt_metadata_record,
    canonical_validation_record,
    select_archive_source_records,
    select_validation_story_records,
)


PHASE1_SELECTION_SEED = "tinyworlds-v2-phase1-reference-v1"
PHASE1_AUDIT_SEED = "tinyworlds-v2-phase1-blinded-audit-v1"
PHASE1_SCREEN_COUNT = 50
PHASE1_FULL_COUNT = 200
PHASE1_AUDIT_COUNT = 100
PHASE1_PROJECTED_CORPUS_ACCEPTED_STORIES = 4_000
PHASE1_GENERATION_WORKERS = 8
# The production client permits two attempts.  Both may be billed (for
# example, a provider-side 5xx after inference), so the cap reserves 100% of
# the first-attempt estimate rather than assuming only ordinary retry rates.
PHASE1_RETRY_ALLOWANCE_BASIS_POINTS = 10_000
PHASE1_VERSION = "tinyworlds-v2-phase1-reference-v1"
OPENROUTER_BYOK_ATTESTATION_FILENAME = (
    "openrouter-tinyworlds-no-byok-attestation.json"
)
PHASE1_SURFACE_MEASUREMENT_VERSION = "tinyworlds-v2-surface-measurements-v2"
PHASE1_STORY_VALIDATOR_VERSION = "tinyworlds-v2-deterministic-story-validator-v3"


@dataclass(frozen=True, slots=True)
class Phase1Paths:
    """All fixed local paths used by the production Phase 1 preset."""

    repository_root: Path
    archive: Path
    validation: Path
    checkpoint: Path
    tokenizer: Path
    raw_cache: Path
    destination: Path

    @classmethod
    def from_repository(cls, repository_root: str | Path) -> "Phase1Paths":
        """Resolve the one production path layout beneath a repository root."""
        root = Path(repository_root).resolve()
        return cls(
            repository_root=root,
            archive=root
            / "data"
            / "tinyworlds-v2"
            / "source"
            / TINYSTORIES_ALL_DATA_SOURCE.archive_file.filename,
            validation=root
            / "data"
            / "tinystories-v2"
            / "TinyStoriesV2-GPT4-valid.txt",
            checkpoint=root / "checkpoints" / "tinystories-8m" / "checkpoint",
            tokenizer=root
            / "checkpoints"
            / "tinystories-8m"
            / "tokenizer"
            / "tokenizer.json",
            raw_cache=root
            / "data"
            / "tinyworlds-v2"
            / "cache"
            / "phase1-openrouter",
            destination=root / "data" / "tinyworlds-v2" / "reference",
        )


@dataclass(frozen=True, slots=True)
class StoryMeasurement:
    """Tokenizer IDs and base-model NLL for one complete story."""

    record_id: str
    model_token_ids: tuple[int, ...]
    normalized_nll: float
    active_token_count: int

    def __post_init__(self) -> None:
        if type(self.record_id) is not str or not self.record_id:
            raise ValueError("story measurement record_id must be nonempty")
        if not self.model_token_ids or any(
            type(token_id) is not int or token_id < 0
            for token_id in self.model_token_ids
        ):
            raise ValueError("story measurements require nonnegative token IDs")
        if (
            type(self.normalized_nll) not in (int, float)
            or not math.isfinite(self.normalized_nll)
            or self.normalized_nll < 0.0
        ):
            raise ValueError("story measurement NLL must be finite and nonnegative")
        if type(self.active_token_count) is not int or self.active_token_count <= 0:
            raise ValueError("story measurement active token count must be positive")

    def as_record(self) -> JsonObject:
        """Return a canonical serializable measurement record."""
        return {
            "active_token_count": self.active_token_count,
            "model_token_ids": list(self.model_token_ids),
            "normalized_nll": self.normalized_nll,
            "record_id": self.record_id,
        }


@dataclass(frozen=True, slots=True)
class MeasurementBatch:
    """Stable measurements plus the exact checkpoint runtime identity."""

    measurements: tuple[StoryMeasurement, ...]
    runtime: JsonObject

    def __post_init__(self) -> None:
        if type(self.measurements) is not tuple or any(
            type(item) is not StoryMeasurement for item in self.measurements
        ):
            raise TypeError("measurement batches must contain StoryMeasurement values")
        ids = tuple(item.record_id for item in self.measurements)
        if ids != tuple(sorted(ids)) or len(ids) != len(set(ids)):
            raise ValueError("measurement IDs must be unique and canonically ordered")
        if type(self.runtime) is not dict:
            raise TypeError("measurement runtime must be a JSON object")

    @property
    def by_id(self) -> dict[str, StoryMeasurement]:
        """Index the immutable measurements by their stable record identity."""
        return {item.record_id: item for item in self.measurements}


@dataclass(frozen=True, slots=True)
class Phase1ReferenceCorpus:
    """All source-derived material needed after the expensive reference phase."""

    briefs: tuple[NeutralStoryBrief, ...]
    reference_records: tuple[ReferenceRecord, ...]
    reference_observations: tuple[ReferenceObservation, ...]
    paired_reference_observations: tuple[ReferenceObservation, ...]
    reference_profile: ReferenceProfile
    paired_reference_profile: ReferenceProfile
    expected_feature_rates: tuple[tuple[str, float], ...]
    source_manifest: JsonObject
    source_artifacts: tuple[tuple[str, tuple[JsonObject, ...]], ...]
    reference_statistics: JsonObject

    def __post_init__(self) -> None:
        if not self.briefs or any(
            type(brief) is not NeutralStoryBrief for brief in self.briefs
        ):
            raise ValueError("reference corpus requires neutral story briefs")
        if len({brief.brief_id for brief in self.briefs}) != len(self.briefs):
            raise ValueError("reference brief IDs must be unique")
        if not self.reference_records or not self.reference_observations:
            raise ValueError("reference corpus requires genuine observations")
        if tuple(item.record_id for item in self.reference_records) != tuple(
            item.record_id for item in self.reference_observations
        ):
            raise ValueError("reference records and observations must align")
        paired_ids = {item.record_id for item in self.paired_reference_observations}
        if paired_ids != {brief.source_record_id for brief in self.briefs}:
            raise ValueError("paired observations must align with every neutral brief")
        if self.reference_profile.record_count != len(self.reference_observations):
            raise ValueError("reference profile count differs from observations")
        if self.paired_reference_profile.record_count != len(
            self.paired_reference_observations
        ):
            raise ValueError("paired reference profile count differs from observations")
        expected_labels = tuple(label for label, _ in self.expected_feature_rates)
        if expected_labels != tuple(sorted(expected_labels)) or len(
            expected_labels
        ) != len(set(expected_labels)):
            raise ValueError("expected feature rate labels must be unique and sorted")
        paths = tuple(path for path, _ in self.source_artifacts)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("source artifact paths must be unique and sorted")


@dataclass(frozen=True, slots=True)
class CatalogEvidence:
    """Exact public catalog payloads and their deterministic route resolution."""

    payloads: CatalogPayloads
    resolved: ResolvedRouteCatalog

    def __post_init__(self) -> None:
        if self.payloads.snapshot_sha256 != self.resolved.snapshot_sha256:
            raise ValueError("resolved routes differ from the catalog payload digest")


ReferenceProgress = Callable[[int], None]
PrepareReferences = Callable[[ReferenceProgress], Phase1ReferenceCorpus]
FetchCatalog = Callable[[], CatalogEvidence]
EncodeText = Callable[[str], tuple[int, ...]]
MeasureStories = Callable[[tuple[NllStory, ...], ReferenceProgress], MeasurementBatch]
LoadApiKey = Callable[[], str]
MakeClient = Callable[[str, ImmutableRawCache], CachedGenerationClient]
RevalidateRoute = Callable[[RouteLock], RouteLock]


def _skip_route_revalidation(route: RouteLock) -> RouteLock:
    return route


class CatalogRouteRevalidationError(RuntimeError):
    """Fresh public catalog evidence no longer matches a paid route lock."""


@dataclass(frozen=True, slots=True)
class Phase1Dependencies:
    """Injected external boundaries for production and fully offline tests."""

    prepare_references: PrepareReferences
    fetch_catalog: FetchCatalog
    encode_text: EncodeText
    measure_stories: MeasureStories
    load_api_key: LoadApiKey
    make_client: MakeClient
    revalidate_route: RevalidateRoute = _skip_route_revalidation


@dataclass(frozen=True, slots=True)
class Phase1RunResult:
    """Promoted Phase 1 result, including explicit scientific stop states."""

    directory: Path
    status: str
    conservative_cost_usd: str
    qualified_route_ids: tuple[str, ...]
    audit_sha256: str | None


@dataclass(frozen=True, slots=True)
class AttemptBilling:
    """All provider-reported costs across every immutable cached attempt."""

    generation_by_route: tuple[tuple[str, Decimal], ...]
    verification_billed_usd: Decimal
    verifier_request_count: int
    cost_by_request: tuple[tuple[str, Decimal], ...]
    route_by_request: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        route_ids = tuple(route_id for route_id, _ in self.generation_by_route)
        if route_ids != tuple(model.route_id for model in CANDIDATE_MODELS):
            raise ValueError("attempt billing must retain all seven route IDs in order")
        costs = tuple(cost for _, cost in self.generation_by_route) + (
            self.verification_billed_usd,
            *(cost for _, cost in self.cost_by_request),
        )
        if any(not cost.is_finite() or cost < 0 for cost in costs):
            raise ValueError("attempt billing costs must be finite and nonnegative")
        if type(self.verifier_request_count) is not int or self.verifier_request_count < 0:
            raise ValueError("verifier request count must be nonnegative")
        request_ids = tuple(request_id for request_id, _ in self.cost_by_request)
        if request_ids != tuple(sorted(request_ids)) or len(request_ids) != len(
            set(request_ids)
        ):
            raise ValueError("attempt billing request IDs must be unique and sorted")
        route_request_ids = tuple(
            request_id for request_id, _ in self.route_by_request
        )
        if route_request_ids != request_ids:
            raise ValueError(
                "attempt billing route assignments must match billed requests"
            )
        valid_route_ids = {*route_ids, VERIFIER_MODEL.route_id}
        if any(
            route_id not in valid_route_ids
            for _, route_id in self.route_by_request
        ):
            raise ValueError("attempt billing contains an unknown request route")
        if self.verifier_request_count != sum(
            route_id == VERIFIER_MODEL.route_id
            for _, route_id in self.route_by_request
        ):
            raise ValueError("attempt billing verifier request count differs")

    @property
    def generation_billed_usd(self) -> Decimal:
        """Return total generation billing across all seven routes."""
        return sum((cost for _, cost in self.generation_by_route), Decimal(0))

    @property
    def actual_billed_usd(self) -> Decimal:
        """Return generation plus independent-verifier billing."""
        return self.generation_billed_usd + self.verification_billed_usd


@dataclass(frozen=True, slots=True)
class _Phase:
    number: int
    name: str
    estimated_seconds: int


PHASES = (
    _Phase(1, "verify and profile pinned TinyStories references", 1_800),
    _Phase(2, "resolve routes and enforce the cost preflight", 30),
    _Phase(3, "screen seven generator routes on 50 paired briefs", 900),
    _Phase(4, "expand selected finalists to 200 paired briefs", 900),
    _Phase(5, "run blind verification and full quality gates", 1_200),
    _Phase(6, "build the blinded mandatory human audit", 30),
    _Phase(7, "validate and atomically promote the artifact", 30),
)


ResultT = TypeVar("ResultT")


class PhaseProgress(Protocol):
    """A phase/ETA surface shared by production and silent offline tests."""

    def run(self, phase: _Phase, operation: Callable[[], ResultT]) -> ResultT:
        """Run one named phase and return its result."""


class _TqdmBar(Protocol):
    n: float

    def update(self, amount: float = 1) -> object:
        """Advance this progress bar."""

    def close(self) -> None:
        """Close this progress bar."""

    def write(self, message: str) -> object:
        """Write a phase line without damaging bars."""


class Phase1Progress:
    """Emit human phase lines, persistent events, and phase/overall ETA bars."""

    def __init__(self, temporary_directory: Path) -> None:
        self._temporary_directory = temporary_directory
        self._overall_bar: _TqdmBar | None = None
        self._tqdm_factory: Callable[..., _TqdmBar] | None = None

    def __enter__(self) -> "Phase1Progress":
        from tqdm.auto import tqdm

        self._tqdm_factory = tqdm
        self._overall_bar = tqdm(
            total=sum(phase.estimated_seconds for phase in PHASES),
            desc="TinyWorlds-v2 Phase 1 overall",
            unit="est-s",
            position=0,
            dynamic_ncols=True,
            leave=True,
        )
        return self

    def __exit__(self, *_exception: object) -> None:
        if self._overall_bar is not None:
            self._overall_bar.close()

    def run(self, phase: _Phase, operation: Callable[[], ResultT]) -> ResultT:
        """Run one operation while persisting its start, finish, and failure."""
        if self._overall_bar is None or self._tqdm_factory is None:
            raise RuntimeError("Phase1Progress must be entered before use")
        self._overall_bar.write(f"Phase {phase.number}/{len(PHASES)}: {phase.name}")
        _append_jsonl(
            self._temporary_directory / "progress.jsonl",
            {"event": "phase_started", "name": phase.name, "phase": phase.number},
        )
        phase_bar = self._tqdm_factory(
            total=phase.estimated_seconds,
            desc=f"Phase {phase.number}/{len(PHASES)}",
            unit="est-s",
            position=1,
            dynamic_ncols=True,
            leave=False,
        )
        stop = Event()
        timer = Thread(
            target=_advance_eta_bars,
            args=(stop, phase_bar, self._overall_bar, phase.estimated_seconds),
            daemon=True,
        )
        timer.start()
        try:
            result = operation()
        except BaseException as error:
            _append_jsonl(
                self._temporary_directory / "progress.jsonl",
                {
                    "error": str(error),
                    "error_type": type(error).__name__,
                    "event": "phase_failed",
                    "name": phase.name,
                    "phase": phase.number,
                },
            )
            raise
        else:
            _append_jsonl(
                self._temporary_directory / "progress.jsonl",
                {
                    "event": "phase_completed",
                    "name": phase.name,
                    "phase": phase.number,
                },
            )
            return result
        finally:
            stop.set()
            timer.join()
            remaining = max(0.0, phase.estimated_seconds - phase_bar.n)
            phase_bar.update(remaining)
            self._overall_bar.update(remaining)
            phase_bar.close()


class _ImmediateProgress:
    """Synchronous phase surface used when a caller intentionally omits bars."""

    def run(self, phase: _Phase, operation: Callable[[], ResultT]) -> ResultT:
        return operation()


def production_dependencies(paths: Phase1Paths) -> Phase1Dependencies:
    """Build the fixed local/GPU/OpenRouter production dependency set."""
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    from apm.lm.text import TokenizersTextTokenizer

    tokenizer = TokenizersTextTokenizer.from_file(paths.tokenizer)

    def measure(
        stories: tuple[NllStory, ...],
        progress_callback: ReferenceProgress,
    ) -> MeasurementBatch:
        run = score_tinystories_checkpoint_nll(
            stories,
            paths.checkpoint,
            paths.tokenizer,
            sequence_length=256,
            batch_size=32,
            require_gpu=True,
            progress_callback=progress_callback,
        )
        scores = {score.record_id: score for score in run.scores}
        return MeasurementBatch(
            measurements=tuple(
                StoryMeasurement(
                    story.record_id,
                    tokenizer.encode(story.text, add_eos=True),
                    scores[story.record_id].normalized_nll,
                    scores[story.record_id].token_count,
                )
                for story in sorted(stories, key=lambda item: item.record_id)
            ),
            runtime=_nll_run_record(run),
        )

    def prepare(progress_callback: ReferenceProgress) -> Phase1ReferenceCorpus:
        return _prepare_production_references(paths, measure, progress_callback)

    transport = HttpxTransport()
    return Phase1Dependencies(
        prepare_references=prepare,
        fetch_catalog=lambda: _catalog_evidence(transport),
        encode_text=lambda text: tokenizer.encode(text, add_eos=False),
        measure_stories=measure,
        load_api_key=lambda: load_openrouter_api_key(paths.repository_root),
        make_client=lambda key, cache: OpenRouterClient(
            api_key=key,
            management_api_key=load_openrouter_management_api_key(),
            byok_attestation_path=(
                paths.repository_root / OPENROUTER_BYOK_ATTESTATION_FILENAME
            ),
            transport=transport,
            cache=cache,
            retry_policy=RetryPolicy(max_attempts=2),
            require_byok_preflight=True,
        ),
        revalidate_route=lambda route: _revalidate_catalog_route(transport, route),
    )


def run_phase1(
    staging_directory: str | Path,
    paths: Phase1Paths,
    dependencies: Phase1Dependencies,
    *,
    progress: PhaseProgress | None = None,
    emit: Callable[[str], None] = print,
) -> Phase1RunResult:
    """Run Phase 1 while exclusively owning its cross-process paid-cost cache."""
    with exclusive_paid_run_lock(paths.raw_cache):
        return _run_phase1_under_paid_lock(
            staging_directory,
            paths,
            dependencies,
            progress=progress,
            emit=emit,
        )


def _run_phase1_under_paid_lock(
    staging_directory: str | Path,
    paths: Phase1Paths,
    dependencies: Phase1Dependencies,
    *,
    progress: PhaseProgress | None = None,
    emit: Callable[[str], None] = print,
) -> Phase1RunResult:
    """Run the fixed seven-route Phase 1 funnel and promote one immutable tree."""
    staging = Path(staging_directory)
    if not staging.is_dir():
        raise FileNotFoundError(f"Phase 1 staging directory does not exist: {staging}")
    if paths.destination.exists() or paths.destination.is_symlink():
        raise FileExistsError(f"Phase 1 destination already exists: {paths.destination}")
    phase_progress = _ImmediateProgress() if progress is None else progress
    builder = Phase1ArtifactBuilder(staging, version=PHASE1_VERSION)
    cache = ImmutableRawCache(paths.raw_cache)

    references = phase_progress.run(
        PHASES[0], lambda: dependencies.prepare_references(lambda _: None)
    )
    _write_reference_artifacts(builder, references)

    catalog, all_jobs, preflight = phase_progress.run(
        PHASES[1],
        lambda: _prepare_catalog_and_costs(
            builder,
            references,
            dependencies,
        ),
    )
    emit(f"Cost preflight: {staging / 'cost_estimates.json'}")
    emit(
        "Conservative Phase 1 OpenRouter estimate: "
        f"${preflight.conservative_usd} / ${preflight.hard_cap_usd} cap"
    )
    all_route_locks = (
        *catalog.resolved.generator_routes,
        catalog.resolved.verifier_route,
    )
    for estimate in preflight.route_estimates:
        emit(
            f"  {estimate.route_id}: expected ${estimate.expected_usd}; "
            f"two-attempt reserve ${estimate.conservative_usd} "
            f"({estimate.request_count} requests)"
        )
    for comparison in preflight.openai_batch_comparisons:
        emit(
            f"  Direct OpenAI Batch estimate only, {comparison.model_snapshot}: "
            f"expected ${comparison.expected_usd}; conservative "
            f"${comparison.conservative_usd}"
        )
    if not preflight.permitted:
        return _publish_stopped_result(
            builder,
            paths,
            cache,
            verifier_route=catalog.resolved.verifier_route,
            all_generation_jobs=all_jobs,
            generated_samples=(),
            verifier_results=(),
            submitted_jobs=(),
            preflight=preflight,
            status="blocked_by_cost_cap",
            screen_reports=(),
            finalist_selection=None,
            full_reports=(),
            qualified=(),
            progress=phase_progress,
        )

    # This is intentionally the first secret read and first point at which a
    # billable client can exist.
    enforce_cost_cap(preflight)
    api_key = dependencies.load_api_key()
    client = dependencies.make_client(api_key, cache)
    if type(client) is OpenRouterClient:
        try:
            # Reconcile prior immutable billing locally before evaluating the
            # current run's authorization. A failed current preflight must not
            # erase historical charges or recovery evidence.
            client.cost_ledger.bootstrap(
                cache,
                (*catalog.resolved.generator_routes, catalog.resolved.verifier_route),
            )
            if client.require_byok_preflight:
                byok_evidence = client.verify_no_byok()
                builder.write_json("byok_preflight.json", byok_evidence.as_record())
        except (
            CostCapExceeded,
            CostJournalRecoveryRequired,
            OpenRouterCostPolicyError,
        ) as error:
            if isinstance(error, OpenRouterCostPolicyError):
                if error.evidence is not None:
                    builder.write_json(
                        "byok_preflight.json",
                        error.evidence.as_record(),
                    )
                client.cost_ledger.halt("byok_preflight_failed")
            attempted = _attempted_jobs(cache, all_jobs)
            _write_runtime_cost_stop(
                builder,
                cache,
                attempted,
                client.cost_ledger,
                all_route_locks,
            )
            return _publish_stopped_result(
                builder,
                paths,
                cache,
                verifier_route=catalog.resolved.verifier_route,
                all_generation_jobs=all_jobs,
                generated_samples=(),
                verifier_results=(),
                submitted_jobs=attempted,
                preflight=preflight,
                status=(
                    "provider_billing_unknown"
                    if isinstance(
                        error,
                        (CostJournalRecoveryRequired, OpenRouterCostPolicyError),
                    )
                    else "blocked_by_runtime_cost_cap"
                ),
                screen_reports=(),
                finalist_selection=None,
                full_reports=(),
                qualified=(),
                progress=phase_progress,
            )

    screen_planned_jobs = _screen_jobs(all_jobs)
    try:
        screen_samples, screen_measurements, screen_reports = phase_progress.run(
            PHASES[2],
            lambda: _run_screen(
                builder,
                all_jobs,
                references,
                dependencies,
                client,
            ),
        )
    except (
        CatalogRouteRevalidationError,
        CostCapExceeded,
        CostJournalRecoveryRequired,
        OpenRouterBillingUnknown,
        OpenRouterContractError,
    ) as error:
        attempted = _attempted_jobs(cache, screen_planned_jobs)
        assert type(client) is OpenRouterClient
        _write_runtime_cost_stop(
            builder,
            cache,
            attempted,
            client.cost_ledger,
            all_route_locks,
        )
        return _publish_stopped_result(
            builder,
            paths,
            cache,
            verifier_route=catalog.resolved.verifier_route,
            all_generation_jobs=all_jobs,
            generated_samples=(),
            verifier_results=(),
            submitted_jobs=attempted,
            preflight=preflight,
            status=_interrupted_generation_status(error, client),
            screen_reports=(),
            finalist_selection=None,
            full_reports=(),
            qualified=(),
            progress=phase_progress,
        )
    screen_jobs = tuple(sample.job for sample in screen_samples)
    screen_billing = _cached_attempt_billing(cache, screen_jobs)
    screen_costs = dict(screen_billing.generation_by_route)
    screen_reports = tuple(
        replace(report, billed_cost_usd=float(screen_costs[report.route_id]))
        for report in screen_reports
    )
    finalist_selection = select_screen_finalists(screen_reports)
    builder.write_json(
        "finalist_decision.json", _quality_selection_record(finalist_selection)
    )
    if finalist_selection.outcome is QualityOutcome.NO_QUALITY_QUALIFIED_ROUTE:
        _write_cost_actuals(
            builder,
            screen_samples,
            (),
            (),
            screen_billing,
            quality_reports=screen_reports,
            qualified_route_ids=(),
        )
        _write_cost_observations(builder, screen_samples, (), (), screen_billing)
        return _publish_stopped_result(
            builder,
            paths,
            cache,
            verifier_route=catalog.resolved.verifier_route,
            all_generation_jobs=all_jobs,
            generated_samples=screen_samples,
            verifier_results=(),
            submitted_jobs=screen_jobs,
            preflight=preflight,
            status="no_quality_qualified_route",
            screen_reports=screen_reports,
            finalist_selection=finalist_selection,
            full_reports=(),
            qualified=(),
            progress=phase_progress,
        )

    finalist_planned_jobs = tuple(
        job
        for job in all_jobs
        if job.route.route_id in finalist_selection.route_ids
    )
    try:
        full_samples, full_measurements = phase_progress.run(
            PHASES[3],
            lambda: _expand_finalists(
                builder,
                all_jobs,
                screen_samples,
                screen_measurements,
                finalist_selection.route_ids,
                dependencies,
                client,
            ),
        )
    except (
        CatalogRouteRevalidationError,
        CostCapExceeded,
        CostJournalRecoveryRequired,
        OpenRouterBillingUnknown,
        OpenRouterContractError,
    ) as error:
        submitted_plan = (*screen_jobs, *finalist_planned_jobs)
        attempted = _attempted_jobs(cache, submitted_plan)
        assert type(client) is OpenRouterClient
        _write_runtime_cost_stop(
            builder,
            cache,
            attempted,
            client.cost_ledger,
            all_route_locks,
        )
        return _publish_stopped_result(
            builder,
            paths,
            cache,
            verifier_route=catalog.resolved.verifier_route,
            all_generation_jobs=all_jobs,
            generated_samples=screen_samples,
            verifier_results=(),
            submitted_jobs=attempted,
            preflight=preflight,
            status=_interrupted_generation_status(error, client),
            screen_reports=screen_reports,
            finalist_selection=finalist_selection,
            full_reports=(),
            qualified=(),
            progress=phase_progress,
        )
    reference_jobs, generated_jobs = _verifier_jobs(
        references,
        catalog.resolved.verifier_route,
        full_samples,
    )
    try:
        reference_verifiers, generated_verifiers, full_reports = phase_progress.run(
            PHASES[4],
            lambda: _verify_and_evaluate(
                builder,
                references,
                catalog.resolved.verifier_route,
                full_samples,
                full_measurements,
                finalist_selection.route_ids,
                client,
                dependencies,
            ),
        )
    except (
        CatalogRouteRevalidationError,
        CostCapExceeded,
        CostJournalRecoveryRequired,
        OpenRouterBillingUnknown,
        OpenRouterContractError,
    ) as error:
        generation_jobs = tuple(sample.job for sample in full_samples) + screen_jobs
        attempted = _attempted_jobs(
            cache,
            (*generation_jobs, *reference_jobs, *generated_jobs),
        )
        assert type(client) is OpenRouterClient
        _write_runtime_cost_stop(
            builder,
            cache,
            attempted,
            client.cost_ledger,
            all_route_locks,
        )
        return _publish_stopped_result(
            builder,
            paths,
            cache,
            verifier_route=catalog.resolved.verifier_route,
            all_generation_jobs=all_jobs,
            generated_samples=full_samples,
            verifier_results=(),
            submitted_jobs=attempted,
            preflight=preflight,
            status=_interrupted_generation_status(error, client),
            screen_reports=screen_reports,
            finalist_selection=finalist_selection,
            full_reports=(),
            qualified=(),
            progress=phase_progress,
        )
    attributed_costs = _attributed_full_route_costs(
        cache,
        full_samples,
        generated_verifiers,
    )
    full_reports = tuple(
        replace(
            report,
            billed_cost_usd=float(attributed_costs[report.route_id]),
        )
        for report in full_reports
    )
    qualified_selection = select_full_quality_routes(
        full_reports,
        finalist_order=finalist_selection.route_ids,
    )
    qualified = qualified_selection.route_ids
    builder.write_json(
        "quality_comparisons.json",
        {
            "audited_route_ids": list(finalist_selection.route_ids),
            "qualified_route_ids": list(qualified),
        },
    )
    builder.write_json(
        "quality_details.json",
        {
            "full_reports": [_quality_report_record(item) for item in full_reports],
            "screen_reports": [_quality_report_record(item) for item in screen_reports],
            "selection": _quality_selection_record(qualified_selection),
        },
    )
    submitted_jobs = tuple(
        sample.job for sample in screen_samples + full_samples
    ) + tuple(
        verified.job for verified in reference_verifiers + generated_verifiers
    )
    billing = _cached_attempt_billing(cache, submitted_jobs)
    _write_cost_actuals(
        builder,
        screen_samples,
        full_samples,
        reference_verifiers + generated_verifiers,
        billing,
        quality_reports=full_reports,
        qualified_route_ids=qualified,
    )
    _write_cost_observations(
        builder,
        screen_samples,
        full_samples,
        reference_verifiers + generated_verifiers,
        billing,
    )
    if type(client) is OpenRouterClient:
        builder.write_json(
            "runtime_cost_ledger.json",
            client.cost_ledger.snapshot().as_record(),
        )
    if qualified_selection.outcome is QualityOutcome.NO_QUALITY_QUALIFIED_ROUTE:
        return _publish_stopped_result(
            builder,
            paths,
            cache,
            verifier_route=catalog.resolved.verifier_route,
            all_generation_jobs=all_jobs,
            generated_samples=screen_samples + full_samples,
            verifier_results=reference_verifiers + generated_verifiers,
            submitted_jobs=submitted_jobs,
            preflight=preflight,
            status="no_quality_qualified_route",
            screen_reports=screen_reports,
            finalist_selection=finalist_selection,
            full_reports=full_reports,
            qualified=(),
            progress=phase_progress,
        )

    audit_result = phase_progress.run(
        PHASES[5],
        lambda: _try_build_and_write_audit(
            builder,
            references,
            full_samples,
            full_measurements,
            reference_verifiers,
            generated_verifiers,
            finalist_selection.route_ids,
        ),
    )
    if audit_result is None:
        builder.write_json(
            "audit_feasibility.json",
            _audit_feasibility_record(full_samples, finalist_selection.route_ids),
        )
        return _publish_stopped_result(
            builder,
            paths,
            cache,
            verifier_route=catalog.resolved.verifier_route,
            all_generation_jobs=all_jobs,
            generated_samples=screen_samples + full_samples,
            verifier_results=reference_verifiers + generated_verifiers,
            submitted_jobs=submitted_jobs,
            preflight=preflight,
            status="audit_insufficient_accepted_samples",
            screen_reports=screen_reports,
            finalist_selection=finalist_selection,
            full_reports=full_reports,
            qualified=qualified,
            progress=phase_progress,
        )
    packet, key = audit_result
    _write_execution_manifests(
        builder,
        cache,
        all_jobs,
        screen_samples + full_samples,
        reference_verifiers + generated_verifiers,
        catalog.resolved.verifier_route,
        attempted_jobs=submitted_jobs,
    )
    _copy_raw_cache(builder, cache, submitted_jobs)
    builder.write_json(
        "status.json",
        {
            "audit_sha256": packet.audit_sha256,
            "phase": 1,
            "status": "awaiting_human_audit",
        },
    )
    # Complete the persistent phase event before sealing the manifest.  A
    # progress writer cannot append after the staging directory is renamed.
    phase_progress.run(PHASES[6], lambda: None)
    directory = _finalize_and_promote(builder, paths.destination)
    return Phase1RunResult(
        directory=directory,
        status="awaiting_human_audit",
        conservative_cost_usd=preflight.conservative_usd,
        qualified_route_ids=qualified,
        audit_sha256=packet.audit_sha256,
    )


def main() -> None:
    """Run the one production preset and stop at the mandatory audit gate."""
    repository_root = Path(__file__).resolve().parents[5]
    paths = Phase1Paths.from_repository(repository_root)
    if paths.destination.exists() and paths.destination.is_dir():
        validate_phase1_semantics(paths.destination)
        print(f"Existing Phase 1 artifact: {paths.destination}")
        print("No source, catalog, GPU, or OpenRouter work was repeated.")
        return
    paths.destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_directory = Path(
        tempfile.mkdtemp(prefix="tinyworlds-v2-phase1-", dir=paths.destination.parent)
    )
    print(f"Temporary artifact directory: {temporary_directory}", flush=True)
    with Phase1Progress(temporary_directory) as progress:
        result = run_phase1(
            temporary_directory,
            paths,
            production_dependencies(paths),
            progress=progress,
        )
    print(f"Phase 1 status: {result.status}")
    print(f"Phase 1 artifact: {result.directory}")
    if result.audit_sha256 is not None:
        print(f"Mandatory audit digest: {result.audit_sha256}")
        print(f"Open the audit: {result.directory / 'audit.html'}")


def _prepare_production_references(
    paths: Phase1Paths,
    measure: MeasureStories,
    progress_callback: ReferenceProgress,
) -> Phase1ReferenceCorpus:
    # Selection streams through ``select_archive_source_records``, whose first
    # operation authenticates the complete pinned archive.  Do not perform a
    # second 1.6 GB checksum pass here.
    selections = select_archive_source_records(
        paths.archive,
        seed=PHASE1_SELECTION_SEED,
    )
    archive_story_hashes = frozenset(
        record.normalized_story_sha256
        for cohort in (
            selections.prompt_metadata_records,
            selections.reference_story_records,
            selections.paired_records,
        )
        for record in cohort
    )
    validation = select_validation_story_records(
        paths.validation,
        seed=PHASE1_SELECTION_SEED,
        exclude_normalized_story_sha256=archive_story_hashes,
    )
    inputs = build_phase1_reference_inputs(selections, validation)
    paired_by_source = {record.record_id: record for record in selections.paired_records}
    stories = tuple(
        NllStory(record.record_id, record.story_text)
        for record in inputs.reference_records
    ) + tuple(
        NllStory(brief.source_record_id, brief.matched_reference_text)
        for brief in inputs.briefs
    )
    measured = measure(tuple(sorted(stories, key=lambda item: item.record_id)), progress_callback)
    values = measured.by_id
    observations = prepare_reference_observations(
        inputs.reference_records,
        inputs.reference_annotations,
        model_token_ids_by_record_id={
            record.record_id: values[record.record_id].model_token_ids
            for record in inputs.reference_records
        },
        normalized_nll_by_record_id={
            record.record_id: values[record.record_id].normalized_nll
            for record in inputs.reference_records
        },
        worker_count=REFERENCE_SURFACE_WORKERS,
    )
    paired_observations = tuple(
        observe_reference(
            ReferenceRecord(
                brief.source_record_id,
                brief.matched_reference_text,
                prompt_text=brief.prompt_text,
                source_model="GPT-4",
            ),
            model_token_ids=values[brief.source_record_id].model_token_ids,
            normalized_nll=values[brief.source_record_id].normalized_nll,
            feature_labels=brief.requested_features,
            required_words=brief.required_words,
        )
        for brief in inputs.briefs
    )
    profile = build_reference_profile(observations)
    paired_profile = build_reference_profile(paired_observations)
    source_manifest: JsonObject = {
        "archive": {
            "dataset_id": TINYSTORIES_ALL_DATA_SOURCE.dataset_id,
            "filename": TINYSTORIES_ALL_DATA_SOURCE.archive_file.filename,
            "revision": TINYSTORIES_ALL_DATA_SOURCE.revision,
            "sha256": TINYSTORIES_ALL_DATA_SOURCE.archive_file.sha256,
            "size_bytes": TINYSTORIES_ALL_DATA_SOURCE.archive_file.size_bytes,
        },
        "counts": {
            "archive_reference": len(selections.reference_story_records),
            "neutral_briefs": len(inputs.briefs),
            "prompt_metadata": len(selections.prompt_metadata_records),
            "reference_profile": len(inputs.reference_records),
            "validation_reference": len(validation),
        },
        "selection_seed": PHASE1_SELECTION_SEED,
        "story_identity_policy": (
            "unicode-nfkc-casefold-whitespace-collapse-sha256-v1"
        ),
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
                    tuple(canonical_neutral_story_brief(item) for item in inputs.briefs),
                ),
                (
                    "paired_reference_observations.jsonl",
                    tuple(canonical_reference_observation(item) for item in paired_observations),
                ),
                (
                    "prompt_metadata_sample.jsonl",
                    tuple(
                        canonical_prompt_metadata_record(item)
                        for item in selections.prompt_metadata_records
                    ),
                ),
                (
                    "reference_annotations.jsonl",
                    tuple(
                        canonical_reference_annotation(item)
                        for item in inputs.reference_annotations
                    ),
                ),
                (
                    "reference_observations.jsonl",
                    tuple(canonical_reference_observation(item) for item in observations),
                ),
                (
                    "reference_story_sample.jsonl",
                    tuple(canonical_reference_record(item) for item in inputs.reference_records),
                ),
                (
                    "validation_source_sample.jsonl",
                    tuple(canonical_validation_record(item) for item in validation),
                ),
            )
        )
    )
    statistics: JsonObject = {
        "ingredient_profile": canonical_prompt_ingredient_profile(inputs.ingredient_profile),
        "nll_runtime": measured.runtime,
        "paired_reference_profile": _reference_profile_record(paired_profile),
        "reference_profile": _reference_profile_record(profile),
        "paired_source_record_ids": [
            paired_by_source[brief.source_record_id].record_id for brief in inputs.briefs
        ],
    }
    return Phase1ReferenceCorpus(
        briefs=inputs.briefs,
        reference_records=inputs.reference_records,
        reference_observations=observations,
        paired_reference_observations=paired_observations,
        reference_profile=profile,
        paired_reference_profile=paired_profile,
        expected_feature_rates=inputs.ingredient_profile.narrative_feature_rates,
        source_manifest=source_manifest,
        source_artifacts=source_artifacts,
        reference_statistics=statistics,
    )


def _catalog_evidence(transport: HttpxTransport) -> CatalogEvidence:
    payloads = fetch_catalog_payloads(transport)
    return CatalogEvidence(payloads, resolve_openrouter_catalog(payloads))


def _revalidate_catalog_route(
    transport: HttpxTransport,
    locked: RouteLock,
) -> RouteLock:
    """Refresh public catalog evidence immediately before one paid batch."""
    fresh = _catalog_evidence(transport).resolved
    routes = (*fresh.generator_routes, fresh.verifier_route)
    observed = next(
        (route for route in routes if route.route_id == locked.route_id),
        None,
    )
    if observed is None:
        raise ValueError(f"fresh catalog omitted locked route {locked.route_id!r}")
    validate_route_semantics(locked, observed)
    return observed


def _prepare_catalog_and_costs(
    builder: Phase1ArtifactBuilder,
    references: Phase1ReferenceCorpus,
    dependencies: Phase1Dependencies,
) -> tuple[CatalogEvidence, tuple[GenerationJob, ...], CostPreflight]:
    catalog = dependencies.fetch_catalog()
    jobs = build_generation_jobs(
        references.briefs,
        CANDIDATE_MODELS,
        catalog.resolved.generator_routes,
    )
    if len(references.briefs) != PHASE1_FULL_COUNT:
        # Small injected fixtures are supported only by explicitly bypassing
        # the production runner's funnel selectors in focused helper tests.
        raise ValueError(
            f"Phase 1 production requires exactly {PHASE1_FULL_COUNT} neutral briefs"
        )
    _write_catalog_artifacts(builder, catalog)
    _write_planned_route_artifacts(builder, jobs, catalog.resolved.generator_routes)
    builder.write_json(
        "configuration.json",
        _configuration_record(references, catalog.resolved),
    )
    preflight = _build_phase1_cost_preflight(
        references,
        jobs,
        catalog.resolved,
        dependencies.encode_text,
    )
    builder.write_json("cost_estimates.json", preflight.as_record())
    return catalog, jobs, preflight


def _build_phase1_cost_preflight(
    references: Phase1ReferenceCorpus,
    jobs: Sequence[GenerationJob],
    routes: ResolvedRouteCatalog,
    encode_text: EncodeText,
) -> CostPreflight:
    jobs_by_route = {
        route.route_id: tuple(
            job for job in jobs if job.route.route_id == route.route_id
        )
        for route in routes.generator_routes
    }
    matched_reference_output = math.ceil(
        sum(
            len(encode_text(brief.matched_reference_text))
            for brief in references.briefs
        )
        / len(references.briefs)
    )
    generator_bounds = tuple(
        (
            route,
            math.ceil(
                sum(
                    len(encode_text(job.request.body_json))
                    for job in jobs_by_route[route.route_id]
                )
                / len(jobs_by_route[route.route_id])
            ),
            max(
                2 * len(encode_text(job.request.body_json)) + 512
                for job in jobs_by_route[route.route_id]
            ),
        )
        for route in routes.generator_routes
    )
    if any(
        conservative_input > PHASE1_PROMPT_TOKEN_UPPER_BOUND
        for _route, _expected_input, conservative_input in generator_bounds
    ):
        raise ValueError(
            "a generator request exceeds the catalog override prompt-token bound"
        )
    # Every route receives 50 screen requests, but only three routes can receive
    # the remaining 150. Reserve those expansions against the three most
    # expensive exact locked routes. This upper-bounds every possible finalist
    # set without incorrectly budgeting seven 200-story finalists.
    finalist_reserve_ids = frozenset(
        route.route_id
        for route, _expected_input_tokens, conservative_input_tokens in sorted(
            generator_bounds,
            key=lambda item: (
                -(
                    Decimal(item[2]) * Decimal(item[0].input_usd_per_million)
                    + Decimal(512) * Decimal(item[0].output_usd_per_million)
                ),
                tuple(routes.generator_routes).index(item[0]),
            ),
        )[:3]
    )
    workloads = tuple(
        RouteWorkload(
            route,
            TokenWorkload(
                label=(
                    "phase1-generator-screen-plus-finalist-reserve"
                    if route.route_id in finalist_reserve_ids
                    else "phase1-generator-screen"
                ),
                request_count=(
                    PHASE1_FULL_COUNT
                    if route.route_id in finalist_reserve_ids
                    else PHASE1_SCREEN_COUNT
                ),
                input_tokens_per_request=expected_input_tokens,
                output_tokens_per_request=matched_reference_output,
                conservative_input_tokens_per_request=conservative_input_tokens,
                conservative_output_tokens_per_request=512,
                retry_allowance_basis_points=PHASE1_RETRY_ALLOWANCE_BASIS_POINTS,
            ),
        )
        for route, expected_input_tokens, conservative_input_tokens in generator_bounds
    )
    verifier_projection_jobs = tuple(
        build_verifier_job(
            source_id=f"preflight-reference:{brief.brief_id}",
            pair_id=brief.brief_id,
            brief=brief,
            story=brief.matched_reference_text,
            model=VERIFIER_MODEL,
            route=routes.verifier_route,
        )
        for brief in references.briefs
    )
    verifier_expected_input = math.ceil(
        sum(len(encode_text(job.request.body_json)) for job in verifier_projection_jobs)
        / len(verifier_projection_jobs)
    )
    verifier_conservative_input = max(
        2 * len(encode_text(job.request.body_json)) + 512
        for job in verifier_projection_jobs
    )
    if verifier_conservative_input > PHASE1_PROMPT_TOKEN_UPPER_BOUND:
        raise ValueError(
            "a verifier request exceeds the catalog override prompt-token bound"
        )
    verifier = RouteWorkload(
        routes.verifier_route,
        TokenWorkload(
            label="phase1-blind-verifier-conservative-800",
            request_count=PHASE1_FULL_COUNT * 4,
            input_tokens_per_request=verifier_expected_input,
            output_tokens_per_request=128,
            conservative_input_tokens_per_request=verifier_conservative_input,
            conservative_output_tokens_per_request=256,
            retry_allowance_basis_points=PHASE1_RETRY_ALLOWANCE_BASIS_POINTS,
        ),
    )
    return build_cost_preflight((*workloads, verifier), hard_cap_usd=PHASE1_HARD_CAP_USD)


def _screen_jobs(all_jobs: Sequence[GenerationJob]) -> tuple[GenerationJob, ...]:
    """Return the fixed 7×50 first-stage funnel in canonical route order."""
    return tuple(
        job
        for model in CANDIDATE_MODELS
        for job in tuple(
            candidate
            for candidate in all_jobs
            if candidate.route.route_id == model.route_id
        )[:PHASE1_SCREEN_COUNT]
    )


def _run_screen(
    builder: Phase1ArtifactBuilder,
    all_jobs: Sequence[GenerationJob],
    references: Phase1ReferenceCorpus,
    dependencies: Phase1Dependencies,
    client: CachedGenerationClient,
) -> tuple[tuple[GeneratedSample, ...], MeasurementBatch, tuple[RouteQualityReport, ...]]:
    screen_jobs = _screen_jobs(all_jobs)
    screen_source_ids = frozenset(job.brief.source_record_id for job in screen_jobs)
    screen_reference_observations = tuple(
        observation
        for observation in references.paired_reference_observations
        if observation.record_id in screen_source_ids
    )
    if len(screen_reference_observations) != PHASE1_SCREEN_COUNT:
        raise ValueError("screen jobs do not have one matched genuine reference each")
    screen_reference_profile = build_reference_profile(
        screen_reference_observations
    )
    samples = _execute_generation_by_route(
        builder,
        screen_jobs,
        client,
        "screen",
        dependencies.revalidate_route,
    )
    measured = _measure_accepted(samples, dependencies.measure_stories)
    _write_measurement_batch(builder, "screen", measured)
    reports = tuple(
        evaluate_route_quality(
            _quality_observations(
                tuple(sample for sample in samples if sample.job.route.route_id == model.route_id),
                measured,
            ),
            references.reference_profile,
            phase=QualityPhase.SCREEN,
            matched_reference_profile=screen_reference_profile,
            expected_feature_rates=references.expected_feature_rates,
        )
        for model in CANDIDATE_MODELS
    )
    return samples, measured, reports


def _expand_finalists(
    builder: Phase1ArtifactBuilder,
    all_jobs: Sequence[GenerationJob],
    screen_samples: tuple[GeneratedSample, ...],
    screen_measurements: MeasurementBatch,
    finalist_ids: tuple[str, ...],
    dependencies: Phase1Dependencies,
    client: CachedGenerationClient,
) -> tuple[tuple[GeneratedSample, ...], MeasurementBatch]:
    extra_jobs = tuple(
        job
        for route_id in finalist_ids
        for job in tuple(
            candidate
            for candidate in all_jobs
            if candidate.route.route_id == route_id
        )[PHASE1_SCREEN_COUNT:PHASE1_FULL_COUNT]
    )
    extras = _execute_generation_by_route(
        builder,
        extra_jobs,
        client,
        "finalist-expansion",
        dependencies.revalidate_route,
    )
    extra_measurements = _measure_accepted(extras, dependencies.measure_stories)
    _write_measurement_batch(builder, "finalist_expansion", extra_measurements)
    finalist_samples = tuple(
        sample
        for sample in screen_samples
        if sample.job.route.route_id in finalist_ids
    ) + extras
    combined = MeasurementBatch(
        measurements=tuple(
            sorted(
                screen_measurements.measurements + extra_measurements.measurements,
                key=lambda item: item.record_id,
            )
        ),
        runtime={
            "expansion": extra_measurements.runtime,
            "screen": screen_measurements.runtime,
        },
    )
    return finalist_samples, combined


def _verify_and_evaluate(
    builder: Phase1ArtifactBuilder,
    references: Phase1ReferenceCorpus,
    verifier_route: RouteLock,
    full_samples: tuple[GeneratedSample, ...],
    full_measurements: MeasurementBatch,
    finalist_ids: tuple[str, ...],
    client: CachedGenerationClient,
    dependencies: Phase1Dependencies,
) -> tuple[tuple[VerifiedStory, ...], tuple[VerifiedStory, ...], tuple[RouteQualityReport, ...]]:
    paired = {item.record_id: item for item in references.paired_reference_observations}
    reference_jobs, generated_jobs = _verifier_jobs(
        references, verifier_route, full_samples
    )

    reference_verified = _execute_verifier_batches(
        builder,
        reference_jobs,
        client,
        "genuine-reference",
        dependencies.revalidate_route,
    )
    generated_verified = _execute_verifier_batches(
        builder,
        generated_jobs,
        client,
        "generated-finalist",
        dependencies.revalidate_route,
    )
    reference_verifier_failed = any(item.payload is None for item in reference_verified)
    # Missing genuine ratings must never make the generator comparison easier.
    reference_means = tuple(
        (
            dimension,
            sum(
                5.0
                if item.payload is None
                else float(getattr(item.payload, dimension))
                for item in reference_verified
            )
            / len(reference_verified),
        )
        for dimension in BLIND_VERIFIER_DIMENSIONS
    )
    verifier_by_source = {item.job.source_id: item for item in generated_verified}
    reports = tuple(
        replace(
            report,
            failures=report.failures
            + (("reference_verifier_failure",) if reference_verifier_failed else ()),
        )
        for route_id in finalist_ids
        for report in (
            evaluate_route_quality(
                _quality_observations(
                    tuple(
                        sample
                        for sample in full_samples
                        if sample.job.route.route_id == route_id
                    ),
                    full_measurements,
                    verifier_by_source=verifier_by_source,
                    verifier_required=True,
                ),
                references.reference_profile,
                phase=QualityPhase.FULL,
                reference_blind_verifier_means=reference_means,
                matched_reference_profile=references.paired_reference_profile,
                expected_feature_rates=references.expected_feature_rates,
            ),
        )
    )
    if set(paired) != {brief.source_record_id for brief in references.briefs}:
        raise ValueError("paired reference observations changed before verification")
    return reference_verified, generated_verified, reports


def _verifier_jobs(
    references: Phase1ReferenceCorpus,
    verifier_route: RouteLock,
    full_samples: Sequence[GeneratedSample],
) -> tuple[tuple[VerifierJob, ...], tuple[VerifierJob, ...]]:
    """Build the fixed genuine/generated verifier requests for execution or stops."""
    reference_jobs = tuple(
        build_verifier_job(
            source_id=f"reference:{brief.brief_id}",
            pair_id=brief.brief_id,
            brief=brief,
            story=brief.matched_reference_text,
            model=VERIFIER_MODEL,
            route=verifier_route,
        )
        for brief in references.briefs
    )
    generated_jobs = tuple(
        build_verifier_job(
            source_id=sample.sample_id,
            pair_id=sample.job.brief.brief_id,
            brief=sample.job.brief,
            story=sample.payload.story,
            model=VERIFIER_MODEL,
            route=verifier_route,
        )
        for sample in full_samples
        if sample.validation.accepted and sample.payload is not None
    )
    return reference_jobs, generated_jobs


def _execute_verifier_batches(
    builder: Phase1ArtifactBuilder,
    jobs: Sequence[VerifierJob],
    client: CachedGenerationClient,
    stage: str,
    revalidate_route: RevalidateRoute = _skip_route_revalidation,
) -> tuple[VerifiedStory, ...]:
    """Persist deterministic verifier batches as soon as each batch completes."""
    batch_size = 50
    completed: list[VerifiedStory] = []
    for start in range(0, len(jobs), batch_size):
        fresh_route = _revalidate_before_paid_batch(
            client,
            revalidate_route,
            jobs[start].route,
        )
        batch_jobs = tuple(
            replace(job, route=fresh_route)
            for job in jobs[start : start + batch_size]
        )
        batch = execute_verifier_jobs(
            batch_jobs,
            client,
            max_workers=PHASE1_GENERATION_WORKERS,
        )
        completed.extend(batch)
        _append_jsonl_records(
            builder.root / "verifier_results.jsonl",
            tuple(item.as_record() for item in batch),
        )
        _append_jsonl(
            builder.root / "sequential_results.jsonl",
            {
                "event": "verifier_batch_completed",
                "request_count": len(batch),
                "stage": stage,
                "start_index": start,
            },
        )
    return tuple(completed)


def _build_and_write_audit(
    builder: Phase1ArtifactBuilder,
    references: Phase1ReferenceCorpus,
    samples: tuple[GeneratedSample, ...],
    measurements: MeasurementBatch,
    reference_verifiers: tuple[VerifiedStory, ...],
    generated_verifiers: tuple[VerifiedStory, ...],
    audit_route_ids: tuple[str, ...],
) -> tuple[BlindedAuditPacket, BlindedAuditKey]:
    paired = {item.record_id: item for item in references.paired_reference_observations}
    measurement_by_id = measurements.by_id
    reference_verifier_by_pair = {item.job.pair_id: item for item in reference_verifiers}
    generated_verifier_by_source = {item.job.source_id: item for item in generated_verifiers}
    reference_records = tuple(
        AuditSourceRecord(
            source_id=f"reference:{brief.brief_id}",
            pair_id=brief.brief_id,
            story_text=brief.matched_reference_text,
            source_prompt=brief.prompt_text,
            token_count=len(paired[brief.source_record_id].model_token_ids),
            base_normalized_nll=paired[brief.source_record_id].normalized_nll,
            automated_style_scores=_style_scores(
                reference_verifier_by_pair[brief.brief_id].payload
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
            token_count=len(measurement_by_id[sample.sample_id].model_token_ids),
            base_normalized_nll=measurement_by_id[sample.sample_id].normalized_nll,
            automated_style_scores=_style_scores(
                generated_verifier_by_source[sample.sample_id].payload
            ),
            source_kind=AuditSourceKind.GENERATED,
            route_id=sample.job.route.route_id,
        )
        for sample in samples
        if sample.job.route.route_id in audit_route_ids
        and sample.validation.accepted
        and sample.payload is not None
    )
    packet, key = build_blinded_audit(
        reference_records,
        generated_records,
        finalist_order=audit_route_ids,
        seed=PHASE1_AUDIT_SEED,
        reference_count=PHASE1_AUDIT_COUNT,
        generated_count=PHASE1_AUDIT_COUNT,
    )
    builder.write_bytes("audit_packet.json", encode_blinded_audit_packet(packet))
    builder.write_bytes("audit_key.json", encode_blinded_audit_key(key))
    builder.write_bytes("audit.html", render_audit_html(packet).encode("utf-8"))
    return packet, key


def _try_build_and_write_audit(
    builder: Phase1ArtifactBuilder,
    references: Phase1ReferenceCorpus,
    samples: tuple[GeneratedSample, ...],
    measurements: MeasurementBatch,
    reference_verifiers: tuple[VerifiedStory, ...],
    generated_verifiers: tuple[VerifiedStory, ...],
    audited_route_ids: tuple[str, ...],
) -> tuple[BlindedAuditPacket, BlindedAuditKey] | None:
    """Convert only mathematical audit-allocation failure into a stop result."""
    try:
        return _build_and_write_audit(
            builder,
            references,
            samples,
            measurements,
            reference_verifiers,
            generated_verifiers,
            audited_route_ids,
        )
    except AuditAllocationError:
        return None


def _audit_feasibility_record(
    samples: Sequence[GeneratedSample],
    audited_route_ids: tuple[str, ...],
) -> JsonObject:
    """Record exact candidates and balanced quotas when audit matching is impossible."""
    quotient, remainder = divmod(PHASE1_AUDIT_COUNT, len(audited_route_ids))
    routes: list[JsonObject] = []
    union_pair_ids: set[str] = set()
    for index, route_id in enumerate(audited_route_ids):
        pair_ids = tuple(
            sorted(
                sample.job.brief.brief_id
                for sample in samples
                if sample.job.route.route_id == route_id
                and sample.validation.accepted
                and sample.payload is not None
            )
        )
        union_pair_ids.update(pair_ids)
        routes.append(
            {
                "eligible_pair_ids": list(pair_ids),
                "required_count": quotient + (index < remainder),
                "route_id": route_id,
            }
        )
    return {
        "audited_route_ids": list(audited_route_ids),
        "failure_reason": "no_distinct_balanced_assignment",
        "generated_audit_count": PHASE1_AUDIT_COUNT,
        "routes": routes,
        "union_eligible_pair_ids": sorted(union_pair_ids),
    }


def _execute_generation_by_route(
    builder: Phase1ArtifactBuilder,
    jobs: Sequence[GenerationJob],
    client: CachedGenerationClient,
    stage: str,
    revalidate_route: RevalidateRoute = _skip_route_revalidation,
) -> tuple[GeneratedSample, ...]:
    route_ids = tuple(
        model.route_id
        for model in CANDIDATE_MODELS
        if any(job.route.route_id == model.route_id for job in jobs)
    )
    completed: list[GeneratedSample] = []
    for route_id in route_ids:
        route_jobs = tuple(
            job for job in jobs if job.route.route_id == route_id
        )
        fresh_route = _revalidate_before_paid_batch(
            client,
            revalidate_route,
            route_jobs[0].route,
        )
        route_jobs = tuple(
            replace(job, route=fresh_route) for job in route_jobs
        )
        samples = execute_generation_jobs(
            route_jobs,
            client,
            max_workers=PHASE1_GENERATION_WORKERS,
        )
        completed.extend(samples)
        _append_jsonl_records(
            builder.root / "generator_bakeoff.jsonl",
            tuple(sample.as_record() for sample in samples),
        )
        for accepted, filename in ((True, "accepted.jsonl"), (False, "rejected.jsonl")):
            _append_jsonl_records(
                builder.root / "routes" / route_id / filename,
                tuple(
                    sample.as_record()
                    for sample in samples
                    if sample.validation.accepted is accepted
                ),
            )
        _append_jsonl_records(
            builder.root / "routes" / route_id / "raw_responses.jsonl",
            tuple(
                {
                    "billed_cost_usd": sample.billed_cost_usd,
                    "error_kind": sample.error_kind,
                    "generation_id": sample.generation_id,
                    "input_tokens": sample.input_tokens,
                    "output_tokens": sample.output_tokens,
                    "request_sha256": sample.job.request.request_sha256,
                    "sample_id": sample.sample_id,
                }
                for sample in samples
            ),
        )
        _append_jsonl(
            builder.root / "sequential_results.jsonl",
            {
                "accepted": sum(item.validation.accepted for item in samples),
                "event": "generation_batch_completed",
                "request_count": len(samples),
                "route_id": route_id,
                "stage": stage,
            },
        )
    return tuple(completed)


def _revalidate_before_paid_batch(
    client: CachedGenerationClient,
    revalidate_route: RevalidateRoute,
    route: RouteLock,
) -> RouteLock:
    """Halt all workers before POST if the fresh semantic route pin drifted."""
    try:
        observed = revalidate_route(route)
        if type(observed) is not RouteLock:
            raise TypeError("route revalidation must return a RouteLock")
        validate_route_semantics(route, observed)
        return observed
    except Exception as error:
        if type(client) is OpenRouterClient:
            client.cost_ledger.halt("catalog_route_drift")
        raise CatalogRouteRevalidationError(
            f"fresh catalog revalidation failed for route {route.route_id!r}"
        ) from error


def _measure_accepted(
    samples: Sequence[GeneratedSample],
    measure: MeasureStories,
) -> MeasurementBatch:
    stories = tuple(
        NllStory(sample.sample_id, sample.payload.story)
        for sample in samples
        if sample.validation.accepted and sample.payload is not None
    )
    if not stories:
        return MeasurementBatch((), {"status": "no_accepted_stories"})
    return measure(tuple(sorted(stories, key=lambda item: item.record_id)), lambda _: None)


def _quality_observations(
    samples: tuple[GeneratedSample, ...],
    measurements: MeasurementBatch,
    *,
    verifier_by_source: Mapping[str, VerifiedStory] | None = None,
    verifier_required: bool = False,
) -> tuple[GeneratedObservation, ...]:
    measured = measurements.by_id
    verifiers = {} if verifier_by_source is None else verifier_by_source
    return tuple(
        replace(
            generated_observation(
                sample,
                model_token_ids=(
                    measured[sample.sample_id].model_token_ids
                    if sample.sample_id in measured
                    else ()
                ),
                normalized_nll=(
                    measured[sample.sample_id].normalized_nll
                    if sample.sample_id in measured
                    else None
                ),
                verifier=verifiers.get(sample.sample_id),
                verifier_required=verifier_required,
            ),
            # Quality comparison is paired by brief, independent of route.
            sample_id=sample.job.brief.brief_id,
        )
        for sample in samples
    )


def _publish_stopped_result(
    builder: Phase1ArtifactBuilder,
    paths: Phase1Paths,
    cache: ImmutableRawCache,
    *,
    verifier_route: RouteLock,
    all_generation_jobs: Sequence[GenerationJob],
    generated_samples: Sequence[GeneratedSample],
    verifier_results: Sequence[VerifiedStory],
    submitted_jobs: Sequence[GenerationJob | VerifierJob],
    preflight: CostPreflight,
    status: str,
    screen_reports: tuple[RouteQualityReport, ...],
    finalist_selection: QualitySelection | None,
    full_reports: tuple[RouteQualityReport, ...],
    qualified: tuple[str, ...],
    progress: PhaseProgress,
) -> Phase1RunResult:
    if not (builder.root / "quality_comparisons.json").exists():
        builder.write_json(
            "quality_comparisons.json",
            {
                "audited_route_ids": (
                    []
                    if finalist_selection is None
                    else list(finalist_selection.route_ids)
                ),
                "qualified_route_ids": list(qualified),
            },
        )
    if not (builder.root / "quality_details.json").exists():
        builder.write_json(
            "quality_details.json",
            {
                "full_reports": [_quality_report_record(item) for item in full_reports],
                "screen_reports": [_quality_report_record(item) for item in screen_reports],
                "selection": (
                    None
                    if finalist_selection is None
                    else _quality_selection_record(finalist_selection)
                ),
            },
        )
    if not (builder.root / "cost_actuals.json").exists():
        builder.write_json(
            "cost_actuals.json",
            {
                "actual_billed_usd": 0.0,
                "generation_billed_usd": 0.0,
                "projection_envelopes": _cost_projection_envelopes((), (), ()),
                "routes": [],
                "verification_billed_usd": 0.0,
            },
        )
    _write_execution_manifests(
        builder,
        cache,
        all_generation_jobs,
        generated_samples,
        verifier_results,
        verifier_route,
        attempted_jobs=submitted_jobs,
    )
    _copy_raw_cache(builder, cache, submitted_jobs)
    builder.write_json(
        "status.json",
        {
            "audit_sha256": None,
            "phase": 1,
            "status": status,
        },
    )
    progress.run(PHASES[6], lambda: None)
    directory = _finalize_and_promote(builder, paths.destination)
    return Phase1RunResult(
        directory=directory,
        status=status,
        conservative_cost_usd=preflight.conservative_usd,
        qualified_route_ids=qualified,
        audit_sha256=None,
    )


def _write_reference_artifacts(
    builder: Phase1ArtifactBuilder,
    references: Phase1ReferenceCorpus,
) -> None:
    builder.write_json("source_manifest.json", references.source_manifest)
    builder.write_json("reference_statistics.json", references.reference_statistics)
    for path, records in references.source_artifacts:
        builder.write_bytes(path, canonical_jsonl_bytes(records))


def _write_measurement_batch(
    builder: Phase1ArtifactBuilder,
    name: str,
    batch: MeasurementBatch,
) -> None:
    """Persist accelerator-derived values so later replay never needs the GPU."""
    builder.write_bytes(
        f"measurements/{name}.jsonl",
        canonical_jsonl_bytes(item.as_record() for item in batch.measurements),
    )
    builder.write_json(f"measurements/{name}_runtime.json", batch.runtime)


def _write_catalog_artifacts(
    builder: Phase1ArtifactBuilder,
    catalog: CatalogEvidence,
) -> None:
    builder.write_bytes("catalog/models.response", catalog.payloads.models)
    endpoint_payloads = dict(catalog.payloads.endpoints)
    for spec in (*CANDIDATE_MODELS, VERIFIER_MODEL):
        builder.write_bytes(
            f"catalog/endpoints/{spec.route_id}.response",
            endpoint_payloads[spec.request_model_id],
        )
    builder.write_json(
        "catalog/routes.json",
        {
            "generator_routes": [
                route.as_record() for route in catalog.resolved.generator_routes
            ],
            "snapshot_sha256": catalog.resolved.snapshot_sha256,
            "verifier_route": catalog.resolved.verifier_route.as_record(),
        },
    )


def _configuration_record(
    references: Phase1ReferenceCorpus,
    routes: ResolvedRouteCatalog,
) -> JsonObject:
    """Expose every fixed Phase 1 identity and policy in one root record."""
    model_record = lambda model: {
        "canonical_slug": model.canonical_slug,
        "first_party_provider_slug": model.first_party_provider_slug,
        "max_token_parameter": model.max_token_parameter,
        "plan_completion_usd_per_million": model.plan_completion_usd_per_million,
        "plan_prompt_usd_per_million": model.plan_prompt_usd_per_million,
        "request_model_id": model.request_model_id,
        "route_id": model.route_id,
    }
    return {
        "audit_count": PHASE1_AUDIT_COUNT,
        "audit_seed": PHASE1_AUDIT_SEED,
        "candidate_models": [model_record(model) for model in CANDIDATE_MODELS],
        "catalog_snapshot_sha256": routes.snapshot_sha256,
        "funnel_policy": "cheapest-closest-pareto-table-order-v1",
        "full_count": PHASE1_FULL_COUNT,
        "generation_workers": PHASE1_GENERATION_WORKERS,
        "hard_cap_usd": PHASE1_HARD_CAP_USD,
        "human_audit_policy": "balanced-100-pairs-mandatory-explicit-approval-v1",
        "neutral_story_prompt_version": "tinyworlds-v2-neutral-story-v1",
        "neutral_story_schema_version": "tinyworlds_v2_neutral_story_v1",
        "projected_accepted_story_count": PHASE1_PROJECTED_CORPUS_ACCEPTED_STORIES,
        "quality_policy": "reference-calibrated-phase1-gates-v2",
        "reference_statistics_sha256": json_sha256(references.reference_statistics),
        "retry_allowance_basis_points": PHASE1_RETRY_ALLOWANCE_BASIS_POINTS,
        "retry_max_attempts": 2,
        "screen_count": PHASE1_SCREEN_COUNT,
        "selection_seed": PHASE1_SELECTION_SEED,
        "surface_worker_count": REFERENCE_SURFACE_WORKERS,
        "surface_measurement_version": PHASE1_SURFACE_MEASUREMENT_VERSION,
        "validator_version": PHASE1_STORY_VALIDATOR_VERSION,
        "verifier_model": model_record(VERIFIER_MODEL),
        "verifier_prompt_version": "tinyworlds-v2-style-verifier-v1",
        "verifier_schema_version": "tinyworlds_v2_style_verifier_v1",
        "version": PHASE1_VERSION,
    }


def _write_planned_route_artifacts(
    builder: Phase1ArtifactBuilder,
    jobs: Sequence[GenerationJob],
    routes: tuple[RouteLock, ...],
) -> None:
    """Persist all exact request bodies before any route can be submitted."""
    for route in routes:
        route_jobs = tuple(
            job for job in jobs if job.route.route_id == route.route_id
        )
        builder.write_bytes(
            f"routes/{route.route_id}/requests.jsonl",
            canonical_jsonl_bytes(
                {
                    **job.request.as_record(),
                    "body": job.request.body,
                    "brief_id": job.brief.brief_id,
                    "route_id": route.route_id,
                }
                for job in route_jobs
            ),
        )
        builder.write_json(
            f"routes/{route.route_id}/batch_submission.json",
            {
                "expansion_request_sha256": [
                    job.request.request_sha256
                    for job in route_jobs[PHASE1_SCREEN_COUNT:]
                ],
                "mode": "individual-content-addressed-requests",
                "screen_request_sha256": [
                    job.request.request_sha256
                    for job in route_jobs[:PHASE1_SCREEN_COUNT]
                ],
            },
        )
        builder.write_json(
            f"routes/{route.route_id}/plan.json",
            {
                "neutral_story_prompt_version": "tinyworlds-v2-neutral-story-v1",
                "neutral_story_schema_version": "tinyworlds_v2_neutral_story_v1",
                "planned_request_count": len(route_jobs),
                "planned_request_sha256": [
                    job.request.request_sha256 for job in route_jobs
                ],
                "route": route.as_record(),
                "route_lock_sha256": route.lock_sha256,
                "surface_measurement_version": PHASE1_SURFACE_MEASUREMENT_VERSION,
                "validator_version": PHASE1_STORY_VALIDATOR_VERSION,
            },
        )
        # Empty canonical streams make absence of accepted/rejected outcomes
        # explicit even for a catastrophic route failure.
        for filename in ("accepted.jsonl", "rejected.jsonl", "raw_responses.jsonl"):
            builder.write_bytes(f"routes/{route.route_id}/{filename}", b"")


def _copy_raw_cache(
    builder: Phase1ArtifactBuilder,
    cache: ImmutableRawCache,
    submitted_jobs: Sequence[GenerationJob | VerifierJob],
) -> None:
    requests = {
        job.request.request_sha256: job.request
        for job in submitted_jobs
    }
    journal_entries = cache.load_cost_journal()
    for request_sha256 in sorted(
        {entry.request_sha256 for entry in journal_entries}
    ):
        requests.setdefault(request_sha256, cache.load_request(request_sha256))
    for request_sha256, request in sorted(requests.items()):
        cache.load_attempts(request)
        source = cache.root / request_sha256
        for path in sorted(item for item in source.rglob("*") if item.is_file()):
            relative = path.relative_to(source).as_posix()
            builder.write_bytes(
                f"raw_cache/requests/{request_sha256}/{relative}",
                path.read_bytes(),
            )
    # The write-ahead journal is independent of individual request entries:
    # it records every pre-POST reservation, including one that triggered a
    # clean fail-closed stop. Authenticate its schema before copying its exact
    # bytes into the published evidence tree.
    journal = cache.root / "runtime-cost-journal"
    if journal.is_dir():
        for path in sorted(item for item in journal.rglob("*") if item.is_file()):
            relative = path.relative_to(journal).as_posix()
            builder.write_bytes(
                f"raw_cache/runtime-cost-journal/{relative}",
                path.read_bytes(),
            )


def _write_execution_manifests(
    builder: Phase1ArtifactBuilder,
    cache: ImmutableRawCache,
    planned_jobs: Sequence[GenerationJob],
    generated_samples: Sequence[GeneratedSample],
    verifier_results: Sequence[VerifiedStory],
    verifier_route: RouteLock,
    *,
    attempted_jobs: Sequence[GenerationJob | VerifierJob] | None = None,
) -> None:
    """Bind every route outcome to its exact request and raw HTTP observations."""
    samples_by_id: dict[str, GeneratedSample] = {}
    for sample in generated_samples:
        previous = samples_by_id.get(sample.sample_id)
        # Repeated screen values inside ``full_samples`` are intentional, but
        # they must still be the exact same derived interpretation.
        if previous is not None and previous.as_record() != sample.as_record():
            raise ValueError("duplicate generated sample records differ")
        samples_by_id[sample.sample_id] = sample
    persisted_sample_records = _persisted_result_records(
        builder.root / "generator_bakeoff.jsonl",
        identity_field="sample_id",
    )
    for sample_id, sample in samples_by_id.items():
        if persisted_sample_records.get(sample_id) != sample.as_record():
            raise ValueError("in-memory generated sample differs from durable result")
    attempted = tuple(attempted_jobs or ())
    attempted_request_ids = frozenset(job.request.request_sha256 for job in attempted)
    for model in CANDIDATE_MODELS:
        route_jobs = tuple(
            job for job in planned_jobs if job.route.route_id == model.route_id
        )
        route_attempted_jobs = tuple(
            job
            for job in route_jobs
            if job.request.request_sha256 in attempted_request_ids
        )
        requests = tuple(
            _execution_request_record(
                job.request,
                cache,
                persisted_sample_records.get(job.sample_id),
                {
                    "brief_id": job.brief.brief_id,
                    "outcome": (
                        "interrupted"
                        if job.sample_id not in persisted_sample_records
                        else "accepted"
                        if _persisted_result_accepted(persisted_sample_records[job.sample_id])
                        else "rejected"
                    ),
                    "sample_id": job.sample_id,
                },
            )
            for job in route_attempted_jobs
        )
        route_records = tuple(
            persisted_sample_records[job.sample_id]
            for job in route_attempted_jobs
            if job.sample_id in persisted_sample_records
        )
        route = route_jobs[0].route
        builder.write_json(
            f"routes/{model.route_id}/manifest.json",
            {
                "accepted_count": sum(_persisted_result_accepted(item) for item in route_records),
                "actual_billed_usd": float(
                    _unique_manifest_request_cost(requests)
                ),
                "neutral_story_prompt_version": "tinyworlds-v2-neutral-story-v1",
                "neutral_story_schema_version": "tinyworlds_v2_neutral_story_v1",
                "interrupted_count": len(route_attempted_jobs) - len(route_records),
                "planned_request_count": len(route_jobs),
                "planned_request_sha256": [job.request.request_sha256 for job in route_jobs],
                "rejected_count": sum(not _persisted_result_accepted(item) for item in route_records),
                "requests": list(requests),
                "route": route.as_record(),
                "route_lock_sha256": route.lock_sha256,
                "submitted_request_count": len(route_attempted_jobs),
                "surface_measurement_version": PHASE1_SURFACE_MEASUREMENT_VERSION,
                "validator_version": PHASE1_STORY_VALIDATOR_VERSION,
            },
        )

    verifier_by_source = {item.job.source_id: item for item in verifier_results}
    if len(verifier_by_source) != len(verifier_results):
        raise ValueError("verifier result source IDs must be unique")
    persisted_verifier_records = _persisted_result_records(
        builder.root / "verifier_results.jsonl",
        identity_field="source_id",
    )
    for source_id, result in verifier_by_source.items():
        if persisted_verifier_records.get(source_id) != result.as_record():
            raise ValueError("in-memory verifier result differs from durable result")
    attempted_verifier_jobs = tuple(
        job for job in attempted if isinstance(job, VerifierJob)
    )
    verifier_requests = tuple(
        _execution_request_record(
            job.request,
            cache,
            persisted_verifier_records.get(job.source_id),
            {
                "outcome": (
                    "interrupted"
                    if job.source_id not in persisted_verifier_records
                    else "accepted"
                    if persisted_verifier_records[job.source_id].get("payload") is not None
                    else "rejected"
                ),
                "pair_id": job.pair_id,
                "source_id": job.source_id,
            },
        )
        for job in attempted_verifier_jobs
    )
    builder.write_json(
        "verifier/manifest.json",
        {
            "accepted_count": sum(
                item.get("payload") is not None
                for item in persisted_verifier_records.values()
            ),
            "actual_billed_usd": float(
                _unique_manifest_request_cost(verifier_requests)
            ),
            "interrupted_count": len(attempted_verifier_jobs) - len(persisted_verifier_records),
            "rejected_count": sum(
                item.get("payload") is None
                for item in persisted_verifier_records.values()
            ),
            "requests": list(verifier_requests),
            "route": verifier_route.as_record(),
            "route_lock_sha256": verifier_route.lock_sha256,
            "submitted_request_count": len(attempted_verifier_jobs),
            "surface_measurement_version": PHASE1_SURFACE_MEASUREMENT_VERSION,
            "validator_version": PHASE1_STORY_VALIDATOR_VERSION,
            "verifier_prompt_version": "tinyworlds-v2-style-verifier-v1",
            "verifier_schema_version": "tinyworlds_v2_style_verifier_v1",
        },
    )


def _execution_request_record(
    request: object,
    cache: ImmutableRawCache,
    result_record: JsonObject | None,
    identity: JsonObject,
) -> JsonObject:
    from apm.data.text.tinyworlds_v2.generation_schema import CanonicalRequest

    if type(request) is not CanonicalRequest:
        raise TypeError("execution manifest request must be canonical")
    attempts = cache.load_attempts(request)
    billed = _provider_billed_request_cost(cache, request)
    observations = [
        {
            "attempt_number": attempt.attempt_number,
            "generation_stats_attempts": (
                []
                if attempt.response is None
                else [
                    {
                        "attempt_number": stats.attempt_number,
                        "billed_cost_usd": stats.billed_cost_usd,
                        "response_body_sha256": (
                            None
                            if stats.response is None
                            else sha256(stats.response.body).hexdigest()
                        ),
                        "status_code": (
                            None if stats.response is None else stats.response.status_code
                        ),
                        "transport_error_type": stats.transport_error_type,
                    }
                    for stats in attempt.response.generation_stats_attempts
                ]
            ),
            "provider_reported_billed_cost_usd": (
                None
                if (cost := _provider_billed_attempt_cost(attempt)) is None
                else format(cost, "f")
            ),
            "response_body_sha256": (
                None
                if attempt.response is None
                else sha256(attempt.response.body).hexdigest()
            ),
            "submission_catalog_sha256": attempt.submission_catalog_sha256,
            "status_code": None if attempt.response is None else attempt.response.status_code,
            "transport_error_type": attempt.transport_error_type,
        }
        for attempt in attempts
    ]
    return {
        **identity,
        "attempts": observations,
        "billed_cost_usd": format(billed, "f"),
        "request_sha256": request.request_sha256,
        "result_sha256": None if result_record is None else json_sha256(result_record),
    }


def _unique_manifest_request_cost(requests: Sequence[JsonObject]) -> Decimal:
    costs: dict[str, Decimal] = {}
    for item in requests:
        request_sha256 = item["request_sha256"]
        billed_cost = item["billed_cost_usd"]
        if type(request_sha256) is not str or type(billed_cost) is not str:
            raise TypeError("execution manifest request cost fields must be strings")
        cost = Decimal(billed_cost)
        previous = costs.get(request_sha256)
        if previous is not None and previous != cost:
            raise ValueError("duplicate execution request costs differ")
        costs[request_sha256] = cost
    return sum(costs.values(), Decimal(0))


def _provider_billed_attempt_cost(attempt: RawAttempt) -> Decimal | None:
    """Resolve actual billed cost from completion or generation-stats evidence."""
    if attempt.response is None:
        return None
    if attempt.response.billed_cost_usd is not None:
        return Decimal(attempt.response.billed_cost_usd)
    return next(
        (
            Decimal(item.billed_cost_usd)
            for item in reversed(attempt.response.generation_stats_attempts)
            if item.billed_cost_usd is not None
        ),
        None,
    )


def _provider_billed_request_cost(
    cache: ImmutableRawCache,
    request: CanonicalRequest,
    *,
    journal_by_identity: Mapping[tuple[str, int], CostJournalEntry] | None = None,
) -> Decimal:
    """Reconcile actual billing from raw responses and journal-only settlements."""
    raw_actuals = {
        attempt.attempt_number: cost
        for attempt in cache.load_attempts(request)
        if (cost := _provider_billed_attempt_cost(attempt)) is not None
    }
    journal = (
        {
            (entry.request_sha256, entry.attempt_number): entry
            for entry in cache.load_cost_journal()
        }
        if journal_by_identity is None
        else journal_by_identity
    )
    journal_actuals: dict[int, Decimal] = {}
    for (request_sha256, attempt_number), entry in journal.items():
        if request_sha256 != request.request_sha256:
            continue
        if entry.provider_reported_actual is True:
            charged_usd = entry.charged_usd
            if type(charged_usd) is not str:
                raise ValueError("provider-actual settlement lacks its charged cost")
            journal_actuals[attempt_number] = Decimal(charged_usd)
    for attempt_number in raw_actuals.keys() & journal_actuals.keys():
        if raw_actuals[attempt_number] != journal_actuals[attempt_number]:
            raise ValueError(
                "raw provider cost differs from its durable cost settlement"
            )
    return sum(
        (
            raw_actuals[attempt_number]
            if attempt_number in raw_actuals
            else journal_actuals[attempt_number]
            for attempt_number in raw_actuals.keys() | journal_actuals.keys()
        ),
        Decimal(0),
    )


def _persisted_result_records(
    path: Path,
    *,
    identity_field: str,
) -> dict[str, JsonObject]:
    """Load one durable append-only result stream after completed batches only."""
    if not path.exists():
        return {}
    payload = path.read_bytes()
    if payload and not payload.endswith(b"\n"):
        raise ValueError(f"persisted result stream has invalid framing: {path.name}")
    records: dict[str, JsonObject] = {}
    for index, line in enumerate(payload.splitlines(), start=1):
        record = require_json_object(
            canonical_json_loads(line, label=f"{path.name} line {index}"),
            label=f"{path.name} line {index}",
        )
        identity = record.get(identity_field)
        if type(identity) is not str or not identity:
            raise ValueError(f"persisted result lacks {identity_field}")
        if identity in records:
            raise ValueError(f"persisted result repeats {identity_field}")
        records[identity] = record
    return records


def _persisted_result_accepted(record: JsonObject) -> bool:
    validation = record.get("validation")
    if type(validation) is not dict or type(validation.get("accepted")) is not bool:
        raise ValueError("persisted generated result lacks an accepted flag")
    return validation["accepted"]


def _cached_attempt_billing(
    cache: ImmutableRawCache,
    submitted_jobs: Sequence[GenerationJob | VerifierJob],
    *,
    route_locks: Sequence[RouteLock] = (),
    include_all_cost_evidence: bool = False,
) -> AttemptBilling:
    """Sum provider actuals once across raw attempts and durable settlements."""
    jobs_by_request: dict[str, GenerationJob | VerifierJob] = {}
    for job in submitted_jobs:
        request = job.request
        route = job.route
        previous = jobs_by_request.get(request.request_sha256)
        if previous is not None and previous.route.route_id != route.route_id:
            raise ValueError("one cached request is assigned to multiple routes")
        jobs_by_request[request.request_sha256] = job
    route_by_lock: dict[str, RouteLock] = {}
    for route in (*route_locks, *(job.route for job in submitted_jobs)):
        previous = route_by_lock.get(route.lock_sha256)
        if previous is not None and previous.route_id != route.route_id:
            raise ValueError("one semantic route lock maps to multiple route IDs")
        route_by_lock[route.lock_sha256] = route
    requests_by_id = {
        request_sha256: job.request
        for request_sha256, job in jobs_by_request.items()
    }
    journal_entries = cache.load_cost_journal()
    if include_all_cost_evidence:
        journal_request_ids = {
            entry.request_sha256
            for entry in journal_entries
            if not entry.cancelled_before_post
        }
        for request in cache.load_all_requests():
            if (
                request.request_sha256 in journal_request_ids
                or cache.load_attempts(request)
            ):
                requests_by_id.setdefault(request.request_sha256, request)
    generator_route_ids = tuple(model.route_id for model in CANDIDATE_MODELS)
    generation_costs = {route_id: Decimal(0) for route_id in generator_route_ids}
    verification_cost = Decimal(0)
    verifier_request_count = 0
    costs_by_request: list[tuple[str, Decimal]] = []
    routes_by_request: list[tuple[str, str]] = []
    journal_by_identity = {
        (entry.request_sha256, entry.attempt_number): entry
        for entry in journal_entries
    }
    if len(journal_by_identity) != len(journal_entries):
        raise ValueError("runtime cost journal identities repeat")
    for request_sha256, request in sorted(requests_by_id.items()):
        job = jobs_by_request.get(request_sha256)
        route = cache.load_route_lock(request_sha256)
        if request.route_lock_sha256 != route.lock_sha256:
            raise ValueError("cost evidence request and historical route lock differ")
        if job is not None and job.route.lock_sha256 != route.lock_sha256:
            raise ValueError("submitted job differs from cached route evidence")
        current = route_by_lock.get(request.route_lock_sha256)
        if current is not None and current.lock_sha256 != route.lock_sha256:
            raise ValueError("current and cached route-lock evidence differ")
        route_id = route.route_id
        cost = _provider_billed_request_cost(
            cache,
            request,
            journal_by_identity=journal_by_identity,
        )
        if route_id == VERIFIER_MODEL.route_id:
            verification_cost += cost
            verifier_request_count += 1
        elif route_id in generation_costs:
            generation_costs[route_id] += cost
        else:
            raise ValueError(f"billing contains unknown route {route_id!r}")
        costs_by_request.append((request_sha256, cost))
        routes_by_request.append((request_sha256, route_id))
    return AttemptBilling(
        generation_by_route=tuple(generation_costs.items()),
        verification_billed_usd=verification_cost,
        verifier_request_count=verifier_request_count,
        cost_by_request=tuple(costs_by_request),
        route_by_request=tuple(routes_by_request),
    )


def _runtime_unknown_costs_by_route(
    cache: ImmutableRawCache,
    billing: AttemptBilling,
) -> dict[str, Decimal]:
    """Attribute fail-closed journal bounds that are not provider actuals."""
    route_by_request = dict(billing.route_by_request)
    route_ids = (
        *(model.route_id for model in CANDIDATE_MODELS),
        VERIFIER_MODEL.route_id,
    )
    totals = {route_id: Decimal(0) for route_id in route_ids}
    raw_actual_identities = {
        (request.request_sha256, attempt.attempt_number)
        for request in cache.load_all_requests()
        for attempt in cache.load_attempts(request)
        if _provider_billed_attempt_cost(attempt) is not None
    }
    for entry in cache.load_cost_journal():
        identity = (entry.request_sha256, entry.attempt_number)
        if (
            entry.cancelled_before_post
            or entry.provider_reported_actual is True
            or identity in raw_actual_identities
        ):
            continue
        route_id = route_by_request.get(entry.request_sha256)
        if route_id is None:
            raise ValueError("unknown journal charge lacks a route attribution")
        amount = Decimal(
            entry.upper_bound_usd
            if entry.charged_usd is None
            else entry.charged_usd
        )
        totals[route_id] += amount
    return totals


def _attributed_full_route_costs(
    cache: ImmutableRawCache,
    samples: Sequence[GeneratedSample],
    generated_verifiers: Sequence[VerifiedStory],
) -> dict[str, Decimal]:
    """Attribute retry-inclusive generation and generated-story verification."""
    route_by_sample = {sample.sample_id: sample.job.route.route_id for sample in samples}
    requests_by_route: dict[str, dict[str, CanonicalRequest]] = {}
    for sample in samples:
        route_id = sample.job.route.route_id
        requests_by_route.setdefault(route_id, {})[
            sample.job.request.request_sha256
        ] = sample.job.request
    for verified in generated_verifiers:
        route_id = route_by_sample.get(verified.job.source_id)
        if route_id is None:
            raise ValueError(
                "generated verifier result does not map to a finalist sample"
            )
        requests_by_route.setdefault(route_id, {})[
            verified.job.request.request_sha256
        ] = verified.job.request
    costs: dict[str, Decimal] = {}
    for route_id, requests in requests_by_route.items():
        costs[route_id] = sum(
            (
                sum(
                    (
                        cost
                        for attempt in cache.load_attempts(request)
                        if (cost := _provider_billed_attempt_cost(attempt))
                        is not None
                    ),
                    Decimal(0),
                )
                for request in requests.values()
            ),
            Decimal(0),
        )
    return costs


def _attempted_jobs(
    cache: ImmutableRawCache,
    jobs: Sequence[GenerationJob | VerifierJob],
) -> tuple[GenerationJob | VerifierJob, ...]:
    """Return jobs that crossed the durable reservation or attempt boundary."""
    attempted: list[GenerationJob | VerifierJob] = []
    seen_jobs: set[tuple[str, str]] = set()
    journal_request_ids = {
        entry.request_sha256
        for entry in cache.load_cost_journal()
        if not entry.cancelled_before_post
    }
    for job in jobs:
        identity = (
            ("generation", job.sample_id)
            if isinstance(job, GenerationJob)
            else ("verifier", job.source_id)
        )
        if identity in seen_jobs:
            continue
        seen_jobs.add(identity)
        request_directory = cache.root / job.request.request_sha256
        if request_directory.is_dir() and (
            cache.load_attempts(job.request)
            or job.request.request_sha256 in journal_request_ids
        ):
            attempted.append(job)
    return tuple(attempted)


def _interrupted_generation_status(
    error: Exception,
    client: OpenRouterClient,
) -> str:
    """Classify a concurrent stop from durable ledger evidence, not race order."""
    if isinstance(error, CatalogRouteRevalidationError):
        return "catalog_route_drift"
    provider_reasons = {
        "billed_attempt_response_missing",
        "byok_preflight_failed",
        "orphaned_cost_reservation",
        "provider_billing_unknown",
        "provider_cost_exceeds_reserved_bound",
        "provider_cost_policy_violation",
        "provider_response_contract_failure",
        "provider_secret_reflection",
        "raw_response_persistence_failure",
    }
    # Another worker may observe the shared halt and surface CostCapExceeded
    # after the causative provider-contract future already failed.  The ledger
    # reason is durable and wins over that nondeterministic completion order.
    if client.cost_ledger.snapshot().halted_reason in provider_reasons or isinstance(
        error,
        (
            CostJournalRecoveryRequired,
            OpenRouterBillingUnknown,
            OpenRouterContractError,
        ),
    ):
        return "provider_billing_unknown"
    return "blocked_by_runtime_cost_cap"


def _write_runtime_cost_stop(
    builder: Phase1ArtifactBuilder,
    cache: ImmutableRawCache,
    attempted_jobs: Sequence[GenerationJob | VerifierJob],
    ledger: RuntimeCostLedger,
    route_locks: Sequence[RouteLock],
) -> None:
    """Persist exact stopped-run attribution without relabeling verifier cost."""
    snapshot = ledger.snapshot()
    builder.write_json("runtime_cost_ledger.json", snapshot.as_record())
    billing = _cached_attempt_billing(
        cache,
        attempted_jobs,
        route_locks=route_locks,
        include_all_cost_evidence=True,
    )
    if billing.actual_billed_usd != Decimal(
        snapshot.provider_reported_actual_usd
    ):
        raise ValueError(
            "runtime ledger actual differs from raw/journal cost attribution"
        )
    unknown_by_route = _runtime_unknown_costs_by_route(cache, billing)
    if sum(unknown_by_route.values(), Decimal(0)) != Decimal(
        snapshot.conservative_unknown_charge_usd
    ):
        raise ValueError(
            "runtime ledger unknown charge differs from journal attribution"
        )
    generation_by_route = dict(billing.generation_by_route)
    route_by_request = dict(billing.route_by_request)
    persisted_samples = _persisted_result_records(
        builder.root / "generator_bakeoff.jsonl",
        identity_field="sample_id",
    )
    accepted_request_ids = {
        job.request.request_sha256
        for job in attempted_jobs
        if isinstance(job, GenerationJob)
        and (record := persisted_samples.get(job.sample_id)) is not None
        and _persisted_result_accepted(record)
    }
    route_records = tuple(
        {
            "accepted_count": sum(
                request_sha256 in accepted_request_ids
                and assigned_route_id == route_id
                for request_sha256, assigned_route_id in billing.route_by_request
            ),
            "actual_billed_usd": float(generation_by_route[route_id]),
            # Interrupted evidence is insufficient for a scientific
            # full-corpus projection; unavailable envelopes below make this
            # numeric schema placeholder non-selectable.
            "projected_full_corpus_usd": 0.0,
            "request_count": sum(
                assigned_route_id == route_id
                for assigned_route_id in route_by_request.values()
            ),
            "route_id": route_id,
        }
        for route_id in (model.route_id for model in CANDIDATE_MODELS)
    )
    builder.write_json(
        "cost_actuals.json",
        {
            "actual_billed_usd": float(billing.actual_billed_usd),
            "generation_billed_usd": float(billing.generation_billed_usd),
            "projection_envelopes": _cost_projection_envelopes((), (), ()),
            "routes": list(route_records),
            "verification_billed_usd": float(billing.verification_billed_usd),
        },
    )
    route_observations = [
        {
            "conservative_unknown_charge_usd": format(
                unknown_by_route[route_id], "f"
            ),
            "provider_reported_actual_usd": format(
                (
                    billing.verification_billed_usd
                    if route_id == VERIFIER_MODEL.route_id
                    else generation_by_route[route_id]
                ),
                "f",
            ),
            "request_count": sum(
                assigned_route_id == route_id
                for assigned_route_id in route_by_request.values()
            ),
            "route_id": route_id,
        }
        for route_id in (
            *(model.route_id for model in CANDIDATE_MODELS),
            VERIFIER_MODEL.route_id,
        )
    ]
    builder.write_json(
        "cost_observations.json",
        {
            "attempted_request_count": len(billing.cost_by_request),
            "billed_generation_usd": format(
                billing.generation_billed_usd, "f"
            ),
            "billed_verifier_usd": format(
                billing.verification_billed_usd, "f"
            ),
            "conservative_unknown_charge_usd": (
                snapshot.conservative_unknown_charge_usd
            ),
            "provider_reported_actual_usd": (
                snapshot.provider_reported_actual_usd
            ),
            "routes": route_observations,
            "runtime_charged_total_usd": snapshot.charged_total_usd,
        },
    )


def _write_cost_observations(
    builder: Phase1ArtifactBuilder,
    screen_samples: Sequence[GeneratedSample],
    full_samples: Sequence[GeneratedSample],
    verified: Sequence[VerifiedStory],
    billing: AttemptBilling,
) -> None:
    builder.write_json(
        "cost_observations.json",
        {
            "billed_generation_usd": format(billing.generation_billed_usd, "f"),
            "billed_screen_returned_success_usd": _sum_costs(screen_samples),
            "billed_verifier_usd": format(billing.verification_billed_usd, "f"),
            "generated_observation_count": len(full_samples),
            "projected_accepted_story_count": PHASE1_PROJECTED_CORPUS_ACCEPTED_STORIES,
            "verifier_observation_count": len(verified),
        },
    )


def _write_cost_actuals(
    builder: Phase1ArtifactBuilder,
    screen_samples: Sequence[GeneratedSample],
    full_samples: Sequence[GeneratedSample],
    verifiers: Sequence[VerifiedStory],
    billing: AttemptBilling,
    *,
    quality_reports: Sequence[RouteQualityReport],
    qualified_route_ids: Sequence[str],
) -> None:
    """Write the strict reconciled billing record used by audit selection."""
    samples_by_id = {
        sample.sample_id: sample for sample in (*screen_samples, *full_samples)
    }
    samples = tuple(samples_by_id[key] for key in sorted(samples_by_id))
    cost_by_request = dict(billing.cost_by_request)
    verifier_cost = {
        verified.job.source_id: cost_by_request[verified.job.request.request_sha256]
        for verified in verifiers
    }
    generation_by_route = dict(billing.generation_by_route)
    routes = tuple(
        {
            "accepted_count": sum(item.validation.accepted for item in route_samples),
            "actual_billed_usd": float(generation_by_route[route_id]),
            "projected_full_corpus_usd": float(
                _projected_route_cost(
                    route_id,
                    samples,
                    verifier_cost,
                    generation_by_route[route_id],
                    billing.verification_billed_usd,
                    billing.verifier_request_count,
                )
            ),
            "request_count": len(route_samples),
            "route_id": route_id,
        }
        for route_id in (model.route_id for model in CANDIDATE_MODELS)
        for route_samples in (
            tuple(item for item in samples if item.job.route.route_id == route_id),
        )
    )
    builder.write_json(
        "cost_actuals.json",
        {
            "actual_billed_usd": float(billing.actual_billed_usd),
            "generation_billed_usd": float(billing.generation_billed_usd),
            "projection_envelopes": _cost_projection_envelopes(
                routes,
                quality_reports,
                qualified_route_ids,
            ),
            "routes": list(routes),
            "verification_billed_usd": float(billing.verification_billed_usd),
        },
    )


def _projected_route_cost(
    route_id: str,
    samples: Sequence[GeneratedSample],
    verifier_cost: Mapping[str, Decimal],
    route_generation_billed: Decimal,
    total_verifier_cost: Decimal,
    verifier_count: int,
) -> Decimal:
    route_samples = tuple(
        item for item in samples if item.job.route.route_id == route_id
    )
    accepted = tuple(item for item in route_samples if item.validation.accepted)
    if not route_samples:
        raise ValueError("every generator route requires observed screen samples")
    generation_per_attempt = route_generation_billed / len(route_samples)
    observed_verifier_costs = tuple(
        verifier_cost[item.sample_id]
        for item in accepted
        if item.sample_id in verifier_cost
    )
    verifier_per_accepted = (
        sum(observed_verifier_costs, Decimal(0)) / len(observed_verifier_costs)
        if observed_verifier_costs
        else total_verifier_cost / verifier_count
        if verifier_count
        else Decimal(0)
    )
    # A zero-acceptance route has no finite empirical cost per accepted story.
    # Retain it as an explicit, very poor 1% reliability projection rather than
    # emitting infinity, which canonical JSON correctly rejects.
    acceptance_rate = max(
        Decimal(len(accepted)) / Decimal(len(route_samples)), Decimal("0.01")
    )
    per_accepted = generation_per_attempt / acceptance_rate + verifier_per_accepted
    return per_accepted * PHASE1_PROJECTED_CORPUS_ACCEPTED_STORIES


def _cost_projection_envelopes(
    route_costs: Sequence[JsonObject],
    quality_reports: Sequence[RouteQualityReport],
    qualified_route_ids: Sequence[str],
) -> JsonObject:
    """Select named 4,000-story projections from fully qualified routes.

    Economy minimizes empirical dollars per accepted story. Quality ceiling
    minimizes the fixed alignment distance without regard to price. Balanced
    gives equal weight to min-max-normalized projected cost and alignment.
    """
    definitions = {
        "balanced": (
            "equal_weight_minmax_projected_cost_and_alignment_among_qualified"
        ),
        "economy": "minimum_projected_cost_among_qualified",
        "quality_ceiling": "minimum_alignment_distance_among_qualified",
    }
    qualified = tuple(qualified_route_ids)
    report_by_route = {report.route_id: report for report in quality_reports}
    cost_by_route = {
        item["route_id"]: float(item["projected_full_corpus_usd"])
        for item in route_costs
        if type(item.get("route_id")) is str
        and type(item.get("projected_full_corpus_usd")) in (int, float)
    }
    eligible = tuple(
        route_id
        for route_id in qualified
        if route_id in report_by_route
        and report_by_route[route_id].passed
        and route_id in cost_by_route
    )
    if not eligible:
        return {
            name: {
                "available": False,
                "definition": definition,
                "projected_accepted_story_count": (
                    PHASE1_PROJECTED_CORPUS_ACCEPTED_STORIES
                ),
                "projected_full_corpus_usd": None,
                "reason": "no_fully_qualified_route",
                "route_id": None,
            }
            for name, definition in sorted(definitions.items())
        }

    route_order = {route_id: index for index, route_id in enumerate(eligible)}
    economy = min(
        eligible,
        key=lambda route_id: (cost_by_route[route_id], route_order[route_id]),
    )
    quality_ceiling = min(
        eligible,
        key=lambda route_id: (
            report_by_route[route_id].alignment_distance,
            route_order[route_id],
        ),
    )
    costs = tuple(cost_by_route[route_id] for route_id in eligible)
    alignments = tuple(
        report_by_route[route_id].alignment_distance for route_id in eligible
    )

    def normalized(value: float, values: tuple[float, ...]) -> float:
        low, high = min(values), max(values)
        return 0.0 if high == low else (value - low) / (high - low)

    balanced = min(
        eligible,
        key=lambda route_id: (
            normalized(cost_by_route[route_id], costs)
            + normalized(
                report_by_route[route_id].alignment_distance,
                alignments,
            ),
            route_order[route_id],
        ),
    )
    selected = {
        "balanced": balanced,
        "economy": economy,
        "quality_ceiling": quality_ceiling,
    }
    return {
        name: {
            "available": True,
            "definition": definitions[name],
            "projected_accepted_story_count": (
                PHASE1_PROJECTED_CORPUS_ACCEPTED_STORIES
            ),
            "projected_full_corpus_usd": cost_by_route[route_id],
            "reason": None,
            "route_id": route_id,
        }
        for name, route_id in sorted(selected.items())
    }


def _style_scores(payload: VerifierPayload | None) -> tuple[tuple[str, float], ...]:
    if payload is None:
        return (
            ("grammar", 0.0),
            ("non_repetition", 0.0),
            ("plot_coherence", 0.0),
            ("preschool_vocabulary", 0.0),
            ("sentence_simplicity", 0.0),
        )
    return tuple(
        (name, float(getattr(payload, name)))
        for name in (
            "grammar",
            "non_repetition",
            "plot_coherence",
            "preschool_vocabulary",
            "sentence_simplicity",
        )
    )


def _quality_report_record(report: RouteQualityReport) -> JsonObject:
    values: JsonObject = {}
    for field in fields(report):
        value = getattr(report, field.name)
        if field.name == "phase":
            values[field.name] = value.value
        elif type(value) is tuple:
            values[field.name] = list(value)
        elif type(value) is float and not math.isfinite(value):
            values[field.name] = None
        else:
            values[field.name] = value
    values["passed"] = report.passed
    return values


def _quality_selection_record(selection: QualitySelection) -> JsonObject:
    return {
        "outcome": selection.outcome.value,
        "reason": selection.reason,
        "route_ids": list(selection.route_ids),
    }


def _reference_profile_record(profile: ReferenceProfile) -> JsonObject:
    return {
        "alphanumeric_identifier_token_rate": profile.alphanumeric_identifier_token_rate,
        "dialogue_rate": profile.dialogue_rate,
        "digit_bearing_token_rate": profile.digit_bearing_token_rate,
        "ending_frequencies": [list(item) for item in profile.ending_frequencies],
        "realized_feature_rates": [
            list(item) for item in profile.realized_feature_rates
        ],
        "requested_feature_rates": [list(item) for item in profile.feature_rates],
        "median_normalized_nll": profile.median_normalized_nll,
        "median_repeated_ngram_fraction": profile.median_repeated_ngram_fraction,
        "median_sentence_words": profile.median_sentence_words,
        "median_story_words": profile.median_story_words,
        "model_token_counts": list(profile.model_token_counts),
        "normalized_nll_iqr": profile.normalized_nll_iqr,
        "normalized_nll_values": list(profile.normalized_nll_values),
        "numeric_token_rate": profile.numeric_token_rate,
        "opening_frequencies": [list(item) for item in profile.opening_frequencies],
        "paragraph_break_rate": profile.paragraph_break_rate,
        "profile_sha256": profile.profile_sha256,
        "record_count": profile.record_count,
        "repeated_ngram_fractions": list(profile.repeated_ngram_fractions),
        "reference_split_token_jsd": profile.reference_split_token_jsd,
        "required_word_frequencies": [list(item) for item in profile.required_word_frequencies],
        "sentence_word_counts": list(profile.sentence_word_counts),
        "story_word_counts": list(profile.story_word_counts),
        "token_probabilities": [list(item) for item in profile.token_probabilities],
        "vocabulary": sorted(profile.vocabulary),
        "word_frequencies": [list(item) for item in profile.word_frequencies],
    }


def _nll_run_record(run: ReferenceNllRun) -> JsonObject:
    return {
        "batch_size": run.batch_size,
        "checkpoint_manifest_sha256": run.checkpoint_manifest_sha256,
        "checkpoint_parameter_checksum": run.checkpoint_parameter_checksum,
        "device_kind": run.device_kind,
        "jax_platform": run.jax_platform,
        "score_count": len(run.scores),
        "sequence_length": run.sequence_length,
        "tokenizer_sha256": run.tokenizer_sha256,
    }


def _sum_costs(values: Sequence[GeneratedSample] | Sequence[VerifiedStory]) -> str:
    return format(_unique_request_cost(values), "f")


def _unique_request_cost(
    values: Sequence[GeneratedSample] | Sequence[VerifiedStory],
) -> Decimal:
    costs: dict[str, Decimal] = {}
    for item in values:
        request_sha256 = item.job.request.request_sha256
        cost = Decimal(item.billed_cost_usd)
        previous = costs.get(request_sha256)
        if previous is not None and previous != cost:
            raise ValueError("one cached request has inconsistent billed costs")
        costs[request_sha256] = cost
    return sum(costs.values(), Decimal(0))


def _finalize_and_promote(
    builder: Phase1ArtifactBuilder,
    destination: Path,
) -> Path:
    builder.finalize()
    # Structural authentication alone cannot detect a self-consistent but
    # scientifically contradictory reseal. Run the complete semantic and
    # zero-network replay gate before the atomic no-replace rename. Promotion
    # itself reloads and authenticates every promoted byte, so repeating the
    # expensive derived replay after an identity-preserving rename adds no
    # evidence.
    validation = validate_phase1_semantics(builder.root)
    return builder.promote(
        destination,
        expected_manifest_sha256=validation.manifest.manifest_sha256,
    )


def _append_jsonl(path: Path, value: JsonValue) -> None:
    from apm.data.text.tinyworlds_v2.json_contracts import canonical_json_line_bytes

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        stream.write(canonical_json_line_bytes(value))


def _append_jsonl_records(path: Path, values: Sequence[JsonValue]) -> None:
    if not values:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        stream.write(canonical_jsonl_bytes(values))


def _advance_eta_bars(
    stop: Event,
    phase_bar: _TqdmBar,
    overall_bar: _TqdmBar,
    estimated_seconds: int,
) -> None:
    while not stop.wait(1.0):
        if phase_bar.n < estimated_seconds - 1:
            phase_bar.update(1)
            overall_bar.update(1)


__all__ = [
    "CatalogEvidence",
    "MeasurementBatch",
    "PHASE1_AUDIT_COUNT",
    "PHASE1_FULL_COUNT",
    "PHASE1_SCREEN_COUNT",
    "Phase1Dependencies",
    "Phase1Paths",
    "Phase1Progress",
    "Phase1ReferenceCorpus",
    "Phase1RunResult",
    "StoryMeasurement",
    "main",
    "production_dependencies",
    "run_phase1",
]

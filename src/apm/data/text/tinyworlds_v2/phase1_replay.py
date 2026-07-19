"""Zero-network replay of Phase 1 results derived from persisted raw evidence.

Replay deliberately starts at the immutable Phase 1 artifact boundary.  It
decodes the persisted briefs and catalog locks, rebuilds canonical requests,
reinterprets cached HTTP response bytes, reruns deterministic validators, and
reconstructs the route/result streams.  It never opens the TinyStories source,
loads a tokenizer/checkpoint or accelerator, reads an API key, or owns a
network-capable transport.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
import math
import os
from pathlib import Path, PurePosixPath
import shutil
from tempfile import TemporaryDirectory
from typing import NoReturn

from apm.data.text.tinyworlds_v2.audit import (
    AuditSourceKind,
    AuditSourceRecord,
    build_blinded_audit,
    render_audit_html,
)
from apm.data.text.tinyworlds_v2.audit_io import (
    encode_blinded_audit_key,
    encode_blinded_audit_packet,
    validate_phase1_tree_with_human_overlays,
)
from apm.data.text.tinyworlds_v2.bakeoff import (
    CANDIDATE_MODELS,
    VERIFIER_MODEL,
    NeutralStoryBrief,
)
from apm.data.text.tinyworlds_v2.generation_cache import ImmutableRawCache
from apm.data.text.tinyworlds_v2.generation_costs import (
    RuntimeCostLedger,
    request_cost_upper_bound,
)
from apm.data.text.tinyworlds_v2.generation_schema import RouteLock
from apm.data.text.tinyworlds_v2.json_contracts import (
    CanonicalJsonError,
    JsonObject,
    JsonValue,
    canonical_json_loads,
    require_exact_fields,
    require_json_object,
)
from apm.data.text.tinyworlds_v2.openrouter import (
    OpenRouterClient,
    OpenRouterBillingUnknown,
    OpenRouterContractError,
    OpenRouterCostPolicyError,
    RetryPolicy,
    TransportResponse,
    _completion_response,
    _derive_response,
    _validate_existing_stats_attempts,
    _validate_observation,
    _validate_success,
)
from apm.data.text.tinyworlds_v2.phase1_artifacts import (
    Phase1ArtifactBuilder,
    canonical_jsonl_bytes,
)
from apm.data.text.tinyworlds_v2.phase1_generation import (
    GeneratedSample,
    GenerationJob,
    VerifiedStory,
    build_generation_jobs,
    build_verifier_job,
    execute_generation_jobs,
    execute_verifier_jobs,
)
from apm.data.text.tinyworlds_v2.route_lock import validate_locked_request_body
from apm.data.text.tinyworlds_v2.phase1_runner import (
    PHASE1_FULL_COUNT,
    PHASE1_SCREEN_COUNT,
    MeasurementBatch,
    StoryMeasurement,
    _attributed_full_route_costs,
    _cached_attempt_billing,
    _quality_observations,
    _quality_report_record,
    _quality_selection_record,
    _style_scores,
    _write_execution_manifests,
)
from apm.data.text.tinyworlds_v2.quality import (
    BLIND_VERIFIER_DIMENSIONS,
    QualityPhase,
    QualitySelection,
    RouteQualityReport,
    evaluate_route_quality,
    select_full_quality_routes,
    select_screen_finalists,
)
from apm.data.text.tinyworlds_v2.reference_profile import (
    ReferenceObservation,
    ReferenceProfile,
    build_reference_profile,
)


class Phase1ReplayError(ValueError):
    """Persisted evidence cannot reproduce the manifested derived artifacts."""


@dataclass(frozen=True, slots=True)
class Phase1ReplayResult:
    """Paths and byte counts authenticated by one zero-network replay."""

    compared_paths: tuple[str, ...]
    compared_size_bytes: int


class _NetworkForbiddenTransport:
    """A transport whose only valid behavior is never to be called."""

    def post(self, **_arguments: object) -> NoReturn:
        raise Phase1ReplayError("replay attempted an OpenRouter POST")

    def get_authenticated(self, **_arguments: object) -> NoReturn:
        raise Phase1ReplayError("replay attempted an authenticated OpenRouter GET")


class _ReplayCostLedger(RuntimeCostLedger):
    """Disable scheduling policy while cached response bytes are interpreted."""

    def attach_cache(self, _cache: ImmutableRawCache) -> None:
        return None

    def reconcile_cached(
        self,
        _request: object,
        _route: object,
        _attempts: object,
    ) -> None:
        return None

    def halt(self, _reason: str) -> None:
        return None

    def reserve(self, *_arguments: object, **_keywords: object) -> NoReturn:
        raise Phase1ReplayError("replay attempted to reserve a new billable POST")


class _ReplayRawCache:
    """Materialize published raw evidence in the cache's native layout."""

    def __init__(self, artifact_root: Path) -> None:
        self._temporary = TemporaryDirectory(prefix="tinyworlds-v2-raw-replay-")
        self.root = Path(self._temporary.name) / "raw-cache"
        self.root.mkdir(parents=True)
        requests = artifact_root / "requests"
        if requests.is_dir():
            for request in requests.iterdir():
                shutil.copytree(
                    request,
                    self.root / request.name,
                    copy_function=_hardlink_or_copy,
                )
        journal = artifact_root / "runtime-cost-journal"
        if journal.is_dir():
            shutil.copytree(
                journal,
                self.root / journal.name,
                copy_function=_hardlink_or_copy,
            )
        for request in self.root.iterdir():
            if request.is_dir() and request.name != "runtime-cost-journal":
                (request / "attempts").mkdir(exist_ok=True)
        for attempt in self.root.glob("*/attempts/*"):
            if attempt.is_dir():
                (attempt / "generation-stats").mkdir(exist_ok=True)
        self.cache = ImmutableRawCache(self.root)
        # Validate the separately published write-ahead journal even when the
        # bundle stopped before producing a replayable result record.
        self.cache.load_cost_journal()

    def close(self) -> None:
        self._temporary.cleanup()


def _hardlink_or_copy(source: str, destination: str) -> str:
    """Materialize immutable replay evidence without recopying file contents."""
    try:
        os.link(source, destination)
    except OSError:
        # Artifact and system temporary roots can live on different filesystems.
        # Replay remains correct there; it merely loses the local hardlink speedup.
        return shutil.copy2(source, destination)
    return destination


def replay_phase1_derived(
    artifact_root: str | Path,
    replay_root: str | Path,
) -> Phase1ReplayResult:
    """Rebuild and byte-compare deterministic result streams from raw bytes."""
    source = Path(artifact_root)
    destination = Path(replay_root)
    validate_phase1_tree_with_human_overlays(source)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Phase 1 replay destination already exists: {destination}")
    destination.mkdir(parents=True)
    builder = Phase1ArtifactBuilder(destination, version="tinyworlds-v2-phase1-replay-v1")

    briefs = _load_briefs(source / "neutral_story_briefs.jsonl")
    generator_routes, verifier_route = _load_routes(source / "catalog" / "routes.json")
    all_jobs = build_generation_jobs(briefs, CANDIDATE_MODELS, generator_routes)
    _validate_planned_requests(source, all_jobs)

    persisted_generator_records = (
        _jsonl_objects(source / "generator_bakeoff.jsonl")
        if (source / "generator_bakeoff.jsonl").is_file()
        else ()
    )
    jobs_by_request = {job.request.request_sha256: job for job in all_jobs}
    try:
        submitted_jobs = tuple(
            jobs_by_request[_text(record.get("request_sha256"), "generator request_sha256")]
            for record in persisted_generator_records
        )
    except KeyError as error:
        raise Phase1ReplayError("generator result names an unplanned request") from error

    replay_cache = _ReplayRawCache(source / "raw_cache")
    cache = replay_cache.cache
    recovery_reason = _validate_raw_cache_journal(
        cache,
        (*generator_routes, verifier_route),
    )
    client = OpenRouterClient(
        api_key="offline-replay-never-sent",
        transport=_NetworkForbiddenTransport(),
        cache=cache,
        retry_policy=RetryPolicy(
            max_attempts=2,
            initial_delay_seconds=0.0,
            maximum_delay_seconds=0.0,
        ),
        cost_ledger=_ReplayCostLedger(),
        sleeper=lambda _seconds: None,
    )
    samples = execute_generation_jobs(submitted_jobs, client, max_workers=1)
    screen_reports, screen_selection, quality_reference_evidence = _recompute_screen_quality(
        source,
        cache,
        briefs,
        samples,
    )
    finalist_ids = () if screen_selection is None else screen_selection.route_ids
    _write_generation_streams(builder, samples, finalist_ids)

    verifier_results: tuple[VerifiedStory, ...] = ()
    replay_verifier_jobs = ()
    all_verifier_jobs = ()
    full_samples: tuple[GeneratedSample, ...] = ()
    if finalist_ids:
        sample_by_id = {sample.sample_id: sample for sample in samples}
        full_samples = tuple(
            sample
            for route_id in finalist_ids
            for brief in briefs
            for sample in (sample_by_id.get(f"{route_id}:{brief.brief_id}"),)
            if sample is not None
        )
        reference_jobs = tuple(
            build_verifier_job(
                source_id=f"reference:{brief.brief_id}",
                pair_id=brief.brief_id,
                brief=brief,
                story=brief.matched_reference_text,
                model=VERIFIER_MODEL,
                route=verifier_route,
            )
            for brief in briefs
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
        all_verifier_jobs = reference_jobs + generated_jobs
    if (source / "verifier_results.jsonl").is_file():
        verifier_records = _jsonl_objects(source / "verifier_results.jsonl")
        verifier_by_identity = {
            (job.source_id, job.request.request_sha256): job for job in all_verifier_jobs
        }
        try:
            replay_verifier_jobs = tuple(
                verifier_by_identity[
                    (
                        _text(record.get("source_id"), "verifier source_id"),
                        _text(record.get("request_sha256"), "verifier request_sha256"),
                    )
                ]
                for record in verifier_records
            )
        except KeyError as error:
            raise Phase1ReplayError("verifier result names an ineligible request") from error
        verifier_results = execute_verifier_jobs(
            replay_verifier_jobs,
            client,
            max_workers=1,
        )
        builder.write_bytes(
            "verifier_results.jsonl",
            canonical_jsonl_bytes(item.as_record() for item in verifier_results),
        )
        _write_verifier_events(
            builder,
            sum(item.job.source_id.startswith("reference:") for item in verifier_results),
            sum(not item.job.source_id.startswith("reference:") for item in verifier_results),
        )

    _write_replayed_quality(
        builder,
        source,
        cache,
        briefs,
        samples,
        verifier_results,
        screen_reports,
        screen_selection,
        quality_reference_evidence,
    )

    if (source / "audit_packet.json").is_file():
        _write_replayed_audit(
            builder,
            source,
            briefs,
            full_samples,
            verifier_results,
            finalist_ids,
        )

    raw_request_ids = (
        frozenset(path.name for path in (source / "raw_cache" / "requests").iterdir())
        if (source / "raw_cache" / "requests").is_dir()
        else frozenset()
    )
    attempted_request_ids = frozenset(
        request.request_sha256
        for request in cache.load_all_requests()
        if cache.load_attempts(request)
    )
    reservation_request_ids = frozenset(
        entry.request_sha256
        for entry in cache.load_cost_journal()
        if not entry.cancelled_before_post
    )
    submitted_boundary_ids = attempted_request_ids | reservation_request_ids
    attempted_generation_jobs = tuple(
        job for job in all_jobs if job.request.request_sha256 in submitted_boundary_ids
    )
    completed_verifier_sources = frozenset(
        job.source_id for job in replay_verifier_jobs
    )
    interrupted_verifier_jobs = tuple(
        job
        for job in all_verifier_jobs
        if job.request.request_sha256 in submitted_boundary_ids
        and job.source_id not in completed_verifier_sources
    )
    # Completed verifier jobs retain the durable result-stream order.  That
    # order is part of the original manifest; canonical job order is used only
    # for attempted jobs that have no completed result.
    attempted_jobs = (
        *attempted_generation_jobs,
        *replay_verifier_jobs,
        *interrupted_verifier_jobs,
    )
    eligible_request_ids = frozenset(
        job.request.request_sha256 for job in (*all_jobs, *all_verifier_jobs)
    )
    historical_request_ids = frozenset(
        request.request_sha256
        for request in cache.load_all_requests()
        if cache.load_route_lock(request.request_sha256).route_id
        in {spec.route_id for spec in (*CANDIDATE_MODELS, VERIFIER_MODEL)}
    )
    if not raw_request_ids.issubset(eligible_request_ids | historical_request_ids):
        raise Phase1ReplayError("raw cache contains a request outside the Phase 1 routes")
    replayed_failure_reasons = _reparse_all_attempted_responses(
        cache,
        (*all_jobs, *all_verifier_jobs),
    )
    _validate_recovery_status(
        source,
        recovery_reason,
        replayed_failure_reasons,
    )

    _write_execution_manifests(
        builder,
        cache,
        all_jobs,
        samples,
        verifier_results,
        verifier_route,
        attempted_jobs=attempted_jobs,
    )
    builder.finalize()

    replay_paths = tuple(
        sorted(
            path.relative_to(destination).as_posix()
            for path in destination.rglob("*")
            # ``Phase1ArtifactBuilder.finalize`` writes the replay tree's own
            # root integrity manifest.  That manifest necessarily differs
            # from the source artifact, but route and verifier execution
            # manifests are derived evidence and must be replayed verbatim.
            if path.is_file()
            and path.relative_to(destination).as_posix() != "manifest.json"
        )
    )
    for relative in replay_paths:
        original = source.joinpath(*PurePosixPath(relative).parts)
        if not original.is_file():
            raise Phase1ReplayError(f"base artifact lacks replay path: {relative}")
        if original.read_bytes() != destination.joinpath(
            *PurePosixPath(relative).parts
        ).read_bytes():
            raise Phase1ReplayError(f"derived replay byte mismatch: {relative}")
    result = Phase1ReplayResult(
        replay_paths,
        sum((source / relative).stat().st_size for relative in replay_paths),
    )
    replay_cache.close()
    return result


def verify_phase1_derived_replay(
    artifact_root: str | Path,
) -> Phase1ReplayResult:
    """Run a disposable zero-network replay and return its comparison evidence."""
    source = Path(artifact_root)
    with TemporaryDirectory(prefix="tinyworlds-v2-phase1-replay-", dir=source.parent) as name:
        return replay_phase1_derived(source, Path(name) / "derived")


def _validate_raw_cache_journal(
    cache: ImmutableRawCache,
    routes: tuple[RouteLock, ...],
) -> str | None:
    """Cross-check every durable reservation against its request and attempt."""
    route_by_lock = {route.lock_sha256: route for route in routes}
    allowed_route_ids = {route.route_id for route in routes}
    requests = cache.load_all_requests()
    attempts_by_identity = {}
    bounds_by_request: dict[str, str] = {}
    for request in requests:
        route = cache.load_route_lock(request.request_sha256)
        try:
            validate_locked_request_body(route, request)
        except (TypeError, ValueError) as error:
            raise Phase1ReplayError(
                "cached request body violates its historical route lock"
            ) from error
        current = route_by_lock.get(request.route_lock_sha256)
        if current is not None and current.lock_sha256 != route.lock_sha256:
            raise Phase1ReplayError("cached historical route lock contradicts current lock")
        if route.route_id not in allowed_route_ids:
            raise Phase1ReplayError("raw request references an unknown Phase 1 route")
        bounds_by_request[request.request_sha256] = request_cost_upper_bound(
            request,
            route,
        ).upper_bound_usd
        for attempt in cache.load_attempts(request):
            identity = (request.request_sha256, attempt.attempt_number)
            if identity in attempts_by_identity:
                raise Phase1ReplayError("raw completion attempt identity repeats")
            attempts_by_identity[identity] = attempt

    journal_by_identity = {
        (entry.request_sha256, entry.attempt_number): entry
        for entry in cache.load_cost_journal()
    }
    if len(journal_by_identity) != len(cache.load_cost_journal()):
        raise Phase1ReplayError("runtime cost journal identity repeats")
    attempts_without_journal = attempts_by_identity.keys() - journal_by_identity.keys()
    if attempts_without_journal:
        raise Phase1ReplayError(
            "a raw completion attempt lacks its runtime cost reservation"
        )
    cancelled_identities = {
        identity
        for identity, entry in journal_by_identity.items()
        if entry.cancelled_before_post
    }
    if cancelled_identities & attempts_by_identity.keys():
        raise Phase1ReplayError(
            "cancelled-before-POST reservation has a raw completion attempt"
        )
    journal_without_attempt = (
        journal_by_identity.keys()
        - attempts_by_identity.keys()
        - cancelled_identities
    )
    evidence_reasons: set[str] = set()
    for identity, entry in journal_by_identity.items():
        upper_bound = bounds_by_request.get(entry.request_sha256)
        if upper_bound is None:
            raise Phase1ReplayError(
                "runtime cost reservation lacks its canonical request"
            )
        if entry.upper_bound_usd != upper_bound:
            raise Phase1ReplayError(
                "runtime cost reservation differs from the request upper bound"
            )
        if entry.cancelled_before_post:
            continue
        if identity in journal_without_attempt:
            continue
        attempt = attempts_by_identity[identity]
        resolved_cost = None
        if attempt.response is not None:
            resolved_cost = attempt.response.billed_cost_usd
            if resolved_cost is None:
                resolved_cost = next(
                    (
                        item.billed_cost_usd
                        for item in reversed(
                            attempt.response.generation_stats_attempts
                        )
                        if item.billed_cost_usd is not None
                    ),
                    None,
                )
        expected_actual = resolved_cost is not None
        expected_charge = upper_bound if resolved_cost is None else resolved_cost
        if (
            resolved_cost is not None
            and Decimal(resolved_cost) > Decimal(upper_bound)
            and entry.charged_usd is None
            and entry.provider_reported_actual is None
        ):
            evidence_reasons.add("provider_cost_exceeds_reserved_bound")
            continue
        if (
            entry.charged_usd is None
            or entry.provider_reported_actual is not expected_actual
            or Decimal(entry.charged_usd) != Decimal(expected_charge)
        ):
            raise Phase1ReplayError(
                "runtime cost settlement differs from the raw completion attempt"
            )
    if journal_without_attempt:
        missing_entries = tuple(
            journal_by_identity[identity] for identity in journal_without_attempt
        )
        if any(entry.provider_reported_actual is True for entry in missing_entries):
            evidence_reasons.add("billed_attempt_response_missing")
        else:
            evidence_reasons.add("orphaned_cost_reservation")
    if len(evidence_reasons) > 1:
        raise Phase1ReplayError(
            "raw cost journal contains conflicting recovery conditions"
        )
    return next(iter(evidence_reasons), None)


def _reparse_all_attempted_responses(
    cache: ImmutableRawCache,
    eligible_jobs: tuple[GenerationJob | object, ...],
) -> frozenset[str]:
    """Reinterpret every raw completion/stats byte without any network fallback.

    Concurrent execution can durably persist attempts whose owning future never
    reached a committed result stream.  Those attempts are scientific and cost
    evidence too, so replay may not use ``generator_bakeoff.jsonl`` or
    ``verifier_results.jsonl`` as its attempted-request index.
    """
    job_by_request = {}
    for job in eligible_jobs:
        request = getattr(job, "request", None)
        route = getattr(job, "route", None)
        if request is None or route is None:
            raise Phase1ReplayError("eligible replay job lacks request/route data")
        previous = job_by_request.get(request.request_sha256)
        if previous is not None and previous[0] != request:
            raise Phase1ReplayError("one replay request identity has differing bodies")
        job_by_request[request.request_sha256] = (request, route)

    failure_reasons: set[str] = set()
    for cached_request in cache.load_all_requests():
        planned = job_by_request.get(cached_request.request_sha256)
        if planned is None:
            request = cached_request
            route = cache.load_route_lock(cached_request.request_sha256)
            if route.route_id not in {
                spec.route_id for spec in (*CANDIDATE_MODELS, VERIFIER_MODEL)
            }:
                raise Phase1ReplayError(
                    "historical raw attempt route is outside the Phase 1 plan"
                )
        else:
            request, route = planned
        if cached_request != request:
            raise Phase1ReplayError("raw attempted request differs from its canonical plan")
        try:
            validate_locked_request_body(route, cached_request)
        except (TypeError, ValueError) as error:
            raise Phase1ReplayError(
                "cached request body violates its historical route lock"
            ) from error
        for attempt in cache.load_attempts(cached_request):
            if attempt.transport_error_type == "authentication_secret_reflection":
                failure_reasons.add("provider_secret_reflection")
                continue
            if attempt.response is None:
                failure_reasons.add("provider_billing_unknown")
                continue
            response = attempt.response
            if any(
                item.transport_error_type == "authentication_secret_reflection"
                for item in response.generation_stats_attempts
            ):
                failure_reasons.add("provider_secret_reflection")
                # Every safe stats response preceding the marker still has to
                # be parsed, rather than letting the terminal marker mask it.
                safe_stats = tuple(
                    item
                    for item in response.generation_stats_attempts
                    if item.transport_error_type
                    != "authentication_secret_reflection"
                )
            else:
                safe_stats = response.generation_stats_attempts
            try:
                _validate_existing_stats_attempts(safe_stats)
                reparsed = _completion_response(
                    TransportResponse(
                        response.status_code,
                        response.headers,
                        response.body,
                    ),
                    route,
                )
                reparsed = _derive_response(
                    replace(
                        reparsed,
                        generation_stats_attempts=safe_stats,
                    ),
                    route,
                )
                expected = replace(
                    response,
                    generation_stats_attempts=safe_stats,
                )
                if reparsed != expected:
                    raise Phase1ReplayError(
                        "cached parsed response differs from exact raw observations"
                    )
                _validate_observation(reparsed, route)
                if 200 <= reparsed.status_code < 300:
                    _validate_success(reparsed, route)
                if reparsed.billed_cost_usd is None:
                    failure_reasons.add("provider_billing_unknown")
            except OpenRouterCostPolicyError:
                failure_reasons.add("provider_cost_policy_violation")
            except OpenRouterBillingUnknown:
                failure_reasons.add("provider_billing_unknown")
            except OpenRouterContractError:
                failure_reasons.add("provider_response_contract_failure")
    return frozenset(failure_reasons)


def _validate_recovery_status(
    source: Path,
    recovery_reason: str | None,
    replayed_failure_reasons: frozenset[str],
) -> None:
    status = _json_object(source / "status.json")
    status_value = status.get("status")
    runtime_path = source / "runtime_cost_ledger.json"
    runtime_reason = (
        _json_object(runtime_path).get("halted_reason")
        if runtime_path.is_file()
        else None
    )
    recovery_reasons = {
        "billed_attempt_response_missing",
        "orphaned_cost_reservation",
        "provider_cost_exceeds_reserved_bound",
    }
    provider_stop_reasons = {
        *recovery_reasons,
        "byok_preflight_failed",
        "provider_cost_exceeds_reserved_bound",
        "provider_cost_policy_violation",
        "provider_billing_unknown",
        "provider_response_contract_failure",
        "provider_secret_reflection",
        "raw_response_persistence_failure",
    }
    if status_value == "provider_billing_unknown":
        if not runtime_path.is_file() or runtime_reason not in provider_stop_reasons:
            raise Phase1ReplayError(
                "provider billing stop lacks its exact runtime halt reason"
            )
    elif status_value == "catalog_route_drift":
        if not runtime_path.is_file() or runtime_reason != "catalog_route_drift":
            raise Phase1ReplayError(
                "catalog route stop lacks its exact runtime halt reason"
            )
    elif runtime_reason in {*provider_stop_reasons, "catalog_route_drift"}:
        raise Phase1ReplayError(
            "provider billing halt reason has a contradictory top-level status"
        )
    if status_value == "blocked_by_runtime_cost_cap" and runtime_reason not in {
        "cached_charges_exceed_runtime_cap",
        "runtime_cap_reservation_denied",
    }:
        raise Phase1ReplayError(
            "runtime cap stop lacks a recognized cap halt reason"
        )
    compatible_recovery_reasons = {
        "billed_attempt_response_missing": {
            "billed_attempt_response_missing",
            "raw_response_persistence_failure",
        },
        "orphaned_cost_reservation": {"orphaned_cost_reservation"},
        "provider_cost_exceeds_reserved_bound": {
            "provider_cost_exceeds_reserved_bound"
        },
    }
    if recovery_reason is not None and (
        status_value != "provider_billing_unknown"
        or runtime_reason not in compatible_recovery_reasons[recovery_reason]
    ):
        raise Phase1ReplayError(
            "raw cost-journal recovery state contradicts the recorded stop"
        )
    if recovery_reason is None and runtime_reason in recovery_reasons:
        raise Phase1ReplayError(
            "recorded cost-journal recovery stop has no missing raw attempt"
        )
    raw_replayable_reasons = {
        "provider_billing_unknown",
        "provider_cost_policy_violation",
        "provider_response_contract_failure",
        "provider_secret_reflection",
    }
    if replayed_failure_reasons & raw_replayable_reasons and (
        status_value != "provider_billing_unknown"
    ):
        raise Phase1ReplayError(
            "raw provider failure has a contradictory top-level status"
        )
    if runtime_reason in raw_replayable_reasons and (
        runtime_reason not in replayed_failure_reasons
    ):
        raise Phase1ReplayError(
            "recorded provider halt reason is not reproduced by raw attempts"
        )


@dataclass(frozen=True, slots=True)
class _QualityReferenceEvidence:
    reference_profile: ReferenceProfile
    paired_observations: tuple[ReferenceObservation, ...]
    paired_profile: ReferenceProfile
    expected_feature_rates: tuple[tuple[str, float], ...]


def _recompute_screen_quality(
    source: Path,
    cache: ImmutableRawCache,
    briefs: tuple[NeutralStoryBrief, ...],
    samples: tuple[GeneratedSample, ...],
) -> tuple[
    tuple[RouteQualityReport, ...],
    QualitySelection | None,
    _QualityReferenceEvidence | None,
]:
    """Rebuild the fixed 7x50 screen without trusting persisted reports."""
    sample_by_id = {sample.sample_id: sample for sample in samples}
    if len(sample_by_id) != len(samples):
        raise Phase1ReplayError("replayed generator sample IDs repeat")
    expected_ids = tuple(
        f"{model.route_id}:{brief.brief_id}"
        for model in CANDIDATE_MODELS
        for brief in briefs[:PHASE1_SCREEN_COUNT]
    )
    if not set(expected_ids).issubset(sample_by_id):
        return (), None, None
    screen_samples = tuple(sample_by_id[sample_id] for sample_id in expected_ids)
    evidence = _load_quality_reference_evidence(source)
    paired_by_id = {
        observation.record_id: observation
        for observation in evidence.paired_observations
    }
    try:
        screen_reference_profile = build_reference_profile(
            tuple(paired_by_id[brief.source_record_id] for brief in briefs[:PHASE1_SCREEN_COUNT])
        )
    except (KeyError, TypeError, ValueError) as error:
        raise Phase1ReplayError(
            "screen quality lacks its exact paired reference observations"
        ) from error
    measurements = _load_measurement_batch(
        source / "measurements" / "screen.jsonl"
    )
    try:
        reports = tuple(
            evaluate_route_quality(
                _quality_observations(
                    tuple(
                        sample
                        for sample in screen_samples
                        if sample.job.route.route_id == model.route_id
                    ),
                    measurements,
                ),
                evidence.reference_profile,
                phase=QualityPhase.SCREEN,
                matched_reference_profile=screen_reference_profile,
                expected_feature_rates=evidence.expected_feature_rates,
            )
            for model in CANDIDATE_MODELS
        )
        billed = dict(
            _cached_attempt_billing(
                cache,
                tuple(sample.job for sample in screen_samples),
            ).generation_by_route
        )
        reports = tuple(
            replace(report, billed_cost_usd=float(billed[report.route_id]))
            for report in reports
        )
        return reports, select_screen_finalists(reports), evidence
    except (KeyError, TypeError, ValueError) as error:
        raise Phase1ReplayError(f"could not recompute screen quality: {error}") from error


def _write_replayed_quality(
    builder: Phase1ArtifactBuilder,
    source: Path,
    cache: ImmutableRawCache,
    briefs: tuple[NeutralStoryBrief, ...],
    samples: tuple[GeneratedSample, ...],
    verifier_results: tuple[VerifiedStory, ...],
    screen_reports: tuple[RouteQualityReport, ...],
    screen_selection: QualitySelection | None,
    evidence: _QualityReferenceEvidence | None,
) -> None:
    """Recompute and persist every quality decision derived from raw evidence."""
    full_reports: tuple[RouteQualityReport, ...] = ()
    full_selection: QualitySelection | None = None
    if screen_selection is not None and screen_selection.route_ids:
        if evidence is None:
            raise Phase1ReplayError("screen selection lacks reference evidence")
        full_reports = _recompute_full_quality(
            source,
            cache,
            briefs,
            samples,
            verifier_results,
            screen_selection.route_ids,
            evidence,
        )
        if full_reports:
            full_selection = select_full_quality_routes(
                full_reports,
                finalist_order=screen_selection.route_ids,
            )

    if screen_selection is not None:
        builder.write_json(
            "finalist_decision.json",
            _quality_selection_record(screen_selection),
        )
    selection = full_selection if full_selection is not None else screen_selection
    builder.write_json(
        "quality_comparisons.json",
        {
            "audited_route_ids": (
                [] if screen_selection is None else list(screen_selection.route_ids)
            ),
            "qualified_route_ids": (
                [] if full_selection is None else list(full_selection.route_ids)
            ),
        },
    )
    builder.write_json(
        "quality_details.json",
        {
            "full_reports": [_quality_report_record(item) for item in full_reports],
            "screen_reports": [_quality_report_record(item) for item in screen_reports],
            "selection": (
                None if selection is None else _quality_selection_record(selection)
            ),
        },
    )


def _recompute_full_quality(
    source: Path,
    cache: ImmutableRawCache,
    briefs: tuple[NeutralStoryBrief, ...],
    samples: tuple[GeneratedSample, ...],
    verifier_results: tuple[VerifiedStory, ...],
    finalist_ids: tuple[str, ...],
    evidence: _QualityReferenceEvidence,
) -> tuple[RouteQualityReport, ...]:
    """Return full reports only when every 200-brief/verifier input is present."""
    sample_by_id = {sample.sample_id: sample for sample in samples}
    expected_sample_ids = tuple(
        f"{route_id}:{brief.brief_id}"
        for route_id in finalist_ids
        for brief in briefs
    )
    if not set(expected_sample_ids).issubset(sample_by_id):
        return ()
    full_samples = tuple(sample_by_id[sample_id] for sample_id in expected_sample_ids)

    verifier_by_source = {item.job.source_id: item for item in verifier_results}
    if len(verifier_by_source) != len(verifier_results):
        raise Phase1ReplayError("replayed verifier source IDs repeat")
    expected_reference_sources = frozenset(
        f"reference:{brief.brief_id}" for brief in briefs
    )
    expected_generated_sources = frozenset(
        sample.sample_id
        for sample in full_samples
        if sample.validation.accepted and sample.payload is not None
    )
    expected_verifier_sources = expected_reference_sources | expected_generated_sources
    if not expected_verifier_sources.issubset(verifier_by_source):
        return ()
    if set(verifier_by_source) != expected_verifier_sources:
        raise Phase1ReplayError("full quality verifier scope differs from the finalists")

    screen_measurements = _load_measurement_batch(
        source / "measurements" / "screen.jsonl"
    )
    expansion_measurements = _load_measurement_batch(
        source / "measurements" / "finalist_expansion.jsonl"
    )
    if screen_measurements.by_id.keys() & expansion_measurements.by_id.keys():
        raise Phase1ReplayError("screen and expansion measurement IDs overlap")
    measurements = MeasurementBatch(
        tuple(
            sorted(
                screen_measurements.measurements + expansion_measurements.measurements,
                key=lambda item: item.record_id,
            )
        ),
        {},
    )
    reference_verified = tuple(
        verifier_by_source[f"reference:{brief.brief_id}"] for brief in briefs
    )
    generated_verified = tuple(
        verifier_by_source[sample_id] for sample_id in sorted(expected_generated_sources)
    )
    reference_verifier_failed = any(item.payload is None for item in reference_verified)
    reference_means = tuple(
        (
            dimension,
            sum(
                5.0 if item.payload is None else float(getattr(item.payload, dimension))
                for item in reference_verified
            )
            / len(reference_verified),
        )
        for dimension in BLIND_VERIFIER_DIMENSIONS
    )
    generated_by_source = {item.job.source_id: item for item in generated_verified}
    try:
        report_values: list[RouteQualityReport] = []
        for route_id in finalist_ids:
            report = evaluate_route_quality(
                _quality_observations(
                    tuple(
                        sample
                        for sample in full_samples
                        if sample.job.route.route_id == route_id
                    ),
                    measurements,
                    verifier_by_source=generated_by_source,
                    verifier_required=True,
                ),
                evidence.reference_profile,
                phase=QualityPhase.FULL,
                reference_blind_verifier_means=reference_means,
                matched_reference_profile=evidence.paired_profile,
                expected_feature_rates=evidence.expected_feature_rates,
            )
            if reference_verifier_failed:
                report = replace(
                    report,
                    failures=report.failures + ("reference_verifier_failure",),
                )
            report_values.append(report)
        reports = tuple(report_values)
        attributed_costs = _attributed_full_route_costs(
            cache,
            full_samples,
            generated_verified,
        )
        return tuple(
            replace(
                report,
                billed_cost_usd=float(attributed_costs[report.route_id]),
            )
            for report in reports
        )
    except (KeyError, TypeError, ValueError) as error:
        raise Phase1ReplayError(f"could not recompute full quality: {error}") from error


def _load_quality_reference_evidence(source: Path) -> _QualityReferenceEvidence:
    paired_observations = _load_reference_observations(
        source / "paired_reference_observations.jsonl"
    )
    if len(paired_observations) != PHASE1_FULL_COUNT:
        raise Phase1ReplayError("quality replay requires 200 paired references")
    statistics = _json_object(source / "reference_statistics.json")
    ingredient_profile = require_json_object(
        statistics.get("ingredient_profile"),
        label="reference ingredient profile",
    )
    expected_feature_rates = _string_float_pairs(
        ingredient_profile.get("narrative_feature_rates"),
        "narrative feature rates",
        maximum=1.0,
    )
    return _QualityReferenceEvidence(
        reference_profile=_decode_reference_profile(
            statistics.get("reference_profile"),
            "reference profile",
        ),
        paired_observations=paired_observations,
        paired_profile=build_reference_profile(paired_observations),
        expected_feature_rates=expected_feature_rates,
    )


def _decode_reference_profile(value: JsonValue, label: str) -> ReferenceProfile:
    try:
        record = require_json_object(value, label=label)
        require_exact_fields(
            record,
            (
                "alphanumeric_identifier_token_rate",
                "dialogue_rate",
                "digit_bearing_token_rate",
                "ending_frequencies",
                "median_normalized_nll",
                "median_repeated_ngram_fraction",
                "median_sentence_words",
                "median_story_words",
                "model_token_counts",
                "normalized_nll_iqr",
                "normalized_nll_values",
                "numeric_token_rate",
                "opening_frequencies",
                "paragraph_break_rate",
                "profile_sha256",
                "realized_feature_rates",
                "record_count",
                "reference_split_token_jsd",
                "repeated_ngram_fractions",
                "requested_feature_rates",
                "required_word_frequencies",
                "sentence_word_counts",
                "story_word_counts",
                "token_probabilities",
                "vocabulary",
                "word_frequencies",
            ),
            label=label,
        )
    except CanonicalJsonError as error:
        raise Phase1ReplayError(str(error)) from error
    record_count = _integer(record["record_count"], f"{label} record count")
    if record_count <= 0:
        raise Phase1ReplayError(f"{label} record count must be positive")
    vocabulary = _text_tuple(record["vocabulary"], f"{label} vocabulary")
    if vocabulary != tuple(sorted(vocabulary)) or len(vocabulary) != len(set(vocabulary)):
        raise Phase1ReplayError(f"{label} vocabulary must be unique and ordered")
    profile = ReferenceProfile(
        record_count=record_count,
        vocabulary=frozenset(vocabulary),
        word_frequencies=_string_integer_pairs(
            record["word_frequencies"], f"{label} word frequencies"
        ),
        required_word_frequencies=_string_integer_pairs(
            record["required_word_frequencies"],
            f"{label} required-word frequencies",
            allow_empty=True,
        ),
        token_probabilities=_integer_float_pairs(
            record["token_probabilities"],
            f"{label} token probabilities",
            maximum=1.0,
        ),
        story_word_counts=_positive_integer_tuple(
            record["story_word_counts"], f"{label} story word counts"
        ),
        model_token_counts=_positive_integer_tuple(
            record["model_token_counts"], f"{label} model token counts"
        ),
        sentence_word_counts=_positive_integer_tuple(
            record["sentence_word_counts"], f"{label} sentence word counts"
        ),
        paragraph_break_rate=_rate(
            record["paragraph_break_rate"], f"{label} paragraph break rate"
        ),
        dialogue_rate=_rate(record["dialogue_rate"], f"{label} dialogue rate"),
        feature_rates=_string_float_pairs(
            record["requested_feature_rates"],
            f"{label} requested feature rates",
            maximum=1.0,
        ),
        realized_feature_rates=_string_float_pairs(
            record["realized_feature_rates"],
            f"{label} realized feature rates",
            maximum=1.0,
        ),
        repeated_ngram_fractions=_number_tuple(
            record["repeated_ngram_fractions"],
            f"{label} repeated ngram fractions",
            maximum=1.0,
        ),
        digit_bearing_token_rate=_rate(
            record["digit_bearing_token_rate"], f"{label} digit-bearing rate"
        ),
        numeric_token_rate=_rate(
            record["numeric_token_rate"], f"{label} numeric rate"
        ),
        alphanumeric_identifier_token_rate=_rate(
            record["alphanumeric_identifier_token_rate"],
            f"{label} alphanumeric identifier rate",
        ),
        opening_frequencies=_string_integer_pairs(
            record["opening_frequencies"], f"{label} opening frequencies"
        ),
        ending_frequencies=_string_integer_pairs(
            record["ending_frequencies"], f"{label} ending frequencies"
        ),
        normalized_nll_values=_number_tuple(
            record["normalized_nll_values"], f"{label} normalized NLL values"
        ),
        reference_split_token_jsd=_rate(
            record["reference_split_token_jsd"], f"{label} split token JSD"
        ),
        profile_sha256=_text(record["profile_sha256"], f"{label} digest"),
    )
    if any(
        len(values) != record_count
        for values in (
            profile.story_word_counts,
            profile.model_token_counts,
            profile.repeated_ngram_fractions,
            profile.normalized_nll_values,
        )
    ):
        raise Phase1ReplayError(f"{label} per-story arrays must match record count")
    if not profile.sentence_word_counts or not profile.token_probabilities:
        raise Phase1ReplayError(f"{label} distributions must be nonempty")
    if not math.isclose(
        sum(probability for _, probability in profile.token_probabilities),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise Phase1ReplayError(f"{label} token probabilities must sum to one")
    derived = (
        (profile.median_normalized_nll, record["median_normalized_nll"]),
        (
            profile.median_repeated_ngram_fraction,
            record["median_repeated_ngram_fraction"],
        ),
        (profile.median_sentence_words, record["median_sentence_words"]),
        (profile.median_story_words, record["median_story_words"]),
        (profile.normalized_nll_iqr, record["normalized_nll_iqr"]),
    )
    if any(
        actual != _number(stored, f"{label} derived statistic")
        for actual, stored in derived
    ):
        raise Phase1ReplayError(f"{label} stored medians/IQR are inconsistent")
    return profile


def _string_integer_pairs(
    value: JsonValue,
    label: str,
    *,
    allow_empty: bool = False,
) -> tuple[tuple[str, int], ...]:
    values = _list(value, label)
    result: list[tuple[str, int]] = []
    for index, item in enumerate(values):
        pair = _list(item, f"{label} pair {index}")
        if len(pair) != 2:
            raise Phase1ReplayError(f"{label} pairs must have length two")
        count = _integer(pair[1], f"{label} count {index}")
        if count <= 0:
            raise Phase1ReplayError(f"{label} counts must be positive")
        result.append((_text(pair[0], f"{label} name {index}"), count))
    output = tuple(result)
    if not output and not allow_empty:
        raise Phase1ReplayError(f"{label} must not be empty")
    if output != tuple(sorted(output)) or len({name for name, _ in output}) != len(output):
        raise Phase1ReplayError(f"{label} must have unique names in order")
    return output


def _string_float_pairs(
    value: JsonValue,
    label: str,
    *,
    maximum: float | None = None,
) -> tuple[tuple[str, float], ...]:
    values = _list(value, label)
    result: list[tuple[str, float]] = []
    for index, item in enumerate(values):
        pair = _list(item, f"{label} pair {index}")
        if len(pair) != 2:
            raise Phase1ReplayError(f"{label} pairs must have length two")
        number = _number(pair[1], f"{label} value {index}")
        if maximum is not None and number > maximum:
            raise Phase1ReplayError(f"{label} values exceed {maximum}")
        result.append((_text(pair[0], f"{label} name {index}"), number))
    output = tuple(result)
    if output != tuple(sorted(output)) or len({name for name, _ in output}) != len(output):
        raise Phase1ReplayError(f"{label} must have unique names in order")
    return output


def _integer_float_pairs(
    value: JsonValue,
    label: str,
    *,
    maximum: float | None = None,
) -> tuple[tuple[int, float], ...]:
    values = _list(value, label)
    result: list[tuple[int, float]] = []
    for index, item in enumerate(values):
        pair = _list(item, f"{label} pair {index}")
        if len(pair) != 2:
            raise Phase1ReplayError(f"{label} pairs must have length two")
        number = _number(pair[1], f"{label} value {index}")
        if maximum is not None and number > maximum:
            raise Phase1ReplayError(f"{label} values exceed {maximum}")
        result.append((_integer(pair[0], f"{label} key {index}"), number))
    output = tuple(result)
    if output != tuple(sorted(output)) or len({key for key, _ in output}) != len(output):
        raise Phase1ReplayError(f"{label} must have unique keys in order")
    return output


def _positive_integer_tuple(value: JsonValue, label: str) -> tuple[int, ...]:
    values = _integer_tuple(value, label)
    if not values or any(item <= 0 for item in values):
        raise Phase1ReplayError(f"{label} must contain positive integers")
    return values


def _number_tuple(
    value: JsonValue,
    label: str,
    *,
    maximum: float | None = None,
) -> tuple[float, ...]:
    values = tuple(
        _number(item, f"{label} value {index}")
        for index, item in enumerate(_list(value, label))
    )
    if not values:
        raise Phase1ReplayError(f"{label} must not be empty")
    if maximum is not None and any(item > maximum for item in values):
        raise Phase1ReplayError(f"{label} values exceed {maximum}")
    return values


def _rate(value: JsonValue, label: str) -> float:
    result = _number(value, label)
    if result > 1.0:
        raise Phase1ReplayError(f"{label} must not exceed one")
    return result


def _write_replayed_audit(
    builder: Phase1ArtifactBuilder,
    source: Path,
    briefs: tuple[NeutralStoryBrief, ...],
    samples: tuple[GeneratedSample, ...],
    verifier_results: tuple[VerifiedStory, ...],
    finalist_ids: tuple[str, ...],
) -> None:
    """Reconstruct the blinded audit from persisted semantic evidence."""
    paired = {
        item.record_id: item
        for item in _load_reference_observations(
            source / "paired_reference_observations.jsonl"
        )
    }
    screen_measurements = _load_measurement_batch(
        source / "measurements" / "screen.jsonl"
    )
    expansion_measurements = _load_measurement_batch(
        source / "measurements" / "finalist_expansion.jsonl"
    )
    if screen_measurements.by_id.keys() & expansion_measurements.by_id.keys():
        raise Phase1ReplayError("screen and expansion measurement IDs overlap")
    measurements = {**screen_measurements.by_id, **expansion_measurements.by_id}
    reference_verifier_by_pair = {
        item.job.pair_id: item
        for item in verifier_results
        if item.job.source_id.startswith("reference:")
    }
    generated_verifier_by_source = {
        item.job.source_id: item
        for item in verifier_results
        if not item.job.source_id.startswith("reference:")
    }
    try:
        references = tuple(
            AuditSourceRecord(
                source_id=f"reference:{brief.brief_id}",
                pair_id=brief.brief_id,
                story_text=brief.matched_reference_text,
                source_prompt=brief.prompt_text,
                token_count=len(paired[brief.source_record_id].model_token_ids),
                base_normalized_nll=paired[
                    brief.source_record_id
                ].normalized_nll,
                automated_style_scores=_style_scores(
                    reference_verifier_by_pair[brief.brief_id].payload
                ),
                source_kind=AuditSourceKind.REFERENCE,
            )
            for brief in briefs
        )
        generated = tuple(
            AuditSourceRecord(
                source_id=sample.sample_id,
                pair_id=sample.job.brief.brief_id,
                story_text=sample.payload.story,
                source_prompt=sample.job.brief.prompt_text,
                token_count=len(measurements[sample.sample_id].model_token_ids),
                base_normalized_nll=measurements[sample.sample_id].normalized_nll,
                automated_style_scores=_style_scores(
                    generated_verifier_by_source[sample.sample_id].payload
                ),
                source_kind=AuditSourceKind.GENERATED,
                route_id=sample.job.route.route_id,
            )
            for sample in samples
            if sample.job.route.route_id in finalist_ids
            and sample.validation.accepted
            and sample.payload is not None
        )
    except KeyError as error:
        raise Phase1ReplayError(
            "audit source lacks paired, measurement, or verifier evidence"
        ) from error
    packet, key = build_blinded_audit(
        references,
        generated,
        finalist_order=finalist_ids,
        seed="tinyworlds-v2-phase1-blinded-audit-v1",
        reference_count=100,
        generated_count=100,
    )
    builder.write_bytes(
        "audit_packet.json",
        encode_blinded_audit_packet(packet),
    )
    builder.write_bytes("audit_key.json", encode_blinded_audit_key(key))
    builder.write_bytes("audit.html", render_audit_html(packet).encode("utf-8"))


def _load_reference_observations(path: Path) -> tuple[ReferenceObservation, ...]:
    observations = []
    for index, record in enumerate(_jsonl_objects(path)):
        require_exact_fields(
            record,
            (
                "dialogue_present",
                "ending_key",
                "model_token_ids",
                "normalized_nll",
                "opening_key",
                "paragraph_count",
                "realized_feature_labels",
                "record_id",
                "repeated_ngram_fraction",
                "requested_feature_labels",
                "required_words",
                "sentence_word_counts",
                "word_tokens",
            ),
            label=f"reference observation {index}",
        )
        try:
            observations.append(
                ReferenceObservation(
                    record_id=_text(record["record_id"], "observation record_id"),
                    word_tokens=_text_tuple(record["word_tokens"], "word tokens"),
                    model_token_ids=_integer_tuple(
                        record["model_token_ids"], "model token IDs"
                    ),
                    sentence_word_counts=_integer_tuple(
                        record["sentence_word_counts"], "sentence word counts"
                    ),
                    paragraph_count=_integer(
                        record["paragraph_count"], "paragraph count"
                    ),
                    dialogue_present=_boolean(
                        record["dialogue_present"], "dialogue present"
                    ),
                    opening_key=_text(record["opening_key"], "opening key"),
                    ending_key=_text(record["ending_key"], "ending key"),
                    feature_labels=_text_tuple(
                        record["requested_feature_labels"],
                        "requested feature labels",
                    ),
                    normalized_nll=_number(
                        record["normalized_nll"], "normalized NLL"
                    ),
                    required_words=_text_tuple(
                        record["required_words"], "required words"
                    ),
                    realized_feature_labels=_text_tuple(
                        record["realized_feature_labels"],
                        "realized feature labels",
                    ),
                    repeated_ngram_fraction=_number(
                        record["repeated_ngram_fraction"],
                        "repeated ngram fraction",
                    ),
                )
            )
        except (TypeError, ValueError) as error:
            raise Phase1ReplayError(
                f"invalid reference observation {index}: {error}"
            ) from error
    return tuple(observations)


def _load_measurement_batch(path: Path) -> MeasurementBatch:
    measurements: list[StoryMeasurement] = []
    record_ids: set[str] = set()
    for index, record in enumerate(_jsonl_objects(path)):
        require_exact_fields(
            record,
            (
                "active_token_count",
                "model_token_ids",
                "normalized_nll",
                "record_id",
            ),
            label=f"story measurement {index}",
        )
        record_id = _text(record["record_id"], "measurement record_id")
        if record_id in record_ids:
            raise Phase1ReplayError("story measurement record ID repeats")
        record_ids.add(record_id)
        token_ids = _integer_tuple(record["model_token_ids"], "model token IDs")
        active_count = _integer(record["active_token_count"], "active token count")
        if not token_ids or active_count < 1 or active_count > len(token_ids):
            raise Phase1ReplayError("story measurement token counts are invalid")
        measurements.append(
            StoryMeasurement(
                record_id,
                token_ids,
                _number(record["normalized_nll"], "measurement NLL"),
                active_count,
            )
        )
    try:
        return MeasurementBatch(tuple(measurements), {})
    except (TypeError, ValueError) as error:
        raise Phase1ReplayError(f"invalid measurement batch: {error}") from error


def _write_generation_streams(
    builder: Phase1ArtifactBuilder,
    samples: tuple[GeneratedSample, ...],
    finalist_ids: tuple[str, ...],
) -> None:
    if samples:
        builder.write_bytes(
            "generator_bakeoff.jsonl",
            canonical_jsonl_bytes(sample.as_record() for sample in samples),
        )
    sample_by_route = {
        model.route_id: tuple(
            sample for sample in samples if sample.job.route.route_id == model.route_id
        )
        for model in CANDIDATE_MODELS
    }
    for model in CANDIDATE_MODELS:
        route_samples = sample_by_route[model.route_id]
        builder.write_bytes(
            f"routes/{model.route_id}/accepted.jsonl",
            canonical_jsonl_bytes(
                sample.as_record() for sample in route_samples if sample.validation.accepted
            ),
        )
        builder.write_bytes(
            f"routes/{model.route_id}/rejected.jsonl",
            canonical_jsonl_bytes(
                sample.as_record() for sample in route_samples if not sample.validation.accepted
            ),
        )
        builder.write_bytes(
            f"routes/{model.route_id}/raw_responses.jsonl",
            canonical_jsonl_bytes(
                {
                    "billed_cost_usd": sample.billed_cost_usd,
                    "error_kind": sample.error_kind,
                    "generation_id": sample.generation_id,
                    "input_tokens": sample.input_tokens,
                    "output_tokens": sample.output_tokens,
                    "request_sha256": sample.job.request.request_sha256,
                    "sample_id": sample.sample_id,
                }
                for sample in route_samples
            ),
        )
    if samples:
        events: list[JsonObject] = []
        for model in CANDIDATE_MODELS:
            screen = sample_by_route[model.route_id][:PHASE1_SCREEN_COUNT]
            if len(screen) == PHASE1_SCREEN_COUNT:
                events.append(
                {
                    "accepted": sum(item.validation.accepted for item in screen),
                    "event": "generation_batch_completed",
                    "request_count": len(screen),
                    "route_id": model.route_id,
                    "stage": "screen",
                }
                )
        for route_id in finalist_ids:
            expansion = sample_by_route[route_id][PHASE1_SCREEN_COUNT:]
            if expansion:
                events.append(
                {
                    "accepted": sum(item.validation.accepted for item in expansion),
                    "event": "generation_batch_completed",
                    "request_count": len(expansion),
                    "route_id": route_id,
                    "stage": "finalist-expansion",
                }
                )
        builder.write_bytes("sequential_results.jsonl", canonical_jsonl_bytes(events))


def _write_verifier_events(
    builder: Phase1ArtifactBuilder,
    reference_count: int,
    generated_count: int,
) -> None:
    path = builder.root / "sequential_results.jsonl"
    existing = path.read_bytes() if path.exists() else b""
    events = tuple(
        {
            "event": "verifier_batch_completed",
            "request_count": min(50, count - start),
            "stage": stage,
            "start_index": start,
        }
        for stage, count in (
            ("genuine-reference", reference_count),
            ("generated-finalist", generated_count),
        )
        for start in range(0, count, 50)
    )
    payload = existing + canonical_jsonl_bytes(events)
    if path.exists():
        path.unlink()
    builder.write_bytes("sequential_results.jsonl", payload)


def _validate_planned_requests(
    root: Path,
    jobs: tuple[GenerationJob, ...],
) -> None:
    by_route = {
        model.route_id: tuple(job for job in jobs if job.route.route_id == model.route_id)
        for model in CANDIDATE_MODELS
    }
    for route_id, route_jobs in by_route.items():
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
            raise Phase1ReplayError(f"planned request records differ for {route_id}")


def _load_briefs(path: Path) -> tuple[NeutralStoryBrief, ...]:
    records = _jsonl_objects(path)
    briefs: list[NeutralStoryBrief] = []
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
            label=f"neutral brief {index}",
        )
        briefs.append(
            NeutralStoryBrief(
                brief_id=_text(record["brief_id"], "brief_id"),
                source_record_id=_text(record["source_record_id"], "source_record_id"),
                prompt_text=_text(record["prompt_text"], "prompt_text"),
                required_words=_string_tuple(record["required_words"], "required_words"),
                requested_features=_string_tuple(
                    record["requested_features"], "requested_features"
                ),
                matched_reference_text=_text(
                    record["matched_reference_text"], "matched_reference_text"
                ),
            )
        )
    if len(briefs) != PHASE1_FULL_COUNT:
        raise Phase1ReplayError("Phase 1 replay requires exactly 200 neutral briefs")
    return tuple(briefs)


def _load_routes(path: Path) -> tuple[tuple[RouteLock, ...], RouteLock]:
    record = _json_object(path)
    require_exact_fields(
        record,
        ("generator_routes", "snapshot_sha256", "verifier_route"),
        label="catalog routes",
    )
    route_values = record["generator_routes"]
    if type(route_values) is not list:
        raise Phase1ReplayError("generator_routes must be a list")
    routes = tuple(_route(value, f"generator route {index}") for index, value in enumerate(route_values))
    verifier = _route(record["verifier_route"], "verifier route")
    if tuple(route.route_id for route in routes) != tuple(
        model.route_id for model in CANDIDATE_MODELS
    ):
        raise Phase1ReplayError("catalog generator route order differs from the preset")
    return routes, verifier


def _route(value: JsonValue, label: str) -> RouteLock:
    record = require_json_object(value, label=label)
    require_exact_fields(
        record,
        (
            "canonical_model",
            "catalog_sha256",
            "input_usd_per_million",
            "output_usd_per_million",
            "provider_slug",
            "quantization",
            "requested_model",
            "returned_provider",
            "route_id",
        ),
        label=label,
    )
    return RouteLock(**{key: _text(value, f"{label} {key}") for key, value in record.items()})


def _json_object(path: Path) -> JsonObject:
    try:
        return require_json_object(
            canonical_json_loads(path.read_bytes(), label=path.as_posix()),
            label=path.as_posix(),
        )
    except (OSError, CanonicalJsonError) as error:
        raise Phase1ReplayError(str(error)) from error


def _jsonl_objects(path: Path) -> tuple[JsonObject, ...]:
    try:
        payload = path.read_bytes()
        if payload and not payload.endswith(b"\n"):
            raise Phase1ReplayError(f"noncanonical JSONL framing: {path}")
        return tuple(
            require_json_object(
                canonical_json_loads(line, label=f"{path} line {index}"),
                label=f"{path} line {index}",
            )
            for index, line in enumerate(payload.splitlines(), start=1)
        )
    except (OSError, CanonicalJsonError) as error:
        raise Phase1ReplayError(str(error)) from error


def _text(value: JsonValue, label: str) -> str:
    if type(value) is not str or not value:
        raise Phase1ReplayError(f"{label} must be nonempty text")
    return value


def _text_tuple(value: JsonValue, label: str) -> tuple[str, ...]:
    if type(value) is not list or any(type(item) is not str for item in value):
        raise Phase1ReplayError(f"{label} must be a string list")
    return tuple(value)


def _list(value: JsonValue, label: str) -> list[JsonValue]:
    if type(value) is not list:
        raise Phase1ReplayError(f"{label} must be a list")
    return value


def _integer_tuple(value: JsonValue, label: str) -> tuple[int, ...]:
    if type(value) is not list or any(
        type(item) is not int or item < 0 for item in value
    ):
        raise Phase1ReplayError(f"{label} must be a nonnegative integer list")
    return tuple(value)


def _integer(value: JsonValue, label: str) -> int:
    if type(value) is not int or value < 0:
        raise Phase1ReplayError(f"{label} must be a nonnegative integer")
    return value


def _number(value: JsonValue, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise Phase1ReplayError(f"{label} must be finite and nonnegative")
    return float(value)


def _boolean(value: JsonValue, label: str) -> bool:
    if type(value) is not bool:
        raise Phase1ReplayError(f"{label} must be boolean")
    return value


def _string_tuple(value: JsonValue, label: str) -> tuple[str, ...]:
    if type(value) is not list or any(type(item) is not str for item in value):
        raise Phase1ReplayError(f"{label} must be a string list")
    return tuple(value)


__all__ = [
    "Phase1ReplayError",
    "Phase1ReplayResult",
    "replay_phase1_derived",
    "verify_phase1_derived_replay",
]

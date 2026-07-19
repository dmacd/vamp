"""Cross-artifact semantic validation for TinyWorlds-v2 Phase 1 bundles."""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime, timezone
from hashlib import sha256
import math
from pathlib import Path
import re
from typing import cast

from apm.data.text.curricula import TINYSTORIES_V2_SOURCE
from apm.data.text.tinyworlds_v2.audit import AuditSourceKind
from apm.data.text.tinyworlds_v2.audit_io import (
    AuditIoError,
    _validate_phase1_reference_overlays,
    decode_audit_pair,
    validate_phase1_tree_with_human_overlays,
)
from apm.data.text.tinyworlds_v2.bakeoff import (
    CANDIDATE_MODELS,
    VERIFIER_MODEL,
    NeutralStoryBrief,
)
from apm.data.text.tinyworlds_v2.generation_cache import (
    GenerationCacheError,
    ImmutableRawCache,
)
from apm.data.text.tinyworlds_v2.ingredients import (
    mechanically_classify_ingredient_roles,
)
from apm.data.text.tinyworlds_v2.json_contracts import (
    JsonObject,
    JsonValue,
    json_sha256,
    require_exact_fields,
)
from apm.data.text.tinyworlds_v2.phase1_artifacts import Phase1ArtifactManifest
from apm.data.text.tinyworlds_v2.phase1_replay import (
    _json_object,
    _jsonl_objects,
    verify_phase1_derived_replay,
)
from apm.data.text.tinyworlds_v2.quality import (
    QualityOutcome,
    QualityPhase,
    QualitySelection,
    RouteQualityReport,
    select_full_quality_routes,
    select_screen_finalists,
    validate_route_quality_report,
)
from apm.data.text.tinyworlds_v2.reference_pipeline import (
    PHASE1_ARCHIVE_REFERENCE_COUNT,
    PHASE1_BRIEF_COUNT,
    PHASE1_PROMPT_METADATA_COUNT,
    PHASE1_VALIDATION_REFERENCE_COUNT,
    REFERENCE_SURFACE_WORKERS,
    ReferenceAnnotation,
    build_prompt_ingredient_profile,
    canonical_prompt_ingredient_profile,
    canonical_reference_observation,
)
from apm.data.text.tinyworlds_v2.reference_profile import (
    ReferenceObservation,
    ReferenceProfile,
    ReferenceRecord,
    build_reference_profile,
    observe_reference,
)
from apm.data.text.tinyworlds_v2.source_data import (
    TINYSTORIES_ALL_DATA_SOURCE,
    ArchiveSourceRecord,
    TinyStoriesInstruction,
    ValidationStoryRecord,
)
from apm.data.text.tinyworlds_v2.surface import (
    canonical_feature_labels,
    normalized_story_sha256,
)


_SCREEN_COUNT = 50
_FULL_COUNT = 200
_AUDIT_COUNT = 100
_PROJECTED_ACCEPTED = 4_000
_SELECTION_SEED = "tinyworlds-v2-phase1-reference-v1"
_AUDIT_SEED = "tinyworlds-v2-phase1-blinded-audit-v1"
_VERSION = "tinyworlds-v2-phase1-reference-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_ARCHIVE_RECORD_ID = re.compile(r"archive:(.+):([0-9]+):([0-9a-f]{64})")
_STATUSES = frozenset(
    (
        "blocked_by_cost_cap",
        "blocked_by_runtime_cost_cap",
        "provider_billing_unknown",
        "catalog_route_drift",
        "no_quality_qualified_route",
        "audit_insufficient_accepted_samples",
        "awaiting_human_audit",
    )
)
_INTERRUPTED_STATUSES = frozenset(
    (
        "blocked_by_runtime_cost_cap",
        "catalog_route_drift",
        "provider_billing_unknown",
    )
)


class Phase1SemanticError(ValueError):
    """A structurally authenticated artifact contains contradictory evidence."""


@dataclass(frozen=True, slots=True)
class Phase1SemanticValidation:
    """Summary of the exact contract proven by semantic validation."""

    manifest: Phase1ArtifactManifest
    status: str
    finalist_route_ids: tuple[str, ...]
    qualified_route_ids: tuple[str, ...]
    generated_request_count: int
    verifier_request_count: int
    replay_file_count: int


@dataclass(frozen=True, slots=True)
class _PromptMetadataEvidence:
    record: ArchiveSourceRecord
    normalized_story_sha256: str


def validate_phase1_semantics(root: str | Path) -> Phase1SemanticValidation:
    """Authenticate and cross-check every Phase 1 selection/evidence boundary."""
    directory = Path(root)
    manifest = validate_phase1_tree_with_human_overlays(directory)
    status_record = _json_object(directory / "status.json")
    require_exact_fields(
        status_record,
        ("audit_sha256", "phase", "status"),
        label="Phase 1 status",
    )
    status = _text(status_record["status"], "Phase 1 status")
    if status not in _STATUSES or status_record["phase"] != 1:
        raise Phase1SemanticError("Phase 1 status or phase is not a fixed contract value")

    source = _json_object(directory / "source_manifest.json")
    _validate_source_manifest(source)
    _validate_source_cohort_counts(directory, manifest)
    configuration = _json_object(directory / "configuration.json")
    _validate_configuration(configuration, directory)
    _validate_byok_preflight(directory, status)

    screen_reports, full_reports, stored_selection = _load_quality_details(
        directory / "quality_details.json"
    )
    finalist = _validate_screen_selection(
        directory,
        status,
        screen_reports,
        stored_selection,
        has_full_reports=bool(full_reports),
    )
    qualified = _validate_full_selection(directory, status, finalist, full_reports)
    generated = _jsonl_objects(directory / "generator_bakeoff.jsonl") if (
        directory / "generator_bakeoff.jsonl"
    ).is_file() else ()
    verifier = _jsonl_objects(directory / "verifier_results.jsonl") if (
        directory / "verifier_results.jsonl"
    ).is_file() else ()
    _validate_result_counts_and_splits(directory, status, finalist, generated, verifier)
    _validate_audit(
        directory,
        status_record,
        status,
        finalist,
        generated,
    )
    _validate_audit_feasibility_stop(directory, status, finalist, generated)
    if status == "awaiting_human_audit":
        # This also validates any permitted human decision/approval overlays.
        _validate_phase1_reference_overlays(directory)
    replay = verify_phase1_derived_replay(directory)
    return Phase1SemanticValidation(
        manifest=manifest,
        status=status,
        finalist_route_ids=finalist,
        qualified_route_ids=qualified,
        generated_request_count=len(generated),
        verifier_request_count=len(verifier),
        replay_file_count=len(replay.compared_paths),
    )


def _validate_byok_preflight(root: Path, status: str) -> None:
    """Bind every possible paid POST boundary to sanitized zero-BYOK proof."""
    evidence_path = root / "byok_preflight.json"
    raw_root = root / "raw_cache"
    # A durable reservation is the last local event before a completion POST.
    # Requiring evidence at this boundary also covers a crash after POST but
    # before raw-response persistence, while ordinary static/runtime cap stops
    # that never authorized a request remain valid without an attestation.
    try:
        journal = ImmutableRawCache(raw_root).load_cost_journal()
    except (GenerationCacheError, TypeError, ValueError) as error:
        raise Phase1SemanticError(
            "runtime cost journal has invalid embedded BYOK authorization"
        ) from error
    # Every journal record has independently validated, digest-bound, allowed
    # authorization evidence.  A cancellation is explicitly before transport
    # and therefore is not itself a paid boundary.
    has_paid_boundary = any(
        not entry.cancelled_before_post for entry in journal
    ) or any(
        path.name == "metadata.json"
        for path in raw_root.glob("requests/*/attempts/*/metadata.json")
    )
    if not evidence_path.is_file():
        if has_paid_boundary:
            raise Phase1SemanticError(
                "paid completion evidence requires byok_preflight.json"
            )
        runtime_path = root / "runtime_cost_ledger.json"
        if runtime_path.is_file():
            runtime = _json_object(runtime_path)
            if runtime.get("halted_reason") == "byok_preflight_failed" and (
                status != "provider_billing_unknown"
            ):
                raise Phase1SemanticError(
                    "BYOK preflight failure has a contradictory top-level status"
                )
        return

    evidence = _json_object(evidence_path)
    require_exact_fields(
        evidence,
        (
            "attestation_sha256",
            "attested_at_utc",
            "checked_at_utc",
            "decision",
            "endpoint",
            "expires_at_utc",
            "method",
            "response_body_sha256",
            "source",
            "status_code",
            "total_count",
        ),
        label="BYOK preflight evidence",
    )
    checked = _utc_timestamp(
        evidence["checked_at_utc"], "BYOK preflight checked_at_utc"
    )
    decision = evidence["decision"]
    source = evidence["source"]
    if decision not in {"allowed", "blocked", "unverified"}:
        raise Phase1SemanticError("BYOK preflight decision is not recognized")
    if source == "management_api":
        if evidence["endpoint"] != "/api/v1/byok" or evidence["method"] != "GET":
            raise Phase1SemanticError("management BYOK endpoint contract differs")
        if any(
            evidence[field] is not None
            for field in (
                "attestation_sha256",
                "attested_at_utc",
                "expires_at_utc",
            )
        ):
            raise Phase1SemanticError(
                "management BYOK evidence contains manual-attestation fields"
            )
        status_code = evidence["status_code"]
        if type(status_code) is not int or not 100 <= status_code <= 599:
            raise Phase1SemanticError("management BYOK status is not an HTTP status")
        digest = evidence["response_body_sha256"]
        total_count = evidence["total_count"]
        if decision == "allowed":
            if status_code != 200 or not _is_sha256(digest) or total_count != 0:
                raise Phase1SemanticError(
                    "allowed management BYOK evidence lacks an exact zero-key proof"
                )
        elif decision == "blocked":
            if (
                status_code != 200
                or not _is_sha256(digest)
                or type(total_count) is not int
                or total_count < 0
            ):
                raise Phase1SemanticError("blocked management BYOK evidence is malformed")
        elif total_count is not None or (
            digest is not None and not _is_sha256(digest)
        ):
            raise Phase1SemanticError("unverified management BYOK evidence is malformed")
    elif source == "manual_attestation":
        if any(
            evidence[field] is not None
            for field in (
                "endpoint",
                "method",
                "response_body_sha256",
                "status_code",
            )
        ) or not _is_sha256(evidence["attestation_sha256"]):
            raise Phase1SemanticError("manual BYOK evidence is not sanitized")
        if decision == "allowed":
            if evidence["total_count"] != 0:
                raise Phase1SemanticError("allowed manual BYOK evidence is not zero-key")
            attested = _utc_timestamp(
                evidence["attested_at_utc"], "BYOK attested_at_utc"
            )
            expires = _utc_timestamp(
                evidence["expires_at_utc"], "BYOK expires_at_utc"
            )
            lifetime = (expires - attested).total_seconds()
            if not 0 < lifetime <= 24 * 60 * 60 or not attested <= checked < expires:
                raise Phase1SemanticError(
                    "manual BYOK attestation was not valid at preflight time"
                )
        elif decision == "unverified":
            if evidence["total_count"] is not None or any(
                evidence[field] is not None
                for field in ("attested_at_utc", "expires_at_utc")
            ):
                raise Phase1SemanticError("unverified manual BYOK evidence is malformed")
        else:
            raise Phase1SemanticError("manual BYOK evidence cannot be blocked")
    else:
        raise Phase1SemanticError("BYOK preflight source is not recognized")

    runtime_path = root / "runtime_cost_ledger.json"
    runtime_reason = (
        _json_object(runtime_path).get("halted_reason")
        if runtime_path.is_file()
        else None
    )
    if decision != "allowed":
        if status != "provider_billing_unknown" or runtime_reason != "byok_preflight_failed":
            raise Phase1SemanticError(
                "failed BYOK preflight lacks its exact terminal status"
            )
    elif runtime_reason == "byok_preflight_failed":
        raise Phase1SemanticError(
            "allowed BYOK evidence contradicts a preflight-failure halt"
        )


def _utc_timestamp(value: JsonValue, label: str) -> datetime:
    if type(value) is not str or not value.endswith("Z"):
        raise Phase1SemanticError(f"{label} must be a canonical UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise Phase1SemanticError(f"{label} must be ISO-8601") from error
    if parsed.tzinfo is None:
        raise Phase1SemanticError(f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _is_sha256(value: JsonValue) -> bool:
    return type(value) is str and _SHA256.fullmatch(value) is not None


def _validate_source_manifest(record: JsonObject) -> None:
    require_exact_fields(
        record,
        (
            "archive",
            "counts",
            "selection_seed",
            "story_identity_policy",
            "validation",
        ),
        label="source manifest",
    )
    archive = _object(record["archive"], "source archive")
    validation = _object(record["validation"], "validation source")
    counts = _object(record["counts"], "source counts")
    expected_archive = {
        "dataset_id": TINYSTORIES_ALL_DATA_SOURCE.dataset_id,
        "filename": TINYSTORIES_ALL_DATA_SOURCE.archive_file.filename,
        "revision": TINYSTORIES_ALL_DATA_SOURCE.revision,
        "sha256": TINYSTORIES_ALL_DATA_SOURCE.archive_file.sha256,
        "size_bytes": TINYSTORIES_ALL_DATA_SOURCE.archive_file.size_bytes,
    }
    expected_validation = {
        "dataset_id": TINYSTORIES_V2_SOURCE.dataset_id,
        "document_separator": TINYSTORIES_V2_SOURCE.document_separator,
        "filename": TINYSTORIES_V2_SOURCE.validation_file.filename,
        "revision": TINYSTORIES_V2_SOURCE.revision,
        "sha256": TINYSTORIES_V2_SOURCE.validation_file.sha256,
        "size_bytes": TINYSTORIES_V2_SOURCE.validation_file.size_bytes,
    }
    expected_counts = {
        "archive_reference": PHASE1_ARCHIVE_REFERENCE_COUNT,
        "neutral_briefs": PHASE1_BRIEF_COUNT,
        "prompt_metadata": PHASE1_PROMPT_METADATA_COUNT,
        "reference_profile": (
            PHASE1_ARCHIVE_REFERENCE_COUNT + PHASE1_VALIDATION_REFERENCE_COUNT
        ),
        "validation_reference": PHASE1_VALIDATION_REFERENCE_COUNT,
    }
    if archive != expected_archive or validation != expected_validation:
        raise Phase1SemanticError("source manifest differs from the pinned corpora")
    if (
        counts != expected_counts
        or record["selection_seed"] != _SELECTION_SEED
        or record["story_identity_policy"]
        != "unicode-nfkc-casefold-whitespace-collapse-sha256-v1"
    ):
        raise Phase1SemanticError("source cohort sizes or selection seed differ")


def _validate_source_cohort_counts(
    root: Path,
    manifest: Phase1ArtifactManifest,
) -> None:
    expected = {
        "neutral_story_briefs.jsonl": PHASE1_BRIEF_COUNT,
        "paired_reference_observations.jsonl": PHASE1_BRIEF_COUNT,
        "prompt_metadata_sample.jsonl": PHASE1_PROMPT_METADATA_COUNT,
        "reference_annotations.jsonl": (
            PHASE1_ARCHIVE_REFERENCE_COUNT + PHASE1_VALIDATION_REFERENCE_COUNT
        ),
        "reference_observations.jsonl": (
            PHASE1_ARCHIVE_REFERENCE_COUNT + PHASE1_VALIDATION_REFERENCE_COUNT
        ),
        "reference_story_sample.jsonl": (
            PHASE1_ARCHIVE_REFERENCE_COUNT + PHASE1_VALIDATION_REFERENCE_COUNT
        ),
        "validation_source_sample.jsonl": PHASE1_VALIDATION_REFERENCE_COUNT,
    }
    counts = {artifact.path: artifact.record_count for artifact in manifest.artifacts}
    if any(counts.get(path) != count for path, count in expected.items()):
        raise Phase1SemanticError("manifested source cohort record counts differ")

    briefs = tuple(
        _decode_neutral_brief(item, index)
        for index, item in enumerate(
            _jsonl_objects(root / "neutral_story_briefs.jsonl")
        )
    )
    prompt_records = tuple(
        _decode_prompt_metadata(item, index)
        for index, item in enumerate(
            _jsonl_objects(root / "prompt_metadata_sample.jsonl")
        )
    )
    reference_records = tuple(
        _decode_reference_record(item, index)
        for index, item in enumerate(
            _jsonl_objects(root / "reference_story_sample.jsonl")
        )
    )
    annotations = tuple(
        _decode_reference_annotation(item, index)
        for index, item in enumerate(
            _jsonl_objects(root / "reference_annotations.jsonl")
        )
    )
    observations = tuple(
        _decode_reference_observation(item, index)
        for index, item in enumerate(
            _jsonl_objects(root / "reference_observations.jsonl")
        )
    )
    paired_observations = tuple(
        _decode_reference_observation(item, index)
        for index, item in enumerate(
            _jsonl_objects(root / "paired_reference_observations.jsonl")
        )
    )
    validation_records = tuple(
        _decode_validation_record(item, index)
        for index, item in enumerate(
            _jsonl_objects(root / "validation_source_sample.jsonl")
        )
    )

    brief_ids = tuple(brief.brief_id for brief in briefs)
    if brief_ids != tuple(sorted(brief_ids)) or len(set(brief_ids)) != len(brief_ids):
        raise Phase1SemanticError("neutral brief IDs are not unique and ordered")
    paired_ids = tuple(brief.source_record_id for brief in briefs)
    if len(set(paired_ids)) != len(paired_ids):
        raise Phase1SemanticError("neutral briefs reuse a paired source record")
    for brief in briefs:
        expected_brief_id = "brief-" + sha256(
            f"tinyworlds-v2-phase1-brief\0{brief.source_record_id}".encode("utf-8")
        ).hexdigest()[:24]
        if brief.brief_id != expected_brief_id:
            raise Phase1SemanticError("neutral brief ID does not bind its source record")
        _require_archive_record_id(brief.source_record_id, "paired source record")
        if (
            mechanically_classify_ingredient_roles(
                brief.prompt_text,
                brief.required_words,
            )
            is None
        ):
            raise Phase1SemanticError(
                "neutral brief words do not have explicit noun/verb/adjective roles"
            )
        if brief.requested_features != canonical_feature_labels(
            brief.requested_features
        ):
            raise Phase1SemanticError("neutral brief feature labels are not canonical")

    prompt_ids = tuple(item.record.record_id for item in prompt_records)
    reference_ids = tuple(record.record_id for record in reference_records)
    annotation_ids = tuple(annotation.record_id for annotation in annotations)
    observation_ids = tuple(observation.record_id for observation in observations)
    validation_ids = tuple(record.record_id for record in validation_records)
    if len(set(prompt_ids)) != len(prompt_ids):
        raise Phase1SemanticError("prompt metadata record IDs are not unique")
    if reference_ids != tuple(sorted(reference_ids)) or len(set(reference_ids)) != len(
        reference_ids
    ):
        raise Phase1SemanticError("reference story IDs are not unique and ordered")
    if annotation_ids != reference_ids or observation_ids != reference_ids:
        raise Phase1SemanticError(
            "reference stories, annotations, and observations do not align by ID"
        )
    if len(set(validation_ids)) != len(validation_ids):
        raise Phase1SemanticError("validation source IDs are not unique")

    annotation_by_id = dict(zip(annotation_ids, annotations, strict=True))
    reference_by_id = dict(zip(reference_ids, reference_records, strict=True))
    validation_by_id = {record.record_id: record for record in validation_records}
    archive_reference_ids = tuple(
        record_id
        for record_id in reference_ids
        if annotation_by_id[record_id].source_partition == "archive"
    )
    validation_reference_ids = tuple(
        record_id
        for record_id in reference_ids
        if annotation_by_id[record_id].source_partition == "validation"
    )
    if (
        len(archive_reference_ids) != PHASE1_ARCHIVE_REFERENCE_COUNT
        or len(validation_reference_ids) != PHASE1_VALIDATION_REFERENCE_COUNT
        or set(validation_reference_ids) != set(validation_ids)
    ):
        raise Phase1SemanticError("reference source partitions do not match their cohorts")
    for record_id in archive_reference_ids:
        _require_archive_record_id(record_id, "archive reference")
    for record_id in validation_reference_ids:
        validation = validation_by_id[record_id]
        reference = reference_by_id[record_id]
        if (
            reference.story_text != validation.story
            or reference.prompt_text is not None
            or reference.source_model != "GPT-4"
        ):
            raise Phase1SemanticError(
                "validation reference does not reproduce its pinned source record"
            )
    for record_id in archive_reference_ids:
        if reference_by_id[record_id].source_model != "GPT-4":
            raise Phase1SemanticError("archive reference is not from the GPT-4 cohort")

    _validate_observation_derivations(
        reference_records,
        annotations,
        observations,
        label="reference",
    )
    paired_observation_ids = tuple(item.record_id for item in paired_observations)
    if paired_observation_ids != paired_ids:
        raise Phase1SemanticError("paired observations do not align with neutral briefs")
    for brief, observation in zip(briefs, paired_observations, strict=True):
        expected_observation = observe_reference(
            ReferenceRecord(
                brief.source_record_id,
                brief.matched_reference_text,
                prompt_text=brief.prompt_text,
                source_model="GPT-4",
            ),
            model_token_ids=observation.model_token_ids,
            normalized_nll=observation.normalized_nll,
            feature_labels=brief.requested_features,
            required_words=brief.required_words,
        )
        if canonical_reference_observation(expected_observation) != (
            canonical_reference_observation(observation)
        ):
            raise Phase1SemanticError(
                "paired observation is stale or contradicts its neutral brief"
            )

    prompt_hashes = tuple(
        item.normalized_story_sha256 for item in prompt_records
    )
    archive_reference_hashes = tuple(
        normalized_story_sha256(reference_by_id[record_id].story_text)
        for record_id in archive_reference_ids
    )
    paired_hashes = tuple(
        normalized_story_sha256(brief.matched_reference_text) for brief in briefs
    )
    validation_hashes = tuple(
        normalized_story_sha256(record.story) for record in validation_records
    )
    _require_disjoint_source_cohorts(
        (
            ("prompt metadata", prompt_ids, prompt_hashes),
            ("archive reference", archive_reference_ids, archive_reference_hashes),
            ("paired reference", paired_ids, paired_hashes),
            ("validation reference", validation_ids, validation_hashes),
        )
    )

    statistics = _json_object(root / "reference_statistics.json")
    require_exact_fields(
        statistics,
        (
            "ingredient_profile",
            "nll_runtime",
            "paired_reference_profile",
            "paired_source_record_ids",
            "reference_profile",
        ),
        label="reference statistics",
    )
    expected_ingredient_profile = canonical_prompt_ingredient_profile(
        build_prompt_ingredient_profile(tuple(item.record for item in prompt_records))
    )
    if statistics["ingredient_profile"] != expected_ingredient_profile:
        raise Phase1SemanticError(
            "prompt ingredient profile is stale or contradicts prompt evidence"
        )
    if statistics["paired_source_record_ids"] != list(paired_ids):
        raise Phase1SemanticError("paired source record IDs contradict neutral briefs")
    expected_reference_profile = _reference_profile_record(
        build_reference_profile(observations)
    )
    expected_paired_profile = _reference_profile_record(
        build_reference_profile(paired_observations)
    )
    if statistics["reference_profile"] != expected_reference_profile:
        raise Phase1SemanticError(
            "reference profile is stale or contradicts serialized observations"
        )
    if statistics["paired_reference_profile"] != expected_paired_profile:
        raise Phase1SemanticError(
            "paired reference profile is stale or contradicts serialized observations"
        )


def _decode_neutral_brief(record: JsonObject, index: int) -> NeutralStoryBrief:
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
    try:
        return NeutralStoryBrief(
            brief_id=_text(record["brief_id"], "neutral brief ID"),
            source_record_id=_text(
                record["source_record_id"], "neutral brief source record ID"
            ),
            prompt_text=_text(record["prompt_text"], "neutral brief prompt"),
            required_words=_text_tuple(
                record["required_words"], "neutral brief required words"
            ),
            requested_features=_text_tuple(
                record["requested_features"],
                "neutral brief requested features",
                allow_empty=True,
            ),
            matched_reference_text=_text(
                record["matched_reference_text"], "neutral brief matched reference"
            ),
        )
    except (TypeError, ValueError) as error:
        raise Phase1SemanticError(f"invalid neutral brief {index}: {error}") from error


def _decode_prompt_metadata(record: JsonObject, index: int) -> _PromptMetadataEvidence:
    require_exact_fields(
        record,
        (
            "content_sha256",
            "features",
            "normalized_story_sha256",
            "prompt",
            "record_id",
            "source",
            "source_index",
            "source_member",
            "story",
            "summary",
            "words",
        ),
        label=f"prompt metadata {index}",
    )
    content_sha256 = _sha256_text(
        record["content_sha256"], "prompt content SHA-256"
    )
    normalized_sha256 = _sha256_text(
        record["normalized_story_sha256"], "prompt normalized story SHA-256"
    )
    story = _text(record["story"], "prompt source story")
    summary = record["summary"]
    if type(summary) is not str:
        raise Phase1SemanticError("prompt source summary must be text")
    if normalized_sha256 != normalized_story_sha256(story):
        raise Phase1SemanticError("prompt normalized story SHA-256 is stale")
    source_member = _text(record["source_member"], "prompt source member")
    source_index = _integer(record["source_index"], "prompt source index")
    record_id = _text(record["record_id"], "prompt record ID")
    if record_id != f"archive:{source_member}:{source_index}:{content_sha256}":
        raise Phase1SemanticError("prompt record ID does not bind source provenance")
    if record["source"] != "GPT-4":
        raise Phase1SemanticError("prompt metadata is not from the GPT-4 cohort")
    words = _text_tuple(record["words"], "prompt words")
    features = _text_tuple(
        record["features"], "prompt features", allow_empty=True
    )
    prompt = _text(record["prompt"], "prompt text")
    expected_content_sha256 = json_sha256(
        {
            "instruction": {
                "features": list(features),
                "prompt:": prompt,
                "words": list(words),
            },
            "source": "GPT-4",
            "story": story,
            "summary": summary,
        }
    )
    if content_sha256 != expected_content_sha256:
        raise Phase1SemanticError("prompt content SHA-256 is stale")
    try:
        source_record = ArchiveSourceRecord(
            record_id=record_id,
            source_member=source_member,
            source_index=source_index,
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
    except (TypeError, ValueError) as error:
        raise Phase1SemanticError(f"invalid prompt metadata {index}: {error}") from error
    return _PromptMetadataEvidence(source_record, normalized_sha256)


def _decode_reference_record(record: JsonObject, index: int) -> ReferenceRecord:
    require_exact_fields(
        record,
        ("prompt_text", "record_id", "source_model", "story_text"),
        label=f"reference story {index}",
    )
    prompt = record["prompt_text"]
    if prompt is not None and (type(prompt) is not str or not prompt.strip()):
        raise Phase1SemanticError("reference prompt must be null or nonempty text")
    try:
        return ReferenceRecord(
            record_id=_text(record["record_id"], "reference record ID"),
            story_text=_text(record["story_text"], "reference story text"),
            prompt_text=cast(str | None, prompt),
            source_model=_text(record["source_model"], "reference source model"),
        )
    except (TypeError, ValueError) as error:
        raise Phase1SemanticError(f"invalid reference story {index}: {error}") from error


def _decode_reference_annotation(
    record: JsonObject,
    index: int,
) -> ReferenceAnnotation:
    require_exact_fields(
        record,
        ("feature_labels", "record_id", "required_words", "source_partition"),
        label=f"reference annotation {index}",
    )
    try:
        return ReferenceAnnotation(
            record_id=_text(record["record_id"], "annotation record ID"),
            source_partition=_text(
                record["source_partition"], "annotation source partition"
            ),
            required_words=_text_tuple(
                record["required_words"],
                "annotation required words",
                allow_empty=True,
            ),
            feature_labels=_text_tuple(
                record["feature_labels"],
                "annotation feature labels",
                allow_empty=True,
            ),
        )
    except (TypeError, ValueError) as error:
        raise Phase1SemanticError(
            f"invalid reference annotation {index}: {error}"
        ) from error


def _decode_reference_observation(
    record: JsonObject,
    index: int,
) -> ReferenceObservation:
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
    dialogue = record["dialogue_present"]
    if type(dialogue) is not bool:
        raise Phase1SemanticError("reference dialogue flag must be boolean")
    normalized_nll = record["normalized_nll"]
    repeated = record["repeated_ngram_fraction"]
    if type(normalized_nll) is not float or type(repeated) is not float:
        raise Phase1SemanticError("reference NLL and repetition values must be floats")
    try:
        return ReferenceObservation(
            record_id=_text(record["record_id"], "observation record ID"),
            word_tokens=_text_tuple(record["word_tokens"], "observation words"),
            model_token_ids=_integer_tuple(
                record["model_token_ids"], "observation model tokens"
            ),
            sentence_word_counts=_positive_integer_tuple(
                record["sentence_word_counts"], "observation sentence lengths"
            ),
            paragraph_count=_positive_integer(
                record["paragraph_count"], "observation paragraph count"
            ),
            dialogue_present=dialogue,
            opening_key=_text(record["opening_key"], "observation opening"),
            ending_key=_text(record["ending_key"], "observation ending"),
            feature_labels=_text_tuple(
                record["requested_feature_labels"],
                "observation requested features",
                allow_empty=True,
            ),
            normalized_nll=normalized_nll,
            required_words=_text_tuple(
                record["required_words"],
                "observation required words",
                allow_empty=True,
            ),
            realized_feature_labels=_text_tuple(
                record["realized_feature_labels"],
                "observation realized features",
                allow_empty=True,
            ),
            repeated_ngram_fraction=repeated,
        )
    except (TypeError, ValueError) as error:
        raise Phase1SemanticError(
            f"invalid reference observation {index}: {error}"
        ) from error


def _decode_validation_record(record: JsonObject, index: int) -> ValidationStoryRecord:
    require_exact_fields(
        record,
        (
            "content_sha256",
            "normalized_story_sha256",
            "record_id",
            "source_index",
            "story",
        ),
        label=f"validation source record {index}",
    )
    story = _text(record["story"], "validation story")
    content_sha256 = _sha256_text(
        record["content_sha256"], "validation content SHA-256"
    )
    normalized_sha256 = _sha256_text(
        record["normalized_story_sha256"], "validation normalized story SHA-256"
    )
    if normalized_sha256 != normalized_story_sha256(story):
        raise Phase1SemanticError("validation normalized story SHA-256 is stale")
    try:
        return ValidationStoryRecord(
            record_id=_text(record["record_id"], "validation record ID"),
            source_index=_integer(record["source_index"], "validation source index"),
            content_sha256=content_sha256,
            story=story,
        )
    except (TypeError, ValueError) as error:
        raise Phase1SemanticError(
            f"invalid validation source record {index}: {error}"
        ) from error


def _validate_observation_derivations(
    records: tuple[ReferenceRecord, ...],
    annotations: tuple[ReferenceAnnotation, ...],
    observations: tuple[ReferenceObservation, ...],
    *,
    label: str,
) -> None:
    for record, annotation, observation in zip(
        records, annotations, observations, strict=True
    ):
        expected = observe_reference(
            record,
            model_token_ids=observation.model_token_ids,
            normalized_nll=observation.normalized_nll,
            feature_labels=annotation.feature_labels,
            required_words=annotation.required_words,
        )
        if canonical_reference_observation(expected) != canonical_reference_observation(
            observation
        ):
            raise Phase1SemanticError(
                f"{label} observation is stale or contradicts story/annotation evidence"
            )


def _require_disjoint_source_cohorts(
    cohorts: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...],
) -> None:
    for label, record_ids, story_hashes in cohorts:
        if len(record_ids) != len(story_hashes):
            raise Phase1SemanticError(f"{label} IDs and story identities do not align")
        if len(set(record_ids)) != len(record_ids):
            raise Phase1SemanticError(f"{label} contains duplicate source IDs")
        if len(set(story_hashes)) != len(story_hashes):
            raise Phase1SemanticError(
                f"{label} is not unique by normalized story content"
            )
    for left in range(len(cohorts)):
        left_label, left_ids, left_hashes = cohorts[left]
        for right in range(left + 1, len(cohorts)):
            right_label, right_ids, right_hashes = cohorts[right]
            if set(left_ids).intersection(right_ids):
                raise Phase1SemanticError(
                    f"{left_label} and {right_label} reuse source record IDs"
                )
            if set(left_hashes).intersection(right_hashes):
                raise Phase1SemanticError(
                    f"{left_label} and {right_label} leak normalized story content"
                )


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
        "required_word_frequencies": [
            list(item) for item in profile.required_word_frequencies
        ],
        "sentence_word_counts": list(profile.sentence_word_counts),
        "story_word_counts": list(profile.story_word_counts),
        "token_probabilities": [list(item) for item in profile.token_probabilities],
        "vocabulary": sorted(profile.vocabulary),
        "word_frequencies": [list(item) for item in profile.word_frequencies],
    }


def _require_archive_record_id(value: str, label: str) -> None:
    if _ARCHIVE_RECORD_ID.fullmatch(value) is None:
        raise Phase1SemanticError(f"{label} does not have archive provenance")


def _sha256_text(value: JsonValue, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise Phase1SemanticError(f"{label} must be lowercase SHA-256")
    return value


def _text_tuple(
    value: JsonValue,
    label: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if type(value) is not list or any(
        type(item) is not str or not item.strip() for item in value
    ):
        raise Phase1SemanticError(f"{label} must contain nonempty strings")
    if not allow_empty and not value:
        raise Phase1SemanticError(f"{label} must not be empty")
    return tuple(value)


def _integer_tuple(value: JsonValue, label: str) -> tuple[int, ...]:
    if type(value) is not list or not value or any(
        type(item) is not int or item < 0 for item in value
    ):
        raise Phase1SemanticError(f"{label} must contain nonnegative integers")
    return tuple(value)


def _positive_integer_tuple(value: JsonValue, label: str) -> tuple[int, ...]:
    if type(value) is not list or not value or any(
        type(item) is not int or item <= 0 for item in value
    ):
        raise Phase1SemanticError(f"{label} must contain positive integers")
    return tuple(value)


def _positive_integer(value: JsonValue, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise Phase1SemanticError(f"{label} must be a positive integer")
    return value


def _validate_configuration(record: JsonObject, root: Path) -> None:
    expected_scalars = {
        "audit_count": _AUDIT_COUNT,
        "audit_seed": _AUDIT_SEED,
        "full_count": _FULL_COUNT,
        "generation_workers": 8,
        "hard_cap_usd": "15.00",
        "projected_accepted_story_count": _PROJECTED_ACCEPTED,
        "retry_allowance_basis_points": 10_000,
        "retry_max_attempts": 2,
        "screen_count": _SCREEN_COUNT,
        "selection_seed": _SELECTION_SEED,
        "surface_measurement_version": "tinyworlds-v2-surface-measurements-v2",
        "surface_worker_count": REFERENCE_SURFACE_WORKERS,
        "validator_version": "tinyworlds-v2-deterministic-story-validator-v3",
        "version": _VERSION,
    }
    for key, expected in expected_scalars.items():
        if record.get(key) != expected:
            raise Phase1SemanticError(f"configuration {key} differs from the preset")
    expected_models = [_model_record(model) for model in CANDIDATE_MODELS]
    if record.get("candidate_models") != expected_models:
        raise Phase1SemanticError("configuration candidate model table differs")
    if record.get("verifier_model") != _model_record(VERIFIER_MODEL):
        raise Phase1SemanticError("configuration verifier model differs")
    statistics = _json_object(root / "reference_statistics.json")
    if record.get("reference_statistics_sha256") != json_sha256(statistics):
        raise Phase1SemanticError("configuration does not bind reference statistics")
    routes = _json_object(root / "catalog" / "routes.json")
    if record.get("catalog_snapshot_sha256") != routes.get("snapshot_sha256"):
        raise Phase1SemanticError("configuration does not bind the route catalog")


def _load_quality_details(
    path: Path,
) -> tuple[tuple[RouteQualityReport, ...], tuple[RouteQualityReport, ...], QualitySelection | None]:
    record = _json_object(path)
    require_exact_fields(
        record,
        ("full_reports", "screen_reports", "selection"),
        label="quality details",
    )
    screen = tuple(_quality_report(value, index) for index, value in enumerate(_list(record["screen_reports"], "screen reports")))
    full = tuple(_quality_report(value, index) for index, value in enumerate(_list(record["full_reports"], "full reports")))
    selection_value = record["selection"]
    selection = None if selection_value is None else _quality_selection(selection_value, "quality selection")
    return screen, full, selection


def _validate_screen_selection(
    root: Path,
    status: str,
    reports: tuple[RouteQualityReport, ...],
    details_selection: QualitySelection | None,
    *,
    has_full_reports: bool,
) -> tuple[str, ...]:
    decision_path = root / "finalist_decision.json"
    if not reports:
        if decision_path.exists() or details_selection is not None:
            raise Phase1SemanticError("preflight-stopped artifact cannot select finalists")
        if status not in {"blocked_by_cost_cap", *_INTERRUPTED_STATUSES}:
            raise Phase1SemanticError("only a cost stop may omit screen reports")
        return ()
    expected = select_screen_finalists(reports)
    if not decision_path.is_file():
        raise Phase1SemanticError("screen reports require finalist_decision.json")
    stored = _quality_selection(_json_object(decision_path), "finalist decision")
    if stored != expected or (
        not has_full_reports and details_selection not in (stored, None)
    ):
        raise Phase1SemanticError("stored finalist decision differs from fixed policy")
    return stored.route_ids


def _validate_full_selection(
    root: Path,
    status: str,
    finalist: tuple[str, ...],
    reports: tuple[RouteQualityReport, ...],
) -> tuple[str, ...]:
    comparison = _json_object(root / "quality_comparisons.json")
    require_exact_fields(
        comparison,
        ("audited_route_ids", "qualified_route_ids"),
        label="quality comparisons",
    )
    audited = _string_tuple_allow_empty(
        comparison["audited_route_ids"], "audited routes"
    )
    qualified = _string_tuple(comparison["qualified_route_ids"], "qualified routes")
    if audited != finalist:
        raise Phase1SemanticError("audited route IDs must equal screen finalists")
    if not reports:
        if qualified:
            raise Phase1SemanticError("qualified routes require full reports")
        return ()
    expected = select_full_quality_routes(reports, finalist_order=finalist)
    details = _json_object(root / "quality_details.json")
    stored = _quality_selection(details["selection"], "full quality selection")
    if stored != expected or qualified != expected.route_ids:
        raise Phase1SemanticError("qualified routes differ from full quality reports")
    passing = tuple(report.route_id for report in reports if report.passed)
    if any(route_id not in passing for route_id in qualified):
        raise Phase1SemanticError("a qualified route did not pass all reported gates")
    if status in {
        "awaiting_human_audit",
        "audit_insufficient_accepted_samples",
    } and not qualified:
        raise Phase1SemanticError(
            "an audit-stage artifact requires an automated-qualified route"
        )
    if status == "no_quality_qualified_route" and qualified:
        raise Phase1SemanticError("stopped quality result cannot contain qualified routes")
    return qualified


def _validate_result_counts_and_splits(
    root: Path,
    status: str,
    finalist: tuple[str, ...],
    generated: tuple[JsonObject, ...],
    verifier: tuple[JsonObject, ...],
) -> None:
    expected_generated = 0 if status == "blocked_by_cost_cap" else (
        len(CANDIDATE_MODELS) * _SCREEN_COUNT
        + len(finalist) * (_FULL_COUNT - _SCREEN_COUNT)
    )
    if status in _INTERRUPTED_STATUSES:
        if len(generated) > expected_generated:
            raise Phase1SemanticError("interrupted generator count exceeds the funnel")
    elif len(generated) != expected_generated:
        raise Phase1SemanticError("generator result count differs from the funnel")
    for model in CANDIDATE_MODELS:
        planned = _jsonl_objects(root / "routes" / model.route_id / "requests.jsonl")
        if len(planned) != _FULL_COUNT:
            raise Phase1SemanticError(f"planned request count differs for {model.route_id}")
        planned_hashes = tuple(
            _text(item.get("request_sha256"), "planned request_sha256")
            for item in planned
        )
        if len(set(planned_hashes)) != _FULL_COUNT:
            raise Phase1SemanticError(f"planned request hashes repeat for {model.route_id}")
        plan = _json_object(root / "routes" / model.route_id / "plan.json")
        if plan.get("planned_request_count") != _FULL_COUNT or tuple(
            _list(plan.get("planned_request_sha256"), "planned request hashes")
        ) != planned_hashes:
            raise Phase1SemanticError(f"route plan request hashes differ for {model.route_id}")
        route_records = tuple(item for item in generated if item.get("route_id") == model.route_id)
        accepted = _jsonl_objects(root / "routes" / model.route_id / "accepted.jsonl")
        rejected = _jsonl_objects(root / "routes" / model.route_id / "rejected.jsonl")
        raw = _jsonl_objects(root / "routes" / model.route_id / "raw_responses.jsonl")
        if accepted != tuple(item for item in route_records if _accepted(item)):
            raise Phase1SemanticError(f"accepted split differs for {model.route_id}")
        if rejected != tuple(item for item in route_records if not _accepted(item)):
            raise Phase1SemanticError(f"rejected split differs for {model.route_id}")
        if len(raw) != len(route_records):
            raise Phase1SemanticError(f"raw response count differs for {model.route_id}")
        manifest = _json_object(root / "routes" / model.route_id / "manifest.json")
        if manifest.get("accepted_count") != len(accepted) or manifest.get("rejected_count") != len(rejected):
            raise Phase1SemanticError(f"execution manifest counts differ for {model.route_id}")
        submitted_count = manifest.get("submitted_request_count")
        interrupted_count = manifest.get("interrupted_count")
        if (
            type(submitted_count) is not int
            or type(interrupted_count) is not int
            or submitted_count != len(route_records) + interrupted_count
        ):
            raise Phase1SemanticError(f"execution manifest submissions differ for {model.route_id}")
        if manifest.get("planned_request_count") != _FULL_COUNT or tuple(
            _list(manifest.get("planned_request_sha256"), "manifest planned hashes")
        ) != planned_hashes:
            raise Phase1SemanticError(f"execution manifest plan differs for {model.route_id}")
        request_hashes = tuple(_text(item.get("request_sha256"), "request_sha256") for item in route_records)
        manifest_requests = _list(manifest.get("requests"), "execution requests")
        completed_manifest_hashes = tuple(
            item.get("request_sha256")
            for item in manifest_requests
            if type(item) is dict and item.get("result_sha256") is not None
        )
        if completed_manifest_hashes != request_hashes:
            raise Phase1SemanticError(f"execution manifest request hashes differ for {model.route_id}")
    if verifier and status not in _INTERRUPTED_STATUSES:
        expected_verifier = _FULL_COUNT + sum(
            _accepted(item)
            for item in generated
            if item.get("route_id") in finalist
        )
        if len(verifier) != expected_verifier:
            raise Phase1SemanticError("verifier result count differs from eligible stories")
    elif (
        finalist
        and status not in {"no_quality_qualified_route", *_INTERRUPTED_STATUSES}
    ):
        raise Phase1SemanticError("full funnel is missing verifier evidence")
    verifier_manifest = _json_object(root / "verifier" / "manifest.json")
    verifier_submitted = verifier_manifest.get("submitted_request_count")
    verifier_interrupted = verifier_manifest.get("interrupted_count")
    if (
        type(verifier_submitted) is not int
        or type(verifier_interrupted) is not int
        or verifier_submitted != len(verifier) + verifier_interrupted
    ):
        raise Phase1SemanticError("verifier execution manifest count differs")


def _validate_audit(
    root: Path,
    status_record: JsonObject,
    status: str,
    finalist: tuple[str, ...],
    generated: tuple[JsonObject, ...],
) -> None:
    packet_path, key_path = root / "audit_packet.json", root / "audit_key.json"
    if status != "awaiting_human_audit":
        if packet_path.exists() or key_path.exists() or status_record["audit_sha256"] is not None:
            raise Phase1SemanticError("stopped artifact cannot contain an audit")
        return
    packet, key = decode_audit_pair(packet_path.read_bytes(), key_path.read_bytes())
    if status_record["audit_sha256"] != packet.audit_sha256:
        raise Phase1SemanticError("status audit digest differs from packet")
    reference = tuple(item for item in key.entries if item.source_kind is AuditSourceKind.REFERENCE)
    generated_entries = tuple(item for item in key.entries if item.source_kind is AuditSourceKind.GENERATED)
    if len(reference) != _AUDIT_COUNT or len(generated_entries) != _AUDIT_COUNT:
        raise Phase1SemanticError("audit must contain exactly 100 reference and generated items")
    eligible_generated = {
        _text(item.get("sample_id"), "sample_id"): item
        for item in generated
        if item.get("route_id") in finalist and _accepted(item)
    }
    if any(entry.source_id not in eligible_generated for entry in generated_entries):
        raise Phase1SemanticError("audit contains a non-finalist or rejected generated story")
    if any(entry.route_id not in finalist for entry in generated_entries):
        raise Phase1SemanticError("audit key route is outside the screen finalists")
    route_counts = {
        route_id: sum(entry.route_id == route_id for entry in generated_entries)
        for route_id in finalist
    }
    if set(entry.route_id for entry in generated_entries) != set(finalist):
        raise Phase1SemanticError("audit generated routes must equal screen finalists")
    if max(route_counts.values()) - min(route_counts.values()) > 1:
        raise Phase1SemanticError("audit generated items are not route-balanced")
    if len(packet.items) != 2 * _AUDIT_COUNT or len(key.entries) != len(packet.items):
        raise Phase1SemanticError("audit packet/key composition count differs")


def _validate_audit_feasibility_stop(
    root: Path,
    status: str,
    finalist: tuple[str, ...],
    generated: tuple[JsonObject, ...],
) -> None:
    path = root / "audit_feasibility.json"
    if status != "audit_insufficient_accepted_samples":
        if path.exists():
            raise Phase1SemanticError("audit feasibility stop evidence has wrong status")
        return
    record = _json_object(path)
    require_exact_fields(
        record,
        (
            "audited_route_ids",
            "failure_reason",
            "generated_audit_count",
            "routes",
            "union_eligible_pair_ids",
        ),
        label="audit feasibility",
    )
    if not finalist:
        raise Phase1SemanticError(
            "audit feasibility stop requires at least one screen finalist"
        )
    audited = _string_tuple(record["audited_route_ids"], "audit feasibility routes")
    if audited != finalist or record["generated_audit_count"] != _AUDIT_COUNT:
        raise Phase1SemanticError("audit feasibility scope/count differs")
    if record["failure_reason"] != "no_distinct_balanced_assignment":
        raise Phase1SemanticError("audit feasibility failure reason differs")
    quotient, remainder = divmod(_AUDIT_COUNT, len(finalist))
    expected_routes = []
    union: set[str] = set()
    for index, route_id in enumerate(finalist):
        pair_ids = tuple(
            sorted(
                _text(item.get("brief_id"), "generated brief_id")
                for item in generated
                if item.get("route_id") == route_id and _accepted(item)
            )
        )
        union.update(pair_ids)
        expected_routes.append(
            {
                "eligible_pair_ids": list(pair_ids),
                "required_count": quotient + (index < remainder),
                "route_id": route_id,
            }
        )
    if record["routes"] != expected_routes or record["union_eligible_pair_ids"] != sorted(union):
        raise Phase1SemanticError("audit feasibility evidence differs from results")
    if _balanced_assignment_exists(expected_routes):
        raise Phase1SemanticError(
            "audit feasibility stop claims failure for an allocatable cohort"
        )


def _balanced_assignment_exists(routes: list[JsonObject]) -> bool:
    """Return whether route quotas admit globally distinct eligible pair IDs."""
    candidates = {
        _text(route["route_id"], "audit feasibility route_id"): tuple(
            _text(pair_id, "audit feasibility pair_id")
            for pair_id in _list(
                route["eligible_pair_ids"],
                "audit feasibility eligible pairs",
            )
        )
        for route in routes
    }
    slots = tuple(
        (route_id, slot_index)
        for route in routes
        for route_id in (_text(route["route_id"], "audit feasibility route_id"),)
        for slot_index in range(_integer(route["required_count"], "required count"))
    )
    pair_to_slot: dict[str, tuple[str, int]] = {}

    def augment(slot: tuple[str, int], visited: set[str]) -> bool:
        for pair_id in candidates[slot[0]]:
            if pair_id in visited:
                continue
            visited.add(pair_id)
            occupied = pair_to_slot.get(pair_id)
            if occupied is None or augment(occupied, visited):
                pair_to_slot[pair_id] = slot
                return True
        return False

    return all(augment(slot, set()) for slot in slots)


def _quality_report(value: JsonValue, index: int) -> RouteQualityReport:
    record = _object(value, f"quality report {index}")
    expected = tuple(field.name for field in fields(RouteQualityReport)) + ("passed",)
    require_exact_fields(record, expected, label=f"quality report {index}")
    values = {key: item for key, item in record.items() if key != "passed"}
    try:
        values["phase"] = QualityPhase(_text(values["phase"], "quality phase"))
        values["sample_ids"] = _string_tuple(values["sample_ids"], "sample_ids")
        values["failures"] = _string_tuple(values["failures"], "failures")
        for field in fields(RouteQualityReport):
            if values[field.name] is None:
                values[field.name] = math.inf
        report = RouteQualityReport(**cast(dict, values))
        validate_route_quality_report(report)
    except (TypeError, ValueError) as error:
        raise Phase1SemanticError(f"invalid quality report {index}: {error}") from error
    if record["passed"] is not report.passed:
        raise Phase1SemanticError("quality report passed flag contradicts failures")
    return report


def _quality_selection(value: JsonValue, label: str) -> QualitySelection:
    record = _object(value, label)
    require_exact_fields(record, ("outcome", "reason", "route_ids"), label=label)
    try:
        return QualitySelection(
            QualityOutcome(_text(record["outcome"], f"{label} outcome")),
            _string_tuple(record["route_ids"], f"{label} route_ids"),
            _text(record["reason"], f"{label} reason"),
        )
    except ValueError as error:
        raise Phase1SemanticError(str(error)) from error


def _model_record(model: object) -> JsonObject:
    return {
        "canonical_slug": model.canonical_slug,
        "first_party_provider_slug": model.first_party_provider_slug,
        "max_token_parameter": model.max_token_parameter,
        "plan_completion_usd_per_million": model.plan_completion_usd_per_million,
        "plan_prompt_usd_per_million": model.plan_prompt_usd_per_million,
        "request_model_id": model.request_model_id,
        "route_id": model.route_id,
    }


def _accepted(record: JsonObject) -> bool:
    validation = _object(record.get("validation"), "generated validation")
    accepted = validation.get("accepted")
    if type(accepted) is not bool:
        raise Phase1SemanticError("generated accepted flag must be boolean")
    return accepted


def _object(value: JsonValue, label: str) -> JsonObject:
    if type(value) is not dict:
        raise Phase1SemanticError(f"{label} must be an object")
    return value


def _list(value: JsonValue, label: str) -> list[JsonValue]:
    if type(value) is not list:
        raise Phase1SemanticError(f"{label} must be a list")
    return value


def _text(value: JsonValue, label: str) -> str:
    if type(value) is not str or not value:
        raise Phase1SemanticError(f"{label} must be nonempty text")
    return value


def _integer(value: JsonValue, label: str) -> int:
    if type(value) is not int or value < 0:
        raise Phase1SemanticError(f"{label} must be a nonnegative integer")
    return value


def _string_tuple(value: JsonValue, label: str) -> tuple[str, ...]:
    if type(value) is not list or any(type(item) is not str or not item for item in value):
        raise Phase1SemanticError(f"{label} must be a list of nonempty strings")
    result = tuple(value)
    if len(set(result)) != len(result):
        raise Phase1SemanticError(f"{label} must contain unique strings")
    return result


def _string_tuple_allow_empty(value: JsonValue, label: str) -> tuple[str, ...]:
    if type(value) is not list or any(type(item) is not str or not item for item in value):
        raise Phase1SemanticError(f"{label} must be a list of nonempty strings")
    result = tuple(value)
    if len(set(result)) != len(result):
        raise Phase1SemanticError(f"{label} must contain unique strings")
    return result


__all__ = [
    "Phase1SemanticError",
    "Phase1SemanticValidation",
    "validate_phase1_semantics",
]

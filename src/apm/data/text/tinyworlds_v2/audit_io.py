"""Strict persistence and fail-closed approval for TinyWorlds-v2 audits."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
from tempfile import TemporaryDirectory
from typing import Callable, TypeVar

from apm.data.text.tinyworlds_v2.audit import (
    AuditApproval,
    AuditDecision,
    AuditDecisionSet,
    AuditEvaluation,
    AuditKeyEntry,
    AuditSourceGuess,
    AuditSourceKind,
    BlindedAuditItem,
    BlindedAuditKey,
    BlindedAuditPacket,
    build_decision_set,
    evaluate_blinded_audit,
    select_human_approved_route,
    validate_audit_approval,
    validate_audit_pair,
)
from apm.data.text.tinyworlds_v2.json_contracts import (
    CanonicalJsonError,
    JsonObject,
    JsonValue,
    canonical_json_bytes,
    canonical_json_loads,
    require_exact_fields,
    require_json_object,
)
from apm.data.text.tinyworlds_v2.phase1_artifacts import (
    Phase1ArtifactFile,
    Phase1ArtifactManifest,
    load_phase1_artifact_tree,
)


REFERENCE_ROOT = Path("data/tinyworlds-v2/reference")
AUDIT_PACKET_FILENAME = "audit_packet.json"
AUDIT_KEY_FILENAME = "audit_key.json"
AUDIT_DECISIONS_FILENAME = "audit_decisions.json"
AUDIT_APPROVAL_REQUEST_FILENAME = "audit_approval_request.json"
AUDIT_APPROVAL_FILENAME = "audit_approval.json"
QUALITY_COMPARISONS_FILENAME = "quality_comparisons.json"
COST_ACTUALS_FILENAME = "cost_actuals.json"

_OVERLAY_FILENAMES = frozenset(
    (
        AUDIT_DECISIONS_FILENAME,
        AUDIT_APPROVAL_REQUEST_FILENAME,
        AUDIT_APPROVAL_FILENAME,
    )
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_UTC_TIMESTAMP = re.compile(
    r"(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-"
    r"(?:0[1-9]|[12]\d|3[01])T(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\dZ"
)
_Result = TypeVar("_Result")


class AuditIoError(ValueError):
    """An audit artifact is malformed, noncanonical, or inconsistent."""


@dataclass(frozen=True, slots=True)
class QualityAuditScope:
    """Screen-finalist audit scope and its automated-qualified subset."""

    audited_route_ids: tuple[str, ...]
    qualified_route_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for label, route_ids in (
            ("audited", self.audited_route_ids),
            ("qualified", self.qualified_route_ids),
        ):
            if (
                type(route_ids) is not tuple
                or not route_ids
                or any(type(route_id) is not str or not route_id for route_id in route_ids)
                or len(route_ids) != len(set(route_ids))
            ):
                raise AuditIoError(f"{label} route IDs must be nonempty and unique")
        if not set(self.qualified_route_ids).issubset(self.audited_route_ids):
            raise AuditIoError("qualified route IDs must be a subset of audited routes")
        if tuple(
            route_id
            for route_id in self.audited_route_ids
            if route_id in self.qualified_route_ids
        ) != self.qualified_route_ids:
            raise AuditIoError("qualified route order must preserve audited route order")


@dataclass(frozen=True, slots=True)
class AuditDecisionExport:
    """Browser-exported decisions before their computed digest is attached."""

    audit_sha256: str
    decisions: tuple[AuditDecision, ...]

    def __post_init__(self) -> None:
        _require_sha256(self.audit_sha256, "audit decision export digest")
        if type(self.decisions) is not tuple or any(
            type(decision) is not AuditDecision for decision in self.decisions
        ):
            raise AuditIoError("audit decision export must contain decisions")
        if len({decision.item_id for decision in self.decisions}) != len(
            self.decisions
        ):
            raise AuditIoError("audit decision export item IDs must be unique")


@dataclass(frozen=True, slots=True)
class AuditApprovalArtifact:
    """The explicit approval, computed evaluation, and selected route."""

    approval: AuditApproval
    evaluation: AuditEvaluation
    selected_route_id: str

    def __post_init__(self) -> None:
        if type(self.approval) is not AuditApproval:
            raise TypeError("approval must be an AuditApproval")
        if type(self.evaluation) is not AuditEvaluation:
            raise TypeError("evaluation must be an AuditEvaluation")
        _require_text(self.selected_route_id, "selected route ID")
        if (
            self.approval.audit_sha256,
            self.approval.decision_sha256,
        ) != (
            self.evaluation.audit_sha256,
            self.evaluation.decision_sha256,
        ):
            raise AuditIoError("approval and evaluation digests differ")


@dataclass(frozen=True, slots=True)
class Phase1AuditValidation:
    """Strict validation result for the base tree and optional human overlay."""

    manifest: Phase1ArtifactManifest
    decision_set: AuditDecisionSet | None
    evaluation: AuditEvaluation | None
    approval_artifact: AuditApprovalArtifact | None


def validate_phase1_semantics(root: str | Path = REFERENCE_ROOT):
    """Run the complete cross-artifact validator, including zero-network replay."""
    # Lazy import prevents the semantic validator's audit decoders from forming
    # an import cycle while this module is initialized.
    from apm.data.text.tinyworlds_v2.phase1_semantics import (
        validate_phase1_semantics as validate,
    )

    return validate(root)


def encode_blinded_audit_packet(packet: BlindedAuditPacket) -> bytes:
    """Encode a blinded packet as canonical JSON bytes."""
    if type(packet) is not BlindedAuditPacket:
        raise TypeError("packet must be a BlindedAuditPacket")
    return canonical_json_bytes(_packet_record(packet))


def decode_blinded_audit_packet(payload: bytes) -> BlindedAuditPacket:
    """Strictly decode a blinded packet, rejecting unknown and noncanonical data."""
    record = _canonical_object(payload, "blinded audit packet")
    require_exact_fields(
        record,
        ("audit_sha256", "items"),
        label="blinded audit packet",
    )
    items = tuple(
        _decode_blinded_item(item, index)
        for index, item in enumerate(
            _require_list(record["items"], "blinded audit packet items")
        )
    )
    return _construct(
        "blinded audit packet",
        lambda: BlindedAuditPacket(
            items=items,
            audit_sha256=_require_sha256(
                record["audit_sha256"], "blinded audit packet digest"
            ),
        ),
    )


def encode_blinded_audit_key(key: BlindedAuditKey) -> bytes:
    """Encode a hidden source key as canonical JSON bytes."""
    if type(key) is not BlindedAuditKey:
        raise TypeError("key must be a BlindedAuditKey")
    return canonical_json_bytes(_key_record(key))


def decode_blinded_audit_key(payload: bytes) -> BlindedAuditKey:
    """Strictly decode a hidden source key."""
    record = _canonical_object(payload, "blinded audit key")
    require_exact_fields(
        record,
        ("audit_sha256", "entries"),
        label="blinded audit key",
    )
    entries = tuple(
        _decode_key_entry(item, index)
        for index, item in enumerate(
            _require_list(record["entries"], "blinded audit key entries")
        )
    )
    return _construct(
        "blinded audit key",
        lambda: BlindedAuditKey(
            entries=entries,
            audit_sha256=_require_sha256(
                record["audit_sha256"], "blinded audit key digest"
            ),
        ),
    )


def decode_audit_pair(
    packet_payload: bytes,
    key_payload: bytes,
) -> tuple[BlindedAuditPacket, BlindedAuditKey]:
    """Decode and authenticate a packet/key pair against their shared digest."""
    packet = decode_blinded_audit_packet(packet_payload)
    key = decode_blinded_audit_key(key_payload)
    _construct("blinded audit pair", lambda: validate_audit_pair(packet, key))
    return packet, key


def encode_audit_decision_export(export: AuditDecisionExport) -> bytes:
    """Encode browser-exported decisions in their canonical interchange form."""
    if type(export) is not AuditDecisionExport:
        raise TypeError("export must be an AuditDecisionExport")
    return canonical_json_bytes(_decision_export_record(export))


def decode_audit_decision_export(payload: bytes) -> AuditDecisionExport:
    """Strictly decode the canonical decision object exported by the audit UI."""
    record = _canonical_object(payload, "audit decision export")
    require_exact_fields(
        record,
        ("audit_sha256", "decisions"),
        label="audit decision export",
    )
    return _construct(
        "audit decision export",
        lambda: AuditDecisionExport(
            audit_sha256=_require_sha256(
                record["audit_sha256"], "audit decision export digest"
            ),
            decisions=tuple(
                _decode_decision(item, index)
                for index, item in enumerate(
                    _require_list(
                        record["decisions"], "audit decision export decisions"
                    )
                )
            ),
        ),
    )


def decision_set_from_export(
    packet: BlindedAuditPacket,
    export: AuditDecisionExport,
) -> AuditDecisionSet:
    """Bind a complete canonical browser export to the exact audit packet."""
    if export.audit_sha256 != packet.audit_sha256:
        raise AuditIoError("audit decision export belongs to another packet")
    return _construct(
        "audit decision export",
        lambda: build_decision_set(packet, export.decisions),
    )


def encode_audit_decision_set(decision_set: AuditDecisionSet) -> bytes:
    """Encode a digest-bound decision set as canonical JSON bytes."""
    if type(decision_set) is not AuditDecisionSet:
        raise TypeError("decision_set must be an AuditDecisionSet")
    return canonical_json_bytes(_decision_set_record(decision_set))


def decode_audit_decision_set(
    payload: bytes,
    packet: BlindedAuditPacket,
) -> AuditDecisionSet:
    """Decode a decision set and recompute its membership, order, and digest."""
    record = _canonical_object(payload, "audit decision set")
    require_exact_fields(
        record,
        ("audit_sha256", "decision_sha256", "decisions"),
        label="audit decision set",
    )
    export = AuditDecisionExport(
        audit_sha256=_require_sha256(
            record["audit_sha256"], "audit decision set audit digest"
        ),
        decisions=tuple(
            _decode_decision(item, index)
            for index, item in enumerate(
                _require_list(record["decisions"], "audit decision set decisions")
            )
        ),
    )
    expected = decision_set_from_export(packet, export)
    stored_digest = _require_sha256(
        record["decision_sha256"], "audit decision set digest"
    )
    if stored_digest != expected.decision_sha256:
        raise AuditIoError("audit decision set digest mismatch")
    return expected


def encode_audit_evaluation(evaluation: AuditEvaluation) -> bytes:
    """Encode computed human-gate metrics as canonical JSON bytes."""
    if type(evaluation) is not AuditEvaluation:
        raise TypeError("evaluation must be an AuditEvaluation")
    return canonical_json_bytes(_evaluation_record(evaluation))


def decode_audit_evaluation(payload: bytes) -> AuditEvaluation:
    """Strictly decode finite, range-checked human-gate metrics."""
    return _decode_evaluation_record(
        _canonical_object(payload, "audit evaluation"),
        "audit evaluation",
    )


def encode_audit_approval(approval: AuditApproval) -> bytes:
    """Encode an explicit digest-bound human approval as canonical JSON bytes."""
    if type(approval) is not AuditApproval:
        raise TypeError("approval must be an AuditApproval")
    _validate_approval_timestamp(approval.approved_at_utc)
    return canonical_json_bytes(_approval_record(approval))


def decode_audit_approval(payload: bytes) -> AuditApproval:
    """Strictly decode an explicit digest-bound human approval."""
    return _decode_approval_record(
        _canonical_object(payload, "audit approval"),
        "audit approval",
    )


def encode_audit_approval_artifact(artifact: AuditApprovalArtifact) -> bytes:
    """Encode the immutable approval, evaluation, and selected route record."""
    if type(artifact) is not AuditApprovalArtifact:
        raise TypeError("artifact must be an AuditApprovalArtifact")
    return canonical_json_bytes(_approval_artifact_record(artifact))


def decode_audit_approval_artifact(payload: bytes) -> AuditApprovalArtifact:
    """Strictly decode the final approval overlay."""
    record = _canonical_object(payload, "audit approval artifact")
    require_exact_fields(
        record,
        (
            "approved",
            "approved_at_utc",
            "approved_by",
            "audit_sha256",
            "decision_sha256",
            "evaluation",
            "selected_route_id",
        ),
        label="audit approval artifact",
    )
    approval = _decode_approval_record(
        {key: record[key] for key in _APPROVAL_FIELDS},
        "audit approval artifact approval",
    )
    return _construct(
        "audit approval artifact",
        lambda: AuditApprovalArtifact(
            approval=approval,
            evaluation=_decode_evaluation_record(
                _require_object(
                    record["evaluation"], "audit approval artifact evaluation"
                ),
                "audit approval artifact evaluation",
            ),
            selected_route_id=_require_text(
                record["selected_route_id"], "selected route ID"
            ),
        ),
    )


_APPROVAL_FIELDS = (
    "approved",
    "approved_at_utc",
    "approved_by",
    "audit_sha256",
    "decision_sha256",
)


def approve_phase1_audit(root: str | Path = REFERENCE_ROOT) -> AuditApprovalArtifact:
    """Run the full scientific/replay gate, then record explicit consent."""
    directory = Path(root)
    # The approval is a human overlay and therefore cannot repair or bless a
    # self-consistent but scientifically contradictory base artifact.
    validate_phase1_semantics(directory)
    return _approve_phase1_audit_after_semantic_validation(directory)


def _approve_phase1_audit_after_semantic_validation(
    root: str | Path,
) -> AuditApprovalArtifact:
    """Approval mechanics for callers that already ran the scientific gate."""
    directory = Path(root)
    output_path = directory / AUDIT_APPROVAL_FILENAME
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"audit approval already exists: {output_path}")
    validate_phase1_tree_with_human_overlays(directory)
    packet, key = _load_audit_pair(directory)
    decision_set = decision_set_from_export(
        packet,
        decode_audit_decision_export(
            _read_required(directory / AUDIT_DECISIONS_FILENAME)
        ),
    )
    audit_scope = decode_quality_audit_scope(
        _read_required(directory / QUALITY_COMPARISONS_FILENAME)
    )
    _validate_audit_scope(key, audit_scope)
    qualified_route_ids = audit_scope.qualified_route_ids
    projected_costs = decode_projected_route_costs(
        _read_required(directory / COST_ACTUALS_FILENAME)
    )
    evaluation = _construct(
        "audit evaluation",
        lambda: evaluate_blinded_audit(
            packet,
            key,
            decision_set,
            selectable_route_ids=qualified_route_ids,
        ),
    )
    selected_route_id = _construct(
        "human-approved route selection",
        lambda: select_human_approved_route(
            evaluation,
            qualified_route_ids=qualified_route_ids,
            projected_costs=projected_costs,
        ),
    )
    approval = decode_audit_approval(
        _read_required(directory / AUDIT_APPROVAL_REQUEST_FILENAME)
    )
    _construct(
        "audit approval request",
        lambda: validate_audit_approval(
            packet,
            key,
            decision_set,
            evaluation,
            approval,
        ),
    )
    artifact = AuditApprovalArtifact(
        approval=approval,
        evaluation=evaluation,
        selected_route_id=selected_route_id,
    )
    with output_path.open("xb") as stream:
        stream.write(encode_audit_approval_artifact(artifact))
    return artifact


def validate_phase1_reference(
    root: str | Path = REFERENCE_ROOT,
) -> Phase1AuditValidation:
    """Run the full scientific/replay gate and validate every human overlay."""
    directory = Path(root)
    validate_phase1_semantics(directory)
    return _validate_phase1_reference_overlays(directory)


def _validate_phase1_reference_overlays(
    root: str | Path,
) -> Phase1AuditValidation:
    """Validate overlays after a caller has authenticated scientific semantics."""
    directory = Path(root)
    manifest = validate_phase1_tree_with_human_overlays(directory)
    packet, key = _load_audit_pair(directory)
    audit_scope = decode_quality_audit_scope(
        _read_required(directory / QUALITY_COMPARISONS_FILENAME)
    )
    _validate_audit_scope(key, audit_scope)
    qualified_route_ids = audit_scope.qualified_route_ids
    projected_costs = decode_projected_route_costs(
        _read_required(directory / COST_ACTUALS_FILENAME)
    )
    decisions_path = directory / AUDIT_DECISIONS_FILENAME
    request_path = directory / AUDIT_APPROVAL_REQUEST_FILENAME
    approval_path = directory / AUDIT_APPROVAL_FILENAME
    if not decisions_path.exists():
        if request_path.exists() or approval_path.exists():
            raise AuditIoError("approval overlays require audit_decisions.json")
        return Phase1AuditValidation(manifest, None, None, None)
    decision_set = decision_set_from_export(
        packet,
        decode_audit_decision_export(_read_required(decisions_path)),
    )
    evaluation = _construct(
        "audit evaluation",
        lambda: evaluate_blinded_audit(
            packet,
            key,
            decision_set,
            selectable_route_ids=qualified_route_ids,
        ),
    )
    if request_path.exists():
        request = decode_audit_approval(_read_required(request_path))
        expected = (packet.audit_sha256, decision_set.decision_sha256)
        if (request.audit_sha256, request.decision_sha256) != expected:
            raise AuditIoError("approval request digests do not match the evidence")
        if not request.approved:
            raise AuditIoError("approval request must explicitly set approved=true")
    if not approval_path.exists():
        return Phase1AuditValidation(manifest, decision_set, evaluation, None)
    if not request_path.exists():
        raise AuditIoError("audit approval requires its explicit approval request")
    artifact = decode_audit_approval_artifact(_read_required(approval_path))
    _construct(
        "audit approval artifact",
        lambda: validate_audit_approval(
            packet,
            key,
            decision_set,
            evaluation,
            artifact.approval,
        ),
    )
    if artifact.evaluation != evaluation:
        raise AuditIoError("stored audit evaluation does not match the decisions")
    selected_route = _construct(
        "human-approved route selection",
        lambda: select_human_approved_route(
            evaluation,
            qualified_route_ids=qualified_route_ids,
            projected_costs=projected_costs,
        ),
    )
    if artifact.selected_route_id != selected_route:
        raise AuditIoError("stored selected route does not match the fixed policy")
    return Phase1AuditValidation(manifest, decision_set, evaluation, artifact)


def validate_phase1_tree_with_human_overlays(
    root: str | Path,
) -> Phase1ArtifactManifest:
    """Authenticate a manifested base tree while allowing only three human files."""
    directory = Path(root)
    manifest = _decode_phase1_manifest(
        _read_required(directory / "manifest.json")
    )
    manifested_files = frozenset(artifact.path for artifact in manifest.artifacts)
    if manifested_files & _OVERLAY_FILENAMES:
        raise AuditIoError("human audit overlays cannot be part of the base manifest")
    expected_files = frozenset(("manifest.json", *manifested_files))
    present_files = _regular_relative_files(directory)
    expected_directories = frozenset(
        parent.as_posix()
        for relative_path in expected_files
        for parent in PurePosixPath(relative_path).parents
        if parent != PurePosixPath(".")
    )
    present_directories = frozenset(
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_dir() and not path.is_symlink()
    )
    if present_directories != expected_directories:
        raise AuditIoError("artifact tree has missing or unknown directories")
    extras = present_files - expected_files
    if not extras.issubset(_OVERLAY_FILENAMES):
        raise AuditIoError(
            f"artifact tree has unknown overlays: {tuple(sorted(extras - _OVERLAY_FILENAMES))}"
        )
    if expected_files - present_files:
        raise AuditIoError("artifact tree has missing manifested files")
    with TemporaryDirectory(prefix="tinyworlds-v2-validate-", dir=directory.parent) as name:
        view = Path(name)
        for relative_path in tuple(sorted(expected_files)):
            source = directory.joinpath(*PurePosixPath(relative_path).parts)
            destination = view.joinpath(*PurePosixPath(relative_path).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, destination)
            except OSError:
                shutil.copyfile(source, destination)
        return load_phase1_artifact_tree(view)


def decode_quality_audit_scope(payload: bytes) -> QualityAuditScope:
    """Load screen-finalist audit routes and the automated-qualified subset."""
    record = _canonical_object(payload, "quality comparisons")
    require_exact_fields(
        record,
        ("audited_route_ids", "qualified_route_ids"),
        label="quality comparisons",
    )
    def route_ids(field: str, label: str) -> tuple[str, ...]:
        return tuple(
            _require_text(item, f"{label} route {index}")
            for index, item in enumerate(
                _require_list(record[field], f"{label} route IDs")
            )
        )
    return QualityAuditScope(
        audited_route_ids=route_ids("audited_route_ids", "audited"),
        qualified_route_ids=route_ids("qualified_route_ids", "qualified"),
    )


def decode_projected_route_costs(payload: bytes) -> tuple[tuple[str, float], ...]:
    """Load projected full-corpus costs used by the fixed human selection rule."""
    record = _canonical_object(payload, "cost actuals")
    require_exact_fields(
        record,
        (
            "actual_billed_usd",
            "generation_billed_usd",
            "projection_envelopes",
            "routes",
            "verification_billed_usd",
        ),
        label="cost actuals",
    )
    routes = _require_list(record["routes"], "cost actual routes")
    route_values = tuple(
        _decode_cost_actual_route(item, index) for index, item in enumerate(routes)
    )
    costs = tuple(
        (route_id, projected_cost)
        for route_id, _, _, _, projected_cost in route_values
    )
    if not costs or len(costs) != len({route_id for route_id, _ in costs}):
        raise AuditIoError("projected route costs must be nonempty and unique")
    _validate_projection_envelopes(record["projection_envelopes"], dict(costs))
    actual_billed = _require_nonnegative_number(
        record["actual_billed_usd"], "cost actuals actual_billed_usd"
    )
    generation_billed = _require_nonnegative_number(
        record["generation_billed_usd"], "cost actuals generation_billed_usd"
    )
    verification_billed = _require_nonnegative_number(
        record["verification_billed_usd"], "cost actuals verification_billed_usd"
    )
    if not math.isclose(
        actual_billed,
        generation_billed + verification_billed,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise AuditIoError("actual billed cost must equal generation plus verification")
    if not math.isclose(
        generation_billed,
        sum(route_actual for _, route_actual, _, _, _ in route_values),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise AuditIoError("generation billed cost must equal the per-route total")
    return costs


def _validate_projection_envelopes(
    value: JsonValue,
    route_costs: Mapping[str, float],
) -> None:
    envelopes = require_json_object(value, label="cost projection envelopes")
    require_exact_fields(
        envelopes,
        ("balanced", "economy", "quality_ceiling"),
        label="cost projection envelopes",
    )
    for name in ("balanced", "economy", "quality_ceiling"):
        envelope = require_json_object(
            envelopes[name], label=f"{name} cost projection envelope"
        )
        require_exact_fields(
            envelope,
            (
                "available",
                "definition",
                "projected_accepted_story_count",
                "projected_full_corpus_usd",
                "reason",
                "route_id",
            ),
            label=f"{name} cost projection envelope",
        )
        available = _require_boolean(
            envelope["available"], f"{name} cost projection availability"
        )
        _require_text(envelope["definition"], f"{name} cost projection definition")
        projected_count = _require_integer(
            envelope["projected_accepted_story_count"],
            f"{name} projected accepted story count",
        )
        if projected_count <= 0:
            raise AuditIoError(
                f"{name} projected accepted story count must be positive"
            )
        if available:
            route_id = _require_text(
                envelope["route_id"], f"{name} cost projection route"
            )
            projected = _require_nonnegative_number(
                envelope["projected_full_corpus_usd"],
                f"{name} projected full-corpus cost",
            )
            if envelope["reason"] is not None:
                raise AuditIoError(
                    f"available {name} cost projection cannot contain a reason"
                )
            if route_id not in route_costs or not math.isclose(
                projected,
                route_costs[route_id],
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise AuditIoError(
                    f"{name} cost projection does not match its route record"
                )
        else:
            if (
                envelope["route_id"] is not None
                or envelope["projected_full_corpus_usd"] is not None
            ):
                raise AuditIoError(
                    f"unavailable {name} cost projection must not select a route"
                )
            _require_text(
                envelope["reason"], f"{name} unavailable projection reason"
            )


def _packet_record(packet: BlindedAuditPacket) -> JsonObject:
    return {
        "audit_sha256": packet.audit_sha256,
        "items": [
            {
                "automated_style_scores": [list(pair) for pair in item.automated_style_scores],
                "base_normalized_nll": item.base_normalized_nll,
                "item_id": item.item_id,
                "source_prompt": item.source_prompt,
                "story_text": item.story_text,
                "token_count": item.token_count,
            }
            for item in packet.items
        ],
    }


def _key_record(key: BlindedAuditKey) -> JsonObject:
    return {
        "audit_sha256": key.audit_sha256,
        "entries": [
            {
                "item_id": entry.item_id,
                "pair_id": entry.pair_id,
                "route_id": entry.route_id,
                "source_id": entry.source_id,
                "source_kind": entry.source_kind.value,
            }
            for entry in key.entries
        ],
    }


def _decision_record(decision: AuditDecision) -> JsonObject:
    return {
        "coherence_rating": decision.coherence_rating,
        "item_id": decision.item_id,
        "simplicity_rating": decision.simplicity_rating,
        "source_guess": decision.source_guess.value,
        "ts_like_accepted": decision.ts_like_accepted,
    }


def _decision_export_record(export: AuditDecisionExport) -> JsonObject:
    return {
        "audit_sha256": export.audit_sha256,
        "decisions": [_decision_record(item) for item in export.decisions],
    }


def _decision_set_record(decision_set: AuditDecisionSet) -> JsonObject:
    return {
        **_decision_export_record(
            AuditDecisionExport(
                decision_set.audit_sha256,
                decision_set.decisions,
            )
        ),
        "decision_sha256": decision_set.decision_sha256,
    }


def _evaluation_record(evaluation: AuditEvaluation) -> JsonObject:
    return {
        "audit_sha256": evaluation.audit_sha256,
        "decision_sha256": evaluation.decision_sha256,
        "failures": list(evaluation.failures),
        "generated_acceptance_rate": evaluation.generated_acceptance_rate,
        "mean_coherence_rating": evaluation.mean_coherence_rating,
        "mean_simplicity_rating": evaluation.mean_simplicity_rating,
        "reference_acceptance_rate": evaluation.reference_acceptance_rate,
        "route_acceptance_rates": [
            {"acceptance_rate": rate, "route_id": route_id}
            for route_id, rate in evaluation.route_acceptance_rates
        ],
        "source_discrimination_accuracy": evaluation.source_discrimination_accuracy,
    }


def _approval_record(approval: AuditApproval) -> JsonObject:
    return {
        "approved": approval.approved,
        "approved_at_utc": approval.approved_at_utc,
        "approved_by": approval.approved_by,
        "audit_sha256": approval.audit_sha256,
        "decision_sha256": approval.decision_sha256,
    }


def _approval_artifact_record(artifact: AuditApprovalArtifact) -> JsonObject:
    return {
        **_approval_record(artifact.approval),
        "evaluation": _evaluation_record(artifact.evaluation),
        "selected_route_id": artifact.selected_route_id,
    }


def _decode_blinded_item(value: JsonValue, index: int) -> BlindedAuditItem:
    label = f"blinded audit item {index}"
    record = _require_object(value, label)
    require_exact_fields(
        record,
        (
            "automated_style_scores",
            "base_normalized_nll",
            "item_id",
            "source_prompt",
            "story_text",
            "token_count",
        ),
        label=label,
    )
    return _construct(
        label,
        lambda: BlindedAuditItem(
            item_id=_require_text(record["item_id"], f"{label} item_id"),
            story_text=_require_text(record["story_text"], f"{label} story_text"),
            source_prompt=_require_text(
                record["source_prompt"], f"{label} source_prompt"
            ),
            token_count=_require_integer(
                record["token_count"], f"{label} token_count"
            ),
            base_normalized_nll=_require_finite_number(
                record["base_normalized_nll"], f"{label} base_normalized_nll"
            ),
            automated_style_scores=_decode_style_scores(
                record["automated_style_scores"], label
            ),
        ),
    )


def _decode_key_entry(value: JsonValue, index: int) -> AuditKeyEntry:
    label = f"blinded audit key entry {index}"
    record = _require_object(value, label)
    require_exact_fields(
        record,
        ("item_id", "pair_id", "route_id", "source_id", "source_kind"),
        label=label,
    )
    source_kind_text = _require_text(record["source_kind"], f"{label} source_kind")
    try:
        source_kind = AuditSourceKind(source_kind_text)
    except ValueError as error:
        raise AuditIoError(f"{label} source_kind is invalid") from error
    route_value = record["route_id"]
    if route_value is not None and type(route_value) is not str:
        raise AuditIoError(f"{label} route_id must be a string or null")
    return _construct(
        label,
        lambda: AuditKeyEntry(
            item_id=_require_text(record["item_id"], f"{label} item_id"),
            source_id=_require_text(record["source_id"], f"{label} source_id"),
            pair_id=_require_text(record["pair_id"], f"{label} pair_id"),
            source_kind=source_kind,
            route_id=route_value,
        ),
    )


def _decode_decision(value: JsonValue, index: int) -> AuditDecision:
    label = f"audit decision {index}"
    record = _require_object(value, label)
    require_exact_fields(
        record,
        (
            "coherence_rating",
            "item_id",
            "simplicity_rating",
            "source_guess",
            "ts_like_accepted",
        ),
        label=label,
    )
    guess_text = _require_text(record["source_guess"], f"{label} source_guess")
    try:
        guess = AuditSourceGuess(guess_text)
    except ValueError as error:
        raise AuditIoError(f"{label} source_guess is invalid") from error
    return _construct(
        label,
        lambda: AuditDecision(
            item_id=_require_text(record["item_id"], f"{label} item_id"),
            ts_like_accepted=_require_boolean(
                record["ts_like_accepted"], f"{label} ts_like_accepted"
            ),
            simplicity_rating=_require_integer(
                record["simplicity_rating"], f"{label} simplicity_rating"
            ),
            coherence_rating=_require_integer(
                record["coherence_rating"], f"{label} coherence_rating"
            ),
            source_guess=guess,
        ),
    )


def _decode_evaluation_record(record: JsonObject, label: str) -> AuditEvaluation:
    require_exact_fields(
        record,
        (
            "audit_sha256",
            "decision_sha256",
            "failures",
            "generated_acceptance_rate",
            "mean_coherence_rating",
            "mean_simplicity_rating",
            "reference_acceptance_rate",
            "route_acceptance_rates",
            "source_discrimination_accuracy",
        ),
        label=label,
    )
    route_rates = tuple(
        _decode_route_rate(item, index, label)
        for index, item in enumerate(
            _require_list(record["route_acceptance_rates"], f"{label} route rates")
        )
    )
    failures = tuple(
        _require_text(item, f"{label} failure {index}")
        for index, item in enumerate(
            _require_list(record["failures"], f"{label} failures")
        )
    )
    if not route_rates or len(route_rates) != len({name for name, _ in route_rates}):
        raise AuditIoError(f"{label} route rates must be nonempty and unique")
    if len(failures) != len(set(failures)):
        raise AuditIoError(f"{label} failures must be unique")
    rate_fields = (
        "reference_acceptance_rate",
        "generated_acceptance_rate",
        "source_discrimination_accuracy",
    )
    rates = {
        field: _require_rate(record[field], f"{label} {field}")
        for field in rate_fields
    }
    return AuditEvaluation(
        audit_sha256=_require_sha256(record["audit_sha256"], f"{label} audit digest"),
        decision_sha256=_require_sha256(
            record["decision_sha256"], f"{label} decision digest"
        ),
        reference_acceptance_rate=rates["reference_acceptance_rate"],
        generated_acceptance_rate=rates["generated_acceptance_rate"],
        route_acceptance_rates=route_rates,
        source_discrimination_accuracy=rates["source_discrimination_accuracy"],
        mean_simplicity_rating=_require_rating_mean(
            record["mean_simplicity_rating"], f"{label} mean simplicity"
        ),
        mean_coherence_rating=_require_rating_mean(
            record["mean_coherence_rating"], f"{label} mean coherence"
        ),
        failures=failures,
    )


def _decode_approval_record(record: JsonObject, label: str) -> AuditApproval:
    require_exact_fields(record, _APPROVAL_FIELDS, label=label)
    approved_at = _require_text(record["approved_at_utc"], f"{label} approved_at_utc")
    _validate_approval_timestamp(approved_at)
    return _construct(
        label,
        lambda: AuditApproval(
            audit_sha256=_require_sha256(
                record["audit_sha256"], f"{label} audit digest"
            ),
            decision_sha256=_require_sha256(
                record["decision_sha256"], f"{label} decision digest"
            ),
            approved_by=_require_text(record["approved_by"], f"{label} approved_by"),
            approved_at_utc=approved_at,
            approved=_require_boolean(record["approved"], f"{label} approved"),
        ),
    )


def _decode_route_rate(
    value: JsonValue,
    index: int,
    parent_label: str,
) -> tuple[str, float]:
    label = f"{parent_label} route rate {index}"
    record = _require_object(value, label)
    require_exact_fields(record, ("acceptance_rate", "route_id"), label=label)
    return (
        _require_text(record["route_id"], f"{label} route_id"),
        _require_rate(record["acceptance_rate"], f"{label} acceptance_rate"),
    )


def _decode_cost_actual_route(
    value: JsonValue,
    index: int,
) -> tuple[str, float, int, int, float]:
    label = f"cost actual route {index}"
    record = _require_object(value, label)
    require_exact_fields(
        record,
        (
            "accepted_count",
            "actual_billed_usd",
            "projected_full_corpus_usd",
            "request_count",
            "route_id",
        ),
        label=label,
    )
    actual_cost = _require_nonnegative_number(
        record["actual_billed_usd"],
        f"{label} actual_billed_usd",
    )
    projected_cost = _require_nonnegative_number(
        record["projected_full_corpus_usd"],
        f"{label} projected_full_corpus_usd",
    )
    request_count = _require_integer(record["request_count"], f"{label} request_count")
    accepted_count = _require_integer(
        record["accepted_count"], f"{label} accepted_count"
    )
    if request_count < 0 or not 0 <= accepted_count <= request_count:
        raise AuditIoError(
            f"{label} counts must satisfy 0 <= accepted_count <= request_count"
        )
    return (
        _require_text(record["route_id"], f"{label} route_id"),
        actual_cost,
        request_count,
        accepted_count,
        projected_cost,
    )


def _decode_style_scores(
    value: JsonValue,
    parent_label: str,
) -> tuple[tuple[str, float], ...]:
    values = _require_list(value, f"{parent_label} style scores")
    scores = tuple(
        _decode_style_score(item, index, parent_label)
        for index, item in enumerate(values)
    )
    if not scores or len(scores) != len({name for name, _ in scores}):
        raise AuditIoError(f"{parent_label} style scores must be nonempty and unique")
    return scores


def _decode_style_score(
    value: JsonValue,
    index: int,
    parent_label: str,
) -> tuple[str, float]:
    label = f"{parent_label} style score {index}"
    if type(value) is not list or len(value) != 2:
        raise AuditIoError(f"{label} must be a name/value pair")
    return (
        _require_text(value[0], f"{label} name"),
        _require_finite_number(value[1], f"{label} value"),
    )


def _decode_phase1_manifest(payload: bytes) -> Phase1ArtifactManifest:
    record = _canonical_object(payload, "Phase 1 manifest")
    require_exact_fields(
        record,
        ("artifacts", "format", "manifest_sha256", "schema_version", "version"),
        label="Phase 1 manifest",
    )
    artifact_records = _require_list(record["artifacts"], "Phase 1 artifacts")
    artifacts = tuple(
        _decode_phase1_artifact(item, index)
        for index, item in enumerate(artifact_records)
    )
    return _construct(
        "Phase 1 manifest",
        lambda: Phase1ArtifactManifest(
            format=_require_text(record["format"], "Phase 1 format"),
            schema_version=_require_integer(
                record["schema_version"], "Phase 1 schema version"
            ),
            version=_require_text(record["version"], "Phase 1 version"),
            artifacts=artifacts,
            manifest_sha256=_require_sha256(
                record["manifest_sha256"], "Phase 1 manifest digest"
            ),
        ),
    )


def _decode_phase1_artifact(value: JsonValue, index: int) -> Phase1ArtifactFile:
    label = f"Phase 1 artifact {index}"
    record = _require_object(value, label)
    require_exact_fields(
        record,
        ("content_format", "path", "record_count", "sha256", "size_bytes"),
        label=label,
    )
    return _construct(
        label,
        lambda: Phase1ArtifactFile(
            path=_require_text(record["path"], f"{label} path"),
            content_format=_require_text(
                record["content_format"], f"{label} content format"
            ),
            sha256=_require_sha256(record["sha256"], f"{label} digest"),
            size_bytes=_require_integer(record["size_bytes"], f"{label} size"),
            record_count=_require_integer(
                record["record_count"], f"{label} record count"
            ),
        ),
    )


def _regular_relative_files(root: Path) -> frozenset[str]:
    if not root.is_dir() or root.is_symlink():
        raise AuditIoError(f"Phase 1 artifact root is not a regular directory: {root}")
    paths = tuple(root.rglob("*"))
    symlinks = tuple(path for path in paths if path.is_symlink())
    if symlinks:
        raise AuditIoError("Phase 1 artifact tree contains a symlink")
    irregular = tuple(path for path in paths if not path.is_dir() and not path.is_file())
    if irregular:
        raise AuditIoError("Phase 1 artifact tree contains an irregular file")
    return frozenset(
        path.relative_to(root).as_posix() for path in paths if path.is_file()
    )


def _canonical_object(payload: bytes, label: str) -> JsonObject:
    if type(payload) is not bytes:
        raise TypeError(f"{label} payload must be bytes")
    try:
        value = canonical_json_loads(payload, label=label)
        return require_json_object(value, label=label)
    except CanonicalJsonError as error:
        raise AuditIoError(str(error)) from error


def _require_object(value: JsonValue, label: str) -> JsonObject:
    try:
        return require_json_object(value, label=label)
    except CanonicalJsonError as error:
        raise AuditIoError(str(error)) from error


def _require_list(value: JsonValue, label: str) -> list[JsonValue]:
    if type(value) is not list:
        raise AuditIoError(f"{label} must be a list")
    return value


def _require_text(value: JsonValue, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise AuditIoError(f"{label} must be a nonempty string")
    return value


def _require_sha256(value: JsonValue, label: str) -> str:
    text = _require_text(value, label)
    if _SHA256.fullmatch(text) is None:
        raise AuditIoError(f"{label} must be a lowercase SHA-256")
    return text


def _require_integer(value: JsonValue, label: str) -> int:
    if type(value) is not int:
        raise AuditIoError(f"{label} must be an integer")
    return value


def _require_boolean(value: JsonValue, label: str) -> bool:
    if type(value) is not bool:
        raise AuditIoError(f"{label} must be a boolean")
    return value


def _require_finite_number(value: JsonValue, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise AuditIoError(f"{label} must be a finite number")
    return float(value)


def _require_nonnegative_number(value: JsonValue, label: str) -> float:
    number = _require_finite_number(value, label)
    if number < 0.0:
        raise AuditIoError(f"{label} must be nonnegative")
    return number


def _require_rate(value: JsonValue, label: str) -> float:
    rate = _require_finite_number(value, label)
    if not 0.0 <= rate <= 1.0:
        raise AuditIoError(f"{label} must be between zero and one")
    return rate


def _require_rating_mean(value: JsonValue, label: str) -> float:
    rating = _require_finite_number(value, label)
    if not 1.0 <= rating <= 5.0:
        raise AuditIoError(f"{label} must be between one and five")
    return rating


def _validate_approval_timestamp(value: str) -> None:
    if _UTC_TIMESTAMP.fullmatch(value) is None:
        raise AuditIoError("approved_at_utc must be a whole-second UTC timestamp")
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise AuditIoError("approved_at_utc is not a valid UTC timestamp") from error


def _read_required(path: Path) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise AuditIoError(f"required audit artifact is missing: {path.name}")
    return path.read_bytes()


def _load_audit_pair(root: Path) -> tuple[BlindedAuditPacket, BlindedAuditKey]:
    return decode_audit_pair(
        _read_required(root / AUDIT_PACKET_FILENAME),
        _read_required(root / AUDIT_KEY_FILENAME),
    )


def _validate_audit_scope(
    key: BlindedAuditKey,
    scope: QualityAuditScope,
) -> None:
    route_counts = Counter(
        entry.route_id
        for entry in key.entries
        if entry.source_kind is AuditSourceKind.GENERATED
    )
    if set(route_counts) != set(scope.audited_route_ids):
        raise AuditIoError(
            "audit key generated routes must exactly match audited route IDs"
        )
    if max(route_counts.values()) - min(route_counts.values()) > 1:
        raise AuditIoError("audit generations must be balanced across audited routes")


def _construct(label: str, operation: Callable[[], _Result]) -> _Result:
    try:
        return operation()
    except AuditIoError:
        raise
    except (TypeError, ValueError) as error:
        raise AuditIoError(f"invalid {label}: {error}") from error

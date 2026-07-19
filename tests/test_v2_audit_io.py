from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from apm.data.text.tinyworlds_v2.audit import (
    AuditApproval,
    AuditDecision,
    AuditSourceGuess,
    AuditSourceKind,
    AuditSourceRecord,
    build_blinded_audit,
    evaluate_blinded_audit,
)
from apm.data.text.tinyworlds_v2.audit_io import (
    AUDIT_APPROVAL_FILENAME,
    AUDIT_APPROVAL_REQUEST_FILENAME,
    AUDIT_DECISIONS_FILENAME,
    AuditApprovalArtifact,
    AuditDecisionExport,
    AuditIoError,
    _approve_phase1_audit_after_semantic_validation,
    _validate_phase1_reference_overlays,
    approve_phase1_audit,
    decision_set_from_export,
    decode_audit_approval,
    decode_audit_approval_artifact,
    decode_audit_decision_export,
    decode_audit_decision_set,
    decode_audit_evaluation,
    decode_audit_pair,
    decode_blinded_audit_key,
    decode_blinded_audit_packet,
    decode_projected_route_costs,
    decode_quality_audit_scope,
    encode_audit_approval,
    encode_audit_approval_artifact,
    encode_audit_decision_export,
    encode_audit_decision_set,
    encode_audit_evaluation,
    encode_blinded_audit_key,
    encode_blinded_audit_packet,
    validate_phase1_reference,
    validate_phase1_tree_with_human_overlays,
)
from apm.data.text.tinyworlds_v2.json_contracts import canonical_json_bytes
from apm.data.text.tinyworlds_v2.phase1_artifacts import Phase1ArtifactBuilder


def _audit_fixture():
    prompts = tuple(f"Write a small story {index}." for index in range(2))
    references = tuple(
        AuditSourceRecord(
            source_id=f"reference-{index}",
            pair_id=f"pair-{index}",
            story_text=f"A little cat was kind in story {index}.",
            source_prompt=prompts[index],
            token_count=10,
            base_normalized_nll=1.2,
            automated_style_scores=(("simplicity", 4.5), ("coherence", 4.0)),
            source_kind=AuditSourceKind.REFERENCE,
        )
        for index in range(2)
    )
    generated = tuple(
        AuditSourceRecord(
            source_id=f"generated-{route_id}",
            pair_id=f"pair-{index}",
            story_text=f"A little dog was kind in story {index}.",
            source_prompt=prompts[index],
            token_count=10,
            base_normalized_nll=1.3,
            automated_style_scores=(("simplicity", 4.2), ("coherence", 4.1)),
            source_kind=AuditSourceKind.GENERATED,
            route_id=route_id,
        )
        for index, route_id in enumerate(("cheap", "best"))
    )
    return build_blinded_audit(
        references,
        generated,
        finalist_order=("cheap", "best"),
        seed="audit-io-fixture",
        reference_count=2,
        generated_count=2,
    )


def _accepted_export(packet) -> AuditDecisionExport:
    return AuditDecisionExport(
        packet.audit_sha256,
        tuple(
            AuditDecision(
                item_id=item.item_id,
                ts_like_accepted=True,
                simplicity_rating=4,
                coherence_rating=5,
                source_guess=AuditSourceGuess.REFERENCE,
            )
            for item in reversed(packet.items)
        ),
    )


def _build_reference_tree(root: Path) -> tuple[object, object]:
    packet, key = _audit_fixture()
    root.mkdir()
    builder = Phase1ArtifactBuilder(root)
    builder.write_bytes("audit_packet.json", encode_blinded_audit_packet(packet))
    builder.write_bytes("audit_key.json", encode_blinded_audit_key(key))
    builder.write_json(
        "quality_comparisons.json",
        {
            "audited_route_ids": ["cheap", "best"],
            "qualified_route_ids": ["cheap", "best"],
        },
    )
    builder.write_json(
        "cost_actuals.json",
        {
            "actual_billed_usd": 0.4,
            "generation_billed_usd": 0.3,
            "projection_envelopes": {
                name: {
                    "available": True,
                    "definition": definition,
                    "projected_accepted_story_count": 4000,
                    "projected_full_corpus_usd": cost,
                    "reason": None,
                    "route_id": route_id,
                }
                for name, definition, route_id, cost in (
                    (
                        "balanced",
                        "equal_weight_minmax_projected_cost_and_alignment_among_qualified",
                        "cheap",
                        1.0,
                    ),
                    (
                        "economy",
                        "minimum_projected_cost_among_qualified",
                        "cheap",
                        1.0,
                    ),
                    (
                        "quality_ceiling",
                        "minimum_alignment_distance_among_qualified",
                        "best",
                        3.0,
                    ),
                )
            },
            "routes": [
                {
                    "accepted_count": 200,
                    "actual_billed_usd": 0.1,
                    "projected_full_corpus_usd": 1.0,
                    "request_count": 200,
                    "route_id": "cheap",
                },
                {
                    "accepted_count": 198,
                    "actual_billed_usd": 0.2,
                    "projected_full_corpus_usd": 3.0,
                    "request_count": 200,
                    "route_id": "best",
                },
            ],
            "verification_billed_usd": 0.1,
        },
    )
    builder.finalize()
    return packet, key


def _write_decisions_and_request(root: Path, packet, *, digest: str | None = None) -> None:
    export = _accepted_export(packet)
    decision_set = decision_set_from_export(packet, export)
    (root / AUDIT_DECISIONS_FILENAME).write_bytes(
        encode_audit_decision_export(export)
    )
    request = AuditApproval(
        audit_sha256=packet.audit_sha256,
        decision_sha256=digest or decision_set.decision_sha256,
        approved_by="human-rater",
        approved_at_utc="2026-07-18T12:00:00Z",
        approved=True,
    )
    (root / AUDIT_APPROVAL_REQUEST_FILENAME).write_bytes(
        encode_audit_approval(request)
    )


def test_packet_key_codecs_are_canonical_strict_and_digest_bound() -> None:
    packet, key = _audit_fixture()
    packet_payload = encode_blinded_audit_packet(packet)
    key_payload = encode_blinded_audit_key(key)

    assert decode_blinded_audit_packet(packet_payload) == packet
    assert decode_blinded_audit_key(key_payload) == key
    assert decode_audit_pair(packet_payload, key_payload) == (packet, key)

    scope = decode_quality_audit_scope(
        canonical_json_bytes(
            {
                "audited_route_ids": ["cheap", "close", "tradeoff"],
                "qualified_route_ids": ["cheap", "tradeoff"],
            }
        )
    )
    assert scope.audited_route_ids == ("cheap", "close", "tradeoff")
    assert scope.qualified_route_ids == ("cheap", "tradeoff")

    with pytest.raises(AuditIoError, match="subset"):
        decode_quality_audit_scope(
            canonical_json_bytes(
                {
                    "audited_route_ids": ["cheap", "close"],
                    "qualified_route_ids": ["other"],
                }
            )
        )

    pretty = json.dumps(json.loads(packet_payload), indent=2).encode()
    with pytest.raises(ValueError, match="non-canonical"):
        decode_blinded_audit_packet(pretty)

    unknown = json.loads(packet_payload)
    unknown["unexpected"] = True
    with pytest.raises(ValueError, match="unknown=.*unexpected"):
        decode_blinded_audit_packet(canonical_json_bytes(unknown))

    changed = json.loads(packet_payload)
    changed["items"][0]["story_text"] = "Tampered story."
    with pytest.raises(ValueError, match="digest"):
        decode_audit_pair(canonical_json_bytes(changed), key_payload)


def test_decision_evaluation_and_approval_codecs_recompute_digests() -> None:
    packet, key = _audit_fixture()
    export = _accepted_export(packet)
    export_payload = encode_audit_decision_export(export)
    decoded_export = decode_audit_decision_export(export_payload)
    decision_set = decision_set_from_export(packet, decoded_export)

    assert decode_audit_decision_set(
        encode_audit_decision_set(decision_set), packet
    ) == decision_set
    changed = json.loads(encode_audit_decision_set(decision_set))
    changed["decision_sha256"] = "0" * 64
    with pytest.raises(AuditIoError, match="digest mismatch"):
        decode_audit_decision_set(canonical_json_bytes(changed), packet)

    evaluation = evaluate_blinded_audit(
        packet,
        key,
        decision_set,
        selectable_route_ids=("cheap", "best"),
    )
    assert evaluation.passed
    assert decode_audit_evaluation(encode_audit_evaluation(evaluation)) == evaluation

    approval = AuditApproval(
        packet.audit_sha256,
        decision_set.decision_sha256,
        approved_by="human-rater",
        approved_at_utc="2026-07-18T12:00:00Z",
        approved=True,
    )
    assert decode_audit_approval(encode_audit_approval(approval)) == approval
    artifact = AuditApprovalArtifact(approval, evaluation, "cheap")
    assert decode_audit_approval_artifact(
        encode_audit_approval_artifact(artifact)
    ) == artifact

    invalid_time = replace(approval, approved_at_utc="sometime")
    with pytest.raises(AuditIoError, match="approved_at_utc"):
        encode_audit_approval(invalid_time)


def test_cost_actuals_bind_observed_billing_counts_and_projection() -> None:
    record = {
        "actual_billed_usd": 0.4,
        "generation_billed_usd": 0.3,
        "projection_envelopes": {
            name: {
                "available": True,
                "definition": f"{name}-definition",
                "projected_accepted_story_count": 4000,
                "projected_full_corpus_usd": 2.5,
                "reason": None,
                "route_id": "route-a",
            }
            for name in ("balanced", "economy", "quality_ceiling")
        },
        "routes": [
            {
                "accepted_count": 49,
                "actual_billed_usd": 0.3,
                "projected_full_corpus_usd": 2.5,
                "request_count": 50,
                "route_id": "route-a",
            }
        ],
        "verification_billed_usd": 0.1,
    }
    assert decode_projected_route_costs(canonical_json_bytes(record)) == (
        ("route-a", 2.5),
    )

    wrong_total = {**record, "actual_billed_usd": 0.5}
    with pytest.raises(AuditIoError, match="generation plus verification"):
        decode_projected_route_costs(canonical_json_bytes(wrong_total))

    invalid_count = json.loads(canonical_json_bytes(record))
    invalid_count["routes"][0]["accepted_count"] = 51
    with pytest.raises(AuditIoError, match="accepted_count"):
        decode_projected_route_costs(canonical_json_bytes(invalid_count))


def test_approval_workflow_is_explicit_selects_cheapest_and_never_overwrites(
    tmp_path: Path,
) -> None:
    root = tmp_path / "reference"
    packet, _ = _build_reference_tree(root)
    _write_decisions_and_request(root, packet)

    artifact = _approve_phase1_audit_after_semantic_validation(root)

    assert artifact.selected_route_id == "cheap"
    assert artifact.evaluation.passed
    assert (root / AUDIT_APPROVAL_FILENAME).read_bytes() == (
        encode_audit_approval_artifact(artifact)
    )
    validated = _validate_phase1_reference_overlays(root)
    assert validated.approval_artifact == artifact
    with pytest.raises(FileExistsError, match="already exists"):
        _approve_phase1_audit_after_semantic_validation(root)

    tampered = json.loads((root / AUDIT_APPROVAL_FILENAME).read_bytes())
    tampered["selected_route_id"] = "best"
    (root / AUDIT_APPROVAL_FILENAME).write_bytes(canonical_json_bytes(tampered))
    with pytest.raises(AuditIoError, match="selected route"):
        _validate_phase1_reference_overlays(root)


def test_approval_fails_closed_for_missing_or_wrong_explicit_digest(
    tmp_path: Path,
) -> None:
    missing_request = tmp_path / "missing-request"
    packet, _ = _build_reference_tree(missing_request)
    (missing_request / AUDIT_DECISIONS_FILENAME).write_bytes(
        encode_audit_decision_export(_accepted_export(packet))
    )
    with pytest.raises(AuditIoError, match="missing"):
        _approve_phase1_audit_after_semantic_validation(missing_request)
    assert not (missing_request / AUDIT_APPROVAL_FILENAME).exists()

    wrong_digest = tmp_path / "wrong-digest"
    packet, _ = _build_reference_tree(wrong_digest)
    _write_decisions_and_request(wrong_digest, packet, digest="0" * 64)
    with pytest.raises(AuditIoError, match="approval digests"):
        _approve_phase1_audit_after_semantic_validation(wrong_digest)
    assert not (wrong_digest / AUDIT_APPROVAL_FILENAME).exists()


def test_overlay_validator_permits_only_exact_human_files(tmp_path: Path) -> None:
    root = tmp_path / "reference"
    packet, _ = _build_reference_tree(root)
    _write_decisions_and_request(root, packet)
    validate_phase1_tree_with_human_overlays(root)

    (root / "unrecognized.json").write_bytes(canonical_json_bytes({"ok": True}))
    with pytest.raises(AuditIoError, match="unknown overlays"):
        validate_phase1_tree_with_human_overlays(root)

    (root / "unrecognized.json").unlink()
    (root / "empty-directory").mkdir()
    with pytest.raises(AuditIoError, match="unknown directories"):
        validate_phase1_tree_with_human_overlays(root)


def test_validator_preserves_failed_human_gates_as_evidence(tmp_path: Path) -> None:
    root = tmp_path / "reference"
    packet, _ = _build_reference_tree(root)
    rejected = AuditDecisionExport(
        packet.audit_sha256,
        tuple(
            AuditDecision(
                item.item_id,
                ts_like_accepted=False,
                simplicity_rating=2,
                coherence_rating=2,
                source_guess=AuditSourceGuess.REFERENCE,
            )
            for item in packet.items
        ),
    )
    (root / AUDIT_DECISIONS_FILENAME).write_bytes(
        encode_audit_decision_export(rejected)
    )

    validation = _validate_phase1_reference_overlays(root)

    assert validation.evaluation is not None
    assert not validation.evaluation.passed
    assert "generated_acceptance_rate" in validation.evaluation.failures

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest

from apm.continual.tinyworlds_calibration import (
    BASELINE_CALIBRATION_CONFIG,
    CalibrationIdentity,
    CalibrationStopReason,
    CalibrationTrialPurpose,
    CalibrationValidationRequest,
    CalibrationValidationTrial,
    CommittedNodeSnapshot,
    CommittedNodeStabilityEvidence,
    TinyWorldsCalibrationEvidence,
    TinyWorldsCalibrationResult,
    calibration_binomial_evidence,
)
from apm.continual.tinyworlds_calibration_profile import (
    calibration_artifact_tree_sha256,
    write_calibration_result,
)
from apm.data.text.tinyworlds import (
    QueryKind,
    TINYWORLDS_VERSION,
    TinyWorldsBundle,
    derive_master_seed,
    generate_calibration_bundle,
    write_tinyworlds_bundle,
)
from apm.interactive.tinyworlds import (
    _load_candidate_scores,
    exact_kg_summary,
    inspect_query,
    load_tinyworlds_lab,
)


_CANONICAL_CALIBRATION_MASTER_SEED = (
    "f070dcf0d4ea88db86ca29486bf2eec3"
    "ec37d9187b73ae547c5b0598c3cb8d1a"
)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def calibration_bundle() -> TinyWorldsBundle:
    return generate_calibration_bundle(
        _CANONICAL_CALIBRATION_MASTER_SEED,
        direct_facts_per_task=36,
    )


def test_exact_kg_summary_checks_each_unique_validation_plan(
    calibration_bundle: TinyWorldsBundle,
) -> None:
    summary = exact_kg_summary(calibration_bundle)

    assert summary.successes == summary.trials == 8
    assert summary.accuracy == 1.0
    assert tuple(item.kind for item in summary.by_kind) == tuple(QueryKind)
    assert all(
        item.successes == item.trials == 1
        for item in summary.by_kind
    )


def test_query_inspection_distinguishes_zero_one_and_two_rule_applications(
    calibration_bundle: TinyWorldsBundle,
) -> None:
    direct = inspect_query(calibration_bundle, QueryKind.DIRECT)
    one_hop = inspect_query(calibration_bundle, QueryKind.ONE_HOP)
    two_hop = inspect_query(calibration_bundle, QueryKind.TWO_HOP)

    assert direct.proof_depth == 0
    assert direct.answers_by_max_depth[0][1] == (direct.answer_entity_id,)

    assert one_hop.proof_depth == 1
    assert one_hop.answers_by_max_depth == (
        (0, ()),
        (1, (one_hop.answer_entity_id,)),
        (2, (one_hop.answer_entity_id,)),
    )
    assert sum(step.rule_id is not None for step in one_hop.proof_steps) == 1

    assert two_hop.proof_depth == 2
    assert two_hop.answers_by_max_depth == (
        (0, ()),
        (1, ()),
        (2, (two_hop.answer_entity_id,)),
    )


def test_one_hop_support_positions_explain_the_facts_axis_result(
    calibration_bundle: TinyWorldsBundle,
) -> None:
    inspection = inspect_query(calibration_bundle, QueryKind.ONE_HOP)
    positions = tuple(
        sorted(fact.exposure_position for fact in inspection.support_facts)
    )

    assert positions == (6, 31)
    assert tuple(
        sum(position <= budget for position in positions)
        for budget in (12, 24, 36)
    ) == (1, 1, 2)
    assert all(
        not fact.answer_survives_removal
        for fact in inspection.support_facts
    )


def test_cross_branch_query_has_no_complete_hard_node_support(
    calibration_bundle: TinyWorldsBundle,
) -> None:
    inspection = inspect_query(calibration_bundle, QueryKind.CROSS_BRANCH)
    recall_by_node = {
        item.node_id: item.required_edge_recall
        for item in inspection.hard_support
    }
    bridge = next(
        item
        for item in inspection.hard_support
        if item.node_id == "calibration_bridge"
    )

    assert inspection.hard_oracle_task_ids == ()
    assert len(inspection.required_edge_ids) == 4
    assert recall_by_node == {
        "calibration_seed": 0.25,
        "calibration_extension": 0.5,
        "calibration_revision": 0.5,
        "calibration_bridge": 0.75,
    }
    assert "edge:calibration_extension" in inspection.required_edge_ids
    assert "edge:calibration_extension" not in bridge.path_edge_ids
    assert max(recall_by_node.values()) < 1.0
    assert all(
        not fact.answer_survives_removal
        for fact in inspection.support_facts
    )


def test_candidate_score_parser_checks_predictions_and_exposes_margin(
    tmp_path: Path,
) -> None:
    score_path = tmp_path / "candidate_scores.jsonl"
    _write_jsonl(
        score_path,
        (
            _exact_kg_row(),
            _neural_score_row(
                query_id="group:direct:0000:prefix-64",
                metric="direct",
                candidate_nll=(2.0, 1.0, 3.0, 4.0),
                correct_index=1,
                predicted_index=1,
            ),
            _neural_score_row(
                query_id="group:one-hop:0001:prefix-64",
                metric="one_hop",
                candidate_nll=(0.5, 1.0, 3.0, 4.0),
                correct_index=1,
                predicted_index=0,
            ),
        ),
    )

    scores, exact_rows = _load_candidate_scores(score_path)

    assert exact_rows == 1
    assert len(scores) == 2
    assert scores[0].correct
    assert scores[0].margin == pytest.approx(1.0)
    assert scores[0].group_id == "group:direct:0000"
    assert not scores[1].correct
    assert scores[1].margin == pytest.approx(-0.5)


@pytest.mark.parametrize(
    ("changed_field", "changed_value", "message"),
    (
        ("predicted_candidate_index", 2, "minimum NLL"),
        ("correct", False, "correctness flag"),
        ("prefix_length", 65, "prefix length"),
    ),
)
def test_candidate_score_parser_rejects_tampered_rows(
    tmp_path: Path,
    changed_field: str,
    changed_value: object,
    message: str,
) -> None:
    record = _neural_score_row(
        query_id="group:direct:0000:prefix-64",
        metric="direct",
        candidate_nll=(2.0, 1.0, 3.0, 4.0),
        correct_index=1,
        predicted_index=1,
    )
    record[changed_field] = changed_value
    score_path = tmp_path / "candidate_scores.jsonl"
    _write_jsonl(score_path, (record,))

    with pytest.raises(ValueError, match=message):
        _load_candidate_scores(score_path)


def test_lab_loader_uses_only_strict_generated_fixture_artifacts(
    tmp_path: Path,
) -> None:
    base_manifest_sha256 = "a" * 64
    base_parameter_checksum = "b" * 64
    fixture_master_seed = derive_master_seed(
        TINYWORLDS_VERSION,
        0,
        base_manifest_sha256,
        base_parameter_checksum,
    )
    fixture_bundle = generate_calibration_bundle(
        fixture_master_seed,
        direct_facts_per_task=36,
    )
    result_directory = tmp_path / "calibration-stopped"
    symbolic_manifest = write_tinyworlds_bundle(
        fixture_bundle,
        result_directory / "symbolic-calibration-pool",
    )
    identity = CalibrationIdentity(
        benchmark_version=TINYWORLDS_VERSION,
        public_seed=0,
        calibration_bundle_sha256=symbolic_manifest.bundle_sha256,
        base_manifest_sha256=base_manifest_sha256,
        base_parameter_checksum=base_parameter_checksum,
        tokenizer_sha256="c" * 64,
    )
    evidence = _failing_evidence()
    requests = (
        CalibrationValidationRequest(
            trial_index=0,
            purpose=CalibrationTrialPurpose.BASELINE,
            config=BASELINE_CALIBRATION_CONFIG,
        ),
        CalibrationValidationRequest(
            trial_index=1,
            purpose=CalibrationTrialPurpose.FACTS,
            config=replace(BASELINE_CALIBRATION_CONFIG, facts_per_task=12),
        ),
        CalibrationValidationRequest(
            trial_index=2,
            purpose=CalibrationTrialPurpose.FACTS,
            config=replace(BASELINE_CALIBRATION_CONFIG, facts_per_task=36),
        ),
    )
    trials = tuple(
        _write_fixture_trial(result_directory, request, evidence)
        for request in requests
    )
    result = TinyWorldsCalibrationResult(
        identity=identity,
        validation_trials=trials,
        profile=None,
        stop_reason=CalibrationStopReason.FACTS_NO_PASS,
    )
    write_calibration_result(result, result_directory)

    lab = load_tinyworlds_lab(
        result_directory,
        repository_root=_REPOSITORY_ROOT,
    )

    assert lab.result == result
    assert lab.calibration_bundle.world.master_seed_sha256 == fixture_master_seed
    assert tuple(
        artifact.trial.request.config.facts_per_task
        for artifact in lab.trials
    ) == (24, 12, 36)
    assert lab.trial_for_facts(36).exact_kg_rows == 1
    assert lab.trial_for_facts(36).learned_graph[0].node_id == "root"
    with pytest.raises(KeyError, match="expected one trial"):
        lab.trial_for_facts(99)

    tampered = lab.trial_for_facts(36).directory / "candidate_scores.jsonl"
    tampered.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="tree checksum mismatch"):
        load_tinyworlds_lab(
            result_directory,
            repository_root=_REPOSITORY_ROOT,
        )


def _neural_score_row(
    *,
    query_id: str,
    metric: str,
    candidate_nll: tuple[float, float, float, float],
    correct_index: int,
    predicted_index: int,
) -> dict[str, object]:
    return {
        "candidate_nll": list(candidate_nll),
        "correct": predicted_index == correct_index,
        "correct_candidate_index": correct_index,
        "method": "fixture_method",
        "metric": metric,
        "predicted_candidate_index": predicted_index,
        "prefix_length": 64,
        "query_id": query_id,
        "task_id": "calibration_seed",
    }


def _exact_kg_row() -> dict[str, object]:
    return {
        "answer_entity_ids": ["entity:calibration_seed:05"],
        "correct": True,
        "group_id": "fixture-group",
        "method": "exact_kg",
        "metric": "exact_kg",
        "query_id": "fixture-query",
        "task_id": "calibration_seed",
    }


def _write_jsonl(path: Path, records: tuple[dict[str, object], ...]) -> None:
    path.write_text(
        "".join(
            json.dumps(record, allow_nan=False, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def _failing_evidence() -> TinyWorldsCalibrationEvidence:
    perfect = calibration_binomial_evidence(1, 1)
    snapshot = CommittedNodeSnapshot(
        node_id="calibration_seed",
        adapter_sha256="d" * 64,
        logits_sha256="e" * 64,
        answers_sha256="f" * 64,
    )
    return TinyWorldsCalibrationEvidence(
        exact_kg=perfect,
        frozen_novel_binding=perfect,
        independent_direct_recall=perfect,
        frozen_one_hop=perfect,
        independent_one_hop=perfect,
        committed_node_stability=CommittedNodeStabilityEvidence(
            before=(snapshot,),
            after=(snapshot,),
        ),
        old_contextual_answer=perfect,
        revision_contextual_answer=perfect,
        paired_revision_consistency=perfect,
    )


def _write_fixture_trial(
    result_directory: Path,
    request: CalibrationValidationRequest,
    evidence: TinyWorldsCalibrationEvidence,
) -> CalibrationValidationTrial:
    artifact_id = f"fixture-validation-{request.trial_index:02d}"
    execution_sha256 = f"{request.trial_index + 1:x}" * 64
    directory = result_directory / "validation" / artifact_id
    directory.mkdir(parents=True)
    candidate_records = (_exact_kg_row(),) + tuple(
        _neural_score_row(
            query_id=f"group:{metric}:0000:prefix-64",
            metric=metric,
            candidate_nll=(1.0, 2.0, 3.0, 4.0),
            correct_index=0,
            predicted_index=0,
        )
        for metric in (
            "frozen_novel_binding",
            "independent_direct_recall",
            "frozen_one_hop",
            "independent_one_hop",
            "old_contextual_answer",
            "revision_contextual_answer",
        )
    )
    candidate_payload = "".join(
        json.dumps(record, allow_nan=False, sort_keys=True) + "\n"
        for record in candidate_records
    ).encode("utf-8")
    checkpoint_payload = b""
    parent_payload = b""
    (directory / "candidate_scores.jsonl").write_bytes(candidate_payload)
    (directory / "checkpointed_transfer.jsonl").write_bytes(checkpoint_payload)
    (directory / "parent_search.jsonl").write_bytes(parent_payload)
    np.savez_compressed(directory / "model.npz")
    config = request.config
    manifest = {
        "artifact_id": artifact_id,
        "artifacts": {
            "candidate_scores.jsonl": sha256(candidate_payload).hexdigest(),
            "checkpointed_transfer.jsonl": sha256(checkpoint_payload).hexdigest(),
            "parent_search.jsonl": sha256(parent_payload).hexdigest(),
        },
        "evidence": _raw_evidence_record(evidence),
        "execution_sha256": execution_sha256,
        "format": "apm.tinyworlds.calibration-trial",
        "model": {
            "graph": [
                {
                    "adapter_sha256": None,
                    "node_id": "root",
                    "parent_id": None,
                    "prefix": None,
                    "train_stage": 0,
                    "trained_task": None,
                }
            ],
            "independent_adapters": [],
            "stability_before": [],
            "tensor_names": [],
        },
        "model_file_sha256": _file_sha256(directory / "model.npz"),
        "request": {
            "config": {
                "distractor_policy": config.distractor_policy.value,
                "exposures_per_fact": config.exposures_per_fact,
                "facts_per_task": config.facts_per_task,
                "lora_rank": config.lora_rank,
                "update_budget": config.update_budget,
            },
            "locked_scratch_rerun": request.locked_scratch_rerun,
            "purpose": request.purpose.value,
            "trial_index": request.trial_index,
        },
        "resource_evidence": {
            "allocator_peak_bytes": 1,
            "allocator_peak_target_bytes": 2,
            "device_kind": "fixture-cpu",
            "platform": "cpu",
        },
        "schema_version": 2,
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, allow_nan=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return CalibrationValidationTrial(
        request=request,
        artifact_id=artifact_id,
        execution_sha256=execution_sha256,
        artifact_sha256=calibration_artifact_tree_sha256(directory),
        evidence=evidence,
    )


def _raw_evidence_record(
    evidence: TinyWorldsCalibrationEvidence,
) -> dict[str, object]:
    def binomial(value) -> dict[str, int]:
        return {"successes": value.successes, "trials": value.trials}

    stability = evidence.committed_node_stability

    def snapshot(value: CommittedNodeSnapshot) -> dict[str, str]:
        return {
            "adapter_sha256": value.adapter_sha256,
            "answers_sha256": value.answers_sha256,
            "logits_sha256": value.logits_sha256,
            "node_id": value.node_id,
        }

    return {
        "committed_node_stability": {
            "after": [snapshot(value) for value in stability.after],
            "before": [snapshot(value) for value in stability.before],
        },
        "exact_kg": binomial(evidence.exact_kg),
        "frozen_novel_binding": binomial(evidence.frozen_novel_binding),
        "frozen_one_hop": binomial(evidence.frozen_one_hop),
        "independent_direct_recall": binomial(
            evidence.independent_direct_recall
        ),
        "independent_one_hop": binomial(evidence.independent_one_hop),
        "old_contextual_answer": binomial(evidence.old_contextual_answer),
        "paired_revision_consistency": binomial(
            evidence.paired_revision_consistency
        ),
        "revision_contextual_answer": binomial(
            evidence.revision_contextual_answer
        ),
    }


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()

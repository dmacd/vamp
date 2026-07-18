from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

import scripts.run_tinyworlds_calibration as calibration_runner
from apm.continual.tinyworlds_calibration import (
    CalibrationIdentity,
    CalibrationNotAuthorizedError,
    CalibrationValidationObservation,
    CommittedNodeSnapshot,
    CommittedNodeStabilityEvidence,
    LockedCalibrationTestObservation,
    TinyWorldsCalibrationEvidence,
    TinyWorldsCalibrationResult,
    calibration_binomial_evidence,
    run_tinyworlds_calibration,
)
from apm.continual.tinyworlds_calibration_profile import (
    CALIBRATION_PROFILE_FILENAME,
    CALIBRATION_RESULT_FILENAME,
    calibration_artifact_tree_sha256,
    calibration_profile_sha256,
    calibration_result_sha256,
    load_calibration_result,
)
from apm.continual.tinyworlds_calibration_run import (
    CALIBRATION_ALLOCATOR_PEAK_TARGET_BYTES,
    CALIBRATION_REQUIRED_DEVICE_KIND,
    CALIBRATION_REQUIRED_PLATFORM,
    CalibrationResourceEvidence,
    _locked_test_request_record,
    _validation_request_record,
    _write_trial_artifact,
)
from apm.data.text.tinyworlds import (
    generate_calibration_bundle,
    write_tinyworlds_bundle,
)
from apm.lm.config import GptNeoConfig
from apm.lm.lora import LoraConfig


def _evidence(*, passing: bool) -> TinyWorldsCalibrationEvidence:
    snapshots = (
        CommittedNodeSnapshot(
            "calibration-seed",
            "1" * 64,
            "2" * 64,
            "3" * 64,
        ),
    )
    return TinyWorldsCalibrationEvidence(
        exact_kg=calibration_binomial_evidence(100 if passing else 99, 100),
        frozen_novel_binding=calibration_binomial_evidence(25, 100),
        independent_direct_recall=calibration_binomial_evidence(80, 100),
        frozen_one_hop=calibration_binomial_evidence(25, 100),
        independent_one_hop=calibration_binomial_evidence(80, 100),
        committed_node_stability=CommittedNodeStabilityEvidence(
            snapshots,
            snapshots,
        ),
        old_contextual_answer=calibration_binomial_evidence(80, 100),
        revision_contextual_answer=calibration_binomial_evidence(80, 100),
        paired_revision_consistency=calibration_binomial_evidence(65, 100),
    )


class _ArtifactEvaluator:
    def __init__(
        self,
        root: Path,
        *,
        passing: bool,
        resource_mutation: str | None = None,
    ) -> None:
        self.root = root
        self.passing = passing
        self.resource_mutation = resource_mutation
        self.model_config = GptNeoConfig(
            vocab_size=16,
            max_position_embeddings=8,
            hidden_size=8,
            intermediate_size=16,
            num_layers=1,
            num_heads=2,
            attention_types=("global",),
            local_window_size=4,
        )
        self.lora_config = LoraConfig(rank=1, alpha=1.0)
        self.resource_evidence = CalibrationResourceEvidence(
            platform=CALIBRATION_REQUIRED_PLATFORM,
            device_kind=CALIBRATION_REQUIRED_DEVICE_KIND,
            allocator_peak_bytes=0,
            allocator_peak_target_bytes=(
                CALIBRATION_ALLOCATOR_PEAK_TARGET_BYTES
            ),
        )

    def evaluate_validation(self, request):
        artifact_id = f"validation-{request.trial_index}"
        execution_sha = sha256(
            f"validation:{request.trial_index}".encode("utf-8")
        ).hexdigest()
        target = self.root / "validation" / artifact_id
        evidence = _evidence(passing=self.passing)
        _write_trial_artifact(
            target,
            artifact_id=artifact_id,
            execution_sha256=execution_sha,
            request_record=_validation_request_record(request),
            evidence=evidence,
            outcome=None,
            model_config=self.model_config,
            lora_config=self.lora_config,
            score_records=(),
            resource_evidence=self.resource_evidence,
        )
        if request.trial_index == 0:
            self._mutate_resource_record(target)
        return CalibrationValidationObservation(
            artifact_id=artifact_id,
            execution_sha256=execution_sha,
            artifact_sha256=calibration_artifact_tree_sha256(target),
            evidence=evidence,
        )

    def _mutate_resource_record(self, target: Path) -> None:
        if self.resource_mutation is None:
            return
        manifest_path = target / "manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        if self.resource_mutation == "missing":
            del manifest["resource_evidence"]
        elif self.resource_mutation == "wrong_device":
            manifest["resource_evidence"]["device_kind"] = (
                "NVIDIA GeForce RTX 3090"
            )
        elif self.resource_mutation == "wrong_target":
            manifest["resource_evidence"]["allocator_peak_target_bytes"] = 1
        else:
            raise ValueError("unknown test resource mutation")
        manifest_path.write_bytes(
            calibration_runner._canonical_json_bytes(manifest)
        )

    def evaluate_locked_test(self, request):
        artifact_id = "test-locked"
        execution_sha = sha256(b"test-locked").hexdigest()
        target = self.root / "test" / artifact_id
        evidence = _evidence(passing=True)
        _write_trial_artifact(
            target,
            artifact_id=artifact_id,
            execution_sha256=execution_sha,
            request_record=_locked_test_request_record(request),
            evidence=evidence,
            outcome=None,
            model_config=self.model_config,
            lora_config=self.lora_config,
            score_records=(),
            resource_evidence=self.resource_evidence,
        )
        return LockedCalibrationTestObservation(
            artifact_id=artifact_id,
            execution_sha256=execution_sha,
            artifact_sha256=calibration_artifact_tree_sha256(target),
            evidence=evidence,
        )


def _calibration_fixture(
    root: Path,
    *,
    passing: bool,
    resource_mutation: str | None = None,
) -> tuple[TinyWorldsCalibrationResult, Path, Path]:
    temporary = root / "temporary"
    temporary.mkdir(parents=True)
    bundle = generate_calibration_bundle("7" * 64, direct_facts_per_task=36)
    manifest = write_tinyworlds_bundle(
        bundle,
        temporary / "symbolic-calibration-pool",
    )
    for name in ("progress.jsonl", "sequential_results.jsonl"):
        (temporary / name).write_text("{}\n", encoding="utf-8")
    artifact_root = root / "artifacts"
    identity = CalibrationIdentity(
        benchmark_version="tinyworlds-v1",
        public_seed=0,
        calibration_bundle_sha256=manifest.bundle_sha256,
        base_manifest_sha256="b" * 64,
        base_parameter_checksum="c" * 64,
        tokenizer_sha256="d" * 64,
    )
    result = run_tinyworlds_calibration(
        identity,
        _ArtifactEvaluator(
            artifact_root,
            passing=passing,
            resource_mutation=resource_mutation,
        ),
    )
    return result, artifact_root, temporary


def test_success_and_stop_outcomes_promote_exact_artifact_sets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results_root = tmp_path / "results"
    monkeypatch.setattr(calibration_runner, "RESULTS_ROOT", results_root)
    success, success_artifacts, success_temporary = _calibration_fixture(
        tmp_path / "success",
        passing=True,
    )
    stopped, stopped_artifacts, stopped_temporary = _calibration_fixture(
        tmp_path / "stopped",
        passing=False,
    )

    success_root = calibration_runner._promote(
        success,
        success_artifacts,
        success_temporary,
    )
    stopped_root = calibration_runner._promote(
        stopped,
        stopped_artifacts,
        stopped_temporary,
    )

    assert load_calibration_result(success_root) == success
    assert (success_root / CALIBRATION_PROFILE_FILENAME).is_file()
    assert (success_root / "test").is_dir()
    assert load_calibration_result(stopped_root) == stopped
    assert (stopped_root / CALIBRATION_RESULT_FILENAME).is_file()
    assert not (stopped_root / CALIBRATION_PROFILE_FILENAME).exists()
    assert not (stopped_root / "test").exists()
    assert {path.name for path in (stopped_root / "validation").iterdir()} == {
        trial.artifact_id for trial in stopped.validation_trials
    }
    with pytest.raises(CalibrationNotAuthorizedError, match="not authorized"):
        load_calibration_result(stopped_root).require_pilot_profile()


def test_raw_artifact_and_outer_manifest_consistent_rehash_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(calibration_runner, "RESULTS_ROOT", tmp_path / "results")
    result, artifacts, temporary = _calibration_fixture(
        tmp_path / "success",
        passing=True,
    )
    promoted = calibration_runner._promote(result, artifacts, temporary)
    result_sha = calibration_result_sha256(result)
    profile_sha = calibration_profile_sha256(result.require_pilot_profile())
    trial_root = promoted / "validation" / result.validation_trials[0].artifact_id
    scores_path = trial_root / "candidate_scores.jsonl"
    scores_path.write_text('{"tampered":true}\n', encoding="utf-8")
    trial_manifest_path = trial_root / "manifest.json"
    trial_manifest = json.loads(trial_manifest_path.read_bytes())
    trial_manifest["artifacts"]["candidate_scores.jsonl"] = sha256(
        scores_path.read_bytes()
    ).hexdigest()
    trial_manifest_path.write_bytes(
        calibration_runner._canonical_json_bytes(trial_manifest)
    )
    outer_manifest = promoted / calibration_runner._PROMOTION_MANIFEST
    outer_manifest.unlink()
    calibration_runner._write_promotion_manifest(
        promoted,
        result_sha,
        profile_sha,
    )

    with pytest.raises(RuntimeError, match="artifact tree changed"):
        calibration_runner._validate_promoted_bundle(promoted, result_sha)


def test_nested_resource_and_outer_manifest_consistent_rehash_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(calibration_runner, "RESULTS_ROOT", tmp_path / "results")
    result, artifacts, temporary = _calibration_fixture(
        tmp_path / "success",
        passing=True,
    )
    promoted = calibration_runner._promote(result, artifacts, temporary)
    result_sha = calibration_result_sha256(result)
    profile_sha = calibration_profile_sha256(result.require_pilot_profile())
    trial_root = promoted / "validation" / result.validation_trials[0].artifact_id
    trial_manifest_path = trial_root / "manifest.json"
    trial_manifest = json.loads(trial_manifest_path.read_bytes())
    trial_manifest["resource_evidence"]["device_kind"] = (
        "NVIDIA GeForce RTX 3090"
    )
    trial_manifest_path.write_bytes(
        calibration_runner._canonical_json_bytes(trial_manifest)
    )
    outer_manifest = promoted / calibration_runner._PROMOTION_MANIFEST
    outer_manifest.unlink()
    calibration_runner._write_promotion_manifest(
        promoted,
        result_sha,
        profile_sha,
    )

    with pytest.raises(RuntimeError, match="artifact tree changed"):
        calibration_runner._validate_promoted_bundle(promoted, result_sha)


@pytest.mark.parametrize(
    ("resource_mutation", "expected_message"),
    (
        ("missing", "resource_evidence"),
        ("wrong_device", "RTX 4090"),
        ("wrong_target", "allocator target changed"),
    ),
)
def test_promotion_rejects_bound_missing_or_wrong_resource_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    resource_mutation: str,
    expected_message: str,
) -> None:
    monkeypatch.setattr(calibration_runner, "RESULTS_ROOT", tmp_path / "results")
    result, artifacts, temporary = _calibration_fixture(
        tmp_path / resource_mutation,
        passing=False,
        resource_mutation=resource_mutation,
    )

    with pytest.raises((RuntimeError, ValueError), match=expected_message):
        calibration_runner._promote(result, artifacts, temporary)

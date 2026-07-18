from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from apm.continual.tinyworlds_calibration import (
    CalibrationIdentity,
    CalibrationNotAuthorizedError,
    CalibrationValidationObservation,
    CommittedNodeSnapshot,
    CommittedNodeStabilityEvidence,
    LockedCalibrationTestObservation,
    TinyWorldsCalibrationEvidence,
    calibration_binomial_evidence,
    run_tinyworlds_calibration,
)
from apm.continual.tinyworlds_calibration_profile import (
    CALIBRATION_PROFILE_FILENAME,
    CALIBRATION_RESULT_FILENAME,
    CalibrationProfileError,
    CalibrationResultError,
    calibration_profile_sha256,
    calibration_result_sha256,
    load_calibration_profile,
    load_calibration_profile_sha256,
    load_calibration_result,
    load_calibration_result_sha256,
    write_calibration_result,
    write_calibration_profile,
)


def _canonical(record: object) -> bytes:
    return (
        json.dumps(
            record,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _evidence() -> TinyWorldsCalibrationEvidence:
    snapshots = (
        CommittedNodeSnapshot(
            "calibration-seed",
            "1" * 64,
            "2" * 64,
            "3" * 64,
        ),
    )
    return TinyWorldsCalibrationEvidence(
        exact_kg=calibration_binomial_evidence(100, 100),
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


class _PassingEvaluator:
    def evaluate_validation(self, request):
        return CalibrationValidationObservation(
            f"validation-{request.trial_index}",
            f"{request.trial_index:064x}",
            f"{request.trial_index + 16:064x}",
            _evidence(),
        )

    def evaluate_locked_test(self, request):
        return LockedCalibrationTestObservation(
            "test-locked",
            "e" * 64,
            "f" * 64,
            _evidence(),
        )


class _StoppedEvaluator:
    def evaluate_validation(self, request):
        evidence = _evidence()
        evidence = TinyWorldsCalibrationEvidence(
            exact_kg=calibration_binomial_evidence(99, 100),
            frozen_novel_binding=evidence.frozen_novel_binding,
            independent_direct_recall=evidence.independent_direct_recall,
            frozen_one_hop=evidence.frozen_one_hop,
            independent_one_hop=evidence.independent_one_hop,
            committed_node_stability=evidence.committed_node_stability,
            old_contextual_answer=evidence.old_contextual_answer,
            revision_contextual_answer=evidence.revision_contextual_answer,
            paired_revision_consistency=evidence.paired_revision_consistency,
        )
        return CalibrationValidationObservation(
            f"validation-{request.trial_index}",
            f"{request.trial_index:064x}",
            f"{request.trial_index + 16:064x}",
            evidence,
        )

    def evaluate_locked_test(self, request):
        raise AssertionError("stopped calibration must not expose test")


def _result():
    return run_tinyworlds_calibration(
        CalibrationIdentity(
            benchmark_version="tinyworlds-v1",
            public_seed=0,
            calibration_bundle_sha256="a" * 64,
            base_manifest_sha256="b" * 64,
            base_parameter_checksum="c" * 64,
            tokenizer_sha256="d" * 64,
        ),
        _PassingEvaluator(),
    )


def _profile():
    return _result().require_pilot_profile()


def _stopped_result():
    return run_tinyworlds_calibration(
        CalibrationIdentity(
            benchmark_version="tinyworlds-v1",
            public_seed=0,
            calibration_bundle_sha256="a" * 64,
            base_manifest_sha256="b" * 64,
            base_parameter_checksum="c" * 64,
            tokenizer_sha256="d" * 64,
        ),
        _StoppedEvaluator(),
    )


def _rewrite_rehashed(path: Path, mutate) -> None:
    record = json.loads(path.read_bytes())
    mutate(record)
    core = {
        key: value for key, value in record.items() if key != "profile_sha256"
    }
    record["profile_sha256"] = sha256(_canonical(core)).hexdigest()
    path.write_bytes(_canonical(record))


def _rewrite_result_rehashed(path: Path, mutate) -> None:
    record = json.loads(path.read_bytes())
    mutate(record)
    core = {
        key: value for key, value in record.items() if key != "result_sha256"
    }
    record["result_sha256"] = sha256(_canonical(core)).hexdigest()
    path.write_bytes(_canonical(record))


def test_profile_round_trip_is_canonical_hashed_and_byte_identical(
    tmp_path: Path,
) -> None:
    profile = _profile()
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_digest = write_calibration_profile(profile, first)
    second_digest = write_calibration_profile(profile, second)

    assert first_digest == second_digest == calibration_profile_sha256(profile)
    assert load_calibration_profile(first) == profile
    assert load_calibration_profile_sha256(first) == first_digest
    assert (first / CALIBRATION_PROFILE_FILENAME).read_bytes() == (
        second / CALIBRATION_PROFILE_FILENAME
    ).read_bytes()
    with pytest.raises(FileExistsError):
        write_calibration_profile(profile, first)


def test_successful_result_round_trip_binds_profile_and_trials(
    tmp_path: Path,
) -> None:
    result = _result()
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_calibration_profile(result.require_pilot_profile(), first)
    write_calibration_profile(result.require_pilot_profile(), second)

    first_digest = write_calibration_result(result, first)
    second_digest = write_calibration_result(result, second)

    assert first_digest == second_digest == calibration_result_sha256(result)
    assert load_calibration_result(first) == result
    assert load_calibration_result_sha256(first) == first_digest
    assert (first / CALIBRATION_RESULT_FILENAME).read_bytes() == (
        second / CALIBRATION_RESULT_FILENAME
    ).read_bytes()


def test_stopped_result_round_trip_has_no_profile_and_cannot_authorize_pilot(
    tmp_path: Path,
) -> None:
    result = _stopped_result()
    root = tmp_path / "stopped"

    digest = write_calibration_result(result, root)
    loaded = load_calibration_result(root)

    assert digest == calibration_result_sha256(result)
    assert loaded == result
    assert not (root / CALIBRATION_PROFILE_FILENAME).exists()
    with pytest.raises(CalibrationNotAuthorizedError, match="not authorized"):
        loaded.require_pilot_profile()


def test_stopped_result_loader_replays_stop_reason_from_evidence(
    tmp_path: Path,
) -> None:
    root = tmp_path / "stopped"
    write_calibration_result(_stopped_result(), root)
    path = root / CALIBRATION_RESULT_FILENAME

    def mutate(record):
        record["result"]["stop_reason"] = "exposures_axis_has_no_passing_configuration"

    _rewrite_result_rehashed(path, mutate)
    with pytest.raises(CalibrationResultError, match="stop_reason does not match"):
        load_calibration_result(root)


def test_result_rejects_consistently_rehashed_profile_binding_change(
    tmp_path: Path,
) -> None:
    result = _result()
    root = tmp_path / "result"
    write_calibration_profile(result.require_pilot_profile(), root)
    write_calibration_result(result, root)
    profile_path = root / CALIBRATION_PROFILE_FILENAME

    def mutate(record):
        record["profile"]["ladder_trials"][0]["artifact_sha256"] = "9" * 64

    _rewrite_rehashed(profile_path, mutate)
    with pytest.raises(CalibrationResultError, match="profile SHA-256 differs"):
        load_calibration_result(root)


def test_profile_loader_rejects_digest_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "profile"
    write_calibration_profile(_profile(), root)
    path = root / CALIBRATION_PROFILE_FILENAME
    path.write_bytes(path.read_bytes().replace(b'"public_seed":0', b'"public_seed":1'))

    with pytest.raises(CalibrationProfileError, match="SHA-256 mismatch"):
        load_calibration_profile(root)


def test_profile_loader_rejects_unknown_fields_after_consistent_rehash(
    tmp_path: Path,
) -> None:
    root = tmp_path / "profile"
    write_calibration_profile(_profile(), root)
    path = root / CALIBRATION_PROFILE_FILENAME

    def mutate(record):
        record["profile"]["identity"]["unexpected"] = "forbidden"

    _rewrite_rehashed(path, mutate)
    with pytest.raises(CalibrationProfileError, match="unknown=.*unexpected"):
        load_calibration_profile(root)


def test_profile_loader_replays_selection_after_consistent_rehash(
    tmp_path: Path,
) -> None:
    root = tmp_path / "profile"
    write_calibration_profile(_profile(), root)
    path = root / CALIBRATION_PROFILE_FILENAME

    def mutate(record):
        record["profile"]["selected_config"]["facts_per_task"] = 12

    _rewrite_rehashed(path, mutate)
    with pytest.raises(CalibrationProfileError, match="fixed ladder selection"):
        load_calibration_profile(root)


def test_profile_loader_rejects_test_not_bound_to_scratch_artifact(
    tmp_path: Path,
) -> None:
    root = tmp_path / "profile"
    write_calibration_profile(_profile(), root)
    path = root / CALIBRATION_PROFILE_FILENAME

    def mutate(record):
        record["profile"]["locked_test"]["request"][
            "validation_artifact_id"
        ] = "validation-9"

    _rewrite_rehashed(path, mutate)
    with pytest.raises(CalibrationProfileError, match="scratch artifact"):
        load_calibration_profile(root)


def test_profile_loader_rejects_gate_claim_inconsistent_with_evidence(
    tmp_path: Path,
) -> None:
    root = tmp_path / "profile"
    write_calibration_profile(_profile(), root)
    path = root / CALIBRATION_PROFILE_FILENAME

    def mutate(record):
        record["profile"]["ladder_trials"][0]["gate_decisions"][0][
            "passed"
        ] = False

    _rewrite_rehashed(path, mutate)
    with pytest.raises(CalibrationProfileError, match="do not match"):
        load_calibration_profile(root)

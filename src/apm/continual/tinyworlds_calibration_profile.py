"""Canonical hashed persistence for TinyWorlds calibration outcomes."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile

from apm.continual.language_benchmarks import WilsonConfidenceInterval
from apm.continual.tinyworlds_calibration import (
    CalibrationBinomialEvidence,
    CalibrationDistractorPolicy,
    CalibrationGateDecision,
    CalibrationGateName,
    CalibrationIdentity,
    CalibrationStopReason,
    CalibrationTrialPurpose,
    CalibrationValidationRequest,
    CalibrationValidationTrial,
    CommittedNodeSnapshot,
    CommittedNodeStabilityEvidence,
    LockedCalibrationTest,
    LockedCalibrationTestRequest,
    TinyWorldsCalibrationConfig,
    TinyWorldsCalibrationEvidence,
    TinyWorldsCalibrationProfile,
    TinyWorldsCalibrationResult,
    calibration_gate_decisions,
)


CALIBRATION_PROFILE_FILENAME = "calibration_profile.json"
CALIBRATION_RESULT_FILENAME = "calibration_result.json"
_FORMAT = "apm.tinyworlds.calibration-profile"
_SCHEMA_VERSION = 1
_RESULT_FORMAT = "apm.tinyworlds.calibration-result"
_RESULT_SCHEMA_VERSION = 1
_TRIAL_TREE_FORMAT = "apm.tinyworlds.calibration-trial-tree"
_TRIAL_TREE_SCHEMA_VERSION = 1


class CalibrationProfileError(ValueError):
    """A persisted calibration profile is malformed or fails validation."""


class CalibrationResultError(ValueError):
    """A persisted calibration outcome is malformed or fails validation."""


def calibration_profile_sha256(profile: TinyWorldsCalibrationProfile) -> str:
    """Return the canonical digest bound into ``calibration_profile.json``."""
    if type(profile) is not TinyWorldsCalibrationProfile:
        raise TypeError("profile must be a TinyWorldsCalibrationProfile")
    return _digest(_canonical_bytes(_profile_core(profile)))


def write_calibration_profile(
    profile: TinyWorldsCalibrationProfile,
    directory: str | Path,
) -> str:
    """Atomically write a successful profile without overwriting prior evidence."""
    if type(profile) is not TinyWorldsCalibrationProfile:
        raise TypeError("profile must be a TinyWorldsCalibrationProfile")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    target = root / CALIBRATION_PROFILE_FILENAME
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"calibration profile already exists: {target}")
    core = _profile_core(profile)
    profile_digest = _digest(_canonical_bytes(core))
    payload = _canonical_bytes({**core, "profile_sha256": profile_digest})
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{CALIBRATION_PROFILE_FILENAME}.tmp-",
        dir=root,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
    except FileExistsError:
        raise FileExistsError(f"calibration profile already exists: {target}")
    finally:
        temporary.unlink(missing_ok=True)
    return profile_digest


def load_calibration_profile(directory: str | Path) -> TinyWorldsCalibrationProfile:
    """Strictly load a canonical profile and replay every ladder decision."""
    path = Path(directory) / CALIBRATION_PROFILE_FILENAME
    if path.is_symlink() or not path.is_file():
        raise CalibrationProfileError(
            f"calibration profile must be a regular file: {path}"
        )
    payload = path.read_bytes()
    record = _loads_strict(payload)
    if payload != _canonical_bytes(record):
        raise CalibrationProfileError("calibration profile JSON is not canonical")
    data = _fields(
        record,
        ("format", "profile", "profile_sha256", "schema_version"),
        "calibration profile artifact",
    )
    if _string(data["format"], "format") != _FORMAT:
        raise CalibrationProfileError("unsupported calibration profile format")
    if _integer(data["schema_version"], "schema_version") != _SCHEMA_VERSION:
        raise CalibrationProfileError("unsupported calibration profile schema")
    profile_digest = _string(data["profile_sha256"], "profile_sha256")
    core = {
        "format": _FORMAT,
        "profile": data["profile"],
        "schema_version": _SCHEMA_VERSION,
    }
    if profile_digest != _digest(_canonical_bytes(core)):
        raise CalibrationProfileError("calibration profile SHA-256 mismatch")
    try:
        return _decode_profile(data["profile"])
    except (KeyError, TypeError, ValueError) as error:
        if type(error) is CalibrationProfileError:
            raise
        raise CalibrationProfileError(
            f"calibration profile contract validation failed: {error}"
        ) from error


def load_calibration_profile_sha256(directory: str | Path) -> str:
    """Load a profile and return its independently recomputed canonical digest."""
    profile = load_calibration_profile(directory)
    return calibration_profile_sha256(profile)


def calibration_result_sha256(result: TinyWorldsCalibrationResult) -> str:
    """Return the digest of one complete successful or stopped outcome."""
    if type(result) is not TinyWorldsCalibrationResult:
        raise TypeError("result must be a TinyWorldsCalibrationResult")
    return _digest(_canonical_bytes(_result_core(result)))


def calibration_artifact_tree_sha256(directory: str | Path) -> str:
    """Hash every immutable raw trial file by canonical relative path."""
    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise CalibrationResultError(
            f"calibration trial artifact must be a regular directory: {root}"
        )
    entries = tuple(sorted(root.rglob("*")))
    if any(path.is_symlink() for path in entries):
        raise CalibrationResultError("calibration trial artifact contains a symlink")
    files = tuple(path for path in entries if path.is_file())
    if not files:
        raise CalibrationResultError("calibration trial artifact is empty")
    return _digest(
        _canonical_bytes(
            {
                "files": [
                    {
                        "path": path.relative_to(root).as_posix(),
                        "sha256": _file_sha256(path),
                        "size_bytes": path.stat().st_size,
                    }
                    for path in files
                ],
                "format": _TRIAL_TREE_FORMAT,
                "schema_version": _TRIAL_TREE_SCHEMA_VERSION,
            }
        )
    )


def write_calibration_result(
    result: TinyWorldsCalibrationResult,
    directory: str | Path,
) -> str:
    """Atomically persist one complete outcome without overwriting evidence."""
    if type(result) is not TinyWorldsCalibrationResult:
        raise TypeError("result must be a TinyWorldsCalibrationResult")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    profile_path = root / CALIBRATION_PROFILE_FILENAME
    if result.profile is None:
        if profile_path.exists() or profile_path.is_symlink():
            raise FileExistsError("stopped calibration must not contain a profile")
    else:
        if load_calibration_profile(root) != result.profile:
            raise ValueError("successful calibration profile does not match result")
    target = root / CALIBRATION_RESULT_FILENAME
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"calibration result already exists: {target}")
    core = _result_core(result)
    result_digest = _digest(_canonical_bytes(core))
    payload = _canonical_bytes({**core, "result_sha256": result_digest})
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{CALIBRATION_RESULT_FILENAME}.tmp-",
        dir=root,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
    except FileExistsError:
        raise FileExistsError(f"calibration result already exists: {target}")
    finally:
        temporary.unlink(missing_ok=True)
    return result_digest


def load_calibration_result(directory: str | Path) -> TinyWorldsCalibrationResult:
    """Strictly load a complete outcome and replay its ladder decisions."""
    root = Path(directory)
    path = root / CALIBRATION_RESULT_FILENAME
    if path.is_symlink() or not path.is_file():
        raise CalibrationResultError(
            f"calibration result must be a regular file: {path}"
        )
    payload = path.read_bytes()
    try:
        record = _loads_strict(payload)
        if payload != _canonical_bytes(record):
            raise CalibrationResultError("calibration result JSON is not canonical")
        data = _fields(
            record,
            ("format", "result", "result_sha256", "schema_version"),
            "calibration result artifact",
        )
        if _string(data["format"], "format") != _RESULT_FORMAT:
            raise CalibrationResultError("unsupported calibration result format")
        if _integer(data["schema_version"], "schema_version") != (
            _RESULT_SCHEMA_VERSION
        ):
            raise CalibrationResultError("unsupported calibration result schema")
        core = {
            "format": _RESULT_FORMAT,
            "result": data["result"],
            "schema_version": _RESULT_SCHEMA_VERSION,
        }
        if _string(data["result_sha256"], "result_sha256") != _digest(
            _canonical_bytes(core)
        ):
            raise CalibrationResultError("calibration result SHA-256 mismatch")
        return _decode_result(data["result"], root)
    except (KeyError, TypeError, ValueError) as error:
        if type(error) is CalibrationResultError:
            raise
        raise CalibrationResultError(
            f"calibration result contract validation failed: {error}"
        ) from error


def load_calibration_result_sha256(directory: str | Path) -> str:
    """Load an outcome and independently recompute its canonical digest."""
    return calibration_result_sha256(load_calibration_result(directory))


def _result_core(result: TinyWorldsCalibrationResult) -> dict[str, object]:
    return {
        "format": _RESULT_FORMAT,
        "result": {
            "identity": _encode_identity(result.identity),
            "outcome": "success" if result.profile is not None else "stopped",
            "profile_sha256": (
                None
                if result.profile is None
                else calibration_profile_sha256(result.profile)
            ),
            "stop_reason": (
                None if result.stop_reason is None else result.stop_reason.value
            ),
            "validation_trials": [
                _encode_validation_trial(trial)
                for trial in result.validation_trials
            ],
        },
        "schema_version": _RESULT_SCHEMA_VERSION,
    }


def _decode_result(
    record: object,
    root: Path,
) -> TinyWorldsCalibrationResult:
    data = _fields(
        record,
        (
            "identity",
            "outcome",
            "profile_sha256",
            "stop_reason",
            "validation_trials",
        ),
        "calibration result",
    )
    identity = _decode_identity(data["identity"])
    trials = tuple(
        _decode_validation_trial(value)
        for value in _list(data["validation_trials"], "validation_trials")
    )
    outcome = _string(data["outcome"], "outcome")
    profile_path = root / CALIBRATION_PROFILE_FILENAME
    if outcome == "success":
        if data["stop_reason"] is not None:
            raise CalibrationResultError("successful result names a stop reason")
        expected_profile_sha = _string(
            data["profile_sha256"], "profile_sha256"
        )
        profile = load_calibration_profile(root)
        if calibration_profile_sha256(profile) != expected_profile_sha:
            raise CalibrationResultError("result profile SHA-256 differs")
        return TinyWorldsCalibrationResult(
            identity=identity,
            validation_trials=trials,
            profile=profile,
            stop_reason=None,
        )
    if outcome != "stopped":
        raise CalibrationResultError("calibration outcome is unknown")
    if data["profile_sha256"] is not None:
        raise CalibrationResultError("stopped result names a profile SHA-256")
    if profile_path.exists() or profile_path.is_symlink():
        raise CalibrationResultError("stopped result must not contain a profile")
    return TinyWorldsCalibrationResult(
        identity=identity,
        validation_trials=trials,
        profile=None,
        stop_reason=CalibrationStopReason(
            _string(data["stop_reason"], "stop_reason")
        ),
    )


def _profile_core(profile: TinyWorldsCalibrationProfile) -> dict[str, object]:
    return {
        "format": _FORMAT,
        "profile": _encode_profile(profile),
        "schema_version": _SCHEMA_VERSION,
    }


def _encode_profile(profile: TinyWorldsCalibrationProfile) -> dict[str, object]:
    return {
        "identity": _encode_identity(profile.identity),
        "ladder_trials": [
            _encode_validation_trial(trial) for trial in profile.ladder_trials
        ],
        "locked_scratch_trial": _encode_validation_trial(
            profile.locked_scratch_trial
        ),
        "locked_test": _encode_locked_test(profile.locked_test),
        "selected_config": _encode_config(profile.selected_config),
    }


def _decode_profile(record: object) -> TinyWorldsCalibrationProfile:
    data = _fields(
        record,
        (
            "identity",
            "ladder_trials",
            "locked_scratch_trial",
            "locked_test",
            "selected_config",
        ),
        "profile",
    )
    return TinyWorldsCalibrationProfile(
        identity=_decode_identity(data["identity"]),
        selected_config=_decode_config(data["selected_config"]),
        ladder_trials=tuple(
            _decode_validation_trial(item)
            for item in _list(data["ladder_trials"], "ladder_trials")
        ),
        locked_scratch_trial=_decode_validation_trial(
            data["locked_scratch_trial"]
        ),
        locked_test=_decode_locked_test(data["locked_test"]),
    )


def _encode_identity(identity: CalibrationIdentity) -> dict[str, object]:
    return {
        "base_manifest_sha256": identity.base_manifest_sha256,
        "base_parameter_checksum": identity.base_parameter_checksum,
        "benchmark_version": identity.benchmark_version,
        "calibration_bundle_sha256": identity.calibration_bundle_sha256,
        "public_seed": identity.public_seed,
        "tokenizer_sha256": identity.tokenizer_sha256,
    }


def _decode_identity(record: object) -> CalibrationIdentity:
    keys = (
        "base_manifest_sha256",
        "base_parameter_checksum",
        "benchmark_version",
        "calibration_bundle_sha256",
        "public_seed",
        "tokenizer_sha256",
    )
    data = _fields(record, keys, "calibration identity")
    return CalibrationIdentity(
        benchmark_version=_string(data["benchmark_version"], "benchmark_version"),
        public_seed=_integer(data["public_seed"], "public_seed"),
        calibration_bundle_sha256=_string(
            data["calibration_bundle_sha256"], "calibration_bundle_sha256"
        ),
        base_manifest_sha256=_string(
            data["base_manifest_sha256"], "base_manifest_sha256"
        ),
        base_parameter_checksum=_string(
            data["base_parameter_checksum"], "base_parameter_checksum"
        ),
        tokenizer_sha256=_string(data["tokenizer_sha256"], "tokenizer_sha256"),
    )


def _encode_config(config: TinyWorldsCalibrationConfig) -> dict[str, object]:
    return {
        "distractor_policy": config.distractor_policy.value,
        "exposures_per_fact": config.exposures_per_fact,
        "facts_per_task": config.facts_per_task,
        "lora_rank": config.lora_rank,
        "update_budget": config.update_budget,
    }


def _decode_config(record: object) -> TinyWorldsCalibrationConfig:
    keys = (
        "distractor_policy",
        "exposures_per_fact",
        "facts_per_task",
        "lora_rank",
        "update_budget",
    )
    data = _fields(record, keys, "calibration config")
    return TinyWorldsCalibrationConfig(
        facts_per_task=_integer(data["facts_per_task"], "facts_per_task"),
        exposures_per_fact=_integer(
            data["exposures_per_fact"], "exposures_per_fact"
        ),
        update_budget=_integer(data["update_budget"], "update_budget"),
        lora_rank=_integer(data["lora_rank"], "lora_rank"),
        distractor_policy=CalibrationDistractorPolicy(
            _string(data["distractor_policy"], "distractor_policy")
        ),
    )


def _encode_validation_trial(
    trial: CalibrationValidationTrial,
) -> dict[str, object]:
    return {
        "artifact_sha256": trial.artifact_sha256,
        "artifact_id": trial.artifact_id,
        "evidence": _encode_evidence(trial.evidence),
        "execution_sha256": trial.execution_sha256,
        "gate_decisions": [
            _encode_gate_decision(decision) for decision in trial.gate_decisions
        ],
        "passed": trial.passed,
        "request": _encode_validation_request(trial.request),
    }


def _decode_validation_trial(record: object) -> CalibrationValidationTrial:
    data = _fields(
        record,
        (
            "artifact_id",
            "artifact_sha256",
            "evidence",
            "execution_sha256",
            "gate_decisions",
            "passed",
            "request",
        ),
        "validation trial",
    )
    evidence = _decode_evidence(data["evidence"])
    expected_decisions = calibration_gate_decisions(evidence)
    supplied_decisions = tuple(
        _decode_gate_decision(item)
        for item in _list(data["gate_decisions"], "gate_decisions")
    )
    if supplied_decisions != expected_decisions:
        raise CalibrationProfileError(
            "persisted gate decisions do not match their evidence"
        )
    passed = _boolean(data["passed"], "trial passed")
    if passed != all(decision.passed for decision in expected_decisions):
        raise CalibrationProfileError("persisted trial pass flag is inconsistent")
    return CalibrationValidationTrial(
        request=_decode_validation_request(data["request"]),
        artifact_id=_string(data["artifact_id"], "artifact_id"),
        execution_sha256=_string(
            data["execution_sha256"], "execution_sha256"
        ),
        artifact_sha256=_string(data["artifact_sha256"], "artifact_sha256"),
        evidence=evidence,
    )


def _encode_validation_request(
    request: CalibrationValidationRequest,
) -> dict[str, object]:
    return {
        "config": _encode_config(request.config),
        "locked_scratch_rerun": request.locked_scratch_rerun,
        "purpose": request.purpose.value,
        "trial_index": request.trial_index,
    }


def _decode_validation_request(record: object) -> CalibrationValidationRequest:
    data = _fields(
        record,
        ("config", "locked_scratch_rerun", "purpose", "trial_index"),
        "validation request",
    )
    return CalibrationValidationRequest(
        trial_index=_integer(data["trial_index"], "trial_index"),
        purpose=CalibrationTrialPurpose(_string(data["purpose"], "purpose")),
        config=_decode_config(data["config"]),
        locked_scratch_rerun=_boolean(
            data["locked_scratch_rerun"], "locked_scratch_rerun"
        ),
    )


def _encode_gate_decision(decision: CalibrationGateDecision) -> dict[str, object]:
    return {
        "criterion": decision.criterion,
        "gate": decision.gate.value,
        "observed": list(decision.observed),
        "passed": decision.passed,
    }


def _decode_gate_decision(record: object) -> CalibrationGateDecision:
    data = _fields(
        record,
        ("criterion", "gate", "observed", "passed"),
        "gate decision",
    )
    return CalibrationGateDecision(
        gate=CalibrationGateName(_string(data["gate"], "gate")),
        passed=_boolean(data["passed"], "gate passed"),
        observed=tuple(
            _float(item, "gate observation")
            for item in _list(data["observed"], "gate observed")
        ),
        criterion=_string(data["criterion"], "gate criterion"),
    )


def _encode_evidence(evidence: TinyWorldsCalibrationEvidence) -> dict[str, object]:
    return {
        "committed_node_stability": _encode_stability(
            evidence.committed_node_stability
        ),
        "exact_kg": _encode_binomial(evidence.exact_kg),
        "frozen_novel_binding": _encode_binomial(evidence.frozen_novel_binding),
        "frozen_one_hop": _encode_binomial(evidence.frozen_one_hop),
        "independent_direct_recall": _encode_binomial(
            evidence.independent_direct_recall
        ),
        "independent_one_hop": _encode_binomial(evidence.independent_one_hop),
        "old_contextual_answer": _encode_binomial(evidence.old_contextual_answer),
        "paired_revision_consistency": _encode_binomial(
            evidence.paired_revision_consistency
        ),
        "revision_contextual_answer": _encode_binomial(
            evidence.revision_contextual_answer
        ),
    }


def _decode_evidence(record: object) -> TinyWorldsCalibrationEvidence:
    keys = (
        "committed_node_stability",
        "exact_kg",
        "frozen_novel_binding",
        "frozen_one_hop",
        "independent_direct_recall",
        "independent_one_hop",
        "old_contextual_answer",
        "paired_revision_consistency",
        "revision_contextual_answer",
    )
    data = _fields(record, keys, "calibration evidence")
    return TinyWorldsCalibrationEvidence(
        exact_kg=_decode_binomial(data["exact_kg"]),
        frozen_novel_binding=_decode_binomial(data["frozen_novel_binding"]),
        independent_direct_recall=_decode_binomial(
            data["independent_direct_recall"]
        ),
        frozen_one_hop=_decode_binomial(data["frozen_one_hop"]),
        independent_one_hop=_decode_binomial(data["independent_one_hop"]),
        committed_node_stability=_decode_stability(
            data["committed_node_stability"]
        ),
        old_contextual_answer=_decode_binomial(data["old_contextual_answer"]),
        revision_contextual_answer=_decode_binomial(
            data["revision_contextual_answer"]
        ),
        paired_revision_consistency=_decode_binomial(
            data["paired_revision_consistency"]
        ),
    )


def _encode_binomial(value: CalibrationBinomialEvidence) -> dict[str, object]:
    interval = value.confidence_interval
    return {
        "observed_rate": interval.observed_rate,
        "successes": value.successes,
        "trials": value.trials,
        "wilson_95": {"lower": interval.lower, "upper": interval.upper},
    }


def _decode_binomial(record: object) -> CalibrationBinomialEvidence:
    data = _fields(
        record,
        ("observed_rate", "successes", "trials", "wilson_95"),
        "binomial evidence",
    )
    bounds = _fields(data["wilson_95"], ("lower", "upper"), "Wilson bounds")
    successes = _integer(data["successes"], "successes")
    trials = _integer(data["trials"], "trials")
    return CalibrationBinomialEvidence(
        successes=successes,
        trials=trials,
        confidence_interval=WilsonConfidenceInterval(
            successes=successes,
            trials=trials,
            observed_rate=_float(data["observed_rate"], "observed_rate"),
            lower=_float(bounds["lower"], "Wilson lower"),
            upper=_float(bounds["upper"], "Wilson upper"),
        ),
    )


def _encode_stability(
    value: CommittedNodeStabilityEvidence,
) -> dict[str, object]:
    return {
        "after": [_encode_snapshot(item) for item in value.after],
        "before": [_encode_snapshot(item) for item in value.before],
        "bit_identical": value.bit_identical,
    }


def _decode_stability(record: object) -> CommittedNodeStabilityEvidence:
    data = _fields(
        record,
        ("after", "before", "bit_identical"),
        "committed-node stability",
    )
    value = CommittedNodeStabilityEvidence(
        before=tuple(
            _decode_snapshot(item)
            for item in _list(data["before"], "stability before")
        ),
        after=tuple(
            _decode_snapshot(item)
            for item in _list(data["after"], "stability after")
        ),
    )
    if _boolean(data["bit_identical"], "bit_identical") != value.bit_identical:
        raise CalibrationProfileError("stability bit-identical flag is inconsistent")
    return value


def _encode_snapshot(value: CommittedNodeSnapshot) -> dict[str, object]:
    return {
        "adapter_sha256": value.adapter_sha256,
        "answers_sha256": value.answers_sha256,
        "logits_sha256": value.logits_sha256,
        "node_id": value.node_id,
    }


def _decode_snapshot(record: object) -> CommittedNodeSnapshot:
    data = _fields(
        record,
        ("adapter_sha256", "answers_sha256", "logits_sha256", "node_id"),
        "committed-node snapshot",
    )
    return CommittedNodeSnapshot(
        node_id=_string(data["node_id"], "node_id"),
        adapter_sha256=_string(data["adapter_sha256"], "adapter_sha256"),
        logits_sha256=_string(data["logits_sha256"], "logits_sha256"),
        answers_sha256=_string(data["answers_sha256"], "answers_sha256"),
    )


def _encode_locked_test(value: LockedCalibrationTest) -> dict[str, object]:
    return {
        "artifact_sha256": value.artifact_sha256,
        "artifact_id": value.artifact_id,
        "evidence": _encode_evidence(value.evidence),
        "execution_sha256": value.execution_sha256,
        "request": {
            "config": _encode_config(value.request.config),
            "validation_artifact_id": value.request.validation_artifact_id,
            "validation_trial_index": value.request.validation_trial_index,
        },
    }


def _decode_locked_test(record: object) -> LockedCalibrationTest:
    data = _fields(
        record,
        (
            "artifact_id",
            "artifact_sha256",
            "evidence",
            "execution_sha256",
            "request",
        ),
        "locked test",
    )
    request_data = _fields(
        data["request"],
        ("config", "validation_artifact_id", "validation_trial_index"),
        "locked test request",
    )
    return LockedCalibrationTest(
        request=LockedCalibrationTestRequest(
            config=_decode_config(request_data["config"]),
            validation_trial_index=_integer(
                request_data["validation_trial_index"], "validation_trial_index"
            ),
            validation_artifact_id=_string(
                request_data["validation_artifact_id"], "validation_artifact_id"
            ),
        ),
        artifact_id=_string(data["artifact_id"], "test artifact_id"),
        execution_sha256=_string(
            data["execution_sha256"], "test execution_sha256"
        ),
        artifact_sha256=_string(
            data["artifact_sha256"], "test artifact_sha256"
        ),
        evidence=_decode_evidence(data["evidence"]),
    )


def _loads_strict(payload: bytes) -> object:
    def pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise CalibrationProfileError(
                    f"duplicate calibration profile field: {key}"
                )
            result[key] = value
        return result

    try:
        return json.loads(payload, object_pairs_hook=pairs_hook)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CalibrationProfileError(f"invalid calibration profile JSON: {error}") from error


def _canonical_bytes(record: object) -> bytes:
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


def _fields(
    record: object,
    expected: tuple[str, ...],
    label: str,
) -> dict[str, object]:
    if type(record) is not dict:
        raise CalibrationProfileError(f"{label} must be a JSON object")
    actual = set(record)
    wanted = set(expected)
    if actual != wanted:
        raise CalibrationProfileError(
            f"{label} fields differ; unknown={tuple(sorted(actual - wanted))}, "
            f"missing={tuple(sorted(wanted - actual))}"
        )
    return record


def _list(value: object, label: str) -> list[object]:
    if type(value) is not list:
        raise CalibrationProfileError(f"{label} must be a JSON array")
    return value


def _string(value: object, label: str) -> str:
    if type(value) is not str:
        raise CalibrationProfileError(f"{label} must be a string")
    return value


def _integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise CalibrationProfileError(f"{label} must be an integer")
    return value


def _float(value: object, label: str) -> float:
    if type(value) is not float:
        raise CalibrationProfileError(f"{label} must be a float")
    return value


def _boolean(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise CalibrationProfileError(f"{label} must be a bool")
    return value


def _digest(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "CALIBRATION_PROFILE_FILENAME",
    "CALIBRATION_RESULT_FILENAME",
    "CalibrationProfileError",
    "CalibrationResultError",
    "calibration_artifact_tree_sha256",
    "calibration_profile_sha256",
    "calibration_result_sha256",
    "load_calibration_profile",
    "load_calibration_profile_sha256",
    "load_calibration_result",
    "load_calibration_result_sha256",
    "write_calibration_profile",
    "write_calibration_result",
]

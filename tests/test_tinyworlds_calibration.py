from __future__ import annotations

from dataclasses import replace

import pytest

from apm.continual.tinyworlds_calibration import (
    BASELINE_CALIBRATION_CONFIG,
    CalibrationGateName,
    CalibrationIdentity,
    CalibrationNotAuthorizedError,
    CalibrationStopReason,
    CalibrationTrialPurpose,
    CalibrationValidationObservation,
    CommittedNodeSnapshot,
    CommittedNodeStabilityEvidence,
    LockedCalibrationTestObservation,
    TinyWorldsCalibrationEvidence,
    calibration_binomial_evidence,
    calibration_evidence_passes,
    calibration_gate_decisions,
    run_tinyworlds_calibration,
)


def _identity() -> CalibrationIdentity:
    return CalibrationIdentity(
        benchmark_version="tinyworlds-v1",
        public_seed=0,
        calibration_bundle_sha256="a" * 64,
        base_manifest_sha256="b" * 64,
        base_parameter_checksum="c" * 64,
        tokenizer_sha256="d" * 64,
    )


def _stability(*, drift: bool = False) -> CommittedNodeStabilityEvidence:
    before = (
        CommittedNodeSnapshot(
            node_id="calibration-seed",
            adapter_sha256="1" * 64,
            logits_sha256="2" * 64,
            answers_sha256="3" * 64,
        ),
        CommittedNodeSnapshot(
            node_id="calibration-extension",
            adapter_sha256="4" * 64,
            logits_sha256="5" * 64,
            answers_sha256="6" * 64,
        ),
    )
    after = before
    if drift:
        after = (replace(before[0], logits_sha256="7" * 64), before[1])
    return CommittedNodeStabilityEvidence(before=before, after=after)


def _passing_evidence() -> TinyWorldsCalibrationEvidence:
    return TinyWorldsCalibrationEvidence(
        exact_kg=calibration_binomial_evidence(100, 100),
        frozen_novel_binding=calibration_binomial_evidence(25, 100),
        independent_direct_recall=calibration_binomial_evidence(80, 100),
        frozen_one_hop=calibration_binomial_evidence(25, 100),
        independent_one_hop=calibration_binomial_evidence(80, 100),
        committed_node_stability=_stability(),
        old_contextual_answer=calibration_binomial_evidence(80, 100),
        revision_contextual_answer=calibration_binomial_evidence(80, 100),
        paired_revision_consistency=calibration_binomial_evidence(65, 100),
    )


def _failing_evidence() -> TinyWorldsCalibrationEvidence:
    return replace(
        _passing_evidence(),
        exact_kg=calibration_binomial_evidence(99, 100),
    )


class _Evaluator:
    def __init__(self, passes):
        self._passes = passes
        self.events: list[tuple[str, object]] = []

    def evaluate_validation(self, request):
        self.events.append(("validation", request))
        evidence = _passing_evidence() if self._passes(request) else _failing_evidence()
        return CalibrationValidationObservation(
            artifact_id=f"validation-artifact-{request.trial_index}",
            execution_sha256=f"{request.trial_index:064x}",
            artifact_sha256=f"{request.trial_index + 16:064x}",
            evidence=evidence,
        )

    def evaluate_locked_test(self, request):
        self.events.append(("test", request))
        # Test is deliberately not a tuning gate: a scientific miss remains in
        # the locked profile instead of selecting another configuration.
        return LockedCalibrationTestObservation(
            artifact_id="locked-test-artifact",
            execution_sha256="e" * 64,
            artifact_sha256="f" * 64,
            evidence=_failing_evidence(),
        )


def test_gate_decisions_use_exact_counts_wilson_bounds_and_stability() -> None:
    evidence = _passing_evidence()
    decisions = calibration_gate_decisions(evidence)

    assert calibration_evidence_passes(evidence)
    assert len(decisions) == len(CalibrationGateName)
    assert all(decision.passed for decision in decisions)
    frozen_wilson = next(
        decision
        for decision in decisions
        if decision.gate is CalibrationGateName.FROZEN_WILSON
    )
    assert frozen_wilson.observed[0] <= 0.25 <= frozen_wilson.observed[2]

    drifted = replace(evidence, committed_node_stability=_stability(drift=True))
    assert not calibration_evidence_passes(drifted)
    stability = next(
        decision
        for decision in calibration_gate_decisions(drifted)
        if decision.gate is CalibrationGateName.COMMITTED_STABILITY
    )
    assert not stability.passed


def test_fixed_ladder_uses_eleven_validation_trials_then_one_test() -> None:
    def passes(request) -> bool:
        config = request.config
        if request.purpose is CalibrationTrialPurpose.LOCKED_SCRATCH:
            return True
        return (
            config.facts_per_task != 12
            and config.exposures_per_fact != 16
            and config.update_budget != 500
            and config.lora_rank != 4
        )

    evaluator = _Evaluator(passes)
    result = run_tinyworlds_calibration(_identity(), evaluator)

    assert result.pilot_authorized
    profile = result.require_pilot_profile()
    assert len(result.validation_trials) == 11
    assert profile.selected_config == replace(
        BASELINE_CALIBRATION_CONFIG,
        facts_per_task=36,
    )
    assert profile.locked_scratch_trial.request.locked_scratch_rerun
    assert not calibration_evidence_passes(profile.locked_test.evidence)
    assert [kind for kind, _ in evaluator.events] == ["validation"] * 11 + [
        "test"
    ]
    test_request = evaluator.events[-1][1]
    assert test_request.validation_artifact_id == (
        profile.locked_scratch_trial.artifact_id
    )


def test_no_axis_pass_stops_without_test_or_pilot_authorization() -> None:
    evaluator = _Evaluator(lambda request: False)

    result = run_tinyworlds_calibration(_identity(), evaluator)

    assert not result.pilot_authorized
    assert result.stop_reason is CalibrationStopReason.FACTS_NO_PASS
    assert len(result.validation_trials) == 3
    assert [kind for kind, _ in evaluator.events] == ["validation"] * 3
    with pytest.raises(CalibrationNotAuthorizedError, match="not authorized"):
        result.require_pilot_profile()


def test_failing_locked_scratch_never_exposes_test_data() -> None:
    evaluator = _Evaluator(
        lambda request: request.purpose is not CalibrationTrialPurpose.LOCKED_SCRATCH
    )

    result = run_tinyworlds_calibration(_identity(), evaluator)

    assert not result.pilot_authorized
    assert result.stop_reason is CalibrationStopReason.LOCKED_SCRATCH_FAILED
    assert len(result.validation_trials) == 11
    assert [kind for kind, _ in evaluator.events] == ["validation"] * 11


def test_ladder_configs_change_exactly_one_axis_and_reuse_current_evidence() -> None:
    evaluator = _Evaluator(lambda request: True)
    result = run_tinyworlds_calibration(_identity(), evaluator)
    trials = result.validation_trials

    assert tuple(trial.request.purpose for trial in trials) == (
        CalibrationTrialPurpose.BASELINE,
        CalibrationTrialPurpose.FACTS,
        CalibrationTrialPurpose.FACTS,
        CalibrationTrialPurpose.EXPOSURES,
        CalibrationTrialPurpose.EXPOSURES,
        CalibrationTrialPurpose.UPDATES,
        CalibrationTrialPurpose.UPDATES,
        CalibrationTrialPurpose.RANK,
        CalibrationTrialPurpose.RANK,
        CalibrationTrialPurpose.STANDARD_POLICY,
        CalibrationTrialPurpose.LOCKED_SCRATCH,
    )
    profile = result.require_pilot_profile()
    assert profile.selected_config.facts_per_task == 36
    assert profile.selected_config.exposures_per_fact == 16
    assert profile.selected_config.update_budget == 500
    assert profile.selected_config.lora_rank == 4
    assert profile.selected_config.distractor_policy.value == "hard"

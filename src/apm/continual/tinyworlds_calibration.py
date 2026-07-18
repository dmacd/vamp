"""Fixed, validation-only calibration control plane for TinyWorlds v1.

This module intentionally knows nothing about JAX or model execution.  A trial
evaluator is injected by the accelerator runner, while the ladder ordering,
selection policy, statistical gates, and test-data boundary remain pure and
mechanically testable on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
import re
from typing import Protocol

from apm.continual.language_benchmarks import (
    WilsonConfidenceInterval,
    wilson_95_confidence_interval,
)


_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_CANONICAL_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]*")


class CalibrationDistractorPolicy(str, Enum):
    """The two fixed distractor policies compared by calibration."""

    HARD = "hard"
    STANDARD_MIX = "standard_mix"


# The standard policy cycles the omitted distractor role across semantic query
# groups.  It is fixed data, not a runner switch.
STANDARD_DISTRACTOR_ROLE_CYCLE = (
    "incompatible_revision",
    "competing_task",
    "partial_proof",
    "same_type_filler",
)


@dataclass(frozen=True, slots=True)
class TinyWorldsCalibrationConfig:
    """One point in the bounded calibration ladder."""

    facts_per_task: int
    exposures_per_fact: int
    update_budget: int
    lora_rank: int
    distractor_policy: CalibrationDistractorPolicy

    def __post_init__(self) -> None:
        for label, value in (
            ("facts_per_task", self.facts_per_task),
            ("exposures_per_fact", self.exposures_per_fact),
            ("update_budget", self.update_budget),
            ("lora_rank", self.lora_rank),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{label} must be a positive integer")
        if type(self.distractor_policy) is not CalibrationDistractorPolicy:
            raise TypeError(
                "distractor_policy must be a CalibrationDistractorPolicy"
            )


BASELINE_CALIBRATION_CONFIG = TinyWorldsCalibrationConfig(
    facts_per_task=24,
    exposures_per_fact=32,
    update_budget=1_000,
    lora_rank=8,
    distractor_policy=CalibrationDistractorPolicy.HARD,
)
FACT_LADDER = (12, 24, 36)
EXPOSURE_LADDER = (16, 32, 64)
UPDATE_LADDER = (500, 1_000, 2_000)
RANK_LADDER = (4, 8, 16)
MAX_CALIBRATION_VALIDATION_TRIALS = 11


@dataclass(frozen=True, slots=True)
class CalibrationIdentity:
    """Immutable identities that bind a profile to one benchmark setup."""

    benchmark_version: str
    public_seed: int
    calibration_bundle_sha256: str
    base_manifest_sha256: str
    base_parameter_checksum: str
    tokenizer_sha256: str

    def __post_init__(self) -> None:
        if type(self.benchmark_version) is not str or not self.benchmark_version:
            raise ValueError("benchmark_version must be a nonempty string")
        if type(self.public_seed) is not int or self.public_seed < 0:
            raise ValueError("public_seed must be a nonnegative integer")
        for label, value in (
            ("calibration_bundle_sha256", self.calibration_bundle_sha256),
            ("base_manifest_sha256", self.base_manifest_sha256),
            ("base_parameter_checksum", self.base_parameter_checksum),
            ("tokenizer_sha256", self.tokenizer_sha256),
        ):
            _require_sha256(value, label)


@dataclass(frozen=True, slots=True)
class CalibrationBinomialEvidence:
    """Exact counts and their validated two-sided 95% Wilson interval."""

    successes: int
    trials: int
    confidence_interval: WilsonConfidenceInterval

    def __post_init__(self) -> None:
        if type(self.confidence_interval) is not WilsonConfidenceInterval:
            raise TypeError("confidence_interval must be a Wilson interval")
        expected = wilson_95_confidence_interval(self.successes, self.trials)
        if self.confidence_interval != expected:
            raise ValueError("confidence_interval must match the exact counts")

    @property
    def rate(self) -> float:
        """Return successes divided by trials."""
        return self.confidence_interval.observed_rate


def calibration_binomial_evidence(
    successes: int,
    trials: int,
) -> CalibrationBinomialEvidence:
    """Construct exact binomial evidence with a validated Wilson interval."""
    return CalibrationBinomialEvidence(
        successes=successes,
        trials=trials,
        confidence_interval=wilson_95_confidence_interval(successes, trials),
    )


@dataclass(frozen=True, slots=True)
class CommittedNodeSnapshot:
    """Checksums sufficient to prove one committed node remained bit-identical."""

    node_id: str
    adapter_sha256: str
    logits_sha256: str
    answers_sha256: str

    def __post_init__(self) -> None:
        _require_canonical_id(self.node_id, "node_id")
        _require_sha256(self.adapter_sha256, "adapter_sha256")
        _require_sha256(self.logits_sha256, "logits_sha256")
        _require_sha256(self.answers_sha256, "answers_sha256")


@dataclass(frozen=True, slots=True)
class CommittedNodeStabilityEvidence:
    """Before/after committed-node snapshots in the same canonical node order."""

    before: tuple[CommittedNodeSnapshot, ...]
    after: tuple[CommittedNodeSnapshot, ...]

    def __post_init__(self) -> None:
        for label, values in (("before", self.before), ("after", self.after)):
            if type(values) is not tuple or not values or any(
                type(value) is not CommittedNodeSnapshot for value in values
            ):
                raise ValueError(
                    f"stability {label} must contain committed-node snapshots"
                )
            ids = tuple(value.node_id for value in values)
            if len(set(ids)) != len(ids):
                raise ValueError(f"stability {label} node IDs must be unique")
        if tuple(item.node_id for item in self.before) != tuple(
            item.node_id for item in self.after
        ):
            raise ValueError("before and after snapshots must align by node ID")

    @property
    def bit_identical(self) -> bool:
        """Whether adapters, logits, and answers all retained exact checksums."""
        return self.before == self.after


@dataclass(frozen=True, slots=True)
class TinyWorldsCalibrationEvidence:
    """Complete validation or test evidence for all fixed calibration gates."""

    exact_kg: CalibrationBinomialEvidence
    frozen_novel_binding: CalibrationBinomialEvidence
    independent_direct_recall: CalibrationBinomialEvidence
    frozen_one_hop: CalibrationBinomialEvidence
    independent_one_hop: CalibrationBinomialEvidence
    committed_node_stability: CommittedNodeStabilityEvidence
    old_contextual_answer: CalibrationBinomialEvidence
    revision_contextual_answer: CalibrationBinomialEvidence
    paired_revision_consistency: CalibrationBinomialEvidence

    def __post_init__(self) -> None:
        metric_values = (
            self.exact_kg,
            self.frozen_novel_binding,
            self.independent_direct_recall,
            self.frozen_one_hop,
            self.independent_one_hop,
            self.old_contextual_answer,
            self.revision_contextual_answer,
            self.paired_revision_consistency,
        )
        if any(type(value) is not CalibrationBinomialEvidence for value in metric_values):
            raise TypeError("calibration metrics must be binomial evidence")
        if type(self.committed_node_stability) is not CommittedNodeStabilityEvidence:
            raise TypeError("committed_node_stability has the wrong type")


class CalibrationGateName(str, Enum):
    """Every immutable validation decision in gate order."""

    EXACT_KG = "exact_kg_100_percent"
    FROZEN_RATE = "frozen_novel_binding_20_to_30_percent"
    FROZEN_WILSON = "frozen_wilson_contains_25_percent"
    DIRECT_RECALL = "independent_direct_recall_at_least_75_percent"
    DIRECT_LIFT = "independent_direct_lift_at_least_30_points"
    ONE_HOP = "independent_one_hop_at_least_60_percent"
    ONE_HOP_WILSON = "one_hop_wilson_separation"
    COMMITTED_STABILITY = "committed_nodes_bit_identical"
    OLD_CONTEXT = "old_contextual_answer_at_least_75_percent"
    REVISION_CONTEXT = "revision_contextual_answer_at_least_75_percent"
    REVISION_CONSISTENCY = "paired_revision_consistency_at_least_60_percent"


@dataclass(frozen=True, slots=True)
class CalibrationGateDecision:
    """One derived calibration decision and the values used to make it."""

    gate: CalibrationGateName
    passed: bool
    observed: tuple[float, ...]
    criterion: str

    def __post_init__(self) -> None:
        if type(self.gate) is not CalibrationGateName:
            raise TypeError("gate must be a CalibrationGateName")
        if type(self.passed) is not bool:
            raise TypeError("passed must be a bool")
        if type(self.observed) is not tuple or not self.observed or any(
            type(value) is not float or not math.isfinite(value)
            for value in self.observed
        ):
            raise ValueError("gate observations must be finite float values")
        if type(self.criterion) is not str or not self.criterion:
            raise ValueError("gate criterion must be nonempty")


def calibration_gate_decisions(
    evidence: TinyWorldsCalibrationEvidence,
) -> tuple[CalibrationGateDecision, ...]:
    """Evaluate all fixed Phase 4 validation gates without side effects."""
    if type(evidence) is not TinyWorldsCalibrationEvidence:
        raise TypeError("evidence must be TinyWorldsCalibrationEvidence")
    frozen = evidence.frozen_novel_binding
    direct = evidence.independent_direct_recall
    frozen_hop = evidence.frozen_one_hop
    independent_hop = evidence.independent_one_hop
    return (
        _decision(
            CalibrationGateName.EXACT_KG,
            evidence.exact_kg.successes == evidence.exact_kg.trials,
            (evidence.exact_kg.rate,),
            "rate == 1.0",
        ),
        _decision(
            CalibrationGateName.FROZEN_RATE,
            0.20 <= frozen.rate <= 0.30,
            (frozen.rate,),
            "0.20 <= rate <= 0.30",
        ),
        _decision(
            CalibrationGateName.FROZEN_WILSON,
            frozen.confidence_interval.lower
            <= 0.25
            <= frozen.confidence_interval.upper,
            (
                frozen.confidence_interval.lower,
                0.25,
                frozen.confidence_interval.upper,
            ),
            "Wilson lower <= 0.25 <= Wilson upper",
        ),
        _decision(
            CalibrationGateName.DIRECT_RECALL,
            direct.rate >= 0.75,
            (direct.rate,),
            "rate >= 0.75",
        ),
        _decision(
            CalibrationGateName.DIRECT_LIFT,
            direct.rate - frozen.rate >= 0.30,
            (direct.rate - frozen.rate,),
            "independent rate - frozen rate >= 0.30",
        ),
        _decision(
            CalibrationGateName.ONE_HOP,
            independent_hop.rate >= 0.60,
            (independent_hop.rate,),
            "rate >= 0.60",
        ),
        _decision(
            CalibrationGateName.ONE_HOP_WILSON,
            independent_hop.confidence_interval.lower
            > frozen_hop.confidence_interval.upper,
            (
                independent_hop.confidence_interval.lower,
                frozen_hop.confidence_interval.upper,
            ),
            "independent Wilson lower > frozen Wilson upper",
        ),
        _decision(
            CalibrationGateName.COMMITTED_STABILITY,
            evidence.committed_node_stability.bit_identical,
            (1.0 if evidence.committed_node_stability.bit_identical else 0.0,),
            "all before/after checksums identical",
        ),
        _decision(
            CalibrationGateName.OLD_CONTEXT,
            evidence.old_contextual_answer.rate >= 0.75,
            (evidence.old_contextual_answer.rate,),
            "rate >= 0.75",
        ),
        _decision(
            CalibrationGateName.REVISION_CONTEXT,
            evidence.revision_contextual_answer.rate >= 0.75,
            (evidence.revision_contextual_answer.rate,),
            "rate >= 0.75",
        ),
        _decision(
            CalibrationGateName.REVISION_CONSISTENCY,
            evidence.paired_revision_consistency.rate >= 0.60,
            (evidence.paired_revision_consistency.rate,),
            "rate >= 0.60",
        ),
    )


def calibration_evidence_passes(evidence: TinyWorldsCalibrationEvidence) -> bool:
    """Whether every validation gate passes."""
    return all(decision.passed for decision in calibration_gate_decisions(evidence))


class CalibrationTrialPurpose(str, Enum):
    """Fixed reason for each validation trial."""

    BASELINE = "baseline"
    FACTS = "facts"
    EXPOSURES = "exposures"
    UPDATES = "updates"
    RANK = "rank"
    STANDARD_POLICY = "standard_policy"
    LOCKED_SCRATCH = "locked_scratch"


@dataclass(frozen=True, slots=True)
class CalibrationValidationRequest:
    """One validation-only trial request in the canonical ladder."""

    trial_index: int
    purpose: CalibrationTrialPurpose
    config: TinyWorldsCalibrationConfig
    locked_scratch_rerun: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.trial_index) is not int
            or not 0 <= self.trial_index < MAX_CALIBRATION_VALIDATION_TRIALS
        ):
            raise ValueError("trial_index must be from zero through ten")
        if type(self.purpose) is not CalibrationTrialPurpose:
            raise TypeError("purpose must be a CalibrationTrialPurpose")
        if type(self.config) is not TinyWorldsCalibrationConfig:
            raise TypeError("config must be a TinyWorldsCalibrationConfig")
        if type(self.locked_scratch_rerun) is not bool:
            raise TypeError("locked_scratch_rerun must be a bool")
        if self.locked_scratch_rerun != (
            self.purpose is CalibrationTrialPurpose.LOCKED_SCRATCH
        ):
            raise ValueError("only the locked trial may be the scratch rerun")


@dataclass(frozen=True, slots=True)
class CalibrationValidationObservation:
    """Evaluator output before the orchestrator binds it to its request."""

    artifact_id: str
    execution_sha256: str
    artifact_sha256: str
    evidence: TinyWorldsCalibrationEvidence

    def __post_init__(self) -> None:
        _require_canonical_id(self.artifact_id, "artifact_id")
        _require_sha256(self.execution_sha256, "execution_sha256")
        _require_sha256(self.artifact_sha256, "artifact_sha256")
        if type(self.evidence) is not TinyWorldsCalibrationEvidence:
            raise TypeError("evidence must be TinyWorldsCalibrationEvidence")


@dataclass(frozen=True, slots=True)
class CalibrationValidationTrial:
    """One immutable request/evidence pair in ladder order."""

    request: CalibrationValidationRequest
    artifact_id: str
    execution_sha256: str
    artifact_sha256: str
    evidence: TinyWorldsCalibrationEvidence

    def __post_init__(self) -> None:
        if type(self.request) is not CalibrationValidationRequest:
            raise TypeError("request must be a CalibrationValidationRequest")
        _require_canonical_id(self.artifact_id, "artifact_id")
        _require_sha256(self.execution_sha256, "execution_sha256")
        _require_sha256(self.artifact_sha256, "artifact_sha256")
        if type(self.evidence) is not TinyWorldsCalibrationEvidence:
            raise TypeError("evidence must be TinyWorldsCalibrationEvidence")

    @property
    def passed(self) -> bool:
        """Whether this trial meets every validation gate."""
        return calibration_evidence_passes(self.evidence)

    @property
    def gate_decisions(self) -> tuple[CalibrationGateDecision, ...]:
        """Return the mechanically derived gate decisions."""
        return calibration_gate_decisions(self.evidence)


@dataclass(frozen=True, slots=True)
class LockedCalibrationTestRequest:
    """The sole test request, bound to the passing locked scratch artifact."""

    config: TinyWorldsCalibrationConfig
    validation_trial_index: int
    validation_artifact_id: str

    def __post_init__(self) -> None:
        if type(self.config) is not TinyWorldsCalibrationConfig:
            raise TypeError("config must be a TinyWorldsCalibrationConfig")
        if self.validation_trial_index != MAX_CALIBRATION_VALIDATION_TRIALS - 1:
            raise ValueError("test evaluation must use locked validation trial ten")
        _require_canonical_id(self.validation_artifact_id, "validation_artifact_id")


@dataclass(frozen=True, slots=True)
class LockedCalibrationTestObservation:
    """Evaluator output for the one post-lock test evaluation."""

    artifact_id: str
    execution_sha256: str
    artifact_sha256: str
    evidence: TinyWorldsCalibrationEvidence

    def __post_init__(self) -> None:
        _require_canonical_id(self.artifact_id, "test artifact_id")
        _require_sha256(self.execution_sha256, "test execution_sha256")
        _require_sha256(self.artifact_sha256, "test artifact_sha256")
        if type(self.evidence) is not TinyWorldsCalibrationEvidence:
            raise TypeError("test evidence must be TinyWorldsCalibrationEvidence")


@dataclass(frozen=True, slots=True)
class LockedCalibrationTest:
    """The one test result retained in a successful calibration profile."""

    request: LockedCalibrationTestRequest
    artifact_id: str
    execution_sha256: str
    artifact_sha256: str
    evidence: TinyWorldsCalibrationEvidence

    def __post_init__(self) -> None:
        if type(self.request) is not LockedCalibrationTestRequest:
            raise TypeError("request must be a LockedCalibrationTestRequest")
        _require_canonical_id(self.artifact_id, "test artifact_id")
        _require_sha256(self.execution_sha256, "test execution_sha256")
        _require_sha256(self.artifact_sha256, "test artifact_sha256")
        if type(self.evidence) is not TinyWorldsCalibrationEvidence:
            raise TypeError("test evidence must be TinyWorldsCalibrationEvidence")


class TinyWorldsCalibrationTrialEvaluator(Protocol):
    """Injected accelerator evaluator with an explicit test-data boundary."""

    def evaluate_validation(
        self,
        request: CalibrationValidationRequest,
    ) -> CalibrationValidationObservation:
        """Train/evaluate one configuration using calibration validation only."""

    def evaluate_locked_test(
        self,
        request: LockedCalibrationTestRequest,
    ) -> LockedCalibrationTestObservation:
        """Evaluate the already-trained locked scratch artifact on test once."""


@dataclass(frozen=True, slots=True)
class TinyWorldsCalibrationProfile:
    """Selected values plus all tuning, scratch-validation, and test evidence."""

    identity: CalibrationIdentity
    selected_config: TinyWorldsCalibrationConfig
    ladder_trials: tuple[CalibrationValidationTrial, ...]
    locked_scratch_trial: CalibrationValidationTrial
    locked_test: LockedCalibrationTest

    def __post_init__(self) -> None:
        if type(self.identity) is not CalibrationIdentity:
            raise TypeError("identity must be a CalibrationIdentity")
        if type(self.selected_config) is not TinyWorldsCalibrationConfig:
            raise TypeError("selected_config has the wrong type")
        if (
            type(self.ladder_trials) is not tuple
            or len(self.ladder_trials) != 10
            or any(
                type(trial) is not CalibrationValidationTrial
                for trial in self.ladder_trials
            )
        ):
            raise ValueError("successful profiles require exactly ten ladder trials")
        if tuple(trial.request.trial_index for trial in self.ladder_trials) != tuple(
            range(10)
        ):
            raise ValueError("ladder trials must use canonical indices zero through nine")
        if len({trial.artifact_id for trial in self.ladder_trials}) != 10:
            raise ValueError("ladder artifact IDs must be unique")
        replayed = _replay_successful_ladder(self.ladder_trials)
        if replayed != self.selected_config:
            raise ValueError("selected_config does not match fixed ladder selection")
        if type(self.locked_scratch_trial) is not CalibrationValidationTrial:
            raise TypeError("locked_scratch_trial has the wrong type")
        scratch = self.locked_scratch_trial
        if (
            scratch.request.trial_index != 10
            or scratch.request.purpose is not CalibrationTrialPurpose.LOCKED_SCRATCH
            or not scratch.request.locked_scratch_rerun
            or scratch.request.config != self.selected_config
            or not scratch.passed
        ):
            raise ValueError(
                "locked scratch trial must rerun the selected config and pass"
            )
        if scratch.artifact_id in {trial.artifact_id for trial in self.ladder_trials}:
            raise ValueError("locked scratch artifact must be a fresh artifact")
        if type(self.locked_test) is not LockedCalibrationTest:
            raise TypeError("locked_test has the wrong type")
        if (
            self.locked_test.request.config != self.selected_config
            or self.locked_test.request.validation_trial_index != 10
            or self.locked_test.request.validation_artifact_id != scratch.artifact_id
        ):
            raise ValueError("locked test must consume the passing scratch artifact")


class CalibrationStopReason(str, Enum):
    """Why calibration did not produce a pilot-authorizing profile."""

    FACTS_NO_PASS = "facts_axis_has_no_passing_configuration"
    EXPOSURES_NO_PASS = "exposures_axis_has_no_passing_configuration"
    UPDATES_NO_PASS = "updates_axis_has_no_passing_configuration"
    RANK_NO_PASS = "rank_axis_has_no_passing_configuration"
    POLICY_NO_PASS = "distractor_policy_has_no_passing_configuration"
    LOCKED_SCRATCH_FAILED = "locked_scratch_validation_failed"


class CalibrationNotAuthorizedError(RuntimeError):
    """Raised when a caller attempts a pilot without a passing profile."""


@dataclass(frozen=True, slots=True)
class TinyWorldsCalibrationResult:
    """Complete successful or early-stopped orchestration result."""

    identity: CalibrationIdentity
    validation_trials: tuple[CalibrationValidationTrial, ...]
    profile: TinyWorldsCalibrationProfile | None
    stop_reason: CalibrationStopReason | None

    def __post_init__(self) -> None:
        if type(self.identity) is not CalibrationIdentity:
            raise TypeError("identity must be a CalibrationIdentity")
        if type(self.validation_trials) is not tuple or not self.validation_trials:
            raise ValueError("calibration result requires validation trials")
        if any(
            type(trial) is not CalibrationValidationTrial
            for trial in self.validation_trials
        ):
            raise TypeError("validation_trials contain an invalid value")
        if len(self.validation_trials) > MAX_CALIBRATION_VALIDATION_TRIALS:
            raise ValueError("calibration exceeded its eleven-trial bound")
        if tuple(trial.request.trial_index for trial in self.validation_trials) != tuple(
            range(len(self.validation_trials))
        ):
            raise ValueError("validation trials must retain contiguous canonical order")
        artifact_ids = tuple(trial.artifact_id for trial in self.validation_trials)
        if len(set(artifact_ids)) != len(artifact_ids):
            raise ValueError("calibration result artifact IDs must be unique")
        if (self.profile is None) == (self.stop_reason is None):
            raise ValueError("result must contain exactly one of profile or stop_reason")
        if self.profile is not None:
            if type(self.profile) is not TinyWorldsCalibrationProfile:
                raise TypeError("profile has the wrong type")
            if self.profile.identity != self.identity:
                raise ValueError("profile identity must match the result")
            expected = self.profile.ladder_trials + (self.profile.locked_scratch_trial,)
            if self.validation_trials != expected:
                raise ValueError("successful result trials must match the profile")
        elif type(self.stop_reason) is not CalibrationStopReason:
            raise TypeError("stop_reason has the wrong type")
        else:
            expected_stop = _replay_stopped_ladder(self.validation_trials)
            if self.stop_reason is not expected_stop:
                raise ValueError(
                    "stop_reason does not match the fixed ladder evidence"
                )

    @property
    def pilot_authorized(self) -> bool:
        """Whether Phase 5 may consume this result."""
        return self.profile is not None

    def require_pilot_profile(self) -> TinyWorldsCalibrationProfile:
        """Return the selected profile or reject an unauthorized pilot launch."""
        if self.profile is None:
            raise CalibrationNotAuthorizedError(
                f"TinyWorlds pilot is not authorized: {self.stop_reason.value}"
            )
        return self.profile


def run_tinyworlds_calibration(
    identity: CalibrationIdentity,
    evaluator: TinyWorldsCalibrationTrialEvaluator,
) -> TinyWorldsCalibrationResult:
    """Run the fixed at-most-eleven validation ladder and one post-lock test."""
    if type(identity) is not CalibrationIdentity:
        raise TypeError("identity must be a CalibrationIdentity")
    trials: list[CalibrationValidationTrial] = []

    current = _evaluate_trial(
        evaluator,
        trials,
        CalibrationTrialPurpose.BASELINE,
        BASELINE_CALIBRATION_CONFIG,
    )
    selection = _evaluate_numeric_axis(
        evaluator,
        trials,
        current,
        purpose=CalibrationTrialPurpose.FACTS,
        field_name="facts_per_task",
        values=FACT_LADDER,
        prefer_largest=True,
    )
    if selection is None:
        return _stopped(identity, trials, CalibrationStopReason.FACTS_NO_PASS)
    current = selection

    selection = _evaluate_numeric_axis(
        evaluator,
        trials,
        current,
        purpose=CalibrationTrialPurpose.EXPOSURES,
        field_name="exposures_per_fact",
        values=EXPOSURE_LADDER,
        prefer_largest=False,
    )
    if selection is None:
        return _stopped(identity, trials, CalibrationStopReason.EXPOSURES_NO_PASS)
    current = selection

    selection = _evaluate_numeric_axis(
        evaluator,
        trials,
        current,
        purpose=CalibrationTrialPurpose.UPDATES,
        field_name="update_budget",
        values=UPDATE_LADDER,
        prefer_largest=False,
    )
    if selection is None:
        return _stopped(identity, trials, CalibrationStopReason.UPDATES_NO_PASS)
    current = selection

    selection = _evaluate_numeric_axis(
        evaluator,
        trials,
        current,
        purpose=CalibrationTrialPurpose.RANK,
        field_name="lora_rank",
        values=RANK_LADDER,
        prefer_largest=False,
    )
    if selection is None:
        return _stopped(identity, trials, CalibrationStopReason.RANK_NO_PASS)
    current = selection

    standard = _evaluate_trial(
        evaluator,
        trials,
        CalibrationTrialPurpose.STANDARD_POLICY,
        replace(
            current.request.config,
            distractor_policy=CalibrationDistractorPolicy.STANDARD_MIX,
        ),
    )
    if current.passed:
        selected = current
    elif standard.passed:
        selected = standard
    else:
        return _stopped(identity, trials, CalibrationStopReason.POLICY_NO_PASS)

    scratch = _evaluate_trial(
        evaluator,
        trials,
        CalibrationTrialPurpose.LOCKED_SCRATCH,
        selected.request.config,
        locked_scratch_rerun=True,
    )
    if not scratch.passed:
        return _stopped(
            identity,
            trials,
            CalibrationStopReason.LOCKED_SCRATCH_FAILED,
        )

    test_request = LockedCalibrationTestRequest(
        config=selected.request.config,
        validation_trial_index=scratch.request.trial_index,
        validation_artifact_id=scratch.artifact_id,
    )
    test_observation = evaluator.evaluate_locked_test(test_request)
    if type(test_observation) is not LockedCalibrationTestObservation:
        raise TypeError(
            "evaluate_locked_test must return LockedCalibrationTestObservation"
        )
    locked_test = LockedCalibrationTest(
        request=test_request,
        artifact_id=test_observation.artifact_id,
        execution_sha256=test_observation.execution_sha256,
        artifact_sha256=test_observation.artifact_sha256,
        evidence=test_observation.evidence,
    )
    profile = TinyWorldsCalibrationProfile(
        identity=identity,
        selected_config=selected.request.config,
        ladder_trials=tuple(trials[:10]),
        locked_scratch_trial=scratch,
        locked_test=locked_test,
    )
    return TinyWorldsCalibrationResult(
        identity=identity,
        validation_trials=tuple(trials),
        profile=profile,
        stop_reason=None,
    )


def _evaluate_numeric_axis(
    evaluator: TinyWorldsCalibrationTrialEvaluator,
    trials: list[CalibrationValidationTrial],
    current: CalibrationValidationTrial,
    *,
    purpose: CalibrationTrialPurpose,
    field_name: str,
    values: tuple[int, int, int],
    prefer_largest: bool,
) -> CalibrationValidationTrial | None:
    current_value = getattr(current.request.config, field_name)
    if current_value not in values:
        raise RuntimeError(f"current config is outside the fixed {field_name} ladder")
    candidates = [current]
    for value in values:
        if value != current_value:
            candidates.append(
                _evaluate_trial(
                    evaluator,
                    trials,
                    purpose,
                    replace(current.request.config, **{field_name: value}),
                )
            )
    passing = [candidate for candidate in candidates if candidate.passed]
    if not passing:
        return None
    return (max if prefer_largest else min)(
        passing,
        key=lambda trial: getattr(trial.request.config, field_name),
    )


def _evaluate_trial(
    evaluator: TinyWorldsCalibrationTrialEvaluator,
    trials: list[CalibrationValidationTrial],
    purpose: CalibrationTrialPurpose,
    config: TinyWorldsCalibrationConfig,
    *,
    locked_scratch_rerun: bool = False,
) -> CalibrationValidationTrial:
    request = CalibrationValidationRequest(
        trial_index=len(trials),
        purpose=purpose,
        config=config,
        locked_scratch_rerun=locked_scratch_rerun,
    )
    observation = evaluator.evaluate_validation(request)
    if type(observation) is not CalibrationValidationObservation:
        raise TypeError(
            "evaluate_validation must return CalibrationValidationObservation"
        )
    if observation.artifact_id in {trial.artifact_id for trial in trials}:
        raise ValueError("every calibration validation trial needs a fresh artifact")
    trial = CalibrationValidationTrial(
        request=request,
        artifact_id=observation.artifact_id,
        execution_sha256=observation.execution_sha256,
        artifact_sha256=observation.artifact_sha256,
        evidence=observation.evidence,
    )
    trials.append(trial)
    return trial


def _replay_successful_ladder(
    trials: tuple[CalibrationValidationTrial, ...],
) -> TinyWorldsCalibrationConfig:
    baseline, facts_low, facts_high = trials[:3]
    if baseline.request.config != BASELINE_CALIBRATION_CONFIG:
        raise ValueError("profile ladder must start from the fixed baseline")
    expected_purposes = (
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
    )
    if tuple(trial.request.purpose for trial in trials) != expected_purposes:
        raise ValueError("profile trials do not follow the fixed ladder purposes")

    current = _select_replayed_axis(
        (baseline, facts_low, facts_high),
        "facts_per_task",
        FACT_LADDER,
        prefer_largest=True,
    )
    current = _select_replayed_axis(
        (current, trials[3], trials[4]),
        "exposures_per_fact",
        EXPOSURE_LADDER,
        prefer_largest=False,
    )
    current = _select_replayed_axis(
        (current, trials[5], trials[6]),
        "update_budget",
        UPDATE_LADDER,
        prefer_largest=False,
    )
    current = _select_replayed_axis(
        (current, trials[7], trials[8]),
        "lora_rank",
        RANK_LADDER,
        prefer_largest=False,
    )
    standard = trials[9]
    expected_standard = replace(
        current.request.config,
        distractor_policy=CalibrationDistractorPolicy.STANDARD_MIX,
    )
    if standard.request.config != expected_standard:
        raise ValueError("standard-policy trial changes more than the policy axis")
    if current.passed:
        return current.request.config
    if standard.passed:
        return standard.request.config
    raise ValueError("successful profile has no passing distractor policy")


def _replay_stopped_ladder(
    trials: tuple[CalibrationValidationTrial, ...],
) -> CalibrationStopReason:
    """Derive the sole valid early-stop reason from immutable trial evidence."""
    expected_purposes = (
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
    if len(trials) not in (3, 5, 7, 9, 10, 11):
        raise ValueError("stopped calibration has a noncanonical trial count")
    if tuple(trial.request.purpose for trial in trials) != expected_purposes[
        : len(trials)
    ]:
        raise ValueError("stopped calibration does not follow the fixed ladder")
    baseline = trials[0]
    if baseline.request.config != BASELINE_CALIBRATION_CONFIG:
        raise ValueError("stopped calibration must start from the fixed baseline")

    current = _optional_replayed_axis(
        (baseline, trials[1], trials[2]),
        "facts_per_task",
        FACT_LADDER,
        prefer_largest=True,
    )
    if current is None:
        if len(trials) != 3:
            raise ValueError("calibration continued after the facts axis failed")
        return CalibrationStopReason.FACTS_NO_PASS
    if len(trials) == 3:
        raise ValueError("calibration stopped despite a passing facts axis")

    current = _optional_replayed_axis(
        (current, trials[3], trials[4]),
        "exposures_per_fact",
        EXPOSURE_LADDER,
        prefer_largest=False,
    )
    if current is None:
        if len(trials) != 5:
            raise ValueError("calibration continued after the exposures axis failed")
        return CalibrationStopReason.EXPOSURES_NO_PASS
    if len(trials) == 5:
        raise ValueError("calibration stopped despite a passing exposures axis")

    current = _optional_replayed_axis(
        (current, trials[5], trials[6]),
        "update_budget",
        UPDATE_LADDER,
        prefer_largest=False,
    )
    if current is None:
        if len(trials) != 7:
            raise ValueError("calibration continued after the updates axis failed")
        return CalibrationStopReason.UPDATES_NO_PASS
    if len(trials) == 7:
        raise ValueError("calibration stopped despite a passing updates axis")

    current = _optional_replayed_axis(
        (current, trials[7], trials[8]),
        "lora_rank",
        RANK_LADDER,
        prefer_largest=False,
    )
    if current is None:
        if len(trials) != 9:
            raise ValueError("calibration continued after the rank axis failed")
        return CalibrationStopReason.RANK_NO_PASS
    if len(trials) == 9:
        raise ValueError("calibration stopped despite a passing rank axis")

    standard = trials[9]
    expected_standard = replace(
        current.request.config,
        distractor_policy=CalibrationDistractorPolicy.STANDARD_MIX,
    )
    if standard.request.config != expected_standard:
        raise ValueError("standard-policy trial changes more than the policy axis")
    selected = current if current.passed else standard if standard.passed else None
    if selected is None:
        if len(trials) != 10:
            raise ValueError("calibration continued after both policies failed")
        return CalibrationStopReason.POLICY_NO_PASS
    if len(trials) == 10:
        raise ValueError("calibration stopped before the locked scratch rerun")

    scratch = trials[10]
    if scratch.request.config != selected.request.config:
        raise ValueError("locked scratch trial does not use the selected config")
    if scratch.passed:
        raise ValueError("a passing locked scratch trial must produce a profile")
    return CalibrationStopReason.LOCKED_SCRATCH_FAILED


def _select_replayed_axis(
    candidates: tuple[CalibrationValidationTrial, ...],
    field_name: str,
    values: tuple[int, int, int],
    *,
    prefer_largest: bool,
) -> CalibrationValidationTrial:
    selected = _optional_replayed_axis(
        candidates,
        field_name,
        values,
        prefer_largest=prefer_largest,
    )
    if selected is None:
        raise ValueError(f"successful profile has no passing {field_name} value")
    return selected


def _optional_replayed_axis(
    candidates: tuple[CalibrationValidationTrial, ...],
    field_name: str,
    values: tuple[int, int, int],
    *,
    prefer_largest: bool,
) -> CalibrationValidationTrial | None:
    if {getattr(item.request.config, field_name) for item in candidates} != set(values):
        raise ValueError(f"profile does not cover fixed {field_name} values")
    anchor = candidates[0].request.config
    for candidate in candidates[1:]:
        for field in (
            "facts_per_task",
            "exposures_per_fact",
            "update_budget",
            "lora_rank",
            "distractor_policy",
        ):
            if field != field_name and getattr(candidate.request.config, field) != getattr(
                anchor, field
            ):
                raise ValueError(f"{field_name} trial changes multiple axes")
    passing = tuple(candidate for candidate in candidates if candidate.passed)
    if not passing:
        return None
    return (max if prefer_largest else min)(
        passing,
        key=lambda trial: getattr(trial.request.config, field_name),
    )


def _stopped(
    identity: CalibrationIdentity,
    trials: list[CalibrationValidationTrial],
    reason: CalibrationStopReason,
) -> TinyWorldsCalibrationResult:
    return TinyWorldsCalibrationResult(
        identity=identity,
        validation_trials=tuple(trials),
        profile=None,
        stop_reason=reason,
    )


def _decision(
    gate: CalibrationGateName,
    passed: bool,
    observed: tuple[float, ...],
    criterion: str,
) -> CalibrationGateDecision:
    return CalibrationGateDecision(
        gate=gate,
        passed=passed,
        observed=observed,
        criterion=criterion,
    )


def _require_sha256(value: object, label: str) -> None:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase hexadecimal SHA-256")


def _require_canonical_id(value: object, label: str) -> None:
    if type(value) is not str or _CANONICAL_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a canonical identifier")


__all__ = [
    "BASELINE_CALIBRATION_CONFIG",
    "CalibrationBinomialEvidence",
    "CalibrationDistractorPolicy",
    "CalibrationGateDecision",
    "CalibrationGateName",
    "CalibrationIdentity",
    "CalibrationNotAuthorizedError",
    "CalibrationStopReason",
    "CalibrationTrialPurpose",
    "CalibrationValidationObservation",
    "CalibrationValidationRequest",
    "CalibrationValidationTrial",
    "CommittedNodeSnapshot",
    "CommittedNodeStabilityEvidence",
    "EXPOSURE_LADDER",
    "FACT_LADDER",
    "LockedCalibrationTest",
    "LockedCalibrationTestObservation",
    "LockedCalibrationTestRequest",
    "MAX_CALIBRATION_VALIDATION_TRIALS",
    "RANK_LADDER",
    "STANDARD_DISTRACTOR_ROLE_CYCLE",
    "TinyWorldsCalibrationConfig",
    "TinyWorldsCalibrationEvidence",
    "TinyWorldsCalibrationProfile",
    "TinyWorldsCalibrationResult",
    "TinyWorldsCalibrationTrialEvaluator",
    "UPDATE_LADDER",
    "calibration_binomial_evidence",
    "calibration_evidence_passes",
    "calibration_gate_decisions",
    "run_tinyworlds_calibration",
]

"""Measured accelerator execution for the fixed TinyWorlds calibration ladder."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Callable, Literal, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np

from apm.continual.knowledge_tasks import KnowledgeQuery
from apm.continual.knowledge_training import (
    KnowledgeCounterfactualTraining,
    KnowledgeParentContext,
    KnowledgeTransferDiagnostics,
    KnowledgeValidationSuite,
    ParentCounterfactualPlan,
    ParentTransferTrialDiagnostic,
    TransferCheckpointDiagnostic,
    commit_selected_counterfactual_edge,
    plan_parent_counterfactuals,
    run_parent_counterfactuals,
    score_knowledge_parent_nodes,
    validate_parent_counterfactual_resume,
)
from apm.continual.language_tasks import CompetenceBatch, RouterBatch
from apm.continual.language_adaptation_artifact import (
    flatten_lora_edge,
    unflatten_lora_edge,
)
from apm.continual.tinyworlds_calibration import (
    CalibrationDistractorPolicy,
    CalibrationIdentity,
    CalibrationTrialPurpose,
    CalibrationValidationObservation,
    CalibrationValidationRequest,
    CommittedNodeSnapshot,
    CommittedNodeStabilityEvidence,
    LockedCalibrationTestObservation,
    LockedCalibrationTestRequest,
    TinyWorldsCalibrationEvidence,
    calibration_binomial_evidence,
)
from apm.continual.tinyworlds_calibration_profile import (
    calibration_artifact_tree_sha256,
)
from apm.data.text.tinyworlds.adapters import (
    PreparedTinyWorldsCurriculum,
    TinyWorldsTrainingDataConfig,
    prepare_tinyworlds_curriculum,
)
from apm.data.text.tinyworlds.closure import answer_query
from apm.data.text.tinyworlds.query_generation import (
    TinyWorldsBundle,
    apply_standard_distractor_mix,
    generate_calibration_bundle,
)
from apm.data.text.tinyworlds.persistence import tinyworlds_bundle_sha256
from apm.data.text.tinyworlds.rendering import (
    RenderedQueryGroup,
    RenderedTinyWorlds,
    TinyWorldsRenderPreset,
    render_tinyworlds_bundle,
    render_tinyworlds_query_groups,
)
from apm.data.text.tinyworlds.schema import (
    CandidateRole,
    DataSplit,
    QueryKind,
    QueryPlan,
    TaskId as SymbolicTaskId,
    TaskKind,
)
from apm.lm.candidate_scoring import (
    score_edge_coefficient_candidates,
    score_frozen_base_candidates,
    score_hard_node_candidates,
)
from apm.lm.checkpoint import BaseCheckpointRef, parameter_checksum
from apm.lm.config import GptNeoConfig
from apm.lm.gpt_neo import apply_gpt_neo
from apm.lm.lora import LoraConfig, LoraEdge, init_lora_edge
from apm.lm.lora_memory import (
    PackedLoraMemory,
    edge_coefficients_for_node,
    pack_lora_memory,
    packed_with_candidate_edge,
)
from apm.lm.parameters import GptNeoParams
from apm.lm.text import TextTokenizer
from apm.lm.text_data import TokenBatch
from apm.lm.training import (
    LmTrainConfig,
    LmTrainState,
    init_candidate_lora_train_state,
)
from apm.lm.training_state_artifact import (
    lm_train_state_checksum,
    load_lm_train_state_artifact,
    write_lm_train_state_artifact,
)
from apm.lm.workflow import run_resumable_candidate_edge_updates
from apm.memory.graph import (
    MemoryGraph,
    NodeId,
    TaskId,
    add_memory_node,
    init_memory_graph,
    memory_node_ids,
)


CALIBRATION_ACCELERATOR_ARTIFACT_VERSION = "2"
CALIBRATION_MAX_FACTS_PER_TASK = 36
CALIBRATION_MAX_EXPOSURES_PER_FACT = 64
CALIBRATION_ALLOCATOR_PEAK_TARGET_BYTES = 12 * 1024**3
CALIBRATION_REQUIRED_PLATFORM = "gpu"
CALIBRATION_REQUIRED_DEVICE_KIND = "NVIDIA GeForce RTX 4090"
CALIBRATION_RENDER_PRESET = TinyWorldsRenderPreset(
    training_stories_per_task=(
        CALIBRATION_MAX_FACTS_PER_TASK * CALIBRATION_MAX_EXPOSURES_PER_FACT
    ),
    validation_stories_per_task=128,
    test_stories_per_task=128,
    validation_query_groups_per_task=256,
    test_query_groups_per_task=512,
    root_validation_stories=128,
    story_token_count=256,
    context_length=256,
)
_ARTIFACT_FORMAT = "apm.tinyworlds.calibration-trial"
_ARTIFACT_SCHEMA_VERSION = 2
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_RESUME_DIRECTORY_PATTERN = re.compile(r"update-[0-9]{7}\Z")
_RESUME_TEMP_PATTERN = re.compile(r"\.update-[0-9]{7}\.tmp-[A-Za-z0-9_-]+\Z")
_TRAINING_CHUNK_PATTERN = re.compile(r"chunk-[0-9]{3}\Z")
_TRAINING_CHUNK_TEMP_PATTERN = re.compile(r"\.chunk\.tmp-[A-Za-z0-9_-]+\Z")


CalibrationSplit: TypeAlias = Literal["validation", "test"]
CalibrationRuntimeEventSink: TypeAlias = Callable[[dict[str, object]], None]


@dataclass(frozen=True, slots=True)
class CalibrationResourceEvidence:
    """Allocator and device evidence captured around one calibration artifact."""

    platform: str
    device_kind: str
    allocator_peak_bytes: int | None
    allocator_peak_target_bytes: int | None

    def __post_init__(self) -> None:
        if type(self.platform) is not str or not self.platform:
            raise ValueError("resource platform must be a nonempty string")
        if type(self.device_kind) is not str or not self.device_kind:
            raise ValueError("resource device kind must be a nonempty string")
        if self.allocator_peak_bytes is not None and (
            type(self.allocator_peak_bytes) is not int
            or self.allocator_peak_bytes < 0
        ):
            raise ValueError("allocator peak must be nonnegative when supplied")
        if self.allocator_peak_target_bytes is not None and (
            type(self.allocator_peak_target_bytes) is not int
            or self.allocator_peak_target_bytes <= 0
        ):
            raise ValueError("allocator peak target must be positive when supplied")


CalibrationResourceProbe: TypeAlias = Callable[
    [int | None],
    CalibrationResourceEvidence,
]


def validate_calibration_resource_evidence(
    evidence: CalibrationResourceEvidence,
    *,
    expected_target_bytes: int | None,
) -> None:
    """Enforce the canonical accelerator contract or the explicit CPU-test seam."""
    if type(evidence) is not CalibrationResourceEvidence:
        raise TypeError("resource evidence has the wrong type")
    if expected_target_bytes is not None and (
        type(expected_target_bytes) is not int or expected_target_bytes <= 0
    ):
        raise ValueError("expected allocator target must be positive when supplied")
    if evidence.allocator_peak_target_bytes != expected_target_bytes:
        raise ValueError("resource evidence allocator target changed")
    if expected_target_bytes is None:
        return
    if (
        evidence.platform != CALIBRATION_REQUIRED_PLATFORM
        or evidence.device_kind != CALIBRATION_REQUIRED_DEVICE_KIND
    ):
        raise RuntimeError(
            "canonical calibration requires one NVIDIA GeForce RTX 4090 GPU"
        )
    if evidence.allocator_peak_bytes is None:
        raise RuntimeError("accelerator allocator peak statistics are unavailable")
    if evidence.allocator_peak_bytes > expected_target_bytes:
        raise MemoryError(
            f"allocator peak {evidence.allocator_peak_bytes} exceeds "
            f"calibration target {expected_target_bytes}"
        )


def measure_calibration_resource_evidence(
    target_bytes: int | None,
) -> CalibrationResourceEvidence:
    """Read the selected local JAX device and its process allocator peak."""
    devices = jax.local_devices()
    if not devices:
        raise RuntimeError("JAX reported no local device for calibration")
    device = devices[0]
    stats = device.memory_stats()
    peak = (
        None
        if stats is None or "peak_bytes_in_use" not in stats
        else int(stats["peak_bytes_in_use"])
    )
    evidence = CalibrationResourceEvidence(
        platform=device.platform,
        device_kind=device.device_kind,
        allocator_peak_bytes=peak,
        allocator_peak_target_bytes=target_bytes,
    )
    validate_calibration_resource_evidence(
        evidence,
        expected_target_bytes=target_bytes,
    )
    return evidence


def _capture_calibration_resource_evidence(
    probe: CalibrationResourceProbe,
    target_bytes: int | None,
) -> CalibrationResourceEvidence:
    evidence = probe(target_bytes)
    validate_calibration_resource_evidence(
        evidence,
        expected_target_bytes=target_bytes,
    )
    return evidence


def _require_same_resource_identity(
    observed: CalibrationResourceEvidence,
    runtime: CalibrationResourceEvidence,
) -> None:
    validate_calibration_resource_evidence(
        observed,
        expected_target_bytes=runtime.allocator_peak_target_bytes,
    )
    validate_calibration_resource_evidence(
        runtime,
        expected_target_bytes=runtime.allocator_peak_target_bytes,
    )
    if (
        observed.platform,
        observed.device_kind,
        observed.allocator_peak_target_bytes,
    ) != (
        runtime.platform,
        runtime.device_kind,
        runtime.allocator_peak_target_bytes,
    ):
        raise ValueError(
            "calibration artifact resource identity differs from this runtime"
        )


@dataclass(frozen=True, slots=True)
class TinyWorldsCalibrationExecutionPreset:
    """Model execution policy, injectable only for bounded integration tests."""

    batch_size: int = 32
    evaluation_examples_per_task: int = 128
    evaluation_microbatch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    evidence_prefix_length: int = 64
    parent_prefix_length: int = 64
    allocator_peak_target_bytes: int | None = (
        CALIBRATION_ALLOCATOR_PEAK_TARGET_BYTES
    )

    def __post_init__(self) -> None:
        integer_values = (
            self.batch_size,
            self.evaluation_examples_per_task,
            self.evaluation_microbatch_size,
            self.evidence_prefix_length,
            self.parent_prefix_length,
        )
        if any(type(value) is not int or value <= 0 for value in integer_values):
            raise ValueError("calibration execution dimensions must be positive")
        for value in (
            self.learning_rate,
            self.weight_decay,
            self.gradient_clip_norm,
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("calibration optimizer values must be finite")
        if self.learning_rate == 0.0 or self.gradient_clip_norm == 0.0:
            raise ValueError("learning rate and gradient clipping must be positive")
        if self.evidence_prefix_length != 64 or self.parent_prefix_length != 64:
            raise ValueError("calibration uses the fixed primary 64-token condition")
        if self.allocator_peak_target_bytes is not None and (
            type(self.allocator_peak_target_bytes) is not int
            or self.allocator_peak_target_bytes <= 0
        ):
            raise ValueError("allocator peak target must be positive when supplied")


CALIBRATION_EXECUTION_PRESET = TinyWorldsCalibrationExecutionPreset()


@dataclass(frozen=True, slots=True)
class TinyWorldsCalibrationPool:
    """One expanded story pool with hard and standard validation/test queries."""

    hard_bundle: TinyWorldsBundle
    standard_bundle: TinyWorldsBundle
    rendered: RenderedTinyWorlds
    standard_query_groups: tuple[RenderedQueryGroup, ...]
    symbolic_bundle_sha256: str
    validation_selection_sha256: str
    content_sha256: str

    def __post_init__(self) -> None:
        if type(self.hard_bundle) is not TinyWorldsBundle or type(
            self.standard_bundle
        ) is not TinyWorldsBundle:
            raise TypeError("calibration pool requires symbolic bundles")
        if type(self.rendered) is not RenderedTinyWorlds:
            raise TypeError("calibration pool requires a rendered story pool")
        if any(
            len(task.direct_fact_ids) != CALIBRATION_MAX_FACTS_PER_TASK
            for task in self.hard_bundle.tasks
        ):
            raise ValueError("calibration pool must contain 36 facts per task")
        if self.hard_bundle.world != self.standard_bundle.world or (
            self.hard_bundle.story_plans != self.standard_bundle.story_plans
        ):
            raise ValueError("distractor policies must share one symbolic world")
        if type(self.standard_query_groups) is not tuple or any(
            type(group) is not RenderedQueryGroup
            for group in self.standard_query_groups
        ):
            raise TypeError("standard_query_groups contain an invalid value")
        hard_group_ids = tuple(group.group_id for group in self.rendered.query_groups)
        standard_group_ids = tuple(group.group_id for group in self.standard_query_groups)
        if hard_group_ids != standard_group_ids:
            raise ValueError("hard and standard query views must align by group ID")
        for value, label in (
            (self.symbolic_bundle_sha256, "symbolic bundle SHA-256"),
            (
                self.validation_selection_sha256,
                "validation-selection SHA-256",
            ),
            (self.content_sha256, "calibration pool SHA-256"),
        ):
            _require_sha256(value, label)
        if self.validation_selection_sha256 != (
            _validation_selection_content_sha256(
                self.hard_bundle,
                self.rendered,
                self.standard_query_groups,
            )
        ):
            raise ValueError("validation-selection content checksum mismatch")
        if self.content_sha256 != _pool_content_sha256(
            self.hard_bundle,
            self.rendered,
            self.standard_query_groups,
            self.symbolic_bundle_sha256,
        ):
            raise ValueError("calibration pool content checksum mismatch")

    def groups(
        self,
        policy: CalibrationDistractorPolicy,
    ) -> tuple[RenderedQueryGroup, ...]:
        """Return the aligned policy-specific query groups."""
        if type(policy) is not CalibrationDistractorPolicy:
            raise TypeError("policy must be a CalibrationDistractorPolicy")
        return (
            self.rendered.query_groups
            if policy is CalibrationDistractorPolicy.HARD
            else self.standard_query_groups
        )

    def bundle(self, policy: CalibrationDistractorPolicy) -> TinyWorldsBundle:
        """Return the policy-specific symbolic candidate plans."""
        if type(policy) is not CalibrationDistractorPolicy:
            raise TypeError("policy must be a CalibrationDistractorPolicy")
        return (
            self.hard_bundle
            if policy is CalibrationDistractorPolicy.HARD
            else self.standard_bundle
        )

    def rendered_for_policy(
        self,
        policy: CalibrationDistractorPolicy,
    ) -> RenderedTinyWorlds:
        """Pair the shared stories with one policy's immutable query view."""
        return RenderedTinyWorlds(
            bundle_id=f"{self.rendered.bundle_id}:{policy.value}",
            registry=self.rendered.registry,
            preset=self.rendered.preset,
            stories=self.rendered.stories,
            query_groups=self.groups(policy),
        )


def build_tinyworlds_calibration_pool(
    master_seed_sha256: str,
    tokenizer: TextTokenizer,
    *,
    render_preset: TinyWorldsRenderPreset = CALIBRATION_RENDER_PRESET,
    symbolic_bundle_sha256: str | None = None,
) -> TinyWorldsCalibrationPool:
    """Generate and render one deterministic expanded calibration data pool."""
    hard_bundle = generate_calibration_bundle(
        master_seed_sha256,
        direct_facts_per_task=CALIBRATION_MAX_FACTS_PER_TASK,
    )
    return render_tinyworlds_calibration_pool(
        hard_bundle,
        tokenizer,
        render_preset=render_preset,
        symbolic_bundle_sha256=symbolic_bundle_sha256,
    )


def render_tinyworlds_calibration_pool(
    hard_bundle: TinyWorldsBundle,
    tokenizer: TextTokenizer,
    *,
    render_preset: TinyWorldsRenderPreset = CALIBRATION_RENDER_PRESET,
    symbolic_bundle_sha256: str | None = None,
) -> TinyWorldsCalibrationPool:
    """Render one already-generated expanded calibration symbolic bundle."""
    if type(hard_bundle) is not TinyWorldsBundle:
        raise TypeError("hard_bundle must be a TinyWorldsBundle")
    if any(
        len(task.direct_fact_ids) != CALIBRATION_MAX_FACTS_PER_TASK
        for task in hard_bundle.tasks
    ):
        raise ValueError("calibration rendering requires 36 facts per task")
    standard_bundle = apply_standard_distractor_mix(hard_bundle)
    rendered = render_tinyworlds_bundle(hard_bundle, tokenizer, render_preset)
    standard_groups = render_tinyworlds_query_groups(
        standard_bundle,
        tokenizer,
        render_preset,
        registry=rendered.registry,
    )
    canonical_symbolic_sha = tinyworlds_bundle_sha256(hard_bundle)
    if symbolic_bundle_sha256 is not None and (
        symbolic_bundle_sha256 != canonical_symbolic_sha
    ):
        raise ValueError("supplied symbolic bundle SHA-256 is not canonical")
    resolved_symbolic_sha = canonical_symbolic_sha
    return TinyWorldsCalibrationPool(
        hard_bundle=hard_bundle,
        standard_bundle=standard_bundle,
        rendered=rendered,
        standard_query_groups=standard_groups,
        symbolic_bundle_sha256=resolved_symbolic_sha,
        validation_selection_sha256=_validation_selection_content_sha256(
            hard_bundle,
            rendered,
            standard_groups,
        ),
        content_sha256=_pool_content_sha256(
            hard_bundle,
            rendered,
            standard_groups,
            resolved_symbolic_sha,
        ),
    )


@dataclass(frozen=True, slots=True)
class _TrainingOutcome:
    independent_adapters: tuple[tuple[TaskId, LoraEdge], ...]
    graph: MemoryGraph[LoraEdge]
    checkpoint_records: tuple[dict[str, object], ...]
    parent_records: tuple[dict[str, object], ...]
    stability_before: tuple[CommittedNodeSnapshot, ...]


class TinyWorldsAcceleratorCalibrationEvaluator:
    """Train, cache, and measure real calibration trials against frozen weights."""

    def __init__(
        self,
        identity: CalibrationIdentity,
        pool: TinyWorldsCalibrationPool,
        tokenizer: TextTokenizer,
        base_checkpoint: BaseCheckpointRef,
        base_params: GptNeoParams,
        model_config: GptNeoConfig,
        artifact_root: str | Path,
        *,
        execution_preset: TinyWorldsCalibrationExecutionPreset = (
            CALIBRATION_EXECUTION_PRESET
        ),
        event_sink: CalibrationRuntimeEventSink | None = None,
        resource_probe: CalibrationResourceProbe | None = None,
    ) -> None:
        if type(identity) is not CalibrationIdentity:
            raise TypeError("identity must be a CalibrationIdentity")
        if type(pool) is not TinyWorldsCalibrationPool:
            raise TypeError("pool must be a TinyWorldsCalibrationPool")
        if not isinstance(tokenizer, TextTokenizer):
            raise TypeError("tokenizer must satisfy TextTokenizer")
        if not isinstance(base_checkpoint, BaseCheckpointRef):
            raise TypeError("base_checkpoint must be a BaseCheckpointRef")
        if not isinstance(model_config, GptNeoConfig):
            raise TypeError("model_config must be a GptNeoConfig")
        if parameter_checksum(base_params, model_config) != (
            base_checkpoint.parameter_checksum
        ):
            raise ValueError("base parameters do not match their checkpoint")
        if identity.calibration_bundle_sha256 != pool.symbolic_bundle_sha256:
            raise ValueError("calibration identity does not match the 36-fact pool")
        if identity.base_manifest_sha256 != base_checkpoint.manifest_sha256 or (
            identity.base_parameter_checksum != base_checkpoint.parameter_checksum
        ):
            raise ValueError("calibration identity does not match the frozen base")
        if type(execution_preset) is not TinyWorldsCalibrationExecutionPreset:
            raise TypeError("execution_preset has the wrong type")
        self.identity = identity
        self.pool = pool
        self.tokenizer = tokenizer
        self.base_checkpoint = base_checkpoint
        self.base_params = base_params
        self.model_config = model_config
        self.execution_preset = execution_preset
        self.resource_probe = (
            measure_calibration_resource_evidence
            if resource_probe is None
            else resource_probe
        )
        if not callable(self.resource_probe):
            raise TypeError("resource_probe must be callable")
        self.runtime_resource_evidence = _capture_calibration_resource_evidence(
            self.resource_probe,
            self.execution_preset.allocator_peak_target_bytes,
        )
        self.artifact_root = Path(artifact_root)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.event_sink = event_sink

    def evaluate_validation(
        self,
        request: CalibrationValidationRequest,
    ) -> CalibrationValidationObservation:
        """Train or reload one validation-only trial and return measured evidence."""
        if type(request) is not CalibrationValidationRequest:
            raise TypeError("request must be a CalibrationValidationRequest")
        execution_sha = self._execution_sha256(request)
        artifact_id = (
            f"calibration-validation-{request.trial_index:02d}-"
            f"{execution_sha[:16]}"
        )
        target = self.artifact_root / "validation" / artifact_id
        if target.exists():
            evidence = _load_trial_artifact(
                target,
                artifact_id,
                execution_sha,
                request_record=_validation_request_record(request),
                runtime_resource_evidence=self.runtime_resource_evidence,
            )[0]
            self._emit(
                "validation_trial_reused",
                trial_index=request.trial_index,
                artifact_id=artifact_id,
            )
            return CalibrationValidationObservation(
                artifact_id=artifact_id,
                execution_sha256=execution_sha,
                artifact_sha256=calibration_artifact_tree_sha256(target),
                evidence=evidence,
            )
        self._emit(
            "validation_trial_started",
            trial_index=request.trial_index,
            artifact_id=artifact_id,
        )
        prepared = self._prepare(request)
        lora_config, train_config = self._model_training_configs(request)
        workspace = self.artifact_root / "work" / artifact_id
        outcome = self._train_or_resume(
            request,
            prepared,
            lora_config,
            train_config,
            workspace,
        )
        evidence, score_records = self._measure_evidence(
            request.config.distractor_policy,
            DataSplit.VALIDATION,
            outcome,
            lora_config,
        )
        resource_evidence = self._capture_resource_evidence()
        _write_trial_artifact(
            target,
            artifact_id=artifact_id,
            execution_sha256=execution_sha,
            request_record=_validation_request_record(request),
            evidence=evidence,
            outcome=outcome,
            model_config=self.model_config,
            lora_config=lora_config,
            score_records=score_records,
            resource_evidence=resource_evidence,
        )
        self._emit(
            "validation_trial_completed",
            trial_index=request.trial_index,
            artifact_id=artifact_id,
        )
        return CalibrationValidationObservation(
            artifact_id=artifact_id,
            execution_sha256=execution_sha,
            artifact_sha256=calibration_artifact_tree_sha256(target),
            evidence=evidence,
        )

    def evaluate_locked_test(
        self,
        request: LockedCalibrationTestRequest,
    ) -> LockedCalibrationTestObservation:
        """Score test exactly once from the persisted locked scratch adapters."""
        if type(request) is not LockedCalibrationTestRequest:
            raise TypeError("request must be a LockedCalibrationTestRequest")
        validation_target = (
            self.artifact_root / "validation" / request.validation_artifact_id
        )
        validation_request = CalibrationValidationRequest(
            trial_index=request.validation_trial_index,
            purpose=CalibrationTrialPurpose.LOCKED_SCRATCH,
            config=request.config,
            locked_scratch_rerun=True,
        )
        execution_sha = self._execution_sha256(validation_request)
        lora_config, _ = self._model_training_configs(validation_request)
        _, outcome = _load_trial_artifact(
            validation_target,
            request.validation_artifact_id,
            execution_sha,
            request_record=_validation_request_record(validation_request),
            model_config=self.model_config,
            lora_config=lora_config,
            runtime_resource_evidence=self.runtime_resource_evidence,
        )
        if outcome is None:
            raise RuntimeError("locked scratch artifact did not retain trained adapters")
        test_execution_sha = self._locked_test_execution_sha256(
            request,
            execution_sha,
        )
        artifact_id = f"calibration-test-locked-{test_execution_sha[:16]}"
        target = self.artifact_root / "test" / artifact_id
        if target.exists():
            evidence = _load_trial_artifact(
                target,
                artifact_id,
                test_execution_sha,
                request_record=_locked_test_request_record(request),
                runtime_resource_evidence=self.runtime_resource_evidence,
            )[0]
            self._emit("locked_test_reused", artifact_id=artifact_id)
            return LockedCalibrationTestObservation(
                artifact_id=artifact_id,
                execution_sha256=test_execution_sha,
                artifact_sha256=calibration_artifact_tree_sha256(target),
                evidence=evidence,
            )
        self._emit("locked_test_started", artifact_id=artifact_id)
        evidence, score_records = self._measure_evidence(
            request.config.distractor_policy,
            DataSplit.TEST,
            outcome,
            lora_config,
        )
        resource_evidence = self._capture_resource_evidence()
        _write_trial_artifact(
            target,
            artifact_id=artifact_id,
            execution_sha256=test_execution_sha,
            request_record=_locked_test_request_record(request),
            evidence=evidence,
            outcome=None,
            model_config=self.model_config,
            lora_config=lora_config,
            score_records=score_records,
            resource_evidence=resource_evidence,
        )
        self._emit("locked_test_completed", artifact_id=artifact_id)
        return LockedCalibrationTestObservation(
            artifact_id=artifact_id,
            execution_sha256=test_execution_sha,
            artifact_sha256=calibration_artifact_tree_sha256(target),
            evidence=evidence,
        )

    def _capture_resource_evidence(self) -> CalibrationResourceEvidence:
        evidence = _capture_calibration_resource_evidence(
            self.resource_probe,
            self.execution_preset.allocator_peak_target_bytes,
        )
        _require_same_resource_identity(
            evidence,
            self.runtime_resource_evidence,
        )
        return evidence

    def _prepare(
        self,
        request: CalibrationValidationRequest,
    ) -> PreparedTinyWorldsCurriculum:
        return prepare_tinyworlds_curriculum(
            self.pool.rendered_for_policy(request.config.distractor_policy),
            self.tokenizer,
            TinyWorldsTrainingDataConfig(
                facts_per_task=request.config.facts_per_task,
                exposures_per_fact=request.config.exposures_per_fact,
                batch_size=self.execution_preset.batch_size,
                context_length=256,
                evaluation_examples_per_task=(
                    self.execution_preset.evaluation_examples_per_task
                ),
            ),
        )

    def _model_training_configs(
        self,
        request: CalibrationValidationRequest,
    ) -> tuple[LoraConfig, LmTrainConfig]:
        return (
            LoraConfig(
                rank=request.config.lora_rank,
                alpha=float(request.config.lora_rank),
            ),
            LmTrainConfig(
                learning_rate=self.execution_preset.learning_rate,
                steps=request.config.update_budget,
                batch_size=self.execution_preset.batch_size,
                weight_decay=self.execution_preset.weight_decay,
                gradient_clip_norm=self.execution_preset.gradient_clip_norm,
            ),
        )

    def _execution_sha256(
        self,
        request: CalibrationValidationRequest,
    ) -> str:
        return _digest_record(
            {
                "artifact_version": CALIBRATION_ACCELERATOR_ARTIFACT_VERSION,
                "base_manifest_sha256": self.base_checkpoint.manifest_sha256,
                "base_parameter_checksum": (
                    self.base_checkpoint.parameter_checksum
                ),
                "execution_preset": _execution_preset_record(
                    self.execution_preset
                ),
                "model_config": _model_config_record(self.model_config),
                "validation_selection_sha256": (
                    self.pool.validation_selection_sha256
                ),
                "request": _validation_request_record(request),
            }
        )

    def _locked_test_execution_sha256(
        self,
        request: LockedCalibrationTestRequest,
        validation_execution_sha256: str,
    ) -> str:
        """Bind locked-test scoring to the complete calibration data pool."""
        _require_sha256(
            validation_execution_sha256,
            "validation execution SHA-256",
        )
        return _digest_record(
            {
                "kind": "locked_test",
                "pool_sha256": self.pool.content_sha256,
                "request": _locked_test_request_record(request),
                "validation_execution_sha256": validation_execution_sha256,
            }
        )

    def _train_or_resume(
        self,
        request: CalibrationValidationRequest,
        prepared: PreparedTinyWorldsCurriculum,
        lora_config: LoraConfig,
        train_config: LmTrainConfig,
        workspace: Path,
    ) -> _TrainingOutcome:
        workspace.mkdir(parents=True, exist_ok=True)
        outcome = _load_latest_training_chunk(
            workspace,
            self._execution_sha256(request),
            self.model_config,
            lora_config,
        ) or _TrainingOutcome(
            independent_adapters=(),
            graph=init_memory_graph(NodeId("root")),
            checkpoint_records=(),
            parent_records=(),
            stability_before=(),
        )
        curriculum_tasks = prepared.language.curriculum.tasks
        if tuple(task.task_id for task in curriculum_tasks) != tuple(
            TaskId(str(task.task_id)) for task in self.pool.hard_bundle.tasks
        ):
            raise ValueError("prepared calibration task order changed")

        for task_index in range(len(outcome.independent_adapters), len(curriculum_tasks)):
            task = curriculum_tasks[task_index]
            validation_suite = self._validation_suite(
                request.config.distractor_policy,
                task.task_id,
            )
            adapter, checkpoint_records = self._train_independent_adapter(
                request,
                task.task_id,
                task.train_batches,
                validation_suite,
                lora_config,
                train_config,
                workspace,
            )
            outcome = _TrainingOutcome(
                independent_adapters=outcome.independent_adapters
                + ((task.task_id, adapter),),
                graph=outcome.graph,
                checkpoint_records=outcome.checkpoint_records
                + checkpoint_records,
                parent_records=outcome.parent_records,
                stability_before=outcome.stability_before,
            )
            _write_training_chunk(
                workspace,
                task_index,
                f"independent:{task.task_id}",
                self._execution_sha256(request),
                outcome,
                self.model_config,
                lora_config,
            )
            self._emit(
                "training_chunk_completed",
                trial_index=request.trial_index,
                stream="independent",
                task_id=str(task.task_id),
            )

        completed_vamp_count = len(outcome.graph.nodes) - 1
        for task_index in range(completed_vamp_count, len(curriculum_tasks)):
            language_task = curriculum_tasks[task_index]
            symbolic_task = self.pool.hard_bundle.world.task(
                SymbolicTaskId(str(language_task.task_id))
            )
            packed = pack_lora_memory(
                outcome.graph,
                self.model_config,
                lora_config,
                max_nodes=len(curriculum_tasks) + 1,
                max_edges=len(curriculum_tasks),
            )
            validation_suite = self._validation_suite(
                request.config.distractor_policy,
                language_task.task_id,
            )
            parent_search = score_knowledge_parent_nodes(
                validation_suite,
                outcome.graph,
                packed,
                self.base_params,
                self.model_config,
                lora_config,
                evaluation_microbatch_size=(
                    self.execution_preset.evaluation_microbatch_size
                ),
            )
            context = KnowledgeParentContext(
                task_id=language_task.task_id,
                family_id=str(symbolic_task.family_id),
                true_parent_node_id=(
                    NodeId("root")
                    if symbolic_task.parent_task_id is None
                    else NodeId(str(symbolic_task.parent_task_id))
                ),
                node_family_ids=tuple(
                    (
                        node.node_id,
                        None
                        if node.trained_task is None
                        else str(
                            self.pool.hard_bundle.world.task(
                                SymbolicTaskId(str(node.trained_task))
                            ).family_id
                        ),
                    )
                    for node in outcome.graph.nodes
                ),
            )
            plan = plan_parent_counterfactuals(parent_search, context)
            initialization_key, training_key = jax.random.split(
                _rng_key(
                    self._execution_sha256(request),
                    f"vamp:{language_task.task_id}",
                )
            )
            initial_state = init_candidate_lora_train_state(
                init_lora_edge(
                    initialization_key,
                    self.model_config,
                    lora_config,
                ),
                training_key,
                train_config,
            )
            trained = self._train_counterfactuals_resumably(
                request,
                workspace,
                plan,
                validation_suite,
                initial_state,
                language_task.train_batches,
                packed,
                lora_config,
                train_config,
            )
            committed_graph = commit_selected_counterfactual_edge(
                outcome.graph,
                trained,
            )
            parent_record = {
                "correct_candidate_nll_by_query_and_node": (
                    parent_search.correct_candidate_nll_by_query_and_node.tolist()
                ),
                "mean_correct_candidate_nll": list(
                    parent_search.mean_correct_candidate_nll
                ),
                "node_ids": [str(value) for value in parent_search.node_ids],
                "selected_node_id": str(parent_search.selected_node_id),
                "selected_node_index": parent_search.selected_node_index,
                "task_id": str(language_task.task_id),
                "validation_query_ids": list(parent_search.validation_query_ids),
                "validation_suite_sha256": (
                    parent_search.validation_suite_sha256
                ),
            }
            transfer_records = tuple(
                {
                    "adapter_checksum": checkpoint.adapter_checksum,
                    "parent_node_id": str(trial.parent_node_id),
                    "roles": list(trial.roles),
                    "stream": "vamp",
                    "task_id": str(language_task.task_id),
                    "training_loss": checkpoint.training_loss,
                    "update": checkpoint.update,
                    "validation_candidate_accuracy": (
                        checkpoint.validation_candidate_accuracy
                    ),
                    "validation_correct_nll": (
                        checkpoint.validation_correct_nll
                    ),
                }
                for trial in trained.diagnostics.trials
                for checkpoint in trial.checkpoints
            )
            stability_before = outcome.stability_before
            if task_index + 1 < len(curriculum_tasks):
                stability_before += (
                    self._committed_node_snapshot(
                        committed_graph,
                        NodeId(str(language_task.task_id)),
                        request.config.distractor_policy,
                        lora_config,
                    ),
                )
            outcome = _TrainingOutcome(
                independent_adapters=outcome.independent_adapters,
                graph=committed_graph,
                checkpoint_records=outcome.checkpoint_records + transfer_records,
                parent_records=outcome.parent_records + (parent_record,),
                stability_before=stability_before,
            )
            _write_training_chunk(
                workspace,
                len(curriculum_tasks) + task_index,
                f"vamp:{language_task.task_id}",
                self._execution_sha256(request),
                outcome,
                self.model_config,
                lora_config,
            )
            self._emit(
                "training_chunk_completed",
                trial_index=request.trial_index,
                stream="vamp",
                task_id=str(language_task.task_id),
            )
        return outcome

    def _train_independent_adapter(
        self,
        request: CalibrationValidationRequest,
        task_id: TaskId,
        train_batches: tuple[TokenBatch, ...],
        validation_suite: KnowledgeValidationSuite,
        lora_config: LoraConfig,
        train_config: LmTrainConfig,
        workspace: Path,
    ) -> tuple[LoraEdge, tuple[dict[str, object], ...]]:
        empty_graph = init_memory_graph(NodeId("root"))
        packed = pack_lora_memory(
            empty_graph,
            self.model_config,
            lora_config,
            max_nodes=2,
            max_edges=1,
        )
        initialization_key, training_key = jax.random.split(
            _rng_key(
                self._execution_sha256(request),
                f"independent:{task_id}",
            )
        )
        state = init_candidate_lora_train_state(
            init_lora_edge(initialization_key, self.model_config, lora_config),
            training_key,
            train_config,
        )

        def validate(adapter: LoraEdge, update: int) -> tuple[float, float]:
            del update
            memory = packed_with_candidate_edge(packed, adapter, 0)
            coefficients = np.ones(
                (len(validation_suite.queries), 1),
                dtype=np.float32,
            )
            scores = score_edge_coefficient_candidates(
                self.base_params,
                self.model_config,
                memory,
                lora_config,
                validation_suite.queries,
                coefficients,
                evaluation_microbatch_size=(
                    self.execution_preset.evaluation_microbatch_size
                ),
            )
            correct = np.asarray(
                tuple(
                    query.correct_candidate_index
                    for query in validation_suite.queries
                ),
                dtype=np.int32,
            )
            rows = np.arange(correct.size)
            return (
                float(np.mean(np.argmin(scores, axis=1) == correct)),
                float(np.mean(scores[rows, correct])),
            )

        resume_directory = workspace / "resume" / "independent" / str(task_id)
        loaded = _load_independent_resume(
            resume_directory,
            self._execution_sha256(request),
            state,
            train_config.steps,
        )
        current_state, records = loaded or (state, ())

        def checkpoint_record(checkpoint) -> dict[str, object]:
            return {
                "adapter_checksum": _tree_checksum(checkpoint.state.trainable),
                "parent_node_id": "root",
                "roles": ["independent_root"],
                "stream": "independent",
                "task_id": str(task_id),
                "training_loss": checkpoint.training_loss,
                "update": checkpoint.update,
                "validation_candidate_accuracy": (
                    checkpoint.validation_candidate_accuracy
                ),
                "validation_correct_nll": (
                    checkpoint.validation_correct_nll
                ),
            }

        if loaded is None:
            current_state, _, checkpoints = run_resumable_candidate_edge_updates(
                current_state,
                train_batches,
                self.base_params,
                self.model_config,
                packed,
                lora_config,
                jnp.zeros((1,), dtype=jnp.float32),
                0,
                train_config,
                stop_update=0,
                validation_function=validate,
            )
            records = tuple(checkpoint_record(value) for value in checkpoints)
            _write_independent_resume(
                resume_directory,
                self._execution_sha256(request),
                state,
                current_state,
                records,
            )
        for target_update in _checkpoint_targets(train_config.steps):
            if target_update <= int(current_state.step):
                continue
            current_state, _, checkpoints = run_resumable_candidate_edge_updates(
                current_state,
                train_batches,
                self.base_params,
                self.model_config,
                packed,
                lora_config,
                jnp.zeros((1,), dtype=jnp.float32),
                0,
                train_config,
                stop_update=target_update,
                validation_function=validate,
            )
            records += tuple(
                checkpoint_record(value) for value in checkpoints[1:]
            )
            _write_independent_resume(
                resume_directory,
                self._execution_sha256(request),
                state,
                current_state,
                records,
            )
        return current_state.trainable, records

    def _train_counterfactuals_resumably(
        self,
        request: CalibrationValidationRequest,
        workspace: Path,
        plan: ParentCounterfactualPlan,
        validation_suite: KnowledgeValidationSuite,
        initial_state: LmTrainState[LoraEdge],
        train_batches: tuple[TokenBatch, ...],
        packed: PackedLoraMemory,
        lora_config: LoraConfig,
        train_config: LmTrainConfig,
    ) -> KnowledgeCounterfactualTraining:
        resume_directory = (
            workspace / "resume" / "vamp" / str(plan.context.task_id)
        )
        loaded = _load_counterfactual_resume(
            resume_directory,
            self._execution_sha256(request),
            plan,
            initial_state,
            train_config.steps,
        )
        current: LmTrainState[LoraEdge] | KnowledgeCounterfactualTraining
        current = initial_state if loaded is None else loaded
        if loaded is None:
            _write_counterfactual_initial_resume(
                resume_directory,
                self._execution_sha256(request),
                initial_state,
            )
        elif isinstance(loaded, KnowledgeCounterfactualTraining):
            validate_parent_counterfactual_resume(
                plan,
                validation_suite,
                loaded,
                train_batches,
                self.base_params,
                self.model_config,
                packed,
                lora_config,
                train_config,
            )
        for target_update in _checkpoint_targets(train_config.steps):
            current_update = (
                int(current.step)
                if isinstance(current, LmTrainState)
                else current.diagnostics.trials[0].final_update
            )
            if target_update <= current_update:
                continue
            current = run_parent_counterfactuals(
                plan,
                validation_suite,
                current,
                train_batches,
                self.base_params,
                self.model_config,
                packed,
                lora_config,
                train_config,
                stop_update=target_update,
                evaluation_microbatch_size=(
                    self.execution_preset.evaluation_microbatch_size
                ),
            )
            _write_counterfactual_resume(
                resume_directory,
                self._execution_sha256(request),
                current,
            )
        if not isinstance(current, KnowledgeCounterfactualTraining):
            raise RuntimeError("counterfactual training did not reach one update")
        return current

    def _validation_suite(
        self,
        policy: CalibrationDistractorPolicy,
        task_id: TaskId,
    ) -> KnowledgeValidationSuite:
        groups = tuple(
            group
            for group in self.pool.groups(policy)
            if group.split is DataSplit.VALIDATION
            and group.task_id == str(task_id)
        )
        queries = tuple(
            variant.knowledge_query
            for group in groups
            for variant in group.variants
            if variant.knowledge_query.prefix_length
            == self.execution_preset.parent_prefix_length
        )
        symbolic_task = self.pool.hard_bundle.world.task(
            SymbolicTaskId(str(task_id))
        )
        return KnowledgeValidationSuite(
            suite_id=f"calibration:{task_id}:validation:prefix-64:{policy.value}",
            split="validation",
            task_id=task_id,
            family_id=str(symbolic_task.family_id),
            queries=queries,
        )

    def _committed_node_snapshot(
        self,
        graph: MemoryGraph[LoraEdge],
        node_id: NodeId,
        policy: CalibrationDistractorPolicy,
        lora_config: LoraConfig,
    ) -> CommittedNodeSnapshot:
        node_index = memory_node_ids(graph).index(node_id)
        node = graph.nodes[node_index]
        if node.incoming_edge is None or node.trained_task is None:
            raise ValueError("only committed task nodes can be snapshotted")
        queries = self._validation_suite(
            policy,
            TaskId(str(node.trained_task)),
        ).queries
        packed = pack_lora_memory(
            graph,
            self.model_config,
            lora_config,
            max_nodes=len(self.pool.hard_bundle.tasks) + 1,
            max_edges=len(self.pool.hard_bundle.tasks),
        )
        coefficients = edge_coefficients_for_node(packed, node_index)
        logits_sha256 = _validation_logits_sha256(
            self.base_params,
            self.model_config,
            packed,
            lora_config,
            coefficients,
            queries,
            self.execution_preset.evaluation_microbatch_size,
        )
        hard_scores = score_hard_node_candidates(
            self.base_params,
            self.model_config,
            packed,
            lora_config,
            queries,
            evaluation_microbatch_size=(
                self.execution_preset.evaluation_microbatch_size
            ),
        )
        answers = np.argmin(hard_scores[:, :, node_index], axis=1).astype(
            np.int32
        )
        answer_digest = sha256()
        answer_digest.update("\n".join(query.query_id for query in queries).encode())
        answer_digest.update(answers.tobytes())
        return CommittedNodeSnapshot(
            node_id=str(node_id),
            adapter_sha256=_tree_checksum(node.incoming_edge),
            logits_sha256=logits_sha256,
            answers_sha256=answer_digest.hexdigest(),
        )

    def _measure_evidence(
        self,
        policy: CalibrationDistractorPolicy,
        split: DataSplit,
        outcome: _TrainingOutcome,
        lora_config: LoraConfig,
    ) -> tuple[TinyWorldsCalibrationEvidence, tuple[dict[str, object], ...]]:
        direct_queries = self._evidence_queries(
            policy,
            split,
            QueryKind.DIRECT,
            prefix_length=self.execution_preset.evidence_prefix_length,
        )
        one_hop_queries = self._evidence_queries(
            policy,
            split,
            QueryKind.ONE_HOP,
            prefix_length=self.execution_preset.evidence_prefix_length,
        )
        revision_candidates = self._evidence_queries(
            policy,
            split,
            QueryKind.REVISION_SENSITIVE,
            cue_regime="cue_sufficient",
        )
        revision_with_old_indices = tuple(
            (query, old_index)
            for query in revision_candidates
            for old_index in (
                self._candidate_role_index(
                    policy,
                    query.query_id,
                    CandidateRole.INCOMPATIBLE_REVISION,
                ),
            )
            if old_index is not None
        )
        revision_queries = tuple(
            query for query, _ in revision_with_old_indices
        )
        if not direct_queries or not one_hop_queries or not revision_queries:
            raise ValueError("calibration evidence slices must all be nonempty")

        frozen_direct = score_frozen_base_candidates(
            self.base_params,
            self.model_config,
            direct_queries,
            evaluation_microbatch_size=(
                self.execution_preset.evaluation_microbatch_size
            ),
        )
        frozen_one_hop = score_frozen_base_candidates(
            self.base_params,
            self.model_config,
            one_hop_queries,
            evaluation_microbatch_size=(
                self.execution_preset.evaluation_microbatch_size
            ),
        )
        independent_direct = self._score_independent_queries(
            direct_queries,
            outcome,
            lora_config,
        )
        independent_one_hop = self._score_independent_queries(
            one_hop_queries,
            outcome,
            lora_config,
        )
        packed = pack_lora_memory(
            outcome.graph,
            self.model_config,
            lora_config,
            max_nodes=len(self.pool.hard_bundle.tasks) + 1,
            max_edges=len(self.pool.hard_bundle.tasks),
        )
        revision_hard_scores = score_hard_node_candidates(
            self.base_params,
            self.model_config,
            packed,
            lora_config,
            revision_queries,
            evaluation_microbatch_size=(
                self.execution_preset.evaluation_microbatch_size
            ),
        )
        seed_task = next(
            task
            for task in self.pool.hard_bundle.tasks
            if task.kind is TaskKind.SEED
        )
        revision_task = next(
            task
            for task in self.pool.hard_bundle.tasks
            if task.kind is TaskKind.REVISION
        )
        node_ids = memory_node_ids(outcome.graph)
        seed_node_index = node_ids.index(NodeId(str(seed_task.task_id)))
        revision_node_index = node_ids.index(NodeId(str(revision_task.task_id)))
        old_expected = np.asarray(
            tuple(old_index for _, old_index in revision_with_old_indices),
            dtype=np.int32,
        )
        revision_expected = _correct_indices(revision_queries)
        old_predictions = np.argmin(
            revision_hard_scores[:, :, seed_node_index],
            axis=1,
        )
        revision_predictions = np.argmin(
            revision_hard_scores[:, :, revision_node_index],
            axis=1,
        )
        exact_successes, exact_trials, exact_records = self._exact_kg_records(
            policy,
            split,
        )
        stability_after = tuple(
            self._committed_node_snapshot(
                outcome.graph,
                NodeId(snapshot.node_id),
                policy,
                lora_config,
            )
            for snapshot in outcome.stability_before
        )
        evidence = TinyWorldsCalibrationEvidence(
            exact_kg=calibration_binomial_evidence(
                exact_successes,
                exact_trials,
            ),
            frozen_novel_binding=_binomial_from_scores(
                direct_queries,
                frozen_direct,
            ),
            independent_direct_recall=_binomial_from_scores(
                direct_queries,
                independent_direct,
            ),
            frozen_one_hop=_binomial_from_scores(
                one_hop_queries,
                frozen_one_hop,
            ),
            independent_one_hop=_binomial_from_scores(
                one_hop_queries,
                independent_one_hop,
            ),
            committed_node_stability=CommittedNodeStabilityEvidence(
                before=outcome.stability_before,
                after=stability_after,
            ),
            old_contextual_answer=calibration_binomial_evidence(
                int(np.sum(old_predictions == old_expected)),
                old_expected.size,
            ),
            revision_contextual_answer=calibration_binomial_evidence(
                int(np.sum(revision_predictions == revision_expected)),
                revision_expected.size,
            ),
            paired_revision_consistency=calibration_binomial_evidence(
                int(
                    np.sum(
                        (old_predictions == old_expected)
                        & (revision_predictions == revision_expected)
                    )
                ),
                revision_expected.size,
            ),
        )
        score_records = (
            _candidate_score_records(
                "frozen_base",
                "frozen_novel_binding",
                direct_queries,
                frozen_direct,
            )
            + _candidate_score_records(
                "independent_root_lora",
                "independent_direct_recall",
                direct_queries,
                independent_direct,
            )
            + _candidate_score_records(
                "frozen_base",
                "frozen_one_hop",
                one_hop_queries,
                frozen_one_hop,
            )
            + _candidate_score_records(
                "independent_root_lora",
                "independent_one_hop",
                one_hop_queries,
                independent_one_hop,
            )
            + _revision_score_records(
                revision_queries,
                revision_hard_scores[:, :, seed_node_index],
                revision_hard_scores[:, :, revision_node_index],
                old_expected,
                revision_expected,
            )
            + exact_records
        )
        return evidence, score_records

    def _evidence_queries(
        self,
        policy: CalibrationDistractorPolicy,
        split: DataSplit,
        kind: QueryKind,
        *,
        prefix_length: int | None = None,
        cue_regime: str | None = None,
    ) -> tuple[KnowledgeQuery, ...]:
        return tuple(
            variant.knowledge_query
            for group in self.pool.groups(policy)
            if group.split is split
            for variant in group.variants
            if variant.knowledge_query.query_kind == kind.value
            and (
                prefix_length is None
                or variant.knowledge_query.prefix_length == prefix_length
            )
            and (
                cue_regime is None
                or variant.knowledge_query.cue_regime == cue_regime
            )
        )

    def _score_independent_queries(
        self,
        queries: tuple[KnowledgeQuery, ...],
        outcome: _TrainingOutcome,
        lora_config: LoraConfig,
    ) -> np.ndarray:
        adapters = dict(outcome.independent_adapters)
        scores_by_query_id: dict[str, np.ndarray] = {}
        for task_id in tuple(dict.fromkeys(query.task_id for query in queries)):
            task_queries = tuple(
                query for query in queries if query.task_id == task_id
            )
            adapter = adapters[TaskId(str(task_id))]
            graph = add_memory_node(
                init_memory_graph(NodeId("root")),
                node_id=NodeId(str(task_id)),
                parent_id=NodeId("root"),
                trained_task=TaskId(str(task_id)),
                train_stage=1,
                incoming_edge=adapter,
            )
            packed = pack_lora_memory(
                graph,
                self.model_config,
                lora_config,
                max_nodes=2,
                max_edges=1,
            )
            task_scores = score_edge_coefficient_candidates(
                self.base_params,
                self.model_config,
                packed,
                lora_config,
                task_queries,
                np.ones((len(task_queries), 1), dtype=np.float32),
                evaluation_microbatch_size=(
                    self.execution_preset.evaluation_microbatch_size
                ),
            )
            scores_by_query_id.update(
                {
                    query.query_id: task_scores[index]
                    for index, query in enumerate(task_queries)
                }
            )
        scores = np.stack(
            tuple(scores_by_query_id[query.query_id] for query in queries)
        ).astype(np.float32)
        scores.flags.writeable = False
        return scores

    def _candidate_role_index(
        self,
        policy: CalibrationDistractorPolicy,
        query_id: str,
        role: CandidateRole,
    ) -> int | None:
        group_id = query_id.rsplit(":prefix-", 1)[0]
        group = next(
            group
            for group in self.pool.groups(policy)
            if group.group_id == group_id
        )
        variant = next(
            variant
            for variant in group.variants
            if variant.knowledge_query.query_id == query_id
        )
        plan = next(
            plan
            for plan in self.pool.bundle(policy).query_plans
            if str(plan.query_ast.query_id) == group.symbolic_query_id
        )
        role_entity_ids = tuple(
            str(candidate.entity_id)
            for candidate in plan.candidates
            if candidate.role is role
        )
        if not role_entity_ids:
            return None
        if len(role_entity_ids) != 1:
            raise ValueError(
                f"query {query_id} does not contain exactly one {role.value} candidate"
            )
        rendered_indices = tuple(
            index
            for index, entity_id in enumerate(variant.candidate_entity_ids)
            if entity_id == role_entity_ids[0]
        )
        if len(rendered_indices) != 1:
            raise ValueError(
                f"rendered query {query_id} lost its {role.value} candidate"
            )
        return rendered_indices[0]

    def _exact_kg_records(
        self,
        policy: CalibrationDistractorPolicy,
        split: DataSplit,
    ) -> tuple[int, int, tuple[dict[str, object], ...]]:
        bundle = self.pool.bundle(policy)
        plan_by_id = {
            str(plan.query_ast.query_id): plan
            for plan in bundle.query_plans
            if plan.split is split
        }
        records = tuple(
            _exact_kg_record(bundle, group, plan_by_id[group.symbolic_query_id])
            for group in self.pool.groups(policy)
            if group.split is split
        )
        return (
            sum(bool(record["correct"]) for record in records),
            len(records),
            records,
        )

    def _emit(self, event: str, **values: object) -> None:
        if self.event_sink is not None:
            self.event_sink({"event": event, **values})


def _correct_indices(queries: tuple[KnowledgeQuery, ...]) -> np.ndarray:
    return np.asarray(
        tuple(query.correct_candidate_index for query in queries),
        dtype=np.int32,
    )


def _validation_logits_sha256(
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    packed: PackedLoraMemory,
    lora_config: LoraConfig,
    edge_coefficients: jax.Array | np.ndarray,
    queries: tuple[KnowledgeQuery, ...],
    microbatch_size: int,
) -> str:
    """Hash every fixed validation probe ID and its complete routed logits."""
    if not queries:
        raise ValueError("committed-node stability requires validation probes")
    if type(microbatch_size) is not int or microbatch_size <= 0:
        raise ValueError("stability microbatch size must be positive")
    digest = sha256()
    for start in range(0, len(queries), microbatch_size):
        block = queries[start : start + microbatch_size]
        input_ids = np.concatenate(
            tuple(query.router_batch.input_ids for query in block),
            axis=0,
        )
        attention_mask = np.concatenate(
            tuple(query.router_batch.attention_mask for query in block),
            axis=0,
        )
        logits = np.asarray(
            apply_gpt_neo(
                base_params,
                model_config,
                jnp.asarray(input_ids, dtype=jnp.int32),
                jnp.asarray(attention_mask, dtype=jnp.bool_),
                lora_memory=packed,
                edge_coefficients=edge_coefficients,
                lora_config=lora_config,
                training=False,
            ).logits
        )
        if logits.shape[0] != len(block):
            raise RuntimeError("stability logits lost validation probes")
        for query, query_logits in zip(block, logits, strict=True):
            digest.update(
                _canonical_json_bytes(
                    {
                        "logits_sha256": _array_sha256(query_logits),
                        "query_id": query.query_id,
                    }
                )
            )
    return digest.hexdigest()


def _binomial_from_scores(
    queries: tuple[KnowledgeQuery, ...],
    scores: np.ndarray,
):
    values = np.asarray(scores, dtype=np.float32)
    if values.shape != (len(queries), 4):
        raise ValueError("candidate evidence scores must have shape [query, 4]")
    correct = _correct_indices(queries)
    return calibration_binomial_evidence(
        int(np.sum(np.argmin(values, axis=1) == correct)),
        len(queries),
    )


def _candidate_score_records(
    method: str,
    metric: str,
    queries: tuple[KnowledgeQuery, ...],
    scores: np.ndarray,
) -> tuple[dict[str, object], ...]:
    values = np.asarray(scores, dtype=np.float32)
    if values.shape != (len(queries), 4):
        raise ValueError("candidate score records require [query, 4] scores")
    return tuple(
        {
            "candidate_nll": [float(value) for value in values[index]],
            "correct_candidate_index": query.correct_candidate_index,
            "correct": int(np.argmin(values[index]))
            == query.correct_candidate_index,
            "method": method,
            "metric": metric,
            "predicted_candidate_index": int(np.argmin(values[index])),
            "prefix_length": query.prefix_length,
            "query_id": query.query_id,
            "task_id": str(query.task_id),
        }
        for index, query in enumerate(queries)
    )


def _revision_score_records(
    queries: tuple[KnowledgeQuery, ...],
    old_scores: np.ndarray,
    revision_scores: np.ndarray,
    old_expected: np.ndarray,
    revision_expected: np.ndarray,
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "candidate_nll": [float(value) for value in scores[index]],
            "correct": int(np.argmin(scores[index])) == int(expected[index]),
            "correct_candidate_index": int(expected[index]),
            "method": method,
            "metric": metric,
            "predicted_candidate_index": int(np.argmin(scores[index])),
            "prefix_length": query.prefix_length,
            "query_id": query.query_id,
            "task_id": str(query.task_id),
        }
        for method, metric, scores, expected in (
            ("vamp_old_node", "old_contextual_answer", old_scores, old_expected),
            (
                "vamp_revision_node",
                "revision_contextual_answer",
                revision_scores,
                revision_expected,
            ),
        )
        for index, query in enumerate(queries)
    )


def _exact_kg_record(
    bundle: TinyWorldsBundle,
    group: RenderedQueryGroup,
    plan: QueryPlan,
) -> dict[str, object]:
    answers = answer_query(
        bundle.closure,
        plan.query_ast,
        bundle.world.registry,
        bundle.entities,
    )
    symbolic_candidate_ids = {
        str(candidate.entity_id) for candidate in plan.candidates
    }
    correct = (
        answers == (plan.answer_entity_id,)
        and all(
            set(variant.candidate_entity_ids) == symbolic_candidate_ids
            for variant in group.variants
        )
        and all(
            variant.candidate_entity_ids[
                variant.knowledge_query.correct_candidate_index
            ]
            == str(plan.answer_entity_id)
            for variant in group.variants
        )
    )
    return {
        "answer_entity_ids": [str(value) for value in answers],
        "correct": correct,
        "group_id": group.group_id,
        "method": "exact_kg",
        "metric": "exact_kg",
        "query_id": group.symbolic_query_id,
        "task_id": group.task_id,
    }


def _checkpoint_targets(update_budget: int) -> tuple[int, ...]:
    return tuple(
        sorted(
            {0, update_budget}
            | {
                2**power
                for power in range(update_budget.bit_length())
                if 2**power <= update_budget
            }
        )
    )


def _write_independent_resume(
    directory: Path,
    execution_sha256: str,
    initial_state: LmTrainState[LoraEdge],
    state: LmTrainState[LoraEdge],
    records: tuple[dict[str, object], ...],
) -> None:
    update = int(state.step)
    target = directory / f"update-{update:07d}"
    _write_resume_directory(
        target,
        {
            "execution_sha256": execution_sha256,
            "initial_state_sha256": lm_train_state_checksum(initial_state),
            "kind": "independent",
            "records": list(records),
            "state_count": 1,
            "update": update,
        },
        (state,),
    )


def _load_independent_resume(
    directory: Path,
    execution_sha256: str,
    template: LmTrainState[LoraEdge],
    update_budget: int,
) -> tuple[LmTrainState[LoraEdge], tuple[dict[str, object], ...]] | None:
    chain = _load_resume_directories(directory, update_budget)
    if not chain:
        return None
    initial_state_sha256 = lm_train_state_checksum(template)
    previous_records: tuple[dict[str, object], ...] = ()
    latest: tuple[
        LmTrainState[LoraEdge],
        tuple[dict[str, object], ...],
    ] | None = None
    for manifest, state_directory in chain:
        if manifest["kind"] != "independent" or (
            manifest["execution_sha256"] != execution_sha256
            or manifest["initial_state_sha256"] != initial_state_sha256
        ):
            raise ValueError("independent resume identity changed")
        if manifest["state_count"] != 1:
            raise ValueError("independent resume must contain exactly one state")
        state = load_lm_train_state_artifact(
            state_directory,
            _require_string(
                manifest["state_identity_sha256"],
                "state identity",
            ),
            (template,),
        )[0]
        update = _require_integer(manifest["update"], "resume update")
        if int(state.step) != update:
            raise ValueError("independent resume update mismatch")
        if update == 0 and lm_train_state_checksum(state) != initial_state_sha256:
            raise ValueError("independent update-zero state changed")
        records = tuple(
            _require_record(value, "checkpoint record")
            for value in _require_list(manifest["records"], "checkpoint records")
        )
        _validate_independent_checkpoint_records(records, state, update)
        if records[: len(previous_records)] != previous_records:
            raise ValueError("independent checkpoint history changed")
        previous_records = records
        latest = (state, records)
    if latest is None:
        raise RuntimeError("independent resume chain unexpectedly vanished")
    return latest


def _write_counterfactual_initial_resume(
    directory: Path,
    execution_sha256: str,
    state: LmTrainState[LoraEdge],
) -> None:
    _write_resume_directory(
        directory / "update-0000000",
        {
            "diagnostics": None,
            "execution_sha256": execution_sha256,
            "initial_state_sha256": lm_train_state_checksum(state),
            "kind": "counterfactual",
            "state_count": 1,
            "training_execution_sha256": None,
            "update": 0,
        },
        (state,),
    )


def _write_counterfactual_resume(
    directory: Path,
    execution_sha256: str,
    training: KnowledgeCounterfactualTraining,
) -> None:
    update = training.diagnostics.trials[0].final_update
    _write_resume_directory(
        directory / f"update-{update:07d}",
        {
            "diagnostics": _counterfactual_diagnostics_record(
                training.diagnostics
            ),
            "execution_sha256": execution_sha256,
            "initial_state_sha256": (
                training.diagnostics.trials[0].initial_state_checksum
            ),
            "kind": "counterfactual",
            "state_count": len(training.final_states),
            "training_execution_sha256": training.execution_sha256,
            "update": update,
        },
        training.final_states,
    )


def _load_counterfactual_resume(
    directory: Path,
    execution_sha256: str,
    plan: ParentCounterfactualPlan,
    state_template: LmTrainState[LoraEdge],
    update_budget: int,
) -> LmTrainState[LoraEdge] | KnowledgeCounterfactualTraining | None:
    chain = _load_resume_directories(directory, update_budget)
    if not chain:
        return None
    initial_state_sha256 = lm_train_state_checksum(state_template)
    previous_diagnostics: KnowledgeTransferDiagnostics | None = None
    training_execution_sha256: str | None = None
    latest: LmTrainState[LoraEdge] | KnowledgeCounterfactualTraining | None = None
    for manifest, state_directory in chain:
        if manifest["kind"] != "counterfactual" or (
            manifest["execution_sha256"] != execution_sha256
            or manifest["initial_state_sha256"] != initial_state_sha256
        ):
            raise ValueError("counterfactual resume identity changed")
        state_count = _require_integer(manifest["state_count"], "state count")
        if state_count <= 0:
            raise ValueError("counterfactual resume state count is invalid")
        states = load_lm_train_state_artifact(
            state_directory,
            _require_string(
                manifest["state_identity_sha256"],
                "state identity",
            ),
            tuple(state_template for _ in range(state_count)),
        )
        update = _require_integer(manifest["update"], "resume update")
        if any(int(state.step) != update for state in states):
            raise ValueError("counterfactual resume state updates differ")
        if update == 0:
            if (
                state_count != 1
                or manifest["diagnostics"] is not None
                or manifest["training_execution_sha256"] is not None
            ):
                raise ValueError("update-zero counterfactual resume is invalid")
            if lm_train_state_checksum(states[0]) != initial_state_sha256:
                raise ValueError("counterfactual initial state changed")
            latest = states[0]
            continue
        diagnostics = _counterfactual_diagnostics_from_record(
            plan,
            _require_record(
                manifest["diagnostics"],
                "counterfactual diagnostics",
            ),
        )
        if any(
            trial.initial_state_checksum != initial_state_sha256
            for trial in diagnostics.trials
        ):
            raise ValueError("counterfactual initial state changed")
        current_training_execution_sha256 = _require_string(
            manifest["training_execution_sha256"],
            "counterfactual training execution SHA-256",
        )
        if training_execution_sha256 is None:
            training_execution_sha256 = current_training_execution_sha256
        elif current_training_execution_sha256 != training_execution_sha256:
            raise ValueError(
                "counterfactual training execution history changed"
            )
        if previous_diagnostics is not None:
            _validate_counterfactual_diagnostics_prefix(
                previous_diagnostics,
                diagnostics,
            )
        previous_diagnostics = diagnostics
        latest = KnowledgeCounterfactualTraining(
            diagnostics=diagnostics,
            final_states=states,
            execution_sha256=current_training_execution_sha256,
        )
    if latest is None:
        raise RuntimeError("counterfactual resume chain unexpectedly vanished")
    return latest


def _validate_counterfactual_diagnostics_prefix(
    previous: KnowledgeTransferDiagnostics,
    current: KnowledgeTransferDiagnostics,
) -> None:
    """Require every earlier transfer trajectory to prefix the next chunk."""
    if len(previous.trials) != len(current.trials):
        raise ValueError("counterfactual diagnostic trial history changed")
    for earlier, later in zip(previous.trials, current.trials, strict=True):
        earlier_static = (
            earlier.parent_node_index,
            earlier.parent_node_id,
            earlier.roles,
            earlier.parent_validation_mean_correct_nll,
            earlier.initial_state_checksum,
        )
        later_static = (
            later.parent_node_index,
            later.parent_node_id,
            later.roles,
            later.parent_validation_mean_correct_nll,
            later.initial_state_checksum,
        )
        if earlier_static != later_static or (
            later.step_losses[: len(earlier.step_losses)]
            != earlier.step_losses
        ) or (
            later.checkpoints[: len(earlier.checkpoints)]
            != earlier.checkpoints
        ):
            raise ValueError("counterfactual diagnostic history changed")


def _write_resume_directory(
    target: Path,
    core: dict[str, object],
    states: tuple[LmTrainState[LoraEdge], ...],
) -> None:
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"immutable resume chunk already exists: {target}")
    if target.parent.is_symlink():
        raise ValueError("calibration resume root cannot be a symlink")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent)
    )
    try:
        state_identity_sha256 = _resume_state_identity_sha256(core)
        state_manifest = write_lm_train_state_artifact(
            temporary / "training_state",
            state_identity_sha256,
            states,
        )
        manifest = {
            **core,
            "format": "apm.tinyworlds.calibration-resume-chunk",
            "schema_version": 1,
            "state_identity_sha256": state_identity_sha256,
            "state_payload_sha256": state_manifest.payload_sha256,
        }
        _write_canonical_json(temporary / "manifest.json", manifest)
        _fsync_tree(temporary)
        os.rename(temporary, target)
        _fsync_directory(target.parent)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _load_resume_directories(
    directory: Path,
    update_budget: int,
) -> tuple[tuple[dict[str, object], Path], ...]:
    if type(update_budget) is not int or update_budget <= 0:
        raise ValueError("resume update budget must be positive")
    if directory.is_symlink():
        raise ValueError("calibration resume root cannot be a symlink")
    if not directory.exists():
        return ()
    if not directory.is_dir():
        raise ValueError("calibration resume root must be a directory")
    paths: list[Path] = []
    for path in directory.iterdir():
        if path.is_symlink():
            raise ValueError("calibration resume targets cannot be symlinks")
        if (
            _RESUME_TEMP_PATTERN.fullmatch(path.name) is not None
            and path.is_dir()
        ):
            continue
        if (
            _RESUME_DIRECTORY_PATTERN.fullmatch(path.name) is not None
            and path.is_dir()
        ):
            paths.append(path)
            continue
        raise ValueError(f"unexpected calibration resume entry: {path.name}")
    paths.sort()
    if not paths:
        return ()
    updates = tuple(
        _require_integer(int(path.name.removeprefix("update-")), "resume path update")
        for path in paths
    )
    expected = _checkpoint_targets(update_budget)
    if updates != expected[: len(updates)]:
        raise ValueError("resume chunks must be an exact checkpoint prefix")
    return tuple(
        _load_resume_directory(path, update)
        for path, update in zip(paths, updates, strict=True)
    )


def _load_latest_resume_directory(
    directory: Path,
    update_budget: int,
) -> tuple[dict[str, object], Path] | None:
    chain = _load_resume_directories(directory, update_budget)
    return None if not chain else chain[-1]


def _load_resume_directory(
    target: Path,
    expected_update: int,
) -> tuple[dict[str, object], Path]:
    entries = tuple(target.iterdir())
    if any(path.is_symlink() for path in entries) or {
        path.name for path in entries
    } != {"manifest.json", "training_state"}:
        raise ValueError("resume chunk entries changed")
    manifest_path = target / "manifest.json"
    state_directory = target / "training_state"
    if manifest_path.is_symlink() or state_directory.is_symlink():
        raise ValueError("resume chunk targets cannot be symlinks")
    manifest = _load_canonical_json(manifest_path)
    required = {
        "execution_sha256",
        "format",
        "initial_state_sha256",
        "kind",
        "schema_version",
        "state_count",
        "state_identity_sha256",
        "state_payload_sha256",
        "update",
    }
    if manifest.get("kind") == "independent":
        required.add("records")
    elif manifest.get("kind") == "counterfactual":
        required.update(("diagnostics", "training_execution_sha256"))
    if set(manifest) != required:
        raise ValueError("resume chunk fields changed")
    if (
        manifest["format"] != "apm.tinyworlds.calibration-resume-chunk"
        or manifest["schema_version"] != 1
        or manifest["update"] != expected_update
    ):
        raise ValueError("resume chunk manifest is inconsistent")
    if manifest["state_identity_sha256"] != _resume_state_identity_sha256(
        manifest
    ):
        raise ValueError("resume state identity changed")
    state_entries = tuple(state_directory.iterdir())
    if any(path.is_symlink() for path in state_entries) or {
        path.name for path in state_entries
    } != {"manifest.json", "state.safetensors"}:
        raise ValueError("resume training-state entries changed")
    state_manifest = _load_canonical_json(state_directory / "manifest.json")
    state_core = {
        key: value
        for key, value in state_manifest.items()
        if key != "payload_sha256"
    }
    if (
        state_manifest.get("payload_sha256")
        != manifest["state_payload_sha256"]
        or state_manifest.get("payload_sha256")
        != sha256(_canonical_json_bytes(state_core)).hexdigest()
        or state_manifest.get("identity_sha256")
        != manifest["state_identity_sha256"]
        or state_manifest.get("state_count") != manifest["state_count"]
        or state_manifest.get("tensor_file") != "state.safetensors"
        or state_manifest.get("tensor_file_sha256")
        != _file_sha256(state_directory / "state.safetensors")
    ):
        raise ValueError("resume training-state payload checksum changed")
    return manifest, state_directory


def _resume_state_identity_sha256(core: dict[str, object]) -> str:
    return _digest_record(
        {
            "execution_sha256": core["execution_sha256"],
            "initial_state_sha256": core["initial_state_sha256"],
            "kind": core["kind"],
            "state_count": core["state_count"],
            "update": core["update"],
        }
    )


def _validate_independent_checkpoint_records(
    records: tuple[dict[str, object], ...],
    state: LmTrainState[LoraEdge],
    update: int,
) -> None:
    updates = tuple(
        _require_integer(record.get("update"), "checkpoint update")
        for record in records
    )
    if updates != _checkpoint_targets(update):
        raise ValueError("independent checkpoints must follow the exact schedule")
    required_fields = (
        "adapter_checksum",
        "parent_node_id",
        "roles",
        "stream",
        "task_id",
        "training_loss",
        "update",
        "validation_candidate_accuracy",
        "validation_correct_nll",
    )
    for record in records:
        _require_record_fields(record, required_fields, "checkpoint record")
        if (
            record["parent_node_id"] != "root"
            or record["roles"] != ["independent_root"]
            or record["stream"] != "independent"
        ):
            raise ValueError("independent checkpoint metadata changed")
    if records[-1]["adapter_checksum"] != _tree_checksum(state.trainable):
        raise ValueError("independent checkpoint adapter changed")


def _counterfactual_diagnostics_record(
    diagnostics: KnowledgeTransferDiagnostics,
) -> dict[str, object]:
    return {
        "trials": [
            {
                "checkpoints": [
                    {
                        "adapter_checksum": checkpoint.adapter_checksum,
                        "training_loss": checkpoint.training_loss,
                        "update": checkpoint.update,
                        "validation_candidate_accuracy": (
                            checkpoint.validation_candidate_accuracy
                        ),
                        "validation_correct_nll": (
                            checkpoint.validation_correct_nll
                        ),
                    }
                    for checkpoint in trial.checkpoints
                ],
                "final_adapter_checksum": trial.final_adapter_checksum,
                "final_state_checksum": trial.final_state_checksum,
                "final_update": trial.final_update,
                "initial_state_checksum": trial.initial_state_checksum,
                "parent_node_id": str(trial.parent_node_id),
                "parent_node_index": trial.parent_node_index,
                "parent_validation_mean_correct_nll": (
                    trial.parent_validation_mean_correct_nll
                ),
                "roles": list(trial.roles),
                "step_losses": list(trial.step_losses),
            }
            for trial in diagnostics.trials
        ]
    }


def _counterfactual_diagnostics_from_record(
    plan: ParentCounterfactualPlan,
    record: dict[str, object],
) -> KnowledgeTransferDiagnostics:
    _require_record_fields(record, ("trials",), "counterfactual diagnostics")
    trials = tuple(
        _parent_transfer_trial_from_record(
            _require_record(value, "parent transfer trial")
        )
        for value in _require_list(record["trials"], "parent transfer trials")
    )
    return KnowledgeTransferDiagnostics(plan=plan, trials=trials)


def _parent_transfer_trial_from_record(
    record: dict[str, object],
) -> ParentTransferTrialDiagnostic:
    _require_record_fields(
        record,
        (
            "checkpoints",
            "final_adapter_checksum",
            "final_state_checksum",
            "final_update",
            "initial_state_checksum",
            "parent_node_id",
            "parent_node_index",
            "parent_validation_mean_correct_nll",
            "roles",
            "step_losses",
        ),
        "parent transfer trial",
    )
    checkpoints = tuple(
        _transfer_checkpoint_from_record(
            _require_record(value, "transfer checkpoint")
        )
        for value in _require_list(record["checkpoints"], "transfer checkpoints")
    )
    return ParentTransferTrialDiagnostic(
        parent_node_index=_require_integer(
            record["parent_node_index"],
            "parent node index",
        ),
        parent_node_id=NodeId(
            _require_string(record["parent_node_id"], "parent node ID")
        ),
        roles=tuple(
            _require_string(value, "counterfactual role")
            for value in _require_list(record["roles"], "counterfactual roles")
        ),
        parent_validation_mean_correct_nll=_require_float(
            record["parent_validation_mean_correct_nll"],
            "parent validation NLL",
        ),
        initial_state_checksum=_require_string(
            record["initial_state_checksum"],
            "initial state checksum",
        ),
        final_adapter_checksum=_require_string(
            record["final_adapter_checksum"],
            "final adapter checksum",
        ),
        final_state_checksum=_require_string(
            record["final_state_checksum"],
            "final state checksum",
        ),
        final_update=_require_integer(record["final_update"], "final update"),
        step_losses=tuple(
            _require_float(value, "step loss")
            for value in _require_list(record["step_losses"], "step losses")
        ),
        checkpoints=checkpoints,
    )


def _transfer_checkpoint_from_record(
    record: dict[str, object],
) -> TransferCheckpointDiagnostic:
    _require_record_fields(
        record,
        (
            "adapter_checksum",
            "training_loss",
            "update",
            "validation_candidate_accuracy",
            "validation_correct_nll",
        ),
        "transfer checkpoint",
    )
    training_loss_value = record["training_loss"]
    return TransferCheckpointDiagnostic(
        update=_require_integer(record["update"], "checkpoint update"),
        training_loss=(
            None
            if training_loss_value is None
            else _require_float(training_loss_value, "checkpoint training loss")
        ),
        validation_candidate_accuracy=_require_float(
            record["validation_candidate_accuracy"],
            "checkpoint validation accuracy",
        ),
        validation_correct_nll=_require_float(
            record["validation_correct_nll"],
            "checkpoint validation NLL",
        ),
        adapter_checksum=_require_string(
            record["adapter_checksum"],
            "checkpoint adapter checksum",
        ),
    )


def _write_training_chunk(
    workspace: Path,
    chunk_index: int,
    label: str,
    execution_sha256: str,
    outcome: _TrainingOutcome,
    model_config: GptNeoConfig,
    lora_config: LoraConfig,
) -> None:
    if workspace.is_symlink():
        raise ValueError("calibration training workspace cannot be a symlink")
    chunks = workspace / "chunks"
    if chunks.is_symlink():
        raise ValueError("calibration training chunk root cannot be a symlink")
    chunks.mkdir(parents=True, exist_ok=True)
    target = chunks / f"chunk-{chunk_index:03d}"
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"immutable calibration chunk already exists: {target}")
    temporary = Path(tempfile.mkdtemp(prefix=".chunk.tmp-", dir=chunks))
    try:
        model_record = _write_outcome_model(
            temporary / "model.npz",
            outcome,
            model_config,
            lora_config,
        )
        checkpoint_payload = _jsonl_bytes(outcome.checkpoint_records)
        parent_payload = _jsonl_bytes(outcome.parent_records)
        (temporary / "checkpointed_transfer.jsonl").write_bytes(
            checkpoint_payload
        )
        (temporary / "parent_search.jsonl").write_bytes(parent_payload)
        manifest = {
            "checkpointed_transfer_sha256": sha256(
                checkpoint_payload
            ).hexdigest(),
            "chunk_index": chunk_index,
            "execution_sha256": execution_sha256,
            "format": "apm.tinyworlds.calibration-training-chunk",
            "label": label,
            "model": model_record,
            "model_file_sha256": _file_sha256(temporary / "model.npz"),
            "parent_search_sha256": sha256(parent_payload).hexdigest(),
            "schema_version": 1,
        }
        _write_canonical_json(temporary / "manifest.json", manifest)
        _fsync_tree(temporary)
        os.rename(temporary, target)
        _fsync_directory(target.parent)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _load_latest_training_chunk(
    workspace: Path,
    execution_sha256: str,
    model_config: GptNeoConfig,
    lora_config: LoraConfig,
) -> _TrainingOutcome | None:
    if workspace.is_symlink():
        raise ValueError("calibration training workspace cannot be a symlink")
    chunks = workspace / "chunks"
    if chunks.is_symlink():
        raise ValueError("calibration training chunk root cannot be a symlink")
    if not chunks.exists():
        return None
    if not chunks.is_dir():
        raise ValueError("calibration training chunks must be a directory")
    paths: list[Path] = []
    for path in chunks.iterdir():
        if path.is_symlink():
            raise ValueError("calibration training chunk targets cannot be symlinks")
        if (
            _TRAINING_CHUNK_TEMP_PATTERN.fullmatch(path.name) is not None
            and path.is_dir()
        ):
            continue
        if (
            _TRAINING_CHUNK_PATTERN.fullmatch(path.name) is not None
            and path.is_dir()
        ):
            paths.append(path)
            continue
        raise ValueError(f"unexpected calibration training entry: {path.name}")
    paths.sort()
    if not paths:
        return None
    expected_names = tuple(f"chunk-{index:03d}" for index in range(len(paths)))
    if tuple(path.name for path in paths) != expected_names:
        raise ValueError("calibration training chunks are not contiguous")
    outcomes = tuple(
        _load_training_chunk(
            path,
            chunk_index,
            execution_sha256,
            model_config,
            lora_config,
        )
        for chunk_index, path in enumerate(paths)
    )
    for previous, current in zip(outcomes, outcomes[1:]):
        if (
            current.checkpoint_records[: len(previous.checkpoint_records)]
            != previous.checkpoint_records
            or current.parent_records[: len(previous.parent_records)]
            != previous.parent_records
            or current.stability_before[: len(previous.stability_before)]
            != previous.stability_before
        ):
            raise ValueError("calibration training chunk history changed")
    return outcomes[-1]


def _load_training_chunk(
    path: Path,
    expected_index: int,
    execution_sha256: str,
    model_config: GptNeoConfig,
    lora_config: LoraConfig,
) -> _TrainingOutcome:
    entries = tuple(path.iterdir())
    if any(value.is_symlink() for value in entries) or {
        value.name for value in entries
    } != {
        "checkpointed_transfer.jsonl",
        "manifest.json",
        "model.npz",
        "parent_search.jsonl",
    }:
        raise ValueError("calibration training chunk entries changed")
    manifest = _load_canonical_json(path / "manifest.json")
    _require_record_fields(
        manifest,
        (
            "checkpointed_transfer_sha256",
            "chunk_index",
            "execution_sha256",
            "format",
            "label",
            "model",
            "model_file_sha256",
            "parent_search_sha256",
            "schema_version",
        ),
        "training chunk",
    )
    if manifest["execution_sha256"] != execution_sha256:
        raise ValueError("calibration training chunk execution identity changed")
    if manifest["chunk_index"] != expected_index:
        raise ValueError("calibration training chunk index changed")
    if manifest["model_file_sha256"] != _file_sha256(path / "model.npz"):
        raise ValueError("calibration training chunk model checksum mismatch")
    checkpoint_payload = (path / "checkpointed_transfer.jsonl").read_bytes()
    parent_payload = (path / "parent_search.jsonl").read_bytes()
    if manifest["checkpointed_transfer_sha256"] != sha256(
        checkpoint_payload
    ).hexdigest() or manifest["parent_search_sha256"] != sha256(
        parent_payload
    ).hexdigest():
        raise ValueError("calibration training chunk record checksum mismatch")
    model_record = _require_record(manifest["model"], "chunk model")
    adapters, graph, snapshots = _load_outcome_model(
        path / "model.npz",
        model_record,
        model_config,
        lora_config,
    )
    return _TrainingOutcome(
        independent_adapters=adapters,
        graph=graph,
        checkpoint_records=_load_jsonl_bytes(checkpoint_payload),
        parent_records=_load_jsonl_bytes(parent_payload),
        stability_before=snapshots,
    )


def _write_trial_artifact(
    target: Path,
    *,
    artifact_id: str,
    execution_sha256: str,
    request_record: dict[str, object],
    evidence: TinyWorldsCalibrationEvidence,
    outcome: _TrainingOutcome | None,
    model_config: GptNeoConfig,
    lora_config: LoraConfig,
    score_records: tuple[dict[str, object], ...],
    resource_evidence: CalibrationResourceEvidence,
) -> None:
    validate_calibration_resource_evidence(
        resource_evidence,
        expected_target_bytes=(
            resource_evidence.allocator_peak_target_bytes
        ),
    )
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"immutable calibration trial exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent)
    )
    try:
        score_payload = _jsonl_bytes(score_records)
        checkpoint_payload = _jsonl_bytes(
            () if outcome is None else outcome.checkpoint_records
        )
        parent_payload = _jsonl_bytes(
            () if outcome is None else outcome.parent_records
        )
        for name, payload in (
            ("candidate_scores.jsonl", score_payload),
            ("checkpointed_transfer.jsonl", checkpoint_payload),
            ("parent_search.jsonl", parent_payload),
        ):
            (temporary / name).write_bytes(payload)
        model_record = None
        model_sha256 = None
        if outcome is not None:
            model_record = _write_outcome_model(
                temporary / "model.npz",
                outcome,
                model_config,
                lora_config,
            )
            model_sha256 = _file_sha256(temporary / "model.npz")
        manifest = {
            "artifact_id": artifact_id,
            "artifacts": {
                "candidate_scores.jsonl": sha256(score_payload).hexdigest(),
                "checkpointed_transfer.jsonl": sha256(
                    checkpoint_payload
                ).hexdigest(),
                "parent_search.jsonl": sha256(parent_payload).hexdigest(),
            },
            "evidence": _evidence_record(evidence),
            "execution_sha256": execution_sha256,
            "format": _ARTIFACT_FORMAT,
            "model": model_record,
            "model_file_sha256": model_sha256,
            "request": request_record,
            "resource_evidence": _resource_evidence_record(resource_evidence),
            "schema_version": _ARTIFACT_SCHEMA_VERSION,
        }
        _write_canonical_json(temporary / "manifest.json", manifest)
        _fsync_tree(temporary)
        os.rename(temporary, target)
        _fsync_directory(target.parent)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


_TRIAL_MANIFEST_FIELDS = (
    "artifact_id",
    "artifacts",
    "evidence",
    "execution_sha256",
    "format",
    "model",
    "model_file_sha256",
    "request",
    "resource_evidence",
    "schema_version",
)


def _load_trial_manifest(target: Path) -> dict[str, object]:
    if target.is_symlink() or not target.is_dir():
        raise ValueError("calibration trial artifact must be a regular directory")
    manifest = _load_canonical_json(target / "manifest.json")
    _require_record_fields(
        manifest,
        _TRIAL_MANIFEST_FIELDS,
        "calibration trial artifact",
    )
    if (
        manifest["format"] != _ARTIFACT_FORMAT
        or manifest["schema_version"] != _ARTIFACT_SCHEMA_VERSION
    ):
        raise ValueError("calibration trial artifact format changed")
    return manifest


def load_calibration_trial_resource_evidence(
    target: str | Path,
    *,
    expected_target_bytes: int | None = (
        CALIBRATION_ALLOCATOR_PEAK_TARGET_BYTES
    ),
) -> CalibrationResourceEvidence:
    """Strictly load and enforce one persisted trial's resource evidence."""
    manifest = _load_trial_manifest(Path(target))
    evidence = _resource_evidence_from_record(
        _require_record(manifest["resource_evidence"], "resource evidence")
    )
    validate_calibration_resource_evidence(
        evidence,
        expected_target_bytes=expected_target_bytes,
    )
    return evidence


def _load_trial_artifact(
    target: Path,
    artifact_id: str,
    execution_sha256: str,
    *,
    request_record: dict[str, object],
    model_config: GptNeoConfig | None = None,
    lora_config: LoraConfig | None = None,
    runtime_resource_evidence: CalibrationResourceEvidence,
) -> tuple[TinyWorldsCalibrationEvidence, _TrainingOutcome | None]:
    manifest = _load_trial_manifest(target)
    if (
        manifest["artifact_id"] != artifact_id
        or manifest["execution_sha256"] != execution_sha256
        or manifest["request"] != request_record
    ):
        raise ValueError("calibration trial artifact identity mismatch")
    stored_resource_evidence = _resource_evidence_from_record(
        _require_record(manifest["resource_evidence"], "resource evidence")
    )
    _require_same_resource_identity(
        stored_resource_evidence,
        runtime_resource_evidence,
    )
    artifact_hashes = _require_record(manifest["artifacts"], "trial artifacts")
    expected_record_files = (
        "candidate_scores.jsonl",
        "checkpointed_transfer.jsonl",
        "parent_search.jsonl",
    )
    if tuple(sorted(artifact_hashes)) != tuple(sorted(expected_record_files)):
        raise ValueError("calibration trial artifact set changed")
    for name in expected_record_files:
        if artifact_hashes[name] != _file_sha256(target / name):
            raise ValueError(f"calibration trial checksum mismatch: {name}")
    expected_entries = {"manifest.json", *expected_record_files}
    if manifest["model"] is not None:
        expected_entries.add("model.npz")
    if {path.name for path in target.iterdir()} != expected_entries:
        raise ValueError("calibration trial directory entries changed")
    model_record_value = manifest["model"]
    model_sha256_value = manifest["model_file_sha256"]
    if model_record_value is None:
        if model_sha256_value is not None:
            raise ValueError("untrained trial must not name a model checksum")
    elif (
        type(model_sha256_value) is not str
        or model_sha256_value != _file_sha256(target / "model.npz")
    ):
        raise ValueError("calibration trial model checksum mismatch")
    evidence = _evidence_from_record(
        _require_record(manifest["evidence"], "trial evidence")
    )
    if model_config is None and lora_config is None:
        return evidence, None
    if model_config is None or lora_config is None or manifest["model"] is None:
        raise ValueError("loading trained trial state requires model metadata")
    model_record = _require_record(model_record_value, "trial model")
    adapters, graph, snapshots = _load_outcome_model(
        target / "model.npz",
        model_record,
        model_config,
        lora_config,
    )
    return evidence, _TrainingOutcome(
        independent_adapters=adapters,
        graph=graph,
        checkpoint_records=_load_jsonl_bytes(
            (target / "checkpointed_transfer.jsonl").read_bytes()
        ),
        parent_records=_load_jsonl_bytes(
            (target / "parent_search.jsonl").read_bytes()
        ),
        stability_before=snapshots,
    )


def _write_outcome_model(
    path: Path,
    outcome: _TrainingOutcome,
    model_config: GptNeoConfig,
    lora_config: LoraConfig,
) -> dict[str, object]:
    arrays: dict[str, np.ndarray] = {}
    independent_records: list[dict[str, object]] = []
    for adapter_index, (task_id, adapter) in enumerate(
        outcome.independent_adapters
    ):
        prefix = f"independent.{adapter_index:03d}."
        arrays.update(
            {
                prefix + name: np.asarray(value)
                for name, value in flatten_lora_edge(
                    adapter,
                    model_config,
                    lora_config,
                ).items()
            }
        )
        independent_records.append(
            {
                "adapter_sha256": _tree_checksum(adapter),
                "prefix": prefix,
                "task_id": str(task_id),
            }
        )
    graph_records: list[dict[str, object]] = []
    for node_index, node in enumerate(outcome.graph.nodes):
        prefix = None
        adapter_sha = None
        if node.incoming_edge is not None:
            prefix = f"vamp.{node_index:03d}."
            arrays.update(
                {
                    prefix + name: np.asarray(value)
                    for name, value in flatten_lora_edge(
                        node.incoming_edge,
                        model_config,
                        lora_config,
                    ).items()
                }
            )
            adapter_sha = _tree_checksum(node.incoming_edge)
        graph_records.append(
            {
                "adapter_sha256": adapter_sha,
                "node_id": str(node.node_id),
                "parent_id": None if node.parent_id is None else str(node.parent_id),
                "prefix": prefix,
                "train_stage": node.train_stage,
                "trained_task": (
                    None if node.trained_task is None else str(node.trained_task)
                ),
            }
        )
    with path.open("wb") as output:
        np.savez_compressed(output, **dict(sorted(arrays.items())))
        output.flush()
        os.fsync(output.fileno())
    return {
        "graph": graph_records,
        "independent_adapters": independent_records,
        "stability_before": [
            _snapshot_record(snapshot) for snapshot in outcome.stability_before
        ],
        "tensor_names": sorted(arrays),
    }


def _load_outcome_model(
    path: Path,
    record: dict[str, object],
    model_config: GptNeoConfig,
    lora_config: LoraConfig,
) -> tuple[
    tuple[tuple[TaskId, LoraEdge], ...],
    MemoryGraph[LoraEdge],
    tuple[CommittedNodeSnapshot, ...],
]:
    _require_record_fields(
        record,
        ("graph", "independent_adapters", "stability_before", "tensor_names"),
        "outcome model",
    )
    with np.load(path, allow_pickle=False) as archive:
        expected_names = tuple(
            _require_string(value, "tensor name")
            for value in _require_list(record["tensor_names"], "tensor names")
        )
        if tuple(sorted(archive.files)) != tuple(sorted(expected_names)):
            raise ValueError("calibration model tensor names changed")
        arrays = {name: np.asarray(archive[name]) for name in archive.files}

    def edge(prefix: str) -> LoraEdge:
        tensors = {
            name.removeprefix(prefix): value
            for name, value in arrays.items()
            if name.startswith(prefix)
        }
        if not tensors:
            raise ValueError(f"calibration model has no tensors for {prefix}")
        return unflatten_lora_edge(tensors, model_config, lora_config)

    independent = tuple(
        (
            TaskId(_require_string(item["task_id"], "independent task ID")),
            edge(_require_string(item["prefix"], "independent prefix")),
        )
        for value in _require_list(
            record["independent_adapters"],
            "independent adapters",
        )
        for item in (_require_record(value, "independent adapter"),)
    )
    for value, (_, adapter) in zip(
        _require_list(record["independent_adapters"], "independent adapters"),
        independent,
    ):
        item = _require_record(value, "independent adapter")
        if item["adapter_sha256"] != _tree_checksum(adapter):
            raise ValueError("independent adapter checksum mismatch")

    graph_values = _require_list(record["graph"], "graph nodes")
    if not graph_values:
        raise ValueError("calibration outcome graph is empty")
    root_record = _require_record(graph_values[0], "root graph node")
    graph_node_fields = (
        "adapter_sha256",
        "node_id",
        "parent_id",
        "prefix",
        "train_stage",
        "trained_task",
    )
    _require_record_fields(root_record, graph_node_fields, "root graph node")
    graph: MemoryGraph[LoraEdge] = init_memory_graph(
        NodeId(_require_string(root_record["node_id"], "root node ID"))
    )
    if any(
        root_record[field] is not None
        for field in (
            "adapter_sha256",
            "parent_id",
            "prefix",
            "trained_task",
        )
    ) or root_record["train_stage"] != 0:
        raise ValueError("calibration graph root metadata is invalid")
    for value in graph_values[1:]:
        item = _require_record(value, "graph node")
        _require_record_fields(item, graph_node_fields, "graph node")
        prefix = _require_string(item["prefix"], "graph adapter prefix")
        adapter = edge(prefix)
        if item["adapter_sha256"] != _tree_checksum(adapter):
            raise ValueError("VAMP graph adapter checksum mismatch")
        graph = add_memory_node(
            graph,
            node_id=NodeId(_require_string(item["node_id"], "graph node ID")),
            parent_id=NodeId(
                _require_string(item["parent_id"], "graph parent ID")
            ),
            trained_task=TaskId(
                _require_string(item["trained_task"], "graph trained task")
            ),
            train_stage=_require_integer(item["train_stage"], "train stage"),
            incoming_edge=adapter,
        )
    snapshots = tuple(
        _snapshot_from_record(_require_record(value, "stability snapshot"))
        for value in _require_list(
            record["stability_before"],
            "stability snapshots",
        )
    )
    return independent, graph, snapshots


def _resource_evidence_record(
    evidence: CalibrationResourceEvidence,
) -> dict[str, object]:
    return {
        "allocator_peak_bytes": evidence.allocator_peak_bytes,
        "allocator_peak_target_bytes": evidence.allocator_peak_target_bytes,
        "device_kind": evidence.device_kind,
        "platform": evidence.platform,
    }


def _resource_evidence_from_record(
    record: dict[str, object],
) -> CalibrationResourceEvidence:
    _require_record_fields(
        record,
        (
            "allocator_peak_bytes",
            "allocator_peak_target_bytes",
            "device_kind",
            "platform",
        ),
        "resource evidence",
    )

    def optional_integer(value: object, label: str) -> int | None:
        return None if value is None else _require_integer(value, label)

    return CalibrationResourceEvidence(
        platform=_require_string(record["platform"], "resource platform"),
        device_kind=_require_string(
            record["device_kind"],
            "resource device kind",
        ),
        allocator_peak_bytes=optional_integer(
            record["allocator_peak_bytes"],
            "allocator peak",
        ),
        allocator_peak_target_bytes=optional_integer(
            record["allocator_peak_target_bytes"],
            "allocator peak target",
        ),
    )


def _evidence_record(evidence: TinyWorldsCalibrationEvidence) -> dict[str, object]:
    binomial = lambda value: {
        "successes": value.successes,
        "trials": value.trials,
    }
    return {
        "committed_node_stability": {
            "after": [
                _snapshot_record(value)
                for value in evidence.committed_node_stability.after
            ],
            "before": [
                _snapshot_record(value)
                for value in evidence.committed_node_stability.before
            ],
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


def _evidence_from_record(
    record: dict[str, object],
) -> TinyWorldsCalibrationEvidence:
    expected_fields = (
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
    _require_record_fields(record, expected_fields, "calibration evidence")

    def binomial(name: str):
        value = _require_record(record[name], name)
        _require_record_fields(value, ("successes", "trials"), name)
        return calibration_binomial_evidence(
            _require_integer(value["successes"], f"{name} successes"),
            _require_integer(value["trials"], f"{name} trials"),
        )

    stability = _require_record(
        record["committed_node_stability"],
        "committed node stability",
    )
    _require_record_fields(stability, ("after", "before"), "node stability")
    return TinyWorldsCalibrationEvidence(
        exact_kg=binomial("exact_kg"),
        frozen_novel_binding=binomial("frozen_novel_binding"),
        independent_direct_recall=binomial("independent_direct_recall"),
        frozen_one_hop=binomial("frozen_one_hop"),
        independent_one_hop=binomial("independent_one_hop"),
        committed_node_stability=CommittedNodeStabilityEvidence(
            before=tuple(
                _snapshot_from_record(_require_record(value, "before snapshot"))
                for value in _require_list(stability["before"], "before snapshots")
            ),
            after=tuple(
                _snapshot_from_record(_require_record(value, "after snapshot"))
                for value in _require_list(stability["after"], "after snapshots")
            ),
        ),
        old_contextual_answer=binomial("old_contextual_answer"),
        revision_contextual_answer=binomial("revision_contextual_answer"),
        paired_revision_consistency=binomial("paired_revision_consistency"),
    )


def _snapshot_record(snapshot: CommittedNodeSnapshot) -> dict[str, object]:
    return {
        "adapter_sha256": snapshot.adapter_sha256,
        "answers_sha256": snapshot.answers_sha256,
        "logits_sha256": snapshot.logits_sha256,
        "node_id": snapshot.node_id,
    }


def _snapshot_from_record(record: dict[str, object]) -> CommittedNodeSnapshot:
    _require_record_fields(
        record,
        ("adapter_sha256", "answers_sha256", "logits_sha256", "node_id"),
        "committed node snapshot",
    )
    return CommittedNodeSnapshot(
        node_id=_require_string(record["node_id"], "snapshot node ID"),
        adapter_sha256=_require_string(
            record["adapter_sha256"],
            "adapter checksum",
        ),
        logits_sha256=_require_string(record["logits_sha256"], "logits checksum"),
        answers_sha256=_require_string(
            record["answers_sha256"],
            "answers checksum",
        ),
    )


def _validation_request_record(
    request: CalibrationValidationRequest,
) -> dict[str, object]:
    return {
        "config": {
            "distractor_policy": request.config.distractor_policy.value,
            "exposures_per_fact": request.config.exposures_per_fact,
            "facts_per_task": request.config.facts_per_task,
            "lora_rank": request.config.lora_rank,
            "update_budget": request.config.update_budget,
        },
        "locked_scratch_rerun": request.locked_scratch_rerun,
        "purpose": request.purpose.value,
        "trial_index": request.trial_index,
    }


def _locked_test_request_record(
    request: LockedCalibrationTestRequest,
) -> dict[str, object]:
    return {
        "config": {
            "distractor_policy": request.config.distractor_policy.value,
            "exposures_per_fact": request.config.exposures_per_fact,
            "facts_per_task": request.config.facts_per_task,
            "lora_rank": request.config.lora_rank,
            "update_budget": request.config.update_budget,
        },
        "validation_artifact_id": request.validation_artifact_id,
        "validation_trial_index": request.validation_trial_index,
    }


def _execution_preset_record(
    preset: TinyWorldsCalibrationExecutionPreset,
) -> dict[str, object]:
    return {
        "allocator_peak_target_bytes": preset.allocator_peak_target_bytes,
        "batch_size": preset.batch_size,
        "evaluation_examples_per_task": preset.evaluation_examples_per_task,
        "evaluation_microbatch_size": preset.evaluation_microbatch_size,
        "evidence_prefix_length": preset.evidence_prefix_length,
        "gradient_clip_norm": preset.gradient_clip_norm,
        "learning_rate": preset.learning_rate,
        "parent_prefix_length": preset.parent_prefix_length,
        "weight_decay": preset.weight_decay,
    }


def _model_config_record(config: GptNeoConfig) -> dict[str, object]:
    return {
        "activation": config.activation,
        "attention_dropout": config.attention_dropout,
        "attention_types": list(config.attention_types),
        "embedding_dropout": config.embedding_dropout,
        "hidden_size": config.hidden_size,
        "initializer_range": config.initializer_range,
        "intermediate_size": config.intermediate_size,
        "layer_norm_epsilon": config.layer_norm_epsilon,
        "local_window_size": config.local_window_size,
        "max_position_embeddings": config.max_position_embeddings,
        "num_heads": config.num_heads,
        "num_layers": config.num_layers,
        "residual_dropout": config.residual_dropout,
        "vocab_size": config.vocab_size,
    }


def _pool_content_sha256(
    bundle: TinyWorldsBundle,
    rendered: RenderedTinyWorlds,
    standard_groups: tuple[RenderedQueryGroup, ...],
    symbolic_bundle_sha256: str,
) -> str:
    return _digest_record(
        {
            "bundle_id": bundle.bundle_id,
            "hard_queries": _query_content_records(
                rendered.query_groups,
                tuple(DataSplit),
            ),
            "render_preset": {
                "context_length": rendered.preset.context_length,
                "root_validation_stories": (
                    rendered.preset.root_validation_stories
                ),
                "story_token_count": rendered.preset.story_token_count,
                "test_query_groups_per_task": (
                    rendered.preset.test_query_groups_per_task
                ),
                "test_stories_per_task": (
                    rendered.preset.test_stories_per_task
                ),
                "training_stories_per_task": (
                    rendered.preset.training_stories_per_task
                ),
                "validation_query_groups_per_task": (
                    rendered.preset.validation_query_groups_per_task
                ),
                "validation_stories_per_task": (
                    rendered.preset.validation_stories_per_task
                ),
            },
            "standard_queries": _query_content_records(
                standard_groups,
                tuple(DataSplit),
            ),
            "story_records": _story_content_records(
                rendered,
                tuple(DataSplit),
            ),
            "symbolic_bundle_sha256": symbolic_bundle_sha256,
            "task_topology": _task_topology_records(bundle),
        }
    )


def _validation_selection_content_sha256(
    bundle: TinyWorldsBundle,
    rendered: RenderedTinyWorlds,
    standard_groups: tuple[RenderedQueryGroup, ...],
) -> str:
    """Hash only records permitted to influence validation-time selection."""
    selection_splits = (DataSplit.TRAIN, DataSplit.VALIDATION)
    return _digest_record(
        {
            "bundle_id": bundle.bundle_id,
            "hard_validation_queries": _query_content_records(
                rendered.query_groups,
                (DataSplit.VALIDATION,),
            ),
            "render_preset": {
                "context_length": rendered.preset.context_length,
                "root_validation_stories": (
                    rendered.preset.root_validation_stories
                ),
                "story_token_count": rendered.preset.story_token_count,
                "training_stories_per_task": (
                    rendered.preset.training_stories_per_task
                ),
                "validation_query_groups_per_task": (
                    rendered.preset.validation_query_groups_per_task
                ),
                "validation_stories_per_task": (
                    rendered.preset.validation_stories_per_task
                ),
            },
            "standard_validation_queries": _query_content_records(
                standard_groups,
                (DataSplit.VALIDATION,),
            ),
            "story_records": _story_content_records(
                rendered,
                selection_splits,
            ),
            "task_topology": _task_topology_records(bundle),
        }
    )


def _task_topology_records(
    bundle: TinyWorldsBundle,
) -> list[dict[str, object]]:
    return [
        {
            "direct_fact_ids": [str(value) for value in task.direct_fact_ids],
            "family_id": str(task.family_id),
            "incoming_edge_id": (
                None
                if task.incoming_edge_id is None
                else str(task.incoming_edge_id)
            ),
            "introduced_entity_ids": [
                str(value) for value in task.introduced_entity_ids
            ],
            "kind": task.kind.value,
            "parent_task_id": (
                None
                if task.parent_task_id is None
                else str(task.parent_task_id)
            ),
            "rule_ids": [str(value) for value in task.rule_ids],
            "task_id": str(task.task_id),
        }
        for task in bundle.tasks
    ]


def _story_content_records(
    rendered: RenderedTinyWorlds,
    splits: tuple[DataSplit, ...],
) -> list[dict[str, object]]:
    return [
        {
            "alignments": [
                {
                    "end_character": alignment.end_character,
                    "fact_ids": list(alignment.fact_ids),
                    "rule_ids": list(alignment.rule_ids),
                    "sentence_index": alignment.sentence_index,
                    "start_character": alignment.start_character,
                }
                for alignment in story.alignments
            ],
            "plot_id": story.plot_id,
            "purpose": story.purpose,
            "split": story.split.value,
            "story_id": story.story_id,
            "task_id": story.task_id,
            "template_family_ids": list(story.template_family_ids),
            "text_sha256": story.text_sha256,
            "token_ids_sha256": _integer_tuple_sha256(story.token_ids),
        }
        for story in rendered.stories
        if story.split in splits
    ]


def _query_content_records(
    groups: tuple[RenderedQueryGroup, ...],
    splits: tuple[DataSplit, ...],
) -> list[dict[str, object]]:
    return [
        {
            "candidate_answers": [
                candidate.answer_text for candidate in query.candidates
            ],
            "candidate_batches": [
                _language_batch_content_record(candidate.competence_batch)
                for candidate in query.candidates
            ],
            "candidate_entity_ids": list(variant.candidate_entity_ids),
            "correct_candidate_index": query.correct_candidate_index,
            "cue_regime": query.cue_regime,
            "eligible_task_ids": [str(value) for value in query.eligible_task_ids],
            "family_id": query.family_id,
            "group_id": group.group_id,
            "holdout_identity_sha256": (
                group.group_plan.holdout_identity_sha256
            ),
            "mode": query.mode,
            "novelty_regime": query.novelty_regime,
            "oracle_node_ids": [str(value) for value in query.oracle_node_ids],
            "prefix_length": query.prefix_length,
            "prefix_sha256": variant.text_sha256,
            "prefix_token_ids_sha256": _integer_tuple_sha256(
                variant.prefix_token_ids
            ),
            "proof_id": query.proof_id,
            "query_core_sha256": variant.query_core_sha256,
            "query_id": query.query_id,
            "query_kind": query.query_kind,
            "reasoning_depth": query.reasoning_depth,
            "reasoning_type": query.reasoning_type,
            "required_edge_ids": [str(value) for value in query.required_edge_ids],
            "router_batch": _language_batch_content_record(query.router_batch),
            "source_query_id": group.symbolic_query_id,
            "split": group.split.value,
            "support_ids": list(query.support_ids),
            "task_id": group.task_id,
            "template_family_id": group.group_plan.template_family_id,
            "variant_id": variant.variant_id,
            "visible_cue_ids": list(query.visible_cue_ids),
        }
        for group in groups
        if group.split in splits
        for variant in group.variants
        for query in (variant.knowledge_query,)
    ]


def _integer_tuple_sha256(values: tuple[int, ...]) -> str:
    array = np.asarray(values, dtype=np.int64)
    return _array_sha256(array)


def _language_batch_content_record(
    batch: RouterBatch | CompetenceBatch,
) -> dict[str, object]:
    return {
        field_name: _array_sha256(np.asarray(getattr(batch, field_name)))
        for field_name in (
            "attention_mask",
            "input_ids",
            "loss_mask",
            "target_ids",
        )
    }


def _rng_key(namespace_sha256: str, label: str) -> jax.Array:
    digest = sha256(f"{namespace_sha256}:{label}".encode("utf-8")).digest()
    return jax.random.PRNGKey(int.from_bytes(digest[:4], "big"))


def _tree_checksum(tree: object) -> str:
    digest = sha256()
    leaves, structure = jax.tree_util.tree_flatten(tree)
    digest.update(str(structure).encode("utf-8"))
    for leaf in leaves:
        value = np.asarray(leaf)
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    value = np.asarray(array)
    digest = sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _digest_record(record: dict[str, object]) -> str:
    return sha256(_canonical_json_bytes(record)).hexdigest()


def _canonical_json_bytes(record: dict[str, object]) -> bytes:
    return (
        json.dumps(
            record,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _write_canonical_json(path: Path, record: dict[str, object]) -> None:
    with path.open("wb") as output:
        output.write(_canonical_json_bytes(record))
        output.flush()
        os.fsync(output.fileno())


def _load_canonical_json(path: Path) -> dict[str, object]:
    payload = path.read_bytes()

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid canonical JSON: {path}") from error
    record = _require_record(value, str(path))
    if payload != _canonical_json_bytes(record):
        raise ValueError(f"non-canonical JSON: {path}")
    return record


def _jsonl_bytes(records: tuple[dict[str, object], ...]) -> bytes:
    return b"".join(_canonical_json_bytes(record) for record in records)


def _load_jsonl_bytes(payload: bytes) -> tuple[dict[str, object], ...]:
    if payload and not payload.endswith(b"\n"):
        raise ValueError("calibration JSONL must end with a newline")
    records: list[dict[str, object]] = []
    for line in payload.splitlines(keepends=True):
        try:
            value = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("invalid calibration JSONL") from error
        record = _require_record(value, "JSONL record")
        if line != _canonical_json_bytes(record):
            raise ValueError("non-canonical calibration JSONL")
        records.append(record)
    return tuple(records)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_tree(directory: Path) -> None:
    for path in directory.rglob("*"):
        if path.is_file():
            with path.open("rb") as source:
                os.fsync(source.fileno())
    for path in sorted(
        (value for value in directory.rglob("*") if value.is_dir()),
        key=lambda value: len(value.parts),
        reverse=True,
    ):
        _fsync_directory(path)
    _fsync_directory(directory)


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_record(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _require_record_fields(
    record: dict[str, object],
    fields: tuple[str, ...],
    label: str,
) -> None:
    expected = set(fields)
    actual = set(record)
    if actual != expected:
        raise ValueError(
            f"{label} fields changed; missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _require_list(value: object, label: str) -> list[object]:
    if type(value) is not list:
        raise ValueError(f"{label} must be a JSON array")
    return value


def _require_string(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _require_integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    return value


def _require_float(value: object, label: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _require_sha256(value: object, label: str) -> None:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")


__all__ = [
    "CALIBRATION_ACCELERATOR_ARTIFACT_VERSION",
    "CALIBRATION_ALLOCATOR_PEAK_TARGET_BYTES",
    "CALIBRATION_EXECUTION_PRESET",
    "CALIBRATION_MAX_EXPOSURES_PER_FACT",
    "CALIBRATION_MAX_FACTS_PER_TASK",
    "CALIBRATION_RENDER_PRESET",
    "CALIBRATION_REQUIRED_DEVICE_KIND",
    "CALIBRATION_REQUIRED_PLATFORM",
    "CalibrationResourceEvidence",
    "TinyWorldsAcceleratorCalibrationEvaluator",
    "TinyWorldsCalibrationExecutionPreset",
    "TinyWorldsCalibrationPool",
    "build_tinyworlds_calibration_pool",
    "load_calibration_trial_resource_evidence",
    "measure_calibration_resource_evidence",
    "render_tinyworlds_calibration_pool",
    "validate_calibration_resource_evidence",
]

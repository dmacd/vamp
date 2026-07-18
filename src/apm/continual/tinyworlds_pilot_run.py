"""Resource-bound execution engine for the fixed TinyWorlds v1 pilot.

The report layer deliberately accepts only an immutable completed result.  This
module owns the complementary side of that boundary: loading the locked
calibration profile, executing real adapter training and candidate scoring,
persisting stage checkpoints, and assembling the evidence consumed by the
report projector.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from time import monotonic
from typing import Protocol, cast

import jax
import jax.numpy as jnp
import numpy as np

from apm.continual.knowledge_evaluation import (
    KnowledgeAddressDecision,
    KnowledgeMethodEvaluation,
    aggregate_knowledge_evaluations,
    evaluate_ebt_knowledge_methods,
    evaluate_knowledge_method,
)
from apm.continual.knowledge_tasks import KnowledgeQuery
from apm.continual.knowledge_training import (
    CounterfactualRole,
    KnowledgeCounterfactualTraining,
    KnowledgeParentContext,
    KnowledgeTransferDiagnostics,
    KnowledgeValidationSuite,
    ParentTransferTrialDiagnostic,
    ParentCounterfactualPlan,
    TransferCheckpointDiagnostic,
    commit_selected_counterfactual_edge,
    plan_parent_counterfactuals,
    run_parent_counterfactuals,
    select_knowledge_parent_from_scores,
    validate_parent_counterfactual_resume,
)
from apm.continual.language_adaptation_artifact import (
    AdapterTrainingRecord,
    LanguageAdaptationArtifact,
    LanguageAdaptationRngState,
    VampTrainingRecord,
    flatten_lora_edge,
    load_language_adaptation_artifact,
    save_language_adaptation_artifact,
)
from apm.continual.language_baseline_training import (
    IndependentRootAdapter,
    IndependentRootLoraRun,
    SequentialLoraStage,
    SequentialLoraRun,
    pack_root_adapter,
)
from apm.continual.language_benchmark_metrics import account_language_memory
from apm.continual.language_benchmark_run import measure_peak_device_memory
from apm.continual.language_benchmarks import (
    ROUTER_BASELINE_NAMES,
    STORED_BASELINE_NAMES,
)
from apm.continual.language_routing import (
    LanguageAddressDecision,
    competence_nll_by_node,
    route_language_prefix,
)
from apm.continual.language_run import LanguageStageMetrics
from apm.continual.language_tasks import (
    AddressBook,
    CompetenceBatch,
    LanguageCurriculum,
    LanguageTask,
    RouterBatch,
)
from apm.continual.tinyworlds_calibration import (
    CalibrationDistractorPolicy,
    TinyWorldsCalibrationProfile,
)
from apm.continual.tinyworlds_calibration_profile import (
    calibration_profile_sha256,
    load_calibration_profile,
)
from apm.continual.tinyworlds_progress import (
    TinyWorldsProgressWriter,
    TinyWorldsSequentialResult,
)
from apm.continual.tinyworlds_report import (
    TINYWORLDS_ADDRESSING_METHODS,
    TINYWORLDS_ALLOCATOR_PEAK_LIMIT_BYTES,
    TINYWORLDS_IMPLEMENTATION_GATES,
    TINYWORLDS_NATURAL_CONTINUATION_METHODS,
    TINYWORLDS_PARENT_COUNTERFACTUALS,
    TINYWORLDS_REPORT_METHODS,
    TINYWORLDS_REPORT_STAGES,
    TINYWORLDS_REPORT_TASK_IDS,
    TinyWorldsCompletedResult,
    TinyWorldsRecord,
    TinyWorldsReportManifest,
    canonical_tinyworlds_config_json,
)
from apm.data.text.tinyworlds.adapters import (
    PreparedTinyWorldsCurriculum,
    TinyWorldsTrainingDataConfig,
    prepare_tinyworlds_curriculum,
)
from apm.data.text.tinyworlds.closure import answer_query
from apm.data.text.tinyworlds.persistence import (
    load_tinyworlds_manifest,
    write_tinyworlds_bundle,
)
from apm.data.text.tinyworlds.query_generation import (
    TinyWorldsBundle,
    apply_standard_distractor_mix,
    generate_pilot_bundle,
)
from apm.data.text.tinyworlds.rendered_persistence import (
    write_rendered_tinyworlds_bundle,
)
from apm.data.text.tinyworlds.rendering import (
    RenderedTinyWorlds,
    TinyWorldsRenderPreset,
    render_tinyworlds_bundle,
)
from apm.data.text.tinyworlds.schema import (
    CandidateRole,
    DataSplit,
    QueryKind,
    TaskId as SymbolicTaskId,
    TaskKind,
)
from apm.data.text.tinyworlds.seeds import (
    derive_master_seed,
    derive_subseed,
    subseed_uint64,
)
from apm.data.text.tinyworlds.world_generation import (
    PILOT_TASK_IDS,
    TINYWORLDS_VERSION,
)
from apm.lm.candidate_scoring import (
    score_edge_coefficient_candidates,
    score_frozen_base_candidates,
    score_hard_node_candidates,
)
from apm.lm.config import GptNeoConfig
from apm.lm.checkpoint import parameter_checksum
from apm.lm.lora import LoraConfig, LoraEdge, init_lora_edge
from apm.lm.lora_memory import (
    PackedLoraMemory,
    edge_coefficients_for_node,
    pack_lora_memory,
)
from apm.lm.parameters import GptNeoParams
from apm.lm.text import TextTokenizer, TokenizersTextTokenizer
from apm.lm.tinystories_conversion import (
    LoadedTinyStoriesArtifact,
    load_tinystories_artifact,
)
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
from apm.memory.content_keys import add_address_key, derive_node_content_key
from apm.memory.graph import (
    MemoryGraph,
    NodeId,
    TaskId,
    init_memory_graph,
    memory_node_ids,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
BASE_ARTIFACT_DIRECTORY = REPOSITORY_ROOT / "checkpoints" / "tinystories-8m"
CALIBRATION_RESULTS_ROOT = (
    REPOSITORY_ROOT
    / "results"
    / "language_cl"
    / "tinyworlds-v1"
    / "knowledge-graph"
)
PILOT_TRAINING_CACHE_ROOT = CALIBRATION_RESULTS_ROOT / ".training-cache"
PUBLIC_SEED = 0
PILOT_PRESET_NAME = "single-gpu"
PILOT_EVALUATION_MICROBATCH_SIZE = 8
PILOT_RANDOM_ROUTER_SEED = 0
PILOT_FORWARD_TRANSFER_POLICY = "root_path_until_task_is_committed"

_SHA256_PATTERN_LENGTH = 64
_GIB = 1024**3
_TRANSFER_CHUNK_FORMAT = "apm.tinyworlds.transfer-chunk"
_TRANSFER_CHUNK_SCHEMA_VERSION = 1
_TRANSFER_STATE_DIRECTORY = "training-state"
_BASELINE_RESUME_FORMAT = "apm.tinyworlds.pilot-baseline-resume"
_BASELINE_RESUME_SCHEMA_VERSION = 1
_BASELINE_UPDATE_PATTERN = re.compile(r"update-([0-9]{7})\Z")
_BASELINE_TEMP_PATTERN = re.compile(
    r"\.update-[0-9]{7}\.tmp-[A-Za-z0-9_-]+\Z"
)


@dataclass(frozen=True, slots=True)
class TinyWorldsPilotExecutionPreset:
    """Fixed accelerator and rendering dimensions for the canonical pilot."""

    render_preset: TinyWorldsRenderPreset = TinyWorldsRenderPreset()
    evaluation_microbatch_size: int = PILOT_EVALUATION_MICROBATCH_SIZE
    random_router_seed: int = PILOT_RANDOM_ROUTER_SEED
    allocator_peak_limit_bytes: int = TINYWORLDS_ALLOCATOR_PEAK_LIMIT_BYTES

    def __post_init__(self) -> None:
        if not isinstance(self.render_preset, TinyWorldsRenderPreset):
            raise TypeError("render_preset must be a TinyWorldsRenderPreset")
        if (
            type(self.evaluation_microbatch_size) is not int
            or self.evaluation_microbatch_size <= 0
            or type(self.random_router_seed) is not int
            or self.random_router_seed < 0
            or type(self.allocator_peak_limit_bytes) is not int
            or self.allocator_peak_limit_bytes <= 0
        ):
            raise ValueError("pilot execution dimensions must be positive integers")


TINYWORLDS_PILOT_EXECUTION_PRESET = TinyWorldsPilotExecutionPreset()


@dataclass(frozen=True, slots=True)
class TinyWorldsPilotInputs:
    """Verified profile, world, rendered data, base model, and fixed configs."""

    profile: TinyWorldsCalibrationProfile
    profile_sha256: str
    master_seed_sha256: str
    symbolic_bundle: TinyWorldsBundle
    rendered: RenderedTinyWorlds
    prepared: PreparedTinyWorldsCurriculum
    base_artifact: LoadedTinyStoriesArtifact
    tokenizer: TextTokenizer
    lora_config: LoraConfig
    train_config: LmTrainConfig
    execution_preset: TinyWorldsPilotExecutionPreset

    def __post_init__(self) -> None:
        if not isinstance(self.profile, TinyWorldsCalibrationProfile):
            raise TypeError("profile must be a TinyWorldsCalibrationProfile")
        for label, digest in (
            ("profile_sha256", self.profile_sha256),
            ("master_seed_sha256", self.master_seed_sha256),
        ):
            if (
                type(digest) is not str
                or len(digest) != _SHA256_PATTERN_LENGTH
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"{label} must be a lowercase SHA-256")
        if not isinstance(self.symbolic_bundle, TinyWorldsBundle):
            raise TypeError("symbolic_bundle must be a TinyWorldsBundle")
        if not isinstance(self.rendered, RenderedTinyWorlds):
            raise TypeError("rendered must be a RenderedTinyWorlds")
        if not isinstance(self.prepared, PreparedTinyWorldsCurriculum):
            raise TypeError("prepared must be a PreparedTinyWorldsCurriculum")
        if not isinstance(self.lora_config, LoraConfig):
            raise TypeError("lora_config must be a LoraConfig")
        if not isinstance(self.train_config, LmTrainConfig):
            raise TypeError("train_config must be an LmTrainConfig")
        if not isinstance(self.execution_preset, TinyWorldsPilotExecutionPreset):
            raise TypeError("execution_preset has the wrong type")
        task_order = tuple(str(task.task_id) for task in self.symbolic_bundle.tasks)
        if task_order != TINYWORLDS_REPORT_TASK_IDS:
            raise ValueError("pilot inputs must use the canonical interleaved task order")
        if self.prepared.rendered_bundle_id != self.rendered.bundle_id:
            raise ValueError("prepared curriculum must come from the rendered bundle")


@dataclass(frozen=True, slots=True)
class TinyWorldsPilotStageResult:
    """All reportable evidence produced after one committed pilot stage."""

    stage: int
    method_evaluations: tuple[KnowledgeMethodEvaluation, ...]
    natural_continuation_metrics: tuple[TinyWorldsRecord, ...]
    parent_search: tuple[TinyWorldsRecord, ...]
    checkpointed_transfer: tuple[TinyWorldsRecord, ...]
    committed_node_drift: tuple[TinyWorldsRecord, ...]
    memory_metrics: tuple[TinyWorldsRecord, ...]
    addressing_cost: tuple[TinyWorldsRecord, ...]
    sequential_result: TinyWorldsRecord

    def __post_init__(self) -> None:
        if self.stage not in TINYWORLDS_REPORT_STAGES:
            raise ValueError("pilot stage is outside the canonical eight stages")
        represented = tuple(item.method for item in self.method_evaluations)
        if represented != TINYWORLDS_REPORT_METHODS or any(
            item.stage != self.stage for item in self.method_evaluations
        ):
            raise ValueError("stage evaluations must follow the canonical method order")
        for label, records in (
            ("natural_continuation_metrics", self.natural_continuation_metrics),
            ("parent_search", self.parent_search),
            ("checkpointed_transfer", self.checkpointed_transfer),
            ("committed_node_drift", self.committed_node_drift),
            ("memory_metrics", self.memory_metrics),
            ("addressing_cost", self.addressing_cost),
        ):
            if not isinstance(records, tuple) or not records or any(
                not isinstance(record, TinyWorldsRecord) for record in records
            ):
                raise ValueError(f"{label} must contain TinyWorlds records")
        if not isinstance(self.sequential_result, TinyWorldsRecord):
            raise TypeError("sequential_result must be a TinyWorldsRecord")


@dataclass(frozen=True, slots=True)
class TinyWorldsPilotFinalEvidence:
    """Cross-stage records that become meaningful after all eight tasks."""

    graph_recovery: tuple[TinyWorldsRecord, ...]
    revision_retention: tuple[TinyWorldsRecord, ...]
    gate_results: tuple[TinyWorldsRecord, ...]
    representative_queries: tuple[TinyWorldsRecord, ...]
    selection_audit: tuple[TinyWorldsRecord, ...]

    def __post_init__(self) -> None:
        for label, records in (
            ("graph_recovery", self.graph_recovery),
            ("revision_retention", self.revision_retention),
            ("gate_results", self.gate_results),
            ("representative_queries", self.representative_queries),
            ("selection_audit", self.selection_audit),
        ):
            if not isinstance(records, tuple) or not records or any(
                not isinstance(record, TinyWorldsRecord) for record in records
            ):
                raise ValueError(f"{label} must contain TinyWorlds records")


@dataclass(frozen=True, slots=True)
class _VampStageState:
    graph: MemoryGraph[LoraEdge]
    address_book: AddressBook
    rng_key: jax.Array
    stage_metrics: tuple[LanguageStageMetrics, ...]
    parent_diagnostics: tuple[KnowledgeTransferDiagnostics, ...]


@dataclass(frozen=True, slots=True)
class _TrainedPilotAdaptations:
    sequential: SequentialLoraRun
    independent: IndependentRootLoraRun
    sequential_rng_by_stage: tuple[jax.Array, ...]
    independent_rng_by_stage: tuple[jax.Array, ...]
    vamp_stages: tuple[_VampStageState, ...]
    training_workspace: Path

    def __post_init__(self) -> None:
        if tuple(stage.stage_metrics[-1].stage_index for stage in self.vamp_stages) != (
            TINYWORLDS_REPORT_STAGES
        ):
            raise ValueError("trained VAMP snapshots must cover all eight stages")
        if (
            len(self.sequential_rng_by_stage) != len(TINYWORLDS_REPORT_STAGES)
            or len(self.independent_rng_by_stage) != len(TINYWORLDS_REPORT_STAGES)
        ):
            raise ValueError("baseline RNG snapshots must cover all eight stages")
        if not isinstance(self.training_workspace, Path):
            raise TypeError("training_workspace must be a Path")


@dataclass(frozen=True, slots=True)
class _SequentialTrainingResult:
    run: SequentialLoraRun
    rng_by_stage: tuple[jax.Array, ...]


@dataclass(frozen=True, slots=True)
class _IndependentTrainingResult:
    run: IndependentRootLoraRun
    rng_by_stage: tuple[jax.Array, ...]


class PilotStageExecutor(Protocol):
    """Injectable CPU-test seam for one already-prepared pilot stage."""

    def __call__(self, stage: int) -> TinyWorldsPilotStageResult:
        """Return real or fixture-backed evidence for one canonical stage."""


def tinyworlds_record(**values: str | int | float | bool | None) -> TinyWorldsRecord:
    """Construct one immutable scalar record without exposing tuple boilerplate."""
    return TinyWorldsRecord(tuple(values.items()))


def assemble_tinyworlds_completed_result(
    manifest: TinyWorldsReportManifest,
    stages: tuple[TinyWorldsPilotStageResult, ...],
    final_evidence: TinyWorldsPilotFinalEvidence,
) -> TinyWorldsCompletedResult:
    """Assemble and validate a completed pilot from immutable stage outputs."""
    if not isinstance(manifest, TinyWorldsReportManifest):
        raise TypeError("manifest must be a TinyWorldsReportManifest")
    if tuple(item.stage for item in stages) != TINYWORLDS_REPORT_STAGES:
        raise ValueError("pilot stage results must cover stages one through eight")
    if not isinstance(final_evidence, TinyWorldsPilotFinalEvidence):
        raise TypeError("final_evidence has the wrong type")
    return TinyWorldsCompletedResult(
        manifest=manifest,
        method_evaluations=tuple(
            evaluation for stage in stages for evaluation in stage.method_evaluations
        ),
        natural_continuation_metrics=tuple(
            record for stage in stages for record in stage.natural_continuation_metrics
        ),
        parent_search=tuple(
            record for stage in stages for record in stage.parent_search
        ),
        checkpointed_transfer=tuple(
            record for stage in stages for record in stage.checkpointed_transfer
        ),
        graph_recovery=final_evidence.graph_recovery,
        revision_retention=final_evidence.revision_retention,
        committed_node_drift=tuple(
            record for stage in stages for record in stage.committed_node_drift
        ),
        memory_metrics=tuple(
            record for stage in stages for record in stage.memory_metrics
        ),
        addressing_cost=tuple(
            record for stage in stages for record in stage.addressing_cost
        ),
        gate_results=final_evidence.gate_results,
        representative_queries=final_evidence.representative_queries,
        selection_audit=final_evidence.selection_audit,
        sequential_results=tuple(stage.sequential_result for stage in stages),
    )


def execute_tinyworlds_pilot_stages(
    executor: PilotStageExecutor,
    progress: TinyWorldsProgressWriter,
) -> tuple[TinyWorldsPilotStageResult, ...]:
    """Execute eight ordered stages and durably append each stage summary."""
    if not callable(executor):
        raise TypeError("executor must be callable")
    if not isinstance(progress, TinyWorldsProgressWriter):
        raise TypeError("progress must be a TinyWorldsProgressWriter")
    stages: list[TinyWorldsPilotStageResult] = []
    for stage in TINYWORLDS_REPORT_STAGES:
        result = executor(stage)
        if not isinstance(result, TinyWorldsPilotStageResult) or result.stage != stage:
            raise ValueError("stage executor returned out-of-order evidence")
        progress.append_sequential(
            TinyWorldsSequentialResult(
                sequence_index=stage - 1,
                stage=stage,
                payload=_without_reserved_stage_fields(result.sequential_result),
            )
        )
        progress.flush_sequential()
        stages.append(result)
    return tuple(stages)


def load_locked_tinyworlds_profile(
    calibration_results_root: str | Path = CALIBRATION_RESULTS_ROOT,
) -> tuple[Path, TinyWorldsCalibrationProfile, str]:
    """Find and strictly load the one successful canonical calibration profile."""
    root = Path(calibration_results_root)
    candidates = tuple(
        sorted(
            path.parent
            for path in root.glob(
                "calibration-seed0-*/calibration_profile.json"
            )
            if path.is_file() and not path.is_symlink()
        )
    )
    if not candidates:
        raise FileNotFoundError(
            "TinyWorlds pilot requires a passing calibration_profile.json"
        )
    loaded = tuple(
        (directory, load_calibration_profile(directory))
        for directory in candidates
    )
    identities = {
        calibration_profile_sha256(profile) for _, profile in loaded
    }
    if len(identities) != 1:
        raise RuntimeError(
            "multiple distinct passing calibration profiles are present"
        )
    directory, profile = loaded[-1]
    digest = calibration_profile_sha256(profile)
    return directory, profile, digest


def prepare_fixed_tinyworlds_pilot_inputs(
    temporary_directory: str | Path,
    *,
    calibration_results_root: str | Path = CALIBRATION_RESULTS_ROOT,
    base_artifact_directory: str | Path = BASE_ARTIFACT_DIRECTORY,
    execution_preset: TinyWorldsPilotExecutionPreset = (
        TINYWORLDS_PILOT_EXECUTION_PRESET
    ),
) -> TinyWorldsPilotInputs:
    """Load identities, generate/render the pilot, and prepare fixed training data."""
    temporary = Path(temporary_directory)
    temporary.mkdir(parents=True, exist_ok=True)
    _, profile, profile_digest = load_locked_tinyworlds_profile(
        calibration_results_root
    )
    base_artifact = load_tinystories_artifact(Path(base_artifact_directory))
    tokenizer_path = Path(base_artifact_directory) / "tokenizer" / "tokenizer.json"
    tokenizer = TokenizersTextTokenizer.from_file(tokenizer_path)
    _validate_profile_base_identity(profile, base_artifact, tokenizer_path)
    identity = profile.identity
    master_seed = derive_master_seed(
        identity.benchmark_version,
        identity.public_seed,
        identity.base_manifest_sha256,
        identity.base_parameter_checksum,
    )
    selected = profile.selected_config
    symbolic = generate_pilot_bundle(
        master_seed,
        direct_facts_per_task=(
            36 if selected.facts_per_task == 36 else 24
        ),
    )
    if selected.distractor_policy is CalibrationDistractorPolicy.STANDARD_MIX:
        symbolic = apply_standard_distractor_mix(symbolic)
    symbolic_directory = temporary / "symbolic-pilot"
    symbolic_manifest = write_tinyworlds_bundle(symbolic, symbolic_directory)
    rendered = render_tinyworlds_bundle(
        symbolic,
        tokenizer,
        execution_preset.render_preset,
    )
    write_rendered_tinyworlds_bundle(
        rendered,
        symbolic,
        tokenizer,
        temporary / "rendered-pilot",
    )
    training_data_config = TinyWorldsTrainingDataConfig(
        facts_per_task=selected.facts_per_task,
        exposures_per_fact=selected.exposures_per_fact,
        batch_size=32,
        context_length=256,
        evaluation_examples_per_task=128,
    )
    prepared = prepare_tinyworlds_curriculum(
        rendered,
        tokenizer,
        training_data_config,
    )
    lora_config = LoraConfig(
        rank=selected.lora_rank,
        alpha=float(selected.lora_rank),
    )
    train_config = LmTrainConfig(
        learning_rate=1e-3,
        steps=selected.update_budget,
        batch_size=training_data_config.batch_size,
        weight_decay=0.01,
        gradient_clip_norm=1.0,
    )
    _write_json_atomic(
        temporary / "prepared_inputs.json",
        {
            "master_seed_sha256": master_seed,
            "pilot_bundle_sha256": symbolic_manifest.bundle_sha256,
            "profile_sha256": profile_digest,
            "rendered_bundle_id": rendered.bundle_id,
            "selected_config": _json_ready(asdict(selected)),
            "training_story_count": len(prepared.training_story_ids),
            "validation_query_count": len(prepared.validation_queries),
            "test_query_count": len(prepared.test_queries),
        },
    )
    return TinyWorldsPilotInputs(
        profile=profile,
        profile_sha256=profile_digest,
        master_seed_sha256=master_seed,
        symbolic_bundle=symbolic,
        rendered=rendered,
        prepared=prepared,
        base_artifact=base_artifact,
        tokenizer=tokenizer,
        lora_config=lora_config,
        train_config=train_config,
        execution_preset=execution_preset,
    )


def _validate_profile_base_identity(
    profile: TinyWorldsCalibrationProfile,
    base_artifact: LoadedTinyStoriesArtifact,
    tokenizer_path: Path,
) -> None:
    checkpoint = base_artifact.checkpoint
    expected = (
        (
            profile.identity.base_manifest_sha256,
            checkpoint.reference.manifest_sha256,
            "base manifest",
        ),
        (
            profile.identity.base_parameter_checksum,
            checkpoint.reference.parameter_checksum,
            "base parameter",
        ),
        (
            profile.identity.tokenizer_sha256,
            sha256(tokenizer_path.read_bytes()).hexdigest(),
            "tokenizer",
        ),
    )
    mismatch = tuple(
        label for actual, wanted, label in expected if actual != wanted
    )
    if mismatch:
        raise ValueError(
            f"calibration profile identity mismatch: {', '.join(mismatch)}"
        )


def _pilot_training_identity_sha256(inputs: TinyWorldsPilotInputs) -> str:
    checkpoint = inputs.base_artifact.checkpoint
    return sha256(
        _canonical_json_compact_bytes(
            {
                "base_manifest_sha256": checkpoint.reference.manifest_sha256,
                "base_parameter_checksum": (
                    checkpoint.reference.parameter_checksum
                ),
                "config_hashes": dict(_adaptation_config_hashes(inputs)),
                "master_seed_sha256": inputs.master_seed_sha256,
                "profile_sha256": inputs.profile_sha256,
                "rendered_bundle_id": inputs.rendered.bundle_id,
                "task_order": [
                    str(task.task_id)
                    for task in inputs.prepared.language.curriculum.tasks
                ],
                "training_story_ids": list(inputs.prepared.training_story_ids),
                "validation_query_ids": [
                    query.query_id for query in inputs.prepared.validation_queries
                ],
                "version": 1,
            }
        )
    ).hexdigest()


def _pilot_training_workspace(
    inputs: TinyWorldsPilotInputs,
    cache_root: str | Path,
    execution_sha256: str,
) -> Path:
    workspace = Path(cache_root) / execution_sha256
    if workspace.is_symlink():
        raise ValueError("pilot training workspace cannot be a symlink")
    workspace.mkdir(parents=True, exist_ok=True)
    _write_or_verify_json(
        workspace / "identity.json",
        {
            "base_parameter_checksum": (
                inputs.base_artifact.checkpoint.reference.parameter_checksum
            ),
            "execution_sha256": execution_sha256,
            "format": "apm.tinyworlds.pilot-training-workspace",
            "profile_sha256": inputs.profile_sha256,
            "rendered_bundle_id": inputs.rendered.bundle_id,
            "schema_version": 1,
        },
    )
    return workspace


def _train_resumable_sequential_lora(
    curriculum: LanguageCurriculum,
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    lora_config: LoraConfig,
    train_config: LmTrainConfig,
    rng_key: jax.Array,
    resume_root: Path,
    execution_sha256: str,
) -> SequentialLoraRun:
    return _train_resumable_sequential_state(
        curriculum,
        base_params,
        model_config,
        lora_config,
        train_config,
        rng_key,
        resume_root,
        execution_sha256,
    ).run


def _train_resumable_sequential_state(
    curriculum: LanguageCurriculum,
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    lora_config: LoraConfig,
    train_config: LmTrainConfig,
    rng_key: jax.Array,
    resume_root: Path,
    execution_sha256: str,
) -> _SequentialTrainingResult:
    base_checksum = parameter_checksum(base_params, model_config)
    initialization_key, current_rng_key = jax.random.split(rng_key)
    current_adapter = init_lora_edge(
        initialization_key,
        model_config,
        lora_config,
    )
    memory = _empty_baseline_memory(model_config, lora_config)
    stages: list[SequentialLoraStage] = []
    rng_by_stage: list[jax.Array] = []
    for task_index, task in enumerate(curriculum.tasks, start=1):
        initial_state = init_candidate_lora_train_state(
            current_adapter,
            current_rng_key,
            train_config,
        )
        task_execution_sha256 = _baseline_task_execution_sha256(
            execution_sha256,
            "sequential",
            task_index,
            task,
            initial_state,
        )
        final_state, step_losses = _train_resumable_baseline_task(
            resume_root / f"{task_index:02d}-{task.task_id}",
            "sequential",
            task_index,
            task,
            task_execution_sha256,
            initial_state,
            base_params,
            model_config,
            memory,
            lora_config,
            train_config,
        )
        current_adapter = final_state.trainable
        current_rng_key = final_state.rng_key
        rng_by_stage.append(current_rng_key)
        stages.append(
            SequentialLoraStage(
                stage_index=task_index,
                task_id=task.task_id,
                adapter=current_adapter,
                step_losses=step_losses,
            )
        )
    if parameter_checksum(base_params, model_config) != base_checksum:
        raise RuntimeError("sequential training mutated the frozen base")
    return _SequentialTrainingResult(
        run=SequentialLoraRun(
            stages=tuple(stages),
            rng_key=current_rng_key,
            train_config=train_config,
            base_parameter_checksum=base_checksum,
        ),
        rng_by_stage=tuple(rng_by_stage),
    )


def _train_resumable_independent_lora(
    curriculum: LanguageCurriculum,
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    lora_config: LoraConfig,
    train_config: LmTrainConfig,
    rng_key: jax.Array,
    resume_root: Path,
    execution_sha256: str,
) -> IndependentRootLoraRun:
    return _train_resumable_independent_state(
        curriculum,
        base_params,
        model_config,
        lora_config,
        train_config,
        rng_key,
        resume_root,
        execution_sha256,
    ).run


def _train_resumable_independent_state(
    curriculum: LanguageCurriculum,
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    lora_config: LoraConfig,
    train_config: LmTrainConfig,
    rng_key: jax.Array,
    resume_root: Path,
    execution_sha256: str,
) -> _IndependentTrainingResult:
    base_checksum = parameter_checksum(base_params, model_config)
    memory = _empty_baseline_memory(model_config, lora_config)
    current_rng_key = rng_key
    adapters: list[IndependentRootAdapter] = []
    rng_by_stage: list[jax.Array] = []
    for task_index, task in enumerate(curriculum.tasks, start=1):
        initialization_key, training_key, current_rng_key = jax.random.split(
            current_rng_key,
            3,
        )
        initial_state = init_candidate_lora_train_state(
            init_lora_edge(initialization_key, model_config, lora_config),
            training_key,
            train_config,
        )
        task_execution_sha256 = _baseline_task_execution_sha256(
            execution_sha256,
            "independent",
            task_index,
            task,
            initial_state,
        )
        final_state, step_losses = _train_resumable_baseline_task(
            resume_root / f"{task_index:02d}-{task.task_id}",
            "independent",
            task_index,
            task,
            task_execution_sha256,
            initial_state,
            base_params,
            model_config,
            memory,
            lora_config,
            train_config,
        )
        adapters.append(
            IndependentRootAdapter(
                task_id=task.task_id,
                adapter=final_state.trainable,
                step_losses=step_losses,
            )
        )
        rng_by_stage.append(current_rng_key)
    if parameter_checksum(base_params, model_config) != base_checksum:
        raise RuntimeError("independent training mutated the frozen base")
    return _IndependentTrainingResult(
        run=IndependentRootLoraRun(
            adapters=tuple(adapters),
            rng_key=current_rng_key,
            train_config=train_config,
            base_parameter_checksum=base_checksum,
        ),
        rng_by_stage=tuple(rng_by_stage),
    )


def _empty_baseline_memory(
    model_config: GptNeoConfig,
    lora_config: LoraConfig,
) -> PackedLoraMemory:
    return pack_lora_memory(
        init_memory_graph(NodeId("root")),
        model_config,
        lora_config,
        max_nodes=2,
        max_edges=1,
    )


def _baseline_task_execution_sha256(
    execution_sha256: str,
    stream: str,
    task_index: int,
    task: LanguageTask,
    initial_state: LmTrainState[LoraEdge],
) -> str:
    digest = sha256()
    digest.update(execution_sha256.encode("ascii"))
    digest.update(stream.encode("ascii"))
    digest.update(str(task_index).encode("ascii"))
    digest.update(str(task.task_id).encode("utf-8"))
    digest.update(lm_train_state_checksum(initial_state).encode("ascii"))
    for batch in task.train_batches:
        for field_name in (
            "input_ids",
            "attention_mask",
            "target_ids",
            "loss_mask",
        ):
            value = np.asarray(getattr(batch, field_name))
            digest.update(field_name.encode("ascii"))
            digest.update(value.dtype.str.encode("ascii"))
            digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
            digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _train_resumable_baseline_task(
    directory: Path,
    stream: str,
    task_index: int,
    task: LanguageTask,
    execution_sha256: str,
    initial_state: LmTrainState[LoraEdge],
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    memory: PackedLoraMemory,
    lora_config: LoraConfig,
    train_config: LmTrainConfig,
) -> tuple[LmTrainState[LoraEdge], tuple[float, ...]]:
    loaded = _load_latest_baseline_chunk(
        directory,
        stream,
        task_index,
        task.task_id,
        execution_sha256,
        initial_state,
        train_config.steps,
    )
    if loaded is None:
        current_state = initial_state
        step_losses: tuple[float, ...] = ()
        _write_baseline_chunk(
            directory / "update-0000000",
            stream,
            task_index,
            task.task_id,
            execution_sha256,
            current_state,
            step_losses,
        )
    else:
        current_state, step_losses = loaded
    parent_coefficients = jnp.zeros(
        (memory.valid_edge_mask.shape[0],),
        dtype=jnp.float32,
    )
    for stop_update in _checkpoint_updates(train_config.steps)[1:]:
        if stop_update <= int(current_state.step):
            continue
        current_state, trace, _ = run_resumable_candidate_edge_updates(
            current_state,
            task.train_batches,
            base_params,
            model_config,
            memory,
            lora_config,
            parent_coefficients,
            0,
            train_config,
            stop_update=stop_update,
        )
        step_losses += trace.step_losses
        _write_baseline_chunk(
            directory / f"update-{stop_update:07d}",
            stream,
            task_index,
            task.task_id,
            execution_sha256,
            current_state,
            step_losses,
        )
    if int(current_state.step) != train_config.steps or (
        len(step_losses) != train_config.steps
    ):
        raise RuntimeError("baseline task did not reach its fixed update budget")
    return current_state, step_losses


def _write_baseline_chunk(
    target: Path,
    stream: str,
    task_index: int,
    task_id: TaskId,
    execution_sha256: str,
    state: LmTrainState[LoraEdge],
    step_losses: tuple[float, ...],
) -> None:
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"baseline resume chunk already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    update = int(state.step)
    core = {
        "execution_sha256": execution_sha256,
        "format": _BASELINE_RESUME_FORMAT,
        "schema_version": _BASELINE_RESUME_SCHEMA_VERSION,
        "step_losses": list(step_losses),
        "stream": stream,
        "task_id": str(task_id),
        "task_index": task_index,
        "update": update,
    }
    state_identity_sha256 = _baseline_state_identity_sha256(core)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent)
    )
    try:
        state_manifest = write_lm_train_state_artifact(
            temporary / _TRANSFER_STATE_DIRECTORY,
            state_identity_sha256,
            (state,),
        )
        metadata = {
            **core,
            "state_identity_sha256": state_identity_sha256,
            "state_payload_sha256": state_manifest.payload_sha256,
        }
        _write_durable_file(
            temporary / "metadata.json",
            _canonical_json_bytes(metadata),
        )
        _fsync_directory(temporary)
        os.rename(temporary, target)
        _fsync_directory(target.parent)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _load_latest_baseline_chunk(
    directory: Path,
    stream: str,
    task_index: int,
    task_id: TaskId,
    execution_sha256: str,
    state_template: LmTrainState[LoraEdge],
    update_budget: int,
) -> tuple[LmTrainState[LoraEdge], tuple[float, ...]] | None:
    if not directory.exists():
        return None
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("baseline resume root must be a nonsymlink directory")
    paths: list[tuple[int, Path]] = []
    for path in directory.iterdir():
        update_match = _BASELINE_UPDATE_PATTERN.fullmatch(path.name)
        if update_match is not None and path.is_dir() and not path.is_symlink():
            paths.append((int(update_match.group(1)), path))
            continue
        if (
            _BASELINE_TEMP_PATTERN.fullmatch(path.name) is not None
            and path.is_dir()
            and not path.is_symlink()
        ):
            continue
        raise ValueError(f"unexpected baseline resume entry: {path.name}")
    if not paths:
        return None
    paths.sort()
    allowed_updates = _checkpoint_updates(update_budget)
    updates = tuple(update for update, _ in paths)
    if updates != allowed_updates[: len(updates)]:
        raise ValueError("baseline resume updates are not a checkpoint prefix")
    loaded: list[tuple[LmTrainState[LoraEdge], tuple[float, ...]]] = []
    for named_update, path in paths:
        loaded.append(
            _load_baseline_chunk(
                path,
                stream,
                task_index,
                task_id,
                execution_sha256,
                state_template,
                named_update,
            )
        )
    return loaded[-1]


def _load_baseline_chunk(
    path: Path,
    stream: str,
    task_index: int,
    task_id: TaskId,
    execution_sha256: str,
    state_template: LmTrainState[LoraEdge],
    named_update: int,
) -> tuple[LmTrainState[LoraEdge], tuple[float, ...]]:
    if {entry.name for entry in path.iterdir()} != {
        "metadata.json",
        _TRANSFER_STATE_DIRECTORY,
    }:
        raise ValueError("baseline resume chunk entries changed")
    metadata = _load_canonical_json_object(
        (path / "metadata.json").read_bytes()
    )
    required = {
        "execution_sha256",
        "format",
        "schema_version",
        "state_identity_sha256",
        "state_payload_sha256",
        "step_losses",
        "stream",
        "task_id",
        "task_index",
        "update",
    }
    if set(metadata) != required:
        raise ValueError("baseline resume metadata fields changed")
    if (
        metadata["execution_sha256"] != execution_sha256
        or metadata["format"] != _BASELINE_RESUME_FORMAT
        or metadata["schema_version"] != _BASELINE_RESUME_SCHEMA_VERSION
        or metadata["stream"] != stream
        or metadata["task_id"] != str(task_id)
        or metadata["task_index"] != task_index
        or metadata["update"] != named_update
    ):
        raise ValueError("baseline resume identity changed")
    losses_value = metadata["step_losses"]
    if type(losses_value) is not list:
        raise ValueError("baseline step losses must be a list")
    step_losses = tuple(float(value) for value in losses_value)
    if len(step_losses) != named_update or any(
        not math.isfinite(value) or value < 0.0 for value in step_losses
    ):
        raise ValueError("baseline step losses do not match the chunk update")
    core = {key: metadata[key] for key in required if not key.startswith("state_")}
    state_identity_sha256 = _baseline_state_identity_sha256(core)
    if metadata["state_identity_sha256"] != state_identity_sha256:
        raise ValueError("baseline state identity changed")
    state = load_lm_train_state_artifact(
        path / _TRANSFER_STATE_DIRECTORY,
        state_identity_sha256,
        (state_template,),
    )[0]
    state_manifest = _load_canonical_json_object(
        (path / _TRANSFER_STATE_DIRECTORY / "manifest.json").read_bytes()
    )
    if metadata["state_payload_sha256"] != state_manifest.get("payload_sha256"):
        raise ValueError("baseline state payload identity changed")
    if int(state.step) != named_update:
        raise ValueError("baseline state update does not match its chunk")
    return state, step_losses


def _baseline_state_identity_sha256(core: dict[str, object]) -> str:
    return sha256(_canonical_json_compact_bytes(core)).hexdigest()


def _save_or_verify_adaptation_artifact(
    target: Path,
    artifact: LanguageAdaptationArtifact,
) -> None:
    if target.is_symlink():
        raise ValueError("pilot adaptation artifact cannot be a symlink")
    if not target.exists():
        save_language_adaptation_artifact(target, artifact)
        return
    existing = load_language_adaptation_artifact(target)
    if (
        existing.tensor_checksums != artifact.tensor_checksums
        or _adaptation_structure(existing) != _adaptation_structure(artifact)
    ):
        raise ValueError("existing pilot adaptation artifact changed")


def _adaptation_structure(
    artifact: LanguageAdaptationArtifact,
) -> dict[str, object]:
    return {
        "base_checkpoint": {
            "directory": str(artifact.base_checkpoint.directory),
            "manifest_sha256": artifact.base_checkpoint.manifest_sha256,
            "parameter_checksum": artifact.base_checkpoint.parameter_checksum,
        },
        "config_hashes": dict(artifact.config_hashes),
        "independent": [
            [record.stage_index, str(record.task_id)]
            for record in artifact.independent_adapters
        ],
        "max_edges": artifact.max_edges,
        "max_nodes": artifact.max_nodes,
        "task_order": [str(task_id) for task_id in artifact.task_order],
        "sequential": [
            [record.stage_index, str(record.task_id)]
            for record in artifact.sequential_stages
        ],
        "vamp_address_node_ids": [
            None if node_id is None else str(node_id)
            for node_id in artifact.address_book.node_ids
        ],
        "vamp_graph": [
            {
                "depth": node.depth,
                "node_id": str(node.node_id),
                "parent_id": (
                    None if node.parent_id is None else str(node.parent_id)
                ),
                "train_stage": node.train_stage,
                "trained_task": (
                    None if node.trained_task is None else str(node.trained_task)
                ),
            }
            for node in artifact.vamp_graph.nodes
        ],
        "vamp_stages": [
            [
                record.stage_index,
                str(record.task_id),
                record.parent_node_index,
                str(record.parent_node_id),
            ]
            for record in artifact.vamp_stages
        ],
    }


def train_fixed_tinyworlds_pilot_adaptations(
    inputs: TinyWorldsPilotInputs,
    temporary_directory: str | Path,
    *,
    training_cache_root: str | Path = PILOT_TRAINING_CACHE_ROOT,
) -> _TrainedPilotAdaptations:
    """Train sequential, independent, and validation-parent VAMP adaptations."""
    if not isinstance(inputs, TinyWorldsPilotInputs):
        raise TypeError("inputs must be TinyWorldsPilotInputs")
    temporary = Path(temporary_directory)
    execution_sha256 = _pilot_training_identity_sha256(inputs)
    workspace = _pilot_training_workspace(
        inputs,
        training_cache_root,
        execution_sha256,
    )
    _write_or_verify_json(
        temporary / "training_cache.json",
        {
            "execution_sha256": execution_sha256,
            "workspace": str(workspace.resolve()),
        },
    )
    checkpoint_directory = workspace / "vamp-resume"
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    checkpoint = inputs.base_artifact.checkpoint
    base_params = checkpoint.params
    model_config = checkpoint.config
    seed = subseed_uint64(
        inputs.master_seed_sha256,
        "pilot-adapter-training",
        inputs.profile_sha256,
    )
    root_key = jax.random.PRNGKey(seed & 0xFFFFFFFF)
    sequential_key, independent_key, vamp_key = jax.random.split(root_key, 3)
    curriculum = inputs.prepared.language.curriculum
    sequential_training = _train_resumable_sequential_state(
        curriculum,
        base_params,
        model_config,
        inputs.lora_config,
        inputs.train_config,
        sequential_key,
        workspace / "baseline-resume" / "sequential",
        execution_sha256,
    )
    independent_training = _train_resumable_independent_state(
        curriculum,
        base_params,
        model_config,
        inputs.lora_config,
        inputs.train_config,
        independent_key,
        workspace / "baseline-resume" / "independent",
        execution_sha256,
    )
    sequential = sequential_training.run
    independent = independent_training.run
    root_address_book = _root_address_book(
        inputs,
        base_params,
        model_config,
    )
    initial = _VampStageState(
        graph=init_memory_graph(NodeId("root")),
        address_book=root_address_book,
        rng_key=vamp_key,
        stage_metrics=(),
        parent_diagnostics=(),
    )
    vamp_stages: list[_VampStageState] = []
    current = initial
    for stage, task in enumerate(curriculum.tasks, start=1):
        current = _train_vamp_stage(
            inputs,
            current,
            task,
            stage,
            base_params,
            model_config,
            checkpoint_directory,
        )
        vamp_stages.append(current)
        artifact = _adaptation_artifact_for_stage(
            inputs,
            sequential,
            independent,
            sequential_training.rng_by_stage,
            independent_training.rng_by_stage,
            current,
            stage,
        )
        _save_or_verify_adaptation_artifact(
            workspace / "adaptation-artifacts" / f"stage-{stage:02d}",
            artifact,
        )
    return _TrainedPilotAdaptations(
        sequential=sequential,
        independent=independent,
        sequential_rng_by_stage=sequential_training.rng_by_stage,
        independent_rng_by_stage=independent_training.rng_by_stage,
        vamp_stages=tuple(vamp_stages),
        training_workspace=workspace,
    )


def _root_address_book(
    inputs: TinyWorldsPilotInputs,
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
) -> AddressBook:
    probes = _stack_router_batches(
        inputs.prepared.language.root_validation_probes
    )
    root_key = derive_node_content_key(
        base_params,
        model_config,
        jnp.asarray(probes.input_ids),
        jnp.asarray(probes.attention_mask),
        expected_probe_count=probes.input_ids.shape[0],
        evaluation_microbatch_size=(
            inputs.execution_preset.evaluation_microbatch_size
        ),
    )
    capacity = len(PILOT_TASK_IDS) + 1
    return add_address_key(
        AddressBook(
            node_ids=(None,) * capacity,
            keys=np.zeros(
                (capacity, model_config.hidden_size),
                dtype=np.float32,
            ),
            valid_node_mask=np.zeros((capacity,), dtype=np.bool_),
        ),
        node_index=0,
        node_id=NodeId("root"),
        key=root_key,
    )


def _train_vamp_stage(
    inputs: TinyWorldsPilotInputs,
    current: _VampStageState,
    task: object,
    stage: int,
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    checkpoint_directory: Path,
) -> _VampStageState:
    from apm.continual.language_tasks import LanguageTask

    if not isinstance(task, LanguageTask):
        raise TypeError("pilot curriculum tasks must be LanguageTask values")
    packed = pack_lora_memory(
        current.graph,
        model_config,
        inputs.lora_config,
        max_nodes=len(PILOT_TASK_IDS) + 1,
        max_edges=len(PILOT_TASK_IDS),
    )
    task_queries = tuple(
        query
        for query in inputs.prepared.validation_queries
        if query.task_id == task.task_id
    )
    suite = KnowledgeValidationSuite(
        suite_id=f"pilot-parent-validation:{task.task_id}",
        split="validation",
        task_id=task.task_id,
        family_id=str(
            inputs.symbolic_bundle.world.task(
                SymbolicTaskId(str(task.task_id))
            ).family_id
        ),
        queries=task_queries,
    )
    validation_hard_scores = _score_hard_candidates_grouped(
        base_params,
        model_config,
        packed,
        inputs.lora_config,
        suite.queries,
        inputs.execution_preset.evaluation_microbatch_size,
    )
    parent_search = select_knowledge_parent_from_scores(
        suite,
        current.graph,
        packed,
        validation_hard_scores,
    )
    symbolic_task = inputs.symbolic_bundle.world.task(
        SymbolicTaskId(str(task.task_id))
    )
    true_parent = NodeId(
        "root"
        if symbolic_task.parent_task_id is None
        else str(symbolic_task.parent_task_id)
    )
    family_by_task = {
        str(specification.task_id): str(specification.family_id)
        for specification in inputs.symbolic_bundle.tasks
    }
    context = KnowledgeParentContext(
        task_id=task.task_id,
        family_id=str(symbolic_task.family_id),
        true_parent_node_id=true_parent,
        node_family_ids=tuple(
            (
                node_id,
                None
                if index == 0
                else family_by_task[str(node_id)],
            )
            for index, node_id in enumerate(memory_node_ids(current.graph))
        ),
    )
    plan = plan_parent_counterfactuals(parent_search, context)
    initialization_key, training_key, next_key = jax.random.split(
        current.rng_key,
        3,
    )
    initial_state = init_candidate_lora_train_state(
        init_lora_edge(initialization_key, model_config, inputs.lora_config),
        training_key,
        inputs.train_config,
    )
    restored = _load_latest_transfer_chunk(
        checkpoint_directory,
        stage,
        plan,
        initial_state,
        inputs.train_config.steps,
    )
    if restored is not None:
        validate_parent_counterfactual_resume(
            plan,
            suite,
            restored,
            task.train_batches,
            base_params,
            model_config,
            packed,
            inputs.lora_config,
            inputs.train_config,
        )
    training: LmTrainState[LoraEdge] | KnowledgeCounterfactualTraining = (
        initial_state if restored is None else restored
    )
    completed_update = (
        0
        if isinstance(training, LmTrainState)
        else training.diagnostics.trials[0].final_update
    )
    for stop_update in tuple(
        update
        for update in _checkpoint_updates(inputs.train_config.steps)[1:]
        if update > completed_update
    ):
        training = run_parent_counterfactuals(
            plan,
            suite,
            training,
            task.train_batches,
            base_params,
            model_config,
            packed,
            inputs.lora_config,
            inputs.train_config,
            stop_update=stop_update,
            evaluation_microbatch_size=(
                inputs.execution_preset.evaluation_microbatch_size
            ),
        )
        if not isinstance(training, KnowledgeCounterfactualTraining):
            raise RuntimeError("counterfactual chunk did not return training state")
        save_tinyworlds_transfer_chunk(
            checkpoint_directory / (
                f"stage-{stage:02d}-chunk-{stop_update:06d}"
            ),
            stage,
            training,
        )
    if not isinstance(training, KnowledgeCounterfactualTraining):
        raise RuntimeError("counterfactual training did not produce a final state")
    graph = commit_selected_counterfactual_edge(
        current.graph,
        training,
        train_stage=stage,
    )
    content_probes = _stack_router_batches(task.content_key_probes)
    content_key = derive_node_content_key(
        base_params,
        model_config,
        jnp.asarray(content_probes.input_ids),
        jnp.asarray(content_probes.attention_mask),
        expected_probe_count=content_probes.input_ids.shape[0],
        evaluation_microbatch_size=(
            inputs.execution_preset.evaluation_microbatch_size
        ),
    )
    address_book = add_address_key(
        current.address_book,
        node_index=stage,
        node_id=NodeId(str(task.task_id)),
        key=content_key,
    )
    selected_trial = training.diagnostics.trial_for_role("selected_parent")
    if selected_trial is None:
        raise RuntimeError("selected-parent counterfactual is unavailable")
    parent_scores = parent_search.mean_correct_candidate_nll + (
        (math.inf,) * (len(PILOT_TASK_IDS) + 1 - len(current.graph.nodes))
    )
    metrics = LanguageStageMetrics(
        stage_index=stage,
        task_id=task.task_id,
        parent_node_index=parent_search.selected_node_index,
        parent_node_id=parent_search.selected_node_id,
        parent_mean_node_nll=parent_scores,
        candidate_step_losses=selected_trial.step_losses,
        task_metrics=(),
    )
    return _VampStageState(
        graph=graph,
        address_book=address_book,
        rng_key=next_key,
        stage_metrics=current.stage_metrics + (metrics,),
        parent_diagnostics=(
            current.parent_diagnostics + (training.diagnostics,)
        ),
    )


def _checkpoint_updates(update_budget: int) -> tuple[int, ...]:
    updates = [0]
    value = 1
    while value < update_budget:
        updates.append(value)
        value *= 2
    updates.append(update_budget)
    return tuple(dict.fromkeys(updates))


def _counterfactual_chunk_record(
    stage: int,
    update: int,
    training: KnowledgeCounterfactualTraining,
) -> dict[str, object]:
    return {
        "execution_sha256": training.execution_sha256,
        "stage": stage,
        "trials": [
            {
                "adapter_sha256": trial.final_adapter_checksum,
                "checkpoints": [
                    {
                        "adapter_sha256": checkpoint.adapter_checksum,
                        "candidate_accuracy": checkpoint.validation_candidate_accuracy,
                        "correct_answer_nll": checkpoint.validation_correct_nll,
                        "training_loss": checkpoint.training_loss,
                        "update": checkpoint.update,
                    }
                    for checkpoint in trial.checkpoints
                ],
                "final_update": trial.final_update,
                "initial_state_sha256": trial.initial_state_checksum,
                "parent_node_index": trial.parent_node_index,
                "parent_node_id": str(trial.parent_node_id),
                "parent_validation_mean_correct_nll": (
                    trial.parent_validation_mean_correct_nll
                ),
                "roles": list(trial.roles),
                "state_sha256": trial.final_state_checksum,
                "step_losses": list(trial.step_losses),
            }
            for trial in training.diagnostics.trials
        ],
        "update": update,
    }


def save_tinyworlds_transfer_chunk(
    directory: str | Path,
    stage: int,
    training: KnowledgeCounterfactualTraining,
) -> Path:
    """Atomically persist complete resumable counterfactual optimizer states."""
    if type(stage) is not int or stage <= 0:
        raise ValueError("transfer chunk stage must be positive")
    if not isinstance(training, KnowledgeCounterfactualTraining):
        raise TypeError("training must be KnowledgeCounterfactualTraining")
    updates = {trial.final_update for trial in training.diagnostics.trials}
    if len(updates) != 1:
        raise ValueError("all transfer trials must end at one chunk update")
    update = next(iter(updates))
    target = Path(directory)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"transfer chunk already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.tmp-{os.getpid()}"
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"transfer chunk temporary path exists: {temporary}")
    temporary.mkdir()
    try:
        diagnostics = _counterfactual_chunk_record(
            stage,
            update,
            training,
        )
        state_identity_sha256 = _transfer_state_identity_sha256(diagnostics)
        state_manifest = write_lm_train_state_artifact(
            temporary / _TRANSFER_STATE_DIRECTORY,
            state_identity_sha256,
            training.final_states,
        )
        core = {
            "diagnostics": diagnostics,
            "format": _TRANSFER_CHUNK_FORMAT,
            "schema_version": _TRANSFER_CHUNK_SCHEMA_VERSION,
            "state_artifact": _TRANSFER_STATE_DIRECTORY,
            "state_artifact_payload_sha256": state_manifest.payload_sha256,
            "state_identity_sha256": state_identity_sha256,
        }
        metadata = {
            **core,
            "metadata_sha256": sha256(_canonical_json_bytes(core)).hexdigest(),
        }
        _write_durable_file(
            temporary / "metadata.json",
            _canonical_json_bytes(metadata),
        )
        _fsync_directory(temporary)
        os.rename(temporary, target)
        _fsync_directory(target.parent)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return target


def load_tinyworlds_transfer_chunk(
    directory: str | Path,
    stage: int,
    plan: ParentCounterfactualPlan,
    state_template: LmTrainState[LoraEdge],
) -> KnowledgeCounterfactualTraining:
    """Strictly reload a transfer chunk against its recomputed parent plan."""
    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("transfer chunk must be a nonsymlink directory")
    represented = tuple(sorted(path.name for path in root.iterdir()))
    if represented != ("metadata.json", _TRANSFER_STATE_DIRECTORY):
        raise ValueError("transfer chunk has missing or unlisted files")
    metadata_payload = (root / "metadata.json").read_bytes()
    metadata = _load_canonical_json_object(metadata_payload)
    required = {
        "diagnostics",
        "format",
        "metadata_sha256",
        "schema_version",
        "state_artifact",
        "state_artifact_payload_sha256",
        "state_identity_sha256",
    }
    if set(metadata) != required:
        raise ValueError("transfer chunk metadata fields are not canonical")
    core = {key: metadata[key] for key in required if key != "metadata_sha256"}
    if (
        metadata["format"] != _TRANSFER_CHUNK_FORMAT
        or metadata["schema_version"] != _TRANSFER_CHUNK_SCHEMA_VERSION
        or metadata["state_artifact"] != _TRANSFER_STATE_DIRECTORY
        or metadata["metadata_sha256"]
        != sha256(_canonical_json_bytes(core)).hexdigest()
    ):
        raise ValueError("transfer chunk metadata identity mismatch")
    diagnostics_record = _require_json_object(
        metadata["diagnostics"],
        "transfer diagnostics",
    )
    if diagnostics_record.get("stage") != stage:
        raise ValueError("transfer chunk stage does not match current execution")
    trials = _decode_transfer_trials(diagnostics_record, plan)
    diagnostics = KnowledgeTransferDiagnostics(plan, trials)
    expected_identity = _transfer_state_identity_sha256(diagnostics_record)
    if metadata["state_identity_sha256"] != expected_identity:
        raise ValueError("transfer state identity does not match diagnostics")
    states = load_lm_train_state_artifact(
        root / _TRANSFER_STATE_DIRECTORY,
        expected_identity,
        (state_template,) * len(trials),
    )
    state_manifest = _load_canonical_json_object(
        (root / _TRANSFER_STATE_DIRECTORY / "manifest.json").read_bytes()
    )
    if (
        metadata["state_artifact_payload_sha256"]
        != state_manifest.get("payload_sha256")
    ):
        raise ValueError("transfer state artifact payload identity changed")
    for state, trial in zip(states, trials):
        if lm_train_state_checksum(state) != trial.final_state_checksum:
            raise ValueError("transfer state checksum does not match diagnostics")
    initial_state_sha256 = lm_train_state_checksum(state_template)
    if any(
        trial.initial_state_checksum != initial_state_sha256
        for trial in trials
    ):
        raise ValueError("transfer initial state does not match current execution")
    execution_sha256 = diagnostics_record.get("execution_sha256")
    if type(execution_sha256) is not str:
        raise ValueError("transfer chunk execution checksum is absent")
    return KnowledgeCounterfactualTraining(
        diagnostics=diagnostics,
        final_states=states,
        execution_sha256=execution_sha256,
    )


def _transfer_state_identity_sha256(
    diagnostics: dict[str, object],
) -> str:
    return sha256(
        _canonical_json_compact_bytes(
            {
                "diagnostics": diagnostics,
                "format": _TRANSFER_CHUNK_FORMAT,
                "schema_version": _TRANSFER_CHUNK_SCHEMA_VERSION,
            }
        )
    ).hexdigest()


def _load_latest_transfer_chunk(
    root: Path,
    stage: int,
    plan: ParentCounterfactualPlan,
    state_template: LmTrainState[LoraEdge],
    update_budget: int,
) -> KnowledgeCounterfactualTraining | None:
    completed_pattern = re.compile(r"stage-(\d{2})-chunk-(\d{6})\Z")
    temporary_pattern = re.compile(
        r"\.stage-(\d{2})-chunk-(\d{6})\.tmp-\d+\Z"
    )
    candidates: list[tuple[int, Path]] = []
    allowed_updates = set(_checkpoint_updates(update_budget))
    for path in root.iterdir():
        match = completed_pattern.fullmatch(path.name)
        if match is not None:
            if path.is_symlink() or not path.is_dir():
                raise ValueError("completed transfer chunks must be directories")
            named_stage = int(match.group(1))
            named_update = int(match.group(2))
            if (
                named_stage not in TINYWORLDS_REPORT_STAGES
                or named_update not in allowed_updates
            ):
                raise ValueError("transfer chunk is outside the fixed schedule")
            if named_stage == stage:
                candidates.append((named_update, path))
            continue
        temporary_match = temporary_pattern.fullmatch(path.name)
        if temporary_match is not None:
            named_stage = int(temporary_match.group(1))
            named_update = int(temporary_match.group(2))
            if (
                named_stage not in TINYWORLDS_REPORT_STAGES
                or named_update not in allowed_updates
                or path.is_symlink()
                or not path.is_dir()
            ):
                raise ValueError(
                    f"unexpected transfer temporary entry: {path.name}"
                )
            continue
        if path.name.startswith(".stage-"):
            raise ValueError(f"unexpected transfer temporary entry: {path.name}")
        raise ValueError(f"unexpected transfer chunk entry: {path.name}")
    if not candidates:
        return None
    candidates.sort()
    completed_updates = tuple(update for update, _ in candidates)
    scheduled_updates = _checkpoint_updates(update_budget)[1:]
    if completed_updates != scheduled_updates[: len(completed_updates)]:
        raise ValueError("transfer resume updates are not a checkpoint prefix")
    loaded_chunks: list[KnowledgeCounterfactualTraining] = []
    for named_update, candidate in candidates:
        loaded = load_tinyworlds_transfer_chunk(
            candidate,
            stage,
            plan,
            state_template,
        )
        actual_update = loaded.diagnostics.trials[0].final_update
        if actual_update != named_update:
            raise ValueError("transfer chunk name does not match its update")
        loaded_chunks.append(loaded)
    return loaded_chunks[-1]


def _decode_transfer_trials(
    diagnostics: dict[str, object],
    plan: ParentCounterfactualPlan,
) -> tuple[ParentTransferTrialDiagnostic, ...]:
    trial_values = diagnostics.get("trials")
    if type(trial_values) is not list:
        raise ValueError("transfer diagnostics trials must be a list")
    trials = tuple(
        _decode_transfer_trial(value) for value in trial_values
    )
    if tuple(trial.parent_node_id for trial in trials) != plan.available_parent_ids:
        raise ValueError("transfer chunk parent plan changed")
    return trials


def _decode_transfer_trial(value: object) -> ParentTransferTrialDiagnostic:
    record = _require_json_object(value, "transfer trial")
    expected = {
        "adapter_sha256",
        "checkpoints",
        "final_update",
        "initial_state_sha256",
        "parent_node_id",
        "parent_node_index",
        "parent_validation_mean_correct_nll",
        "roles",
        "state_sha256",
        "step_losses",
    }
    if set(record) != expected:
        raise ValueError("transfer trial fields are not canonical")
    checkpoints_value = record["checkpoints"]
    roles_value = record["roles"]
    losses_value = record["step_losses"]
    if (
        type(checkpoints_value) is not list
        or type(roles_value) is not list
        or type(losses_value) is not list
    ):
        raise ValueError("transfer trial sequences must be lists")
    return ParentTransferTrialDiagnostic(
        parent_node_index=int(record["parent_node_index"]),
        parent_node_id=NodeId(str(record["parent_node_id"])),
        roles=cast(
            tuple[CounterfactualRole, ...],
            tuple(str(role) for role in roles_value),
        ),
        parent_validation_mean_correct_nll=float(
            record["parent_validation_mean_correct_nll"]
        ),
        initial_state_checksum=str(record["initial_state_sha256"]),
        final_adapter_checksum=str(record["adapter_sha256"]),
        final_state_checksum=str(record["state_sha256"]),
        final_update=int(record["final_update"]),
        step_losses=tuple(float(loss) for loss in losses_value),
        checkpoints=tuple(
            _decode_transfer_checkpoint(checkpoint)
            for checkpoint in checkpoints_value
        ),
    )


def _decode_transfer_checkpoint(value: object) -> TransferCheckpointDiagnostic:
    record = _require_json_object(value, "transfer checkpoint")
    expected = {
        "adapter_sha256",
        "candidate_accuracy",
        "correct_answer_nll",
        "training_loss",
        "update",
    }
    if set(record) != expected:
        raise ValueError("transfer checkpoint fields are not canonical")
    loss = record["training_loss"]
    return TransferCheckpointDiagnostic(
        update=int(record["update"]),
        training_loss=None if loss is None else float(loss),
        validation_candidate_accuracy=float(record["candidate_accuracy"]),
        validation_correct_nll=float(record["correct_answer_nll"]),
        adapter_checksum=str(record["adapter_sha256"]),
    )


def _adaptation_artifact_for_stage(
    inputs: TinyWorldsPilotInputs,
    sequential: SequentialLoraRun,
    independent: IndependentRootLoraRun,
    sequential_rng_by_stage: tuple[jax.Array, ...],
    independent_rng_by_stage: tuple[jax.Array, ...],
    vamp: _VampStageState,
    stage: int,
) -> LanguageAdaptationArtifact:
    task_order = tuple(
        TaskId(str(task_id)) for task_id in PILOT_TASK_IDS[:stage]
    )
    return LanguageAdaptationArtifact(
        base_checkpoint=inputs.base_artifact.checkpoint.reference,
        model_config=inputs.base_artifact.checkpoint.config,
        lora_config=inputs.lora_config,
        train_config=inputs.train_config,
        config_hashes=_adaptation_config_hashes(inputs),
        task_order=task_order,
        sequential_stages=tuple(
            AdapterTrainingRecord(
                item.stage_index,
                item.task_id,
                item.adapter,
                item.step_losses,
            )
            for item in sequential.stages[:stage]
        ),
        independent_adapters=tuple(
            AdapterTrainingRecord(
                index,
                item.task_id,
                item.adapter,
                item.step_losses,
            )
            for index, item in enumerate(independent.adapters[:stage], start=1)
        ),
        vamp_graph=vamp.graph,
        address_book=vamp.address_book,
        rng_state=LanguageAdaptationRngState(
            sequential_rng_by_stage[stage - 1],
            independent_rng_by_stage[stage - 1],
            vamp.rng_key,
        ),
        vamp_stages=tuple(
            VampTrainingRecord(
                item.stage_index,
                item.task_id,
                item.parent_node_index,
                item.parent_node_id,
                item.parent_mean_node_nll,
                item.candidate_step_losses,
            )
            for item in vamp.stage_metrics
        ),
        max_nodes=len(PILOT_TASK_IDS) + 1,
        max_edges=len(PILOT_TASK_IDS),
    )


@dataclass(frozen=True, slots=True)
class _KnowledgeStageEvaluation:
    evaluations: tuple[KnowledgeMethodEvaluation, ...]
    hard_candidate_nll: np.ndarray
    addressing_cost: tuple[TinyWorldsRecord, ...]


@dataclass(frozen=True, slots=True)
class _ExactKgEvidence:
    successes: int
    trials: int

    @property
    def accuracy(self) -> float:
        return self.successes / self.trials


def _evaluate_stage_knowledge(
    inputs: TinyWorldsPilotInputs,
    adaptations: _TrainedPilotAdaptations,
    stage: int,
) -> _KnowledgeStageEvaluation:
    checkpoint = inputs.base_artifact.checkpoint
    base_params = checkpoint.params
    model_config = checkpoint.config
    vamp = adaptations.vamp_stages[stage - 1]
    graph = vamp.graph
    packed = pack_lora_memory(
        graph,
        model_config,
        inputs.lora_config,
        len(PILOT_TASK_IDS) + 1,
        len(PILOT_TASK_IDS),
    )
    queries = inputs.prepared.test_queries
    microbatch_size = inputs.execution_preset.evaluation_microbatch_size
    hard_scores = _score_hard_candidates_grouped(
        base_params,
        model_config,
        packed,
        inputs.lora_config,
        queries,
        microbatch_size,
    )
    future_task_ids = tuple(
        str(task_id) for task_id in PILOT_TASK_IDS[stage:]
    )
    shared_arguments = {
        "queries": queries,
        "hard_candidate_nll": hard_scores,
        "graph": graph,
        "packed_memory": packed,
        "stage": stage,
        "unavailable_node_ids": future_task_ids,
        "unavailable_edge_ids": future_task_ids,
    }
    frozen_scores = _score_frozen_candidates_grouped(
        base_params,
        model_config,
        queries,
        microbatch_size,
    )
    sequential_scores = _score_one_adapter(
        adaptations.sequential.stages[stage - 1].adapter,
        inputs,
        queries,
    )
    independent_scores = _score_independent_adapters(
        adaptations.independent,
        inputs,
        queries,
        stage,
        frozen_scores,
    )
    vamp_oracle_scores = _score_vamp_forward_oracle(
        queries,
        hard_scores,
        graph,
    )
    stored_scores = {
        "frozen_base": frozen_scores,
        "sequential_single_lora": sequential_scores,
        "independent_root_lora": independent_scores,
        "vamp_oracle": vamp_oracle_scores,
    }
    evaluations: dict[str, KnowledgeMethodEvaluation] = {
        method: evaluate_knowledge_method(
            method=method,
            candidate_nll=scores,
            **shared_arguments,
        )
        for method, scores in stored_scores.items()
    }
    costs: list[TinyWorldsRecord] = []
    for router in (
        "vamp_exhaustive",
        "vamp_hopfield",
        "deterministic_random_node",
    ):
        cold_seconds = _time_hard_router_cold_queries(
            router,
            inputs,
            graph,
            packed,
            vamp.address_book,
            queries,
        )
        started = monotonic()
        decision = _route_knowledge_queries(
            router,
            inputs,
            graph,
            packed,
            vamp.address_book,
            queries,
        )
        _block_decision(decision)
        warm_seconds = monotonic() - started
        evaluations[router] = evaluate_knowledge_method(
            method=router,
            hard_decision=decision,
            **shared_arguments,
        )
        costs.append(
            tinyworlds_record(
                stage=stage,
                method=router,
                cold_seconds=cold_seconds,
                warm_seconds=warm_seconds,
            )
        )
    for router in ("vamp_ebt_uniform", "vamp_ebt_hopfield"):
        cold_seconds = _time_ebt_cold_queries(
            router,
            inputs,
            graph,
            packed,
            vamp.address_book,
            queries,
            hard_scores,
            stage,
            future_task_ids,
        )
        started = monotonic()
        hard_evaluation, soft_evaluation = _evaluate_ebt_router_grouped(
            router,
            inputs,
            graph,
            packed,
            vamp.address_book,
            queries,
            hard_scores,
            stage,
            future_task_ids,
        )
        _block_method_evaluation(soft_evaluation)
        warm_seconds = monotonic() - started
        evaluations[router] = hard_evaluation
        evaluations[f"{router}_soft"] = soft_evaluation
        costs.extend(
            tinyworlds_record(
                stage=stage,
                method=method,
                cold_seconds=cold_seconds,
                warm_seconds=warm_seconds,
            )
            for method in (router, f"{router}_soft")
        )
    return _KnowledgeStageEvaluation(
        evaluations=tuple(evaluations[method] for method in TINYWORLDS_REPORT_METHODS),
        hard_candidate_nll=np.asarray(hard_scores, dtype=np.float32),
        addressing_cost=tuple(
            next(record for record in costs if record.require("method") == method)
            for method in TINYWORLDS_ADDRESSING_METHODS
        ),
    )


def _score_one_adapter(
    adapter: LoraEdge,
    inputs: TinyWorldsPilotInputs,
    queries: tuple[KnowledgeQuery, ...],
) -> np.ndarray:
    graph, packed = pack_root_adapter(
        adapter,
        inputs.base_artifact.checkpoint.config,
        inputs.lora_config,
    )
    del graph
    coefficients = np.repeat(
        np.asarray(edge_coefficients_for_node(packed, 1))[None, :],
        len(queries),
        axis=0,
    )
    scores = np.empty((len(queries), 4), dtype=np.float32)
    for indices in _query_indices_by_prefix(queries):
        subset = tuple(queries[index] for index in indices)
        subset_coefficients = coefficients[np.asarray(indices)]
        scores[np.asarray(indices)] = score_edge_coefficient_candidates(
            inputs.base_artifact.checkpoint.params,
            inputs.base_artifact.checkpoint.config,
            packed,
            inputs.lora_config,
            subset,
            subset_coefficients,
            evaluation_microbatch_size=(
                inputs.execution_preset.evaluation_microbatch_size
            ),
        )
    scores.flags.writeable = False
    return scores


def _score_hard_candidates_grouped(
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    packed: PackedLoraMemory,
    lora_config: LoraConfig,
    queries: tuple[KnowledgeQuery, ...],
    microbatch_size: int,
) -> np.ndarray:
    scores = np.full(
        (len(queries), 4, packed.node_path_matrix.shape[0]),
        np.inf,
        dtype=np.float32,
    )
    for indices in _query_indices_by_prefix(queries):
        scores[np.asarray(indices)] = score_hard_node_candidates(
            base_params,
            model_config,
            packed,
            lora_config,
            tuple(queries[index] for index in indices),
            evaluation_microbatch_size=microbatch_size,
        )
    scores.flags.writeable = False
    return scores


def _score_frozen_candidates_grouped(
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    queries: tuple[KnowledgeQuery, ...],
    microbatch_size: int,
) -> np.ndarray:
    scores = np.empty((len(queries), 4), dtype=np.float32)
    for indices in _query_indices_by_prefix(queries):
        scores[np.asarray(indices)] = score_frozen_base_candidates(
            base_params,
            model_config,
            tuple(queries[index] for index in indices),
            evaluation_microbatch_size=microbatch_size,
        )
    scores.flags.writeable = False
    return scores


def _score_independent_adapters(
    independent: IndependentRootLoraRun,
    inputs: TinyWorldsPilotInputs,
    queries: tuple[KnowledgeQuery, ...],
    stage: int,
    frozen_scores: np.ndarray,
) -> np.ndarray:
    scores = np.array(frozen_scores, dtype=np.float32, copy=True)
    for adapter in independent.adapters[:stage]:
        indices = tuple(
            index
            for index, query in enumerate(queries)
            if query.task_id == adapter.task_id
        )
        if indices:
            selected_queries = tuple(queries[index] for index in indices)
            scores[np.asarray(indices)] = _score_one_adapter(
                adapter.adapter,
                inputs,
                selected_queries,
            )
    scores.flags.writeable = False
    return scores


def _score_vamp_forward_oracle(
    queries: tuple[KnowledgeQuery, ...],
    hard_scores: np.ndarray,
    graph: MemoryGraph[LoraEdge],
) -> np.ndarray:
    node_index = {str(node_id): index for index, node_id in enumerate(memory_node_ids(graph))}
    selected = np.asarray(
        tuple(node_index.get(str(query.task_id), 0) for query in queries),
        dtype=np.int32,
    )
    rows = np.arange(len(queries))[:, None]
    candidates = np.arange(4)[None, :]
    scores = np.asarray(hard_scores)[rows, candidates, selected[:, None]]
    result = np.array(scores, dtype=np.float32, copy=True)
    result.flags.writeable = False
    return result


def _route_knowledge_queries(
    router: str,
    inputs: TinyWorldsPilotInputs,
    graph: MemoryGraph[LoraEdge],
    packed: PackedLoraMemory,
    address_book: AddressBook,
    queries: tuple[KnowledgeQuery, ...],
) -> KnowledgeAddressDecision:
    del graph
    decisions = tuple(
        (
            indices,
            route_language_prefix(
                router,
                inputs.base_artifact.checkpoint.params,
                inputs.base_artifact.checkpoint.config,
                packed,
                inputs.lora_config,
                address_book,
                _stack_router_batches(
                    tuple(queries[index].router_batch for index in indices)
                ),
                random_seed=inputs.execution_preset.random_router_seed,
                evaluation_microbatch_size=(
                    inputs.execution_preset.evaluation_microbatch_size
                ),
            ),
        )
        for indices in _query_indices_by_prefix(queries)
    )
    return _merge_address_decisions(decisions, len(queries))


def _query_indices_by_prefix(
    queries: tuple[KnowledgeQuery, ...],
) -> tuple[tuple[int, ...], ...]:
    execution_shapes = tuple(
        dict.fromkeys(
            (
                query.router_batch.input_ids.shape[1],
                query.candidates[0].competence_batch.input_ids.shape[1],
            )
            for query in queries
        )
    )
    return tuple(
        tuple(
            index
            for index, query in enumerate(queries)
            if (
                query.router_batch.input_ids.shape[1],
                query.candidates[0].competence_batch.input_ids.shape[1],
            )
            == execution_shape
        )
        for execution_shape in execution_shapes
    )


def _merge_address_decisions(
    indexed_decisions: tuple[tuple[tuple[int, ...], object], ...],
    query_count: int,
) -> KnowledgeAddressDecision:
    if not indexed_decisions:
        raise ValueError("address decisions cannot be empty")
    first = indexed_decisions[0][1]
    node_capacity = np.asarray(first.node_scores).shape[1]
    top_k_size = np.asarray(first.top_k_indices).shape[1]
    integer_fields = {
        "selected_indices": np.empty((query_count,), dtype=np.int32),
        "top_k_indices": np.empty(
            (query_count, top_k_size),
            dtype=np.int32,
        ),
    }
    float_fields = {
        "node_probabilities": np.empty(
            (query_count, node_capacity),
            dtype=np.float32,
        ),
        "node_scores": np.empty(
            (query_count, node_capacity),
            dtype=np.float32,
        ),
        "score_margin": np.empty((query_count,), dtype=np.float32),
        "entropy": np.empty((query_count,), dtype=np.float32),
    }
    represented: list[int] = []
    for indices, decision in indexed_decisions:
        rows = np.asarray(indices, dtype=np.int32)
        represented.extend(indices)
        for field_name, values in integer_fields.items():
            values[rows] = np.asarray(getattr(decision, field_name))
        for field_name, values in float_fields.items():
            values[rows] = np.asarray(getattr(decision, field_name))
    if tuple(sorted(represented)) != tuple(range(query_count)):
        raise ValueError("grouped decisions must cover every query exactly once")
    return KnowledgeAddressDecision(
        selected_indices=integer_fields["selected_indices"],
        node_probabilities=float_fields["node_probabilities"],
        node_scores=float_fields["node_scores"],
        score_margin=float_fields["score_margin"],
        entropy=float_fields["entropy"],
        top_k_indices=integer_fields["top_k_indices"],
    )


def _evaluate_ebt_router_grouped(
    router: str,
    inputs: TinyWorldsPilotInputs,
    graph: MemoryGraph[LoraEdge],
    packed: PackedLoraMemory,
    address_book: AddressBook,
    queries: tuple[KnowledgeQuery, ...],
    hard_scores: np.ndarray,
    stage: int,
    future_task_ids: tuple[str, ...],
) -> tuple[KnowledgeMethodEvaluation, KnowledgeMethodEvaluation]:
    grouped = tuple(
        (
            indices,
            evaluate_ebt_knowledge_methods(
                inputs.base_artifact.checkpoint.params,
                inputs.base_artifact.checkpoint.config,
                graph,
                packed,
                inputs.lora_config,
                address_book,
                tuple(queries[index] for index in indices),
                np.asarray(hard_scores)[np.asarray(indices)],
                stage=stage,
                evaluation_microbatch_size=(
                    inputs.execution_preset.evaluation_microbatch_size
                ),
                unavailable_node_ids=future_task_ids,
                unavailable_edge_ids=future_task_ids,
                routers=(router,),
            ),
        )
        for indices in _query_indices_by_prefix(queries)
    )
    merged = tuple(
        _merge_method_evaluations(
            tuple(
                (indices, evaluations[method_index])
                for indices, evaluations in grouped
            ),
            queries,
        )
        for method_index in range(2)
    )
    if len(merged) != 2:
        raise RuntimeError("one EBT router must produce hard and soft results")
    return merged[0], merged[1]


def _merge_method_evaluations(
    indexed: tuple[tuple[tuple[int, ...], KnowledgeMethodEvaluation], ...],
    queries: tuple[KnowledgeQuery, ...],
) -> KnowledgeMethodEvaluation:
    first = indexed[0][1]
    query_rows = {
        row.query_id: row
        for _, evaluation in indexed
        for row in evaluation.queries
    }
    ordered_rows = tuple(query_rows[query.query_id] for query in queries)
    address = (
        None
        if first.address_decision is None
        else _merge_address_decisions(
            tuple(
                (indices, evaluation.address_decision)
                for indices, evaluation in indexed
                if evaluation.address_decision is not None
            ),
            len(queries),
        )
    )
    coefficients = None
    if first.edge_coefficients is not None:
        edge_capacity = first.edge_coefficients.shape[1]
        merged = np.zeros(
            (len(queries), edge_capacity),
            dtype=np.float32,
        )
        for indices, evaluation in indexed:
            if evaluation.edge_coefficients is None:
                raise ValueError("grouped soft evaluations lost edge coefficients")
            merged[np.asarray(indices)] = evaluation.edge_coefficients
        coefficients = merged
    return KnowledgeMethodEvaluation(
        stage=first.stage,
        method=first.method,
        queries=ordered_rows,
        aggregates=aggregate_knowledge_evaluations(ordered_rows),
        address_decision=address,
        edge_coefficients=coefficients,
    )


def _time_hard_router_cold_queries(
    router: str,
    inputs: TinyWorldsPilotInputs,
    graph: MemoryGraph[LoraEdge],
    packed: PackedLoraMemory,
    address_book: AddressBook,
    queries: tuple[KnowledgeQuery, ...],
) -> float:
    started = monotonic()
    decision = _route_knowledge_queries(
        router,
        inputs,
        graph,
        packed,
        address_book,
        queries,
    )
    _block_decision(decision)
    return monotonic() - started


def _time_ebt_cold_queries(
    router: str,
    inputs: TinyWorldsPilotInputs,
    graph: MemoryGraph[LoraEdge],
    packed: PackedLoraMemory,
    address_book: AddressBook,
    queries: tuple[KnowledgeQuery, ...],
    hard_scores: np.ndarray,
    stage: int,
    future_task_ids: tuple[str, ...],
) -> float:
    started = monotonic()
    evaluations = _evaluate_ebt_router_grouped(
        router,
        inputs,
        graph,
        packed,
        address_book,
        queries,
        hard_scores,
        stage,
        future_task_ids,
    )
    _block_method_evaluation(evaluations[-1])
    return monotonic() - started


def _block_decision(decision: KnowledgeAddressDecision) -> None:
    jax.block_until_ready(jnp.asarray(decision.node_probabilities))


def _block_method_evaluation(evaluation: KnowledgeMethodEvaluation) -> None:
    if evaluation.edge_coefficients is not None:
        jax.block_until_ready(jnp.asarray(evaluation.edge_coefficients))
    else:
        jax.block_until_ready(jnp.asarray(evaluation.queries[0].candidate_nll))


def _natural_continuation_records(
    inputs: TinyWorldsPilotInputs,
    adaptations: _TrainedPilotAdaptations,
    stage: int,
) -> tuple[TinyWorldsRecord, ...]:
    checkpoint = inputs.base_artifact.checkpoint
    vamp = adaptations.vamp_stages[stage - 1]
    vamp_memory = pack_lora_memory(
        vamp.graph,
        checkpoint.config,
        inputs.lora_config,
        len(PILOT_TASK_IDS) + 1,
        len(PILOT_TASK_IDS),
    )
    frozen_memory = pack_lora_memory(
        init_memory_graph(NodeId("root")),
        checkpoint.config,
        inputs.lora_config,
        1,
        0,
    )
    _, sequential_memory = pack_root_adapter(
        adaptations.sequential.stages[stage - 1].adapter,
        checkpoint.config,
        inputs.lora_config,
    )
    independent_by_task = {
        str(adapter.task_id): adapter.adapter
        for adapter in adaptations.independent.adapters[:stage]
    }
    vamp_node_index = {
        str(node_id): index
        for index, node_id in enumerate(memory_node_ids(vamp.graph))
    }
    records: list[TinyWorldsRecord] = []
    for task in inputs.prepared.language.curriculum.tasks:
        prefix_lengths = tuple(
            dict.fromkeys(
                example.router_batch.input_ids.shape[1] + 1
                for example in task.test_examples
            )
        )
        for prefix_length in prefix_lengths:
            examples = tuple(
                example
                for example in task.test_examples
                if example.router_batch.input_ids.shape[1] + 1 == prefix_length
            )
            competence = _stack_competence_batches(
                tuple(example.competence_batch for example in examples)
            )
            router_batch = _stack_router_batches(
                tuple(example.router_batch for example in examples)
            )
            weights = np.sum(competence.loss_mask, axis=1).astype(np.float64)
            frozen_rows = competence_nll_by_node(
                checkpoint.params,
                checkpoint.config,
                frozen_memory,
                inputs.lora_config,
                competence,
                evaluation_microbatch_size=(
                    inputs.execution_preset.evaluation_microbatch_size
                ),
            )[:, 0]
            sequential_rows = competence_nll_by_node(
                checkpoint.params,
                checkpoint.config,
                sequential_memory,
                inputs.lora_config,
                competence,
                evaluation_microbatch_size=(
                    inputs.execution_preset.evaluation_microbatch_size
                ),
            )[:, 1]
            independent_adapter = independent_by_task.get(str(task.task_id))
            if independent_adapter is None:
                independent_rows = frozen_rows
            else:
                _, independent_memory = pack_root_adapter(
                    independent_adapter,
                    checkpoint.config,
                    inputs.lora_config,
                )
                independent_rows = competence_nll_by_node(
                    checkpoint.params,
                    checkpoint.config,
                    independent_memory,
                    inputs.lora_config,
                    competence,
                    evaluation_microbatch_size=(
                        inputs.execution_preset.evaluation_microbatch_size
                    ),
                )[:, 1]
            vamp_rows = competence_nll_by_node(
                checkpoint.params,
                checkpoint.config,
                vamp_memory,
                inputs.lora_config,
                competence,
                evaluation_microbatch_size=(
                    inputs.execution_preset.evaluation_microbatch_size
                ),
            )
            oracle_index = vamp_node_index.get(str(task.task_id), 0)
            values = {
                "frozen_base": _weighted_nll(frozen_rows, weights),
                "sequential_single_lora": _weighted_nll(
                    sequential_rows,
                    weights,
                ),
                "independent_root_lora": _weighted_nll(
                    independent_rows,
                    weights,
                ),
                "vamp_oracle": _weighted_nll(
                    vamp_rows[:, oracle_index],
                    weights,
                ),
            }
            for router in ROUTER_BASELINE_NAMES:
                decision = route_language_prefix(
                    router,
                    checkpoint.params,
                    checkpoint.config,
                    vamp_memory,
                    inputs.lora_config,
                    vamp.address_book,
                    router_batch,
                    random_seed=inputs.execution_preset.random_router_seed,
                    evaluation_microbatch_size=(
                        inputs.execution_preset.evaluation_microbatch_size
                    ),
                )
                values[router] = _weighted_nll(
                    vamp_rows[
                        np.arange(len(examples)),
                        np.asarray(decision.selected_indices, dtype=np.int32),
                    ],
                    weights,
                )
            records.extend(
                tinyworlds_record(
                    stage=stage,
                    method=method,
                    task_id=str(task.task_id),
                    prefix_length=prefix_length,
                    suffix_nll=values[method],
                    example_count=len(examples),
                    suffix_token_count=int(np.sum(weights)),
                )
                for method in TINYWORLDS_NATURAL_CONTINUATION_METHODS
            )
    return tuple(records)


def _weighted_nll(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(np.asarray(values) * weights) / np.sum(weights))


def _parent_search_records(
    diagnostics: KnowledgeTransferDiagnostics,
    stage: int,
) -> tuple[TinyWorldsRecord, ...]:
    search = diagnostics.plan.parent_search
    ranked_indices = tuple(
        sorted(
            range(len(search.node_ids)),
            key=lambda index: (
                search.mean_correct_candidate_nll[index],
                index,
            ),
        )
    )
    rank_by_index = {
        node_index: rank for rank, node_index in enumerate(ranked_indices)
    }
    return tuple(
        tinyworlds_record(
            stage=stage,
            task_id=str(search.task_id),
            candidate_parent_id=str(node_id),
            rank=rank_by_index[index],
            mean_candidate_nll=search.mean_correct_candidate_nll[index],
            selected=index == search.selected_node_index,
            scoring_basis=search.scoring_basis,
            validation_suite_id=search.validation_suite_id,
            validation_suite_sha256=search.validation_suite_sha256,
        )
        for index, node_id in enumerate(search.node_ids)
    )


def _counterfactual_transfer_records(
    diagnostics: KnowledgeTransferDiagnostics,
    stage: int,
) -> tuple[TinyWorldsRecord, ...]:
    return tuple(
        record
        for role in TINYWORLDS_PARENT_COUNTERFACTUALS
        for record in _counterfactual_transfer_role_records(
            diagnostics,
            role,
            stage,
        )
    )


def _counterfactual_transfer_role_records(
    diagnostics: KnowledgeTransferDiagnostics,
    role: str,
    stage: int,
) -> tuple[TinyWorldsRecord, ...]:
    target = next(
        item for item in diagnostics.plan.targets if item.role == role
    )
    if not target.available:
        return (
            tinyworlds_record(
                stage=stage,
                task_id=str(diagnostics.plan.context.task_id),
                parent_kind=role,
                parent_node_id=None,
                available=False,
                update=None,
                training_loss=None,
                candidate_accuracy=None,
                correct_answer_nll=None,
                adapter_sha256=None,
                final_update=None,
            ),
        )
    trial = diagnostics.trial_for_role(target.role)
    if trial is None:
        raise RuntimeError("available parent counterfactual omitted its trial")
    return tuple(
        tinyworlds_record(
            stage=stage,
            task_id=str(diagnostics.plan.context.task_id),
            parent_kind=role,
            parent_node_id=str(trial.parent_node_id),
            available=True,
            update=checkpoint.update,
            training_loss=checkpoint.training_loss,
            candidate_accuracy=checkpoint.validation_candidate_accuracy,
            correct_answer_nll=checkpoint.validation_correct_nll,
            adapter_sha256=checkpoint.adapter_checksum,
            final_update=trial.final_update,
            initial_state_sha256=trial.initial_state_checksum,
            final_adapter_sha256=trial.final_adapter_checksum,
            final_state_sha256=trial.final_state_checksum,
        )
        for checkpoint in trial.checkpoints
    )


def _committed_node_drift_records(
    inputs: TinyWorldsPilotInputs,
    adaptations: _TrainedPilotAdaptations,
    stage: int,
    current: _KnowledgeStageEvaluation,
    previous: _KnowledgeStageEvaluation | None,
) -> tuple[TinyWorldsRecord, ...]:
    graph = adaptations.vamp_stages[stage - 1].graph
    if previous is None:
        return (
            tinyworlds_record(
                stage=stage,
                node_id="root",
                logit_max_abs_drift=0.0,
                answer_change_count=0,
                checksum_match=True,
            ),
        )
    previous_node_count = len(adaptations.vamp_stages[stage - 2].graph.nodes)
    old_scores = np.asarray(previous.hard_candidate_nll)[
        :, :, :previous_node_count
    ]
    new_scores = np.asarray(current.hard_candidate_nll)[
        :, :, :previous_node_count
    ]
    if not np.array_equal(old_scores, new_scores):
        difference = float(np.max(np.abs(old_scores - new_scores)))
        raise RuntimeError(
            f"committed-node candidate logits drifted at stage {stage}: {difference}"
        )
    old_answers = np.argmin(old_scores, axis=1)
    new_answers = np.argmin(new_scores, axis=1)
    if not np.array_equal(old_answers, new_answers):
        raise RuntimeError("committed-node candidate answers changed")
    previous_graph = adaptations.vamp_stages[stage - 2].graph
    records = tuple(
        tinyworlds_record(
            stage=stage,
            node_id=str(node.node_id),
            logit_max_abs_drift=0.0,
            answer_change_count=0,
            checksum_match=(
                node.incoming_edge is None
                or _lora_checksum(node.incoming_edge, inputs)
                == _lora_checksum(
                    graph.nodes[node_index].incoming_edge,
                    inputs,
                )
            ),
        )
        for node_index, node in enumerate(previous_graph.nodes)
    )
    if any(record.require("checksum_match") is not True for record in records):
        raise RuntimeError("committed-node adapter checksum changed")
    return records


def _memory_record(
    inputs: TinyWorldsPilotInputs,
    vamp: _VampStageState,
    stage: int,
) -> TinyWorldsRecord:
    packed = pack_lora_memory(
        vamp.graph,
        inputs.base_artifact.checkpoint.config,
        inputs.lora_config,
        len(PILOT_TASK_IDS) + 1,
        len(PILOT_TASK_IDS),
    )
    accounting = account_language_memory(
        inputs.base_artifact.checkpoint.params,
        vamp.graph,
        vamp.address_book,
        packed,
        inputs.lora_config,
    )
    peak = measure_peak_device_memory(
        inputs.execution_preset.allocator_peak_limit_bytes
    )
    if peak.peak_bytes_in_use is None:
        raise RuntimeError("pilot accelerator did not expose allocator peak bytes")
    return tinyworlds_record(
        stage=stage,
        persistent_bytes=accounting.persistent_bytes,
        runtime_bytes=accounting.packed_runtime_bytes,
        allocator_peak_bytes=peak.peak_bytes_in_use,
        optimizer_peak_bytes=accounting.optimizer_peak_bytes,
        committed_lora_bytes=accounting.committed_lora_bytes,
        address_key_bytes=accounting.address_key_bytes,
    )


def _lora_checksum(
    edge: LoraEdge | None,
    inputs: TinyWorldsPilotInputs,
) -> str:
    if edge is None:
        return sha256(b"root").hexdigest()
    digest = sha256()
    for name, value in sorted(
        flatten_lora_edge(
            edge,
            inputs.base_artifact.checkpoint.config,
            inputs.lora_config,
        ).items()
    ):
        array = np.asarray(value, dtype=np.float32)
        digest.update(name.encode("utf-8"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _exact_kg_test_evidence(
    inputs: TinyWorldsPilotInputs,
) -> _ExactKgEvidence:
    bundle = inputs.symbolic_bundle
    plan_by_id = {
        str(plan.query_ast.query_id): plan for plan in bundle.query_plans
    }
    successes = 0
    trials = 0
    for group in inputs.rendered.query_groups:
        if group.split is not DataSplit.TEST:
            continue
        plan = plan_by_id.get(group.symbolic_query_id)
        if plan is None or plan.split is not DataSplit.TEST:
            raise RuntimeError("test rendering lost its symbolic query plan")
        answers = answer_query(
            bundle.world.closure,
            plan.query_ast,
            bundle.world.registry,
            bundle.entities,
        )
        canonical_proof = bundle.world.closure.proof_for(
            plan.proof.conclusion_atom_id
        )
        proof_matches = (
            canonical_proof.proof_id == plan.proof.proof_id
            and canonical_proof.conclusion.atom_id
            == plan.proof.conclusion_atom_id
            and canonical_proof.supporting_fact_ids
            == plan.proof.supporting_fact_ids
            and canonical_proof.supporting_rule_ids
            == plan.proof.supporting_rule_ids
            and canonical_proof.depth == plan.proof.depth
        )
        expected_support_ids = tuple(
            str(item)
            for item in (
                *plan.proof.supporting_fact_ids,
                *plan.proof.supporting_rule_ids,
            )
        )
        for variant in group.variants:
            query = variant.knowledge_query
            candidate_ids = variant.candidate_entity_ids
            local_correct_index = query.correct_candidate_index
            correct = (
                answers == (plan.answer_entity_id,)
                and proof_matches
                and len(candidate_ids) == 4
                and len(set(candidate_ids)) == 4
                and candidate_ids.count(str(plan.answer_entity_id)) == 1
                and 0 <= local_correct_index < len(candidate_ids)
                and candidate_ids[local_correct_index]
                == str(plan.answer_entity_id)
                and query.proof_id == str(plan.proof.proof_id)
                and query.support_ids == expected_support_ids
            )
            successes += int(correct)
            trials += 1
    if trials == 0:
        raise RuntimeError("exact KG integrity has no held-out test variants")
    return _ExactKgEvidence(successes, trials)


def _no_test_selection_passed(
    inputs: TinyWorldsPilotInputs,
    adaptations: _TrainedPilotAdaptations,
) -> bool:
    validation_ids = {
        query.query_id for query in inputs.prepared.validation_queries
    }
    test_ids = {query.query_id for query in inputs.prepared.test_queries}
    selected_ids = {
        query_id
        for diagnostics in adaptations.vamp_stages[-1].parent_diagnostics
        for query_id in diagnostics.plan.parent_search.validation_query_ids
    }
    audit = _selection_audit_records(inputs)
    return (
        validation_ids.isdisjoint(test_ids)
        and selected_ids == validation_ids
        and selected_ids.isdisjoint(test_ids)
        and all(
            record.require("used_for_tuning") is False
            and (
                record.require("used_for_parent_selection")
                == (record.require("split") == "validation")
            )
            for record in audit
        )
    )


def _final_evidence(
    inputs: TinyWorldsPilotInputs,
    adaptations: _TrainedPilotAdaptations,
    stages: tuple[TinyWorldsPilotStageResult, ...],
    final_knowledge: _KnowledgeStageEvaluation,
    exact_kg: _ExactKgEvidence,
) -> TinyWorldsPilotFinalEvidence:
    final_graph = adaptations.vamp_stages[-1].graph
    node_by_id = {str(node.node_id): node for node in final_graph.nodes}
    graph_recovery = tuple(
        tinyworlds_record(
            task_id=str(task.task_id),
            expected_parent_id=(
                "root"
                if task.parent_task_id is None
                else str(task.parent_task_id)
            ),
            learned_parent_id=str(node_by_id[str(task.task_id)].parent_id),
            recovered=(
                str(node_by_id[str(task.task_id)].parent_id)
                == (
                    "root"
                    if task.parent_task_id is None
                    else str(task.parent_task_id)
                )
            ),
        )
        for task in inputs.symbolic_bundle.tasks
    )
    revision = _revision_retention_records(
        inputs,
        final_graph,
        final_knowledge.hard_candidate_nll,
    )
    implementation_gates = (
        tinyworlds_record(
            gate="exact_kg_integrity",
            category="implementation",
            passed=exact_kg.successes == exact_kg.trials,
            successes=exact_kg.successes,
            trials=exact_kg.trials,
            accuracy=exact_kg.accuracy,
        ),
        tinyworlds_record(
            gate="committed_node_drift",
            category="implementation",
            passed=all(
                row.require("checksum_match") is True
                and row.require("answer_change_count") == 0
                and float(row.require("logit_max_abs_drift")) == 0.0
                for stage in stages
                for row in stage.committed_node_drift
            ),
        ),
        tinyworlds_record(
            gate="no_test_selection",
            category="implementation",
            passed=_no_test_selection_passed(inputs, adaptations),
            validation_query_count=len(inputs.prepared.validation_queries),
            test_query_count=len(inputs.prepared.test_queries),
        ),
        tinyworlds_record(
            gate="allocator_peak_12gib",
            category="implementation",
            passed=max(
                int(row.require("allocator_peak_bytes"))
                for stage in stages
                for row in stage.memory_metrics
            )
            <= TINYWORLDS_ALLOCATOR_PEAK_LIMIT_BYTES,
        ),
    )
    final_accuracy = {
        evaluation.method: next(
            row.candidate_accuracy
            for row in evaluation.aggregates
            if row.grouping_axis == "all"
        )
        for evaluation in final_knowledge.evaluations
    }
    scientific_gates = tuple(
        tinyworlds_record(
            gate=f"{method}_above_four_way_chance",
            category="scientific",
            passed=accuracy > 0.25,
            observed_accuracy=accuracy,
        )
        for method, accuracy in final_accuracy.items()
    )
    return TinyWorldsPilotFinalEvidence(
        graph_recovery=graph_recovery,
        revision_retention=revision,
        gate_results=implementation_gates + scientific_gates,
        representative_queries=_representative_query_records(inputs),
        selection_audit=_selection_audit_records(inputs),
    )


def _revision_retention_records(
    inputs: TinyWorldsPilotInputs,
    graph: MemoryGraph[LoraEdge],
    hard_scores: np.ndarray,
) -> tuple[TinyWorldsRecord, ...]:
    query_index = {
        query.query_id: index
        for index, query in enumerate(inputs.prepared.test_queries)
    }
    plan_by_id = {
        str(plan.query_ast.query_id): plan
        for plan in inputs.symbolic_bundle.query_plans
    }
    node_index = {
        str(node_id): index for index, node_id in enumerate(memory_node_ids(graph))
    }
    records: list[TinyWorldsRecord] = []
    for family_id in ("willow", "sunny"):
        family_tasks = tuple(
            task
            for task in inputs.symbolic_bundle.tasks
            if str(task.family_id) == family_id
        )
        seed_task = next(
            task for task in family_tasks if task.kind is TaskKind.SEED
        )
        revision_task = next(
            task for task in family_tasks if task.kind is TaskKind.REVISION
        )
        observations: list[tuple[bool, bool]] = []
        for group in inputs.rendered.query_groups:
            if (
                group.split is not DataSplit.TEST
                or group.task_id != str(revision_task.task_id)
            ):
                continue
            plan = plan_by_id[group.symbolic_query_id]
            if plan.kind is not QueryKind.REVISION_SENSITIVE:
                continue
            old_entity_ids = tuple(
                str(candidate.entity_id)
                for candidate in plan.candidates
                if candidate.role is CandidateRole.INCOMPATIBLE_REVISION
            )
            if len(old_entity_ids) != 1:
                continue
            for variant in group.variants:
                try:
                    old_candidate_index = variant.candidate_entity_ids.index(
                        old_entity_ids[0]
                    )
                except ValueError as error:
                    raise RuntimeError(
                        "revision variant lost its incompatible candidate"
                    ) from error
                row = query_index[variant.knowledge_query.query_id]
                old_prediction = int(
                    np.argmin(
                        hard_scores[
                            row,
                            :,
                            node_index[str(seed_task.task_id)],
                        ]
                    )
                )
                revision_prediction = int(
                    np.argmin(
                        hard_scores[
                            row,
                            :,
                            node_index[str(revision_task.task_id)],
                        ]
                    )
                )
                observations.append(
                    (
                        old_prediction == old_candidate_index,
                        revision_prediction
                        == variant.knowledge_query.correct_candidate_index,
                    )
                )
        if not observations:
            raise RuntimeError(
                f"revision retention has no paired observations for {family_id}"
            )
        records.append(
            tinyworlds_record(
                stage=8,
                family_id=family_id,
                old_context_accuracy=float(
                    np.mean(tuple(old for old, _ in observations))
                ),
                revision_context_accuracy=float(
                    np.mean(tuple(new for _, new in observations))
                ),
                paired_revision_consistency=float(
                    np.mean(tuple(old and new for old, new in observations))
                ),
                pair_count=len(observations),
            )
        )
    return tuple(records)


def _representative_query_records(
    inputs: TinyWorldsPilotInputs,
) -> tuple[TinyWorldsRecord, ...]:
    represented: set[tuple[str, str]] = set()
    records: list[TinyWorldsRecord] = []
    for group in inputs.rendered.query_groups:
        if group.split is not DataSplit.TEST:
            continue
        variant = group.variants[0]
        query = variant.knowledge_query
        identity = (query.family_id, query.query_kind)
        if identity in represented:
            continue
        represented.add(identity)
        records.append(
            tinyworlds_record(
                query_id=query.query_id,
                proof_id=query.proof_id,
                query_text=variant.prefix_text,
                answer_text=query.candidates[query.correct_candidate_index].answer_text,
                support_ids=",".join(query.support_ids),
                family_id=query.family_id,
                query_kind=query.query_kind,
            )
        )
    return tuple(records)


def _selection_audit_records(
    inputs: TinyWorldsPilotInputs,
) -> tuple[TinyWorldsRecord, ...]:
    return tuple(
        tinyworlds_record(
            record_id=query.query_id,
            split=split,
            used_for_tuning=False,
            used_for_parent_selection=split == "validation",
        )
        for split, queries in (
            ("validation", inputs.prepared.validation_queries),
            ("test", inputs.prepared.test_queries),
        )
        for query in queries
    )


def build_tinyworlds_pilot_manifest(
    inputs: TinyWorldsPilotInputs,
    temporary_directory: str | Path,
) -> TinyWorldsReportManifest:
    """Build the complete content-addressed identity for the executed pilot."""
    symbolic_manifest = load_tinyworlds_manifest(
        Path(temporary_directory) / "symbolic-pilot"
    )
    checkpoint = inputs.base_artifact.checkpoint
    tokenizer_path = (
        inputs.base_artifact.directory / "tokenizer" / "tokenizer.json"
    )
    config = {
        "generator_version": TINYWORLDS_VERSION,
        "renderer_version": inputs.rendered.registry.version,
        "seeds": {
            "calibration_profile_sha256": inputs.profile_sha256,
            "master_seed_sha256": inputs.master_seed_sha256,
            "pilot_namespace_sha256": derive_subseed(
                inputs.master_seed_sha256,
                "world",
                "pilot",
            ),
            "public": PUBLIC_SEED,
        },
        "checkpoint_identity": {
            "manifest_sha256": checkpoint.reference.manifest_sha256,
            "parameter_checksum": checkpoint.reference.parameter_checksum,
            "source": _json_ready(asdict(checkpoint.source)),
        },
        "tokenizer_identity": {
            "metadata": _json_ready(asdict(checkpoint.tokenizer)),
            "tokenizer_json_sha256": sha256(tokenizer_path.read_bytes()).hexdigest(),
        },
        "corpus_identity": {
            "filename": "TinyStories-train.txt",
            "revision": "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64",
            "sha256": "c5cf5e22ff13614e830afbe61a99fbcbe8bcb7dd72252b989fa1117a368d401f",
            "size_bytes": 1_924_281_556,
        },
        "topology": [
            {
                "family_id": str(task.family_id),
                "kind": task.kind.value,
                "parent_task_id": (
                    None
                    if task.parent_task_id is None
                    else str(task.parent_task_id)
                ),
                "task_id": str(task.task_id),
            }
            for task in inputs.symbolic_bundle.tasks
        ],
        "ontology": {
            "entity_types": [
                str(entity_type)
                for entity_type in inputs.symbolic_bundle.world.registry.entity_types
            ],
            "predicate_ids": [
                str(signature.predicate_id)
                for signature in inputs.symbolic_bundle.world.registry.predicates
            ],
            "symbolic_bundle_sha256": symbolic_manifest.bundle_sha256,
        },
        "calibration_profile": {
            "selected_config": _json_ready(
                asdict(inputs.profile.selected_config)
            ),
            "sha256": inputs.profile_sha256,
        },
        "story_policy": _json_ready(asdict(inputs.rendered.preset)),
        "query_policy": {
            "future_oracle_policy": PILOT_FORWARD_TRANSFER_POLICY,
            "prefix_lengths": [64, 128, 192],
            "semantic_boundary_api": True,
            "validation_query_count": len(inputs.prepared.validation_queries),
            "test_query_count": len(inputs.prepared.test_queries),
        },
        "candidate_policy": {
            "candidate_count": 4,
            "distractor_policy": (
                inputs.profile.selected_config.distractor_policy.value
            ),
            "equal_token_lengths": True,
        },
        "adapter_settings": _json_ready(asdict(inputs.lora_config)),
        "optimizer_settings": _json_ready(asdict(inputs.train_config)),
        "routers": list(TINYWORLDS_REPORT_METHODS),
        "microbatching": {
            "candidate": inputs.execution_preset.evaluation_microbatch_size,
            "natural_continuation": (
                inputs.execution_preset.evaluation_microbatch_size
            ),
            "routing": inputs.execution_preset.evaluation_microbatch_size,
        },
        "timing_targets": {
            "logical_batch": "complete_test_suite",
            "pilot_seconds": 28_800,
            "synchronize_device": True,
        },
        "memory_targets": {
            "allocator_peak_bytes": (
                inputs.execution_preset.allocator_peak_limit_bytes
            ),
            "allocator_peak_gib": (
                inputs.execution_preset.allocator_peak_limit_bytes / _GIB
            ),
        },
        "model": _json_ready(asdict(checkpoint.config)),
    }
    return TinyWorldsReportManifest(
        preset=PILOT_PRESET_NAME,
        seed=PUBLIC_SEED,
        config_json=canonical_tinyworlds_config_json(config),
    )


def run_fixed_tinyworlds_pilot(
    temporary_directory: Path,
    progress: TinyWorldsProgressWriter,
) -> TinyWorldsCompletedResult:
    """Execute the one calibrated eight-task pilot and return immutable evidence."""
    if not isinstance(temporary_directory, Path):
        raise TypeError("temporary_directory must be a Path")
    if not isinstance(progress, TinyWorldsProgressWriter):
        raise TypeError("progress must be a TinyWorldsProgressWriter")
    inputs = prepare_fixed_tinyworlds_pilot_inputs(temporary_directory)
    exact_kg = _exact_kg_test_evidence(inputs)
    adaptations = train_fixed_tinyworlds_pilot_adaptations(
        inputs,
        temporary_directory,
    )
    knowledge_results: list[_KnowledgeStageEvaluation] = []
    stage_results: list[TinyWorldsPilotStageResult] = []
    for stage in TINYWORLDS_REPORT_STAGES:
        knowledge = _evaluate_stage_knowledge(inputs, adaptations, stage)
        knowledge_results.append(knowledge)
        vamp = adaptations.vamp_stages[stage - 1]
        diagnostics = vamp.parent_diagnostics[-1]
        drift = _committed_node_drift_records(
            inputs,
            adaptations,
            stage,
            knowledge,
            None if stage == 1 else knowledge_results[-2],
        )
        stage_result = TinyWorldsPilotStageResult(
            stage=stage,
            method_evaluations=knowledge.evaluations,
            natural_continuation_metrics=_natural_continuation_records(
                inputs,
                adaptations,
                stage,
            ),
            parent_search=_parent_search_records(diagnostics, stage),
            checkpointed_transfer=_counterfactual_transfer_records(
                diagnostics,
                stage,
            ),
            committed_node_drift=drift,
            memory_metrics=(_memory_record(inputs, vamp, stage),),
            addressing_cost=knowledge.addressing_cost,
            sequential_result=tinyworlds_record(
                sequence_index=stage - 1,
                stage=stage,
                event="stage_completed",
                task_id=TINYWORLDS_REPORT_TASK_IDS[stage - 1],
                learned_parent_id=str(vamp.graph.nodes[-1].parent_id),
                future_oracle_policy=PILOT_FORWARD_TRANSFER_POLICY,
                exact_kg_accuracy=exact_kg.accuracy,
                exact_kg_successes=exact_kg.successes,
                exact_kg_trials=exact_kg.trials,
            ),
        )
        progress.append_sequential(
            TinyWorldsSequentialResult(
                sequence_index=stage - 1,
                stage=stage,
                payload=_without_reserved_stage_fields(
                    stage_result.sequential_result
                ),
            )
        )
        progress.flush_sequential()
        stage_results.append(stage_result)
    stages = tuple(stage_results)
    final_evidence = _final_evidence(
        inputs,
        adaptations,
        stages,
        knowledge_results[-1],
        exact_kg,
    )
    manifest = build_tinyworlds_pilot_manifest(inputs, temporary_directory)
    completed = assemble_tinyworlds_completed_result(
        manifest,
        stages,
        final_evidence,
    )
    _write_json_atomic(
        temporary_directory / "completed_execution.json",
        {
            "adaptation_checkpoint": str(
                adaptations.training_workspace
                / "adaptation-artifacts"
                / "stage-08"
            ),
            "method_evaluation_count": len(completed.method_evaluations),
            "profile_sha256": inputs.profile_sha256,
            "run_id": completed.manifest.run_id,
            "stage_count": len(stages),
            "test_query_count": len(inputs.prepared.test_queries),
        },
    )
    return completed


def _adaptation_config_hashes(
    inputs: TinyWorldsPilotInputs,
) -> tuple[tuple[str, str], ...]:
    model = inputs.base_artifact.checkpoint.config
    lora = inputs.lora_config
    training = inputs.train_config
    payloads = {
        "model": {
            "vocab_size": model.vocab_size,
            "max_position_embeddings": model.max_position_embeddings,
            "hidden_size": model.hidden_size,
            "intermediate_size": model.intermediate_size,
            "num_layers": model.num_layers,
            "num_heads": model.num_heads,
            "attention_types": list(model.attention_types),
            "local_window_size": model.local_window_size,
            "layer_norm_epsilon": model.layer_norm_epsilon,
            "initializer_range": model.initializer_range,
            "activation": model.activation,
            "embedding_dropout": model.embedding_dropout,
            "attention_dropout": model.attention_dropout,
            "residual_dropout": model.residual_dropout,
        },
        "lora": {
            "rank": lora.rank,
            "alpha": float(lora.alpha),
            "target_mask": {
                name: getattr(lora.target_mask, name)
                for name in (
                    "query",
                    "key",
                    "value",
                    "attention_output",
                    "mlp_input",
                    "mlp_output",
                )
            },
        },
        "training": {
            "learning_rate": training.learning_rate,
            "steps": training.steps,
            "batch_size": training.batch_size,
            "weight_decay": training.weight_decay,
            "gradient_clip_norm": training.gradient_clip_norm,
        },
    }
    return tuple(
        sorted(
            (
                name,
                sha256(_canonical_json_compact_bytes(payload)).hexdigest(),
            )
            for name, payload in payloads.items()
        )
    ) + tuple(
        sorted(
            (
                ("tinyworlds_profile", inputs.profile_sha256),
                (
                    "tinyworlds_rendered",
                    sha256(inputs.rendered.bundle_id.encode("utf-8")).hexdigest(),
                ),
            )
        )
    )


def _stack_router_batches(batches: tuple[RouterBatch, ...]) -> RouterBatch:
    if not batches or any(not isinstance(batch, RouterBatch) for batch in batches):
        raise ValueError("router batches must contain RouterBatch values")
    if len({batch.input_ids.shape[1] for batch in batches}) != 1:
        raise ValueError("stacked router batches must share one sequence width")
    return RouterBatch(
        input_ids=np.concatenate(tuple(batch.input_ids for batch in batches), axis=0),
        attention_mask=np.concatenate(
            tuple(batch.attention_mask for batch in batches),
            axis=0,
        ),
        target_ids=np.concatenate(
            tuple(batch.target_ids for batch in batches),
            axis=0,
        ),
        loss_mask=np.concatenate(
            tuple(batch.loss_mask for batch in batches),
            axis=0,
        ),
    )


def _stack_competence_batches(
    batches: tuple[CompetenceBatch, ...],
) -> CompetenceBatch:
    if not batches or any(
        not isinstance(batch, CompetenceBatch) for batch in batches
    ):
        raise ValueError("competence batches must contain CompetenceBatch values")
    if len({batch.input_ids.shape[1] for batch in batches}) != 1:
        raise ValueError("stacked competence batches must share one sequence width")
    return CompetenceBatch(
        input_ids=np.concatenate(tuple(batch.input_ids for batch in batches), axis=0),
        attention_mask=np.concatenate(
            tuple(batch.attention_mask for batch in batches),
            axis=0,
        ),
        target_ids=np.concatenate(
            tuple(batch.target_ids for batch in batches),
            axis=0,
        ),
        loss_mask=np.concatenate(
            tuple(batch.loss_mask for batch in batches),
            axis=0,
        ),
    )


def _without_reserved_stage_fields(record: TinyWorldsRecord) -> TinyWorldsRecord:
    return TinyWorldsRecord(
        tuple(
            (key, value)
            for key, value in record.entries
            if key not in {"sequence_index", "stage"}
        )
    )


def _json_ready(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value


def _canonical_json_bytes(value: object) -> bytes:
    return _canonical_json_compact_bytes(value) + b"\n"


def _canonical_json_compact_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _load_canonical_json_object(payload: bytes) -> dict[str, object]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("JSON artifact is malformed") from error
    if payload != _canonical_json_bytes(value):
        raise ValueError("JSON artifact is not canonical")
    return _require_json_object(value, "JSON artifact")


def _require_json_object(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise ValueError(f"{label} must be a string-keyed object")
    return value


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"immutable JSON artifact already exists: {path}")
    payload = _canonical_json_bytes(value)
    with temporary.open("xb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    os.link(temporary, path)
    temporary.unlink()


def _write_or_verify_json(path: Path, value: object) -> None:
    expected = _canonical_json_bytes(value)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != expected:
            raise ValueError(f"immutable JSON artifact changed: {path}")
        return
    try:
        _write_json_atomic(path, value)
    except FileExistsError:
        if path.is_symlink() or path.read_bytes() != expected:
            raise ValueError(f"immutable JSON artifact changed: {path}")


def _write_durable_file(path: Path, payload: bytes) -> None:
    with path.open("xb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "PILOT_FORWARD_TRANSFER_POLICY",
    "PILOT_TRAINING_CACHE_ROOT",
    "TINYWORLDS_PILOT_EXECUTION_PRESET",
    "PilotStageExecutor",
    "TinyWorldsPilotExecutionPreset",
    "TinyWorldsPilotFinalEvidence",
    "TinyWorldsPilotInputs",
    "TinyWorldsPilotStageResult",
    "assemble_tinyworlds_completed_result",
    "build_tinyworlds_pilot_manifest",
    "execute_tinyworlds_pilot_stages",
    "load_locked_tinyworlds_profile",
    "load_tinyworlds_transfer_chunk",
    "prepare_fixed_tinyworlds_pilot_inputs",
    "run_fixed_tinyworlds_pilot",
    "save_tinyworlds_transfer_chunk",
    "tinyworlds_record",
    "train_fixed_tinyworlds_pilot_adaptations",
]

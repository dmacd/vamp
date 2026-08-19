"""Authenticated joint-IID LoRA rank sweep for the temporal final evaluation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
from pathlib import Path
from typing import TypeAlias

import jax
import numpy as np

from apm.data.text.tinyworlds_nouns_v2.contracts import (
    TASK_IDS,
    record_sha256,
    require_sha256,
)
from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation import (
    TemporalStudyInputs,
    assert_canonical_artifacts_unchanged,
    authenticate_temporal_study_inputs,
)
from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_contracts import (
    ALLOCATOR_LIMIT_BYTES,
    ARRIVAL_COUNT,
    BOOTSTRAP_REPETITIONS,
    CONTEXT_LENGTH,
    EVALUATION_ROW_FORMAT,
    FIXED_EPOCHS,
    GRADIENT_CLIP_NORM,
    LORA_LEARNING_RATE,
    PHYSICAL_BATCH_SIZE,
    SEED,
    SHARDS_PER_TASK,
    WEIGHT_DECAY,
)
from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_evaluation import (
    AdapterCandidate,
    MidpointCase,
    build_adapter_bank,
    evaluate_to_ledger,
    validate_evaluation_rows,
)
from apm.continual.artifacts import (
    ChainedJsonlLedger,
    file_sha256,
    load_canonical_json,
    publish_immutable_json,
)
from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_training import (
    AdapterArtifact,
    FullModelArtifact,
    StoryEpochBatches,
    TrainingJob,
    load_adapter_artifact,
    load_full_model_artifact,
    train_or_load_lora,
)
from apm.lm.lora import LoraConfig


STUDY_ID = "tinyworlds-nouns-v2-joint-iid-lora-rank-sweep"
CONTRACT_FORMAT = f"{STUDY_ID}-contract-v1"
REPORT_FORMAT = f"{STUDY_ID}-report-v1"
RANKS = (4, 8, 16, 32)
PARENT_CONTRACT_SHA256 = (
    "3f4ef4a10fd471b418a32a8f7b45431602c1f6abc080c19a7822ea2c2dd839b4"
)
PARENT_MANIFEST_SHA256 = (
    "15f3ee2a5a2c5054b158ba62d7a0d1b9fcaa22e40634a73c9cbffceca5888bcb"
)
CANONICAL_RANK8_JOB_SHA256 = (
    "cd4605c8240b459058c5a916ac6747edfd7712e99fcfd3710bd80cad1470a3cb"
)
CANONICAL_FULL_MODEL_JOB_SHA256 = (
    "61376ee6e474516ab6471d74ca97dfe2737586863c2d5b3a50c123147120bc80"
)
CANONICAL_RANK8_LEDGER_SHA256 = (
    "a0a5308b77bfc632dc91fb1b027e9f2fa1b9e7d51a6f913f5808733d3685692b"
)
CANONICAL_FULL_MODEL_LEDGER_SHA256 = (
    "46b9fef540af40897e08461a123839ee39f98c5b4566e22bd0583bf5a7ecbbe8"
)
EXPECTED_STORY_COUNT = 4_440
EXPECTED_TOKEN_COUNT = 476_035
EXPECTED_OPTIMIZER_UPDATES = 15_024
BASE_PARITY_TOLERANCE = 2e-5

TrainingProgress = Callable[[int, int, int, float, float], None]
EvaluationProgress = Callable[[int, int, int, Mapping[str, float]], None]
RankArtifacts: TypeAlias = tuple[tuple[int, AdapterArtifact], ...]
RankLedgers: TypeAlias = tuple[tuple[int, Path], ...]


@dataclass(frozen=True, slots=True)
class JointIidRankSweepInputs:
    """Authenticated parent controls and independent rank-sweep contract."""

    parent: TemporalStudyInputs
    canonical_rank8_job: TrainingJob
    canonical_rank8: AdapterArtifact
    canonical_full_model_job: TrainingJob
    canonical_full_model: FullModelArtifact
    rank8_rows: tuple[dict[str, object], ...]
    full_model_rows: tuple[dict[str, object], ...]
    contract: dict[str, object]
    result_directory: Path
    checkpoint_directory: Path
    work_directory: Path
    protected_files: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        for name in (
            "result_directory",
            "checkpoint_directory",
            "work_directory",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))
        if len(self.rank8_rows) != EXPECTED_STORY_COUNT:
            raise ValueError("rank-sweep reference coverage changed")

    @property
    def contract_sha256(self) -> str:
        """Return the independent sweep contract identity."""
        return str(self.contract["contract_sha256"])

    @property
    def all_story_ids(self) -> tuple[str, ...]:
        """Return the canonical joint-IID training population order."""
        return self.canonical_rank8_job.source_story_ids

    @property
    def source_shard_ids(self) -> tuple[str, ...]:
        """Return the canonical joint-IID shard order."""
        return self.canonical_rank8_job.source_shard_ids


def rank_lora_config(rank: int) -> LoraConfig:
    """Return the scale-one LoRA configuration for a preregistered rank."""
    if rank not in RANKS:
        raise ValueError(f"rank is outside the fixed sweep: {rank}")
    return LoraConfig(rank=rank, alpha=float(rank))


def rank_training_job(inputs: JointIidRankSweepInputs, rank: int) -> TrainingJob:
    """Return the canonical rank-8 job or one independently bound sweep job."""
    rank_lora_config(rank)
    if rank == 8:
        return inputs.canonical_rank8_job
    namespace = inputs.canonical_rank8_job.identity_sha256
    return TrainingJob(
        contract_sha256=inputs.contract_sha256,
        job_id=f"joint-iid-lora-rank-{rank}",
        family="joint_iid_lora",
        source_story_ids=inputs.all_story_ids,
        source_shard_ids=inputs.source_shard_ids,
        lora_rank=rank,
        lora_alpha=float(rank),
        batch_namespace_sha256=namespace,
        random_namespace_sha256=namespace,
    )


def authenticate_joint_iid_rank_sweep_inputs(
    repository_root: str | Path,
) -> JointIidRankSweepInputs:
    """Authenticate parent artifacts and publish the immutable sweep contract."""
    parent = authenticate_temporal_study_inputs(repository_root)
    if parent.contract_sha256 != PARENT_CONTRACT_SHA256:
        raise ValueError("joint-IID rank sweep parent contract changed")
    root = parent.repository_root
    parent_manifest_path = parent.result_directory / "manifest.json"
    parent_analysis_path = parent.result_directory / "analysis.json"
    parent_manifest = load_canonical_json(parent_manifest_path)
    manifest_core = {
        key: value
        for key, value in parent_manifest.items()
        if key != "manifest_sha256"
    }
    if (
        parent_manifest.get("manifest_sha256") != PARENT_MANIFEST_SHA256
        or parent_manifest.get("manifest_sha256") != record_sha256(manifest_core)
        or parent_manifest.get("contract_sha256") != parent.contract_sha256
        or type(parent_manifest.get("artifacts")) is not dict
    ):
        raise ValueError("joint-IID rank sweep parent manifest changed")
    if any(
        file_sha256(parent.result_directory / str(relative)) != str(digest)
        for relative, digest in dict(parent_manifest["artifacts"]).items()
    ):
        raise ValueError("joint-IID rank sweep parent publication files changed")
    parent_analysis = load_canonical_json(parent_analysis_path)
    analysis_core = {
        key: value
        for key, value in parent_analysis.items()
        if key != "analysis_sha256"
    }
    if parent_analysis.get("analysis_sha256") != record_sha256(analysis_core):
        raise ValueError("joint-IID rank sweep parent analysis changed")

    all_story_ids = tuple(
        story_id for shard in parent.shards for story_id in shard.story_ids
    )
    source_shard_ids = tuple(shard.shard_id for shard in parent.shards)
    rank8_job = TrainingJob(
        parent.contract_sha256,
        "joint-iid-lora",
        "joint_iid_lora",
        all_story_ids,
        source_shard_ids,
    )
    full_model_job = TrainingJob(
        parent.contract_sha256,
        "joint-iid-full-model",
        "joint_iid_full_model",
        all_story_ids,
        source_shard_ids,
    )
    if rank8_job.identity_sha256 != CANONICAL_RANK8_JOB_SHA256:
        raise ValueError("canonical joint-IID rank-8 job identity changed")
    if full_model_job.identity_sha256 != CANONICAL_FULL_MODEL_JOB_SHA256:
        raise ValueError("canonical joint-IID full-model job identity changed")
    rank8_directory = (
        parent.checkpoint_directory
        / "joint-iid-lora"
        / rank8_job.identity_sha256
    )
    full_model_directory = (
        parent.checkpoint_directory
        / "joint-iid-full-model"
        / full_model_job.identity_sha256
    )
    rank8_artifact = load_adapter_artifact(
        rank8_directory,
        rank8_job,
        parent.loaded_base.config,
        rank_lora_config(8),
    )
    full_model_artifact = load_full_model_artifact(
        full_model_directory,
        full_model_job,
    )
    if (
        rank8_artifact.optimizer_updates != EXPECTED_OPTIMIZER_UPDATES
        or full_model_artifact.optimizer_updates != EXPECTED_OPTIMIZER_UPDATES
    ):
        raise ValueError("canonical joint-IID optimizer coverage changed")

    rank8_ledger_path = (
        parent.work_directory
        / "evaluation-final-controls/final-stage-192-joint_iid_lora.jsonl"
    )
    full_model_ledger_path = (
        parent.work_directory
        / "evaluation-final-controls/final-stage-192-joint_iid_full_model.jsonl"
    )
    rank8_rows = _load_reference_rows(
        rank8_ledger_path,
        parent,
        "joint_iid_lora",
        CANONICAL_RANK8_LEDGER_SHA256,
    )
    full_model_rows = _load_reference_rows(
        full_model_ledger_path,
        parent,
        "joint_iid_full_model",
        CANONICAL_FULL_MODEL_LEDGER_SHA256,
    )
    if _row_identities(rank8_rows) != _row_identities(full_model_rows):
        raise ValueError("parent joint-IID controls changed story order")
    _validate_parent_aggregates(parent_analysis, rank8_rows, full_model_rows)

    manifest_artifacts = dict(parent_manifest["artifacts"])
    parent_publication_paths = tuple(
        parent.result_directory / relative
        for relative in (*sorted(manifest_artifacts), "manifest.json")
    )
    protected_paths = tuple(
        sorted(
            {
                *parent_publication_paths,
                *(
                    path
                    for directory in (rank8_directory, full_model_directory)
                    for path in directory.rglob("*")
                    if path.is_file()
                ),
                rank8_ledger_path,
                full_model_ledger_path,
            }
        )
    )
    protected_files = _file_snapshot(root, protected_paths)
    validation_order_sha256 = record_sha256(
        [[str(row["task_id"]), str(row["story_id"])] for row in rank8_rows]
    )
    core = {
        "bindings": {
            "base_manifest_sha256": parent.selected_base.reference.manifest_sha256,
            "base_parameter_checksum": parent.selected_base.reference.parameter_checksum,
            "canonical_full_model": _artifact_binding(root, full_model_directory),
            "canonical_full_model_job_sha256": full_model_job.identity_sha256,
            "canonical_full_model_ledger": {
                "path": full_model_ledger_path.relative_to(root).as_posix(),
                "row_count": len(full_model_rows),
                "sha256": file_sha256(full_model_ledger_path),
            },
            "canonical_rank8": _artifact_binding(root, rank8_directory),
            "canonical_rank8_job_sha256": rank8_job.identity_sha256,
            "canonical_rank8_ledger": {
                "path": rank8_ledger_path.relative_to(root).as_posix(),
                "row_count": len(rank8_rows),
                "sha256": file_sha256(rank8_ledger_path),
            },
            "parent_analysis_sha256": file_sha256(parent_analysis_path),
            "parent_contract_sha256": parent.contract_sha256,
            "parent_manifest_sha256": PARENT_MANIFEST_SHA256,
            "partition_sha256": parent.partition.partition_sha256,
            "source_shard_ids_sha256": record_sha256(list(source_shard_ids)),
            "source_story_count": len(all_story_ids),
            "source_story_ids_sha256": record_sha256(list(all_story_ids)),
            "validation_order_sha256": validation_order_sha256,
        },
        "bootstrap": {"repetitions": BOOTSTRAP_REPETITIONS, "seed": SEED},
        "evaluation": {
            "dataset": "official_final_4440_midpoint_suffix_cases",
            "metric_primary": ["story_mean_nll", "token_mean_nll"],
            "routing": "forced_single_adapter",
            "suffix_windowing": "canonical_reset_at_256",
        },
        "format": CONTRACT_FORMAT,
        "schema_version": 1,
        "study_id": STUDY_ID,
        "training": {
            "alpha_policy": "alpha_equals_rank",
            "batch_namespace_sha256": rank8_job.identity_sha256,
            "batch_size": PHYSICAL_BATCH_SIZE,
            "context_length": CONTEXT_LENGTH,
            "epochs": FIXED_EPOCHS,
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
            "learning_rate": LORA_LEARNING_RATE,
            "optimizer": "adamw",
            "optimizer_updates": rank8_artifact.optimizer_updates,
            "random_namespace_sha256": rank8_job.identity_sha256,
            "ranks": list(RANKS),
            "rank8_policy": "strict_load_parent_artifact_and_ledger",
            "scale": 1.0,
            "weight_decay": WEIGHT_DECAY,
        },
    }
    contract = {**core, "contract_sha256": record_sha256(core)}
    result_directory = parent.result_directory / "joint-iid-lora-rank-sweep-v1"
    checkpoint_directory = (
        parent.checkpoint_directory
        / "joint-iid-lora-rank-sweep-v1"
        / str(contract["contract_sha256"])
    )
    work_directory = (
        parent.work_directory
        / "joint-iid-lora-rank-sweep-v1"
        / str(contract["contract_sha256"])
    )
    for directory in (result_directory, checkpoint_directory, work_directory):
        directory.mkdir(parents=True, exist_ok=True)
    publish_immutable_json(result_directory / "contract.json", contract)
    return JointIidRankSweepInputs(
        parent,
        rank8_job,
        rank8_artifact,
        full_model_job,
        full_model_artifact,
        rank8_rows,
        full_model_rows,
        contract,
        result_directory,
        checkpoint_directory,
        work_directory,
        protected_files,
    )


def run_or_resume_rank_training(
    inputs: JointIidRankSweepInputs,
    *,
    progress: TrainingProgress | None = None,
) -> RankArtifacts:
    """Train or strict-load every rank while reusing canonical rank-8 evidence."""
    entry_lookup = inputs.parent.train_entry_lookup
    entries = tuple(entry_lookup[story_id] for story_id in inputs.all_story_ids)
    batches = StoryEpochBatches(
        inputs.parent.partition,
        entries,
        context_length=CONTEXT_LENGTH,
        batch_size=PHYSICAL_BATCH_SIZE,
        namespace=inputs.canonical_rank8_job.identity_sha256,
    )
    if len(batches) != EXPECTED_OPTIMIZER_UPDATES:
        raise ValueError("joint-IID optimizer update count changed")
    artifacts: list[tuple[int, AdapterArtifact]] = []
    for rank in RANKS:
        if rank == 8:
            artifact = inputs.canonical_rank8
        else:
            job = rank_training_job(inputs, rank)
            artifact = train_or_load_lora(
                job,
                batches,
                inputs.parent.loaded_base.params,
                inputs.parent.loaded_base.config,
                inputs.checkpoint_directory / f"rank-{rank:02d}",
                inputs.work_directory / "training",
                lora_config=rank_lora_config(rank),
                progress=(
                    None
                    if progress is None
                    else lambda _job, update, total, loss, elapsed, active_rank=rank: progress(
                        active_rank,
                        update,
                        total,
                        loss,
                        elapsed,
                    )
                ),
            )
        if artifact.optimizer_updates != EXPECTED_OPTIMIZER_UPDATES:
            raise ValueError(f"rank-{rank} optimizer coverage changed")
        if progress is not None:
            progress(
                rank,
                artifact.optimizer_updates,
                artifact.optimizer_updates,
                _last_loss(artifact.directory / "losses.jsonl"),
                artifact.runtime_seconds,
            )
        artifacts.append((rank, artifact))
    return tuple(artifacts)


def run_or_resume_rank_evaluation(
    inputs: JointIidRankSweepInputs,
    artifacts: RankArtifacts,
    cases: Sequence[MidpointCase],
    *,
    progress: EvaluationProgress | None = None,
) -> RankLedgers:
    """Evaluate each new rank on the exact parent final-case protocol."""
    artifact_by_rank = dict(artifacts)
    if tuple(artifact_by_rank) != RANKS or len(cases) != EXPECTED_STORY_COUNT:
        raise ValueError("rank-sweep artifacts or validation cases changed")
    ledgers: list[tuple[int, Path]] = []
    parent_rank8_path = (
        inputs.parent.work_directory
        / "evaluation-final-controls/final-stage-192-joint_iid_lora.jsonl"
    )
    for rank in RANKS:
        if rank == 8:
            if progress is not None:
                progress(rank, EXPECTED_STORY_COUNT, EXPECTED_STORY_COUNT, {})
            ledgers.append((rank, parent_rank8_path))
            continue
        artifact = artifact_by_rank[rank]
        bank = build_adapter_bank(
            (
                AdapterCandidate(
                    f"joint-iid-lora-rank-{rank}",
                    artifact.adapter_sha256,
                    artifact.adapter,
                    tuple((task_id, SHARDS_PER_TASK) for task_id in TASK_IDS),
                ),
            ),
            inputs.parent.loaded_base.config,
            rank_lora_config(rank),
        )
        ledger = ChainedJsonlLedger(
            inputs.work_directory / "evaluation" / f"rank-{rank:02d}.jsonl",
            EVALUATION_ROW_FORMAT,
        )
        _validate_sweep_rows(ledger.rows, inputs, rank, allow_prefix=True)
        evaluate_to_ledger(
            cases,
            contract_sha256=inputs.contract_sha256,
            evaluation_id="joint-iid-lora-rank-sweep",
            dataset="final",
            method=f"joint_iid_lora_rank_{rank}",
            order=None,
            stage=ARRIVAL_COUNT,
            routing="forced_adapter",
            base_params=inputs.parent.loaded_base.params,
            model_config=inputs.parent.loaded_base.config,
            bank=bank,
            ledger=ledger,
            progress=(
                None
                if progress is None
                else lambda completed, total, metrics, active_rank=rank: progress(
                    active_rank,
                    completed,
                    total,
                    metrics,
                )
            ),
        )
        _validate_sweep_rows(ledger.rows, inputs, rank, allow_prefix=False)
        if progress is not None:
            progress(rank, len(ledger.rows), EXPECTED_STORY_COUNT, {})
        ledgers.append((rank, ledger.path))
    return tuple(ledgers)


def analyze_joint_iid_rank_sweep(
    inputs: JointIidRankSweepInputs,
    artifacts: RankArtifacts,
    ledgers: RankLedgers,
    *,
    execution: Mapping[str, float],
    allocator: Mapping[str, object],
) -> dict[str, object]:
    """Aggregate exact comparable NLLs, per-task rows, and paired intervals."""
    artifact_by_rank = dict(artifacts)
    ledger_by_rank = dict(ledgers)
    if tuple(artifact_by_rank) != RANKS or tuple(ledger_by_rank) != RANKS:
        raise ValueError("rank-sweep analysis requires all ranks in fixed order")
    rows_by_condition: dict[str, tuple[dict[str, object], ...]] = {
        "full_model": inputs.full_model_rows,
        "rank_8": inputs.rank8_rows,
    }
    for rank in RANKS:
        if rank == 8:
            continue
        ledger = ChainedJsonlLedger(ledger_by_rank[rank], EVALUATION_ROW_FORMAT)
        _validate_sweep_rows(ledger.rows, inputs, rank, allow_prefix=False)
        rows_by_condition[f"rank_{rank}"] = ledger.rows
    reference_identities = _row_identities(inputs.rank8_rows)
    if any(
        _row_identities(rows) != reference_identities
        for rows in rows_by_condition.values()
    ):
        raise ValueError("rank-sweep ledgers do not share exact story order")
    reference_tokens = tuple(
        int(row["suffix_token_count"]) for row in inputs.rank8_rows
    )
    if any(
        tuple(int(row["suffix_token_count"]) for row in rows) != reference_tokens
        for rows in rows_by_condition.values()
    ) or sum(reference_tokens) != EXPECTED_TOKEN_COUNT:
        raise ValueError("rank-sweep ledgers do not share exact suffix tokens")
    base_drift = max(
        abs(
            float(row["suffix_mean_nll_by_candidate"][0])
            - float(reference["suffix_mean_nll_by_candidate"][0])
        )
        for rank in (4, 16, 32)
        for row, reference in zip(
            rows_by_condition[f"rank_{rank}"],
            inputs.rank8_rows,
            strict=True,
        )
    )
    if base_drift > BASE_PARITY_TOLERANCE:
        raise RuntimeError(f"rank-sweep base-path drift is too large: {base_drift}")

    ordered_conditions = ("full_model",) + tuple(f"rank_{rank}" for rank in RANKS)
    aggregate = tuple(
        _aggregate_rows(condition, rows_by_condition[condition])
        for condition in ordered_conditions
    )
    per_task = tuple(
        _aggregate_rows(
            condition,
            tuple(row for row in rows_by_condition[condition] if row["task_id"] == task_id),
            task_id=task_id,
        )
        for condition in ordered_conditions
        for task_id in TASK_IDS
    )
    bootstrap = _paired_bootstrap(rows_by_condition)
    training = tuple(
        _training_summary(
            inputs,
            rank,
            artifact_by_rank[rank],
            reused=rank == 8,
        )
        for rank in RANKS
    )
    return {
        "aggregate": aggregate,
        "allocator": dict(allocator),
        "bootstrap": bootstrap,
        "comparability": {
            "base_path_max_abs_story_nll_drift": base_drift,
            "batch_namespace_sha256": inputs.canonical_rank8_job.identity_sha256,
            "exact_story_order": True,
            "exact_suffix_token_masks": True,
            "full_model_source_ledger_sha256": file_sha256(
                inputs.parent.work_directory
                / "evaluation-final-controls/final-stage-192-joint_iid_full_model.jsonl"
            ),
            "rank8_source_ledger_sha256": file_sha256(ledger_by_rank[8]),
            "random_namespace_sha256": inputs.canonical_rank8_job.identity_sha256,
        },
        "execution": dict(execution),
        "ledger_provenance": tuple(
            {
                "path": path.relative_to(inputs.parent.repository_root).as_posix(),
                "rank": rank,
                "row_count": EXPECTED_STORY_COUNT,
                "sha256": file_sha256(path),
                "source": "canonical_parent" if rank == 8 else "rank_sweep",
            }
            for rank, path in ledgers
        ),
        "per_task": per_task,
        "provenance": {
            "base_parameter_checksum": inputs.parent.selected_base.reference.parameter_checksum,
            "canonical_full_model_job_sha256": inputs.canonical_full_model_job.identity_sha256,
            "canonical_rank8_job_sha256": inputs.canonical_rank8_job.identity_sha256,
            "contract_sha256": inputs.contract_sha256,
            "parent_contract_sha256": inputs.parent.contract_sha256,
            "parent_manifest_sha256": PARENT_MANIFEST_SHA256,
            "partition_sha256": inputs.parent.partition.partition_sha256,
        },
        "training": training,
    }


def assert_rank_sweep_inputs_unchanged(inputs: JointIidRankSweepInputs) -> None:
    """Reject mutations of canonical nouns or any bound parent control file."""
    assert_canonical_artifacts_unchanged(inputs.parent)
    paths = tuple(
        inputs.parent.repository_root / relative
        for relative, _ in inputs.protected_files
    )
    after = _file_snapshot(inputs.parent.repository_root, paths)
    if after != inputs.protected_files:
        before_map, after_map = dict(inputs.protected_files), dict(after)
        changed = tuple(
            path
            for path in sorted(set(before_map) | set(after_map))
            if before_map.get(path) != after_map.get(path)
        )
        raise RuntimeError(f"rank-sweep bound parent files changed: {changed}")


def _load_reference_rows(
    path: Path,
    parent: TemporalStudyInputs,
    method: str,
    expected_sha256: str,
) -> tuple[dict[str, object], ...]:
    if file_sha256(path) != expected_sha256:
        raise ValueError(f"canonical {method} evaluation ledger changed")
    ledger = ChainedJsonlLedger(path, EVALUATION_ROW_FORMAT)
    validate_evaluation_rows(ledger.rows, parent.contract_sha256)
    expected_ids = {
        (task_id, entry.story_id)
        for task_id, entries in parent.validation_entries
        for entry in entries
    }
    observed_ids = {
        (str(row["task_id"]), str(row["story_id"])) for row in ledger.rows
    }
    if (
        len(ledger.rows) != EXPECTED_STORY_COUNT
        or observed_ids != expected_ids
        or any(
            row["evaluation_id"] != "final-controls"
            or row["dataset"] != "final"
            or row["method"] != method
            or row["order"] is not None
            or row["stage"] != ARRIVAL_COUNT
            for row in ledger.rows
        )
    ):
        raise ValueError(f"canonical {method} evaluation coverage changed")
    return ledger.rows


def _validate_sweep_rows(
    rows: Sequence[Mapping[str, object]],
    inputs: JointIidRankSweepInputs,
    rank: int,
    *,
    allow_prefix: bool,
) -> None:
    validate_evaluation_rows(rows, inputs.contract_sha256)
    expected_order = _row_identities(inputs.rank8_rows)
    observed_order = _row_identities(rows)
    if (
        len(rows) > EXPECTED_STORY_COUNT
        or (not allow_prefix and len(rows) != EXPECTED_STORY_COUNT)
        or observed_order != expected_order[: len(rows)]
        or any(
            row["evaluation_id"] != "joint-iid-lora-rank-sweep"
            or row["dataset"] != "final"
            or row["method"] != f"joint_iid_lora_rank_{rank}"
            or row["order"] is not None
            or row["stage"] != ARRIVAL_COUNT
            or row["candidate_ids"] != ["base", f"joint-iid-lora-rank-{rank}"]
            or row["selected_index"] != 1
            for row in rows
        )
    ):
        raise ValueError(f"rank-{rank} evaluation ledger is not the canonical prefix")


def _validate_parent_aggregates(
    analysis: Mapping[str, object],
    rank8_rows: Sequence[Mapping[str, object]],
    full_model_rows: Sequence[Mapping[str, object]],
) -> None:
    nested = analysis.get("analysis")
    if type(nested) is not dict or type(nested.get("aggregate")) is not list:
        raise ValueError("parent temporal aggregate is malformed")
    published = {
        str(row["method"]): row
        for row in nested["aggregate"]
        if row["dataset"] == "final"
        and row["method"] in ("joint_iid_lora", "joint_iid_full_model")
    }
    derived = {
        "joint_iid_lora": _aggregate_rows("rank_8", rank8_rows),
        "joint_iid_full_model": _aggregate_rows("full_model", full_model_rows),
    }
    if set(published) != set(derived) or any(
        abs(float(published[method][metric]) - float(values[metric])) > 1e-12
        for method, values in derived.items()
        for metric in ("story_mean_nll", "token_mean_nll", "suffix_token_accuracy")
    ):
        raise ValueError("parent joint-IID aggregate no longer matches its ledgers")


def _aggregate_rows(
    condition: str,
    rows: Sequence[Mapping[str, object]],
    *,
    task_id: str | None = None,
) -> dict[str, object]:
    if not rows:
        raise ValueError("cannot aggregate empty rank-sweep rows")
    token_count = sum(int(row["suffix_token_count"]) for row in rows)
    total_nll = sum(float(row["suffix_total_nll"]) for row in rows)
    correct = sum(int(row["suffix_correct_tokens"]) for row in rows)
    rank = int(condition.removeprefix("rank_")) if condition.startswith("rank_") else None
    return {
        "condition": condition,
        "label": "Joint-IID full model" if rank is None else f"Joint-IID LoRA rank {rank}",
        "rank": rank,
        "story_count": len(rows),
        "story_mean_nll": float(np.mean([float(row["suffix_mean_nll"]) for row in rows])),
        "suffix_token_accuracy": correct / token_count,
        "task_id": task_id,
        "token_count": token_count,
        "token_mean_nll": total_nll / token_count,
    }


def _paired_bootstrap(
    rows_by_condition: Mapping[str, Sequence[Mapping[str, object]]],
) -> tuple[dict[str, object], ...]:
    conditions = tuple(f"rank_{rank}" for rank in RANKS)
    ordered = tuple(rows_by_condition[condition] for condition in ("full_model", *conditions))
    story_nll = np.asarray(
        [[float(row["suffix_mean_nll"]) for row in rows] for rows in ordered],
        dtype=np.float64,
    )
    total_nll = np.asarray(
        [[float(row["suffix_total_nll"]) for row in rows] for rows in ordered],
        dtype=np.float64,
    )
    token_count = np.asarray(
        [[int(row["suffix_token_count"]) for row in rows] for rows in ordered],
        dtype=np.float64,
    )
    task_positions = tuple(
        np.asarray(
            [index for index, row in enumerate(ordered[0]) if row["task_id"] == task_id],
            dtype=np.int32,
        )
        for task_id in TASK_IDS
    )
    if any(len(positions) <= 0 for positions in task_positions):
        raise ValueError("paired bootstrap lost a noun stratum")
    rng = np.random.default_rng(SEED)
    story_samples = np.empty(
        (BOOTSTRAP_REPETITIONS, len(ordered)),
        dtype=np.float64,
    )
    token_samples = np.empty_like(story_samples)
    for repetition in range(BOOTSTRAP_REPETITIONS):
        sampled = np.concatenate(
            tuple(
                rng.choice(positions, size=len(positions), replace=True)
                for positions in task_positions
            )
        )
        story_samples[repetition] = np.mean(story_nll[:, sampled], axis=1)
        token_samples[repetition] = np.sum(
            total_nll[:, sampled],
            axis=1,
        ) / np.sum(token_count[:, sampled], axis=1)
    condition_positions = {condition: index + 1 for index, condition in enumerate(conditions)}
    reference_positions = {"full_model": 0, "rank_8": condition_positions["rank_8"]}
    rows = []
    for condition in conditions:
        for reference, reference_position in reference_positions.items():
            if condition == reference:
                continue
            for metric, samples in (
                ("story_mean_nll", story_samples),
                ("token_mean_nll", token_samples),
            ):
                differences = (
                    samples[:, condition_positions[condition]]
                    - samples[:, reference_position]
                )
                observed_values = (
                    story_nll if metric == "story_mean_nll" else total_nll / token_count
                )
                if metric == "story_mean_nll":
                    estimate = float(
                        np.mean(observed_values[condition_positions[condition]])
                        - np.mean(observed_values[reference_position])
                    )
                else:
                    estimate = float(
                        np.sum(total_nll[condition_positions[condition]])
                        / np.sum(token_count[condition_positions[condition]])
                        - np.sum(total_nll[reference_position])
                        / np.sum(token_count[reference_position])
                    )
                lower, upper = np.quantile(differences, (0.025, 0.975))
                rows.append(
                    {
                        "condition": condition,
                        "estimate": estimate,
                        "lower_95": float(lower),
                        "metric": metric,
                        "reference": reference,
                        "upper_95": float(upper),
                    }
                )
    return tuple(rows)


def _training_summary(
    inputs: JointIidRankSweepInputs,
    rank: int,
    artifact: AdapterArtifact,
    *,
    reused: bool,
) -> dict[str, object]:
    parameter_count = sum(
        int(np.prod(np.asarray(value).shape, dtype=np.int64))
        for value in jax.tree_util.tree_leaves(artifact.adapter)
    )
    base_parameter_count = sum(
        int(np.prod(np.asarray(value).shape, dtype=np.int64))
        for value in jax.tree_util.tree_leaves(inputs.parent.loaded_base.params)
    )
    return {
        "adapter_file_bytes": (artifact.directory / "adapter.safetensors").stat().st_size,
        "adapter_parameter_count": parameter_count,
        "adapter_to_base_parameter_fraction": parameter_count / base_parameter_count,
        "allocator_peak_bytes": artifact.allocator_peak_bytes,
        "alpha": float(rank),
        "final_training_loss": _last_loss(artifact.directory / "losses.jsonl"),
        "job_sha256": artifact.job.identity_sha256,
        "optimizer_updates": artifact.optimizer_updates,
        "rank": rank,
        "reused_canonical_artifact": reused,
        "runtime_seconds": artifact.runtime_seconds,
        "scale": rank_lora_config(rank).scale,
        "tensor_file_sha256": artifact.tensor_file_sha256,
    }


def _artifact_binding(root: Path, directory: Path) -> dict[str, object]:
    files = tuple(path for path in directory.rglob("*") if path.is_file())
    return {
        "directory": directory.relative_to(root).as_posix(),
        "files": {path.relative_to(directory).as_posix(): file_sha256(path) for path in sorted(files)},
    }


def _file_snapshot(
    root: Path,
    paths: Sequence[Path],
) -> tuple[tuple[str, str], ...]:
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("rank-sweep protected file set is incomplete")
    return tuple(
        (path.relative_to(root).as_posix(), file_sha256(path))
        for path in sorted(paths)
    )


def _row_identities(
    rows: Sequence[Mapping[str, object]],
) -> tuple[tuple[str, str], ...]:
    return tuple((str(row["task_id"]), str(row["story_id"])) for row in rows)


def _last_loss(path: Path) -> float:
    rows = ChainedJsonlLedger(path, "tinyworlds-nouns-v2-temporal-consolidation-training-row-v1").rows
    if not rows:
        raise ValueError("rank-sweep training trace is empty")
    value = float(rows[-1]["loss"])
    if not math.isfinite(value):
        raise ValueError("rank-sweep final training loss is not finite")
    return value


__all__ = [
    "ALLOCATOR_LIMIT_BYTES",
    "BASE_PARITY_TOLERANCE",
    "EXPECTED_OPTIMIZER_UPDATES",
    "EXPECTED_STORY_COUNT",
    "JointIidRankSweepInputs",
    "RANKS",
    "REPORT_FORMAT",
    "RankArtifacts",
    "RankLedgers",
    "analyze_joint_iid_rank_sweep",
    "assert_rank_sweep_inputs_unchanged",
    "authenticate_joint_iid_rank_sweep_inputs",
    "rank_lora_config",
    "rank_training_job",
    "run_or_resume_rank_evaluation",
    "run_or_resume_rank_training",
]

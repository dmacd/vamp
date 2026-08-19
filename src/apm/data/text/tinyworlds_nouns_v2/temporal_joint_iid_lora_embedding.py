"""Joint-IID projection LoRA with a jointly trained tied token embedding."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
import math
import os
from pathlib import Path
import shutil
import tempfile
from time import monotonic
from typing import NamedTuple, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
import optax

from apm.continual.language_adaptation_artifact import (
    flatten_lora_edge,
    read_safetensors_archive,
    unflatten_lora_edge,
    write_safetensors_archive,
)
from apm.data.text.tinyworlds_nouns_v2.contracts import (
    TASK_IDS,
    canonical_json_bytes,
    record_sha256,
    require_sha256,
)
from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation import (
    assert_canonical_artifacts_unchanged,
)
from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_contracts import (
    ALLOCATOR_LIMIT_BYTES,
    ARRIVAL_COUNT,
    BOOTSTRAP_REPETITIONS,
    CONTEXT_LENGTH,
    EVALUATION_ROW_FORMAT,
    FIXED_EPOCHS,
    FULL_MODEL_LEARNING_RATE,
    GRADIENT_CLIP_NORM,
    LORA_LEARNING_RATE,
    PHYSICAL_BATCH_SIZE,
    SEED,
    SHARDS_PER_TASK,
    TRAINING_ROW_FORMAT,
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
    atomic_write,
    file_sha256,
    load_canonical_json,
    publish_immutable_json,
)
from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_training import (
    AdapterArtifact,
    StoryEpochBatches,
    TrainingInterrupted,
    load_adapter_artifact,
)
from apm.data.text.tinyworlds_nouns_v2.temporal_joint_iid_rank_sweep import (
    EXPECTED_OPTIMIZER_UPDATES,
    EXPECTED_STORY_COUNT,
    EXPECTED_TOKEN_COUNT,
    JointIidRankSweepInputs,
    _validate_sweep_rows,
    authenticate_joint_iid_rank_sweep_inputs,
    rank_lora_config,
    rank_training_job,
)
from apm.lm.config import GptNeoConfig
from apm.lm.gpt_neo import apply_gpt_neo
from apm.lm.lora import LoraConfig, LoraEdge, init_lora_edge
from apm.lm.lora_memory import pack_lora_memory, packed_with_candidate_edge
from apm.lm.losses import mean_token_nll
from apm.lm.parameters import GptNeoParams
from apm.lm.text_data import TokenBatch
from apm.lm.training import LmTrainState
from apm.lm.training_state_artifact import (
    load_lm_train_state_artifact,
    write_lm_train_state_artifact,
)
from apm.memory.graph import NodeId, init_memory_graph
from apm.data.text.tinyworlds_p.training import allocator_peak_bytes


STUDY_ID = "tinyworlds-nouns-v2-joint-iid-lora-trainable-embedding"
CONTRACT_FORMAT = f"{STUDY_ID}-contract-v1"
ARTIFACT_FORMAT = f"{STUDY_ID}-artifact-v1"
REPORT_FORMAT = f"{STUDY_ID}-report-v1"
RANKS = (8, 32)
PARENT_RANK_SWEEP_CONTRACT_SHA256 = (
    "e87a835334a64c22b634a5e51f300cf5ad5fd529bd9fdcdf2268842fbd3df301"
)
PARENT_RANK_SWEEP_MANIFEST_SHA256 = (
    "bf8b74cdb996679adf501234aaf4f540ba92cf599ac44590960d47ffc83676bb"
)
CHECKPOINT_UPDATE_INTERVAL = 512
CHECKPOINT_SECONDS = 180.0

TrainingProgress = Callable[[int, int, int, float, float], None]
EvaluationProgress = Callable[[int, int, int, Mapping[str, float]], None]


class LoraEmbeddingTrainable(NamedTuple):
    """One projection adapter and the tied input/output token matrix."""

    adapter: LoraEdge
    token_embedding: jax.Array


@dataclass(frozen=True, slots=True)
class LoraEmbeddingJob:
    """Immutable optimizer identity for one rank-shaped joint training run."""

    contract_sha256: str
    rank: int
    source_story_ids: tuple[str, ...]
    source_shard_ids: tuple[str, ...]
    batch_namespace_sha256: str
    random_namespace_sha256: str

    def __post_init__(self) -> None:
        require_sha256(self.contract_sha256, "embedding-LoRA contract")
        if self.rank not in RANKS or not self.source_story_ids:
            raise ValueError("embedding-LoRA job rank or stories are invalid")
        if len(set(self.source_story_ids)) != len(self.source_story_ids):
            raise ValueError("embedding-LoRA job stories must be unique")
        for digest in (
            *self.source_story_ids,
            *self.source_shard_ids,
            self.batch_namespace_sha256,
            self.random_namespace_sha256,
        ):
            require_sha256(digest, "embedding-LoRA source or namespace")

    @property
    def identity_sha256(self) -> str:
        """Return the content identity of this optimizer job."""
        return record_sha256(self.as_record())

    def as_record(self) -> dict[str, object]:
        """Return the canonical job payload without its derived identity."""
        return {
            "alpha": float(self.rank),
            "batch_namespace_sha256": self.batch_namespace_sha256,
            "contract_sha256": self.contract_sha256,
            "embedding_learning_rate": FULL_MODEL_LEARNING_RATE,
            "epochs": FIXED_EPOCHS,
            "family": "joint_iid_lora_trainable_tied_embedding",
            "format": f"{STUDY_ID}-training-job-v1",
            "lora_learning_rate": LORA_LEARNING_RATE,
            "random_namespace_sha256": self.random_namespace_sha256,
            "rank": self.rank,
            "source_shard_ids": list(self.source_shard_ids),
            "source_story_count": len(self.source_story_ids),
            "source_story_ids_sha256": record_sha256(list(self.source_story_ids)),
            "weight_decay": WEIGHT_DECAY,
        }


@dataclass(frozen=True, slots=True)
class LoraEmbeddingArtifact:
    """Strict-loaded adapter, tied embedding, trace, and measurements."""

    directory: Path
    job: LoraEmbeddingJob
    trainable: LoraEmbeddingTrainable
    trainable_sha256: str
    adapter_sha256: str
    embedding_sha256: str
    tensor_file_sha256: str
    loss_trace_sha256: str
    optimizer_updates: int
    runtime_seconds: float
    allocator_peak_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "directory", Path(self.directory))
        for digest in (
            self.trainable_sha256,
            self.adapter_sha256,
            self.embedding_sha256,
            self.tensor_file_sha256,
            self.loss_trace_sha256,
        ):
            require_sha256(digest, "embedding-LoRA artifact")
        if (
            self.optimizer_updates <= 0
            or not math.isfinite(self.runtime_seconds)
            or self.runtime_seconds < 0.0
            or self.allocator_peak_bytes < 0
        ):
            raise ValueError("embedding-LoRA artifact measurements are invalid")


@dataclass(frozen=True, slots=True)
class LoraEmbeddingInputs:
    """Authenticated parent sweep, reference controls, and new contract."""

    rank_sweep: JointIidRankSweepInputs
    standard_rank32: AdapterArtifact
    standard_rank32_rows: tuple[dict[str, object], ...]
    contract: dict[str, object]
    result_directory: Path
    checkpoint_directory: Path
    work_directory: Path
    protected_files: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        for name in ("result_directory", "checkpoint_directory", "work_directory"):
            object.__setattr__(self, name, Path(getattr(self, name)))
        if len(self.standard_rank32_rows) != EXPECTED_STORY_COUNT:
            raise ValueError("embedding-LoRA rank-32 reference coverage changed")

    @property
    def contract_sha256(self) -> str:
        """Return the independent addendum contract identity."""
        return str(self.contract["contract_sha256"])

    @property
    def parent(self):
        """Return the authenticated temporal-study inputs."""
        return self.rank_sweep.parent


Artifacts: TypeAlias = tuple[tuple[int, LoraEmbeddingArtifact], ...]
Ledgers: TypeAlias = tuple[tuple[int, Path], ...]


def embedding_lora_job(inputs: LoraEmbeddingInputs, rank: int) -> LoraEmbeddingJob:
    """Build one job bound to the inherited rank-8 batch and random streams."""
    if rank not in RANKS:
        raise ValueError(f"rank is outside the embedding-LoRA study: {rank}")
    namespace = inputs.rank_sweep.canonical_rank8_job.identity_sha256
    return LoraEmbeddingJob(
        inputs.contract_sha256,
        rank,
        inputs.rank_sweep.all_story_ids,
        inputs.rank_sweep.source_shard_ids,
        namespace,
        namespace,
    )


def authenticate_lora_embedding_inputs(
    repository_root: str | Path,
) -> LoraEmbeddingInputs:
    """Authenticate both parent studies and publish the independent contract."""
    sweep = authenticate_joint_iid_rank_sweep_inputs(repository_root)
    if sweep.contract_sha256 != PARENT_RANK_SWEEP_CONTRACT_SHA256:
        raise ValueError("embedding-LoRA parent rank-sweep contract changed")
    root = sweep.parent.repository_root
    sweep_manifest_path = sweep.result_directory / "manifest.json"
    sweep_manifest = load_canonical_json(sweep_manifest_path)
    manifest_core = {
        key: value for key, value in sweep_manifest.items() if key != "manifest_sha256"
    }
    if (
        sweep_manifest.get("manifest_sha256") != PARENT_RANK_SWEEP_MANIFEST_SHA256
        or sweep_manifest.get("manifest_sha256") != record_sha256(manifest_core)
        or sweep_manifest.get("contract_sha256") != sweep.contract_sha256
        or type(sweep_manifest.get("artifacts")) is not dict
    ):
        raise ValueError("embedding-LoRA parent rank-sweep manifest changed")
    if any(
        file_sha256(sweep.result_directory / str(relative)) != str(digest)
        for relative, digest in dict(sweep_manifest["artifacts"]).items()
    ):
        raise ValueError("embedding-LoRA parent rank-sweep publication changed")

    rank32_job = rank_training_job(sweep, 32)
    rank32_directory = (
        sweep.checkpoint_directory / "rank-32" / rank32_job.identity_sha256
    )
    rank32_artifact = load_adapter_artifact(
        rank32_directory,
        rank32_job,
        sweep.parent.loaded_base.config,
        rank_lora_config(32),
    )
    rank32_ledger_path = sweep.work_directory / "evaluation/rank-32.jsonl"
    rank32_ledger = ChainedJsonlLedger(rank32_ledger_path, EVALUATION_ROW_FORMAT)
    _validate_sweep_rows(rank32_ledger.rows, sweep, 32, allow_prefix=False)

    protected_paths = tuple(
        sorted(
            {
                *(root / relative for relative, _ in sweep.protected_files),
                *(
                    path
                    for path in sweep.result_directory.iterdir()
                    if path.is_file()
                ),
                *(
                    path
                    for path in rank32_directory.rglob("*")
                    if path.is_file()
                ),
                rank32_ledger_path,
            }
        )
    )
    protected_files = _file_snapshot(root, protected_paths)
    namespace = sweep.canonical_rank8_job.identity_sha256
    core = {
        "bindings": {
            "base_manifest_sha256": sweep.parent.selected_base.reference.manifest_sha256,
            "base_parameter_checksum": sweep.parent.selected_base.reference.parameter_checksum,
            "parent_rank_sweep_contract_sha256": sweep.contract_sha256,
            "parent_rank_sweep_manifest_sha256": PARENT_RANK_SWEEP_MANIFEST_SHA256,
            "rank32_adapter_sha256": rank32_artifact.adapter_sha256,
            "rank32_ledger_sha256": file_sha256(rank32_ledger_path),
            "rank8_adapter_sha256": sweep.canonical_rank8.adapter_sha256,
            "rank8_ledger_sha256": file_sha256(
                sweep.parent.work_directory
                / "evaluation-final-controls/final-stage-192-joint_iid_lora.jsonl"
            ),
            "source_story_ids_sha256": record_sha256(list(sweep.all_story_ids)),
            "validation_order_sha256": record_sha256(
                [
                    [str(row["task_id"]), str(row["story_id"])]
                    for row in sweep.rank8_rows
                ]
            ),
        },
        "bootstrap": {"repetitions": BOOTSTRAP_REPETITIONS, "seed": SEED},
        "evaluation": {
            "dataset": "official_final_4440_midpoint_suffix_cases",
            "routing": "forced_single_adapter",
            "suffix_target_count": EXPECTED_TOKEN_COUNT,
            "suffix_windowing": "canonical_reset_at_256",
        },
        "format": CONTRACT_FORMAT,
        "schema_version": 1,
        "study_id": STUDY_ID,
        "training": {
            "alpha_policy": "alpha_equals_rank",
            "batch_namespace_sha256": namespace,
            "batch_size": PHYSICAL_BATCH_SIZE,
            "context_length": CONTEXT_LENGTH,
            "embedding_initialization": "authenticated_base_token_embedding",
            "embedding_learning_rate": FULL_MODEL_LEARNING_RATE,
            "embedding_semantics": "one_matrix_tied_for_input_and_output_logits",
            "epochs": FIXED_EPOCHS,
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
            "lora_learning_rate": LORA_LEARNING_RATE,
            "lora_targets": [
                "query",
                "key",
                "value",
                "attention_output",
                "mlp_input",
                "mlp_output",
            ],
            "optimizer": "joint_adamw_with_parameter_group_learning_rates",
            "optimizer_updates": EXPECTED_OPTIMIZER_UPDATES,
            "random_namespace_sha256": namespace,
            "ranks": list(RANKS),
            "scale": 1.0,
            "weight_decay": WEIGHT_DECAY,
        },
    }
    contract = {**core, "contract_sha256": record_sha256(core)}
    result_directory = sweep.parent.result_directory / "joint-iid-lora-embedding-v1"
    checkpoint_directory = (
        sweep.parent.checkpoint_directory
        / "joint-iid-lora-embedding-v1"
        / str(contract["contract_sha256"])
    )
    work_directory = (
        sweep.parent.work_directory
        / "joint-iid-lora-embedding-v1"
        / str(contract["contract_sha256"])
    )
    for directory in (result_directory, checkpoint_directory, work_directory):
        directory.mkdir(parents=True, exist_ok=True)
    publish_immutable_json(result_directory / "contract.json", contract)
    return LoraEmbeddingInputs(
        sweep,
        rank32_artifact,
        rank32_ledger.rows,
        contract,
        result_directory,
        checkpoint_directory,
        work_directory,
        protected_files,
    )


def run_or_resume_lora_embedding_training(
    inputs: LoraEmbeddingInputs,
    *,
    progress: TrainingProgress | None = None,
) -> Artifacts:
    """Train or strict-load the rank-8 and rank-32 joint parameterizations."""
    entry_lookup = inputs.parent.train_entry_lookup
    entries = tuple(
        entry_lookup[story_id] for story_id in inputs.rank_sweep.all_story_ids
    )
    batches = StoryEpochBatches(
        inputs.parent.partition,
        entries,
        context_length=CONTEXT_LENGTH,
        batch_size=PHYSICAL_BATCH_SIZE,
        namespace=inputs.rank_sweep.canonical_rank8_job.identity_sha256,
    )
    if len(batches) != EXPECTED_OPTIMIZER_UPDATES:
        raise ValueError("embedding-LoRA optimizer update count changed")
    artifacts = tuple(
        (
            rank,
            train_or_load_lora_embedding(
                embedding_lora_job(inputs, rank),
                batches,
                inputs.parent.loaded_base.params,
                inputs.parent.loaded_base.config,
                inputs.checkpoint_directory / f"rank-{rank:02d}",
                inputs.work_directory / "training",
                progress=(
                    None
                    if progress is None
                    else lambda update, total, loss, elapsed, active_rank=rank: progress(
                        active_rank,
                        update,
                        total,
                        loss,
                        elapsed,
                    )
                ),
            ),
        )
        for rank in RANKS
    )
    if any(
        artifact.optimizer_updates != EXPECTED_OPTIMIZER_UPDATES
        for _, artifact in artifacts
    ):
        raise ValueError("embedding-LoRA optimizer coverage changed")
    if progress is not None:
        for rank, artifact in artifacts:
            progress(
                rank,
                artifact.optimizer_updates,
                artifact.optimizer_updates,
                _last_loss(artifact.directory / "losses.jsonl"),
                artifact.runtime_seconds,
            )
    return artifacts


def train_or_load_lora_embedding(
    job: LoraEmbeddingJob,
    batches: Sequence[TokenBatch],
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    output_directory: str | Path,
    work_directory: str | Path,
    *,
    progress: Callable[[int, int, float, float], None] | None = None,
    stop_after_update: int | None = None,
) -> LoraEmbeddingArtifact:
    """Train, exactly resume, or strict-load one LoRA plus tied embedding."""
    if not batches:
        raise ValueError("embedding-LoRA training requires finite batches")
    lora_config = rank_lora_config(job.rank)
    target = Path(output_directory) / job.identity_sha256
    if target.is_dir():
        return load_lora_embedding_artifact(
            target,
            job,
            model_config,
            lora_config,
        )
    work = Path(work_directory) / job.identity_sha256
    work.mkdir(parents=True, exist_ok=True)
    ledger = ChainedJsonlLedger(work / "losses.jsonl", TRAINING_ROW_FORMAT)
    initialization_key, training_key = jax.random.split(
        _job_key(job.random_namespace_sha256)
    )
    starting = LoraEmbeddingTrainable(
        init_lora_edge(initialization_key, model_config, lora_config),
        jnp.asarray(base_params.token_embedding, dtype=jnp.float32),
    )
    optimizer = _joint_optimizer(model_config, lora_config)
    template = LmTrainState(
        starting,
        optimizer.init(starting),
        training_key,
        jnp.asarray(0, dtype=jnp.int32),
    )
    state = _latest_state(work / "states", job.identity_sha256, template)
    start_update = int(state.step)
    if len(ledger.rows) < start_update:
        raise ValueError("embedding-LoRA loss ledger is behind optimizer state")
    ledger.truncate(start_update)
    empty_memory = pack_lora_memory(
        init_memory_graph(NodeId("root")),
        model_config,
        lora_config,
        max_nodes=2,
        max_edges=1,
    )
    compiled_step = _compiled_joint_step(model_config, lora_config)
    prior_elapsed = _load_elapsed(work / "runtime.json", job.identity_sha256)
    started = monotonic()
    last_checkpoint = monotonic()
    current = state
    for update in range(start_update, len(batches)):
        current, loss = compiled_step(
            current,
            batches[update],
            base_params,
            empty_memory,
        )
        scalar_loss = float(jax.block_until_ready(loss))
        elapsed = prior_elapsed + monotonic() - started
        ledger.append(
            {
                "job_sha256": job.identity_sha256,
                "loss": scalar_loss,
                "update": update + 1,
            }
        )
        if progress is not None:
            progress(update + 1, len(batches), scalar_loss, elapsed)
        checkpoint_due = (
            (update + 1) % CHECKPOINT_UPDATE_INTERVAL == 0
            or monotonic() - last_checkpoint >= CHECKPOINT_SECONDS
            or update + 1 == len(batches)
            or stop_after_update == update + 1
        )
        if checkpoint_due:
            _write_latest_state(
                work / "states",
                job.identity_sha256,
                current,
            )
            _write_elapsed(
                work / "runtime.json",
                job.identity_sha256,
                update + 1,
                elapsed,
            )
            last_checkpoint = monotonic()
        if stop_after_update == update + 1:
            raise TrainingInterrupted(
                f"intentional interruption of embedding-LoRA rank {job.rank} "
                f"at update {update + 1}"
            )
    jax.tree_util.tree_map(jax.block_until_ready, current)
    peak = allocator_peak_bytes()
    if peak > ALLOCATOR_LIMIT_BYTES:
        raise RuntimeError(
            f"embedding-LoRA allocator peak {peak} exceeds {ALLOCATOR_LIMIT_BYTES}"
        )
    return _publish_lora_embedding_artifact(
        target,
        job,
        current.trainable,
        model_config,
        lora_config,
        work / "losses.jsonl",
        len(batches),
        _load_elapsed(work / "runtime.json", job.identity_sha256),
        peak,
    )


@lru_cache(maxsize=None)
def _joint_optimizer(
    model_config: GptNeoConfig,
    lora_config: LoraConfig,
) -> optax.GradientTransformation:
    label_adapter = jax.tree_util.tree_map(
        lambda _: "lora",
        init_lora_edge(jax.random.PRNGKey(0), model_config, lora_config),
    )
    labels = LoraEmbeddingTrainable(label_adapter, "embedding")
    return optax.chain(
        optax.clip_by_global_norm(GRADIENT_CLIP_NORM),
        optax.multi_transform(
            {
                "embedding": optax.adamw(
                    learning_rate=FULL_MODEL_LEARNING_RATE,
                    weight_decay=WEIGHT_DECAY,
                ),
                "lora": optax.adamw(
                    learning_rate=LORA_LEARNING_RATE,
                    weight_decay=WEIGHT_DECAY,
                ),
            },
            labels,
        ),
    )


@lru_cache(maxsize=None)
def _compiled_joint_step(
    model_config: GptNeoConfig,
    lora_config: LoraConfig,
):
    optimizer = _joint_optimizer(model_config, lora_config)
    coefficients = jnp.ones((1,), dtype=jnp.float32)

    @jax.jit
    def compiled(
        state: LmTrainState[LoraEmbeddingTrainable],
        batch: TokenBatch,
        base_params: GptNeoParams,
        empty_memory,
    ) -> tuple[LmTrainState[LoraEmbeddingTrainable], jax.Array]:
        frozen_base = jax.tree_util.tree_map(jax.lax.stop_gradient, base_params)
        next_rng_key, dropout_key = jax.random.split(state.rng_key)

        def loss_function(trainable: LoraEmbeddingTrainable) -> jax.Array:
            effective_base = frozen_base._replace(
                token_embedding=trainable.token_embedding
            )
            memory = packed_with_candidate_edge(
                empty_memory,
                trainable.adapter,
                0,
            )
            result = apply_gpt_neo(
                effective_base,
                model_config,
                jnp.asarray(batch.input_ids, dtype=jnp.int32),
                jnp.asarray(batch.attention_mask, dtype=jnp.bool_),
                lora_memory=memory,
                edge_coefficients=coefficients,
                lora_config=lora_config,
                training=True,
                rng_key=dropout_key,
            )
            return mean_token_nll(
                result.logits,
                jnp.asarray(batch.target_ids, dtype=jnp.int32),
                jnp.asarray(batch.loss_mask, dtype=jnp.float32),
            )

        loss, gradients = jax.value_and_grad(loss_function)(state.trainable)
        updates, next_optimizer_state = optimizer.update(
            gradients,
            state.opt_state,
            state.trainable,
        )
        return (
            LmTrainState(
                optax.apply_updates(state.trainable, updates),
                next_optimizer_state,
                next_rng_key,
                state.step + jnp.asarray(1, dtype=jnp.int32),
            ),
            loss,
        )

    return compiled


def lora_embedding_checksum(
    trainable: LoraEmbeddingTrainable,
    model_config: GptNeoConfig,
    lora_config: LoraConfig,
) -> str:
    """Hash names, shapes, dtypes, and bytes of the complete trainable value."""
    digest = sha256()
    for name, value in sorted(
        _trainable_tensors(trainable, model_config, lora_config).items()
    ):
        array = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
        digest.update(name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def embedding_checksum(embedding: jax.Array) -> str:
    """Hash the tied embedding independently of the adapter factors."""
    value = np.ascontiguousarray(np.asarray(embedding, dtype=np.float32))
    digest = sha256()
    digest.update(b"token_embedding")
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def load_lora_embedding_artifact(
    directory: str | Path,
    job: LoraEmbeddingJob,
    model_config: GptNeoConfig,
    lora_config: LoraConfig,
) -> LoraEmbeddingArtifact:
    """Strict-load one published adapter and tied-embedding artifact."""
    root = Path(directory)
    if {path.name for path in root.iterdir()} != {
        "losses.jsonl",
        "manifest.json",
        "trainable.safetensors",
    }:
        raise ValueError("embedding-LoRA artifact entries changed")
    manifest = load_canonical_json(root / "manifest.json")
    supplied = manifest.get("result_sha256")
    core = {key: value for key, value in manifest.items() if key != "result_sha256"}
    if (
        manifest.get("format") != ARTIFACT_FORMAT
        or manifest.get("job") != job.as_record()
        or manifest.get("job_sha256") != job.identity_sha256
        or supplied != record_sha256(core)
    ):
        raise ValueError("embedding-LoRA artifact manifest changed")
    tensor_path = root / "trainable.safetensors"
    loss_path = root / "losses.jsonl"
    if (
        manifest.get("tensor_file_sha256") != file_sha256(tensor_path)
        or manifest.get("loss_trace_sha256") != file_sha256(loss_path)
    ):
        raise ValueError("embedding-LoRA artifact file hash changed")
    tensors, metadata = read_safetensors_archive(tensor_path)
    if metadata != {
        "format": ARTIFACT_FORMAT,
        "job_sha256": job.identity_sha256,
        "trainable_sha256": str(manifest["trainable_sha256"]),
    }:
        raise ValueError("embedding-LoRA safetensors metadata changed")
    if "token_embedding" not in tensors:
        raise ValueError("embedding-LoRA token embedding is missing")
    embedding = np.asarray(tensors["token_embedding"], dtype=np.float32)
    if embedding.shape != (model_config.vocab_size, model_config.hidden_size):
        raise ValueError("embedding-LoRA token embedding shape changed")
    lora_tensors = {
        name.removeprefix("lora."): value
        for name, value in tensors.items()
        if name.startswith("lora.")
    }
    if len(lora_tensors) != len(tensors) - 1:
        raise ValueError("embedding-LoRA tensor names changed")
    adapter = unflatten_lora_edge(lora_tensors, model_config, lora_config)
    trainable = LoraEmbeddingTrainable(adapter, jnp.asarray(embedding))
    from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_training import (
        adapter_checksum,
    )

    if (
        lora_embedding_checksum(trainable, model_config, lora_config)
        != manifest.get("trainable_sha256")
        or adapter_checksum(adapter, model_config, lora_config)
        != manifest.get("adapter_sha256")
        or embedding_checksum(trainable.token_embedding)
        != manifest.get("embedding_sha256")
    ):
        raise ValueError("embedding-LoRA tensor checksum changed")
    ledger = ChainedJsonlLedger(loss_path, TRAINING_ROW_FORMAT)
    updates = int(manifest["optimizer_updates"])
    if (
        len(ledger.rows) != updates
        or not ledger.rows
        or int(ledger.rows[-1]["update"]) != updates
    ):
        raise ValueError("embedding-LoRA training trace coverage changed")
    return LoraEmbeddingArtifact(
        root,
        job,
        trainable,
        str(manifest["trainable_sha256"]),
        str(manifest["adapter_sha256"]),
        str(manifest["embedding_sha256"]),
        str(manifest["tensor_file_sha256"]),
        str(manifest["loss_trace_sha256"]),
        updates,
        float(manifest["runtime_seconds"]),
        int(manifest["allocator_peak_bytes"]),
    )


def _publish_lora_embedding_artifact(
    target: Path,
    job: LoraEmbeddingJob,
    trainable: LoraEmbeddingTrainable,
    model_config: GptNeoConfig,
    lora_config: LoraConfig,
    loss_path: Path,
    updates: int,
    runtime: float,
    peak: int,
) -> LoraEmbeddingArtifact:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    try:
        from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_training import (
            adapter_checksum,
        )

        trainable_sha256 = lora_embedding_checksum(
            trainable,
            model_config,
            lora_config,
        )
        tensor_path = temporary / "trainable.safetensors"
        write_safetensors_archive(
            tensor_path,
            {
                name: np.asarray(value, dtype=np.float32)
                for name, value in sorted(
                    _trainable_tensors(trainable, model_config, lora_config).items()
                )
            },
            {
                "format": ARTIFACT_FORMAT,
                "job_sha256": job.identity_sha256,
                "trainable_sha256": trainable_sha256,
            },
        )
        shutil.copyfile(loss_path, temporary / "losses.jsonl")
        core = {
            "adapter_sha256": adapter_checksum(
                trainable.adapter,
                model_config,
                lora_config,
            ),
            "allocator_peak_bytes": peak,
            "embedding_sha256": embedding_checksum(trainable.token_embedding),
            "format": ARTIFACT_FORMAT,
            "job": job.as_record(),
            "job_sha256": job.identity_sha256,
            "loss_trace_sha256": file_sha256(temporary / "losses.jsonl"),
            "optimizer_updates": updates,
            "runtime_seconds": runtime,
            "tensor_file": "trainable.safetensors",
            "tensor_file_sha256": file_sha256(tensor_path),
            "trainable_sha256": trainable_sha256,
        }
        atomic_write(
            temporary / "manifest.json",
            canonical_json_bytes({**core, "result_sha256": record_sha256(core)}),
        )
        os.replace(temporary, target)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return load_lora_embedding_artifact(target, job, model_config, lora_config)


def _trainable_tensors(
    trainable: LoraEmbeddingTrainable,
    model_config: GptNeoConfig,
    lora_config: LoraConfig,
) -> dict[str, jax.Array]:
    tensors = {
        f"lora.{name}": value
        for name, value in flatten_lora_edge(
            trainable.adapter,
            model_config,
            lora_config,
        ).items()
    }
    return {**tensors, "token_embedding": trainable.token_embedding}


def _latest_state(
    states_root: Path,
    identity_sha256: str,
    template: LmTrainState[LoraEmbeddingTrainable],
) -> LmTrainState[LoraEmbeddingTrainable]:
    states_root.mkdir(parents=True, exist_ok=True)
    candidates = tuple(
        sorted(
            path
            for path in states_root.iterdir()
            if path.is_dir() and path.name.startswith("state-")
        )
    )
    if not candidates:
        return template
    return load_lm_train_state_artifact(
        candidates[-1],
        identity_sha256,
        (template,),
    )[0]


def _write_latest_state(
    states_root: Path,
    identity_sha256: str,
    state: LmTrainState[LoraEmbeddingTrainable],
) -> None:
    states_root.mkdir(parents=True, exist_ok=True)
    target = states_root / f"state-{int(state.step):08d}"
    if not target.is_dir():
        write_lm_train_state_artifact(target, identity_sha256, (state,))
    loaded = load_lm_train_state_artifact(target, identity_sha256, (state,))[0]
    if int(loaded.step) != int(state.step):
        raise ValueError("embedding-LoRA persisted update changed")
    for path in tuple(states_root.iterdir()):
        if path != target and path.is_dir() and path.name.startswith("state-"):
            shutil.rmtree(path)


def _job_key(identity_sha256: str) -> jax.Array:
    require_sha256(identity_sha256, "embedding-LoRA random namespace")
    return jax.random.PRNGKey(int(identity_sha256[:8], 16) ^ SEED)


def _load_elapsed(path: Path, identity_sha256: str) -> float:
    if not path.is_file():
        return 0.0
    record = load_canonical_json(path)
    supplied = record.get("result_sha256")
    core = {key: value for key, value in record.items() if key != "result_sha256"}
    elapsed = record.get("elapsed_seconds")
    if (
        record.get("format") != f"{STUDY_ID}-training-runtime-v1"
        or record.get("job_sha256") != identity_sha256
        or supplied != record_sha256(core)
        or type(elapsed) not in (int, float)
        or not math.isfinite(float(elapsed))
        or float(elapsed) < 0.0
    ):
        raise ValueError("embedding-LoRA runtime state changed")
    return float(elapsed)


def _write_elapsed(
    path: Path,
    identity_sha256: str,
    update: int,
    elapsed_seconds: float,
) -> None:
    core = {
        "elapsed_seconds": elapsed_seconds,
        "format": f"{STUDY_ID}-training-runtime-v1",
        "job_sha256": identity_sha256,
        "update": update,
    }
    atomic_write(
        path,
        canonical_json_bytes({**core, "result_sha256": record_sha256(core)}),
    )


def run_or_resume_lora_embedding_evaluation(
    inputs: LoraEmbeddingInputs,
    artifacts: Artifacts,
    cases: Sequence[MidpointCase],
    *,
    progress: EvaluationProgress | None = None,
) -> Ledgers:
    """Force each joint model over the exact parent final suffix cases."""
    artifact_by_rank = dict(artifacts)
    if tuple(artifact_by_rank) != RANKS or len(cases) != EXPECTED_STORY_COUNT:
        raise ValueError("embedding-LoRA artifacts or cases changed")
    ledgers: list[tuple[int, Path]] = []
    for rank in RANKS:
        artifact = artifact_by_rank[rank]
        candidate_id = f"joint-iid-lora-embedding-rank-{rank}"
        bank = build_adapter_bank(
            (
                AdapterCandidate(
                    candidate_id,
                    artifact.adapter_sha256,
                    artifact.trainable.adapter,
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
        _validate_joint_rows(ledger.rows, inputs, rank, allow_prefix=True)
        effective_params = inputs.parent.loaded_base.params._replace(
            token_embedding=artifact.trainable.token_embedding
        )
        evaluate_to_ledger(
            cases,
            contract_sha256=inputs.contract_sha256,
            evaluation_id="joint-iid-lora-trainable-embedding",
            dataset="final",
            method=f"joint_iid_lora_embedding_rank_{rank}",
            order=None,
            stage=ARRIVAL_COUNT,
            routing="forced_adapter",
            base_params=effective_params,
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
        _validate_joint_rows(ledger.rows, inputs, rank, allow_prefix=False)
        if progress is not None:
            progress(rank, len(ledger.rows), EXPECTED_STORY_COUNT, {})
        ledgers.append((rank, ledger.path))
    return tuple(ledgers)


def analyze_lora_embedding(
    inputs: LoraEmbeddingInputs,
    artifacts: Artifacts,
    ledgers: Ledgers,
    *,
    execution: Mapping[str, float],
    allocator: Mapping[str, object],
) -> dict[str, object]:
    """Aggregate direct controls, embedding diagnostics, and paired intervals."""
    artifact_by_rank = dict(artifacts)
    ledger_by_rank = dict(ledgers)
    if tuple(artifact_by_rank) != RANKS or tuple(ledger_by_rank) != RANKS:
        raise ValueError("embedding-LoRA analysis requires both fixed ranks")
    new_rows = {
        rank: ChainedJsonlLedger(ledger_by_rank[rank], EVALUATION_ROW_FORMAT).rows
        for rank in RANKS
    }
    for rank in RANKS:
        _validate_joint_rows(new_rows[rank], inputs, rank, allow_prefix=False)
    rows_by_condition: dict[str, Sequence[Mapping[str, object]]] = {
        "full_model": inputs.rank_sweep.full_model_rows,
        "lora_rank_8": inputs.rank_sweep.rank8_rows,
        "lora_rank_32": inputs.standard_rank32_rows,
        "lora_embedding_rank_8": new_rows[8],
        "lora_embedding_rank_32": new_rows[32],
    }
    reference_identities = _row_identities(inputs.rank_sweep.rank8_rows)
    reference_tokens = tuple(
        int(row["suffix_token_count"]) for row in inputs.rank_sweep.rank8_rows
    )
    if (
        any(_row_identities(rows) != reference_identities for rows in rows_by_condition.values())
        or any(
            tuple(int(row["suffix_token_count"]) for row in rows) != reference_tokens
            for rows in rows_by_condition.values()
        )
        or sum(reference_tokens) != EXPECTED_TOKEN_COUNT
    ):
        raise ValueError("embedding-LoRA conditions lost exact suffix alignment")

    labels = {
        "full_model": "Joint-IID full model",
        "lora_rank_8": "Projection LoRA rank 8",
        "lora_rank_32": "Projection LoRA rank 32",
        "lora_embedding_rank_8": "Projection LoRA + tied embedding rank 8",
        "lora_embedding_rank_32": "Projection LoRA + tied embedding rank 32",
    }
    ordered_conditions = tuple(labels)
    aggregate = tuple(
        _aggregate_rows(condition, labels[condition], rows_by_condition[condition])
        for condition in ordered_conditions
    )
    per_task = tuple(
        _aggregate_rows(
            condition,
            labels[condition],
            tuple(row for row in rows_by_condition[condition] if row["task_id"] == task_id),
            task_id=task_id,
        )
        for condition in ordered_conditions
        for task_id in TASK_IDS
    )
    embedding_only = tuple(
        _aggregate_candidate_zero(rank, new_rows[rank]) for rank in RANKS
    )
    training = tuple(
        _training_summary(inputs, rank, artifact_by_rank[rank]) for rank in RANKS
    )
    return {
        "aggregate": aggregate,
        "allocator": dict(allocator),
        "bootstrap": _paired_bootstrap(rows_by_condition),
        "comparability": {
            "batch_namespace_sha256": inputs.rank_sweep.canonical_rank8_job.identity_sha256,
            "exact_story_order": True,
            "exact_suffix_target_count": EXPECTED_TOKEN_COUNT,
            "exact_suffix_token_masks": True,
            "random_namespace_sha256": inputs.rank_sweep.canonical_rank8_job.identity_sha256,
        },
        "embedding_only": embedding_only,
        "execution": dict(execution),
        "ledger_provenance": tuple(
            {
                "path": path.relative_to(inputs.parent.repository_root).as_posix(),
                "rank": rank,
                "row_count": EXPECTED_STORY_COUNT,
                "sha256": file_sha256(path),
            }
            for rank, path in ledgers
        ),
        "per_task": per_task,
        "provenance": {
            "base_parameter_checksum": inputs.parent.selected_base.reference.parameter_checksum,
            "contract_sha256": inputs.contract_sha256,
            "full_model_job_sha256": inputs.rank_sweep.canonical_full_model_job.identity_sha256,
            "parent_rank_sweep_contract_sha256": inputs.rank_sweep.contract_sha256,
            "parent_rank_sweep_manifest_sha256": PARENT_RANK_SWEEP_MANIFEST_SHA256,
            "rank8_job_sha256": inputs.rank_sweep.canonical_rank8_job.identity_sha256,
        },
        "training": training,
    }


def assert_lora_embedding_inputs_unchanged(inputs: LoraEmbeddingInputs) -> None:
    """Reject changes to canonical temporal or bound rank-sweep evidence."""
    assert_canonical_artifacts_unchanged(inputs.parent)
    paths = tuple(
        inputs.parent.repository_root / relative
        for relative, _ in inputs.protected_files
    )
    after = _file_snapshot(inputs.parent.repository_root, paths)
    if after != inputs.protected_files:
        before_map, after_map = dict(inputs.protected_files), dict(after)
        changed = tuple(
            relative
            for relative in sorted(set(before_map) | set(after_map))
            if before_map.get(relative) != after_map.get(relative)
        )
        raise RuntimeError(f"embedding-LoRA bound parent files changed: {changed}")


def _validate_joint_rows(
    rows: Sequence[dict[str, object]],
    inputs: LoraEmbeddingInputs,
    rank: int,
    *,
    allow_prefix: bool,
) -> None:
    validate_evaluation_rows(rows, inputs.contract_sha256)
    expected_order = _row_identities(inputs.rank_sweep.rank8_rows)
    candidate_id = f"joint-iid-lora-embedding-rank-{rank}"
    if (
        len(rows) > EXPECTED_STORY_COUNT
        or (not allow_prefix and len(rows) != EXPECTED_STORY_COUNT)
        or _row_identities(rows) != expected_order[: len(rows)]
        or any(
            row["evaluation_id"] != "joint-iid-lora-trainable-embedding"
            or row["dataset"] != "final"
            or row["method"] != f"joint_iid_lora_embedding_rank_{rank}"
            or row["order"] is not None
            or row["stage"] != ARRIVAL_COUNT
            or row["candidate_ids"] != ["base", candidate_id]
            or row["selected_index"] != 1
            for row in rows
        )
    ):
        raise ValueError(f"embedding-LoRA rank-{rank} ledger is not canonical")


def _aggregate_rows(
    condition: str,
    label: str,
    rows: Sequence[Mapping[str, object]],
    *,
    task_id: str | None = None,
) -> dict[str, object]:
    if not rows:
        raise ValueError("cannot aggregate empty embedding-LoRA rows")
    token_count = sum(int(row["suffix_token_count"]) for row in rows)
    total_nll = sum(float(row["suffix_total_nll"]) for row in rows)
    correct = sum(int(row["suffix_correct_tokens"]) for row in rows)
    rank = 32 if condition.endswith("32") else 8 if condition.endswith("8") else None
    return {
        "condition": condition,
        "label": label,
        "rank": rank,
        "story_count": len(rows),
        "story_mean_nll": float(
            np.mean([float(row["suffix_mean_nll"]) for row in rows])
        ),
        "suffix_token_accuracy": correct / token_count,
        "task_id": task_id,
        "token_count": token_count,
        "token_mean_nll": total_nll / token_count,
    }


def _aggregate_candidate_zero(
    rank: int,
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    token_counts = np.asarray(
        [int(row["suffix_token_count"]) for row in rows],
        dtype=np.float64,
    )
    means = np.asarray(
        [float(row["suffix_mean_nll_by_candidate"][0]) for row in rows],
        dtype=np.float64,
    )
    return {
        "condition": f"trained_embedding_without_lora_rank_{rank}",
        "label": f"Rank-{rank} jointly trained embedding without its LoRA",
        "rank": rank,
        "story_count": len(rows),
        "story_mean_nll": float(np.mean(means)),
        "token_count": int(np.sum(token_counts)),
        "token_mean_nll": float(np.sum(means * token_counts) / np.sum(token_counts)),
    }


def _paired_bootstrap(
    rows_by_condition: Mapping[str, Sequence[Mapping[str, object]]],
) -> tuple[dict[str, object], ...]:
    conditions = tuple(rows_by_condition)
    ordered = tuple(rows_by_condition[condition] for condition in conditions)
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
        raise ValueError("embedding-LoRA bootstrap lost a noun stratum")
    rng = np.random.default_rng(SEED)
    story_samples = np.empty(
        (BOOTSTRAP_REPETITIONS, len(conditions)),
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
            total_nll[:, sampled], axis=1
        ) / np.sum(token_count[:, sampled], axis=1)
    positions = {condition: index for index, condition in enumerate(conditions)}
    comparisons = (
        ("lora_embedding_rank_8", "lora_rank_8"),
        ("lora_embedding_rank_32", "lora_rank_32"),
        ("lora_embedding_rank_8", "full_model"),
        ("lora_embedding_rank_32", "full_model"),
        ("lora_embedding_rank_32", "lora_embedding_rank_8"),
    )
    output: list[dict[str, object]] = []
    for condition, reference in comparisons:
        condition_index = positions[condition]
        reference_index = positions[reference]
        for metric, samples in (
            ("story_mean_nll", story_samples),
            ("token_mean_nll", token_samples),
        ):
            differences = samples[:, condition_index] - samples[:, reference_index]
            estimate = (
                float(np.mean(story_nll[condition_index]) - np.mean(story_nll[reference_index]))
                if metric == "story_mean_nll"
                else float(
                    np.sum(total_nll[condition_index]) / np.sum(token_count[condition_index])
                    - np.sum(total_nll[reference_index]) / np.sum(token_count[reference_index])
                )
            )
            lower, upper = np.quantile(differences, (0.025, 0.975))
            output.append(
                {
                    "condition": condition,
                    "estimate": estimate,
                    "lower_95": float(lower),
                    "metric": metric,
                    "reference": reference,
                    "upper_95": float(upper),
                }
            )
    return tuple(output)


def _training_summary(
    inputs: LoraEmbeddingInputs,
    rank: int,
    artifact: LoraEmbeddingArtifact,
) -> dict[str, object]:
    adapter_parameters = sum(
        int(np.prod(np.asarray(value).shape, dtype=np.int64))
        for value in jax.tree_util.tree_leaves(artifact.trainable.adapter)
    )
    embedding_parameters = int(np.prod(artifact.trainable.token_embedding.shape))
    base_parameters = sum(
        int(np.prod(np.asarray(value).shape, dtype=np.int64))
        for value in jax.tree_util.tree_leaves(inputs.parent.loaded_base.params)
    )
    base_embedding = np.asarray(
        inputs.parent.loaded_base.params.token_embedding,
        dtype=np.float64,
    )
    learned_embedding = np.asarray(
        artifact.trainable.token_embedding,
        dtype=np.float64,
    )
    delta = learned_embedding - base_embedding
    return {
        "adapter_parameter_count": adapter_parameters,
        "allocator_peak_bytes": artifact.allocator_peak_bytes,
        "alpha": float(rank),
        "embedding_delta_max_abs": float(np.max(np.abs(delta))),
        "embedding_delta_mean_abs": float(np.mean(np.abs(delta))),
        "embedding_delta_relative_frobenius": float(
            np.linalg.norm(delta) / np.linalg.norm(base_embedding)
        ),
        "embedding_delta_rms": float(np.sqrt(np.mean(np.square(delta)))),
        "embedding_learning_rate": FULL_MODEL_LEARNING_RATE,
        "embedding_parameter_count": embedding_parameters,
        "final_training_loss": _last_loss(artifact.directory / "losses.jsonl"),
        "job_sha256": artifact.job.identity_sha256,
        "lora_learning_rate": LORA_LEARNING_RATE,
        "optimizer_updates": artifact.optimizer_updates,
        "rank": rank,
        "runtime_seconds": artifact.runtime_seconds,
        "tensor_file_bytes": (artifact.directory / "trainable.safetensors").stat().st_size,
        "tensor_file_sha256": artifact.tensor_file_sha256,
        "total_trainable_parameter_count": adapter_parameters + embedding_parameters,
        "trainable_to_base_parameter_fraction": (
            adapter_parameters + embedding_parameters
        ) / base_parameters,
    }


def _row_identities(
    rows: Sequence[Mapping[str, object]],
) -> tuple[tuple[str, str], ...]:
    return tuple((str(row["task_id"]), str(row["story_id"])) for row in rows)


def _last_loss(path: Path) -> float:
    rows = ChainedJsonlLedger(path, TRAINING_ROW_FORMAT).rows
    if not rows:
        raise ValueError("embedding-LoRA training trace is empty")
    value = float(rows[-1]["loss"])
    if not math.isfinite(value):
        raise ValueError("embedding-LoRA final training loss is not finite")
    return value


def _file_snapshot(
    root: Path,
    paths: Sequence[Path],
) -> tuple[tuple[str, str], ...]:
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("embedding-LoRA protected file set is incomplete")
    return tuple(
        (path.relative_to(root).as_posix(), file_sha256(path))
        for path in sorted(paths)
    )


__all__ = [
    "ALLOCATOR_LIMIT_BYTES",
    "ARTIFACT_FORMAT",
    "Artifacts",
    "CONTRACT_FORMAT",
    "LoraEmbeddingArtifact",
    "LoraEmbeddingInputs",
    "LoraEmbeddingJob",
    "LoraEmbeddingTrainable",
    "Ledgers",
    "RANKS",
    "REPORT_FORMAT",
    "analyze_lora_embedding",
    "assert_lora_embedding_inputs_unchanged",
    "authenticate_lora_embedding_inputs",
    "embedding_lora_job",
    "load_lora_embedding_artifact",
    "lora_embedding_checksum",
    "run_or_resume_lora_embedding_evaluation",
    "run_or_resume_lora_embedding_training",
    "train_or_load_lora_embedding",
]

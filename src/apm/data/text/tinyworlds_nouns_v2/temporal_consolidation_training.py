"""Finite-epoch, exact-resume training for temporal LoRA and IID controls."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
import math
import os
from pathlib import Path
import shutil
import tempfile
from time import monotonic
from typing import Literal, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from apm.continual.language_adaptation_artifact import (
    flatten_lora_edge,
    read_safetensors_archive,
    unflatten_lora_edge,
    write_safetensors_archive,
)
from apm.data.text.tinyworlds_nouns_v1.experiment import (
    IndexedStoryStore,
    NounSelectedBase,
    StoryIndexEntry,
)
from apm.data.text.tinyworlds_nouns_v2.contracts import (
    canonical_json_bytes,
    record_sha256,
    require_sha256,
)
from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_contracts import (
    ADAPTER_FORMAT,
    ALLOCATOR_LIMIT_BYTES,
    CHECKPOINT_SECONDS,
    CHECKPOINT_UPDATE_INTERVAL,
    FIXED_EPOCHS,
    FULL_MODEL_FORMAT,
    FULL_MODEL_LEARNING_RATE,
    GRADIENT_CLIP_NORM,
    LORA_ALPHA,
    LORA_LEARNING_RATE,
    LORA_RANK,
    PHYSICAL_BATCH_SIZE,
    SEED,
    STUDY_ID,
    TRAINING_ROW_FORMAT,
    WEIGHT_DECAY,
)
from apm.continual.artifacts import (
    ChainedJsonlLedger,
    atomic_write,
    file_sha256,
    load_canonical_json,
)
from apm.lm.checkpoint import (
    BaseCheckpointRef,
    LoadedGptNeoCheckpoint,
    load_gpt_neo_checkpoint,
    save_gpt_neo_checkpoint,
)
from apm.lm.config import GptNeoConfig
from apm.lm.lora import LoraConfig, LoraEdge, init_lora_edge
from apm.lm.lora_memory import PackedLoraMemory, pack_lora_memory
from apm.lm.parameters import GptNeoParams
from apm.lm.text_data import TokenBatch
from apm.lm.training import (
    LmTrainConfig,
    LmTrainState,
    base_train_step,
    candidate_lora_train_step,
    init_base_train_state,
    init_candidate_lora_train_state,
)
from apm.lm.training_state_artifact import (
    load_lm_train_state_artifact,
    write_lm_train_state_artifact,
)
from apm.memory.graph import NodeId, init_memory_graph
from apm.data.text.tinyworlds_p.training import allocator_peak_bytes


TrainingFamily: TypeAlias = Literal[
    "level_zero",
    "merge",
    "sequential",
    "independent_noun",
    "joint_iid_lora",
    "joint_iid_full_model",
]
TrainingProgress = Callable[[str, int, int, float, float], None]
LoraCompiledStep: TypeAlias = Callable[
    [LmTrainState[LoraEdge], TokenBatch, GptNeoParams, PackedLoraMemory],
    tuple[LmTrainState[LoraEdge], jax.Array],
]
FullModelCompiledStep: TypeAlias = Callable[
    [LmTrainState[GptNeoParams], TokenBatch],
    tuple[LmTrainState[GptNeoParams], jax.Array],
]


@dataclass(frozen=True, slots=True)
class TrainingJob:
    """One immutable finite-epoch optimizer job bound to exact source stories."""

    contract_sha256: str
    job_id: str
    family: TrainingFamily
    source_story_ids: tuple[str, ...]
    source_shard_ids: tuple[str, ...]
    lineage_ids: tuple[str, ...] = ()
    order: str | None = None
    level: int | None = None
    start_arrival: int | None = None
    end_arrival: int | None = None
    initial_adapter_sha256: str | None = None
    lora_rank: int | None = None
    lora_alpha: float | None = None
    batch_namespace_sha256: str | None = None
    random_namespace_sha256: str | None = None

    def __post_init__(self) -> None:
        require_sha256(self.contract_sha256, "temporal training contract")
        if not self.job_id or self.family not in (
            "level_zero",
            "merge",
            "sequential",
            "independent_noun",
            "joint_iid_lora",
            "joint_iid_full_model",
        ):
            raise ValueError("temporal training job identity is invalid")
        if not self.source_story_ids or len(set(self.source_story_ids)) != len(
            self.source_story_ids
        ):
            raise ValueError("temporal training jobs require unique source stories")
        for digest in (
            *self.source_story_ids,
            *self.source_shard_ids,
            *self.lineage_ids,
        ):
            require_sha256(digest, "temporal training source or lineage")
        if self.initial_adapter_sha256 is not None:
            require_sha256(self.initial_adapter_sha256, "initial adapter")
        if (self.lora_rank is None) != (self.lora_alpha is None):
            raise ValueError("LoRA rank and alpha overrides must be declared together")
        if self.lora_rank is not None and (
            self.lora_rank <= 0
            or self.lora_alpha is None
            or not math.isfinite(self.lora_alpha)
            or self.lora_alpha <= 0.0
        ):
            raise ValueError("training job LoRA configuration is invalid")
        for namespace in (
            self.batch_namespace_sha256,
            self.random_namespace_sha256,
        ):
            if namespace is not None:
                require_sha256(namespace, "training namespace")
        interval_values = (self.level, self.start_arrival, self.end_arrival)
        if any(value is not None for value in interval_values) and any(
            type(value) is not int or value < 0 for value in interval_values
        ):
            raise ValueError("temporal training interval metadata is incomplete")

    @property
    def identity_sha256(self) -> str:
        """Return the independent content identity of this optimizer job."""
        return record_sha256(self.as_record())

    def as_record(self) -> dict[str, object]:
        """Return the canonical job payload without its derived hash."""
        record = {
            "contract_sha256": self.contract_sha256,
            "end_arrival": self.end_arrival,
            "epochs": FIXED_EPOCHS,
            "family": self.family,
            "format": f"{STUDY_ID}-training-job-v1",
            "initial_adapter_sha256": self.initial_adapter_sha256,
            "job_id": self.job_id,
            "level": self.level,
            "lineage_ids": list(self.lineage_ids),
            "order": self.order,
            "source_shard_ids": list(self.source_shard_ids),
            "source_story_ids_sha256": record_sha256(list(self.source_story_ids)),
            "source_story_count": len(self.source_story_ids),
            "start_arrival": self.start_arrival,
        }
        if self.lora_rank is not None:
            record.update(
                {
                    "batch_namespace_sha256": self.batch_namespace_sha256,
                    "lora_alpha": self.lora_alpha,
                    "lora_rank": self.lora_rank,
                    "random_namespace_sha256": self.random_namespace_sha256,
                }
            )
        return record


@dataclass(frozen=True, slots=True)
class AdapterArtifact:
    """Strict-loaded standalone LoRA and its immutable study manifest."""

    directory: Path
    job: TrainingJob
    adapter: LoraEdge
    adapter_sha256: str
    tensor_file_sha256: str
    loss_trace_sha256: str
    optimizer_updates: int
    runtime_seconds: float
    allocator_peak_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "directory", Path(self.directory))
        for digest in (
            self.adapter_sha256,
            self.tensor_file_sha256,
            self.loss_trace_sha256,
        ):
            require_sha256(digest, "temporal adapter artifact")
        if (
            self.optimizer_updates <= 0
            or not math.isfinite(self.runtime_seconds)
            or self.runtime_seconds < 0.0
            or self.allocator_peak_bytes < 0
        ):
            raise ValueError("temporal adapter measurements are invalid")


@dataclass(frozen=True, slots=True)
class FullModelArtifact:
    """Strict-loaded joint-IID full model and its study identity."""

    directory: Path
    job: TrainingJob
    checkpoint: BaseCheckpointRef
    loss_trace_sha256: str
    optimizer_updates: int
    runtime_seconds: float
    allocator_peak_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "directory", Path(self.directory))
        require_sha256(self.loss_trace_sha256, "full-model loss trace")
        if self.job.family != "joint_iid_full_model" or self.optimizer_updates <= 0:
            raise ValueError("full-model artifact has the wrong training family")


class TrainingInterrupted(RuntimeError):
    """Intentional test/smoke interruption after an exact state checkpoint."""


class StoryEpochBatches(Sequence[TokenBatch]):
    """Four deterministic full-coverage epochs over arbitrary story pointers."""

    def __init__(
        self,
        partition: object,
        entries: Sequence[StoryIndexEntry],
        *,
        context_length: int,
        batch_size: int,
        namespace: str,
        epochs: int = FIXED_EPOCHS,
    ) -> None:
        if not entries or not namespace or epochs <= 0:
            raise ValueError("story epoch batches require data, namespace, and epochs")
        if context_length <= 0 or batch_size <= 0:
            raise ValueError("story epoch batch dimensions must be positive")
        self._partition = partition
        self._context_length = context_length
        self._batch_size = batch_size
        self._store = IndexedStoryStore(partition)  # type: ignore[arg-type]
        partition_sha256 = str(getattr(partition, "partition_sha256"))
        benchmark_id = str(getattr(partition, "benchmark_id", STUDY_ID))
        self._epoch_entries = tuple(
            tuple(
                sorted(
                    entries,
                    key=lambda entry: (
                        sha256(
                            f"{benchmark_id}\0{namespace}\0epoch-{epoch}\0"
                            f"{partition_sha256}\0{entry.story_id}".encode("utf-8")
                        ).hexdigest(),
                        entry.story_id,
                    ),
                )
            )
            for epoch in range(epochs)
        )
        window_counts = {
            entry.story_id: math.ceil((entry.token_count - 1) / context_length)
            for entry in entries
        }
        self._window_count = sum(window_counts.values())
        self._batches_per_epoch = math.ceil(self._window_count / batch_size)
        self._window_stops = tuple(
            tuple(
                np.cumsum(
                    tuple(window_counts[entry.story_id] for entry in epoch_entries),
                    dtype=np.int64,
                ).tolist()
            )
            for epoch_entries in self._epoch_entries
        )

    def __len__(self) -> int:
        return len(self._epoch_entries) * self._batches_per_epoch

    def __getitem__(self, index: int | slice) -> TokenBatch | tuple[TokenBatch, ...]:
        if isinstance(index, slice):
            return tuple(self[position] for position in range(*index.indices(len(self))))
        position = index + len(self) if index < 0 else index
        if not 0 <= position < len(self):
            raise IndexError(index)
        epoch, batch = divmod(position, self._batches_per_epoch)
        entries = self._epoch_entries[epoch]
        stops = self._window_stops[epoch]
        shape = (self._batch_size, self._context_length)
        pad_token_id = int(getattr(self._partition, "pad_token_id"))
        input_ids = np.full(shape, pad_token_id, dtype=np.int32)
        target_ids = np.full(shape, pad_token_id, dtype=np.int32)
        attention = np.zeros(shape, dtype=np.bool_)
        losses = np.zeros(shape, dtype=np.bool_)
        first_window = batch * self._batch_size
        stop_window = min(first_window + self._batch_size, self._window_count)
        for row, global_window in enumerate(range(first_window, stop_window)):
            story_position = bisect_right(stops, global_window)
            prior_stop = 0 if story_position == 0 else stops[story_position - 1]
            entry = entries[story_position]
            local_window = global_window - prior_stop
            tokens = self._store.tokens(entry)
            token_start = local_window * self._context_length
            chunk = tokens[token_start : token_start + self._context_length + 1]
            transitions = len(chunk) - 1
            input_ids[row, :transitions] = chunk[:-1]
            target_ids[row, :transitions] = chunk[1:]
            attention[row, :transitions] = True
            losses[row, :transitions] = True
        return TokenBatch(input_ids, attention, target_ids, losses)

    @property
    def story_count(self) -> int:
        """Return unique source stories per epoch."""
        return len(self._epoch_entries[0])

    @property
    def window_count_per_epoch(self) -> int:
        """Return causal windows visited exactly once in each epoch."""
        return self._window_count

    @property
    def epochs(self) -> int:
        """Return the number of finite epoch permutations."""
        return len(self._epoch_entries)


def _compiled_lora_train_step(
    model_config: GptNeoConfig,
    lora_config: LoraConfig | None = None,
) -> LoraCompiledStep:
    """Cache one fixed-setting LoRA executable per architecture and rank."""
    return _compiled_lora_train_step_for_config(
        model_config,
        lora_config or LoraConfig(rank=LORA_RANK, alpha=LORA_ALPHA),
    )


@lru_cache(maxsize=None)
def _compiled_lora_train_step_for_config(
    model_config: GptNeoConfig,
    lora_config: LoraConfig,
) -> LoraCompiledStep:
    step_config = LmTrainConfig(
        learning_rate=LORA_LEARNING_RATE,
        steps=1,
        batch_size=PHYSICAL_BATCH_SIZE,
        weight_decay=WEIGHT_DECAY,
        gradient_clip_norm=GRADIENT_CLIP_NORM,
    )
    parent_coefficients = jnp.zeros((1,), dtype=jnp.float32)

    @jax.jit
    def compiled(
        current: LmTrainState[LoraEdge],
        batch: TokenBatch,
        base_params: GptNeoParams,
        empty_memory: PackedLoraMemory,
    ) -> tuple[LmTrainState[LoraEdge], jax.Array]:
        return candidate_lora_train_step(
            current,
            batch,
            base_params,
            model_config,
            empty_memory,
            lora_config,
            parent_coefficients,
            0,
            step_config,
        )

    return compiled


@lru_cache(maxsize=None)
def _compiled_full_model_train_step(
    model_config: GptNeoConfig,
) -> FullModelCompiledStep:
    """Cache one fixed-setting full-model executable per model architecture."""
    step_config = LmTrainConfig(
        learning_rate=FULL_MODEL_LEARNING_RATE,
        steps=1,
        batch_size=PHYSICAL_BATCH_SIZE,
        weight_decay=WEIGHT_DECAY,
        gradient_clip_norm=GRADIENT_CLIP_NORM,
    )

    @jax.jit
    def compiled(
        current: LmTrainState[GptNeoParams],
        batch: TokenBatch,
    ) -> tuple[LmTrainState[GptNeoParams], jax.Array]:
        return base_train_step(current, batch, model_config, step_config)

    return compiled


def train_or_load_lora(
    job: TrainingJob,
    batches: Sequence[TokenBatch],
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    output_directory: str | Path,
    work_directory: str | Path,
    *,
    initial_adapter: LoraEdge | None = None,
    lora_config: LoraConfig | None = None,
    progress: TrainingProgress | None = None,
    stop_after_update: int | None = None,
) -> AdapterArtifact:
    """Train, exactly resume, or strict-load one standalone base-relative LoRA."""
    if job.family == "joint_iid_full_model":
        raise ValueError("full-model jobs require train_or_load_full_model")
    if not batches:
        raise ValueError("LoRA training requires finite epoch batches")
    resolved_lora_config = lora_config or LoraConfig(
        rank=LORA_RANK,
        alpha=LORA_ALPHA,
    )
    declared_lora_config = (
        None
        if job.lora_rank is None
        else LoraConfig(rank=job.lora_rank, alpha=float(job.lora_alpha))
    )
    if (
        declared_lora_config is not None
        and declared_lora_config != resolved_lora_config
    ) or (
        declared_lora_config is None
        and resolved_lora_config != LoraConfig(rank=LORA_RANK, alpha=LORA_ALPHA)
    ):
        raise ValueError("LoRA configuration is not bound to the training job")
    target = Path(output_directory) / job.identity_sha256
    if target.is_dir():
        return load_adapter_artifact(
            target,
            job,
            model_config,
            resolved_lora_config,
        )
    work = Path(work_directory) / job.identity_sha256
    work.mkdir(parents=True, exist_ok=True)
    loss_ledger = ChainedJsonlLedger(work / "losses.jsonl", TRAINING_ROW_FORMAT)
    train_config = LmTrainConfig(
        learning_rate=LORA_LEARNING_RATE,
        steps=len(batches),
        batch_size=PHYSICAL_BATCH_SIZE,
        weight_decay=WEIGHT_DECAY,
        gradient_clip_norm=GRADIENT_CLIP_NORM,
    )
    random_namespace = job.random_namespace_sha256 or job.identity_sha256
    initialization_key, training_key = jax.random.split(_job_key(random_namespace))
    starting_adapter = initial_adapter or init_lora_edge(
        initialization_key,
        model_config,
        resolved_lora_config,
    )
    if job.initial_adapter_sha256 is not None and (
        adapter_checksum(starting_adapter, model_config, resolved_lora_config)
        != job.initial_adapter_sha256
    ):
        raise ValueError("sequential job initial adapter checksum changed")
    template = init_candidate_lora_train_state(
        starting_adapter,
        training_key,
        train_config,
    )
    state = _latest_state(work / "states", job.identity_sha256, template)
    start_update = int(state.step)
    if len(loss_ledger.rows) < start_update:
        raise ValueError("training loss ledger is behind its optimizer state")
    loss_ledger.truncate(start_update)
    empty_memory = pack_lora_memory(
        init_memory_graph(NodeId("root")),
        model_config,
        resolved_lora_config,
        max_nodes=2,
        max_edges=1,
    )
    compiled_step = _compiled_lora_train_step(model_config, resolved_lora_config)
    prior_elapsed = _load_elapsed(work / "runtime.json", job.identity_sha256)
    started = monotonic()
    last_checkpoint = monotonic()
    current = state
    for update in range(start_update, train_config.steps):
        current, loss = compiled_step(
            current,
            batches[update],
            base_params,
            empty_memory,
        )
        scalar_loss = float(jax.block_until_ready(loss))
        elapsed = prior_elapsed + monotonic() - started
        loss_ledger.append(
            {
                "job_sha256": job.identity_sha256,
                "loss": scalar_loss,
                "update": update + 1,
            }
        )
        if progress is not None:
            progress(job.job_id, update + 1, train_config.steps, scalar_loss, elapsed)
        checkpoint_due = (
            (update + 1) % CHECKPOINT_UPDATE_INTERVAL == 0
            or monotonic() - last_checkpoint >= CHECKPOINT_SECONDS
            or update + 1 == train_config.steps
            or stop_after_update == update + 1
        )
        if checkpoint_due:
            _write_state_if_missing(
                work / "states" / f"state-{update + 1:08d}",
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
                f"intentional interruption of {job.job_id} at update {update + 1}"
            )
    jax.tree_util.tree_map(jax.block_until_ready, current)
    peak = allocator_peak_bytes()
    if peak > ALLOCATOR_LIMIT_BYTES:
        raise RuntimeError(
            f"temporal LoRA allocator peak {peak} exceeds {ALLOCATOR_LIMIT_BYTES}"
        )
    return _publish_adapter_artifact(
        target,
        job,
        current.trainable,
        model_config,
        resolved_lora_config,
        work / "losses.jsonl",
        train_config.steps,
        _load_elapsed(work / "runtime.json", job.identity_sha256),
        peak,
    )


def train_or_load_full_model(
    job: TrainingJob,
    batches: Sequence[TokenBatch],
    selected_base: NounSelectedBase,
    output_directory: str | Path,
    work_directory: str | Path,
    *,
    progress: TrainingProgress | None = None,
    stop_after_update: int | None = None,
) -> FullModelArtifact:
    """Train, exactly resume, or strict-load the four-epoch joint-IID full model."""
    if job.family != "joint_iid_full_model" or not batches:
        raise ValueError("full-model training requires its fixed IID job and batches")
    target = Path(output_directory) / job.identity_sha256
    if target.is_dir():
        return load_full_model_artifact(target, job)
    loaded = load_gpt_neo_checkpoint(selected_base.reference)
    train_config = LmTrainConfig(
        learning_rate=FULL_MODEL_LEARNING_RATE,
        steps=len(batches),
        batch_size=PHYSICAL_BATCH_SIZE,
        weight_decay=WEIGHT_DECAY,
        gradient_clip_norm=GRADIENT_CLIP_NORM,
    )
    work = Path(work_directory) / job.identity_sha256
    work.mkdir(parents=True, exist_ok=True)
    loss_ledger = ChainedJsonlLedger(work / "losses.jsonl", TRAINING_ROW_FORMAT)
    template = init_base_train_state(
        loaded.params,
        _job_key(job.identity_sha256),
        train_config,
    )
    state = _latest_state(work / "states", job.identity_sha256, template)
    start_update = int(state.step)
    if len(loss_ledger.rows) < start_update:
        raise ValueError("full-model loss ledger is behind its optimizer state")
    loss_ledger.truncate(start_update)
    compiled_step = _compiled_full_model_train_step(loaded.config)
    prior_elapsed = _load_elapsed(work / "runtime.json", job.identity_sha256)
    started = monotonic()
    last_checkpoint = monotonic()
    current = state
    for update in range(start_update, train_config.steps):
        current, loss = compiled_step(current, batches[update])
        scalar_loss = float(jax.block_until_ready(loss))
        elapsed = prior_elapsed + monotonic() - started
        loss_ledger.append(
            {
                "job_sha256": job.identity_sha256,
                "loss": scalar_loss,
                "update": update + 1,
            }
        )
        if progress is not None:
            progress(job.job_id, update + 1, train_config.steps, scalar_loss, elapsed)
        checkpoint_due = (
            (update + 1) % CHECKPOINT_UPDATE_INTERVAL == 0
            or monotonic() - last_checkpoint >= CHECKPOINT_SECONDS
            or update + 1 == train_config.steps
            or stop_after_update == update + 1
        )
        if checkpoint_due:
            _write_state_if_missing(
                work / "states" / f"state-{update + 1:08d}",
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
                f"intentional interruption of {job.job_id} at update {update + 1}"
            )
    jax.tree_util.tree_map(jax.block_until_ready, current)
    peak = allocator_peak_bytes()
    if peak > ALLOCATOR_LIMIT_BYTES:
        raise RuntimeError(
            f"temporal full-model allocator peak {peak} exceeds {ALLOCATOR_LIMIT_BYTES}"
        )
    return _publish_full_model_artifact(
        target,
        job,
        current.trainable,
        loaded,
        work / "losses.jsonl",
        train_config.steps,
        _load_elapsed(work / "runtime.json", job.identity_sha256),
        peak,
    )


def adapter_checksum(
    adapter: LoraEdge,
    model_config: GptNeoConfig,
    lora_config: LoraConfig,
) -> str:
    """Hash stable LoRA tensor names, shapes, dtypes, and values."""
    digest = sha256()
    for name, tensor in sorted(
        flatten_lora_edge(adapter, model_config, lora_config).items()
    ):
        value = np.ascontiguousarray(np.asarray(tensor, dtype=np.float32))
        digest.update(name.encode("utf-8"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def load_adapter_artifact(
    directory: str | Path,
    job: TrainingJob,
    model_config: GptNeoConfig,
    lora_config: LoraConfig,
) -> AdapterArtifact:
    """Strict-load one standalone temporal LoRA artifact."""
    root = Path(directory)
    if {path.name for path in root.iterdir()} != {
        "adapter.safetensors",
        "losses.jsonl",
        "manifest.json",
    }:
        raise ValueError("temporal adapter artifact entries changed")
    manifest = _validated_study_manifest(root / "manifest.json", ADAPTER_FORMAT, job)
    tensor_path = root / "adapter.safetensors"
    loss_path = root / "losses.jsonl"
    if (
        manifest.get("tensor_file_sha256") != file_sha256(tensor_path)
        or manifest.get("loss_trace_sha256") != file_sha256(loss_path)
    ):
        raise ValueError("temporal adapter file hashes changed")
    tensors, metadata = read_safetensors_archive(tensor_path)
    expected_metadata = {
        "adapter_sha256": str(manifest["adapter_sha256"]),
        "format": ADAPTER_FORMAT,
        "job_sha256": job.identity_sha256,
    }
    if metadata != expected_metadata:
        raise ValueError("temporal adapter safetensors metadata changed")
    adapter = unflatten_lora_edge(tensors, model_config, lora_config)
    if adapter_checksum(adapter, model_config, lora_config) != manifest["adapter_sha256"]:
        raise ValueError("temporal adapter tensor checksum changed")
    ledger = ChainedJsonlLedger(loss_path, TRAINING_ROW_FORMAT)
    updates = int(manifest["optimizer_updates"])
    if len(ledger.rows) != updates or int(ledger.rows[-1]["update"]) != updates:
        raise ValueError("temporal adapter loss trace coverage changed")
    return AdapterArtifact(
        root,
        job,
        adapter,
        str(manifest["adapter_sha256"]),
        str(manifest["tensor_file_sha256"]),
        str(manifest["loss_trace_sha256"]),
        updates,
        float(manifest["runtime_seconds"]),
        int(manifest["allocator_peak_bytes"]),
    )


def load_full_model_artifact(
    directory: str | Path,
    job: TrainingJob,
) -> FullModelArtifact:
    """Strict-load the joint-IID full-model study artifact."""
    root = Path(directory)
    if {path.name for path in root.iterdir()} != {
        "checkpoint",
        "losses.jsonl",
        "manifest.json",
    }:
        raise ValueError("temporal full-model artifact entries changed")
    manifest = _validated_study_manifest(root / "manifest.json", FULL_MODEL_FORMAT, job)
    loss_path = root / "losses.jsonl"
    if manifest.get("loss_trace_sha256") != file_sha256(loss_path):
        raise ValueError("temporal full-model loss trace changed")
    loaded = load_gpt_neo_checkpoint(root / "checkpoint")
    if (
        loaded.reference.manifest_sha256 != manifest.get("checkpoint_manifest_sha256")
        or loaded.reference.parameter_checksum != manifest.get("parameter_checksum")
    ):
        raise ValueError("temporal full-model checkpoint identity changed")
    ledger = ChainedJsonlLedger(loss_path, TRAINING_ROW_FORMAT)
    updates = int(manifest["optimizer_updates"])
    if len(ledger.rows) != updates:
        raise ValueError("temporal full-model loss coverage changed")
    return FullModelArtifact(
        root,
        job,
        loaded.reference,
        str(manifest["loss_trace_sha256"]),
        updates,
        float(manifest["runtime_seconds"]),
        int(manifest["allocator_peak_bytes"]),
    )


def _publish_adapter_artifact(
    target: Path,
    job: TrainingJob,
    adapter: LoraEdge,
    model_config: GptNeoConfig,
    lora_config: LoraConfig,
    loss_path: Path,
    updates: int,
    runtime: float,
    peak: int,
) -> AdapterArtifact:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    try:
        checksum = adapter_checksum(adapter, model_config, lora_config)
        tensors = {
            name: np.asarray(value, dtype=np.float32)
            for name, value in flatten_lora_edge(
                adapter,
                model_config,
                lora_config,
            ).items()
        }
        tensor_path = temporary / "adapter.safetensors"
        write_safetensors_archive(
            tensor_path,
            dict(sorted(tensors.items())),
            {
                "adapter_sha256": checksum,
                "format": ADAPTER_FORMAT,
                "job_sha256": job.identity_sha256,
            },
        )
        shutil.copyfile(loss_path, temporary / "losses.jsonl")
        core = {
            "adapter_sha256": checksum,
            "allocator_peak_bytes": peak,
            "format": ADAPTER_FORMAT,
            "job": job.as_record(),
            "job_sha256": job.identity_sha256,
            "loss_trace_sha256": file_sha256(temporary / "losses.jsonl"),
            "optimizer_updates": updates,
            "runtime_seconds": runtime,
            "tensor_file": "adapter.safetensors",
            "tensor_file_sha256": file_sha256(tensor_path),
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
    return load_adapter_artifact(target, job, model_config, lora_config)


def _publish_full_model_artifact(
    target: Path,
    job: TrainingJob,
    params: GptNeoParams,
    loaded_base: LoadedGptNeoCheckpoint,
    loss_path: Path,
    updates: int,
    runtime: float,
    peak: int,
) -> FullModelArtifact:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    try:
        reference = save_gpt_neo_checkpoint(
            temporary / "checkpoint",
            params,
            loaded_base.config,
            tokenizer=loaded_base.tokenizer,
            source=loaded_base.source,
        )
        shutil.copyfile(loss_path, temporary / "losses.jsonl")
        core = {
            "allocator_peak_bytes": peak,
            "checkpoint_manifest_sha256": reference.manifest_sha256,
            "format": FULL_MODEL_FORMAT,
            "job": job.as_record(),
            "job_sha256": job.identity_sha256,
            "loss_trace_sha256": file_sha256(temporary / "losses.jsonl"),
            "optimizer_updates": updates,
            "parameter_checksum": reference.parameter_checksum,
            "runtime_seconds": runtime,
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
    return load_full_model_artifact(target, job)


def _validated_study_manifest(
    path: Path,
    expected_format: str,
    job: TrainingJob,
) -> dict[str, object]:
    record = load_canonical_json(path)
    supplied = record.get("result_sha256")
    core = {key: value for key, value in record.items() if key != "result_sha256"}
    if (
        record.get("format") != expected_format
        or record.get("job_sha256") != job.identity_sha256
        or record.get("job") != job.as_record()
        or supplied != record_sha256(core)
    ):
        raise ValueError("temporal training artifact manifest changed")
    return record


def _latest_state(
    states_root: Path,
    identity_sha256: str,
    template: LmTrainState[object],
) -> LmTrainState[object]:
    states_root.mkdir(parents=True, exist_ok=True)
    candidates = tuple(
        sorted(
            (
                path
                for path in states_root.iterdir()
                if path.is_dir() and path.name.startswith("state-")
            ),
            key=lambda path: path.name,
        )
    )
    if not candidates:
        return template
    return load_lm_train_state_artifact(
        candidates[-1],
        identity_sha256,
        (template,),
    )[0]


def _write_state_if_missing(
    directory: Path,
    identity_sha256: str,
    state: LmTrainState[object],
) -> None:
    if directory.is_dir():
        loaded = load_lm_train_state_artifact(
            directory,
            identity_sha256,
            (state,),
        )[0]
        if int(loaded.step) != int(state.step):
            raise ValueError("persisted training state update changed")
        return
    write_lm_train_state_artifact(directory, identity_sha256, (state,))


def _job_key(identity_sha256: str) -> jax.Array:
    require_sha256(identity_sha256, "training job key")
    seed = int(identity_sha256[:8], 16) ^ SEED
    return jax.random.PRNGKey(seed)


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
        or type(record.get("update")) is not int
        or type(elapsed) not in (int, float)
        or not math.isfinite(float(elapsed))
        or float(elapsed) < 0.0
    ):
        raise ValueError("temporal training runtime state changed")
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


__all__ = [
    "AdapterArtifact",
    "FullModelArtifact",
    "StoryEpochBatches",
    "TrainingInterrupted",
    "TrainingJob",
    "adapter_checksum",
    "load_adapter_artifact",
    "load_full_model_artifact",
    "train_or_load_full_model",
    "train_or_load_lora",
]

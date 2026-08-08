"""Lazy batches, resumable base training, and VAMP-only noun adaptation."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, fields, replace
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from apm.continual.language_adaptation_artifact import (
    LanguageAdaptationArtifact,
    extract_language_vamp_artifact,
    flatten_lora_edge,
    load_language_adaptation_artifact,
    save_language_adaptation_artifact,
)
from apm.continual.language_run import (
    LanguageStageMetrics,
    LanguageVampRun,
    advance_language_vamp_run,
    init_language_vamp_run,
    score_parent_nodes,
)
from apm.continual.language_tasks import LanguageTask, RouterBatch
from apm.data.text.tinyworlds_nouns_v1.contracts import (
    BENCHMARK_ID,
    CHECKPOINT_ROOT,
    NounPartitionArtifact,
    NounsExperimentPreset,
    canonical_json_bytes,
    record_sha256,
    require_sha256,
)
from apm.data.text.tinyworlds_p.contracts import CANONICAL_TOKENIZER_IDENTITY
from apm.lm.checkpoint import (
    BaseCheckpointRef,
    CheckpointFileHash,
    SourceCheckpointMetadata,
    TokenizerCheckpointMetadata,
    load_gpt_neo_checkpoint,
    parameter_checksum,
    save_gpt_neo_checkpoint,
)
from apm.lm.config import GptNeoConfig
from apm.lm.gpt_neo import apply_gpt_neo
from apm.lm.lora import LoraConfig
from apm.lm.losses import per_token_nll
from apm.lm.parameters import GptNeoParams, init_gpt_neo_params
from apm.lm.text_data import TokenBatch
from apm.lm.training import LmTrainConfig, LmTrainState
from apm.lm.training_state_artifact import (
    load_lm_train_state_artifact,
    write_lm_train_state_artifact,
)
from apm.memory.graph import TaskId


BASE_TRAINING_FORMAT = "tinyworlds-nouns-base-training-v1"
BASE_SELECTION_FORMAT = "tinyworlds-nouns-selected-base-v1"
VAMP_STAGE_FORMAT = "tinyworlds-nouns-vamp-stage-v1"
GPU_PREFLIGHT_FORMAT = "tinyworlds-nouns-gpu-preflight-v1"
ProgressCallback = Callable[[str, int, int, float], None]


@dataclass(frozen=True, slots=True)
class StoryIndexEntry:
    """One compact pointer into partition text and token stores."""

    story_id: str
    story_index: int
    story_offset: int
    byte_length: int
    token_offset: int
    token_count: int

    def __post_init__(self) -> None:
        require_sha256(self.story_id, "indexed story")
        if any(
            type(value) is not int or value < 0
            for value in (
                self.story_index,
                self.story_offset,
                self.byte_length,
                self.token_offset,
                self.token_count,
            )
        ):
            raise ValueError("story index offsets and counts must be nonnegative")
        if self.byte_length == 0 or self.token_count < 2:
            raise ValueError("indexed stories require text and one causal target")


@dataclass(frozen=True, slots=True)
class NounBaseCursor:
    """Next epoch/batch and completed optimizer update for exact base resume."""

    epoch: int
    next_batch: int
    optimizer_update: int

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value < 0
            for value in (self.epoch, self.next_batch, self.optimizer_update)
        ):
            raise ValueError("base cursor values must be nonnegative")

    def as_record(self) -> dict[str, int]:
        """Return the canonical resume cursor."""
        return {
            "epoch": self.epoch,
            "next_batch": self.next_batch,
            "optimizer_update": self.optimizer_update,
        }


@dataclass(frozen=True, slots=True)
class NounSelectedBase:
    """Accepted fresh base and its two held-in epoch NLL measurements."""

    directory: Path
    reference: BaseCheckpointRef
    training_sha256: str
    preflight_sha256: str
    epoch_validation_nll: tuple[float, float]
    peak_allocator_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "directory", Path(self.directory))
        require_sha256(self.training_sha256, "noun base training")
        require_sha256(self.preflight_sha256, "noun base preflight")
        if len(self.epoch_validation_nll) != 2 or any(
            not math.isfinite(value) or value < 0.0
            for value in self.epoch_validation_nll
        ):
            raise ValueError("base epoch NLL must be finite and nonnegative")
        if self.peak_allocator_bytes < 0:
            raise ValueError("base allocator peak must be nonnegative")


@dataclass(frozen=True, slots=True)
class NounResourceEstimate:
    """Frozen scale estimate for training, routing, and result artifacts."""

    task_count: int
    max_nodes: int
    base_training_batches_per_epoch: int
    adapter_updates: int
    parent_node_story_scores: int
    whole_story_result_rows: int
    generation_result_rows: int
    estimated_result_bytes: int
    estimated_training_peak_bytes: int
    estimated_routing_peak_bytes: int

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value <= 0
            for field in fields(self)
            for value in (getattr(self, field.name),)
        ):
            raise ValueError("noun resource estimates must be positive integers")

    def as_record(self) -> dict[str, int]:
        """Return every resource estimate in canonical form."""
        return {
            field: getattr(self, field) for field in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class NounGpuPreflight:
    """Measured representative GPU allocation bound to partition and config."""

    preflight_sha256: str
    partition_sha256: str
    config_sha256: str
    device_platform: str
    measured_peak_bytes: int
    allocator_limit_bytes: int
    estimate: NounResourceEstimate
    artifact_path: Path

    def __post_init__(self) -> None:
        for digest in (
            self.preflight_sha256,
            self.partition_sha256,
            self.config_sha256,
        ):
            require_sha256(digest, "noun GPU preflight")
        object.__setattr__(self, "artifact_path", Path(self.artifact_path))
        if (
            self.device_platform != "gpu"
            or type(self.measured_peak_bytes) is not int
            or self.measured_peak_bytes < 0
            or type(self.allocator_limit_bytes) is not int
            or self.allocator_limit_bytes <= 0
            or self.measured_peak_bytes > self.allocator_limit_bytes
            or type(self.estimate) is not NounResourceEstimate
        ):
            raise ValueError("noun GPU preflight did not satisfy its frozen device limit")


class IndexedTokenBatchSequence(Sequence[TokenBatch]):
    """Deterministic random-access causal batches backed by a token memmap."""

    def __init__(
        self,
        artifact: NounPartitionArtifact,
        index_name: str,
        *,
        context_length: int,
        batch_size: int,
        order_namespace: str,
    ) -> None:
        if not index_name or not order_namespace:
            raise ValueError("indexed batches require index and ordering names")
        if context_length <= 0 or batch_size <= 0:
            raise ValueError("indexed batch dimensions must be positive")
        entries = load_story_index(artifact, index_name)
        if not entries:
            raise ValueError(f"noun index contains no stories: {index_name}")
        self._artifact = artifact
        self._context_length = context_length
        self._batch_size = batch_size
        self._entries = tuple(
            sorted(
                entries,
                key=lambda entry: (
                    sha256(
                        f"{_benchmark_id(artifact)}\0{order_namespace}\0"
                        f"{artifact.partition_sha256}\0{entry.story_id}".encode("utf-8")
                    ).hexdigest(),
                    entry.story_id,
                ),
            )
        )
        window_counts = tuple(
            math.ceil((entry.token_count - 1) / context_length)
            for entry in self._entries
        )
        self._window_stops = tuple(np.cumsum(window_counts, dtype=np.int64).tolist())
        self._window_count = self._window_stops[-1]
        self._tokens = np.memmap(
            _token_store_path(artifact),
            mode="r",
            dtype="<u2",
        )

    def __len__(self) -> int:
        return math.ceil(self._window_count / self._batch_size)

    def __getitem__(self, index: int | slice) -> TokenBatch | tuple[TokenBatch, ...]:
        if isinstance(index, slice):
            return tuple(self[position] for position in range(*index.indices(len(self))))
        position = index + len(self) if index < 0 else index
        if not 0 <= position < len(self):
            raise IndexError(index)
        shape = (self._batch_size, self._context_length)
        input_ids = np.full(shape, self._artifact.pad_token_id, dtype=np.int32)
        target_ids = np.full(shape, self._artifact.pad_token_id, dtype=np.int32)
        attention_mask = np.zeros(shape, dtype=np.bool_)
        loss_mask = np.zeros(shape, dtype=np.bool_)
        first_window = position * self._batch_size
        stop_window = min(first_window + self._batch_size, self._window_count)
        for row, global_window in enumerate(range(first_window, stop_window)):
            story_position = bisect_right(self._window_stops, global_window)
            prior_stop = 0 if story_position == 0 else self._window_stops[story_position - 1]
            entry = self._entries[story_position]
            local_window = global_window - prior_stop
            token_start = entry.token_offset + local_window * self._context_length
            token_stop = min(
                entry.token_offset + entry.token_count,
                token_start + self._context_length + 1,
            )
            chunk = np.asarray(self._tokens[token_start:token_stop], dtype=np.int32)
            transitions = len(chunk) - 1
            input_ids[row, :transitions] = chunk[:-1]
            target_ids[row, :transitions] = chunk[1:]
            attention_mask[row, :transitions] = True
            loss_mask[row, :transitions] = True
        return TokenBatch(input_ids, attention_mask, target_ids, loss_mask)

    @property
    def batch_size(self) -> int:
        """Return the fixed number of rows in every materialized batch."""
        return self._batch_size

    @property
    def sequence_width(self) -> int:
        """Return the fixed causal width of every materialized batch."""
        return self._context_length

    @property
    def story_count(self) -> int:
        """Return the indexed source population size."""
        return len(self._entries)

    @property
    def window_count(self) -> int:
        """Return the number of non-overlapping causal windows."""
        return self._window_count

    def consumed_story_ids(self, update_count: int) -> tuple[str, ...]:
        """Return unique stories touched by the fixed prefix of optimizer batches."""
        if type(update_count) is not int or update_count < 0:
            raise ValueError("update_count must be nonnegative")
        consumed_windows = min(update_count * self._batch_size, self._window_count)
        consumed_story_count = bisect_right(self._window_stops, consumed_windows - 1) + 1 if consumed_windows else 0
        return tuple(entry.story_id for entry in self._entries[:consumed_story_count])


class IndexedStoryStore:
    """Bounded random-access reader for exact partition stories and tokens."""

    def __init__(self, artifact: NounPartitionArtifact) -> None:
        self._artifact = artifact
        self._stories = np.memmap(
            _story_store_path(artifact), mode="r", dtype=np.uint8
        )
        self._tokens = np.memmap(
            _token_store_path(artifact), mode="r", dtype="<u2"
        )

    def text(self, entry: StoryIndexEntry) -> str:
        """Read and verify one exact normalized story."""
        payload = bytes(
            self._stories[entry.story_offset : entry.story_offset + entry.byte_length]
        )
        if sha256(payload).hexdigest() != entry.story_id:
            raise ValueError("indexed story bytes changed")
        return payload.decode("utf-8", errors="strict")

    def tokens(self, entry: StoryIndexEntry) -> tuple[int, ...]:
        """Read one immutable EOS-terminated token sequence."""
        values = self._tokens[
            entry.token_offset : entry.token_offset + entry.token_count
        ]
        return tuple(int(value) for value in values)


def load_story_index(
    artifact: NounPartitionArtifact,
    index_name: str,
) -> tuple[StoryIndexEntry, ...]:
    """Load one verified partition index into compact immutable pointers."""
    if not index_name or Path(index_name).name != index_name:
        raise ValueError("noun index name must be one basename")
    path = artifact.root / "indexes" / f"{index_name}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    entries = tuple(
        _index_entry(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    )
    if len({entry.story_id for entry in entries}) != len(entries):
        raise ValueError(f"noun index contains duplicate stories: {index_name}")
    return entries


def router_batch_from_index(
    artifact: NounPartitionArtifact,
    index_name: str,
    context_length: int,
) -> RouterBatch:
    """Build one fixed-width full-story router batch from context-fitting probes."""
    entries = load_story_index(artifact, index_name)
    store = IndexedStoryStore(artifact)
    shape = (len(entries), context_length)
    input_ids = np.full(shape, artifact.pad_token_id, dtype=np.int32)
    target_ids = np.full(shape, artifact.pad_token_id, dtype=np.int32)
    attention_mask = np.zeros(shape, dtype=np.bool_)
    for row, entry in enumerate(entries):
        tokens = store.tokens(entry)
        transitions = len(tokens) - 1
        if not 1 <= transitions <= context_length:
            raise ValueError("router probe does not fit one model context")
        input_ids[row, :transitions] = tokens[:-1]
        target_ids[row, :transitions] = tokens[1:]
        attention_mask[row, :transitions] = True
    return RouterBatch(input_ids, attention_mask, target_ids, attention_mask)


def noun_model_config(vocab_size: int = 50_257) -> GptNeoConfig:
    """Return the retained eight-layer TinyStories GPT-Neo architecture."""
    return GptNeoConfig(
        vocab_size=vocab_size,
        max_position_embeddings=2_048,
        hidden_size=256,
        intermediate_size=1_024,
        num_layers=8,
        num_heads=16,
        attention_types=("global", "local") * 4,
        local_window_size=256,
        embedding_dropout=0.0,
        attention_dropout=0.0,
        residual_dropout=0.0,
    )


def _model_config_record(config: GptNeoConfig) -> dict[str, object]:
    return {field.name: getattr(config, field.name) for field in fields(config)}


def estimate_noun_resources(
    artifact: NounPartitionArtifact,
    preset: NounsExperimentPreset,
) -> NounResourceEstimate:
    """Estimate every scale-sensitive phase from the approved dynamic manifest."""
    base_batches = len(
        IndexedTokenBatchSequence(
            artifact,
            "base-train",
            context_length=preset.context_length,
            batch_size=preset.microbatch_size,
            order_namespace="resource-estimate",
        )
    )
    validation_pairs = sum(task.validation_story_count for task in artifact.tasks)
    generation_pairs = sum(task.generation_story_count for task in artifact.tasks)
    vocab_size = artifact.tokenizer_identity.get("vocab_size")
    if type(vocab_size) is not int:
        raise ValueError("partition tokenizer identity has no integer vocabulary size")
    model = noun_model_config(vocab_size)
    parameter_count = (
        model.vocab_size * model.hidden_size
        + model.max_position_embeddings * model.hidden_size
        + model.num_layers
        * (
            4 * model.hidden_size * model.hidden_size
            + 2 * model.hidden_size * model.intermediate_size
        )
    )
    parameter_bytes = parameter_count * 4
    logits_bytes = (
        preset.microbatch_size
        * preset.context_length
        * model.vocab_size
        * 4
    )
    task_count = len(artifact.task_ids)
    return NounResourceEstimate(
        task_count=task_count,
        max_nodes=artifact.max_nodes,
        base_training_batches_per_epoch=base_batches,
        adapter_updates=task_count * preset.adapter_updates,
        parent_node_story_scores=36 * sum(range(1, task_count + 1)),
        whole_story_result_rows=validation_pairs * 6,
        generation_result_rows=generation_pairs,
        estimated_result_bytes=(
            validation_pairs * 6 * 512
            + generation_pairs * 32_768
        ),
        estimated_training_peak_bytes=parameter_bytes * 4 + logits_bytes * 2,
        estimated_routing_peak_bytes=parameter_bytes * 2 + logits_bytes,
    )


def run_or_load_noun_gpu_preflight(
    artifact: NounPartitionArtifact,
    preset: NounsExperimentPreset,
    checkpoint_root: str | Path = CHECKPOINT_ROOT,
) -> NounGpuPreflight:
    """Measure one representative base update and enforce the 12 GiB GPU limit."""
    estimate = estimate_noun_resources(artifact, preset)
    vocab_size = artifact.tokenizer_identity.get("vocab_size")
    if type(vocab_size) is not int:
        raise ValueError("partition tokenizer identity has no integer vocabulary size")
    model_config = noun_model_config(vocab_size)
    preflight_format = _engine_format(
        artifact, "gpu_preflight_format", GPU_PREFLIGHT_FORMAT
    )
    identity_core = {
        "estimate": estimate.as_record(),
        "format": preflight_format,
        "model_config": _model_config_record(model_config),
        "partition_sha256": artifact.partition_sha256,
        "preset_sha256": preset.config_sha256,
    }
    identity = record_sha256(identity_core)
    path = Path(checkpoint_root) / "preflight" / f"{identity}.json"
    if path.is_file():
        return _load_gpu_preflight(path, artifact, preset, estimate)
    devices = jax.local_devices()
    if not devices or any(device.platform != "gpu" for device in devices):
        raise RuntimeError("noun GPU preflight requires only GPU-backed JAX devices")
    sequence = IndexedTokenBatchSequence(
        artifact,
        "base-train",
        context_length=preset.context_length,
        batch_size=preset.microbatch_size,
        order_namespace="gpu-preflight",
    )
    optimizer = _base_optimizer(preset, max(2, estimate.base_training_batches_per_epoch))
    params = init_gpt_neo_params(
        jax.random.PRNGKey(preset.seed), model_config, dtype=jnp.float32
    )
    state = LmTrainState(
        trainable=params,
        opt_state=optimizer.init(params),
        rng_key=jax.random.PRNGKey(preset.seed + 1),
        step=jnp.asarray(0, dtype=jnp.int32),
    )
    gradient_function, update_function = _compiled_base_steps(
        model_config,
        preset,
        optimizer,
    )
    gradient = gradient_function(state.trainable, state.rng_key, sequence[0])
    state = update_function(state, gradient[2], gradient[1])
    jax.tree_util.tree_map(jax.block_until_ready, state)
    measured = allocator_peak_bytes()
    if measured > preset.allocator_peak_limit_bytes:
        raise RuntimeError(
            f"noun GPU preflight measured {measured} bytes, above the frozen "
            f"{preset.allocator_peak_limit_bytes}-byte limit"
        )
    core = {
        **identity_core,
        "allocator_limit_bytes": preset.allocator_peak_limit_bytes,
        "device_platform": "gpu",
        "measured_peak_bytes": measured,
    }
    preflight_sha256 = record_sha256(core)
    _atomic_write(
        path,
        canonical_json_bytes({**core, "preflight_sha256": preflight_sha256}),
    )
    preview = NounGpuPreflight(
        preflight_sha256=preflight_sha256,
        partition_sha256=artifact.partition_sha256,
        config_sha256=preset.config_sha256,
        device_platform="gpu",
        measured_peak_bytes=measured,
        allocator_limit_bytes=preset.allocator_peak_limit_bytes,
        estimate=estimate,
        artifact_path=path,
    )
    _atomic_write(path.with_suffix(".md"), _render_gpu_preflight(preview).encode("utf-8"))
    return _load_gpu_preflight(path, artifact, preset, estimate)


def _load_gpu_preflight(
    path: Path,
    artifact: NounPartitionArtifact,
    preset: NounsExperimentPreset,
    estimate: NounResourceEstimate,
) -> NounGpuPreflight:
    payload = path.read_bytes()
    record = json.loads(payload)
    if payload != canonical_json_bytes(record):
        raise ValueError("noun GPU preflight JSON is not canonical")
    supplied = record.pop("preflight_sha256", None)
    preflight_format = _engine_format(
        artifact, "gpu_preflight_format", GPU_PREFLIGHT_FORMAT
    )
    identity_core = {
        "estimate": estimate.as_record(),
        "format": preflight_format,
        "model_config": _model_config_record(
            noun_model_config(int(artifact.tokenizer_identity["vocab_size"]))
        ),
        "partition_sha256": artifact.partition_sha256,
        "preset_sha256": preset.config_sha256,
    }
    expected_path = record_sha256(identity_core)
    if (
        set(record)
        != {
            "allocator_limit_bytes",
            "device_platform",
            "estimate",
            "format",
            "measured_peak_bytes",
            "model_config",
            "partition_sha256",
            "preset_sha256",
        }
        or supplied != record_sha256(record)
        or path.stem != expected_path
        or record.get("format") != preflight_format
        or record.get("partition_sha256") != artifact.partition_sha256
        or record.get("preset_sha256") != preset.config_sha256
        or record.get("estimate") != estimate.as_record()
        or record.get("allocator_limit_bytes") != preset.allocator_peak_limit_bytes
        or type(record.get("measured_peak_bytes")) is not int
        or type(record.get("allocator_limit_bytes")) is not int
    ):
        raise ValueError("noun GPU preflight binding changed")
    preflight = NounGpuPreflight(
        preflight_sha256=str(supplied),
        partition_sha256=artifact.partition_sha256,
        config_sha256=preset.config_sha256,
        device_platform=str(record["device_platform"]),
        measured_peak_bytes=int(record["measured_peak_bytes"]),
        allocator_limit_bytes=int(record["allocator_limit_bytes"]),
        estimate=estimate,
        artifact_path=path,
    )
    markdown = path.with_suffix(".md")
    if (
        not markdown.is_file()
        or markdown.read_text(encoding="utf-8") != _render_gpu_preflight(preflight)
    ):
        raise ValueError("noun GPU preflight report changed")
    return preflight


def _render_gpu_preflight(preflight: NounGpuPreflight) -> str:
    return (
        "# TinyWorlds nouns GPU preflight\n\n"
        f"Measured allocator peak: {preflight.measured_peak_bytes / 2**30:.3f} GiB\n\n"
        f"Frozen limit: {preflight.allocator_limit_bytes / 2**30:.3f} GiB\n\n"
        f"Tasks/nodes: {preflight.estimate.task_count}/{preflight.estimate.max_nodes}\n\n"
        f"Whole-story rows: {preflight.estimate.whole_story_result_rows:,}\n\n"
        f"Generation/judge cases: {preflight.estimate.generation_result_rows:,}\n"
    )


def run_or_resume_noun_base(
    artifact: NounPartitionArtifact,
    preset: NounsExperimentPreset,
    preflight: NounGpuPreflight,
    checkpoint_root: str | Path = CHECKPOINT_ROOT,
    *,
    progress: ProgressCallback | None = None,
) -> NounSelectedBase:
    """Train, exactly resume, quality-check, and publish the fresh noun base."""
    if type(preflight) is not NounGpuPreflight:
        raise TypeError("noun base requires a NounGpuPreflight")
    authenticated_preflight = _load_gpu_preflight(
        preflight.artifact_path,
        artifact,
        preset,
        estimate_noun_resources(artifact, preset),
    )
    if authenticated_preflight.preflight_sha256 != preflight.preflight_sha256:
        raise ValueError("noun base requires its exact passing GPU preflight")
    preflight = authenticated_preflight
    vocab_size = artifact.tokenizer_identity.get("vocab_size")
    if type(vocab_size) is not int:
        raise ValueError("partition tokenizer identity has no integer vocabulary size")
    model_config = noun_model_config(vocab_size)
    base_training_format = _engine_format(
        artifact, "base_training_format", BASE_TRAINING_FORMAT
    )
    training_core = {
        "format": base_training_format,
        "model_config": _model_config_record(model_config),
        "partition_sha256": artifact.partition_sha256,
        "preflight_sha256": preflight.preflight_sha256,
        "preset": preset.as_record(),
    }
    training_sha256 = record_sha256(training_core)
    root = Path(checkpoint_root) / "base" / training_sha256
    selected_manifest = root / "selected.json"
    if selected_manifest.is_file():
        return _load_selected_base(
            root,
            artifact,
            preset,
            preflight,
            training_sha256,
        )
    work = root / "work"
    states_root = work / "states"
    states_root.mkdir(parents=True, exist_ok=True)
    trace_path = work / "losses.jsonl"
    sequences = tuple(
        IndexedTokenBatchSequence(
            artifact,
            "base-train",
            context_length=preset.context_length,
            batch_size=preset.microbatch_size,
            order_namespace=f"base-epoch-{epoch}",
        )
        for epoch in range(preset.base_epochs)
    )
    planned_updates = sum(
        math.ceil(len(sequence) / preset.accumulation_microbatches)
        for sequence in sequences
    )
    optimizer = _base_optimizer(preset, planned_updates)
    initial_params = init_gpt_neo_params(
        jax.random.PRNGKey(preset.seed), model_config, dtype=jnp.float32
    )
    template = LmTrainState(
        trainable=initial_params,
        opt_state=optimizer.init(initial_params),
        rng_key=jax.random.PRNGKey(preset.seed + 1),
        step=jnp.asarray(0, dtype=jnp.int32),
    )
    latest = _latest_base_state(states_root, training_sha256)
    state, cursor = (
        (template, NounBaseCursor(0, 0, 0))
        if latest is None
        else _load_base_state(
            latest, training_sha256, template, base_training_format
        )
    )
    _trim_trace(trace_path, cursor.optimizer_update)
    evidence = _load_epoch_evidence(work, training_sha256)
    validation = IndexedTokenBatchSequence(
        artifact,
        "base-validation",
        context_length=preset.context_length,
        batch_size=preset.microbatch_size,
        order_namespace="base-validation",
    )
    if len(evidence) > cursor.epoch or cursor.epoch - len(evidence) > 1:
        raise ValueError("noun base resume epoch evidence and cursor differ")
    if len(evidence) < cursor.epoch:
        evidence = evidence + (
            evaluate_token_weighted_nll(state.trainable, model_config, validation),
        )
        _write_epoch_evidence(work, training_sha256, evidence)
    gradient_function, update_function = _compiled_base_steps(
        model_config,
        preset,
        optimizer,
    )
    trace = trace_path.open("ab")
    try:
        for epoch in range(cursor.epoch, preset.base_epochs):
            sequence = sequences[epoch]
            batch_start = cursor.next_batch if cursor.epoch == epoch else 0
            gradient_sum = jax.tree_util.tree_map(jnp.zeros_like, state.trainable)
            loss_sum = jnp.asarray(0.0, dtype=jnp.float32)
            token_sum = jnp.asarray(0.0, dtype=jnp.float32)
            accumulation_count = 0
            for batch_index in range(batch_start, len(sequence)):
                gradient = gradient_function(state.trainable, state.rng_key, sequence[batch_index])
                state = replace(state, rng_key=gradient[3])
                gradient_sum = jax.tree_util.tree_map(jnp.add, gradient_sum, gradient[2])
                loss_sum += gradient[0]
                token_sum += gradient[1]
                accumulation_count += 1
                is_epoch_end = batch_index + 1 == len(sequence)
                if (
                    accumulation_count == preset.accumulation_microbatches
                    or is_epoch_end
                ):
                    state = update_function(state, gradient_sum, token_sum)
                    cursor = NounBaseCursor(epoch, batch_index + 1, int(state.step))
                    nll = float(loss_sum / token_sum)
                    _append_trace(trace, cursor, nll)
                    if progress is not None:
                        progress("base-training", int(state.step), planned_updates, nll)
                    gradient_sum = jax.tree_util.tree_map(jnp.zeros_like, gradient_sum)
                    loss_sum = jnp.asarray(0.0, dtype=jnp.float32)
                    token_sum = jnp.asarray(0.0, dtype=jnp.float32)
                    accumulation_count = 0
                    if int(state.step) % preset.base_checkpoint_interval == 0:
                        _write_base_state(
                            states_root,
                            training_sha256,
                            state,
                            cursor,
                            base_training_format,
                        )
            cursor = NounBaseCursor(epoch + 1, 0, int(state.step))
            _write_base_state(
                states_root,
                training_sha256,
                state,
                cursor,
                base_training_format,
            )
            if len(evidence) <= epoch:
                epoch_nll = evaluate_token_weighted_nll(
                    state.trainable,
                    model_config,
                    validation,
                )
                evidence = evidence + (epoch_nll,)
                _write_epoch_evidence(work, training_sha256, evidence)
    finally:
        trace.flush()
        os.fsync(trace.fileno())
        trace.close()
    if len(evidence) != 2:
        raise RuntimeError("noun base did not produce exactly two epoch measurements")
    if (
        evidence[1] >= evidence[0]
        or not all(math.isfinite(value) for value in evidence)
    ):
        raise RuntimeError(
            "noun base quality gate failed: epoch NLLs "
            f"{evidence[0]:.6f}, {evidence[1]:.6f}"
        )
    peak_bytes = max(allocator_peak_bytes(), preflight.measured_peak_bytes)
    if peak_bytes > preset.allocator_peak_limit_bytes:
        raise RuntimeError("noun base exceeded the frozen 12 GiB allocator limit")
    checkpoint_directory = root / "checkpoint"
    if not checkpoint_directory.exists():
        save_gpt_neo_checkpoint(
            checkpoint_directory,
            state.trainable,
            model_config,
            tokenizer=_tokenizer_checkpoint_metadata(),
            source=SourceCheckpointMetadata(
                identifier="roneneldan/TinyStories",
                revision=str(artifact.source_identity["revision"]),
                sha256=record_sha256(artifact.source_identity),
            ),
        )
    reference = load_gpt_neo_checkpoint(checkpoint_directory).reference
    selected_core = {
        "epoch_validation_nll": list(evidence),
        "format": _engine_format(
            artifact, "base_selection_format", BASE_SELECTION_FORMAT
        ),
        "parameter_checksum": reference.parameter_checksum,
        "partition_sha256": artifact.partition_sha256,
        "peak_allocator_bytes": peak_bytes,
        "preflight_sha256": preflight.preflight_sha256,
        "preset_sha256": preset.config_sha256,
        "training_sha256": training_sha256,
    }
    _atomic_write(
        selected_manifest,
        canonical_json_bytes(
            {**selected_core, "selection_sha256": record_sha256(selected_core)}
        ),
    )
    return NounSelectedBase(
        root,
        reference,
        training_sha256,
        preflight.preflight_sha256,
        (evidence[0], evidence[1]),
        peak_bytes,
    )


def run_or_resume_noun_vamp(
    artifact: NounPartitionArtifact,
    preset: NounsExperimentPreset,
    selected_base: NounSelectedBase,
    checkpoint_root: str | Path = CHECKPOINT_ROOT,
    *,
    progress: ProgressCallback | None = None,
) -> LanguageAdaptationArtifact:
    """Resume completed stages and train exactly one VAMP edge per noun task."""
    loaded_base, lora_config, train_config, run_sha256 = _noun_vamp_execution(
        artifact,
        preset,
        selected_base,
    )
    model_config = loaded_base.config
    root = Path(checkpoint_root) / "vamp" / run_sha256
    root.mkdir(parents=True, exist_ok=True)
    tasks = tuple(
        _language_task(artifact, task_id, preset) for task_id in artifact.task_ids
    )
    latest = _latest_vamp_stage(root, artifact.task_ids)
    if latest is None:
        root_probes = router_batch_from_index(
            artifact,
            "root-probes",
            preset.context_length,
        )
        run = init_language_vamp_run(
            selected_base.reference,
            loaded_base.params,
            model_config,
            (root_probes,),
            jax.random.PRNGKey(preset.seed + 2),
            max_nodes=artifact.max_nodes,
            max_edges=artifact.max_edges,
            key_probe_count=36,
            evaluation_microbatch_size=preset.evaluation_chunk_size,
        )
    else:
        stage_record = _load_vamp_stage_record(latest)
        persisted = load_language_adaptation_artifact(latest / "adaptation")
        _require_vamp_bindings(
            persisted,
            selected_base,
            model_config,
            lora_config,
            train_config,
            artifact,
            preset,
        )
        _require_vamp_stage_record(
            stage_record,
            persisted,
            artifact,
            preset,
            selected_base,
            latest,
        )
        run = _run_from_artifact(persisted, tasks[: len(persisted.task_order)])
    for stage_index in range(len(run.completed_tasks), len(tasks)):
        task = tasks[stage_index]
        eligibility = (
            (True,)
            if stage_index == 0
            else (False,) + (True,) * stage_index
        )
        parent = score_parent_nodes(
            run,
            task.parent_probes,
            loaded_base.params,
            model_config,
            lora_config,
            eligible_node_mask=eligibility,
            evaluation_microbatch_size=preset.evaluation_chunk_size,
        )
        old_checksums = _edge_checksums(run, model_config, lora_config)
        started = time.monotonic()

        def stage_progress(update: int, loss: float, total: int) -> None:
            if progress is not None:
                progress(f"vamp-{task.task_id}", update, total, loss)

        run = advance_language_vamp_run(
            run,
            task,
            loaded_base.params,
            model_config,
            lora_config,
            train_config,
            parent,
            key_probe_count=36,
            evaluation_microbatch_size=preset.evaluation_chunk_size,
            training_progress=stage_progress,
        )
        if _edge_checksums(run, model_config, lora_config)[: len(old_checksums)] != old_checksums:
            raise RuntimeError("committing a noun edge changed an older VAMP edge")
        stage = extract_language_vamp_artifact(
            run,
            model_config,
            lora_config,
            train_config,
            config_hashes={
                "experiment": preset.config_sha256,
                "partition": artifact.partition_sha256,
            },
        )
        target = root / f"stage-{stage_index + 1:03d}-{task.task_id}"
        temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=root))
        try:
            save_language_adaptation_artifact(temporary / "adaptation", stage)
            if not isinstance(task.train_batches, IndexedTokenBatchSequence):
                raise TypeError("noun task lost its indexed training sequence")
            consumed = task.train_batches.consumed_story_ids(preset.adapter_updates)
            stage_record = {
                "adaptation_manifest_sha256": _file_sha256(
                    temporary / "adaptation" / "manifest.json"
                ),
                "adaptation_tensor_checksum": stage.tensor_checksum,
                "base_training_sha256": selected_base.training_sha256,
                "elapsed_seconds": time.monotonic() - started,
                "eligible_node_mask": list(parent.eligible_node_mask),
                "format": _engine_format(
                    artifact, "vamp_stage_format", VAMP_STAGE_FORMAT
                ),
                "parent_node_id": str(parent.selected_node_id),
                "parent_scores": list(parent.mean_candidate_nll),
                "partition_sha256": artifact.partition_sha256,
                "preset_sha256": preset.config_sha256,
                "source_story_count": task.train_batches.story_count,
                "source_window_count": task.train_batches.window_count,
                "stage_index": stage_index + 1,
                "task_id": str(task.task_id),
                "unique_consumed_story_count": len(consumed),
            }
            _atomic_write(
                temporary / "stage.json",
                canonical_json_bytes(
                    {**stage_record, "stage_sha256": record_sha256(stage_record)}
                ),
            )
            os.replace(temporary, target)
        except BaseException:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
    final_stage = _latest_vamp_stage(root, artifact.task_ids)
    if final_stage is None:
        raise RuntimeError("noun VAMP produced no committed task stage")
    final_record = _load_vamp_stage_record(final_stage)
    final_artifact = load_language_adaptation_artifact(final_stage / "adaptation")
    _require_vamp_stage_record(
        final_record,
        final_artifact,
        artifact,
        preset,
        selected_base,
        final_stage,
    )
    return final_artifact


def load_noun_vamp_stages(
    artifact: NounPartitionArtifact,
    preset: NounsExperimentPreset,
    selected_base: NounSelectedBase,
    checkpoint_root: str | Path = CHECKPOINT_ROOT,
) -> tuple[LanguageAdaptationArtifact, ...]:
    """Strict-load and authenticate every committed stage of a complete VAMP run."""
    loaded_base, lora_config, train_config, run_sha256 = _noun_vamp_execution(
        artifact,
        preset,
        selected_base,
    )
    root = Path(checkpoint_root) / "vamp" / run_sha256
    latest = _latest_vamp_stage(root, artifact.task_ids)
    expected_paths = tuple(
        root / f"stage-{index:03d}-{task_id}"
        for index, task_id in enumerate(artifact.task_ids, start=1)
    )
    if latest != expected_paths[-1] or any(not path.is_dir() for path in expected_paths):
        raise ValueError("noun VAMP stagewise audit requires all canonical stages")
    stages: list[LanguageAdaptationArtifact] = []
    prior_edge_checksums: tuple[str, ...] = ()
    prior_keys: np.ndarray | None = None
    for path in expected_paths:
        record = _load_vamp_stage_record(path)
        persisted = load_language_adaptation_artifact(path / "adaptation")
        _require_vamp_bindings(
            persisted,
            selected_base,
            loaded_base.config,
            lora_config,
            train_config,
            artifact,
            preset,
        )
        _require_vamp_stage_record(
            record,
            persisted,
            artifact,
            preset,
            selected_base,
            path,
        )
        edge_checksums = _artifact_edge_checksums(persisted)
        if edge_checksums[: len(prior_edge_checksums)] != prior_edge_checksums:
            raise ValueError("persisted noun VAMP stage changed an earlier edge")
        keys = np.asarray(persisted.address_book.keys)
        if prior_keys is not None and not np.array_equal(
            keys[: prior_keys.shape[0]], prior_keys
        ):
            raise ValueError("persisted noun VAMP stage changed an earlier content key")
        valid_count = len(persisted.vamp_graph.nodes)
        prior_edge_checksums = edge_checksums
        prior_keys = np.array(keys[:valid_count], copy=True)
        stages.append(persisted)
    return tuple(stages)


def _noun_vamp_execution(
    artifact: NounPartitionArtifact,
    preset: NounsExperimentPreset,
    selected_base: NounSelectedBase,
):
    """Resolve the one content-addressed VAMP configuration used by run and audit."""
    loaded_base = load_gpt_neo_checkpoint(selected_base.reference)
    lora_config = LoraConfig(rank=preset.lora_rank, alpha=preset.lora_alpha)
    train_config = LmTrainConfig(
        learning_rate=preset.adapter_learning_rate,
        steps=preset.adapter_updates,
        batch_size=preset.microbatch_size,
        weight_decay=preset.adapter_weight_decay,
        gradient_clip_norm=preset.gradient_clip_norm,
    )
    run_sha256 = record_sha256(
        {
            "base": selected_base.training_sha256,
            "lora_config": {
                "alpha": lora_config.alpha,
                "rank": lora_config.rank,
                "target_mask": {
                    field.name: getattr(lora_config.target_mask, field.name)
                    for field in fields(lora_config.target_mask)
                },
            },
            "model_config": _model_config_record(loaded_base.config),
            "partition": artifact.partition_sha256,
            "preset": preset.as_record(),
            "train_config": {
                field.name: getattr(train_config, field.name)
                for field in fields(train_config)
            },
            "vamp_only": True,
        }
    )
    return loaded_base, lora_config, train_config, run_sha256


def evaluate_token_weighted_nll(
    params: GptNeoParams,
    model_config: GptNeoConfig,
    batches: Sequence[TokenBatch],
) -> float:
    """Evaluate total causal NLL over a bounded or lazy batch sequence."""
    if not batches:
        raise ValueError("NLL evaluation requires batches")

    @jax.jit
    def evaluate(batch: TokenBatch) -> tuple[jax.Array, jax.Array]:
        result = apply_gpt_neo(
            params,
            model_config,
            jnp.asarray(batch.input_ids),
            jnp.asarray(batch.attention_mask),
            training=False,
        )
        mask = jnp.asarray(batch.loss_mask, dtype=jnp.float32)
        losses = per_token_nll(result.logits, jnp.asarray(batch.target_ids))
        return jnp.sum(losses * mask), jnp.sum(mask)

    total_loss = 0.0
    total_tokens = 0.0
    for batch in batches:
        loss, tokens = evaluate(batch)
        total_loss += float(loss)
        total_tokens += float(tokens)
    if total_tokens <= 0.0:
        raise ValueError("NLL evaluation contains no active targets")
    return total_loss / total_tokens


def allocator_peak_bytes() -> int:
    """Return the largest visible JAX device allocator peak when available."""
    peaks = tuple(
        int(stats.get("peak_bytes_in_use", stats.get("bytes_in_use", 0)))
        for device in jax.local_devices()
        for stats in (device.memory_stats() or {},)
    )
    return max(peaks, default=0)


def _language_task(
    artifact: NounPartitionArtifact,
    task_id: str,
    preset: NounsExperimentPreset,
) -> LanguageTask:
    probes = router_batch_from_index(
        artifact,
        f"task-{task_id}-probes",
        preset.context_length,
    )
    batches = IndexedTokenBatchSequence(
        artifact,
        f"task-{task_id}-train",
        context_length=preset.context_length,
        batch_size=preset.microbatch_size,
        order_namespace=f"adapter-{task_id}",
    )
    return LanguageTask(
        task_id=TaskId(task_id),
        train_batches=batches,
        validation_examples=(),
        test_examples=(),
        parent_probes=(probes,),
        content_key_probes=(probes,),
    )


def _run_from_artifact(
    artifact: LanguageAdaptationArtifact,
    completed_tasks: tuple[LanguageTask, ...],
) -> LanguageVampRun:
    metrics = tuple(
        LanguageStageMetrics(
            stage_index=record.stage_index,
            task_id=record.task_id,
            parent_node_index=record.parent_node_index,
            parent_node_id=record.parent_node_id,
            parent_mean_node_nll=record.parent_mean_node_nll,
            candidate_step_losses=record.training_trace,
            task_metrics=(),
        )
        for record in artifact.vamp_stages
    )
    return LanguageVampRun(
        base_checkpoint=artifact.base_checkpoint,
        graph=artifact.vamp_graph,
        address_book=artifact.address_book,
        rng_key=jnp.asarray(artifact.rng_state.vamp, dtype=jnp.uint32),
        completed_tasks=completed_tasks,
        stage_metrics=metrics,
        max_nodes=artifact.max_nodes,
        max_edges=artifact.max_edges,
    )


def _require_vamp_bindings(
    persisted: LanguageAdaptationArtifact,
    selected_base: NounSelectedBase,
    model_config: GptNeoConfig,
    lora_config: LoraConfig,
    train_config: LmTrainConfig,
    partition: NounPartitionArtifact,
    preset: NounsExperimentPreset,
) -> None:
    if (
        persisted.base_checkpoint.parameter_checksum
        != selected_base.reference.parameter_checksum
        or persisted.model_config != model_config
        or persisted.lora_config != lora_config
        or persisted.train_config != train_config
        or dict(persisted.config_hashes).get("partition")
        != partition.partition_sha256
        or dict(persisted.config_hashes).get("experiment") != preset.config_sha256
        or tuple(str(task_id) for task_id in persisted.task_order)
        != partition.task_ids[: len(persisted.task_order)]
    ):
        raise ValueError("persisted noun VAMP stage has different bindings")


def _edge_checksums(
    run: LanguageVampRun,
    model_config: GptNeoConfig,
    lora_config: LoraConfig,
) -> tuple[str, ...]:
    return _graph_edge_checksums(run.graph.nodes, model_config, lora_config)


def _artifact_edge_checksums(
    artifact: LanguageAdaptationArtifact,
) -> tuple[str, ...]:
    return _graph_edge_checksums(
        artifact.vamp_graph.nodes,
        artifact.model_config,
        artifact.lora_config,
    )


def _graph_edge_checksums(nodes, model_config, lora_config) -> tuple[str, ...]:
    return tuple(
        sha256(
            b"".join(
                np.asarray(value).tobytes()
                for value in flatten_lora_edge(
                    node.incoming_edge, model_config, lora_config
                ).values()
            )
        ).hexdigest()
        for node in nodes[1:]
        if node.incoming_edge is not None
    )


def _base_optimizer(
    preset: NounsExperimentPreset,
    total_updates: int,
) -> optax.GradientTransformation:
    warmup = max(1, math.ceil(preset.warmup_fraction * total_updates))

    def schedule(update: jax.Array) -> jax.Array:
        position = jnp.asarray(update, dtype=jnp.float32)
        warmup_rate = preset.maximum_learning_rate * (position + 1.0) / warmup
        decay_position = jnp.clip(
            (position - warmup) / max(1, total_updates - warmup - 1),
            0.0,
            1.0,
        )
        cosine = 0.5 * (1.0 + jnp.cos(jnp.pi * decay_position))
        decay_rate = preset.minimum_learning_rate + (
            preset.maximum_learning_rate - preset.minimum_learning_rate
        ) * cosine
        return jnp.where(position < warmup, warmup_rate, decay_rate)

    return optax.chain(
        optax.clip_by_global_norm(preset.gradient_clip_norm),
        optax.adamw(
            learning_rate=schedule,
            b1=preset.adam_beta1,
            b2=preset.adam_beta2,
            eps=preset.adam_epsilon,
            weight_decay=preset.base_weight_decay,
        ),
    )


def _compiled_base_steps(
    model_config: GptNeoConfig,
    preset: NounsExperimentPreset,
    optimizer: optax.GradientTransformation,
):
    def gradients(
        params: GptNeoParams,
        rng_key: jax.Array,
        batch: TokenBatch,
    ):
        next_rng, dropout_rng = jax.random.split(rng_key)

        def loss(current: GptNeoParams):
            result = apply_gpt_neo(
                current,
                model_config,
                jnp.asarray(batch.input_ids),
                jnp.asarray(batch.attention_mask),
                training=True,
                rng_key=dropout_rng,
            )
            mask = jnp.asarray(batch.loss_mask, dtype=jnp.float32)
            losses = per_token_nll(result.logits, jnp.asarray(batch.target_ids))
            return jnp.sum(losses * mask), jnp.sum(mask)

        (loss_sum, tokens), gradient = jax.value_and_grad(loss, has_aux=True)(params)
        return loss_sum, tokens, gradient, next_rng

    def update(
        state: LmTrainState[GptNeoParams],
        gradient_sum: GptNeoParams,
        token_sum: jax.Array,
    ) -> LmTrainState[GptNeoParams]:
        normalized = jax.tree_util.tree_map(
            lambda value: value / token_sum, gradient_sum
        )
        updates, opt_state = optimizer.update(
            normalized,
            state.opt_state,
            state.trainable,
        )
        return replace(
            state,
            trainable=optax.apply_updates(state.trainable, updates),
            opt_state=opt_state,
            step=state.step + jnp.asarray(1, dtype=jnp.int32),
        )

    return jax.jit(gradients), jax.jit(update)


def _write_base_state(
    root: Path,
    identity: str,
    state: LmTrainState[GptNeoParams],
    cursor: NounBaseCursor,
    base_training_format: str,
) -> Path:
    target = root / f"update-{cursor.optimizer_update:09d}-epoch-{cursor.epoch:02d}"
    if target.is_dir():
        return target
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=root))
    try:
        write_lm_train_state_artifact(temporary / "state", identity, (state,))
        resume_core = {
            "cursor": cursor.as_record(),
            "format": base_training_format,
            "identity_sha256": identity,
        }
        _atomic_write(
            temporary / "resume.json",
            canonical_json_bytes(
                {**resume_core, "resume_sha256": record_sha256(resume_core)}
            ),
        )
        os.replace(temporary, target)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return target


def _load_base_state(
    root: Path,
    identity: str,
    template: LmTrainState[GptNeoParams],
    base_training_format: str,
) -> tuple[LmTrainState[GptNeoParams], NounBaseCursor]:
    record = json.loads((root / "resume.json").read_text(encoding="utf-8"))
    supplied = record.pop("resume_sha256", None)
    if (
        supplied != record_sha256(record)
        or record.get("format") != base_training_format
        or record.get("identity_sha256") != identity
    ):
        raise ValueError("noun base resume identity changed")
    raw_cursor = record.get("cursor")
    if type(raw_cursor) is not dict:
        raise ValueError("noun base resume cursor is invalid")
    cursor = NounBaseCursor(
        int(raw_cursor["epoch"]),
        int(raw_cursor["next_batch"]),
        int(raw_cursor["optimizer_update"]),
    )
    state = load_lm_train_state_artifact(root / "state", identity, (template,))[0]
    if int(state.step) != cursor.optimizer_update:
        raise ValueError("noun base state and cursor updates differ")
    return state, cursor


def _latest_base_state(root: Path, identity: str) -> Path | None:
    candidates = tuple(path for path in root.glob("update-*-epoch-*") if path.is_dir())
    if not candidates:
        return None
    validated = tuple(
        (
            json.loads((path / "resume.json").read_text(encoding="utf-8"))["cursor"]["optimizer_update"],
            path,
        )
        for path in candidates
    )
    del identity
    return max(validated, key=lambda item: (item[0], item[1].name))[1]


def _append_trace(stream, cursor: NounBaseCursor, nll: float) -> None:
    if not math.isfinite(nll):
        raise RuntimeError("noun base produced non-finite training NLL")
    stream.write(
        canonical_json_bytes(
            {"cursor": cursor.as_record(), "training_nll": nll}
        )
    )
    stream.flush()


def _trim_trace(path: Path, update: int) -> None:
    if not path.is_file():
        return
    retained = tuple(
        line
        for line in path.read_bytes().splitlines(keepends=True)
        if int(json.loads(line)["cursor"]["optimizer_update"]) <= update
    )
    _atomic_write(path, b"".join(retained))


def _write_epoch_evidence(
    root: Path,
    identity: str,
    evidence: tuple[float, ...],
) -> None:
    core = {
        "epoch_validation_nll": list(evidence),
        "identity_sha256": identity,
    }
    _atomic_write(
        root / "epoch-evidence.json",
        canonical_json_bytes({**core, "evidence_sha256": record_sha256(core)}),
    )


def _load_epoch_evidence(root: Path, identity: str) -> tuple[float, ...]:
    path = root / "epoch-evidence.json"
    if not path.is_file():
        return ()
    record = json.loads(path.read_text(encoding="utf-8"))
    supplied = record.pop("evidence_sha256", None)
    if supplied != record_sha256(record) or record.get("identity_sha256") != identity:
        raise ValueError("noun base epoch evidence changed")
    return tuple(float(value) for value in record["epoch_validation_nll"])


def _load_selected_base(
    root: Path,
    artifact: NounPartitionArtifact,
    preset: NounsExperimentPreset,
    preflight: NounGpuPreflight,
    identity: str,
) -> NounSelectedBase:
    record = json.loads((root / "selected.json").read_text(encoding="utf-8"))
    supplied = record.pop("selection_sha256", None)
    if (
        supplied != record_sha256(record)
        or record.get("format")
        != _engine_format(artifact, "base_selection_format", BASE_SELECTION_FORMAT)
        or record.get("partition_sha256") != artifact.partition_sha256
        or record.get("preset_sha256") != preset.config_sha256
        or record.get("preflight_sha256") != preflight.preflight_sha256
        or record.get("training_sha256") != identity
    ):
        raise ValueError("selected noun base binding changed")
    loaded = load_gpt_neo_checkpoint(root / "checkpoint")
    if loaded.reference.parameter_checksum != record.get("parameter_checksum"):
        raise ValueError("selected noun base parameters changed")
    evidence = tuple(float(value) for value in record["epoch_validation_nll"])
    if len(evidence) != 2:
        raise ValueError("selected noun base must retain exactly two epoch NLLs")
    return NounSelectedBase(
        root,
        loaded.reference,
        identity,
        preflight.preflight_sha256,
        (evidence[0], evidence[1]),
        int(record["peak_allocator_bytes"]),
    )


def _tokenizer_checkpoint_metadata() -> TokenizerCheckpointMetadata:
    identity = CANONICAL_TOKENIZER_IDENTITY
    return TokenizerCheckpointMetadata(
        kind=identity.kind,
        identifier=identity.identifier,
        revision=identity.revision,
        files=tuple(
            CheckpointFileHash(item.name, item.sha256) for item in identity.files
        ),
    )


def _latest_vamp_stage(
    root: Path,
    task_ids: tuple[str, ...],
) -> Path | None:
    stages = tuple(sorted(path for path in root.glob("stage-*-*") if path.is_dir()))
    expected = tuple(
        f"stage-{index:03d}-{task_id}"
        for index, task_id in enumerate(task_ids[: len(stages)], start=1)
    )
    if tuple(path.name for path in stages) != expected:
        raise ValueError("noun VAMP stage directories are not one canonical prefix")
    return stages[-1] if stages else None


def _load_vamp_stage_record(path: Path) -> dict[str, object]:
    payload = (path / "stage.json").read_bytes()
    record = json.loads(payload)
    if payload != canonical_json_bytes(record) or type(record) is not dict:
        raise ValueError("noun VAMP stage JSON is not canonical")
    supplied = record.pop("stage_sha256", None)
    if supplied != record_sha256(record):
        raise ValueError("noun VAMP stage identity changed")
    return record


def _require_vamp_stage_record(
    record: dict[str, object],
    adaptation: LanguageAdaptationArtifact,
    partition: NounPartitionArtifact,
    preset: NounsExperimentPreset,
    selected_base: NounSelectedBase,
    path: Path,
) -> None:
    expected_fields = {
        "adaptation_manifest_sha256",
        "adaptation_tensor_checksum",
        "base_training_sha256",
        "elapsed_seconds",
        "eligible_node_mask",
        "format",
        "parent_node_id",
        "parent_scores",
        "partition_sha256",
        "preset_sha256",
        "source_story_count",
        "source_window_count",
        "stage_index",
        "task_id",
        "unique_consumed_story_count",
    }
    stage_index = len(adaptation.task_order)
    stage = adaptation.vamp_stages[-1]
    expected_eligibility = (
        (True,) if stage_index == 1 else (False,) + (True,) * (stage_index - 1)
    )
    parent_scores = tuple(float(value) for value in record.get("parent_scores", ()))
    eligibility = tuple(record.get("eligible_node_mask", ()))
    numeric_counts = tuple(
        record.get(field)
        for field in (
            "source_story_count",
            "source_window_count",
            "unique_consumed_story_count",
        )
    )
    elapsed = record.get("elapsed_seconds")
    if (
        set(record) != expected_fields
        or record.get("format")
        != _engine_format(partition, "vamp_stage_format", VAMP_STAGE_FORMAT)
        or record.get("partition_sha256") != partition.partition_sha256
        or record.get("preset_sha256") != preset.config_sha256
        or record.get("base_training_sha256") != selected_base.training_sha256
        or record.get("stage_index") != stage_index
        or record.get("task_id") != str(adaptation.task_order[-1])
        or path.name != f"stage-{stage_index:03d}-{adaptation.task_order[-1]}"
        or record.get("parent_node_id") != str(stage.parent_node_id)
        or parent_scores != stage.parent_mean_node_nll[:stage_index]
        or eligibility != expected_eligibility
        or record.get("adaptation_tensor_checksum") != adaptation.tensor_checksum
        or record.get("adaptation_manifest_sha256")
        != _file_sha256(path / "adaptation" / "manifest.json")
        or type(elapsed) not in (int, float)
        or not math.isfinite(float(elapsed))
        or float(elapsed) <= 0.0
        or any(type(value) is not int or value <= 0 for value in numeric_counts)
        or int(numeric_counts[2]) > int(numeric_counts[0])
    ):
        raise ValueError("noun VAMP stage metadata changed")


def _index_entry(record: object) -> StoryIndexEntry:
    if type(record) is not dict:
        raise TypeError("story index row must be an object")
    return StoryIndexEntry(
        story_id=str(record["story_id"]),
        story_index=int(record["story_index"]),
        story_offset=int(record["story_offset"]),
        byte_length=int(record["byte_length"]),
        token_offset=int(record["token_offset"]),
        token_count=int(record["token_count"]),
    )


def _benchmark_id(artifact: object) -> str:
    value = getattr(artifact, "benchmark_id", BENCHMARK_ID)
    if type(value) is not str or not value:
        raise ValueError("noun artifact benchmark namespace must be nonempty")
    return value


def _engine_format(artifact: object, attribute: str, default: str) -> str:
    value = getattr(artifact, attribute, default)
    if type(value) is not str or not value:
        raise ValueError(f"noun artifact {attribute} must be nonempty")
    return value


def _story_store_path(artifact: object) -> Path:
    return Path(getattr(artifact, "story_store_path", Path(artifact.root) / "stories.bin"))


def _token_store_path(artifact: object) -> Path:
    return Path(getattr(artifact, "token_store_path", Path(artifact.root) / "tokens.uint16"))


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "IndexedStoryStore",
    "IndexedTokenBatchSequence",
    "NounBaseCursor",
    "NounGpuPreflight",
    "NounResourceEstimate",
    "NounSelectedBase",
    "StoryIndexEntry",
    "allocator_peak_bytes",
    "estimate_noun_resources",
    "evaluate_token_weighted_nll",
    "load_story_index",
    "load_noun_vamp_stages",
    "noun_model_config",
    "router_batch_from_index",
    "run_or_load_noun_gpu_preflight",
    "run_or_resume_noun_base",
    "run_or_resume_noun_vamp",
]

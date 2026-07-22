"""Streaming token-weighted scratch training and immutable resume checkpoints."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from apm.data.text.tinyworlds_p.batching import (
    count_partition_microbatches,
    iter_partition_batch_blocks,
)
from apm.data.text.tinyworlds_p.contracts import (
    BASE_TRAINING_PRESET,
    BaseTrainingPreset,
    PartitionArtifact,
    canonical_record_bytes,
)
from apm.data.text.tinyworlds_p.schedule import (
    EpochValidation,
    GridDecision,
    WorldGap,
    calibration_grid_decision,
    epoch_satisfies_gap_gates,
    select_best_eligible_epoch,
)
from apm.lm.config import GptNeoConfig
from apm.lm.gpt_neo import apply_gpt_neo
from apm.lm.losses import per_token_nll
from apm.lm.parameters import GptNeoParams, init_gpt_neo_params
from apm.lm.text_data import TokenBatch
from apm.lm.training import LmTrainState
from apm.lm.training_state_artifact import (
    lm_train_state_checksum,
    load_lm_train_state_artifact,
    write_lm_train_state_artifact,
)


@dataclass(frozen=True, slots=True)
class StreamingTrainingConfig:
    """A validated trainer contract, fixed to archive-v1 values in production."""

    model_config: GptNeoConfig
    epochs: int
    context_length: int
    microbatch_size: int
    accumulation_microbatches: int
    maximum_learning_rate: float
    minimum_learning_rate: float
    warmup_fraction: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    weight_decay: float
    gradient_clip_norm: float
    parameter_seed: int
    state_interval_updates: int
    allocator_peak_limit_bytes: int

    def __post_init__(self) -> None:
        if type(self.model_config) is not GptNeoConfig:
            raise TypeError("model_config must be GptNeoConfig")
        integers = (
            self.epochs,
            self.context_length,
            self.microbatch_size,
            self.accumulation_microbatches,
            self.state_interval_updates,
            self.allocator_peak_limit_bytes,
        )
        if any(type(value) is not int or value <= 0 for value in integers):
            raise ValueError("trainer dimensions and budgets must be positive")
        if type(self.parameter_seed) is not int or self.parameter_seed < 0:
            raise ValueError("parameter seed must be nonnegative")
        if self.context_length > self.model_config.max_position_embeddings:
            raise ValueError("training context exceeds model positions")
        optimizer_values = (
            self.maximum_learning_rate,
            self.minimum_learning_rate,
            self.warmup_fraction,
            self.adam_beta1,
            self.adam_beta2,
            self.adam_epsilon,
            self.weight_decay,
            self.gradient_clip_norm,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in optimizer_values):
            raise ValueError("optimizer values must be finite and positive")
        if not 0.0 < self.minimum_learning_rate < self.maximum_learning_rate:
            raise ValueError("learning-rate bounds are invalid")
        if not 0.0 < self.warmup_fraction < 1.0:
            raise ValueError("warmup_fraction must lie in (0, 1)")
        if not 0.0 < self.adam_beta1 < 1.0 or not 0.0 < self.adam_beta2 < 1.0:
            raise ValueError("Adam betas must lie in (0, 1)")

    @classmethod
    def from_preset(
        cls,
        preset: BaseTrainingPreset = BASE_TRAINING_PRESET,
    ) -> StreamingTrainingConfig:
        """Expand the archive-v1 preset into the executable trainer contract."""
        return cls(
            model_config=preset.model_config,
            epochs=preset.epochs,
            context_length=preset.context_length,
            microbatch_size=preset.microbatch_size,
            accumulation_microbatches=preset.accumulation_microbatches,
            maximum_learning_rate=preset.maximum_learning_rate,
            minimum_learning_rate=preset.minimum_learning_rate,
            warmup_fraction=preset.warmup_fraction,
            adam_beta1=preset.adam_beta1,
            adam_beta2=preset.adam_beta2,
            adam_epsilon=preset.adam_epsilon,
            weight_decay=preset.weight_decay,
            gradient_clip_norm=preset.gradient_clip_norm,
            parameter_seed=preset.parameter_seed,
            state_interval_updates=preset.state_interval_updates,
            allocator_peak_limit_bytes=preset.allocator_peak_limit_bytes,
        )

    def as_record(self) -> dict[str, object]:
        """Return every executable training choice in canonical form."""
        model = self.model_config
        return {
            "accumulation_microbatches": self.accumulation_microbatches,
            "adam_beta1": self.adam_beta1,
            "adam_beta2": self.adam_beta2,
            "adam_epsilon": self.adam_epsilon,
            "allocator_peak_limit_bytes": self.allocator_peak_limit_bytes,
            "context_length": self.context_length,
            "epochs": self.epochs,
            "gradient_clip_norm": self.gradient_clip_norm,
            "maximum_learning_rate": self.maximum_learning_rate,
            "microbatch_size": self.microbatch_size,
            "minimum_learning_rate": self.minimum_learning_rate,
            "model": {
                "activation": model.activation,
                "attention_types": list(model.attention_types),
                "attention_dropout": model.attention_dropout,
                "embedding_dropout": model.embedding_dropout,
                "hidden_size": model.hidden_size,
                "initializer_range": model.initializer_range,
                "intermediate_size": model.intermediate_size,
                "layer_norm_epsilon": model.layer_norm_epsilon,
                "local_window_size": model.local_window_size,
                "max_position_embeddings": model.max_position_embeddings,
                "num_heads": model.num_heads,
                "num_layers": model.num_layers,
                "residual_dropout": model.residual_dropout,
                "tied_embeddings": True,
                "vocab_size": model.vocab_size,
            },
            "parameter_seed": self.parameter_seed,
            "state_interval_updates": self.state_interval_updates,
            "warmup_fraction": self.warmup_fraction,
            "weight_decay": self.weight_decay,
        }


@dataclass(frozen=True, slots=True)
class TrainingCursor:
    """The exact next epoch/block/microbatch and optimizer schedule position."""

    epoch: int
    block: int
    microbatch: int
    optimizer_update: int
    schedule_position: int

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value < 0
            for value in (
                self.epoch,
                self.block,
                self.microbatch,
                self.optimizer_update,
                self.schedule_position,
            )
        ):
            raise ValueError("training cursor positions must be nonnegative")
        if self.optimizer_update != self.schedule_position:
            raise ValueError("optimizer update and schedule position must agree")

    def as_record(self) -> dict[str, int]:
        """Return the canonical persisted cursor."""
        return {
            "block": self.block,
            "epoch": self.epoch,
            "microbatch": self.microbatch,
            "optimizer_update": self.optimizer_update,
            "schedule_position": self.schedule_position,
        }


@dataclass(frozen=True, slots=True)
class TrainingCheckpoint:
    """One complete immutable state and its resume cursor."""

    directory: Path
    cursor: TrainingCursor
    state_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "directory", Path(self.directory))
        if (
            type(self.state_sha256) is not str
            or len(self.state_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.state_sha256)
        ):
            raise ValueError("training checkpoint state hash must be SHA-256")


@dataclass(frozen=True, slots=True)
class StreamingTrainingResult:
    """The final state/cursor, immutable checkpoints, and continuous loss trace."""

    state: LmTrainState[GptNeoParams]
    cursor: TrainingCursor
    checkpoints: tuple[TrainingCheckpoint, ...]
    trace_path: Path
    training_sha256: str
    planned_optimizer_updates: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "trace_path", Path(self.trace_path))
        if not self.trace_path.is_file():
            raise FileNotFoundError(self.trace_path)
        if self.planned_optimizer_updates <= 0:
            raise ValueError("planned optimizer updates must be positive")


class _MicrobatchGradient(NamedTuple):
    loss_sum: jax.Array
    active_tokens: jax.Array
    gradients: GptNeoParams
    next_rng_key: jax.Array


def cosine_learning_rate(
    update: int | jax.Array,
    total_updates: int,
    config: StreamingTrainingConfig,
) -> jax.Array:
    """Apply 1% linear warmup then cosine decay with exact boundary values."""
    if type(total_updates) is not int or total_updates <= 0:
        raise ValueError("total_updates must be positive")
    warmup_updates = max(1, math.ceil(config.warmup_fraction * total_updates))
    position = jnp.asarray(update, dtype=jnp.float32)
    warmup_rate = config.maximum_learning_rate * (position + 1.0) / warmup_updates
    decay_denominator = max(1, total_updates - warmup_updates - 1)
    decay_progress = jnp.clip(
        (position - warmup_updates) / decay_denominator,
        0.0,
        1.0,
    )
    cosine = 0.5 * (1.0 + jnp.cos(jnp.pi * decay_progress))
    decay_rate = config.minimum_learning_rate + (
        config.maximum_learning_rate - config.minimum_learning_rate
    ) * cosine
    return jnp.where(position < warmup_updates, warmup_rate, decay_rate).astype(
        jnp.float32
    )


def init_streaming_train_state(
    config: StreamingTrainingConfig,
    planned_optimizer_updates: int,
) -> LmTrainState[GptNeoParams]:
    """Initialize every float32 model and optimizer value from parameter seed zero."""
    params = init_gpt_neo_params(
        jax.random.PRNGKey(config.parameter_seed),
        config.model_config,
        dtype=jnp.float32,
    )
    optimizer = _optimizer(config, planned_optimizer_updates)
    return LmTrainState(
        trainable=params,
        opt_state=optimizer.init(params),
        rng_key=jax.random.PRNGKey(config.parameter_seed + 1),
        step=jnp.asarray(0, dtype=jnp.int32),
    )


def run_streaming_base_training(
    artifact: PartitionArtifact,
    working_directory: str | Path,
    config: StreamingTrainingConfig | None = None,
    *,
    resume_from: str | Path | None = None,
    stop_after_epoch: int | None = None,
    stop_after_update: int | None = None,
    progress: Callable[[TrainingCursor, float, int], None] | None = None,
) -> StreamingTrainingResult:
    """Train or resume from memory-mapped base shards with token-weighted gradients."""
    training_config = config or StreamingTrainingConfig.from_preset()
    if artifact.preset.context_length != training_config.context_length:
        raise ValueError("partition and trainer context lengths differ")
    if artifact.preset.batch_size != training_config.microbatch_size:
        raise ValueError("partition and trainer microbatch sizes differ")
    if artifact.tokenizer_identity.vocab_size != training_config.model_config.vocab_size:
        raise ValueError("partition tokenizer and model vocabularies differ")
    microbatches_per_epoch = count_partition_microbatches(artifact, "base/train")
    updates_per_epoch = math.ceil(
        microbatches_per_epoch / training_config.accumulation_microbatches
    )
    planned_updates = updates_per_epoch * training_config.epochs
    training_sha256 = sha256(
        canonical_record_bytes(
            {
                "partition_sha256": artifact.partition_sha256,
                "training": training_config.as_record(),
                "updates_per_epoch": updates_per_epoch,
            }
        )
    ).hexdigest()
    working = Path(working_directory)
    working.mkdir(parents=True, exist_ok=True)
    checkpoints_directory = working / "states"
    checkpoints_directory.mkdir(exist_ok=True)
    trace_path = working / "progress.jsonl"
    optimizer = _optimizer(training_config, planned_updates)
    template = init_streaming_train_state(training_config, planned_updates)
    if resume_from is None:
        state = template
        cursor = TrainingCursor(0, 0, 0, 0, 0)
        if trace_path.exists() and trace_path.stat().st_size:
            raise FileExistsError(f"new training trace already exists: {trace_path}")
    else:
        state, cursor = load_streaming_checkpoint(
            resume_from,
            training_sha256,
            template,
        )
        if int(state.step) != cursor.optimizer_update:
            raise ValueError("resume state step and cursor update disagree")
    trace_stream = trace_path.open("ab")
    checkpoints: list[TrainingCheckpoint] = []
    accumulated_gradients = jax.tree_util.tree_map(jnp.zeros_like, state.trainable)
    accumulated_loss = jnp.asarray(0.0, dtype=jnp.float32)
    accumulated_tokens = jnp.asarray(0.0, dtype=jnp.float32)
    accumulated_microbatches = 0

    def microbatch_gradient(
        params: GptNeoParams,
        rng_key: jax.Array,
        batch: TokenBatch,
    ) -> _MicrobatchGradient:
        next_rng_key, dropout_key = jax.random.split(rng_key)

        def loss_sum_function(
            current_params: GptNeoParams,
        ) -> tuple[jax.Array, jax.Array]:
            result = apply_gpt_neo(
                current_params,
                training_config.model_config,
                jnp.asarray(batch.input_ids, dtype=jnp.int32),
                jnp.asarray(batch.attention_mask, dtype=jnp.bool_),
                training=True,
                rng_key=dropout_key,
            )
            mask = jnp.asarray(batch.loss_mask, dtype=jnp.float32)
            losses = per_token_nll(
                result.logits,
                jnp.asarray(batch.target_ids, dtype=jnp.int32),
            )
            return jnp.sum(losses * mask), jnp.sum(mask)

        (loss_sum, active_tokens), gradients = jax.value_and_grad(
            loss_sum_function,
            has_aux=True,
        )(params)
        return _MicrobatchGradient(
            loss_sum,
            active_tokens,
            gradients,
            next_rng_key,
        )

    compiled_gradient = jax.jit(microbatch_gradient)

    def apply_accumulated(
        current_state: LmTrainState[GptNeoParams],
        gradient_sums: GptNeoParams,
        token_count: jax.Array,
    ) -> LmTrainState[GptNeoParams]:
        normalized = jax.tree_util.tree_map(
            lambda gradient: gradient / token_count,
            gradient_sums,
        )
        updates, next_opt_state = optimizer.update(
            normalized,
            current_state.opt_state,
            current_state.trainable,
        )
        return LmTrainState(
            trainable=optax.apply_updates(current_state.trainable, updates),
            opt_state=next_opt_state,
            rng_key=current_state.rng_key,
            step=current_state.step + jnp.asarray(1, dtype=jnp.int32),
        )

    compiled_update = jax.jit(apply_accumulated)
    try:
        for epoch in range(cursor.epoch, training_config.epochs):
            start_block = cursor.block if epoch == cursor.epoch else 0
            start_microbatch = cursor.microbatch if epoch == cursor.epoch else 0
            for block in iter_partition_batch_blocks(artifact, "base/train", epoch):
                if block.shuffled_block < start_block:
                    continue
                for microbatch_index, batch in enumerate(block.batches):
                    if (
                        block.shuffled_block == start_block
                        and microbatch_index < start_microbatch
                    ):
                        continue
                    gradient = compiled_gradient(state.trainable, state.rng_key, batch)
                    state = replace(state, rng_key=gradient.next_rng_key)
                    accumulated_gradients = jax.tree_util.tree_map(
                        jnp.add,
                        accumulated_gradients,
                        gradient.gradients,
                    )
                    accumulated_loss += gradient.loss_sum
                    accumulated_tokens += gradient.active_tokens
                    accumulated_microbatches += 1
                    cursor = TrainingCursor(
                        epoch=epoch,
                        block=block.shuffled_block,
                        microbatch=microbatch_index + 1,
                        optimizer_update=int(state.step),
                        schedule_position=int(state.step),
                    )
                    if accumulated_microbatches == training_config.accumulation_microbatches:
                        state = compiled_update(
                            state,
                            accumulated_gradients,
                            accumulated_tokens,
                        )
                        cursor = replace(
                            cursor,
                            optimizer_update=int(state.step),
                            schedule_position=int(state.step),
                        )
                        update_nll = _append_update_trace(
                            trace_stream,
                            cursor,
                            accumulated_loss,
                            accumulated_tokens,
                            training_config,
                            planned_updates,
                        )
                        if progress is not None:
                            progress(cursor, update_nll, planned_updates)
                        accumulated_gradients = jax.tree_util.tree_map(
                            jnp.zeros_like,
                            accumulated_gradients,
                        )
                        accumulated_loss = jnp.asarray(0.0, dtype=jnp.float32)
                        accumulated_tokens = jnp.asarray(0.0, dtype=jnp.float32)
                        accumulated_microbatches = 0
                        if int(state.step) % training_config.state_interval_updates == 0:
                            checkpoints.append(
                                write_streaming_checkpoint(
                                    checkpoints_directory
                                    / f"update-{int(state.step):09d}",
                                    training_sha256,
                                    state,
                                    cursor,
                                )
                            )
                        if stop_after_update is not None and int(state.step) >= stop_after_update:
                            checkpoint = write_streaming_checkpoint(
                                checkpoints_directory
                                / f"interrupted-{int(state.step):09d}",
                                training_sha256,
                                state,
                                cursor,
                            )
                            checkpoints.append(checkpoint)
                            return StreamingTrainingResult(
                                state,
                                cursor,
                                tuple(checkpoints),
                                trace_path,
                                training_sha256,
                                planned_updates,
                            )
            if accumulated_microbatches:
                state = compiled_update(state, accumulated_gradients, accumulated_tokens)
                cursor = TrainingCursor(
                    epoch=epoch + 1,
                    block=0,
                    microbatch=0,
                    optimizer_update=int(state.step),
                    schedule_position=int(state.step),
                )
                update_nll = _append_update_trace(
                    trace_stream,
                    cursor,
                    accumulated_loss,
                    accumulated_tokens,
                    training_config,
                    planned_updates,
                )
                if progress is not None:
                    progress(cursor, update_nll, planned_updates)
                accumulated_gradients = jax.tree_util.tree_map(
                    jnp.zeros_like,
                    accumulated_gradients,
                )
                accumulated_loss = jnp.asarray(0.0, dtype=jnp.float32)
                accumulated_tokens = jnp.asarray(0.0, dtype=jnp.float32)
                accumulated_microbatches = 0
            else:
                cursor = TrainingCursor(
                    epoch=epoch + 1,
                    block=0,
                    microbatch=0,
                    optimizer_update=int(state.step),
                    schedule_position=int(state.step),
                )
            checkpoints.append(
                write_streaming_checkpoint(
                    checkpoints_directory
                    / f"epoch-{epoch + 1:02d}-update-{int(state.step):09d}",
                    training_sha256,
                    state,
                    cursor,
                )
            )
            if stop_after_epoch is not None and epoch + 1 >= stop_after_epoch:
                break
    finally:
        trace_stream.flush()
        os.fsync(trace_stream.fileno())
        trace_stream.close()
    return StreamingTrainingResult(
        state=state,
        cursor=cursor,
        checkpoints=tuple(checkpoints),
        trace_path=trace_path,
        training_sha256=training_sha256,
        planned_optimizer_updates=planned_updates,
    )


def write_streaming_checkpoint(
    directory: str | Path,
    training_sha256: str,
    state: LmTrainState[GptNeoParams],
    cursor: TrainingCursor,
) -> TrainingCheckpoint:
    """Persist complete model/optimizer/RNG state and an exact next-batch cursor."""
    target = Path(directory)
    if target.exists():
        raise FileExistsError(f"training checkpoint already exists: {target}")
    target.mkdir(parents=True)
    write_lm_train_state_artifact(target / "state", training_sha256, (state,))
    state_sha256 = lm_train_state_checksum(state)
    resume_path = target / "resume.json"
    _write_file(
        resume_path,
        canonical_record_bytes(
            {
                "cursor": cursor.as_record(),
                "format": "tinyworlds-p-archive-training-resume",
                "state_sha256": state_sha256,
                "training_sha256": training_sha256,
                "version": 1,
            }
        ),
    )
    return TrainingCheckpoint(target, cursor, state_sha256)


def load_streaming_checkpoint(
    directory: str | Path,
    training_sha256: str,
    template: LmTrainState[GptNeoParams],
) -> tuple[LmTrainState[GptNeoParams], TrainingCursor]:
    """Strictly restore complete state and validate its resume cursor binding."""
    root = Path(directory)
    if root.is_symlink() or not root.is_dir() or {path.name for path in root.iterdir()} != {
        "resume.json",
        "state",
    }:
        raise ValueError("training checkpoint entries are not canonical")
    resume_payload = (root / "resume.json").read_bytes()
    try:
        resume = json.loads(resume_payload)
    except (json.JSONDecodeError, UnicodeError) as error:
        raise ValueError("invalid training resume manifest") from error
    if type(resume) is not dict or canonical_record_bytes(resume) != resume_payload:
        raise ValueError("training resume manifest is not canonical")
    if set(resume) != {
        "cursor",
        "format",
        "state_sha256",
        "training_sha256",
        "version",
    }:
        raise ValueError("training resume fields changed")
    if (
        resume["format"] != "tinyworlds-p-archive-training-resume"
        or resume["version"] != 1
        or resume["training_sha256"] != training_sha256
    ):
        raise ValueError("training resume identity changed")
    cursor_record = resume["cursor"]
    if type(cursor_record) is not dict or set(cursor_record) != {
        "block",
        "epoch",
        "microbatch",
        "optimizer_update",
        "schedule_position",
    }:
        raise ValueError("training resume cursor fields changed")
    cursor = TrainingCursor(
        epoch=_strict_integer(cursor_record, "epoch"),
        block=_strict_integer(cursor_record, "block"),
        microbatch=_strict_integer(cursor_record, "microbatch"),
        optimizer_update=_strict_integer(cursor_record, "optimizer_update"),
        schedule_position=_strict_integer(cursor_record, "schedule_position"),
    )
    state = load_lm_train_state_artifact(
        root / "state",
        training_sha256,
        (template,),
    )[0]
    if (
        lm_train_state_checksum(state) != resume["state_sha256"]
        or int(state.step) != cursor.optimizer_update
    ):
        raise ValueError("training resume state checksum or update changed")
    return state, cursor


def allocator_peak_bytes() -> int:
    """Return the largest JAX device allocator peak currently reported."""
    peaks = tuple(
        int(statistics.get("peak_bytes_in_use", 0))
        for device in jax.devices()
        if (statistics := device.memory_stats()) is not None
    )
    return max(peaks, default=0)


def _optimizer(
    config: StreamingTrainingConfig,
    total_updates: int,
) -> optax.GradientTransformation:
    schedule = lambda update: cosine_learning_rate(update, total_updates, config)
    return optax.chain(
        optax.clip_by_global_norm(config.gradient_clip_norm),
        optax.adamw(
            learning_rate=schedule,
            b1=config.adam_beta1,
            b2=config.adam_beta2,
            eps=config.adam_epsilon,
            weight_decay=config.weight_decay,
        ),
    )


def _append_update_trace(
    stream,
    cursor: TrainingCursor,
    loss_sum: jax.Array,
    active_tokens: jax.Array,
    config: StreamingTrainingConfig,
    planned_updates: int,
) -> float:
    token_count = float(active_tokens)
    if token_count <= 0.0:
        raise ValueError("optimizer update accumulated no active tokens")
    learning_rate = float(
        cosine_learning_rate(cursor.optimizer_update - 1, planned_updates, config)
    )
    normalized_nll = float(loss_sum) / token_count
    stream.write(
        canonical_record_bytes(
            {
                "active_tokens": int(token_count),
                "block": cursor.block,
                "epoch": cursor.epoch,
                "learning_rate": learning_rate,
                "microbatch": cursor.microbatch,
                "nll": normalized_nll,
                "optimizer_update": cursor.optimizer_update,
                "schedule_position": cursor.schedule_position,
            }
        )
    )
    stream.flush()
    return normalized_nll


def _strict_integer(record: dict[str, object], field: str) -> int:
    value = record.get(field)
    if type(value) is not int or value < 0:
        raise ValueError(f"training cursor {field} must be nonnegative")
    return value


def _write_file(path: Path, payload: bytes) -> None:
    with path.open("wb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())


__all__ = [
    "EpochValidation",
    "GridDecision",
    "StreamingTrainingConfig",
    "StreamingTrainingResult",
    "TrainingCheckpoint",
    "TrainingCursor",
    "WorldGap",
    "allocator_peak_bytes",
    "calibration_grid_decision",
    "cosine_learning_rate",
    "epoch_satisfies_gap_gates",
    "init_streaming_train_state",
    "load_streaming_checkpoint",
    "run_streaming_base_training",
    "select_best_eligible_epoch",
    "write_streaming_checkpoint",
]

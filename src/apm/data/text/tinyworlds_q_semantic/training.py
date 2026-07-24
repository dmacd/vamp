"""Query-native scratch-base training, validation, and exact resume state."""

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

from apm.data.text.tinyworlds_q_semantic.batching import (
    count_query_partition_microbatches,
    iter_query_partition_batch_blocks,
    iter_query_partition_batches,
)
from apm.data.text.tinyworlds_q_semantic.contracts import (
    BENCHMARK_ID,
    QueryExperimentPreset,
    QueryPartitionArtifact,
    canonical_json_bytes,
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


QUERY_BASE_RESUME_FORMAT = "tinyworlds-q-semantic-base-training-resume-v1"

TrainingProgress = Callable[["QueryTrainingCursor", float, int], None]
EvaluationProgress = Callable[[str, int, int], None]


@dataclass(frozen=True, slots=True)
class QueryBaseTrainingConfig:
    """Complete executable base-training contract, with test-size overrides."""

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
            raise TypeError("query base model_config must be GptNeoConfig")
        positive_integers = (
            self.epochs,
            self.context_length,
            self.microbatch_size,
            self.accumulation_microbatches,
            self.state_interval_updates,
            self.allocator_peak_limit_bytes,
        )
        if any(type(value) is not int or value <= 0 for value in positive_integers):
            raise ValueError("query base dimensions and limits must be positive")
        if type(self.parameter_seed) is not int or self.parameter_seed < 0:
            raise ValueError("query base parameter seed must be nonnegative")
        if self.context_length > self.model_config.max_position_embeddings:
            raise ValueError("query base context exceeds model positions")
        positive_floats = (
            self.maximum_learning_rate,
            self.minimum_learning_rate,
            self.warmup_fraction,
            self.adam_beta1,
            self.adam_beta2,
            self.adam_epsilon,
            self.weight_decay,
            self.gradient_clip_norm,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive_floats):
            raise ValueError("query base optimizer values must be finite and positive")
        if not 0.0 < self.minimum_learning_rate < self.maximum_learning_rate:
            raise ValueError("query base learning-rate bounds are invalid")
        if not 0.0 < self.warmup_fraction < 1.0:
            raise ValueError("query base warmup_fraction must lie in (0, 1)")
        if not 0.0 < self.adam_beta1 < 1.0 or not 0.0 < self.adam_beta2 < 1.0:
            raise ValueError("query base Adam betas must lie in (0, 1)")

    @classmethod
    def from_preset(
        cls,
        preset: QueryExperimentPreset,
    ) -> QueryBaseTrainingConfig:
        """Expand the registered query experiment into its executable trainer."""
        return cls(
            model_config=preset.model_config,
            epochs=preset.base_epochs,
            context_length=preset.context_length,
            microbatch_size=preset.microbatch_size,
            accumulation_microbatches=preset.accumulation_microbatches,
            maximum_learning_rate=preset.maximum_learning_rate,
            minimum_learning_rate=preset.minimum_learning_rate,
            warmup_fraction=preset.warmup_fraction,
            adam_beta1=preset.adam_beta1,
            adam_beta2=preset.adam_beta2,
            adam_epsilon=preset.adam_epsilon,
            weight_decay=preset.base_weight_decay,
            gradient_clip_norm=preset.base_gradient_clip_norm,
            parameter_seed=preset.seed,
            state_interval_updates=preset.base_state_interval_updates,
            allocator_peak_limit_bytes=preset.allocator_peak_limit_bytes,
        )

    def as_record(self) -> dict[str, object]:
        """Return every model and optimizer choice in canonical form."""
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
                "attention_dropout": model.attention_dropout,
                "attention_types": list(model.attention_types),
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
class QueryTrainingCursor:
    """The exact next epoch, block, microbatch, and optimizer position."""

    epoch: int
    block: int
    microbatch: int
    optimizer_update: int
    schedule_position: int

    def __post_init__(self) -> None:
        values = (
            self.epoch,
            self.block,
            self.microbatch,
            self.optimizer_update,
            self.schedule_position,
        )
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("query training cursor positions must be nonnegative")
        if self.optimizer_update != self.schedule_position:
            raise ValueError("query optimizer update and schedule position must agree")

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
class QueryTrainingCheckpoint:
    """One complete immutable query-base state and resume cursor."""

    directory: Path
    cursor: QueryTrainingCursor
    state_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "directory", Path(self.directory))
        if (
            type(self.state_sha256) is not str
            or len(self.state_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.state_sha256)
        ):
            raise ValueError("query training state hash must be lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class QueryBaseTrainingResult:
    """Current base state plus checkpoints and the append-only loss ledger."""

    state: LmTrainState[GptNeoParams]
    cursor: QueryTrainingCursor
    checkpoints: tuple[QueryTrainingCheckpoint, ...]
    trace_path: Path
    training_sha256: str
    planned_optimizer_updates: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "trace_path", Path(self.trace_path))
        if not self.trace_path.is_file():
            raise FileNotFoundError(self.trace_path)
        if self.planned_optimizer_updates <= 0:
            raise ValueError("query base training must plan at least one update")


@dataclass(frozen=True, slots=True)
class QuerySplitNll:
    """Token-weighted NLL for one base validation or test selector."""

    split: str
    active_tokens: int
    nll: float

    def __post_init__(self) -> None:
        if self.split not in ("validation", "test"):
            raise ValueError("query base evaluation split must be validation or test")
        if type(self.active_tokens) is not int or self.active_tokens <= 0:
            raise ValueError("query base evaluation requires active tokens")
        if not math.isfinite(self.nll) or self.nll < 0.0:
            raise ValueError("query base NLL must be finite and nonnegative")


class _MicrobatchGradient(NamedTuple):
    loss_sum: jax.Array
    active_tokens: jax.Array
    gradients: GptNeoParams
    next_rng_key: jax.Array


def query_cosine_learning_rate(
    update: int | jax.Array,
    total_updates: int,
    config: QueryBaseTrainingConfig,
) -> jax.Array:
    """Apply the registered one-percent warmup and cosine decay."""
    if type(total_updates) is not int or total_updates <= 0:
        raise ValueError("query base total_updates must be positive")
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


def init_query_base_train_state(
    config: QueryBaseTrainingConfig,
    planned_optimizer_updates: int,
) -> LmTrainState[GptNeoParams]:
    """Initialize fresh float32 GPT-Neo and AdamW state from seed zero."""
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


def run_query_base_training(
    artifact: QueryPartitionArtifact,
    preset: QueryExperimentPreset,
    working_directory: str | Path,
    config: QueryBaseTrainingConfig | None = None,
    *,
    resume_from: str | Path | None = None,
    stop_after_epoch: int | None = None,
    stop_after_update: int | None = None,
    progress: TrainingProgress | None = None,
) -> QueryBaseTrainingResult:
    """Train or resume the seed-zero base from query-native indexed documents."""
    training_config = config or QueryBaseTrainingConfig.from_preset(preset)
    _require_training_bindings(artifact, preset, training_config)
    training_sha256, planned_updates = query_base_training_identity(
        artifact,
        preset,
        training_config,
    )
    if stop_after_epoch is not None and (
        type(stop_after_epoch) is not int
        or not 1 <= stop_after_epoch <= training_config.epochs
    ):
        raise ValueError("query base stop_after_epoch lies outside training")
    if stop_after_update is not None and (
        type(stop_after_update) is not int
        or not 1 <= stop_after_update <= planned_updates
    ):
        raise ValueError("query base stop_after_update lies outside training")

    working = Path(working_directory)
    working.mkdir(parents=True, exist_ok=True)
    print(f"TinyWorlds-Q base training artifacts: {working.resolve()}", flush=True)
    checkpoints_directory = working / "states"
    checkpoints_directory.mkdir(exist_ok=True)
    trace_path = working / "progress.jsonl"
    optimizer = _optimizer(training_config, planned_updates)
    template = init_query_base_train_state(training_config, planned_updates)
    if resume_from is None:
        state = template
        cursor = QueryTrainingCursor(0, 0, 0, 0, 0)
        if trace_path.exists() and trace_path.stat().st_size:
            raise FileExistsError(f"new query training trace already exists: {trace_path}")
    else:
        state, cursor = load_query_training_checkpoint(
            resume_from,
            training_sha256,
            template,
        )
        _trim_trace_to_cursor(trace_path, cursor)
        if stop_after_update is not None and stop_after_update < cursor.optimizer_update:
            raise ValueError("query base stop update precedes the resume checkpoint")

    checkpoints: list[QueryTrainingCheckpoint] = []
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
    trace_stream = trace_path.open("ab")
    try:
        for epoch in range(cursor.epoch, training_config.epochs):
            start_block = cursor.block if epoch == cursor.epoch else 0
            start_microbatch = cursor.microbatch if epoch == cursor.epoch else 0
            for block in iter_query_partition_batch_blocks(
                artifact,
                preset,
                role="base",
                split="train",
                epoch=epoch,
                context_length=training_config.context_length,
                microbatch_size=training_config.microbatch_size,
            ):
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
                    cursor = QueryTrainingCursor(
                        epoch,
                        block.shuffled_block,
                        microbatch_index + 1,
                        int(state.step),
                        int(state.step),
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
                                write_query_training_checkpoint(
                                    checkpoints_directory / f"update-{int(state.step):09d}",
                                    training_sha256,
                                    state,
                                    cursor,
                                )
                            )
                        if stop_after_update is not None and int(state.step) >= stop_after_update:
                            checkpoints.append(
                                write_query_training_checkpoint(
                                    checkpoints_directory / f"interrupted-{int(state.step):09d}",
                                    training_sha256,
                                    state,
                                    cursor,
                                )
                            )
                            return QueryBaseTrainingResult(
                                state,
                                cursor,
                                tuple(checkpoints),
                                trace_path,
                                training_sha256,
                                planned_updates,
                            )
            if accumulated_microbatches:
                state = compiled_update(state, accumulated_gradients, accumulated_tokens)
                cursor = QueryTrainingCursor(
                    epoch + 1,
                    0,
                    0,
                    int(state.step),
                    int(state.step),
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
                cursor = QueryTrainingCursor(
                    epoch + 1,
                    0,
                    0,
                    int(state.step),
                    int(state.step),
                )
            checkpoints.append(
                write_query_training_checkpoint(
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
    return QueryBaseTrainingResult(
        state,
        cursor,
        tuple(checkpoints),
        trace_path,
        training_sha256,
        planned_updates,
    )


def evaluate_query_base_nll(
    params: GptNeoParams,
    artifact: QueryPartitionArtifact,
    preset: QueryExperimentPreset,
    split: str,
    config: QueryBaseTrainingConfig | None = None,
    *,
    progress: EvaluationProgress | None = None,
) -> QuerySplitNll:
    """Stream held-in base NLL without ever using adapter or query-test data."""
    if split not in ("validation", "test"):
        raise ValueError("query base evaluation split must be validation or test")
    training_config = config or QueryBaseTrainingConfig.from_preset(preset)
    _require_training_bindings(artifact, preset, training_config)

    def evaluate_batch(batch: TokenBatch) -> tuple[jax.Array, jax.Array]:
        result = apply_gpt_neo(
            params,
            training_config.model_config,
            jnp.asarray(batch.input_ids, dtype=jnp.int32),
            jnp.asarray(batch.attention_mask, dtype=jnp.bool_),
        )
        mask = jnp.asarray(batch.loss_mask, dtype=jnp.float32)
        losses = per_token_nll(
            result.logits,
            jnp.asarray(batch.target_ids, dtype=jnp.int32),
        )
        return jnp.sum(losses * mask), jnp.sum(mask)

    compiled = jax.jit(evaluate_batch)
    planned_batches = count_query_partition_microbatches(
        artifact,
        preset,
        role="base",
        split=split,
        context_length=training_config.context_length,
        microbatch_size=training_config.microbatch_size,
    )
    total_loss = 0.0
    active_tokens = 0
    for completed, batch in enumerate(
        iter_query_partition_batches(
            artifact,
            preset,
            role="base",
            split=split,
            epoch=0,
            context_length=training_config.context_length,
            microbatch_size=training_config.microbatch_size,
        ),
        start=1,
    ):
        loss_sum, token_count = compiled(batch)
        total_loss += float(loss_sum)
        active_tokens += int(token_count)
        if progress is not None:
            progress(split, completed, planned_batches)
    if active_tokens <= 0:
        raise ValueError(f"query base {split} split contains no active tokens")
    return QuerySplitNll(split, active_tokens, total_loss / active_tokens)


def query_base_training_identity(
    artifact: QueryPartitionArtifact,
    preset: QueryExperimentPreset,
    config: QueryBaseTrainingConfig | None = None,
) -> tuple[str, int]:
    """Return the exact resume identity and planned update count without training."""
    training_config = config or QueryBaseTrainingConfig.from_preset(preset)
    _require_training_bindings(artifact, preset, training_config)
    microbatches_per_epoch = count_query_partition_microbatches(
        artifact,
        preset,
        role="base",
        split="train",
        context_length=training_config.context_length,
        microbatch_size=training_config.microbatch_size,
    )
    updates_per_epoch = math.ceil(
        microbatches_per_epoch / training_config.accumulation_microbatches
    )
    planned_updates = updates_per_epoch * training_config.epochs
    identity = sha256(
        canonical_json_bytes(
            {
                "benchmark_id": BENCHMARK_ID,
                "format": QUERY_BASE_RESUME_FORMAT,
                "partition_sha256": artifact.partition_sha256,
                "training": training_config.as_record(),
                "updates_per_epoch": updates_per_epoch,
            }
        )
    ).hexdigest()
    return identity, planned_updates


def write_query_training_checkpoint(
    directory: str | Path,
    training_sha256: str,
    state: LmTrainState[GptNeoParams],
    cursor: QueryTrainingCursor,
) -> QueryTrainingCheckpoint:
    """Persist complete model, optimizer, RNG, schedule, and next-batch state."""
    target = Path(directory)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"query training checkpoint already exists: {target}")
    target.mkdir(parents=True)
    write_lm_train_state_artifact(target / "state", training_sha256, (state,))
    state_sha256 = lm_train_state_checksum(state)
    _write_file(
        target / "resume.json",
        canonical_json_bytes(
            {
                "cursor": cursor.as_record(),
                "format": QUERY_BASE_RESUME_FORMAT,
                "state_sha256": state_sha256,
                "training_sha256": training_sha256,
                "version": 1,
            }
        ),
    )
    return QueryTrainingCheckpoint(target.resolve(), cursor, state_sha256)


def load_query_training_checkpoint(
    directory: str | Path,
    training_sha256: str,
    template: LmTrainState[GptNeoParams],
) -> tuple[LmTrainState[GptNeoParams], QueryTrainingCursor]:
    """Strictly restore one query-native state and its exact cursor binding."""
    root = Path(directory)
    if root.is_symlink() or not root.is_dir() or {
        path.name for path in root.iterdir()
    } != {"resume.json", "state"}:
        raise ValueError("query training checkpoint entries are not canonical")
    resume_payload = (root / "resume.json").read_bytes()
    try:
        resume = json.loads(resume_payload)
    except (json.JSONDecodeError, UnicodeError) as error:
        raise ValueError("invalid query training resume manifest") from error
    if type(resume) is not dict or canonical_json_bytes(resume) != resume_payload:
        raise ValueError("query training resume manifest is not canonical")
    if set(resume) != {
        "cursor",
        "format",
        "state_sha256",
        "training_sha256",
        "version",
    } or (
        resume.get("format") != QUERY_BASE_RESUME_FORMAT
        or resume.get("version") != 1
        or resume.get("training_sha256") != training_sha256
    ):
        raise ValueError("query training resume identity changed")
    cursor_record = resume.get("cursor")
    if type(cursor_record) is not dict or set(cursor_record) != {
        "block",
        "epoch",
        "microbatch",
        "optimizer_update",
        "schedule_position",
    }:
        raise ValueError("query training resume cursor fields changed")
    cursor = QueryTrainingCursor(
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
        lm_train_state_checksum(state) != resume.get("state_sha256")
        or int(state.step) != cursor.optimizer_update
    ):
        raise ValueError("query training state checksum or update changed")
    return state, cursor


def allocator_peak_bytes() -> int:
    """Return the largest JAX device allocator peak currently reported."""
    peaks = tuple(
        int(statistics.get("peak_bytes_in_use", 0))
        for device in jax.devices()
        if (statistics := device.memory_stats()) is not None
    )
    return max(peaks, default=0)


def _require_training_bindings(
    artifact: QueryPartitionArtifact,
    preset: QueryExperimentPreset,
    config: QueryBaseTrainingConfig,
) -> None:
    if artifact.concept_ids[: preset.active_world_count] != preset.concept_ids:
        raise ValueError("query base preset is not an active partition prefix")
    if artifact.tokenizer_identity.vocab_size != config.model_config.vocab_size:
        raise ValueError("query base model vocabulary differs from the tokenizer")


def _optimizer(
    config: QueryBaseTrainingConfig,
    total_updates: int,
) -> optax.GradientTransformation:
    schedule = lambda update: query_cosine_learning_rate(update, total_updates, config)
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
    cursor: QueryTrainingCursor,
    loss_sum: jax.Array,
    active_tokens: jax.Array,
    config: QueryBaseTrainingConfig,
    planned_updates: int,
) -> float:
    token_count = float(active_tokens)
    if token_count <= 0.0:
        raise ValueError("query optimizer update accumulated no active tokens")
    learning_rate = float(
        query_cosine_learning_rate(
            cursor.optimizer_update - 1,
            planned_updates,
            config,
        )
    )
    normalized_nll = float(loss_sum) / token_count
    stream.write(
        canonical_json_bytes(
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


def _trim_trace_to_cursor(path: Path, cursor: QueryTrainingCursor) -> None:
    if not path.exists():
        if cursor.optimizer_update:
            raise ValueError("query resume trace is missing completed updates")
        return
    retained: list[bytes] = []
    removed: list[bytes] = []
    with path.open("rb") as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except (UnicodeError, json.JSONDecodeError) as error:
                raise ValueError("query training trace contains invalid JSON") from error
            if type(record) is not dict or canonical_json_bytes(record) != line:
                raise ValueError("query training trace is not canonical")
            update = _strict_integer(record, "optimizer_update")
            (retained if update <= cursor.optimizer_update else removed).append(line)
    if tuple(_strict_integer(json.loads(line), "optimizer_update") for line in retained) != tuple(
        range(1, cursor.optimizer_update + 1)
    ):
        raise ValueError("query training trace does not match the resume cursor")
    if not removed:
        return
    recovery = path.parent / "recovery"
    recovery.mkdir(exist_ok=True)
    recovery_path = recovery / f"progress-after-{cursor.optimizer_update:09d}.jsonl"
    if recovery_path.exists():
        raise FileExistsError("query training trace recovery already exists")
    _write_file(recovery_path, b"".join(removed))
    temporary = path.with_name(f".{path.name}.trimmed")
    _write_file(temporary, b"".join(retained))
    os.replace(temporary, path)


def _strict_integer(record: dict[str, object], field: str) -> int:
    value = record.get(field)
    if type(value) is not int or value < 0:
        raise ValueError(f"query training {field} must be nonnegative")
    return value


def _write_file(path: Path, payload: bytes) -> None:
    with path.open("wb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())


__all__ = [
    "QUERY_BASE_RESUME_FORMAT",
    "QueryBaseTrainingConfig",
    "QueryBaseTrainingResult",
    "QuerySplitNll",
    "QueryTrainingCheckpoint",
    "QueryTrainingCursor",
    "allocator_peak_bytes",
    "evaluate_query_base_nll",
    "init_query_base_train_state",
    "load_query_training_checkpoint",
    "query_cosine_learning_rate",
    "query_base_training_identity",
    "run_query_base_training",
    "write_query_training_checkpoint",
]

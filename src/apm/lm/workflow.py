"""Deterministic immutable workflows and presets for language-model training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from apm.lm.config import GptNeoConfig
from apm.lm.gpt_neo import apply_gpt_neo
from apm.lm.lora import LoraConfig, LoraEdge
from apm.lm.lora_memory import PackedLoraMemory
from apm.lm.losses import per_token_nll
from apm.lm.parameters import GptNeoParams
from apm.lm.text_data import TokenBatch
from apm.lm.training import (
    LmTrainConfig,
    LmTrainState,
    base_train_step,
    candidate_lora_train_step,
)


@dataclass(frozen=True)
class LmLossTrace:
    """Immutable ordered scalar losses produced by a fixed update budget."""

    step_losses: tuple[float, ...]


class CandidateValidationFunction(Protocol):
    """Validation callback evaluated only at immutable checkpoint updates."""

    def __call__(self, adapter: LoraEdge, update: int) -> tuple[float, float]:
        """Return candidate accuracy and correct-answer NLL."""
        ...


@dataclass(frozen=True)
class CandidateTrainingCheckpoint:
    """One resumable optimizer state and its validation evidence."""

    update: int
    state: LmTrainState[LoraEdge]
    training_loss: float | None
    validation_candidate_accuracy: float | None
    validation_correct_nll: float | None

    def __post_init__(self) -> None:
        if type(self.update) is not int or self.update < 0:
            raise ValueError("candidate checkpoint update must be nonnegative")
        if not isinstance(self.state, LmTrainState):
            raise TypeError("candidate checkpoint state must be an LmTrainState")
        if int(self.state.step) != self.update:
            raise ValueError("candidate checkpoint state step must equal update")
        if self.update == 0 and self.training_loss is not None:
            raise ValueError("update-zero candidate checkpoints have no training loss")
        if self.training_loss is not None and not np.isfinite(self.training_loss):
            raise ValueError("candidate checkpoint training loss must be finite")
        validation_values = (
            self.validation_candidate_accuracy,
            self.validation_correct_nll,
        )
        if (validation_values[0] is None) != (validation_values[1] is None):
            raise ValueError("candidate validation accuracy and NLL are paired")
        if validation_values[0] is not None and (
            not 0.0 <= validation_values[0] <= 1.0
            or validation_values[1] is None
            or not np.isfinite(validation_values[1])
            or validation_values[1] < 0.0
        ):
            raise ValueError("candidate validation values are outside their domains")


def tiny_shakespeare_model_config(vocab_size: int) -> GptNeoConfig:
    """Return the standard TinyShakespeare GPT-Neo smoke configuration."""
    return GptNeoConfig(
        vocab_size=vocab_size,
        max_position_embeddings=256,
        hidden_size=128,
        intermediate_size=512,
        num_layers=4,
        num_heads=4,
        attention_types=("global", "local", "global", "local"),
        local_window_size=64,
        embedding_dropout=0.0,
        attention_dropout=0.0,
        residual_dropout=0.0,
    )


def tiny_shakespeare_unit_model_config(vocab_size: int) -> GptNeoConfig:
    """Return the smaller CPU unit-test GPT-Neo configuration."""
    return GptNeoConfig(
        vocab_size=vocab_size,
        max_position_embeddings=64,
        hidden_size=64,
        intermediate_size=256,
        num_layers=2,
        num_heads=4,
        attention_types=("global", "local"),
        local_window_size=64,
        embedding_dropout=0.0,
        attention_dropout=0.0,
        residual_dropout=0.0,
    )


def evaluate_normalized_nll(
    params: GptNeoParams,
    model_config: GptNeoConfig,
    batches: Sequence[TokenBatch],
    *,
    lora_memory: PackedLoraMemory | None = None,
    edge_coefficients: jax.Array | None = None,
    lora_config: LoraConfig | None = None,
) -> float:
    """Evaluate total NLL normalized over all active tokens in all batches."""
    if not batches:
        raise ValueError("NLL evaluation requires at least one TokenBatch")

    def evaluate_batch(batch: TokenBatch) -> tuple[jax.Array, jax.Array]:
        result = apply_gpt_neo(
            params,
            model_config,
            jnp.asarray(batch.input_ids, dtype=jnp.int32),
            jnp.asarray(batch.attention_mask, dtype=jnp.bool_),
            lora_memory=lora_memory,
            edge_coefficients=edge_coefficients,
            lora_config=lora_config,
        )
        mask = jnp.asarray(batch.loss_mask, dtype=jnp.float32)
        losses = per_token_nll(
            result.logits,
            jnp.asarray(batch.target_ids, dtype=jnp.int32),
        )
        return jnp.sum(losses * mask), jnp.sum(mask)

    compiled_evaluation = jax.jit(evaluate_batch)
    totals = tuple(compiled_evaluation(batch) for batch in batches)
    total_tokens = sum(float(token_count) for _, token_count in totals)
    if total_tokens == 0.0:
        raise ValueError("NLL evaluation requires at least one active loss token")
    return sum(float(total_loss) for total_loss, _ in totals) / total_tokens


def run_base_updates(
    state: LmTrainState[GptNeoParams],
    batches: Sequence[TokenBatch],
    model_config: GptNeoConfig,
    train_config: LmTrainConfig,
) -> tuple[LmTrainState[GptNeoParams], LmLossTrace]:
    """Run exactly the configured base updates while cycling batches in order."""
    _require_training_batches(batches)
    compiled_step = jax.jit(
        lambda current_state, batch: base_train_step(
            current_state,
            batch,
            model_config,
            train_config,
        )
    )
    current_state = state
    losses: list[float] = []
    for step_index in range(train_config.steps):
        current_state, loss = compiled_step(
            current_state,
            batches[step_index % len(batches)],
        )
        losses.append(float(loss))
    return current_state, LmLossTrace(step_losses=tuple(losses))


def run_candidate_edge_updates(
    state: LmTrainState[LoraEdge],
    batches: Sequence[TokenBatch],
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    packed_memory: PackedLoraMemory,
    lora_config: LoraConfig,
    parent_edge_coefficients: jax.Array,
    candidate_index: int,
    train_config: LmTrainConfig,
    *,
    progress: Callable[[int, float, int], None] | None = None,
) -> tuple[LmTrainState[LoraEdge], LmLossTrace]:
    """Run exactly the configured candidate-edge updates while cycling batches."""
    _require_training_batches(batches)
    compiled_step = jax.jit(
        lambda current_state, batch: candidate_lora_train_step(
            current_state,
            batch,
            base_params,
            model_config,
            packed_memory,
            lora_config,
            parent_edge_coefficients,
            candidate_index,
            train_config,
        )
    )
    current_state = state
    losses: list[float] = []
    for step_index in range(train_config.steps):
        current_state, loss = compiled_step(
            current_state,
            batches[step_index % len(batches)],
        )
        scalar_loss = float(loss)
        losses.append(scalar_loss)
        if progress is not None:
            progress(step_index + 1, scalar_loss, train_config.steps)
    return current_state, LmLossTrace(step_losses=tuple(losses))


def run_resumable_candidate_edge_updates(
    state: LmTrainState[LoraEdge],
    batches: Sequence[TokenBatch],
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    packed_memory: PackedLoraMemory,
    lora_config: LoraConfig,
    parent_edge_coefficients: jax.Array,
    candidate_index: int,
    train_config: LmTrainConfig,
    *,
    stop_update: int | None = None,
    validation_function: CandidateValidationFunction | None = None,
) -> tuple[LmTrainState[LoraEdge], LmLossTrace, tuple[CandidateTrainingCheckpoint, ...]]:
    """Resume to one absolute update and retain power-of-two checkpoints."""
    _require_training_batches(batches)
    start_update = int(state.step)
    target_update = train_config.steps if stop_update is None else stop_update
    if (
        type(target_update) is not int
        or target_update < start_update
        or target_update > train_config.steps
    ):
        raise ValueError("stop_update must lie between state.step and the budget")
    checkpoint_updates = frozenset(
        (start_update, target_update)
        + tuple(
            2**power
            for power in range(train_config.steps.bit_length())
            if start_update < 2**power <= target_update
        )
    )
    compiled_step = jax.jit(
        lambda current_state, batch: candidate_lora_train_step(
            current_state,
            batch,
            base_params,
            model_config,
            packed_memory,
            lora_config,
            parent_edge_coefficients,
            candidate_index,
            train_config,
        )
    )

    def checkpoint(
        current_state: LmTrainState[LoraEdge],
        training_loss: float | None,
    ) -> CandidateTrainingCheckpoint:
        update = int(current_state.step)
        validation = (
            (None, None)
            if validation_function is None
            else validation_function(current_state.trainable, update)
        )
        return CandidateTrainingCheckpoint(
            update=update,
            state=current_state,
            training_loss=training_loss,
            validation_candidate_accuracy=validation[0],
            validation_correct_nll=validation[1],
        )

    current_state = state
    losses: list[float] = []
    checkpoints = [checkpoint(state, None)]
    for update in range(start_update + 1, target_update + 1):
        current_state, loss = compiled_step(
            current_state,
            batches[(update - 1) % len(batches)],
        )
        scalar_loss = float(loss)
        losses.append(scalar_loss)
        if update in checkpoint_updates:
            checkpoints.append(checkpoint(current_state, scalar_loss))
    return current_state, LmLossTrace(tuple(losses)), tuple(checkpoints)


def _require_training_batches(batches: Sequence[TokenBatch]) -> None:
    if not batches:
        raise ValueError("training requires at least one TokenBatch")

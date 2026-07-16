"""Immutable Optax training state and pure language-model update steps."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Generic, TypeVar

import jax
import jax.numpy as jnp
import optax

from apm.lm.config import GptNeoConfig
from apm.lm.gpt_neo import apply_gpt_neo
from apm.lm.lora import LoraConfig, LoraEdge
from apm.lm.lora_memory import PackedLoraMemory, packed_with_candidate_edge
from apm.lm.losses import mean_token_nll
from apm.lm.parameters import GptNeoParams
from apm.lm.text_data import TokenBatch

TrainableT = TypeVar("TrainableT")


@dataclass(frozen=True)
class LmTrainConfig:
    """Static AdamW budget, batching, clipping, and regularization values."""

    learning_rate: float
    steps: int
    batch_size: int
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0

    def __post_init__(self) -> None:
        """Reject invalid optimizer and fixed-budget values."""
        if not isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if not isfinite(self.gradient_clip_norm) or self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be finite and positive")


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class LmTrainState(Generic[TrainableT]):
    """One immutable trainable value, optimizer state, RNG stream, and step."""

    trainable: TrainableT
    opt_state: optax.OptState
    rng_key: jax.Array
    step: jax.Array

    def tree_flatten(self):
        """Expose trainable state as dynamic JAX PyTree leaves."""
        return (
            self.trainable,
            self.opt_state,
            self.rng_key,
            self.step,
        ), None

    @classmethod
    def tree_unflatten(cls, auxiliary_data, children):
        """Rebuild immutable state during JAX transformations."""
        del auxiliary_data
        return cls(*children)


BASE_TRAINING_PRESET = LmTrainConfig(
    learning_rate=3e-4,
    steps=5_000,
    batch_size=32,
    weight_decay=0.01,
    gradient_clip_norm=1.0,
)

EDGE_TRAINING_PRESET = LmTrainConfig(
    learning_rate=1e-3,
    steps=1_000,
    batch_size=32,
    weight_decay=0.01,
    gradient_clip_norm=1.0,
)

EDGE_LORA_PRESET = LoraConfig(rank=4, alpha=4.0)


def make_adamw_optimizer(config: LmTrainConfig) -> optax.GradientTransformation:
    """Build the deterministic clipped AdamW transform for a training config."""
    return optax.chain(
        optax.clip_by_global_norm(config.gradient_clip_norm),
        optax.adamw(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        ),
    )


def init_base_train_state(
    params: GptNeoParams,
    rng_key: jax.Array,
    train_config: LmTrainConfig,
) -> LmTrainState[GptNeoParams]:
    """Initialize optimizer state around base GPT-Neo parameters."""
    return _init_train_state(params, rng_key, train_config)


def init_candidate_lora_train_state(
    candidate_edge: LoraEdge,
    rng_key: jax.Array,
    train_config: LmTrainConfig,
) -> LmTrainState[LoraEdge]:
    """Initialize optimizer state around one trainable candidate LoRA edge."""
    return _init_train_state(candidate_edge, rng_key, train_config)


def base_train_step(
    state: LmTrainState[GptNeoParams],
    batch: TokenBatch,
    model_config: GptNeoConfig,
    train_config: LmTrainConfig,
) -> tuple[LmTrainState[GptNeoParams], jax.Array]:
    """Apply one pure clipped-AdamW update to all base model parameters."""
    _validate_batch(batch, train_config)
    next_rng_key, dropout_key = jax.random.split(state.rng_key)

    def loss_function(params: GptNeoParams) -> jax.Array:
        result = apply_gpt_neo(
            params,
            model_config,
            jnp.asarray(batch.input_ids, dtype=jnp.int32),
            jnp.asarray(batch.attention_mask, dtype=jnp.bool_),
            training=True,
            rng_key=dropout_key,
        )
        return mean_token_nll(
            result.logits,
            jnp.asarray(batch.target_ids, dtype=jnp.int32),
            jnp.asarray(batch.loss_mask, dtype=jnp.float32),
        )

    loss, gradients = jax.value_and_grad(loss_function)(state.trainable)
    optimizer = make_adamw_optimizer(train_config)
    updates, opt_state = optimizer.update(
        gradients,
        state.opt_state,
        state.trainable,
    )
    return (
        LmTrainState(
            trainable=optax.apply_updates(state.trainable, updates),
            opt_state=opt_state,
            rng_key=next_rng_key,
            step=state.step + jnp.asarray(1, dtype=jnp.int32),
        ),
        loss,
    )


def candidate_lora_train_step(
    state: LmTrainState[LoraEdge],
    batch: TokenBatch,
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    packed_memory: PackedLoraMemory,
    lora_config: LoraConfig,
    parent_edge_coefficients: jax.Array,
    candidate_index: int,
    train_config: LmTrainConfig,
) -> tuple[LmTrainState[LoraEdge], jax.Array]:
    """Update only a candidate edge over a frozen base and committed edge bank."""
    _validate_batch(batch, train_config)
    coefficients = _candidate_edge_coefficients(
        parent_edge_coefficients,
        candidate_index,
        packed_memory.valid_edge_mask.shape[0],
        batch.input_ids.shape[0],
    )
    frozen_base = jax.tree_util.tree_map(jax.lax.stop_gradient, base_params)
    frozen_coefficients = jax.lax.stop_gradient(coefficients)
    next_rng_key, dropout_key = jax.random.split(state.rng_key)

    def loss_function(candidate_edge: LoraEdge) -> jax.Array:
        candidate_memory = packed_with_candidate_edge(
            packed_memory,
            candidate_edge,
            candidate_index,
        )
        result = apply_gpt_neo(
            frozen_base,
            model_config,
            jnp.asarray(batch.input_ids, dtype=jnp.int32),
            jnp.asarray(batch.attention_mask, dtype=jnp.bool_),
            lora_memory=candidate_memory,
            edge_coefficients=frozen_coefficients,
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
    optimizer = make_adamw_optimizer(train_config)
    updates, opt_state = optimizer.update(
        gradients,
        state.opt_state,
        state.trainable,
    )
    return (
        LmTrainState(
            trainable=optax.apply_updates(state.trainable, updates),
            opt_state=opt_state,
            rng_key=next_rng_key,
            step=state.step + jnp.asarray(1, dtype=jnp.int32),
        ),
        loss,
    )


def _init_train_state(
    trainable: TrainableT,
    rng_key: jax.Array,
    train_config: LmTrainConfig,
) -> LmTrainState[TrainableT]:
    optimizer = make_adamw_optimizer(train_config)
    return LmTrainState(
        trainable=trainable,
        opt_state=optimizer.init(trainable),
        rng_key=rng_key,
        step=jnp.asarray(0, dtype=jnp.int32),
    )


def _validate_batch(batch: TokenBatch, train_config: LmTrainConfig) -> None:
    if batch.input_ids.shape[0] != train_config.batch_size:
        raise ValueError("TokenBatch row count must equal train_config.batch_size")


def _candidate_edge_coefficients(
    parent_edge_coefficients: jax.Array,
    candidate_index: int,
    edge_capacity: int,
    batch_size: int,
) -> jax.Array:
    if candidate_index < 0 or candidate_index >= edge_capacity:
        raise IndexError(f"candidate edge index is outside capacity: {candidate_index}")
    coefficients = jnp.asarray(parent_edge_coefficients, dtype=jnp.float32)
    if coefficients.ndim not in (1, 2) or coefficients.shape[-1] != edge_capacity:
        raise ValueError(
            "parent_edge_coefficients must have shape [edges] or [batch, edges]"
        )
    if coefficients.ndim == 2 and coefficients.shape[0] != batch_size:
        raise ValueError("batched parent coefficients must match TokenBatch rows")
    return coefficients.at[..., candidate_index].set(1.0)

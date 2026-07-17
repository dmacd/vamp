"""Per-example energy-based refinement of task-free node addresses."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import math
from typing import Callable, Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from apm.continual.language_tasks import RouterBatch
from apm.lm.config import GptNeoConfig
from apm.lm.gpt_neo import apply_gpt_neo
from apm.lm.lora import LoraConfig
from apm.lm.lora_memory import PackedLoraMemory, node_weights_to_edge_coefficients
from apm.lm.losses import per_token_nll
from apm.lm.parameters import GptNeoParams
from apm.memory.content_addressing import HopfieldAddressResult

EbtInitialization = Literal[
    "uniform",
    "hopfield",
    "full_node",
    "hopfield_top_k",
]
_FULL_NODE_INITIAL_LOGIT = 8.0


def _validate_positive_real(value: float, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(f"{field_name} must be a real number")
    if not math.isfinite(float(value)) or value <= 0.0:
        raise ValueError(f"{field_name} must be finite and positive")


@dataclass(frozen=True)
class EbtConfig:
    """Fixed Adam budget, softmax temperature, penalty, and initialization."""

    steps: int = 20
    learning_rate: float = 0.1
    tau: float = 1.0
    entropy_penalty: float = 0.01
    initialization: EbtInitialization = "uniform"

    def __post_init__(self) -> None:
        if type(self.steps) is not int or self.steps <= 0:
            raise ValueError("steps must be a positive integer")
        _validate_positive_real(self.learning_rate, "learning_rate")
        _validate_positive_real(self.tau, "tau")
        if isinstance(self.entropy_penalty, bool) or not isinstance(
            self.entropy_penalty,
            (int, float, np.integer, np.floating),
        ):
            raise TypeError("entropy_penalty must be a real number")
        if (
            not math.isfinite(float(self.entropy_penalty))
            or self.entropy_penalty < 0.0
        ):
            raise ValueError("entropy_penalty must be finite and nonnegative")
        if self.initialization not in (
            "uniform",
            "hopfield",
            "full_node",
            "hopfield_top_k",
        ):
            raise ValueError(f"unknown EBT initialization: {self.initialization}")
        object.__setattr__(self, "learning_rate", float(self.learning_rate))
        object.__setattr__(self, "tau", float(self.tau))
        object.__setattr__(self, "entropy_penalty", float(self.entropy_penalty))


class EbtAddressResult(NamedTuple):
    """Final addresses plus aligned objective, node, and edge step traces."""

    final_node_logits: jax.Array
    node_probabilities: jax.Array
    edge_coefficients: jax.Array
    selected_indices: jax.Array
    soft_mixture_nll: jax.Array
    hard_node_nll: jax.Array
    objective_trace: jax.Array
    node_probability_trace: jax.Array
    edge_coefficient_trace: jax.Array


_PrefixNllFunction = Callable[
    [
        GptNeoParams,
        GptNeoConfig,
        PackedLoraMemory,
        LoraConfig,
        RouterBatch,
        jax.Array,
    ],
    jax.Array,
]
_ObjectiveDependencies = tuple[_PrefixNllFunction, Callable[..., object]]


def masked_node_probabilities(
    node_logits: jax.Array,
    candidate_node_mask: jax.Array,
    tau: float | jax.Array,
) -> jax.Array:
    """Return temperature-scaled probabilities with exact zero masked mass."""
    logits = jnp.asarray(node_logits, dtype=jnp.float32)
    candidate_mask = jnp.asarray(candidate_node_mask, dtype=jnp.bool_)
    if logits.ndim != 2 or candidate_mask.shape != logits.shape:
        raise ValueError("node logits and candidate mask must share shape [batch, nodes]")
    masked_logits = jnp.where(
        candidate_mask,
        logits / jnp.asarray(tau, dtype=jnp.float32),
        jnp.asarray(-jnp.inf, dtype=jnp.float32),
    )
    probabilities = jax.nn.softmax(masked_logits, axis=-1)
    return jnp.where(
        candidate_mask,
        probabilities,
        jnp.asarray(0.0, dtype=jnp.float32),
    ).astype(jnp.float32)


def soft_mixture_prefix_nll(
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    packed_memory: PackedLoraMemory,
    lora_config: LoraConfig,
    prefix_batch: RouterBatch,
    node_probabilities: jax.Array,
) -> jax.Array:
    """Return per-example prefix NLL under continuous path coefficients."""
    probabilities = jnp.asarray(node_probabilities, dtype=jnp.float32)
    expected_shape = (
        prefix_batch.input_ids.shape[0],
        packed_memory.node_path_matrix.shape[0],
    )
    if probabilities.shape != expected_shape:
        raise ValueError(
            f"node_probabilities must have shape {expected_shape}"
        )
    frozen_base = jax.tree_util.tree_map(jax.lax.stop_gradient, base_params)
    frozen_memory = jax.tree_util.tree_map(jax.lax.stop_gradient, packed_memory)
    edge_coefficients = node_weights_to_edge_coefficients(
        probabilities,
        frozen_memory,
    )
    return _prefix_nll_for_edge_coefficients(
        frozen_base,
        model_config,
        frozen_memory,
        lora_config,
        prefix_batch,
        edge_coefficients,
    )


@partial(
    jax.jit,
    static_argnames=(
        "model_config",
        "lora_config",
        "config",
        "objective_dependencies",
    ),
)
def _optimize_node_logits(
    starting_logits: jax.Array,
    candidate_mask: jax.Array,
    current_base: GptNeoParams,
    current_memory: PackedLoraMemory,
    current_batch: RouterBatch,
    *,
    model_config: GptNeoConfig,
    lora_config: LoraConfig,
    config: EbtConfig,
    objective_dependencies: _ObjectiveDependencies,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Run one shape-stable compiled Adam refinement over node logits."""
    prefix_nll_function, _model_application_cache_token = objective_dependencies
    optimizer = optax.adam(config.learning_rate)

    def per_example_objective(node_logits: jax.Array) -> jax.Array:
        probabilities = masked_node_probabilities(
            node_logits,
            candidate_mask,
            config.tau,
        )
        prefix_nll = prefix_nll_function(
            current_base,
            model_config,
            current_memory,
            lora_config,
            current_batch,
            probabilities,
        )
        entropy = _masked_entropy(probabilities, candidate_mask)
        return prefix_nll + jnp.asarray(
            config.entropy_penalty,
            dtype=jnp.float32,
        ) * entropy

    starting_optimizer_state = optimizer.init(starting_logits)

    def update_logits(carry, unused_step):
        del unused_step
        current_logits, optimizer_state = carry

        def summed_objective(logits):
            objectives = per_example_objective(logits)
            return jnp.sum(objectives), objectives

        (_, objectives), gradients = jax.value_and_grad(
            summed_objective,
            has_aux=True,
        )(current_logits)
        updates, next_optimizer_state = optimizer.update(
            gradients,
            optimizer_state,
            current_logits,
        )
        next_logits = optax.apply_updates(current_logits, updates)
        return (next_logits, next_optimizer_state), (objectives, next_logits)

    (final_logits, _), (pre_update_trace, post_update_logits) = jax.lax.scan(
        update_logits,
        (starting_logits, starting_optimizer_state),
        xs=None,
        length=config.steps,
    )
    final_objective = per_example_objective(final_logits)
    return (
        final_logits,
        jnp.concatenate(
            (pre_update_trace, final_objective[None, :]),
            axis=0,
        ),
        jnp.concatenate(
            (starting_logits[None, :, :], post_update_logits),
            axis=0,
        ),
    )


def refine_ebt_address(
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    packed_memory: PackedLoraMemory,
    lora_config: LoraConfig,
    prefix_batch: RouterBatch,
    config: EbtConfig = EbtConfig(),
    *,
    hopfield_result: HopfieldAddressResult | None = None,
    initial_node_indices: jax.Array | np.ndarray | None = None,
) -> EbtAddressResult:
    """Optimize independent node logits against prefix NLL with Adam."""
    if not isinstance(prefix_batch, RouterBatch):
        raise TypeError("prefix_batch must be a RouterBatch")
    if not isinstance(config, EbtConfig):
        raise TypeError("config must be an EbtConfig")
    batch_size = prefix_batch.input_ids.shape[0]
    valid_node_mask = _validated_packed_memory(packed_memory)
    if np.any(np.sum(prefix_batch.loss_mask, axis=-1) == 0):
        raise ValueError("every prefix row must enable at least one loss token")
    initial_logits, candidate_node_mask = _initial_node_logits(
        config,
        batch_size,
        valid_node_mask,
        hopfield_result,
        initial_node_indices,
    )
    frozen_base = jax.tree_util.tree_map(jax.lax.stop_gradient, base_params)
    frozen_memory = jax.tree_util.tree_map(jax.lax.stop_gradient, packed_memory)
    candidate_mask = jnp.asarray(candidate_node_mask, dtype=jnp.bool_)
    optimized_logits, objective_trace, node_logit_trace = _optimize_node_logits(
        jnp.asarray(initial_logits, dtype=jnp.float32),
        candidate_mask,
        frozen_base,
        frozen_memory,
        prefix_batch,
        model_config=model_config,
        lora_config=lora_config,
        config=config,
        objective_dependencies=(soft_mixture_prefix_nll, apply_gpt_neo),
    )
    final_node_logits = jnp.where(
        candidate_mask,
        optimized_logits,
        jnp.asarray(-jnp.inf, dtype=jnp.float32),
    ).astype(jnp.float32)
    node_probability_trace = jax.vmap(
        lambda node_logits: masked_node_probabilities(
            node_logits,
            candidate_mask,
            config.tau,
        )
    )(node_logit_trace).astype(jnp.float32)
    edge_coefficient_trace = node_weights_to_edge_coefficients(
        node_probability_trace,
        frozen_memory,
    ).astype(jnp.float32)
    node_probabilities = node_probability_trace[-1]
    edge_coefficients = edge_coefficient_trace[-1]
    selected_indices = jnp.argmax(node_probabilities, axis=-1).astype(jnp.int32)
    soft_mixture_nll = soft_mixture_prefix_nll(
        frozen_base,
        model_config,
        frozen_memory,
        lora_config,
        prefix_batch,
        node_probabilities,
    ).astype(jnp.float32)
    hard_node_probabilities = jax.nn.one_hot(
        selected_indices,
        node_probabilities.shape[1],
        dtype=jnp.float32,
    )
    hard_node_nll = soft_mixture_prefix_nll(
        frozen_base,
        model_config,
        frozen_memory,
        lora_config,
        prefix_batch,
        hard_node_probabilities,
    ).astype(jnp.float32)
    return EbtAddressResult(
        final_node_logits=final_node_logits,
        node_probabilities=node_probabilities,
        edge_coefficients=edge_coefficients,
        selected_indices=selected_indices,
        soft_mixture_nll=soft_mixture_nll,
        hard_node_nll=hard_node_nll,
        objective_trace=objective_trace.astype(jnp.float32),
        node_probability_trace=node_probability_trace,
        edge_coefficient_trace=edge_coefficient_trace,
    )


def _prefix_nll_for_edge_coefficients(
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    packed_memory: PackedLoraMemory,
    lora_config: LoraConfig,
    prefix_batch: RouterBatch,
    edge_coefficients: jax.Array,
) -> jax.Array:
    logits = apply_gpt_neo(
        base_params,
        model_config,
        jnp.asarray(prefix_batch.input_ids, dtype=jnp.int32),
        jnp.asarray(prefix_batch.attention_mask, dtype=jnp.bool_),
        lora_memory=packed_memory,
        edge_coefficients=edge_coefficients,
        lora_config=lora_config,
        training=False,
    ).logits
    token_nll = per_token_nll(
        logits,
        jnp.asarray(prefix_batch.target_ids, dtype=jnp.int32),
    )
    loss_mask = jnp.asarray(prefix_batch.loss_mask, dtype=jnp.float32)
    return jnp.sum(token_nll * loss_mask, axis=-1) / jnp.sum(
        loss_mask,
        axis=-1,
    )


def _masked_entropy(
    node_probabilities: jax.Array,
    candidate_node_mask: jax.Array,
) -> jax.Array:
    safe_probabilities = jnp.where(
        candidate_node_mask,
        jnp.maximum(
            node_probabilities,
            jnp.asarray(jnp.finfo(jnp.float32).tiny, dtype=jnp.float32),
        ),
        jnp.asarray(1.0, dtype=jnp.float32),
    )
    entropy_terms = jnp.where(
        candidate_node_mask,
        node_probabilities * jnp.log(safe_probabilities),
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    return -jnp.sum(entropy_terms, axis=-1)


def _initial_node_logits(
    config: EbtConfig,
    batch_size: int,
    valid_node_mask: np.ndarray,
    hopfield_result: HopfieldAddressResult | None,
    initial_node_indices: jax.Array | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    valid_candidates = np.broadcast_to(
        valid_node_mask[None, :],
        (batch_size, valid_node_mask.shape[0]),
    ).copy()
    if config.initialization == "uniform":
        _reject_initialization_inputs(hopfield_result, initial_node_indices)
        return np.zeros(valid_candidates.shape, dtype=np.float32), valid_candidates
    if config.initialization == "full_node":
        if hopfield_result is not None:
            raise ValueError("full_node initialization does not accept hopfield_result")
        indices = _validated_initial_node_indices(
            initial_node_indices,
            batch_size,
            valid_node_mask,
        )
        logits = np.zeros(valid_candidates.shape, dtype=np.float32)
        logits[np.arange(batch_size), indices] = _FULL_NODE_INITIAL_LOGIT
        return logits, valid_candidates
    if initial_node_indices is not None:
        raise ValueError(
            f"{config.initialization} initialization does not accept initial_node_indices"
        )
    hopfield_probabilities, top_k_indices = _validated_hopfield_initialization(
        hopfield_result,
        batch_size,
        valid_node_mask,
        require_top_k=config.initialization == "hopfield_top_k",
    )
    stabilized_logits = np.log(
        np.maximum(hopfield_probabilities, np.finfo(np.float32).tiny)
    ).astype(np.float32)
    if config.initialization == "hopfield":
        return stabilized_logits, valid_candidates
    top_k_candidates = np.zeros(valid_candidates.shape, dtype=np.bool_)
    assert top_k_indices is not None
    top_k_candidates[
        np.arange(batch_size)[:, None],
        top_k_indices,
    ] = True
    return stabilized_logits, top_k_candidates


def _validated_hopfield_initialization(
    hopfield_result: HopfieldAddressResult | None,
    batch_size: int,
    valid_node_mask: np.ndarray,
    *,
    require_top_k: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    if not isinstance(hopfield_result, HopfieldAddressResult):
        raise TypeError("Hopfield initialization requires a HopfieldAddressResult")
    probabilities = np.asarray(hopfield_result.node_probabilities, dtype=np.float32)
    expected_shape = (batch_size, valid_node_mask.shape[0])
    if probabilities.shape != expected_shape:
        raise ValueError(
            f"Hopfield node probabilities must have shape {expected_shape}"
        )
    if (
        np.any(~np.isfinite(probabilities))
        or np.any(probabilities < 0.0)
        or not np.allclose(np.sum(probabilities, axis=-1), 1.0, atol=1e-6)
        or np.any(probabilities[:, ~valid_node_mask] != 0.0)
    ):
        raise ValueError("Hopfield probabilities must be normalized over valid nodes")
    if not require_top_k:
        return probabilities, None
    top_k_indices = np.asarray(hopfield_result.top_k_indices)
    if (
        top_k_indices.ndim != 2
        or top_k_indices.shape[0] != batch_size
        or top_k_indices.shape[1] < 1
        or top_k_indices.dtype.kind not in "iu"
    ):
        raise ValueError("Hopfield top-k indices must have shape [batch, nonempty k]")
    if np.any(
        (top_k_indices < 0)
        | (top_k_indices >= valid_node_mask.shape[0])
    ):
        raise ValueError("Hopfield top-k indices are outside node capacity")
    if np.any(~valid_node_mask[top_k_indices]):
        raise ValueError("Hopfield top-k indices must identify valid nodes")
    if any(len(set(row.tolist())) != len(row) for row in top_k_indices):
        raise ValueError("Hopfield top-k indices must be unique per example")
    return probabilities, top_k_indices.astype(np.int32, copy=False)


def _validated_initial_node_indices(
    initial_node_indices: jax.Array | np.ndarray | None,
    batch_size: int,
    valid_node_mask: np.ndarray,
) -> np.ndarray:
    if initial_node_indices is None:
        raise ValueError("full_node initialization requires initial_node_indices")
    indices = np.asarray(initial_node_indices)
    if indices.shape != (batch_size,) or indices.dtype.kind not in "iu":
        raise ValueError(
            f"initial_node_indices must contain {batch_size} integer values"
        )
    if np.any((indices < 0) | (indices >= valid_node_mask.shape[0])):
        raise ValueError("initial_node_indices are outside node capacity")
    if np.any(~valid_node_mask[indices]):
        raise ValueError("initial_node_indices must identify valid nodes")
    return indices.astype(np.int32, copy=False)


def _reject_initialization_inputs(
    hopfield_result: HopfieldAddressResult | None,
    initial_node_indices: jax.Array | np.ndarray | None,
) -> None:
    if hopfield_result is not None or initial_node_indices is not None:
        raise ValueError("uniform initialization does not accept initialization inputs")


def _validated_packed_memory(packed_memory: PackedLoraMemory) -> np.ndarray:
    if not isinstance(packed_memory, PackedLoraMemory):
        raise TypeError("packed_memory must be a PackedLoraMemory")
    node_paths = np.asarray(packed_memory.node_path_matrix)
    valid_node_mask = np.asarray(packed_memory.valid_node_mask, dtype=np.bool_)
    valid_edge_mask = np.asarray(packed_memory.valid_edge_mask, dtype=np.bool_)
    if node_paths.ndim != 2:
        raise ValueError("packed node path matrix must have rank two")
    if valid_node_mask.shape != (node_paths.shape[0],):
        raise ValueError("packed valid-node mask must match node capacity")
    if valid_edge_mask.shape != (node_paths.shape[1],):
        raise ValueError("packed valid-edge mask must match edge capacity")
    if not np.any(valid_node_mask):
        raise ValueError("EBT refinement requires at least one valid node")
    if not np.all(np.isfinite(node_paths)):
        raise ValueError("packed node paths must be finite")
    return valid_node_mask

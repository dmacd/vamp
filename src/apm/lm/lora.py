"""Fixed-shape pathwise LoRA values and linear projection application."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import jax
import jax.numpy as jnp

from apm.lm.config import GptNeoConfig
from apm.lm.parameters import LinearParams


@dataclass(frozen=True)
class LoraTargetMask:
    """Static switches for the six supported transformer projections."""

    query: bool = True
    key: bool = True
    value: bool = True
    attention_output: bool = True
    mlp_input: bool = True
    mlp_output: bool = True


@dataclass(frozen=True)
class LoraConfig:
    """Immutable rank, scale, and target-site configuration for one run."""

    rank: int
    alpha: float
    target_mask: LoraTargetMask = field(default_factory=LoraTargetMask)

    def __post_init__(self) -> None:
        """Reject ranks and scales that cannot define a LoRA update."""
        if self.rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if self.alpha <= 0.0:
            raise ValueError("LoRA alpha must be positive")

    @property
    def scale(self) -> float:
        """Return the fixed multiplier applied to every completed edge update."""
        return self.alpha / self.rank


class LoraProjection(NamedTuple):
    """One low-rank projection update with shapes [input, rank] and [rank, output]."""

    left: jax.Array
    right: jax.Array


class LoraBlock(NamedTuple):
    """LoRA projections for every supported linear site in one transformer block."""

    query: LoraProjection
    key: LoraProjection
    value: LoraProjection
    attention_output: LoraProjection
    mlp_input: LoraProjection
    mlp_output: LoraProjection


class LoraEdge(NamedTuple):
    """One immutable LoRA residual edge spanning all transformer blocks."""

    blocks: tuple[LoraBlock, ...]


class LoraProjectionBank(NamedTuple):
    """Corresponding projection factors stacked on a leading edge axis."""

    left: jax.Array
    right: jax.Array


class LoraBlockBank(NamedTuple):
    """Stacked LoRA projection banks for one transformer block."""

    query: LoraProjectionBank
    key: LoraProjectionBank
    value: LoraProjectionBank
    attention_output: LoraProjectionBank
    mlp_input: LoraProjectionBank
    mlp_output: LoraProjectionBank


class LoraEdgeBank(NamedTuple):
    """A fixed-capacity bank of LoRA edges for every transformer block."""

    blocks: tuple[LoraBlockBank, ...]


def init_lora_edge(
    rng_key: jax.Array,
    model_config: GptNeoConfig,
    lora_config: LoraConfig,
) -> LoraEdge:
    """Initialize fan-in-scaled left factors and exactly zero right factors."""
    keys = jax.random.split(rng_key, 6 * model_config.num_layers)
    return LoraEdge(
        blocks=tuple(
            _init_lora_block(
                keys[6 * layer_index : 6 * (layer_index + 1)],
                model_config,
                lora_config,
            )
            for layer_index in range(model_config.num_layers)
        )
    )


def stack_lora_edges(
    edges: tuple[LoraEdge, ...],
    model_config: GptNeoConfig,
    lora_config: LoraConfig,
    max_edges: int,
) -> LoraEdgeBank:
    """Stack immutable edges into a zero-padded fixed-capacity bank."""
    if max_edges < 0:
        raise ValueError("max_edges cannot be negative")
    if len(edges) > max_edges:
        raise ValueError("LoRA edge count exceeds max_edges")
    _validate_edges(edges, model_config, lora_config)
    return LoraEdgeBank(
        blocks=tuple(
            _stack_lora_block(
                tuple(edge.blocks[layer_index] for edge in edges),
                model_config,
                lora_config.rank,
                max_edges,
            )
            for layer_index in range(model_config.num_layers)
        )
    )


def insert_lora_edge(
    bank: LoraEdgeBank,
    edge: LoraEdge,
    index: int,
) -> LoraEdgeBank:
    """Return a bank with one edge functionally inserted at a padded slot."""
    if len(bank.blocks) != len(edge.blocks):
        raise ValueError("LoRA edge block count does not match the bank")
    max_edges = _bank_capacity(bank)
    if not 0 <= index < max_edges:
        raise IndexError(f"LoRA edge index {index} is outside capacity {max_edges}")
    return LoraEdgeBank(
        blocks=tuple(
            _insert_lora_block(bank_block, edge_block, index)
            for bank_block, edge_block in zip(bank.blocks, edge.blocks)
        )
    )


def apply_lora_linear(
    base_params: LinearParams,
    inputs: jax.Array,
    projection_bank: LoraProjectionBank,
    edge_coefficients: jax.Array,
    scale: float,
    target_enabled: bool,
) -> jax.Array:
    """Apply a base linear map plus independent weighted LoRA edge outputs."""
    base_output = inputs @ base_params.kernel
    if base_params.bias is not None:
        base_output = base_output + base_params.bias
    if not target_enabled:
        return base_output
    _validate_apply_shapes(inputs, projection_bank, edge_coefficients)
    edge_hidden = jnp.einsum(
        "...i,eir->...er",
        inputs.astype(jnp.float32),
        projection_bank.left,
    )
    if edge_coefficients.ndim == 1:
        update = jnp.einsum(
            "...er,ero,e->...o",
            edge_hidden,
            projection_bank.right,
            edge_coefficients,
        )
    else:
        update = jnp.einsum(
            "b...er,ero,be->b...o",
            edge_hidden,
            projection_bank.right,
            edge_coefficients,
        )
    return base_output + jnp.asarray(scale, dtype=jnp.float32) * update


def _init_lora_block(
    keys: jax.Array,
    model_config: GptNeoConfig,
    lora_config: LoraConfig,
) -> LoraBlock:
    hidden_size = model_config.hidden_size
    intermediate_size = model_config.intermediate_size
    rank = lora_config.rank
    targets = lora_config.target_mask
    return LoraBlock(
        query=_init_lora_projection(
            keys[0], hidden_size, hidden_size, rank, targets.query
        ),
        key=_init_lora_projection(keys[1], hidden_size, hidden_size, rank, targets.key),
        value=_init_lora_projection(
            keys[2], hidden_size, hidden_size, rank, targets.value
        ),
        attention_output=_init_lora_projection(
            keys[3], hidden_size, hidden_size, rank, targets.attention_output
        ),
        mlp_input=_init_lora_projection(
            keys[4], hidden_size, intermediate_size, rank, targets.mlp_input
        ),
        mlp_output=_init_lora_projection(
            keys[5], intermediate_size, hidden_size, rank, targets.mlp_output
        ),
    )


def _init_lora_projection(
    rng_key: jax.Array,
    input_size: int,
    output_size: int,
    rank: int,
    enabled: bool,
) -> LoraProjection:
    left = (
        jax.random.normal(
            rng_key,
            (input_size, rank),
            dtype=jnp.float32,
        )
        / jnp.sqrt(jnp.asarray(input_size, dtype=jnp.float32))
        if enabled
        else jnp.zeros((input_size, rank), dtype=jnp.float32)
    )
    return LoraProjection(
        left=left,
        right=jnp.zeros((rank, output_size), dtype=jnp.float32),
    )


def _stack_lora_block(
    blocks: tuple[LoraBlock, ...],
    model_config: GptNeoConfig,
    rank: int,
    max_edges: int,
) -> LoraBlockBank:
    hidden_size = model_config.hidden_size
    intermediate_size = model_config.intermediate_size
    specifications = (
        ("query", hidden_size, hidden_size),
        ("key", hidden_size, hidden_size),
        ("value", hidden_size, hidden_size),
        ("attention_output", hidden_size, hidden_size),
        ("mlp_input", hidden_size, intermediate_size),
        ("mlp_output", intermediate_size, hidden_size),
    )
    banks = {
        name: _stack_lora_projections(
            tuple(getattr(block, name) for block in blocks),
            input_size,
            output_size,
            rank,
            max_edges,
        )
        for name, input_size, output_size in specifications
    }
    return LoraBlockBank(**banks)


def _stack_lora_projections(
    projections: tuple[LoraProjection, ...],
    input_size: int,
    output_size: int,
    rank: int,
    max_edges: int,
) -> LoraProjectionBank:
    left = jnp.zeros((max_edges, input_size, rank), dtype=jnp.float32)
    right = jnp.zeros((max_edges, rank, output_size), dtype=jnp.float32)
    if projections:
        left = left.at[: len(projections)].set(
            jnp.stack(tuple(projection.left for projection in projections))
        )
        right = right.at[: len(projections)].set(
            jnp.stack(tuple(projection.right for projection in projections))
        )
    return LoraProjectionBank(left=left, right=right)


def _insert_lora_block(
    bank: LoraBlockBank,
    edge: LoraBlock,
    index: int,
) -> LoraBlockBank:
    return LoraBlockBank(
        **{
            name: _insert_lora_projection(
                getattr(bank, name),
                getattr(edge, name),
                index,
            )
            for name in LoraBlock._fields
        }
    )


def _insert_lora_projection(
    bank: LoraProjectionBank,
    edge: LoraProjection,
    index: int,
) -> LoraProjectionBank:
    expected_left_shape = bank.left.shape[1:]
    expected_right_shape = bank.right.shape[1:]
    if edge.left.shape != expected_left_shape or edge.right.shape != expected_right_shape:
        raise ValueError(
            "LoRA projection shape does not match bank slot: "
            f"expected {expected_left_shape}/{expected_right_shape}, "
            f"received {edge.left.shape}/{edge.right.shape}"
        )
    return LoraProjectionBank(
        left=bank.left.at[index].set(edge.left.astype(jnp.float32)),
        right=bank.right.at[index].set(edge.right.astype(jnp.float32)),
    )


def _validate_edges(
    edges: tuple[LoraEdge, ...],
    model_config: GptNeoConfig,
    lora_config: LoraConfig,
) -> None:
    expected_shapes = (
        (model_config.hidden_size, lora_config.rank, model_config.hidden_size),
        (model_config.hidden_size, lora_config.rank, model_config.hidden_size),
        (model_config.hidden_size, lora_config.rank, model_config.hidden_size),
        (model_config.hidden_size, lora_config.rank, model_config.hidden_size),
        (
            model_config.hidden_size,
            lora_config.rank,
            model_config.intermediate_size,
        ),
        (
            model_config.intermediate_size,
            lora_config.rank,
            model_config.hidden_size,
        ),
    )
    for edge in edges:
        if len(edge.blocks) != model_config.num_layers:
            raise ValueError("LoRA edge must contain one block per model layer")
        for block in edge.blocks:
            for projection, (input_size, rank, output_size) in zip(
                block,
                expected_shapes,
            ):
                if projection.left.shape != (input_size, rank) or projection.right.shape != (
                    rank,
                    output_size,
                ):
                    raise ValueError("LoRA projection shape does not match configuration")


def _bank_capacity(bank: LoraEdgeBank) -> int:
    if not bank.blocks:
        raise ValueError("LoRA edge bank must contain at least one model block")
    capacities = {
        projection_bank.left.shape[0]
        for block in bank.blocks
        for projection_bank in block
    } | {
        projection_bank.right.shape[0]
        for block in bank.blocks
        for projection_bank in block
    }
    if len(capacities) != 1:
        raise ValueError("LoRA bank projections do not share one edge capacity")
    return capacities.pop()


def _validate_apply_shapes(
    inputs: jax.Array,
    projection_bank: LoraProjectionBank,
    edge_coefficients: jax.Array,
) -> None:
    if projection_bank.left.ndim != 3 or projection_bank.right.ndim != 3:
        raise ValueError("LoRA projection bank factors must each have rank three")
    edge_count, input_size, rank = projection_bank.left.shape
    right_edge_count, right_rank, _ = projection_bank.right.shape
    if (edge_count, rank) != (right_edge_count, right_rank):
        raise ValueError("LoRA projection bank factor shapes are incompatible")
    if inputs.shape[-1] != input_size:
        raise ValueError("input width does not match LoRA left factor")
    if edge_coefficients.ndim not in (1, 2):
        raise ValueError("edge_coefficients must have shape [edges] or [batch, edges]")
    if edge_coefficients.shape[-1] != edge_count:
        raise ValueError("edge coefficient count does not match bank capacity")
    if edge_coefficients.ndim == 2 and (
        inputs.ndim < 2 or edge_coefficients.shape[0] != inputs.shape[0]
    ):
        raise ValueError("batched edge coefficients must match the input batch")

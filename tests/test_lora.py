from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from apm.lm import GptNeoConfig, LinearParams
from apm.lm.lora import (
    LoraConfig,
    LoraProjection,
    LoraProjectionBank,
    LoraTargetMask,
    apply_lora_linear,
    init_lora_edge,
    insert_lora_edge,
    stack_lora_edges,
)


def _model_config() -> GptNeoConfig:
    return GptNeoConfig(
        vocab_size=19,
        max_position_embeddings=8,
        hidden_size=4,
        intermediate_size=7,
        num_layers=2,
        num_heads=2,
        attention_types=("global", "local"),
        local_window_size=4,
    )


def test_lora_initialization_has_fixed_shapes_and_exact_zero_right_factors() -> None:
    model_config = _model_config()
    lora_config = LoraConfig(rank=2, alpha=4.0)
    edge = init_lora_edge(jax.random.PRNGKey(0), model_config, lora_config)

    assert lora_config.scale == 2.0
    assert len(edge.blocks) == model_config.num_layers
    assert edge.blocks[0].query.left.shape == (4, 2)
    assert edge.blocks[0].query.right.shape == (2, 4)
    assert edge.blocks[0].mlp_input.left.shape == (4, 2)
    assert edge.blocks[0].mlp_input.right.shape == (2, 7)
    assert edge.blocks[0].mlp_output.left.shape == (7, 2)
    assert edge.blocks[0].mlp_output.right.shape == (2, 4)
    assert all(
        projection.left.dtype == jnp.float32
        and projection.right.dtype == jnp.float32
        and bool(jnp.all(projection.right == 0.0))
        for block in edge.blocks
        for projection in block
    )


def test_zero_effect_lora_returns_the_base_linear_result() -> None:
    model_config = _model_config()
    lora_config = LoraConfig(rank=2, alpha=2.0)
    edge = init_lora_edge(jax.random.PRNGKey(1), model_config, lora_config)
    bank = stack_lora_edges((edge,), model_config, lora_config, max_edges=3)
    base = LinearParams(
        kernel=jnp.arange(16, dtype=jnp.float32).reshape(4, 4) / 10.0,
        bias=jnp.arange(4, dtype=jnp.float32),
    )
    inputs = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4) / 7.0
    expected = jnp.einsum("...i,io->...o", inputs, base.kernel) + base.bias

    actual = apply_lora_linear(
        base,
        inputs,
        bank.blocks[0].query,
        jnp.asarray([1.0, 0.0, 0.0], dtype=jnp.float32),
        lora_config.scale,
        True,
    )

    np.testing.assert_array_equal(actual, expected)


def test_lora_linear_sums_completed_edge_outputs_not_factor_coordinates() -> None:
    base = LinearParams(
        kernel=jnp.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32),
        bias=jnp.asarray([0.5, -0.5], dtype=jnp.float32),
    )
    projection_bank = LoraProjectionBank(
        left=jnp.asarray(
            [
                [[1.0], [2.0]],
                [[-1.0], [1.0]],
            ],
            dtype=jnp.float32,
        ),
        right=jnp.asarray(
            [
                [[3.0, 4.0]],
                [[2.0, -1.0]],
            ],
            dtype=jnp.float32,
        ),
    )
    inputs = jnp.asarray([[5.0, 6.0]], dtype=jnp.float32)
    coefficients = jnp.asarray([0.5, 2.0], dtype=jnp.float32)
    scale = 0.25
    first_edge = (inputs @ projection_bank.left[0]) @ projection_bank.right[0]
    second_edge = (inputs @ projection_bank.left[1]) @ projection_bank.right[1]
    expected = (
        inputs @ base.kernel
        + base.bias
        + scale * (coefficients[0] * first_edge + coefficients[1] * second_edge)
    )

    actual = apply_lora_linear(
        base,
        inputs,
        projection_bank,
        coefficients,
        scale,
        True,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_lora_linear_applies_batched_coefficients_independently() -> None:
    base = LinearParams(kernel=jnp.eye(2, dtype=jnp.float32), bias=None)
    projection_bank = LoraProjectionBank(
        left=jnp.asarray(
            [
                [[1.0], [0.0]],
                [[0.0], [1.0]],
            ],
            dtype=jnp.float32,
        ),
        right=jnp.asarray(
            [
                [[2.0, 0.0]],
                [[0.0, 3.0]],
            ],
            dtype=jnp.float32,
        ),
    )
    inputs = jnp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    coefficients = jnp.asarray([[1.0, 0.0], [0.0, 0.5]], dtype=jnp.float32)
    expected = jnp.asarray([[3.0, 2.0], [3.0, 10.0]], dtype=jnp.float32)

    actual = apply_lora_linear(
        base,
        inputs,
        projection_bank,
        coefficients,
        1.0,
        True,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_disabled_target_ignores_lora_update() -> None:
    target_mask = LoraTargetMask(query=False)
    base = LinearParams(kernel=jnp.eye(2, dtype=jnp.float32), bias=None)
    inputs = jnp.asarray([[1.0, 2.0]], dtype=jnp.float32)
    projection_bank = LoraProjectionBank(
        left=jnp.ones((1, 2, 1), dtype=jnp.float32),
        right=jnp.ones((1, 1, 2), dtype=jnp.float32),
    )

    actual = apply_lora_linear(
        base,
        inputs,
        projection_bank,
        jnp.ones((1,), dtype=jnp.float32),
        1.0,
        target_mask.query,
    )

    np.testing.assert_array_equal(actual, inputs)


def test_disabled_target_initializes_zero_factors_without_changing_shapes() -> None:
    model_config = _model_config()
    lora_config = LoraConfig(
        rank=2,
        alpha=2.0,
        target_mask=LoraTargetMask(query=False, mlp_output=False),
    )

    edge = init_lora_edge(jax.random.PRNGKey(11), model_config, lora_config)

    for block in edge.blocks:
        assert block.query.left.shape == (4, 2)
        assert block.mlp_output.left.shape == (7, 2)
        np.testing.assert_array_equal(block.query.left, 0.0)
        np.testing.assert_array_equal(block.mlp_output.left, 0.0)
        assert np.count_nonzero(np.asarray(block.key.left)) > 0


def test_stack_and_insert_preserve_edges_and_zero_padding() -> None:
    model_config = _model_config()
    lora_config = LoraConfig(rank=2, alpha=2.0)
    first = init_lora_edge(jax.random.PRNGKey(2), model_config, lora_config)
    second_zero = init_lora_edge(jax.random.PRNGKey(3), model_config, lora_config)
    second_query = LoraProjection(
        left=second_zero.blocks[0].query.left,
        right=jnp.ones_like(second_zero.blocks[0].query.right),
    )
    second = second_zero._replace(
        blocks=(
            second_zero.blocks[0]._replace(query=second_query),
            second_zero.blocks[1],
        )
    )
    bank = stack_lora_edges((first,), model_config, lora_config, max_edges=3)

    inserted = insert_lora_edge(bank, second, 1)

    np.testing.assert_array_equal(inserted.blocks[0].query.left[0], first.blocks[0].query.left)
    np.testing.assert_array_equal(inserted.blocks[0].query.right[1], second.blocks[0].query.right)
    np.testing.assert_array_equal(inserted.blocks[0].query.left[2], 0.0)
    np.testing.assert_array_equal(inserted.blocks[0].query.right[2], 0.0)
    np.testing.assert_array_equal(bank.blocks[0].query.right[1], 0.0)


def test_root_only_bank_has_zero_edge_capacity() -> None:
    model_config = _model_config()
    bank = stack_lora_edges(
        (),
        model_config,
        LoraConfig(rank=2, alpha=2.0),
        max_edges=0,
    )

    assert bank.blocks[0].query.left.shape == (0, 4, 2)
    assert bank.blocks[0].query.right.shape == (0, 2, 4)

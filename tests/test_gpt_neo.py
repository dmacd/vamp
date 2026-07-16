from __future__ import annotations

from dataclasses import FrozenInstanceError

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from apm.lm.attention import apply_attention, attention_pattern
from apm.lm.config import GptNeoConfig
from apm.lm.gpt_neo import (
    CapturePoint,
    CaptureSpec,
    apply_gpt_neo,
    apply_gpt_neo_embeddings,
    embed_tokens,
    gelu_new,
)
from apm.lm.losses import mean_token_nll, per_token_nll
from apm.lm.parameters import AttentionParams, LinearParams, init_gpt_neo_params


def _tiny_config(**overrides: object) -> GptNeoConfig:
    values: dict[str, object] = {
        "vocab_size": 11,
        "max_position_embeddings": 8,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_layers": 1,
        "num_heads": 2,
        "attention_types": ("global",),
        "local_window_size": 2,
        "layer_norm_epsilon": 1e-5,
        "activation": "gelu_new",
        "embedding_dropout": 0.0,
        "attention_dropout": 0.0,
        "residual_dropout": 0.0,
        "initializer_range": 0.02,
    }
    values.update(overrides)
    return GptNeoConfig(**values)  # type: ignore[arg-type]


def _identity_attention_params(hidden_size: int) -> AttentionParams:
    identity = jnp.eye(hidden_size, dtype=jnp.float32)
    bias = jnp.zeros((hidden_size,), dtype=jnp.float32)
    return AttentionParams(
        query=LinearParams(kernel=identity, bias=None),
        key=LinearParams(kernel=identity, bias=None),
        value=LinearParams(kernel=identity, bias=None),
        output=LinearParams(kernel=identity, bias=bias),
    )


def _assert_tree_allclose(left: object, right: object, **kwargs: float) -> None:
    left_leaves, left_structure = jax.tree_util.tree_flatten(left)
    right_leaves, right_structure = jax.tree_util.tree_flatten(right)
    assert left_structure == right_structure
    assert len(left_leaves) == len(right_leaves)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves):
        np.testing.assert_allclose(np.asarray(left_leaf), np.asarray(right_leaf), **kwargs)


@pytest.mark.parametrize(
    "override",
    (
        {"vocab_size": 0},
        {"max_position_embeddings": 0},
        {"hidden_size": 0},
        {"intermediate_size": 0},
        {"num_layers": 0, "attention_types": ()},
        {"num_heads": 0},
        {"hidden_size": 7, "num_heads": 2},
        {"num_layers": 2, "attention_types": ("global",)},
        {"attention_types": ("diagonal",)},
        {"local_window_size": 0},
        {"layer_norm_epsilon": 0.0},
        {"activation": "relu"},
        {"embedding_dropout": -0.1},
        {"attention_dropout": 1.0},
        {"residual_dropout": 1.1},
        {"initializer_range": 0.0},
    ),
)
def test_gpt_neo_config_rejects_invalid_values(override: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _tiny_config(**override)


def test_gpt_neo_config_is_frozen() -> None:
    config = _tiny_config()

    with pytest.raises(FrozenInstanceError):
        config.hidden_size = 16  # type: ignore[misc]


def test_parameter_shapes_and_projection_biases() -> None:
    config = _tiny_config(num_layers=2, attention_types=("global", "local"))
    params = init_gpt_neo_params(jax.random.PRNGKey(0), config)

    assert params.token_embedding.shape == (config.vocab_size, config.hidden_size)
    assert params.position_embedding.shape == (config.max_position_embeddings, config.hidden_size)
    assert params.token_embedding.dtype == jnp.float32
    assert len(params.blocks) == config.num_layers
    assert params.final_norm.scale.shape == (config.hidden_size,)
    assert params.final_norm.bias.shape == (config.hidden_size,)

    for block in params.blocks:
        assert block.attention_norm.scale.shape == (config.hidden_size,)
        assert block.attention_norm.bias.shape == (config.hidden_size,)
        assert block.mlp_norm.scale.shape == (config.hidden_size,)
        assert block.mlp_norm.bias.shape == (config.hidden_size,)
        for projection in (block.attention.query, block.attention.key, block.attention.value):
            assert projection.kernel.shape == (config.hidden_size, config.hidden_size)
            assert projection.bias is None
        assert block.attention.output.kernel.shape == (config.hidden_size, config.hidden_size)
        assert block.attention.output.bias is not None
        assert block.attention.output.bias.shape == (config.hidden_size,)
        assert block.mlp.input_projection.kernel.shape == (config.hidden_size, config.intermediate_size)
        assert block.mlp.input_projection.bias is not None
        assert block.mlp.input_projection.bias.shape == (config.intermediate_size,)
        assert block.mlp.output_projection.kernel.shape == (config.intermediate_size, config.hidden_size)
        assert block.mlp.output_projection.bias is not None
        assert block.mlp.output_projection.bias.shape == (config.hidden_size,)


def test_global_and_local_causal_mask_boundaries() -> None:
    global_mask = np.asarray(attention_pattern(4, "global", local_window_size=2)).reshape((4, 4))
    local_mask = np.asarray(attention_pattern(4, "local", local_window_size=2)).reshape((4, 4))

    np.testing.assert_array_equal(
        global_mask,
        np.asarray(
            (
                (True, False, False, False),
                (True, True, False, False),
                (True, True, True, False),
                (True, True, True, True),
            )
        ),
    )
    np.testing.assert_array_equal(
        local_mask,
        np.asarray(
            (
                (True, False, False, False),
                (True, True, False, False),
                (False, True, True, False),
                (False, False, True, True),
            )
        ),
    )
    assert global_mask.dtype == np.bool_
    assert local_mask.dtype == np.bool_


def test_attention_uses_raw_unscaled_query_key_scores() -> None:
    config = _tiny_config(hidden_size=2, intermediate_size=4, num_heads=1)
    hidden_states = jnp.asarray([[[1.0, 0.0], [1.0, 1.0]]], dtype=jnp.float32)

    output = apply_attention(
        _identity_attention_params(config.hidden_size),
        config,
        hidden_states,
        jnp.ones((1, 2), dtype=jnp.bool_),
        "global",
        training=False,
        probability_dropout_key=None,
        output_dropout_key=None,
    )

    raw_second_key_weight = float(jax.nn.softmax(jnp.asarray([1.0, 2.0]))[1])
    expected = np.asarray([[[1.0, 0.0], [1.0, raw_second_key_weight]]], dtype=np.float32)
    np.testing.assert_allclose(np.asarray(output), expected, rtol=1e-6, atol=1e-6)


def test_padding_mask_excludes_keys_but_does_not_zero_queries() -> None:
    config = _tiny_config(hidden_size=2, intermediate_size=4, num_heads=1)
    first_hidden = jnp.asarray([[[1.0, 0.0], [100.0, 100.0], [0.0, 1.0]]], dtype=jnp.float32)
    second_hidden = first_hidden.at[0, 1].set(jnp.asarray([-100.0, 50.0], dtype=jnp.float32))
    attention_mask = jnp.asarray([[True, False, True]])
    params = _identity_attention_params(config.hidden_size)

    first_output = apply_attention(
        params,
        config,
        first_hidden,
        attention_mask,
        "global",
        training=False,
        probability_dropout_key=None,
        output_dropout_key=None,
    )
    second_output = apply_attention(
        params,
        config,
        second_hidden,
        attention_mask,
        "global",
        training=False,
        probability_dropout_key=None,
        output_dropout_key=None,
    )

    np.testing.assert_allclose(np.asarray(first_output[0, 2]), np.asarray(second_output[0, 2]), atol=1e-6)
    np.testing.assert_allclose(np.asarray(first_output[0, 1]), np.asarray([1.0, 0.0]), atol=1e-6)


@pytest.mark.parametrize("attention_type", ("global", "local"))
def test_future_tokens_do_not_influence_prefix_logits(attention_type: str) -> None:
    config = _tiny_config(attention_types=(attention_type,))
    params = init_gpt_neo_params(jax.random.PRNGKey(1), config)
    token_ids = jnp.asarray(((1, 2, 3, 4), (1, 2, 7, 8)), dtype=jnp.int32)
    attention_mask = jnp.ones_like(token_ids, dtype=jnp.bool_)

    result = apply_gpt_neo(params, config, token_ids, attention_mask)

    np.testing.assert_allclose(np.asarray(result.logits[0, :2]), np.asarray(result.logits[1, :2]), atol=1e-6)


def test_right_padding_preserves_nonpadding_hidden_states() -> None:
    config = _tiny_config()
    params = init_gpt_neo_params(jax.random.PRNGKey(2), config)
    short_ids = jnp.asarray(((1, 2, 3),), dtype=jnp.int32)
    padded_ids = jnp.asarray(((1, 2, 3, 9, 10),), dtype=jnp.int32)

    short_result = apply_gpt_neo(params, config, short_ids, jnp.ones_like(short_ids, dtype=jnp.bool_))
    padded_result = apply_gpt_neo(
        params,
        config,
        padded_ids,
        jnp.asarray(((True, True, True, False, False),)),
    )

    np.testing.assert_allclose(
        np.asarray(short_result.final_hidden),
        np.asarray(padded_result.final_hidden[:, :3]),
        rtol=1e-6,
        atol=1e-6,
    )


def test_gelu_new_matches_gpt_neo_formula() -> None:
    inputs = jnp.asarray((-3.0, -1.0, 0.0, 0.5, 2.0), dtype=jnp.float32)
    expected = 0.5 * inputs * (
        1.0 + jnp.tanh(jnp.sqrt(jnp.asarray(2.0 / np.pi)) * (inputs + 0.044715 * inputs**3))
    )

    np.testing.assert_allclose(np.asarray(gelu_new(inputs)), np.asarray(expected), rtol=1e-6, atol=1e-6)


def test_embeddings_use_explicit_positions_and_match_embeddings_forward() -> None:
    config = _tiny_config()
    params = init_gpt_neo_params(jax.random.PRNGKey(3), config)
    token_ids = jnp.asarray(((1, 2, 3), (4, 5, 6)), dtype=jnp.int32)
    position_ids = jnp.asarray(((2, 1, 0), (0, 2, 1)), dtype=jnp.int32)
    attention_mask = jnp.ones_like(token_ids, dtype=jnp.bool_)

    input_embeddings = embed_tokens(params, token_ids, position_ids)
    expected_embeddings = params.token_embedding[token_ids] + params.position_embedding[position_ids]
    token_result = apply_gpt_neo(
        params,
        config,
        token_ids,
        attention_mask,
        position_ids=position_ids,
    )
    embedding_result = apply_gpt_neo_embeddings(
        params,
        config,
        input_embeddings,
        attention_mask,
    )

    np.testing.assert_allclose(np.asarray(input_embeddings), np.asarray(expected_embeddings), atol=0.0)
    _assert_tree_allclose(token_result, embedding_result, rtol=1e-6, atol=1e-6)


def test_default_positions_are_left_to_right_indices() -> None:
    config = _tiny_config()
    params = init_gpt_neo_params(jax.random.PRNGKey(4), config)
    token_ids = jnp.asarray(((1, 2, 3, 4),), dtype=jnp.int32)
    attention_mask = jnp.ones_like(token_ids, dtype=jnp.bool_)
    explicit_positions = jnp.arange(token_ids.shape[1], dtype=jnp.int32)[None, :]

    implicit_result = apply_gpt_neo(params, config, token_ids, attention_mask)
    explicit_result = apply_gpt_neo(
        params,
        config,
        token_ids,
        attention_mask,
        position_ids=explicit_positions,
    )

    _assert_tree_allclose(implicit_result, explicit_result, rtol=1e-6, atol=1e-6)


def test_output_logits_are_tied_to_token_embeddings() -> None:
    config = _tiny_config()
    params = init_gpt_neo_params(jax.random.PRNGKey(5), config)
    token_ids = jnp.asarray(((1, 2, 3),), dtype=jnp.int32)
    result = apply_gpt_neo(params, config, token_ids, jnp.ones_like(token_ids, dtype=jnp.bool_))

    expected_logits = jnp.einsum("bsh,vh->bsv", result.final_hidden, params.token_embedding)

    np.testing.assert_allclose(np.asarray(result.logits), np.asarray(expected_logits), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("dropout_field", ("embedding_dropout", "attention_dropout", "residual_dropout"))
def test_training_with_configured_dropout_requires_rng(dropout_field: str) -> None:
    config = _tiny_config(**{dropout_field: 0.25})
    params = init_gpt_neo_params(jax.random.PRNGKey(6), config)
    token_ids = jnp.asarray(((1, 2, 3),), dtype=jnp.int32)

    with pytest.raises(ValueError, match="rng|RNG|key"):
        apply_gpt_neo(
            params,
            config,
            token_ids,
            jnp.ones_like(token_ids, dtype=jnp.bool_),
            training=True,
        )


def test_dropout_is_keyed_and_disabled_during_evaluation() -> None:
    config = _tiny_config(embedding_dropout=0.5, attention_dropout=0.25, residual_dropout=0.25)
    params = init_gpt_neo_params(jax.random.PRNGKey(7), config)
    token_ids = jnp.tile(jnp.arange(6, dtype=jnp.int32)[None, :], (4, 1))
    attention_mask = jnp.ones_like(token_ids, dtype=jnp.bool_)

    first = apply_gpt_neo(
        params,
        config,
        token_ids,
        attention_mask,
        training=True,
        rng_key=jax.random.PRNGKey(8),
    )
    repeated = apply_gpt_neo(
        params,
        config,
        token_ids,
        attention_mask,
        training=True,
        rng_key=jax.random.PRNGKey(8),
    )
    different = apply_gpt_neo(
        params,
        config,
        token_ids,
        attention_mask,
        training=True,
        rng_key=jax.random.PRNGKey(9),
    )
    evaluation = apply_gpt_neo(params, config, token_ids, attention_mask, training=False)

    _assert_tree_allclose(first, repeated, atol=0.0)
    assert not np.array_equal(np.asarray(first.logits), np.asarray(different.logits))
    assert np.isfinite(np.asarray(evaluation.logits)).all()


def test_zero_dropout_training_does_not_require_rng() -> None:
    config = _tiny_config()
    params = init_gpt_neo_params(jax.random.PRNGKey(10), config)
    token_ids = jnp.asarray(((1, 2),), dtype=jnp.int32)

    result = apply_gpt_neo(
        params,
        config,
        token_ids,
        jnp.ones_like(token_ids, dtype=jnp.bool_),
        training=True,
    )

    assert np.isfinite(np.asarray(result.logits)).all()


def test_capture_is_selective_and_preserves_requested_order() -> None:
    config = _tiny_config(num_layers=2, attention_types=("global", "local"))
    params = init_gpt_neo_params(jax.random.PRNGKey(11), config)
    token_ids = jnp.asarray(((1, 2, 3, 4),), dtype=jnp.int32)
    attention_mask = jnp.ones_like(token_ids, dtype=jnp.bool_)
    points = (
        CapturePoint(layer_index=1, location="post_mlp"),
        CapturePoint(layer_index=0, location="post_attention"),
        CapturePoint(layer_index=1, location="post_attention"),
    )

    default_result = apply_gpt_neo(params, config, token_ids, attention_mask)
    combined_result = apply_gpt_neo(
        params,
        config,
        token_ids,
        attention_mask,
        capture=CaptureSpec(points=points),
    )
    individually_captured = tuple(
        apply_gpt_neo(
            params,
            config,
            token_ids,
            attention_mask,
            capture=CaptureSpec(points=(point,)),
        ).captured_hidden[0]
        for point in points
    )

    assert default_result.captured_hidden == ()
    assert len(combined_result.captured_hidden) == len(points)
    for actual, expected in zip(combined_result.captured_hidden, individually_captured):
        assert actual.shape == (token_ids.shape[0], token_ids.shape[1], config.hidden_size)
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ("layer_index", "location"),
    ((-1, "post_attention"), (0, "before_attention")),
)
def test_invalid_capture_points_are_rejected_at_construction(layer_index: int, location: str) -> None:
    with pytest.raises(ValueError):
        CapturePoint(layer_index=layer_index, location=location)  # type: ignore[arg-type]


def test_capture_points_outside_model_are_rejected_at_apply() -> None:
    config = _tiny_config()
    params = init_gpt_neo_params(jax.random.PRNGKey(12), config)
    token_ids = jnp.asarray(((1, 2),), dtype=jnp.int32)

    with pytest.raises(ValueError):
        apply_gpt_neo(
            params,
            config,
            token_ids,
            jnp.ones_like(token_ids, dtype=jnp.bool_),
            capture=CaptureSpec(points=(CapturePoint(layer_index=1, location="post_mlp"),)),
        )


def test_jitted_forward_matches_eager_with_fixed_capture_structure() -> None:
    config = _tiny_config()
    params = init_gpt_neo_params(jax.random.PRNGKey(13), config)
    token_ids = jnp.asarray(((1, 2, 3), (4, 5, 6)), dtype=jnp.int32)
    attention_mask = jnp.ones_like(token_ids, dtype=jnp.bool_)
    capture = CaptureSpec(points=(CapturePoint(0, "post_mlp"),))

    eager_result = apply_gpt_neo(params, config, token_ids, attention_mask, capture=capture)
    compiled_apply = jax.jit(
        lambda model_params, inputs, mask: apply_gpt_neo(
            model_params,
            config,
            inputs,
            mask,
            capture=capture,
        )
    )
    compiled_result = compiled_apply(params, token_ids, attention_mask)

    _assert_tree_allclose(eager_result, compiled_result, rtol=1e-6, atol=1e-6)


def test_language_model_loss_has_finite_parameter_gradients() -> None:
    config = _tiny_config()
    params = init_gpt_neo_params(jax.random.PRNGKey(14), config)
    token_ids = jnp.asarray(((1, 2, 3, 4), (4, 3, 2, 1)), dtype=jnp.int32)
    target_ids = jnp.asarray(((2, 3, 4, 5), (3, 2, 1, 0)), dtype=jnp.int32)
    attention_mask = jnp.ones_like(token_ids, dtype=jnp.bool_)

    gradients = jax.grad(
        lambda model_params: mean_token_nll(
            apply_gpt_neo(model_params, config, token_ids, attention_mask).logits,
            target_ids,
            attention_mask,
        )
    )(params)
    gradient_leaves = jax.tree_util.tree_leaves(gradients)

    assert gradient_leaves
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in gradient_leaves)
    assert any(np.any(np.asarray(leaf) != 0.0) for leaf in gradient_leaves)


def test_token_nll_and_masked_mean_are_normalized_over_valid_targets() -> None:
    logits = jnp.asarray(
        (
            ((3.0, 1.0, -1.0), (0.0, 2.0, 1.0)),
            ((-1.0, 0.5, 2.0), (4.0, 1.0, 0.0)),
        ),
        dtype=jnp.float32,
    )
    targets = jnp.asarray(((0, 1), (2, 1)), dtype=jnp.int32)
    loss_mask = jnp.asarray(((1.0, 0.0), (1.0, 0.0)), dtype=jnp.float32)
    expected_per_token = -jnp.take_along_axis(jax.nn.log_softmax(logits), targets[..., None], axis=-1)[..., 0]

    actual_per_token = per_token_nll(logits, targets)
    actual_mean = mean_token_nll(logits, targets, loss_mask)

    assert actual_per_token.dtype == jnp.float32
    assert actual_mean.dtype == jnp.float32
    np.testing.assert_allclose(np.asarray(actual_per_token), np.asarray(expected_per_token), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(actual_mean),
        np.asarray((expected_per_token[0, 0] + expected_per_token[1, 0]) / 2.0),
        rtol=1e-6,
        atol=1e-6,
    )


def test_empty_loss_mask_is_fail_visible() -> None:
    logits = jnp.asarray((((1.0, 2.0), (3.0, 4.0)),), dtype=jnp.float32)
    targets = jnp.asarray(((0, 1),), dtype=jnp.int32)

    loss = mean_token_nll(logits, targets, jnp.zeros_like(targets, dtype=jnp.float32))

    assert loss.dtype == jnp.float32
    assert np.isnan(np.asarray(loss))


def test_tiny_model_overfits_fixed_batch_within_bounded_steps() -> None:
    config = _tiny_config(
        vocab_size=5,
        max_position_embeddings=4,
        hidden_size=8,
        intermediate_size=16,
        num_heads=2,
    )
    params = init_gpt_neo_params(jax.random.PRNGKey(15), config)
    input_ids = jnp.tile(jnp.asarray(((0, 1, 2, 3),), dtype=jnp.int32), (2, 1))
    target_ids = jnp.tile(jnp.asarray(((1, 2, 3, 4),), dtype=jnp.int32), (2, 1))
    loss_mask = jnp.ones_like(input_ids, dtype=jnp.float32)
    attention_mask = jnp.ones_like(input_ids, dtype=jnp.bool_)
    optimizer = optax.adam(learning_rate=0.02)
    optimizer_state = optimizer.init(params)

    @jax.jit
    def train_step(model_params, state):
        loss, gradients = jax.value_and_grad(
            lambda candidate_params: mean_token_nll(
                apply_gpt_neo(candidate_params, config, input_ids, attention_mask).logits,
                target_ids,
                loss_mask,
            )
        )(model_params)
        updates, next_state = optimizer.update(gradients, state, model_params)
        return optax.apply_updates(model_params, updates), next_state, loss

    initial_loss = mean_token_nll(
        apply_gpt_neo(params, config, input_ids, attention_mask).logits,
        target_ids,
        loss_mask,
    )
    for _ in range(200):
        params, optimizer_state, _ = train_step(params, optimizer_state)
    final_loss = mean_token_nll(
        apply_gpt_neo(params, config, input_ids, attention_mask).logits,
        target_ids,
        loss_mask,
    )

    assert float(final_loss) < float(initial_loss)
    assert float(final_loss) < 0.05

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from apm.lm.attention import apply_attention, apply_linear
from apm.lm.config import GptNeoConfig
from apm.lm.gpt_neo import (
    CapturePoint,
    CaptureSpec,
    apply_gpt_neo,
    apply_layer_norm,
    embed_tokens,
    gelu_new,
)
from apm.lm.lora import (
    LoraConfig,
    LoraEdge,
    LoraProjection,
    LoraTargetMask,
    init_lora_edge,
    insert_lora_edge,
)
from apm.lm.lora_memory import (
    PackedLoraMemory,
    pack_lora_memory,
    packed_with_candidate_edge,
)
from apm.lm.parameters import GptNeoParams, init_gpt_neo_params
from apm.memory.graph import NodeId, TaskId, add_memory_node, init_memory_graph


LORA_SITES = (
    "query",
    "key",
    "value",
    "attention_output",
    "mlp_input",
    "mlp_output",
)


def _model_config() -> GptNeoConfig:
    return GptNeoConfig(
        vocab_size=13,
        max_position_embeddings=8,
        hidden_size=4,
        intermediate_size=8,
        num_layers=1,
        num_heads=2,
        attention_types=("global",),
        local_window_size=3,
    )


def _lora_config(
    *,
    alpha: float = 2.0,
    target_mask: LoraTargetMask | None = None,
) -> LoraConfig:
    return LoraConfig(
        rank=2,
        alpha=alpha,
        target_mask=LoraTargetMask() if target_mask is None else target_mask,
    )


def _inputs() -> tuple[jax.Array, jax.Array]:
    token_ids = jnp.asarray(
        (
            (1, 2, 3, 4),
            (4, 3, 2, 1),
        ),
        dtype=jnp.int32,
    )
    return token_ids, jnp.ones_like(token_ids, dtype=jnp.bool_)


def _edge_with_only_site(
    site: str,
    *,
    multiplier: float = 1.0,
) -> LoraEdge:
    model_config = _model_config()
    edge = jax.tree_util.tree_map(
        jnp.zeros_like,
        init_lora_edge(jax.random.PRNGKey(100), model_config, _lora_config()),
    )
    projection = getattr(edge.blocks[0], site)
    left = multiplier * jnp.linspace(
        -0.35,
        0.55,
        projection.left.size,
        dtype=jnp.float32,
    ).reshape(projection.left.shape)
    right = multiplier * jnp.linspace(
        0.45,
        -0.25,
        projection.right.size,
        dtype=jnp.float32,
    ).reshape(projection.right.shape)
    updated_block = edge.blocks[0]._replace(
        **{site: LoraProjection(left=left, right=right)}
    )
    return edge._replace(blocks=(updated_block,))


def _packed_memory(
    edges: tuple[LoraEdge, ...],
    *,
    lora_config: LoraConfig | None = None,
    max_edges: int | None = None,
) -> PackedLoraMemory:
    graph = init_memory_graph(NodeId("root"))
    for edge_index, edge in enumerate(edges):
        graph = add_memory_node(
            graph,
            NodeId(f"node_{edge_index}"),
            NodeId("root"),
            TaskId(f"task_{edge_index}"),
            edge_index + 1,
            edge,
        )
    capacity = len(edges) if max_edges is None else max_edges
    return pack_lora_memory(
        graph,
        _model_config(),
        _lora_config() if lora_config is None else lora_config,
        max_nodes=capacity + 1,
        max_edges=capacity,
    )


def _apply(
    params: GptNeoParams,
    *,
    token_ids: jax.Array | None = None,
    attention_mask: jax.Array | None = None,
    lora_memory: PackedLoraMemory | None = None,
    edge_coefficients: jax.Array | None = None,
    lora_config: LoraConfig | None = None,
    capture: CaptureSpec = CaptureSpec(),
):
    default_ids, default_mask = _inputs()
    return apply_gpt_neo(
        params,
        _model_config(),
        default_ids if token_ids is None else token_ids,
        default_mask if attention_mask is None else attention_mask,
        lora_memory=lora_memory,
        edge_coefficients=edge_coefficients,
        lora_config=lora_config,
        capture=capture,
    )


def _base_reference(
    params: GptNeoParams,
    token_ids: jax.Array,
    attention_mask: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    config = _model_config()
    position_ids = jnp.arange(token_ids.shape[1], dtype=jnp.int32)[None, :]
    hidden_states = embed_tokens(params, token_ids, position_ids)
    for block, attention_type in zip(params.blocks, config.attention_types):
        attention_input = apply_layer_norm(
            block.attention_norm,
            hidden_states,
            config.layer_norm_epsilon,
        )
        attention_output = apply_attention(
            block.attention,
            config,
            attention_input,
            attention_mask,
            attention_type,
            training=False,
            probability_dropout_key=None,
            output_dropout_key=None,
        )
        post_attention = hidden_states + attention_output
        mlp_input = apply_layer_norm(
            block.mlp_norm,
            post_attention,
            config.layer_norm_epsilon,
        )
        mlp_hidden = gelu_new(apply_linear(block.mlp.input_projection, mlp_input))
        hidden_states = post_attention + apply_linear(
            block.mlp.output_projection,
            mlp_hidden,
        )
    final_hidden = apply_layer_norm(
        params.final_norm,
        hidden_states,
        config.layer_norm_epsilon,
    )
    return final_hidden, jnp.einsum(
        "bth,vh->btv",
        final_hidden,
        params.token_embedding,
    )


def _assert_tree_zero(tree: object) -> None:
    assert all(
        np.count_nonzero(np.asarray(leaf)) == 0
        for leaf in jax.tree_util.tree_leaves(tree)
    )


def test_base_path_is_bitwise_identical_to_the_phase_two_reference() -> None:
    params = init_gpt_neo_params(jax.random.PRNGKey(0), _model_config())
    token_ids, attention_mask = _inputs()
    expected_hidden, expected_logits = _base_reference(
        params,
        token_ids,
        attention_mask,
    )

    actual = _apply(params, token_ids=token_ids, attention_mask=attention_mask)

    np.testing.assert_array_equal(actual.final_hidden, expected_hidden)
    np.testing.assert_array_equal(actual.logits, expected_logits)


def test_zero_right_edge_is_exactly_equal_to_the_base_model() -> None:
    params = init_gpt_neo_params(jax.random.PRNGKey(1), _model_config())
    lora_config = _lora_config()
    zero_effect_edge = init_lora_edge(
        jax.random.PRNGKey(2),
        _model_config(),
        lora_config,
    )
    packed = _packed_memory((zero_effect_edge,), lora_config=lora_config)

    base = _apply(params)
    adapted = _apply(
        params,
        lora_memory=packed,
        edge_coefficients=jnp.asarray((1.0,), dtype=jnp.float32),
        lora_config=lora_config,
    )

    np.testing.assert_array_equal(adapted.final_hidden, base.final_hidden)
    np.testing.assert_array_equal(adapted.logits, base.logits)


@pytest.mark.parametrize("site", LORA_SITES)
def test_each_transformer_projection_site_affects_the_complete_model(site: str) -> None:
    params = init_gpt_neo_params(jax.random.PRNGKey(3), _model_config())
    edge = _edge_with_only_site(site)
    packed = _packed_memory((edge,))

    base = _apply(params)
    adapted = _apply(
        params,
        lora_memory=packed,
        edge_coefficients=jnp.asarray((1.0,), dtype=jnp.float32),
        lora_config=_lora_config(),
    )

    assert not np.array_equal(np.asarray(adapted.logits), np.asarray(base.logits))


def test_alpha_over_rank_scales_one_projection_update_exactly_once() -> None:
    params = init_gpt_neo_params(jax.random.PRNGKey(4), _model_config())
    edge = _edge_with_only_site("mlp_output")
    packed = _packed_memory((edge,))
    capture = CaptureSpec(points=(CapturePoint(0, "post_mlp"),))
    base_hidden = _apply(params, capture=capture).captured_hidden[0]

    half_scale_hidden = _apply(
        params,
        lora_memory=packed,
        edge_coefficients=jnp.asarray((1.0,), dtype=jnp.float32),
        lora_config=_lora_config(alpha=1.0),
        capture=capture,
    ).captured_hidden[0]
    unit_scale_hidden = _apply(
        params,
        lora_memory=packed,
        edge_coefficients=jnp.asarray((1.0,), dtype=jnp.float32),
        lora_config=_lora_config(alpha=2.0),
        capture=capture,
    ).captured_hidden[0]

    np.testing.assert_allclose(
        np.asarray(unit_scale_hidden - base_hidden),
        2.0 * np.asarray(half_scale_hidden - base_hidden),
        rtol=2e-5,
        atol=2e-6,
    )


def test_batched_edge_coefficients_apply_independently_per_example() -> None:
    params = init_gpt_neo_params(jax.random.PRNGKey(5), _model_config())
    packed = _packed_memory(
        (
            _edge_with_only_site("mlp_output", multiplier=0.7),
            _edge_with_only_site("mlp_output", multiplier=1.3),
        )
    )
    token_ids, attention_mask = _inputs()
    batched = _apply(
        params,
        token_ids=token_ids,
        attention_mask=attention_mask,
        lora_memory=packed,
        edge_coefficients=jnp.asarray(((1.0, 0.0), (0.0, 1.0)), dtype=jnp.float32),
        lora_config=_lora_config(),
    )

    separately_applied = tuple(
        _apply(
            params,
            token_ids=token_ids[index : index + 1],
            attention_mask=attention_mask[index : index + 1],
            lora_memory=packed,
            edge_coefficients=jnp.asarray(coefficients, dtype=jnp.float32),
            lora_config=_lora_config(),
        ).logits[0]
        for index, coefficients in enumerate(((1.0, 0.0), (0.0, 1.0)))
    )

    np.testing.assert_allclose(
        np.asarray(batched.logits),
        np.stack(tuple(np.asarray(logits) for logits in separately_applied)),
        rtol=1e-6,
        atol=1e-6,
    )


def test_invalid_edge_slot_is_masked_even_when_its_factors_and_coefficient_are_nonzero() -> None:
    params = init_gpt_neo_params(jax.random.PRNGKey(6), _model_config())
    packed = _packed_memory(
        (_edge_with_only_site("mlp_output"),),
        max_edges=2,
    )
    polluted = packed._replace(
        edge_bank=insert_lora_edge(
            packed.edge_bank,
            _edge_with_only_site("mlp_output", multiplier=20.0),
            1,
        )
    )

    base = _apply(params)
    actual = _apply(
        params,
        lora_memory=polluted,
        edge_coefficients=jnp.asarray((0.0, 100.0), dtype=jnp.float32),
        lora_config=_lora_config(),
    )

    np.testing.assert_array_equal(actual.logits, base.logits)


def test_static_target_mask_disables_a_nonzero_projection_without_changing_the_bank() -> None:
    params = init_gpt_neo_params(jax.random.PRNGKey(7), _model_config())
    edge = _edge_with_only_site("query")
    packed = _packed_memory((edge,))
    disabled_config = _lora_config(
        target_mask=replace(LoraTargetMask(), query=False)
    )

    base = _apply(params)
    disabled = _apply(
        params,
        lora_memory=packed,
        edge_coefficients=jnp.asarray((1.0,), dtype=jnp.float32),
        lora_config=disabled_config,
    )

    np.testing.assert_array_equal(disabled.final_hidden, base.final_hidden)
    np.testing.assert_array_equal(disabled.logits, base.logits)
    assert jax.tree_util.tree_structure(packed.edge_bank) == jax.tree_util.tree_structure(
        _packed_memory((edge,), lora_config=disabled_config).edge_bank
    )


def test_candidate_training_has_gradients_only_for_the_candidate_edge() -> None:
    params = init_gpt_neo_params(jax.random.PRNGKey(8), _model_config())
    committed = _edge_with_only_site("mlp_output", multiplier=0.5)
    candidate = _edge_with_only_site("mlp_output", multiplier=0.9)
    packed = _packed_memory((committed,), max_edges=2)
    coefficients = jnp.asarray((1.0, 1.0), dtype=jnp.float32)

    def loss(candidate_edge, base_params, committed_bank):
        frozen_base = jax.tree_util.tree_map(jax.lax.stop_gradient, base_params)
        candidate_memory = packed_with_candidate_edge(
            packed._replace(edge_bank=committed_bank),
            candidate_edge,
            1,
        )
        logits = _apply(
            frozen_base,
            lora_memory=candidate_memory,
            edge_coefficients=coefficients,
            lora_config=_lora_config(),
        ).logits
        return jnp.mean(jnp.square(logits))

    candidate_gradient, base_gradient, committed_gradient = jax.grad(
        loss,
        argnums=(0, 1, 2),
    )(candidate, params, packed.edge_bank)

    candidate_leaves = jax.tree_util.tree_leaves(candidate_gradient)
    assert candidate_leaves
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in candidate_leaves)
    assert any(np.count_nonzero(np.asarray(leaf)) > 0 for leaf in candidate_leaves)
    _assert_tree_zero(base_gradient)
    _assert_tree_zero(committed_gradient)


def test_filling_a_later_slot_keeps_old_node_logits_bitwise_stable() -> None:
    params = init_gpt_neo_params(jax.random.PRNGKey(9), _model_config())
    committed = _edge_with_only_site("mlp_output", multiplier=0.6)
    packed = _packed_memory((committed,), max_edges=2)
    old_coefficients = jnp.asarray((1.0, 0.0), dtype=jnp.float32)
    before = _apply(
        params,
        lora_memory=packed,
        edge_coefficients=old_coefficients,
        lora_config=_lora_config(),
    )
    after_commit = packed_with_candidate_edge(
        packed,
        _edge_with_only_site("mlp_output", multiplier=10.0),
        1,
    )

    after = _apply(
        params,
        lora_memory=after_commit,
        edge_coefficients=old_coefficients,
        lora_config=_lora_config(),
    )

    np.testing.assert_array_equal(after.final_hidden, before.final_hidden)
    np.testing.assert_array_equal(after.logits, before.logits)

from __future__ import annotations

from inspect import Parameter, signature
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from apm.continual.language_tasks import RouterBatch
from apm.lm.config import GptNeoConfig
from apm.lm.gpt_neo import ForwardResult
from apm.lm.lora import LoraConfig, stack_lora_edges
from apm.lm.lora_memory import PackedLoraMemory
from apm.lm.parameters import init_gpt_neo_params
from apm.memory import prefix_energy
from apm.memory.prefix_energy import (
    exhaustive_prefix_nll_address,
    exhaustive_prefix_nll_core,
)


def _model_config() -> GptNeoConfig:
    return GptNeoConfig(
        vocab_size=3,
        max_position_embeddings=8,
        hidden_size=4,
        intermediate_size=8,
        num_layers=1,
        num_heads=2,
        attention_types=("global",),
        local_window_size=2,
    )


def _lora_config() -> LoraConfig:
    return LoraConfig(rank=2, alpha=2.0)


def _packed_memory(max_nodes: int = 4) -> PackedLoraMemory:
    max_edges = max_nodes - 1
    path_rows = (
        jnp.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (9.0, 9.0, 9.0),
            ),
            dtype=jnp.float32,
        )
        if max_nodes == 4
        else jnp.zeros((1, 0), dtype=jnp.float32)
    )
    return PackedLoraMemory(
        edge_bank=stack_lora_edges(
            (),
            _model_config(),
            _lora_config(),
            max_edges,
        ),
        node_path_matrix=path_rows,
        valid_node_mask=(
            jnp.asarray((True, True, True, False))
            if max_nodes == 4
            else jnp.asarray((True,))
        ),
        valid_edge_mask=(
            jnp.asarray((True, True, False))
            if max_nodes == 4
            else jnp.zeros((0,), dtype=jnp.bool_)
        ),
    )


def _router_batch(
    desired_nodes: tuple[int, ...],
    attention_mask: np.ndarray | None = None,
    target_ids: np.ndarray | None = None,
) -> RouterBatch:
    batch_size = len(desired_nodes)
    sequence_length = 3
    input_ids = np.zeros((batch_size, sequence_length), dtype=np.int32)
    input_ids[:, 0] = desired_nodes
    resolved_attention_mask = (
        np.ones_like(input_ids, dtype=np.bool_)
        if attention_mask is None
        else attention_mask
    )
    return RouterBatch(
        input_ids=input_ids,
        attention_mask=resolved_attention_mask,
        target_ids=(
            np.zeros_like(input_ids, dtype=np.int32)
            if target_ids is None
            else target_ids
        ),
        loss_mask=resolved_attention_mask,
    )


def _install_path_sensitive_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def apply_path_sensitive_model(
        params,
        config,
        token_ids,
        attention_mask,
        *,
        position_ids=None,
        lora_memory=None,
        edge_coefficients=None,
        lora_config=None,
        capture=None,
        training=False,
        rng_key=None,
    ) -> ForwardResult:
        del params, attention_mask, position_ids, lora_config, capture, training, rng_key
        assert lora_memory is not None
        assert edge_coefficients is not None
        edge_numbers = jnp.arange(
            1,
            edge_coefficients.shape[0] + 1,
            dtype=jnp.float32,
        )
        routed_node = jnp.sum(edge_coefficients * edge_numbers)
        desired_node = token_ids[:, 0].astype(jnp.float32)
        target_logit = -3.0 * jnp.abs(desired_node - routed_node)
        logits = jnp.zeros(
            (*token_ids.shape, config.vocab_size),
            dtype=jnp.float32,
        ).at[:, :, 0].set(target_logit[:, None])
        return ForwardResult(
            logits=logits,
            final_hidden=jnp.zeros(
                (*token_ids.shape, config.hidden_size),
                dtype=jnp.float32,
            ),
            captured_hidden=(),
        )

    monkeypatch.setattr(prefix_energy, "apply_gpt_neo", apply_path_sensitive_model)


def _route(batch: RouterBatch):
    config = _model_config()
    return exhaustive_prefix_nll_address(
        init_gpt_neo_params(jax.random.PRNGKey(0), config),
        config,
        _packed_memory(),
        _lora_config(),
        batch,
    )


def test_known_best_nodes_are_selected_independently_and_padding_is_masked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_path_sensitive_model(monkeypatch)
    batch = _router_batch((2, 0, 1))

    result = _route(batch)

    np.testing.assert_array_equal(result.selected_indices, (2, 0, 1))
    assert result.selected_indices.shape == (3,)
    assert result.selected_indices.dtype == jnp.int32
    assert result.node_scores.shape == (3, 4)
    assert result.node_scores.dtype == jnp.float32
    assert result.node_probabilities.shape == (3, 4)
    assert result.node_probabilities.dtype == jnp.float32
    assert result.score_margin.shape == (3,)
    assert result.score_margin.dtype == jnp.float32
    assert result.entropy.shape == (3,)
    assert result.entropy.dtype == jnp.float32
    assert np.all(np.isinf(np.asarray(result.node_scores[:, 3])))
    np.testing.assert_array_equal(result.node_probabilities[:, 3], 0.0)

    single_results = tuple(_route(_router_batch((node,))) for node in (2, 0, 1))
    for field_index, batched_field in enumerate(result):
        np.testing.assert_allclose(
            np.asarray(batched_field),
            np.concatenate(
                tuple(np.asarray(single[field_index]) for single in single_results),
                axis=0,
            ),
            rtol=1e-6,
            atol=1e-6,
        )


def test_probabilities_margins_and_entropy_are_derived_from_prefix_nll(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_path_sensitive_model(monkeypatch)

    result = _route(_router_batch((1,)))
    valid_scores = np.asarray(result.node_scores[0, :3])
    unnormalized = np.exp(-valid_scores - np.max(-valid_scores))
    expected_probabilities = unnormalized / np.sum(unnormalized)
    expected_entropy = -np.sum(
        expected_probabilities * np.log(expected_probabilities)
    )

    np.testing.assert_allclose(
        result.node_probabilities[0, :3],
        expected_probabilities,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        result.score_margin[0],
        np.sort(valid_scores)[1] - np.sort(valid_scores)[0],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        result.entropy[0],
        expected_entropy,
        rtol=1e-6,
        atol=1e-6,
    )


def test_scores_normalize_valid_tokens_and_ignore_masked_padding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_path_sensitive_model(monkeypatch)
    attention_mask = np.asarray(
        ((True, True, True), (True, False, False)),
        dtype=np.bool_,
    )
    target_ids = np.asarray(((0, 0, 0), (0, 2, 1)), dtype=np.int32)

    result = _route(
        _router_batch(
            (1, 1),
            attention_mask=attention_mask,
            target_ids=target_ids,
        )
    )

    np.testing.assert_allclose(
        result.node_scores[0, :3],
        result.node_scores[1, :3],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        result.node_probabilities[0],
        result.node_probabilities[1],
        rtol=1e-6,
        atol=1e-6,
    )


def test_microbatched_router_matches_full_scores_and_bounds_every_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_path_sensitive_model(monkeypatch)
    config = _model_config()
    params = init_gpt_neo_params(jax.random.PRNGKey(0), config)
    packed_memory = _packed_memory()
    lora_config = _lora_config()
    batch = _router_batch((2, 0, 1, 2, 1))
    expected = exhaustive_prefix_nll_address(
        params,
        config,
        packed_memory,
        lora_config,
        batch,
    )
    original_apply = prefix_energy.apply_gpt_neo
    observed_batch_sizes: list[int] = []

    def counted_apply(*args, **kwargs):
        observed_batch_sizes.append(int(args[2].shape[0]))
        return original_apply(*args, **kwargs)

    monkeypatch.setattr(prefix_energy, "apply_gpt_neo", counted_apply)
    actual = exhaustive_prefix_nll_address(
        params,
        config,
        packed_memory,
        lora_config,
        batch,
        evaluation_microbatch_size=2,
    )

    for expected_field, actual_field in zip(expected, actual):
        np.testing.assert_allclose(
            actual_field,
            expected_field,
            rtol=1e-6,
            atol=1e-6,
        )
    assert observed_batch_sizes
    assert max(observed_batch_sizes) == 2
    assert len(observed_batch_sizes) == 9


def test_router_rejects_a_row_without_loss_tokens() -> None:
    empty_mask = np.zeros((1, 3), dtype=np.bool_)
    malformed_batch = SimpleNamespace(
        input_ids=np.zeros((1, 3), dtype=np.int32),
        attention_mask=empty_mask,
        target_ids=np.zeros((1, 3), dtype=np.int32),
        loss_mask=empty_mask,
    )
    config = _model_config()

    with pytest.raises(ValueError, match="at least one loss token"):
        exhaustive_prefix_nll_address(
            init_gpt_neo_params(jax.random.PRNGKey(0), config),
            config,
            _packed_memory(),
            _lora_config(),
            malformed_batch,  # type: ignore[arg-type]
        )


def test_root_only_memory_has_certain_probability_and_infinite_margin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_path_sensitive_model(monkeypatch)
    config = _model_config()

    result = exhaustive_prefix_nll_address(
        init_gpt_neo_params(jax.random.PRNGKey(0), config),
        config,
        _packed_memory(max_nodes=1),
        _lora_config(),
        _router_batch((0,)),
    )

    np.testing.assert_array_equal(result.selected_indices, (0,))
    np.testing.assert_array_equal(result.node_probabilities, ((1.0,),))
    assert np.isposinf(np.asarray(result.score_margin[0]))
    np.testing.assert_allclose(result.entropy, 0.0, atol=0.0)


def test_array_core_is_deterministic_under_jit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_path_sensitive_model(monkeypatch)
    config = _model_config()
    params = init_gpt_neo_params(jax.random.PRNGKey(0), config)
    packed_memory = _packed_memory()
    lora_config = _lora_config()
    batch = _router_batch((0, 2))

    route_arrays = lambda input_ids, attention_mask, target_ids, loss_mask: (
        exhaustive_prefix_nll_core(
            params,
            config,
            packed_memory,
            lora_config,
            input_ids,
            attention_mask,
            target_ids,
            loss_mask,
        )
    )
    arrays = tuple(jnp.asarray(leaf) for leaf in batch.tree_flatten()[0])
    eager = route_arrays(*arrays)
    compiled = jax.jit(route_arrays)(*arrays)
    repeated = jax.jit(route_arrays)(*arrays)

    for expected, actual, repeated_actual in zip(eager, compiled, repeated):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(repeated_actual, actual)


def test_router_api_contains_only_frozen_state_and_prefix_batch() -> None:
    router_signature = signature(exhaustive_prefix_nll_address)

    assert tuple(router_signature.parameters) == (
        "base_params",
        "model_config",
        "packed_memory",
        "lora_config",
        "prefix_batch",
        "evaluation_microbatch_size",
    )
    assert all(
        parameter.kind
        not in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD)
        for parameter in router_signature.parameters.values()
    )
    assert set(RouterBatch.__dataclass_fields__) == {
        "input_ids",
        "attention_mask",
        "target_ids",
        "loss_mask",
    }
    forbidden_fragments = ("task", "oracle", "suffix")
    assert not any(
        fragment in parameter_name.lower()
        for parameter_name in router_signature.parameters
        for fragment in forbidden_fragments
    )

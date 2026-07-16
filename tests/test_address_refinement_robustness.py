from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import struct

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from apm.continual.language_tasks import RouterBatch
from apm.lm.config import GptNeoConfig
from apm.lm.gpt_neo import ForwardResult
from apm.lm.lora import LoraConfig, init_lora_edge, stack_lora_edges
from apm.lm.lora_memory import (
    PackedLoraMemory,
    edge_coefficients_for_node,
    node_weights_to_edge_coefficients,
)
from apm.lm.parameters import GptNeoParams, init_gpt_neo_params
import apm.memory.address_refinement as refinement_module
from apm.memory.address_refinement import (
    EbtConfig,
    masked_node_probabilities,
    refine_ebt_address,
    soft_mixture_prefix_nll,
)
from apm.memory.content_addressing import HopfieldAddressResult


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


def _lora_config(alpha: float = 2.0) -> LoraConfig:
    return LoraConfig(rank=2, alpha=alpha)


def _params() -> GptNeoParams:
    return init_gpt_neo_params(jax.random.PRNGKey(0), _model_config())


def _packed_memory() -> PackedLoraMemory:
    lora_config = _lora_config()
    edges = tuple(
        init_lora_edge(jax.random.PRNGKey(index + 1), _model_config(), lora_config)
        for index in range(2)
    )
    return PackedLoraMemory(
        edge_bank=stack_lora_edges(edges, _model_config(), lora_config, max_edges=3),
        node_path_matrix=jnp.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (9.0, 9.0, 9.0),
            ),
            dtype=jnp.float32,
        ),
        valid_node_mask=jnp.asarray((True, True, True, False)),
        valid_edge_mask=jnp.asarray((True, True, False)),
    )


def _router_batch(desired_nodes: tuple[int, ...]) -> RouterBatch:
    input_ids = np.zeros((len(desired_nodes), 3), dtype=np.int32)
    input_ids[:, 0] = desired_nodes
    mask = np.ones_like(input_ids, dtype=np.bool_)
    return RouterBatch(
        input_ids=input_ids,
        attention_mask=mask,
        target_ids=np.zeros_like(input_ids),
        loss_mask=mask,
    )


def _slice_batch(batch: RouterBatch, row: int) -> RouterBatch:
    return RouterBatch(
        input_ids=batch.input_ids[row : row + 1],
        attention_mask=batch.attention_mask[row : row + 1],
        target_ids=batch.target_ids[row : row + 1],
        loss_mask=batch.loss_mask[row : row + 1],
    )


def _hopfield_result() -> HopfieldAddressResult:
    return HopfieldAddressResult(
        selected_indices=jnp.asarray((0,), dtype=jnp.int32),
        node_probabilities=jnp.asarray(((0.6, 0.3, 0.1, 0.0),), dtype=jnp.float32),
        node_scores=jnp.asarray(((1.0, 0.5, 0.0, -jnp.inf),), dtype=jnp.float32),
        score_margin=jnp.asarray((0.5,), dtype=jnp.float32),
        entropy=jnp.asarray((0.9,), dtype=jnp.float32),
        top_k_indices=jnp.asarray(((0, 1),), dtype=jnp.int32),
    )


def _install_path_sensitive_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def apply_path_sensitive_model(
        params: GptNeoParams,
        config: GptNeoConfig,
        token_ids: jax.Array,
        attention_mask: jax.Array,
        *,
        position_ids: jax.Array | None = None,
        lora_memory: PackedLoraMemory | None = None,
        edge_coefficients: jax.Array | None = None,
        lora_config: LoraConfig | None = None,
        capture: object = None,
        training: bool = False,
        rng_key: jax.Array | None = None,
    ) -> ForwardResult:
        del params, attention_mask, position_ids, lora_config, capture, training, rng_key
        assert lora_memory is not None
        assert edge_coefficients is not None
        edge_numbers = jnp.arange(
            1,
            edge_coefficients.shape[-1] + 1,
            dtype=jnp.float32,
        )
        routed_node = jnp.sum(edge_coefficients * edge_numbers, axis=-1)
        desired_node = token_ids[:, 0].astype(jnp.float32)
        target_logit = -4.0 * jnp.abs(desired_node - routed_node)
        logits = jnp.zeros((*token_ids.shape, config.vocab_size), dtype=jnp.float32)
        logits = logits.at[:, :, 0].set(target_logit[:, None])
        return ForwardResult(
            logits=logits,
            final_hidden=jnp.zeros((*token_ids.shape, config.hidden_size)),
            captured_hidden=(),
        )

    monkeypatch.setattr(
        refinement_module,
        "apply_gpt_neo",
        apply_path_sensitive_model,
    )


def _install_zero_nll(monkeypatch: pytest.MonkeyPatch) -> None:
    def zero_nll(
        base_params: GptNeoParams,
        model_config: GptNeoConfig,
        packed_memory: PackedLoraMemory,
        lora_config: LoraConfig,
        prefix_batch: RouterBatch,
        node_probabilities: jax.Array,
    ) -> jax.Array:
        del base_params, model_config, packed_memory, lora_config, prefix_batch
        return jnp.zeros((node_probabilities.shape[0],), dtype=jnp.float32)

    monkeypatch.setattr(refinement_module, "soft_mixture_prefix_nll", zero_nll)


def test_entropy_penalty_has_the_documented_positive_low_entropy_sign(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_zero_nll(monkeypatch)
    initial_logits = jnp.asarray(((8.0, 0.0, 0.0, -jnp.inf),), dtype=jnp.float32)
    initial_probabilities = masked_node_probabilities(
        initial_logits,
        jnp.asarray(((True, True, True, False),)),
        tau=1.0,
    )

    result = refine_ebt_address(
        _params(),
        _model_config(),
        _packed_memory(),
        _lora_config(),
        _router_batch((0,)),
        EbtConfig(
            steps=3,
            learning_rate=0.1,
            tau=1.0,
            entropy_penalty=1.0,
            initialization="full_node",
        ),
        initial_node_indices=np.asarray((0,), dtype=np.int32),
    )

    assert result.node_probabilities[0, 0] > initial_probabilities[0, 0]
    assert result.objective_trace[-1, 0] < result.objective_trace[0, 0]


def test_masked_softmax_has_exact_zero_mass_and_zero_gradient() -> None:
    logits = jnp.asarray(((2.0, 100.0, 0.0), (-1.0, 0.5, -100.0)))
    candidate_mask = jnp.asarray(((True, False, True), (False, True, True)))

    probabilities = masked_node_probabilities(logits, candidate_mask, tau=0.7)
    gradients = jax.grad(
        lambda values: jnp.sum(
            masked_node_probabilities(values, candidate_mask, tau=0.7)
            * jnp.asarray(((1.0, 7.0, 2.0), (8.0, 3.0, 4.0)))
        )
    )(logits)

    np.testing.assert_array_equal(probabilities[~candidate_mask], 0.0)
    np.testing.assert_array_equal(gradients[~candidate_mask], 0.0)
    np.testing.assert_allclose(jnp.sum(probabilities, axis=-1), 1.0, atol=1e-7)


def test_batched_sum_objective_matches_independent_refinements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_path_sensitive_model(monkeypatch)
    batch = _router_batch((1, 2))
    config = EbtConfig(
        steps=5,
        learning_rate=0.05,
        tau=1.0,
        entropy_penalty=0.0,
    )

    batched = refine_ebt_address(
        _params(),
        _model_config(),
        _packed_memory(),
        _lora_config(),
        batch,
        config,
    )
    independent = tuple(
        refine_ebt_address(
            _params(),
            _model_config(),
            _packed_memory(),
            _lora_config(),
            _slice_batch(batch, row),
            config,
        )
        for row in range(2)
    )

    np.testing.assert_allclose(
        batched.final_node_logits,
        jnp.concatenate(tuple(value.final_node_logits for value in independent)),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        batched.node_probabilities,
        jnp.concatenate(tuple(value.node_probabilities for value in independent)),
        rtol=1e-6,
        atol=1e-6,
    )
    for row, value in enumerate(independent):
        np.testing.assert_allclose(
            batched.objective_trace[:, row],
            value.objective_trace[:, 0],
            rtol=1e-6,
            atol=1e-6,
        )
    assert np.all(np.isfinite(np.asarray(batched.objective_trace)))


def test_full_node_refines_all_valid_nodes_but_hopfield_top_k_is_restricted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_zero_nll(monkeypatch)
    common_arguments = (
        _params(),
        _model_config(),
        _packed_memory(),
        _lora_config(),
        _router_batch((0,)),
    )
    full_node = refine_ebt_address(
        *common_arguments,
        EbtConfig(steps=1, entropy_penalty=0.0, initialization="full_node"),
        initial_node_indices=np.asarray((1,), dtype=np.int32),
    )
    top_k = refine_ebt_address(
        *common_arguments,
        EbtConfig(steps=1, entropy_penalty=0.0, initialization="hopfield_top_k"),
        hopfield_result=_hopfield_result(),
    )

    assert np.all(np.isfinite(np.asarray(full_node.final_node_logits[0, :3])))
    assert np.all(np.asarray(full_node.node_probabilities[0, :3]) > 0.0)
    assert np.isneginf(full_node.final_node_logits[0, 3])
    np.testing.assert_array_equal(top_k.node_probabilities[0, 2:], 0.0)
    assert np.all(np.isfinite(np.asarray(top_k.final_node_logits[0, :2])))


def test_one_hot_weights_map_to_the_unscaled_hard_path() -> None:
    packed_memory = _packed_memory()
    one_hot_node = jax.nn.one_hot(
        jnp.asarray((2,)),
        packed_memory.node_path_matrix.shape[0],
    )

    soft_coefficients = node_weights_to_edge_coefficients(one_hot_node, packed_memory)
    hard_coefficients = edge_coefficients_for_node(packed_memory, 2)

    np.testing.assert_array_equal(soft_coefficients[0], hard_coefficients)
    np.testing.assert_array_equal(
        soft_coefficients[0],
        packed_memory.node_path_matrix[2] * packed_memory.valid_edge_mask,
    )
    assert np.max(np.asarray(soft_coefficients)) == 1.0
    assert _lora_config(alpha=20.0).scale != _lora_config(alpha=2.0).scale


def test_only_logits_receive_gradients_and_refinement_preserves_frozen_hashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_path_sensitive_model(monkeypatch)
    params = _params()
    packed_memory = _packed_memory()
    prefix_batch = _router_batch((2,))
    base_hash = _tree_hash(params)
    memory_hash = _tree_hash(packed_memory)
    candidate_mask = jnp.asarray(((True, True, True, False),))

    def objective(
        logits: jax.Array,
        current_params: GptNeoParams,
        current_memory: PackedLoraMemory,
    ) -> jax.Array:
        probabilities = masked_node_probabilities(logits, candidate_mask, tau=1.0)
        return jnp.sum(
            soft_mixture_prefix_nll(
                current_params,
                _model_config(),
                current_memory,
                _lora_config(),
                prefix_batch,
                probabilities,
            )
        )

    logits_gradient, base_gradient, memory_gradient = jax.grad(
        objective,
        argnums=(0, 1, 2),
        allow_int=True,
    )(
        jnp.zeros((1, 4), dtype=jnp.float32),
        params,
        packed_memory,
    )
    result = refine_ebt_address(
        params,
        _model_config(),
        packed_memory,
        _lora_config(),
        prefix_batch,
        EbtConfig(steps=2, entropy_penalty=0.0),
    )

    assert np.any(np.abs(np.asarray(logits_gradient)) > 0.0)
    _assert_zero_gradient_tree(base_gradient)
    _assert_zero_gradient_tree(memory_gradient)
    assert _tree_hash(params) == base_hash
    assert _tree_hash(packed_memory) == memory_hash
    assert result.objective_trace.shape == (3, 1)


def test_default_budget_temperature_and_config_are_fixed_and_immutable() -> None:
    config = EbtConfig()

    assert config.steps == 20
    assert config.learning_rate == 0.1
    assert config.tau == 1.0
    assert config.entropy_penalty == 0.01
    assert config.initialization == "uniform"
    with pytest.raises(FrozenInstanceError):
        config.tau = 2.0  # type: ignore[misc]
    with pytest.raises(ValueError, match="positive"):
        EbtConfig(steps=0)
    with pytest.raises(ValueError, match="positive"):
        EbtConfig(tau=0.0)


def _tree_hash(tree: object) -> str:
    digest = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(tree):
        array = np.asarray(leaf)
        descriptor = f"{array.dtype.str}:{array.shape}".encode("ascii")
        digest.update(struct.pack("<Q", len(descriptor)))
        digest.update(descriptor)
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _assert_zero_gradient_tree(tree: object) -> None:
    for leaf in jax.tree_util.tree_leaves(tree):
        array = np.asarray(leaf)
        if array.dtype == jax.dtypes.float0:
            continue
        np.testing.assert_array_equal(array, np.zeros_like(array))

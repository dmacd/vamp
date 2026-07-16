from __future__ import annotations

from hashlib import sha256

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from apm.continual.language_tasks import AddressBook, NodeId, RouterBatch
from apm.lm.checkpoint import parameter_checksum
from apm.lm.config import GptNeoConfig
from apm.lm.gpt_neo import ForwardResult
from apm.lm.lora import LoraConfig, stack_lora_edges
from apm.lm.lora_memory import (
    PackedLoraMemory,
    edge_coefficients_for_node,
)
from apm.lm.losses import per_token_nll
from apm.lm.parameters import init_gpt_neo_params
import apm.memory.address_refinement as refinement_module
from apm.memory.address_refinement import (
    EbtAddressResult,
    EbtConfig,
    masked_node_probabilities,
    refine_ebt_address,
    soft_mixture_prefix_nll,
)
from apm.memory.content_addressing import (
    HopfieldAddressResult,
    HopfieldConfig,
    hopfield_address,
)


def _model_config() -> GptNeoConfig:
    return GptNeoConfig(
        vocab_size=5,
        max_position_embeddings=4,
        hidden_size=4,
        intermediate_size=8,
        num_layers=1,
        num_heads=2,
        attention_types=("global",),
        local_window_size=4,
    )


def _lora_config() -> LoraConfig:
    return LoraConfig(rank=1, alpha=1.0)


def _branching_memory() -> PackedLoraMemory:
    return PackedLoraMemory(
        edge_bank=stack_lora_edges(
            (),
            _model_config(),
            _lora_config(),
            max_edges=4,
        ),
        node_path_matrix=jnp.asarray(
            (
                (0.0, 0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (1.0, 0.0, 1.0, 0.0),
                (9.0, 9.0, 9.0, 9.0),
            ),
            dtype=jnp.float32,
        ),
        valid_node_mask=jnp.asarray((True, True, True, True, False)),
        valid_edge_mask=jnp.asarray((True, True, True, False)),
    )


def _prefix_batch(desired_nodes: tuple[int, ...]) -> RouterBatch:
    input_ids = np.ones((len(desired_nodes), 3), dtype=np.int32)
    input_ids[:, 0] = desired_nodes
    mask = np.ones_like(input_ids, dtype=np.bool_)
    return RouterBatch(
        input_ids=input_ids,
        attention_mask=mask,
        target_ids=np.zeros_like(input_ids),
        loss_mask=mask,
    )


def _hopfield_start(packed_memory: PackedLoraMemory) -> HopfieldAddressResult:
    address_book = AddressBook(
        node_ids=(
            NodeId("root"),
            NodeId("left"),
            NodeId("right"),
            NodeId("left-child"),
            None,
        ),
        keys=np.asarray(
            (
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
                (0.0, 0.0, 0.0, 0.0),
            ),
            dtype=np.float32,
        ),
        valid_node_mask=np.asarray(packed_memory.valid_node_mask),
    )
    return hopfield_address(
        np.asarray(
            (
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
            dtype=np.float32,
        ),
        address_book,
        HopfieldConfig(beta=1.0, top_k=4),
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
        coefficients = jnp.asarray(edge_coefficients, dtype=jnp.float32)
        if coefficients.ndim == 1:
            coefficients = jnp.broadcast_to(
                coefficients[None, :],
                (token_ids.shape[0], coefficients.shape[0]),
            )
        desired_coefficients = lora_memory.node_path_matrix[token_ids[:, 0]]
        squared_path_distance = jnp.sum(
            jnp.square(coefficients - desired_coefficients)
            * lora_memory.valid_edge_mask,
            axis=-1,
        )
        target_logit = 4.0 - 8.0 * squared_path_distance
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

    monkeypatch.setattr(
        refinement_module,
        "apply_gpt_neo",
        apply_path_sensitive_model,
    )


def _tree_checksum(tree: object) -> str:
    digest = sha256()
    for leaf in jax.tree_util.tree_leaves(tree):
        array = np.asarray(leaf)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _assert_batch_matches_single_results(
    batched: EbtAddressResult,
    singles: tuple[EbtAddressResult, ...],
) -> None:
    batch_axis_fields = (
        "final_node_logits",
        "node_probabilities",
        "edge_coefficients",
        "selected_indices",
        "soft_mixture_nll",
        "hard_node_nll",
    )
    for field_name in batch_axis_fields:
        np.testing.assert_allclose(
            getattr(batched, field_name),
            np.concatenate(
                tuple(np.asarray(getattr(single, field_name)) for single in singles),
                axis=0,
            ),
            rtol=1e-5,
            atol=1e-5,
        )
    np.testing.assert_allclose(
        batched.objective_trace,
        np.concatenate(
            tuple(np.asarray(single.objective_trace) for single in singles),
            axis=1,
        ),
        rtol=1e-5,
        atol=1e-5,
    )


def test_uniform_and_hopfield_ebt_refine_independent_branch_addresses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_path_sensitive_model(monkeypatch)
    model_config = _model_config()
    lora_config = _lora_config()
    base_params = init_gpt_neo_params(jax.random.PRNGKey(3), model_config)
    packed_memory = _branching_memory()
    prefix_batch = _prefix_batch((2, 3))
    expected_nodes = np.asarray((2, 3), dtype=np.int32)
    base_checksum = parameter_checksum(base_params, model_config)
    bank_checksum = _tree_checksum(packed_memory.edge_bank)
    path_snapshot = np.asarray(packed_memory.node_path_matrix).copy()
    valid_node_snapshot = np.asarray(packed_memory.valid_node_mask).copy()
    valid_edge_snapshot = np.asarray(packed_memory.valid_edge_mask).copy()
    candidate_mask = jnp.broadcast_to(
        packed_memory.valid_node_mask[None, :],
        (2, 5),
    )
    uniform_start = masked_node_probabilities(
        jnp.zeros((2, 5), dtype=jnp.float32),
        candidate_mask,
        1.0,
    )
    hopfield_result = _hopfield_start(packed_memory)
    uniform_initial_nll = soft_mixture_prefix_nll(
        base_params,
        model_config,
        packed_memory,
        lora_config,
        prefix_batch,
        uniform_start,
    )
    hopfield_initial_nll = soft_mixture_prefix_nll(
        base_params,
        model_config,
        packed_memory,
        lora_config,
        prefix_batch,
        hopfield_result.node_probabilities,
    )
    uniform_config = EbtConfig(
        steps=20,
        learning_rate=0.2,
        tau=1.0,
        entropy_penalty=0.01,
        initialization="uniform",
    )
    hopfield_config = EbtConfig(
        steps=20,
        learning_rate=0.2,
        tau=1.0,
        entropy_penalty=0.01,
        initialization="hopfield",
    )

    uniform_result = refine_ebt_address(
        base_params,
        model_config,
        packed_memory,
        lora_config,
        prefix_batch,
        uniform_config,
    )
    hopfield_refined = refine_ebt_address(
        base_params,
        model_config,
        packed_memory,
        lora_config,
        prefix_batch,
        hopfield_config,
        hopfield_result=hopfield_result,
    )

    for result, initial_nll in (
        (uniform_result, uniform_initial_nll),
        (hopfield_refined, hopfield_initial_nll),
    ):
        np.testing.assert_array_equal(result.selected_indices, expected_nodes)
        assert result.objective_trace.shape == (21, 2)
        assert np.all(np.isfinite(np.asarray(result.objective_trace)))
        assert np.all(
            np.asarray(result.objective_trace[-1])
            < np.asarray(result.objective_trace[0])
        )
        assert np.all(np.asarray(result.soft_mixture_nll) < np.asarray(initial_nll))
        np.testing.assert_array_equal(result.node_probabilities[:, 4], 0.0)
        assert np.all(np.isneginf(np.asarray(result.final_node_logits[:, 4])))
        np.testing.assert_allclose(
            np.sum(np.asarray(result.node_probabilities), axis=-1),
            1.0,
            rtol=1e-6,
            atol=1e-6,
        )

    one_hot_nodes = jax.nn.one_hot(
        jnp.asarray(expected_nodes),
        packed_memory.node_path_matrix.shape[0],
        dtype=jnp.float32,
    )
    one_hot_soft_nll = soft_mixture_prefix_nll(
        base_params,
        model_config,
        packed_memory,
        lora_config,
        prefix_batch,
        one_hot_nodes,
    )
    hard_coefficients = jax.vmap(
        lambda node_index: edge_coefficients_for_node(
            packed_memory,
            node_index,
        )
    )(jnp.asarray(expected_nodes))
    hard_logits = refinement_module.apply_gpt_neo(
        base_params,
        model_config,
        jnp.asarray(prefix_batch.input_ids),
        jnp.asarray(prefix_batch.attention_mask),
        lora_memory=packed_memory,
        edge_coefficients=hard_coefficients,
        lora_config=lora_config,
        training=False,
    ).logits
    hard_token_nll = per_token_nll(
        hard_logits,
        jnp.asarray(prefix_batch.target_ids),
    )
    hard_loss_mask = jnp.asarray(prefix_batch.loss_mask, dtype=jnp.float32)
    discrete_hard_nll = jnp.sum(
        hard_token_nll * hard_loss_mask,
        axis=-1,
    ) / jnp.sum(hard_loss_mask, axis=-1)
    np.testing.assert_allclose(
        one_hot_soft_nll,
        discrete_hard_nll,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        uniform_result.hard_node_nll,
        discrete_hard_nll,
        rtol=1e-6,
        atol=1e-6,
    )

    single_results = tuple(
        refine_ebt_address(
            base_params,
            model_config,
            packed_memory,
            lora_config,
            _prefix_batch((desired_node,)),
            uniform_config,
        )
        for desired_node in (2, 3)
    )
    _assert_batch_matches_single_results(uniform_result, single_results)

    assert parameter_checksum(base_params, model_config) == base_checksum
    assert _tree_checksum(packed_memory.edge_bank) == bank_checksum
    np.testing.assert_array_equal(packed_memory.node_path_matrix, path_snapshot)
    np.testing.assert_array_equal(packed_memory.valid_node_mask, valid_node_snapshot)
    np.testing.assert_array_equal(packed_memory.valid_edge_mask, valid_edge_snapshot)

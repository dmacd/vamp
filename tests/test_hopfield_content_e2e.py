from __future__ import annotations

from inspect import Parameter, signature

import jax
import jax.numpy as jnp
import numpy as np

from apm.continual.language_tasks import AddressBook, NodeId
from apm.lm.config import GptNeoConfig
from apm.lm.parameters import init_gpt_neo_params
from apm.memory.content_addressing import HopfieldConfig, hopfield_address
from apm.memory.content_keys import (
    add_address_key,
    derive_node_content_key,
    encode_frozen_base_content,
)


def _model_config() -> GptNeoConfig:
    return GptNeoConfig(
        vocab_size=16,
        max_position_embeddings=8,
        hidden_size=8,
        intermediate_size=16,
        num_layers=1,
        num_heads=2,
        attention_types=("global",),
        local_window_size=4,
    )


def _empty_address_book(config: GptNeoConfig) -> AddressBook:
    return AddressBook(
        node_ids=(None,) * 5,
        keys=np.zeros((5, config.hidden_size), dtype=np.float32),
        valid_node_mask=np.zeros((5,), dtype=np.bool_),
    )


def test_real_content_keys_retrieve_themselves_independently() -> None:
    config = _model_config()
    base_params = init_gpt_neo_params(jax.random.PRNGKey(7), config)
    active_token_ids = (2, 7, 12)
    probe_mask = jnp.asarray(((True, False, False, False),))
    keys = tuple(
        derive_node_content_key(
            base_params,
            config,
            jnp.asarray(((token_id, 0, 0, 0),), dtype=jnp.int32),
            probe_mask,
            expected_probe_count=1,
        )
        for token_id in active_token_ids
    )
    stacked_keys = np.stack(tuple(np.asarray(key) for key in keys))
    node_ids = (NodeId("root"), NodeId("node-a"), NodeId("node-b"))
    address_book = _empty_address_book(config)
    for node_index, (node_id, key) in enumerate(zip(node_ids, keys)):
        address_book = add_address_key(
            address_book,
            node_index,
            node_id,
            key,
        )

    query_input_ids = jnp.asarray(
        (
            (2, 15, 14, 13),
            (7, 13, 15, 14),
            (12, 14, 13, 15),
        ),
        dtype=jnp.int32,
    )
    query_mask = jnp.asarray(
        (
            (True, False, False, False),
            (True, False, False, False),
            (True, False, False, False),
        )
    )
    query_embeddings = encode_frozen_base_content(
        base_params,
        config,
        query_input_ids,
        query_mask,
    )
    result = hopfield_address(
        query_embeddings,
        address_book,
        HopfieldConfig(),
    )

    np.testing.assert_allclose(
        np.linalg.norm(stacked_keys, axis=-1),
        1.0,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        query_embeddings,
        stacked_keys,
        rtol=1e-6,
        atol=1e-6,
    )
    assert all(
        not np.allclose(stacked_keys[left], stacked_keys[right])
        for left, right in ((0, 1), (0, 2), (1, 2))
    )
    np.testing.assert_array_equal(result.selected_indices, (0, 1, 2))
    np.testing.assert_array_equal(result.top_k_indices[:, 0], (0, 1, 2))
    assert result.top_k_indices.shape == (3, 3)
    assert result.node_probabilities.shape == (3, 5)
    assert result.node_probabilities.dtype == jnp.float32
    np.testing.assert_array_equal(result.node_probabilities[:, 3:], 0.0)
    assert np.all(np.isneginf(np.asarray(result.node_scores[:, 3:])))
    assert np.all(np.asarray(result.score_margin) > 0.0)
    assert np.all(np.isfinite(np.asarray(result.entropy)))
    np.testing.assert_allclose(
        np.sum(np.asarray(result.node_probabilities), axis=-1),
        1.0,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_array_equal(address_book.keys[3:], 0.0)
    np.testing.assert_array_equal(
        address_book.valid_node_mask,
        (True, True, True, False, False),
    )

    single_results = tuple(
        hopfield_address(
            query_embeddings[row_index : row_index + 1],
            address_book,
            HopfieldConfig(),
        )
        for row_index in range(3)
    )
    for field_index, batched_field in enumerate(result):
        np.testing.assert_allclose(
            batched_field,
            np.concatenate(
                tuple(
                    np.asarray(single_result[field_index])
                    for single_result in single_results
                ),
                axis=0,
            ),
            rtol=1e-6,
            atol=1e-6,
        )


def test_content_query_key_and_hopfield_apis_exclude_identity_and_adapters() -> None:
    parameter_names = {
        encode_frozen_base_content: tuple(
            signature(encode_frozen_base_content).parameters
        ),
        derive_node_content_key: tuple(
            signature(derive_node_content_key).parameters
        ),
        hopfield_address: tuple(signature(hopfield_address).parameters),
    }

    assert parameter_names[encode_frozen_base_content] == (
        "base_params",
        "base_config",
        "prefix_input_ids",
        "prefix_attention_mask",
    )
    assert parameter_names[derive_node_content_key] == (
        "base_params",
        "base_config",
        "probe_input_ids",
        "probe_attention_mask",
        "expected_probe_count",
    )
    assert parameter_names[hopfield_address] == (
        "query_embeddings",
        "address_book",
        "config",
    )
    forbidden_fragments = (
        "adapter",
        "edge",
        "lora",
        "oracle",
        "task",
    )
    assert not any(
        fragment in parameter_name.lower()
        for names in parameter_names.values()
        for parameter_name in names
        for fragment in forbidden_fragments
    )
    assert all(
        parameter.kind
        not in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD)
        for function in parameter_names
        for parameter in signature(function).parameters.values()
    )

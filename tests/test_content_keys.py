from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from apm.continual.language_tasks import AddressBook, NodeId
from apm.lm.config import GptNeoConfig
from apm.lm.gpt_neo import ForwardResult, apply_gpt_neo
from apm.lm.parameters import GptNeoParams, init_gpt_neo_params
import apm.memory.content_keys as content_keys_module
from apm.memory.content_keys import (
    DEFAULT_CONTENT_KEY_PROBE_COUNT,
    add_address_key,
    derive_node_content_key,
    encode_frozen_base_content,
)


def _config() -> GptNeoConfig:
    return GptNeoConfig(
        vocab_size=17,
        max_position_embeddings=8,
        hidden_size=8,
        intermediate_size=16,
        num_layers=1,
        num_heads=2,
        attention_types=("global",),
        local_window_size=3,
    )


def _params() -> GptNeoParams:
    return init_gpt_neo_params(jax.random.PRNGKey(42), _config())


def _probe_batch(count: int = 4) -> tuple[jax.Array, jax.Array]:
    base = jnp.asarray(
        (
            (1, 2, 3, 4, 5),
            (2, 4, 6, 8, 0),
            (3, 6, 9, 0, 0),
            (4, 8, 12, 16, 1),
        ),
        dtype=jnp.int32,
    )
    base_mask = jnp.asarray(
        (
            (True, True, True, True, True),
            (True, True, True, True, False),
            (True, True, True, False, False),
            (True, True, True, True, True),
        )
    )
    repetitions = (count + len(base) - 1) // len(base)
    return (
        jnp.tile(base, (repetitions, 1))[:count],
        jnp.tile(base_mask, (repetitions, 1))[:count],
    )


def _empty_address_book(max_nodes: int = 3) -> AddressBook:
    return AddressBook(
        node_ids=(None,) * max_nodes,
        keys=np.zeros((max_nodes, _config().hidden_size), dtype=np.float32),
        valid_node_mask=np.zeros((max_nodes,), dtype=np.bool_),
    )


def test_content_encoder_signature_structurally_excludes_adapters_and_identity() -> None:
    encoder_parameters = tuple(inspect.signature(encode_frozen_base_content).parameters)
    key_parameters = tuple(inspect.signature(derive_node_content_key).parameters)

    assert encoder_parameters == (
        "base_params",
        "base_config",
        "prefix_input_ids",
        "prefix_attention_mask",
        "evaluation_microbatch_size",
    )
    assert key_parameters == (
        "base_params",
        "base_config",
        "probe_input_ids",
        "probe_attention_mask",
        "expected_probe_count",
        "evaluation_microbatch_size",
    )
    forbidden_fragments = ("lora", "adapter", "memory", "task", "node")
    assert not any(
        fragment in parameter
        for parameter in encoder_parameters[:-1] + key_parameters[:-2]
        for fragment in forbidden_fragments
    )


def test_content_encoder_masks_nonzero_padded_hidden_states(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_hidden = jnp.asarray(
        (
            (
                (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                (1_000.0,) * 8,
            ),
        ),
        dtype=jnp.float32,
    )

    def fake_base_apply(
        params: GptNeoParams,
        config: GptNeoConfig,
        input_ids: jax.Array,
        attention_mask: jax.Array,
    ) -> ForwardResult:
        assert params is base_params
        assert config == _config()
        assert input_ids.shape == attention_mask.shape == (1, 3)
        return ForwardResult(
            logits=jnp.zeros((1, 3, config.vocab_size), dtype=jnp.float32),
            final_hidden=final_hidden,
            captured_hidden=(),
        )

    base_params = _params()
    monkeypatch.setattr(content_keys_module, "apply_gpt_neo", fake_base_apply)
    query = encode_frozen_base_content(
        base_params,
        _config(),
        jnp.asarray(((1, 2, 16),), dtype=jnp.int32),
        jnp.asarray(((True, True, False),)),
    )
    expected = np.asarray((1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    expected /= np.linalg.norm(expected)

    np.testing.assert_allclose(np.asarray(query[0]), expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(np.asarray(query), axis=-1), 1.0)


def test_real_base_content_is_padding_neutral_and_unit_normalized() -> None:
    params = _params()
    short_ids = jnp.asarray(((1, 2, 3), (4, 5, 6)), dtype=jnp.int32)
    short_mask = jnp.ones_like(short_ids, dtype=jnp.bool_)
    padded_ids = jnp.pad(short_ids, ((0, 0), (0, 2)), constant_values=16)
    padded_mask = jnp.pad(short_mask, ((0, 0), (0, 2)), constant_values=False)

    short_queries = encode_frozen_base_content(params, _config(), short_ids, short_mask)
    padded_queries = encode_frozen_base_content(
        params,
        _config(),
        padded_ids,
        padded_mask,
    )
    padded_hidden = apply_gpt_neo(
        params,
        _config(),
        padded_ids,
        padded_mask,
    ).final_hidden[:, 3:]

    assert np.any(np.abs(np.asarray(padded_hidden)) > 1e-6)
    np.testing.assert_allclose(short_queries, padded_queries, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        np.linalg.norm(np.asarray(short_queries), axis=-1),
        np.ones((2,)),
        rtol=1e-6,
        atol=1e-6,
    )


def test_node_key_is_deterministic_normalized_centroid_of_normalized_queries() -> None:
    params = _params()
    probe_ids, probe_mask = _probe_batch()
    queries = encode_frozen_base_content(params, _config(), probe_ids, probe_mask)

    first = derive_node_content_key(
        params,
        _config(),
        probe_ids,
        probe_mask,
        expected_probe_count=4,
    )
    repeated = derive_node_content_key(
        params,
        _config(),
        probe_ids,
        probe_mask,
        expected_probe_count=4,
    )
    expected = np.mean(np.asarray(queries), axis=0)
    expected /= np.linalg.norm(expected)

    np.testing.assert_array_equal(first, repeated)
    np.testing.assert_allclose(first, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(np.asarray(first)), 1.0, atol=1e-6)


def test_content_encoding_microbatches_without_changing_queries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    params = _params()
    probe_ids, probe_mask = _probe_batch(5)
    expected = encode_frozen_base_content(
        params,
        _config(),
        probe_ids,
        probe_mask,
    )
    original_apply = content_keys_module.apply_gpt_neo
    observed_batch_sizes: list[int] = []

    def counted_apply(*args, **kwargs):
        observed_batch_sizes.append(int(args[2].shape[0]))
        return original_apply(*args, **kwargs)

    monkeypatch.setattr(content_keys_module, "apply_gpt_neo", counted_apply)
    actual = encode_frozen_base_content(
        params,
        _config(),
        probe_ids,
        probe_mask,
        evaluation_microbatch_size=2,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    assert observed_batch_sizes == [2, 2, 1]


def test_default_key_derivation_requires_exactly_256_probes() -> None:
    params = _params()
    complete_ids, complete_mask = _probe_batch(DEFAULT_CONTENT_KEY_PROBE_COUNT)
    incomplete_ids, incomplete_mask = _probe_batch(DEFAULT_CONTENT_KEY_PROBE_COUNT - 1)

    key = derive_node_content_key(
        params,
        _config(),
        complete_ids,
        complete_mask,
    )

    assert key.shape == (_config().hidden_size,)
    np.testing.assert_allclose(np.linalg.norm(np.asarray(key)), 1.0, atol=1e-6)
    with pytest.raises(ValueError, match="expected exactly 256"):
        derive_node_content_key(
            params,
            _config(),
            incomplete_ids,
            incomplete_mask,
        )


def test_root_key_uses_probe_derivation_and_inserts_without_mutation() -> None:
    address_book = _empty_address_book()
    original_node_ids = address_book.node_ids
    original_keys = address_book.keys.copy()
    original_mask = address_book.valid_node_mask.copy()
    probe_ids, probe_mask = _probe_batch()
    root_key = derive_node_content_key(
        _params(),
        _config(),
        probe_ids,
        probe_mask,
        expected_probe_count=4,
    )

    with_root = add_address_key(address_book, 0, NodeId("root"), root_key)

    assert address_book.node_ids == original_node_ids
    np.testing.assert_array_equal(address_book.keys, original_keys)
    np.testing.assert_array_equal(address_book.valid_node_mask, original_mask)
    assert with_root.node_ids == (NodeId("root"), None, None)
    np.testing.assert_array_equal(with_root.valid_node_mask, (True, False, False))
    np.testing.assert_allclose(with_root.keys[0], root_key, atol=0.0)
    np.testing.assert_array_equal(with_root.keys[1:], np.zeros((2, _config().hidden_size)))
    assert not with_root.keys.flags.writeable
    assert not with_root.valid_node_mask.flags.writeable


def test_address_key_insertion_preserves_capacity_masking_and_rejects_bad_slots() -> None:
    address_book = _empty_address_book(max_nodes=2)
    key = np.zeros((_config().hidden_size,), dtype=np.float32)
    key[0] = 1.0
    first = add_address_key(address_book, 0, NodeId("root"), key)
    second = add_address_key(first, 1, NodeId("child"), key)

    np.testing.assert_array_equal(first.valid_node_mask, (True, False))
    np.testing.assert_array_equal(second.valid_node_mask, (True, True))
    with pytest.raises(ValueError, match="already valid"):
        add_address_key(first, 0, NodeId("replacement"), key)
    with pytest.raises(ValueError, match="already exists"):
        add_address_key(first, 1, NodeId("root"), key)
    with pytest.raises(ValueError, match="capacity"):
        add_address_key(first, 2, NodeId("outside"), key)
    with pytest.raises(ValueError, match="L2 normalized"):
        add_address_key(first, 1, NodeId("child"), np.ones_like(key))

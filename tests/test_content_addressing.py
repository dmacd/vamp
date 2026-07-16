from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, is_dataclass
import inspect

import jax
import numpy as np
import pytest

from apm.continual.language_tasks import AddressBook, NodeId
from apm.memory.content_addressing import (
    HopfieldAddressResult,
    HopfieldConfig,
    hopfield_address,
    hopfield_address_core,
)


def _address_book() -> AddressBook:
    return AddressBook(
        node_ids=(
            NodeId("axis-x"),
            NodeId("diagonal"),
            NodeId("axis-y"),
            NodeId("negative-x"),
            None,
        ),
        keys=np.asarray(
            (
                (1.0, 0.0),
                (0.8, 0.6),
                (0.0, 1.0),
                (-1.0, 0.0),
                (0.0, 0.0),
            ),
            dtype=np.float32,
        ),
        valid_node_mask=np.asarray((True, True, True, True, False)),
    )


def _assert_results_close(
    first: HopfieldAddressResult,
    second: HopfieldAddressResult,
) -> None:
    for first_value, second_value in zip(first, second):
        np.testing.assert_allclose(first_value, second_value, rtol=1e-6, atol=1e-6)


def test_exact_keys_retrieve_themselves_and_mask_padded_nodes() -> None:
    address_book = _address_book()
    queries = address_book.keys[:3]

    result = hopfield_address(
        queries,
        address_book,
        HopfieldConfig(beta=10.0, top_k=3),
    )

    np.testing.assert_array_equal(result.selected_indices, (0, 1, 2))
    np.testing.assert_array_equal(result.top_k_indices[0], (0, 1, 2))
    np.testing.assert_array_equal(result.node_probabilities[:, 4], 0.0)
    assert np.all(np.isneginf(np.asarray(result.node_scores)[:, 4]))
    np.testing.assert_allclose(
        np.asarray(result.node_scores)[:, :4],
        queries @ address_book.keys[:4].T,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.sum(result.node_probabilities, axis=-1),
        1.0,
        rtol=1e-6,
        atol=1e-6,
    )
    assert result.selected_indices.dtype == np.int32
    assert result.top_k_indices.dtype == np.int32
    assert result.node_scores.dtype == np.float32
    assert result.node_probabilities.dtype == np.float32
    assert result.score_margin.dtype == np.float32
    assert result.entropy.dtype == np.float32


def test_beta_controls_probability_sharpness_without_changing_scores() -> None:
    query = np.asarray(((0.8, 0.6),), dtype=np.float32)
    low_beta = hopfield_address(
        query,
        _address_book(),
        HopfieldConfig(beta=0.5, top_k=4),
    )
    high_beta = hopfield_address(
        query,
        _address_book(),
        HopfieldConfig(beta=30.0, top_k=4),
    )

    np.testing.assert_array_equal(low_beta.node_scores, high_beta.node_scores)
    assert int(low_beta.selected_indices[0]) == int(high_beta.selected_indices[0]) == 1
    assert high_beta.node_probabilities[0, 1] > low_beta.node_probabilities[0, 1]
    assert high_beta.entropy[0] < low_beta.entropy[0]


def test_top_k_uses_effective_valid_count_and_descending_score_order() -> None:
    address_book = _address_book()
    query = np.asarray(((1.0, 0.0),), dtype=np.float32)

    requested_three = hopfield_address(
        query,
        address_book,
        HopfieldConfig(top_k=3),
    )
    requested_beyond_capacity = hopfield_address(
        query,
        address_book,
        HopfieldConfig(top_k=20),
    )

    assert requested_three.top_k_indices.shape == (1, 3)
    np.testing.assert_array_equal(requested_three.top_k_indices[0], (0, 1, 2))
    assert requested_beyond_capacity.top_k_indices.shape == (1, 4)
    np.testing.assert_array_equal(
        requested_beyond_capacity.top_k_indices[0],
        (0, 1, 2, 3),
    )
    assert 4 not in np.asarray(requested_beyond_capacity.top_k_indices)


def test_one_valid_node_has_unit_probability_infinite_margin_and_zero_entropy() -> None:
    address_book = AddressBook(
        node_ids=(NodeId("only"), None, None),
        keys=np.asarray(((0.0, 1.0), (0.0, 0.0), (0.0, 0.0))),
        valid_node_mask=np.asarray((True, False, False)),
    )

    result = hopfield_address(
        np.asarray(((0.6, 0.8),), dtype=np.float32),
        address_book,
    )

    np.testing.assert_array_equal(result.selected_indices, (0,))
    np.testing.assert_array_equal(result.top_k_indices, ((0,),))
    np.testing.assert_array_equal(result.node_probabilities, ((1.0, 0.0, 0.0),))
    assert np.isposinf(result.score_margin[0])
    assert result.entropy[0] == pytest.approx(0.0)


def test_batch_rows_are_independent_and_core_matches_jit() -> None:
    address_book = _address_book()
    queries = np.asarray(((1.0, 0.0), (0.0, 1.0)), dtype=np.float32)
    config = HopfieldConfig(beta=7.0, top_k=2)
    batched = hopfield_address(queries, address_book, config)
    separate = tuple(
        hopfield_address(query[None, :], address_book, config)
        for query in queries
    )

    for row_index, row_result in enumerate(separate):
        for batched_value, row_value in zip(batched, row_result):
            np.testing.assert_allclose(
                np.asarray(batched_value)[row_index],
                np.asarray(row_value)[0],
                rtol=1e-6,
                atol=1e-6,
            )

    core_arguments = (
        queries,
        address_book.keys,
        address_book.valid_node_mask,
    )
    eager = hopfield_address_core(
        *core_arguments,
        beta=config.beta,
        top_k_size=2,
    )
    compiled_core = jax.jit(
        hopfield_address_core,
        static_argnames=("top_k_size",),
    )
    compiled = compiled_core(
        *core_arguments,
        beta=config.beta,
        top_k_size=2,
    )
    _assert_results_close(eager, compiled)
    _assert_results_close(eager, batched)


def test_config_result_and_public_signature_exclude_oracle_metadata() -> None:
    config = HopfieldConfig()

    assert config == HopfieldConfig(beta=10.0, top_k=4)
    assert is_dataclass(config)
    assert tuple(field.name for field in fields(config)) == ("beta", "top_k")
    with pytest.raises(FrozenInstanceError):
        config.beta = 1.0
    assert HopfieldAddressResult._fields == (
        "selected_indices",
        "node_probabilities",
        "node_scores",
        "score_margin",
        "entropy",
        "top_k_indices",
    )
    signature_names = tuple(inspect.signature(hopfield_address).parameters)
    assert signature_names == ("query_embeddings", "address_book", "config")
    assert not any(
        forbidden in parameter_name
        for parameter_name in signature_names
        for forbidden in ("task", "oracle", "adapter")
    )

    with pytest.raises(ValueError, match="positive"):
        HopfieldConfig(beta=0.0)
    with pytest.raises(ValueError, match="positive"):
        HopfieldConfig(top_k=0)
    with pytest.raises(ValueError, match="L2 normalized"):
        hopfield_address(
            np.asarray(((0.5, 0.0),), dtype=np.float32),
            _address_book(),
        )

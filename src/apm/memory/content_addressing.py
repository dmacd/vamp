"""Masked Hopfield retrieval over frozen-base content embeddings."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from apm.continual.language_tasks import AddressBook


@dataclass(frozen=True)
class HopfieldConfig:
    """Fixed inverse temperature and retrieval width for content addressing."""

    beta: float = 10.0
    top_k: int = 4

    def __post_init__(self) -> None:
        if isinstance(self.beta, bool) or not isinstance(
            self.beta,
            (int, float, np.integer, np.floating),
        ):
            raise TypeError("beta must be a real number")
        if not math.isfinite(float(self.beta)) or self.beta <= 0.0:
            raise ValueError("beta must be finite and positive")
        if type(self.top_k) is not int or self.top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        object.__setattr__(self, "beta", float(self.beta))


class HopfieldAddressResult(NamedTuple):
    """Batched masked similarities, probabilities, choices, and uncertainty."""

    selected_indices: jax.Array
    node_probabilities: jax.Array
    node_scores: jax.Array
    score_margin: jax.Array
    entropy: jax.Array
    top_k_indices: jax.Array


def hopfield_address(
    query_embeddings: jax.Array | np.ndarray,
    address_book: AddressBook,
    config: HopfieldConfig = HopfieldConfig(),
) -> HopfieldAddressResult:
    """Validate and independently retrieve content-addressed nodes for each query."""
    if not isinstance(address_book, AddressBook):
        raise TypeError("address_book must be an AddressBook")
    if not isinstance(config, HopfieldConfig):
        raise TypeError("config must be a HopfieldConfig")
    queries = _validated_unit_queries(query_embeddings, address_book.key_dim)
    valid_node_mask = np.asarray(address_book.valid_node_mask, dtype=np.bool_)
    valid_node_count = int(np.sum(valid_node_mask))
    if valid_node_count == 0:
        raise ValueError("Hopfield addressing requires at least one valid node key")
    valid_keys = address_book.keys[valid_node_mask]
    valid_key_norms = np.linalg.norm(valid_keys, axis=-1)
    if not np.allclose(valid_key_norms, 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("valid address-book keys must be L2 normalized")
    return hopfield_address_core(
        jnp.asarray(queries, dtype=jnp.float32),
        jnp.asarray(address_book.keys, dtype=jnp.float32),
        jnp.asarray(valid_node_mask, dtype=jnp.bool_),
        beta=config.beta,
        top_k_size=min(config.top_k, valid_node_count),
    )


def hopfield_address_core(
    query_embeddings: jax.Array,
    address_keys: jax.Array,
    valid_node_mask: jax.Array,
    *,
    beta: float | jax.Array,
    top_k_size: int,
) -> HopfieldAddressResult:
    """Compute pure fixed-shape Hopfield retrieval suitable for JAX compilation."""
    queries = jnp.asarray(query_embeddings, dtype=jnp.float32)
    keys = jnp.asarray(address_keys, dtype=jnp.float32)
    valid_mask = jnp.asarray(valid_node_mask, dtype=jnp.bool_)
    similarities = queries @ keys.T
    node_scores = jnp.where(
        valid_mask[None, :],
        similarities,
        jnp.asarray(-jnp.inf, dtype=jnp.float32),
    ).astype(jnp.float32)
    node_probabilities = jax.nn.softmax(
        jnp.asarray(beta, dtype=jnp.float32) * node_scores,
        axis=-1,
    ).astype(jnp.float32)
    selected_indices = jnp.argmax(node_scores, axis=-1).astype(jnp.int32)
    top_k_indices = jax.lax.top_k(node_scores, top_k_size)[1].astype(jnp.int32)
    if node_scores.shape[1] == 1:
        score_margin = jnp.full(
            (node_scores.shape[0],),
            jnp.inf,
            dtype=jnp.float32,
        )
    else:
        descending_scores = jnp.sort(node_scores, axis=-1)[:, ::-1]
        score_margin = (
            descending_scores[:, 0] - descending_scores[:, 1]
        ).astype(jnp.float32)
    entropy_terms = jnp.where(
        node_probabilities > 0.0,
        node_probabilities * jnp.log(node_probabilities),
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    entropy = -jnp.sum(entropy_terms, axis=-1).astype(jnp.float32)
    return HopfieldAddressResult(
        selected_indices=selected_indices,
        node_probabilities=node_probabilities,
        node_scores=node_scores,
        score_margin=score_margin,
        entropy=entropy,
        top_k_indices=top_k_indices,
    )


def _validated_unit_queries(
    query_embeddings: jax.Array | np.ndarray,
    key_dimension: int,
) -> np.ndarray:
    queries = np.asarray(query_embeddings)
    if queries.ndim != 2:
        raise ValueError("query_embeddings must have shape [batch, key_dimension]")
    if queries.shape[0] < 1 or queries.shape[1] != key_dimension:
        raise ValueError(
            "query_embeddings must have a nonempty batch and match the key dimension"
        )
    if queries.dtype.kind not in "fiu" or not np.all(np.isfinite(queries)):
        raise ValueError("query_embeddings must contain only finite numeric values")
    float_queries = np.asarray(queries, dtype=np.float32)
    query_norms = np.linalg.norm(float_queries, axis=-1)
    if not np.allclose(query_norms, 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("query_embeddings must be independently L2 normalized")
    return float_queries

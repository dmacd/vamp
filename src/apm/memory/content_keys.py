"""Frozen-base content embeddings and immutable address-key insertion."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from apm.continual.language_tasks import AddressBook, NodeId
from apm.lm.config import GptNeoConfig
from apm.lm.evaluation import evaluation_microbatch_slices
from apm.lm.gpt_neo import apply_gpt_neo
from apm.lm.parameters import GptNeoParams


DEFAULT_CONTENT_KEY_PROBE_COUNT = 256


def encode_frozen_base_content(
    base_params: GptNeoParams,
    base_config: GptNeoConfig,
    prefix_input_ids: jax.Array,
    prefix_attention_mask: jax.Array,
    *,
    evaluation_microbatch_size: int | None = None,
) -> jax.Array:
    """Return normalized masked-mean final states from the adapter-free base."""
    _validate_prefix_shapes(prefix_input_ids, prefix_attention_mask)
    slices = evaluation_microbatch_slices(
        prefix_input_ids.shape[0],
        evaluation_microbatch_size,
    )
    if len(slices) > 1:
        encoded = tuple(
            np.asarray(
                _encode_frozen_base_content_batch(
                    base_params,
                    base_config,
                    prefix_input_ids[row_slice],
                    prefix_attention_mask[row_slice],
                ),
                dtype=np.float32,
            )
            for row_slice in slices
        )
        return jnp.asarray(np.concatenate(encoded, axis=0), dtype=jnp.float32)
    return _encode_frozen_base_content_batch(
        base_params,
        base_config,
        prefix_input_ids,
        prefix_attention_mask,
    )


def _encode_frozen_base_content_batch(
    base_params: GptNeoParams,
    base_config: GptNeoConfig,
    prefix_input_ids: jax.Array,
    prefix_attention_mask: jax.Array,
) -> jax.Array:
    """Encode one already-bounded batch without retaining its vocabulary logits."""
    final_hidden = jax.lax.stop_gradient(
        apply_gpt_neo(
            base_params,
            base_config,
            prefix_input_ids,
            prefix_attention_mask,
        ).final_hidden
    )
    valid_tokens = jnp.asarray(prefix_attention_mask, dtype=jnp.bool_)
    weights = valid_tokens.astype(final_hidden.dtype)[..., None]
    pooled = jnp.sum(final_hidden * weights, axis=1) / jnp.sum(
        weights,
        axis=1,
    )
    return _l2_normalize(pooled)


def derive_node_content_key(
    base_params: GptNeoParams,
    base_config: GptNeoConfig,
    probe_input_ids: jax.Array,
    probe_attention_mask: jax.Array,
    *,
    expected_probe_count: int = DEFAULT_CONTENT_KEY_PROBE_COUNT,
    evaluation_microbatch_size: int | None = None,
) -> jax.Array:
    """Return the normalized centroid of an exact deterministic probe batch."""
    if type(expected_probe_count) is not int or expected_probe_count <= 0:
        raise ValueError("expected_probe_count must be a positive integer")
    _validate_prefix_shapes(probe_input_ids, probe_attention_mask)
    if probe_input_ids.shape[0] != expected_probe_count:
        raise ValueError(
            f"content-key probe batch has {probe_input_ids.shape[0]} examples; "
            f"expected exactly {expected_probe_count}"
        )
    query_embeddings = encode_frozen_base_content(
        base_params,
        base_config,
        probe_input_ids,
        probe_attention_mask,
        evaluation_microbatch_size=evaluation_microbatch_size,
    )
    return _l2_normalize(jnp.mean(query_embeddings, axis=0, keepdims=True))[0]


def add_address_key(
    address_book: AddressBook,
    node_index: int,
    node_id: NodeId,
    key: jax.Array | np.ndarray,
) -> AddressBook:
    """Return a new address book with one normalized key inserted into an empty slot."""
    if type(node_index) is not int or not 0 <= node_index < address_book.max_nodes:
        raise ValueError("node_index is outside the address-book capacity")
    if address_book.valid_node_mask[node_index]:
        raise ValueError(f"address-book slot is already valid: {node_index}")
    if node_id in address_book.node_ids:
        raise ValueError(f"address-book node ID already exists: {node_id}")
    key_array = np.asarray(key, dtype=np.float32)
    if key_array.shape != (address_book.key_dim,):
        raise ValueError(
            f"address key has shape {key_array.shape}; "
            f"expected {(address_book.key_dim,)}"
        )
    if not np.all(np.isfinite(key_array)):
        raise ValueError("address key must contain only finite values")
    if not np.isclose(np.linalg.norm(key_array), 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("address key must be L2 normalized")
    node_ids = (
        address_book.node_ids[:node_index]
        + (node_id,)
        + address_book.node_ids[node_index + 1 :]
    )
    keys = np.array(address_book.keys, copy=True)
    keys[node_index] = key_array
    valid_node_mask = np.array(address_book.valid_node_mask, copy=True)
    valid_node_mask[node_index] = True
    return AddressBook(
        node_ids=node_ids,
        keys=keys,
        valid_node_mask=valid_node_mask,
    )


def _validate_prefix_shapes(
    prefix_input_ids: jax.Array,
    prefix_attention_mask: jax.Array,
) -> None:
    if prefix_input_ids.ndim != 2:
        raise ValueError("prefix_input_ids must have shape [batch, sequence]")
    if prefix_attention_mask.shape != prefix_input_ids.shape:
        raise ValueError("prefix_attention_mask must match prefix_input_ids")
    if prefix_input_ids.shape[0] == 0 or prefix_input_ids.shape[1] == 0:
        raise ValueError("prefix inputs must have nonempty batch and sequence axes")


def _l2_normalize(vectors: jax.Array) -> jax.Array:
    return vectors / jnp.linalg.norm(vectors, axis=-1, keepdims=True)

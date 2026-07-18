from __future__ import annotations

from dataclasses import replace
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from apm.lm.config import GptNeoConfig
from apm.lm.lora import LoraConfig, init_lora_edge
from apm.lm.training import (
    LmTrainConfig,
    init_candidate_lora_train_state,
)
from apm.lm.training_state_artifact import (
    LM_TRAIN_STATE_MANIFEST,
    LM_TRAIN_STATE_TENSORS,
    lm_train_state_checksum,
    load_lm_train_state_artifact,
    write_lm_train_state_artifact,
)


def _state():
    model_config = GptNeoConfig(
        vocab_size=16,
        max_position_embeddings=8,
        hidden_size=8,
        intermediate_size=16,
        num_layers=1,
        num_heads=2,
        attention_types=("global",),
        local_window_size=4,
    )
    train_config = LmTrainConfig(
        learning_rate=1e-3,
        steps=8,
        batch_size=1,
    )
    return init_candidate_lora_train_state(
        init_lora_edge(
            jax.random.PRNGKey(1),
            model_config,
            LoraConfig(rank=1, alpha=1.0),
        ),
        jax.random.PRNGKey(2),
        train_config,
    )


def _mutated_state(state):
    return replace(
        state,
        trainable=jax.tree_util.tree_map(
            lambda leaf: leaf + jnp.asarray(0.125, dtype=leaf.dtype),
            state.trainable,
        ),
        opt_state=jax.tree_util.tree_map(
            lambda leaf: leaf + jnp.asarray(3, dtype=leaf.dtype),
            state.opt_state,
        ),
        rng_key=jax.random.PRNGKey(99),
        step=jnp.asarray(7, dtype=jnp.int32),
    )


def _assert_trees_equal(first, second) -> None:
    first_leaves, first_structure = jax.tree_util.tree_flatten(first)
    second_leaves, second_structure = jax.tree_util.tree_flatten(second)
    assert first_structure == second_structure
    assert len(first_leaves) == len(second_leaves)
    for first_leaf, second_leaf in zip(first_leaves, second_leaves):
        np.testing.assert_array_equal(np.asarray(first_leaf), np.asarray(second_leaf))


def test_training_state_artifact_round_trips_every_state_leaf(tmp_path) -> None:
    initial = _state()
    advanced = _mutated_state(initial)
    identity = "1" * 64

    manifest = write_lm_train_state_artifact(
        tmp_path / "checkpoint",
        identity,
        (initial, advanced),
    )
    restored = load_lm_train_state_artifact(
        tmp_path / "checkpoint",
        identity,
        (initial, initial),
    )

    assert manifest.state_count == 2
    assert manifest.identity_sha256 == identity
    assert int(restored[0].step) == 0
    assert int(restored[1].step) == 7
    assert lm_train_state_checksum(restored[0]) == lm_train_state_checksum(initial)
    assert lm_train_state_checksum(restored[1]) == lm_train_state_checksum(advanced)
    _assert_trees_equal(restored[0], initial)
    _assert_trees_equal(restored[1], advanced)


def test_training_state_artifact_is_immutable_and_rejects_tensor_tampering(
    tmp_path,
) -> None:
    state = _state()
    target = tmp_path / "checkpoint"
    write_lm_train_state_artifact(target, "2" * 64, (state,))

    with pytest.raises(FileExistsError, match="already exists"):
        write_lm_train_state_artifact(target, "2" * 64, (state,))

    tensor_path = target / LM_TRAIN_STATE_TENSORS
    payload = bytearray(tensor_path.read_bytes())
    payload[-1] ^= 1
    tensor_path.write_bytes(payload)
    with pytest.raises(ValueError, match="tensor file checksum"):
        load_lm_train_state_artifact(target, "2" * 64, (state,))


def test_training_state_artifact_rejects_unknown_manifest_fields(tmp_path) -> None:
    state = _state()
    target = tmp_path / "checkpoint"
    write_lm_train_state_artifact(target, "3" * 64, (state,))
    manifest_path = target / LM_TRAIN_STATE_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["unknown"] = True
    manifest_path.write_text(
        json.dumps(manifest, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="fields changed"):
        load_lm_train_state_artifact(target, "3" * 64, (state,))

from __future__ import annotations

import jax
import numpy as np
import pytest

from apm.continual.language_full_finetune_training import (
    run_full_parameter_updates,
)
from apm.data.text.tinyworlds_nouns_v2.contracts import NounsV2ExperimentPreset
from apm.data.text.tinyworlds_nouns_v2.full_finetune import (
    nouns_v2_full_finetune_train_config,
)
from apm.lm.config import GptNeoConfig
from apm.lm.parameters import init_gpt_neo_params
from apm.lm.text_data import TokenBatch
from apm.lm.training import LmTrainConfig, init_base_train_state
from apm.lm.training_state_artifact import (
    lm_train_state_checksum,
    load_lm_train_state_artifact,
    write_lm_train_state_artifact,
)


def _model_config() -> GptNeoConfig:
    return GptNeoConfig(
        vocab_size=8,
        max_position_embeddings=4,
        hidden_size=8,
        intermediate_size=16,
        num_layers=1,
        num_heads=2,
        attention_types=("global",),
        local_window_size=2,
    )


def _batch(target_offset: int = 1, *, rows: int = 2) -> TokenBatch:
    inputs = np.asarray((0, 1, 2, 3), dtype=np.int32)
    targets = (inputs + target_offset) % 8
    return TokenBatch(
        input_ids=np.stack((inputs,) * rows),
        attention_mask=np.ones((rows, 4), dtype=np.bool_),
        target_ids=np.stack((targets,) * rows),
        loss_mask=np.ones((rows, 4), dtype=np.bool_),
    )


def _train_config() -> LmTrainConfig:
    return LmTrainConfig(
        learning_rate=2e-2,
        steps=4,
        batch_size=2,
        weight_decay=0.01,
        gradient_clip_norm=1.0,
    )


def test_nouns_v2_full_finetune_uses_the_matched_task_budget() -> None:
    preset = NounsV2ExperimentPreset()
    train_config = nouns_v2_full_finetune_train_config(preset)
    assert train_config.learning_rate == 5e-5
    assert train_config.steps == 2_000
    assert train_config.batch_size == 32
    assert train_config.weight_decay == 0.01
    assert train_config.gradient_clip_norm == 1.0


def test_full_parameter_updates_resume_exactly_from_authenticated_state(
    tmp_path,
) -> None:
    model_config = _model_config()
    train_config = _train_config()
    initial = init_base_train_state(
        init_gpt_neo_params(jax.random.PRNGKey(1), model_config),
        jax.random.PRNGKey(2),
        train_config,
    )
    batches = (_batch(1), _batch(2))

    uninterrupted, uninterrupted_losses = run_full_parameter_updates(
        initial,
        batches,
        model_config,
        train_config,
    )
    partial, first_losses = run_full_parameter_updates(
        initial,
        batches,
        model_config,
        train_config,
        stop_update=2,
    )
    identity = "a" * 64
    write_lm_train_state_artifact(tmp_path / "state", identity, (partial,))
    restored = load_lm_train_state_artifact(
        tmp_path / "state",
        identity,
        (initial,),
    )[0]
    resumed, second_losses = run_full_parameter_updates(
        restored,
        batches,
        model_config,
        train_config,
    )

    assert first_losses + second_losses == uninterrupted_losses
    assert lm_train_state_checksum(resumed) == lm_train_state_checksum(uninterrupted)
    assert int(resumed.step) == train_config.steps


def test_task_boundary_preserves_parameters_and_rng_but_resets_adamw() -> None:
    model_config = _model_config()
    train_config = _train_config()
    initial = init_base_train_state(
        init_gpt_neo_params(jax.random.PRNGKey(3), model_config),
        jax.random.PRNGKey(4),
        train_config,
    )
    completed, _ = run_full_parameter_updates(
        initial,
        (_batch(),),
        model_config,
        train_config,
        stop_update=2,
    )
    next_task = init_base_train_state(
        completed.trainable,
        completed.rng_key,
        train_config,
    )

    assert int(next_task.step) == 0
    np.testing.assert_array_equal(next_task.rng_key, completed.rng_key)
    completed_leaves = jax.tree_util.tree_leaves(completed.trainable)
    reset_leaves = jax.tree_util.tree_leaves(next_task.trainable)
    for completed_leaf, reset_leaf in zip(completed_leaves, reset_leaves, strict=True):
        np.testing.assert_array_equal(completed_leaf, reset_leaf)
    assert lm_train_state_checksum(next_task) != lm_train_state_checksum(completed)


def test_full_parameter_updates_reject_mixed_or_wrong_batch_shapes() -> None:
    model_config = _model_config()
    train_config = _train_config()
    initial = init_base_train_state(
        init_gpt_neo_params(jax.random.PRNGKey(5), model_config),
        jax.random.PRNGKey(6),
        train_config,
    )
    with pytest.raises(ValueError, match="row count"):
        run_full_parameter_updates(
            initial,
            (_batch(rows=1),),
            model_config,
            train_config,
        )
    short = TokenBatch(
        input_ids=np.ones((2, 3), dtype=np.int32),
        attention_mask=np.ones((2, 3), dtype=np.bool_),
        target_ids=np.ones((2, 3), dtype=np.int32),
        loss_mask=np.ones((2, 3), dtype=np.bool_),
    )
    with pytest.raises(ValueError, match="sequence width"):
        run_full_parameter_updates(
            initial,
            (_batch(), short),
            model_config,
            train_config,
        )

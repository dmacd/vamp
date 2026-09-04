from __future__ import annotations

from pathlib import Path

import torch

from apm.continual.vision.imagenetr.behavior_replay_config import (
    load_behavior_replay_config,
)
from apm.continual.vision.imagenetr.behavior_replay_workflow import (
    COMMON_STATE_NAME,
    seed_row_cache_shards,
)
from apm.continual.vision.imagenetr.integrator_config import load_integrator_config
from apm.continual.vision.imagenetr.integrator_model import create_integrator_state
from apm.continual.vision.imagenetr.proxy_memory import TensorCache


def test_behavior_replay_config_freezes_adapted_latents_and_capacity_matrix() -> None:
    config = load_behavior_replay_config()
    assert config.feature_variant == "behavior"
    assert config.historical_capacities == (2048, 4096, 8192)
    assert config.tasks == 50
    assert config.cache_seed_mode == "hardlink_train_then_test"


def test_capacity_arms_have_identical_initial_parameters() -> None:
    integrator = load_integrator_config(
        "configs/vision/imagenetr/logt_prediction_integrator_full_union_ungated_v3.yaml"
    )
    states = tuple(
        create_integrator_state(
            COMMON_STATE_NAME,
            integrator.maximum_levels,
            "behavior",
            integrator.optimization,
            integrator.seed,
            torch.device("cpu"),
        )
        for _capacity in (2048, 4096, 8192)
    )
    assert states[0].model.input_dim == 8214
    assert all(
        torch.equal(value, states[index].model.state_dict()[name])
        for name, value in states[0].model.state_dict().items()
        for index in (1, 2)
    )


def test_cache_seed_keeps_test_rows_out_until_the_training_seal_boundary(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source = TensorCache(source_root, "unit-behavior-cache-v1")
    source.get_or_compute_rows(
        {"node": "fixed"},
        ("train-image",),
        lambda _ids: {"latent": torch.tensor([[1.0]])},
    )
    source.get_or_compute_rows(
        {"node": "fixed"},
        ("test-image",),
        lambda _ids: {"latent": torch.tensor([[2.0]])},
    )
    splits = {"train-image": "train", "test-image": "test"}
    training_seed = seed_row_cache_shards(
        source_root, target_root, splits, frozenset({"train"})
    )
    assert training_seed["eligible_shards"] == 1
    assert training_seed["skipped_shards"] == 1
    target = TensorCache(target_root, "unit-behavior-cache-v1")
    train, hits, misses = target.get_or_compute_rows(
        {"node": "fixed"},
        ("train-image",),
        lambda _ids: (_ for _ in ()).throw(AssertionError("train cache miss")),
    )
    assert (hits, misses) == (1, 0)
    assert train["latent"].item() == 1.0

    complete_seed = seed_row_cache_shards(
        source_root, target_root, splits, frozenset({"train", "test"})
    )
    assert complete_seed["eligible_shards"] == 2
    assert complete_seed["reused_shards"] == 1
    test, hits, misses = target.get_or_compute_rows(
        {"node": "fixed"},
        ("test-image",),
        lambda _ids: (_ for _ in ()).throw(AssertionError("test cache miss")),
    )
    assert (hits, misses) == (1, 0)
    assert test["latent"].item() == 2.0

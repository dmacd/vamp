from dataclasses import fields
from pathlib import Path

import pytest

from apm.continual.vision.imagenetr.router_config import load_router_config
from apm.continual.vision.imagenetr.router_protocol import (
    RouterPolicy,
    RouterSplit,
)
from apm.continual.vision.imagenetr.router_scores import RouterQuery


PROJECT_ROOT = Path(__file__).parents[4]


def test_recursive_router_config_freezes_promoted_r3_protocol() -> None:
    config = load_router_config(
        PROJECT_ROOT
        / "configs/vision/imagenetr/recursive_router_oracle_recovery_v1.yaml"
    )
    assert config.response_blocks == (0, 4, 7, 11)
    assert config.response_targets == ("attn.qkv", "mlp.fc1")
    assert config.primary_rank == 8
    assert config.router_seeds == (1993, 1994, 1995)
    assert tuple(config.policy_map) == ("I-U100", "I-SVD0", "I-SVD5")


def test_router_query_structurally_excludes_teacher_truth() -> None:
    assert {field.name for field in fields(RouterQuery)} == {
        "image_ids",
        "prelogits",
        "cls_activations",
    }
    assert not {
        "label",
        "labels",
        "class_id",
        "task_id",
        "oracle",
        "node_id",
    } & {field.name for field in fields(RouterQuery)}


def test_router_split_is_disjoint_content_addressed_and_immutable() -> None:
    split = RouterSplit(
        "a" * 64,
        ("1" * 64, "2" * 64),
        ("3" * 64,),
        "unit",
    )
    same = RouterSplit("a" * 64, ("1" * 64, "2" * 64), ("3" * 64,), "unit")
    changed = RouterSplit("a" * 64, ("1" * 64,), ("2" * 64, "3" * 64), "unit")
    assert split.content_hash == same.content_hash
    assert split.content_hash != changed.content_hash
    with pytest.raises(ValueError, match="invalid router split"):
        RouterSplit("a" * 64, ("1" * 64,), ("1" * 64,), "unit")


def test_r3_is_valid_for_every_main_maintenance_policy() -> None:
    base = dict(
        inference_condition="I-U100",
        inference_policy_hash="a" * 64,
        architecture="r3",
        rank=8,
        router_seed=1993,
        descriptor_config_hash="b" * 64,
        response_config_hash="c" * 64,
        training_config_hash="d" * 64,
    )
    for maintenance, fraction in (
        ("flat_seen_data", 0.0),
        ("exact", 0.0),
        ("u100", 0.0),
        ("svd0", 0.0),
        ("svd5", 0.05),
    ):
        policy = RouterPolicy(
            **base, maintenance=maintenance, repair_fraction=fraction
        )
        assert policy.architecture == "r3"

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from apm.continual.artifacts import file_sha256
from apm.continual.vision.imagenetr.data import ImageRecord
from apm.continual.vision.imagenetr.persistent_affine_config import (
    load_persistent_affine_config,
)
from apm.continual.vision.imagenetr.persistent_affine_model import (
    node_transition,
    raw_union_logits,
    true_node_logits,
)
from apm.continual.vision.imagenetr.persistent_affine_reporting import (
    LABEL_H4096,
    LABEL_H8192,
    LABEL_ORACLE_H4096,
    LABEL_ORACLE_H8192,
    LABEL_RANK_JOINT,
    LABEL_STAGE_JOINT,
    _fragmentation_rows,
    _lifecycle_rows,
    _series,
    _stage_rows,
    _summary_rows,
)
from apm.continual.vision.imagenetr.persistent_affine_training import (
    rotating_replay_population,
)
from apm.continual.vision.imagenetr.persistent_integrator_head import PersistentIntegratorHead
from apm.continual.vision.imagenetr.persistent_affine_workflow import (
    PersistentAffineProtocol,
    _material_paths,
)


CONFIG = Path("configs/vision/imagenetr/logt_persistent_affine_v15.yaml")


def _image(class_id: int, copy: int, split: str = "train") -> ImageRecord:
    integer_id = 1 + class_id * 10 + copy
    identity = f"{integer_id:064x}"
    return ImageRecord(
        identity,
        f"class-{class_id}/image-{copy}.jpg",
        f"{split}/class-{class_id}/image-{copy}.jpg",
        f"{integer_id + 10_000:064x}",
        f"class-{class_id}",
        class_id,
        class_id,
        class_id // 4,
        split,
        f"{integer_id + 20_000:064x}",
        1,
    )


def test_config_freezes_only_the_two_persistent_affine_arms() -> None:
    config = load_persistent_affine_config(CONFIG)
    assert config.historical_capacities == (4_096, 8_192)
    assert config.epoch_map == {4_096: 4, 8_192: 5}
    assert config.integrator_kind == "single_affine"
    assert config.feature_source == "node_preclassifier"
    assert config.train_frontier_loras
    assert not config.train_node_classifiers
    assert config.joint_rank_policy == "source_rank_times_live_nodes"
    assert (
        config.joint_epochs,
        config.joint_batch_size,
        config.joint_optimizer,
        config.joint_momentum,
        config.joint_weight_decay,
        config.joint_lora_learning_rate,
        config.joint_head_learning_rate,
    ) == (5, 64, "sgd", 0.9, 0.0005, 0.0005, 0.01)
    assert file_sha256(config.source_config) == config.source_config_sha256
    with pytest.raises(ValueError, match="persistent affine protocol"):
        replace(config, integrator_kind="macro_token")


def test_rotating_replay_is_exact_resumable_and_stage_keyed() -> None:
    rows = tuple(_image(class_id, copy) for class_id in range(20) for copy in range(3))
    first = rotating_replay_population(rows, stage=5, historical_capacity=32, seed=1993)
    repeated = rotating_replay_population(rows, stage=5, historical_capacity=32, seed=1993)
    changed_seed = rotating_replay_population(rows, stage=5, historical_capacity=32, seed=1994)
    assert first == repeated
    assert first.current_examples == 12
    assert first.historical_examples == 32
    assert all(row.task_index < 5 and row.split == "train" for row in first.rows)
    assert {row.image_id for row in rows if row.task_index == 4} <= {
        row.image_id for row in first.rows
    }
    assert first.historical_image_ids_hash != changed_seed.historical_image_ids_hash
    assert "stage=005" in first.namespace


def test_node_transition_resets_only_replaced_hierarchy_nodes() -> None:
    arrival = node_transition(("parent-01",), ("leaf-02", "parent-01"))
    assert arrival.continued_node_hashes == ("parent-01",)
    assert arrival.reset_node_hashes == ("leaf-02",)
    assert arrival.retired_node_hashes == ()
    consolidation = node_transition(
        ("leaf-02", "parent-01"), ("parent-03",)
    )
    assert consolidation.continued_node_hashes == ()
    assert consolidation.reset_node_hashes == ("parent-03",)
    assert consolidation.retired_node_hashes == ("leaf-02", "parent-01")


def test_affine_optimizer_moments_follow_node_hashes_not_positions() -> None:
    old = torch.cat(
        (
            torch.full((200, 768), 1.0),
            torch.full((200, 768), 2.0),
        ),
        dim=1,
    )
    parameter = nn.Parameter(torch.empty(200, 3 * 768))
    head = PersistentIntegratorHead(torch.zeros_like(parameter), torch.zeros(200), 0, 1993)
    transplanted = head.project_parameter(
        "input.weight", old, ("node-a", "node-b"), ("node-b", "node-c", "node-a"), (), torch.zeros_like(parameter)
    )
    torch.testing.assert_close(transplanted[:, :768], torch.full((200, 768), 2.0))
    assert int(torch.count_nonzero(transplanted[:, 768:1536])) == 0
    torch.testing.assert_close(transplanted[:, 1536:], torch.full((200, 768), 1.0))
    bias = head.project_parameter(
        "input.bias", torch.arange(200, dtype=torch.float32), (), (),
        (0, 3, 17), torch.zeros(200),
    )
    assert bias[[0, 3, 17]].tolist() == [0.0, 3.0, 17.0]
    assert int(torch.count_nonzero(bias)) == 2


def test_parameter_free_diagnostics_have_explicit_label_boundary() -> None:
    model = SimpleNamespace(class_ids=((0, 1), (2, 3)))
    local_scores = (
        torch.tensor([[4.0, 1.0], [3.0, 2.0]]),
        torch.tensor([[8.0, 7.0], [1.0, 6.0]]),
    )
    union = raw_union_logits(model, local_scores)
    oracle = true_node_logits(model, local_scores, torch.tensor([0, 3]))
    assert union.argmax(dim=1).tolist() == [2, 3]
    assert oracle.argmax(dim=1).tolist() == [0, 3]
    assert bool(torch.isneginf(oracle[0, 2:]).all())
    assert bool(torch.isneginf(oracle[1, :2]).all())


def test_protocol_material_surface_excludes_reporting_and_macro_modules() -> None:
    identities = tuple(f"{index:064x}" for index in range(1, 16))
    protocol = PersistentAffineProtocol(*identities)
    assert len(protocol.content_hash) == 64
    config = load_persistent_affine_config(CONFIG)
    names = {path.name for path in _material_paths(Path.cwd(), CONFIG.resolve())}
    assert {
        "persistent_affine_config.py",
        "persistent_affine_model.py",
        "persistent_affine_training.py",
        "persistent_affine_workflow.py",
        "imagenetr50_logt_persistent_affine_protocol.md",
        "run_persistent_affine_local.sh",
        "frontier_adaptation_training.py",
        "primary.yaml",
    } <= names
    assert "persistent_affine_reporting.py" not in names
    assert not any("macro_token" in name for name in names)
    assert all(path.is_file() for path in _material_paths(Path.cwd(), CONFIG.resolve()))


def test_report_uses_one_set_of_exact_condition_names() -> None:
    arm = tuple(
        {
            "evaluation": {
                "accuracy": 70.0 + stage / 10,
                "nll": 1.0,
                "true_node_oracle_accuracy": 80.0,
            }
        }
        for stage in range(1, 51)
    )
    joint = tuple({"accuracy": 80.0, "stage": stage} for stage in range(1, 51))
    rank = tuple(
        {"accuracy": 81.0, "nll": 0.8, "rank": 16 * stage.bit_count(), "stage": stage}
        for stage in range(1, 51)
    )
    result = {
        "arms": {"4096": arm, "8192": arm},
        "stage_matched_joint": joint,
        "rank_matched_joint": rank,
    }
    assert tuple(_series(result)) == (
        LABEL_H4096,
        LABEL_H8192,
        LABEL_STAGE_JOINT,
        LABEL_RANK_JOINT,
        LABEL_ORACLE_H4096,
        LABEL_ORACLE_H8192,
    )
    assert len(_summary_rows(result)) == 6
    assert len(_stage_rows(result)) == 50
    assert "macro" not in " ".join(_series(result)).lower()


def test_report_lifecycle_distinguishes_exact_carry_from_source_reset() -> None:
    stages = (
        {
            "carry": {
                "continued_node_hashes": [],
                "reset_node_hashes": ["node-a"],
                "retired_node_hashes": [],
            },
            "node_hashes": ["node-a"],
            "slots": [1],
        },
        {
            "carry": {
                "continued_node_hashes": ["node-a"],
                "reset_node_hashes": ["node-b"],
                "retired_node_hashes": [],
            },
            "node_hashes": ["node-b", "node-a"],
            "slots": [0, 1],
        },
    )
    rows = _lifecycle_rows({"arms": {"4096": stages, "8192": stages}})
    assert rows == (
        {
            "stage": 1,
            "slot": 1,
            "represented_task_capacity": 2,
            "node_hash": "node-a",
            "entry": "source_reset",
        },
        {
            "stage": 2,
            "slot": 0,
            "represented_task_capacity": 1,
            "node_hash": "node-b",
            "entry": "source_reset",
        },
        {
            "stage": 2,
            "slot": 1,
            "represented_task_capacity": 2,
            "node_hash": "node-a",
            "entry": "continued",
        },
    )


def test_report_aggregates_fragmentation_by_live_node_count() -> None:
    stages = tuple(
        {
            "evaluation": {
                "accuracy": 70.0 + live_nodes,
                "true_node_oracle_accuracy": 72.0 + 2 * live_nodes,
            },
            "live_nodes": live_nodes,
        }
        for live_nodes in range(1, 6)
    )
    rows = _fragmentation_rows({"arms": {"4096": stages, "8192": stages}})
    assert len(rows) == 10
    assert rows[0] == {
        "historical_capacity": 4096,
        "integrator_kind": "affine",
        "live_nodes": 1,
        "stages": 1,
        "mean_accuracy": 71.0,
        "mean_true_node_oracle_accuracy": 74.0,
        "mean_true_node_oracle_gap": 3.0,
    }
    assert rows[-1]["mean_true_node_oracle_gap"] == 7.0

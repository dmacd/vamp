from dataclasses import fields
from pathlib import Path
import yaml

import pytest
import torch

from apm.continual.vision.imagenetr.bank import simulate_topology
from apm.continual.vision.imagenetr.config import load_config
from apm.continual.vision.imagenetr.data import (
    ImageRecord,
    deterministic_bottom_k,
    largest_remainder_counts,
    pycil_class_order,
    task_classes,
)
from apm.continual.vision.imagenetr.routing import GroundTruth, TaskFreeQuery


PROJECT_ROOT = Path(__file__).parents[4]


def test_seed_1993_matches_pycil_legacy_permutation() -> None:
    order = pycil_class_order()
    assert order[:20] == (
        168,
        136,
        51,
        9,
        183,
        101,
        171,
        99,
        42,
        159,
        191,
        70,
        16,
        188,
        27,
        10,
        175,
        26,
        68,
        187,
    )
    tasks = task_classes(order)
    assert len(tasks) == 50
    assert all(len(task) == 4 for task in tasks)
    assert sorted(class_id for task in tasks for class_id in task) == list(range(200))


def test_largest_remainder_is_exact_and_stable() -> None:
    counts = {"a": 7, "b": 8, "c": 9}
    allocation = largest_remainder_counts(counts, 5)
    assert allocation == {"a": 1, "b": 2, "c": 2}
    assert sum(allocation.values()) == 5


def test_fifty_arrivals_have_exact_required_topology() -> None:
    events, snapshots = simulate_topology(50)
    assert len(events) == 42
    final = snapshots[-1]
    by_level = {
        level: [node.one_based_interval for node in final.live_nodes if node.level == level]
        for level in range(5)
    }
    assert by_level == {
        0: ["49", "50"],
        1: ["45-46", "47-48"],
        2: ["41-44"],
        3: ["33-40"],
        4: ["1-16", "17-32"],
    }
    assert len(final.live_nodes) == 8


def test_task_free_query_has_no_label_or_task_surface() -> None:
    assert {field.name for field in fields(TaskFreeQuery)} == {"image_ids"}
    assert {field.name for field in fields(GroundTruth)} == {"image_ids", "labels"}
    query = TaskFreeQuery(("a", "b"))
    truth = GroundTruth(query.image_ids, torch.tensor([0, 1]))
    assert truth.image_ids == query.image_ids


def test_hash_reservoir_is_order_independent_and_training_only() -> None:
    rows = tuple(
        ImageRecord(
            image_id=f"{index:064x}",
            source_relative_path=f"n/a{index}.jpg",
            prepared_relative_path=f"train/n/a{index}.jpg",
            image_sha256=f"{100 + index:064x}",
            original_class_name="n",
            original_class_index=0,
            remapped_class_index=0,
            task_index=0,
            split="train",
            priority=f"{10 - index:064x}",
            size_bytes=1,
        )
        for index in range(10)
    )
    assert deterministic_bottom_k(rows, 4, "proxy") == deterministic_bottom_k(
        tuple(reversed(rows)), 4, "proxy"
    )
    test_row = ImageRecord(
        image_id="f" * 64,
        source_relative_path="n/test.jpg",
        prepared_relative_path="test/n/test.jpg",
        image_sha256="e" * 64,
        original_class_name="n",
        original_class_index=0,
        remapped_class_index=0,
        task_index=0,
        split="test",
        priority="d" * 64,
        size_bytes=1,
    )
    with pytest.raises(ValueError, match="training"):
        deterministic_bottom_k(rows + (test_row,), 4, "proxy")


def test_primary_yaml_resolves_to_one_strict_protocol() -> None:
    config = load_config(PROJECT_ROOT / "configs/vision/imagenetr/primary.yaml")
    assert config.tasks == 50
    assert config.lora_targets == ("attention.qkv", "mlp.fc1")
    assert config.artifact_root == PROJECT_ROOT / "artifacts/imagenetr50"


def test_primary_yaml_rejects_unknown_nested_scientific_choice(tmp_path: Path) -> None:
    source = PROJECT_ROOT / "configs/vision/imagenetr/primary.yaml"
    record = yaml.safe_load(source.read_text(encoding="utf-8"))
    record["lora"]["untracked_choice"] = 7
    target = tmp_path / "configs" / "vision" / "imagenetr" / "primary.yaml"
    target.parent.mkdir(parents=True)
    target.write_text(yaml.safe_dump(record), encoding="utf-8")
    with pytest.raises(ValueError, match="lora keys"):
        load_config(target)

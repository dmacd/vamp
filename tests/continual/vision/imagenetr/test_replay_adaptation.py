from __future__ import annotations

from pathlib import Path

import torch

from apm.continual.logt_behavioral_router import sample_example_balanced
from apm.continual.vision.imagenetr.data import ImageRecord
from apm.continual.vision.imagenetr.integrator_bank import class_stratified_reservoir
from apm.continual.vision.imagenetr.integrator_model import IntegratorFitResult
from apm.continual.vision.imagenetr.replay_adaptation_config import (
    load_replay_adaptation_config,
)
from apm.continual.vision.imagenetr.replay_adaptation_training import (
    task_uniform_weights,
)
from apm.continual.vision.imagenetr.replay_adaptation_workflow import (
    _full_history_candidate_rows,
    online_conditions,
)
from apm.experiments.vamp_logt_router_data import ExampleBatch, named_seed


def _image_row(class_id: int, index: int) -> ImageRecord:
    return ImageRecord(
        image_id=f"{class_id * 100 + index:064x}",
        source_relative_path=f"class-{class_id}/{index}.jpg",
        prepared_relative_path=f"train/class-{class_id}/{index}.jpg",
        image_sha256=f"{class_id * 1000 + index + 1:064x}",
        original_class_name=f"class-{class_id}",
        original_class_index=class_id,
        remapped_class_index=class_id,
        task_index=class_id // 4,
        split="train",
        priority=f"{10_000 - class_id * 100 - index:064x}",
        size_bytes=1,
    )


def test_replay_adaptation_config_expands_the_eight_paired_conditions() -> None:
    config = load_replay_adaptation_config()
    conditions = online_conditions(config)
    assert len(conditions) == 8
    assert len({condition.condition_id for condition in conditions}) == 8
    assert config.diagnostic_stages == (31, 50)
    assert config.historical_capacity == 8192


def test_task_uniform_weights_give_each_seen_task_equal_total_mass() -> None:
    labels = torch.tensor((0, 0, 1, 4, 5, 6, 7, 8), dtype=torch.int64)
    weights = task_uniform_weights(labels)
    totals = torch.stack(
        tuple(weights[labels // 4 == task].sum() for task in range(3))
    )
    assert torch.allclose(totals, totals[:1].expand_as(totals))
    assert torch.isclose(weights.mean(), torch.tensor(1.0))


def test_stage_keyed_hash_namespaces_rotate_a_bounded_stratified_draw() -> None:
    rows = tuple(
        _image_row(class_id, index)
        for class_id in range(8)
        for index in range(20)
    )
    first = class_stratified_reservoir(rows, 64, "stage-keyed:31")
    replay = class_stratified_reservoir(rows, 64, "stage-keyed:31")
    second = class_stratified_reservoir(rows, 64, "stage-keyed:32")
    assert first.image_ids == replay.image_ids
    assert set(first.image_ids) != set(second.image_ids)
    assert len(first.image_ids) == len(second.image_ids) == 64


def test_permuted_mnist_uniform_replay_used_a_fresh_macro_step_seed() -> None:
    rows = 80
    archive = ExampleBatch(
        torch.zeros(rows, 1, 28, 28),
        torch.arange(rows, dtype=torch.int64) % 10,
        torch.arange(rows, dtype=torch.int64) % 4,
        torch.arange(rows, dtype=torch.int64),
        torch.ones(rows, dtype=torch.int64),
    )
    first_seed = named_seed(1993, "integrator-uniform", 2)
    second_seed = named_seed(1993, "integrator-uniform", 3)
    first = sample_example_balanced(archive, 24, first_seed, 2)
    replay = sample_example_balanced(archive, 24, first_seed, 2)
    second = sample_example_balanced(archive, 24, second_seed, 3)
    assert first.archive_indices == replay.archive_indices
    assert set(first.archive_indices) != set(second.archive_indices)
    assert first.duplicate_draws == second.duplicate_draws == 0


def test_full_history_selection_identity_excludes_transient_reuse_state(
    tmp_path: Path,
) -> None:
    fit = IntegratorFitResult(
        1, 1, 1, 1.0, 50.0, 1.1, 49.0, 4, 8, 4, 4, 0, 0.1, False
    )
    artifact = tmp_path / "integrators" / "restart_0"
    fresh = ((object(), fit, artifact, False),)
    reused = ((object(), fit, artifact, True),)
    assert _full_history_candidate_rows(tmp_path, fresh) == (
        _full_history_candidate_rows(tmp_path, reused)
    )

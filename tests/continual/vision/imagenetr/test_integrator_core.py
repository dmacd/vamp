from __future__ import annotations

from dataclasses import fields

import pytest
import torch

from apm.continual.vision.imagenetr.data import ImageRecord
from apm.continual.vision.imagenetr.integrator_bank import (
    class_stratified_reservoir,
    merge_stratified_reservoirs,
    require_binary_work_bound,
    resize_stratified_reservoir,
    simulate_binary_topology,
)
from apm.continual.vision.imagenetr.integrator_config import load_integrator_config
from apm.continual.vision.imagenetr.integrator_model import (
    VARIANT_SLOT_DIMS,
    ImageNetResidualIntegrator,
    IntegratorObservations,
    IntegratorSupervision,
    create_integrator_state,
    fit_integrator_epochs,
    select_smallest_near_best,
)
from apm.continual.vision.imagenetr.integrator_observations import FrontierTensors


def _row(class_id: int, index: int) -> ImageRecord:
    identity = f"{class_id * 100 + index:064x}"
    return ImageRecord(
        image_id=identity,
        source_relative_path=f"{class_id}/{index}.jpg",
        prepared_relative_path=f"train/{class_id}/{index}.jpg",
        image_sha256=f"{class_id * 1000 + index + 1:064x}",
        original_class_name=f"class-{class_id}",
        original_class_index=class_id,
        remapped_class_index=class_id,
        task_index=class_id // 4,
        split="train",
        priority=f"{10_000 - class_id * 100 - index:064x}",
        size_bytes=1,
    )


def _frontier() -> FrontierTensors:
    rows, slots = 5, 6
    normalized = torch.zeros(rows, slots, 768, dtype=torch.bfloat16)
    raw = torch.zeros(rows, slots, 200)
    local = torch.zeros_like(raw)
    base = torch.zeros_like(raw)
    ownership = torch.zeros(slots, 200, dtype=torch.bool)
    active = torch.zeros(slots, dtype=torch.bool)
    for slot, class_ids in ((0, torch.arange(4)), (2, torch.arange(4, 8))):
        normalized[:, slot] = torch.randn(rows, 768).to(torch.bfloat16)
        values = torch.randn(rows, 4)
        raw[:, slot, class_ids] = values
        local[:, slot, class_ids] = torch.log_softmax(values, dim=1)
        base[:, slot, class_ids] = torch.randn(rows, 4)
        ownership[slot, class_ids] = True
        active[slot] = True
    return FrontierTensors(
        tuple(f"{index:064x}" for index in range(rows)),
        torch.tensor((0, 1, 4, 5, 7)),
        normalized,
        raw,
        raw.clone(),
        local,
        base,
        ownership,
        active,
        ownership.any(dim=0),
        0,
        3,
        5,
        15,
    )


def test_resolved_config_and_capacity_one_topology_match_the_protocol() -> None:
    config = load_integrator_config(
        "configs/vision/imagenetr/logt_prediction_integrator_v1.yaml"
    )
    events, snapshots = simulate_binary_topology(config.tasks)
    require_binary_work_bound(snapshots, events)
    assert len(events) == 47
    assert [(node.level, node.task_ids) for node in snapshots[-1].live_nodes] == [
        (1, (48, 49)),
        (4, tuple(range(32, 48))),
        (5, tuple(range(32))),
    ]
    assert all(len(snapshot.live_nodes) == snapshot.stage.bit_count() for snapshot in snapshots)


def test_stratified_child_merge_equals_direct_permanent_priority_selection() -> None:
    rows = tuple(_row(class_id, index) for class_id in range(8) for index in range(4))
    left_rows = tuple(row for row in rows if row.remapped_class_index < 4)
    right_rows = tuple(row for row in rows if row.remapped_class_index >= 4)
    namespace = "unit-test-bottom-k"
    children = tuple(
        class_stratified_reservoir(part, 8, namespace)
        for part in (left_rows, right_rows)
    )
    merged = merge_stratified_reservoirs(
        children, {row.image_id: row for row in rows}, 8, namespace
    )
    direct = class_stratified_reservoir(rows, 8, namespace)
    assert merged.image_ids == direct.image_ids
    assert merged.selected_class_counts == direct.selected_class_counts
    assert merged.source_class_counts == direct.source_class_counts
    assert all(count >= 1 for _class_id, count in merged.selected_class_counts)


def test_larger_shared_leaf_reservoir_projects_exactly_to_policy_capacity() -> None:
    rows = tuple(_row(class_id, index) for class_id in range(8) for index in range(8))
    retained = class_stratified_reservoir(rows, 32, "unit-test-bottom-k")
    projected = resize_stratified_reservoir(
        retained, {row.image_id: row for row in rows}, 16
    )
    direct = class_stratified_reservoir(rows, 16, "unit-test-bottom-k")
    assert projected.image_ids == direct.image_ids
    assert projected.selected_class_counts == direct.selected_class_counts
    assert projected.source_class_counts == direct.source_class_counts


def test_observation_variants_are_label_free_fixed_slots_with_exact_raw_parity() -> None:
    frontier = _frontier()
    assert "labels" not in {field.name for field in fields(IntegratorObservations)}
    for variant, slot_dim in VARIANT_SLOT_DIMS.items():
        observations = frontier.observations(variant)
        assert observations.features.shape == (5, 6 * slot_dim)
        slots = observations.features.reshape(5, 6, slot_dim)
        assert torch.equal(slots[:, 1], torch.zeros_like(slots[:, 1]))
        assert torch.equal(slots[:, 3:], torch.zeros_like(slots[:, 3:]))
        model = ImageNetResidualIntegrator(6, slot_dim, (1024, 512, 256), 0.0)
        logits = model(
            observations.features.float(),
            observations.baseline_logits,
            observations.seen_class_mask,
        )
        assert torch.equal(
            logits[:, observations.seen_class_mask],
            observations.baseline_logits[:, observations.seen_class_mask],
        )
        assert torch.isneginf(logits[:, ~observations.seen_class_mask]).all()
    supervision = IntegratorSupervision(frontier.observations("scores"), frontier.labels)
    assert torch.equal(supervision.labels, frontier.labels)


def test_true_node_oracle_is_confined_to_explicit_diagnostic_method() -> None:
    frontier = _frontier()
    predictions = frontier.true_node_oracle_predictions()
    assert predictions.shape == frontier.labels.shape
    with pytest.raises(ValueError):
        IntegratorSupervision(
            frontier.observations("scores"), torch.full((5,), 199, dtype=torch.int64)
        )


def test_feature_selection_chooses_smallest_family_near_the_best() -> None:
    assert (
        select_smallest_near_best(
            {"scores": 84.9, "behavior": 85.1, "behavior_base": 85.0}, 0.25
        )
        == "scores"
    )
    assert (
        select_smallest_near_best(
            {"scores": 84.0, "behavior": 85.1, "behavior_base": 85.2}, 0.25
        )
        == "behavior"
    )


def test_persistent_training_is_independent_of_ambient_dropout_rng() -> None:
    config = load_integrator_config(
        "configs/vision/imagenetr/logt_prediction_integrator_v1.yaml"
    )
    seen = torch.zeros(200, dtype=torch.bool)
    seen[:4] = True
    baseline = torch.full((8, 200), -torch.inf)
    baseline[:, :4] = torch.randn(8, 4)
    observations = IntegratorObservations(
        torch.randn(8, VARIANT_SLOT_DIMS["scores"]).to(torch.bfloat16),
        baseline,
        seen,
        torch.ones(1, dtype=torch.bool),
        "scores",
        VARIANT_SLOT_DIMS["scores"],
    )
    supervision = IntegratorSupervision(
        observations, torch.tensor((0, 1, 2, 3, 0, 1, 2, 3))
    )
    states = tuple(
        create_integrator_state(
            "deterministic", 1, "scores", config.optimization, 17, torch.device("cpu")
        )
        for _ in range(2)
    )
    torch.manual_seed(123)
    first = fit_integrator_epochs(
        states[0], supervision, 1, config.optimization, 17, 1, torch.device("cpu")
    )
    torch.manual_seed(987654)
    second = fit_integrator_epochs(
        states[1], supervision, 1, config.optimization, 17, 1, torch.device("cpu")
    )
    assert first.training_backward_example_passes == 8
    assert first.training_forward_example_passes == 16
    assert first.train_loss == second.train_loss
    assert all(
        torch.equal(first_value, states[1].model.state_dict()[name])
        for name, first_value in states[0].model.state_dict().items()
    )

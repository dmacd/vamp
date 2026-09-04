from __future__ import annotations

from pathlib import Path

from apm.continual.artifacts import publish_immutable_json, record_sha256
from apm.continual.vision.imagenetr.config import TrainingConfig
from apm.continual.vision.imagenetr.integrator_hierarchy import HierarchyParentRecipe
from apm.continual.vision.imagenetr.parent_recipe_factorial import (
    condition_matrix,
    load_parent_recipe_config,
    seed_recipe,
    select_followup,
)
from apm.continual.vision.imagenetr.parent_recipe_reporting import factorial_effects
from apm.continual.vision.imagenetr.promoted_integrator_config import (
    load_promoted_integrator_config,
)
from apm.continual.vision.imagenetr.promoted_integrator_reporting import (
    comparison_rows,
)


def test_factorial_has_eight_uniquely_and_consistently_named_cells() -> None:
    config = load_parent_recipe_config()
    conditions = condition_matrix(config)

    assert len(conditions) == 8
    assert len({condition.key for condition in conditions}) == 8
    assert conditions[0].key == "fresh__wd5e4__joint"
    assert conditions[-1].key == "inherited_union__wd0__parent"
    assert conditions[0].label == "Fresh head | wd=5e-4 | joint seed/order"


def test_seed_recipe_exactly_reconstructs_joint_and_parent_schedules() -> None:
    joint, parent = condition_matrix(load_parent_recipe_config())[::7]

    assert seed_recipe(joint, 1993, 30) == (1993, 51_993)
    assert seed_recipe(parent, 1993, 30) == (302_023, 302_023)


def test_promoted_hierarchy_recipe_uses_the_selected_joint_seed_schedule() -> None:
    training = TrainingConfig(5, 64, 0.9, 0.0005, 0.0005, 0.01)
    recipe = HierarchyParentRecipe("fresh", "joint", training)

    assert recipe.seeds(1993, 30) == (1993, 51_993)
    assert recipe.content_hash != HierarchyParentRecipe(
        "inherited_union", "joint", training
    ).content_hash


def test_full50_config_pins_the_factorial_selected_recipe() -> None:
    config = load_promoted_integrator_config()

    assert config.selected_condition == "fresh__wd5e4__joint"
    assert config.head_initialization == "fresh"
    assert config.weight_decay == 0.0005
    assert config.seed_schedule == "joint"


def test_promoted_report_aligns_persistent_oracle_and_joint_curves(
    tmp_path: Path,
) -> None:
    joint_rows = [
        {
            "accuracy": 90.0 - stage / 10,
            "evaluation_seconds": 0.5,
            "image_presentations": stage * 10,
            "optimizer_steps": stage,
            "peak_vram_bytes": 1024,
            "reused_source_model": stage == 50,
            "stage": stage,
            "test_examples": stage * 4,
            "train_examples": stage * 16,
            "training_seconds": 1.0,
        }
        for stage in range(1, 51)
    ]
    joint_core: dict[str, object] = {
        "incremental_accuracy": sum(
            float(row["accuracy"]) for row in joint_rows
        )
        / 50,
        "rows": joint_rows,
        "schema_version": "imagenetr50-stage-matched-joint-iid-summary-v1",
    }
    publish_immutable_json(
        tmp_path / "evaluations" / "stage_matched_joint_iid.json",
        {**joint_core, "content_hash": record_sha256(joint_core)},
    )
    locked = {
        "stage_metrics": [
            {
                "accuracy": 80.0 - stage / 10,
                "controls": {"true_node_oracle": 85.0 - stage / 10},
                "live_nodes": stage.bit_count(),
                "stage": stage,
            }
            for stage in range(1, 51)
        ]
    }

    rows = comparison_rows(tmp_path, locked)

    assert len(rows) == 50
    assert rows[15] == {
        "joint_iid_minus_true_node_pp": 5.0,
        "live_nodes": 1,
        "persistent_logt_accuracy": 78.4,
        "stage": 16,
        "stage_matched_joint_iid_accuracy": 88.4,
        "true_node_minus_persistent_pp": 5.0,
        "true_node_oracle_accuracy": 83.4,
    }


def _synthetic_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    conditions = [condition.as_record() for condition in condition_matrix(load_parent_recipe_config())]
    rows = [
        {
            "condition_key": condition["condition_key"],
            "final_cross_entropy": 1.0,
            "final_validation_accuracy": (
                80.0
                + (2.0 if condition["head_initialization"] == "inherited_union" else 0.0)
                + (1.0 if condition["weight_decay"] == 0.0 else 0.0)
                + (0.5 if condition["seed_schedule"] == "parent" else 0.0)
                - stage / 100.0
            ),
            "replication_seed": 1993,
            "stage": stage,
        }
        for stage in (8, 16, 32)
        for condition in conditions
    ]
    return rows, conditions


def test_factor_effects_hold_the_other_two_factors_fixed() -> None:
    rows, conditions = _synthetic_rows()

    effects = factorial_effects(rows, conditions)

    stage_16 = {row["factor"]: row for row in effects if row["stage"] == 16}
    assert stage_16["head_initialization"]["mean_effect_pp"] == 2.0
    assert stage_16["weight_decay"]["mean_effect_pp"] == 1.0
    assert stage_16["seed_schedule"]["mean_effect_pp"] == 0.5


def test_followup_trigger_requires_half_gap_closure_at_both_late_stages() -> None:
    rows, _conditions = _synthetic_rows()
    parent_key = "inherited_union__wd0__parent"
    joint_key = "fresh__wd5e4__joint"
    # Make the reference direction realistic and one candidate close 60% at
    # both selection stages.
    for row in rows:
        stage = int(row["stage"])
        if row["condition_key"] == parent_key:
            row["final_validation_accuracy"] = 70.0
        elif row["condition_key"] == joint_key:
            row["final_validation_accuracy"] = 80.0
        else:
            row["final_validation_accuracy"] = 76.0 if stage in (16, 32) else 75.0

    selection = select_followup(rows, 0.5)

    assert selection["full50_triggered"] is True
    assert selection["selected_condition"] != parent_key

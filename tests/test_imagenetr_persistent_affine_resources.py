"""Scientific accounting includes source training, path multiplicity, and cache reuse."""

from copy import deepcopy

import pytest

from apm.continual.vision.imagenetr.persistent_affine_resources import resource_stage_rows


def _five_stage_evidence() -> tuple[dict[str, object], dict[str, object]]:
    nodes = [
        {"node_hash": f"leaf{stage}", "last_task": stage - 1,
         "metrics": {"image_presentations": 10, "wall_seconds": 1.0}}
        for stage in range(1, 6)
    ] + [
        {"node_hash": name, "last_task": last_task,
         "metrics": {"image_presentations": images, "wall_seconds": images / 10}}
        for name, last_task, images in (("p12", 1, 20), ("p34", 3, 20), ("p14", 3, 40))
    ]
    frontiers = (("leaf1",), ("p12",), ("leaf3", "p12"), ("p14",), ("leaf5", "p14"))
    arms = [
        {
            "stage": stage, "live_nodes": len(frontier), "node_hashes": list(frontier),
            "fit": {"epochs": 4, "image_presentations": 8 if stage == 1 else 12, "wall_seconds": 1.0},
            "population": {"training_examples": 2 if stage == 1 else 3},
            "evaluation": {"examples": stage, "wall_seconds": 0.1},
        }
        for stage, frontier in enumerate(frontiers, 1)
    ]
    joint = [
        {
            "stage": stage, "train_examples": 2 * stage, "test_examples": stage,
            "image_presentations": 10 * stage if stage < 5 else 0,
            "training_seconds": float(stage) if stage < 5 else 0.0,
            "evaluation_seconds": 0.1,
            "reused_source_model": stage == 5,
            "source_joint_node_hash": "original_joint" if stage == 5 else None,
        }
        for stage in range(1, 6)
    ]
    rank = [
        {
            "stage": stage, "train_examples": 2 * stage,
            "image_presentations": 10 * stage,
            "training_seconds": float(stage if stage.bit_count() == 1 else 2 * stage),
            "reused_stage_matched_rank16": stage.bit_count() == 1,
        }
        for stage in range(1, 6)
    ]
    return (
        {"arms": {"4096": arms}, "stage_matched_joint": joint, "rank_matched_joint": rank},
        {"hierarchy_nodes": nodes, "joint_final": {
            "node_hash": "original_joint", "metrics": {"image_presentations": 50, "wall_seconds": 5.0},
        }},
    )


def test_resource_totals_charge_intermediate_parents_and_original_joint_training() -> None:
    result, sources = _five_stage_evidence()
    rows = resource_stage_rows(result, sources, True)
    final = {row["condition_id"]: row for row in rows if row["stage"] == 5}
    persistent = final["persistent_h4096"]
    assert persistent["cumulative_hierarchy_forward_images"] == 130
    assert persistent["cumulative_adaptation_forward_images"] == 80
    assert persistent["cumulative_training_backward_images"] == 210
    assert persistent["cumulative_recompute_forward_images"] == 80
    assert persistent["cumulative_training_forward_images_including_recompute"] == 290
    assert persistent["cumulative_training_wall_seconds"] == 18.0
    assert persistent["cumulative_evaluation_forward_images"] == 23
    assert rows[3]["hierarchy_forward_images"] == 70  # p34 never enters a post-stage frontier.
    joint, rank = final["joint_rank16"], final["joint_rank_matched"]
    assert joint["cumulative_training_forward_images"] == rank["cumulative_training_forward_images"] == 150
    assert joint["cumulative_training_wall_seconds"] == 15.0
    assert rank["cumulative_training_wall_seconds"] == 23.0
    assert joint["cumulative_evaluation_wall_seconds"] == pytest.approx(0.5)
    assert rank["cumulative_evaluation_wall_seconds"] is None


def test_resource_accounting_rejects_missing_live_nodes_or_incomplete_presentations() -> None:
    result, sources = _five_stage_evidence()
    changed = deepcopy(result)
    changed["arms"]["4096"][2]["live_nodes"] = 1
    with pytest.raises(ValueError, match="multiplicity"):
        resource_stage_rows(changed, sources, True)
    changed = deepcopy(result)
    changed["arms"]["4096"][2]["fit"]["image_presentations"] = 8
    with pytest.raises(ValueError, match="completed epochs"):
        resource_stage_rows(changed, sources, True)
    rows = resource_stage_rows(result, sources, False)
    assert rows[4]["cumulative_recompute_forward_images"] == 0
    assert rows[4]["cumulative_training_forward_images_including_recompute"] == 210

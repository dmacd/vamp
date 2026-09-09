"""The new report condition must authenticate and retain comparable work accounting."""

from pathlib import Path

import pytest

from apm.continual.artifacts import canonical_json_bytes, file_sha256, record_sha256
from apm.continual.vision.imagenetr.persistent_mlp_reporting import (
    MLP_CONDITION_ID, load_mlp_extension, mlp_resource_rows,
)


def _sealed(record: dict[str, object]) -> dict[str, object]:
    return {**record, "content_hash": record_sha256(record)}


def test_mlp_report_extension_authenticates_every_stage_against_its_affine_reference(tmp_path: Path) -> None:
    run = tmp_path / "persistent_affine_v15/runs/affine"
    reports = run / "reports"
    reports.mkdir(parents=True)
    mlp_run = tmp_path / "persistent_mlp_v16/runs/mlp"
    (mlp_run / "evaluations").mkdir(parents=True)
    rows = [
        {"stage": stage, "node_hashes": [f"node{stage}"], "slots": [0], "carry": {},
         "population": {"identity": str(stage)}, "fit": {"image_presentations": stage * 4}}
        for stage in range(1, 51)
    ]
    reference = {"content_hash": "a" * 64, "arms": {"4096": rows}}
    core = {
        "schema_version": "imagenetr50-persistent-mlp-result-v1",
        "reference_result_hash": reference["content_hash"], "hidden_dimension": 1024,
        "historical_capacity": 4096, "source_hierarchy_unchanged": True,
        "matched_replay_and_exposure": True, "rows": rows,
    }
    path = mlp_run / "evaluations/result.json"
    result = _sealed(core)
    path.write_bytes(canonical_json_bytes(result))
    pointer = {
        "schema_version": "imagenetr50-persistent-mlp-report-extension-v1",
        "reference_result_hash": reference["content_hash"], "run": str(mlp_run),
        "result_sha256": file_sha256(path), "result_hash": result["content_hash"],
    }
    (reports / "mlp_extension.json").write_bytes(canonical_json_bytes(_sealed(pointer)))
    assert load_mlp_extension(run, reference) == result
    changed_rows = [{**row, "node_hashes": ["different"]} if row["stage"] == 31 else row for row in rows]
    changed = _sealed({**core, "rows": changed_rows})
    path.write_bytes(canonical_json_bytes(changed))
    with pytest.raises(ValueError, match="source file changed"):
        load_mlp_extension(run, reference)
    pointer = {**pointer, "result_sha256": file_sha256(path), "result_hash": changed["content_hash"]}
    (reports / "mlp_extension.json").write_bytes(canonical_json_bytes(_sealed(pointer)))
    with pytest.raises(ValueError, match="matched affine task stream"):
        load_mlp_extension(run, reference)


def test_mlp_resources_charge_one_common_hierarchy_and_actual_node_multiplicity() -> None:
    result = {"mlp": {"rows": [
        {"live_nodes": nodes, "fit": {"image_presentations": 20, "wall_seconds": 2.0},
         "evaluation": {"examples": 3, "wall_seconds": 0.1}}
        for nodes in (1, 1, 2)
    ]}}
    base_rows = tuple(
        {"condition_id": "persistent_h4096", "hierarchy_forward_images": 10, "hierarchy_wall_seconds": 1.0}
        for _ in range(3)
    )
    rows = mlp_resource_rows(result, base_rows)
    assert len(rows) == 3 and rows[-1]["condition_id"] == MLP_CONDITION_ID
    assert rows[-1]["cumulative_hierarchy_forward_images"] == 30
    assert rows[-1]["cumulative_adaptation_forward_images"] == 80
    assert rows[-1]["cumulative_training_forward_images_including_recompute"] == 190
    assert rows[-1]["cumulative_training_backward_images"] == 110
    assert rows[-1]["cumulative_training_wall_seconds"] == 9.0
    assert rows[-1]["cumulative_evaluation_forward_images"] == 12
    assert mlp_resource_rows({}, base_rows) == ()

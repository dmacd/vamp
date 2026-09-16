"""Explicit, non-training authentication of the completed H=512 follow-up."""

from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path

import pytest

from apm.continual.artifacts import file_sha256, record_sha256
from apm.continual.vision.imagenetr.srt_evidence import read_sealed


@pytest.mark.integration
def test_completed_h512_pair_and_report_authenticate() -> None:
    """Verify every new checkpoint, common test identity, paired count and preserved prior summary."""
    import pyarrow.parquet as pq
    from apm.continual.vision.imagenetr.data import load_dataset_manifest, validate_prepared_dataset
    from apm.continual.vision.imagenetr.srt_followup import load_followup_config
    from apm.continual.vision.imagenetr.srt_reporting import LOW_BUDGET_STYLES, followup_report_jobs

    project = Path(__file__).resolve().parents[1]
    config = load_followup_config(project / "configs/vision/imagenetr/srt_h512_rho80_unit8.yaml")
    source = project / config.source_run
    pointer_path = source / "reports/fixed_policy_followup_h512_standard_rho80_unit8.json"
    if not pointer_path.is_file():
        pytest.skip("the H=512 experiment has not completed locally")
    pointer = read_sealed(pointer_path)
    run = source / "followups" / pointer["run_hash"]
    result = read_sealed(run / "result.json")
    assert set(result["conditions"]) == set(LOW_BUDGET_STYLES)
    assert result["content_hash"] == pointer["result_hash"]
    assert result["zero_step_reuse"] and result["source_unchanged"]
    assert all(row["optimizer_steps"] == 0 for row in result["reuse"].values())

    dataset_root = project / "data/imagenetr50/imagenet-r"
    manifest = load_dataset_manifest(dataset_root / "dataset_manifest.json")
    validate_prepared_dataset(dataset_root, manifest)
    train, test = manifest.select("train"), manifest.select("test")
    expected_labels = {row.image_id: (row.remapped_class_index, row.task_index + 1) for row in test}
    assert len(train) == 24000 and len(test) == 6000
    assert not {row.image_id for row in train} & expected_labels.keys()
    roots, evidence = followup_report_jobs(source, config.source_result_hash)
    assert len(roots) == 6 and len(evidence) == 2

    checks = tuple((project / row["path"], row["sha256"]) for row in read_sealed(source / "code_manifest.json")["files"])
    original = read_sealed(source / "result.json")
    for name, job in result["conditions"].items():
        root = run / "final" / name
        definition = read_sealed(root / "job.json")
        assert definition["training_ids_hash"] == record_sha256([row.image_id for row in train])
        assert definition["evaluation_ids_hash"] == record_sha256([row.image_id for row in test])
        assert job["image_presentations"] == 196204
        assert len(job["rows"]) == 50
        # History contains fewer than 512 images during the first two tasks,
        # so both budgets must produce exactly the same learned checkpoints.
        previous = original["conditions"][f"{job['method']}_h1024"]
        assert [(row["model_sha256"], row["predictions_sha256"]) for row in job["rows"][:2]] == [
            (row["model_sha256"], row["predictions_sha256"]) for row in previous["rows"][:2]]
        for row in job["rows"]:
            stage = root / f"stages/{row['stage']:03d}"
            assert read_sealed(stage / "result.json") == row
            checks += ((stage / "model.safetensors", row["model_sha256"]),
                       (stage / "predictions.parquet", row["predictions_sha256"]))
        predictions = pq.read_table(root / "stages/050/predictions.parquet").to_pylist()
        assert {row["image_id"]: (row["label"], row["task"]) for row in predictions} == expected_labels
        assert len(predictions) == 6000
        metrics = job["rows"][-1]["evaluation"]
        assert metrics["accuracy"] == pytest.approx(100 * sum(row["prediction"] == row["label"] for row in predictions) / 6000)
        assert metrics["nll"] == pytest.approx(math.fsum(row["nll"] for row in predictions) / 6000)
    with ThreadPoolExecutor(max_workers=8) as executor:
        assert tuple(executor.map(file_sha256, (path for path, _ in checks))) == tuple(expected for _, expected in checks)
    first, second = result["conditions"].values()
    assert first["optimizer_steps"] == second["optimizer_steps"]

    reports = source / "reports"
    report = read_sealed(reports / "report_manifest.json")
    assert report["event_audit_passed"] and report["paired_exposure_audit_passed"]
    assert set(LOW_BUDGET_STYLES) <= report["condition_names"].keys()
    assert "standard_budget_comparison" in report["figures"]
    assert file_sha256(Path(report["pdf"])) == report["pdf_sha256"]
    stages = pq.read_table(reports / "stage_metrics.parquet").to_pylist()
    assert len(stages) == 500
    samples = pq.read_table(reports / "replay_samples.parquet").to_pylist()
    assert len(samples) == 240000
    frozen = report["frozen_replay_history_input"]
    assert file_sha256(reports / frozen["path"]) == frozen["sha256"]
    previous = json.loads((project / "artifacts/imagenetr50/srt_h512/prior_condition_summary.json").read_text())
    current = {row["condition"]: row for row in json.loads((reports / "condition_summary.json").read_text())}
    assert all(current[row["condition"]] == row for row in previous)

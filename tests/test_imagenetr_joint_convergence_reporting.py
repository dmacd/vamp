"""Synthetic-only checks for task-50 markers and the convergence report appendix."""

from dataclasses import replace

import matplotlib.pyplot as plt
import pytest

from apm.continual.artifacts import file_sha256, publish_immutable_json
from apm.continual.vision.imagenetr.joint_convergence_reporting import (
    draw_joint_endpoint, endpoint_summary, joint_report_parts, load_joint_reference,
)
from apm.continual.vision.imagenetr.joint_convergence_training import select_epochs
from apm.continual.vision.imagenetr.srt_evidence import sealed_record, write_parquet


@pytest.fixture
def synthetic_joint_reference():
    """Build explicitly synthetic layout data, never an authenticated experiment."""
    endpoints = {"accuracy_selected": 25, "five_epoch": 5, "nll_selected": 20, "terminal": 40}
    def epoch_record(epoch, seed, development):
        values = {"accuracy": min(99., 70 + epoch * .9), "nll": max(.1, 1.2 - epoch * .03)}
        validation = {"accuracy": min(81., 66 + epoch * .6), "nll": max(.8, 1.3 - epoch * .025)}
        return {"epoch": epoch, "learning_rate_scale": .2 ** min(3, epoch // 10),
                "lora_learning_rate": .0005 * .2 ** min(3, epoch // 10), "head_learning_rate": .01 * .2 ** min(3, epoch // 10),
                "training": {**values, "presentations": 19200 if development else 24000,
                             "optimizer_steps": 300 if development else 375, "wall_seconds": 55.},
                "fit_probe": values, "validation": validation if development else None,
                "evaluation_wall_seconds": 15., "plateau": {"stale_epochs": epoch % 10, "reductions": min(3, epoch // 10)},
                "model_sha256": "f" * 64, "content_hash": "a" * 64}
    jobs = {name: {"seed": seed, "phase": "development" if name == "development" else "full_data_refit",
                   "epochs": [epoch_record(epoch, seed, name == "development") for epoch in range(1, 41)],
                   "training_presentations": 40 * (19200 if name == "development" else 24000), "optimizer_steps": 40 * 375,
                   "model_forward_images": 1100000, "training_wall_seconds": 2200., "evaluation_wall_seconds": 600.}
            for name, seed in (("development", 1993), ("seed_1993", 1993), ("seed_1994", 1994), ("seed_1995", 1995))}
    evaluations = [{"seed": seed, "epoch": epoch, "endpoint_roles": [role],
                    "metrics": {"accuracy": 80. + seed - 1993 if role == "accuracy_selected" else 77. + seed - 1993,
                                "nll": .8 + .1 * (seed - 1993), "examples": 6000, "wall_seconds": 15.},
                    "model_sha256": "f" * 64, "predictions_sha256": "b" * 64, "selection_hash": "c" * 64}
                   for seed in (1993, 1994, 1995) for role, epoch in endpoints.items()]
    return {"protocol": {"config": {"rule": {"plateau_patience": 8, "terminal_patience": 12}}},
            "result": {"selection": {"endpoints": endpoints, "validation_converged": True}, "jobs": jobs, "evaluations": evaluations}}


def test_endpoint_mean_sd_and_stage50_only_marker(synthetic_joint_reference) -> None:
    rows = endpoint_summary(synthetic_joint_reference)
    selected = next(row for row in rows if row["role"] == "accuracy_selected")
    assert selected["accuracy_mean"] == 81 and selected["accuracy_sample_sd"] == 1
    assert selected["nll_mean"] == pytest.approx(.9) and selected["nll_sample_sd"] == pytest.approx(.1)
    assert selected["training_presentations_per_seed_to_endpoint"] == 600000
    figure, axis = plt.subplots()
    draw_joint_endpoint(axis, {"joint_convergence": synthetic_joint_reference})
    assert list(axis.lines[0].get_xdata()) == [50] and list(axis.lines[0].get_ydata()) == [81]
    plt.close(figure)


def test_incomplete_seed_matrix_and_unsafe_pointer_are_rejected(tmp_path, synthetic_joint_reference) -> None:
    result = synthetic_joint_reference["result"]
    altered = {**synthetic_joint_reference, "result": {**result, "evaluations": result["evaluations"][:-1]}}
    with pytest.raises(ValueError, match="three seed"):
        endpoint_summary(altered)
    source = tmp_path / "a/b/c/d/source"
    assert load_joint_reference(source, "a" * 64) is None
    publish_immutable_json(source / "reports/joint_convergence.json", sealed_record({
        "schema_version": "imagenetr50-joint-convergence-pointer-v1", "run": "../../outside",
    }))
    with pytest.raises(ValueError, match="namespace"):
        load_joint_reference(source, "a" * 64)


def test_authenticated_reference_reconstructs_metrics_and_rejects_changed_predictions(tmp_path) -> None:
    """Exercise the report's complete evidence path using synthetic, sealed files."""
    source = tmp_path / "artifacts/imagenetr50/srt_r16_v1/runs/source"
    predictions = tuple({"image_id": str(index), "task": index % 50 + 1, "label": index % 200,
                         "prediction": index % 200, "nll": .5} for index in range(6000))
    original_path = source / "final/srt_h1024/stages/050/predictions.parquet"
    prediction_hash = write_parquet(original_path, predictions)
    original = sealed_record({"conditions": {"srt_h1024": {"rows": [{"predictions_sha256": prediction_hash}]}}})
    source_protocol = sealed_record({"dataset_hash": "d" * 64, "model_sha256": "m" * 64})
    for name, record in (("result.json", original), ("protocol.json", source_protocol)):
        publish_immutable_json(source / name, record)
    protocol = sealed_record({"schema_version": "imagenetr50-joint-convergence-protocol-v1",
                              "source_result_hash": original["content_hash"],
                              **{name: source_protocol[name] for name in ("dataset_hash", "model_sha256")}})
    root = tmp_path / "artifacts/imagenetr50/joint_convergence_r16/runs" / protocol["content_hash"]
    publish_immutable_json(root / "protocol.json", protocol)
    jobs = {}
    for name in ("development", "seed_1993", "seed_1994", "seed_1995"):
        development = name == "development"
        job_root = root / name if development else root / "refits" / name
        definition = sealed_record({"schema_version": "imagenetr50-joint-convergence-job-v1", "protocol_hash": root.name,
                                    "fitting_examples": 19200 if development else 24000, "validation_examples": 4800 if development else 0})
        publish_immutable_json(job_root / "job.json", definition)
        epochs = []
        for epoch in range(1, 6):
            epoch_root = job_root / "epochs" / f"{epoch:03d}"
            epoch_root.mkdir(parents=True)
            (epoch_root / "model.safetensors").write_bytes(b"synthetic hash fixture, not model weights")
            validation_hash = write_parquet(epoch_root / "validation_predictions.parquet", predictions[:1]) if development else None
            row = sealed_record({"epoch": epoch, "learning_rate_scale": 1., "model_sha256": file_sha256(epoch_root / "model.safetensors"),
                                 "validation": {"accuracy": 70 + epoch, "nll": 1 / epoch} if development else None,
                                 "validation_predictions_sha256": validation_hash})
            publish_immutable_json(epoch_root / "result.json", row)
            epochs.append(row)
        jobs[name] = sealed_record({"schema_version": "imagenetr50-joint-convergence-job-result-v1",
                                    "job_hash": definition["content_hash"], "epochs": epochs})
        publish_immutable_json(job_root / "result.json", jobs[name])
    development = jobs["development"]
    selection = sealed_record({"schema_version": "imagenetr50-joint-convergence-selection-v1", "protocol_hash": root.name,
                               "development_result_hash": development["content_hash"], "endpoints": select_epochs(tuple(development["epochs"])),
                               "learning_rate_schedule": [1.] * 5, "test_used": False})
    publish_immutable_json(root / "selection.json", selection)
    evaluations = []
    for seed in (1993, 1994, 1995):
        output = root / "evaluations" / f"seed_{seed}/epoch_005"
        digest = write_parquet(output / "predictions.parquet", predictions)
        evaluated = sealed_record({"seed": seed, "epoch": 5, "selection_hash": selection["content_hash"],
                                   "model_sha256": jobs[f"seed_{seed}"]["epochs"][-1]["model_sha256"],
                                   "predictions_sha256": digest, "metrics": {"examples": 6000, "accuracy": 100., "nll": .5}})
        publish_immutable_json(output / "result.json", evaluated)
        evaluations.append(evaluated)
    result = sealed_record({"schema_version": "imagenetr50-joint-convergence-result-v1", "protocol_hash": root.name,
                            "source_result_hash": original["content_hash"], "selection": selection, "jobs": jobs,
                            "reuse": {name: {"optimizer_steps": 0, "result_hash": job["content_hash"]} for name, job in jobs.items()},
                            "zero_step_reuse": True, "source_unchanged": True, "evaluations": evaluations})
    publish_immutable_json(root / "result.json", result)
    publish_immutable_json(source / "reports/joint_convergence.json", sealed_record({
        "schema_version": "imagenetr50-joint-convergence-pointer-v1", "run": str(root.relative_to(tmp_path)),
        "source_result_hash": original["content_hash"], "result_hash": result["content_hash"],
    }))
    assert load_joint_reference(source, original["content_hash"])["result"] == result
    (root / "evaluations/seed_1993/epoch_005/predictions.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="predictions changed"):
        load_joint_reference(source, original["content_hash"])


@pytest.mark.integration
def test_joint_convergence_appendix_layout(tmp_path, synthetic_joint_reference) -> None:
    """Render the complete synthetic appendix for explicit visual QA."""
    from pypdf import PdfReader
    from apm.continual.vision.imagenetr.srt_reporting import render_report
    replay = ({"condition": "uniform_h4096_standard_rho80_unit8", "final_accuracy": 80.9, "final_nll": .9},)
    sections, figures, tables = joint_report_parts(tmp_path, synthetic_joint_reference, replay)
    sections = (replace(sections[0], paragraphs=("SYNTHETIC LAYOUT FIXTURE - NOT MEASURED RESULTS", *sections[0].paragraphs)), *sections[1:])
    assert len(figures) == 2 and len(tables["joint_convergence_endpoints"]) == 12
    assert len(tables["joint_convergence_epochs"]) == 160
    assert tables["joint_convergence_comparisons"][0]["joint_minus_replay_accuracy_points"] == pytest.approx(.1)
    output = tmp_path / "synthetic_joint_layout.pdf"
    render_report(sections, tmp_path, output)
    reader = PdfReader(output)
    assert len(reader.pages) == 3
    assert "SYNTHETIC LAYOUT" in reader.pages[0].extract_text()
    assert "validation accuracy-selected" in " ".join(reader.pages[0].extract_text().split())
    assert "Full-data refits" in reader.pages[-1].extract_text()

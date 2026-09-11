"""Synthetic-only checks for the fixed offline endpoint and its report appendix."""

from dataclasses import asdict, replace

import matplotlib.pyplot as plt
import pytest

from apm.continual.artifacts import atomic_write, canonical_json_bytes, file_sha256, publish_immutable_json, record_sha256
from apm.continual.vision.imagenetr.schedule_matched_reporting import (
    CONDITION, PLOT_LABEL, control_summary, draw_schedule_endpoint, load_schedule_reference, prediction_loss_parts, schedule_report_parts,
)
from apm.continual.vision.imagenetr.srt_evidence import sealed_record, write_parquet


@pytest.fixture
def synthetic_schedule_reference():
    """Build layout data explicitly separated from authenticated scientific results."""
    jobs = {f"seed_{seed}": {
        "seed": seed, "optimizer_steps": 56243, "training_presentations": 844640, "model_forward_images": 947040,
        "training_wall_seconds": 5000., "checkpoint_wall_seconds": 60., "evaluation_and_artifact_seconds": 150.,
        "blocks": [{"block": block, "steps_total": round(block * 56243 / 50),
                    "presentations_total": round(block * 844640 / 50), "training_presentations": 16000,
                    "optimizer_steps": 1000, "fit_probe": {"accuracy": min(99., 40 + 2 * block), "nll": 5 / block},
                    "training_accuracy": min(98., 30 + 2 * block), "training_nll": 6 / block,
                    "training_wall_seconds": 100., "checkpoint_wall_seconds": 1.2, "evaluation_and_artifact_seconds": 3.,
                    "model_sha256": "f" * 64} for block in range(1, 51)],
    } for seed in (1993, 1994, 1995)}
    evaluations = [{"seed": seed, "metrics": {"accuracy": 80. + seed - 1993, "nll": .8 + .1 * (seed - 1993),
                                               "examples": 6000, "wall_seconds": 10.},
                    "model_sha256": "f" * 64, "predictions_sha256": "b" * 64} for seed in (1993, 1994, 1995)]
    quality = tuple({"condition": CONDITION, "seed": row["seed"], "correct_nll_contribution": .05,
                     "wrong_nll_contribution": row["metrics"]["nll"] - .05} for row in evaluations)
    quality += ({"condition": "uniform_h4096_standard_rho80_unit8", "seed": 1993,
                 "correct_nll_contribution": .07, "wrong_nll_contribution": .83},)
    return {"result": {"jobs": jobs, "evaluations": evaluations}, "prediction_quality": quality}


def test_prediction_loss_parts_use_whole_test_denominator() -> None:
    rows = [{"prediction": prediction, "label": 0, "nll": nll} for prediction, nll in ((0, .1), (0, .2), (1, 2.), (1, 3.))]
    parts = prediction_loss_parts(CONDITION, 1993, rows)
    assert parts["correct_examples"] == parts["wrong_examples"] == 2
    assert parts["correct_mean_nll"] == pytest.approx(.15) and parts["wrong_mean_nll"] == 2.5
    assert parts["correct_nll_contribution"] == pytest.approx(.075) and parts["wrong_nll_contribution"] == 1.25
    assert parts["nll"] == pytest.approx(parts["correct_nll_contribution"] + parts["wrong_nll_contribution"])
    assert prediction_loss_parts(CONDITION, 1993, rows[:2])["wrong_mean_nll"] is None


def test_fixed_endpoint_mean_sd_and_final_only_marker(synthetic_schedule_reference) -> None:
    row = control_summary(synthetic_schedule_reference)
    assert row["accuracy_mean"] == 81. and row["accuracy_sample_sd"] == 1.
    assert row["nll_mean"] == pytest.approx(.9) and row["nll_sample_sd"] == pytest.approx(.1)
    assert row["condition"] == CONDITION and row["optimizer_steps_per_seed"] == 56243
    figure, axis = plt.subplots()
    draw_schedule_endpoint(axis, {"schedule_matched_joint": synthetic_schedule_reference})
    assert list(axis.lines[0].get_xdata()) == [50] and list(axis.lines[0].get_ydata()) == [81]
    assert axis.lines[0].get_marker() == "s" and axis.lines[0].get_markerfacecolor() == "white"
    assert axis.get_legend_handles_labels()[1] == [PLOT_LABEL]
    plt.close(figure)


def test_incomplete_seeds_and_unsafe_pointer_are_rejected(tmp_path, synthetic_schedule_reference) -> None:
    result = synthetic_schedule_reference["result"]
    with pytest.raises(ValueError, match="exactly three seeds"):
        control_summary({"result": {**result, "evaluations": result["evaluations"][:-1]}})
    source = tmp_path / "a/b/c/d/source"
    assert load_schedule_reference(source, "a" * 64) is None
    publish_immutable_json(source / "reports/schedule_matched_joint.json", sealed_record({
        "schema_version": "imagenetr50-schedule-matched-pointer-v1", "run": "../../outside",
    }))
    with pytest.raises(ValueError, match="namespace"):
        load_schedule_reference(source, "a" * 64)


@pytest.mark.integration
def test_report_authenticates_complete_synthetic_control(tmp_path, monkeypatch) -> None:
    """Exercise real evidence validation; only source replay extraction uses a synthetic schedule."""
    from apm.continual.vision.imagenetr import schedule_matched_reporting as reporting
    from apm.continual.vision.imagenetr.schedule_matched_joint import load_config
    from apm.continual.vision.imagenetr.schedule_matched_training import ScheduledBatch, require_schedule
    source = tmp_path / "artifacts/imagenetr50/srt_r16_v1/runs/source"
    predictions = tuple({"image_id": f"test_{index}", "task": index % 50 + 1, "label": index % 200,
                         "prediction": index % 200, "nll": .5} for index in range(6000))
    digest = write_parquet(source / "final/srt_h1024/stages/050/predictions.parquet", predictions)
    original = sealed_record({"conditions": {"srt_h1024": {"rows": [{"predictions_sha256": digest}]}}})
    source_protocol = sealed_record({"dataset_hash": "d" * 64, "model_sha256": "m" * 64})
    publish_immutable_json(source / "result.json", original)
    publish_immutable_json(source / "protocol.json", source_protocol)
    fitting = [f"train_{index}" for index in range(24000)]
    write_parquet(source / "final/image_index.parquet", tuple({"image_id": name} for name in fitting))
    batches = tuple(ScheduledBatch(step, min(50, (step - 1) // 1125 + 1), 16 if step <= 995 else 15)
                    for step in range(1, 56244))
    schedule_hash = require_schedule(batches, 64)
    config = replace(load_config(), source_result_hash=original["content_hash"])
    source_folder = source / "followups" / config.followup_hash / "final" / config.source_condition / "stages/050"
    source_digest = write_parquet(source_folder / "predictions.parquet", predictions)
    source_stage = sealed_record({"predictions_sha256": source_digest, "evaluation": {"accuracy": 100., "nll": .5}})
    publish_immutable_json(source_folder / "result.json", source_stage)
    evidence = {"schedule_hash": schedule_hash, "source_job_hash": "s" * 64,
                "blocks": ({"block": 1, "source_stage_hash": source_stage["content_hash"],
                            "batch_evidence": [{"path": "synthetic", "batches_sha256": "b" * 64}]},)}
    monkeypatch.setattr(reporting, "source_schedule", lambda _source, _config: (batches, evidence))
    optimizer = {"lora_learning_rate": .0005, "head_learning_rate": .01, "momentum": .9, "weight_decay": .0005}
    protocol = sealed_record({"schema_version": "imagenetr50-schedule-matched-protocol-v1", "config": asdict(config),
                              "source_result_hash": original["content_hash"], **evidence, "optimizer": optimizer,
                              "optimizer_config_hash": "o" * 64,
                              **{key: source_protocol[key] for key in ("dataset_hash", "model_sha256")}})
    root = tmp_path / "artifacts/imagenetr50/schedule_matched_joint_r16/runs" / protocol["content_hash"]
    publish_immutable_json(root / "protocol.json", protocol)
    schedule_digest = write_parquet(root / "schedule.parquet", tuple(asdict(batch) for batch in batches))
    publish_immutable_json(root / "schedule.json", sealed_record({**evidence, "schedule_parquet_sha256": schedule_digest}))
    populations = sealed_record({"fitting": fitting, "probe": fitting[:2048], "test": [row["image_id"] for row in predictions]})
    publish_immutable_json(root / "populations.json", populations)
    updates = tuple({"step": batch.step, "block": batch.block, "batch_size": batch.size,
                     "loss_sum": float(batch.size), "gradient_norm": float(batch.step),
                     "lora_learning_rate": .0005, "head_learning_rate": .01} for batch in batches)
    jobs, audits, evaluations = {}, {}, []
    for seed in (1993, 1994, 1995):
        name = f"seed_{seed}"
        folder = root / "seeds" / name
        definition = sealed_record({"schema_version": "imagenetr50-schedule-matched-job-v1", "protocol_hash": root.name,
                                    "fitting_examples": 24000, "probe_examples": 2048,
                                    "fitting_ids_hash": record_sha256(fitting), "probe_ids_hash": record_sha256(fitting[:2048]),
                                    "schedule_hash": schedule_hash, "optimizer": optimizer, "optimizer_config_hash": "o" * 64,
                                    "planned_steps": 56243, "planned_presentations": 844640, "maximum_batch_size": 64})
        publish_immutable_json(folder / "job.json", definition)
        blocks = []
        for block in range(1, 51):
            block_root = folder / "blocks" / f"{block:03d}"
            atomic_write(block_root / "model.safetensors", f"SYNTHETIC HASH FIXTURE {seed} {block}".encode())
            row = sealed_record({"block": block, "model_sha256": file_sha256(block_root / "model.safetensors")})
            publish_immutable_json(block_root / "result.json", row)
            blocks.append(row)
        chunk_hash = write_parquet(folder / "updates/synthetic.parquet", updates)
        exposure_hash = write_parquet(folder / "exposures.parquet", ({"fixture": "synthetic"},))
        job = sealed_record({"schema_version": "imagenetr50-schedule-matched-job-result-v1", "job_hash": definition["content_hash"],
                             "seed": seed, "optimizer_steps": 56243, "training_presentations": 844640, "model_forward_images": 947040,
                             "blocks": blocks, "chunks": [{"path": "updates/synthetic.parquet", "sha256": chunk_hash, "steps": 56243}],
                             "exposures_sha256": exposure_hash, "test_used": False, "stop_reason": "complete_frozen_schedule"})
        publish_immutable_json(folder / "result.json", job)
        audit = sealed_record({"result_hash": job["content_hash"], "verified_updates": 56243, "verified_presentations": 844640,
                               "training_only": True, "schedule_and_rates_matched": True})
        publish_immutable_json(folder / "draw_audit.json", audit)
        jobs[name], audits[name] = job, audit
        output = root / "evaluations" / name
        prediction_hash = write_parquet(output / "predictions.parquet", predictions)
        evaluated = sealed_record({"seed": seed, "protocol_hash": root.name, "model_sha256": blocks[-1]["model_sha256"],
                                   "predictions_sha256": prediction_hash, "metrics": {"examples": 6000, "accuracy": 100., "nll": .5}})
        publish_immutable_json(output / "result.json", evaluated)
        evaluations.append(evaluated)
    result = sealed_record({"schema_version": "imagenetr50-schedule-matched-result-v1", "protocol_hash": root.name,
                            "source_result_hash": original["content_hash"], "schedule_hash": schedule_hash,
                            "jobs": jobs, "draw_audits": audits, "evaluations": evaluations,
                            "reuse": {name: {"optimizer_steps": 0, "result_hash": job["content_hash"]} for name, job in jobs.items()},
                            "zero_step_reuse": True, "source_unchanged": True})
    publish_immutable_json(root / "result.json", result)
    publish_immutable_json(source / "reports/schedule_matched_joint.json", sealed_record({
        "schema_version": "imagenetr50-schedule-matched-pointer-v1", "run": str(root.relative_to(tmp_path)),
        "source_result_hash": original["content_hash"], "result_hash": result["content_hash"],
    }))
    monkeypatch.chdir(tmp_path)
    reference = load_schedule_reference(source.relative_to(tmp_path), original["content_hash"])
    assert reference["result"] == result
    assert len(reference["prediction_quality"]) == 4 and reference["prediction_quality"][-1]["wrong_examples"] == 0
    export = reporting.export_schedule_updates(source, reference)
    assert len(export["tables"]) == 3 and sum(row["optimizer_steps"] for row in export["tables"]) == 168729
    assert export["tables"][0]["early_peak_batch_nll"] == 1. and export["tables"][0]["early_peak_gradient_norm"] == 1125.
    assert reporting.export_schedule_updates(source, reference) == export
    import pyarrow.parquet as pq
    assert pq.read_table(root / export["tables"][0]["path"]).to_pylist() == list(updates)
    changed = sealed_record({**{key: value for key, value in populations.items() if key != "content_hash"},
                             "probe": [*populations["probe"][:-1], "test_0"]})
    atomic_write(root / "populations.json", canonical_json_bytes(changed))
    with pytest.raises(ValueError, match="not isolated"):
        load_schedule_reference(source, original["content_hash"])
    atomic_write(root / "populations.json", canonical_json_bytes(populations))
    atomic_write(root / "evaluations/seed_1993/predictions.parquet", b"altered synthetic prediction")
    with pytest.raises(ValueError, match="predictions changed"):
        load_schedule_reference(source, original["content_hash"])


@pytest.mark.integration
def test_schedule_matched_appendix_layout(tmp_path, synthetic_schedule_reference) -> None:
    """Render synthetic pages and check work totals, comparisons, and reusable output."""
    from pypdf import PdfReader
    from apm.continual.artifacts import file_sha256
    from apm.continual.vision.imagenetr.srt_reporting import render_report
    joint = {"result": {
        "selection": {"endpoints": {"accuracy_selected": 19, "five_epoch": 5, "nll_selected": 3, "terminal": 47}},
        "evaluations": [{"seed": seed, "epoch": epoch, "metrics": {"accuracy": 79., "nll": 1.}}
                        for seed in (1993, 1994, 1995) for epoch in (3, 5, 19, 47)],
    }}
    replay = tuple({"condition": name, "final_accuracy": 80.9, "final_nll": .9}
                   for name in ("uniform_h4096_standard_rho80_unit8", "uniform_h1024"))
    update_export = {"tables": [{"seed": seed, "first_block_updates": 106, "early_peak_batch_nll": 10. * (seed - 1992),
                                "early_peak_loss_step": 90, "early_peak_loss_batch_size": 1,
                                "early_peak_gradient_norm": 200. * (seed - 1992), "early_peak_gradient_step": 78}
                               for seed in (1993, 1994, 1995)]}
    sections, figures, tables = schedule_report_parts(tmp_path, synthetic_schedule_reference, joint, replay, update_export)
    sections = tuple(replace(section, paragraphs=("SYNTHETIC LAYOUT FIXTURE - NOT MEASURED RESULTS", *section.paragraphs))
                     for section in sections)
    assert len(figures) == 2 and len(tables) == 7 and len(tables["schedule_matched_blocks"]) == 150
    assert len(tables["schedule_matched_comparisons"]) == 7 and len(tables["schedule_matched_endpoints"]) == 3
    assert sum(row["all_forward_images"] for row in tables["schedule_matched_resources"]) == 2859120
    assert tables["schedule_matched_comparisons"][-1]["control_minus_reference_accuracy_points"] == pytest.approx(.1)
    output = tmp_path / "synthetic_schedule_layout.pdf"
    render_report(sections, tmp_path, output)
    reader = PdfReader(output)
    assert len(reader.pages) == 4 and all("SYNTHETIC LAYOUT FIXTURE" in page.extract_text() for page in reader.pages)
    assert "56,243" in reader.pages[0].extract_text() and "2,859,120" in reader.pages[2].extract_text()
    digest = file_sha256(output)
    render_report(sections, tmp_path, output)
    assert file_sha256(output) == digest

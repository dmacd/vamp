"""Bounded synthetic evidence tests for report auditing and common rendering."""

from dataclasses import asdict, replace
from hashlib import sha256

import pytest

from apm.continual.artifacts import publish_immutable_json, record_sha256
from apm.continual.vision.imagenetr.srt_analysis import analyze_replay_job
from apm.continual.vision.imagenetr.srt_evidence import TraceBuffer, sealed_record, write_parquet
from apm.continual.vision.imagenetr.srt_scheduler import (
    RecallPolicy, ReviewBatch, ReviewScheduler, observe_batch, select_due,
)


@pytest.fixture
def tiny_full_stream(tmp_path):
    root = tmp_path / "srt"
    index = tuple({"image": image, "image_id": sha256(str(image).encode()).hexdigest(),
                   "arrival_stage": image + 1, "class_id": 4 * image} for image in range(50))
    index_path = tmp_path / "image_index.parquet"
    write_parquet(index_path, index)
    job = sealed_record({"schema_version": "imagenetr50-srt-job-v1", "training_ids_hash": record_sha256([row["image_id"] for row in index])})
    publish_immutable_json(root / "job.json", job)
    policy = RecallPolicy("standard", (.1, .25, .5, .75, .9), .5, 8)
    scheduler, steps, presentations = ReviewScheduler(), 0, 0
    rows = []
    for stage in range(1, 51):
        scheduler = scheduler.arrive(stage)
        stage_presentations, stage_steps, trace = 0, 0, TraceBuffer()
        while stage_presentations < 4 * stage:
            if stage_steps == 0:
                batch = ReviewBatch((stage - 1,), 0, 1, True)
                selected = scheduler
            else:
                batch, selected = select_due(scheduler, policy, min(64, 4 * stage - stage_presentations), 1993, steps + 1)
            scheduler, events = observe_batch(selected, batch, (.85,) * len(batch.images), policy, "srt", steps + 1, presentations)
            steps += 1
            stage_steps += 1
            presentations += len(batch.images)
            stage_presentations += len(batch.images)
            trace = trace.append(events, {"stage": stage, "step": steps, "presentations": len(batch.images),
                                           "historical_count": batch.historical_count, "current_count": batch.current_count,
                                           "introduction": batch.introduction, "clock_advance": batch.clock_advance})
        chunk = trace.publish(root)
        predictions = tuple({"image_id": index[image]["image_id"], "task": image + 1, "label": 4 * image,
                             "prediction": 4 * image, "nll": .2} for image in range(stage))
        prediction_hash = write_parquet(root / f"stages/{stage:03d}/predictions.parquet", predictions)
        row = sealed_record({
            "schema_version": "imagenetr50-srt-stage-v1", "stage": stage,
            "current_examples": 1, "historical_examples_available": stage - 1, "trace_chunks": [chunk],
            "predictions_sha256": prediction_hash,
            "evaluation": {"accuracy": 100.0, "nll": .2, "examples": stage, "wall_seconds": .001},
            "fit": {"image_presentations": stage_presentations, "optimizer_steps": stage_steps,
                    "wall_seconds": .01, "loader_wall_seconds": .001, "scheduler_wall_seconds": .001,
                    "checkpoint_wall_seconds": .001, "peak_vram_bytes": 0},
            "scheduler": {"clock": scheduler.clock, "historical_due": len(scheduler.ready().historical),
                          "current_due": len(scheduler.ready().current),
                          "never_historically_replayed": sum(not state.historical_reviews for state in scheduler.states.values())},
        })
        publish_immutable_json(root / f"stages/{stage:03d}/result.json", row)
        rows.append(row)
    publish_immutable_json(root / "result.json", sealed_record({
        "schema_version": "imagenetr50-srt-job-result-v1", "job_hash": job["content_hash"], "rows": rows,
        "image_presentations": presentations, "optimizer_steps": steps, "method": "srt", "capacity": 1024, "policy": asdict(policy),
    }))
    return root, index_path


def test_streaming_report_audit_counts_all_samples_and_censored_waits(tiny_full_stream) -> None:
    root, index_path = tiny_full_stream
    analysis = analyze_replay_job(root, index_path, "srt_h1024")
    assert len(analysis.stages) == len(analysis.samples) == 50
    assert analysis.totals["training_presentations"] == 5100
    assert sum(row["presentations"] for row in analysis.samples) == 5100
    assert all(row["next_review_censored"] for row in analysis.samples)
    assert analysis.samples[-1]["historical_reviews"] == 0
    assert analysis.totals["final_accuracy"] == 100
    completed_gaps = sum(row["count"] for row in analysis.histograms if row["metric"] == "step_gap" and row["previous_quality"] == "all")
    assert completed_gaps == 5050
    assert {row["arrival_stage"] for row in analysis.timelines} == {1, 16, 31, 50}


def test_report_rejects_changed_image_identity_mapping(tiny_full_stream, tmp_path) -> None:
    root, index_path = tiny_full_stream
    import pyarrow.parquet as pq
    records = pq.read_table(index_path).to_pylist()
    records[0]["image_id"] = "f" * 64
    altered = tmp_path / "altered.parquet"
    write_parquet(altered, records)
    with pytest.raises(ValueError, match="ordered image index"):
        analyze_replay_job(root, altered, "srt_h1024")


def test_common_pdf_html_markdown_rendering(tmp_path) -> None:
    from pypdf import PdfReader
    from apm.continual.vision.imagenetr.srt_reporting import ReportSection, ReportTable, render_report
    sections = (
        ReportSection("Synthetic layout fixture", ("This is a unit-test fixture, not an experimental result.",),
                      table=ReportTable(("Condition", "Accuracy"), (("SRT fixture", "80.0%"),))),
        ReportSection("Second page", ("Repeated condition names remain identical across formats.",)),
    )
    output = tmp_path / "layout.pdf"
    render_report(sections, tmp_path, output)
    reader = PdfReader(output)
    assert len(reader.pages) == 2
    assert "SRT fixture" in reader.pages[0].extract_text()
    assert "SRT fixture" in (tmp_path / "REPORT.html").read_text()
    assert "SRT fixture" in (tmp_path / "REPORT.md").read_text()


def test_analysis_tables_preserve_unbounded_virtual_clocks(tmp_path) -> None:
    import json
    import pyarrow.parquet as pq
    from apm.continual.vision.imagenetr.srt_reporting import _write_table
    records = ({"image": 1, "next_due_clock": 10 ** 80, "clock": 10 ** 79, "terminal_overdue_ticks": None},)
    _write_table(tmp_path, "exact_clocks", records)
    expected = [{"image": 1, "next_due_clock": str(10 ** 80), "clock": str(10 ** 79), "terminal_overdue_ticks": None}]
    assert json.loads((tmp_path / "exact_clocks.json").read_text()) == expected
    assert pq.read_table(tmp_path / "exact_clocks.parquet").to_pylist() == expected
    assert str(10 ** 80) in (tmp_path / "exact_clocks.csv").read_text()


@pytest.fixture
def fixed_policy_report_source(tmp_path):
    """Seal minimal explicit extension provenance without fabricating model results."""
    from apm.continual.vision.imagenetr.srt_config import load_srt_config
    from apm.continual.vision.imagenetr.srt_followup import followup_conditions, load_followup_config
    training, config = load_srt_config(), load_followup_config()
    jobs = followup_conditions(config, training)
    source_hash = "a" * 64
    protocol = sealed_record({"schema_version": "imagenetr50-srt-followup-protocol-v1", "source_result_hash": source_hash,
                              "conditions": [name for name, _, _ in jobs]})
    root = tmp_path / "followups" / protocol["content_hash"]
    publish_immutable_json(root / "protocol.json", protocol)
    publish_immutable_json(tmp_path / "config_resolved.json", sealed_record(asdict(training)))
    results, definitions = {}, {}
    for name, method, policy in jobs:
        definitions[name] = sealed_record({"schema_version": "imagenetr50-srt-job-v1", "protocol_hash": protocol["content_hash"],
                                           "policy": asdict(policy), "method": method, "capacity": 4096,
                                           "paired_job_hash": definitions[name.replace("uniform_", "srt_", 1)]["content_hash"] if method == "uniform" else None})
        results[name] = sealed_record({"schema_version": "imagenetr50-srt-job-result-v1", "job_hash": definitions[name]["content_hash"],
                                      "policy": asdict(policy), "method": method, "capacity": 4096,
                                      "rows": [{}] * 50, "image_presentations": 844640})
        for filename, record in (("job.json", definitions[name]), ("result.json", results[name])):
            publish_immutable_json(root / "final" / name / filename, record)
    result = sealed_record({"schema_version": "imagenetr50-srt-followup-result-v1", "source_result_hash": source_hash,
                            "protocol_hash": protocol["content_hash"], "conditions": results, "source_unchanged": True,
                            "zero_step_reuse": True, "reuse": {name: {"optimizer_steps": 0, "result_hash": job["content_hash"]}
                                                                 for name, job in results.items()}})
    publish_immutable_json(root / "result.json", result)
    publish_immutable_json(tmp_path / "reports/fixed_policy_followup.json", sealed_record({
        "schema_version": "imagenetr50-srt-followup-pointer-v1", "source_result_hash": source_hash,
        "run_hash": protocol["content_hash"], "result_hash": result["content_hash"],
    }))
    return tmp_path, source_hash, root


def test_report_loads_only_complete_authenticated_followups(fixed_policy_report_source) -> None:
    from apm.continual.vision.imagenetr.srt_reporting import FOLLOWUP_STYLES, followup_report_jobs
    source, source_hash, _ = fixed_policy_report_source
    roots, evidence = followup_report_jobs(source, source_hash)
    assert set(roots) == set(FOLLOWUP_STYLES)
    assert evidence["pointer"]["source_result_hash"] == source_hash
    with pytest.raises(ValueError, match="identity or source"):
        followup_report_jobs(source, "b" * 64)


def test_report_rejects_changed_followup_recipe(fixed_policy_report_source) -> None:
    from apm.continual.artifacts import atomic_write, canonical_json_bytes
    from apm.continual.vision.imagenetr.srt_evidence import read_sealed
    from apm.continual.vision.imagenetr.srt_reporting import followup_report_jobs
    source, source_hash, root = fixed_policy_report_source
    path = root / "final/srt_h4096_standard_rho80_unit8/job.json"
    record = read_sealed(path)
    record = {**record, "policy": {**record["policy"], "historical_fraction": .5}}
    atomic_write(path, canonical_json_bytes(sealed_record({key: value for key, value in record.items() if key != "content_hash"})))
    with pytest.raises(ValueError, match="explicit requested recipe"):
        followup_report_jobs(source, source_hash)


@pytest.mark.integration
@pytest.mark.parametrize("include_followup", (False, True))
def test_complete_synthetic_report_layout(tiny_full_stream, tmp_path, include_followup) -> None:
    """Render all paper pages from explicitly synthetic measurements for visual QA."""
    import pandas as pd
    from pypdf import PdfReader
    from apm.continual.vision.imagenetr import srt_reporting as reporting
    root, index_path = tiny_full_stream
    source = analyze_replay_job(root, index_path, "srt_h1024")
    analyses = {
        condition: replace(source, **{
            name: tuple({**row, "condition": condition} for row in getattr(source, name))
            for name in ("stages", "samples", "histograms", "timelines")
        }, totals={**source.totals, "condition": condition})
        for condition in (reporting.ALL_STYLES if include_followup else reporting.NEW_STYLES)
    }
    evaluation = {"accuracy": 80., "nll": .8, "true_node_oracle_accuracy": 85.}
    references = {"arms": {str(capacity): [{"evaluation": evaluation}] * 50 for capacity in (4096, 8192)},
                  "mlp": {"rows": [{"evaluation": evaluation}] * 50},
                  "stage_matched_joint": [evaluation] * 50, "rank_matched_joint": [evaluation] * 50}
    reports = tmp_path / "reports"
    reports.mkdir()
    records = {name: tuple(row for analysis in analyses.values() for row in getattr(analysis, name))
               for name in ("stages", "samples", "histograms", "timelines")}
    for name, rows in records.items():
        reporting._write_table(reports, name, rows)
    stages, samples, histograms, timelines = (pd.DataFrame(records[name]) for name in records)
    resources = tuple({**row, "cumulative_training_backwards": row["cumulative_presentations"],
                       "cumulative_recompute_forwards": 0} for row in records["stages"])
    resources += tuple({**row, "condition": condition} for row in resources[:50]
                       for condition in ("persistent_h4096", "persistent_h8192", "persistent_mlp_h4096", "joint_rank16", "joint_rank_matched"))
    figures = {
        "accuracy": reporting.plot_accuracy(reports, references, stages),
        "nll": reporting._plot_nll_and_gaps(reports, references, stages),
        "resources": reporting._plot_resources(reports, pd.DataFrame(resources)),
        "replay": reporting._plot_replay(reports, histograms, stages),
        "intervals": reporting._plot_requested_intervals(reports, histograms),
        "coverage": reporting._plot_sample_coverage(reports, samples, stages),
        "timelines": reporting._plot_timelines(reports, timelines[timelines.condition.isin(reporting.NEW_STYLES)]),
    }
    if include_followup:
        figures.update({"followup_accuracy": reporting.plot_followup_accuracy(reports, references, stages),
                        "policy_comparisons": reporting.plot_policy_comparisons(reports, references, stages),
                        "optimizer_work": reporting.plot_optimizer_work(reports, stages),
                        "followup_timelines": reporting._plot_timelines(reports, timelines[timelines.condition.isin(reporting.FOLLOWUP_STYLES)],
                                                                        "fixed_policy_sample_timelines.png")})
    from apm.continual.artifacts import atomic_write
    atomic_write(tmp_path / "references/previous_stage_accuracy.png", figures["accuracy"].read_bytes())
    calibration = tuple({"capacity": capacity, "profile": profile, "historical_fraction": fraction,
                         "interval_unit": unit, "screen_accuracy": 80., "full_mean_accuracy": 80. if profile == "relaxed" and fraction == .2 else None,
                         "selected": profile == "relaxed" and fraction == .2 and unit == 1,
                         "presentations": 100, "training_wall_seconds": 1., "evaluation_wall_seconds": .1}
                        for capacity in (1024, 4096) for profile in ("relaxed", "standard", "strict")
                        for fraction in (.2, .5, .8) for unit in (1, 8))
    sections = reporting._sections(tmp_path, reports, {}, references, analyses, calibration, resources, figures)
    sections = tuple(replace(section, paragraphs=("SYNTHETIC LAYOUT FIXTURE - NOT EXPERIMENTAL RESULTS.", *section.paragraphs))
                     for section in sections)
    pdf = reports / "synthetic_layout.pdf"
    reporting.render_report(sections, reports, pdf)
    pages = PdfReader(pdf).pages
    assert len(pages) == len(sections)
    assert all("SYNTHETIC LAYOUT FIXTURE" in page.extract_text() for page in pages)
    first_hash = sha256(pdf.read_bytes()).hexdigest()
    reporting.render_report(sections, reports, pdf)
    assert sha256(pdf.read_bytes()).hexdigest() == first_hash

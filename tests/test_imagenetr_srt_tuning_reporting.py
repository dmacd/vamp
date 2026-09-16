"""Bounded provenance guards and explicit synthetic layout checks for H=128 tuning."""

from dataclasses import asdict, replace
from pathlib import Path

import pytest

from apm.continual.artifacts import file_sha256, publish_immutable_json
from apm.continual.vision.imagenetr.srt_evidence import sealed_record
from apm.continual.vision.imagenetr.srt_tuning_reporting import load_tuning_reference, tuning_report_parts


def test_tuning_report_is_optional_and_rejects_pointer_escape(tmp_path):
    source = tmp_path / "source"
    assert load_tuning_reference(source, "s" * 64) is None
    assert tuning_report_parts(tmp_path, None, {}, {}) == ((), {}, {})
    publish_immutable_json(source / "reports/srt_h128_tuning.json", sealed_record({
        "schema_version": "imagenetr50-srt-tuning-pointer-v1", "run_hash": "../../outside"}))
    with pytest.raises(ValueError, match="namespace"):
        load_tuning_reference(source, "s" * 64)


@pytest.mark.integration
@pytest.mark.parametrize("baseline_reused", (False, True))
def test_tuning_appendix_layout_and_complete_candidate_table(tmp_path, baseline_reused):
    """Render visibly synthetic search and final-pair evidence, never fabricated science artifacts."""
    from pypdf import PdfReader
    from apm.continual.vision.imagenetr.srt_analysis import ReplayAnalysis
    from apm.continual.vision.imagenetr.srt_config import load_srt_config
    from apm.continual.vision.imagenetr.srt_reporting import render_report
    from apm.continual.vision.imagenetr.srt_tuning_config import load_config, policy_candidates, rate_candidates, refinement_candidates, unique_candidates
    config, training = load_config(), load_srt_config()
    initial = policy_candidates(config, training)
    baseline = next(candidate for candidate in initial if candidate.policy.name == "standard_rho80_unit8")
    candidates = unique_candidates(initial + rate_candidates(config, training, baseline) + refinement_candidates(config, baseline))
    chosen = baseline if baseline_reused else candidates[-1]
    summaries = tuple({"candidate": asdict(candidate), "candidate_hash": candidate.content_hash, "job": candidate.name,
                       "status": "complete", "validation_accuracy": 80. + index / 100, "validation_nll": .8,
                       "mean_stage_accuracy": 85., "image_presentations": 101888, "optimizer_steps": 2000,
                       "training_wall_seconds": 60., "validation_wall_seconds": 10., "validation_forward_images": 120000}
                      for index, candidate in enumerate(candidates))
    final_names = tuple(f"{method}_h128_{'standard_rho80_unit8' if baseline_reused else 'tuned'}" for method in ("srt", "uniform"))
    reference = {"selection": {"candidates": summaries, "selected": next(row for row in summaries if row["candidate_hash"] == chosen.content_hash)},
                 "result": {"baseline_reused": baseline_reused, "conditions": dict.fromkeys(final_names)},
                 "candidate_phases": {candidate.content_hash: "policy" if index < 18 else "rates" if index < 26 else "refinement"
                                      for index, candidate in enumerate(candidates)}}
    names = ("srt_h128_standard_rho80_unit8", "uniform_h128_standard_rho80_unit8", *final_names)
    stages = tuple({"stage": stage, "accuracy": 80., "nll": .8} for stage in range(1, 51))
    totals = {"final_accuracy": 80., "final_nll": .8, "mean_stage_accuracy": 85., "optimizer_steps": 2200}
    analyses = {name: ReplayAnalysis(stages, (), tuple({"condition": name, "metric": metric, "kind": "historical",
                                                       "previous_quality": "all", "count": 10, "upper": 8}
                                                      for metric in ("interval_before", "clock_gap")), (), totals) for name in names}
    references = {"stage_matched_joint": [{"accuracy": 79.}] * 50}
    sections, figures, tables = tuning_report_parts(tmp_path, reference, references, analyses)
    assert len(sections) == (5 if baseline_reused else 6)
    assert len(figures) == (2 if baseline_reused else 3)
    assert len(tables["h128_tuning_candidates"]) == 32
    assert sum(row["selected"] for row in tables["h128_tuning_candidates"]) == 1
    sections = tuple(replace(section, paragraphs=("SYNTHETIC LAYOUT FIXTURE - NOT EXPERIMENTAL RESULTS.", *section.paragraphs)) for section in sections)
    pdf = tmp_path / "synthetic_tuning.pdf"
    render_report(sections, tmp_path, pdf)
    pages = PdfReader(pdf).pages
    assert len(pages) == len(sections)
    assert all("SYNTHETIC LAYOUT FIXTURE" in page.extract_text() for page in pages)
    assert ("exactly equals the original" in pages[0].extract_text()) == baseline_reused
    digest = file_sha256(pdf)
    render_report(sections, tmp_path, pdf)
    assert file_sha256(pdf) == digest


@pytest.mark.integration
def test_completed_real_tuning_selection_and_report_authenticate():
    """Audit frozen phase choices, held-out predictions, chosen refits and report exports."""
    from concurrent.futures import ThreadPoolExecutor
    import pyarrow.parquet as pq
    from apm.continual.vision.imagenetr.srt_evidence import read_sealed
    from apm.continual.vision.imagenetr.srt_followup import load_followup_config

    project = Path(__file__).resolve().parents[1]
    config = load_followup_config(project / "configs/vision/imagenetr/srt_h128_rho80_unit8.yaml")
    source = project / config.source_run
    if not (source / "reports/srt_h128_tuning.json").is_file():
        pytest.skip("the full H=128 validation search has not completed locally")
    reference = load_tuning_reference(source, config.source_result_hash)
    report = read_sealed(source / "reports/report_manifest.json")
    assert report["h128_tuning"]["result_hash"] == reference["result"]["content_hash"]
    assert report["event_audit_passed"] and report["paired_exposure_audit_passed"]
    assert file_sha256(Path(report["pdf"])) == report["pdf_sha256"]
    candidates = pq.read_table(source / "reports/h128_tuning_candidates.parquet").to_pylist()
    assert {row["candidate_hash"] for row in candidates} == {
        row["candidate_hash"] for row in reference["selection"]["candidates"]}
    assert len(candidates) <= 32 and sum(row["selected"] for row in candidates) == 1
    selected = next(row for row in candidates if row["selected"])
    assert selected["candidate_hash"] == reference["selection"]["selected"]["candidate_hash"]
    conditions = set(reference["result"]["conditions"])
    assert conditions <= report["condition_names"].keys()
    stages = pq.read_table(source / "reports/stage_metrics.parquet").to_pylist()
    assert len([row for row in stages if row["condition"] in conditions]) == 100
    samples = pq.read_table(source / "reports/replay_samples.parquet").to_pylist()
    assert len([row for row in samples if row["condition"] in conditions]) == 48000
    candidate_roots = tuple(reference["root"] / "calibration" / row["job"] for row in reference["selection"]["candidates"]
                            if row["status"] == "complete")
    jobs = tuple((root, read_sealed(root / "result.json")) for root in (*candidate_roots, *reference["roots"].values()))
    for root, job in jobs:
        assert len(job["rows"]) == 50
        assert all(read_sealed(root / f"stages/{row['stage']:03d}/result.json") == row for row in job["rows"])
    checks = tuple((root / f"stages/{row['stage']:03d}" / filename, row[field]) for root, job in jobs for row in job["rows"]
                   for filename, field in (("model.safetensors", "model_sha256"), ("predictions.parquet", "predictions_sha256")))
    with ThreadPoolExecutor(max_workers=8) as executor:
        assert tuple(executor.map(file_sha256, (path for path, _ in checks))) == tuple(expected for _, expected in checks)

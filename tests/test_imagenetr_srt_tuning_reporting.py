"""Bounded provenance guards and explicit synthetic layout checks for H=128 tuning."""

from dataclasses import asdict, replace

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
    analyses = {name: ReplayAnalysis(stages, (), (), (), totals) for name in names}
    references = {"stage_matched_joint": [{"accuracy": 79.}] * 50}
    sections, figures, tables = tuning_report_parts(tmp_path, reference, references, analyses)
    assert len(sections) == 5 and len(figures) == 2
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

"""Synthetic layout and explicit real-artifact checks for checkpoint diagnostics."""

from dataclasses import replace
from pathlib import Path

import pytest

from apm.continual.artifacts import file_sha256, publish_immutable_json
from apm.continual.vision.imagenetr.checkpoint_diagnostic_reporting import LABELS, checkpoint_report_parts, load_checkpoint_diagnostics
from apm.continual.vision.imagenetr.srt_evidence import sealed_record


def synthetic_diagnostics():
    """Supply visibly synthetic, complete layout inputs without claiming model measurements."""
    summary = tuple({"condition": name, "score_mode": mode, "accuracy": 80., "nll": 1. if mode == "raw" else .7,
                     "temperature_min": 1.2, "temperature_max": 1.3} for name in LABELS for mode in ("raw", "calibrated"))
    groups = ("all", "unrevisited_tasks40_50", "unrevisited_and_last_probability_ge90", "revisited_tasks40_50",
              "top1pct_presentations", *(f"last_review_{lo:02d}_{hi:02d}" for lo, hi in ((1, 10), (11, 20), (21, 30), (31, 39), (40, 49), (50, 50))))
    cohorts = tuple({"profile": profile, "method": method, "cohort": group, "accuracy": 75. if method == "srt" else 85.,
                     "nll": 1. if method == "srt" else .5, "examples": 240, "now_wrong": 60 if method == "srt" else 36}
                    for profile in ("standard", "strict") for method in ("srt", "uniform") for group in groups)
    reliability = tuple({"condition": name, "score_mode": mode, "examples": 400, "confidence": (index + .5) / 15,
                         "accuracy": (index + .5) / 15} for name in LABELS for mode in ("raw", "calibrated") for index in range(15))
    return {"tables": {"checkpoint_probability_summary": summary, "checkpoint_training_cohorts": cohorts,
                       "checkpoint_reliability": reliability, "checkpoint_resources": ({"wall_seconds": 1.},),
                       "checkpoint_temperature_fits": ({"at_numerical_bound": False},),
                       "checkpoint_prediction_parity": ({"maximum_nll_difference": 1e-6},)}}


def test_missing_diagnostics_are_optional_but_unsafe_pointer_is_rejected(tmp_path):
    source = tmp_path / "artifacts/imagenetr50/srt_r16_v1/runs/source"
    assert load_checkpoint_diagnostics(source, "s" * 64) is None
    assert checkpoint_report_parts(tmp_path, None) == ((), {}, {})
    publish_immutable_json(source / "reports/checkpoint_diagnostics.json", sealed_record({
        "schema_version": "imagenetr50-checkpoint-diagnostics-pointer-v1", "run": "../../outside"}))
    with pytest.raises(ValueError, match="namespace"):
        load_checkpoint_diagnostics(source, "s" * 64)


@pytest.mark.integration
def test_checkpoint_diagnostic_layout_is_four_deterministic_pages(tmp_path):
    """Keep the post-hoc appendix separate and readable in the shared PDF renderer."""
    from pypdf import PdfReader
    from apm.continual.vision.imagenetr.srt_reporting import render_report
    sections, figures, tables = checkpoint_report_parts(tmp_path, synthetic_diagnostics())
    sections = tuple(replace(section, paragraphs=("SYNTHETIC LAYOUT FIXTURE - NOT EXPERIMENT RESULTS", *section.paragraphs)) for section in sections)
    assert len(sections) == 4 and len(figures) == 2
    assert len(tables["checkpoint_probability_summary"]) == 14
    output = tmp_path / "synthetic_diagnostics.pdf"
    render_report(sections, tmp_path, output)
    reader = PdfReader(output)
    assert len(reader.pages) == 4
    assert all("SYNTHETIC LAYOUT FIXTURE" in page.extract_text() for page in reader.pages)
    assert "4,800" in reader.pages[0].extract_text()
    digest = file_sha256(output)
    render_report(sections, tmp_path, output)
    assert file_sha256(output) == digest


@pytest.mark.integration
def test_completed_real_checkpoint_diagnostics_authenticate_and_reconstruct():
    """Explicitly audit the finished seven-checkpoint result without GPU inference."""
    from apm.continual.vision.imagenetr.checkpoint_diagnostics import load_config
    config = load_config()
    source = Path(config.source_run)
    if not (source / "reports/checkpoint_diagnostics.json").is_file():
        pytest.skip("real checkpoint diagnostics not complete")
    reference = load_checkpoint_diagnostics(source, config.source_result_hash)
    assert reference["result"]["optimizer_steps"] == 0
    assert len(reference["tables"]["checkpoint_test_predictions"]) == 42000
    assert not any(row["at_numerical_bound"] for row in reference["tables"]["checkpoint_temperature_fits"])

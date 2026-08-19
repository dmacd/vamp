from __future__ import annotations

from pathlib import Path

import pytest

from apm.continual.artifacts import publish_immutable_json, record_sha256
from apm.continual.trace.jobs import JobLedger, JobSpec
from apm.continual.trace.reporting import (
    _validation_policy_rows,
    _vamp_comparison_gaps,
    build_report,
)


def test_report_calculates_registered_gaps_for_task_free_vamp_routers() -> None:
    records = tuple(
        {
            "condition": "joint_iid_lora",
            "router_scores": {"direct": 50.0},
            "stage": 8,
            "task_index": task_index,
        }
        for task_index in range(1, 9)
    )
    summaries = (
        {"condition": "seq_lora_reference", "op": 40.0, "router": "direct"},
        {"condition": "vamp_svd_r8_repair000", "op": 44.0, "router": "prompt_nll"},
        {"condition": "vamp_svd_r8_repair000", "op": 55.0, "router": "task_aware"},
    )

    gaps = _vamp_comparison_gaps(records, summaries)

    assert len(gaps) == 1
    assert gaps[0]["joint_iid_vs_vamp"] == 6.0
    assert gaps[0]["vamp_vs_sequential"] == 4.0


def test_validation_policy_selection_never_uses_test_scores() -> None:
    records = tuple(
        {
            "condition": condition,
            "router_scores": {"prompt_nll": test_score},
            "stage": 8,
            "validation_router_scores": {"prompt_nll": validation_score},
        }
        for condition, validation_score, test_score in (
            ("vamp_core_tsv_r8_scale03_repair000", 40.0, 90.0),
            ("vamp_core_tsv_r8_scale05_repair000", 50.0, 10.0),
        )
        for _task_index in range(1, 9)
    )

    rows = _validation_policy_rows(records)

    assert next(row for row in rows if row["selected"])["condition"].endswith(
        "scale05_repair000"
    )


def test_interim_report_is_self_contained_and_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = tmp_path / "runs" / record_sha256({"run": "report-test"})
    for name in (
        "baselines",
        "derived",
        "evaluations",
        "leaves",
        "logs",
        "manifests",
        "merge_cache",
        "reports",
        "state/sessions",
    ):
        (run / name).mkdir(parents=True, exist_ok=True)
    arrivals = tuple(record_sha256({"arrival": index}) for index in range(1, 41))
    publish_immutable_json(
        run / "manifests" / "arrivals.json",
        {"arrival_ids": list(arrivals), "format": "trace-arrivals-v1"},
    )
    JobLedger(run / "manifests" / "jobs.jsonl").register(
        (JobSpec.create("unfinished", "gpu", 1, (), {"value": 1}),)
    )
    monkeypatch.setattr(
        "apm.continual.trace.reporting._score_parquet",
        lambda _rows: b"PAR1trace-testPAR1",
    )
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))

    first = build_report(run, interim=True, name="INTERIM_TEST")
    first_bytes = {
        path.name: path.read_bytes()
        for path in (
            first.markdown_path,
            first.html_path,
            first.csv_path,
            first.parquet_path,
            first.calibration_csv_path,
            first.lineage_svg_path,
            first.merge_plot_path,
            first.merge_diagnostics_csv_path,
        )
    }
    second = build_report(run, interim=True, name="INTERIM_TEST")

    assert "PRELIMINARY — RUN PAUSED BEFORE COMPLETION" in first.markdown_path.read_text()
    assert "data:image/png;base64" in first.html_path.read_text()
    assert "<svg" in first.html_path.read_text()
    assert first.parquet_path.read_bytes().startswith(b"PAR1")
    assert first_bytes == {
        path.name: path.read_bytes()
        for path in (
            second.markdown_path,
            second.html_path,
            second.csv_path,
            second.parquet_path,
            second.calibration_csv_path,
            second.lineage_svg_path,
            second.merge_plot_path,
            second.merge_diagnostics_csv_path,
        )
    }

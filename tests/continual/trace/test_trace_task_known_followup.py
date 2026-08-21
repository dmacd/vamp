from __future__ import annotations

from pathlib import Path

import pytest

from apm.continual.artifacts import canonical_json_bytes, file_sha256, record_sha256
from apm.continual.trace.data import TraceExample
from apm.continual.trace.task_known import build_task_known_routes
from apm.continual.trace.task_known_followup import (
    CandidateIndexEntry,
    _load_candidate_outputs,
    build_task_known_followup,
)


def _arrival_ids() -> tuple[str, ...]:
    return tuple(record_sha256({"arrival": index}) for index in range(1, 41))


def test_task_known_routes_apply_all_three_predeclared_ranking_terms() -> None:
    routes = {
        (route.stage, route.task): route
        for route in build_task_known_routes(_arrival_ids())
    }

    assert routes[(2, "C-STANCE")].interval == "1–4"
    assert routes[(2, "C-STANCE")].coverage_count == 4
    assert routes[(3, "MeetingBank")].interval == "13–14"
    assert routes[(3, "MeetingBank")].purity == 1.0
    assert routes[(1, "C-STANCE")].interval == "3–4"


def test_task_known_routes_match_the_complete_registered_interval_audit() -> None:
    routes = build_task_known_routes(_arrival_ids())
    observed = tuple(
        (route.stage, route.task, route.interval, route.coverage_count, route.purity)
        for route in routes
    )
    expected = (
        (1, "C-STANCE", "3–4", 2, 1.0),
        (2, "C-STANCE", "1–4", 4, 1.0),
        (2, "FOMC", "7–8", 2, 1.0),
        (3, "C-STANCE", "1–8", 5, 0.625),
        (3, "FOMC", "1–8", 3, 0.375),
        (3, "MeetingBank", "13–14", 2, 1.0),
        (4, "C-STANCE", "1–8", 5, 0.625),
        (4, "FOMC", "1–8", 3, 0.375),
        (4, "MeetingBank", "13–16", 3, 0.75),
        (4, "Py150", "17–18", 2, 1.0),
        (5, "C-STANCE", "1–8", 5, 0.625),
        (5, "FOMC", "1–8", 3, 0.375),
        (5, "MeetingBank", "9–16", 5, 0.625),
        (5, "Py150", "17–20", 4, 1.0),
        (5, "ScienceQA", "23–24", 2, 1.0),
        (6, "C-STANCE", "1–8", 5, 0.625),
        (6, "FOMC", "1–8", 3, 0.375),
        (6, "MeetingBank", "9–16", 5, 0.625),
        (6, "Py150", "17–20", 4, 1.0),
        (6, "ScienceQA", "21–24", 4, 1.0),
        (6, "NumGLUE-cm", "27–28", 2, 1.0),
        (7, "C-STANCE", "1–16", 5, 0.3125),
        (7, "FOMC", "1–16", 5, 0.3125),
        (7, "MeetingBank", "1–16", 5, 0.3125),
        (7, "Py150", "17–24", 4, 0.5),
        (7, "ScienceQA", "17–24", 4, 0.5),
        (7, "NumGLUE-cm", "25–28", 3, 0.75),
        (7, "NumGLUE-ds", "33–34", 2, 1.0),
        (8, "C-STANCE", "1–16", 5, 0.3125),
        (8, "FOMC", "1–16", 5, 0.3125),
        (8, "MeetingBank", "1–16", 5, 0.3125),
        (8, "Py150", "17–24", 4, 0.5),
        (8, "ScienceQA", "17–24", 4, 0.5),
        (8, "NumGLUE-cm", "25–32", 5, 0.625),
        (8, "NumGLUE-ds", "33–36", 3, 0.75),
        (8, "20Minuten", "37–38", 2, 1.0),
    )

    assert observed == expected


def test_candidate_loader_requires_the_exact_candidate_example_product(
    tmp_path: Path,
) -> None:
    examples = tuple(
        TraceExample(
            example_id=record_sha256({"example": index}),
            task="C-STANCE",
            split="test",
            source_index=index,
            prompt=f"prompt {index}",
            answer="A",
        )
        for index in range(2)
    )
    candidate_order = ("base", "node")
    rows = tuple(
        {
            "candidate_id": candidate_id,
            "example_id": example.example_id,
            "format": "trace-candidate-evaluation-v1",
            "prediction": "A",
            "prompt_nll": 1.0,
            "split": "test",
            "stage": 1,
            "task": "C-STANCE",
        }
        for candidate_id in candidate_order
        for example in examples
    )
    path = tmp_path / "candidates.jsonl"
    path.write_bytes(b"".join(canonical_json_bytes(row) for row in rows))
    entry = CandidateIndexEntry(
        relative_path="candidates.jsonl",
        condition="vamp_svd_r8_repair000",
        policy_hash=record_sha256({"policy": 1}),
        stage=1,
        task="C-STANCE",
        split="test",
        rows=4,
        bytes=path.stat().st_size,
        sha256=file_sha256(path),
    )

    assert len(_load_candidate_outputs(path, entry, examples, candidate_order)) == 4
    path.write_bytes(b"".join(canonical_json_bytes(row) for row in rows[:-1]))
    with pytest.raises(ValueError, match="exact deterministic"):
        _load_candidate_outputs(path, entry, examples, candidate_order)


@pytest.mark.integration
def test_sealed_bundle_followup_is_complete_and_byte_deterministic(
    tmp_path: Path,
) -> None:
    bundle = (
        Path(__file__).resolve().parents[3]
        / "docs"
        / "experiments"
        / "trace-logt-vamp"
    )
    first = build_task_known_followup(bundle, tmp_path / "first")
    second = build_task_known_followup(bundle, tmp_path / "second")

    assert len(first.summaries) == 12
    assert {
        path.relative_to(first.output_root).as_posix(): file_sha256(path)
        for path in first.output_root.iterdir()
    } == {
        path.relative_to(second.output_root).as_posix(): file_sha256(path)
        for path in second.output_root.iterdir()
    }
    assert "data:image/png;base64" in first.report_html.read_text(encoding="utf-8")

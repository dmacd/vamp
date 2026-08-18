from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from apm.data.text.tinyworlds_nouns_v2.contracts import TASK_IDS
from apm.data.text.tinyworlds_nouns_v2.temporal_full_story_routing import (
    AUDIT_MARGIN_THRESHOLD,
    SOURCE_SPECIFICATIONS,
    _bootstrap_differences,
    reconstructed_whole_story_scores,
    select_audit_story_ids,
    stable_minimum,
)
from apm.data.text.tinyworlds_nouns_v2.temporal_full_story_routing_report import (
    _accessible_svg,
    _html_report,
    _markdown_report,
)


def _source_row(task_id: str, story_id: str, margin: float, prefix_tokens: int = 10):
    return {
        "prefix_scores": [1.0, 1.0 + margin],
        "prefix_token_count": prefix_tokens,
        "story_id": story_id,
        "suffix_mean_nll_by_candidate": [1.0, 1.0 + margin],
        "suffix_token_count": 10,
        "task_id": task_id,
    }


def _case_row(task_id: str, offset: float) -> dict[str, object]:
    return {
        "full_route_hit": offset > 0,
        "full_suffix_mean_nll": 1.0 - offset,
        "full_whole_mean_nll": 1.1 - offset,
        "midpoint_route_hit": False,
        "midpoint_suffix_mean_nll": 1.0,
        "midpoint_whole_mean_nll": 1.1,
        "task_id": task_id,
    }


def _analysis() -> dict[str, object]:
    aggregates = [
        {
            "accuracy_kind": "noun_support" if index < 2 else "exact_noun",
            "condition": specification.condition,
            "full_route_accuracy": 0.8,
            "full_suffix_story_nll": 1.5,
            "label": specification.label,
            "midpoint_route_accuracy": 0.7,
            "midpoint_suffix_story_nll": 1.6,
            "oracle_suffix_story_nll": 1.45,
            "route_accuracy_change_pp": 10.0,
            "suffix_gap_recovered_fraction": 2.0 / 3.0,
            "suffix_story_nll_change": -0.1,
        }
        for index, specification in enumerate(SOURCE_SPECIFICATIONS)
    ]
    return {
        "aggregate": aggregates,
        "audit": {
            "audited_condition_story_rows": 570,
            "audited_story_count": 190,
            "long_story_count": 111,
            "maximum_short_score_absolute_error": 1e-6,
            "minimum_unaudited_margin": 0.001,
            "selection_mismatches": 0,
        },
        "bootstrap": [
            {
                "condition": "blocked_log_t",
                "estimate": -0.1,
                "lower_95": -0.11,
                "metric": "suffix_story_nll_change",
                "upper_95": -0.09,
            }
        ],
    }


def test_weighted_reconstruction_uses_every_transition_and_stable_first_ties() -> None:
    row = {
        "prefix_scores": [1.0, 2.0, 2.0],
        "prefix_token_count": 3,
        "suffix_mean_nll_by_candidate": [4.0, 2.0, 2.0],
        "suffix_token_count": 1,
    }
    assert reconstructed_whole_story_scores(row) == (1.75, 2.0, 2.0)
    assert stable_minimum((2.0, 1.0, 1.0)) == (1, 0.0)
    with pytest.raises(ValueError, match="finite"):
        stable_minimum((1.0, np.nan))


def test_audit_selection_includes_long_near_tie_and_each_noun_minimum() -> None:
    rows = tuple(
        row
        for task_index, task_id in enumerate(TASK_IDS)
        for row in (
            _source_row(task_id, f"{task_index:064x}", 0.01),
            _source_row(task_id, f"{task_index + 100:064x}", 0.02),
        )
    )
    rows = tuple(
        {
            **row,
            "prefix_tokens": 257,
            "prefix_token_count": 257,
        }
        if index == 1
        else row
        for index, row in enumerate(rows)
    )
    rows = tuple(
        {
            **row,
            "prefix_scores": [1.0, 1.0 + AUDIT_MARGIN_THRESHOLD],
            "suffix_mean_nll_by_candidate": [1.0, 1.0 + AUDIT_MARGIN_THRESHOLD],
        }
        if index == 3
        else row
        for index, row in enumerate(rows)
    )
    selected = select_audit_story_ids(
        {specification.condition: rows for specification in SOURCE_SPECIFICATIONS}
    )
    assert len(selected) >= len(TASK_IDS)
    assert rows[1]["story_id"] in selected
    assert rows[3]["story_id"] in selected
    assert all(
        any(row["task_id"] == task_id and row["story_id"] in selected for row in rows)
        for task_id in TASK_IDS
    )


def test_stratified_bootstrap_is_seed_zero_deterministic() -> None:
    rows = tuple(
        _case_row(task_id, 0.01 + task_index / 10_000)
        for task_index, task_id in enumerate(TASK_IDS)
        for _ in range(task_index + 1)
    )
    by_condition = {
        specification.condition: rows for specification in SOURCE_SPECIFICATIONS
    }
    first = _bootstrap_differences(by_condition)
    second = _bootstrap_differences(by_condition)
    assert first == second
    assert len(first) == 9
    assert all(row["repetitions"] == 10_000 and row["seed"] == 0 for row in first)
    assert all(row["lower_95"] <= row["estimate"] <= row["upper_95"] for row in first)


def test_reports_are_standalone_accessible_and_explicit_about_selection_leakage() -> None:
    inputs = SimpleNamespace(
        contract_sha256="a" * 64,
        parent=SimpleNamespace(contract_sha256="b" * 64),
    )
    analysis = _analysis()
    markdown = _markdown_report(
        inputs,
        analysis,
        {"end_to_end_seconds": 10.0},
        {"peak_bytes_in_use": 1024},
    )
    svg = _accessible_svg("<svg><path/></svg>", "Title", "Description")
    html = _html_report(
        inputs,
        analysis,
        {"end_to_end_seconds": 10.0},
        {"peak_bytes_in_use": 1024},
        svg,
    )
    assert "selection-leaking" in markdown
    assert markdown.count("<details>") >= 4
    assert "self-selected" in markdown
    assert "aria-labelledby" in svg and "<desc" in svg
    assert html.startswith("<!doctype html>")
    assert "<details" in html and "http://" not in html and "https://" not in html
    assert "Diagnostic-only selection leakage" in html


def test_default_runner_has_no_options_and_fixes_gpu_zero_and_allocator() -> None:
    runner = (
        Path(__file__).resolve().parents[1]
        / "scripts/run_tinyworlds_nouns_v2_full_story_routing.py"
    )
    source = runner.read_text(encoding="utf-8")
    module = ast.parse(source)
    main = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    assert not main.args.args
    assert "argparse" not in source
    assert 'os.environ["CUDA_VISIBLE_DEVICES"] = "0"' in source
    assert 'os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "cuda_async"' in source
    assert "Persistent temporary directory:" in source
    assert "ALLOCATOR_LIMIT_BYTES" in source

from __future__ import annotations

import ast
from pathlib import Path
import runpy
import shutil
from types import SimpleNamespace

from apm.data.text.tinyworlds_nouns_v2.addressing_study_contracts import KEY_SCHEMES
from apm.data.text.tinyworlds_nouns_v2.addressing_study_report import (
    publish_inclusion_graph,
    render_cumulative_cost_svg,
    render_html_report,
    render_markdown_report,
    render_quality_latency_svg,
)
from apm.memory.graph import NodeId, TaskId, add_memory_node, init_memory_graph


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPOSITORY_ROOT / "scripts/run_tinyworlds_nouns_v2_addressing_study.py"


def _method(scheme: str, mode: str, width: int) -> dict[str, object]:
    return {
        "candidate_width": width,
        "mean_active_lora_edge_evaluations": 120.0,
        "mean_final_entropy": 0.7,
        "mean_final_margin": 0.2,
        "mean_gathered_edge_count": 5.0 if mode == "compact" else 24.0,
        "mean_hopfield_dot_products": 25.0,
        "mean_model_forward_equivalent_prefix_tokens": 92.0,
        "mean_oracle_regret": 0.1,
        "mean_physical_edge_capacity": 8.0 if mode == "compact" else 24.0,
        "mean_selected_path_edge_count": 2.0,
        "mode": mode,
        "route_accuracy": 0.75,
        "row_count": 4_440,
        "scheme": scheme,
        "story_mean_nll": 2.0 + 0.001 * width,
        "token_mean_nll": 1.9,
        "true_node_recall_at_4": 0.7,
        "true_node_recall_at_8": 0.8,
        "warm_gpu_latency_seconds": 0.03,
        "warm_throughput_examples_per_second": 266.0,
    }


def _analysis() -> dict[str, object]:
    canonical = (
        _method("canonical_full_centroid", "dense_all", 25),
        _method("canonical_full_centroid", "compact", 4),
        _method("canonical_full_centroid", "compact", 8),
    )
    top_8 = tuple(
        _method(scheme, "compact", 8)
        for scheme in KEY_SCHEMES
        if scheme != "canonical_full_centroid"
    )
    retrieval = tuple(
        {
            "mean_entropy": 1.0,
            "mean_margin": 0.1,
            "row_count": 4_440,
            "scheme": scheme,
            "top_1_recall": 0.5,
            "top_4_recall": 0.7,
            "top_8_recall": 0.8,
        }
        for scheme in KEY_SCHEMES
    )
    intervals = tuple(
        {
            "difference": 0.0,
            "lower_95": -0.01,
            "metric": metric,
            "reference": "canonical_full_centroid",
            "scheme": scheme,
            "upper_95": 0.01,
        }
        for metric in ("top_8_recall", "compact_top_8_story_nll")
        for scheme in KEY_SCHEMES
    )
    per_task = tuple(
        {
            "candidate_width": 8,
            "confusion_counts": {str(index): int(index == 1) for index in range(25)},
            "mode": "compact",
            "route_accuracy": 0.75,
            "scheme": scheme,
            "story_count": 1,
            "story_mean_nll": 2.0,
            "task": "pear",
            "top_1_recall": 0.5,
            "top_4_recall": 0.7,
            "top_8_recall": 0.8,
        }
        for scheme in KEY_SCHEMES
    )
    timing = (
        {
            "candidate_width": 8,
            "cold_compile_seconds": 1.0,
            "mode": "compact",
            "physical_edge_capacity": 8,
            "prefix_width_bucket": 32,
            "warm_kernel_mean_seconds": 0.03,
            "warm_throughput_examples_per_second": 266.0,
        },
    )
    return {
        "ebt_aggregate": (*canonical, *top_8),
        "ebt_bootstrap": intervals[5:],
        "experiment_1": canonical,
        "noninferiority": {
            "accuracy_loss": 0.0,
            "accuracy_margin": 0.02,
            "accuracy_pass": True,
            "story_nll_increase": -0.017,
            "story_nll_margin": 0.02,
            "story_nll_pass": True,
        },
        "per_task": per_task,
        "provenance": {
            "addressing_key_artifact_sha256": "f" * 64,
            "base_parameter_checksum": "a" * 64,
            "base_training_sha256": "b" * 64,
            "canonical_artifact_hashes": {},
            "canonical_ledger_hashes": {},
            "canonical_run_sha256": "c" * 64,
            "ebt_contract_sha256": "1" * 64,
            "partition_sha256": "d" * 64,
            "retrieval_contract_sha256": "2" * 64,
            "vamp_tensor_checksum": "e" * 64,
        },
        "retrieval_aggregate": retrieval,
        "retrieval_bootstrap": intervals[:5],
        "runtimes": {"end_to_end": 10.0},
        "stage_costs": (),
        "timing": timing,
    }


def test_reports_are_separate_self_contained_collapsible_and_accessible() -> None:
    analysis = _analysis()
    parity = {"maximum_absolute_differences": {"top_8_soft_nll": 1e-6}, "tolerance": 2e-4}
    allocator = {"allocator_limit_bytes": 12 * 2**30, "peak_bytes_in_use": 2**30}
    runtimes = {"end_to_end": 10.0}
    cost_rows = tuple(
        {
            "active_lora_edge_evaluations": stage * method_index,
            "hopfield_dot_products": stage * method_index,
            "method": method,
            "model_forward_equivalent_prefix_tokens": stage * method_index,
            "stage": stage,
        }
        for stage in range(1, 25)
        for method_index, method in enumerate(
            (
                "canonical_exhaustive",
                "canonical_hopfield",
                "canonical_ebt_uniform",
                "canonical_ebt_hopfield",
            ),
            start=1,
        )
    )
    cost_svg = render_cumulative_cost_svg(cost_rows)
    quality_svg = render_quality_latency_svg(analysis["experiment_1"])
    assert cost_svg == render_cumulative_cost_svg(cost_rows)
    assert quality_svg == render_quality_latency_svg(analysis["experiment_1"])
    for svg in (cost_svg, quality_svg):
        assert 'role="img"' in svg
        assert "<title id=" in svg
        assert "<desc id=" in svg
    embedded = {
        "cumulative-addressing-cost.svg": cost_svg,
        "final-checkpoint-quality-latency.svg": quality_svg,
        "vamp-graph-top4.svg": '<svg role="img"><title>top 4</title></svg>',
        "vamp-graph-top8.svg": '<svg role="img"><title>top 8</title></svg>',
    }

    markdown = render_markdown_report(analysis, parity, allocator, runtimes)
    html = render_html_report(analysis, parity, allocator, runtimes, embedded)

    assert markdown.startswith("# TinyWorlds Nouns-v2 bounded addressing study")
    assert markdown.count("<details>") >= 5
    assert "softmax(logits_t)" in markdown
    assert "all 36 registered probes for every node, including the root" in markdown
    assert "20 Adam steps" in markdown
    assert "true-node recall@4" in markdown
    assert "active path edges" in markdown
    assert html.startswith("<!doctype html>")
    assert html.count("<details>") >= 5
    assert "<style>" in html and "<script" not in html
    assert "<link " not in html and "<script src=" not in html
    assert 'src="http://' not in html and 'src="https://' not in html
    assert "True-node recall@8" in html
    assert "Active path edges" in html
    assert "Every EBT run uses 20 Adam steps" in html
    assert cost_svg in html and quality_svg in html


def test_graphviz_outputs_cover_every_node_and_edge(tmp_path: Path) -> None:
    if shutil.which("dot") is None:
        return
    graph = init_memory_graph(NodeId("root"))
    for index in range(1, 25):
        graph = add_memory_node(
            graph,
            NodeId(f"node-{index}"),
            NodeId("root"),
            TaskId(f"task-{index}"),
            index,
            f"edge-{index}",
        )
    inputs = SimpleNamespace(adaptation=SimpleNamespace(vamp_graph=graph))
    retrieval_rows = (
        {"top_8_indices": list(range(8))},
        {"top_8_indices": list(range(8, 0, -1))},
    )

    for width in (4, 8):
        dot_path, svg_path = publish_inclusion_graph(
            inputs,
            retrieval_rows,
            width,
            tmp_path,
        )
        dot = dot_path.read_text(encoding="utf-8")
        svg = svg_path.read_text(encoding="utf-8")
        assert sum(" -> " in line for line in dot.splitlines()) == 24
        assert all(f'"node-{index}"' in dot for index in range(1, 25))
        assert "\\ncandidate" in dot and "\\\\ncandidate" not in dot
        assert 'role="img"' in svg and "<desc id=" in svg


def test_default_runner_has_one_fixed_load_only_gpu_zero_workflow() -> None:
    namespace = runpy.run_path(str(RUNNER), run_name="addressing_runner_surface")
    syntax = ast.parse(RUNNER.read_text(encoding="utf-8"))
    calls = tuple(
        node.func.id
        for node in ast.walk(syntax)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    )
    main = next(
        node
        for node in syntax.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )

    assert len(namespace["PHASES"]) == 6
    assert namespace["os"].environ["CUDA_VISIBLE_DEVICES"] == "0"
    assert not main.args.args
    assert "argparse" not in RUNNER.read_text(encoding="utf-8")
    assert "run_or_resume_nouns_v2_base" not in RUNNER.read_text(encoding="utf-8")
    assert "authenticate_addressing_study_inputs" in calls
    assert "run_or_resume_addressing_evaluation" in calls
    assert calls.count("publish_addressing_study_report") == 2
    assert "notify-send" in RUNNER.read_text(encoding="utf-8")

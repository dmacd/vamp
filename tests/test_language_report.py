from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apm.continual.language_benchmarks import (
    GeneratedLanguageSample,
    ROUTER_BASELINE_NAMES,
    STORED_BASELINE_NAMES,
)
from apm.continual.language_report import (
    AddressConfusion,
    LanguageReportBundle,
    LanguageReportManifest,
    ReportRecord,
    canonical_config_json,
    language_report_directory,
    write_language_report,
)
from apm.memory.graph import NodeId, TaskId, add_memory_node, init_memory_graph
from apm.memory.visualization import EdgeVisualStats, NodeVisualStats


def _record(**values: str | int | float | bool | None) -> ReportRecord:
    return ReportRecord(tuple(values.items()))


def _bundle() -> LanguageReportBundle:
    graph = add_memory_node(
        init_memory_graph(NodeId("root")),
        NodeId("task_0"),
        NodeId("root"),
        TaskId("task_0"),
        1,
        object(),
    )
    stored = tuple(
        _record(stage=1, baseline=name, task_id="task_0", suffix_nll=1.0)
        for name in STORED_BASELINE_NAMES
    )
    routing = tuple(
        _record(
            stage=1,
            router=name,
            task_id="task_0",
            prefix_length=32,
            routed_suffix_nll=1.0,
            negative_control_chance_accuracy=(
                1.0 if name == "deterministic_random_node" else None
            ),
            negative_control_ci95_lower=(
                0.75 if name == "deterministic_random_node" else None
            ),
            negative_control_ci95_upper=(
                1.0 if name == "deterministic_random_node" else None
            ),
            negative_control_chance_in_ci95=(
                True if name == "deterministic_random_node" else None
            ),
            leakage_audit_required=(
                False if name == "deterministic_random_node" else None
            ),
        )
        for name in ROUTER_BASELINE_NAMES
    )
    return LanguageReportBundle(
        manifest=LanguageReportManifest(
            dataset="tinyshakespeare",
            curriculum="character-permutation",
            preset="bounded",
            seed=0,
            config_json=canonical_config_json(
                {"lora": {"alpha": 1, "rank": 1}, "steps": 2}
            ),
        ),
        stage_metrics=(_record(stage=1, task_id="task_0", parent="root"),),
        stored_competence=stored,
        routing_metrics=routing,
        transfer_metrics=(_record(stage=1, task_id="task_0", transfer=0.5),),
        memory_metrics=(
            _record(stage=1, persistent_bytes=100, runtime_bytes=200),
        ),
        addressing_cost=tuple(
            _record(
                stage=1,
                router=name,
                cold_seconds=0.2,
                warm_seconds=0.01,
            )
            for name in ROUTER_BASELINE_NAMES
        ),
        competence_curve=(
            _record(
                stage=0,
                frozen_base=2.0,
                sequential_single_lora=2.0,
                independent_root_lora=2.0,
                vamp_oracle=2.0,
            ),
            _record(
                stage=1,
                frozen_base=2.0,
                sequential_single_lora=1.4,
                independent_root_lora=1.2,
                vamp_oracle=1.1,
            ),
        ),
        routing_curve=(
            _record(stage=0, exhaustive_accuracy=1.0, hopfield_accuracy=1.0),
            _record(stage=1, exhaustive_accuracy=1.0, hopfield_accuracy=0.8),
        ),
        memory_curve=(
            _record(stage=0, persistent_bytes=80, runtime_bytes=120),
            _record(stage=1, persistent_bytes=100, runtime_bytes=200),
        ),
        address_confusion=AddressConfusion(
            ("root", "task_0"),
            np.asarray(((1, 0), (0, 2)), dtype=np.int32),
        ),
        graph=graph,
        node_stats=(
            NodeVisualStats("root", "root", 0, 80, (), 0.0),
            NodeVisualStats("task_0", "task_0", 1, 20, ("task_0",), 1.0),
        ),
        edge_stats=(
            EdgeVisualStats("root", "task_0", "task_0", 0.5, 20, 0.4),
        ),
        samples=tuple(
            GeneratedLanguageSample(name, "task_0", "Once", " upon a time")
            for name in (*STORED_BASELINE_NAMES, *ROUTER_BASELINE_NAMES)
        ),
    )


def test_language_report_writes_complete_idempotent_artifact_set(tmp_path) -> None:
    bundle = _bundle()
    output_directory = write_language_report(tmp_path, bundle)
    expected_files = {
        "manifest.json",
        "stage_metrics.jsonl",
        "stored_competence.jsonl",
        "routing_metrics.jsonl",
        "transfer_metrics.jsonl",
        "memory_metrics.jsonl",
        "addressing_cost.jsonl",
        "address_confusion.svg",
        "competence_curves.svg",
        "routing_curves.svg",
        "memory_curves.svg",
        "graph.svg",
        "report.html",
        "samples.md",
    }

    assert {path.name for path in output_directory.iterdir()} == expected_files
    first_contents = {
        path.name: path.read_bytes() for path in output_directory.iterdir()
    }
    assert write_language_report(tmp_path, bundle) == output_directory
    second_contents = {
        path.name: path.read_bytes() for path in output_directory.iterdir()
    }
    assert second_contents == first_contents
    assert output_directory == language_report_directory(tmp_path, bundle.manifest)
    assert output_directory.parts[-3:] == (
        "tinyshakespeare",
        "character-permutation",
        bundle.manifest.run_id,
    )
    assert bundle.manifest.run_id.startswith("bounded-seed0-")
    assert len(bundle.manifest.run_id.rsplit("-", 1)[1]) == 12

    html_text = (output_directory / "report.html").read_text(encoding="utf-8")
    samples_text = (output_directory / "samples.md").read_text(encoding="utf-8")
    assert "graph.svg" in html_text
    assert "samples.md" in html_text
    assert "Once" in html_text and "upon a time" in samples_text
    assert "report-lightbox" in html_text
    assert "95% Wilson" in html_text
    assert "leakage audit" in html_text
    assert "reconstruction" not in html_text.casefold()
    assert "mnist" not in html_text.casefold()
    for method_name in (*STORED_BASELINE_NAMES, *ROUTER_BASELINE_NAMES):
        assert method_name in html_text
        assert method_name in samples_text


def test_language_report_rejects_missing_method_coverage_and_unsafe_identity() -> None:
    bundle = _bundle()
    with pytest.raises(ValueError, match="coverage omits"):
        LanguageReportBundle(
            **{
                **bundle.__dict__,
                "routing_metrics": bundle.routing_metrics[:-1],
            }
        )
    with pytest.raises(ValueError, match="lowercase"):
        LanguageReportManifest(
            dataset="../TinyStories",
            curriculum="topic",
            preset="bounded",
            seed=0,
            config_json=canonical_config_json({"steps": 1}),
        )


def test_report_records_and_config_are_strictly_immutable_inputs() -> None:
    counts = np.eye(2, dtype=np.int32)
    confusion = AddressConfusion(("a", "b"), counts)
    counts[0, 0] = 99
    np.testing.assert_array_equal(confusion.counts, np.eye(2, dtype=np.int64))
    assert not confusion.counts.flags.writeable
    with pytest.raises(ValueError, match="canonical"):
        LanguageReportManifest(
            dataset="tinystories",
            curriculum="topic",
            preset="bounded",
            seed=0,
            config_json='{"z": 1, "a": 2}',
        )
    with pytest.raises(TypeError, match="finite JSON scalars"):
        ReportRecord((("loss", float("nan")),))

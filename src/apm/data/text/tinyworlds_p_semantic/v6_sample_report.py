"""Validation-only semantic-v6 sample report publication and loading."""

from __future__ import annotations

import os
from pathlib import Path

from apm.data.text.tinyworlds_p_semantic.contracts import WORLD_LABELS, record_sha256
from apm.data.text.tinyworlds_p_semantic.sample_report import (
    SampleReportError,
    _file_sha256,
    _fsync_directory,
    _html,
    _load_json,
    _markdown,
    _materialize_sample,
    _sample_selections,
    _selected_assignments,
    _text,
    _write_json,
    _write_text,
)
from apm.data.text.tinyworlds_p_semantic.v6_partition_contracts import (
    V6_PARTITION_SCHEMA_VERSION,
    V6_SAMPLE_REPORT_FORMAT,
    V6_SAMPLE_REPORT_TREE_FORMAT,
    V6SemanticPartitionArtifact,
    V6SemanticSampleReport,
)


def publish_v6_sample_report(
    artifact: V6SemanticPartitionArtifact,
    output_root: str | Path,
    temporary_directory: str | Path,
) -> V6SemanticSampleReport:
    """Publish held-in, five-world, and ten v6 comparison examples."""
    if type(artifact) is not V6SemanticPartitionArtifact:
        raise TypeError("semantic-v6 report requires its strict partition")
    selections = _sample_selections(artifact)
    assignment_by_group = _selected_assignments(
        artifact.root / "assignments.jsonl",
        {item["normalized_story_sha256"] for item in selections},
    )
    samples = tuple(
        _materialize_sample(artifact, selection, assignment_by_group)
        for selection in selections
    )
    clusters = [
        {
            "index": cluster.index,
            "role": cluster.role,
            "token_mass": cluster.token_mass,
            "words": list(cluster.words),
        }
        for cluster in artifact.semantic_catalog.clusters
    ]
    content = {
        "catalog_sha256": artifact.semantic_catalog.catalog_sha256,
        "clusters": clusters,
        "format": V6_SAMPLE_REPORT_FORMAT,
        "partition_sha256": artifact.partition_sha256,
        "samples": list(samples),
        "schema_version": V6_PARTITION_SCHEMA_VERSION,
        "sealed_test_opened": False,
    }
    report_sha256 = record_sha256(content)
    working = Path(temporary_directory)
    if working.exists() and any(working.iterdir()):
        raise FileExistsError(f"semantic-v6 report work directory is not empty: {working}")
    working.mkdir(parents=True, exist_ok=True)
    _write_json(
        working / "sample-report.json",
        {**content, "report_sha256": report_sha256},
    )
    _write_text(
        working / "sample-report.md",
        _markdown(report_sha256, content, "v6"),
    )
    _write_text(
        working / "sample-report.html",
        _html(report_sha256, content, "v6"),
    )
    _write_tree(working, report_sha256)
    partition_root = Path(output_root) / artifact.partition_sha256
    partition_root.mkdir(parents=True, exist_ok=True)
    target = partition_root / report_sha256
    if target.exists():
        raise FileExistsError(f"semantic-v6 sample report already exists: {target}")
    os.rename(working, target)
    _fsync_directory(partition_root)
    return load_v6_sample_report(target)


def load_v6_sample_report(path: str | Path) -> V6SemanticSampleReport:
    """Strictly authenticate a complete v6 validation-only report."""
    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        raise SampleReportError("semantic-v6 report must be a regular directory")
    tree = _load_json(root / "tree.json")
    if (
        set(tree) != {"files", "format", "report_sha256", "schema_version"}
        or tree["format"] != V6_SAMPLE_REPORT_TREE_FORMAT
        or tree["schema_version"] != V6_PARTITION_SCHEMA_VERSION
    ):
        raise SampleReportError("semantic-v6 report tree changed")
    report_sha256 = _text(tree, "report_sha256")
    if root.name != report_sha256:
        raise SampleReportError("semantic-v6 report directory identity changed")
    raw_files = tree.get("files")
    if type(raw_files) is not list or any(type(item) is not dict for item in raw_files):
        raise SampleReportError("semantic-v6 report descriptors changed")
    paths = tuple(_text(item, "name") for item in raw_files)
    if paths != ("sample-report.html", "sample-report.json", "sample-report.md"):
        raise SampleReportError("semantic-v6 report files are incomplete")
    for descriptor in raw_files:
        candidate = root / _text(descriptor, "name")
        size = descriptor.get("size_bytes")
        if (
            type(size) is not int
            or size < 0
            or candidate.stat().st_size != size
            or _file_sha256(candidate) != _text(descriptor, "sha256")
        ):
            raise SampleReportError("semantic-v6 report file changed")
    record = _load_json(root / "sample-report.json")
    if record.get("report_sha256") != report_sha256:
        raise SampleReportError("semantic-v6 report identity changed")
    content = {key: value for key, value in record.items() if key != "report_sha256"}
    if (
        record_sha256(content) != report_sha256
        or record.get("format") != V6_SAMPLE_REPORT_FORMAT
        or record.get("schema_version") != V6_PARTITION_SCHEMA_VERSION
    ):
        raise SampleReportError("semantic-v6 report content identity changed")
    if record.get("sealed_test_opened") is not False:
        raise SampleReportError("semantic-v6 report opened sealed test")
    samples = record.get("samples")
    if type(samples) is not list or len(samples) != 16:
        raise SampleReportError("semantic-v6 report condition coverage changed")
    conditions = tuple(
        _text(item, "condition") for item in samples if type(item) is dict
    )
    expected = (
        "base/validation",
        *(f"world/{world}/validation" for world in WORLD_LABELS),
        *(
            f"control/{world}/validation/{arm}"
            for world in WORLD_LABELS
            for arm in ("row", "column")
        ),
    )
    if conditions != expected:
        raise SampleReportError("semantic-v6 report conditions changed")
    return V6SemanticSampleReport(
        root=root.resolve(),
        report_sha256=report_sha256,
        partition_sha256=_text(record, "partition_sha256"),
        catalog_sha256=_text(record, "catalog_sha256"),
        sample_count=len(samples),
    )


def _write_tree(root: Path, report_sha256: str) -> None:
    files = tuple(
        {
            "name": name,
            "sha256": _file_sha256(root / name),
            "size_bytes": (root / name).stat().st_size,
        }
        for name in ("sample-report.html", "sample-report.json", "sample-report.md")
    )
    _write_json(
        root / "tree.json",
        {
            "files": list(files),
            "format": V6_SAMPLE_REPORT_TREE_FORMAT,
            "report_sha256": report_sha256,
            "schema_version": V6_PARTITION_SCHEMA_VERSION,
        },
    )


__all__ = ["load_v6_sample_report", "publish_v6_sample_report"]

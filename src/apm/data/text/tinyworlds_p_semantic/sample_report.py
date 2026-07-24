"""Pre-training semantic validation sample reports with exact archive provenance."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from html import escape
import json
import os
from pathlib import Path

from apm.data.text.tinyworlds_p_semantic.contracts import (
    WORLD_LABELS,
    SemanticPartitionArtifact,
    canonical_json_bytes,
    record_sha256,
)


@dataclass(frozen=True, slots=True)
class SemanticSampleReport:
    """Authenticated pre-training samples bound to one partition and catalog."""

    root: Path
    report_sha256: str
    partition_sha256: str
    catalog_sha256: str
    sample_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        if type(self.sample_count) is not int or self.sample_count != 16:
            raise ValueError("semantic sample report must cover exactly 16 conditions")


class SampleReportError(ValueError):
    """A semantic sample report is incomplete, changed, or opens sealed test."""


def publish_sample_report(
    artifact: SemanticPartitionArtifact,
    output_root: str | Path,
    temporary_directory: str | Path,
) -> SemanticSampleReport:
    """Publish held-in, five-world, and ten control-arm validation examples."""
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
        "format": "tinyworlds-p-semantic-sample-report",
        "partition_sha256": artifact.partition_sha256,
        "samples": list(samples),
        "schema_version": 1,
        "sealed_test_opened": False,
    }
    report_sha = record_sha256(content)
    working = Path(temporary_directory)
    if working.exists() and any(working.iterdir()):
        raise FileExistsError(f"sample-report temporary directory is not empty: {working}")
    working.mkdir(parents=True, exist_ok=True)
    _write_json(working / "sample-report.json", {**content, "report_sha256": report_sha})
    _write_text(working / "sample-report.md", _markdown(report_sha, content))
    _write_text(working / "sample-report.html", _html(report_sha, content))
    _write_tree(working, report_sha)
    partition_root = Path(output_root) / artifact.partition_sha256
    partition_root.mkdir(parents=True, exist_ok=True)
    target = partition_root / report_sha
    if target.exists():
        raise FileExistsError(f"semantic sample report already exists: {target}")
    os.rename(working, target)
    _fsync_directory(partition_root)
    return load_sample_report(target)


def load_sample_report(path: str | Path) -> SemanticSampleReport:
    """Strictly authenticate a complete validation-only sample report."""
    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        raise SampleReportError("semantic sample report must be a regular directory")
    tree = _load_json(root / "tree.json")
    if (
        set(tree) != {"files", "format", "report_sha256", "schema_version"}
        or tree["format"] != "tinyworlds-p-semantic-sample-report-tree"
        or tree["schema_version"] != 1
    ):
        raise SampleReportError("semantic sample report tree contract changed")
    report_sha = _text(tree, "report_sha256")
    if root.name != report_sha:
        raise SampleReportError("semantic sample report directory identity changed")
    raw_files = tree.get("files")
    if type(raw_files) is not list or any(type(item) is not dict for item in raw_files):
        raise SampleReportError("semantic sample report file descriptors changed")
    paths = tuple(_text(item, "name") for item in raw_files)
    if paths != ("sample-report.html", "sample-report.json", "sample-report.md"):
        raise SampleReportError("semantic sample report files are incomplete")
    for descriptor in raw_files:
        candidate = root / _text(descriptor, "name")
        if (
            candidate.stat().st_size != _integer(descriptor, "size_bytes")
            or _file_sha256(candidate) != _text(descriptor, "sha256")
        ):
            raise SampleReportError("semantic sample report file changed")
    record = _load_json(root / "sample-report.json")
    if record.get("report_sha256") != report_sha:
        raise SampleReportError("semantic sample report identity changed")
    content = {key: value for key, value in record.items() if key != "report_sha256"}
    if record_sha256(content) != report_sha:
        raise SampleReportError("semantic sample report content identity changed")
    if record.get("sealed_test_opened") is not False:
        raise SampleReportError("semantic sample report crossed the sealed-test boundary")
    samples = record.get("samples")
    if type(samples) is not list or len(samples) != 16:
        raise SampleReportError("semantic sample report condition coverage changed")
    conditions = tuple(_text(item, "condition") for item in samples if type(item) is dict)
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
        raise SampleReportError("semantic sample report conditions changed")
    return SemanticSampleReport(
        root=root.resolve(),
        report_sha256=report_sha,
        partition_sha256=_text(record, "partition_sha256"),
        catalog_sha256=_text(record, "catalog_sha256"),
        sample_count=len(samples),
    )


def _sample_selections(artifact: SemanticPartitionArtifact) -> tuple[dict[str, object], ...]:
    held_group = min(
        _text(record, "normalized_story_sha256")
        for record in _iter_jsonl(artifact.root / "indexes" / "base-validation.jsonl")
    )
    selections: list[dict[str, object]] = [
        {
            "condition": "base/validation",
            "index": "base-validation.jsonl",
            "normalized_story_sha256": held_group,
        }
    ]
    for world in WORLD_LABELS:
        world_group = min(
            item.world_group_sha256
            for item in artifact.pairings
            if item.world == world and item.split == "validation"
        )
        selections.append(
            {
                "condition": f"world/{world}/validation",
                "index": f"world-{world}-validation.jsonl",
                "normalized_story_sha256": world_group,
            }
        )
    for world in WORLD_LABELS:
        for arm in ("row", "column"):
            control_group = min(
                item.control_group_sha256
                for item in artifact.pairings
                if item.world == world
                and item.split == "validation"
                and item.arm == arm
            )
            selections.append(
                {
                    "condition": f"control/{world}/validation/{arm}",
                    "index": f"control-{world}-validation.jsonl",
                    "normalized_story_sha256": control_group,
                }
            )
    return tuple(selections)


def _selected_assignments(
    path: Path,
    selected: set[str],
) -> dict[str, dict[str, object]]:
    result = {}
    for record in _iter_jsonl(path):
        digest = _text(record, "normalized_story_sha256")
        if digest in selected:
            if record.get("status") != "eligible":
                raise SampleReportError("sample selection refers to an excluded group")
            result[digest] = record
    if set(result) != selected:
        raise SampleReportError("sample assignments are incomplete")
    return result


def _materialize_sample(
    artifact: SemanticPartitionArtifact,
    selection: Mapping[str, object],
    assignments: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    group = _text(selection, "normalized_story_sha256")
    index_path = artifact.root / "indexes" / _text(selection, "index")
    matching = tuple(
        record
        for record in _iter_jsonl(index_path)
        if record.get("normalized_story_sha256") == group
    )
    if not matching:
        raise SampleReportError("sample group is absent from its validation index")
    record = min(matching, key=lambda item: _text(item, "record_id"))
    shard = artifact.root / "shards" / f"text-{_integer(record, 'text_shard'):06d}.bin"
    with shard.open("rb") as source:
        source.seek(_integer(record, "text_offset"))
        raw_story = source.read(_integer(record, "text_bytes"))
    story_sha = sha256(raw_story).hexdigest()
    if story_sha != _text(record, "story_sha256"):
        raise SampleReportError("sample exact story bytes changed")
    assignment = assignments[group]
    recipe = assignment.get("recipe")
    if type(recipe) is not dict:
        raise SampleReportError("sample assignment recipe is malformed")
    return {
        "condition": _text(selection, "condition"),
        "content_sha256": _text(record, "content_sha256"),
        "normalized_story_sha256": group,
        "noun_cluster": _integer(assignment, "noun_bucket"),
        "provenance": assignment.get("provenance"),
        "recipe": recipe,
        "record_id": _text(record, "record_id"),
        "source": _text(record, "source"),
        "source_index": _integer(record, "source_index"),
        "source_member": _text(record, "source_member"),
        "story": raw_story.decode("utf-8", errors="strict"),
        "story_sha256": story_sha,
        "token_count": _integer(record, "token_count"),
        "verb_cluster": _integer(assignment, "verb_bucket"),
    }


def _markdown(
    report_sha: str,
    content: Mapping[str, object],
    benchmark_version: str = "v1",
) -> str:
    lines = [
        f"# TinyWorlds-P Semantic {benchmark_version} Pre-Training Sample Report",
        "",
        f"Report SHA-256: `{report_sha}`  ",
        f"Partition SHA-256: `{content['partition_sha256']}`  ",
        f"Semantic catalog SHA-256: `{content['catalog_sha256']}`",
        "",
        "This report reads validation indexes only. The sealed test has not been opened.",
        "",
        "## Cluster word inventories",
        "",
    ]
    for cluster in content["clusters"]:
        lines.extend(
            (
                f"### {cluster['role']} cluster {cluster['index']}",
                "",
                ", ".join(f"`{word}`" for word in cluster["words"]),
                "",
            )
        )
    lines.extend(("## Exact validation samples", ""))
    for sample in content["samples"]:
        lines.extend(
            (
                f"### `{sample['condition']}`",
                "",
                f"Record `{sample['record_id']}`; source `{sample['source_member']}:{sample['source_index']}`; story SHA-256 `{sample['story_sha256']}`; noun cluster {sample['noun_cluster']}; verb cluster {sample['verb_cluster']}.",
                "",
                sample["story"],
                "",
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def _html(
    report_sha: str,
    content: Mapping[str, object],
    benchmark_version: str = "v1",
) -> str:
    inventories = "".join(
        f"<h3>{cluster['role']} cluster {cluster['index']}</h3><p>{escape(', '.join(cluster['words']))}</p>"
        for cluster in content["clusters"]
    )
    samples = "".join(
        f"<article><h3><code>{escape(sample['condition'])}</code></h3>"
        f"<p><small>{escape(sample['record_id'])}; {escape(sample['source_member'])}:{sample['source_index']}; story {sample['story_sha256']}; N{sample['noun_cluster']}×V{sample['verb_cluster']}</small></p>"
        f"<pre>{escape(sample['story'])}</pre></article>"
        for sample in content["samples"]
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>TinyWorlds-P Semantic Sample Report</title><style>body{{font:15px/1.5 system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem}}pre{{white-space:pre-wrap;background:#f4f6fa;padding:1rem}}code{{font-family:ui-monospace,monospace}}small{{color:#596273}}</style></head><body><h1>TinyWorlds-P Semantic {benchmark_version} Pre-Training Sample Report</h1><p>Report <code>{report_sha}</code><br>Partition <code>{content['partition_sha256']}</code><br>Catalog <code>{content['catalog_sha256']}</code></p><p>Validation only; sealed test unopened.</p><h2>Cluster word inventories</h2>{inventories}<h2>Exact validation samples</h2>{samples}</body></html>"""


def _write_tree(root: Path, report_sha: str) -> None:
    files = tuple(
        {
            "name": path.name,
            "sha256": _file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(root.iterdir())
        if path.is_file() and path.name != "tree.json"
    )
    _write_json(
        root / "tree.json",
        {
            "files": list(files),
            "format": "tinyworlds-p-semantic-sample-report-tree",
            "report_sha256": report_sha,
            "schema_version": 1,
        },
    )


def _write_json(path: Path, value: object) -> None:
    with path.open("wb") as output:
        output.write(canonical_json_bytes(value))
        output.flush()
        os.fsync(output.fileno())


def _write_text(path: Path, value: str) -> None:
    with path.open("wb") as output:
        output.write(value.encode("utf-8"))
        output.flush()
        os.fsync(output.fileno())


def _load_json(path: Path) -> dict[str, object]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SampleReportError(f"invalid semantic sample report JSON: {path}") from error
    if type(value) is not dict or canonical_json_bytes(value) != payload:
        raise SampleReportError(f"noncanonical semantic sample report JSON: {path}")
    return value


def _iter_jsonl(path: Path) -> Iterator[dict[str, object]]:
    if "test" in path.name:
        raise SampleReportError("sample reporting cannot open sealed-test indexes")
    with path.open("rb") as source:
        for line in source:
            value = json.loads(line)
            if type(value) is not dict or canonical_json_bytes(value) != line:
                raise SampleReportError(f"noncanonical sample source JSONL: {path}")
            yield value


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _text(record: Mapping[str, object], field: str) -> str:
    value = record.get(field)
    if type(value) is not str or not value:
        raise SampleReportError(f"field {field!r} must be nonempty text")
    return value


def _integer(record: Mapping[str, object], field: str) -> int:
    value = record.get(field)
    if type(value) is not int or value < 0:
        raise SampleReportError(f"field {field!r} must be a nonnegative integer")
    return value


__all__ = [
    "SampleReportError",
    "SemanticSampleReport",
    "load_sample_report",
    "publish_sample_report",
]

"""Authenticated validation-only story and query samples before main training."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from hashlib import sha256
import html
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from apm.data.text.tinyworlds_q_semantic.catalog import ValidationCatalogView
from apm.data.text.tinyworlds_q_semantic.contracts import (
    BENCHMARK_ID,
    QueryPartitionArtifact,
    canonical_json_bytes,
    record_sha256,
    require_sha256,
)


QUERY_SAMPLE_REPORT_FORMAT = "tinyworlds-q-semantic-validation-sample-report-v1"
QUERY_SAMPLE_REPORT_TREE_FORMAT = (
    "tinyworlds-q-semantic-validation-sample-report-tree-v1"
)


@dataclass(frozen=True, slots=True)
class QueryValidationSampleReport:
    """Strict validation-only examples bound to one catalog and partition."""

    root: Path
    report_sha256: str
    partition_sha256: str
    catalog_sha256: str
    sample_count: int
    validation_query_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        for value, label in (
            (self.report_sha256, "validation sample report"),
            (self.partition_sha256, "validation sample partition"),
            (self.catalog_sha256, "validation sample catalog"),
        ):
            require_sha256(value, label)
        if (
            type(self.sample_count) is not int
            or self.sample_count < 2
            or type(self.validation_query_count) is not int
            or self.validation_query_count <= 0
        ):
            raise ValueError("validation sample report coverage is invalid")


def publish_query_validation_sample_report(
    artifact: QueryPartitionArtifact,
    catalog: ValidationCatalogView,
    output_root: str | Path,
) -> QueryValidationSampleReport:
    """Publish one exact base story, one per node, and all validation queries."""
    _validate_sources(artifact, catalog)
    samples = _validation_samples(artifact)
    content = {
        "benchmark_id": BENCHMARK_ID,
        "catalog_sha256": catalog.catalog_sha256,
        "concept_ids": list(catalog.concept_ids),
        "format": QUERY_SAMPLE_REPORT_FORMAT,
        "partition_sha256": artifact.partition_sha256,
        "samples": list(samples),
        "schema_version": 1,
        "sealed_test_opened": False,
        "validation_queries": [
            template.as_record() for template in catalog.templates
        ],
    }
    report_sha256 = record_sha256(content)
    root = (
        Path(output_root)
        / artifact.partition_sha256
        / report_sha256
    )
    if root.exists():
        return load_query_validation_sample_report(root, artifact, catalog)
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".validation-samples-", dir=root.parent))
    try:
        record = {**content, "report_sha256": report_sha256}
        payloads = {
            "sample-report.json": canonical_json_bytes(record),
            "sample-report.md": _markdown(record).encode("utf-8"),
            "sample-report.html": _standalone_html(record).encode("utf-8"),
        }
        for name, payload in payloads.items():
            (staging / name).write_bytes(payload)
        manifest = {
            "files": [
                {
                    "name": name,
                    "sha256": sha256(payload).hexdigest(),
                    "size_bytes": len(payload),
                }
                for name, payload in sorted(payloads.items())
            ],
            "format": QUERY_SAMPLE_REPORT_TREE_FORMAT,
            "report_sha256": report_sha256,
            "schema_version": 1,
        }
        (staging / "manifest.json").write_bytes(canonical_json_bytes(manifest))
        os.replace(staging, root)
    except BaseException:
        _remove_staging(staging)
        raise
    return load_query_validation_sample_report(root, artifact, catalog)


def load_query_validation_sample_report(
    directory: str | Path,
    artifact: QueryPartitionArtifact,
    catalog: ValidationCatalogView,
) -> QueryValidationSampleReport:
    """Authenticate every report byte and reproduce its validation selections."""
    _validate_sources(artifact, catalog)
    root = Path(directory)
    manifest = _load_json(root / "manifest.json")
    if (
        set(manifest) != {"files", "format", "report_sha256", "schema_version"}
        or manifest.get("format") != QUERY_SAMPLE_REPORT_TREE_FORMAT
        or manifest.get("report_sha256") != root.name
        or manifest.get("schema_version") != 1
    ):
        raise ValueError("validation sample report manifest changed")
    descriptors = manifest.get("files")
    if type(descriptors) is not list or any(
        type(item) is not dict for item in descriptors
    ):
        raise ValueError("validation sample report descriptors changed")
    expected_names = {
        "manifest.json",
        *(str(item.get("name")) for item in descriptors),
    }
    if (
        not root.is_dir()
        or root.is_symlink()
        or {path.name for path in root.iterdir()} != expected_names
        or any(path.is_symlink() for path in root.iterdir())
    ):
        raise ValueError("validation sample report tree entries changed")
    for descriptor in descriptors:
        name = descriptor.get("name")
        size_bytes = descriptor.get("size_bytes")
        expected_sha256 = descriptor.get("sha256")
        if (
            type(name) is not str
            or type(size_bytes) is not int
            or type(expected_sha256) is not str
        ):
            raise ValueError("validation sample report descriptor is invalid")
        payload = (root / name).read_bytes()
        if (
            len(payload) != size_bytes
            or sha256(payload).hexdigest() != expected_sha256
        ):
            raise ValueError("validation sample report file changed")
    record = _load_json(root / "sample-report.json")
    report_sha256 = _text(record, "report_sha256")
    content = {key: value for key, value in record.items() if key != "report_sha256"}
    expected_samples = list(_validation_samples(artifact))
    expected_queries = [template.as_record() for template in catalog.templates]
    if (
        report_sha256 != root.name
        or record_sha256(content) != report_sha256
        or record.get("format") != QUERY_SAMPLE_REPORT_FORMAT
        or record.get("benchmark_id") != BENCHMARK_ID
        or record.get("schema_version") != 1
        or record.get("partition_sha256") != artifact.partition_sha256
        or record.get("catalog_sha256") != catalog.catalog_sha256
        or record.get("concept_ids") != list(catalog.concept_ids)
        or record.get("sealed_test_opened") is not False
        or record.get("samples") != expected_samples
        or record.get("validation_queries") != expected_queries
        or (root / "sample-report.md").read_text(encoding="utf-8")
        != _markdown(record)
        or (root / "sample-report.html").read_text(encoding="utf-8")
        != _standalone_html(record)
    ):
        raise ValueError("validation sample report semantic content changed")
    return QueryValidationSampleReport(
        root=root.resolve(),
        report_sha256=report_sha256,
        partition_sha256=artifact.partition_sha256,
        catalog_sha256=catalog.catalog_sha256,
        sample_count=len(expected_samples),
        validation_query_count=len(expected_queries),
    )


def _validate_sources(
    artifact: QueryPartitionArtifact,
    catalog: ValidationCatalogView,
) -> None:
    if (
        type(artifact) is not QueryPartitionArtifact
        or type(catalog) is not ValidationCatalogView
        or artifact.catalog_sha256 != catalog.catalog_sha256
        or artifact.concept_ids != catalog.concept_ids
    ):
        raise ValueError("validation sample report sources do not match")


def _validation_samples(
    artifact: QueryPartitionArtifact,
) -> tuple[dict[str, object], ...]:
    selections = (
        ("base/validation", artifact.root / "indexes" / "base-validation.jsonl"),
        *(
            (
                f"node/{concept_id}/validation",
                artifact.root
                / "indexes"
                / f"node-{concept_id}-validation.jsonl",
            )
            for concept_id in artifact.concept_ids
        ),
    )
    return tuple(
        _materialize_sample(artifact, condition, _minimum_document(index_path))
        for condition, index_path in selections
    )


def _minimum_document(path: Path) -> dict[str, object]:
    selected = min(
        _iter_jsonl(path),
        key=lambda record: (
            _text(record, "group_sha256"),
            _text(record, "record_id"),
        ),
        default=None,
    )
    if selected is None:
        raise ValueError(f"validation sample index is empty: {path.name}")
    return selected


def _materialize_sample(
    artifact: QueryPartitionArtifact,
    condition: str,
    record: dict[str, object],
) -> dict[str, object]:
    expected_role = "base" if condition == "base/validation" else "node"
    expected_world = None if expected_role == "base" else condition.split("/")[1]
    world = record.get("world")
    if (
        record.get("role") != expected_role
        or record.get("split") != "validation"
        or world != expected_world
    ):
        raise ValueError("validation sample index role or world changed")
    story_offset = _integer(record, "story_offset")
    story_size = _integer(record, "story_bytes")
    token_offset = _integer(record, "token_offset")
    token_count = _integer(record, "token_count")
    with (artifact.root / "stories.bin").open("rb") as stories:
        stories.seek(story_offset)
        story_bytes = stories.read(story_size)
    with (artifact.root / "tokens.uint16").open("rb") as tokens:
        tokens.seek(token_offset * 2)
        token_payload = tokens.read(token_count * 2)
    token_ids = tuple(
        int(value) for value in np.frombuffer(token_payload, dtype="<u2")
    )
    if (
        len(story_bytes) != story_size
        or sha256(story_bytes).hexdigest() != _text(record, "story_sha256")
        or len(token_ids) != token_count
    ):
        raise ValueError("validation sample payload changed or is truncated")
    return {
        "condition": condition,
        "content_sha256": _text(record, "content_sha256"),
        "group_sha256": _text(record, "group_sha256"),
        "record_id": _text(record, "record_id"),
        "role": expected_role,
        "source": _text(record, "source"),
        "source_index": _integer(record, "source_index"),
        "source_member": _text(record, "source_member"),
        "split": "validation",
        "story": story_bytes.decode("utf-8", errors="strict"),
        "story_sha256": _text(record, "story_sha256"),
        "token_ids": list(token_ids),
        "world": world,
    }


def _markdown(record: dict[str, object]) -> str:
    samples = record.get("samples")
    queries = record.get("validation_queries")
    if type(samples) is not list or type(queries) is not list:
        raise ValueError("validation sample report rendering input changed")
    lines = [
        "# TinyWorlds-Q validation-only pre-training sample report",
        "",
        f"Report: `{record['report_sha256']}`  ",
        f"Partition: `{record['partition_sha256']}`  ",
        f"Catalog: `{record['catalog_sha256']}`",
        "",
        "This report reads validation indexes and validation queries only. "
        "The sealed test was not opened.",
        "",
        "## Exact validation stories",
        "",
    ]
    for sample in samples:
        if type(sample) is not dict:
            raise ValueError("validation sample report story changed")
        lines.extend(
            (
                f"### `{sample['condition']}`",
                "",
                f"Record `{sample['record_id']}`; source "
                f"`{sample['source_member']}:{sample['source_index']}`; "
                f"group `{sample['group_sha256']}`; story "
                f"`{sample['story_sha256']}`.",
                "",
                str(sample["story"]),
                "",
                f"Token IDs: `{tuple(sample['token_ids'])}`",
                "",
            )
        )
    lines.extend(
        (
            "## Validation query coverage",
            "",
            f"All {len(queries)} validation query records are included in "
            "`sample-report.json`; sealed query records are absent.",
            "",
        )
    )
    return "\n".join(lines)


def _standalone_html(record: dict[str, object]) -> str:
    return (
        "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<title>TinyWorlds-Q validation samples</title>"
        "<style>body{font:15px/1.5 system-ui;max-width:1100px;margin:2rem auto;"
        "padding:0 1rem}pre{white-space:pre-wrap;overflow-wrap:anywhere;"
        "background:#f5f7f9;padding:1.25rem;border-radius:8px}</style></head>"
        f"<body data-report-sha256=\"{record['report_sha256']}\"><pre>"
        f"{html.escape(_markdown(record))}</pre></body></html>\n"
    )


def _iter_jsonl(path: Path) -> Iterator[dict[str, object]]:
    with path.open("rb") as source:
        for line in source:
            record = json.loads(line)
            if type(record) is not dict or canonical_json_bytes(record) != line:
                raise ValueError(f"validation sample index is noncanonical: {path}")
            yield record


def _load_json(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    record = json.loads(payload)
    if type(record) is not dict or canonical_json_bytes(record) != payload:
        raise ValueError(f"validation sample JSON is noncanonical: {path.name}")
    return record


def _text(record: dict[str, object], field: str) -> str:
    value = record.get(field)
    if type(value) is not str:
        raise ValueError(f"validation sample {field} must be text")
    return value


def _integer(record: dict[str, object], field: str) -> int:
    value = record.get(field)
    if type(value) is not int:
        raise ValueError(f"validation sample {field} must be an integer")
    return value


def _remove_staging(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            path.rmdir()
    if root.exists():
        root.rmdir()


__all__ = [
    "QueryValidationSampleReport",
    "load_query_validation_sample_report",
    "publish_query_validation_sample_report",
]

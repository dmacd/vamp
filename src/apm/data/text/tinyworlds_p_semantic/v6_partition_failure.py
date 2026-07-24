"""Authenticated evidence for a semantic-v6 feasibility stop."""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
from html import escape
import json
import os
from pathlib import Path
import shutil

from apm.data.text.tinyworlds_p import artifact as archive_artifact
from apm.data.text.tinyworlds_p.contracts import NORMALIZATION_IDENTITY
from apm.data.text.tinyworlds_p_semantic.builder import _source_record
from apm.data.text.tinyworlds_p_semantic.contracts import (
    canonical_json_bytes,
    record_sha256,
)
from apm.data.text.tinyworlds_p_semantic.v4_catalog import (
    V4SemanticCatalogError,
    load_v4_semantic_catalog,
)
from apm.data.text.tinyworlds_p_semantic.v4_contracts import V4SemanticCatalog
from apm.data.text.tinyworlds_p_semantic.v5_partition_contracts import (
    V5_BENCHMARK_ID,
    V5SemanticPartitionFailure,
)
from apm.data.text.tinyworlds_p_semantic.v5_partition_failure import (
    V5SemanticPartitionFailureError,
    load_v5_partition_failure,
    load_v5_partition_failure_evidence,
)
from apm.data.text.tinyworlds_p_semantic.v6_partition_contracts import (
    V6_BENCHMARK_ID,
    V6_FEASIBILITY_FAILURE_REASON,
    V6_FEASIBILITY_FAILURE_STAGE,
    V6_PARENT_CATALOG_SHA256,
    V6_PARENT_PARTITION_FAILURE_SHA256,
    V6_PARTITION_FAILURE_FORMAT,
    V6_PARTITION_FAILURE_TREE_FORMAT,
    V6_PARTITION_SCHEMA_VERSION,
    V6SemanticPartitionFailure,
    V6SemanticPartitionInputs,
    V6SemanticPartitionPreset,
)
from apm.data.text.tinyworlds_p_semantic.v6_topology import (
    V6TopologyEvidenceError,
    validate_topology_selection,
)


class V6SemanticPartitionFailureError(ValueError):
    """A semantic-v6 partition failure is malformed or inconsistent."""


def publish_v6_partition_failure(
    inputs: V6SemanticPartitionInputs,
    preset: V6SemanticPartitionPreset,
    catalog: V4SemanticCatalog,
    parent_failure: V5SemanticPartitionFailure,
    seed_identity_sha256: str,
    semantic_exclusions: Mapping[str, int],
    topology_selection: Mapping[str, object],
) -> V6SemanticPartitionFailure:
    """Publish and strictly reload a no-feasible-candidate v6 stop."""
    if type(inputs) is not V6SemanticPartitionInputs:
        raise TypeError("semantic-v6 failure requires its dedicated inputs")
    if type(preset) is not V6SemanticPartitionPreset:
        raise TypeError("semantic-v6 failure requires its frozen preset")
    if type(catalog) is not V4SemanticCatalog:
        raise TypeError("semantic-v6 failure requires the strict v4 catalog")
    if type(parent_failure) is not V5SemanticPartitionFailure:
        raise TypeError("semantic-v6 failure requires the strict v5 parent")
    if (
        catalog.catalog_sha256 != V6_PARENT_CATALOG_SHA256
        or parent_failure.failure_sha256 != V6_PARENT_PARTITION_FAILURE_SHA256
    ):
        raise ValueError("semantic-v6 failure parents changed")
    parent_evidence = load_v5_partition_failure_evidence(parent_failure)
    parent_source = _parent_source(parent_failure)
    sources = _source_record(
        inputs,
        catalog,
        additional_sources=parent_source,
    )
    if {
        name: sources[name]
        for name in ("archive", "semantic_catalog", "tokenizer")
    } != {
        name: parent_evidence.sources[name]
        for name in ("archive", "semantic_catalog", "tokenizer")
    }:
        raise ValueError("semantic-v6 failure sources differ from the parent")
    if (
        preset.v4_shape.as_record()
        != parent_evidence.parent_v4_evidence.partition_preset
    ):
        raise ValueError("semantic-v6 failure settings differ from the parent")
    try:
        validate_topology_selection(
            topology_selection,
            parent_evidence.parent_v4_evidence.topology_candidates,
            seed_identity_sha256,
            preset,
        )
    except V6TopologyEvidenceError as error:
        raise ValueError("semantic-v6 failure topology evidence changed") from error
    if topology_selection.get("selected") is not None:
        raise ValueError("semantic-v6 failure has a feasible candidate")
    exclusions = dict(sorted(semantic_exclusions.items()))
    if exclusions != parent_evidence.semantic_exclusions:
        raise ValueError("semantic-v6 failure exclusions differ from the parent")
    evaluations = topology_selection.get("evaluations")
    if type(evaluations) is not list:
        raise ValueError("semantic-v6 failure has no feasibility records")
    feasibility_sha256 = record_sha256(evaluations)

    working = inputs.temporary_directory / "partition-failure-publication"
    if working.exists():
        raise FileExistsError(f"semantic-v6 failure staging path exists: {working}")
    working.mkdir(parents=True)
    audit = {
        "semantic_exclusions": exclusions,
        "topology_selection": dict(topology_selection),
    }
    _write_json(working / "audit.json", audit)
    content = {
        "audit_sha256": _file_sha256(working / "audit.json"),
        "benchmark_id": V6_BENCHMARK_ID,
        "catalog_sha256": catalog.catalog_sha256,
        "failure_sha256": "",
        "feasibility_sha256": feasibility_sha256,
        "format": V6_PARTITION_FAILURE_FORMAT,
        "normalization": NORMALIZATION_IDENTITY.as_record(),
        "parent_partition_failure_sha256": parent_failure.failure_sha256,
        "preset": preset.as_record(),
        "reason": V6_FEASIBILITY_FAILURE_REASON,
        "schema_version": V6_PARTITION_SCHEMA_VERSION,
        "seed_identity_sha256": seed_identity_sha256,
        "sources": sources,
        "stage": V6_FEASIBILITY_FAILURE_STAGE,
    }
    failure_sha256 = record_sha256(
        {key: value for key, value in content.items() if key != "failure_sha256"}
    )
    content["failure_sha256"] = failure_sha256
    _write_json(working / "failure.json", content)
    markdown = _markdown(failure_sha256, topology_selection)
    _write_text(working / "audit.md", markdown)
    _write_text(working / "audit.html", _html(markdown))
    shutil.copytree(
        catalog.root,
        working / "semantic-catalog" / catalog.catalog_sha256,
    )
    shutil.copytree(
        parent_failure.root,
        working / "parent-partition-failure" / parent_failure.failure_sha256,
    )
    _write_tree(working, failure_sha256)

    failure_root = inputs.output_root / "failures"
    target = failure_root / failure_sha256
    if target.exists():
        if (working / "tree.json").read_bytes() != (target / "tree.json").read_bytes():
            raise RuntimeError("semantic-v6 failure rebuild is not byte-identical")
        shutil.rmtree(working)
        return load_v6_partition_failure(target)
    failure_root.mkdir(parents=True, exist_ok=True)
    os.rename(working, target)
    _fsync_directory(failure_root)
    return load_v6_partition_failure(target)


def load_v6_partition_failure(path: str | Path) -> V6SemanticPartitionFailure:
    """Strictly authenticate a semantic-v6 feasibility failure."""
    try:
        requested_root = Path(path)
        if requested_root.is_symlink() or not requested_root.is_dir():
            raise V6SemanticPartitionFailureError(
                "semantic-v6 failure must be a regular directory"
            )
        root = requested_root.resolve()
        tree = _load_json(root / "tree.json", "semantic-v6 failure tree")
        if (
            set(tree) != {"failure_sha256", "files", "format", "schema_version"}
            or tree.get("format") != V6_PARTITION_FAILURE_TREE_FORMAT
            or tree.get("schema_version") != V6_PARTITION_SCHEMA_VERSION
        ):
            raise V6SemanticPartitionFailureError("semantic-v6 failure tree changed")
        failure_sha256 = _text(tree, "failure_sha256")
        if root.name != failure_sha256:
            raise V6SemanticPartitionFailureError(
                "semantic-v6 failure directory identity changed"
            )
        _validate_tree(root, tree)
        failure = _load_json(root / "failure.json", "semantic-v6 failure identity")
        required = {
            "audit_sha256",
            "benchmark_id",
            "catalog_sha256",
            "failure_sha256",
            "feasibility_sha256",
            "format",
            "normalization",
            "parent_partition_failure_sha256",
            "preset",
            "reason",
            "schema_version",
            "seed_identity_sha256",
            "sources",
            "stage",
        }
        identity = {
            key: value for key, value in failure.items() if key != "failure_sha256"
        }
        if (
            set(failure) != required
            or failure.get("failure_sha256") != failure_sha256
            or record_sha256(identity) != failure_sha256
            or failure.get("benchmark_id") != V6_BENCHMARK_ID
            or failure.get("format") != V6_PARTITION_FAILURE_FORMAT
            or failure.get("schema_version") != V6_PARTITION_SCHEMA_VERSION
            or failure.get("stage") != V6_FEASIBILITY_FAILURE_STAGE
            or failure.get("normalization") != NORMALIZATION_IDENTITY.as_record()
        ):
            raise V6SemanticPartitionFailureError(
                "semantic-v6 failure identity changed"
            )
        catalog_sha256 = _sha_text(failure, "catalog_sha256")
        parent_sha256 = _sha_text(failure, "parent_partition_failure_sha256")
        seed_identity_sha256 = _sha_text(failure, "seed_identity_sha256")
        feasibility_sha256 = _sha_text(failure, "feasibility_sha256")
        if (
            catalog_sha256 != V6_PARENT_CATALOG_SHA256
            or parent_sha256 != V6_PARENT_PARTITION_FAILURE_SHA256
        ):
            raise V6SemanticPartitionFailureError(
                "semantic-v6 failure parents changed"
            )
        sources = _mapping(failure, "sources")
        if set(sources) != {
            "archive",
            "parent_partition_failure",
            "semantic_catalog",
            "tokenizer",
        }:
            raise V6SemanticPartitionFailureError(
                "semantic-v6 failure source set changed"
            )
        from apm.data.text.tinyworlds_p_semantic.v6_partition_artifact import _preset

        try:
            archive_artifact._source_identity(_mapping(sources, "archive"))
            archive_artifact._tokenizer_identity(_mapping(sources, "tokenizer"))
            preset = _preset(_mapping(failure, "preset"))
        except (
            TypeError,
            ValueError,
            archive_artifact.PartitionArtifactError,
        ) as error:
            raise V6SemanticPartitionFailureError(
                "semantic-v6 failure source or preset is invalid"
            ) from error
        catalog_source = _mapping(sources, "semantic_catalog")
        if (
            set(catalog_source)
            != {"catalog_sha256", "encoder_identity_sha256", "evidence_sha256"}
            or catalog_source.get("catalog_sha256") != catalog_sha256
        ):
            raise V6SemanticPartitionFailureError(
                "semantic-v6 failure catalog source changed"
            )
        try:
            catalog = load_v4_semantic_catalog(
                root / "semantic-catalog" / catalog_sha256
            )
        except V4SemanticCatalogError as error:
            raise V6SemanticPartitionFailureError(
                "embedded semantic-v6 catalog changed"
            ) from error
        if (
            catalog.encoder_identity.identity_sha256
            != _sha_text(catalog_source, "encoder_identity_sha256")
            or catalog.evidence_sha256
            != _sha_text(catalog_source, "evidence_sha256")
        ):
            raise V6SemanticPartitionFailureError(
                "semantic-v6 failure catalog binding changed"
            )
        parent_source = _mapping(sources, "parent_partition_failure")
        if (
            set(parent_source) != {"benchmark_id", "failure_sha256", "tree_sha256"}
            or parent_source.get("benchmark_id") != V5_BENCHMARK_ID
            or parent_source.get("failure_sha256") != parent_sha256
        ):
            raise V6SemanticPartitionFailureError(
                "semantic-v6 parent source changed"
            )
        try:
            parent = load_v5_partition_failure(
                root / "parent-partition-failure" / parent_sha256
            )
        except V5SemanticPartitionFailureError as error:
            raise V6SemanticPartitionFailureError(
                "embedded semantic-v6 parent changed"
            ) from error
        if _file_sha256(parent.root / "tree.json") != _sha_text(
            parent_source,
            "tree_sha256",
        ):
            raise V6SemanticPartitionFailureError(
                "semantic-v6 parent tree binding changed"
            )
        parent_evidence = load_v5_partition_failure_evidence(parent)
        if {
            name: sources[name]
            for name in ("archive", "semantic_catalog", "tokenizer")
        } != {
            name: parent_evidence.sources[name]
            for name in ("archive", "semantic_catalog", "tokenizer")
        }:
            raise V6SemanticPartitionFailureError(
                "semantic-v6 sources differ from the parent"
            )
        if (
            preset.v4_shape.as_record()
            != parent_evidence.parent_v4_evidence.partition_preset
        ):
            raise V6SemanticPartitionFailureError(
                "semantic-v6 settings differ from the parent"
            )
        expected_seed = record_sha256(
            {
                "benchmark_id": V6_BENCHMARK_ID,
                "normalization": NORMALIZATION_IDENTITY.as_record(),
                "preset": preset.as_record(),
                "sources": sources,
            }
        )
        if seed_identity_sha256 != expected_seed:
            raise V6SemanticPartitionFailureError(
                "semantic-v6 failure seed changed"
            )
        if _file_sha256(root / "audit.json") != _sha_text(failure, "audit_sha256"):
            raise V6SemanticPartitionFailureError("semantic-v6 audit changed")
        audit = _load_json(root / "audit.json", "semantic-v6 failure audit")
        if set(audit) != {"semantic_exclusions", "topology_selection"}:
            raise V6SemanticPartitionFailureError(
                "semantic-v6 failure audit fields changed"
            )
        if _mapping(audit, "semantic_exclusions") != parent_evidence.semantic_exclusions:
            raise V6SemanticPartitionFailureError(
                "semantic-v6 exclusions differ from the parent"
            )
        topology_selection = _mapping(audit, "topology_selection")
        try:
            validate_topology_selection(
                topology_selection,
                parent_evidence.parent_v4_evidence.topology_candidates,
                seed_identity_sha256,
                preset,
            )
        except V6TopologyEvidenceError as error:
            raise V6SemanticPartitionFailureError(
                "semantic-v6 feasibility evidence changed"
            ) from error
        evaluations = topology_selection.get("evaluations")
        if (
            topology_selection.get("selected") is not None
            or type(evaluations) is not list
            or record_sha256(evaluations) != feasibility_sha256
            or failure.get("reason") != V6_FEASIBILITY_FAILURE_REASON
        ):
            raise V6SemanticPartitionFailureError(
                "semantic-v6 failure result changed"
            )
        return V6SemanticPartitionFailure(
            root=root,
            failure_sha256=failure_sha256,
            catalog_sha256=catalog_sha256,
            parent_partition_failure_sha256=parent_sha256,
            seed_identity_sha256=seed_identity_sha256,
            feasibility_sha256=feasibility_sha256,
            reason=V6_FEASIBILITY_FAILURE_REASON,
            topology_selection=topology_selection,
        )
    except V6SemanticPartitionFailureError:
        raise
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise V6SemanticPartitionFailureError(
            "semantic-v6 failure payload changed"
        ) from error


def _parent_source(parent: V5SemanticPartitionFailure) -> dict[str, object]:
    return {
        "parent_partition_failure": {
            "benchmark_id": V5_BENCHMARK_ID,
            "failure_sha256": parent.failure_sha256,
            "tree_sha256": _file_sha256(parent.root / "tree.json"),
        }
    }


def _markdown(
    failure_sha256: str,
    topology_selection: Mapping[str, object],
) -> str:
    evaluations = topology_selection.get("evaluations")
    assert type(evaluations) is list
    failures = [
        record
        for record in evaluations
        if type(record) is dict and record.get("control_feasible") is False
    ]
    lines = [
        "# TinyWorlds-P semantic-v6 partition failure",
        "",
        f"Failure identity: `{failure_sha256}`",
        "",
        "Version 6 measured every balanced layout with the complete split and "
        "comparison allocator. None could construct all required comparisons.",
        "",
        f"Measured balanced layouts: {len(evaluations)}.",
        "",
        "## Candidate failures",
        "",
    ]
    lines.extend(
        f"- Semantic rank {record.get('semantic_rank')}: "
        f"{record.get('failure_reason')}."
        for record in failures
    )
    lines.extend(
        [
            "",
            "No model was trained and the sealed test was not opened.",
            "",
        ]
    )
    return "\n".join(lines)


def _html(markdown: str) -> str:
    body = "\n".join(
        f"<p>{escape(line)}</p>" if line else ""
        for line in markdown.splitlines()
    )
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<title>TinyWorlds-P semantic-v6 partition failure</title>"
        "<style>body{font:16px system-ui;max-width:900px;margin:2rem auto;"
        "padding:0 1rem;line-height:1.5}p{overflow-wrap:anywhere}</style>"
        f"</head><body>{body}</body></html>\n"
    )


def _write_tree(root: Path, failure_sha256: str) -> None:
    files = tuple(
        {
            "relative_path": path.relative_to(root).as_posix(),
            "sha256": _file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(
            root.rglob("*"),
            key=lambda candidate: candidate.relative_to(root).as_posix(),
        )
        if path.is_file() and path.name != "tree.json"
    )
    _write_json(
        root / "tree.json",
        {
            "failure_sha256": failure_sha256,
            "files": list(files),
            "format": V6_PARTITION_FAILURE_TREE_FORMAT,
            "schema_version": V6_PARTITION_SCHEMA_VERSION,
        },
    )


def _validate_tree(root: Path, tree: Mapping[str, object]) -> None:
    raw_files = tree.get("files")
    if type(raw_files) is not list or any(type(item) is not dict for item in raw_files):
        raise V6SemanticPartitionFailureError(
            "semantic-v6 failure descriptors changed"
        )
    described = tuple(_text(item, "relative_path") for item in raw_files)
    actual = tuple(
        path.relative_to(root).as_posix()
        for path in sorted(
            root.rglob("*"),
            key=lambda candidate: candidate.relative_to(root).as_posix(),
        )
        if path.is_file() and path.name != "tree.json"
    )
    if described != actual or any((root / path).is_symlink() for path in actual):
        raise V6SemanticPartitionFailureError(
            "semantic-v6 failure membership changed"
        )
    for descriptor in raw_files:
        candidate = root / _text(descriptor, "relative_path")
        size = descriptor.get("size_bytes")
        if (
            type(size) is not int
            or size < 0
            or candidate.stat().st_size != size
            or _file_sha256(candidate) != _sha_text(descriptor, "sha256")
        ):
            raise V6SemanticPartitionFailureError(
                f"semantic-v6 failure file changed: {candidate.relative_to(root)}"
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


def _load_json(path: Path, label: str) -> dict[str, object]:
    value = json.loads(path.read_bytes())
    if type(value) is not dict:
        raise V6SemanticPartitionFailureError(f"invalid {label}")
    if canonical_json_bytes(value) != path.read_bytes():
        raise V6SemanticPartitionFailureError(f"noncanonical {label}")
    return value


def _mapping(record: Mapping[str, object], field: str) -> dict[str, object]:
    value = record.get(field)
    if type(value) is not dict:
        raise V6SemanticPartitionFailureError(
            f"semantic-v6 failure field {field!r} must be a mapping"
        )
    return value


def _text(record: Mapping[str, object], field: str) -> str:
    value = record.get(field)
    if type(value) is not str or not value:
        raise V6SemanticPartitionFailureError(
            f"semantic-v6 failure field {field!r} must be text"
        )
    return value


def _sha_text(record: Mapping[str, object], field: str) -> str:
    value = _text(record, field)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise V6SemanticPartitionFailureError(
            f"semantic-v6 failure field {field!r} must be SHA-256"
        )
    return value


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


__all__ = [
    "V6SemanticPartitionFailureError",
    "load_v6_partition_failure",
    "publish_v6_partition_failure",
]

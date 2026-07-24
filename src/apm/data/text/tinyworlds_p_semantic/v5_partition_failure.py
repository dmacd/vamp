"""Authenticated evidence for the semantic-v5 control-allocation stop."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from html import escape
import json
import os
from pathlib import Path
import re
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
from apm.data.text.tinyworlds_p_semantic.v4_contracts import (
    V4_BENCHMARK_ID,
    V4SemanticCatalog,
)
from apm.data.text.tinyworlds_p_semantic.v4_partition_failure import (
    V4SemanticPartitionFailureEvidence,
    V4SemanticPartitionFailureError,
    load_v4_partition_failure,
    load_v4_partition_failure_evidence,
)
from apm.data.text.tinyworlds_p_semantic.v4_partition_contracts import (
    V4SemanticPartitionFailure,
)
from apm.data.text.tinyworlds_p_semantic.v5_partition_contracts import (
    V5_BENCHMARK_ID,
    V5_CONTROL_ALLOCATION_FAILURE_STAGE,
    V5_PARENT_CATALOG_SHA256,
    V5_PARENT_PARTITION_FAILURE_SHA256,
    V5_PARTITION_FAILURE_FORMAT,
    V5_PARTITION_FAILURE_TREE_FORMAT,
    V5_PARTITION_SCHEMA_VERSION,
    V5ControlShortfall,
    V5SemanticPartitionFailure,
    V5SemanticPartitionInputs,
    V5SemanticPartitionPreset,
)
from apm.data.text.tinyworlds_p_semantic.v5_topology import (
    V5TopologyEvidenceError,
    topology_selection_from_parent_candidates,
    validate_topology_selection,
)


_CONTROL_SHORTFALL_PATTERN = re.compile(
    r"^control:([A-E]):(validation|test):(row|column) has "
    r"([0-9]+) candidates for ([0-9]+) controls$"
)


class V5SemanticPartitionFailureError(ValueError):
    """A semantic-v5 partition failure is malformed or inconsistent."""


@dataclass(frozen=True, slots=True)
class V5SemanticPartitionFailureEvidence:
    """Strict v5 settings and inherited evidence needed by a successor."""

    partition_preset: dict[str, object]
    semantic_exclusions: dict[str, int]
    sources: dict[str, object]
    parent_v4_failure: V4SemanticPartitionFailure
    parent_v4_evidence: V4SemanticPartitionFailureEvidence


def load_v5_partition_failure_evidence(
    failure: V5SemanticPartitionFailure,
) -> V5SemanticPartitionFailureEvidence:
    """Load the authenticated v5 evidence required by a later benchmark."""
    if type(failure) is not V5SemanticPartitionFailure:
        raise TypeError("v5 partition evidence requires its strict failure artifact")
    restored = load_v5_partition_failure(failure.root)
    if restored.failure_sha256 != failure.failure_sha256:
        raise V5SemanticPartitionFailureError("v5 partition failure identity changed")
    identity = _load_json(restored.root / "failure.json", "semantic-v5 failure")
    audit = _load_json(restored.root / "audit.json", "semantic-v5 failure audit")
    raw_preset = identity.get("preset")
    raw_sources = identity.get("sources")
    raw_exclusions = audit.get("semantic_exclusions")
    if (
        type(raw_preset) is not dict
        or type(raw_sources) is not dict
        or type(raw_exclusions) is not dict
    ):
        raise V5SemanticPartitionFailureError("v5 successor evidence changed")
    parent = load_v4_partition_failure(
        restored.root
        / "parent-partition-failure"
        / restored.parent_partition_failure_sha256
    )
    return V5SemanticPartitionFailureEvidence(
        partition_preset=dict(raw_preset),
        semantic_exclusions={
            str(name): int(value) for name, value in raw_exclusions.items()
        },
        sources=dict(raw_sources),
        parent_v4_failure=parent,
        parent_v4_evidence=load_v4_partition_failure_evidence(parent),
    )


def parse_v5_control_shortfall(reason: str) -> V5ControlShortfall:
    """Parse the canonical archive partitioner's control-shortage message."""
    if type(reason) is not str:
        raise V5SemanticPartitionFailureError(
            "semantic-v5 control failure reason must be text"
        )
    matched = _CONTROL_SHORTFALL_PATTERN.fullmatch(reason)
    if matched is None:
        raise V5SemanticPartitionFailureError(
            "semantic-v5 stopped for an unregistered allocation reason"
        )
    world, split, arm, available, required = matched.groups()
    try:
        return V5ControlShortfall(
            world=world,
            split=split,
            arm=arm,
            available_count=int(available),
            required_count=int(required),
        )
    except ValueError as error:
        raise V5SemanticPartitionFailureError(
            "semantic-v5 control failure counts are inconsistent"
        ) from error


def publish_v5_partition_failure(
    inputs: V5SemanticPartitionInputs,
    preset: V5SemanticPartitionPreset,
    catalog: V4SemanticCatalog,
    parent_failure: V4SemanticPartitionFailure,
    seed_identity_sha256: str,
    semantic_exclusions: Mapping[str, int],
    topology_selection: Mapping[str, object],
    assignments_path: str | Path,
    reason: str,
) -> V5SemanticPartitionFailure:
    """Publish and strictly reload one content-addressed v5 control stop."""
    if type(inputs) is not V5SemanticPartitionInputs:
        raise TypeError("semantic-v5 failure requires its dedicated inputs")
    if type(preset) is not V5SemanticPartitionPreset:
        raise TypeError("semantic-v5 failure requires its frozen preset")
    if type(catalog) is not V4SemanticCatalog:
        raise TypeError("semantic-v5 failure requires the strict v4 catalog")
    if type(parent_failure) is not V4SemanticPartitionFailure:
        raise TypeError("semantic-v5 failure requires the strict v4 parent failure")
    if (
        catalog.catalog_sha256 != V5_PARENT_CATALOG_SHA256
        or parent_failure.failure_sha256
        != V5_PARENT_PARTITION_FAILURE_SHA256
    ):
        raise ValueError("semantic-v5 failure parents changed")
    shortfall = parse_v5_control_shortfall(reason)
    assignment_source = Path(assignments_path)
    if assignment_source.is_symlink() or not assignment_source.is_file():
        raise FileNotFoundError(assignment_source)
    assignments_sha256 = _file_sha256(assignment_source)
    parent_evidence = load_v4_partition_failure_evidence(parent_failure)
    parent_source = _parent_source(parent_failure)
    sources = _source_record(
        inputs,
        catalog,
        additional_sources=parent_source,
    )
    if {
        name: sources[name]
        for name in ("archive", "semantic_catalog", "tokenizer")
    } != parent_evidence.sources:
        raise ValueError("semantic-v5 failure sources differ from the v4 parent")
    if preset.v4_shape.as_record() != parent_evidence.partition_preset:
        raise ValueError("semantic-v5 failure settings differ from the v4 parent")
    expected_selection = topology_selection_from_parent_candidates(
        parent_evidence.topology_candidates,
        seed_identity_sha256,
        preset,
    )
    if dict(topology_selection) != expected_selection:
        raise ValueError("semantic-v5 failure topology selection changed")
    exclusions = dict(sorted(semantic_exclusions.items()))
    if exclusions != parent_evidence.semantic_exclusions:
        raise ValueError("semantic-v5 failure exclusions differ from the v4 parent")

    working = inputs.temporary_directory / "partition-failure-publication"
    if working.exists():
        raise FileExistsError(f"semantic-v5 failure staging path exists: {working}")
    working.mkdir(parents=True)
    audit = {
        "assignments_sha256": assignments_sha256,
        "control_shortfall": shortfall.as_record(),
        "semantic_exclusions": exclusions,
        "topology_selection": expected_selection,
    }
    _write_json(working / "audit.json", audit)
    content = {
        "assignments_sha256": assignments_sha256,
        "audit_sha256": _file_sha256(working / "audit.json"),
        "benchmark_id": V5_BENCHMARK_ID,
        "catalog_sha256": catalog.catalog_sha256,
        "failure_sha256": "",
        "format": V5_PARTITION_FAILURE_FORMAT,
        "normalization": NORMALIZATION_IDENTITY.as_record(),
        "parent_partition_failure_sha256": parent_failure.failure_sha256,
        "preset": preset.as_record(),
        "reason": shortfall.reason,
        "schema_version": V5_PARTITION_SCHEMA_VERSION,
        "seed_identity_sha256": seed_identity_sha256,
        "sources": sources,
        "stage": V5_CONTROL_ALLOCATION_FAILURE_STAGE,
    }
    failure_sha256 = record_sha256(
        {key: value for key, value in content.items() if key != "failure_sha256"}
    )
    content["failure_sha256"] = failure_sha256
    _write_json(working / "failure.json", content)
    markdown = _markdown(failure_sha256, shortfall, expected_selection)
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
            raise RuntimeError("semantic-v5 failure rebuild is not byte-identical")
        shutil.rmtree(working)
        return load_v5_partition_failure(target)
    failure_root.mkdir(parents=True, exist_ok=True)
    os.rename(working, target)
    _fsync_directory(failure_root)
    return load_v5_partition_failure(target)


def load_v5_partition_failure(path: str | Path) -> V5SemanticPartitionFailure:
    """Strictly authenticate a semantic-v5 control-allocation failure."""
    try:
        requested_root = Path(path)
        if requested_root.is_symlink() or not requested_root.is_dir():
            raise V5SemanticPartitionFailureError(
                "semantic-v5 partition failure must be a regular directory"
            )
        root = requested_root.resolve()
        tree = _load_json(root / "tree.json", "semantic-v5 failure tree")
        if (
            set(tree) != {"failure_sha256", "files", "format", "schema_version"}
            or tree.get("format") != V5_PARTITION_FAILURE_TREE_FORMAT
            or tree.get("schema_version") != V5_PARTITION_SCHEMA_VERSION
        ):
            raise V5SemanticPartitionFailureError(
                "semantic-v5 partition failure tree changed"
            )
        failure_sha256 = _text(tree, "failure_sha256")
        if root.name != failure_sha256:
            raise V5SemanticPartitionFailureError(
                "semantic-v5 partition failure directory identity changed"
            )
        _validate_tree(root, tree)
        failure = _load_json(root / "failure.json", "semantic-v5 failure identity")
        required = {
            "assignments_sha256",
            "audit_sha256",
            "benchmark_id",
            "catalog_sha256",
            "failure_sha256",
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
            or failure.get("benchmark_id") != V5_BENCHMARK_ID
            or failure.get("format") != V5_PARTITION_FAILURE_FORMAT
            or failure.get("schema_version") != V5_PARTITION_SCHEMA_VERSION
            or failure.get("stage") != V5_CONTROL_ALLOCATION_FAILURE_STAGE
            or failure.get("normalization") != NORMALIZATION_IDENTITY.as_record()
        ):
            raise V5SemanticPartitionFailureError(
                "semantic-v5 partition failure identity changed"
            )
        assignments_sha256 = _sha_text(failure, "assignments_sha256")
        seed_identity_sha256 = _sha_text(failure, "seed_identity_sha256")
        catalog_sha256 = _sha_text(failure, "catalog_sha256")
        parent_sha256 = _sha_text(failure, "parent_partition_failure_sha256")
        if (
            catalog_sha256 != V5_PARENT_CATALOG_SHA256
            or parent_sha256 != V5_PARENT_PARTITION_FAILURE_SHA256
        ):
            raise V5SemanticPartitionFailureError(
                "semantic-v5 partition failure parents changed"
            )
        sources = _mapping(failure, "sources")
        if set(sources) != {
            "archive",
            "parent_partition_failure",
            "semantic_catalog",
            "tokenizer",
        }:
            raise V5SemanticPartitionFailureError(
                "semantic-v5 partition failure source set changed"
            )
        from apm.data.text.tinyworlds_p_semantic.v5_partition_artifact import _preset

        try:
            archive_artifact._source_identity(_mapping(sources, "archive"))
            archive_artifact._tokenizer_identity(_mapping(sources, "tokenizer"))
            preset = _preset(_mapping(failure, "preset"))
        except (
            TypeError,
            ValueError,
            archive_artifact.PartitionArtifactError,
        ) as error:
            raise V5SemanticPartitionFailureError(
                "semantic-v5 failure source or preset is invalid"
            ) from error
        catalog_source = _mapping(sources, "semantic_catalog")
        if (
            set(catalog_source)
            != {"catalog_sha256", "encoder_identity_sha256", "evidence_sha256"}
            or catalog_source.get("catalog_sha256") != catalog_sha256
        ):
            raise V5SemanticPartitionFailureError(
                "semantic-v5 failure catalog source changed"
            )
        try:
            catalog = load_v4_semantic_catalog(
                root / "semantic-catalog" / catalog_sha256
            )
        except V4SemanticCatalogError as error:
            raise V5SemanticPartitionFailureError(
                "embedded semantic-v5 failure catalog changed"
            ) from error
        if (
            catalog.encoder_identity.identity_sha256
            != _sha_text(catalog_source, "encoder_identity_sha256")
            or catalog.evidence_sha256 != _sha_text(catalog_source, "evidence_sha256")
        ):
            raise V5SemanticPartitionFailureError(
                "semantic-v5 failure catalog binding changed"
            )
        parent_source = _mapping(sources, "parent_partition_failure")
        if (
            set(parent_source) != {"benchmark_id", "failure_sha256", "tree_sha256"}
            or parent_source.get("benchmark_id") != V4_BENCHMARK_ID
            or parent_source.get("failure_sha256") != parent_sha256
        ):
            raise V5SemanticPartitionFailureError(
                "semantic-v5 failure parent source changed"
            )
        try:
            parent = load_v4_partition_failure(
                root / "parent-partition-failure" / parent_sha256
            )
        except V4SemanticPartitionFailureError as error:
            raise V5SemanticPartitionFailureError(
                "embedded semantic-v5 parent failure changed"
            ) from error
        if _file_sha256(parent.root / "tree.json") != _sha_text(
            parent_source,
            "tree_sha256",
        ):
            raise V5SemanticPartitionFailureError(
                "semantic-v5 failure parent tree binding changed"
            )
        parent_evidence = load_v4_partition_failure_evidence(parent)
        core_sources = {
            name: sources[name]
            for name in ("archive", "semantic_catalog", "tokenizer")
        }
        if core_sources != parent_evidence.sources:
            raise V5SemanticPartitionFailureError(
                "semantic-v5 failure sources differ from the v4 parent"
            )
        if preset.v4_shape.as_record() != parent_evidence.partition_preset:
            raise V5SemanticPartitionFailureError(
                "semantic-v5 failure settings differ from the v4 parent"
            )
        expected_seed = record_sha256(
            {
                "benchmark_id": V5_BENCHMARK_ID,
                "normalization": NORMALIZATION_IDENTITY.as_record(),
                "preset": preset.as_record(),
                "sources": sources,
            }
        )
        if seed_identity_sha256 != expected_seed:
            raise V5SemanticPartitionFailureError(
                "semantic-v5 partition failure seed changed"
            )
        if _file_sha256(root / "audit.json") != _sha_text(failure, "audit_sha256"):
            raise V5SemanticPartitionFailureError(
                "semantic-v5 partition failure audit changed"
            )
        audit = _load_json(root / "audit.json", "semantic-v5 failure audit")
        if set(audit) != {
            "assignments_sha256",
            "control_shortfall",
            "semantic_exclusions",
            "topology_selection",
        }:
            raise V5SemanticPartitionFailureError(
                "semantic-v5 partition failure audit fields changed"
            )
        if audit.get("assignments_sha256") != assignments_sha256:
            raise V5SemanticPartitionFailureError(
                "semantic-v5 failed assignment identity changed"
            )
        if _mapping(audit, "semantic_exclusions") != (
            parent_evidence.semantic_exclusions
        ):
            raise V5SemanticPartitionFailureError(
                "semantic-v5 failure exclusions differ from the v4 parent"
            )
        topology_selection = _mapping(audit, "topology_selection")
        try:
            validate_topology_selection(
                topology_selection,
                parent_evidence.topology_candidates,
                seed_identity_sha256,
                preset,
            )
        except V5TopologyEvidenceError as error:
            raise V5SemanticPartitionFailureError(
                "semantic-v5 failure topology evidence changed"
            ) from error
        shortfall = _shortfall(_mapping(audit, "control_shortfall"))
        reason = _text(failure, "reason")
        if reason != shortfall.reason:
            raise V5SemanticPartitionFailureError(
                "semantic-v5 partition failure reason changed"
            )
        _validate_embedded_membership(root, catalog.root, parent.root)
        return V5SemanticPartitionFailure(
            root=root.resolve(),
            failure_sha256=failure_sha256,
            catalog_sha256=catalog_sha256,
            parent_partition_failure_sha256=parent_sha256,
            seed_identity_sha256=seed_identity_sha256,
            assignments_sha256=assignments_sha256,
            reason=reason,
            shortfall=shortfall,
            topology_selection=topology_selection,
        )
    except V5SemanticPartitionFailureError:
        raise
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise V5SemanticPartitionFailureError(
            "semantic-v5 partition failure payload changed"
        ) from error


def _parent_source(
    parent_failure: V4SemanticPartitionFailure,
) -> dict[str, object]:
    return {
        "parent_partition_failure": {
            "benchmark_id": V4_BENCHMARK_ID,
            "failure_sha256": parent_failure.failure_sha256,
            "tree_sha256": _file_sha256(parent_failure.root / "tree.json"),
        }
    }


def _shortfall(record: Mapping[str, object]) -> V5ControlShortfall:
    if set(record) != {
        "arm",
        "available_count",
        "required_count",
        "shortage_count",
        "split",
        "world",
    }:
        raise V5SemanticPartitionFailureError(
            "semantic-v5 control shortfall fields changed"
        )
    shortfall = V5ControlShortfall(
        world=_text(record, "world"),
        split=_text(record, "split"),
        arm=_text(record, "arm"),
        available_count=_integer(record, "available_count"),
        required_count=_integer(record, "required_count"),
    )
    if record.get("shortage_count") != (
        shortfall.required_count - shortfall.available_count
    ):
        raise V5SemanticPartitionFailureError(
            "semantic-v5 control shortage calculation changed"
        )
    return shortfall


def _markdown(
    failure_sha256: str,
    shortfall: V5ControlShortfall,
    topology_selection: Mapping[str, object],
) -> str:
    selected = _mapping(topology_selection, "selected")
    return "\n".join(
        (
            "# TinyWorlds-P semantic-v5 partition failure",
            "",
            f"Failure SHA-256: `{failure_sha256}`",
            "",
            "Version 5 successfully selected a layout whose five conditions "
            "all passed the frozen 10% text-balance rule.",
            "",
            f"The selected cells for A through E were `{selected['cells']}`. "
            f"Their active-token counts were `{selected['token_masses']}`.",
            "",
            f"Construction then stopped while matching the {shortfall.arm} "
            f"comparison for condition {shortfall.world} in the "
            f"{shortfall.split} split. It needed {shortfall.required_count:,} "
            f"distinct comparison-story groups, but only "
            f"{shortfall.available_count:,} were available. The shortage was "
            f"{shortfall.required_count - shortfall.available_count:,} groups.",
            "",
            "No tolerance was loosened and no different layout was substituted. "
            "No partition, sample report, GPU training run, or sealed-test result "
            "was published.",
            "",
        )
    )


def _html(markdown: str) -> str:
    paragraphs = "".join(
        f"<p>{escape(line)}</p>" for line in markdown.splitlines() if line
    )
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<title>TinyWorlds-P semantic-v5 partition failure</title><style>"
        "body{font-family:system-ui,sans-serif;max-width:76rem;margin:2rem auto;"
        "padding:0 1rem;line-height:1.5}code{font-family:ui-monospace,monospace}"
        "</style></head><body>"
        f"{paragraphs}</body></html>\n"
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
            key=lambda item: item.relative_to(root).as_posix(),
        )
        if path.is_file() and path != root / "tree.json"
    )
    _write_json(
        root / "tree.json",
        {
            "failure_sha256": failure_sha256,
            "files": list(files),
            "format": V5_PARTITION_FAILURE_TREE_FORMAT,
            "schema_version": V5_PARTITION_SCHEMA_VERSION,
        },
    )


def _validate_tree(root: Path, tree: Mapping[str, object]) -> None:
    raw = tree.get("files")
    if type(raw) is not list or any(type(item) is not dict for item in raw):
        raise V5SemanticPartitionFailureError(
            "semantic-v5 partition failure descriptors changed"
        )
    paths = tuple(_text(item, "relative_path") for item in raw)
    if paths != tuple(sorted(set(paths))):
        raise V5SemanticPartitionFailureError(
            "semantic-v5 partition failure paths are not canonical"
        )
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual != set(paths) | {"tree.json"}:
        raise V5SemanticPartitionFailureError(
            "semantic-v5 partition failure membership changed"
        )
    if any(path.is_symlink() for path in root.rglob("*")):
        raise V5SemanticPartitionFailureError(
            "semantic-v5 partition failure contains a symlink"
        )
    for descriptor in raw:
        relative = _text(descriptor, "relative_path")
        candidate = root / relative
        if (
            candidate.stat().st_size != _integer(descriptor, "size_bytes")
            or _file_sha256(candidate) != _sha_text(descriptor, "sha256")
        ):
            raise V5SemanticPartitionFailureError(
                f"semantic-v5 partition failure file changed: {relative}"
            )


def _validate_embedded_membership(
    root: Path,
    catalog_root: Path,
    parent_root: Path,
) -> None:
    top_level = {"audit.html", "audit.json", "audit.md", "failure.json", "tree.json"}
    catalog_prefix = catalog_root.relative_to(root).as_posix()
    parent_prefix = parent_root.relative_to(root).as_posix()
    expected = top_level | {
        f"{catalog_prefix}/{path.relative_to(catalog_root).as_posix()}"
        for path in catalog_root.rglob("*")
        if path.is_file()
    } | {
        f"{parent_prefix}/{path.relative_to(parent_root).as_posix()}"
        for path in parent_root.rglob("*")
        if path.is_file()
    }
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual != expected:
        raise V5SemanticPartitionFailureError(
            "semantic-v5 partition failure embedded membership changed"
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
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise V5SemanticPartitionFailureError(f"invalid {label}") from error
    if type(value) is not dict or canonical_json_bytes(value) != payload:
        raise V5SemanticPartitionFailureError(f"noncanonical {label}")
    return value


def _mapping(
    record: Mapping[str, object],
    field: str,
) -> dict[str, object]:
    value = record.get(field)
    if type(value) is not dict:
        raise V5SemanticPartitionFailureError(
            f"semantic-v5 partition failure field {field!r} must be a mapping"
        )
    return value


def _text(record: Mapping[str, object], field: str) -> str:
    value = record.get(field)
    if type(value) is not str or not value:
        raise V5SemanticPartitionFailureError(
            f"semantic-v5 partition failure field {field!r} must be text"
        )
    return value


def _sha_text(record: Mapping[str, object], field: str) -> str:
    value = _text(record, field)
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise V5SemanticPartitionFailureError(
            f"semantic-v5 partition failure field {field!r} must be SHA-256"
        )
    return value


def _integer(record: Mapping[str, object], field: str) -> int:
    value = record.get(field)
    if type(value) is not int or value < 0:
        raise V5SemanticPartitionFailureError(
            f"semantic-v5 partition failure field {field!r} must be nonnegative"
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
    "V5SemanticPartitionFailureEvidence",
    "V5SemanticPartitionFailureError",
    "load_v5_partition_failure",
    "load_v5_partition_failure_evidence",
    "parse_v5_control_shortfall",
    "publish_v5_partition_failure",
]

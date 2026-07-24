"""Authenticated failure evidence for the fixed semantic-v4 topology gate."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
from html import escape
import json
from math import isfinite
import os
from pathlib import Path
import shutil

from apm.data.text.tinyworlds_p import artifact as archive_artifact
from apm.data.text.tinyworlds_p.contracts import (
    NORMALIZATION_IDENTITY,
    WordBucket,
)
from apm.data.text.tinyworlds_p_semantic import artifact as shared_artifact
from apm.data.text.tinyworlds_p_semantic.builder import _bucket_record, _source_record
from apm.data.text.tinyworlds_p_semantic.contracts import (
    canonical_json_bytes,
    record_sha256,
    require_sha256,
)
from apm.data.text.tinyworlds_p_semantic.partitioning import SemanticTopologyAudit
from apm.data.text.tinyworlds_p_semantic.v4_contracts import (
    V4_BENCHMARK_ID,
    V4SemanticCatalog,
)
from apm.data.text.tinyworlds_p_semantic.v4_partition_contracts import (
    V4_PARTITION_FAILURE_FORMAT,
    V4_PARTITION_FAILURE_TREE_FORMAT,
    V4_PARTITION_SCHEMA_VERSION,
    V4SemanticPartitionFailure,
    V4SemanticPartitionInputs,
    V4SemanticPartitionPreset,
)


class V4SemanticPartitionFailureError(ValueError):
    """A semantic-v4 partition failure bundle is malformed or inconsistent."""


@dataclass(frozen=True, slots=True)
class V4SemanticPartitionFailureEvidence:
    """The parent adjective mapping, exclusions, and ranked topology records."""

    adjective_buckets: tuple[WordBucket, ...]
    partition_preset: dict[str, object]
    semantic_exclusions: dict[str, int]
    sources: dict[str, object]
    topology_candidates: tuple[dict[str, object], ...]


def load_v4_partition_failure_evidence(
    failure: V4SemanticPartitionFailure,
) -> V4SemanticPartitionFailureEvidence:
    """Load the strictly authenticated parent evidence needed by a successor."""
    if type(failure) is not V4SemanticPartitionFailure:
        raise TypeError("v4 partition evidence requires its strict failure artifact")
    restored = load_v4_partition_failure(failure.root)
    if restored.failure_sha256 != failure.failure_sha256:
        raise V4SemanticPartitionFailureError("v4 partition failure identity changed")
    record = _load_json(restored.root / "failure.json")
    raw_buckets = record.get("adjective_buckets")
    raw_exclusions = record.get("semantic_exclusions")
    raw_preset = record.get("preset")
    raw_sources = record.get("sources")
    if (
        type(raw_buckets) is not list
        or type(raw_exclusions) is not dict
        or type(raw_preset) is not dict
        or type(raw_sources) is not dict
    ):
        raise V4SemanticPartitionFailureError("v4 successor evidence changed")
    return V4SemanticPartitionFailureEvidence(
        adjective_buckets=tuple(
            shared_artifact._word_bucket(item) for item in raw_buckets
        ),
        partition_preset=dict(raw_preset),
        semantic_exclusions={
            str(name): int(value) for name, value in raw_exclusions.items()
        },
        sources=dict(raw_sources),
        topology_candidates=tuple(
            _iter_jsonl(restored.root / "topology-candidates.jsonl")
        ),
    )


def publish_v4_partition_failure(
    inputs: V4SemanticPartitionInputs,
    preset: V4SemanticPartitionPreset,
    catalog: V4SemanticCatalog,
    seed_identity_sha256: str,
    adjective_buckets: Sequence[WordBucket],
    semantic_exclusions: Mapping[str, int],
    audit: SemanticTopologyAudit,
    reason: str,
) -> V4SemanticPartitionFailure:
    """Publish and strictly reload one content-addressed topology stop."""
    if type(inputs) is not V4SemanticPartitionInputs:
        raise TypeError("semantic-v4 partition failure requires dedicated inputs")
    if type(preset) is not V4SemanticPartitionPreset:
        raise TypeError("semantic-v4 partition failure requires its frozen preset")
    if type(catalog) is not V4SemanticCatalog:
        raise TypeError("semantic-v4 partition failure requires its strict catalog")
    if type(audit) is not SemanticTopologyAudit:
        raise TypeError("semantic-v4 partition failure requires a topology audit")
    require_sha256(seed_identity_sha256, "semantic-v4 partition failure seed")
    if type(reason) is not str or not reason:
        raise ValueError("semantic-v4 partition failure requires a reason")
    selected = audit.selected
    if selected is None or selected.passes_median_gate(audit.median_tolerance):
        raise ValueError("semantic-v4 failure publication requires a failed winner")

    working = inputs.temporary_directory / "partition-failure-publication"
    if working.exists():
        raise FileExistsError(f"semantic-v4 failure staging path exists: {working}")
    working.mkdir(parents=True)
    with (working / "topology-candidates.jsonl").open("wb") as output:
        for candidate in audit.candidates:
            output.write(
                canonical_json_bytes(candidate.as_record(audit.median_tolerance))
            )
        output.flush()
        os.fsync(output.fileno())

    audit_record = _audit_summary(audit)
    _write_json(working / "audit.json", audit_record)
    sources = _source_record(inputs, catalog)
    content = {
        "adjective_buckets": [_bucket_record(item) for item in adjective_buckets],
        "audit_sha256": _file_sha256(working / "audit.json"),
        "benchmark_id": V4_BENCHMARK_ID,
        "catalog_sha256": catalog.catalog_sha256,
        "failure_sha256": "",
        "format": V4_PARTITION_FAILURE_FORMAT,
        "normalization": NORMALIZATION_IDENTITY.as_record(),
        "preset": preset.as_record(),
        "reason": reason,
        "schema_version": V4_PARTITION_SCHEMA_VERSION,
        "seed_identity_sha256": seed_identity_sha256,
        "semantic_exclusions": dict(sorted(semantic_exclusions.items())),
        "sources": sources,
        "topology_candidates_sha256": _file_sha256(
            working / "topology-candidates.jsonl"
        ),
    }
    failure_sha = record_sha256(
        {key: value for key, value in content.items() if key != "failure_sha256"}
    )
    content["failure_sha256"] = failure_sha
    _write_json(working / "failure.json", content)
    markdown = _markdown(failure_sha, reason, audit_record)
    _write_text(working / "audit.md", markdown)
    _write_text(working / "audit.html", _html(markdown))
    _write_tree(working, failure_sha)

    failure_root = inputs.output_root / "failures"
    target = failure_root / failure_sha
    if target.exists():
        if _tree_bytes(working) != _tree_bytes(target):
            raise RuntimeError(
                "semantic-v4 partition failure rebuild is not byte-identical"
            )
        shutil.rmtree(working)
        return load_v4_partition_failure(target)
    failure_root.mkdir(parents=True, exist_ok=True)
    os.rename(working, target)
    _fsync_directory(failure_root)
    return load_v4_partition_failure(target)


def load_v4_partition_failure(path: str | Path) -> V4SemanticPartitionFailure:
    """Strictly authenticate and replay a semantic-v4 topology failure."""
    try:
        root = Path(path)
        if root.is_symlink() or not root.is_dir():
            raise V4SemanticPartitionFailureError(
                "semantic-v4 partition failure must be a regular directory"
            )
        tree = _load_json(root / "tree.json")
        if (
            set(tree) != {"failure_sha256", "files", "format", "schema_version"}
            or tree.get("format") != V4_PARTITION_FAILURE_TREE_FORMAT
            or tree.get("schema_version") != V4_PARTITION_SCHEMA_VERSION
        ):
            raise V4SemanticPartitionFailureError(
                "semantic-v4 partition failure tree changed"
            )
        failure_sha = _text(tree, "failure_sha256")
        require_sha256(failure_sha, "semantic-v4 partition failure")
        if root.name != failure_sha:
            raise V4SemanticPartitionFailureError(
                "semantic-v4 partition failure directory identity changed"
            )
        _validate_tree_files(root, tree)
        failure = _load_json(root / "failure.json")
        required = {
            "adjective_buckets",
            "audit_sha256",
            "benchmark_id",
            "catalog_sha256",
            "failure_sha256",
            "format",
            "normalization",
            "preset",
            "reason",
            "schema_version",
            "seed_identity_sha256",
            "semantic_exclusions",
            "sources",
            "topology_candidates_sha256",
        }
        if set(failure) != required:
            raise V4SemanticPartitionFailureError(
                "semantic-v4 partition failure fields changed"
            )
        identity_content = {
            key: value for key, value in failure.items() if key != "failure_sha256"
        }
        if (
            failure.get("failure_sha256") != failure_sha
            or record_sha256(identity_content) != failure_sha
            or failure.get("benchmark_id") != V4_BENCHMARK_ID
            or failure.get("format") != V4_PARTITION_FAILURE_FORMAT
            or failure.get("schema_version") != V4_PARTITION_SCHEMA_VERSION
            or failure.get("normalization") != NORMALIZATION_IDENTITY.as_record()
        ):
            raise V4SemanticPartitionFailureError(
                "semantic-v4 partition failure identity changed"
            )
        median_tolerance = _validate_sources(failure)
        if _file_sha256(root / "audit.json") != _text(failure, "audit_sha256"):
            raise V4SemanticPartitionFailureError("semantic-v4 failure audit changed")
        if _file_sha256(root / "topology-candidates.jsonl") != _text(
            failure, "topology_candidates_sha256"
        ):
            raise V4SemanticPartitionFailureError(
                "semantic-v4 failure candidates changed"
            )
        audit = _load_json(root / "audit.json")
        candidates = tuple(_iter_jsonl(root / "topology-candidates.jsonl"))
        _validate_audit(audit, candidates, median_tolerance)
        reason = _text(failure, "reason")
        if reason != "best semantic topology violates the selected-cell token median gate":
            raise V4SemanticPartitionFailureError(
                "semantic-v4 partition stopped for an unregistered reason"
            )
        return V4SemanticPartitionFailure(
            root=root.resolve(),
            failure_sha256=failure_sha,
            catalog_sha256=_text(failure, "catalog_sha256"),
            seed_identity_sha256=_text(failure, "seed_identity_sha256"),
            reason=reason,
            audit=audit,
        )
    except V4SemanticPartitionFailureError:
        raise
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise V4SemanticPartitionFailureError(
            "semantic-v4 partition failure payload changed"
        ) from error


def _audit_summary(audit: SemanticTopologyAudit) -> dict[str, object]:
    selected = audit.selected
    assert selected is not None
    median_feasible = audit.median_feasible_candidates
    return {
        "control_capable_candidate_count": audit.control_capable_candidate_count,
        "diagnostic_best_median_feasible": (
            median_feasible[0].as_record(audit.median_tolerance)
            if median_feasible
            else None
        ),
        "median_feasible_candidate_count": len(median_feasible),
        "nonempty_candidate_count": audit.nonempty_candidate_count,
        "physical_candidate_count": audit.physical_candidate_count,
        "selected": selected.as_record(audit.median_tolerance),
        "selection_policy": (
            "semantic-dispersion,token-imbalance,nuisance-imbalance,"
            "negative-control-capacity,canonical-hash; median gate after selection"
        ),
        "visible_candidate_count": audit.visible_candidate_count,
    }


def _validate_audit(
    audit: Mapping[str, object],
    candidates: Sequence[Mapping[str, object]],
    median_tolerance: float,
) -> None:
    required = {
        "control_capable_candidate_count",
        "diagnostic_best_median_feasible",
        "median_feasible_candidate_count",
        "nonempty_candidate_count",
        "physical_candidate_count",
        "selected",
        "selection_policy",
        "visible_candidate_count",
    }
    if set(audit) != required or not candidates:
        raise V4SemanticPartitionFailureError("semantic-v4 failure audit fields changed")
    counts = tuple(
        _integer(audit, name)
        for name in (
            "physical_candidate_count",
            "nonempty_candidate_count",
            "visible_candidate_count",
            "control_capable_candidate_count",
        )
    )
    if counts[0] != 28_224 or tuple(sorted(counts, reverse=True)) != counts:
        raise V4SemanticPartitionFailureError(
            "semantic-v4 topology screen counts changed"
        )
    if counts[-1] != len(candidates):
        raise V4SemanticPartitionFailureError(
            "semantic-v4 control-capable candidate count changed"
        )
    scores = tuple(_candidate_score(item) for item in candidates)
    if scores != tuple(sorted(scores)):
        raise V4SemanticPartitionFailureError(
            "semantic-v4 topology ranking changed"
        )
    selected = _mapping(audit, "selected")
    if selected != candidates[0] or _candidate_passes_median(
        selected,
        median_tolerance,
    ):
        raise V4SemanticPartitionFailureError(
            "semantic-v4 selected topology no longer demonstrates failure"
        )
    feasible = tuple(
        item
        for item in candidates
        if _candidate_passes_median(item, median_tolerance)
    )
    if _integer(audit, "median_feasible_candidate_count") != len(feasible):
        raise V4SemanticPartitionFailureError(
            "semantic-v4 median-feasible candidate count changed"
        )
    diagnostic = audit.get("diagnostic_best_median_feasible")
    expected = feasible[0] if feasible else None
    if diagnostic != expected:
        raise V4SemanticPartitionFailureError(
            "semantic-v4 diagnostic balanced candidate changed"
        )


def _candidate_score(
    record: Mapping[str, object],
) -> tuple[float, Fraction, Fraction, Fraction, str]:
    required = {
        "cells",
        "control_capacity",
        "group_counts",
        "median_gate",
        "nuisance_imbalance",
        "semantic_dispersion",
        "tie_sha256",
        "token_imbalance",
        "token_masses",
    }
    if set(record) != required:
        raise V4SemanticPartitionFailureError(
            "semantic-v4 topology candidate fields changed"
        )
    dispersion = record.get("semantic_dispersion")
    if (
        type(dispersion) not in (int, float)
        or not isfinite(float(dispersion))
        or not 0.0 <= float(dispersion) <= 2.0
    ):
        raise V4SemanticPartitionFailureError(
            "semantic-v4 topology dispersion changed"
        )
    tie = _text(record, "tie_sha256")
    require_sha256(tie, "semantic-v4 topology tie")
    cells = record.get("cells")
    masses = record.get("token_masses")
    groups = record.get("group_counts")
    if (
        type(cells) is not list
        or len(cells) != 5
        or any(
            type(cell) is not list
            or len(cell) != 2
            or any(type(value) is not int or not 0 <= value < 8 for value in cell)
            for cell in cells
        )
        or type(masses) is not list
        or len(masses) != 5
        or any(type(value) is not int or value <= 0 for value in masses)
        or type(groups) is not list
        or len(groups) != 5
        or any(type(value) is not int or value <= 0 for value in groups)
    ):
        raise V4SemanticPartitionFailureError(
            "semantic-v4 topology candidate shape changed"
        )
    cell_tuple = tuple(tuple(cell) for cell in cells)
    a, b, c, d, e = cell_tuple
    if (
        len(set(cell_tuple)) != 5
        or not (
            a[0] == d[0]
            and b[0] == c[0]
            and a[0] != b[0]
            and a[1] == b[1]
            and c[1] == d[1]
            and a[1] != c[1]
            and e[0] not in (a[0], b[0])
            and e[1] not in (a[1], c[1])
        )
    ):
        raise V4SemanticPartitionFailureError(
            "semantic-v4 topology candidate geometry changed"
        )
    token_imbalance = _fraction(_mapping(record, "token_imbalance"))
    nuisance_imbalance = _fraction(_mapping(record, "nuisance_imbalance"))
    control_capacity = _fraction(_mapping(record, "control_capacity"))
    if control_capacity <= 0:
        raise V4SemanticPartitionFailureError(
            "semantic-v4 topology control capacity changed"
        )
    return (
        float(dispersion),
        token_imbalance,
        nuisance_imbalance,
        -control_capacity,
        tie,
    )


def _candidate_passes_median(
    record: Mapping[str, object],
    tolerance: float,
) -> bool:
    gate = _mapping(record, "median_gate")
    required = {
        "lower_token_mass",
        "median_token_mass",
        "passes",
        "tolerance",
        "upper_token_mass",
    }
    masses = record.get("token_masses")
    if set(gate) != required or type(masses) is not list:
        raise V4SemanticPartitionFailureError("semantic-v4 median gate changed")
    persisted_tolerance = gate.get("tolerance")
    median = gate.get("median_token_mass")
    passed = gate.get("passes")
    if (
        persisted_tolerance != tolerance
        or type(median) is not int
        or type(passed) is not bool
    ):
        raise V4SemanticPartitionFailureError("semantic-v4 median gate changed")
    expected_median = sorted(masses)[2]
    lower = expected_median * (1.0 - tolerance)
    upper = expected_median * (1.0 + tolerance)
    expected_passed = all(lower <= mass <= upper for mass in masses)
    if (
        median != expected_median
        or gate.get("lower_token_mass") != lower
        or gate.get("upper_token_mass") != upper
        or passed != expected_passed
    ):
        raise V4SemanticPartitionFailureError(
            "semantic-v4 candidate median calculation changed"
        )
    return passed


def _validate_sources(failure: Mapping[str, object]) -> float:
    sources = _mapping(failure, "sources")
    catalog = _mapping(sources, "semantic_catalog")
    if (
        set(sources) != {"archive", "semantic_catalog", "tokenizer"}
        or catalog.get("catalog_sha256") != failure.get("catalog_sha256")
        or set(catalog)
        != {"catalog_sha256", "encoder_identity_sha256", "evidence_sha256"}
    ):
        raise V4SemanticPartitionFailureError(
            "semantic-v4 partition failure sources changed"
        )
    require_sha256(_text(catalog, "catalog_sha256"), "semantic-v4 failure catalog")
    require_sha256(_text(catalog, "evidence_sha256"), "semantic-v4 failure evidence")
    require_sha256(
        _text(catalog, "encoder_identity_sha256"),
        "semantic-v4 failure encoder",
    )
    require_sha256(
        _text(failure, "seed_identity_sha256"),
        "semantic-v4 partition failure seed",
    )
    try:
        archive_artifact._source_identity(_mapping(sources, "archive"))
        archive_artifact._tokenizer_identity(_mapping(sources, "tokenizer"))
        from apm.data.text.tinyworlds_p_semantic.v4_partition_artifact import _preset

        preset = _preset(_mapping(failure, "preset"))
    except (TypeError, ValueError, archive_artifact.PartitionArtifactError) as error:
        raise V4SemanticPartitionFailureError(
            "semantic-v4 partition failure source or preset is invalid"
        ) from error
    expected_seed = record_sha256(
        {
            "benchmark_id": V4_BENCHMARK_ID,
            "normalization": NORMALIZATION_IDENTITY.as_record(),
            "preset": preset.as_record(),
            "sources": sources,
        }
    )
    if failure.get("seed_identity_sha256") != expected_seed:
        raise V4SemanticPartitionFailureError(
            "semantic-v4 partition failure seed identity changed"
        )
    raw_buckets = failure.get("adjective_buckets")
    if type(raw_buckets) is not list:
        raise V4SemanticPartitionFailureError(
            "semantic-v4 partition failure adjective buckets changed"
        )
    buckets = tuple(shared_artifact._word_bucket(item) for item in raw_buckets)
    if tuple(item.index for item in buckets) != tuple(range(len(buckets))):
        raise V4SemanticPartitionFailureError(
            "semantic-v4 partition failure adjective buckets are incomplete"
        )
    exclusions = failure.get("semantic_exclusions")
    if (
        type(exclusions) is not dict
        or not exclusions
        or any(
            type(name) is not str
            or not name
            or type(value) is not int
            or value < 0
            for name, value in exclusions.items()
        )
        or exclusions.get("retained_tokens", 0) <= 0
    ):
        raise V4SemanticPartitionFailureError(
            "semantic-v4 partition failure exclusions changed"
        )
    return preset.selected_cell_median_tolerance


def _markdown(
    failure_sha: str,
    reason: str,
    audit: Mapping[str, object],
) -> str:
    selected = _mapping(audit, "selected")
    gate = _mapping(selected, "median_gate")
    diagnostic = audit.get("diagnostic_best_median_feasible")
    lines = [
        "# TinyWorlds-P semantic-v4 partition failure",
        "",
        f"- Failure SHA-256: `{failure_sha}`",
        f"- Reason: {reason}",
        f"- Physical candidates: {_integer(audit, 'physical_candidate_count'):,}",
        f"- Visibility-passing candidates: {_integer(audit, 'visible_candidate_count'):,}",
        f"- Control-capable candidates: {_integer(audit, 'control_capable_candidate_count'):,}",
        "- Median-feasible candidates (diagnostic only): "
        f"{_integer(audit, 'median_feasible_candidate_count'):,}",
        "",
        "## Preregistered winner",
        "",
        f"- Cells A--E: `{selected['cells']}`",
        f"- Active-token masses: `{selected['token_masses']}`",
        f"- Median: `{gate['median_token_mass']}`; allowed interval: "
        f"`[{gate['lower_token_mass']}, {gate['upper_token_mass']}]`",
        f"- Semantic dispersion: `{selected['semantic_dispersion']}`",
        f"- Token imbalance: `{selected['token_imbalance']}`",
        f"- Nuisance imbalance: `{selected['nuisance_imbalance']}`",
        f"- Control capacity: `{selected['control_capacity']}`",
        "",
        "The winner failed the frozen downstream mass gate, so no partition, "
        "sample report, GPU preflight, calibration, or sealed-test access was produced.",
    ]
    if type(diagnostic) is dict:
        lines.extend(
            (
                "",
                "## Diagnostic balanced candidate",
                "",
                f"Cells A--E: `{diagnostic['cells']}`; masses: "
                f"`{diagnostic['token_masses']}`.",
                "",
                "This is recorded only to audit the screen. Substituting it after "
                "observing the preregistered winner would change v4's selection rule.",
            )
        )
    lines.extend(("", "All ranked candidates are in `topology-candidates.jsonl`.", ""))
    return "\n".join(lines)


def _html(markdown: str) -> str:
    body = "\n".join(f"<p>{escape(line)}</p>" for line in markdown.splitlines())
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>"
        "TinyWorlds-P semantic-v4 partition failure</title><style>"
        "body{font-family:system-ui,sans-serif;max-width:76rem;margin:2rem auto;"
        "padding:0 1rem;line-height:1.45}p{white-space:pre-wrap}"
        "</style></head><body>"
        f"{body}</body></html>\n"
    )


def _write_tree(root: Path, failure_sha: str) -> None:
    files = tuple(
        {
            "relative_path": path.relative_to(root).as_posix(),
            "sha256": _file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(root.iterdir(), key=lambda item: item.name)
        if path.is_file() and path.name != "tree.json"
    )
    _write_json(
        root / "tree.json",
        {
            "failure_sha256": failure_sha,
            "files": list(files),
            "format": V4_PARTITION_FAILURE_TREE_FORMAT,
            "schema_version": V4_PARTITION_SCHEMA_VERSION,
        },
    )


def _validate_tree_files(root: Path, tree: Mapping[str, object]) -> None:
    raw = tree.get("files")
    if type(raw) is not list or any(type(item) is not dict for item in raw):
        raise V4SemanticPartitionFailureError(
            "semantic-v4 partition failure descriptors changed"
        )
    expected = (
        "audit.html",
        "audit.json",
        "audit.md",
        "failure.json",
        "topology-candidates.jsonl",
    )
    names = tuple(_text(item, "relative_path") for item in raw)
    if names != expected:
        raise V4SemanticPartitionFailureError(
            "semantic-v4 partition failure files are incomplete"
        )
    actual = tuple(
        path.name
        for path in sorted(root.iterdir(), key=lambda item: item.name)
        if path.name != "tree.json"
    )
    if actual != expected:
        raise V4SemanticPartitionFailureError(
            "semantic-v4 partition failure has unknown files"
        )
    for descriptor in raw:
        candidate = root / _text(descriptor, "relative_path")
        size = descriptor.get("size_bytes")
        if (
            candidate.is_symlink()
            or not candidate.is_file()
            or type(size) is not int
            or size < 0
            or candidate.stat().st_size != size
            or _file_sha256(candidate) != _text(descriptor, "sha256")
        ):
            raise V4SemanticPartitionFailureError(
                "semantic-v4 partition failure file changed"
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
    payload = path.read_bytes()
    value = json.loads(payload)
    if type(value) is not dict or canonical_json_bytes(value) != payload:
        raise V4SemanticPartitionFailureError(
            f"noncanonical semantic-v4 partition failure JSON: {path}"
        )
    return value


def _iter_jsonl(path: Path) -> Iterator[dict[str, object]]:
    with path.open("rb") as source:
        for line in source:
            value = json.loads(line)
            if type(value) is not dict or canonical_json_bytes(value) != line:
                raise V4SemanticPartitionFailureError(
                    "noncanonical semantic-v4 topology candidate"
                )
            yield value


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _mapping(record: Mapping[str, object], field: str) -> dict[str, object]:
    value = record.get(field)
    if type(value) is not dict:
        raise V4SemanticPartitionFailureError(
            f"semantic-v4 partition failure field {field!r} must be a mapping"
        )
    return value


def _text(record: Mapping[str, object], field: str) -> str:
    value = record.get(field)
    if type(value) is not str or not value:
        raise V4SemanticPartitionFailureError(
            f"semantic-v4 partition failure field {field!r} must be text"
        )
    return value


def _integer(record: Mapping[str, object], field: str) -> int:
    value = record.get(field)
    if type(value) is not int or value < 0:
        raise V4SemanticPartitionFailureError(
            f"semantic-v4 partition failure field {field!r} must be nonnegative"
        )
    return value


def _fraction(record: Mapping[str, object]) -> Fraction:
    if set(record) != {"denominator", "numerator"}:
        raise V4SemanticPartitionFailureError(
            "semantic-v4 topology fraction fields changed"
        )
    denominator = _integer(record, "denominator")
    numerator = _integer(record, "numerator")
    if denominator <= 0:
        raise V4SemanticPartitionFailureError(
            "semantic-v4 topology fraction denominator changed"
        )
    value = Fraction(numerator, denominator)
    if value.numerator != numerator or value.denominator != denominator:
        raise V4SemanticPartitionFailureError(
            "semantic-v4 topology fraction is not reduced"
        )
    return value


__all__ = [
    "V4SemanticPartitionFailureEvidence",
    "V4SemanticPartitionFailureError",
    "load_v4_partition_failure_evidence",
    "load_v4_partition_failure",
    "publish_v4_partition_failure",
]

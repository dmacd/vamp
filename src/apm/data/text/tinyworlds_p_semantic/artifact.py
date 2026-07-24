"""Strict loading and independent semantic validation of partition artifacts."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path

from apm.data.text.tinyworlds_p import artifact as archive_artifact
from apm.data.text.tinyworlds_p.contracts import (
    NORMALIZATION_IDENTITY,
    ControlSelection,
    SplitCount,
    WordBucket,
    WorldCell,
)
from apm.data.text.tinyworlds_p_semantic.builder import (
    _bucket_record,
    _cell_record,
    _control_record,
    _pair_record,
    _source_record,
)
from apm.data.text.tinyworlds_p_semantic.catalog import (
    SemanticCatalogError,
    load_semantic_catalog,
)
from apm.data.text.tinyworlds_p_semantic.contracts import (
    BENCHMARK_ID,
    PARTITION_FORMAT,
    SCHEMA_VERSION,
    WORLD_LABELS,
    ControlPair,
    SemanticCatalog,
    SemanticPartitionArtifact,
    SemanticPartitionInputs,
    SemanticPartitionPreset,
    canonical_json_bytes,
    record_sha256,
)
from apm.data.text.tinyworlds_p_semantic.evidence import is_construction_group


_CORE_FILES = {
    "assignments.jsonl",
    "audit.json",
    "buckets.json",
    "controls.json",
    "documents.jsonl",
    "normalization.json",
    "pairings.json",
    "partition.json",
    "shards.json",
    "sources.json",
    "topology.json",
    "tree.json",
    "manifests/base.json",
    "manifests/controls.json",
    *(f"manifests/world-{world}.json" for world in WORLD_LABELS),
    *(f"indexes/base-{split}.jsonl" for split in ("train", "validation", "test")),
    *(
        f"indexes/world-{world}-{split}.jsonl"
        for world in WORLD_LABELS
        for split in ("train", "validation", "test")
    ),
    *(
        f"indexes/control-{world}-{split}.jsonl"
        for world in WORLD_LABELS
        for split in ("validation", "test")
    ),
}
_PARALLEL_VALIDATION_MIN_BYTES = 256 * 1024 * 1024
_MAX_PARALLEL_WORKERS = max(1, (os.cpu_count() or 1) * 3 // 4)


class PartitionArtifactError(ValueError):
    """A semantic partition tree or source/catalog binding is invalid."""


def load_partition(path: str | Path) -> SemanticPartitionArtifact:
    """Authenticate the complete tree, embedded catalog, assignments, and shards."""
    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        raise PartitionArtifactError("semantic partition must be a regular directory")
    tree_path = root / "tree.json"
    tree = _load_json(tree_path, "semantic partition tree")
    if set(tree) != {"files", "format", "partition_sha256", "schema_version"}:
        raise PartitionArtifactError("semantic partition tree fields changed")
    if tree["format"] != "tinyworlds-p-semantic-tree" or tree["schema_version"] != 1:
        raise PartitionArtifactError("unsupported semantic partition tree format")
    partition_sha = _text(tree, "partition_sha256")
    if root.name != partition_sha:
        raise PartitionArtifactError("semantic partition directory identity changed")
    _validate_tree(root, tree)
    partition = _load_json(root / "partition.json", "semantic partition identity")
    required = {
        "assignments_sha256",
        "benchmark_id",
        "eos_token_id",
        "format",
        "normalization",
        "pad_token_id",
        "partition_sha256",
        "preset",
        "schema_version",
        "seed_identity_sha256",
        "sources",
    }
    if set(partition) != required:
        raise PartitionArtifactError("semantic partition identity fields changed")
    if (
        partition["benchmark_id"] != BENCHMARK_ID
        or partition["format"] != PARTITION_FORMAT
        or partition["schema_version"] != SCHEMA_VERSION
        or partition["partition_sha256"] != partition_sha
    ):
        raise PartitionArtifactError("semantic partition identity contract changed")
    sources = _mapping(partition, "sources")
    if set(sources) != {"archive", "semantic_catalog", "tokenizer"}:
        raise PartitionArtifactError(
            "semantic partition sources must be archive, tokenizer, and semantic_catalog"
        )
    if _load_json(root / "sources.json", "semantic sources") != sources:
        raise PartitionArtifactError("semantic partition source files disagree")
    if _load_json(root / "normalization.json", "normalization") != partition["normalization"]:
        raise PartitionArtifactError("semantic normalization files disagree")
    try:
        archive_identity = archive_artifact._source_identity(_mapping(sources, "archive"))
        tokenizer_identity = archive_artifact._tokenizer_identity(_mapping(sources, "tokenizer"))
        normalization = archive_artifact._normalization(_mapping(partition, "normalization"))
        preset = _preset(_mapping(partition, "preset"))
    except (TypeError, ValueError, archive_artifact.PartitionArtifactError) as error:
        raise PartitionArtifactError("semantic partition source or preset is invalid") from error
    catalog_source = _mapping(sources, "semantic_catalog")
    if set(catalog_source) != {
        "catalog_sha256",
        "encoder_identity_sha256",
        "evidence_sha256",
    }:
        raise PartitionArtifactError("semantic catalog source fields changed")
    catalog_sha = _text(catalog_source, "catalog_sha256")
    catalog_root = root / "semantic-catalog" / catalog_sha
    try:
        catalog = load_semantic_catalog(catalog_root)
    except SemanticCatalogError as error:
        raise PartitionArtifactError("embedded semantic catalog failed authentication") from error
    if (
        catalog.encoder_identity.identity_sha256
        != _text(catalog_source, "encoder_identity_sha256")
        or catalog.evidence_sha256 != _text(catalog_source, "evidence_sha256")
    ):
        raise PartitionArtifactError("semantic catalog encoder/evidence binding changed")
    bucket_payload = _load_json(root / "buckets.json", "semantic adjective buckets")
    if set(bucket_payload) != {"adjective_buckets", "catalog_sha256"} or bucket_payload[
        "catalog_sha256"
    ] != catalog_sha:
        raise PartitionArtifactError("semantic bucket/catalog binding changed")
    raw_buckets = bucket_payload["adjective_buckets"]
    if type(raw_buckets) is not list:
        raise PartitionArtifactError("semantic adjective buckets must be a list")
    adjective_buckets = tuple(_word_bucket(item) for item in raw_buckets)
    if tuple(item.index for item in adjective_buckets) != tuple(range(catalog.config.cluster_count)):
        raise PartitionArtifactError("semantic adjective buckets are incomplete")
    topology = _load_json(root / "topology.json", "semantic topology")
    if set(topology) != {"cells"} or type(topology["cells"]) is not list:
        raise PartitionArtifactError("semantic topology fields changed")
    cells = tuple(_world_cell(item) for item in topology["cells"])
    _validate_topology(cells, catalog.config.cluster_count)
    control_payload = _load_json(root / "controls.json", "semantic controls")
    if set(control_payload) != {"controls"} or type(control_payload["controls"]) is not list:
        raise PartitionArtifactError("semantic controls fields changed")
    controls = tuple(_control(item) for item in control_payload["controls"])
    _validate_controls(controls)
    pairing_payload = _load_json(root / "pairings.json", "semantic pairings")
    if set(pairing_payload) != {"pairings"} or type(pairing_payload["pairings"]) is not list:
        raise PartitionArtifactError("semantic pairings fields changed")
    pairings = tuple(_pair(item) for item in pairing_payload["pairings"])
    audit = _load_json(root / "audit.json", "semantic partition audit")
    required_audit = {
        "archive_ingest",
        "component_visibility",
        "semantic_exclusions",
        "split_counts",
    }
    if set(audit) != required_audit or type(audit["split_counts"]) is not list:
        raise PartitionArtifactError("semantic partition audit fields changed")
    split_counts = tuple(_split_count(item) for item in audit["split_counts"])
    assignments_sha = _text(partition, "assignments_sha256")
    if _file_sha256(root / "assignments.jsonl") != assignments_sha:
        raise PartitionArtifactError("semantic assignment checksum changed")
    assignment_index = _validate_assignments(
        root / "assignments.jsonl",
        catalog,
        cells,
        controls,
        split_counts,
        pairings,
        _mapping(audit, "semantic_exclusions"),
    )
    _validate_pairings(pairings, controls, cells, assignment_index)
    try:
        archive_artifact._validate_documents_against_assignments(root)
        archive_artifact._validate_document_storage_and_indexes(
            root,
            controls,
            _integer(partition, "eos_token_id"),
            tokenizer_identity.vocab_size,
        )
    except (ValueError, OSError) as error:
        raise PartitionArtifactError("semantic partition document/index proof failed") from error
    pairings_sha = record_sha256([_pair_record(item) for item in pairings])
    expected_partition_sha = record_sha256(
        {
            "adjective_buckets": [_bucket_record(item) for item in adjective_buckets],
            "assignments_sha256": assignments_sha,
            "benchmark_id": BENCHMARK_ID,
            "cells": [_cell_record(item) for item in cells],
            "controls": [_control_record(item) for item in controls],
            "normalization": normalization.as_record(),
            "pairings_sha256": pairings_sha,
            "preset": preset.as_record(),
            "sources": sources,
        }
    )
    if expected_partition_sha != partition_sha:
        raise PartitionArtifactError("semantic partition content identity is inconsistent")
    expected_seed = record_sha256(
        {
            "benchmark_id": BENCHMARK_ID,
            "normalization": normalization.as_record(),
            "preset": preset.as_record(),
            "sources": sources,
        }
    )
    if partition["seed_identity_sha256"] != expected_seed:
        raise PartitionArtifactError("semantic partition seed identity changed")
    return SemanticPartitionArtifact(
        root=root.resolve(),
        partition_sha256=partition_sha,
        manifest_sha256=_file_sha256(tree_path),
        archive_identity=archive_identity,
        tokenizer_identity=tokenizer_identity,
        semantic_catalog=catalog,
        normalization=normalization,
        preset=preset,
        cells=cells,
        controls=controls,
        pairings=pairings,
        split_counts=split_counts,
        pad_token_id=_integer(partition, "pad_token_id"),
        eos_token_id=_integer(partition, "eos_token_id"),
    )


def _validate_assignments(
    path: Path,
    catalog: SemanticCatalog,
    cells: Sequence[WorldCell],
    controls: Sequence[ControlSelection],
    split_counts: Sequence[SplitCount],
    pairings: Sequence[ControlPair],
    expected_exclusions: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    selected = {(cell.noun_bucket, cell.verb_bucket): cell.label for cell in cells}
    nouns, verbs = catalog.word_cluster("noun"), catalog.word_cluster("verb")
    control_owner = {
        digest: (control.world, control.split)
        for control in controls
        for digest in control.group_sha256
    }
    if len(control_owner) != sum(len(item.group_sha256) for item in controls):
        raise PartitionArtifactError("semantic controls reuse a group globally")
    expected_counts = {
        (item.role, item.world, item.split): item
        for item in split_counts
    }
    group_counts: Counter[tuple[str, str | None, str]] = Counter()
    occurrence_counts: Counter[tuple[str, str | None, str]] = Counter()
    token_counts: Counter[tuple[str, str | None, str]] = Counter()
    exclusion_counts: Counter[str] = Counter()
    required_assignments = {
        digest
        for pair in pairings
        for digest in (pair.world_group_sha256, pair.control_group_sha256)
    }
    result: dict[str, dict[str, object]] = {}
    previous = ""
    for record in _iter_jsonl(path):
        required = {
            "active_token_count",
            "adjective_bucket",
            "canonical_token_count",
            "excluded_recipe",
            "normalized_story_sha256",
            "noun_bucket",
            "provenance",
            "recipe",
            "record_ids",
            "role",
            "split",
            "status",
            "verb_bucket",
            "world",
        }
        if set(record) != required:
            raise PartitionArtifactError("semantic assignment fields changed")
        digest = _text(record, "normalized_story_sha256")
        if digest <= previous:
            raise PartitionArtifactError("semantic assignments are not unique and sorted")
        previous = digest
        status = record.get("status")
        record_ids = record.get("record_ids")
        if type(record_ids) is not list or record_ids != sorted(set(record_ids)):
            raise PartitionArtifactError("semantic assignment record IDs changed")
        if status == "eligible":
            if record.get("excluded_recipe") is not None:
                raise PartitionArtifactError("retained semantic assignment has an exclusion recipe")
            recipe = _mapping(record, "recipe")
            noun, verb = _text(recipe, "noun"), _text(recipe, "verb")
            if is_construction_group(digest, catalog.config):
                raise PartitionArtifactError("semantic construction group leaked into a split")
            noun_bucket, verb_bucket = _integer(record, "noun_bucket"), _integer(record, "verb_bucket")
            if nouns.get(noun) != noun_bucket or verbs.get(verb) != verb_bucket:
                raise PartitionArtifactError("assignment differs from semantic catalog clusters")
            role, split = _text(record, "role"), _text(record, "split")
            world = record.get("world")
            selected_world = selected.get((noun_bucket, verb_bucket))
            if role == "base" and (world is not None or selected_world is not None):
                raise PartitionArtifactError("semantic world cell leaked into held-in base")
            if role == "world" and world != selected_world:
                raise PartitionArtifactError("semantic world assignment differs from topology")
            if role not in ("base", "world") or split not in ("train", "validation", "test"):
                raise PartitionArtifactError("semantic assignment role or split is invalid")
            key = (role, world, split)
            group_counts[key] += 1
            occurrence_counts[key] += len(record_ids)
            active_tokens = _integer(record, "active_token_count")
            token_counts[key] += active_tokens
            exclusion_counts["retained_groups"] += 1
            exclusion_counts["retained_occurrences"] += len(record_ids)
            exclusion_counts["retained_tokens"] += active_tokens
            if digest in control_owner and (role, split) != ("base", control_owner[digest][1]):
                raise PartitionArtifactError("semantic control is outside its held-in split")
        elif status in ("semantic_construction", "semantic_word_exclusion"):
            recipe = _mapping(record, "excluded_recipe")
            construction = is_construction_group(digest, catalog.config)
            if (status == "semantic_construction") != construction:
                raise PartitionArtifactError("semantic construction status changed")
            if status == "semantic_word_exclusion" and (
                _text(recipe, "noun") in nouns and _text(recipe, "verb") in verbs
            ):
                raise PartitionArtifactError("retained catalog words were marked excluded")
            exclusion_counts[f"{status}_groups"] += 1
            exclusion_counts[f"{status}_tokens"] += _integer(
                record, "active_token_count"
            )
            _require_unassigned(record)
        elif status in (
            "empty_story",
            "unclassifiable_metadata",
            "conflicting_metadata",
        ):
            if record.get("excluded_recipe") is not None:
                raise PartitionArtifactError("source exclusion acquired a semantic recipe")
            _require_unassigned(record)
        else:
            raise PartitionArtifactError("semantic assignment status is invalid")
        if digest in required_assignments:
            result[digest] = record
    for key, expected in expected_counts.items():
        if (
            group_counts[key] != expected.group_count
            or occurrence_counts[key] != expected.occurrence_count
            or token_counts[key] != expected.active_token_count
        ):
            raise PartitionArtifactError(f"semantic split counts changed for {key}")
    if not set(control_owner).issubset(result):
        raise PartitionArtifactError("semantic control references an unknown group")
    measured_exclusions = dict(sorted(exclusion_counts.items()))
    persisted_exclusions = {
        key: _integer(expected_exclusions, key)
        for key in expected_exclusions
    }
    if measured_exclusions != persisted_exclusions:
        raise PartitionArtifactError("semantic exclusion audit changed")
    if exclusion_counts["retained_tokens"] != catalog.retained_token_count:
        raise PartitionArtifactError("semantic retained token mass changed")
    if not required_assignments.issubset(result):
        raise PartitionArtifactError("semantic pairings reference unknown assignments")
    return result


def _require_unassigned(record: Mapping[str, object]) -> None:
    if any(
        record.get(field) is not None
        for field in (
            "adjective_bucket",
            "canonical_token_count",
            "noun_bucket",
            "recipe",
            "role",
            "split",
            "verb_bucket",
            "world",
        )
    ):
        raise PartitionArtifactError("excluded semantic group received a split")


def _validate_pairings(
    pairings: Sequence[ControlPair],
    controls: Sequence[ControlSelection],
    cells: Sequence[WorldCell],
    assignments: Mapping[str, Mapping[str, object]],
) -> None:
    expected_order = tuple(
        sorted(
            pairings,
            key=lambda item: (
                ("validation", "test").index(item.split),
                WORLD_LABELS.index(item.world),
                item.world_group_sha256,
                item.control_group_sha256,
            ),
        )
    )
    if tuple(pairings) != expected_order:
        raise PartitionArtifactError("semantic pairings are not canonically ordered")
    cells_by_world = {item.label: item for item in cells}
    for split in ("validation", "test"):
        for world in WORLD_LABELS:
            current = tuple(
                item for item in pairings if item.world == world and item.split == split
            )
            control = next(item for item in controls if item.world == world and item.split == split)
            expected_world = {
                digest
                for digest, record in assignments.items()
                if record.get("status") == "eligible"
                and record.get("role") == "world"
                and record.get("world") == world
                and record.get("split") == split
            }
            if (
                {item.world_group_sha256 for item in current} != expected_world
                or {item.control_group_sha256 for item in current} != set(control.group_sha256)
                or len(current) != len(expected_world)
            ):
                raise PartitionArtifactError("semantic world/control pairing coverage changed")
            cell = cells_by_world[world]
            for pair in current:
                record = assignments[pair.control_group_sha256]
                row = record.get("noun_bucket") == cell.noun_bucket
                column = record.get("verb_bucket") == cell.verb_bucket
                if (pair.arm == "row") != row or (pair.arm == "column") != column or row == column:
                    raise PartitionArtifactError("semantic control pairing arm changed")


def _validate_topology(cells: Sequence[WorldCell], count: int) -> None:
    if tuple(item.label for item in cells) != WORLD_LABELS:
        raise PartitionArtifactError("semantic worlds A-E are incomplete")
    coordinates = tuple((item.noun_bucket, item.verb_bucket) for item in cells)
    if any(not 0 <= value < count for pair in coordinates for value in pair):
        raise PartitionArtifactError("semantic topology cluster index is out of range")
    a, b, c, d, e = coordinates
    if not (
        a[1] == b[1]
        and b[0] == c[0]
        and c[1] == d[1]
        and d[0] == a[0]
        and e[0] not in {a[0], b[0]}
        and e[1] not in {a[1], c[1]}
    ):
        raise PartitionArtifactError("semantic topology is not a 2x2 corner plus E")


def _validate_controls(controls: Sequence[ControlSelection]) -> None:
    expected = tuple((split, world) for split in ("validation", "test") for world in WORLD_LABELS)
    if tuple((item.split, item.world) for item in controls) != expected:
        raise PartitionArtifactError("semantic controls are incomplete or unordered")
    all_groups = [digest for item in controls for digest in item.group_sha256]
    if len(all_groups) != len(set(all_groups)):
        raise PartitionArtifactError("semantic controls reuse groups globally")


def _preset(record: Mapping[str, object]) -> SemanticPartitionPreset:
    default = SemanticPartitionPreset()
    if set(record) != set(default.as_record()):
        raise PartitionArtifactError("semantic partition preset fields changed")
    triple = lambda field: _integer_triple(record, field)
    return SemanticPartitionPreset(
        public_seed=_integer(record, "public_seed"),
        worker_count=default.worker_count,
        run_record_count=default.run_record_count,
        shard_target_bytes=_integer(record, "shard_target_bytes"),
        batch_block_documents=_integer(record, "batch_block_documents"),
        context_length=_integer(record, "context_length"),
        batch_size=_integer(record, "batch_size"),
        minimum_role_coverage=_number(record, "minimum_role_coverage"),
        selected_cell_median_tolerance=_number(record, "selected_cell_median_tolerance"),
        minimum_component_outside_groups=_integer(record, "minimum_component_outside_groups"),
        world_split_weights=triple("world_split_weights"),
        base_split_weights=triple("base_split_weights"),
        control_token_tolerance=_number(record, "control_token_tolerance"),
        control_source_feature_tolerance=_number(record, "control_source_feature_tolerance"),
        control_adjective_length_tolerance=_number(record, "control_adjective_length_tolerance"),
        control_mean_length_tolerance=_number(record, "control_mean_length_tolerance"),
    )


def _word_bucket(record: object) -> WordBucket:
    if type(record) is not dict:
        raise PartitionArtifactError("semantic adjective bucket must be an object")
    words = record.get("words")
    if type(words) is not list or any(type(item) is not str for item in words):
        raise PartitionArtifactError("semantic adjective bucket words are malformed")
    return WordBucket(
        namespace=_text(record, "namespace"),
        index=_integer(record, "index"),
        token_mass=_integer(record, "token_mass"),
        words=tuple(words),
    )


def _world_cell(record: object) -> WorldCell:
    if type(record) is not dict:
        raise PartitionArtifactError("semantic world cell must be an object")
    return WorldCell(
        label=_text(record, "label"),
        noun_bucket=_integer(record, "noun_bucket"),
        verb_bucket=_integer(record, "verb_bucket"),
        active_token_count=_integer(record, "active_token_count"),
        group_count=_integer(record, "group_count"),
    )


def _control(record: object) -> ControlSelection:
    if type(record) is not dict:
        raise PartitionArtifactError("semantic control must be an object")
    groups = record.get("group_sha256")
    if type(groups) is not list or any(type(item) is not str for item in groups):
        raise PartitionArtifactError("semantic control groups are malformed")
    return ControlSelection(
        world=_text(record, "world"),
        split=_text(record, "split"),
        group_sha256=tuple(groups),
        row_group_count=_integer(record, "row_group_count"),
        column_group_count=_integer(record, "column_group_count"),
        active_token_count=_integer(record, "active_token_count"),
    )


def _pair(record: object) -> ControlPair:
    if type(record) is not dict:
        raise PartitionArtifactError("semantic control pair must be an object")
    return ControlPair(
        world=_text(record, "world"),
        split=_text(record, "split"),
        arm=_text(record, "arm"),
        world_group_sha256=_text(record, "world_group_sha256"),
        control_group_sha256=_text(record, "control_group_sha256"),
    )


def _split_count(record: object) -> SplitCount:
    if type(record) is not dict:
        raise PartitionArtifactError("semantic split count must be an object")
    world = record.get("world")
    if world is not None and type(world) is not str:
        raise PartitionArtifactError("semantic split world must be text or null")
    return SplitCount(
        role=_text(record, "role"),
        world=world,
        split=_text(record, "split"),
        group_count=_integer(record, "group_count"),
        occurrence_count=_integer(record, "occurrence_count"),
        active_token_count=_integer(record, "active_token_count"),
    )


def _validate_tree(root: Path, tree: Mapping[str, object]) -> None:
    raw = tree.get("files")
    if type(raw) is not list or any(type(item) is not dict for item in raw):
        raise PartitionArtifactError("semantic partition tree files are malformed")
    relative_paths = tuple(_text(item, "relative_path") for item in raw)
    if relative_paths != tuple(sorted(set(relative_paths))):
        raise PartitionArtifactError("semantic partition tree paths are not canonical")
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual != set(relative_paths) | {"tree.json"} or not _CORE_FILES.issubset(actual):
        raise PartitionArtifactError("semantic partition tree membership changed")
    if any(path.is_symlink() for path in root.rglob("*")):
        raise PartitionArtifactError("semantic partition cannot contain symlinks")
    for descriptor in raw:
        relative = _text(descriptor, "relative_path")
        path = root / relative
        if path.stat().st_size != _integer(descriptor, "size_bytes"):
            raise PartitionArtifactError(f"semantic partition file size changed: {relative}")
    total_bytes = sum(_integer(descriptor, "size_bytes") for descriptor in raw)
    file_paths = tuple(root / _text(descriptor, "relative_path") for descriptor in raw)
    if total_bytes < _PARALLEL_VALIDATION_MIN_BYTES:
        measured = tuple(_file_sha256(path) for path in file_paths)
    else:
        with ThreadPoolExecutor(
            max_workers=min(_MAX_PARALLEL_WORKERS, len(file_paths))
        ) as executor:
            measured = tuple(executor.map(_file_sha256, file_paths))
    for descriptor, digest in zip(raw, measured, strict=True):
        relative = _text(descriptor, "relative_path")
        if digest != _text(descriptor, "sha256"):
            raise PartitionArtifactError(f"semantic partition file checksum changed: {relative}")


def _load_json(path: Path, label: str) -> dict[str, object]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PartitionArtifactError(f"invalid {label}") from error
    if type(value) is not dict or canonical_json_bytes(value) != payload:
        raise PartitionArtifactError(f"{label} is not canonical JSON")
    return value


def _iter_jsonl(path: Path) -> Iterator[dict[str, object]]:
    with path.open("rb") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                value = json.loads(line)
            except (UnicodeError, json.JSONDecodeError) as error:
                raise PartitionArtifactError(f"invalid JSONL at {path}:{line_number}") from error
            if type(value) is not dict or canonical_json_bytes(value) != line:
                raise PartitionArtifactError(f"noncanonical JSONL at {path}:{line_number}")
            yield value


def _mapping(record: Mapping[str, object], field: str) -> dict[str, object]:
    value = record.get(field)
    if type(value) is not dict:
        raise PartitionArtifactError(f"field {field!r} must be an object")
    return value


def _text(record: Mapping[str, object], field: str) -> str:
    value = record.get(field)
    if type(value) is not str or not value:
        raise PartitionArtifactError(f"field {field!r} must be nonempty text")
    return value


def _integer(record: Mapping[str, object], field: str) -> int:
    value = record.get(field)
    if type(value) is not int or value < 0:
        raise PartitionArtifactError(f"field {field!r} must be a nonnegative integer")
    return value


def _number(record: Mapping[str, object], field: str) -> float:
    value = record.get(field)
    if type(value) not in (int, float):
        raise PartitionArtifactError(f"field {field!r} must be numeric")
    return float(value)


def _integer_triple(record: Mapping[str, object], field: str) -> tuple[int, int, int]:
    value = record.get(field)
    if type(value) is not list or len(value) != 3 or any(type(item) is not int for item in value):
        raise PartitionArtifactError(f"field {field!r} must be three integers")
    return value[0], value[1], value[2]


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["PartitionArtifactError", "load_partition"]

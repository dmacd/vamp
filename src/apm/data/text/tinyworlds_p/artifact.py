"""Strict loading and semantic validation of TinyWorlds-P partition trees."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping
from hashlib import sha256
import json
from pathlib import Path
from typing import BinaryIO

import numpy as np

from apm.data.text.tinyworlds_p.contracts import (
    BENCHMARK_ID,
    PARTITION_FORMAT,
    PARTITION_SCHEMA_VERSION,
    ArtifactFile,
    ControlSelection,
    DocumentIndex,
    HashedFile,
    NormalizationIdentity,
    PartitionArtifact,
    PartitionPreset,
    SourceIdentity,
    SplitCount,
    TokenizerIdentity,
    WORLD_LABELS,
    WordBucket,
    WorldCell,
    canonical_record_bytes,
)


_CORE_FILES = {
    "assignments.jsonl",
    "audit.json",
    "buckets.json",
    "controls.json",
    "documents.jsonl",
    "normalization.json",
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


class PartitionArtifactError(ValueError):
    """A persisted partition tree or semantic binding is invalid."""


def load_partition(path: str | Path) -> PartitionArtifact:
    """Strictly authenticate a complete partition tree and its split semantics."""
    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        raise PartitionArtifactError(f"partition must be a non-symlink directory: {root}")
    tree_path = root / "tree.json"
    tree = _load_json(tree_path, "tree manifest")
    _require_fields(
        tree,
        ("files", "format", "partition_sha256", "schema_version"),
        "tree manifest",
    )
    if (
        tree["format"] != "tinyworlds-p-archive-tree"
        or tree["schema_version"] != 1
    ):
        raise PartitionArtifactError("unsupported partition tree format")
    partition_sha256 = _sha256_string(tree, "partition_sha256")
    if root.name != partition_sha256:
        raise PartitionArtifactError("partition directory name does not match its identity")
    raw_files = tree["files"]
    if type(raw_files) is not list:
        raise PartitionArtifactError("tree files must be a list")
    files = tuple(_artifact_file(value) for value in raw_files)
    paths = tuple(item.relative_path for item in files)
    if tuple(sorted(set(paths))) != paths:
        raise PartitionArtifactError("tree file paths must be unique and sorted")
    actual_paths = {
        candidate.relative_to(root).as_posix()
        for candidate in root.rglob("*")
        if candidate.is_file()
    }
    if actual_paths != set(paths) | {"tree.json"}:
        raise PartitionArtifactError("partition tree entries differ from the manifest")
    if not _CORE_FILES.issubset(actual_paths):
        raise PartitionArtifactError(
            "partition is missing canonical files: "
            + repr(tuple(sorted(_CORE_FILES - actual_paths)))
        )
    if any(candidate.is_symlink() for candidate in root.rglob("*")):
        raise PartitionArtifactError("partition trees cannot contain symbolic links")
    for descriptor in files:
        candidate = root / descriptor.relative_path
        if candidate.stat().st_size != descriptor.size_bytes:
            raise PartitionArtifactError(
                f"partition file size changed: {descriptor.relative_path}"
            )
        if _file_sha256(candidate) != descriptor.sha256:
            raise PartitionArtifactError(
                f"partition file checksum changed: {descriptor.relative_path}"
            )
    partition = _load_json(root / "partition.json", "partition identity")
    _require_fields(
        partition,
        (
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
        ),
        "partition identity",
    )
    if (
        partition["benchmark_id"] != BENCHMARK_ID
        or partition["format"] != PARTITION_FORMAT
        or partition["schema_version"] != PARTITION_SCHEMA_VERSION
        or partition["partition_sha256"] != partition_sha256
    ):
        raise PartitionArtifactError("partition identity contract changed")
    assignments_sha256 = _sha256_string(partition, "assignments_sha256")
    if _file_sha256(root / "assignments.jsonl") != assignments_sha256:
        raise PartitionArtifactError("assignment checksum differs from partition identity")
    sources = _object(partition, "sources")
    _require_fields(sources, ("archive", "tokenizer"), "partition sources")
    archive_identity = _source_identity(_object(sources, "archive"))
    tokenizer_identity = _tokenizer_identity(_object(sources, "tokenizer"))
    normalization = _normalization(_object(partition, "normalization"))
    preset = _preset(_object(partition, "preset"))
    persisted_sources = _load_json(root / "sources.json", "source identities")
    if persisted_sources != sources:
        raise PartitionArtifactError("source identity files disagree")
    if _load_json(root / "normalization.json", "normalization") != partition[
        "normalization"
    ]:
        raise PartitionArtifactError("normalization identity files disagree")
    bucket_payload = _load_json(root / "buckets.json", "word buckets")
    _require_fields(bucket_payload, ("buckets", "seed_identity_sha256"), "word buckets")
    if bucket_payload["seed_identity_sha256"] != partition["seed_identity_sha256"]:
        raise PartitionArtifactError("bucket and partition seed identities disagree")
    raw_buckets = bucket_payload["buckets"]
    if type(raw_buckets) is not list:
        raise PartitionArtifactError("word buckets must be a list")
    buckets = tuple(_word_bucket(value) for value in raw_buckets)
    _validate_buckets(buckets, preset.bucket_count)
    topology = _load_json(root / "topology.json", "selected topology")
    _require_fields(topology, ("cells",), "selected topology")
    if type(topology["cells"]) is not list:
        raise PartitionArtifactError("topology cells must be a list")
    cells = tuple(_world_cell(value) for value in topology["cells"])
    _validate_topology(cells)
    control_payload = _load_json(root / "controls.json", "matched controls")
    _require_fields(control_payload, ("controls",), "matched controls")
    if type(control_payload["controls"]) is not list:
        raise PartitionArtifactError("controls must be a list")
    controls = tuple(_control(value) for value in control_payload["controls"])
    _validate_controls(controls)
    audit = _load_json(root / "audit.json", "partition audit")
    _require_fields(
        audit,
        (
            "archive_group_count",
            "archive_ingest",
            "archive_record_count",
            "component_visibility",
            "duplicate_group_count",
            "exclusions",
            "forced_word_visibility",
            "split_counts",
        ),
        "partition audit",
    )
    raw_split_counts = audit.get("split_counts")
    if type(raw_split_counts) is not list:
        raise PartitionArtifactError("partition audit split_counts must be a list")
    split_counts = tuple(_split_count(value) for value in raw_split_counts)
    _validate_assignments(root / "assignments.jsonl", cells, controls, split_counts)
    pad_token_id = _integer(partition, "pad_token_id")
    eos_token_id = _integer(partition, "eos_token_id")
    _validate_documents_and_indexes(
        root,
        controls,
        eos_token_id,
        tokenizer_identity.vocab_size,
    )
    expected_partition_sha256 = _recompute_partition_identity(
        archive_identity,
        tokenizer_identity,
        normalization,
        preset,
        buckets,
        cells,
        controls,
        assignments_sha256,
    )
    if expected_partition_sha256 != partition_sha256:
        raise PartitionArtifactError("partition content identity is inconsistent")
    return PartitionArtifact(
        root=root.resolve(),
        partition_sha256=partition_sha256,
        manifest_sha256=_file_sha256(tree_path),
        archive_identity=archive_identity,
        tokenizer_identity=tokenizer_identity,
        normalization=normalization,
        preset=preset,
        buckets=buckets,
        cells=cells,
        controls=controls,
        split_counts=split_counts,
        files=files,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
    )


def _validate_assignments(
    path: Path,
    cells: tuple[WorldCell, ...],
    controls: tuple[ControlSelection, ...],
    split_counts: tuple[SplitCount, ...],
) -> None:
    selected_cells = {
        (cell.noun_bucket, cell.verb_bucket): cell.label for cell in cells
    }
    expected_counts = {
        (count.role, count.world, count.split): count for count in split_counts
    }
    measured_groups: Counter[tuple[str, str | None, str]] = Counter()
    measured_occurrences: Counter[tuple[str, str | None, str]] = Counter()
    measured_tokens: Counter[tuple[str, str | None, str]] = Counter()
    control_owner = {
        group_sha256: (control.world, control.split)
        for control in controls
        for group_sha256 in control.group_sha256
    }
    if len(control_owner) != sum(len(control.group_sha256) for control in controls):
        raise PartitionArtifactError("a held-in group is reused across matched controls")
    seen_controls: set[str] = set()
    previous_sha256 = ""
    for line_number, record in _iter_jsonl(path):
        _require_fields(
            record,
            (
                "active_token_count",
                "adjective_bucket",
                "canonical_token_count",
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
            ),
            f"assignment line {line_number}",
        )
        group_sha256 = _sha256_string(record, "normalized_story_sha256")
        if group_sha256 <= previous_sha256:
            raise PartitionArtifactError("assignment groups are not unique and sorted")
        previous_sha256 = group_sha256
        status = record.get("status")
        if status not in (
            "eligible",
            "empty_story",
            "unclassifiable_metadata",
            "conflicting_metadata",
        ):
            raise PartitionArtifactError("assignment status is invalid")
        record_ids = record.get("record_ids")
        if (
            type(record_ids) is not list
            or not record_ids
            or any(type(value) is not str or not value for value in record_ids)
            or record_ids != sorted(set(record_ids))
        ):
            raise PartitionArtifactError("assignment record IDs are not canonical")
        provenance = record.get("provenance")
        if type(provenance) is not list or len(provenance) != len(record_ids):
            raise PartitionArtifactError("assignment provenance multiplicity changed")
        if [item.get("record_id") for item in provenance if type(item) is dict] != record_ids:
            raise PartitionArtifactError("assignment provenance and record IDs disagree")
        if status == "eligible":
            role = _text(record, "role")
            split = _text(record, "split")
            world = record.get("world")
            if world is not None and type(world) is not str:
                raise PartitionArtifactError("assignment world must be text or null")
            noun_bucket = _integer(record, "noun_bucket")
            verb_bucket = _integer(record, "verb_bucket")
            selected_world = selected_cells.get((noun_bucket, verb_bucket))
            if role == "base" and (world is not None or selected_world is not None):
                raise PartitionArtifactError("held-out cell leaked into a base assignment")
            if role == "world" and (world != selected_world or world not in WORLD_LABELS):
                raise PartitionArtifactError("world assignment does not match topology")
            key = (role, world, split)
            measured_groups[key] += 1
            measured_occurrences[key] += len(record_ids)
            measured_tokens[key] += _integer(record, "active_token_count")
            if group_sha256 in control_owner:
                owner_world, owner_split = control_owner[group_sha256]
                if role != "base" or split != owner_split:
                    raise PartitionArtifactError("matched control is not held-in")
                seen_controls.add(group_sha256)
        else:
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
                raise PartitionArtifactError("excluded assignment received a split")
    if seen_controls != set(control_owner):
        raise PartitionArtifactError("matched control refers to an unknown assignment")
    for key, expected in expected_counts.items():
        if (
            measured_groups[key] != expected.group_count
            or measured_occurrences[key] != expected.occurrence_count
            or measured_tokens[key] != expected.active_token_count
        ):
            raise PartitionArtifactError(f"persisted split counts disagree for {key}")


def _validate_documents_and_indexes(
    root: Path,
    controls: tuple[ControlSelection, ...],
    eos_token_id: int,
    vocab_size: int,
) -> None:
    """Replay every archive/document/shard/index binding with bounded memory."""
    index_names = tuple(
        [f"base-{split}.jsonl" for split in ("train", "validation", "test")]
        + [
            f"world-{world}-{split}.jsonl"
            for world in WORLD_LABELS
            for split in ("train", "validation", "test")
        ]
        + [
            f"control-{world}-{split}.jsonl"
            for world in WORLD_LABELS
            for split in ("validation", "test")
        ]
    )
    index_streams = {
        name: (root / "indexes" / name).open("rb") for name in index_names
    }
    text_streams: dict[int, BinaryIO] = {}
    token_streams: dict[int, BinaryIO] = {}
    next_text_offset: Counter[int] = Counter()
    next_token_offset: Counter[int] = Counter()
    text_story_counts: Counter[int] = Counter()
    token_story_counts: Counter[int] = Counter()
    control_owner = {
        group_sha256: (control.world, control.split)
        for control in controls
        for group_sha256 in control.group_sha256
    }
    documents = _iter_jsonl(root / "documents.jsonl")
    document_current = next(documents, None)
    seen_record_ids: set[str] = set()
    try:
        for _, assignment in _iter_jsonl(root / "assignments.jsonl"):
            if assignment.get("status") != "eligible":
                continue
            group_sha256 = _sha256_string(
                assignment,
                "normalized_story_sha256",
            )
            record_ids = assignment["record_ids"]
            provenance = assignment["provenance"]
            assert type(record_ids) is list and type(provenance) is list
            provenance_by_id = {
                _text(item, "record_id"): item
                for item in provenance
                if type(item) is dict
            }
            for expected_record_id in record_ids:
                if document_current is None:
                    raise PartitionArtifactError(
                        "eligible assignments exceed persisted documents"
                    )
                _, document = document_current
                document_current = next(documents, None)
                _require_fields(
                    document,
                    (
                        "active_group_token_count",
                        "adjective_bucket",
                        "byte_length",
                        "content_sha256",
                        "normalized_story_sha256",
                        "noun_bucket",
                        "provenance",
                        "recipe",
                        "record_id",
                        "role",
                        "source",
                        "source_index",
                        "source_member",
                        "split",
                        "story_sha256",
                        "text_bytes",
                        "text_offset",
                        "text_shard",
                        "token_count",
                        "token_offset",
                        "token_shard",
                        "verb_bucket",
                        "world",
                    ),
                    "document index",
                )
                record_id = _text(document, "record_id")
                if record_id != expected_record_id or record_id in seen_record_ids:
                    raise PartitionArtifactError(
                        "document record identities are duplicated or out of order"
                    )
                seen_record_ids.add(record_id)
                source_member = _text(document, "source_member")
                source_index = _integer(document, "source_index")
                content_sha256 = _sha256_string(document, "content_sha256")
                story_sha256 = _sha256_string(document, "story_sha256")
                if record_id != (
                    f"archive:{source_member}:{source_index}:{content_sha256}"
                ):
                    raise PartitionArtifactError(
                        "document record ID does not bind archive location and content"
                    )
                source = _text(document, "source")
                if source not in ("GPT-3.5", "GPT-4"):
                    raise PartitionArtifactError("document archive source is invalid")
                source_record = provenance_by_id.get(record_id)
                if source_record is None or source_record != {
                    "content_sha256": content_sha256,
                    "record_id": record_id,
                    "source": source,
                    "source_index": source_index,
                    "source_member": source_member,
                    "story_sha256": story_sha256,
                }:
                    raise PartitionArtifactError(
                        "document archive identity disagrees with group provenance"
                    )
                if document["provenance"] != provenance:
                    raise PartitionArtifactError(
                        "document group provenance differs from its assignment"
                    )
                role = _text(document, "role")
                split = _text(document, "split")
                world = document.get("world")
                if world is not None and type(world) is not str:
                    raise PartitionArtifactError("document world must be text or null")
                for field in (
                    "active_group_token_count",
                    "adjective_bucket",
                    "noun_bucket",
                    "recipe",
                    "role",
                    "split",
                    "verb_bucket",
                    "world",
                ):
                    assignment_field = (
                        "active_token_count"
                        if field == "active_group_token_count"
                        else field
                    )
                    if document[field] != assignment[assignment_field]:
                        raise PartitionArtifactError(
                            f"document {field} differs from its assignment"
                        )
                if _sha256_string(
                    document,
                    "normalized_story_sha256",
                ) != group_sha256:
                    raise PartitionArtifactError(
                        "document duplicate identity differs from its assignment"
                    )
                index = DocumentIndex(
                    record_id=record_id,
                    source_member=source_member,
                    source_index=source_index,
                    content_sha256=content_sha256,
                    story_sha256=story_sha256,
                    normalized_story_sha256=group_sha256,
                    text_shard=_integer(document, "text_shard"),
                    text_offset=_integer(document, "text_offset"),
                    text_bytes=_integer(document, "text_bytes"),
                    token_shard=_integer(document, "token_shard"),
                    token_offset=_integer(document, "token_offset"),
                    token_count=_integer(document, "token_count"),
                    role=role,
                    world=world,
                    split=split,
                )
                if _integer(document, "byte_length") != index.text_bytes:
                    raise PartitionArtifactError(
                        "document source and text-shard byte lengths differ"
                    )
                if index.text_offset != next_text_offset[index.text_shard]:
                    raise PartitionArtifactError("text shard offsets are not contiguous")
                if index.token_offset != next_token_offset[index.token_shard]:
                    raise PartitionArtifactError("token shard offsets are not contiguous")
                if index.text_shard not in text_streams:
                    text_streams[index.text_shard] = (
                        root
                        / "shards"
                        / f"text-{index.text_shard:06d}.bin"
                    ).open("rb")
                text_stream = text_streams[index.text_shard]
                raw_story = text_stream.read(index.text_bytes)
                if (
                    len(raw_story) != index.text_bytes
                    or sha256(raw_story).hexdigest() != story_sha256
                ):
                    raise PartitionArtifactError(
                        "text shard does not reconstruct the exact archive story"
                    )
                if index.token_shard not in token_streams:
                    token_streams[index.token_shard] = (
                        root
                        / "shards"
                        / f"tokens-{index.token_shard:06d}.uint16"
                    ).open("rb")
                token_stream = token_streams[index.token_shard]
                token_payload = token_stream.read(index.token_count * 2)
                tokens = np.frombuffer(token_payload, dtype="<u2")
                if (
                    len(tokens) != index.token_count
                    or int(tokens[-1]) != eos_token_id
                    or bool(np.any(tokens >= vocab_size))
                ):
                    raise PartitionArtifactError("token shard IDs are invalid")
                next_text_offset[index.text_shard] += index.text_bytes
                next_token_offset[index.token_shard] += index.token_count
                text_story_counts[index.text_shard] += 1
                token_story_counts[index.token_shard] += 1
                compact = {
                    "content_sha256": content_sha256,
                    "normalized_story_sha256": group_sha256,
                    "record_id": record_id,
                    "source": source,
                    "source_index": source_index,
                    "source_member": source_member,
                    "story_sha256": story_sha256,
                    "text_bytes": index.text_bytes,
                    "text_offset": index.text_offset,
                    "text_shard": index.text_shard,
                    "token_count": index.token_count,
                    "token_offset": index.token_offset,
                    "token_shard": index.token_shard,
                }
                primary_name = (
                    f"base-{split}.jsonl"
                    if role == "base"
                    else f"world-{world}-{split}.jsonl"
                )
                _require_index_record(index_streams[primary_name], compact, primary_name)
                owner = control_owner.get(group_sha256)
                if owner is not None:
                    control_world, control_split = owner
                    if role != "base" or split != control_split:
                        raise PartitionArtifactError(
                            "control document is not in the matching held-in split"
                        )
                    control_name = f"control-{control_world}-{control_split}.jsonl"
                    _require_index_record(
                        index_streams[control_name],
                        compact,
                        control_name,
                    )
        if document_current is not None:
            raise PartitionArtifactError(
                "persisted documents contain an unknown archive record"
            )
        for name, stream in index_streams.items():
            if stream.read(1):
                raise PartitionArtifactError(
                    f"partition index {name} contains extra records"
                )
        _validate_shard_descriptors(
            root,
            next_text_offset,
            next_token_offset,
            text_story_counts,
            token_story_counts,
        )
    finally:
        for stream in (
            *index_streams.values(),
            *text_streams.values(),
            *token_streams.values(),
        ):
            stream.close()


def _require_index_record(
    stream: BinaryIO,
    expected: dict[str, object],
    name: str,
) -> None:
    if stream.readline() != canonical_record_bytes(expected):
        raise PartitionArtifactError(
            f"partition index {name} differs from the document ledger"
        )


def _validate_shard_descriptors(
    root: Path,
    text_offsets: Mapping[int, int],
    token_offsets: Mapping[int, int],
    text_counts: Mapping[int, int],
    token_counts: Mapping[int, int],
) -> None:
    payload = _load_json(root / "shards.json", "shard descriptors")
    _require_fields(payload, ("shards",), "shard descriptors")
    raw_shards = payload["shards"]
    if type(raw_shards) is not list:
        raise PartitionArtifactError("shard descriptors must be a list")
    expected_order = tuple(
        (kind, shard_id)
        for kind, offsets in (("text", text_offsets), ("tokens", token_offsets))
        for shard_id in sorted(offsets)
    )
    measured_order: list[tuple[str, int]] = []
    for value in raw_shards:
        record = _value_object(value, "shard descriptor")
        _require_fields(
            record,
            ("kind", "relative_path", "shard_id", "size_bytes", "story_count"),
            "shard descriptor",
        )
        kind = _text(record, "kind")
        if kind not in ("text", "tokens"):
            raise PartitionArtifactError("shard descriptor kind is invalid")
        shard_id = _integer(record, "shard_id")
        measured_order.append((kind, shard_id))
        offsets = text_offsets if kind == "text" else token_offsets
        counts = text_counts if kind == "text" else token_counts
        expected_size = offsets.get(shard_id, -1) * (1 if kind == "text" else 2)
        expected_path = f"shards/{kind}-{shard_id:06d}"
        expected_path += ".bin" if kind == "text" else ".uint16"
        if (
            _text(record, "relative_path") != expected_path
            or _integer(record, "size_bytes") != expected_size
            or _integer(record, "story_count") != counts.get(shard_id, -1)
        ):
            raise PartitionArtifactError(
                "shard descriptor differs from document coordinates"
            )
    if tuple(measured_order) != expected_order:
        raise PartitionArtifactError("shard descriptors are incomplete or out of order")


def _validate_buckets(buckets: tuple[WordBucket, ...], bucket_count: int) -> None:
    for namespace in ("noun", "verb", "adjective"):
        selected = tuple(bucket for bucket in buckets if bucket.namespace == namespace)
        if tuple(bucket.index for bucket in selected) != tuple(range(bucket_count)):
            raise PartitionArtifactError(f"{namespace} bucket indexes are incomplete")
        words = tuple(word for bucket in selected for word in bucket.words)
        if len(set(words)) != len(words):
            raise PartitionArtifactError(f"{namespace} word occurs in multiple buckets")


def _validate_topology(cells: tuple[WorldCell, ...]) -> None:
    if tuple(cell.label for cell in cells) != WORLD_LABELS:
        raise PartitionArtifactError("selected cells are not canonically labelled")
    by_label = {cell.label: cell for cell in cells}
    a, b, c, d, e = (by_label[label] for label in WORLD_LABELS)
    if not (
        a.noun_bucket == d.noun_bucket
        and b.noun_bucket == c.noun_bucket
        and a.verb_bucket == b.verb_bucket
        and c.verb_bucket == d.verb_bucket
        and a.noun_bucket != b.noun_bucket
        and a.verb_bucket != c.verb_bucket
        and e.noun_bucket not in (a.noun_bucket, b.noun_bucket)
        and e.verb_bucket not in (a.verb_bucket, c.verb_bucket)
    ):
        raise PartitionArtifactError(
            "selected cells do not form the archive-v1 corner topology"
        )


def _validate_controls(controls: tuple[ControlSelection, ...]) -> None:
    expected = tuple(
        (world, split)
        for split in ("validation", "test")
        for world in WORLD_LABELS
    )
    if tuple((control.world, control.split) for control in controls) != expected:
        raise PartitionArtifactError("matched controls are incomplete or out of order")


def _recompute_partition_identity(
    archive: SourceIdentity,
    tokenizer: TokenizerIdentity,
    normalization: NormalizationIdentity,
    preset: PartitionPreset,
    buckets: tuple[WordBucket, ...],
    cells: tuple[WorldCell, ...],
    controls: tuple[ControlSelection, ...],
    assignments_sha256: str,
) -> str:
    return sha256(
        canonical_record_bytes(
            {
                "assignments_sha256": assignments_sha256,
                "benchmark_id": BENCHMARK_ID,
                "buckets": [_bucket_record(bucket) for bucket in buckets],
                "cells": [_cell_record(cell) for cell in cells],
                "controls": [_control_record(control) for control in controls],
                "normalization": normalization.as_record(),
                "preset": preset.as_record(),
                "sources": {
                    "archive": archive.as_record(),
                    "tokenizer": tokenizer.as_record(),
                },
            }
        )
    ).hexdigest()


def _source_identity(record: dict[str, object]) -> SourceIdentity:
    _require_fields(
        record,
        ("dataset_id", "filename", "revision", "sha256", "size_bytes"),
        "source identity",
    )
    return SourceIdentity(
        dataset_id=_text(record, "dataset_id"),
        revision=_text(record, "revision"),
        filename=_text(record, "filename"),
        size_bytes=_integer(record, "size_bytes"),
        sha256=_sha256_string(record, "sha256"),
    )


def _tokenizer_identity(record: dict[str, object]) -> TokenizerIdentity:
    _require_fields(
        record,
        ("files", "identifier", "kind", "revision", "vocab_size"),
        "tokenizer identity",
    )
    raw_files = record["files"]
    if type(raw_files) is not list:
        raise PartitionArtifactError("tokenizer files must be a list")
    files = tuple(_hashed_file(value) for value in raw_files)
    return TokenizerIdentity(
        kind=_text(record, "kind"),
        identifier=_text(record, "identifier"),
        revision=_text(record, "revision"),
        vocab_size=_integer(record, "vocab_size"),
        files=files,
    )


def _hashed_file(value: object) -> HashedFile:
    record = _value_object(value, "tokenizer file")
    _require_fields(record, ("name", "sha256", "size_bytes"), "tokenizer file")
    return HashedFile(
        name=_text(record, "name"),
        size_bytes=_integer(record, "size_bytes"),
        sha256=_sha256_string(record, "sha256"),
    )


def _normalization(record: dict[str, object]) -> NormalizationIdentity:
    _require_fields(
        record,
        (
            "canonical_straight_quotes",
            "case_folding",
            "unicode_form",
            "version",
            "whitespace_collapse",
        ),
        "normalization identity",
    )
    unicode_form = _text(record, "unicode_form")
    if unicode_form != "NFKC":
        raise PartitionArtifactError("normalization Unicode form changed")
    return NormalizationIdentity(
        version=_text(record, "version"),
        unicode_form="NFKC",
        case_folding=_boolean(record, "case_folding"),
        whitespace_collapse=_boolean(record, "whitespace_collapse"),
        canonical_straight_quotes=_boolean(record, "canonical_straight_quotes"),
    )


def _preset(record: dict[str, object]) -> PartitionPreset:
    expected = set(PartitionPreset().as_record())
    if set(record) != expected:
        raise PartitionArtifactError("partition preset fields changed")
    world_weights = _integer_triple(record, "world_split_weights")
    base_weights = _integer_triple(record, "base_split_weights")
    return PartitionPreset(
        bucket_count=_integer(record, "bucket_count"),
        public_seed=_integer(record, "public_seed"),
        shard_target_bytes=_integer(record, "shard_target_bytes"),
        batch_block_documents=_integer(record, "batch_block_documents"),
        context_length=_integer(record, "context_length"),
        batch_size=_integer(record, "batch_size"),
        minimum_role_coverage=_number(record, "minimum_role_coverage"),
        selected_cell_median_tolerance=_number(
            record, "selected_cell_median_tolerance"
        ),
        minimum_component_outside_groups=_integer(
            record, "minimum_component_outside_groups"
        ),
        world_split_weights=world_weights,
        base_split_weights=base_weights,
        control_token_tolerance=_number(record, "control_token_tolerance"),
        control_source_feature_tolerance=_number(
            record, "control_source_feature_tolerance"
        ),
        control_adjective_length_tolerance=_number(
            record, "control_adjective_length_tolerance"
        ),
        control_mean_length_tolerance=_number(
            record, "control_mean_length_tolerance"
        ),
    )


def _word_bucket(value: object) -> WordBucket:
    record = _value_object(value, "word bucket")
    _require_fields(record, ("index", "namespace", "token_mass", "words"), "word bucket")
    words = record["words"]
    if type(words) is not list or any(type(word) is not str for word in words):
        raise PartitionArtifactError("bucket words must be a string list")
    namespace = _text(record, "namespace")
    if namespace not in ("noun", "verb", "adjective"):
        raise PartitionArtifactError("unknown word bucket namespace")
    return WordBucket(
        namespace=namespace,
        index=_integer(record, "index"),
        token_mass=_integer(record, "token_mass"),
        words=tuple(words),
    )


def _world_cell(value: object) -> WorldCell:
    record = _value_object(value, "world cell")
    _require_fields(
        record,
        ("active_token_count", "group_count", "label", "noun_bucket", "verb_bucket"),
        "world cell",
    )
    return WorldCell(
        label=_text(record, "label"),
        noun_bucket=_integer(record, "noun_bucket"),
        verb_bucket=_integer(record, "verb_bucket"),
        active_token_count=_integer(record, "active_token_count"),
        group_count=_integer(record, "group_count"),
    )


def _control(value: object) -> ControlSelection:
    record = _value_object(value, "control selection")
    _require_fields(
        record,
        (
            "active_token_count",
            "column_group_count",
            "group_sha256",
            "row_group_count",
            "split",
            "world",
        ),
        "control selection",
    )
    group_values = record["group_sha256"]
    if type(group_values) is not list or any(type(item) is not str for item in group_values):
        raise PartitionArtifactError("control group hashes must be a string list")
    split = _text(record, "split")
    if split not in ("validation", "test"):
        raise PartitionArtifactError("control split must be validation or test")
    return ControlSelection(
        world=_text(record, "world"),
        split=split,
        group_sha256=tuple(group_values),
        row_group_count=_integer(record, "row_group_count"),
        column_group_count=_integer(record, "column_group_count"),
        active_token_count=_integer(record, "active_token_count"),
    )


def _split_count(value: object) -> SplitCount:
    record = _value_object(value, "split count")
    _require_fields(
        record,
        (
            "active_token_count",
            "group_count",
            "occurrence_count",
            "role",
            "split",
            "world",
        ),
        "split count",
    )
    role = _text(record, "role")
    split = _text(record, "split")
    world = record["world"]
    if role not in ("base", "world") or split not in ("train", "validation", "test"):
        raise PartitionArtifactError("split count role or split is invalid")
    if world is not None and type(world) is not str:
        raise PartitionArtifactError("split count world must be text or null")
    return SplitCount(
        role=role,
        world=world,
        split=split,
        group_count=_integer(record, "group_count"),
        occurrence_count=_integer(record, "occurrence_count"),
        active_token_count=_integer(record, "active_token_count"),
    )


def _artifact_file(value: object) -> ArtifactFile:
    record = _value_object(value, "artifact file")
    _require_fields(record, ("relative_path", "sha256", "size_bytes"), "artifact file")
    return ArtifactFile(
        relative_path=_text(record, "relative_path"),
        size_bytes=_integer(record, "size_bytes"),
        sha256=_sha256_string(record, "sha256"),
    )


def _bucket_record(bucket: WordBucket) -> dict[str, object]:
    return {
        "index": bucket.index,
        "namespace": bucket.namespace,
        "token_mass": bucket.token_mass,
        "words": list(bucket.words),
    }


def _cell_record(cell: WorldCell) -> dict[str, object]:
    return {
        "active_token_count": cell.active_token_count,
        "group_count": cell.group_count,
        "label": cell.label,
        "noun_bucket": cell.noun_bucket,
        "verb_bucket": cell.verb_bucket,
    }


def _control_record(control: ControlSelection) -> dict[str, object]:
    return {
        "active_token_count": control.active_token_count,
        "column_group_count": control.column_group_count,
        "group_sha256": list(control.group_sha256),
        "row_group_count": control.row_group_count,
        "split": control.split,
        "world": control.world,
    }


def _load_json(path: Path, label: str) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise PartitionArtifactError(f"missing non-symlink {label}: {path}")
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except (json.JSONDecodeError, UnicodeError) as error:
        raise PartitionArtifactError(f"invalid {label}: {error}") from error
    if type(value) is not dict:
        raise PartitionArtifactError(f"{label} must be an object")
    if canonical_record_bytes(value) != payload:
        raise PartitionArtifactError(f"{label} is not canonically encoded")
    return value


def _iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, object]]]:
    with path.open("rb") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                value = json.loads(line)
            except (json.JSONDecodeError, UnicodeError) as error:
                raise PartitionArtifactError(f"invalid JSONL at {path}:{line_number}") from error
            if type(value) is not dict or canonical_record_bytes(value) != line:
                raise PartitionArtifactError(f"noncanonical JSONL at {path}:{line_number}")
            yield line_number, value


def _value_object(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise PartitionArtifactError(f"{label} must be an object")
    return value


def _object(record: dict[str, object], field: str) -> dict[str, object]:
    return _value_object(record.get(field), field)


def _require_fields(record: dict[str, object], fields: tuple[str, ...], label: str) -> None:
    if set(record) != set(fields):
        raise PartitionArtifactError(f"{label} fields changed")


def _text(record: dict[str, object], field: str) -> str:
    value = record.get(field)
    if type(value) is not str or not value:
        raise PartitionArtifactError(f"{field} must be nonempty text")
    return value


def _integer(record: dict[str, object], field: str) -> int:
    value = record.get(field)
    if type(value) is not int:
        raise PartitionArtifactError(f"{field} must be an integer")
    return value


def _number(record: dict[str, object], field: str) -> float:
    value = record.get(field)
    if type(value) not in (int, float):
        raise PartitionArtifactError(f"{field} must be a number")
    return float(value)


def _boolean(record: dict[str, object], field: str) -> bool:
    value = record.get(field)
    if type(value) is not bool:
        raise PartitionArtifactError(f"{field} must be a boolean")
    return value


def _sha256_string(record: dict[str, object], field: str) -> str:
    value = _text(record, field)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise PartitionArtifactError(f"{field} must be a lowercase SHA-256")
    return value


def _integer_triple(record: dict[str, object], field: str) -> tuple[int, int, int]:
    value = record.get(field)
    if type(value) is not list or len(value) != 3 or any(type(item) is not int for item in value):
        raise PartitionArtifactError(f"{field} must contain three integers")
    return value[0], value[1], value[2]


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["PartitionArtifactError", "load_partition"]

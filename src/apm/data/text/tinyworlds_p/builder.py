"""Deterministic construction and publication of TinyWorlds-P partitions."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from hashlib import sha256
import heapq
import json
import multiprocessing
import os
from pathlib import Path
import shutil
from typing import BinaryIO

import numpy as np

from apm.data.text.tinyworlds_p.contracts import (
    BENCHMARK_ID,
    NORMALIZATION_IDENTITY,
    PARTITION_FORMAT,
    PARTITION_SCHEMA_VERSION,
    ArtifactFile,
    ControlSelection,
    PartitionArtifact,
    PartitionInputs,
    PartitionPreset,
    ProgressEvent,
    SplitCount,
    WORLD_LABELS,
    WordBucket,
    WorldCell,
    canonical_record_bytes,
)
from apm.data.text.tinyworlds_p.partitioning import (
    AllocationGroup,
    PartitionGateError,
    balance_word_buckets,
    bucket_word_lookup,
    require_component_visibility,
    select_matched_control,
    select_world_cells,
    summarize_cells,
)
from apm.data.text.tinyworlds_p.source_join import (
    JoinedSources,
    build_source_join,
    iter_joined_groups,
)
from apm.lm.text import TokenizersTextTokenizer


_TOKENIZER_WORKER: TokenizersTextTokenizer | None = None
_TOKENIZATION_BATCH_SIZE = 128


@dataclass(frozen=True, slots=True)
class _DomainTotals:
    token_count: int
    marginal_token_counts: tuple[tuple[str, str, int], ...]


@dataclass(frozen=True, slots=True)
class _PreparedAllocation:
    assignments_path: Path
    controls: tuple[ControlSelection, ...]
    control_group_owners: tuple[tuple[str, str, str], ...]
    split_counts: tuple[SplitCount, ...]
    allocation_groups_by_evaluation_domain: Mapping[
        tuple[str, str], tuple[AllocationGroup, ...]
    ]


class _ShardWriter:
    """Write deterministic shards, rolling only before a complete story."""

    def __init__(self, directory: Path, stem: str, suffix: str, target_bytes: int) -> None:
        self._directory = directory
        self._stem = stem
        self._suffix = suffix
        self._target_bytes = target_bytes
        self._index = -1
        self._stream: BinaryIO | None = None
        self._size = 0
        self._stories = 0
        self._completed: list[dict[str, object]] = []

    def write(self, payload: bytes) -> tuple[int, int]:
        """Append one whole story payload and return shard ID and byte offset."""
        if not payload:
            raise ValueError("shard story payload must not be empty")
        if self._stream is None or (
            self._size > 0 and self._size + len(payload) > self._target_bytes
        ):
            self._roll()
        assert self._stream is not None
        offset = self._size
        self._stream.write(payload)
        self._size += len(payload)
        self._stories += 1
        return self._index, offset

    def finish(self) -> tuple[dict[str, object], ...]:
        """Close the final shard and return canonical shard descriptors."""
        self._close_current()
        return tuple(self._completed)

    def _roll(self) -> None:
        self._close_current()
        self._index += 1
        path = self._directory / f"{self._stem}-{self._index:06d}{self._suffix}"
        self._stream = path.open("wb")
        self._size = 0
        self._stories = 0

    def _close_current(self) -> None:
        if self._stream is None:
            return
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._stream.close()
        relative_path = f"shards/{self._stem}-{self._index:06d}{self._suffix}"
        self._completed.append(
            {
                "kind": "text" if self._stem == "text" else "tokens",
                "relative_path": relative_path,
                "shard_id": self._index,
                "size_bytes": self._size,
                "story_count": self._stories,
            }
        )
        self._stream = None


def build_partition(
    inputs: PartitionInputs,
    preset: PartitionPreset,
) -> PartitionArtifact:
    """Build, gate, atomically publish, and strictly reload one partition."""
    if type(inputs) is not PartitionInputs or type(preset) is not PartitionPreset:
        raise TypeError("build_partition requires PartitionInputs and PartitionPreset")
    joined = build_source_join(inputs, preset, NORMALIZATION_IDENTITY)
    seed_identity = _seed_identity(inputs, preset)
    _progress(inputs, "buckets", 0, None, "balancing ingredient token mass")
    buckets = _build_buckets(joined, preset, seed_identity)
    bucket_lookups = {
        namespace: bucket_word_lookup(
            tuple(bucket for bucket in buckets if bucket.namespace == namespace)
        )
        for namespace in ("noun", "verb", "adjective")
    }
    allocation_factory = lambda: _iter_allocation_groups(
        joined.groups_path,
        bucket_lookups["noun"],
        bucket_lookups["verb"],
        bucket_lookups["adjective"],
    )
    cells = select_world_cells(
        summarize_cells(allocation_factory()),
        preset.bucket_count,
        seed_identity,
        public_seed=preset.public_seed,
        median_tolerance=preset.selected_cell_median_tolerance,
    )
    visibility = require_component_visibility(
        allocation_factory(),
        cells,
        preset.minimum_component_outside_groups,
    )
    _progress(inputs, "splits", 0, None, "allocating groups and strict matched controls")
    allocation = _prepare_allocations(
        inputs,
        preset,
        joined,
        cells,
        allocation_factory,
        seed_identity,
    )
    assignments_sha256 = _file_sha256(allocation.assignments_path)
    partition_sha256 = _partition_identity(
        inputs,
        preset,
        buckets,
        cells,
        allocation.controls,
        assignments_sha256,
    )
    target = inputs.output_root / partition_sha256
    if target.exists():
        raise FileExistsError(f"partition artifact already exists: {target}")
    publication = inputs.temporary_directory / "publication"
    if publication.exists():
        raise FileExistsError(f"partition publication staging path exists: {publication}")
    publication.mkdir(parents=True)
    (publication / "shards").mkdir()
    (publication / "indexes").mkdir()
    (publication / "manifests").mkdir()
    shutil.copyfile(allocation.assignments_path, publication / "assignments.jsonl")
    _progress(inputs, "shards", 0, None, "writing original bytes and uint16 token shards")
    shard_records, occurrence_counts = _write_shards_and_indexes(
        inputs,
        preset,
        joined,
        allocation.assignments_path,
        allocation.control_group_owners,
        publication,
    )
    _write_partition_metadata(
        publication,
        inputs,
        preset,
        joined,
        partition_sha256,
        seed_identity,
        assignments_sha256,
        buckets,
        cells,
        allocation.controls,
        allocation.split_counts,
        visibility,
        shard_records,
        occurrence_counts,
    )
    tree_manifest = _write_tree_manifest(publication, partition_sha256)
    manifest_sha256 = _file_sha256(tree_manifest)
    inputs.output_root.mkdir(parents=True, exist_ok=True)
    os.rename(publication, target)
    _fsync_directory(inputs.output_root)
    _progress(inputs, "publish", 1, 1, f"published {partition_sha256}")
    from apm.data.text.tinyworlds_p.artifact import load_partition

    artifact = load_partition(target)
    if artifact.manifest_sha256 != manifest_sha256:
        raise RuntimeError("published partition manifest identity changed on strict reload")
    return artifact


def _build_buckets(
    joined: JoinedSources,
    preset: PartitionPreset,
    seed_identity: str,
) -> tuple[WordBucket, ...]:
    word_masses: dict[str, Counter[str]] = {
        "noun": Counter(),
        "verb": Counter(),
        "adjective": Counter(),
    }
    for group in iter_joined_groups(joined.groups_path):
        if group.get("status") != "eligible":
            continue
        recipe = _recipe(group)
        active_tokens = _integer(group, "active_token_count")
        for namespace in word_masses:
            word_masses[namespace][_string(recipe, namespace)] += active_tokens
    return tuple(
        bucket
        for namespace in ("noun", "verb", "adjective")
        for bucket in balance_word_buckets(
            word_masses[namespace],
            namespace,
            preset.bucket_count,
            seed_identity,
            public_seed=preset.public_seed,
        )
    )


def _iter_allocation_groups(
    joined_groups_path: Path,
    noun_lookup: Mapping[str, int],
    verb_lookup: Mapping[str, int],
    adjective_lookup: Mapping[str, int],
) -> Iterator[AllocationGroup]:
    for group in iter_joined_groups(joined_groups_path):
        if group.get("status") != "eligible":
            continue
        recipe = _recipe(group)
        occurrences = _record_list(group, "occurrences")
        canonical_occurrence = min(
            occurrences,
            key=lambda occurrence: _string(occurrence, "occurrence_id"),
        )
        provenance = _record_list(group, "provenance")
        sources = tuple(sorted({_string(record, "source") for record in provenance}))
        noun = _string(recipe, "noun")
        verb = _string(recipe, "verb")
        adjective = _string(recipe, "adjective")
        raw_features = recipe.get("features")
        if type(raw_features) is not list or any(type(item) is not str for item in raw_features):
            raise ValueError("joined recipe features must be a string list")
        yield AllocationGroup(
            normalized_sha256=_string(group, "normalized_sha256"),
            active_token_count=_integer(group, "active_token_count"),
            canonical_token_count=_integer(canonical_occurrence, "token_count"),
            noun=noun,
            verb=verb,
            adjective=adjective,
            noun_bucket=noun_lookup[noun],
            verb_bucket=verb_lookup[verb],
            adjective_bucket=adjective_lookup[adjective],
            source="+".join(sources),
            feature_signature="+".join(raw_features) if raw_features else "none",
        )


def _prepare_allocations(
    inputs: PartitionInputs,
    preset: PartitionPreset,
    joined: JoinedSources,
    cells: Sequence[WorldCell],
    allocation_factory,
    seed_identity: str,
) -> _PreparedAllocation:
    cell_labels = {(cell.noun_bucket, cell.verb_bucket): cell.label for cell in cells}
    allocation_runs_directory = inputs.temporary_directory / "allocation-runs"
    assignment_runs_directory = inputs.temporary_directory / "assignment-runs"
    allocation_runs_directory.mkdir()
    assignment_runs_directory.mkdir()
    totals_tokens: Counter[str] = Counter()
    totals_marginals: Counter[tuple[str, str, str]] = Counter()
    pending: list[dict[str, object]] = []
    allocation_run_paths: list[Path] = []
    for group in allocation_factory():
        domain = cell_labels.get((group.noun_bucket, group.verb_bucket), "base")
        totals_tokens[domain] += group.active_token_count
        for dimension, category in group.marginals:
            totals_marginals[(domain, dimension, category)] += group.active_token_count
        pending.append(
            {
                **_allocation_group_record(group),
                "domain": domain,
                "order_sha256": _namespaced_sha256(
                    seed_identity,
                    f"allocator-order:{domain}",
                    group.normalized_sha256,
                ),
            }
        )
        if len(pending) == preset.run_record_count:
            allocation_run_paths.append(
                _flush_records(
                    pending,
                    allocation_runs_directory,
                    "allocation",
                    len(allocation_run_paths),
                    _allocation_sort_key,
                )
            )
            pending = []
    if pending:
        allocation_run_paths.append(
            _flush_records(
                pending,
                allocation_runs_directory,
                "allocation",
                len(allocation_run_paths),
                _allocation_sort_key,
            )
        )
    expected_domains = {"base", *WORLD_LABELS}
    if set(totals_tokens) != expected_domains:
        raise PartitionGateError(
            f"partition allocation domains are incomplete: {tuple(sorted(totals_tokens))}"
        )
    domain_totals = {
        domain: _DomainTotals(
            token_count=totals_tokens[domain],
            marginal_token_counts=tuple(
                sorted(
                    (dimension, category, count)
                    for (current_domain, dimension, category), count in totals_marginals.items()
                    if current_domain == domain
                )
            ),
        )
        for domain in expected_domains
    }
    assignment_run_paths = _allocate_sorted_runs(
        allocation_run_paths,
        assignment_runs_directory,
        domain_totals,
        preset,
        seed_identity,
    )
    eligible_assignments_path = inputs.temporary_directory / "eligible-assignments.jsonl"
    _merge_records_to_path(
        assignment_run_paths,
        eligible_assignments_path,
        lambda record: (_string(record, "normalized_sha256"),),
    )
    assignments_path = inputs.temporary_directory / "assignments.jsonl"
    evaluation_groups: dict[tuple[str, str], list[AllocationGroup]] = defaultdict(list)
    split_counter: Counter[tuple[str, str | None, str, str]] = Counter()
    with assignments_path.open("wb") as output:
        eligible_iterator = _iter_jsonl(eligible_assignments_path)
        eligible_current = next(eligible_iterator, None)
        for joined_group in iter_joined_groups(joined.groups_path):
            group_sha256 = _string(joined_group, "normalized_sha256")
            if joined_group.get("status") == "eligible":
                if eligible_current is None or _string(
                    eligible_current, "normalized_sha256"
                ) != group_sha256:
                    raise ValueError(f"missing eligible assignment for {group_sha256}")
                assignment = eligible_current
                eligible_current = next(eligible_iterator, None)
                allocation_group = _allocation_group_from_record(assignment)
                domain = _string(assignment, "domain")
                split = _string(assignment, "split")
                role = "base" if domain == "base" else "world"
                world = None if domain == "base" else domain
                if split in ("validation", "test"):
                    evaluation_groups[(domain, split)].append(allocation_group)
                occurrence_count = len(_record_list(joined_group, "occurrences"))
                token_count = _integer(joined_group, "active_token_count")
                for measure, amount in (
                    ("groups", 1),
                    ("occurrences", occurrence_count),
                    ("tokens", token_count),
                ):
                    split_counter[(role, world, split, measure)] += amount
                assignment_record = {
                    "active_token_count": token_count,
                    "adjective_bucket": allocation_group.adjective_bucket,
                    "canonical_token_count": allocation_group.canonical_token_count,
                    "normalized_sha256": group_sha256,
                    "noun_bucket": allocation_group.noun_bucket,
                    "occurrence_ids": [
                        _string(occurrence, "occurrence_id")
                        for occurrence in _record_list(joined_group, "occurrences")
                    ],
                    "provenance": joined_group["provenance"],
                    "recipe": joined_group["recipe"],
                    "role": role,
                    "split": split,
                    "status": "eligible",
                    "verb_bucket": allocation_group.verb_bucket,
                    "world": world,
                }
            else:
                assignment_record = {
                    "active_token_count": _integer(joined_group, "active_token_count"),
                    "adjective_bucket": None,
                    "canonical_token_count": None,
                    "normalized_sha256": group_sha256,
                    "noun_bucket": None,
                    "occurrence_ids": [
                        _string(occurrence, "occurrence_id")
                        for occurrence in _record_list(joined_group, "occurrences")
                    ],
                    "provenance": joined_group["provenance"],
                    "recipe": None,
                    "role": None,
                    "split": None,
                    "status": joined_group["status"],
                    "verb_bucket": None,
                    "world": None,
                }
            output.write(canonical_record_bytes(assignment_record))
        if eligible_current is not None:
            raise ValueError("eligible assignment stream contains an unknown group")
    controls, owners = _build_controls(
        evaluation_groups,
        cells,
        preset,
        seed_identity,
    )
    split_counts = tuple(
        SplitCount(
            role=role,
            world=world,
            split=split,
            group_count=split_counter[(role, world, split, "groups")],
            occurrence_count=split_counter[(role, world, split, "occurrences")],
            active_token_count=split_counter[(role, world, split, "tokens")],
        )
        for role, world in (("base", None), *(('world', label) for label in WORLD_LABELS))
        for split in ("train", "validation", "test")
    )
    return _PreparedAllocation(
        assignments_path=assignments_path,
        controls=controls,
        control_group_owners=owners,
        split_counts=split_counts,
        allocation_groups_by_evaluation_domain={
            key: tuple(value) for key, value in evaluation_groups.items()
        },
    )


def _allocate_sorted_runs(
    allocation_run_paths: Sequence[Path],
    assignment_directory: Path,
    domain_totals: Mapping[str, _DomainTotals],
    preset: PartitionPreset,
    seed_identity: str,
) -> tuple[Path, ...]:
    split_labels = ("train", "validation", "test")
    active_domain: str | None = None
    split_tokens: Counter[str] = Counter()
    split_marginals: Counter[tuple[str, str, str]] = Counter()
    pending: list[dict[str, object]] = []
    paths: list[Path] = []
    for record in _merge_records(allocation_run_paths, _allocation_sort_key):
        domain = _string(record, "domain")
        if domain != active_domain:
            active_domain = domain
            split_tokens = Counter()
            split_marginals = Counter()
        group = _allocation_group_from_record(record)
        weights = preset.base_split_weights if domain == "base" else preset.world_split_weights
        totals = domain_totals[domain]
        total_marginals = {
            (dimension, category): count
            for dimension, category, count in totals.marginal_token_counts
        }
        scores = tuple(
            (
                _streaming_allocator_score(
                    label,
                    weight,
                    sum(weights),
                    totals.token_count,
                    total_marginals,
                    split_tokens,
                    split_marginals,
                    group,
                ),
                _namespaced_sha256(
                    seed_identity,
                    f"allocator-tie:{domain}",
                    f"{group.normalized_sha256}\0{label}",
                ),
                label,
            )
            for label, weight in zip(split_labels, weights, strict=True)
        )
        split = min(scores)[2]
        split_tokens[split] += group.active_token_count
        for dimension, category in group.marginals:
            split_marginals[(split, dimension, category)] += group.active_token_count
        pending.append({**record, "split": split})
        if len(pending) == preset.run_record_count:
            paths.append(
                _flush_records(
                    pending,
                    assignment_directory,
                    "assignment",
                    len(paths),
                    lambda value: (_string(value, "normalized_sha256"),),
                )
            )
            pending = []
    if pending:
        paths.append(
            _flush_records(
                pending,
                assignment_directory,
                "assignment",
                len(paths),
                lambda value: (_string(value, "normalized_sha256"),),
            )
        )
    return tuple(paths)


def _streaming_allocator_score(
    split: str,
    weight: int,
    total_weight: int,
    total_tokens: int,
    total_marginals: Mapping[tuple[str, str], int],
    split_tokens: Mapping[str, int],
    split_marginals: Mapping[tuple[str, str, str], int],
    group: AllocationGroup,
) -> float:
    target_tokens = total_tokens * weight / total_weight
    overall_fill = (split_tokens.get(split, 0) + group.active_token_count) / target_tokens
    marginal_fill = sum(
        (
            (
                split_marginals.get((split, dimension, category), 0)
                + group.active_token_count
            )
            / (total_marginals[(dimension, category)] * weight / total_weight)
        )
        ** 2
        for dimension, category in group.marginals
    )
    return overall_fill**2 + marginal_fill


def _build_controls(
    evaluation_groups: Mapping[tuple[str, str], Sequence[AllocationGroup]],
    cells: Sequence[WorldCell],
    preset: PartitionPreset,
    seed_identity: str,
) -> tuple[tuple[ControlSelection, ...], tuple[tuple[str, str, str], ...]]:
    controls: list[ControlSelection] = []
    owners: list[tuple[str, str, str]] = []
    cells_by_label = {cell.label: cell for cell in cells}
    used_by_split: dict[str, set[str]] = {"validation": set(), "test": set()}
    for split in ("validation", "test"):
        base_candidates = tuple(evaluation_groups[("base", split)])
        # E draws from all four complement arms, so reserve its balanced sample
        # first; opposite corner pairs then consume symmetric remaining pools.
        for world in ("E", "A", "C", "B", "D"):
            cell = cells_by_label[world]
            unused = tuple(
                group
                for group in base_candidates
                if group.normalized_sha256 not in used_by_split[split]
            )
            row_candidates = tuple(
                group
                for group in unused
                if group.noun_bucket == cell.noun_bucket
                and group.verb_bucket != cell.verb_bucket
            )
            column_candidates = tuple(
                group
                for group in unused
                if group.verb_bucket == cell.verb_bucket
                and group.noun_bucket != cell.noun_bucket
            )
            control, _ = select_matched_control(
                evaluation_groups[(world, split)],
                row_candidates,
                column_candidates,
                world,
                split,
                seed_identity,
                preset,
            )
            controls.append(control)
            used_by_split[split].update(control.group_sha256)
            owners.extend((group_sha256, world, split) for group_sha256 in control.group_sha256)
    canonical_controls = tuple(
        sorted(
            controls,
            key=lambda control: (
                ("validation", "test").index(control.split),
                WORLD_LABELS.index(control.world),
            ),
        )
    )
    return canonical_controls, tuple(sorted(owners))


def _write_shards_and_indexes(
    inputs: PartitionInputs,
    preset: PartitionPreset,
    joined: JoinedSources,
    assignments_path: Path,
    control_owners: Sequence[tuple[str, str, str]],
    publication: Path,
) -> tuple[tuple[dict[str, object], ...], Counter[tuple[str, str | None, str]]]:
    control_by_group = {
        group_sha256: (world, split) for group_sha256, world, split in control_owners
    }
    if len(control_by_group) != len(control_owners):
        raise ValueError("control groups must not be reused")
    text_writer = _ShardWriter(
        publication / "shards",
        "text",
        ".bin",
        preset.shard_target_bytes,
    )
    token_writer = _ShardWriter(
        publication / "shards",
        "tokens",
        ".uint16",
        preset.shard_target_bytes,
    )
    index_names = tuple(
        [
            f"base-{split}.jsonl"
            for split in ("train", "validation", "test")
        ]
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
        name: (publication / "indexes" / name).open("wb") for name in index_names
    }
    occurrence_counts: Counter[tuple[str, str | None, str]] = Counter()
    documents_path = publication / "documents.jsonl"
    try:
        with documents_path.open("wb") as documents:
            occurrences = _iter_assigned_raw_occurrences(
                inputs.corpus_path,
                joined.groups_path,
                assignments_path,
            )
            tokenized = _ordered_tokenize(
                occurrences,
                inputs.tokenizer_directory / "tokenizer.json",
                preset.worker_count,
            )
            for occurrence, raw_story, token_ids in tokenized:
                if len(token_ids) != _integer(occurrence, "token_count"):
                    raise ValueError("replayed tokenizer count differs from source join")
                if any(not 0 <= token_id <= np.iinfo(np.uint16).max for token_id in token_ids):
                    raise ValueError("token shard ID does not fit uint16")
                text_shard, text_offset = text_writer.write(raw_story)
                token_payload = np.asarray(token_ids, dtype="<u2").tobytes(order="C")
                token_shard, token_byte_offset = token_writer.write(token_payload)
                assignment = _object_record(occurrence, "assignment")
                group_sha256 = _string(assignment, "normalized_sha256")
                role = _string(assignment, "role")
                split = _string(assignment, "split")
                world_value = assignment.get("world")
                if world_value is not None and type(world_value) is not str:
                    raise ValueError("assignment world must be text or null")
                recipe = _object_record(assignment, "recipe")
                document = {
                    "active_group_token_count": _integer(assignment, "active_token_count"),
                    "adjective_bucket": _integer(occurrence, "adjective_bucket"),
                    "byte_length": len(raw_story),
                    "normalized_sha256": group_sha256,
                    "noun_bucket": _integer(occurrence, "noun_bucket"),
                    "occurrence_id": _string(occurrence, "occurrence_id"),
                    "provenance": assignment["provenance"],
                    "raw_sha256": _string(occurrence, "raw_sha256"),
                    "recipe": recipe,
                    "role": role,
                    "source_byte_offset": _integer(occurrence, "byte_offset"),
                    "split": split,
                    "text_bytes": len(raw_story),
                    "text_offset": text_offset,
                    "text_shard": text_shard,
                    "token_count": len(token_ids),
                    "token_offset": token_byte_offset // 2,
                    "token_shard": token_shard,
                    "verb_bucket": _integer(occurrence, "verb_bucket"),
                    "world": world_value,
                }
                documents.write(canonical_record_bytes(document))
                compact = {
                    "normalized_sha256": group_sha256,
                    "occurrence_id": document["occurrence_id"],
                    "text_bytes": document["text_bytes"],
                    "text_offset": text_offset,
                    "text_shard": text_shard,
                    "token_count": len(token_ids),
                    "token_offset": document["token_offset"],
                    "token_shard": token_shard,
                }
                primary_name = (
                    f"base-{split}.jsonl"
                    if role == "base"
                    else f"world-{world_value}-{split}.jsonl"
                )
                index_streams[primary_name].write(canonical_record_bytes(compact))
                occurrence_counts[(role, world_value, split)] += 1
                if group_sha256 in control_by_group:
                    control_world, control_split = control_by_group[group_sha256]
                    if split != control_split or role != "base":
                        raise ValueError("control owner is not in the corresponding held-in split")
                    index_streams[f"control-{control_world}-{control_split}.jsonl"].write(
                        canonical_record_bytes(compact)
                    )
                    occurrence_counts[("control", control_world, control_split)] += 1
    finally:
        for stream in index_streams.values():
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
    return text_writer.finish() + token_writer.finish(), occurrence_counts


def _iter_assigned_raw_occurrences(
    corpus_path: Path,
    joined_path: Path,
    assignments_path: Path,
) -> Iterator[tuple[dict[str, object], bytes]]:
    assignments = _iter_jsonl(assignments_path)
    assignment = next(assignments, None)
    with corpus_path.open("rb") as corpus:
        for group in iter_joined_groups(joined_path):
            if assignment is None or _string(assignment, "normalized_sha256") != _string(
                group, "normalized_sha256"
            ):
                raise ValueError("joined groups and assignments are not aligned")
            current_assignment = assignment
            assignment = next(assignments, None)
            if current_assignment.get("status") != "eligible":
                continue
            noun_bucket = _integer(current_assignment, "noun_bucket")
            verb_bucket = _integer(current_assignment, "verb_bucket")
            adjective_bucket = _integer(current_assignment, "adjective_bucket")
            for occurrence in _record_list(group, "occurrences"):
                offset = _integer(occurrence, "byte_offset")
                length = _integer(occurrence, "byte_length")
                corpus.seek(offset)
                raw_story = corpus.read(length)
                if len(raw_story) != length or sha256(raw_story).hexdigest() != _string(
                    occurrence, "raw_sha256"
                ):
                    raise ValueError("raw corpus occurrence changed after source join")
                yield (
                    {
                        **occurrence,
                        "adjective_bucket": adjective_bucket,
                        "assignment": current_assignment,
                        "noun_bucket": noun_bucket,
                        "verb_bucket": verb_bucket,
                    },
                    raw_story,
                )
        if assignment is not None:
            raise ValueError("assignment stream contains unknown trailing groups")


def _ordered_tokenize(
    occurrences: Iterable[tuple[dict[str, object], bytes]],
    tokenizer_path: Path,
    worker_count: int,
) -> Iterator[tuple[dict[str, object], bytes, tuple[int, ...]]]:
    batches = _chunked_occurrences(occurrences, _TOKENIZATION_BATCH_SIZE)
    if worker_count == 1:
        _init_tokenizer_worker(str(tokenizer_path))
        for batch in batches:
            yield from _tokenize_occurrence_batch(batch)
        return
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_tokenizer_worker,
        initargs=(str(tokenizer_path),),
    ) as executor:
        pending: dict[Future[tuple[tuple[dict[str, object], bytes, tuple[int, ...]], ...]], int] = {}
        completed: dict[int, tuple[tuple[dict[str, object], bytes, tuple[int, ...]], ...]] = {}
        batch_iterator = enumerate(batches)
        next_output = 0
        for _ in range(worker_count * 2):
            item = next(batch_iterator, None)
            if item is None:
                break
            index, batch = item
            pending[executor.submit(_tokenize_occurrence_batch, batch)] = index
        while pending:
            finished, _ = wait(set(pending), return_when=FIRST_COMPLETED)
            for future in finished:
                index = pending.pop(future)
                completed[index] = future.result()
                item = next(batch_iterator, None)
                if item is not None:
                    following_index, batch = item
                    pending[executor.submit(_tokenize_occurrence_batch, batch)] = following_index
            while next_output in completed:
                yield from completed.pop(next_output)
                next_output += 1


def _init_tokenizer_worker(tokenizer_path: str) -> None:
    global _TOKENIZER_WORKER
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _TOKENIZER_WORKER = TokenizersTextTokenizer.from_file(tokenizer_path)


def _tokenize_occurrence_batch(
    batch: tuple[tuple[dict[str, object], bytes], ...],
) -> tuple[tuple[dict[str, object], bytes, tuple[int, ...]], ...]:
    if _TOKENIZER_WORKER is None:
        raise RuntimeError("tokenizer worker was not initialized")
    return tuple(
        (
            occurrence,
            raw_story,
            _TOKENIZER_WORKER.encode(raw_story.decode("utf-8", errors="strict"), add_eos=True),
        )
        for occurrence, raw_story in batch
    )


def _write_partition_metadata(
    publication: Path,
    inputs: PartitionInputs,
    preset: PartitionPreset,
    joined: JoinedSources,
    partition_sha256: str,
    seed_identity: str,
    assignments_sha256: str,
    buckets: Sequence[WordBucket],
    cells: Sequence[WorldCell],
    controls: Sequence[ControlSelection],
    split_counts: Sequence[SplitCount],
    visibility: Sequence[tuple[str, str, int]],
    shards: Sequence[dict[str, object]],
    occurrence_counts: Mapping[tuple[str, str | None, str], int],
) -> None:
    source_record = {
        "corpus": inputs.corpus_identity.as_record(),
        "metadata": inputs.metadata_identity.as_record(),
        "tokenizer": inputs.tokenizer_identity.as_record(),
    }
    _write_json(publication / "sources.json", source_record)
    _write_json(publication / "normalization.json", NORMALIZATION_IDENTITY.as_record())
    _write_json(
        publication / "buckets.json",
        {
            "buckets": [_bucket_record(bucket) for bucket in buckets],
            "seed_identity_sha256": seed_identity,
        },
    )
    _write_json(publication / "topology.json", {"cells": [_cell_record(cell) for cell in cells]})
    _write_json(
        publication / "controls.json",
        {"controls": [_control_record(control) for control in controls]},
    )
    _write_json(publication / "shards.json", {"shards": list(shards)})
    _write_json(
        publication / "audit.json",
        {
            "component_visibility": [
                {"outside_group_count": count, "role": role, "word": word}
                for role, word, count in visibility
            ],
            "duplicate_group_count": joined.audit.corpus_group_count,
            "duplicate_occurrence_count": joined.audit.corpus_occurrence_count,
            "exclusions": {
                "conflicting_metadata_groups": joined.audit.conflicting_group_count,
                "unclassifiable_metadata_groups": joined.audit.unclassifiable_group_count,
                "unmatched_metadata_groups": joined.audit.unmatched_group_count,
            },
            "forced_word_visibility": [
                {"outside_group_count": count, "role": role, "word": word}
                for role, word, count in visibility
            ],
            "source_join": joined.audit.as_record(),
            "split_counts": [_split_count_record(count) for count in split_counts],
        },
    )
    _write_manifests(publication, split_counts, controls, occurrence_counts)
    _write_json(
        publication / "partition.json",
        {
            "assignments_sha256": assignments_sha256,
            "benchmark_id": BENCHMARK_ID,
            "eos_token_id": joined.eos_token_id,
            "format": PARTITION_FORMAT,
            "normalization": NORMALIZATION_IDENTITY.as_record(),
            "pad_token_id": joined.pad_token_id,
            "partition_sha256": partition_sha256,
            "preset": preset.as_record(),
            "schema_version": PARTITION_SCHEMA_VERSION,
            "seed_identity_sha256": seed_identity,
            "sources": source_record,
        },
    )


def _write_manifests(
    publication: Path,
    split_counts: Sequence[SplitCount],
    controls: Sequence[ControlSelection],
    occurrence_counts: Mapping[tuple[str, str | None, str], int],
) -> None:
    base_counts = [count for count in split_counts if count.role == "base"]
    _write_json(
        publication / "manifests" / "base.json",
        {
            "indexes": [f"indexes/base-{count.split}.jsonl" for count in base_counts],
            "splits": [_split_count_record(count) for count in base_counts],
        },
    )
    for world in WORLD_LABELS:
        world_counts = [count for count in split_counts if count.world == world]
        _write_json(
            publication / "manifests" / f"world-{world}.json",
            {
                "indexes": [
                    f"indexes/world-{world}-{count.split}.jsonl" for count in world_counts
                ],
                "splits": [_split_count_record(count) for count in world_counts],
                "world": world,
            },
        )
    _write_json(
        publication / "manifests" / "controls.json",
        {
            "controls": [
                {
                    **_control_record(control),
                    "index": f"indexes/control-{control.world}-{control.split}.jsonl",
                    "occurrence_count": occurrence_counts.get(
                        ("control", control.world, control.split), 0
                    ),
                }
                for control in controls
            ]
        },
    )


def _write_tree_manifest(publication: Path, partition_sha256: str) -> Path:
    files = tuple(
        ArtifactFile(
            relative_path=path.relative_to(publication).as_posix(),
            size_bytes=path.stat().st_size,
            sha256=_file_sha256(path),
        )
        for path in sorted(
            publication.rglob("*"),
            key=lambda candidate: candidate.relative_to(publication).as_posix(),
        )
        if path.is_file() and path.name != "tree.json"
    )
    tree_path = publication / "tree.json"
    _write_json(
        tree_path,
        {
            "files": [
                {
                    "relative_path": item.relative_path,
                    "sha256": item.sha256,
                    "size_bytes": item.size_bytes,
                }
                for item in files
            ],
            "format": "tinyworlds-p-tree",
            "partition_sha256": partition_sha256,
            "schema_version": 1,
        },
    )
    return tree_path


def _partition_identity(
    inputs: PartitionInputs,
    preset: PartitionPreset,
    buckets: Sequence[WordBucket],
    cells: Sequence[WorldCell],
    controls: Sequence[ControlSelection],
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
                "normalization": NORMALIZATION_IDENTITY.as_record(),
                "preset": preset.as_record(),
                "sources": {
                    "corpus": inputs.corpus_identity.as_record(),
                    "metadata": inputs.metadata_identity.as_record(),
                    "tokenizer": inputs.tokenizer_identity.as_record(),
                },
            }
        )
    ).hexdigest()


def _seed_identity(inputs: PartitionInputs, preset: PartitionPreset) -> str:
    return sha256(
        canonical_record_bytes(
            {
                "benchmark_id": BENCHMARK_ID,
                "normalization": NORMALIZATION_IDENTITY.as_record(),
                "preset": preset.as_record(),
                "sources": {
                    "corpus": inputs.corpus_identity.as_record(),
                    "metadata": inputs.metadata_identity.as_record(),
                    "tokenizer": inputs.tokenizer_identity.as_record(),
                },
            }
        )
    ).hexdigest()


def _allocation_group_record(group: AllocationGroup) -> dict[str, object]:
    return {
        "active_token_count": group.active_token_count,
        "adjective": group.adjective,
        "adjective_bucket": group.adjective_bucket,
        "canonical_token_count": group.canonical_token_count,
        "feature_signature": group.feature_signature,
        "normalized_sha256": group.normalized_sha256,
        "noun": group.noun,
        "noun_bucket": group.noun_bucket,
        "source": group.source,
        "verb": group.verb,
        "verb_bucket": group.verb_bucket,
    }


def _allocation_group_from_record(record: dict[str, object]) -> AllocationGroup:
    return AllocationGroup(
        normalized_sha256=_string(record, "normalized_sha256"),
        active_token_count=_integer(record, "active_token_count"),
        canonical_token_count=_integer(record, "canonical_token_count"),
        noun=_string(record, "noun"),
        verb=_string(record, "verb"),
        adjective=_string(record, "adjective"),
        noun_bucket=_integer(record, "noun_bucket"),
        verb_bucket=_integer(record, "verb_bucket"),
        adjective_bucket=_integer(record, "adjective_bucket"),
        source=_string(record, "source"),
        feature_signature=_string(record, "feature_signature"),
    )


def _allocation_sort_key(record: dict[str, object]) -> tuple[object, ...]:
    return (
        _string(record, "domain"),
        -_integer(record, "active_token_count"),
        _string(record, "order_sha256"),
        _string(record, "normalized_sha256"),
    )


def _flush_records(
    records: list[dict[str, object]],
    directory: Path,
    prefix: str,
    index: int,
    key,
) -> Path:
    path = directory / f"{prefix}-{index:06d}.jsonl"
    with path.open("wb") as output:
        for record in sorted(records, key=key):
            output.write(canonical_record_bytes(record))
    return path


def _merge_records(
    paths: Sequence[Path],
    key,
) -> Iterator[dict[str, object]]:
    streams = tuple(path.open("rb") for path in paths)
    iterators = tuple(_iter_json_stream(stream, path) for stream, path in zip(streams, paths, strict=True))
    heap: list[tuple[tuple[object, ...], int, dict[str, object]]] = []
    try:
        for index, iterator in enumerate(iterators):
            record = next(iterator, None)
            if record is not None:
                heapq.heappush(heap, (key(record), index, record))
        while heap:
            _, index, record = heapq.heappop(heap)
            yield record
            following = next(iterators[index], None)
            if following is not None:
                heapq.heappush(heap, (key(following), index, following))
    finally:
        for stream in streams:
            stream.close()


def _merge_records_to_path(
    paths: Sequence[Path],
    destination: Path,
    key,
) -> None:
    with destination.open("wb") as output:
        for record in _merge_records(paths, key):
            output.write(canonical_record_bytes(record))


def _iter_jsonl(path: Path) -> Iterator[dict[str, object]]:
    with path.open("rb") as stream:
        yield from _iter_json_stream(stream, path)


def _iter_json_stream(stream: BinaryIO, path: Path) -> Iterator[dict[str, object]]:
    for line_number, line in enumerate(stream, start=1):
        try:
            value = json.loads(line)
        except (json.JSONDecodeError, UnicodeError) as error:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
        if type(value) is not dict:
            raise ValueError(f"JSONL record is not an object at {path}:{line_number}")
        yield value


def _chunked_occurrences(
    values: Iterable[tuple[dict[str, object], bytes]],
    size: int,
) -> Iterator[tuple[tuple[dict[str, object], bytes], ...]]:
    pending: list[tuple[dict[str, object], bytes]] = []
    for value in values:
        pending.append(value)
        if len(pending) == size:
            yield tuple(pending)
            pending = []
    if pending:
        yield tuple(pending)


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


def _split_count_record(count: SplitCount) -> dict[str, object]:
    return {
        "active_token_count": count.active_token_count,
        "group_count": count.group_count,
        "occurrence_count": count.occurrence_count,
        "role": count.role,
        "split": count.split,
        "world": count.world,
    }


def _write_json(path: Path, record: object) -> None:
    payload = canonical_record_bytes(record)
    with path.open("wb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())


def _recipe(record: dict[str, object]) -> dict[str, object]:
    return _object_record(record, "recipe")


def _object_record(record: dict[str, object], field: str) -> dict[str, object]:
    value = record.get(field)
    if type(value) is not dict:
        raise ValueError(f"field {field!r} must be an object")
    return value


def _record_list(record: dict[str, object], field: str) -> tuple[dict[str, object], ...]:
    value = record.get(field)
    if type(value) is not list or any(type(item) is not dict for item in value):
        raise ValueError(f"field {field!r} must be a list of objects")
    return tuple(value)


def _string(record: Mapping[str, object], field: str) -> str:
    value = record.get(field)
    if type(value) is not str or not value:
        raise ValueError(f"field {field!r} must be nonempty text")
    return value


def _integer(record: Mapping[str, object], field: str) -> int:
    value = record.get(field)
    if type(value) is not int:
        raise ValueError(f"field {field!r} must be an integer")
    return value


def _namespaced_sha256(identity: str, namespace: str, value: str) -> str:
    return sha256(
        f"tinyworlds-p-v1\0{identity}\0{namespace}\0{value}".encode("utf-8")
    ).hexdigest()


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


def _progress(
    inputs: PartitionInputs,
    phase: str,
    completed: int,
    total: int | None,
    detail: str,
) -> None:
    if inputs.progress is not None:
        inputs.progress(ProgressEvent(phase, completed, total, detail))


__all__ = ["build_partition"]

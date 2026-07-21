"""Bounded external-sort join of raw TinyStories text to released metadata."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from hashlib import sha256
from itertools import groupby
import heapq
import json
import multiprocessing
import os
from pathlib import Path
from typing import BinaryIO, TypeVar

from apm.data.text.curricula import PinnedDatasetFile
from apm.data.text.tinyworlds_p.contracts import (
    NormalizationIdentity,
    PartitionInputs,
    PartitionPreset,
    ProgressEvent,
    SourceIdentity,
    canonical_record_bytes,
)
from apm.data.text.tinyworlds_p.normalization import (
    normalized_story_bytes_sha256,
    normalized_story_sha256,
)
from apm.data.text.tinyworlds_p.roles import recover_released_recipe
from apm.data.text.tinyworlds_v2.source_data import (
    ArchiveSourceRecord,
    TinyStoriesArchiveSource,
    iter_archive_source_records,
)
from apm.lm.text import TokenizersTextTokenizer


_DOCUMENT_SEPARATOR = b"<|endoftext|>"
_READ_BYTES = 4 * 1024 * 1024
_WORK_BATCH_SIZE = 256
_RecordT = TypeVar("_RecordT", bound=dict[str, object])
_WORKER_TOKENIZER: TokenizersTextTokenizer | None = None


class SourceJoinError(ValueError):
    """A source, external-sort run, join, or coverage gate is invalid."""


@dataclass(frozen=True, slots=True)
class JoinAudit:
    """Token-weighted source coverage and exclusion counts from the join."""

    corpus_group_count: int
    corpus_occurrence_count: int
    corpus_token_count: int
    matched_token_count: int
    classified_matched_token_count: int
    eligible_token_count: int
    unmatched_group_count: int
    unclassifiable_group_count: int
    conflicting_group_count: int
    eligible_group_count: int

    @property
    def hash_match_coverage(self) -> float:
        """Return matched corpus-token mass divided by all corpus-token mass."""
        return self.matched_token_count / self.corpus_token_count

    @property
    def role_classification_coverage(self) -> float:
        """Return mechanically classified token mass among matched mass."""
        return self.classified_matched_token_count / self.matched_token_count

    @property
    def eligible_coverage(self) -> float:
        """Return final eligible token mass divided by all corpus-token mass."""
        return self.eligible_token_count / self.corpus_token_count

    def as_record(self) -> dict[str, int | float]:
        """Return canonical persisted coverage and exclusion evidence."""
        return {
            "classified_matched_token_count": self.classified_matched_token_count,
            "conflicting_group_count": self.conflicting_group_count,
            "corpus_group_count": self.corpus_group_count,
            "corpus_occurrence_count": self.corpus_occurrence_count,
            "corpus_token_count": self.corpus_token_count,
            "eligible_coverage": self.eligible_coverage,
            "eligible_group_count": self.eligible_group_count,
            "eligible_token_count": self.eligible_token_count,
            "hash_match_coverage": self.hash_match_coverage,
            "matched_token_count": self.matched_token_count,
            "role_classification_coverage": self.role_classification_coverage,
            "unclassifiable_group_count": self.unclassifiable_group_count,
            "unmatched_group_count": self.unmatched_group_count,
        }


@dataclass(frozen=True, slots=True)
class JoinedSources:
    """Canonical joined-groups file plus tokenizer IDs and its audit."""

    groups_path: Path
    audit: JoinAudit
    pad_token_id: int
    eos_token_id: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "groups_path", Path(self.groups_path))
        if not self.groups_path.is_file():
            raise FileNotFoundError(self.groups_path)
        if any(type(value) is not int or value < 0 for value in (self.pad_token_id, self.eos_token_id)):
            raise ValueError("joined tokenizer IDs must be nonnegative")


def verify_partition_inputs(inputs: PartitionInputs) -> TokenizersTextTokenizer:
    """Verify every pinned byte identity and load the local tokenizer backend."""
    _verify_source_file(inputs.corpus_path, inputs.corpus_identity)
    _verify_source_file(inputs.metadata_archive_path, inputs.metadata_identity)
    if not inputs.tokenizer_directory.is_dir():
        raise FileNotFoundError(inputs.tokenizer_directory)
    actual_names = {path.name for path in inputs.tokenizer_directory.iterdir() if path.is_file()}
    expected_names = {item.name for item in inputs.tokenizer_identity.files}
    if actual_names != expected_names:
        raise SourceJoinError(
            "tokenizer directory entries changed; "
            f"missing={tuple(sorted(expected_names - actual_names))}, "
            f"unexpected={tuple(sorted(actual_names - expected_names))}"
        )
    for expected in inputs.tokenizer_identity.files:
        path = inputs.tokenizer_directory / expected.name
        measured_size, measured_sha256 = _file_identity(path)
        if (measured_size, measured_sha256) != (expected.size_bytes, expected.sha256):
            raise SourceJoinError(f"tokenizer file identity changed: {expected.name}")
    tokenizer = TokenizersTextTokenizer.from_file(
        inputs.tokenizer_directory / "tokenizer.json"
    )
    if tokenizer.vocab_size != inputs.tokenizer_identity.vocab_size:
        raise SourceJoinError("loaded tokenizer vocabulary differs from its contract")
    return tokenizer


def build_source_join(
    inputs: PartitionInputs,
    preset: PartitionPreset,
    normalization: NormalizationIdentity,
) -> JoinedSources:
    """Normalize/tokenize bounded runs, external-merge, join, and enforce coverage."""
    if type(inputs) is not PartitionInputs or type(preset) is not PartitionPreset:
        raise TypeError("source join requires PartitionInputs and PartitionPreset")
    if type(normalization) is not NormalizationIdentity:
        raise TypeError("normalization must be NormalizationIdentity")
    tokenizer = verify_partition_inputs(inputs)
    working = inputs.temporary_directory
    if working.exists() and any(working.iterdir()):
        raise FileExistsError(f"partition temporary directory is not empty: {working}")
    working.mkdir(parents=True, exist_ok=True)
    corpus_runs_directory = working / "corpus-runs"
    metadata_runs_directory = working / "metadata-runs"
    corpus_runs_directory.mkdir()
    metadata_runs_directory.mkdir()
    _progress(inputs, "corpus", 0, None, "normalizing and tokenizing raw occurrences")
    corpus_runs = _build_corpus_runs(
        inputs.corpus_path,
        inputs.tokenizer_directory / "tokenizer.json",
        corpus_runs_directory,
        preset,
    )
    _progress(inputs, "metadata", 0, None, "normalizing released recipes")
    metadata_runs = _build_metadata_runs(
        inputs.metadata_archive_path,
        inputs.metadata_identity,
        metadata_runs_directory,
        preset,
    )
    _progress(inputs, "join", 0, None, "external-merging normalized SHA-256 groups")
    groups_path = working / "joined-groups.jsonl"
    audit = _join_runs(corpus_runs, metadata_runs, groups_path)
    _write_join_audit(working / "join-audit.json", audit, preset)
    _require_coverage(audit, preset)
    _progress(inputs, "join", audit.corpus_group_count, audit.corpus_group_count, "coverage gates passed")
    return JoinedSources(
        groups_path=groups_path,
        audit=audit,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )


def iter_joined_groups(path: str | Path) -> Iterator[dict[str, object]]:
    """Stream canonical joined group records from a preparation run."""
    source_path = Path(path)
    with source_path.open("rb") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                value = json.loads(line)
            except (json.JSONDecodeError, UnicodeError) as error:
                raise SourceJoinError(
                    f"invalid joined group at line {line_number}: {error}"
                ) from error
            if type(value) is not dict:
                raise SourceJoinError(f"joined group line {line_number} is not an object")
            yield value


def iter_raw_corpus_occurrences(
    path: str | Path,
) -> Iterator[tuple[int, int, bytes]]:
    """Yield source index, byte offset, and exact bytes between EOS separators."""
    source_path = Path(path)
    with source_path.open("rb") as source:
        buffer = b""
        buffer_offset = 0
        source_index = 0
        while chunk := source.read(_READ_BYTES):
            buffer += chunk
            while (separator_index := buffer.find(_DOCUMENT_SEPARATOR)) >= 0:
                raw_story = buffer[:separator_index]
                if raw_story.strip():
                    yield source_index, buffer_offset, raw_story
                    source_index += 1
                consumed = separator_index + len(_DOCUMENT_SEPARATOR)
                buffer = buffer[consumed:]
                buffer_offset += consumed
        if buffer.strip():
            yield source_index, buffer_offset, buffer


def _build_corpus_runs(
    corpus_path: Path,
    tokenizer_path: Path,
    runs_directory: Path,
    preset: PartitionPreset,
) -> tuple[Path, ...]:
    raw_batches = _chunked(iter_raw_corpus_occurrences(corpus_path), _WORK_BATCH_SIZE)
    if preset.worker_count == 1:
        _init_corpus_worker(str(tokenizer_path))
        records = (
            record
            for batch in raw_batches
            for record in _normalize_corpus_batch(batch)
        )
    else:
        records = (
            record
            for result in _bounded_process_map(
                raw_batches,
                _normalize_corpus_batch,
                preset.worker_count,
                initializer=_init_corpus_worker,
                initargs=(str(tokenizer_path),),
            )
            for record in result
        )
    return _write_sorted_runs(
        records,
        runs_directory,
        "corpus",
        preset.run_record_count,
        ("normalized_sha256", "occurrence_id"),
    )


def _build_metadata_runs(
    archive_path: Path,
    identity: SourceIdentity,
    runs_directory: Path,
    preset: PartitionPreset,
) -> tuple[Path, ...]:
    source = TinyStoriesArchiveSource(
        dataset_id=identity.dataset_id,
        revision=identity.revision,
        archive_file=PinnedDatasetFile(
            identity.filename,
            identity.size_bytes,
            identity.sha256,
        ),
    )
    record_batches = _chunked(iter_archive_source_records(archive_path, source), _WORK_BATCH_SIZE)
    if preset.worker_count == 1:
        records = (
            record
            for batch in record_batches
            for record in _normalize_metadata_batch(batch)
        )
    else:
        records = (
            record
            for result in _bounded_process_map(
                record_batches,
                _normalize_metadata_batch,
                preset.worker_count,
            )
            for record in result
        )
    return _write_sorted_runs(
        records,
        runs_directory,
        "metadata",
        preset.run_record_count,
        ("normalized_sha256", "record_id"),
    )


def _init_corpus_worker(tokenizer_path: str) -> None:
    global _WORKER_TOKENIZER
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _WORKER_TOKENIZER = TokenizersTextTokenizer.from_file(tokenizer_path)


def _normalize_corpus_batch(
    batch: tuple[tuple[int, int, bytes], ...],
) -> tuple[dict[str, object], ...]:
    if _WORKER_TOKENIZER is None:
        raise RuntimeError("corpus worker tokenizer was not initialized")
    records: list[dict[str, object]] = []
    for source_index, byte_offset, raw_story in batch:
        text = raw_story.decode("utf-8", errors="strict")
        token_count = len(_WORKER_TOKENIZER.encode(text, add_eos=True))
        if token_count == 0:
            raise SourceJoinError(f"corpus occurrence {source_index} tokenized empty")
        records.append(
            {
                "byte_length": len(raw_story),
                "byte_offset": byte_offset,
                "normalized_sha256": normalized_story_bytes_sha256(raw_story),
                "occurrence_id": f"train:{source_index:09d}",
                "raw_sha256": sha256(raw_story).hexdigest(),
                "source_index": source_index,
                "token_count": token_count,
            }
        )
    return tuple(records)


def _normalize_metadata_batch(
    batch: tuple[ArchiveSourceRecord, ...],
) -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    for record in batch:
        try:
            recipe = recover_released_recipe(
                record.instruction.prompt,
                record.instruction.words,
                record.instruction.features,
            )
        except ValueError:
            recipe = None
        records.append(
            {
                "content_sha256": record.content_sha256,
                "normalized_sha256": normalized_story_sha256(record.story),
                "recipe": None if recipe is None else recipe.as_record(),
                "record_id": record.record_id,
                "source": record.source,
                "source_index": record.source_index,
                "source_member": record.source_member,
            }
        )
    return tuple(records)


def _join_runs(
    corpus_runs: Sequence[Path],
    metadata_runs: Sequence[Path],
    destination: Path,
) -> JoinAudit:
    corpus_groups = _group_sorted_records(
        _merge_sorted_runs(corpus_runs, ("normalized_sha256", "occurrence_id"))
    )
    metadata_groups = _group_sorted_records(
        _merge_sorted_runs(metadata_runs, ("normalized_sha256", "record_id"))
    )
    corpus_current = next(corpus_groups, None)
    metadata_current = next(metadata_groups, None)
    counters = {
        "corpus_group_count": 0,
        "corpus_occurrence_count": 0,
        "corpus_token_count": 0,
        "matched_token_count": 0,
        "classified_matched_token_count": 0,
        "eligible_token_count": 0,
        "unmatched_group_count": 0,
        "unclassifiable_group_count": 0,
        "conflicting_group_count": 0,
        "eligible_group_count": 0,
    }
    with destination.open("wb") as output:
        while corpus_current is not None:
            normalized_sha256, occurrences = corpus_current
            while metadata_current is not None and metadata_current[0] < normalized_sha256:
                metadata_current = next(metadata_groups, None)
            metadata = (
                metadata_current[1]
                if metadata_current is not None and metadata_current[0] == normalized_sha256
                else ()
            )
            if metadata:
                metadata_current = next(metadata_groups, None)
            token_count = sum(_integer(item, "token_count") for item in occurrences)
            status, recipe = _resolve_group_recipe(metadata)
            counters["corpus_group_count"] += 1
            counters["corpus_occurrence_count"] += len(occurrences)
            counters["corpus_token_count"] += token_count
            counter_name = {
                "eligible": "eligible_group_count",
                "unmatched_metadata": "unmatched_group_count",
                "unclassifiable_metadata": "unclassifiable_group_count",
                "conflicting_metadata": "conflicting_group_count",
            }[status]
            counters[counter_name] += 1
            if metadata:
                counters["matched_token_count"] += token_count
            if metadata and all(item.get("recipe") is not None for item in metadata):
                counters["classified_matched_token_count"] += token_count
            if status == "eligible":
                counters["eligible_token_count"] += token_count
            output.write(
                canonical_record_bytes(
                    {
                        "active_token_count": token_count,
                        "normalized_sha256": normalized_sha256,
                        "occurrences": list(occurrences),
                        "provenance": [
                            {
                                "content_sha256": item["content_sha256"],
                                "record_id": item["record_id"],
                                "source": item["source"],
                                "source_index": item["source_index"],
                                "source_member": item["source_member"],
                            }
                            for item in metadata
                        ],
                        "recipe": recipe,
                        "status": status,
                    }
                )
            )
            corpus_current = next(corpus_groups, None)
    if counters["corpus_token_count"] == 0 or counters["matched_token_count"] == 0:
        raise SourceJoinError("source join produced no token-bearing matched corpus groups")
    return JoinAudit(**counters)


def _resolve_group_recipe(
    metadata: Sequence[dict[str, object]],
) -> tuple[str, dict[str, object] | None]:
    if not metadata:
        return "unmatched_metadata", None
    recipes = tuple(item.get("recipe") for item in metadata)
    if any(recipe is None for recipe in recipes):
        return "unclassifiable_metadata", None
    canonical_recipes = {canonical_record_bytes(recipe) for recipe in recipes}
    if len(canonical_recipes) != 1:
        return "conflicting_metadata", None
    recipe = recipes[0]
    if type(recipe) is not dict:
        raise SourceJoinError("classified metadata recipe is not an object")
    return "eligible", recipe


def _require_coverage(audit: JoinAudit, preset: PartitionPreset) -> None:
    failures = tuple(
        f"{label}={measured:.6f} < {minimum:.6f}"
        for label, measured, minimum in (
            (
                "hash_match_coverage",
                audit.hash_match_coverage,
                preset.minimum_hash_match_coverage,
            ),
            (
                "role_classification_coverage",
                audit.role_classification_coverage,
                preset.minimum_role_coverage,
            ),
            (
                "eligible_coverage",
                audit.eligible_coverage,
                preset.minimum_eligible_coverage,
            ),
        )
        if measured < minimum
    )
    if failures:
        raise SourceJoinError("partition source coverage gates failed: " + "; ".join(failures))


def _write_join_audit(
    path: Path,
    audit: JoinAudit,
    preset: PartitionPreset,
) -> None:
    measured_gates = (
        (
            "hash_match_coverage",
            audit.hash_match_coverage,
            preset.minimum_hash_match_coverage,
        ),
        (
            "role_classification_coverage",
            audit.role_classification_coverage,
            preset.minimum_role_coverage,
        ),
        (
            "eligible_coverage",
            audit.eligible_coverage,
            preset.minimum_eligible_coverage,
        ),
    )
    payload = canonical_record_bytes(
        {
            "coverage": audit.as_record(),
            "format": "tinyworlds-p-source-join-audit",
            "gates": [
                {
                    "minimum": minimum,
                    "name": name,
                    "passed": measured >= minimum,
                    "value": measured,
                }
                for name, measured, minimum in measured_gates
            ],
            "passed": all(
                measured >= minimum for _, measured, minimum in measured_gates
            ),
            "schema_version": 1,
        }
    )
    with path.open("wb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())


def _write_sorted_runs(
    records: Iterable[dict[str, object]],
    directory: Path,
    prefix: str,
    run_record_count: int,
    key_fields: tuple[str, ...],
) -> tuple[Path, ...]:
    paths: list[Path] = []
    pending: list[dict[str, object]] = []
    for record in records:
        pending.append(record)
        if len(pending) >= run_record_count:
            paths.append(_flush_run(pending, directory, prefix, len(paths), key_fields))
            pending = []
    if pending:
        paths.append(_flush_run(pending, directory, prefix, len(paths), key_fields))
    if not paths:
        raise SourceJoinError(f"{prefix} source produced no sortable records")
    return tuple(paths)


def _flush_run(
    records: list[dict[str, object]],
    directory: Path,
    prefix: str,
    index: int,
    key_fields: tuple[str, ...],
) -> Path:
    path = directory / f"{prefix}-{index:06d}.jsonl"
    ordered = sorted(records, key=lambda record: tuple(record[field] for field in key_fields))
    with path.open("wb") as output:
        for record in ordered:
            output.write(canonical_record_bytes(record))
    return path


def _merge_sorted_runs(
    paths: Sequence[Path],
    key_fields: tuple[str, ...],
) -> Iterator[dict[str, object]]:
    streams = tuple(path.open("rb") for path in paths)
    try:
        iterators = tuple(_iter_run(stream, path) for stream, path in zip(streams, paths, strict=True))
        heap: list[tuple[tuple[object, ...], int, dict[str, object]]] = []
        for index, iterator in enumerate(iterators):
            record = next(iterator, None)
            if record is not None:
                heapq.heappush(
                    heap,
                    (tuple(record[field] for field in key_fields), index, record),
                )
        while heap:
            _, index, record = heapq.heappop(heap)
            yield record
            following = next(iterators[index], None)
            if following is not None:
                heapq.heappush(
                    heap,
                    (
                        tuple(following[field] for field in key_fields),
                        index,
                        following,
                    ),
                )
    finally:
        for stream in streams:
            stream.close()


def _iter_run(stream: BinaryIO, path: Path) -> Iterator[dict[str, object]]:
    for line_number, line in enumerate(stream, start=1):
        try:
            value = json.loads(line)
        except (json.JSONDecodeError, UnicodeError) as error:
            raise SourceJoinError(f"invalid sort run {path}:{line_number}") from error
        if type(value) is not dict:
            raise SourceJoinError(f"sort run record is not an object: {path}:{line_number}")
        yield value


def _group_sorted_records(
    records: Iterable[dict[str, object]],
) -> Iterator[tuple[str, tuple[dict[str, object], ...]]]:
    for normalized_sha256, grouped in groupby(
        records,
        key=lambda record: record["normalized_sha256"],
    ):
        if type(normalized_sha256) is not str:
            raise SourceJoinError("normalized group key is not text")
        yield normalized_sha256, tuple(grouped)


def _bounded_process_map(
    batches: Iterable[tuple[object, ...]],
    function,
    worker_count: int,
    *,
    initializer=None,
    initargs: tuple[object, ...] = (),
) -> Iterator[tuple[dict[str, object], ...]]:
    batch_iterator = iter(batches)
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=initializer,
        initargs=initargs,
    ) as executor:
        pending: set[Future[tuple[dict[str, object], ...]]] = set()
        for _ in range(worker_count * 2):
            batch = next(batch_iterator, None)
            if batch is None:
                break
            pending.add(executor.submit(function, batch))
        while pending:
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                yield future.result()
                batch = next(batch_iterator, None)
                if batch is not None:
                    pending.add(executor.submit(function, batch))


def _chunked(values: Iterable[object], size: int) -> Iterator[tuple[object, ...]]:
    pending: list[object] = []
    for value in values:
        pending.append(value)
        if len(pending) == size:
            yield tuple(pending)
            pending = []
    if pending:
        yield tuple(pending)


def _verify_source_file(path: Path, identity: SourceIdentity) -> None:
    if path.name != identity.filename:
        raise SourceJoinError(
            f"expected source filename {identity.filename!r}, got {path.name!r}"
        )
    measured = _file_identity(path)
    if measured != (identity.size_bytes, identity.sha256):
        raise SourceJoinError(
            f"source identity changed for {identity.filename}: "
            f"measured_size={measured[0]}, measured_sha256={measured[1]}"
        )


def _file_identity(path: Path) -> tuple[int, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = sha256()
    size = 0
    with path.open("rb") as source:
        while chunk := source.read(_READ_BYTES):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _integer(record: dict[str, object], field: str) -> int:
    value = record[field]
    if type(value) is not int:
        raise SourceJoinError(f"joined field {field!r} must be an integer")
    return value


def _progress(
    inputs: PartitionInputs,
    phase: str,
    completed: int,
    total: int | None,
    detail: str,
) -> None:
    if inputs.progress is not None:
        inputs.progress(ProgressEvent(phase, completed, total, detail))


__all__ = [
    "JoinAudit",
    "JoinedSources",
    "SourceJoinError",
    "build_source_join",
    "iter_joined_groups",
    "iter_raw_corpus_occurrences",
    "verify_partition_inputs",
]

"""Strict, bounded, archive-native ingestion for TinyWorlds-P."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
import codecs
from dataclasses import dataclass
from hashlib import sha256
import heapq
from itertools import groupby
import json
import multiprocessing
import os
from pathlib import Path
import tarfile
from typing import BinaryIO, TypeVar

import numpy as np

from apm.data.text.tinyworlds_p.contracts import (
    ArchiveIngestAudit,
    ArchiveOccurrence,
    NormalizationIdentity,
    PartitionInputs,
    PartitionPreset,
    ProgressEvent,
    Recipe,
    canonical_record_bytes,
)
from apm.data.text.tinyworlds_p.normalization import normalized_story_sha256
from apm.data.text.tinyworlds_p.roles import recover_released_recipe
from apm.lm.text import TokenizersTextTokenizer


_JSON_READ_CHARACTERS = 64 * 1024
_MAX_SOURCE_RECORD_CHARACTERS = 16 * 1024 * 1024
_READ_BYTES = 4 * 1024 * 1024
_WORK_BATCH_SIZE = 256
_RecordT = TypeVar("_RecordT", bound=dict[str, object])
_WORKER_TOKENIZER: TokenizersTextTokenizer | None = None


class ArchiveIngestError(ValueError):
    """The archive, tokenizer, sorted runs, or role gate is invalid."""


@dataclass(frozen=True, slots=True)
class ArchiveIngestResult:
    """Canonical grouped archive stream, exact-byte spool, audit, and token IDs."""

    groups_path: Path
    story_spool_path: Path
    token_spool_path: Path
    audit_path: Path
    audit: ArchiveIngestAudit
    pad_token_id: int
    eos_token_id: int

    def __post_init__(self) -> None:
        for name in (
            "groups_path",
            "story_spool_path",
            "token_spool_path",
            "audit_path",
        ):
            path = Path(getattr(self, name))
            object.__setattr__(self, name, path)
            if not path.is_file():
                raise FileNotFoundError(path)
        if type(self.audit) is not ArchiveIngestAudit:
            raise TypeError("archive ingest result requires an ArchiveIngestAudit")
        if any(
            type(value) is not int or value < 0
            for value in (self.pad_token_id, self.eos_token_id)
        ):
            raise ValueError("archive ingest tokenizer IDs must be nonnegative")


@dataclass(frozen=True, slots=True)
class _ParsedArchiveRecord:
    member: str
    index: int
    content_sha256: str
    record_id: str
    story: str
    prompt: str
    words: tuple[str, ...]
    features: tuple[str, ...]
    source: str


@dataclass(frozen=True, slots=True)
class _ArchiveWorkRecord:
    member: str
    index: int
    content_sha256: str
    record_id: str
    story: str
    prompt: str
    words: tuple[str, ...]
    features: tuple[str, ...]
    source: str
    spool_offset: int
    byte_length: int


@dataclass(slots=True)
class _ArchiveStreamCounts:
    member_count: int = 0
    record_count: int = 0


class _HashingReader:
    """Hash compressed archive bytes while tarfile consumes one stream."""

    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream
        self._digest = sha256()
        self.size_bytes = 0

    @property
    def hexdigest(self) -> str:
        return self._digest.hexdigest()

    def read(self, size: int = -1) -> bytes:
        payload = self._stream.read(size)
        self._digest.update(payload)
        self.size_bytes += len(payload)
        return payload


class _Utf8ChunkReader:
    """Incrementally decode a non-seekable binary tar member as strict UTF-8."""

    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
        self._finished = False

    def read(self, size: int) -> str:
        if self._finished:
            return ""
        payload = self._stream.read(size)
        if payload:
            return self._decoder.decode(payload, final=False)
        self._finished = True
        return self._decoder.decode(b"", final=True)


class _StreamingJsonArray:
    """Stateful decoder that retains no more than one bounded JSON record."""

    def __init__(self, stream: _Utf8ChunkReader, label: str) -> None:
        self._stream = stream
        self._label = label
        self._buffer = ""
        self._position = 0
        self._eof = False
        self._decoder = json.JSONDecoder(
            object_pairs_hook=self._strict_object,
            parse_constant=self._reject_constant,
        )

    def records(self) -> Iterator[object]:
        """Yield top-level array values while enforcing exact JSON syntax."""
        if self._next_character() != "[":
            raise ArchiveIngestError(f"{self._label} must contain a JSON array")
        self._position += 1
        if self._next_character() == "]":
            self._position += 1
            self._require_only_trailing_whitespace()
            return
        while True:
            yield self._decode_value()
            character = self._next_character()
            if character == "]":
                self._position += 1
                self._require_only_trailing_whitespace()
                return
            if character != ",":
                detail = "end of input" if character is None else repr(character)
                raise ArchiveIngestError(
                    f"expected array delimiter in {self._label}, got {detail}"
                )
            self._position += 1
            if self._next_character() == "]":
                raise ArchiveIngestError(
                    f"trailing comma in JSON array {self._label}"
                )
            self._compact()

    def _decode_value(self) -> object:
        start = self._position
        while True:
            try:
                value, end = self._decoder.raw_decode(self._buffer, self._position)
            except json.JSONDecodeError as error:
                if self._eof:
                    raise ArchiveIngestError(
                        f"invalid JSON record in {self._label}: {error.msg}"
                    ) from error
                if len(self._buffer) - start > _MAX_SOURCE_RECORD_CHARACTERS:
                    raise ArchiveIngestError(
                        f"JSON record exceeds safety limit in {self._label}"
                    )
                self._read_more()
                continue
            if end - start > _MAX_SOURCE_RECORD_CHARACTERS:
                raise ArchiveIngestError(
                    f"JSON record exceeds safety limit in {self._label}"
                )
            self._position = end
            return value

    def _next_character(self) -> str | None:
        while True:
            while (
                self._position < len(self._buffer)
                and self._buffer[self._position].isspace()
            ):
                self._position += 1
            if self._position < len(self._buffer):
                return self._buffer[self._position]
            if self._eof:
                return None
            self._compact()
            self._read_more()

    def _read_more(self) -> None:
        chunk = self._stream.read(_JSON_READ_CHARACTERS)
        if chunk:
            self._buffer += chunk
        else:
            self._eof = True

    def _compact(self) -> None:
        if self._position:
            self._buffer = self._buffer[self._position :]
            self._position = 0

    def _require_only_trailing_whitespace(self) -> None:
        while not self._eof:
            self._read_more()
        if self._buffer[self._position :].strip():
            raise ArchiveIngestError(f"trailing JSON data in {self._label}")

    def _strict_object(self, pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ArchiveIngestError(
                    f"duplicate field {key!r} in {self._label}"
                )
            result[key] = value
        return result

    def _reject_constant(self, value: str) -> float:
        raise ArchiveIngestError(
            f"non-finite number {value!r} in {self._label}"
        )


def verify_partition_inputs(inputs: PartitionInputs) -> TokenizersTextTokenizer:
    """Check source boundaries and authenticate every local tokenizer file."""
    if type(inputs) is not PartitionInputs:
        raise TypeError("archive ingestion requires PartitionInputs")
    if inputs.archive_path.name != inputs.archive_identity.filename:
        raise ArchiveIngestError(
            f"expected archive filename {inputs.archive_identity.filename!r}, "
            f"got {inputs.archive_path.name!r}"
        )
    if not inputs.archive_path.is_file():
        raise FileNotFoundError(inputs.archive_path)
    if inputs.archive_path.stat().st_size != inputs.archive_identity.size_bytes:
        raise ArchiveIngestError("archive byte size differs from its pinned identity")
    if not inputs.tokenizer_directory.is_dir():
        raise FileNotFoundError(inputs.tokenizer_directory)
    actual_names = {
        path.name for path in inputs.tokenizer_directory.iterdir() if path.is_file()
    }
    expected_names = {item.name for item in inputs.tokenizer_identity.files}
    if actual_names != expected_names:
        raise ArchiveIngestError(
            "tokenizer directory entries changed; "
            f"missing={tuple(sorted(expected_names - actual_names))}, "
            f"unexpected={tuple(sorted(actual_names - expected_names))}"
        )
    for expected in inputs.tokenizer_identity.files:
        path = inputs.tokenizer_directory / expected.name
        if _file_identity(path) != (expected.size_bytes, expected.sha256):
            raise ArchiveIngestError(
                f"tokenizer file identity changed: {expected.name}"
            )
    tokenizer = TokenizersTextTokenizer.from_file(
        inputs.tokenizer_directory / "tokenizer.json"
    )
    if tokenizer.vocab_size != inputs.tokenizer_identity.vocab_size:
        raise ArchiveIngestError(
            "loaded tokenizer vocabulary differs from its contract"
        )
    return tokenizer


def build_archive_ingest(
    inputs: PartitionInputs,
    preset: PartitionPreset,
    normalization: NormalizationIdentity,
) -> ArchiveIngestResult:
    """Stream, spool, classify, sort, group, audit, and gate the archive."""
    if type(preset) is not PartitionPreset:
        raise TypeError("archive ingestion requires a PartitionPreset")
    if type(normalization) is not NormalizationIdentity:
        raise TypeError("normalization must be NormalizationIdentity")
    tokenizer = verify_partition_inputs(inputs)
    working = inputs.temporary_directory
    if working.exists() and any(working.iterdir()):
        raise FileExistsError(f"partition temporary directory is not empty: {working}")
    working.mkdir(parents=True, exist_ok=True)
    runs_directory = working / "archive-runs"
    runs_directory.mkdir()
    spool_path = working / "archive-stories.bin"
    token_spool_path = working / "archive-tokens.uint16"
    counts = _ArchiveStreamCounts()
    _progress(
        inputs,
        "archive",
        0,
        inputs.archive_identity.size_bytes,
        "streaming, spooling, and classifying records",
    )
    batches = _archive_work_batches(inputs, spool_path, counts)
    tokenizer_path = inputs.tokenizer_directory / "tokenizer.json"
    if preset.worker_count == 1:
        _init_archive_worker(str(tokenizer_path))
        processed_records = (
            record for batch in batches for record in _process_archive_batch(batch)
        )
    else:
        processed_records = (
            record
            for result in _bounded_process_map(
                batches,
                _process_archive_batch,
                preset.worker_count,
                initializer=_init_archive_worker,
                initargs=(str(tokenizer_path),),
            )
            for record in result
        )
    runs = _write_sorted_runs(
        _spool_processed_tokens(processed_records, token_spool_path),
        runs_directory,
        "archive",
        preset.run_record_count,
        ("normalized_story_sha256", "record_id"),
    )
    groups_path = working / "archive-groups.jsonl"
    audit = _write_groups_and_audit(runs, groups_path, counts)
    audit_path = working / "archive-ingest.json"
    _write_ingest_audit(audit_path, audit, preset)
    _require_role_coverage(audit, preset)
    _progress(
        inputs,
        "archive",
        inputs.archive_identity.size_bytes,
        inputs.archive_identity.size_bytes,
        f"archive identity and role gate passed for {audit.archive_record_count:,} records",
    )
    return ArchiveIngestResult(
        groups_path=groups_path,
        story_spool_path=spool_path,
        token_spool_path=token_spool_path,
        audit_path=audit_path,
        audit=audit,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )


def iter_archive_groups(path: str | Path) -> Iterator[dict[str, object]]:
    """Stream canonical archive duplicate-group records."""
    source_path = Path(path)
    with source_path.open("rb") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                value = json.loads(line)
            except (json.JSONDecodeError, UnicodeError) as error:
                raise ArchiveIngestError(
                    f"invalid archive group at line {line_number}: {error}"
                ) from error
            if type(value) is not dict:
                raise ArchiveIngestError(
                    f"archive group line {line_number} is not an object"
                )
            if canonical_record_bytes(value) != line:
                raise ArchiveIngestError(
                    f"archive group line {line_number} is not canonical JSON"
                )
            yield value


def read_spooled_story(spool_path: str | Path, occurrence: dict[str, object]) -> bytes:
    """Read and authenticate one exact story from the temporary archive spool."""
    offset = _integer(occurrence, "spool_offset")
    byte_length = _integer(occurrence, "byte_length")
    with Path(spool_path).open("rb") as spool:
        spool.seek(offset)
        payload = spool.read(byte_length)
    if len(payload) != byte_length:
        raise ArchiveIngestError("archive story spool coordinates exceed the file")
    expected = _text(occurrence, "story_sha256")
    if sha256(payload).hexdigest() != expected:
        raise ArchiveIngestError("archive story spool content hash changed")
    return payload


def read_spooled_tokens(
    spool_path: str | Path,
    occurrence: dict[str, object],
) -> tuple[int, ...]:
    """Read one exact token sequence from the temporary archive token spool."""
    offset = _integer(occurrence, "token_spool_offset")
    token_count = _integer(occurrence, "token_count")
    with Path(spool_path).open("rb") as spool:
        spool.seek(offset * 2)
        payload = spool.read(token_count * 2)
    tokens = np.frombuffer(payload, dtype="<u2")
    if len(tokens) != token_count:
        raise ArchiveIngestError("archive token spool coordinates exceed the file")
    return tuple(int(token) for token in tokens)


def _archive_work_batches(
    inputs: PartitionInputs,
    spool_path: Path,
    counts: _ArchiveStreamCounts,
) -> Iterator[tuple[_ArchiveWorkRecord, ...]]:
    pending: list[_ArchiveWorkRecord] = []
    with spool_path.open("wb") as spool:
        for record in _iter_authenticated_archive(inputs, counts):
            story_bytes = record.story.encode("utf-8")
            offset = spool.tell()
            spool.write(story_bytes)
            pending.append(
                _ArchiveWorkRecord(
                    member=record.member,
                    index=record.index,
                    content_sha256=record.content_sha256,
                    record_id=record.record_id,
                    story=record.story,
                    prompt=record.prompt,
                    words=record.words,
                    features=record.features,
                    source=record.source,
                    spool_offset=offset,
                    byte_length=len(story_bytes),
                )
            )
            if len(pending) == _WORK_BATCH_SIZE:
                yield tuple(pending)
                pending = []
        if pending:
            yield tuple(pending)
        spool.flush()
        os.fsync(spool.fileno())


def _iter_authenticated_archive(
    inputs: PartitionInputs,
    counts: _ArchiveStreamCounts,
) -> Iterator[_ParsedArchiveRecord]:
    path = inputs.archive_path
    identity = inputs.archive_identity
    initial_stat = path.stat()
    if initial_stat.st_size != identity.size_bytes:
        raise ArchiveIngestError("archive size changed before streaming")
    json_members: set[str] = set()
    try:
        with path.open("rb") as raw:
            hashing = _HashingReader(raw)
            with tarfile.open(fileobj=hashing, mode="r|gz") as archive:
                for member in archive:
                    _validate_member_name(member.name)
                    if member.isdir():
                        continue
                    if not member.isfile():
                        raise ArchiveIngestError(
                            f"unsupported archive member type: {member.name!r}"
                        )
                    if not member.name.endswith(".json"):
                        raise ArchiveIngestError(
                            f"unexpected non-JSON archive member: {member.name!r}"
                        )
                    if member.name in json_members:
                        raise ArchiveIngestError(
                            f"duplicate JSON archive member: {member.name!r}"
                        )
                    json_members.add(member.name)
                    counts.member_count += 1
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        raise ArchiveIngestError(
                            f"could not stream archive member: {member.name!r}"
                        )
                    for value in _records_from_member(extracted, member.name):
                        counts.record_count += 1
                        yield value
                    _progress(
                        inputs,
                        "archive",
                        min(hashing.size_bytes, identity.size_bytes),
                        identity.size_bytes,
                        f"streamed {counts.record_count:,} archive records",
                    )
            while hashing.read(_READ_BYTES):
                pass
    except ArchiveIngestError:
        raise
    except (tarfile.TarError, UnicodeError, OSError) as error:
        raise ArchiveIngestError(
            f"could not read pinned archive {path}: {error}"
        ) from error
    if not json_members or counts.record_count == 0:
        raise ArchiveIngestError("TinyStories archive contains no JSON records")
    final_stat = path.stat()
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(initial_stat, name) != getattr(final_stat, name) for name in stable_fields):
        raise ArchiveIngestError("archive identity changed while it was being streamed")
    measured = hashing.size_bytes, hashing.hexdigest
    if measured != (identity.size_bytes, identity.sha256):
        raise ArchiveIngestError(
            f"archive identity changed: measured_size={measured[0]}, "
            f"measured_sha256={measured[1]}"
        )


def _records_from_member(
    extracted: BinaryIO,
    member_name: str,
) -> Iterator[_ParsedArchiveRecord]:
    text = _Utf8ChunkReader(extracted)
    for source_index, value in enumerate(
        _StreamingJsonArray(text, member_name).records()
    ):
        yield _parse_archive_record(value, member_name, source_index)


def _parse_archive_record(
    value: object,
    member_name: str,
    source_index: int,
) -> _ParsedArchiveRecord:
    label = f"{member_name}[{source_index}]"
    record = _require_exact_object(
        value,
        ("story", "instruction", "summary", "source"),
        label,
    )
    instruction = _require_exact_object(
        record["instruction"],
        ("prompt:", "words", "features"),
        f"{label}.instruction",
    )
    prompt = _require_string(instruction["prompt:"], f"{label}.instruction.prompt:")
    words = _require_string_array(instruction["words"], f"{label}.instruction.words")
    features = _require_string_array(
        instruction["features"],
        f"{label}.instruction.features",
    )
    story = _require_string(record["story"], f"{label}.story")
    summary = _require_string(record["summary"], f"{label}.summary")
    source = _require_string(record["source"], f"{label}.source")
    if source not in ("GPT-3.5", "GPT-4"):
        raise ArchiveIngestError(
            f"unsupported released source label at {label}: {source!r}"
        )
    source_record = {
        "instruction": {
            "features": list(features),
            "prompt:": prompt,
            "words": list(words),
        },
        "source": source,
        "story": story,
        "summary": summary,
    }
    content_sha256 = sha256(_canonical_json_bytes(source_record)).hexdigest()
    record_id = f"archive:{member_name}:{source_index}:{content_sha256}"
    return _ParsedArchiveRecord(
        member=member_name,
        index=source_index,
        content_sha256=content_sha256,
        record_id=record_id,
        story=story,
        prompt=prompt,
        words=words,
        features=features,
        source=source,
    )


def _init_archive_worker(tokenizer_path: str) -> None:
    global _WORKER_TOKENIZER
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _WORKER_TOKENIZER = TokenizersTextTokenizer.from_file(tokenizer_path)


def _process_archive_batch(
    batch: tuple[_ArchiveWorkRecord, ...],
) -> tuple[dict[str, object], ...]:
    if _WORKER_TOKENIZER is None:
        raise RuntimeError("archive worker tokenizer was not initialized")
    processed: list[dict[str, object]] = []
    for record in batch:
        story_bytes = record.story.encode("utf-8")
        empty = not record.story.strip()
        token_ids = (
            ()
            if empty
            else _WORKER_TOKENIZER.encode(record.story, add_eos=True)
        )
        token_count = len(token_ids)
        if not empty and token_count == 0:
            raise ArchiveIngestError(
                f"archive record {record.record_id} tokenized empty"
            )
        token_payload = np.asarray(token_ids, dtype="<u2").tobytes(order="C")
        try:
            recipe = (
                None
                if empty
                else recover_released_recipe(
                    record.prompt,
                    record.words,
                    record.features,
                )
            )
        except ValueError:
            recipe = None
        occurrence = ArchiveOccurrence(
            record_id=record.record_id,
            source_member=record.member,
            source_index=record.index,
            content_sha256=record.content_sha256,
            story_sha256=sha256(story_bytes).hexdigest(),
            source=record.source,
            spool_offset=record.spool_offset,
            token_spool_offset=0,
            byte_length=record.byte_length,
            token_count=token_count,
        )
        processed.append(
            {
                "empty": empty,
                "normalized_story_sha256": normalized_story_sha256(record.story),
                "occurrence": occurrence.as_record(),
                "recipe": None if recipe is None else recipe.as_record(),
                "record_id": record.record_id,
                "token_payload": token_payload,
            }
        )
    return tuple(processed)


def _spool_processed_tokens(
    records: Iterable[dict[str, object]],
    destination: Path,
) -> Iterator[dict[str, object]]:
    """Persist worker-produced token IDs once and replace them with coordinates."""
    with destination.open("wb") as output:
        for record in records:
            token_payload = record.get("token_payload")
            if type(token_payload) is not bytes:
                raise ArchiveIngestError("archive worker returned invalid token bytes")
            occurrence = _object(record, "occurrence")
            if len(token_payload) != _integer(occurrence, "token_count") * 2:
                raise ArchiveIngestError("archive worker token count changed")
            token_offset = output.tell() // 2
            output.write(token_payload)
            yield {
                key: value
                for key, value in record.items()
                if key != "token_payload"
            } | {
                "occurrence": {
                    **occurrence,
                    "token_spool_offset": token_offset,
                }
            }
        output.flush()
        os.fsync(output.fileno())


def _write_groups_and_audit(
    runs: Sequence[Path],
    destination: Path,
    source_counts: _ArchiveStreamCounts,
) -> ArchiveIngestAudit:
    counters = {
        "archive_member_count": source_counts.member_count,
        "archive_record_count": 0,
        "archive_group_count": 0,
        "nonempty_record_count": 0,
        "nonempty_token_count": 0,
        "classified_record_count": 0,
        "classified_token_count": 0,
        "empty_group_count": 0,
        "empty_record_count": 0,
        "unclassifiable_group_count": 0,
        "unclassifiable_record_count": 0,
        "unclassifiable_token_count": 0,
        "conflicting_group_count": 0,
        "conflicting_record_count": 0,
        "conflicting_token_count": 0,
        "eligible_group_count": 0,
        "eligible_record_count": 0,
        "eligible_token_count": 0,
        "duplicate_group_count": 0,
        "maximum_group_multiplicity": 0,
    }
    records = _merge_sorted_runs(
        runs,
        ("normalized_story_sha256", "record_id"),
    )
    with destination.open("wb") as output:
        for normalized_hash, grouped_iterator in groupby(
            records,
            key=lambda record: record["normalized_story_sha256"],
        ):
            if type(normalized_hash) is not str:
                raise ArchiveIngestError("normalized archive group key is not text")
            grouped = tuple(grouped_iterator)
            occurrences = tuple(_object(record, "occurrence") for record in grouped)
            token_count = sum(_integer(item, "token_count") for item in occurrences)
            status, recipe = _resolve_group(grouped)
            multiplicity = len(grouped)
            counters["archive_group_count"] += 1
            counters["archive_record_count"] += multiplicity
            counters["maximum_group_multiplicity"] = max(
                counters["maximum_group_multiplicity"],
                multiplicity,
            )
            if multiplicity > 1:
                counters["duplicate_group_count"] += 1
            for record, occurrence in zip(grouped, occurrences, strict=True):
                occurrence_tokens = _integer(occurrence, "token_count")
                if not _boolean(record, "empty"):
                    counters["nonempty_record_count"] += 1
                    counters["nonempty_token_count"] += occurrence_tokens
                    if record.get("recipe") is not None:
                        counters["classified_record_count"] += 1
                        counters["classified_token_count"] += occurrence_tokens
            prefix = {
                "empty_story": "empty",
                "unclassifiable_metadata": "unclassifiable",
                "conflicting_metadata": "conflicting",
                "eligible": "eligible",
            }[status]
            counters[f"{prefix}_group_count"] += 1
            counters[f"{prefix}_record_count"] += multiplicity
            if prefix != "empty":
                counters[f"{prefix}_token_count"] += token_count
            output.write(
                canonical_record_bytes(
                    {
                        "active_token_count": token_count,
                        "canonical_token_count": _integer(
                            occurrences[0],
                            "token_count",
                        ),
                        "normalized_story_sha256": normalized_hash,
                        "occurrences": list(occurrences),
                        "provenance": [
                            {
                                "content_sha256": occurrence["content_sha256"],
                                "record_id": occurrence["record_id"],
                                "source": occurrence["source"],
                                "source_index": occurrence["source_index"],
                                "source_member": occurrence["source_member"],
                                "story_sha256": occurrence["story_sha256"],
                            }
                            for occurrence in occurrences
                        ],
                        "recipe": recipe,
                        "status": status,
                    }
                )
            )
        output.flush()
        os.fsync(output.fileno())
    if counters["archive_record_count"] != source_counts.record_count:
        raise ArchiveIngestError("external sort lost archive records")
    try:
        return ArchiveIngestAudit(**counters)
    except (TypeError, ValueError) as error:
        raise ArchiveIngestError(f"invalid archive ingest totals: {error}") from error


def _resolve_group(
    records: Sequence[dict[str, object]],
) -> tuple[str, dict[str, object] | None]:
    if all(_boolean(record, "empty") for record in records):
        return "empty_story", None
    if any(_boolean(record, "empty") for record in records):
        raise ArchiveIngestError(
            "one normalized duplicate group mixes empty and nonempty stories"
        )
    recipes = tuple(record.get("recipe") for record in records)
    if any(recipe is None for recipe in recipes):
        return "unclassifiable_metadata", None
    canonical = {canonical_record_bytes(recipe) for recipe in recipes}
    if len(canonical) != 1:
        return "conflicting_metadata", None
    recipe = recipes[0]
    if type(recipe) is not dict:
        raise ArchiveIngestError("classified archive recipe is not an object")
    return "eligible", recipe


def _write_ingest_audit(
    path: Path,
    audit: ArchiveIngestAudit,
    preset: PartitionPreset,
) -> None:
    coverage = audit.role_classification_coverage
    payload = canonical_record_bytes(
        {
            "coverage": audit.as_record(),
            "format": "tinyworlds-p-archive-ingest-audit",
            "gates": [
                {
                    "minimum": preset.minimum_role_coverage,
                    "name": "role_classification_coverage",
                    "passed": coverage >= preset.minimum_role_coverage,
                    "value": coverage,
                }
            ],
            "passed": coverage >= preset.minimum_role_coverage,
            "schema_version": 1,
        }
    )
    with path.open("wb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())


def _require_role_coverage(
    audit: ArchiveIngestAudit,
    preset: PartitionPreset,
) -> None:
    if audit.role_classification_coverage < preset.minimum_role_coverage:
        raise ArchiveIngestError(
            "archive role-classification gate failed: "
            f"role_classification_coverage={audit.role_classification_coverage:.6f} "
            f"< {preset.minimum_role_coverage:.6f}"
        )


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
            paths.append(
                _flush_run(pending, directory, prefix, len(paths), key_fields)
            )
            pending = []
    if pending:
        paths.append(_flush_run(pending, directory, prefix, len(paths), key_fields))
    if not paths:
        raise ArchiveIngestError(f"{prefix} source produced no sortable records")
    return tuple(paths)


def _flush_run(
    records: list[dict[str, object]],
    directory: Path,
    prefix: str,
    index: int,
    key_fields: tuple[str, ...],
) -> Path:
    path = directory / f"{prefix}-{index:06d}.jsonl"
    ordered = sorted(
        records,
        key=lambda record: tuple(record[field] for field in key_fields),
    )
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
        iterators = tuple(
            _iter_run(stream, path)
            for stream, path in zip(streams, paths, strict=True)
        )
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
            raise ArchiveIngestError(
                f"invalid sort run {path}:{line_number}"
            ) from error
        if type(value) is not dict:
            raise ArchiveIngestError(
                f"sort run record is not an object: {path}:{line_number}"
            )
        yield value


def _bounded_process_map(
    batches: Iterable[tuple[_ArchiveWorkRecord, ...]],
    function,
    worker_count: int,
    *,
    initializer=None,
    initargs: tuple[object, ...] = (),
) -> Iterator[tuple[dict[str, object], ...]]:
    batch_iterator = iter(batches)
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=context,
        initializer=initializer,
        initargs=initargs,
    ) as executor:
        pending: dict[Future[tuple[dict[str, object], ...]], int] = {}
        completed: dict[int, tuple[dict[str, object], ...]] = {}
        indexed_batches = enumerate(batch_iterator)
        next_output = 0
        for _ in range(worker_count * 2):
            item = next(indexed_batches, None)
            if item is None:
                break
            index, batch = item
            pending[executor.submit(function, batch)] = index
        while pending:
            finished, _ = wait(set(pending), return_when=FIRST_COMPLETED)
            for future in finished:
                index = pending.pop(future)
                completed[index] = future.result()
            while next_output in completed:
                yield completed.pop(next_output)
                next_output += 1
                item = next(indexed_batches, None)
                if item is not None:
                    following_index, batch = item
                    pending[executor.submit(function, batch)] = following_index


def _file_identity(path: Path) -> tuple[int, str]:
    digest = sha256()
    size = 0
    with path.open("rb") as source:
        while chunk := source.read(_READ_BYTES):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _require_exact_object(
    value: object,
    fields: tuple[str, ...],
    label: str,
) -> dict[str, object]:
    if type(value) is not dict:
        raise ArchiveIngestError(f"{label} must be a JSON object")
    record: dict[str, object] = value
    actual = set(record)
    expected = set(fields)
    if actual != expected:
        raise ArchiveIngestError(
            f"{label} fields differ; unknown={tuple(sorted(actual - expected))}, "
            f"missing={tuple(sorted(expected - actual))}"
        )
    return record


def _require_string(value: object, label: str) -> str:
    if type(value) is not str:
        raise ArchiveIngestError(f"{label} must be a string")
    return value


def _require_string_array(value: object, label: str) -> tuple[str, ...]:
    if type(value) is not list or any(type(item) is not str for item in value):
        raise ArchiveIngestError(f"{label} must contain only strings")
    return tuple(value)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _validate_member_name(member_name: str) -> None:
    path = Path(member_name)
    if (
        not member_name
        or path.is_absolute()
        or ".." in path.parts
        or "\x00" in member_name
    ):
        raise ArchiveIngestError(f"unsafe archive member name: {member_name!r}")


def _object(record: dict[str, object], field: str) -> dict[str, object]:
    value = record.get(field)
    if type(value) is not dict:
        raise ArchiveIngestError(f"archive field {field!r} must be an object")
    return value


def _integer(record: dict[str, object], field: str) -> int:
    value = record.get(field)
    if type(value) is not int or value < 0:
        raise ArchiveIngestError(
            f"archive field {field!r} must be a nonnegative integer"
        )
    return value


def _text(record: dict[str, object], field: str) -> str:
    value = record.get(field)
    if type(value) is not str:
        raise ArchiveIngestError(f"archive field {field!r} must be text")
    return value


def _boolean(record: dict[str, object], field: str) -> bool:
    value = record.get(field)
    if type(value) is not bool:
        raise ArchiveIngestError(f"archive field {field!r} must be boolean")
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
    "ArchiveIngestError",
    "ArchiveIngestResult",
    "build_archive_ingest",
    "iter_archive_groups",
    "read_spooled_story",
    "read_spooled_tokens",
    "verify_partition_inputs",
]

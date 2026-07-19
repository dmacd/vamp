"""Pinned TinyStories source parsing and deterministic Phase 1 sampling."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
import codecs
from dataclasses import dataclass
from hashlib import sha256
import heapq
import json
from pathlib import Path
import tarfile
from typing import BinaryIO
import unicodedata

from apm.data.text.curricula import (
    PinnedDatasetFile,
    TINYSTORIES_DATASET_REVISION,
    TINYSTORIES_DOCUMENT_SEPARATOR,
    TINYSTORIES_V2_SOURCE,
    verify_pinned_dataset_file,
)
from apm.data.text.tinyworlds_v2.ingredients import (
    mechanically_classify_ingredient_roles,
)
from apm.data.text.tinyworlds_v2.surface import normalized_story_sha256


TINYSTORIES_ALL_DATA_FILENAME = "TinyStories_all_data.tar.gz"
TINYSTORIES_ALL_DATA_SIZE_BYTES = 1_608_001_638
TINYSTORIES_ALL_DATA_SHA256 = (
    "26cf7605aca15bc4ea6fa637256400d9d01317b28ed296172b2d1dd160cd7699"
)
TINYSTORIES_GPT4_SOURCE = "GPT-4"
_JSON_READ_CHARACTERS = 64 * 1024
_MAX_SOURCE_RECORD_CHARACTERS = 16 * 1024 * 1024
_TEXT_READ_CHARACTERS = 1024 * 1024


class TinyStoriesSourceError(ValueError):
    """A pinned TinyStories source or one of its records is malformed."""


@dataclass(frozen=True, slots=True)
class TinyStoriesArchiveSource:
    """Complete immutable identity of the released TinyStories JSON archive."""

    dataset_id: str
    revision: str
    archive_file: PinnedDatasetFile

    def __post_init__(self) -> None:
        if type(self.dataset_id) is not str or not self.dataset_id:
            raise ValueError("archive dataset_id must be nonempty")
        if (
            type(self.revision) is not str
            or len(self.revision) != 40
            or any(character not in "0123456789abcdef" for character in self.revision)
        ):
            raise ValueError("archive revision must be a lowercase Git SHA")
        if type(self.archive_file) is not PinnedDatasetFile:
            raise TypeError("archive_file must be a PinnedDatasetFile")

    @property
    def download_url(self) -> str:
        """Return the exact revision-pinned Hugging Face source URL."""
        return (
            f"https://huggingface.co/datasets/{self.dataset_id}/resolve/"
            f"{self.revision}/{self.archive_file.filename}"
        )


TINYSTORIES_ALL_DATA_SOURCE = TinyStoriesArchiveSource(
    dataset_id="roneneldan/TinyStories",
    revision=TINYSTORIES_DATASET_REVISION,
    archive_file=PinnedDatasetFile(
        filename=TINYSTORIES_ALL_DATA_FILENAME,
        size_bytes=TINYSTORIES_ALL_DATA_SIZE_BYTES,
        sha256=TINYSTORIES_ALL_DATA_SHA256,
    ),
)


@dataclass(frozen=True, slots=True)
class TinyStoriesInstruction:
    """The exact released prompt metadata attached to one archive story."""

    prompt: str
    words: tuple[str, ...]
    features: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_text(self.prompt, "instruction prompt:")
        for label, values in (("words", self.words), ("features", self.features)):
            if type(values) is not tuple or any(
                type(value) is not str for value in values
            ):
                raise TinyStoriesSourceError(
                    f"instruction {label} must contain only strings"
                )


@dataclass(frozen=True, slots=True)
class ArchiveSourceRecord:
    """One strict source record with a location- and content-bound identity."""

    record_id: str
    source_member: str
    source_index: int
    content_sha256: str
    story: str
    instruction: TinyStoriesInstruction
    summary: str
    source: str

    def __post_init__(self) -> None:
        _require_nonempty_text(self.source_member, "source member")
        if type(self.source_index) is not int or self.source_index < 0:
            raise TinyStoriesSourceError("source index must be nonnegative")
        _require_sha256(self.content_sha256, "record content SHA-256")
        _require_text(self.story, "story")
        if type(self.instruction) is not TinyStoriesInstruction:
            raise TypeError("instruction must be TinyStoriesInstruction")
        _require_text(self.summary, "summary")
        if self.source not in ("GPT-4", "GPT-3.5"):
            raise TinyStoriesSourceError(
                f"unsupported released source label: {self.source!r}"
            )
        expected_id = _archive_record_id(
            self.source_member,
            self.source_index,
            self.content_sha256,
        )
        if self.record_id != expected_id:
            raise TinyStoriesSourceError("record_id does not bind source and content")

    @property
    def normalized_story_sha256(self) -> str:
        """Return the separately derived normalized story identity."""
        return normalized_story_sha256(self.story)


@dataclass(frozen=True, slots=True)
class ArchiveSourceSelections:
    """Three hash-ranked, mutually disjoint GPT-4 archive cohorts."""

    prompt_metadata_records: tuple[ArchiveSourceRecord, ...]
    reference_story_records: tuple[ArchiveSourceRecord, ...]
    paired_records: tuple[ArchiveSourceRecord, ...]

    def __post_init__(self) -> None:
        cohorts = (
            self.prompt_metadata_records,
            self.reference_story_records,
            self.paired_records,
        )
        if any(type(cohort) is not tuple for cohort in cohorts):
            raise TypeError("archive source cohorts must be tuples")
        if any(
            type(record) is not ArchiveSourceRecord
            or record.source != TINYSTORIES_GPT4_SOURCE
            for cohort in cohorts
            for record in cohort
        ):
            raise TinyStoriesSourceError("source cohorts must contain only GPT-4 records")
        id_sets = tuple({record.record_id for record in cohort} for cohort in cohorts)
        if any(len(ids) != len(cohort) for ids, cohort in zip(id_sets, cohorts)):
            raise TinyStoriesSourceError("source records must be unique within cohorts")
        if any(
            id_sets[left] & id_sets[right]
            for left, right in ((0, 1), (0, 2), (1, 2))
        ):
            raise TinyStoriesSourceError("archive source cohorts must be disjoint")
        story_sets = tuple(
            {record.normalized_story_sha256 for record in cohort}
            for cohort in cohorts
        )
        if any(
            len(story_hashes) != len(cohort)
            for story_hashes, cohort in zip(story_sets, cohorts, strict=True)
        ):
            raise TinyStoriesSourceError(
                "archive source cohorts must be content-unique within cohorts"
            )
        if any(
            story_sets[left] & story_sets[right]
            for left, right in ((0, 1), (0, 2), (1, 2))
        ):
            raise TinyStoriesSourceError(
                "archive source cohorts must be content-disjoint"
            )
        prompt_records = self.prompt_metadata_records + self.paired_records
        story_records = self.reference_story_records + self.paired_records
        if any(
            not record.instruction.prompt.strip()
            or not record.instruction.words
            or any(not word.strip() for word in record.instruction.words)
            for record in prompt_records
        ):
            raise TinyStoriesSourceError("prompt cohorts contain unusable metadata")
        if any(not record.story.strip() for record in story_records):
            raise TinyStoriesSourceError("story cohorts contain an empty story")


@dataclass(frozen=True, slots=True)
class ValidationStoryRecord:
    """One unique story selected from the pinned GPT-4 validation aggregate."""

    record_id: str
    source_index: int
    content_sha256: str
    story: str

    def __post_init__(self) -> None:
        if type(self.source_index) is not int or self.source_index < 0:
            raise TinyStoriesSourceError("validation source index must be nonnegative")
        _require_sha256(self.content_sha256, "validation story SHA-256")
        _require_nonempty_text(self.story, "validation story")
        if self.content_sha256 != sha256(self.story.encode("utf-8")).hexdigest():
            raise TinyStoriesSourceError("validation content SHA-256 mismatch")
        expected = f"v2-validation:{self.source_index}:{self.content_sha256}"
        if self.record_id != expected:
            raise TinyStoriesSourceError("validation record_id does not bind its source")

    @property
    def normalized_story_sha256(self) -> str:
        """Return the separately derived normalized story identity."""
        return normalized_story_sha256(self.story)


class _Utf8ChunkReader:
    """Incrementally decode a non-seekable binary tar member as UTF-8."""

    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
        self._finished = False

    def read(self, size: int) -> str:
        """Read and decode at most one binary chunk from the member."""
        if self._finished:
            return ""
        payload = self._stream.read(size)
        if payload:
            return self._decoder.decode(payload, final=False)
        self._finished = True
        return self._decoder.decode(b"", final=True)


class _StreamingJsonArray:
    """Small stateful decoder that never retains more than one JSON record."""

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
        """Yield the top-level array values while enforcing exact JSON syntax."""
        if self._next_character() != "[":
            raise TinyStoriesSourceError(f"{self._label} must contain a JSON array")
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
                raise TinyStoriesSourceError(
                    f"expected array delimiter in {self._label}, got {detail}"
                )
            self._position += 1
            if self._next_character() == "]":
                raise TinyStoriesSourceError(
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
                    raise TinyStoriesSourceError(
                        f"invalid JSON record in {self._label}: {error.msg}"
                    ) from error
                if len(self._buffer) - start > _MAX_SOURCE_RECORD_CHARACTERS:
                    raise TinyStoriesSourceError(
                        f"JSON record exceeds safety limit in {self._label}"
                    )
                self._read_more()
                continue
            if end - start > _MAX_SOURCE_RECORD_CHARACTERS:
                raise TinyStoriesSourceError(
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
            raise TinyStoriesSourceError(f"trailing JSON data in {self._label}")

    def _strict_object(self, pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise TinyStoriesSourceError(
                    f"duplicate field {key!r} in {self._label}"
                )
            result[key] = value
        return result

    def _reject_constant(self, value: str) -> float:
        raise TinyStoriesSourceError(
            f"non-finite number {value!r} in {self._label}"
        )


class _BoundedArchiveSample:
    """Retain only the lowest namespaced hash ranks from a record stream."""

    def __init__(self, capacity: int, seed: str, namespace: str) -> None:
        self._capacity = capacity
        self._seed = seed
        self._namespace = namespace
        self._heap: list[tuple[int, str, str]] = []
        self._by_story: dict[str, tuple[int, str, ArchiveSourceRecord]] = {}

    def offer(self, record: ArchiveSourceRecord) -> None:
        if self._capacity == 0:
            return
        story_sha256 = record.normalized_story_sha256
        # Rank the unique story identity, not the released record location.
        # Otherwise duplicated story text receives multiple lottery tickets
        # and biases every empirical cohort.  Record ID only breaks ties when
        # choosing which released occurrence supplies provenance/metadata.
        rank = _archive_rank(self._seed, self._namespace, story_sha256)
        existing = self._by_story.get(story_sha256)
        if existing is not None and (rank, record.record_id) >= existing[:2]:
            return
        self._by_story[story_sha256] = (rank, record.record_id, record)
        heapq.heappush(self._heap, (-rank, record.record_id, story_sha256))
        self._trim()

    def _trim(self) -> None:
        while len(self._by_story) > self._capacity:
            negative_rank, record_id, story_sha256 = heapq.heappop(self._heap)
            current = self._by_story.get(story_sha256)
            if current is None:
                continue
            if (current[0], current[1]) != (-negative_rank, record_id):
                continue
            del self._by_story[story_sha256]

    def ordered(self) -> tuple[ArchiveSourceRecord, ...]:
        return tuple(
            sorted(
                (entry[2] for entry in self._by_story.values()),
                key=lambda record: (
                    _archive_rank(
                        self._seed,
                        self._namespace,
                        record.normalized_story_sha256,
                    ),
                    record.record_id,
                ),
            )
        )


def verify_tinystories_archive(
    path: str | Path,
    source: TinyStoriesArchiveSource = TINYSTORIES_ALL_DATA_SOURCE,
) -> Path:
    """Verify the archive filename, byte count, and SHA-256 by streaming once."""
    if type(source) is not TinyStoriesArchiveSource:
        raise TypeError("source must be TinyStoriesArchiveSource")
    return verify_pinned_dataset_file(path, source.archive_file)


def iter_archive_source_records(
    path: str | Path,
    source: TinyStoriesArchiveSource = TINYSTORIES_ALL_DATA_SOURCE,
) -> Iterator[ArchiveSourceRecord]:
    """Verify and stream strict records from every JSON-array archive member."""
    verified_path = verify_tinystories_archive(path, source)
    json_members: set[str] = set()
    try:
        with tarfile.open(verified_path, mode="r|gz") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                _validate_member_name(member.name)
                if not member.name.endswith(".json"):
                    raise TinyStoriesSourceError(
                        f"unexpected non-JSON archive member: {member.name!r}"
                    )
                if member.name in json_members:
                    raise TinyStoriesSourceError(
                        f"duplicate JSON archive member: {member.name!r}"
                    )
                json_members.add(member.name)
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise TinyStoriesSourceError(
                        f"could not stream archive member: {member.name!r}"
                    )
                yield from _records_from_member(extracted, member.name)
    except (tarfile.TarError, UnicodeError, OSError) as error:
        raise TinyStoriesSourceError(
            f"could not read pinned archive {verified_path}: {error}"
        ) from error
    if not json_members:
        raise TinyStoriesSourceError("TinyStories archive contains no JSON members")


def select_gpt4_archive_records(
    records: Iterable[ArchiveSourceRecord],
    *,
    seed: str,
    prompt_metadata_count: int = 10_000,
    reference_story_count: int = 10_000,
    paired_count: int = 200,
) -> ArchiveSourceSelections:
    """Select the prescribed disjoint cohorts with bounded hash-ranked heaps."""
    _require_nonempty_text(seed, "selection seed")
    counts = (paired_count, prompt_metadata_count, reference_story_count)
    if any(type(count) is not int or count < 0 for count in counts):
        raise ValueError("source sample counts must be nonnegative integers")
    capacities = (
        paired_count,
        paired_count + prompt_metadata_count,
        paired_count + prompt_metadata_count + reference_story_count,
    )
    paired_sample, prompt_sample, reference_sample = (
        _BoundedArchiveSample(capacity, seed, namespace)
        for capacity, namespace in zip(
            capacities,
            ("paired-reference", "prompt-metadata", "reference-story"),
        )
    )
    paired_eligible_count = 0
    prompt_eligible_count = 0
    reference_eligible_count = 0
    for record in records:
        if type(record) is not ArchiveSourceRecord:
            raise TypeError("records must contain ArchiveSourceRecord values")
        if record.source != TINYSTORIES_GPT4_SOURCE:
            continue
        prompt_eligible = bool(record.story.strip()) and bool(
            record.instruction.prompt.strip()
        ) and bool(
            record.instruction.words
        ) and all(word.strip() for word in record.instruction.words)
        reference_eligible = bool(record.story.strip())
        paired_words = tuple(
            word.strip().casefold() for word in record.instruction.words
        )
        paired_eligible = (
            prompt_eligible
            and reference_eligible
            and len(paired_words) == 3
            and len(set(paired_words)) == 3
            and mechanically_classify_ingredient_roles(
                record.instruction.prompt,
                record.instruction.words,
            )
            is not None
        )
        if paired_eligible:
            paired_sample.offer(record)
            paired_eligible_count += 1
        if prompt_eligible:
            prompt_sample.offer(record)
            prompt_eligible_count += 1
        if reference_eligible:
            reference_sample.offer(record)
            reference_eligible_count += 1

    paired = paired_sample.ordered()[:paired_count]
    paired_story_hashes = frozenset(
        record.normalized_story_sha256 for record in paired
    )
    prompt = tuple(
        record
        for record in prompt_sample.ordered()
        if record.normalized_story_sha256 not in paired_story_hashes
    )[:prompt_metadata_count]
    excluded_story_hashes = paired_story_hashes | frozenset(
        record.normalized_story_sha256 for record in prompt
    )
    reference = tuple(
        record
        for record in reference_sample.ordered()
        if record.normalized_story_sha256 not in excluded_story_hashes
    )[:reference_story_count]
    actual = (len(paired), len(prompt), len(reference))
    if actual != counts:
        raise TinyStoriesSourceError(
            "not enough eligible GPT-4 archive records for disjoint samples: "
            f"paired={paired_eligible_count}, prompts={prompt_eligible_count}, "
            f"stories={reference_eligible_count}, requested={counts}"
        )
    return ArchiveSourceSelections(
        prompt_metadata_records=prompt,
        reference_story_records=reference,
        paired_records=paired,
    )


def select_archive_source_records(
    path: str | Path,
    *,
    seed: str,
    prompt_metadata_count: int = 10_000,
    reference_story_count: int = 10_000,
    paired_count: int = 200,
    source: TinyStoriesArchiveSource = TINYSTORIES_ALL_DATA_SOURCE,
) -> ArchiveSourceSelections:
    """Verify, stream, and select the complete Phase 1 archive cohorts."""
    return select_gpt4_archive_records(
        iter_archive_source_records(path, source),
        seed=seed,
        prompt_metadata_count=prompt_metadata_count,
        reference_story_count=reference_story_count,
        paired_count=paired_count,
    )


def select_validation_story_records(
    path: str | Path,
    *,
    seed: str,
    count: int = 10_000,
    expected_file: PinnedDatasetFile = TINYSTORIES_V2_SOURCE.validation_file,
    exclude_normalized_story_sha256: frozenset[str] = frozenset(),
) -> tuple[ValidationStoryRecord, ...]:
    """Verify and hash-select unique stories from the GPT-4 validation text."""
    _require_nonempty_text(seed, "selection seed")
    if type(count) is not int or count < 0:
        raise ValueError("validation sample count must be nonnegative")
    if type(exclude_normalized_story_sha256) is not frozenset or any(
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in exclude_normalized_story_sha256
    ):
        raise ValueError("excluded story hashes must be a frozenset of SHA-256 values")
    verified_path = verify_pinned_dataset_file(path, expected_file)
    selected: dict[str, ValidationStoryRecord] = {}
    largest: list[tuple[int, str]] = []
    seen_story_sha256: set[str] = set()
    unique_count = 0
    for source_index, raw_story in enumerate(_iter_delimited_text(verified_path)):
        story = unicodedata.normalize("NFC", raw_story).strip()
        if not story:
            continue
        content_sha256 = sha256(story.encode("utf-8")).hexdigest()
        story_sha256 = normalized_story_sha256(story)
        if (
            story_sha256 in seen_story_sha256
            or story_sha256 in exclude_normalized_story_sha256
        ):
            continue
        seen_story_sha256.add(story_sha256)
        rank = _validation_rank(seed, story_sha256)
        record = ValidationStoryRecord(
            record_id=f"v2-validation:{source_index}:{content_sha256}",
            source_index=source_index,
            content_sha256=content_sha256,
            story=story,
        )
        unique_count += 1
        if len(selected) < count:
            selected[story_sha256] = record
            heapq.heappush(largest, (-rank, story_sha256))
        elif count and rank < -largest[0][0]:
            _, removed_story_sha256 = heapq.heapreplace(
                largest,
                (-rank, story_sha256),
            )
            del selected[removed_story_sha256]
            selected[story_sha256] = record
    if len(selected) != count:
        raise TinyStoriesSourceError(
            f"requested {count} validation stories from only {unique_count} unique documents"
        )
    return tuple(
        sorted(
            selected.values(),
            key=lambda record: (
                _validation_rank(seed, record.normalized_story_sha256),
                record.normalized_story_sha256,
            ),
        )
    )


def canonical_archive_record(record: ArchiveSourceRecord) -> dict[str, object]:
    """Return the released source payload with stable provenance fields."""
    if type(record) is not ArchiveSourceRecord:
        raise TypeError("record must be ArchiveSourceRecord")
    return {
        "content_sha256": record.content_sha256,
        "normalized_story_sha256": record.normalized_story_sha256,
        "instruction": {
            "features": list(record.instruction.features),
            "prompt:": record.instruction.prompt,
            "words": list(record.instruction.words),
        },
        "record_id": record.record_id,
        "source": record.source,
        "source_index": record.source_index,
        "source_member": record.source_member,
        "story": record.story,
        "summary": record.summary,
    }


def canonical_prompt_metadata_record(
    record: ArchiveSourceRecord,
) -> dict[str, object]:
    """Return prompt evidence sufficient to reauthenticate its source record."""
    if type(record) is not ArchiveSourceRecord:
        raise TypeError("record must be ArchiveSourceRecord")
    return {
        "content_sha256": record.content_sha256,
        "normalized_story_sha256": record.normalized_story_sha256,
        "features": list(record.instruction.features),
        "prompt": record.instruction.prompt,
        "record_id": record.record_id,
        "source": record.source,
        "source_index": record.source_index,
        "source_member": record.source_member,
        "story": record.story,
        "summary": record.summary,
        "words": list(record.instruction.words),
    }


def canonical_reference_story_record(
    record: ArchiveSourceRecord,
) -> dict[str, object]:
    """Return one genuine archive story with immutable source provenance."""
    if type(record) is not ArchiveSourceRecord:
        raise TypeError("record must be ArchiveSourceRecord")
    return {
        "content_sha256": record.content_sha256,
        "normalized_story_sha256": record.normalized_story_sha256,
        "record_id": record.record_id,
        "source": record.source,
        "source_index": record.source_index,
        "source_member": record.source_member,
        "story": record.story,
    }


def canonical_validation_record(record: ValidationStoryRecord) -> dict[str, object]:
    """Return one validation story as a canonical artifact record."""
    if type(record) is not ValidationStoryRecord:
        raise TypeError("record must be ValidationStoryRecord")
    return {
        "content_sha256": record.content_sha256,
        "normalized_story_sha256": record.normalized_story_sha256,
        "record_id": record.record_id,
        "source_index": record.source_index,
        "story": record.story,
    }


def canonical_jsonl(records: Sequence[dict[str, object]]) -> bytes:
    """Encode source records as sorted-key UTF-8 JSONL with one final newline."""
    return b"".join(
        json.dumps(
            record,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
        for record in records
    )


def _records_from_member(
    extracted: BinaryIO,
    member_name: str,
) -> Iterator[ArchiveSourceRecord]:
    text = _Utf8ChunkReader(extracted)
    for source_index, value in enumerate(
        _StreamingJsonArray(text, member_name).records()
    ):
        yield _parse_archive_record(value, member_name, source_index)


def _parse_archive_record(
    value: object,
    member_name: str,
    source_index: int,
) -> ArchiveSourceRecord:
    record = _require_exact_object(
        value,
        ("story", "instruction", "summary", "source"),
        f"{member_name}[{source_index}]",
    )
    instruction_record = _require_exact_object(
        record["instruction"],
        ("prompt:", "words", "features"),
        f"{member_name}[{source_index}].instruction",
    )
    words = _require_string_array(
        instruction_record["words"],
        f"{member_name}[{source_index}].instruction.words",
    )
    features = _require_string_array(
        instruction_record["features"],
        f"{member_name}[{source_index}].instruction.features",
    )
    prompt = _require_string(
        instruction_record["prompt:"],
        f"{member_name}[{source_index}].instruction.prompt:",
    )
    source_label = _require_string(record["source"], "source")
    story = _require_string(record["story"], "story")
    summary = _require_string(record["summary"], "summary")
    source_payload: dict[str, object] = {
        "instruction": {
            "features": list(features),
            "prompt:": prompt,
            "words": list(words),
        },
        "source": source_label,
        "story": story,
        "summary": summary,
    }
    content_sha256 = sha256(_canonical_json_bytes(source_payload)).hexdigest()
    return ArchiveSourceRecord(
        record_id=_archive_record_id(member_name, source_index, content_sha256),
        source_member=member_name,
        source_index=source_index,
        content_sha256=content_sha256,
        story=story,
        instruction=TinyStoriesInstruction(
            prompt=prompt,
            words=words,
            features=features,
        ),
        summary=summary,
        source=source_label,
    )


def _require_exact_object(
    value: object,
    fields: tuple[str, ...],
    label: str,
) -> dict[str, object]:
    if type(value) is not dict:
        raise TinyStoriesSourceError(f"{label} must be a JSON object")
    record: dict[str, object] = value
    actual = set(record)
    expected = set(fields)
    if actual != expected:
        raise TinyStoriesSourceError(
            f"{label} fields differ; unknown={tuple(sorted(actual - expected))}, "
            f"missing={tuple(sorted(expected - actual))}"
        )
    return record


def _require_string(value: object, label: str) -> str:
    if type(value) is not str:
        raise TinyStoriesSourceError(f"{label} must be a string")
    return value


def _require_string_array(value: object, label: str) -> tuple[str, ...]:
    if type(value) is not list or any(type(item) is not str for item in value):
        raise TinyStoriesSourceError(f"{label} must contain only strings")
    return tuple(value)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _archive_record_id(member: str, index: int, content_sha256: str) -> str:
    return f"archive:{member}:{index}:{content_sha256}"


def _archive_rank(seed: str, namespace: str, story_sha256: str) -> int:
    payload = f"tinyworlds-v2\0{seed}\0{namespace}\0{story_sha256}".encode("utf-8")
    return int.from_bytes(sha256(payload).digest(), "big")


def _validation_rank(seed: str, content_sha256: str) -> int:
    payload = (
        f"tinyworlds-v2\0{seed}\0v2-gpt4-validation\0{content_sha256}"
    ).encode("utf-8")
    return int.from_bytes(sha256(payload).digest(), "big")


def _iter_delimited_text(path: Path) -> Iterator[str]:
    pending = ""
    with path.open("r", encoding="utf-8", newline="") as source:
        while chunk := source.read(_TEXT_READ_CHARACTERS):
            documents = (pending + chunk).split(TINYSTORIES_DOCUMENT_SEPARATOR)
            yield from documents[:-1]
            pending = documents[-1]
    yield pending


def _validate_member_name(member_name: str) -> None:
    path = Path(member_name)
    if (
        not member_name
        or path.is_absolute()
        or ".." in path.parts
        or "\x00" in member_name
    ):
        raise TinyStoriesSourceError(f"unsafe archive member name: {member_name!r}")


def _require_nonempty_text(value: object, label: str) -> None:
    if type(value) is not str or not value.strip():
        raise TinyStoriesSourceError(f"{label} must be a nonempty string")


def _require_text(value: object, label: str) -> None:
    if type(value) is not str:
        raise TinyStoriesSourceError(f"{label} must be a string")


def _require_sha256(value: object, label: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TinyStoriesSourceError(f"{label} must be lowercase hexadecimal")


__all__ = [
    "ArchiveSourceRecord",
    "ArchiveSourceSelections",
    "TINYSTORIES_ALL_DATA_FILENAME",
    "TINYSTORIES_ALL_DATA_SHA256",
    "TINYSTORIES_ALL_DATA_SIZE_BYTES",
    "TINYSTORIES_ALL_DATA_SOURCE",
    "TINYSTORIES_GPT4_SOURCE",
    "TinyStoriesArchiveSource",
    "TinyStoriesInstruction",
    "TinyStoriesSourceError",
    "ValidationStoryRecord",
    "canonical_archive_record",
    "canonical_jsonl",
    "canonical_prompt_metadata_record",
    "canonical_reference_story_record",
    "canonical_validation_record",
    "iter_archive_source_records",
    "select_archive_source_records",
    "select_gpt4_archive_records",
    "select_validation_story_records",
    "verify_tinystories_archive",
]

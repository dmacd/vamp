"""Exact duplicate-group source records shared by review and partition builds."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from collections.abc import Callable, Iterator
from typing import BinaryIO

import numpy as np

from apm.data.text.tinyworlds_p.archive_ingest import (
    iter_archive_groups,
)
from apm.data.text.tinyworlds_p.contracts import ProgressCallback, ProgressEvent
from apm.data.text.tinyworlds_p.normalization import normalized_story_bytes_sha256
from apm.data.text.tinyworlds_q_semantic.contracts import require_sha256


@dataclass(frozen=True, slots=True)
class QueryStoryOccurrence:
    """One exact archive story and token sequence inside a duplicate group."""

    record_id: str
    source_member: str
    source_index: int
    content_sha256: str
    story_sha256: str
    source: str
    story_bytes: bytes
    token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if any(
            type(value) is not str or not value
            for value in (self.record_id, self.source_member, self.source)
        ):
            raise ValueError("query occurrence provenance must be nonempty")
        if type(self.source_index) is not int or self.source_index < 0:
            raise ValueError("query occurrence source index must be nonnegative")
        for value, label in (
            (self.content_sha256, "query occurrence content"),
            (self.story_sha256, "query occurrence story"),
        ):
            require_sha256(value, label)
        if type(self.story_bytes) is not bytes or not self.story_bytes:
            raise ValueError("query occurrence story bytes must be nonempty")
        if sha256(self.story_bytes).hexdigest() != self.story_sha256:
            raise ValueError("query occurrence story bytes changed")
        self.story_bytes.decode("utf-8", errors="strict")
        if (
            type(self.token_ids) is not tuple
            or not self.token_ids
            or any(type(token) is not int or not 0 <= token <= 65_535 for token in self.token_ids)
        ):
            raise ValueError("query occurrence token IDs must be nonempty uint16 values")


@dataclass(frozen=True, slots=True)
class QueryStoryGroup:
    """One preserved normalized duplicate group with all exact occurrences."""

    normalized_story_sha256: str
    occurrences: tuple[QueryStoryOccurrence, ...]

    def __post_init__(self) -> None:
        require_sha256(self.normalized_story_sha256, "query story group")
        if type(self.occurrences) is not tuple or not self.occurrences:
            raise ValueError("query story groups require at least one occurrence")
        if any(type(item) is not QueryStoryOccurrence for item in self.occurrences):
            raise TypeError("query story groups require QueryStoryOccurrence values")
        if len({item.record_id for item in self.occurrences}) != len(self.occurrences):
            raise ValueError("query duplicate group occurrence IDs must be unique")
        if any(
            normalized_story_bytes_sha256(item.story_bytes)
            != self.normalized_story_sha256
            for item in self.occurrences
        ):
            raise ValueError("query duplicate group contains a different normalized story")

    @property
    def canonical_occurrence(self) -> QueryStoryOccurrence:
        """Return the deterministic first archive occurrence."""
        return min(
            self.occurrences,
            key=lambda item: (item.source_member, item.source_index, item.record_id),
        )

    @property
    def normalized_text(self) -> str:
        """Return the canonical exact story decoded as UTF-8."""
        return self.canonical_occurrence.story_bytes.decode("utf-8")

    @property
    def token_count(self) -> int:
        """Return aggregate token mass across preserved occurrences."""
        return sum(len(item.token_ids) for item in self.occurrences)


def iter_query_story_groups(
    groups_path: str | Path,
    story_spool_path: str | Path,
    token_spool_path: str | Path,
    *,
    group_filter: Callable[[str], bool] | None = None,
    progress: ProgressCallback | None = None,
    total_group_count: int | None = None,
) -> Iterator[QueryStoryGroup]:
    """Adapt a fresh archive ingest stream into query-native source groups."""
    if total_group_count is not None and (
        type(total_group_count) is not int or total_group_count <= 0
    ):
        raise ValueError("query source total_group_count must be positive")
    progress_interval = max(
        1,
        min(100_000, (total_group_count or 1_000_000) // 100),
    )
    processed_group_count = 0
    _emit_progress(
        progress,
        0,
        total_group_count,
        "scanning duplicate-group identities for the construction slice",
    )
    with (
        Path(story_spool_path).open("rb") as story_spool,
        Path(token_spool_path).open("rb") as token_spool,
    ):
        for record in iter_archive_groups(groups_path):
            if record.get("status") == "empty_story":
                continue
            processed_group_count += 1
            if processed_group_count % progress_interval == 0:
                _emit_progress(
                    progress,
                    processed_group_count,
                    total_group_count,
                    "selecting construction groups before loading exact payloads",
                )
            group_sha256 = _text(record, "normalized_story_sha256")
            if group_filter is not None and not group_filter(group_sha256):
                continue
            raw_occurrences = _records(record, "occurrences")
            occurrences = tuple(
                _read_occurrence(item, story_spool, token_spool)
                for item in raw_occurrences
            )
            yield QueryStoryGroup(group_sha256, occurrences)
    if total_group_count is not None and processed_group_count != total_group_count:
        raise ValueError("query source group count differs from the archive audit")
    _emit_progress(
        progress,
        processed_group_count,
        total_group_count,
        "construction-slice source scan completed",
    )


def _emit_progress(
    progress: ProgressCallback | None,
    completed: int,
    total: int | None,
    detail: str,
) -> None:
    if progress is not None:
        progress(ProgressEvent("buckets", completed, total, detail))


def _read_occurrence(
    record: dict[str, object],
    story_spool: BinaryIO,
    token_spool: BinaryIO,
) -> QueryStoryOccurrence:
    story_offset = _integer(record, "spool_offset")
    byte_length = _integer(record, "byte_length")
    token_offset = _integer(record, "token_spool_offset")
    token_count = _integer(record, "token_count")
    story_spool.seek(story_offset)
    story = story_spool.read(byte_length)
    token_spool.seek(token_offset * 2)
    token_payload = token_spool.read(token_count * 2)
    if len(story) != byte_length or len(token_payload) != token_count * 2:
        raise ValueError("query source spool coordinates are truncated")
    return QueryStoryOccurrence(
        record_id=_text(record, "record_id"),
        source_member=_text(record, "source_member"),
        source_index=_integer(record, "source_index"),
        content_sha256=_text(record, "content_sha256"),
        story_sha256=_text(record, "story_sha256"),
        source=_text(record, "source"),
        story_bytes=story,
        token_ids=tuple(
            int(token) for token in np.frombuffer(token_payload, dtype="<u2")
        ),
    )


def _records(record: dict[str, object], field: str) -> tuple[dict[str, object], ...]:
    value = record.get(field)
    if type(value) is not list or any(type(item) is not dict for item in value):
        raise ValueError(f"archive group {field} must contain records")
    return tuple(value)  # type: ignore[arg-type]


def _text(record: dict[str, object], field: str) -> str:
    value = record.get(field)
    if type(value) is not str or not value:
        raise ValueError(f"archive group {field} must be nonempty text")
    return value


def _integer(record: dict[str, object], field: str) -> int:
    value = record.get(field)
    if type(value) is not int or value < 0:
        raise ValueError(f"archive group {field} must be nonnegative")
    return value


__all__ = [
    "QueryStoryGroup",
    "QueryStoryOccurrence",
    "iter_query_story_groups",
]

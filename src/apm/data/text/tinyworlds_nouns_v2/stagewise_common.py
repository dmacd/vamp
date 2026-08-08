"""Shared deterministic mechanics for nouns-v2 longitudinal ledgers."""

from __future__ import annotations

from collections.abc import Iterable
from hashlib import sha256
import os
from pathlib import Path

from apm.data.text.tinyworlds_nouns_v1.experiment import (
    StoryIndexEntry,
    load_story_index,
)
from apm.data.text.tinyworlds_nouns_v2.contracts import (
    STAGEWISE_CASE_COUNT,
    TASK_IDS,
    NounsV2PartitionArtifact,
)


def generation_entries(
    partition: NounsV2PartitionArtifact,
) -> dict[str, tuple[StoryIndexEntry, ...]]:
    """Load every task's canonical midpoint-evaluation index."""
    return {
        task_id: load_story_index(partition, f"task-{task_id}-generation")
        for task_id in partition.task_ids
    }


def expected_stagewise_keys(
    task_ids: tuple[str, ...],
    entries_by_task: dict[str, tuple[StoryIndexEntry, ...]],
) -> set[tuple[str, str, str]]:
    """Return the exact learned-task triangle keys in ledger form."""
    expected = {
        (str(stage), task_id, entry.story_id)
        for stage in range(1, len(task_ids) + 1)
        for task_id in task_ids[:stage]
        for entry in entries_by_task[task_id]
    }
    if task_ids == TASK_IDS and len(expected) != STAGEWISE_CASE_COUNT:
        raise ValueError("canonical stagewise case count changed")
    return expected


def repair_interrupted_tail(path: Path) -> None:
    """Truncate only a final incomplete JSONL record, preserving prior rows."""
    if not path.is_file() or path.stat().st_size == 0:
        return
    with path.open("r+b") as stream:
        stream.seek(-1, os.SEEK_END)
        if stream.read(1) == b"\n":
            return
        position = stream.tell() - 1
        while position > 0:
            chunk_start = max(0, position - 64 * 1024)
            stream.seek(chunk_start)
            chunk = stream.read(position - chunk_start)
            newline = chunk.rfind(b"\n")
            if newline >= 0:
                stream.truncate(chunk_start + newline + 1)
                stream.flush()
                os.fsync(stream.fileno())
                return
            position = chunk_start
        stream.truncate(0)
        stream.flush()
        os.fsync(stream.fileno())


def file_sha256(path: Path) -> str:
    """Hash one published ledger without loading it into memory."""
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def longitudinal_metrics(
    series: tuple[tuple[int, dict[str, object]], ...],
) -> dict[str, object]:
    """Summarize one task from introduction through the final stage."""
    introduction = series[0][1]
    final = series[-1][1]
    best_stage, best = min(
        series,
        key=lambda item: float(item[1]["story_mean_nll"]),
    )
    return {
        "backward_transfer": float(introduction["story_mean_nll"])
        - float(final["story_mean_nll"]),
        "best_stage": best_stage,
        "best_story_mean_nll": float(best["story_mean_nll"]),
        "final_story_mean_nll": float(final["story_mean_nll"]),
        "forgetting": float(final["story_mean_nll"])
        - float(best["story_mean_nll"]),
        "introduction_story_mean_nll": float(introduction["story_mean_nll"]),
    }


def mean(values: Iterable[float]) -> float:
    """Return a strict nonempty arithmetic mean."""
    measured = tuple(values)
    if not measured:
        raise ValueError("stagewise mean requires values")
    return sum(measured) / len(measured)


def object_record(value: object, label: str) -> dict[str, object]:
    """Narrow one decoded JSON value to an exact object."""
    if type(value) is not dict:
        raise TypeError(f"{label} must be an object")
    return value


__all__ = [
    "expected_stagewise_keys",
    "file_sha256",
    "generation_entries",
    "longitudinal_metrics",
    "mean",
    "object_record",
    "repair_interrupted_tail",
]

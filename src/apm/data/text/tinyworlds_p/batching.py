"""Memory-mapped deterministic block-shuffled batches from partition shards."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from apm.data.text.tinyworlds_p.contracts import PartitionArtifact, WORLD_LABELS
from apm.lm.text_data import TokenBatch


@dataclass(frozen=True, slots=True)
class TokenBatchBlock:
    """One canonical source block at its epoch-specific shuffled position."""

    source_block: int
    shuffled_block: int
    batches: tuple[TokenBatch, ...]

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value < 0
            for value in (self.source_block, self.shuffled_block)
        ):
            raise ValueError("batch block indexes must be nonnegative")
        if type(self.batches) is not tuple or not self.batches:
            raise ValueError("token batch blocks must contain batches")
        if any(type(batch) is not TokenBatch for batch in self.batches):
            raise TypeError("batch blocks must contain TokenBatch values")


@dataclass(frozen=True, slots=True)
class _BlockDescriptor:
    source_block: int
    byte_start: int
    byte_stop: int


def iter_partition_batches(
    artifact: PartitionArtifact,
    split: str,
    epoch: int,
) -> Iterator[TokenBatch]:
    """Yield fixed microbatches from a canonical split such as ``base/train``."""
    for block in iter_partition_batch_blocks(artifact, split, epoch):
        yield from block.batches


def iter_partition_batch_blocks(
    artifact: PartitionArtifact,
    split: str,
    epoch: int,
) -> Iterator[TokenBatchBlock]:
    """Yield epoch-shuffled blocks while preserving canonical order within blocks."""
    if type(artifact) is not PartitionArtifact:
        raise TypeError("artifact must be PartitionArtifact")
    if type(epoch) is not int or epoch < 0:
        raise ValueError("epoch must be a nonnegative integer")
    index_path = artifact.root / "indexes" / _index_filename(split)
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    descriptors = _index_blocks(index_path, artifact.preset.batch_block_documents)
    if not descriptors:
        raise ValueError(f"partition split {split!r} contains no documents")
    ordered = tuple(
        sorted(
            descriptors,
            key=lambda descriptor: (
                sha256(
                    (
                        "tinyworlds-p-block-shuffle-v1\0"
                        + artifact.partition_sha256
                        + "\0"
                        + split
                        + "\0"
                        + str(epoch)
                        + "\0"
                        + str(descriptor.source_block)
                    ).encode("utf-8")
                ).hexdigest(),
                descriptor.source_block,
            ),
        )
    )
    token_shards: dict[int, np.memmap] = {}
    try:
        for shuffled_block, descriptor in enumerate(ordered):
            records = _read_block_records(index_path, descriptor)
            batches = _records_to_batches(
                records,
                artifact,
                token_shards,
            )
            if batches:
                yield TokenBatchBlock(
                    source_block=descriptor.source_block,
                    shuffled_block=shuffled_block,
                    batches=batches,
                )
    finally:
        token_shards.clear()


def count_partition_microbatches(artifact: PartitionArtifact, split: str) -> int:
    """Count stable padded microbatches per epoch without mapping token shards."""
    index_path = artifact.root / "indexes" / _index_filename(split)
    descriptors = _index_blocks(index_path, artifact.preset.batch_block_documents)
    total = 0
    for descriptor in descriptors:
        window_count = sum(
            max(0, (_integer(record, "token_count") - 2) // artifact.preset.context_length + 1)
            for record in _read_block_records(index_path, descriptor)
        )
        total += (window_count + artifact.preset.batch_size - 1) // artifact.preset.batch_size
    return total


def _index_filename(split: str) -> str:
    if type(split) is not str:
        raise TypeError("partition split selector must be text")
    parts = split.split("/")
    if len(parts) == 2 and parts[0] == "base" and parts[1] in (
        "train",
        "validation",
        "test",
    ):
        return f"base-{parts[1]}.jsonl"
    if (
        len(parts) == 3
        and parts[0] in ("world", "control")
        and parts[1] in WORLD_LABELS
        and parts[2] in ("train", "validation", "test")
        and not (parts[0] == "control" and parts[2] == "train")
    ):
        return f"{parts[0]}-{parts[1]}-{parts[2]}.jsonl"
    raise ValueError(
        "split must be base/{train,validation,test}, "
        "world/{A-E}/{train,validation,test}, or control/{A-E}/{validation,test}"
    )


def _index_blocks(path: Path, documents_per_block: int) -> tuple[_BlockDescriptor, ...]:
    descriptors: list[_BlockDescriptor] = []
    with path.open("rb") as source:
        source_block = 0
        block_start = 0
        document_count = 0
        while line := source.readline():
            if not line.endswith(b"\n"):
                raise ValueError(f"partition index has an unterminated line: {path}")
            document_count += 1
            if document_count == documents_per_block:
                descriptors.append(
                    _BlockDescriptor(source_block, block_start, source.tell())
                )
                source_block += 1
                block_start = source.tell()
                document_count = 0
        if document_count:
            descriptors.append(_BlockDescriptor(source_block, block_start, source.tell()))
    return tuple(descriptors)


def _read_block_records(
    path: Path,
    descriptor: _BlockDescriptor,
) -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    with path.open("rb") as source:
        source.seek(descriptor.byte_start)
        while source.tell() < descriptor.byte_stop:
            line = source.readline()
            try:
                value = json.loads(line)
            except (json.JSONDecodeError, UnicodeError) as error:
                raise ValueError(f"invalid partition document index: {path}") from error
            if type(value) is not dict:
                raise ValueError("partition document index record must be an object")
            records.append(value)
        if source.tell() != descriptor.byte_stop:
            raise ValueError("partition block descriptor does not end between documents")
    return tuple(records)


def _records_to_batches(
    records: tuple[dict[str, object], ...],
    artifact: PartitionArtifact,
    token_shards: dict[int, np.memmap],
) -> tuple[TokenBatch, ...]:
    context_length = artifact.preset.context_length
    batch_size = artifact.preset.batch_size
    rows: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    batches: list[TokenBatch] = []
    for record in records:
        shard_id = _integer(record, "token_shard")
        token_count = _integer(record, "token_count")
        token_offset = _integer(record, "token_offset")
        shard = token_shards.get(shard_id)
        if shard is None:
            shard_path = artifact.root / "shards" / f"tokens-{shard_id:06d}.uint16"
            shard = np.memmap(shard_path, dtype="<u2", mode="r")
            token_shards[shard_id] = shard
        stop = token_offset + token_count
        if stop > shard.shape[0]:
            raise ValueError("partition token index points outside its shard")
        tokens = np.asarray(shard[token_offset:stop], dtype=np.int32)
        for start in range(0, max(token_count - 1, 0), context_length):
            chunk = tokens[start : start + context_length + 1]
            transition_count = len(chunk) - 1
            if transition_count <= 0:
                continue
            input_ids = np.full(context_length, artifact.pad_token_id, dtype=np.int32)
            target_ids = np.full(context_length, artifact.pad_token_id, dtype=np.int32)
            attention_mask = np.zeros(context_length, dtype=np.bool_)
            loss_mask = np.zeros(context_length, dtype=np.bool_)
            input_ids[:transition_count] = chunk[:-1]
            target_ids[:transition_count] = chunk[1:]
            attention_mask[:transition_count] = True
            loss_mask[:transition_count] = True
            rows.append((input_ids, attention_mask, target_ids, loss_mask))
            if len(rows) == batch_size:
                batches.append(_stack_rows(rows))
                rows = []
    if rows:
        empty_count = batch_size - len(rows)
        rows.extend(
            (
                np.full(context_length, artifact.pad_token_id, dtype=np.int32),
                np.zeros(context_length, dtype=np.bool_),
                np.full(context_length, artifact.pad_token_id, dtype=np.int32),
                np.zeros(context_length, dtype=np.bool_),
            )
            for _ in range(empty_count)
        )
        batches.append(_stack_rows(rows))
    return tuple(batches)


def _stack_rows(
    rows: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> TokenBatch:
    return TokenBatch(
        input_ids=np.stack(tuple(row[0] for row in rows)),
        attention_mask=np.stack(tuple(row[1] for row in rows)),
        target_ids=np.stack(tuple(row[2] for row in rows)),
        loss_mask=np.stack(tuple(row[3] for row in rows)),
    )


def _integer(record: dict[str, object], field: str) -> int:
    value = record.get(field)
    if type(value) is not int or value < 0:
        raise ValueError(f"partition index {field!r} must be a nonnegative integer")
    return value


__all__ = [
    "TokenBatchBlock",
    "count_partition_microbatches",
    "iter_partition_batch_blocks",
    "iter_partition_batches",
]

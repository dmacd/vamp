"""Memory-mapped dynamic base/node batching for query-native partitions."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Iterator

import numpy as np

from apm.data.text.tinyworlds_q_semantic.contracts import (
    BENCHMARK_ID,
    QueryExperimentPreset,
    QueryPartitionArtifact,
    canonical_json_bytes,
)
from apm.lm.text_data import TokenBatch


DOCUMENTS_PER_BLOCK = 1_024


@dataclass(frozen=True, slots=True)
class QueryTokenBatchBlock:
    """One epoch-shuffled source block containing fixed-shape microbatches."""

    source_block: int
    shuffled_block: int
    batches: tuple[TokenBatch, ...]

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value < 0
            for value in (self.source_block, self.shuffled_block)
        ):
            raise ValueError("query batch block indexes must be nonnegative")
        if type(self.batches) is not tuple or not self.batches or any(
            type(batch) is not TokenBatch for batch in self.batches
        ):
            raise ValueError("query batch blocks require immutable TokenBatch values")


@dataclass(frozen=True, slots=True)
class _BlockDescriptor:
    source_block: int
    byte_start: int
    byte_stop: int


def iter_query_partition_batches(
    artifact: QueryPartitionArtifact,
    preset: QueryExperimentPreset,
    *,
    role: str,
    split: str,
    epoch: int,
    concept_id: str | None = None,
    context_length: int | None = None,
    microbatch_size: int | None = None,
) -> Iterator[TokenBatch]:
    """Yield deterministic fixed-shape base or node microbatches."""
    for block in iter_query_partition_batch_blocks(
        artifact,
        preset,
        role=role,
        split=split,
        epoch=epoch,
        concept_id=concept_id,
        context_length=context_length,
        microbatch_size=microbatch_size,
    ):
        yield from block.batches


def iter_query_partition_batch_blocks(
    artifact: QueryPartitionArtifact,
    preset: QueryExperimentPreset,
    *,
    role: str,
    split: str,
    epoch: int,
    concept_id: str | None = None,
    context_length: int | None = None,
    microbatch_size: int | None = None,
) -> Iterator[QueryTokenBatchBlock]:
    """Shuffle bounded document blocks while preserving order inside each block."""
    if type(epoch) is not int or epoch < 0:
        raise ValueError("query batch epoch must be nonnegative")
    _require_preset_binding(artifact, preset)
    effective_context_length = _positive_override(
        context_length,
        preset.context_length,
        "context_length",
    )
    effective_microbatch_size = _positive_override(
        microbatch_size,
        preset.microbatch_size,
        "microbatch_size",
    )
    if effective_context_length > preset.model_config.max_position_embeddings:
        raise ValueError("query batch context exceeds model positions")
    index_path = artifact.root / "indexes" / _index_filename(
        artifact,
        role,
        split,
        concept_id,
    )
    descriptors = _index_blocks(index_path)
    if not descriptors:
        raise ValueError("query partition selector contains no documents")
    selector = f"{role}:{concept_id or 'root'}:{split}"
    ordered = tuple(
        sorted(
            descriptors,
            key=lambda descriptor: (
                sha256(
                    (
                        f"{BENCHMARK_ID}\0block-shuffle\0{artifact.partition_sha256}"
                        f"\0{selector}\0{epoch}\0{descriptor.source_block}"
                    ).encode("utf-8")
                ).hexdigest(),
                descriptor.source_block,
            ),
        )
    )
    token_values = np.memmap(
        artifact.root / "tokens.uint16",
        dtype="<u2",
        mode="r",
    )
    try:
        for shuffled_block, descriptor in enumerate(ordered):
            batches = _records_to_batches(
                _read_block_records(index_path, descriptor),
                token_values,
                artifact.pad_token_id,
                effective_context_length,
                effective_microbatch_size,
            )
            if batches:
                yield QueryTokenBatchBlock(
                    descriptor.source_block,
                    shuffled_block,
                    batches,
                )
    finally:
        del token_values


def count_query_partition_microbatches(
    artifact: QueryPartitionArtifact,
    preset: QueryExperimentPreset,
    *,
    role: str,
    split: str,
    concept_id: str | None = None,
    context_length: int | None = None,
    microbatch_size: int | None = None,
) -> int:
    """Count stable padded microbatches without mapping the token payload."""
    _require_preset_binding(artifact, preset)
    effective_context_length = _positive_override(
        context_length,
        preset.context_length,
        "context_length",
    )
    effective_microbatch_size = _positive_override(
        microbatch_size,
        preset.microbatch_size,
        "microbatch_size",
    )
    index_path = artifact.root / "indexes" / _index_filename(
        artifact,
        role,
        split,
        concept_id,
    )
    return sum(
        math.ceil(
            sum(
                math.ceil(
                    max(0, _integer(record, "token_count") - 1)
                    / effective_context_length
                )
                for record in _read_block_records(index_path, descriptor)
            )
            / effective_microbatch_size
        )
        for descriptor in _index_blocks(index_path)
    )


def _require_preset_binding(
    artifact: QueryPartitionArtifact,
    preset: QueryExperimentPreset,
) -> None:
    if artifact.concept_ids[: preset.active_world_count] != preset.concept_ids:
        raise ValueError("query batching preset is not an active partition prefix")
    if artifact.tokenizer_identity.vocab_size != preset.model_config.vocab_size:
        raise ValueError("query partition and model vocabularies differ")


def _positive_override(value: int | None, default: int, label: str) -> int:
    if value is None:
        return default
    if type(value) is not int or value <= 0:
        raise ValueError(f"query batch {label} must be positive")
    return value


def _index_filename(
    artifact: QueryPartitionArtifact,
    role: str,
    split: str,
    concept_id: str | None,
) -> str:
    if role == "base" and concept_id is None and split in (
        "train",
        "validation",
        "test",
    ):
        return f"base-{split}.jsonl"
    if (
        role == "node"
        and concept_id in artifact.concept_ids
        and split in ("train", "validation")
    ):
        return f"node-{concept_id}-{split}.jsonl"
    raise ValueError("query batch selector is invalid")


def _index_blocks(path: Path) -> tuple[_BlockDescriptor, ...]:
    descriptors: list[_BlockDescriptor] = []
    with path.open("rb") as source:
        source_block = 0
        block_start = 0
        document_count = 0
        while line := source.readline():
            if not line.endswith(b"\n"):
                raise ValueError("query partition index contains an unterminated line")
            document_count += 1
            if document_count == DOCUMENTS_PER_BLOCK:
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
            except (UnicodeError, json.JSONDecodeError) as error:
                raise ValueError("invalid query partition index record") from error
            if type(value) is not dict or canonical_json_bytes(value) != line:
                raise ValueError("query partition index record is not canonical")
            records.append(value)
        if source.tell() != descriptor.byte_stop:
            raise ValueError("query partition block ends inside a document record")
    return tuple(records)


def _records_to_batches(
    records: tuple[dict[str, object], ...],
    token_values: np.memmap,
    pad_token_id: int,
    context_length: int,
    batch_size: int,
) -> tuple[TokenBatch, ...]:
    rows: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    batches: list[TokenBatch] = []
    for record in records:
        token_offset = _integer(record, "token_offset")
        token_count = _integer(record, "token_count")
        stop = token_offset + token_count
        if stop > token_values.shape[0]:
            raise ValueError("query partition token index exceeds its payload")
        tokens = np.asarray(token_values[token_offset:stop], dtype=np.int32)
        for start in range(0, max(token_count - 1, 0), context_length):
            chunk = tokens[start : start + context_length + 1]
            transition_count = len(chunk) - 1
            if transition_count <= 0:
                continue
            input_ids = np.full(context_length, pad_token_id, dtype=np.int32)
            target_ids = np.full(context_length, pad_token_id, dtype=np.int32)
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
        rows.extend(
            (
                np.full(context_length, pad_token_id, dtype=np.int32),
                np.zeros(context_length, dtype=np.bool_),
                np.full(context_length, pad_token_id, dtype=np.int32),
                np.zeros(context_length, dtype=np.bool_),
            )
            for _ in range(batch_size - len(rows))
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
        raise ValueError(f"query partition {field} must be nonnegative")
    return value


__all__ = [
    "DOCUMENTS_PER_BLOCK",
    "QueryTokenBatchBlock",
    "count_query_partition_microbatches",
    "iter_query_partition_batch_blocks",
    "iter_query_partition_batches",
]

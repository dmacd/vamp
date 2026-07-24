"""Memory-mapped batching for strictly loaded semantic-v1 partitions."""

from __future__ import annotations

from collections.abc import Iterator

from apm.data.text.tinyworlds_p.batching import (
    TokenBatchBlock,
    count_partition_microbatches as _count_archive_microbatches,
    iter_partition_batch_blocks as _iter_archive_blocks,
    iter_partition_batches as _iter_archive_batches,
)
from apm.data.text.tinyworlds_p.contracts import PartitionArtifact
from apm.data.text.tinyworlds_p_semantic.contracts import SemanticPartitionArtifact
from apm.data.text.tinyworlds_p_semantic.partition_runtime import semantic_runtime_view
from apm.lm.text_data import TokenBatch


def iter_partition_batches(
    artifact: SemanticPartitionArtifact,
    split: str,
    epoch: int,
) -> Iterator[TokenBatch]:
    """Yield semantic-v1 microbatches without exposing archive-v1 loaders."""
    yield from _iter_archive_batches(_runtime_view(artifact), split, epoch)


def iter_partition_batch_blocks(
    artifact: SemanticPartitionArtifact,
    split: str,
    epoch: int,
) -> Iterator[TokenBatchBlock]:
    """Yield deterministic block-shuffled semantic-v1 batches."""
    yield from _iter_archive_blocks(_runtime_view(artifact), split, epoch)


def count_partition_microbatches(
    artifact: SemanticPartitionArtifact,
    split: str,
) -> int:
    """Count stable semantic-v1 microbatches without mapping token shards."""
    return _count_archive_microbatches(_runtime_view(artifact), split)


def _runtime_view(artifact: SemanticPartitionArtifact) -> PartitionArtifact:
    return semantic_runtime_view(artifact, SemanticPartitionArtifact)


__all__ = [
    "TokenBatchBlock",
    "count_partition_microbatches",
    "iter_partition_batch_blocks",
    "iter_partition_batches",
]

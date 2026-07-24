"""Memory-mapped batching for strictly loaded semantic-v6 partitions."""

from __future__ import annotations

from collections.abc import Iterator

from apm.data.text.tinyworlds_p.batching import (
    TokenBatchBlock,
    count_partition_microbatches as count_archive_microbatches,
    iter_partition_batch_blocks as iter_archive_blocks,
    iter_partition_batches as iter_archive_batches,
)
from apm.data.text.tinyworlds_p.contracts import PartitionArtifact
from apm.data.text.tinyworlds_p_semantic.partition_runtime import semantic_runtime_view
from apm.data.text.tinyworlds_p_semantic.v6_partition_contracts import (
    V6SemanticPartitionArtifact,
)
from apm.lm.text_data import TokenBatch


def iter_v6_partition_batches(
    artifact: V6SemanticPartitionArtifact,
    split: str,
    epoch: int,
) -> Iterator[TokenBatch]:
    """Yield semantic-v6 batches through the authenticated archive runtime."""
    yield from iter_archive_batches(_runtime_view(artifact), split, epoch)


def iter_v6_partition_batch_blocks(
    artifact: V6SemanticPartitionArtifact,
    split: str,
    epoch: int,
) -> Iterator[TokenBatchBlock]:
    """Yield deterministic semantic-v6 block-shuffled batches."""
    yield from iter_archive_blocks(_runtime_view(artifact), split, epoch)


def count_v6_partition_microbatches(
    artifact: V6SemanticPartitionArtifact,
    split: str,
) -> int:
    """Count semantic-v6 microbatches without mapping token shards."""
    return count_archive_microbatches(_runtime_view(artifact), split)


def _runtime_view(artifact: V6SemanticPartitionArtifact) -> PartitionArtifact:
    return semantic_runtime_view(artifact, V6SemanticPartitionArtifact)


__all__ = [
    "TokenBatchBlock",
    "count_v6_partition_microbatches",
    "iter_v6_partition_batch_blocks",
    "iter_v6_partition_batches",
]

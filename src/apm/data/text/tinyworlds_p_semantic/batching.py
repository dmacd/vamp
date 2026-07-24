"""Memory-mapped batching for strictly loaded semantic-v1 partitions."""

from __future__ import annotations

from collections.abc import Iterator

from apm.data.text.tinyworlds_p.batching import (
    TokenBatchBlock,
    count_partition_microbatches as _count_archive_microbatches,
    iter_partition_batch_blocks as _iter_archive_blocks,
    iter_partition_batches as _iter_archive_batches,
)
from apm.data.text.tinyworlds_p.contracts import (
    ArtifactFile,
    PartitionArtifact,
    PartitionPreset,
)
from apm.data.text.tinyworlds_p_semantic.contracts import SemanticPartitionArtifact
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
    if type(artifact) is not SemanticPartitionArtifact:
        raise TypeError("semantic batching requires SemanticPartitionArtifact")
    preset = artifact.preset
    archive_preset = PartitionPreset(
        bucket_count=artifact.semantic_catalog.config.cluster_count,
        public_seed=preset.public_seed,
        worker_count=preset.worker_count,
        run_record_count=preset.run_record_count,
        shard_target_bytes=preset.shard_target_bytes,
        batch_block_documents=preset.batch_block_documents,
        context_length=preset.context_length,
        batch_size=preset.batch_size,
        minimum_role_coverage=preset.minimum_role_coverage,
        selected_cell_median_tolerance=preset.selected_cell_median_tolerance,
        minimum_component_outside_groups=preset.minimum_component_outside_groups,
        world_split_weights=preset.world_split_weights,
        base_split_weights=preset.base_split_weights,
        control_token_tolerance=preset.control_token_tolerance,
        control_source_feature_tolerance=preset.control_source_feature_tolerance,
        control_adjective_length_tolerance=preset.control_adjective_length_tolerance,
        control_mean_length_tolerance=preset.control_mean_length_tolerance,
    )
    return PartitionArtifact(
        root=artifact.root,
        partition_sha256=artifact.partition_sha256,
        manifest_sha256=artifact.manifest_sha256,
        archive_identity=artifact.archive_identity,
        tokenizer_identity=artifact.tokenizer_identity,
        normalization=artifact.normalization,
        preset=archive_preset,
        buckets=(),
        cells=artifact.cells,
        controls=artifact.controls,
        split_counts=artifact.split_counts,
        files=(),
        pad_token_id=artifact.pad_token_id,
        eos_token_id=artifact.eos_token_id,
    )


__all__ = [
    "TokenBatchBlock",
    "count_partition_microbatches",
    "iter_partition_batch_blocks",
    "iter_partition_batches",
]

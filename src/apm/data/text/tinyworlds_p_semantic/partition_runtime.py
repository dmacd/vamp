"""Source-neutral runtime projection for strictly typed semantic partitions."""

from __future__ import annotations

from typing import TypeVar

from apm.data.text.tinyworlds_p.contracts import PartitionArtifact, PartitionPreset
from apm.data.text.tinyworlds_p_semantic.contracts import SemanticPartitionArtifact
from apm.data.text.tinyworlds_p_semantic.v6_partition_contracts import (
    V6SemanticPartitionArtifact,
)


SemanticRuntimeArtifact = SemanticPartitionArtifact | V6SemanticPartitionArtifact
ArtifactT = TypeVar("ArtifactT", SemanticPartitionArtifact, V6SemanticPartitionArtifact)


def semantic_runtime_view(
    artifact: ArtifactT,
    expected_type: type[ArtifactT],
) -> PartitionArtifact:
    """Project one strict semantic artifact into the shared archive runtime."""
    if type(artifact) is not expected_type:
        raise TypeError(
            f"semantic runtime requires {expected_type.__name__}, "
            f"not {type(artifact).__name__}"
        )
    preset = artifact.preset
    return PartitionArtifact(
        root=artifact.root,
        partition_sha256=artifact.partition_sha256,
        manifest_sha256=artifact.manifest_sha256,
        archive_identity=artifact.archive_identity,
        tokenizer_identity=artifact.tokenizer_identity,
        normalization=artifact.normalization,
        preset=PartitionPreset(
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
            control_adjective_length_tolerance=(
                preset.control_adjective_length_tolerance
            ),
            control_mean_length_tolerance=preset.control_mean_length_tolerance,
        ),
        buckets=(),
        cells=artifact.cells,
        controls=artifact.controls,
        split_counts=artifact.split_counts,
        files=(),
        pad_token_id=artifact.pad_token_id,
        eos_token_id=artifact.eos_token_id,
    )


__all__ = ["SemanticRuntimeArtifact", "semantic_runtime_view"]

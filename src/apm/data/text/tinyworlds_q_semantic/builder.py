"""Fresh pinned-archive construction entry points for review and partition work."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from apm.data.text.tinyworlds_p.archive_ingest import build_archive_ingest
from apm.data.text.tinyworlds_p.contracts import (
    CANONICAL_ARCHIVE_IDENTITY,
    CANONICAL_TOKENIZER_IDENTITY,
    NORMALIZATION_IDENTITY,
    PartitionInputs,
    PartitionPreset,
    ProgressCallback,
    ProgressEvent,
)
from apm.data.text.tinyworlds_q_semantic.contracts import (
    ConceptDefinition,
    QueryPartitionArtifact,
    SemanticQueryCatalog,
)
from apm.data.text.tinyworlds_q_semantic.partition import (
    QUERY_PARTITION_PRESET,
    QueryPartitionPreset,
    build_query_partition,
)
from apm.data.text.tinyworlds_q_semantic.review import (
    SemanticReviewPacket,
    discover_review_packet,
    is_construction_group,
    publish_review_packet,
)
from apm.data.text.tinyworlds_q_semantic.source import iter_query_story_groups


@dataclass(frozen=True, slots=True)
class QueryReviewBuildInputs:
    """Pinned sources and bounded work locations for one fresh review build."""

    archive_path: Path
    tokenizer_directory: Path
    output_root: Path
    temporary_directory: Path
    concepts: tuple[ConceptDefinition, ...]
    worker_count: int = 24
    run_record_count: int = 50_000
    progress: ProgressCallback | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        for name in (
            "archive_path",
            "tokenizer_directory",
            "output_root",
            "temporary_directory",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))
        if not self.concepts:
            raise ValueError("review construction requires an ordered concept manifest")
        if any(type(value) is not int or value <= 0 for value in (self.worker_count, self.run_record_count)):
            raise ValueError("review worker and sort-run counts must be positive")


@dataclass(frozen=True, slots=True)
class QueryPartitionBuildInputs:
    """Pinned sources and work locations for a fresh full partition replay."""

    archive_path: Path
    tokenizer_directory: Path
    output_root: Path
    temporary_directory: Path
    catalog: SemanticQueryCatalog
    worker_count: int = 24
    run_record_count: int = 50_000
    progress: ProgressCallback | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        for name in (
            "archive_path",
            "tokenizer_directory",
            "output_root",
            "temporary_directory",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))
        if type(self.catalog) is not SemanticQueryCatalog:
            raise TypeError("partition construction requires a sealed reviewed catalog")
        if any(
            type(value) is not int or value <= 0
            for value in (self.worker_count, self.run_record_count)
        ):
            raise ValueError("partition worker and sort-run counts must be positive")


def build_query_review_from_archive(
    inputs: QueryReviewBuildInputs,
) -> tuple[SemanticReviewPacket, Path]:
    """Rebuild duplicate groups from the pinned archive and publish proposals only."""
    print(
        f"TinyWorlds-Q review temporary artifacts: {inputs.temporary_directory}",
        flush=True,
    )
    archive = build_archive_ingest(
        PartitionInputs(
            archive_path=inputs.archive_path,
            tokenizer_directory=inputs.tokenizer_directory,
            output_root=inputs.output_root,
            temporary_directory=inputs.temporary_directory,
            archive_identity=CANONICAL_ARCHIVE_IDENTITY,
            tokenizer_identity=CANONICAL_TOKENIZER_IDENTITY,
            progress=inputs.progress,
        ),
        PartitionPreset(
            worker_count=inputs.worker_count,
            run_record_count=inputs.run_record_count,
        ),
        NORMALIZATION_IDENTITY,
    )
    nonempty_group_count = (
        archive.audit.archive_group_count - archive.audit.empty_group_count
    )
    groups = iter_query_story_groups(
        archive.groups_path,
        archive.story_spool_path,
        archive.token_spool_path,
        group_filter=is_construction_group,
        progress=inputs.progress,
        total_group_count=nonempty_group_count,
    )
    packet = discover_review_packet(groups, inputs.concepts)
    if inputs.progress is not None:
        inputs.progress(
            ProgressEvent(
                "publish",
                0,
                1,
                "rendering and authenticating the human review packet",
            )
        )
    review_root = publish_review_packet(packet, inputs.output_root)
    if inputs.progress is not None:
        inputs.progress(
            ProgressEvent(
                "publish",
                1,
                1,
                "review packet JSON, Markdown, and standalone HTML published",
            )
        )
    return packet, review_root


def build_query_partition_from_archive(
    inputs: QueryPartitionBuildInputs,
    preset: QueryPartitionPreset = QUERY_PARTITION_PRESET,
) -> QueryPartitionArtifact:
    """Replay the pinned archive and publish a fresh query-native partition."""
    print(
        f"TinyWorlds-Q partition/archive temporary artifacts: "
        f"{inputs.temporary_directory}",
        flush=True,
    )
    archive = build_archive_ingest(
        PartitionInputs(
            archive_path=inputs.archive_path,
            tokenizer_directory=inputs.tokenizer_directory,
            output_root=inputs.output_root,
            temporary_directory=inputs.temporary_directory,
            archive_identity=CANONICAL_ARCHIVE_IDENTITY,
            tokenizer_identity=CANONICAL_TOKENIZER_IDENTITY,
            progress=inputs.progress,
        ),
        PartitionPreset(
            worker_count=inputs.worker_count,
            run_record_count=inputs.run_record_count,
        ),
        NORMALIZATION_IDENTITY,
    )
    return build_query_partition(
        iter_query_story_groups(
            archive.groups_path,
            archive.story_spool_path,
            archive.token_spool_path,
        ),
        inputs.catalog,
        inputs.output_root,
        pad_token_id=archive.pad_token_id,
        eos_token_id=archive.eos_token_id,
        preset=preset,
        progress=inputs.progress,
        total_group_count=(
            archive.audit.archive_group_count - archive.audit.empty_group_count
        ),
    )


__all__ = [
    "QueryReviewBuildInputs",
    "QueryPartitionBuildInputs",
    "build_query_partition_from_archive",
    "build_query_review_from_archive",
]

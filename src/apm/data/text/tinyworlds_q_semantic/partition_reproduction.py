"""Common retained-index and fresh-archive partition reproduction workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import tempfile

from apm.data.text.tinyworlds_p.contracts import ProgressCallback
from apm.data.text.tinyworlds_q_semantic.builder import (
    QueryPartitionBuildInputs,
    build_query_partition_from_archive,
)
from apm.data.text.tinyworlds_q_semantic.catalog import publish_catalog
from apm.data.text.tinyworlds_q_semantic.contracts import (
    QueryPartitionArtifact,
    SemanticQueryCatalog,
    canonical_json_bytes,
)
from apm.data.text.tinyworlds_q_semantic.partition import (
    build_query_partition,
    load_query_partition,
    tree_sha256,
)
from apm.data.text.tinyworlds_q_semantic.source import iter_query_story_groups
from apm.lm.text import TokenizersTextTokenizer


@dataclass(frozen=True, slots=True)
class QueryPartitionReproductionInputs:
    """Exact sources and distinct roots for one two-path reproduction proof."""

    primary_catalog: SemanticQueryCatalog
    rebuilt_catalog: SemanticQueryCatalog
    primary_output_root: Path
    rebuilt_output_root: Path
    retained_archive_directory: Path
    archive_path: Path
    tokenizer_directory: Path
    temporary_root: Path
    worker_count: int = 24
    rebuild_run_record_count: int = 37_000
    progress: ProgressCallback | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        for name in (
            "primary_output_root",
            "rebuilt_output_root",
            "retained_archive_directory",
            "archive_path",
            "tokenizer_directory",
            "temporary_root",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))
        if (
            type(self.primary_catalog) is not SemanticQueryCatalog
            or type(self.rebuilt_catalog) is not SemanticQueryCatalog
            or self.primary_catalog != self.rebuilt_catalog
        ):
            raise ValueError("partition reproduction requires byte-equal catalogs")
        if self.primary_output_root == self.rebuilt_output_root:
            raise ValueError("partition reproduction requires independent output roots")
        if any(
            type(value) is not int or value <= 0
            for value in (self.worker_count, self.rebuild_run_record_count)
        ):
            raise ValueError("partition reproduction worker and run counts must be positive")


@dataclass(frozen=True, slots=True)
class QueryPartitionReproduction:
    """Authenticated identities from two byte-identical build paths."""

    primary: QueryPartitionArtifact
    rebuilt: QueryPartitionArtifact
    catalog_tree_sha256: str
    partition_tree_sha256: str


def reproduce_query_partition(
    inputs: QueryPartitionReproductionInputs,
) -> QueryPartitionReproduction:
    """Build from a retained index and fresh archive, then compare every byte."""
    primary_catalog_root = publish_catalog(
        inputs.primary_catalog,
        inputs.primary_output_root,
    )
    rebuilt_catalog_root = publish_catalog(
        inputs.rebuilt_catalog,
        inputs.rebuilt_output_root,
    )
    catalog_tree = tree_sha256(primary_catalog_root)
    if catalog_tree != tree_sha256(rebuilt_catalog_root):
        raise RuntimeError("independent catalog trees are not byte-identical")
    primary = _existing_partition(
        inputs.primary_output_root,
        inputs.primary_catalog,
    )
    if primary is None:
        tokenizer = TokenizersTextTokenizer.from_file(
            inputs.tokenizer_directory / "tokenizer.json"
        )
        group_count = _nonempty_archive_group_count(
            inputs.retained_archive_directory / "archive-ingest.json"
        )
        primary = build_query_partition(
            iter_query_story_groups(
                inputs.retained_archive_directory / "archive-groups.jsonl",
                inputs.retained_archive_directory / "archive-stories.bin",
                inputs.retained_archive_directory / "archive-tokens.uint16",
                progress=inputs.progress,
                total_group_count=group_count,
            ),
            inputs.primary_catalog,
            inputs.primary_output_root,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            progress=inputs.progress,
            total_group_count=group_count,
        )
    rebuilt = _existing_partition(
        inputs.rebuilt_output_root,
        inputs.rebuilt_catalog,
    )
    if rebuilt is None:
        inputs.temporary_root.mkdir(parents=True, exist_ok=True)
        temporary_directory = Path(
            tempfile.mkdtemp(
                prefix="query-partition-rebuild-",
                dir=inputs.temporary_root,
            )
        )
        rebuilt = build_query_partition_from_archive(
            QueryPartitionBuildInputs(
                archive_path=inputs.archive_path,
                tokenizer_directory=inputs.tokenizer_directory,
                output_root=inputs.rebuilt_output_root,
                temporary_directory=temporary_directory,
                catalog=inputs.rebuilt_catalog,
                worker_count=inputs.worker_count,
                run_record_count=inputs.rebuild_run_record_count,
                progress=inputs.progress,
            )
        )
    primary_tree = tree_sha256(primary.root)
    rebuilt_tree = tree_sha256(rebuilt.root)
    if (
        primary.partition_sha256 != rebuilt.partition_sha256
        or primary_tree != rebuilt_tree
    ):
        raise RuntimeError("independent query partitions are not byte-identical")
    return QueryPartitionReproduction(
        primary=primary,
        rebuilt=rebuilt,
        catalog_tree_sha256=catalog_tree,
        partition_tree_sha256=primary_tree,
    )


def _existing_partition(
    output_root: Path,
    catalog: SemanticQueryCatalog,
) -> QueryPartitionArtifact | None:
    partition_root = output_root / "partitions"
    matching_roots = (
        tuple(
            path
            for path in sorted(partition_root.iterdir())
            if path.is_dir()
            and len(path.name) == 64
            and _partition_catalog_sha256(path) == catalog.catalog_sha256
        )
        if partition_root.is_dir()
        else ()
    )
    if len(matching_roots) > 1:
        raise RuntimeError("multiple partitions bind the same catalog")
    return (
        load_query_partition(matching_roots[0], catalog)
        if matching_roots
        else None
    )


def _partition_catalog_sha256(root: Path) -> str | None:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return None
    record = json.loads(manifest_path.read_bytes())
    if type(record) is not dict:
        return None
    value = record.get("catalog_sha256")
    return value if type(value) is str else None


def _nonempty_archive_group_count(audit_path: Path) -> int:
    payload = audit_path.read_bytes()
    record = json.loads(payload)
    if type(record) is not dict:
        raise RuntimeError("retained archive ingest audit is not an object")
    coverage = record.get("coverage")
    if (
        canonical_json_bytes(record) != payload
        or record.get("format") != "tinyworlds-p-archive-ingest-audit"
        or record.get("passed") is not True
        or type(coverage) is not dict
        or type(coverage.get("archive_group_count")) is not int
        or type(coverage.get("empty_group_count")) is not int
    ):
        raise RuntimeError("retained archive ingest audit changed or did not pass")
    archive_groups = coverage["archive_group_count"]
    empty_groups = coverage["empty_group_count"]
    assert type(archive_groups) is int and type(empty_groups) is int
    return archive_groups - empty_groups


__all__ = [
    "QueryPartitionReproduction",
    "QueryPartitionReproductionInputs",
    "reproduce_query_partition",
]

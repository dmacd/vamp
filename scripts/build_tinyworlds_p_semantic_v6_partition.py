#!/usr/bin/env python3
"""Build, independently reproduce, and sample semantic-v6."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from apm.data.text.tinyworlds_p_semantic import (
        V4SemanticCatalog,
        V5SemanticPartitionFailure,
        V6SemanticPartitionArtifact,
        V6SemanticPartitionFailure,
        V6SemanticSampleReport,
    )


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SEMANTIC_DATA_ROOT = REPOSITORY_ROOT / "data" / "tinyworlds-p-semantic"
CATALOG_ROOT = SEMANTIC_DATA_ROOT / "catalog" / "v4"
PARENT_FAILURE_ROOT = SEMANTIC_DATA_ROOT / "v5" / "failures"
PARTITION_ROOT = SEMANTIC_DATA_ROOT / "v6"
REBUILD_ROOT = SEMANTIC_DATA_ROOT / "rebuild-verification" / "v6"
SAMPLE_REPORT_ROOT = SEMANTIC_DATA_ROOT / "sample-reports" / "v6"
WORK_ROOT = SEMANTIC_DATA_ROOT / "work"
ARCHIVE_PATH = (
    REPOSITORY_ROOT
    / "data"
    / "tinyworlds-v2"
    / "source"
    / "TinyStories_all_data.tar.gz"
)
TOKENIZER_DIRECTORY = (
    REPOSITORY_ROOT / "checkpoints" / "tinystories-8m" / "tokenizer"
)
V6_PHASE_WEIGHTS = {
    "archive": 10.0,
    "semantic-filter": 3.0,
    "topology": 2.0,
    "feasibility": 45.0,
    "splits": 12.0,
    "pairing": 2.0,
    "shards": 18.0,
    "publish": 8.0,
}


def _fixed_sources() -> tuple[V4SemanticCatalog, V5SemanticPartitionFailure]:
    from apm.data.text.tinyworlds_p_semantic import (
        V6_PARENT_CATALOG_SHA256,
        V6_PARENT_PARTITION_FAILURE_SHA256,
        load_v4_semantic_catalog,
        load_v5_partition_failure,
    )

    catalog = load_v4_semantic_catalog(CATALOG_ROOT / V6_PARENT_CATALOG_SHA256)
    failure = load_v5_partition_failure(
        PARENT_FAILURE_ROOT / V6_PARENT_PARTITION_FAILURE_SHA256
    )
    if failure.catalog_sha256 != catalog.catalog_sha256:
        raise RuntimeError("The semantic-v6 catalog and parent failure disagree.")
    return catalog, failure


def _existing_partition(
    root: Path,
    catalog: V4SemanticCatalog,
    parent: V5SemanticPartitionFailure,
) -> V6SemanticPartitionArtifact | None:
    from apm.data.text.tinyworlds_p_semantic import load_v6_partition

    candidates = (
        tuple(
            load_v6_partition(path)
            for path in sorted(root.glob("[0-9a-f]" * 64))
            if (path / "tree.json").is_file()
        )
        if root.is_dir()
        else ()
    )
    matches = tuple(
        item
        for item in candidates
        if item.semantic_catalog.catalog_sha256 == catalog.catalog_sha256
        and item.parent_partition_failure.failure_sha256 == parent.failure_sha256
    )
    if len(matches) > 1:
        raise RuntimeError("Multiple semantic-v6 partitions bind the fixed parents.")
    return matches[0] if matches else None


def _existing_failure(
    root: Path,
    catalog: V4SemanticCatalog,
    parent: V5SemanticPartitionFailure,
) -> V6SemanticPartitionFailure | None:
    from apm.data.text.tinyworlds_p_semantic import load_v6_partition_failure

    failure_root = root / "failures"
    candidates = (
        tuple(
            load_v6_partition_failure(path)
            for path in sorted(failure_root.glob("[0-9a-f]" * 64))
            if (path / "tree.json").is_file()
        )
        if failure_root.is_dir()
        else ()
    )
    matches = tuple(
        item
        for item in candidates
        if item.catalog_sha256 == catalog.catalog_sha256
        and item.parent_partition_failure_sha256 == parent.failure_sha256
    )
    if len(matches) > 1:
        raise RuntimeError("Multiple semantic-v6 failures bind the fixed parents.")
    return matches[0] if matches else None


def _existing_outcome(
    root: Path,
    catalog: V4SemanticCatalog,
    parent: V5SemanticPartitionFailure,
) -> V6SemanticPartitionArtifact | V6SemanticPartitionFailure | None:
    partition = _existing_partition(root, catalog, parent)
    failure = _existing_failure(root, catalog, parent)
    if partition is not None and failure is not None:
        raise RuntimeError("Semantic-v6 has both a partition and a failure artifact.")
    return partition if partition is not None else failure


def _build_outcome(
    catalog: V4SemanticCatalog,
    parent: V5SemanticPartitionFailure,
    output_root: Path,
    temporary_directory: Path,
    run_record_count: int,
) -> V6SemanticPartitionArtifact | V6SemanticPartitionFailure:
    from apm.data.text.tinyworlds_p_semantic import (
        CANONICAL_ARCHIVE_IDENTITY,
        CANONICAL_TOKENIZER_IDENTITY,
        V6_SEMANTIC_PARTITION_PRESET,
        V6SemanticPartitionInputs,
        build_v6_partition,
    )
    from apm.data.text.tinyworlds_p_semantic.partitioning import (
        SemanticPartitionGateError,
    )
    from apm.data.text.tinyworlds_p_semantic.progress import (
        SemanticProgressReporter,
    )

    reporter = SemanticProgressReporter(
        f"TinyWorlds-P semantic-v6 ({run_record_count:,}-record runs)",
        V6_PHASE_WEIGHTS,
    )
    try:
        try:
            return build_v6_partition(
                V6SemanticPartitionInputs(
                    archive_path=ARCHIVE_PATH,
                    tokenizer_directory=TOKENIZER_DIRECTORY,
                    semantic_catalog_directory=catalog.root,
                    parent_partition_failure_directory=parent.root,
                    output_root=output_root,
                    temporary_directory=temporary_directory,
                    archive_identity=CANONICAL_ARCHIVE_IDENTITY,
                    tokenizer_identity=CANONICAL_TOKENIZER_IDENTITY,
                    progress=reporter.archive_event,
                ),
                replace(
                    V6_SEMANTIC_PARTITION_PRESET,
                    worker_count=24,
                    run_record_count=run_record_count,
                ),
            )
        except SemanticPartitionGateError:
            failure = _existing_failure(output_root, catalog, parent)
            if failure is None:
                raise
            return failure
    finally:
        reporter.close()


def _sample_report(
    artifact: V6SemanticPartitionArtifact,
    working: Path,
) -> V6SemanticSampleReport:
    from apm.data.text.tinyworlds_p_semantic import (
        load_v6_sample_report,
        publish_v6_sample_report,
    )

    partition_root = SAMPLE_REPORT_ROOT / artifact.partition_sha256
    existing = (
        tuple(
            load_v6_sample_report(path)
            for path in sorted(partition_root.glob("[0-9a-f]" * 64))
            if (path / "tree.json").is_file()
        )
        if partition_root.is_dir()
        else ()
    )
    if len(existing) > 1:
        raise RuntimeError("Multiple semantic-v6 sample reports bind the partition.")
    if existing:
        return existing[0]
    return publish_v6_sample_report(
        artifact,
        SAMPLE_REPORT_ROOT,
        working,
    )


def main() -> int:
    """Build and reproduce either the semantic-v6 partition or its stop."""
    from apm.data.text.tinyworlds_p_semantic import (
        V6SemanticPartitionArtifact,
        V6SemanticPartitionFailure,
    )

    catalog, parent = _fixed_sources()
    primary = _existing_outcome(PARTITION_ROOT, catalog, parent)
    rebuild = _existing_outcome(REBUILD_ROOT, catalog, parent)
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    working = Path(tempfile.mkdtemp(prefix="semantic-partition-v6-", dir=WORK_ROOT))
    print(f"The semantic-v6 temporary artifacts are being written to {working}.")

    if primary is None:
        primary_directory = working / "primary"
        primary_directory.mkdir()
        primary = _build_outcome(
            catalog,
            parent,
            PARTITION_ROOT,
            primary_directory,
            50_000,
        )
    else:
        print(f"The strict primary semantic-v6 result is being reused from {primary.root}.")
    if rebuild is None:
        rebuild_directory = working / "independent-rebuild"
        rebuild_directory.mkdir()
        rebuild = _build_outcome(
            catalog,
            parent,
            REBUILD_ROOT,
            rebuild_directory,
            37_000,
        )
    else:
        print(f"The strict semantic-v6 rebuild is being reused from {rebuild.root}.")

    if type(primary) is not type(rebuild):
        raise RuntimeError("The primary and independent semantic-v6 runs disagree.")
    if isinstance(primary, V6SemanticPartitionFailure):
        if not isinstance(rebuild, V6SemanticPartitionFailure):
            raise RuntimeError("The independent run did not reproduce the v6 stop.")
        if (
            primary.failure_sha256 != rebuild.failure_sha256
            or (primary.root / "tree.json").read_bytes()
            != (rebuild.root / "tree.json").read_bytes()
        ):
            raise RuntimeError("The semantic-v6 failure rebuild is not byte-identical.")
        print(
            "Every balanced layout failed the exact comparison allocation. "
            f"The reproduced failure is {primary.failure_sha256}."
        )
        print(f"The failure audit is available at {primary.root / 'audit.md'}.")
        return 2
    if not isinstance(primary, V6SemanticPartitionArtifact):
        raise RuntimeError("The primary semantic-v6 result has an unknown type.")
    if not isinstance(rebuild, V6SemanticPartitionArtifact):
        raise RuntimeError("The independent run did not reproduce the v6 partition.")
    if (
        primary.partition_sha256 != rebuild.partition_sha256
        or (primary.root / "tree.json").read_bytes()
        != (rebuild.root / "tree.json").read_bytes()
    ):
        raise RuntimeError("The semantic-v6 partition rebuild is not byte-identical.")
    report = _sample_report(primary, working / "sample-report-publication")
    selected_cells = ", ".join(
        f"{cell.label}=({cell.noun_bucket},{cell.verb_bucket})"
        for cell in primary.cells
    )
    print(
        "Semantic-v6 produced a reproducible partition with complete comparisons. "
        f"The selected cells are {selected_cells}."
    )
    print(f"The partition identity is {primary.partition_sha256}.")
    print(f"The validation-only sample report is available at {report.root}.")
    print("The sealed test was not opened.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

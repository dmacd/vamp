#!/usr/bin/env python3
"""Build, independently reproduce, and sample-report semantic-v1 partitioning."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from apm.data.text.tinyworlds_p_semantic import (
        SemanticCatalog,
        SemanticPartitionArtifact,
        SemanticSampleReport,
    )


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SEMANTIC_DATA_ROOT = REPOSITORY_ROOT / "data" / "tinyworlds-p-semantic"
CATALOG_ROOT = SEMANTIC_DATA_ROOT / "catalog" / "v1"
PARTITION_ROOT = SEMANTIC_DATA_ROOT / "v1"
REBUILD_ROOT = SEMANTIC_DATA_ROOT / "rebuild-verification" / "v1"
SAMPLE_REPORT_ROOT = SEMANTIC_DATA_ROOT / "sample-reports" / "v1"
ARCHIVE_PATH = (
    REPOSITORY_ROOT / "data" / "tinyworlds-v2" / "source" / "TinyStories_all_data.tar.gz"
)
TOKENIZER_DIRECTORY = REPOSITORY_ROOT / "checkpoints" / "tinystories-8m" / "tokenizer"


def _fixed_catalog() -> SemanticCatalog | None:
    from apm.data.text.tinyworlds_p_semantic import (
        CANONICAL_ARCHIVE_IDENTITY,
        ENCODER_IDENTIFIER,
        ENCODER_REVISION,
        ENCODER_SNAPSHOT_IDENTITY_SHA256,
        SEMANTIC_CONFIG,
        load_semantic_catalog,
        load_semantic_catalog_failure,
        load_semantic_evidence,
    )

    candidates = tuple(
        load_semantic_catalog(path)
        for path in sorted(CATALOG_ROOT.glob("[0-9a-f]" * 64))
        if (path / "tree.json").is_file()
    ) if CATALOG_ROOT.is_dir() else ()
    matches = tuple(
        item
        for item in candidates
        if item.config == SEMANTIC_CONFIG
        and item.encoder_identity.identifier == ENCODER_IDENTIFIER
        and item.encoder_identity.revision == ENCODER_REVISION
        and item.encoder_identity.identity_sha256
        == ENCODER_SNAPSHOT_IDENTITY_SHA256
    )
    if len(matches) != 1:
        failures_root = CATALOG_ROOT / "failures"
        failures = tuple(
            load_semantic_catalog_failure(path)
            for path in sorted(failures_root.glob("[0-9a-f]" * 64))
            if (path / "tree.json").is_file()
        ) if failures_root.is_dir() else ()
        if len(failures) == 1 and not matches:
            print(
                f"[stop] semantic-v1 construction failed its frozen grid gate: "
                f"{failures[0].reason}; audit: {failures[0].root}",
                flush=True,
            )
            return None
        raise RuntimeError(
            "the fixed partition builder requires exactly one strict semantic-v1 catalog; "
            "run scripts/prepare_tinyworlds_p_semantic.py first"
        )
    catalog = matches[0]
    evidence = load_semantic_evidence(
        SEMANTIC_DATA_ROOT / "evidence" / "v1" / catalog.evidence_sha256
    )
    if (
        evidence.archive_identity != CANONICAL_ARCHIVE_IDENTITY
        or evidence.encoder_identity != catalog.encoder_identity
        or evidence.config.evidence_record() != catalog.config.evidence_record()
    ):
        raise RuntimeError("semantic catalog differs from its strict source evidence")
    return catalog


def _existing_partition(
    root: Path,
    catalog: SemanticCatalog,
) -> SemanticPartitionArtifact | None:
    from apm.data.text.tinyworlds_p_semantic import load_partition

    candidates = tuple(
        load_partition(path)
        for path in sorted(root.glob("[0-9a-f]" * 64))
        if (path / "tree.json").is_file()
    ) if root.is_dir() else ()
    matches = tuple(
        item
        for item in candidates
        if item.semantic_catalog.catalog_sha256 == catalog.catalog_sha256
    )
    if len(matches) > 1:
        raise RuntimeError("multiple strict partitions bind the fixed semantic catalog")
    return matches[0] if matches else None


def _build(
    catalog: SemanticCatalog,
    output_root: Path,
    temporary_directory: Path,
    *,
    run_record_count: int,
) -> SemanticPartitionArtifact:
    from apm.data.text.tinyworlds_p_semantic import (
        CANONICAL_ARCHIVE_IDENTITY,
        CANONICAL_TOKENIZER_IDENTITY,
        SEMANTIC_PARTITION_PRESET,
        SemanticPartitionInputs,
        build_partition,
    )
    from apm.data.text.tinyworlds_p_semantic.progress import (
        PARTITION_PHASE_WEIGHTS,
        SemanticProgressReporter,
    )

    reporter = SemanticProgressReporter(
        f"TinyWorlds-P semantic partition ({run_record_count:,}-record runs)",
        PARTITION_PHASE_WEIGHTS,
    )
    try:
        return build_partition(
            SemanticPartitionInputs(
                archive_path=ARCHIVE_PATH,
                tokenizer_directory=TOKENIZER_DIRECTORY,
                semantic_catalog_directory=catalog.root,
                output_root=output_root,
                temporary_directory=temporary_directory,
                archive_identity=CANONICAL_ARCHIVE_IDENTITY,
                tokenizer_identity=CANONICAL_TOKENIZER_IDENTITY,
                progress=reporter.archive_event,
            ),
            replace(
                SEMANTIC_PARTITION_PRESET,
                worker_count=24,
                run_record_count=run_record_count,
            ),
        )
    finally:
        reporter.close()


def _sample_report(
    artifact: SemanticPartitionArtifact,
    work_directory: Path,
) -> SemanticSampleReport:
    from apm.data.text.tinyworlds_p_semantic import (
        load_sample_report,
        publish_sample_report,
    )

    partition_root = SAMPLE_REPORT_ROOT / artifact.partition_sha256
    candidates = tuple(
        load_sample_report(path)
        for path in sorted(partition_root.glob("[0-9a-f]" * 64))
        if (path / "tree.json").is_file()
    ) if partition_root.is_dir() else ()
    matches = tuple(
        item
        for item in candidates
        if item.partition_sha256 == artifact.partition_sha256
        and item.catalog_sha256 == artifact.semantic_catalog.catalog_sha256
    )
    if len(matches) > 1:
        raise RuntimeError("multiple sample reports bind the fixed semantic partition")
    if matches:
        return matches[0]
    print(
        "[sample-report] publishing validation-only held-in, five-world, and "
        "ten control-arm provenance",
        flush=True,
    )
    return publish_sample_report(
        artifact,
        SAMPLE_REPORT_ROOT,
        work_directory / "sample-report-publication",
    )


def main() -> int:
    """Build two strict byte-identical partitions and the pre-training report."""
    catalog = _fixed_catalog()
    if catalog is None:
        print("partition/sample report: not authorized", flush=True)
        return 2
    work_root = SEMANTIC_DATA_ROOT / "work"
    work_root.mkdir(parents=True, exist_ok=True)
    working = Path(tempfile.mkdtemp(prefix="semantic-partition-v1-", dir=work_root))
    print(f"temporary artifact directory: {working}", flush=True)
    primary = _existing_partition(PARTITION_ROOT, catalog)
    if primary is None:
        primary = _build(
            catalog,
            PARTITION_ROOT,
            working / "primary",
            run_record_count=50_000,
        )
    else:
        print(f"[partition] strict primary reused: {primary.root}", flush=True)
    rebuild = _existing_partition(REBUILD_ROOT, catalog)
    if rebuild is None:
        rebuild = _build(
            catalog,
            REBUILD_ROOT,
            working / "independent-rebuild",
            run_record_count=37_000,
        )
    else:
        print(f"[rebuild] strict independent rebuild reused: {rebuild.root}", flush=True)
    if (
        rebuild.partition_sha256 != primary.partition_sha256
        or (rebuild.root / "tree.json").read_bytes()
        != (primary.root / "tree.json").read_bytes()
    ):
        raise RuntimeError("independent semantic partition rebuild is not byte-identical")
    print(
        "[rebuild] partition identity and complete authenticated tree are byte-identical",
        flush=True,
    )
    report = _sample_report(primary, working)
    print(f"partition: {primary.root}")
    print(f"partition SHA-256: {primary.partition_sha256}")
    print(f"independent rebuild: {rebuild.root}")
    print(f"sample report: {report.root}")
    print(f"sample report SHA-256: {report.report_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

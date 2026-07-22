#!/usr/bin/env python3
"""Build the fixed TinyWorlds-P Archive v1 8x8 partition from pinned data."""

from __future__ import annotations

from pathlib import Path
import tempfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    """Verify pinned inputs and publish the fixed content-addressed partition."""
    from apm.data.text.tinyworlds_p import (
        ArchiveIngestError,
        CANONICAL_ARCHIVE_IDENTITY,
        CANONICAL_TOKENIZER_IDENTITY,
        PARTITION_PRESET,
        PartitionInputs,
        build_partition,
    )
    from apm.data.text.tinyworlds_p.progress import PreparationReporter

    work_root = REPOSITORY_ROOT / "data" / "tinyworlds-p-archive" / "work"
    work_root.mkdir(parents=True, exist_ok=True)
    temporary_directory = Path(
        tempfile.mkdtemp(prefix="prepare-archive-v1-", dir=work_root)
    )
    print(f"temporary directory: {temporary_directory}", flush=True)
    reporter = PreparationReporter()
    try:
        artifact = build_partition(
            PartitionInputs(
                archive_path=(
                    REPOSITORY_ROOT
                    / "data"
                    / "tinyworlds-v2"
                    / "source"
                    / CANONICAL_ARCHIVE_IDENTITY.filename
                ),
                tokenizer_directory=(
                    REPOSITORY_ROOT
                    / "checkpoints"
                    / "tinystories-8m"
                    / "tokenizer"
                ),
                output_root=(
                    REPOSITORY_ROOT / "data" / "tinyworlds-p-archive" / "v1"
                ),
                temporary_directory=temporary_directory,
                archive_identity=CANONICAL_ARCHIVE_IDENTITY,
                tokenizer_identity=CANONICAL_TOKENIZER_IDENTITY,
                progress=reporter,
            ),
            PARTITION_PRESET,
        )
    except ArchiveIngestError as error:
        reporter.close()
        audit_path = temporary_directory / "archive-ingest.json"
        print(f"[stopped] {error}", flush=True)
        if audit_path.is_file():
            print(f"source audit: {audit_path}", flush=True)
        return 2
    print(f"partition: {artifact.root}")
    print(f"partition SHA-256: {artifact.partition_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

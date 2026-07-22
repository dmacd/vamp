#!/usr/bin/env python3
"""Build the fixed TinyWorlds-P Archive v1 8x8 partition from pinned data."""

from __future__ import annotations

from pathlib import Path
import tempfile
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tqdm import tqdm as Tqdm

    from apm.data.text.tinyworlds_p import ProgressEvent


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class _PreparationReporter:
    """Render phase and overall ETAs for the fixed long-running preparation."""

    _PHASE_MINUTES = {
        "archive": 35.0,
        "buckets": 2.0,
        "splits": 8.0,
        "shards": 20.0,
        "publish": 2.0,
    }

    def __init__(self) -> None:
        from tqdm import tqdm

        self._started = time.monotonic()
        self._seen: list[str] = []
        self._progress: Tqdm = tqdm(
            total=len(self._PHASE_MINUTES),
            desc="TinyWorlds-P preparation",
            unit="phase",
        )

    def __call__(self, event: ProgressEvent) -> None:
        """Advance phase progress and print stable human-facing ETA estimates."""
        if event.phase not in self._seen:
            if self._seen:
                self._progress.update(1)
            self._seen.append(event.phase)
        remaining_minutes = sum(
            minutes
            for phase, minutes in self._PHASE_MINUTES.items()
            if phase not in self._seen
        ) + self._PHASE_MINUTES.get(event.phase, 0.0)
        phase_minutes = self._PHASE_MINUTES.get(event.phase, 0.0)
        elapsed_minutes = (time.monotonic() - self._started) / 60.0
        self._progress.set_postfix_str(
            f"phase ETA {phase_minutes:.0f}m, overall ETA {remaining_minutes:.0f}m"
        )
        self._progress.write(
            f"[{event.phase}] {event.detail} | elapsed {elapsed_minutes:.1f}m | "
            f"phase ETA {phase_minutes:.0f}m | overall ETA {remaining_minutes:.0f}m"
        )
        if event.phase == "publish" and event.completed == event.total:
            self._progress.update(self._progress.total - self._progress.n)
            self._progress.close()

    def close(self) -> None:
        """Close the progress bar after a planned stopped preparation."""
        if not self._progress.disable:
            self._progress.close()


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

    work_root = REPOSITORY_ROOT / "data" / "tinyworlds-p-archive" / "work"
    work_root.mkdir(parents=True, exist_ok=True)
    temporary_directory = Path(
        tempfile.mkdtemp(prefix="prepare-archive-v1-", dir=work_root)
    )
    print(f"temporary directory: {temporary_directory}", flush=True)
    reporter = _PreparationReporter()
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

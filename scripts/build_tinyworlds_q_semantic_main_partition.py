#!/usr/bin/env python3
"""Publish and independently reproduce the approved five-world partition."""

from __future__ import annotations

from pathlib import Path
import time

from apm.data.text.tinyworlds_p.contracts import ProgressEvent
from apm.data.text.tinyworlds_q_semantic.partition_reproduction import (
    QueryPartitionReproductionInputs,
    reproduce_query_partition,
)
from apm.data.text.tinyworlds_q_semantic.registered_main_catalog import (
    publish_registered_main_catalog,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPOSITORY_ROOT / "data" / "tinyworlds-q-semantic"
REBUILD_ROOT = DATA_ROOT / "rebuild-verification" / "main"
ARCHIVE_WORK = DATA_ROOT / "work" / "pilot-review-primary"
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


class _Progress:
    """Render phase elapsed time and measured phase/overall ETAs."""

    def __init__(self) -> None:
        self._started_at = time.monotonic()
        self._phase_started_at: dict[str, float] = {}
        self._phase_eta: dict[str, float | None] = {}

    def __call__(self, event: ProgressEvent) -> None:
        """Print one archive or partition progress event."""
        now = time.monotonic()
        if event.phase not in self._phase_started_at:
            self._phase_started_at[event.phase] = now
            print(f"Phase: {event.phase} — {event.detail}", flush=True)
        phase_elapsed = now - self._phase_started_at[event.phase]
        phase_eta = (
            None
            if event.total is None or event.completed <= 0
            else phase_elapsed * (event.total - event.completed) / event.completed
        )
        self._phase_eta[event.phase] = phase_eta
        overall_elapsed = now - self._started_at
        overall_eta = max(
            (value for value in self._phase_eta.values() if value is not None),
            default=None,
        )
        total = "?" if event.total is None else f"{event.total:,}"
        print(
            f"TinyWorlds-Q [{event.phase}] {event.completed:,}/{total} "
            f"phase_elapsed={_duration(phase_elapsed)} "
            f"phase_eta={_duration(phase_eta)} "
            f"overall_elapsed={_duration(overall_elapsed)} "
            f"overall_eta={_duration(overall_eta)}: {event.detail}",
            flush=True,
        )


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "pending"
    rounded = max(0, round(seconds))
    hours, remainder = divmod(rounded, 3_600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}"


def main() -> int:
    """Prove exact catalog and partition reconstruction before GPU work."""
    print("Phase 1/4: rebuilding the approved catalog in two roots.", flush=True)
    _primary_freeze, primary_catalog, _primary_validation = (
        publish_registered_main_catalog(DATA_ROOT)
    )
    _rebuilt_freeze, rebuilt_catalog, _rebuilt_validation = (
        publish_registered_main_catalog(REBUILD_ROOT)
    )
    print("Phase 2/4: building from the retained 50,000-record sort index.", flush=True)
    print("Phase 3/4 follows with a fresh 37,000-record archive replay.", flush=True)
    reproduction = reproduce_query_partition(
        QueryPartitionReproductionInputs(
            primary_catalog=primary_catalog,
            rebuilt_catalog=rebuilt_catalog,
            primary_output_root=DATA_ROOT,
            rebuilt_output_root=REBUILD_ROOT,
            retained_archive_directory=ARCHIVE_WORK,
            archive_path=ARCHIVE_PATH,
            tokenizer_directory=TOKENIZER_DIRECTORY,
            temporary_root=DATA_ROOT / "work",
            worker_count=24,
            rebuild_run_record_count=37_000,
            progress=_Progress(),
        )
    )
    print("Phase 4/4: independent trees are byte-identical.", flush=True)
    print(
        f"Main catalog identity: {primary_catalog.catalog_sha256}",
        flush=True,
    )
    print(
        f"Main partition identity: {reproduction.primary.partition_sha256}",
        flush=True,
    )
    print(
        f"Catalog tree identity: {reproduction.catalog_tree_sha256}",
        flush=True,
    )
    print(
        f"Partition tree identity: {reproduction.partition_tree_sha256}",
        flush=True,
    )
    print(f"Partition audit: {reproduction.primary.root / 'audit.md'}", flush=True)
    print("The sealed test file was never opened.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

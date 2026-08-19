#!/usr/bin/env python3
"""Serve the active TinyWorlds nouns-v2 temporal dashboard independently."""

from __future__ import annotations

from pathlib import Path
import signal
from threading import Event

from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_dashboard import (
    start_dashboard_server,
)
from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_io import (
    load_canonical_json,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TEMPORAL_RESULT_ROOT = (
    REPOSITORY_ROOT
    / "results"
    / "language_cl"
    / "tinyworlds-nouns-v2"
    / "temporal-consolidation"
)


def main() -> int:
    """Serve the sole active contract until the service receives a signal."""
    work_root = TEMPORAL_RESULT_ROOT / ".work-v1"
    work_directories = tuple(
        sorted(
            path
            for path in work_root.iterdir()
            if path.is_dir() and (path / "status.json").is_file()
        )
    )
    if len(work_directories) != 1:
        raise RuntimeError(
            "dashboard requires exactly one materialized temporal contract"
        )
    work_directory = work_directories[0]
    status = load_canonical_json(work_directory / "status.json")
    if status.get("contract_sha256") != work_directory.name:
        raise ValueError("dashboard status contract identity changed")

    server = start_dashboard_server(
        work_directory,
        TEMPORAL_RESULT_ROOT / work_directory.name,
        first_port=8765,
        last_port=8765,
    )
    stopped = Event()
    for signal_number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signal_number, lambda *_: stopped.set())
    print(f"Live dashboard: {server.url}", flush=True)
    print(f"Persistent temporary directory: {work_directory}", flush=True)
    try:
        stopped.wait()
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Verify TinyWorlds-v2 Phase 1 derived records from immutable raw evidence."""

from __future__ import annotations

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from apm.data.text.tinyworlds_v2.phase1_replay import (  # noqa: E402
    verify_phase1_derived_replay,
)


def main() -> None:
    artifact = REPOSITORY_ROOT / "data" / "tinyworlds-v2" / "reference"
    result = verify_phase1_derived_replay(artifact)
    print(
        "Zero-network Phase 1 replay matched "
        f"{len(result.compared_paths)} files "
        f"({result.compared_size_bytes} bytes)."
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train or resume the frozen five-world seed-zero base and apply its gates."""

from __future__ import annotations

import os
from pathlib import Path


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from apm.data.text.tinyworlds_q_semantic.base_execution import (  # noqa: E402
    run_or_resume_query_selected_base,
)
from apm.data.text.tinyworlds_q_semantic.registered_main_partition import (  # noqa: E402
    load_registered_main_partition,
)
from apm.data.text.tinyworlds_q_semantic.registered_main_preflight import (  # noqa: E402
    load_registered_main_gpu_preflight,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_ROOT = REPOSITORY_ROOT / "checkpoints" / "tinyworlds-q-semantic-v1"
WORK_ROOT = CHECKPOINT_ROOT / "work" / "main"


def main() -> int:
    """Authenticate every prerequisite, then resume or publish the main base."""
    print("Phase 1/3: authenticating the registered main partition.", flush=True)
    _frozen, preset, catalog, partition = load_registered_main_partition()
    print("Phase 2/3: authenticating the passing main GPU preflight.", flush=True)
    preflight = load_registered_main_gpu_preflight(
        partition,
        catalog,
        preset,
        CHECKPOINT_ROOT,
    )
    print(
        f"GPU preflight {preflight.preflight_sha256}; measured peak "
        f"{preflight.measurement.allocator_peak_bytes / 2**30:.3f} GiB.",
        flush=True,
    )
    print("Phase 3/3: running the resumable fresh seed-zero base.", flush=True)
    run_or_resume_query_selected_base(
        partition,
        preset,
        preflight,
        WORK_ROOT,
        CHECKPOINT_ROOT,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

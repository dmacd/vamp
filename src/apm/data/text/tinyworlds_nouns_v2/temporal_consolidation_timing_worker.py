"""Private one-shape GPU worker for the temporal timing audit."""

from __future__ import annotations

from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_timing import (
    timing_worker_main,
)


if __name__ == "__main__":
    raise SystemExit(timing_worker_main())

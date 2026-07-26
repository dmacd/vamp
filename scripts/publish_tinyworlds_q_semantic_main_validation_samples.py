#!/usr/bin/env python3
"""Publish exact validation-only stories and queries before main GPU work."""

from pathlib import Path

from apm.data.text.tinyworlds_q_semantic.registered_main_partition import (
    MAIN_VALIDATION_SAMPLE_REPORT_SHA256,
    load_registered_main_partition,
)
from apm.data.text.tinyworlds_q_semantic.sample_report import (
    publish_query_validation_sample_report,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_ROOT = (
    REPOSITORY_ROOT
    / "data"
    / "tinyworlds-q-semantic"
    / "sample-reports"
    / "main"
)


def main() -> int:
    """Strictly load the partition and publish only validation-visible examples."""
    _frozen, _preset, catalog, partition = load_registered_main_partition()
    report = publish_query_validation_sample_report(
        partition,
        catalog,
        SAMPLE_ROOT,
    )
    if report.report_sha256 != MAIN_VALIDATION_SAMPLE_REPORT_SHA256:
        raise RuntimeError("registered main validation sample report changed")
    print(f"Validation sample report: {report.report_sha256}", flush=True)
    print(f"Exact validation stories: {report.sample_count}", flush=True)
    print(f"Validation query records: {report.validation_query_count}", flush=True)
    print(f"Report: {report.root / 'sample-report.md'}", flush=True)
    print("The sealed test file was authenticated but not opened.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

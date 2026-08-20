#!/usr/bin/env python3
"""Verify the TRACE reviewer bundle after Git LFS objects are materialized."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUN_ID = "c9743521129b5c35389903eea8e381891a582fe24c54f374395013cf746327e5"
EVIDENCE = ROOT / "evidence-volume"
EVALUATIONS = EVIDENCE / "runs" / RUN_ID / "evaluations"


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest for a local artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source_hashes() -> int:
    """Verify every file copied from the mounted evidence volume."""

    count = 0
    manifest = EVIDENCE / "SOURCE_SHA256SUMS"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        path = EVIDENCE / relative
        if not path.is_file():
            raise AssertionError(f"missing source evidence: {relative}")
        if sha256(path) != expected:
            raise AssertionError(f"source hash mismatch: {relative}")
        count += 1
    if count != 1798:
        raise AssertionError(f"expected 1798 source files, found {count}")
    return count


def verify_final_report() -> None:
    """Verify terminal job counts and every manifest-bound report component."""

    final = ROOT / "final"
    reports = final / "reports"
    marker = json.loads((final / "SAFE_TO_TERMINATE.json").read_text())
    expected_counts = {
        "CHECKPOINTED": 0,
        "COMPLETE": 562,
        "FAILED": 0,
        "PAUSED": 0,
        "PENDING": 0,
        "RUNNING": 0,
    }
    if marker["job_state_counts"] != expected_counts:
        raise AssertionError("final marker does not describe a clean 562-job run")
    for report in marker["reports"]:
        path = reports / Path(report["path"]).name
        if sha256(path) != report["sha256"]:
            raise AssertionError(f"marker-bound report mismatch: {path.name}")

    manifest = json.loads((reports / "primary-manifest.json").read_text())
    components = {
        "markdown_sha256": "primary-report.md",
        "html_sha256": "primary-report.html",
        "lineage_svg_sha256": "primary-lineage.svg",
        "merge_diagnostics_sha256": "primary-merge-diagnostics.csv",
        "merge_plot_sha256": "primary-merge-diagnostics.png",
        "calibration_sha256": "primary-retrained-parent-calibration.csv",
        "scores_sha256": "primary-scores.csv",
        "parquet_sha256": "primary-scores.parquet",
    }
    if manifest["evaluation_rows"] != 312 or manifest["interim"]:
        raise AssertionError("primary manifest row count or finality is wrong")
    for field, filename in components.items():
        if sha256(reports / filename) != manifest[field]:
            raise AssertionError(f"primary component mismatch: {filename}")


def verify_candidate_index() -> tuple[int, int]:
    """Verify indexed candidate paths, byte counts, hashes, and row counts."""

    files = 0
    rows_total = 0
    seen: set[str] = set()
    with (ROOT / "candidate-index.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            relative = row["relative_path"]
            if relative in seen:
                raise AssertionError(f"duplicate candidate index path: {relative}")
            seen.add(relative)
            path = EVALUATIONS / relative
            if path.stat().st_size != int(row["bytes"]):
                raise AssertionError(f"candidate byte count mismatch: {relative}")
            if sha256(path) != row["sha256"]:
                raise AssertionError(f"candidate hash mismatch: {relative}")
            with path.open("rb") as candidate:
                rows = sum(1 for line in candidate if line.strip())
            if rows != int(row["rows"]):
                raise AssertionError(f"candidate row count mismatch: {relative}")
            files += 1
            rows_total += rows
    if (files, rows_total) != (532, 315397):
        raise AssertionError(
            f"expected 532 files/315397 rows, found {files}/{rows_total}"
        )
    return files, rows_total


def main() -> None:
    """Run all bundle checks and print a compact success summary."""

    source_files = verify_source_hashes()
    verify_final_report()
    candidate_files, candidate_rows = verify_candidate_index()
    print(
        "TRACE reviewer bundle verified: "
        f"{source_files} source files, {candidate_files} candidate files, "
        f"{candidate_rows} candidate rows, 562 complete jobs, 312 report rows"
    )


if __name__ == "__main__":
    main()

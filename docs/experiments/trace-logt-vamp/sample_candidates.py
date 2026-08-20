#!/usr/bin/env python3
"""Index and deterministically sample TRACE raw candidate generations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import sys
from pathlib import Path
from typing import Iterable, Iterator, TypeAlias


RUN_ID = "c9743521129b5c35389903eea8e381891a582fe24c54f374395013cf746327e5"
ROOT = Path(__file__).resolve().parent
EVALUATIONS = ROOT / "evidence-volume" / "runs" / RUN_ID / "evaluations"
POLICY_NAMES = {
    "1e51d2973353ad68f99979f822cde7713401e16b4f79106bc90c540e0e18c8c7": "vamp_svd_r8_repair000",
    "ae101c7f8b800eed7ae750626b2d1db93697b662160356b34e2df60989a3489e": "vamp_svd_r8_repair005",
    "546c828af41198edfcc3520d8c0f283eb2555357e275e50fddef434791470b03": "vamp_core_tsv_r8_scale03_repair000",
    "f6efd136ccfdc34cc5e3b54607f1802f063a986aa0c678d5c8b92e7da2d34457": "vamp_core_tsv_r8_scale03_repair005",
    "b2be4a776959c3c511e2b5faf99712d69b9df713bfcf53e538c4bc9786fe5e18": "vamp_core_tsv_r8_scale05_repair000",
    "828c74a74e3d66a2cfc9d94f28364e1c66a68299ff34912bbb000bfb137ccfed": "vamp_core_tsv_r8_scale05_repair010",
}
SAMPLE_SEED = 1234
JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)
JsonObject: TypeAlias = dict[str, JsonValue]


def candidate_files() -> list[Path]:
    return sorted(EVALUATIONS.rglob("*candidates.jsonl"))


def metadata(path: Path) -> dict[str, str]:
    """Describe one candidate file using report-facing condition names."""

    relative = path.relative_to(EVALUATIONS)
    parts = relative.parts
    directory = parts[0]
    condition = POLICY_NAMES.get(directory, directory)
    policy_hash = directory if directory in POLICY_NAMES else ""
    stage = ""
    task = ""
    if len(parts) >= 2:
        stage = parts[1].removeprefix("stage-")
        if stage.isdigit():
            stage = str(int(stage))
    if len(parts) >= 4:
        task = parts[2]
    name = path.name
    if name.startswith("test-"):
        split = "test"
    elif name.startswith("validation-"):
        split = "validation"
    else:
        split = "diagnostic"
    return {
        "relative_path": relative.as_posix(),
        "condition": condition,
        "policy_hash": policy_hash,
        "stage": stage,
        "task": task,
        "split": split,
    }


def matches(meta: dict[str, str], args: argparse.Namespace) -> bool:
    """Return whether file metadata satisfies all requested exact filters."""

    filters = (
        (args.condition, "condition"),
        (args.task, "task"),
        (args.split, "split"),
        (str(args.stage) if args.stage is not None else None, "stage"),
    )
    return all(value is None or meta[key] == value for value, key in filters)


def iter_records(path: Path) -> Iterator[tuple[int, JsonObject]]:
    """Stream nonempty JSON records with their one-based source line."""

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                yield line_number, json.loads(line)


def priority(relative_path: str, line_number: int) -> int:
    material = f"{SAMPLE_SEED}\0{relative_path}\0{line_number}".encode()
    return int.from_bytes(hashlib.sha256(material).digest(), "big")


def sample_records(args: argparse.Namespace) -> Iterable[JsonObject]:
    """Select the globally lowest stable hash priorities among matching rows."""

    selected: list[tuple[int, str, int, JsonObject]] = []
    for path in candidate_files():
        meta = metadata(path)
        if not matches(meta, args):
            continue
        for line_number, record in iter_records(path):
            score = priority(meta["relative_path"], line_number)
            item = (-score, meta["relative_path"], line_number, record)
            if len(selected) < args.limit:
                heapq.heappush(selected, item)
            elif score < -selected[0][0]:
                heapq.heapreplace(selected, item)
    for negative_score, source_path, source_line, record in sorted(
        selected, key=lambda item: -item[0]
    ):
        yield {
            "sample_priority": -negative_score,
            "source_path": source_path,
            "source_line": source_line,
            "record": record,
        }


def inspect_file(path: Path) -> dict[str, JsonValue]:
    """Compute navigation and integrity metadata for one candidate file."""

    meta = metadata(path)
    digest = hashlib.sha256()
    rows = 0
    first_result_sha256 = ""
    last_result_sha256 = ""
    with path.open("rb") as handle:
        for raw_line in handle:
            digest.update(raw_line)
            if not raw_line.strip():
                continue
            rows += 1
            record = json.loads(raw_line)
            result_sha256 = str(record.get("result_sha256", ""))
            if rows == 1:
                first_result_sha256 = result_sha256
            last_result_sha256 = result_sha256
    return {
        **meta,
        "rows": rows,
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "first_result_sha256": first_result_sha256,
        "last_result_sha256": last_result_sha256,
    }


def write_index(path: Path) -> None:
    """Write the deterministic candidate-file inventory to a fixed CSV."""

    rows = [inspect_file(candidate) for candidate in candidate_files()]
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_one_per_file(path: Path) -> None:
    """Write one stable hash-priority sample from every candidate file."""

    with path.open("w", encoding="utf-8") as handle:
        for candidate in candidate_files():
            meta = metadata(candidate)
            best: tuple[int, int, JsonObject] | None = None
            for line_number, record in iter_records(candidate):
                score = priority(meta["relative_path"], line_number)
                if best is None or score < best[0]:
                    best = (score, line_number, record)
            if best is None:
                continue
            score, line_number, record = best
            envelope = {
                **meta,
                "sample_priority": score,
                "source_line": line_number,
                "record": record,
            }
            handle.write(json.dumps(envelope, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    """Parse exact sampling filters or the single derived-file rebuild action."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", help="exact report condition name")
    parser.add_argument("--task", help="exact task name, for example Py150")
    parser.add_argument("--split", choices=("test", "validation", "diagnostic"))
    parser.add_argument("--stage", type=int)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument(
        "--rebuild-derived",
        action="store_true",
        help="rewrite candidate-index.csv and candidate-sample.jsonl",
    )
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    return args


def main() -> int:
    """Rebuild derived navigation files or print a filtered stable sample."""

    args = parse_args()
    if not EVALUATIONS.is_dir():
        raise SystemExit(
            f"missing {EVALUATIONS}; materialize the Git LFS evidence first"
        )
    if args.rebuild_derived:
        write_index(ROOT / "candidate-index.csv")
        write_one_per_file(ROOT / "candidate-sample.jsonl")
        return 0
    found = False
    for record in sample_records(args):
        found = True
        print(json.dumps(record, sort_keys=True))
    if not found:
        print("no matching candidate records", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

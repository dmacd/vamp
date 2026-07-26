#!/usr/bin/env python3
"""Run the frozen five-world GPU preflight after validation-only publication."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from apm.data.text.tinyworlds_q_semantic.catalog import ValidationCatalogView
from apm.data.text.tinyworlds_q_semantic.contracts import (
    QueryExperimentPreset,
    QueryPartitionArtifact,
    canonical_json_bytes,
)
from apm.data.text.tinyworlds_q_semantic.preflight import (
    QueryGpuPreflight,
    load_query_gpu_preflight,
    run_and_publish_query_gpu_preflight,
)
from apm.data.text.tinyworlds_q_semantic.registered_main_partition import (
    MAIN_VALIDATION_SAMPLE_REPORT_SHA256,
    load_registered_main_partition,
)
from apm.data.text.tinyworlds_q_semantic.sample_report import (
    publish_query_validation_sample_report,
)
from apm.lm.text import TokenizersTextTokenizer


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_ROOT = REPOSITORY_ROOT / "checkpoints" / "tinyworlds-q-semantic-v1"
TOKENIZER_DIRECTORY = (
    REPOSITORY_ROOT / "checkpoints" / "tinystories-8m" / "tokenizer"
)
SAMPLE_ROOT = (
    REPOSITORY_ROOT
    / "data"
    / "tinyworlds-q-semantic"
    / "sample-reports"
    / "main"
)


def main() -> int:
    """Require validation samples, then publish two disposable GPU updates."""
    _frozen, preset, catalog, partition = load_registered_main_partition()
    sample_report = publish_query_validation_sample_report(
        partition,
        catalog,
        SAMPLE_ROOT,
    )
    if sample_report.report_sha256 != MAIN_VALIDATION_SAMPLE_REPORT_SHA256:
        raise RuntimeError("registered main validation sample report changed")
    print(
        f"Authenticated validation-only sample report "
        f"{sample_report.report_sha256}.",
        flush=True,
    )
    existing = _matching_preflight(partition, catalog, preset)
    if existing is not None:
        print(f"Using strict main GPU preflight {existing.directory}.", flush=True)
        return 0
    work_root = CHECKPOINT_ROOT / "work"
    work_root.mkdir(parents=True, exist_ok=True)
    working = Path(tempfile.mkdtemp(prefix="main-gpu-preflight-", dir=work_root))
    tokenizer = TokenizersTextTokenizer.from_file(
        TOKENIZER_DIRECTORY / "tokenizer.json"
    )
    preflight = run_and_publish_query_gpu_preflight(
        partition,
        catalog,
        tokenizer,
        preset,
        working,
        CHECKPOINT_ROOT / "preflight",
    )
    print(f"Main GPU preflight: {preflight.preflight_sha256}", flush=True)
    print(f"Report: {preflight.directory / 'preflight.md'}", flush=True)
    print(
        "The disposable parameters are not reusable; sealed test remains closed.",
        flush=True,
    )
    return 0


def _matching_preflight(
    partition: QueryPartitionArtifact,
    catalog: ValidationCatalogView,
    preset: QueryExperimentPreset,
) -> QueryGpuPreflight | None:
    root = CHECKPOINT_ROOT / "preflight"
    matches = (
        tuple(
            load_query_gpu_preflight(path, partition, catalog, preset)
            for path in sorted(root.iterdir())
            if path.is_dir()
            and len(path.name) == 64
            and _preflight_matches(
                path,
                partition.partition_sha256,
                catalog.catalog_sha256,
                preset.config_sha256,
            )
        )
        if root.is_dir()
        else ()
    )
    if len(matches) > 1:
        raise RuntimeError("multiple GPU preflights bind the main sources")
    return matches[0] if matches else None


def _preflight_matches(
    root: Path,
    partition_sha256: str,
    catalog_sha256: str,
    config_sha256: str,
) -> bool:
    path = root / "preflight.json"
    if not path.is_file():
        return False
    payload = path.read_bytes()
    record = json.loads(payload)
    if type(record) is not dict or canonical_json_bytes(record) != payload:
        raise ValueError(f"noncanonical GPU preflight candidate: {root}")
    return (
        record.get("partition_sha256") == partition_sha256
        and record.get("catalog_sha256") == catalog_sha256
        and record.get("config_sha256") == config_sha256
    )


if __name__ == "__main__":
    raise SystemExit(main())

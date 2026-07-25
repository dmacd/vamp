#!/usr/bin/env python3
"""Run the registered TinyWorlds-Q rabbit/horse pilot gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from apm.data.text.tinyworlds_q_semantic.catalog import load_validation_catalog
from apm.data.text.tinyworlds_q_semantic.contracts import QueryExperimentPreset
from apm.data.text.tinyworlds_q_semantic.contracts import canonical_json_bytes
from apm.data.text.tinyworlds_q_semantic.manifests import PILOT_CONCEPTS
from apm.data.text.tinyworlds_q_semantic.partition import load_query_partition
from apm.data.text.tinyworlds_q_semantic.preflight import (
    load_query_gpu_preflight,
    run_and_publish_query_gpu_preflight,
)
from apm.lm.text import TokenizersTextTokenizer


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPOSITORY_ROOT / "data" / "tinyworlds-q-semantic"
CHECKPOINT_ROOT = REPOSITORY_ROOT / "checkpoints" / "tinyworlds-q-semantic-v1"
WORK_ROOT = CHECKPOINT_ROOT / "work"
TOKENIZER_DIRECTORY = (
    REPOSITORY_ROOT / "checkpoints" / "tinystories-8m" / "tokenizer"
)
CATALOG_SHA256 = (
    "5c9c892e5d010370f9533e73c8b0ad9c9a79c244db9e2a5d7f2b4e12d4a8aa4f"
)
PARTITION_SHA256 = (
    "419e6c8b6362add9af081885066559cc34b18f5c7044894f343c7caf0091ad0c"
)


def _sources():
    catalog = load_validation_catalog(DATA_ROOT / "catalog" / CATALOG_SHA256)
    partition = load_query_partition(
        DATA_ROOT / "partitions" / PARTITION_SHA256,
        catalog,
    )
    tokenizer = TokenizersTextTokenizer.from_file(
        TOKENIZER_DIRECTORY / "tokenizer.json"
    )
    preset = QueryExperimentPreset(
        tuple(concept.concept_id for concept in PILOT_CONCEPTS),
        adapter_updates=2_000,
    )
    return catalog, partition, tokenizer, preset


def _matching_preflight(catalog, partition, preset):
    root = CHECKPOINT_ROOT / "preflight"
    if not root.is_dir():
        return None
    matches = []
    for path in sorted(root.iterdir()):
        if not path.is_dir() or len(path.name) != 64:
            continue
        record_path = path / "preflight.json"
        if not record_path.is_file():
            continue
        payload = record_path.read_bytes()
        record = json.loads(payload)
        if type(record) is not dict or canonical_json_bytes(record) != payload:
            raise ValueError(f"noncanonical GPU preflight candidate: {path}")
        if (
            record.get("catalog_sha256") != catalog.catalog_sha256
            or record.get("partition_sha256") != partition.partition_sha256
            or record.get("config_sha256") != preset.config_sha256
        ):
            continue
        preflight = load_query_gpu_preflight(
            path,
            partition,
            catalog,
            preset,
        )
        matches.append(preflight)
    if len(matches) > 1:
        raise RuntimeError("multiple GPU preflights bind the pilot sources")
    return matches[0] if matches else None


def _run_preflight():
    catalog, partition, tokenizer, preset = _sources()
    existing = _matching_preflight(catalog, partition, preset)
    if existing is not None:
        print(f"Using strict GPU preflight {existing.directory}.", flush=True)
        return existing
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    working = Path(tempfile.mkdtemp(prefix="pilot-gpu-preflight-", dir=WORK_ROOT))
    preflight = run_and_publish_query_gpu_preflight(
        partition,
        catalog,
        tokenizer,
        preset,
        working,
        CHECKPOINT_ROOT / "preflight",
    )
    print(f"GPU preflight: {preflight.preflight_sha256}", flush=True)
    print(f"Preflight report: {preflight.directory / 'preflight.md'}", flush=True)
    print("The sealed test was not opened.", flush=True)
    return preflight


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("preflight",),
        default="preflight",
        help="highest registered pilot stage to execute",
    )
    parser.parse_args()
    _run_preflight()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

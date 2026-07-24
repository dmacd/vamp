#!/usr/bin/env python3
"""Recover the authenticated topology audit from one completed failed v4 scan."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import time


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SEMANTIC_ROOT = REPOSITORY_ROOT / "data" / "tinyworlds-p-semantic"
CATALOG_ROOT = SEMANTIC_ROOT / "catalog" / "v4"
PARTITION_ROOT = SEMANTIC_ROOT / "v4"
WORK_ROOT = SEMANTIC_ROOT / "work"
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


def _duration(seconds: float) -> str:
    rounded = max(0, round(seconds))
    hours, remainder = divmod(rounded, 3_600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}"


def _catalog():
    from apm.data.text.tinyworlds_p_semantic import (
        ENCODER_SNAPSHOT_IDENTITY_SHA256,
        V4_SEMANTIC_CONFIG,
        load_v4_semantic_catalog,
    )

    matches = tuple(
        catalog
        for path in sorted(CATALOG_ROOT.glob("[0-9a-f]" * 64))
        if (path / "tree.json").is_file()
        for catalog in (load_v4_semantic_catalog(path),)
        if catalog.config == V4_SEMANTIC_CONFIG
        and catalog.encoder_identity.identity_sha256
        == ENCODER_SNAPSHOT_IDENTITY_SHA256
    )
    if len(matches) != 1:
        raise RuntimeError("failure recovery requires exactly one strict v4 catalog")
    return matches[0]


def _failed_work() -> Path:
    matches = tuple(
        path.parent
        for path in sorted(WORK_ROOT.glob("semantic-partition-v4-*/primary/semantic-groups.jsonl"))
        if (path.parent / "archive-ingest.json").is_file()
    )
    if len(matches) != 1:
        raise RuntimeError(
            "failure recovery requires exactly one completed v4 semantic-filter scan"
        )
    return matches[0]


def main() -> int:
    """Replay the topology screen without repeating archive ingestion."""
    from apm.data.text.tinyworlds_p import builder as archive_builder
    from apm.data.text.tinyworlds_p.archive_ingest import iter_archive_groups
    from apm.data.text.tinyworlds_p.partitioning import (
        balance_word_buckets,
        bucket_word_lookup,
    )
    from apm.data.text.tinyworlds_p_semantic import (
        CANONICAL_ARCHIVE_IDENTITY,
        CANONICAL_TOKENIZER_IDENTITY,
        V4_SEMANTIC_PARTITION_PRESET,
        V4SemanticPartitionInputs,
        publish_v4_partition_failure,
    )
    from apm.data.text.tinyworlds_p_semantic.builder import _seed_identity
    from apm.data.text.tinyworlds_p_semantic.partitioning import (
        audit_semantic_world_cells,
    )
    from apm.data.text.tinyworlds_p_semantic.v4_contracts import V4_BENCHMARK_ID
    from tqdm import tqdm

    catalog = _catalog()
    failed_work = _failed_work()
    groups_path = failed_work / "semantic-groups.jsonl"
    preset = replace(
        V4_SEMANTIC_PARTITION_PRESET,
        worker_count=24,
        run_record_count=50_000,
    )
    inputs = V4SemanticPartitionInputs(
        archive_path=ARCHIVE_PATH,
        tokenizer_directory=TOKENIZER_DIRECTORY,
        semantic_catalog_directory=catalog.root,
        output_root=PARTITION_ROOT,
        temporary_directory=failed_work,
        archive_identity=CANONICAL_ARCHIVE_IDENTITY,
        tokenizer_identity=CANONICAL_TOKENIZER_IDENTITY,
    )
    seed = _seed_identity(
        inputs,
        preset,
        catalog,
        benchmark_id=V4_BENCHMARK_ID,
    )
    ingest_record = json.loads((failed_work / "archive-ingest.json").read_bytes())
    total_groups = int(ingest_record["coverage"]["archive_group_count"])

    counts: Counter[str] = Counter()
    adjective_masses: Counter[str] = Counter()
    workflow_started = time.monotonic()
    started = workflow_started
    with tqdm(total=total_groups, desc="recover exclusions/adjectives", unit="group") as bar:
        for completed, group in enumerate(iter_archive_groups(groups_path), start=1):
            status = group["status"]
            active_tokens = int(group["active_token_count"])
            if status == "eligible":
                counts["retained_groups"] += 1
                counts["retained_occurrences"] += len(group["occurrences"])
                counts["retained_tokens"] += active_tokens
                adjective_masses[group["recipe"]["adjective"]] += active_tokens
            elif status in ("semantic_construction", "semantic_word_exclusion"):
                counts[f"{status}_groups"] += 1
                counts[f"{status}_tokens"] += active_tokens
            if completed % 100_000 == 0 or completed == total_groups:
                bar.update(completed - bar.n)
                elapsed = time.monotonic() - started
                eta = elapsed * max(0, total_groups - completed) / completed
                overall_elapsed = time.monotonic() - workflow_started
                overall_eta = (
                    overall_elapsed * max(0, 2 * total_groups - completed) / completed
                )
                bar.set_postfix_str(
                    f"phase ETA {_duration(eta)}, "
                    f"overall ETA ~{_duration(overall_eta)}",
                    refresh=False,
                )
    if counts["retained_tokens"] != catalog.retained_token_count:
        raise RuntimeError("recovered retained mass differs from the v4 catalog")
    adjective_buckets = balance_word_buckets(
        adjective_masses,
        "adjective",
        catalog.config.cluster_count,
        seed,
        public_seed=preset.public_seed,
    )
    noun_lookup = catalog.word_cluster("noun")
    verb_lookup = catalog.word_cluster("verb")
    adjective_lookup = bucket_word_lookup(adjective_buckets)
    source_groups = archive_builder._iter_allocation_groups(
        groups_path,
        noun_lookup,
        verb_lookup,
        adjective_lookup,
    )
    started = time.monotonic()

    def tracked_groups():
        with tqdm(
            total=counts["retained_groups"],
            desc="recover topology audit",
            unit="group",
        ) as bar:
            for completed, group in enumerate(source_groups, start=1):
                yield group
                if completed % 100_000 == 0 or completed == counts["retained_groups"]:
                    bar.update(completed - bar.n)
                    elapsed = time.monotonic() - started
                    eta = elapsed * max(0, counts["retained_groups"] - completed) / completed
                    overall_completed = total_groups + completed
                    overall_total = total_groups + counts["retained_groups"]
                    overall_elapsed = time.monotonic() - workflow_started
                    overall_eta = (
                        overall_elapsed
                        * max(0, overall_total - overall_completed)
                        / overall_completed
                    )
                    bar.set_postfix_str(
                        f"phase ETA {_duration(eta)}, "
                        f"overall ETA ~{_duration(overall_eta)}",
                        refresh=False,
                    )

    audit = audit_semantic_world_cells(
        tracked_groups(),
        catalog,
        seed,
        preset,
        benchmark_id=V4_BENCHMARK_ID,
    )
    selected = audit.selected
    if selected is None or selected.passes_median_gate(preset.selected_cell_median_tolerance):
        raise RuntimeError("recovered v4 topology does not reproduce the observed stop")
    failure = publish_v4_partition_failure(
        inputs,
        preset,
        catalog,
        seed,
        adjective_buckets,
        counts,
        audit,
        "best semantic topology violates the selected-cell token median gate",
    )
    print(f"failure audit: {failure.root}")
    print(f"failure SHA-256: {failure.failure_sha256}")
    print(f"selected cells: {selected.cells}")
    print(f"selected token masses: {selected.token_masses}")
    print(
        "median-feasible candidates (diagnostic only): "
        f"{len(audit.median_feasible_candidates):,}"
    )
    print("partition, sample report, GPU preflight, training, and sealed test remain absent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

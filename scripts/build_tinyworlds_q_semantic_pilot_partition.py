#!/usr/bin/env python3
"""Publish and independently reproduce the approved query-v1 pilot partition."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

from apm.data.text.tinyworlds_p.contracts import ProgressEvent
from apm.data.text.tinyworlds_q_semantic.approval import (
    load_primary_review_approval,
)
from apm.data.text.tinyworlds_q_semantic.builder import (
    QueryPartitionBuildInputs,
    build_query_partition_from_archive,
)
from apm.data.text.tinyworlds_q_semantic.catalog import (
    load_validation_catalog,
    publish_catalog,
)
from apm.data.text.tinyworlds_q_semantic.contracts import (
    SemanticQueryCatalog,
    canonical_json_bytes,
)
from apm.data.text.tinyworlds_q_semantic.partition import (
    QueryPartitionArtifact,
    build_query_partition,
    load_query_partition,
    tree_sha256,
)
from apm.data.text.tinyworlds_q_semantic.pilot_catalog import (
    build_approved_pilot_catalog,
)
from apm.data.text.tinyworlds_q_semantic.review import load_review_packet
from apm.data.text.tinyworlds_q_semantic.reverse_review import (
    build_pilot_reverse_review,
    load_reverse_review_approval,
)
from apm.data.text.tinyworlds_q_semantic.shortlist import (
    build_pilot_review_shortlist,
)
from apm.data.text.tinyworlds_q_semantic.source import iter_query_story_groups
from apm.lm.text import TokenizersTextTokenizer


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPOSITORY_ROOT / "data" / "tinyworlds-q-semantic"
REBUILD_ROOT = DATA_ROOT / "rebuild-verification" / "pilot"
WORK_ROOT = DATA_ROOT / "work"
PRIMARY_ARCHIVE_WORK = WORK_ROOT / "pilot-review-primary"
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
REVIEW_PACKET_SHA256 = (
    "1603f089988125c2a0782d5bb41ebb0ce113ec466ed6248b14ad4a8e0040d071"
)
PRIMARY_APPROVAL_SHA256 = (
    "fbe0db124a77ce0215b2632d12cc97320e7eeda60de77b3fe8d48384eaef539b"
)
REVERSE_APPROVAL_SHA256 = (
    "bc184647bfec6f33c04a0e527d1c70e4c1415555695fedbf5d09d4066a41bbb8"
)
CATALOG_SHA256 = (
    "5c9c892e5d010370f9533e73c8b0ad9c9a79c244db9e2a5d7f2b4e12d4a8aa4f"
)


def _approved_catalog(output_root: Path) -> tuple[SemanticQueryCatalog, Path]:
    """Strictly reload the human authority chain and compile its exact catalog."""
    tokenizer = TokenizersTextTokenizer.from_file(
        TOKENIZER_DIRECTORY / "tokenizer.json"
    )
    packet = load_review_packet(DATA_ROOT / "review" / REVIEW_PACKET_SHA256)
    shortlist = build_pilot_review_shortlist(packet, tokenizer)
    primary = load_primary_review_approval(
        DATA_ROOT / "review-approvals" / PRIMARY_APPROVAL_SHA256
    )
    reverse_review = build_pilot_reverse_review(shortlist, primary, tokenizer)
    reverse = load_reverse_review_approval(
        DATA_ROOT / "reverse-approvals" / REVERSE_APPROVAL_SHA256
    )
    catalog = build_approved_pilot_catalog(
        review_packet=packet,
        shortlist=shortlist,
        primary_approval=primary,
        reverse_review=reverse_review,
        reverse_approval=reverse,
        tokenizer=tokenizer,
    )
    if catalog.catalog_sha256 != CATALOG_SHA256:
        raise RuntimeError("the approved pilot catalog identity changed")
    root = publish_catalog(catalog, output_root)
    validation = load_validation_catalog(root)
    if validation.catalog_sha256 != catalog.catalog_sha256:
        raise RuntimeError("the validation-only catalog reload changed identity")
    return catalog, root


def _progress(event: ProgressEvent) -> None:
    total = "?" if event.total is None else f"{event.total:,}"
    print(
        f"TinyWorlds-Q [{event.phase}] {event.completed:,}/{total}: "
        f"{event.detail}",
        flush=True,
    )


def _nonempty_archive_group_count() -> int:
    audit_path = PRIMARY_ARCHIVE_WORK / "archive-ingest.json"
    payload = audit_path.read_bytes()
    record = json.loads(payload)
    if canonical_json_bytes(record) != payload:
        raise RuntimeError("the retained archive audit is not canonical JSON")
    if (
        record.get("format") != "tinyworlds-p-archive-ingest-audit"
        or record.get("passed") is not True
    ):
        raise RuntimeError("the retained archive audit did not pass")
    coverage = record.get("coverage")
    if not isinstance(coverage, dict):
        raise RuntimeError("the retained archive coverage record is missing")
    archive_groups = coverage.get("archive_group_count")
    empty_groups = coverage.get("empty_group_count")
    if type(archive_groups) is not int or type(empty_groups) is not int:
        raise RuntimeError("the retained archive group counts are invalid")
    return archive_groups - empty_groups


def _existing_partition(
    output_root: Path,
    catalog: SemanticQueryCatalog,
) -> QueryPartitionArtifact | None:
    root = output_root / "partitions"
    candidates = (
        tuple(
            load_query_partition(path, catalog)
            for path in sorted(root.iterdir())
            if path.is_dir()
            and len(path.name) == 64
            and _partition_catalog_sha256(path) == catalog.catalog_sha256
        )
        if root.is_dir()
        else ()
    )
    matches = tuple(
        artifact
        for artifact in candidates
        if artifact.catalog_sha256 == catalog.catalog_sha256
    )
    if len(matches) > 1:
        raise RuntimeError("multiple pilot partitions bind the approved catalog")
    return matches[0] if matches else None


def _partition_catalog_sha256(root: Path) -> str | None:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return None
    record = json.loads(manifest_path.read_bytes())
    value = record.get("catalog_sha256")
    return value if type(value) is str else None


def _primary_partition(catalog: SemanticQueryCatalog) -> QueryPartitionArtifact:
    existing = _existing_partition(DATA_ROOT, catalog)
    if existing is not None:
        print(f"Reusing strict primary partition {existing.root}.", flush=True)
        return existing
    tokenizer = TokenizersTextTokenizer.from_file(
        TOKENIZER_DIRECTORY / "tokenizer.json"
    )
    total_group_count = _nonempty_archive_group_count()
    return build_query_partition(
        iter_query_story_groups(
            PRIMARY_ARCHIVE_WORK / "archive-groups.jsonl",
            PRIMARY_ARCHIVE_WORK / "archive-stories.bin",
            PRIMARY_ARCHIVE_WORK / "archive-tokens.uint16",
            progress=_progress,
            total_group_count=total_group_count,
        ),
        catalog,
        DATA_ROOT,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        progress=_progress,
        total_group_count=total_group_count,
    )


def _rebuilt_partition(catalog: SemanticQueryCatalog) -> QueryPartitionArtifact:
    existing = _existing_partition(REBUILD_ROOT, catalog)
    if existing is not None:
        print(f"Reusing strict independent partition {existing.root}.", flush=True)
        return existing
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    temporary_directory = Path(
        tempfile.mkdtemp(prefix="pilot-partition-rebuild-", dir=WORK_ROOT)
    )
    return build_query_partition_from_archive(
        QueryPartitionBuildInputs(
            archive_path=ARCHIVE_PATH,
            tokenizer_directory=TOKENIZER_DIRECTORY,
            output_root=REBUILD_ROOT,
            temporary_directory=temporary_directory,
            catalog=catalog,
            worker_count=24,
            run_record_count=37_000,
            progress=_progress,
        )
    )


def main() -> int:
    """Publish the primary partition and prove a fresh replay is byte-identical."""
    primary_catalog, primary_catalog_root = _approved_catalog(DATA_ROOT)
    rebuilt_catalog, rebuilt_catalog_root = _approved_catalog(REBUILD_ROOT)
    if (
        primary_catalog.catalog_sha256 != rebuilt_catalog.catalog_sha256
        or tree_sha256(primary_catalog_root) != tree_sha256(rebuilt_catalog_root)
    ):
        raise RuntimeError("the independent pilot catalog is not byte-identical")
    print(
        f"Pilot catalog {primary_catalog.catalog_sha256} reproduced byte-for-byte; "
        "the sealed test was not deserialized.",
        flush=True,
    )
    primary = _primary_partition(primary_catalog)
    rebuilt = _rebuilt_partition(rebuilt_catalog)
    primary_tree = tree_sha256(primary.root)
    rebuilt_tree = tree_sha256(rebuilt.root)
    if (
        primary.partition_sha256 != rebuilt.partition_sha256
        or primary_tree != rebuilt_tree
    ):
        raise RuntimeError("the independent pilot partition is not byte-identical")
    print(f"Pilot partition identity: {primary.partition_sha256}", flush=True)
    print(f"Byte-identical tree identity: {primary_tree}", flush=True)
    print(f"Partition audit: {primary.root / 'audit.md'}", flush=True)
    print("The sealed test was not opened.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

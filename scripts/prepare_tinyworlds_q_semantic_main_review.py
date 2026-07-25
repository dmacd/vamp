#!/usr/bin/env python3
"""Build the proposal-only five-world review packet from the retained archive index."""

from __future__ import annotations

import json
from pathlib import Path
import time

from apm.data.text.tinyworlds_p.contracts import ProgressEvent
from apm.data.text.tinyworlds_q_semantic.contracts import canonical_json_bytes
from apm.data.text.tinyworlds_q_semantic.manifests import MAIN_CONCEPTS
from apm.data.text.tinyworlds_q_semantic.main_shortlist import (
    build_main_review_shortlist,
    main_evidence_predicates,
)
from apm.data.text.tinyworlds_q_semantic.review import (
    PredicateDefinition,
    SemanticReviewPacket,
    build_review_packet,
    discover_review_packet,
    is_construction_group,
    load_review_packet,
    publish_review_packet,
)
from apm.data.text.tinyworlds_q_semantic.shortlist import publish_review_shortlist
from apm.data.text.tinyworlds_q_semantic.source import iter_query_story_groups
from apm.data.text.tinyworlds_q_semantic.registered_main import (
    load_registered_main_authority,
)
from apm.lm.text import TokenizersTextTokenizer


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPOSITORY_ROOT / "data" / "tinyworlds-q-semantic"
ARCHIVE_WORK = DATA_ROOT / "work" / "pilot-review-primary"
RAW_REVIEW_PACKET_SHA256 = (
    "7164cd2cc18be5ba29d7106a44f23dbec5bf39a9a962b9c441ccf07501a8132f"
)
TARGETED_EVIDENCE_PACKET_SHA256 = (
    "ce1b06c7f7a325cedded9970ac008329c93d97d29c84344b93d22b450db14374"
)
TOKENIZER_PATH = (
    REPOSITORY_ROOT
    / "checkpoints"
    / "tinystories-8m"
    / "tokenizer"
    / "tokenizer.json"
)


class _Progress:
    """Small phase-aware progress renderer with elapsed time and ETA."""

    def __init__(self) -> None:
        self.started_at = time.monotonic()

    def __call__(self, event: ProgressEvent) -> None:
        elapsed = time.monotonic() - self.started_at
        eta = (
            None
            if event.total is None or event.completed <= 0
            else elapsed * (event.total - event.completed) / event.completed
        )
        eta_text = "?" if eta is None else _duration(eta)
        total = "?" if event.total is None else f"{event.total:,}"
        print(
            f"TinyWorlds-Q [{event.phase}] {event.completed:,}/{total} "
            f"elapsed={_duration(elapsed)} eta={eta_text}: {event.detail}",
            flush=True,
        )


def _duration(seconds: float) -> str:
    rounded = max(0, int(seconds))
    hours, remainder = divmod(rounded, 3_600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}"


def _nonempty_group_count() -> int:
    payload = (ARCHIVE_WORK / "archive-ingest.json").read_bytes()
    record = json.loads(payload)
    if type(record) is not dict or canonical_json_bytes(record) != payload:
        raise RuntimeError("retained archive ingest audit is not canonical JSON")
    coverage = record.get("coverage")
    if (
        record.get("format") != "tinyworlds-p-archive-ingest-audit"
        or record.get("passed") is not True
        or type(coverage) is not dict
        or type(coverage.get("archive_group_count")) is not int
        or type(coverage.get("empty_group_count")) is not int
    ):
        raise RuntimeError("retained archive ingest audit changed or did not pass")
    archive_group_count = coverage["archive_group_count"]
    empty_group_count = coverage["empty_group_count"]
    assert type(archive_group_count) is int and type(empty_group_count) is int
    return archive_group_count - empty_group_count


def _registered_targeted_packet(
    predicates: tuple[PredicateDefinition, ...],
) -> SemanticReviewPacket | None:
    """Strictly load the exact targeted packet when it is already published."""
    root = DATA_ROOT / "review" / TARGETED_EVIDENCE_PACKET_SHA256
    if not root.is_dir():
        return None
    packet = load_review_packet(root)
    if packet.concepts != MAIN_CONCEPTS or packet.predicates != predicates:
        raise RuntimeError("registered targeted evidence packet changed")
    return packet


def main() -> int:
    """Authenticate the main freeze and publish ranked, non-authoritative proposals."""
    *_authority, frozen = load_registered_main_authority()
    print(f"Phase 1/4: authenticated main freeze {frozen.freeze_sha256}.", flush=True)
    raw_root = DATA_ROOT / "review" / RAW_REVIEW_PACKET_SHA256
    if raw_root.is_dir():
        print(f"Using registered main review packet {RAW_REVIEW_PACKET_SHA256}.", flush=True)
        print(f"Review audit: {raw_root / 'review.md'}", flush=True)
    else:
        print("Phase 2/4: discovering broad ranked proposals.", flush=True)
        print(f"Temporary archive artifacts retained at {ARCHIVE_WORK}.", flush=True)
        progress = _Progress()
        total_group_count = _nonempty_group_count()
        groups = iter_query_story_groups(
            ARCHIVE_WORK / "archive-groups.jsonl",
            ARCHIVE_WORK / "archive-stories.bin",
            ARCHIVE_WORK / "archive-tokens.uint16",
            group_filter=is_construction_group,
            progress=progress,
            total_group_count=total_group_count,
        )
        raw_packet = discover_review_packet(groups, MAIN_CONCEPTS)
        root = publish_review_packet(raw_packet, DATA_ROOT)
        strict = load_review_packet(root)
        if strict != raw_packet:
            raise RuntimeError("published main review packet changed on strict reload")
        print(f"Main review packet: {raw_packet.packet_sha256}", flush=True)
        print(f"Review audit: {root / 'review.md'}", flush=True)
    print("Phase 3/4: collecting targeted evidence for candidate semantic facts.", flush=True)
    predicates = main_evidence_predicates()
    targeted = _registered_targeted_packet(predicates)
    if targeted is not None:
        print(f"Using strict targeted evidence packet {targeted.packet_sha256}.", flush=True)
        print(f"Targeted evidence: {DATA_ROOT / 'review' / targeted.packet_sha256 / 'review.md'}", flush=True)
    else:
        progress = _Progress()
        total_group_count = _nonempty_group_count()
        groups = iter_query_story_groups(
            ARCHIVE_WORK / "archive-groups.jsonl",
            ARCHIVE_WORK / "archive-stories.bin",
            ARCHIVE_WORK / "archive-tokens.uint16",
            group_filter=is_construction_group,
            progress=progress,
            total_group_count=total_group_count,
        )
        targeted = build_review_packet(groups, MAIN_CONCEPTS, predicates)
        root = publish_review_packet(targeted, DATA_ROOT)
        strict = load_review_packet(root)
        if strict != targeted:
            raise RuntimeError("published main review packet changed on strict reload")
        print(f"Targeted evidence packet: {targeted.packet_sha256}", flush=True)
        print(f"Targeted evidence: {root / 'review.md'}", flush=True)
    print("Phase 4/4: publishing the compact human decision sheet.", flush=True)
    tokenizer = TokenizersTextTokenizer.from_file(TOKENIZER_PATH)
    shortlist = build_main_review_shortlist(targeted, tokenizer)
    shortlist_root = publish_review_shortlist(shortlist, DATA_ROOT)
    print(f"Main shortlist: {shortlist.shortlist_sha256}", flush=True)
    print(f"Approval sheet: {shortlist_root / 'review.md'}", flush=True)
    print("All candidates remain proposals only; none is human-approved.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

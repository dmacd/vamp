"""Exact primary authority and reverse queue for the five-world main review."""

from __future__ import annotations

from pathlib import Path

from apm.data.text.tinyworlds_q_semantic.approval import (
    PrimaryReviewApproval,
    approve_all_primary_proposals,
    load_primary_review_approval,
    publish_primary_review_approval,
)
from apm.data.text.tinyworlds_q_semantic.contracts import CATALOG_ROOT
from apm.data.text.tinyworlds_q_semantic.main_freeze import MainExperimentFreeze
from apm.data.text.tinyworlds_q_semantic.main_reverse_review import (
    build_main_reverse_review,
)
from apm.data.text.tinyworlds_q_semantic.main_shortlist import (
    build_main_review_shortlist,
)
from apm.data.text.tinyworlds_q_semantic.registered_main import (
    load_registered_main_authority,
)
from apm.data.text.tinyworlds_q_semantic.review import (
    SemanticReviewPacket,
    load_review_packet,
)
from apm.data.text.tinyworlds_q_semantic.reverse_review import (
    ReverseReviewApproval,
    SemanticReverseReview,
    approve_all_reverse_choices,
    load_reverse_review_approval,
    publish_reverse_review,
    publish_reverse_review_approval,
)
from apm.data.text.tinyworlds_q_semantic.shortlist import SemanticReviewShortlist
from apm.lm.text import TokenizersTextTokenizer


MAIN_REVIEW_PACKET_SHA256 = (
    "ce1b06c7f7a325cedded9970ac008329c93d97d29c84344b93d22b450db14374"
)
MAIN_SHORTLIST_SHA256 = (
    "fe2f78e92e1c4e0d26280f2741beea728ea3125c932c3126b770da6cd90104cc"
)
MAIN_PRIMARY_APPROVAL_SHA256 = (
    "8b0f2868b216b837f2b2c90c0f7faaa141874fe87b2387c6fecd62faed8f616b"
)
MAIN_REVERSE_REVIEW_SHA256 = (
    "c805da6c075920f85a58b0c4ed25ee4aa6dac2e5763e2578648efd0c0800e1f0"
)
MAIN_PRIMARY_REVIEWED_AT = "2026-07-25T22:38:51Z"
MAIN_PRIMARY_AUTHORIZATION = "Approve all primaries"
MAIN_REVERSE_APPROVAL_SHA256 = (
    "c643731930ae9721ea4c4420f14a830c04ca8179bee8caccb8a73756ec0c1067"
)
MAIN_REVERSE_REVIEWED_AT = "2026-07-25T22:54:49Z"
MAIN_REVERSE_AUTHORIZATION = "Approve all reverse choices"
TOKENIZER_PATH = Path("checkpoints/tinystories-8m/tokenizer/tokenizer.json")


def load_registered_main_primary_authority() -> tuple[
    MainExperimentFreeze,
    SemanticReviewPacket,
    SemanticReviewShortlist,
    PrimaryReviewApproval,
    SemanticReverseReview,
]:
    """Rebuild and authenticate the exact main primary approval and reverse queue."""
    *_pilot_authority, frozen = load_registered_main_authority()
    tokenizer = TokenizersTextTokenizer.from_file(TOKENIZER_PATH)
    packet = load_review_packet(
        CATALOG_ROOT / "review" / MAIN_REVIEW_PACKET_SHA256
    )
    shortlist = build_main_review_shortlist(packet, tokenizer)
    if shortlist.shortlist_sha256 != MAIN_SHORTLIST_SHA256:
        raise RuntimeError("registered main shortlist identity changed")
    approval = approve_all_primary_proposals(
        shortlist,
        reviewer="interactive-user",
        reviewed_at=MAIN_PRIMARY_REVIEWED_AT,
    )
    if approval.approval_sha256 != MAIN_PRIMARY_APPROVAL_SHA256:
        raise RuntimeError("registered main primary approval identity changed")
    approval_root = publish_primary_review_approval(approval, CATALOG_ROOT)
    if load_primary_review_approval(approval_root) != approval:
        raise RuntimeError("registered main primary approval strict reload changed")
    reverse_review = build_main_reverse_review(shortlist, approval, tokenizer)
    if reverse_review.reverse_review_sha256 != MAIN_REVERSE_REVIEW_SHA256:
        raise RuntimeError("registered main reverse review identity changed")
    reverse_root = publish_reverse_review(reverse_review, CATALOG_ROOT)
    if reverse_root.name != reverse_review.reverse_review_sha256:
        raise RuntimeError("registered main reverse review publication changed")
    return frozen, packet, shortlist, approval, reverse_review


def load_registered_main_complete_review_authority() -> tuple[
    MainExperimentFreeze,
    SemanticReviewPacket,
    SemanticReviewShortlist,
    PrimaryReviewApproval,
    SemanticReverseReview,
    ReverseReviewApproval,
]:
    """Rebuild and authenticate both explicit main review approvals."""
    frozen, packet, shortlist, primary, reverse_review = (
        load_registered_main_primary_authority()
    )
    reverse = approve_all_reverse_choices(
        reverse_review,
        reviewer="interactive-user",
        reviewed_at=MAIN_REVERSE_REVIEWED_AT,
    )
    if reverse.approval_sha256 != MAIN_REVERSE_APPROVAL_SHA256:
        raise RuntimeError("registered main reverse approval identity changed")
    reverse_root = publish_reverse_review_approval(reverse, CATALOG_ROOT)
    if load_reverse_review_approval(reverse_root) != reverse:
        raise RuntimeError("registered main reverse approval strict reload changed")
    return frozen, packet, shortlist, primary, reverse_review, reverse


__all__ = [
    "MAIN_PRIMARY_APPROVAL_SHA256",
    "MAIN_PRIMARY_AUTHORIZATION",
    "MAIN_PRIMARY_REVIEWED_AT",
    "MAIN_REVIEW_PACKET_SHA256",
    "MAIN_REVERSE_APPROVAL_SHA256",
    "MAIN_REVERSE_AUTHORIZATION",
    "MAIN_REVERSE_REVIEW_SHA256",
    "MAIN_REVERSE_REVIEWED_AT",
    "MAIN_SHORTLIST_SHA256",
    "load_registered_main_complete_review_authority",
    "load_registered_main_primary_authority",
]

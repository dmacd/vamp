"""Approved five-world main catalog compilation."""

from __future__ import annotations

from apm.data.text.tinyworlds_q_semantic.approval import PrimaryReviewApproval
from apm.data.text.tinyworlds_q_semantic.approved_catalog import (
    build_approved_catalog,
)
from apm.data.text.tinyworlds_q_semantic.contracts import SemanticQueryCatalog
from apm.data.text.tinyworlds_q_semantic.main_shortlist import MAIN_SHORTLIST_SPECS
from apm.data.text.tinyworlds_q_semantic.manifests import MAIN_CONCEPTS
from apm.data.text.tinyworlds_q_semantic.reverse_review import (
    ReverseReviewApproval,
    SemanticReverseReview,
)
from apm.data.text.tinyworlds_q_semantic.review import SemanticReviewPacket
from apm.data.text.tinyworlds_q_semantic.shortlist import SemanticReviewShortlist
from apm.lm.text import TextTokenizer


def build_approved_main_catalog(
    *,
    review_packet: SemanticReviewPacket,
    shortlist: SemanticReviewShortlist,
    primary_approval: PrimaryReviewApproval,
    reverse_review: SemanticReverseReview,
    reverse_approval: ReverseReviewApproval,
    tokenizer: TextTokenizer,
) -> SemanticQueryCatalog:
    """Compile the official five-world catalog after both explicit approvals."""
    return build_approved_catalog(
        concepts=MAIN_CONCEPTS,
        shortlist_specs=MAIN_SHORTLIST_SPECS,
        review_packet=review_packet,
        shortlist=shortlist,
        primary_approval=primary_approval,
        reverse_review=reverse_review,
        reverse_approval=reverse_approval,
        tokenizer=tokenizer,
    )


__all__ = ["build_approved_main_catalog"]

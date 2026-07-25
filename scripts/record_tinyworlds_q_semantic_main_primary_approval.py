#!/usr/bin/env python3
"""Record the authorized main primaries and publish reverse choices."""

from apm.data.text.tinyworlds_q_semantic.registered_main_review import (
    load_registered_main_primary_authority,
)


def main() -> int:
    """Authenticate the primary decision and publish its reverse review queue."""
    _frozen, _packet, _shortlist, approval, reverse_review = (
        load_registered_main_primary_authority()
    )
    print(f"Main primary approval: {approval.approval_sha256}", flush=True)
    print(
        "Approval record: "
        f"data/tinyworlds-q-semantic/review-approvals/{approval.approval_sha256}/approval.md",
        flush=True,
    )
    print(f"Main reverse review: {reverse_review.reverse_review_sha256}", flush=True)
    print(
        "Reverse approval sheet: "
        f"data/tinyworlds-q-semantic/reverse-reviews/{reverse_review.reverse_review_sha256}/review.md",
        flush=True,
    )
    print("The reverse choices are not yet approved; the sealed test remains closed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

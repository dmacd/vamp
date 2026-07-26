#!/usr/bin/env python3
"""Record the authorized main reverse choices and authenticate the review chain."""

from apm.data.text.tinyworlds_q_semantic.registered_main_review import (
    load_registered_main_complete_review_authority,
)


def main() -> int:
    """Publish and strictly reload the complete five-world review authority."""
    *_authority, approval = load_registered_main_complete_review_authority()
    print(f"Main reverse approval: {approval.approval_sha256}", flush=True)
    print(
        "Approval record: "
        f"data/tinyworlds-q-semantic/reverse-approvals/"
        f"{approval.approval_sha256}/approval.md",
        flush=True,
    )
    print("All 60 facts now have complete human review authority.", flush=True)
    print("The sealed test remains closed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

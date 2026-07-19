#!/usr/bin/env python3
"""Apply explicit approval at the fixed TinyWorlds-v2 Phase 1 location."""

from apm.data.text.tinyworlds_v2.audit_io import approve_phase1_audit


def main() -> int:
    """Fail closed unless the exact passing evidence has explicit approval."""
    artifact = approve_phase1_audit()
    print(
        "Approved TinyWorlds-v2 audit "
        f"{artifact.approval.audit_sha256}; selected route "
        f"{artifact.selected_route_id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

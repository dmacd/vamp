#!/usr/bin/env python3
"""Validate the fixed TinyWorlds-v2 Phase 1 artifact location."""

from apm.data.text.tinyworlds_v2.audit_io import (
    validate_phase1_reference,
    validate_phase1_semantics,
)


def main() -> int:
    """Validate base and human-overlay artifacts and print their gate state."""
    semantic = validate_phase1_semantics()
    print(f"Phase 1 manifest: {semantic.manifest.manifest_sha256}")
    print(
        f"Semantic/replay gate: {semantic.status}; "
        f"{semantic.replay_file_count} derived files matched"
    )
    if semantic.status != "awaiting_human_audit":
        print("Human audit: not available for this recorded stop state")
        print("Phase 2 lock: closed")
        return 0
    result = validate_phase1_reference()
    if result.decision_set is None:
        print("Human audit: awaiting audit_decisions.json")
    else:
        assert result.evaluation is not None
        print(f"Audit decision digest: {result.decision_set.decision_sha256}")
        gate_state = (
            "passed"
            if result.evaluation.passed
            else "failed: " + ", ".join(result.evaluation.failures)
        )
        print(
            "Human gates: " + gate_state
        )
    if result.approval_artifact is None:
        print("Phase 2 lock: closed")
    else:
        print(
            "Phase 2 lock: explicitly approved; selected route "
            f"{result.approval_artifact.selected_route_id}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

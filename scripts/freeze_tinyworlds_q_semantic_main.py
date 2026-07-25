#!/usr/bin/env python3
"""Freeze the five-world main configuration from the successful pilot chain."""

from apm.data.text.tinyworlds_q_semantic.registered_main import (
    load_registered_main_authority,
)


def main() -> int:
    """Authenticate the pilot and publish the one registered main freeze."""
    *_authority, frozen = load_registered_main_authority()
    print(f"Main experiment freeze: {frozen.freeze_sha256}", flush=True)
    print(f"Freeze report: {frozen.root / 'report.md'}", flush=True)
    print("The main catalog may now be constructed; the sealed test remains closed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

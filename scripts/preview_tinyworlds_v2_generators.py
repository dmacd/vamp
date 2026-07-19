#!/usr/bin/env python3
"""Generate the fixed three-brief preview for all seven Phase 1 routes."""

from pathlib import Path

from apm.data.text.tinyworlds_v2.generator_preview import run_generator_preview


def main() -> None:
    """Run the isolated generator preview and stop for human review."""
    run_generator_preview(
        Path(__file__).resolve().parents[1],
        authorize_paid_preview=True,
    )


if __name__ == "__main__":
    main()

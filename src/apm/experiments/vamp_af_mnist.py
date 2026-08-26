"""Direct config-driven entry point for the VAMP-AF MNIST proof of concept."""

from __future__ import annotations

import argparse
from pathlib import Path

from apm.experiments.vamp_af_config import load_config
from apm.experiments.vamp_af_workflow import run_workflow


DEFAULT_CONFIG = Path("configs/vamp_af_mnist/poc.yaml")


def main() -> None:
    """Run or resume the one resolved smoke/main/consolidation workflow."""
    parser = argparse.ArgumentParser(description="VAMP-AF Addressable Rotated MNIST proof of concept")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="stop after the frozen-address and top-two-layer capacity preflight",
    )
    parser.add_argument(
        "--stop-after-pass",
        choices=("smoke", "main", "consolidation_stress"),
        help="durably stop after completing the selected pass; a later run resumes",
    )
    arguments = parser.parse_args()
    result = run_workflow(
        load_config(arguments.config),
        preflight_only=arguments.preflight_only,
        stop_after_pass=arguments.stop_after_pass,
    )
    print(f"Completed VAMP-AF artifacts: {result}", flush=True)


if __name__ == "__main__":
    main()


__all__ = ["DEFAULT_CONFIG", "main"]

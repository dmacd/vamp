"""Entry point for behavioral LogT routing on VAMP-AF Rotated-MNIST."""

from __future__ import annotations

import argparse
from pathlib import Path

from apm.experiments.vamp_logt_router_rotated_config import load_config
from apm.experiments.vamp_logt_router_rotated_workflow import run_workflow


DEFAULT_CONFIG = Path("configs/vamp_logt_router_rotated_mnist/primary.yaml")


def main() -> None:
    """Run or resume the selected sealed workflow phase."""
    parser = argparse.ArgumentParser(
        description="Integrated LogT behavioral routing on VAMP-AF Rotated-MNIST"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase", choices=("smoke", "primary", "all"), default="all")
    arguments = parser.parse_args()
    result = run_workflow(load_config(arguments.config), arguments.phase)
    print(f"Rotated behavioral-router artifacts: {result}", flush=True)


if __name__ == "__main__":
    main()


__all__ = ["DEFAULT_CONFIG", "main"]

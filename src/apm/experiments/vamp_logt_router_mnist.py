"""Config-driven entry point for integrated LogT behavioral routing."""

from __future__ import annotations

import argparse
from pathlib import Path

from apm.experiments.vamp_logt_router_config import load_config
from apm.experiments.vamp_logt_router_workflow import run_workflow


DEFAULT_CONFIG = Path("configs/vamp_logt_router_mnist/primary.yaml")


def main() -> None:
    """Run or resume the selected preregistered workflow phase."""
    parser = argparse.ArgumentParser(
        description="Integrated behavioral routing for true-LogT VAMP on MNIST"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase", choices=("smoke", "primary", "all"), default="all")
    arguments = parser.parse_args()
    result = run_workflow(load_config(arguments.config), arguments.phase)
    print(f"Behavioral-router workflow artifacts: {result}", flush=True)


if __name__ == "__main__":
    main()


__all__ = ["DEFAULT_CONFIG", "main"]

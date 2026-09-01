"""Command-line entry point for the dense Permuted-MNIST LogT study."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from apm.experiments.vamp_logt_mlp_permuted_config import load_config
from apm.experiments.vamp_logt_mlp_permuted_workflow import run_workflow


DEFAULT_CONFIG = Path("configs/vamp_logt_mlp_permuted_mnist/primary.yaml")


def main() -> None:
    """Run or resume one declared phase with prerequisites handled automatically."""
    parser = argparse.ArgumentParser(
        description="Dense-base LogT routing, integration, and ceiling on Permuted-MNIST"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--phase",
        choices=("calibration", "hierarchy", "online", "ceiling", "all"),
        default="all",
    )
    arguments = parser.parse_args()
    result = run_workflow(load_config(arguments.config), arguments.phase)
    print(f"Dense Permuted-MNIST artifacts: {result}", flush=True)


if __name__ == "__main__":
    main()


__all__ = ["DEFAULT_CONFIG", "main"]

"""Command entry point for the exact generalized-Gauss-Newton PC experiment."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.65")

from apm.experiments.vamp_logt_pc_config import load_config  # noqa: E402


DEFAULT_CONFIG = Path("configs/vamp_logt_pc_mnist/gauss_newton.yaml")


def main() -> None:
    """Run or resume the exact-GGN experiment in its fixed phase order."""
    parser = argparse.ArgumentParser(
        description="Exact generalized-Gauss-Newton PC evidence for true-LogT VAMP on MNIST"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    arguments = parser.parse_args()
    from apm.experiments.vamp_logt_pc_gn_workflow import run_gn_workflow

    result = run_gn_workflow(load_config(arguments.config))
    print(f"Generative-PC GN workflow artifacts: {result}", flush=True)


if __name__ == "__main__":
    main()


__all__ = ["DEFAULT_CONFIG", "main"]

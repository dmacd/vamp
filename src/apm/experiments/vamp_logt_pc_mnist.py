"""Single config-driven entry point for generative-PC MAP routing."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.65")

from apm.experiments.vamp_logt_pc_config import load_config  # noqa: E402


DEFAULT_CONFIG = Path("configs/vamp_logt_pc_mnist/minimal.yaml")


def main() -> None:
    """Run or resume all eligible phases in their fixed order."""
    parser = argparse.ArgumentParser(
        description="Normalized generative-PC MAP routing for true-LogT VAMP on MNIST"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    arguments = parser.parse_args()
    from apm.experiments.vamp_logt_pc_workflow import run_workflow

    result = run_workflow(load_config(arguments.config))
    print(f"Generative-PC workflow artifacts: {result}", flush=True)


if __name__ == "__main__":
    main()


__all__ = ["DEFAULT_CONFIG", "main"]

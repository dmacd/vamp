"""Entry point for the converged Permuted-MNIST integrator ceiling."""

from __future__ import annotations

import argparse
from pathlib import Path

from apm.experiments.vamp_logt_integrator_ceiling_permuted_config import load_config
from apm.experiments.vamp_logt_integrator_ceiling_permuted_workflow import run_workflow


DEFAULT_CONFIG = Path(
    "configs/vamp_logt_integrator_ceiling_permuted_mnist/primary.yaml"
)


def main() -> None:
    """Run or resume the selected converged full-replay phase."""
    parser = argparse.ArgumentParser(
        description="Converged full-replay LogT integrator ceiling on Permuted-MNIST"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase", choices=("smoke", "primary", "all"), default="all")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    result = run_workflow(config, arguments.phase)
    print(f"Permuted integrator-ceiling artifacts: {result}", flush=True)


if __name__ == "__main__":
    main()


__all__ = ["DEFAULT_CONFIG", "main"]

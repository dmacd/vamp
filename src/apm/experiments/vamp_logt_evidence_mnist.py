"""Single config-driven entry point for the LogT NCE/TRE MNIST experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from apm.experiments.vamp_logt_evidence_config import load_config
from apm.experiments.vamp_logt_evidence_workflow import run_workflow


DEFAULT_CONFIG = Path("configs/vamp_logt_evidence_mnist/nce_tre.yaml")


def main() -> None:
    """Run or resume every preregistered phase in its fixed de-risking order."""
    parser = argparse.ArgumentParser(
        description="NCE/TRE evidence routing for true-LogT VAMP on MNIST"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    arguments = parser.parse_args()
    result = run_workflow(load_config(arguments.config))
    print(f"NCE/TRE workflow artifacts: {result}", flush=True)


if __name__ == "__main__":
    main()


__all__ = ["DEFAULT_CONFIG", "main"]

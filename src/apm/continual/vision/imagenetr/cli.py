"""Minimal config-driven command surface for the local ImageNet-R experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
import json

from apm.continual.vision.imagenetr.scheduler import LocalScheduler
from apm.continual.vision.imagenetr.workflow import latest_run_path, run_workflow


DEFAULT_CONFIG = Path("configs/vision/imagenetr/primary.yaml")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ImageNet-R-50 logarithmic VAMP")
    parser.add_argument(
        "command",
        choices=("run", "status", "report"),
        nargs="?",
        default="run",
        help="run the one resolved workflow or inspect its durable outputs",
    )
    parser.add_argument(
        "config",
        type=Path,
        nargs="?",
        default=DEFAULT_CONFIG,
        help="resolved YAML protocol (scientific choices live here, not in CLI flags)",
    )
    return parser


def main() -> None:
    """Dispatch the single workflow or one of its two read-only views."""
    arguments = _parser().parse_args()
    if arguments.command == "run":
        run_workflow(arguments.config)
        return
    _config, run = latest_run_path(arguments.config)
    if arguments.command == "status":
        scheduler = LocalScheduler(
            run / "state" / "scheduler_state.json", run.name
        )
        print(json.dumps(scheduler.summary(), indent=2, sort_keys=True))
        return
    report = run / "reports" / "REPORT.md"
    if not report.is_file():
        raise SystemExit(f"No completed report exists for run {run.name}")
    print(report)


if __name__ == "__main__":
    main()


__all__ = ["main"]

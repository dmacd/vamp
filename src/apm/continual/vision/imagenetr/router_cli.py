"""Dedicated config-driven CLI for the ImageNet-R recursive-router follow-up."""

from __future__ import annotations

import argparse
from pathlib import Path
import json

from apm.continual.vision.imagenetr.router_experiment import latest_router_run
from apm.continual.vision.imagenetr.router_workflow import (
    DEFAULT_ROUTER_CONFIG,
    run_router_workflow,
)
from apm.continual.vision.imagenetr.scheduler import LocalScheduler


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ImageNet-R-50 recursive learned-router oracle recovery"
    )
    parser.add_argument(
        "command",
        choices=("run", "status", "report"),
        nargs="?",
        default="run",
    )
    parser.add_argument(
        "config",
        type=Path,
        nargs="?",
        default=DEFAULT_ROUTER_CONFIG,
        help="resolved YAML protocol; scientific choices are not CLI flags",
    )
    return parser


def main() -> None:
    """Run the workflow or inspect existing durable state without mutation."""
    arguments = _parser().parse_args()
    if arguments.command == "run":
        print(run_router_workflow(arguments.config))
        return
    _config, run = latest_router_run(arguments.config)
    if arguments.command == "status":
        scheduler = LocalScheduler(run / "state" / "scheduler_state.json", run.name)
        workflow_path = run / "state" / "router_workflow_state.json"
        workflow = (
            json.loads(workflow_path.read_text(encoding="utf-8"))
            if workflow_path.is_file()
            else None
        )
        print(
            json.dumps(
                {"scheduler": scheduler.summary(), "workflow": workflow},
                indent=2,
                sort_keys=True,
            )
        )
        return
    report = run / "reports" / "REPORT.md"
    if not report.is_file():
        raise SystemExit(f"No report exists for router run {run.name}")
    print(report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()


__all__ = ["main"]

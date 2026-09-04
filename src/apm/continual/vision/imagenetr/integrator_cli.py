"""Single config-driven CLI for the ImageNet-R prediction-integrator study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apm.continual.vision.imagenetr.integrator_artifacts import latest_integrator_run
from apm.continual.vision.imagenetr.integrator_pdf import render_integrator_pdf
from apm.continual.vision.imagenetr.integrator_reporting import write_integrator_report
from apm.continual.vision.imagenetr.integrator_workflow import (
    DEFAULT_INTEGRATOR_CONFIG,
    run_integrator_workflow,
)
from apm.continual.vision.imagenetr.stage_matched_joint import (
    run_stage_matched_joint_control,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ImageNet-R-50 LogT prediction integrator")
    parser.add_argument(
        "command",
        choices=("run", "stage-matched", "status", "report"),
        nargs="?",
        default="run",
    )
    parser.add_argument(
        "config",
        type=Path,
        nargs="?",
        default=DEFAULT_INTEGRATOR_CONFIG,
        help="resolved YAML protocol; scientific choices are not CLI flags",
    )
    return parser


def main() -> None:
    """Run the experiment or provide one of its read-only views."""
    arguments = _parser().parse_args()
    if arguments.command == "run":
        print(run_integrator_workflow(arguments.config))
        return
    if arguments.command == "stage-matched":
        summary = run_stage_matched_joint_control(arguments.config)
        _config, run = latest_integrator_run(arguments.config)
        report = write_integrator_report(run)
        pdf = render_integrator_pdf(report)
        print(
            json.dumps(
                {"html_report": str(report), "pdf_report": str(pdf), "summary": str(summary)},
                indent=2,
            )
        )
        return
    _config, run = latest_integrator_run(arguments.config)
    if arguments.command == "status":
        state = run / "state" / "workflow.json"
        print(
            json.dumps(
                json.loads(state.read_text(encoding="utf-8")) if state.is_file() else {"phase": "NOT_STARTED"},
                indent=2,
                sort_keys=True,
            )
        )
        return
    report = run / "reports" / "REPORT.md"
    if not report.is_file():
        raise SystemExit(f"No report exists for integrator run {run.name}")
    print(report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()


__all__ = ["main"]

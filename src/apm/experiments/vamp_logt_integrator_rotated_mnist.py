"""Entry point for direct LogT prediction integration on Rotated-MNIST."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from apm.continual.artifacts import load_canonical_json
from apm.experiments.vamp_logt_integrator_rotated_config import load_config
from apm.experiments.vamp_logt_integrator_rotated_reporting import rerender_results
from apm.experiments.vamp_logt_integrator_rotated_workflow import run_workflow


DEFAULT_CONFIG = Path("configs/vamp_logt_integrator_rotated_mnist/primary.yaml")
CEILING_ARTIFACT_DIRECTORY = "vamp-logt-integrator-ceiling-rotated-mnist"


def _matching_ceiling_run_root(
    parent_run_root: Path,
    parent_artifact_root: Path,
) -> Path | None:
    """Return the latest ceiling only when it declares this exact parent run."""
    latest_path = (
        parent_artifact_root.parent
        / CEILING_ARTIFACT_DIRECTORY
        / "LATEST_RUN.json"
    )
    if not latest_path.is_file():
        return None
    latest = load_canonical_json(latest_path)
    candidate = Path(str(latest.get("run_root", "")))
    protocol_path = candidate / "protocol.json"
    if not protocol_path.is_file():
        raise ValueError("latest converged-integrator ceiling lacks protocol.json")
    protocol = load_canonical_json(protocol_path)
    ceiling_config = protocol.get("config")
    parent = (
        ceiling_config.get("parent_integrator")
        if isinstance(ceiling_config, Mapping)
        else None
    )
    if not isinstance(parent, Mapping) or parent.get("run_id") != parent_run_root.name:
        return None
    return candidate


def main() -> None:
    """Run or resume the selected sealed prediction-integrator phase."""
    parser = argparse.ArgumentParser(
        description="Direct LogT prediction integration on VAMP-AF Rotated-MNIST"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase", choices=("smoke", "primary", "all"), default="all")
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="regenerate derived figures and standalone HTML without running training",
    )
    parser.add_argument(
        "--ceiling-run-root",
        type=Path,
        help=(
            "authenticated converged full-replay run to overlay; render-only "
            "auto-discovers the latest matching run when omitted"
        ),
    )
    arguments = parser.parse_args()
    if arguments.ceiling_run_root is not None and not arguments.render_only:
        parser.error("--ceiling-run-root requires --render-only")
    config = load_config(arguments.config)
    if arguments.render_only:
        parent_run_root = config.artifact_root / "runs" / config.config_hash
        ceiling_run_root = arguments.ceiling_run_root or _matching_ceiling_run_root(
            parent_run_root,
            config.artifact_root,
        )
        if ceiling_run_root is not None:
            print(
                "Overlaying certified converged full-replay integrator rows from "
                f"{ceiling_run_root}",
                flush=True,
            )
        result = rerender_results(
            parent_run_root,
            config,
            ceiling_run_root,
        )
    else:
        result = run_workflow(config, arguments.phase)
    print(f"Rotated prediction-integrator artifacts: {result}", flush=True)


if __name__ == "__main__":
    main()


__all__ = ["DEFAULT_CONFIG", "_matching_ceiling_run_root", "main"]

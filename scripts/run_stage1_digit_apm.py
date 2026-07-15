"""Run Stage 1 dense-delta APM on one MNIST digit task at a time."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from apm.data import load_mnist
from apm.data.mnist import make_digit_mnist_stream
from apm.training import EnergyConvergenceSchedule, FixedEpochSchedule, TrainingSchedule
from run_stage1_apm import run_stage1_benchmark

RUN_DIR = Path("results") / "stage1_apm" / "digit_mnist_dense_delta"
DIGITS = tuple(range(10))
TRAIN_EXAMPLES_PER_DIGIT = 2_000
TEST_EXAMPLES_PER_DIGIT = 400
REPLAY_EXAMPLES_PER_TASK = 200
TASK_EPOCHS = 5
PARENT_PROBE_COUNT = 512
REPORT_CANVAS_COUNT = 32
MIN_EPOCHS = 10
MAX_EPOCHS = 100
ENERGY_RELATIVE_DELTA = 1e-3
ENERGY_PATIENCE = 5
ENERGY_PROBE_COUNT = 1_024


def main() -> None:
    """Run the digit-incremental Stage 1 benchmark and write report artifacts."""
    args = _parse_args()
    training_schedule = _training_schedule(args)
    run_dir = _resolved_run_dir(args)
    tasks = make_digit_mnist_stream(
        load_mnist(allow_download=True),
        digits=tuple(args.digits),
        train_count=args.train_count,
        test_count=args.test_count,
    )
    run_stage1_benchmark(
        run_dir,
        tasks,
        {
            "kind": "digit_mnist",
            "digits": tuple(args.digits),
            "train_examples_per_digit": args.train_count,
            "test_examples_per_digit": args.test_count,
            "replay_examples_per_task": args.replay_count,
        },
        f"Stage 1 DigitMNIST Dense-Delta APM ({args.model_kind}, {args.training_mode})",
        training_schedule=training_schedule,
        replay_examples_per_task=args.replay_count,
        parent_probe_count=args.parent_probe_count,
        report_canvas_count=args.report_canvas_count,
        model_kind=args.model_kind,
        show_progress=not args.no_progress,
        include_baselines=args.include_baselines,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-kind", choices=("vae", "fabricpc"), default="vae")
    parser.add_argument("--output-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--digits", type=int, nargs="+", default=DIGITS)
    parser.add_argument("--train-count", type=int, default=TRAIN_EXAMPLES_PER_DIGIT)
    parser.add_argument("--test-count", type=int, default=TEST_EXAMPLES_PER_DIGIT)
    parser.add_argument("--replay-count", type=int, default=REPLAY_EXAMPLES_PER_TASK)
    parser.add_argument("--epochs", type=int, default=TASK_EPOCHS)
    parser.add_argument(
        "--training-mode",
        choices=("fixed", "energy-convergence"),
        default="fixed",
        help="Use a fixed epoch count or stop when digit-only energy converges.",
    )
    parser.add_argument("--min-epochs", type=int, default=MIN_EPOCHS)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--energy-relative-delta", type=float, default=ENERGY_RELATIVE_DELTA)
    parser.add_argument("--energy-patience", type=int, default=ENERGY_PATIENCE)
    parser.add_argument("--energy-probe-count", type=int, default=ENERGY_PROBE_COUNT)
    parser.add_argument("--parent-probe-count", type=int, default=PARENT_PROBE_COUNT)
    parser.add_argument("--report-canvas-count", type=int, default=REPORT_CANVAS_COUNT)
    parser.add_argument("--include-baselines", action="store_true", help="Also run online_sgd and replay_sgd baselines.")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def _training_schedule(args: argparse.Namespace) -> TrainingSchedule:
    if args.training_mode == "fixed":
        return FixedEpochSchedule(args.epochs)
    return EnergyConvergenceSchedule(
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        relative_delta=args.energy_relative_delta,
        patience=args.energy_patience,
        probe_count=args.energy_probe_count,
    )


def _resolved_run_dir(args: argparse.Namespace) -> Path:
    if args.output_dir != RUN_DIR:
        return args.output_dir
    suffix = "" if args.model_kind == "vae" else f"_{args.model_kind}"
    if args.training_mode == "energy-convergence":
        suffix += "_energy_converged"
    return RUN_DIR.parent / f"{RUN_DIR.name}{suffix}"


if __name__ == "__main__":
    main()

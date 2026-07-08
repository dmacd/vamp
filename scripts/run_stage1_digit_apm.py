"""Run Stage 1 dense-delta APM on one MNIST digit task at a time."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from apm.data import load_mnist
from apm.data.mnist import make_digit_mnist_stream
from run_stage1_apm import run_stage1_benchmark

RUN_DIR = Path("results") / "stage1_apm" / "digit_mnist_dense_delta"
DIGITS = tuple(range(10))
TRAIN_EXAMPLES_PER_DIGIT = 2_000
TEST_EXAMPLES_PER_DIGIT = 400
REPLAY_EXAMPLES_PER_TASK = 200
TASK_EPOCHS = 5
PARENT_PROBE_COUNT = 512
REPORT_CANVAS_COUNT = 32


def main() -> None:
    """Run the digit-incremental Stage 1 benchmark and write report artifacts."""
    args = _parse_args()
    run_dir = args.output_dir
    if run_dir == RUN_DIR and args.model_kind != "vae":
        run_dir = RUN_DIR.parent / f"{RUN_DIR.name}_{args.model_kind}"
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
        f"Stage 1 DigitMNIST Dense-Delta APM ({args.model_kind})",
        task_epochs=args.epochs,
        replay_examples_per_task=args.replay_count,
        parent_probe_count=args.parent_probe_count,
        report_canvas_count=args.report_canvas_count,
        model_kind=args.model_kind,
        show_progress=not args.no_progress,
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
    parser.add_argument("--parent-probe-count", type=int, default=PARENT_PROBE_COUNT)
    parser.add_argument("--report-canvas-count", type=int, default=REPORT_CANVAS_COUNT)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()

"""Run Stage 1 dense-delta APM on one MNIST digit task at a time."""

from __future__ import annotations

from pathlib import Path

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
    tasks = make_digit_mnist_stream(
        load_mnist(allow_download=True),
        digits=DIGITS,
        train_count=TRAIN_EXAMPLES_PER_DIGIT,
        test_count=TEST_EXAMPLES_PER_DIGIT,
    )
    run_stage1_benchmark(
        RUN_DIR,
        tasks,
        {
            "kind": "digit_mnist",
            "digits": DIGITS,
            "train_examples_per_digit": TRAIN_EXAMPLES_PER_DIGIT,
            "test_examples_per_digit": TEST_EXAMPLES_PER_DIGIT,
            "replay_examples_per_task": REPLAY_EXAMPLES_PER_TASK,
        },
        "Stage 1 DigitMNIST Dense-Delta APM",
        task_epochs=TASK_EPOCHS,
        replay_examples_per_task=REPLAY_EXAMPLES_PER_TASK,
        parent_probe_count=PARENT_PROBE_COUNT,
        report_canvas_count=REPORT_CANVAS_COUNT,
    )


if __name__ == "__main__":
    main()

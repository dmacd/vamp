from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import run_stage1_digit_apm as digit_runner
import run_stage1_apm as stage_runner

from apm.training import EnergyConvergenceSchedule, FixedEpochSchedule


def test_digit_runner_builds_fixed_schedule() -> None:
    schedule = digit_runner._training_schedule(_args(training_mode="fixed", epochs=7))

    assert schedule == FixedEpochSchedule(7)


def test_digit_runner_builds_convergence_schedule_and_separate_output_dir() -> None:
    args = _args(training_mode="energy-convergence", model_kind="fabricpc")

    schedule = digit_runner._training_schedule(args)

    assert schedule == EnergyConvergenceSchedule(
        min_epochs=10,
        max_epochs=100,
        relative_delta=1e-3,
        patience=5,
        probe_count=1_024,
    )
    assert digit_runner._resolved_run_dir(args).name == "digit_mnist_dense_delta_fabricpc_energy_converged"


def test_digit_runner_preserves_explicit_output_dir() -> None:
    args = _args(output_dir=Path("results/custom"), training_mode="energy-convergence")

    assert digit_runner._resolved_run_dir(args) == Path("results/custom")


def test_convergence_work_estimate_includes_training_and_probe_batches() -> None:
    backend = SimpleNamespace(train_config=SimpleNamespace(batch_size=4, eval_batch_size=2))
    schedule = EnergyConvergenceSchedule(
        min_epochs=2,
        max_epochs=3,
        relative_delta=1e-3,
        patience=1,
        probe_count=5,
    )

    assert stage_runner._training_work_units(backend, 10, schedule) == 18


def _args(**overrides: object) -> argparse.Namespace:
    values = {
        "output_dir": digit_runner.RUN_DIR,
        "model_kind": "vae",
        "training_mode": "fixed",
        "epochs": 5,
        "min_epochs": 10,
        "max_epochs": 100,
        "energy_relative_delta": 1e-3,
        "energy_patience": 5,
        "energy_probe_count": 1_024,
    }
    values.update(overrides)
    return argparse.Namespace(**values)

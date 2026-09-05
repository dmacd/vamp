from __future__ import annotations

from pathlib import Path

import pytest

from apm.continual.vision.imagenetr.macro_convergence_config import (
    load_macro_convergence_config,
)
from apm.continual.vision.imagenetr.macro_convergence_training import (
    ConvergenceFit,
    MacroConvergenceCell,
    convergence_learning_rate,
)


CONFIG = Path("configs/vision/imagenetr/logt_macro_token_convergence_v9.yaml")


def test_convergence_config_expands_the_declared_nine_cell_matrix() -> None:
    config = load_macro_convergence_config(CONFIG)
    assert config.stage == 31
    assert config.matrix == tuple(
        (batch_size, learning_rate)
        for batch_size in (64, 128, 512)
        for learning_rate in (0.00003, 0.0001, 0.0003)
    )
    assert config.source_macro_run_hash == (
        "323104b00589e606b67a4e084c832a99166d161b68e1ff3fe19966793b4a18b2"
    )
    assert config.shuffle_population_hash == (
        "a3c79696a30925a01471ae2b8d35fb3f4e87bf5f959c17b6b3a10255e0ba3e21"
    )


def test_warmup_cosine_schedule_reaches_peak_then_decays_to_floor() -> None:
    cell = MacroConvergenceCell("warmup_cosine", 128, 0.0001, 50, 1993)
    rates = tuple(
        convergence_learning_rate(cell, step, 100, 0.05, 0.01)
        for step in range(100)
    )
    assert rates[0] == pytest.approx(0.00002)
    assert rates[4] == pytest.approx(0.0001)
    assert rates[5] == pytest.approx(0.0001)
    assert rates[-1] == pytest.approx(0.000001)
    assert all(left >= right for left, right in zip(rates[4:-1], rates[5:], strict=True))


def test_legacy_schedule_is_constant_and_cells_have_distinct_names() -> None:
    cells = tuple(
        MacroConvergenceCell("warmup_cosine", batch, rate, 50, 1993)
        for batch in (64, 128, 512)
        for rate in (0.00003, 0.0001, 0.0003)
    )
    assert len({cell.condition for cell in cells}) == 9
    legacy = MacroConvergenceCell("legacy_constant", 512, 0.0003, 20, 1993)
    assert convergence_learning_rate(legacy, 19, 20, 0.0, 0.01) == 0.0003


def test_convergence_fit_rejects_a_best_step_after_total_work() -> None:
    with pytest.raises(ValueError, match="incomplete"):
        ConvergenceFit(
            best_epoch=2,
            epochs=20,
            optimizer_steps=40,
            best_optimizer_steps=41,
            image_presentations=100,
            train_nll=0.2,
            train_accuracy=90.0,
            validation_nll=1.0,
            validation_accuracy=75.0,
            peak_vram_bytes=10,
            wall_seconds=1.0,
            history_rows=20,
        )

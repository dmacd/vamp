from __future__ import annotations

from dataclasses import replace
from hashlib import sha256

import pytest

from apm.data.text.tinyworlds_p_semantic.statistics import (
    EmpiricalGap,
    GroupLoss,
    PairedLoss,
    SemanticEpochValidation,
    WorldEmpiricalGap,
    calibration_decision,
    epoch_satisfies_semantic_gap_gate,
    holm_rejections,
    paired_empirical_gap,
    select_best_eligible_epoch,
    summarize_empirical_gaps,
)


def _pairs(world: str, gap: float = 0.10) -> tuple[PairedLoss, ...]:
    return tuple(
        PairedLoss(
            world=world,
            world_loss=GroupLoss(
                sha256(f"{world}-world-{index}".encode()).hexdigest(),
                (1.0 + gap + index / 10_000) * (50 + index),
                50 + index,
            ),
            control_loss=GroupLoss(
                sha256(f"{world}-control-{index}".encode()).hexdigest(),
                (1.0 + index / 10_000) * (48 + index),
                48 + index,
            ),
        )
        for index in range(24)
    )


def _validation(epoch: int, held_in: float, gap: float = 0.10) -> SemanticEpochValidation:
    empirical = EmpiricalGap(gap, gap - 0.01, gap + 0.01, 0.001, 10_000)
    worlds = tuple(
        WorldEmpiricalGap(world, 1.0 + gap, 1.0, empirical)
        for world in "ABCDE"
    )
    return SemanticEpochValidation(
        epoch=epoch,
        held_in_nll=held_in,
        worlds=worlds,
        mean_empirical=empirical,
        allocator_peak_bytes=8 * 1024**3,
    )


def test_paired_bootstrap_and_placebo_are_sha_seeded_and_order_independent() -> None:
    pairs = _pairs("A")
    first = paired_empirical_gap(pairs, "a" * 64, "test", replicates=2_000)
    replay = paired_empirical_gap(tuple(reversed(pairs)), "a" * 64, "test", replicates=2_000)

    assert replay == first
    assert first.observed_gap == pytest.approx(0.1, abs=1e-4)
    assert first.bootstrap_lower > 0.09
    assert first.placebo_probability <= 0.01


def test_mean_empirical_summary_holm_and_semantic_decision() -> None:
    pairs = tuple(_pairs(world) for world in "ABCDE")
    worlds, mean = summarize_empirical_gaps(pairs, "b" * 64, replicates=2_000)

    assert tuple(item.world for item in worlds) == tuple("ABCDE")
    assert mean.observed_gap == pytest.approx(0.1, abs=1e-4)
    assert mean.bootstrap_lower > 0.0
    assert all(holm_rejections(tuple(item.empirical.placebo_probability for item in worlds)))

    passing = _validation(2, 2.0)
    assert epoch_satisfies_semantic_gap_gate(passing)
    assert calibration_decision(_validation(1, 2.1), passing, 12 * 1024**3) == "pass"
    assert calibration_decision(
        _validation(1, 2.01), passing, 12 * 1024**3
    ) == "training_quality_failure"
    failed_gap = replace(
        passing,
        mean_empirical=replace(passing.mean_empirical, observed_gap=0.048),
    )
    assert calibration_decision(
        _validation(1, 2.1), failed_gap, 12 * 1024**3
    ) == "semantic_grid_failure"
    epoch_three = _validation(3, 1.9)
    epoch_four = _validation(4, 1.9)
    assert select_best_eligible_epoch((passing, epoch_four, epoch_three)).epoch == 3


def test_holm_requires_every_ordered_hypothesis_to_pass() -> None:
    assert holm_rejections((0.001, 0.008, 0.011, 0.012, 0.02)) == (
        True,
        True,
        True,
        True,
        True,
    )
    assert holm_rejections((0.001, 0.02, 0.021, 0.022, 0.023)) == (
        True,
        False,
        False,
        False,
        False,
    )

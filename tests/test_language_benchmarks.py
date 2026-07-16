from __future__ import annotations

import math

import numpy as np
import pytest

from apm.continual.language_benchmarks import (
    CANONICAL_BASELINE_MATRIX,
    ROUTER_BASELINE_NAMES,
    STORED_BASELINE_NAMES,
    AddressingOperationCounts,
    deterministic_random_valid_node_indices,
    evaluate_route_results,
    summarize_negative_control,
    time_synchronized_addressing,
    wilson_95_confidence_interval,
)


def test_canonical_baseline_matrix_has_exact_names_categories_and_order() -> None:
    assert STORED_BASELINE_NAMES == (
        "frozen_base",
        "sequential_single_lora",
        "independent_root_lora",
        "vamp_oracle",
    )
    assert ROUTER_BASELINE_NAMES == (
        "vamp_exhaustive",
        "vamp_hopfield",
        "vamp_ebt_uniform",
        "vamp_ebt_hopfield",
        "deterministic_random_node",
    )
    assert tuple(spec.name for spec in CANONICAL_BASELINE_MATRIX) == (
        STORED_BASELINE_NAMES + ROUTER_BASELINE_NAMES
    )
    assert tuple(spec.category for spec in CANONICAL_BASELINE_MATRIX) == (
        ("stored",) * len(STORED_BASELINE_NAMES)
        + ("router",) * len(ROUTER_BASELINE_NAMES)
    )
    assert len({spec.name for spec in CANONICAL_BASELINE_MATRIX}) == 9


def test_deterministic_random_node_is_stable_identity_keyed_and_valid() -> None:
    valid_mask = np.asarray((True, False, True, True, False), dtype=np.bool_)
    identities = tuple(f"story-{index}" for index in range(12))

    selected = deterministic_random_valid_node_indices(
        valid_mask,
        seed=17,
        example_identities=identities,
    )
    repeated = deterministic_random_valid_node_indices(
        valid_mask,
        seed=17,
        example_identities=identities,
    )
    reversed_order = deterministic_random_valid_node_indices(
        valid_mask,
        seed=17,
        example_identities=tuple(reversed(identities)),
    )
    other_seed = deterministic_random_valid_node_indices(
        valid_mask,
        seed=18,
        example_identities=identities,
    )

    np.testing.assert_array_equal(selected, repeated)
    np.testing.assert_array_equal(selected, reversed_order[::-1])
    assert np.all(valid_mask[selected])
    assert not np.array_equal(selected, other_seed)
    assert not selected.flags.writeable


def test_route_evaluation_produces_suffix_regret_uncertainty_and_confusion() -> None:
    suffix_nll = np.asarray(
        (
            (1.0, 2.0, 0.5, np.inf),
            (1.0, 0.8, 1.4, np.inf),
        ),
        dtype=np.float32,
    )
    probabilities = np.asarray(
        ((0.2, 0.5, 0.3, 0.0), (0.7, 0.2, 0.1, 0.0)),
        dtype=np.float32,
    )

    rows = evaluate_route_results(
        selected_indices=np.asarray((1, 0), dtype=np.int32),
        suffix_nll_by_node=suffix_nll,
        valid_node_mask=np.asarray((True, True, True, False), dtype=np.bool_),
        task_oracle_indices=np.asarray((0, 1), dtype=np.int32),
        node_probabilities=probabilities,
        top_k_indices=np.asarray(((1, 0), (0, 2)), dtype=np.int32),
    )

    assert rows[0].selected_suffix_nll == pytest.approx(2.0)
    assert rows[0].task_oracle_regret == pytest.approx(1.0)
    assert rows[0].best_node_index == 2
    assert rows[0].best_node_regret == pytest.approx(1.5)
    assert rows[0].task_oracle_correct is False
    assert rows[0].top_k_task_oracle_hit is True
    assert rows[0].top_two_probability_margin == pytest.approx(0.2)
    assert rows[0].address_entropy == pytest.approx(
        -sum(probability * math.log(probability) for probability in (0.2, 0.5, 0.3))
    )
    assert rows[0].confusion_pair == (0, 1)
    assert rows[1].task_oracle_regret == pytest.approx(0.2)
    assert rows[1].best_node_regret == pytest.approx(0.2)
    assert rows[1].top_k_task_oracle_hit is False
    assert rows[1].confusion_pair == (1, 0)


class _PendingResult:
    def __init__(self) -> None:
        self.block_count = 0

    def block_until_ready(self) -> _PendingResult:
        self.block_count += 1
        return self


class _FakeClock:
    def __init__(self, readings: tuple[float, ...]) -> None:
        self._readings = iter(readings)

    def __call__(self) -> float:
        return next(self._readings)


def test_address_timing_synchronizes_cold_and_each_warm_result() -> None:
    pending = _PendingResult()
    invocation_count = 0

    def address() -> tuple[dict[str, _PendingResult], int]:
        nonlocal invocation_count
        invocation_count += 1
        return ({"address": pending}, invocation_count)

    operations = AddressingOperationCounts(
        prefix_tokens=96,
        candidates_available=3,
        candidates_scored=3,
        full_model_forward_equivalent_tokens=288,
        base_forwards=3,
        edge_evaluations=6,
        hopfield_dot_products=0,
        ebt_steps=0,
        ebt_mask_size=0,
        selected_execution_cost=2,
    )
    timing = time_synchronized_addressing(
        address,
        operations,
        batch_size=6,
        warm_repetitions=3,
        clock=_FakeClock((0.0, 5.0, 10.0, 12.0, 20.0, 23.0, 30.0, 34.0)),
    )

    assert invocation_count == 4
    assert pending.block_count == 4
    assert timing.cold_compile_seconds == pytest.approx(5.0)
    assert timing.warm_latency_samples_seconds == pytest.approx((2.0, 3.0, 4.0))
    assert timing.warm_latency_seconds == pytest.approx(3.0)
    assert timing.warm_throughput_examples_per_second == pytest.approx(2.0)
    assert timing.operations is operations


def test_operation_counts_distinguish_logical_candidates_and_ebt_mask() -> None:
    counts = AddressingOperationCounts(
        prefix_tokens=64,
        candidates_available=3,
        candidates_scored=5,
        full_model_forward_equivalent_tokens=1280,
        base_forwards=20,
        edge_evaluations=60,
        hopfield_dot_products=5,
        ebt_steps=20,
        ebt_mask_size=2,
        selected_execution_cost=3,
    )

    assert counts.candidates_available == 3
    assert counts.candidates_scored == 5
    assert counts.ebt_mask_size == 2
    with pytest.raises(ValueError, match="both be zero or positive"):
        AddressingOperationCounts(
            prefix_tokens=64,
            candidates_available=5,
            candidates_scored=5,
            full_model_forward_equivalent_tokens=0,
            base_forwards=0,
            edge_evaluations=0,
            hopfield_dot_products=0,
            ebt_steps=20,
            ebt_mask_size=0,
            selected_execution_cost=0,
        )


@pytest.mark.parametrize(
    ("successes", "trials", "expected_lower", "expected_upper"),
    (
        (50, 100, 0.4038315, 0.5961685),
        (0, 10, 0.0, 0.2775328),
        (10, 10, 0.7224672, 1.0),
    ),
)
def test_wilson_95_interval_matches_binomial_reference_values(
    successes: int,
    trials: int,
    expected_lower: float,
    expected_upper: float,
) -> None:
    interval = wilson_95_confidence_interval(successes, trials)

    assert interval.observed_rate == pytest.approx(successes / trials)
    assert interval.lower == pytest.approx(expected_lower, abs=1e-7)
    assert interval.upper == pytest.approx(expected_upper, abs=1e-7)


def test_negative_control_summary_flags_only_material_above_chance_accuracy() -> None:
    oracle = np.tile(np.arange(1, 5, dtype=np.int32), 25)
    at_chance = np.ones((100,), dtype=np.int32)
    materially_high = np.roll(oracle, 1)
    materially_high[:50] = oracle[:50]

    chance_summary = summarize_negative_control(at_chance, oracle, 4)
    high_summary = summarize_negative_control(materially_high, oracle, 4)

    assert chance_summary.correct_count == 25
    assert chance_summary.observed_accuracy == pytest.approx(0.25)
    assert chance_summary.chance_accuracy == pytest.approx(0.25)
    assert chance_summary.chance_rate_in_interval is True
    assert chance_summary.leakage_audit_required is False
    assert high_summary.correct_count == 50
    assert high_summary.confidence_interval.lower > 0.25
    assert high_summary.chance_rate_in_interval is False
    assert high_summary.leakage_audit_required is True


def test_negative_control_summary_counts_root_routes_as_incorrect() -> None:
    summary = summarize_negative_control(
        np.asarray((0, 1, 0, 2), dtype=np.int32),
        np.asarray((1, 1, 2, 2), dtype=np.int32),
        2,
    )

    assert summary.correct_count == 2
    assert summary.observed_accuracy == 0.5
    assert summary.chance_accuracy == 0.5


@pytest.mark.parametrize(
    ("successes", "trials"),
    ((-1, 10), (11, 10), (0, 0), (True, 10)),
)
def test_wilson_95_interval_rejects_invalid_counts(
    successes: int,
    trials: int,
) -> None:
    with pytest.raises(ValueError, match="0 <= successes <= trials"):
        wilson_95_confidence_interval(successes, trials)

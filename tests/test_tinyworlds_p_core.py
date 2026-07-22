from __future__ import annotations

from dataclasses import replace
from hashlib import sha256

import pytest

from apm.data.text.tinyworlds_p import (
    AllocationGroup,
    BASE_TRAINING_PRESET,
    EpochValidation,
    PartitionPreset,
    WorldGap,
    allocate_stratified_splits,
    balance_word_buckets,
    bucket_word_lookup,
    calibration_grid_decision,
    cosine_learning_rate,
    fallback_partition_preset,
    normalize_story_identity,
    normalized_story_sha256,
    recover_released_recipe,
    require_component_visibility,
    select_best_eligible_epoch,
    select_matched_control,
    select_world_cells,
    summarize_cells,
)


_IDENTITY = "0" * 64


def _group(
    name: str,
    *,
    noun: str = "cat",
    verb: str = "help",
    adjective: str = "kind",
    noun_bucket: int = 0,
    verb_bucket: int = 0,
    adjective_bucket: int = 0,
    source: str = "GPT-4",
    feature: str = "none",
    active_tokens: int = 100,
    canonical_tokens: int = 80,
) -> AllocationGroup:
    return AllocationGroup(
        normalized_sha256=sha256(name.encode()).hexdigest(),
        active_token_count=active_tokens,
        canonical_token_count=canonical_tokens,
        noun=noun,
        verb=verb,
        adjective=adjective,
        noun_bucket=noun_bucket,
        verb_bucket=verb_bucket,
        adjective_bucket=adjective_bucket,
        source=source,
        feature_signature=feature,
    )


def _validation(epoch: int, held_in_nll: float, gaps: tuple[float, ...]) -> EpochValidation:
    return EpochValidation(
        epoch=epoch,
        held_in_nll=held_in_nll,
        world_gaps=tuple(
            WorldGap(world, 2.0 + gap, 2.0)
            for world, gap in zip(("A", "B", "C", "D", "E"), gaps, strict=True)
        ),
        allocator_peak_bytes=8 * 1024**3,
    )


def test_normalization_collapses_quote_and_whitespace_variants() -> None:
    left = "  The CAT said, “I’m here.”\n"
    right = 'the cat said, "i\'m   here."'

    assert normalize_story_identity(left) == normalize_story_identity(right)
    assert normalized_story_sha256(left) == normalized_story_sha256(right)


def test_role_recovery_requires_three_explicit_unique_labels() -> None:
    recovered = recover_released_recipe(
        "Use the verb ‘HELP’, the noun \"Cat\", and adjective 'Kind'.",
        ("help", "cat", "kind"),
        (" dialogue ",),
    )

    assert recovered is not None
    assert recovered.roles == ("cat", "help", "kind")
    assert recovered.features == ("dialogue",)
    assert (
        recover_released_recipe(
            'Use the verb "win", noun "win", and adjective "kind".',
            ("win", "win", "kind"),
        )
        is None
    )
    assert (
        recover_released_recipe(
            'Use the verb "help", noun "cat", noun "robot", and adjective "kind".',
            ("help", "cat", "kind"),
        )
        is None
    )


def test_balanced_buckets_topology_visibility_and_splits_are_deterministic() -> None:
    masses = {f"word-{index}": 10 + index for index in range(9)}
    first = balance_word_buckets(masses, "noun", 3, _IDENTITY)
    second = balance_word_buckets(dict(reversed(tuple(masses.items()))), "noun", 3, _IDENTITY)

    assert second == first
    assert set(bucket_word_lookup(first)) == set(masses)

    groups = tuple(
        _group(
            f"cell-{noun_bucket}-{verb_bucket}-{index}",
            noun=f"noun-{noun_bucket}",
            verb=f"verb-{verb_bucket}",
            adjective=f"adjective-{index % 3}",
            noun_bucket=noun_bucket,
            verb_bucket=verb_bucket,
            adjective_bucket=index % 3,
            source=f"source-{index % 2}",
            feature=f"feature-{index % 2}",
        )
        for noun_bucket in range(3)
        for verb_bucket in range(3)
        for index in range(12)
    )
    cells = select_world_cells(summarize_cells(groups), 3, _IDENTITY)

    assert tuple(cell.label for cell in cells) == ("A", "B", "C", "D", "E")
    assert cells[0].verb_bucket == cells[1].verb_bucket
    assert cells[1].noun_bucket == cells[2].noun_bucket
    assert cells[2].verb_bucket == cells[3].verb_bucket
    assert cells[3].noun_bucket == cells[0].noun_bucket
    assert cells[4].noun_bucket not in {cells[0].noun_bucket, cells[1].noun_bucket}
    assert cells[4].verb_bucket not in {cells[0].verb_bucket, cells[2].verb_bucket}
    visibility = require_component_visibility(groups, cells, 1)
    assert visibility

    assignments = allocate_stratified_splits(
        groups,
        (80, 10, 10),
        _IDENTITY,
        "fixture",
    )
    replayed = allocate_stratified_splits(
        tuple(reversed(groups)),
        (80, 10, 10),
        _IDENTITY,
        "fixture",
    )
    assert replayed == assignments
    assert set(assignments.values()) == {"train", "validation", "test"}


def test_matched_control_balances_marginals_when_joint_strata_are_absent() -> None:
    targets = tuple(
        [
            _group(
                f"target-left-{index}",
                source="s0",
                feature="f0",
                adjective_bucket=0,
                canonical_tokens=50,
            )
            for index in range(10)
        ]
        + [
            _group(
                f"target-right-{index}",
                source="s1",
                feature="f1",
                adjective_bucket=1,
                canonical_tokens=100,
            )
            for index in range(10)
        ]
    )
    row_candidates = tuple(
        [
            _group(
                f"row-left-{index}",
                source="s0",
                feature="f1",
                adjective_bucket=0,
                canonical_tokens=50,
            )
            for index in range(10)
        ]
        + [
            _group(
                f"row-right-{index}",
                source="s1",
                feature="f0",
                adjective_bucket=1,
                canonical_tokens=100,
            )
            for index in range(10)
        ]
    )
    column_candidates = tuple(
        replace(group, normalized_sha256=sha256(f"column-{index}".encode()).hexdigest())
        for index, group in enumerate(row_candidates)
    )
    target_strata = {item.full_stratum for item in targets}
    assert not any(
        item.full_stratum in target_strata
        for item in (*row_candidates, *column_candidates)
    )

    selection, diagnostics = select_matched_control(
        targets,
        row_candidates,
        column_candidates,
        "A",
        "validation",
        _IDENTITY,
        replace(
            PartitionPreset(),
            control_source_feature_tolerance=0.01,
            control_adjective_length_tolerance=0.01,
        ),
    )

    assert selection.row_group_count == selection.column_group_count == 10
    assert len(set(selection.group_sha256)) == 20
    assert diagnostics.token_relative_error == 0.0
    assert diagnostics.maximum_source_feature_prevalence_error == 0.0
    assert diagnostics.maximum_adjective_length_prevalence_error == 0.0
    assert diagnostics.mean_length_relative_error == 0.0


def test_schedule_grid_fallback_and_best_epoch_boundaries() -> None:
    config = replace(
        BASE_TRAINING_PRESET,
        maximum_learning_rate=1e-2,
        minimum_learning_rate=1e-3,
        warmup_fraction=0.1,
    )
    assert cosine_learning_rate(0, 100, config) == pytest.approx(1e-3)
    assert cosine_learning_rate(9, 100, config) == pytest.approx(1e-2)
    assert cosine_learning_rate(99, 100, config) == pytest.approx(1e-3)

    epoch_one = _validation(1, 2.1, (0.10,) * 5)
    passing = _validation(2, 2.0, (0.10, 0.11, 0.12, 0.13, 0.14))
    assert calibration_grid_decision(epoch_one, passing, 12 * 1024**3) == "pass"
    assert calibration_grid_decision(
        epoch_one,
        _validation(2, 2.0, (0.04, 0.10, 0.10, 0.10, 0.10)),
        12 * 1024**3,
    ) == "fallback_6x6"
    assert calibration_grid_decision(
        epoch_one,
        _validation(2, 2.0, (0.31,) * 5),
        12 * 1024**3,
    ) == "fallback_10x10"
    assert calibration_grid_decision(
        epoch_one,
        _validation(2, 2.09, (0.10,) * 5),
        12 * 1024**3,
    ) == "training_quality_failure"

    low_gap_preset = fallback_partition_preset(
        "fallback_6x6",
        PartitionPreset(),
    )
    assert low_gap_preset.bucket_count == 6
    assert low_gap_preset.base_split_weights == (94, 3, 3)
    excessive_gap_preset = fallback_partition_preset(
        "fallback_10x10",
        PartitionPreset(),
    )
    assert excessive_gap_preset.bucket_count == 10
    assert excessive_gap_preset.base_split_weights == (96, 2, 2)

    earlier = _validation(3, 1.9, (0.10,) * 5)
    tied_later = _validation(4, 1.9, (0.11,) * 5)
    assert select_best_eligible_epoch((passing, tied_later, earlier)).epoch == 3

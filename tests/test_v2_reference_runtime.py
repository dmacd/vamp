from __future__ import annotations

import numpy as np
import pytest

from apm.data.text.tinyworlds_v2.reference_runtime import (
    NllStory,
    aggregate_story_window_nll,
    build_story_token_windows,
)


class _IntegerTokenizer:
    pad_token_id = 0

    def encode(self, text: str, *, add_eos: bool = False) -> tuple[int, ...]:
        tokens = tuple(int(part) for part in text.split())
        return tokens + ((99,) if add_eos else ())


def test_story_windows_cover_each_target_once_across_boundaries() -> None:
    stories = (
        NllStory("story-a", "1 2 3 4 5 6"),
        NllStory("story-b", "7 8"),
    )
    windows = build_story_token_windows(
        stories,
        _IntegerTokenizer(),
        sequence_length=3,
        pad_token_id=0,
    )

    assert len(windows) == 3
    assert [(window.story_index, window.active_tokens) for window in windows] == [
        (0, 3),
        (0, 3),
        (1, 2),
    ]
    assert windows[0].input_ids == (1, 2, 3)
    assert windows[0].target_ids == (2, 3, 4)
    assert windows[1].input_ids == (4, 5, 6)
    assert windows[1].target_ids == (5, 6, 99)
    assert windows[2].input_ids == (7, 8, 0)
    assert windows[2].target_ids == (8, 99, 0)


def test_window_nll_microbatching_ignores_padded_rows_and_aggregates_stably() -> None:
    stories = (
        NllStory("story-a", "1 2 3 4 5 6"),
        NllStory("story-b", "7 8"),
    )
    windows = build_story_token_windows(
        stories,
        _IntegerTokenizer(),
        sequence_length=3,
        pad_token_id=0,
    )
    observed_shapes: list[tuple[int, int]] = []

    def scorer(
        input_ids: np.ndarray,
        target_ids: np.ndarray,
        attention_mask: np.ndarray,
        loss_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        del input_ids, target_ids, attention_mask
        observed_shapes.append(loss_mask.shape)
        counts = loss_mask.sum(axis=1).astype(np.int64)
        return counts.astype(np.float64) * 2.5, counts

    scores = aggregate_story_window_nll(
        stories,
        windows,
        batch_size=2,
        scorer=scorer,
    )

    assert observed_shapes == [(2, 3), (2, 3)]
    assert [(score.record_id, score.token_count) for score in scores] == [
        ("story-a", 6),
        ("story-b", 2),
    ]
    assert [score.total_nll for score in scores] == [15.0, 5.0]
    assert [score.normalized_nll for score in scores] == [2.5, 2.5]


def test_window_scorer_must_report_exact_mask_counts() -> None:
    stories = (NllStory("story", "1 2"),)
    windows = build_story_token_windows(
        stories,
        _IntegerTokenizer(),
        sequence_length=4,
        pad_token_id=0,
    )

    def bad_scorer(
        input_ids: np.ndarray,
        target_ids: np.ndarray,
        attention_mask: np.ndarray,
        loss_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        del target_ids, attention_mask, loss_mask
        return np.zeros(input_ids.shape[0]), np.zeros(input_ids.shape[0], dtype=np.int64)

    with pytest.raises(ValueError, match="token count"):
        aggregate_story_window_nll(
            stories,
            windows,
            batch_size=2,
            scorer=bad_scorer,
        )

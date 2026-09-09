"""Fast deterministic scheduler tests independent of accelerator runtime."""

from dataclasses import replace
import math

import numpy as np
import pytest

from apm.continual.vision.imagenetr.srt_config import load_srt_config, presentation_budget
from apm.continual.vision.imagenetr.srt_scheduler import (
    DueHeap, IndexedPool, RecallPolicy, ReviewBatch, ReviewScheduler, ReviewState,
    merge_heaps, observe_batch, restore_scheduler, review_transition, scheduler_record, select_due,
)


POLICY = RecallPolicy("standard", (.1, .25, .5, .75, .9), .5, 1)


def test_quality_boundaries_and_sm2_reference() -> None:
    assert [POLICY.quality(value) for value in (0, .1, .25, .5, .75, .9, 1)] == [0, 1, 2, 3, 4, 5, 5]
    state = ReviewState(1)
    for clock, interval, ease in ((1, 1, 2.6), (2, 6, 2.7), (8, 17, 2.8)):
        state = review_transition(state, 5, clock, 1)
        assert state.interval == interval and state.due == clock + interval
        assert state.ease == pytest.approx(ease)
    state = review_transition(state, 0, 25, 8)
    assert state.successes == 0 and state.interval == 8
    for _ in range(10):
        state = review_transition(state, 0, 25, 8)
    assert state.ease == 1.3
    with pytest.raises(ValueError):
        POLICY.quality(math.nan)


def test_interval_ceiling_uses_exact_hundredths_without_float_overflow() -> None:
    state = ReviewState(1, exposures=4, ease=1.3, successes=4, interval=10)
    assert review_transition(state, 4, 10, 1).interval == 13
    enormous = replace(state, interval=10 ** 400)
    assert review_transition(enormous, 4, 10 ** 401, 1).interval == 13 * 10 ** 399


def test_persistent_heap_and_pool_do_not_mutate_ancestors() -> None:
    first = DueHeap((9, 1))
    heap = first
    for value in range(15):
        heap = merge_heaps(heap, DueHeap((value, value + 10)))
    ordered = []
    while heap:
        ordered.append(heap.key)
        heap = merge_heaps(heap.left, heap.right)
    assert ordered == sorted(ordered) and first.key == (9, 1) and first.left is None
    pool = IndexedPool()
    for value in range(100):
        pool = pool.add(value)
    drawn, remaining = pool.draw(70, np.random.RandomState(7))
    assert len(set(drawn)) == 70 and len(remaining) == 30 and len(pool) == 100
    assert set(drawn).isdisjoint(remaining.values)
    assert all(remaining.values[position] == image for image, position in remaining.positions.items())


def _introduced(unit: int = 1) -> ReviewScheduler:
    scheduler = ReviewScheduler().arrive(1)
    return observe_batch(scheduler, ReviewBatch((0, 1, 2, 3), 0, 4, True), (.95,) * 4,
                         replace(POLICY, interval_unit=unit), "srt", 1, 0)[0]


def test_empty_due_pools_advance_only_scheduling_clock() -> None:
    scheduler = _introduced(8)
    batch, selected = select_due(scheduler, replace(POLICY, interval_unit=8), 64, 1993, 2)
    assert batch.clock_advance == 7 and selected.clock == 9
    assert len(batch.images) == 4 and batch.current_count == 4
    following, events = observe_batch(selected, batch, (.95,) * 4, replace(POLICY, interval_unit=8), "srt", 2, 4)
    assert all(event["step_gap"] == 1 and event["clock_gap"] == 8 and event["lateness"] == 0 for event in events)
    assert following.clock == 10 and scheduler.clock == 2


def test_pool_transfer_lateness_and_budget_release() -> None:
    scheduler = replace(_introduced(), clock=20).ready().arrive(2)
    assert len(scheduler.historical) == 4 and not len(scheduler.current)
    batch, selected = select_due(scheduler, POLICY, 3, 1993, 2)
    assert batch.historical_count == 3 and batch.current_count == 0
    _, events = observe_batch(selected, batch, (.3,) * 3, POLICY, "srt", 2, 4)
    assert all(event["kind"] == "historical" and event["lateness"] == 18 for event in events)
    assert all(event["stage_gap"] == 1 and event["quality"] == 2 for event in events)


def test_checkpoint_restoration_preserves_future_draws_and_event_identity() -> None:
    scheduler = _introduced().ready()
    restored = restore_scheduler(scheduler_record(scheduler))
    assert scheduler_record(restored) == scheduler_record(scheduler)
    for step in range(2, 30):
        first, before = select_due(scheduler, POLICY, 3, 1993, step)
        second, resumed = select_due(restored, POLICY, 3, 1993, step)
        assert first == second
        confidences = tuple(.95 if image % 2 else .05 for image in first.images)
        scheduler, first_events = observe_batch(before, first, confidences, POLICY, "srt", step, step * 3)
        restored, second_events = observe_batch(resumed, second, confidences, POLICY, "srt", step, step * 3)
        assert first_events == second_events


def test_first_exposure_and_duplicate_guards() -> None:
    scheduler = _introduced()
    with pytest.raises(ValueError, match="initial coverage"):
        observe_batch(scheduler, ReviewBatch((0,), 0, 1, True), (.5,), POLICY, "srt", 2, 4)
    with pytest.raises(ValueError, match="distinct"):
        observe_batch(scheduler, ReviewBatch((0, 0), 0, 2, False), (.5, .5), POLICY, "srt", 2, 4)
    uniform, events = observe_batch(ReviewScheduler().arrive(1), ReviewBatch((0,), 0, 1, True), (.5,), POLICY, "uniform", 1, 0)
    assert uniform.future is None and events[0]["due_after"] is None and events[0]["quality"] is None


def test_finite_config_grid_and_budget_counts() -> None:
    config = load_srt_config()
    assert len(config.policies) == len({policy.name for policy in config.policies}) == 18
    assert presentation_budget(475, 0, 4096) == 1900
    assert presentation_budget(444, 23556, 4096) == 18160
    assert presentation_budget(444, 23556, 1024) == 5872

"""Pure SM-2 transitions and persistent due queues for image-level rehearsal."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from hashlib import sha256
import math
from typing import Literal

import numpy as np
from pyrsistent import PMap, PVector, pmap, pvector


@dataclass(frozen=True, slots=True)
class RecallPolicy:
    """A calibrated quality mapping, old/current mixture, and SM-2 time unit."""

    profile: str
    thresholds: tuple[float, ...]
    historical_fraction: float
    interval_unit: int

    def __post_init__(self) -> None:
        if (
            len(self.thresholds) != 5
            or self.thresholds != tuple(sorted(set(self.thresholds)))
            or not 0 < self.thresholds[0] < self.thresholds[-1] < 1
            or not 0 <= self.historical_fraction <= 1
            or self.interval_unit < 1
        ):
            raise ValueError("invalid SRT recall policy")

    @property
    def name(self) -> str:
        return f"{self.profile}_rho{round(100 * self.historical_fraction):02d}_unit{self.interval_unit}"

    def quality(self, confidence: float) -> int:
        """Map pre-update correct-label probability to an integer in [0, 5]."""
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("recall confidence must be a finite probability")
        return sum(confidence >= boundary for boundary in self.thresholds)


@dataclass(frozen=True, slots=True)
class ReviewState:
    """An image's committed review history, never shared as mutable state."""

    arrival_stage: int
    exposures: int = 0
    historical_reviews: int = 0
    ease: float = 2.5
    successes: int = 0
    interval: int = 0
    due: int = 0
    last_step: int = 0
    last_clock: int = 0
    last_presentation: int = 0
    last_stage: int = 0
    last_quality: int | None = None


def review_transition(
    previous: ReviewState, quality: int, clock: int, unit: int,
) -> ReviewState:
    """Apply equations 8-10 of Atreya et al. using the newly updated ease."""
    if quality not in range(6) or unit < 1 or clock < previous.last_clock:
        raise ValueError("invalid SM-2 transition")
    distance = 5 - quality
    # Every SM-2 increment is an exact hundredth. Integer arithmetic avoids
    # ceil errors at integer boundaries and overflow for very long intervals.
    ease_hundredths = max(130, round(previous.ease * 100) + 10 - distance * (8 + 2 * distance))
    ease = ease_hundredths / 100
    successes = previous.successes + 1 if quality >= 3 else 0
    interval = (
        unit if successes <= 1 else 6 * unit if successes == 2
        else (previous.interval * ease_hundredths + 99) // 100
    )
    return replace(previous, ease=ease, successes=successes, interval=interval,
                   due=clock + interval, last_quality=quality)


@dataclass(frozen=True, slots=True)
class DueHeap:
    """A persistent leftist min-heap with logarithmic insertion and removal."""

    key: tuple[int, int]
    left: DueHeap | None = None
    right: DueHeap | None = None
    rank: int = 1


def merge_heaps(first: DueHeap | None, second: DueHeap | None) -> DueHeap | None:
    """Merge two immutable due heaps while sharing their untouched branches."""
    if first is None or second is None:
        return first if second is None else second
    if first.key > second.key:
        first, second = second, first
    left, right = first.left, merge_heaps(first.right, second)
    if (left.rank if left else 0) < (right.rank if right else 0):
        left, right = right, left
    return DueHeap(first.key, left, right, 1 + (right.rank if right else 0))


@dataclass(frozen=True, slots=True)
class IndexedPool:
    """Persistent indexed set supporting unbiased draws without history scans."""

    values: PVector = field(default_factory=pvector)
    positions: PMap = field(default_factory=pmap)

    def __len__(self) -> int:
        return len(self.values)

    def add(self, image: int) -> IndexedPool:
        """Return a pool containing a previously absent image."""
        if image in self.positions:
            raise ValueError("image is already present in this due pool")
        return IndexedPool(self.values.append(image), self.positions.set(image, len(self)))

    def remove(self, image: int) -> IndexedPool:
        """Remove one image by swapping the last entry into its position."""
        position, last = self.positions[image], self.values[-1]
        values = self.values.set(position, last).delete(len(self) - 1)
        positions = self.positions.set(last, position).remove(image)
        return IndexedPool(values, positions)

    def draw(self, count: int, generator: np.random.RandomState) -> tuple[tuple[int, ...], IndexedPool]:
        """Draw uniformly without replacement using sparse partial shuffling."""
        if not 0 <= count <= len(self):
            raise ValueError("due draw exceeds available images")
        pool, selected = self, ()
        for _ in range(count):
            image = pool.values[int(generator.randint(len(pool)))]
            selected += (image,)
            pool = pool.remove(image)
        return selected, pool


@dataclass(frozen=True, slots=True)
class ReviewScheduler:
    """Committed per-image state and distinct old/current due pools."""

    states: PMap = field(default_factory=pmap)
    future: DueHeap | None = None
    historical: IndexedPool = field(default_factory=IndexedPool)
    current: IndexedPool = field(default_factory=IndexedPool)
    stage: int = 0
    clock: int = 1

    def arrive(self, stage: int) -> ReviewScheduler:
        """Move the preceding task's ready images into the historical pool."""
        if stage != self.stage + 1:
            raise ValueError("SRT stages must arrive consecutively")
        historical = self.historical
        for image in self.current.values:
            historical = historical.add(image)
        return replace(self, stage=stage, historical=historical, current=IndexedPool())

    def ready(self) -> ReviewScheduler:
        """Release only newly due entries, without scanning retained examples."""
        future, historical, current = self.future, self.historical, self.current
        while future is not None and future.key[0] <= self.clock:
            image = future.key[1]
            if self.states[image].arrival_stage < self.stage:
                historical = historical.add(image)
            else:
                current = current.add(image)
            future = merge_heaps(future.left, future.right)
        return replace(self, future=future, historical=historical, current=current)


@dataclass(frozen=True, slots=True)
class ReviewBatch:
    """An already-selected batch; loaders cannot choose future training work."""

    images: tuple[int, ...]
    historical_count: int
    current_count: int
    introduction: bool
    clock_advance: int = 0
    historical_due: int = 0
    current_due: int = 0


def step_generator(seed: int, stage: int, step: int, method: str) -> np.random.RandomState:
    """Use a counter-derived RNG independent of worker order or resume timing."""
    value = f"imagenetr50-srt-draw-v1:{seed}:{stage}:{step}:{method}"
    return np.random.RandomState(int(sha256(value.encode()).hexdigest()[:8], 16))


def uniform_indices(length: int, count: int, generator: np.random.RandomState) -> tuple[int, ...]:
    """Sample without replacement in work proportional to the requested count."""
    if not 0 <= count <= length:
        raise ValueError("uniform draw exceeds its arrived population")
    swaps, selected = pmap(), ()
    for offset in range(count):
        remaining = length - offset
        position = int(generator.randint(remaining))
        selected += (swaps.get(position, position),)
        swaps = swaps.set(position, swaps.get(remaining - 1, remaining - 1))
    return selected


def select_due(
    scheduler: ReviewScheduler, policy: RecallPolicy, limit: int, seed: int, step: int,
) -> tuple[ReviewBatch, ReviewScheduler]:
    """Select a due-only batch, advancing virtual time if both pools are empty."""
    ready = scheduler.ready()
    if not len(ready.current) + len(ready.historical):
        if ready.future is None:
            raise ValueError("no introduced images are available for SRT")
        ready = replace(ready, clock=ready.future.key[0]).ready()
    old_count = min(len(ready.historical), math.floor(policy.historical_fraction * limit))
    new_count = min(len(ready.current), limit - old_count)
    old_count = min(len(ready.historical), limit - new_count)
    generator = step_generator(seed, ready.stage, step, "srt")
    old, historical = ready.historical.draw(old_count, generator)
    new, current = ready.current.draw(new_count, generator)
    batch = ReviewBatch(old + new, old_count, new_count, False,
                        ready.clock - scheduler.clock, len(ready.historical), len(ready.current))
    return batch, replace(ready, historical=historical, current=current)


def observe_batch(
    scheduler: ReviewScheduler, batch: ReviewBatch, confidences: tuple[float, ...],
    policy: RecallPolicy, method: Literal["srt", "uniform"], step: int, presentations: int,
) -> tuple[ReviewScheduler, tuple[dict[str, object], ...]]:
    """Commit retained confidence and timing evidence after a successful update."""
    if len(batch.images) != len(confidences) or len(set(batch.images)) != len(batch.images):
        raise ValueError("a review batch must contain distinct scored images")
    states, future, events = scheduler.states, scheduler.future, ()
    for image, confidence in zip(batch.images, confidences, strict=True):
        previous = states.get(image, ReviewState(scheduler.stage))
        if (previous.exposures == 0) != batch.introduction:
            raise ValueError("only the initial coverage pass may introduce an image")
        historical = previous.arrival_stage < scheduler.stage
        quality = policy.quality(confidence) if method == "srt" else None
        updated = review_transition(previous, quality, scheduler.clock, policy.interval_unit) if method == "srt" else previous
        updated = replace(
            updated, exposures=previous.exposures + 1,
            historical_reviews=previous.historical_reviews + int(historical),
            last_step=step, last_clock=scheduler.clock, last_stage=scheduler.stage,
            last_presentation=presentations,
        )
        states = states.set(image, updated)
        if method == "srt":
            future = merge_heaps(future, DueHeap((updated.due, image)))
        events += ({
            "image": image, "arrival_stage": previous.arrival_stage, "stage": scheduler.stage,
            "exposure": updated.exposures, "kind": "introduction" if batch.introduction else "historical" if historical else "current_review",
            "step": step, "presentations": presentations, "clock": scheduler.clock,
            "confidence": confidence, "quality": quality, "previous_quality": previous.last_quality,
            "step_gap": step - previous.last_step if previous.exposures else None,
            "presentation_gap": presentations - previous.last_presentation if previous.exposures else None,
            "stage_gap": scheduler.stage - previous.last_stage if previous.exposures else None,
            "clock_gap": scheduler.clock - previous.last_clock if previous.exposures else None,
            "due_before": previous.due if method == "srt" and previous.exposures else None,
            "lateness": scheduler.clock - previous.due if method == "srt" and previous.exposures else None,
            **{f"{name}_{when}": getattr(state, name) if method == "srt" else None
               for name in ("ease", "successes", "interval", "due")
               for when, state in (("before", previous), ("after", updated))
               if not (name == "due" and when == "before")},
        },)
    return replace(scheduler, states=states, future=future, clock=scheduler.clock + 1), events


def scheduler_record(scheduler: ReviewScheduler) -> dict[str, object]:
    """Serialize exact indexed-pool and heap state, including deterministic order."""
    heap = pvector()
    pending = (scheduler.future,) if scheduler.future else ()
    while pending:
        node, pending = pending[-1], pending[:-1]
        heap = heap.append(node.key)
        pending += tuple(child for child in (node.left, node.right) if child is not None)
    return {
        "states": {image: asdict(state) for image, state in scheduler.states.items()},
        "future": tuple(sorted(heap)), "historical": tuple(scheduler.historical.values),
        "current": tuple(scheduler.current.values), "stage": scheduler.stage, "clock": scheduler.clock,
    }


def restore_scheduler(record: dict[str, object]) -> ReviewScheduler:
    """Restore states and random-access pool ordering from a trusted checkpoint."""
    future = None
    for key in record["future"]:
        future = merge_heaps(future, DueHeap(tuple(key)))
    pools = tuple(
        IndexedPool(pvector(record[name]), pmap({image: index for index, image in enumerate(record[name])}))
        for name in ("historical", "current")
    )
    return ReviewScheduler(pmap({int(image): ReviewState(**state) for image, state in record["states"].items()}),
                           future, *pools, int(record["stage"]), int(record["clock"]))

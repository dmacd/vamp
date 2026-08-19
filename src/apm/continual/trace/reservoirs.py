"""Deterministic replay reservoirs for TRACE merge repair."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
from typing import Iterable, Sequence

from apm.continual.artifacts import require_sha256
from apm.continual.trace.protocol import REPAIR_FRACTION_GRID, SEED


@dataclass(frozen=True, slots=True, order=True)
class ReservoirEntry:
    """One permanently prioritized training example."""

    priority: str
    example_id: str

    def __post_init__(self) -> None:
        require_sha256(self.priority, "reservoir priority")
        require_sha256(self.example_id, "reservoir example")


def example_priority(example_id: str, seed: int = SEED) -> str:
    """Return the permanent priority assigned to a training example."""
    require_sha256(example_id, "reservoir example")
    return sha256(f"trace-reservoir-v1\0{seed}\0{example_id}".encode()).hexdigest()


def prioritized_entries(
    example_ids: Iterable[str],
    seed: int = SEED,
) -> tuple[ReservoirEntry, ...]:
    """Return unique examples sorted by permanent priority and identity."""
    identities = tuple(example_ids)
    if len(set(identities)) != len(identities):
        raise ValueError("reservoir source example identities must be unique")
    return tuple(
        sorted(
            ReservoirEntry(example_priority(identity, seed), identity)
            for identity in identities
        )
    )


def select_reservoir(
    eligible: Sequence[ReservoirEntry],
    represented_example_count: int,
    fraction: float,
) -> tuple[ReservoirEntry, ...]:
    """Select the lowest-priority eligible entries for one node."""
    if fraction not in REPAIR_FRACTION_GRID:
        raise ValueError("repair fraction is outside the registered grid")
    if represented_example_count < 0:
        raise ValueError("represented example count must be nonnegative")
    target = math.ceil(fraction * represented_example_count)
    unique = {entry.example_id: entry for entry in eligible}
    if len(unique) != len(eligible):
        raise ValueError("eligible reservoir entries contain duplicates")
    ordered = tuple(sorted(unique.values()))
    if len(ordered) < target:
        raise ValueError("eligible child reservoirs cannot satisfy parent capacity")
    return ordered[:target]


def merge_reservoirs(
    left: Sequence[ReservoirEntry],
    right: Sequence[ReservoirEntry],
    represented_example_count: int,
    fraction: float,
) -> tuple[tuple[ReservoirEntry, ...], tuple[ReservoirEntry, ...]]:
    """Return the repair union and deterministic parent reservoir."""
    repair_examples = tuple(sorted((*left, *right)))
    if len({entry.example_id for entry in repair_examples}) != len(repair_examples):
        raise ValueError("child repair reservoirs overlap")
    return repair_examples, select_reservoir(
        repair_examples,
        represented_example_count,
        fraction,
    )


__all__ = [
    "ReservoirEntry",
    "example_priority",
    "merge_reservoirs",
    "prioritized_entries",
    "select_reservoir",
]

"""Semantic topology selection and deterministic one-to-one control pairing."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from hashlib import sha256
from itertools import combinations
from typing import Protocol

import numpy as np

from apm.data.text.tinyworlds_p.contracts import (
    ControlSelection,
    WorldCell,
)
from apm.data.text.tinyworlds_p.partitioning import AllocationGroup
from apm.data.text.tinyworlds_p_semantic.contracts import (
    BENCHMARK_ID,
    WORLD_LABELS,
    ControlPair,
)


class _CatalogConfig(Protocol):
    cluster_count: int


class _CatalogWord(Protocol):
    role: str
    word: str
    cluster: int | None
    vector: tuple[float, ...] | None


class _CatalogCluster(Protocol):
    role: str
    index: int
    centroid: tuple[float, ...]


class _Catalog(Protocol):
    config: _CatalogConfig
    words: Sequence[_CatalogWord]
    clusters: Sequence[_CatalogCluster]


class _PartitionPreset(Protocol):
    minimum_component_outside_groups: int
    selected_cell_median_tolerance: float


class SemanticPartitionGateError(ValueError):
    """A semantic partition topology, visibility, or pairing gate failed."""

    def __init__(
        self,
        message: str,
        audit: SemanticTopologyAudit | None = None,
    ) -> None:
        super().__init__(message)
        self.audit = audit


@dataclass(frozen=True, slots=True)
class _CellAggregate:
    token_mass: int
    group_count: int
    semantic_quality_sum: float
    nuisance_counts: tuple[tuple[tuple[str, str], int], ...]


@dataclass(frozen=True, slots=True)
class SemanticTopologyCandidate:
    """One fully visible and control-capable five-cell topology."""

    cells: tuple[tuple[int, int], ...]
    token_masses: tuple[int, ...]
    group_counts: tuple[int, ...]
    semantic_dispersion: float
    token_imbalance: Fraction
    nuisance_imbalance: Fraction
    control_capacity: Fraction
    tie_sha256: str

    @property
    def score(self) -> tuple[float, Fraction, Fraction, Fraction, str]:
        """Return the preregistered lexicographic objective."""
        return (
            self.semantic_dispersion,
            self.token_imbalance,
            self.nuisance_imbalance,
            -self.control_capacity,
            self.tie_sha256,
        )

    @property
    def median_token_mass(self) -> int:
        """Return the middle of the five selected cell masses."""
        return sorted(self.token_masses)[2]

    def passes_median_gate(self, tolerance: float) -> bool:
        """Check the frozen relative-to-median mass gate."""
        median = self.median_token_mass
        lower = median * (1.0 - tolerance)
        upper = median * (1.0 + tolerance)
        return all(lower <= mass <= upper for mass in self.token_masses)

    def as_record(self, tolerance: float) -> dict[str, object]:
        """Return exact score components and the diagnostic median result."""
        median = self.median_token_mass
        return {
            "cells": [list(cell) for cell in self.cells],
            "control_capacity": _fraction_record(self.control_capacity),
            "group_counts": list(self.group_counts),
            "median_gate": {
                "lower_token_mass": median * (1.0 - tolerance),
                "median_token_mass": median,
                "passes": self.passes_median_gate(tolerance),
                "tolerance": tolerance,
                "upper_token_mass": median * (1.0 + tolerance),
            },
            "nuisance_imbalance": _fraction_record(self.nuisance_imbalance),
            "semantic_dispersion": self.semantic_dispersion,
            "tie_sha256": self.tie_sha256,
            "token_imbalance": _fraction_record(self.token_imbalance),
            "token_masses": list(self.token_masses),
        }


@dataclass(frozen=True, slots=True)
class SemanticTopologyAudit:
    """Complete deterministic topology-screen counts and ranked candidates."""

    physical_candidate_count: int
    nonempty_candidate_count: int
    visible_candidate_count: int
    control_capable_candidate_count: int
    candidates: tuple[SemanticTopologyCandidate, ...]
    median_tolerance: float

    @property
    def selected(self) -> SemanticTopologyCandidate | None:
        """Return the winner under the preregistered objective."""
        return self.candidates[0] if self.candidates else None

    @property
    def median_feasible_candidates(self) -> tuple[SemanticTopologyCandidate, ...]:
        """Return diagnostic-only candidates satisfying the downstream gate."""
        return tuple(
            candidate
            for candidate in self.candidates
            if candidate.passes_median_gate(self.median_tolerance)
        )

    def as_record(self) -> dict[str, object]:
        """Return the complete ranked topology evidence."""
        feasible = self.median_feasible_candidates
        return {
            "control_capable_candidate_count": self.control_capable_candidate_count,
            "median_feasible_candidate_count": len(feasible),
            "median_tolerance": self.median_tolerance,
            "nonempty_candidate_count": self.nonempty_candidate_count,
            "physical_candidate_count": self.physical_candidate_count,
            "ranked_candidates": [
                candidate.as_record(self.median_tolerance)
                for candidate in self.candidates
            ],
            "selected_rank": 0 if self.selected is not None else None,
            "visible_candidate_count": self.visible_candidate_count,
        }


def select_semantic_world_cells(
    groups: Iterable[AllocationGroup],
    catalog: _Catalog,
    identity_sha256: str,
    preset: _PartitionPreset,
    *,
    benchmark_id: str = BENCHMARK_ID,
) -> tuple[WorldCell, ...]:
    """Choose a 2x2 corner plus E using only semantic/data/control quality."""
    audit = audit_semantic_world_cells(
        groups,
        catalog,
        identity_sha256,
        preset,
        benchmark_id=benchmark_id,
    )
    selected = audit.selected
    if selected is None:
        raise SemanticPartitionGateError(
            "no complete semantic topology has visible row/column controls",
            audit,
        )
    if not selected.passes_median_gate(preset.selected_cell_median_tolerance):
        raise SemanticPartitionGateError(
            "best semantic topology violates the selected-cell token median gate",
            audit,
        )
    return world_cells_from_semantic_candidate(selected)


def select_balance_eligible_semantic_world_cells(
    audit: SemanticTopologyAudit,
) -> tuple[WorldCell, ...]:
    """Select the semantic leader only after applying the fixed mass gate."""
    eligible = audit.median_feasible_candidates
    if not eligible:
        raise SemanticPartitionGateError(
            "no semantic topology satisfies the selected-cell token median gate",
            audit,
        )
    return world_cells_from_semantic_candidate(eligible[0])


def retie_semantic_topology_audit(
    audit: SemanticTopologyAudit,
    identity_sha256: str,
    benchmark_id: str,
) -> SemanticTopologyAudit:
    """Apply a new version's hash authority without changing measured scores."""
    candidates = tuple(
        replace(
            candidate,
            tie_sha256=_tie(
                identity_sha256,
                "topology",
                repr(candidate.cells),
                benchmark_id,
            ),
        )
        for candidate in audit.candidates
    )
    return replace(
        audit,
        candidates=tuple(sorted(candidates, key=lambda candidate: candidate.score)),
    )


def world_cells_from_semantic_candidate(
    selected: SemanticTopologyCandidate,
) -> tuple[WorldCell, ...]:
    """Assign canonical world labels to one already-selected topology."""
    return tuple(
        WorldCell(
            label=label,
            noun_bucket=cell[0],
            verb_bucket=cell[1],
            active_token_count=selected.token_masses[index],
            group_count=selected.group_counts[index],
        )
        for index, (label, cell) in enumerate(
            zip(WORLD_LABELS, selected.cells, strict=True)
        )
    )


def audit_semantic_world_cells(
    groups: Iterable[AllocationGroup],
    catalog: _Catalog,
    identity_sha256: str,
    preset: _PartitionPreset,
    *,
    benchmark_id: str = BENCHMARK_ID,
) -> SemanticTopologyAudit:
    """Aggregate and rank every topology without applying the final mass gate."""
    count = catalog.config.cluster_count
    token_mass: Counter[tuple[int, int]] = Counter()
    group_count: Counter[tuple[int, int]] = Counter()
    semantic_quality_sum: Counter[tuple[int, int]] = Counter()
    nuisance_counts: Counter[tuple[int, int, str, str]] = Counter()
    word_group_counts: Counter[tuple[str, str]] = Counter()
    cell_word_group_counts: dict[
        tuple[int, int], Counter[tuple[str, str]]
    ] = defaultdict(Counter)
    word_quality = _word_quality(catalog)
    for group in groups:
        cell = (group.noun_bucket, group.verb_bucket)
        token_mass[cell] += group.active_token_count
        group_count[cell] += 1
        semantic_quality_sum[cell] += group.active_token_count * (
            word_quality[("noun", group.noun)]
            + word_quality[("verb", group.verb)]
        ) / 2.0
        for role_word in (("noun", group.noun), ("verb", group.verb)):
            word_group_counts[role_word] += 1
            cell_word_group_counts[cell][role_word] += 1
        for dimension, category in group.marginals:
            nuisance_counts[(cell[0], cell[1], dimension, category)] += group.active_token_count
    by_cell = {
        cell: _CellAggregate(
            token_mass=mass,
            group_count=group_count[cell],
            semantic_quality_sum=semantic_quality_sum[cell],
            nuisance_counts=tuple(
                sorted(
                    ((dimension, category), value)
                    for (noun, verb, dimension, category), value in nuisance_counts.items()
                    if (noun, verb) == cell
                )
            ),
        )
        for cell, mass in token_mass.items()
    }
    candidates = (
        (
            (rows[0], columns[0]),
            (rows[1], columns[0]),
            (rows[1], columns[1]),
            (rows[0], columns[1]),
            (extra_row, extra_column),
        )
        for rows in combinations(range(count), 2)
        for columns in combinations(range(count), 2)
        for extra_row in range(count)
        if extra_row not in rows
        for extra_column in range(count)
        if extra_column not in columns
    )
    physical_candidate_count = 0
    nonempty_candidate_count = 0
    visible_candidate_count = 0
    control_capable_candidate_count = 0
    scored: list[SemanticTopologyCandidate] = []
    for cells in candidates:
        physical_candidate_count += 1
        if any(not by_cell.get(cell) for cell in cells):
            continue
        nonempty_candidate_count += 1
        selected = set(cells)
        cell_groups = tuple(by_cell[cell] for cell in cells)
        if not _has_component_visibility(
            word_group_counts,
            cell_word_group_counts,
            selected,
            preset.minimum_component_outside_groups,
        ):
            continue
        visible_candidate_count += 1
        if not _has_control_capacity(by_cell, selected, cells, cell_groups):
            continue
        control_capable_candidate_count += 1
        control_capacity = _control_capacity(cell_groups, by_cell, selected, cells)
        scored.append(
            SemanticTopologyCandidate(
                cells=cells,
                token_masses=tuple(cell.token_mass for cell in cell_groups),
                group_counts=tuple(cell.group_count for cell in cell_groups),
                semantic_dispersion=_semantic_dispersion(cell_groups),
                token_imbalance=_token_imbalance(cell_groups),
                nuisance_imbalance=_nuisance_imbalance(cell_groups),
                control_capacity=control_capacity,
                tie_sha256=_tie(
                    identity_sha256,
                    "topology",
                    repr(cells),
                    benchmark_id,
                ),
            )
        )
    return SemanticTopologyAudit(
        physical_candidate_count=physical_candidate_count,
        nonempty_candidate_count=nonempty_candidate_count,
        visible_candidate_count=visible_candidate_count,
        control_capable_candidate_count=control_capable_candidate_count,
        candidates=tuple(sorted(scored, key=lambda candidate: candidate.score)),
        median_tolerance=preset.selected_cell_median_tolerance,
    )


def pair_world_controls(
    evaluation_groups: Mapping[tuple[str, str], Sequence[AllocationGroup]],
    cells: Sequence[WorldCell],
    controls: Sequence[ControlSelection],
    identity_sha256: str,
    *,
    benchmark_id: str = BENCHMARK_ID,
) -> tuple[ControlPair, ...]:
    """Persist one nuisance-first, length/mass/hash pairing for each control arm."""
    cells_by_world = {cell.label: cell for cell in cells}
    base_by_split = {
        split: {
            group.normalized_sha256: group
            for group in evaluation_groups[("base", split)]
        }
        for split in ("validation", "test")
    }
    pairings: list[ControlPair] = []
    for split in ("validation", "test"):
        for world in WORLD_LABELS:
            selection = next(
                (
                    item
                    for item in controls
                    if item.world == world and item.split == split
                ),
                None,
            )
            if selection is None:
                raise SemanticPartitionGateError("control selection family is incomplete")
            cell = cells_by_world[world]
            selected_groups = tuple(base_by_split[split][digest] for digest in selection.group_sha256)
            row = tuple(
                group
                for group in selected_groups
                if group.noun_bucket == cell.noun_bucket
                and group.verb_bucket != cell.verb_bucket
            )
            column = tuple(
                group
                for group in selected_groups
                if group.verb_bucket == cell.verb_bucket
                and group.noun_bucket != cell.noun_bucket
            )
            if (
                len(row) != selection.row_group_count
                or len(column) != selection.column_group_count
            ):
                raise SemanticPartitionGateError("control arm membership changed")
            targets = tuple(evaluation_groups[(world, split)])
            if len(targets) != len(selected_groups):
                raise SemanticPartitionGateError("world and control group counts differ")
            ordered_controls = tuple(
                sorted(
                    (
                        (arm, group)
                        for arm, control_groups in (("row", row), ("column", column))
                        for group in control_groups
                    ),
                    key=lambda item: (
                        item[1].full_stratum,
                        item[1].canonical_token_count,
                        item[1].active_token_count,
                        _tie(
                            identity_sha256,
                            f"pair-control:{world}:{split}:{item[0]}",
                            item[1].normalized_sha256,
                            benchmark_id,
                        ),
                    ),
                )
            )
            ordered_targets = tuple(
                sorted(
                    targets,
                    key=lambda item: (
                        item.full_stratum,
                        item.canonical_token_count,
                        item.active_token_count,
                        _tie(
                            identity_sha256,
                            f"pair-target:{world}:{split}",
                            item.normalized_sha256,
                            benchmark_id,
                        ),
                    ),
                )
            )
            pairings.extend(
                ControlPair(
                    world=world,
                    split=split,
                    arm=arm,
                    world_group_sha256=target.normalized_sha256,
                    control_group_sha256=control.normalized_sha256,
                )
                for target, (arm, control) in zip(
                    ordered_targets,
                    ordered_controls,
                    strict=True,
                )
            )
    return tuple(
        sorted(
            pairings,
            key=lambda item: (
                ("validation", "test").index(item.split),
                WORLD_LABELS.index(item.world),
                item.world_group_sha256,
                item.control_group_sha256,
            ),
        )
    )


def _word_quality(catalog: _Catalog) -> dict[tuple[str, str], float]:
    centroids = {
        (cluster.role, cluster.index): np.asarray(cluster.centroid, dtype=np.float64)
        for cluster in catalog.clusters
    }
    return {
        (word.role, word.word): float(
            centroids[(word.role, int(word.cluster))] @ np.asarray(word.vector)
        )
        for word in catalog.words
        if word.cluster is not None and word.vector is not None
    }


def _semantic_dispersion(
    cells: Sequence[_CellAggregate],
) -> float:
    weighted_quality = sum(cell.semantic_quality_sum for cell in cells)
    mass = sum(cell.token_mass for cell in cells)
    return 1.0 - weighted_quality / mass


def _token_imbalance(cells: Sequence[_CellAggregate]) -> Fraction:
    masses = tuple(cell.token_mass for cell in cells)
    total = sum(masses)
    return Fraction(sum(abs(len(masses) * mass - total) for mass in masses), total)


def _nuisance_imbalance(cells: Sequence[_CellAggregate]) -> Fraction:
    totals = [cell.token_mass for cell in cells]
    categories = tuple(
        sorted(
            {
                marginal
                for cell in cells
                for marginal, _ in cell.nuisance_counts
            }
        )
    )
    counts = [
        Counter(dict(cell.nuisance_counts))
        for cell in cells
    ]
    return sum(
        (
            Fraction(counts[left][category], totals[left])
            - Fraction(counts[right][category], totals[right])
        )
        ** 2
        for left in range(len(cells))
        for right in range(left + 1, len(cells))
        for category in categories
    )


def _has_control_capacity(
    by_cell: Mapping[tuple[int, int], _CellAggregate],
    selected: set[tuple[int, int]],
    cells: Sequence[tuple[int, int]],
    targets: Sequence[_CellAggregate],
) -> bool:
    for cell, target in zip(cells, targets, strict=True):
        row_count = sum(
            aggregate.group_count
            for candidate, aggregate in by_cell.items()
            if candidate not in selected and candidate[0] == cell[0]
        )
        column_count = sum(
            aggregate.group_count
            for candidate, aggregate in by_cell.items()
            if candidate not in selected and candidate[1] == cell[1]
        )
        required_row = target.group_count // 2
        required_column = target.group_count - required_row
        if row_count < required_row or column_count < required_column:
            return False
    return True


def _has_component_visibility(
    totals: Mapping[tuple[str, str], int],
    by_cell: Mapping[tuple[int, int], Counter[tuple[str, str]]],
    selected: set[tuple[int, int]],
    minimum_outside_groups: int,
) -> bool:
    selected_counts: Counter[tuple[str, str]] = Counter()
    for cell in selected:
        selected_counts.update(by_cell[cell])
    return all(
        totals[role_word] - selected_count >= minimum_outside_groups
        for role_word, selected_count in selected_counts.items()
    )


def _control_capacity(
    cell_groups: Sequence[_CellAggregate],
    by_cell: Mapping[tuple[int, int], _CellAggregate],
    selected: set[tuple[int, int]],
    cells: Sequence[tuple[int, int]],
) -> Fraction:
    capacities = []
    for target, cell in zip(cell_groups, cells, strict=True):
        row = sum(
            aggregate.group_count
            for candidate, aggregate in by_cell.items()
            if candidate not in selected and candidate[0] == cell[0]
        )
        column = sum(
            aggregate.group_count
            for candidate, aggregate in by_cell.items()
            if candidate not in selected and candidate[1] == cell[1]
        )
        capacities.append(Fraction(min(row, column), target.group_count))
    return min(capacities)


def _tie(
    identity_sha256: str,
    namespace: str,
    value: str,
    benchmark_id: str,
) -> str:
    return sha256(
        f"{benchmark_id}\0{identity_sha256}\0{namespace}\0{value}".encode("utf-8")
    ).hexdigest()


def _fraction_record(value: Fraction) -> dict[str, int]:
    return {
        "denominator": value.denominator,
        "numerator": value.numerator,
    }


__all__ = [
    "SemanticPartitionGateError",
    "SemanticTopologyAudit",
    "SemanticTopologyCandidate",
    "audit_semantic_world_cells",
    "pair_world_controls",
    "retie_semantic_topology_audit",
    "select_balance_eligible_semantic_world_cells",
    "select_semantic_world_cells",
    "world_cells_from_semantic_candidate",
]

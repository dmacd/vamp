"""Pure reconstruction of semantic-v6 feasibility and topology selection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction

from apm.data.text.tinyworlds_p.contracts import WorldCell
from apm.data.text.tinyworlds_p_semantic.contracts import WORLD_LABELS
from apm.data.text.tinyworlds_p_semantic.partitioning import _tie
from apm.data.text.tinyworlds_p_semantic.v6_partition_contracts import (
    V6_BENCHMARK_ID,
    V6CandidateFeasibility,
    V6SemanticPartitionPreset,
)


class V6TopologyEvidenceError(ValueError):
    """Semantic-v6 topology or feasibility evidence is inconsistent."""


def ranked_balance_candidates(
    parent_candidates: Sequence[Mapping[str, object]],
    seed_identity_sha256: str,
) -> tuple[dict[str, object], ...]:
    """Return all balanced parent candidates in the semantic-v6 score order."""
    if not parent_candidates:
        raise V6TopologyEvidenceError("semantic-v6 parent has no candidates")
    eligible = tuple(
        {
            **candidate,
            "tie_sha256": _tie(
                seed_identity_sha256,
                "topology",
                repr(_cells(candidate)),
                V6_BENCHMARK_ID,
            ),
        }
        for candidate in parent_candidates
        if _mapping(candidate, "median_gate").get("passes") is True
    )
    if not eligible:
        raise V6TopologyEvidenceError("semantic-v6 has no balanced candidate")
    return tuple(sorted(eligible, key=_candidate_score))


def topology_selection_from_feasibility(
    parent_candidates: Sequence[Mapping[str, object]],
    feasibility: Sequence[V6CandidateFeasibility],
    seed_identity_sha256: str,
    preset: V6SemanticPartitionPreset,
) -> dict[str, object]:
    """Select the semantic leader after complete exact-control measurement."""
    ranked = ranked_balance_candidates(parent_candidates, seed_identity_sha256)
    if len(feasibility) != len(ranked):
        raise V6TopologyEvidenceError(
            "semantic-v6 did not measure every balanced candidate"
        )
    for rank, (candidate, measured) in enumerate(
        zip(ranked, feasibility, strict=True)
    ):
        if (
            measured.semantic_rank != rank
            or measured.cells != _cells(candidate)
        ):
            raise V6TopologyEvidenceError(
                "semantic-v6 feasibility order differs from semantic ranking"
            )
    feasible_ranks = tuple(
        item.semantic_rank for item in feasibility if item.control_feasible
    )
    selected = ranked[feasible_ranks[0]] if feasible_ranks else None
    return {
        "balance_eligible_candidate_count": len(ranked),
        "control_feasible_candidate_count": len(feasible_ranks),
        "evaluations": [item.as_record() for item in feasibility],
        "measured_candidate_count": len(parent_candidates),
        "selected": selected,
        "selection_method": preset.topology_selection_method,
    }


def validate_topology_selection(
    selection: Mapping[str, object],
    parent_candidates: Sequence[Mapping[str, object]],
    seed_identity_sha256: str,
    preset: V6SemanticPartitionPreset,
    cells: Sequence[WorldCell] | None = None,
) -> None:
    """Reconstruct and validate complete v6 evidence and published cells."""
    raw_evaluations = selection.get("evaluations")
    if type(raw_evaluations) is not list:
        raise V6TopologyEvidenceError("semantic-v6 evaluations changed")
    feasibility = tuple(_feasibility(item) for item in raw_evaluations)
    expected = topology_selection_from_feasibility(
        parent_candidates,
        feasibility,
        seed_identity_sha256,
        preset,
    )
    if dict(selection) != expected:
        raise V6TopologyEvidenceError("semantic-v6 topology selection changed")
    selected = selection.get("selected")
    if cells is None:
        return
    if type(selected) is not dict:
        raise V6TopologyEvidenceError("semantic-v6 successful partition has no winner")
    if _cells(selected) != tuple(
        (cell.noun_bucket, cell.verb_bucket) for cell in cells
    ):
        raise V6TopologyEvidenceError("semantic-v6 published cells changed")


def world_cells_from_topology_selection(
    selection: Mapping[str, object],
) -> tuple[WorldCell, ...]:
    """Construct the five labeled cells from a successful v6 selection."""
    selected = selection.get("selected")
    if type(selected) is not dict:
        raise V6TopologyEvidenceError("semantic-v6 has no feasible topology")
    cells = _cells(selected)
    masses = _positive_integer_five(selected, "token_masses")
    group_counts = _positive_integer_five(selected, "group_counts")
    return tuple(
        WorldCell(
            label=label,
            noun_bucket=coordinates[0],
            verb_bucket=coordinates[1],
            active_token_count=masses[index],
            group_count=group_counts[index],
        )
        for index, (label, coordinates) in enumerate(
            zip(WORLD_LABELS, cells, strict=True)
        )
    )


def selected_feasibility(
    selection: Mapping[str, object],
) -> V6CandidateFeasibility:
    """Return the exact feasibility record associated with the selected cells."""
    selected = selection.get("selected")
    evaluations = selection.get("evaluations")
    if type(selected) is not dict or type(evaluations) is not list:
        raise V6TopologyEvidenceError("semantic-v6 selection has no feasible winner")
    cells = _cells(selected)
    matches = tuple(
        item
        for item in (_feasibility(record) for record in evaluations)
        if item.cells == cells and item.control_feasible
    )
    if len(matches) != 1:
        raise V6TopologyEvidenceError("semantic-v6 selected feasibility changed")
    return matches[0]


def _feasibility(record: object) -> V6CandidateFeasibility:
    if type(record) is not dict or set(record) != {
        "cells",
        "control_feasible",
        "controls_sha256",
        "failure_reason",
        "semantic_rank",
        "split_assignments_sha256",
    }:
        raise V6TopologyEvidenceError("semantic-v6 feasibility fields changed")
    feasible = record.get("control_feasible")
    controls_sha = record.get("controls_sha256")
    failure_reason = record.get("failure_reason")
    if (
        type(feasible) is not bool
        or controls_sha is not None and type(controls_sha) is not str
        or failure_reason is not None and type(failure_reason) is not str
    ):
        raise V6TopologyEvidenceError("semantic-v6 feasibility values changed")
    try:
        return V6CandidateFeasibility(
            semantic_rank=_integer(record, "semantic_rank"),
            cells=_cells(record),
            split_assignments_sha256=_text(record, "split_assignments_sha256"),
            control_feasible=feasible,
            controls_sha256=controls_sha,
            failure_reason=failure_reason,
        )
    except (TypeError, ValueError) as error:
        raise V6TopologyEvidenceError(
            "semantic-v6 feasibility record changed"
        ) from error


def _candidate_score(
    candidate: Mapping[str, object],
) -> tuple[float, Fraction, Fraction, Fraction, str]:
    dispersion = candidate.get("semantic_dispersion")
    tie_sha256 = candidate.get("tie_sha256")
    if type(dispersion) not in (int, float) or type(tie_sha256) is not str:
        raise V6TopologyEvidenceError("semantic-v6 candidate score changed")
    return (
        float(dispersion),
        _fraction(_mapping(candidate, "token_imbalance")),
        _fraction(_mapping(candidate, "nuisance_imbalance")),
        -_fraction(_mapping(candidate, "control_capacity")),
        tie_sha256,
    )


def _cells(candidate: Mapping[str, object]) -> tuple[tuple[int, int], ...]:
    raw = candidate.get("cells")
    if (
        type(raw) is not list
        or len(raw) != 5
        or any(
            type(cell) is not list
            or len(cell) != 2
            or any(type(value) is not int or value < 0 for value in cell)
            for cell in raw
        )
    ):
        raise V6TopologyEvidenceError("semantic-v6 candidate cells changed")
    return tuple((cell[0], cell[1]) for cell in raw)


def _positive_integer_five(
    record: Mapping[str, object],
    field: str,
) -> tuple[int, int, int, int, int]:
    raw = record.get(field)
    if (
        type(raw) is not list
        or len(raw) != 5
        or any(type(value) is not int or value <= 0 for value in raw)
    ):
        raise V6TopologyEvidenceError(f"semantic-v6 candidate {field} changed")
    return raw[0], raw[1], raw[2], raw[3], raw[4]


def _mapping(record: Mapping[str, object], field: str) -> dict[str, object]:
    value = record.get(field)
    if type(value) is not dict:
        raise V6TopologyEvidenceError(f"semantic-v6 field {field!r} changed")
    return value


def _fraction(record: Mapping[str, object]) -> Fraction:
    if set(record) != {"denominator", "numerator"}:
        raise V6TopologyEvidenceError("semantic-v6 candidate fraction changed")
    numerator = record.get("numerator")
    denominator = record.get("denominator")
    if (
        type(numerator) is not int
        or numerator < 0
        or type(denominator) is not int
        or denominator <= 0
    ):
        raise V6TopologyEvidenceError("semantic-v6 candidate fraction changed")
    return Fraction(numerator, denominator)


def _integer(record: Mapping[str, object], field: str) -> int:
    value = record.get(field)
    if type(value) is not int:
        raise V6TopologyEvidenceError(f"semantic-v6 field {field!r} changed")
    return value


def _text(record: Mapping[str, object], field: str) -> str:
    value = record.get(field)
    if type(value) is not str or not value:
        raise V6TopologyEvidenceError(f"semantic-v6 field {field!r} changed")
    return value


__all__ = [
    "V6TopologyEvidenceError",
    "ranked_balance_candidates",
    "selected_feasibility",
    "topology_selection_from_feasibility",
    "validate_topology_selection",
    "world_cells_from_topology_selection",
]

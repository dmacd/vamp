"""Pure reconstruction and validation of semantic-v5 topology selection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction

from apm.data.text.tinyworlds_p.contracts import WorldCell
from apm.data.text.tinyworlds_p_semantic.contracts import WORLD_LABELS
from apm.data.text.tinyworlds_p_semantic.partitioning import _tie
from apm.data.text.tinyworlds_p_semantic.v5_partition_contracts import (
    V5_BENCHMARK_ID,
    V5SemanticPartitionPreset,
)


class V5TopologyEvidenceError(ValueError):
    """Semantic-v5 topology evidence is incomplete or inconsistent."""


def topology_selection_from_parent_candidates(
    parent_candidates: Sequence[Mapping[str, object]],
    seed_identity_sha256: str,
    preset: V5SemanticPartitionPreset,
) -> dict[str, object]:
    """Apply the frozen v5 eligibility and ranking rule to v4 records."""
    if not parent_candidates:
        raise V5TopologyEvidenceError("semantic-v5 parent has no topology candidates")
    eligible = tuple(
        candidate
        for candidate in parent_candidates
        if _median_gate(candidate).get("passes") is True
    )
    if not eligible:
        raise V5TopologyEvidenceError("semantic-v5 has no balance-eligible candidate")
    retied = tuple(
        {
            **candidate,
            "tie_sha256": _tie(
                seed_identity_sha256,
                "topology",
                repr(_cells(candidate)),
                V5_BENCHMARK_ID,
            ),
        }
        for candidate in eligible
    )
    selected = min(retied, key=_candidate_score)
    return {
        "balance_eligible_candidate_count": len(eligible),
        "measured_candidate_count": len(parent_candidates),
        "parent_unconstrained_winner": dict(parent_candidates[0]),
        "selected": selected,
        "selection_method": preset.topology_selection_method,
    }


def validate_topology_selection(
    selection: Mapping[str, object],
    parent_candidates: Sequence[Mapping[str, object]],
    seed_identity_sha256: str,
    preset: V5SemanticPartitionPreset,
    cells: Sequence[WorldCell] | None = None,
) -> None:
    """Require an exact v5 selection and, when supplied, its published cells."""
    expected = topology_selection_from_parent_candidates(
        parent_candidates,
        seed_identity_sha256,
        preset,
    )
    if dict(selection) != expected:
        raise V5TopologyEvidenceError("semantic-v5 topology selection changed")
    selected = _mapping(selection, "selected")
    if _median_gate(selected).get("tolerance") != (
        preset.selected_cell_median_tolerance
    ):
        raise V5TopologyEvidenceError("semantic-v5 balance tolerance changed")
    if cells is not None and _cells(selected) != tuple(
        (cell.noun_bucket, cell.verb_bucket) for cell in cells
    ):
        raise V5TopologyEvidenceError("semantic-v5 published cells changed")


def world_cells_from_topology_selection(
    selection: Mapping[str, object],
) -> tuple[WorldCell, ...]:
    """Construct the five labeled cells from a validated selection record."""
    selected = _mapping(selection, "selected")
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


def _candidate_score(
    candidate: Mapping[str, object],
) -> tuple[float, Fraction, Fraction, Fraction, str]:
    dispersion = candidate.get("semantic_dispersion")
    tie_sha256 = candidate.get("tie_sha256")
    if type(dispersion) not in (int, float) or type(tie_sha256) is not str:
        raise V5TopologyEvidenceError("semantic-v5 candidate score changed")
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
        raise V5TopologyEvidenceError("semantic-v5 candidate cells changed")
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
        raise V5TopologyEvidenceError(f"semantic-v5 candidate {field} changed")
    return raw[0], raw[1], raw[2], raw[3], raw[4]


def _mapping(
    record: Mapping[str, object],
    field: str,
) -> dict[str, object]:
    value = record.get(field)
    if type(value) is not dict:
        raise V5TopologyEvidenceError(f"semantic-v5 field {field!r} changed")
    return value


def _median_gate(candidate: Mapping[str, object]) -> dict[str, object]:
    return _mapping(candidate, "median_gate")


def _fraction(record: Mapping[str, object]) -> Fraction:
    if set(record) != {"denominator", "numerator"}:
        raise V5TopologyEvidenceError("semantic-v5 candidate fraction changed")
    numerator = record.get("numerator")
    denominator = record.get("denominator")
    if (
        type(numerator) is not int
        or numerator < 0
        or type(denominator) is not int
        or denominator <= 0
    ):
        raise V5TopologyEvidenceError("semantic-v5 candidate fraction changed")
    return Fraction(numerator, denominator)


__all__ = [
    "V5TopologyEvidenceError",
    "topology_selection_from_parent_candidates",
    "validate_topology_selection",
    "world_cells_from_topology_selection",
]

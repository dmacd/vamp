from __future__ import annotations

from fractions import Fraction
from hashlib import sha256
from pathlib import Path

import pytest

from apm.data.text.tinyworlds_p_semantic import (
    PartitionArtifactError,
    V5_SEMANTIC_PARTITION_PRESET,
    V5SemanticPartitionPreset,
    build_v4_partition,
    load_v5_partition,
    parse_v5_control_shortfall,
)
from apm.data.text.tinyworlds_p_semantic.partitioning import (
    SemanticTopologyAudit,
    SemanticTopologyCandidate,
    retie_semantic_topology_audit,
    select_balance_eligible_semantic_world_cells,
)
from apm.data.text.tinyworlds_p_semantic.v5_partition_contracts import (
    V5_BENCHMARK_ID,
)
from apm.data.text.tinyworlds_p_semantic.v5_partition_failure import (
    V5SemanticPartitionFailureError,
)
from apm.data.text.tinyworlds_p_semantic.v5_topology import (
    topology_selection_from_parent_candidates,
    validate_topology_selection,
    world_cells_from_topology_selection,
)
from test_tinyworlds_p_semantic_v4_partition import _fixture


def _candidate(
    cells: tuple[tuple[int, int], ...],
    masses: tuple[int, ...],
    dispersion: float,
    name: str,
) -> SemanticTopologyCandidate:
    return SemanticTopologyCandidate(
        cells=cells,
        token_masses=masses,
        group_counts=(10, 10, 10, 10, 10),
        semantic_dispersion=dispersion,
        token_imbalance=Fraction(1, 10),
        nuisance_imbalance=Fraction(1, 20),
        control_capacity=Fraction(2, 1),
        tie_sha256=sha256(name.encode("utf-8")).hexdigest(),
    )


def test_v5_filters_for_balance_before_semantic_ranking() -> None:
    semantic_winner = _candidate(
        ((0, 0), (1, 0), (1, 1), (0, 1), (2, 2)),
        (100, 100, 100, 100, 180),
        0.10,
        "semantic-winner",
    )
    balanced_runner_up = _candidate(
        ((2, 0), (3, 0), (3, 2), (2, 2), (1, 1)),
        (100, 102, 98, 101, 99),
        0.20,
        "balanced-runner-up",
    )
    audit = SemanticTopologyAudit(
        physical_candidate_count=2,
        nonempty_candidate_count=2,
        visible_candidate_count=2,
        control_capable_candidate_count=2,
        candidates=(semantic_winner, balanced_runner_up),
        median_tolerance=0.10,
    )

    cells = select_balance_eligible_semantic_world_cells(audit)
    assert tuple((cell.noun_bucket, cell.verb_bucket) for cell in cells) == (
        balanced_runner_up.cells
    )
    assert audit.selected == semantic_winner
    assert audit.median_feasible_candidates == (balanced_runner_up,)

    retied = retie_semantic_topology_audit(audit, "a" * 64, V5_BENCHMARK_ID)
    assert tuple(candidate.tie_sha256 for candidate in retied.candidates) != tuple(
        candidate.tie_sha256 for candidate in audit.candidates
    )
    assert {
        (candidate.cells, candidate.token_masses, candidate.semantic_dispersion)
        for candidate in retied.candidates
    } == {
        (candidate.cells, candidate.token_masses, candidate.semantic_dispersion)
        for candidate in audit.candidates
    }

    parent_records = tuple(
        candidate.as_record(audit.median_tolerance) for candidate in audit.candidates
    )
    selection = topology_selection_from_parent_candidates(
        parent_records,
        "a" * 64,
        V5_SEMANTIC_PARTITION_PRESET,
    )
    validate_topology_selection(
        selection,
        parent_records,
        "a" * 64,
        V5_SEMANTIC_PARTITION_PRESET,
    )
    selected_cells = world_cells_from_topology_selection(selection)
    assert tuple(
        (cell.noun_bucket, cell.verb_bucket) for cell in selected_cells
    ) == balanced_runner_up.cells


def test_v5_requires_a_balanced_candidate_and_fixed_parent() -> None:
    only_unbalanced = _candidate(
        ((0, 0), (1, 0), (1, 1), (0, 1), (2, 2)),
        (100, 100, 100, 100, 180),
        0.10,
        "unbalanced",
    )
    audit = SemanticTopologyAudit(1, 1, 1, 1, (only_unbalanced,), 0.10)
    with pytest.raises(ValueError, match="no semantic topology satisfies"):
        select_balance_eligible_semantic_world_cells(audit)
    with pytest.raises(ValueError, match="fixed partition choice changed"):
        V5SemanticPartitionPreset(parent_partition_failure_sha256="f" * 64)
    with pytest.raises(ValueError, match="frozen v4 partition setting"):
        V5SemanticPartitionPreset(control_token_tolerance=0.01)
    assert V5_SEMANTIC_PARTITION_PRESET.selected_cell_median_tolerance == 0.10


def test_v5_control_shortfall_is_structured_and_strict() -> None:
    reason = "control:B:validation:column has 1511 candidates for 2314 controls"
    shortfall = parse_v5_control_shortfall(reason)
    assert shortfall.reason == reason
    assert shortfall.required_count - shortfall.available_count == 803
    with pytest.raises(V5SemanticPartitionFailureError, match="unregistered"):
        parse_v5_control_shortfall("not enough controls")
    with pytest.raises(V5SemanticPartitionFailureError, match="inconsistent"):
        parse_v5_control_shortfall(
            "control:B:validation:column has 2314 candidates for 2314 controls"
        )


def test_v5_loader_rejects_a_v4_partition(tmp_path: Path) -> None:
    inputs, preset, _ = _fixture(tmp_path)
    v4 = build_v4_partition(inputs, preset)
    with pytest.raises(PartitionArtifactError, match="unsupported semantic-v5"):
        load_v5_partition(v4.root)

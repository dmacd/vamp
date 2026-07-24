from __future__ import annotations

from fractions import Fraction
from hashlib import sha256
from pathlib import Path

import pytest

from apm.data.text.tinyworlds_p_semantic import (
    CANONICAL_ARCHIVE_IDENTITY,
    CANONICAL_TOKENIZER_IDENTITY,
    PartitionArtifactError,
    V6_SEMANTIC_PARTITION_PRESET,
    V6CandidateFeasibility,
    V6SemanticPartitionPreset,
    build_v4_partition,
    load_v6_partition,
)
from apm.data.text.tinyworlds_p.builder import _prepare_control_feasibility
from apm.data.text.tinyworlds_p.contracts import (
    PartitionInputs,
    PartitionPreset,
    WorldCell,
)
from apm.data.text.tinyworlds_p.partitioning import (
    AllocationGroup,
    PartitionGateError,
)
from apm.data.text.tinyworlds_p_semantic.partitioning import (
    SemanticTopologyCandidate,
)
from apm.data.text.tinyworlds_p_semantic.v6_topology import (
    V6TopologyEvidenceError,
    ranked_balance_candidates,
    selected_feasibility,
    topology_selection_from_feasibility,
    validate_topology_selection,
    world_cells_from_topology_selection,
)
from test_tinyworlds_p_semantic_v4_partition import _fixture


def _candidate(
    cells: tuple[tuple[int, int], ...],
    dispersion: float,
    name: str,
) -> SemanticTopologyCandidate:
    return SemanticTopologyCandidate(
        cells=cells,
        token_masses=(100, 102, 98, 101, 99),
        group_counts=(10, 10, 10, 10, 10),
        semantic_dispersion=dispersion,
        token_imbalance=Fraction(1, 10),
        nuisance_imbalance=Fraction(1, 20),
        control_capacity=Fraction(2, 1),
        tie_sha256=sha256(name.encode("utf-8")).hexdigest(),
    )


def _feasibility(
    rank: int,
    cells: tuple[tuple[int, int], ...],
    feasible: bool,
) -> V6CandidateFeasibility:
    return V6CandidateFeasibility(
        semantic_rank=rank,
        cells=cells,
        split_assignments_sha256=sha256(f"split-{rank}".encode()).hexdigest(),
        control_feasible=feasible,
        controls_sha256=(
            sha256(f"controls-{rank}".encode()).hexdigest() if feasible else None
        ),
        failure_reason=(
            None
            if feasible
            else f"control:A:validation:row has {rank} candidates for 5 controls"
        ),
    )


def test_v6_filters_exact_control_failures_before_semantic_selection() -> None:
    semantic_winner = _candidate(
        ((0, 0), (1, 0), (1, 1), (0, 1), (2, 2)),
        0.10,
        "semantic-winner",
    )
    feasible_runner_up = _candidate(
        ((2, 0), (3, 0), (3, 2), (2, 2), (1, 1)),
        0.20,
        "feasible-runner-up",
    )
    parent_records = tuple(
        candidate.as_record(0.10)
        for candidate in (semantic_winner, feasible_runner_up)
    )
    ranked = ranked_balance_candidates(parent_records, "a" * 64)
    assert tuple(record["semantic_dispersion"] for record in ranked) == (0.10, 0.20)
    feasibility = (
        _feasibility(0, semantic_winner.cells, False),
        _feasibility(1, feasible_runner_up.cells, True),
    )

    selection = topology_selection_from_feasibility(
        parent_records,
        feasibility,
        "a" * 64,
        V6_SEMANTIC_PARTITION_PRESET,
    )
    validate_topology_selection(
        selection,
        parent_records,
        "a" * 64,
        V6_SEMANTIC_PARTITION_PRESET,
    )
    cells = world_cells_from_topology_selection(selection)
    assert tuple((cell.noun_bucket, cell.verb_bucket) for cell in cells) == (
        feasible_runner_up.cells
    )
    assert selected_feasibility(selection) == feasibility[1]
    assert selection["control_feasible_candidate_count"] == 1


def test_v6_requires_every_balanced_candidate_to_be_measured() -> None:
    candidates = (
        _candidate(((0, 0), (1, 0), (1, 1), (0, 1), (2, 2)), 0.10, "one"),
        _candidate(((2, 0), (3, 0), (3, 2), (2, 2), (1, 1)), 0.20, "two"),
    )
    parent_records = tuple(candidate.as_record(0.10) for candidate in candidates)
    with pytest.raises(V6TopologyEvidenceError, match="every balanced candidate"):
        topology_selection_from_feasibility(
            parent_records,
            (_feasibility(0, candidates[0].cells, False),),
            "b" * 64,
            V6_SEMANTIC_PARTITION_PRESET,
        )


def test_v6_stops_when_every_exact_control_allocation_fails() -> None:
    candidate = _candidate(
        ((0, 0), (1, 0), (1, 1), (0, 1), (2, 2)),
        0.10,
        "only",
    )
    records = (candidate.as_record(0.10),)
    selection = topology_selection_from_feasibility(
        records,
        (_feasibility(0, candidate.cells, False),),
        "c" * 64,
        V6_SEMANTIC_PARTITION_PRESET,
    )
    assert selection["selected"] is None
    assert selection["control_feasible_candidate_count"] == 0
    with pytest.raises(V6TopologyEvidenceError, match="no feasible topology"):
        world_cells_from_topology_selection(selection)


def test_v6_contract_freezes_parent_and_all_prior_partition_settings() -> None:
    with pytest.raises(ValueError, match="fixed partition choice changed"):
        V6SemanticPartitionPreset(parent_partition_failure_sha256="f" * 64)
    with pytest.raises(ValueError, match="frozen partition setting"):
        V6SemanticPartitionPreset(control_token_tolerance=0.01)
    assert V6_SEMANTIC_PARTITION_PRESET.selected_cell_median_tolerance == 0.10
    assert V6_SEMANTIC_PARTITION_PRESET.world_split_weights == (80, 10, 10)
    assert V6_SEMANTIC_PARTITION_PRESET.base_split_weights == (96, 2, 2)


def test_exact_control_feasibility_runs_real_splits_and_global_controls(
    tmp_path: Path,
) -> None:
    cells = tuple(
        WorldCell(label, noun, verb, 1_200, 12)
        for label, (noun, verb) in zip(
            ("A", "B", "C", "D", "E"),
            ((0, 0), (1, 0), (1, 1), (0, 1), (2, 2)),
            strict=True,
        )
    )

    def groups(include_column_pool: bool) -> tuple[AllocationGroup, ...]:
        selected = {(cell.noun_bucket, cell.verb_bucket) for cell in cells}
        counts = {
            coordinates: 12 if coordinates in selected else 36
            for coordinates in (
                (0, 0),
                (0, 1),
                (0, 2),
                (1, 0),
                (1, 1),
                (1, 2),
                (2, 0),
                (2, 1),
                (2, 2),
            )
            if include_column_pool or coordinates != (2, 0)
        }
        return tuple(
            AllocationGroup(
                normalized_sha256=sha256(
                    f"{noun}-{verb}-{index}".encode("utf-8")
                ).hexdigest(),
                active_token_count=100,
                canonical_token_count=100,
                noun=f"noun-{noun}",
                verb=f"verb-{verb}",
                adjective="calm",
                noun_bucket=noun,
                verb_bucket=verb,
                adjective_bucket=0,
                source="fixture",
                feature_signature="none",
            )
            for (noun, verb), count in counts.items()
            for index in range(count)
        )

    preset = PartitionPreset(
        bucket_count=3,
        worker_count=1,
        run_record_count=10,
        world_split_weights=(1, 1, 1),
        base_split_weights=(1, 1, 1),
    )

    def inputs(name: str) -> PartitionInputs:
        working = tmp_path / name
        working.mkdir()
        return PartitionInputs(
            archive_path=tmp_path / "archive",
            tokenizer_directory=tmp_path / "tokenizer",
            output_root=tmp_path / "output",
            temporary_directory=working,
            archive_identity=CANONICAL_ARCHIVE_IDENTITY,
            tokenizer_identity=CANONICAL_TOKENIZER_IDENTITY,
        )

    feasible = _prepare_control_feasibility(
        inputs("feasible"),
        preset,
        cells,
        lambda: iter(groups(True)),
        "d" * 64,
    )
    assert len(feasible.controls) == 10
    assert feasible.split_assignments_path.is_file()
    with pytest.raises(PartitionGateError, match="candidates for"):
        _prepare_control_feasibility(
            inputs("infeasible"),
            preset,
            cells,
            lambda: iter(groups(False)),
            "d" * 64,
        )


def test_v6_loader_rejects_a_v4_partition(tmp_path: Path) -> None:
    inputs, preset, _ = _fixture(tmp_path)
    v4 = build_v4_partition(inputs, preset)
    with pytest.raises(PartitionArtifactError, match="unsupported semantic-v6"):
        load_v6_partition(v4.root)

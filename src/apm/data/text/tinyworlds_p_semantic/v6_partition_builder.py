"""Archive-native construction of exact-control-feasible semantic-v6."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
import os
from pathlib import Path
import shutil

from apm.data.text.tinyworlds_p import builder as archive_builder
from apm.data.text.tinyworlds_p.archive_ingest import build_archive_ingest
from apm.data.text.tinyworlds_p.contracts import (
    NORMALIZATION_IDENTITY,
    PartitionInputs,
    PartitionPreset,
)
from apm.data.text.tinyworlds_p.partitioning import (
    AllocationGroup,
    PartitionGateError as ArchivePartitionGateError,
    bucket_word_lookup,
    require_component_visibility,
)
from apm.data.text.tinyworlds_p_semantic.builder import (
    _archive_contracts,
    _control_record,
    _enhance_assignment_exclusions,
    _file_sha256,
    _filter_semantic_groups,
    _fsync_directory,
    _pair_record,
    _partition_identity,
    _progress,
    _seed_identity,
    _source_record,
    _write_metadata,
    _write_tree,
)
from apm.data.text.tinyworlds_p_semantic.contracts import record_sha256
from apm.data.text.tinyworlds_p_semantic.partitioning import (
    SemanticPartitionGateError,
    audit_semantic_world_cells,
    pair_world_controls,
)
from apm.data.text.tinyworlds_p_semantic.v4_catalog import load_v4_semantic_catalog
from apm.data.text.tinyworlds_p_semantic.v4_contracts import V4_BENCHMARK_ID
from apm.data.text.tinyworlds_p_semantic.v5_partition_contracts import (
    V5_BENCHMARK_ID,
    V5_SEMANTIC_PARTITION_PRESET,
    V5SemanticPartitionFailure,
)
from apm.data.text.tinyworlds_p_semantic.v5_partition_failure import (
    load_v5_partition_failure,
    load_v5_partition_failure_evidence,
)
from apm.data.text.tinyworlds_p_semantic.v6_partition_contracts import (
    V6_BENCHMARK_ID,
    V6_FEASIBILITY_FAILURE_REASON,
    V6_PARENT_CATALOG_SHA256,
    V6_PARENT_PARTITION_FAILURE_SHA256,
    V6_PARTITION_FORMAT,
    V6_PARTITION_SCHEMA_VERSION,
    V6_PARTITION_TREE_FORMAT,
    V6CandidateFeasibility,
    V6SemanticPartitionArtifact,
    V6SemanticPartitionInputs,
    V6SemanticPartitionPreset,
)
from apm.data.text.tinyworlds_p_semantic.v6_topology import (
    ranked_balance_candidates,
    selected_feasibility,
    topology_selection_from_feasibility,
    world_cells_from_topology_selection,
)


def build_v6_partition(
    inputs: V6SemanticPartitionInputs,
    preset: V6SemanticPartitionPreset,
) -> V6SemanticPartitionArtifact:
    """Build, publish, and strictly reload one feasible semantic-v6 partition."""
    if type(inputs) is not V6SemanticPartitionInputs:
        raise TypeError("semantic-v6 partition requires its dedicated inputs")
    if type(preset) is not V6SemanticPartitionPreset:
        raise TypeError("semantic-v6 partition requires its dedicated preset")
    catalog = load_v4_semantic_catalog(inputs.semantic_catalog_directory)
    if catalog.catalog_sha256 != V6_PARENT_CATALOG_SHA256:
        raise ValueError("semantic-v6 requires the canonical successful v4 catalog")
    parent = load_v5_partition_failure(inputs.parent_partition_failure_directory)
    if (
        parent.failure_sha256 != V6_PARENT_PARTITION_FAILURE_SHA256
        or parent.catalog_sha256 != catalog.catalog_sha256
    ):
        raise ValueError("semantic-v6 requires the canonical v5 partition failure")
    parent_evidence = load_v5_partition_failure_evidence(parent)
    core_sources = {
        name: parent_evidence.sources[name]
        for name in ("archive", "semantic_catalog", "tokenizer")
    }
    if _source_record(inputs, catalog) != core_sources:
        raise ValueError("semantic-v6 archive or tokenizer differs from the v5 parent")
    if parent_evidence.partition_preset != V5_SEMANTIC_PARTITION_PRESET.as_record():
        raise ValueError("semantic-v6 parent does not use the canonical v5 settings")
    if (
        preset.v4_shape.as_record()
        != parent_evidence.parent_v4_evidence.partition_preset
    ):
        raise ValueError("semantic-v6 downstream settings differ from the parent")

    parent_source = _parent_source(parent)
    archive_inputs, archive_preset = _archive_contracts(inputs, preset, catalog)
    ingest = build_archive_ingest(
        archive_inputs,
        archive_preset,
        NORMALIZATION_IDENTITY,
    )
    seed_identity = _seed_identity(
        inputs,
        preset,
        catalog,
        benchmark_id=V6_BENCHMARK_ID,
        additional_sources=parent_source,
    )
    filtered_path, exclusions = _filter_semantic_groups(ingest, catalog, inputs)
    if exclusions != parent_evidence.semantic_exclusions:
        raise ValueError("semantic-v6 archive exclusions differ from the v5 parent")
    filtered_ingest = replace(ingest, groups_path=filtered_path)
    adjective_buckets = parent_evidence.parent_v4_evidence.adjective_buckets
    noun_lookup = catalog.word_cluster("noun")
    verb_lookup = catalog.word_cluster("verb")
    adjective_lookup = bucket_word_lookup(adjective_buckets)
    allocation_factory = lambda: archive_builder._iter_allocation_groups(
        filtered_path,
        noun_lookup,
        verb_lookup,
        adjective_lookup,
    )

    _progress(inputs, "topology", 0, 1, "replaying the complete parent topology audit")
    parent_audit = audit_semantic_world_cells(
        allocation_factory(),
        catalog,
        parent_evidence.parent_v4_failure.seed_identity_sha256,
        preset,
        benchmark_id=V4_BENCHMARK_ID,
    )
    parent_candidates = tuple(
        candidate.as_record(parent_audit.median_tolerance)
        for candidate in parent_audit.candidates
    )
    if parent_candidates != parent_evidence.parent_v4_evidence.topology_candidates:
        raise ValueError("semantic-v6 archive replay differs from the parent audit")
    ranked = ranked_balance_candidates(parent_candidates, seed_identity)
    _progress(
        inputs,
        "topology",
        1,
        1,
        f"reproduced the audit and found {len(ranked)} balanced candidates",
    )
    feasibility = _measure_all_candidates(
        inputs,
        preset,
        archive_inputs,
        archive_preset,
        allocation_factory,
        ranked,
        seed_identity,
    )
    topology_selection = topology_selection_from_feasibility(
        parent_candidates,
        feasibility,
        seed_identity,
        preset,
    )
    if topology_selection["selected"] is None:
        from apm.data.text.tinyworlds_p_semantic.v6_partition_failure import (
            publish_v6_partition_failure,
        )

        failure = publish_v6_partition_failure(
            inputs,
            preset,
            catalog,
            parent,
            seed_identity,
            exclusions,
            topology_selection,
        )
        raise SemanticPartitionGateError(
            f"{V6_FEASIBILITY_FAILURE_REASON}; failure audit: {failure.root}"
        )
    cells = world_cells_from_topology_selection(topology_selection)
    visibility = require_component_visibility(
        allocation_factory(),
        cells,
        preset.minimum_component_outside_groups,
    )

    final_directory = inputs.temporary_directory / "selected-allocation"
    final_directory.mkdir()
    final_inputs = replace(
        archive_inputs,
        temporary_directory=final_directory,
    )
    _progress(inputs, "splits", 0, 4, "rebuilding the selected exact allocation")
    allocation = archive_builder._prepare_allocations(
        final_inputs,
        archive_preset,
        filtered_ingest,
        cells,
        allocation_factory,
        seed_identity,
    )
    selected_evidence = selected_feasibility(topology_selection)
    split_assignments_sha256 = _file_sha256(
        final_directory / "eligible-assignments.jsonl"
    )
    controls_sha256 = record_sha256(
        [_control_record(control) for control in allocation.controls]
    )
    if (
        split_assignments_sha256 != selected_evidence.split_assignments_sha256
        or controls_sha256 != selected_evidence.controls_sha256
    ):
        raise RuntimeError("semantic-v6 selected allocation differs from its screen")

    assignments_path = _enhance_assignment_exclusions(
        allocation.assignments_path,
        filtered_path,
        final_directory / "semantic-assignments.jsonl",
    )
    _progress(inputs, "pairing", 0, 1, "pairing both comparison arms one-to-one")
    pairings = pair_world_controls(
        allocation.allocation_groups_by_evaluation_domain,
        cells,
        allocation.controls,
        seed_identity,
        benchmark_id=V6_BENCHMARK_ID,
    )
    _progress(inputs, "pairing", 1, 1, "semantic-v6 pairings passed")
    assignments_sha256 = _file_sha256(assignments_path)
    pairings_sha256 = record_sha256([_pair_record(item) for item in pairings])
    partition_sha256 = _partition_identity(
        inputs,
        preset,
        catalog,
        adjective_buckets,
        cells,
        allocation.controls,
        assignments_sha256,
        pairings_sha256,
        benchmark_id=V6_BENCHMARK_ID,
        additional_sources=parent_source,
    )
    target = inputs.output_root / partition_sha256
    if target.exists():
        raise FileExistsError(f"semantic-v6 partition already exists: {target}")
    publication = final_directory / "publication"
    if publication.exists():
        raise FileExistsError(f"semantic-v6 staging path exists: {publication}")
    publication.mkdir(parents=True)
    (publication / "shards").mkdir()
    (publication / "indexes").mkdir()
    (publication / "manifests").mkdir()
    shutil.copyfile(assignments_path, publication / "assignments.jsonl")
    shutil.copytree(
        catalog.root,
        publication / "semantic-catalog" / catalog.catalog_sha256,
    )
    shutil.copytree(
        parent.root,
        publication / "parent-partition-failure" / parent.failure_sha256,
    )
    retained_records = exclusions["retained_occurrences"]
    _progress(
        inputs,
        "shards",
        0,
        retained_records,
        "publishing exact semantic-v6 archive bytes and tokens",
    )
    shards, occurrence_counts = archive_builder._write_shards_and_indexes(
        archive_inputs,
        archive_preset,
        filtered_ingest,
        assignments_path,
        allocation.control_group_owners,
        publication,
        progress_total_occurrences=retained_records,
    )
    _progress(
        inputs,
        "shards",
        retained_records,
        retained_records,
        "semantic-v6 shards and indexes are complete",
    )
    _write_metadata(
        publication,
        inputs,
        preset,
        catalog,
        ingest,
        exclusions,
        seed_identity,
        partition_sha256,
        assignments_sha256,
        adjective_buckets,
        cells,
        allocation.controls,
        pairings,
        allocation.split_counts,
        visibility,
        shards,
        benchmark_id=V6_BENCHMARK_ID,
        partition_format=V6_PARTITION_FORMAT,
        schema_version=V6_PARTITION_SCHEMA_VERSION,
        additional_sources=parent_source,
        topology_selection=topology_selection,
    )
    archive_builder._write_manifests(
        publication,
        allocation.split_counts,
        allocation.controls,
        occurrence_counts,
    )
    _progress(inputs, "publish", 0, 1, "hashing and strictly reloading semantic-v6")
    tree_path = _write_tree(
        publication,
        partition_sha256,
        tree_format=V6_PARTITION_TREE_FORMAT,
        schema_version=V6_PARTITION_SCHEMA_VERSION,
    )
    tree_sha256 = _file_sha256(tree_path)
    inputs.output_root.mkdir(parents=True, exist_ok=True)
    os.rename(publication, target)
    _fsync_directory(inputs.output_root)
    from apm.data.text.tinyworlds_p_semantic.v6_partition_artifact import (
        load_v6_partition,
    )

    restored = load_v6_partition(target)
    if restored.manifest_sha256 != tree_sha256:
        raise RuntimeError("semantic-v6 tree changed during strict reload")
    _progress(inputs, "publish", 1, 1, "strict semantic-v6 reload passed")
    return restored


def _measure_all_candidates(
    inputs: V6SemanticPartitionInputs,
    preset: V6SemanticPartitionPreset,
    archive_inputs: PartitionInputs,
    archive_preset: PartitionPreset,
    allocation_factory: Callable[[], Iterable[AllocationGroup]],
    ranked: tuple[dict[str, object], ...],
    seed_identity: str,
) -> tuple[V6CandidateFeasibility, ...]:
    """Measure every balanced candidate with bounded parallel exact allocators."""
    feasibility_root = inputs.temporary_directory / "candidate-feasibility"
    feasibility_root.mkdir()
    parallel_candidates = min(4, len(ranked), max(1, preset.worker_count))
    candidate_workers = max(1, preset.worker_count // parallel_candidates)
    _progress(
        inputs,
        "feasibility",
        0,
        len(ranked),
        f"measuring {len(ranked)} balanced layouts with exact comparisons",
    )

    def measure(rank: int) -> V6CandidateFeasibility:
        candidate = ranked[rank]
        cells = world_cells_from_topology_selection({"selected": candidate})
        candidate_directory = feasibility_root / f"candidate-{rank:03d}"
        candidate_directory.mkdir()
        candidate_inputs = replace(
            archive_inputs,
            temporary_directory=candidate_directory,
            progress=None,
        )
        candidate_preset = replace(
            archive_preset,
            worker_count=candidate_workers,
        )
        try:
            prepared = archive_builder._prepare_control_feasibility(
                candidate_inputs,
                candidate_preset,
                cells,
                allocation_factory,
                seed_identity,
            )
        except ArchivePartitionGateError as error:
            split_path = candidate_directory / "eligible-assignments.jsonl"
            if not split_path.is_file():
                raise
            return V6CandidateFeasibility(
                semantic_rank=rank,
                cells=tuple((cell.noun_bucket, cell.verb_bucket) for cell in cells),
                split_assignments_sha256=_file_sha256(split_path),
                control_feasible=False,
                controls_sha256=None,
                failure_reason=str(error),
            )
        return V6CandidateFeasibility(
            semantic_rank=rank,
            cells=tuple((cell.noun_bucket, cell.verb_bucket) for cell in cells),
            split_assignments_sha256=_file_sha256(
                prepared.split_assignments_path
            ),
            control_feasible=True,
            controls_sha256=record_sha256(
                [_control_record(control) for control in prepared.controls]
            ),
            failure_reason=None,
        )

    completed: dict[int, V6CandidateFeasibility] = {}
    with ThreadPoolExecutor(max_workers=parallel_candidates) as executor:
        futures = {
            executor.submit(measure, rank): rank for rank in range(len(ranked))
        }
        for count, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            completed[result.semantic_rank] = result
            outcome = "passed" if result.control_feasible else "failed"
            _progress(
                inputs,
                "feasibility",
                count,
                len(ranked),
                f"layout rank {result.semantic_rank} {outcome} exact comparisons",
            )
    return tuple(completed[rank] for rank in range(len(ranked)))


def _parent_source(
    parent: V5SemanticPartitionFailure,
) -> dict[str, object]:
    return {
        "parent_partition_failure": {
            "benchmark_id": V5_BENCHMARK_ID,
            "failure_sha256": parent.failure_sha256,
            "tree_sha256": _file_sha256(parent.root / "tree.json"),
        }
    }


__all__ = ["build_v6_partition"]

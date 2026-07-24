"""Archive-native construction of the balance-eligible semantic-v5 partition."""

from __future__ import annotations

from dataclasses import replace
import os
import shutil

from apm.data.text.tinyworlds_p import builder as archive_builder
from apm.data.text.tinyworlds_p.archive_ingest import build_archive_ingest
from apm.data.text.tinyworlds_p.contracts import NORMALIZATION_IDENTITY
from apm.data.text.tinyworlds_p.partitioning import (
    PartitionGateError as ArchivePartitionGateError,
    bucket_word_lookup,
    require_component_visibility,
)
from apm.data.text.tinyworlds_p_semantic.builder import (
    _archive_contracts,
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
    retie_semantic_topology_audit,
    select_balance_eligible_semantic_world_cells,
)
from apm.data.text.tinyworlds_p_semantic.v4_catalog import load_v4_semantic_catalog
from apm.data.text.tinyworlds_p_semantic.v4_contracts import V4_BENCHMARK_ID
from apm.data.text.tinyworlds_p_semantic.v4_partition_failure import (
    V4SemanticPartitionFailure,
    load_v4_partition_failure,
    load_v4_partition_failure_evidence,
)
from apm.data.text.tinyworlds_p_semantic.v5_partition_contracts import (
    V5_BENCHMARK_ID,
    V5_PARENT_CATALOG_SHA256,
    V5_PARENT_PARTITION_FAILURE_SHA256,
    V5_PARTITION_FORMAT,
    V5_PARTITION_SCHEMA_VERSION,
    V5_PARTITION_TREE_FORMAT,
    V5SemanticPartitionArtifact,
    V5SemanticPartitionInputs,
    V5SemanticPartitionPreset,
)
from apm.data.text.tinyworlds_p_semantic.v5_topology import (
    topology_selection_from_parent_candidates,
)


def build_v5_partition(
    inputs: V5SemanticPartitionInputs,
    preset: V5SemanticPartitionPreset,
) -> V5SemanticPartitionArtifact:
    """Build, publish, and strictly reload one balance-eligible v5 partition."""
    if type(inputs) is not V5SemanticPartitionInputs:
        raise TypeError("semantic-v5 partition requires its dedicated inputs")
    if type(preset) is not V5SemanticPartitionPreset:
        raise TypeError("semantic-v5 partition requires its dedicated preset")
    catalog = load_v4_semantic_catalog(inputs.semantic_catalog_directory)
    if catalog.catalog_sha256 != V5_PARENT_CATALOG_SHA256:
        raise ValueError("semantic-v5 requires the canonical successful v4 catalog")
    parent = load_v4_partition_failure(inputs.parent_partition_failure_directory)
    if (
        parent.failure_sha256 != V5_PARENT_PARTITION_FAILURE_SHA256
        or parent.catalog_sha256 != catalog.catalog_sha256
    ):
        raise ValueError("semantic-v5 requires the canonical v4 partition failure")
    parent_evidence = load_v4_partition_failure_evidence(parent)
    if _source_record(inputs, catalog) != parent_evidence.sources:
        raise ValueError("semantic-v5 archive or tokenizer differs from the v4 parent")
    if preset.v4_shape.as_record() != parent_evidence.partition_preset:
        raise ValueError("semantic-v5 downstream settings differ from the v4 parent")
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
        benchmark_id=V5_BENCHMARK_ID,
        additional_sources=parent_source,
    )
    filtered_path, exclusions = _filter_semantic_groups(ingest, catalog, inputs)
    if exclusions != parent_evidence.semantic_exclusions:
        raise ValueError("semantic-v5 archive exclusions differ from the v4 parent")
    filtered_ingest = replace(ingest, groups_path=filtered_path)
    adjective_buckets = parent_evidence.adjective_buckets
    noun_lookup = catalog.word_cluster("noun")
    verb_lookup = catalog.word_cluster("verb")
    adjective_lookup = bucket_word_lookup(adjective_buckets)
    allocation_factory = lambda: archive_builder._iter_allocation_groups(
        filtered_path,
        noun_lookup,
        verb_lookup,
        adjective_lookup,
    )
    _progress(inputs, "topology", 0, 3, "replaying the complete v4 topology audit")
    parent_audit = audit_semantic_world_cells(
        allocation_factory(),
        catalog,
        parent.seed_identity_sha256,
        preset,
        benchmark_id=V4_BENCHMARK_ID,
    )
    parent_candidates = tuple(
        candidate.as_record(parent_audit.median_tolerance)
        for candidate in parent_audit.candidates
    )
    if parent_candidates != parent_evidence.topology_candidates:
        raise ValueError("semantic-v5 archive replay differs from the v4 topology audit")
    _progress(inputs, "topology", 1, 3, "the v4 topology audit reproduced exactly")
    v5_audit = retie_semantic_topology_audit(
        parent_audit,
        seed_identity,
        V5_BENCHMARK_ID,
    )
    cells = select_balance_eligible_semantic_world_cells(v5_audit)
    selected = v5_audit.median_feasible_candidates[0]
    topology_selection = topology_selection_from_parent_candidates(
        parent_evidence.topology_candidates,
        seed_identity,
        preset,
    )
    if topology_selection["selected"] != selected.as_record(
        v5_audit.median_tolerance
    ):
        raise ValueError("semantic-v5 in-memory and parent-record selections disagree")
    _progress(
        inputs,
        "topology",
        2,
        3,
        "selected the semantic leader among "
        f"{len(v5_audit.median_feasible_candidates)} balanced candidates",
    )
    visibility = require_component_visibility(
        allocation_factory(),
        cells,
        preset.minimum_component_outside_groups,
    )
    _progress(inputs, "topology", 3, 3, "v5 topology and visibility gates passed")
    try:
        allocation = archive_builder._prepare_allocations(
            archive_inputs,
            archive_preset,
            filtered_ingest,
            cells,
            allocation_factory,
            seed_identity,
        )
    except ArchivePartitionGateError as error:
        assignments_path = inputs.temporary_directory / "assignments.jsonl"
        if not assignments_path.is_file():
            raise
        from apm.data.text.tinyworlds_p_semantic.v5_partition_failure import (
            publish_v5_partition_failure,
        )

        failure = publish_v5_partition_failure(
            inputs,
            preset,
            catalog,
            parent,
            seed_identity,
            exclusions,
            topology_selection,
            assignments_path,
            str(error),
        )
        raise SemanticPartitionGateError(
            f"{error}; failure audit: {failure.root}",
            v5_audit,
        ) from error
    assignments_path = _enhance_assignment_exclusions(
        allocation.assignments_path,
        filtered_path,
        inputs.temporary_directory / "semantic-assignments.jsonl",
    )
    _progress(inputs, "pairing", 0, 1, "pairing both control arms one-to-one")
    pairings = pair_world_controls(
        allocation.allocation_groups_by_evaluation_domain,
        cells,
        allocation.controls,
        seed_identity,
        benchmark_id=V5_BENCHMARK_ID,
    )
    _progress(inputs, "pairing", 1, 1, "v5 control pairings passed")
    assignments_sha = _file_sha256(assignments_path)
    pairings_sha = record_sha256([_pair_record(item) for item in pairings])
    partition_sha = _partition_identity(
        inputs,
        preset,
        catalog,
        adjective_buckets,
        cells,
        allocation.controls,
        assignments_sha,
        pairings_sha,
        benchmark_id=V5_BENCHMARK_ID,
        additional_sources=parent_source,
    )
    target = inputs.output_root / partition_sha
    if target.exists():
        raise FileExistsError(f"semantic-v5 partition already exists: {target}")
    publication = inputs.temporary_directory / "publication"
    if publication.exists():
        raise FileExistsError(f"semantic-v5 staging path exists: {publication}")
    publication.mkdir(parents=True)
    (publication / "shards").mkdir()
    (publication / "indexes").mkdir()
    (publication / "manifests").mkdir()
    shutil.copyfile(assignments_path, publication / "assignments.jsonl")
    embedded_catalog = publication / "semantic-catalog" / catalog.catalog_sha256
    shutil.copytree(catalog.root, embedded_catalog)
    embedded_parent = publication / "parent-partition-failure" / parent.failure_sha256
    shutil.copytree(parent.root, embedded_parent)
    retained_records = exclusions["retained_occurrences"]
    _progress(
        inputs,
        "shards",
        0,
        retained_records,
        "publishing exact v5 archive bytes and tokens",
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
        "semantic-v5 shards and indexes complete",
    )
    _write_metadata(
        publication,
        inputs,
        preset,
        catalog,
        ingest,
        exclusions,
        seed_identity,
        partition_sha,
        assignments_sha,
        adjective_buckets,
        cells,
        allocation.controls,
        pairings,
        allocation.split_counts,
        visibility,
        shards,
        benchmark_id=V5_BENCHMARK_ID,
        partition_format=V5_PARTITION_FORMAT,
        schema_version=V5_PARTITION_SCHEMA_VERSION,
        additional_sources=parent_source,
        topology_selection=topology_selection,
    )
    archive_builder._write_manifests(
        publication,
        allocation.split_counts,
        allocation.controls,
        occurrence_counts,
    )
    _progress(inputs, "publish", 0, 1, "hashing and strictly reloading v5")
    tree_path = _write_tree(
        publication,
        partition_sha,
        tree_format=V5_PARTITION_TREE_FORMAT,
        schema_version=V5_PARTITION_SCHEMA_VERSION,
    )
    tree_sha = _file_sha256(tree_path)
    inputs.output_root.mkdir(parents=True, exist_ok=True)
    os.rename(publication, target)
    _fsync_directory(inputs.output_root)
    from apm.data.text.tinyworlds_p_semantic.v5_partition_artifact import (
        load_v5_partition,
    )

    restored = load_v5_partition(target)
    if restored.manifest_sha256 != tree_sha:
        raise RuntimeError("semantic-v5 tree changed during strict reload")
    _progress(inputs, "publish", 1, 1, "strict semantic-v5 reload passed")
    return restored


def _parent_source(
    parent: V4SemanticPartitionFailure,
) -> dict[str, object]:
    return {
        "parent_partition_failure": {
            "benchmark_id": V4_BENCHMARK_ID,
            "failure_sha256": parent.failure_sha256,
            "tree_sha256": _file_sha256(parent.root / "tree.json"),
        }
    }


__all__ = ["build_v5_partition"]

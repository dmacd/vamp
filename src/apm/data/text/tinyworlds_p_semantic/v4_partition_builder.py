"""Archive-native construction and publication of semantic-v4 partitions."""

from __future__ import annotations

from dataclasses import replace
import os
import shutil

from apm.data.text.tinyworlds_p import builder as archive_builder
from apm.data.text.tinyworlds_p.archive_ingest import build_archive_ingest
from apm.data.text.tinyworlds_p.contracts import NORMALIZATION_IDENTITY
from apm.data.text.tinyworlds_p.partitioning import (
    bucket_word_lookup,
    require_component_visibility,
)
from apm.data.text.tinyworlds_p_semantic.builder import (
    _adjective_buckets,
    _archive_contracts,
    _enhance_assignment_exclusions,
    _file_sha256,
    _filter_semantic_groups,
    _fsync_directory,
    _pair_record,
    _partition_identity,
    _progress,
    _seed_identity,
    _write_metadata,
    _write_tree,
)
from apm.data.text.tinyworlds_p_semantic.contracts import record_sha256
from apm.data.text.tinyworlds_p_semantic.partitioning import (
    SemanticPartitionGateError,
    pair_world_controls,
    select_semantic_world_cells,
)
from apm.data.text.tinyworlds_p_semantic.v4_catalog import load_v4_semantic_catalog
from apm.data.text.tinyworlds_p_semantic.v4_contracts import V4_BENCHMARK_ID
from apm.data.text.tinyworlds_p_semantic.v4_partition_contracts import (
    V4_PARTITION_FORMAT,
    V4_PARTITION_SCHEMA_VERSION,
    V4_PARTITION_TREE_FORMAT,
    V4SemanticPartitionArtifact,
    V4SemanticPartitionInputs,
    V4SemanticPartitionPreset,
)


def build_v4_partition(
    inputs: V4SemanticPartitionInputs,
    preset: V4SemanticPartitionPreset,
) -> V4SemanticPartitionArtifact:
    """Build, atomically publish, and strictly reload one v4 partition."""
    if type(inputs) is not V4SemanticPartitionInputs:
        raise TypeError("semantic-v4 partition requires its dedicated inputs")
    if type(preset) is not V4SemanticPartitionPreset:
        raise TypeError("semantic-v4 partition requires its dedicated preset")
    catalog = load_v4_semantic_catalog(inputs.semantic_catalog_directory)
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
        benchmark_id=V4_BENCHMARK_ID,
    )
    filtered_path, exclusions = _filter_semantic_groups(ingest, catalog, inputs)
    filtered_ingest = replace(ingest, groups_path=filtered_path)
    adjective_buckets = _adjective_buckets(
        filtered_path,
        archive_preset,
        seed_identity,
    )
    noun_lookup = catalog.word_cluster("noun")
    verb_lookup = catalog.word_cluster("verb")
    adjective_lookup = bucket_word_lookup(adjective_buckets)
    allocation_factory = lambda: archive_builder._iter_allocation_groups(
        filtered_path,
        noun_lookup,
        verb_lookup,
        adjective_lookup,
    )
    _progress(
        inputs,
        "topology",
        0,
        2,
        "scoring v4 topology without model losses",
    )
    try:
        cells = select_semantic_world_cells(
            allocation_factory(),
            catalog,
            seed_identity,
            preset,
            benchmark_id=V4_BENCHMARK_ID,
        )
    except SemanticPartitionGateError as error:
        if error.audit is None:
            raise
        from apm.data.text.tinyworlds_p_semantic.v4_partition_failure import (
            publish_v4_partition_failure,
        )

        failure = publish_v4_partition_failure(
            inputs,
            preset,
            catalog,
            seed_identity,
            adjective_buckets,
            exclusions,
            error.audit,
            str(error),
        )
        raise SemanticPartitionGateError(
            f"{error}; failure audit: {failure.root}",
            error.audit,
        ) from error
    _progress(inputs, "topology", 1, 2, "auditing component visibility")
    visibility = require_component_visibility(
        allocation_factory(),
        cells,
        preset.minimum_component_outside_groups,
    )
    _progress(inputs, "topology", 2, 2, "v4 topology and visibility gates passed")
    allocation = archive_builder._prepare_allocations(
        archive_inputs,
        archive_preset,
        filtered_ingest,
        cells,
        allocation_factory,
        seed_identity,
    )
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
        benchmark_id=V4_BENCHMARK_ID,
    )
    _progress(inputs, "pairing", 1, 1, "v4 control pairings passed")
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
        benchmark_id=V4_BENCHMARK_ID,
    )
    target = inputs.output_root / partition_sha
    if target.exists():
        raise FileExistsError(f"semantic-v4 partition already exists: {target}")
    publication = inputs.temporary_directory / "publication"
    if publication.exists():
        raise FileExistsError(f"semantic-v4 staging path exists: {publication}")
    publication.mkdir(parents=True)
    (publication / "shards").mkdir()
    (publication / "indexes").mkdir()
    (publication / "manifests").mkdir()
    shutil.copyfile(assignments_path, publication / "assignments.jsonl")
    embedded_catalog = publication / "semantic-catalog" / catalog.catalog_sha256
    shutil.copytree(catalog.root, embedded_catalog)
    retained_records = exclusions["retained_occurrences"]
    _progress(
        inputs,
        "shards",
        0,
        retained_records,
        "publishing exact v4 archive bytes and tokens",
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
        "semantic-v4 shards and indexes complete",
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
        benchmark_id=V4_BENCHMARK_ID,
        partition_format=V4_PARTITION_FORMAT,
        schema_version=V4_PARTITION_SCHEMA_VERSION,
    )
    archive_builder._write_manifests(
        publication,
        allocation.split_counts,
        allocation.controls,
        occurrence_counts,
    )
    _progress(inputs, "publish", 0, 1, "hashing and strictly reloading v4")
    tree_path = _write_tree(
        publication,
        partition_sha,
        tree_format=V4_PARTITION_TREE_FORMAT,
        schema_version=V4_PARTITION_SCHEMA_VERSION,
    )
    tree_sha = _file_sha256(tree_path)
    inputs.output_root.mkdir(parents=True, exist_ok=True)
    os.rename(publication, target)
    _fsync_directory(inputs.output_root)
    from apm.data.text.tinyworlds_p_semantic.v4_partition_artifact import (
        load_v4_partition,
    )

    restored = load_v4_partition(target)
    if restored.manifest_sha256 != tree_sha:
        raise RuntimeError("semantic-v4 tree changed during strict reload")
    _progress(inputs, "publish", 1, 1, "strict semantic-v4 reload passed")
    return restored


__all__ = ["build_v4_partition"]

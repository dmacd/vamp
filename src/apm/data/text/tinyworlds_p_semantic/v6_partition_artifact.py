"""Strict independent loading of exact-control-feasible semantic-v6."""

from __future__ import annotations

from pathlib import Path

from apm.data.text.tinyworlds_p import artifact as archive_artifact
from apm.data.text.tinyworlds_p_semantic import artifact as shared
from apm.data.text.tinyworlds_p_semantic.builder import (
    _bucket_record,
    _cell_record,
    _control_record,
    _pair_record,
)
from apm.data.text.tinyworlds_p_semantic.contracts import record_sha256
from apm.data.text.tinyworlds_p_semantic.v4_catalog import (
    V4SemanticCatalogError,
    load_v4_semantic_catalog,
)
from apm.data.text.tinyworlds_p_semantic.v5_partition_contracts import V5_BENCHMARK_ID
from apm.data.text.tinyworlds_p_semantic.v5_partition_failure import (
    V5SemanticPartitionFailureError,
    load_v5_partition_failure,
    load_v5_partition_failure_evidence,
)
from apm.data.text.tinyworlds_p_semantic.v6_partition_contracts import (
    V6_BENCHMARK_ID,
    V6_PARENT_CATALOG_SHA256,
    V6_PARENT_PARTITION_FAILURE_SHA256,
    V6_PARTITION_FORMAT,
    V6_PARTITION_SCHEMA_VERSION,
    V6_PARTITION_TREE_FORMAT,
    V6SemanticPartitionArtifact,
    V6SemanticPartitionPreset,
)
from apm.data.text.tinyworlds_p_semantic.v6_topology import (
    V6TopologyEvidenceError,
    selected_feasibility,
    validate_topology_selection,
)


PartitionArtifactError = shared.PartitionArtifactError


def load_v6_partition(path: str | Path) -> V6SemanticPartitionArtifact:
    """Authenticate the complete v6 tree, feasibility, and archive data."""
    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        raise PartitionArtifactError("semantic-v6 partition must be a regular directory")
    tree_path = root / "tree.json"
    tree = shared._load_json(tree_path, "semantic-v6 partition tree")
    if set(tree) != {"files", "format", "partition_sha256", "schema_version"}:
        raise PartitionArtifactError("semantic-v6 tree fields changed")
    if (
        tree["format"] != V6_PARTITION_TREE_FORMAT
        or tree["schema_version"] != V6_PARTITION_SCHEMA_VERSION
    ):
        raise PartitionArtifactError("unsupported semantic-v6 partition tree")
    partition_sha256 = shared._text(tree, "partition_sha256")
    if root.name != partition_sha256:
        raise PartitionArtifactError("semantic-v6 partition directory identity changed")
    shared._validate_tree(root, tree)
    partition = shared._load_json(root / "partition.json", "semantic-v6 identity")
    required = {
        "assignments_sha256",
        "benchmark_id",
        "eos_token_id",
        "format",
        "normalization",
        "pad_token_id",
        "partition_sha256",
        "preset",
        "schema_version",
        "seed_identity_sha256",
        "sources",
    }
    if set(partition) != required:
        raise PartitionArtifactError("semantic-v6 identity fields changed")
    if (
        partition["benchmark_id"] != V6_BENCHMARK_ID
        or partition["format"] != V6_PARTITION_FORMAT
        or partition["schema_version"] != V6_PARTITION_SCHEMA_VERSION
        or partition["partition_sha256"] != partition_sha256
    ):
        raise PartitionArtifactError("semantic-v6 partition identity changed")
    sources = shared._mapping(partition, "sources")
    if set(sources) != {
        "archive",
        "parent_partition_failure",
        "semantic_catalog",
        "tokenizer",
    }:
        raise PartitionArtifactError("semantic-v6 source set changed")
    if shared._load_json(root / "sources.json", "semantic-v6 sources") != sources:
        raise PartitionArtifactError("semantic-v6 source files disagree")
    if (
        shared._load_json(root / "normalization.json", "semantic-v6 normalization")
        != partition["normalization"]
    ):
        raise PartitionArtifactError("semantic-v6 normalization files disagree")
    try:
        archive_identity = archive_artifact._source_identity(
            shared._mapping(sources, "archive")
        )
        tokenizer_identity = archive_artifact._tokenizer_identity(
            shared._mapping(sources, "tokenizer")
        )
        normalization = archive_artifact._normalization(
            shared._mapping(partition, "normalization")
        )
        preset = _preset(shared._mapping(partition, "preset"))
    except (TypeError, ValueError, archive_artifact.PartitionArtifactError) as error:
        raise PartitionArtifactError("semantic-v6 source or preset is invalid") from error
    catalog_source = shared._mapping(sources, "semantic_catalog")
    if set(catalog_source) != {
        "catalog_sha256",
        "encoder_identity_sha256",
        "evidence_sha256",
    }:
        raise PartitionArtifactError("semantic-v6 catalog source fields changed")
    catalog_sha256 = shared._text(catalog_source, "catalog_sha256")
    if catalog_sha256 != V6_PARENT_CATALOG_SHA256:
        raise PartitionArtifactError("semantic-v6 parent catalog identity changed")
    try:
        catalog = load_v4_semantic_catalog(
            root / "semantic-catalog" / catalog_sha256
        )
    except V4SemanticCatalogError as error:
        raise PartitionArtifactError("embedded v4 catalog failed authentication") from error
    if (
        catalog.encoder_identity.identity_sha256
        != shared._text(catalog_source, "encoder_identity_sha256")
        or catalog.evidence_sha256
        != shared._text(catalog_source, "evidence_sha256")
    ):
        raise PartitionArtifactError("semantic-v6 catalog source binding changed")
    parent_source = shared._mapping(sources, "parent_partition_failure")
    if (
        set(parent_source) != {"benchmark_id", "failure_sha256", "tree_sha256"}
        or parent_source.get("benchmark_id") != V5_BENCHMARK_ID
        or parent_source.get("failure_sha256")
        != V6_PARENT_PARTITION_FAILURE_SHA256
    ):
        raise PartitionArtifactError("semantic-v6 parent failure source changed")
    try:
        parent = load_v5_partition_failure(
            root
            / "parent-partition-failure"
            / V6_PARENT_PARTITION_FAILURE_SHA256
        )
    except V5SemanticPartitionFailureError as error:
        raise PartitionArtifactError("embedded v5 failure failed authentication") from error
    if (
        parent.catalog_sha256 != catalog.catalog_sha256
        or shared._file_sha256(parent.root / "tree.json")
        != shared._text(parent_source, "tree_sha256")
    ):
        raise PartitionArtifactError("semantic-v6 parent failure binding changed")
    parent_evidence = load_v5_partition_failure_evidence(parent)
    core_sources = {
        name: sources[name]
        for name in ("archive", "semantic_catalog", "tokenizer")
    }
    if core_sources != {
        name: parent_evidence.sources[name]
        for name in ("archive", "semantic_catalog", "tokenizer")
    }:
        raise PartitionArtifactError("semantic-v6 sources differ from the parent")
    if (
        preset.v4_shape.as_record()
        != parent_evidence.parent_v4_evidence.partition_preset
    ):
        raise PartitionArtifactError("semantic-v6 settings differ from the parent")
    bucket_payload = shared._load_json(
        root / "buckets.json",
        "semantic-v6 adjective buckets",
    )
    if (
        set(bucket_payload) != {"adjective_buckets", "catalog_sha256"}
        or bucket_payload["catalog_sha256"] != catalog_sha256
        or type(bucket_payload["adjective_buckets"]) is not list
    ):
        raise PartitionArtifactError("semantic-v6 bucket/catalog binding changed")
    adjective_buckets = tuple(
        shared._word_bucket(item) for item in bucket_payload["adjective_buckets"]
    )
    if adjective_buckets != parent_evidence.parent_v4_evidence.adjective_buckets:
        raise PartitionArtifactError("semantic-v6 adjective mapping differs from parent")
    topology = shared._load_json(root / "topology.json", "semantic-v6 topology")
    if set(topology) != {"cells"} or type(topology["cells"]) is not list:
        raise PartitionArtifactError("semantic-v6 topology fields changed")
    cells = tuple(shared._world_cell(item) for item in topology["cells"])
    shared._validate_topology(cells, catalog.config.cluster_count)
    control_payload = shared._load_json(root / "controls.json", "semantic-v6 controls")
    if (
        set(control_payload) != {"controls"}
        or type(control_payload["controls"]) is not list
    ):
        raise PartitionArtifactError("semantic-v6 controls fields changed")
    controls = tuple(shared._control(item) for item in control_payload["controls"])
    shared._validate_controls(controls)
    pairing_payload = shared._load_json(root / "pairings.json", "semantic-v6 pairings")
    if (
        set(pairing_payload) != {"pairings"}
        or type(pairing_payload["pairings"]) is not list
    ):
        raise PartitionArtifactError("semantic-v6 pairings fields changed")
    pairings = tuple(shared._pair(item) for item in pairing_payload["pairings"])
    audit = shared._load_json(root / "audit.json", "semantic-v6 partition audit")
    required_audit = {
        "archive_ingest",
        "component_visibility",
        "semantic_exclusions",
        "split_counts",
        "topology_selection",
    }
    if set(audit) != required_audit or type(audit["split_counts"]) is not list:
        raise PartitionArtifactError("semantic-v6 audit fields changed")
    if (
        shared._mapping(audit, "semantic_exclusions")
        != parent_evidence.semantic_exclusions
    ):
        raise PartitionArtifactError("semantic-v6 exclusions differ from parent")
    expected_seed = record_sha256(
        {
            "benchmark_id": V6_BENCHMARK_ID,
            "normalization": normalization.as_record(),
            "preset": preset.as_record(),
            "sources": sources,
        }
    )
    if partition["seed_identity_sha256"] != expected_seed:
        raise PartitionArtifactError("semantic-v6 seed identity changed")
    topology_selection = shared._mapping(audit, "topology_selection")
    try:
        validate_topology_selection(
            topology_selection,
            parent_evidence.parent_v4_evidence.topology_candidates,
            expected_seed,
            preset,
            cells,
        )
        selected = selected_feasibility(topology_selection)
    except V6TopologyEvidenceError as error:
        raise PartitionArtifactError("semantic-v6 topology evidence changed") from error
    if selected.controls_sha256 != record_sha256(
        [_control_record(item) for item in controls]
    ):
        raise PartitionArtifactError("semantic-v6 selected controls differ from screen")
    split_counts = tuple(shared._split_count(item) for item in audit["split_counts"])
    assignments_sha256 = shared._text(partition, "assignments_sha256")
    if shared._file_sha256(root / "assignments.jsonl") != assignments_sha256:
        raise PartitionArtifactError("semantic-v6 assignment checksum changed")
    assignment_index = shared._validate_assignments(
        root / "assignments.jsonl",
        catalog,
        cells,
        controls,
        split_counts,
        pairings,
        shared._mapping(audit, "semantic_exclusions"),
    )
    shared._validate_pairings(pairings, controls, cells, assignment_index)
    try:
        archive_artifact._validate_documents_against_assignments(root)
        archive_artifact._validate_document_storage_and_indexes(
            root,
            controls,
            shared._integer(partition, "eos_token_id"),
            tokenizer_identity.vocab_size,
        )
    except (ValueError, OSError) as error:
        raise PartitionArtifactError("semantic-v6 document/index proof failed") from error
    pairings_sha256 = record_sha256([_pair_record(item) for item in pairings])
    expected_partition_sha256 = record_sha256(
        {
            "adjective_buckets": [_bucket_record(item) for item in adjective_buckets],
            "assignments_sha256": assignments_sha256,
            "benchmark_id": V6_BENCHMARK_ID,
            "cells": [_cell_record(item) for item in cells],
            "controls": [_control_record(item) for item in controls],
            "normalization": normalization.as_record(),
            "pairings_sha256": pairings_sha256,
            "preset": preset.as_record(),
            "sources": sources,
        }
    )
    if expected_partition_sha256 != partition_sha256:
        raise PartitionArtifactError("semantic-v6 content identity is inconsistent")
    return V6SemanticPartitionArtifact(
        root=root.resolve(),
        partition_sha256=partition_sha256,
        manifest_sha256=shared._file_sha256(tree_path),
        archive_identity=archive_identity,
        tokenizer_identity=tokenizer_identity,
        semantic_catalog=catalog,
        parent_partition_failure=parent,
        normalization=normalization,
        preset=preset,
        cells=cells,
        controls=controls,
        pairings=pairings,
        split_counts=split_counts,
        pad_token_id=shared._integer(partition, "pad_token_id"),
        eos_token_id=shared._integer(partition, "eos_token_id"),
    )


def _preset(record: dict[str, object]) -> V6SemanticPartitionPreset:
    default = V6SemanticPartitionPreset()
    if set(record) != set(default.as_record()):
        raise PartitionArtifactError("semantic-v6 partition preset fields changed")
    return V6SemanticPartitionPreset(
        version=shared._text(record, "version"),
        parent_catalog_sha256=shared._text(record, "parent_catalog_sha256"),
        parent_partition_failure_sha256=shared._text(
            record,
            "parent_partition_failure_sha256",
        ),
        topology_selection_method=shared._text(record, "topology_selection_method"),
        public_seed=shared._integer(record, "public_seed"),
        worker_count=default.worker_count,
        run_record_count=default.run_record_count,
        shard_target_bytes=shared._integer(record, "shard_target_bytes"),
        batch_block_documents=shared._integer(record, "batch_block_documents"),
        context_length=shared._integer(record, "context_length"),
        batch_size=shared._integer(record, "batch_size"),
        minimum_role_coverage=shared._number(record, "minimum_role_coverage"),
        selected_cell_median_tolerance=shared._number(
            record,
            "selected_cell_median_tolerance",
        ),
        minimum_component_outside_groups=shared._integer(
            record,
            "minimum_component_outside_groups",
        ),
        world_split_weights=shared._integer_triple(record, "world_split_weights"),
        base_split_weights=shared._integer_triple(record, "base_split_weights"),
        control_token_tolerance=shared._number(record, "control_token_tolerance"),
        control_source_feature_tolerance=shared._number(
            record,
            "control_source_feature_tolerance",
        ),
        control_adjective_length_tolerance=shared._number(
            record,
            "control_adjective_length_tolerance",
        ),
        control_mean_length_tolerance=shared._number(
            record,
            "control_mean_length_tolerance",
        ),
    )


__all__ = ["PartitionArtifactError", "load_v6_partition"]

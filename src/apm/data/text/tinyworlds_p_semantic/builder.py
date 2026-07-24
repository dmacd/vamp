"""Archive-native construction and publication of semantic-v1 partitions."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil

from apm.data.text.tinyworlds_p import builder as archive_builder
from apm.data.text.tinyworlds_p.archive_ingest import (
    ArchiveIngestResult,
    build_archive_ingest,
    iter_archive_groups,
)
from apm.data.text.tinyworlds_p.contracts import (
    NORMALIZATION_IDENTITY,
    ControlSelection,
    PartitionInputs,
    PartitionPreset,
    ProgressEvent,
    SplitCount,
    WordBucket,
    WorldCell,
    canonical_record_bytes,
)
from apm.data.text.tinyworlds_p.partitioning import (
    AllocationGroup,
    balance_word_buckets,
    bucket_word_lookup,
    require_component_visibility,
)
from apm.data.text.tinyworlds_p_semantic.catalog import load_semantic_catalog
from apm.data.text.tinyworlds_p_semantic.contracts import (
    BENCHMARK_ID,
    PARTITION_FORMAT,
    SCHEMA_VERSION,
    ControlPair,
    SemanticCatalog,
    SemanticPartitionArtifact,
    SemanticPartitionInputs,
    SemanticPartitionPreset,
    canonical_json_bytes,
    record_sha256,
)
from apm.data.text.tinyworlds_p_semantic.evidence import is_construction_group
from apm.data.text.tinyworlds_p_semantic.partitioning import (
    pair_world_controls,
    select_semantic_world_cells,
)


def build_partition(
    inputs: SemanticPartitionInputs,
    preset: SemanticPartitionPreset,
) -> SemanticPartitionArtifact:
    """Build, atomically publish, and strictly reload one semantic partition."""
    if type(inputs) is not SemanticPartitionInputs or type(preset) is not SemanticPartitionPreset:
        raise TypeError("semantic partition requires its dedicated inputs and preset")
    catalog = load_semantic_catalog(inputs.semantic_catalog_directory)
    archive_inputs, archive_preset = _archive_contracts(inputs, preset, catalog)
    ingest = build_archive_ingest(archive_inputs, archive_preset, NORMALIZATION_IDENTITY)
    seed_identity = _seed_identity(inputs, preset, catalog)
    filtered_path, exclusions = _filter_semantic_groups(ingest, catalog, inputs)
    filtered_ingest = replace(ingest, groups_path=filtered_path)
    adjective_buckets = _adjective_buckets(filtered_path, archive_preset, seed_identity)
    noun_lookup = catalog.word_cluster("noun")
    verb_lookup = catalog.word_cluster("verb")
    adjective_lookup = bucket_word_lookup(adjective_buckets)
    allocation_factory = lambda: archive_builder._iter_allocation_groups(
        filtered_path,
        noun_lookup,
        verb_lookup,
        adjective_lookup,
    )
    _progress(inputs, "topology", 0, 2, "scoring semantic topology without model losses")
    cells = select_semantic_world_cells(
        allocation_factory(),
        catalog,
        seed_identity,
        preset,
    )
    _progress(inputs, "topology", 1, 2, "auditing component visibility")
    visibility = require_component_visibility(
        allocation_factory(),
        cells,
        preset.minimum_component_outside_groups,
    )
    _progress(inputs, "topology", 2, 2, "semantic topology and visibility gates passed")
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
    _progress(inputs, "pairing", 0, 1, "pairing every world group across both control arms")
    pairings = pair_world_controls(
        allocation.allocation_groups_by_evaluation_domain,
        cells,
        allocation.controls,
        seed_identity,
    )
    _progress(inputs, "pairing", 1, 1, "deterministic one-to-one control pairings passed")
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
    )
    target = inputs.output_root / partition_sha
    if target.exists():
        raise FileExistsError(f"semantic partition already exists: {target}")
    publication = inputs.temporary_directory / "publication"
    if publication.exists():
        raise FileExistsError(f"semantic publication staging path exists: {publication}")
    publication.mkdir(parents=True)
    (publication / "shards").mkdir()
    (publication / "indexes").mkdir()
    (publication / "manifests").mkdir()
    shutil.copyfile(assignments_path, publication / "assignments.jsonl")
    embedded_catalog = publication / "semantic-catalog" / catalog.catalog_sha256
    shutil.copytree(catalog.root, embedded_catalog)
    retained_records = exclusions["retained_occurrences"]
    _progress(inputs, "shards", 0, retained_records, "publishing exact archive bytes and tokens")
    shards, occurrence_counts = archive_builder._write_shards_and_indexes(
        archive_inputs,
        archive_preset,
        filtered_ingest,
        assignments_path,
        allocation.control_group_owners,
        publication,
        progress_total_occurrences=retained_records,
    )
    _progress(inputs, "shards", retained_records, retained_records, "semantic shards and indexes complete")
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
    )
    archive_builder._write_manifests(
        publication,
        allocation.split_counts,
        allocation.controls,
        occurrence_counts,
    )
    _progress(inputs, "publish", 0, 1, "hashing and strictly reloading semantic publication")
    tree_path = _write_tree(publication, partition_sha)
    tree_sha = _file_sha256(tree_path)
    inputs.output_root.mkdir(parents=True, exist_ok=True)
    os.rename(publication, target)
    _fsync_directory(inputs.output_root)
    from apm.data.text.tinyworlds_p_semantic.artifact import load_partition

    restored = load_partition(target)
    if restored.manifest_sha256 != tree_sha:
        raise RuntimeError("semantic partition tree changed during strict reload")
    _progress(inputs, "publish", 1, 1, "strict semantic partition reload passed")
    return restored


def _archive_contracts(
    inputs: SemanticPartitionInputs,
    preset: SemanticPartitionPreset,
    catalog: SemanticCatalog,
) -> tuple[PartitionInputs, PartitionPreset]:
    progress = None
    if inputs.progress is not None:
        progress = lambda event: inputs.progress(event)
    return (
        PartitionInputs(
            archive_path=inputs.archive_path,
            tokenizer_directory=inputs.tokenizer_directory,
            output_root=inputs.output_root,
            temporary_directory=inputs.temporary_directory,
            archive_identity=inputs.archive_identity,
            tokenizer_identity=inputs.tokenizer_identity,
            progress=progress,
        ),
        PartitionPreset(
            bucket_count=catalog.config.cluster_count,
            public_seed=preset.public_seed,
            worker_count=preset.worker_count,
            run_record_count=preset.run_record_count,
            shard_target_bytes=preset.shard_target_bytes,
            batch_block_documents=preset.batch_block_documents,
            context_length=preset.context_length,
            batch_size=preset.batch_size,
            minimum_role_coverage=preset.minimum_role_coverage,
            selected_cell_median_tolerance=preset.selected_cell_median_tolerance,
            minimum_component_outside_groups=preset.minimum_component_outside_groups,
            world_split_weights=preset.world_split_weights,
            base_split_weights=preset.base_split_weights,
            control_token_tolerance=preset.control_token_tolerance,
            control_source_feature_tolerance=preset.control_source_feature_tolerance,
            control_adjective_length_tolerance=preset.control_adjective_length_tolerance,
            control_mean_length_tolerance=preset.control_mean_length_tolerance,
        ),
    )


def _filter_semantic_groups(
    ingest: ArchiveIngestResult,
    catalog: SemanticCatalog,
    inputs: SemanticPartitionInputs,
) -> tuple[Path, dict[str, int]]:
    noun_lookup = catalog.word_cluster("noun")
    verb_lookup = catalog.word_cluster("verb")
    output_path = inputs.temporary_directory / "semantic-groups.jsonl"
    counts: Counter[str] = Counter()
    _progress(
        inputs,
        "semantic-filter",
        0,
        ingest.audit.archive_group_count,
        "excluding construction and rejected-role groups",
    )
    with output_path.open("wb") as output:
        for completed, group in enumerate(iter_archive_groups(ingest.groups_path), start=1):
            status = group.get("status")
            if status == "eligible":
                group_sha = _text(group, "normalized_story_sha256")
                recipe = _object(group, "recipe")
                noun, verb = _text(recipe, "noun"), _text(recipe, "verb")
                if is_construction_group(group_sha, catalog.config):
                    status = "semantic_construction"
                elif noun not in noun_lookup or verb not in verb_lookup:
                    status = "semantic_word_exclusion"
                else:
                    counts["retained_groups"] += 1
                    counts["retained_occurrences"] += len(_objects(group, "occurrences"))
                    counts["retained_tokens"] += _integer(group, "active_token_count")
            if status in ("semantic_construction", "semantic_word_exclusion"):
                counts[f"{status}_groups"] += 1
                counts[f"{status}_tokens"] += _integer(group, "active_token_count")
                group = {**group, "status": status}
            output.write(canonical_record_bytes(group))
            if completed % 100_000 == 0:
                _progress(
                    inputs,
                    "semantic-filter",
                    completed,
                    ingest.audit.archive_group_count,
                    "excluding construction and rejected-role groups",
                )
        output.flush()
        os.fsync(output.fileno())
    _progress(
        inputs,
        "semantic-filter",
        ingest.audit.archive_group_count,
        ingest.audit.archive_group_count,
        "construction and rejected-role groups excluded",
    )
    if counts["retained_tokens"] != catalog.retained_token_count:
        raise ValueError(
            "semantic catalog retained mass differs from the authenticated archive replay"
        )
    return output_path, dict(counts)


def _adjective_buckets(
    groups_path: Path,
    preset: PartitionPreset,
    seed_identity: str,
) -> tuple[WordBucket, ...]:
    masses: Counter[str] = Counter()
    for group in iter_archive_groups(groups_path):
        if group.get("status") == "eligible":
            masses[_text(_object(group, "recipe"), "adjective")] += _integer(
                group, "active_token_count"
            )
    return balance_word_buckets(
        masses,
        "adjective",
        preset.bucket_count,
        seed_identity,
        public_seed=preset.public_seed,
    )


def _enhance_assignment_exclusions(
    assignments_path: Path,
    groups_path: Path,
    destination: Path,
) -> Path:
    assignments = _iter_jsonl(assignments_path)
    with destination.open("wb") as output:
        for group, assignment in zip(
            iter_archive_groups(groups_path), assignments, strict=True
        ):
            if _text(group, "normalized_story_sha256") != _text(
                assignment, "normalized_story_sha256"
            ):
                raise ValueError("semantic groups and assignments are not aligned")
            status = assignment.get("status")
            excluded_recipe = (
                group.get("recipe")
                if status in ("semantic_construction", "semantic_word_exclusion")
                else None
            )
            output.write(
                canonical_json_bytes(
                    {**assignment, "excluded_recipe": excluded_recipe}
                )
            )
        output.flush()
        os.fsync(output.fileno())
    return destination


def _write_metadata(
    publication: Path,
    inputs: SemanticPartitionInputs,
    preset: SemanticPartitionPreset,
    catalog: SemanticCatalog,
    ingest: ArchiveIngestResult,
    exclusions: Mapping[str, int],
    seed_identity: str,
    partition_sha: str,
    assignments_sha: str,
    adjective_buckets: Sequence[WordBucket],
    cells: Sequence[WorldCell],
    controls: Sequence[ControlSelection],
    pairings: Sequence[ControlPair],
    split_counts: Sequence[SplitCount],
    visibility: Sequence[tuple[str, str, int]],
    shards: Sequence[dict[str, object]],
    *,
    benchmark_id: str = BENCHMARK_ID,
    partition_format: str = PARTITION_FORMAT,
    schema_version: int = SCHEMA_VERSION,
    additional_sources: Mapping[str, object] | None = None,
    topology_selection: Mapping[str, object] | None = None,
) -> None:
    sources = _source_record(inputs, catalog, additional_sources=additional_sources)
    _write_json(publication / "sources.json", sources)
    _write_json(publication / "normalization.json", NORMALIZATION_IDENTITY.as_record())
    _write_json(
        publication / "buckets.json",
        {
            "adjective_buckets": [_bucket_record(item) for item in adjective_buckets],
            "catalog_sha256": catalog.catalog_sha256,
        },
    )
    _write_json(
        publication / "topology.json",
        {"cells": [_cell_record(item) for item in cells]},
    )
    _write_json(
        publication / "controls.json",
        {"controls": [_control_record(item) for item in controls]},
    )
    _write_json(
        publication / "pairings.json",
        {"pairings": [_pair_record(item) for item in pairings]},
    )
    _write_json(publication / "shards.json", {"shards": list(shards)})
    audit_record = {
        "archive_ingest": ingest.audit.as_record(),
        "component_visibility": [
            {"outside_group_count": count, "role": role, "word": word}
            for role, word, count in visibility
        ],
        "semantic_exclusions": dict(sorted(exclusions.items())),
        "split_counts": [_split_count_record(item) for item in split_counts],
    }
    if topology_selection is not None:
        audit_record["topology_selection"] = dict(topology_selection)
    _write_json(publication / "audit.json", audit_record)
    _write_json(
        publication / "partition.json",
        {
            "assignments_sha256": assignments_sha,
            "benchmark_id": benchmark_id,
            "eos_token_id": ingest.eos_token_id,
            "format": partition_format,
            "normalization": NORMALIZATION_IDENTITY.as_record(),
            "pad_token_id": ingest.pad_token_id,
            "partition_sha256": partition_sha,
            "preset": preset.as_record(),
            "schema_version": schema_version,
            "seed_identity_sha256": seed_identity,
            "sources": sources,
        },
    )


def _partition_identity(
    inputs: SemanticPartitionInputs,
    preset: SemanticPartitionPreset,
    catalog: SemanticCatalog,
    adjective_buckets: Sequence[WordBucket],
    cells: Sequence[WorldCell],
    controls: Sequence[ControlSelection],
    assignments_sha: str,
    pairings_sha: str,
    *,
    benchmark_id: str = BENCHMARK_ID,
    additional_sources: Mapping[str, object] | None = None,
) -> str:
    return record_sha256(
        {
            "adjective_buckets": [_bucket_record(item) for item in adjective_buckets],
            "assignments_sha256": assignments_sha,
            "benchmark_id": benchmark_id,
            "cells": [_cell_record(item) for item in cells],
            "controls": [_control_record(item) for item in controls],
            "normalization": NORMALIZATION_IDENTITY.as_record(),
            "pairings_sha256": pairings_sha,
            "preset": preset.as_record(),
            "sources": _source_record(
                inputs,
                catalog,
                additional_sources=additional_sources,
            ),
        }
    )


def _seed_identity(
    inputs: SemanticPartitionInputs,
    preset: SemanticPartitionPreset,
    catalog: SemanticCatalog,
    *,
    benchmark_id: str = BENCHMARK_ID,
    additional_sources: Mapping[str, object] | None = None,
) -> str:
    return record_sha256(
        {
            "benchmark_id": benchmark_id,
            "normalization": NORMALIZATION_IDENTITY.as_record(),
            "preset": preset.as_record(),
            "sources": _source_record(
                inputs,
                catalog,
                additional_sources=additional_sources,
            ),
        }
    )


def _source_record(
    inputs: SemanticPartitionInputs,
    catalog: SemanticCatalog,
    *,
    additional_sources: Mapping[str, object] | None = None,
) -> dict[str, object]:
    sources = {
        "archive": inputs.archive_identity.as_record(),
        "semantic_catalog": {
            "catalog_sha256": catalog.catalog_sha256,
            "encoder_identity_sha256": catalog.encoder_identity.identity_sha256,
            "evidence_sha256": catalog.evidence_sha256,
        },
        "tokenizer": inputs.tokenizer_identity.as_record(),
    }
    additions = dict(additional_sources or {})
    if set(sources) & set(additions):
        raise ValueError("additional semantic partition sources overlap core sources")
    return {**sources, **additions}


def _write_tree(
    publication: Path,
    partition_sha: str,
    *,
    tree_format: str = "tinyworlds-p-semantic-tree",
    schema_version: int = 1,
) -> Path:
    files = tuple(
        {
            "relative_path": path.relative_to(publication).as_posix(),
            "sha256": _file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(
            publication.rglob("*"),
            key=lambda item: item.relative_to(publication).as_posix(),
        )
        if path.is_file() and path != publication / "tree.json"
    )
    path = publication / "tree.json"
    _write_json(
        path,
        {
            "files": list(files),
            "format": tree_format,
            "partition_sha256": partition_sha,
            "schema_version": schema_version,
        },
    )
    return path


def _bucket_record(bucket: WordBucket) -> dict[str, object]:
    return {
        "index": bucket.index,
        "namespace": bucket.namespace,
        "token_mass": bucket.token_mass,
        "words": list(bucket.words),
    }


def _cell_record(cell: WorldCell) -> dict[str, object]:
    return {
        "active_token_count": cell.active_token_count,
        "group_count": cell.group_count,
        "label": cell.label,
        "noun_bucket": cell.noun_bucket,
        "verb_bucket": cell.verb_bucket,
    }


def _control_record(control: ControlSelection) -> dict[str, object]:
    return {
        "active_token_count": control.active_token_count,
        "column_group_count": control.column_group_count,
        "group_sha256": list(control.group_sha256),
        "row_group_count": control.row_group_count,
        "split": control.split,
        "world": control.world,
    }


def _pair_record(pair: ControlPair) -> dict[str, str]:
    return {
        "arm": pair.arm,
        "control_group_sha256": pair.control_group_sha256,
        "split": pair.split,
        "world": pair.world,
        "world_group_sha256": pair.world_group_sha256,
    }


def _split_count_record(count: SplitCount) -> dict[str, object]:
    return {
        "active_token_count": count.active_token_count,
        "group_count": count.group_count,
        "occurrence_count": count.occurrence_count,
        "role": count.role,
        "split": count.split,
        "world": count.world,
    }


def _write_json(path: Path, value: object) -> None:
    with path.open("wb") as output:
        output.write(canonical_json_bytes(value))
        output.flush()
        os.fsync(output.fileno())


def _iter_jsonl(path: Path) -> Iterator[dict[str, object]]:
    with path.open("rb") as source:
        for line in source:
            value = json.loads(line)
            if type(value) is not dict or canonical_json_bytes(value) != line:
                raise ValueError(f"noncanonical semantic JSONL: {path}")
            yield value


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _progress(
    inputs: SemanticPartitionInputs,
    phase: str,
    completed: int,
    total: int,
    detail: str,
) -> None:
    if inputs.progress is not None:
        inputs.progress(ProgressEvent(phase, completed, total, detail))


def _object(record: Mapping[str, object], field: str) -> dict[str, object]:
    value = record.get(field)
    if type(value) is not dict:
        raise ValueError(f"field {field!r} must be an object")
    return value


def _objects(record: Mapping[str, object], field: str) -> tuple[dict[str, object], ...]:
    value = record.get(field)
    if type(value) is not list or any(type(item) is not dict for item in value):
        raise ValueError(f"field {field!r} must contain objects")
    return tuple(value)


def _text(record: Mapping[str, object], field: str) -> str:
    value = record.get(field)
    if type(value) is not str or not value:
        raise ValueError(f"field {field!r} must be nonempty text")
    return value


def _integer(record: Mapping[str, object], field: str) -> int:
    value = record.get(field)
    if type(value) is not int or value < 0:
        raise ValueError(f"field {field!r} must be a nonnegative integer")
    return value


__all__ = ["build_partition"]

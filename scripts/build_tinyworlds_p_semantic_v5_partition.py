#!/usr/bin/env python3
"""Build, independently reproduce, and sample the semantic-v5 partition."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from apm.data.text.tinyworlds_p.partitioning import AllocationGroup

    from apm.data.text.tinyworlds_p_semantic import (
        V4SemanticCatalog,
        V4SemanticPartitionFailure,
        V5SemanticPartitionArtifact,
        V5SemanticPartitionFailure,
        V5SemanticSampleReport,
    )


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SEMANTIC_DATA_ROOT = REPOSITORY_ROOT / "data" / "tinyworlds-p-semantic"
CATALOG_ROOT = SEMANTIC_DATA_ROOT / "catalog" / "v4"
PARENT_FAILURE_ROOT = SEMANTIC_DATA_ROOT / "v4" / "failures"
PARTITION_ROOT = SEMANTIC_DATA_ROOT / "v5"
REBUILD_ROOT = SEMANTIC_DATA_ROOT / "rebuild-verification" / "v5"
SAMPLE_REPORT_ROOT = SEMANTIC_DATA_ROOT / "sample-reports" / "v5"
WORK_ROOT = SEMANTIC_DATA_ROOT / "work"
ARCHIVE_PATH = (
    REPOSITORY_ROOT
    / "data"
    / "tinyworlds-v2"
    / "source"
    / "TinyStories_all_data.tar.gz"
)
TOKENIZER_DIRECTORY = (
    REPOSITORY_ROOT / "checkpoints" / "tinystories-8m" / "tokenizer"
)


def _fixed_sources() -> tuple[V4SemanticCatalog, V4SemanticPartitionFailure]:
    from apm.data.text.tinyworlds_p_semantic import (
        V5_PARENT_CATALOG_SHA256,
        V5_PARENT_PARTITION_FAILURE_SHA256,
        load_v4_partition_failure,
        load_v4_semantic_catalog,
    )

    catalog = load_v4_semantic_catalog(CATALOG_ROOT / V5_PARENT_CATALOG_SHA256)
    failure = load_v4_partition_failure(
        PARENT_FAILURE_ROOT / V5_PARENT_PARTITION_FAILURE_SHA256
    )
    if failure.catalog_sha256 != catalog.catalog_sha256:
        raise RuntimeError("semantic-v5 parent catalog and failure disagree")
    return catalog, failure


def _existing_partition(
    root: Path,
    catalog: V4SemanticCatalog,
    parent: V4SemanticPartitionFailure,
) -> V5SemanticPartitionArtifact | None:
    from apm.data.text.tinyworlds_p_semantic import load_v5_partition

    candidates = (
        tuple(
            load_v5_partition(path)
            for path in sorted(root.glob("[0-9a-f]" * 64))
            if (path / "tree.json").is_file()
        )
        if root.is_dir()
        else ()
    )
    matches = tuple(
        item
        for item in candidates
        if item.semantic_catalog.catalog_sha256 == catalog.catalog_sha256
        and item.parent_partition_failure.failure_sha256 == parent.failure_sha256
    )
    if len(matches) > 1:
        raise RuntimeError("multiple v5 partitions bind the fixed parents")
    return matches[0] if matches else None


def _existing_failure(
    root: Path,
    catalog: V4SemanticCatalog,
    parent: V4SemanticPartitionFailure,
) -> V5SemanticPartitionFailure | None:
    from apm.data.text.tinyworlds_p_semantic import load_v5_partition_failure

    failure_root = root / "failures"
    candidates = (
        tuple(
            load_v5_partition_failure(path)
            for path in sorted(failure_root.glob("[0-9a-f]" * 64))
            if (path / "tree.json").is_file()
        )
        if failure_root.is_dir()
        else ()
    )
    matches = tuple(
        item
        for item in candidates
        if item.catalog_sha256 == catalog.catalog_sha256
        and item.parent_partition_failure_sha256 == parent.failure_sha256
    )
    if len(matches) > 1:
        raise RuntimeError("multiple v5 failures bind the fixed parents")
    return matches[0] if matches else None


def _build(
    catalog: V4SemanticCatalog,
    parent: V4SemanticPartitionFailure,
    output_root: Path,
    temporary_directory: Path,
    *,
    run_record_count: int,
) -> V5SemanticPartitionArtifact:
    from apm.data.text.tinyworlds_p_semantic import (
        CANONICAL_ARCHIVE_IDENTITY,
        CANONICAL_TOKENIZER_IDENTITY,
        V5_SEMANTIC_PARTITION_PRESET,
        V5SemanticPartitionInputs,
        build_v5_partition,
    )
    from apm.data.text.tinyworlds_p_semantic.progress import (
        PARTITION_PHASE_WEIGHTS,
        SemanticProgressReporter,
    )

    reporter = SemanticProgressReporter(
        f"TinyWorlds-P semantic-v5 partition ({run_record_count:,}-record runs)",
        PARTITION_PHASE_WEIGHTS,
    )
    try:
        return build_v5_partition(
            V5SemanticPartitionInputs(
                archive_path=ARCHIVE_PATH,
                tokenizer_directory=TOKENIZER_DIRECTORY,
                semantic_catalog_directory=catalog.root,
                parent_partition_failure_directory=parent.root,
                output_root=output_root,
                temporary_directory=temporary_directory,
                archive_identity=CANONICAL_ARCHIVE_IDENTITY,
                tokenizer_identity=CANONICAL_TOKENIZER_IDENTITY,
                progress=reporter.archive_event,
            ),
            replace(
                V5_SEMANTIC_PARTITION_PRESET,
                worker_count=24,
                run_record_count=run_record_count,
            ),
        )
    finally:
        reporter.close()


def _build_outcome(
    catalog: V4SemanticCatalog,
    parent: V4SemanticPartitionFailure,
    output_root: Path,
    temporary_directory: Path,
    *,
    run_record_count: int,
) -> V5SemanticPartitionArtifact | V5SemanticPartitionFailure:
    from apm.data.text.tinyworlds_p_semantic.partitioning import (
        SemanticPartitionGateError,
    )

    try:
        return _build(
            catalog,
            parent,
            output_root,
            temporary_directory,
            run_record_count=run_record_count,
        )
    except SemanticPartitionGateError:
        failure = _existing_failure(output_root, catalog, parent)
        if failure is None:
            raise
        print(
            "[stop] The strict v5 control-allocation failure was published at "
            f"{failure.root}.",
            flush=True,
        )
        return failure


def _recover_unpublished_primary_failure(
    catalog: V4SemanticCatalog,
    parent: V4SemanticPartitionFailure,
) -> V5SemanticPartitionFailure | None:
    assignment_paths = tuple(
        path
        for path in sorted(
            WORK_ROOT.glob("semantic-partition-v5-*/primary/assignments.jsonl")
        )
        if not (path.parent / "partition-failure-publication").exists()
    )
    if not assignment_paths:
        return None
    if len(assignment_paths) != 1:
        raise RuntimeError(
            "v5 failure recovery requires exactly one unpublished assignment ledger"
        )
    print(
        "[recovery] Replaying the failed comparison allocation from the completed "
        f"ledger at {assignment_paths[0]}.",
        flush=True,
    )
    return _replay_control_failure(catalog, parent, assignment_paths[0])


def _replay_control_failure(
    catalog: V4SemanticCatalog,
    parent: V4SemanticPartitionFailure,
    assignments_path: Path,
) -> V5SemanticPartitionFailure:
    from apm.data.text.tinyworlds_p import builder as archive_builder
    from apm.data.text.tinyworlds_p.partitioning import (
        PartitionGateError as ArchivePartitionGateError,
    )
    from apm.data.text.tinyworlds_p_semantic import (
        CANONICAL_ARCHIVE_IDENTITY,
        CANONICAL_TOKENIZER_IDENTITY,
        V5_BENCHMARK_ID,
        V5_SEMANTIC_PARTITION_PRESET,
        V5SemanticPartitionInputs,
        publish_v5_partition_failure,
    )
    from apm.data.text.tinyworlds_p_semantic.builder import (
        _archive_contracts,
        _seed_identity,
    )
    from apm.data.text.tinyworlds_p_semantic.v4_partition_failure import (
        load_v4_partition_failure_evidence,
    )
    from apm.data.text.tinyworlds_p_semantic.v5_partition_builder import (
        _parent_source,
    )
    from apm.data.text.tinyworlds_p_semantic.v5_topology import (
        topology_selection_from_parent_candidates,
        world_cells_from_topology_selection,
    )
    preset = replace(
        V5_SEMANTIC_PARTITION_PRESET,
        worker_count=24,
        run_record_count=50_000,
    )
    inputs = V5SemanticPartitionInputs(
        archive_path=ARCHIVE_PATH,
        tokenizer_directory=TOKENIZER_DIRECTORY,
        semantic_catalog_directory=catalog.root,
        parent_partition_failure_directory=parent.root,
        output_root=PARTITION_ROOT,
        temporary_directory=assignments_path.parent,
        archive_identity=CANONICAL_ARCHIVE_IDENTITY,
        tokenizer_identity=CANONICAL_TOKENIZER_IDENTITY,
    )
    seed_identity = _seed_identity(
        inputs,
        preset,
        catalog,
        benchmark_id=V5_BENCHMARK_ID,
        additional_sources=_parent_source(parent),
    )
    parent_evidence = load_v4_partition_failure_evidence(parent)
    topology_selection = topology_selection_from_parent_candidates(
        parent_evidence.topology_candidates,
        seed_identity,
        preset,
    )
    cells = world_cells_from_topology_selection(topology_selection)
    _, archive_preset = _archive_contracts(inputs, preset, catalog)
    evaluation_groups = _recover_evaluation_groups(assignments_path)
    expected_domains = {
        (domain, split)
        for domain in ("base", "A", "B", "C", "D", "E")
        for split in ("validation", "test")
    }
    if set(evaluation_groups) != expected_domains:
        raise RuntimeError("the failed v5 evaluation assignments are incomplete")
    print(
        "[recovery] The validation and test groups were reconstructed. The strict "
        "comparison matcher is now being replayed.",
        flush=True,
    )
    try:
        archive_builder._build_controls(
            evaluation_groups,
            cells,
            archive_preset,
            seed_identity,
        )
    except ArchivePartitionGateError as error:
        return publish_v5_partition_failure(
            inputs,
            preset,
            catalog,
            parent,
            seed_identity,
            parent_evidence.semantic_exclusions,
            topology_selection,
            assignments_path,
            str(error),
        )
    raise RuntimeError("v5 failure recovery did not reproduce the control shortage")


def _recover_evaluation_groups(
    assignments_path: Path,
) -> dict[tuple[str, str], list[AllocationGroup]]:
    from tqdm import tqdm

    evaluation_groups: dict[tuple[str, str], list[AllocationGroup]] = defaultdict(
        list
    )
    with assignments_path.open("rb") as source, tqdm(
        total=assignments_path.stat().st_size,
        desc="  recover-controls",
        unit="B",
        unit_scale=True,
    ) as progress:
        pending_progress_bytes = 0
        for line_number, line in enumerate(source, start=1):
            pending_progress_bytes += len(line)
            if line_number % 10_000 == 0:
                progress.update(pending_progress_bytes)
                pending_progress_bytes = 0
            record = json.loads(line)
            if (
                type(record) is not dict
                or record.get("status") != "eligible"
                or record.get("split") not in ("validation", "test")
            ):
                continue
            domain, split, group = _allocation_group_from_assignment(record)
            evaluation_groups[(domain, split)].append(group)
        progress.update(pending_progress_bytes)
    return evaluation_groups


def _allocation_group_from_assignment(
    record: Mapping[str, object],
) -> tuple[str, str, AllocationGroup]:
    from apm.data.text.tinyworlds_p.partitioning import AllocationGroup

    recipe = record.get("recipe")
    provenance = record.get("provenance")
    if type(recipe) is not dict or type(provenance) is not list:
        raise RuntimeError("the failed v5 assignment ledger is malformed")
    raw_features = recipe.get("features")
    if type(raw_features) is not list or any(
        type(feature) is not str for feature in raw_features
    ):
        raise RuntimeError("the failed v5 assignment features are malformed")
    if any(
        type(item) is not dict or type(item.get("source")) is not str
        for item in provenance
    ):
        raise RuntimeError("the failed v5 assignment sources are malformed")
    sources = tuple(sorted({item["source"] for item in provenance}))
    if not sources:
        raise RuntimeError("the failed v5 assignment sources are empty")
    split = _assignment_text(record, "split")
    if split not in ("validation", "test"):
        raise RuntimeError("the failed v5 assignment split is malformed")
    role = _assignment_text(record, "role")
    world = record.get("world")
    if role == "base" and world is None:
        domain = "base"
    elif role == "world" and world in ("A", "B", "C", "D", "E"):
        domain = world
    else:
        raise RuntimeError("the failed v5 assignment domain is malformed")
    return (
        domain,
        split,
        AllocationGroup(
            normalized_sha256=_assignment_text(
                record,
                "normalized_story_sha256",
            ),
            active_token_count=_assignment_positive_integer(
                record,
                "active_token_count",
            ),
            canonical_token_count=_assignment_positive_integer(
                record,
                "canonical_token_count",
            ),
            noun=_assignment_text(recipe, "noun"),
            verb=_assignment_text(recipe, "verb"),
            adjective=_assignment_text(recipe, "adjective"),
            noun_bucket=_assignment_bucket(record, "noun_bucket"),
            verb_bucket=_assignment_bucket(record, "verb_bucket"),
            adjective_bucket=_assignment_bucket(record, "adjective_bucket"),
            source="+".join(sources),
            feature_signature="+".join(raw_features) if raw_features else "none",
        ),
    )


def _assignment_text(record: Mapping[str, object], field: str) -> str:
    value = record.get(field)
    if type(value) is not str or not value:
        raise RuntimeError(f"the failed v5 assignment field {field!r} is malformed")
    return value


def _assignment_positive_integer(
    record: Mapping[str, object],
    field: str,
) -> int:
    value = record.get(field)
    if type(value) is not int or value <= 0:
        raise RuntimeError(f"the failed v5 assignment field {field!r} is malformed")
    return value


def _assignment_bucket(record: Mapping[str, object], field: str) -> int:
    value = record.get(field)
    if type(value) is not int or value < 0:
        raise RuntimeError(f"the failed v5 assignment field {field!r} is malformed")
    return value


def _sample_report(
    artifact: V5SemanticPartitionArtifact,
    work_directory: Path,
) -> V5SemanticSampleReport:
    from apm.data.text.tinyworlds_p_semantic import (
        load_v5_sample_report,
        publish_v5_sample_report,
    )

    partition_root = SAMPLE_REPORT_ROOT / artifact.partition_sha256
    candidates = (
        tuple(
            load_v5_sample_report(path)
            for path in sorted(partition_root.glob("[0-9a-f]" * 64))
            if (path / "tree.json").is_file()
        )
        if partition_root.is_dir()
        else ()
    )
    matches = tuple(
        item
        for item in candidates
        if item.partition_sha256 == artifact.partition_sha256
        and item.catalog_sha256 == artifact.semantic_catalog.catalog_sha256
    )
    if len(matches) > 1:
        raise RuntimeError("multiple v5 reports bind the fixed partition")
    if matches:
        return matches[0]
    print(
        "[sample-report] Publishing held-in validation, all five worlds, and "
        "both control arms for every world.",
        flush=True,
    )
    return publish_v5_sample_report(
        artifact,
        SAMPLE_REPORT_ROOT,
        work_directory / "sample-report-publication",
    )


def main() -> int:
    """Build and reproduce either the v5 partition or its frozen stop."""
    from apm.data.text.tinyworlds_p_semantic import (
        V5SemanticPartitionArtifact,
        V5SemanticPartitionFailure,
    )

    catalog, parent = _fixed_sources()
    primary_partition = _existing_partition(PARTITION_ROOT, catalog, parent)
    primary_failure = _existing_failure(PARTITION_ROOT, catalog, parent)
    if primary_partition is not None and primary_failure is not None:
        raise RuntimeError("v5 has both a success partition and a failure artifact")
    if primary_partition is None and primary_failure is None:
        primary_failure = _recover_unpublished_primary_failure(catalog, parent)

    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    working = Path(tempfile.mkdtemp(prefix="semantic-partition-v5-", dir=WORK_ROOT))
    print(f"Temporary artifact directory: {working}", flush=True)
    if primary_partition is None and primary_failure is None:
        primary_outcome = _build_outcome(
            catalog,
            parent,
            PARTITION_ROOT,
            working / "primary",
            run_record_count=50_000,
        )
    elif primary_failure is not None:
        primary_outcome = primary_failure
        print(
            f"[stop] Reusing the strict v5 primary failure at {primary_failure.root}.",
            flush=True,
        )
    else:
        assert primary_partition is not None
        primary_outcome = primary_partition
        print(
            f"[partition] Reusing the strict v5 primary at {primary_partition.root}.",
            flush=True,
        )

    rebuild_partition = _existing_partition(REBUILD_ROOT, catalog, parent)
    rebuild_failure = _existing_failure(REBUILD_ROOT, catalog, parent)
    if rebuild_partition is not None and rebuild_failure is not None:
        raise RuntimeError("v5 rebuild has both a partition and a failure artifact")
    if rebuild_partition is None and rebuild_failure is None:
        rebuild_outcome = _build_outcome(
            catalog,
            parent,
            REBUILD_ROOT,
            working / "independent-rebuild",
            run_record_count=37_000,
        )
    elif rebuild_failure is not None:
        rebuild_outcome = rebuild_failure
        print(
            f"[rebuild] Reusing the strict v5 failure at {rebuild_failure.root}.",
            flush=True,
        )
    else:
        assert rebuild_partition is not None
        rebuild_outcome = rebuild_partition
        print(
            f"[rebuild] Reusing the strict v5 partition at {rebuild_partition.root}.",
            flush=True,
        )

    if isinstance(primary_outcome, V5SemanticPartitionFailure):
        if not isinstance(rebuild_outcome, V5SemanticPartitionFailure):
            raise RuntimeError("the independent v5 run disagrees with the primary stop")
        if (
            rebuild_outcome.failure_sha256 != primary_outcome.failure_sha256
            or (rebuild_outcome.root / "tree.json").read_bytes()
            != (primary_outcome.root / "tree.json").read_bytes()
        ):
            raise RuntimeError("the independent v5 failure is not byte-identical")
        selected = primary_outcome.topology_selection["selected"]
        if type(selected) is not dict:
            raise RuntimeError("the strict v5 failure lost its selected topology")
        print(
            "[rebuild] The v5 failure identity and authenticated tree are "
            "byte-identical.",
            flush=True,
        )
        print(
            "Version 5 selected the balanced cells "
            f"{selected['cells']} with active-token counts "
            f"{selected['token_masses']}.",
        )
        print(
            f"The {primary_outcome.shortfall.arm} comparison for condition "
            f"{primary_outcome.shortfall.world} in validation needed "
            f"{primary_outcome.shortfall.required_count:,} groups, but only "
            f"{primary_outcome.shortfall.available_count:,} were available."
        )
        print(f"Primary failure: {primary_outcome.root}")
        print(f"Failure SHA-256: {primary_outcome.failure_sha256}")
        print(f"Independent failure rebuild: {rebuild_outcome.root}")
        print(
            "No partition, sample report, GPU training run, or sealed-test result "
            "was produced."
        )
        return 2

    if not isinstance(primary_outcome, V5SemanticPartitionArtifact):
        raise RuntimeError("the v5 primary outcome has an unknown type")
    if not isinstance(rebuild_outcome, V5SemanticPartitionArtifact):
        raise RuntimeError("the independent v5 run disagrees with the primary result")
    primary = primary_outcome
    rebuild = rebuild_outcome
    if (
        rebuild.partition_sha256 != primary.partition_sha256
        or (rebuild.root / "tree.json").read_bytes()
        != (primary.root / "tree.json").read_bytes()
    ):
        raise RuntimeError("independent semantic-v5 rebuild is not byte-identical")
    print(
        "[rebuild] The v5 partition identity and authenticated tree are "
        "byte-identical.",
        flush=True,
    )
    report = _sample_report(primary, working)
    print(f"Partition: {primary.root}")
    print(f"Partition SHA-256: {primary.partition_sha256}")
    print(f"Independent rebuild: {rebuild.root}")
    print(f"Sample report: {report.root}")
    print(f"Sample report SHA-256: {report.report_sha256}")
    print("GPU training and the sealed test remain unopened.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest

from apm.data.text.tinyworlds_p import (
    NORMALIZATION_IDENTITY,
    PartitionPreset,
    build_archive_ingest,
    iter_archive_groups,
)
from apm.data.text.tinyworlds_p_semantic import (
    ENCODER_DIMENSION,
    ENCODER_IDENTIFIER,
    ENCODER_REVISION,
    EncoderIdentity,
    ModelFile,
    PartitionArtifactError,
    V4_SEMANTIC_CONFIG,
    V4_SEMANTIC_PARTITION_PRESET,
    V4SemanticPartitionFailureError,
    V4SemanticPartitionInputs,
    V4SemanticPartitionPreset,
    WordEvidence,
    build_v4_partition,
    build_v4_semantic_catalog,
    is_construction_group,
    load_partition,
    load_v4_partition,
    load_v4_partition_failure,
    load_v4_partition_failure_evidence,
    load_v4_sample_report,
    publish_v4_sample_report,
    publish_v4_partition_failure,
)
from apm.data.text.tinyworlds_p.contracts import WordBucket
from apm.data.text.tinyworlds_p_semantic.builder import _seed_identity
from apm.data.text.tinyworlds_p_semantic.contracts import canonical_json_bytes
from apm.data.text.tinyworlds_p_semantic.partitioning import (
    SemanticTopologyAudit,
    SemanticTopologyCandidate,
)
from apm.data.text.tinyworlds_p_semantic.v4_contracts import V4_BENCHMARK_ID
from test_tinyworlds_p_partition import _fixture_inputs


def _unit(index: int) -> np.ndarray:
    result = np.zeros(ENCODER_DIMENSION, dtype=np.float32)
    result[index] = 1.0
    return result


def _fixture(
    tmp_path: Path,
) -> tuple[V4SemanticPartitionInputs, V4SemanticPartitionPreset, dict[tuple[str, int], bytes]]:
    archive_inputs, expected = _fixture_inputs(
        tmp_path,
        "unused-v4-archive-output",
        "v4-catalog-ingest",
    )
    config = replace(
        V4_SEMANTIC_CONFIG,
        role_calibration_fold_count=3,
        minimum_calibration_reference_words=1,
        minimum_contexts_per_word=4,
        maximum_context_silhouette=1.0,
        cluster_count=3,
        minimum_nouns_per_cluster=1,
        minimum_verbs_per_cluster=1,
        maximum_centroid_pair_cosine=0.95,
    )
    ingest = build_archive_ingest(
        archive_inputs,
        PartitionPreset(
            bucket_count=3,
            worker_count=1,
            run_record_count=47,
            shard_target_bytes=1_024,
            batch_block_documents=7,
            context_length=8,
            batch_size=4,
            minimum_role_coverage=1.0,
            selected_cell_median_tolerance=0.25,
            minimum_component_outside_groups=1,
            world_split_weights=(80, 10, 10),
            base_split_weights=(20, 40, 40),
        ),
        NORMALIZATION_IDENTITY,
    )
    pair_masses: Counter[tuple[str, str]] = Counter()
    for group in iter_archive_groups(ingest.groups_path):
        if group["status"] == "eligible" and not is_construction_group(
            group["normalized_story_sha256"],
            config,
        ):
            recipe = group["recipe"]
            pair_masses[(recipe["noun"], recipe["verb"])] += group[
                "active_token_count"
            ]
    noun_masses: Counter[str] = Counter()
    verb_masses: Counter[str] = Counter()
    for (noun, verb), mass in pair_masses.items():
        noun_masses[noun] += mass
        verb_masses[verb] += mass
    evidence = tuple(
        WordEvidence(
            role=role,
            word=word,
            token_mass=masses[word],
            target_anchor_embeddings=np.stack((_unit(axis),) * 3),
            opposite_anchor_embeddings=np.stack((-_unit(axis),) * 3),
            context_embeddings=np.stack((_unit(axis),) * 4),
        )
        for role, words, masses, offset in (
            ("noun", ("cat", "robot", "garden"), noun_masses, 0),
            ("verb", ("help", "find", "carry"), verb_masses, 4),
        )
        for axis, word in enumerate(words, start=offset)
    )
    encoder = EncoderIdentity(
        identifier=ENCODER_IDENTIFIER,
        revision=ENCODER_REVISION,
        dimension=ENCODER_DIMENSION,
        files=(ModelFile("model.safetensors", 1, sha256(b"v4-partition").hexdigest()),),
    )
    catalog = build_v4_semantic_catalog(
        evidence,
        pair_masses,
        encoder,
        "e" * 64,
        sum(pair_masses.values()),
        tmp_path / "v4-semantic-catalogs",
        tmp_path / "v4-catalog-work",
        config,
    )
    inputs = V4SemanticPartitionInputs(
        archive_path=archive_inputs.archive_path,
        tokenizer_directory=archive_inputs.tokenizer_directory,
        semantic_catalog_directory=catalog.root,
        output_root=tmp_path / "v4-semantic-partitions-a",
        temporary_directory=tmp_path / "v4-semantic-partition-work-a",
        archive_identity=archive_inputs.archive_identity,
        tokenizer_identity=archive_inputs.tokenizer_identity,
    )
    preset = V4SemanticPartitionPreset(
        worker_count=1,
        run_record_count=47,
        shard_target_bytes=1_024,
        batch_block_documents=7,
        context_length=8,
        batch_size=4,
        minimum_role_coverage=1.0,
        selected_cell_median_tolerance=0.25,
        minimum_component_outside_groups=1,
        world_split_weights=(80, 10, 10),
        base_split_weights=(20, 40, 40),
        control_token_tolerance=0.25,
        control_source_feature_tolerance=0.25,
        control_adjective_length_tolerance=0.25,
        control_mean_length_tolerance=0.25,
    )
    return inputs, preset, expected


def _tree_bytes(root: Path) -> dict[Path, bytes]:
    return {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_v4_partition_reconstructs_pairs_reports_and_rebuilds(
    tmp_path: Path,
) -> None:
    inputs, preset, expected = _fixture(tmp_path)
    first = build_v4_partition(inputs, preset)

    assert first.semantic_catalog.catalog_sha256 == inputs.semantic_catalog_directory.name
    assert len(first.pairings) == sum(len(item.group_sha256) for item in first.controls)
    assert len({item.control_group_sha256 for item in first.pairings}) == len(
        first.pairings
    )
    assignments = tuple(
        json.loads(line)
        for line in (first.root / "assignments.jsonl").read_bytes().splitlines()
    )
    assert any(item["status"] == "semantic_construction" for item in assignments)
    assert all(
        not is_construction_group(
            item["normalized_story_sha256"],
            first.semantic_catalog.config,
        )
        for item in assignments
        if item["status"] == "eligible"
    )
    documents = tuple(
        json.loads(line)
        for line in (first.root / "documents.jsonl").read_bytes().splitlines()
    )
    for document in documents:
        shard = first.root / "shards" / f"text-{document['text_shard']:06d}.bin"
        with shard.open("rb") as source:
            source.seek(document["text_offset"])
            restored = source.read(document["text_bytes"])
        assert restored == expected[(document["source_member"], document["source_index"])]

    report = publish_v4_sample_report(
        first,
        tmp_path / "v4-sample-reports",
        tmp_path / "v4-sample-report-work",
    )
    restored_report = load_v4_sample_report(report.root)
    payload = json.loads((report.root / "sample-report.json").read_bytes())
    assert restored_report.partition_sha256 == first.partition_sha256
    assert restored_report.catalog_sha256 == first.semantic_catalog.catalog_sha256
    assert payload["sealed_test_opened"] is False
    assert len(payload["samples"]) == 16
    assert "Semantic v4 Pre-Training" in (report.root / "sample-report.md").read_text()

    second = build_v4_partition(
        replace(
            inputs,
            output_root=tmp_path / "v4-semantic-partitions-b",
            temporary_directory=tmp_path / "v4-semantic-partition-work-b",
        ),
        replace(preset, worker_count=2, run_record_count=31),
    )
    assert second.partition_sha256 == first.partition_sha256
    assert _tree_bytes(second.root) == _tree_bytes(first.root)


def test_v4_partition_rejects_old_formats_and_tampering(tmp_path: Path) -> None:
    old = tmp_path / ("a" * 64)
    old.mkdir()
    (old / "tree.json").write_bytes(
        canonical_json_bytes(
            {
                "files": [],
                "format": "tinyworlds-p-semantic-tree",
                "partition_sha256": old.name,
                "schema_version": 1,
            }
        )
    )
    with pytest.raises(PartitionArtifactError, match="unsupported semantic-v4"):
        load_v4_partition(old)

    inputs, preset, _ = _fixture(tmp_path)
    artifact = build_v4_partition(inputs, preset)
    with pytest.raises(PartitionArtifactError, match="unsupported semantic partition"):
        load_partition(artifact.root)
    shard = next((artifact.root / "shards").glob("tokens-*.uint16"))
    payload = bytearray(shard.read_bytes())
    payload[0] ^= 1
    shard.write_bytes(payload)
    with pytest.raises(PartitionArtifactError, match="checksum changed"):
        load_v4_partition(artifact.root)


def test_v4_topology_failure_is_content_addressed_and_strict(tmp_path: Path) -> None:
    inputs, _, _ = _fixture(tmp_path)
    from apm.data.text.tinyworlds_p_semantic import load_v4_semantic_catalog

    catalog = load_v4_semantic_catalog(inputs.semantic_catalog_directory)
    failed = SemanticTopologyCandidate(
        cells=((0, 0), (1, 0), (1, 1), (0, 1), (2, 2)),
        token_masses=(100, 100, 100, 100, 130),
        group_counts=(10, 10, 10, 10, 13),
        semantic_dispersion=0.1,
        token_imbalance=Fraction(3, 10),
        nuisance_imbalance=Fraction(1, 10),
        control_capacity=Fraction(2, 1),
        tie_sha256=sha256(b"failed").hexdigest(),
    )
    balanced = SemanticTopologyCandidate(
        cells=((0, 2), (1, 2), (1, 1), (0, 1), (2, 0)),
        token_masses=(100, 100, 100, 100, 100),
        group_counts=(10, 10, 10, 10, 10),
        semantic_dispersion=0.2,
        token_imbalance=Fraction(0, 1),
        nuisance_imbalance=Fraction(0, 1),
        control_capacity=Fraction(2, 1),
        tie_sha256=sha256(b"balanced").hexdigest(),
    )
    audit = SemanticTopologyAudit(
        physical_candidate_count=28_224,
        nonempty_candidate_count=4,
        visible_candidate_count=3,
        control_capable_candidate_count=2,
        candidates=(failed, balanced),
        median_tolerance=0.10,
    )
    seed = _seed_identity(
        inputs,
        V4_SEMANTIC_PARTITION_PRESET,
        catalog,
        benchmark_id=V4_BENCHMARK_ID,
    )
    adjective_buckets = (
        WordBucket("adjective", 0, 10, ("kind",)),
        WordBucket("adjective", 1, 10, ("small",)),
        WordBucket("adjective", 2, 10, ("red",)),
    )
    failure, rebuilt = tuple(
        publish_v4_partition_failure(
            inputs,
            V4_SEMANTIC_PARTITION_PRESET,
            catalog,
            seed,
            adjective_buckets,
            {"retained_groups": 50, "retained_tokens": 500},
            audit,
            "best semantic topology violates the selected-cell token median gate",
        )
        for _ in range(2)
    )
    restored = load_v4_partition_failure(failure.root)
    evidence = load_v4_partition_failure_evidence(restored)
    assert restored.failure_sha256 == failure.failure_sha256
    assert restored.audit["median_feasible_candidate_count"] == 1
    assert restored.audit["selected"]["median_gate"]["passes"] is False
    assert evidence.adjective_buckets == adjective_buckets
    assert evidence.partition_preset == V4_SEMANTIC_PARTITION_PRESET.as_record()
    assert evidence.sources["semantic_catalog"]["catalog_sha256"] == (
        catalog.catalog_sha256
    )
    assert len(evidence.topology_candidates) == 2
    assert not (inputs.output_root / failure.failure_sha256).exists()
    assert rebuilt.failure_sha256 == failure.failure_sha256

    payload = bytearray((failure.root / "audit.json").read_bytes())
    payload[payload.index(b"28224")] = ord("1")
    (failure.root / "audit.json").write_bytes(payload)
    with pytest.raises(V4SemanticPartitionFailureError, match="file changed"):
        load_v4_partition_failure(failure.root)

from __future__ import annotations

from collections import Counter
from dataclasses import replace
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
from apm.data.text.tinyworlds_p import training as archive_training
from apm.data.text.tinyworlds_p_semantic import (
    ENCODER_DIMENSION,
    ENCODER_IDENTIFIER,
    ENCODER_REVISION,
    EncoderIdentity,
    ModelFile,
    PartitionArtifactError,
    SEMANTIC_CONFIG,
    SemanticPartitionInputs,
    SemanticPartitionPreset,
    StreamingTrainingConfig,
    WordEvidence,
    build_partition,
    build_semantic_catalog,
    evaluate_partition_split,
    is_construction_group,
    iter_partition_batches,
    load_streaming_checkpoint,
    load_partition,
    load_sample_report,
    publish_sample_report,
    run_streaming_base_training,
)
from apm.data.text.tinyworlds_p_semantic.contracts import canonical_json_bytes
from apm.lm.config import GptNeoConfig
from apm.lm.parameters import init_gpt_neo_params
from apm.lm.training_state_artifact import lm_train_state_checksum
import jax
from test_tinyworlds_p_partition import _fixture_inputs


def _unit(index: int) -> np.ndarray:
    result = np.zeros(ENCODER_DIMENSION, dtype=np.float32)
    result[index] = 1.0
    return result


def _semantic_fixture(
    tmp_path: Path,
) -> tuple[SemanticPartitionInputs, SemanticPartitionPreset, dict[tuple[str, int], bytes]]:
    archive_inputs, expected = _fixture_inputs(
        tmp_path,
        "unused-archive-output",
        "catalog-ingest",
    )
    config = replace(
        SEMANTIC_CONFIG,
        cluster_count=3,
        minimum_contexts_per_word=4,
        maximum_context_silhouette=1.0,
        minimum_cluster_mass_fraction=0.75,
        maximum_cluster_mass_fraction=1.25,
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
            shard_target_bytes=1024,
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
            group["normalized_story_sha256"], config
        ):
            recipe = group["recipe"]
            pair_masses[(recipe["noun"], recipe["verb"])] += group["active_token_count"]
    noun_masses: Counter[str] = Counter()
    verb_masses: Counter[str] = Counter()
    for (noun, verb), mass in pair_masses.items():
        noun_masses[noun] += mass
        verb_masses[verb] += mass
    nouns = ("cat", "robot", "garden")
    verbs = ("help", "find", "carry")
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
            ("noun", nouns, noun_masses, 0),
            ("verb", verbs, verb_masses, 4),
        )
        for axis, word in enumerate(words, start=offset)
    )
    encoder = EncoderIdentity(
        identifier=ENCODER_IDENTIFIER,
        revision=ENCODER_REVISION,
        dimension=ENCODER_DIMENSION,
        files=(ModelFile("model.safetensors", 1, sha256(b"fixture").hexdigest()),),
    )
    catalog = build_semantic_catalog(
        evidence,
        pair_masses,
        encoder,
        "e" * 64,
        sum(pair_masses.values()),
        tmp_path / "semantic-catalogs",
        tmp_path / "catalog-work",
        config,
    )
    inputs = SemanticPartitionInputs(
        archive_path=archive_inputs.archive_path,
        tokenizer_directory=archive_inputs.tokenizer_directory,
        semantic_catalog_directory=catalog.root,
        output_root=tmp_path / "semantic-partitions-a",
        temporary_directory=tmp_path / "semantic-partition-work-a",
        archive_identity=archive_inputs.archive_identity,
        tokenizer_identity=archive_inputs.tokenizer_identity,
    )
    preset = SemanticPartitionPreset(
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


def test_semantic_partition_excludes_construction_pairs_controls_and_rebuilds(
    tmp_path: Path,
) -> None:
    inputs, preset, expected = _semantic_fixture(tmp_path)
    first = build_partition(inputs, preset)

    assert first.semantic_catalog.catalog_sha256 == inputs.semantic_catalog_directory.name
    assert len(first.pairings) == sum(len(item.group_sha256) for item in first.controls)
    assert tuple(iter_partition_batches(first, "base/train", epoch=0))
    assignments = tuple(
        json.loads(line)
        for line in (first.root / "assignments.jsonl").read_bytes().splitlines()
    )
    assert any(item["status"] == "semantic_construction" for item in assignments)
    assert all(
        not is_construction_group(item["normalized_story_sha256"], first.semantic_catalog.config)
        for item in assignments
        if item["status"] == "eligible"
    )
    documents = tuple(
        json.loads(line)
        for line in (first.root / "documents.jsonl").read_bytes().splitlines()
    )
    for document in documents:
        text_shard = first.root / "shards" / f"text-{document['text_shard']:06d}.bin"
        with text_shard.open("rb") as source:
            source.seek(document["text_offset"])
            restored = source.read(document["text_bytes"])
        assert restored == expected[(document["source_member"], document["source_index"])]

    second_inputs = replace(
        inputs,
        output_root=tmp_path / "semantic-partitions-b",
        temporary_directory=tmp_path / "semantic-partition-work-b",
    )
    second = build_partition(
        second_inputs,
        replace(preset, worker_count=2, run_record_count=31),
    )
    assert second.partition_sha256 == first.partition_sha256
    assert _tree_bytes(second.root) == _tree_bytes(first.root)


def test_semantic_loader_rejects_archive_v1_and_tampering(tmp_path: Path) -> None:
    archive_v1 = tmp_path / ("a" * 64)
    archive_v1.mkdir()
    (archive_v1 / "tree.json").write_bytes(
        canonical_json_bytes(
            {
                "files": [],
                "format": "tinyworlds-p-archive-tree",
                "partition_sha256": archive_v1.name,
                "schema_version": 1,
            }
        )
    )
    with pytest.raises(PartitionArtifactError, match="unsupported semantic partition"):
        load_partition(archive_v1)

    inputs, preset, _ = _semantic_fixture(tmp_path)
    artifact = build_partition(inputs, preset)
    shard = next((artifact.root / "shards").glob("tokens-*.uint16"))
    payload = bytearray(shard.read_bytes())
    payload[0] ^= 1
    shard.write_bytes(payload)
    with pytest.raises(PartitionArtifactError, match="checksum changed"):
        load_partition(artifact.root)


def test_semantic_evaluation_persists_sorted_group_loss_sums(tmp_path: Path) -> None:
    inputs, preset, _ = _semantic_fixture(tmp_path)
    artifact = build_partition(inputs, preset)
    config = GptNeoConfig(
        vocab_size=artifact.tokenizer_identity.vocab_size,
        max_position_embeddings=8,
        hidden_size=8,
        intermediate_size=16,
        num_layers=1,
        num_heads=2,
        attention_types=("global",),
        local_window_size=4,
    )
    params = init_gpt_neo_params(jax.random.PRNGKey(0), config)
    evaluation = evaluate_partition_split(
        params,
        artifact,
        "world/A/validation",
        tmp_path / "world-a.groups.jsonl",
        config,
    )
    records = tuple(json.loads(line) for line in evaluation.ledger_path.read_bytes().splitlines())

    assert [item["normalized_story_sha256"] for item in records] == sorted(
        item["normalized_story_sha256"] for item in records
    )
    assert sum(item["active_tokens"] for item in records) == evaluation.active_tokens
    assert sum(item["loss_sum"] for item in records) == pytest.approx(evaluation.loss_sum)
    assert np.isfinite(evaluation.nll)


def test_semantic_training_resume_is_identical_and_uses_its_own_contract(
    tmp_path: Path,
) -> None:
    inputs, preset, _ = _semantic_fixture(tmp_path)
    artifact = build_partition(inputs, preset)
    config = StreamingTrainingConfig(
        model_config=GptNeoConfig(
            vocab_size=artifact.tokenizer_identity.vocab_size,
            max_position_embeddings=8,
            hidden_size=8,
            intermediate_size=16,
            num_layers=1,
            num_heads=2,
            attention_types=("global",),
            local_window_size=4,
        ),
        epochs=2,
        calibration_epochs=1,
        context_length=8,
        microbatch_size=4,
        accumulation_microbatches=4,
        maximum_learning_rate=1e-2,
        minimum_learning_rate=1e-3,
        warmup_fraction=0.1,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        weight_decay=0.01,
        gradient_clip_norm=1.0,
        parameter_seed=0,
        state_interval_updates=100,
        allocator_peak_limit_bytes=1024**3,
    )
    uninterrupted = run_streaming_base_training(
        artifact,
        tmp_path / "uninterrupted",
        config,
    )
    interrupted = run_streaming_base_training(
        artifact,
        tmp_path / "resumed",
        config,
        stop_after_update=3,
    )
    resume_manifest = json.loads(
        (interrupted.checkpoints[-1].directory / "resume.json").read_bytes()
    )
    assert resume_manifest["format"] == "tinyworlds-p-semantic-training-resume"
    resumed = run_streaming_base_training(
        artifact,
        tmp_path / "resumed",
        config,
        resume_from=interrupted.checkpoints[-1].directory,
    )

    assert resumed.cursor == uninterrupted.cursor
    assert lm_train_state_checksum(resumed.state) == lm_train_state_checksum(
        uninterrupted.state
    )
    assert resumed.trace_path.read_bytes() == uninterrupted.trace_path.read_bytes()

    archive_resume = archive_training.write_streaming_checkpoint(
        tmp_path / "archive-v1-resume",
        uninterrupted.training_sha256,
        uninterrupted.state,
        uninterrupted.cursor,
    )
    with pytest.raises(ValueError, match="resume identity"):
        load_streaming_checkpoint(
            archive_resume.directory,
            uninterrupted.training_sha256,
            uninterrupted.state,
        )


def test_semantic_sample_report_covers_validation_and_both_control_arms(
    tmp_path: Path,
) -> None:
    inputs, preset, _ = _semantic_fixture(tmp_path)
    artifact = build_partition(inputs, preset)
    report = publish_sample_report(
        artifact,
        tmp_path / "sample-reports",
        tmp_path / "sample-report-work",
    )
    restored = load_sample_report(report.root)
    payload = json.loads((restored.root / "sample-report.json").read_bytes())

    assert restored.partition_sha256 == artifact.partition_sha256
    assert restored.catalog_sha256 == artifact.semantic_catalog.catalog_sha256
    assert payload["sealed_test_opened"] is False
    assert len(payload["samples"]) == 16
    assert {
        sample["condition"].rsplit("/", 1)[-1]
        for sample in payload["samples"]
        if sample["condition"].startswith("control/")
    } == {"row", "column"}
    assert "Cluster word inventories" in (restored.root / "sample-report.md").read_text()

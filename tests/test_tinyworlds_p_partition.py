from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import io
import json
from pathlib import Path
import tarfile

import numpy as np
import pytest

from apm.data.text.tinyworlds_p import (
    EpochValidation,
    HashedFile,
    PartitionInputs,
    PartitionPreset,
    SourceIdentity,
    SourceJoinError,
    StreamingTrainingConfig,
    TokenizerIdentity,
    WorldGap,
    build_partition,
    calibration_grid_decision,
    cosine_learning_rate,
    evaluate_partition_split,
    iter_partition_batches,
    load_partition,
    normalize_story_identity,
    normalized_story_sha256,
    recover_released_recipe,
    run_streaming_base_training,
    select_best_eligible_epoch,
)
from apm.lm.config import GptNeoConfig
from apm.lm.training_state_artifact import lm_train_state_checksum


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _source_identity(path: Path) -> SourceIdentity:
    return SourceIdentity(
        dataset_id="tinyworlds-p/fixture",
        revision="0" * 40,
        filename=path.name,
        size_bytes=path.stat().st_size,
        sha256=_sha256(path),
    )


def _fixture_inputs(tmp_path: Path, output_name: str, work_name: str) -> PartitionInputs:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    nouns = ("cat", "robot", "garden")
    verbs = ("help", "find", "carry")
    adjectives = ("kind", "bright", "quiet")
    vocabulary_words = (
        "A",
        "kind",
        "bright",
        "quiet",
        "cat",
        "robot",
        "garden",
        "will",
        "help",
        "find",
        "carry",
        "today",
    )
    vocabulary = {"<unk>": 0, "<|endoftext|>": 1}
    vocabulary.update({word: index for index, word in enumerate(vocabulary_words, start=2)})
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer_directory = tmp_path / "tokenizer"
    tokenizer_directory.mkdir(exist_ok=True)
    tokenizer_path = tokenizer_directory / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    records = []
    stories = []
    for noun_index, noun in enumerate(nouns):
        for verb_index, verb in enumerate(verbs):
            for story_index in range(60):
                adjective = adjectives[story_index % len(adjectives)]
                story = (
                    f"A {adjective} {noun} will {verb} today {noun_index} "
                    f"{verb_index} {story_index}."
                )
                stories.append(story)
                records.append(
                    {
                        "instruction": {
                            "features": [],
                            "prompt:": (
                                f'Write a story using the verb "{verb}", '
                                f'the noun "{noun}" and the adjective "{adjective}".'
                            ),
                            "words": [verb, noun, adjective],
                        },
                        "source": "GPT-4",
                        "story": story,
                        "summary": "A small fixture story.",
                    }
                )
    corpus_path = tmp_path / "TinyStories-train.txt"
    corpus_path.write_bytes(b"<|endoftext|>".join(story.encode("utf-8") for story in stories))
    archive_path = tmp_path / "TinyStories_all_data.tar.gz"
    archive_payload = json.dumps(records, separators=(",", ":")).encode("utf-8")
    with tarfile.open(archive_path, "w:gz") as archive:
        member = tarfile.TarInfo("./data00.json")
        member.size = len(archive_payload)
        member.mtime = 0
        archive.addfile(member, io.BytesIO(archive_payload))
    tokenizer_file = HashedFile(
        tokenizer_path.name,
        tokenizer_path.stat().st_size,
        _sha256(tokenizer_path),
    )
    return PartitionInputs(
        corpus_path=corpus_path,
        metadata_archive_path=archive_path,
        tokenizer_directory=tokenizer_directory,
        output_root=tmp_path / output_name,
        temporary_directory=tmp_path / work_name,
        corpus_identity=_source_identity(corpus_path),
        metadata_identity=_source_identity(archive_path),
        tokenizer_identity=TokenizerIdentity(
            kind="word-level-fixture",
            identifier="tinyworlds-p/fixture",
            revision="0" * 40,
            vocab_size=len(vocabulary),
            files=(tokenizer_file,),
        ),
    )


def _fixture_preset(worker_count: int = 1) -> PartitionPreset:
    return PartitionPreset(
        bucket_count=3,
        worker_count=worker_count,
        run_record_count=47,
        shard_target_bytes=1_024,
        batch_block_documents=7,
        context_length=8,
        batch_size=4,
        minimum_hash_match_coverage=1.0,
        minimum_role_coverage=1.0,
        minimum_eligible_coverage=1.0,
        selected_cell_median_tolerance=0.10,
        minimum_component_outside_groups=1,
        world_split_weights=(80, 10, 10),
        base_split_weights=(20, 40, 40),
    )


def test_normalization_collapses_quote_and_whitespace_variants() -> None:
    left = "  The CAT said, “I’m here.”\n"
    right = 'the cat said, "i\'m   here."'

    assert normalize_story_identity(left) == normalize_story_identity(right)
    assert normalized_story_sha256(left) == normalized_story_sha256(right)


def test_role_recovery_requires_three_explicit_unique_labels() -> None:
    recovered = recover_released_recipe(
        "Use the verb ‘HELP’, the noun \"Cat\", and adjective 'Kind'.",
        ("help", "cat", "kind"),
        (" dialogue ",),
    )

    assert recovered is not None
    assert (recovered.noun, recovered.verb, recovered.adjective) == (
        "cat",
        "help",
        "kind",
    )
    assert recovered.features == ("dialogue",)
    assert (
        recover_released_recipe(
            'Use the verb "win", noun "win", and adjective "kind".',
            ("win", "win", "kind"),
        )
        is None
    )
    assert (
        recover_released_recipe(
            'Use the verb "help", noun "cat", noun "robot", and adjective "kind".',
            ("help", "cat", "kind"),
        )
        is None
    )


def test_cpu_fixture_builds_loads_batches_and_rebuilds_identically(tmp_path: Path) -> None:
    first_inputs = _fixture_inputs(tmp_path, "first-output", "first-work")
    preset = _fixture_preset()

    first = build_partition(first_inputs, preset)
    restored = load_partition(first.root)
    batches = tuple(iter_partition_batches(restored, "base/train", epoch=0))
    replayed = tuple(iter_partition_batches(restored, "base/train", epoch=0))

    assert restored.partition_sha256 == first.partition_sha256
    assert len(restored.cells) == 5
    assert len(restored.controls) == 10
    assert batches
    assert len(batches) == len(replayed)
    for left, right in zip(batches, replayed, strict=True):
        np.testing.assert_array_equal(left.input_ids, right.input_ids)
        np.testing.assert_array_equal(left.loss_mask, right.loss_mask)

    second_inputs = replace(
        first_inputs,
        output_root=tmp_path / "second-output",
        temporary_directory=tmp_path / "second-work",
    )
    second = build_partition(
        second_inputs,
        replace(preset, worker_count=2, run_record_count=31),
    )

    assert second.partition_sha256 == first.partition_sha256
    first_files = {
        path.relative_to(first.root): path.read_bytes()
        for path in first.root.rglob("*")
        if path.is_file()
    }
    second_files = {
        path.relative_to(second.root): path.read_bytes()
        for path in second.root.rglob("*")
        if path.is_file()
    }
    assert second_files == first_files

    source_bytes = first_inputs.corpus_path.read_bytes()
    tokenizer_path = first_inputs.tokenizer_directory / "tokenizer.json"
    from apm.lm.text import TokenizersTextTokenizer

    tokenizer = TokenizersTextTokenizer.from_file(tokenizer_path)
    documents = tuple(
        json.loads(line)
        for line in (restored.root / "documents.jsonl").read_text().splitlines()
    )
    for document in documents[:20]:
        source_start = document["source_byte_offset"]
        raw_story = source_bytes[source_start : source_start + document["byte_length"]]
        text_path = (
            restored.root
            / "shards"
            / f"text-{document['text_shard']:06d}.bin"
        )
        text_payload = text_path.read_bytes()
        assert raw_story == text_payload[
            document["text_offset"] : document["text_offset"] + document["text_bytes"]
        ]
        token_path = (
            restored.root
            / "shards"
            / f"tokens-{document['token_shard']:06d}.uint16"
        )
        token_payload = np.memmap(token_path, dtype="<u2", mode="r")
        token_start = document["token_offset"]
        token_stop = token_start + document["token_count"]
        assert tuple(int(token) for token in token_payload[token_start:token_stop]) == (
            tokenizer.encode(raw_story.decode("utf-8"), add_eos=True)
        )

    tampered_shard = next((restored.root / "shards").glob("tokens-*.uint16"))
    tampered = bytearray(tampered_shard.read_bytes())
    tampered[0] ^= 1
    tampered_shard.write_bytes(tampered)
    with pytest.raises(ValueError, match="checksum changed"):
        load_partition(restored.root)


def test_failed_coverage_gate_persists_a_canonical_audit(tmp_path: Path) -> None:
    inputs = _fixture_inputs(tmp_path, "failed-output", "failed-work")
    with tarfile.open(inputs.metadata_archive_path, "r:gz") as archive:
        member = next(item for item in archive if item.isfile())
        stream = archive.extractfile(member)
        assert stream is not None
        records = json.loads(stream.read())
    archive_payload = json.dumps(records[:-1], separators=(",", ":")).encode()
    with tarfile.open(inputs.metadata_archive_path, "w:gz") as archive:
        member = tarfile.TarInfo("./data00.json")
        member.size = len(archive_payload)
        member.mtime = 0
        archive.addfile(member, io.BytesIO(archive_payload))
    inputs = replace(
        inputs,
        metadata_identity=_source_identity(inputs.metadata_archive_path),
    )

    with pytest.raises(SourceJoinError, match="coverage gates failed"):
        build_partition(inputs, _fixture_preset())

    audit_path = inputs.temporary_directory / "join-audit.json"
    payload = audit_path.read_bytes()
    audit = json.loads(payload)
    assert payload.endswith(b"\n")
    assert audit["format"] == "tinyworlds-p-source-join-audit"
    assert audit["passed"] is False
    assert [gate["passed"] for gate in audit["gates"]] == [False, True, False]
    assert audit["coverage"]["unmatched_group_count"] == 1


def _tiny_training_config(vocab_size: int) -> StreamingTrainingConfig:
    return StreamingTrainingConfig(
        model_config=GptNeoConfig(
            vocab_size=vocab_size,
            max_position_embeddings=8,
            hidden_size=8,
            intermediate_size=16,
            num_layers=1,
            num_heads=2,
            attention_types=("global",),
            local_window_size=4,
        ),
        epochs=2,
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


def test_streaming_training_resume_is_bit_identical(tmp_path: Path) -> None:
    artifact = build_partition(
        _fixture_inputs(tmp_path, "training-output", "partition-work"),
        _fixture_preset(),
    )
    config = _tiny_training_config(artifact.tokenizer_identity.vocab_size)

    uninterrupted = run_streaming_base_training(
        artifact,
        tmp_path / "uninterrupted",
        config,
    )
    interrupted = run_streaming_base_training(
        artifact,
        tmp_path / "resumed",
        config,
        stop_after_update=5,
    )
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
    validation = evaluate_partition_split(
        resumed.state.trainable,
        artifact,
        "base/validation",
        config.model_config,
    )
    assert validation.active_tokens > 0
    assert np.isfinite(validation.nll)


def _validation(epoch: int, held_in_nll: float, gaps: tuple[float, ...]) -> EpochValidation:
    return EpochValidation(
        epoch=epoch,
        held_in_nll=held_in_nll,
        world_gaps=tuple(
            WorldGap(world, 2.0 + gap, 2.0)
            for world, gap in zip(("A", "B", "C", "D", "E"), gaps, strict=True)
        ),
        allocator_peak_bytes=8 * 1024**3,
    )


def test_schedule_grid_fallback_and_best_epoch_boundaries() -> None:
    config = _tiny_training_config(32)

    assert float(cosine_learning_rate(0, 100, config)) == pytest.approx(1e-3)
    assert float(cosine_learning_rate(9, 100, config)) == pytest.approx(1e-2)
    assert float(cosine_learning_rate(99, 100, config)) == pytest.approx(1e-3)

    epoch_one = _validation(1, 2.1, (0.10,) * 5)
    passing = _validation(2, 2.0, (0.10, 0.11, 0.12, 0.13, 0.14))
    assert calibration_grid_decision(epoch_one, passing, 12 * 1024**3) == "pass"
    assert calibration_grid_decision(
        epoch_one,
        _validation(2, 2.0, (0.04, 0.10, 0.10, 0.10, 0.10)),
        12 * 1024**3,
    ) == "fallback_6x6"
    assert calibration_grid_decision(
        epoch_one,
        _validation(2, 2.0, (0.31,) * 5),
        12 * 1024**3,
    ) == "fallback_10x10"
    assert calibration_grid_decision(
        epoch_one,
        _validation(2, 2.09, (0.10,) * 5),
        12 * 1024**3,
    ) == "training_quality_failure"

    earlier = _validation(3, 1.9, (0.10,) * 5)
    tied_later = _validation(4, 1.9, (0.11,) * 5)
    assert select_best_eligible_epoch((passing, tied_later, earlier)).epoch == 3

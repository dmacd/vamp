from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import io
import json
from pathlib import Path
import shutil
import tarfile

import numpy as np
import pytest

from apm.data.text.tinyworlds_p import (
    HashedFile,
    PartitionArtifactError,
    PartitionInputs,
    PartitionPreset,
    SourceIdentity,
    TokenizerIdentity,
    build_partition,
    iter_partition_batches,
    load_partition,
)
from apm.data.text.tinyworlds_p.contracts import canonical_record_bytes
from apm.lm.text import TokenizersTextTokenizer


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _source_identity(path: Path) -> SourceIdentity:
    return SourceIdentity(
        dataset_id="tinyworlds-p/partition-fixture",
        revision="0" * 40,
        filename=path.name,
        size_bytes=path.stat().st_size,
        sha256=_sha256(path),
    )


def _fixture_inputs(
    tmp_path: Path,
    output_name: str,
    work_name: str,
) -> tuple[PartitionInputs, dict[tuple[str, int], bytes]]:
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
    vocabulary.update(
        {word: index for index, word in enumerate(vocabulary_words, start=2)}
    )
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer_directory = tmp_path / "tokenizer"
    tokenizer_directory.mkdir(exist_ok=True)
    tokenizer_path = tokenizer_directory / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    records: list[dict[str, object]] = []
    expected_stories: dict[tuple[str, int], bytes] = {}
    member_name = "./data00.json"
    for noun_index, noun in enumerate(nouns):
        for verb_index, verb in enumerate(verbs):
            for story_index in range(60):
                adjective = adjectives[story_index % len(adjectives)]
                story = (
                    f"A {adjective} {noun} will {verb} today {noun_index} "
                    f"{verb_index} {story_index}."
                )
                record = {
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
                expected_stories[(member_name, len(records))] = story.encode("utf-8")
                records.append(record)
    archive_path = tmp_path / "TinyStories_all_data.tar.gz"
    payload = json.dumps(records, separators=(",", ":")).encode("utf-8")
    with tarfile.open(archive_path, "w:gz") as archive:
        member = tarfile.TarInfo(member_name)
        member.size = len(payload)
        member.mtime = 0
        archive.addfile(member, io.BytesIO(payload))
    tokenizer_file = HashedFile(
        tokenizer_path.name,
        tokenizer_path.stat().st_size,
        _sha256(tokenizer_path),
    )
    inputs = PartitionInputs(
        archive_path=archive_path,
        tokenizer_directory=tokenizer_directory,
        output_root=tmp_path / output_name,
        temporary_directory=tmp_path / work_name,
        archive_identity=_source_identity(archive_path),
        tokenizer_identity=TokenizerIdentity(
            kind="word-level-fixture",
            identifier="tinyworlds-p/partition-fixture",
            revision="0" * 40,
            vocab_size=len(vocabulary),
            files=(tokenizer_file,),
        ),
    )
    return inputs, expected_stories


def _fixture_preset(worker_count: int = 1, run_record_count: int = 47) -> PartitionPreset:
    return PartitionPreset(
        bucket_count=3,
        worker_count=worker_count,
        run_record_count=run_record_count,
        shard_target_bytes=1_024,
        batch_block_documents=7,
        context_length=8,
        batch_size=4,
        minimum_role_coverage=1.0,
        selected_cell_median_tolerance=0.10,
        minimum_component_outside_groups=1,
        world_split_weights=(80, 10, 10),
        base_split_weights=(20, 40, 40),
    )


def _tree_bytes(root: Path) -> dict[Path, bytes]:
    return {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_archive_partition_builds_reconstructs_and_rebuilds_identically(
    tmp_path: Path,
) -> None:
    first_inputs, expected_stories = _fixture_inputs(
        tmp_path,
        "first-output",
        "first-work",
    )
    first = build_partition(first_inputs, _fixture_preset())
    restored = load_partition(first.root)

    assert restored.partition_sha256 == first.partition_sha256
    assert restored.archive_identity == first_inputs.archive_identity
    assert len(restored.cells) == 5
    assert len(restored.controls) == 10
    assert len(
        {group for control in restored.controls for group in control.group_sha256}
    ) == sum(len(control.group_sha256) for control in restored.controls)
    assert tuple(iter_partition_batches(restored, "base/train", epoch=0))

    documents = tuple(
        json.loads(line)
        for line in (restored.root / "documents.jsonl").read_text().splitlines()
    )
    tokenizer = TokenizersTextTokenizer.from_file(
        first_inputs.tokenizer_directory / "tokenizer.json"
    )
    text_shards: dict[int, bytes] = {}
    token_shards: dict[int, np.memmap] = {}
    for document in documents:
        expected = expected_stories[
            (document["source_member"], document["source_index"])
        ]
        assert sha256(expected).hexdigest() == document["story_sha256"]
        text_shards.setdefault(
            document["text_shard"],
            (
                restored.root
                / "shards"
                / f"text-{document['text_shard']:06d}.bin"
            ).read_bytes(),
        )
        reconstructed = text_shards[document["text_shard"]][
            document["text_offset"] : document["text_offset"]
            + document["text_bytes"]
        ]
        assert reconstructed == expected
        token_shards.setdefault(
            document["token_shard"],
            np.memmap(
                restored.root
                / "shards"
                / f"tokens-{document['token_shard']:06d}.uint16",
                dtype="<u2",
                mode="r",
            ),
        )
        token_start = document["token_offset"]
        token_stop = token_start + document["token_count"]
        assert tuple(
            int(token)
            for token in token_shards[document["token_shard"]][token_start:token_stop]
        ) == tokenizer.encode(expected.decode("utf-8"), add_eos=True)

    assignments = tuple(
        json.loads(line)
        for line in (restored.root / "assignments.jsonl").read_text().splitlines()
    )
    selected = {
        (cell.noun_bucket, cell.verb_bucket): cell.label for cell in restored.cells
    }
    assert all(
        (
            assignment["role"] == "world"
            and assignment["world"]
            == selected[(assignment["noun_bucket"], assignment["verb_bucket"])]
        )
        or (
            assignment["role"] == "base"
            and (assignment["noun_bucket"], assignment["verb_bucket"]) not in selected
        )
        for assignment in assignments
        if assignment["status"] == "eligible"
    )
    assert all(
        {"train", "validation", "test"}
        == {
            assignment["split"]
            for assignment in assignments
            if assignment["status"] == "eligible"
            and assignment["role"] == role
            and assignment["world"] == world
        }
        for role, world in (("base", None), *(('world', label) for label in "ABCDE"))
    )

    second_inputs = replace(
        first_inputs,
        output_root=tmp_path / "second-output",
        temporary_directory=tmp_path / "second-work",
    )
    second = build_partition(
        second_inputs,
        _fixture_preset(worker_count=2, run_record_count=31),
    )
    assert second.partition_sha256 == first.partition_sha256
    assert _tree_bytes(second.root) == _tree_bytes(first.root)


def test_partition_loader_rejects_shard_tampering(tmp_path: Path) -> None:
    inputs, _ = _fixture_inputs(tmp_path, "output", "work")
    artifact = build_partition(inputs, _fixture_preset())
    shard = next((artifact.root / "shards").glob("tokens-*.uint16"))
    payload = bytearray(shard.read_bytes())
    payload[0] ^= 1
    shard.write_bytes(payload)

    with pytest.raises(PartitionArtifactError, match="checksum changed"):
        load_partition(artifact.root)


def test_partition_loader_rejects_old_source_keys_even_when_tree_is_resealed(
    tmp_path: Path,
) -> None:
    inputs, _ = _fixture_inputs(tmp_path, "output", "work")
    artifact = build_partition(inputs, _fixture_preset())
    old_root = tmp_path / "old-artifact" / artifact.root.name
    shutil.copytree(artifact.root, old_root)

    partition_path = old_root / "partition.json"
    sources_path = old_root / "sources.json"
    partition = json.loads(partition_path.read_bytes())
    sources = partition["sources"]
    old_sources = {
        "corpus": sources["archive"],
        "metadata": sources["archive"],
        "tokenizer": sources["tokenizer"],
    }
    partition["sources"] = old_sources
    partition_path.write_bytes(canonical_record_bytes(partition))
    sources_path.write_bytes(canonical_record_bytes(old_sources))

    tree_path = old_root / "tree.json"
    tree = json.loads(tree_path.read_bytes())
    for descriptor in tree["files"]:
        if descriptor["relative_path"] in {"partition.json", "sources.json"}:
            path = old_root / descriptor["relative_path"]
            descriptor["size_bytes"] = path.stat().st_size
            descriptor["sha256"] = _sha256(path)
    tree_path.write_bytes(canonical_record_bytes(tree))

    with pytest.raises(PartitionArtifactError, match="partition sources"):
        load_partition(old_root)

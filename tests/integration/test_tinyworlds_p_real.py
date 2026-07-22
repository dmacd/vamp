"""Opt-in real archive replay, tokenizer validation, and rebuild gates."""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path

import numpy as np
import pytest

from apm.data.text.tinyworlds_p import (
    CANONICAL_ARCHIVE_IDENTITY,
    CANONICAL_TOKENIZER_IDENTITY,
    NORMALIZATION_IDENTITY,
    PARTITION_PRESET,
    PartitionInputs,
    build_archive_ingest,
    build_partition,
    iter_archive_groups,
    load_partition,
    read_spooled_story,
)
from apm.lm.text import TokenizersTextTokenizer


pytestmark = pytest.mark.integration

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_ARCHIVE_PATH = (
    _REPOSITORY_ROOT
    / "data"
    / "tinyworlds-v2"
    / "source"
    / CANONICAL_ARCHIVE_IDENTITY.filename
)
_TOKENIZER_DIRECTORY = (
    _REPOSITORY_ROOT / "checkpoints" / "tinystories-8m" / "tokenizer"
)
_PARTITION_ROOT = _REPOSITORY_ROOT / "data" / "tinyworlds-p-archive" / "v1"


def _published_partition():
    if not _PARTITION_ROOT.is_dir():
        pytest.skip("the canonical TinyWorlds-P partition has not been prepared")
    candidates = tuple(
        load_partition(path)
        for path in sorted(_PARTITION_ROOT.iterdir())
        if path.is_dir() and len(path.name) == 64 and (path / "tree.json").is_file()
    )
    selected = tuple(
        artifact for artifact in candidates if artifact.preset.bucket_count == 8
    )
    if len(selected) != 1:
        pytest.skip("exactly one strict canonical 8x8 partition is required")
    return selected[0]


def _canonical_inputs(output_root: Path, temporary_directory: Path) -> PartitionInputs:
    return PartitionInputs(
        archive_path=_ARCHIVE_PATH,
        tokenizer_directory=_TOKENIZER_DIRECTORY,
        output_root=output_root,
        temporary_directory=temporary_directory,
        archive_identity=CANONICAL_ARCHIVE_IDENTITY,
        tokenizer_identity=CANONICAL_TOKENIZER_IDENTITY,
    )


@pytest.mark.benchmark
def test_canonical_partition_replays_archive_and_real_tokenizer(tmp_path: Path) -> None:
    """Replay exact archive entities and authenticate their published tokens."""
    artifact = _published_partition()
    replay = build_archive_ingest(
        _canonical_inputs(tmp_path / "unused-output", tmp_path / "archive-replay"),
        PARTITION_PRESET,
        NORMALIZATION_IDENTITY,
    )
    tokenizer = TokenizersTextTokenizer.from_file(
        _TOKENIZER_DIRECTORY / "tokenizer.json"
    )
    documents = tuple(
        json.loads(line)
        for line in (artifact.root / "documents.jsonl").read_text().splitlines()[:256]
    )
    expected = {document["record_id"]: document for document in documents}
    replayed: dict[str, bytes] = {}
    for group in iter_archive_groups(replay.groups_path):
        for occurrence in group["occurrences"]:
            if occurrence["record_id"] in expected:
                replayed[occurrence["record_id"]] = read_spooled_story(
                    replay.story_spool_path,
                    occurrence,
                )
    assert set(replayed) == set(expected)
    text_shards: dict[int, bytes] = {}
    token_shards: dict[int, np.memmap] = {}
    for record_id, raw_story in replayed.items():
        document = expected[record_id]
        text_shard_id = document["text_shard"]
        text_shards.setdefault(
            text_shard_id,
            (artifact.root / "shards" / f"text-{text_shard_id:06d}.bin").read_bytes(),
        )
        assert raw_story == text_shards[text_shard_id][
            document["text_offset"] : document["text_offset"] + document["text_bytes"]
        ]
        token_shard_id = document["token_shard"]
        token_shards.setdefault(
            token_shard_id,
            np.memmap(
                artifact.root / "shards" / f"tokens-{token_shard_id:06d}.uint16",
                dtype="<u2",
                mode="r",
            ),
        )
        token_start = document["token_offset"]
        token_stop = token_start + document["token_count"]
        assert tuple(
            int(token)
            for token in token_shards[token_shard_id][token_start:token_stop]
        ) == tokenizer.encode(raw_story.decode("utf-8"), add_eos=True)


@pytest.mark.benchmark
def test_canonical_partition_full_rebuild_is_byte_identical(tmp_path: Path) -> None:
    """Rebuild all real sources with different execution-only parallelism."""
    if os.environ.get("TINYWORLDS_P_FULL_REBUILD") != "1":
        pytest.skip("set TINYWORLDS_P_FULL_REBUILD=1 for the multi-hour rebuild gate")
    published = _published_partition()
    rebuilt = build_partition(
        _canonical_inputs(tmp_path / "output", tmp_path / "work"),
        replace(PARTITION_PRESET, worker_count=7, run_record_count=37_001),
    )
    assert rebuilt.partition_sha256 == published.partition_sha256
    published_files = {
        path.relative_to(published.root): path.read_bytes()
        for path in published.root.rglob("*")
        if path.is_file()
    }
    rebuilt_files = {
        path.relative_to(rebuilt.root): path.read_bytes()
        for path in rebuilt.root.rglob("*")
        if path.is_file()
    }
    assert rebuilt_files == published_files

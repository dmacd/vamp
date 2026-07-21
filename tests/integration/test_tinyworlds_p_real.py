"""Opt-in real-source, rebuild, tokenizer-replay, and RTX training gates."""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path

import numpy as np
import pytest

from apm.data.text.tinyworlds_p import (
    CANONICAL_CORPUS_IDENTITY,
    CANONICAL_METADATA_IDENTITY,
    CANONICAL_TOKENIZER_IDENTITY,
    PARTITION_PRESET,
    PartitionInputs,
    StreamingTrainingConfig,
    build_partition,
    load_partition,
    run_streaming_base_training,
)
from apm.lm.text import TokenizersTextTokenizer


pytestmark = pytest.mark.integration

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_CORPUS_PATH = (
    _REPOSITORY_ROOT
    / "data"
    / "tinystories-original"
    / CANONICAL_CORPUS_IDENTITY.filename
)
_METADATA_PATH = (
    _REPOSITORY_ROOT
    / "data"
    / "tinyworlds-v2"
    / "source"
    / CANONICAL_METADATA_IDENTITY.filename
)
_TOKENIZER_DIRECTORY = (
    _REPOSITORY_ROOT / "checkpoints" / "tinystories-8m" / "tokenizer"
)
_PARTITION_ROOT = _REPOSITORY_ROOT / "data" / "tinyworlds-p" / "v1"


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
        corpus_path=_CORPUS_PATH,
        metadata_archive_path=_METADATA_PATH,
        tokenizer_directory=_TOKENIZER_DIRECTORY,
        output_root=output_root,
        temporary_directory=temporary_directory,
        corpus_identity=CANONICAL_CORPUS_IDENTITY,
        metadata_identity=CANONICAL_METADATA_IDENTITY,
        tokenizer_identity=CANONICAL_TOKENIZER_IDENTITY,
    )


def test_canonical_partition_replays_original_bytes_and_real_tokenizer() -> None:
    """Authenticate a real publication and replay source and token offsets."""
    artifact = _published_partition()
    tokenizer = TokenizersTextTokenizer.from_file(
        _TOKENIZER_DIRECTORY / "tokenizer.json"
    )
    checked = 0
    with (
        _CORPUS_PATH.open("rb") as corpus,
        (artifact.root / "documents.jsonl").open("rb") as documents,
    ):
        text_shards: dict[int, bytes] = {}
        token_shards: dict[int, np.memmap] = {}
        for line in documents:
            document = json.loads(line)
            corpus.seek(document["source_byte_offset"])
            raw_story = corpus.read(document["byte_length"])
            assert len(raw_story) == document["byte_length"]
            text_shard_id = document["text_shard"]
            if text_shard_id not in text_shards:
                text_shards[text_shard_id] = (
                    artifact.root
                    / "shards"
                    / f"text-{text_shard_id:06d}.bin"
                ).read_bytes()
            assert raw_story == text_shards[text_shard_id][
                document["text_offset"] : document["text_offset"]
                + document["text_bytes"]
            ]
            token_shard_id = document["token_shard"]
            if token_shard_id not in token_shards:
                token_shards[token_shard_id] = np.memmap(
                    artifact.root
                    / "shards"
                    / f"tokens-{token_shard_id:06d}.uint16",
                    dtype="<u2",
                    mode="r",
                )
            token_start = document["token_offset"]
            token_stop = token_start + document["token_count"]
            assert tuple(
                int(token)
                for token in token_shards[token_shard_id][token_start:token_stop]
            ) == tokenizer.encode(raw_story.decode("utf-8"), add_eos=True)
            checked += 1
            if checked == 256:
                break
    assert checked == 256


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


@pytest.mark.benchmark
def test_rtx_4090_training_and_resume_smoke(tmp_path: Path) -> None:
    """Compile production training and prove one-update GPU resume parity."""
    if os.environ.get("TINYWORLDS_P_GPU_SMOKE") != "1":
        pytest.skip("set TINYWORLDS_P_GPU_SMOKE=1 for the RTX resume smoke")
    jax = pytest.importorskip("jax")
    devices = tuple(device for device in jax.devices() if device.platform == "gpu")
    if len(devices) != 1:
        pytest.skip("the RTX smoke requires exactly one visible GPU")
    artifact = _published_partition()
    config = StreamingTrainingConfig.from_preset()
    interrupted = run_streaming_base_training(
        artifact,
        tmp_path / "training",
        config,
        stop_after_update=1,
    )
    resumed = run_streaming_base_training(
        artifact,
        tmp_path / "training",
        config,
        resume_from=interrupted.checkpoints[-1].directory,
        stop_after_update=2,
    )
    assert interrupted.cursor.optimizer_update == 1
    assert resumed.cursor.optimizer_update == 2
    assert tuple(device.platform for device in jax.devices()) == ("gpu",)

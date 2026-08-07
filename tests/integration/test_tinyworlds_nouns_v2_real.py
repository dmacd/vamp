from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest

from apm.data.text.tinyworlds_nouns_v1.partition import load_noun_partition
from apm.data.text.tinyworlds_nouns_v2.contracts import (
    BASE_UNIVERSE_STORY_COUNT,
    EXCLUDED_TRAIN_STORY_COUNT,
    EXCLUDED_VALIDATION_STORY_COUNT,
    EXPECTED_PURE_COUNTS,
    PARENT_PARTITION_SHA256,
    PURE_TASK_TRAIN_STORY_COUNT,
    PURE_TASK_VALIDATION_STORY_COUNT,
    TRAIN_UNIQUE_STORY_COUNT,
)
from apm.data.text.tinyworlds_nouns_v2.partition import (
    authenticate_parent_manifest,
    find_partition,
    verify_byte_identical_rebuild,
)


pytestmark = pytest.mark.integration
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PARENT_ROOT = (
    REPOSITORY_ROOT
    / "data/tinyworlds-nouns-v1/partitions"
    / PARENT_PARTITION_SHA256
)
V2_ROOT = REPOSITORY_ROOT / "data/tinyworlds-nouns-v2"


def test_real_24_task_counts_and_byte_identical_rebuild() -> None:
    manifest = authenticate_parent_manifest(PARENT_ROOT)
    partition = find_partition(manifest, V2_ROOT)
    assert partition is not None
    assert partition.base_universe_story_count == BASE_UNIVERSE_STORY_COUNT
    assert partition.base_universe_story_count / TRAIN_UNIQUE_STORY_COUNT == pytest.approx(
        0.8136, abs=5e-5
    )
    assert sum(task.train_story_count for task in partition.tasks) == (
        PURE_TASK_TRAIN_STORY_COUNT
    )
    assert sum(task.validation_story_count for task in partition.tasks) == (
        PURE_TASK_VALIDATION_STORY_COUNT
    )
    assert partition.excluded_train_story_count == EXCLUDED_TRAIN_STORY_COUNT
    assert partition.excluded_validation_story_count == EXCLUDED_VALIDATION_STORY_COUNT
    assert tuple(
        (task.task_id, task.train_story_count, task.validation_story_count)
        for task in partition.tasks
    ) == EXPECTED_PURE_COUNTS
    verify_byte_identical_rebuild(partition, manifest)


def test_completed_nouns_v1_strict_load_and_published_hashes_are_unchanged() -> None:
    partition = load_noun_partition(PARENT_ROOT / "partition.json")
    assert partition.partition_sha256 == PARENT_PARTITION_SHA256
    expected = {
        "results/language_cl/tinyworlds-nouns-v1/report.md": (
            "74f0035c755f95ff57624f8270615f3c9171f7b792d7b5ee22bae147ac15c4ae"
        ),
        "results/language_cl/tinyworlds-nouns-v1/report.html": (
            "9ef9cfea2c827836da8999c3d032fa63ff3a109664ee65b250066577a09d1526"
        ),
        "results/language_cl/tinyworlds-nouns-v1/run-manifest.json": (
            "fffa0e0f64f1c63ae3efb16363042907169737dc6858944cbcd0f46b178cc628"
        ),
    }
    for relative, digest in expected.items():
        assert sha256((REPOSITORY_ROOT / relative).read_bytes()).hexdigest() == digest

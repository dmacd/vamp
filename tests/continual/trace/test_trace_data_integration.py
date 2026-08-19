from __future__ import annotations

from collections import Counter
import os
from pathlib import Path

import pytest

from apm.continual.trace.data import extract_pinned_archive, prepare_dataset


@pytest.mark.integration
def test_pinned_tree_lora_archive_reproduces_the_registered_manifest(
    tmp_path: Path,
) -> None:
    archive = Path(
        os.environ.get(
            "TRACE_DATASET_ARCHIVE",
            "/tmp/LLM-CL-Benchmark_500.tar.xz",
        )
    )
    if not archive.is_file():
        pytest.skip("set TRACE_DATASET_ARCHIVE to the pinned TreeLoRA archive")
    source = extract_pinned_archive(archive, tmp_path / "source")

    manifest = prepare_dataset(
        source,
        tmp_path / "dataset.json",
        tmp_path / "examples.jsonl",
    )

    assert manifest.manifest_sha256 == (
        "19fe258e74f5dba6408e9b498fb1b5e4c4dac16d4840363d142bd89a19e47ba2"
    )
    assert Counter(example.split for example in manifest.examples) == {
        "train": 4_000,
        "validation": 741,
        "test": 781,
    }

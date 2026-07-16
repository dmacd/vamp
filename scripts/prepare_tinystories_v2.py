"""Download and verify the exact supported TinyStories V2/GPT-4 aggregates."""

from __future__ import annotations

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATA_DIRECTORY = REPOSITORY_ROOT / "data" / "tinystories-v2"


def main() -> None:
    """Materialize the pinned files locally and verify size plus SHA-256."""
    from huggingface_hub import snapshot_download

    from apm.data.text.curricula import (
        TINYSTORIES_V2_SOURCE,
        verify_pinned_dataset_file,
    )

    expected_files = (
        TINYSTORIES_V2_SOURCE.train_file,
        TINYSTORIES_V2_SOURCE.validation_file,
    )
    snapshot_download(
        repo_id=TINYSTORIES_V2_SOURCE.dataset_id,
        repo_type="dataset",
        revision=TINYSTORIES_V2_SOURCE.revision,
        allow_patterns=[file.filename for file in expected_files],
        local_dir=DATA_DIRECTORY,
    )
    for expected_file in expected_files:
        verified = verify_pinned_dataset_file(
            DATA_DIRECTORY / expected_file.filename,
            expected_file,
        )
        print(verified)


if __name__ == "__main__":
    main()

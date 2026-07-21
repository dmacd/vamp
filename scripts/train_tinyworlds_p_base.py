#!/usr/bin/env python3
"""Calibrate, train, select, test, and publish the fixed TinyWorlds-P base."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from pathlib import Path
import tempfile
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tqdm import tqdm as Tqdm

    from apm.data.text.tinyworlds_p import (
        PartitionArtifact,
        PartitionInputs,
        TrainingCursor,
    )


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class _TrainingReporter:
    """Render optimizer progress with update and overall ETA."""

    def __init__(self, description: str) -> None:
        self._description = description
        self._progress: Tqdm | None = None
        self._started = time.monotonic()
        self._initial_update = 0

    def __call__(
        self,
        cursor: TrainingCursor,
        nll: float,
        planned_updates: int,
    ) -> None:
        """Advance the human-facing optimizer progress bar."""
        from tqdm import tqdm

        if self._progress is None:
            self._initial_update = max(0, cursor.optimizer_update - 1)
            self._progress = tqdm(
                total=planned_updates,
                initial=self._initial_update,
                desc=self._description,
                unit="update",
            )
        self._progress.update(cursor.optimizer_update - self._progress.n)
        completed_here = max(1, cursor.optimizer_update - self._initial_update)
        remaining_seconds = (
            (time.monotonic() - self._started)
            * (planned_updates - cursor.optimizer_update)
            / completed_here
        )
        self._progress.set_postfix_str(
            f"epoch {cursor.epoch + 1}, NLL {nll:.4f}, "
            f"overall ETA {timedelta(seconds=max(0, round(remaining_seconds)))}"
        )

    def close(self) -> None:
        """Close a constructed progress bar."""
        if self._progress is not None:
            self._progress.close()


def _fixed_partition_inputs(temporary_directory: Path) -> PartitionInputs:
    from apm.data.text.tinyworlds_p import (
        CANONICAL_CORPUS_IDENTITY,
        CANONICAL_METADATA_IDENTITY,
        CANONICAL_TOKENIZER_IDENTITY,
        PartitionInputs,
    )

    return PartitionInputs(
        corpus_path=(
            REPOSITORY_ROOT
            / "data"
            / "tinystories-original"
            / CANONICAL_CORPUS_IDENTITY.filename
        ),
        metadata_archive_path=(
            REPOSITORY_ROOT
            / "data"
            / "tinyworlds-v2"
            / "source"
            / CANONICAL_METADATA_IDENTITY.filename
        ),
        tokenizer_directory=(
            REPOSITORY_ROOT / "checkpoints" / "tinystories-8m" / "tokenizer"
        ),
        output_root=REPOSITORY_ROOT / "data" / "tinyworlds-p" / "v1",
        temporary_directory=temporary_directory,
        corpus_identity=CANONICAL_CORPUS_IDENTITY,
        metadata_identity=CANONICAL_METADATA_IDENTITY,
        tokenizer_identity=CANONICAL_TOKENIZER_IDENTITY,
        progress=lambda event: print(
            f"[{event.phase}] {event.detail} | phase ETA pending | overall ETA pending",
            flush=True,
        ),
    )


def _initial_partition() -> PartitionArtifact:
    from apm.data.text.tinyworlds_p import load_partition

    partition_root = REPOSITORY_ROOT / "data" / "tinyworlds-p" / "v1"
    candidates = tuple(
        load_partition(path)
        for path in sorted(partition_root.iterdir())
        if path.is_dir() and len(path.name) == 64 and (path / "tree.json").is_file()
    )
    eight_bucket = tuple(
        artifact for artifact in candidates if artifact.preset.bucket_count == 8
    )
    if len(eight_bucket) != 1:
        raise RuntimeError(
            "the fixed trainer requires exactly one strict 8x8 partition; "
            "run scripts/prepare_tinyworlds_p.py first"
        )
    return eight_bucket[0]


def main() -> int:
    """Execute the fixed one-fallback calibration and final publication policy."""
    import jax

    from apm.data.text.tinyworlds_p import (
        PARTITION_PRESET,
        StreamingTrainingConfig,
        build_partition,
    )
    from apm.data.text.tinyworlds_p.milestone import (
        finish_and_publish_base,
        run_calibration_attempt,
    )

    devices = tuple(device for device in jax.devices() if device.platform == "gpu")
    if len(devices) != 1:
        raise RuntimeError(
            "TinyWorlds-P base training requires the single RTX GPU; "
            "run this fixed script outside the filesystem sandbox"
        )
    work_root = REPOSITORY_ROOT / "checkpoints" / "tinyworlds-p-v1" / "work"
    work_root.mkdir(parents=True, exist_ok=True)
    temporary_directory = Path(
        tempfile.mkdtemp(prefix="base-v1-", dir=work_root)
    )
    print(f"temporary directory: {temporary_directory}", flush=True)
    config = StreamingTrainingConfig.from_preset()
    artifact = _initial_partition()
    print(
        f"[calibration] grid 8x8, epochs 1-2 | phase ETA 2h | overall ETA 6h",
        flush=True,
    )
    reporter = _TrainingReporter("TinyWorlds-P 8x8 calibration")
    calibration = run_calibration_attempt(
        artifact,
        temporary_directory / "grid-8-training",
        config,
        progress=reporter,
    )
    reporter.close()
    fallback_count = {
        "fallback_6x6": 6,
        "fallback_10x10": 10,
    }.get(calibration.decision)
    if fallback_count is not None:
        print(
            f"[fallback] rebuilding the single allowed {fallback_count}x{fallback_count} grid "
            "| phase ETA 1h | overall ETA 7h",
            flush=True,
        )
        fallback_partition = build_partition(
            _fixed_partition_inputs(
                temporary_directory / f"grid-{fallback_count}-partition"
            ),
            replace(PARTITION_PRESET, bucket_count=fallback_count),
        )
        reporter = _TrainingReporter(
            f"TinyWorlds-P {fallback_count}x{fallback_count} calibration"
        )
        calibration = run_calibration_attempt(
            fallback_partition,
            temporary_directory / f"grid-{fallback_count}-training",
            config,
            progress=reporter,
        )
        reporter.close()
    if calibration.decision != "pass":
        print(
            f"[stop] calibration did not pass: {calibration.decision}; no final base trained",
            flush=True,
        )
        return 2
    print(
        "[final] resuming the passing run through epoch 5 | phase ETA 3h | overall ETA 4h",
        flush=True,
    )
    reporter = _TrainingReporter("TinyWorlds-P final base")
    published = finish_and_publish_base(
        calibration,
        REPOSITORY_ROOT / "checkpoints" / "tinyworlds-p-v1",
        REPOSITORY_ROOT / "checkpoints" / "tinystories-8m" / "tokenizer",
        progress=reporter,
    )
    reporter.close()
    print(f"training publication: {published.directory}")
    print(f"selected epoch: {published.selected_epoch}")
    print(f"training SHA-256: {published.training_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

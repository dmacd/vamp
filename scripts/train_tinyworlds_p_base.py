#!/usr/bin/env python3
"""Calibrate, train, select, test, and publish the fixed TinyWorlds-P base."""

from __future__ import annotations

from datetime import timedelta
import math
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
        CANONICAL_ARCHIVE_IDENTITY,
        CANONICAL_TOKENIZER_IDENTITY,
        PartitionInputs,
    )

    return PartitionInputs(
        archive_path=(
            REPOSITORY_ROOT
            / "data"
            / "tinyworlds-v2"
            / "source"
            / CANONICAL_ARCHIVE_IDENTITY.filename
        ),
        tokenizer_directory=(
            REPOSITORY_ROOT / "checkpoints" / "tinystories-8m" / "tokenizer"
        ),
        output_root=(
            REPOSITORY_ROOT / "data" / "tinyworlds-p-archive" / "v1"
        ),
        temporary_directory=temporary_directory,
        archive_identity=CANONICAL_ARCHIVE_IDENTITY,
        tokenizer_identity=CANONICAL_TOKENIZER_IDENTITY,
        progress=lambda event: print(
            f"[{event.phase}] {event.detail} | phase ETA pending | overall ETA pending",
            flush=True,
        ),
    )


def _initial_partition() -> PartitionArtifact:
    from apm.data.text.tinyworlds_p import load_partition

    partition_root = REPOSITORY_ROOT / "data" / "tinyworlds-p-archive" / "v1"
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


def _print_archive_training_plan(artifact: PartitionArtifact, config) -> None:
    """Derive update and runtime planning numbers from this archive partition."""
    from apm.data.text.tinyworlds_p import count_partition_microbatches

    active_tokens = next(
        count.active_token_count
        for count in artifact.split_counts
        if count.role == "base" and count.world is None and count.split == "train"
    )
    microbatches = count_partition_microbatches(artifact, "base/train")
    updates = math.ceil(microbatches / config.accumulation_microbatches)
    planned_tokens = active_tokens * config.epochs
    planning_throughput = 100_000
    estimated_seconds = planned_tokens / planning_throughput
    print(
        f"[plan] archive base/train tokens per epoch {active_tokens:,}; "
        f"microbatches {microbatches:,}; updates per epoch {updates:,}; "
        f"five-epoch tokens {planned_tokens:,}; updates {updates * config.epochs:,}; "
        f"planning ETA {timedelta(seconds=round(estimated_seconds))} "
        f"at {planning_throughput:,} active tokens/s",
        flush=True,
    )


def main() -> int:
    """Execute the fixed one-fallback calibration and final publication policy."""
    import jax

    from apm.data.text.tinyworlds_p import (
        PARTITION_PRESET,
        StreamingTrainingConfig,
        build_partition,
        fallback_partition_preset,
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
    work_root = REPOSITORY_ROOT / "checkpoints" / "tinyworlds-p-archive-v1" / "work"
    work_root.mkdir(parents=True, exist_ok=True)
    temporary_directory = Path(
        tempfile.mkdtemp(prefix="base-archive-v1-", dir=work_root)
    )
    print(f"temporary directory: {temporary_directory}", flush=True)
    config = StreamingTrainingConfig.from_preset()
    artifact = _initial_partition()
    _print_archive_training_plan(artifact, config)
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
        fallback_preset = fallback_partition_preset(
            calibration.decision,
            PARTITION_PRESET,
        )
        fallback_partition = build_partition(
            _fixed_partition_inputs(
                temporary_directory / f"grid-{fallback_count}-partition"
            ),
            fallback_preset,
        )
        _print_archive_training_plan(fallback_partition, config)
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
        REPOSITORY_ROOT / "checkpoints" / "tinyworlds-p-archive-v1",
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

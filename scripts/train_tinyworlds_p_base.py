#!/usr/bin/env python3
"""Calibrate, train, select, test, and publish the fixed TinyWorlds-P base."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
import math
import os
from pathlib import Path
import tempfile
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tqdm import tqdm as Tqdm

    from apm.data.text.tinyworlds_p import (
        PartitionArtifact,
        PartitionInputs,
        ProgressEvent,
        StreamingTrainingConfig,
        TrainingCursor,
    )


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_PLANNING_THROUGHPUT = 100_000
_PARTITION_REBUILD_ESTIMATE_SECONDS = 40 * 60

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


@dataclass(frozen=True, slots=True)
class _ArchiveTrainingPlan:
    active_tokens_per_epoch: int
    microbatches_per_epoch: int
    updates_per_epoch: int
    epoch_seconds: float
    epochs: int
    calibration_epochs: int

    def __post_init__(self) -> None:
        if not 0 < self.calibration_epochs < self.epochs:
            raise ValueError("calibration epochs must be a strict prefix of training")

    @property
    def total_updates(self) -> int:
        return self.updates_per_epoch * self.epochs

    @property
    def calibration_updates(self) -> int:
        return self.updates_per_epoch * self.calibration_epochs

    @property
    def remaining_epochs(self) -> int:
        return self.epochs - self.calibration_epochs

    def seconds_for_epochs(self, epochs: int) -> float:
        return self.epoch_seconds * epochs


class _TrainingReporter:
    """Render one training phase with adaptive phase and pass-path ETAs."""

    def __init__(
        self,
        description: str,
        phase_end_update: int,
        overall_end_update: int,
    ) -> None:
        self._description = description
        self._phase_end_update = phase_end_update
        self._overall_end_update = overall_end_update
        self._progress: Tqdm | None = None
        self._started: float | None = None
        self._initial_update = 0
        self._timed_initial_update = 0

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
            if not (
                self._initial_update < self._phase_end_update
                <= self._overall_end_update
                <= planned_updates
            ):
                raise ValueError("training reporter update bounds are inconsistent")
            self._progress = tqdm(
                total=self._phase_end_update - self._initial_update,
                desc=self._description,
                unit="update",
                mininterval=120.0,
                maxinterval=300.0,
            )
            self._started = time.monotonic()
            self._timed_initial_update = cursor.optimizer_update
        phase_updates = cursor.optimizer_update - self._initial_update
        self._progress.update(phase_updates - self._progress.n)
        assert self._started is not None
        timed_updates = cursor.optimizer_update - self._timed_initial_update
        if timed_updates:
            seconds_per_update = (time.monotonic() - self._started) / timed_updates
            phase_eta = _duration(
                seconds_per_update
                * max(0, self._phase_end_update - cursor.optimizer_update)
            )
            overall_eta = _duration(
                seconds_per_update
                * max(0, self._overall_end_update - cursor.optimizer_update)
            )
        else:
            phase_eta = overall_eta = "pending"
        self._progress.set_postfix_str(
            f"epoch {cursor.epoch + 1}, NLL {nll:.4f}, "
            f"phase ETA {phase_eta}, overall ETA {overall_eta}",
            refresh=False,
        )

    def close(self) -> None:
        """Close a constructed progress bar."""
        if self._progress is not None:
            self._progress.close()


class _EvaluationReporter:
    """Render ordered GPU evaluation with measured phase and pass-path ETAs."""

    def __init__(
        self,
        artifact: PartitionArtifact,
        description: str,
        phases: tuple[tuple[int | None, str], ...],
        overall_after_seconds: float,
    ) -> None:
        from apm.data.text.tinyworlds_p import count_partition_microbatches

        split_totals = {
            split: count_partition_microbatches(artifact, split)
            for split in dict.fromkeys(split for _, split in phases)
        }
        self._description = description
        self._planned = {
            (epoch, split): split_totals[split] for epoch, split in phases
        }
        self._overall_after_seconds = overall_after_seconds
        self._completed: dict[tuple[int | None, str], int] = {}
        self._progress: Tqdm | None = None
        self._started: float | None = None
        self._timed_initial_batches = 0

    def __call__(self, epoch: int, split: str, completed: int, total: int) -> None:
        """Advance the evaluation bar after one synchronized GPU batch."""
        from tqdm import tqdm

        key = (None if split.endswith("/test") else epoch, split)
        expected = self._planned.get(key)
        previous = self._completed.get(key, 0)
        if expected is None or total != expected or not previous <= completed <= total:
            raise ValueError("evaluation reporter progress is inconsistent")
        if self._progress is None:
            self._progress = tqdm(
                total=sum(self._planned.values()),
                desc=self._description,
                unit="batch",
                mininterval=120.0,
                maxinterval=300.0,
            )
            self._started = time.monotonic()
            self._timed_initial_batches = self._progress.n
        self._completed[key] = completed
        self._progress.update(completed - previous)
        assert self._started is not None
        measured_batches = self._progress.n - self._timed_initial_batches
        if measured_batches:
            seconds_per_batch = (time.monotonic() - self._started) / measured_batches
            phase_seconds = seconds_per_batch * (
                self._progress.total - self._progress.n
            )
            phase_eta = _duration(phase_seconds)
            overall_eta = _duration(phase_seconds + self._overall_after_seconds)
        else:
            phase_eta = overall_eta = "pending"
        self._progress.set_postfix_str(
            f"epoch {epoch}, {split}, phase ETA {phase_eta}, "
            f"pass-path ETA {overall_eta}",
            refresh=False,
        )

    def close(self) -> None:
        """Close a constructed evaluation progress bar."""
        if self._progress is not None:
            self._progress.close()


def _evaluation_splits(split: str) -> tuple[str, ...]:
    """Return held-in, world, and matched-control splits in evaluation order."""
    from apm.data.text.tinyworlds_p.contracts import WORLD_LABELS

    return (
        f"base/{split}",
        *(
            f"{role}/{world}/{split}"
            for world in WORLD_LABELS
            for role in ("world", "control")
        ),
    )


def _evaluation_phases(
    epochs: tuple[int, ...],
    split: str,
) -> tuple[tuple[int, str], ...]:
    """Expand ordered epochs over one complete evaluation split family."""
    return tuple(
        (epoch, split_name)
        for epoch in epochs
        for split_name in _evaluation_splits(split)
    )


def _fixed_partition_inputs(
    temporary_directory: Path,
    progress: Callable[[ProgressEvent], None],
) -> PartitionInputs:
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
        progress=progress,
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


def _print_archive_training_plan(
    artifact: PartitionArtifact,
    config: StreamingTrainingConfig,
) -> _ArchiveTrainingPlan:
    """Derive update and runtime planning numbers from this archive partition."""
    from apm.data.text.tinyworlds_p import count_partition_microbatches

    active_tokens = next(
        count.active_token_count
        for count in artifact.split_counts
        if count.role == "base" and count.world is None and count.split == "train"
    )
    microbatches = count_partition_microbatches(artifact, "base/train")
    updates = math.ceil(microbatches / config.accumulation_microbatches)
    plan = _ArchiveTrainingPlan(
        active_tokens_per_epoch=active_tokens,
        microbatches_per_epoch=microbatches,
        updates_per_epoch=updates,
        epoch_seconds=active_tokens / _PLANNING_THROUGHPUT,
        epochs=config.epochs,
        calibration_epochs=config.calibration_epochs,
    )
    planned_tokens = active_tokens * plan.epochs
    print(
        f"[plan] archive base/train tokens per epoch {active_tokens:,}; "
        f"microbatches {microbatches:,}; updates per epoch {updates:,}; "
        f"{plan.epochs}-epoch tokens {planned_tokens:,}; updates {plan.total_updates:,}; "
        f"planning ETA {_duration(plan.seconds_for_epochs(plan.epochs))} "
        f"at {_PLANNING_THROUGHPUT:,} active tokens/s",
        flush=True,
    )
    return plan


def _duration(seconds: float) -> str:
    return str(timedelta(seconds=max(0, round(seconds))))


def _print_training_phase(
    label: str,
    detail: str,
    plan: _ArchiveTrainingPlan,
    phase_epochs: int,
    overall_epochs: int | None = None,
) -> None:
    remaining_epochs = plan.epochs if overall_epochs is None else overall_epochs
    print(
        f"[{label}] {detail} | "
        f"phase ETA {_duration(plan.seconds_for_epochs(phase_epochs))} | "
        f"overall ETA {_duration(plan.seconds_for_epochs(remaining_epochs))}",
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
    from apm.data.text.tinyworlds_p.progress import PreparationReporter

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
    print(
        "[load] strictly authenticating the 18 GB archive partition with 24 checksum "
        "workers and three semantic workers | phase ETA 0:03:00 | overall ETA pending",
        flush=True,
    )
    load_started = time.monotonic()
    artifact = _initial_partition()
    print(
        f"[load] strict partition authentication passed in "
        f"{_duration(time.monotonic() - load_started)}",
        flush=True,
    )
    plan = _print_archive_training_plan(artifact, config)
    _print_training_phase(
        "calibration",
        f"grid 8x8, epochs 1-{plan.calibration_epochs}",
        plan,
        plan.calibration_epochs,
    )
    reporter = _TrainingReporter(
        "TinyWorlds-P 8x8 calibration",
        plan.calibration_updates,
        plan.total_updates,
    )
    evaluation_reporter = _EvaluationReporter(
        artifact,
        "TinyWorlds-P 8x8 calibration validation",
        _evaluation_phases(tuple(range(1, plan.calibration_epochs + 1)), "validation"),
        plan.seconds_for_epochs(plan.remaining_epochs),
    )
    try:
        calibration = run_calibration_attempt(
            artifact,
            temporary_directory / "grid-8-training",
            config,
            progress=reporter,
            evaluation_progress=evaluation_reporter,
        )
    finally:
        reporter.close()
        evaluation_reporter.close()
    fallback_count = {
        "fallback_6x6": 6,
        "fallback_10x10": 10,
    }.get(calibration.decision)
    if fallback_count is not None:
        print(
            f"[fallback] rebuilding the single allowed {fallback_count}x{fallback_count} grid "
            f"| phase ETA {_duration(_PARTITION_REBUILD_ESTIMATE_SECONDS)} | "
            f"preliminary overall ETA "
            f"{_duration(_PARTITION_REBUILD_ESTIMATE_SECONDS + plan.seconds_for_epochs(plan.epochs))}",
            flush=True,
        )
        fallback_preset = fallback_partition_preset(
            calibration.decision,
            PARTITION_PRESET,
        )
        partition_reporter = PreparationReporter()
        try:
            fallback_partition = build_partition(
                _fixed_partition_inputs(
                    temporary_directory / f"grid-{fallback_count}-partition",
                    partition_reporter,
                ),
                fallback_preset,
            )
        finally:
            partition_reporter.close()
        plan = _print_archive_training_plan(fallback_partition, config)
        _print_training_phase(
            "calibration",
            f"grid {fallback_count}x{fallback_count}, epochs 1-{plan.calibration_epochs}",
            plan,
            plan.calibration_epochs,
        )
        reporter = _TrainingReporter(
            f"TinyWorlds-P {fallback_count}x{fallback_count} calibration",
            plan.calibration_updates,
            plan.total_updates,
        )
        evaluation_reporter = _EvaluationReporter(
            fallback_partition,
            f"TinyWorlds-P {fallback_count}x{fallback_count} calibration validation",
            _evaluation_phases(
                tuple(range(1, plan.calibration_epochs + 1)),
                "validation",
            ),
            plan.seconds_for_epochs(plan.remaining_epochs),
        )
        try:
            calibration = run_calibration_attempt(
                fallback_partition,
                temporary_directory / f"grid-{fallback_count}-training",
                config,
                progress=reporter,
                evaluation_progress=evaluation_reporter,
            )
        finally:
            reporter.close()
            evaluation_reporter.close()
    if calibration.decision != "pass":
        print(
            f"[stop] calibration did not pass: {calibration.decision}; no final base trained",
            flush=True,
        )
        return 2
    remaining_epochs = plan.remaining_epochs
    _print_training_phase(
        "final",
        f"resuming the passing run through epoch {config.epochs}",
        plan,
        remaining_epochs,
        remaining_epochs,
    )
    reporter = _TrainingReporter(
        "TinyWorlds-P final base",
        plan.total_updates,
        plan.total_updates,
    )
    validation_epochs = tuple(range(plan.calibration_epochs + 1, plan.epochs + 1))
    evaluation_reporter = _EvaluationReporter(
        calibration.artifact,
        "TinyWorlds-P final validation and sealed test",
        _evaluation_phases(validation_epochs, "validation")
        + tuple((None, split) for split in _evaluation_splits("test")),
        0.0,
    )
    try:
        published = finish_and_publish_base(
            calibration,
            REPOSITORY_ROOT / "checkpoints" / "tinyworlds-p-archive-v1",
            REPOSITORY_ROOT / "checkpoints" / "tinystories-8m" / "tokenizer",
            progress=reporter,
            evaluation_progress=evaluation_reporter,
        )
    finally:
        reporter.close()
        evaluation_reporter.close()
    print(f"training publication: {published.directory}")
    print(f"selected epoch: {published.selected_epoch}")
    print(f"training SHA-256: {published.training_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

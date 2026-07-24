#!/usr/bin/env python3
"""Calibrate, gate, select, test, and publish the fixed semantic-v1 base."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
import os
from pathlib import Path
import tempfile
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tqdm import tqdm as Tqdm

    from apm.data.text.tinyworlds_p_semantic import (
        SemanticPartitionArtifact,
        SemanticSampleReport,
        StreamingTrainingConfig,
        TrainingCursor,
    )


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SEMANTIC_DATA_ROOT = REPOSITORY_ROOT / "data" / "tinyworlds-p-semantic"
PARTITION_ROOT = SEMANTIC_DATA_ROOT / "v1"
SAMPLE_REPORT_ROOT = SEMANTIC_DATA_ROOT / "sample-reports" / "v1"
CHECKPOINT_ROOT = REPOSITORY_ROOT / "checkpoints" / "tinyworlds-p-semantic-v1"
TOKENIZER_DIRECTORY = REPOSITORY_ROOT / "checkpoints" / "tinystories-8m" / "tokenizer"

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


@dataclass(frozen=True, slots=True)
class _RuntimePlan:
    active_tokens_per_epoch: int
    microbatches_per_epoch: int
    updates_per_epoch: int
    seconds_per_update: float
    validation_batches_per_epoch: int
    test_batches: int
    epochs: int

    @property
    def total_updates(self) -> int:
        return self.updates_per_epoch * self.epochs

    def estimate_seconds(self, epochs: int, validation_epochs: int, test: bool) -> float:
        training = self.seconds_per_update * self.updates_per_epoch * epochs
        evaluation_batches = self.validation_batches_per_epoch * validation_epochs
        evaluation_batches += self.test_batches if test else 0
        return training + evaluation_batches * self.seconds_per_update / 8.0


class _TrainingReporter:
    """Render measured optimizer phase and pass-path ETAs."""

    def __init__(self, description: str, phase_end: int, overall_end: int) -> None:
        self._description = description
        self._phase_end = phase_end
        self._overall_end = overall_end
        self._progress: Tqdm | None = None
        self._initial_update = 0
        self._started = 0.0

    def __call__(self, cursor: TrainingCursor, nll: float, planned_updates: int) -> None:
        """Advance the optimizer bar after one synchronized update."""
        from tqdm import tqdm

        if self._progress is None:
            self._initial_update = cursor.optimizer_update - 1
            if not self._initial_update < self._phase_end <= self._overall_end <= planned_updates:
                raise ValueError("semantic training reporter bounds are inconsistent")
            self._started = time.monotonic()
            self._progress = tqdm(
                total=self._phase_end - self._initial_update,
                desc=self._description,
                unit="update",
            )
        completed = cursor.optimizer_update - self._initial_update
        self._progress.update(completed - self._progress.n)
        elapsed = time.monotonic() - self._started
        seconds_per_update = elapsed / max(1, completed)
        self._progress.set_postfix_str(
            f"NLL {nll:.4f}, phase ETA "
            f"{_duration(seconds_per_update * max(0, self._phase_end - cursor.optimizer_update))}, "
            f"pass-path ETA "
            f"{_duration(seconds_per_update * max(0, self._overall_end - cursor.optimizer_update))}",
            refresh=False,
        )

    def close(self) -> None:
        """Close the optimizer progress bar if training reached it."""
        if self._progress is not None:
            self._progress.close()


class _EvaluationReporter:
    """Render an ordered set of semantic evaluation splits with overall ETA."""

    def __init__(
        self,
        artifact: SemanticPartitionArtifact,
        description: str,
        selectors: Sequence[str],
    ) -> None:
        from apm.data.text.tinyworlds_p_semantic import count_partition_microbatches
        from tqdm import tqdm

        self._expected = tuple(
            (selector, count_partition_microbatches(artifact, selector))
            for selector in selectors
        )
        self._index = -1
        self._active_completed = 0
        self._started = time.monotonic()
        self._progress: Tqdm = tqdm(
            total=sum(total for _, total in self._expected),
            desc=description,
            unit="batch",
        )

    def __call__(self, split: str, completed: int, total: int) -> None:
        """Advance one split while enforcing the known evaluation sequence."""
        if self._index < 0 or split != self._expected[self._index][0] or completed <= self._active_completed:
            self._index += 1
            self._active_completed = 0
        if self._index >= len(self._expected) or self._expected[self._index] != (split, total):
            raise ValueError("semantic evaluation progress differs from its fixed sequence")
        self._progress.update(completed - self._active_completed)
        self._active_completed = completed
        elapsed = time.monotonic() - self._started
        seconds_per_batch = elapsed / max(1, self._progress.n)
        self._progress.set_postfix_str(
            f"{split}, phase ETA "
            f"{_duration(seconds_per_batch * max(0, total - completed))}, "
            f"overall ETA "
            f"{_duration(seconds_per_batch * max(0, self._progress.total - self._progress.n))}",
            refresh=False,
        )

    def close(self) -> None:
        """Close the evaluation progress bar."""
        self._progress.close()


def _fixed_partition() -> SemanticPartitionArtifact | None:
    from apm.data.text.tinyworlds_p_semantic import (
        CANONICAL_ARCHIVE_IDENTITY,
        CANONICAL_TOKENIZER_IDENTITY,
        ENCODER_SNAPSHOT_IDENTITY_SHA256,
        SEMANTIC_CONFIG,
        load_semantic_catalog_failure,
        load_partition,
    )

    candidates = tuple(
        load_partition(path)
        for path in sorted(PARTITION_ROOT.glob("[0-9a-f]" * 64))
        if (path / "tree.json").is_file()
    ) if PARTITION_ROOT.is_dir() else ()
    matches = tuple(
        item
        for item in candidates
        if item.archive_identity == CANONICAL_ARCHIVE_IDENTITY
        and item.tokenizer_identity == CANONICAL_TOKENIZER_IDENTITY
        and item.semantic_catalog.config == SEMANTIC_CONFIG
        and item.semantic_catalog.encoder_identity.identity_sha256
        == ENCODER_SNAPSHOT_IDENTITY_SHA256
    )
    if len(matches) != 1:
        failures_root = SEMANTIC_DATA_ROOT / "catalog" / "v1" / "failures"
        failures = tuple(
            load_semantic_catalog_failure(path)
            for path in sorted(failures_root.glob("[0-9a-f]" * 64))
            if (path / "tree.json").is_file()
        ) if failures_root.is_dir() else ()
        if len(failures) == 1 and not matches:
            print(
                f"[stop] semantic-v1 construction failed its frozen grid gate: "
                f"{failures[0].reason}; audit: {failures[0].root}",
                flush=True,
            )
            return None
        raise RuntimeError(
            "semantic training requires exactly one strict semantic-v1 partition; "
            "run scripts/build_tinyworlds_p_semantic_partition.py first"
        )
    return matches[0]


def _fixed_sample_report(artifact: SemanticPartitionArtifact) -> SemanticSampleReport:
    from apm.data.text.tinyworlds_p_semantic import load_sample_report

    root = SAMPLE_REPORT_ROOT / artifact.partition_sha256
    candidates = tuple(
        load_sample_report(path)
        for path in sorted(root.glob("[0-9a-f]" * 64))
        if (path / "tree.json").is_file()
    ) if root.is_dir() else ()
    matches = tuple(
        item
        for item in candidates
        if item.partition_sha256 == artifact.partition_sha256
        and item.catalog_sha256 == artifact.semantic_catalog.catalog_sha256
    )
    if len(matches) != 1:
        raise RuntimeError("semantic training requires exactly one complete bound sample report")
    return matches[0]


def _selectors(split: str) -> tuple[str, ...]:
    from apm.data.text.tinyworlds_p_semantic.contracts import WORLD_LABELS

    return (
        f"base/{split}",
        *(
            f"{role}/{world}/{split}"
            for world in WORLD_LABELS
            for role in ("world", "control")
        ),
    )


def _measure_and_plan(
    artifact: SemanticPartitionArtifact,
    config: StreamingTrainingConfig,
    working: Path,
) -> _RuntimePlan:
    from apm.data.text.tinyworlds_p_semantic import (
        allocator_peak_bytes,
        count_partition_microbatches,
        run_streaming_base_training,
    )

    base_microbatches = count_partition_microbatches(artifact, "base/train")
    updates_per_epoch = math.ceil(base_microbatches / config.accumulation_microbatches)
    callback_times: list[float] = []
    reporter = _TrainingReporter("semantic GPU preflight", 3, 3)
    try:
        run_streaming_base_training(
            artifact,
            working / "gpu-preflight",
            config,
            stop_after_update=3,
            progress=lambda cursor, nll, planned: (
                callback_times.append(time.monotonic()),
                reporter(cursor, nll, planned),
            )[-1],
        )
    finally:
        reporter.close()
    if len(callback_times) != 3:
        raise RuntimeError("semantic GPU preflight did not complete three updates")
    peak = allocator_peak_bytes()
    if peak > config.allocator_peak_limit_bytes:
        raise RuntimeError(
            f"semantic GPU preflight peak {peak:,} exceeds {config.allocator_peak_limit_bytes:,}"
        )
    seconds_per_update = callback_times[-1] - callback_times[-2]
    if not math.isfinite(seconds_per_update) or seconds_per_update <= 0.0:
        raise RuntimeError("semantic GPU preflight produced an invalid update rate")
    active_tokens = next(
        item.active_token_count
        for item in artifact.split_counts
        if item.role == "base" and item.world is None and item.split == "train"
    )
    validation_batches = sum(
        count_partition_microbatches(artifact, selector)
        for selector in _selectors("validation")
    )
    test_batches = sum(
        count_partition_microbatches(artifact, selector)
        for selector in _selectors("test")
    )
    plan = _RuntimePlan(
        active_tokens_per_epoch=active_tokens,
        microbatches_per_epoch=base_microbatches,
        updates_per_epoch=updates_per_epoch,
        seconds_per_update=seconds_per_update,
        validation_batches_per_epoch=validation_batches,
        test_batches=test_batches,
        epochs=config.epochs,
    )
    print(
        f"[runtime] retained base/train active tokens {active_tokens:,}/epoch; "
        f"microbatches {base_microbatches:,}/epoch; updates {updates_per_epoch:,}/epoch; "
        f"measured {seconds_per_update:.3f}s/update; allocator peak {peak / 2**30:.3f} GiB; "
        f"two-epoch calibration ETA { _duration(plan.estimate_seconds(2, 2, False)) }; "
        f"five-epoch pass-path ETA { _duration(plan.estimate_seconds(5, 5, True)) }",
        flush=True,
    )
    return plan


def _duration(seconds: float) -> str:
    rounded = max(0, round(seconds))
    hours, remainder = divmod(rounded, 3_600)
    minutes, remainder = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{remainder:02d}"


def main() -> int:
    """Run the fresh seed-zero semantic calibration and its fixed pass/stop policy."""
    from apm.data.text.tinyworlds_p_semantic import StreamingTrainingConfig
    from apm.data.text.tinyworlds_p_semantic.milestone import (
        finish_and_publish_base,
        run_calibration_attempt,
    )

    artifact = _fixed_partition()
    if artifact is None:
        print("GPU preflight/training/sealed test: not authorized", flush=True)
        return 2
    import jax

    devices = tuple(device for device in jax.devices() if device.platform == "gpu")
    if len(devices) != 1:
        raise RuntimeError(
            "semantic-v1 training requires the single CUDA GPU and must run outside the sandbox"
        )
    sample_report = _fixed_sample_report(artifact)
    config = StreamingTrainingConfig.from_preset()
    if (config.parameter_seed, config.calibration_epochs, config.epochs) != (0, 2, 5):
        raise RuntimeError("semantic-v1 seed/calibration/epoch policy changed")
    work_root = CHECKPOINT_ROOT / "work"
    work_root.mkdir(parents=True, exist_ok=True)
    working = Path(tempfile.mkdtemp(prefix="semantic-base-v1-", dir=work_root))
    print(f"temporary artifact directory: {working}", flush=True)
    print(
        f"[load] strict partition {artifact.partition_sha256} and sample report "
        f"{sample_report.report_sha256} authenticated; sealed test remains closed",
        flush=True,
    )
    plan = _measure_and_plan(artifact, config, working)
    training_reporter = _TrainingReporter(
        "semantic-v1 calibration",
        2 * plan.updates_per_epoch,
        plan.total_updates,
    )
    validation_reporter = _EvaluationReporter(
        artifact,
        "semantic-v1 calibration validation",
        _selectors("validation") * 2,
    )
    try:
        calibration = run_calibration_attempt(
            artifact,
            sample_report.root,
            working / "calibration",
            config,
            progress=training_reporter,
            evaluation_progress=validation_reporter,
        )
    finally:
        training_reporter.close()
        validation_reporter.close()
    if calibration.decision != "pass":
        print(
            f"[stop] fixed semantic grid decision: {calibration.decision}; "
            "epochs 3-5, selection, sealed test, and publication were not run",
            flush=True,
        )
        return 2
    training_reporter = _TrainingReporter(
        "semantic-v1 epochs 3-5",
        plan.total_updates,
        plan.total_updates,
    )
    evaluation_reporter = _EvaluationReporter(
        artifact,
        "semantic-v1 final validation and sealed test",
        _selectors("validation") * 3 + _selectors("test"),
    )
    try:
        published = finish_and_publish_base(
            calibration,
            CHECKPOINT_ROOT,
            TOKENIZER_DIRECTORY,
            progress=training_reporter,
            evaluation_progress=evaluation_reporter,
        )
    finally:
        training_reporter.close()
        evaluation_reporter.close()
    print(f"training publication: {published.directory}")
    print(f"selected epoch: {published.selected_epoch}")
    print(f"training SHA-256: {published.training_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

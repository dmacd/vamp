"""Two-epoch semantic calibration, gated continuation, selection, and publication."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

from apm.data.text.tinyworlds_p_semantic.contracts import (
    BENCHMARK_ID,
    SemanticPartitionArtifact,
    canonical_json_bytes,
    record_sha256,
)
from apm.data.text.tinyworlds_p_semantic.evaluation import (
    EvaluationProgress,
    SemanticSealedTest,
    evaluate_epoch_validation,
    evaluate_sealed_test_once,
    semantic_validation_record,
)
from apm.data.text.tinyworlds_p_semantic.sample_report import (
    SemanticSampleReport,
    load_sample_report,
)
from apm.data.text.tinyworlds_p_semantic.statistics import (
    CalibrationDecision,
    SemanticEpochValidation,
    calibration_decision,
    select_best_eligible_epoch,
)
from apm.data.text.tinyworlds_p_semantic.training import (
    StreamingTrainingConfig,
    StreamingTrainingResult,
    TrainingCursor,
    init_streaming_train_state,
    load_streaming_checkpoint,
    run_streaming_base_training,
)
from apm.lm.checkpoint import (
    CheckpointFileHash,
    SourceCheckpointMetadata,
    TokenizerCheckpointMetadata,
    load_gpt_neo_checkpoint,
    save_gpt_neo_checkpoint,
)
from apm.lm.parameters import GptNeoParams
from apm.lm.training import LmTrainState


TrainingProgress = Callable[[TrainingCursor, float, int], None]


class TrainingGateError(ValueError):
    """Semantic training, selection, sample, memory, or publication gate failed."""


@dataclass(frozen=True, slots=True)
class CalibrationAttempt:
    """Fresh two-epoch semantic calibration and its fixed stop/pass decision."""

    artifact: SemanticPartitionArtifact
    config: StreamingTrainingConfig
    sample_report: SemanticSampleReport
    training: StreamingTrainingResult
    validations: tuple[SemanticEpochValidation, SemanticEpochValidation]
    decision: CalibrationDecision
    working_directory: Path
    runtime_seconds: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "working_directory", Path(self.working_directory))
        if tuple(item.epoch for item in self.validations) != (1, 2):
            raise ValueError("semantic calibration must contain epochs one and two")
        if self.runtime_seconds < 0.0:
            raise ValueError("semantic calibration runtime cannot be negative")


@dataclass(frozen=True, slots=True)
class PublishedBase:
    """Selected semantic base, all validation evidence, and one sealed test."""

    directory: Path
    publication_sha256: str
    training_sha256: str
    selected_epoch: int
    validations: tuple[SemanticEpochValidation, ...]
    sealed_test: SemanticSealedTest

    def __post_init__(self) -> None:
        object.__setattr__(self, "directory", Path(self.directory))
        if not self.directory.is_dir():
            raise FileNotFoundError(self.directory)


def run_calibration_attempt(
    artifact: SemanticPartitionArtifact,
    sample_report_directory: str | Path,
    working_directory: str | Path,
    config: StreamingTrainingConfig,
    *,
    progress: TrainingProgress | None = None,
    evaluation_progress: EvaluationProgress | None = None,
    replicates: int = 10_000,
) -> CalibrationAttempt:
    """Train exactly two fresh epochs after requiring the full validation report."""
    started = time.monotonic()
    sample_report = load_sample_report(sample_report_directory)
    _require_sample_binding(artifact, sample_report)
    working = Path(working_directory)
    if working.exists() and any(working.iterdir()):
        raise FileExistsError("fresh semantic calibration working directory is not empty")
    training = run_streaming_base_training(
        artifact,
        working,
        config,
        stop_after_epoch=2,
        progress=progress,
    )
    template = init_streaming_train_state(config, training.planned_optimizer_updates)
    validations = tuple(
        evaluate_epoch_validation(
            _load_epoch_state(
                working / "states",
                epoch,
                training.training_sha256,
                template,
            )[0].trainable,
            artifact,
            epoch,
            working / "evaluations" / f"epoch-{epoch:02d}",
            config.model_config,
            replicates=replicates,
            progress=evaluation_progress,
        )
        for epoch in (1, 2)
    )
    decision = calibration_decision(
        validations[0],
        validations[1],
        config.allocator_peak_limit_bytes,
    )
    _write_json(
        working / "calibration.json",
        {
            "benchmark_id": BENCHMARK_ID,
            "decision": decision,
            "partition_sha256": artifact.partition_sha256,
            "sample_report_sha256": sample_report.report_sha256,
            "training_sha256": training.training_sha256,
            "validations": [semantic_validation_record(item) for item in validations],
        },
    )
    return CalibrationAttempt(
        artifact=artifact,
        config=config,
        sample_report=sample_report,
        training=training,
        validations=(validations[0], validations[1]),
        decision=decision,
        working_directory=working,
        runtime_seconds=time.monotonic() - started,
    )


def finish_and_publish_base(
    calibration: CalibrationAttempt,
    publication_root: str | Path,
    tokenizer_directory: str | Path,
    *,
    progress: TrainingProgress | None = None,
    evaluation_progress: EvaluationProgress | None = None,
    replicates: int = 10_000,
) -> PublishedBase:
    """Continue a passing grid through epoch five, select, open test once, and publish."""
    if calibration.decision != "pass":
        raise TrainingGateError(
            f"cannot continue semantic calibration decision {calibration.decision!r}"
        )
    _require_sample_binding(calibration.artifact, calibration.sample_report)
    working = calibration.working_directory
    epoch_two = _epoch_checkpoint(working / "states", 2)
    final_training = run_streaming_base_training(
        calibration.artifact,
        working,
        calibration.config,
        resume_from=epoch_two,
        progress=progress,
    )
    if final_training.cursor.epoch != calibration.config.epochs:
        raise TrainingGateError("semantic training did not reach the five-epoch boundary")
    template = init_streaming_train_state(
        calibration.config,
        final_training.planned_optimizer_updates,
    )
    later = tuple(
        evaluate_epoch_validation(
            _load_epoch_state(
                working / "states",
                epoch,
                final_training.training_sha256,
                template,
            )[0].trainable,
            calibration.artifact,
            epoch,
            working / "evaluations" / f"epoch-{epoch:02d}",
            calibration.config.model_config,
            replicates=replicates,
            progress=evaluation_progress,
        )
        for epoch in range(3, calibration.config.epochs + 1)
    )
    validations = (*calibration.validations, *later)
    selected = select_best_eligible_epoch(validations)
    if selected.held_in_nll > 2.0:
        raise TrainingGateError(
            "best semantic-gap checkpoint fails final held-in NLL: "
            f"epoch={selected.epoch}, nll={selected.held_in_nll:.6f}"
        )
    if max(item.allocator_peak_bytes for item in validations) > calibration.config.allocator_peak_limit_bytes:
        raise TrainingGateError("semantic validation exceeded the allocator peak limit")
    selected_checkpoint = _epoch_checkpoint(working / "states", selected.epoch)
    selected_state, _ = load_streaming_checkpoint(
        selected_checkpoint,
        final_training.training_sha256,
        template,
    )
    sealed = evaluate_sealed_test_once(
        selected_state.trainable,
        calibration.artifact,
        selected.epoch,
        working / "sealed-test",
        calibration.config.model_config,
        replicates=replicates,
        progress=evaluation_progress,
    )
    return _publish(
        calibration,
        final_training,
        validations,
        selected,
        selected_checkpoint,
        selected_state.trainable,
        sealed,
        Path(publication_root),
        Path(tokenizer_directory),
    )


def _publish(
    calibration: CalibrationAttempt,
    training: StreamingTrainingResult,
    validations: tuple[SemanticEpochValidation, ...],
    selected: SemanticEpochValidation,
    selected_checkpoint: Path,
    selected_params: GptNeoParams,
    sealed: SemanticSealedTest,
    publication_root: Path,
    tokenizer_directory: Path,
) -> PublishedBase:
    validation_trees = {
        f"epoch-{item.epoch:02d}": _file_sha256(
            calibration.working_directory
            / "evaluations"
            / f"epoch-{item.epoch:02d}"
            / "tree.json"
        )
        for item in validations
    }
    publication_root.mkdir(parents=True, exist_ok=True)
    work_root = publication_root / "work"
    work_root.mkdir(exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="publish-semantic-v1-", dir=work_root))
    tokenizer_target = staging / "tokenizer"
    tokenizer_target.mkdir()
    for expected in calibration.artifact.tokenizer_identity.files:
        source = tokenizer_directory / expected.name
        if (
            source.stat().st_size != expected.size_bytes
            or _file_sha256(source) != expected.sha256
        ):
            raise TrainingGateError(f"semantic tokenizer source changed: {expected.name}")
        shutil.copyfile(source, tokenizer_target / expected.name)
    checkpoint_tokenizer = TokenizerCheckpointMetadata(
        kind=calibration.artifact.tokenizer_identity.kind,
        identifier=calibration.artifact.tokenizer_identity.identifier,
        revision=calibration.artifact.tokenizer_identity.revision,
        files=tuple(
            CheckpointFileHash(item.name, item.sha256)
            for item in calibration.artifact.tokenizer_identity.files
        ),
    )
    base_reference = save_gpt_neo_checkpoint(
        staging / "base",
        selected_params,
        calibration.config.model_config,
        tokenizer=checkpoint_tokenizer,
        source=SourceCheckpointMetadata(
            identifier=BENCHMARK_ID,
            revision=calibration.artifact.partition_sha256,
            sha256=calibration.artifact.partition_sha256,
        ),
    )
    content = {
        "benchmark_id": BENCHMARK_ID,
        "base_checkpoint": {
            "manifest_sha256": base_reference.manifest_sha256,
            "parameter_checksum": base_reference.parameter_checksum,
        },
        "catalog_sha256": calibration.artifact.semantic_catalog.catalog_sha256,
        "partition_sha256": calibration.artifact.partition_sha256,
        "sample_report_sha256": calibration.sample_report.report_sha256,
        "sealed_test_tree_sha256": _file_sha256(sealed.directory / "tree.json"),
        "selected_epoch": selected.epoch,
        "training": calibration.config.as_record(),
        "training_sha256": training.training_sha256,
        "validation_tree_sha256": validation_trees,
    }
    publication_sha = record_sha256(content)
    target = publication_root / publication_sha
    if target.exists():
        raise FileExistsError(f"semantic base publication already exists: {target}")
    shutil.copytree(selected_checkpoint, staging / "selected-checkpoint")
    shutil.copytree(calibration.sample_report.root, staging / "sample-report")
    shutil.copytree(calibration.working_directory / "evaluations", staging / "evaluations")
    shutil.copytree(sealed.directory, staging / "sealed-test")
    _write_json(staging / "manifest.json", {**content, "publication_sha256": publication_sha})
    _write_text(staging / "report.md", _report_text(publication_sha, validations, sealed))
    _write_tree(staging, publication_sha)
    os.rename(staging, target)
    _fsync_directory(publication_root)
    loaded = load_gpt_neo_checkpoint(target / "base")
    if loaded.reference.parameter_checksum != base_reference.parameter_checksum:
        raise RuntimeError("published semantic base checkpoint changed on strict reload")
    return PublishedBase(
        directory=target.resolve(),
        publication_sha256=publication_sha,
        training_sha256=training.training_sha256,
        selected_epoch=selected.epoch,
        validations=validations,
        sealed_test=sealed,
    )


def _report_text(
    publication_sha: str,
    validations: Sequence[SemanticEpochValidation],
    sealed: SemanticSealedTest,
) -> str:
    lines = [
        "# TinyWorlds-P Semantic v1 Base Report",
        "",
        f"Publication SHA-256: `{publication_sha}`",
        "",
        "## Validation",
        "",
        "| Epoch | Held-in NLL | Mean gap | Mean 95% CI | Mean placebo p |",
        "|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {item.epoch} | {item.held_in_nll:.6f} | {item.mean_empirical.observed_gap:.6f} | [{item.mean_empirical.bootstrap_lower:.6f}, {item.mean_empirical.bootstrap_upper:.6f}] | {item.mean_empirical.placebo_probability:.6f} |"
        for item in validations
    )
    lines.extend(
        (
            "",
            f"## Sealed test — selected epoch {sealed.selected_epoch}",
            "",
            f"Held-in NLL: {sealed.held_in.nll:.6f}",
            "",
            "| World | Gap | 95% paired CI | Placebo p |",
            "|---|---:|---:|---:|",
        )
    )
    lines.extend(
        f"| {item.world} | {item.empirical.observed_gap:.6f} | [{item.empirical.bootstrap_lower:.6f}, {item.empirical.bootstrap_upper:.6f}] | {item.empirical.placebo_probability:.6f} |"
        for item in sealed.validation.worlds
    )
    mean = sealed.validation.mean_empirical
    lines.append(
        f"| mean | {mean.observed_gap:.6f} | [{mean.bootstrap_lower:.6f}, {mean.bootstrap_upper:.6f}] | {mean.placebo_probability:.6f} |"
    )
    return "\n".join(lines) + "\n"


def _require_sample_binding(
    artifact: SemanticPartitionArtifact,
    sample: SemanticSampleReport,
) -> None:
    if (
        sample.partition_sha256 != artifact.partition_sha256
        or sample.catalog_sha256 != artifact.semantic_catalog.catalog_sha256
    ):
        raise TrainingGateError("pre-training sample report does not bind this semantic partition")


def _load_epoch_state(
    states_directory: Path,
    epoch: int,
    training_sha256: str,
    template: LmTrainState[GptNeoParams],
) -> tuple[LmTrainState[GptNeoParams], TrainingCursor]:
    return load_streaming_checkpoint(
        _epoch_checkpoint(states_directory, epoch),
        training_sha256,
        template,
    )


def _epoch_checkpoint(states_directory: Path, epoch: int) -> Path:
    matches = tuple(states_directory.glob(f"epoch-{epoch:02d}-update-*"))
    if len(matches) != 1:
        raise TrainingGateError(
            f"expected one complete semantic epoch-{epoch} checkpoint, found {len(matches)}"
        )
    return matches[0]


def _write_tree(root: Path, publication_sha: str) -> None:
    files = tuple(
        {
            "relative_path": path.relative_to(root).as_posix(),
            "sha256": _file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())
        if path.is_file() and path != root / "tree.json"
    )
    _write_json(
        root / "tree.json",
        {
            "files": list(files),
            "format": "tinyworlds-p-semantic-base-tree",
            "publication_sha256": publication_sha,
            "schema_version": 1,
        },
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        output.write(canonical_json_bytes(value))
        output.flush()
        os.fsync(output.fileno())


def _write_text(path: Path, value: str) -> None:
    with path.open("wb") as output:
        output.write(value.encode("utf-8"))
        output.flush()
        os.fsync(output.fileno())


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "CalibrationAttempt",
    "PublishedBase",
    "TrainingGateError",
    "finish_and_publish_base",
    "run_calibration_attempt",
]

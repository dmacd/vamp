"""Validation-gated semantic-v6 base selection before sealed-test access."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import math

from apm.data.text.tinyworlds_p_semantic.contracts import (
    canonical_json_bytes,
    record_sha256,
)
from apm.data.text.tinyworlds_p_semantic.evaluation import semantic_validation_record
from apm.data.text.tinyworlds_p_semantic.statistics import (
    CANONICAL_REPLICATES,
    CalibrationDecision,
    SemanticEpochValidation,
    calibration_decision,
    select_best_eligible_epoch,
)
from apm.data.text.tinyworlds_p_semantic.v6_evaluation import (
    V6EvaluationProgress,
    evaluate_v6_epoch_validation,
    load_v6_epoch_validation,
    v6_validation_from_record,
)
from apm.data.text.tinyworlds_p_semantic.v6_batching import (
    count_v6_partition_microbatches,
)
from apm.data.text.tinyworlds_p_semantic.v6_partition_contracts import (
    V6_BENCHMARK_ID,
    V6SemanticPartitionArtifact,
    V6SemanticSampleReport,
)
from apm.data.text.tinyworlds_p_semantic.v6_sample_report import (
    load_v6_sample_report,
)
from apm.data.text.tinyworlds_p_semantic.v6_training import (
    TrainingCursor,
    V6StreamingTrainingConfig,
    V6StreamingTrainingResult,
    init_v6_streaming_train_state,
    load_latest_v6_streaming_result,
    load_v6_streaming_checkpoint,
    run_v6_streaming_base_training,
)
from apm.data.text.tinyworlds_p_semantic.v6_vamp_contracts import (
    V6_VAMP_EXPERIMENT_PRESET,
)
from apm.lm.checkpoint import (
    BaseCheckpointRef,
    CheckpointFileHash,
    SourceCheckpointMetadata,
    TokenizerCheckpointMetadata,
    load_gpt_neo_checkpoint,
    save_gpt_neo_checkpoint,
)
from apm.lm.config import GptNeoConfig
from apm.lm.parameters import GptNeoParams
from apm.lm.training import LmTrainState


V6_SELECTED_BASE_FORMAT = "tinyworlds-p-semantic-v6-selected-base"
V6_SELECTED_BASE_TREE_FORMAT = "tinyworlds-p-semantic-v6-selected-base-tree"
V6TrainingProgress = Callable[[TrainingCursor, float, int], None]


class V6TrainingGateError(ValueError):
    """A semantic-v6 source, training, validation, or publication gate failed."""


@dataclass(frozen=True, slots=True)
class V6CalibrationAttempt:
    """The fresh two-epoch semantic-v6 run and its frozen decision."""

    artifact: V6SemanticPartitionArtifact
    config: V6StreamingTrainingConfig
    sample_report: V6SemanticSampleReport
    training: V6StreamingTrainingResult
    validations: tuple[SemanticEpochValidation, SemanticEpochValidation]
    decision: CalibrationDecision
    working_directory: Path
    runtime_seconds: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "working_directory", Path(self.working_directory))
        if tuple(item.epoch for item in self.validations) != (1, 2):
            raise ValueError("semantic-v6 calibration must contain epochs one and two")
        if self.runtime_seconds < 0.0:
            raise ValueError("semantic-v6 calibration runtime cannot be negative")


@dataclass(frozen=True, slots=True)
class V6SelectedBase:
    """A validation-selected immutable base whose test split remains sealed."""

    directory: Path
    selection_sha256: str
    training_sha256: str
    partition_sha256: str
    catalog_sha256: str
    sample_report_sha256: str
    selected_epoch: int
    checkpoint: BaseCheckpointRef

    def __post_init__(self) -> None:
        object.__setattr__(self, "directory", Path(self.directory))
        if not self.directory.is_dir():
            raise FileNotFoundError(self.directory)
        for value, label in (
            (self.selection_sha256, "semantic-v6 selected base"),
            (self.training_sha256, "semantic-v6 selected training"),
            (self.partition_sha256, "semantic-v6 selected partition"),
            (self.catalog_sha256, "semantic-v6 selected catalog"),
            (self.sample_report_sha256, "semantic-v6 selected sample report"),
        ):
            _require_sha256(value, label)
        if not 2 <= self.selected_epoch <= 5:
            raise ValueError("semantic-v6 selected epoch must lie in 2-5")


def run_v6_calibration_attempt(
    artifact: V6SemanticPartitionArtifact,
    sample_report_directory: str | Path,
    working_directory: str | Path,
    config: V6StreamingTrainingConfig,
    *,
    progress: V6TrainingProgress | None = None,
    evaluation_progress: V6EvaluationProgress | None = None,
    replicates: int = CANONICAL_REPLICATES,
) -> V6CalibrationAttempt:
    """Train or strictly resume two epochs after authenticating the sample report."""
    if type(artifact) is not V6SemanticPartitionArtifact:
        raise TypeError("semantic-v6 calibration requires its strict partition")
    if type(config) is not V6StreamingTrainingConfig:
        raise TypeError("semantic-v6 calibration requires its strict training config")
    started = time.monotonic()
    sample_report = load_v6_sample_report(sample_report_directory)
    _require_sample_binding(artifact, sample_report)
    working = Path(working_directory)
    prior = (
        load_latest_v6_streaming_result(artifact, working, config)
        if working.exists() and any(working.iterdir())
        else None
    )
    if working.exists() and any(working.iterdir()) and prior is None:
        raise FileExistsError(
            "semantic-v6 calibration has work but no complete resume checkpoint"
        )
    if prior is not None and prior.cursor.epoch > 2:
        raise V6TrainingGateError("calibration resume advanced beyond epoch two")
    training = (
        prior
        if prior is not None and prior.cursor.epoch == 2
        else run_v6_streaming_base_training(
            artifact,
            working,
            config,
            resume_from=(
                None if prior is None else prior.checkpoints[-1].directory
            ),
            stop_after_epoch=2,
            progress=progress,
        )
    )
    template = init_v6_streaming_train_state(
        config,
        training.planned_optimizer_updates,
    )
    validations = tuple(
        _evaluate_or_load_epoch_validation(
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
    paired_validations = (validations[0], validations[1])
    decision = calibration_decision(
        *paired_validations,
        config.allocator_peak_limit_bytes,
    )
    _write_json(
        working / "calibration.json",
        {
            "benchmark_id": V6_BENCHMARK_ID,
            "decision": decision,
            "partition_sha256": artifact.partition_sha256,
            "sample_report_sha256": sample_report.report_sha256,
            "training_config": config.as_record(),
            "training_sha256": training.training_sha256,
            "validations": [
                semantic_validation_record(item) for item in paired_validations
            ],
        },
    )
    return V6CalibrationAttempt(
        artifact=artifact,
        config=config,
        sample_report=sample_report,
        training=training,
        validations=paired_validations,
        decision=decision,
        working_directory=working,
        runtime_seconds=time.monotonic() - started,
    )


def finish_v6_base_selection(
    calibration: V6CalibrationAttempt,
    selection_root: str | Path,
    tokenizer_directory: str | Path,
    *,
    progress: V6TrainingProgress | None = None,
    evaluation_progress: V6EvaluationProgress | None = None,
    replicates: int = CANONICAL_REPLICATES,
) -> V6SelectedBase:
    """Finish five epochs and publish the validation-selected base without test access."""
    if type(calibration) is not V6CalibrationAttempt:
        raise TypeError("semantic-v6 selection requires V6CalibrationAttempt")
    if calibration.decision != "pass":
        raise V6TrainingGateError(
            f"cannot continue semantic-v6 decision {calibration.decision!r}"
        )
    _require_sample_binding(calibration.artifact, calibration.sample_report)
    working = calibration.working_directory
    prior = load_latest_v6_streaming_result(
        calibration.artifact,
        working,
        calibration.config,
    )
    if prior is None or prior.cursor.epoch < 2:
        raise V6TrainingGateError("semantic-v6 continuation lacks epoch two")
    final_training = (
        prior
        if prior.cursor.epoch == calibration.config.epochs
        else run_v6_streaming_base_training(
            calibration.artifact,
            working,
            calibration.config,
            resume_from=prior.checkpoints[-1].directory,
            progress=progress,
        )
    )
    if final_training.cursor.epoch != calibration.config.epochs:
        raise V6TrainingGateError("semantic-v6 training did not reach epoch five")
    template = init_v6_streaming_train_state(
        calibration.config,
        final_training.planned_optimizer_updates,
    )
    later = tuple(
        _evaluate_or_load_epoch_validation(
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
        raise V6TrainingGateError("selected semantic-v6 base exceeds held-in NLL 2.0")
    if (
        max(item.allocator_peak_bytes for item in validations)
        > calibration.config.allocator_peak_limit_bytes
    ):
        raise V6TrainingGateError("semantic-v6 validation exceeded 12 GiB")
    selected_checkpoint = _epoch_checkpoint(
        working / "states",
        selected.epoch,
    )
    selected_state, _ = load_v6_streaming_checkpoint(
        selected_checkpoint,
        final_training.training_sha256,
        template,
    )
    return _publish_selected_base(
        calibration,
        final_training,
        validations,
        selected,
        selected_checkpoint,
        selected_state.trainable,
        Path(selection_root),
        Path(tokenizer_directory),
    )


def load_v6_calibration_attempt(
    artifact: V6SemanticPartitionArtifact,
    sample_report_directory: str | Path,
    working_directory: str | Path,
    config: V6StreamingTrainingConfig | None = None,
) -> V6CalibrationAttempt:
    """Reconstruct a completed two-epoch attempt for stage-boundary resume."""
    if type(artifact) is not V6SemanticPartitionArtifact:
        raise TypeError("semantic-v6 calibration resume requires its strict partition")
    effective = config or V6StreamingTrainingConfig.from_preset()
    if type(effective) is not V6StreamingTrainingConfig:
        raise TypeError("semantic-v6 calibration resume requires its strict config")
    working = Path(working_directory)
    record = _load_json(working / "calibration.json")
    if (
        record.get("benchmark_id") != V6_BENCHMARK_ID
        or record.get("partition_sha256") != artifact.partition_sha256
        or record.get("training_config") != effective.as_record()
    ):
        raise V6TrainingGateError("semantic-v6 calibration resume identity changed")
    decision = record.get("decision")
    if decision not in (
        "pass",
        "semantic_grid_failure",
        "training_quality_failure",
    ):
        raise V6TrainingGateError("semantic-v6 calibration decision changed")
    validations_record = record.get("validations")
    if (
        type(validations_record) is not list
        or len(validations_record) != 2
        or any(type(item) is not dict for item in validations_record)
    ):
        raise V6TrainingGateError("semantic-v6 calibration validations changed")
    persisted_validations = tuple(
        v6_validation_from_record(item) for item in validations_record
    )
    validations = tuple(
        load_v6_epoch_validation(
            working / "evaluations" / f"epoch-{epoch:02d}",
            artifact,
            epoch,
        )
        for epoch in (1, 2)
    )
    if (
        validations != persisted_validations
        or decision
        != calibration_decision(
            validations[0],
            validations[1],
            effective.allocator_peak_limit_bytes,
        )
    ):
        raise V6TrainingGateError("semantic-v6 calibration decision evidence changed")
    training_sha256 = _text(record, "training_sha256")
    planned_updates = (
        math.ceil(
            count_v6_partition_microbatches(artifact, "base/train")
            / effective.accumulation_microbatches
        )
        * effective.epochs
    )
    template = init_v6_streaming_train_state(effective, planned_updates)
    state, cursor = load_v6_streaming_checkpoint(
        _epoch_checkpoint(working / "states", 2),
        training_sha256,
        template,
    )
    sample = load_v6_sample_report(sample_report_directory)
    _require_sample_binding(artifact, sample)
    if sample.report_sha256 != _text(record, "sample_report_sha256"):
        raise V6TrainingGateError("semantic-v6 calibration sample binding changed")
    training = V6StreamingTrainingResult(
        state=state,
        cursor=cursor,
        checkpoints=(),
        trace_path=working / "progress.jsonl",
        training_sha256=training_sha256,
        planned_optimizer_updates=planned_updates,
    )
    return V6CalibrationAttempt(
        artifact=artifact,
        config=effective,
        sample_report=sample,
        training=training,
        validations=(validations[0], validations[1]),
        decision=decision,
        working_directory=working,
        runtime_seconds=0.0,
    )


def load_v6_selected_base(directory: str | Path) -> V6SelectedBase:
    """Authenticate a validation-selected v6 base and every bound file."""
    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise V6TrainingGateError("semantic-v6 selected base must be a directory")
    tree = _load_json(root / "tree.json")
    if (
        set(tree) != {"files", "format", "schema_version", "selection_sha256"}
        or tree.get("format") != V6_SELECTED_BASE_TREE_FORMAT
        or tree.get("schema_version") != 1
        or tree.get("selection_sha256") != root.name
    ):
        raise V6TrainingGateError("semantic-v6 selected-base tree changed")
    descriptors = tree.get("files")
    if type(descriptors) is not list or any(type(item) is not dict for item in descriptors):
        raise V6TrainingGateError("semantic-v6 selected-base descriptors changed")
    expected_paths = tuple(
        path.relative_to(root).as_posix()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != root / "tree.json"
    )
    described_paths = tuple(_text(item, "relative_path") for item in descriptors)
    if described_paths != expected_paths:
        raise V6TrainingGateError("semantic-v6 selected-base file set changed")
    for descriptor in descriptors:
        path = root / _text(descriptor, "relative_path")
        size = descriptor.get("size_bytes")
        if (
            type(size) is not int
            or size < 0
            or path.is_symlink()
            or path.stat().st_size != size
            or _file_sha256(path) != _text(descriptor, "sha256")
        ):
            raise V6TrainingGateError(f"semantic-v6 selected-base file changed: {path}")
    manifest = _load_json(root / "manifest.json")
    required = {
        "base_checkpoint",
        "benchmark_id",
        "catalog_sha256",
        "format",
        "partition_sha256",
        "sample_report_sha256",
        "selected_epoch",
        "selection_sha256",
        "training_config",
        "training_sha256",
        "validation_tree_sha256",
    }
    if (
        set(manifest) != required
        or manifest.get("format") != V6_SELECTED_BASE_FORMAT
        or manifest.get("benchmark_id") != V6_BENCHMARK_ID
        or manifest.get("selection_sha256") != root.name
    ):
        raise V6TrainingGateError("semantic-v6 selected-base manifest changed")
    content = {key: value for key, value in manifest.items() if key != "selection_sha256"}
    if record_sha256(content) != root.name:
        raise V6TrainingGateError("semantic-v6 selected-base identity changed")
    checkpoint = load_gpt_neo_checkpoint(root / "base")
    checkpoint_record = _mapping(manifest, "base_checkpoint")
    if checkpoint_record != {
        "manifest_sha256": checkpoint.reference.manifest_sha256,
        "parameter_checksum": checkpoint.reference.parameter_checksum,
    } or (
        manifest.get("partition_sha256")
        != V6_VAMP_EXPERIMENT_PRESET.partition_sha256
        or manifest.get("catalog_sha256")
        != V6_VAMP_EXPERIMENT_PRESET.catalog_sha256
        or manifest.get("sample_report_sha256")
        != V6_VAMP_EXPERIMENT_PRESET.sample_report_sha256
        or manifest.get("training_config")
        != V6StreamingTrainingConfig.from_preset().as_record()
        or checkpoint.config != V6StreamingTrainingConfig.from_preset().model_config
        or checkpoint.source.identifier != V6_BENCHMARK_ID
        or checkpoint.source.revision != manifest.get("partition_sha256")
        or checkpoint.source.sha256 != manifest.get("partition_sha256")
    ):
        raise V6TrainingGateError("semantic-v6 selected checkpoint binding changed")
    sample_report = load_v6_sample_report(root / "sample-report")
    if (
        sample_report.report_sha256 != manifest.get("sample_report_sha256")
        or sample_report.partition_sha256 != manifest.get("partition_sha256")
        or sample_report.catalog_sha256 != manifest.get("catalog_sha256")
    ):
        raise V6TrainingGateError("semantic-v6 selected sample report changed")
    validation_trees = _mapping(manifest, "validation_tree_sha256")
    expected_validation_names = tuple(
        f"epoch-{epoch:02d}" for epoch in range(1, 6)
    )
    if tuple(sorted(validation_trees)) != expected_validation_names or any(
        _file_sha256(root / "evaluations" / name / "tree.json")
        != validation_trees[name]
        for name in expected_validation_names
    ):
        raise V6TrainingGateError("semantic-v6 selected validation evidence changed")
    return V6SelectedBase(
        directory=root.resolve(),
        selection_sha256=root.name,
        training_sha256=_text(manifest, "training_sha256"),
        partition_sha256=_text(manifest, "partition_sha256"),
        catalog_sha256=_text(manifest, "catalog_sha256"),
        sample_report_sha256=_text(manifest, "sample_report_sha256"),
        selected_epoch=_integer(manifest, "selected_epoch"),
        checkpoint=checkpoint.reference,
    )


def _publish_selected_base(
    calibration: V6CalibrationAttempt,
    training: V6StreamingTrainingResult,
    validations: tuple[SemanticEpochValidation, ...],
    selected: SemanticEpochValidation,
    selected_checkpoint: Path,
    selected_params: GptNeoParams,
    selection_root: Path,
    tokenizer_directory: Path,
) -> V6SelectedBase:
    selection_root.mkdir(parents=True, exist_ok=True)
    work_root = selection_root / "work"
    work_root.mkdir(exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="select-semantic-v6-", dir=work_root))
    tokenizer_target = staging / "tokenizer"
    tokenizer_target.mkdir()
    for expected in calibration.artifact.tokenizer_identity.files:
        source = tokenizer_directory / expected.name
        if (
            source.stat().st_size != expected.size_bytes
            or _file_sha256(source) != expected.sha256
        ):
            raise V6TrainingGateError(
                f"semantic-v6 tokenizer source changed: {expected.name}"
            )
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
            identifier=V6_BENCHMARK_ID,
            revision=calibration.artifact.partition_sha256,
            sha256=calibration.artifact.partition_sha256,
        ),
    )
    validation_trees = {
        f"epoch-{item.epoch:02d}": _file_sha256(
            calibration.working_directory
            / "evaluations"
            / f"epoch-{item.epoch:02d}"
            / "tree.json"
        )
        for item in validations
    }
    content = {
        "base_checkpoint": {
            "manifest_sha256": base_reference.manifest_sha256,
            "parameter_checksum": base_reference.parameter_checksum,
        },
        "benchmark_id": V6_BENCHMARK_ID,
        "catalog_sha256": calibration.artifact.semantic_catalog.catalog_sha256,
        "format": V6_SELECTED_BASE_FORMAT,
        "partition_sha256": calibration.artifact.partition_sha256,
        "sample_report_sha256": calibration.sample_report.report_sha256,
        "selected_epoch": selected.epoch,
        "training_config": calibration.config.as_record(),
        "training_sha256": training.training_sha256,
        "validation_tree_sha256": validation_trees,
    }
    selection_sha256 = record_sha256(content)
    target = selection_root / selection_sha256
    if target.exists():
        raise FileExistsError(f"semantic-v6 selected base already exists: {target}")
    shutil.copytree(selected_checkpoint, staging / "selected-resume-state")
    shutil.copytree(calibration.sample_report.root, staging / "sample-report")
    shutil.copytree(
        calibration.working_directory / "evaluations",
        staging / "evaluations",
    )
    _write_json(
        staging / "manifest.json",
        {**content, "selection_sha256": selection_sha256},
    )
    _write_text(
        staging / "selection-report.md",
        _selection_report(selection_sha256, validations, selected.epoch),
    )
    _write_tree(staging, selection_sha256)
    os.rename(staging, target)
    _fsync_directory(selection_root)
    return load_v6_selected_base(target)


def _selection_report(
    selection_sha256: str,
    validations: Sequence[SemanticEpochValidation],
    selected_epoch: int,
) -> str:
    rows = "\n".join(
        f"| {item.epoch} | {item.held_in_nll:.6f} | "
        f"{item.mean_empirical.observed_gap:.6f} | "
        f"[{item.mean_empirical.bootstrap_lower:.6f}, "
        f"{item.mean_empirical.bootstrap_upper:.6f}] | "
        f"{item.mean_empirical.placebo_probability:.6f} |"
        for item in validations
    )
    return (
        "# TinyWorlds-P Semantic-v6 Selected Base\n\n"
        f"Selection SHA-256: `{selection_sha256}`\n\n"
        f"Selected validation epoch: {selected_epoch}.\n\n"
        "The sealed test remains unopened.\n\n"
        "| Epoch | Held-in NLL | Mean gap | Mean 95% interval | Placebo p |\n"
        "|---:|---:|---:|---:|---:|\n"
        f"{rows}\n"
    )


def _require_sample_binding(
    artifact: V6SemanticPartitionArtifact,
    sample: V6SemanticSampleReport,
) -> None:
    if (
        sample.partition_sha256 != artifact.partition_sha256
        or sample.catalog_sha256 != artifact.semantic_catalog.catalog_sha256
    ):
        raise V6TrainingGateError(
            "semantic-v6 sample report does not bind this partition"
        )


def _load_epoch_state(
    states_directory: Path,
    epoch: int,
    training_sha256: str,
    template: LmTrainState[GptNeoParams],
) -> tuple[LmTrainState[GptNeoParams], TrainingCursor]:
    return load_v6_streaming_checkpoint(
        _epoch_checkpoint(states_directory, epoch),
        training_sha256,
        template,
    )


def _evaluate_or_load_epoch_validation(
    params: GptNeoParams,
    artifact: V6SemanticPartitionArtifact,
    epoch: int,
    directory: Path,
    model_config: GptNeoConfig,
    *,
    replicates: int,
    progress: V6EvaluationProgress | None,
) -> SemanticEpochValidation:
    if (directory / "tree.json").is_file():
        return load_v6_epoch_validation(directory, artifact, epoch)
    if directory.exists():
        recovery_root = directory.parents[1] / "recovery"
        recovery_root.mkdir(exist_ok=True)
        index = 1
        while (
            target := recovery_root / f"{directory.name}-incomplete-{index:02d}"
        ).exists():
            index += 1
        os.rename(directory, target)
        _fsync_directory(directory.parent)
        _fsync_directory(recovery_root)
    return evaluate_v6_epoch_validation(
        params,
        artifact,
        epoch,
        directory,
        model_config,
        replicates=replicates,
        progress=progress,
    )


def _epoch_checkpoint(states_directory: Path, epoch: int) -> Path:
    matches = tuple(states_directory.glob(f"epoch-{epoch:02d}-update-*"))
    if len(matches) != 1:
        raise V6TrainingGateError(
            f"expected one complete semantic-v6 epoch-{epoch} checkpoint, "
            f"found {len(matches)}"
        )
    return matches[0]


def _write_tree(root: Path, selection_sha256: str) -> None:
    files = tuple(
        {
            "relative_path": path.relative_to(root).as_posix(),
            "sha256": _file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != root / "tree.json"
    )
    _write_json(
        root / "tree.json",
        {
            "files": list(files),
            "format": V6_SELECTED_BASE_TREE_FORMAT,
            "schema_version": 1,
            "selection_sha256": selection_sha256,
        },
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(canonical_json_bytes(value))
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _write_text(path: Path, value: str) -> None:
    with path.open("wb") as output:
        output.write(value.encode("utf-8"))
        output.flush()
        os.fsync(output.fileno())


def _load_json(path: Path) -> dict[str, object]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise V6TrainingGateError(f"invalid semantic-v6 selected-base JSON: {path}") from error
    if type(value) is not dict or canonical_json_bytes(value) != raw:
        raise V6TrainingGateError(f"noncanonical semantic-v6 selected-base JSON: {path}")
    return value


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


def _mapping(record: Mapping[str, object], field: str) -> dict[str, object]:
    value = record.get(field)
    if type(value) is not dict:
        raise V6TrainingGateError(f"semantic-v6 field {field!r} must be an object")
    return value


def _text(record: Mapping[str, object], field: str) -> str:
    value = record.get(field)
    if type(value) is not str or not value:
        raise V6TrainingGateError(f"semantic-v6 field {field!r} must be text")
    return value


def _integer(record: Mapping[str, object], field: str) -> int:
    value = record.get(field)
    if type(value) is not int or value < 0:
        raise V6TrainingGateError(
            f"semantic-v6 field {field!r} must be a nonnegative integer"
        )
    return value


def _require_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


__all__ = [
    "V6CalibrationAttempt",
    "V6SelectedBase",
    "V6TrainingGateError",
    "V6_SELECTED_BASE_FORMAT",
    "V6_SELECTED_BASE_TREE_FORMAT",
    "finish_v6_base_selection",
    "load_v6_calibration_attempt",
    "load_v6_selected_base",
    "run_v6_calibration_attempt",
]

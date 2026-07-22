"""Calibration, final selection, sealed test, and base-model publication."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import time

import jax.numpy as jnp
import numpy as np

from apm.data.text.tinyworlds_p.contracts import (
    BENCHMARK_ID,
    PartitionArtifact,
    WORLD_LABELS,
    canonical_record_bytes,
)
from apm.data.text.tinyworlds_p.evaluation import (
    SealedTestResults,
    SplitNll,
    evaluate_epoch_validation,
    evaluate_sealed_test_once,
)
from apm.data.text.tinyworlds_p.training import (
    EpochValidation,
    GridDecision,
    StreamingTrainingConfig,
    StreamingTrainingResult,
    TrainingCursor,
    calibration_grid_decision,
    init_streaming_train_state,
    load_streaming_checkpoint,
    run_streaming_base_training,
    select_best_eligible_epoch,
)
from apm.lm.checkpoint import (
    CheckpointFileHash,
    SourceCheckpointMetadata,
    TokenizerCheckpointMetadata,
    load_gpt_neo_checkpoint,
    save_gpt_neo_checkpoint,
)
from apm.lm.generation import greedy_generate
from apm.lm.parameters import GptNeoParams
from apm.lm.text import TokenizersTextTokenizer
from apm.lm.training import LmTrainState


TrainingProgress = Callable[[TrainingCursor, float, int], None]
EvaluationProgress = Callable[[int, str, int, int], None]


class TrainingGateError(ValueError):
    """Calibration or final publication failed a predeclared training gate."""


@dataclass(frozen=True, slots=True)
class CalibrationAttempt:
    """One fresh two-epoch grid calibration with its complete states and decision."""

    artifact: PartitionArtifact
    config: StreamingTrainingConfig
    training: StreamingTrainingResult
    validations: tuple[EpochValidation, EpochValidation]
    decision: GridDecision
    working_directory: Path
    runtime_seconds: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "working_directory", Path(self.working_directory))
        if tuple(validation.epoch for validation in self.validations) != (1, 2):
            raise ValueError("calibration attempt requires epochs one and two")
        if self.runtime_seconds <= 0.0:
            raise ValueError("calibration runtime must be positive")


@dataclass(frozen=True, slots=True)
class PublishedBase:
    """The selected strict base checkpoint and complete publication tree."""

    directory: Path
    training_sha256: str
    selected_epoch: int
    validations: tuple[EpochValidation, ...]
    sealed_test: SealedTestResults

    def __post_init__(self) -> None:
        object.__setattr__(self, "directory", Path(self.directory))
        if not self.directory.is_dir():
            raise FileNotFoundError(self.directory)


def run_calibration_attempt(
    artifact: PartitionArtifact,
    working_directory: str | Path,
    config: StreamingTrainingConfig,
    *,
    progress: TrainingProgress | None = None,
    evaluation_progress: EvaluationProgress | None = None,
) -> CalibrationAttempt:
    """Train exactly two epochs from scratch, validate both, and decide the grid."""
    started = time.monotonic()
    working = Path(working_directory)
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
            config.model_config,
            progress=_split_evaluation_progress(evaluation_progress, epoch),
        )
        for epoch in (1, 2)
    )
    validation_path = working / "validation.jsonl"
    with validation_path.open("wb") as output:
        for validation in validations:
            output.write(canonical_record_bytes(_validation_record(validation)))
        output.flush()
        os.fsync(output.fileno())
    decision = calibration_grid_decision(
        validations[0],
        validations[1],
        config.allocator_peak_limit_bytes,
    )
    _write_json(
        working / "calibration.json",
        {
            "decision": decision,
            "partition_sha256": artifact.partition_sha256,
            "training_sha256": training.training_sha256,
            "validations": [_validation_record(item) for item in validations],
        },
    )
    return CalibrationAttempt(
        artifact=artifact,
        config=config,
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
) -> PublishedBase:
    """Resume a passing grid through epoch five, select, test once, and publish."""
    if calibration.decision != "pass":
        raise TrainingGateError(
            f"cannot finish nonpassing calibration decision: {calibration.decision}"
        )
    started = time.monotonic()
    working = calibration.working_directory
    epoch_two_checkpoint = _epoch_checkpoint(working / "states", 2)
    final_training = run_streaming_base_training(
        calibration.artifact,
        working,
        calibration.config,
        resume_from=epoch_two_checkpoint,
        progress=progress,
    )
    template = init_streaming_train_state(
        calibration.config,
        final_training.planned_optimizer_updates,
    )
    later_validations = tuple(
        evaluate_epoch_validation(
            _load_epoch_state(
                working / "states",
                epoch,
                final_training.training_sha256,
                template,
            )[0].trainable,
            calibration.artifact,
            epoch,
            calibration.config.model_config,
            progress=_split_evaluation_progress(evaluation_progress, epoch),
        )
        for epoch in (3, 4, 5)
    )
    validations = calibration.validations + later_validations
    with (working / "validation.jsonl").open("ab") as output:
        for validation in later_validations:
            output.write(canonical_record_bytes(_validation_record(validation)))
        output.flush()
        os.fsync(output.fileno())
    selected = select_best_eligible_epoch(validations)
    if selected.held_in_nll > 2.0:
        raise TrainingGateError(
            "best gap-eligible checkpoint fails final held-in validation NLL: "
            f"epoch={selected.epoch}, nll={selected.held_in_nll:.6f}"
        )
    selected_state, _ = _load_epoch_state(
        working / "states",
        selected.epoch,
        final_training.training_sha256,
        template,
    )
    sealed_path = working / "sealed-test.json"
    if sealed_path.exists():
        raise TrainingGateError("sealed test was already opened for this training run")
    sealed_test = evaluate_sealed_test_once(
        selected_state.trainable,
        calibration.artifact,
        selected.epoch,
        calibration.config.model_config,
        progress=_split_evaluation_progress(evaluation_progress, selected.epoch),
    )
    _write_json(sealed_path, _sealed_test_record(sealed_test))
    publication_directory = Path(publication_root) / final_training.training_sha256
    if publication_directory.exists():
        raise FileExistsError(f"training publication already exists: {publication_directory}")
    publication_directory.parent.mkdir(parents=True, exist_ok=True)
    staging = working / "publication"
    if staging.exists():
        raise FileExistsError(f"training publication staging exists: {staging}")
    staging.mkdir()
    tokenizer_source = Path(tokenizer_directory)
    tokenizer_target = staging / "tokenizer"
    tokenizer_target.mkdir()
    for expected in calibration.artifact.tokenizer_identity.files:
        source = tokenizer_source / expected.name
        if source.stat().st_size != expected.size_bytes or _file_sha256(source) != expected.sha256:
            raise TrainingGateError(f"tokenizer source changed: {expected.name}")
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
        selected_state.trainable,
        calibration.config.model_config,
        tokenizer=checkpoint_tokenizer,
        source=SourceCheckpointMetadata(
            identifier=BENCHMARK_ID,
            revision=calibration.artifact.partition_sha256,
            sha256=calibration.artifact.partition_sha256,
        ),
    )
    shutil.copytree(working / "states", staging / "states")
    for name in ("progress.jsonl", "validation.jsonl", "calibration.json", "sealed-test.json"):
        shutil.copyfile(working / name, staging / name)
    samples = _generation_samples(
        selected_state.trainable,
        calibration.config,
        tokenizer_target / "tokenizer.json",
        calibration.artifact.eos_token_id,
        calibration.artifact.pad_token_id,
    )
    _write_json(staging / "samples.json", {"samples": samples})
    total_runtime = calibration.runtime_seconds + (time.monotonic() - started)
    active_training_tokens = _trace_active_tokens(staging / "progress.jsonl")
    _write_json(
        staging / "training.json",
        {
            "base_checkpoint": {
                "manifest_sha256": base_reference.manifest_sha256,
                "parameter_checksum": base_reference.parameter_checksum,
            },
            "partition_sha256": calibration.artifact.partition_sha256,
            "selected_epoch": selected.epoch,
            "training": calibration.config.as_record(),
            "training_sha256": final_training.training_sha256,
        },
    )
    (staging / "report.md").write_text(
        _report_text(
            calibration.artifact,
            final_training.training_sha256,
            validations,
            selected,
            sealed_test,
            total_runtime,
            active_training_tokens,
        ),
        encoding="utf-8",
    )
    _write_training_tree(staging, final_training.training_sha256)
    os.rename(staging, publication_directory)
    _fsync_directory(publication_directory.parent)
    loaded = load_gpt_neo_checkpoint(publication_directory / "base")
    if loaded.reference.parameter_checksum != base_reference.parameter_checksum:
        raise RuntimeError("published base checkpoint changed on strict reload")
    return PublishedBase(
        directory=publication_directory,
        training_sha256=final_training.training_sha256,
        selected_epoch=selected.epoch,
        validations=validations,
        sealed_test=sealed_test,
    )


def _split_evaluation_progress(
    progress: EvaluationProgress | None,
    epoch: int,
) -> Callable[[str, int, int], None] | None:
    """Bind an epoch to split-level evaluator progress."""
    return (
        None
        if progress is None
        else lambda split, completed, total: progress(epoch, split, completed, total)
    )


def _load_epoch_state(
    states_directory: Path,
    epoch: int,
    training_sha256: str,
    template: LmTrainState[GptNeoParams],
) -> tuple[LmTrainState[GptNeoParams], TrainingCursor]:
    checkpoint = _epoch_checkpoint(states_directory, epoch)
    return load_streaming_checkpoint(checkpoint, training_sha256, template)


def _epoch_checkpoint(states_directory: Path, epoch: int) -> Path:
    matches = tuple(states_directory.glob(f"epoch-{epoch:02d}-update-*"))
    if len(matches) != 1:
        raise TrainingGateError(
            f"expected one complete epoch-{epoch} checkpoint, found {len(matches)}"
        )
    return matches[0]


def _validation_record(validation: EpochValidation) -> dict[str, object]:
    return {
        "allocator_peak_bytes": validation.allocator_peak_bytes,
        "epoch": validation.epoch,
        "held_in_nll": validation.held_in_nll,
        "mean_gap": validation.mean_gap,
        "worlds": [
            {
                "control_nll": item.control_nll,
                "gap": item.gap,
                "world": item.world,
                "world_nll": item.world_nll,
            }
            for item in validation.world_gaps
        ],
    }


def _sealed_test_record(results: SealedTestResults) -> dict[str, object]:
    return {
        "controls": [_split_nll_record(item) for item in results.controls],
        "held_in": _split_nll_record(results.held_in),
        "selected_epoch": results.selected_epoch,
        "worlds": [_split_nll_record(item) for item in results.worlds],
    }


def _split_nll_record(result: SplitNll) -> dict[str, object]:
    return {
        "active_tokens": result.active_tokens,
        "nll": result.nll,
        "split": result.split,
    }


def _generation_samples(
    params: GptNeoParams,
    config: StreamingTrainingConfig,
    tokenizer_path: Path,
    eos_token_id: int,
    pad_token_id: int,
) -> list[dict[str, str]]:
    tokenizer = TokenizersTextTokenizer.from_file(tokenizer_path)
    prompts = (
        "Once upon a time",
        "One day, a little girl",
        "There was a small dog",
        "Lily looked at the sky",
        "A kind boy found",
    )
    encoded = tuple(tokenizer.encode(prompt) for prompt in prompts)
    width = max(len(tokens) for tokens in encoded)
    prompt_ids = np.full((len(prompts), width), pad_token_id, dtype=np.int32)
    attention_mask = np.zeros((len(prompts), width), dtype=np.bool_)
    for row, tokens in enumerate(encoded):
        prompt_ids[row, : len(tokens)] = tokens
        attention_mask[row, : len(tokens)] = True
    generated = greedy_generate(
        params,
        config.model_config,
        jnp.asarray(prompt_ids),
        jnp.asarray(attention_mask),
        64,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )
    return [
        {
            "prompt": prompt,
            "text": tokenizer.decode(tuple(int(token) for token in np.asarray(row))),
        }
        for prompt, row in zip(prompts, generated, strict=True)
    ]


def _trace_active_tokens(path: Path) -> int:
    return sum(
        int(json.loads(line)["active_tokens"])
        for line in path.read_bytes().splitlines()
    )


def _report_text(
    artifact: PartitionArtifact,
    training_sha256: str,
    validations: Sequence[EpochValidation],
    selected: EpochValidation,
    sealed: SealedTestResults,
    runtime_seconds: float,
    active_tokens: int,
) -> str:
    validation_rows = "\n".join(
        f"| {item.epoch} | {item.held_in_nll:.6f} | {item.mean_gap:.6f} |"
        for item in validations
    )
    test_rows = "\n".join(
        f"| {world} | {world_result.nll:.6f} | {control_result.nll:.6f} | "
        f"{world_result.nll - control_result.nll:.6f} |"
        for world, world_result, control_result in zip(
            WORLD_LABELS,
            sealed.worlds,
            sealed.controls,
            strict=True,
        )
    )
    throughput = active_tokens / runtime_seconds
    role_coverage, eligible_fraction = _archive_ingest_coverage(artifact)
    return (
        "# TinyWorlds-P Archive v1 Base Training Report\n\n"
        f"- Partition: `{artifact.partition_sha256}`\n"
        f"- Training: `{training_sha256}`\n"
        f"- Archive role-classification coverage: {role_coverage:.6f}\n"
        f"- Eligible archive token fraction: {eligible_fraction:.6f}\n"
        f"- Selected epoch: {selected.epoch}\n"
        f"- Runtime: {runtime_seconds / 3600.0:.3f} hours\n"
        f"- Training throughput: {throughput:,.0f} active tokens/s\n"
        f"- Allocator peak: {max(item.allocator_peak_bytes for item in validations) / 1024**3:.3f} GiB\n\n"
        "## Validation learning curve\n\n"
        "| Epoch | Held-in NLL | Mean world-control gap |\n"
        "|---:|---:|---:|\n"
        f"{validation_rows}\n\n"
        "## Sealed test\n\n"
        f"Held-in test NLL: {sealed.held_in.nll:.6f}\n\n"
        "| World | World NLL | Control NLL | Gap |\n"
        "|:---:|---:|---:|---:|\n"
        f"{test_rows}\n"
    )


def _archive_ingest_coverage(artifact: PartitionArtifact) -> tuple[float, float]:
    audit = json.loads((artifact.root / "audit.json").read_bytes())
    ingest = audit["archive_ingest"]
    nonempty_tokens = int(ingest["nonempty_token_count"])
    return (
        float(ingest["role_classification_coverage"]),
        int(ingest["eligible_token_count"]) / nonempty_tokens,
    )


def _write_training_tree(directory: Path, training_sha256: str) -> None:
    files = [
        {
            "relative_path": path.relative_to(directory).as_posix(),
            "sha256": _file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(
            directory.rglob("*"),
            key=lambda item: item.relative_to(directory).as_posix(),
        )
        if path.is_file() and path.name != "tree.json"
    ]
    _write_json(
        directory / "tree.json",
        {
            "files": files,
            "format": "tinyworlds-p-archive-training-tree",
            "schema_version": 1,
            "training_sha256": training_sha256,
        },
    )


def _write_json(path: Path, value: object) -> None:
    payload = canonical_record_bytes(value)
    with path.open("wb") as output:
        output.write(payload)
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

"""Resumable orchestration for a registered TinyWorlds-Q scratch base."""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import time

from apm.data.text.tinyworlds_q_semantic.contracts import (
    QueryExperimentPreset,
    QueryPartitionArtifact,
    canonical_json_bytes,
    record_sha256,
)
from apm.data.text.tinyworlds_q_semantic.execution import BaseQualityDecision
from apm.data.text.tinyworlds_q_semantic.preflight import QueryGpuPreflight
from apm.data.text.tinyworlds_q_semantic.selected_base import (
    QueryBaseEpochEvidence,
    QuerySelectedBase,
    load_query_selected_base,
    publish_query_selected_base,
)
from apm.data.text.tinyworlds_q_semantic.training import (
    QueryBaseTrainingConfig,
    QueryBaseTrainingResult,
    QuerySplitNll,
    allocator_peak_bytes,
    evaluate_query_base_nll,
    init_query_base_train_state,
    load_query_training_checkpoint,
    query_base_training_identity,
    run_query_base_training,
)
from apm.lm.parameters import GptNeoParams


QUERY_BASE_EPOCH_EVIDENCE_FORMAT = (
    "tinyworlds-q-semantic-base-epoch-evidence-v1"
)
QUERY_BASE_DECISION_FORMAT = "tinyworlds-q-semantic-base-decision-v1"


def run_or_resume_query_selected_base(
    artifact: QueryPartitionArtifact,
    preset: QueryExperimentPreset,
    preflight: QueryGpuPreflight,
    working_root: str | Path,
    publication_root: str | Path,
) -> QuerySelectedBase:
    """Resume the exact two-epoch run and publish it only if every gate passes."""
    _validate_preflight(artifact, preset, preflight)
    selected = _matching_selected_base(artifact, preset, Path(publication_root))
    if selected is not None:
        print(f"Using strict selected base {selected.directory}.", flush=True)
        return selected

    config = QueryBaseTrainingConfig.from_preset(preset)
    config_sha256 = record_sha256(config.as_record())
    training_sha256, planned_updates = query_base_training_identity(
        artifact,
        preset,
        config,
    )
    working = Path(working_root) / f"base-{training_sha256}"
    working.mkdir(parents=True, exist_ok=True)
    initial_checkpoint = _latest_base_checkpoint(working, training_sha256)
    initial_update = (
        0
        if initial_checkpoint is None
        else _checkpoint_update(initial_checkpoint, training_sha256)
    )
    phase_started = time.monotonic()
    print(
        f"Phase: seed-zero base training ({planned_updates:,} optimizer updates; "
        f"resuming at {initial_update:,}).",
        flush=True,
    )

    def training_progress(cursor, nll: float, planned: int) -> None:
        update = cursor.optimizer_update
        if update == 1 or update % 100 == 0 or update == planned:
            elapsed = time.monotonic() - phase_started
            completed = max(1, update - initial_update)
            remaining = elapsed * (planned - update) / completed
            print(
                f"TinyWorlds-Q base update {update:,}/{planned:,}; "
                f"NLL {nll:.6f}; phase ETA {_duration(remaining)}",
                flush=True,
            )

    evidence: list[QueryBaseEpochEvidence] = []
    result: QueryBaseTrainingResult | None = None
    for epoch in range(1, config.epochs + 1):
        existing_evidence = _load_epoch_evidence(
            working,
            epoch,
            training_sha256,
            artifact.partition_sha256,
            config_sha256,
        )
        latest = _latest_base_checkpoint(working, training_sha256)
        completed_epoch = (
            0
            if latest is None
            else _checkpoint_epoch(latest, training_sha256)
        )
        if completed_epoch < epoch:
            result = run_query_base_training(
                artifact,
                preset,
                working,
                config,
                resume_from=latest,
                stop_after_epoch=epoch,
                progress=training_progress,
            )
        if existing_evidence is None:
            params = _epoch_parameters(
                result,
                working,
                epoch,
                training_sha256,
                config,
                planned_updates,
            )
            existing_evidence = _evaluate_epoch(
                epoch,
                params,
                artifact,
                preset,
                config,
                working,
                training_sha256,
                config_sha256,
            )
        evidence.append(existing_evidence)
        if result is not None and epoch < config.epochs:
            del result
            result = None
            gc.collect()

    latest = _latest_base_checkpoint(working, training_sha256)
    if latest is None:
        raise RuntimeError("complete base training has no resumable state")
    if result is None or result.cursor.epoch != config.epochs:
        result = run_query_base_training(
            artifact,
            preset,
            working,
            config,
            resume_from=latest,
            progress=training_progress,
        )
    peak = max(allocator_peak_bytes(), preflight.measurement.allocator_peak_bytes)
    ordered_evidence = tuple(evidence)
    if len(ordered_evidence) != 2:
        raise RuntimeError("registered base requires exactly two epochs")
    decision = BaseQualityDecision(
        tuple(item.validation.nll for item in ordered_evidence),  # type: ignore[arg-type]
        peak,
        preset.allocator_peak_limit_bytes,
    )
    _publish_base_decision(
        working,
        ordered_evidence,  # type: ignore[arg-type]
        peak,
        decision,
        preflight,
        artifact.partition_sha256,
        training_sha256,
    )
    if not decision.passed:
        raise RuntimeError(f"query base quality gate failed: {decision.reason}")
    selected = publish_query_selected_base(
        result,
        ordered_evidence,  # type: ignore[arg-type]
        peak,
        artifact,
        preset,
        publication_root,
    )
    print(f"Selected base: {selected.selection_sha256}", flush=True)
    print(f"Selected-base report: {selected.directory / 'training-report.md'}", flush=True)
    print("The sealed test was not opened.", flush=True)
    return selected


def _validate_preflight(
    artifact: QueryPartitionArtifact,
    preset: QueryExperimentPreset,
    preflight: QueryGpuPreflight,
) -> None:
    if (
        type(preflight) is not QueryGpuPreflight
        or preflight.partition_sha256 != artifact.partition_sha256
        or preflight.catalog_sha256 != artifact.catalog_sha256
        or preflight.config_sha256 != preset.config_sha256
        or preflight.measurement.allocator_peak_bytes
        > preset.allocator_peak_limit_bytes
    ):
        raise ValueError("selected-base run requires its exact passing GPU preflight")


def _matching_selected_base(
    artifact: QueryPartitionArtifact,
    preset: QueryExperimentPreset,
    publication_root: Path,
) -> QuerySelectedBase | None:
    root = publication_root / "base"
    matches = (
        tuple(
            load_query_selected_base(path, artifact, preset)
            for path in sorted(root.iterdir())
            if path.is_dir()
            and len(path.name) == 64
            and _selected_base_matches(
                path,
                artifact.catalog_sha256,
                artifact.partition_sha256,
            )
        )
        if root.is_dir()
        else ()
    )
    if len(matches) > 1:
        raise RuntimeError("multiple selected bases bind the query partition")
    return matches[0] if matches else None


def _selected_base_matches(
    root: Path,
    catalog_sha256: str,
    partition_sha256: str,
) -> bool:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return False
    record = _canonical_json(manifest_path, "selected-base candidate")
    return (
        record.get("catalog_sha256") == catalog_sha256
        and record.get("partition_sha256") == partition_sha256
    )


def _latest_base_checkpoint(
    working: Path,
    training_sha256: str,
) -> Path | None:
    states = working / "states"
    candidates = (
        tuple(
            (_checkpoint_cursor(path, training_sha256), path)
            for path in sorted(states.iterdir())
            if path.is_dir() and (path / "resume.json").is_file()
        )
        if states.is_dir()
        else ()
    )
    return max(candidates, default=((), None))[1]


def _checkpoint_cursor(
    root: Path,
    training_sha256: str,
) -> tuple[int, int, int, int, int]:
    record = _canonical_json(root / "resume.json", "base checkpoint")
    cursor = record.get("cursor")
    if record.get("training_sha256") != training_sha256 or type(cursor) is not dict:
        raise ValueError(f"base checkpoint binding changed: {root}")
    values = tuple(
        cursor.get(field)
        for field in (
            "optimizer_update",
            "epoch",
            "block",
            "microbatch",
            "schedule_position",
        )
    )
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError(f"base checkpoint cursor changed: {root}")
    return values  # type: ignore[return-value]


def _checkpoint_update(root: Path, training_sha256: str) -> int:
    return _checkpoint_cursor(root, training_sha256)[0]


def _checkpoint_epoch(root: Path, training_sha256: str) -> int:
    return _checkpoint_cursor(root, training_sha256)[1]


def _epoch_checkpoint(working: Path, epoch: int) -> Path:
    matches = tuple(
        sorted((working / "states").glob(f"epoch-{epoch:02d}-update-*"))
    )
    if len(matches) != 1:
        raise RuntimeError(f"expected one complete checkpoint for epoch {epoch}")
    return matches[0]


def _epoch_parameters(
    result: QueryBaseTrainingResult | None,
    working: Path,
    epoch: int,
    training_sha256: str,
    config: QueryBaseTrainingConfig,
    planned_updates: int,
) -> GptNeoParams:
    if result is not None and result.cursor.epoch == epoch:
        return result.state.trainable
    template = init_query_base_train_state(config, planned_updates)
    state, cursor = load_query_training_checkpoint(
        _epoch_checkpoint(working, epoch),
        training_sha256,
        template,
    )
    if cursor.epoch != epoch:
        raise ValueError("epoch checkpoint cursor changed")
    return state.trainable


def _load_epoch_evidence(
    working: Path,
    epoch: int,
    training_sha256: str,
    partition_sha256: str,
    config_sha256: str,
) -> QueryBaseEpochEvidence | None:
    path = working / f"epoch-{epoch:02d}-validation.json"
    if not path.is_file():
        return None
    record = _canonical_json(path, f"base epoch-{epoch} evidence")
    required = {
        "active_tokens",
        "base_config_sha256",
        "epoch",
        "format",
        "nll",
        "partition_sha256",
        "split",
        "training_sha256",
    }
    if (
        set(record) != required
        or record.get("format") != QUERY_BASE_EPOCH_EVIDENCE_FORMAT
        or record.get("epoch") != epoch
        or record.get("split") != "validation"
        or record.get("training_sha256") != training_sha256
        or record.get("partition_sha256") != partition_sha256
        or record.get("base_config_sha256") != config_sha256
        or type(record.get("active_tokens")) is not int
        or type(record.get("nll")) not in (int, float)
    ):
        raise ValueError(f"base epoch-{epoch} evidence changed")
    return QueryBaseEpochEvidence(
        epoch,
        QuerySplitNll(
            "validation",
            int(record["active_tokens"]),
            float(record["nll"]),
        ),
    )


def _evaluate_epoch(
    epoch: int,
    params: GptNeoParams,
    artifact: QueryPartitionArtifact,
    preset: QueryExperimentPreset,
    config: QueryBaseTrainingConfig,
    working: Path,
    training_sha256: str,
    config_sha256: str,
) -> QueryBaseEpochEvidence:
    started = time.monotonic()

    def progress(_split: str, completed: int, total: int) -> None:
        if completed == 1 or completed % 100 == 0 or completed == total:
            elapsed = time.monotonic() - started
            remaining = elapsed * (total - completed) / max(1, completed)
            print(
                f"TinyWorlds-Q epoch {epoch} validation "
                f"{completed:,}/{total:,}; phase ETA {_duration(remaining)}",
                flush=True,
            )

    validation = evaluate_query_base_nll(
        params,
        artifact,
        preset,
        "validation",
        config,
        progress=progress,
    )
    evidence = QueryBaseEpochEvidence(epoch, validation)
    _publish_epoch_evidence(
        working,
        evidence,
        training_sha256,
        artifact.partition_sha256,
        config_sha256,
    )
    print(
        f"TinyWorlds-Q epoch {epoch} held-in NLL: {validation.nll:.9f}",
        flush=True,
    )
    return evidence


def _publish_epoch_evidence(
    working: Path,
    evidence: QueryBaseEpochEvidence,
    training_sha256: str,
    partition_sha256: str,
    config_sha256: str,
) -> Path:
    path = working / f"epoch-{evidence.epoch:02d}-validation.json"
    payload = canonical_json_bytes(
        {
            "active_tokens": evidence.validation.active_tokens,
            "base_config_sha256": config_sha256,
            "epoch": evidence.epoch,
            "format": QUERY_BASE_EPOCH_EVIDENCE_FORMAT,
            "nll": evidence.validation.nll,
            "partition_sha256": partition_sha256,
            "split": evidence.validation.split,
            "training_sha256": training_sha256,
        }
    )
    _write_once(path, payload, "epoch evidence")
    return path


def _publish_base_decision(
    working: Path,
    evidence: tuple[QueryBaseEpochEvidence, QueryBaseEpochEvidence],
    allocator_peak: int,
    decision: BaseQualityDecision,
    preflight: QueryGpuPreflight,
    partition_sha256: str,
    training_sha256: str,
) -> Path:
    path = working / "base-decision.json"
    payload = canonical_json_bytes(
        {
            "allocator_peak_bytes": allocator_peak,
            "epoch_nll": [item.validation.nll for item in evidence],
            "format": QUERY_BASE_DECISION_FORMAT,
            "partition_sha256": partition_sha256,
            "passed": decision.passed,
            "preflight_sha256": preflight.preflight_sha256,
            "reason": decision.reason,
            "training_sha256": training_sha256,
        }
    )
    _write_once(path, payload, "base quality decision")
    return path


def _write_once(path: Path, payload: bytes, label: str) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"different {label} already exists: {path}")
        return
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_bytes(payload)
    with temporary.open("rb") as source:
        os.fsync(source.fileno())
    os.replace(temporary, path)


def _canonical_json(path: Path, label: str) -> dict[str, object]:
    payload = path.read_bytes()
    record = json.loads(payload)
    if type(record) is not dict or canonical_json_bytes(record) != payload:
        raise ValueError(f"noncanonical {label}: {path}")
    return record


def _duration(seconds: float) -> str:
    rounded = max(0, round(seconds))
    hours, remainder = divmod(rounded, 3_600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


__all__ = [
    "QUERY_BASE_DECISION_FORMAT",
    "QUERY_BASE_EPOCH_EVIDENCE_FORMAT",
    "run_or_resume_query_selected_base",
]

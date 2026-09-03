"""Crash-safe checkpoints and immutable residual-integrator artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
import shutil

import torch
from torch import Tensor

from apm.continual.artifacts import file_sha256, load_canonical_json, publish_immutable_json
from apm.continual.vision.imagenetr.artifacts import (
    publish_artifact_directory,
    validate_artifact_directory,
)
from apm.continual.vision.imagenetr.checkpoints import atomic_torch_save
from apm.continual.vision.imagenetr.integrator_artifacts import IntegratorStore
from apm.continual.vision.imagenetr.integrator_model import (
    IntegratorFitResult,
    IntegratorState,
)


def save_integrator_checkpoint(
    path: str | Path,
    state: IntegratorState,
    stage: int,
    variant: str,
    frontier_hash: str,
    training_ids_hash: str,
) -> Path:
    """Atomically record a persistent model, optimizer, and exact data boundary."""
    return atomic_torch_save(
        path,
        {
            "frontier_hash": frontier_hash,
            "maximum_slots": state.model.maximum_slots,
            "model": state.model.state_dict(),
            "name": state.name,
            "optimizer": state.optimizer.state_dict(),
            "optimizer_steps": state.optimizer_steps,
            "schema_version": "imagenetr50-integrator-checkpoint-v1",
            "slot_dim": state.model.slot_dim,
            "stage": stage,
            "training_ids_hash": training_ids_hash,
            "variant": variant,
        },
    )


def restore_integrator_checkpoint(
    path: str | Path,
    state: IntegratorState,
    expected_stage: int,
    variant: str,
    frontier_hash: str,
    training_ids_hash: str,
) -> None:
    """Restore a trusted local checkpoint only at its authenticated boundary."""
    record = torch.load(Path(path), map_location="cpu", weights_only=False)
    if (
        type(record) is not dict
        or record.get("schema_version") != "imagenetr50-integrator-checkpoint-v1"
        or record.get("name") != state.name
        or record.get("maximum_slots") != state.model.maximum_slots
        or record.get("slot_dim") != state.model.slot_dim
        or record.get("stage") != expected_stage
        or record.get("variant") != variant
        or record.get("frontier_hash") != frontier_hash
        or record.get("training_ids_hash") != training_ids_hash
    ):
        raise ValueError("integrator checkpoint differs from the requested boundary")
    state.model.load_state_dict(record["model"], strict=True)
    state.optimizer.load_state_dict(record["optimizer"])
    state.optimizer_steps = int(record["optimizer_steps"])


def publish_integrator_fit(
    store: IntegratorStore,
    family: str,
    job_hash: str,
    state: IntegratorState,
    result: IntegratorFitResult,
    metadata: Mapping[str, object],
) -> Path:
    """Publish a selected model and its full fit evidence as one immutable directory."""
    from safetensors.torch import save_file

    target = store.run / "integrators" / family / job_hash
    if target.is_dir():
        validate_artifact_directory(target)
        return target
    work = store.run / "work" / f"integrator_{job_hash}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    model_path = work / "model.safetensors"
    tensors: dict[str, Tensor] = {
        name: value.detach().cpu().contiguous()
        for name, value in sorted(state.model.state_dict().items())
    }
    save_file(
        tensors,
        model_path,
        metadata={
            "schema_version": "imagenetr50-integrator-model-v1",
            "name": state.name,
        },
    )
    publish_immutable_json(
        work / "fit.json",
        {
            **dict(metadata),
            "fit": asdict(result),
            "model_sha256": file_sha256(model_path),
            "name": state.name,
            "schema_version": "imagenetr50-integrator-fit-v1",
        },
    )
    publish_artifact_directory(work, target)
    shutil.rmtree(work)
    return target


def load_integrator_fit(path: str | Path, state: IntegratorState) -> IntegratorFitResult:
    """Validate and load one immutable selected integrator into a matching model."""
    from safetensors.torch import load_file

    root = Path(path)
    validate_artifact_directory(root)
    record = load_canonical_json(root / "fit.json")
    model_path = root / "model.safetensors"
    if (
        record.get("schema_version") != "imagenetr50-integrator-fit-v1"
        or record.get("name") != state.name
        or record.get("model_sha256") != file_sha256(model_path)
    ):
        raise ValueError("integrator fit artifact changed")
    state.model.load_state_dict(load_file(model_path, device="cpu"), strict=True)
    values = dict(record["fit"])
    return IntegratorFitResult(**values)


__all__ = [
    "load_integrator_fit",
    "publish_integrator_fit",
    "restore_integrator_checkpoint",
    "save_integrator_checkpoint",
]

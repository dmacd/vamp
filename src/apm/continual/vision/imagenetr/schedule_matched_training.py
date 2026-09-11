"""Fixed replay-batch schedules applied to a fully available training population."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
import math
from pathlib import Path
import time

import pyarrow.parquet as pq
from pyrsistent import PVector, pvector
from safetensors.torch import save as tensor_bytes
import torch
from torch import nn
from tqdm import tqdm

from apm.continual.artifacts import atomic_write, canonical_json_bytes, file_sha256, publish_immutable_json, record_sha256
from apm.continual.vision.imagenetr.checkpoints import atomic_torch_save
from apm.continual.vision.imagenetr.joint_convergence_training import JointPopulation
from apm.continual.vision.imagenetr.srt_config import SRTConfig
from apm.continual.vision.imagenetr.srt_data import Presentation, SelectedImageLoader
from apm.continual.vision.imagenetr.srt_evidence import read_sealed, sealed_record, write_parquet
from apm.continual.vision.imagenetr.srt_scheduler import step_generator, uniform_indices
from apm.continual.vision.imagenetr.srt_training import evaluate_model, optimization_step, optimizer_for, restore_trainables, trainable_state


@dataclass(frozen=True, slots=True)
class ScheduledBatch:
    """One optimizer update and an informational source work-checkpoint index."""

    step: int
    block: int
    size: int


def require_schedule(schedule: tuple[ScheduledBatch, ...], maximum_batch: int) -> str:
    """Authenticate contiguous updates and work blocks, without inspecting labels."""
    if not schedule or schedule[0].block != 1:
        raise ValueError("schedule must start at work block one")
    for index, batch in enumerate(schedule):
        previous_block = schedule[index - 1].block if index else 1
        if (batch.step != index + 1 or batch.block not in (previous_block, previous_block + 1)
                or not 1 <= batch.size <= maximum_batch):
            raise ValueError("schedule contains invalid update, work block, or batch size")
    return record_sha256([asdict(batch) for batch in schedule])


def draw_presentations(population: JointPopulation, exposures: PVector, seed: int, batch: ScheduledBatch) -> tuple[tuple[Presentation, ...], PVector]:
    """Draw globally without replacement within a batch and advance persistent exposure counts."""
    indices = uniform_indices(len(population.fitting), batch.size, step_generator(seed, 0, batch.step, "offline_schedule_matched"))
    chosen = tuple(Presentation(population.fitting[index], exposures[index] + 1) for index in indices)
    for index in indices:
        exposures = exposures.set(index, exposures[index] + 1)
    return chosen, exposures


@dataclass(frozen=True, slots=True)
class WorkPosition:
    """Committed optimizer progress and partial work-block accounting."""

    steps: int = 0
    presentations: int = 0
    block_presentations: int = 0
    block_steps: int = 0
    block_loss: float = 0.0
    block_correct: int = 0
    block_wall_seconds: float = 0.0
    block_loader_seconds: float = 0.0
    block_checkpoint_seconds: float = 0.0


def validate_schedule_job(root: Path, expected_hash: str | None = None) -> dict[str, object]:
    """Authenticate committed work-block weights, update chunks, and final counts."""
    result = read_sealed(root / "result.json", "imagenetr50-schedule-matched-job-result-v1")
    job = read_sealed(root / "job.json", "imagenetr50-schedule-matched-job-v1")
    if result["job_hash"] != job["content_hash"] or (expected_hash is not None and job["content_hash"] != expected_hash):
        raise ValueError("schedule-matched job identity changed")
    for index, row in enumerate(result["blocks"], 1):
        folder = root / "blocks" / f"{index:03d}"
        if row["block"] != index or read_sealed(folder / "result.json") != row or file_sha256(folder / "model.safetensors") != row["model_sha256"]:
            raise ValueError("schedule-matched block evidence changed")
    next_step, presentations, observed_schedule = 1, 0, pvector()
    for chunk in result["chunks"]:
        path = root / chunk["path"]
        if not path.resolve().is_relative_to(root.resolve() / "updates") or file_sha256(path) != chunk["sha256"]:
            raise ValueError("schedule-matched update chunk changed")
        rows = pq.read_table(path).to_pylist()
        if not rows or len(rows) != chunk["steps"] or rows[0]["step"] != next_step:
            raise ValueError("schedule-matched update sequence changed")
        for row in rows:
            if row["step"] != next_step or row["batch_size"] < 1:
                raise ValueError("schedule-matched update sequence changed")
            if any(row[name] != job["optimizer"][name] for name in ("lora_learning_rate", "head_learning_rate")):
                raise ValueError("schedule-matched learning rates changed")
            observed_schedule = observed_schedule.append(ScheduledBatch(row["step"], row["block"], row["batch_size"]))
            next_step += 1
            presentations += row["batch_size"]
    if (next_step - 1 != result["optimizer_steps"] or presentations != result["training_presentations"]
            or result["optimizer_steps"] != job["planned_steps"] or presentations != job["planned_presentations"]
            or require_schedule(tuple(observed_schedule), job["maximum_batch_size"]) != job["schedule_hash"]
            or len(result["blocks"]) != observed_schedule[-1].block):
        raise ValueError("schedule-matched work counters disagree")
    if file_sha256(root / "exposures.parquet") != result["exposures_sha256"]:
        raise ValueError("schedule-matched exposure evidence changed")
    return result


def audit_schedule_draws(root: Path, population: JointPopulation, schedule: tuple[ScheduledBatch, ...]) -> dict[str, object]:
    """Reconstruct every sample draw and final exposure count without forwarding a model."""
    result = validate_schedule_job(root)
    job = read_sealed(root / "job.json")
    if (job["fitting_ids_hash"] != record_sha256([row.image_id for row in population.fitting])
            or job["schedule_hash"] != require_schedule(schedule, job["maximum_batch_size"])):
        raise ValueError("draw audit population or schedule changed")
    if (root / "draw_audit.json").is_file():
        audit = read_sealed(root / "draw_audit.json")
        if audit["result_hash"] != result["content_hash"]:
            raise ValueError("draw audit refers to a different training result")
        return audit
    exposures, verified = pvector([0] * len(population.fitting)), 0
    for chunk in result["chunks"]:
        for row in pq.read_table(root / chunk["path"]).to_pylist():
            chosen, exposures = draw_presentations(population, exposures, result["seed"], schedule[verified])
            if row["draw_hash"] != record_sha256([(item.row.image_id, item.ordinal) for item in chosen]):
                raise ValueError("offline sample draw or augmentation ordinal changed")
            verified += 1
    expected = [{"image_id": row.image_id, "task": row.task_index + 1, "label": row.remapped_class_index, "presentations": count}
                for row, count in zip(population.fitting, exposures, strict=True)]
    if pq.read_table(root / "exposures.parquet").to_pylist() != expected or sum(exposures) != result["training_presentations"]:
        raise ValueError("offline per-image exposures disagree with actual draws")
    audit = sealed_record({"schema_version": "imagenetr50-schedule-matched-draw-audit-v1", "result_hash": result["content_hash"],
                           "verified_updates": verified, "verified_presentations": sum(exposures),
                           "minimum_exposures": min(exposures), "maximum_exposures": max(exposures),
                           "training_only": True, "schedule_and_rates_matched": True})
    publish_immutable_json(root / "draw_audit.json", audit)
    return audit


def _commit(root: Path, job_hash: str, model: nn.Module, optimizer: torch.optim.Optimizer,
            position: WorkPosition, exposures: PVector, history: tuple[dict[str, object], ...],
            chunks: tuple[dict[str, object], ...], pending: tuple[dict[str, object], ...]) -> tuple[WorkPosition, tuple[dict[str, object], ...]]:
    """Publish bounded update evidence before atomically committing its optimizer state."""
    started = time.monotonic()
    if pending:
        relative = f"updates/{record_sha256(pending)}.parquet"
        digest = write_parquet(root / relative, pending)
        chunks += ({"path": relative, "sha256": digest, "steps": len(pending)},)
    atomic_torch_save(root / "checkpoint.pt", {
        "schema_version": "imagenetr50-schedule-matched-checkpoint-v1", "job_hash": job_hash,
        "trainables": trainable_state(model), "optimizer": optimizer.state_dict(), "position": asdict(position),
        "exposures": tuple(exposures), "blocks": history, "chunks": chunks,
        "cpu_rng": torch.random.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else (),
    })
    return replace(position, block_checkpoint_seconds=position.block_checkpoint_seconds + time.monotonic() - started), chunks


def train_schedule_job(
    root: Path, protocol_hash: str, seed: int, population: JointPopulation, schedule: tuple[ScheduledBatch, ...],
    optimizer_config: SRTConfig, loader: SelectedImageLoader, device: torch.device, model_factory: Callable[[int], nn.Module],
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    """Train a cold all-class model on a frozen update schedule with exact step resume.

    No validation or test rows enter this API. Work boundaries only trigger
    diagnostics/checkpoints; no model reset, classifier growth, or data filtering
    occurs. A checkpoint owns every committed block and update chunk, so orphan
    files from a failed commit cannot alter the reconstructed optimizer history.
    """
    schedule_hash = require_schedule(schedule, optimizer_config.batch_size)
    if population.validation or len(population.fitting) < max(batch.size for batch in schedule):
        raise ValueError("offline control requires one full training-only population")
    job = sealed_record({
        "schema_version": "imagenetr50-schedule-matched-job-v1", "protocol_hash": protocol_hash, "seed": seed,
        "schedule_hash": schedule_hash, "optimizer_config_hash": optimizer_config.content_hash,
        "maximum_batch_size": optimizer_config.batch_size,
        "optimizer": {name: getattr(optimizer_config, name) for name in ("lora_learning_rate", "head_learning_rate", "momentum", "weight_decay")},
        "fitting_ids_hash": record_sha256([row.image_id for row in population.fitting]), "fitting_examples": len(population.fitting),
        "probe_ids_hash": record_sha256([row.image_id for row in population.probe]), "probe_examples": len(population.probe),
        "planned_steps": len(schedule), "planned_presentations": sum(batch.size for batch in schedule),
        "data_policy": "all training images available from update one; uniform within each batch",
    })
    publish_immutable_json(root / "job.json", job)
    if (root / "result.json").is_file():
        return {**validate_schedule_job(root, job["content_hash"]), "invocation_optimizer_steps": 0}
    model = model_factory(seed).to(device)
    optimizer = optimizer_for(model, optimizer_config)
    position, exposures, history, chunks = WorkPosition(), pvector([0] * len(population.fitting)), (), ()
    if (root / "checkpoint.pt").is_file():
        state = torch.load(root / "checkpoint.pt", map_location="cpu", weights_only=False)
        if state.get("schema_version") != "imagenetr50-schedule-matched-checkpoint-v1" or state["job_hash"] != job["content_hash"]:
            raise ValueError("offline checkpoint identity changed")
        restore_trainables(model, state["trainables"])
        optimizer.load_state_dict(state["optimizer"])
        position, exposures = WorkPosition(**state["position"]), pvector(state["exposures"])
        history, chunks = tuple(state["blocks"]), tuple(state["chunks"])
        torch.random.set_rng_state(state["cpu_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        for row in history:
            if read_sealed(root / "blocks" / f"{row['block']:03d}/result.json") != row:
                raise ValueError("committed offline block changed")
        for chunk in chunks:
            if file_sha256(root / chunk["path"]) != chunk["sha256"]:
                raise ValueError("committed offline updates changed")
        del state
    initial_steps, pending = position.steps, ()
    progress = tqdm(total=len(schedule), initial=position.steps, desc=f"Offline seed {seed}", unit="update", mininterval=5)
    try:
        for batch in schedule[position.steps:]:
            started = time.monotonic()
            chosen, exposures = draw_presentations(population, exposures, seed, batch)
            images, labels = loader.load(chosen, True)
            loader_seconds = time.monotonic() - started
            _, loss_sum, correct, gradient = optimization_step(model, optimizer, images, labels, device)
            elapsed = time.monotonic() - started
            position = replace(position, steps=batch.step, presentations=position.presentations + batch.size,
                               block_presentations=position.block_presentations + batch.size, block_steps=position.block_steps + 1,
                               block_loss=position.block_loss + loss_sum, block_correct=position.block_correct + correct,
                               block_wall_seconds=position.block_wall_seconds + elapsed,
                               block_loader_seconds=position.block_loader_seconds + loader_seconds)
            pending += ({"step": batch.step, "block": batch.block, "batch_size": batch.size,
                         "draw_hash": record_sha256([(item.row.image_id, item.ordinal) for item in chosen]),
                         "loss_sum": loss_sum, "correct": correct, "gradient_norm": gradient,
                         "lora_learning_rate": optimizer.param_groups[0]["lr"], "head_learning_rate": optimizer.param_groups[1]["lr"],
                         "batch_wall_seconds": elapsed, "loader_seconds": loader_seconds},)
            boundary = batch.step == len(schedule) or schedule[batch.step].block != batch.block
            if boundary:
                evaluation_started = time.monotonic()
                probe, _ = evaluate_model(model, population.probe, loader, device, optimizer_config.batch_size, 50)
                folder = root / "blocks" / f"{batch.block:03d}"
                atomic_write(folder / "model.safetensors", tensor_bytes(trainable_state(model)))
                row = sealed_record({
                    "schema_version": "imagenetr50-schedule-matched-block-v1", "job_hash": job["content_hash"], "block": batch.block,
                    "steps_total": position.steps, "presentations_total": position.presentations,
                    "optimizer_steps": position.block_steps, "training_presentations": position.block_presentations,
                    "training_accuracy": 100 * position.block_correct / position.block_presentations,
                    "training_nll": position.block_loss / position.block_presentations,
                    "training_wall_seconds": position.block_wall_seconds, "loader_seconds": position.block_loader_seconds,
                    "checkpoint_wall_seconds": position.block_checkpoint_seconds, "fit_probe": probe,
                    "evaluation_and_artifact_seconds": time.monotonic() - evaluation_started,
                    "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                    "model_sha256": file_sha256(folder / "model.safetensors"),
                })
                atomic_write(folder / "result.json", canonical_json_bytes(row))
                history += (row,)
                position = WorkPosition(steps=position.steps, presentations=position.presentations)
                print(f"Work block {batch.block}: {position.steps:,} updates, {position.presentations:,} training images; "
                      f"clean fit probe {probe['accuracy']:.3f}% / NLL {probe['nll']:.5f}", flush=True)
            if boundary or batch.step % optimizer_config.checkpoint_steps == 0:
                position, chunks = _commit(root, job["content_hash"], model, optimizer, position, exposures, history, chunks, pending)
                pending = ()
                if progress_callback:
                    progress_callback({"seed": seed, "block": batch.block, "steps": position.steps,
                                       "presentations": position.presentations, "blocks": history,
                                       "partial_training_seconds": position.block_wall_seconds, "boundary": boundary})
            progress.update(1)
    finally:
        progress.close()
    exposure_hash = write_parquet(root / "exposures.parquet", tuple(
        {"image_id": row.image_id, "task": row.task_index + 1, "label": row.remapped_class_index, "presentations": count}
        for row, count in zip(population.fitting, exposures, strict=True)))
    result = sealed_record({
        "schema_version": "imagenetr50-schedule-matched-job-result-v1", "job_hash": job["content_hash"], "seed": seed,
        "schedule_hash": schedule_hash, "blocks": history, "chunks": chunks, "exposures_sha256": exposure_hash,
        "optimizer_steps": position.steps, "training_presentations": position.presentations,
        "model_forward_images": position.presentations + len(history) * len(population.probe),
        "training_wall_seconds": math.fsum(row["training_wall_seconds"] for row in history),
        "checkpoint_wall_seconds": math.fsum(row["checkpoint_wall_seconds"] for row in history) + position.block_checkpoint_seconds,
        "evaluation_and_artifact_seconds": math.fsum(row["evaluation_and_artifact_seconds"] for row in history),
        "stop_reason": "complete_frozen_schedule", "test_used": False,
    })
    publish_immutable_json(root / "result.json", result)
    return {**result, "invocation_optimizer_steps": position.steps - initial_steps}

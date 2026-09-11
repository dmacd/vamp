"""Validation-plateau stopping and resumable, full-population joint LoRA fitting."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
import math
from pathlib import Path
import time

from safetensors.torch import save as tensor_bytes
import torch
from torch import nn
from tqdm.auto import tqdm

from apm.continual.artifacts import atomic_write, canonical_json_bytes, file_sha256, publish_immutable_json, record_sha256
from apm.continual.vision.imagenetr.checkpoints import atomic_torch_save
from apm.continual.vision.imagenetr.data import ImageRecord
from apm.continual.vision.imagenetr.srt_config import SRTConfig
from apm.continual.vision.imagenetr.srt_data import Presentation, SelectedImageLoader
from apm.continual.vision.imagenetr.srt_evidence import read_sealed, sealed_record, write_parquet
from apm.continual.vision.imagenetr.srt_training import evaluate_model, optimization_step, optimizer_for, restore_trainables, trainable_state
from apm.continual.vision.imagenetr.training import deterministic_epoch_order


@dataclass(frozen=True, slots=True)
class ConvergenceRule:
    """Explicit validation tolerances and learning-rate refinement budget."""

    minimum_epochs: int = 30
    maximum_epochs: int = 160
    plateau_patience: int = 8
    terminal_patience: int = 12
    learning_rate_factor: float = .2
    learning_rate_reductions: int = 3
    accuracy_improvement_points: float = .1
    nll_improvement: float = .002

    def __post_init__(self) -> None:
        if (not 1 <= self.minimum_epochs <= self.maximum_epochs or self.plateau_patience < 1
                or self.terminal_patience < 1 or self.learning_rate_reductions < 1
                or not 0 < self.learning_rate_factor < 1 or self.accuracy_improvement_points <= 0
                or self.nll_improvement <= 0):
            raise ValueError("invalid joint convergence rule")


@dataclass(frozen=True, slots=True)
class PlateauState:
    """Separate significant-improvement anchors with one deterministic due counter."""

    accuracy_anchor: float | None = None
    nll_anchor: float | None = None
    stale_epochs: int = 0
    reductions: int = 0
    next_scale: float = 1.0
    stop_reason: str | None = None


def observe_validation(state: PlateauState, accuracy: float, nll: float, epoch: int, rule: ConvergenceRule) -> PlateauState:
    """Reduce rates on a two-metric plateau, never treating an epoch cap as convergence."""
    if state.stop_reason is not None or epoch < 1 or not 0 <= accuracy <= 100 or not math.isfinite(nll) or nll < 0:
        raise ValueError("invalid validation observation")
    accuracy_gain = state.accuracy_anchor is None or accuracy >= state.accuracy_anchor + rule.accuracy_improvement_points
    nll_gain = state.nll_anchor is None or nll <= state.nll_anchor - rule.nll_improvement
    following = replace(state, accuracy_anchor=accuracy if accuracy_gain else state.accuracy_anchor,
                        nll_anchor=nll if nll_gain else state.nll_anchor,
                        stale_epochs=0 if accuracy_gain or nll_gain else state.stale_epochs + 1)
    if following.reductions < rule.learning_rate_reductions and following.stale_epochs >= rule.plateau_patience:
        following = replace(following, reductions=following.reductions + 1, stale_epochs=0,
                            next_scale=following.next_scale * rule.learning_rate_factor)
    elif (following.reductions == rule.learning_rate_reductions and epoch >= rule.minimum_epochs
          and following.stale_epochs >= rule.terminal_patience):
        following = replace(following, stop_reason="validation_plateau")
    if epoch >= rule.maximum_epochs and following.stop_reason is None:
        following = replace(following, stop_reason="maximum_epochs_without_plateau")
    return following


def select_epochs(rows: tuple[dict[str, object], ...]) -> dict[str, int]:
    """Choose raw validation accuracy and NLL endpoints without test input."""
    if not rows or any(row.get("validation") is None for row in rows):
        raise ValueError("checkpoint selection requires validation-only epoch metrics")
    primary = min(rows, key=lambda row: (-row["validation"]["accuracy"], row["validation"]["nll"], row["epoch"]))
    probabilistic = min(rows, key=lambda row: (row["validation"]["nll"], -row["validation"]["accuracy"], row["epoch"]))
    return {"accuracy_selected": primary["epoch"], "nll_selected": probabilistic["epoch"], "terminal": rows[-1]["epoch"], "five_epoch": 5}


@dataclass(frozen=True, slots=True)
class JointPopulation:
    """Training-only fit, validation, and diagnostic identities with explicit isolation."""

    fitting: tuple[ImageRecord, ...]
    validation: tuple[ImageRecord, ...]
    probe: tuple[ImageRecord, ...]

    def __post_init__(self) -> None:
        fit_ids = frozenset(row.image_id for row in self.fitting)
        validation_ids = frozenset(row.image_id for row in self.validation)
        if (not self.fitting or not self.probe or len(fit_ids) != len(self.fitting)
                or len(validation_ids) != len(self.validation) or fit_ids & validation_ids
                or any(row.split != "train" for row in self.fitting + self.validation + self.probe)
                or not frozenset(row.image_id for row in self.probe) <= fit_ids):
            raise ValueError("joint fitting, validation, and probe populations are not isolated training identities")


@dataclass(frozen=True, slots=True)
class EpochPosition:
    """Partial-epoch statistics committed with the actual optimizer state."""

    epoch: int = 1
    next_batch: int = 0
    steps: int = 0
    presentations: int = 0
    epoch_loss: float = 0.0
    epoch_correct: int = 0
    epoch_presentations: int = 0
    epoch_wall_seconds: float = 0.0
    epoch_loader_seconds: float = 0.0
    gradient_norm_max: float = 0.0


def _save_checkpoint(root: Path, job_hash: str, model: nn.Module, optimizer: torch.optim.Optimizer,
                     position: EpochPosition, plateau: PlateauState, history: tuple[dict[str, object], ...]) -> None:
    atomic_torch_save(root / "checkpoint.pt", {
        "schema_version": "imagenetr50-joint-convergence-checkpoint-v1", "job_hash": job_hash,
        "trainables": trainable_state(model), "optimizer": optimizer.state_dict(),
        "position": asdict(position), "plateau": asdict(plateau), "history": history,
        "cpu_rng": torch.random.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else (),
    })


def validate_joint_job(root: Path, expected_hash: str | None = None) -> dict[str, object]:
    """Authenticate all committed epoch metrics and weights before reuse or reporting."""
    result = read_sealed(root / "result.json", "imagenetr50-joint-convergence-job-result-v1")
    job = read_sealed(root / "job.json", "imagenetr50-joint-convergence-job-v1")
    if result["job_hash"] != job["content_hash"] or (expected_hash is not None and result["job_hash"] != expected_hash):
        raise ValueError("joint convergence job identity changed")
    for expected_epoch, row in enumerate(result["epochs"], 1):
        epoch_root = root / "epochs" / f"{expected_epoch:03d}"
        if (row["epoch"] != expected_epoch or read_sealed(epoch_root / "result.json") != row
                or file_sha256(epoch_root / "model.safetensors") != row["model_sha256"]):
            raise ValueError("joint convergence epoch evidence changed")
        if row["validation"] is not None and file_sha256(epoch_root / "validation_predictions.parquet") != row["validation_predictions_sha256"]:
            raise ValueError("joint validation predictions changed")
    return result


def train_joint_job(
    root: Path, protocol_hash: str, seed: int, population: JointPopulation,
    optimizer_config: SRTConfig, rule: ConvergenceRule, loader: SelectedImageLoader,
    device: torch.device, model_factory: Callable[[int], nn.Module],
    fixed_schedule: tuple[float, ...] | None = None,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    """Fit one cold joint model with exact step resume and immutable epoch evidence.

    Development alone supplies validation identities. Full-data refits receive
    a sealed schedule and cannot access validation or test feedback. Checkpoints
    own committed history; a crash before epoch commit cannot adopt orphan files.
    """
    if ((fixed_schedule is None) != bool(population.validation)
            or (fixed_schedule is not None and (not fixed_schedule or any(not math.isfinite(value) or value <= 0 for value in fixed_schedule)))):
        raise ValueError("only development may select a schedule from validation")
    job = sealed_record({
        "schema_version": "imagenetr50-joint-convergence-job-v1", "protocol_hash": protocol_hash,
        "seed": seed, "phase": "development" if fixed_schedule is None else "full_data_refit",
        "fitting_ids_hash": record_sha256([row.image_id for row in population.fitting]),
        "validation_ids_hash": record_sha256([row.image_id for row in population.validation]),
        "probe_ids_hash": record_sha256([row.image_id for row in population.probe]),
        "fitting_examples": len(population.fitting), "validation_examples": len(population.validation),
        "probe_examples": len(population.probe), "fixed_schedule": fixed_schedule,
        "rule": asdict(rule), "optimizer_config_hash": optimizer_config.content_hash,
    })
    publish_immutable_json(root / "job.json", job)
    if (root / "result.json").is_file():
        return {**validate_joint_job(root, job["content_hash"]), "invocation_optimizer_steps": 0}
    model = model_factory(seed).to(device)
    optimizer = optimizer_for(model, optimizer_config)
    position, plateau, history = EpochPosition(), PlateauState(), ()
    if (root / "checkpoint.pt").is_file():
        state = torch.load(root / "checkpoint.pt", map_location="cpu", weights_only=False)
        if state.get("schema_version") != "imagenetr50-joint-convergence-checkpoint-v1" or state["job_hash"] != job["content_hash"]:
            raise ValueError("joint checkpoint identity changed")
        restore_trainables(model, state["trainables"])
        optimizer.load_state_dict(state["optimizer"])
        torch.random.set_rng_state(state["cpu_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        position, plateau, history = EpochPosition(**state["position"]), PlateauState(**state["plateau"]), tuple(state["history"])
        for row in history:
            if read_sealed(root / "epochs" / f"{row['epoch']:03d}" / "result.json") != row:
                raise ValueError("checkpoint epoch history changed")
    initial_steps = position.steps
    epoch_limit = rule.maximum_epochs if fixed_schedule is None else len(fixed_schedule)
    batches_per_epoch = math.ceil(len(population.fitting) / optimizer_config.batch_size)
    while position.epoch <= epoch_limit and plateau.stop_reason is None:
        epoch = position.epoch
        scale = plateau.next_scale if fixed_schedule is None else fixed_schedule[epoch - 1]
        for group, base_rate in zip(optimizer.param_groups, (optimizer_config.lora_learning_rate, optimizer_config.head_learning_rate), strict=True):
            group["lr"] = base_rate * scale
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        order = deterministic_epoch_order(len(population.fitting), seed + 50_000, epoch - 1)
        ordered = tuple(population.fitting[index] for index in order)
        progress = tqdm(total=batches_per_epoch, initial=position.next_batch, desc=f"{root.name} epoch {epoch}/{epoch_limit}", unit="batch", mininterval=5)
        for batch_index in range(position.next_batch, batches_per_epoch):
            started = time.monotonic()
            selected = ordered[batch_index * optimizer_config.batch_size:(batch_index + 1) * optimizer_config.batch_size]
            images, labels = loader.load(tuple(Presentation(row, epoch - 1) for row in selected), True)
            loader_seconds = time.monotonic() - started
            _, loss_sum, correct, gradient = optimization_step(model, optimizer, images, labels, device)
            position = replace(position, next_batch=batch_index + 1, steps=position.steps + 1,
                               presentations=position.presentations + len(selected), epoch_loss=position.epoch_loss + loss_sum,
                               epoch_correct=position.epoch_correct + correct, epoch_presentations=position.epoch_presentations + len(selected),
                               epoch_wall_seconds=position.epoch_wall_seconds + time.monotonic() - started,
                               epoch_loader_seconds=position.epoch_loader_seconds + loader_seconds,
                               gradient_norm_max=max(position.gradient_norm_max, gradient))
            if position.steps % optimizer_config.checkpoint_steps == 0:
                _save_checkpoint(root, job["content_hash"], model, optimizer, position, plateau, history)
            progress.update(1)
            if progress_callback is not None:
                progress_callback({"job": root.name, "epoch": epoch, "epoch_limit": epoch_limit, "batch": batch_index + 1,
                                   "batches": batches_per_epoch, "steps": position.steps, "presentations": position.presentations,
                                   "history": history, "learning_rate_scale": scale})
        progress.close()
        evaluation_started = time.monotonic()
        probe, _ = evaluate_model(model, population.probe, loader, device, optimizer_config.batch_size, 50)
        validation = None
        validation_prediction_hash = None
        epoch_root = root / "epochs" / f"{epoch:03d}"
        if population.validation:
            validation, predictions = evaluate_model(model, population.validation, loader, device, optimizer_config.batch_size, 50)
            validation_prediction_hash = write_parquet(epoch_root / "validation_predictions.parquet", predictions)
            plateau = observe_validation(plateau, validation["accuracy"], validation["nll"], epoch, rule)
        payload = tensor_bytes(trainable_state(model))
        atomic_write(epoch_root / "model.safetensors", payload)
        row = sealed_record({
            "schema_version": "imagenetr50-joint-convergence-epoch-v1", "job_hash": job["content_hash"], "epoch": epoch,
            "learning_rate_scale": scale, "lora_learning_rate": optimizer_config.lora_learning_rate * scale,
            "head_learning_rate": optimizer_config.head_learning_rate * scale,
            "training": {"accuracy": 100 * position.epoch_correct / position.epoch_presentations,
                         "nll": position.epoch_loss / position.epoch_presentations, "presentations": position.epoch_presentations,
                         "optimizer_steps": batches_per_epoch, "wall_seconds": position.epoch_wall_seconds,
                         "loader_seconds": position.epoch_loader_seconds, "gradient_norm_max": position.gradient_norm_max},
            "fit_probe": probe, "validation": validation, "plateau": asdict(plateau),
            "validation_predictions_sha256": validation_prediction_hash,
            "evaluation_wall_seconds": time.monotonic() - evaluation_started,
            "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
            "model_sha256": file_sha256(epoch_root / "model.safetensors"),
        })
        atomic_write(epoch_root / "result.json", canonical_json_bytes(row))
        history += (row,)
        position = EpochPosition(epoch + 1, 0, position.steps, position.presentations)
        _save_checkpoint(root, job["content_hash"], model, optimizer, position, plateau, history)
        metric = validation or probe
        print(f"Epoch {epoch}: {'validation' if validation else 'fit probe'} accuracy {metric['accuracy']:.3f}%, NLL {metric['nll']:.5f}; "
              f"scale {scale:g}; stale {plateau.stale_epochs}; reductions {plateau.reductions}; "
              f"{position.presentations:,} training images. {plateau.stop_reason or ''}", flush=True)
    result = sealed_record({
        "schema_version": "imagenetr50-joint-convergence-job-result-v1", "job_hash": job["content_hash"], "seed": seed,
        "phase": job["phase"], "epochs": history, "optimizer_steps": position.steps, "training_presentations": position.presentations,
        "stop_reason": plateau.stop_reason if fixed_schedule is None else "frozen_development_schedule_complete",
        "validation_converged": plateau.stop_reason == "validation_plateau" if fixed_schedule is None else None,
        "model_forward_images": position.presentations + len(history) * (len(population.probe) + len(population.validation)),
        "training_wall_seconds": math.fsum(row["training"]["wall_seconds"] for row in history),
        "evaluation_wall_seconds": math.fsum(row["evaluation_wall_seconds"] for row in history),
    })
    publish_immutable_json(root / "result.json", result)
    return {**result, "invocation_optimizer_steps": position.steps - initial_steps}

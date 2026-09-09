"""Resumable single-adapter training shared by SRT and exposure-matched replay."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
import gc
import math
from pathlib import Path
import time
from typing import Literal

import torch
from safetensors.torch import save as tensor_bytes
from torch import Tensor, nn
from torch.nn import functional as F
from tqdm.auto import tqdm

from apm.continual.artifacts import (
    atomic_write, canonical_json_bytes, file_sha256, publish_immutable_bytes,
    publish_immutable_json, record_sha256,
)
from apm.continual.vision.imagenetr.checkpoints import atomic_torch_save
from apm.continual.vision.imagenetr.data import ImageRecord
from apm.continual.vision.imagenetr.heads import AffineClassifier
from apm.continual.vision.imagenetr.model import AdapterVisionModel, create_pinned_backbone, require_trainable_boundary
from apm.continual.vision.imagenetr.srt_config import SRTConfig, presentation_budget
from apm.continual.vision.imagenetr.srt_data import Presentation, SelectedImageLoader
from apm.continual.vision.imagenetr.srt_evidence import (
    TraceBuffer, read_sealed, sealed_record, trace_batches, write_parquet,
)
from apm.continual.vision.imagenetr.srt_scheduler import (
    RecallPolicy, ReviewBatch, ReviewScheduler, observe_batch,
    restore_scheduler, scheduler_record, select_due, step_generator,
    uniform_indices,
)


Method = Literal["srt", "uniform"]


@dataclass(frozen=True, slots=True)
class TrainingPosition:
    """Counters committed atomically with weights, optimizer, and review state."""

    stage: int = 0
    steps: int = 0
    presentations: int = 0
    stage_steps: int = 0
    stage_presentations: int = 0
    introduced: int = 0
    stage_loss_sum: float = 0.0
    stage_correct: int = 0
    stage_wall: float = 0.0
    stage_loader_wall: float = 0.0
    stage_scheduler_wall: float = 0.0
    stage_checkpoint_wall: float = 0.0
    stage_peak_vram: int = 0


def trainable_state(model: nn.Module) -> dict[str, Tensor]:
    """Copy only adapter and classifier parameters for checkpoint persistence."""
    return {name: value.detach().cpu().contiguous().clone()
            for name, value in model.named_parameters() if value.requires_grad}


def restore_trainables(model: nn.Module, state: dict[str, Tensor]) -> None:
    """Require exact trainable keys and shapes before restoring committed weights."""
    parameters = {name: value for name, value in model.named_parameters() if value.requires_grad}
    if set(parameters) != set(state):
        raise ValueError("SRT checkpoint trainable boundary changed")
    with torch.no_grad():
        for name, parameter in parameters.items():
            if parameter.shape != state[name].shape:
                raise ValueError("SRT checkpoint parameter shape changed")
            parameter.copy_(state[name].to(parameter))


def grow_classifier(model: AdapterVisionModel, optimizer: torch.optim.Optimizer, classes: int, seed: int) -> None:
    """Append cold rows while preserving all old rows and SGD momentum exactly."""
    previous = model.classifier
    if classes != len(previous.class_ids) + 4:
        raise ValueError("the classifier must grow by exactly four classes")
    cold = AffineClassifier(tuple(range(200)), previous.weight.shape[1], seed + 10_000)
    following = AffineClassifier(tuple(range(classes)), previous.weight.shape[1],
                                 initial_rows=cold.selected_rows(range(classes))).to(previous.weight.device)
    following.restore_rows(previous.rows())
    for old, new in ((previous.weight, following.weight), (previous.bias, following.bias)):
        for group in optimizer.param_groups:
            group["params"] = [new if parameter is old else parameter for parameter in group["params"]]
        old_state = optimizer.state.pop(old, {})
        if set(old_state) - {"momentum_buffer"}:
            raise ValueError("SRT carry expects ordinary SGD momentum state")
        if "momentum_buffer" in old_state:
            momentum = torch.zeros_like(new)
            momentum[:old.shape[0]].copy_(old_state["momentum_buffer"])
            optimizer.state[new] = {"momentum_buffer": momentum}
    model.classifier = following


def optimizer_for(model: AdapterVisionModel, config: SRTConfig) -> torch.optim.SGD:
    """Use the unchanged joint-reference SGD recipe with separate head/LoRA rates."""
    parameters = tuple(parameter for name, parameter in model.named_parameters()
                       if parameter.requires_grad and not name.startswith("classifier."))
    return torch.optim.SGD(
        ({"params": parameters, "lr": config.lora_learning_rate},
         {"params": tuple(model.classifier.parameters()), "lr": config.head_learning_rate}),
        momentum=config.momentum, weight_decay=config.weight_decay,
    )


def optimization_step(
    model: nn.Module, optimizer: torch.optim.Optimizer, images: Tensor, labels: Tensor, device: torch.device,
) -> tuple[tuple[float, ...], float, int, float]:
    """Score recall before the update, then perform one finite all-seen-class step."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = model(images)
        losses = F.cross_entropy(logits.float(), labels, reduction="none")
        loss = losses.mean()
    confidences = tuple(float(value) for value in (-losses.detach()).exp().cpu().tolist())
    loss_sum, correct = float(losses.detach().sum().item()), int((logits.argmax(-1) == labels).sum().item())
    if not math.isfinite(loss_sum):
        raise FloatingPointError("nonfinite pre-update SRT cross-entropy")
    loss.backward()
    gradient_norm = float(torch.stack(tuple(parameter.grad.detach().float().norm()
                                          for parameter in model.parameters() if parameter.grad is not None)).norm().item())
    if not math.isfinite(gradient_norm):
        raise FloatingPointError("nonfinite SRT gradient")
    optimizer.step()
    return confidences, loss_sum, correct, gradient_norm


def evaluate_model(
    model: nn.Module, rows: tuple[ImageRecord, ...], loader: SelectedImageLoader,
    device: torch.device, batch_size: int, stage: int,
) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    """Evaluate one sealed prefix without passing task identities into the model."""
    if not rows or any(row.task_index >= stage for row in rows):
        raise ValueError("evaluation rows must belong to arrived classes")
    started, loss_sum, correct = time.monotonic(), 0.0, 0
    task_counts, task_correct, predictions = [0] * stage, [0] * stage, []
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(rows), batch_size):
            selected = rows[offset:offset + batch_size]
            images, labels = loader.load(tuple(Presentation(row, 0) for row in selected), False)
            images, labels = images.to(device), labels.to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(images).float()
                losses = F.cross_entropy(logits, labels, reduction="none")
            predicted = logits.argmax(-1).cpu().tolist()
            nll = losses.cpu().tolist()
            loss_sum += math.fsum(nll)
            for row, prediction, value in zip(selected, predicted, nll, strict=True):
                is_correct = prediction == row.remapped_class_index
                correct += int(is_correct)
                task_counts[row.task_index] += 1
                task_correct[row.task_index] += int(is_correct)
                predictions.append({"image_id": row.image_id, "task": row.task_index + 1,
                                    "label": row.remapped_class_index, "prediction": prediction, "nll": value})
    return {
        "accuracy": 100 * correct / len(rows), "nll": loss_sum / len(rows), "examples": len(rows),
        "task_examples": task_counts, "task_correct": task_correct, "wall_seconds": time.monotonic() - started,
    }, tuple(predictions)


def _checkpoint(
    root: Path, job_hash: str, model: nn.Module, optimizer: torch.optim.Optimizer,
    scheduler: ReviewScheduler, position: TrainingPosition, chunks: tuple[dict[str, object], ...],
    stages: tuple[dict[str, object], ...], buffer: TraceBuffer,
) -> tuple[tuple[dict[str, object], ...], TrainingPosition]:
    started = time.monotonic()
    if buffer.batches:
        chunks += (buffer.publish(root),)
    atomic_torch_save(root / "checkpoint.pt", {
        "schema_version": "imagenetr50-srt-checkpoint-v1", "job_hash": job_hash,
        "trainables": trainable_state(model), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler_record(scheduler), "position": asdict(position),
        "chunks": chunks, "stages": stages, "cpu_rng": torch.random.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else (),
    })
    elapsed = time.monotonic() - started
    return chunks, replace(position, stage_checkpoint_wall=position.stage_checkpoint_wall + elapsed,
                           stage_wall=position.stage_wall + elapsed)


def _choose_batch(
    scheduler: ReviewScheduler, position: TrainingPosition, policy: RecallPolicy, method: Method,
    current: tuple[int, ...], historical: tuple[int, ...], budget: int, config: SRTConfig,
    paired: tuple[dict[str, object], ...],
) -> tuple[ReviewBatch, ReviewScheduler]:
    limit = min(config.batch_size, budget - position.stage_presentations)
    if position.introduced < len(current):
        images = current[position.introduced:position.introduced + limit]
        return ReviewBatch(images, 0, len(images), True), scheduler
    if method == "srt":
        return select_due(scheduler, policy, limit, config.seed, position.steps + 1)
    template = paired[position.stage_steps]
    old_count, new_count = int(template["historical_count"]), int(template["current_count"])
    if template["introduction"] or old_count + new_count > limit:
        raise ValueError("uniform replay differs from its paired SRT presentation boundary")
    generator = step_generator(config.seed, position.stage, position.steps + 1, "uniform")
    old = tuple(historical[index] for index in uniform_indices(len(historical), old_count, generator))
    new = tuple(current[index] for index in uniform_indices(len(current), new_count, generator))
    return ReviewBatch(old + new, old_count, new_count, False), scheduler


def _train_stage(
    root: Path, job_hash: str, model: AdapterVisionModel, optimizer: torch.optim.Optimizer,
    scheduler: ReviewScheduler, position: TrainingPosition, chunks: tuple[dict[str, object], ...],
    stages: tuple[dict[str, object], ...], rows: tuple[ImageRecord, ...], current: tuple[int, ...],
    historical: tuple[int, ...], policy: RecallPolicy, method: Method, capacity: int,
    loader: SelectedImageLoader, device: torch.device, config: SRTConfig,
    paired: tuple[dict[str, object], ...], progress_callback: Callable[[dict[str, object]], None] | None,
    stop_after_steps: int | None,
) -> tuple[ReviewScheduler, TrainingPosition, tuple[dict[str, object], ...], bool]:
    budget = presentation_budget(len(current), len(historical), capacity)
    buffer = TraceBuffer()
    progress = tqdm(total=budget, initial=position.stage_presentations,
                    desc=f"{method} H{capacity} task {position.stage:02d}", unit="image", mininterval=5)
    try:
        while position.stage_presentations < budget:
            started = time.monotonic()
            batch, selected_scheduler = _choose_batch(scheduler, position, policy, method, current, historical, budget, config, paired)
            selection_wall = time.monotonic() - started
            chosen = tuple(Presentation(rows[image], scheduler.states[image].exposures + 1 if image in scheduler.states else 1)
                           for image in batch.images)
            if any(item.row.task_index >= position.stage for item in chosen):
                raise ValueError("future task image reached the optimizer")
            loading_started = time.monotonic()
            images, labels = loader.load(chosen, True)
            loading_wall = time.monotonic() - loading_started
            confidences, loss_sum, correct, gradient_norm = optimization_step(model, optimizer, images, labels, device)
            scheduling_started = time.monotonic()
            scheduler, events = observe_batch(selected_scheduler, batch, confidences, policy, method,
                                               position.steps + 1, position.presentations)
            scheduling_wall = selection_wall + time.monotonic() - scheduling_started
            elapsed = time.monotonic() - started
            position = replace(
                position, steps=position.steps + 1, presentations=position.presentations + len(batch.images),
                stage_steps=position.stage_steps + 1, stage_presentations=position.stage_presentations + len(batch.images),
                introduced=position.introduced + (len(batch.images) if batch.introduction else 0),
                stage_loss_sum=position.stage_loss_sum + loss_sum, stage_correct=position.stage_correct + correct,
                stage_wall=position.stage_wall + elapsed, stage_loader_wall=position.stage_loader_wall + loading_wall,
                stage_scheduler_wall=position.stage_scheduler_wall + scheduling_wall,
                stage_peak_vram=max(position.stage_peak_vram, torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0),
            )
            batch_record = {
                "stage": position.stage, "step": position.steps, "stage_step": position.stage_steps,
                "presentations": len(batch.images), "historical_count": batch.historical_count,
                "current_count": batch.current_count, "introduction": batch.introduction,
                "clock": selected_scheduler.clock, "clock_advance": batch.clock_advance,
                "historical_due": batch.historical_due, "current_due": batch.current_due,
                "loss_sum": loss_sum, "correct": correct, "gradient_norm": gradient_norm,
                "wall_seconds": elapsed, "loader_wall_seconds": loading_wall, "scheduler_wall_seconds": scheduling_wall,
            }
            if method == "uniform" and any(batch_record[name] != paired[position.stage_steps - 1][name]
                                            for name in ("historical_count", "current_count", "presentations", "introduction", "step")):
                raise ValueError("uniform control no longer matches realized SRT exposure")
            buffer = buffer.append(events, batch_record)
            progress.update(len(batch.images))
            should_stop = stop_after_steps is not None and position.steps >= stop_after_steps
            if position.steps % config.checkpoint_steps == 0 or position.stage_presentations == budget or should_stop:
                chunks, position = _checkpoint(root, job_hash, model, optimizer, scheduler, position, chunks, stages, buffer)
                buffer = TraceBuffer()
                if progress_callback:
                    progress_callback({"job": root.name, "method": method, "capacity": capacity, "stage": position.stage,
                                       "stage_presentations": position.stage_presentations, "stage_budget": budget,
                                       "optimizer_steps": position.steps, "image_presentations": position.presentations})
            if should_stop:
                return scheduler, position, chunks, True
    finally:
        progress.close()
    return scheduler, position, chunks, False


def run_training_job(
    root: Path, protocol_hash: str, checkpoint_path: Path, config: SRTConfig, policy: RecallPolicy,
    capacity: int, method: Method, training_rows: tuple[ImageRecord, ...], evaluation_rows: tuple[ImageRecord, ...],
    loader: SelectedImageLoader, device: torch.device, target_stage: int,
    paired_root: Path | None = None, progress_callback: Callable[[dict[str, object]], None] | None = None,
    stop_after_steps: int | None = None,
) -> dict[str, object]:
    """Continue one immutable training stream, returning without model loading on reuse."""
    if (not training_rows or any(row.split != "train" for row in training_rows)
            or {row.image_id for row in training_rows} & {row.image_id for row in evaluation_rows}
            or method not in {"srt", "uniform"} or (method == "uniform") != (paired_root is not None)):
        raise ValueError("invalid SRT job population or paired replay definition")
    job = sealed_record({
        "schema_version": "imagenetr50-srt-job-v1", "protocol_hash": protocol_hash, "policy": asdict(policy),
        "capacity": capacity, "method": method, "config_hash": config.content_hash,
        "training_ids_hash": record_sha256([row.image_id for row in training_rows]),
        "evaluation_ids_hash": record_sha256([row.image_id for row in evaluation_rows]),
        "paired_job_hash": read_sealed(paired_root / "job.json")["content_hash"] if paired_root else None,
    })
    publish_immutable_json(root / "job.json", job)
    if (root / "result.json").is_file():
        result = read_sealed(root / "result.json", "imagenetr50-srt-job-result-v1")
        if result["job_hash"] != job["content_hash"]:
            raise ValueError("SRT result belongs to another job")
        if len(result["rows"]) >= target_stage:
            return {**result, "invocation_optimizer_steps": 0}
    saved = torch.load(root / "checkpoint.pt", map_location="cpu", weights_only=False) if (root / "checkpoint.pt").is_file() else None
    if saved and (saved.get("schema_version") != "imagenetr50-srt-checkpoint-v1" or saved["job_hash"] != job["content_hash"]):
        raise ValueError("SRT checkpoint identity changed")
    position = TrainingPosition(**saved["position"]) if saved else TrainingPosition()
    scheduler = restore_scheduler(saved["scheduler"]) if saved else ReviewScheduler()
    chunks, stages = (tuple(saved["chunks"]), tuple(saved["stages"])) if saved else ((), ())
    invocation_steps = position.steps
    classes = 4 * max(1, position.stage)
    cold = AffineClassifier(tuple(range(200)), 768, config.seed + 10_000)
    model = AdapterVisionModel(create_pinned_backbone(checkpoint_path), tuple(range(classes)),
                               initialization_seed=config.seed, initial_rows=cold.selected_rows(range(classes))).to(device)
    require_trainable_boundary(model)
    optimizer = optimizer_for(model, config)
    if saved:
        restore_trainables(model, saved["trainables"])
        optimizer.load_state_dict(saved["optimizer"])
        torch.random.set_rng_state(saved["cpu_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        # Validate only committed files; orphaned chunks cannot affect a resume.
        for chunk in chunks:
            if file_sha256(root / chunk["path"] / "events.parquet") != chunk["events_sha256"]:
                raise ValueError("committed SRT event evidence changed")
    del saved, cold
    try:
        for stage in range(len(stages) + 1, target_stage + 1):
            if position.stage < stage:
                if stage > 1:
                    grow_classifier(model, optimizer, 4 * stage, config.seed)
                scheduler = scheduler.arrive(stage)
                position = TrainingPosition(stage=stage, steps=position.steps, presentations=position.presentations)
            current = tuple(index for index, row in enumerate(training_rows) if row.task_index == stage - 1)
            generator = step_generator(config.seed, stage, 0, "introduction")
            current = tuple(current[int(index)] for index in generator.permutation(len(current)))
            historical = tuple(index for index, row in enumerate(training_rows) if row.task_index < stage - 1)
            paired_stage = read_sealed(paired_root / f"stages/{stage:03d}/result.json") if paired_root else None
            paired = trace_batches(paired_root, paired_stage["trace_chunks"]) if paired_root else ()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            scheduler, position, chunks, stopped = _train_stage(
                root, job["content_hash"], model, optimizer, scheduler, position, chunks, stages,
                training_rows, current, historical, policy, method, capacity, loader, device, config, paired,
                progress_callback, stop_after_steps,
            )
            if stopped:
                return {"stopped_for_resume_test": True, "optimizer_steps": position.steps,
                        "invocation_optimizer_steps": position.steps - invocation_steps}
            stage_root = root / f"stages/{stage:03d}"
            model_payload = tensor_bytes(trainable_state(model))
            publish_immutable_bytes(stage_root / "model.safetensors", model_payload)
            model_hash = file_sha256(stage_root / "model.safetensors")
            stage_chunks = tuple(chunk for chunk in chunks if chunk["stage"] == stage)
            if (stage_root / "result.json").is_file():
                row = read_sealed(stage_root / "result.json", "imagenetr50-srt-stage-v1")
                if row["model_sha256"] != model_hash or row["job_hash"] != job["content_hash"]:
                    raise ValueError("sealed SRT stage differs from resumed trainables")
            else:
                evaluation, predictions = evaluate_model(model, tuple(row for row in evaluation_rows if row.task_index < stage),
                                                         loader, device, config.batch_size, stage)
                prediction_hash = write_parquet(stage_root / "predictions.parquet", predictions)
                due = scheduler.ready()
                row = sealed_record({
                    "schema_version": "imagenetr50-srt-stage-v1", "job_hash": job["content_hash"], "stage": stage,
                    "capacity": capacity, "method": method, "policy": policy.name,
                    "current_examples": len(current), "historical_examples_available": len(historical),
                    "model_sha256": model_hash, "predictions_sha256": prediction_hash, "trace_chunks": stage_chunks,
                    "fit": {"image_presentations": position.stage_presentations, "optimizer_steps": position.stage_steps,
                            "optimizer_steps_total": position.steps, "wall_seconds": position.stage_wall,
                            "loader_wall_seconds": position.stage_loader_wall, "scheduler_wall_seconds": position.stage_scheduler_wall,
                            "checkpoint_wall_seconds": position.stage_checkpoint_wall,
                            "train_nll": position.stage_loss_sum / position.stage_presentations,
                            "train_accuracy": 100 * position.stage_correct / position.stage_presentations,
                            "peak_vram_bytes": position.stage_peak_vram},
                    "evaluation": evaluation,
                    "scheduler": {"clock": scheduler.clock, "historical_due": len(due.historical), "current_due": len(due.current),
                                  "retained_images": len(scheduler.states), "introduced_images": position.introduced,
                                  "never_historically_replayed": sum(not state.historical_reviews for state in scheduler.states.values())},
                    "lora_parameters": sum(parameter.numel() for name, parameter in model.named_parameters()
                                           if parameter.requires_grad and not name.startswith("classifier.")),
                    "classifier_parameters": sum(parameter.numel() for parameter in model.classifier.parameters()),
                })
                publish_immutable_json(stage_root / "result.json", row)
            stages += (row,)
            chunks, position = _checkpoint(root, job["content_hash"], model, optimizer, scheduler, position, chunks, stages, TraceBuffer())
            result = sealed_record({"schema_version": "imagenetr50-srt-job-result-v1", "job_hash": job["content_hash"],
                                    "method": method, "capacity": capacity, "policy": asdict(policy), "rows": stages,
                                    "optimizer_steps": position.steps, "image_presentations": position.presentations})
            atomic_write(root / "result.json", canonical_json_bytes(result))
            print(f"Completed {root.name} task {stage}/50: {row['evaluation']['accuracy']:.3f}% "
                  f"NLL {row['evaluation']['nll']:.4f}; {position.presentations:,} cumulative presentations", flush=True)
        return {**result, "invocation_optimizer_steps": position.steps - invocation_steps}
    finally:
        del optimizer, model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

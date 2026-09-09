"""Training, replay, evaluation, and named optimizer carry for persistent affine frontiers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import time

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from apm.continual.artifacts import ChainedJsonlLedger, record_sha256
from apm.continual.vision.imagenetr.checkpoints import atomic_torch_save
from apm.continual.vision.imagenetr.data import ImageRecord, ManifestDataset
from apm.continual.vision.imagenetr.frontier_adaptation_training import warmup_cosine_multiplier
from apm.continual.vision.imagenetr.integrator_bank import class_stratified_reservoir
from apm.continual.vision.imagenetr.persistent_affine_config import PersistentAffineConfig
from apm.continual.vision.imagenetr.persistent_affine_model import (
    PersistentFrontierIntegrator,
    export_model_state,
    load_exact_model_state,
    raw_union_logits,
    true_node_logits,
)
from apm.continual.vision.imagenetr.training import deterministic_epoch_order, os_cpu_workers


EPOCH_LEDGER_SCHEMA = "imagenetr50-persistent-affine-epoch-v1"
REPLAY_NAMESPACE = "imagenetr50-persistent-affine-rotating-replay-v1"


@dataclass(frozen=True, slots=True)
class StageFit:
    """Finite work and final training objective for one online arrival."""

    epochs: int
    optimizer_steps: int
    optimizer_steps_total: int
    image_presentations: int
    train_accuracy: float
    train_nll: float
    peak_vram_bytes: int
    wall_seconds: float

    def as_record(self) -> dict[str, object]:
        """Return canonical JSON-compatible fit evidence."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PrefixEvaluation:
    """Task-free and diagnostic measurements for one sealed test prefix."""

    accuracy: float
    nll: float
    examples: int
    raw_union_accuracy: float
    true_node_oracle_accuracy: float
    task_correct: tuple[int, ...]
    task_examples: tuple[int, ...]
    wall_seconds: float

    def as_record(self) -> dict[str, object]:
        """Return canonical JSON-compatible evaluation evidence."""
        record = asdict(self)
        record["task_correct"] = list(self.task_correct)
        record["task_examples"] = list(self.task_examples)
        return record


@dataclass(frozen=True, slots=True)
class ReplayPopulation:
    """Exact current-plus-rotating-history population for one stage."""

    rows: tuple[ImageRecord, ...]
    current_examples: int
    historical_examples: int
    current_image_ids_hash: str
    historical_image_ids_hash: str
    image_ids_hash: str
    namespace: str

    def as_record(self) -> dict[str, object]:
        """Return compact identity evidence without duplicating all rows."""
        return {
            "current_examples": self.current_examples,
            "current_image_ids_hash": self.current_image_ids_hash,
            "historical_examples": self.historical_examples,
            "historical_image_ids_hash": self.historical_image_ids_hash,
            "image_ids_hash": self.image_ids_hash,
            "namespace": self.namespace,
            "training_examples": len(self.rows),
        }


def rotating_replay_population(
    all_train_rows: Sequence[ImageRecord],
    stage: int,
    historical_capacity: int,
    seed: int,
) -> ReplayPopulation:
    """Select all current rows plus a deterministic stage-keyed historical draw."""
    if (
        not 1 <= stage <= 50
        or historical_capacity < 1
        or seed < 0
        or any(row.split != "train" for row in all_train_rows)
    ):
        raise ValueError("rotating replay inputs are invalid")
    current = tuple(row for row in all_train_rows if row.task_index == stage - 1)
    history = tuple(row for row in all_train_rows if row.task_index < stage - 1)
    if not current:
        raise ValueError("current task has no training rows")
    namespace = f"{REPLAY_NAMESPACE}:seed={seed}:stage={stage:03d}"
    if len(history) <= historical_capacity:
        historical = history
    else:
        reservoir = class_stratified_reservoir(history, historical_capacity, namespace)
        by_id = {row.image_id: row for row in history}
        historical = tuple(by_id[image_id] for image_id in reservoir.image_ids)
    rows = tuple(sorted((*current, *historical), key=lambda row: row.image_id))
    current_ids = tuple(sorted(row.image_id for row in current))
    historical_ids = tuple(sorted(row.image_id for row in historical))
    if (
        len(rows) != len(current) + len(historical)
        or len({row.image_id for row in rows}) != len(rows)
        or len(historical) > historical_capacity
    ):
        raise ValueError("rotating replay population overlaps or exceeds H")
    return ReplayPopulation(
        rows,
        len(current),
        len(historical),
        record_sha256(list(current_ids)),
        record_sha256(list(historical_ids)),
        record_sha256([row.image_id for row in rows]),
        namespace,
    )


def _stable_parameters(model: PersistentFrontierIntegrator) -> dict[str, nn.Parameter]:
    parameters = {f"integrator.{name}": parameter for name, parameter in model.integrator.named_parameters()}
    head_count = len(parameters)
    for node_hash, node_model in zip(model.node_hashes, model.node_models, strict=True):
        for name, parameter in node_model.named_parameters():
            if name.endswith("lora_a") or name.endswith("lora_b"):
                parameters[f"node:{node_hash}:{name}"] = parameter
    if len(parameters) != head_count + 48 * len(model.node_hashes):
        raise ValueError("stable optimizer parameter map is incomplete")
    return parameters


def create_optimizer(
    model: PersistentFrontierIntegrator, config: PersistentAffineConfig
) -> torch.optim.AdamW:
    """Create the two-group AdamW optimizer used at every arrival."""
    return torch.optim.AdamW(
        (
            {
                "params": tuple(model.integrator.parameters()),
                "lr": config.integrator_peak_learning_rate,
                "name": "integrator",
            },
            {
                "params": model.lora_parameters,
                "lr": config.lora_peak_learning_rate,
                "name": "lora",
            },
        ),
        weight_decay=config.weight_decay,
    )


def export_named_optimizer_state(
    optimizer: torch.optim.AdamW, model: PersistentFrontierIntegrator
) -> dict[str, object]:
    """Serialize AdamW state by semantic node identity instead of parameter index."""
    states: dict[str, dict[str, object]] = {}
    for name, parameter in _stable_parameters(model).items():
        if parameter not in optimizer.state:
            continue
        states[name] = {
            key: value.detach().cpu().clone() if isinstance(value, Tensor) else value
            for key, value in optimizer.state[parameter].items()
        }
    return {
        "node_hashes": model.node_hashes,
        "seen_class_ids": model.seen_class_ids,
        "states": states,
    }


def _to_parameter_state(value: object, parameter: nn.Parameter) -> object:
    if not isinstance(value, Tensor):
        return value
    if value.ndim == 0:
        return value.detach().cpu().clone()
    return value.to(device=parameter.device, dtype=parameter.dtype).clone()


def restore_named_optimizer_state(
    optimizer: torch.optim.AdamW,
    model: PersistentFrontierIntegrator,
    snapshot: Mapping[str, object] | None,
) -> int:
    """Carry affine and surviving-node moments across a changing frontier."""
    if snapshot is None:
        return 0
    old_hashes = tuple(str(value) for value in snapshot.get("node_hashes", ()))
    old_seen = tuple(int(value) for value in snapshot.get("seen_class_ids", ()))
    raw_states = snapshot.get("states")
    if not isinstance(raw_states, Mapping) or len(set(old_hashes)) != len(old_hashes):
        raise ValueError("named optimizer snapshot is malformed")
    parameters = _stable_parameters(model)
    restored = 0
    for name, parameter in parameters.items():
        raw_state = raw_states.get(name)
        if not isinstance(raw_state, Mapping):
            continue
        state: dict[str, object] = {}
        for key, value in raw_state.items():
            if (
                name.startswith("integrator.")
                and isinstance(value, Tensor)
                and value.ndim > 0
            ):
                state[key] = model.integrator.project_parameter(
                    name.removeprefix("integrator."), value, old_hashes,
                    model.node_hashes, old_seen, torch.zeros_like(parameter),
                )
            else:
                if isinstance(value, Tensor) and value.ndim > 0 and value.shape != parameter.shape:
                    raise ValueError("named optimizer tensor shape changed unexpectedly")
                state[key] = _to_parameter_state(value, parameter)
        optimizer.state[parameter] = state
        restored += 1
    return restored


def _checkpoint_record(
    *,
    protocol_hash: str,
    capacity: int,
    stage: int,
    frontier_hash: str,
    population_hash: str,
    epoch: int,
    optimizer_steps_before_stage: int,
    optimizer_steps_total: int,
    image_presentations: int,
    history_rows: int,
    model: PersistentFrontierIntegrator,
    optimizer: torch.optim.AdamW,
) -> dict[str, object]:
    return {
        "capacity": capacity,
        "epoch": epoch,
        "frontier_hash": frontier_hash,
        "history_rows": history_rows,
        "image_presentations": image_presentations,
        "model_state": export_model_state(model),
        "optimizer_state": export_named_optimizer_state(optimizer, model),
        "optimizer_steps_before_stage": optimizer_steps_before_stage,
        "optimizer_steps_total": optimizer_steps_total,
        "population_hash": population_hash,
        "protocol_hash": protocol_hash,
        "schema_version": "imagenetr50-persistent-frontier-checkpoint-v1",
        "stage": stage,
    }


def fit_online_stage(
    *,
    model: PersistentFrontierIntegrator,
    optimizer: torch.optim.AdamW,
    config: PersistentAffineConfig,
    protocol_hash: str,
    capacity: int,
    stage: int,
    frontier_hash: str,
    population: ReplayPopulation,
    prepared_root: Path,
    train_transform: object,
    checkpoint_path: Path,
    history: ChainedJsonlLedger,
    optimizer_steps_total: int,
    device: torch.device,
) -> tuple[StageFit, dict[str, object], dict[str, object]]:
    """Fit or resume one fixed-budget arrival and return continuing state."""
    epochs = config.epoch_map[capacity]
    start_epoch = 0
    optimizer_steps_before_stage = optimizer_steps_total
    presentations = 0
    elapsed_before = 0.0
    peak_before = 0
    if checkpoint_path.is_file():
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (
            saved.get("schema_version") != "imagenetr50-persistent-frontier-checkpoint-v1"
            or saved.get("protocol_hash") != protocol_hash
            or saved.get("capacity") != capacity
            or saved.get("stage") != stage
            or saved.get("frontier_hash") != frontier_hash
            or saved.get("population_hash") != population.image_ids_hash
        ):
            raise ValueError("persistent affine checkpoint identity changed")
        raw_model = saved.get("model_state")
        raw_optimizer = saved.get("optimizer_state")
        if not isinstance(raw_model, Mapping) or not isinstance(raw_optimizer, Mapping):
            raise ValueError("persistent affine checkpoint state is malformed")
        load_exact_model_state(model, raw_model)
        optimizer.state.clear()
        restore_named_optimizer_state(optimizer, model, raw_optimizer)
        start_epoch = int(saved["epoch"])
        optimizer_steps_total = int(saved["optimizer_steps_total"])
        optimizer_steps_before_stage = int(saved["optimizer_steps_before_stage"])
        presentations = int(saved["image_presentations"])
        history.truncate(int(saved["history_rows"]))
        retained = tuple(row for row in history.rows if row["stage"] == stage and row["capacity"] == capacity)
        if retained:
            elapsed_before = float(retained[-1]["wall_seconds_stage"])
            peak_before = int(retained[-1]["peak_vram_bytes_stage"])
    steps_per_epoch = math.ceil(len(population.rows) / config.microbatch_size)
    horizon_steps = steps_per_epoch * config.schedule_horizon_epochs
    started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    from tqdm.auto import tqdm

    progress = tqdm(
        range(start_epoch + 1, epochs + 1),
        initial=start_epoch,
        total=epochs,
        desc=f"integrator H={capacity:,} stage {stage:02d}",
        unit="epoch",
    )
    final_nll = final_accuracy = math.nan
    for epoch in progress:
        order = deterministic_epoch_order(
            len(population.rows),
            config.seed + capacity + stage * 10_000,
            epoch - 1,
        )
        ordered = tuple(population.rows[index] for index in order)
        loader = DataLoader(
            ManifestDataset(
                prepared_root,
                ordered,
                train_transform,
                config.seed + capacity + stage * 10_000,
                epoch - 1,
            ),
            batch_size=config.microbatch_size,
            shuffle=False,
            num_workers=min(config.num_workers, os_cpu_workers()),
            pin_memory=device.type == "cuda",
            persistent_workers=False,
        )
        model.set_training_mode()
        nll_sum = 0.0
        correct = examples = 0
        gradient_norm_sum = 0.0
        for batch_index, (images, labels, _image_ids) in enumerate(loader):
            local_step = (epoch - 1) * steps_per_epoch + batch_index
            multiplier = warmup_cosine_multiplier(
                local_step,
                horizon_steps,
                config.warmup_fraction,
                config.minimum_learning_rate_ratio,
            )
            for group in optimizer.param_groups:
                group["lr"] = multiplier * (
                    config.integrator_peak_learning_rate
                    if group["name"] == "integrator"
                    else config.lora_peak_learning_rate
                )
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = model(
                    images,
                    adapt_lora=True,
                    activation_recomputation=config.activation_recomputation,
                )
                loss = F.cross_entropy(logits, labels)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.trainable_parameters, config.gradient_clip_norm
            )
            optimizer.step()
            optimizer_steps_total += 1
            presentations += len(labels)
            nll_sum += float(loss.detach()) * len(labels)
            correct += int((logits.detach().argmax(dim=1) == labels).sum())
            examples += len(labels)
            gradient_norm_sum += float(gradient_norm.detach())
        final_nll = nll_sum / examples
        final_accuracy = 100.0 * correct / examples
        elapsed = elapsed_before + time.monotonic() - started
        peak = max(
            peak_before,
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        )
        history.append(
            {
                "capacity": capacity,
                "epoch": epoch,
                "frontier_hash": frontier_hash,
                "gradient_norm_mean": gradient_norm_sum / steps_per_epoch,
                "historical_examples": population.historical_examples,
                "image_presentations_stage": presentations,
                "optimizer_steps_total": optimizer_steps_total,
                "peak_vram_bytes_stage": int(peak),
                "stage": stage,
                "train_accuracy": final_accuracy,
                "train_nll": final_nll,
                "training_examples": len(population.rows),
                "wall_seconds_stage": elapsed,
            }
        )
        atomic_torch_save(
            checkpoint_path,
            _checkpoint_record(
                protocol_hash=protocol_hash,
                capacity=capacity,
                stage=stage,
                frontier_hash=frontier_hash,
                population_hash=population.image_ids_hash,
                epoch=epoch,
                optimizer_steps_before_stage=optimizer_steps_before_stage,
                optimizer_steps_total=optimizer_steps_total,
                image_presentations=presentations,
                history_rows=len(history.rows),
                model=model,
                optimizer=optimizer,
            ),
        )
        progress.set_postfix(acc=f"{final_accuracy:.1f}", nll=f"{final_nll:.3f}")
    progress.close()
    if start_epoch == epochs:
        rows = [
            row for row in history.rows
            if int(row["stage"]) == stage and int(row["capacity"]) == capacity
        ]
        final = rows[-1]
        final_nll = float(final["train_nll"])
        final_accuracy = float(final["train_accuracy"])
    peak = max(
        peak_before,
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
    )
    fit = StageFit(
        epochs,
        optimizer_steps_total - optimizer_steps_before_stage,
        optimizer_steps_total,
        presentations,
        final_accuracy,
        final_nll,
        int(peak),
        elapsed_before + time.monotonic() - started,
    )
    return fit, export_model_state(model), export_named_optimizer_state(optimizer, model)


def evaluate_prefix(
    *,
    model: PersistentFrontierIntegrator,
    prepared_root: Path,
    rows: Sequence[ImageRecord],
    transform: object,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> PrefixEvaluation:
    """Evaluate task-free affine predictions and explicitly diagnostic controls."""
    if not rows or any(row.split != "test" for row in rows):
        raise ValueError("prefix evaluation requires only test rows")
    loader = DataLoader(
        ManifestDataset(prepared_root, rows, transform, 0, 0),
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(num_workers, os_cpu_workers()),
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )
    model.set_evaluation_mode()
    nll_sum = 0.0
    correct = raw_correct = oracle_correct = examples = 0
    task_correct = [0] * (max(row.task_index for row in rows) + 1)
    task_examples = [0] * len(task_correct)
    started = time.monotonic()
    with torch.inference_mode():
        for images, labels, _image_ids in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits, _features, local_scores = model.forward_components(
                    images, adapt_lora=False, activation_recomputation=False
                )
                nll_sum += float(F.cross_entropy(logits, labels, reduction="sum"))
            predictions = logits.argmax(dim=1)
            raw_predictions = raw_union_logits(model, local_scores).argmax(dim=1)
            oracle_predictions = true_node_logits(model, local_scores, labels).argmax(dim=1)
            matches = predictions == labels
            correct += int(matches.sum())
            raw_correct += int((raw_predictions == labels).sum())
            oracle_correct += int((oracle_predictions == labels).sum())
            examples += len(labels)
            tasks = labels // 4
            for task in torch.unique(tasks).tolist():
                selected = tasks == task
                task_examples[task] += int(selected.sum())
                task_correct[task] += int(matches[selected].sum())
    return PrefixEvaluation(
        100.0 * correct / examples,
        nll_sum / examples,
        examples,
        100.0 * raw_correct / examples,
        100.0 * oracle_correct / examples,
        tuple(task_correct),
        tuple(task_examples),
        time.monotonic() - started,
    )


__all__ = [
    "EPOCH_LEDGER_SCHEMA",
    "PrefixEvaluation",
    "ReplayPopulation",
    "StageFit",
    "create_optimizer",
    "evaluate_prefix",
    "export_named_optimizer_state",
    "fit_online_stage",
    "restore_named_optimizer_state",
    "rotating_replay_population",
]

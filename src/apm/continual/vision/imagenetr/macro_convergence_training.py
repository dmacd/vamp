"""Resumable optimization and full histories for the macro-token convergence audit."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
import math
import time

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader

from apm.continual.artifacts import ChainedJsonlLedger, record_sha256
from apm.continual.vision.imagenetr.checkpoints import atomic_torch_save
from apm.continual.vision.imagenetr.config import TrainingConfig
from apm.continual.vision.imagenetr.data import ImageRecord, ManifestDataset
from apm.continual.vision.imagenetr.heads import ClassifierRows
from apm.continual.vision.imagenetr.lora import (
    adapter_factors,
    load_adapter_factors,
    trainable_lora_parameters,
)
from apm.continual.vision.imagenetr.macro_token_cache import (
    MacroTokenPopulation,
    MacroTokenShard,
)
from apm.continual.vision.imagenetr.macro_token_model import MacroTokenClassifier
from apm.continual.vision.imagenetr.macro_token_training import (
    PopulationMetrics,
    evaluate_population,
)
from apm.continual.vision.imagenetr.merging.common import LoRAFactors
from apm.continual.vision.imagenetr.model import (
    AdapterVisionModel,
    require_trainable_boundary,
)
from apm.continual.vision.imagenetr.training import (
    deterministic_epoch_order,
    os_cpu_workers,
)


MACRO_HISTORY_FORMAT = "imagenetr50-macro-convergence-epoch-v1"
JOINT_HISTORY_FORMAT = "imagenetr50-clean-joint-epoch-v1"


@dataclass(frozen=True, slots=True)
class MacroConvergenceCell:
    """One immutable macro-token optimizer schedule and initialization."""

    schedule: str
    effective_batch_size: int
    peak_learning_rate: float
    epochs: int
    seed: int
    depth: int = 1

    def __post_init__(self) -> None:
        if (
            self.schedule not in {"warmup_cosine", "legacy_constant"}
            or self.effective_batch_size not in {64, 128, 512}
            or self.peak_learning_rate <= 0.0
            or self.epochs < 1
            or self.seed < 0
            or self.depth != 1
        ):
            raise ValueError("macro convergence cell is invalid")

    @property
    def accumulation_steps(self) -> int:
        """Return the number of 64-image shards in one optimizer update."""
        return self.effective_batch_size // 64

    @property
    def condition(self) -> str:
        """Return a stable report and artifact identifier."""
        rate = f"{self.peak_learning_rate:.0e}".replace("-", "m")
        return f"{self.schedule}_b{self.effective_batch_size}_lr{rate}_s{self.seed}"

    def as_record(self) -> dict[str, object]:
        """Return canonical cell fields."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ConvergenceFit:
    """Best-checkpoint measurements and resource totals for one fit."""

    best_epoch: int
    epochs: int
    optimizer_steps: int
    best_optimizer_steps: int
    image_presentations: int
    train_nll: float
    train_accuracy: float
    validation_nll: float
    validation_accuracy: float
    peak_vram_bytes: int
    wall_seconds: float
    history_rows: int

    def __post_init__(self) -> None:
        if (
            not 1 <= self.best_epoch <= self.epochs
            or not 1 <= self.best_optimizer_steps <= self.optimizer_steps
            or self.image_presentations < 1
            or self.peak_vram_bytes < 0
            or self.history_rows != self.epochs
            or not all(
                math.isfinite(value)
                for value in (
                    self.train_nll,
                    self.train_accuracy,
                    self.validation_nll,
                    self.validation_accuracy,
                    self.wall_seconds,
                )
            )
            or not 0.0 <= self.train_accuracy <= 100.0
            or not 0.0 <= self.validation_accuracy <= 100.0
        ):
            raise ValueError("convergence fit is incomplete")

    def as_record(self) -> dict[str, object]:
        """Return canonical summary fields."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class JointConvergenceFit:
    """Fixed five-epoch joint result plus its diagnostic best validation epoch."""

    epochs: int
    optimizer_steps: int
    image_presentations: int
    fixed_train_nll: float
    fixed_train_accuracy: float
    fixed_validation_nll: float
    fixed_validation_accuracy: float
    best_epoch: int
    best_validation_nll: float
    best_validation_accuracy: float
    peak_vram_bytes: int
    wall_seconds: float
    history_rows: int

    def __post_init__(self) -> None:
        accuracies = (
            self.fixed_train_accuracy,
            self.fixed_validation_accuracy,
            self.best_validation_accuracy,
        )
        if (
            self.epochs != 5
            or self.optimizer_steps < 1
            or self.image_presentations < 1
            or not 1 <= self.best_epoch <= self.epochs
            or self.peak_vram_bytes < 0
            or self.history_rows != self.epochs
            or not all(
                math.isfinite(value)
                for value in (
                    self.fixed_train_nll,
                    self.fixed_validation_nll,
                    self.best_validation_nll,
                    self.wall_seconds,
                    *accuracies,
                )
            )
            or any(not 0.0 <= value <= 100.0 for value in accuracies)
        ):
            raise ValueError("clean joint fit is incomplete")

    def as_record(self) -> dict[str, object]:
        """Return canonical joint-control summary fields."""
        return asdict(self)


def convergence_learning_rate(
    cell: MacroConvergenceCell,
    step: int,
    total_steps: int,
    warmup_fraction: float,
    minimum_learning_rate_ratio: float,
) -> float:
    """Return the exact per-update constant or warmup-cosine learning rate."""
    if (
        not 0 <= step < total_steps
        or not 0.0 <= warmup_fraction < 1.0
        or not 0.0 < minimum_learning_rate_ratio <= 1.0
    ):
        raise ValueError("learning-rate schedule inputs are invalid")
    if cell.schedule == "legacy_constant":
        return cell.peak_learning_rate
    warmup_steps = max(1, round(total_steps * warmup_fraction))
    if step < warmup_steps:
        return cell.peak_learning_rate * (step + 1) / warmup_steps
    decay_steps = max(1, total_steps - warmup_steps - 1)
    progress = min(1.0, (step - warmup_steps) / decay_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    multiplier = minimum_learning_rate_ratio + (
        1.0 - minimum_learning_rate_ratio
    ) * cosine
    return cell.peak_learning_rate * multiplier


def _v8_derived_seed(seed: int, *parts: object) -> int:
    return int(
        record_sha256(
            {
                "parts": [str(part) for part in parts],
                "schema_version": "imagenetr50-macro-token-derived-seed-v1",
                "seed": seed,
            }
        )[:15],
        16,
    )


def _ordered_shards(
    population: MacroTokenPopulation,
    seed: int,
    epoch: int,
    shuffle_population_hash: str,
) -> tuple[MacroTokenShard, ...]:
    generator = torch.Generator().manual_seed(
        int(
            record_sha256(
                {
                    "epoch": epoch,
                    "population": shuffle_population_hash,
                    "seed": seed,
                    "schema_version": "imagenetr50-macro-token-shuffle-v1",
                }
            )[:15],
            16,
        )
    )
    order = torch.randperm(len(population.shards), generator=generator).tolist()
    return tuple(population.shards[index] for index in order)


def _restore_ledger_to_checkpoint(
    ledger: ChainedJsonlLedger, checkpoint: Mapping[str, object] | None
) -> None:
    retained_rows = 0 if checkpoint is None else int(checkpoint["history_rows"])
    if retained_rows > len(ledger.rows):
        raise ValueError("checkpoint refers beyond the durable history")
    ledger.truncate(retained_rows)


def _train_macro_epoch(
    model: MacroTokenClassifier,
    optimizer: torch.optim.AdamW,
    population: MacroTokenPopulation,
    cell: MacroConvergenceCell,
    epoch: int,
    optimizer_steps: int,
    total_steps: int,
    warmup_fraction: float,
    minimum_learning_rate_ratio: float,
    gradient_clip_norm: float,
    shuffle_population_hash: str,
    device: torch.device,
) -> tuple[int, float, float, float, float]:
    """Train one deterministic epoch and return steps, objective metrics, and norm."""
    model.train()
    ordered = _ordered_shards(
        population, cell.seed, epoch, shuffle_population_hash
    )
    examples = correct = 0
    nll_sum = gradient_norm_sum = 0.0
    device_ids = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=device_ids):
        torch.manual_seed(
            _v8_derived_seed(cell.seed, "macro_classifier", "dropout", epoch)
        )
        for offset in range(0, len(ordered), cell.accumulation_steps):
            window = ordered[offset : offset + cell.accumulation_steps]
            window_examples = sum(len(shard.rows) for shard in window)
            current_rate = convergence_learning_rate(
                cell,
                optimizer_steps,
                total_steps,
                warmup_fraction,
                minimum_learning_rate_ratio,
            )
            for group in optimizer.param_groups:
                group["lr"] = current_rate
            optimizer.zero_grad(set_to_none=True)
            for local_index, shard in enumerate(window):
                supervision = population.load(
                    shard,
                    _v8_derived_seed(
                        cell.seed, "epoch", epoch, "shard", offset + local_index
                    ),
                )
                moved = supervision.inputs.to(device)
                labels = supervision.labels.to(device, non_blocking=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    logits = model(moved)
                    batch_nll = F.cross_entropy(logits, labels, reduction="sum")
                    loss = batch_nll / window_examples
                loss.backward()
                nll_sum += float(batch_nll.detach())
                correct += int((logits.detach().argmax(dim=1) == labels).sum())
                examples += len(labels)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), gradient_clip_norm
            )
            gradient_norm_sum += float(gradient_norm.detach())
            optimizer.step()
            optimizer_steps += 1
    updates = math.ceil(len(ordered) / cell.accumulation_steps)
    return (
        optimizer_steps,
        nll_sum / examples,
        100.0 * correct / examples,
        gradient_norm_sum / updates,
        current_rate,
    )


def fit_macro_convergence_cell(
    *,
    model: MacroTokenClassifier,
    cell: MacroConvergenceCell,
    training: MacroTokenPopulation,
    validation: MacroTokenPopulation,
    dropout: float,
    weight_decay: float,
    gradient_clip_norm: float,
    warmup_fraction: float,
    minimum_learning_rate_ratio: float,
    shuffle_population_hash: str,
    checkpoint_path: Path,
    history_path: Path,
    job_hash: str,
    device: torch.device,
) -> ConvergenceFit:
    """Fit one complete schedule while persisting each epoch before continuing."""
    if dropout != 0.1 or training.partition != "fit" or validation.partition != "validation":
        raise ValueError("macro fit must use the frozen clean populations")
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cell.peak_learning_rate, weight_decay=weight_decay
    )
    ledger = ChainedJsonlLedger(history_path, MACRO_HISTORY_FORMAT)
    checkpoint = (
        torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint_path.is_file()
        else None
    )
    _restore_ledger_to_checkpoint(ledger, checkpoint)
    epoch = optimizer_steps = presentations = best_epoch = best_steps = 0
    elapsed_before = 0.0
    peak_before = 0
    best_nll = math.inf
    best_state: dict[str, Tensor] = {}
    if checkpoint is not None:
        if (
            checkpoint.get("schema_version")
            != "imagenetr50-macro-convergence-checkpoint-v1"
            or checkpoint.get("job_hash") != job_hash
        ):
            raise ValueError("macro convergence checkpoint identity changed")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch = int(checkpoint["epoch"])
        optimizer_steps = int(checkpoint["optimizer_steps"])
        presentations = int(checkpoint["image_presentations"])
        best_epoch = int(checkpoint["best_epoch"])
        best_steps = int(checkpoint["best_optimizer_steps"])
        best_nll = float(checkpoint["best_validation_nll"])
        best_state = dict(checkpoint["best_state"])
        elapsed_before = float(checkpoint["wall_seconds"])
        peak_before = int(checkpoint["peak_vram_bytes"])
    steps_per_epoch = math.ceil(len(training.shards) / cell.accumulation_steps)
    total_steps = steps_per_epoch * cell.epochs
    started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    from tqdm.auto import tqdm

    progress = tqdm(
        range(epoch + 1, cell.epochs + 1),
        desc=cell.condition,
        unit="epoch",
        initial=epoch,
        total=cell.epochs,
    )
    for current_epoch in progress:
        optimizer_steps, train_nll, train_accuracy, gradient_norm, final_rate = (
            _train_macro_epoch(
                model,
                optimizer,
                training,
                cell,
                current_epoch,
                optimizer_steps,
                total_steps,
                warmup_fraction,
                minimum_learning_rate_ratio,
                gradient_clip_norm,
                shuffle_population_hash,
                device,
            )
        )
        presentations += len(training.rows)
        validation_metrics = evaluate_population(
            model, validation, "macro_classifier", device
        )
        if validation_metrics.nll < best_nll:
            best_nll = validation_metrics.nll
            best_epoch = current_epoch
            best_steps = optimizer_steps
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        elapsed = elapsed_before + time.monotonic() - started
        peak = max(
            peak_before,
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        )
        ledger.append(
            {
                "condition": cell.condition,
                "epoch": current_epoch,
                "gradient_norm_mean": gradient_norm,
                "image_presentations": presentations,
                "learning_rate": final_rate,
                "optimizer_steps": optimizer_steps,
                "schema_version": MACRO_HISTORY_FORMAT,
                "train_objective_accuracy": train_accuracy,
                "train_objective_nll": train_nll,
                "validation_accuracy": validation_metrics.accuracy,
                "validation_examples": validation_metrics.examples,
                "validation_nll": validation_metrics.nll,
                "wall_seconds": elapsed,
            }
        )
        atomic_torch_save(
            checkpoint_path,
            {
                "best_epoch": best_epoch,
                "best_optimizer_steps": best_steps,
                "best_state": best_state,
                "best_validation_nll": best_nll,
                "epoch": current_epoch,
                "history_rows": len(ledger.rows),
                "image_presentations": presentations,
                "job_hash": job_hash,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "optimizer_steps": optimizer_steps,
                "peak_vram_bytes": peak,
                "schema_version": "imagenetr50-macro-convergence-checkpoint-v1",
                "wall_seconds": elapsed,
            },
        )
        progress.set_postfix(
            best_epoch=best_epoch,
            best_nll=f"{best_nll:.4f}",
            validation=f"{validation_metrics.accuracy:.2f}%",
        )
    progress.close()
    if not best_state:
        raise RuntimeError("macro convergence fit produced no finite checkpoint")
    model.load_state_dict(best_state, strict=True)
    train_metrics, validation_metrics = tuple(
        evaluate_population(model, population, "macro_classifier", device)
        for population in (training, validation)
    )
    final_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return ConvergenceFit(
        best_epoch,
        cell.epochs,
        optimizer_steps,
        best_steps,
        presentations,
        train_metrics.nll,
        train_metrics.accuracy,
        validation_metrics.nll,
        validation_metrics.accuracy,
        int(final_checkpoint["peak_vram_bytes"]),
        float(final_checkpoint["wall_seconds"]),
        len(ledger.rows),
    )


def evaluate_clean_joint(
    model: AdapterVisionModel,
    prepared_root: Path,
    rows: Sequence[ImageRecord],
    transform: object,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> PopulationMetrics:
    """Evaluate an adapter on clean training-derived identities only."""
    if not rows or any(row.split != "train" for row in rows):
        raise ValueError("clean joint evaluation cannot consume test rows")
    model.to(device).eval()
    loader = DataLoader(
        ManifestDataset(prepared_root, rows, transform, 0, 0),
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(num_workers, os_cpu_workers()),
        pin_memory=device.type == "cuda",
    )
    class_ids = torch.tensor(model.classifier.class_ids, dtype=torch.int64)
    nll_sum = 0.0
    correct = examples = 0
    task_correct: dict[int, int] = {}
    task_examples: dict[int, int] = {}
    with torch.inference_mode():
        for images, labels, _image_ids in loader:
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = model(images.to(device, non_blocking=True))
                nll_sum += float(F.cross_entropy(logits, labels, reduction="sum"))
            predictions = class_ids[logits.argmax(dim=1).cpu()]
            cpu_labels = labels.cpu()
            matches = predictions == cpu_labels
            correct += int(matches.sum())
            examples += len(labels)
            for task in torch.unique(cpu_labels // 4).tolist():
                selected = cpu_labels // 4 == task
                task_examples[task] = task_examples.get(task, 0) + int(selected.sum())
                task_correct[task] = task_correct.get(task, 0) + int(
                    (matches & selected).sum()
                )
    return PopulationMetrics(
        nll_sum / examples,
        100.0 * correct / examples,
        examples,
        tuple(
            (task, 100.0 * task_correct[task] / task_examples[task])
            for task in sorted(task_examples)
        ),
    )


def _joint_trainable_state(model: AdapterVisionModel) -> dict[str, object]:
    factors = adapter_factors(model)
    return {
        "adapter": {
            name: tensor.detach().cpu()
            for module, values in factors.items()
            for name, tensor in (
                (f"{module}.a", values.a),
                (f"{module}.b", values.b),
                (f"{module}.scale", torch.tensor(values.scale)),
            )
        },
        "classifier_bias": model.classifier.bias.detach().cpu(),
        "classifier_class_ids": list(model.classifier.class_ids),
        "classifier_weight": model.classifier.weight.detach().cpu(),
    }


def _load_joint_trainable_state(
    model: AdapterVisionModel, state: Mapping[str, object]
) -> None:
    raw_adapter = state["adapter"]
    if not isinstance(raw_adapter, Mapping):
        raise ValueError("clean joint adapter state is malformed")
    modules = tuple(
        sorted(name[: -len(".a")] for name in raw_adapter if name.endswith(".a"))
    )
    load_adapter_factors(
        model,
        {
            module: LoRAFactors(
                raw_adapter[f"{module}.a"],
                raw_adapter[f"{module}.b"],
                float(raw_adapter[f"{module}.scale"].item()),
            )
            for module in modules
        },
    )
    model.classifier.load_rows(
        ClassifierRows(
            tuple(int(value) for value in state["classifier_class_ids"]),
            state["classifier_weight"],
            state["classifier_bias"],
        )
    )


def fit_clean_joint_control(
    *,
    model: AdapterVisionModel,
    prepared_root: Path,
    training_rows: Sequence[ImageRecord],
    validation_rows: Sequence[ImageRecord],
    train_transform: object,
    evaluation_transform: object,
    config: TrainingConfig,
    training_seed: int,
    num_workers: int,
    checkpoint_path: Path,
    history_path: Path,
    job_hash: str,
    device: torch.device,
) -> tuple[JointConvergenceFit, PopulationMetrics, PopulationMetrics]:
    """Train and measure the exact clean-split five-epoch joint-IID control."""
    if (
        config.epochs != 5
        or config.batch_size != 64
        or not training_rows
        or not validation_rows
        or any(row.split != "train" for row in (*training_rows, *validation_rows))
        or set(row.image_id for row in training_rows)
        & set(row.image_id for row in validation_rows)
    ):
        raise ValueError("clean joint control differs from its frozen protocol")
    model.to(device)
    require_trainable_boundary(model)
    optimizer = torch.optim.SGD(
        (
            {"params": trainable_lora_parameters(model), "lr": config.lora_lr},
            {
                "params": (model.classifier.weight, model.classifier.bias),
                "lr": config.head_lr,
            },
        ),
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    ledger = ChainedJsonlLedger(history_path, JOINT_HISTORY_FORMAT)
    checkpoint = (
        torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint_path.is_file()
        else None
    )
    _restore_ledger_to_checkpoint(ledger, checkpoint)
    epoch = optimizer_steps = presentations = best_epoch = best_steps = 0
    elapsed_before = 0.0
    peak_before = 0
    best_nll = math.inf
    if checkpoint is not None:
        if (
            checkpoint.get("schema_version")
            != "imagenetr50-clean-joint-checkpoint-v1"
            or checkpoint.get("job_hash") != job_hash
        ):
            raise ValueError("clean joint checkpoint identity changed")
        _load_joint_trainable_state(model, checkpoint)
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch = int(checkpoint["epoch"])
        optimizer_steps = int(checkpoint["optimizer_steps"])
        presentations = int(checkpoint["image_presentations"])
        best_epoch = int(checkpoint["best_epoch"])
        best_steps = int(checkpoint["best_optimizer_steps"])
        best_nll = float(checkpoint["best_validation_nll"])
        elapsed_before = float(checkpoint["wall_seconds"])
        peak_before = int(checkpoint["peak_vram_bytes"])
    started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    from tqdm.auto import tqdm

    progress = tqdm(
        range(epoch, config.epochs),
        desc="clean_joint_iid",
        unit="epoch",
        initial=epoch,
        total=config.epochs,
    )
    final_train_objective_nll = final_train_objective_accuracy = math.nan
    for current_epoch in progress:
        order = deterministic_epoch_order(len(training_rows), training_seed, current_epoch)
        ordered_rows = tuple(training_rows[index] for index in order)
        loader = DataLoader(
            ManifestDataset(
                prepared_root,
                ordered_rows,
                train_transform,
                training_seed,
                current_epoch,
            ),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=min(num_workers, os_cpu_workers()),
            pin_memory=device.type == "cuda",
            persistent_workers=False,
        )
        model.train()
        train_nll_sum = 0.0
        train_correct = train_examples = 0
        for images, labels, _image_ids in loader:
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = model(images.to(device, non_blocking=True))
                loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            batch_examples = len(labels)
            train_nll_sum += float(loss.detach()) * batch_examples
            train_correct += int((logits.detach().argmax(dim=1) == labels).sum())
            train_examples += batch_examples
            optimizer_steps += 1
            presentations += batch_examples
        final_train_objective_nll = train_nll_sum / train_examples
        final_train_objective_accuracy = 100.0 * train_correct / train_examples
        validation_metrics = evaluate_clean_joint(
            model,
            prepared_root,
            validation_rows,
            evaluation_transform,
            config.batch_size,
            num_workers,
            device,
        )
        completed_epoch = current_epoch + 1
        if validation_metrics.nll < best_nll:
            best_nll = validation_metrics.nll
            best_epoch = completed_epoch
            best_steps = optimizer_steps
        elapsed = elapsed_before + time.monotonic() - started
        peak = max(
            peak_before,
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        )
        ledger.append(
            {
                "condition": "clean_joint_iid",
                "epoch": completed_epoch,
                "head_learning_rate": config.head_lr,
                "image_presentations": presentations,
                "lora_learning_rate": config.lora_lr,
                "optimizer_steps": optimizer_steps,
                "schema_version": JOINT_HISTORY_FORMAT,
                "train_objective_accuracy": final_train_objective_accuracy,
                "train_objective_nll": final_train_objective_nll,
                "validation_accuracy": validation_metrics.accuracy,
                "validation_examples": validation_metrics.examples,
                "validation_nll": validation_metrics.nll,
                "wall_seconds": elapsed,
            }
        )
        atomic_torch_save(
            checkpoint_path,
            {
                **_joint_trainable_state(model),
                "best_epoch": best_epoch,
                "best_optimizer_steps": best_steps,
                "best_validation_nll": best_nll,
                "epoch": completed_epoch,
                "history_rows": len(ledger.rows),
                "image_presentations": presentations,
                "job_hash": job_hash,
                "optimizer": optimizer.state_dict(),
                "optimizer_steps": optimizer_steps,
                "peak_vram_bytes": peak,
                "schema_version": "imagenetr50-clean-joint-checkpoint-v1",
                "wall_seconds": elapsed,
            },
        )
        progress.set_postfix(
            best_epoch=best_epoch,
            best_nll=f"{best_nll:.4f}",
            validation=f"{validation_metrics.accuracy:.2f}%",
        )
    progress.close()
    final_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    train_metrics, validation_metrics = tuple(
        evaluate_clean_joint(
            model,
            prepared_root,
            rows,
            evaluation_transform,
            config.batch_size,
            num_workers,
            device,
        )
        for rows in (training_rows, validation_rows)
    )
    best_row = min(ledger.rows, key=lambda row: float(row["validation_nll"]))
    fit = JointConvergenceFit(
        config.epochs,
        optimizer_steps,
        presentations,
        train_metrics.nll,
        train_metrics.accuracy,
        validation_metrics.nll,
        validation_metrics.accuracy,
        best_epoch,
        float(best_row["validation_nll"]),
        float(best_row["validation_accuracy"]),
        int(final_checkpoint["peak_vram_bytes"]),
        float(final_checkpoint["wall_seconds"]),
        len(ledger.rows),
    )
    return fit, train_metrics, validation_metrics


__all__ = [
    "ConvergenceFit",
    "JointConvergenceFit",
    "JOINT_HISTORY_FORMAT",
    "MACRO_HISTORY_FORMAT",
    "MacroConvergenceCell",
    "convergence_learning_rate",
    "evaluate_clean_joint",
    "fit_clean_joint_control",
    "fit_macro_convergence_cell",
]

"""Streaming optimization, evaluation, and persistence for macro-token heads."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
import math
import shutil
import time
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from apm.continual.artifacts import file_sha256, load_canonical_json, publish_immutable_json, record_sha256
from apm.continual.vision.imagenetr.artifacts import (
    publish_artifact_directory,
    validate_artifact_directory,
)
from apm.continual.vision.imagenetr.checkpoints import atomic_torch_save
from apm.continual.vision.imagenetr.integrator_model import ImageNetResidualIntegrator
from apm.continual.vision.imagenetr.macro_token_cache import (
    MacroTokenPopulation,
    MacroTokenShard,
)
from apm.continual.vision.imagenetr.macro_token_config import (
    MacroTokenConfig,
    MacroTokenOptimization,
)
from apm.continual.vision.imagenetr.macro_token_model import (
    CLASS_COUNT,
    MAXIMUM_SLOTS,
    MacroOwnerClassifier,
    MacroTokenClassifier,
    MacroTokenSupervision,
    behavior_control_features,
    parameter_count,
    predicted_owner_class_predictions,
)


MODEL_KINDS = frozenset({"macro_classifier", "owner_end_to_end", "v6_control"})


@dataclass(frozen=True, slots=True)
class MacroFitResult:
    """Selected epoch metrics and exact head-optimization resource accounting."""

    epochs: int
    best_epoch: int
    optimizer_steps: int
    train_nll: float
    train_accuracy: float
    validation_nll: float | None
    validation_accuracy: float | None
    image_presentations: int
    peak_vram_bytes: int
    wall_seconds: float
    converged: bool

    def __post_init__(self) -> None:
        optional = (self.validation_nll, self.validation_accuracy)
        if (
            self.epochs < 1
            or not 1 <= self.best_epoch <= self.epochs
            or self.optimizer_steps < 1
            or self.image_presentations < 1
            or self.peak_vram_bytes < 0
            or not all(
                math.isfinite(value)
                for value in (
                    self.train_nll,
                    self.train_accuracy,
                    self.wall_seconds,
                )
            )
            or any(value is not None and not math.isfinite(value) for value in optional)
            or (self.validation_nll is None) != (self.validation_accuracy is None)
        ):
            raise ValueError("macro-token fit result is incomplete")


@dataclass(frozen=True, slots=True)
class PopulationMetrics:
    """Streaming aggregate metrics without retaining per-image predictions."""

    nll: float
    accuracy: float
    examples: int
    task_accuracies: tuple[tuple[int, float], ...]
    owner_routed_accuracy: float | None = None

    def __post_init__(self) -> None:
        values = (self.nll, self.accuracy, *(value for _task, value in self.task_accuracies))
        if (
            self.examples < 1
            or not all(math.isfinite(value) for value in values)
            or any(not 0.0 <= value <= 100.0 for value in values[1:])
            or (
                self.owner_routed_accuracy is not None
                and (
                    not math.isfinite(self.owner_routed_accuracy)
                    or not 0.0 <= self.owner_routed_accuracy <= 100.0
                )
            )
        ):
            raise ValueError("population metrics are invalid")

    def as_record(self) -> dict[str, object]:
        """Return stable JSON fields with one row per observed task."""
        return {
            "accuracy": self.accuracy,
            "examples": self.examples,
            "nll": self.nll,
            "owner_routed_accuracy": self.owner_routed_accuracy,
            "task_accuracies": {
                str(task): value for task, value in self.task_accuracies
            },
        }


@dataclass(frozen=True, slots=True)
class MacroModelSpec:
    """One model construction and optimizer cell in the frozen matrix."""

    kind: str
    depth: int
    learning_rate: float
    seed: int

    def __post_init__(self) -> None:
        if (
            self.kind not in MODEL_KINDS
            or self.depth not in (1, 2)
            or self.learning_rate <= 0.0
            or self.seed < 0
        ):
            raise ValueError("macro-token model specification is invalid")

    @property
    def condition(self) -> str:
        """Return one report-safe condition identifier."""
        if self.kind == "v6_control":
            return "v6_final_cls_behavior_mlp"
        return f"macro_token_depth{self.depth}_{self.kind}"

    def as_record(self) -> dict[str, object]:
        """Return the construction fields used in job identities."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CompactControlPopulation:
    """In-memory v6 fields derived once from a full-token stage population."""

    identity: str
    features: Tensor
    baseline_logits: Tensor
    labels: Tensor
    seen_class_mask: Tensor

    def __post_init__(self) -> None:
        rows = len(self.labels)
        if (
            len(self.identity) != 64
            or self.features.shape != (rows, 8_214)
            or self.features.dtype != torch.bfloat16
            or self.baseline_logits.shape != (rows, CLASS_COUNT)
            or self.labels.shape != (rows,)
            or self.labels.dtype != torch.int64
            or self.seen_class_mask.shape != (CLASS_COUNT,)
            or self.seen_class_mask.dtype != torch.bool
            or not bool(torch.isfinite(self.features).all())
            or not bool(
                torch.isfinite(
                    self.baseline_logits[:, self.seen_class_mask]
                ).all()
            )
            or not bool(
                torch.isneginf(
                    self.baseline_logits[:, ~self.seen_class_mask]
                ).all()
            )
        ):
            raise ValueError("compact v6 control population is malformed")


def compact_control_population(
    population: MacroTokenPopulation,
) -> CompactControlPopulation:
    """Project full token shards to the exact 8,214-value v6 input once."""
    features: list[Tensor] = []
    baseline: list[Tensor] = []
    labels: list[Tensor] = []
    seen: Tensor | None = None
    for shard in population.shards:
        supervision = population.load(shard)
        shard_features, shard_baseline = behavior_control_features(
            supervision.inputs
        )
        features.append(shard_features.to(torch.bfloat16))
        baseline.append(shard_baseline)
        labels.append(supervision.labels)
        if seen is None:
            seen = supervision.inputs.seen_class_mask
        elif not torch.equal(seen, supervision.inputs.seen_class_mask):
            raise ValueError("control shards expose different seen-class masks")
    if seen is None:
        raise ValueError("cannot compact an empty control population")
    return CompactControlPopulation(
        record_sha256(
            {
                "source_population": population.identity,
                "schema_version": "imagenetr50-v6-compact-control-population-v1",
            }
        ),
        torch.cat(features),
        torch.cat(baseline),
        torch.cat(labels),
        seen,
    )


def _derived_seed(seed: int, *parts: object) -> int:
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


def create_trainable_model(
    spec: MacroModelSpec, config: MacroTokenConfig, device: torch.device
) -> nn.Module:
    """Construct one deterministic classifier, owner model, or v6 MLP control."""
    if spec.kind == "macro_classifier":
        return MacroTokenClassifier(
            spec.depth, config.macro_optimization.dropout, spec.seed
        ).to(device)
    if spec.kind == "owner_end_to_end":
        return MacroOwnerClassifier(
            spec.depth, config.macro_optimization.dropout, spec.seed
        ).to(device)
    device_ids = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=device_ids):
        torch.manual_seed(_derived_seed(spec.seed, "v6-control-initialization"))
        return ImageNetResidualIntegrator(
            MAXIMUM_SLOTS,
            1369,
            cast(tuple[int, int, int], tuple(config.control_hidden_widths)),
            config.control_dropout,
        ).to(device)


def _target(supervision: MacroTokenSupervision, kind: str) -> Tensor:
    return supervision.owner_targets if kind == "owner_end_to_end" else supervision.labels


def _logits(
    model: nn.Module,
    supervision: MacroTokenSupervision,
    kind: str,
    device: torch.device,
) -> tuple[Tensor, MacroTokenSupervision]:
    moved = MacroTokenSupervision(
        supervision.inputs.to(device),
        supervision.labels.to(device, non_blocking=True),
        supervision.owner_targets.to(device, non_blocking=True),
    )
    if kind == "v6_control":
        if not isinstance(model, ImageNetResidualIntegrator):
            raise TypeError("v6 control model has the wrong type")
        features, baseline = behavior_control_features(moved.inputs)
        return model(features, baseline, moved.inputs.seen_class_mask), moved
    if kind == "macro_classifier":
        if not isinstance(model, MacroTokenClassifier):
            raise TypeError("macro classifier has the wrong type")
        return model(moved.inputs), moved
    if not isinstance(model, MacroOwnerClassifier):
        raise TypeError("owner model has the wrong type")
    return model(moved.inputs), moved


def evaluate_population(
    model: nn.Module,
    population: MacroTokenPopulation,
    kind: str,
    device: torch.device,
) -> PopulationMetrics:
    """Evaluate one model by streaming cache shards exactly once."""
    if kind not in MODEL_KINDS:
        raise ValueError("unknown macro-token evaluation kind")
    model.eval()
    loss_sum = correct = routed_correct = examples = 0
    task_correct = [0] * 50
    task_total = [0] * 50
    with torch.inference_mode():
        for shard in population.shards:
            supervision = population.load(shard)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits, moved = _logits(model, supervision, kind, device)
                targets = _target(moved, kind)
                loss_sum += float(F.cross_entropy(logits, targets, reduction="sum"))
            predictions = logits.argmax(dim=1)
            correct += int((predictions == targets).sum())
            if kind == "owner_end_to_end":
                routed = predicted_owner_class_predictions(moved.inputs, predictions)
                routed_correct += int((routed == moved.labels).sum())
            class_predictions = predictions if kind != "owner_end_to_end" else None
            for task in torch.unique(moved.labels // 4).tolist():
                rows = moved.labels // 4 == task
                task_total[task] += int(rows.sum())
                if class_predictions is not None:
                    task_correct[task] += int(
                        (class_predictions[rows] == moved.labels[rows]).sum()
                    )
            examples += len(targets)
    task_accuracies = tuple(
        (task + 1, 100.0 * task_correct[task] / count)
        for task, count in enumerate(task_total)
        if count and kind != "owner_end_to_end"
    )
    return PopulationMetrics(
        loss_sum / examples,
        100.0 * correct / examples,
        examples,
        task_accuracies,
        100.0 * routed_correct / examples if kind == "owner_end_to_end" else None,
    )


def evaluate_frontier_controls(
    population: MacroTokenPopulation,
) -> dict[str, object]:
    """Stream raw-union and label-aware true-node accuracy from cached behaviors."""
    raw_correct = oracle_correct = examples = 0
    raw_task_correct = [0] * 50
    oracle_task_correct = [0] * 50
    task_total = [0] * 50
    for shard in population.shards:
        supervision = population.load(shard)
        _features, raw_union = behavior_control_features(supervision.inputs)
        raw_predictions = raw_union.argmax(dim=1)
        true_owners = supervision.owner_targets
        oracle_predictions = predicted_owner_class_predictions(
            supervision.inputs, true_owners
        )
        raw_correct += int((raw_predictions == supervision.labels).sum())
        oracle_correct += int((oracle_predictions == supervision.labels).sum())
        for task in torch.unique(supervision.labels // 4).tolist():
            rows = supervision.labels // 4 == task
            task_total[task] += int(rows.sum())
            raw_task_correct[task] += int(
                (raw_predictions[rows] == supervision.labels[rows]).sum()
            )
            oracle_task_correct[task] += int(
                (oracle_predictions[rows] == supervision.labels[rows]).sum()
            )
        examples += len(supervision.labels)
    return {
        "examples": examples,
        "raw_union_accuracy": 100.0 * raw_correct / examples,
        "raw_union_task_accuracies": {
            str(task + 1): 100.0 * raw_task_correct[task] / count
            for task, count in enumerate(task_total)
            if count
        },
        "true_node_oracle_accuracy": 100.0 * oracle_correct / examples,
        "true_node_oracle_task_accuracies": {
            str(task + 1): 100.0 * oracle_task_correct[task] / count
            for task, count in enumerate(task_total)
            if count
        },
    }


def evaluate_compact_control(
    model: ImageNetResidualIntegrator,
    population: CompactControlPopulation,
    batch_size: int,
    device: torch.device,
) -> PopulationMetrics:
    """Evaluate the v6 MLP without rereading discarded patch tokens."""
    model.eval()
    loss_sum = correct = 0
    task_correct = [0] * 50
    task_total = [0] * 50
    with torch.inference_mode():
        for offset in range(0, len(population.labels), batch_size):
            features = population.features[offset : offset + batch_size].to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            baseline = population.baseline_logits[offset : offset + batch_size].to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            labels = population.labels[offset : offset + batch_size].to(
                device, non_blocking=True
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = model(features, baseline, population.seen_class_mask.to(device))
                loss_sum += float(F.cross_entropy(logits, labels, reduction="sum"))
            predictions = logits.argmax(dim=1)
            correct += int((predictions == labels).sum())
            for task in torch.unique(labels // 4).tolist():
                rows = labels // 4 == task
                task_total[task] += int(rows.sum())
                task_correct[task] += int((predictions[rows] == labels[rows]).sum())
    examples = len(population.labels)
    return PopulationMetrics(
        loss_sum / examples,
        100.0 * correct / examples,
        examples,
        tuple(
            (task + 1, 100.0 * task_correct[task] / count)
            for task, count in enumerate(task_total)
            if count
        ),
    )


def fit_compact_control(
    *,
    model: ImageNetResidualIntegrator,
    spec: MacroModelSpec,
    training: CompactControlPopulation,
    validation: CompactControlPopulation | None,
    optimization: MacroTokenOptimization,
    batch_size: int,
    fixed_epochs: int | None,
    checkpoint_path: Path,
    checkpoint_key: str,
    device: torch.device,
    progress: bool = True,
) -> MacroFitResult:
    """Fit the v6 control on compact fields with its native 512-example batches."""
    if spec.kind != "v6_control" or (validation is None) == (fixed_epochs is None):
        raise ValueError("compact control fit has an invalid model or stopping mode")
    started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=spec.learning_rate, weight_decay=optimization.weight_decay
    )
    start_epoch = optimizer_steps = best_epoch = stale = 0
    best_nll = math.inf
    best_accuracy = 0.0
    best_state: dict[str, Tensor] = {}
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (
            type(checkpoint) is not dict
            or checkpoint.get("schema_version")
            != "imagenetr50-compact-control-checkpoint-v1"
            or checkpoint.get("checkpoint_key") != checkpoint_key
        ):
            raise ValueError("compact control checkpoint identity changed")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"])
        optimizer_steps = int(checkpoint["optimizer_steps"])
        best_epoch = int(checkpoint["best_epoch"])
        best_nll = float(checkpoint["best_nll"])
        best_accuracy = float(checkpoint["best_accuracy"])
        best_state = dict(checkpoint["best_state"])
        stale = int(checkpoint["stale"])
    maximum_epochs = fixed_epochs or optimization.maximum_epochs
    already_complete = start_epoch >= maximum_epochs or (
        validation is not None
        and start_epoch >= optimization.minimum_epochs
        and stale >= optimization.patience
    )
    from tqdm.auto import tqdm

    epochs = range(maximum_epochs + 1, maximum_epochs + 1) if already_complete else range(start_epoch + 1, maximum_epochs + 1)
    progress_bar = tqdm(
        epochs,
        desc=f"{spec.condition} seed {spec.seed}",
        unit="epoch",
        leave=False,
        disable=not progress,
    )
    epochs_ran = start_epoch
    for epoch in progress_bar:
        model.train()
        order = torch.randperm(
            len(training.labels),
            generator=torch.Generator().manual_seed(
                _derived_seed(spec.seed, "compact-control", epoch)
            ),
        )
        device_ids = [device.index or 0] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=device_ids):
            torch.manual_seed(_derived_seed(spec.seed, "control-dropout", epoch))
            for offset in range(0, len(order), batch_size):
                indices = order[offset : offset + batch_size]
                features = training.features[indices].to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                baseline = training.baseline_logits[indices].to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                labels = training.labels[indices].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    logits = model(
                        features, baseline, training.seen_class_mask.to(device)
                    )
                    loss = F.cross_entropy(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), optimization.gradient_clip_norm
                )
                optimizer.step()
                optimizer_steps += 1
        selected = (
            None
            if validation is None
            else evaluate_compact_control(model, validation, batch_size, device)
        )
        improved = bool(
            selected is not None
            and selected.nll <= best_nll - optimization.improvement_delta
        )
        if selected is None or selected.nll < best_nll:
            best_nll = 0.0 if selected is None else selected.nll
            best_accuracy = 0.0 if selected is None else selected.accuracy
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        stale = 0 if improved else stale + 1
        epochs_ran = epoch
        atomic_torch_save(
            checkpoint_path,
            {
                "best_accuracy": best_accuracy,
                "best_epoch": best_epoch,
                "best_nll": best_nll,
                "best_state": best_state,
                "checkpoint_key": checkpoint_key,
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "optimizer_steps": optimizer_steps,
                "schema_version": "imagenetr50-compact-control-checkpoint-v1",
                "stale": stale,
            },
        )
        progress_bar.set_postfix(
            **(
                {"epoch": epoch}
                if selected is None
                else {"best": f"{best_accuracy:.2f}%", "nll": f"{best_nll:.4f}"}
            )
        )
        if (
            selected is not None
            and epoch >= optimization.minimum_epochs
            and stale >= optimization.patience
        ):
            break
    if not best_state:
        raise RuntimeError("compact v6 control produced no finite checkpoint")
    model.load_state_dict(best_state, strict=True)
    train_metrics = evaluate_compact_control(model, training, batch_size, device)
    validation_metrics = (
        None
        if validation is None
        else evaluate_compact_control(model, validation, batch_size, device)
    )
    return MacroFitResult(
        epochs_ran,
        best_epoch,
        optimizer_steps,
        train_metrics.nll,
        train_metrics.accuracy,
        None if validation_metrics is None else validation_metrics.nll,
        None if validation_metrics is None else validation_metrics.accuracy,
        epochs_ran * len(training.labels),
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        time.monotonic() - started,
        validation is not None and epochs_ran < optimization.maximum_epochs,
    )


def _batch_seed(seed: int, epoch: int, shard_index: int) -> int:
    return _derived_seed(seed, "epoch", epoch, "shard", shard_index)


def _train_epoch(
    model: nn.Module,
    optimizer: torch.optim.AdamW,
    population: MacroTokenPopulation,
    spec: MacroModelSpec,
    optimization: MacroTokenOptimization,
    epoch: int,
    device: torch.device,
) -> int:
    model.train()
    shards = population.ordered_shards(spec.seed, epoch, True)
    steps = 0
    device_ids = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=device_ids):
        torch.manual_seed(_derived_seed(spec.seed, spec.kind, "dropout", epoch))
        for offset in range(0, len(shards), optimization.accumulation_steps):
            window = shards[offset : offset + optimization.accumulation_steps]
            window_examples = sum(len(shard.rows) for shard in window)
            optimizer.zero_grad(set_to_none=True)
            for local_index, shard in enumerate(window):
                supervision = population.load(
                    shard, _batch_seed(spec.seed, epoch, offset + local_index)
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    logits, moved = _logits(model, supervision, spec.kind, device)
                    loss = F.cross_entropy(
                        logits, _target(moved, spec.kind), reduction="sum"
                    ) / window_examples
                loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), optimization.gradient_clip_norm
            )
            optimizer.step()
            steps += 1
    return steps


def _checkpoint_record(
    *,
    model: nn.Module,
    optimizer: torch.optim.AdamW,
    epoch: int,
    optimizer_steps: int,
    best_epoch: int,
    best_nll: float,
    best_accuracy: float,
    best_state: Mapping[str, Tensor],
    stale: int,
    checkpoint_key: str,
) -> dict[str, object]:
    return {
        "best_accuracy": best_accuracy,
        "best_epoch": best_epoch,
        "best_nll": best_nll,
        "best_state": dict(best_state),
        "checkpoint_key": checkpoint_key,
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "optimizer_steps": optimizer_steps,
        "schema_version": "imagenetr50-macro-token-training-checkpoint-v1",
        "stale": stale,
    }


def fit_model(
    *,
    model: nn.Module,
    spec: MacroModelSpec,
    training: MacroTokenPopulation,
    validation: MacroTokenPopulation | None,
    optimization: MacroTokenOptimization,
    fixed_epochs: int | None,
    checkpoint_path: Path,
    checkpoint_key: str,
    device: torch.device,
    progress: bool = True,
) -> MacroFitResult:
    """Fit to validation-selected convergence or an exact clean-selected epoch."""
    if (validation is None) == (fixed_epochs is None):
        raise ValueError("fit must use either validation stopping or fixed refit epochs")
    if fixed_epochs is not None and fixed_epochs < 1:
        raise ValueError("fixed refit epochs must be positive")
    started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=spec.learning_rate, weight_decay=optimization.weight_decay
    )
    start_epoch = optimizer_steps = best_epoch = stale = 0
    best_nll = math.inf
    best_accuracy = 0.0
    best_state: dict[str, Tensor] = {}
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (
            type(checkpoint) is not dict
            or checkpoint.get("schema_version")
            != "imagenetr50-macro-token-training-checkpoint-v1"
            or checkpoint.get("checkpoint_key") != checkpoint_key
        ):
            raise ValueError("macro-token checkpoint identity changed")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"])
        optimizer_steps = int(checkpoint["optimizer_steps"])
        best_epoch = int(checkpoint["best_epoch"])
        best_nll = float(checkpoint["best_nll"])
        best_accuracy = float(checkpoint["best_accuracy"])
        best_state = dict(checkpoint["best_state"])
        stale = int(checkpoint["stale"])
    maximum_epochs = fixed_epochs or optimization.maximum_epochs
    already_complete = start_epoch >= maximum_epochs or (
        validation is not None
        and start_epoch >= optimization.minimum_epochs
        and stale >= optimization.patience
    )
    from tqdm.auto import tqdm

    epochs = range(maximum_epochs + 1, maximum_epochs + 1) if already_complete else range(start_epoch + 1, maximum_epochs + 1)
    progress_bar = tqdm(
        epochs,
        desc=f"{spec.condition} seed {spec.seed}",
        unit="epoch",
        leave=False,
        disable=not progress,
    )
    epochs_ran = start_epoch
    for epoch in progress_bar:
        optimizer_steps += _train_epoch(
            model, optimizer, training, spec, optimization, epoch, device
        )
        selection_metrics = (
            evaluate_population(model, validation, spec.kind, device)
            if validation is not None
            else None
        )
        improved = bool(
            selection_metrics is not None
            and selection_metrics.nll <= best_nll - optimization.improvement_delta
        )
        if validation is None or (
            selection_metrics is not None and selection_metrics.nll < best_nll
        ):
            best_nll = 0.0 if selection_metrics is None else selection_metrics.nll
            best_accuracy = (
                0.0 if selection_metrics is None else selection_metrics.accuracy
            )
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        stale = 0 if improved else stale + 1
        epochs_ran = epoch
        atomic_torch_save(
            checkpoint_path,
            _checkpoint_record(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                optimizer_steps=optimizer_steps,
                best_epoch=best_epoch,
                best_nll=best_nll,
                best_accuracy=best_accuracy,
                best_state=best_state,
                stale=stale,
                checkpoint_key=checkpoint_key,
            ),
        )
        progress_bar.set_postfix(
            **(
                {"epoch": epoch}
                if selection_metrics is None
                else {"best": f"{best_accuracy:.2f}%", "nll": f"{best_nll:.4f}"}
            )
        )
        if (
            validation is not None
            and epoch >= optimization.minimum_epochs
            and stale >= optimization.patience
        ):
            break
    if not best_state:
        raise RuntimeError("macro-token optimization produced no finite checkpoint")
    model.load_state_dict(best_state, strict=True)
    training_metrics = evaluate_population(model, training, spec.kind, device)
    validation_metrics = (
        evaluate_population(model, validation, spec.kind, device)
        if validation is not None
        else None
    )
    return MacroFitResult(
        epochs_ran,
        best_epoch,
        optimizer_steps,
        training_metrics.nll,
        training_metrics.accuracy,
        None if validation_metrics is None else validation_metrics.nll,
        None if validation_metrics is None else validation_metrics.accuracy,
        epochs_ran * len(training.rows),
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        time.monotonic() - started,
        validation is not None and epochs_ran < optimization.maximum_epochs,
    )


def publish_fitted_model(
    *,
    run_root: Path,
    family: str,
    job_hash: str,
    model: nn.Module,
    spec: MacroModelSpec,
    fit: MacroFitResult,
    metadata: Mapping[str, object],
) -> Path:
    """Publish one selected or fixed-epoch head as an immutable directory."""
    try:
        from safetensors.torch import save_file
    except ImportError as error:  # pragma: no cover - vision environment gate
        raise RuntimeError("safetensors is required by the vision environment") from error
    target = run_root / "models" / family / job_hash
    if target.is_dir():
        validate_artifact_directory(target)
        return target
    work = run_root / "work" / f"macro_model_{job_hash}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    model_path = work / "model.safetensors"
    save_file(
        {
            name: value.detach().cpu().contiguous()
            for name, value in sorted(model.state_dict().items())
        },
        model_path,
        metadata={
            "kind": spec.kind,
            "schema_version": "imagenetr50-macro-token-model-v1",
        },
    )
    publish_immutable_json(
        work / "fit.json",
        {
            **dict(metadata),
            "fit": asdict(fit),
            "model_sha256": file_sha256(model_path),
            "parameter_count": parameter_count(model),
            "schema_version": "imagenetr50-macro-token-fit-v1",
            "spec": spec.as_record(),
        },
    )
    publish_artifact_directory(work, target)
    shutil.rmtree(work)
    return target


def load_fitted_model(
    path: Path,
    factory: Callable[[], nn.Module],
    expected_spec: MacroModelSpec,
) -> tuple[nn.Module, MacroFitResult, dict[str, object]]:
    """Authenticate and load a fitted model with its immutable fit evidence."""
    try:
        from safetensors.torch import load_file
    except ImportError as error:  # pragma: no cover - vision environment gate
        raise RuntimeError("safetensors is required by the vision environment") from error
    validate_artifact_directory(path)
    record = load_canonical_json(path / "fit.json")
    model_path = path / "model.safetensors"
    if (
        record.get("schema_version") != "imagenetr50-macro-token-fit-v1"
        or record.get("spec") != expected_spec.as_record()
        or record.get("model_sha256") != file_sha256(model_path)
    ):
        raise ValueError("fitted macro-token model changed")
    model = factory()
    model.load_state_dict(load_file(model_path, device="cpu"), strict=True)
    return model, MacroFitResult(**dict(record["fit"])), record


def model_job_hash(
    *,
    protocol_hash: str,
    phase: str,
    stage: int,
    spec: MacroModelSpec,
    fit_population_hash: str,
    validation_population_hash: str | None,
    fixed_epochs: int | None,
    source_artifact_hash: str | None = None,
) -> str:
    """Bind one fit to its data, architecture, phase, and optional source model."""
    return record_sha256(
        {
            "fit_population_hash": fit_population_hash,
            "fixed_epochs": fixed_epochs,
            "phase": phase,
            "protocol_hash": protocol_hash,
            "schema_version": "imagenetr50-macro-token-model-job-v1",
            "source_artifact_hash": source_artifact_hash,
            "spec": spec.as_record(),
            "stage": stage,
            "validation_population_hash": validation_population_hash,
        }
    )


def _macro_representations(
    source: MacroTokenClassifier,
    population: MacroTokenPopulation,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    """Encode one population once for the frozen linear owner probe."""
    source.eval()
    features: list[Tensor] = []
    labels: list[Tensor] = []
    owner_targets: list[Tensor] = []
    with torch.inference_mode():
        for shard in population.shards:
            supervision = population.load(shard)
            moved = supervision.inputs.to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                features.append(source.encode(moved).cpu().to(torch.bfloat16))
            labels.append(supervision.labels)
            owner_targets.append(supervision.owner_targets)
    return torch.cat(features), torch.cat(labels), torch.cat(owner_targets)


def _probe_metrics(
    probe: nn.Linear,
    features: Tensor,
    targets: Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[float, float]:
    probe.eval()
    loss_sum = correct = 0
    with torch.inference_mode():
        for offset in range(0, len(targets), batch_size):
            selected_features = features[offset : offset + batch_size].to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            selected_targets = targets[offset : offset + batch_size].to(
                device, non_blocking=True
            )
            logits = probe(selected_features)
            loss_sum += float(F.cross_entropy(logits, selected_targets, reduction="sum"))
            correct += int((logits.argmax(dim=1) == selected_targets).sum())
    return loss_sum / len(targets), 100.0 * correct / len(targets)


def _new_owner_probe(seed: int, device: torch.device) -> nn.Linear:
    device_ids = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=device_ids):
        torch.manual_seed(_derived_seed(seed, "frozen-owner-probe"))
        probe = nn.Linear(768, MAXIMUM_SLOTS).to(device)
        nn.init.xavier_uniform_(probe.weight)
        nn.init.zeros_(probe.bias)
    return probe


def fit_frozen_owner_probe(
    *,
    source: MacroTokenClassifier,
    training: MacroTokenPopulation,
    validation: MacroTokenPopulation | None,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    minimum_epochs: int,
    maximum_epochs: int,
    patience: int,
    improvement_delta: float,
    fixed_epochs: int | None,
    seed: int,
    checkpoint_path: Path,
    checkpoint_key: str,
    device: torch.device,
) -> tuple[nn.Linear, MacroFitResult]:
    """Fit a diagnostic linear owner probe over a frozen macro-CLS representation."""
    if (validation is None) == (fixed_epochs is None):
        raise ValueError("owner probe needs validation stopping or fixed refit epochs")
    started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train_features, _train_labels, train_targets = _macro_representations(
        source, training, device
    )
    validation_data = (
        None
        if validation is None
        else _macro_representations(source, validation, device)
    )
    probe = _new_owner_probe(seed, device)
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    start_epoch = optimizer_steps = best_epoch = stale = 0
    best_nll = math.inf
    best_accuracy = 0.0
    best_state: dict[str, Tensor] = {}
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (
            type(checkpoint) is not dict
            or checkpoint.get("schema_version")
            != "imagenetr50-macro-token-probe-checkpoint-v1"
            or checkpoint.get("checkpoint_key") != checkpoint_key
        ):
            raise ValueError("frozen owner probe checkpoint identity changed")
        probe.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"])
        optimizer_steps = int(checkpoint["optimizer_steps"])
        best_epoch = int(checkpoint["best_epoch"])
        best_nll = float(checkpoint["best_nll"])
        best_accuracy = float(checkpoint["best_accuracy"])
        best_state = dict(checkpoint["best_state"])
        stale = int(checkpoint["stale"])
    target_epochs = fixed_epochs or maximum_epochs
    already_complete = start_epoch >= target_epochs or (
        validation_data is not None and start_epoch >= minimum_epochs and stale >= patience
    )
    from tqdm.auto import tqdm

    epochs = range(target_epochs + 1, target_epochs + 1) if already_complete else range(start_epoch + 1, target_epochs + 1)
    progress = tqdm(epochs, desc=f"frozen owner probe seed {seed}", unit="epoch", leave=False)
    epochs_ran = start_epoch
    for epoch in progress:
        probe.train()
        order = torch.randperm(
            len(train_targets),
            generator=torch.Generator().manual_seed(_derived_seed(seed, "probe", epoch)),
        )
        for offset in range(0, len(order), batch_size):
            indices = order[offset : offset + batch_size]
            batch_features = train_features[indices].to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            batch_targets = train_targets[indices].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(probe(batch_features), batch_targets)
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
        selected = (
            None
            if validation_data is None
            else _probe_metrics(
                probe,
                validation_data[0],
                validation_data[2],
                batch_size,
                device,
            )
        )
        improved = bool(selected is not None and selected[0] <= best_nll - improvement_delta)
        if selected is None or selected[0] < best_nll:
            best_nll = 0.0 if selected is None else selected[0]
            best_accuracy = 0.0 if selected is None else selected[1]
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in probe.state_dict().items()
            }
        stale = 0 if improved else stale + 1
        epochs_ran = epoch
        atomic_torch_save(
            checkpoint_path,
            {
                "best_accuracy": best_accuracy,
                "best_epoch": best_epoch,
                "best_nll": best_nll,
                "best_state": best_state,
                "checkpoint_key": checkpoint_key,
                "epoch": epoch,
                "model": probe.state_dict(),
                "optimizer": optimizer.state_dict(),
                "optimizer_steps": optimizer_steps,
                "schema_version": "imagenetr50-macro-token-probe-checkpoint-v1",
                "stale": stale,
            },
        )
        progress.set_postfix(
            **(
                {"epoch": epoch}
                if selected is None
                else {"best": f"{best_accuracy:.2f}%", "nll": f"{best_nll:.4f}"}
            )
        )
        if selected is not None and epoch >= minimum_epochs and stale >= patience:
            break
    if not best_state:
        raise RuntimeError("frozen owner probe produced no finite checkpoint")
    probe.load_state_dict(best_state, strict=True)
    train_nll, train_accuracy = _probe_metrics(
        probe, train_features, train_targets, batch_size, device
    )
    validation_metrics = (
        None
        if validation_data is None
        else _probe_metrics(
            probe,
            validation_data[0],
            validation_data[2],
            batch_size,
            device,
        )
    )
    return probe, MacroFitResult(
        epochs_ran,
        best_epoch,
        optimizer_steps,
        train_nll,
        train_accuracy,
        None if validation_metrics is None else validation_metrics[0],
        None if validation_metrics is None else validation_metrics[1],
        epochs_ran * len(train_targets),
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        time.monotonic() - started,
        validation_data is not None and epochs_ran < maximum_epochs,
    )


def evaluate_frozen_owner_probe(
    source: MacroTokenClassifier,
    probe: nn.Linear,
    population: MacroTokenPopulation,
    device: torch.device,
) -> PopulationMetrics:
    """Measure owner classification and predicted-owner routing for a frozen probe."""
    source.eval()
    probe.eval()
    loss_sum = owner_correct = routed_correct = examples = 0
    with torch.inference_mode():
        for shard in population.shards:
            supervision = population.load(shard)
            moved = MacroTokenSupervision(
                supervision.inputs.to(device),
                supervision.labels.to(device),
                supervision.owner_targets.to(device),
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                owner_logits = probe(source.encode(moved.inputs))
                owner_logits = owner_logits.masked_fill(
                    ~moved.inputs.active_slot_mask, -torch.inf
                )
                loss_sum += float(
                    F.cross_entropy(
                        owner_logits, moved.owner_targets, reduction="sum"
                    )
                )
            predictions = owner_logits.argmax(dim=1)
            owner_correct += int((predictions == moved.owner_targets).sum())
            routed = predicted_owner_class_predictions(moved.inputs, predictions)
            routed_correct += int((routed == moved.labels).sum())
            examples += len(predictions)
    return PopulationMetrics(
        loss_sum / examples,
        100.0 * owner_correct / examples,
        examples,
        (),
        100.0 * routed_correct / examples,
    )


def publish_frozen_owner_probe(
    *,
    run_root: Path,
    job_hash: str,
    probe: nn.Linear,
    fit: MacroFitResult,
    metadata: Mapping[str, object],
) -> Path:
    """Persist a frozen linear probe separately from its source representation."""
    try:
        from safetensors.torch import save_file
    except ImportError as error:  # pragma: no cover - vision environment gate
        raise RuntimeError("safetensors is required by the vision environment") from error
    target = run_root / "models" / "frozen_owner_probe" / job_hash
    if target.is_dir():
        validate_artifact_directory(target)
        return target
    work = run_root / "work" / f"owner_probe_{job_hash}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    model_path = work / "model.safetensors"
    save_file(
        {name: value.detach().cpu().contiguous() for name, value in probe.state_dict().items()},
        model_path,
        metadata={"schema_version": "imagenetr50-frozen-owner-probe-v1"},
    )
    publish_immutable_json(
        work / "fit.json",
        {
            **dict(metadata),
            "fit": asdict(fit),
            "model_sha256": file_sha256(model_path),
            "parameter_count": parameter_count(probe),
            "schema_version": "imagenetr50-frozen-owner-probe-fit-v1",
        },
    )
    publish_artifact_directory(work, target)
    shutil.rmtree(work)
    return target


def load_frozen_owner_probe(path: Path, device: torch.device) -> tuple[nn.Linear, MacroFitResult, dict[str, object]]:
    """Authenticate and load a diagnostic frozen linear owner probe."""
    try:
        from safetensors.torch import load_file
    except ImportError as error:  # pragma: no cover - vision environment gate
        raise RuntimeError("safetensors is required by the vision environment") from error
    validate_artifact_directory(path)
    record = load_canonical_json(path / "fit.json")
    model_path = path / "model.safetensors"
    if (
        record.get("schema_version") != "imagenetr50-frozen-owner-probe-fit-v1"
        or record.get("model_sha256") != file_sha256(model_path)
    ):
        raise ValueError("frozen owner probe artifact changed")
    probe = _new_owner_probe(int(record["seed"]), device)
    probe.load_state_dict(load_file(model_path, device="cpu"), strict=True)
    return probe, MacroFitResult(**dict(record["fit"])), record


__all__ = [
    "MODEL_KINDS",
    "CompactControlPopulation",
    "MacroFitResult",
    "MacroModelSpec",
    "PopulationMetrics",
    "create_trainable_model",
    "compact_control_population",
    "evaluate_compact_control",
    "evaluate_population",
    "evaluate_frozen_owner_probe",
    "evaluate_frontier_controls",
    "fit_frozen_owner_probe",
    "fit_compact_control",
    "fit_model",
    "load_fitted_model",
    "load_frozen_owner_probe",
    "model_job_hash",
    "publish_frozen_owner_probe",
    "publish_fitted_model",
]

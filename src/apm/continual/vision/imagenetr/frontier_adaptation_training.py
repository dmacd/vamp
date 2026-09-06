"""Differentiable frontier-node and macro-head training for ImageNet-R."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import islice
import math
from pathlib import Path
import time

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader

from apm.continual.artifacts import ChainedJsonlLedger, record_sha256
from apm.continual.vision.imagenetr.checkpoints import atomic_torch_save
from apm.continual.vision.imagenetr.data import ImageRecord, ManifestDataset
from apm.continual.vision.imagenetr.frontier_adaptation_config import (
    FrontierAdaptationConfig,
)
from apm.continual.vision.imagenetr.integrator_observations import BehaviorNode
from apm.continual.vision.imagenetr.lora import (
    adapter_factors,
    load_adapter_factors,
    trainable_lora_parameters,
)
from apm.continual.vision.imagenetr.macro_token_model import (
    CLASS_COUNT,
    MAXIMUM_SLOTS,
    TOKEN_DIMENSION,
    MacroTokenClassifier,
    behavior_meta_features,
)
from apm.continual.vision.imagenetr.macro_token_training import PopulationMetrics
from apm.continual.vision.imagenetr.merging.common import LoRAFactors
from apm.continual.vision.imagenetr.model import AdapterVisionModel
from apm.continual.vision.imagenetr.training import (
    deterministic_epoch_order,
    os_cpu_workers,
)


HISTORY_FORMAT = "imagenetr50-frontier-adaptation-epoch-v1"
REPLAY_NAMESPACE = "imagenetr50-stage31-nested-uniform-replay-v1"


@dataclass(frozen=True, slots=True)
class AdaptationCell:
    """One independently initialized replay-capacity and trainability condition."""

    historical_capacity: int
    adapt_lora: bool
    seed: int

    def __post_init__(self) -> None:
        if self.historical_capacity < 1 or self.seed < 0:
            raise ValueError("frontier adaptation cell is invalid")

    @property
    def condition(self) -> str:
        """Return the exact condition name shared by artifacts and plots."""
        return (
            f"frontier_lora_adapt_h{self.historical_capacity}"
            if self.adapt_lora
            else "frozen_frontier_full_fit_control"
        )

    def as_record(self) -> dict[str, object]:
        """Return canonical cell fields."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AdaptationFit:
    """Selected checkpoint, accuracy optimum, and measured training work."""

    best_nll_epoch: int
    best_validation_nll: float
    validation_accuracy_at_best_nll: float
    max_accuracy_epoch: int
    max_validation_accuracy: float
    validation_nll_at_max_accuracy: float
    epochs: int
    optimizer_steps: int
    best_optimizer_steps: int
    image_presentations: int
    train_nll_at_best: float
    train_accuracy_at_best: float
    peak_vram_bytes: int
    wall_seconds: float
    history_rows: int
    trainable_parameters: int

    def __post_init__(self) -> None:
        finite = (
            self.best_validation_nll,
            self.validation_accuracy_at_best_nll,
            self.max_validation_accuracy,
            self.validation_nll_at_max_accuracy,
            self.train_nll_at_best,
            self.train_accuracy_at_best,
            self.wall_seconds,
        )
        if (
            not 1 <= self.best_nll_epoch <= self.epochs
            or not 1 <= self.max_accuracy_epoch <= self.epochs
            or not 1 <= self.best_optimizer_steps <= self.optimizer_steps
            or self.image_presentations < 1
            or self.peak_vram_bytes < 0
            or self.history_rows != self.epochs
            or self.trainable_parameters < 1
            or not all(math.isfinite(value) for value in finite)
            or any(
                not 0.0 <= value <= 100.0
                for value in (
                    self.validation_accuracy_at_best_nll,
                    self.max_validation_accuracy,
                    self.train_accuracy_at_best,
                )
            )
        ):
            raise ValueError("frontier adaptation fit is incomplete")

    def as_record(self) -> dict[str, object]:
        """Return canonical fit fields."""
        return asdict(self)


def nested_replay_order(
    rows: Sequence[ImageRecord], seed: int
) -> tuple[ImageRecord, ...]:
    """Return one deterministic uniform ordering whose prefixes define every H."""
    if not rows or seed < 0 or any(row.split != "train" for row in rows):
        raise ValueError("nested replay requires nonempty training-derived rows")
    if len({row.image_id for row in rows}) != len(rows):
        raise ValueError("nested replay rows contain duplicate identities")
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                sha256(
                    f"{REPLAY_NAMESPACE}\0{seed}\0{row.image_id}".encode()
                ).hexdigest(),
                row.image_id,
            ),
        )
    )


def warmup_cosine_multiplier(
    step: int,
    total_steps: int,
    warmup_fraction: float,
    minimum_learning_rate_ratio: float,
) -> float:
    """Return a shared unit-peak warmup/cosine multiplier for both parameter groups."""
    if (
        not 0 <= step < total_steps
        or not 0.0 <= warmup_fraction < 1.0
        or not 0.0 < minimum_learning_rate_ratio <= 1.0
    ):
        raise ValueError("frontier learning-rate schedule inputs are invalid")
    warmup_steps = max(1, round(total_steps * warmup_fraction))
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    decay_steps = max(1, total_steps - warmup_steps - 1)
    progress = min(1.0, (step - warmup_steps) / decay_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_learning_rate_ratio + (
        1.0 - minimum_learning_rate_ratio
    ) * cosine


class AdaptiveFrontierModel(nn.Module):
    """Five sealed node models feeding one differentiable macro-token head."""

    def __init__(
        self,
        nodes: Sequence[BehaviorNode],
        slot_indices: Sequence[int],
        backbone_factory: Callable[[], nn.Module],
        rank: int,
        alpha: int,
        depth: int,
        dropout: float,
        seed: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        ordered = tuple(
            sorted(zip(nodes, slot_indices, strict=True), key=lambda pair: pair[1])
        )
        if (
            not ordered
            or len(ordered) > MAXIMUM_SLOTS
            or len({slot for _node, slot in ordered}) != len(ordered)
            or any(not 0 <= slot < MAXIMUM_SLOTS for _node, slot in ordered)
        ):
            raise ValueError("adaptive frontier nodes or slots are malformed")
        models = []
        for node, _slot in ordered:
            model = AdapterVisionModel(
                backbone_factory(),
                node.classifier.class_ids,
                rank,
                alpha,
                0.0,
                0,
                node.classifier,
            )
            load_adapter_factors(model, node.adapter)
            model.classifier.weight.requires_grad_(False)
            model.classifier.bias.requires_grad_(False)
            model.eval()
            models.append(model)
        self.node_models = nn.ModuleList(models)
        self.macro = MacroTokenClassifier(depth, dropout, seed)
        slots = torch.tensor(
            tuple(slot for _node, slot in ordered), dtype=torch.int64
        )
        ownership = torch.zeros((MAXIMUM_SLOTS, CLASS_COUNT), dtype=torch.bool)
        for node, slot in ordered:
            ownership[slot, torch.tensor(node.classifier.class_ids)] = True
        if int(ownership.sum()) != 4 * 31:
            raise ValueError("adaptive frontier must own exactly the 124 seen classes")
        self.register_buffer("slot_indices", slots, persistent=True)
        self.register_buffer("ownership", ownership, persistent=True)
        self.register_buffer("active_slot_mask", ownership.any(dim=1), persistent=True)
        self.register_buffer("seen_class_mask", ownership.any(dim=0), persistent=True)
        self.class_ids = tuple(node.classifier.class_ids for node, _slot in ordered)
        self.to(device)
        require_frontier_trainable_boundary(self, adapt_lora=True)

    @property
    def lora_parameters(self) -> tuple[nn.Parameter, ...]:
        """Return all and only LoRA factors from the live frontier nodes."""
        return tuple(
            parameter
            for node_model in self.node_models
            for parameter in trainable_lora_parameters(node_model)
        )

    def set_lora_trainable(self, enabled: bool) -> None:
        """Enable or freeze every node LoRA factor as one experimental boundary."""
        for parameter in self.lora_parameters:
            parameter.requires_grad_(enabled)
        require_frontier_trainable_boundary(self, adapt_lora=enabled)

    def set_training_mode(self) -> None:
        """Train only macro dropout while keeping deterministic node inference state."""
        self.macro.train()
        for node_model in self.node_models:
            node_model.eval()

    def set_evaluation_mode(self) -> None:
        """Disable macro dropout and retain evaluation mode in every node."""
        self.macro.eval()
        for node_model in self.node_models:
            node_model.eval()

    @staticmethod
    def _node_components(
        node_model: AdapterVisionModel, images: Tensor
    ) -> tuple[Tensor, Tensor]:
        tokens = node_model.token_sequence(images)
        features = node_model.backbone.forward_head(tokens, pre_logits=True)
        normalized = F.layer_norm(
            tokens.float(), (TOKEN_DIMENSION,), eps=1e-5
        ).to(tokens.dtype)
        return normalized, node_model.classifier(features).float()

    def forward(
        self, images: Tensor, adapt_lora: bool, activation_recomputation: bool
    ) -> Tensor:
        """Return task-free global logits while optionally adapting node LoRAs."""
        if adapt_lora and not all(
            parameter.requires_grad for parameter in self.lora_parameters
        ):
            raise ValueError("requested adaptation differs from the trainable boundary")
        components: list[tuple[Tensor, Tensor]] = []
        for node_model in self.node_models:
            if adapt_lora and activation_recomputation:
                values = checkpoint(
                    lambda current_images, selected=node_model: self._node_components(
                        selected, current_images
                    ),
                    images,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            elif adapt_lora:
                values = self._node_components(node_model, images)
            else:
                with torch.no_grad():
                    values = self._node_components(node_model, images)
            components.append(values)
        node_tokens = torch.stack(tuple(values[0] for values in components), dim=1)
        local_scores = tuple(values[1] for values in components)
        score_slots = []
        node_by_slot = {
            int(slot): index for index, slot in enumerate(self.slot_indices.tolist())
        }
        for slot in range(MAXIMUM_SLOTS):
            if slot not in node_by_slot:
                score_slots.append(
                    torch.zeros(
                        (len(images), CLASS_COUNT),
                        dtype=torch.float32,
                        device=images.device,
                    )
                )
                continue
            node_index = node_by_slot[slot]
            class_ids = torch.tensor(
                self.class_ids[node_index], dtype=torch.int64, device=images.device
            )
            score_slots.append(
                torch.zeros(
                    (len(images), CLASS_COUNT),
                    dtype=torch.float32,
                    device=images.device,
                ).index_copy(1, class_ids, local_scores[node_index])
            )
        raw_scores = torch.stack(tuple(score_slots), dim=1)
        meta_features = behavior_meta_features(
            raw_scores, self.ownership, self.active_slot_mask
        )
        return self.macro.forward_components(
            node_tokens,
            self.slot_indices,
            meta_features,
            self.seen_class_mask,
        )


def require_frontier_trainable_boundary(
    model: AdaptiveFrontierModel, adapt_lora: bool
) -> None:
    """Fail unless only the macro head and optionally node LoRAs can update."""
    expected_macro = {
        f"macro.{name}" for name, _parameter in model.macro.named_parameters()
    }
    expected_lora = {
        name
        for name, _parameter in model.named_parameters()
        if name.startswith("node_models.")
        and (name.endswith("lora_a") or name.endswith("lora_b"))
    }
    expected = expected_macro | (expected_lora if adapt_lora else set())
    actual = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if actual != expected or len(expected_lora) != 5 * 48:
        raise ValueError("frontier trainable parameters crossed the declared boundary")


def _trainable_state(model: AdaptiveFrontierModel) -> dict[str, object]:
    return {
        "macro": {
            name: value.detach().cpu().clone()
            for name, value in model.macro.state_dict().items()
        },
        "node_adapters": tuple(
            {
                module: LoRAFactors(
                    factors.a.detach().cpu().clone(),
                    factors.b.detach().cpu().clone(),
                    factors.scale,
                )
                for module, factors in adapter_factors(node_model).items()
            }
            for node_model in model.node_models
        ),
    }


def _load_trainable_state(
    model: AdaptiveFrontierModel, state: Mapping[str, object]
) -> None:
    macro = state.get("macro")
    adapters = state.get("node_adapters")
    if not isinstance(macro, Mapping) or not isinstance(adapters, Sequence):
        raise ValueError("frontier trainable checkpoint state is malformed")
    model.macro.load_state_dict(macro, strict=True)
    if len(adapters) != len(model.node_models):
        raise ValueError("frontier checkpoint node count changed")
    for node_model, raw_factors in zip(model.node_models, adapters, strict=True):
        if not isinstance(raw_factors, Mapping) or not all(
            isinstance(value, LoRAFactors) for value in raw_factors.values()
        ):
            raise ValueError("frontier checkpoint adapter factors are malformed")
        load_adapter_factors(node_model, raw_factors)


def _restore_ledger(
    ledger: ChainedJsonlLedger, checkpoint_record: Mapping[str, object] | None
) -> None:
    rows = 0 if checkpoint_record is None else int(checkpoint_record["history_rows"])
    if rows > len(ledger.rows):
        raise ValueError("frontier checkpoint refers beyond its durable history")
    ledger.truncate(rows)


def _batched_windows(
    loader: DataLoader, accumulation_steps: int
) -> Iterator[tuple[Sequence[object], ...]]:
    iterator = iter(loader)
    while window := tuple(islice(iterator, accumulation_steps)):
        yield window


def evaluate_frontier(
    model: AdaptiveFrontierModel,
    prepared_root: Path,
    rows: Sequence[ImageRecord],
    transform: object,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> PopulationMetrics:
    """Evaluate one attached or frozen frontier on training-derived identities."""
    if not rows or any(row.split != "train" for row in rows):
        raise ValueError("frontier evaluation may consume only training-derived rows")
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
                logits = model(
                    images.to(device, non_blocking=True),
                    adapt_lora=False,
                    activation_recomputation=False,
                )
                nll_sum += float(F.cross_entropy(logits, labels, reduction="sum"))
            predictions = logits.argmax(dim=1)
            matches = predictions == labels
            correct += int(matches.sum())
            examples += len(labels)
            for task in torch.unique(labels // 4).tolist():
                selected = labels // 4 == task
                task_examples[task] = task_examples.get(task, 0) + int(selected.sum())
                task_correct[task] = task_correct.get(task, 0) + int(
                    matches[selected].sum()
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


def adapter_displacements(
    model: AdaptiveFrontierModel, nodes: Sequence[BehaviorNode]
) -> tuple[dict[str, object], ...]:
    """Measure scale-aware dense LoRA-update movement from every sealed node."""
    if len(nodes) != len(model.node_models):
        raise ValueError("source and adapted frontier node counts differ")
    rows = []
    for index, (node_model, source) in enumerate(
        zip(model.node_models, nodes, strict=True)
    ):
        current = adapter_factors(node_model)
        squared_change = squared_source = 0.0
        for module in sorted(current):
            adapted_update = current[module].dense().float()
            source_update = source.adapter[module].dense().to(adapted_update).float()
            squared_change += float(torch.sum((adapted_update - source_update) ** 2))
            squared_source += float(torch.sum(source_update**2))
        change = math.sqrt(squared_change)
        source_norm = math.sqrt(squared_source)
        rows.append(
            {
                "dense_update_change_frobenius": change,
                "dense_update_relative_change": change / max(source_norm, 1e-12),
                "dense_update_source_frobenius": source_norm,
                "level": source.level,
                "node_hash": source.node_hash,
                "node_index": index,
                "represented_task_ids": list(source.represented_task_ids),
            }
        )
    return tuple(rows)


def fit_adaptation_cell(
    *,
    model: AdaptiveFrontierModel,
    nodes: Sequence[BehaviorNode],
    cell: AdaptationCell,
    prepared_root: Path,
    training_rows: Sequence[ImageRecord],
    validation_rows: Sequence[ImageRecord],
    train_transform: object,
    evaluation_transform: object,
    config: FrontierAdaptationConfig,
    checkpoint_path: Path,
    history_path: Path,
    job_hash: str,
    device: torch.device,
) -> tuple[
    AdaptationFit,
    PopulationMetrics,
    PopulationMetrics,
    tuple[dict[str, object], ...],
]:
    """Fit or exactly resume one nested-H cell and restore its minimum-NLL state."""
    if (
        len(training_rows) != cell.historical_capacity
        or len(validation_rows) != 3049
        or any(row.split != "train" for row in (*training_rows, *validation_rows))
        or set(row.image_id for row in training_rows)
        & set(row.image_id for row in validation_rows)
    ):
        raise ValueError("frontier adaptation populations differ from the clean protocol")
    model.set_lora_trainable(cell.adapt_lora)
    macro_parameters = tuple(model.macro.parameters())
    lora_parameters = model.lora_parameters if cell.adapt_lora else ()
    optimizer = torch.optim.AdamW(
        tuple(
            {"params": parameters, "lr": learning_rate, "name": name}
            for parameters, learning_rate, name in (
                (macro_parameters, config.macro_peak_learning_rate, "macro"),
                (lora_parameters, config.lora_peak_learning_rate, "lora"),
            )
            if parameters
        ),
        weight_decay=config.weight_decay,
    )
    ledger = ChainedJsonlLedger(history_path, HISTORY_FORMAT)
    saved = (
        torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint_path.is_file()
        else None
    )
    _restore_ledger(ledger, saved)
    epoch = optimizer_steps = presentations = best_epoch = best_steps = 0
    best_nll = math.inf
    best_state: dict[str, object] = {}
    elapsed_before = 0.0
    peak_before = 0
    if saved is not None:
        if (
            saved.get("schema_version")
            != "imagenetr50-frontier-adaptation-checkpoint-v1"
            or saved.get("job_hash") != job_hash
        ):
            raise ValueError("frontier adaptation checkpoint identity changed")
        raw_state = saved.get("trainable_state")
        if not isinstance(raw_state, Mapping):
            raise ValueError("frontier checkpoint lacks trainable state")
        _load_trainable_state(model, raw_state)
        optimizer.load_state_dict(saved["optimizer"])
        epoch = int(saved["epoch"])
        optimizer_steps = int(saved["optimizer_steps"])
        presentations = int(saved["image_presentations"])
        best_epoch = int(saved["best_nll_epoch"])
        best_steps = int(saved["best_optimizer_steps"])
        best_nll = float(saved["best_validation_nll"])
        best_state = dict(saved["best_state"])
        elapsed_before = float(saved["wall_seconds"])
        peak_before = int(saved["peak_vram_bytes"])
    accumulation_steps = config.effective_batch_size // config.microbatch_size
    steps_per_epoch = math.ceil(
        math.ceil(len(training_rows) / config.microbatch_size) / accumulation_steps
    )
    total_steps = steps_per_epoch * config.epochs
    started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    from tqdm.auto import tqdm

    progress = tqdm(
        range(epoch + 1, config.epochs + 1),
        desc=cell.condition,
        unit="epoch",
        initial=epoch,
        total=config.epochs,
    )
    for current_epoch in progress:
        order = deterministic_epoch_order(
            len(training_rows),
            cell.seed + cell.historical_capacity,
            current_epoch - 1,
        )
        ordered_rows = tuple(training_rows[index] for index in order)
        loader = DataLoader(
            ManifestDataset(
                prepared_root,
                ordered_rows,
                train_transform,
                cell.seed,
                current_epoch - 1,
            ),
            batch_size=config.microbatch_size,
            shuffle=False,
            num_workers=min(config.num_workers, os_cpu_workers()),
            pin_memory=device.type == "cuda",
            persistent_workers=False,
        )
        model.set_training_mode()
        train_nll_sum = gradient_norm_sum = 0.0
        train_correct = train_examples = updates = 0
        macro_rate = lora_rate = 0.0
        device_ids = [device.index or 0] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=device_ids):
            torch.manual_seed(
                int(
                    record_sha256(
                        {
                            "condition": cell.condition,
                            "epoch": current_epoch,
                            "schema_version": "imagenetr50-frontier-dropout-seed-v1",
                            "seed": cell.seed,
                        }
                    )[:15],
                    16,
                )
            )
            for window in _batched_windows(loader, accumulation_steps):
                window_examples = sum(len(batch[1]) for batch in window)
                multiplier = warmup_cosine_multiplier(
                    optimizer_steps,
                    total_steps,
                    config.warmup_fraction,
                    config.minimum_learning_rate_ratio,
                )
                macro_rate = config.macro_peak_learning_rate * multiplier
                lora_rate = config.lora_peak_learning_rate * multiplier
                for group in optimizer.param_groups:
                    group["lr"] = (
                        macro_rate if group["name"] == "macro" else lora_rate
                    )
                optimizer.zero_grad(set_to_none=True)
                for images, labels, _image_ids in window:
                    labels = labels.to(device, non_blocking=True)
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=device.type == "cuda",
                    ):
                        logits = model(
                            images.to(device, non_blocking=True),
                            adapt_lora=cell.adapt_lora,
                            activation_recomputation=(
                                config.activation_recomputation
                                and cell.adapt_lora
                            ),
                        )
                        batch_nll = F.cross_entropy(
                            logits, labels, reduction="sum"
                        )
                        loss = batch_nll / window_examples
                    loss.backward()
                    train_nll_sum += float(batch_nll.detach())
                    train_correct += int(
                        (logits.detach().argmax(dim=1) == labels).sum()
                    )
                    train_examples += len(labels)
                trainables = macro_parameters + lora_parameters
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    trainables, config.gradient_clip_norm
                )
                gradient_norm_sum += float(gradient_norm.detach())
                optimizer.step()
                optimizer_steps += 1
                updates += 1
        presentations += len(training_rows)
        validation_metrics = evaluate_frontier(
            model,
            prepared_root,
            validation_rows,
            evaluation_transform,
            config.evaluation_batch_size,
            config.num_workers,
            device,
        )
        if validation_metrics.nll < best_nll:
            best_nll = validation_metrics.nll
            best_epoch = current_epoch
            best_steps = optimizer_steps
            best_state = _trainable_state(model)
        elapsed = elapsed_before + time.monotonic() - started
        peak = max(
            peak_before,
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        )
        ledger.append(
            {
                "adapt_lora": cell.adapt_lora,
                "condition": cell.condition,
                "epoch": current_epoch,
                "gradient_norm_mean": gradient_norm_sum / updates,
                "historical_capacity": cell.historical_capacity,
                "image_presentations": presentations,
                "lora_learning_rate": lora_rate if cell.adapt_lora else None,
                "macro_learning_rate": macro_rate,
                "optimizer_steps": optimizer_steps,
                "schema_version": HISTORY_FORMAT,
                "train_objective_accuracy": 100.0
                * train_correct
                / train_examples,
                "train_objective_nll": train_nll_sum / train_examples,
                "validation_accuracy": validation_metrics.accuracy,
                "validation_examples": validation_metrics.examples,
                "validation_nll": validation_metrics.nll,
                "wall_seconds": elapsed,
            }
        )
        atomic_torch_save(
            checkpoint_path,
            {
                "best_nll_epoch": best_epoch,
                "best_optimizer_steps": best_steps,
                "best_state": best_state,
                "best_validation_nll": best_nll,
                "epoch": current_epoch,
                "history_rows": len(ledger.rows),
                "image_presentations": presentations,
                "job_hash": job_hash,
                "optimizer": optimizer.state_dict(),
                "optimizer_steps": optimizer_steps,
                "peak_vram_bytes": peak,
                "schema_version": "imagenetr50-frontier-adaptation-checkpoint-v1",
                "trainable_state": _trainable_state(model),
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
        raise RuntimeError("frontier adaptation produced no finite checkpoint")
    _load_trainable_state(model, best_state)
    model.set_lora_trainable(cell.adapt_lora)
    train_metrics, validation_metrics = tuple(
        evaluate_frontier(
            model,
            prepared_root,
            rows,
            evaluation_transform,
            config.evaluation_batch_size,
            config.num_workers,
            device,
        )
        for rows in (training_rows, validation_rows)
    )
    maximum_accuracy = max(
        ledger.rows,
        key=lambda row: (float(row["validation_accuracy"]), -int(row["epoch"])),
    )
    final_checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    fit = AdaptationFit(
        best_epoch,
        validation_metrics.nll,
        validation_metrics.accuracy,
        int(maximum_accuracy["epoch"]),
        float(maximum_accuracy["validation_accuracy"]),
        float(maximum_accuracy["validation_nll"]),
        config.epochs,
        optimizer_steps,
        best_steps,
        presentations,
        train_metrics.nll,
        train_metrics.accuracy,
        int(final_checkpoint["peak_vram_bytes"]),
        float(final_checkpoint["wall_seconds"]),
        len(ledger.rows),
        sum(parameter.numel() for parameter in macro_parameters + lora_parameters),
    )
    return fit, train_metrics, validation_metrics, adapter_displacements(model, nodes)


__all__ = [
    "AdaptationCell",
    "AdaptationFit",
    "AdaptiveFrontierModel",
    "HISTORY_FORMAT",
    "REPLAY_NAMESPACE",
    "adapter_displacements",
    "evaluate_frontier",
    "fit_adaptation_cell",
    "nested_replay_order",
    "require_frontier_trainable_boundary",
    "warmup_cosine_multiplier",
]

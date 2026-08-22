"""Deterministic flat, causal-leaf, full-replay, and repair router fitting."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from collections.abc import Mapping, Sequence
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from apm.continual.artifacts import record_sha256
from apm.continual.vision.imagenetr.checkpoints import atomic_torch_save
from apm.continual.vision.imagenetr.router_config import RouterTrainingConfig
from apm.continual.vision.imagenetr.router_features import RouterFeatureUniverse
from apm.continual.vision.imagenetr.router_protocol import RouterSplit
from apm.continual.vision.imagenetr.router_scores import (
    RouterQuery,
    ScoringNode,
    move_scorer,
    scorer_state_hash,
    score_nodes,
)
from apm.continual.vision.imagenetr.router_teacher import (
    ImageNetRouterTeacher,
    require_class_partition,
)


@dataclass(frozen=True, slots=True)
class RouterTrainingResult:
    """Bounded optimizer work and validation-selected state evidence."""

    epochs: int
    optimizer_steps: int
    best_validation_loss: float
    training_image_ids: tuple[str, ...]
    validation_image_ids: tuple[str, ...]

    @property
    def training_ids_hash(self) -> str:
        return record_sha256(list(self.training_image_ids))

    def as_record(self) -> dict[str, object]:
        return {
            "best_validation_loss": self.best_validation_loss,
            "epochs": self.epochs,
            "optimizer_steps": self.optimizer_steps,
            "schema_version": "imagenetr50-router-training-result-v1",
            "training_ids_hash": self.training_ids_hash,
            "training_image_count": len(self.training_image_ids),
            "validation_image_count": len(self.validation_image_ids),
        }


class RouterTrainingData:
    """Leakage-checked views over the frozen training-feature universe."""

    def __init__(self, universe: RouterFeatureUniverse, split: RouterSplit) -> None:
        if set(universe.image_ids) != set(split.fit_image_ids) | set(
            split.validation_image_ids
        ):
            raise ValueError("router split does not partition the training feature universe")
        self.universe = universe
        self.split = split
        self._index = universe.index
        self._fit = frozenset(split.fit_image_ids)
        self._validation = frozenset(split.validation_image_ids)

    def ids(self, partition: str, stage: int, tasks: Sequence[int] | None = None) -> tuple[str, ...]:
        """Return stable rows available in one split and historical task view."""
        if partition not in {"fit", "validation"} or not 1 <= stage <= 50:
            raise ValueError("invalid router training partition or stage")
        allowed_ids = self._fit if partition == "fit" else self._validation
        allowed_tasks = set(range(stage)) if tasks is None else set(tasks)
        if not allowed_tasks <= set(range(stage)):
            raise ValueError("router training view requests future tasks")
        return tuple(
            image_id
            for image_id, task in zip(
                self.universe.image_ids, self.universe.task_ids.tolist()
            )
            if image_id in allowed_ids and task in allowed_tasks
        )

    def batch(
        self, image_ids: Sequence[str], device: torch.device
    ) -> tuple[RouterQuery, Tensor, Tensor]:
        """Return task-free inputs separately from teacher-only labels/tasks."""
        if len(set(image_ids)) != len(image_ids) or any(value not in self._index for value in image_ids):
            raise ValueError("router batch contains unknown or duplicate image IDs")
        indices = torch.tensor([self._index[value] for value in image_ids], dtype=torch.long)
        query = RouterQuery(
            tuple(image_ids),
            self.universe.prelogits[indices],
            {name: values[indices] for name, values in self.universe.cls_activations.items()},
        ).to(device)
        return (
            query,
            self.universe.labels[indices].to(device),
            self.universe.task_ids[indices].to(device),
        )

    def require_fit(self, image_ids: Sequence[str], stage: int) -> None:
        """Reject validation, test, future, unknown, or duplicate optimizer inputs."""
        supplied = tuple(image_ids)
        if len(set(supplied)) != len(supplied) or not set(supplied) <= self._fit:
            raise ValueError("router optimizer inputs are not exclusively router-fit identities")
        tasks = {int(self.universe.task_ids[self._index[value]]) for value in supplied}
        if not tasks <= set(range(stage)):
            raise ValueError("router optimizer inputs contain future tasks")


def _seed(namespace: str, seed: int, epoch: int) -> int:
    return int(
        sha256(f"imagenetr50-router-order-v1\0{namespace}\0{seed}\0{epoch}".encode()).hexdigest()[:16],
        16,
    )


def _batches(
    image_ids: Sequence[str],
    batch_size: int,
    namespace: str,
    seed: int,
    epoch: int,
) -> tuple[tuple[str, ...], ...]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_seed(namespace, seed, epoch))
    order = torch.randperm(len(image_ids), generator=generator).tolist()
    return tuple(
        tuple(image_ids[index] for index in order[offset : offset + batch_size])
        for offset in range(0, len(order), batch_size)
    )


def deterministic_reservoir(
    image_ids: Sequence[str], count: int, namespace: str
) -> tuple[str, ...]:
    """Return an order-independent bottom-hash reservoir."""
    if count < 0 or not namespace or len(set(image_ids)) != len(image_ids):
        raise ValueError("invalid deterministic router reservoir request")
    return tuple(
        sorted(
            image_ids,
            key=lambda value: (sha256(f"{namespace}\0{value}".encode()).hexdigest(), value),
        )[:count]
    )


def negative_reservoirs(
    data: RouterTrainingData,
    nodes: Sequence[ScoringNode],
    stage: int,
    count: int,
    namespace: str,
) -> tuple[str, ...]:
    """Select exactly bounded fit-only negatives from every represented live node."""
    result = []
    for node in nodes:
        candidates = data.ids("fit", stage, node.represented_task_ids)
        result.extend(
            deterministic_reservoir(
                candidates,
                min(count, len(candidates)),
                f"{namespace}:{node.node_id}",
            )
        )
    if len(set(result)) != len(result):
        raise ValueError("live-node negative reservoirs overlap")
    return tuple(result)


def repair_reservoir(
    data: RouterTrainingData,
    represented_tasks: Sequence[int],
    stage: int,
    fraction: float,
    namespace: str,
) -> tuple[str, ...]:
    """Select exactly ceil(f*N) represented router-fit positives."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("router repair fraction must be in (0, 1]")
    candidates = data.ids("fit", stage, represented_tasks)
    return deterministic_reservoir(
        candidates, math.ceil(fraction * len(candidates)), namespace
    )


def _live_classes(nodes: Sequence[ScoringNode]) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple((node.node_id, node.represented_class_ids) for node in nodes)


def _target_indices(
    teacher: ImageNetRouterTeacher,
    stage: int,
    query: RouterQuery,
    labels: Tensor,
    task_ids: Tensor,
    nodes: Sequence[ScoringNode],
) -> Tensor:
    classes = _live_classes(nodes)
    require_class_partition(stage, classes)
    return teacher.target_indices(
        stage,
        query.image_ids,
        labels.detach().cpu(),
        task_ids.detach().cpu(),
        classes,
    ).to(labels.device)


def _state(modules: Sequence[nn.Module]) -> tuple[dict[str, Tensor], ...]:
    return tuple(
        {
            key: value.detach().to(device="cpu").clone()
            for key, value in module.state_dict().items()
        }
        for module in modules
    )


def _restore(modules: Sequence[nn.Module], state: Sequence[Mapping[str, Tensor]]) -> None:
    for module, values in zip(modules, state):
        module.load_state_dict(dict(values), strict=True)


@dataclass(frozen=True, slots=True)
class _ResumeState:
    next_epoch: int
    steps: int
    best_loss: float
    stale: int
    best: tuple[dict[str, Tensor], ...]
    completed: bool


def _checkpoint_identity(
    kind: str,
    namespace: str,
    stage: int,
    seed: int,
    training_ids: Sequence[str],
    validation_ids: Sequence[str],
    training: RouterTrainingConfig,
    dependencies: Mapping[str, object],
) -> str:
    return record_sha256(
        {
            "kind": kind,
            "namespace": namespace,
            "dependencies": dict(dependencies),
            "router_seed": seed,
            "schema_version": "imagenetr50-router-training-checkpoint-identity-v1",
            "stage": stage,
            "training": {
                "batch_size": training.batch_size,
                "lr": training.lr,
                "lse_weight": training.lse_weight,
                "margin": training.margin,
                "max_epochs": training.max_epochs,
                "negatives_per_live_node": training.negatives_per_live_node,
                "patience": training.patience,
                "weight_decay": training.weight_decay,
            },
            "training_ids_hash": record_sha256(list(training_ids)),
            "validation_ids_hash": record_sha256(list(validation_ids)),
        }
    )


def _optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for values in optimizer.state.values():
        for key, value in values.items():
            if isinstance(value, Tensor):
                values[key] = value.to(device)


def _resume(
    checkpoint_path: str | Path | None,
    identity: str,
    modules: Sequence[nn.Module],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> _ResumeState:
    initial = _ResumeState(0, 0, math.inf, 0, _state(modules), False)
    if checkpoint_path is None or not Path(checkpoint_path).is_file():
        return initial
    record = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    if (
        type(record) is not dict
        or record.get("schema_version")
        != "imagenetr50-router-training-checkpoint-v1"
        or record.get("identity") != identity
    ):
        raise ValueError("router training checkpoint identity changed")
    current = tuple(record["module_state"])
    best = tuple(record["best_state"])
    if len(current) != len(modules) or len(best) != len(modules):
        raise ValueError("router training checkpoint module count changed")
    _restore(modules, current)
    optimizer.load_state_dict(record["optimizer_state"])
    _optimizer_to(optimizer, device)
    return _ResumeState(
        int(record["next_epoch"]),
        int(record["optimizer_steps"]),
        float(record["best_validation_loss"]),
        int(record["stale_epochs"]),
        best,
        bool(record["completed"]),
    )


def _save_checkpoint(
    checkpoint_path: str | Path | None,
    identity: str,
    modules: Sequence[nn.Module],
    optimizer: torch.optim.Optimizer,
    next_epoch: int,
    steps: int,
    best_loss: float,
    stale: int,
    best: Sequence[Mapping[str, Tensor]],
    completed: bool,
) -> None:
    if checkpoint_path is None:
        return
    atomic_torch_save(
        checkpoint_path,
        {
            "best_state": tuple(dict(values) for values in best),
            "best_validation_loss": best_loss,
            "completed": completed,
            "cpu_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else ()
            ),
            "identity": identity,
            "module_state": _state(modules),
            "next_epoch": next_epoch,
            "optimizer_state": optimizer.state_dict(),
            "optimizer_steps": steps,
            "schema_version": "imagenetr50-router-training-checkpoint-v1",
            "stale_epochs": stale,
        },
    )


def _validation_ce(
    data: RouterTrainingData,
    nodes: Sequence[ScoringNode],
    image_ids: Sequence[str],
    stage: int,
    teacher: ImageNetRouterTeacher,
    batch_size: int,
    device: torch.device,
) -> float:
    losses, rows = 0.0, 0
    with torch.no_grad():
        for offset in range(0, len(image_ids), batch_size):
            batch_ids = tuple(image_ids[offset : offset + batch_size])
            query, labels, tasks = data.batch(batch_ids, device)
            targets = _target_indices(teacher, stage, query, labels, tasks, nodes)
            loss = F.cross_entropy(score_nodes(query, nodes), targets, reduction="sum")
            losses += float(loss.item())
            rows += len(batch_ids)
    return losses / rows


def _validation_parent_loss(
    data: RouterTrainingData,
    post: Sequence[ScoringNode],
    parent: ScoringNode,
    left: ScoringNode,
    right: ScoringNode,
    image_ids: Sequence[str],
    stage: int,
    teacher: ImageNetRouterTeacher,
    lse_weight: float,
    batch_size: int,
    device: torch.device,
) -> float:
    """Evaluate the exact route-plus-distillation objective used for parent fitting."""
    losses, rows = 0.0, 0
    with torch.no_grad():
        for offset in range(0, len(image_ids), batch_size):
            batch_ids = tuple(image_ids[offset : offset + batch_size])
            query, labels, tasks = data.batch(batch_ids, device)
            targets = _target_indices(teacher, stage, query, labels, tasks, post)
            scores = score_nodes(query, post)
            parent_score = parent.scorer.score(query, parent.features)
            child_lse = torch.logaddexp(
                left.scorer.score(query, left.features),
                right.scorer.score(query, right.features),
            )
            loss = F.cross_entropy(scores, targets, reduction="sum")
            loss += lse_weight * F.mse_loss(
                parent_score, child_lse, reduction="sum"
            )
            losses += float(loss.item())
            rows += len(batch_ids)
    return losses / rows


def fit_flat_frontier(
    data: RouterTrainingData,
    nodes: Sequence[ScoringNode],
    stage: int,
    training: RouterTrainingConfig,
    seed: int,
    validation_batch_size: int,
    device: torch.device,
    checkpoint_path: str | Path | None = None,
    namespace: str = "flat",
) -> RouterTrainingResult:
    """Jointly fit a complete frontier as a non-scaling capacity control."""
    frontier = tuple(nodes)
    require_class_partition(stage, _live_classes(frontier))
    modules = tuple(
        node.scorer for node in frontier if isinstance(node.scorer, nn.Module)
    )
    if len(modules) != len(frontier):
        raise ValueError("flat fitting requires independently trainable scorers")
    for node in frontier:
        move_scorer(node.scorer, device)
    fit_ids = data.ids("fit", stage)
    validation_ids = data.ids("validation", stage)
    data.require_fit(fit_ids, stage)
    optimizer = torch.optim.AdamW(
        (parameter for module in modules for parameter in module.parameters()),
        lr=training.lr,
        weight_decay=training.weight_decay,
    )
    teacher = ImageNetRouterTeacher()
    identity = _checkpoint_identity(
        "flat",
        namespace,
        stage,
        seed,
        fit_ids,
        validation_ids,
        training,
        {
            "nodes": [
                {
                    "descriptor": node.features.descriptor_sha256,
                    "node_id": node.node_id,
                    "response": node.features.response_kernel_sha256,
                }
                for node in frontier
            ]
        },
    )
    resumed = _resume(checkpoint_path, identity, modules, optimizer, device)
    best, best_loss, stale, steps = (
        resumed.best,
        resumed.best_loss,
        resumed.stale,
        resumed.steps,
    )
    epochs = resumed.next_epoch
    if resumed.completed:
        _restore(modules, best)
        for module in modules:
            module.to("cpu")
        return RouterTrainingResult(epochs, steps, best_loss, fit_ids, validation_ids)
    for epoch in range(resumed.next_epoch, training.max_epochs):
        for batch_ids in _batches(
            fit_ids, training.batch_size, namespace, seed, epoch
        ):
            query, labels, tasks = data.batch(batch_ids, device)
            targets = _target_indices(teacher, stage, query, labels, tasks, frontier)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(score_nodes(query, frontier), targets)
            loss.backward()
            optimizer.step()
            steps += 1
        epochs = epoch + 1
        value = _validation_ce(
            data,
            frontier,
            validation_ids,
            stage,
            teacher,
            validation_batch_size,
            device,
        )
        if value < best_loss - 1.0e-8:
            best, best_loss, stale = _state(modules), value, 0
        else:
            stale += 1
        _save_checkpoint(
            checkpoint_path,
            identity,
            modules,
            optimizer,
            epochs,
            steps,
            best_loss,
            stale,
            best,
            stale >= training.patience,
        )
        if stale >= training.patience:
            break
    _restore(modules, best)
    _save_checkpoint(
        checkpoint_path,
        identity,
        modules,
        optimizer,
        epochs,
        steps,
        best_loss,
        stale,
        best,
        True,
    )
    for module in modules:
        module.to("cpu")
    return RouterTrainingResult(epochs, steps, best_loss, fit_ids, validation_ids)


def fit_new_leaf(
    data: RouterTrainingData,
    new_leaf: ScoringNode,
    old_frontier: Sequence[ScoringNode],
    stage: int,
    training: RouterTrainingConfig,
    seed: int,
    validation_batch_size: int,
    device: torch.device,
    namespace: str,
    checkpoint_path: str | Path | None = None,
) -> RouterTrainingResult:
    """Calibrate one new leaf while every old score function remains frozen."""
    old = tuple(old_frontier)
    if not isinstance(new_leaf.scorer, nn.Module) or new_leaf.represented_task_ids != (stage - 1,):
        raise ValueError("causal insertion requires one trainable current-task leaf")
    old_hashes = tuple(scorer_state_hash(node.scorer) for node in old)
    positives = data.ids("fit", stage, (stage - 1,))
    negatives = negative_reservoirs(
        data,
        old,
        stage,
        training.negatives_per_live_node,
        f"{namespace}:leaf-negatives",
    )
    train_ids = positives + negatives
    data.require_fit(train_ids, stage)
    positive_set = frozenset(positives)
    frontier = old + (new_leaf,)
    validation_ids = data.ids("validation", stage)
    move_scorer(new_leaf.scorer, device)
    for node in old:
        move_scorer(node.scorer, device)
    optimizer = torch.optim.AdamW(
        new_leaf.scorer.parameters(), lr=training.lr, weight_decay=training.weight_decay
    )
    teacher = ImageNetRouterTeacher()
    modules = (new_leaf.scorer,)
    identity = _checkpoint_identity(
        "leaf",
        namespace,
        stage,
        seed,
        train_ids,
        validation_ids,
        training,
        {
            "new_descriptor": new_leaf.features.descriptor_sha256,
            "new_node_id": new_leaf.node_id,
            "new_response": new_leaf.features.response_kernel_sha256,
            "old_frontier": [
                {"node_id": node.node_id, "state": scorer_state_hash(node.scorer)}
                for node in old
            ],
        },
    )
    resumed = _resume(checkpoint_path, identity, modules, optimizer, device)
    best, best_loss, stale, steps = (
        resumed.best,
        resumed.best_loss,
        resumed.stale,
        resumed.steps,
    )
    epochs = resumed.next_epoch
    if resumed.completed:
        _restore(modules, best)
    for epoch in range(resumed.next_epoch, 0 if resumed.completed else training.max_epochs):
        for batch_ids in _batches(train_ids, training.batch_size, namespace, seed, epoch):
            query, labels, tasks = data.batch(batch_ids, device)
            new_scores = new_leaf.scorer.score(query, new_leaf.features)
            with torch.no_grad():
                old_scores = score_nodes(query, old)
            mask = torch.tensor(
                [value in positive_set for value in batch_ids],
                dtype=torch.bool,
                device=device,
            )
            losses = []
            if torch.any(mask):
                losses.append(
                    F.softplus(
                        torch.logsumexp(old_scores[mask], dim=-1)
                        - new_scores[mask]
                        + training.margin
                    )
                )
            if torch.any(~mask):
                negative_query = RouterQuery(
                    tuple(value for value, keep in zip(batch_ids, (~mask).tolist()) if keep),
                    query.prelogits[~mask],
                    {name: values[~mask] for name, values in query.cls_activations.items()},
                )
                old_targets = teacher.target_indices(
                    stage,
                    negative_query.image_ids,
                    labels[~mask].detach().cpu(),
                    tasks[~mask].detach().cpu(),
                    _live_classes(old),
                ).to(device)
                owner = old_scores[~mask].gather(1, old_targets[:, None]).squeeze(1)
                losses.append(F.softplus(new_scores[~mask] - owner + training.margin))
            loss = torch.cat(tuple(losses)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            steps += 1
        epochs = epoch + 1
        value = _validation_ce(
            data,
            frontier,
            validation_ids,
            stage,
            teacher,
            validation_batch_size,
            device,
        )
        if value < best_loss - 1.0e-8:
            best, best_loss, stale = _state((new_leaf.scorer,)), value, 0
        else:
            stale += 1
        _save_checkpoint(
            checkpoint_path,
            identity,
            modules,
            optimizer,
            epochs,
            steps,
            best_loss,
            stale,
            best,
            stale >= training.patience,
        )
        if stale >= training.patience:
            break
    _restore(modules, best)
    _save_checkpoint(
        checkpoint_path,
        identity,
        modules,
        optimizer,
        epochs,
        steps,
        best_loss,
        stale,
        best,
        True,
    )
    new_leaf.scorer.to("cpu")
    for node in old:
        move_scorer(node.scorer, torch.device("cpu"))
    new_old_hashes = tuple(scorer_state_hash(node.scorer) for node in old)
    if new_old_hashes != old_hashes:
        raise ValueError("causal leaf insertion changed an old router state")
    return RouterTrainingResult(epochs, steps, best_loss, train_ids, validation_ids)


def fit_parent(
    data: RouterTrainingData,
    parent: ScoringNode,
    left: ScoringNode,
    right: ScoringNode,
    other_frontier: Sequence[ScoringNode],
    stage: int,
    training_ids: Sequence[str],
    training: RouterTrainingConfig,
    seed: int,
    validation_batch_size: int,
    device: torch.device,
    namespace: str,
    checkpoint_path: str | Path | None = None,
) -> RouterTrainingResult:
    """Fit or repair one parent against route truth and exact child LSE."""
    if not isinstance(parent.scorer, nn.Module):
        raise ValueError("parent fitting requires a trainable fixed-size scorer")
    data.require_fit(training_ids, stage)
    post = tuple(other_frontier) + (parent,)
    require_class_partition(stage, _live_classes(post))
    validation_ids = data.ids("validation", stage)
    move_scorer(parent.scorer, device)
    for node in (left, right, *other_frontier):
        move_scorer(node.scorer, device)
    optimizer = torch.optim.AdamW(
        parent.scorer.parameters(), lr=training.lr, weight_decay=training.weight_decay
    )
    teacher = ImageNetRouterTeacher()
    modules = (parent.scorer,)
    identity = _checkpoint_identity(
        "parent",
        namespace,
        stage,
        seed,
        training_ids,
        validation_ids,
        training,
        {
            "left": scorer_state_hash(left.scorer),
            "other_frontier": [
                {"node_id": node.node_id, "state": scorer_state_hash(node.scorer)}
                for node in other_frontier
            ],
            "parent_descriptor": parent.features.descriptor_sha256,
            "parent_node_id": parent.node_id,
            "parent_response": parent.features.response_kernel_sha256,
            "right": scorer_state_hash(right.scorer),
        },
    )
    resumed = _resume(checkpoint_path, identity, modules, optimizer, device)
    best, best_loss, stale, steps = (
        resumed.best,
        resumed.best_loss,
        resumed.stale,
        resumed.steps,
    )
    epochs = resumed.next_epoch
    if resumed.completed:
        _restore(modules, best)
    for epoch in range(resumed.next_epoch, 0 if resumed.completed else training.max_epochs):
        for batch_ids in _batches(training_ids, training.batch_size, namespace, seed, epoch):
            query, labels, tasks = data.batch(batch_ids, device)
            targets = _target_indices(teacher, stage, query, labels, tasks, post)
            parent_score = parent.scorer.score(query, parent.features)
            with torch.no_grad():
                lse = torch.logaddexp(
                    left.scorer.score(query, left.features),
                    right.scorer.score(query, right.features),
                )
                other_scores = (
                    score_nodes(query, other_frontier)
                    if other_frontier
                    else torch.empty((len(batch_ids), 0), device=device)
                )
            scores = torch.cat((other_scores, parent_score[:, None]), dim=-1)
            loss = F.cross_entropy(scores, targets) + training.lse_weight * F.mse_loss(
                parent_score, lse
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            steps += 1
        epochs = epoch + 1
        value = _validation_parent_loss(
            data,
            post,
            parent,
            left,
            right,
            validation_ids,
            stage,
            teacher,
            training.lse_weight,
            validation_batch_size,
            device,
        )
        if value < best_loss - 1.0e-8:
            best, best_loss, stale = _state((parent.scorer,)), value, 0
        else:
            stale += 1
        _save_checkpoint(
            checkpoint_path,
            identity,
            modules,
            optimizer,
            epochs,
            steps,
            best_loss,
            stale,
            best,
            stale >= training.patience,
        )
        if stale >= training.patience:
            break
    _restore(modules, best)
    _save_checkpoint(
        checkpoint_path,
        identity,
        modules,
        optimizer,
        epochs,
        steps,
        best_loss,
        stale,
        best,
        True,
    )
    parent.scorer.to("cpu")
    for node in (left, right, *other_frontier):
        move_scorer(node.scorer, torch.device("cpu"))
    return RouterTrainingResult(
        epochs,
        steps,
        best_loss,
        tuple(training_ids),
        validation_ids,
    )


__all__ = [
    "RouterTrainingData",
    "RouterTrainingResult",
    "deterministic_reservoir",
    "fit_flat_frontier",
    "fit_new_leaf",
    "fit_parent",
    "negative_reservoirs",
    "repair_reservoir",
]

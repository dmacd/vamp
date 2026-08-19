"""Deterministic presentation schedules for leaves, repair, and baselines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import random

from apm.continual.artifacts import record_sha256, require_sha256
from apm.continual.trace.data import TraceExample
from apm.continual.trace.protocol import SEED, TASKS, stable_seed


@dataclass(frozen=True, slots=True)
class TrainingPlan:
    """One fully materialized and content-addressed example presentation order."""

    name: str
    example_ids: tuple[str, ...]
    phase_boundaries: tuple[tuple[str, int], ...]
    seed: int = SEED

    def __post_init__(self) -> None:
        if not self.name or not self.example_ids or self.seed != SEED:
            raise ValueError("training plans require a name, examples, and seed 1234")
        for identity in self.example_ids:
            require_sha256(identity, "training plan example")
        if (
            not self.phase_boundaries
            or self.phase_boundaries[-1][1] != len(self.example_ids)
            or any(
                not name or end <= (self.phase_boundaries[index - 1][1] if index else 0)
                for index, (name, end) in enumerate(self.phase_boundaries)
            )
        ):
            raise ValueError("training phase boundaries do not cover the plan")

    @property
    def plan_hash(self) -> str:
        """Return the exact order identity bound into checkpoints."""
        return record_sha256(self.as_record())

    def as_record(self) -> dict[str, object]:
        """Return a canonical schedule record."""
        return {
            "example_ids": list(self.example_ids),
            "format": "trace-training-plan-v1",
            "name": self.name,
            "phase_boundaries": [list(boundary) for boundary in self.phase_boundaries],
            "seed": self.seed,
        }


def shuffled_epochs(
    example_ids: Sequence[str],
    epochs: int,
    namespace: str,
) -> tuple[str, ...]:
    """Return independently and deterministically shuffled finite epochs."""
    if epochs <= 0 or not example_ids:
        raise ValueError("finite-epoch schedules require examples and positive epochs")
    orders: list[str] = []
    for epoch in range(epochs):
        current = list(example_ids)
        random.Random(stable_seed(namespace, epoch)).shuffle(current)
        orders.extend(current)
    return tuple(orders)


def leaf_plan(examples: Sequence[TraceExample], arrival: int) -> TrainingPlan:
    """Build one fresh leaf's exact task-specific finite-epoch order."""
    selected = tuple(example for example in examples if example.arrival == arrival)
    if len(selected) != 100 or len({example.task for example in selected}) != 1:
        raise ValueError("leaf plans require one complete 100-example arrival")
    task = next(task for task in TASKS if task.name == selected[0].task)
    order = shuffled_epochs(
        tuple(example.example_id for example in selected),
        task.epochs,
        f"leaf-{arrival:02d}",
    )
    return TrainingPlan(
        name=f"leaf-{arrival:02d}",
        example_ids=order,
        phase_boundaries=((selected[0].task, len(order)),),
    )


def sequential_reference_plan(examples: Sequence[TraceExample]) -> TrainingPlan:
    """Build the eight-task sequential reference order with one optimizer lifetime."""
    return _segmented_plan(examples, "seq_lora_reference", group_by_arrival=False)


def sequential_40_plan(examples: Sequence[TraceExample]) -> TrainingPlan:
    """Build the 40-arrival sequential control with one optimizer lifetime."""
    return _segmented_plan(examples, "seq_lora_40", group_by_arrival=True)


def joint_iid_plan(examples: Sequence[TraceExample]) -> TrainingPlan:
    """Build one global shuffle of the matched 20,000-presentation multiset."""
    training = tuple(example for example in examples if example.split == "train")
    by_task = {task.name: tuple(e for e in training if e.task == task.name) for task in TASKS}
    presentations = [
        example.example_id
        for task in TASKS
        for _ in range(task.epochs)
        for example in by_task[task.name]
    ]
    random.Random(stable_seed("joint_iid_lora")).shuffle(presentations)
    if len(presentations) != 20_000:
        raise ValueError("joint-IID plan must contain exactly 20,000 presentations")
    return TrainingPlan(
        name="joint_iid_lora",
        example_ids=tuple(presentations),
        phase_boundaries=(("joint_iid", len(presentations)),),
    )


def taskwise_plans(examples: Sequence[TraceExample]) -> tuple[TrainingPlan, ...]:
    """Build eight independent taskwise baseline schedules."""
    training = tuple(example for example in examples if example.split == "train")
    return tuple(
        TrainingPlan(
            name=f"taskwise_lora-{task.name}",
            example_ids=order,
            phase_boundaries=((task.name, len(order)),),
        )
        for task in TASKS
        for order in (
            shuffled_epochs(
                tuple(example.example_id for example in training if example.task == task.name),
                task.epochs,
                f"taskwise_lora-{task.name}",
            ),
        )
    )


def repair_plan(
    examples_by_id: Mapping[str, TraceExample],
    example_ids: Sequence[str],
    node_id: str,
) -> TrainingPlan:
    """Build the one-epoch deterministic repair order for a merged node."""
    require_sha256(node_id, "repair node")
    if any(identity not in examples_by_id for identity in example_ids):
        raise ValueError("repair plan references an unknown example")
    order = shuffled_epochs(example_ids, 1, f"repair-{node_id}")
    return TrainingPlan(
        name=f"repair-{node_id}",
        example_ids=order,
        phase_boundaries=(("repair", len(order)),),
    )


def retrained_parent_plan(
    examples: Sequence[TraceExample],
    start_arrival: int,
    end_arrival: int,
) -> TrainingPlan:
    """Build a fresh-parent calibration plan with matched task epoch weighting."""
    if not 1 <= start_arrival <= end_arrival <= 40:
        raise ValueError("retrained-parent interval is outside the TRACE stream")
    selected = tuple(
        example
        for example in examples
        if example.arrival is not None
        and start_arrival <= example.arrival <= end_arrival
    )
    if len(selected) != (end_arrival - start_arrival + 1) * 100:
        raise ValueError("retrained-parent interval is incomplete")
    order: list[str] = []
    boundaries: list[tuple[str, int]] = []
    for task in TASKS:
        task_examples = tuple(example.example_id for example in selected if example.task == task.name)
        if not task_examples:
            continue
        order.extend(
            shuffled_epochs(
                task_examples,
                task.epochs,
                f"retrained-parent-{start_arrival:02d}-{end_arrival:02d}-{task.name}",
            )
        )
        boundaries.append((task.name, len(order)))
    return TrainingPlan(
        name=f"retrained-parent-{start_arrival:02d}-{end_arrival:02d}",
        example_ids=tuple(order),
        phase_boundaries=tuple(boundaries),
    )


def _segmented_plan(
    examples: Sequence[TraceExample],
    name: str,
    *,
    group_by_arrival: bool,
) -> TrainingPlan:
    training = tuple(example for example in examples if example.split == "train")
    orders: list[str] = []
    boundaries: list[tuple[str, int]] = []
    groups = (
        tuple(
            (
                f"arrival-{arrival:02d}",
                tuple(example for example in training if example.arrival == arrival),
                next(task.epochs for task in TASKS if task.name == next(example.task for example in training if example.arrival == arrival)),
            )
            for arrival in range(1, 41)
        )
        if group_by_arrival
        else tuple(
            (
                task.name,
                tuple(example for example in training if example.task == task.name),
                task.epochs,
            )
            for task in TASKS
        )
    )
    for group_name, group_examples, epochs in groups:
        orders.extend(
            shuffled_epochs(
                tuple(example.example_id for example in group_examples),
                epochs,
                f"{name}-{group_name}",
            )
        )
        boundaries.append((group_name, len(orders)))
    if len(orders) != 20_000:
        raise ValueError(f"{name} must contain exactly 20,000 presentations")
    return TrainingPlan(name=name, example_ids=tuple(orders), phase_boundaries=tuple(boundaries))


__all__ = [
    "TrainingPlan",
    "joint_iid_plan",
    "leaf_plan",
    "repair_plan",
    "retrained_parent_plan",
    "sequential_40_plan",
    "sequential_reference_plan",
    "shuffled_epochs",
    "taskwise_plans",
]

"""Generic router-teacher boundary with an ImageNet class-membership teacher."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import Protocol

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class TeacherExample:
    """Label-aware training/diagnostic row kept outside the runtime query API."""

    image_id: str
    label: int
    task_id: int

    def __post_init__(self) -> None:
        if len(self.image_id) != 64 or not 0 <= self.label < 200 or self.task_id != self.label // 4:
            raise ValueError("invalid ImageNet router-teacher example")


class RouterTeacher(Protocol):
    """Teacher API suitable for class membership now and utility/NLL later."""

    def target_node(
        self,
        stage: int,
        example: TeacherExample,
        live_node_classes: Sequence[tuple[str, tuple[int, ...]]],
    ) -> str: ...

    def utility(
        self,
        stage: int,
        example: TeacherExample,
        node_id: str,
        represented_classes: tuple[int, ...],
    ) -> float: ...


class ImageNetRouterTeacher:
    """Training-only teacher matching the existing true-node oracle."""

    def target_node(
        self,
        stage: int,
        example: TeacherExample,
        live_node_classes: Sequence[tuple[str, tuple[int, ...]]],
    ) -> str:
        if not 1 <= stage <= 50 or example.task_id >= stage:
            raise ValueError("teacher example is unavailable at this stage")
        matches = tuple(
            node_id for node_id, classes in live_node_classes if example.label in classes
        )
        if len(matches) != 1:
            raise ValueError("class must map to exactly one live router node")
        return matches[0]

    def utility(
        self,
        stage: int,
        example: TeacherExample,
        node_id: str,
        represented_classes: tuple[int, ...],
    ) -> float:
        del node_id
        if not 1 <= stage <= 50 or example.task_id >= stage:
            raise ValueError("teacher example is unavailable at this stage")
        return float(example.label in represented_classes)

    def target_indices(
        self,
        stage: int,
        image_ids: Sequence[str],
        labels: Tensor,
        task_ids: Tensor,
        live_node_classes: Sequence[tuple[str, tuple[int, ...]]],
    ) -> Tensor:
        """Vectorize target construction while preserving the isolated API."""
        if labels.ndim != 1 or task_ids.ndim != 1 or len(image_ids) != labels.numel() or labels.shape != task_ids.shape:
            raise ValueError("teacher batch tensors have inconsistent shapes")
        nodes = tuple(live_node_classes)
        index = {node_id: position for position, (node_id, _classes) in enumerate(nodes)}
        targets = tuple(
            index[
                self.target_node(
                    stage,
                    TeacherExample(image_id, int(label), int(task)),
                    nodes,
                )
            ]
            for image_id, label, task in zip(
                image_ids, labels.tolist(), task_ids.tolist()
            )
        )
        return torch.tensor(targets, dtype=torch.long)


def require_class_partition(
    stage: int,
    live_node_classes: Sequence[tuple[str, tuple[int, ...]]],
) -> None:
    """Require the unique teacher target property at every historical stage."""
    classes = tuple(value for _node, values in live_node_classes for value in values)
    if len(classes) != len(set(classes)) or set(classes) != set(range(4 * stage)):
        raise ValueError("live router nodes do not partition all seen classes")


__all__ = [
    "ImageNetRouterTeacher",
    "RouterTeacher",
    "TeacherExample",
    "require_class_partition",
]

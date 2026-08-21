"""Task-known provenance routing over the immutable TRACE hierarchy."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

from apm.continual.trace.lineage import HierarchyNode, HierarchyState, build_hierarchy
from apm.continual.trace.protocol import ARRIVALS_PER_TASK, TASKS


@dataclass(frozen=True, slots=True)
class TaskKnownRoute:
    """One score-independent task-to-node decision at a TRACE stage."""

    stage: int
    task_index: int
    task: str
    candidate_id: str
    start_arrival: int
    end_arrival: int
    represented_arrivals: int
    task_arrivals: int
    coverage_count: int
    coverage: float
    purity: float

    def __post_init__(self) -> None:
        if (
            not 1 <= self.task_index <= self.stage <= len(TASKS)
            or self.task != TASKS[self.task_index - 1].name
            or self.start_arrival < 1
            or self.end_arrival < self.start_arrival
            or self.represented_arrivals != self.end_arrival - self.start_arrival + 1
            or self.task_arrivals != ARRIVALS_PER_TASK
            or not 1 <= self.coverage_count <= self.task_arrivals
            or self.coverage != self.coverage_count / self.task_arrivals
            or self.purity != self.coverage_count / self.represented_arrivals
        ):
            raise ValueError("invalid task-known provenance route")

    @property
    def interval(self) -> str:
        """Return the selected arrival interval in compact report form."""
        return f"{self.start_arrival}–{self.end_arrival}"

    def as_record(self) -> dict[str, object]:
        """Return a report-ready representation of the route."""
        return {
            "candidate_id": self.candidate_id,
            "coverage": self.coverage,
            "coverage_count": self.coverage_count,
            "end_arrival": self.end_arrival,
            "interval": self.interval,
            "purity": self.purity,
            "represented_arrivals": self.represented_arrivals,
            "stage": self.stage,
            "start_arrival": self.start_arrival,
            "task": self.task,
            "task_arrivals": self.task_arrivals,
            "task_index": self.task_index,
        }


def select_task_known_route(
    hierarchy: HierarchyState,
    stage: int,
    task_index: int,
) -> TaskKnownRoute:
    """Choose coverage, then purity, then recency without using task scores."""
    if hierarchy.arrival_count != stage * ARRIVALS_PER_TASK:
        raise ValueError("task-known routing requires the complete requested stage")
    if not 1 <= task_index <= stage <= len(TASKS):
        raise ValueError("task-known routing requires a task seen by this stage")
    task_start = (task_index - 1) * ARRIVALS_PER_TASK + 1
    task_end = task_index * ARRIVALS_PER_TASK
    ranked = tuple(
        (
            _task_overlap(node, task_start, task_end),
            node,
        )
        for node in hierarchy.active_nodes
    )
    coverage_count, selected = max(
        ranked,
        key=lambda item: (
            item[0],
            Fraction(item[0], item[1].represented_arrivals),
            item[1].end_arrival,
        ),
    )
    if coverage_count == 0:
        raise ValueError("no live hierarchy node represents the requested task")
    return TaskKnownRoute(
        stage=stage,
        task_index=task_index,
        task=TASKS[task_index - 1].name,
        candidate_id=selected.node_id,
        start_arrival=selected.start_arrival,
        end_arrival=selected.end_arrival,
        represented_arrivals=selected.represented_arrivals,
        task_arrivals=ARRIVALS_PER_TASK,
        coverage_count=coverage_count,
        coverage=coverage_count / ARRIVALS_PER_TASK,
        purity=coverage_count / selected.represented_arrivals,
    )


def build_task_known_routes(arrival_ids: Sequence[str]) -> tuple[TaskKnownRoute, ...]:
    """Build all 36 stage-local routes in canonical stage/task order."""
    expected = len(TASKS) * ARRIVALS_PER_TASK
    if len(arrival_ids) != expected:
        raise ValueError(f"task-known routing requires exactly {expected} arrivals")
    return tuple(
        select_task_known_route(
            build_hierarchy(arrival_ids[: stage * ARRIVALS_PER_TASK])[0],
            stage,
            task_index,
        )
        for stage in range(1, len(TASKS) + 1)
        for task_index in range(1, stage + 1)
    )


def _task_overlap(node: HierarchyNode, task_start: int, task_end: int) -> int:
    return max(0, min(node.end_arrival, task_end) - max(node.start_arrival, task_start) + 1)


__all__ = [
    "TaskKnownRoute",
    "build_task_known_routes",
    "select_task_known_route",
]

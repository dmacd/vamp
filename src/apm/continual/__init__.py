"""Immutable continual-learning contracts and task-free addressing."""

from apm.continual.language_tasks import (
    AddressBook,
    AddressResult,
    BaseCheckpointRef,
    CompetenceBatch,
    LanguageCurriculum,
    LanguageEvaluationExample,
    LanguageTask,
    NodeId,
    RouterBatch,
    TaskId,
    build_prefix_suffix_batches,
)

__all__ = [
    "AddressBook",
    "AddressResult",
    "BaseCheckpointRef",
    "CompetenceBatch",
    "LanguageCurriculum",
    "LanguageEvaluationExample",
    "LanguageTask",
    "NodeId",
    "RouterBatch",
    "TaskId",
    "build_prefix_suffix_batches",
]

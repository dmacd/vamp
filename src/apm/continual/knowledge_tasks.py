"""Immutable candidate-answer contracts for knowledge evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np

from apm.continual.language_tasks import CompetenceBatch, RouterBatch
from apm.memory.graph import NodeId, TaskId


CueRegime: TypeAlias = Literal[
    "cue_sufficient",
    "cue_present",
    "cue_hidden_or_ambiguous",
    "cue_free_control",
]
KnowledgeMode: TypeAlias = Literal["closed_book", "open_book"]

_CUE_REGIMES: tuple[CueRegime, ...] = (
    "cue_sufficient",
    "cue_present",
    "cue_hidden_or_ambiguous",
    "cue_free_control",
)
_KNOWLEDGE_MODES: tuple[KnowledgeMode, ...] = ("closed_book", "open_book")


@dataclass(frozen=True)
class KnowledgeCandidate:
    """One answer string and its exact prefix-plus-answer competence batch."""

    answer_text: str
    competence_batch: CompetenceBatch

    def __post_init__(self) -> None:
        if not isinstance(self.answer_text, str) or not self.answer_text.strip():
            raise ValueError("candidate answer_text must contain visible text")
        if not isinstance(self.competence_batch, CompetenceBatch):
            raise TypeError("candidate competence_batch must be a CompetenceBatch")
        if self.competence_batch.input_ids.shape[0] != 1:
            raise ValueError("each knowledge candidate must contain exactly one row")


@dataclass(frozen=True)
class KnowledgeQuery:
    """One four-way semantic query with proof, support, and routing metadata."""

    query_id: str
    task_id: TaskId
    family_id: str
    query_kind: str
    candidates: tuple[KnowledgeCandidate, ...]
    router_batch: RouterBatch
    correct_candidate_index: int
    proof_id: str
    support_ids: tuple[str, ...]
    required_edge_ids: tuple[NodeId, ...]
    cue_regime: CueRegime
    visible_cue_ids: tuple[str, ...]
    eligible_task_ids: tuple[TaskId, ...]
    novelty_regime: str
    reasoning_type: str
    reasoning_depth: int
    prefix_length: int
    mode: KnowledgeMode
    oracle_node_ids: tuple[NodeId, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "query_id",
            "task_id",
            "family_id",
            "query_kind",
            "proof_id",
            "novelty_regime",
            "reasoning_type",
        ):
            _validate_identifier(getattr(self, field_name), field_name)
        if (
            not isinstance(self.candidates, tuple)
            or len(self.candidates) != 4
            or any(
                not isinstance(candidate, KnowledgeCandidate)
                for candidate in self.candidates
            )
        ):
            raise ValueError("knowledge queries require exactly four candidates")
        answer_texts = tuple(candidate.answer_text for candidate in self.candidates)
        if len(set(answer_texts)) != len(answer_texts):
            raise ValueError("knowledge candidate answer texts must be unique")
        candidate_shapes = {
            candidate.competence_batch.input_ids.shape
            for candidate in self.candidates
        }
        if len(candidate_shapes) != 1:
            raise ValueError("all four candidate batches must share one shape")
        candidate_token_counts = tuple(
            int(candidate.competence_batch.loss_mask.sum())
            for candidate in self.candidates
        )
        if len(set(candidate_token_counts)) != 1:
            raise ValueError("all four candidates must have equal active token counts")
        if type(self.prefix_length) is not int or self.prefix_length < 2:
            raise ValueError("prefix_length must contain at least two tokens")
        _validate_paired_candidate_prefixes(self.candidates, self.prefix_length)
        _validate_router_prefix(
            self.router_batch,
            self.candidates,
            self.prefix_length,
        )
        if (
            type(self.correct_candidate_index) is not int
            or not 0 <= self.correct_candidate_index < len(self.candidates)
        ):
            raise ValueError(
                "correct_candidate_index must identify one of four candidates"
            )
        _validate_identifier_tuple(
            self.support_ids,
            "support_ids",
            require_nonempty=True,
        )
        _validate_identifier_tuple(self.required_edge_ids, "required_edge_ids")
        _validate_identifier_tuple(self.visible_cue_ids, "visible_cue_ids")
        _validate_identifier_tuple(self.eligible_task_ids, "eligible_task_ids")
        _validate_identifier_tuple(self.oracle_node_ids, "oracle_node_ids")
        if self.cue_regime not in _CUE_REGIMES:
            raise ValueError(f"unknown cue regime: {self.cue_regime}")
        if self.mode not in _KNOWLEDGE_MODES:
            raise ValueError(f"unknown knowledge mode: {self.mode}")
        if (
            type(self.reasoning_depth) is not int
            or not 0 <= self.reasoning_depth <= 2
        ):
            raise ValueError("reasoning_depth must be an integer from zero through two")
        is_cross_branch = self.reasoning_type == "cross_branch"
        if is_cross_branch and self.oracle_node_ids:
            raise ValueError("cross-branch queries cannot claim one hard-node oracle")
        if is_cross_branch and not self.required_edge_ids:
            raise ValueError("cross-branch queries require explicit edge support")
        if not self.oracle_node_ids and not is_cross_branch:
            raise ValueError("only cross-branch queries may omit hard-node oracles")


def _validate_identifier(
    value: str,
    field_name: str,
) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(character in value for character in "\r\n\t")
    ):
        raise ValueError(f"{field_name} must be one canonical nonempty identifier")


def _validate_paired_candidate_prefixes(
    candidates: tuple[KnowledgeCandidate, ...],
    prefix_length: int,
) -> None:
    batches = tuple(candidate.competence_batch for candidate in candidates)
    reference = batches[0]
    reference_loss = reference.loss_mask[0]
    first_candidate_transition = int(np.argmax(reference_loss))
    if not np.any(reference_loss) or first_candidate_transition != prefix_length - 1:
        raise ValueError("prefix_length must match the candidate loss boundary")
    paired_fields = tuple(
        (
            batch.attention_mask,
            batch.loss_mask,
            batch.input_ids[:, : first_candidate_transition + 1],
            batch.target_ids[:, :first_candidate_transition],
        )
        for batch in batches
    )
    if any(
        not np.array_equal(reference_value, candidate_value)
        for reference_value, *candidate_values in zip(*paired_fields)
        for candidate_value in candidate_values
    ):
        raise ValueError("all four candidates must share one exact visible prefix")


def _validate_router_prefix(
    router_batch: RouterBatch,
    candidates: tuple[KnowledgeCandidate, ...],
    prefix_length: int,
) -> None:
    if not isinstance(router_batch, RouterBatch):
        raise TypeError("knowledge query router_batch must be a RouterBatch")
    if router_batch.input_ids.shape[0] != 1:
        raise ValueError("knowledge query router_batch must contain exactly one row")
    router_width = router_batch.input_ids.shape[1]
    if router_width != prefix_length - 1:
        raise ValueError("router width must equal prefix_length minus one transition")
    for candidate in candidates:
        competence_batch = candidate.competence_batch
        if competence_batch.input_ids.shape[1] <= router_width:
            raise ValueError("candidate capacity must extend beyond the router prefix")
        prefix_pairs = (
            (router_batch.input_ids, competence_batch.input_ids),
            (router_batch.attention_mask, competence_batch.attention_mask),
            (router_batch.target_ids, competence_batch.target_ids),
        )
        if any(
            not np.array_equal(router_values, competence_values[:, :router_width])
            for router_values, competence_values in prefix_pairs
        ):
            raise ValueError(
                "router transitions must exactly match every candidate prefix"
            )
        if np.any(competence_batch.loss_mask[:, :router_width]):
            raise ValueError(
                "candidate loss must remain inactive across router transitions"
            )


def _validate_identifier_tuple(
    values: tuple[str, ...],
    field_name: str,
    *,
    require_nonempty: bool = False,
) -> None:
    if not isinstance(values, tuple):
        raise TypeError(f"{field_name} must be a tuple")
    if require_nonempty and not values:
        raise ValueError(f"{field_name} must not be empty")
    for value in values:
        _validate_identifier(value, field_name)
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must contain unique identifiers")


__all__ = [
    "CueRegime",
    "KnowledgeCandidate",
    "KnowledgeMode",
    "KnowledgeQuery",
]

"""Immutable language-task, evaluation, and addressing contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Sequence

import jax
import numpy as np

from apm.lm.checkpoint import BaseCheckpointRef
from apm.lm.text_data import TokenBatch
from apm.memory.graph import NodeId, TaskId


@dataclass(frozen=True, eq=False)
class _LanguageBatch:
    input_ids: np.ndarray
    attention_mask: np.ndarray
    target_ids: np.ndarray
    loss_mask: np.ndarray

    def __post_init__(self) -> None:
        input_ids = _immutable_token_ids(self.input_ids, "input_ids")
        target_ids = _immutable_token_ids(self.target_ids, "target_ids")
        attention_mask = _immutable_mask(self.attention_mask, "attention_mask")
        loss_mask = _immutable_mask(self.loss_mask, "loss_mask")
        shapes = {
            input_ids.shape,
            attention_mask.shape,
            target_ids.shape,
            loss_mask.shape,
        }
        if input_ids.ndim != 2 or len(shapes) != 1:
            raise ValueError("language batch fields must share one rank-two shape")
        if input_ids.shape[0] < 1:
            raise ValueError("language batches must contain at least one row")
        if np.any(attention_mask[:, 1:] & ~attention_mask[:, :-1]):
            raise ValueError("attention_mask must describe right-padded rows")
        if np.any(loss_mask & ~attention_mask):
            raise ValueError("loss_mask cannot activate a padded transition")
        object.__setattr__(self, "input_ids", input_ids)
        object.__setattr__(self, "attention_mask", attention_mask)
        object.__setattr__(self, "target_ids", target_ids)
        object.__setattr__(self, "loss_mask", loss_mask)

    def tree_flatten(self):
        """Expose batch arrays as dynamic JAX leaves."""
        return (
            self.input_ids,
            self.attention_mask,
            self.target_ids,
            self.loss_mask,
        ), None

    @classmethod
    def tree_unflatten(cls, auxiliary_data, children):
        """Rebuild a batch without coercing transformed JAX leaves."""
        del auxiliary_data
        batch = object.__new__(cls)
        for field_name, child in zip(
            ("input_ids", "attention_mask", "target_ids", "loss_mask"),
            children,
        ):
            object.__setattr__(batch, field_name, child)
        return batch


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False)
class RouterBatch(_LanguageBatch):
    """Prefix-only inputs and targets exposed to task-free routers."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not np.array_equal(self.loss_mask, self.attention_mask):
            raise ValueError("router loss_mask must activate every valid prefix transition")


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False)
class CompetenceBatch(_LanguageBatch):
    """Prefix context plus suffix targets, with loss active only on the suffix."""

    def __post_init__(self) -> None:
        super().__post_init__()
        for attention_row, loss_row in zip(self.attention_mask, self.loss_mask):
            active_loss = loss_row[: int(np.sum(attention_row))]
            if not np.any(active_loss):
                raise ValueError("each competence row must contain a suffix loss target")
            first_active = int(np.argmax(active_loss))
            if not np.all(active_loss[first_active:]):
                raise ValueError("competence loss must be one contiguous suffix")


@dataclass(frozen=True)
class LanguageEvaluationExample:
    """Disjoint router/competence views plus evaluator-only oracle metadata."""

    router_batch: RouterBatch
    competence_batch: CompetenceBatch
    task_id: TaskId
    oracle_node_id: NodeId

    def __post_init__(self) -> None:
        if self.router_batch.input_ids.shape[0] != self.competence_batch.input_ids.shape[0]:
            raise ValueError("router and competence batches must have equal row counts")
        router_width = self.router_batch.input_ids.shape[1]
        if self.competence_batch.input_ids.shape[1] <= router_width:
            raise ValueError("competence capacity must extend beyond the router prefix")
        prefix_pairs = (
            (self.router_batch.input_ids, self.competence_batch.input_ids),
            (self.router_batch.attention_mask, self.competence_batch.attention_mask),
            (self.router_batch.target_ids, self.competence_batch.target_ids),
        )
        if any(
            not np.array_equal(router_values, competence_values[:, :router_width])
            for router_values, competence_values in prefix_pairs
        ):
            raise ValueError("router transitions must exactly match the competence prefix")
        if np.any(self.competence_batch.loss_mask[:, :router_width]):
            raise ValueError("competence loss must be inactive across router transitions")
        if not self.task_id or not self.oracle_node_id:
            raise ValueError("evaluation task and oracle node IDs must not be empty")


@dataclass(frozen=True)
class LanguageTask:
    """One immutable training task and its held-out evaluator examples."""

    task_id: TaskId
    train_batches: tuple[TokenBatch, ...]
    validation_examples: tuple[LanguageEvaluationExample, ...]
    test_examples: tuple[LanguageEvaluationExample, ...]

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("language task ID must not be empty")
        if not self.train_batches:
            raise ValueError("language tasks must contain training batches")
        if any(not isinstance(batch, TokenBatch) for batch in self.train_batches):
            raise TypeError("train_batches must contain TokenBatch values")
        examples = self.validation_examples + self.test_examples
        if any(example.task_id != self.task_id for example in examples):
            raise ValueError("evaluation example task IDs must match their LanguageTask")


@dataclass(frozen=True)
class LanguageCurriculum:
    """An ordered task sequence with exact rooted-tree node and edge capacities."""

    tasks: tuple[LanguageTask, ...]
    max_nodes: int
    max_edges: int

    def __post_init__(self) -> None:
        task_ids = tuple(task.task_id for task in self.tasks)
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("language curriculum task IDs must be unique")
        if self.max_nodes != len(self.tasks) + 1:
            raise ValueError("max_nodes must equal task count plus the root")
        if self.max_edges != len(self.tasks) or self.max_edges != self.max_nodes - 1:
            raise ValueError("max_edges must equal task count and max_nodes - 1")


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False)
class AddressBook:
    """Fixed-capacity node IDs, content keys, and valid-node mask."""

    node_ids: tuple[NodeId | None, ...]
    keys: np.ndarray
    valid_node_mask: np.ndarray

    def __post_init__(self) -> None:
        keys = np.array(self.keys, dtype=np.float32, copy=True)
        valid_node_mask = _immutable_mask(self.valid_node_mask, "valid_node_mask")
        if keys.ndim != 2 or keys.shape[0] < 1 or keys.shape[1] < 1:
            raise ValueError("address keys must have shape [max_nodes, key_dim]")
        if valid_node_mask.shape != (keys.shape[0],):
            raise ValueError("valid_node_mask must match address-book node capacity")
        if len(self.node_ids) != keys.shape[0]:
            raise ValueError("node_ids must match address-book node capacity")
        if any((node_id is not None) != bool(valid) for node_id, valid in zip(self.node_ids, valid_node_mask)):
            raise ValueError("node IDs must be present exactly at valid mask positions")
        valid_ids = tuple(node_id for node_id in self.node_ids if node_id is not None)
        if any(not node_id for node_id in valid_ids) or len(set(valid_ids)) != len(valid_ids):
            raise ValueError("valid address-book node IDs must be nonempty and unique")
        if not np.all(np.isfinite(keys)):
            raise ValueError("address keys must be finite")
        if np.any(keys[~valid_node_mask] != 0.0):
            raise ValueError("invalid address-book rows must contain zero keys")
        keys.flags.writeable = False
        object.__setattr__(self, "node_ids", tuple(self.node_ids))
        object.__setattr__(self, "keys", keys)
        object.__setattr__(self, "valid_node_mask", valid_node_mask)

    @property
    def max_nodes(self) -> int:
        """Return the fixed node capacity."""
        return self.keys.shape[0]

    @property
    def key_dim(self) -> int:
        """Return the fixed content-key width."""
        return self.keys.shape[1]

    def tree_flatten(self):
        """Expose key and mask arrays while keeping node IDs static."""
        return (self.keys, self.valid_node_mask), self.node_ids

    @classmethod
    def tree_unflatten(cls, node_ids, children):
        """Rebuild an address book without coercing transformed JAX leaves."""
        address_book = object.__new__(cls)
        object.__setattr__(address_book, "node_ids", node_ids)
        object.__setattr__(address_book, "keys", children[0])
        object.__setattr__(address_book, "valid_node_mask", children[1])
        return address_book


class AddressResult(NamedTuple):
    """Task-free node scores, probabilities, hard choices, and uncertainty."""

    selected_indices: jax.Array
    node_probabilities: jax.Array
    node_scores: jax.Array
    score_margin: jax.Array
    entropy: jax.Array


def build_prefix_suffix_batches(
    token_ids: Sequence[int],
    prefix_length: int,
    suffix_length: int,
    *,
    pad_token_id: int = 0,
) -> tuple[RouterBatch, CompetenceBatch]:
    """Build prefix-only routing and suffix-only competence views of one sequence."""
    if prefix_length < 2:
        raise ValueError("prefix_length must be at least two tokens")
    if suffix_length < 1:
        raise ValueError("suffix_length must be positive")
    if pad_token_id < 0:
        raise ValueError("pad_token_id must be nonnegative")
    raw_tokens = np.asarray(tuple(token_ids))
    if raw_tokens.ndim != 1 or raw_tokens.dtype.kind not in "iu":
        raise TypeError("token_ids must be a one-dimensional integer sequence")
    if raw_tokens.size < 1 or np.any(raw_tokens < 0):
        raise ValueError("token_ids must contain nonnegative tokens")
    if raw_tokens.size <= prefix_length:
        raise ValueError("token_ids must contain at least one suffix target")
    tokens = raw_tokens.astype(np.int32, copy=False)
    prefix_tokens = tokens[:prefix_length]
    router_capacity = prefix_length - 1
    router_transition_count = max(len(prefix_tokens) - 1, 0)
    router_input, router_target, router_mask = _padded_transitions(
        prefix_tokens,
        router_capacity,
        router_transition_count,
        pad_token_id,
    )
    combined_tokens = tokens[: prefix_length + suffix_length]
    competence_capacity = prefix_length + suffix_length - 1
    competence_transition_count = max(len(combined_tokens) - 1, 0)
    competence_input, competence_target, competence_attention = _padded_transitions(
        combined_tokens,
        competence_capacity,
        competence_transition_count,
        pad_token_id,
    )
    competence_loss = np.zeros((1, competence_capacity), dtype=np.bool_)
    suffix_start = prefix_length - 1
    competence_loss[0, suffix_start:competence_transition_count] = True
    return (
        RouterBatch(router_input, router_mask, router_target, router_mask),
        CompetenceBatch(
            competence_input,
            competence_attention,
            competence_target,
            competence_loss,
        ),
    )


def _padded_transitions(
    tokens: np.ndarray,
    capacity: int,
    transition_count: int,
    pad_token_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    input_ids = np.full((1, capacity), pad_token_id, dtype=np.int32)
    target_ids = np.full((1, capacity), pad_token_id, dtype=np.int32)
    mask = np.zeros((1, capacity), dtype=np.bool_)
    input_ids[0, :transition_count] = tokens[:transition_count]
    target_ids[0, :transition_count] = tokens[1 : transition_count + 1]
    mask[0, :transition_count] = True
    return input_ids, target_ids, mask


def _immutable_token_ids(values: np.ndarray, field_name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind not in "iu":
        raise TypeError(f"{field_name} must contain integer token IDs")
    if np.any(array < 0):
        raise ValueError(f"{field_name} must contain nonnegative token IDs")
    normalized = np.array(array, dtype=np.int32, copy=True)
    normalized.flags.writeable = False
    return normalized


def _immutable_mask(values: np.ndarray, field_name: str) -> np.ndarray:
    array = np.asarray(values)
    if not np.all((array == 0) | (array == 1)):
        raise ValueError(f"{field_name} must contain only zero/one values")
    normalized = np.array(array, dtype=np.bool_, copy=True)
    normalized.flags.writeable = False
    return normalized


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

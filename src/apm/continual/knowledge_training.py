"""Validation-only parent selection and counterfactual TinyWorlds training."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
import re
from typing import Literal, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np

from apm.continual.knowledge_tasks import KnowledgeQuery
from apm.lm.candidate_scoring import (
    score_edge_coefficient_candidates,
    score_hard_node_candidates,
)
from apm.lm.config import GptNeoConfig
from apm.lm.lora import LoraConfig, LoraEdge
from apm.lm.lora_memory import (
    PackedLoraMemory,
    edge_coefficients_for_node,
    packed_with_candidate_edge,
)
from apm.lm.parameters import GptNeoParams
from apm.lm.text_data import TokenBatch
from apm.lm.training import LmTrainConfig, LmTrainState
from apm.lm.workflow import (
    CandidateTrainingCheckpoint,
    run_resumable_candidate_edge_updates,
)
from apm.memory.graph import (
    MemoryGraph,
    NodeId,
    TaskId,
    add_memory_node,
    memory_node_ids,
    path_incidence_matrix,
)


CounterfactualRole: TypeAlias = Literal[
    "root",
    "true_parent",
    "selected_parent",
    "strongest_other_family",
]
_COUNTERFACTUAL_ROLES: tuple[CounterfactualRole, ...] = (
    "root",
    "true_parent",
    "selected_parent",
    "strongest_other_family",
)
_SCORING_BASIS = "mean_validation_correct_candidate_nll"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _transfer_checkpoint_updates(final_update: int) -> tuple[int, ...]:
    """Return update zero, powers of two, and the exact final update."""
    updates = [0]
    value = 1
    while value < final_update:
        updates.append(value)
        value *= 2
    if final_update > 0:
        updates.append(final_update)
    return tuple(dict.fromkeys(updates))


@dataclass(frozen=True)
class KnowledgeValidationSuite:
    """Explicit validation-only queries admitted to selection and tuning."""

    suite_id: str
    split: Literal["validation"]
    task_id: TaskId
    family_id: str
    queries: tuple[KnowledgeQuery, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.suite_id, str) or not self.suite_id:
            raise ValueError("validation suite_id must not be empty")
        if self.split != "validation":
            raise ValueError("knowledge parent selection accepts validation data only")
        if (
            not self.task_id
            or not isinstance(self.family_id, str)
            or not self.family_id
        ):
            raise ValueError("validation task and family IDs must not be empty")
        if (
            not isinstance(self.queries, tuple)
            or not self.queries
            or any(not isinstance(query, KnowledgeQuery) for query in self.queries)
        ):
            raise ValueError("validation suites require KnowledgeQuery values")
        query_ids = tuple(query.query_id for query in self.queries)
        if len(set(query_ids)) != len(query_ids):
            raise ValueError("validation query IDs must be unique")
        if any(
            query.task_id != self.task_id or query.family_id != self.family_id
            for query in self.queries
        ):
            raise ValueError(
                "validation queries must match their suite task and family"
            )

    @property
    def query_ids(self) -> tuple[str, ...]:
        """Return ordered validation query IDs for provenance evidence."""
        return tuple(query.query_id for query in self.queries)

    @property
    def content_sha256(self) -> str:
        """Return a digest over query metadata and every scored token array."""
        return _validation_suite_checksum(self)


@dataclass(frozen=True, eq=False)
class KnowledgeParentSearchResult:
    """Insertion-ordered validation correct-NLL evidence and parent choice."""

    task_id: TaskId
    family_id: str
    validation_suite_id: str
    validation_suite_sha256: str
    validation_query_ids: tuple[str, ...]
    node_ids: tuple[NodeId, ...]
    correct_candidate_nll_by_query_and_node: np.ndarray
    mean_correct_candidate_nll: tuple[float, ...]
    selected_node_index: int
    selected_node_id: NodeId
    scoring_basis: str = _SCORING_BASIS

    def __post_init__(self) -> None:
        if not self.task_id or not self.family_id or not self.validation_suite_id:
            raise ValueError("parent search provenance IDs must not be empty")
        _validate_sha256(
            self.validation_suite_sha256,
            "parent search validation suite checksum",
        )
        if (
            not isinstance(self.validation_query_ids, tuple)
            or not self.validation_query_ids
            or any(not query_id for query_id in self.validation_query_ids)
            or len(set(self.validation_query_ids)) != len(self.validation_query_ids)
        ):
            raise ValueError("parent search validation query IDs must be unique")
        if (
            not isinstance(self.node_ids, tuple)
            or not self.node_ids
            or len(set(self.node_ids)) != len(self.node_ids)
        ):
            raise ValueError("parent search node IDs must be nonempty and unique")
        scores = _immutable_float_array(
            self.correct_candidate_nll_by_query_and_node,
            "correct_candidate_nll_by_query_and_node",
            ndim=2,
        )
        expected_shape = (len(self.validation_query_ids), len(self.node_ids))
        if scores.shape != expected_shape:
            raise ValueError(
                f"parent correct-candidate NLL must have shape {expected_shape}"
            )
        if np.any(~np.isfinite(scores)) or np.any(scores < 0.0):
            raise ValueError(
                "parent correct-candidate NLL must be finite and nonnegative"
            )
        if (
            not isinstance(self.mean_correct_candidate_nll, tuple)
            or len(self.mean_correct_candidate_nll) != len(self.node_ids)
            or any(
                not math.isfinite(value) or value < 0.0
                for value in self.mean_correct_candidate_nll
            )
        ):
            raise ValueError("mean parent correct-candidate NLL is invalid")
        expected_means = np.mean(scores, axis=0)
        if not np.allclose(
            self.mean_correct_candidate_nll,
            expected_means,
            rtol=1e-7,
            atol=1e-7,
        ):
            raise ValueError("mean parent NLL must average every validation query")
        if (
            type(self.selected_node_index) is not int
            or not 0 <= self.selected_node_index < len(self.node_ids)
        ):
            raise ValueError("selected parent index is outside current nodes")
        expected_selected = int(np.argmin(expected_means))
        if self.selected_node_index != expected_selected:
            raise ValueError("parent ties must resolve by graph insertion order")
        if self.selected_node_id != self.node_ids[self.selected_node_index]:
            raise ValueError("selected parent ID must match its insertion index")
        if self.scoring_basis != _SCORING_BASIS:
            raise ValueError(f"parent scoring basis must be {_SCORING_BASIS}")
        object.__setattr__(
            self,
            "correct_candidate_nll_by_query_and_node",
            scores,
        )


@dataclass(frozen=True)
class KnowledgeParentContext:
    """Current task topology and family ownership of committed graph nodes."""

    task_id: TaskId
    family_id: str
    true_parent_node_id: NodeId
    node_family_ids: tuple[tuple[NodeId, str | None], ...]

    def __post_init__(self) -> None:
        if not self.task_id or not self.family_id or not self.true_parent_node_id:
            raise ValueError("parent context IDs must not be empty")
        if not isinstance(self.node_family_ids, tuple) or not self.node_family_ids:
            raise ValueError("parent context requires insertion-ordered node families")
        node_ids = tuple(node_id for node_id, _ in self.node_family_ids)
        if any(not node_id for node_id in node_ids) or len(set(node_ids)) != len(
            node_ids
        ):
            raise ValueError("parent context node IDs must be nonempty and unique")
        if self.node_family_ids[0][1] is not None:
            raise ValueError("the graph root must not claim a task family")
        if any(
            family_id is None or not isinstance(family_id, str) or not family_id
            for _, family_id in self.node_family_ids[1:]
        ):
            raise ValueError("every committed task node must identify one family")
        if self.true_parent_node_id not in node_ids:
            raise ValueError("true parent must already exist in the graph")


@dataclass(frozen=True)
class ParentCounterfactualTarget:
    """One required transfer role, optionally unavailable for one-family runs."""

    role: CounterfactualRole
    parent_node_index: int | None
    parent_node_id: NodeId | None
    validation_mean_correct_nll: float | None

    def __post_init__(self) -> None:
        if self.role not in _COUNTERFACTUAL_ROLES:
            raise ValueError(f"unknown parent counterfactual role: {self.role}")
        values = (
            self.parent_node_index,
            self.parent_node_id,
            self.validation_mean_correct_nll,
        )
        if all(value is None for value in values):
            if self.role != "strongest_other_family":
                raise ValueError("only the other-family role may be unavailable")
            return
        if any(value is None for value in values):
            raise ValueError(
                "available counterfactual targets require complete metadata"
            )
        if type(self.parent_node_index) is not int or self.parent_node_index < 0:
            raise ValueError("counterfactual parent index must be nonnegative")
        if not self.parent_node_id:
            raise ValueError("counterfactual parent ID must not be empty")
        if (
            not math.isfinite(self.validation_mean_correct_nll)
            or self.validation_mean_correct_nll < 0.0
        ):
            raise ValueError("counterfactual validation parent NLL is invalid")

    @property
    def available(self) -> bool:
        """Return whether this topology has a parent for the role."""
        return self.parent_node_index is not None


@dataclass(frozen=True)
class ParentCounterfactualPlan:
    """Fixed four-role plan derived only from validation parent evidence."""

    parent_search: KnowledgeParentSearchResult
    context: KnowledgeParentContext
    targets: tuple[ParentCounterfactualTarget, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.parent_search, KnowledgeParentSearchResult):
            raise TypeError("counterfactual plan requires parent search evidence")
        if not isinstance(self.context, KnowledgeParentContext):
            raise TypeError("counterfactual plan requires a parent context")
        if self.parent_search.task_id != self.context.task_id or (
            self.parent_search.family_id != self.context.family_id
        ):
            raise ValueError("counterfactual task context must match parent search")
        if tuple(node_id for node_id, _ in self.context.node_family_ids) != (
            self.parent_search.node_ids
        ):
            raise ValueError(
                "counterfactual node families must match parent candidates"
            )
        if (
            not isinstance(self.targets, tuple)
            or tuple(target.role for target in self.targets) != _COUNTERFACTUAL_ROLES
        ):
            raise ValueError("counterfactual targets must use the canonical four roles")
        for target in self.targets:
            if target.available and (
                target.parent_node_id
                != self.parent_search.node_ids[target.parent_node_index]
                or not math.isclose(
                    target.validation_mean_correct_nll,
                    self.parent_search.mean_correct_candidate_nll[
                        target.parent_node_index
                    ],
                    rel_tol=1e-7,
                    abs_tol=1e-7,
                )
            ):
                raise ValueError(
                    "counterfactual target must match parent search evidence"
                )
        if self.targets[0].parent_node_index != 0:
            raise ValueError("root counterfactual must use the graph root")
        if self.targets[1].parent_node_id != self.context.true_parent_node_id:
            raise ValueError("true-parent counterfactual must use symbolic topology")
        if self.targets[2].parent_node_id != self.parent_search.selected_node_id:
            raise ValueError(
                "selected-parent counterfactual must use validation argmin"
            )
        other_family_indices = tuple(
            index
            for index, (_, family_id) in enumerate(self.context.node_family_ids)
            if family_id is not None and family_id != self.context.family_id
        )
        expected_other_index = (
            None
            if not other_family_indices
            else min(
                other_family_indices,
                key=lambda index: (
                    self.parent_search.mean_correct_candidate_nll[index],
                    index,
                ),
            )
        )
        if self.targets[3].parent_node_index != expected_other_index:
            raise ValueError(
                "other-family counterfactual must use validation-best insertion tie"
            )

    @property
    def available_parent_ids(self) -> tuple[NodeId, ...]:
        """Return unique available parents in first-role order."""
        return tuple(
            dict.fromkeys(
                target.parent_node_id
                for target in self.targets
                if target.parent_node_id is not None
            )
        )


@dataclass(frozen=True)
class TransferCheckpointDiagnostic:
    """Reportable checkpoint validation without mutable optimizer payloads."""

    update: int
    training_loss: float | None
    validation_candidate_accuracy: float
    validation_correct_nll: float
    adapter_checksum: str

    def __post_init__(self) -> None:
        if type(self.update) is not int or self.update < 0:
            raise ValueError("transfer checkpoint update must be nonnegative")
        if self.update == 0 and self.training_loss is not None:
            raise ValueError("update-zero transfer checkpoints have no training loss")
        if self.training_loss is not None and (
            not math.isfinite(self.training_loss) or self.training_loss < 0.0
        ):
            raise ValueError("transfer checkpoint training loss is invalid")
        if (
            not math.isfinite(self.validation_candidate_accuracy)
            or not 0.0 <= self.validation_candidate_accuracy <= 1.0
        ):
            raise ValueError("checkpoint candidate accuracy must lie in [0, 1]")
        if (
            not math.isfinite(self.validation_correct_nll)
            or self.validation_correct_nll < 0.0
        ):
            raise ValueError("checkpoint validation correct NLL is invalid")
        _validate_sha256(self.adapter_checksum, "checkpoint adapter checksum")


@dataclass(frozen=True)
class ParentTransferTrialDiagnostic:
    """One unique parent's same-initialization transfer trajectory."""

    parent_node_index: int
    parent_node_id: NodeId
    roles: tuple[CounterfactualRole, ...]
    parent_validation_mean_correct_nll: float
    initial_state_checksum: str
    final_adapter_checksum: str
    final_state_checksum: str
    final_update: int
    step_losses: tuple[float, ...]
    checkpoints: tuple[TransferCheckpointDiagnostic, ...]

    def __post_init__(self) -> None:
        if type(self.parent_node_index) is not int or self.parent_node_index < 0:
            raise ValueError("transfer parent index must be nonnegative")
        if not self.parent_node_id:
            raise ValueError("transfer parent ID must not be empty")
        if (
            not isinstance(self.roles, tuple)
            or not self.roles
            or any(role not in _COUNTERFACTUAL_ROLES for role in self.roles)
            or tuple(sorted(self.roles, key=_COUNTERFACTUAL_ROLES.index)) != self.roles
        ):
            raise ValueError("transfer roles must be unique and canonically ordered")
        if len(set(self.roles)) != len(self.roles):
            raise ValueError("transfer roles must not repeat")
        if (
            not math.isfinite(self.parent_validation_mean_correct_nll)
            or self.parent_validation_mean_correct_nll < 0.0
        ):
            raise ValueError("transfer parent validation NLL is invalid")
        _validate_sha256(self.initial_state_checksum, "initial state checksum")
        _validate_sha256(self.final_adapter_checksum, "final adapter checksum")
        _validate_sha256(self.final_state_checksum, "final optimizer-state checksum")
        if type(self.final_update) is not int or self.final_update < 0:
            raise ValueError("transfer final update must be nonnegative")
        if (
            not isinstance(self.step_losses, tuple)
            or len(self.step_losses) != self.final_update
            or any(not math.isfinite(loss) or loss < 0.0 for loss in self.step_losses)
        ):
            raise ValueError("transfer step losses must cover updates zero-to-final")
        if (
            not isinstance(self.checkpoints, tuple)
            or not self.checkpoints
            or any(
                not isinstance(checkpoint, TransferCheckpointDiagnostic)
                for checkpoint in self.checkpoints
            )
        ):
            raise ValueError("transfer trials require checkpoint diagnostics")
        updates = tuple(checkpoint.update for checkpoint in self.checkpoints)
        if updates != _transfer_checkpoint_updates(self.final_update):
            raise ValueError(
                "transfer checkpoints must follow the zero/power-of-two/final schedule"
            )
        if self.checkpoints[-1].adapter_checksum != self.final_adapter_checksum:
            raise ValueError("final checkpoint and adapter checksums must match")

    @property
    def validation_correct_nll_improvement(self) -> float:
        """Return initial minus final validation correct-answer NLL."""
        return (
            self.checkpoints[0].validation_correct_nll
            - self.checkpoints[-1].validation_correct_nll
        )


@dataclass(frozen=True)
class KnowledgeTransferDiagnostics:
    """Immutable four-role plan and all unique-parent transfer trajectories."""

    plan: ParentCounterfactualPlan
    trials: tuple[ParentTransferTrialDiagnostic, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.plan, ParentCounterfactualPlan):
            raise TypeError("transfer diagnostics require a counterfactual plan")
        if (
            not isinstance(self.trials, tuple)
            or not self.trials
            or any(
                not isinstance(trial, ParentTransferTrialDiagnostic)
                for trial in self.trials
            )
        ):
            raise ValueError("transfer diagnostics require trial records")
        expected_ids = self.plan.available_parent_ids
        if tuple(trial.parent_node_id for trial in self.trials) != expected_ids:
            raise ValueError(
                "transfer trials must cover unique planned parents in order"
            )
        initial_checksums = {trial.initial_state_checksum for trial in self.trials}
        if len(initial_checksums) != 1:
            raise ValueError("all counterfactual trials must share one initial state")
        for trial in self.trials:
            expected_roles = tuple(
                target.role
                for target in self.plan.targets
                if target.parent_node_id == trial.parent_node_id
            )
            if trial.roles != expected_roles:
                raise ValueError("transfer trial roles must match the plan")

    @property
    def selected_parent_recovered(self) -> bool:
        """Return whether validation parent selection recovered symbolic topology."""
        return (
            self.plan.parent_search.selected_node_id
            == self.plan.context.true_parent_node_id
        )

    def trial_for_role(
        self,
        role: CounterfactualRole,
    ) -> ParentTransferTrialDiagnostic | None:
        """Return the unique trial serving a role, or None when unavailable."""
        if role not in _COUNTERFACTUAL_ROLES:
            raise ValueError(f"unknown parent counterfactual role: {role}")
        target = self.plan.targets[_COUNTERFACTUAL_ROLES.index(role)]
        if target.parent_node_id is None:
            return None
        return next(
            trial
            for trial in self.trials
            if trial.parent_node_id == target.parent_node_id
        )


@dataclass(frozen=True, eq=False)
class KnowledgeCounterfactualTraining:
    """Report diagnostics plus resumable final optimizer states per unique parent."""

    diagnostics: KnowledgeTransferDiagnostics
    final_states: tuple[LmTrainState[LoraEdge], ...]
    execution_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.diagnostics, KnowledgeTransferDiagnostics):
            raise TypeError("counterfactual training requires transfer diagnostics")
        _validate_sha256(
            self.execution_sha256,
            "counterfactual execution checksum",
        )
        if (
            not isinstance(self.final_states, tuple)
            or len(self.final_states) != len(self.diagnostics.trials)
            or any(not isinstance(state, LmTrainState) for state in self.final_states)
        ):
            raise ValueError("counterfactual final states must align with trials")
        for state, trial in zip(self.final_states, self.diagnostics.trials):
            if not isinstance(state.trainable, LoraEdge):
                raise TypeError("counterfactual states must train LoraEdge values")
            if int(state.step) != trial.final_update:
                raise ValueError("counterfactual state steps must match diagnostics")
            if _tree_checksum(state.trainable) != trial.final_adapter_checksum:
                raise ValueError("counterfactual state adapters must match diagnostics")
            if _tree_checksum(state) != trial.final_state_checksum:
                raise ValueError(
                    "counterfactual optimizer states must match diagnostics"
                )

    @property
    def selected_state(self) -> LmTrainState[LoraEdge]:
        """Return only the validation-selected parent's trained state."""
        selected_id = self.diagnostics.plan.parent_search.selected_node_id
        selected_index = tuple(
            trial.parent_node_id for trial in self.diagnostics.trials
        ).index(selected_id)
        return self.final_states[selected_index]


def select_knowledge_parent_from_scores(
    validation_suite: KnowledgeValidationSuite,
    graph: MemoryGraph[object],
    packed_memory: PackedLoraMemory,
    hard_candidate_nll: np.ndarray,
) -> KnowledgeParentSearchResult:
    """Select a parent from mean validation correct-candidate NLL only."""
    if not isinstance(validation_suite, KnowledgeValidationSuite):
        raise TypeError("parent selection requires a KnowledgeValidationSuite")
    valid_node_mask = _validate_graph_packing(graph, packed_memory)
    scores = np.asarray(hard_candidate_nll, dtype=np.float32)
    node_capacity = valid_node_mask.size
    expected_shape = (len(validation_suite.queries), 4, node_capacity)
    if scores.shape != expected_shape:
        raise ValueError(f"hard_candidate_nll must have shape {expected_shape}")
    if np.any(~np.isfinite(scores[:, :, valid_node_mask])) or np.any(
        scores[:, :, valid_node_mask] < 0.0
    ):
        raise ValueError("valid hard-node candidate NLL must be finite and nonnegative")
    if np.any(~np.isposinf(scores[:, :, ~valid_node_mask])):
        raise ValueError("invalid hard-node candidate NLL must be positive infinity")
    correct_indices = np.asarray(
        tuple(query.correct_candidate_index for query in validation_suite.queries),
        dtype=np.int32,
    )
    query_rows = np.arange(len(validation_suite.queries))[:, None]
    node_columns = np.flatnonzero(valid_node_mask)[None, :]
    correct_nll = scores[query_rows, correct_indices[:, None], node_columns]
    means = np.mean(correct_nll, axis=0)
    selected_index = int(np.argmin(means))
    node_ids = memory_node_ids(graph)
    return KnowledgeParentSearchResult(
        task_id=validation_suite.task_id,
        family_id=validation_suite.family_id,
        validation_suite_id=validation_suite.suite_id,
        validation_suite_sha256=validation_suite.content_sha256,
        validation_query_ids=validation_suite.query_ids,
        node_ids=node_ids,
        correct_candidate_nll_by_query_and_node=correct_nll,
        mean_correct_candidate_nll=tuple(float(value) for value in means),
        selected_node_index=selected_index,
        selected_node_id=node_ids[selected_index],
    )


def score_knowledge_parent_nodes(
    validation_suite: KnowledgeValidationSuite,
    graph: MemoryGraph[object],
    packed_memory: PackedLoraMemory,
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    lora_config: LoraConfig,
    *,
    evaluation_microbatch_size: int | None = None,
) -> KnowledgeParentSearchResult:
    """Score current hard nodes and select from an explicit validation suite."""
    if not isinstance(validation_suite, KnowledgeValidationSuite):
        raise TypeError("parent scoring requires a KnowledgeValidationSuite")
    hard_scores = score_hard_node_candidates(
        base_params,
        model_config,
        packed_memory,
        lora_config,
        validation_suite.queries,
        evaluation_microbatch_size=evaluation_microbatch_size,
    )
    return select_knowledge_parent_from_scores(
        validation_suite,
        graph,
        packed_memory,
        hard_scores,
    )


def plan_parent_counterfactuals(
    parent_search: KnowledgeParentSearchResult,
    context: KnowledgeParentContext,
) -> ParentCounterfactualPlan:
    """Plan root, true, selected, and strongest-other-family trials."""
    if not isinstance(parent_search, KnowledgeParentSearchResult):
        raise TypeError("counterfactual planning requires parent search evidence")
    if not isinstance(context, KnowledgeParentContext):
        raise TypeError("counterfactual planning requires a parent context")
    if parent_search.task_id != context.task_id or (
        parent_search.family_id != context.family_id
    ):
        raise ValueError("counterfactual context must match parent search task")
    node_ids = parent_search.node_ids
    family_by_node = dict(context.node_family_ids)
    if tuple(family_by_node) != node_ids:
        raise ValueError(
            "counterfactual node families must match parent insertion order"
        )
    true_parent_index = node_ids.index(context.true_parent_node_id)
    other_family_indices = tuple(
        index
        for index, node_id in enumerate(node_ids)
        if family_by_node[node_id] is not None
        and family_by_node[node_id] != context.family_id
    )
    strongest_other_index = (
        None
        if not other_family_indices
        else min(
            other_family_indices,
            key=lambda index: (
                parent_search.mean_correct_candidate_nll[index],
                index,
            ),
        )
    )

    def target(
        role: CounterfactualRole,
        index: int | None,
    ) -> ParentCounterfactualTarget:
        return ParentCounterfactualTarget(
            role=role,
            parent_node_index=index,
            parent_node_id=None if index is None else node_ids[index],
            validation_mean_correct_nll=(
                None
                if index is None
                else parent_search.mean_correct_candidate_nll[index]
            ),
        )

    return ParentCounterfactualPlan(
        parent_search=parent_search,
        context=context,
        targets=(
            target("root", 0),
            target("true_parent", true_parent_index),
            target("selected_parent", parent_search.selected_node_index),
            target("strongest_other_family", strongest_other_index),
        ),
    )


def run_parent_counterfactuals(
    plan: ParentCounterfactualPlan,
    validation_suite: KnowledgeValidationSuite,
    training: LmTrainState[LoraEdge] | KnowledgeCounterfactualTraining,
    train_batches: tuple[TokenBatch, ...],
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    packed_memory: PackedLoraMemory,
    lora_config: LoraConfig,
    train_config: LmTrainConfig,
    *,
    stop_update: int | None = None,
    evaluation_microbatch_size: int | None = None,
) -> KnowledgeCounterfactualTraining:
    """Run or resume same-initialization trials with validation checkpoints."""
    execution_sha256 = _validated_counterfactual_execution_checksum(
        plan,
        validation_suite,
        training,
        train_batches,
        base_params,
        model_config,
        packed_memory,
        lora_config,
        train_config,
    )
    candidate_index = len(plan.parent_search.node_ids) - 1
    previous_by_parent = (
        {}
        if isinstance(training, LmTrainState)
        else {
            trial.parent_node_id: (trial, state)
            for trial, state in zip(
                training.diagnostics.trials,
                training.final_states,
            )
        }
    )
    shared_initial_checksum = (
        _tree_checksum(training)
        if isinstance(training, LmTrainState)
        else training.diagnostics.trials[0].initial_state_checksum
    )
    diagnostics: list[ParentTransferTrialDiagnostic] = []
    final_states: list[LmTrainState[LoraEdge]] = []
    for parent_node_id in plan.available_parent_ids:
        parent_index = plan.parent_search.node_ids.index(parent_node_id)
        previous = previous_by_parent.get(parent_node_id)
        starting_state = training if previous is None else previous[1]
        assert isinstance(starting_state, LmTrainState)
        parent_coefficients = edge_coefficients_for_node(
            packed_memory,
            parent_index,
        )
        validation_function = _candidate_validation_function(
            validation_suite,
            base_params,
            model_config,
            packed_memory,
            lora_config,
            parent_coefficients,
            candidate_index,
            evaluation_microbatch_size,
        )
        final_state, loss_trace, checkpoints = (
            run_resumable_candidate_edge_updates(
                starting_state,
                train_batches,
                base_params,
                model_config,
                packed_memory,
                lora_config,
                parent_coefficients,
                candidate_index,
                train_config,
                stop_update=stop_update,
                validation_function=validation_function,
            )
        )
        new_checkpoints = tuple(
            _checkpoint_diagnostic(checkpoint) for checkpoint in checkpoints
        )
        prior_diagnostic = None if previous is None else previous[0]
        all_checkpoints = (
            new_checkpoints
            if prior_diagnostic is None
            else prior_diagnostic.checkpoints + new_checkpoints[1:]
        )
        required_updates = set(_transfer_checkpoint_updates(int(final_state.step)))
        merged_checkpoints = tuple(
            checkpoint
            for checkpoint in all_checkpoints
            if checkpoint.update in required_updates
        )
        merged_losses = (
            loss_trace.step_losses
            if prior_diagnostic is None
            else prior_diagnostic.step_losses + loss_trace.step_losses
        )
        roles = tuple(
            target.role
            for target in plan.targets
            if target.parent_node_id == parent_node_id
        )
        trial = ParentTransferTrialDiagnostic(
            parent_node_index=parent_index,
            parent_node_id=parent_node_id,
            roles=roles,
            parent_validation_mean_correct_nll=(
                plan.parent_search.mean_correct_candidate_nll[parent_index]
            ),
            initial_state_checksum=shared_initial_checksum,
            final_adapter_checksum=_tree_checksum(final_state.trainable),
            final_state_checksum=_tree_checksum(final_state),
            final_update=int(final_state.step),
            step_losses=merged_losses,
            checkpoints=merged_checkpoints,
        )
        diagnostics.append(trial)
        final_states.append(final_state)
    transfer = KnowledgeTransferDiagnostics(plan, tuple(diagnostics))
    return KnowledgeCounterfactualTraining(
        transfer,
        tuple(final_states),
        execution_sha256,
    )


def validate_parent_counterfactual_resume(
    plan: ParentCounterfactualPlan,
    validation_suite: KnowledgeValidationSuite,
    training: KnowledgeCounterfactualTraining,
    train_batches: tuple[TokenBatch, ...],
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    packed_memory: PackedLoraMemory,
    lora_config: LoraConfig,
    train_config: LmTrainConfig,
) -> None:
    """Bind a loaded counterfactual chunk to the complete current execution."""
    if not isinstance(training, KnowledgeCounterfactualTraining):
        raise TypeError("resume validation requires counterfactual training")
    _validated_counterfactual_execution_checksum(
        plan,
        validation_suite,
        training,
        train_batches,
        base_params,
        model_config,
        packed_memory,
        lora_config,
        train_config,
    )


def commit_selected_counterfactual_edge(
    graph: MemoryGraph[LoraEdge],
    training: KnowledgeCounterfactualTraining,
    *,
    train_stage: int | None = None,
) -> MemoryGraph[LoraEdge]:
    """Commit exactly the selected-parent adapter; diagnostics remain uncommitted."""
    if not isinstance(graph, MemoryGraph):
        raise TypeError("selected counterfactual commit requires a MemoryGraph")
    if not isinstance(training, KnowledgeCounterfactualTraining):
        raise TypeError("selected commit requires counterfactual training")
    plan = training.diagnostics.plan
    if memory_node_ids(graph) != plan.parent_search.node_ids:
        raise ValueError("commit graph must equal the counterfactual source graph")
    resolved_stage = len(graph.nodes) if train_stage is None else train_stage
    if type(resolved_stage) is not int or resolved_stage != len(graph.nodes):
        raise ValueError("committed train_stage must equal the next graph stage")
    return add_memory_node(
        graph,
        node_id=NodeId(str(plan.context.task_id)),
        parent_id=plan.parent_search.selected_node_id,
        trained_task=plan.context.task_id,
        train_stage=resolved_stage,
        incoming_edge=training.selected_state.trainable,
    )


def _candidate_validation_function(
    validation_suite: KnowledgeValidationSuite,
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    packed_memory: PackedLoraMemory,
    lora_config: LoraConfig,
    parent_coefficients: jax.Array,
    candidate_index: int,
    evaluation_microbatch_size: int | None,
):
    coefficients = np.asarray(parent_coefficients, dtype=np.float32).copy()
    coefficients[candidate_index] = 1.0
    coefficient_rows = np.repeat(
        coefficients[None, :],
        len(validation_suite.queries),
        axis=0,
    )
    correct_indices = np.asarray(
        tuple(query.correct_candidate_index for query in validation_suite.queries),
        dtype=np.int32,
    )

    def validate(adapter: LoraEdge, update: int) -> tuple[float, float]:
        del update
        candidate_memory = packed_with_candidate_edge(
            packed_memory,
            adapter,
            candidate_index,
        )
        scores = np.empty(
            (len(validation_suite.queries), 4),
            dtype=np.float32,
        )
        execution_shapes = tuple(
            dict.fromkeys(
                query.candidates[0].competence_batch.input_ids.shape
                for query in validation_suite.queries
            )
        )
        for execution_shape in execution_shapes:
            indices = tuple(
                index
                for index, query in enumerate(validation_suite.queries)
                if query.candidates[0].competence_batch.input_ids.shape
                == execution_shape
            )
            scores[np.asarray(indices)] = score_edge_coefficient_candidates(
                base_params,
                model_config,
                candidate_memory,
                lora_config,
                tuple(validation_suite.queries[index] for index in indices),
                coefficient_rows[np.asarray(indices)],
                evaluation_microbatch_size=evaluation_microbatch_size,
            )
        rows = np.arange(len(validation_suite.queries))
        return (
            float(np.mean(np.argmin(scores, axis=1) == correct_indices)),
            float(np.mean(scores[rows, correct_indices])),
        )

    return validate


def _checkpoint_diagnostic(
    checkpoint: CandidateTrainingCheckpoint,
) -> TransferCheckpointDiagnostic:
    if (
        checkpoint.validation_candidate_accuracy is None
        or checkpoint.validation_correct_nll is None
    ):
        raise ValueError("counterfactual checkpoints require validation evidence")
    return TransferCheckpointDiagnostic(
        update=checkpoint.update,
        training_loss=checkpoint.training_loss,
        validation_candidate_accuracy=checkpoint.validation_candidate_accuracy,
        validation_correct_nll=checkpoint.validation_correct_nll,
        adapter_checksum=_tree_checksum(checkpoint.state.trainable),
    )


def _validate_counterfactual_execution(
    plan: ParentCounterfactualPlan,
    validation_suite: KnowledgeValidationSuite,
    training: LmTrainState[LoraEdge] | KnowledgeCounterfactualTraining,
    train_batches: tuple[TokenBatch, ...],
    packed_memory: PackedLoraMemory,
    train_config: LmTrainConfig,
) -> None:
    if not isinstance(plan, ParentCounterfactualPlan):
        raise TypeError("counterfactual execution requires a plan")
    if not isinstance(validation_suite, KnowledgeValidationSuite):
        raise TypeError("counterfactual execution requires a validation suite")
    search = plan.parent_search
    if (
        validation_suite.suite_id != search.validation_suite_id
        or validation_suite.content_sha256 != search.validation_suite_sha256
        or validation_suite.query_ids != search.validation_query_ids
        or validation_suite.task_id != search.task_id
    ):
        raise ValueError("counterfactual validation suite must match parent selection")
    if (
        not isinstance(train_batches, tuple)
        or not train_batches
        or any(not isinstance(batch, TokenBatch) for batch in train_batches)
    ):
        raise ValueError("counterfactual training requires TokenBatch values")
    if not isinstance(train_config, LmTrainConfig):
        raise TypeError("counterfactual training requires an LmTrainConfig")
    valid_nodes = _validate_graph_free_packing(
        search.node_ids,
        packed_memory,
    )
    candidate_index = len(search.node_ids) - 1
    if candidate_index >= packed_memory.valid_edge_mask.shape[0]:
        raise ValueError("packed memory has no candidate edge capacity")
    if np.asarray(packed_memory.valid_edge_mask)[candidate_index]:
        raise ValueError("candidate edge slot must be uncommitted")
    if not np.all(valid_nodes[: len(search.node_ids)]):
        raise ValueError("packed memory must contain every parent candidate")
    if isinstance(training, LmTrainState):
        if not isinstance(training.trainable, LoraEdge):
            raise TypeError("counterfactual initial state must train one LoraEdge")
        if int(training.step) != 0:
            raise ValueError(
                "new counterfactual trials must share one update-zero state"
            )
    elif isinstance(training, KnowledgeCounterfactualTraining):
        if not _same_counterfactual_plan(training.diagnostics.plan, plan):
            raise ValueError("resumed counterfactual training must use the same plan")
    else:
        raise TypeError("training must be an initial state or prior result")


def _validated_counterfactual_execution_checksum(
    plan: ParentCounterfactualPlan,
    validation_suite: KnowledgeValidationSuite,
    training: LmTrainState[LoraEdge] | KnowledgeCounterfactualTraining,
    train_batches: tuple[TokenBatch, ...],
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    packed_memory: PackedLoraMemory,
    lora_config: LoraConfig,
    train_config: LmTrainConfig,
) -> str:
    _validate_counterfactual_execution(
        plan,
        validation_suite,
        training,
        train_batches,
        packed_memory,
        train_config,
    )
    execution_sha256 = _counterfactual_execution_checksum(
        plan,
        validation_suite,
        train_batches,
        base_params,
        model_config,
        packed_memory,
        lora_config,
        train_config,
        len(plan.parent_search.node_ids) - 1,
    )
    if isinstance(training, KnowledgeCounterfactualTraining) and (
        training.execution_sha256 != execution_sha256
    ):
        raise ValueError("resumed counterfactual execution inputs have changed")
    return execution_sha256


def _validate_graph_packing(
    graph: MemoryGraph[object],
    packed_memory: PackedLoraMemory,
) -> np.ndarray:
    if not isinstance(graph, MemoryGraph) or not graph.nodes:
        raise ValueError("parent selection requires a nonempty MemoryGraph")
    valid_nodes = _validate_graph_free_packing(memory_node_ids(graph), packed_memory)
    paths = np.asarray(packed_memory.node_path_matrix, dtype=np.float32)
    edge_count = len(graph.nodes) - 1
    expected = np.zeros_like(paths)
    expected[: len(graph.nodes), :edge_count] = path_incidence_matrix(graph)
    if not np.array_equal(paths, expected):
        raise ValueError("packed path matrix must exactly encode the supplied graph")
    return valid_nodes


def _validate_graph_free_packing(
    node_ids: tuple[NodeId, ...],
    packed_memory: PackedLoraMemory,
) -> np.ndarray:
    if not isinstance(packed_memory, PackedLoraMemory):
        raise TypeError("knowledge training requires PackedLoraMemory")
    paths = np.asarray(packed_memory.node_path_matrix)
    valid_nodes = np.asarray(packed_memory.valid_node_mask, dtype=np.bool_)
    valid_edges = np.asarray(packed_memory.valid_edge_mask, dtype=np.bool_)
    if (
        paths.ndim != 2
        or valid_nodes.shape != (paths.shape[0],)
        or valid_edges.shape != (paths.shape[1],)
        or len(node_ids) > paths.shape[0]
        or len(node_ids) - 1 > paths.shape[1]
    ):
        raise ValueError("packed memory shapes cannot contain current graph")
    expected_nodes = np.arange(paths.shape[0]) < len(node_ids)
    expected_edges = np.arange(paths.shape[1]) < len(node_ids) - 1
    if not np.array_equal(valid_nodes, expected_nodes) or not np.array_equal(
        valid_edges,
        expected_edges,
    ):
        raise ValueError("packed validity masks must match current graph insertion")
    return valid_nodes


def _same_counterfactual_plan(
    first: ParentCounterfactualPlan,
    second: ParentCounterfactualPlan,
) -> bool:
    return (
        first.parent_search.validation_suite_id
        == second.parent_search.validation_suite_id
        and first.parent_search.validation_suite_sha256
        == second.parent_search.validation_suite_sha256
        and first.parent_search.validation_query_ids
        == second.parent_search.validation_query_ids
        and first.parent_search.node_ids == second.parent_search.node_ids
        and first.parent_search.mean_correct_candidate_nll
        == second.parent_search.mean_correct_candidate_nll
        and first.context == second.context
        and first.targets == second.targets
    )


def _tree_checksum(tree: object) -> str:
    digest = sha256()
    leaves, structure = jax.tree_util.tree_flatten(tree)
    digest.update(str(structure).encode("utf-8"))
    for leaf in leaves:
        array = np.asarray(leaf)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _validation_suite_checksum(suite: KnowledgeValidationSuite) -> str:
    digest = sha256()
    digest.update(suite.suite_id.encode("utf-8"))
    digest.update(suite.split.encode("ascii"))
    digest.update(str(suite.task_id).encode("utf-8"))
    digest.update(suite.family_id.encode("utf-8"))
    for query in suite.queries:
        metadata = (
            query.query_id,
            str(query.task_id),
            query.family_id,
            query.query_kind,
            tuple(candidate.answer_text for candidate in query.candidates),
            query.correct_candidate_index,
            query.proof_id,
            query.support_ids,
            tuple(str(value) for value in query.required_edge_ids),
            query.cue_regime,
            query.visible_cue_ids,
            tuple(str(value) for value in query.eligible_task_ids),
            query.novelty_regime,
            query.reasoning_type,
            query.reasoning_depth,
            query.prefix_length,
            query.mode,
            tuple(str(value) for value in query.oracle_node_ids),
        )
        digest.update(repr(metadata).encode("utf-8"))
        digest.update(
            _tree_checksum(
                (
                    query.router_batch,
                    *(candidate.competence_batch for candidate in query.candidates),
                )
            ).encode("ascii")
        )
    return digest.hexdigest()


def _counterfactual_execution_checksum(
    plan: ParentCounterfactualPlan,
    validation_suite: KnowledgeValidationSuite,
    train_batches: tuple[TokenBatch, ...],
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    packed_memory: PackedLoraMemory,
    lora_config: LoraConfig,
    train_config: LmTrainConfig,
    candidate_index: int,
) -> str:
    digest = sha256()
    metadata = (
        plan.parent_search.validation_suite_sha256,
        plan.parent_search.node_ids,
        plan.parent_search.mean_correct_candidate_nll,
        plan.context,
        plan.targets,
        validation_suite.content_sha256,
        model_config,
        lora_config,
        train_config,
        candidate_index,
    )
    digest.update(repr(metadata).encode("utf-8"))
    for value in (base_params, packed_memory, train_batches):
        digest.update(_tree_checksum(value).encode("ascii"))
    return digest.hexdigest()


def _immutable_float_array(
    values: object,
    field_name: str,
    *,
    ndim: int,
) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype.kind not in "fiu":
        raise TypeError(f"{field_name} must be numeric")
    result = np.array(raw, dtype=np.float32, copy=True)
    if result.ndim != ndim:
        raise ValueError(f"{field_name} must have rank {ndim}")
    result.flags.writeable = False
    return result


def _validate_sha256(value: str, field_name: str) -> None:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


__all__ = [
    "CounterfactualRole",
    "KnowledgeCounterfactualTraining",
    "KnowledgeParentContext",
    "KnowledgeParentSearchResult",
    "KnowledgeTransferDiagnostics",
    "KnowledgeValidationSuite",
    "ParentCounterfactualPlan",
    "ParentCounterfactualTarget",
    "ParentTransferTrialDiagnostic",
    "TransferCheckpointDiagnostic",
    "commit_selected_counterfactual_edge",
    "plan_parent_counterfactuals",
    "run_parent_counterfactuals",
    "score_knowledge_parent_nodes",
    "select_knowledge_parent_from_scores",
    "validate_parent_counterfactual_resume",
]

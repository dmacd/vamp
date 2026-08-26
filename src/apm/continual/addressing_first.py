"""Immutable addressing-first tree transitions for frozen-feature classifiers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import math
from collections.abc import Iterable, Sequence

import numpy as np
from pyrsistent import pmap, pvector
from pyrsistent.typing import PMap, PVector
import torch
from torch import Tensor

from apm.continual.top_two_adapter import (
    TopTwoAdapterState,
    TopTwoAdamWState,
    TopTwoBaseState,
    TopTwoOptimizerConfig,
    sum_top_two_adapters,
    top_two_logits,
    train_top_two_adapter_step,
    zero_top_two_adapter,
    zero_top_two_adamw,
)


@dataclass(frozen=True, slots=True)
class StoredExampleTable:
    """Columnar frozen examples addressed by stable integer row IDs."""

    embeddings: Tensor
    trunk_features: Tensor
    base_logits: Tensor
    labels: Tensor
    context_ids: Tensor
    stream_steps: Tensor

    def __post_init__(self) -> None:
        rows = int(self.embeddings.shape[0])
        if (
            self.embeddings.ndim != 2
            or self.trunk_features.ndim != 2
            or self.trunk_features.shape[0] != rows
            or self.base_logits.ndim != 2
            or self.base_logits.shape[0] != rows
            or self.labels.shape != (rows,)
            or self.context_ids.shape != (rows,)
            or self.stream_steps.shape != (rows,)
            or self.embeddings.device.type != "cpu"
            or self.trunk_features.device.type != "cpu"
            or self.base_logits.device.type != "cpu"
            or self.embeddings.dtype != torch.float32
            or self.trunk_features.dtype != torch.float32
            or self.base_logits.dtype != torch.float32
            or self.labels.dtype != torch.int64
            or self.context_ids.dtype != torch.int64
            or self.stream_steps.dtype != torch.int64
        ):
            raise ValueError("stored example columns have incompatible shapes, devices, or dtypes")
        if (
            not torch.isfinite(self.embeddings).all()
            or not torch.isfinite(self.trunk_features).all()
            or not torch.isfinite(self.base_logits).all()
        ):
            raise ValueError("stored examples must be finite")

    @property
    def embedding_dim(self) -> int:
        """Return the frozen address dimension."""
        return int(self.embeddings.shape[1])

    @property
    def classes(self) -> int:
        """Return the number of base-logit classes."""
        return int(self.base_logits.shape[1])


@dataclass(frozen=True, slots=True)
class WorkCounters:
    """Logical online and structural work measured in per-example operations."""

    embedding_evaluations: int = 0
    hyperplane_evaluations: int = 0
    adapter_evaluations: int = 0
    online_training_examples: int = 0
    split_replay_examples: int = 0
    consolidation_replay_examples: int = 0
    historical_examples_repartitioned: int = 0
    pca_fit_examples: int = 0

    @property
    def counted_work(self) -> int:
        """Return the additive work proxy used by the scaling diagnostic."""
        return sum(
            (
                self.embedding_evaluations,
                self.hyperplane_evaluations,
                self.adapter_evaluations,
                self.online_training_examples,
                self.split_replay_examples,
                self.consolidation_replay_examples,
                self.historical_examples_repartitioned,
                self.pca_fit_examples,
            )
        )


@dataclass(frozen=True, slots=True)
class AFNode:
    """One immutable address region and pathwise top-two-layer ancestor."""

    node_id: int
    parent_id: int | None
    depth: int
    adapter: TopTwoAdapterState
    optimizer: TopTwoAdamWState | None
    split_direction: Tensor | None
    split_threshold: float | None
    left_id: int | None
    right_id: int | None
    total_arrivals: int
    arrivals_since_structure_change: int
    last_consolidated_subtree_size: int
    created_at_step: int

    @property
    def is_leaf(self) -> bool:
        """Return whether this node owns a current example buffer."""
        return self.left_id is None and self.right_id is None

    def __post_init__(self) -> None:
        children = (self.left_id, self.right_id)
        if (
            self.node_id < 0
            or self.depth < 0
            or self.total_arrivals < 0
            or self.arrivals_since_structure_change < 0
            or self.last_consolidated_subtree_size < 0
            or self.created_at_step < 0
            or ((children[0] is None) != (children[1] is None))
        ):
            raise ValueError("invalid AF node metadata")
        if self.is_leaf:
            if self.split_direction is not None or self.split_threshold is not None or self.optimizer is None:
                raise ValueError("leaf routing and optimizer state are inconsistent")
        elif self.split_direction is None or self.split_threshold is None or self.optimizer is not None:
            raise ValueError("internal routing and optimizer state are inconsistent")


@dataclass(frozen=True, slots=True)
class AFState:
    """Persistent tree, leaf memberships, ID allocator, and online work totals."""

    root_id: int
    base: TopTwoBaseState
    nodes: PMap[int, AFNode]
    leaf_buffers: PMap[int, PVector[int]]
    next_node_id: int
    counters: WorkCounters


@dataclass(frozen=True, slots=True)
class AFHyperparameters:
    """Fixed online, split, and consolidation settings."""

    leaf_capacity: int = 512
    split_fit_samples: int = 2_048
    batch_size: int = 64
    adapter_lr: float = 0.01
    weight_decay: float = 0.0001
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1.0e-8
    split_epochs: int = 2
    consolidation_epochs: int = 3
    depth_cap_override: int | None = None

    def __post_init__(self) -> None:
        if (
            self.leaf_capacity < 2
            or self.split_fit_samples < 2
            or self.batch_size < 1
            or self.adapter_lr <= 0.0
            or self.weight_decay < 0.0
            or not 0.0 <= self.beta1 < 1.0
            or not 0.0 <= self.beta2 < 1.0
            or self.epsilon <= 0.0
            or self.split_epochs < 1
            or self.consolidation_epochs < 1
            or (self.depth_cap_override is not None and self.depth_cap_override < 0)
        ):
            raise ValueError("invalid AF hyperparameters")


@dataclass(frozen=True, slots=True)
class RouteResult:
    """One deterministic root-to-leaf route and its comparison count."""

    path_ids: tuple[int, ...]
    leaf_id: int
    hyperplane_evaluations: int


@dataclass(frozen=True, slots=True)
class MicrobatchResult:
    """Updated state plus the leaves touched by one pre-structure microbatch."""

    state: AFState
    touched_leaf_ids: tuple[int, ...]
    routes: tuple[RouteResult, ...]
    replay_example_ids: tuple[tuple[int, tuple[int, ...]], ...]


@dataclass(frozen=True, slots=True)
class SplitEvent:
    """Installed zero-child split and its exhaustive partition."""

    parent_id: int
    left_id: int
    right_id: int
    left_examples: tuple[int, ...]
    right_examples: tuple[int, ...]
    stream_step: int


@dataclass(frozen=True, slots=True)
class ConsolidationEvent:
    """Completed two-leaf replay collapse."""

    parent_id: int
    removed_child_ids: tuple[int, int]
    example_ids: tuple[int, ...]
    stream_step: int


def init_af_state(
    base: TopTwoBaseState,
    device: torch.device | str = "cpu",
) -> AFState:
    """Create the root-only zero-adapter AF state."""
    adapter = zero_top_two_adapter(base, device)
    root = AFNode(
        0,
        None,
        0,
        adapter,
        zero_top_two_adamw(adapter),
        None,
        None,
        None,
        None,
        0,
        0,
        0,
        0,
    )
    return AFState(0, base, pmap({0: root}), pmap({0: pvector()}), 1, WorkCounters())


def current_depth_cap(stream_examples: int, hyperparameters: AFHyperparameters) -> int:
    """Return the configured fixed or logarithmic maximum leaf depth."""
    if stream_examples < 0:
        raise ValueError("stream example count must be nonnegative")
    if hyperparameters.depth_cap_override is not None:
        return hyperparameters.depth_cap_override
    return 1 + math.ceil(math.log2(1.0 + stream_examples / hyperparameters.leaf_capacity))


def route(state: AFState, embedding: Tensor) -> RouteResult:
    """Route one frozen embedding without consulting labels or context metadata."""
    if embedding.ndim != 1:
        raise ValueError("routing expects one embedding vector")
    node = state.nodes[state.root_id]
    path = [node.node_id]
    evaluations = 0
    while not node.is_leaf:
        if node.split_direction is None or node.left_id is None or node.right_id is None:
            raise RuntimeError("internal node is missing a routing rule")
        score = torch.dot(node.split_direction.to(embedding.device), embedding)
        node = state.nodes[node.left_id if float(score.item()) <= float(node.split_threshold) else node.right_id]
        path.append(node.node_id)
        evaluations += 1
    return RouteResult(tuple(path), node.node_id, evaluations)


def effective_adapter(state: AFState, node_id: int) -> TopTwoAdapterState:
    """Sum committed top-two-layer deltas on the root-to-node path."""
    return sum_top_two_adapters(
        tuple(state.nodes[item].adapter for item in node_path(state, node_id)),
        state.base,
    )


def node_path(state: AFState, node_id: int) -> tuple[int, ...]:
    """Return one root-to-node ID path."""
    reverse = []
    current = state.nodes[node_id]
    while True:
        reverse.append(current.node_id)
        if current.parent_id is None:
            return tuple(reversed(reverse))
        current = state.nodes[current.parent_id]


def predict_for_node(
    state: AFState,
    table: StoredExampleTable,
    node_id: int,
    example_ids: Sequence[int],
) -> Tensor:
    """Return logits from the frozen base plus cumulative path deltas."""
    ids = torch.as_tensor(tuple(example_ids), dtype=torch.int64)
    adapter = effective_adapter(state, node_id)
    return top_two_logits(
        table.trunk_features[ids].to(adapter.embedding_weight.device),
        state.base,
        adapter,
    )


def update_microbatch(
    state: AFState,
    table: StoredExampleTable,
    example_ids: Sequence[int],
    hyperparameters: AFHyperparameters,
    seed: int,
) -> MicrobatchResult:
    """Train touched leaves, append arrivals, and return pre-structure state."""
    ids = tuple(int(example_id) for example_id in example_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("a microbatch must contain distinct stored example IDs")
    routes = tuple(route(state, table.embeddings[example_id]) for example_id in ids)
    grouped = {
        leaf_id: tuple(example_id for example_id, result in zip(ids, routes) if result.leaf_id == leaf_id)
        for leaf_id in sorted({result.leaf_id for result in routes})
    }
    updated = state
    replay_records = []
    online_presentations = 0
    adapter_evaluations = 0
    for leaf_id, arrivals in grouped.items():
        existing = tuple(updated.leaf_buffers[leaf_id])
        replay = _sample_replay(existing, len(arrivals), _derived_seed(seed, "online", leaf_id, ids[0]))
        replay_records.append((leaf_id, replay))
        training_ids = arrivals + replay
        node = updated.nodes[leaf_id]
        if node.optimizer is None:
            raise RuntimeError("destination leaf lacks optimizer state")
        trained_head, trained_optimizer, _loss = _train_step_for_node(
            updated, table, leaf_id, training_ids, node.adapter, node.optimizer, hyperparameters
        )
        updated = replace(
            updated,
            nodes=updated.nodes.set(leaf_id, replace(node, adapter=trained_head, optimizer=trained_optimizer)),
        )
        online_presentations += len(training_ids)
        adapter_evaluations += len(node_path(updated, leaf_id)) * len(training_ids)

    nodes = updated.nodes
    for path_result in routes:
        for node_id in path_result.path_ids:
            node = nodes[node_id]
            nodes = nodes.set(node_id, replace(node, total_arrivals=node.total_arrivals + 1))
    buffers = updated.leaf_buffers
    for leaf_id, arrivals in grouped.items():
        node = nodes[leaf_id]
        nodes = nodes.set(
            leaf_id,
            replace(
                node,
                arrivals_since_structure_change=node.arrivals_since_structure_change + len(arrivals),
            ),
        )
        buffers = buffers.set(leaf_id, buffers[leaf_id].extend(arrivals))
    counters = replace(
        updated.counters,
        embedding_evaluations=updated.counters.embedding_evaluations + len(ids),
        hyperplane_evaluations=(
            updated.counters.hyperplane_evaluations
            + sum(result.hyperplane_evaluations for result in routes)
        ),
        adapter_evaluations=updated.counters.adapter_evaluations + adapter_evaluations,
        online_training_examples=updated.counters.online_training_examples + online_presentations,
    )
    return MicrobatchResult(
        replace(updated, nodes=nodes, leaf_buffers=buffers, counters=counters),
        tuple(grouped),
        routes,
        tuple(replay_records),
    )


def structural_action(
    state: AFState,
    leaf_id: int,
    stream_examples: int,
    hyperparameters: AFHyperparameters,
) -> str | None:
    """Return the sole permitted structure action for one full current leaf."""
    if leaf_id not in state.nodes or not state.nodes[leaf_id].is_leaf:
        return None
    leaf = state.nodes[leaf_id]
    if leaf.arrivals_since_structure_change < hyperparameters.leaf_capacity:
        return None
    cap = current_depth_cap(stream_examples, hyperparameters)
    if leaf.depth < cap:
        return "split"
    if leaf.depth != cap or leaf.parent_id is None:
        return None
    parent = state.nodes[leaf.parent_id]
    if parent.left_id is None or parent.right_id is None:
        return None
    children = (state.nodes[parent.left_id], state.nodes[parent.right_id])
    if not all(child.is_leaf for child in children):
        return None
    subtree_size = sum(len(state.leaf_buffers[child.node_id]) for child in children)
    return "collapse" if subtree_size >= 2 * parent.last_consolidated_subtree_size else None


def install_split(
    state: AFState,
    table: StoredExampleTable,
    leaf_id: int,
    stream_step: int,
    hyperparameters: AFHyperparameters,
    seed: int,
) -> tuple[AFState, SplitEvent]:
    """Replace a leaf buffer with two zero-adapter children and a frozen rule."""
    leaf = state.nodes[leaf_id]
    if not leaf.is_leaf:
        raise ValueError("only a current leaf can split")
    example_ids = tuple(state.leaf_buffers[leaf_id])
    direction, threshold, fit_count = _pca_median_rule(
        table.embeddings,
        example_ids,
        min(hyperparameters.split_fit_samples, len(example_ids)),
        _derived_seed(seed, "split", leaf_id, stream_step),
    )
    scores = table.embeddings[torch.as_tensor(example_ids)] @ direction
    left_examples = tuple(example_id for example_id, score in zip(example_ids, scores) if float(score) <= threshold)
    right_examples = tuple(example_id for example_id, score in zip(example_ids, scores) if float(score) > threshold)
    if not left_examples or not right_examples or set(left_examples) & set(right_examples):
        raise RuntimeError("PCA-median split failed to produce two disjoint nonempty children")
    left_id, right_id = state.next_node_id, state.next_node_id + 1
    device = leaf.adapter.embedding_weight.device
    left_adapter = zero_top_two_adapter(state.base, device)
    right_adapter = zero_top_two_adapter(state.base, device)
    child = lambda node_id, members, offset: AFNode(
        node_id,
        leaf_id,
        leaf.depth + 1,
        left_adapter if offset == 0 else right_adapter,
        zero_top_two_adamw(left_adapter if offset == 0 else right_adapter),
        None,
        None,
        None,
        None,
        len(members),
        0,
        0,
        stream_step,
    )
    internal = replace(
        leaf,
        optimizer=None,
        split_direction=direction.cpu(),
        split_threshold=threshold,
        left_id=left_id,
        right_id=right_id,
        arrivals_since_structure_change=0,
    )
    nodes = state.nodes.set(leaf_id, internal).set(left_id, child(left_id, left_examples, 0)).set(
        right_id, child(right_id, right_examples, 1)
    )
    buffers = (
        state.leaf_buffers.remove(leaf_id)
        .set(left_id, pvector(left_examples))
        .set(right_id, pvector(right_examples))
    )
    counters = replace(
        state.counters,
        historical_examples_repartitioned=(
            state.counters.historical_examples_repartitioned + len(example_ids)
        ),
        pca_fit_examples=state.counters.pca_fit_examples + fit_count,
    )
    split_state = replace(
        state,
        nodes=nodes,
        leaf_buffers=buffers,
        next_node_id=right_id + 1,
        counters=counters,
    )
    return split_state, SplitEvent(
        leaf_id, left_id, right_id, left_examples, right_examples, stream_step
    )


def initialize_split_children(
    state: AFState,
    table: StoredExampleTable,
    event: SplitEvent,
    hyperparameters: AFHyperparameters,
    seed: int,
) -> AFState:
    """Replay-train both zero children for the configured fixed epoch count."""
    updated = state
    presentations = 0
    adapter_evaluations = 0
    for child_id in (event.left_id, event.right_id):
        members = tuple(updated.leaf_buffers[child_id])
        node = updated.nodes[child_id]
        if node.optimizer is None:
            raise RuntimeError("new split child lacks optimizer state")
        head, optimizer, _loss, count = _train_epochs_for_node(
            updated,
            table,
            child_id,
            members,
            node.adapter,
            node.optimizer,
            hyperparameters.split_epochs,
            hyperparameters,
            _derived_seed(seed, "split-init", child_id, event.stream_step),
        )
        updated = replace(
            updated,
            nodes=updated.nodes.set(child_id, replace(node, adapter=head, optimizer=optimizer)),
        )
        presentations += count
        adapter_evaluations += len(node_path(updated, child_id)) * count
    return replace(
        updated,
        counters=replace(
            updated.counters,
            split_replay_examples=updated.counters.split_replay_examples + presentations,
            adapter_evaluations=updated.counters.adapter_evaluations + adapter_evaluations,
        ),
    )


def collapse_leaf_pair(
    state: AFState,
    table: StoredExampleTable,
    triggering_leaf_id: int,
    stream_step: int,
    hyperparameters: AFHyperparameters,
    seed: int,
) -> tuple[AFState, ConsolidationEvent]:
    """Replay-fit a parent replacement and atomically remove its two leaf children."""
    leaf = state.nodes[triggering_leaf_id]
    if leaf.parent_id is None:
        raise ValueError("the root cannot trigger sibling collapse")
    parent = state.nodes[leaf.parent_id]
    if parent.left_id is None or parent.right_id is None:
        raise ValueError("collapse parent is not internal")
    child_ids = (parent.left_id, parent.right_id)
    if not all(state.nodes[child_id].is_leaf for child_id in child_ids):
        raise ValueError("collapse requires two leaf children")
    members = tuple(
        sorted(
            tuple(state.leaf_buffers[child_ids[0]]) + tuple(state.leaf_buffers[child_ids[1]]),
            key=lambda example_id: (int(table.stream_steps[example_id]), example_id),
        )
    )
    replacement_optimizer = zero_top_two_adamw(parent.adapter)
    replacement_head, replacement_optimizer, _loss, presentations = _train_epochs_for_node(
        state,
        table,
        parent.node_id,
        members,
        parent.adapter,
        replacement_optimizer,
        hyperparameters.consolidation_epochs,
        hyperparameters,
        _derived_seed(seed, "collapse", parent.node_id, stream_step),
    )
    collapsed_parent = replace(
        parent,
        adapter=replacement_head,
        optimizer=replacement_optimizer,
        split_direction=None,
        split_threshold=None,
        left_id=None,
        right_id=None,
        arrivals_since_structure_change=0,
        last_consolidated_subtree_size=len(members),
    )
    nodes = state.nodes.set(parent.node_id, collapsed_parent)
    buffers = state.leaf_buffers.set(parent.node_id, pvector(members))
    for child_id in child_ids:
        nodes = nodes.remove(child_id)
        buffers = buffers.remove(child_id)
    collapsed = replace(
        state,
        nodes=nodes,
        leaf_buffers=buffers,
        counters=replace(
            state.counters,
            consolidation_replay_examples=(
                state.counters.consolidation_replay_examples + presentations
            ),
            adapter_evaluations=(
                state.counters.adapter_evaluations
                + len(node_path(replace(state, nodes=nodes), parent.node_id)) * presentations
            ),
        ),
    )
    return collapsed, ConsolidationEvent(parent.node_id, child_ids, members, stream_step)


def validate_af_state(state: AFState, arrived_example_ids: Iterable[int] | None = None) -> None:
    """Raise if reachability, parent links, depths, or leaf ownership changed."""
    if state.root_id not in state.nodes or state.nodes[state.root_id].parent_id is not None:
        raise ValueError("AF root is missing or has a parent")
    reachable = set()
    pending = [state.root_id]
    while pending:
        node_id = pending.pop()
        if node_id in reachable:
            raise ValueError("AF tree contains a cycle or duplicate child reference")
        reachable.add(node_id)
        node = state.nodes[node_id]
        if node.is_leaf:
            if node_id not in state.leaf_buffers:
                raise ValueError("leaf buffer is missing")
            continue
        if node.left_id is None or node.right_id is None:
            raise ValueError("internal node is missing children")
        for child_id in (node.left_id, node.right_id):
            child = state.nodes[child_id]
            if child.parent_id != node_id or child.depth != node.depth + 1:
                raise ValueError("child parent or depth metadata changed")
            pending.append(child_id)
    if reachable != set(state.nodes) or set(state.leaf_buffers) != {
        node_id for node_id, node in state.nodes.items() if node.is_leaf
    }:
        raise ValueError("unreachable nodes or non-leaf buffers exist")
    memberships = tuple(example_id for buffer in state.leaf_buffers.values() for example_id in buffer)
    if len(set(memberships)) != len(memberships):
        raise ValueError("an example appears in more than one leaf buffer")
    if arrived_example_ids is not None and set(memberships) != set(arrived_example_ids):
        raise ValueError("leaf buffers do not exhaust arrived examples")


def _sample_replay(existing: Sequence[int], count: int, seed: int) -> tuple[int, ...]:
    if not existing or count == 0:
        return ()
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(existing), size=count, replace=len(existing) < count)
    return tuple(int(existing[int(index)]) for index in indices)


def _derived_seed(seed: int, *parts: object) -> int:
    payload = "\0".join((str(seed), *(str(part) for part in parts))).encode("utf-8")
    return int.from_bytes(sha256(payload).digest()[:8], "big")


def _pca_median_rule(
    embeddings: Tensor,
    example_ids: Sequence[int],
    fit_count: int,
    seed: int,
) -> tuple[Tensor, float, int]:
    if len(example_ids) < 2 or fit_count < 2:
        raise ValueError("PCA split needs at least two examples")
    rng = np.random.default_rng(seed)
    selected = np.asarray(example_ids, dtype=np.int64)
    if selected.shape[0] > fit_count:
        selected = rng.choice(selected, size=fit_count, replace=False)
    matrix = embeddings[torch.from_numpy(selected)].numpy().astype(np.float64, copy=False)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(1, centered.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    direction = eigenvectors[:, int(np.argmax(eigenvalues))]
    pivot = int(np.argmax(np.abs(direction)))
    if direction[pivot] < 0.0:
        direction = -direction
    if not np.isfinite(direction).all() or float(np.linalg.norm(direction)) <= 0.0:
        raise RuntimeError("PCA produced a non-finite or zero split direction")
    direction = direction / np.linalg.norm(direction)
    all_matrix = embeddings[torch.as_tensor(tuple(example_ids), dtype=torch.int64)].numpy()
    threshold = float(np.median(all_matrix @ direction))
    if not math.isfinite(threshold):
        raise RuntimeError("PCA produced a non-finite split threshold")
    return torch.from_numpy(direction.astype(np.float32)), threshold, int(selected.shape[0])


def _train_step_for_node(
    state: AFState,
    table: StoredExampleTable,
    node_id: int,
    example_ids: Sequence[int],
    adapter: TopTwoAdapterState,
    optimizer: TopTwoAdamWState,
    hyperparameters: AFHyperparameters,
) -> tuple[TopTwoAdapterState, TopTwoAdamWState, float]:
    if not example_ids:
        return adapter, optimizer, math.nan
    ids = torch.as_tensor(tuple(example_ids), dtype=torch.int64)
    device = adapter.embedding_weight.device
    trunk_features = table.trunk_features[ids].to(device)
    labels = table.labels[ids].to(device)
    path = node_path(state, node_id)
    if path[-1] != node_id:
        raise RuntimeError("adapter target is absent from its own ancestry path")
    ancestor_ids = path[:-1]
    fixed_adapter = sum_top_two_adapters(
        tuple(state.nodes[ancestor_id].adapter for ancestor_id in ancestor_ids),
        state.base,
    )
    return train_top_two_adapter_step(
        trunk_features,
        labels,
        state.base,
        fixed_adapter,
        adapter,
        optimizer,
        TopTwoOptimizerConfig(
            hyperparameters.adapter_lr,
            hyperparameters.weight_decay,
            hyperparameters.beta1,
            hyperparameters.beta2,
            hyperparameters.epsilon,
        ),
    )


def _train_epochs_for_node(
    state: AFState,
    table: StoredExampleTable,
    node_id: int,
    example_ids: Sequence[int],
    adapter: TopTwoAdapterState,
    optimizer: TopTwoAdamWState,
    epochs: int,
    hyperparameters: AFHyperparameters,
    seed: int,
) -> tuple[TopTwoAdapterState, TopTwoAdamWState, float, int]:
    current_adapter, current_optimizer, final_loss, presentations = adapter, optimizer, math.nan, 0
    for epoch in range(epochs):
        order = np.random.default_rng(_derived_seed(seed, epoch)).permutation(len(example_ids))
        ordered = tuple(int(example_ids[int(index)]) for index in order)
        for offset in range(0, len(ordered), hyperparameters.batch_size):
            batch = ordered[offset : offset + hyperparameters.batch_size]
            current_adapter, current_optimizer, final_loss = _train_step_for_node(
                state,
                table,
                node_id,
                batch,
                current_adapter,
                current_optimizer,
                hyperparameters,
            )
            presentations += len(batch)
    return current_adapter, current_optimizer, final_loss, presentations


__all__ = [
    "AFHyperparameters",
    "AFNode",
    "AFState",
    "ConsolidationEvent",
    "MicrobatchResult",
    "RouteResult",
    "SplitEvent",
    "StoredExampleTable",
    "WorkCounters",
    "collapse_leaf_pair",
    "current_depth_cap",
    "effective_adapter",
    "init_af_state",
    "initialize_split_children",
    "install_split",
    "node_path",
    "predict_for_node",
    "route",
    "structural_action",
    "update_microbatch",
    "validate_af_state",
]

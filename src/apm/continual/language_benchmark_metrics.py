"""Pure language benchmark decomposition and memory accounting."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import cast

import numpy as np

from apm.continual.language_tasks import AddressBook
from apm.lm.lora import LoraConfig, LoraEdge
from apm.lm.lora_memory import PackedLoraMemory
from apm.memory.dense import tree_nbytes, tree_parameter_count
from apm.memory.graph import (
    MemoryGraph,
    memory_node_path,
    path_incidence_matrix,
)


@dataclass(frozen=True)
class StageTaskObservation:
    """One task's stored and routed competence observed after one stage."""

    stage_index: int
    task_id: str
    stored_nll: float
    routed_nll: float
    best_node_nll: float
    base_checksum: str
    committed_path_checksum: str

    def __post_init__(self) -> None:
        if type(self.stage_index) is not int or self.stage_index <= 0:
            raise ValueError("stage_index must be a positive integer")
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError("task_id must be a nonempty string")
        nll_values = (self.stored_nll, self.routed_nll, self.best_node_nll)
        if any(not math.isfinite(value) or value < 0.0 for value in nll_values):
            raise ValueError("stage/task NLL values must be finite and nonnegative")
        if self.routed_nll + 1e-7 < self.best_node_nll:
            raise ValueError("best-node NLL cannot exceed routed NLL")
        if any(
            not isinstance(checksum, str) or not checksum
            for checksum in (self.base_checksum, self.committed_path_checksum)
        ):
            raise ValueError("stage/task checksums must be nonempty strings")


@dataclass(frozen=True)
class TransferObservation:
    """Pretraining parent comparison and fixed-budget curve for one new task."""

    stage_index: int
    task_id: str
    root_initial_nll: float
    selected_parent_initial_nll: float
    learning_curve_nll: tuple[float, ...]
    tokens_per_update: int

    def __post_init__(self) -> None:
        if type(self.stage_index) is not int or self.stage_index <= 0:
            raise ValueError("stage_index must be a positive integer")
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError("task_id must be a nonempty string")
        if any(
            not math.isfinite(value) or value < 0.0
            for value in (self.root_initial_nll, self.selected_parent_initial_nll)
        ):
            raise ValueError("initial transfer NLL values must be nonnegative")
        if len(self.learning_curve_nll) < 2 or any(
            not math.isfinite(value) or value < 0.0
            for value in self.learning_curve_nll
        ):
            raise ValueError("learning curve must contain initial and updated NLL values")
        if not math.isclose(
            self.learning_curve_nll[0],
            self.selected_parent_initial_nll,
            abs_tol=1e-7,
        ):
            raise ValueError("learning curve must begin at selected-parent initial NLL")
        if type(self.tokens_per_update) is not int or self.tokens_per_update <= 0:
            raise ValueError("tokens_per_update must be a positive integer")


@dataclass(frozen=True)
class StoredMetricRow:
    """Stored competence, best-so-far forgetting, and immutability drift."""

    stage_index: int
    task_id: str
    stored_nll: float
    best_so_far_stored_nll: float
    stored_forgetting: float
    base_checksum: str
    base_checksum_drift: bool
    committed_path_checksum: str
    committed_path_checksum_drift: bool


@dataclass(frozen=True)
class RoutingMetricRow:
    """Routed competence, distinct forgetting, and both regret references."""

    stage_index: int
    task_id: str
    routed_nll: float
    best_so_far_routed_nll: float
    routing_forgetting: float
    task_oracle_nll: float
    task_oracle_regret: float
    best_node_nll: float
    best_node_regret: float


@dataclass(frozen=True)
class TransferMetricRow:
    """Root-parent advantage and threshold counts for one learning curve."""

    stage_index: int
    task_id: str
    root_initial_nll: float
    selected_parent_initial_nll: float
    parent_advantage: float
    learning_curve_nll: tuple[float, ...]
    first_step_improvement: float
    fixed_budget_improvement: float
    update_budget: int
    nll_threshold: float
    updates_to_threshold: int | None
    tokens_to_threshold: int | None


@dataclass(frozen=True)
class BenchmarkMetricDecomposition:
    """Synchronized stored, routing, and transfer rows for report writers."""

    stored: tuple[StoredMetricRow, ...]
    routing: tuple[RoutingMetricRow, ...]
    transfer: tuple[TransferMetricRow, ...]

    def __post_init__(self) -> None:
        if len(self.stored) != len(self.routing):
            raise ValueError("stored and routing rows must remain aligned")
        if any(
            (stored.stage_index, stored.task_id)
            != (routing.stage_index, routing.task_id)
            for stored, routing in zip(self.stored, self.routing)
        ):
            raise ValueError("stored and routing row identities must remain aligned")


def decompose_benchmark_metrics(
    stage_task_observations: tuple[StageTaskObservation, ...],
    transfer_observations: tuple[TransferObservation, ...],
    *,
    nll_threshold: float,
) -> BenchmarkMetricDecomposition:
    """Compute forgetting, regret, transfer, threshold, and checksum fields."""
    if not stage_task_observations:
        raise ValueError("stage_task_observations must not be empty")
    if not math.isfinite(nll_threshold) or nll_threshold < 0.0:
        raise ValueError("nll_threshold must be finite and nonnegative")
    stage_identities = tuple(
        (observation.stage_index, observation.task_id)
        for observation in stage_task_observations
    )
    if len(set(stage_identities)) != len(stage_identities):
        raise ValueError("stage/task observations must have unique identities")
    introduction_identities = _validate_curriculum_prefix_order(
        stage_task_observations
    )
    transfer_identities = tuple(
        (observation.stage_index, observation.task_id)
        for observation in transfer_observations
    )
    if len(set(transfer_identities)) != len(transfer_identities):
        raise ValueError("transfer observations must have unique identities")
    if transfer_identities and transfer_identities != introduction_identities:
        raise ValueError("transfer observations must match task introduction order")

    base_checksum_reference = stage_task_observations[0].base_checksum
    stored_best: dict[str, float] = {}
    routed_best: dict[str, float] = {}
    path_checksum_reference: dict[str, str] = {}
    stored_rows: list[StoredMetricRow] = []
    routing_rows: list[RoutingMetricRow] = []
    for observation in stage_task_observations:
        best_stored = min(
            stored_best.get(observation.task_id, math.inf),
            observation.stored_nll,
        )
        best_routed = min(
            routed_best.get(observation.task_id, math.inf),
            observation.routed_nll,
        )
        reference_path_checksum = path_checksum_reference.setdefault(
            observation.task_id,
            observation.committed_path_checksum,
        )
        stored_rows.append(
            StoredMetricRow(
                stage_index=observation.stage_index,
                task_id=observation.task_id,
                stored_nll=observation.stored_nll,
                best_so_far_stored_nll=best_stored,
                stored_forgetting=observation.stored_nll - best_stored,
                base_checksum=observation.base_checksum,
                base_checksum_drift=(
                    observation.base_checksum != base_checksum_reference
                ),
                committed_path_checksum=observation.committed_path_checksum,
                committed_path_checksum_drift=(
                    observation.committed_path_checksum != reference_path_checksum
                ),
            )
        )
        routing_rows.append(
            RoutingMetricRow(
                stage_index=observation.stage_index,
                task_id=observation.task_id,
                routed_nll=observation.routed_nll,
                best_so_far_routed_nll=best_routed,
                routing_forgetting=observation.routed_nll - best_routed,
                task_oracle_nll=observation.stored_nll,
                task_oracle_regret=(
                    observation.routed_nll - observation.stored_nll
                ),
                best_node_nll=observation.best_node_nll,
                best_node_regret=(
                    observation.routed_nll - observation.best_node_nll
                ),
            )
        )
        stored_best[observation.task_id] = best_stored
        routed_best[observation.task_id] = best_routed

    transfer_rows = tuple(
        _decompose_transfer(observation, nll_threshold)
        for observation in transfer_observations
    )
    return BenchmarkMetricDecomposition(
        stored=tuple(stored_rows),
        routing=tuple(routing_rows),
        transfer=transfer_rows,
    )


@dataclass(frozen=True)
class ProjectionEffectiveRanks:
    """Uncompressed pathwise rank upper bound at each supported projection."""

    query: int
    key: int
    value: int
    attention_output: int
    mlp_input: int
    mlp_output: int

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value < 0
            for value in (
                self.query,
                self.key,
                self.value,
                self.attention_output,
                self.mlp_input,
                self.mlp_output,
            )
        ):
            raise ValueError("effective projection ranks must be nonnegative integers")


@dataclass(frozen=True)
class EffectivePathMemory:
    """Logical LoRA storage and rank along one root-to-node path."""

    node_id: str
    edge_count: int
    lora_bytes: int
    effective_model_bytes: int
    projection_ranks: ProjectionEffectiveRanks

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("effective-path node_id must not be empty")
        if any(
            type(value) is not int or value < 0
            for value in (
                self.edge_count,
                self.lora_bytes,
                self.effective_model_bytes,
            )
        ):
            raise ValueError("effective-path counts and bytes must be nonnegative")
        if self.effective_model_bytes < self.lora_bytes:
            raise ValueError("effective-model bytes must include path LoRA bytes")
        if not isinstance(self.projection_ranks, ProjectionEffectiveRanks):
            raise TypeError("projection_ranks must be ProjectionEffectiveRanks")


@dataclass(frozen=True)
class LanguageMemoryAccounting:
    """Persistent logical storage, padded runtime storage, and optimizer peak."""

    base_parameter_count: int
    base_bytes: int
    edge_bytes: tuple[int, ...]
    committed_lora_bytes: int
    effective_paths: tuple[EffectivePathMemory, ...]
    address_key_bytes_per_node: tuple[int, ...]
    address_key_bytes: int
    graph_metadata_bytes: int
    persistent_bytes: int
    packed_edge_bank_bytes: int
    packed_path_matrix_bytes: int
    packed_validity_mask_bytes: int
    packed_runtime_bytes: int
    packed_padding_bytes: int
    optimizer_peak_bytes: int
    completed_task_count: int
    bytes_per_task: float | None
    nll_improvement: float | None
    bytes_per_nll_improvement: float | None

    def __post_init__(self) -> None:
        integer_fields = (
            self.base_parameter_count,
            self.base_bytes,
            self.committed_lora_bytes,
            self.address_key_bytes,
            self.graph_metadata_bytes,
            self.persistent_bytes,
            self.packed_edge_bank_bytes,
            self.packed_path_matrix_bytes,
            self.packed_validity_mask_bytes,
            self.packed_runtime_bytes,
            self.packed_padding_bytes,
            self.optimizer_peak_bytes,
            self.completed_task_count,
        )
        if any(type(value) is not int or value < 0 for value in integer_fields):
            raise ValueError("memory byte and count fields must be nonnegative integers")
        if any(type(value) is not int or value < 0 for value in self.edge_bytes):
            raise ValueError("edge_bytes must contain nonnegative integers")
        if self.committed_lora_bytes != sum(self.edge_bytes):
            raise ValueError("committed LoRA bytes must equal the edge-byte sum")
        if self.completed_task_count != len(self.edge_bytes):
            raise ValueError("completed task count must equal committed edge count")
        if len(self.effective_paths) != self.completed_task_count + 1:
            raise ValueError("effective paths must contain root plus one row per task")
        if self.address_key_bytes != sum(self.address_key_bytes_per_node):
            raise ValueError("address-key bytes must equal the per-node byte sum")
        if self.packed_padding_bytes > self.packed_runtime_bytes:
            raise ValueError("packed padding cannot exceed packed runtime storage")
        if self.nll_improvement is not None and not math.isfinite(
            self.nll_improvement
        ):
            raise ValueError("nll_improvement must be finite when provided")
        expected_persistent = (
            self.base_bytes
            + self.committed_lora_bytes
            + self.address_key_bytes
            + self.graph_metadata_bytes
        )
        if self.persistent_bytes != expected_persistent:
            raise ValueError("persistent bytes must count base, edges, keys, and metadata")
        expected_runtime = (
            self.packed_edge_bank_bytes
            + self.packed_path_matrix_bytes
            + self.packed_validity_mask_bytes
        )
        if self.packed_runtime_bytes != expected_runtime:
            raise ValueError("packed runtime component bytes must sum exactly")
        adaptation_bytes = self.persistent_bytes - self.base_bytes
        expected_bytes_per_task = (
            None
            if self.completed_task_count == 0
            else adaptation_bytes / self.completed_task_count
        )
        if not _optional_float_matches(self.bytes_per_task, expected_bytes_per_task):
            raise ValueError("bytes_per_task must divide adaptation storage by tasks")
        expected_bytes_per_improvement = (
            None
            if self.nll_improvement is None or self.nll_improvement <= 0.0
            else adaptation_bytes / self.nll_improvement
        )
        if not _optional_float_matches(
            self.bytes_per_nll_improvement,
            expected_bytes_per_improvement,
        ):
            raise ValueError("bytes-per-NLL must use positive improvement only")


def account_language_memory(
    base_params: object,
    graph: MemoryGraph[LoraEdge],
    address_book: AddressBook,
    packed_memory: PackedLoraMemory,
    lora_config: LoraConfig,
    *,
    optimizer_state_snapshots: tuple[object, ...] = (),
    nll_improvement: float | None = None,
) -> LanguageMemoryAccounting:
    """Account logical persistent bytes separately from padded runtime bytes."""
    _validate_memory_alignment(graph, address_book, packed_memory)
    if not isinstance(lora_config, LoraConfig):
        raise TypeError("lora_config must be a LoraConfig")
    if nll_improvement is not None and not math.isfinite(nll_improvement):
        raise ValueError("nll_improvement must be finite when provided")

    base_bytes = tree_nbytes(base_params)
    non_root_nodes = tuple(node for node in graph.nodes if node.parent_id is not None)
    edge_bytes = tuple(
        tree_nbytes(cast(LoraEdge, node.incoming_edge))
        for node in non_root_nodes
    )
    committed_lora_bytes = sum(edge_bytes)
    path_records = tuple(
        _effective_path_memory(graph, node.node_id, base_bytes, lora_config)
        for node in graph.nodes
    )
    valid_key_count = len(graph.nodes)
    key_bytes_per_node = address_book.keys.shape[1] * address_book.keys.dtype.itemsize
    address_key_bytes_per_node = (key_bytes_per_node,) * valid_key_count
    address_key_bytes = sum(address_key_bytes_per_node)
    metadata_bytes = graph_metadata_nbytes(graph)
    persistent_bytes = (
        base_bytes + committed_lora_bytes + address_key_bytes + metadata_bytes
    )

    packed_edge_bank_bytes = tree_nbytes(packed_memory.edge_bank)
    packed_path_matrix_bytes = int(np.asarray(packed_memory.node_path_matrix).nbytes)
    packed_validity_mask_bytes = int(
        np.asarray(packed_memory.valid_node_mask).nbytes
        + np.asarray(packed_memory.valid_edge_mask).nbytes
    )
    packed_runtime_bytes = (
        packed_edge_bank_bytes
        + packed_path_matrix_bytes
        + packed_validity_mask_bytes
    )
    logical_path_matrix_bytes = (
        len(graph.nodes)
        * len(non_root_nodes)
        * np.asarray(packed_memory.node_path_matrix).dtype.itemsize
    )
    packed_padding_bytes = (
        packed_edge_bank_bytes
        - committed_lora_bytes
        + packed_path_matrix_bytes
        - logical_path_matrix_bytes
    )
    optimizer_peak_bytes = max(
        (tree_nbytes(snapshot) for snapshot in optimizer_state_snapshots),
        default=0,
    )
    task_count = len(non_root_nodes)
    adaptation_bytes = persistent_bytes - base_bytes
    bytes_per_task = None if task_count == 0 else adaptation_bytes / task_count
    bytes_per_nll_improvement = (
        None
        if nll_improvement is None or nll_improvement <= 0.0
        else adaptation_bytes / nll_improvement
    )
    return LanguageMemoryAccounting(
        base_parameter_count=tree_parameter_count(base_params),
        base_bytes=base_bytes,
        edge_bytes=edge_bytes,
        committed_lora_bytes=committed_lora_bytes,
        effective_paths=path_records,
        address_key_bytes_per_node=address_key_bytes_per_node,
        address_key_bytes=address_key_bytes,
        graph_metadata_bytes=metadata_bytes,
        persistent_bytes=persistent_bytes,
        packed_edge_bank_bytes=packed_edge_bank_bytes,
        packed_path_matrix_bytes=packed_path_matrix_bytes,
        packed_validity_mask_bytes=packed_validity_mask_bytes,
        packed_runtime_bytes=packed_runtime_bytes,
        packed_padding_bytes=packed_padding_bytes,
        optimizer_peak_bytes=optimizer_peak_bytes,
        completed_task_count=task_count,
        bytes_per_task=bytes_per_task,
        nll_improvement=nll_improvement,
        bytes_per_nll_improvement=bytes_per_nll_improvement,
    )


def graph_metadata_nbytes(graph: MemoryGraph[LoraEdge]) -> int:
    """Return canonical compact UTF-8 bytes for payload-free graph metadata."""
    payload = {
        "nodes": [
            {
                "node_id": str(node.node_id),
                "parent_id": None if node.parent_id is None else str(node.parent_id),
                "trained_task": (
                    None if node.trained_task is None else str(node.trained_task)
                ),
                "train_stage": node.train_stage,
                "depth": node.depth,
            }
            for node in graph.nodes
        ]
    }
    return len(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    )


def _decompose_transfer(
    observation: TransferObservation,
    nll_threshold: float,
) -> TransferMetricRow:
    threshold_updates = next(
        (
            update
            for update, nll in enumerate(observation.learning_curve_nll)
            if nll <= nll_threshold
        ),
        None,
    )
    return TransferMetricRow(
        stage_index=observation.stage_index,
        task_id=observation.task_id,
        root_initial_nll=observation.root_initial_nll,
        selected_parent_initial_nll=observation.selected_parent_initial_nll,
        parent_advantage=(
            observation.root_initial_nll - observation.selected_parent_initial_nll
        ),
        learning_curve_nll=observation.learning_curve_nll,
        first_step_improvement=(
            observation.learning_curve_nll[0] - observation.learning_curve_nll[1]
        ),
        fixed_budget_improvement=(
            observation.learning_curve_nll[0] - observation.learning_curve_nll[-1]
        ),
        update_budget=len(observation.learning_curve_nll) - 1,
        nll_threshold=nll_threshold,
        updates_to_threshold=threshold_updates,
        tokens_to_threshold=(
            None
            if threshold_updates is None
            else threshold_updates * observation.tokens_per_update
        ),
    )


def _validate_curriculum_prefix_order(
    observations: tuple[StageTaskObservation, ...],
) -> tuple[tuple[int, str], ...]:
    stage_indices = tuple(
        dict.fromkeys(observation.stage_index for observation in observations)
    )
    if stage_indices != tuple(range(stage_indices[0], stage_indices[-1] + 1)):
        raise ValueError("stage/task observations must use contiguous stage indices")
    curriculum_order: list[str] = []
    introductions: list[tuple[int, str]] = []
    for stage_index in stage_indices:
        stage_task_ids = tuple(
            observation.task_id
            for observation in observations
            if observation.stage_index == stage_index
        )
        new_task_ids = tuple(
            task_id for task_id in stage_task_ids if task_id not in curriculum_order
        )
        if len(new_task_ids) != 1:
            raise ValueError("each stage must introduce exactly one new task")
        curriculum_order.append(new_task_ids[0])
        introductions.append((stage_index, new_task_ids[0]))
        if stage_task_ids != tuple(curriculum_order):
            raise ValueError(
                "each stage must cover the stable first-seen curriculum prefix"
            )
    return tuple(introductions)


def _effective_path_memory(
    graph: MemoryGraph[LoraEdge],
    node_id: str,
    base_bytes: int,
    lora_config: LoraConfig,
) -> EffectivePathMemory:
    path = memory_node_path(graph, node_id)
    path_edges = tuple(
        cast(LoraEdge, node.incoming_edge)
        for node in path
        if node.parent_id is not None
    )
    edge_count = len(path_edges)
    path_bytes = sum(tree_nbytes(edge) for edge in path_edges)
    target_mask = lora_config.target_mask
    rank = lora_config.rank * edge_count
    projection_ranks = ProjectionEffectiveRanks(
        **{
            name: rank if getattr(target_mask, name) else 0
            for name in ProjectionEffectiveRanks.__dataclass_fields__
        }
    )
    return EffectivePathMemory(
        node_id=str(node_id),
        edge_count=edge_count,
        lora_bytes=path_bytes,
        effective_model_bytes=base_bytes + path_bytes,
        projection_ranks=projection_ranks,
    )


def _validate_memory_alignment(
    graph: MemoryGraph[LoraEdge],
    address_book: AddressBook,
    packed_memory: PackedLoraMemory,
) -> None:
    if not isinstance(graph, MemoryGraph) or not graph.nodes:
        raise ValueError("graph must be a nonempty MemoryGraph")
    if not isinstance(address_book, AddressBook):
        raise TypeError("address_book must be an AddressBook")
    if not isinstance(packed_memory, PackedLoraMemory):
        raise TypeError("packed_memory must be PackedLoraMemory")
    non_root_nodes = tuple(node for node in graph.nodes if node.parent_id is not None)
    if any(not isinstance(node.incoming_edge, LoraEdge) for node in non_root_nodes):
        raise TypeError("every non-root graph node must contain a LoraEdge")
    node_capacity, edge_capacity = packed_memory.node_path_matrix.shape
    if address_book.max_nodes != node_capacity:
        raise ValueError("address-book and packed node capacities must match")
    if packed_memory.valid_edge_mask.shape != (edge_capacity,):
        raise ValueError("packed edge mask must match edge capacity")
    expected_node_mask = np.arange(node_capacity) < len(graph.nodes)
    expected_edge_mask = np.arange(edge_capacity) < len(non_root_nodes)
    if not np.array_equal(address_book.valid_node_mask, expected_node_mask):
        raise ValueError("address-book valid nodes must align with graph insertion order")
    if not np.array_equal(
        np.asarray(packed_memory.valid_node_mask),
        expected_node_mask,
    ) or not np.array_equal(
        np.asarray(packed_memory.valid_edge_mask),
        expected_edge_mask,
    ):
        raise ValueError("packed validity masks must align with the graph")
    graph_node_ids = tuple(node.node_id for node in graph.nodes)
    if address_book.node_ids[: len(graph.nodes)] != graph_node_ids:
        raise ValueError("address-book node IDs must align with graph insertion order")
    expected_incidence = path_incidence_matrix(graph)
    packed_incidence = np.asarray(packed_memory.node_path_matrix)
    if not np.array_equal(
        packed_incidence[: len(graph.nodes), : len(non_root_nodes)],
        expected_incidence,
    ):
        raise ValueError("packed path incidence must match graph semantics")
    if np.any(packed_incidence[len(graph.nodes) :, :] != 0.0) or np.any(
        packed_incidence[:, len(non_root_nodes) :] != 0.0
    ):
        raise ValueError("packed path padding must be zero")


def _optional_float_matches(
    actual: float | None,
    expected: float | None,
) -> bool:
    return (
        actual is None and expected is None
        or actual is not None
        and expected is not None
        and math.isclose(actual, expected, rel_tol=1e-12)
    )

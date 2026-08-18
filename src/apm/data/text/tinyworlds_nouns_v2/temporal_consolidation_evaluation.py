"""Prefix-isolated routing and streamed suffix evaluation for temporal banks."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import partial
import math
from typing import Literal, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np

from apm.continual.language_tasks import RouterBatch
from apm.data.text.tinyworlds_nouns_v1.experiment import (
    IndexedStoryStore,
    StoryIndexEntry,
)
from apm.data.text.tinyworlds_nouns_v2.contracts import require_sha256
from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_contracts import (
    EVALUATION_BATCH_SIZE,
    EVALUATION_ROW_FORMAT,
    LORA_ALPHA,
    LORA_RANK,
    TASK_IDS,
)
from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_io import (
    ChainedJsonlLedger,
)
from apm.lm.config import GptNeoConfig
from apm.lm.gpt_neo import apply_gpt_neo
from apm.lm.lora import LoraConfig, LoraEdge
from apm.lm.lora_memory import (
    PackedLoraMemory,
    edge_coefficients_for_node,
    pack_lora_memory,
)
from apm.lm.losses import per_token_nll
from apm.lm.parameters import GptNeoParams
from apm.lm.text_data import TokenBatch
from apm.memory.graph import NodeId, TaskId, add_memory_node, init_memory_graph
from apm.memory.prefix_energy import exhaustive_prefix_nll_core


RoutingMode: TypeAlias = Literal["exhaustive", "forced_base", "forced_adapter"]
DatasetKind: TypeAlias = Literal[
    "sentinel",
    "macro",
    "final",
    "merge_source",
    "merge_validation",
    "timing",
]
EvaluationProgress = Callable[[int, int, dict[str, float]], None]
# Calibrated so the largest eight-row timing shape stays below 9 GiB on the
# production RTX 4090.  The next prefix-width bucket drops to four rows before
# XLA's GEMM autotuner needs a transient allocation that crosses the 12 GiB
# study gate.
_PREFIX_ROUTER_WORK_BUDGET = 20_736
_PREFIX_ROUTER_ROW_BUCKETS = (8, 4, 2, 1)


@dataclass(frozen=True, slots=True)
class AdapterCandidate:
    """One standalone adapter and its non-router descriptive metadata."""

    candidate_id: str
    adapter_sha256: str
    adapter: LoraEdge
    task_counts: tuple[tuple[str, int], ...]
    level: int | None = None
    start_arrival: int | None = None
    end_arrival: int | None = None

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("adapter candidate ID must be nonempty")
        require_sha256(self.adapter_sha256, "adapter candidate")
        if tuple(task for task, _ in self.task_counts) != tuple(
            task for task in TASK_IDS if task in dict(self.task_counts)
        ) or any(count <= 0 for _, count in self.task_counts):
            raise ValueError("adapter candidate task counts are not canonical")
        interval = (self.level, self.start_arrival, self.end_arrival)
        if any(value is not None for value in interval) and any(
            type(value) is not int or value < 0 for value in interval
        ):
            raise ValueError("adapter candidate interval metadata is incomplete")


@dataclass(frozen=True, slots=True)
class AdapterBank:
    """Base plus insertion-ordered standalone root adapters."""

    candidates: tuple[AdapterCandidate, ...]
    packed: PackedLoraMemory
    candidate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.candidate_ids != ("base",) + tuple(
            candidate.candidate_id for candidate in self.candidates
        ):
            raise ValueError("adapter bank candidate order changed")
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("adapter bank candidate IDs must be unique")


@dataclass(frozen=True, slots=True)
class MidpointCase:
    """One story split into a router-only prefix and evaluator-only suffix."""

    task_id: str
    entry: StoryIndexEntry
    midpoint: int
    prefix: RouterBatch
    suffix_windows: TokenBatch

    def __post_init__(self) -> None:
        if self.task_id not in TASK_IDS or self.midpoint < 2:
            raise ValueError("midpoint case task or split is invalid")
        if self.prefix.input_ids.shape[0] != 1:
            raise ValueError("midpoint case prefixes require one row")
        if not np.any(self.suffix_windows.loss_mask):
            raise ValueError("midpoint case suffix contains no targets")

    @property
    def prefix_width_bucket(self) -> int:
        """Return the 32-token JIT width bucket for this prefix."""
        width = self.prefix.input_ids.shape[1]
        return 32 * math.ceil(width / 32)


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """One routed or direct condition's complete suffix evidence."""

    contract_sha256: str
    evaluation_id: str
    dataset: DatasetKind
    method: str
    order: str | None
    stage: int
    task_id: str
    story_id: str
    candidate_ids: tuple[str, ...]
    selected_index: int
    prefix_scores: tuple[float, ...] | None
    prefix_token_count: int
    prefix_entropy: float | None
    prefix_margin: float | None
    suffix_mean_nll_by_candidate: tuple[float, ...]
    suffix_total_nll: float
    suffix_token_count: int
    suffix_correct_tokens: int
    oracle_index: int
    oracle_suffix_mean_nll: float
    oracle_suffix_correct_tokens: int
    noun_support_hit: bool | None
    exact_noun_route_hit: bool | None

    def __post_init__(self) -> None:
        require_sha256(self.contract_sha256, "temporal evaluation contract")
        require_sha256(self.story_id, "temporal evaluation story")
        if (
            not self.evaluation_id
            or not self.method
            or self.dataset not in (
                "sentinel",
                "macro",
                "final",
                "merge_source",
                "merge_validation",
                "timing",
            )
            or self.task_id not in TASK_IDS
            or self.stage < 0
            or not self.candidate_ids
            or len(set(self.candidate_ids)) != len(self.candidate_ids)
            or not 0 <= self.selected_index < len(self.candidate_ids)
            or not 0 <= self.oracle_index < len(self.candidate_ids)
            or len(self.suffix_mean_nll_by_candidate) != len(self.candidate_ids)
        ):
            raise ValueError("temporal evaluation identity or candidates are invalid")
        if self.prefix_scores is not None and len(self.prefix_scores) != len(
            self.candidate_ids
        ):
            raise ValueError("temporal prefix scores do not match candidates")
        numeric = (
            *self.suffix_mean_nll_by_candidate,
            self.suffix_total_nll,
            self.oracle_suffix_mean_nll,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in numeric):
            raise ValueError("temporal evaluation NLL values are invalid")
        if self.prefix_scores is not None and any(
            not math.isfinite(value) or value < 0.0 for value in self.prefix_scores
        ):
            raise ValueError("temporal prefix scores are invalid")
        if (
            self.prefix_token_count <= 0
            or self.suffix_token_count <= 0
            or not 0 <= self.suffix_correct_tokens <= self.suffix_token_count
            or not 0 <= self.oracle_suffix_correct_tokens <= self.suffix_token_count
        ):
            raise ValueError("temporal evaluation token counts are invalid")

    @property
    def suffix_mean_nll(self) -> float:
        """Return selected suffix token NLL."""
        return self.suffix_total_nll / self.suffix_token_count

    @property
    def suffix_token_accuracy(self) -> float:
        """Return selected teacher-forced suffix token accuracy."""
        return self.suffix_correct_tokens / self.suffix_token_count

    @property
    def oracle_regret(self) -> float:
        """Return selected minus suffix-oracle mean NLL."""
        return self.suffix_mean_nll - self.oracle_suffix_mean_nll

    def as_values(self) -> dict[str, object]:
        """Return values reserved for the chained evaluation ledger."""
        return {
            "candidate_ids": list(self.candidate_ids),
            "contract_sha256": self.contract_sha256,
            "dataset": self.dataset,
            "evaluation_id": self.evaluation_id,
            "exact_noun_route_hit": self.exact_noun_route_hit,
            "method": self.method,
            "noun_support_hit": self.noun_support_hit,
            "oracle_index": self.oracle_index,
            "oracle_regret": self.oracle_regret,
            "oracle_suffix_correct_tokens": self.oracle_suffix_correct_tokens,
            "oracle_suffix_mean_nll": self.oracle_suffix_mean_nll,
            "order": self.order,
            "prefix_entropy": self.prefix_entropy,
            "prefix_margin": self.prefix_margin,
            "prefix_scores": None if self.prefix_scores is None else list(self.prefix_scores),
            "prefix_token_count": self.prefix_token_count,
            "selected_candidate_id": self.candidate_ids[self.selected_index],
            "selected_index": self.selected_index,
            "stage": self.stage,
            "story_id": self.story_id,
            "suffix_correct_tokens": self.suffix_correct_tokens,
            "suffix_mean_nll": self.suffix_mean_nll,
            "suffix_mean_nll_by_candidate": list(self.suffix_mean_nll_by_candidate),
            "suffix_token_accuracy": self.suffix_token_accuracy,
            "suffix_token_count": self.suffix_token_count,
            "suffix_total_nll": self.suffix_total_nll,
            "task_id": self.task_id,
        }


@dataclass(frozen=True, slots=True)
class CandidateScore:
    """Token-weighted loss and accuracy for one explicitly scored candidate."""

    candidate_id: str
    total_nll: float
    token_count: int
    correct_tokens: int

    def __post_init__(self) -> None:
        if (
            not self.candidate_id
            or not math.isfinite(self.total_nll)
            or self.total_nll < 0.0
            or self.token_count <= 0
            or not 0 <= self.correct_tokens <= self.token_count
        ):
            raise ValueError("candidate score fields are invalid")

    @property
    def mean_nll(self) -> float:
        """Return token-weighted mean NLL."""
        return self.total_nll / self.token_count

    @property
    def token_accuracy(self) -> float:
        """Return teacher-forced top-one token accuracy."""
        return self.correct_tokens / self.token_count


def build_adapter_bank(
    candidates: Sequence[AdapterCandidate],
    model_config: GptNeoConfig,
) -> AdapterBank:
    """Pack standalone candidates as root children in their exact tie order."""
    resolved = tuple(candidates)
    graph = init_memory_graph(NodeId("base"))
    for index, candidate in enumerate(resolved, start=1):
        graph = add_memory_node(
            graph,
            NodeId(candidate.candidate_id),
            NodeId("base"),
            TaskId(candidate.candidate_id),
            index,
            candidate.adapter,
        )
    lora_config = LoraConfig(rank=LORA_RANK, alpha=LORA_ALPHA)
    return AdapterBank(
        candidates=resolved,
        packed=pack_lora_memory(
            graph,
            model_config,
            lora_config,
            max_nodes=len(resolved) + 1,
            max_edges=len(resolved),
        ),
        candidate_ids=("base",) + tuple(
            candidate.candidate_id for candidate in resolved
        ),
    )


def build_midpoint_case(
    partition: object,
    store: IndexedStoryStore,
    task_id: str,
    entry: StoryIndexEntry,
    *,
    context_length: int,
    maximum_position_embeddings: int,
) -> MidpointCase:
    """Build structurally separate midpoint prefix and true-suffix tensors."""
    tokens = store.tokens(entry)
    midpoint = len(tokens) // 2
    prefix_tokens = tokens[:midpoint]
    if len(prefix_tokens) < 2 or len(prefix_tokens) >= maximum_position_embeddings:
        raise ValueError("temporal midpoint prefix does not fit the model")
    width = len(prefix_tokens) - 1
    prefix = RouterBatch(
        np.asarray(prefix_tokens[:-1], dtype=np.int32).reshape(1, width),
        np.ones((1, width), dtype=np.bool_),
        np.asarray(prefix_tokens[1:], dtype=np.int32).reshape(1, width),
        np.ones((1, width), dtype=np.bool_),
    )
    suffix = _story_windows(
        tokens,
        context_length,
        int(getattr(partition, "pad_token_id")),
        first_target_index=midpoint,
    )
    return MidpointCase(task_id, entry, midpoint, prefix, suffix)


def score_token_batches_by_candidate(
    batches: Sequence[TokenBatch],
    *,
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    bank: AdapterBank,
    candidate_indices: Sequence[int],
    evaluation_batch_size: int = EVALUATION_BATCH_SIZE,
) -> tuple[CandidateScore, ...]:
    """Score selected base/bank candidates over a finite masked batch stream."""
    indices = tuple(candidate_indices)
    if (
        not batches
        or not indices
        or len(set(indices)) != len(indices)
        or any(not 0 <= index < len(bank.candidate_ids) for index in indices)
    ):
        raise ValueError("candidate batch scoring requires valid data and indices")
    totals = np.zeros((len(indices),), dtype=np.float64)
    correct = np.zeros((len(indices),), dtype=np.int64)
    token_count = 0
    for batch in batches:
        batch_tokens = int(np.sum(batch.loss_mask))
        if batch_tokens <= 0:
            continue
        batch_totals, batch_correct = _score_nodes_per_window(
            base_params,
            model_config,
            bank.packed,
            batch,
            evaluation_batch_size,
            node_indices=indices,
        )
        totals += np.sum(batch_totals, axis=1, dtype=np.float64)
        correct += np.sum(batch_correct, axis=1, dtype=np.int64)
        token_count += batch_tokens
    if token_count <= 0:
        raise ValueError("candidate batch scoring contains no active tokens")
    return tuple(
        CandidateScore(
            bank.candidate_ids[node_index],
            float(totals[position]),
            token_count,
            int(correct[position]),
        )
        for position, node_index in enumerate(indices)
    )


def score_midpoint_cases_by_candidate(
    cases: Sequence[MidpointCase],
    *,
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    bank: AdapterBank,
    candidate_indices: Sequence[int],
    evaluation_batch_size: int = EVALUATION_BATCH_SIZE,
) -> tuple[CandidateScore, ...]:
    """Score selected candidates on only the held-back suffixes of cases."""
    if not cases:
        raise ValueError("midpoint candidate scoring requires cases")
    return score_token_batches_by_candidate(
        (_stack_token_batches(tuple(case.suffix_windows for case in cases)),),
        base_params=base_params,
        model_config=model_config,
        bank=bank,
        candidate_indices=candidate_indices,
        evaluation_batch_size=evaluation_batch_size,
    )


def prepare_prefix_kernel_batch(
    cases: Sequence[MidpointCase],
    *,
    row_count: int = 8,
) -> RouterBatch:
    """Build one fixed-row, fixed-width prefix batch for synchronized timing."""
    if not cases or len({case.prefix_width_bucket for case in cases}) != 1:
        raise ValueError("prefix timing cases require one nonempty width bucket")
    return _stack_prefixes(
        cases,
        cases[0].prefix_width_bucket,
        row_count=row_count,
    )


def prefix_router_row_capacity(
    candidate_count: int,
    prefix_width: int,
    *,
    maximum_rows: int = 8,
) -> int:
    """Choose a fixed power-of-two router microbatch under the GPU work gate."""
    if candidate_count < 0 or prefix_width <= 0 or maximum_rows <= 0:
        raise ValueError("router capacity inputs must be positive")
    affordable = _PREFIX_ROUTER_WORK_BUDGET // (
        (candidate_count + 1) * prefix_width
    )
    for rows in _PREFIX_ROUTER_ROW_BUCKETS:
        if rows <= maximum_rows and rows <= affordable:
            return rows
    return 1


def run_prefix_kernel(
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    bank: AdapterBank,
    batch: RouterBatch,
):
    """Run the exact compiled exhaustive-prefix kernel used by evaluation."""
    return run_packed_prefix_kernel(base_params, model_config, bank.packed, batch)


def run_packed_prefix_kernel(
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    packed: PackedLoraMemory,
    batch: RouterBatch,
):
    """Run the evaluation prefix kernel from an authenticated packed bank."""
    return _compiled_exhaustive_prefix_nll(
        base_params,
        model_config,
        packed,
        LoraConfig(rank=LORA_RANK, alpha=LORA_ALPHA),
        jnp.asarray(batch.input_ids),
        jnp.asarray(batch.attention_mask),
        jnp.asarray(batch.target_ids),
        jnp.asarray(batch.loss_mask),
    )


def prepare_suffix_kernel_batch(
    case: MidpointCase,
    *,
    row_count: int = EVALUATION_BATCH_SIZE,
) -> TokenBatch:
    """Build one fixed-row suffix-window batch for synchronized timing."""
    stop = min(row_count, case.suffix_windows.input_ids.shape[0])
    return _padded_slice(case.suffix_windows, 0, stop, row_count)


def run_suffix_kernel(
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    bank: AdapterBank,
    batch: TokenBatch,
    *,
    node_index: int = 0,
) -> tuple[jax.Array, jax.Array]:
    """Run one exact compiled candidate suffix-scoring kernel."""
    return run_packed_suffix_kernel(
        base_params,
        model_config,
        bank.packed,
        batch,
        node_index=node_index,
    )


def run_packed_suffix_kernel(
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    packed: PackedLoraMemory,
    batch: TokenBatch,
    *,
    node_index: int = 0,
) -> tuple[jax.Array, jax.Array]:
    """Run the evaluation suffix kernel from an authenticated packed bank."""
    coefficients = edge_coefficients_for_node(packed, node_index)
    return _compiled_window_scores(
        base_params,
        model_config,
        packed,
        LoraConfig(rank=LORA_RANK, alpha=LORA_ALPHA),
        jnp.asarray(batch.input_ids),
        jnp.asarray(batch.attention_mask),
        jnp.asarray(batch.target_ids),
        jnp.asarray(batch.loss_mask),
        coefficients,
    )


def evaluate_to_ledger(
    cases: Sequence[MidpointCase],
    *,
    contract_sha256: str,
    evaluation_id: str,
    dataset: DatasetKind,
    method: str,
    order: str | None,
    stage: int,
    routing: RoutingMode,
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    bank: AdapterBank,
    ledger: ChainedJsonlLedger,
    router_batch_size: int = 8,
    evaluation_batch_size: int = EVALUATION_BATCH_SIZE,
    progress: EvaluationProgress | None = None,
) -> tuple[EvaluationResult, ...]:
    """Evaluate pending cases in bounded batches and fsync each result batch."""
    if ledger.row_format != EVALUATION_ROW_FORMAT:
        raise ValueError("temporal evaluation requires its independent row format")
    _validate_existing_evaluation_rows(ledger.rows, contract_sha256)
    ledger.require_unique_keys(
        ("evaluation_id", "method", "order", "stage", "task_id", "story_id")
    )
    completed = {
        (
            str(row.get("evaluation_id")),
            str(row.get("method")),
            str(row.get("order")),
            int(row.get("stage", -1)),
            str(row.get("task_id")),
            str(row.get("story_id")),
        )
        for row in ledger.rows
    }
    expected_case_keys = {(case.task_id, case.entry.story_id) for case in cases}
    expected_case_order = tuple(
        (case.task_id, case.entry.story_id)
        for batch in _routed_case_batches(
            cases,
            router_batch_size,
            len(bank.candidates),
        )
        for case in batch
    )
    observed_call_keys = tuple(
        (str(row["task_id"]), str(row["story_id"]))
        for row in ledger.rows
        if (
            row.get("evaluation_id"),
            row.get("method"),
            str(row.get("order")),
            row.get("stage"),
        )
        == (evaluation_id, method, str(order), stage)
    )
    if (
        set(observed_call_keys) - expected_case_keys
        or observed_call_keys != expected_case_order[: len(observed_call_keys)]
    ):
        raise ValueError("evaluation ledger is not a canonical case prefix")
    pending = tuple(
        case
        for case in cases
        if (
            evaluation_id,
            method,
            str(order),
            stage,
            case.task_id,
            case.entry.story_id,
        )
        not in completed
    )
    existing_for_call = tuple(
        row
        for row in ledger.rows
        if (
            row.get("evaluation_id"),
            row.get("method"),
            str(row.get("order")),
            row.get("stage"),
        )
        == (evaluation_id, method, str(order), stage)
    )
    aggregate_nll = [float(row["suffix_mean_nll"]) for row in existing_for_call]
    aggregate_correct = sum(int(row["suffix_correct_tokens"]) for row in existing_for_call)
    aggregate_tokens = sum(int(row["suffix_token_count"]) for row in existing_for_call)
    results: list[EvaluationResult] = []
    finished = len(existing_for_call)
    for batch_cases in _routed_case_batches(
        pending,
        router_batch_size,
        len(bank.candidates),
    ):
        physical_router_rows = prefix_router_row_capacity(
            len(bank.candidates),
            batch_cases[0].prefix_width_bucket,
            maximum_rows=router_batch_size,
        )
        batch_results = evaluate_case_batch(
            batch_cases,
            contract_sha256=contract_sha256,
            evaluation_id=evaluation_id,
            dataset=dataset,
            method=method,
            order=order,
            stage=stage,
            routing=routing,
            base_params=base_params,
            model_config=model_config,
            bank=bank,
            evaluation_batch_size=evaluation_batch_size,
            physical_router_rows=physical_router_rows,
        )
        ledger.append_many(result.as_values() for result in batch_results)
        results.extend(batch_results)
        aggregate_nll.extend(result.suffix_mean_nll for result in batch_results)
        aggregate_correct += sum(result.suffix_correct_tokens for result in batch_results)
        aggregate_tokens += sum(result.suffix_token_count for result in batch_results)
        finished += len(batch_results)
        if progress is not None:
            progress(
                finished,
                len(cases),
                {
                    "coverage": finished / len(cases),
                    "story_nll": float(np.mean(aggregate_nll)),
                    "token_accuracy": aggregate_correct / aggregate_tokens,
                },
            )
    return tuple(results)


def evaluate_case_batch(
    cases: Sequence[MidpointCase],
    *,
    contract_sha256: str,
    evaluation_id: str,
    dataset: DatasetKind,
    method: str,
    order: str | None,
    stage: int,
    routing: RoutingMode,
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    bank: AdapterBank,
    evaluation_batch_size: int = EVALUATION_BATCH_SIZE,
    physical_router_rows: int | None = None,
) -> tuple[EvaluationResult, ...]:
    """Route and score one same-prefix-bucket case batch."""
    resolved = tuple(cases)
    if not resolved or len({case.prefix_width_bucket for case in resolved}) != 1:
        raise ValueError("evaluation batches require one nonempty prefix-width bucket")
    if routing == "forced_adapter" and len(bank.candidates) != 1:
        raise ValueError("forced-adapter evaluation requires exactly one adapter")
    router_rows = physical_router_rows or len(resolved)
    if router_rows < len(resolved):
        raise ValueError("physical router rows cannot be smaller than the case batch")
    prefix_batch = _stack_prefixes(
        resolved,
        resolved[0].prefix_width_bucket,
        row_count=router_rows,
    )
    if routing == "exhaustive":
        address = _compiled_exhaustive_prefix_nll(
            base_params,
            model_config,
            bank.packed,
            LoraConfig(rank=LORA_RANK, alpha=LORA_ALPHA),
            jnp.asarray(prefix_batch.input_ids),
            jnp.asarray(prefix_batch.attention_mask),
            jnp.asarray(prefix_batch.target_ids),
            jnp.asarray(prefix_batch.loss_mask),
        )
        selected = np.asarray(address.selected_indices, dtype=np.int32)[: len(resolved)]
        scores = np.asarray(address.node_scores, dtype=np.float64)[: len(resolved)]
        entropies = np.asarray(address.entropy, dtype=np.float64)[: len(resolved)]
        margins = np.asarray(address.score_margin, dtype=np.float64)[: len(resolved)]
    else:
        forced_index = 0 if routing == "forced_base" else 1
        selected = np.full((len(resolved),), forced_index, dtype=np.int32)
        scores = None
        entropies = None
        margins = None
    windows = _stack_token_batches(tuple(case.suffix_windows for case in resolved))
    window_boundaries = np.cumsum(
        (0,) + tuple(case.suffix_windows.input_ids.shape[0] for case in resolved)
    )
    node_window_nll, node_window_correct = _score_nodes_per_window(
        base_params,
        model_config,
        bank.packed,
        windows,
        evaluation_batch_size,
    )
    node_totals = np.asarray(
        [
            [
                np.sum(node_window_nll[node, start:stop], dtype=np.float64)
                for node in range(len(bank.candidate_ids))
            ]
            for start, stop in zip(window_boundaries[:-1], window_boundaries[1:])
        ],
        dtype=np.float64,
    )
    node_correct = np.asarray(
        [
            [
                np.sum(node_window_correct[node, start:stop], dtype=np.int64)
                for node in range(len(bank.candidate_ids))
            ]
            for start, stop in zip(window_boundaries[:-1], window_boundaries[1:])
        ],
        dtype=np.int64,
    )
    token_counts = np.asarray(
        [int(np.sum(case.suffix_windows.loss_mask)) for case in resolved],
        dtype=np.int64,
    )
    mean_nll = node_totals / token_counts[:, None]
    oracle = np.argmin(mean_nll, axis=1)
    return tuple(
        _evaluation_result(
            case,
            contract_sha256,
            evaluation_id,
            dataset,
            method,
            order,
            stage,
            bank,
            int(selected[index]),
            None if scores is None else tuple(float(value) for value in scores[index]),
            None if entropies is None else float(entropies[index]),
            None if margins is None else float(margins[index]),
            tuple(float(value) for value in mean_nll[index]),
            node_totals[index],
            node_correct[index],
            int(token_counts[index]),
            int(oracle[index]),
        )
        for index, case in enumerate(resolved)
    )


def _evaluation_result(
    case: MidpointCase,
    contract_sha256: str,
    evaluation_id: str,
    dataset: DatasetKind,
    method: str,
    order: str | None,
    stage: int,
    bank: AdapterBank,
    selected: int,
    prefix_scores: tuple[float, ...] | None,
    entropy: float | None,
    margin: float | None,
    mean_nll: tuple[float, ...],
    total_nll: np.ndarray,
    correct: np.ndarray,
    token_count: int,
    oracle: int,
) -> EvaluationResult:
    selected_candidate = None if selected == 0 else bank.candidates[selected - 1]
    noun_support = (
        selected_candidate is not None
        and case.task_id in dict(selected_candidate.task_counts)
        if method in ("log_t", "independent_noun_exhaustive")
        else None
    )
    exact_route = (
        None
        if method != "independent_noun_exhaustive"
        else selected_candidate is not None
        and selected_candidate.task_counts == ((case.task_id, 8),)
    )
    return EvaluationResult(
        contract_sha256=contract_sha256,
        evaluation_id=evaluation_id,
        dataset=dataset,
        method=method,
        order=order,
        stage=stage,
        task_id=case.task_id,
        story_id=case.entry.story_id,
        candidate_ids=bank.candidate_ids,
        selected_index=selected,
        prefix_scores=prefix_scores,
        prefix_token_count=int(np.sum(case.prefix.loss_mask)),
        prefix_entropy=entropy,
        prefix_margin=margin,
        suffix_mean_nll_by_candidate=mean_nll,
        suffix_total_nll=float(total_nll[selected]),
        suffix_token_count=token_count,
        suffix_correct_tokens=int(correct[selected]),
        oracle_index=oracle,
        oracle_suffix_mean_nll=mean_nll[oracle],
        oracle_suffix_correct_tokens=int(correct[oracle]),
        noun_support_hit=noun_support,
        exact_noun_route_hit=exact_route,
    )


@partial(jax.jit, static_argnames=("model_config", "lora_config"))
def _compiled_exhaustive_prefix_nll(
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    packed: PackedLoraMemory,
    lora_config: LoraConfig,
    input_ids: jax.Array,
    attention_mask: jax.Array,
    target_ids: jax.Array,
    loss_mask: jax.Array,
):
    return exhaustive_prefix_nll_core(
        base_params,
        model_config,
        packed,
        lora_config,
        input_ids,
        attention_mask,
        target_ids,
        loss_mask,
    )


@partial(jax.jit, static_argnames=("model_config", "lora_config"))
def _compiled_window_scores(
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    packed: PackedLoraMemory,
    lora_config: LoraConfig,
    input_ids: jax.Array,
    attention_mask: jax.Array,
    target_ids: jax.Array,
    loss_mask: jax.Array,
    coefficients: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    logits = apply_gpt_neo(
        base_params,
        model_config,
        input_ids,
        attention_mask,
        lora_memory=packed,
        edge_coefficients=coefficients,
        lora_config=lora_config,
        training=False,
    ).logits
    mask = jnp.asarray(loss_mask, dtype=jnp.float32)
    losses = per_token_nll(logits, target_ids)
    predictions = jnp.argmax(logits, axis=-1)
    return (
        jnp.sum(losses * mask, axis=-1),
        jnp.sum((predictions == target_ids) * mask, axis=-1),
    )


def _score_nodes_per_window(
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    packed: PackedLoraMemory,
    windows: TokenBatch,
    batch_size: int,
    *,
    node_indices: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    window_count = windows.input_ids.shape[0]
    node_count = int(np.sum(np.asarray(packed.valid_node_mask, dtype=np.int32)))
    selected = tuple(range(node_count)) if node_indices is None else tuple(node_indices)
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(not 0 <= node < node_count for node in selected)
    ):
        raise ValueError("window scoring node indices are invalid")
    totals = np.empty((len(selected), window_count), dtype=np.float64)
    correct = np.empty((len(selected), window_count), dtype=np.int64)
    lora_config = LoraConfig(rank=LORA_RANK, alpha=LORA_ALPHA)

    for output_index, node in enumerate(selected):
        coefficients = edge_coefficients_for_node(packed, node)
        for start in range(0, window_count, batch_size):
            stop = min(start + batch_size, window_count)
            batch = _padded_slice(windows, start, stop, batch_size)
            batch_totals, batch_correct = _compiled_window_scores(
                base_params,
                model_config,
                packed,
                lora_config,
                jnp.asarray(batch.input_ids),
                jnp.asarray(batch.attention_mask),
                jnp.asarray(batch.target_ids),
                jnp.asarray(batch.loss_mask),
                coefficients,
            )
            totals[output_index, start:stop] = np.asarray(batch_totals)[: stop - start]
            correct[output_index, start:stop] = np.asarray(batch_correct)[: stop - start]
    return totals, correct


def _case_batches(
    cases: Sequence[MidpointCase],
    maximum_rows: int,
) -> Iterable[tuple[MidpointCase, ...]]:
    buckets = tuple(sorted({case.prefix_width_bucket for case in cases}))
    for bucket in buckets:
        bucket_cases = tuple(
            case for case in cases if case.prefix_width_bucket == bucket
        )
        for start in range(0, len(bucket_cases), maximum_rows):
            yield bucket_cases[start : start + maximum_rows]


def _routed_case_batches(
    cases: Sequence[MidpointCase],
    maximum_rows: int,
    candidate_count: int,
) -> Iterable[tuple[MidpointCase, ...]]:
    buckets = tuple(sorted({case.prefix_width_bucket for case in cases}))
    for bucket in buckets:
        rows = prefix_router_row_capacity(
            candidate_count,
            bucket,
            maximum_rows=maximum_rows,
        )
        bucket_cases = tuple(case for case in cases if case.prefix_width_bucket == bucket)
        for start in range(0, len(bucket_cases), rows):
            yield bucket_cases[start : start + rows]


def _stack_prefixes(
    cases: Sequence[MidpointCase],
    width: int,
    *,
    row_count: int | None = None,
) -> RouterBatch:
    rows = row_count or len(cases)
    if rows < len(cases):
        raise ValueError("prefix row capacity is smaller than its cases")
    shape = (rows, width)
    inputs = np.zeros(shape, dtype=np.int32)
    targets = np.zeros(shape, dtype=np.int32)
    attention = np.zeros(shape, dtype=np.bool_)
    for row, case in enumerate(cases):
        case_width = case.prefix.input_ids.shape[1]
        inputs[row, :case_width] = case.prefix.input_ids[0]
        targets[row, :case_width] = case.prefix.target_ids[0]
        attention[row, :case_width] = True
    if rows > len(cases):
        repeats = rows - len(cases)
        inputs[len(cases) :] = np.repeat(inputs[:1], repeats, axis=0)
        targets[len(cases) :] = np.repeat(targets[:1], repeats, axis=0)
        attention[len(cases) :] = np.repeat(attention[:1], repeats, axis=0)
    return RouterBatch(inputs, attention, targets, attention)


def _story_windows(
    token_ids: tuple[int, ...],
    context_length: int,
    pad_token_id: int,
    *,
    first_target_index: int,
) -> TokenBatch:
    starts = tuple(range(0, len(token_ids) - 1, context_length))
    shape = (len(starts), context_length)
    inputs = np.full(shape, pad_token_id, dtype=np.int32)
    targets = np.full(shape, pad_token_id, dtype=np.int32)
    attention = np.zeros(shape, dtype=np.bool_)
    losses = np.zeros(shape, dtype=np.bool_)
    for row, start in enumerate(starts):
        chunk = token_ids[start : start + context_length + 1]
        transitions = len(chunk) - 1
        inputs[row, :transitions] = chunk[:-1]
        targets[row, :transitions] = chunk[1:]
        attention[row, :transitions] = True
        target_indices = np.arange(start + 1, start + transitions + 1)
        losses[row, :transitions] = target_indices >= first_target_index
    return TokenBatch(inputs, attention, targets, losses)


def _stack_token_batches(batches: Sequence[TokenBatch]) -> TokenBatch:
    if not batches:
        raise ValueError("suffix window stacking requires at least one batch")
    return TokenBatch(
        np.concatenate(tuple(batch.input_ids for batch in batches)),
        np.concatenate(tuple(batch.attention_mask for batch in batches)),
        np.concatenate(tuple(batch.target_ids for batch in batches)),
        np.concatenate(tuple(batch.loss_mask for batch in batches)),
    )


def _padded_slice(
    batch: TokenBatch,
    start: int,
    stop: int,
    row_count: int,
) -> TokenBatch:
    size = stop - start
    if size == row_count:
        return TokenBatch(
            batch.input_ids[start:stop],
            batch.attention_mask[start:stop],
            batch.target_ids[start:stop],
            batch.loss_mask[start:stop],
        )
    repeats = row_count - size
    return TokenBatch(
        np.concatenate((batch.input_ids[start:stop], np.repeat(batch.input_ids[start:start + 1], repeats, axis=0))),
        np.concatenate((batch.attention_mask[start:stop], np.repeat(batch.attention_mask[start:start + 1], repeats, axis=0))),
        np.concatenate((batch.target_ids[start:stop], np.repeat(batch.target_ids[start:start + 1], repeats, axis=0))),
        np.concatenate((batch.loss_mask[start:stop], np.zeros((repeats, batch.loss_mask.shape[1]), dtype=np.bool_))),
    )


def _validate_existing_evaluation_rows(
    rows: Sequence[dict[str, object]],
    contract_sha256: str,
) -> None:
    """Reject cross-contract rows and arithmetically inconsistent evidence."""
    value_fields = {
        "candidate_ids",
        "contract_sha256",
        "dataset",
        "evaluation_id",
        "exact_noun_route_hit",
        "method",
        "noun_support_hit",
        "oracle_index",
        "oracle_regret",
        "oracle_suffix_correct_tokens",
        "oracle_suffix_mean_nll",
        "order",
        "prefix_entropy",
        "prefix_margin",
        "prefix_scores",
        "prefix_token_count",
        "selected_candidate_id",
        "selected_index",
        "stage",
        "story_id",
        "suffix_correct_tokens",
        "suffix_mean_nll",
        "suffix_mean_nll_by_candidate",
        "suffix_token_accuracy",
        "suffix_token_count",
        "suffix_total_nll",
        "task_id",
    }
    ledger_fields = value_fields | {
        "format",
        "previous_sha256",
        "result_sha256",
        "sequence",
    }
    for row in rows:
        candidates = row.get("candidate_ids")
        means = row.get("suffix_mean_nll_by_candidate")
        selected = row.get("selected_index")
        oracle = row.get("oracle_index")
        token_count = row.get("suffix_token_count")
        total = row.get("suffix_total_nll")
        mean = row.get("suffix_mean_nll")
        correct = row.get("suffix_correct_tokens")
        accuracy = row.get("suffix_token_accuracy")
        regret = row.get("oracle_regret")
        oracle_mean = row.get("oracle_suffix_mean_nll")
        prefix_scores = row.get("prefix_scores")
        if (
            set(row) != ledger_fields
            or row.get("contract_sha256") != contract_sha256
            or type(candidates) is not list
            or not candidates
            or len(set(candidates)) != len(candidates)
            or type(means) is not list
            or len(means) != len(candidates)
            or type(selected) is not int
            or type(oracle) is not int
            or not 0 <= selected < len(candidates)
            or not 0 <= oracle < len(candidates)
            or row.get("selected_candidate_id") != candidates[selected]
            or type(token_count) is not int
            or token_count <= 0
            or type(correct) is not int
            or not 0 <= correct <= token_count
            or not all(_finite_nonnegative(value) for value in means)
            or not _finite_nonnegative(total)
            or not _finite_nonnegative(mean)
            or not _finite_nonnegative(accuracy)
            or not _finite_nonnegative(oracle_mean)
            or not _finite_number(regret)
            or not math.isclose(float(mean), float(total) / token_count, abs_tol=1e-9)
            or not math.isclose(float(accuracy), correct / token_count, abs_tol=1e-12)
            or not math.isclose(
                float(regret),
                float(mean) - float(oracle_mean),
                abs_tol=1e-9,
            )
            or (
                prefix_scores is not None
                and (
                    type(prefix_scores) is not list
                    or len(prefix_scores) != len(candidates)
                    or not all(_finite_nonnegative(value) for value in prefix_scores)
                )
            )
        ):
            raise ValueError("temporal evaluation ledger row changed")


def _finite_nonnegative(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value)) and value >= 0


def _finite_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


__all__ = [
    "AdapterBank",
    "AdapterCandidate",
    "CandidateScore",
    "EvaluationResult",
    "MidpointCase",
    "build_adapter_bank",
    "build_midpoint_case",
    "evaluate_case_batch",
    "evaluate_to_ledger",
    "prepare_prefix_kernel_batch",
    "prepare_suffix_kernel_batch",
    "prefix_router_row_capacity",
    "run_packed_prefix_kernel",
    "run_packed_suffix_kernel",
    "run_prefix_kernel",
    "run_suffix_kernel",
    "score_midpoint_cases_by_candidate",
    "score_token_batches_by_candidate",
]

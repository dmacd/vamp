"""Task-free language routers and evaluator-only suffix competence."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from apm.continual.language_benchmarks import (
    ROUTER_BASELINE_NAMES,
    RouteExampleEvaluation,
    RouterBaselineName,
    deterministic_random_valid_node_indices,
    evaluate_route_results,
)
from apm.continual.language_tasks import (
    AddressBook,
    AddressResult,
    CompetenceBatch,
    LanguageEvaluationExample,
    RouterBatch,
)
from apm.continual.language_metrics import resolve_node_index
from apm.lm.config import GptNeoConfig
from apm.lm.evaluation import (
    evaluation_microbatch_slices,
    validate_evaluation_microbatch_size,
)
from apm.lm.gpt_neo import apply_gpt_neo
from apm.lm.lora import LoraConfig, LoraEdge
from apm.lm.lora_memory import PackedLoraMemory, edge_coefficients_for_node
from apm.lm.losses import per_token_nll
from apm.lm.parameters import GptNeoParams
from apm.memory.address_refinement import (
    EbtAddressResult,
    EbtConfig,
    refine_ebt_address,
)
from apm.memory.content_addressing import (
    HopfieldAddressResult,
    HopfieldConfig,
    hopfield_address,
)
from apm.memory.content_keys import encode_frozen_base_content
from apm.memory.graph import MemoryGraph, path_incidence_matrix
from apm.memory.prefix_energy import exhaustive_prefix_nll_address


LANGUAGE_ROUTER_TOP_K = 4


class LanguageAddressDecision(NamedTuple):
    """Common higher-is-better scores, probabilities, and task-free choices."""

    selected_indices: jax.Array
    node_probabilities: jax.Array
    node_scores: jax.Array
    score_margin: jax.Array
    entropy: jax.Array
    top_k_indices: jax.Array


@dataclass(frozen=True, eq=False)
class LanguageRouterEvaluation:
    """Evaluator-only suffix results and confusion after task-free routing."""

    router: RouterBaselineName
    decision: LanguageAddressDecision
    suffix_nll_by_node: np.ndarray
    examples: tuple[RouteExampleEvaluation, ...]
    confusion_counts: np.ndarray

    def __post_init__(self) -> None:
        if self.router not in ROUTER_BASELINE_NAMES:
            raise ValueError(f"unknown router: {self.router}")
        suffix_nll = np.array(self.suffix_nll_by_node, dtype=np.float32, copy=True)
        confusion = np.array(self.confusion_counts, dtype=np.int64, copy=True)
        if suffix_nll.ndim != 2 or suffix_nll.shape[0] == 0:
            raise ValueError("suffix_nll_by_node must have shape [batch, nodes]")
        batch_size, node_count = suffix_nll.shape
        if not isinstance(self.examples, tuple) or len(self.examples) != batch_size:
            raise ValueError("route evaluations must match suffix-NLL rows")
        if any(not isinstance(example, RouteExampleEvaluation) for example in self.examples):
            raise TypeError("examples must contain RouteExampleEvaluation values")
        if confusion.shape != (node_count, node_count) or np.any(confusion < 0):
            raise ValueError("confusion_counts must be nonnegative [nodes, nodes]")
        if int(np.sum(confusion)) != batch_size:
            raise ValueError("confusion counts must contain every example exactly once")
        expected_confusion = np.zeros_like(confusion)
        np.add.at(
            expected_confusion,
            tuple(zip(*(example.confusion_pair for example in self.examples))),
            1,
        )
        if not np.array_equal(confusion, expected_confusion):
            raise ValueError("confusion counts must match route evaluation pairs")
        suffix_nll.flags.writeable = False
        confusion.flags.writeable = False
        object.__setattr__(self, "suffix_nll_by_node", suffix_nll)
        object.__setattr__(self, "confusion_counts", confusion)


def route_language_prefix(
    router: RouterBaselineName,
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    packed_memory: PackedLoraMemory,
    lora_config: LoraConfig,
    address_book: AddressBook,
    prefix_batch: RouterBatch,
    *,
    random_seed: int = 0,
    hopfield_config: HopfieldConfig = HopfieldConfig(),
    ebt_config: EbtConfig = EbtConfig(),
    evaluation_microbatch_size: int | None = None,
) -> LanguageAddressDecision:
    """Route from prefix arrays only; task and suffix identity are unavailable."""
    if router not in ROUTER_BASELINE_NAMES:
        raise ValueError(f"unknown task-free router: {router}")
    if not isinstance(prefix_batch, RouterBatch):
        raise TypeError("task-free routers accept RouterBatch prefix data only")
    if np.any(np.sum(prefix_batch.attention_mask, axis=-1) == 0):
        raise ValueError("every router row must contain active prefix tokens")
    _validate_address_alignment(packed_memory, address_book)
    microbatch_size = validate_evaluation_microbatch_size(
        evaluation_microbatch_size
    )
    slices = evaluation_microbatch_slices(
        prefix_batch.input_ids.shape[0],
        microbatch_size,
    )
    decisions = tuple(
        _route_language_prefix_batch(
            router,
            base_params,
            model_config,
            packed_memory,
            lora_config,
            address_book,
            _slice_router_batch(prefix_batch, row_slice),
            random_seed=random_seed,
            hopfield_config=hopfield_config,
            ebt_config=ebt_config,
            evaluation_microbatch_size=microbatch_size,
        )
        for row_slice in slices
    )
    decision = (
        decisions[0]
        if len(decisions) == 1
        else _concatenate_address_decisions(decisions)
    )
    return _validated_decision(
        decision,
        packed_memory.valid_node_mask,
        prefix_batch.input_ids.shape[0],
    )


def trace_ebt_language_prefix(
    router: RouterBaselineName,
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    packed_memory: PackedLoraMemory,
    lora_config: LoraConfig,
    address_book: AddressBook,
    prefix_batch: RouterBatch,
    *,
    hopfield_config: HopfieldConfig = HopfieldConfig(),
    ebt_config: EbtConfig = EbtConfig(),
) -> EbtAddressResult:
    """Return the full EBT coefficient trajectory for one prefix batch."""
    if router not in ("vamp_ebt_uniform", "vamp_ebt_hopfield"):
        raise ValueError(f"coefficient traces require an EBT router: {router}")
    if not isinstance(prefix_batch, RouterBatch):
        raise TypeError("task-free routers accept RouterBatch prefix data only")
    if np.any(np.sum(prefix_batch.attention_mask, axis=-1) == 0):
        raise ValueError("every router row must contain active prefix tokens")
    _validate_address_alignment(packed_memory, address_book)
    hopfield_result = (
        _hopfield_prefix_address(
            base_params,
            model_config,
            address_book,
            prefix_batch,
            hopfield_config,
        )
        if router == "vamp_ebt_hopfield"
        else None
    )
    return refine_ebt_address(
        base_params,
        model_config,
        packed_memory,
        lora_config,
        prefix_batch,
        replace(
            ebt_config,
            initialization=(
                "uniform" if router == "vamp_ebt_uniform" else "hopfield"
            ),
        ),
        hopfield_result=hopfield_result,
    )


def _route_language_prefix_batch(
    router: RouterBaselineName,
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    packed_memory: PackedLoraMemory,
    lora_config: LoraConfig,
    address_book: AddressBook,
    prefix_batch: RouterBatch,
    *,
    random_seed: int,
    hopfield_config: HopfieldConfig,
    ebt_config: EbtConfig,
    evaluation_microbatch_size: int | None,
) -> LanguageAddressDecision:
    """Route one evaluation microbatch with unchanged per-example semantics."""
    if router == "vamp_exhaustive":
        result = exhaustive_prefix_nll_address(
            base_params,
            model_config,
            packed_memory,
            lora_config,
            prefix_batch,
            evaluation_microbatch_size=evaluation_microbatch_size,
        )
        decision = _decision_from_exhaustive(
            result,
            packed_memory.valid_node_mask,
        )
    else:
        hopfield_result = (
            _hopfield_prefix_address(
                base_params,
                model_config,
                address_book,
                prefix_batch,
                hopfield_config,
            )
            if router == "vamp_hopfield"
            else None
        )
        if router == "vamp_hopfield":
            assert hopfield_result is not None
            decision = LanguageAddressDecision(*hopfield_result)
        elif router in ("vamp_ebt_uniform", "vamp_ebt_hopfield"):
            refinement = trace_ebt_language_prefix(
                router,
                base_params,
                model_config,
                packed_memory,
                lora_config,
                address_book,
                prefix_batch,
                hopfield_config=hopfield_config,
                ebt_config=ebt_config,
            )
            decision = _decision_from_ebt(
                refinement,
                packed_memory.valid_node_mask,
            )
        else:
            decision = _random_node_decision(
                prefix_batch,
                packed_memory.valid_node_mask,
                random_seed,
            )
    return decision


def evaluate_language_router(
    router: RouterBaselineName,
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    graph: MemoryGraph[LoraEdge],
    packed_memory: PackedLoraMemory,
    lora_config: LoraConfig,
    address_book: AddressBook,
    examples: tuple[LanguageEvaluationExample, ...],
    *,
    random_seed: int = 0,
    hopfield_config: HopfieldConfig = HopfieldConfig(),
    ebt_config: EbtConfig = EbtConfig(),
    evaluation_microbatch_size: int | None = None,
    suffix_nll_by_node: np.ndarray | None = None,
) -> LanguageRouterEvaluation:
    """Route on stacked prefixes, then score every valid node on suffixes."""
    if not examples:
        raise ValueError("router evaluation requires at least one example")
    if any(not isinstance(example, LanguageEvaluationExample) for example in examples):
        raise TypeError("examples must contain LanguageEvaluationExample values")
    if any(example.router_batch.input_ids.shape[0] != 1 for example in examples):
        raise ValueError("each router evaluation example must contain exactly one row")
    _validate_evaluation_alignment(graph, packed_memory, address_book)
    router_batch = _stack_router_batches(
        tuple(example.router_batch for example in examples)
    )
    competence_batch = _stack_competence_batches(
        tuple(example.competence_batch for example in examples)
    )
    decision = route_language_prefix(
        router,
        base_params,
        model_config,
        packed_memory,
        lora_config,
        address_book,
        router_batch,
        random_seed=random_seed,
        hopfield_config=hopfield_config,
        ebt_config=ebt_config,
        evaluation_microbatch_size=evaluation_microbatch_size,
    )
    suffix_nll = (
        competence_nll_by_node(
            base_params,
            model_config,
            packed_memory,
            lora_config,
            competence_batch,
            evaluation_microbatch_size=evaluation_microbatch_size,
        )
        if suffix_nll_by_node is None
        else _validated_suffix_nll_by_node(
            suffix_nll_by_node,
            competence_batch.input_ids.shape[0],
            packed_memory.valid_node_mask,
        )
    )
    oracle_indices = np.asarray(
        tuple(resolve_node_index(graph, example.oracle_node_id) for example in examples),
        dtype=np.int32,
    )
    route_examples = evaluate_route_results(
        decision.selected_indices,
        suffix_nll,
        packed_memory.valid_node_mask,
        oracle_indices,
        node_probabilities=decision.node_probabilities,
        top_k_indices=decision.top_k_indices,
    )
    node_capacity = packed_memory.node_path_matrix.shape[0]
    confusion = np.zeros((node_capacity, node_capacity), dtype=np.int64)
    np.add.at(
        confusion,
        (
            oracle_indices,
            np.asarray(decision.selected_indices, dtype=np.int32),
        ),
        1,
    )
    return LanguageRouterEvaluation(
        router=router,
        decision=decision,
        suffix_nll_by_node=suffix_nll,
        examples=route_examples,
        confusion_counts=confusion,
    )


def competence_nll_by_node(
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    packed_memory: PackedLoraMemory,
    lora_config: LoraConfig,
    competence_batch: CompetenceBatch,
    *,
    evaluation_microbatch_size: int | None = None,
) -> np.ndarray:
    """Return per-example suffix NLL for every fixed-capacity node."""
    if not isinstance(competence_batch, CompetenceBatch):
        raise TypeError("suffix competence requires a CompetenceBatch")
    loss_mask = jnp.asarray(competence_batch.loss_mask, dtype=jnp.float32)
    if np.any(np.sum(competence_batch.loss_mask, axis=-1) == 0):
        raise ValueError("every competence row must contain suffix loss tokens")

    microbatch_size = validate_evaluation_microbatch_size(
        evaluation_microbatch_size
    )
    if microbatch_size is not None:
        return _microbatched_competence_nll_by_node(
            base_params,
            model_config,
            packed_memory,
            lora_config,
            competence_batch,
            microbatch_size,
        )

    def score_node(node_index: jax.Array) -> jax.Array:
        coefficients = edge_coefficients_for_node(packed_memory, node_index)
        logits = apply_gpt_neo(
            base_params,
            model_config,
            jnp.asarray(competence_batch.input_ids, dtype=jnp.int32),
            jnp.asarray(competence_batch.attention_mask, dtype=jnp.bool_),
            lora_memory=packed_memory,
            edge_coefficients=coefficients,
            lora_config=lora_config,
            training=False,
        ).logits
        token_nll = per_token_nll(
            logits,
            jnp.asarray(competence_batch.target_ids, dtype=jnp.int32),
        )
        return jnp.sum(token_nll * loss_mask, axis=-1) / jnp.sum(loss_mask, axis=-1)

    node_capacity = packed_memory.node_path_matrix.shape[0]
    values = jax.vmap(score_node)(jnp.arange(node_capacity, dtype=jnp.int32)).T
    masked = jnp.where(
        jnp.asarray(packed_memory.valid_node_mask, dtype=jnp.bool_)[None, :],
        values,
        jnp.asarray(jnp.inf, dtype=jnp.float32),
    )
    result = np.asarray(masked, dtype=np.float32)
    result.flags.writeable = False
    return result


def _microbatched_competence_nll_by_node(
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    packed_memory: PackedLoraMemory,
    lora_config: LoraConfig,
    competence_batch: CompetenceBatch,
    evaluation_microbatch_size: int,
) -> np.ndarray:
    """Score one node and bounded row slice at a time."""
    row_count = competence_batch.input_ids.shape[0]
    node_capacity = packed_memory.node_path_matrix.shape[0]
    valid_node_mask = np.asarray(packed_memory.valid_node_mask, dtype=np.bool_)
    result = np.full((row_count, node_capacity), np.inf, dtype=np.float32)
    for row_slice in evaluation_microbatch_slices(
        row_count,
        evaluation_microbatch_size,
    ):
        input_ids = jnp.asarray(
            competence_batch.input_ids[row_slice],
            dtype=jnp.int32,
        )
        attention_mask = jnp.asarray(
            competence_batch.attention_mask[row_slice],
            dtype=jnp.bool_,
        )
        target_ids = jnp.asarray(
            competence_batch.target_ids[row_slice],
            dtype=jnp.int32,
        )
        loss_mask = jnp.asarray(
            competence_batch.loss_mask[row_slice],
            dtype=jnp.float32,
        )
        active_token_counts = jnp.sum(loss_mask, axis=-1)
        for node_index in np.flatnonzero(valid_node_mask):
            coefficients = edge_coefficients_for_node(packed_memory, node_index)
            logits = apply_gpt_neo(
                base_params,
                model_config,
                input_ids,
                attention_mask,
                lora_memory=packed_memory,
                edge_coefficients=coefficients,
                lora_config=lora_config,
                training=False,
            ).logits
            token_nll = per_token_nll(logits, target_ids)
            node_nll = jnp.sum(token_nll * loss_mask, axis=-1) / active_token_counts
            result[row_slice, node_index] = np.asarray(
                node_nll,
                dtype=np.float32,
            )
    result.flags.writeable = False
    return result


def _validated_suffix_nll_by_node(
    values: np.ndarray,
    batch_size: int,
    valid_node_mask: jax.Array,
) -> np.ndarray:
    """Validate evaluator-provided suffix scores before baseline reuse."""
    valid_mask = np.asarray(valid_node_mask, dtype=np.bool_)
    suffix_nll = np.array(values, dtype=np.float32, copy=True)
    expected_shape = (batch_size, valid_mask.shape[0])
    if suffix_nll.shape != expected_shape:
        raise ValueError(f"suffix_nll_by_node must have shape {expected_shape}")
    if np.any(~np.isfinite(suffix_nll[:, valid_mask])) or np.any(
        suffix_nll[:, valid_mask] < 0.0
    ):
        raise ValueError("valid-node suffix NLL values must be finite and nonnegative")
    if np.any(~np.isposinf(suffix_nll[:, ~valid_mask])):
        raise ValueError("invalid-node suffix NLL values must be positive infinity")
    suffix_nll.flags.writeable = False
    return suffix_nll


def _slice_router_batch(batch: RouterBatch, row_slice: slice) -> RouterBatch:
    return RouterBatch(
        input_ids=batch.input_ids[row_slice],
        attention_mask=batch.attention_mask[row_slice],
        target_ids=batch.target_ids[row_slice],
        loss_mask=batch.loss_mask[row_slice],
    )


def _concatenate_address_decisions(
    decisions: tuple[LanguageAddressDecision, ...],
) -> LanguageAddressDecision:
    return LanguageAddressDecision(
        *(jnp.concatenate(values, axis=0) for values in zip(*decisions))
    )


def _hopfield_prefix_address(
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    address_book: AddressBook,
    prefix_batch: RouterBatch,
    config: HopfieldConfig,
) -> HopfieldAddressResult:
    queries = encode_frozen_base_content(
        base_params,
        model_config,
        jnp.asarray(prefix_batch.input_ids, dtype=jnp.int32),
        jnp.asarray(prefix_batch.attention_mask, dtype=jnp.bool_),
    )
    return hopfield_address(
        queries,
        address_book,
        replace(config, top_k=LANGUAGE_ROUTER_TOP_K),
    )


def _decision_from_exhaustive(
    result: AddressResult,
    valid_node_mask: jax.Array,
) -> LanguageAddressDecision:
    valid_count = int(np.sum(np.asarray(valid_node_mask, dtype=np.bool_)))
    normalized_scores = (-result.node_scores).astype(jnp.float32)
    top_k = jax.lax.top_k(
        normalized_scores,
        min(LANGUAGE_ROUTER_TOP_K, valid_count),
    )[1].astype(jnp.int32)
    return LanguageAddressDecision(
        selected_indices=result.selected_indices,
        node_probabilities=result.node_probabilities,
        node_scores=normalized_scores,
        score_margin=result.score_margin,
        entropy=result.entropy,
        top_k_indices=top_k,
    )


def _decision_from_ebt(
    result: EbtAddressResult,
    valid_node_mask: jax.Array,
) -> LanguageAddressDecision:
    valid_mask = jnp.asarray(valid_node_mask, dtype=jnp.bool_)
    valid_count = int(np.sum(np.asarray(valid_mask)))
    masked_logits = jnp.where(
        valid_mask[None, :],
        result.final_node_logits,
        jnp.asarray(-jnp.inf, dtype=jnp.float32),
    ).astype(jnp.float32)
    top_k = jax.lax.top_k(
        masked_logits,
        min(LANGUAGE_ROUTER_TOP_K, valid_count),
    )[1].astype(jnp.int32)
    sorted_logits = jnp.sort(masked_logits, axis=-1)[:, ::-1]
    margin = (
        jnp.full((sorted_logits.shape[0],), jnp.inf, dtype=jnp.float32)
        if valid_count == 1
        else (sorted_logits[:, 0] - sorted_logits[:, 1]).astype(jnp.float32)
    )
    entropy_terms = jnp.where(
        result.node_probabilities > 0.0,
        result.node_probabilities * jnp.log(result.node_probabilities),
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    return LanguageAddressDecision(
        selected_indices=result.selected_indices,
        node_probabilities=result.node_probabilities,
        node_scores=masked_logits,
        score_margin=margin,
        entropy=-jnp.sum(entropy_terms, axis=-1).astype(jnp.float32),
        top_k_indices=top_k,
    )


def _random_node_decision(
    prefix_batch: RouterBatch,
    valid_node_mask: jax.Array,
    seed: int,
) -> LanguageAddressDecision:
    identities = tuple(
        _active_prefix_identity(input_row, attention_row, target_row)
        for input_row, attention_row, target_row in zip(
            prefix_batch.input_ids,
            prefix_batch.attention_mask,
            prefix_batch.target_ids,
        )
    )
    valid_mask = np.asarray(valid_node_mask, dtype=np.bool_)
    task_node_mask = np.array(valid_mask, copy=True)
    task_node_mask[0] = False
    selection_mask = task_node_mask if np.any(task_node_mask) else valid_mask
    selected = deterministic_random_valid_node_indices(
        selection_mask,
        seed,
        identities,
    )
    node_capacity = valid_node_mask.shape[0]
    task_indices = np.flatnonzero(task_node_mask)
    root_indices = np.flatnonzero(valid_mask & ~task_node_mask)
    rankings = np.asarray(
        [
            (int(selected[row]),)
            + tuple(
                sorted(
                    (
                        int(index)
                        for index in task_indices
                        if index != selected[row]
                    ),
                    key=lambda index: sha256(
                        seed.to_bytes(8, byteorder="big", signed=False)
                        + identities[row].encode("ascii")
                        + index.to_bytes(4, byteorder="big", signed=False)
                    ).digest(),
                )
            )
            + tuple(
                int(index) for index in root_indices if index != selected[row]
            )
            for row in range(len(selected))
        ],
        dtype=np.int32,
    )
    probabilities = jax.nn.one_hot(selected, node_capacity, dtype=jnp.float32)
    scores = np.full((len(selected), node_capacity), -np.inf, dtype=np.float32)
    scores[np.arange(len(selected))[:, None], rankings] = np.arange(
        rankings.shape[1],
        0,
        -1,
        dtype=np.float32,
    )[None, :]
    return LanguageAddressDecision(
        selected_indices=jnp.asarray(selected, dtype=jnp.int32),
        node_probabilities=probabilities,
        node_scores=jnp.asarray(scores),
        score_margin=(
            jnp.full((len(selected),), jnp.inf, dtype=jnp.float32)
            if int(np.sum(np.asarray(valid_node_mask, dtype=np.bool_))) == 1
            else jnp.ones((len(selected),), dtype=jnp.float32)
        ),
        entropy=jnp.zeros((len(selected),), dtype=jnp.float32),
        top_k_indices=jnp.asarray(
            rankings[:, : min(LANGUAGE_ROUTER_TOP_K, int(np.sum(valid_mask)))],
            dtype=jnp.int32,
        ),
    )


def _stack_router_batches(batches: tuple[RouterBatch, ...]) -> RouterBatch:
    if not batches or any(not isinstance(batch, RouterBatch) for batch in batches):
        raise TypeError("router batches must contain RouterBatch values")
    arrays = _stack_language_batch_arrays(batches, "router")
    return RouterBatch(
        input_ids=arrays.input_ids,
        attention_mask=arrays.attention_mask,
        target_ids=arrays.target_ids,
        loss_mask=arrays.loss_mask,
    )


def _stack_competence_batches(
    batches: tuple[CompetenceBatch, ...],
) -> CompetenceBatch:
    if not batches or any(not isinstance(batch, CompetenceBatch) for batch in batches):
        raise TypeError("competence batches must contain CompetenceBatch values")
    arrays = _stack_language_batch_arrays(batches, "competence")
    return CompetenceBatch(
        input_ids=arrays.input_ids,
        attention_mask=arrays.attention_mask,
        target_ids=arrays.target_ids,
        loss_mask=arrays.loss_mask,
    )


def _validate_address_alignment(
    packed_memory: PackedLoraMemory,
    address_book: AddressBook,
) -> None:
    node_capacity = packed_memory.node_path_matrix.shape[0]
    if address_book.max_nodes != node_capacity:
        raise ValueError("address book and packed memory must share node capacity")
    if not np.array_equal(
        address_book.valid_node_mask,
        np.asarray(packed_memory.valid_node_mask, dtype=np.bool_),
    ):
        raise ValueError("address book and packed memory valid nodes must align")
    if not np.any(address_book.valid_node_mask):
        raise ValueError("routing requires at least one valid node")


def _validate_evaluation_alignment(
    graph: MemoryGraph[LoraEdge],
    packed_memory: PackedLoraMemory,
    address_book: AddressBook,
) -> None:
    if not isinstance(graph, MemoryGraph) or not graph.nodes:
        raise ValueError("router evaluation requires a nonempty MemoryGraph")
    _validate_address_alignment(packed_memory, address_book)
    valid_node_count = int(
        np.sum(np.asarray(packed_memory.valid_node_mask, dtype=np.bool_))
    )
    if len(graph.nodes) != valid_node_count:
        raise ValueError("graph nodes must match packed valid-node count")
    graph_node_ids = tuple(node.node_id for node in graph.nodes)
    if address_book.node_ids[:valid_node_count] != graph_node_ids:
        raise ValueError("graph and address-book node order must match")
    edge_count = len(graph.nodes) - 1
    packed_incidence = np.asarray(packed_memory.node_path_matrix)
    if not np.array_equal(
        packed_incidence[:valid_node_count, :edge_count],
        path_incidence_matrix(graph),
    ):
        raise ValueError("graph and packed path incidence must match")
    if np.any(packed_incidence[valid_node_count:, :] != 0.0) or np.any(
        packed_incidence[:, edge_count:] != 0.0
    ):
        raise ValueError("packed path incidence padding must be zero")


def _validated_decision(
    decision: LanguageAddressDecision,
    valid_node_mask: jax.Array,
    batch_size: int,
) -> LanguageAddressDecision:
    valid_nodes = np.asarray(valid_node_mask, dtype=np.bool_)
    node_count = valid_nodes.size
    valid_count = int(np.sum(valid_nodes))
    selected = np.asarray(decision.selected_indices)
    probabilities = np.asarray(decision.node_probabilities)
    scores = np.asarray(decision.node_scores)
    margins = np.asarray(decision.score_margin)
    entropy = np.asarray(decision.entropy)
    top_k = np.asarray(decision.top_k_indices)
    if selected.shape != (batch_size,) or selected.dtype.kind not in "iu":
        raise ValueError("selected indices must be integer [batch]")
    if probabilities.shape != (batch_size, node_count) or scores.shape != (
        batch_size,
        node_count,
    ):
        raise ValueError("router probabilities and scores must be [batch, nodes]")
    if np.any(~np.isfinite(probabilities)) or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise ValueError("router probabilities must be finite and in [0, 1]")
    if not np.allclose(np.sum(probabilities, axis=1), 1.0, atol=1e-6):
        raise ValueError("router probability rows must sum to one")
    if np.any(probabilities[:, ~valid_nodes] != 0.0):
        raise ValueError("invalid nodes must have exactly zero probability")
    if np.any(~np.isfinite(scores[:, valid_nodes])) or np.any(
        ~np.isneginf(scores[:, ~valid_nodes])
    ):
        raise ValueError("valid scores must be finite and invalid scores exactly -inf")
    if np.any((selected < 0) | (selected >= node_count)) or np.any(
        ~valid_nodes[selected]
    ):
        raise ValueError("selected indices must identify valid nodes")
    if np.any(np.argmax(scores, axis=1) != selected) or np.any(
        np.argmax(probabilities, axis=1) != selected
    ):
        raise ValueError("selected indices must maximize scores and probabilities")
    if margins.shape != (batch_size,) or np.any(np.isnan(margins)) or np.any(
        margins < 0.0
    ):
        raise ValueError("router margins must be nonnegative [batch]")
    sorted_valid_scores = np.sort(scores[:, valid_nodes], axis=1)[:, ::-1]
    expected_margins = (
        np.full((batch_size,), np.inf, dtype=np.float32)
        if valid_count == 1
        else sorted_valid_scores[:, 0] - sorted_valid_scores[:, 1]
    )
    if not np.allclose(margins, expected_margins):
        raise ValueError("router margins must equal the top-two score difference")
    if entropy.shape != (batch_size,) or np.any(~np.isfinite(entropy)) or np.any(
        entropy < 0.0
    ):
        raise ValueError("router entropy must be finite and nonnegative [batch]")
    expected_entropy = -np.sum(
        probabilities
        * np.log(np.where(probabilities > 0.0, probabilities, 1.0)),
        axis=1,
    )
    if not np.allclose(entropy, expected_entropy, atol=1e-6):
        raise ValueError("router entropy must match node probabilities")
    if (
        top_k.ndim != 2
        or top_k.shape[0] != batch_size
        or top_k.shape[1] != min(LANGUAGE_ROUTER_TOP_K, valid_count)
        or top_k.dtype.kind not in "iu"
    ):
        raise ValueError("top-k width must use the canonical valid-node limit")
    if np.any((top_k < 0) | (top_k >= node_count)) or np.any(
        ~valid_nodes[top_k]
    ):
        raise ValueError("top-k indices must identify valid nodes")
    if any(len(set(row.tolist())) != row.size for row in top_k):
        raise ValueError("top-k indices must be unique within each row")
    if np.any(top_k[:, 0] != selected):
        raise ValueError("the selected node must lead each top-k row")
    return decision


class _StackedLanguageBatchArrays(NamedTuple):
    input_ids: np.ndarray
    attention_mask: np.ndarray
    target_ids: np.ndarray
    loss_mask: np.ndarray


def _stack_language_batch_arrays(
    batches: tuple[RouterBatch, ...] | tuple[CompetenceBatch, ...],
    batch_name: str,
) -> _StackedLanguageBatchArrays:
    if len({batch.input_ids.shape[1] for batch in batches}) != 1:
        raise ValueError(f"{batch_name} batches must share a fixed sequence width")
    return _StackedLanguageBatchArrays(
        input_ids=np.concatenate(tuple(batch.input_ids for batch in batches)),
        attention_mask=np.concatenate(tuple(batch.attention_mask for batch in batches)),
        target_ids=np.concatenate(tuple(batch.target_ids for batch in batches)),
        loss_mask=np.concatenate(tuple(batch.loss_mask for batch in batches)),
    )


def _active_prefix_identity(
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    target_ids: np.ndarray,
) -> str:
    active_length = int(np.sum(attention_mask))
    return sha256(
        active_length.to_bytes(4, byteorder="big", signed=False)
        + np.asarray(input_ids[:active_length], dtype="<i4").tobytes()
        + np.asarray(target_ids[:active_length], dtype="<i4").tobytes()
    ).hexdigest()

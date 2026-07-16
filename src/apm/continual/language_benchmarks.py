"""Canonical language baselines, route evaluation, and addressing timing."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import time
from typing import Callable, Literal, Protocol, TypeVar, runtime_checkable

import jax
import numpy as np


StoredBaselineName = Literal[
    "frozen_base",
    "sequential_single_lora",
    "independent_root_lora",
    "vamp_oracle",
]
RouterBaselineName = Literal[
    "vamp_exhaustive",
    "vamp_hopfield",
    "vamp_ebt_uniform",
    "vamp_ebt_hopfield",
    "deterministic_random_node",
]
BaselineName = StoredBaselineName | RouterBaselineName
BaselineCategory = Literal["stored", "router"]


STORED_BASELINE_NAMES: tuple[StoredBaselineName, ...] = (
    "frozen_base",
    "sequential_single_lora",
    "independent_root_lora",
    "vamp_oracle",
)
ROUTER_BASELINE_NAMES: tuple[RouterBaselineName, ...] = (
    "vamp_exhaustive",
    "vamp_hopfield",
    "vamp_ebt_uniform",
    "vamp_ebt_hopfield",
    "deterministic_random_node",
)


@dataclass(frozen=True)
class BaselineSpec:
    """One canonical benchmark identifier, category, and scientific purpose."""

    name: BaselineName
    category: BaselineCategory
    purpose: str

    def __post_init__(self) -> None:
        expected_category = (
            "stored" if self.name in STORED_BASELINE_NAMES else "router"
        )
        if self.name not in STORED_BASELINE_NAMES + ROUTER_BASELINE_NAMES:
            raise ValueError(f"unknown language benchmark baseline: {self.name}")
        if self.category != expected_category:
            raise ValueError(
                f"baseline {self.name} belongs to category {expected_category}"
            )
        if not self.purpose:
            raise ValueError("baseline purpose must not be empty")


CANONICAL_BASELINE_MATRIX: tuple[BaselineSpec, ...] = (
    BaselineSpec("frozen_base", "stored", "no-adaptation floor"),
    BaselineSpec(
        "sequential_single_lora",
        "stored",
        "ordinary adapter forgetting baseline",
    ),
    BaselineSpec(
        "independent_root_lora",
        "stored",
        "per-task adapter ceiling without transfer",
    ),
    BaselineSpec("vamp_oracle", "stored", "task-oracle stored competence"),
    BaselineSpec(
        "vamp_exhaustive",
        "router",
        "gold-standard task-free prefix routing",
    ),
    BaselineSpec("vamp_hopfield", "router", "cheap content routing"),
    BaselineSpec(
        "vamp_ebt_uniform",
        "router",
        "continuous addressing without a content prior",
    ),
    BaselineSpec(
        "vamp_ebt_hopfield",
        "router",
        "content retrieval followed by continuous refinement",
    ),
    BaselineSpec(
        "deterministic_random_node",
        "router",
        "task-free routing negative control",
    ),
)


@dataclass(frozen=True)
class RouteExampleEvaluation:
    """Evaluator-only suffix and address metrics for one routed example."""

    selected_index: int
    task_oracle_index: int
    best_node_index: int
    selected_suffix_nll: float
    task_oracle_suffix_nll: float
    best_node_suffix_nll: float
    task_oracle_regret: float
    best_node_regret: float
    task_oracle_correct: bool
    top_k_task_oracle_hit: bool | None
    address_entropy: float | None
    top_two_probability_margin: float | None
    confusion_pair: tuple[int, int]

    def __post_init__(self) -> None:
        if any(
            type(index) is not int or index < 0
            for index in (
                self.selected_index,
                self.task_oracle_index,
                self.best_node_index,
            )
        ):
            raise ValueError("route indices must be nonnegative integers")
        nll_values = (
            self.selected_suffix_nll,
            self.task_oracle_suffix_nll,
            self.best_node_suffix_nll,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in nll_values):
            raise ValueError("suffix NLL values must be finite and nonnegative")
        expected_regrets = (
            self.selected_suffix_nll - self.task_oracle_suffix_nll,
            self.selected_suffix_nll - self.best_node_suffix_nll,
        )
        if not all(
            math.isclose(actual, expected, abs_tol=1e-7)
            for actual, expected in zip(
                (self.task_oracle_regret, self.best_node_regret),
                expected_regrets,
            )
        ):
            raise ValueError("route regrets must be suffix-NLL differences")
        if self.task_oracle_correct != (
            self.selected_index == self.task_oracle_index
        ):
            raise ValueError("task-oracle correctness must match route indices")
        optional_metrics = (
            self.address_entropy,
            self.top_two_probability_margin,
        )
        if any(
            value is not None and (not math.isfinite(value) or value < 0.0)
            for value in optional_metrics
        ):
            raise ValueError("address entropy and margin must be nonnegative")
        if self.confusion_pair != (
            self.task_oracle_index,
            self.selected_index,
        ):
            raise ValueError("confusion_pair must be (task oracle, selected)")


@dataclass(frozen=True)
class WilsonConfidenceInterval:
    """Validated two-sided 95% Wilson interval for one binomial proportion."""

    successes: int
    trials: int
    observed_rate: float
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if (
            type(self.successes) is not int
            or type(self.trials) is not int
            or self.trials <= 0
            or not 0 <= self.successes <= self.trials
        ):
            raise ValueError("Wilson counts must satisfy 0 <= successes <= trials")
        expected_rate = self.successes / self.trials
        if not math.isclose(self.observed_rate, expected_rate, abs_tol=1e-12):
            raise ValueError("observed_rate must equal successes divided by trials")
        if not 0.0 <= self.lower <= self.observed_rate <= self.upper <= 1.0:
            raise ValueError("Wilson bounds must contain the observed rate in [0, 1]")
        expected_lower, expected_upper = _wilson_95_bounds(
            self.successes,
            self.trials,
        )
        if not math.isclose(self.lower, expected_lower, abs_tol=1e-12) or not math.isclose(
            self.upper,
            expected_upper,
            abs_tol=1e-12,
        ):
            raise ValueError("Wilson bounds must match the validated count formula")


@dataclass(frozen=True)
class NegativeControlSummary:
    """Task-node accuracy, chance coverage, and leakage-audit decision."""

    task_node_count: int
    example_count: int
    correct_count: int
    observed_accuracy: float
    chance_accuracy: float
    confidence_interval: WilsonConfidenceInterval
    chance_rate_in_interval: bool
    leakage_audit_required: bool

    def __post_init__(self) -> None:
        if type(self.task_node_count) is not int or self.task_node_count <= 0:
            raise ValueError("task_node_count must be a positive integer")
        if (self.example_count, self.correct_count) != (
            self.confidence_interval.trials,
            self.confidence_interval.successes,
        ):
            raise ValueError("negative-control counts must match the confidence interval")
        if not math.isclose(
            self.observed_accuracy,
            self.confidence_interval.observed_rate,
            abs_tol=1e-12,
        ):
            raise ValueError("observed accuracy must match the confidence interval")
        expected_chance = 1.0 / self.task_node_count
        if not math.isclose(self.chance_accuracy, expected_chance, abs_tol=1e-12):
            raise ValueError("chance accuracy must equal one over task-node count")
        chance_is_covered = (
            self.confidence_interval.lower
            <= self.chance_accuracy
            <= self.confidence_interval.upper
        )
        if self.chance_rate_in_interval != chance_is_covered:
            raise ValueError("chance-rate coverage must match the Wilson bounds")
        expected_audit = (
            self.observed_accuracy > self.chance_accuracy
            and not chance_is_covered
        )
        if self.leakage_audit_required != expected_audit:
            raise ValueError("leakage audit must flag only material above-chance accuracy")


def wilson_95_confidence_interval(
    successes: int,
    trials: int,
) -> WilsonConfidenceInterval:
    """Return the two-sided 95% Wilson interval for binomial successes."""
    if (
        type(successes) is not int
        or type(trials) is not int
        or trials <= 0
        or not 0 <= successes <= trials
    ):
        raise ValueError("binomial counts must satisfy 0 <= successes <= trials")
    observed_rate = successes / trials
    lower, upper = _wilson_95_bounds(successes, trials)
    return WilsonConfidenceInterval(
        successes=successes,
        trials=trials,
        observed_rate=observed_rate,
        lower=lower,
        upper=upper,
    )


def summarize_negative_control(
    selected_indices: jax.Array | np.ndarray,
    task_oracle_indices: jax.Array | np.ndarray,
    task_node_count: int,
) -> NegativeControlSummary:
    """Summarize task-node accuracy against chance on a negative curriculum.

    Oracle indices identify non-root task nodes. A learned router may select
    the root, which counts as an ordinary incorrect prediction.
    """
    if type(task_node_count) is not int or task_node_count <= 0:
        raise ValueError("task_node_count must be a positive integer")
    selected = np.asarray(selected_indices)
    oracle = np.asarray(task_oracle_indices)
    if selected.ndim != 1 or selected.shape != oracle.shape or selected.size == 0:
        raise ValueError("selected and oracle indices must share nonempty [examples]")
    if selected.dtype.kind not in "iu" or oracle.dtype.kind not in "iu":
        raise TypeError("selected and oracle indices must contain integers")
    if np.any((selected < 0) | (selected > task_node_count)) or np.any(
        (oracle < 1) | (oracle > task_node_count)
    ):
        raise ValueError(
            "negative-control predictions must identify root/task nodes and "
            "oracles must identify task nodes"
        )
    correct_count = int(np.sum(selected == oracle))
    interval = wilson_95_confidence_interval(correct_count, int(selected.size))
    chance_accuracy = 1.0 / task_node_count
    chance_is_covered = interval.lower <= chance_accuracy <= interval.upper
    return NegativeControlSummary(
        task_node_count=task_node_count,
        example_count=int(selected.size),
        correct_count=correct_count,
        observed_accuracy=interval.observed_rate,
        chance_accuracy=chance_accuracy,
        confidence_interval=interval,
        chance_rate_in_interval=chance_is_covered,
        leakage_audit_required=(
            interval.observed_rate > chance_accuracy and not chance_is_covered
        ),
    )


def _wilson_95_bounds(successes: int, trials: int) -> tuple[float, float]:
    z_95 = 1.959963984540054
    observed_rate = successes / trials
    z_squared = z_95**2
    denominator = 1.0 + z_squared / trials
    center = (observed_rate + z_squared / (2.0 * trials)) / denominator
    half_width = z_95 / denominator * math.sqrt(
        observed_rate * (1.0 - observed_rate) / trials
        + z_squared / (4.0 * trials**2)
    )
    return (
        0.0 if successes == 0 else max(0.0, center - half_width),
        1.0 if successes == trials else min(1.0, center + half_width),
    )


def deterministic_random_valid_node_indices(
    valid_node_mask: jax.Array | np.ndarray,
    seed: int,
    example_identities: tuple[str, ...],
) -> np.ndarray:
    """Select a stable random valid node from each seed/example identity pair."""
    mask = np.asarray(valid_node_mask)
    if mask.ndim != 1 or mask.dtype != np.dtype(np.bool_):
        raise TypeError("valid_node_mask must be a rank-one boolean array")
    valid_indices = np.flatnonzero(mask)
    if valid_indices.size == 0:
        raise ValueError("valid_node_mask must contain at least one valid node")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    if not isinstance(example_identities, tuple) or not example_identities:
        raise ValueError("example_identities must be a nonempty tuple")
    if any(
        not isinstance(identity, str) or not identity
        for identity in example_identities
    ):
        raise ValueError("example identities must be nonempty strings")
    seed_bytes = seed.to_bytes(8, byteorder="big", signed=False)
    selected = np.asarray(
        [
            valid_indices[
                int.from_bytes(
                    hashlib.sha256(
                        seed_bytes + identity.encode("utf-8")
                    ).digest(),
                    byteorder="big",
                    signed=False,
                )
                % valid_indices.size
            ]
            for identity in example_identities
        ],
        dtype=np.int32,
    )
    selected.flags.writeable = False
    return selected


def evaluate_route_results(
    selected_indices: jax.Array | np.ndarray,
    suffix_nll_by_node: jax.Array | np.ndarray,
    valid_node_mask: jax.Array | np.ndarray,
    task_oracle_indices: jax.Array | np.ndarray,
    *,
    node_probabilities: jax.Array | np.ndarray | None = None,
    top_k_indices: jax.Array | np.ndarray | None = None,
) -> tuple[RouteExampleEvaluation, ...]:
    """Turn task-free route outputs into evaluator-only suffix metric inputs."""
    suffix_nll = np.asarray(suffix_nll_by_node, dtype=np.float64)
    if suffix_nll.ndim != 2 or suffix_nll.shape[0] == 0:
        raise ValueError("suffix_nll_by_node must have shape [batch, nodes]")
    batch_size, node_count = suffix_nll.shape
    valid_nodes = np.asarray(valid_node_mask)
    if valid_nodes.shape != (node_count,) or valid_nodes.dtype != np.dtype(np.bool_):
        raise TypeError("valid_node_mask must be boolean with shape [nodes]")
    if not np.any(valid_nodes):
        raise ValueError("at least one node must be valid")
    if np.any(~np.isfinite(suffix_nll[:, valid_nodes])) or np.any(
        suffix_nll[:, valid_nodes] < 0.0
    ):
        raise ValueError("valid-node suffix NLL values must be finite and nonnegative")

    selected = _integer_indices(selected_indices, "selected_indices", batch_size)
    task_oracle = _integer_indices(
        task_oracle_indices,
        "task_oracle_indices",
        batch_size,
    )
    _validate_valid_indices(selected, valid_nodes, "selected_indices")
    _validate_valid_indices(task_oracle, valid_nodes, "task_oracle_indices")
    probabilities = _validated_probabilities(
        node_probabilities,
        (batch_size, node_count),
        valid_nodes,
    )
    top_k = _validated_top_k(
        top_k_indices,
        batch_size,
        valid_nodes,
    )

    masked_nll = np.where(valid_nodes[None, :], suffix_nll, np.inf)
    best_indices = np.argmin(masked_nll, axis=1)
    rows = np.arange(batch_size)
    selected_nll = suffix_nll[rows, selected]
    task_oracle_nll = suffix_nll[rows, task_oracle]
    best_nll = suffix_nll[rows, best_indices]
    entropy = (
        None
        if probabilities is None
        else -np.sum(
            probabilities
            * np.log(np.where(probabilities > 0.0, probabilities, 1.0)),
            axis=1,
        )
    )
    margin = (
        None
        if probabilities is None
        else _top_two_probability_margins(probabilities[:, valid_nodes])
    )
    return tuple(
        RouteExampleEvaluation(
            selected_index=int(selected[row]),
            task_oracle_index=int(task_oracle[row]),
            best_node_index=int(best_indices[row]),
            selected_suffix_nll=float(selected_nll[row]),
            task_oracle_suffix_nll=float(task_oracle_nll[row]),
            best_node_suffix_nll=float(best_nll[row]),
            task_oracle_regret=float(selected_nll[row] - task_oracle_nll[row]),
            best_node_regret=float(selected_nll[row] - best_nll[row]),
            task_oracle_correct=bool(selected[row] == task_oracle[row]),
            top_k_task_oracle_hit=(
                None
                if top_k is None
                else bool(np.any(top_k[row] == task_oracle[row]))
            ),
            address_entropy=None if entropy is None else float(entropy[row]),
            top_two_probability_margin=(
                None if margin is None else float(margin[row])
            ),
            confusion_pair=(int(task_oracle[row]), int(selected[row])),
        )
        for row in range(batch_size)
    )


@dataclass(frozen=True)
class AddressingOperationCounts:
    """Static operation counts for one synchronized addressing measurement."""

    prefix_tokens: int
    candidates_available: int
    candidates_scored: int
    full_model_forward_equivalent_tokens: int
    base_forwards: int
    edge_evaluations: int
    hopfield_dot_products: int
    ebt_steps: int
    ebt_mask_size: int
    selected_execution_cost: int

    def __post_init__(self) -> None:
        values = tuple(getattr(self, name) for name in self.__dataclass_fields__)
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("addressing operation counts must be nonnegative integers")
        if self.prefix_tokens == 0 or self.candidates_available == 0:
            raise ValueError("prefix tokens and available candidates must be positive")
        if self.ebt_mask_size > self.candidates_available:
            raise ValueError("EBT mask size cannot exceed available candidates")
        if (self.ebt_steps == 0) != (self.ebt_mask_size == 0):
            raise ValueError("EBT steps and mask size must either both be zero or positive")


@dataclass(frozen=True)
class AddressingTiming:
    """Separate synchronized first-call and repeated warm address timings."""

    cold_compile_seconds: float
    warm_latency_samples_seconds: tuple[float, ...]
    warm_latency_seconds: float
    warm_throughput_examples_per_second: float
    batch_size: int
    operations: AddressingOperationCounts

    def __post_init__(self) -> None:
        if not math.isfinite(self.cold_compile_seconds) or self.cold_compile_seconds <= 0.0:
            raise ValueError("cold compile timing must be finite and positive")
        if not self.warm_latency_samples_seconds or any(
            not math.isfinite(value) or value <= 0.0
            for value in self.warm_latency_samples_seconds
        ):
            raise ValueError("warm latency samples must be finite and positive")
        expected_latency = math.fsum(self.warm_latency_samples_seconds) / len(
            self.warm_latency_samples_seconds
        )
        if not math.isclose(self.warm_latency_seconds, expected_latency, abs_tol=1e-12):
            raise ValueError("warm latency must be the mean of synchronized samples")
        if type(self.batch_size) is not int or self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        expected_throughput = self.batch_size / self.warm_latency_seconds
        if not math.isclose(
            self.warm_throughput_examples_per_second,
            expected_throughput,
            rel_tol=1e-12,
        ):
            raise ValueError("warm throughput must equal batch size divided by latency")
        if not isinstance(self.operations, AddressingOperationCounts):
            raise TypeError("operations must be AddressingOperationCounts")


@runtime_checkable
class _BlockUntilReady(Protocol):
    def block_until_ready(self) -> object:
        """Wait until this asynchronous result is materialized."""


ResultT = TypeVar("ResultT")


def time_synchronized_addressing(
    address: Callable[[], ResultT],
    operations: AddressingOperationCounts,
    *,
    batch_size: int,
    warm_repetitions: int = 5,
    clock: Callable[[], float] = time.perf_counter,
) -> AddressingTiming:
    """Measure one synchronized cold call separately from synchronized warm calls."""
    if not callable(address):
        raise TypeError("address must be callable")
    if not isinstance(operations, AddressingOperationCounts):
        raise TypeError("operations must be AddressingOperationCounts")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if type(warm_repetitions) is not int or warm_repetitions <= 0:
        raise ValueError("warm_repetitions must be a positive integer")

    cold_start = clock()
    _block_until_ready(address())
    cold_seconds = clock() - cold_start
    warm_samples: list[float] = []
    for _ in range(warm_repetitions):
        warm_start = clock()
        _block_until_ready(address())
        warm_samples.append(clock() - warm_start)
    samples = tuple(warm_samples)
    if cold_seconds <= 0.0 or any(sample <= 0.0 for sample in samples):
        raise ValueError("timing clock must advance for every synchronized call")
    warm_latency = math.fsum(samples) / len(samples)
    return AddressingTiming(
        cold_compile_seconds=cold_seconds,
        warm_latency_samples_seconds=samples,
        warm_latency_seconds=warm_latency,
        warm_throughput_examples_per_second=batch_size / warm_latency,
        batch_size=batch_size,
        operations=operations,
    )


def _integer_indices(
    values: jax.Array | np.ndarray,
    name: str,
    batch_size: int,
) -> np.ndarray:
    indices = np.asarray(values)
    if indices.shape != (batch_size,):
        raise ValueError(f"{name} must have shape ({batch_size},)")
    if indices.dtype.kind not in "iu":
        raise TypeError(f"{name} must contain integers")
    return indices.astype(np.int64, copy=False)


def _validate_valid_indices(
    indices: np.ndarray,
    valid_node_mask: np.ndarray,
    name: str,
) -> None:
    if np.any((indices < 0) | (indices >= valid_node_mask.size)):
        raise ValueError(f"{name} contains an index outside node capacity")
    if np.any(~valid_node_mask[indices]):
        raise ValueError(f"{name} must identify valid nodes")


def _validated_probabilities(
    values: jax.Array | np.ndarray | None,
    expected_shape: tuple[int, int],
    valid_node_mask: np.ndarray,
) -> np.ndarray | None:
    if values is None:
        return None
    probabilities = np.asarray(values, dtype=np.float64)
    if probabilities.shape != expected_shape:
        raise ValueError(f"node_probabilities must have shape {expected_shape}")
    if np.any(~np.isfinite(probabilities)) or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise ValueError("node probabilities must be finite and in [0, 1]")
    if np.any(probabilities[:, ~valid_node_mask] != 0.0):
        raise ValueError("invalid nodes must have exactly zero probability")
    if not np.allclose(np.sum(probabilities, axis=1), 1.0, atol=1e-6):
        raise ValueError("node probability rows must sum to one")
    return probabilities


def _validated_top_k(
    values: jax.Array | np.ndarray | None,
    batch_size: int,
    valid_node_mask: np.ndarray,
) -> np.ndarray | None:
    if values is None:
        return None
    top_k = np.asarray(values)
    if top_k.ndim != 2 or top_k.shape[0] != batch_size or top_k.shape[1] == 0:
        raise ValueError("top_k_indices must have shape [batch, nonempty top-k]")
    if top_k.dtype.kind not in "iu":
        raise TypeError("top_k_indices must contain integers")
    top_k = top_k.astype(np.int64, copy=False)
    if top_k.shape[1] > int(np.sum(valid_node_mask)):
        raise ValueError("top-k width cannot exceed the valid-node count")
    _validate_valid_indices(top_k.reshape(-1), valid_node_mask, "top_k_indices")
    if any(len(set(row.tolist())) != row.size for row in top_k):
        raise ValueError("top-k indices must be unique within each row")
    return top_k


def _top_two_probability_margins(probabilities: np.ndarray) -> np.ndarray:
    if probabilities.shape[1] == 1:
        return probabilities[:, 0]
    sorted_probabilities = np.sort(probabilities, axis=1)
    return sorted_probabilities[:, -1] - sorted_probabilities[:, -2]


def _block_until_ready(result: ResultT) -> None:
    for leaf in jax.tree_util.tree_leaves(result):
        if isinstance(leaf, _BlockUntilReady):
            leaf.block_until_ready()

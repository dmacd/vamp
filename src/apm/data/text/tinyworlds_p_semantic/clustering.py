"""Deterministic semantic diagnostics and spherical clustering."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
from typing import Mapping, Protocol, Sequence

import numpy as np

from apm.data.text.tinyworlds_p_semantic.contracts import (
    BENCHMARK_ID,
    Role,
    SemanticConstructionConfig,
)


class SemanticGridError(ValueError):
    """A frozen semantic-v1 construction or coherence gate failed."""


class _ClusterQualityConfig(Protocol):
    cluster_count: int
    minimum_nouns_per_cluster: int
    minimum_verbs_per_cluster: int
    maximum_centroid_pair_cosine: float


@dataclass(frozen=True, slots=True)
class WordVector:
    """One normalized role-word vector and its non-construction token mass."""

    role: Role
    word: str
    token_mass: int
    vector: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.role not in ("noun", "verb") or not self.word:
            raise ValueError("word vector requires a role and word")
        if type(self.token_mass) is not int or self.token_mass <= 0:
            raise ValueError("word vector mass must be positive")
        values = np.asarray(self.vector, dtype=np.float64)
        if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
            raise ValueError("word vector must be one finite vector")
        if not np.isclose(np.linalg.norm(values), 1.0, atol=1e-5):
            raise ValueError("word vectors must be L2 normalized")


@dataclass(frozen=True, slots=True)
class SphericalClustering:
    """One complete deterministic assignment with normalized centroids."""

    role: Role
    assignments: tuple[tuple[str, int], ...]
    centroids: tuple[tuple[float, ...], ...]
    cluster_masses: tuple[int, ...]
    iterations: int

    def __post_init__(self) -> None:
        if self.role not in ("noun", "verb"):
            raise ValueError("clustering role is invalid")
        if tuple(sorted(self.assignments)) != self.assignments:
            raise ValueError("clustering assignments must be sorted by word")
        if not self.centroids or len(self.centroids) != len(self.cluster_masses):
            raise ValueError("clustering centroid and mass counts differ")
        if any(value <= 0 for value in self.cluster_masses):
            raise ValueError("every semantic cluster must have positive mass")
        if self.iterations <= 0:
            raise ValueError("spherical clustering must perform an iteration")

    def margin_by_word(self, vectors: Mapping[str, Sequence[float]]) -> dict[str, float]:
        """Return assigned-centroid versus best-alternative cosine margins."""
        centroid_matrix = np.asarray(self.centroids, dtype=np.float64)
        return {
            word: _assigned_margin(
                np.asarray(vectors[word], dtype=np.float64),
                centroid_matrix,
                cluster,
            )
            for word, cluster in self.assignments
        }


@dataclass(frozen=True, slots=True)
class BoundaryClustering:
    """Final clusters plus words excluded during boundary/recluster passes."""

    clustering: SphericalClustering
    excluded_margins: tuple[tuple[str, float], ...]
    passes: int


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """Normalize one vector in float32, rejecting a zero or nonfinite norm."""
    values = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(values))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("cannot normalize a zero or nonfinite vector")
    return np.asarray(values / np.float32(norm), dtype=np.float32)


def normalized_centroid(vectors: np.ndarray) -> np.ndarray:
    """Return the L2-normalized float32 arithmetic mean of row vectors."""
    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("centroid requires a nonempty matrix")
    return l2_normalize(np.mean(values, axis=0, dtype=np.float32))


def compose_word_vector(
    target_anchor_embeddings: np.ndarray,
    context_embeddings: np.ndarray,
) -> np.ndarray:
    """Equally combine normalized role-anchor and archive-context centroids."""
    anchor_centroid = normalized_centroid(target_anchor_embeddings)
    context_centroid = normalized_centroid(context_embeddings)
    return l2_normalize(
        (anchor_centroid + context_centroid).astype(np.float32) * np.float32(0.5)
    )


def role_margin_quantile(
    target_anchor_embeddings: np.ndarray,
    opposite_anchor_embeddings: np.ndarray,
    context_embeddings: np.ndarray,
    quantile: float,
) -> float:
    """Measure the requested context-to-target versus opposite-role margin quantile."""
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("role-margin quantile must lie in [0, 1]")
    target = normalized_centroid(target_anchor_embeddings)
    opposite = normalized_centroid(opposite_anchor_embeddings)
    contexts = _normalized_rows(context_embeddings)
    margins = contexts @ target - contexts @ opposite
    return float(np.quantile(margins, quantile, method="linear"))


def deterministic_two_means_silhouette(context_embeddings: np.ndarray) -> float:
    """Return cosine silhouette for deterministic farthest-first spherical 2-means."""
    values = _normalized_rows(context_embeddings)
    count = values.shape[0]
    if count < 3:
        return 0.0
    first = 0
    second = min(
        range(1, count),
        key=lambda index: (float(values[index] @ values[first]), index),
    )
    centroids = np.stack((values[first], values[second]))
    assignments = np.zeros(count, dtype=np.int8)
    for _ in range(100):
        similarities = values @ centroids.T
        following = np.asarray(
            [0 if row[0] >= row[1] else 1 for row in similarities],
            dtype=np.int8,
        )
        if len(set(int(value) for value in following)) < 2:
            return 0.0
        new_centroids = np.stack(
            [normalized_centroid(values[following == cluster]) for cluster in range(2)]
        )
        if np.array_equal(following, assignments) and np.allclose(
            new_centroids, centroids, atol=1e-7
        ):
            assignments = following
            break
        assignments, centroids = following, new_centroids
    distances = np.maximum(0.0, 1.0 - values @ values.T)
    silhouettes = []
    for index, cluster in enumerate(assignments):
        same = np.flatnonzero(assignments == cluster)
        same = same[same != index]
        other = np.flatnonzero(assignments != cluster)
        if not len(same) or not len(other):
            silhouettes.append(0.0)
            continue
        within = float(np.mean(distances[index, same]))
        outside = float(np.mean(distances[index, other]))
        denominator = max(within, outside)
        silhouettes.append(0.0 if denominator == 0.0 else (outside - within) / denominator)
    return float(np.mean(silhouettes))


def capacity_constrained_spherical_kmeans(
    words: Sequence[WordVector],
    cluster_count: int,
    *,
    minimum_mass_fraction: float = 0.90,
    maximum_mass_fraction: float = 1.10,
    maximum_iterations: int = 100,
    benchmark_id: str = BENCHMARK_ID,
    repair_assignment_dead_ends: bool = False,
) -> SphericalClustering:
    """Cluster words with farthest-first seeds and deterministic mass-feasible assignment."""
    if type(cluster_count) is not int or cluster_count <= 1:
        raise ValueError("spherical clustering requires at least two clusters")
    if type(maximum_iterations) is not int or maximum_iterations <= 0:
        raise ValueError("maximum clustering iterations must be positive")
    if type(benchmark_id) is not str or not benchmark_id:
        raise ValueError("spherical clustering requires a benchmark identity")
    if type(repair_assignment_dead_ends) is not bool:
        raise TypeError("assignment repair choice must be boolean")
    canonical = tuple(sorted(words, key=lambda item: (item.word, item.role)))
    if len(canonical) < cluster_count:
        raise SemanticGridError("fewer role words than requested semantic clusters")
    roles = {item.role for item in canonical}
    if len(roles) != 1 or len({item.word for item in canonical}) != len(canonical):
        raise ValueError("spherical clustering requires unique words from one role")
    role = canonical[0].role
    matrix = _normalized_rows(np.asarray([item.vector for item in canonical]))
    masses = np.asarray([item.token_mass for item in canonical], dtype=np.int64)
    total_mass = int(np.sum(masses))
    target_mass = total_mass / cluster_count
    lower = minimum_mass_fraction * target_mass
    upper = maximum_mass_fraction * target_mass
    if int(np.max(masses)) > upper:
        raise SemanticGridError("one word exceeds the semantic cluster upper mass bound")
    seed_indexes = _farthest_first_indexes(
        canonical,
        matrix,
        cluster_count,
        role,
        benchmark_id,
    )
    centroids = matrix[np.asarray(seed_indexes)].copy()
    previous: tuple[int, ...] | None = None
    assignments: tuple[int, ...] = ()
    cluster_masses: tuple[int, ...] = ()
    for iteration in range(1, maximum_iterations + 1):
        assignments, cluster_masses = _mass_feasible_assignment(
            canonical,
            matrix,
            masses,
            centroids,
            lower,
            upper,
            role,
            benchmark_id,
            repair_assignment_dead_ends,
        )
        following = np.stack(
            [
                normalized_centroid(matrix[np.asarray(assignments) == cluster])
                for cluster in range(cluster_count)
            ]
        )
        if assignments == previous:
            centroids = following
            break
        previous, centroids = assignments, following
    else:
        raise SemanticGridError("spherical k-means exceeded its fixed iteration budget")
    return SphericalClustering(
        role=role,
        assignments=tuple(
            sorted(
                (item.word, assignments[index])
                for index, item in enumerate(canonical)
            )
        ),
        centroids=tuple(tuple(float(value) for value in row) for row in centroids),
        cluster_masses=cluster_masses,
        iterations=iteration,
    )


def semantic_first_spherical_kmeans(
    words: Sequence[WordVector],
    cluster_count: int,
    *,
    maximum_iterations: int = 100,
    benchmark_id: str = BENCHMARK_ID,
) -> SphericalClustering:
    """Cluster words by nearest cosine only, without consulting token mass."""
    if type(cluster_count) is not int or cluster_count <= 1:
        raise ValueError("spherical clustering requires at least two clusters")
    if type(maximum_iterations) is not int or maximum_iterations <= 0:
        raise ValueError("maximum clustering iterations must be positive")
    if type(benchmark_id) is not str or not benchmark_id:
        raise ValueError("spherical clustering requires a benchmark identity")
    canonical = tuple(sorted(words, key=lambda item: (item.word, item.role)))
    if len(canonical) < cluster_count:
        raise SemanticGridError("fewer role words than requested semantic clusters")
    roles = {item.role for item in canonical}
    if len(roles) != 1 or len({item.word for item in canonical}) != len(canonical):
        raise ValueError("spherical clustering requires unique words from one role")
    role = canonical[0].role
    matrix = _normalized_rows(np.asarray([item.vector for item in canonical]))
    seed_indexes = _farthest_first_indexes(
        canonical,
        matrix,
        cluster_count,
        role,
        benchmark_id,
    )
    centroids = matrix[np.asarray(seed_indexes)].copy()
    previous: tuple[int, ...] | None = None
    assignments: tuple[int, ...] = ()
    for iteration in range(1, maximum_iterations + 1):
        similarities = matrix @ centroids.T
        assignments = tuple(
            min(
                range(cluster_count),
                key=lambda cluster: (
                    -float(similarities[index, cluster]),
                    _tie_hash(
                        role,
                        "semantic-nearest-assignment",
                        f"{item.word}\0{cluster}",
                        benchmark_id,
                    ),
                    cluster,
                ),
            )
            for index, item in enumerate(canonical)
        )
        if any(cluster not in assignments for cluster in range(cluster_count)):
            raise SemanticGridError("nearest-centroid assignment emptied a semantic cluster")
        following = np.stack(
            [
                normalized_centroid(matrix[np.asarray(assignments) == cluster])
                for cluster in range(cluster_count)
            ]
        )
        if assignments == previous:
            centroids = following
            break
        previous, centroids = assignments, following
    else:
        raise SemanticGridError("spherical k-means exceeded its fixed iteration budget")
    cluster_masses = tuple(
        sum(
            item.token_mass
            for item, assignment in zip(canonical, assignments)
            if assignment == cluster
        )
        for cluster in range(cluster_count)
    )
    return SphericalClustering(
        role=role,
        assignments=tuple(
            sorted(
                (item.word, assignments[index])
                for index, item in enumerate(canonical)
            )
        ),
        centroids=tuple(tuple(float(value) for value in row) for row in centroids),
        cluster_masses=cluster_masses,
        iterations=iteration,
    )


def cluster_with_boundary_exclusions(
    words: Sequence[WordVector],
    config: SemanticConstructionConfig,
    *,
    benchmark_id: str = BENCHMARK_ID,
    repair_assignment_dead_ends: bool = False,
) -> BoundaryClustering:
    """Exclude low nearest-cluster margins and recluster for at most five passes."""
    retained = {item.word: item for item in words}
    excluded: dict[str, float] = {}
    for pass_index in range(config.maximum_exclusion_passes + 1):
        clustering = capacity_constrained_spherical_kmeans(
            tuple(retained.values()),
            config.cluster_count,
            minimum_mass_fraction=config.minimum_cluster_mass_fraction,
            maximum_mass_fraction=config.maximum_cluster_mass_fraction,
            maximum_iterations=config.maximum_centroid_iterations,
            benchmark_id=benchmark_id,
            repair_assignment_dead_ends=repair_assignment_dead_ends,
        )
        vectors = {word: item.vector for word, item in retained.items()}
        margins = clustering.margin_by_word(vectors)
        failing = tuple(
            sorted(
                word for word, margin in margins.items()
                if margin < config.minimum_cluster_margin
            )
        )
        if not failing:
            return BoundaryClustering(
                clustering=clustering,
                excluded_margins=tuple(sorted(excluded.items())),
                passes=pass_index,
            )
        if pass_index == config.maximum_exclusion_passes:
            raise SemanticGridError(
                "cluster boundary exclusions did not converge within five passes"
            )
        excluded.update((word, margins[word]) for word in failing)
        retained = {word: item for word, item in retained.items() if word not in failing}
        if len(retained) < config.cluster_count:
            raise SemanticGridError("boundary exclusions emptied the semantic grid")
    raise AssertionError("boundary exclusion loop must return or raise")


def validate_cluster_quality(
    result: SphericalClustering,
    config: _ClusterQualityConfig,
) -> None:
    """Apply per-role size and inter-centroid coherence gates."""
    counts = np.bincount(
        np.asarray([cluster for _, cluster in result.assignments]),
        minlength=config.cluster_count,
    )
    minimum_count = (
        config.minimum_nouns_per_cluster
        if result.role == "noun"
        else config.minimum_verbs_per_cluster
    )
    if np.any(counts < minimum_count):
        raise SemanticGridError(
            f"{result.role} semantic cluster contains fewer than {minimum_count} words"
        )
    centroids = np.asarray(result.centroids, dtype=np.float64)
    maximum_pair = max(
        float(centroids[left] @ centroids[right])
        for left in range(len(centroids))
        for right in range(left + 1, len(centroids))
    )
    if maximum_pair >= config.maximum_centroid_pair_cosine:
        raise SemanticGridError(
            "semantic cluster centroids violate the maximum pair cosine gate"
        )


def _normalized_rows(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("semantic embeddings must be a nonempty matrix")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.all(np.isfinite(matrix)) or np.any(norms <= 0.0):
        raise ValueError("semantic embeddings must be finite and nonzero")
    return np.asarray(matrix / norms, dtype=np.float32)


def _tie_hash(
    role: Role,
    namespace: str,
    word: str,
    benchmark_id: str = BENCHMARK_ID,
) -> str:
    return sha256(
        f"{benchmark_id}\0{role}\0{namespace}\0{word}".encode("utf-8")
    ).hexdigest()


def _farthest_first_indexes(
    words: Sequence[WordVector],
    matrix: np.ndarray,
    cluster_count: int,
    role: Role,
    benchmark_id: str,
) -> tuple[int, ...]:
    first = min(
        range(len(words)),
        key=lambda index: (
            _tie_hash(role, "centroid-seed", words[index].word, benchmark_id),
            words[index].word,
        ),
    )
    selected = [first]
    while len(selected) < cluster_count:
        following = min(
            (index for index in range(len(words)) if index not in selected),
            key=lambda index: (
                max(float(matrix[index] @ matrix[seed]) for seed in selected),
                _tie_hash(
                    role,
                    "centroid-farthest",
                    words[index].word,
                    benchmark_id,
                ),
                words[index].word,
            ),
        )
        selected.append(following)
    return tuple(selected)


def _mass_feasible_assignment(
    words: Sequence[WordVector],
    matrix: np.ndarray,
    masses: np.ndarray,
    centroids: np.ndarray,
    lower: float,
    upper: float,
    role: Role,
    benchmark_id: str,
    repair_dead_ends: bool,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    count = len(words)
    cluster_count = len(centroids)
    assignments = [-1] * count
    cluster_masses = [0] * cluster_count
    order = sorted(
        range(count),
        key=lambda index: (
            -int(masses[index]),
            _tie_hash(role, "mass-order", words[index].word, benchmark_id),
            words[index].word,
        ),
    )
    remaining_mass = int(np.sum(masses))
    for position, index in enumerate(order):
        mass = int(masses[index])
        remaining_after = remaining_mass - mass
        remaining_count = count - position - 1
        similarities = matrix[index] @ centroids.T
        feasible = []
        for cluster in range(cluster_count):
            proposed = tuple(
                value + (mass if current == cluster else 0)
                for current, value in enumerate(cluster_masses)
            )
            deficits = tuple(max(0.0, lower - value) for value in proposed)
            upper_capacity = sum(max(0.0, upper - value) for value in proposed)
            if (
                proposed[cluster] <= upper + 1e-9
                and sum(deficits) <= remaining_after + 1e-9
                and (
                    not repair_dead_ends
                    or sum(value > 1e-9 for value in deficits) <= remaining_count
                )
                and upper_capacity + 1e-9 >= remaining_after
            ):
                feasible.append(cluster)
        if not feasible:
            repair = (
                _single_reassignment_repair(
                    words,
                    matrix,
                    masses,
                    centroids,
                    assignments,
                    cluster_masses,
                    index,
                    mass,
                    remaining_after,
                    remaining_count,
                    lower,
                    upper,
                    role,
                    benchmark_id,
                )
                if repair_dead_ends
                else None
            )
            if repair is None:
                raise SemanticGridError(
                    "descending-mass assignment cannot satisfy fixed cluster mass bounds"
                )
            selected, moved_index, target = repair
            source = assignments[moved_index]
            if source < 0:
                raise AssertionError("assignment repair selected an unassigned word")
            moved_mass = int(masses[moved_index])
            assignments[moved_index] = target
            cluster_masses[source] -= moved_mass
            cluster_masses[target] += moved_mass
        else:
            selected = min(
                feasible,
                key=lambda cluster: (
                    -float(similarities[cluster]),
                    _tie_hash(
                        role,
                        "assignment-tie",
                        f"{words[index].word}\0{cluster}",
                        benchmark_id,
                    ),
                    cluster,
                ),
            )
        assignments[index] = selected
        cluster_masses[selected] += mass
        remaining_mass = remaining_after
    if any(not lower - 1e-9 <= value <= upper + 1e-9 for value in cluster_masses):
        raise SemanticGridError("final semantic cluster masses violate fixed bounds")
    if any(cluster not in assignments for cluster in range(cluster_count)):
        raise SemanticGridError("capacity assignment produced an empty semantic cluster")
    return tuple(assignments), tuple(cluster_masses)


def _single_reassignment_repair(
    words: Sequence[WordVector],
    matrix: np.ndarray,
    masses: np.ndarray,
    centroids: np.ndarray,
    assignments: Sequence[int],
    cluster_masses: Sequence[int],
    current_index: int,
    current_mass: int,
    remaining_mass: int,
    remaining_count: int,
    lower: float,
    upper: float,
    role: Role,
    benchmark_id: str,
) -> tuple[int, int, int] | None:
    """Repair one discrete packing dead end by moving one prior word."""
    current_similarities = matrix[current_index] @ centroids.T
    candidates: list[tuple[float, str, int, int, int]] = []
    for selected in range(len(centroids)):
        if cluster_masses[selected] + current_mass > upper + 1e-9:
            continue
        for moved_index, source in enumerate(assignments):
            if source < 0:
                continue
            moved_mass = int(masses[moved_index])
            moved_similarities = matrix[moved_index] @ centroids.T
            for target in range(len(centroids)):
                if target == source:
                    continue
                proposed = list(cluster_masses)
                proposed[source] -= moved_mass
                proposed[target] += moved_mass
                proposed[selected] += current_mass
                deficits = tuple(max(0.0, lower - value) for value in proposed)
                capacity = sum(max(0.0, upper - value) for value in proposed)
                if (
                    max(proposed) > upper + 1e-9
                    or sum(deficits) > remaining_mass + 1e-9
                    or sum(value > 1e-9 for value in deficits) > remaining_count
                    or capacity + 1e-9 < remaining_mass
                ):
                    continue
                score = (
                    float(current_similarities[selected])
                    + float(moved_similarities[target])
                    - float(moved_similarities[source])
                )
                tie = _tie_hash(
                    role,
                    "assignment-repair",
                    f"{words[current_index].word}\0{selected}\0"
                    f"{words[moved_index].word}\0{target}",
                    benchmark_id,
                )
                candidates.append((-score, tie, selected, moved_index, target))
    if not candidates:
        return None
    _, _, selected, moved_index, target = min(candidates)
    return selected, moved_index, target


def _assigned_margin(
    vector: np.ndarray,
    centroids: np.ndarray,
    assigned_cluster: int,
) -> float:
    similarities = tuple(float(value) for value in centroids @ vector)
    if len(similarities) < 2 or not 0 <= assigned_cluster < len(similarities):
        raise ValueError("cluster margin requires at least two centroids")
    return similarities[assigned_cluster] - max(
        value
        for cluster, value in enumerate(similarities)
        if cluster != assigned_cluster
    )


__all__ = [
    "BoundaryClustering",
    "SemanticGridError",
    "SphericalClustering",
    "WordVector",
    "capacity_constrained_spherical_kmeans",
    "cluster_with_boundary_exclusions",
    "compose_word_vector",
    "deterministic_two_means_silhouette",
    "l2_normalize",
    "normalized_centroid",
    "role_margin_quantile",
    "semantic_first_spherical_kmeans",
    "validate_cluster_quality",
]

"""Deterministic bucket, topology, split, visibility, and control algorithms."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from bisect import bisect_left
from fractions import Fraction
from hashlib import sha256
from itertools import combinations
import math
from typing import Literal

from apm.data.text.tinyworlds_p.contracts import (
    BENCHMARK_ID,
    ControlSelection,
    PartitionPreset,
    SplitLabel,
    WORLD_LABELS,
    WordBucket,
    WorldCell,
)


MarginalName = Literal["source", "feature", "adjective_bucket", "length_bin"]


class PartitionGateError(ValueError):
    """A predeclared partition quality or leakage gate did not pass."""


@dataclass(frozen=True, slots=True)
class AllocationGroup:
    """The compact recipe and nuisance evidence used by deterministic allocation."""

    normalized_sha256: str
    active_token_count: int
    canonical_token_count: int
    noun: str
    verb: str
    adjective: str
    noun_bucket: int
    verb_bucket: int
    adjective_bucket: int
    source: str
    feature_signature: str

    def __post_init__(self) -> None:
        if (
            type(self.normalized_sha256) is not str
            or len(self.normalized_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.normalized_sha256)
        ):
            raise ValueError("allocation group requires a lowercase SHA-256")
        if any(
            type(value) is not int or value <= 0
            for value in (self.active_token_count, self.canonical_token_count)
        ):
            raise ValueError("allocation token counts must be positive")
        if any(
            type(value) is not int or value < 0
            for value in (self.noun_bucket, self.verb_bucket, self.adjective_bucket)
        ):
            raise ValueError("allocation bucket indexes must be nonnegative")
        if any(
            type(value) is not str or not value
            for value in (
                self.noun,
                self.verb,
                self.adjective,
                self.source,
                self.feature_signature,
            )
        ):
            raise ValueError("allocation categorical values must be nonempty")

    @property
    def length_bin(self) -> str:
        """Return the predeclared canonical-story token-length stratum."""
        return token_length_bin(self.canonical_token_count)

    @property
    def marginals(self) -> tuple[tuple[MarginalName, str], ...]:
        """Return every multi-marginal allocator category in canonical order."""
        return (
            ("source", self.source),
            ("feature", self.feature_signature),
            ("adjective_bucket", str(self.adjective_bucket)),
            ("length_bin", self.length_bin),
        )

    @property
    def full_stratum(self) -> tuple[str, str, str, str]:
        """Return the joint nuisance stratum used for matched controls."""
        return (
            self.source,
            self.feature_signature,
            str(self.adjective_bucket),
            self.length_bin,
        )


@dataclass(frozen=True, slots=True)
class CellStatistics:
    """Token/group totals and token-weighted nuisance counts for one grid cell."""

    noun_bucket: int
    verb_bucket: int
    active_token_count: int
    group_count: int
    nuisance_token_counts: tuple[tuple[str, str, int], ...]

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value < 0
            for value in (
                self.noun_bucket,
                self.verb_bucket,
                self.active_token_count,
                self.group_count,
            )
        ):
            raise ValueError("cell statistics indexes and counts must be nonnegative")
        if type(self.nuisance_token_counts) is not tuple or any(
            type(dimension) is not str
            or not dimension
            or type(category) is not str
            or not category
            or type(count) is not int
            or count < 0
            for dimension, category, count in self.nuisance_token_counts
        ):
            raise ValueError("cell nuisance counts must be canonical triples")
        if tuple(sorted(self.nuisance_token_counts)) != self.nuisance_token_counts:
            raise ValueError("cell nuisance counts must be sorted")


@dataclass(frozen=True, slots=True)
class ControlDiagnostics:
    """Measured target/control differences for one strict matched selection."""

    token_relative_error: float
    maximum_source_feature_prevalence_error: float
    maximum_adjective_length_prevalence_error: float
    mean_length_relative_error: float


def token_length_bin(token_count: int) -> str:
    """Map a positive canonical token count to the fixed four length strata."""
    if type(token_count) is not int or token_count <= 0:
        raise ValueError("token_count must be positive")
    if token_count <= 64:
        return "le64"
    if token_count <= 128:
        return "65-128"
    if token_count <= 192:
        return "129-192"
    return "gt192"


def balance_word_buckets(
    word_token_mass: Mapping[str, int],
    namespace: Literal["noun", "verb", "adjective"],
    bucket_count: int,
    identity_sha256: str,
    *,
    public_seed: int = 0,
) -> tuple[WordBucket, ...]:
    """Greedily balance largest word masses with namespaced deterministic ties."""
    if namespace not in ("noun", "verb", "adjective"):
        raise ValueError("unknown word-bucket namespace")
    if type(bucket_count) is not int or bucket_count <= 0:
        raise ValueError("bucket_count must be positive")
    _require_identity(identity_sha256)
    if type(public_seed) is not int or public_seed < 0:
        raise ValueError("public_seed must be nonnegative")
    masses = {
        word: count
        for word, count in word_token_mass.items()
        if type(word) is str and word and type(count) is int and count > 0
    }
    if len(masses) != len(word_token_mass):
        raise ValueError("word masses require unique nonempty words and positive integers")
    if len(masses) < bucket_count:
        raise PartitionGateError(
            f"{namespace} vocabulary has {len(masses)} words for {bucket_count} buckets"
        )
    namespace_seed = _namespace_hash(
        identity_sha256,
        public_seed,
        f"bucket-order:{namespace}",
        "",
    )
    ordered_words = tuple(
        sorted(
            masses,
            key=lambda word: (
                -masses[word],
                _namespace_hash(namespace_seed, public_seed, "word", word),
                word,
            ),
        )
    )
    bucket_masses = [0] * bucket_count
    bucket_words: list[list[str]] = [[] for _ in range(bucket_count)]
    for word in ordered_words:
        selected_bucket = min(
            range(bucket_count),
            key=lambda index: (
                bucket_masses[index],
                _namespace_hash(
                    namespace_seed,
                    public_seed,
                    "bucket-tie",
                    f"{word}\0{index}",
                ),
                index,
            ),
        )
        bucket_masses[selected_bucket] += masses[word]
        bucket_words[selected_bucket].append(word)
    return tuple(
        WordBucket(
            namespace=namespace,
            index=index,
            token_mass=bucket_masses[index],
            words=tuple(sorted(bucket_words[index])),
        )
        for index in range(bucket_count)
    )


def bucket_word_lookup(buckets: Sequence[WordBucket]) -> dict[str, int]:
    """Invert one complete namespace of buckets while rejecting overlap."""
    if not buckets:
        raise ValueError("bucket lookup requires at least one bucket")
    namespaces = {bucket.namespace for bucket in buckets}
    if len(namespaces) != 1:
        raise ValueError("bucket lookup requires exactly one namespace")
    pairs = tuple((word, bucket.index) for bucket in buckets for word in bucket.words)
    lookup = dict(pairs)
    if len(lookup) != len(pairs):
        raise ValueError("word appears in more than one bucket")
    return lookup


def summarize_cells(groups: Iterable[AllocationGroup]) -> tuple[CellStatistics, ...]:
    """Aggregate token-weighted grid and nuisance counts from eligible groups."""
    token_counts: Counter[tuple[int, int]] = Counter()
    group_counts: Counter[tuple[int, int]] = Counter()
    nuisance: Counter[tuple[int, int, str, str]] = Counter()
    for group in groups:
        if type(group) is not AllocationGroup:
            raise TypeError("cell summaries require AllocationGroup values")
        key = (group.noun_bucket, group.verb_bucket)
        token_counts[key] += group.active_token_count
        group_counts[key] += 1
        for dimension, category in group.marginals:
            nuisance[(key[0], key[1], dimension, category)] += group.active_token_count
    return tuple(
        CellStatistics(
            noun_bucket=noun_bucket,
            verb_bucket=verb_bucket,
            active_token_count=token_counts[(noun_bucket, verb_bucket)],
            group_count=group_counts[(noun_bucket, verb_bucket)],
            nuisance_token_counts=tuple(
                sorted(
                    (dimension, category, count)
                    for (
                        current_noun,
                        current_verb,
                        dimension,
                        category,
                    ),
                    count in nuisance.items()
                    if (current_noun, current_verb) == (noun_bucket, verb_bucket)
                )
            ),
        )
        for noun_bucket, verb_bucket in sorted(token_counts)
    )


def select_world_cells(
    cell_statistics: Sequence[CellStatistics],
    bucket_count: int,
    identity_sha256: str,
    *,
    public_seed: int = 0,
    median_tolerance: float = 0.10,
) -> tuple[WorldCell, ...]:
    """Select the minimum-imbalance 2x2 corner plus an unrelated fifth cell."""
    if type(bucket_count) is not int or bucket_count < 3:
        raise ValueError("world topology requires at least three buckets")
    _require_identity(identity_sha256)
    if not math.isfinite(median_tolerance) or not 0.0 < median_tolerance < 1.0:
        raise ValueError("median_tolerance must lie in (0, 1)")
    by_cell = {
        (statistics.noun_bucket, statistics.verb_bucket): statistics
        for statistics in cell_statistics
    }
    if len(by_cell) != len(cell_statistics):
        raise ValueError("cell statistics must be unique by bucket pair")
    candidates = (
        (
            rows,
            columns,
            extra_row,
            extra_column,
            (
                (rows[0], columns[0]),
                (rows[1], columns[0]),
                (rows[1], columns[1]),
                (rows[0], columns[1]),
                (extra_row, extra_column),
            ),
        )
        for rows in combinations(range(bucket_count), 2)
        for columns in combinations(range(bucket_count), 2)
        for extra_row in range(bucket_count)
        if extra_row not in rows
        for extra_column in range(bucket_count)
        if extra_column not in columns
    )
    scored = tuple(
        (
            _cell_imbalance_score(tuple(by_cell[cell] for cell in cells)),
            _namespace_hash(
                identity_sha256,
                public_seed,
                "cell-topology-tie",
                ":".join(f"{noun},{verb}" for noun, verb in cells),
            ),
            cells,
        )
        for _, _, _, _, cells in candidates
        if all(cell in by_cell and by_cell[cell].active_token_count > 0 for cell in cells)
    )
    if not scored:
        raise PartitionGateError("no complete nonempty five-cell topology exists")
    _, _, selected = min(scored)
    selected_statistics = tuple(by_cell[cell] for cell in selected)
    median_tokens = sorted(item.active_token_count for item in selected_statistics)[2]
    lower = median_tokens * (1.0 - median_tolerance)
    upper = median_tokens * (1.0 + median_tolerance)
    if any(
        not lower <= statistics.active_token_count <= upper
        for statistics in selected_statistics
    ):
        counts = tuple(item.active_token_count for item in selected_statistics)
        raise PartitionGateError(
            "minimum-imbalance selected cells exceed the median token gate: "
            f"counts={counts}, median={median_tokens}, tolerance={median_tolerance}"
        )
    return tuple(
        WorldCell(
            label=label,
            noun_bucket=statistics.noun_bucket,
            verb_bucket=statistics.verb_bucket,
            active_token_count=statistics.active_token_count,
            group_count=statistics.group_count,
        )
        for label, statistics in zip(WORLD_LABELS, selected_statistics, strict=True)
    )


def require_component_visibility(
    groups: Iterable[AllocationGroup],
    cells: Sequence[WorldCell],
    minimum_outside_groups: int,
) -> tuple[tuple[str, str, int], ...]:
    """Require each selected-world noun and verb to remain visible in base groups."""
    if type(minimum_outside_groups) is not int or minimum_outside_groups < 0:
        raise ValueError("minimum_outside_groups must be nonnegative")
    selected_cells = {(cell.noun_bucket, cell.verb_bucket) for cell in cells}
    selected_components: set[tuple[str, str]] = set()
    outside_counts: Counter[tuple[str, str]] = Counter()
    for group in groups:
        if (group.noun_bucket, group.verb_bucket) in selected_cells:
            selected_components.update((("noun", group.noun), ("verb", group.verb)))
        else:
            outside_counts[("noun", group.noun)] += 1
            outside_counts[("verb", group.verb)] += 1
    visibility = tuple(
        (role, word, outside_counts[(role, word)])
        for role, word in sorted(selected_components)
    )
    failures = tuple(item for item in visibility if item[2] < minimum_outside_groups)
    if failures:
        raise PartitionGateError(
            "selected-world components lack required outside story groups: "
            + repr(failures[:20])
        )
    return visibility


def allocate_stratified_splits(
    groups: Sequence[AllocationGroup],
    weights: tuple[int, int, int],
    identity_sha256: str,
    namespace: str,
) -> dict[str, SplitLabel]:
    """Allocate indivisible groups while balancing token mass and four marginals."""
    if not groups:
        raise PartitionGateError(f"cannot split empty allocation domain {namespace!r}")
    if (
        type(weights) is not tuple
        or len(weights) != 3
        or any(type(weight) is not int or weight <= 0 for weight in weights)
    ):
        raise ValueError("split weights must be three positive integers")
    _require_identity(identity_sha256)
    labels: tuple[SplitLabel, SplitLabel, SplitLabel] = (
        "train",
        "validation",
        "test",
    )
    total_weight = sum(weights)
    total_tokens = sum(group.active_token_count for group in groups)
    marginal_totals: Counter[tuple[MarginalName, str]] = Counter()
    for group in groups:
        for marginal in group.marginals:
            marginal_totals[marginal] += group.active_token_count
    split_tokens: Counter[SplitLabel] = Counter()
    split_marginals: Counter[tuple[SplitLabel, MarginalName, str]] = Counter()
    assignments: dict[str, SplitLabel] = {}
    ordered = sorted(
        groups,
        key=lambda group: (
            -group.active_token_count,
            _namespace_hash(
                identity_sha256,
                0,
                f"split-order:{namespace}",
                group.normalized_sha256,
            ),
            group.normalized_sha256,
        ),
    )
    for group in ordered:
        scored_labels = tuple(
            (
                _allocator_score(
                    label,
                    weight,
                    total_weight,
                    total_tokens,
                    marginal_totals,
                    split_tokens,
                    split_marginals,
                    group,
                ),
                _namespace_hash(
                    identity_sha256,
                    0,
                    f"split-tie:{namespace}",
                    f"{group.normalized_sha256}\0{label}",
                ),
                label,
            )
            for label, weight in zip(labels, weights, strict=True)
        )
        selected_label = min(scored_labels)[2]
        assignments[group.normalized_sha256] = selected_label
        split_tokens[selected_label] += group.active_token_count
        for dimension, category in group.marginals:
            split_marginals[(selected_label, dimension, category)] += (
                group.active_token_count
            )
    if len(assignments) != len(groups):
        raise ValueError("allocation group hashes must be unique")
    if set(assignments.values()) != set(labels):
        raise PartitionGateError(
            f"allocation domain {namespace!r} did not populate every split"
        )
    return assignments


def select_matched_control(
    target_groups: Sequence[AllocationGroup],
    row_candidates: Sequence[AllocationGroup],
    column_candidates: Sequence[AllocationGroup],
    world: str,
    split: Literal["validation", "test"],
    identity_sha256: str,
    preset: PartitionPreset,
) -> tuple[ControlSelection, ControlDiagnostics]:
    """Select and strictly validate one half-row/half-column matched control."""
    if not target_groups:
        raise PartitionGateError("matched control target must not be empty")
    if world not in WORLD_LABELS or split not in ("validation", "test"):
        raise ValueError("matched controls require a world validation/test split")
    _require_identity(identity_sha256)
    target_count = len(target_groups)
    row_count = target_count // 2
    column_count = target_count - row_count
    target_strata = Counter(group.full_stratum for group in target_groups)
    selected_row = _select_control_arm(
        row_candidates,
        target_strata,
        row_count,
        target_count,
        identity_sha256,
        f"control:{world}:{split}:row",
    )
    selected_column = _select_control_arm(
        column_candidates,
        target_strata,
        column_count,
        target_count,
        identity_sha256,
        f"control:{world}:{split}:column",
    )
    selected = _improve_control_marginal_match(
        tuple(selected_row + selected_column),
        tuple(row_candidates),
        tuple(column_candidates),
        row_count,
        target_groups,
        identity_sha256,
        f"control:{world}:{split}:marginal-swap",
        preset,
    )
    selected = _improve_control_token_match(
        selected,
        tuple(row_candidates),
        tuple(column_candidates),
        row_count,
        sum(group.active_token_count for group in target_groups),
        identity_sha256,
        f"control:{world}:{split}:token-swap",
    )
    selected = _improve_control_numeric_match(
        selected,
        tuple(row_candidates),
        tuple(column_candidates),
        row_count,
        target_groups,
        identity_sha256,
        f"control:{world}:{split}:numeric-swap",
        preset,
    )
    selected_row_hashes = {
        group.normalized_sha256 for group in selected[:row_count]
    }
    if len(selected_row_hashes) != row_count:
        raise PartitionGateError("control row arm contains duplicate groups")
    if len({group.normalized_sha256 for group in selected}) != len(selected):
        raise PartitionGateError("matched control reused a group across its arms")
    diagnostics = matched_control_diagnostics(target_groups, selected)
    _require_control_bounds(diagnostics, preset)
    selection = ControlSelection(
        world=world,
        split=split,
        group_sha256=tuple(sorted(group.normalized_sha256 for group in selected)),
        row_group_count=row_count,
        column_group_count=column_count,
        active_token_count=sum(group.active_token_count for group in selected),
    )
    return selection, diagnostics


def matched_control_diagnostics(
    target_groups: Sequence[AllocationGroup],
    control_groups: Sequence[AllocationGroup],
) -> ControlDiagnostics:
    """Measure all predeclared matched-control bounds."""
    if not target_groups or not control_groups:
        raise ValueError("control diagnostics require nonempty target and control groups")
    target_tokens = sum(group.active_token_count for group in target_groups)
    control_tokens = sum(group.active_token_count for group in control_groups)
    target_mean = sum(group.canonical_token_count for group in target_groups) / len(
        target_groups
    )
    control_mean = sum(group.canonical_token_count for group in control_groups) / len(
        control_groups
    )
    source_feature_error = max(
        _prevalence_errors(
            target_groups,
            control_groups,
            lambda group: (
                ("source", group.source),
                ("feature", group.feature_signature),
            ),
        ),
        default=0.0,
    )
    adjective_length_error = max(
        _prevalence_errors(
            target_groups,
            control_groups,
            lambda group: (
                ("adjective", str(group.adjective_bucket)),
                ("length", group.length_bin),
            ),
        ),
        default=0.0,
    )
    return ControlDiagnostics(
        token_relative_error=abs(control_tokens - target_tokens) / target_tokens,
        maximum_source_feature_prevalence_error=source_feature_error,
        maximum_adjective_length_prevalence_error=adjective_length_error,
        mean_length_relative_error=abs(control_mean - target_mean) / target_mean,
    )


def _cell_imbalance_score(
    cells: tuple[CellStatistics, ...],
) -> tuple[Fraction, Fraction, Fraction]:
    total_tokens = sum(cell.active_token_count for cell in cells)
    token_imbalance = Fraction(
        sum(abs(len(cells) * cell.active_token_count - total_tokens) for cell in cells),
        total_tokens,
    )
    nuisance_maps = tuple(
        {(dimension, category): count for dimension, category, count in cell.nuisance_token_counts}
        for cell in cells
    )
    categories = sorted(set().union(*(set(counts) for counts in nuisance_maps)))
    nuisance_imbalance = Fraction(
        sum(
            abs(
                len(cells) * counts.get(category, 0)
                - sum(other.get(category, 0) for other in nuisance_maps)
            )
            for category in categories
            for counts in nuisance_maps
        ),
        total_tokens,
    )
    return token_imbalance + nuisance_imbalance, token_imbalance, nuisance_imbalance


def _allocator_score(
    label: SplitLabel,
    weight: int,
    total_weight: int,
    total_tokens: int,
    marginal_totals: Mapping[tuple[MarginalName, str], int],
    split_tokens: Mapping[SplitLabel, int],
    split_marginals: Mapping[tuple[SplitLabel, MarginalName, str], int],
    group: AllocationGroup,
) -> Fraction:
    target_tokens = Fraction(total_tokens * weight, total_weight)
    projected_tokens = split_tokens.get(label, 0) + group.active_token_count
    overall_fill = Fraction(projected_tokens, 1) / target_tokens
    marginal_fill = sum(
        (
            Fraction(
                split_marginals.get((label, dimension, category), 0)
                + group.active_token_count,
                1,
            )
            / Fraction(marginal_totals[(dimension, category)] * weight, total_weight)
        )
        ** 2
        for dimension, category in group.marginals
    )
    return overall_fill**2 + marginal_fill


def _select_control_arm(
    candidates: Sequence[AllocationGroup],
    target_strata: Counter[tuple[str, str, str, str]],
    arm_count: int,
    target_count: int,
    identity_sha256: str,
    namespace: str,
) -> list[AllocationGroup]:
    if arm_count == 0:
        return []
    unique_candidates = {group.normalized_sha256: group for group in candidates}
    if len(unique_candidates) != len(candidates):
        raise ValueError("control candidate hashes must be unique")
    if len(candidates) < arm_count:
        raise PartitionGateError(
            f"{namespace} has {len(candidates)} candidates for {arm_count} controls"
        )
    exact_quotas = {
        stratum: Fraction(count * arm_count, target_count)
        for stratum, count in target_strata.items()
    }
    quotas = {stratum: int(quota) for stratum, quota in exact_quotas.items()}
    remaining = arm_count - sum(quotas.values())
    for stratum in sorted(
        exact_quotas,
        key=lambda value: (
            -(exact_quotas[value] - quotas[value]),
            _namespace_hash(identity_sha256, 0, f"{namespace}:quota", repr(value)),
            value,
        ),
    )[:remaining]:
        quotas[stratum] += 1
    by_stratum: dict[tuple[str, str, str, str], list[AllocationGroup]] = defaultdict(list)
    for group in candidates:
        by_stratum[group.full_stratum].append(group)
    target_mean = sum(
        count * _length_bin_midpoint(stratum[3])
        for stratum, count in target_strata.items()
    ) / target_count
    for stratum in by_stratum:
        by_stratum[stratum].sort(
            key=lambda group: (
                abs(group.canonical_token_count - target_mean),
                _namespace_hash(
                    identity_sha256,
                    0,
                    f"{namespace}:candidate",
                    group.normalized_sha256,
                ),
                group.normalized_sha256,
            )
        )
    selected = [
        group
        for stratum in sorted(quotas)
        for group in by_stratum.get(stratum, ())[: quotas[stratum]]
    ]
    selected_hashes = {group.normalized_sha256 for group in selected}
    if len(selected) < arm_count:
        remaining_candidates = sorted(
            (
                group
                for group in candidates
                if group.normalized_sha256 not in selected_hashes
            ),
            key=lambda group: (
                _stratum_distance(group.full_stratum, target_strata),
                abs(group.canonical_token_count - target_mean),
                _namespace_hash(
                    identity_sha256,
                    0,
                    f"{namespace}:fill",
                    group.normalized_sha256,
                ),
                group.normalized_sha256,
            ),
        )
        selected.extend(remaining_candidates[: arm_count - len(selected)])
    if len(selected) != arm_count:
        raise PartitionGateError(f"{namespace} could not fill its control arm")
    return selected


def _improve_control_marginal_match(
    initial: tuple[AllocationGroup, ...],
    row_candidates: tuple[AllocationGroup, ...],
    column_candidates: tuple[AllocationGroup, ...],
    row_count: int,
    target_groups: Sequence[AllocationGroup],
    identity_sha256: str,
    namespace: str,
    preset: PartitionPreset,
) -> tuple[AllocationGroup, ...]:
    """Satisfy the declared marginal bounds without requiring joint matches."""
    selected = list(initial)
    target_count = len(target_groups)
    if len(selected) != target_count:
        raise ValueError("control and target group counts must agree")

    category_cache: dict[
        str,
        tuple[tuple[str, str], tuple[str, str], tuple[str, str], tuple[str, str]],
    ] = {}

    def categories(
        group: AllocationGroup,
    ) -> tuple[tuple[str, str], tuple[str, str], tuple[str, str], tuple[str, str]]:
        cached = category_cache.get(group.normalized_sha256)
        if cached is not None:
            return cached
        value = (
            ("source", group.source),
            ("feature", group.feature_signature),
            ("adjective", str(group.adjective_bucket)),
            ("length", group.length_bin),
        )
        category_cache[group.normalized_sha256] = value
        return value

    target_counts = Counter(
        category for group in target_groups for category in categories(group)
    )
    selected_counts = Counter(
        category for group in selected for category in categories(group)
    )
    all_categories = set(target_counts) | set(selected_counts)
    differences = Counter(
        {
            category: selected_counts[category] - target_counts[category]
            for category in all_categories
        }
    )
    source_feature_tolerance = Fraction(
        str(preset.control_source_feature_tolerance)
    )
    adjective_length_tolerance = Fraction(
        str(preset.control_adjective_length_tolerance)
    )

    source_feature_threshold = (
        target_count * source_feature_tolerance.numerator
    )
    adjective_length_threshold = (
        target_count * adjective_length_tolerance.numerator
    )
    common_denominator = math.lcm(
        source_feature_threshold**2,
        adjective_length_threshold**2,
    )
    score_parameters = {
        "source": (
            source_feature_tolerance.denominator,
            source_feature_threshold,
            common_denominator // source_feature_threshold**2,
        ),
        "feature": (
            source_feature_tolerance.denominator,
            source_feature_threshold,
            common_denominator // source_feature_threshold**2,
        ),
        "adjective": (
            adjective_length_tolerance.denominator,
            adjective_length_threshold,
            common_denominator // adjective_length_threshold**2,
        ),
        "length": (
            adjective_length_tolerance.denominator,
            adjective_length_threshold,
            common_denominator // adjective_length_threshold**2,
        ),
    }

    def category_scores(
        category: tuple[str, str],
        difference: int,
    ) -> tuple[int, int]:
        scale, threshold, weight = score_parameters[category[0]]
        scaled_difference = abs(difference) * scale
        excess = max(scaled_difference - threshold, 0)
        return excess**2 * weight, scaled_difference**2 * weight

    violation_score = sum(
        category_scores(category, difference)[0]
        for category, difference in differences.items()
    )
    balance_score = sum(
        category_scores(category, difference)[1]
        for category, difference in differences.items()
    )
    pools = (row_candidates, column_candidates)
    spans = ((0, row_count), (row_count, len(selected)))
    pools_by_signature: list[
        dict[tuple[tuple[str, str], ...], list[AllocationGroup]]
    ] = []
    for arm_index, pool in enumerate(pools):
        by_signature: dict[
            tuple[tuple[str, str], ...], list[AllocationGroup]
        ] = defaultdict(list)
        for group in pool:
            by_signature[categories(group)].append(group)
        for values in by_signature.values():
            values.sort(
                key=lambda group: (
                    _namespace_hash(
                        identity_sha256,
                        0,
                        f"{namespace}:pool:{arm_index}",
                        group.normalized_sha256,
                    ),
                    group.normalized_sha256,
                )
            )
        pools_by_signature.append(by_signature)
    signature_changes: dict[
        tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]],
        tuple[tuple[tuple[str, str], int], ...],
    ] = {}
    selected_order_hashes: dict[tuple[int, str], str] = {}

    def selected_order_hash(arm_index: int, group: AllocationGroup) -> str:
        key = arm_index, group.normalized_sha256
        cached = selected_order_hashes.get(key)
        if cached is None:
            cached = _namespace_hash(
                identity_sha256,
                0,
                f"{namespace}:selected:{arm_index}",
                group.normalized_sha256,
            )
            selected_order_hashes[key] = cached
        return cached

    for iteration in range(min(512, max(1, len(selected)))):
        if violation_score == 0:
            break
        selected_hashes = {group.normalized_sha256 for group in selected}
        violating_directions = {
            category: 1 if difference > 0 else -1
            for category, difference in differences.items()
            if category_scores(category, difference)[0] > 0
        }
        best_swap: tuple[
            int,
            int,
            str,
            int,
            AllocationGroup,
            tuple[tuple[tuple[str, str], int], ...],
        ] | None = None
        for arm_index, (start, stop) in enumerate(spans):
            outgoing_by_signature: dict[
                tuple[tuple[str, str], ...], tuple[int, AllocationGroup]
            ] = {}
            for index in range(start, stop):
                group = selected[index]
                signature = categories(group)
                current = outgoing_by_signature.get(signature)
                candidate_key = (
                    selected_order_hash(arm_index, group),
                    group.normalized_sha256,
                )
                if current is None or candidate_key < (
                    selected_order_hash(arm_index, current[1]),
                    current[1].normalized_sha256,
                ):
                    outgoing_by_signature[signature] = (index, group)
            incoming_by_signature = {
                signature: incoming
                for signature, values in pools_by_signature[arm_index].items()
                for incoming in (
                    next(
                        (
                            group
                            for group in values
                            if group.normalized_sha256 not in selected_hashes
                        ),
                        None,
                    ),
                )
                if incoming is not None
            }
            for outgoing_signature, (
                outgoing_index,
                outgoing,
            ) in outgoing_by_signature.items():
                for incoming_signature, incoming in incoming_by_signature.items():
                    if outgoing_signature == incoming_signature:
                        continue
                    signature_pair = (outgoing_signature, incoming_signature)
                    delta = signature_changes.get(signature_pair)
                    if delta is None:
                        changes: Counter[tuple[str, str]] = Counter(incoming_signature)
                        changes.subtract(outgoing_signature)
                        delta = tuple(
                            sorted(
                                (category, change)
                                for category, change in changes.items()
                                if change
                            )
                        )
                        signature_changes[signature_pair] = delta
                    if not any(
                        (direction > 0 and change < 0)
                        or (direction < 0 and change > 0)
                        for category, change in delta
                        for direction in (violating_directions.get(category),)
                        if direction is not None
                    ):
                        continue
                    next_violation = violation_score
                    next_balance = balance_score
                    for category, change in delta:
                        previous = category_scores(category, differences[category])
                        following = category_scores(
                            category,
                            differences[category] + change,
                        )
                        next_violation += following[0] - previous[0]
                        next_balance += following[1] - previous[1]
                    if best_swap is not None and (
                        next_violation,
                        next_balance,
                    ) > best_swap[:2]:
                        continue
                    candidate = (
                        next_violation,
                        next_balance,
                        _namespace_hash(
                            identity_sha256,
                            0,
                            f"{namespace}:choice:{iteration}",
                            f"{outgoing.normalized_sha256}\0{incoming.normalized_sha256}",
                        ),
                        outgoing_index,
                        incoming,
                        delta,
                    )
                    if best_swap is None or candidate < best_swap:
                        best_swap = candidate
        if best_swap is None or best_swap[:2] >= (
            violation_score,
            balance_score,
        ):
            break
        (
            violation_score,
            balance_score,
            _,
            outgoing_index,
            incoming,
            delta,
        ) = best_swap
        selected[outgoing_index] = incoming
        for category, change in delta:
            differences[category] += change
    return tuple(selected)


def _improve_control_token_match(
    initial: tuple[AllocationGroup, ...],
    row_candidates: tuple[AllocationGroup, ...],
    column_candidates: tuple[AllocationGroup, ...],
    row_count: int,
    target_tokens: int,
    identity_sha256: str,
    namespace: str,
) -> tuple[AllocationGroup, ...]:
    selected = list(initial)
    pools = (row_candidates, column_candidates)
    spans = ((0, row_count), (row_count, len(selected)))
    for arm_index, (start, stop) in enumerate(spans):
        selected_hashes = {group.normalized_sha256 for group in selected}
        alternatives = [
            group
            for group in pools[arm_index]
            if group.normalized_sha256 not in selected_hashes
        ]
        for _ in range(min(128, max(1, stop - start))):
            current_tokens = sum(group.active_token_count for group in selected)
            current_error = abs(current_tokens - target_tokens)
            desired_delta = target_tokens - current_tokens
            alternatives_by_stratum: dict[
                tuple[str, str, str, str], list[AllocationGroup]
            ] = defaultdict(list)
            for group in alternatives:
                alternatives_by_stratum[group.full_stratum].append(group)
            for values in alternatives_by_stratum.values():
                values.sort(
                    key=lambda group: (
                        group.active_token_count,
                        group.normalized_sha256,
                    )
                )
            alternative_token_counts = {
                stratum: [group.active_token_count for group in values]
                for stratum, values in alternatives_by_stratum.items()
            }
            best_swap: tuple[int, str, int, AllocationGroup] | None = None
            for index, outgoing in enumerate(selected[start:stop], start=start):
                candidates = alternatives_by_stratum.get(outgoing.full_stratum, [])
                if not candidates:
                    continue
                token_counts = alternative_token_counts[outgoing.full_stratum]
                desired_incoming = outgoing.active_token_count + desired_delta
                insertion = bisect_left(token_counts, desired_incoming)
                for candidate_index in {max(0, insertion - 1), min(len(candidates) - 1, insertion)}:
                    incoming = candidates[candidate_index]
                    error = abs(
                        current_tokens
                        - outgoing.active_token_count
                        + incoming.active_token_count
                        - target_tokens
                    )
                    if best_swap is not None and error > best_swap[0]:
                        continue
                    swap = (
                        error,
                        _namespace_hash(
                            identity_sha256,
                            0,
                            namespace,
                            f"{outgoing.normalized_sha256}\0{incoming.normalized_sha256}",
                        ),
                        index,
                        incoming,
                    )
                    if best_swap is None or swap < best_swap:
                        best_swap = swap
            if best_swap is None or best_swap[0] >= current_error:
                break
            _, _, index, incoming = best_swap
            outgoing = selected[index]
            selected[index] = incoming
            alternatives = [
                group
                for group in (*alternatives, outgoing)
                if group.normalized_sha256 != incoming.normalized_sha256
            ]
    return tuple(selected)


def _improve_control_numeric_match(
    initial: tuple[AllocationGroup, ...],
    row_candidates: tuple[AllocationGroup, ...],
    column_candidates: tuple[AllocationGroup, ...],
    row_count: int,
    target_groups: Sequence[AllocationGroup],
    identity_sha256: str,
    namespace: str,
    preset: PartitionPreset,
) -> tuple[AllocationGroup, ...]:
    """Use same-stratum swaps to jointly satisfy token and mean-length bounds."""
    selected = list(initial)
    target_tokens = sum(group.active_token_count for group in target_groups)
    target_canonical_tokens = sum(
        group.canonical_token_count for group in target_groups
    )
    selected_tokens = sum(group.active_token_count for group in selected)
    selected_canonical_tokens = sum(
        group.canonical_token_count for group in selected
    )
    pools = (row_candidates, column_candidates)
    spans = ((0, row_count), (row_count, len(selected)))
    token_tolerance = Fraction(str(preset.control_token_tolerance))
    mean_tolerance = Fraction(str(preset.control_mean_length_tolerance))

    def score(active_tokens: int, canonical_tokens: int) -> tuple[Fraction, Fraction]:
        token_ratio = Fraction(
            abs(active_tokens - target_tokens),
            target_tokens,
        ) / token_tolerance
        mean_ratio = Fraction(
            abs(canonical_tokens - target_canonical_tokens),
            target_canonical_tokens,
        ) / mean_tolerance
        return max(token_ratio, mean_ratio), token_ratio + mean_ratio

    for _ in range(min(256, max(1, len(selected)))):
        current_score = score(selected_tokens, selected_canonical_tokens)
        if current_score[0] <= 1:
            break
        selected_hashes = {group.normalized_sha256 for group in selected}
        best_swap: tuple[
            tuple[Fraction, Fraction],
            str,
            int,
            AllocationGroup,
            int,
            int,
        ] | None = None
        for arm_index, (start, stop) in enumerate(spans):
            alternatives_by_stratum: dict[
                tuple[str, str, str, str], list[AllocationGroup]
            ] = defaultdict(list)
            for group in pools[arm_index]:
                if group.normalized_sha256 not in selected_hashes:
                    alternatives_by_stratum[group.full_stratum].append(group)
            for index in range(start, stop):
                outgoing = selected[index]
                for incoming in alternatives_by_stratum.get(
                    outgoing.full_stratum,
                    (),
                ):
                    next_tokens = (
                        selected_tokens
                        - outgoing.active_token_count
                        + incoming.active_token_count
                    )
                    next_canonical_tokens = (
                        selected_canonical_tokens
                        - outgoing.canonical_token_count
                        + incoming.canonical_token_count
                    )
                    next_score = score(next_tokens, next_canonical_tokens)
                    if best_swap is not None and next_score > best_swap[0]:
                        continue
                    candidate = (
                        next_score,
                        _namespace_hash(
                            identity_sha256,
                            0,
                            namespace,
                            f"{outgoing.normalized_sha256}\0{incoming.normalized_sha256}",
                        ),
                        index,
                        incoming,
                        next_tokens,
                        next_canonical_tokens,
                    )
                    if best_swap is None or candidate < best_swap:
                        best_swap = candidate
        if best_swap is None or best_swap[0] >= current_score:
            break
        (
            _,
            _,
            index,
            incoming,
            selected_tokens,
            selected_canonical_tokens,
        ) = best_swap
        selected[index] = incoming
    return tuple(selected)


def _prevalence_errors(
    target_groups: Sequence[AllocationGroup],
    control_groups: Sequence[AllocationGroup],
    categories,
) -> tuple[float, ...]:
    target_counts = Counter(category for group in target_groups for category in categories(group))
    control_counts = Counter(category for group in control_groups for category in categories(group))
    all_categories = set(target_counts) | set(control_counts)
    return tuple(
        abs(
            target_counts[category] / len(target_groups)
            - control_counts[category] / len(control_groups)
        )
        for category in all_categories
    )


def _require_control_bounds(
    diagnostics: ControlDiagnostics,
    preset: PartitionPreset,
) -> None:
    failures = tuple(
        label
        for label, measured, bound in (
            (
                "active token count",
                diagnostics.token_relative_error,
                preset.control_token_tolerance,
            ),
            (
                "source/feature prevalence",
                diagnostics.maximum_source_feature_prevalence_error,
                preset.control_source_feature_tolerance,
            ),
            (
                "adjective/length prevalence",
                diagnostics.maximum_adjective_length_prevalence_error,
                preset.control_adjective_length_tolerance,
            ),
            (
                "mean length",
                diagnostics.mean_length_relative_error,
                preset.control_mean_length_tolerance,
            ),
        )
        if measured > bound + 1e-15
    )
    if failures:
        raise PartitionGateError(
            "matched-control bounds failed for "
            + ", ".join(failures)
            + f": {diagnostics!r}"
        )


def _stratum_distance(
    stratum: tuple[str, str, str, str],
    target: Mapping[tuple[str, str, str, str], int],
) -> int:
    return min(
        sum(left != right for left, right in zip(stratum, target_stratum, strict=True))
        for target_stratum in target
    )


def _length_bin_midpoint(length_bin: str) -> float:
    return {"le64": 48.0, "65-128": 96.0, "129-192": 160.0, "gt192": 224.0}[
        length_bin
    ]


def _namespace_hash(
    identity_sha256: str,
    public_seed: int,
    namespace: str,
    value: str,
) -> str:
    return sha256(
        (
            BENCHMARK_ID
            + "\0"
            + identity_sha256
            + "\0"
            + str(public_seed)
            + "\0"
            + namespace
            + "\0"
            + value
        ).encode("utf-8")
    ).hexdigest()


def _require_identity(value: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("identity_sha256 must be lowercase hexadecimal")


__all__ = [
    "AllocationGroup",
    "CellStatistics",
    "ControlDiagnostics",
    "PartitionGateError",
    "allocate_stratified_splits",
    "balance_word_buckets",
    "bucket_word_lookup",
    "matched_control_diagnostics",
    "require_component_visibility",
    "select_matched_control",
    "select_world_cells",
    "summarize_cells",
    "token_length_bin",
]

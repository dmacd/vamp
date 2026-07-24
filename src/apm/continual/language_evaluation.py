"""Evaluation-only language-suite contracts and cue coverage summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import re

import numpy as np

from apm.continual.language_tasks import LanguageEvaluationExample, TaskId


CueRegime = Literal[
    "cue_sufficient",
    "cue_present",
    "cue_hidden_or_ambiguous",
]
EvaluationSplit = Literal["validation", "test"]
IN_DOMAIN_TOPIC_SPECIALIZATION = "in-domain topic specialization"
_CUE_REGIMES: tuple[CueRegime, ...] = (
    "cue_sufficient",
    "cue_present",
    "cue_hidden_or_ambiguous",
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class LanguageEvaluationCondition:
    """One explicit address-prefix and competence-suffix condition."""

    condition_id: str
    prefix_tokens: int
    suffix_tokens: int

    def __post_init__(self) -> None:
        if not self.condition_id:
            raise ValueError("evaluation condition_id must not be empty")
        if type(self.prefix_tokens) is not int or self.prefix_tokens < 2:
            raise ValueError("evaluation prefixes must contain at least two tokens")
        if type(self.suffix_tokens) is not int or self.suffix_tokens < 1:
            raise ValueError("evaluation suffixes must contain at least one token")


@dataclass(frozen=True)
class LanguageExampleProvenance:
    """Stable span-level identity retained independently of rendered batches."""

    source_document_id: str
    token_offset: int
    pair_hash: str

    def __post_init__(self) -> None:
        if not self.source_document_id:
            raise ValueError("source_document_id must not be empty")
        if type(self.token_offset) is not int or self.token_offset < 0:
            raise ValueError("token_offset must be a nonnegative integer")
        if _SHA256_PATTERN.fullmatch(self.pair_hash) is None:
            raise ValueError("pair_hash must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class LanguageSuiteExample:
    """One condition-specific view of a paired evaluation anchor."""

    pair_id: str
    condition_id: str
    split: EvaluationSplit
    example: LanguageEvaluationExample
    provenance: LanguageExampleProvenance
    cue_regime: CueRegime
    visible_concept_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.pair_id or not self.condition_id:
            raise ValueError("paired example IDs must not be empty")
        if self.split not in ("validation", "test"):
            raise ValueError("language evaluation split must be validation or test")
        if not isinstance(self.example, LanguageEvaluationExample):
            raise TypeError("example must be a LanguageEvaluationExample")
        if not isinstance(self.provenance, LanguageExampleProvenance):
            raise TypeError("provenance must be LanguageExampleProvenance")
        if self.cue_regime not in _CUE_REGIMES:
            raise ValueError(f"unknown cue regime: {self.cue_regime}")
        if (
            not isinstance(self.visible_concept_ids, tuple)
            or tuple(sorted(set(self.visible_concept_ids)))
            != self.visible_concept_ids
            or any(not value for value in self.visible_concept_ids)
        ):
            raise ValueError("visible concept IDs must be sorted unique strings")
        if self.cue_regime != "cue_hidden_or_ambiguous" and not self.visible_concept_ids:
            raise ValueError("a visible semantic cue must name at least one concept")

    @property
    def task_id(self) -> TaskId:
        """Return evaluator task metadata without exposing it to a router."""
        return self.example.task_id


@dataclass(frozen=True)
class LanguageEvaluationSuite:
    """An evaluation-only collection with paired conditions and provenance."""

    suite_id: str
    benchmark_label: str
    primary_condition_id: str
    conditions: tuple[LanguageEvaluationCondition, ...]
    examples: tuple[LanguageSuiteExample, ...]

    def __post_init__(self) -> None:
        if not self.suite_id:
            raise ValueError("language evaluation suite_id must not be empty")
        if not self.benchmark_label:
            raise ValueError("language evaluation benchmark_label must not be empty")
        if not self.conditions or any(
            not isinstance(condition, LanguageEvaluationCondition)
            for condition in self.conditions
        ):
            raise ValueError("evaluation suite must contain explicit conditions")
        condition_ids = tuple(condition.condition_id for condition in self.conditions)
        if len(set(condition_ids)) != len(condition_ids):
            raise ValueError("evaluation condition IDs must be unique")
        if self.primary_condition_id not in condition_ids:
            raise ValueError("primary_condition_id must name one suite condition")
        if not self.examples or any(
            not isinstance(example, LanguageSuiteExample)
            for example in self.examples
        ):
            raise ValueError("evaluation suite must contain paired examples")
        if any(example.condition_id not in condition_ids for example in self.examples):
            raise ValueError("suite examples must reference declared conditions")
        _validate_paired_examples(self.conditions, self.examples)

    def examples_for(
        self,
        task_id: TaskId,
        condition_id: str,
        split: EvaluationSplit = "test",
    ) -> tuple[LanguageSuiteExample, ...]:
        """Return one task/condition slice in canonical pair order."""
        if condition_id not in tuple(value.condition_id for value in self.conditions):
            raise KeyError(f"unknown evaluation condition: {condition_id}")
        if split not in ("validation", "test"):
            raise ValueError("language evaluation split must be validation or test")
        return tuple(
            value
            for value in self.examples
            if value.task_id == task_id
            and value.condition_id == condition_id
            and value.split == split
        )


@dataclass(frozen=True)
class LanguageCueCoverage:
    """One nonempty cue stratum or its derived all-example aggregate."""

    task_id: TaskId
    condition_id: str
    split: EvaluationSplit
    cue_regime: CueRegime | Literal["all"]
    example_count: int
    pair_count: int
    source_story_count: int

    def __post_init__(self) -> None:
        if not self.task_id or not self.condition_id:
            raise ValueError("cue coverage identity must not be empty")
        if self.split not in ("validation", "test"):
            raise ValueError("cue coverage split must be validation or test")
        if self.cue_regime not in _CUE_REGIMES + ("all",):
            raise ValueError(f"unknown cue coverage regime: {self.cue_regime}")
        counts = (self.example_count, self.pair_count, self.source_story_count)
        if any(type(value) is not int or value <= 0 for value in counts):
            raise ValueError("cue coverage counts must be positive integers")
        if self.pair_count > self.example_count:
            raise ValueError("cue coverage pair_count cannot exceed example_count")


def summarize_language_cue_coverage(
    suite: LanguageEvaluationSuite,
) -> tuple[LanguageCueCoverage, ...]:
    """Return every nonempty visible-prefix stratum plus derived aggregates."""
    if not isinstance(suite, LanguageEvaluationSuite):
        raise TypeError("suite must be a LanguageEvaluationSuite")
    identities = tuple(
        dict.fromkeys(
            (example.task_id, example.condition_id, example.split)
            for example in suite.examples
        )
    )
    rows: list[LanguageCueCoverage] = []
    for task_id, condition_id, split in identities:
        group = tuple(
            example
            for example in suite.examples
            if (example.task_id, example.condition_id, example.split)
            == (task_id, condition_id, split)
        )
        for cue_regime in _CUE_REGIMES + ("all",):
            stratum = (
                group
                if cue_regime == "all"
                else tuple(
                    example
                    for example in group
                    if example.cue_regime == cue_regime
                )
            )
            if stratum:
                rows.append(
                    LanguageCueCoverage(
                        task_id=task_id,
                        condition_id=condition_id,
                        split=split,
                        cue_regime=cue_regime,
                        example_count=len(stratum),
                        pair_count=len({example.pair_id for example in stratum}),
                        source_story_count=len(
                            {
                                example.provenance.source_document_id
                                for example in stratum
                            }
                        ),
                    )
                )
    return tuple(rows)


def _validate_paired_examples(
    conditions: tuple[LanguageEvaluationCondition, ...],
    examples: tuple[LanguageSuiteExample, ...],
) -> None:
    condition_by_id = {condition.condition_id: condition for condition in conditions}
    pair_keys = tuple(
        dict.fromkeys(
            (example.task_id, example.split, example.pair_id)
            for example in examples
        )
    )
    expected_condition_ids = tuple(condition_by_id)
    for pair_key in pair_keys:
        paired = tuple(
            example
            for example in examples
            if (example.task_id, example.split, example.pair_id) == pair_key
        )
        if tuple(example.condition_id for example in paired) != expected_condition_ids:
            raise ValueError("every pair must contain all conditions in canonical order")
        if len({example.provenance for example in paired}) != 1:
            raise ValueError("paired conditions must share identical provenance")
        prefixes = tuple(
            _validated_prefix_tokens(example, condition_by_id[example.condition_id])
            for example in paired
        )
        ordered_prefixes = tuple(
            prefix
            for _, prefix in sorted(
                zip(
                    (condition.prefix_tokens for condition in conditions),
                    prefixes,
                )
            )
        )
        if any(
            not np.array_equal(shorter, longer[: shorter.size])
            for shorter, longer in zip(ordered_prefixes, ordered_prefixes[1:])
        ):
            raise ValueError("paired condition prefixes must be exactly nested")


def _validated_prefix_tokens(
    suite_example: LanguageSuiteExample,
    condition: LanguageEvaluationCondition,
) -> np.ndarray:
    example = suite_example.example
    router = example.router_batch
    competence = example.competence_batch
    if router.input_ids.shape != (1, condition.prefix_tokens - 1):
        raise ValueError("router width does not match its evaluation condition")
    if competence.input_ids.shape != (
        1,
        condition.prefix_tokens + condition.suffix_tokens - 1,
    ):
        raise ValueError("competence width does not match its evaluation condition")
    if int(np.sum(competence.loss_mask)) != condition.suffix_tokens:
        raise ValueError("evaluation conditions require their full suffix token count")
    return np.concatenate(
        (
            router.input_ids[0],
            router.target_ids[0, -1:],
        )
    )


__all__ = [
    "CueRegime",
    "EvaluationSplit",
    "IN_DOMAIN_TOPIC_SPECIALIZATION",
    "LanguageCueCoverage",
    "LanguageEvaluationCondition",
    "LanguageEvaluationSuite",
    "LanguageExampleProvenance",
    "LanguageSuiteExample",
    "summarize_language_cue_coverage",
]

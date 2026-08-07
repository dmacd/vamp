"""Versioned nouns-v2 wrappers around the bounded shared evaluators."""

from __future__ import annotations

from pathlib import Path

from apm.continual.language_adaptation_artifact import LanguageAdaptationArtifact
from apm.data.text.tinyworlds_nouns_v1.evaluation import (
    CompletionResult,
    HalfStoryGenerationRow,
    PrefixOnlyQuery,
    build_prefix_only_query,
    evaluate_half_story_generations as _evaluate_half_story_generations,
    evaluate_whole_story_nll as _evaluate_whole_story_nll,
)
from apm.data.text.tinyworlds_nouns_v1.experiment import NounSelectedBase
from apm.data.text.tinyworlds_nouns_v2.contracts import (
    NounsV2ExperimentPreset,
    NounsV2PartitionArtifact,
)
from apm.lm.text import TextTokenizer


def evaluate_whole_story_nll(
    partition: NounsV2PartitionArtifact,
    preset: NounsV2ExperimentPreset,
    selected_base: NounSelectedBase,
    adaptation: LanguageAdaptationArtifact,
    output_path: str | Path,
    *,
    progress=None,
) -> Path:
    """Stream exactly 4,440 × 6 authenticated v2 whole-story rows."""
    return _evaluate_whole_story_nll(
        partition,
        preset,
        selected_base,
        adaptation,
        output_path,
        progress=progress,
    )


def evaluate_half_story_generations(
    partition: NounsV2PartitionArtifact,
    preset: NounsV2ExperimentPreset,
    selected_base: NounSelectedBase,
    adaptation: LanguageAdaptationArtifact,
    tokenizer: TextTokenizer,
    output_path: str | Path,
    *,
    progress=None,
) -> Path:
    """Stream midpoint-only routing, suffix NLL, and equal-budget generations."""
    return _evaluate_half_story_generations(
        partition,
        preset,
        selected_base,
        adaptation,
        tokenizer,
        output_path,
        progress=progress,
    )


__all__ = [
    "CompletionResult",
    "HalfStoryGenerationRow",
    "PrefixOnlyQuery",
    "build_prefix_only_query",
    "evaluate_half_story_generations",
    "evaluate_whole_story_nll",
]

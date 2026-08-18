"""Versioned nouns-v2 entry points for the shared resumable noun engine."""

from __future__ import annotations

from pathlib import Path

from apm.continual.language_adaptation_artifact import LanguageAdaptationArtifact
from apm.data.text.tinyworlds_nouns_v1.experiment import (
    IndexedStoryStore,
    IndexedTokenBatchSequence,
    NounGpuPreflight,
    NounSelectedBase,
    StoryIndexEntry,
    load_noun_gpu_preflight,
    load_noun_selected_base,
    load_noun_vamp_stages,
    load_story_index,
    noun_model_config,
    router_batch_from_index,
    run_or_load_noun_gpu_preflight,
    run_or_resume_noun_base,
    run_or_resume_noun_vamp,
)
from apm.data.text.tinyworlds_nouns_v2.contracts import (
    CHECKPOINT_ROOT,
    NounsV2ExperimentPreset,
    NounsV2PartitionArtifact,
)
from apm.data.text.tinyworlds_nouns_v2.baselines import (
    load_nouns_v2_baseline_stages,
    run_or_resume_nouns_v2_baselines,
)
from apm.data.text.tinyworlds_nouns_v2.full_finetune import (
    FullFinetuneStage,
    load_nouns_v2_full_finetune_stages,
    run_or_resume_nouns_v2_full_finetune,
)


def run_or_load_nouns_v2_gpu_preflight(
    partition: NounsV2PartitionArtifact,
    preset: NounsV2ExperimentPreset,
    checkpoint_root: str | Path = CHECKPOINT_ROOT,
) -> NounGpuPreflight:
    """Run or strict-load the GPU preflight under v2 formats and paths."""
    return run_or_load_noun_gpu_preflight(partition, preset, checkpoint_root)


def load_nouns_v2_gpu_preflight(
    partition: NounsV2PartitionArtifact,
    preset: NounsV2ExperimentPreset,
    path: str | Path,
) -> NounGpuPreflight:
    """Strict-load a published v2 GPU preflight without measuring or writing."""
    return load_noun_gpu_preflight(partition, preset, path)


def load_nouns_v2_selected_base(
    partition: NounsV2PartitionArtifact,
    preset: NounsV2ExperimentPreset,
    preflight: NounGpuPreflight,
    root: str | Path,
) -> NounSelectedBase:
    """Strict-load a published v2 selected base without training or resuming."""
    return load_noun_selected_base(root, partition, preset, preflight)


def run_or_resume_nouns_v2_base(
    partition: NounsV2PartitionArtifact,
    preset: NounsV2ExperimentPreset,
    preflight: NounGpuPreflight,
    checkpoint_root: str | Path = CHECKPOINT_ROOT,
    *,
    progress=None,
) -> NounSelectedBase:
    """Train or exact-resume the fresh seed-zero nouns-v2 base."""
    return run_or_resume_noun_base(
        partition,
        preset,
        preflight,
        checkpoint_root,
        progress=progress,
    )


def run_or_resume_nouns_v2_vamp(
    partition: NounsV2PartitionArtifact,
    preset: NounsV2ExperimentPreset,
    selected_base: NounSelectedBase,
    checkpoint_root: str | Path = CHECKPOINT_ROOT,
    *,
    progress=None,
) -> LanguageAdaptationArtifact:
    """Train or stage-resume the ordered 24-edge nouns-v2 VAMP graph."""
    return run_or_resume_noun_vamp(
        partition,
        preset,
        selected_base,
        checkpoint_root,
        progress=progress,
    )


def load_nouns_v2_vamp_stages(
    partition: NounsV2PartitionArtifact,
    preset: NounsV2ExperimentPreset,
    selected_base: NounSelectedBase,
    checkpoint_root: str | Path = CHECKPOINT_ROOT,
) -> tuple[LanguageAdaptationArtifact, ...]:
    """Strict-load all 24 immutable VAMP stages for longitudinal evaluation."""
    return load_noun_vamp_stages(
        partition,
        preset,
        selected_base,
        checkpoint_root,
    )


__all__ = [
    "IndexedStoryStore",
    "IndexedTokenBatchSequence",
    "NounGpuPreflight",
    "NounSelectedBase",
    "StoryIndexEntry",
    "FullFinetuneStage",
    "load_nouns_v2_gpu_preflight",
    "load_nouns_v2_selected_base",
    "load_story_index",
    "load_nouns_v2_vamp_stages",
    "load_nouns_v2_baseline_stages",
    "load_nouns_v2_full_finetune_stages",
    "noun_model_config",
    "router_batch_from_index",
    "run_or_load_nouns_v2_gpu_preflight",
    "run_or_resume_nouns_v2_base",
    "run_or_resume_nouns_v2_baselines",
    "run_or_resume_nouns_v2_full_finetune",
    "run_or_resume_nouns_v2_vamp",
]

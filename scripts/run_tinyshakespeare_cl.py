"""Run the canonical four-task TinyShakespeare VAMP benchmark and report."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from apm.continual.language_benchmark_run import (
    LanguageBenchmarkSettings,
    run_language_benchmark,
)
from apm.continual.language_report import (
    LanguageReportManifest,
    canonical_config_json,
)
from apm.continual.language_report_build import write_language_benchmark_report
from apm.data.text.curricula import (
    CorpusSplits,
    TINY_SHAKESPEARE_EVALUATION_EXAMPLES_PER_TASK_AND_PREFIX,
    TINY_SHAKESPEARE_EVALUATION_PRESET,
    TINY_SHAKESPEARE_MACRO_DOCUMENT_CHARACTERS,
    build_tiny_shakespeare_permutation_curriculum,
    build_tiny_shakespeare_region_curriculum,
    build_tiny_shakespeare_stable_hash_curriculum,
)
from apm.data.text.language_tasks import (
    LanguageDataBuildConfig,
    prepare_language_curriculum,
    raw_tasks_from_character_curriculum,
    raw_tasks_from_corpus_curriculum,
)
from apm.lm.checkpoint import load_gpt_neo_checkpoint
from apm.lm.lora import LoraConfig
from apm.lm.text import CharTokenizer
from apm.lm.text_data import (
    TINY_SHAKESPEARE_SOURCE,
    load_tiny_shakespeare,
    split_text_contiguously,
)
from apm.lm.training import EDGE_TRAINING_PRESET
from apm.lm.workflow import tiny_shakespeare_model_config

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = REPOSITORY_ROOT / "data" / "tinyshakespeare" / "input.txt"
CHECKPOINT_PATH = REPOSITORY_ROOT / "checkpoints" / "tinyshakespeare-base"
RESULTS_ROOT = REPOSITORY_ROOT / "results"
CURRICULUM_CHOICES = (
    "character-permutation",
    "corpus-region",
    "stable-hash",
)


def main() -> None:
    """Load pinned local inputs, execute every baseline, and emit one report."""
    arguments = _parse_args()
    curriculum_name = arguments.curriculum
    corpus = load_tiny_shakespeare(CORPUS_PATH)
    raw_splits = split_text_contiguously(corpus)
    tokenizer = CharTokenizer.from_training_text(raw_splits.train)
    corpus_splits = CorpusSplits(
        train=raw_splits.train,
        validation=raw_splits.validation,
        test=raw_splits.test,
    )
    if curriculum_name == "character-permutation":
        character_curriculum = build_tiny_shakespeare_permutation_curriculum(
            corpus_splits
        )
        curriculum_id = character_curriculum.curriculum_id
        raw_tasks = raw_tasks_from_character_curriculum(character_curriculum)
    elif curriculum_name == "corpus-region":
        corpus_curriculum = build_tiny_shakespeare_region_curriculum(corpus_splits)
        curriculum_id = corpus_curriculum.curriculum_id
        raw_tasks = raw_tasks_from_corpus_curriculum(corpus_curriculum)
    else:
        corpus_curriculum = build_tiny_shakespeare_stable_hash_curriculum(
            corpus_splits
        )
        curriculum_id = corpus_curriculum.curriculum_id
        raw_tasks = raw_tasks_from_corpus_curriculum(corpus_curriculum)
    build_config = LanguageDataBuildConfig(
        context_length=256,
        batch_size=EDGE_TRAINING_PRESET.batch_size,
        stride=256,
        prefix_lengths=TINY_SHAKESPEARE_EVALUATION_PRESET.prefix_lengths,
        suffix_length=TINY_SHAKESPEARE_EVALUATION_PRESET.suffix_length,
        examples_per_task_and_prefix=(
            TINY_SHAKESPEARE_EVALUATION_EXAMPLES_PER_TASK_AND_PREFIX
        ),
        primary_prefix_length=64,
    )
    prepared = prepare_language_curriculum(
        curriculum_id,
        raw_tasks,
        (raw_splits.validation,),
        tokenizer,
        build_config,
    )
    checkpoint = load_gpt_neo_checkpoint(CHECKPOINT_PATH)
    expected_config = tiny_shakespeare_model_config(tokenizer.vocab_size)
    if checkpoint.config != expected_config:
        raise ValueError("TinyShakespeare checkpoint does not match the canonical model")
    lora_config = LoraConfig(rank=4, alpha=4.0)
    settings = LanguageBenchmarkSettings(
        seed=0,
        random_router_seed=0,
        negative_control_curriculum=curriculum_name == "stable-hash",
    )
    benchmark = run_language_benchmark(
        prepared,
        checkpoint.reference,
        checkpoint.params,
        checkpoint.config,
        lora_config,
        EDGE_TRAINING_PRESET,
        settings,
    )
    manifest = LanguageReportManifest(
        dataset="tinyshakespeare",
        curriculum=curriculum_name,
        preset="standard",
        seed=settings.seed,
        config_json=canonical_config_json(
            {
                "adapter": asdict(lora_config),
                "base_checkpoint": {
                    "manifest_sha256": checkpoint.reference.manifest_sha256,
                    "parameter_checksum": checkpoint.reference.parameter_checksum,
                },
                "benchmark": asdict(settings),
                "curriculum": {
                    "macro_document_characters": (
                        TINY_SHAKESPEARE_MACRO_DOCUMENT_CHARACTERS
                        if curriculum_name == "stable-hash"
                        else None
                    ),
                    "max_edges": prepared.curriculum.max_edges,
                    "max_nodes": prepared.curriculum.max_nodes,
                    "name": curriculum_name,
                    "task_ids": [
                        str(task.task_id) for task in prepared.curriculum.tasks
                    ],
                },
                "data": asdict(build_config),
                "dataset_revision": TINY_SHAKESPEARE_SOURCE.revision,
                "model": asdict(checkpoint.config),
                "optimizer": asdict(EDGE_TRAINING_PRESET),
                "tokenizer": {
                    "eos_token_id": tokenizer.eos_token_id,
                    "pad_token_id": tokenizer.pad_token_id,
                    "vocabulary": list(tokenizer.vocabulary),
                },
            }
        ),
    )
    output_directory = write_language_benchmark_report(
        RESULTS_ROOT,
        manifest,
        prepared,
        benchmark,
        checkpoint.params,
        checkpoint.config,
        lora_config,
        tokenizer,
    )
    print(output_directory)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--curriculum",
        choices=CURRICULUM_CHOICES,
        default="character-permutation",
        help="Four-task TinyShakespeare curriculum to run.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()

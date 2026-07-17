"""Run the canonical bounded TinyStories V2 topic benchmark and report."""

from __future__ import annotations

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
    TINYSTORIES_SINGLE_GPU_PRESET,
    TINYSTORIES_TOPICS,
    TINYSTORIES_V2_SOURCE,
    load_tinystories_topic_dataset,
)
from apm.data.text.language_tasks import (
    LanguageDataBuildConfig,
    prepare_language_curriculum,
    raw_tasks_from_document_curriculum,
)
from apm.lm.lora import LoraConfig
from apm.lm.text import TokenizersTextTokenizer
from apm.lm.tinystories_conversion import load_tinystories_artifact
from apm.lm.training import LmTrainConfig

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATA_DIRECTORY = REPOSITORY_ROOT / "data" / "tinystories-v2"
ARTIFACT_DIRECTORY = REPOSITORY_ROOT / "checkpoints" / "tinystories-8m"
RESULTS_ROOT = REPOSITORY_ROOT / "results"


def main() -> None:
    """Verify local data/model artifacts, run the topic benchmark, and report."""
    preset = TINYSTORIES_SINGLE_GPU_PRESET
    topic_dataset = load_tinystories_topic_dataset(
        DATA_DIRECTORY / TINYSTORIES_V2_SOURCE.train_file.filename,
        DATA_DIRECTORY / TINYSTORIES_V2_SOURCE.validation_file.filename,
        TINYSTORIES_V2_SOURCE,
        preset.stories_per_task,
    )
    artifact = load_tinystories_artifact(ARTIFACT_DIRECTORY)
    tokenizer = TokenizersTextTokenizer.from_file(
        ARTIFACT_DIRECTORY / "tokenizer" / "tokenizer.json"
    )
    build_config = LanguageDataBuildConfig(
        context_length=preset.context_length,
        batch_size=preset.batch_size,
        stride=preset.context_length,
        prefix_lengths=preset.evaluation.prefix_lengths,
        suffix_length=preset.evaluation.suffix_length,
        examples_per_task_and_prefix=(
            preset.evaluation_examples_per_task_and_prefix
        ),
        primary_prefix_length=64,
    )
    prepared = prepare_language_curriculum(
        topic_dataset.curriculum.curriculum_id,
        raw_tasks_from_document_curriculum(topic_dataset.curriculum),
        tuple(document.text for document in topic_dataset.root_validation),
        tokenizer,
        build_config,
    )
    lora_config = LoraConfig(rank=preset.lora_rank, alpha=preset.lora_alpha)
    train_config = LmTrainConfig(
        learning_rate=1e-3,
        steps=preset.adapter_steps_per_task,
        batch_size=preset.batch_size,
        weight_decay=0.01,
        gradient_clip_norm=1.0,
    )
    settings = LanguageBenchmarkSettings(
        seed=0,
        random_router_seed=0,
        evaluation_microbatch_size=8,
        peak_device_memory_target_bytes=(
            preset.peak_device_memory_gib * 1024**3
        ),
    )
    benchmark = run_language_benchmark(
        prepared,
        artifact.checkpoint.reference,
        artifact.checkpoint.params,
        artifact.checkpoint.config,
        lora_config,
        train_config,
        tokenizer,
        settings,
    )
    manifest = LanguageReportManifest(
        dataset="tinystories-v2-gpt4",
        curriculum="topic",
        preset="single-gpu",
        seed=settings.seed,
        config_json=canonical_config_json(
            {
                "adapter": asdict(lora_config),
                "base_checkpoint": {
                    "manifest_sha256": (
                        artifact.checkpoint.reference.manifest_sha256
                    ),
                    "parameter_checksum": (
                        artifact.checkpoint.reference.parameter_checksum
                    ),
                },
                "benchmark": asdict(settings),
                "checkpoint_source": asdict(artifact.checkpoint.source),
                "curriculum": {
                    "max_edges": prepared.curriculum.max_edges,
                    "max_nodes": prepared.curriculum.max_nodes,
                    "task_ids": [
                        str(task.task_id) for task in prepared.curriculum.tasks
                    ],
                },
                "data": asdict(build_config),
                "dataset_source": asdict(TINYSTORIES_V2_SOURCE),
                "model": asdict(artifact.checkpoint.config),
                "optimizer": asdict(train_config),
                "preset": asdict(preset),
                "tokenizer": asdict(artifact.checkpoint.tokenizer),
                "topic_lexicons": {
                    topic.name: {
                        concept.name: list(concept.forms)
                        for concept in topic.concepts
                    }
                    for topic in TINYSTORIES_TOPICS
                },
                "topic_rule": {
                    "matching": "case-folded whole words",
                    "minimum_distinct_concepts": 2,
                    "minimum_winner_margin": 1,
                    "selection": "lowest normalized-content SHA-256",
                },
            }
        ),
    )
    output_directory = write_language_benchmark_report(
        RESULTS_ROOT,
        manifest,
        prepared,
        benchmark,
        artifact.checkpoint.params,
    )
    print(output_directory)


if __name__ == "__main__":
    main()

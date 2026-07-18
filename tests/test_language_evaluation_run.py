from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import jax

from apm.continual.language_baseline_training import (
    train_language_adaptation_baselines,
)
from apm.continual.language_adaptation_artifact import (
    extract_language_adaptation_artifact,
    load_language_adaptation_artifact,
    save_language_adaptation_artifact,
)
from apm.continual.language_benchmark_run import LanguageBenchmarkSettings
from apm.continual.language_evaluation import (
    LanguageEvaluationCondition,
    LanguageEvaluationSuite,
    LanguageExampleProvenance,
    LanguageSuiteExample,
)
from apm.continual.language_evaluation_run import evaluate_language_benchmark
from apm.data.text.language_tasks import (
    LanguageDataBuildConfig,
    RawTextTask,
    prepare_language_curriculum,
)
from apm.lm.checkpoint import BaseCheckpointRef, parameter_checksum
from apm.lm.config import GptNeoConfig
from apm.lm.lora import LoraConfig
from apm.lm.parameters import init_gpt_neo_params
from apm.lm.text import CharTokenizer
from apm.lm.training import LmTrainConfig
from apm.memory.address_refinement import EbtConfig


def test_already_trained_adaptations_evaluate_without_tensor_changes(tmp_path) -> None:
    text = "abcdefghijklmnopqrstuvwxyz" * 3
    tokenizer = CharTokenizer.from_training_text(text)
    prepared = prepare_language_curriculum(
        "evaluation-only",
        (
            RawTextTask(
                task_id="task_0",
                train_texts=(text,),
                validation_texts=(text,),
                test_texts=(text[::-1],),
            ),
        ),
        (text,),
        tokenizer,
        LanguageDataBuildConfig(
            context_length=8,
            batch_size=1,
            stride=8,
            prefix_lengths=(4,),
            suffix_length=2,
            examples_per_task_and_prefix=1,
            primary_prefix_length=4,
        ),
    )
    model_config = GptNeoConfig(
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=8,
        hidden_size=4,
        intermediate_size=8,
        num_layers=1,
        num_heads=1,
        attention_types=("global",),
        local_window_size=4,
    )
    base_params = init_gpt_neo_params(jax.random.PRNGKey(0), model_config)
    checkpoint = BaseCheckpointRef(
        Path("base"),
        "a" * 64,
        parameter_checksum(base_params, model_config),
    )
    lora_config = LoraConfig(rank=1, alpha=1.0)
    train_config = LmTrainConfig(
        learning_rate=1e-2,
        steps=1,
        batch_size=1,
        weight_decay=0.0,
    )
    adaptations = train_language_adaptation_baselines(
        prepared.curriculum,
        prepared.root_validation_probes,
        checkpoint,
        base_params,
        model_config,
        lora_config,
        train_config,
        jax.random.PRNGKey(1),
    )
    raw_example = prepared.evaluation_sweeps[0].test_examples[0]
    suite = LanguageEvaluationSuite(
        suite_id="bounded-suite",
        benchmark_label="bounded",
        primary_condition_id="p4-s2",
        conditions=(LanguageEvaluationCondition("p4-s2", 4, 2),),
        examples=(
            LanguageSuiteExample(
                pair_id="pair-0",
                condition_id="p4-s2",
                split="test",
                example=raw_example,
                provenance=LanguageExampleProvenance(
                    "document-0",
                    0,
                    sha256(b"pair-0").hexdigest(),
                ),
                cue_regime="cue_hidden_or_ambiguous",
                visible_concept_ids=(),
            ),
        ),
    )

    result = evaluate_language_benchmark(
        adaptations,
        suite,
        base_params,
        model_config,
        lora_config,
        LanguageBenchmarkSettings(
            ebt=EbtConfig(steps=1, learning_rate=0.1),
            timing_warm_repetitions=1,
            sample_new_tokens=1,
        ),
    )

    assert result.adaptation_checksum_before == result.adaptation_checksum_after
    assert {row.method for row in result.measurements} == {
        "frozen_base",
        "sequential_single_lora",
        "independent_root_lora",
        "vamp_oracle",
        "vamp_exhaustive",
        "vamp_hopfield",
        "vamp_ebt_uniform",
        "vamp_ebt_hopfield",
        "deterministic_random_node",
    }
    assert {row.cue_regime for row in result.measurements} == {
        "cue_hidden_or_ambiguous",
        "all",
    }

    artifact = extract_language_adaptation_artifact(
        adaptations,
        model_config,
        lora_config,
    )
    save_language_adaptation_artifact(tmp_path / "adaptation", artifact)
    loaded = load_language_adaptation_artifact(tmp_path / "adaptation")
    loaded_result = evaluate_language_benchmark(
        loaded,
        suite,
        base_params,
        model_config,
        lora_config,
        LanguageBenchmarkSettings(
            ebt=EbtConfig(steps=1, learning_rate=0.1),
            timing_warm_repetitions=1,
            sample_new_tokens=1,
        ),
    )
    assert loaded_result.measurements == result.measurements
    assert loaded_result.adaptation_checksum_before == artifact.tensor_checksum

from __future__ import annotations

import json
from pathlib import Path

import jax
import numpy as np
import pytest

import apm.continual.language_benchmark_run as language_benchmark_run
from apm.continual.language_benchmark_run import (
    LanguageBenchmarkSettings,
    _peak_device_memory_from_stats,
    run_language_benchmark,
)
from apm.continual.language_report import (
    LanguageReportManifest,
    canonical_config_json,
    write_language_report,
)
from apm.continual.language_report_build import build_language_report_bundle
from apm.continual.language_tasks import BaseCheckpointRef
from apm.data.text.language_tasks import (
    LanguageDataBuildConfig,
    RawTextTask,
    prepare_language_curriculum,
)
from apm.lm.checkpoint import parameter_checksum
from apm.lm.config import GptNeoConfig
from apm.lm.lora import LoraConfig
from apm.lm.parameters import init_gpt_neo_params
from apm.lm.text import CharTokenizer
from apm.lm.training import LmTrainConfig
from apm.memory.address_refinement import EbtConfig


def test_bounded_language_benchmark_emits_every_baseline_metric_family(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    text = "abcdefghijklmnopqrstuvwxyz" * 3
    tokenizer = CharTokenizer.from_training_text(text)
    prepared = prepare_language_curriculum(
        "bounded",
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
        directory=Path("bounded-base"),
        manifest_sha256="a" * 64,
        parameter_checksum=parameter_checksum(base_params, model_config),
    )
    lifecycle_events = []
    generate_language_samples = language_benchmark_run._generate_language_samples
    measure_peak_device_memory = language_benchmark_run.measure_peak_device_memory

    def observe_sample_generation(*args, **kwargs):
        lifecycle_events.append("samples")
        return generate_language_samples(*args, **kwargs)

    def observe_peak_measurement(*args, **kwargs):
        lifecycle_events.append("peak")
        return measure_peak_device_memory(*args, **kwargs)

    monkeypatch.setattr(
        language_benchmark_run,
        "_generate_language_samples",
        observe_sample_generation,
    )
    monkeypatch.setattr(
        language_benchmark_run,
        "measure_peak_device_memory",
        observe_peak_measurement,
    )
    result = run_language_benchmark(
        prepared,
        checkpoint,
        base_params,
        model_config,
        LoraConfig(rank=1, alpha=1.0),
        LmTrainConfig(
            learning_rate=1e-2,
            steps=1,
            batch_size=1,
            weight_decay=0.0,
        ),
        tokenizer,
        LanguageBenchmarkSettings(
            seed=0,
            random_router_seed=1,
            ebt=EbtConfig(steps=1, learning_rate=0.1),
            timing_warm_repetitions=1,
            sample_new_tokens=1,
            negative_control_curriculum=True,
        ),
    )

    assert lifecycle_events == ["samples", "peak"]
    assert {row.baseline for row in result.stored_competence} == {
        "frozen_base",
        "sequential_single_lora",
        "independent_root_lora",
        "vamp_oracle",
    }
    assert {row.router for row in result.routing} == {
        "vamp_exhaustive",
        "vamp_hopfield",
        "vamp_ebt_uniform",
        "vamp_ebt_hopfield",
        "deterministic_random_node",
    }
    assert len(result.transfer) == 1
    assert len(result.memory) == 1
    assert len(result.addressing_cost) == 5
    assert len(result.addressing_traces) == 2
    assert {trace.router for trace in result.addressing_traces} == {
        "vamp_ebt_uniform",
        "vamp_ebt_hopfield",
    }
    assert all(trace.objective_trace.shape == (2,) for trace in result.addressing_traces)
    assert all(
        trace.node_probabilities.shape == (2, 2)
        and trace.edge_coefficients.shape == (2, 1)
        for trace in result.addressing_traces
    )
    assert len(result.samples) == 9
    assert int(np.sum(result.final_confusion)) == 1
    assert all(row.base_checksum_stable for row in result.stored_competence)
    assert result.memory[0].accounting.persistent_bytes > 0
    assert result.memory[0].accounting.packed_runtime_bytes > 0
    assert all(
        timing.timing.cold_compile_seconds > 0.0
        and timing.timing.warm_latency_seconds > 0.0
        for timing in result.addressing_cost
    )
    assert result.peak_device_memory.platform
    assert result.peak_device_memory.target_bytes is None
    assert result.peak_device_memory.within_target is None
    random_row = next(
        row
        for row in result.routing
        if row.router == "deterministic_random_node"
    )
    assert random_row.negative_control_chance_accuracy == 1.0
    assert random_row.negative_control_chance_in_ci95 is True
    assert random_row.leakage_audit_required is False
    assert all(
        row.negative_control_chance_accuracy == 1.0
        and row.negative_control_chance_in_ci95 is not None
        and row.leakage_audit_required is not None
        for row in result.routing
    )

    bundle = build_language_report_bundle(
        LanguageReportManifest(
            dataset="tinyshakespeare",
            curriculum="character-permutation",
            preset="bounded",
            seed=0,
            config_json=canonical_config_json({"examples": 1, "steps": 1}),
        ),
        prepared,
        result,
        base_params,
    )
    assert lifecycle_events == ["samples", "peak"]
    assert bundle.samples is result.samples
    assert bundle.addressing_traces is result.addressing_traces
    output_directory = write_language_report(tmp_path, bundle)
    assert len(bundle.samples) == 9
    assert (output_directory / "report.html").is_file()
    assert (output_directory / "graph.svg").is_file()
    assert (output_directory / "ebt_objective_trace.svg").is_file()
    routing_record = next(
        json.loads(line)
        for line in (output_directory / "routing_metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if '"router":"deterministic_random_node"' in line
    )
    assert routing_record["negative_control_chance_in_ci95"] is True
    memory_record = json.loads(
        (output_directory / "memory_metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[-1]
    )
    assert "peak_device_memory_bytes" in memory_record
    assert "peak_device_memory_target_bytes" in memory_record
    assert memory_record["device_platform"] == result.peak_device_memory.platform


def test_peak_device_memory_target_is_validated_enforced_and_reportable() -> None:
    measurement = _peak_device_memory_from_stats(
        platform="gpu",
        device_kind="test accelerator",
        memory_stats={"peak_bytes_in_use": 9, "bytes_limit": 16},
        target_bytes=10,
    )

    assert measurement.peak_bytes_in_use == 9
    assert measurement.bytes_limit == 16
    assert measurement.within_target is True
    with pytest.raises(MemoryError, match="exceeded target"):
        _peak_device_memory_from_stats(
            platform="gpu",
            device_kind="test accelerator",
            memory_stats={"peak_bytes_in_use": 11},
            target_bytes=10,
        )
    with pytest.raises(RuntimeError, match="does not expose"):
        _peak_device_memory_from_stats(
            platform="cpu",
            device_kind="test cpu",
            memory_stats=None,
            target_bytes=10,
        )


def test_benchmark_evaluation_microbatch_setting_is_optional_and_validated() -> None:
    assert LanguageBenchmarkSettings().evaluation_microbatch_size is None
    assert (
        LanguageBenchmarkSettings(
            evaluation_microbatch_size=8
        ).evaluation_microbatch_size
        == 8
    )
    with pytest.raises(ValueError, match="positive integer"):
        LanguageBenchmarkSettings(evaluation_microbatch_size=0)
    with pytest.raises(ValueError, match="sample_new_tokens must be positive"):
        LanguageBenchmarkSettings(sample_new_tokens=0)

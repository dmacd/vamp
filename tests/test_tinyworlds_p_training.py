from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from apm.data.text.tinyworlds_p import (
    BASE_TRAINING_PRESET,
    StreamingTrainingConfig,
    build_partition,
    evaluate_partition_split,
    iter_partition_batches,
    load_streaming_checkpoint,
    run_streaming_base_training,
)
from apm.lm.config import GptNeoConfig
from apm.lm.training_state_artifact import lm_train_state_checksum
from test_tinyworlds_p_partition import _fixture_inputs, _fixture_preset


def _tiny_training_config(vocab_size: int) -> StreamingTrainingConfig:
    return StreamingTrainingConfig(
        model_config=GptNeoConfig(
            vocab_size=vocab_size,
            max_position_embeddings=8,
            hidden_size=8,
            intermediate_size=16,
            num_layers=1,
            num_heads=2,
            attention_types=("global",),
            local_window_size=4,
        ),
        epochs=2,
        calibration_epochs=1,
        context_length=8,
        microbatch_size=4,
        accumulation_microbatches=4,
        maximum_learning_rate=1e-2,
        minimum_learning_rate=1e-3,
        warmup_fraction=0.1,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        weight_decay=0.01,
        gradient_clip_norm=1.0,
        parameter_seed=0,
        state_interval_updates=100,
        allocator_peak_limit_bytes=1024**3,
    )


def test_fixed_runner_uses_the_base_policy_calibration_prefix() -> None:
    config = StreamingTrainingConfig.from_preset()

    assert config.epochs == BASE_TRAINING_PRESET.epochs == 5
    assert config.calibration_epochs == BASE_TRAINING_PRESET.calibration_epochs == 2


def test_streaming_training_is_token_weighted_and_resume_identical(
    tmp_path: Path,
) -> None:
    inputs, _ = _fixture_inputs(tmp_path, "partition-output", "partition-work")
    artifact = build_partition(inputs, _fixture_preset())
    config = _tiny_training_config(artifact.tokenizer_identity.vocab_size)
    active_tokens_per_epoch = sum(
        int(np.count_nonzero(batch.loss_mask))
        for batch in iter_partition_batches(artifact, "base/train", epoch=0)
    )

    uninterrupted = run_streaming_base_training(
        artifact,
        tmp_path / "uninterrupted",
        config,
    )
    interrupted = run_streaming_base_training(
        artifact,
        tmp_path / "resumed",
        config,
        stop_after_update=5,
    )
    resumed = run_streaming_base_training(
        artifact,
        tmp_path / "resumed",
        config,
        resume_from=interrupted.checkpoints[-1].directory,
    )

    assert resumed.cursor == uninterrupted.cursor
    assert lm_train_state_checksum(resumed.state) == lm_train_state_checksum(
        uninterrupted.state
    )
    assert resumed.trace_path.read_bytes() == uninterrupted.trace_path.read_bytes()
    trace = tuple(json.loads(line) for line in resumed.trace_path.read_bytes().splitlines())
    assert sum(record["active_tokens"] for record in trace) == (
        active_tokens_per_epoch * config.epochs
    )
    assert all(record["active_tokens"] > 0 for record in trace)
    validation = evaluate_partition_split(
        resumed.state.trainable,
        artifact,
        "base/validation",
        config.model_config,
    )
    assert validation.active_tokens > 0
    assert np.isfinite(validation.nll)


def test_training_checkpoint_rejects_old_resume_format(tmp_path: Path) -> None:
    inputs, _ = _fixture_inputs(tmp_path, "partition-output", "partition-work")
    artifact = build_partition(inputs, _fixture_preset())
    config = _tiny_training_config(artifact.tokenizer_identity.vocab_size)
    interrupted = run_streaming_base_training(
        artifact,
        tmp_path / "training",
        config,
        stop_after_update=1,
    )
    checkpoint = interrupted.checkpoints[-1]

    with pytest.raises(ValueError, match="resume identity changed"):
        load_streaming_checkpoint(
            checkpoint.directory,
            "0" * 64,
            interrupted.state,
        )

    resume_path = checkpoint.directory / "resume.json"
    resume = json.loads(resume_path.read_bytes())
    resume["format"] = "tinyworlds-p-training-resume"
    resume_path.write_text(
        json.dumps(resume, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="resume identity changed"):
        load_streaming_checkpoint(
            checkpoint.directory,
            interrupted.training_sha256,
            interrupted.state,
        )

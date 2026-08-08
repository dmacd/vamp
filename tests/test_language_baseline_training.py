from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import jax
import numpy as np
import pytest

from apm.continual.language_adaptation_artifact import (
    attach_language_baseline_runs,
    extract_language_adaptation_artifact,
    extract_language_vamp_artifact,
)
from apm.continual.language_baseline_training import (
    advance_independent_root_lora_progress,
    advance_sequential_lora_progress,
    complete_independent_root_lora_progress,
    complete_sequential_lora_progress,
    init_independent_root_lora_progress,
    init_sequential_lora_progress,
    train_independent_root_lora,
    train_language_adaptation_baselines,
    train_sequential_single_lora,
)
from apm.continual.language_tasks import (
    LanguageCurriculum,
    LanguageEvaluationExample,
    LanguageTask,
    RouterBatch,
    TaskId,
    build_prefix_suffix_batches,
)
from apm.lm.checkpoint import BaseCheckpointRef, parameter_checksum
from apm.lm.config import GptNeoConfig
from apm.lm.lora import LoraConfig
from apm.lm.parameters import GptNeoParams, init_gpt_neo_params
from apm.lm.text_data import TokenBatch
from apm.lm.training import LmTrainConfig
from apm.memory.graph import NodeId


def _model_config() -> GptNeoConfig:
    return GptNeoConfig(
        vocab_size=10,
        max_position_embeddings=8,
        hidden_size=4,
        intermediate_size=8,
        num_layers=1,
        num_heads=2,
        attention_types=("global",),
        local_window_size=2,
    )


def _train_config(steps: int = 1) -> LmTrainConfig:
    return LmTrainConfig(
        learning_rate=1e-2,
        steps=steps,
        batch_size=1,
        weight_decay=0.0,
    )


def _training_batch(tokens: tuple[int, ...]) -> TokenBatch:
    return TokenBatch(
        input_ids=np.asarray((tokens[:-1],), dtype=np.int32),
        attention_mask=np.ones((1, len(tokens) - 1), dtype=np.bool_),
        target_ids=np.asarray((tokens[1:],), dtype=np.int32),
        loss_mask=np.ones((1, len(tokens) - 1), dtype=np.bool_),
    )


def _task(task_id: str, tokens: tuple[int, ...], *, evaluated: bool) -> LanguageTask:
    evaluation_examples: tuple[LanguageEvaluationExample, ...] = ()
    if evaluated:
        router_batch, competence_batch = build_prefix_suffix_batches(
            tokens + (tokens[-1],),
            prefix_length=3,
            suffix_length=2,
        )
        evaluation_examples = (
            LanguageEvaluationExample(
                router_batch=router_batch,
                competence_batch=competence_batch,
                task_id=TaskId(task_id),
                oracle_node_id=NodeId(task_id),
            ),
        )
    return LanguageTask(
        task_id=TaskId(task_id),
        train_batches=(_training_batch(tokens),),
        validation_examples=evaluation_examples,
        test_examples=evaluation_examples,
    )


def _curriculum(tasks: tuple[LanguageTask, ...]) -> LanguageCurriculum:
    return LanguageCurriculum(
        tasks=tasks,
        max_nodes=len(tasks) + 1,
        max_edges=len(tasks),
    )


def _base_reference(
    params: GptNeoParams,
    config: GptNeoConfig,
) -> BaseCheckpointRef:
    return BaseCheckpointRef(
        directory=Path("unused-baseline-checkpoint"),
        manifest_sha256="0" * 64,
        parameter_checksum=parameter_checksum(params, config),
    )


def _tree_checksum(tree: object) -> str:
    digest = sha256()
    for leaf in jax.tree_util.tree_leaves(tree):
        array = np.asarray(leaf)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _stack_router_rows(batch: RouterBatch, count: int) -> RouterBatch:
    return RouterBatch(
        input_ids=np.repeat(batch.input_ids, count, axis=0),
        attention_mask=np.repeat(batch.attention_mask, count, axis=0),
        target_ids=np.repeat(batch.target_ids, count, axis=0),
        loss_mask=np.repeat(batch.loss_mask, count, axis=0),
    )


def test_sequential_snapshots_and_independent_streams_are_immutable() -> None:
    config = _model_config()
    params = init_gpt_neo_params(jax.random.PRNGKey(0), config)
    lora_config = LoraConfig(rank=1, alpha=1.0)
    train_config = _train_config(steps=2)
    tasks = (
        _task("task-a", (1, 2, 3, 4, 5), evaluated=False),
        _task("task-b", (5, 4, 3, 2, 1), evaluated=False),
    )
    curriculum = _curriculum(tasks)
    initial_rng = jax.random.PRNGKey(11)
    base_checksum = parameter_checksum(params, config)

    sequential = train_sequential_single_lora(
        curriculum,
        params,
        config,
        lora_config,
        train_config,
        initial_rng,
    )
    first_task_only = train_sequential_single_lora(
        _curriculum(tasks[:1]),
        params,
        config,
        lora_config,
        train_config,
        initial_rng,
    )
    independent = train_independent_root_lora(
        curriculum,
        params,
        config,
        lora_config,
        train_config,
        initial_rng,
    )

    assert parameter_checksum(params, config) == base_checksum
    assert sequential.base_parameter_checksum == base_checksum
    assert independent.base_parameter_checksum == base_checksum
    assert sequential.train_config == independent.train_config == train_config
    assert tuple(len(stage.step_losses) for stage in sequential.stages) == (2, 2)
    assert tuple(len(adapter.step_losses) for adapter in independent.adapters) == (2, 2)
    assert _tree_checksum(sequential.stages[0].adapter) == _tree_checksum(
        first_task_only.stages[0].adapter
    )
    assert _tree_checksum(sequential.stages[0].adapter) != _tree_checksum(
        sequential.stages[1].adapter
    )
    assert _tree_checksum(independent.adapters[0].adapter) != _tree_checksum(
        independent.adapters[1].adapter
    )
    assert not np.array_equal(sequential.rng_key, initial_rng)
    assert not np.array_equal(independent.rng_key, initial_rng)


def test_combined_baselines_share_budget_base_hash_and_distinct_rng_streams() -> None:
    config = _model_config()
    params = init_gpt_neo_params(jax.random.PRNGKey(3), config)
    lora_config = LoraConfig(rank=1, alpha=1.0)
    train_config = _train_config()
    task = _task("task-a", (1, 2, 3, 4, 5), evaluated=True)
    root_probe = _stack_router_rows(task.validation_examples[0].router_batch, 2)
    base_checksum = parameter_checksum(params, config)

    baselines = train_language_adaptation_baselines(
        _curriculum((task,)),
        (root_probe,),
        _base_reference(params, config),
        params,
        config,
        lora_config,
        train_config,
        jax.random.PRNGKey(19),
    )

    assert baselines.base_parameter_checksum == base_checksum
    assert parameter_checksum(params, config) == base_checksum
    assert baselines.train_config == train_config
    assert len(baselines.sequential_single_lora.stages[0].step_losses) == 1
    assert len(baselines.independent_root_lora.adapters[0].step_losses) == 1
    assert len(baselines.vamp.stage_metrics[0].candidate_step_losses) == 1
    final_rng_keys = (
        baselines.sequential_single_lora.rng_key,
        baselines.independent_root_lora.rng_key,
        baselines.vamp.rng_key,
    )
    assert len({tuple(np.asarray(key).tolist()) for key in final_rng_keys}) == 3


def test_separately_trained_baselines_attach_to_the_exact_vamp_artifact() -> None:
    config = _model_config()
    params = init_gpt_neo_params(jax.random.PRNGKey(31), config)
    lora_config = LoraConfig(rank=1, alpha=1.0)
    train_config = _train_config()
    task = _task("task-a", (1, 2, 3, 4, 5), evaluated=True)
    root_probe = _stack_router_rows(task.validation_examples[0].router_batch, 2)
    trained = train_language_adaptation_baselines(
        _curriculum((task,)),
        (root_probe,),
        _base_reference(params, config),
        params,
        config,
        lora_config,
        train_config,
        jax.random.PRNGKey(37),
    )
    expected = extract_language_adaptation_artifact(
        trained,
        config,
        lora_config,
    )
    vamp_only = extract_language_vamp_artifact(
        trained.vamp,
        config,
        lora_config,
        train_config,
    )
    attached = attach_language_baseline_runs(
        vamp_only,
        trained.sequential_single_lora,
        trained.independent_root_lora,
    )

    assert attached.tensor_checksum == expected.tensor_checksum
    assert attached.tensor_checksums == expected.tensor_checksums
    assert attached.task_order == expected.task_order
    assert dict(attached.config_hashes) == dict(expected.config_hashes)


def test_training_contract_rejects_different_task_sequence_widths() -> None:
    config = _model_config()
    params = init_gpt_neo_params(jax.random.PRNGKey(5), config)
    curriculum = _curriculum(
        (
            _task("short", (1, 2, 3, 4), evaluated=False),
            _task("long", (1, 2, 3, 4, 5), evaluated=False),
        )
    )

    with pytest.raises(ValueError, match="one sequence width"):
        train_sequential_single_lora(
            curriculum,
            params,
            config,
            LoraConfig(rank=1, alpha=1.0),
            _train_config(),
            jax.random.PRNGKey(6),
        )


def test_task_boundary_resume_matches_uninterrupted_baselines() -> None:
    config = _model_config()
    params = init_gpt_neo_params(jax.random.PRNGKey(8), config)
    lora_config = LoraConfig(rank=1, alpha=1.0)
    train_config = _train_config(steps=1)
    tasks = (
        _task("task-a", (1, 2, 3, 4, 5), evaluated=False),
        _task("task-b", (5, 4, 3, 2, 1), evaluated=False),
    )
    curriculum = _curriculum(tasks)
    rng_key = jax.random.PRNGKey(27)
    uninterrupted_sequential = train_sequential_single_lora(
        curriculum,
        params,
        config,
        lora_config,
        train_config,
        rng_key,
    )
    uninterrupted_independent = train_independent_root_lora(
        curriculum,
        params,
        config,
        lora_config,
        train_config,
        rng_key,
    )

    sequential_progress = init_sequential_lora_progress(
        params,
        config,
        lora_config,
        train_config,
        rng_key,
    )
    independent_progress = init_independent_root_lora_progress(
        params,
        config,
        train_config,
        rng_key,
    )
    sequential_progress = advance_sequential_lora_progress(
        sequential_progress,
        tasks[0],
        params,
        config,
        lora_config,
    )
    independent_progress = advance_independent_root_lora_progress(
        independent_progress,
        tasks[0],
        params,
        config,
        lora_config,
    )
    resumed_sequential = complete_sequential_lora_progress(
        advance_sequential_lora_progress(
            sequential_progress,
            tasks[1],
            params,
            config,
            lora_config,
        )
    )
    resumed_independent = complete_independent_root_lora_progress(
        advance_independent_root_lora_progress(
            independent_progress,
            tasks[1],
            params,
            config,
            lora_config,
        )
    )

    assert tuple(stage.step_losses for stage in resumed_sequential.stages) == tuple(
        stage.step_losses for stage in uninterrupted_sequential.stages
    )
    assert tuple(adapter.step_losses for adapter in resumed_independent.adapters) == tuple(
        adapter.step_losses for adapter in uninterrupted_independent.adapters
    )
    assert _tree_checksum(resumed_sequential.stages[-1].adapter) == _tree_checksum(
        uninterrupted_sequential.stages[-1].adapter
    )
    assert _tree_checksum(resumed_independent.adapters[-1].adapter) == _tree_checksum(
        uninterrupted_independent.adapters[-1].adapter
    )
    np.testing.assert_array_equal(
        resumed_sequential.rng_key,
        uninterrupted_sequential.rng_key,
    )
    np.testing.assert_array_equal(
        resumed_independent.rng_key,
        uninterrupted_independent.rng_key,
    )

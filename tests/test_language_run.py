from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import pytest

from apm.continual.language_run import (
    advance_language_vamp_run,
    init_language_vamp_run,
    score_parent_nodes,
)
from apm.continual.language_tasks import (
    LanguageEvaluationExample,
    LanguageTask,
    NodeId,
    TaskId,
    build_prefix_suffix_batches,
)
from apm.lm.checkpoint import BaseCheckpointRef, parameter_checksum
from apm.lm.config import GptNeoConfig
from apm.lm.gpt_neo import apply_gpt_neo
from apm.lm.lora import LoraConfig
from apm.lm.lora_memory import edge_coefficients_for_node, pack_lora_memory
from apm.lm.parameters import GptNeoParams, init_gpt_neo_params
from apm.lm.text_data import TokenBatch
from apm.lm.training import LmTrainConfig
from apm.memory.graph import memory_node_path


def _model_config() -> GptNeoConfig:
    return GptNeoConfig(
        vocab_size=8,
        max_position_embeddings=4,
        hidden_size=4,
        intermediate_size=8,
        num_layers=1,
        num_heads=2,
        attention_types=("global",),
        local_window_size=4,
    )


def _task(task_name: str, tokens: tuple[int, ...]) -> LanguageTask:
    task_id = TaskId(task_name)
    router_batch, competence_batch = build_prefix_suffix_batches(
        tokens,
        prefix_length=3,
        suffix_length=2,
    )
    example = LanguageEvaluationExample(
        router_batch=router_batch,
        competence_batch=competence_batch,
        task_id=task_id,
        oracle_node_id=NodeId(task_name),
    )
    return LanguageTask(
        task_id=task_id,
        train_batches=(
            TokenBatch(
                input_ids=np.asarray((tokens[:-1],), dtype=np.int32),
                attention_mask=np.ones((1, 4), dtype=np.bool_),
                target_ids=np.asarray((tokens[1:],), dtype=np.int32),
                loss_mask=np.ones((1, 4), dtype=np.bool_),
            ),
        ),
        validation_examples=(example,),
        test_examples=(example,),
    )


def _base_reference(params: GptNeoParams, config: GptNeoConfig) -> BaseCheckpointRef:
    return BaseCheckpointRef(
        directory=Path("unused-language-run-checkpoint"),
        manifest_sha256="0" * 64,
        parameter_checksum=parameter_checksum(params, config),
    )


def _assert_trees_equal(first, second) -> None:
    first_leaves, first_structure = jax.tree_util.tree_flatten(first)
    second_leaves, second_structure = jax.tree_util.tree_flatten(second)
    assert first_structure == second_structure
    assert all(
        np.array_equal(np.asarray(first_leaf), np.asarray(second_leaf))
        for first_leaf, second_leaf in zip(first_leaves, second_leaves)
    )


def _assert_graphs_equal(first, second) -> None:
    assert len(first.nodes) == len(second.nodes)
    for first_node, second_node in zip(first.nodes, second.nodes):
        assert (
            first_node.node_id,
            first_node.parent_id,
            first_node.trained_task,
            first_node.train_stage,
            first_node.depth,
        ) == (
            second_node.node_id,
            second_node.parent_id,
            second_node.trained_task,
            second_node.train_stage,
            second_node.depth,
        )
        if first_node.incoming_edge is None:
            assert second_node.incoming_edge is None
        else:
            _assert_trees_equal(first_node.incoming_edge, second_node.incoming_edge)


def _node_logits(run, params, config, lora_config, node_index, batch):
    packed_memory = pack_lora_memory(
        run.graph,
        config,
        lora_config,
        run.max_nodes,
        run.max_edges,
    )
    return apply_gpt_neo(
        params,
        config,
        batch.input_ids,
        batch.attention_mask,
        lora_memory=packed_memory,
        edge_coefficients=edge_coefficients_for_node(packed_memory, node_index),
        lora_config=lora_config,
        training=False,
    ).logits


def test_two_task_transition_is_immutable_stable_and_deterministic() -> None:
    config = _model_config()
    params = init_gpt_neo_params(jax.random.PRNGKey(0), config)
    base_reference = _base_reference(params, config)
    tasks = (
        _task("task-a", (1, 2, 3, 4, 5)),
        _task("task-b", (5, 4, 3, 2, 1)),
    )
    lora_config = LoraConfig(rank=2, alpha=2.0)
    train_config = LmTrainConfig(
        learning_rate=1e-2,
        steps=1,
        batch_size=1,
        weight_decay=0.0,
    )
    initial_run = init_language_vamp_run(
        base_reference,
        params,
        config,
        (tasks[0].validation_examples[0].router_batch,),
        jax.random.PRNGKey(1),
        max_nodes=3,
        max_edges=2,
        key_probe_count=1,
    )

    first_run = advance_language_vamp_run(
        initial_run,
        tasks[0],
        params,
        config,
        lora_config,
        train_config,
        score_parent_nodes(
            initial_run,
            tasks[0].parent_probes,
            params,
            config,
            lora_config,
        ),
        key_probe_count=1,
    )
    first_logits = _node_logits(
        first_run,
        params,
        config,
        lora_config,
        node_index=1,
        batch=tasks[0].train_batches[0],
    )
    first_edge = first_run.graph.nodes[1].incoming_edge
    second_run = advance_language_vamp_run(
        first_run,
        tasks[1],
        params,
        config,
        lora_config,
        train_config,
        score_parent_nodes(
            first_run,
            tasks[1].parent_probes,
            params,
            config,
            lora_config,
        ),
        key_probe_count=1,
    )
    repeated_first_run = advance_language_vamp_run(
        initial_run,
        tasks[0],
        params,
        config,
        lora_config,
        train_config,
        score_parent_nodes(
            initial_run,
            tasks[0].parent_probes,
            params,
            config,
            lora_config,
        ),
        key_probe_count=1,
    )
    repeated_second_run = advance_language_vamp_run(
        repeated_first_run,
        tasks[1],
        params,
        config,
        lora_config,
        train_config,
        score_parent_nodes(
            repeated_first_run,
            tasks[1].parent_probes,
            params,
            config,
            lora_config,
        ),
        key_probe_count=1,
    )

    assert len(initial_run.graph.nodes) == 1
    assert len(first_run.graph.nodes) == 2
    assert len(second_run.graph.nodes) == second_run.max_nodes == 3
    assert second_run.address_book.node_ids == (
        NodeId("root"),
        NodeId("task-a"),
        NodeId("task-b"),
    )
    np.testing.assert_array_equal(
        second_run.address_book.valid_node_mask,
        (True, True, True),
    )
    assert tuple(
        node.node_id
        for node in memory_node_path(second_run.graph, NodeId("task-b"))
    ) in (
        (NodeId("root"), NodeId("task-b")),
        (NodeId("root"), NodeId("task-a"), NodeId("task-b")),
    )
    assert tuple(metric.task_id for metric in second_run.stage_metrics[-1].task_metrics) == (
        TaskId("task-a"),
        TaskId("task-b"),
    )
    assert parameter_checksum(params, config) == base_reference.parameter_checksum
    _assert_trees_equal(first_edge, first_run.graph.nodes[1].incoming_edge)
    _assert_trees_equal(first_edge, second_run.graph.nodes[1].incoming_edge)
    np.testing.assert_array_equal(
        first_logits,
        _node_logits(
            second_run,
            params,
            config,
            lora_config,
            node_index=1,
            batch=tasks[0].train_batches[0],
        ),
    )
    _assert_graphs_equal(second_run.graph, repeated_second_run.graph)
    np.testing.assert_array_equal(
        second_run.address_book.keys,
        repeated_second_run.address_book.keys,
    )
    np.testing.assert_array_equal(second_run.rng_key, repeated_second_run.rng_key)
    assert second_run.stage_metrics == repeated_second_run.stage_metrics


def test_init_requires_exact_probe_count_and_advance_rejects_node_collisions() -> None:
    config = _model_config()
    params = init_gpt_neo_params(jax.random.PRNGKey(2), config)
    task = _task("task-a", (1, 3, 5, 7, 2))
    base_reference = _base_reference(params, config)
    probes = (task.validation_examples[0].router_batch,)

    with pytest.raises(ValueError, match="expected exactly 2"):
        init_language_vamp_run(
            base_reference,
            params,
            config,
            probes,
            jax.random.PRNGKey(3),
            max_nodes=3,
            max_edges=2,
            key_probe_count=2,
        )

    initial_run = init_language_vamp_run(
        base_reference,
        params,
        config,
        probes,
        jax.random.PRNGKey(3),
        max_nodes=3,
        max_edges=2,
        key_probe_count=1,
    )
    train_config = LmTrainConfig(learning_rate=1e-2, steps=1, batch_size=1)
    lora_config = LoraConfig(rank=2, alpha=2.0)
    first_run = advance_language_vamp_run(
        initial_run,
        task,
        params,
        config,
        lora_config,
        train_config,
        score_parent_nodes(
            initial_run,
            task.parent_probes,
            params,
            config,
            lora_config,
        ),
        key_probe_count=1,
    )

    with pytest.raises(ValueError, match="already exists"):
        advance_language_vamp_run(
            first_run,
            task,
            params,
            config,
            lora_config,
            train_config,
            score_parent_nodes(
                first_run,
                task.parent_probes,
                params,
                config,
                lora_config,
            ),
            key_probe_count=1,
        )

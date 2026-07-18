from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from apm.continual.language_run import (
    LanguageVampRun,
    advance_language_vamp_run,
    init_language_vamp_run,
    score_parent_nodes,
)
from apm.continual.language_tasks import (
    BaseCheckpointRef,
    LanguageEvaluationExample,
    LanguageTask,
    NodeId,
    RouterBatch,
    TaskId,
    build_prefix_suffix_batches,
)
from apm.lm.checkpoint import parameter_checksum
from apm.lm.config import GptNeoConfig
from apm.lm.gpt_neo import apply_gpt_neo
from apm.lm.lora import LoraConfig, LoraEdge
from apm.lm.lora_memory import (
    PackedLoraMemory,
    edge_coefficients_for_node,
    pack_lora_memory,
)
from apm.lm.parameters import GptNeoParams, init_gpt_neo_params
from apm.lm.text import CharTokenizer
from apm.lm.text_data import TokenBatch
from apm.lm.training import LmTrainConfig
from apm.memory.graph import MemoryGraph, memory_node_ids
from apm.memory.prefix_energy import exhaustive_prefix_nll_address


_TINY_SHAKESPEARE_EXCERPT = (
    "to be, or not to be: that is the question.\n"
    "whether tis nobler in the mind to suffer.\n"
)
_KEY_PROBE_COUNT = 2
_EDGE_STEPS = 3

GraphMetadata = tuple[
    tuple[NodeId, NodeId | None, TaskId | None, int, int],
    ...,
]
GraphSnapshot = tuple[GraphMetadata, tuple[tuple[np.ndarray, ...], ...]]


def _model_config(vocab_size: int) -> GptNeoConfig:
    return GptNeoConfig(
        vocab_size=vocab_size,
        max_position_embeddings=8,
        hidden_size=8,
        intermediate_size=16,
        num_layers=1,
        num_heads=2,
        attention_types=("global",),
        local_window_size=4,
    )


def _permute_letters(text: str, offset: int) -> str:
    letters = tuple(sorted({character for character in text if character.isalpha()}))
    permutation = {
        letter: letters[(index + offset) % len(letters)]
        for index, letter in enumerate(letters)
    }
    return "".join(permutation.get(character, character) for character in text)


def _token_batch(token_ids: tuple[int, ...], start: int) -> TokenBatch:
    transition_tokens = token_ids[start : start + 6]
    mask = np.ones((1, 5), dtype=np.bool_)
    return TokenBatch(
        input_ids=np.asarray((transition_tokens[:-1],), dtype=np.int32),
        attention_mask=mask,
        target_ids=np.asarray((transition_tokens[1:],), dtype=np.int32),
        loss_mask=mask,
    )


def _evaluation_example(
    token_ids: tuple[int, ...],
    start: int,
    task_id: TaskId,
) -> LanguageEvaluationExample:
    router_batch, competence_batch = build_prefix_suffix_batches(
        token_ids[start : start + 6],
        prefix_length=4,
        suffix_length=2,
    )
    return LanguageEvaluationExample(
        router_batch=router_batch,
        competence_batch=competence_batch,
        task_id=task_id,
        oracle_node_id=NodeId(str(task_id)),
    )


def _permutation_task(
    tokenizer: CharTokenizer,
    task_id: TaskId,
    offset: int,
) -> LanguageTask:
    token_ids = tokenizer.encode(
        _permute_letters(_TINY_SHAKESPEARE_EXCERPT, offset)
    )
    return LanguageTask(
        task_id=task_id,
        train_batches=tuple(
            _token_batch(token_ids, start)
            for start in (0, 7)
        ),
        validation_examples=tuple(
            _evaluation_example(token_ids, start, task_id)
            for start in (14, 21)
        ),
        test_examples=tuple(
            _evaluation_example(token_ids, start, task_id)
            for start in (28, 35)
        ),
    )


def _combine_router_batches(batches: tuple[RouterBatch, ...]) -> RouterBatch:
    return RouterBatch(
        input_ids=np.concatenate(tuple(batch.input_ids for batch in batches)),
        attention_mask=np.concatenate(
            tuple(batch.attention_mask for batch in batches)
        ),
        target_ids=np.concatenate(tuple(batch.target_ids for batch in batches)),
        loss_mask=np.concatenate(tuple(batch.loss_mask for batch in batches)),
    )


def _packed_memory(
    run: LanguageVampRun,
    model_config: GptNeoConfig,
    lora_config: LoraConfig,
) -> PackedLoraMemory:
    return pack_lora_memory(
        run.graph,
        model_config,
        lora_config,
        run.max_nodes,
        run.max_edges,
    )


def _hard_node_logits(
    run: LanguageVampRun,
    node_index: int,
    batch: RouterBatch,
    base_params: GptNeoParams,
    model_config: GptNeoConfig,
    lora_config: LoraConfig,
) -> np.ndarray:
    packed_memory = _packed_memory(run, model_config, lora_config)
    return np.asarray(
        apply_gpt_neo(
            base_params,
            model_config,
            jnp.asarray(batch.input_ids),
            jnp.asarray(batch.attention_mask),
            lora_memory=packed_memory,
            edge_coefficients=edge_coefficients_for_node(
                packed_memory,
                node_index,
            ),
            lora_config=lora_config,
        ).logits
    ).copy()


def _graph_snapshot(
    graph: MemoryGraph[LoraEdge],
) -> GraphSnapshot:
    metadata = tuple(
        (
            node.node_id,
            node.parent_id,
            node.trained_task,
            node.train_stage,
            node.depth,
        )
        for node in graph.nodes
    )
    edges = tuple(
        tuple(
            np.asarray(leaf).copy()
            for leaf in jax.tree_util.tree_leaves(node.incoming_edge)
        )
        for node in graph.nodes
        if node.incoming_edge is not None
    )
    return metadata, edges


def _assert_graph_matches_snapshot(
    graph: MemoryGraph[LoraEdge],
    snapshot: GraphSnapshot,
) -> None:
    metadata, edge_snapshots = snapshot
    current_metadata, current_edges = _graph_snapshot(graph)
    assert current_metadata == metadata
    assert len(current_edges) == len(edge_snapshots)
    for current_edge, expected_edge in zip(current_edges, edge_snapshots):
        assert len(current_edge) == len(expected_edge)
        for current_leaf, expected_leaf in zip(current_edge, expected_edge):
            np.testing.assert_array_equal(current_leaf, expected_leaf)


def test_two_task_character_permutation_run_is_immutable_and_task_free() -> None:
    tokenizer = CharTokenizer.from_training_text(_TINY_SHAKESPEARE_EXCERPT)
    task_one = _permutation_task(tokenizer, TaskId("letter-permutation-1"), 1)
    task_two = _permutation_task(tokenizer, TaskId("letter-permutation-2"), 5)
    model_config = _model_config(tokenizer.vocab_size)
    base_params = init_gpt_neo_params(jax.random.PRNGKey(0), model_config)
    base_checksum = parameter_checksum(base_params, model_config)
    checkpoint = BaseCheckpointRef(
        directory=Path("tiny-shakespeare-base"),
        manifest_sha256="a" * 64,
        parameter_checksum=base_checksum,
    )
    lora_config = LoraConfig(rank=1, alpha=1.0)
    train_config = LmTrainConfig(
        learning_rate=5e-2,
        steps=_EDGE_STEPS,
        batch_size=1,
        weight_decay=0.0,
        gradient_clip_norm=1.0,
    )
    root_tokens = tokenizer.encode(_TINY_SHAKESPEARE_EXCERPT)
    root_probes = tuple(
        build_prefix_suffix_batches(
            root_tokens[start : start + 6],
            prefix_length=4,
            suffix_length=2,
        )[0]
        for start in (14, 21)
    )
    initial_run = init_language_vamp_run(
        checkpoint,
        base_params,
        model_config,
        root_probes,
        jax.random.PRNGKey(1),
        max_nodes=3,
        max_edges=2,
        key_probe_count=_KEY_PROBE_COUNT,
    )
    initial_graph_snapshot = _graph_snapshot(initial_run.graph)
    initial_keys = initial_run.address_book.keys.copy()

    first_run = advance_language_vamp_run(
        initial_run,
        task_one,
        base_params,
        model_config,
        lora_config,
        train_config,
        score_parent_nodes(
            initial_run,
            task_one.parent_probes,
            base_params,
            model_config,
            lora_config,
        ),
        key_probe_count=_KEY_PROBE_COUNT,
    )
    first_graph_snapshot = _graph_snapshot(first_run.graph)
    first_keys = first_run.address_book.keys.copy()
    first_mask = first_run.address_book.valid_node_mask.copy()
    first_rng_key = np.asarray(first_run.rng_key).copy()
    first_stage_metrics = first_run.stage_metrics
    committed_probe = task_one.test_examples[0].router_batch
    committed_logits = tuple(
        _hard_node_logits(
            first_run,
            node_index,
            committed_probe,
            base_params,
            model_config,
            lora_config,
        )
        for node_index in (0, 1)
    )

    second_run = advance_language_vamp_run(
        first_run,
        task_two,
        base_params,
        model_config,
        lora_config,
        train_config,
        score_parent_nodes(
            first_run,
            task_two.parent_probes,
            base_params,
            model_config,
            lora_config,
        ),
        key_probe_count=_KEY_PROBE_COUNT,
    )

    assert memory_node_ids(initial_run.graph) == (NodeId("root"),)
    _assert_graph_matches_snapshot(initial_run.graph, initial_graph_snapshot)
    np.testing.assert_array_equal(initial_run.address_book.keys, initial_keys)
    assert tuple(task.task_id for task in initial_run.completed_tasks) == ()
    assert initial_run.stage_metrics == ()
    assert memory_node_ids(first_run.graph) == (
        NodeId("root"),
        NodeId(str(task_one.task_id)),
    )
    _assert_graph_matches_snapshot(first_run.graph, first_graph_snapshot)
    np.testing.assert_array_equal(first_run.address_book.keys, first_keys)
    np.testing.assert_array_equal(first_run.address_book.valid_node_mask, first_mask)
    np.testing.assert_array_equal(first_run.rng_key, first_rng_key)
    assert first_run.stage_metrics is first_stage_metrics
    assert tuple(task.task_id for task in first_run.completed_tasks) == (
        task_one.task_id,
    )
    assert memory_node_ids(second_run.graph) == (
        NodeId("root"),
        NodeId(str(task_one.task_id)),
        NodeId(str(task_two.task_id)),
    )
    assert tuple(task.task_id for task in second_run.completed_tasks) == (
        task_one.task_id,
        task_two.task_id,
    )

    assert parameter_checksum(base_params, model_config) == base_checksum
    assert all(
        run.base_checkpoint.parameter_checksum == base_checksum
        for run in (initial_run, first_run, second_run)
    )
    for node_index, expected_logits in enumerate(committed_logits):
        np.testing.assert_array_equal(
            _hard_node_logits(
                second_run,
                node_index,
                committed_probe,
                base_params,
                model_config,
                lora_config,
            ),
            expected_logits,
        )

    assert len(second_run.stage_metrics) == 2
    first_stage, second_stage = second_run.stage_metrics
    assert first_stage.stage_index == 1
    assert first_stage.parent_node_index == 0
    assert first_stage.parent_node_id == NodeId("root")
    assert second_stage.stage_index == 2
    assert second_stage.parent_node_index in (0, 1)
    assert second_stage.parent_node_id == second_run.graph.nodes[
        second_stage.parent_node_index
    ].node_id
    assert len(first_stage.candidate_step_losses) == _EDGE_STEPS
    assert len(second_stage.candidate_step_losses) == _EDGE_STEPS
    assert np.all(np.isfinite(first_stage.candidate_step_losses))
    assert np.all(np.isfinite(second_stage.candidate_step_losses))
    assert np.all(np.isfinite(first_stage.parent_mean_node_nll[:1]))
    assert np.all(np.isinf(first_stage.parent_mean_node_nll[1:]))
    assert np.all(np.isfinite(second_stage.parent_mean_node_nll[:2]))
    assert np.isinf(second_stage.parent_mean_node_nll[2])

    assert tuple(metric.task_id for metric in second_stage.task_metrics) == (
        task_one.task_id,
        task_two.task_id,
    )
    valid_node_ids = set(memory_node_ids(second_run.graph))
    reported_test_selections = tuple(
        example_metric.selected_node_id
        for task_metric in second_stage.task_metrics
        for example_metric in task_metric.example_metrics
    )
    assert len(reported_test_selections) == 4
    assert set(reported_test_selections).issubset(valid_node_ids)
    assert all(
        len(task_metric.example_metrics) == 2
        and task_metric.valid_suffix_tokens == 4
        and np.isfinite(task_metric.oracle_competence_nll)
        and np.isfinite(task_metric.task_free_competence_nll)
        for task_metric in second_stage.task_metrics
    )

    packed_second_run = _packed_memory(second_run, model_config, lora_config)
    mixed_test_batches = tuple(
        example.router_batch
        for task in (task_one, task_two)
        for example in task.test_examples
    )
    independent_test_routing = exhaustive_prefix_nll_address(
        base_params,
        model_config,
        packed_second_run,
        lora_config,
        _combine_router_batches(mixed_test_batches),
    )
    independently_selected_ids = tuple(
        second_run.graph.nodes[int(index)].node_id
        for index in np.asarray(independent_test_routing.selected_indices)
    )
    assert independently_selected_ids == reported_test_selections

    mixed_validation_batches = tuple(
        example.router_batch
        for task in (task_one, task_two)
        for example in task.validation_examples
    )
    mixed_validation_routing = exhaustive_prefix_nll_address(
        base_params,
        model_config,
        packed_second_run,
        lora_config,
        _combine_router_batches(mixed_validation_batches),
    )
    assert mixed_validation_routing.selected_indices.shape == (4,)
    assert mixed_validation_routing.node_scores.shape == (4, 3)
    assert np.all(np.isfinite(np.asarray(mixed_validation_routing.node_scores)))
    np.testing.assert_allclose(
        np.sum(np.asarray(mixed_validation_routing.node_probabilities), axis=-1),
        1.0,
        rtol=1e-6,
        atol=1e-6,
    )

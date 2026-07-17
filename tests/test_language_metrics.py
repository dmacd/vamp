from __future__ import annotations

from dataclasses import fields

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from apm.continual.language_metrics import (
    LanguageExampleMetric,
    LanguageTaskMetrics,
    aggregate_language_task_metrics,
    evaluate_language_example,
    hard_node_competence_nll,
    resolve_node_index,
)
import apm.continual.language_metrics as language_metrics_module
from apm.continual.language_tasks import (
    AddressResult,
    CompetenceBatch,
    LanguageEvaluationExample,
    NodeId,
    RouterBatch,
    TaskId,
)
from apm.lm.config import GptNeoConfig
from apm.lm.gpt_neo import ForwardResult
from apm.lm.lora import LoraConfig, LoraEdge, init_lora_edge
from apm.lm.lora_memory import PackedLoraMemory, pack_lora_memory
from apm.lm.losses import per_token_nll
from apm.lm.parameters import GptNeoParams, init_gpt_neo_params
from apm.memory import prefix_energy
from apm.memory.graph import MemoryGraph, add_memory_node, init_memory_graph


def _model_config() -> GptNeoConfig:
    return GptNeoConfig(
        vocab_size=3,
        max_position_embeddings=8,
        hidden_size=4,
        intermediate_size=8,
        num_layers=1,
        num_heads=2,
        attention_types=("global",),
        local_window_size=2,
    )


def _lora_config() -> LoraConfig:
    return LoraConfig(rank=2, alpha=2.0)


def _params() -> GptNeoParams:
    return init_gpt_neo_params(jax.random.PRNGKey(0), _model_config())


def _graph(node_count: int = 3) -> MemoryGraph[LoraEdge]:
    graph: MemoryGraph[LoraEdge] = init_memory_graph(NodeId("root"))
    for node_number in range(1, node_count):
        graph = add_memory_node(
            graph,
            NodeId(f"node-{node_number}"),
            NodeId("root"),
            TaskId(f"task-{node_number}"),
            node_number,
            init_lora_edge(
                jax.random.PRNGKey(node_number),
                _model_config(),
                _lora_config(),
            ),
        )
    return graph


def _packed_memory(
    graph: MemoryGraph[LoraEdge] | None = None,
    max_nodes: int = 4,
) -> PackedLoraMemory:
    return pack_lora_memory(
        _graph() if graph is None else graph,
        _model_config(),
        _lora_config(),
        max_nodes=max_nodes,
        max_edges=max_nodes - 1,
    )


def _router_batch(desired_node: int) -> RouterBatch:
    input_ids = np.asarray(((desired_node, 0, 0),), dtype=np.int32)
    mask = np.ones_like(input_ids, dtype=np.bool_)
    return RouterBatch(
        input_ids=input_ids,
        attention_mask=mask,
        target_ids=np.zeros_like(input_ids),
        loss_mask=mask,
    )


def _competence_batch(oracle_node: int = 1) -> CompetenceBatch:
    return CompetenceBatch(
        input_ids=np.asarray(((oracle_node, 0, 0, 0, 2),), dtype=np.int32),
        attention_mask=np.asarray(((True, True, True, True, False),)),
        target_ids=np.asarray(((0, 0, 0, 0, 2),), dtype=np.int32),
        loss_mask=np.asarray(((False, False, True, True, False),)),
    )


def _example(
    *,
    desired_node: int,
    oracle_node: int = 1,
) -> LanguageEvaluationExample:
    router_batch = _router_batch(desired_node)
    return LanguageEvaluationExample(
        router_batch=router_batch,
        competence_batch=CompetenceBatch(
            input_ids=np.asarray(
                ((desired_node, 0, 0, oracle_node, 2),),
                dtype=np.int32,
            ),
            attention_mask=np.asarray(((True, True, True, True, False),)),
            target_ids=np.asarray(((0, 0, 0, 0, 2),), dtype=np.int32),
            loss_mask=np.asarray(((False, False, False, True, False),)),
        ),
        task_id=TaskId("evaluator-only-task"),
        oracle_node_id=NodeId("root" if oracle_node == 0 else f"node-{oracle_node}"),
    )


def _install_path_sensitive_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def apply_path_sensitive_model(
        params: GptNeoParams,
        config: GptNeoConfig,
        token_ids: jax.Array,
        attention_mask: jax.Array,
        *,
        position_ids: jax.Array | None = None,
        lora_memory: PackedLoraMemory | None = None,
        edge_coefficients: jax.Array | None = None,
        lora_config: LoraConfig | None = None,
        capture: object = None,
        training: bool = False,
        rng_key: jax.Array | None = None,
    ) -> ForwardResult:
        del params, attention_mask, position_ids, lora_config, capture, training, rng_key
        assert lora_memory is not None
        assert edge_coefficients is not None
        edge_numbers = jnp.arange(
            1,
            edge_coefficients.shape[0] + 1,
            dtype=jnp.float32,
        )
        routed_node = jnp.sum(edge_coefficients * edge_numbers)
        desired_node = token_ids[:, 0 if token_ids.shape[1] == 3 else 3].astype(
            jnp.float32
        )
        target_logit = -3.0 * jnp.abs(desired_node - routed_node)
        logits = jnp.zeros(
            (*token_ids.shape, config.vocab_size),
            dtype=jnp.float32,
        ).at[:, :, 0].set(target_logit[:, None])
        return ForwardResult(
            logits=logits,
            final_hidden=jnp.zeros(
                (*token_ids.shape, config.hidden_size),
                dtype=jnp.float32,
            ),
            captured_hidden=(),
        )

    monkeypatch.setattr(prefix_energy, "apply_gpt_neo", apply_path_sensitive_model)
    monkeypatch.setattr(
        language_metrics_module,
        "apply_gpt_neo",
        apply_path_sensitive_model,
    )


def test_metric_records_have_the_stage_contract_fields() -> None:
    assert tuple(field.name for field in fields(LanguageExampleMetric)) == (
        "oracle_node_id",
        "selected_node_id",
        "oracle_competence_nll",
        "task_free_competence_nll",
        "routing_correct",
        "routing_regret",
        "valid_suffix_tokens",
    )
    assert tuple(field.name for field in fields(LanguageTaskMetrics)) == (
        "task_id",
        "example_metrics",
        "oracle_competence_nll",
        "task_free_competence_nll",
        "routing_accuracy",
        "routing_regret",
        "valid_suffix_tokens",
    )


def test_hard_node_uses_exact_path_and_suffix_only_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packed_memory = _packed_memory()
    captured_coefficients: list[np.ndarray] = []

    def apply_position_sensitive_model(
        params: GptNeoParams,
        config: GptNeoConfig,
        token_ids: jax.Array,
        attention_mask: jax.Array,
        *,
        lora_memory: PackedLoraMemory,
        edge_coefficients: jax.Array,
        lora_config: LoraConfig,
        training: bool,
    ) -> ForwardResult:
        del params, attention_mask, lora_memory, lora_config, training
        captured_coefficients.append(np.asarray(edge_coefficients))
        target_logits = jnp.asarray((-100.0, -50.0, 1.0, 2.0, -100.0))
        logits = jnp.zeros((*token_ids.shape, config.vocab_size), dtype=jnp.float32)
        logits = logits.at[0, :, 0].set(target_logits)
        return ForwardResult(
            logits=logits,
            final_hidden=jnp.zeros((*token_ids.shape, config.hidden_size)),
            captured_hidden=(),
        )

    monkeypatch.setattr(
        language_metrics_module,
        "apply_gpt_neo",
        apply_position_sensitive_model,
    )
    competence_batch = _competence_batch(oracle_node=2)

    nll = hard_node_competence_nll(
        _params(),
        _model_config(),
        packed_memory,
        _lora_config(),
        competence_batch,
        node_index=2,
    )
    expected_losses = per_token_nll(
        jnp.zeros((1, 5, 3), dtype=jnp.float32).at[0, :, 0].set(
            jnp.asarray((-100.0, -50.0, 1.0, 2.0, -100.0))
        ),
        jnp.asarray(competence_batch.target_ids),
    )
    expected = np.mean(np.asarray(expected_losses)[0, 2:4])

    np.testing.assert_array_equal(
        captured_coefficients,
        np.asarray((packed_memory.node_path_matrix[2],)),
    )
    np.testing.assert_allclose(nll, expected, rtol=1e-6, atol=1e-6)


def test_task_free_prefix_selection_drives_competence_and_regret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_path_sensitive_model(monkeypatch)

    metric = evaluate_language_example(
        _params(),
        _model_config(),
        _graph(),
        _packed_memory(),
        _lora_config(),
        _example(desired_node=2, oracle_node=1),
    )

    assert metric.oracle_node_id == NodeId("node-1")
    assert metric.selected_node_id == NodeId("node-2")
    assert not metric.routing_correct
    assert metric.task_free_competence_nll > metric.oracle_competence_nll
    assert metric.routing_regret == pytest.approx(
        metric.task_free_competence_nll - metric.oracle_competence_nll
    )
    assert metric.valid_suffix_tokens == 1


def test_evaluator_metadata_never_enters_the_router(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    example = _example(desired_node=2, oracle_node=1)
    routed_batches: list[RouterBatch] = []
    scored_nodes: list[int] = []

    def route_prefix_only(
        base_params: GptNeoParams,
        model_config: GptNeoConfig,
        packed_memory: PackedLoraMemory,
        lora_config: LoraConfig,
        prefix_batch: RouterBatch,
        *,
        evaluation_microbatch_size: int | None = None,
    ) -> AddressResult:
        del (
            base_params,
            model_config,
            packed_memory,
            lora_config,
            evaluation_microbatch_size,
        )
        routed_batches.append(prefix_batch)
        assert set(prefix_batch.__dataclass_fields__) == {
            "input_ids",
            "attention_mask",
            "target_ids",
            "loss_mask",
        }
        return AddressResult(
            selected_indices=jnp.asarray((2,), dtype=jnp.int32),
            node_probabilities=jnp.asarray(((0.1, 0.2, 0.7, 0.0),)),
            node_scores=jnp.asarray(((2.0, 1.0, 0.0, jnp.inf),)),
            score_margin=jnp.asarray((1.0,)),
            entropy=jnp.asarray((0.8,)),
        )

    def score_competence(
        base_params: GptNeoParams,
        model_config: GptNeoConfig,
        packed_memory: PackedLoraMemory,
        lora_config: LoraConfig,
        competence_batch: CompetenceBatch,
        node_index: int,
    ) -> jax.Array:
        del base_params, model_config, packed_memory, lora_config
        assert competence_batch is example.competence_batch
        scored_nodes.append(node_index)
        return jnp.asarray(0.25 + node_index, dtype=jnp.float32)

    monkeypatch.setattr(
        language_metrics_module,
        "exhaustive_prefix_nll_address",
        route_prefix_only,
    )
    monkeypatch.setattr(
        language_metrics_module,
        "hard_node_competence_nll",
        score_competence,
    )

    metric = evaluate_language_example(
        _params(),
        _model_config(),
        _graph(),
        _packed_memory(),
        _lora_config(),
        example,
    )

    assert routed_batches == [example.router_batch]
    assert scored_nodes == [1, 2]
    assert metric.oracle_node_id == NodeId("node-1")
    assert metric.selected_node_id == NodeId("node-2")


def test_old_path_evaluation_is_stable_after_graph_growth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_path_sensitive_model(monkeypatch)
    original_graph = _graph(node_count=2)
    grown_graph = _graph(node_count=3)
    example = _example(desired_node=1, oracle_node=1)

    before = evaluate_language_example(
        _params(),
        _model_config(),
        original_graph,
        _packed_memory(original_graph),
        _lora_config(),
        example,
    )
    after = evaluate_language_example(
        _params(),
        _model_config(),
        grown_graph,
        _packed_memory(grown_graph),
        _lora_config(),
        example,
    )

    assert before == after


def test_node_resolution_uses_graph_insertion_order() -> None:
    graph = _graph()

    assert resolve_node_index(graph, NodeId("root")) == 0
    assert resolve_node_index(graph, NodeId("node-2")) == 2
    with pytest.raises(KeyError, match="unknown"):
        resolve_node_index(graph, NodeId("missing"))


def test_task_aggregation_is_token_weighted_with_per_example_accuracy() -> None:
    metrics = (
        LanguageExampleMetric(
            NodeId("root"),
            NodeId("root"),
            1.0,
            1.0,
            True,
            0.0,
            1,
        ),
        LanguageExampleMetric(
            NodeId("node-1"),
            NodeId("node-2"),
            3.0,
            5.0,
            False,
            2.0,
            3,
        ),
    )

    aggregate = aggregate_language_task_metrics(TaskId("task"), metrics)

    assert aggregate.example_metrics is metrics
    assert aggregate.oracle_competence_nll == pytest.approx(2.5)
    assert aggregate.task_free_competence_nll == pytest.approx(4.0)
    assert aggregate.routing_accuracy == pytest.approx(0.5)
    assert aggregate.routing_regret == pytest.approx(1.5)
    assert aggregate.valid_suffix_tokens == 4

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import apm.continual.language_routing as language_routing_module
import apm.memory.address_refinement as address_refinement_module
from apm.continual.language_routing import (
    LANGUAGE_ROUTER_TOP_K,
    competence_nll_by_node,
    evaluate_language_router,
    route_language_prefix,
)
from apm.continual.language_tasks import (
    AddressBook,
    RouterBatch,
    LanguageEvaluationExample,
    NodeId,
    TaskId,
    build_prefix_suffix_batches,
)
from apm.lm.config import GptNeoConfig
from apm.lm.lora import LoraConfig, init_lora_edge
from apm.lm.lora_memory import pack_lora_memory
from apm.lm.parameters import init_gpt_neo_params
from apm.memory.address_refinement import EbtAddressResult, EbtConfig
from apm.memory.graph import add_memory_node, init_memory_graph


def _setup():
    model_config = GptNeoConfig(
        vocab_size=10,
        max_position_embeddings=8,
        hidden_size=8,
        intermediate_size=16,
        num_layers=1,
        num_heads=2,
        attention_types=("global",),
        local_window_size=4,
    )
    lora_config = LoraConfig(rank=1, alpha=1.0)
    base_params = init_gpt_neo_params(jax.random.PRNGKey(0), model_config)
    graph = init_memory_graph(NodeId("root"))
    for index, node_name in enumerate(
        ("task_0", "aux_1", "aux_2", "aux_3"),
        start=1,
    ):
        graph = add_memory_node(
            graph,
            NodeId(node_name),
            NodeId("root"),
            TaskId(node_name),
            index,
            init_lora_edge(
                jax.random.PRNGKey(index),
                model_config,
                lora_config,
            ),
        )
    packed_memory = pack_lora_memory(
        graph,
        model_config,
        lora_config,
        max_nodes=6,
        max_edges=5,
    )
    keys = np.zeros((6, 8), dtype=np.float32)
    keys[:5] = np.eye(8, dtype=np.float32)[:5]
    address_book = AddressBook(
        node_ids=(
            NodeId("root"),
            NodeId("task_0"),
            NodeId("aux_1"),
            NodeId("aux_2"),
            NodeId("aux_3"),
            None,
        ),
        keys=keys,
        valid_node_mask=np.asarray((True, True, True, True, True, False)),
    )
    examples = tuple(
        LanguageEvaluationExample(
            router_batch=router_batch,
            competence_batch=competence_batch,
            task_id=TaskId("task_0"),
            oracle_node_id=NodeId("task_0"),
        )
        for sequence in ((2, 3, 4, 5, 6, 7), (2, 3, 4, 5, 7, 6))
        for router_batch, competence_batch in (
            build_prefix_suffix_batches(sequence, 4, 2),
        )
    )
    return (
        base_params,
        model_config,
        graph,
        packed_memory,
        lora_config,
        address_book,
        examples,
    )


def test_every_router_returns_valid_task_free_decisions_and_suffix_metrics() -> None:
    setup = _setup()
    routers = (
        "vamp_exhaustive",
        "vamp_hopfield",
        "vamp_ebt_uniform",
        "vamp_ebt_hopfield",
        "deterministic_random_node",
    )
    results = tuple(
        evaluate_language_router(
            router,
            *setup,
            ebt_config=EbtConfig(steps=2, learning_rate=0.1),
        )
        for router in routers
    )

    for result in results:
        probabilities = np.asarray(result.decision.node_probabilities)
        assert probabilities.shape == (2, 6)
        np.testing.assert_allclose(np.sum(probabilities, axis=1), 1.0)
        np.testing.assert_array_equal(probabilities[:, 5], 0.0)
        assert np.all(np.isfinite(result.suffix_nll_by_node[:, :5]))
        assert np.all(np.isinf(result.suffix_nll_by_node[:, 5]))
        scores = np.asarray(result.decision.node_scores)
        np.testing.assert_array_equal(scores[:, 5], -np.inf)
        np.testing.assert_array_equal(
            np.argmax(scores, axis=1),
            result.decision.selected_indices,
        )
        top_k = np.asarray(result.decision.top_k_indices)
        assert top_k.shape == (2, LANGUAGE_ROUTER_TOP_K)
        np.testing.assert_array_equal(
            top_k[:, 0],
            result.decision.selected_indices,
        )
        assert np.all(top_k < 5)
        assert all(len(set(row.tolist())) == LANGUAGE_ROUTER_TOP_K for row in top_k)
        assert len(result.examples) == 2
        assert int(np.sum(result.confusion_counts)) == 2
        assert all(example.task_oracle_index == 1 for example in result.examples)


@pytest.mark.parametrize(
    "router",
    (
        "vamp_exhaustive",
        "vamp_hopfield",
        "vamp_ebt_uniform",
        "vamp_ebt_hopfield",
        "deterministic_random_node",
    ),
)
def test_router_microbatching_preserves_every_decision_field(router: str) -> None:
    setup = _setup()
    prefix = RouterBatch(
        input_ids=np.concatenate(
            tuple(example.router_batch.input_ids for example in setup[6])
        ),
        attention_mask=np.concatenate(
            tuple(example.router_batch.attention_mask for example in setup[6])
        ),
        target_ids=np.concatenate(
            tuple(example.router_batch.target_ids for example in setup[6])
        ),
        loss_mask=np.concatenate(
            tuple(example.router_batch.loss_mask for example in setup[6])
        ),
    )
    kwargs = {
        "random_seed": 7,
        "ebt_config": EbtConfig(steps=2, learning_rate=0.1),
    }
    expected = route_language_prefix(
        router,
        *setup[:2],
        setup[3],
        setup[4],
        setup[5],
        prefix,
        **kwargs,
    )
    actual = route_language_prefix(
        router,
        *setup[:2],
        setup[3],
        setup[4],
        setup[5],
        prefix,
        evaluation_microbatch_size=1,
        **kwargs,
    )

    for expected_field, actual_field in zip(expected, actual):
        np.testing.assert_allclose(
            actual_field,
            expected_field,
            rtol=1e-5,
            atol=1e-5,
        )


def test_ebt_microbatches_and_repeated_calls_reuse_optimizer_compilation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = _setup()
    prefix = RouterBatch(
        input_ids=np.concatenate(
            tuple(example.router_batch.input_ids for example in setup[6])
        ),
        attention_mask=np.concatenate(
            tuple(example.router_batch.attention_mask for example in setup[6])
        ),
        target_ids=np.concatenate(
            tuple(example.router_batch.target_ids for example in setup[6])
        ),
        loss_mask=np.concatenate(
            tuple(example.router_batch.loss_mask for example in setup[6])
        ),
    )
    original_adam = address_refinement_module.optax.adam
    optimizer_constructions = 0

    def counted_adam(learning_rate: float):
        nonlocal optimizer_constructions
        optimizer_constructions += 1
        return original_adam(learning_rate)

    address_refinement_module._optimize_node_logits.clear_cache()
    monkeypatch.setattr(address_refinement_module.optax, "adam", counted_adam)
    config = EbtConfig(steps=2, learning_rate=0.1)
    results = tuple(
        route_language_prefix(
            "vamp_ebt_uniform",
            *setup[:2],
            setup[3],
            setup[4],
            setup[5],
            prefix,
            ebt_config=config,
            evaluation_microbatch_size=1,
        )
        for _ in range(2)
    )
    jax.block_until_ready(results[-1].node_probabilities)

    assert optimizer_constructions == 1
    for first_field, second_field in zip(*results):
        np.testing.assert_array_equal(first_field, second_field)


def test_competence_microbatching_and_router_suffix_reuse_are_equivalent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = _setup()
    competence_batch = language_routing_module._stack_competence_batches(
        tuple(example.competence_batch for example in setup[6])
    )
    expected = competence_nll_by_node(
        setup[0],
        setup[1],
        setup[3],
        setup[4],
        competence_batch,
    )
    original_apply = language_routing_module.apply_gpt_neo
    observed_batch_sizes: list[int] = []

    def counted_apply(*args, **kwargs):
        observed_batch_sizes.append(int(args[2].shape[0]))
        return original_apply(*args, **kwargs)

    monkeypatch.setattr(
        language_routing_module,
        "apply_gpt_neo",
        counted_apply,
    )
    microbatched = competence_nll_by_node(
        setup[0],
        setup[1],
        setup[3],
        setup[4],
        competence_batch,
        evaluation_microbatch_size=1,
    )
    np.testing.assert_allclose(microbatched, expected, rtol=1e-6, atol=1e-6)
    assert observed_batch_sizes
    assert max(observed_batch_sizes) == 1

    def reject_duplicate_scoring(*args, **kwargs):
        del args, kwargs
        raise AssertionError("precomputed suffix NLL should be reused")

    monkeypatch.setattr(
        language_routing_module,
        "competence_nll_by_node",
        reject_duplicate_scoring,
    )
    reused = evaluate_language_router(
        "deterministic_random_node",
        *setup,
        evaluation_microbatch_size=1,
        suffix_nll_by_node=microbatched,
    )

    np.testing.assert_array_equal(reused.suffix_nll_by_node, microbatched)


def test_evaluation_microbatch_and_reused_suffix_values_are_validated() -> None:
    setup = _setup()

    with pytest.raises(ValueError, match="positive integer"):
        evaluate_language_router(
            "deterministic_random_node",
            *setup,
            evaluation_microbatch_size=0,
        )
    with pytest.raises(ValueError, match="must have shape"):
        evaluate_language_router(
            "deterministic_random_node",
            *setup,
            suffix_nll_by_node=np.zeros((1, 6), dtype=np.float32),
        )


def test_random_router_is_repeatable_and_router_cannot_observe_changed_suffix() -> None:
    (
        base_params,
        model_config,
        _,
        packed_memory,
        lora_config,
        address_book,
        examples,
    ) = _setup()
    prefix = type(examples[0].router_batch)(
        input_ids=np.concatenate(tuple(example.router_batch.input_ids for example in examples)),
        attention_mask=np.concatenate(
            tuple(example.router_batch.attention_mask for example in examples)
        ),
        target_ids=np.concatenate(tuple(example.router_batch.target_ids for example in examples)),
        loss_mask=np.concatenate(tuple(example.router_batch.loss_mask for example in examples)),
    )
    assert np.array_equal(
        examples[0].router_batch.input_ids,
        examples[1].router_batch.input_ids,
    )
    first = route_language_prefix(
        "deterministic_random_node",
        base_params,
        model_config,
        packed_memory,
        lora_config,
        address_book,
        prefix,
        random_seed=9,
    )
    second = route_language_prefix(
        "deterministic_random_node",
        base_params,
        model_config,
        packed_memory,
        lora_config,
        address_book,
        prefix,
        random_seed=9,
    )
    np.testing.assert_array_equal(first.selected_indices, second.selected_indices)
    assert int(first.selected_indices[0]) == int(first.selected_indices[1])
    assert np.all(jnp.isfinite(first.entropy))


@pytest.mark.parametrize(
    "router",
    (
        "vamp_exhaustive",
        "vamp_hopfield",
        "vamp_ebt_uniform",
        "vamp_ebt_hopfield",
        "deterministic_random_node",
    ),
)
def test_every_router_rejects_suffix_bearing_competence_batch(router: str) -> None:
    (
        base_params,
        model_config,
        _,
        packed_memory,
        lora_config,
        address_book,
        examples,
    ) = _setup()

    with pytest.raises(TypeError, match="RouterBatch prefix data only"):
        route_language_prefix(
            router,
            base_params,
            model_config,
            packed_memory,
            lora_config,
            address_book,
            examples[0].competence_batch,
        )


def test_suffix_matrix_rejects_router_batch() -> None:
    base_params, model_config, _, packed, lora_config, _, examples = _setup()

    with pytest.raises(TypeError, match="CompetenceBatch"):
        competence_nll_by_node(
            base_params,
            model_config,
            packed,
            lora_config,
            examples[0].router_batch,
        )


def test_random_router_ignores_values_outside_active_prefix() -> None:
    base_params, model_config, _, packed, lora_config, address_book, _ = _setup()
    attention_mask = np.asarray(
        ((True, True, True, False, False),) * 2,
        dtype=np.bool_,
    )
    prefix = RouterBatch(
        input_ids=np.asarray(((1, 2, 3, 0, 0), (1, 2, 3, 8, 9))),
        attention_mask=attention_mask,
        target_ids=np.asarray(((2, 3, 4, 0, 0), (2, 3, 4, 9, 8))),
        loss_mask=attention_mask,
    )

    for seed in range(10):
        decision = route_language_prefix(
            "deterministic_random_node",
            base_params,
            model_config,
            packed,
            lora_config,
            address_book,
            prefix,
            random_seed=seed,
        )
        assert int(decision.selected_indices[0]) == int(
            decision.selected_indices[1]
        )
        np.testing.assert_array_equal(
            decision.top_k_indices[0],
            decision.top_k_indices[1],
        )


@pytest.mark.parametrize(
    "router",
    (
        "vamp_exhaustive",
        "vamp_hopfield",
        "vamp_ebt_uniform",
        "vamp_ebt_hopfield",
        "deterministic_random_node",
    ),
)
def test_every_router_rejects_an_empty_prefix_row(router: str) -> None:
    base_params, model_config, _, packed, lora_config, address_book, _ = _setup()
    empty = np.zeros((1, 3), dtype=np.bool_)
    prefix = RouterBatch(
        input_ids=np.zeros((1, 3), dtype=np.int32),
        attention_mask=empty,
        target_ids=np.zeros((1, 3), dtype=np.int32),
        loss_mask=empty,
    )

    with pytest.raises(ValueError, match="active prefix"):
        route_language_prefix(
            router,
            base_params,
            model_config,
            packed,
            lora_config,
            address_book,
            prefix,
        )


def test_ebt_padding_logits_cannot_enter_scores_or_top_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_params, model_config, _, packed, lora_config, address_book, examples = _setup()

    def fake_refinement(*args: object, **kwargs: object) -> EbtAddressResult:
        del args, kwargs
        return EbtAddressResult(
            final_node_logits=jnp.asarray(((-1.0, -2.0, -3.0, -4.0, -5.0, 999.0),)),
            node_probabilities=jnp.asarray(((0.4, 0.3, 0.2, 0.1, 0.0, 0.0),)),
            edge_coefficients=jnp.zeros((1, 5), dtype=jnp.float32),
            selected_indices=jnp.asarray((0,), dtype=jnp.int32),
            soft_mixture_nll=jnp.zeros((1,), dtype=jnp.float32),
            hard_node_nll=jnp.zeros((1,), dtype=jnp.float32),
            objective_trace=jnp.zeros((2, 1), dtype=jnp.float32),
        )

    monkeypatch.setattr(
        "apm.continual.language_routing.refine_ebt_address",
        fake_refinement,
    )
    decision = route_language_prefix(
        "vamp_ebt_uniform",
        base_params,
        model_config,
        packed,
        lora_config,
        address_book,
        examples[0].router_batch,
    )

    assert np.isneginf(decision.node_scores[0, 5])
    assert 5 not in np.asarray(decision.top_k_indices[0]).tolist()


def test_evaluator_rejects_address_book_order_that_disagrees_with_graph() -> None:
    setup = _setup()
    address_book = setup[5]
    swapped = AddressBook(
        node_ids=(
            NodeId("task_0"),
            NodeId("root"),
        )
        + address_book.node_ids[2:],
        keys=address_book.keys,
        valid_node_mask=address_book.valid_node_mask,
    )

    with pytest.raises(ValueError, match="node order"):
        evaluate_language_router(
            "deterministic_random_node",
            *setup[:5],
            swapped,
            setup[6],
        )


def test_random_root_only_margin_is_infinite() -> None:
    base_params, model_config, _, _, lora_config, _, examples = _setup()
    graph = init_memory_graph(NodeId("root"))
    packed = pack_lora_memory(graph, model_config, lora_config, 2, 1)
    key = np.zeros((2, model_config.hidden_size), dtype=np.float32)
    key[0, 0] = 1.0
    address_book = AddressBook(
        node_ids=(NodeId("root"), None),
        keys=key,
        valid_node_mask=np.asarray((True, False)),
    )

    decision = route_language_prefix(
        "deterministic_random_node",
        base_params,
        model_config,
        packed,
        lora_config,
        address_book,
        examples[0].router_batch,
    )

    assert np.isposinf(decision.score_margin[0])
    assert decision.top_k_indices.shape == (1, 1)


def test_random_negative_control_selects_four_task_nodes_and_ranks_root_last() -> None:
    base_params, model_config, _, packed, lora_config, address_book, examples = _setup()
    selected_nodes: set[int] = set()

    for seed in range(64):
        decision = route_language_prefix(
            "deterministic_random_node",
            base_params,
            model_config,
            packed,
            lora_config,
            address_book,
            examples[0].router_batch,
            random_seed=seed,
        )
        selected_nodes.add(int(decision.selected_indices[0]))
        assert int(decision.selected_indices[0]) in (1, 2, 3, 4)
        assert 0 not in np.asarray(decision.top_k_indices[0]).tolist()
        assert float(decision.node_scores[0, 0]) < float(
            np.min(np.asarray(decision.node_scores[0, 1:5]))
        )

    assert selected_nodes == {1, 2, 3, 4}

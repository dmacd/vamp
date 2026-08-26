from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from apm.continual.addressing_first import (
    AFHyperparameters,
    collapse_leaf_pair,
    current_depth_cap,
    init_af_state,
    initialize_split_children,
    install_split,
    predict_for_node,
    route,
    structural_action,
    update_microbatch,
    validate_af_state,
)
from apm.continual.addressing_first import StoredExampleTable
from apm.continual.top_two_adapter import top_two_base_state


def _base():
    return top_two_base_state(
        torch.tensor([[0.4, -0.2], [0.1, 0.3], [-0.5, 0.2]], dtype=torch.float32),
        torch.tensor([0.1, 0.0, -0.1], dtype=torch.float32),
        torch.tensor([[0.3, -0.4, 0.2], [-0.2, 0.5, 0.1]], dtype=torch.float32),
        torch.tensor([0.05, -0.05], dtype=torch.float32),
    )


def _table(rows: int = 8) -> StoredExampleTable:
    embeddings = torch.stack(
        (torch.linspace(-2.0, 2.0, rows), torch.arange(rows, dtype=torch.float32) / rows),
        dim=1,
    )
    labels = (torch.arange(rows) % 2).to(torch.int64)
    return StoredExampleTable(
        embeddings,
        embeddings.clone(),
        torch.zeros((rows, 2), dtype=torch.float32),
        labels,
        torch.arange(rows, dtype=torch.int64) % 2,
        torch.arange(rows, dtype=torch.int64),
    )


def _split_state(train_children: bool = False):
    table = _table()
    hyperparameters = AFHyperparameters(leaf_capacity=4, split_fit_samples=8, batch_size=4)
    state = init_af_state(_base())
    state = update_microbatch(state, table, tuple(range(8)), hyperparameters, 0).state
    split, event = install_split(state, table, 0, 8, hyperparameters, 0)
    return (
        initialize_split_children(split, table, event, hyperparameters, 0)
        if train_children
        else split,
        event,
        table,
        hyperparameters,
    )


def test_deterministic_routing_and_exhaustive_disjoint_split() -> None:
    state, event, table, _hyperparameters = _split_state()
    left = set(event.left_examples)
    right = set(event.right_examples)
    assert left and right and not left & right and left | right == set(range(8))
    assert tuple(route(state, table.embeddings[index]).leaf_id for index in range(8)) == tuple(
        route(state, table.embeddings[index]).leaf_id for index in range(8)
    )
    assert {index for index in range(8) if route(state, table.embeddings[index]).leaf_id == event.left_id} == left
    validate_af_state(state, range(8))


def test_zero_child_split_preserves_predictions_exactly() -> None:
    table = _table()
    hyperparameters = AFHyperparameters(leaf_capacity=4, split_fit_samples=8)
    state = init_af_state(_base())
    state = update_microbatch(state, table, tuple(range(8)), hyperparameters, 0).state
    before = torch.stack(
        tuple(predict_for_node(state, table, 0, (index,))[0] for index in range(8))
    )
    split, _event = install_split(state, table, 0, 8, hyperparameters, 0)
    after = torch.stack(
        tuple(
            predict_for_node(split, table, route(split, table.embeddings[index]).leaf_id, (index,))[0]
            for index in range(8)
        )
    )
    torch.testing.assert_close(after, before, rtol=0.0, atol=0.0)


def test_updating_one_child_does_not_change_sibling_effective_logits() -> None:
    state, event, table, hyperparameters = _split_state()
    sibling_examples = event.right_examples
    before = predict_for_node(state, table, event.right_id, sibling_examples)
    updated = update_microbatch(
        state, table, event.left_examples[:2], hyperparameters, seed=7
    ).state
    after = predict_for_node(updated, table, event.right_id, sibling_examples)
    torch.testing.assert_close(after, before, rtol=0.0, atol=0.0)
    assert updated.nodes[event.left_id].adapter is not state.nodes[event.left_id].adapter
    assert updated.nodes[event.right_id].adapter is state.nodes[event.right_id].adapter


def test_route_depth_and_structure_action_respect_cap() -> None:
    table = _table()
    hyperparameters = AFHyperparameters(leaf_capacity=4, depth_cap_override=0)
    state = init_af_state(_base())
    state = update_microbatch(state, table, tuple(range(4)), hyperparameters, 0).state
    assert current_depth_cap(4, hyperparameters) == 0
    assert structural_action(state, 0, 4, hyperparameters) is None
    assert route(state, table.embeddings[0]).path_ids == (0,)


def test_collapse_buffer_is_union_and_children_become_unreachable() -> None:
    state, event, table, _hyperparameters = _split_state(train_children=True)
    hyperparameters = AFHyperparameters(
        leaf_capacity=2,
        split_fit_samples=8,
        batch_size=4,
        depth_cap_override=1,
    )
    triggering = state.nodes[event.left_id]
    state = replace(
        state,
        nodes=state.nodes.set(
            event.left_id,
            replace(triggering, arrivals_since_structure_change=2),
        ),
    )
    assert structural_action(state, event.left_id, 8, hyperparameters) == "collapse"
    collapsed, collapse = collapse_leaf_pair(
        state, table, event.left_id, 8, hyperparameters, 3
    )
    assert set(collapse.example_ids) == set(event.left_examples) | set(event.right_examples)
    assert set(collapsed.leaf_buffers[0]) == set(range(8))
    assert event.left_id not in collapsed.nodes and event.right_id not in collapsed.nodes
    validate_af_state(collapsed, range(8))


def test_complexity_counters_increment_once_per_operation() -> None:
    table = _table(6)
    hyperparameters = AFHyperparameters(leaf_capacity=6, batch_size=2)
    state = init_af_state(_base())
    first = update_microbatch(state, table, (0, 1), hyperparameters, 0)
    assert first.state.counters.embedding_evaluations == 2
    assert first.state.counters.online_training_examples == 2
    assert first.state.counters.adapter_evaluations == 2
    second = update_microbatch(first.state, table, (2, 3), hyperparameters, 0)
    assert second.replay_example_ids[0][1]
    assert second.state.counters.embedding_evaluations == 4
    assert second.state.counters.online_training_examples == 6
    assert second.state.counters.adapter_evaluations == 6
    assert second.state.counters.hyperplane_evaluations == 0


def test_explicit_adamw_step_matches_torch_reference() -> None:
    table = _table(4)
    hyperparameters = AFHyperparameters(leaf_capacity=4, batch_size=4)
    base = _base()
    state = init_af_state(base)
    result = update_microbatch(state, table, tuple(range(4)), hyperparameters, 0).state

    parameters = [torch.nn.Parameter(torch.zeros_like(tensor)) for tensor in base.tensors]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=hyperparameters.adapter_lr,
        betas=(hyperparameters.beta1, hyperparameters.beta2),
        eps=hyperparameters.epsilon,
        weight_decay=hyperparameters.weight_decay,
    )
    embedding_weight, embedding_bias, classifier_weight, classifier_bias = parameters
    hidden = F.relu(
        F.linear(
            table.trunk_features,
            base.embedding_weight + embedding_weight,
            base.embedding_bias + embedding_bias,
        )
    )
    logits = F.linear(
        hidden,
        base.classifier_weight + classifier_weight,
        base.classifier_bias + classifier_bias,
    )
    loss = F.cross_entropy(logits, table.labels)
    loss.backward()
    optimizer.step()
    for committed, reference in zip(result.nodes[0].adapter.tensors, parameters):
        torch.testing.assert_close(committed, reference, rtol=1e-6, atol=1e-7)


def test_identical_embeddings_fail_instead_of_installing_empty_child() -> None:
    table = StoredExampleTable(
        torch.ones((4, 2), dtype=torch.float32),
        torch.ones((4, 2), dtype=torch.float32),
        torch.zeros((4, 2), dtype=torch.float32),
        torch.tensor([0, 1, 0, 1]),
        torch.zeros(4, dtype=torch.int64),
        torch.arange(4, dtype=torch.int64),
    )
    hyperparameters = AFHyperparameters(leaf_capacity=4)
    state = update_microbatch(
        init_af_state(_base()), table, tuple(range(4)), hyperparameters, 0
    ).state
    with pytest.raises(RuntimeError, match="nonempty"):
        install_split(state, table, 0, 4, hyperparameters, 0)

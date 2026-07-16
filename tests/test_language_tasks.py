from __future__ import annotations

from dataclasses import fields
import inspect

import jax
import numpy as np
import pytest

from apm.continual.language_tasks import (
    AddressBook,
    AddressResult,
    BaseCheckpointRef,
    CompetenceBatch,
    LanguageCurriculum,
    LanguageEvaluationExample,
    LanguageTask,
    NodeId,
    RouterBatch,
    TaskId,
    build_prefix_suffix_batches,
)
from apm.lm.checkpoint import BaseCheckpointRef as CheckpointBaseRef
from apm.lm.text_data import TokenBatch


def test_suffix_changes_cannot_change_the_prefix_only_router_batch() -> None:
    first_router, first_competence = build_prefix_suffix_batches(
        (2, 3, 4, 5, 6, 7, 8),
        prefix_length=4,
        suffix_length=3,
    )
    second_router, second_competence = build_prefix_suffix_batches(
        (2, 3, 4, 5, 20, 21, 22),
        prefix_length=4,
        suffix_length=3,
    )

    for first_value, second_value in zip(
        jax.tree_util.tree_leaves(first_router),
        jax.tree_util.tree_leaves(second_router),
    ):
        np.testing.assert_array_equal(first_value, second_value)
    assert not np.array_equal(first_competence.target_ids, second_competence.target_ids)


def test_router_and_competence_masks_select_disjoint_target_spans() -> None:
    router, competence = build_prefix_suffix_batches(
        (10, 11, 12, 13, 14, 15, 16),
        prefix_length=4,
        suffix_length=3,
    )

    np.testing.assert_array_equal(router.input_ids, [[10, 11, 12]])
    np.testing.assert_array_equal(router.target_ids, [[11, 12, 13]])
    np.testing.assert_array_equal(router.loss_mask, [[True, True, True]])
    np.testing.assert_array_equal(
        competence.loss_mask,
        [[False, False, False, True, True, True]],
    )
    np.testing.assert_array_equal(
        competence.target_ids[competence.loss_mask],
        [14, 15, 16],
    )
    assert not np.any(competence.loss_mask[:, : router.input_ids.shape[1]])


def test_short_suffix_is_right_padded_to_fixed_capacities() -> None:
    router, competence = build_prefix_suffix_batches(
        (2, 3, 4, 5, 6),
        prefix_length=4,
        suffix_length=4,
        pad_token_id=0,
    )

    assert router.input_ids.shape == (1, 3)
    assert competence.input_ids.shape == (1, 7)
    np.testing.assert_array_equal(competence.input_ids, [[2, 3, 4, 5, 0, 0, 0]])
    np.testing.assert_array_equal(competence.target_ids, [[3, 4, 5, 6, 0, 0, 0]])
    np.testing.assert_array_equal(
        competence.attention_mask,
        [[True, True, True, True, False, False, False]],
    )
    np.testing.assert_array_equal(
        competence.loss_mask,
        [[False, False, False, True, False, False, False]],
    )
    assert not router.input_ids.flags.writeable
    assert not competence.loss_mask.flags.writeable


def test_builder_and_router_contract_structurally_exclude_task_identity() -> None:
    signature_names = tuple(inspect.signature(build_prefix_suffix_batches).parameters)

    assert signature_names == (
        "token_ids",
        "prefix_length",
        "suffix_length",
        "pad_token_id",
    )
    assert not any(
        identity_word in parameter_name
        for parameter_name in signature_names
        for identity_word in ("task", "oracle", "node")
    )
    assert tuple(field.name for field in fields(RouterBatch)) == (
        "input_ids",
        "attention_mask",
        "target_ids",
        "loss_mask",
    )
    assert AddressResult._fields == (
        "selected_indices",
        "node_probabilities",
        "node_scores",
        "score_margin",
        "entropy",
    )


def test_manual_evaluation_example_enforces_the_no_leak_boundary() -> None:
    router, competence = build_prefix_suffix_batches(
        (2, 3, 4, 5, 6),
        prefix_length=4,
        suffix_length=2,
    )
    example = LanguageEvaluationExample(
        router,
        competence,
        TaskId("task"),
        NodeId("node"),
    )

    assert example.task_id == "task"
    invalid_competence = CompetenceBatch(
        competence.input_ids,
        competence.attention_mask,
        competence.target_ids,
        np.asarray([[False, False, True, True, False]]),
    )
    with pytest.raises(ValueError, match="inactive across router"):
        LanguageEvaluationExample(
            router,
            invalid_competence,
            TaskId("task"),
            NodeId("node"),
        )


def test_builder_requires_at_least_one_suffix_target() -> None:
    with pytest.raises(ValueError, match="suffix target"):
        build_prefix_suffix_batches(
            (2, 3, 4, 5),
            prefix_length=4,
            suffix_length=2,
        )


def test_address_book_is_fixed_capacity_validated_and_immutable() -> None:
    address_book = AddressBook(
        (NodeId("root"), None, None),
        np.asarray(
            [
                [1.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        np.asarray([True, False, False]),
    )

    assert address_book.max_nodes == 3
    assert address_book.key_dim == 2
    assert address_book.keys.dtype == np.float32
    assert not address_book.keys.flags.writeable
    assert not address_book.valid_node_mask.flags.writeable
    with pytest.raises(ValueError, match="zero keys"):
        AddressBook(
            (NodeId("root"), None),
            np.asarray([[1.0, 0.0], [0.5, 0.5]]),
            np.asarray([True, False]),
        )


def test_curriculum_capacities_match_one_root_plus_one_edge_per_task() -> None:
    router, competence = build_prefix_suffix_batches(
        (2, 3, 4),
        prefix_length=2,
        suffix_length=1,
    )
    example = LanguageEvaluationExample(
        router,
        competence,
        TaskId("task"),
        NodeId("root"),
    )
    train_batch = TokenBatch(
        np.asarray([[2]], dtype=np.int32),
        np.asarray([[True]]),
        np.asarray([[3]], dtype=np.int32),
        np.asarray([[True]]),
    )
    task = LanguageTask(
        TaskId("task"),
        (train_batch,),
        (example,),
        (example,),
    )

    curriculum = LanguageCurriculum((task,), max_nodes=2, max_edges=1)

    assert curriculum.max_nodes == 2
    assert curriculum.max_edges == 1
    with pytest.raises(ValueError, match="max_nodes"):
        LanguageCurriculum((task,), max_nodes=3, max_edges=2)
    assert BaseCheckpointRef is CheckpointBaseRef

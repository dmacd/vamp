from __future__ import annotations

import numpy as np
import pytest

import apm.data.text.language_tasks as language_tasks
from apm.data.text.curricula import (
    CorpusCurriculum,
    CorpusDocumentSplits,
    CorpusTask,
)
from apm.data.text.language_tasks import (
    LanguageDataBuildConfig,
    RawTextTask,
    prepare_language_curriculum,
    raw_tasks_from_corpus_curriculum,
)
from apm.lm.text import CharTokenizer


def _task(task_id: str, offset: int) -> RawTextTask:
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    shifted = alphabet[offset:] + alphabet[:offset]
    return RawTextTask(
        task_id=task_id,
        train_texts=(shifted * 2, (shifted[::-1]) * 2),
        validation_texts=(shifted * 3,),
        test_texts=((shifted[::-1]) * 3,),
    )


def _config() -> LanguageDataBuildConfig:
    return LanguageDataBuildConfig(
        context_length=8,
        batch_size=2,
        stride=8,
        prefix_lengths=(4, 6),
        suffix_length=2,
        examples_per_task_and_prefix=2,
        primary_prefix_length=4,
    )


def _full_sort_evaluation_sequences(
    texts: tuple[str, ...],
    tokenizer: CharTokenizer,
    config: LanguageDataBuildConfig,
    *,
    split_identity: str,
) -> tuple[tuple[int, ...], ...]:
    maximum_length = max(config.prefix_lengths) + config.suffix_length
    minimum_length = max(config.prefix_lengths) + 1
    evaluation_stride = max(1, maximum_length // 4)
    candidates = tuple(
        (
            language_tasks._sequence_identity(
                split_identity,
                text_index,
                start,
                sequence,
            ),
            sequence,
        )
        for text_index, text in enumerate(texts)
        for tokens in (tokenizer.encode(text, add_eos=True),)
        for start in range(
            0,
            max(len(tokens) - minimum_length + 1, 0),
            evaluation_stride,
        )
        for sequence in (tokens[start : start + maximum_length],)
    )
    return tuple(
        sequence
        for _, sequence in sorted(candidates, key=lambda candidate: candidate[0])[
            : config.examples_per_task_and_prefix
        ]
    )


def test_prepare_language_curriculum_builds_fixed_batches_and_complete_sweeps() -> None:
    raw_tasks = (_task("task_0", 0), _task("task_1", 3))
    tokenizer = CharTokenizer.from_training_text("abcdefghijklmnopqrstuvwxyz")
    prepared = prepare_language_curriculum(
        "unit-curriculum",
        raw_tasks,
        ("abcdefghijklmnopqrstuvwxyz" * 3,),
        tokenizer,
        _config(),
    )

    assert prepared.curriculum.max_nodes == 3
    assert prepared.curriculum.max_edges == 2
    assert tuple(task.task_id for task in prepared.curriculum.tasks) == (
        "task_0",
        "task_1",
    )
    assert len(prepared.root_validation_probes) == 2
    assert all(probe.input_ids.shape == (1, 3) for probe in prepared.root_validation_probes)
    assert tuple(
        (sweep.task_id, sweep.prefix_length)
        for sweep in prepared.evaluation_sweeps
    ) == (
        ("task_0", 4),
        ("task_0", 6),
        ("task_1", 4),
        ("task_1", 6),
    )
    assert all(
        len(sweep.validation_examples) == 2 and len(sweep.test_examples) == 2
        for sweep in prepared.evaluation_sweeps
    )
    assert all(
        batch.input_ids.shape == (2, 8)
        for task in prepared.curriculum.tasks
        for batch in task.train_batches
    )
    assert all(
        example.router_batch.input_ids.shape[1] == 3
        for task in prepared.curriculum.tasks
        for example in task.validation_examples
    )


def test_document_training_windows_do_not_create_cross_document_transitions() -> None:
    tokenizer = CharTokenizer.from_training_text("ab")
    raw_task = RawTextTask(
        task_id="separate-documents",
        train_texts=("aaaaa", "bbbbb"),
        validation_texts=("a" * 20,),
        test_texts=("b" * 20,),
    )
    prepared = prepare_language_curriculum(
        "no-crossing",
        (raw_task,),
        ("ab" * 12,),
        tokenizer,
        LanguageDataBuildConfig(
            context_length=8,
            batch_size=2,
            stride=8,
            prefix_lengths=(4,),
            suffix_length=2,
            examples_per_task_and_prefix=1,
            primary_prefix_length=4,
        ),
    )
    task = prepared.curriculum.tasks[0]
    active_pairs = tuple(
        (int(input_id), int(target_id))
        for batch in task.train_batches
        for input_row, target_row, mask_row in zip(
            batch.input_ids,
            batch.target_ids,
            batch.loss_mask,
        )
        for input_id, target_id, active in zip(input_row, target_row, mask_row)
        if active
    )
    a_id, b_id = tokenizer.encode("ab")
    assert (a_id, b_id) not in active_pairs
    assert (b_id, a_id) not in active_pairs
    assert any(target == tokenizer.eos_token_id for _, target in active_pairs)


def test_corpus_curriculum_adapter_preserves_every_raw_document_boundary() -> None:
    curriculum = CorpusCurriculum(
        curriculum_id="raw-boundaries",
        tasks=tuple(
            CorpusTask(
                task_id=f"task-{task_index}",
                splits=CorpusDocumentSplits(
                    train=(f"train-{task_index}-a\n", f"train-{task_index}-b\t"),
                    validation=(f"validation-{task_index}\r\n",),
                    test=(f"test-{task_index}  raw",),
                ),
            )
            for task_index in range(4)
        ),
    )

    raw_tasks = raw_tasks_from_corpus_curriculum(curriculum)

    assert tuple(task.task_id for task in raw_tasks) == tuple(
        task.task_id for task in curriculum.tasks
    )
    assert all(
        raw_task.train_texts == corpus_task.splits.train
        and raw_task.validation_texts == corpus_task.splits.validation
        and raw_task.test_texts == corpus_task.splits.test
        for raw_task, corpus_task in zip(raw_tasks, curriculum.tasks)
    )


def test_language_data_builder_is_deterministic_and_requires_suffix_tokens() -> None:
    task = _task("task_0", 0)
    tokenizer = CharTokenizer.from_training_text("abcdefghijklmnopqrstuvwxyz")
    first = prepare_language_curriculum(
        "deterministic",
        (task,),
        ("abcdefghijklmnopqrstuvwxyz" * 3,),
        tokenizer,
        _config(),
    )
    second = prepare_language_curriculum(
        "deterministic",
        (task,),
        ("abcdefghijklmnopqrstuvwxyz" * 3,),
        tokenizer,
        _config(),
    )
    for first_sweep, second_sweep in zip(
        first.evaluation_sweeps,
        second.evaluation_sweeps,
    ):
        np.testing.assert_array_equal(
            first_sweep.test_examples[0].router_batch.input_ids,
            second_sweep.test_examples[0].router_batch.input_ids,
        )
    with pytest.raises(ValueError, match="evaluation spans"):
        prepare_language_curriculum(
            "too-short",
            (
                RawTextTask(
                    task_id="short",
                    train_texts=("abcdefgh",),
                    validation_texts=("abc",),
                    test_texts=("abc",),
                ),
            ),
            ("abc",),
            tokenizer,
            _config(),
        )


def test_evaluation_sequences_right_pad_a_shorter_suffix() -> None:
    tokenizer = CharTokenizer.from_training_text("abcdefgh")
    config = LanguageDataBuildConfig(
        context_length=8,
        batch_size=2,
        stride=8,
        prefix_lengths=(4, 6),
        suffix_length=2,
        examples_per_task_and_prefix=1,
        primary_prefix_length=4,
    )
    task = RawTextTask(
        task_id="short-suffix",
        train_texts=("abcdefgh" * 2,),
        validation_texts=("abcdef",),
        test_texts=("abcdef",),
    )

    prepared = prepare_language_curriculum(
        "short-suffix",
        (task,),
        ("abcdef",),
        tokenizer,
        config,
    )
    longest_prefix_sweep = prepared.evaluation_sweeps[1]
    competence = longest_prefix_sweep.validation_examples[0].competence_batch

    assert competence.input_ids.shape == (1, 7)
    assert np.sum(competence.attention_mask) == 6
    assert np.sum(competence.loss_mask) == 1
    assert not competence.attention_mask[0, -1]


def test_bounded_evaluation_selection_matches_full_sort_for_unicode_duplicates() -> None:
    texts = (
        "åβ🙂漢字abc" * 4,
        "åβ🙂漢字abc" * 4,
        "Ωé🦊defgh" * 4,
    )
    tokenizer = CharTokenizer.from_training_text("".join(texts))
    config = LanguageDataBuildConfig(
        context_length=8,
        batch_size=2,
        stride=8,
        prefix_lengths=(4, 6),
        suffix_length=2,
        examples_per_task_and_prefix=5,
        primary_prefix_length=4,
    )
    split_identity = "验证:評価:🌱"

    expected = _full_sort_evaluation_sequences(
        texts,
        tokenizer,
        config,
        split_identity=split_identity,
    )
    actual = language_tasks._select_evaluation_sequences(
        texts,
        tokenizer,
        config,
        split_identity=split_identity,
    )

    assert actual == expected


def test_bounded_evaluation_selection_preserves_stable_identity_ties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    texts = (
        "abcdefghijklmnopqrstuvwx",
        "zyxwvutsrqponmlkjihgfedcba",
    )
    tokenizer = CharTokenizer.from_training_text("".join(texts))
    config = LanguageDataBuildConfig(
        context_length=8,
        batch_size=2,
        stride=8,
        prefix_lengths=(4, 6),
        suffix_length=2,
        examples_per_task_and_prefix=5,
        primary_prefix_length=4,
    )
    monkeypatch.setattr(
        language_tasks,
        "_sequence_identity",
        lambda split_identity, text_index, start, sequence: "same-identity",
    )

    expected = _full_sort_evaluation_sequences(
        texts,
        tokenizer,
        config,
        split_identity="tie-boundary",
    )
    actual = language_tasks._select_evaluation_sequences(
        texts,
        tokenizer,
        config,
        split_identity="tie-boundary",
    )

    assert actual == expected

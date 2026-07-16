"""Deterministic conversion from document curricula to language-task arrays."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Sequence

import numpy as np

from apm.continual.language_tasks import (
    LanguageCurriculum,
    LanguageEvaluationExample,
    LanguageTask,
    NodeId,
    RouterBatch,
    TaskId,
    build_prefix_suffix_batches,
)
from apm.data.text.curricula import (
    CharacterPermutationCurriculum,
    CorpusCurriculum,
    DocumentCurriculum,
)
from apm.lm.text import TextTokenizer
from apm.lm.text_data import (
    TokenBatch,
    batch_token_windows,
    causal_token_windows,
)


@dataclass(frozen=True)
class RawTextTask:
    """One task's document-level text before tokenization and packing."""

    task_id: str
    train_texts: tuple[str, ...]
    validation_texts: tuple[str, ...]
    test_texts: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("raw text task_id must not be empty")
        for split_name in ("train_texts", "validation_texts", "test_texts"):
            texts = getattr(self, split_name)
            if not isinstance(texts, tuple) or not texts:
                raise ValueError(f"{split_name} must be a nonempty tuple")
            if any(not isinstance(text, str) or not text for text in texts):
                raise ValueError(f"{split_name} must contain nonempty strings")


@dataclass(frozen=True)
class LanguageDataBuildConfig:
    """Fixed train packing, probe count, and evaluation-prefix sweep."""

    context_length: int
    batch_size: int
    stride: int
    prefix_lengths: tuple[int, ...]
    suffix_length: int
    examples_per_task_and_prefix: int
    primary_prefix_length: int

    def __post_init__(self) -> None:
        positive_values = (
            self.context_length,
            self.batch_size,
            self.stride,
            self.suffix_length,
            self.examples_per_task_and_prefix,
            self.primary_prefix_length,
        )
        if any(type(value) is not int or value <= 0 for value in positive_values):
            raise ValueError("language data dimensions must be positive integers")
        if (
            not isinstance(self.prefix_lengths, tuple)
            or not self.prefix_lengths
            or tuple(sorted(set(self.prefix_lengths))) != self.prefix_lengths
            or any(type(length) is not int or length < 2 for length in self.prefix_lengths)
        ):
            raise ValueError("prefix lengths must be unique increasing integers >= 2")
        if self.primary_prefix_length not in self.prefix_lengths:
            raise ValueError("primary_prefix_length must belong to prefix_lengths")
        if max(self.prefix_lengths) + self.suffix_length - 1 > self.context_length:
            raise ValueError("the largest prefix/suffix span exceeds context_length")


@dataclass(frozen=True)
class TaskEvaluationSweep:
    """Validation and test examples for one task at one prefix length."""

    task_id: TaskId
    prefix_length: int
    validation_examples: tuple[LanguageEvaluationExample, ...]
    test_examples: tuple[LanguageEvaluationExample, ...]

    def __post_init__(self) -> None:
        if not self.task_id or type(self.prefix_length) is not int or self.prefix_length < 2:
            raise ValueError("evaluation sweep identity and prefix length are invalid")
        for examples in (self.validation_examples, self.test_examples):
            if not examples or any(example.task_id != self.task_id for example in examples):
                raise ValueError("evaluation sweeps require nonempty task-aligned examples")
            expected_width = self.prefix_length - 1
            if any(example.router_batch.input_ids.shape[1] != expected_width for example in examples):
                raise ValueError("evaluation router width must match prefix_length")


@dataclass(frozen=True)
class PreparedLanguageCurriculum:
    """Training curriculum, root probes, and complete prefix evaluation sweep."""

    curriculum_id: str
    curriculum: LanguageCurriculum
    root_validation_probes: tuple[RouterBatch, ...]
    evaluation_sweeps: tuple[TaskEvaluationSweep, ...]
    build_config: LanguageDataBuildConfig

    def __post_init__(self) -> None:
        if not self.curriculum_id:
            raise ValueError("prepared curriculum_id must not be empty")
        if not isinstance(self.curriculum, LanguageCurriculum):
            raise TypeError("curriculum must be a LanguageCurriculum")
        expected_count = self.build_config.examples_per_task_and_prefix
        if len(self.root_validation_probes) != expected_count:
            raise ValueError("root probe count must equal the fixed example count")
        expected_sweeps = len(self.curriculum.tasks) * len(
            self.build_config.prefix_lengths
        )
        if len(self.evaluation_sweeps) != expected_sweeps:
            raise ValueError("evaluation sweep must cover every task/prefix pair")
        expected_pairs = tuple(
            (task.task_id, prefix_length)
            for task in self.curriculum.tasks
            for prefix_length in self.build_config.prefix_lengths
        )
        actual_pairs = tuple(
            (sweep.task_id, sweep.prefix_length)
            for sweep in self.evaluation_sweeps
        )
        if actual_pairs != expected_pairs:
            raise ValueError("evaluation sweeps must follow task then prefix order")


def raw_tasks_from_document_curriculum(
    curriculum: DocumentCurriculum,
) -> tuple[RawTextTask, ...]:
    """Expose normalized document curriculum splits as raw text task values."""
    if not isinstance(curriculum, DocumentCurriculum):
        raise TypeError("curriculum must be a DocumentCurriculum")
    return tuple(
        RawTextTask(
            task_id=task.task_id,
            train_texts=tuple(document.text for document in task.splits.train),
            validation_texts=tuple(
                document.text for document in task.splits.validation
            ),
            test_texts=tuple(document.text for document in task.splits.test),
        )
        for task in curriculum.tasks
    )


def raw_tasks_from_character_curriculum(
    curriculum: CharacterPermutationCurriculum,
) -> tuple[RawTextTask, ...]:
    """Expose each permuted TinyShakespeare split without changing formatting."""
    if not isinstance(curriculum, CharacterPermutationCurriculum):
        raise TypeError("curriculum must be a CharacterPermutationCurriculum")
    return tuple(
        RawTextTask(
            task_id=task.task_id,
            train_texts=(task.splits.train,),
            validation_texts=(task.splits.validation,),
            test_texts=(task.splits.test,),
        )
        for task in curriculum.tasks
    )


def raw_tasks_from_corpus_curriculum(
    curriculum: CorpusCurriculum,
) -> tuple[RawTextTask, ...]:
    """Preserve raw region or macro-document boundaries as language tasks."""
    if not isinstance(curriculum, CorpusCurriculum):
        raise TypeError("curriculum must be a CorpusCurriculum")
    return tuple(
        RawTextTask(
            task_id=task.task_id,
            train_texts=task.splits.train,
            validation_texts=task.splits.validation,
            test_texts=task.splits.test,
        )
        for task in curriculum.tasks
    )


def prepare_language_curriculum(
    curriculum_id: str,
    raw_tasks: tuple[RawTextTask, ...],
    root_validation_texts: tuple[str, ...],
    tokenizer: TextTokenizer,
    config: LanguageDataBuildConfig,
) -> PreparedLanguageCurriculum:
    """Tokenize, pack, and split supplied documents without crossing boundaries."""
    if not curriculum_id:
        raise ValueError("curriculum_id must not be empty")
    if not raw_tasks or any(not isinstance(task, RawTextTask) for task in raw_tasks):
        raise ValueError("raw_tasks must be a nonempty tuple of RawTextTask values")
    if len({task.task_id for task in raw_tasks}) != len(raw_tasks):
        raise ValueError("raw task IDs must be unique")
    if not isinstance(root_validation_texts, tuple) or not root_validation_texts:
        raise ValueError("root_validation_texts must be a nonempty tuple")
    if not isinstance(tokenizer, TextTokenizer):
        raise TypeError("tokenizer must satisfy TextTokenizer")
    if not isinstance(config, LanguageDataBuildConfig):
        raise TypeError("config must be a LanguageDataBuildConfig")

    task_material = tuple(
        _prepare_task_material(task, tokenizer, config)
        for task in raw_tasks
    )
    primary_tasks = tuple(material[0] for material in task_material)
    curriculum = LanguageCurriculum(
        tasks=primary_tasks,
        max_nodes=len(primary_tasks) + 1,
        max_edges=len(primary_tasks),
    )
    root_sequences = _select_evaluation_sequences(
        root_validation_texts,
        tokenizer,
        config,
        split_identity="root-validation",
    )
    root_probes = tuple(
        build_prefix_suffix_batches(
            sequence,
            config.primary_prefix_length,
            config.suffix_length,
            pad_token_id=tokenizer.pad_token_id,
        )[0]
        for sequence in root_sequences
    )
    return PreparedLanguageCurriculum(
        curriculum_id=curriculum_id,
        curriculum=curriculum,
        root_validation_probes=root_probes,
        evaluation_sweeps=tuple(
            sweep
            for _, sweeps in task_material
            for sweep in sweeps
        ),
        build_config=config,
    )


def _prepare_task_material(
    raw_task: RawTextTask,
    tokenizer: TextTokenizer,
    config: LanguageDataBuildConfig,
) -> tuple[LanguageTask, tuple[TaskEvaluationSweep, ...]]:
    task_id = TaskId(raw_task.task_id)
    validation_sequences = _select_evaluation_sequences(
        raw_task.validation_texts,
        tokenizer,
        config,
        split_identity=f"{raw_task.task_id}:validation",
    )
    test_sequences = _select_evaluation_sequences(
        raw_task.test_texts,
        tokenizer,
        config,
        split_identity=f"{raw_task.task_id}:test",
    )
    sweeps = tuple(
        TaskEvaluationSweep(
            task_id=task_id,
            prefix_length=prefix_length,
            validation_examples=_evaluation_examples(
                validation_sequences,
                task_id,
                prefix_length,
                config.suffix_length,
                tokenizer.pad_token_id,
            ),
            test_examples=_evaluation_examples(
                test_sequences,
                task_id,
                prefix_length,
                config.suffix_length,
                tokenizer.pad_token_id,
            ),
        )
        for prefix_length in config.prefix_lengths
    )
    primary_sweep = next(
        sweep
        for sweep in sweeps
        if sweep.prefix_length == config.primary_prefix_length
    )
    return (
        LanguageTask(
            task_id=task_id,
            train_batches=_training_batches(
                raw_task.train_texts,
                tokenizer,
                config,
            ),
            validation_examples=primary_sweep.validation_examples,
            test_examples=primary_sweep.test_examples,
        ),
        sweeps,
    )


def _training_batches(
    texts: tuple[str, ...],
    tokenizer: TextTokenizer,
    config: LanguageDataBuildConfig,
) -> tuple[TokenBatch, ...]:
    document_windows = tuple(
        causal_token_windows(
            tokenizer.encode(text, add_eos=True),
            config.context_length,
            tokenizer.pad_token_id,
            stride=config.stride,
        )
        for text in texts
    )
    nonempty_windows = tuple(
        windows for windows in document_windows if windows.input_ids.shape[0] > 0
    )
    if not nonempty_windows:
        raise ValueError("training texts contain no causal transitions")
    windows = TokenBatch(
        input_ids=np.concatenate(tuple(batch.input_ids for batch in nonempty_windows)),
        attention_mask=np.concatenate(
            tuple(batch.attention_mask for batch in nonempty_windows)
        ),
        target_ids=np.concatenate(tuple(batch.target_ids for batch in nonempty_windows)),
        loss_mask=np.concatenate(tuple(batch.loss_mask for batch in nonempty_windows)),
    )
    return batch_token_windows(windows, config.batch_size, tokenizer.pad_token_id)


def _select_evaluation_sequences(
    texts: Sequence[str],
    tokenizer: TextTokenizer,
    config: LanguageDataBuildConfig,
    *,
    split_identity: str,
) -> tuple[tuple[int, ...], ...]:
    required_length = max(config.prefix_lengths) + config.suffix_length
    evaluation_stride = max(1, required_length // 4)
    candidates = tuple(
        (
            _sequence_identity(split_identity, text_index, start, sequence),
            sequence,
        )
        for text_index, text in enumerate(texts)
        for tokens in (tokenizer.encode(text, add_eos=True),)
        for start in range(
            0,
            max(len(tokens) - required_length + 1, 0),
            evaluation_stride,
        )
        for sequence in (tokens[start : start + required_length],)
    )
    ordered = tuple(sequence for _, sequence in sorted(candidates, key=lambda item: item[0]))
    required_count = config.examples_per_task_and_prefix
    if len(ordered) < required_count:
        raise ValueError(
            f"{split_identity} has {len(ordered)} full evaluation spans; "
            f"requires exactly {required_count}"
        )
    return ordered[:required_count]


def _sequence_identity(
    split_identity: str,
    text_index: int,
    start: int,
    sequence: tuple[int, ...],
) -> str:
    token_bytes = np.asarray(sequence, dtype="<i4").tobytes()
    prefix = f"{split_identity}:{text_index}:{start}:".encode("utf-8")
    return sha256(prefix + token_bytes).hexdigest()


def _evaluation_examples(
    sequences: tuple[tuple[int, ...], ...],
    task_id: TaskId,
    prefix_length: int,
    suffix_length: int,
    pad_token_id: int,
) -> tuple[LanguageEvaluationExample, ...]:
    return tuple(
        LanguageEvaluationExample(
            router_batch=router_batch,
            competence_batch=competence_batch,
            task_id=task_id,
            oracle_node_id=NodeId(str(task_id)),
        )
        for sequence in sequences
        for router_batch, competence_batch in (
            build_prefix_suffix_batches(
                sequence,
                prefix_length,
                suffix_length,
                pad_token_id=pad_token_id,
            ),
        )
    )

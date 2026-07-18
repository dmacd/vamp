"""Paired, provenance-rich TinyStories post-mortem evaluation preparation."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import struct

import numpy as np

from apm.continual.language_evaluation import (
    IN_DOMAIN_TOPIC_SPECIALIZATION,
    CueRegime,
    LanguageEvaluationCondition,
    LanguageEvaluationSuite,
    LanguageExampleProvenance,
    LanguageSuiteExample,
)
from apm.continual.language_tasks import (
    LanguageEvaluationExample,
    NodeId,
    TaskId,
    build_prefix_suffix_batches,
)
from apm.data.text.curricula import (
    TINYSTORIES_TOPICS,
    TextDocument,
    TinyStoriesSourceContract,
    classify_tinystory_topic,
    load_pinned_dataset_text,
    parse_tinystories,
    topic_scores,
)
from apm.lm.text import TextTokenizer


TINYSTORIES_POSTMORTEM_CONDITIONS = (
    LanguageEvaluationCondition("prefix64_suffix128", 64, 128),
    LanguageEvaluationCondition("prefix128_suffix128", 128, 128),
    LanguageEvaluationCondition("prefix192_suffix64", 192, 64),
)


@dataclass(frozen=True)
class TinyStoriesEvaluationTaskDocuments:
    """Every classified document from one topic in the official test half."""

    task_id: TaskId
    topic: str
    documents: tuple[TextDocument, ...]

    def __post_init__(self) -> None:
        topic_names = tuple(topic.name for topic in TINYSTORIES_TOPICS)
        if not self.task_id or self.topic not in topic_names:
            raise ValueError("TinyStories evaluation task identity is invalid")
        if not self.documents or any(
            not isinstance(document, TextDocument) for document in self.documents
        ):
            raise ValueError("TinyStories evaluation tasks require classified documents")
        content_ids = tuple(document.content_id for document in self.documents)
        if len(set(content_ids)) != len(content_ids):
            raise ValueError("TinyStories evaluation task documents must be unique")
        if any(
            (assignment := classify_tinystory_topic(document.text)) is None
            or assignment.topic != self.topic
            for document in self.documents
        ):
            raise ValueError("every evaluation document must match its assigned topic")


@dataclass(frozen=True)
class _AnchorSpan:
    document_id: str
    token_offset: int
    token_ids: tuple[int, ...]
    pair_hash: str


def load_complete_classified_tinystories_test_half(
    official_validation_path: str | Path,
    source: TinyStoriesSourceContract,
) -> tuple[TinyStoriesEvaluationTaskDocuments, ...]:
    """Verify and classify the complete official hash-ordered test half."""
    if not isinstance(source, TinyStoriesSourceContract):
        raise TypeError("source must be a TinyStoriesSourceContract")
    documents = parse_tinystories(
        load_pinned_dataset_text(official_validation_path, source.validation_file)
    )
    if len(documents) % 2:
        raise ValueError("deduplicated official validation must divide exactly in half")
    ordered = tuple(sorted(documents, key=lambda document: document.content_id))
    test_documents = ordered[len(ordered) // 2 :]
    return tuple(
        TinyStoriesEvaluationTaskDocuments(
            task_id=TaskId(f"tinystories-topic-{topic.name}"),
            topic=topic.name,
            documents=tuple(
                document
                for document in test_documents
                if (
                    (assignment := classify_tinystory_topic(document.text))
                    is not None
                    and assignment.topic == topic.name
                )
            ),
        )
        for topic in TINYSTORIES_TOPICS
    )


def build_tinystories_postmortem_suite(
    task_documents: tuple[TinyStoriesEvaluationTaskDocuments, ...],
    tokenizer: TextTokenizer,
) -> LanguageEvaluationSuite:
    """Build the fixed three-condition suite from exact 256-token anchors."""
    if (
        not isinstance(task_documents, tuple)
        or not task_documents
        or any(
            not isinstance(task, TinyStoriesEvaluationTaskDocuments)
            for task in task_documents
        )
    ):
        raise ValueError("task_documents must contain classified TinyStories tasks")
    if len({task.task_id for task in task_documents}) != len(task_documents):
        raise ValueError("TinyStories evaluation task IDs must be unique")
    if not isinstance(tokenizer, TextTokenizer):
        raise TypeError("tokenizer must satisfy TextTokenizer")

    examples = tuple(
        suite_example
        for task in task_documents
        for anchor_index, anchor in enumerate(
            _round_robin_anchor_spans(task, tokenizer)
        )
        for suite_example in _paired_anchor_examples(
            task,
            anchor,
            anchor_index,
            tokenizer,
        )
    )
    return LanguageEvaluationSuite(
        suite_id="tinystories-topic-postmortem-v1",
        benchmark_label=IN_DOMAIN_TOPIC_SPECIALIZATION,
        primary_condition_id=TINYSTORIES_POSTMORTEM_CONDITIONS[0].condition_id,
        conditions=TINYSTORIES_POSTMORTEM_CONDITIONS,
        examples=examples,
    )


def _round_robin_anchor_spans(
    task: TinyStoriesEvaluationTaskDocuments,
    tokenizer: TextTokenizer,
    *,
    anchor_count: int = 128,
    anchor_tokens: int = 256,
    stride: int = 32,
) -> tuple[_AnchorSpan, ...]:
    document_candidates = tuple(
        (
            document,
            tokens,
            tuple(range(0, len(tokens) - anchor_tokens + 1, stride)),
        )
        for document in sorted(task.documents, key=lambda value: value.content_id)
        for tokens in (tokenizer.encode(document.text, add_eos=True),)
        if len(tokens) >= anchor_tokens
    )
    if not document_candidates:
        raise ValueError(f"task {task.task_id!r} has no exact 256-token anchors")
    selected: list[_AnchorSpan] = []
    round_index = 0
    while len(selected) < anchor_count:
        before_round = len(selected)
        for document, tokens, offsets in document_candidates:
            if round_index >= len(offsets):
                continue
            offset = offsets[round_index]
            token_span = tuple(tokens[offset : offset + anchor_tokens])
            selected.append(
                _anchor_span(document.content_id, offset, token_span)
            )
            if len(selected) == anchor_count:
                return tuple(selected)
        if len(selected) == before_round:
            break
        round_index += 1
    raise ValueError(
        f"task {task.task_id!r} has {len(selected)} exact anchor spans; "
        f"requires {anchor_count}"
    )


def _anchor_span(
    document_id: str,
    token_offset: int,
    token_ids: tuple[int, ...],
) -> _AnchorSpan:
    digest = sha256()
    digest.update(document_id.encode("ascii"))
    digest.update(struct.pack("<Q", token_offset))
    digest.update(np.asarray(token_ids, dtype="<i4").tobytes())
    return _AnchorSpan(document_id, token_offset, token_ids, digest.hexdigest())


def _paired_anchor_examples(
    task: TinyStoriesEvaluationTaskDocuments,
    anchor: _AnchorSpan,
    anchor_index: int,
    tokenizer: TextTokenizer,
) -> tuple[LanguageSuiteExample, ...]:
    pair_id = f"{task.task_id}:span-{anchor_index:03d}-{anchor.pair_hash[:12]}"
    provenance = LanguageExampleProvenance(
        source_document_id=anchor.document_id,
        token_offset=anchor.token_offset,
        pair_hash=anchor.pair_hash,
    )
    return tuple(
        LanguageSuiteExample(
            pair_id=pair_id,
            condition_id=condition.condition_id,
            split="test",
            example=LanguageEvaluationExample(
                router_batch=router_batch,
                competence_batch=competence_batch,
                task_id=task.task_id,
                oracle_node_id=NodeId(str(task.task_id)),
            ),
            provenance=provenance,
            cue_regime=cue_regime,
            visible_concept_ids=visible_concepts,
        )
        for condition in TINYSTORIES_POSTMORTEM_CONDITIONS
        for router_batch, competence_batch in (
            build_prefix_suffix_batches(
                anchor.token_ids,
                condition.prefix_tokens,
                condition.suffix_tokens,
                pad_token_id=tokenizer.pad_token_id,
            ),
        )
        for cue_regime, visible_concepts in (
            _visible_prefix_cues(
                task.topic,
                tokenizer.decode(anchor.token_ids[: condition.prefix_tokens]),
            ),
        )
    )


def _visible_prefix_cues(
    assigned_topic: str,
    visible_prefix: str,
) -> tuple[CueRegime, tuple[str, ...]]:
    visible_assignment = classify_tinystory_topic(visible_prefix)
    assigned_score = next(
        score for score in topic_scores(visible_prefix) if score.topic == assigned_topic
    )
    concepts = tuple(
        sorted(f"{assigned_topic}:{concept}" for concept in assigned_score.matched_concepts)
    )
    cue_regime: CueRegime = (
        "cue_sufficient"
        if visible_assignment is not None
        and visible_assignment.topic == assigned_topic
        else "cue_present"
        if concepts
        else "cue_hidden_or_ambiguous"
    )
    return cue_regime, concepts


__all__ = [
    "TINYSTORIES_POSTMORTEM_CONDITIONS",
    "TinyStoriesEvaluationTaskDocuments",
    "build_tinystories_postmortem_suite",
    "load_complete_classified_tinystories_test_half",
]

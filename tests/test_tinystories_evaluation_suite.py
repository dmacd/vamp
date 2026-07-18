from __future__ import annotations

from apm.continual.language_evaluation import (
    IN_DOMAIN_TOPIC_SPECIALIZATION,
    summarize_language_cue_coverage,
)
from apm.data.text.curricula import normalized_documents
from apm.data.text.tinystories_evaluation import (
    TINYSTORIES_POSTMORTEM_CONDITIONS,
    TinyStoriesEvaluationTaskDocuments,
    build_tinystories_postmortem_suite,
)
from apm.lm.text import CharTokenizer
from apm.memory.graph import TaskId


def test_postmortem_suite_uses_exact_paired_anchors_and_visible_cues() -> None:
    texts = tuple(
        ("softly " * 45) + f"number {index:03d}. dog cat rabbit."
        for index in range(128)
    )
    documents = normalized_documents(texts)
    tokenizer = CharTokenizer.from_training_text("".join(texts))
    suite = build_tinystories_postmortem_suite(
        (
            TinyStoriesEvaluationTaskDocuments(
                task_id=TaskId("tinystories-topic-animals"),
                topic="animals",
                documents=documents,
            ),
        ),
        tokenizer,
    )

    assert suite.benchmark_label == IN_DOMAIN_TOPIC_SPECIALIZATION
    assert suite.conditions == TINYSTORIES_POSTMORTEM_CONDITIONS
    assert len(suite.examples) == 128 * 3
    pair_ids = tuple(dict.fromkeys(example.pair_id for example in suite.examples))
    assert len(pair_ids) == 128
    assert all(
        tuple(
            example.condition_id
            for example in suite.examples
            if example.pair_id == pair_id
        )
        == tuple(condition.condition_id for condition in suite.conditions)
        for pair_id in pair_ids
    )
    assert len(
        {
            example.provenance.source_document_id
            for example in suite.examples
        }
    ) == 128
    assert all(example.provenance.token_offset == 0 for example in suite.examples)
    assert all(
        example.cue_regime == "cue_hidden_or_ambiguous"
        for example in suite.examples
        if example.condition_id == "prefix64_suffix128"
    )
    coverage = summarize_language_cue_coverage(suite)
    assert len(coverage) == 6
    assert {
        (row.condition_id, row.cue_regime, row.example_count)
        for row in coverage
    } == {
        (condition.condition_id, cue_regime, 128)
        for condition in suite.conditions
        for cue_regime in ("cue_hidden_or_ambiguous", "all")
    }

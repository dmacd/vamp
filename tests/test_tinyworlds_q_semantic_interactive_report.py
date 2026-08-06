from __future__ import annotations

from dataclasses import replace

import pytest

from apm.data.text.tinyworlds_q_semantic.contracts import SemanticQueryResult
from apm.data.text.tinyworlds_q_semantic.interactive_report import (
    build_interactive_report_data,
    parse_opened_catalog_audit,
    render_interactive_report,
)


_METHODS = (
    "base",
    "independent",
    "sequential",
    "vamp_oracle",
    "vamp_exhaustive",
    "vamp_hopfield",
    "vamp_ebt_uniform",
    "vamp_ebt_hopfield",
    "vamp_random",
)
_CONCEPTS = ("cat", "dog")


def _audit() -> str:
    blocks = []
    for concept in _CONCEPTS:
        blocks.append(f"## {concept}")
        for fact_number in range(1, 13):
            fact_id = f"{concept}-fact-{fact_number:02d}"
            lines = [
                f"### {fact_id}",
                "",
                "Relation: `taxonomy`; answer type: `category`",
                "Answer: `animal`",
                "Accepted: `animal`",
                "Triggers: `animal`, `animals`",
                "",
                "Evidence:",
                f"- `{'a' * 64}` / `story-{concept}-{fact_number}` sentence 1: {concept.title()}s are animals.",
                "",
                "Queries:",
            ]
            query_specs = (
                ("validation", 0, "forward"),
                ("validation", 1, "forward"),
                ("validation", 2, "reverse"),
                ("test", 0, "forward"),
                ("test", 1, "forward"),
                ("test", 2, "forward"),
                ("test", 3, "reverse"),
                ("test", 4, "reverse"),
            )
            for split, query_number, direction in query_specs:
                template_id = f"{fact_id}-{split}-{query_number:02d}"
                prompt = (
                    f"Which kind of thing is a {concept}? Answer:"
                    if direction == "forward"
                    else "Which concept is an animal? Answer:"
                )
                if (
                    concept == "cat"
                    and fact_number == 1
                    and split == "test"
                    and query_number == 0
                ):
                    prompt = "Can a </script><script> marker escape? Answer:"
                lines.extend(
                    (
                        f"- `{template_id}` ({split}, {direction})",
                        f"  - prompt: {prompt}",
                        "  - candidates: ('animal', 'object', 'plant', 'insect')",
                        "  - correct position: 0",
                        "  - prompt tokens: (1, 2)",
                        "  - answer tokens: ((3,), (4,), (5,), (6,))",
                    )
                )
            blocks.append("\n".join(lines))
    return "# Opened audit\n\n" + "\n\n".join(blocks) + "\n"


def _results() -> tuple[SemanticQueryResult, ...]:
    rows = []
    final_stage = len(_CONCEPTS)
    for concept_index, concept in enumerate(_CONCEPTS, start=1):
        for fact_number in range(1, 13):
            fact_id = f"{concept}-fact-{fact_number:02d}"
            for query_number, direction in (
                (0, "forward"),
                (1, "forward"),
                (2, "forward"),
                (3, "reverse"),
                (4, "reverse"),
            ):
                template_id = f"{fact_id}-test-{query_number:02d}"
                for method in _METHODS:
                    rows.append(
                        _result(
                            stage=0 if method == "base" else final_stage,
                            method=method,
                            concept=concept,
                            fact_id=fact_id,
                            template_id=template_id,
                            direction=direction,
                            adapter_concept_id=(
                                concept if method == "independent" else None
                            ),
                        )
                    )
                for adapter_concept in _CONCEPTS:
                    if adapter_concept != concept:
                        rows.append(
                            _result(
                                stage=final_stage,
                                method="independent",
                                concept=concept,
                                fact_id=fact_id,
                                template_id=template_id,
                                direction=direction,
                                adapter_concept_id=adapter_concept,
                            )
                        )
                if concept_index < final_stage:
                    for method in ("independent", "sequential"):
                        rows.append(
                            _result(
                                stage=concept_index,
                                method=method,
                                concept=concept,
                                fact_id=fact_id,
                                template_id=template_id,
                                direction=direction,
                                adapter_concept_id=(
                                    concept if method == "independent" else None
                                ),
                            )
                        )
    return tuple(rows)


def _result(
    *,
    stage: int,
    method: str,
    concept: str,
    fact_id: str,
    template_id: str,
    direction: str,
    adapter_concept_id: str | None,
) -> SemanticQueryResult:
    oracle_node_index = _CONCEPTS.index(concept) + 1
    return SemanticQueryResult(
        stage=stage,
        method=method,
        concept_id=concept,
        fact_id=fact_id,
        template_id=template_id,
        direction=direction,
        split="test",
        adapter_concept_id=adapter_concept_id,
        candidate_nll=(0.0, 1.0, 2.0, 3.0),
        correct_candidate_index=0,
        predicted_candidate_index=0,
        answer_correct=True,
        correct_answer_margin=1.0,
        selected_node_index=(
            oracle_node_index
            if method.startswith("vamp_") and method != "vamp_oracle"
            else None
        ),
        oracle_node_index=oracle_node_index,
        routed_regret=None,
    )


def _result_record() -> dict[str, object]:
    return {
        "report_sha256": "1" * 64,
        "catalog_sha256": "2" * 64,
        "transaction_sha256": "3" * 64,
        "results_sha256": "4" * 64,
        "runtime_seconds": {"sealed_evaluation": 60.0},
        "memory_bytes": {"allocator_peak": 1_024, "allocator_limit": 2_048},
    }


def test_opened_audit_builds_forward_only_foldable_report() -> None:
    audit = _audit()
    facts, queries = parse_opened_catalog_audit(audit)
    assert len(facts) == 24
    assert len(queries) == 192

    data = build_interactive_report_data(
        audit,
        _results(),
        _result_record(),
        _CONCEPTS,
    )
    assert data.source_question_count == 120
    assert data.question_count == 72
    assert data.excluded_reverse_question_count == 48
    assert data.sample_count == 24
    assert {case.direction for case in data.cases} == {"forward"}
    assert all("already_knew" in case.tags for case in data.cases)
    assert len(data.router_comparisons) == 5
    assert all(item.node_accuracy == 1.0 for item in data.router_comparisons)

    html = render_interactive_report(data)
    assert html.startswith("<!doctype html>")
    assert '<script type="application/json" id="report-data">' in html
    assert 'class="chapter"' in html
    assert 'id="case-list"' in html
    assert 'id="direction-filter"' not in html
    assert 'id="router-table"' in html
    assert "Explore 60 forward test cases" in html
    assert "</script><script> marker" not in html
    assert "\\u003c/script\\u003e\\u003cscript\\u003e marker" in html


def test_interactive_report_rejects_audit_ledger_disagreement() -> None:
    results = _results()
    tampered = (replace(results[0], fact_id="cat-fact-02"), *results[1:])
    with pytest.raises(ValueError, match="audit questions disagree"):
        build_interactive_report_data(
            _audit(),
            tampered,
            _result_record(),
            _CONCEPTS,
        )

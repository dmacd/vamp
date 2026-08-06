"""Build the post-result interactive TinyWorlds-Q presentation report."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import exp
import os
from pathlib import Path
import re

from apm.data.text.tinyworlds_q_semantic.contracts import SemanticQueryResult
from apm.data.text.tinyworlds_q_semantic.statistics import (
    CANONICAL_BOOTSTRAP_REPLICATES,
    BootstrapEstimate,
    FactObservation,
    average_direction_paraphrases,
    bootstrap_fact_metric,
    paired_fact_effect,
    specificity_effect,
)


@dataclass(frozen=True, slots=True)
class AuditFact:
    """One reviewed fact recovered from the published opened-test audit."""

    concept_id: str
    fact_id: str
    relation: str
    relation_label: str
    answer_type: str
    answer: str
    accepted_forms: tuple[str, ...]
    triggers: tuple[str, ...]
    evidence: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AuditQuery:
    """One question recovered from the published opened-test audit."""

    concept_id: str
    fact_id: str
    template_id: str
    split: str
    direction: str
    prompt: str
    candidates: tuple[str, str, str, str]
    correct_index: int


@dataclass(frozen=True, slots=True)
class MethodDefinition:
    """Plain-language presentation metadata for one evaluated method."""

    method_id: str
    label: str
    short_label: str
    description: str


@dataclass(frozen=True, slots=True)
class CaseMethodResult:
    """One method's answer and within-question preference for an explorer case."""

    method_id: str
    predicted_index: int
    answer_correct: bool
    margin: float
    choice_shares: tuple[float, float, float, float]
    selected_node: str | None
    oracle_node: str


@dataclass(frozen=True, slots=True)
class ExplorerCase:
    """One balanced test example with all final method outcomes."""

    template_id: str
    concept_id: str
    concept_label: str
    fact_id: str
    relation: str
    relation_label: str
    answer: str
    direction: str
    direction_label: str
    prompt: str
    candidates: tuple[str, str, str, str]
    correct_index: int
    evidence: tuple[str, ...]
    tags: tuple[str, ...]
    takeaway: str
    methods: tuple[CaseMethodResult, ...]


@dataclass(frozen=True, slots=True)
class HeadlineMetric:
    """One final accuracy shown in the presentation overview."""

    metric_id: str
    label: str
    value: float
    lower: float
    upper: float
    note: str


@dataclass(frozen=True, slots=True)
class EffectStory:
    """One change-over-time result translated into a short narrative."""

    effect_id: str
    label: str
    value: float
    lower: float
    upper: float
    note: str


@dataclass(frozen=True, slots=True)
class WorldResult:
    """Final question accuracy for one world and method."""

    concept_id: str
    method_id: str
    accuracy: float


@dataclass(frozen=True, slots=True)
class RouterComparison:
    """Forward-only node selection and answer accuracy for one task-free router."""

    method_id: str
    label: str
    node_accuracy: float
    node_lower: float
    node_upper: float
    answer_accuracy: float
    answer_lower: float
    answer_upper: float


@dataclass(frozen=True, slots=True)
class InteractiveReportData:
    """Complete deterministic payload embedded into the standalone page."""

    report_sha256: str
    catalog_sha256: str
    transaction_sha256: str
    result_ledger_sha256: str
    audit_sha256: str
    source_question_count: int
    question_count: int
    excluded_reverse_question_count: int
    sample_count: int
    fact_count: int
    methods: tuple[MethodDefinition, ...]
    facts: tuple[AuditFact, ...]
    cases: tuple[ExplorerCase, ...]
    headlines: tuple[HeadlineMetric, ...]
    effects: tuple[EffectStory, ...]
    world_results: tuple[WorldResult, ...]
    router_comparisons: tuple[RouterComparison, ...]
    tag_counts: tuple[tuple[str, int], ...]
    runtime_seconds: float
    allocator_peak_bytes: int
    allocator_limit_bytes: int


_METHODS = (
    MethodDefinition(
        "base",
        "Base model",
        "Base",
        "The freshly trained story model before any fact-specific adapter is attached.",
    ),
    MethodDefinition(
        "independent",
        "Independent LoRA",
        "Independent",
        "The matching world's small adapter is supplied directly; no routing is needed.",
    ),
    MethodDefinition(
        "sequential",
        "Sequential LoRA",
        "Sequential",
        "One adapter is repeatedly overwritten as cat, dog, bird, robot, and dragon arrive.",
    ),
    MethodDefinition(
        "vamp_oracle",
        "VAMP with the right node",
        "VAMP · right node",
        "The graph is told which world the question belongs to. This isolates stored knowledge from routing.",
    ),
    MethodDefinition(
        "vamp_exhaustive",
        "VAMP exhaustive router",
        "VAMP · exhaustive",
        "Every available node is checked, then the router chooses one without seeing the answer.",
    ),
    MethodDefinition(
        "vamp_hopfield",
        "VAMP content router",
        "VAMP · content",
        "A content-addressed memory chooses a node from the wording of the question.",
    ),
    MethodDefinition(
        "vamp_ebt_uniform",
        "VAMP EBT, uniform start",
        "VAMP · EBT uniform",
        "Energy-based routing starts with equal weight on the available nodes.",
    ),
    MethodDefinition(
        "vamp_ebt_hopfield",
        "VAMP EBT, content start",
        "VAMP · EBT content",
        "Energy-based routing starts from the content router's suggestion.",
    ),
    MethodDefinition(
        "vamp_random",
        "VAMP random node",
        "VAMP · random",
        "A deterministic random node is used as a sanity check.",
    ),
)

_METHOD_IDS = tuple(item.method_id for item in _METHODS)
_RELATION_LABELS = {
    "anatomy": "body",
    "appearance": "appearance",
    "behavior": "behavior",
    "diet": "food",
    "fantasy": "storybook trait",
    "function": "what it does",
    "habitat": "home",
    "locomotion": "movement",
    "reproduction": "young and life cycle",
    "taxonomy": "what it is",
    "vocalization": "sound",
}
_TAG_ORDER = (
    "learned",
    "still_missed",
    "already_knew",
    "regressed",
    "sequential_loss",
    "routing_miss",
    "clear_win",
    "close_call",
)


def parse_opened_catalog_audit(
    markdown: str,
) -> tuple[tuple[AuditFact, ...], tuple[AuditQuery, ...]]:
    """Recover reviewed facts and questions from the transaction-published audit."""
    lines = markdown.splitlines()
    fact_starts = tuple(
        index for index, line in enumerate(lines) if line.startswith("### ")
    )
    facts_and_queries = tuple(
        _parse_fact_block(
            lines[start : fact_starts[offset + 1] if offset + 1 < len(fact_starts) else len(lines)]
        )
        for offset, start in enumerate(fact_starts)
    )
    facts = tuple(item[0] for item in facts_and_queries)
    queries = tuple(query for item in facts_and_queries for query in item[1])
    if not facts or not queries:
        raise ValueError("opened catalog audit contains no facts or questions")
    if len({item.fact_id for item in facts}) != len(facts):
        raise ValueError("opened catalog audit repeats a fact")
    if len({item.template_id for item in queries}) != len(queries):
        raise ValueError("opened catalog audit repeats a question")
    return facts, queries


def build_interactive_report_data(
    audit_markdown: str,
    results: tuple[SemanticQueryResult, ...],
    result_record: dict[str, object],
    concept_ids: tuple[str, ...],
) -> InteractiveReportData:
    """Build the forward-only one-question-per-fact presentation view."""
    facts, queries = parse_opened_catalog_audit(audit_markdown)
    fact_by_id = {item.fact_id: item for item in facts}
    test_queries = tuple(item for item in queries if item.split == "test")
    if (
        tuple(dict.fromkeys(item.concept_id for item in facts)) != concept_ids
        or len(facts) != 12 * len(concept_ids)
        or len(test_queries) != 5 * len(facts)
    ):
        raise ValueError("opened catalog audit does not match the five-query fact protocol")
    forward_queries = tuple(
        item for item in test_queries if item.direction == "forward"
    )
    reverse_queries = tuple(
        item for item in test_queries if item.direction == "reverse"
    )
    if len(forward_queries) != 3 * len(facts) or len(reverse_queries) != 2 * len(facts):
        raise ValueError("opened catalog audit does not match the directional protocol")
    primary_results = _primary_result_index(results, forward_queries, concept_ids)
    selected_queries = tuple(
        item
        for item in forward_queries
        if item.template_id.endswith("-test-00")
    )
    if len(selected_queries) != len(facts):
        raise ValueError("interactive sample must contain one forward query per fact")
    cases = tuple(
        _explorer_case(
            query,
            fact_by_id[query.fact_id],
            primary_results,
            concept_ids,
        )
        for query in selected_queries
    )
    forward_observations = average_direction_paraphrases(results, "forward")
    runtime = _required_record(result_record, "runtime_seconds")
    memory = _required_record(result_record, "memory_bytes")
    return InteractiveReportData(
        report_sha256=_required_string(result_record, "report_sha256"),
        catalog_sha256=_required_string(result_record, "catalog_sha256"),
        transaction_sha256=_required_string(result_record, "transaction_sha256"),
        result_ledger_sha256=_required_string(result_record, "results_sha256"),
        audit_sha256=sha256(audit_markdown.encode("utf-8")).hexdigest(),
        source_question_count=len(test_queries),
        question_count=len(forward_queries),
        excluded_reverse_question_count=len(reverse_queries),
        sample_count=len(cases),
        fact_count=len(facts),
        methods=_METHODS,
        facts=facts,
        cases=cases,
        headlines=_headline_metrics(forward_observations, len(concept_ids)),
        effects=_effect_stories(forward_observations, concept_ids),
        world_results=_world_results(forward_queries, primary_results, concept_ids),
        router_comparisons=_router_comparisons(
            forward_observations,
            len(concept_ids),
        ),
        tag_counts=tuple(
            (tag, sum(tag in case.tags for case in cases)) for tag in _TAG_ORDER
        ),
        runtime_seconds=_required_number(runtime, "sealed_evaluation"),
        allocator_peak_bytes=_required_integer(memory, "allocator_peak"),
        allocator_limit_bytes=_required_integer(memory, "allocator_limit"),
    )


def render_interactive_report(data: InteractiveReportData) -> str:
    """Render one dependency-free HTML document with the report payload embedded."""
    payload = json.dumps(
        asdict(data),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    safe_payload = (
        payload.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )
    return (
        _HTML_TEMPLATE.replace("__REPORT_DATA__", safe_payload)
        .replace("__REPORT_SHA256__", data.report_sha256)
        .replace("__CATALOG_SHA256__", data.catalog_sha256)
    )


def publish_interactive_report(
    data: InteractiveReportData,
    output_path: str | Path,
) -> Path:
    """Atomically publish the deterministic standalone presentation page."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = render_interactive_report(data).encode("utf-8")
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, destination)
    return destination.resolve()


def _parse_fact_block(
    lines: list[str],
) -> tuple[AuditFact, tuple[AuditQuery, ...]]:
    fact_id = lines[0].removeprefix("### ").strip()
    concept_id = fact_id.split("-fact-", maxsplit=1)[0]
    block = "\n".join(lines)
    relation_match = re.search(
        r"^Relation: `([^`]+)`; answer type: `([^`]+)`",
        block,
        re.MULTILINE,
    )
    answer_match = re.search(r"^Answer: `([^`]+)`", block, re.MULTILINE)
    accepted_match = re.search(r"^Accepted: (.+?)\s*$", block, re.MULTILINE)
    triggers_match = re.search(r"^Triggers: (.+?)\s*$", block, re.MULTILINE)
    if None in (relation_match, answer_match, accepted_match, triggers_match):
        raise ValueError(f"opened audit fact metadata is incomplete: {fact_id}")
    assert relation_match is not None
    assert answer_match is not None
    assert accepted_match is not None
    assert triggers_match is not None
    relation = relation_match.group(1)
    evidence = tuple(
        match.group(1).strip()
        for line in lines
        for match in (
            re.match(
                r"^- `[0-9a-f]{64}` / `[^`]+` sentence \d+: (.+)$",
                line,
            ),
        )
        if match is not None and match.group(1).strip()
    )
    queries = tuple(
        _parse_query(lines, index, fact_id, concept_id)
        for index, line in enumerate(lines)
        if re.match(r"^- `[^`]+` \((validation|test), (forward|reverse)\)$", line)
    )
    return (
        AuditFact(
            concept_id=concept_id,
            fact_id=fact_id,
            relation=relation,
            relation_label=_RELATION_LABELS.get(
                relation,
                relation.replace("-", " ").replace("_", " "),
            ),
            answer_type=relation_match.group(2),
            answer=answer_match.group(1),
            accepted_forms=tuple(re.findall(r"`([^`]+)`", accepted_match.group(1))),
            triggers=tuple(re.findall(r"`([^`]+)`", triggers_match.group(1))),
            evidence=evidence[:3],
        ),
        queries,
    )


def _parse_query(
    lines: list[str],
    start: int,
    fact_id: str,
    concept_id: str,
) -> AuditQuery:
    heading = re.fullmatch(
        r"- `([^`]+)` \((validation|test), (forward|reverse)\)",
        lines[start],
    )
    if heading is None:
        raise ValueError(f"invalid audit question heading in {fact_id}")
    fields = {
        key: value
        for line in lines[start + 1 : start + 7]
        for key, prefix in (
            ("prompt", "  - prompt: "),
            ("candidates", "  - candidates: "),
            ("correct", "  - correct position: "),
        )
        if line.startswith(prefix)
        for value in (line.removeprefix(prefix),)
    }
    if set(fields) != {"prompt", "candidates", "correct"}:
        raise ValueError(f"audit question fields are incomplete: {heading.group(1)}")
    candidates_value = ast.literal_eval(fields["candidates"])
    if (
        type(candidates_value) is not tuple
        or len(candidates_value) != 4
        or any(type(item) is not str or not item for item in candidates_value)
    ):
        raise ValueError(f"audit candidates changed: {heading.group(1)}")
    correct_index = int(fields["correct"])
    if not 0 <= correct_index < 4:
        raise ValueError(f"audit correct position changed: {heading.group(1)}")
    return AuditQuery(
        concept_id=concept_id,
        fact_id=fact_id,
        template_id=heading.group(1),
        split=heading.group(2),
        direction=heading.group(3),
        prompt=fields["prompt"],
        candidates=candidates_value,  # type: ignore[arg-type]
        correct_index=correct_index,
    )


def _primary_result_index(
    results: tuple[SemanticQueryResult, ...],
    test_queries: tuple[AuditQuery, ...],
    concept_ids: tuple[str, ...],
) -> dict[tuple[str, str], SemanticQueryResult]:
    template_ids = frozenset(item.template_id for item in test_queries)
    final_stage = len(concept_ids)

    def is_primary(row: SemanticQueryResult) -> bool:
        if row.template_id not in template_ids or row.split != "test":
            return False
        if row.method == "base":
            return row.stage == 0 and row.adapter_concept_id is None
        if row.stage != final_stage:
            return False
        if row.method == "independent":
            return row.adapter_concept_id == row.concept_id
        return row.adapter_concept_id is None and row.method in _METHOD_IDS

    primary_rows = tuple(row for row in results if is_primary(row))
    index = {(row.template_id, row.method): row for row in primary_rows}
    expected_count = len(test_queries) * len(_METHOD_IDS)
    if len(primary_rows) != len(index) or len(index) != expected_count:
        raise ValueError("final report does not contain one primary row per question and method")
    query_by_id = {item.template_id: item for item in test_queries}
    if any(
        (
            row.concept_id,
            row.fact_id,
            row.direction,
            row.correct_candidate_index,
        )
        != (
            query_by_id[row.template_id].concept_id,
            query_by_id[row.template_id].fact_id,
            query_by_id[row.template_id].direction,
            query_by_id[row.template_id].correct_index,
        )
        for row in primary_rows
    ):
        raise ValueError("opened audit questions disagree with the final result ledger")
    return index


def _explorer_case(
    query: AuditQuery,
    fact: AuditFact,
    result_index: dict[tuple[str, str], SemanticQueryResult],
    concept_ids: tuple[str, ...],
) -> ExplorerCase:
    methods = tuple(
        _case_method_result(result_index[(query.template_id, method_id)], concept_ids)
        for method_id in _METHOD_IDS
    )
    by_method = {item.method_id: item for item in methods}
    base = by_method["base"]
    independent = by_method["independent"]
    sequential = by_method["sequential"]
    oracle = by_method["vamp_oracle"]
    routed = by_method["vamp_ebt_hopfield"]
    tags = tuple(
        tag
        for tag, condition in (
            ("learned", not base.answer_correct and independent.answer_correct),
            ("still_missed", not base.answer_correct and not independent.answer_correct),
            ("already_knew", base.answer_correct and independent.answer_correct),
            ("regressed", base.answer_correct and not independent.answer_correct),
            ("sequential_loss", independent.answer_correct and not sequential.answer_correct),
            (
                "routing_miss",
                oracle.answer_correct
                and not routed.answer_correct
                and routed.selected_node != routed.oracle_node,
            ),
            ("clear_win", independent.answer_correct and independent.margin >= 0.75),
            ("close_call", abs(independent.margin) < 0.15),
        )
        if condition
    )
    takeaway = _case_takeaway(query, base, independent, sequential, oracle, routed)
    return ExplorerCase(
        template_id=query.template_id,
        concept_id=query.concept_id,
        concept_label=query.concept_id.title(),
        fact_id=query.fact_id,
        relation=fact.relation,
        relation_label=fact.relation_label,
        answer=fact.answer,
        direction=query.direction,
        direction_label=(
            "Concept → fact" if query.direction == "forward" else "Fact → concept"
        ),
        prompt=query.prompt,
        candidates=query.candidates,
        correct_index=query.correct_index,
        evidence=fact.evidence[:2],
        tags=tags,
        takeaway=takeaway,
        methods=methods,
    )


def _case_method_result(
    row: SemanticQueryResult,
    concept_ids: tuple[str, ...],
) -> CaseMethodResult:
    weights = tuple(exp(-(value - min(row.candidate_nll))) for value in row.candidate_nll)
    total = sum(weights)
    node_names = ("base", *concept_ids)
    if row.oracle_node_index is None or not 0 <= row.oracle_node_index < len(node_names):
        raise ValueError("interactive result lacks its task-oracle node")
    if row.selected_node_index is not None and not 0 <= row.selected_node_index < len(node_names):
        raise ValueError("interactive result selected-node index is outside the graph")
    return CaseMethodResult(
        method_id=row.method,
        predicted_index=row.predicted_candidate_index,
        answer_correct=row.answer_correct,
        margin=row.correct_answer_margin,
        choice_shares=tuple(value / total for value in weights),  # type: ignore[arg-type]
        selected_node=(
            None
            if row.selected_node_index is None
            else node_names[row.selected_node_index]
        ),
        oracle_node=node_names[row.oracle_node_index],
    )


def _case_takeaway(
    query: AuditQuery,
    base: CaseMethodResult,
    independent: CaseMethodResult,
    sequential: CaseMethodResult,
    oracle: CaseMethodResult,
    routed: CaseMethodResult,
) -> str:
    if not base.answer_correct and independent.answer_correct:
        opening = "The matching adapter repairs a base-model miss."
    elif base.answer_correct and not independent.answer_correct:
        opening = "The base knew this one, but the matching adapter changed it to a wrong answer."
    elif base.answer_correct:
        opening = "Both the base and matching adapter answer correctly."
    else:
        opening = "The matching adapter still misses this question."
    observations = (
        " The continually overwritten adapter loses it."
        if independent.answer_correct and not sequential.answer_correct
        else ""
    ) + (
        " VAMP stores a correct answer, but the content-start router chooses the wrong node."
        if oracle.answer_correct
        and not routed.answer_correct
        and routed.selected_node != routed.oracle_node
        else ""
    )
    direction = (
        " This version asks for the fact."
        if query.direction == "forward"
        else " This version gives the fact and asks for the world."
    )
    return opening + observations + direction


def _headline_metrics(
    observations: tuple[FactObservation, ...],
    final_stage: int,
) -> tuple[HeadlineMetric, ...]:
    specifications = (
        (
            "base",
            "Base model",
            0,
            "Before a matching fact adapter is attached.",
        ),
        (
            "independent",
            "Independent LoRA",
            final_stage,
            "The correct world's adapter is supplied directly.",
        ),
        (
            "sequential",
            "Sequential LoRA",
            final_stage,
            "One adapter has been overwritten by all five worlds.",
        ),
        (
            "vamp_oracle",
            "VAMP, right node",
            final_stage,
            "The graph is told which memory node to use.",
        ),
        (
            "vamp_ebt_hopfield",
            "VAMP, self-routed",
            final_stage,
            "The graph must infer its node from the world named in the question.",
        ),
    )
    return tuple(
        HeadlineMetric(
            metric_id=metric_id,
            label=label,
            value=estimate.point,
            lower=estimate.lower,
            upper=estimate.upper,
            note=note,
        )
        for metric_id, label, stage, note in specifications
        for selected in (
            _primary_observations(observations, metric_id, stage),
        )
        for estimate in (
            bootstrap_fact_metric(
                selected,
                "accuracy",
                replicates=CANONICAL_BOOTSTRAP_REPLICATES,
                identity=f"forward-only-headline:{metric_id}:stage-{stage}",
            ),
        )
    )


def _effect_stories(
    observations: tuple[FactObservation, ...],
    concept_ids: tuple[str, ...],
) -> tuple[EffectStory, ...]:
    final_stage = len(concept_ids)
    base = _primary_observations(observations, "base", 0)
    independent_acquisition = _acquisition_observations(
        observations,
        "independent",
        concept_ids,
    )
    independent_final = _primary_observations(
        observations,
        "independent",
        final_stage,
    )
    sequential_acquisition = _acquisition_observations(
        observations,
        "sequential",
        concept_ids,
    )
    sequential_final = _primary_observations(
        observations,
        "sequential",
        final_stage,
    )
    specificity_rows = tuple(
        item
        for item in observations
        if item.stage == final_stage and item.method == "independent"
    )
    specifications: tuple[
        tuple[str, str, BootstrapEstimate, str],
        ...,
    ] = (
        (
            "independent_gain",
            "Knowledge added by independent adapters",
            paired_fact_effect(
                base,
                independent_acquisition,
                "accuracy",
                replicates=CANONICAL_BOOTSTRAP_REPLICATES,
                identity="forward-only-base-to-adapter-acquisition:independent",
            ),
            "How many more questions become correct when the matching adapter is attached.",
        ),
        (
            "independent_retention",
            "Independent knowledge retained",
            paired_fact_effect(
                independent_acquisition,
                independent_final,
                "accuracy",
                replicates=CANONICAL_BOOTSTRAP_REPLICATES,
                identity="forward-only-acquisition-to-final-retention:independent",
            ),
            "Change from first learning to the final five-world checkpoint; zero means no loss.",
        ),
        (
            "sequential_retention",
            "Sequential knowledge retained",
            paired_fact_effect(
                sequential_acquisition,
                sequential_final,
                "accuracy",
                replicates=CANONICAL_BOOTSTRAP_REPLICATES,
                identity="forward-only-acquisition-to-final-retention:sequential",
            ),
            "The negative value is forgetting after later worlds overwrite the same adapter.",
        ),
        (
            "specificity",
            "Right adapter versus wrong adapter",
            specificity_effect(
                specificity_rows,
                "accuracy",
                replicates=CANONICAL_BOOTSTRAP_REPLICATES,
                identity="forward-only-node-specificity",
            ),
            "The advantage of using the matching world's adapter rather than another world's.",
        ),
    )
    return tuple(
        EffectStory(
            effect_id=effect_id,
            label=label,
            value=estimate.point,
            lower=estimate.lower,
            upper=estimate.upper,
            note=note,
        )
        for effect_id, label, estimate, note in specifications
    )


def _primary_observations(
    observations: tuple[FactObservation, ...],
    method: str,
    stage: int,
) -> tuple[FactObservation, ...]:
    selected = tuple(
        item
        for item in observations
        if item.stage == stage
        and item.method == method
        and (
            item.adapter_concept_id == item.concept_id
            if method == "independent"
            else item.adapter_concept_id is None
        )
    )
    expected_facts = {
        (item.concept_id, item.fact_id) for item in observations
    }
    selected_facts = {
        (item.concept_id, item.fact_id) for item in selected
    }
    if selected_facts != expected_facts or len(selected) != len(selected_facts):
        raise ValueError(f"forward-only primary facts are incomplete for {method}")
    return selected


def _acquisition_observations(
    observations: tuple[FactObservation, ...],
    method: str,
    concept_ids: tuple[str, ...],
) -> tuple[FactObservation, ...]:
    selected = tuple(
        item
        for stage, concept_id in enumerate(concept_ids, start=1)
        for item in observations
        if item.stage == stage
        and item.method == method
        and item.concept_id == concept_id
        and (
            item.adapter_concept_id == concept_id
            if method == "independent"
            else item.adapter_concept_id is None
        )
    )
    if len(selected) != 12 * len(concept_ids):
        raise ValueError(f"forward-only acquisition facts are incomplete for {method}")
    return selected


def _router_comparisons(
    observations: tuple[FactObservation, ...],
    final_stage: int,
) -> tuple[RouterComparison, ...]:
    definitions = {item.method_id: item for item in _METHODS}
    return tuple(
        RouterComparison(
            method_id=method_id,
            label=definitions[method_id].label,
            node_accuracy=node.point,
            node_lower=node.lower,
            node_upper=node.upper,
            answer_accuracy=answer.point,
            answer_lower=answer.lower,
            answer_upper=answer.upper,
        )
        for method_id in _METHOD_IDS
        if method_id.startswith("vamp_") and method_id != "vamp_oracle"
        for selected in (
            _primary_observations(observations, method_id, final_stage),
        )
        for node, answer in (
            (
                bootstrap_fact_metric(
                    selected,
                    "router_accuracy",
                    replicates=CANONICAL_BOOTSTRAP_REPLICATES,
                    identity=f"forward-only-router-node:{method_id}",
                ),
                bootstrap_fact_metric(
                    selected,
                    "accuracy",
                    replicates=CANONICAL_BOOTSTRAP_REPLICATES,
                    identity=f"forward-only-router-answer:{method_id}",
                ),
            ),
        )
    )


def _world_results(
    queries: tuple[AuditQuery, ...],
    result_index: dict[tuple[str, str], SemanticQueryResult],
    concept_ids: tuple[str, ...],
) -> tuple[WorldResult, ...]:
    return tuple(
        WorldResult(
            concept_id=concept_id,
            method_id=method_id,
            accuracy=sum(
                result_index[(query.template_id, method_id)].answer_correct
                for query in queries
                if query.concept_id == concept_id
            )
            / sum(query.concept_id == concept_id for query in queries),
        )
        for concept_id in concept_ids
        for method_id in _METHOD_IDS
    )


def _required_record(record: dict[str, object], field: str) -> dict[str, object]:
    value = record.get(field)
    if type(value) is not dict:
        raise ValueError(f"interactive report {field} must be a record")
    return value


def _required_string(record: dict[str, object], field: str) -> str:
    value = record.get(field)
    if type(value) is not str or not value:
        raise ValueError(f"interactive report {field} must be nonempty text")
    return value


def _required_number(record: dict[str, object], field: str) -> float:
    value = record.get(field)
    if type(value) not in (int, float):
        raise ValueError(f"interactive report {field} must be numeric")
    return float(value)


def _required_integer(record: dict[str, object], field: str) -> int:
    value = record.get(field)
    if type(value) is not int:
        raise ValueError(f"interactive report {field} must be an integer")
    return value


_HTML_TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light">
  <meta name="report-sha256" content="__REPORT_SHA256__">
  <meta name="catalog-sha256" content="__CATALOG_SHA256__">
  <title>TinyWorlds-Q — forward-only routable result</title>
  <style>
    :root {
      --ink: #18302f;
      --muted: #5f6f6c;
      --paper: #f7f3e9;
      --paper-deep: #eee7d8;
      --card: #fffdf7;
      --line: #d9d1c1;
      --teal: #177e79;
      --teal-dark: #0f5c58;
      --teal-soft: #d9efeb;
      --coral: #dc6d52;
      --coral-soft: #f9dfd6;
      --gold: #d3a33f;
      --gold-soft: #f6ebc8;
      --blue: #4a6fa5;
      --blue-soft: #e1e9f5;
      --shadow: 0 16px 44px rgba(42, 54, 50, .10);
      --radius: 18px;
    }

    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at 12% 4%, rgba(211, 163, 63, .16), transparent 24rem),
        radial-gradient(circle at 88% 10%, rgba(23, 126, 121, .12), transparent 26rem),
        var(--paper);
      font: 16px/1.55 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    button, select, input { font: inherit; }
    button, select { cursor: pointer; }
    a { color: var(--teal-dark); }
    code { overflow-wrap: anywhere; }
    .wrap { width: min(1180px, calc(100% - 32px)); margin-inline: auto; }

    .topbar {
      position: sticky;
      top: 0;
      z-index: 20;
      border-bottom: 1px solid rgba(24, 48, 47, .10);
      background: rgba(247, 243, 233, .92);
      backdrop-filter: blur(12px);
    }
    .topbar-inner {
      min-height: 58px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
    }
    .brand { font-weight: 850; letter-spacing: -.02em; }
    .jump-links { display: flex; gap: 18px; align-items: center; flex-wrap: wrap; }
    .jump-links a { color: var(--ink); text-decoration: none; font-size: .9rem; font-weight: 700; }
    .utility-button {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 7px 12px;
      color: var(--ink);
      background: var(--card);
      font-weight: 750;
    }

    .hero { padding: 78px 0 46px; }
    .eyebrow {
      display: inline-flex;
      gap: 8px;
      align-items: center;
      padding: 6px 11px;
      border-radius: 999px;
      background: var(--teal-soft);
      color: var(--teal-dark);
      font-weight: 800;
      font-size: .82rem;
      letter-spacing: .04em;
      text-transform: uppercase;
    }
    h1 {
      max-width: 920px;
      margin: 20px 0 16px;
      font-family: Georgia, "Times New Roman", serif;
      font-size: clamp(2.7rem, 7vw, 5.5rem);
      line-height: .98;
      letter-spacing: -.055em;
      font-weight: 700;
    }
    .hero-lede { max-width: 790px; margin: 0; color: var(--muted); font-size: clamp(1.08rem, 2.3vw, 1.38rem); }
    .world-row { display: flex; gap: 10px; flex-wrap: wrap; margin: 28px 0; }
    .world-chip {
      min-width: 94px;
      padding: 9px 14px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: rgba(255, 253, 247, .75);
      box-shadow: 0 4px 14px rgba(42, 54, 50, .05);
      text-align: center;
      font-weight: 800;
    }
    .hero-actions { display: flex; gap: 12px; flex-wrap: wrap; }
    .primary-link, .secondary-link {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 46px;
      padding: 10px 17px;
      border-radius: 12px;
      text-decoration: none;
      font-weight: 850;
    }
    .primary-link { color: white; background: var(--teal-dark); }
    .secondary-link { color: var(--ink); border: 1px solid var(--line); background: var(--card); }

    .one-sentence {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 26px;
      align-items: center;
      margin-bottom: 34px;
      padding: 24px 28px;
      border-left: 5px solid var(--coral);
      border-radius: 0 var(--radius) var(--radius) 0;
      background: var(--card);
      box-shadow: var(--shadow);
    }
    .one-sentence strong { display: block; margin-bottom: 4px; color: var(--coral); font-size: .82rem; text-transform: uppercase; letter-spacing: .06em; }
    .one-sentence p { margin: 0; font-family: Georgia, "Times New Roman", serif; font-size: clamp(1.2rem, 2.3vw, 1.7rem); line-height: 1.35; }
    .answer-badge { min-width: 110px; font-size: 2rem; font-weight: 900; color: var(--teal-dark); text-align: center; }

    .chapter-controls { display: flex; justify-content: flex-end; gap: 8px; margin: 10px 0 16px; }
    .chapter {
      margin: 0 0 18px;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: rgba(255, 253, 247, .82);
      box-shadow: 0 8px 26px rgba(42, 54, 50, .06);
      overflow: clip;
    }
    .chapter > summary {
      display: flex;
      align-items: center;
      gap: 14px;
      min-height: 74px;
      padding: 18px 22px;
      list-style: none;
      cursor: pointer;
      font-family: Georgia, "Times New Roman", serif;
      font-size: clamp(1.25rem, 2.3vw, 1.65rem);
      font-weight: 700;
    }
    .chapter > summary::-webkit-details-marker, .case-card > summary::-webkit-details-marker { display: none; }
    .chapter > summary::before {
      content: "+";
      width: 30px;
      height: 30px;
      display: grid;
      place-items: center;
      flex: 0 0 30px;
      border-radius: 50%;
      color: white;
      background: var(--teal);
      font: 800 1.2rem/1 system-ui;
    }
    .chapter[open] > summary::before { content: "−"; }
    .chapter-body { padding: 2px 24px 28px 68px; }
    .section-intro { max-width: 790px; margin: 0 0 24px; color: var(--muted); font-size: 1.05rem; }
    h2, h3 { letter-spacing: -.02em; }
    h3 { margin: 28px 0 12px; }

    .tour-grid { display: grid; grid-template-columns: repeat(5, minmax(150px, 1fr)); gap: 12px; }
    .tour-step { padding: 17px; border-radius: 15px; background: var(--paper-deep); }
    .tour-step b { display: grid; place-items: center; width: 30px; height: 30px; margin-bottom: 11px; border-radius: 50%; background: var(--ink); color: white; }
    .tour-step strong { display: block; margin-bottom: 5px; }
    .tour-step p { margin: 0; color: var(--muted); font-size: .92rem; }

    .callout { margin: 22px 0; padding: 17px 19px; border-radius: 14px; background: var(--gold-soft); }
    .callout strong { color: #725612; }
    .disclosure { background: var(--coral-soft); }
    .disclosure strong { color: #843a29; }

    .metric-grid { display: grid; grid-template-columns: repeat(5, minmax(145px, 1fr)); gap: 12px; }
    .metric-card { min-height: 170px; padding: 18px; border-radius: 16px; background: var(--paper-deep); }
    .metric-card.featured { color: white; background: var(--teal-dark); }
    .metric-value { display: block; margin: 8px 0 3px; font-size: clamp(2rem, 4vw, 3.2rem); line-height: 1; font-weight: 900; letter-spacing: -.06em; }
    .metric-card small { display: block; color: var(--muted); }
    .metric-card.featured small { color: rgba(255,255,255,.74); }
    .metric-interval { font-size: .78rem; font-weight: 750; }

    .bar-chart { display: grid; gap: 10px; margin-top: 22px; }
    .bar-row { display: grid; grid-template-columns: minmax(150px, 220px) 1fr 62px; gap: 12px; align-items: center; }
    .bar-label { font-weight: 750; }
    .bar-track { height: 16px; border-radius: 999px; background: var(--paper-deep); overflow: hidden; }
    .bar-fill { height: 100%; border-radius: inherit; background: var(--teal); }
    .bar-fill.base { background: var(--gold); }
    .bar-fill.weak { background: var(--coral); }
    .bar-value { text-align: right; font-variant-numeric: tabular-nums; font-weight: 850; }
    .chance-line { margin: 10px 0 0 232px; color: var(--muted); font-size: .83rem; }

    .story-grid { display: grid; grid-template-columns: repeat(4, minmax(170px, 1fr)); gap: 12px; margin-top: 24px; }
    .story-card { padding: 17px; border: 1px solid var(--line); border-radius: 15px; background: var(--card); }
    .story-card .delta { display: block; margin: 7px 0; color: var(--teal-dark); font-size: 2rem; font-weight: 900; letter-spacing: -.04em; }
    .story-card.negative .delta { color: var(--coral); }
    .story-card p { margin: 4px 0 0; color: var(--muted); font-size: .9rem; }

    .world-table-wrap { overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; }
    th, td { padding: 10px 12px; border-bottom: 1px solid var(--line); text-align: left; }
    th { color: var(--muted); font-size: .8rem; text-transform: uppercase; letter-spacing: .04em; }
    td.number { text-align: right; font-variant-numeric: tabular-nums; }
    .table-note { display: block; margin-top: 2px; color: var(--muted); font-size: .75rem; font-weight: 500; text-transform: none; letter-spacing: 0; }

    .fact-atlas { display: grid; grid-template-columns: repeat(5, minmax(160px, 1fr)); gap: 12px; }
    .fact-world { border: 1px solid var(--line); border-radius: 15px; background: var(--card); overflow: hidden; }
    .fact-world summary { padding: 14px 15px; cursor: pointer; font-weight: 850; background: var(--paper-deep); }
    .fact-list { list-style: none; margin: 0; padding: 7px 15px 13px; }
    .fact-list li { padding: 9px 0; border-bottom: 1px dashed var(--line); }
    .fact-list li:last-child { border: 0; }
    .fact-list small { display: block; color: var(--muted); }

    .preset-row { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 16px; }
    .preset-button { border: 1px solid var(--line); border-radius: 999px; padding: 8px 12px; background: var(--card); color: var(--ink); font-weight: 750; }
    .preset-button:hover, .preset-button.active { color: white; border-color: var(--teal-dark); background: var(--teal-dark); }
    .filters { display: grid; grid-template-columns: repeat(4, minmax(150px, 1fr)); gap: 11px; padding: 15px; border-radius: 16px; background: var(--paper-deep); }
    .field label { display: block; margin: 0 0 5px; color: var(--muted); font-size: .78rem; font-weight: 800; text-transform: uppercase; letter-spacing: .04em; }
    .field select, .field input { width: 100%; min-height: 42px; padding: 8px 10px; border: 1px solid var(--line); border-radius: 10px; color: var(--ink); background: var(--card); }
    .explorer-status { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin: 15px 0 8px; color: var(--muted); }
    .case-list { display: grid; gap: 11px; }
    .case-card { border: 1px solid var(--line); border-radius: 15px; background: var(--card); overflow: hidden; }
    .case-card > summary { display: grid; grid-template-columns: auto minmax(0,1fr) auto; gap: 13px; align-items: center; padding: 15px 17px; cursor: pointer; list-style: none; }
    .case-card[open] > summary { border-bottom: 1px solid var(--line); background: #fffaf0; }
    .world-dot { width: 12px; height: 12px; border-radius: 50%; background: var(--teal); box-shadow: 0 0 0 4px var(--teal-soft); }
    .case-heading { min-width: 0; }
    .case-heading strong { display: block; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .case-heading small { color: var(--muted); }
    .outcome-pill { padding: 5px 9px; border-radius: 999px; font-size: .78rem; font-weight: 850; }
    .outcome-pill.correct { color: var(--teal-dark); background: var(--teal-soft); }
    .outcome-pill.wrong { color: #8c3e2c; background: var(--coral-soft); }
    .case-body { padding: 18px; }
    .takeaway { margin: 0 0 17px; padding: 13px 15px; border-radius: 12px; background: var(--blue-soft); color: #29486f; }
    .question-text { margin: 0 0 14px; font-family: Georgia, "Times New Roman", serif; font-size: 1.28rem; }
    .choice-list { display: grid; gap: 8px; }
    .choice { position: relative; display: grid; grid-template-columns: 28px minmax(110px, auto) 1fr auto; gap: 10px; align-items: center; min-height: 43px; padding: 8px 11px; border: 1px solid var(--line); border-radius: 11px; overflow: hidden; }
    .choice::before { content: ""; position: absolute; inset: 0 auto 0 0; width: var(--share); background: rgba(23, 126, 121, .10); z-index: 0; }
    .choice > * { position: relative; z-index: 1; }
    .choice.selected { border-color: var(--coral); }
    .choice.correct-answer { box-shadow: inset 4px 0 0 var(--teal); }
    .choice-letter { display: grid; place-items: center; width: 24px; height: 24px; border-radius: 50%; background: var(--paper-deep); font-weight: 850; }
    .choice-flags { color: var(--muted); font-size: .78rem; text-align: right; }
    .share-label { font-variant-numeric: tabular-nums; color: var(--muted); font-size: .78rem; }
    .case-folds { display: grid; grid-template-columns: 1fr 1fr; gap: 11px; margin-top: 16px; }
    .mini-fold { border: 1px solid var(--line); border-radius: 12px; background: #fffdf9; }
    .mini-fold summary { padding: 10px 12px; cursor: pointer; font-weight: 800; }
    .mini-fold-content { padding: 0 12px 12px; }
    .mini-fold-content p { margin: 8px 0; }
    .method-table td:first-child { font-weight: 750; }
    .method-table .correct-text { color: var(--teal-dark); font-weight: 850; }
    .method-table .wrong-text { color: #9a432f; font-weight: 850; }
    .tag-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 13px; }
    .tag { padding: 4px 8px; border-radius: 999px; background: var(--paper-deep); color: var(--muted); font-size: .74rem; font-weight: 800; }
    .load-more { display: block; margin: 16px auto 0; border: 0; border-radius: 11px; padding: 10px 17px; color: white; background: var(--teal-dark); font-weight: 850; }
    .empty-state { padding: 28px; border: 1px dashed var(--line); border-radius: 14px; text-align: center; color: var(--muted); }

    .glossary { display: grid; grid-template-columns: repeat(3, minmax(220px, 1fr)); gap: 11px; }
    .glossary-item { padding: 14px; border-left: 4px solid var(--teal); background: var(--paper-deep); }
    .glossary-item strong { display: block; margin-bottom: 4px; }
    .technical { color: var(--muted); font-size: .88rem; }
    footer { padding: 34px 0 60px; color: var(--muted); font-size: .84rem; }

    .presentation-mode { font-size: 19px; }
    .presentation-mode .technical, .presentation-mode .jump-links a:nth-child(n+4) { display: none; }
    .presentation-mode .wrap { width: min(1320px, calc(100% - 36px)); }

    @media (max-width: 980px) {
      .tour-grid, .metric-grid, .fact-atlas { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .story-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .filters { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .glossary { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .jump-links a { display: none; }
    }
    @media (max-width: 640px) {
      .hero { padding-top: 48px; }
      .one-sentence { grid-template-columns: 1fr; }
      .answer-badge { text-align: left; }
      .chapter-body { padding: 2px 15px 22px; }
      .chapter > summary { padding-inline: 15px; }
      .tour-grid, .metric-grid, .story-grid, .fact-atlas, .filters, .glossary, .case-folds { grid-template-columns: 1fr; }
      .bar-row { grid-template-columns: 115px 1fr 48px; gap: 7px; font-size: .82rem; }
      .chance-line { margin-left: 122px; }
      .case-card > summary { grid-template-columns: auto minmax(0,1fr); }
      .outcome-pill { grid-column: 2; justify-self: start; }
      .choice { grid-template-columns: 26px minmax(85px, auto) 1fr; }
      .choice-flags { grid-column: 2 / -1; text-align: left; }
    }
    @media (prefers-reduced-motion: reduce) { html { scroll-behavior: auto; } }
    @media print {
      .topbar, .hero-actions, .chapter-controls, .filters, .preset-row, .load-more { display: none !important; }
      body { background: white; }
      .chapter { break-inside: avoid; box-shadow: none; }
    }
  </style>
</head>
<body>
  <nav class="topbar" aria-label="Report sections">
    <div class="wrap topbar-inner">
      <span class="brand">TinyWorlds-Q</span>
      <div class="jump-links">
        <a href="#story">How it works</a>
        <a href="#results">Results</a>
        <a href="#explorer">Examples</a>
        <a href="#methods">Caveats</a>
        <button class="utility-button" id="presentation-toggle" type="button">Presentation mode</button>
      </div>
    </div>
  </nav>

  <header class="hero wrap">
    <span class="eyebrow">Forward-only post-result diagnostic</span>
    <h1>When the question names the world, do the adapters help?</h1>
    <p class="hero-lede">This view uses only the 180 sealed questions that explicitly say cat, dog, bird, robot, or dragon. The 120 reverse questions are excluded because some did not uniquely identify a world.</p>
    <div class="world-row" aria-label="Experiment worlds">
      <span class="world-chip">Cat</span><span class="world-chip">Dog</span><span class="world-chip">Bird</span><span class="world-chip">Robot</span><span class="world-chip">Dragon</span>
    </div>
    <div class="hero-actions">
      <a class="primary-link" href="#explorer">Explore real test cases</a>
      <a class="secondary-link" href="#results">See the main result</a>
    </div>
  </header>

  <main class="wrap">
    <section class="one-sentence" aria-label="Result in one sentence">
      <div><strong>In one sentence</strong><p>On routable forward questions, matching adapters moved answer accuracy from about 43% to 53%; self-routed VAMP reached about 52% and selected the named world about 77% of the time.</p></div>
      <div class="answer-badge" id="headline-change">—</div>
    </section>

    <div class="chapter-controls">
      <button class="utility-button" id="open-all" type="button">Open all chapters</button>
      <button class="utility-button" id="fold-all" type="button">Fold all</button>
    </div>

    <details class="chapter" id="story">
      <summary>How the experiment works</summary>
      <div class="chapter-body">
        <p class="section-intro">The goal is simple: separate ordinary word familiarity from knowledge of a specific fact. The base model can see the word “cat,” but it cannot see registered stories saying, for example, that cats have fur.</p>
        <div class="tour-grid">
          <div class="tour-step"><b>1</b><strong>Review facts</strong><p>Twelve human-reviewed facts for each of five worlds.</p></div>
          <div class="tour-step"><b>2</b><strong>Withhold them</strong><p>Every story expressing one of those facts is removed from base training.</p></div>
          <div class="tour-step"><b>3</b><strong>Train adapters</strong><p>Small rank-eight LoRAs read the fact-bearing stories; word embeddings stay frozen.</p></div>
          <div class="tour-step"><b>4</b><strong>Freeze choices</strong><p>Model settings and routing keys are fixed using validation questions only.</p></div>
          <div class="tour-step"><b>5</b><strong>Keep routable questions</strong><p>Three unseen forward phrasings per fact; every prompt names its world.</p></div>
        </div>
        <div class="callout"><strong>What counts as success?</strong> The model scores four answer choices. Only the answer words count—not the wording of the question. Random guessing would average 25 correct answers out of 100.</div>
        <div class="callout disclosure"><strong>Post-result disclosure:</strong> the registered 300-question result remains immutable. After we found that some reverse prompts did not identify a unique world, this derived view excluded all 120 reverse questions by one uniform rule. It is an honest directional diagnostic, not a replacement preregistered result, and it was not used to tune any model.</div>
        <h3>The 60 reviewed facts</h3>
        <p class="section-intro">Open a world to see all twelve knowledge targets. These are the human-approved meanings that the benchmark treats as ground truth.</p>
        <div class="fact-atlas" id="fact-atlas"></div>
      </div>
    </details>

    <details class="chapter" id="results" open>
      <summary>What happened</summary>
      <div class="chapter-body">
        <p class="section-intro">Among the questions that name their world, the base answered 42.8% correctly. Matching independent LoRAs reached 53.3%, VAMP with the right node supplied reached 52.8%, and VAMP choosing its own node reached 51.7%.</p>
        <div class="metric-grid" id="headline-metrics"></div>
        <div class="bar-chart" id="headline-bars" aria-label="Final test accuracy by method"></div>
        <p class="chance-line">Random four-choice guessing: 25%</p>
        <div class="story-grid" id="effect-stories"></div>
        <div class="callout"><strong>What does “points” mean?</strong> It is the direct gap between two percentages. Moving from 42.8% correct to 53.3% correct is a gain of 10.6 percentage points. That is different from saying accuracy grew by 10.6 percent.</div>
        <h3>Can VAMP find the world named in the question?</h3>
        <p class="section-intro">These two columns separate routing from answering. “Right memory” asks whether the router selected the node for the named world. “Correct answer” asks whether the answer ranked first after that selection.</p>
        <div class="world-table-wrap"><table id="router-table"><thead><tr><th>Router</th><th>Right memory<span class="table-note">world node selected</span></th><th>Correct answer<span class="table-note">after routing</span></th></tr></thead><tbody></tbody></table></div>
        <div class="callout"><strong>Plain-English routing result:</strong> the two EBT routers selected the named world 77.2% of the time, while routed answer accuracy was 51.7%—only 1.1 points below the 52.8% right-node result. On these forward prompts, routing still leaves room to improve, but it is no longer the main explanation for most wrong answers.</div>
        <h3>Did every world behave the same way?</h3>
        <p class="section-intro">No. This forward-only table keeps the worlds separate so an overall average cannot hide an easier or harder concept family.</p>
        <div class="world-table-wrap"><table id="world-table"><thead><tr><th>World</th><th>Base</th><th>Independent</th><th>Sequential</th><th>VAMP, right node</th><th>VAMP, self-routed</th></tr></thead><tbody></tbody></table></div>
        <div class="callout"><strong>Plain-English reading:</strong> the matching adapters add useful, world-specific knowledge. Continually overwriting one adapter causes forgetting. VAMP preserves nearly all of the right-node answer performance when the question itself says which world it concerns.</div>
      </div>
    </details>

    <details class="chapter" id="explorer" open>
      <summary>Explore 60 forward test cases</summary>
      <div class="chapter-body">
        <p class="section-intro">This is a fixed sample, not a highlight reel: forward test question 00 for every one of the 60 reviewed facts. Every prompt names its world. Open any case to compare all nine methods and see the source evidence.</p>
        <div class="preset-row" aria-label="Example presets">
          <button class="preset-button active" data-preset="all" type="button">Balanced sample</button>
          <button class="preset-button" data-preset="learned" type="button">Learned after adapter</button>
          <button class="preset-button" data-preset="still_missed" type="button">Still missed</button>
          <button class="preset-button" data-preset="sequential_loss" type="button">Lost in sequence</button>
          <button class="preset-button" data-preset="routing_miss" type="button">Router picked wrong memory</button>
        </div>
        <div class="filters">
          <div class="field"><label for="world-filter">World</label><select id="world-filter"><option value="all">All five worlds</option></select></div>
          <div class="field"><label for="relation-filter">Kind of fact</label><select id="relation-filter"><option value="all">All kinds</option></select></div>
          <div class="field"><label for="method-filter">Answer shown</label><select id="method-filter"></select></div>
          <div class="field"><label for="search-filter">Find text</label><input id="search-filter" type="search" placeholder="fur, fly, dragon…"></div>
        </div>
        <div class="explorer-status"><strong id="case-count"></strong><span id="sample-note">One fixed forward question per fact</span></div>
        <div class="case-list" id="case-list"></div>
        <button class="load-more" id="load-more" type="button">Show 12 more</button>
      </div>
    </details>

    <details class="chapter" id="methods">
      <summary>Method guide and caveats</summary>
      <div class="chapter-body">
        <p class="section-intro">These labels are easy to blur together. This guide separates “what knowledge exists” from “how the system decides where to look.”</p>
        <div class="glossary" id="method-glossary"></div>
        <h3>What this result does—and does not—show</h3>
        <ul>
          <li>It shows that, on forward questions naming their world, small LoRA adapters improve direct answer ranking for deliberately withheld facts by 10.6 points, with a fact-resampled 95% interval from 3.3 to 17.8 points.</li>
          <li>It shows that the EBT router selected the named world 77.2% of the time and answered 51.7% correctly, compared with 52.8% when the correct node was supplied.</li>
          <li>It excludes reverse questions because some of them do not uniquely identify a world; it makes no claim about solving that ambiguous routing task.</li>
          <li>It is a post-result directional analysis. The original full test, including its reverse questions, remains the registered primary result.</li>
          <li>It is one seed, one architecture, five reviewed concept families, and a descriptive result—not a universal pass/fail claim.</li>
        </ul>
      </div>
    </details>

    <details class="chapter technical" id="provenance">
      <summary>Reproducibility and exact identities</summary>
      <div class="chapter-body">
        <p>This presentation is a deterministic forward-only view of the already-completed transaction. It reads the transaction-published opened audit and final JSONL ledger; it does not rerun a model, change a stored result, or reopen the sealed catalog loader.</p>
        <table><tbody>
          <tr><th>Final report</th><td><code id="report-id"></code></td></tr>
          <tr><th>Catalog</th><td><code id="catalog-id"></code></td></tr>
          <tr><th>Transaction</th><td><code id="transaction-id"></code></td></tr>
          <tr><th>Result ledger</th><td><code id="ledger-id"></code></td></tr>
          <tr><th>Opened audit</th><td><code id="audit-id"></code></td></tr>
          <tr><th>Source test</th><td id="source-question-total"></td></tr>
          <tr><th>Included here</th><td id="question-total"></td></tr>
          <tr><th>Excluded here</th><td id="excluded-question-total"></td></tr>
          <tr><th>Explorer sample</th><td id="sample-total"></td></tr>
          <tr><th>Sealed evaluation time</th><td id="runtime"></td></tr>
          <tr><th>Peak GPU allocation</th><td id="memory"></td></tr>
        </tbody></table>
      </div>
    </details>
  </main>

  <footer class="wrap">TinyWorlds-Q Semantic-v1 · forward-only post-result diagnostic · original registered result unchanged</footer>

  <script type="application/json" id="report-data">__REPORT_DATA__</script>
  <script>
    (() => {
      "use strict";
      const data = JSON.parse(document.getElementById("report-data").textContent);
      const $ = selector => document.querySelector(selector);
      const $$ = selector => [...document.querySelectorAll(selector)];
      const node = (tag, className, text) => {
        const item = document.createElement(tag);
        if (className) item.className = className;
        if (text !== undefined) item.textContent = text;
        return item;
      };
      const percent = value => `${(100 * value).toFixed(1)}%`;
      const points = value => `${value >= 0 ? "+" : ""}${(100 * value).toFixed(1)} points`;
      const methodById = Object.fromEntries(data.methods.map(item => [item.method_id, item]));
      const resultFor = (caseItem, methodId) => caseItem.methods.find(item => item.method_id === methodId);
      const tagLabels = {
        learned: "adapter learned it",
        still_missed: "still missed",
        already_knew: "base already knew it",
        regressed: "adapter regression",
        sequential_loss: "lost in sequence",
        routing_miss: "routing miss",
        clear_win: "clear adapter win",
        close_call: "close call"
      };

      const renderFactAtlas = () => {
        const root = $("#fact-atlas");
        [...new Set(data.facts.map(item => item.concept_id))].forEach(world => {
          const fold = node("details", "fact-world");
          const summary = node("summary", "", `${world[0].toUpperCase()}${world.slice(1)} · 12 facts`);
          const list = node("ul", "fact-list");
          data.facts.filter(item => item.concept_id === world).forEach(fact => {
            const row = node("li");
            row.append(node("strong", "", fact.answer), node("small", "", `${fact.relation_label} · ${fact.fact_id}`));
            list.append(row);
          });
          fold.append(summary, list);
          root.append(fold);
        });
      };

      const renderResults = () => {
        const cards = $("#headline-metrics");
        data.headlines.forEach(metric => {
          const card = node("article", `metric-card ${metric.metric_id === "independent" ? "featured" : ""}`);
          card.append(
            node("strong", "", metric.label),
            node("span", "metric-value", percent(metric.value)),
            node("span", "metric-interval", `Likely range ${percent(metric.lower)}–${percent(metric.upper)}`),
            node("small", "", metric.note)
          );
          cards.append(card);
        });
        const bars = $("#headline-bars");
        data.headlines.forEach(metric => {
          const row = node("div", "bar-row");
          const track = node("div", "bar-track");
          const fill = node("div", `bar-fill ${metric.metric_id === "base" ? "base" : metric.value < .5 ? "weak" : ""}`);
          fill.style.width = percent(metric.value);
          track.append(fill);
          row.append(node("span", "bar-label", metric.label), track, node("span", "bar-value", percent(metric.value)));
          bars.append(row);
        });
        const base = data.headlines.find(item => item.metric_id === "base");
        const independent = data.headlines.find(item => item.metric_id === "independent");
        $("#headline-change").textContent = `${Math.round(100 * base.value)}% → ${Math.round(100 * independent.value)}%`;
        const stories = $("#effect-stories");
        data.effects.forEach(effect => {
          const card = node("article", `story-card ${effect.value < 0 ? "negative" : ""}`);
          card.append(
            node("strong", "", effect.label),
            node("span", "delta", points(effect.value)),
            node("small", "", `Range ${points(effect.lower)} to ${points(effect.upper)}`),
            node("p", "", effect.note)
          );
          stories.append(card);
        });
        const shownMethods = ["base", "independent", "sequential", "vamp_oracle", "vamp_ebt_hopfield"];
        const body = $("#world-table tbody");
        [...new Set(data.world_results.map(item => item.concept_id))].forEach(world => {
          const row = node("tr");
          row.append(node("td", "", `${world[0].toUpperCase()}${world.slice(1)}`));
          shownMethods.forEach(method => {
            const value = data.world_results.find(item => item.concept_id === world && item.method_id === method).accuracy;
            row.append(node("td", "number", percent(value)));
          });
          body.append(row);
        });
        const routerBody = $("#router-table tbody");
        data.router_comparisons.forEach(router => {
          const row = node("tr");
          const nodeCell = node("td", "number", percent(router.node_accuracy));
          nodeCell.append(node("span", "table-note", `${percent(router.node_lower)}–${percent(router.node_upper)}`));
          const answerCell = node("td", "number", percent(router.answer_accuracy));
          answerCell.append(node("span", "table-note", `${percent(router.answer_lower)}–${percent(router.answer_upper)}`));
          row.append(node("td", "", router.label), nodeCell, answerCell);
          routerBody.append(row);
        });
      };

      const state = { preset: "all", world: "all", relation: "all", method: "independent", search: "", limit: 12 };
      const controls = {
        world: $("#world-filter"), relation: $("#relation-filter"),
        method: $("#method-filter"), search: $("#search-filter")
      };

      const fillControls = () => {
        [...new Set(data.cases.map(item => item.concept_id))].forEach(world => controls.world.add(new Option(`${world[0].toUpperCase()}${world.slice(1)}`, world)));
        [...new Map(data.cases.map(item => [item.relation, item.relation_label])).entries()]
          .sort((a, b) => a[1].localeCompare(b[1]))
          .forEach(([value, label]) => controls.relation.add(new Option(label[0].toUpperCase() + label.slice(1), value)));
        data.methods.forEach(method => controls.method.add(new Option(method.label, method.method_id)));
        controls.method.value = state.method;
        const counts = Object.fromEntries(data.tag_counts);
        $$(".preset-button").forEach(button => {
          const count = button.dataset.preset === "all" ? data.sample_count : counts[button.dataset.preset];
          button.textContent = `${button.textContent} · ${count}`;
        });
      };

      const filteredCases = () => data.cases.filter(item =>
        (state.preset === "all" || item.tags.includes(state.preset)) &&
        (state.world === "all" || item.concept_id === state.world) &&
        (state.relation === "all" || item.relation === state.relation) &&
        (!state.search || `${item.prompt} ${item.answer} ${item.candidates.join(" ")} ${item.fact_id}`.toLowerCase().includes(state.search))
      );

      const choiceRows = (caseItem, methodResult) => {
        const root = node("div", "choice-list");
        caseItem.candidates.forEach((answer, index) => {
          const classes = ["choice"];
          if (index === methodResult.predicted_index) classes.push("selected");
          if (index === caseItem.correct_index) classes.push("correct-answer");
          const row = node("div", classes.join(" "));
          row.style.setProperty("--share", percent(methodResult.choice_shares[index]));
          const flags = [];
          if (index === methodResult.predicted_index) flags.push("model choice");
          if (index === caseItem.correct_index) flags.push("correct answer");
          row.append(
            node("span", "choice-letter", String.fromCharCode(65 + index)),
            node("strong", "", answer),
            node("span", "share-label", `${Math.round(methodResult.choice_shares[index] * 100)}% preference share`),
            node("span", "choice-flags", flags.join(" · "))
          );
          root.append(row);
        });
        return root;
      };

      const comparisonFold = caseItem => {
        const fold = node("details", "mini-fold");
        fold.append(node("summary", "", "Compare all nine methods"));
        const content = node("div", "mini-fold-content world-table-wrap");
        const table = node("table", "method-table");
        const head = node("thead");
        const headRow = node("tr");
        ["Method", "Answer", "Result", "Chosen memory"].forEach(label => headRow.append(node("th", "", label)));
        head.append(headRow);
        const body = node("tbody");
        caseItem.methods.forEach(result => {
          const row = node("tr");
          row.append(
            node("td", "", methodById[result.method_id].short_label),
            node("td", "", caseItem.candidates[result.predicted_index]),
            node("td", result.answer_correct ? "correct-text" : "wrong-text", result.answer_correct ? "Correct" : "Wrong"),
            node("td", "", result.selected_node || "supplied directly")
          );
          body.append(row);
        });
        table.append(head, body);
        content.append(table, node("p", "technical", "Preference share compares the four choices within this question. It is not a calibrated probability."));
        fold.append(content);
        return fold;
      };

      const evidenceFold = caseItem => {
        const fold = node("details", "mini-fold");
        fold.append(node("summary", "", "Where this reviewed fact came from"));
        const content = node("div", "mini-fold-content");
        content.append(node("p", "", `Registered answer: “${caseItem.answer}” · kind: ${caseItem.relation_label}`));
        caseItem.evidence.forEach(text => content.append(node("p", "", `“${text}”`)));
        content.append(node("p", "technical", "These examples came from the construction slice used for review. That slice was permanently excluded from every model input."));
        fold.append(content);
        return fold;
      };

      const caseCard = caseItem => {
        const result = resultFor(caseItem, state.method);
        const card = node("details", "case-card");
        const summary = node("summary");
        const heading = node("span", "case-heading");
        heading.append(node("strong", "", caseItem.prompt), node("small", "", `${caseItem.concept_label} · ${caseItem.direction_label} · ${caseItem.relation_label}`));
        summary.append(
          node("span", "world-dot"),
          heading,
          node("span", `outcome-pill ${result.answer_correct ? "correct" : "wrong"}`, result.answer_correct ? "Correct" : "Wrong")
        );
        const body = node("div", "case-body");
        body.append(
          node("p", "takeaway", caseItem.takeaway),
          node("p", "question-text", caseItem.prompt),
          choiceRows(caseItem, result)
        );
        const folds = node("div", "case-folds");
        folds.append(comparisonFold(caseItem), evidenceFold(caseItem));
        body.append(folds);
        const tags = node("div", "tag-row");
        caseItem.tags.forEach(tag => tags.append(node("span", "tag", tagLabels[tag] || tag)));
        body.append(tags);
        card.append(summary, body);
        return card;
      };

      const renderCases = () => {
        const matches = filteredCases();
        const list = $("#case-list");
        list.replaceChildren();
        matches.slice(0, state.limit).forEach(item => list.append(caseCard(item)));
        if (!matches.length) list.append(node("div", "empty-state", "No cases match these filters. Try another preset or clear the search."));
        $("#case-count").textContent = `${matches.length} matching case${matches.length === 1 ? "" : "s"}`;
        $("#load-more").hidden = state.limit >= matches.length;
      };

      const resetAndRender = () => { state.limit = 12; renderCases(); };
      Object.entries(controls).forEach(([key, control]) => {
        control.addEventListener(key === "search" ? "input" : "change", event => {
          state[key] = key === "search" ? event.target.value.trim().toLowerCase() : event.target.value;
          resetAndRender();
        });
      });
      $$(".preset-button").forEach(button => button.addEventListener("click", () => {
        state.preset = button.dataset.preset;
        $$(".preset-button").forEach(item => item.classList.toggle("active", item === button));
        resetAndRender();
      }));
      $("#load-more").addEventListener("click", () => { state.limit += 12; renderCases(); });

      const renderGlossary = () => {
        const root = $("#method-glossary");
        data.methods.forEach(method => {
          const item = node("div", "glossary-item");
          item.append(node("strong", "", method.label), node("span", "", method.description));
          root.append(item);
        });
      };

      const renderProvenance = () => {
        $("#report-id").textContent = data.report_sha256;
        $("#catalog-id").textContent = data.catalog_sha256;
        $("#transaction-id").textContent = data.transaction_sha256;
        $("#ledger-id").textContent = data.result_ledger_sha256;
        $("#audit-id").textContent = data.audit_sha256;
        $("#source-question-total").textContent = `${data.source_question_count} sealed test questions in the registered result`;
        $("#question-total").textContent = `${data.question_count} forward questions that name their world`;
        $("#excluded-question-total").textContent = `${data.excluded_reverse_question_count} reverse questions, excluded uniformly`;
        $("#sample-total").textContent = `${data.sample_count} fixed forward examples across ${data.fact_count} facts`;
        $("#runtime").textContent = `${Math.round(data.runtime_seconds / 60)} minutes`;
        $("#memory").textContent = `${(data.allocator_peak_bytes / 2 ** 30).toFixed(2)} GiB of ${(data.allocator_limit_bytes / 2 ** 30).toFixed(0)} GiB allowed`;
      };

      $("#open-all").addEventListener("click", () => $$(".chapter").forEach(item => item.open = true));
      $("#fold-all").addEventListener("click", () => $$(".chapter").forEach(item => item.open = false));
      $("#presentation-toggle").addEventListener("click", event => {
        const enabled = document.body.classList.toggle("presentation-mode");
        event.target.textContent = enabled ? "Exit presentation mode" : "Presentation mode";
      });

      const revealChapter = () => {
        const target = document.getElementById(location.hash.slice(1));
        if (!target || !target.matches("details.chapter")) return;
        target.open = true;
        requestAnimationFrame(() => target.scrollIntoView({block: "start"}));
      };
      $$('a[href^="#"]').forEach(link => link.addEventListener("click", () => {
        const target = document.querySelector(link.getAttribute("href"));
        if (target && target.matches("details.chapter")) target.open = true;
      }));

      renderFactAtlas();
      renderResults();
      fillControls();
      renderCases();
      renderGlossary();
      renderProvenance();
      revealChapter();
    })();
  </script>
</body>
</html>
'''


__all__ = [
    "AuditFact",
    "AuditQuery",
    "InteractiveReportData",
    "build_interactive_report_data",
    "parse_opened_catalog_audit",
    "publish_interactive_report",
    "render_interactive_report",
]

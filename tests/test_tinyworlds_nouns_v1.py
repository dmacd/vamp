from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest

import apm.data.text.tinyworlds_nouns_v1.evaluation as noun_evaluation
from apm.continual.language_run import ParentSearchResult
from apm.continual.language_tasks import NodeId
from apm.data.text.tinyworlds_nouns_v1.contracts import (
    CONDITIONS,
    NounPartitionArtifact,
    NounTaskSummary,
    NounsExperimentPreset,
    canonical_json_bytes,
    record_sha256,
)
from apm.data.text.tinyworlds_nouns_v1.evaluation import (
    _story_windows,
    build_prefix_only_query,
)
from apm.data.text.tinyworlds_nouns_v1.experiment import IndexedTokenBatchSequence
from apm.data.text.tinyworlds_nouns_v1.judging import (
    JudgeHttpResponse,
    JudgeResponseError,
    anonymize_judge_request,
    judge_generation_ledger,
    parse_judge_content,
)
from apm.data.text.tinyworlds_nouns_v1.partition import (
    NounApprovalRequired,
    approve_noun_breakdown,
    build_breakdown_from_documents,
    build_noun_partition,
    decisions_record,
    initial_noun_decisions,
    load_noun_breakdown,
    load_noun_partition,
    match_noun_forms,
    publish_noun_breakdown,
    require_noun_approval,
    _initialize_scan_database,
)
from apm.data.text.tinyworlds_nouns_v1.report import (
    build_report_data,
    render_report_html,
    render_report_markdown,
)
from apm.lm.text import CharTokenizer


def _tokenizer(documents: tuple[str, ...]) -> CharTokenizer:
    return CharTokenizer.from_training_text("".join(documents))


def _story_id(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


def _small_breakdown():
    train = (
        "A cat and dog played.",
        "A cat met a bird.",
        "The dog met a bird.",
        "A cat went home.",
        "A dog saw a house.",
        "The bird found a castle.",
        "Two cats purred.",
    )
    validation = (
        "A cat slept.",
        "A dog slept.",
        "A bird slept.",
    )
    tokenizer = _tokenizer(train + validation)
    return build_breakdown_from_documents(
        train,
        validation,
        tokenizer,
        initial_noun_decisions(),
        evidence_count=2,
    )[0]


def test_whole_word_alternate_forms_and_known_exclusions() -> None:
    decisions = initial_noun_decisions()
    matches = dict(match_noun_forms("Cats chased mice near a catalog.", decisions))
    assert matches["cat"] == ("cats",)
    assert matches["mouse"] == ("mice",)
    assert "cat" not in dict(match_noun_forms("A catalog opened.", decisions))
    assert {decision.concept_id for decision in decisions if not decision.included} == {
        "friend",
        "saw",
    }


def test_overlap_and_validation_precedence_are_retained() -> None:
    train = ("A cat and dog played.", "A cat and dog played.", "A cat slept.")
    validation = ("A cat and dog played.", "A dog slept.")
    tokenizer = _tokenizer(train + validation)
    breakdown, stories = build_breakdown_from_documents(
        train,
        validation,
        tokenizer,
        initial_noun_decisions(),
        evidence_count=1,
    )
    rows = {row.decision.concept_id: row for row in breakdown.rows}
    assert breakdown.train_unique_story_count == 1
    assert breakdown.validation_unique_story_count == 2
    assert rows["cat"].train_story_count == 1
    assert rows["dog"].train_story_count == 0
    overlap_story = next(story for story in stories if "cat and dog" in story.text)
    cat_index = tuple(
        decision.concept_id for decision in initial_noun_decisions()
    ).index("cat")
    dog_index = tuple(
        decision.concept_id for decision in initial_noun_decisions()
    ).index("dog")
    assert overlap_story.concept_mask & (1 << cat_index)
    assert overlap_story.concept_mask & (1 << dog_index)


def test_greedy_base_stops_at_first_union_crossing() -> None:
    breakdown = _small_breakdown()
    assert breakdown.base_selection[-1].cumulative_story_coverage >= 0.5
    assert all(
        step.cumulative_story_coverage < 0.5
        for step in breakdown.base_selection[:-1]
    )
    counts = tuple(step.noun_story_count for step in breakdown.base_selection)
    assert counts == tuple(sorted(counts, reverse=True))


def test_review_packet_and_exact_manual_approval_gate(tmp_path: Path) -> None:
    decisions = initial_noun_decisions()
    breakdown = _small_breakdown()
    packet = publish_noun_breakdown(breakdown, decisions, tmp_path)
    assert load_noun_breakdown(packet / "noun-breakdown.json") == breakdown
    with pytest.raises(NounApprovalRequired):
        require_noun_approval(breakdown, decisions, tmp_path)
    with pytest.raises(ValueError, match="does not match"):
        approve_noun_breakdown(breakdown, decisions, "0" * 64, tmp_path)
    approval_path = approve_noun_breakdown(
        breakdown,
        decisions,
        breakdown.breakdown_sha256,
        tmp_path,
    )
    approval = require_noun_approval(breakdown, decisions, tmp_path)
    assert approval_path.stem == approval.approval_sha256
    payload = (packet / "noun-breakdown.json").read_bytes()
    (packet / "noun-breakdown.json").write_bytes(payload.replace(b"cat", b"bat", 1))
    with pytest.raises(ValueError):
        load_noun_breakdown(packet / "noun-breakdown.json")


def _fixture_partition(tmp_path: Path, output_name: str) -> NounPartitionArtifact:
    train = tuple(
        [f"A cat story number {index}." for index in range(400)]
        + [f"A cat and dog story number {index}." for index in range(200)]
        + [f"A dog story number {index}." for index in range(100)]
        + [f"A dog and bird story number {index}." for index in range(100)]
        + [f"A bird story number {index}." for index in range(200)]
    )
    validation = tuple(
        [f"A dog validation tale number {index}." for index in range(70)]
        + [f"A bird validation tale number {index}." for index in range(70)]
        + [f"A dog and bird validation tale number {index}." for index in range(10)]
    )
    tokenizer = _tokenizer(train + validation)
    decisions = initial_noun_decisions()
    breakdown, stories = build_breakdown_from_documents(
        train,
        validation,
        tokenizer,
        decisions,
        evidence_count=2,
    )
    database = tmp_path / f"{output_name}.sqlite3"
    connection = sqlite3.connect(database)
    _initialize_scan_database(connection)
    connection.executemany(
        "INSERT INTO stories VALUES (?, ?, ?, ?, ?, ?)",
        (
            (
                story.story_id,
                story.source_split,
                story.text.encode("utf-8"),
                np.asarray(story.token_ids, dtype="<u2").tobytes(),
                len(story.token_ids),
                story.concept_mask,
            )
            for story in stories
        ),
    )
    scan = {
        "breakdown_sha256": breakdown.breakdown_sha256,
        "decisions_sha256": record_sha256(decisions_record(decisions)),
        "format": "tinyworlds-nouns-scan-v1",
        "source_sha256": record_sha256(breakdown.source_identity),
    }
    connection.execute(
        "INSERT INTO metadata VALUES (?, ?)",
        ("scan", canonical_json_bytes(scan).decode("utf-8")),
    )
    connection.commit()
    connection.close()
    review_root = tmp_path / "review"
    approval_path = approve_noun_breakdown(
        breakdown,
        decisions,
        breakdown.breakdown_sha256,
        review_root,
    )
    approval = require_noun_approval(breakdown, decisions, review_root)
    assert approval_path.stem == approval.approval_sha256
    return build_noun_partition(
        breakdown,
        approval,
        decisions,
        database,
        tmp_path / output_name,
    )


def _index_ids(partition: NounPartitionArtifact, name: str) -> set[str]:
    return {
        json.loads(line)["story_id"]
        for line in (partition.root / "indexes" / f"{name}.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    }


def test_partition_retains_overlap_and_selects_probes_deterministically(
    tmp_path: Path,
) -> None:
    first = _fixture_partition(tmp_path, "first")
    second = _fixture_partition(tmp_path, "second")
    assert first.partition_sha256 == second.partition_sha256
    assert first.base_concept_ids == ("cat",)
    assert first.task_ids == ("dog", "bird")
    dog, bird = first.tasks
    assert dog.base_overlap_story_count == 200
    assert dict(dog.overlap_counts)["bird"] == 100
    assert dict(bird.overlap_counts)["dog"] == 100
    assert dog.generation_story_count == dog.validation_story_count == 80
    assert bird.generation_story_count == bird.validation_story_count == 80
    dog_probes = set(dog.probe_story_ids)
    bird_probes = set(bird.probe_story_ids)
    dog_train = _index_ids(first, "task-dog-train")
    bird_train = _index_ids(first, "task-bird-train")
    assert dog_probes.isdisjoint(dog_train)
    assert bird_probes.isdisjoint(bird_train)
    overlap_ids = dog_train & bird_train
    assert overlap_ids
    ledger = tuple(
        json.loads(line)
        for line in (first.root / "stories.jsonl").read_text(encoding="utf-8").splitlines()
    )
    overlap_row = next(
        row for row in ledger if set(row["concept_ids"]) == {"dog", "bird"}
    )
    assert overlap_row["matched_forms"] == {"bird": ["bird"], "dog": ["dog"]}
    extra = first.root / "unexpected.txt"
    extra.write_text("tamper", encoding="utf-8")
    with pytest.raises(ValueError, match="entries"):
        load_noun_partition(first.root / "partition.json")


def _fake_partition(tmp_path: Path) -> NounPartitionArtifact:
    root = tmp_path / ("a" * 64)
    indexes = root / "indexes"
    indexes.mkdir(parents=True)
    token_sequences = ((1, 2, 3, 4), (5, 6, 7), (8, 9, 10, 11, 12))
    offsets = np.cumsum((0,) + tuple(len(tokens) for tokens in token_sequences))
    np.concatenate(
        tuple(np.asarray(tokens, dtype="<u2") for tokens in token_sequences)
    ).tofile(root / "tokens.uint16")
    (root / "stories.bin").write_bytes(b"unused")
    rows = tuple(
        {
            "byte_length": 1,
            "story_id": _story_id(f"story-{index}"),
            "story_index": index,
            "story_offset": 0,
            "token_count": len(tokens),
            "token_offset": int(offsets[index]),
        }
        for index, tokens in enumerate(token_sequences)
    )
    (indexes / "fixture.jsonl").write_bytes(
        b"".join(canonical_json_bytes(row) for row in rows)
    )
    return NounPartitionArtifact(
        root=root,
        partition_sha256="a" * 64,
        breakdown_sha256="b" * 64,
        approval_sha256="c" * 64,
        source_identity={"fixture": True},
        tokenizer_identity={"vocab_size": 20},
        pad_token_id=0,
        eos_token_id=1,
        base_concept_ids=("cat",),
        task_ids=(),
        base_train_story_count=3,
        base_validation_story_count=1,
        root_probe_story_ids=tuple(_story_id(f"root-{index}") for index in range(36)),
        tasks=(),
    )


def test_lazy_indexed_batches_are_deterministic_and_count_each_target_once(
    tmp_path: Path,
) -> None:
    partition = _fake_partition(tmp_path)
    first = IndexedTokenBatchSequence(
        partition,
        "fixture",
        context_length=2,
        batch_size=2,
        order_namespace="test-order",
    )
    second = IndexedTokenBatchSequence(
        partition,
        "fixture",
        context_length=2,
        batch_size=2,
        order_namespace="test-order",
    )
    assert first.window_count == 5
    assert sum(int(np.sum(batch.loss_mask)) for batch in first) == 9
    assert all(
        np.array_equal(first[index].input_ids, second[index].input_ids)
        for index in range(len(first))
    )
    assert 0 < len(first.consumed_story_ids(1)) <= 2


def test_prefix_only_query_has_no_continuation_surface() -> None:
    query = build_prefix_only_query(_story_id("prefix"), (1, 2, 3, 4, 5, 6), 0, 8)
    assert query.prompt_token_ids == (1, 2, 3)
    assert not hasattr(query, "reference_continuation")
    assert not hasattr(query, "full_story")


def test_long_story_windows_count_every_target_once() -> None:
    token_ids = tuple(range(600))
    whole = _story_windows(token_ids, 256, 0)
    suffix = _story_windows(token_ids, 256, 0, first_target_index=300)
    query = build_prefix_only_query(_story_id("long-prefix"), token_ids, 0, 2_048)
    assert int(np.sum(whole.loss_mask)) == len(token_ids) - 1
    assert int(np.sum(suffix.loss_mask)) == len(token_ids) - 300
    assert query.prompt_token_ids == token_ids[:300]
    assert query.router_batch.input_ids.shape == (1, 299)


def test_all_completion_conditions_share_one_batched_budget(monkeypatch) -> None:
    calls: list[tuple[tuple[int, ...], int, tuple[int, ...]]] = []

    def generate(
        _params,
        _config,
        prompt_ids,
        _attention_mask,
        maximum_tokens,
        **keywords,
    ):
        node_indices = tuple(int(value) for value in keywords["node_index"])
        calls.append((prompt_ids.shape, maximum_tokens, node_indices))
        endings = np.tile(
            np.arange(10, 10 + maximum_tokens, dtype=np.int32),
            (prompt_ids.shape[0], 1),
        )
        return np.concatenate((prompt_ids, endings), axis=1)

    monkeypatch.setattr(noun_evaluation, "greedy_generate", generate)
    monkeypatch.setattr(
        noun_evaluation,
        "_node_path",
        lambda _adaptation, node_index: ("root", f"node-{node_index}"),
    )
    query = build_prefix_only_query(_story_id("batched-generation"), (1, 2, 3, 4), 0, 32)
    nodes = tuple(SimpleNamespace(node_id=f"node-{index}") for index in range(3))
    adaptation = SimpleNamespace(
        lora_config=object(),
        model_config=object(),
        vamp_graph=SimpleNamespace(nodes=nodes),
    )
    selections = {
        condition: index % len(nodes)
        for index, condition in enumerate(CONDITIONS)
    }
    results = noun_evaluation._completion_results(
        selections,
        query,
        (3.0, 6.0, 9.0),
        3,
        5,
        object(),
        adaptation,
        object(),
        SimpleNamespace(decode=lambda values: " ".join(map(str, values))),
        SimpleNamespace(eos_token_id=99, pad_token_id=0),
    )
    assert calls == [((len(CONDITIONS), 2), 5, tuple(selections.values()))]
    assert tuple(result.condition for result in results) == CONDITIONS
    assert {result.generated_token_count for result in results} == {5}


def test_generation_batches_multiple_stories_without_mixing_budgets(monkeypatch) -> None:
    calls: list[tuple[tuple[int, ...], tuple[int, ...], int, tuple[int, ...]]] = []

    def generate(
        _params,
        _config,
        prompt_ids,
        attention_mask,
        maximum_tokens,
        **keywords,
    ):
        node_indices = tuple(int(value) for value in keywords["node_index"])
        lengths = tuple(int(value) for value in np.sum(attention_mask, axis=1))
        calls.append((prompt_ids.shape, lengths, maximum_tokens, node_indices))
        output = np.pad(prompt_ids, ((0, 0), (0, maximum_tokens)))
        for row, length in enumerate(lengths):
            output[row, length : length + maximum_tokens] = 100 + row
        return output

    monkeypatch.setattr(noun_evaluation, "greedy_generate", generate)
    monkeypatch.setattr(
        noun_evaluation,
        "_node_path",
        lambda _adaptation, node_index: ("root", f"node-{node_index}"),
    )
    first_query = build_prefix_only_query(
        _story_id("first-batched-story"),
        (1, 2, 3, 4),
        0,
        32,
    )
    second_query = build_prefix_only_query(
        _story_id("second-batched-story"),
        (5, 6, 7, 8, 9, 10),
        0,
        32,
    )
    first = SimpleNamespace(query=first_query, budget=2)
    second = SimpleNamespace(query=second_query, budget=4)
    nodes = tuple(SimpleNamespace(node_id=f"node-{index}") for index in range(3))
    adaptation = SimpleNamespace(
        lora_config=object(),
        model_config=SimpleNamespace(max_position_embeddings=32),
        vamp_graph=SimpleNamespace(nodes=nodes),
    )
    selections = tuple(
        {
            condition: (condition_index + story_index) % len(nodes)
            for condition_index, condition in enumerate(CONDITIONS)
        }
        for story_index in range(2)
    )
    results = noun_evaluation._generate_completion_chunk(
        (first, second),
        selections,
        ((3.0, 6.0, 9.0), (4.0, 8.0, 12.0)),
        (3, 4),
        object(),
        adaptation,
        object(),
        SimpleNamespace(decode=lambda values: " ".join(map(str, values))),
        SimpleNamespace(eos_token_id=99, pad_token_id=0),
        row_capacity=8,
    )
    assert calls == [
        (
            (8, 3),
            (2,) * 3 + (3,) * 3 + (2,) * 2,
            4,
            (0, 1, 2, 1, 2, 0, 0, 0),
        )
    ]
    assert {result.generated_token_count for result in results[0]} == {2}
    assert {result.generated_token_count for result in results[1]} == {4}
    assert results[0][0].generated_continuation == "100 100"
    assert results[1][0].generated_continuation == "103 103 103 103"


def test_generation_bins_sort_by_budget_and_restore_bounded_groups() -> None:
    budgets = (2, 8, 7)
    cases = tuple(
        SimpleNamespace(
            budget=budget,
            query=build_prefix_only_query(
                _story_id(f"generation-bin-{index}"),
                (1, 2, 3, 4),
                0,
                16,
            ),
        )
        for index, budget in enumerate(budgets)
    )
    selections = tuple(
        {
            condition: condition_index % 2
            for condition_index, condition in enumerate(CONDITIONS)
        }
        for _ in cases
    )

    bins = noun_evaluation._generation_case_bins(
        cases,
        selections,
        16,
        row_capacity=4,
    )

    assert bins == ((1, 2), (0,))


def test_prefix_routing_uses_shape_stable_eight_story_subbatches(monkeypatch) -> None:
    calls: list[tuple[str, tuple[int, ...], int]] = []

    def route(condition, _params, _config, _packed, _lora, _book, batch, **kwargs):
        calls.append(
            (
                condition,
                batch.input_ids.shape,
                kwargs["evaluation_microbatch_size"],
            )
        )
        return SimpleNamespace(
            selected_indices=np.arange(batch.input_ids.shape[0], dtype=np.int32) % 3
        )

    monkeypatch.setattr(noun_evaluation, "route_language_prefix", route)
    cases = tuple(
        SimpleNamespace(
            oracle_index=index + 1,
            query=build_prefix_only_query(
                _story_id(f"routing-{index}"),
                tuple(range(1, 7 + index)),
                0,
                64,
            ),
        )
        for index in range(3)
    )
    adaptation = SimpleNamespace(
        model_config=SimpleNamespace(max_position_embeddings=64),
        lora_config=object(),
        address_book=object(),
    )
    selections = noun_evaluation._prefix_chunk_selections(
        cases,
        object(),
        adaptation,
        object(),
        32,
    )
    assert calls == [
        (condition, (8, 32), 8)
        for condition in CONDITIONS[2:]
    ]
    assert tuple(selection["oracle"] for selection in selections) == (1, 2, 3)
    assert tuple(selection["vamp_exhaustive"] for selection in selections) == (
        0,
        1,
        2,
    )


def test_parent_search_mask_keeps_raw_root_score_but_selects_nonroot() -> None:
    result = ParentSearchResult(
        node_ids=(NodeId("root"), NodeId("cat"), NodeId("dog")),
        mean_candidate_nll=(0.1, 0.4, 0.3),
        selected_node_index=2,
        selected_node_id=NodeId("dog"),
        scoring_basis="mean_prefix_nll",
        eligible_node_mask=(False, True, True),
    )
    assert result.mean_candidate_nll[0] == 0.1
    assert result.selected_node_id == "dog"
    with pytest.raises(ValueError, match="eligible"):
        ParentSearchResult(
            result.node_ids,
            result.mean_candidate_nll,
            0,
            NodeId("root"),
            result.scoring_basis,
            result.eligible_node_mask,
        )


def _generation_record(index: int = 0) -> dict[str, object]:
    return {
        "format": "tinyworlds-nouns-half-story-generation-v1",
        "full_original_story": "Once there was a cat. It slept.",
        "prefix": "Once there was a cat.",
        "reference_continuation": " It slept.",
        "results": {
            condition: {
                "condition": condition,
                "generated_continuation": f" ending {condition} {index}",
                "mean_nll": 1.0 if condition == "oracle" else 2.0,
                "selected_node": "cat" if condition != "base" else "root",
                "selected_path": ["root", "cat"] if condition != "base" else ["root"],
                "token_count": 3,
                "total_nll": 3.0 if condition == "oracle" else 6.0,
            }
            for condition in (
                "base",
                "oracle",
                "vamp_exhaustive",
                "vamp_hopfield",
                "vamp_ebt_uniform",
                "vamp_ebt_hopfield",
            )
        },
        "story_id": _story_id(f"judge-{index}"),
        "task_noun": "cat",
    }


def _judge_content() -> str:
    labels = tuple(chr(ord("A") + index) for index in range(7))
    return json.dumps(
        {
            "ranking": list(labels),
            "scores": [
                {
                    "candidate": label,
                    "coherence": 4,
                    "writing_quality": 4,
                    "ending_quality": 4,
                    "overall": 4,
                    "reason": "clear",
                }
                for label in labels
            ],
        }
    )


def test_judge_anonymization_and_schema_are_deterministic() -> None:
    first = anonymize_judge_request(_generation_record(), "fixture/model")
    second = anonymize_judge_request(_generation_record(), "fixture/model")
    assert first.body == second.body
    assert first.label_sources == second.label_sources
    assert set(dict(first.label_sources).values()) == {
        "base",
        "oracle",
        "vamp_exhaustive",
        "vamp_hopfield",
        "vamp_ebt_uniform",
        "vamp_ebt_hopfield",
        "reference",
    }
    parsed = parse_judge_content(_judge_content())
    assert len(parsed.scores) == len(parsed.ranking) == 7
    incomplete = json.loads(_judge_content())
    incomplete["ranking"].pop()
    with pytest.raises(JudgeResponseError):
        parse_judge_content(json.dumps(incomplete))


class _FakeJudgeTransport:
    def __init__(self) -> None:
        self.posts = 0

    def model_available(self, model: str) -> bool:
        return model == "fixture/model"

    def post(self, api_key: str, body: bytes) -> JudgeHttpResponse:
        assert api_key == "secret"
        assert b"secret" not in body
        self.posts += 1
        return JudgeHttpResponse(
            200,
            json.dumps(
                {"choices": [{"message": {"content": _judge_content()}}]}
            ).encode("utf-8"),
        )


def test_judge_persists_every_case_and_resumes(tmp_path: Path) -> None:
    case_count = 3
    generations = tmp_path / "generations.jsonl"
    generations.write_bytes(
        b"".join(
            canonical_json_bytes(_generation_record(index))
            for index in range(case_count)
        )
    )
    transport = _FakeJudgeTransport()
    result = judge_generation_ledger(
        generations,
        tmp_path / "judge",
        api_key="secret",
        model="fixture/model",
        transport=transport,
    )
    assert transport.posts == case_count
    assert len(result.read_text(encoding="utf-8").splitlines()) == case_count
    assert (
        judge_generation_ledger(
            generations,
            tmp_path / "judge",
            api_key="secret",
            model="fixture/model",
            transport=transport,
        )
        == result
    )
    assert transport.posts == case_count


def test_report_renders_overall_confusion_overlap_and_examples(tmp_path: Path) -> None:
    del tmp_path
    task_ids = ("cat", "dog")
    tasks = tuple(
        NounTaskSummary(
            task_id=task_id,
            train_story_count=300,
            update_story_count=264,
            validation_story_count=1,
            generation_story_count=1,
            probe_story_ids=tuple(
                _story_id(f"{task_id}-probe-{index}") for index in range(36)
            ),
            base_overlap_story_count=150 if task_id == "cat" else 30,
            overlap_counts=(("cat", 300 if task_id == "cat" else 60), ("dog", 60 if task_id == "cat" else 300)),
        )
        for task_id in task_ids
    )
    partition = NounPartitionArtifact(
        root=Path("unused-report-partition"),
        partition_sha256="a" * 64,
        breakdown_sha256="b" * 64,
        approval_sha256="c" * 64,
        source_identity={"fixture": True},
        tokenizer_identity={"vocab_size": 20},
        pad_token_id=0,
        eos_token_id=1,
        base_concept_ids=("bird",),
        task_ids=task_ids,
        base_train_story_count=500,
        base_validation_story_count=10,
        root_probe_story_ids=tuple(
            _story_id(f"report-root-{index}") for index in range(36)
        ),
        tasks=tasks,
    )
    graph = SimpleNamespace(
        nodes=(
            SimpleNamespace(node_id="root", parent_id=None, depth=0, train_stage=0),
            SimpleNamespace(node_id="cat", parent_id="root", depth=1, train_stage=1),
            SimpleNamespace(node_id="dog", parent_id="cat", depth=2, train_stage=2),
        )
    )
    adaptation = SimpleNamespace(vamp_graph=graph)
    whole_rows = tuple(
        {
            "condition": condition,
            "mean_nll": 2.0 if condition == "base" else 1.0,
            "oracle_match": condition != "base",
            "perplexity": 7.0 if condition == "base" else 3.0,
            "regret_vs_oracle": 1.0 if condition == "base" else 0.0,
            "selected_node": "root" if condition == "base" else task_id,
            "story_id": _story_id(f"report-{task_id}"),
            "task_noun": task_id,
            "token_count": 3,
            "total_nll": 6.0 if condition == "base" else 3.0,
        }
        for task_id in task_ids
        for condition in (
            "base",
            "oracle",
            "vamp_exhaustive",
            "vamp_hopfield",
            "vamp_ebt_uniform",
            "vamp_ebt_hopfield",
        )
    )
    generations = tuple(
        {**_generation_record(index), "task_noun": task_id}
        for index, task_id in enumerate(task_ids)
    )
    data = build_report_data(
        partition,
        _small_breakdown(),
        NounsExperimentPreset(),
        adaptation,
        whole_rows,
        generations,
        (),
    )
    markdown = render_report_markdown(data)
    html = render_report_html(data)
    assert "story-weighted loss" in markdown
    assert "largest task overlaps" in markdown
    assert "Routing confusion matrices" in html
    assert "Dataset overlap by noun" in html
    assert "Why shown" in html
    assert len(data["examples"]) == 2

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import apm.data.text.tinyworlds_nouns_v1.evaluation as shared_evaluation
from apm.data.text.tinyworlds_nouns_v1.contracts import NounsExperimentPreset
from apm.data.text.tinyworlds_nouns_v1.experiment import StoryIndexEntry
from apm.data.text.tinyworlds_nouns_v2.contracts import (
    BASE_SELECTION_FORMAT,
    BASE_TRAINING_FORMAT,
    GPU_PREFLIGHT_FORMAT,
    HALF_STORY_FORMAT,
    STAGEWISE_FORMAT,
    TASK_IDS,
    VAMP_STAGE_FORMAT,
    WHOLE_STORY_FORMAT,
    NounConceptFamily,
    NounsV2ExperimentPreset,
    StagewiseClRow,
    StagewiseConditionResult,
    WholeStoryNllRow,
    canonical_json_bytes,
    record_sha256,
)
from apm.data.text.tinyworlds_nouns_v2.evaluation import build_prefix_only_query
from apm.data.text.tinyworlds_nouns_v2.partition import (
    build_fixture_disjoint_partition,
    classify_selected_concepts,
    load_fixture_disjoint_partition,
    match_selected_forms,
)
from apm.data.text.tinyworlds_nouns_v2.report import (
    render_report_html,
    render_report_markdown,
    render_vamp_graph_svg,
)
import apm.data.text.tinyworlds_nouns_v2.stagewise as stagewise
from apm.data.text.tinyworlds_nouns_v2.stagewise import summarize_stagewise_rows
from apm.lm.text import CharTokenizer
from apm.memory.graph import NodeId, TaskId, add_memory_node, init_memory_graph


FAMILIES = (
    NounConceptFamily("mouse", "animals", ("mouse", "mice")),
    NounConceptFamily("boat", "vehicles_tools", ("boat", "boats")),
)


def _tokenizer(documents: tuple[str, ...]) -> CharTokenizer:
    return CharTokenizer.from_training_text("".join(documents))


def _fixture():
    train = (
        "A child smiled in the garden.",
        "A teacher read a little story.",
        "One mouse ate cheese.",
        "Two mice played together.",
        "A mouse met another mouse.",
        "A boat crossed the lake.",
        "Two boats reached shore.",
        "A mouse climbed into a boat.",
    )
    validation = (
        "The mice slept quietly.",
        "The boat was blue.",
        "A mouse waved from the boat.",
    )
    return build_fixture_disjoint_partition(
        train,
        validation,
        _tokenizer(train + validation),
        FAMILIES,
        minimum_train_stories=2,
        minimum_validation_stories=1,
        probe_count=1,
        base_validation_bucket_count=2,
    )


def test_alternate_forms_repeat_mentions_and_whole_words() -> None:
    matches = dict(
        match_selected_forms(
            "Mice saw a mouse near two boats and a mouseboat.", FAMILIES
        )
    )
    assert matches == {
        "mouse": ("mouse", "mice"),
        "boat": ("boats",),
    }
    assert classify_selected_concepts(("mouse", "mouse")).role == "task"
    assert classify_selected_concepts(("mouse", "boat")).role == "excluded"


def test_fixture_base_tasks_and_task_pairs_are_story_disjoint() -> None:
    partition = _fixture()
    assert partition.task_ids == ("mouse", "boat")
    base = set(partition.base_universe_story_ids)
    task_sets = tuple(set(task.train_story_ids) for task in partition.tasks)
    assert not any(base & task_set for task_set in task_sets)
    assert not task_sets[0] & task_sets[1]
    assert len(partition.excluded_train_story_ids) == 1
    assert len(partition.excluded_validation_story_ids) == 1
    assert sum(len(task.train_story_ids) for task in partition.tasks) == 5


def test_holdout_probes_order_and_reconstruction_are_deterministic() -> None:
    first = _fixture()
    second = _fixture()
    assert first == second
    assert first.partition_sha256 == second.partition_sha256
    assert first.base_universe_story_ids == tuple(
        sorted(first.base_universe_story_ids)
    )
    assert all(
        len(task.probe_story_ids) == 1
        and not set(task.probe_story_ids) & set(task.update_story_ids)
        for task in first.tasks
    )
    payload = canonical_json_bytes(first.as_record())
    assert load_fixture_disjoint_partition(payload) == first


def test_fixture_threshold_and_tamper_rejection() -> None:
    train = ("A mouse ran.", "A boat sailed.")
    validation = ("A mouse slept.", "A boat stopped.")
    with pytest.raises(ValueError, match="threshold"):
        build_fixture_disjoint_partition(
            train,
            validation,
            _tokenizer(train + validation),
            FAMILIES,
            minimum_train_stories=2,
        )
    payload = canonical_json_bytes(_fixture().as_record())
    with pytest.raises(ValueError, match="identity"):
        load_fixture_disjoint_partition(payload.replace(b"mouse", b"house", 1))


def test_v2_preset_and_result_contracts_have_independent_hashes() -> None:
    v1 = NounsExperimentPreset()
    v2 = NounsV2ExperimentPreset()
    assert v1.config_sha256 != v2.config_sha256
    assert v2.as_record()["format"] == "tinyworlds-nouns-experiment-preset-v2"
    row = WholeStoryNllRow(
        task_noun="mouse",
        story_id=sha256(b"story").hexdigest(),
        condition="oracle",
        selected_node="mouse",
        selected_path=("root", "mouse"),
        oracle_node="mouse",
        oracle_match=True,
        total_nll=3.0,
        token_count=3,
        mean_nll=1.0,
        perplexity=2.718,
        regret_vs_oracle=0.0,
    ).as_record()
    supplied = row.pop("result_sha256")
    assert row["format"] == WHOLE_STORY_FORMAT
    assert supplied == record_sha256(row)

    stage_results = tuple(
        StagewiseConditionResult(
            condition=condition,
            selected_node="mouse" if condition != "base" else "root",
            selected_path=("root", "mouse") if condition != "base" else ("root",),
            oracle_match=condition != "base",
            total_nll=2.0,
            token_count=2,
            mean_nll=1.0,
            regret_vs_oracle=0.0,
        )
        for condition in (
            "base",
            "oracle",
            "vamp_exhaustive",
            "vamp_hopfield",
            "vamp_ebt_uniform",
            "vamp_ebt_hopfield",
        )
    )
    stage_row = StagewiseClRow(
        stage_index=1,
        introduced_task="mouse",
        stage_tensor_checksum="1" * 64,
        task_noun="mouse",
        story_id=sha256(b"stage-story").hexdigest(),
        results=stage_results,
    ).as_record()
    stage_hash = stage_row.pop("result_sha256")
    assert stage_row["format"] == STAGEWISE_FORMAT
    assert stage_hash == record_sha256(stage_row)


def test_shared_base_and_vamp_resume_formats_are_v2_bound() -> None:
    assert len(TASK_IDS) == 24
    assert {
        BASE_TRAINING_FORMAT,
        BASE_SELECTION_FORMAT,
        GPU_PREFLIGHT_FORMAT,
        VAMP_STAGE_FORMAT,
    } == {
        "tinyworlds-nouns-base-training-v2",
        "tinyworlds-nouns-selected-base-v2",
        "tinyworlds-nouns-gpu-preflight-v2",
        "tinyworlds-nouns-vamp-stage-v2",
    }


def test_prefix_contract_cannot_expose_continuation() -> None:
    query = build_prefix_only_query(
        sha256(b"prefix").hexdigest(),
        tuple(range(10)),
        0,
        32,
    )
    assert query.prompt_token_ids == tuple(range(5))
    assert not hasattr(query, "continuation")
    assert not hasattr(query, "full_story")


def test_versioned_evaluation_work_ledger_recovers_an_interrupted_tail(
    tmp_path: Path,
) -> None:
    path = tmp_path / "whole.jsonl.work"
    core = {
        "condition": "base",
        "format": WHOLE_STORY_FORMAT,
        "story_id": sha256(b"resume-story").hexdigest(),
        "task_noun": "mouse",
    }
    record = {**core, "result_sha256": record_sha256(core)}
    path.write_bytes(canonical_json_bytes(record) + b'{"interrupted"')
    keys = shared_evaluation._completed_keys(
        path, ("task_noun", "story_id", "condition")
    )
    assert keys == {("mouse", core["story_id"], "base")}
    assert path.read_bytes() == canonical_json_bytes(record)


def test_report_rendering_is_byte_stable_after_resume() -> None:
    condition_summary = {
        "mean_regret": 0.0,
        "routing_accuracy": 1.0,
        "story_count": 1,
        "story_mean_nll": 1.0,
        "story_perplexity": 2.718,
        "token_count": 2,
        "token_mean_nll": 1.0,
    }
    conditions = {condition: condition_summary for condition in (
        "base",
        "oracle",
        "vamp_exhaustive",
        "vamp_hopfield",
        "vamp_ebt_uniform",
        "vamp_ebt_hopfield",
    )}
    continual = _stagewise_fixture_summary()
    data = {
        "base": {
            "optimizer_share": 0.79,
            "optimizer_train_story_count": 2,
            "universe_share": 0.8136,
            "universe_story_count": 2,
        },
        "construction": {
            "excluded_train_story_count": 1,
            "excluded_validation_story_count": 1,
            "pure_task_train_story_count": 2,
            "pure_validation_pair_count": 2,
        },
        "continual_learning": continual,
        "examples": [],
        "graph": [{"depth": 0, "node": "root", "parent": None, "stage": 0}],
        "judge": {"available": False},
        "overall_conditions": conditions,
        "report_sha256": "0" * 64,
        "suffix_conditions": {
            condition: {
                "routing_accuracy": 1.0,
                "story_mean_nll": 1.0,
                "token_mean_nll": 1.0,
            }
            for condition in conditions
        },
        "task_metrics": [],
    }
    assert render_report_markdown(data) == render_report_markdown(data)
    assert render_report_html(data) == render_report_html(data)
    assert HALF_STORY_FORMAT.endswith("-v2")


def test_stagewise_metrics_separate_stored_retention_from_router_decay() -> None:
    summary = _stagewise_fixture_summary()
    assert summary["row_count"] == 3
    assert summary["oracle_max_absolute_drift"] == pytest.approx(0.0)
    exhaustive = summary["condition_summaries"]["vamp_exhaustive"]
    assert exhaustive["mean_task_forgetting"] == pytest.approx(0.15)
    assert exhaustive["mean_backward_transfer"] == pytest.approx(-0.15)
    assert exhaustive["mean_route_accuracy_change"] == pytest.approx(-0.5)


def test_vamp_dependency_svg_is_deterministic_and_contains_every_edge() -> None:
    graph = [
        {"depth": 0, "node": "root", "parent": None, "stage": 0},
        {"depth": 1, "node": "mouse", "parent": "root", "stage": 1},
        {"depth": 2, "node": "boat", "parent": "mouse", "stage": 2},
    ]
    rendered = render_vamp_graph_svg(graph)
    assert rendered == render_vamp_graph_svg(graph)
    assert "Learned VAMP node dependencies" in rendered
    assert "01 · mouse" in rendered
    assert "parent root" in rendered
    assert "parent mouse" in rendered


def test_stagewise_ledger_resume_and_tamper_rejection(tmp_path: Path) -> None:
    story_id = sha256(b"stagewise-ledger-story").hexdigest()
    checksum = "2" * 64
    graph = add_memory_node(
        init_memory_graph(NodeId("root")),
        NodeId("mouse"),
        NodeId("root"),
        TaskId("mouse"),
        1,
        1,
    )
    adaptation = SimpleNamespace(
        task_order=("mouse",),
        tensor_checksum=checksum,
        vamp_graph=graph,
        vamp_stages=(object(),),
    )
    partition = SimpleNamespace(task_ids=("mouse",))
    entries = {
        "mouse": (
            StoryIndexEntry(story_id, 0, 0, 1, 0, 2),
        )
    }
    results = tuple(
        StagewiseConditionResult(
            condition=condition,
            selected_node="root" if condition == "base" else "mouse",
            selected_path=("root",) if condition == "base" else ("root", "mouse"),
            oracle_match=condition != "base",
            total_nll=2.0,
            token_count=2,
            mean_nll=1.0,
            regret_vs_oracle=0.0,
        )
        for condition in (
            "base",
            "oracle",
            "vamp_exhaustive",
            "vamp_hopfield",
            "vamp_ebt_uniform",
            "vamp_ebt_hopfield",
        )
    )
    record = StagewiseClRow(
        1,
        "mouse",
        checksum,
        "mouse",
        story_id,
        results,
    ).as_record()
    ledger = tmp_path / "stagewise.jsonl.work"
    ledger.write_bytes(canonical_json_bytes(record) + b'{"interrupted"')
    stagewise._repair_interrupted_tail(ledger)
    assert stagewise.validate_stagewise_ledger(
        ledger,
        partition,
        (adaptation,),
        require_complete=True,
        entries_by_task=entries,
    ) == {("1", "mouse", story_id)}

    tampered = json.loads(canonical_json_bytes(record))
    tampered["results"]["vamp_hopfield"]["selected_path"] = ["root"]
    core = {key: value for key, value in tampered.items() if key != "result_sha256"}
    tampered["result_sha256"] = record_sha256(core)
    ledger.write_bytes(canonical_json_bytes(tampered))
    with pytest.raises(ValueError, match="route metadata"):
        stagewise.validate_stagewise_ledger(
            ledger,
            partition,
            (adaptation,),
            require_complete=True,
            entries_by_task=entries,
        )


def _stagewise_fixture_summary() -> dict[str, object]:
    task_ids = ("mouse", "boat")

    def row(
        stage: int,
        task: str,
        oracle_nll: float,
        exhaustive_nll: float,
        exhaustive_match: bool,
    ) -> dict[str, object]:
        results = {}
        for condition in (
            "base",
            "oracle",
            "vamp_exhaustive",
            "vamp_hopfield",
            "vamp_ebt_uniform",
            "vamp_ebt_hopfield",
        ):
            mean_nll = (
                oracle_nll + 0.5
                if condition == "base"
                else exhaustive_nll
                if condition == "vamp_exhaustive"
                else oracle_nll
            )
            oracle_match = (
                False
                if condition == "base"
                else exhaustive_match
                if condition == "vamp_exhaustive"
                else True
            )
            results[condition] = {
                "mean_nll": mean_nll,
                "oracle_match": oracle_match,
                "regret_vs_oracle": mean_nll - oracle_nll,
                "token_count": 2,
                "total_nll": mean_nll * 2,
            }
        return {"results": results, "stage_index": stage, "task_noun": task}

    return summarize_stagewise_rows(
        (
            row(1, "mouse", 1.0, 1.2, True),
            row(2, "mouse", 1.0, 1.5, False),
            row(2, "boat", 2.0, 2.2, True),
        ),
        task_ids,
    )

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import runpy

import numpy as np
import pytest

from apm.continual.knowledge_evaluation import (
    KnowledgeMethodEvaluation,
    KnowledgeQueryEvaluation,
    aggregate_knowledge_evaluations,
)
from apm.continual.tinyworlds_progress import (
    TinyWorldsProgressEvent,
    TinyWorldsProgressWriter,
    TinyWorldsSequentialResult,
)
from apm.continual.tinyworlds_report import (
    TINYWORLDS_ADDRESSING_METHODS,
    TINYWORLDS_IMPLEMENTATION_GATES,
    TINYWORLDS_NATURAL_CONTINUATION_METHODS,
    TINYWORLDS_PARENT_COUNTERFACTUALS,
    TINYWORLDS_REPORT_CUE_REGIMES,
    TINYWORLDS_REPORT_FILENAMES,
    TINYWORLDS_REPORT_METHODS,
    TINYWORLDS_REPORT_PREFIX_LENGTHS,
    TINYWORLDS_REPORT_QUERY_KINDS,
    TINYWORLDS_REPORT_STAGES,
    TINYWORLDS_REPORT_TASK_IDS,
    TINYWORLDS_REQUIRED_CONFIG_FIELDS,
    TinyWorldsCompletedResult,
    TinyWorldsRecord,
    TinyWorldsReportManifest,
    atomically_promote_tinyworlds_report,
    build_tinyworlds_report_bundle,
    canonical_tinyworlds_config_json,
    tinyworlds_report_directory,
    validate_tinyworlds_report_artifact,
    write_tinyworlds_report,
)


def _record(**values: object) -> TinyWorldsRecord:
    return TinyWorldsRecord(tuple(values.items()))  # type: ignore[arg-type]


def _manifest() -> TinyWorldsReportManifest:
    config = {
        field_name: f"fixed-{field_name}"
        for field_name in TINYWORLDS_REQUIRED_CONFIG_FIELDS
    }
    config.update(
        {
            "routers": list(TINYWORLDS_REPORT_METHODS),
            "seeds": {
                "calibration": "a" * 64,
                "pilot": "b" * 64,
                "public": 0,
            },
            "topology": list(TINYWORLDS_REPORT_TASK_IDS),
            "ontology": {"version": "ontology-v1"},
            "calibration_profile": {"sha256": "c" * 64},
            "optimizer_settings": {"steps": 4},
            "microbatching": {"candidate": 8, "routing": 16},
            "timing_targets": {"pilot_seconds": 28_800},
            "memory_targets": {"allocator_peak_bytes": 12 * 1024**3},
        }
    )
    return TinyWorldsReportManifest(
        preset="single-gpu",
        seed=0,
        config_json=canonical_tinyworlds_config_json(config),
    )


def _query(
    stage: int,
    method: str,
    index: int,
) -> KnowledgeQueryEvaluation:
    correct_index = index % 4
    scores = np.full((4,), 0.8, dtype=np.float32)
    scores[correct_index] = 0.2
    task_id = TINYWORLDS_REPORT_TASK_IDS[index]
    query_kind = TINYWORLDS_REPORT_QUERY_KINDS[index]
    return KnowledgeQueryEvaluation(
        stage=stage,
        method=method,
        query_id=f"query-{index}",
        task_id=task_id,
        family_id="willow" if task_id.startswith("willow") else "sunny",
        query_kind=query_kind,
        proof_id=f"proof-{index}",
        support_ids=(f"fact-{index}",),
        required_edge_ids=(f"edge-{index}",),
        cue_regime=TINYWORLDS_REPORT_CUE_REGIMES[index % 4],
        visible_cue_ids=(f"cue-{index}",),
        eligible_task_ids=(task_id,),
        novelty_regime="novel_binding",
        reasoning_type=query_kind,
        reasoning_depth=index % 3,
        prefix_length=TINYWORLDS_REPORT_PREFIX_LENGTHS[index % 3],
        mode="open_book" if query_kind == "open_book" else "closed_book",
        oracle_node_ids=(task_id,),
        candidate_answer_texts=tuple(f"answer-{index}-{item}" for item in range(4)),
        candidate_nll=scores,
        correct_candidate_index=correct_index,
        predicted_candidate_index=correct_index,
        candidate_correct=True,
        candidate_margin=0.6,
        correct_answer_nll=0.2,
        selected_node_index=None,
        task_oracle_node_index=0,
        best_hard_node_index=0,
        routed_correct_answer_nll=None,
        task_oracle_correct_answer_nll=0.2,
        best_hard_node_correct_answer_nll=0.2,
        routed_regret=None,
        task_oracle_regret=0.0,
        best_hard_node_regret=0.0,
        node_accuracy=None,
        top_k_accuracy=None,
        address_entropy=None,
        address_margin=None,
        hard_required_edge_recall=None,
        soft_required_edge_mean_coefficient=None,
    )


def _method_evaluations() -> tuple[KnowledgeMethodEvaluation, ...]:
    evaluations = []
    for stage in TINYWORLDS_REPORT_STAGES:
        for method in TINYWORLDS_REPORT_METHODS:
            rows = tuple(
                _query(stage, method, index)
                for index in range(len(TINYWORLDS_REPORT_TASK_IDS))
            )
            evaluations.append(
                KnowledgeMethodEvaluation(
                    stage=stage,
                    method=method,
                    queries=rows,
                    aggregates=aggregate_knowledge_evaluations(rows),
                )
            )
    return tuple(evaluations)


@pytest.fixture(scope="module")
def completed_result() -> TinyWorldsCompletedResult:
    natural = tuple(
        _record(
            stage=stage,
            method=method,
            task_id=task_id,
            prefix_length=prefix_length,
            suffix_nll=1.0,
        )
        for stage in TINYWORLDS_REPORT_STAGES
        for method in TINYWORLDS_NATURAL_CONTINUATION_METHODS
        for task_id in TINYWORLDS_REPORT_TASK_IDS
        for prefix_length in TINYWORLDS_REPORT_PREFIX_LENGTHS
    )
    parent_search = tuple(
        _record(
            stage=stage,
            task_id=TINYWORLDS_REPORT_TASK_IDS[stage - 1],
            candidate_parent_id="root",
            rank=0,
            mean_candidate_nll=0.4,
            selected=True,
        )
        for stage in TINYWORLDS_REPORT_STAGES
    )
    transfer = tuple(
        _record(
            stage=stage,
            task_id=TINYWORLDS_REPORT_TASK_IDS[stage - 1],
            parent_kind=parent_kind,
            available=(
                stage != 1 or parent_kind != "strongest_other_family"
            ),
            parent_node_id=(
                None
                if stage == 1 and parent_kind == "strongest_other_family"
                else "root"
            ),
            update=update,
            training_loss=(
                None if update in (None, 0) else 0.5
            ),
            candidate_accuracy=(
                None
                if stage == 1 and parent_kind == "strongest_other_family"
                else 0.75
            ),
            correct_answer_nll=(
                None
                if stage == 1 and parent_kind == "strongest_other_family"
                else 0.3
            ),
            adapter_sha256=(
                None
                if stage == 1 and parent_kind == "strongest_other_family"
                else "d" * 64
            ),
            final_update=(
                None
                if stage == 1 and parent_kind == "strongest_other_family"
                else 4
            ),
        )
        for stage in TINYWORLDS_REPORT_STAGES
        for parent_kind in TINYWORLDS_PARENT_COUNTERFACTUALS
        for update in (
            (None,)
            if stage == 1 and parent_kind == "strongest_other_family"
            else (0, 1, 2, 4)
        )
    )
    graph_recovery = tuple(
        _record(
            task_id=task_id,
            expected_parent_id="root",
            learned_parent_id="root",
            recovered=True,
        )
        for task_id in TINYWORLDS_REPORT_TASK_IDS
    )
    revision_retention = tuple(
        _record(
            stage=8,
            family_id=family_id,
            old_context_accuracy=0.8,
            revision_context_accuracy=0.8,
            paired_revision_consistency=0.7,
        )
        for family_id in ("willow", "sunny")
    )
    drift = tuple(
        _record(
            stage=stage,
            node_id=TINYWORLDS_REPORT_TASK_IDS[stage - 1],
            logit_max_abs_drift=0.0,
            answer_change_count=0,
            checksum_match=True,
        )
        for stage in TINYWORLDS_REPORT_STAGES
    )
    memory = tuple(
        _record(
            stage=stage,
            persistent_bytes=1_000 * stage,
            runtime_bytes=2_000 * stage,
            allocator_peak_bytes=3_000 * stage,
        )
        for stage in TINYWORLDS_REPORT_STAGES
    )
    addressing = tuple(
        _record(
            stage=stage,
            method=method,
            cold_seconds=0.1,
            warm_seconds=0.05,
        )
        for stage in TINYWORLDS_REPORT_STAGES
        for method in TINYWORLDS_ADDRESSING_METHODS
    )
    gates = tuple(
        _record(gate=name, category="implementation", passed=True)
        for name in TINYWORLDS_IMPLEMENTATION_GATES
    ) + (
        _record(
            gate="independent_direct_recall_hypothesis",
            category="scientific",
            passed=False,
        ),
    )
    selection = tuple(
        _record(
            record_id=f"query-{index}",
            split="test",
            used_for_tuning=False,
            used_for_parent_selection=False,
        )
        for index in range(len(TINYWORLDS_REPORT_TASK_IDS))
    )
    return TinyWorldsCompletedResult(
        manifest=_manifest(),
        method_evaluations=_method_evaluations(),
        natural_continuation_metrics=natural,
        parent_search=parent_search,
        checkpointed_transfer=transfer,
        graph_recovery=graph_recovery,
        revision_retention=revision_retention,
        committed_node_drift=drift,
        memory_metrics=memory,
        addressing_cost=addressing,
        gate_results=gates,
        representative_queries=(
            _record(
                query_id="query-0",
                proof_id="proof-0",
                query_text="Which nonce attribute is correct?",
                answer_text="answer-0-0",
                support_ids="fact-0",
            ),
        ),
        selection_audit=selection,
        sequential_results=tuple(
            _record(sequence_index=stage - 1, stage=stage, event="stage_completed")
            for stage in TINYWORLDS_REPORT_STAGES
        ),
    )


def _tree(directory: Path) -> dict[str, bytes]:
    return {
        path.name: path.read_bytes()
        for path in sorted(directory.iterdir())
        if path.is_file()
    }


def test_report_identity_covers_every_required_policy_field() -> None:
    manifest = _manifest()

    assert manifest.run_id.startswith("single-gpu-seed0-")
    assert len(manifest.run_id.rsplit("-", 1)[1]) == 12
    assert tinyworlds_report_directory("results", manifest).parts[-3:] == (
        "tinyworlds-v1",
        "knowledge-graph",
        manifest.run_id,
    )
    config = json.loads(manifest.config_json)
    assert set(TINYWORLDS_REQUIRED_CONFIG_FIELDS).issubset(config)
    config.pop("candidate_policy")
    with pytest.raises(ValueError, match="omits required"):
        TinyWorldsReportManifest(
            preset="single-gpu",
            seed=0,
            config_json=canonical_tinyworlds_config_json(config),
        )


def test_completed_result_projects_and_writes_byte_identically(
    tmp_path: Path,
    completed_result: TinyWorldsCompletedResult,
) -> None:
    bundle = build_tinyworlds_report_bundle(completed_result)
    first = write_tinyworlds_report(tmp_path / "first", bundle)
    second = write_tinyworlds_report(tmp_path / "second", bundle)

    assert _tree(first) == _tree(second)
    assert set(_tree(first)) == set(TINYWORLDS_REPORT_FILENAMES)
    candidate_rows = (first / "candidate_scores.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(candidate_rows) == (
        len(TINYWORLDS_REPORT_STAGES)
        * len(TINYWORLDS_REPORT_METHODS)
        * len(TINYWORLDS_REPORT_TASK_IDS)
    )
    html_text = (first / "report.html").read_text(encoding="utf-8")
    assert bundle.manifest.run_id in html_text
    assert "Scientific hypothesis failures remain visible" in html_text
    assert "candidate_reasoning.svg" in html_text
    assert "expected_vs_learned_graph.svg" in html_text
    assert "in-domain topic specialization" not in html_text
    (first / "candidate_reasoning.svg").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        validate_tinyworlds_report_artifact(first, bundle)


def test_validated_report_is_atomically_promoted(
    tmp_path: Path,
    completed_result: TinyWorldsCompletedResult,
) -> None:
    bundle = build_tinyworlds_report_bundle(completed_result)
    staging = write_tinyworlds_report(tmp_path / "staging", bundle)

    promoted = atomically_promote_tinyworlds_report(
        staging,
        tmp_path / "results",
        bundle,
    )

    assert not staging.exists()
    assert promoted == tinyworlds_report_directory(
        tmp_path / "results", bundle.manifest
    )
    assert set(_tree(promoted)) == set(TINYWORLDS_REPORT_FILENAMES)


def test_strict_coverage_drift_selection_and_gate_validation(
    completed_result: TinyWorldsCompletedResult,
) -> None:
    with pytest.raises(ValueError, match="coverage mismatch"):
        replace(
            completed_result,
            method_evaluations=completed_result.method_evaluations[:-1],
        )
    missing_checkpoint = tuple(
        row
        for row in completed_result.checkpointed_transfer
        if not (
            row.require("stage") == 1
            and row.require("parent_kind") == "root"
            and row.require("update") == 1
        )
    )
    with pytest.raises(ValueError, match="exact checkpoint schedule"):
        replace(
            completed_result,
            checkpointed_transfer=missing_checkpoint,
        )
    bad_drift = _record(
        stage=1,
        node_id="willow_seed",
        logit_max_abs_drift=0.001,
        answer_change_count=0,
        checksum_match=True,
    )
    with pytest.raises(ValueError, match="exactly zero"):
        replace(
            completed_result,
            committed_node_drift=(
                bad_drift,
                *completed_result.committed_node_drift[1:],
            ),
        )
    selected_test = _record(
        record_id="query-0",
        split="test",
        used_for_tuning=True,
        used_for_parent_selection=False,
    )
    with pytest.raises(ValueError, match="test records"):
        replace(
            completed_result,
            selection_audit=(selected_test, *completed_result.selection_audit[1:]),
        )
    failed_implementation = _record(
        gate=TINYWORLDS_IMPLEMENTATION_GATES[0],
        category="implementation",
        passed=False,
    )
    with pytest.raises(ValueError, match="implementation gates"):
        replace(
            completed_result,
            gate_results=(
                failed_implementation,
                *completed_result.gate_results[1:],
            ),
        )
    assert any(
        row.require("category") == "scientific"
        and row.require("passed") is False
        for row in completed_result.gate_results
    )
    with pytest.raises(TypeError, match="finite JSON scalars"):
        _record(loss=float("nan"))


def test_progress_and_sequential_jsonl_are_durably_batched(tmp_path: Path) -> None:
    writer = TinyWorldsProgressWriter(tmp_path, batch_size=2)
    first = TinyWorldsProgressEvent(
        event="phase_started",
        phase=1,
        phase_count=2,
        name="pilot",
        completed_units=0.0,
        total_units=10.0,
        elapsed_seconds=0.0,
        eta_seconds=10.0,
    )
    second = replace(
        first,
        event="phase_progress",
        completed_units=1.0,
        elapsed_seconds=1.0,
        eta_seconds=9.0,
    )
    writer.append_progress(first)
    assert not writer.progress_path.exists()
    writer.append_progress(second)
    assert len(writer.progress_path.read_text(encoding="utf-8").splitlines()) == 2

    writer.append_sequential(
        TinyWorldsSequentialResult(0, 1, _record(event="stage_completed"))
    )
    assert not writer.sequential_results_path.exists()
    with pytest.raises(ValueError, match="contiguously"):
        writer.append_sequential(
            TinyWorldsSequentialResult(2, 2, _record(event="stage_completed"))
        )
    writer.append_sequential(
        TinyWorldsSequentialResult(1, 2, _record(event="stage_completed"))
    )
    assert len(
        writer.sequential_results_path.read_text(encoding="utf-8").splitlines()
    ) == 2
    writer.close()
    with pytest.raises(RuntimeError, match="closed"):
        writer.append_progress(first)


def test_fixed_pilot_script_has_no_research_cli_and_prints_temp_first(
    tmp_path: Path,
    completed_result: TinyWorldsCompletedResult,
    capsys: pytest.CaptureFixture[str],
) -> None:
    script_path = Path(__file__).parents[1] / "scripts" / "run_tinyworlds_pilot.py"
    source = script_path.read_text(encoding="utf-8")
    assert "argparse" not in source
    namespace = runpy.run_path(str(script_path), run_name="tinyworlds_pilot_test")
    phases = namespace["PHASES"]
    assert tuple(phase.number for phase in phases) == (1, 2, 3, 4)

    destination = namespace["main"](
        executor=lambda _temporary, _progress: completed_result,
        results_root=tmp_path / "results",
    )

    stdout = capsys.readouterr().out.splitlines()
    assert stdout
    assert Path(stdout[0]).name.startswith("tinyworlds-v1-pilot-")
    assert destination.is_dir()

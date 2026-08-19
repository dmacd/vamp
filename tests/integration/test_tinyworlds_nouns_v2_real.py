from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from apm.data.text.tinyworlds_nouns_v1.partition import load_noun_partition
from apm.data.text.tinyworlds_nouns_v2.contracts import (
    BASE_UNIVERSE_STORY_COUNT,
    EXCLUDED_TRAIN_STORY_COUNT,
    EXCLUDED_VALIDATION_STORY_COUNT,
    EXPECTED_PURE_COUNTS,
    PARENT_PARTITION_SHA256,
    PURE_TASK_TRAIN_STORY_COUNT,
    PURE_TASK_VALIDATION_STORY_COUNT,
    STAGEWISE_CASE_COUNT,
    TRAIN_UNIQUE_STORY_COUNT,
)
from apm.data.text.tinyworlds_nouns_v2.partition import (
    authenticate_parent_manifest,
    find_partition,
    verify_byte_identical_rebuild,
)


pytestmark = pytest.mark.integration
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PARENT_ROOT = (
    REPOSITORY_ROOT
    / "data/tinyworlds-nouns-v1/partitions"
    / PARENT_PARTITION_SHA256
)
V2_ROOT = REPOSITORY_ROOT / "data/tinyworlds-nouns-v2"
V2_CHECKPOINT_ROOT = REPOSITORY_ROOT / "checkpoints/tinyworlds-nouns-v2"
V2_RESULT_ROOT = REPOSITORY_ROOT / "results/language_cl/tinyworlds-nouns-v2"


def test_real_24_task_counts_and_byte_identical_rebuild() -> None:
    manifest = authenticate_parent_manifest(PARENT_ROOT)
    partition = find_partition(manifest, V2_ROOT)
    assert partition is not None
    assert partition.base_universe_story_count == BASE_UNIVERSE_STORY_COUNT
    assert partition.base_universe_story_count / TRAIN_UNIQUE_STORY_COUNT == pytest.approx(
        0.8136, abs=5e-5
    )
    assert sum(task.train_story_count for task in partition.tasks) == (
        PURE_TASK_TRAIN_STORY_COUNT
    )
    assert sum(task.validation_story_count for task in partition.tasks) == (
        PURE_TASK_VALIDATION_STORY_COUNT
    )
    assert partition.excluded_train_story_count == EXCLUDED_TRAIN_STORY_COUNT
    assert partition.excluded_validation_story_count == EXCLUDED_VALIDATION_STORY_COUNT
    assert tuple(
        (task.task_id, task.train_story_count, task.validation_story_count)
        for task in partition.tasks
    ) == EXPECTED_PURE_COUNTS
    assert sum(
        task.validation_story_count * (len(partition.tasks) - index)
        for index, task in enumerate(partition.tasks)
    ) == STAGEWISE_CASE_COUNT
    verify_byte_identical_rebuild(partition, manifest)


def test_completed_nouns_v1_strict_load_and_published_hashes_are_unchanged() -> None:
    partition = load_noun_partition(PARENT_ROOT / "partition.json")
    assert partition.partition_sha256 == PARENT_PARTITION_SHA256
    expected = {
        "results/language_cl/tinyworlds-nouns-v1/report.md": (
            "74f0035c755f95ff57624f8270615f3c9171f7b792d7b5ee22bae147ac15c4ae"
        ),
        "results/language_cl/tinyworlds-nouns-v1/report.html": (
            "9ef9cfea2c827836da8999c3d032fa63ff3a109664ee65b250066577a09d1526"
        ),
        "results/language_cl/tinyworlds-nouns-v1/run-manifest.json": (
            "fffa0e0f64f1c63ae3efb16363042907169737dc6858944cbcd0f46b178cc628"
        ),
    }
    for relative, digest in expected.items():
        assert sha256((REPOSITORY_ROOT / relative).read_bytes()).hexdigest() == digest


def test_all_real_v2_vamp_stages_strict_load_as_one_immutable_prefix() -> None:
    manifest = authenticate_parent_manifest(PARENT_ROOT)
    partition = find_partition(manifest, V2_ROOT)
    assert partition is not None

    from apm.data.text.tinyworlds_nouns_v2.contracts import NounsV2ExperimentPreset
    from apm.data.text.tinyworlds_nouns_v2.compact_stagewise import (
        load_compact_stagewise_contract,
        validate_compact_stagewise_ledger,
    )
    from apm.data.text.tinyworlds_nouns_v2.experiment import (
        load_nouns_v2_vamp_stages,
        run_or_load_nouns_v2_gpu_preflight,
        run_or_resume_nouns_v2_base,
    )

    preset = NounsV2ExperimentPreset()
    preflight = run_or_load_nouns_v2_gpu_preflight(
        partition,
        preset,
        V2_CHECKPOINT_ROOT,
    )
    selected_base = run_or_resume_nouns_v2_base(
        partition,
        preset,
        preflight,
        V2_CHECKPOINT_ROOT,
    )
    stages = load_nouns_v2_vamp_stages(
        partition,
        preset,
        selected_base,
        V2_CHECKPOINT_ROOT,
    )
    assert len(stages) == 24
    assert tuple(
        tuple(str(task) for task in stage.task_order) for stage in stages
    ) == tuple(partition.task_ids[:index] for index in range(1, 25))
    assert stages[-1].tensor_checksum == (
        "97414ac3d8656ab083b2e570a4162dc69b024f90cf819b80b1cab94213553e63"
    )
    compact_contract = load_compact_stagewise_contract(
        V2_RESULT_ROOT / "compact-stagewise-contract.json"
    )
    compact_ledger = V2_RESULT_ROOT / "compact-stagewise-cl.jsonl"
    assert len(
        validate_compact_stagewise_ledger(
            compact_ledger,
            str(compact_contract["contract_sha256"]),
            partition,
            stages,
            require_complete=True,
        )
    ) == STAGEWISE_CASE_COUNT
    run = json.loads((V2_RESULT_ROOT / "run-manifest.json").read_bytes())
    assert sha256(compact_ledger.read_bytes()).hexdigest() == run[
        "compact_stagewise_sha256"
    ]


def test_real_v2_baselines_and_comparison_ledger_strict_load() -> None:
    manifest = authenticate_parent_manifest(PARENT_ROOT)
    partition = find_partition(manifest, V2_ROOT)
    assert partition is not None

    from apm.data.text.tinyworlds_nouns_v2.baseline_stagewise import (
        validate_baseline_stagewise_ledger,
    )
    from apm.data.text.tinyworlds_nouns_v2.contracts import NounsV2ExperimentPreset
    from apm.data.text.tinyworlds_nouns_v2.experiment import (
        load_nouns_v2_baseline_stages,
        load_nouns_v2_vamp_stages,
        run_or_load_nouns_v2_gpu_preflight,
        run_or_resume_nouns_v2_base,
    )

    preset = NounsV2ExperimentPreset()
    preflight = run_or_load_nouns_v2_gpu_preflight(
        partition,
        preset,
        V2_CHECKPOINT_ROOT,
    )
    selected_base = run_or_resume_nouns_v2_base(
        partition,
        preset,
        preflight,
        V2_CHECKPOINT_ROOT,
    )
    vamp_stages = load_nouns_v2_vamp_stages(
        partition,
        preset,
        selected_base,
        V2_CHECKPOINT_ROOT,
    )
    baseline_stages = load_nouns_v2_baseline_stages(
        partition,
        preset,
        selected_base,
        vamp_stages,
        V2_CHECKPOINT_ROOT,
    )
    assert len(baseline_stages) == 24
    assert tuple(
        tuple(str(task) for task in stage.task_order)
        for stage in baseline_stages
    ) == tuple(partition.task_ids[:index] for index in range(1, 25))

    ledger = V2_RESULT_ROOT / "baseline-stagewise-cl.jsonl"
    assert len(
        validate_baseline_stagewise_ledger(
            ledger,
            partition,
            baseline_stages,
            vamp_stages,
            require_complete=True,
        )
    ) == STAGEWISE_CASE_COUNT
    run = json.loads((V2_RESULT_ROOT / "run-manifest.json").read_bytes())
    assert baseline_stages[-1].tensor_checksum == run["baseline_tensor_checksum"]
    assert sha256(ledger.read_bytes()).hexdigest() == run[
        "baseline_stagewise_sha256"
    ]


def test_real_v2_full_finetune_and_stagewise_ledger_strict_load() -> None:
    manifest = authenticate_parent_manifest(PARENT_ROOT)
    partition = find_partition(manifest, V2_ROOT)
    assert partition is not None

    from apm.data.text.tinyworlds_nouns_v2.contracts import NounsV2ExperimentPreset
    from apm.data.text.tinyworlds_nouns_v2.experiment import (
        load_nouns_v2_full_finetune_stages,
        load_nouns_v2_vamp_stages,
        run_or_load_nouns_v2_gpu_preflight,
        run_or_resume_nouns_v2_base,
    )
    from apm.data.text.tinyworlds_nouns_v2.full_finetune_stagewise import (
        validate_full_finetune_stagewise_ledger,
    )

    preset = NounsV2ExperimentPreset()
    preflight = run_or_load_nouns_v2_gpu_preflight(
        partition,
        preset,
        V2_CHECKPOINT_ROOT,
    )
    selected_base = run_or_resume_nouns_v2_base(
        partition,
        preset,
        preflight,
        V2_CHECKPOINT_ROOT,
    )
    vamp_stages = load_nouns_v2_vamp_stages(
        partition,
        preset,
        selected_base,
        V2_CHECKPOINT_ROOT,
    )
    full_stages = load_nouns_v2_full_finetune_stages(
        partition,
        preset,
        selected_base,
        V2_CHECKPOINT_ROOT,
    )
    assert len(full_stages) == 24
    assert tuple(stage.task_order for stage in full_stages) == tuple(
        partition.task_ids[:index] for index in range(1, 25)
    )

    ledger = V2_RESULT_ROOT / "full-finetune-stagewise-cl.jsonl"
    assert len(
        validate_full_finetune_stagewise_ledger(
            ledger,
            partition,
            full_stages,
            vamp_stages,
            require_complete=True,
        )
    ) == STAGEWISE_CASE_COUNT
    run = json.loads((V2_RESULT_ROOT / "run-manifest.json").read_bytes())
    assert full_stages[-1].parameter_checksum == run[
        "full_finetune_parameter_checksum"
    ]
    assert full_stages[-1].run_sha256 == run["full_finetune_run_sha256"]
    assert sha256(ledger.read_bytes()).hexdigest() == run[
        "full_finetune_stagewise_sha256"
    ]


def test_completed_bounded_addressing_study_strict_loads_every_artifact() -> None:
    from apm.data.text.tinyworlds_nouns_v2.addressing_study import (
        assert_canonical_hashes_unchanged,
        authenticate_addressing_study_inputs,
        build_study_contracts,
        expected_ebt_keys,
        expected_retrieval_keys,
        load_timing_ledger,
        validate_ebt_ledger,
        validate_retrieval_ledger,
    )
    from apm.data.text.tinyworlds_nouns_v2.addressing_study_contracts import (
        EBT_CONTRACT_FORMAT,
        EBT_ROW_COUNT,
        RETRIEVAL_CONTRACT_FORMAT,
        RETRIEVAL_ROW_COUNT,
        load_contract,
        record_sha256,
    )
    from apm.data.text.tinyworlds_nouns_v2.addressing_study_keys import (
        load_addressing_keys,
    )

    study = V2_RESULT_ROOT / "addressing-study"
    inputs = authenticate_addressing_study_inputs(REPOSITORY_ROOT)
    retrieval_contract = load_contract(
        study / "retrieval-contract.json",
        RETRIEVAL_CONTRACT_FORMAT,
    )
    ebt_contract = load_contract(study / "ebt-contract.json", EBT_CONTRACT_FORMAT)
    keys = load_addressing_keys(study / "keys")
    assert len(keys.node_ids) == 25
    assert tuple(map(len, keys.probe_story_ids)) == (36,) * 25
    replayed_retrieval, replayed_ebt = build_study_contracts(inputs, keys, study)
    assert replayed_retrieval == retrieval_contract
    assert replayed_ebt == ebt_contract
    assert len(
        validate_retrieval_ledger(
            study / "retrieval.jsonl",
            str(retrieval_contract["contract_sha256"]),
            expected_retrieval_keys(inputs),
            require_complete=True,
        )
    ) == RETRIEVAL_ROW_COUNT
    assert len(
        validate_ebt_ledger(
            study / "ebt.jsonl",
            str(ebt_contract["contract_sha256"]),
            expected_ebt_keys(inputs),
            require_complete=True,
        )
    ) == EBT_ROW_COUNT
    assert load_timing_ledger(
        study / "timing.jsonl",
        str(ebt_contract["contract_sha256"]),
    )
    manifest = json.loads((study / "manifest.json").read_bytes())
    manifest_core = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    assert manifest["manifest_sha256"] == record_sha256(manifest_core)
    assert all(
        sha256((study / relative).read_bytes()).hexdigest() == digest
        for relative, digest in manifest["artifacts"].items()
    )
    for width in (4, 8):
        dot = (study / f"vamp-graph-top{width}.dot").read_text(encoding="utf-8")
        assert sum(" -> " in line for line in dot.splitlines()) == 24
    html = (study / "report.html").read_text(encoding="utf-8")
    assert html.startswith("<!doctype html>")
    assert html.count("<details>") >= 5
    assert html.count('role="img"') >= 4
    assert "<script src=" not in html and "<link " not in html
    assert_canonical_hashes_unchanged(REPOSITORY_ROOT, inputs.canonical_hashes)


def test_temporal_consolidation_real_sources_authenticate_without_mutation() -> None:
    from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation import (
        assert_canonical_artifacts_unchanged,
        authenticate_temporal_study_inputs,
        study_dashboard_jobs,
    )
    from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_contracts import (
        ARRIVAL_COUNT,
        CONTRACT_FORMAT,
        SHARDS_PER_TASK,
        STORIES_PER_SHARD,
        TASK_STORY_COUNT,
        TEMPORAL_ORDERS,
        expected_final_intervals,
        simulate_hierarchy,
    )

    inputs = authenticate_temporal_study_inputs(REPOSITORY_ROOT)
    assert inputs.contract["format"] == CONTRACT_FORMAT
    assert inputs.contract_sha256 == (
        "3f4ef4a10fd471b418a32a8f7b45431602c1f6abc080c19a7822ea2c2dd839b4"
    )
    assert len(inputs.shards) == ARRIVAL_COUNT
    assert all(len(shard.story_ids) == STORIES_PER_SHARD for shard in inputs.shards)
    assert all(
        sum(shard.task_id == task_id for shard in inputs.shards) == SHARDS_PER_TASK
        for task_id in inputs.partition.task_ids
    )
    assert len({story for shard in inputs.shards for story in shard.story_ids}) == (
        len(inputs.partition.task_ids) * TASK_STORY_COUNT
    )
    assert sum(len(entries) for _, entries in inputs.validation_entries) == 4_440
    assert tuple(task for task, _ in inputs.sentinel) == tuple(inputs.partition.task_ids)
    assert all(len(stories) == 16 for _, stories in inputs.sentinel)
    for order in TEMPORAL_ORDERS:
        state, merges = simulate_hierarchy(inputs.shards, order)
        assert len(merges) == 183
        assert tuple(
            (chunk.start_arrival, chunk.end_arrival)
            for chunk in state.active_chunks
        ) == expected_final_intervals()
    jobs = study_dashboard_jobs(inputs)
    assert len(jobs) == 26
    assert all(job.total > 0 for job in jobs)
    assert_canonical_artifacts_unchanged(inputs)

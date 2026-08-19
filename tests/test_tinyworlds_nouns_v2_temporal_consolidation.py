from __future__ import annotations

import ast
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
import urllib.error
import urllib.request

import pytest
import jax
import jax.numpy as jnp
import numpy as np

import apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_timing as timing_module
from apm.data.text.tinyworlds_nouns_v1.experiment import StoryIndexEntry
from apm.data.text.tinyworlds_nouns_v2.contracts import TASK_IDS
from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_contracts import (
    ARRIVAL_COUNT,
    CONTRACT_FORMAT,
    LEVEL_CAPACITY,
    MERGE_ROW_FORMAT,
    PROGRESS_ROW_FORMAT,
    SHARDS_PER_TASK,
    STORIES_PER_SHARD,
    TemporalShard,
    TIMING_ROW_FORMAT,
    build_contract_record,
    empty_hierarchy,
    expected_final_intervals,
    insert_arrival,
    select_temporal_shards,
    select_validation_sentinel,
    simulate_hierarchy,
    temporal_arrivals,
    validate_contract_record,
)
from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_dashboard import (
    ProgressRecorder,
    StudyJob,
    publish_frozen_dashboard,
    start_dashboard_server,
)
from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_evaluation import (
    AdapterCandidate,
    MidpointCase,
    build_adapter_bank,
    evaluate_case_batch,
    evaluate_to_ledger,
    prefix_router_row_capacity,
)
from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_distortion import (
    expected_distortion_keys,
    require_lineage_identities,
    validate_distortion_rows,
)
from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation import (
    OrderingArtifacts,
)
from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_report import (
    _accessible_svg,
    _compact_lineage_dot,
    _full_lineage_dot,
    _html_report,
    _markdown_report,
    _publish_lineage_graphs,
    _publish_plots,
)
from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_timing import (
    validate_timing_rows,
)
from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_contracts import (
    EVALUATION_ROW_FORMAT,
)
from apm.continual.artifacts import (
    ChainedJsonlLedger,
    load_canonical_json,
    publish_immutable_json,
)
from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_training import (
    StoryEpochBatches,
    TrainingInterrupted,
    TrainingJob,
    _compiled_full_model_train_step,
    _compiled_lora_train_step,
    train_or_load_full_model,
    train_or_load_lora,
)
from apm.lm.checkpoint import (
    CheckpointFileHash,
    SourceCheckpointMetadata,
    TokenizerCheckpointMetadata,
    save_gpt_neo_checkpoint,
)
from apm.lm.parameters import init_gpt_neo_params
from apm.lm.lora import LoraConfig, init_lora_edge
from apm.lm.text_data import TokenBatch
from apm.lm.workflow import tiny_shakespeare_unit_model_config
from apm.continual.language_tasks import RouterBatch


def _digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _entry(value: str, index: int) -> StoryIndexEntry:
    return StoryIndexEntry(
        story_id=_digest(value),
        story_index=index,
        story_offset=index,
        byte_length=1,
        token_offset=index * 2,
        token_count=2,
    )


def _shards() -> tuple[TemporalShard, ...]:
    return tuple(
        TemporalShard(
            shard_id=_digest(f"shard:{task}:{shard_index}"),
            task_id=task,
            shard_index=shard_index,
            story_ids=tuple(
                _digest(f"story:{task}:{shard_index}:{row}")
                for row in range(STORIES_PER_SHARD)
            ),
        )
        for task in TASK_IDS
        for shard_index in range(SHARDS_PER_TASK)
    )


def test_default_runner_is_one_fixed_gpu_zero_workflow_without_options() -> None:
    runner = (
        Path(__file__).resolve().parents[1]
        / "scripts/run_tinyworlds_nouns_v2_temporal_consolidation.py"
    )
    source = runner.read_text(encoding="utf-8")
    module = ast.parse(source)
    main_function = next(
        node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "main"
    )
    imports = {
        alias.name
        for node in module.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "argparse" not in imports
    assert not main_function.args.args
    assert 'os.environ["CUDA_VISIBLE_DEVICES"] = "0"' in source
    assert 'os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"' in source
    assert 'os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "cuda_async"' in source
    assert "--xla_gpu_enable_command_buffer=" in source
    assert "start_dashboard_server" in source
    assert "Persistent temporary directory:" in source


def test_training_reuses_one_compiled_wrapper_per_model_architecture() -> None:
    config = tiny_shakespeare_unit_model_config(vocab_size=32)
    assert _compiled_lora_train_step(config) is _compiled_lora_train_step(config)
    assert _compiled_full_model_train_step(config) is _compiled_full_model_train_step(
        config
    )


def test_b2_schedule_has_exact_final_intervals_and_oldest_first_carries() -> None:
    shards = _shards()
    for order in ("blocked", "round_robin"):
        state = empty_hierarchy(order)
        merge_count = 0
        for arrival_index, shard in enumerate(temporal_arrivals(shards, order), start=1):
            state, merges = insert_arrival(state, shard)
            merge_count += len(merges)
            assert state.arrival_count == arrival_index
            assert all(len(level) <= LEVEL_CAPACITY for level in state.levels)
            assert len(state.active_chunks) <= LEVEL_CAPACITY * (
                1 + (arrival_index.bit_length() - 1)
            )
            assert all(merge.left.end_arrival < merge.right.start_arrival for merge in merges)
        assert merge_count == 183
        assert tuple(
            (chunk.start_arrival, chunk.end_arrival) for chunk in state.active_chunks
        ) == expected_final_intervals()
        assert len(state.active_chunks) == 9


def test_two_orders_reuse_shards_but_change_arrival_locality() -> None:
    shards = _shards()
    blocked = temporal_arrivals(shards, "blocked")
    round_robin = temporal_arrivals(shards, "round_robin")
    assert {shard.shard_id for shard in blocked} == {
        shard.shard_id for shard in round_robin
    }
    assert len({shard.task_id for shard in blocked[:8]}) == 1
    assert tuple(shard.task_id for shard in round_robin[:24]) == TASK_IDS
    blocked_state, blocked_merges = simulate_hierarchy(shards, "blocked")
    interleaved_state, interleaved_merges = simulate_hierarchy(shards, "round_robin")
    assert len(blocked_merges) == len(interleaved_merges) == 183
    assert interleaved_state.active_chunks[0].noun_entropy > (
        blocked_state.active_chunks[0].noun_entropy
    )


def test_selection_excludes_probes_and_validation_and_is_deterministic() -> None:
    entries = {
        task: tuple(
            _entry(f"train:{task}:{index}", task_index * 5_000 + index)
            for index in range(4_140)
        )
        for task_index, task in enumerate(TASK_IDS)
    }
    probes = {
        task: tuple(entry.story_id for entry in entries[task][:36])
        for task in TASK_IDS
    }
    validation = tuple(_digest(f"validation:{task}:{index}") for task in TASK_IDS for index in range(185))
    first = select_temporal_shards(entries, probes, validation)
    second = select_temporal_shards(entries, probes, validation)
    selected_ids = {story for shard in first for story in shard.story_ids}
    assert first == second
    assert len(first) == ARRIVAL_COUNT
    assert len(selected_ids) == ARRIVAL_COUNT * STORIES_PER_SHARD
    assert not selected_ids & {story for values in probes.values() for story in values}
    assert not selected_ids & set(validation)


def test_sentinel_and_contract_are_independently_hashed() -> None:
    validation_entries = {
        task: tuple(
            _entry(f"validation:{task}:{index}", task_index * 200 + index)
            for index in range(185)
        )
        for task_index, task in enumerate(TASK_IDS)
    }
    sentinel = select_validation_sentinel(validation_entries)
    contract = build_contract_record(
        bindings={"base_parameter_checksum": _digest("base")},
        shards=_shards(),
        sentinel=sentinel,
    )
    assert contract["format"] == CONTRACT_FORMAT
    assert validate_contract_record(contract) == contract
    mutated = json.loads(json.dumps(contract))
    mutated["training"]["epochs"] = 5
    with pytest.raises(ValueError, match="identity changed"):
        validate_contract_record(mutated)


def test_chained_ledger_repairs_only_an_incomplete_tail_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"
    ledger = ChainedJsonlLedger(path, PROGRESS_ROW_FORMAT)
    first = ledger.append({"name": "one"})
    second = ledger.append({"name": "two"})
    assert second["previous_sha256"] == first["result_sha256"]
    with path.open("ab") as output:
        output.write(b'{"partial"')
    repaired = ChainedJsonlLedger(path, PROGRESS_ROW_FORMAT)
    assert [row["name"] for row in repaired.rows] == ["one", "two"]
    lines = path.read_bytes().splitlines(keepends=True)
    altered = json.loads(lines[0])
    altered["name"] = "changed"
    lines[0] = json.dumps(altered, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    path.write_bytes(b"".join(lines))
    with pytest.raises(ValueError, match="hash changed"):
        ChainedJsonlLedger(path, PROGRESS_ROW_FORMAT)


def test_immutable_json_rejects_republication_with_different_bytes(tmp_path: Path) -> None:
    path = tmp_path / "contract.json"
    publish_immutable_json(path, {"a": 1})
    publish_immutable_json(path, {"a": 1})
    assert load_canonical_json(path) == {"a": 1}
    with pytest.raises(ValueError, match="immutable JSON changed"):
        publish_immutable_json(path, {"a": 2})


def test_lora_training_resume_matches_uninterrupted_tensor_identity(tmp_path: Path) -> None:
    config = tiny_shakespeare_unit_model_config(vocab_size=32)
    params = init_gpt_neo_params(jax.random.PRNGKey(7), config, dtype=jnp.float32)
    rows, width = 32, 8
    inputs = np.tile(np.arange(width, dtype=np.int32), (rows, 1)) % 31
    targets = (inputs + 1) % 31
    batch = TokenBatch(
        inputs,
        np.ones((rows, width), dtype=np.bool_),
        targets,
        np.ones((rows, width), dtype=np.bool_),
    )
    batches = (batch, batch, batch)
    job = TrainingJob(
        contract_sha256=_digest("contract"),
        job_id="fixture-lora",
        family="level_zero",
        source_story_ids=(_digest("story-a"), _digest("story-b")),
        source_shard_ids=(_digest("shard"),),
    )
    with pytest.raises(TrainingInterrupted):
        train_or_load_lora(
            job,
            batches,
            params,
            config,
            tmp_path / "resumed-output",
            tmp_path / "resumed-work",
            stop_after_update=2,
        )
    resumed = train_or_load_lora(
        job,
        batches,
        params,
        config,
        tmp_path / "resumed-output",
        tmp_path / "resumed-work",
    )
    uninterrupted = train_or_load_lora(
        job,
        batches,
        params,
        config,
        tmp_path / "direct-output",
        tmp_path / "direct-work",
    )
    assert resumed.adapter_sha256 == uninterrupted.adapter_sha256
    assert resumed.loss_trace_sha256 == uninterrupted.loss_trace_sha256
    assert resumed.optimizer_updates == uninterrupted.optimizer_updates == 3


def test_story_epoch_batches_cover_every_transition_and_mask_padding(
    tmp_path: Path,
) -> None:
    stories = ((1, 2, 3), (4, 5, 6, 7, 8, 9), (10, 11))
    token_path = tmp_path / "tokens.uint16"
    np.asarray([token for story in stories for token in story], dtype="<u2").tofile(
        token_path
    )
    (tmp_path / "stories.bin").write_bytes(b"abc")
    offsets = np.cumsum((0,) + tuple(len(story) for story in stories))
    entries = tuple(
        StoryIndexEntry(
            _digest(f"epoch-story-{index}"),
            index,
            index,
            1,
            int(offsets[index]),
            len(story),
        )
        for index, story in enumerate(stories)
    )
    partition = SimpleNamespace(
        benchmark_id="fixture",
        pad_token_id=0,
        partition_sha256=_digest("partition"),
        root=tmp_path,
        story_store_path=tmp_path / "stories.bin",
        token_store_path=token_path,
    )
    first = StoryEpochBatches(
        partition,
        entries,
        context_length=4,
        batch_size=3,
        namespace="coverage",
        epochs=2,
    )
    second = StoryEpochBatches(
        partition,
        entries,
        context_length=4,
        batch_size=3,
        namespace="coverage",
        epochs=2,
    )
    assert len(first) == 4
    assert first.window_count_per_epoch == 4
    expected_active = sum(len(story) - 1 for story in stories)
    assert tuple(
        sum(int(np.sum(first[index].loss_mask)) for index in range(epoch * 2, epoch * 2 + 2))
        for epoch in range(2)
    ) == (expected_active, expected_active)
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left.input_ids, right.input_ids)
        np.testing.assert_array_equal(left.target_ids, right.target_ids)
        np.testing.assert_array_equal(left.loss_mask, right.loss_mask)
    assert np.all(first[1].loss_mask[-1] == 0)
    assert np.all(first[3].loss_mask[-1] == 0)


def test_midpoint_router_is_suffix_isolated_and_evaluation_resume_is_exact(
    tmp_path: Path,
) -> None:
    config = tiny_shakespeare_unit_model_config(vocab_size=32)
    params = init_gpt_neo_params(jax.random.PRNGKey(3), config, dtype=jnp.float32)
    lora_config = LoraConfig(rank=8, alpha=8.0)
    adapter = init_lora_edge(jax.random.PRNGKey(4), config, lora_config)
    candidate = AdapterCandidate(
        "mouse-adapter",
        _digest("adapter"),
        adapter,
        (("mouse", 8),),
    )
    bank = build_adapter_bank((candidate,), config)
    prefix = RouterBatch(
        np.asarray([[1, 2, 3, 4]], dtype=np.int32),
        np.ones((1, 4), dtype=np.bool_),
        np.asarray([[2, 3, 4, 5]], dtype=np.int32),
        np.ones((1, 4), dtype=np.bool_),
    )
    suffix_a = TokenBatch(
        np.asarray([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int32),
        np.ones((1, 8), dtype=np.bool_),
        np.asarray([[2, 3, 4, 5, 6, 7, 8, 9]], dtype=np.int32),
        np.asarray([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=np.bool_),
    )
    suffix_b = TokenBatch(
        suffix_a.input_ids,
        suffix_a.attention_mask,
        np.asarray([[2, 3, 4, 5, 9, 8, 7, 6]], dtype=np.int32),
        suffix_a.loss_mask,
    )
    entry_a = _entry("case-a", 1)
    entry_b = _entry("case-b", 2)
    cases = (
        MidpointCase("mouse", entry_a, 5, prefix, suffix_a),
        MidpointCase("mouse", entry_b, 5, prefix, suffix_b),
    )
    results = evaluate_case_batch(
        cases,
        contract_sha256=_digest("contract"),
        evaluation_id="fixture",
        dataset="final",
        method="independent_noun_exhaustive",
        order=None,
        stage=192,
        routing="exhaustive",
        base_params=params,
        model_config=config,
        bank=bank,
        evaluation_batch_size=2,
    )
    assert results[0].prefix_scores == results[1].prefix_scores
    assert results[0].selected_index == results[1].selected_index == 0
    assert results[0].suffix_mean_nll != results[1].suffix_mean_nll
    ledger = ChainedJsonlLedger(tmp_path / "evaluation.jsonl", EVALUATION_ROW_FORMAT)
    evaluate_to_ledger(
        cases,
        contract_sha256=_digest("contract"),
        evaluation_id="fixture",
        dataset="final",
        method="independent_noun_exhaustive",
        order=None,
        stage=192,
        routing="exhaustive",
        base_params=params,
        model_config=config,
        bank=bank,
        ledger=ledger,
        router_batch_size=2,
        evaluation_batch_size=2,
    )
    first_payload = ledger.path.read_bytes()
    evaluate_to_ledger(
        cases,
        contract_sha256=_digest("contract"),
        evaluation_id="fixture",
        dataset="final",
        method="independent_noun_exhaustive",
        order=None,
        stage=192,
        routing="exhaustive",
        base_params=params,
        model_config=config,
        bank=bank,
        ledger=ledger,
        router_batch_size=2,
        evaluation_batch_size=2,
    )
    assert ledger.path.read_bytes() == first_payload
    assert len(ledger.rows) == 2


def test_loopback_dashboard_exposes_etags_events_and_get_only_artifacts(
    tmp_path: Path,
) -> None:
    work = tmp_path / "work"
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "result.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    jobs = (
        StudyJob("short", "preflight", "short job", 2, "checks", 10.0),
        StudyJob("long", "training", "long job", 10, "updates", 301.0),
    )
    recorder = ProgressRecorder(work, _digest("contract"), jobs)
    recorder.update("short", 2, status="complete", elapsed_seconds=1.0)
    recorder.update(
        "long",
        3,
        elapsed_seconds=30.0,
        metrics={"story_nll": 1.5},
        detail={"order": "blocked", "arrival": 3, "active_chunk_count": 2},
    )
    try:
        server = start_dashboard_server(work, artifacts, first_port=0, last_port=0)
    except RuntimeError:
        pytest.skip("the filesystem sandbox denies loopback socket binding")
    try:
        with urllib.request.urlopen(server.url + "api/v1/snapshot") as response:
            snapshot = json.loads(response.read())
            etag = response.headers["ETag"]
        assert snapshot["event_count"] == 2
        assert snapshot["jobs"][1]["is_long"] is True
        request = urllib.request.Request(
            server.url + "api/v1/snapshot",
            headers={"If-None-Match": etag},
        )
        with pytest.raises(urllib.error.HTTPError) as unchanged:
            urllib.request.urlopen(request)
        assert unchanged.value.code == 304
        with urllib.request.urlopen(server.url + "api/v1/events?after=0") as response:
            assert len(json.loads(response.read())["events"]) == 1
        with urllib.request.urlopen(server.url + "artifacts/result.csv") as response:
            assert response.read() == b"a,b\n1,2\n"
        post = urllib.request.Request(server.url, method="POST", data=b"x")
        with pytest.raises(urllib.error.HTTPError) as rejected:
            urllib.request.urlopen(post)
        assert rejected.value.code == 405
        with pytest.raises(urllib.error.HTTPError) as traversal:
            urllib.request.urlopen(server.url + "artifacts/../secret.json")
        assert traversal.value.code in (403, 404)
    finally:
        server.stop()


def test_dashboard_resume_ignores_only_stale_replayed_progress(tmp_path: Path) -> None:
    recorder = ProgressRecorder(
        tmp_path / "work",
        _digest("contract"),
        (StudyJob("training", "training", "train", 10, "updates", 301.0),),
    )
    complete = recorder.update(
        "training",
        10,
        status="complete",
        elapsed_seconds=20.0,
    )
    ledger_before = recorder.ledger.path.read_bytes()

    replayed = recorder.update(
        "training",
        3,
        elapsed_seconds=3.0,
        ignore_stale_replay=True,
    )
    replayed_at_boundary = recorder.update(
        "training",
        10,
        elapsed_seconds=10.0,
        ignore_stale_replay=True,
    )

    assert replayed == complete
    assert replayed_at_boundary == complete
    assert recorder.ledger.path.read_bytes() == ledger_before
    assert recorder.snapshot()["jobs"][0]["status"] == "complete"
    with pytest.raises(ValueError, match="cannot move backward"):
        recorder.update("training", 9, elapsed_seconds=21.0)

    running_recorder = ProgressRecorder(
        tmp_path / "running-work",
        _digest("running-contract"),
        (StudyJob("evaluation", "evaluation", "evaluate", 10, "rows", 301.0),),
    )
    running = running_recorder.update(
        "evaluation",
        4,
        elapsed_seconds=2.0,
        metrics={"story_nll": 1.5},
    )
    running_bytes = running_recorder.ledger.path.read_bytes()
    assert running_recorder.update(
        "evaluation",
        4,
        elapsed_seconds=3.0,
        metrics={},
        ignore_stale_replay=True,
    ) == running
    assert running_recorder.ledger.path.read_bytes() == running_bytes
    assert running_recorder.snapshot()["jobs"][0]["metrics"] == {"story_nll": 1.5}

    frozen = publish_frozen_dashboard(tmp_path / "dashboard.html", recorder.snapshot())
    html = frozen.read_text(encoding="utf-8")
    assert "Content-Security-Policy" not in html
    assert "http://" not in html and "https://" not in html
    assert "log-t temporal consolidation" in html


def test_merge_distortion_rows_telescope_for_every_leaf_and_kind(tmp_path: Path) -> None:
    shards = _shards()
    final_state, merges = simulate_hierarchy(shards, "blocked")
    ordering = OrderingArtifacts("blocked", (), (), final_state, merges)
    task_by_shard = {shard.shard_id: shard.task_id for shard in shards}
    current = {
        (shard.shard_id, kind): 1.0
        for shard in shards
        for kind in ("source", "validation")
    }
    ledger = ChainedJsonlLedger(tmp_path / "distortion.jsonl", MERGE_ROW_FORMAT)
    for merge in merges:
        for child in (merge.left, merge.right):
            for shard_id in child.shard_ids:
                for kind in ("source", "validation"):
                    child_mean = current[(shard_id, kind)]
                    increment = 0.001 * (merge.parent.level + (kind == "validation"))
                    parent_mean = child_mean + increment
                    ledger.append(
                        {
                            "child_mean_nll": child_mean,
                            "child_chunk_id": child.chunk_id,
                            "contract_sha256": _digest("contract"),
                            "kind": kind,
                            "order": "blocked",
                            "parent_chunk_id": merge.parent.chunk_id,
                            "parent_mean_nll": parent_mean,
                            "positive_increment": increment,
                            "signed_increment": increment,
                            "source_shard_id": shard_id,
                            "task_id": task_by_shard[shard_id],
                            "token_count": 10,
                        }
                    )
                    current[(shard_id, kind)] = parent_mean
    assert len(ledger.rows) == 2_052
    assert validate_distortion_rows(
        ledger.rows,
        _digest("contract"),
        "blocked",
    ) == expected_distortion_keys(ordering)
    audits = require_lineage_identities(ledger.rows, final_state, tolerance=1e-12)
    assert len(audits) == ARRIVAL_COUNT * 2
    assert max(abs(audit.telescoping_residual) for audit in audits) < 1e-15
    assert min(audit.positive_bound_slack for audit in audits) >= -1e-15
    altered = [dict(row) for row in ledger.rows]
    altered[0]["parent_mean_nll"] = float(altered[0]["parent_mean_nll"]) + 0.1
    with pytest.raises(ValueError, match="row changed"):
        validate_distortion_rows(altered, _digest("contract"), "blocked")


def test_lineage_graph_sources_cover_all_leaves_merges_and_final_chunks() -> None:
    final_state, merges = simulate_hierarchy(_shards(), "round_robin")
    ordering = OrderingArtifacts("round_robin", (), (), final_state, merges)
    compact = _compact_lineage_dot(ordering)
    full = _full_lineage_dot(ordering)
    assert compact.count(" -> deployment") == 9
    assert full.count("[label=") == ARRIVAL_COUNT + len(merges)
    assert full.count(" -> ") == 2 * len(merges)
    assert all(chunk.chunk_id in compact for chunk in final_state.active_chunks)
    accessible = _accessible_svg(
        "<svg><title>old</title><g/></svg>",
        "Lineage title",
        "Complete lineage description",
    )
    assert 'role="img"' in accessible
    assert 'aria-labelledby="svg-title svg-description"' in accessible
    assert "Complete lineage description" in accessible


def test_timing_ledger_validation_rejects_changed_repetition_mean(tmp_path: Path) -> None:
    ledger = ChainedJsonlLedger(tmp_path / "timing.jsonl", TIMING_ROW_FORMAT)
    repetitions = [0.1, 0.2, 0.3, 0.4, 0.5]
    ledger.append(
        {
            "candidate_count": 3,
            "cold_seconds": 1.0,
            "contract_sha256": _digest("contract"),
            "kind": "prefix",
            "prefix_width": 32,
            "warm_mean_seconds": sum(repetitions) / len(repetitions),
            "warm_repetitions": repetitions,
        }
    )
    assert validate_timing_rows(ledger.rows, _digest("contract")) == (
        ("prefix", 3, 32),
    )
    altered = [dict(ledger.rows[0])]
    altered[0]["warm_mean_seconds"] = 0.7
    with pytest.raises(ValueError, match="timing ledger row changed"):
        validate_timing_rows(altered, _digest("contract"))


def test_timing_audit_isolates_each_cold_shape_compile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    fake_batch = SimpleNamespace(
        input_ids=np.zeros((8, 32), dtype=np.int32),
        loss_mask=np.ones((8, 32), dtype=np.bool_),
    )
    fake_case = SimpleNamespace(
        prefix_width_bucket=32,
        suffix_windows=fake_batch,
    )
    inputs = SimpleNamespace(
        work_directory=tmp_path,
        contract_sha256=_digest("contract"),
        loaded_base=SimpleNamespace(params=object(), config=object()),
    )
    monkeypatch.setattr(
        timing_module,
        "expected_timing_shapes",
        lambda _: (("prefix", 1, 32), ("suffix", 1, None)),
    )
    monkeypatch.setattr(
        timing_module,
        "_representative_banks",
        lambda *_: {1: SimpleNamespace(candidate_ids=("base",))},
    )
    monkeypatch.setattr(
        timing_module,
        "prepare_prefix_kernel_batch",
        lambda _, **_keywords: fake_batch,
    )
    monkeypatch.setattr(
        timing_module,
        "prepare_suffix_kernel_batch",
        lambda _: fake_batch,
    )
    monkeypatch.setattr(
        timing_module,
        "run_prefix_kernel",
        lambda *_: events.append("prefix"),
    )
    monkeypatch.setattr(
        timing_module,
        "run_suffix_kernel",
        lambda *_args, **_keywords: events.append("suffix"),
    )
    monkeypatch.setattr(
        timing_module,
        "_synchronized_seconds",
        lambda operation: (operation(), 0.01)[1],
    )
    monkeypatch.setattr(
        timing_module.jax,
        "clear_caches",
        lambda: events.append("clear"),
    )

    timing_module.run_or_resume_timing_audit(
        inputs,
        (),
        {"fixture": (fake_case,)},
    )
    assert events == [
        "clear",
        *("prefix",) * 6,
        "clear",
        *("suffix",) * 6,
    ]


def test_prefix_router_capacity_uses_bounded_power_of_two_microbatches() -> None:
    assert prefix_router_row_capacity(7, 320) == 8
    assert prefix_router_row_capacity(7, 352) == 4
    assert prefix_router_row_capacity(7, 384) == 4
    assert prefix_router_row_capacity(7, 416) == 4
    assert prefix_router_row_capacity(7, 512) == 4
    assert prefix_router_row_capacity(8, 288) == 8
    assert prefix_router_row_capacity(8, 320) == 4
    assert prefix_router_row_capacity(24, 512) == 1
    assert prefix_router_row_capacity(24, 128) == 4
    with pytest.raises(ValueError, match="capacity inputs"):
        prefix_router_row_capacity(7, 0)


def test_isolated_timing_workers_resume_at_exact_shape_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _digest("contract")
    shapes = (("prefix", 1, 32), ("suffix", 1, None))
    inputs = SimpleNamespace(
        contract_sha256=contract,
        repository_root=tmp_path,
        work_directory=tmp_path,
    )
    invocations: list[int] = []
    updates: list[tuple[int, int]] = []
    monkeypatch.setattr(timing_module, "expected_timing_shapes", lambda _: shapes)
    monkeypatch.setattr(
        timing_module,
        "_prepare_timing_worker_bundle",
        lambda *_: tmp_path / "bundle",
    )

    def invoke(_inputs, _bundle, ledger_path, shape_index):
        invocations.append(shape_index)
        kind, candidate_count, prefix_width = shapes[shape_index]
        repetitions = [0.01] * 5
        ChainedJsonlLedger(ledger_path, TIMING_ROW_FORMAT).append(
            {
                "allocator_peak_bytes": 1024,
                "candidate_count": candidate_count,
                "cold_seconds": 0.02,
                "contract_sha256": contract,
                "kind": kind,
                "prefix_width": prefix_width,
                "warm_mean_seconds": 0.01,
                "warm_repetitions": repetitions,
            }
        )

    monkeypatch.setattr(timing_module, "_invoke_timing_worker", invoke)
    timing_module.run_or_resume_isolated_timing_audit(
        inputs,
        (),
        {},
        progress=lambda completed, total, _metrics: updates.append((completed, total)),
    )
    first_payload = (tmp_path / "timing.jsonl").read_bytes()
    timing_module.run_or_resume_isolated_timing_audit(inputs, (), {})

    assert invocations == [0, 1]
    assert updates == [(1, 2), (2, 2)]
    assert (tmp_path / "timing.jsonl").read_bytes() == first_payload
    assert timing_module.maximum_timing_allocator_peak_bytes(
        tmp_path / "timing.jsonl",
        contract,
    ) == 1024


def test_evaluation_rejects_cross_contract_and_reordered_resume(tmp_path: Path) -> None:
    config = tiny_shakespeare_unit_model_config(vocab_size=32)
    params = init_gpt_neo_params(jax.random.PRNGKey(21), config, dtype=jnp.float32)
    bank = build_adapter_bank((), config)
    prefix = RouterBatch(
        np.asarray([[1, 2, 3, 4]], dtype=np.int32),
        np.ones((1, 4), dtype=np.bool_),
        np.asarray([[2, 3, 4, 5]], dtype=np.int32),
        np.ones((1, 4), dtype=np.bool_),
    )
    suffix = TokenBatch(
        np.asarray([[1, 2, 3, 4]], dtype=np.int32),
        np.ones((1, 4), dtype=np.bool_),
        np.asarray([[2, 3, 4, 5]], dtype=np.int32),
        np.ones((1, 4), dtype=np.bool_),
    )
    cases = tuple(
        MidpointCase("mouse", _entry(f"ordered-{index}", index), 3, prefix, suffix)
        for index in range(2)
    )
    source = ChainedJsonlLedger(tmp_path / "source.jsonl", EVALUATION_ROW_FORMAT)
    evaluate_to_ledger(
        cases,
        contract_sha256=_digest("contract"),
        evaluation_id="ordered",
        dataset="final",
        method="base",
        order=None,
        stage=192,
        routing="forced_base",
        base_params=params,
        model_config=config,
        bank=bank,
        ledger=source,
        router_batch_size=2,
        evaluation_batch_size=2,
    )
    values = tuple(
        {
            key: value
            for key, value in row.items()
            if key not in {"format", "previous_sha256", "result_sha256", "sequence"}
        }
        for row in source.rows
    )
    reordered = ChainedJsonlLedger(tmp_path / "reordered.jsonl", EVALUATION_ROW_FORMAT)
    reordered.append_many(reversed(values))
    with pytest.raises(ValueError, match="canonical case prefix"):
        evaluate_to_ledger(
            cases,
            contract_sha256=_digest("contract"),
            evaluation_id="ordered",
            dataset="final",
            method="base",
            order=None,
            stage=192,
            routing="forced_base",
            base_params=params,
            model_config=config,
            bank=bank,
            ledger=reordered,
            router_batch_size=2,
            evaluation_batch_size=2,
        )
    cross_contract = ChainedJsonlLedger(
        tmp_path / "cross-contract.jsonl",
        EVALUATION_ROW_FORMAT,
    )
    cross_values = dict(values[0])
    cross_values["contract_sha256"] = _digest("different-contract")
    cross_contract.append(cross_values)
    with pytest.raises(ValueError, match="ledger row changed"):
        evaluate_to_ledger(
            cases,
            contract_sha256=_digest("contract"),
            evaluation_id="ordered",
            dataset="final",
            method="base",
            order=None,
            stage=192,
            routing="forced_base",
            base_params=params,
            model_config=config,
            bank=bank,
            ledger=cross_contract,
            router_batch_size=2,
            evaluation_batch_size=2,
        )


def test_full_model_resume_matches_uninterrupted_parameter_identity(tmp_path: Path) -> None:
    config = tiny_shakespeare_unit_model_config(vocab_size=32)
    params = init_gpt_neo_params(jax.random.PRNGKey(31), config, dtype=jnp.float32)
    digest = _digest("checkpoint-file")
    reference = save_gpt_neo_checkpoint(
        tmp_path / "base-checkpoint",
        params,
        config,
        tokenizer=TokenizerCheckpointMetadata(
            "character",
            "fixture-tokenizer",
            _digest("tokenizer-revision"),
            (CheckpointFileHash("tokenizer.json", digest),),
        ),
        source=SourceCheckpointMetadata(
            "fixture-source",
            _digest("source-revision"),
            _digest("source-bytes"),
        ),
    )
    selected = SimpleNamespace(reference=reference)
    values = np.tile(np.arange(8, dtype=np.int32), (32, 1)) % 31
    batch = TokenBatch(
        values,
        np.ones_like(values, dtype=np.bool_),
        (values + 1) % 31,
        np.ones_like(values, dtype=np.bool_),
    )
    batches = (batch, batch, batch)
    job = TrainingJob(
        _digest("contract"),
        "fixture-full-model",
        "joint_iid_full_model",
        (_digest("story-a"), _digest("story-b")),
        (_digest("shard"),),
    )
    with pytest.raises(TrainingInterrupted):
        train_or_load_full_model(
            job,
            batches,
            selected,
            tmp_path / "resumed-output",
            tmp_path / "resumed-work",
            stop_after_update=2,
        )
    resumed = train_or_load_full_model(
        job,
        batches,
        selected,
        tmp_path / "resumed-output",
        tmp_path / "resumed-work",
    )
    direct = train_or_load_full_model(
        job,
        batches,
        selected,
        tmp_path / "direct-output",
        tmp_path / "direct-work",
    )
    assert resumed.checkpoint.parameter_checksum == direct.checkpoint.parameter_checksum
    assert resumed.loss_trace_sha256 == direct.loss_trace_sha256


def test_dashboard_contains_phase_nested_memory_and_live_estimate_surfaces(
    tmp_path: Path,
) -> None:
    recorder = ProgressRecorder(
        tmp_path / "work",
        _digest("contract"),
        (StudyJob("stack-blocked", "blocked", "stack", 192, "arrivals", 600.0),),
    )
    recorder.update(
        "stack-blocked",
        1,
        elapsed_seconds=2.0,
        metrics={"story_nll": 1.5},
        detail={
            "active_adapter_bytes": 100,
            "active_chunk_count": 1,
            "active_intervals": ["1-1 (L0)"],
            "archive_adapter_bytes": 100,
            "archive_chunk_count": 1,
            "arrival": 1,
            "carry": [],
            "order": "blocked",
        },
    )
    html = publish_frozen_dashboard(tmp_path / "dashboard.html", recorder.snapshot()).read_text()
    assert "phaseFraction" in html
    assert "active_adapter_bytes" in html
    assert "provisional story NLL" in html
    assert "is_long" in html
    assert ".join('\\n')" in html
    assert "paused/stale" in html
    assert "snapshot.event_count||0)-501" in html


def test_report_plots_graphviz_and_standalone_surfaces_regenerate_identically(
    tmp_path: Path,
) -> None:
    final_state, merges = simulate_hierarchy(_shards(), "blocked")
    blocked = OrderingArtifacts("blocked", (), (), final_state, merges)
    round_state, round_merges = simulate_hierarchy(_shards(), "round_robin")
    round_robin = OrderingArtifacts(
        "round_robin",
        (),
        (),
        round_state,
        round_merges,
    )
    metric = {
        "address_oracle_agreement": 0.5,
        "candidate_evaluations": 10,
        "exact_noun_route_accuracy": None,
        "mean_oracle_regret": 0.01,
        "mean_prefix_entropy": 0.5,
        "mean_prefix_margin": 0.2,
        "model_forward_equivalent_prefix_tokens": 100,
        "noun_support_accuracy": 0.7,
        "oracle_story_mean_nll": 1.4,
        "story_count": 10,
        "story_mean_nll": 1.5,
        "suffix_token_accuracy": 0.6,
        "token_count": 100,
        "token_mean_nll": 1.6,
    }
    stage = tuple(
        {
            **metric,
            "dataset": "macro",
            "method": method,
            "order": order,
            "stage": stage_index,
            "story_mean_nll": 1.5 + 0.001 * stage_index,
        }
        for order in ("blocked", "round_robin")
        for method in ("base", "sequential_lora", "log_t")
        for stage_index in (8, 192)
    )
    aggregate = tuple(
        {
            **metric,
            "dataset": "macro" if order != "offline" else "final",
            "method": method,
            "order": order,
            "stage": 192,
        }
        for order, method in (
            ("blocked", "base"),
            ("blocked", "sequential_lora"),
            ("blocked", "log_t"),
            ("round_robin", "base"),
            ("round_robin", "sequential_lora"),
            ("round_robin", "log_t"),
            ("offline", "independent_noun_exhaustive"),
            ("offline", "joint_iid_lora"),
            ("offline", "joint_iid_full_model"),
        )
    )
    cost = tuple(
        {
            "active_adapter_bytes": 100 * arrival,
            "active_adapter_count": min(arrival, 9),
            "arrival": arrival,
            "insertion_optimizer_updates": arrival * 2,
            "insertion_runtime_seconds": arrival / 10,
            "order": order,
        }
        for order in ("blocked", "round_robin")
        for arrival in (1, 64, 128, 192)
    )
    merge = tuple(
        {
            "delta": 0.01 * level,
            "kind": kind,
            "level": level,
            "order": order,
        }
        for order in ("blocked", "round_robin")
        for kind in ("source", "validation")
        for level in (1, 2, 3)
    )
    timing = tuple(
        {
            "candidate_count": count,
            "kind": kind,
            "prefix_width": 32 if kind == "prefix" else None,
            "warm_mean_seconds": 0.001 * count,
        }
        for kind in ("prefix", "suffix")
        for count in (1, 5, 9)
    )
    analysis = {
        "aggregate": aggregate,
        "bootstrap": tuple(
            {
                "estimate": 0.01,
                "lower_95": -0.01,
                "metric": name,
                "upper_95": 0.02,
            }
            for name in ("story_nll", "token_accuracy")
        ),
        "cost": cost,
        "distortion": {
            "lineage": (
                {"positive_bound_slack": 0.0, "telescoping_residual": 0.0},
            ),
            "merge": merge,
        },
        "provenance": {
            "base_parameter_checksum": _digest("base"),
            "final_vamp_tensor_checksum": _digest("vamp"),
        },
        "stage": stage,
        "timing": timing,
    }
    first_plots = _publish_plots(tmp_path, analysis)
    first_hashes = tuple(sha256(path.read_bytes()).hexdigest() for path in first_plots)
    second_plots = _publish_plots(tmp_path, analysis)
    assert tuple(sha256(path.read_bytes()).hexdigest() for path in second_plots) == first_hashes
    graph_paths = (
        *_publish_lineage_graphs(tmp_path, blocked),
        *_publish_lineage_graphs(tmp_path, round_robin),
    )
    assert all(path.is_file() for path in graph_paths)
    svg_map = {
        path.name: path.read_text(encoding="utf-8")
        for path in (*second_plots, *graph_paths)
        if path.suffix == ".svg"
    }
    inputs = SimpleNamespace(
        contract_sha256=_digest("contract"),
        final_vamp_tensor_checksum=_digest("vamp"),
        partition=SimpleNamespace(partition_sha256=_digest("partition")),
        selected_base=SimpleNamespace(
            reference=SimpleNamespace(parameter_checksum=_digest("base"))
        ),
    )
    markdown = _markdown_report(
        inputs,
        analysis,
        {"end_to_end_seconds": 3600.0},
        {"peak_bytes_in_use": 1024**3},
    )
    html = _html_report(
        inputs,
        analysis,
        {"end_to_end_seconds": 3600.0},
        {"peak_bytes_in_use": 1024**3},
        svg_map,
    )
    assert "<details>" in markdown
    assert html.startswith("<!doctype html>")
    assert html.count('role="img"') >= 9
    assert "<script src=" not in html and "<link " not in html
    assert "lineage-blocked-full" not in html
    assert "Complete lineage audits" in html

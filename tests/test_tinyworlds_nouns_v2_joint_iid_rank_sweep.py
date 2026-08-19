from __future__ import annotations

import ast
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from apm.continual.language_tasks import RouterBatch
from apm.data.text.tinyworlds_nouns_v1.experiment import StoryIndexEntry
from apm.data.text.tinyworlds_nouns_v2.contracts import TASK_IDS, canonical_json_bytes
from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_contracts import (
    EVALUATION_ROW_FORMAT,
)
from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_evaluation import (
    AdapterCandidate,
    MidpointCase,
    build_adapter_bank,
    evaluate_to_ledger,
)
from apm.continual.artifacts import (
    ChainedJsonlLedger,
    file_sha256,
)
from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_training import (
    TrainingInterrupted,
    TrainingJob,
    train_or_load_lora,
)
from apm.data.text.tinyworlds_nouns_v2.temporal_joint_iid_rank_sweep import (
    RANKS,
    _paired_bootstrap,
    rank_lora_config,
    rank_training_job,
)
from apm.data.text.tinyworlds_nouns_v2.temporal_joint_iid_rank_sweep_report import (
    publish_joint_iid_rank_sweep_report,
)
from apm.lm.lora import init_lora_edge
from apm.lm.parameters import init_gpt_neo_params
from apm.lm.text_data import TokenBatch
from apm.lm.workflow import tiny_shakespeare_unit_model_config


def _digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def test_default_training_job_identity_surface_is_unchanged() -> None:
    job = TrainingJob(
        _digest("contract"),
        "canonical-fixture",
        "joint_iid_lora",
        (_digest("story-a"), _digest("story-b")),
        (_digest("shard"),),
    )
    assert not {
        "batch_namespace_sha256",
        "lora_alpha",
        "lora_rank",
        "random_namespace_sha256",
    } & set(job.as_record())
    assert job.identity_sha256 == TrainingJob(
        _digest("contract"),
        "canonical-fixture",
        "joint_iid_lora",
        (_digest("story-a"), _digest("story-b")),
        (_digest("shard"),),
    ).identity_sha256


def test_rank_configs_hold_scale_constant_and_jobs_bind_namespaces() -> None:
    canonical = TrainingJob(
        _digest("parent-contract"),
        "joint-iid-lora",
        "joint_iid_lora",
        (_digest("story-a"), _digest("story-b")),
        (_digest("shard"),),
    )
    inputs = SimpleNamespace(
        canonical_rank8_job=canonical,
        contract_sha256=_digest("sweep-contract"),
        all_story_ids=canonical.source_story_ids,
        source_shard_ids=canonical.source_shard_ids,
    )
    assert tuple(rank_lora_config(rank).scale for rank in RANKS) == (1.0,) * 4
    assert rank_training_job(inputs, 8) is canonical
    job = rank_training_job(inputs, 16)
    assert (job.lora_rank, job.lora_alpha) == (16, 16.0)
    assert job.batch_namespace_sha256 == canonical.identity_sha256
    assert job.random_namespace_sha256 == canonical.identity_sha256
    assert job.as_record()["source_story_ids_sha256"] == canonical.as_record()[
        "source_story_ids_sha256"
    ]


def test_rank_four_training_resume_matches_uninterrupted(tmp_path: Path) -> None:
    config = tiny_shakespeare_unit_model_config(vocab_size=32)
    params = init_gpt_neo_params(jax.random.PRNGKey(71), config, dtype=jnp.float32)
    values = np.tile(np.arange(8, dtype=np.int32), (32, 1)) % 31
    batch = TokenBatch(
        values,
        np.ones_like(values, dtype=np.bool_),
        (values + 1) % 31,
        np.ones_like(values, dtype=np.bool_),
    )
    namespace = _digest("canonical-rank-eight-namespace")
    job = TrainingJob(
        _digest("rank-sweep-contract"),
        "rank-four-fixture",
        "joint_iid_lora",
        (_digest("story-a"), _digest("story-b")),
        (_digest("shard"),),
        lora_rank=4,
        lora_alpha=4.0,
        batch_namespace_sha256=namespace,
        random_namespace_sha256=namespace,
    )
    with pytest.raises(TrainingInterrupted):
        train_or_load_lora(
            job,
            (batch, batch, batch),
            params,
            config,
            tmp_path / "resumed-output",
            tmp_path / "resumed-work",
            lora_config=rank_lora_config(4),
            stop_after_update=2,
        )
    resumed = train_or_load_lora(
        job,
        (batch, batch, batch),
        params,
        config,
        tmp_path / "resumed-output",
        tmp_path / "resumed-work",
        lora_config=rank_lora_config(4),
    )
    direct = train_or_load_lora(
        job,
        (batch, batch, batch),
        params,
        config,
        tmp_path / "direct-output",
        tmp_path / "direct-work",
        lora_config=rank_lora_config(4),
    )
    assert resumed.adapter_sha256 == direct.adapter_sha256
    assert resumed.loss_trace_sha256 == direct.loss_trace_sha256
    with pytest.raises(ValueError, match="not bound"):
        train_or_load_lora(
            job,
            (batch,),
            params,
            config,
            tmp_path / "wrong-output",
            tmp_path / "wrong-work",
            lora_config=rank_lora_config(16),
        )


def test_arbitrary_rank_bank_scores_the_existing_suffix_schema(tmp_path: Path) -> None:
    config = tiny_shakespeare_unit_model_config(vocab_size=32)
    params = init_gpt_neo_params(jax.random.PRNGKey(72), config, dtype=jnp.float32)
    lora_config = rank_lora_config(4)
    adapter = init_lora_edge(jax.random.PRNGKey(73), config, lora_config)
    bank = build_adapter_bank(
        (
            AdapterCandidate(
                "joint-iid-lora-rank-4",
                _digest("rank-four-adapter"),
                adapter,
                tuple((task_id, 8) for task_id in TASK_IDS),
            ),
        ),
        config,
        lora_config,
    )
    prefix = RouterBatch(
        np.asarray([[1, 2, 3, 4]], dtype=np.int32),
        np.ones((1, 4), dtype=np.bool_),
        np.asarray([[2, 3, 4, 5]], dtype=np.int32),
        np.ones((1, 4), dtype=np.bool_),
    )
    suffix = TokenBatch(
        np.asarray([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int32),
        np.ones((1, 8), dtype=np.bool_),
        np.asarray([[2, 3, 4, 5, 6, 7, 8, 9]], dtype=np.int32),
        np.asarray([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=np.bool_),
    )
    entry = StoryIndexEntry(_digest("evaluation-story"), 0, 0, 1, 0, 9)
    ledger = ChainedJsonlLedger(tmp_path / "rank-four.jsonl", EVALUATION_ROW_FORMAT)
    evaluate_to_ledger(
        (MidpointCase(TASK_IDS[0], entry, 5, prefix, suffix),),
        contract_sha256=_digest("sweep-contract"),
        evaluation_id="joint-iid-lora-rank-sweep",
        dataset="final",
        method="joint_iid_lora_rank_4",
        order=None,
        stage=192,
        routing="forced_adapter",
        base_params=params,
        model_config=config,
        bank=bank,
        ledger=ledger,
        router_batch_size=1,
        evaluation_batch_size=1,
    )
    assert bank.lora_config == lora_config
    assert ledger.rows[0]["selected_candidate_id"] == "joint-iid-lora-rank-4"
    assert ledger.rows[0]["suffix_token_count"] == 4


def test_paired_bootstrap_is_deterministic_and_noun_stratified() -> None:
    identities = tuple((task_id, index) for index, task_id in enumerate(TASK_IDS))

    def rows(offset: float) -> tuple[dict[str, object], ...]:
        return tuple(
            {
                "story_id": _digest(f"{task_id}-{index}"),
                "suffix_mean_nll": 1.0 + offset + index / 1_000,
                "suffix_token_count": 10 + index,
                "suffix_total_nll": (1.0 + offset + index / 1_000) * (10 + index),
                "task_id": task_id,
            }
            for task_id, index in identities
        )

    by_condition = {
        "full_model": rows(0.0),
        "rank_4": rows(0.3),
        "rank_8": rows(0.2),
        "rank_16": rows(0.1),
        "rank_32": rows(0.05),
    }
    first = _paired_bootstrap(by_condition)
    second = _paired_bootstrap(by_condition)
    assert first == second
    rank32_story = next(
        row
        for row in first
        if row["condition"] == "rank_32"
        and row["reference"] == "rank_8"
        and row["metric"] == "story_mean_nll"
    )
    assert float(rank32_story["estimate"]) == pytest.approx(-0.15)
    assert float(rank32_story["lower_95"]) == pytest.approx(-0.15)
    assert float(rank32_story["upper_95"]) == pytest.approx(-0.15)


def test_report_is_standalone_accessible_and_byte_identical(tmp_path: Path) -> None:
    contract = _digest("report-contract")
    inputs = SimpleNamespace(result_directory=tmp_path, contract_sha256=contract)
    for name in ("contract.json", "execution.json", "allocator.json"):
        (tmp_path / name).write_bytes(canonical_json_bytes({"name": name}))
    aggregate = tuple(
        {
            "condition": "full_model" if rank is None else f"rank_{rank}",
            "label": "Joint-IID full model" if rank is None else f"Joint-IID LoRA rank {rank}",
            "rank": rank,
            "story_count": 4_440,
            "story_mean_nll": 1.40 if rank is None else 1.6 - rank / 1_000,
            "suffix_token_accuracy": 0.61,
            "task_id": None,
            "token_count": 476_035,
            "token_mean_nll": 1.45 if rank is None else 1.64 - rank / 1_000,
        }
        for rank in (None, *RANKS)
    )
    analysis = {
        "aggregate": aggregate,
        "allocator": {"peak_bytes_in_use": 2 * 1024**3},
        "bootstrap": (
            {
                "condition": "rank_32",
                "estimate": -0.01,
                "lower_95": -0.02,
                "metric": "story_mean_nll",
                "reference": "rank_8",
                "upper_95": -0.001,
            },
        ),
        "comparability": {
            "base_path_max_abs_story_nll_drift": 1e-7,
            "batch_namespace_sha256": _digest("namespace"),
        },
        "execution": {"end_to_end_seconds": 60.0},
        "ledger_provenance": tuple(
            {"path": f"rank-{rank}.jsonl", "rank": rank, "row_count": 4_440, "sha256": _digest(str(rank)), "source": "fixture"}
            for rank in RANKS
        ),
        "per_task": tuple(
            {**row, "task_id": TASK_IDS[0], "story_count": 1}
            for row in aggregate
        ),
        "provenance": {
            "canonical_full_model_job_sha256": _digest("full"),
            "canonical_rank8_job_sha256": _digest("rank8"),
            "parent_contract_sha256": _digest("parent"),
            "parent_manifest_sha256": _digest("manifest"),
        },
        "training": tuple(
            {
                "adapter_file_bytes": rank * 100,
                "adapter_parameter_count": rank * 1_000,
                "adapter_to_base_parameter_fraction": rank / 1_000,
                "allocator_peak_bytes": 1024,
                "alpha": float(rank),
                "final_training_loss": 1.0,
                "job_sha256": _digest(f"job-{rank}"),
                "optimizer_updates": 15_024,
                "rank": rank,
                "reused_canonical_artifact": rank == 8,
                "runtime_seconds": 100.0,
                "scale": 1.0,
                "tensor_file_sha256": _digest(f"tensor-{rank}"),
            }
            for rank in RANKS
        ),
    }
    publish_joint_iid_rank_sweep_report(inputs, analysis)
    first = tuple(
        (path.name, file_sha256(path))
        for path in sorted(tmp_path.iterdir())
        if path.is_file()
    )
    publish_joint_iid_rank_sweep_report(inputs, analysis)
    second = tuple(
        (path.name, file_sha256(path))
        for path in sorted(tmp_path.iterdir())
        if path.is_file()
    )
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    svg = (tmp_path / "rank-sweep-nll.svg").read_text(encoding="utf-8")
    assert first == second
    assert "<details>" in html and "<svg" in html
    assert "<script src=" not in html and '<link rel="stylesheet"' not in html
    assert 'role="img"' in svg and "<title id=" in svg and "<desc id=" in svg


def test_runner_has_no_options_and_fixes_gpu_zero() -> None:
    path = Path("scripts/run_tinyworlds_nouns_v2_joint_iid_rank_sweep.py")
    source = path.read_text(encoding="utf-8")
    module = ast.parse(source)
    imports = {
        alias.name
        for node in module.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    main_function = next(
        node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "main"
    )
    assert "argparse" not in imports
    assert not main_function.args.args
    assert 'os.environ["CUDA_VISIBLE_DEVICES"] = "0"' in source
    assert "Persistent temporary directory:" in source

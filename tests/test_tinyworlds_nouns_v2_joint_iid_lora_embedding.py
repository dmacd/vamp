from __future__ import annotations

import ast
from hashlib import sha256
from pathlib import Path
import runpy
import shutil
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from apm.data.text.tinyworlds_nouns_v2.contracts import (
    TASK_IDS,
    canonical_json_bytes,
)
from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_io import file_sha256
from apm.data.text.tinyworlds_nouns_v2.temporal_consolidation_training import (
    TrainingInterrupted,
)
from apm.data.text.tinyworlds_nouns_v2.temporal_joint_iid_lora_embedding import (
    LoraEmbeddingJob,
    RANKS,
    _paired_bootstrap,
    load_lora_embedding_artifact,
    train_or_load_lora_embedding,
)
from apm.data.text.tinyworlds_nouns_v2.temporal_joint_iid_lora_embedding_report import (
    publish_lora_embedding_report,
)
from apm.data.text.tinyworlds_nouns_v2.temporal_joint_iid_rank_sweep import (
    rank_lora_config,
)
from apm.lm.parameters import init_gpt_neo_params
from apm.lm.text_data import TokenBatch
from apm.lm.workflow import tiny_shakespeare_unit_model_config


def _digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _fixture():
    config = tiny_shakespeare_unit_model_config(vocab_size=32)
    params = init_gpt_neo_params(jax.random.PRNGKey(401), config, dtype=jnp.float32)
    values = np.tile(np.arange(8, dtype=np.int32), (32, 1)) % 31
    batch = TokenBatch(
        values,
        np.ones_like(values, dtype=np.bool_),
        (values + 1) % 31,
        np.ones_like(values, dtype=np.bool_),
    )
    namespace = _digest("canonical-rank-eight-namespace")
    job = LoraEmbeddingJob(
        _digest("embedding-lora-contract"),
        8,
        (_digest("story-a"), _digest("story-b")),
        (_digest("shard"),),
        namespace,
        namespace,
    )
    return config, params, batch, job


def test_joint_training_updates_tied_embedding_without_mutating_base(tmp_path: Path) -> None:
    config, params, batch, job = _fixture()
    before = tuple(np.asarray(value).copy() for value in jax.tree_util.tree_leaves(params))
    artifact = train_or_load_lora_embedding(
        job,
        (batch,),
        params,
        config,
        tmp_path / "output",
        tmp_path / "work",
    )
    after = tuple(np.asarray(value) for value in jax.tree_util.tree_leaves(params))
    assert all(np.array_equal(left, right) for left, right in zip(before, after, strict=True))
    assert not np.array_equal(
        np.asarray(artifact.trainable.token_embedding),
        np.asarray(params.token_embedding),
    )
    assert any(
        np.any(np.asarray(value) != 0.0)
        for block in artifact.trainable.adapter.blocks
        for projection in block
        for value in (projection.right,)
    )


def test_embedding_lora_resume_matches_uninterrupted(tmp_path: Path) -> None:
    config, params, batch, job = _fixture()
    with pytest.raises(TrainingInterrupted):
        train_or_load_lora_embedding(
            job,
            (batch, batch, batch),
            params,
            config,
            tmp_path / "resumed-output",
            tmp_path / "resumed-work",
            stop_after_update=2,
        )
    resumed = train_or_load_lora_embedding(
        job,
        (batch, batch, batch),
        params,
        config,
        tmp_path / "resumed-output",
        tmp_path / "resumed-work",
    )
    direct = train_or_load_lora_embedding(
        job,
        (batch, batch, batch),
        params,
        config,
        tmp_path / "direct-output",
        tmp_path / "direct-work",
    )
    assert resumed.trainable_sha256 == direct.trainable_sha256
    assert resumed.loss_trace_sha256 == direct.loss_trace_sha256
    states = tuple((tmp_path / "resumed-work" / job.identity_sha256 / "states").iterdir())
    assert len(states) == 1 and states[0].name == "state-00000003"


def test_artifact_loader_rejects_tensor_tampering(tmp_path: Path) -> None:
    config, params, batch, job = _fixture()
    artifact = train_or_load_lora_embedding(
        job,
        (batch,),
        params,
        config,
        tmp_path / "output",
        tmp_path / "work",
    )
    copied = tmp_path / "copied"
    shutil.copytree(artifact.directory, copied)
    tensor_path = copied / "trainable.safetensors"
    payload = bytearray(tensor_path.read_bytes())
    payload[-1] ^= 1
    tensor_path.write_bytes(payload)
    with pytest.raises(ValueError, match="file hash"):
        load_lora_embedding_artifact(
            copied,
            job,
            config,
            rank_lora_config(8),
        )


def test_embedding_bootstrap_is_deterministic_and_paired() -> None:
    identities = tuple((task_id, index) for index, task_id in enumerate(TASK_IDS))

    def rows(offset: float) -> tuple[dict[str, object], ...]:
        return tuple(
            {
                "story_id": _digest(f"{task_id}-{index}"),
                "suffix_mean_nll": 1.0 + offset + index / 1000,
                "suffix_token_count": 10 + index,
                "suffix_total_nll": (1.0 + offset + index / 1000) * (10 + index),
                "task_id": task_id,
            }
            for task_id, index in identities
        )

    conditions = {
        "full_model": rows(0.0),
        "lora_rank_8": rows(0.4),
        "lora_rank_32": rows(0.5),
        "lora_embedding_rank_8": rows(0.2),
        "lora_embedding_rank_32": rows(0.1),
    }
    first = _paired_bootstrap(conditions)
    assert first == _paired_bootstrap(conditions)
    comparison = next(
        row
        for row in first
        if row["condition"] == "lora_embedding_rank_8"
        and row["reference"] == "lora_rank_8"
        and row["metric"] == "story_mean_nll"
    )
    assert float(comparison["estimate"]) == pytest.approx(-0.2)
    assert float(comparison["lower_95"]) == pytest.approx(-0.2)
    assert float(comparison["upper_95"]) == pytest.approx(-0.2)


def test_report_is_standalone_accessible_and_byte_identical(tmp_path: Path) -> None:
    contract = _digest("report-contract")
    inputs = SimpleNamespace(result_directory=tmp_path, contract_sha256=contract)
    for name in ("contract.json", "execution.json", "allocator.json"):
        (tmp_path / name).write_bytes(canonical_json_bytes({"name": name}))
    condition_specs = (
        ("full_model", "Joint-IID full model", None, 1.40),
        ("lora_rank_8", "Projection LoRA rank 8", 8, 1.55),
        ("lora_rank_32", "Projection LoRA rank 32", 32, 1.57),
        (
            "lora_embedding_rank_8",
            "Projection LoRA + tied embedding rank 8",
            8,
            1.44,
        ),
        (
            "lora_embedding_rank_32",
            "Projection LoRA + tied embedding rank 32",
            32,
            1.43,
        ),
    )
    aggregate = tuple(
        {
            "condition": condition,
            "label": label,
            "rank": rank,
            "story_count": 4_440,
            "story_mean_nll": nll,
            "suffix_token_accuracy": 0.62,
            "task_id": None,
            "token_count": 476_035,
            "token_mean_nll": nll + 0.03,
        }
        for condition, label, rank, nll in condition_specs
    )
    analysis = {
        "aggregate": aggregate,
        "allocator": {"peak_bytes_in_use": 2 * 1024**3},
        "bootstrap": (
            {
                "condition": "lora_embedding_rank_8",
                "estimate": -0.11,
                "lower_95": -0.12,
                "metric": "story_mean_nll",
                "reference": "lora_rank_8",
                "upper_95": -0.10,
            },
        ),
        "comparability": {
            "batch_namespace_sha256": _digest("namespace"),
            "exact_suffix_target_count": 476_035,
        },
        "embedding_only": tuple(
            {
                "condition": f"trained_embedding_without_lora_rank_{rank}",
                "label": "diagnostic",
                "rank": rank,
                "story_count": 4_440,
                "story_mean_nll": 1.48,
                "token_count": 476_035,
                "token_mean_nll": 1.51,
            }
            for rank in RANKS
        ),
        "execution": {"end_to_end_seconds": 60.0},
        "ledger_provenance": tuple(
            {
                "path": f"rank-{rank}.jsonl",
                "rank": rank,
                "row_count": 4_440,
                "sha256": _digest(str(rank)),
            }
            for rank in RANKS
        ),
        "per_task": tuple(
            {**row, "task_id": TASK_IDS[0], "story_count": 1}
            for row in aggregate
        ),
        "provenance": {
            "parent_rank_sweep_contract_sha256": _digest("parent"),
            "parent_rank_sweep_manifest_sha256": _digest("manifest"),
        },
        "training": tuple(
            {
                "adapter_parameter_count": rank * 36_864,
                "allocator_peak_bytes": 1024,
                "alpha": float(rank),
                "embedding_delta_max_abs": 0.1,
                "embedding_delta_mean_abs": 0.01,
                "embedding_delta_relative_frobenius": 0.2,
                "embedding_delta_rms": 0.03,
                "embedding_learning_rate": 5e-5,
                "embedding_parameter_count": 12_865_792,
                "final_training_loss": 1.2,
                "job_sha256": _digest(f"job-{rank}"),
                "lora_learning_rate": 1e-3,
                "optimizer_updates": 15_024,
                "rank": rank,
                "runtime_seconds": 100.0,
                "tensor_file_bytes": 100,
                "tensor_file_sha256": _digest(f"tensor-{rank}"),
                "total_trainable_parameter_count": 12_865_792 + rank * 36_864,
                "trainable_to_base_parameter_fraction": 0.67,
            }
            for rank in RANKS
        ),
    }
    publish_lora_embedding_report(inputs, analysis)
    first = tuple(
        (path.name, file_sha256(path))
        for path in sorted(tmp_path.iterdir())
        if path.is_file()
    )
    publish_lora_embedding_report(inputs, analysis)
    second = tuple(
        (path.name, file_sha256(path))
        for path in sorted(tmp_path.iterdir())
        if path.is_file()
    )
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    svg = (tmp_path / "embedding-lora-nll.svg").read_text(encoding="utf-8")
    assert first == second
    assert "<details>" in html and "<svg" in html
    assert "<script src=" not in html and '<link rel="stylesheet"' not in html
    assert 'role="img"' in svg and "<title id=" in svg and "<desc id=" in svg


def test_runner_has_no_options_and_fixes_gpu_zero() -> None:
    path = Path("scripts/run_tinyworlds_nouns_v2_joint_iid_lora_embedding.py")
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
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "main"
    )
    assert "argparse" not in imports
    assert not main_function.args.args
    assert 'os.environ["CUDA_VISIBLE_DEVICES"] = "0"' in source
    assert "Persistent temporary directory:" in source


def test_runner_strips_allocator_authentication_envelope() -> None:
    namespace = runpy.run_path(
        "scripts/run_tinyworlds_nouns_v2_joint_iid_lora_embedding.py"
    )
    normalize = namespace["_allocator_analysis_payload"]
    raw = {
        "allocator_limit_bytes": 12 * 1024**3,
        "device_kind": ["GPU"],
        "device_platform": "gpu",
        "peak_bytes_in_use": 8 * 1024**3,
    }
    authenticated = {
        **raw,
        "contract_sha256": _digest("contract"),
        "format": "allocator-v1",
        "result_sha256": _digest("result"),
    }
    assert normalize(raw) == raw
    assert normalize(authenticated) == raw

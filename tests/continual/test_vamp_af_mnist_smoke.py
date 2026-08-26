from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

from apm.continual.addressing_first import StoredExampleTable
from apm.continual.top_two_adapter import top_two_base_state
from apm.experiments.vamp_af_config import (
    BaseTrainingConfig,
    DataConfig,
    AdapterTrainingConfig,
    PassConfig,
    PreflightConfig,
    RuntimeConfig,
    StructureConfig,
    VampAFConfig,
    load_config,
)
from apm.experiments.vamp_af_data import FeatureTables, pass_training_table
from apm.experiments.vamp_af_reporting import REQUIRED_ARTIFACTS
from apm.experiments.vamp_af_workflow import run_pass_seed


def _config(tmp_path: Path) -> VampAFConfig:
    conditions = ("frozen_base", "global_replay", "af", "oracle_context", "joint_iid")
    return VampAFConfig(
        "vamp-af-test",
        "top-two-v3",
        tmp_path,
        tmp_path,
        BaseTrainingConfig(0, 8, 0.001, 0.0001, 2, 1, 0.0001, 10),
        DataConfig((0.0, 18.0, 36.0, 54.0, 72.0), (0, 2, 4, 6, 8), 4, 2, "bilinear"),
        AdapterTrainingConfig(0.001, 0.0001, 0.9, 0.999, 1e-8, 2),
        PreflightConfig(1, 4, 0.0, 0.0, 0.01),
        StructureConfig(4, 16, 1, 1),
        (
            PassConfig("smoke", (0,), 2, 4, conditions, None, False),
            PassConfig("main", (0, 1, 2), 4, 4, conditions, None, False),
            PassConfig("consolidation_stress", (0,), 4, 4, ("af",), 3, True),
        ),
        RuntimeConfig("cpu", 2, True, False),
    )


def _tables() -> FeatureTables:
    rows = 20
    embeddings = torch.stack(
        (
            torch.linspace(-1.0, 1.0, rows),
            torch.sin(torch.arange(rows, dtype=torch.float32)),
            torch.cos(torch.arange(rows, dtype=torch.float32)),
            torch.arange(rows, dtype=torch.float32) / rows,
        ),
        dim=1,
    )
    embeddings = torch.nn.functional.normalize(embeddings, dim=1)
    contexts = torch.arange(5, dtype=torch.int64).repeat_interleave(4)
    labels = (torch.arange(rows) + contexts) % 2
    table = StoredExampleTable(
        embeddings,
        embeddings.clone(),
        torch.zeros((rows, 2), dtype=torch.float32),
        labels,
        contexts,
        torch.arange(rows, dtype=torch.int64),
    )
    return FeatureTables(table, table, tuple(range(4)))


def _base():
    return top_two_base_state(
        torch.arange(12, dtype=torch.float32).reshape(3, 4) / 20.0,
        torch.zeros(3, dtype=torch.float32),
        torch.tensor([[0.2, -0.1, 0.3], [-0.2, 0.4, 0.1]], dtype=torch.float32),
        torch.zeros(2, dtype=torch.float32),
    )


def test_primary_config_is_strict_and_resolved() -> None:
    source = Path("configs/vamp_af_mnist/poc.yaml")
    config = load_config(source)
    assert config.data.rotations_deg == (0.0, 18.0, 36.0, 54.0, 72.0)
    assert config.passes[1].seeds == (0, 1, 2)
    assert config.passes[2].depth_cap_override == 3


def test_primary_config_rejects_unknown_scientific_key(tmp_path: Path) -> None:
    record = yaml.safe_load(Path("configs/vamp_af_mnist/poc.yaml").read_text(encoding="utf-8"))
    record["structure"]["hidden_choice"] = 7
    path = tmp_path / "configs" / "vamp_af_mnist" / "poc.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(record), encoding="utf-8")
    with pytest.raises(ValueError, match="structure keys"):
        load_config(path)


def test_pass_table_is_blocked_deterministic_and_context_complete() -> None:
    tables = _tables()
    left = pass_training_table(tables, 2, 7)
    right = pass_training_table(tables, 2, 7)
    assert torch.equal(left.embeddings, right.embeddings)
    assert left.context_ids.tolist() == [context for context in range(5) for _ in range(2)]
    assert left.stream_steps.tolist() == list(range(10))


def test_tiny_end_to_end_run_writes_every_required_artifact(tmp_path: Path) -> None:
    config = _config(tmp_path)
    result = run_pass_seed(
        config,
        config.passes[0],
        0,
        _tables(),
        _base(),
        tmp_path / "run",
        torch.device("cpu"),
    )
    assert result.summary["final_leaves"] > 1
    assert all((result.directory / name).is_file() for name in REQUIRED_ARTIFACTS)
    assert result.summary["conditions"].keys() == {
        "af",
        "frozen_base",
        "global_replay",
        "joint_iid",
        "oracle_context",
    }
    resumed = run_pass_seed(
        config,
        config.passes[0],
        0,
        _tables(),
        _base(),
        tmp_path / "run",
        torch.device("cpu"),
    )
    assert resumed.summary == result.summary

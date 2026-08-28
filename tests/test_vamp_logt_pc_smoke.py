from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from apm.experiments.vamp_logt_pc_data import PcRawTable
from apm.experiments.vamp_logt_pc_state import load_bank_checkpoint
from apm.experiments.vamp_logt_pc_training import SelectedPcProtocol
from apm.experiments.vamp_logt_pc_workflow import _build_bank


def test_two_block_pc_bank_writes_resumes_and_retires_children(tmp_path: Path) -> None:
    config = _tiny_config()
    stream = _tiny_stream()
    selected = SelectedPcProtocol(2.0, 1.0, 0.01)

    first = _build_bank(config, selected, stream, tmp_path, 2)
    second = _build_bank(config, selected, stream, tmp_path, 2)
    loaded = load_bank_checkpoint(tmp_path / "bank.json")

    assert first == second == loaded
    assert first.topology.processed_blocks == 2
    assert tuple(node.level for node in first.topology.active_nodes) == (1,)
    assert first.counters.pc_leaf_example_presentations == 4
    assert first.counters.pc_merge_example_presentations == 4
    assert first.counters.classifier_example_presentations == 8
    live_node = first.topology.active_nodes[0]
    model_directories = tuple(path.name for path in (tmp_path / "models").iterdir())
    assert model_directories == (live_node.node_id,)
    assert (tmp_path / "models" / live_node.node_id / "replica-0" / "manifest.json").is_file()


def _tiny_config() -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(
            latent_dim=1,
            hidden_dim=2,
            image_dim=784,
            weight_init_std=0.02,
        ),
        training=SimpleNamespace(
            epochs=1,
            batch_size=2,
            learning_rate=1.0e-3,
            weight_decay=0.0,
            infer_steps=1,
            score_batch_size=2,
            classifier_epochs=1,
            classifier_batch_size=2,
            classifier_learning_rate=1.0e-2,
            classifier_weight_decay=0.0,
        ),
        stream=SimpleNamespace(block_size=2, model_seeds=(0,)),
        runtime=SimpleNamespace(progress=False),
    )


def _tiny_stream() -> PcRawTable:
    raw = np.zeros((4, 784), dtype=np.uint8)
    raw[0, :4] = (0, 64, 128, 255)
    raw[1, :4] = (255, 128, 64, 0)
    raw[2, :4] = (32, 96, 160, 224)
    raw[3, :4] = (224, 160, 96, 32)
    return PcRawTable(
        raw,
        np.asarray([0, 1, 0, 1], dtype=np.int64),
        np.zeros(4, dtype=np.int64),
        np.arange(4, dtype=np.int64),
    )

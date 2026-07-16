from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

import apm.lm.text_data as text_data

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "train_tinyshakespeare.py"


def test_training_script_import_has_no_network_or_training_side_effects(monkeypatch) -> None:
    monkeypatch.setattr(
        text_data,
        "urlopen",
        lambda *args, **kwargs: pytest.fail("script import attempted network access"),
    )
    specification = importlib.util.spec_from_file_location(
        "train_tinyshakespeare_smoke",
        SCRIPT_PATH,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)

    specification.loader.exec_module(module)

    assert module.SEED == 0
    assert module.CORPUS_PATH == REPOSITORY_ROOT / "data/tinyshakespeare/input.txt"
    assert module.CHECKPOINT_PATH == REPOSITORY_ROOT / "checkpoints/tinyshakespeare-base"
    assert callable(module.main)

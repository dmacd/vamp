"""Atomic exact-boundary training checkpoints for short vision jobs."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from collections.abc import Mapping
import os
import tempfile

import torch
from torch import Tensor

from apm.continual.artifacts import fsync_directory


def atomic_torch_save(path: str | Path, value: object) -> Path:
    """Serialize a trusted local PyTorch state and atomically replace its checkpoint."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            torch.save(value, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, target)
        fsync_directory(target.parent)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return target


def load_training_checkpoint(path: str | Path) -> dict[str, object]:
    """Load a local trusted checkpoint and require the exact vision schema."""
    value = torch.load(Path(path), map_location="cpu", weights_only=False)
    if type(value) is not dict or value.get("schema_version") != "imagenetr50-training-checkpoint-v1":
        raise ValueError("unknown or malformed vision training checkpoint")
    return value


__all__ = ["atomic_torch_save", "load_training_checkpoint"]

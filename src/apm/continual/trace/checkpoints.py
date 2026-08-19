"""Exact-resume adapter training checkpoints with atomic publication."""

from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import Tensor

from apm.continual.artifacts import atomic_write
from apm.continual.trace.protocol import stable_seed


@dataclass(frozen=True, slots=True)
class TrainingProgress:
    """Exact next-presentation cursor and authenticated ledger boundary."""

    plan_hash: str
    next_presentation: int
    optimizer_steps: int
    ledger_rows: int
    elapsed_seconds: float = 0.0

    def __post_init__(self) -> None:
        if (
            min(self.next_presentation, self.optimizer_steps, self.ledger_rows) < 0
            or self.elapsed_seconds < 0.0
        ):
            raise ValueError("training checkpoint counters must be nonnegative")


def trainable_state(model: torch.nn.Module) -> dict[str, Tensor]:
    """Copy only trainable model tensors to CPU for a compact checkpoint."""
    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if name in trainable_names
    }


def save_training_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    progress: TrainingProgress,
) -> Path:
    """Atomically save trainable tensors, optimizer, scheduler, and all RNG states."""
    payload: dict[str, object] = {
        "cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else (),
        "format": "trace-training-checkpoint-v1",
        "optimizer": optimizer.state_dict(),
        "progress": {
            "gradient_accumulation_position": 0,
            "ledger_rows": progress.ledger_rows,
            "elapsed_seconds": progress.elapsed_seconds,
            "next_presentation": progress.next_presentation,
            "optimizer_steps": progress.optimizer_steps,
            "plan_hash": progress.plan_hash,
        },
        "scheduler": scheduler.state_dict(),
        "trainable_state": trainable_state(model),
    }
    stream = io.BytesIO()
    torch.save(payload, stream)
    return atomic_write(path, stream.getvalue())


def load_training_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    expected_plan_hash: str,
) -> TrainingProgress:
    """Restore an exact safe-boundary checkpoint and return its next cursor."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if type(payload) is not dict or payload.get("format") != "trace-training-checkpoint-v1":
        raise ValueError("training checkpoint format differs")
    raw_progress = payload.get("progress")
    raw_state = payload.get("trainable_state")
    if type(raw_progress) is not dict or type(raw_state) is not dict:
        raise ValueError("training checkpoint payload is incomplete")
    if raw_progress.get("gradient_accumulation_position") != 0:
        raise ValueError("TRACE checkpoints must be saved at optimizer boundaries")
    progress = TrainingProgress(
        plan_hash=str(raw_progress["plan_hash"]),
        next_presentation=int(raw_progress["next_presentation"]),
        optimizer_steps=int(raw_progress["optimizer_steps"]),
        ledger_rows=int(raw_progress["ledger_rows"]),
        elapsed_seconds=float(raw_progress["elapsed_seconds"]),
    )
    if progress.plan_hash != expected_plan_hash:
        raise ValueError("training checkpoint belongs to another presentation plan")
    current = model.state_dict()
    if not set(raw_state) <= set(current):
        raise ValueError("checkpoint trainable tensors do not match the model")
    for name, value in raw_state.items():
        if not isinstance(value, Tensor) or current[name].shape != value.shape:
            raise ValueError("checkpoint trainable tensor shape differs")
        current[name].copy_(value.to(device=current[name].device, dtype=current[name].dtype))
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    torch.set_rng_state(payload["cpu_rng_state"])
    cuda_states = payload["cuda_rng_states"]
    if torch.cuda.is_available() and isinstance(cuda_states, Sequence) and cuda_states:
        torch.cuda.set_rng_state_all(cuda_states)
    return progress


def initialize_training_rng(plan_hash: str) -> None:
    """Seed CPU and CUDA generators from one presentation-plan identity."""
    seed = stable_seed("training", plan_hash)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


__all__ = [
    "TrainingProgress",
    "initialize_training_rng",
    "load_training_checkpoint",
    "save_training_checkpoint",
    "trainable_state",
]

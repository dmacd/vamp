"""Deterministic adapter-only training with exact safe-boundary resumption."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import time

import torch
from torch import Tensor
from tqdm.auto import tqdm

from apm.continual.artifacts import ChainedJsonlLedger
from apm.continual.trace.checkpoints import (
    TrainingProgress,
    initialize_training_rng,
    load_training_checkpoint,
    save_training_checkpoint,
)
from apm.continual.trace.collator import Tokenizer, TraceDataCollator
from apm.continual.trace.data import TraceExample
from apm.continual.trace.protocol import TrainingConfig
from apm.continual.trace.training_plans import TrainingPlan


TRAINING_LEDGER_FORMAT = "trace-training-step-v1"


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Completed optimizer work and presentation accounting for one plan."""

    plan_hash: str
    presentations: int
    tokens: int
    optimizer_steps: int
    mean_loss: float
    elapsed_seconds: float
    checkpoint_path: Path

    def as_record(self) -> dict[str, object]:
        """Return JSON-compatible training metrics."""
        return {
            "checkpoint_path": str(self.checkpoint_path),
            "elapsed_seconds": self.elapsed_seconds,
            "mean_loss": self.mean_loss,
            "optimizer_steps": self.optimizer_steps,
            "plan_hash": self.plan_hash,
            "presentations": self.presentations,
            "tokens": self.tokens,
        }


def train_adapter(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    examples_by_id: Mapping[str, TraceExample],
    plan: TrainingPlan,
    checkpoint_path: str | Path,
    ledger_path: str | Path,
    config: TrainingConfig = TrainingConfig(),
    *,
    checkpoint_step_interval: int = 50,
    checkpoint_seconds: float = 120.0,
    should_pause: Callable[[], bool] = lambda: False,
    on_phase_boundary: Callable[[torch.nn.Module, str, int], None] = lambda _model, _name, _index: None,
) -> TrainingResult:
    """Train one adapter plan and checkpoint every safe optimizer boundary."""
    if set(plan.example_ids) - set(examples_by_id):
        raise ValueError("training plan references unavailable examples")
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    if not parameters:
        raise ValueError("adapter training found no trainable parameters")
    optimizer = torch.optim.Adam(
        parameters,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    checkpoint = Path(checkpoint_path)
    ledger = ChainedJsonlLedger(ledger_path, TRAINING_LEDGER_FORMAT)
    initialize_training_rng(plan.plan_hash)
    progress = TrainingProgress(plan.plan_hash, 0, 0, 0)
    if checkpoint.is_file():
        progress = load_training_checkpoint(
            checkpoint,
            model,
            optimizer,
            scheduler,
            plan.plan_hash,
        )
        if progress.next_presentation > len(plan.example_ids):
            raise ValueError("checkpoint cursor exceeds its training plan")
        ledger.truncate(progress.ledger_rows)
    elif ledger.rows:
        raise ValueError("training ledger exists without an exact-resume checkpoint")

    device = parameters[0].device
    collator = TraceDataCollator(tokenizer, config)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    last_checkpoint = started
    accumulated_loss = 0.0
    accumulated_tokens = 0
    total_loss = sum(float(row["mean_loss"]) for row in ledger.rows)
    total_tokens = sum(int(row["tokens"]) for row in ledger.rows)
    completed_steps = progress.optimizer_steps
    start_index = progress.next_presentation
    bar = tqdm(
        total=len(plan.example_ids),
        initial=start_index,
        desc=f"TRACE {plan.name}",
        unit="example",
        dynamic_ncols=True,
    )
    print(f"TRACE phase: training {plan.name} ({len(plan.example_ids):,} presentations)")
    pending = 0
    next_index = start_index
    boundary_names = {end: name for name, end in plan.phase_boundaries}
    try:
        for index in range(start_index, len(plan.example_ids)):
            example = examples_by_id[plan.example_ids[index]]
            batch = collator.training_batch(
                ({"answer": example.answer, "prompt": example.prompt},)
            )
            device_batch = {name: value.to(device) for name, value in batch.items()}
            context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device.type == "cuda"
                else nullcontext()
            )
            with context:
                output = model(**device_batch)
                loss = getattr(output, "loss", None)
                if not isinstance(loss, Tensor) or loss.ndim != 0:
                    raise TypeError("causal LM did not return one scalar loss")
                scaled_loss = loss / config.gradient_accumulation_steps
            scaled_loss.backward()
            accumulated_loss += float(loss.detach().to(torch.float32).cpu().item())
            accumulated_tokens += int(device_batch["attention_mask"].sum().item())
            pending += 1
            next_index = index + 1
            phase_name = boundary_names.get(next_index)
            at_boundary = pending == config.gradient_accumulation_steps
            at_end = next_index == len(plan.example_ids)
            if at_boundary or at_end or phase_name is not None:
                if pending < config.gradient_accumulation_steps:
                    correction = config.gradient_accumulation_steps / pending
                    for parameter in parameters:
                        if parameter.grad is not None:
                            parameter.grad.mul_(correction)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                completed_steps += 1
                mean_step_loss = accumulated_loss / pending
                total_loss += mean_step_loss
                total_tokens += accumulated_tokens
                ledger.append(
                    {
                        "mean_loss": mean_step_loss,
                        "next_presentation": next_index,
                        "optimizer_step": completed_steps,
                        "plan_hash": plan.plan_hash,
                        "tokens": accumulated_tokens,
                    }
                )
                pending = 0
                accumulated_loss = 0.0
                accumulated_tokens = 0
                now = time.monotonic()
                pause_requested = should_pause()
                due = (
                    completed_steps % checkpoint_step_interval == 0
                    or now - last_checkpoint >= checkpoint_seconds
                    or at_end
                    or phase_name is not None
                    or pause_requested
                )
                if phase_name is not None:
                    on_phase_boundary(model, phase_name, next_index)
                if due:
                    save_training_checkpoint(
                        checkpoint,
                        model,
                        optimizer,
                        scheduler,
                        TrainingProgress(
                            plan_hash=plan.plan_hash,
                            next_presentation=next_index,
                            optimizer_steps=completed_steps,
                            ledger_rows=ledger.next_sequence,
                            elapsed_seconds=progress.elapsed_seconds
                            + now
                            - started,
                        ),
                    )
                    last_checkpoint = now
                if pause_requested:
                    raise InterruptedError("TRACE training paused at a safe boundary")
            bar.update(1)
    finally:
        bar.close()
    elapsed = progress.elapsed_seconds + time.monotonic() - started
    step_count = len(ledger.rows)
    return TrainingResult(
        plan_hash=plan.plan_hash,
        presentations=len(plan.example_ids),
        tokens=total_tokens,
        optimizer_steps=completed_steps,
        mean_loss=total_loss / step_count if step_count else float("nan"),
        elapsed_seconds=elapsed,
        checkpoint_path=checkpoint,
    )


__all__ = ["TRAINING_LEDGER_FORMAT", "TrainingResult", "train_adapter"]

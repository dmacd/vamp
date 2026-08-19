"""Matched TRACE baseline presentation-plan construction."""

from __future__ import annotations

from collections.abc import Sequence

from apm.continual.trace.data import TraceExample
from apm.continual.trace.training_plans import (
    TrainingPlan,
    joint_iid_plan,
    sequential_40_plan,
    sequential_reference_plan,
    taskwise_plans,
)


def baseline_plans(examples: Sequence[TraceExample]) -> dict[str, tuple[TrainingPlan, ...]]:
    """Return all registered baselines with independent-adapter groups explicit."""
    return {
        "joint_iid_lora": (joint_iid_plan(examples),),
        "seq_lora_40": (sequential_40_plan(examples),),
        "seq_lora_reference": (sequential_reference_plan(examples),),
        "taskwise_lora": taskwise_plans(examples),
    }


__all__ = ["baseline_plans"]

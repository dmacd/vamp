"""Direct-prediction, retention, and frozen-node reference metrics."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from apm.continual.logt_behavioral_integrator import IntegratorObservations
from apm.continual.logt_evidence_bank import TemporalNode
from apm.experiments.vamp_logt_router_data import ExampleBatch


FIXED_CONTROLS = (
    "mean_ensemble",
    "most_recent_range",
    "largest_range",
    "uniform_active",
    "best_single_node",
)


def fixed_control_logits(
    control: str,
    nodes: tuple[TemporalNode, ...],
    observations: IntegratorObservations,
    labels: Tensor,
    seed: int,
) -> Tensor:
    """Return final class logits for one fixed node-combination control."""
    if not nodes or labels.shape != (len(observations.features),):
        raise ValueError("fixed integration control received an invalid frontier")
    if control == "mean_ensemble":
        return observations.baseline_log_probabilities.clone()
    levels = tuple(sorted(node.level for node in nodes))
    if control == "most_recent_range":
        selected = torch.full(
            (len(labels),),
            max(nodes, key=lambda node: node.last_block).level,
            dtype=torch.int64,
        )
    elif control == "largest_range":
        selected = torch.full(
            (len(labels),), max(levels), dtype=torch.int64
        )
    elif control == "uniform_active":
        active = torch.tensor(levels, dtype=torch.int64)
        generator = torch.Generator().manual_seed(seed)
        selected = active[
            torch.randint(len(active), (len(labels),), generator=generator)
        ]
    elif control == "best_single_node":
        losses = -observations.node_log_probabilities.gather(
            2,
            labels[:, None, None].expand(-1, len(observations.active_mask), 1),
        ).squeeze(2)
        losses[:, ~observations.active_mask] = torch.inf
        selected = losses.argmin(dim=1)
    else:
        raise ValueError(f"unknown integration control: {control}")
    return observations.node_log_probabilities[
        torch.arange(len(labels)), selected
    ].clone()


def prediction_metric_rows(
    *,
    condition: str,
    logits: Tensor,
    examples: ExampleBatch,
    node_observations: IntegratorObservations,
    nodes: tuple[TemporalNode, ...],
    run_seed: int,
    macro_step: int,
    evaluation_scope: str,
    joint_logits: Tensor | None = None,
) -> tuple[dict[str, object], ...]:
    """Return micro, temporal, age, and domain views for one predictor."""
    if logits.shape != (len(examples.labels), 10):
        raise ValueError("prediction metrics received misaligned class logits")
    groups = _group_masks(examples, nodes, macro_step, evaluation_scope)
    ranges = tuple((name, mask) for name, mask in groups if name.startswith("range:"))
    range_losses = tuple(
        float(F.cross_entropy(logits[mask], examples.labels[mask]).item())
        for _name, mask in ranges
        if bool(mask.any())
    )
    common = {
        "active_levels": [node.level for node in sorted(nodes, key=lambda value: value.level)],
        "active_node_count": len(nodes),
        "condition": condition,
        "evaluation_scope": evaluation_scope,
        "macro_step": macro_step,
        "range_macro_mean_cross_entropy": (
            float(np.mean(range_losses)) if range_losses else None
        ),
        "run_seed": run_seed,
        "temporal_ranges": [
            [node.first_block + 1, node.last_block + 1]
            for node in sorted(nodes, key=lambda value: value.first_block)
        ],
        "worst_range_mean_cross_entropy": max(range_losses) if range_losses else None,
    }
    best_logits = fixed_control_logits(
        "best_single_node", nodes, node_observations, examples.labels, 0
    )
    return tuple(
        {
            **common,
            "group": name,
            **_metrics(
                logits[mask],
                examples.labels[mask],
                best_logits[mask],
                None if joint_logits is None else joint_logits[mask],
            ),
        }
        for name, mask in groups
        if bool(mask.any())
    )


def _metrics(
    logits: Tensor,
    labels: Tensor,
    best_single_logits: Tensor,
    joint_logits: Tensor | None,
) -> dict[str, object]:
    losses = F.cross_entropy(logits, labels, reduction="none")
    best_losses = F.cross_entropy(best_single_logits, labels, reduction="none")
    probabilities = F.softmax(logits, dim=1)
    one_hot = F.one_hot(labels, num_classes=10).to(torch.float32)
    accuracy = float((logits.argmax(dim=1) == labels).float().mean().item())
    best_accuracy = float(
        (best_single_logits.argmax(dim=1) == labels).float().mean().item()
    )
    result: dict[str, object] = {
        "accuracy": accuracy,
        "accuracy_gap_from_best_single_node": best_accuracy - accuracy,
        "best_single_node_accuracy": best_accuracy,
        "best_single_node_mean_cross_entropy": float(best_losses.mean().item()),
        "brier_score": float(((probabilities - one_hot) ** 2).sum(dim=1).mean().item()),
        "cross_entropy_gap_from_best_single_node": float(
            (losses - best_losses).mean().item()
        ),
        "example_count": len(labels),
        "mean_cross_entropy": float(losses.mean().item()),
    }
    if joint_logits is None:
        result.update(
            {
                "accuracy_gap_from_joint_iid": None,
                "cross_entropy_gap_from_joint_iid": None,
                "joint_iid_accuracy": None,
                "joint_iid_mean_cross_entropy": None,
            }
        )
    else:
        joint_losses = F.cross_entropy(joint_logits, labels, reduction="none")
        joint_accuracy = float(
            (joint_logits.argmax(dim=1) == labels).float().mean().item()
        )
        result.update(
            {
                "accuracy_gap_from_joint_iid": joint_accuracy - accuracy,
                "cross_entropy_gap_from_joint_iid": float(
                    (losses - joint_losses).mean().item()
                ),
                "joint_iid_accuracy": joint_accuracy,
                "joint_iid_mean_cross_entropy": float(joint_losses.mean().item()),
            }
        )
    if any(
        not math.isfinite(float(result[name]))
        for name in ("accuracy", "mean_cross_entropy", "brier_score")
    ):
        raise RuntimeError("prediction-integrator metrics contain non-finite values")
    return result


def _group_masks(
    examples: ExampleBatch,
    nodes: tuple[TemporalNode, ...],
    macro_step: int,
    scope: str,
) -> tuple[tuple[str, Tensor], ...]:
    count = len(examples.labels)
    groups: list[tuple[str, Tensor]] = [
        ("micro", torch.ones(count, dtype=torch.bool))
    ]
    if bool(torch.any(examples.macro_steps > 0)):
        origins = examples.macro_steps - 1
        groups.extend(
            (
                f"range:{node.first_block + 1}-{node.last_block + 1}",
                (origins >= node.first_block) & (origins <= node.last_block),
            )
            for node in sorted(nodes, key=lambda value: value.first_block)
        )
        current_node = max(nodes, key=lambda value: value.last_block)
        current = (
            (origins >= current_node.first_block)
            & (origins <= current_node.last_block)
        )
        groups.extend((("current_range", current), ("older_ranges", ~current)))
        groups.extend(
            (f"age:{int(age)}", macro_step - examples.macro_steps == age)
            for age in sorted(set((macro_step - examples.macro_steps).tolist()))
        )
    if scope in {"test_subset", "full_test"}:
        groups.extend(
            (f"domain:{int(domain)}", examples.domain_ids == domain)
            for domain in sorted(set(examples.domain_ids.tolist()))
        )
    return tuple(groups)


__all__ = ["FIXED_CONTROLS", "fixed_control_logits", "prediction_metric_rows"]

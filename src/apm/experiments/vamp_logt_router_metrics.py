"""Routing-regret metrics and hierarchy-versus-routing decomposition."""

from __future__ import annotations

from collections.abc import Mapping
import math

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from apm.continual.logt_behavioral_router import RouterSupervision
from apm.continual.logt_evidence_bank import TemporalNode
from apm.experiments.vamp_logt_router_data import ExampleBatch


def fixed_policy_selections(
    policy: str,
    nodes: tuple[TemporalNode, ...],
    supervision: RouterSupervision,
    seed: int,
) -> Tensor:
    """Return deterministic selections for one fixed or oracle policy."""
    if not nodes:
        raise ValueError("fixed routing requires a nonempty active frontier")
    if policy == "oracle":
        return supervision.hard_targets.clone()
    if policy == "most_recent_range":
        level = max(nodes, key=lambda node: node.last_block).level
        return torch.full((len(supervision.features),), level, dtype=torch.int64)
    if policy == "largest_range":
        level = max(node.level for node in nodes)
        return torch.full((len(supervision.features),), level, dtype=torch.int64)
    if policy == "uniform_active":
        levels = torch.tensor(
            sorted(node.level for node in nodes), dtype=torch.int64
        )
        generator = torch.Generator().manual_seed(seed)
        return levels[
            torch.randint(len(levels), (len(supervision.features),), generator=generator)
        ]
    raise ValueError(f"unknown fixed routing policy: {policy}")


def routing_metric_rows(
    *,
    condition: str,
    selections: Tensor,
    probabilities: Tensor | None,
    inactive_attempts: Tensor | None,
    examples: ExampleBatch,
    supervision: RouterSupervision,
    nodes: tuple[TemporalNode, ...],
    run_seed: int,
    macro_step: int,
    evaluation_scope: str,
    near_oracle_thresholds: tuple[float, ...],
    joint_logits: Tensor | None = None,
) -> tuple[dict[str, object], ...]:
    """Return aggregate and retention views for one aligned routing policy."""
    rows = len(examples.labels)
    if (
        selections.shape != (rows,)
        or (probabilities is not None and probabilities.shape != supervision.node_losses.shape)
        or (inactive_attempts is not None and inactive_attempts.shape != (rows,))
    ):
        raise ValueError("routing metrics received misaligned policy outputs")
    group_masks = _group_masks(examples, nodes, macro_step, evaluation_scope)
    range_masks = tuple((name, mask) for name, mask in group_masks if name.startswith("range:"))
    range_means = tuple(
        _mean_regret(selections[mask], supervision.node_losses[mask])
        for _name, mask in range_masks
        if bool(mask.any())
    )
    common = {
        "active_levels": [node.level for node in sorted(nodes, key=lambda value: value.level)],
        "active_node_count": len(nodes),
        "condition": condition,
        "evaluation_scope": evaluation_scope,
        "macro_step": macro_step,
        "range_macro_mean_regret": float(np.mean(range_means)) if range_means else None,
        "run_seed": run_seed,
        "temporal_ranges": [
            [node.first_block + 1, node.last_block + 1]
            for node in sorted(nodes, key=lambda value: value.first_block)
        ],
        "worst_range_mean_regret": max(range_means) if range_means else None,
    }
    return tuple(
        {
            **common,
            "group": name,
            **_metrics(
                selections[mask],
                None if probabilities is None else probabilities[mask],
                None if inactive_attempts is None else inactive_attempts[mask],
                examples.labels[mask],
                supervision.node_logits[mask],
                supervision.node_losses[mask],
                supervision.hard_targets[mask],
                supervision.active_mask,
                near_oracle_thresholds,
                None if joint_logits is None else joint_logits[mask],
            ),
        }
        for name, mask in group_masks
        if bool(mask.any())
    )


def _metrics(
    selections: Tensor,
    probabilities: Tensor | None,
    inactive_attempts: Tensor | None,
    labels: Tensor,
    node_logits: Tensor,
    node_losses: Tensor,
    hard_targets: Tensor,
    active_mask: Tensor,
    thresholds: tuple[float, ...],
    joint_logits: Tensor | None,
) -> dict[str, object]:
    rows = torch.arange(len(labels))
    selected_losses = node_losses[rows, selections]
    oracle_losses = node_losses[rows, hard_targets]
    regret = selected_losses - oracle_losses
    if float(regret.min().item()) < -1.0e-6:
        raise RuntimeError("reported routing regret is negative beyond tolerance")
    regret = regret.clamp_min(0.0)
    selected_logits = node_logits[rows, selections]
    oracle_logits = node_logits[rows, hard_targets]
    selected_accuracy = float((selected_logits.argmax(dim=1) == labels).float().mean().item())
    oracle_accuracy = float((oracle_logits.argmax(dim=1) == labels).float().mean().item())
    active_losses = node_losses[:, active_mask]
    second_margin = (
        active_losses.topk(min(2, active_losses.shape[1]), largest=False, dim=1).values
    )
    margins = (
        torch.zeros(len(labels))
        if second_margin.shape[1] == 1
        else second_margin[:, 1] - second_margin[:, 0]
    )
    margin_edges = torch.tensor(
        [0.0, 0.001, 0.005, 0.01, 0.05, 0.10, 0.25, 0.50, 1.0, 2.0, 5.0, 1.0e9]
    )
    margin_histogram = torch.histogram(margins, bins=margin_edges).hist.to(torch.int64)
    confusion = torch.zeros(
        active_mask.shape[0],
        active_mask.shape[0],
        dtype=torch.int64,
    )
    for target, selected in zip(hard_targets.tolist(), selections.tolist()):
        confusion[int(target), int(selected)] += 1
    result: dict[str, object] = {
        "accuracy_gap_from_oracle": oracle_accuracy - selected_accuracy,
        "best_second_loss_margin_mean": float(margins.mean().item()),
        "best_second_loss_margin_p10": float(torch.quantile(margins, 0.10).item()),
        "best_second_loss_margin_histogram": {
            "counts": margin_histogram.tolist(),
            "edges": margin_edges.tolist(),
        },
        "cross_entropy_gap_from_oracle": float(regret.mean().item()),
        "example_count": len(labels),
        "inactive_attempt_rate": (
            None
            if inactive_attempts is None
            else float(inactive_attempts.float().mean().item())
        ),
        "mean_regret": float(regret.mean().item()),
        "median_regret": float(regret.median().item()),
        "near_oracle_rates": {
            f"{threshold:.2f}": float((regret <= threshold).float().mean().item())
            for threshold in thresholds
        },
        "oracle_accuracy": oracle_accuracy,
        "oracle_match_rate": float(
            (selections == hard_targets).float().mean().item()
        ),
        "oracle_mean_cross_entropy": float(oracle_losses.mean().item()),
        "p90_regret": float(torch.quantile(regret, 0.90).item()),
        "router_entropy": (
            None
            if probabilities is None
            else float(
                (-(probabilities * probabilities.clamp_min(1.0e-12).log()).sum(dim=1))
                .mean()
                .item()
            )
        ),
        "selected_accuracy": selected_accuracy,
        "selected_mean_cross_entropy": float(selected_losses.mean().item()),
        "selection_counts": torch.bincount(
            selections, minlength=active_mask.shape[0]
        ).tolist(),
        "target_counts": torch.bincount(
            hard_targets, minlength=active_mask.shape[0]
        ).tolist(),
        "target_selection_confusion": confusion.tolist(),
    }
    if joint_logits is not None:
        joint_losses = F.cross_entropy(joint_logits, labels, reduction="none")
        joint_accuracy = float((joint_logits.argmax(dim=1) == labels).float().mean().item())
        result.update(
            {
                "hierarchy_accuracy_gap": joint_accuracy - oracle_accuracy,
                "hierarchy_cross_entropy_gap": float(
                    (oracle_losses - joint_losses).mean().item()
                ),
                "joint_iid_accuracy": joint_accuracy,
                "joint_iid_mean_cross_entropy": float(joint_losses.mean().item()),
                "total_cross_entropy_gap_from_joint": float(
                    (selected_losses - joint_losses).mean().item()
                ),
            }
        )
    else:
        result.update(
            {
                "hierarchy_accuracy_gap": None,
                "hierarchy_cross_entropy_gap": None,
                "joint_iid_accuracy": None,
                "joint_iid_mean_cross_entropy": None,
                "total_cross_entropy_gap_from_joint": None,
            }
        )
    if any(not math.isfinite(float(value)) for value in (
        result["mean_regret"],
        result["selected_accuracy"],
        result["oracle_accuracy"],
    )):
        raise RuntimeError("routing metrics contain non-finite primary values")
    return result


def _mean_regret(selections: Tensor, losses: Tensor) -> float:
    rows = torch.arange(len(selections))
    return float((losses[rows, selections] - losses.min(dim=1).values).mean().item())


def _group_masks(
    examples: ExampleBatch,
    nodes: tuple[TemporalNode, ...],
    macro_step: int,
    scope: str,
) -> tuple[tuple[str, Tensor], ...]:
    count = len(examples.labels)
    groups: list[tuple[str, Tensor]] = [("micro", torch.ones(count, dtype=torch.bool))]
    if bool(torch.any(examples.macro_steps > 0)):
        origins = examples.macro_steps - 1
        for node in sorted(nodes, key=lambda value: value.first_block):
            mask = (origins >= node.first_block) & (origins <= node.last_block)
            groups.append((f"range:{node.first_block + 1}-{node.last_block + 1}", mask))
        current_node = max(nodes, key=lambda value: value.last_block)
        current = (origins >= current_node.first_block) & (origins <= current_node.last_block)
        groups.extend((("current_range", current), ("older_ranges", ~current)))
        for age in sorted(set((macro_step - examples.macro_steps).tolist())):
            groups.append((f"age:{int(age)}", macro_step - examples.macro_steps == age))
    if scope.startswith("test"):
        for domain in sorted(set(examples.domain_ids.tolist())):
            groups.append((f"domain:{int(domain)}", examples.domain_ids == domain))
    return tuple(groups)


__all__ = ["fixed_policy_selections", "routing_metric_rows"]

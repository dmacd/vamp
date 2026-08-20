"""Deterministic training-proxy-only per-node temperature and offset fitting."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from apm.continual.artifacts import record_sha256


@dataclass(frozen=True, slots=True)
class CalibrationExamples:
    """Training-derived labels kept structurally outside task-free evaluator inputs."""

    image_ids: tuple[str, ...]
    global_labels: Tensor

    def __post_init__(self) -> None:
        if (
            len(self.image_ids) < 1
            or len(set(self.image_ids)) != len(self.image_ids)
            or self.global_labels.ndim != 1
            or self.global_labels.shape[0] != len(self.image_ids)
            or torch.any((self.global_labels < 0) | (self.global_labels >= 200))
        ):
            raise ValueError("invalid calibration examples")


@dataclass(frozen=True, slots=True)
class AffineCalibration:
    """Positive temperatures and identifiable zero-mean offsets for live nodes."""

    node_hashes: tuple[str, ...]
    temperatures: tuple[float, ...]
    offsets: tuple[float, ...]
    proxy_hash: str

    def __post_init__(self) -> None:
        if (
            self.node_hashes != tuple(sorted(set(self.node_hashes)))
            or len(self.temperatures) != len(self.node_hashes)
            or len(self.offsets) != len(self.node_hashes)
            or any(value <= 0.0 for value in self.temperatures)
            or abs(sum(self.offsets)) > 1.0e-5
            or len(self.proxy_hash) != 64
        ):
            raise ValueError("invalid affine calibration")

    def parameters_for(self, node_hash: str) -> tuple[float, float]:
        """Return the fitted temperature and offset for one live node."""
        index = self.node_hashes.index(node_hash)
        return self.temperatures[index], self.offsets[index]


def fit_affine_calibration(
    examples: CalibrationExamples,
    raw_logits_by_node: Mapping[str, Tensor],
    class_ids_by_node: Mapping[str, Sequence[int]],
    steps: int = 250,
    learning_rate: float = 0.03,
) -> AffineCalibration:
    """Fit node temperatures/offsets by deterministic full-batch proxy cross entropy."""
    node_hashes = tuple(sorted(raw_logits_by_node))
    if (
        not node_hashes
        or set(node_hashes) != set(class_ids_by_node)
        or steps < 1
        or learning_rate <= 0.0
    ):
        raise ValueError("invalid calibration node inputs")
    rows = len(examples.image_ids)
    if any(
        tuple(raw_logits_by_node[node].shape)
        != (rows, len(tuple(class_ids_by_node[node])))
        for node in node_hashes
    ):
        raise ValueError("calibration logits do not cover the exact proxy rows")
    all_classes = tuple(
        class_id for node in node_hashes for class_id in class_ids_by_node[node]
    )
    if len(all_classes) != len(set(all_classes)):
        raise ValueError("live calibration nodes have overlapping class rows")
    ordered_classes = tuple(sorted(all_classes))
    target_lookup = torch.full((200,), -1, dtype=torch.long)
    target_lookup[torch.tensor(ordered_classes)] = torch.arange(len(ordered_classes))
    local_targets = target_lookup[examples.global_labels.to(device="cpu")]
    if torch.any(local_targets < 0):
        raise ValueError("calibration proxy label is outside live represented classes")

    log_temperatures = torch.zeros(len(node_hashes), dtype=torch.float64, requires_grad=True)
    offsets = torch.zeros(len(node_hashes), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam((log_temperatures, offsets), lr=learning_rate)
    class_positions = {class_id: index for index, class_id in enumerate(ordered_classes)}
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        centered_offsets = offsets - offsets.mean()
        assembled = torch.full((rows, len(ordered_classes)), -torch.inf, dtype=torch.float64)
        for index, node in enumerate(node_hashes):
            positions = torch.tensor(
                [class_positions[value] for value in class_ids_by_node[node]], dtype=torch.long
            )
            assembled[:, positions] = (
                raw_logits_by_node[node].detach().to(torch.float64)
                / torch.exp(log_temperatures[index])
                + centered_offsets[index]
            )
        loss = F.cross_entropy(assembled, local_targets)
        loss.backward()
        optimizer.step()
    final_offsets = offsets.detach() - offsets.detach().mean()
    return AffineCalibration(
        node_hashes=node_hashes,
        temperatures=tuple(float(value) for value in torch.exp(log_temperatures.detach()).tolist()),
        offsets=tuple(float(value) for value in final_offsets.tolist()),
        proxy_hash=record_sha256(
            {
                "image_ids": list(examples.image_ids),
                "labels": [int(value) for value in examples.global_labels.tolist()],
                "schema_version": "imagenetr50-calibration-proxies-v1",
            }
        ),
    )


__all__ = ["AffineCalibration", "CalibrationExamples", "fit_affine_calibration"]

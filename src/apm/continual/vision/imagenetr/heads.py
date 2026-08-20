"""Disjoint affine classifier rows and exact union operations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable, Sequence
import json

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from apm.continual.artifacts import file_sha256


@dataclass(frozen=True, slots=True)
class ClassifierRows:
    """Owned global class IDs with ordinary affine weight and bias rows."""

    class_ids: tuple[int, ...]
    weight: Tensor
    bias: Tensor

    def __post_init__(self) -> None:
        if (
            self.class_ids != tuple(sorted(set(self.class_ids)))
            or any(not 0 <= class_id < 200 for class_id in self.class_ids)
            or self.weight.ndim != 2
            or self.bias.ndim != 1
            or self.weight.shape[0] != len(self.class_ids)
            or self.bias.shape[0] != len(self.class_ids)
        ):
            raise ValueError("invalid affine classifier rows")


class AffineClassifier(nn.Module):
    """Trainable ordinary affine rows over an explicit global class subset."""

    def __init__(
        self,
        class_ids: Sequence[int],
        feature_dim: int,
        initialization_seed: int = 0,
        initial_rows: ClassifierRows | None = None,
    ) -> None:
        super().__init__()
        ids = tuple(sorted(set(int(value) for value in class_ids)))
        if not ids or any(not 0 <= value < 200 for value in ids) or feature_dim < 1:
            raise ValueError("classifier requires valid represented classes and features")
        self.class_ids = ids
        self.weight = nn.Parameter(torch.empty(len(ids), feature_dim))
        self.bias = nn.Parameter(torch.zeros(len(ids)))
        lookup = torch.full((200,), -1, dtype=torch.long)
        lookup[torch.tensor(ids)] = torch.arange(len(ids))
        self.register_buffer("global_to_local", lookup, persistent=True)
        if initial_rows is None:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(initialization_seed)
                nn.init.trunc_normal_(self.weight, std=0.02)
                nn.init.zeros_(self.bias)
        else:
            self.load_rows(initial_rows)

    def forward(self, features: Tensor) -> Tensor:
        """Return raw affine logits in the stored class-row order."""
        return F.linear(features, self.weight, self.bias)

    def cosine_scores(self, features: Tensor, scale: float) -> Tensor:
        """Return bias-free normalized feature/weight scores at one global scale."""
        if scale <= 0.0:
            raise ValueError("cosine score scale must be positive")
        return scale * (F.normalize(features, dim=-1) @ F.normalize(self.weight, dim=-1).T)

    def local_targets(self, global_targets: Tensor) -> Tensor:
        """Map represented global labels to local cross-entropy row indices."""
        local = self.global_to_local[global_targets]
        if torch.any(local < 0):
            raise ValueError("training targets include an unrepresented classifier class")
        return local

    def rows(self) -> ClassifierRows:
        """Export detached cloned affine rows for immutable union and storage."""
        return ClassifierRows(
            self.class_ids,
            self.weight.detach().clone(),
            self.bias.detach().clone(),
        )

    def selected_rows(self, class_ids: Iterable[int]) -> ClassifierRows:
        """Snapshot an exact represented-row subset for later restoration."""
        selected = tuple(sorted(set(int(value) for value in class_ids)))
        if not selected or not set(selected) <= set(self.class_ids):
            raise ValueError("selected classifier rows are not represented")
        positions = torch.tensor(
            [self.class_ids.index(class_id) for class_id in selected],
            device=self.weight.device,
            dtype=torch.long,
        )
        return ClassifierRows(
            selected,
            self.weight.detach()[positions].clone(),
            self.bias.detach()[positions].clone(),
        )

    def restore_rows(self, rows: ClassifierRows) -> None:
        """Restore an exact represented-row subset without touching other rows."""
        if not set(rows.class_ids) <= set(self.class_ids) or rows.weight.shape[1] != self.weight.shape[1]:
            raise ValueError("restored classifier rows are not represented")
        positions = torch.tensor(
            [self.class_ids.index(class_id) for class_id in rows.class_ids],
            device=self.weight.device,
            dtype=torch.long,
        )
        with torch.no_grad():
            self.weight.index_copy_(0, positions, rows.weight.to(self.weight))
            self.bias.index_copy_(0, positions, rows.bias.to(self.bias))

    def load_rows(self, rows: ClassifierRows) -> None:
        """Load exact same-class affine rows."""
        if rows.class_ids != self.class_ids or rows.weight.shape != self.weight.shape:
            raise ValueError("classifier rows do not match this head")
        with torch.no_grad():
            self.weight.copy_(rows.weight.to(self.weight))
            self.bias.copy_(rows.bias.to(self.bias))

    def mask_inactive_gradients(self, active_class_ids: Iterable[int]) -> None:
        """Zero gradient rows outside the current task for sequential training."""
        active = frozenset(active_class_ids)
        if not active <= frozenset(self.class_ids):
            raise ValueError("active classifier rows are not represented")
        mask = torch.tensor(
            [class_id in active for class_id in self.class_ids],
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        if self.weight.grad is not None:
            self.weight.grad.mul_(mask[:, None])
        if self.bias.grad is not None:
            self.bias.grad.mul_(mask)


def union_classifier_rows(children: Sequence[ClassifierRows]) -> ClassifierRows:
    """Union disjoint child rows exactly, sorted by global class ID."""
    if len(children) < 2:
        raise ValueError("classifier union requires at least two children")
    dimensions = {rows.weight.shape[1] for rows in children}
    all_ids = tuple(class_id for rows in children for class_id in rows.class_ids)
    if len(dimensions) != 1 or len(all_ids) != len(set(all_ids)):
        raise ValueError("classifier child rows overlap or have different feature dimensions")
    indexed = {
        class_id: (rows.weight[index], rows.bias[index])
        for rows in children
        for index, class_id in enumerate(rows.class_ids)
    }
    ordered = tuple(sorted(indexed))
    return ClassifierRows(
        ordered,
        torch.stack(tuple(indexed[class_id][0] for class_id in ordered)),
        torch.stack(tuple(indexed[class_id][1] for class_id in ordered)),
    )


def save_classifier(path: str | Path, rows: ClassifierRows) -> str:
    """Save ordinary affine rows to safetensors and return their file identity."""
    try:
        from safetensors.torch import save_file
    except ImportError as error:  # pragma: no cover - vision environment gate
        raise RuntimeError("safetensors is required by the vision environment") from error
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "bias": rows.bias.detach().cpu().contiguous(),
            "weight": rows.weight.detach().cpu().contiguous(),
        },
        target,
        metadata={
            "class_ids": json.dumps(list(rows.class_ids), separators=(",", ":")),
            "schema_version": "imagenetr50-affine-classifier-v1",
        },
    )
    return file_sha256(target)


def load_classifier(path: str | Path) -> ClassifierRows:
    """Load and validate ordinary affine classifier rows from safetensors."""
    try:
        from safetensors import safe_open
    except ImportError as error:  # pragma: no cover - vision environment gate
        raise RuntimeError("safetensors is required by the vision environment") from error
    with safe_open(Path(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        if metadata.get("schema_version") != "imagenetr50-affine-classifier-v1":
            raise ValueError("unknown classifier safetensors schema")
        if set(handle.keys()) != {"bias", "weight"}:
            raise ValueError("classifier safetensors keys changed")
        return ClassifierRows(
            tuple(int(value) for value in json.loads(metadata["class_ids"])),
            handle.get_tensor("weight"),
            handle.get_tensor("bias"),
        )


__all__ = [
    "AffineClassifier",
    "ClassifierRows",
    "load_classifier",
    "save_classifier",
    "union_classifier_rows",
]

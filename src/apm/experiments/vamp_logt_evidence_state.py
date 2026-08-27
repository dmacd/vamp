"""Immutable active-bank state and crash-safe node-model artifact lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import re

import torch
from pyrsistent import PMap, pmap

from apm.continual.artifacts import require_sha256
from apm.continual.logt_evidence_bank import (
    EvidenceWorkCounters,
    LogTState,
    empty_logt_state,
)
from apm.continual.nce_tre_evidence import FrozenEvidenceState
from apm.continual.top_two_adapter import TopTwoAdapterState
from apm.continual.vision.imagenetr.checkpoints import atomic_torch_save


_CONDITION = re.compile(r"[a-z0-9][a-z0-9-]*\Z")


@dataclass(frozen=True, slots=True)
class ActiveEvidenceBank:
    """One functional LogT frontier with adapters and zero or more evidence families."""

    topology: LogTState
    adapters: PMap[str, TopTwoAdapterState]
    evidence_by_condition: PMap[str, PMap[str, FrozenEvidenceState]]
    counters: EvidenceWorkCounters
    adapter_example_updates: int

    def __post_init__(self) -> None:
        active = {node.node_id for node in self.topology.active_nodes}
        if (
            set(self.adapters) != active
            or any(set(models) != active for models in self.evidence_by_condition.values())
            or any(_CONDITION.fullmatch(condition) is None for condition in self.evidence_by_condition)
            or self.counters.active_evidence_models
            != len(active) * len(self.evidence_by_condition)
            or self.adapter_example_updates < 0
        ):
            raise ValueError("active model state does not exactly match the LogT frontier")


@dataclass(frozen=True, slots=True)
class StoredAdapterResult:
    """A loaded or newly published node-adapter artifact."""

    adapter: TopTwoAdapterState
    final_loss: float
    example_updates: int


@dataclass(frozen=True, slots=True)
class StoredEvidenceResult:
    """A loaded or newly published node-evidence artifact."""

    state: FrozenEvidenceState
    final_loss: float
    example_updates: int
    reference_examples: int


def empty_active_bank(block_size: int) -> ActiveEvidenceBank:
    """Return an empty adapter/evidence bank for a fixed stream block size."""
    return ActiveEvidenceBank(
        empty_logt_state(block_size),
        pmap(),
        pmap(),
        EvidenceWorkCounters(),
        0,
    )


def save_bank_checkpoint(path: Path, bank: ActiveEvidenceBank) -> None:
    """Atomically publish the latest exact online boundary."""
    atomic_torch_save(
        path,
        {"bank": bank, "schema_version": "vamp-logt-active-bank-v1"},
    )


def load_bank_checkpoint(path: Path) -> ActiveEvidenceBank:
    """Load and validate one exact active-bank checkpoint."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != "vamp-logt-active-bank-v1" or not isinstance(
        payload.get("bank"), ActiveEvidenceBank
    ):
        raise ValueError("active-bank checkpoint is malformed or has the wrong schema")
    return payload["bank"]


def adapter_artifact_path(nodes_root: Path, node_id: str) -> Path:
    """Return the scoped adapter artifact path for one content-addressed node."""
    require_sha256(node_id, "node artifact ID")
    return nodes_root / node_id / "adapter.pt"


def evidence_artifact_path(
    nodes_root: Path,
    node_id: str,
    condition: str,
) -> Path:
    """Return the scoped evidence artifact path for one node and condition."""
    require_sha256(node_id, "node artifact ID")
    if _CONDITION.fullmatch(condition) is None:
        raise ValueError("evidence condition is not a safe artifact component")
    return nodes_root / node_id / "evidence" / f"{condition}.pt"


def publish_adapter_result(
    path: Path,
    node_id: str,
    seed: int,
    adapter: TopTwoAdapterState,
    final_loss: float,
    example_updates: int,
) -> StoredAdapterResult:
    """Publish one adapter once, or require semantic identity with the prior artifact."""
    if path.is_file():
        return load_adapter_result(path, node_id, seed)
    if not math.isfinite(final_loss) or example_updates < 1:
        raise ValueError("cannot publish an incomplete adapter result")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(
        path,
        {
            "example_updates": example_updates,
            "final_loss": final_loss,
            "node_id": node_id,
            "parameters": tuple(tensor.detach().cpu() for tensor in adapter.tensors),
            "schema_version": "vamp-logt-node-adapter-v1",
            "seed": seed,
        },
    )
    return load_adapter_result(path, node_id, seed)


def load_adapter_result(path: Path, node_id: str, seed: int) -> StoredAdapterResult:
    """Load one adapter artifact and validate its semantic coordinates."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("schema_version") != "vamp-logt-node-adapter-v1"
        or payload.get("node_id") != node_id
        or payload.get("seed") != seed
        or not isinstance(payload.get("parameters"), tuple)
        or len(payload["parameters"]) != 4
    ):
        raise ValueError("node adapter artifact coordinates changed")
    return StoredAdapterResult(
        TopTwoAdapterState(*payload["parameters"]),
        float(payload["final_loss"]),
        int(payload["example_updates"]),
    )


def publish_evidence_result(
    path: Path,
    node_id: str,
    condition: str,
    seed: int,
    state: FrozenEvidenceState,
    final_loss: float,
    example_updates: int,
    reference_examples: int,
) -> StoredEvidenceResult:
    """Publish one evidence state once, or require its prior semantic coordinates."""
    if path.is_file():
        return load_evidence_result(path, node_id, condition, seed)
    if (
        not math.isfinite(final_loss)
        or example_updates < 1
        or reference_examples != example_updates
    ):
        raise ValueError("cannot publish an incomplete balanced evidence result")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(
        path,
        {
            "bridges": state.bridges,
            "condition": condition,
            "example_updates": example_updates,
            "final_loss": final_loss,
            "node_id": node_id,
            "parameters": state.parameters,
            "reference_examples": reference_examples,
            "schema_version": "vamp-logt-node-evidence-v1",
            "seed": seed,
        },
    )
    return load_evidence_result(path, node_id, condition, seed)


def load_evidence_result(
    path: Path,
    node_id: str,
    condition: str,
    seed: int,
) -> StoredEvidenceResult:
    """Load one evidence artifact and validate its semantic coordinates."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("schema_version") != "vamp-logt-node-evidence-v1"
        or payload.get("node_id") != node_id
        or payload.get("condition") != condition
        or payload.get("seed") != seed
    ):
        raise ValueError("node evidence artifact coordinates changed")
    return StoredEvidenceResult(
        FrozenEvidenceState(int(payload["bridges"]), tuple(payload["parameters"])),
        float(payload["final_loss"]),
        int(payload["example_updates"]),
        int(payload["reference_examples"]),
    )


def retire_inactive_node_artifacts(nodes_root: Path, active_node_ids: set[str]) -> tuple[str, ...]:
    """Delete only complete content-addressed node directories outside the live frontier."""
    if not nodes_root.is_dir():
        return ()
    for node_id in active_node_ids:
        require_sha256(node_id, "active node artifact ID")
    retired = []
    for directory in sorted(path for path in nodes_root.iterdir() if path.is_dir()):
        require_sha256(directory.name, "stored node artifact ID")
        if directory.name in active_node_ids:
            continue
        for file_path in sorted(path for path in directory.rglob("*") if path.is_file()):
            file_path.unlink()
        for child in sorted(
            (path for path in directory.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            child.rmdir()
        directory.rmdir()
        retired.append(directory.name)
    return tuple(retired)


__all__ = [
    "ActiveEvidenceBank",
    "StoredAdapterResult",
    "StoredEvidenceResult",
    "adapter_artifact_path",
    "empty_active_bank",
    "evidence_artifact_path",
    "load_adapter_result",
    "load_bank_checkpoint",
    "load_evidence_result",
    "publish_adapter_result",
    "publish_evidence_result",
    "retire_inactive_node_artifacts",
    "save_bank_checkpoint",
]

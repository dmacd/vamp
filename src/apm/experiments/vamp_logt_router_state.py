"""Immutable adapter frontier and crash-safe node artifact lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping

import torch
from pyrsistent import PMap, pmap

from apm.continual.logt_evidence_bank import LogTState, TemporalNode, empty_logt_state, insert_block
from apm.continual.top_two_adapter import TopTwoAdapterState
from apm.continual.vision.imagenetr.checkpoints import atomic_torch_save
from apm.experiments.vamp_logt_evidence_training import train_node_adapter
from apm.experiments.vamp_logt_router_config import VampLogTRouterConfig
from apm.experiments.vamp_logt_router_data import (
    ExampleBatch,
    FrozenClassifierDependency,
    named_seed,
)
from apm.continual.logt_behavioral_router import frozen_trunk_features


@dataclass(frozen=True, slots=True)
class ActiveAdapterBank:
    """One functional binary-counter frontier of de-novo adapters."""

    topology: LogTState
    adapters: PMap[str, TopTwoAdapterState]
    adapter_example_updates: int = 0

    def __post_init__(self) -> None:
        active = {node.node_id for node in self.topology.active_nodes}
        if set(self.adapters) != active or self.adapter_example_updates < 0:
            raise ValueError("active adapters do not exactly match the LogT frontier")


def empty_adapter_bank(block_size: int) -> ActiveAdapterBank:
    """Return an empty adapter-only LogT frontier."""
    return ActiveAdapterBank(empty_logt_state(block_size), pmap())


def advance_adapter_bank(
    config: VampLogTRouterConfig,
    bank: ActiveAdapterBank,
    model_archive: ExampleBatch,
    dependency: FrozenClassifierDependency,
    run_seed: int,
    nodes_root: Path,
    device: torch.device,
) -> ActiveAdapterBank:
    """Insert one model batch and train every induced node de novo."""
    block_size = config.benchmark.model_batch_size
    first_example = bank.topology.processed_blocks * block_size
    if len(model_archive.labels) != first_example + block_size:
        raise ValueError("model archive does not end at exactly one new LogT block")
    topology, leaf, merges = insert_block(
        bank.topology,
        tuple(range(first_example, first_example + block_size)),
    )
    adapters = bank.adapters
    updates = bank.adapter_example_updates
    for node, is_merge in ((leaf, False), *((merge.parent, True) for merge in merges)):
        adapter, example_updates = _fit_or_load_adapter(
            config,
            node,
            model_archive,
            dependency,
            run_seed,
            nodes_root,
            device,
        )
        if is_merge:
            adapters = adapters.remove(node.parent_node_ids[0]).remove(node.parent_node_ids[1])
        adapters = adapters.set(node.node_id, adapter)
        updates += example_updates
    return ActiveAdapterBank(topology, adapters, updates)


def retire_inactive_nodes(nodes_root: Path, active_node_ids: set[str]) -> tuple[str, ...]:
    """Delete complete inactive node directories after the enclosing checkpoint commits."""
    if not nodes_root.is_dir():
        return ()
    retired = []
    for directory in sorted(path for path in nodes_root.iterdir() if path.is_dir()):
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


def bank_record(bank: ActiveAdapterBank) -> dict[str, object]:
    """Return the explicit checkpoint payload for one active frontier."""
    return {
        "adapter_example_updates": bank.adapter_example_updates,
        "adapters": {
            node_id: tuple(tensor.detach().cpu() for tensor in adapter.tensors)
            for node_id, adapter in bank.adapters.items()
        },
        "topology": bank.topology,
    }


def bank_from_record(record: Mapping[str, object]) -> ActiveAdapterBank:
    """Reconstruct and validate one adapter frontier checkpoint record."""
    topology = record.get("topology")
    adapters = record.get("adapters")
    if not isinstance(topology, LogTState) or not isinstance(adapters, Mapping):
        raise ValueError("adapter-bank checkpoint is malformed")
    return ActiveAdapterBank(
        topology,
        pmap(
            {
                str(node_id): TopTwoAdapterState(*tuple(tensors))
                for node_id, tensors in adapters.items()
            }
        ),
        int(record["adapter_example_updates"]),
    )


def _fit_or_load_adapter(
    config: VampLogTRouterConfig,
    node: TemporalNode,
    archive: ExampleBatch,
    dependency: FrozenClassifierDependency,
    run_seed: int,
    nodes_root: Path,
    device: torch.device,
) -> tuple[TopTwoAdapterState, int]:
    path = nodes_root / node.node_id / "adapter.pt"
    seed = named_seed(run_seed, "node-adapter", node.node_id)
    if path.is_file():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (
            payload.get("schema_version") != "vamp-logt-router-node-adapter-v1"
            or payload.get("node_id") != node.node_id
            or payload.get("seed") != seed
        ):
            raise ValueError("stored router-experiment adapter coordinates changed")
        return TopTwoAdapterState(*payload["parameters"]), int(payload["example_updates"])
    ids = torch.tensor(node.example_ids, dtype=torch.int64)
    trunk = frozen_trunk_features(
        dependency.model,
        archive.images[ids],
        device,
        config.evaluation.inference_batch_size,
    )
    result = train_node_adapter(
        trunk,
        archive.labels[ids],
        dependency.base,
        config.adapter,
        seed,
        device,
        f"adapter L{node.level} n={len(ids)}",
        config.runtime.progress,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(
        path,
        {
            "example_updates": result.example_updates,
            "node_id": node.node_id,
            "parameters": tuple(tensor.detach().cpu() for tensor in result.adapter.tensors),
            "schema_version": "vamp-logt-router-node-adapter-v1",
            "seed": seed,
        },
    )
    return result.adapter, result.example_updates


__all__ = [
    "ActiveAdapterBank",
    "advance_adapter_bank",
    "bank_from_record",
    "bank_record",
    "empty_adapter_bank",
    "retire_inactive_nodes",
]

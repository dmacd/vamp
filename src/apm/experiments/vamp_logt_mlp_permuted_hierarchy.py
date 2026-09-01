"""Immutable full-model LogT hierarchy tape for dense Permuted-MNIST."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F
from pyrsistent import PMap, pmap

from apm.continual.artifacts import (
    file_sha256,
    load_canonical_json,
    publish_immutable_json,
    record_sha256,
)
from apm.continual.dense_mlp_adapter import (
    DenseExamples,
    DenseMlpState,
    dense_delta,
    dense_hidden_logits,
    fit_dense_model,
    zero_dense_delta,
)
from apm.continual.logt_behavioral_integrator import IntegratorObservations
from apm.continual.logt_behavioral_router import RouterSupervision
from apm.continual.logt_evidence_bank import (
    LogTState,
    TemporalNode,
    empty_logt_state,
    insert_block,
)
from apm.continual.vision.imagenetr.checkpoints import atomic_torch_save
from apm.experiments.vamp_logt_mlp_permuted_calibration import load_calibrated_base
from apm.experiments.vamp_logt_mlp_permuted_config import VampLogTDenseConfig
from apm.experiments.vamp_logt_mlp_permuted_data import (
    ExampleBatch,
    PermutedMnistBenchmark,
    build_benchmark,
    concatenate_batches,
    named_seed,
)


@dataclass(frozen=True, slots=True)
class DenseFrontier:
    """One authenticated step of the immutable hierarchy tape."""

    macro_step: int
    nodes: tuple[TemporalNode, ...]
    deltas: PMap[str, DenseMlpState]
    node_checkpoint_sha256: PMap[str, str]

    def __post_init__(self) -> None:
        node_ids = {node.node_id for node in self.nodes}
        if (
            self.macro_step < 1
            or set(self.deltas) != node_ids
            or set(self.node_checkpoint_sha256) != node_ids
            or len({node.level for node in self.nodes}) != len(self.nodes)
        ):
            raise ValueError("dense frontier is incomplete or misaligned")


@dataclass(frozen=True, slots=True)
class DenseObservations:
    """Shared label-free features plus router targets behind a strict boundary."""

    integrator: IntegratorObservations
    router: RouterSupervision


def build_hierarchy_tape(
    config: VampLogTDenseConfig,
    run_root: Path,
    device: torch.device,
) -> tuple[dict[str, object], ...]:
    """Run or resume every seed's hierarchy and retain all created nodes."""
    base = load_calibrated_base(config, run_root)
    summaries = tuple(
        _build_seed_tape(config, run_root, seed, base, device)
        for seed in config.online.seeds
    )
    publish_immutable_json(
        run_root / "hierarchy" / "summary.json",
        {
            "config_hash": config.config_hash,
            "schema_version": "vamp-logt-dense-hierarchy-summary-v1",
            "seeds": list(summaries),
            "status": "complete",
        },
    )
    return summaries


def load_frontier(
    config: VampLogTDenseConfig,
    run_root: Path,
    seed: int,
    macro_step: int,
) -> DenseFrontier:
    """Authenticate one frontier manifest and load its immutable node deltas."""
    path = run_root / "hierarchy" / f"seed-{seed}" / "frontiers" / f"step-{macro_step:03d}.json"
    manifest = load_canonical_json(path)
    if (
        manifest.get("config_hash") != config.config_hash
        or int(manifest.get("run_seed", -1)) != seed
        or int(manifest.get("macro_step", -1)) != macro_step
    ):
        raise ValueError("dense hierarchy frontier coordinates changed")
    nodes = tuple(_node_from_record(row) for row in manifest["active_nodes"])
    deltas = {}
    hashes = {}
    for node, row in zip(nodes, manifest["active_nodes"], strict=True):
        checkpoint = run_root / "hierarchy" / f"seed-{seed}" / "nodes" / node.node_id / "delta.pt"
        expected = str(row["checkpoint_sha256"])
        if file_sha256(checkpoint) != expected:
            raise ValueError(f"dense hierarchy node changed: {node.node_id}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if payload.get("node_id") != node.node_id or payload.get("config_hash") != config.config_hash:
            raise ValueError("dense node checkpoint metadata changed")
        deltas[node.node_id] = DenseMlpState(tuple(payload["delta_parameters"]))
        hashes[node.node_id] = expected
    _require_frontier_partition(nodes, macro_step, config.benchmark.model_batch_size)
    return DenseFrontier(macro_step, nodes, pmap(deltas), pmap(hashes))


def build_dense_observations(
    frontier: DenseFrontier,
    examples: ExampleBatch,
    base: DenseMlpState,
    maximum_levels: int,
    temperature: float,
    device: torch.device,
    batch_size: int,
) -> DenseObservations:
    """Build shared fixed-slot node behaviors and current-frontier targets."""
    if (
        not frontier.nodes
        or any(not 0 <= node.level < maximum_levels for node in frontier.nodes)
        or temperature <= 0.0
        or batch_size < 1
    ):
        raise ValueError("invalid dense observer request")
    rows = len(examples.labels)
    slot_dim = base.embedding_dim + 10 + 1
    features = torch.zeros((rows, maximum_levels, slot_dim), dtype=torch.float32)
    node_logits = torch.zeros((rows, maximum_levels, 10), dtype=torch.float32)
    node_log_probabilities = torch.zeros_like(node_logits)
    active_mask = torch.zeros(maximum_levels, dtype=torch.bool)
    flattened = examples.images.flatten(1)
    target_base = DenseMlpState(tuple(tensor.to(device) for tensor in base.tensors))
    for node in sorted(frontier.nodes, key=lambda value: value.level):
        delta = DenseMlpState(tuple(tensor.to(device) for tensor in frontier.deltas[node.node_id].tensors))
        hidden_rows, logit_rows = [], []
        with torch.inference_mode():
            for offset in range(0, rows, batch_size):
                hidden, logits = dense_hidden_logits(
                    flattened[offset : offset + batch_size].to(device),
                    target_base,
                    delta,
                )
                hidden_rows.append(hidden.cpu())
                logit_rows.append(logits.cpu())
        hidden = torch.cat(hidden_rows)
        logits = torch.cat(logit_rows)
        log_probabilities = F.log_softmax(logits, dim=1)
        features[:, node.level, :-1] = torch.cat(
            (F.layer_norm(hidden, (base.embedding_dim,)), log_probabilities),
            dim=1,
        )
        features[:, node.level, -1] = 1.0
        node_logits[:, node.level] = logits
        node_log_probabilities[:, node.level] = log_probabilities
        active_mask[node.level] = True
    active_probabilities = node_log_probabilities[:, active_mask]
    baseline = torch.logsumexp(active_probabilities, dim=1) - torch.log(
        torch.tensor(float(active_probabilities.shape[1]))
    )
    expanded_labels = examples.labels[:, None, None].expand(-1, maximum_levels, 1)
    node_losses = -F.log_softmax(node_logits, dim=2).gather(2, expanded_labels).squeeze(2)
    node_losses[:, ~active_mask] = torch.inf
    hard_targets = node_losses.argmin(dim=1)
    teacher_logits = -(node_losses - node_losses.min(dim=1, keepdim=True).values) / temperature
    teacher_logits[:, ~active_mask] = -torch.inf
    flattened_features = features.flatten(1).detach()
    return DenseObservations(
        IntegratorObservations(
            flattened_features,
            node_log_probabilities.detach(),
            active_mask.detach(),
            baseline.detach(),
        ),
        RouterSupervision(
            flattened_features,
            node_logits.detach(),
            node_losses.detach(),
            hard_targets.detach(),
            F.softmax(teacher_logits, dim=1).detach(),
            active_mask.detach(),
        ),
    )


def build_base_observations(
    examples: ExampleBatch,
    base: DenseMlpState,
    maximum_levels: int,
    device: torch.device,
    batch_size: int,
) -> IntegratorObservations:
    """Place frozen-base behavior in slot zero for the matched base-only control."""
    hidden_rows, logit_rows = [], []
    target_base = DenseMlpState(tuple(tensor.to(device) for tensor in base.tensors))
    target_delta = zero_dense_delta(target_base)
    flattened = examples.images.flatten(1)
    with torch.inference_mode():
        for offset in range(0, len(examples.labels), batch_size):
            hidden, logits = dense_hidden_logits(
                flattened[offset : offset + batch_size].to(device),
                target_base,
                target_delta,
            )
            hidden_rows.append(hidden.cpu())
            logit_rows.append(logits.cpu())
    hidden = torch.cat(hidden_rows)
    log_probabilities = F.log_softmax(torch.cat(logit_rows), dim=1)
    slot_dim = base.embedding_dim + 11
    features = torch.zeros((len(examples.labels), maximum_levels, slot_dim))
    node_log_probabilities = torch.zeros((len(examples.labels), maximum_levels, 10))
    features[:, 0, :-1] = torch.cat(
        (F.layer_norm(hidden, (base.embedding_dim,)), log_probabilities), dim=1
    )
    features[:, 0, -1] = 1.0
    node_log_probabilities[:, 0] = log_probabilities
    active_mask = torch.zeros(maximum_levels, dtype=torch.bool)
    active_mask[0] = True
    return IntegratorObservations(
        features.flatten(1).detach(),
        node_log_probabilities.detach(),
        active_mask,
        log_probabilities.detach(),
    )


def _build_seed_tape(
    config: VampLogTDenseConfig,
    run_root: Path,
    seed: int,
    base: DenseMlpState,
    device: torch.device,
) -> dict[str, object]:
    directory = run_root / "hierarchy" / f"seed-{seed}"
    summary_path = directory / "summary.json"
    if summary_path.is_file():
        summary = load_canonical_json(summary_path)
        stored_hashes = summary.get("all_node_checkpoint_sha256")
        if not isinstance(stored_hashes, dict):
            raise ValueError("dense hierarchy summary lacks the complete immutable node tape")
        for node_id, expected in stored_hashes.items():
            checkpoint = directory / "nodes" / str(node_id) / "delta.pt"
            if not checkpoint.is_file() or file_sha256(checkpoint) != expected:
                raise ValueError(f"dense hierarchy tape node changed: {node_id}")
        final_manifest = directory / "frontiers" / f"step-{config.benchmark.macro_steps:03d}.json"
        if file_sha256(final_manifest) != summary.get("final_frontier_sha256"):
            raise ValueError("dense hierarchy final frontier manifest changed")
        load_frontier(config, run_root, seed, config.benchmark.macro_steps)
        return summary
    benchmark = build_benchmark(config, seed)
    topology: LogTState = empty_logt_state(config.benchmark.model_batch_size)
    model_batches: tuple[ExampleBatch, ...] = ()
    node_hashes: dict[str, str] = {}
    example_updates = 0
    created_nodes = 0
    for macro_step in range(1, config.benchmark.macro_steps + 1):
        model_batches = (*model_batches, benchmark.step(macro_step).model)
        archive = concatenate_batches(model_batches)
        first_example = topology.processed_blocks * config.benchmark.model_batch_size
        topology, leaf, merges = insert_block(
            topology,
            tuple(range(first_example, first_example + config.benchmark.model_batch_size)),
        )
        for node in (leaf, *(merge.parent for merge in merges)):
            checkpoint_hash, updates = _fit_or_load_node(
                config, directory, seed, node, archive, base, device
            )
            node_hashes[node.node_id] = checkpoint_hash
            example_updates += updates
            created_nodes += 1
        active_nodes = topology.active_nodes
        publish_immutable_json(
            directory / "frontiers" / f"step-{macro_step:03d}.json",
            {
                "active_nodes": [
                    {**_node_record(node), "checkpoint_sha256": node_hashes[node.node_id]}
                    for node in active_nodes
                ],
                "config_hash": config.config_hash,
                "frontier_sha256": record_sha256([
                    [node.node_id, node_hashes[node.node_id]] for node in active_nodes
                ]),
                "macro_step": macro_step,
                "run_seed": seed,
                "schema_version": "vamp-logt-dense-frontier-v1",
            },
        )
    expected_nodes = 2 * config.benchmark.macro_steps - config.benchmark.macro_steps.bit_count()
    if created_nodes != expected_nodes:
        raise RuntimeError("dense hierarchy did not create the binary-counter node count")
    summary = {
        "config_hash": config.config_hash,
        "all_node_checkpoint_sha256": dict(sorted(node_hashes.items())),
        "created_node_count": created_nodes,
        "final_active_node_count": len(topology.active_nodes),
        "final_frontier_sha256": file_sha256(
            directory / "frontiers" / f"step-{config.benchmark.macro_steps:03d}.json"
        ),
        "node_example_updates": example_updates,
        "run_seed": seed,
        "schema_version": "vamp-logt-dense-hierarchy-seed-v1",
        "status": "complete",
    }
    publish_immutable_json(summary_path, summary)
    return summary


def _fit_or_load_node(
    config: VampLogTDenseConfig,
    directory: Path,
    seed: int,
    node: TemporalNode,
    archive: ExampleBatch,
    base: DenseMlpState,
    device: torch.device,
) -> tuple[str, int]:
    path = directory / "nodes" / node.node_id / "delta.pt"
    node_seed = named_seed(seed, "node", node.node_id)
    if path.is_file():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (
            payload.get("schema_version") != "vamp-logt-dense-node-v1"
            or payload.get("config_hash") != config.config_hash
            or payload.get("node_id") != node.node_id
            or int(payload.get("seed", -1)) != node_seed
        ):
            raise ValueError("stored dense node coordinates changed")
        return file_sha256(path), int(payload["example_updates"])
    ids = torch.tensor(node.example_ids, dtype=torch.int64)
    training = DenseExamples(
        archive.images[ids],
        archive.labels[ids],
        (torch.arange(784, dtype=torch.int64),),
    )
    result = fit_dense_model(
        training,
        base,
        config.node.optimizer,
        node_seed,
        device,
        fixed_epochs=config.node.epochs,
        dropout=config.node.dropout,
        progress_label=f"dense node L{node.level} n={len(ids)}",
        progress=config.runtime.progress,
    )
    delta = dense_delta(base, result.state)
    atomic_torch_save(
        path,
        {
            "config_hash": config.config_hash,
            "delta_parameters": delta.tensors,
            "example_updates": result.training_example_presentations,
            "node": _node_record(node),
            "node_id": node.node_id,
            "schema_version": "vamp-logt-dense-node-v1",
            "seed": node_seed,
        },
    )
    return file_sha256(path), result.training_example_presentations


def _node_record(node: TemporalNode) -> dict[str, object]:
    return {
        "example_ids": list(node.example_ids),
        "first_block": node.first_block,
        "last_block": node.last_block,
        "level": node.level,
        "node_id": node.node_id,
        "parent_node_ids": list(node.parent_node_ids),
    }


def _node_from_record(value: object) -> TemporalNode:
    if not isinstance(value, dict):
        raise ValueError("dense hierarchy node manifest is malformed")
    return TemporalNode(
        str(value["node_id"]),
        int(value["level"]),
        int(value["first_block"]),
        int(value["last_block"]),
        tuple(int(item) for item in value["example_ids"]),
        tuple(str(item) for item in value["parent_node_ids"]),
    )


def _require_frontier_partition(
    nodes: tuple[TemporalNode, ...],
    macro_step: int,
    block_size: int,
) -> None:
    blocks = tuple(
        block
        for node in sorted(nodes, key=lambda value: value.first_block)
        for block in range(node.first_block, node.last_block + 1)
    )
    examples = tuple(
        example
        for node in sorted(nodes, key=lambda value: value.first_block)
        for example in node.example_ids
    )
    if blocks != tuple(range(macro_step)) or examples != tuple(range(macro_step * block_size)):
        raise ValueError("dense hierarchy frontier does not partition the stream prefix")


__all__ = [
    "DenseFrontier",
    "DenseObservations",
    "build_base_observations",
    "build_dense_observations",
    "build_hierarchy_tape",
    "load_frontier",
]

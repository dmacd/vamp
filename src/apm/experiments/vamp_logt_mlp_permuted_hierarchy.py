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
    TemporalMerge,
    TemporalNode,
    merge_temporal_nodes,
    temporal_leaf,
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
    max_nodes_per_level: int = 1

    def __post_init__(self) -> None:
        node_ids = {node.node_id for node in self.nodes}
        level_counts = {
            level: sum(node.level == level for node in self.nodes)
            for level in {node.level for node in self.nodes}
        }
        if (
            self.macro_step < 1
            or self.max_nodes_per_level < 1
            or set(self.deltas) != node_ids
            or set(self.node_checkpoint_sha256) != node_ids
            or any(count > self.max_nodes_per_level for count in level_counts.values())
        ):
            raise ValueError("dense frontier is incomplete or misaligned")

    def node_slots(self, maximum_levels: int) -> PMap[str, int]:
        """Return stable rank-major slots for all active nodes."""
        return _node_slots(self.nodes, maximum_levels, self.max_nodes_per_level)


@dataclass(frozen=True, slots=True)
class DenseObservations:
    """Shared label-free features plus router targets behind a strict boundary."""

    integrator: IntegratorObservations
    router: RouterSupervision


@dataclass(frozen=True, slots=True)
class _DenseLogTState:
    """Immutable exponential-histogram frontier with a fixed level capacity."""

    block_size: int
    processed_blocks: int
    max_nodes_per_level: int
    active_by_level: PMap[int, tuple[TemporalNode, ...]]

    def __post_init__(self) -> None:
        groups = tuple(self.active_by_level.items())
        if (
            self.block_size < 1
            or self.processed_blocks < 0
            or self.max_nodes_per_level < 1
            or any(
                not nodes
                or len(nodes) > self.max_nodes_per_level
                or any(node.level != level for node in nodes)
                or tuple(sorted(nodes, key=lambda node: node.first_block)) != nodes
                for level, nodes in groups
            )
        ):
            raise ValueError("invalid dense LogT topology")
        chronological = self.active_nodes
        blocks = tuple(
            block
            for node in chronological
            for block in range(node.first_block, node.last_block + 1)
        )
        examples = tuple(example for node in chronological for example in node.example_ids)
        if (
            blocks != tuple(range(self.processed_blocks))
            or examples != tuple(range(self.processed_blocks * self.block_size))
        ):
            raise ValueError("dense LogT nodes do not partition the stream")

    @property
    def active_nodes(self) -> tuple[TemporalNode, ...]:
        """Return active nodes in chronological interval order."""
        return tuple(sorted(
            (node for nodes in self.active_by_level.values() for node in nodes),
            key=lambda node: node.first_block,
        ))


def _empty_dense_logt_state(block_size: int, max_nodes_per_level: int) -> _DenseLogTState:
    return _DenseLogTState(block_size, 0, max_nodes_per_level, pmap())


def _insert_dense_block(
    state: _DenseLogTState,
    example_ids: tuple[int, ...],
) -> tuple[_DenseLogTState, TemporalNode, tuple[TemporalMerge, ...]]:
    expected = tuple(range(
        state.processed_blocks * state.block_size,
        (state.processed_blocks + 1) * state.block_size,
    ))
    if example_ids != expected:
        raise ValueError("dense LogT blocks must be complete and chronological")
    leaf = temporal_leaf(state.processed_blocks, example_ids)
    active = state.active_by_level
    pending = leaf
    merges = []
    while True:
        residents = active.get(pending.level, ())
        candidates = (*residents, pending)
        if len(candidates) <= state.max_nodes_per_level:
            active = active.set(pending.level, candidates)
            break
        merge = merge_temporal_nodes(candidates[0], candidates[1])
        survivors = candidates[2:]
        active = (
            active.set(pending.level, survivors)
            if survivors
            else active.remove(pending.level)
        )
        merges.append(merge)
        pending = merge.parent
    return (
        _DenseLogTState(
            state.block_size,
            state.processed_blocks + 1,
            state.max_nodes_per_level,
            active,
        ),
        leaf,
        tuple(merges),
    )


def build_hierarchy_tape(
    config: VampLogTDenseConfig,
    run_root: Path,
    device: torch.device,
    *,
    max_nodes_per_level: int = 1,
    hierarchy_root: Path | None = None,
    training_sample_multiplier: int = 1,
    stop_after_step: int | None = None,
) -> tuple[dict[str, object], ...]:
    """Run or resume every seed's hierarchy through a complete or prefix target."""
    base = load_calibrated_base(config, run_root)
    target_root = run_root / "hierarchy" if hierarchy_root is None else hierarchy_root
    target_step = config.benchmark.macro_steps if stop_after_step is None else stop_after_step
    if not 1 <= target_step <= config.benchmark.macro_steps:
        raise ValueError("hierarchy target step is outside the configured stream")
    try:
        from tqdm.auto import tqdm
    except ImportError as error:  # pragma: no cover - environment dependency
        raise RuntimeError("tqdm is required by the vision environment") from error
    summaries = tuple(
        _build_seed_tape(
            config,
            run_root,
            target_root,
            seed,
            base,
            device,
            max_nodes_per_level,
            training_sample_multiplier,
            target_step,
        )
        for seed in tqdm(
            config.online.seeds,
            desc=(
                f"dense hierarchies capacity={max_nodes_per_level} "
                f"samples={training_sample_multiplier}x"
            ),
            disable=not config.runtime.progress,
            unit="seed",
        )
    )
    hierarchy_summary = {
            "config_hash": config.config_hash,
            "final_macro_step": target_step,
            "max_nodes_per_level": max_nodes_per_level,
            "seeds": list(summaries),
            "status": (
                "complete"
                if target_step == config.benchmark.macro_steps
                else "prefix_complete"
            ),
        }
    if _uses_sample_multiplier(config):
        hierarchy_summary.update({
            "schema_version": "vamp-logt-dense-hierarchy-summary-v2",
            "training_sample_multiplier": training_sample_multiplier,
        })
    else:
        hierarchy_summary["schema_version"] = "vamp-logt-dense-hierarchy-summary-v1"
    summary_name = (
        "summary.json"
        if target_step == config.benchmark.macro_steps
        else f"prefix-step-{target_step:03d}-summary.json"
    )
    publish_immutable_json(target_root / summary_name, hierarchy_summary)
    return summaries


def load_frontier(
    config: VampLogTDenseConfig,
    run_root: Path,
    seed: int,
    macro_step: int,
    *,
    max_nodes_per_level: int = 1,
    hierarchy_root: Path | None = None,
    training_sample_multiplier: int = 1,
) -> DenseFrontier:
    """Authenticate one frontier manifest and load its immutable node deltas."""
    target_root = run_root / "hierarchy" if hierarchy_root is None else hierarchy_root
    path = target_root / f"seed-{seed}" / "frontiers" / f"step-{macro_step:03d}.json"
    manifest = load_canonical_json(path)
    if (
        manifest.get("config_hash") != config.config_hash
        or int(manifest.get("run_seed", -1)) != seed
        or int(manifest.get("macro_step", -1)) != macro_step
        or int(manifest.get("max_nodes_per_level", -1)) != max_nodes_per_level
        or (
            _uses_sample_multiplier(config)
            and int(manifest.get("training_sample_multiplier", -1))
            != training_sample_multiplier
        )
    ):
        raise ValueError("dense hierarchy frontier coordinates changed")
    nodes = tuple(_node_from_record(row) for row in manifest["active_nodes"])
    deltas = {}
    hashes = {}
    for node, row in zip(nodes, manifest["active_nodes"], strict=True):
        checkpoint = target_root / f"seed-{seed}" / "nodes" / node.node_id / "delta.pt"
        expected = str(row["checkpoint_sha256"])
        if file_sha256(checkpoint) != expected:
            raise ValueError(f"dense hierarchy node changed: {node.node_id}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (
            payload.get("node_id") != node.node_id
            or payload.get("config_hash") != config.config_hash
            or (
                _uses_sample_multiplier(config)
                and int(payload.get("training_sample_multiplier", -1))
                != training_sample_multiplier
            )
        ):
            raise ValueError("dense node checkpoint metadata changed")
        deltas[node.node_id] = DenseMlpState(tuple(payload["delta_parameters"]))
        hashes[node.node_id] = expected
    _require_frontier_partition(
        nodes,
        macro_step,
        config.benchmark.model_batch_size * training_sample_multiplier,
    )
    frontier = DenseFrontier(
        macro_step,
        nodes,
        pmap(deltas),
        pmap(hashes),
        max_nodes_per_level,
    )
    declared_slots = {
        str(row["node_id"]): int(row["slot"])
        for row in manifest["active_nodes"]
    }
    if dict(frontier.node_slots(config.observer.maximum_levels)) != declared_slots:
        raise ValueError("dense hierarchy node slots changed")
    return frontier


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
    total_slots = maximum_levels * frontier.max_nodes_per_level
    features = torch.zeros((rows, total_slots, slot_dim), dtype=torch.float32)
    node_logits = torch.zeros((rows, total_slots, 10), dtype=torch.float32)
    node_log_probabilities = torch.zeros_like(node_logits)
    active_mask = torch.zeros(total_slots, dtype=torch.bool)
    node_slots = frontier.node_slots(maximum_levels)
    flattened = examples.images.flatten(1)
    target_base = DenseMlpState(tuple(tensor.to(device) for tensor in base.tensors))
    for node in sorted(frontier.nodes, key=lambda value: node_slots[value.node_id]):
        slot = node_slots[node.node_id]
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
        features[:, slot, :-1] = torch.cat(
            (F.layer_norm(hidden, (base.embedding_dim,)), log_probabilities),
            dim=1,
        )
        features[:, slot, -1] = 1.0
        node_logits[:, slot] = logits
        node_log_probabilities[:, slot] = log_probabilities
        active_mask[slot] = True
    active_probabilities = node_log_probabilities[:, active_mask]
    baseline = torch.logsumexp(active_probabilities, dim=1) - torch.log(
        torch.tensor(float(active_probabilities.shape[1]))
    )
    expanded_labels = examples.labels[:, None, None].expand(-1, total_slots, 1)
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
    hierarchy_root: Path,
    seed: int,
    base: DenseMlpState,
    device: torch.device,
    max_nodes_per_level: int,
    training_sample_multiplier: int,
    target_step: int,
) -> dict[str, object]:
    directory = hierarchy_root / f"seed-{seed}"
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
        if int(summary.get("max_nodes_per_level", -1)) != max_nodes_per_level:
            raise ValueError("dense hierarchy capacity changed")
        if (
            _uses_sample_multiplier(config)
            and int(summary.get("training_sample_multiplier", -1))
            != training_sample_multiplier
        ):
            raise ValueError("dense hierarchy training sample count changed")
        load_frontier(
            config,
            run_root,
            seed,
            config.benchmark.macro_steps,
            max_nodes_per_level=max_nodes_per_level,
            hierarchy_root=hierarchy_root,
            training_sample_multiplier=training_sample_multiplier,
        )
        return summary
    benchmark = (
        build_benchmark(config, seed)
        if training_sample_multiplier == 1
        else build_benchmark(
            config,
            seed,
            training_sample_multiplier=training_sample_multiplier,
        )
    )
    block_size = config.benchmark.model_batch_size * training_sample_multiplier
    topology = _empty_dense_logt_state(
        block_size,
        max_nodes_per_level,
    )
    model_batches: tuple[ExampleBatch, ...] = ()
    node_hashes: dict[str, str] = {}
    example_updates = 0
    created_nodes = 0
    try:
        from tqdm.auto import tqdm
    except ImportError as error:  # pragma: no cover - environment dependency
        raise RuntimeError("tqdm is required by the vision environment") from error
    macro_steps = tqdm(
        range(1, target_step + 1),
        total=target_step,
        desc=(
            f"hierarchy seed={seed} samples={training_sample_multiplier}x"
        ),
        disable=not config.runtime.progress,
        leave=False,
        unit="permutation",
    )
    for macro_step in macro_steps:
        model_batches = (*model_batches, benchmark.step(macro_step).model)
        archive = concatenate_batches(model_batches)
        first_example = topology.processed_blocks * block_size
        topology, leaf, merges = _insert_dense_block(
            topology,
            tuple(range(first_example, first_example + block_size)),
        )
        for node in (leaf, *(merge.parent for merge in merges)):
            checkpoint_hash, updates = _fit_or_load_node(
                config,
                directory,
                seed,
                node,
                archive,
                base,
                device,
                training_sample_multiplier,
            )
            node_hashes[node.node_id] = checkpoint_hash
            example_updates += updates
            created_nodes += 1
        active_nodes = topology.active_nodes
        slots = _node_slots(
            active_nodes,
            config.observer.maximum_levels,
            max_nodes_per_level,
        )
        frontier_record = {
                "active_nodes": [
                    {
                        **_node_record(node),
                        "checkpoint_sha256": node_hashes[node.node_id],
                        "slot": slots[node.node_id],
                    }
                    for node in active_nodes
                ],
                "config_hash": config.config_hash,
                "frontier_sha256": record_sha256([
                    [node.node_id, node_hashes[node.node_id]] for node in active_nodes
                ]),
                "macro_step": macro_step,
                "max_nodes_per_level": max_nodes_per_level,
                "run_seed": seed,
            }
        if _uses_sample_multiplier(config):
            frontier_record.update({
                "schema_version": "vamp-logt-dense-frontier-v2",
                "training_sample_multiplier": training_sample_multiplier,
            })
        else:
            frontier_record["schema_version"] = "vamp-logt-dense-frontier-v1"
        publish_immutable_json(
            directory / "frontiers" / f"step-{macro_step:03d}.json",
            frontier_record,
        )
    expected_nodes = 2 * target_step - len(topology.active_nodes)
    if created_nodes != expected_nodes:
        raise RuntimeError("dense hierarchy did not create the binary-counter node count")
    summary = {
        "config_hash": config.config_hash,
        "all_node_checkpoint_sha256": dict(sorted(node_hashes.items())),
        "created_node_count": created_nodes,
        "final_active_node_count": len(topology.active_nodes),
        "final_frontier_sha256": file_sha256(
            directory / "frontiers" / f"step-{target_step:03d}.json"
        ),
        "final_macro_step": target_step,
        "node_example_updates": example_updates,
        "max_nodes_per_level": max_nodes_per_level,
        "run_seed": seed,
        "status": (
            "complete"
            if target_step == config.benchmark.macro_steps
            else "prefix_complete"
        ),
    }
    if _uses_sample_multiplier(config):
        summary.update({
            "schema_version": "vamp-logt-dense-hierarchy-seed-v2",
            "training_sample_multiplier": training_sample_multiplier,
        })
    else:
        summary["schema_version"] = "vamp-logt-dense-hierarchy-seed-v1"
    target_summary_path = (
        summary_path
        if target_step == config.benchmark.macro_steps
        else directory / f"prefix-step-{target_step:03d}-summary.json"
    )
    publish_immutable_json(target_summary_path, summary)
    return summary


def _fit_or_load_node(
    config: VampLogTDenseConfig,
    directory: Path,
    seed: int,
    node: TemporalNode,
    archive: ExampleBatch,
    base: DenseMlpState,
    device: torch.device,
    training_sample_multiplier: int,
) -> tuple[str, int]:
    path = directory / "nodes" / node.node_id / "delta.pt"
    sample_multiplier_study = _uses_sample_multiplier(config)
    node_seed = named_seed(
        seed,
        "capacity-node",
        node.level,
        node.first_block,
        node.last_block,
    ) if sample_multiplier_study else named_seed(seed, "node", node.node_id)
    schema_version = (
        "vamp-logt-dense-node-v2"
        if sample_multiplier_study
        else "vamp-logt-dense-node-v1"
    )
    if path.is_file():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (
            payload.get("schema_version") != schema_version
            or payload.get("config_hash") != config.config_hash
            or payload.get("node_id") != node.node_id
            or int(payload.get("seed", -1)) != node_seed
            or (
                sample_multiplier_study
                and int(payload.get("training_sample_multiplier", -1))
                != training_sample_multiplier
            )
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
    payload = {
            "config_hash": config.config_hash,
            "delta_parameters": delta.tensors,
            "example_updates": result.training_example_presentations,
            "node": _node_record(node),
            "node_id": node.node_id,
            "schema_version": schema_version,
            "seed": node_seed,
        }
    if sample_multiplier_study:
        payload["training_sample_multiplier"] = training_sample_multiplier
    atomic_torch_save(path, payload)
    return file_sha256(path), result.training_example_presentations


def _uses_sample_multiplier(config: VampLogTDenseConfig) -> bool:
    """Return whether sample multiplier is an authenticated coordinate."""
    return config.protocol_revision in {
        "dense-full-model-v5-scaling-capacity",
        "dense-full-model-v6-sample-calibrated",
    }


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


def _node_slots(
    nodes: tuple[TemporalNode, ...],
    maximum_levels: int,
    max_nodes_per_level: int,
) -> PMap[str, int]:
    if (
        maximum_levels < 1
        or max_nodes_per_level < 1
        or any(node.level >= maximum_levels for node in nodes)
    ):
        raise ValueError("dense frontier exceeds the configured slot geometry")
    by_level = {
        level: tuple(sorted(
            (node for node in nodes if node.level == level),
            key=lambda node: node.first_block,
        ))
        for level in {node.level for node in nodes}
    }
    if any(len(group) > max_nodes_per_level for group in by_level.values()):
        raise ValueError("dense frontier exceeds its per-level node capacity")
    return pmap({
        node.node_id: rank * maximum_levels + level
        for level, group in by_level.items()
        for rank, node in enumerate(group)
    })


__all__ = [
    "DenseFrontier",
    "DenseObservations",
    "build_base_observations",
    "build_dense_observations",
    "build_hierarchy_tape",
    "load_frontier",
]

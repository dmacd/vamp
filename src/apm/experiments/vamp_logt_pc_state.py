"""Immutable PC bank metadata, exact work counters, and safe retirement."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path

from pyrsistent import PMap, pmap

from apm.continual.artifacts import atomic_write, canonical_json_bytes, load_canonical_json, require_sha256
from apm.continual.logt_evidence_bank import LogTState, TemporalNode, evidence_update_bound


@dataclass(frozen=True, slots=True)
class PcWorkCounters:
    """Exact logical work for fixed-cost PC fitting, settling, and scoring."""

    pc_leaf_example_presentations: int = 0
    pc_merge_example_presentations: int = 0
    classifier_example_presentations: int = 0
    pc_inference_state_updates: int = 0
    pc_route_model_evals: int = 0
    pc_laplace_hessian_evals: int = 0
    pc_importance_audit_samples: int = 0
    active_pc_models: int = 0
    pc_gauss_newton_matrix_evals: int = 0
    pc_gauss_newton_cholesky_solves: int = 0
    pc_exact_hessian_evals: int = 0
    pc_negative_direction_probes: int = 0

    def __post_init__(self) -> None:
        if any(type(value) is not int or value < 0 for value in asdict(self).values()):
            raise ValueError("PC work counters must be nonnegative integers")

    @property
    def pc_example_presentations(self) -> int:
        """Return all leaf and merge density presentations."""
        return self.pc_leaf_example_presentations + self.pc_merge_example_presentations

    def with_fit(
        self,
        density_presentations: int,
        classifier_presentations: int,
        classifier_settling_examples: int,
        *,
        merge: bool,
        infer_steps: int,
    ) -> "PcWorkCounters":
        """Return counters after one completed node/replica fit."""
        if (
            density_presentations < 1
            or classifier_presentations < 1
            or classifier_settling_examples < 1
            or infer_steps < 1
        ):
            raise ValueError("completed PC fits require positive fixed work")
        return PcWorkCounters(
            pc_leaf_example_presentations=self.pc_leaf_example_presentations
            + (0 if merge else density_presentations),
            pc_merge_example_presentations=self.pc_merge_example_presentations
            + (density_presentations if merge else 0),
            classifier_example_presentations=self.classifier_example_presentations
            + classifier_presentations,
            pc_inference_state_updates=self.pc_inference_state_updates
            + (density_presentations + classifier_settling_examples) * infer_steps,
            pc_route_model_evals=self.pc_route_model_evals,
            pc_laplace_hessian_evals=self.pc_laplace_hessian_evals,
            pc_importance_audit_samples=self.pc_importance_audit_samples,
            active_pc_models=self.active_pc_models,
            pc_gauss_newton_matrix_evals=self.pc_gauss_newton_matrix_evals,
            pc_gauss_newton_cholesky_solves=self.pc_gauss_newton_cholesky_solves,
            pc_exact_hessian_evals=self.pc_exact_hessian_evals,
            pc_negative_direction_probes=self.pc_negative_direction_probes,
        )

    def with_scoring(
        self,
        examples: int,
        models: int,
        infer_steps: int,
        *,
        hessians: bool,
        importance_samples: int = 0,
        active_models: int | None = None,
        settle_passes: int = 2,
    ) -> "PcWorkCounters":
        """Return counters after exhaustive image-only scoring."""
        if (
            examples < 0
            or models < 0
            or infer_steps < 1
            or importance_samples < 0
            or settle_passes < 1
        ):
            raise ValueError("invalid PC scoring work")
        evaluations = examples * models
        return PcWorkCounters(
            pc_leaf_example_presentations=self.pc_leaf_example_presentations,
            pc_merge_example_presentations=self.pc_merge_example_presentations,
            classifier_example_presentations=self.classifier_example_presentations,
            pc_inference_state_updates=self.pc_inference_state_updates
            + evaluations * infer_steps * settle_passes,
            pc_route_model_evals=self.pc_route_model_evals + evaluations,
            pc_laplace_hessian_evals=self.pc_laplace_hessian_evals
            + (evaluations if hessians else 0),
            pc_importance_audit_samples=self.pc_importance_audit_samples
            + importance_samples,
            active_pc_models=self.active_pc_models if active_models is None else active_models,
            pc_gauss_newton_matrix_evals=self.pc_gauss_newton_matrix_evals,
            pc_gauss_newton_cholesky_solves=self.pc_gauss_newton_cholesky_solves,
            pc_exact_hessian_evals=self.pc_exact_hessian_evals,
            pc_negative_direction_probes=self.pc_negative_direction_probes,
        )

    def with_gauss_newton_scoring(
        self,
        examples: int,
        models: int,
        infer_steps: int,
        *,
        negative_hessian_states: int,
        direction_epsilon_count: int,
        active_models: int | None = None,
        settle_passes: int = 1,
    ) -> "PcWorkCounters":
        """Return counters after paired GN and exact-H scoring at one state."""
        if (
            examples < 0
            or models < 0
            or infer_steps < 1
            or negative_hessian_states < 0
            or direction_epsilon_count < 1
            or settle_passes < 1
        ):
            raise ValueError("invalid Gauss-Newton scoring work")
        evaluations = examples * models
        if negative_hessian_states > evaluations:
            raise ValueError("negative-Hessian state count exceeds scored states")
        return PcWorkCounters(
            pc_leaf_example_presentations=self.pc_leaf_example_presentations,
            pc_merge_example_presentations=self.pc_merge_example_presentations,
            classifier_example_presentations=self.classifier_example_presentations,
            pc_inference_state_updates=self.pc_inference_state_updates
            + evaluations * infer_steps * settle_passes,
            pc_route_model_evals=self.pc_route_model_evals + evaluations,
            pc_laplace_hessian_evals=self.pc_laplace_hessian_evals,
            pc_importance_audit_samples=self.pc_importance_audit_samples,
            active_pc_models=self.active_pc_models if active_models is None else active_models,
            pc_gauss_newton_matrix_evals=self.pc_gauss_newton_matrix_evals + evaluations,
            pc_gauss_newton_cholesky_solves=self.pc_gauss_newton_cholesky_solves
            + evaluations,
            pc_exact_hessian_evals=self.pc_exact_hessian_evals + evaluations,
            pc_negative_direction_probes=self.pc_negative_direction_probes
            + 2 * direction_epsilon_count * negative_hessian_states,
        )


def require_pc_work_bound(
    counters: PcWorkCounters,
    processed_blocks: int,
    block_size: int,
    density_epochs: int,
    classifier_epochs: int,
    replicas: int,
) -> None:
    """Assert fixed-multiple LogT bounds for both density and head training."""
    if replicas < 1 or classifier_epochs < 1:
        raise ValueError("PC work bounds require positive fixed replicas and epochs")
    density_ceiling = replicas * evidence_update_bound(processed_blocks, block_size, density_epochs)
    classifier_ceiling = replicas * evidence_update_bound(processed_blocks, block_size, classifier_epochs)
    if counters.pc_example_presentations > density_ceiling:
        raise RuntimeError(
            f"PC density work {counters.pc_example_presentations} exceeded {density_ceiling}"
        )
    if counters.classifier_example_presentations > classifier_ceiling:
        raise RuntimeError(
            f"PC classifier work {counters.classifier_example_presentations} exceeded {classifier_ceiling}"
        )


@dataclass(frozen=True, slots=True)
class ActivePcBank:
    """One live topology with authenticated on-disk model replicas."""

    topology: LogTState
    replicas_by_node: PMap[str, tuple[int, ...]]
    counters: PcWorkCounters

    def __post_init__(self) -> None:
        active = {node.node_id for node in self.topology.active_nodes}
        if set(self.replicas_by_node) != active:
            raise ValueError("active PC replicas do not exactly match the LogT frontier")
        replica_counts = {len(seeds) for seeds in self.replicas_by_node.values()}
        if any(tuple(sorted(set(seeds))) != seeds for seeds in self.replicas_by_node.values()):
            raise ValueError("PC replica seeds must be unique and sorted")
        expected_models = len(active) * (next(iter(replica_counts)) if replica_counts else 0)
        if len(replica_counts) > 1 or self.counters.active_pc_models != expected_models:
            raise ValueError("active PC model gauge differs from bank contents")


def save_bank_checkpoint(path: Path, bank: ActivePcBank) -> None:
    """Atomically replace the latest bank boundary before child retirement."""
    record = {
        "counters": asdict(bank.counters),
        "replicas_by_node": {node: list(seeds) for node, seeds in sorted(bank.replicas_by_node.items())},
        "schema_version": "vamp-logt-pc-bank-v1",
        "topology": {
            "block_size": bank.topology.block_size,
            "processed_blocks": bank.topology.processed_blocks,
            "active_nodes": [_node_record(node) for node in bank.topology.active_nodes],
        },
    }
    atomic_write(path, canonical_json_bytes(record))


def load_bank_checkpoint(path: Path) -> ActivePcBank:
    """Load and validate one exact PC bank boundary."""
    record = load_canonical_json(path)
    topology_record = record.get("topology")
    replicas = record.get("replicas_by_node")
    counters = record.get("counters")
    if (
        record.get("schema_version") != "vamp-logt-pc-bank-v1"
        or not isinstance(topology_record, dict)
        or not isinstance(replicas, dict)
        or not isinstance(counters, dict)
    ):
        raise ValueError("PC bank checkpoint is malformed")
    nodes = tuple(_node_from_record(value) for value in topology_record["active_nodes"])
    topology = LogTState(
        int(topology_record["block_size"]),
        int(topology_record["processed_blocks"]),
        pmap({node.level: node for node in nodes}),
    )
    return ActivePcBank(
        topology,
        pmap({str(node): tuple(int(seed) for seed in seeds) for node, seeds in replicas.items()}),
        PcWorkCounters(**{name: int(value) for name, value in counters.items()}),
    )


def retire_inactive_models(models_root: Path, active_node_ids: set[str]) -> tuple[str, ...]:
    """Delete only complete content-addressed node directories outside the bank."""
    if not models_root.is_dir():
        return ()
    for node_id in active_node_ids:
        require_sha256(node_id, "active PC node ID")
    retired: list[str] = []
    for directory in sorted(path for path in models_root.iterdir() if path.is_dir()):
        require_sha256(directory.name, "stored PC node ID")
        if directory.name in active_node_ids:
            continue
        for file_path in sorted(path for path in directory.rglob("*") if path.is_file()):
            file_path.unlink()
        for child in sorted(
            (path for path in directory.rglob("*") if path.is_dir()),
            key=lambda value: len(value.parts),
            reverse=True,
        ):
            child.rmdir()
        directory.rmdir()
        retired.append(directory.name)
    return tuple(retired)


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
        raise ValueError("PC bank node record is malformed")
    return TemporalNode(
        str(value["node_id"]),
        int(value["level"]),
        int(value["first_block"]),
        int(value["last_block"]),
        tuple(int(item) for item in value["example_ids"]),
        tuple(str(item) for item in value["parent_node_ids"]),
    )


__all__ = [
    "ActivePcBank",
    "PcWorkCounters",
    "load_bank_checkpoint",
    "require_pc_work_bound",
    "retire_inactive_models",
    "save_bank_checkpoint",
]

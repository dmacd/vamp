"""Run matched replay-capacity sweeps for affine-frontier and rank-80 models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import shutil
import tempfile
import time

import torch

from apm.continual.artifacts import (
    ChainedJsonlLedger,
    atomic_write,
    canonical_json_bytes,
    file_sha256,
    load_canonical_json,
    publish_immutable_bytes,
    publish_immutable_json,
    record_sha256,
    require_sha256,
)
from apm.continual.vision.imagenetr.artifacts import (
    publish_artifact_directory,
    validate_artifact_directory,
)
from apm.continual.vision.imagenetr.data import ImageRecord
from apm.continual.vision.imagenetr.frontier_adaptation_training import (
    HISTORY_FORMAT,
    AdaptationFit,
    fit_adaptation_cell,
)
from apm.continual.vision.imagenetr.frontier_adaptation_workflow import (
    _frontier,
    _training_rows,
)
from apm.continual.vision.imagenetr.frontier_architecture_replay_config import (
    DEFAULT_FRONTIER_ARCHITECTURE_REPLAY_CONFIG,
    FrontierArchitectureReplayConfig,
    load_frontier_architecture_replay_config,
)
from apm.continual.vision.imagenetr.frontier_linear_preclassifier_workflow import (
    FrontierLinearPreclassifierBootstrap,
    _material_paths as _linear_material_paths,
    _new_model as _new_linear_model,
    _publish_model as _publish_linear_model,
    _validated_control_result as _validated_linear_result,
    bootstrap_frontier_linear_preclassifier,
)
from apm.continual.vision.imagenetr.frontier_rank_matched_workflow import (
    FrontierRankMatchedBootstrap,
    _material_paths as _rank80_material_paths,
    _new_model as _new_rank80_model,
    _validated_control_result as _validated_rank80_result,
    bootstrap_frontier_rank_matched,
)
from apm.continual.vision.imagenetr.heads import save_classifier
from apm.continual.vision.imagenetr.integrator_hierarchy import HierarchyBuildResult
from apm.continual.vision.imagenetr.integrator_observations import BehaviorNode
from apm.continual.vision.imagenetr.lora import adapter_factors, save_adapter
from apm.continual.vision.imagenetr.macro_convergence_training import (
    JOINT_HISTORY_FORMAT,
    JointConvergenceFit,
    fit_clean_joint_control,
)
from apm.continual.vision.imagenetr.manifests import installed_environment_manifest
from apm.continual.vision.imagenetr.model import require_trainable_boundary
from apm.continual.vision.imagenetr.promoted_integrator_workflow import (
    PROMOTED_PACKAGES,
    _build,
)
from apm.continual.vision.imagenetr.protocol import material_tree_manifest


SWEEP_RESULT = Path("evaluations/frontier_architecture_replay_sweep.json")
FAMILIES = ("single_affine", "joint_iid_rank80")


@dataclass(frozen=True, slots=True)
class ArchitectureReplayCell:
    """One architecture and one frozen historical replay prefix."""

    family: str
    historical_capacity: int
    seed: int

    def __post_init__(self) -> None:
        if self.family not in FAMILIES or self.historical_capacity not in {
            1_024,
            2_048,
            4_096,
            8_192,
        } or self.seed < 0:
            raise ValueError("architecture replay cell is outside the frozen matrix")

    @property
    def adapt_lora(self) -> bool:
        """Expose the interface required by frontier adaptation training."""
        return True

    @property
    def condition(self) -> str:
        """Return the stable cell artifact and progress label."""
        return f"{self.family}_h{self.historical_capacity}"

    def as_record(self) -> dict[str, object]:
        """Return canonical cell fields."""
        return {
            "adapt_lora": self.adapt_lora,
            "family": self.family,
            "historical_capacity": self.historical_capacity,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class ReplayPopulation:
    """Exact current-plus-history rows for one nested H value."""

    historical_capacity: int
    rows: tuple[ImageRecord, ...]
    historical_image_ids_hash: str
    image_ids_hash: str

    def as_record(self) -> dict[str, object]:
        """Return identity and size fields without duplicating the parent manifest."""
        return {
            "historical_capacity": self.historical_capacity,
            "historical_image_ids_hash": self.historical_image_ids_hash,
            "image_ids_hash": self.image_ids_hash,
            "training_examples": len(self.rows),
        }


@dataclass(frozen=True, slots=True)
class FrontierArchitectureReplayProtocol:
    """Content identity for both matched replay-capacity sweeps."""

    parent_run_hash: str
    parent_result_hash: str
    parent_replay_sha256: str
    parent_replay_content_hash: str
    linear_full_result_sha256: str
    linear_full_result_hash: str
    rank80_full_result_sha256: str
    rank80_full_result_hash: str
    dataset_manifest_hash: str
    model_manifest_hash: str
    split_hash: str
    population_manifest_hash: str
    validation_image_ids_hash: str
    config_hash: str
    code_manifest_hash: str
    environment_manifest_hash: str
    schema_version: str = "imagenetr50-frontier-architecture-replay-protocol-v1"

    def __post_init__(self) -> None:
        for label, identity in (
            ("parent run", self.parent_run_hash),
            ("parent result", self.parent_result_hash),
            ("parent replay file", self.parent_replay_sha256),
            ("parent replay", self.parent_replay_content_hash),
            ("linear full result file", self.linear_full_result_sha256),
            ("linear full result", self.linear_full_result_hash),
            ("rank-80 full result file", self.rank80_full_result_sha256),
            ("rank-80 full result", self.rank80_full_result_hash),
            ("dataset", self.dataset_manifest_hash),
            ("model", self.model_manifest_hash),
            ("split", self.split_hash),
            ("population manifest", self.population_manifest_hash),
            ("validation identities", self.validation_image_ids_hash),
            ("configuration", self.config_hash),
            ("code", self.code_manifest_hash),
            ("environment", self.environment_manifest_hash),
        ):
            require_sha256(identity, label)
        if (
            self.schema_version
            != "imagenetr50-frontier-architecture-replay-protocol-v1"
        ):
            raise ValueError("architecture replay protocol schema changed")

    @property
    def content_hash(self) -> str:
        """Return the immutable sweep namespace."""
        return record_sha256(self.as_record(False))

    def as_record(self, include_hash: bool = True) -> dict[str, object]:
        """Return canonical protocol fields with an optional derived hash."""
        core = asdict(self)
        return {**core, "content_hash": self.content_hash} if include_hash else core


@dataclass(frozen=True, slots=True)
class FrontierArchitectureReplayBootstrap:
    """Authenticated sources, populations, protocol, and output paths."""

    project_root: Path
    config: FrontierArchitectureReplayConfig
    linear: FrontierLinearPreclassifierBootstrap
    rank80: FrontierRankMatchedBootstrap
    linear_full_result: dict[str, object]
    rank80_full_result: dict[str, object]
    replay: dict[str, object]
    populations: tuple[ReplayPopulation, ...]
    validation_rows: tuple[ImageRecord, ...]
    protocol: FrontierArchitectureReplayProtocol
    control_root: Path


def _material_paths(
    project_root: Path,
    config_path: Path,
    linear_config_path: Path,
    linear_parent_config_path: Path,
    rank80_config_path: Path,
) -> tuple[Path, ...]:
    package = project_root / "src/apm/continual/vision/imagenetr"
    additions = (
        config_path,
        project_root
        / "docs/imagenetr50_frontier_architecture_replay_sweep_protocol.md",
        project_root
        / "scripts/vision/imagenetr/run_frontier_architecture_replay_sweep_local.sh",
        package / "frontier_architecture_replay_config.py",
        package / "frontier_architecture_replay_workflow.py",
    )
    return tuple(
        dict.fromkeys(
            (
                *_linear_material_paths(
                    project_root, linear_config_path, linear_parent_config_path
                ),
                *_rank80_material_paths(project_root, rank80_config_path),
                *additions,
            )
        )
    )


def _validated_replay(
    path: Path, config: FrontierArchitectureReplayConfig
) -> dict[str, object]:
    replay = load_canonical_json(path)
    core = {key: value for key, value in replay.items() if key != "content_hash"}
    if (
        file_sha256(path) != config.parent_replay_sha256
        or replay.get("content_hash") != config.parent_replay_content_hash
        or replay.get("content_hash") != record_sha256(core)
        or replay.get("schema_version")
        != "imagenetr50-frontier-adaptation-replay-v1"
        or replay.get("sampler") != "uniform_hash_order_without_replacement"
    ):
        raise ValueError("parent replay manifest does not authenticate")
    return replay


def _replay_populations(
    fit_rows: Sequence[ImageRecord],
    replay: Mapping[str, object],
    historical_capacities: Sequence[int],
    current_task_examples: int,
    available_historical_examples: int,
) -> tuple[ReplayPopulation, ...]:
    rows_by_id = {row.image_id: row for row in fit_rows}
    current_ids = tuple(str(value) for value in replay["current_image_ids"])
    historical_ids = tuple(
        str(value) for value in replay["nested_historical_image_ids"]
    )
    if (
        len(rows_by_id) != len(fit_rows)
        or len(current_ids) != current_task_examples
        or len(historical_ids) != available_historical_examples
        or set(current_ids) & set(historical_ids)
        or set(current_ids) | set(historical_ids) != set(rows_by_id)
    ):
        raise ValueError("parent replay identities differ from the clean fit rows")
    try:
        current_rows = tuple(rows_by_id[image_id] for image_id in current_ids)
        historical_rows = tuple(rows_by_id[image_id] for image_id in historical_ids)
    except KeyError as error:
        raise ValueError("parent replay refers to an unknown fit identity") from error
    entries = {
        int(dict(entry)["historical_capacity"]): dict(entry)
        for entry in replay["entries"]
    }
    populations = tuple(
        ReplayPopulation(
            capacity,
            rows,
            record_sha256(list(historical_ids[:capacity])),
            record_sha256([row.image_id for row in rows]),
        )
        for capacity in historical_capacities
        for rows in (_training_rows(current_rows, historical_rows, capacity),)
    )
    if any(
        population.historical_capacity not in entries
        or entries[population.historical_capacity].get("training_examples")
        != len(population.rows)
        or entries[population.historical_capacity].get(
            "historical_image_ids_hash"
        )
        != population.historical_image_ids_hash
        or entries[population.historical_capacity].get("image_ids_hash")
        != population.image_ids_hash
        for population in populations
    ):
        raise ValueError("reconstructed sweep populations differ from parent entries")
    return populations


def bootstrap_frontier_architecture_replay(
    config_path: str | Path = DEFAULT_FRONTIER_ARCHITECTURE_REPLAY_CONFIG,
) -> FrontierArchitectureReplayBootstrap:
    """Authenticate both full-fit sources and reconstruct every exact H cell."""
    resolved = Path(config_path).resolve()
    project_root = resolved.parents[3]
    config = load_frontier_architecture_replay_config(resolved)
    if (
        file_sha256(config.linear_config) != config.linear_config_sha256
        or file_sha256(config.rank80_config) != config.rank80_config_sha256
    ):
        raise ValueError("source control configuration bytes changed")
    linear = bootstrap_frontier_linear_preclassifier(config.linear_config)
    rank80 = bootstrap_frontier_rank_matched(config.rank80_config)
    parent_run = config.parent_artifact_root / "runs" / config.parent_run_hash
    linear_result_path = parent_run / "evaluations/frontier_linear_preclassifier.json"
    rank80_result_path = parent_run / "evaluations/joint_iid_lora_r80.json"
    replay_path = parent_run / "protocol/replay_populations.json"
    linear_result = _validated_linear_result(linear_result_path, linear)
    rank80_result = _validated_rank80_result(rank80_result_path, rank80)
    replay = _validated_replay(replay_path, config)
    if (
        linear.parent.run != parent_run
        or rank80.parent.run != parent_run
        or file_sha256(linear_result_path) != config.linear_full_result_sha256
        or linear_result["content_hash"] != config.linear_full_result_content_hash
        or file_sha256(rank80_result_path) != config.rank80_full_result_sha256
        or rank80_result["content_hash"] != config.rank80_full_result_content_hash
        or tuple(row.image_id for row in linear.fit_rows)
        != tuple(row.image_id for row in rank80.fit_rows)
        or tuple(row.image_id for row in linear.validation_rows)
        != tuple(row.image_id for row in rank80.validation_rows)
        or linear.parent_result["content_hash"]
        != rank80.parent_result["content_hash"]
    ):
        raise ValueError("architecture sweep sources do not share one parent")
    populations = _replay_populations(
        linear.fit_rows,
        replay,
        config.historical_capacities,
        config.current_task_examples,
        config.available_historical_examples,
    )
    population_core: dict[str, object] = {
        "current_image_ids_hash": record_sha256(replay["current_image_ids"]),
        "entries": [population.as_record() for population in populations],
        "parent_replay_content_hash": replay["content_hash"],
        "parent_replay_sha256": file_sha256(replay_path),
        "schema_version": "imagenetr50-frontier-architecture-replay-populations-v1",
    }
    population_manifest = {
        **population_core,
        "content_hash": record_sha256(population_core),
    }
    code = material_tree_manifest(
        _material_paths(
            project_root,
            resolved,
            config.linear_config,
            linear.config.parent_config,
            config.rank80_config,
        )
    )
    environment = installed_environment_manifest(PROMOTED_PACKAGES)
    missing = tuple(
        str(row["name"])
        for row in environment["packages"]
        if row["version"] == "MISSING"
    )
    if missing:
        raise RuntimeError(f"isolated vision environment is incomplete: {missing}")
    protocol = FrontierArchitectureReplayProtocol(
        config.parent_run_hash,
        str(linear.parent_result["content_hash"]),
        config.parent_replay_sha256,
        config.parent_replay_content_hash,
        config.linear_full_result_sha256,
        config.linear_full_result_content_hash,
        config.rank80_full_result_sha256,
        config.rank80_full_result_content_hash,
        linear.parent.protocol.dataset_manifest_hash,
        linear.parent.protocol.model_manifest_hash,
        linear.parent.protocol.split_hash,
        str(population_manifest["content_hash"]),
        record_sha256([row.image_id for row in linear.validation_rows]),
        config.config_hash,
        str(code["content_hash"]),
        str(environment["content_hash"]),
    )
    control_root = (
        parent_run
        / "controls/frontier_architecture_replay_sweep"
        / protocol.content_hash
    )
    for relative in ("cells", "state", "work"):
        (control_root / relative).mkdir(parents=True, exist_ok=True)
    for filename, record in (
        ("protocol.json", protocol.as_record()),
        ("code_manifest.json", code),
        ("environment_manifest.json", environment),
        ("population_manifest.json", population_manifest),
    ):
        publish_immutable_json(control_root / filename, record)
    publish_immutable_bytes(
        control_root / "config_resolved.json",
        canonical_json_bytes(config.as_record()),
    )
    return FrontierArchitectureReplayBootstrap(
        project_root,
        config,
        linear,
        rank80,
        linear_result,
        rank80_result,
        replay,
        populations,
        linear.validation_rows,
        protocol,
        control_root,
    )


def _phase(
    bootstrap: FrontierArchitectureReplayBootstrap,
    number: int,
    total: int,
    message: str,
) -> None:
    """Print and persist one human-readable workflow phase."""
    print(f"[phase {number}/{total}] {message}", flush=True)
    ChainedJsonlLedger(
        bootstrap.control_root / "workflow_events.jsonl",
        "imagenetr50-frontier-architecture-replay-event-v1",
    ).append(
        {
            "message": message,
            "phase": number,
            "schema_version": "imagenetr50-frontier-architecture-replay-event-v1",
            "wall_time_unix": time.time(),
        }
    )


def _cell_root(
    bootstrap: FrontierArchitectureReplayBootstrap, cell: ArchitectureReplayCell
) -> Path:
    return (
        bootstrap.control_root
        / "cells"
        / cell.family
        / f"h{cell.historical_capacity:05d}"
    )


def _preflight(
    bootstrap: FrontierArchitectureReplayBootstrap,
    hierarchy: HierarchyBuildResult,
    frontier_hash: str,
) -> dict[str, object]:
    """Authenticate prior model checks and the new eight-cell data boundary."""
    target = bootstrap.control_root / "preflight.json"
    if target.is_file():
        record = load_canonical_json(target)
        core = {key: value for key, value in record.items() if key != "content_hash"}
        if (
            record.get("content_hash") != record_sha256(core)
            or record.get("protocol_hash") != bootstrap.protocol.content_hash
            or record.get("test_evaluations") != 0
        ):
            raise ValueError("architecture replay preflight does not authenticate")
        return record
    linear_preflight = dict(bootstrap.linear_full_result["preflight"])
    rank80_preflight = dict(bootstrap.rank80_full_result["preflight"])
    integrator = bootstrap.linear.parent.source.source.integrator
    validation_ids = frozenset(row.image_id for row in bootstrap.validation_rows)
    test_ids = frozenset(
        row.image_id for row in integrator.manifest.images if row.split == "test"
    )
    core: dict[str, object] = {
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "cells": len(FAMILIES) * len(bootstrap.populations),
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0),
        "fit_examples": [len(population.rows) for population in bootstrap.populations],
        "fit_validation_overlaps": [
            len(
                frozenset(row.image_id for row in population.rows)
                & validation_ids
            )
            for population in bootstrap.populations
        ],
        "frontier_hash": frontier_hash,
        "linear_initial_union_max_logit_error": linear_preflight[
            "initial_union_max_logit_error"
        ],
        "linear_preflight_hash": linear_preflight["content_hash"],
        "linear_trainable_parameters": dict(
            bootstrap.linear_full_result["architecture"]
        )["trainable_parameters"],
        "protocol_hash": bootstrap.protocol.content_hash,
        "rank80_preflight_hash": rank80_preflight["content_hash"],
        "rank80_trainable_parameters": dict(
            bootstrap.rank80_full_result["architecture"]
        )["trainable_parameters"],
        "rank80_zero_lora_max_logit_error": rank80_preflight[
            "zero_lora_max_logit_error_vs_rank16"
        ],
        "schema_version": "imagenetr50-frontier-architecture-replay-preflight-v1",
        "source_hierarchy_leaf_optimizer_steps": hierarchy.work.leaf_optimizer_steps,
        "source_hierarchy_parent_optimizer_steps": (
            hierarchy.work.parent_optimizer_steps
        ),
        "test_evaluations": 0,
        "test_fit_overlaps": [
            len(frozenset(row.image_id for row in population.rows) & test_ids)
            for population in bootstrap.populations
        ],
        "test_validation_overlap": len(test_ids & validation_ids),
        "validation_examples": len(validation_ids),
    }
    if (
        not core["cuda_available"]
        or not core["bf16_supported"]
        or core["cells"] != 8
        or core["fit_examples"] != [1_391, 2_415, 4_463, 8_559]
        or any(core["fit_validation_overlaps"])
        or any(core["test_fit_overlaps"])
        or core["test_validation_overlap"] != 0
        or core["validation_examples"] != bootstrap.config.validation_examples
        or core["linear_initial_union_max_logit_error"] > 1e-4
        or core["linear_trainable_parameters"] != 7_403_720
        or core["rank80_zero_lora_max_logit_error"] != 0.0
        or core["rank80_trainable_parameters"] != 6_730_876
        or core["source_hierarchy_leaf_optimizer_steps"] != 0
        or core["source_hierarchy_parent_optimizer_steps"] != 0
    ):
        raise RuntimeError(f"architecture replay preflight failed: {core}")
    record = {**core, "content_hash": record_sha256(core)}
    publish_immutable_json(target, record)
    return record


def _cell_seal(
    bootstrap: FrontierArchitectureReplayBootstrap,
    cell: ArchitectureReplayCell,
    population: ReplayPopulation,
    hierarchy: HierarchyBuildResult,
) -> dict[str, object]:
    """Prove population isolation and unchanged source artifacts for one cell."""
    integrator = bootstrap.linear.parent.source.source.integrator
    fit_ids = frozenset(row.image_id for row in population.rows)
    validation_ids = frozenset(row.image_id for row in bootstrap.validation_rows)
    test_ids = frozenset(
        row.image_id for row in integrator.manifest.images if row.split == "test"
    )
    parent_run = bootstrap.linear.parent.run
    source_hashes = {
        "linear_full_result": file_sha256(
            parent_run / "evaluations/frontier_linear_preclassifier.json"
        ),
        "parent_replay": file_sha256(
            parent_run / "protocol/replay_populations.json"
        ),
        "rank80_full_result": file_sha256(
            parent_run / "evaluations/joint_iid_lora_r80.json"
        ),
    }
    expected = {
        "linear_full_result": bootstrap.config.linear_full_result_sha256,
        "parent_replay": bootstrap.config.parent_replay_sha256,
        "rank80_full_result": bootstrap.config.rank80_full_result_sha256,
    }
    core: dict[str, object] = {
        "cell": cell.as_record(),
        "fit_examples": len(fit_ids),
        "fit_image_ids_hash": population.image_ids_hash,
        "fit_validation_overlap": len(fit_ids & validation_ids),
        "source_files_unchanged": source_hashes == expected,
        "source_hashes_after": source_hashes,
        "source_hierarchy_leaf_optimizer_steps": hierarchy.work.leaf_optimizer_steps,
        "source_hierarchy_parent_optimizer_steps": (
            hierarchy.work.parent_optimizer_steps
        ),
        "test_evaluations": 0,
        "test_fit_overlap": len(test_ids & fit_ids),
        "test_validation_overlap": len(test_ids & validation_ids),
        "training_derived_only": all(
            row.split == "train"
            for row in (*population.rows, *bootstrap.validation_rows)
        ),
        "validation_examples": len(validation_ids),
        "schema_version": "imagenetr50-frontier-architecture-replay-cell-seal-v1",
    }
    if (
        core["fit_validation_overlap"] != 0
        or core["test_fit_overlap"] != 0
        or core["test_validation_overlap"] != 0
        or not core["source_files_unchanged"]
        or not core["training_derived_only"]
        or core["source_hierarchy_leaf_optimizer_steps"] != 0
        or core["source_hierarchy_parent_optimizer_steps"] != 0
    ):
        raise RuntimeError(f"architecture replay cell seal failed: {cell.condition}")
    record = {**core, "content_hash": record_sha256(core)}
    publish_immutable_json(_cell_root(bootstrap, cell) / "training_seal.json", record)
    return record


def _validated_cell_result(
    bootstrap: FrontierArchitectureReplayBootstrap,
    cell: ArchitectureReplayCell,
    population: ReplayPopulation,
) -> dict[str, object]:
    path = _cell_root(bootstrap, cell) / "result.json"
    record = load_canonical_json(path)
    core = {key: value for key, value in record.items() if key != "content_hash"}
    history_path = (
        bootstrap.linear.parent.run / str(record.get("history", ""))
    ).resolve()
    artifact_path = (
        bootstrap.linear.parent.run / str(record.get("artifact", ""))
    ).resolve()
    expected_epochs = (
        bootstrap.config.linear_epochs
        if cell.family == "single_affine"
        else bootstrap.config.rank80_epochs
    )
    expected_history_format = (
        HISTORY_FORMAT if cell.family == "single_affine" else JOINT_HISTORY_FORMAT
    )
    if (
        record.get("schema_version")
        != "imagenetr50-frontier-architecture-replay-cell-v1"
        or record.get("content_hash") != record_sha256(core)
        or record.get("protocol_hash") != bootstrap.protocol.content_hash
        or record.get("cell") != cell.as_record()
        or record.get("population") != population.as_record()
        or record.get("test_evaluations") != 0
        or bootstrap.linear.parent.run not in history_path.parents
        or bootstrap.linear.parent.run not in artifact_path.parents
        or not history_path.is_file()
        or file_sha256(history_path) != record.get("history_sha256")
        or len(ChainedJsonlLedger(history_path, expected_history_format).rows)
        != expected_epochs
        or validate_artifact_directory(artifact_path)
        != record.get("artifact_sha256")
    ):
        raise ValueError(
            f"architecture replay cell does not authenticate: {cell.condition}"
        )
    return record


def _fit_linear_cell(
    bootstrap: FrontierArchitectureReplayBootstrap,
    cell: ArchitectureReplayCell,
    population: ReplayPopulation,
    hierarchy: HierarchyBuildResult,
    nodes: Sequence[BehaviorNode],
    slots: Sequence[int],
    frontier_hash: str,
    device: torch.device,
) -> tuple[dict[str, object], bool]:
    """Fit or authenticate one single-affine replay cell."""
    cell_root = _cell_root(bootstrap, cell)
    cell_root.mkdir(parents=True, exist_ok=True)
    (cell_root / "work").mkdir(exist_ok=True)
    if (cell_root / "result.json").is_file():
        return _validated_cell_result(bootstrap, cell, population), True
    model = _new_linear_model(bootstrap.linear, nodes, slots, device)
    history_path = cell_root / "history.jsonl"
    checkpoint_path = cell_root / "checkpoint.pt"
    job_hash = record_sha256(
        {
            "cell": cell.as_record(),
            "frontier_hash": frontier_hash,
            "population": population.as_record(),
            "protocol": bootstrap.protocol.content_hash,
            "schema_version": "imagenetr50-frontier-architecture-replay-job-v1",
            "validation_image_ids_hash": bootstrap.protocol.validation_image_ids_hash,
        }
    )
    integrator = bootstrap.linear.parent.source.source.integrator
    fit, train_metrics, validation_metrics, displacements = fit_adaptation_cell(
        model=model,
        nodes=nodes,
        cell=cell,
        prepared_root=integrator.config.data_root / "imagenet-r",
        training_rows=population.rows,
        validation_rows=bootstrap.validation_rows,
        train_transform=integrator.train_transform,
        evaluation_transform=integrator.test_transform,
        config=bootstrap.linear.config,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        job_hash=job_hash,
        device=device,
    )
    cell_bootstrap = replace(
        bootstrap.linear,
        fit_rows=population.rows,
        protocol=bootstrap.protocol,
        control_root=cell_root,
    )
    model_path, artifact_hash = _publish_linear_model(
        cell_bootstrap, model, nodes, fit, history_path, job_hash
    )
    history = tuple(
        json.loads(line)
        for line in history_path.read_text(encoding="utf-8").splitlines()
        if line
    )
    fixed = next(row for row in history if int(row["epoch"]) == 5)
    seal = _cell_seal(bootstrap, cell, population, hierarchy)
    core: dict[str, object] = {
        "architecture": dict(bootstrap.linear_full_result["architecture"]),
        "artifact": str(model_path.relative_to(bootstrap.linear.parent.run)),
        "artifact_sha256": artifact_hash,
        "cell": cell.as_record(),
        "displacements": [dict(row) for row in displacements],
        "fit": {
            **fit.as_record(),
            "fixed_epoch": 5,
            "fixed_image_presentations": int(fixed["image_presentations"]),
            "fixed_validation_accuracy": float(fixed["validation_accuracy"]),
            "fixed_validation_nll": float(fixed["validation_nll"]),
        },
        "history": str(history_path.relative_to(bootstrap.linear.parent.run)),
        "history_sha256": file_sha256(history_path),
        "population": population.as_record(),
        "protocol_hash": bootstrap.protocol.content_hash,
        "schema_version": "imagenetr50-frontier-architecture-replay-cell-v1",
        "source_full_result_hash": bootstrap.linear_full_result["content_hash"],
        "test_evaluations": 0,
        "train_metrics": train_metrics.as_record(),
        "training": dict(bootstrap.linear_full_result["training"]),
        "training_seal": seal,
        "validation_metrics": validation_metrics.as_record(),
    }
    record = {**core, "content_hash": record_sha256(core)}
    publish_immutable_json(cell_root / "result.json", record)
    checkpoint_path.unlink(missing_ok=True)
    del model
    torch.cuda.empty_cache()
    return record, False


def _publish_rank80_model(
    bootstrap: FrontierArchitectureReplayBootstrap,
    cell: ArchitectureReplayCell,
    model: torch.nn.Module,
    fit: JointConvergenceFit,
    history_path: Path,
) -> tuple[Path, str]:
    """Publish one local rank-80 model directory and its compact manifest."""
    target = _cell_root(bootstrap, cell) / "model"
    if target.is_dir():
        return target, validate_artifact_directory(target)
    work = Path(
        tempfile.mkdtemp(
            prefix="joint-r80-replay-", dir=bootstrap.control_root / "work"
        )
    )
    try:
        adapter_sha256 = save_adapter(
            work / "adapter.safetensors", adapter_factors(model)
        )
        classifier_sha256 = save_classifier(
            work / "classifier.safetensors", model.classifier.rows()
        )
        publish_immutable_json(
            work / "fit.json",
            {
                "adapter_sha256": adapter_sha256,
                "classifier_sha256": classifier_sha256,
                "fit": fit.as_record(),
                "history_sha256": file_sha256(history_path),
                "protocol_hash": bootstrap.protocol.content_hash,
                "schema_version": "imagenetr50-frontier-rank80-replay-fit-v1",
            },
        )
        artifact_hash = publish_artifact_directory(work, target)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return target, artifact_hash


def _fit_rank80_cell(
    bootstrap: FrontierArchitectureReplayBootstrap,
    cell: ArchitectureReplayCell,
    population: ReplayPopulation,
    hierarchy: HierarchyBuildResult,
    device: torch.device,
) -> tuple[dict[str, object], bool]:
    """Fit or authenticate one five-epoch rank-80 joint-IID replay cell."""
    cell_root = _cell_root(bootstrap, cell)
    cell_root.mkdir(parents=True, exist_ok=True)
    if (cell_root / "result.json").is_file():
        return _validated_cell_result(bootstrap, cell, population), True
    rank80 = bootstrap.rank80
    model = _new_rank80_model(
        rank80, rank80.config.target_rank, rank80.config.target_alpha
    ).to(device)
    require_trainable_boundary(model)
    history_path = cell_root / "history.jsonl"
    checkpoint_path = cell_root / "checkpoint.pt"
    job_hash = record_sha256(
        {
            "cell": cell.as_record(),
            "population": population.as_record(),
            "protocol": bootstrap.protocol.content_hash,
            "schema_version": "imagenetr50-frontier-architecture-replay-job-v1",
            "validation_image_ids_hash": bootstrap.protocol.validation_image_ids_hash,
        }
    )
    integrator = bootstrap.linear.parent.source.source.integrator
    fit, train_metrics, validation_metrics = fit_clean_joint_control(
        model=model,
        prepared_root=integrator.config.data_root / "imagenet-r",
        training_rows=population.rows,
        validation_rows=bootstrap.validation_rows,
        train_transform=integrator.train_transform,
        evaluation_transform=integrator.test_transform,
        config=rank80.config.training,
        training_seed=bootstrap.config.seed + 50_000,
        num_workers=rank80.config.num_workers,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        job_hash=job_hash,
        device=device,
    )
    model_path, artifact_hash = _publish_rank80_model(
        bootstrap, cell, model, fit, history_path
    )
    seal = _cell_seal(bootstrap, cell, population, hierarchy)
    core: dict[str, object] = {
        "architecture": dict(bootstrap.rank80_full_result["architecture"]),
        "artifact": str(model_path.relative_to(bootstrap.linear.parent.run)),
        "artifact_sha256": artifact_hash,
        "cell": cell.as_record(),
        "fit": fit.as_record(),
        "history": str(history_path.relative_to(bootstrap.linear.parent.run)),
        "history_sha256": file_sha256(history_path),
        "population": population.as_record(),
        "protocol_hash": bootstrap.protocol.content_hash,
        "schema_version": "imagenetr50-frontier-architecture-replay-cell-v1",
        "source_full_result_hash": bootstrap.rank80_full_result["content_hash"],
        "test_evaluations": 0,
        "train_metrics": train_metrics.as_record(),
        "training": asdict(rank80.config.training),
        "training_seal": seal,
        "validation_metrics": validation_metrics.as_record(),
    }
    record = {**core, "content_hash": record_sha256(core)}
    publish_immutable_json(cell_root / "result.json", record)
    checkpoint_path.unlink(missing_ok=True)
    del model
    torch.cuda.empty_cache()
    return record, False


def _fit_or_load_cell(
    bootstrap: FrontierArchitectureReplayBootstrap,
    cell: ArchitectureReplayCell,
    population: ReplayPopulation,
    hierarchy: HierarchyBuildResult,
    nodes: Sequence[BehaviorNode],
    slots: Sequence[int],
    frontier_hash: str,
    device: torch.device,
) -> tuple[dict[str, object], bool]:
    """Dispatch one cell to its shared architecture-specific training engine."""
    if cell.family == "single_affine":
        return _fit_linear_cell(
            bootstrap,
            cell,
            population,
            hierarchy,
            nodes,
            slots,
            frontier_hash,
            device,
        )
    return _fit_rank80_cell(bootstrap, cell, population, hierarchy, device)


def _aggregate_result(
    bootstrap: FrontierArchitectureReplayBootstrap,
    cell_results: Sequence[tuple[ArchitectureReplayCell, Mapping[str, object]]],
) -> dict[str, object]:
    """Publish and return the compact eight-cell result index."""
    parent_run = bootstrap.linear.parent.run
    indexed = tuple(
        {
            "cell": cell.as_record(),
            "result": str(
                (_cell_root(bootstrap, cell) / "result.json").relative_to(parent_run)
            ),
            "result_content_hash": result["content_hash"],
            "result_sha256": file_sha256(_cell_root(bootstrap, cell) / "result.json"),
        }
        for cell, result in cell_results
    )
    core: dict[str, object] = {
        "cells": list(indexed),
        "families": list(FAMILIES),
        "historical_capacities": list(bootstrap.config.historical_capacities),
        "parent_result_hash": bootstrap.linear.parent_result["content_hash"],
        "protocol": str(
            (bootstrap.control_root / "protocol.json").relative_to(parent_run)
        ),
        "protocol_hash": bootstrap.protocol.content_hash,
        "schema_version": "imagenetr50-frontier-architecture-replay-result-v1",
        "source_full_results": {
            "joint_iid_rank80": bootstrap.rank80_full_result["content_hash"],
            "single_affine": bootstrap.linear_full_result["content_hash"],
        },
        "test_evaluations": 0,
    }
    record = {**core, "content_hash": record_sha256(core)}
    target = parent_run / SWEEP_RESULT
    publish_immutable_json(target, record)
    return record


def run_frontier_architecture_replay_sweep(
    config_path: str | Path = DEFAULT_FRONTIER_ARCHITECTURE_REPLAY_CONFIG,
) -> Path:
    """Run or resume all eight cells and rebuild the existing stage-31 report."""
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("the architecture replay sweep requires BF16 CUDA")
    started = time.monotonic()
    bootstrap = bootstrap_frontier_architecture_replay(config_path)
    print(
        f"Temporary/resumable artifact directory: {bootstrap.control_root}",
        flush=True,
    )
    device = torch.device("cuda:0")
    from tqdm.auto import tqdm

    _phase(bootstrap, 1, 5, "Authenticate both full-fit controls and replay identities")
    hierarchy = _build(
        bootstrap.linear.parent.source.source, "fit", 50, device, progress=False
    )
    if (
        hierarchy.policy.content_hash
        != bootstrap.linear.parent.source.config.fit_hierarchy_policy_hash
        or hierarchy.work.leaf_optimizer_steps
        or hierarchy.work.parent_optimizer_steps
    ):
        raise RuntimeError("source hierarchy was not reused exactly")
    nodes, slots, frontier_hash = _frontier(hierarchy, bootstrap.config.stage)
    _preflight(bootstrap, hierarchy, frontier_hash)
    cells = tuple(
        ArchitectureReplayCell(
            family, population.historical_capacity, bootstrap.config.seed
        )
        for population in bootstrap.populations
        for family in FAMILIES
    )
    populations = {
        population.historical_capacity: population
        for population in bootstrap.populations
    }
    _phase(
        bootstrap,
        2,
        5,
        "Fit or authenticate four single-affine and four rank-80 cells",
    )
    total_presentations = sum(
        len(populations[cell.historical_capacity].rows)
        * (
            bootstrap.config.linear_epochs
            if cell.family == "single_affine"
            else bootstrap.config.rank80_epochs
        )
        for cell in cells
    )
    overall = tqdm(
        total=total_presentations,
        desc="ImageNet-R architecture replay sweep",
        unit="image-pass",
    )
    cell_results = []
    initially_reused = []
    for index, cell in enumerate(cells, start=1):
        population = populations[cell.historical_capacity]
        print(
            f"[cell {index}/{len(cells)}] {cell.condition}: "
            f"{len(population.rows):,} fit identities",
            flush=True,
        )
        result, reused = _fit_or_load_cell(
            bootstrap,
            cell,
            population,
            hierarchy,
            nodes,
            slots,
            frontier_hash,
            device,
        )
        cell_results.append((cell, result))
        initially_reused.append(reused)
        epochs = (
            bootstrap.config.linear_epochs
            if cell.family == "single_affine"
            else bootstrap.config.rank80_epochs
        )
        overall.update(len(population.rows) * epochs)
        overall.set_postfix(completed=f"{index}/{len(cells)}")
    overall.close()
    result = _aggregate_result(bootstrap, cell_results)
    _phase(
        bootstrap,
        3,
        5,
        "Authenticate all eight cells without another optimizer step",
    )
    repeated = tuple(
        _fit_or_load_cell(
            bootstrap,
            cell,
            populations[cell.historical_capacity],
            hierarchy,
            nodes,
            slots,
            frontier_hash,
            device,
        )
        for cell in cells
    )
    if any(
        not reused or repeated_result != original_result
        for (_cell, original_result), (repeated_result, reused) in zip(
            cell_results, repeated, strict=True
        )
    ):
        raise RuntimeError("architecture replay sweep did not reuse exactly")
    reuse_core: dict[str, object] = {
        "all_cells_reused": True,
        "new_optimizer_steps": 0,
        "result_hash": result["content_hash"],
        "schema_version": "imagenetr50-frontier-architecture-replay-reuse-v1",
        "source_hierarchy_leaf_optimizer_steps": hierarchy.work.leaf_optimizer_steps,
        "source_hierarchy_parent_optimizer_steps": (
            hierarchy.work.parent_optimizer_steps
        ),
    }
    publish_immutable_json(
        bootstrap.control_root / "reuse_proof.json",
        {**reuse_core, "content_hash": record_sha256(reuse_core)},
    )
    _phase(bootstrap, 4, 5, "Rebuild tables, scaling plots, and learning curves")
    from apm.continual.vision.imagenetr.frontier_adaptation_reporting import (
        write_frontier_adaptation_report,
    )

    report = write_frontier_adaptation_report(bootstrap.linear.parent.run)
    _phase(bootstrap, 5, 5, "Write the fresh-process reuse and resource record")
    elapsed = time.monotonic() - started
    atomic_write(
        bootstrap.control_root / "state/last_invocation.json",
        canonical_json_bytes(
            {
                "elapsed_seconds": elapsed,
                "initially_reused_cells": sum(initially_reused),
                "newly_completed_cells": len(cells) - sum(initially_reused),
                "result_hash": result["content_hash"],
                "schema_version": (
                    "imagenetr50-frontier-architecture-replay-invocation-v1"
                ),
            }
        ),
    )
    print(
        f"Architecture replay sweep complete in {elapsed / 60:.1f} minutes; "
        f"report updated: {report}",
        flush=True,
    )
    return report


if __name__ == "__main__":
    print(run_frontier_architecture_replay_sweep())


__all__ = [
    "ArchitectureReplayCell",
    "DEFAULT_FRONTIER_ARCHITECTURE_REPLAY_CONFIG",
    "FAMILIES",
    "FrontierArchitectureReplayBootstrap",
    "FrontierArchitectureReplayProtocol",
    "ReplayPopulation",
    "SWEEP_RESULT",
    "bootstrap_frontier_architecture_replay",
    "run_frontier_architecture_replay_sweep",
]

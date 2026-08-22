"""Immutable protocol, policy, node, and split records for learned routing."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence

from apm.continual.artifacts import record_sha256, require_sha256


def _hashes(values: Sequence[str], label: str) -> tuple[str, ...]:
    result = tuple(values)
    if len(set(result)) != len(result):
        raise ValueError(f"{label} hashes must be unique")
    for value in result:
        require_sha256(value, label)
    return result


@dataclass(frozen=True, slots=True)
class RouterSplit:
    """Frozen disjoint router-fit and router-validation membership."""

    sealed_dataset_hash: str
    fit_image_ids: tuple[str, ...]
    validation_image_ids: tuple[str, ...]
    namespace: str
    schema_version: str = "imagenetr50-router-split-v1"

    def __post_init__(self) -> None:
        require_sha256(self.sealed_dataset_hash, "sealed dataset")
        fit = _hashes(self.fit_image_ids, "router-fit image")
        validation = _hashes(self.validation_image_ids, "router-validation image")
        if (
            self.schema_version != "imagenetr50-router-split-v1"
            or not self.namespace
            or not fit
            or not validation
            or set(fit) & set(validation)
        ):
            raise ValueError("invalid router split")

    @property
    def content_hash(self) -> str:
        return record_sha256(self.as_record(False))

    def as_record(self, include_hash: bool = True) -> dict[str, object]:
        core: dict[str, object] = {
            "fit_image_ids": list(self.fit_image_ids),
            "namespace": self.namespace,
            "schema_version": self.schema_version,
            "sealed_dataset_hash": self.sealed_dataset_hash,
            "validation_image_ids": list(self.validation_image_ids),
        }
        return {**core, "content_hash": self.content_hash} if include_hash else core


@dataclass(frozen=True, slots=True)
class RouterProtocol:
    """Router run identity linked to immutable inference artifacts."""

    sealed_run_hash: str
    sealed_dataset_hash: str
    sealed_model_hash: str
    inference_inventory_hash: str
    inference_policies: tuple[tuple[str, str], ...]
    split_hash: str
    config_hash: str
    code_manifest_hash: str
    environment_manifest_hash: str
    schema_version: str = "imagenetr50-router-protocol-v1"

    def __post_init__(self) -> None:
        for label, value in (
            ("sealed run", self.sealed_run_hash),
            ("sealed dataset", self.sealed_dataset_hash),
            ("sealed model", self.sealed_model_hash),
            ("inference inventory", self.inference_inventory_hash),
            ("router split", self.split_hash),
            ("router config", self.config_hash),
            ("router code", self.code_manifest_hash),
            ("router environment", self.environment_manifest_hash),
        ):
            require_sha256(value, label)
        if (
            self.schema_version != "imagenetr50-router-protocol-v1"
            or tuple(name for name, _ in self.inference_policies)
            != ("I-U100", "I-SVD0", "I-SVD5")
        ):
            raise ValueError("invalid router protocol")
        for name, value in self.inference_policies:
            require_sha256(value, f"{name} inference policy")

    @property
    def content_hash(self) -> str:
        return record_sha256(self.as_record(False))

    def as_record(self, include_hash: bool = True) -> dict[str, object]:
        core: dict[str, object] = {
            "code_manifest_hash": self.code_manifest_hash,
            "config_hash": self.config_hash,
            "environment_manifest_hash": self.environment_manifest_hash,
            "inference_inventory_hash": self.inference_inventory_hash,
            "inference_policies": [list(value) for value in self.inference_policies],
            "schema_version": self.schema_version,
            "sealed_dataset_hash": self.sealed_dataset_hash,
            "sealed_model_hash": self.sealed_model_hash,
            "sealed_run_hash": self.sealed_run_hash,
            "split_hash": self.split_hash,
        }
        return {**core, "content_hash": self.content_hash} if include_hash else core


@dataclass(frozen=True, slots=True)
class RouterPolicy:
    """Scientific identity of one learned-router condition."""

    inference_condition: str
    inference_policy_hash: str
    architecture: str
    rank: int
    maintenance: str
    repair_fraction: float
    router_seed: int
    descriptor_config_hash: str
    response_config_hash: str
    training_config_hash: str
    schema_version: str = "imagenetr50-router-policy-v1"

    def __post_init__(self) -> None:
        for label, value in (
            ("inference policy", self.inference_policy_hash),
            ("descriptor config", self.descriptor_config_hash),
            ("response config", self.response_config_hash),
            ("training config", self.training_config_hash),
        ):
            require_sha256(value, label)
        if (
            self.schema_version != "imagenetr50-router-policy-v1"
            or self.inference_condition not in {"I-U100", "I-SVD0", "I-SVD5"}
            or self.architecture not in {"r0", "r1", "r2", "r3"}
            or self.rank < 1
            or self.maintenance
            not in {"flat_full", "flat_seen_data", "exact", "u100", "svd0", "svd5"}
            or not 0.0 <= self.repair_fraction <= 1.0
            or (self.maintenance == "svd5") != (self.repair_fraction > 0.0)
            or self.router_seed < 0
        ):
            raise ValueError("invalid router policy")

    @property
    def content_hash(self) -> str:
        return record_sha256(self.as_record(False))

    def as_record(self, include_hash: bool = True) -> dict[str, object]:
        core: dict[str, object] = {
            "architecture": self.architecture,
            "descriptor_config_hash": self.descriptor_config_hash,
            "inference_condition": self.inference_condition,
            "inference_policy_hash": self.inference_policy_hash,
            "maintenance": self.maintenance,
            "rank": self.rank,
            "repair_fraction": self.repair_fraction,
            "response_config_hash": self.response_config_hash,
            "router_seed": self.router_seed,
            "schema_version": self.schema_version,
            "training_config_hash": self.training_config_hash,
        }
        return {**core, "content_hash": self.content_hash} if include_hash else core


@dataclass(frozen=True, slots=True)
class RouterNodeArtifact:
    """Immutable learned score state paired with one inference node."""

    router_run_hash: str
    policy_hash: str
    inference_node_hash: str
    logical_node_id: str
    stage_created: int
    level: int
    represented_task_ids: tuple[int, ...]
    represented_class_ids: tuple[int, ...]
    represented_fit_count: int
    architecture: str
    rank: int
    maintenance: str
    scorer_sha256: str
    descriptor_sha256: str
    response_kernel_sha256: str | None
    parent_router_hashes: tuple[str, ...]
    training_ids_hash: str | None
    repair_ids_hash: str | None
    optimizer_steps: int
    schema_version: str = "imagenetr50-router-node-v1"

    def __post_init__(self) -> None:
        for label, value in (
            ("router run", self.router_run_hash),
            ("router policy", self.policy_hash),
            ("inference node", self.inference_node_hash),
            ("logical node", self.logical_node_id),
            ("scorer tensors", self.scorer_sha256),
            ("descriptor tensors", self.descriptor_sha256),
        ):
            require_sha256(value, label)
        for value in self.parent_router_hashes:
            require_sha256(value, "parent router")
        for label, value in (
            ("response kernel", self.response_kernel_sha256),
            ("training IDs", self.training_ids_hash),
            ("repair IDs", self.repair_ids_hash),
        ):
            if value is not None:
                require_sha256(value, label)
        tasks = tuple(self.represented_task_ids)
        classes = tuple(self.represented_class_ids)
        if (
            self.schema_version != "imagenetr50-router-node-v1"
            or self.stage_created < 1
            or self.level < 0
            or tasks != tuple(sorted(set(tasks)))
            or classes != tuple(sorted(set(classes)))
            or len(classes) != 4 * len(tasks)
            or self.represented_fit_count < 1
            or self.architecture not in {"r0", "r1", "r2", "r3", "exact"}
            or self.rank < 1
            or self.maintenance not in {"leaf", "exact", "u100", "svd0", "svd5", "flat"}
            or len(self.parent_router_hashes) not in {0, 2}
            or self.optimizer_steps < 0
            or (self.architecture == "r3") != (self.response_kernel_sha256 is not None)
        ):
            raise ValueError("invalid router node artifact")

    @property
    def content_hash(self) -> str:
        return record_sha256(self.as_record(False))

    def as_record(self, include_hash: bool = True) -> dict[str, object]:
        core: dict[str, object] = {
            "architecture": self.architecture,
            "descriptor_sha256": self.descriptor_sha256,
            "inference_node_hash": self.inference_node_hash,
            "level": self.level,
            "logical_node_id": self.logical_node_id,
            "maintenance": self.maintenance,
            "optimizer_steps": self.optimizer_steps,
            "parent_router_hashes": list(self.parent_router_hashes),
            "policy_hash": self.policy_hash,
            "rank": self.rank,
            "repair_ids_hash": self.repair_ids_hash,
            "represented_class_ids": list(self.represented_class_ids),
            "represented_fit_count": self.represented_fit_count,
            "represented_task_ids": list(self.represented_task_ids),
            "response_kernel_sha256": self.response_kernel_sha256,
            "router_run_hash": self.router_run_hash,
            "schema_version": self.schema_version,
            "scorer_sha256": self.scorer_sha256,
            "stage_created": self.stage_created,
            "training_ids_hash": self.training_ids_hash,
        }
        return {**core, "content_hash": self.content_hash} if include_hash else core


@dataclass(frozen=True, slots=True)
class RouterStageSnapshot:
    """Immutable learned-router frontier aligned to one sealed inference snapshot."""

    router_run_hash: str
    policy_hash: str
    inference_condition: str
    stage: int
    logical_node_ids: tuple[str, ...]
    inference_node_hashes: tuple[str, ...]
    router_node_hashes: tuple[str, ...]
    cumulative_router_optimizer_steps: int
    leaf_optimizer_steps: int = 0
    inference_parent_optimizer_steps: int = 0
    schema_version: str = "imagenetr50-router-stage-snapshot-v1"

    def __post_init__(self) -> None:
        for label, value in (
            ("router run", self.router_run_hash),
            ("router policy", self.policy_hash),
        ):
            require_sha256(value, label)
        for label, values in (
            ("logical node", self.logical_node_ids),
            ("inference node", self.inference_node_hashes),
            ("router node", self.router_node_hashes),
        ):
            _hashes(values, label)
        if (
            self.schema_version != "imagenetr50-router-stage-snapshot-v1"
            or self.inference_condition not in {"I-U100", "I-SVD0", "I-SVD5"}
            or not 1 <= self.stage <= 50
            or not (
                len(self.logical_node_ids)
                == len(self.inference_node_hashes)
                == len(self.router_node_hashes)
            )
            or self.cumulative_router_optimizer_steps < 0
            or self.leaf_optimizer_steps != 0
            or self.inference_parent_optimizer_steps != 0
        ):
            raise ValueError("invalid learned-router stage snapshot")

    @property
    def content_hash(self) -> str:
        return record_sha256(self.as_record(False))

    def as_record(self, include_hash: bool = True) -> dict[str, object]:
        core: dict[str, object] = {
            "cumulative_router_optimizer_steps": self.cumulative_router_optimizer_steps,
            "inference_condition": self.inference_condition,
            "inference_node_hashes": list(self.inference_node_hashes),
            "inference_parent_optimizer_steps": self.inference_parent_optimizer_steps,
            "leaf_optimizer_steps": self.leaf_optimizer_steps,
            "logical_node_ids": list(self.logical_node_ids),
            "policy_hash": self.policy_hash,
            "router_node_hashes": list(self.router_node_hashes),
            "router_run_hash": self.router_run_hash,
            "schema_version": self.schema_version,
            "stage": self.stage,
        }
        return {**core, "content_hash": self.content_hash} if include_hash else core


def router_node_from_record(record: Mapping[str, object]) -> RouterNodeArtifact:
    """Parse and authenticate a persisted router-node record."""
    values = dict(record)
    supplied = str(values.pop("content_hash", ""))
    for key in ("parent_router_hashes", "represented_class_ids", "represented_task_ids"):
        values[key] = tuple(values[key])
    artifact = RouterNodeArtifact(**values)
    if artifact.content_hash != supplied:
        raise ValueError("router node content hash changed")
    return artifact


def router_snapshot_from_record(record: Mapping[str, object]) -> RouterStageSnapshot:
    """Parse and authenticate a persisted learned-router stage snapshot."""
    values = dict(record)
    supplied = str(values.pop("content_hash", ""))
    for key in ("inference_node_hashes", "logical_node_ids", "router_node_hashes"):
        values[key] = tuple(values[key])
    snapshot = RouterStageSnapshot(**values)
    if snapshot.content_hash != supplied:
        raise ValueError("router stage snapshot content hash changed")
    return snapshot


def policy_hashes(config: Mapping[str, object]) -> tuple[str, str, str]:
    """Return descriptor, response, and training subconfiguration identities."""
    return tuple(
        record_sha256(config[key]) for key in ("descriptor", "response", "training")
    )  # type: ignore[return-value]


__all__ = [
    "RouterNodeArtifact",
    "RouterPolicy",
    "RouterProtocol",
    "RouterSplit",
    "RouterStageSnapshot",
    "policy_hashes",
    "router_node_from_record",
    "router_snapshot_from_record",
]

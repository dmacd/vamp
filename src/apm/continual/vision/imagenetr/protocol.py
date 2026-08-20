"""Immutable protocol, policy, node, and job identities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping, Sequence
import os

from apm.continual.artifacts import record_sha256, require_sha256
from apm.continual.vision.imagenetr.constants import (
    JOB_SCHEMA,
    MERGE_POLICY_SCHEMA,
    NODE_SCHEMA,
    PROTOCOL_SCHEMA,
)


def _sorted_unique(values: Sequence[int], label: str) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if result != tuple(sorted(set(result))):
        raise ValueError(f"{label} must be sorted and unique")
    return result


@dataclass(frozen=True, slots=True)
class ResolvedProtocol:
    """Content identity binding all material inputs to one experiment run."""

    dataset_manifest_hash: str
    model_manifest_hash: str
    config_hash: str
    code_manifest_hash: str
    environment_manifest_hash: str
    class_order: tuple[int, ...]
    schema_version: str = PROTOCOL_SCHEMA

    def __post_init__(self) -> None:
        for label, value in (
            ("dataset manifest", self.dataset_manifest_hash),
            ("model manifest", self.model_manifest_hash),
            ("config", self.config_hash),
            ("code manifest", self.code_manifest_hash),
            ("environment manifest", self.environment_manifest_hash),
        ):
            require_sha256(value, label)
        if self.schema_version != PROTOCOL_SCHEMA:
            raise ValueError("unknown resolved protocol schema")
        if tuple(sorted(self.class_order)) != tuple(range(200)):
            raise ValueError("class order must be a permutation of 0..199")

    @property
    def content_hash(self) -> str:
        """Return the stable run namespace identity."""
        return record_sha256(self.as_record(include_hash=False))

    def as_record(self, include_hash: bool = True) -> dict[str, object]:
        """Return the canonical JSON-compatible protocol record."""
        core: dict[str, object] = {
            "class_order": list(self.class_order),
            "code_manifest_hash": self.code_manifest_hash,
            "config_hash": self.config_hash,
            "dataset_manifest_hash": self.dataset_manifest_hash,
            "environment_manifest_hash": self.environment_manifest_hash,
            "model_manifest_hash": self.model_manifest_hash,
            "schema_version": self.schema_version,
        }
        return {**core, "content_hash": self.content_hash} if include_hash else core


@dataclass(frozen=True, slots=True)
class MergePolicy:
    """Scientific identity of one hierarchy construction policy."""

    method: str
    output_rank: int
    scale: float
    weighting: str
    repair_fraction: float
    repair_config_hash: str
    proxy_size: int
    core_space_revision: str | None = None
    schema_version: str = MERGE_POLICY_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema_version != MERGE_POLICY_SCHEMA
            or self.method
            not in {"retrain_union", "svd", "core_tsv", "output_drift"}
            or self.output_rank < 1
            or self.scale <= 0.0
            or self.weighting != "source_image_count"
            or not 0.0 <= self.repair_fraction <= 1.0
            or self.proxy_size < 1
        ):
            raise ValueError("invalid merge policy")
        require_sha256(self.repair_config_hash, "repair configuration")
        if (self.method == "core_tsv") != (self.core_space_revision is not None):
            raise ValueError("Core+TSV alone must bind the pinned source revision")

    @property
    def content_hash(self) -> str:
        """Return the policy identity used by tree and cache namespaces."""
        return record_sha256(self.as_record(include_hash=False))

    @property
    def merge_cache_hash(self) -> str:
        """Return the replay-independent parameter-merge identity."""
        return record_sha256(
            {
                "core_space_revision": self.core_space_revision,
                "method": self.method,
                "output_rank": self.output_rank,
                "scale": self.scale,
                "schema_version": "imagenetr50-merge-cache-policy-v1",
                "weighting": self.weighting,
            }
        )

    def as_record(self, include_hash: bool = True) -> dict[str, object]:
        """Return a canonical policy record."""
        core: dict[str, object] = {
            "core_space_revision": self.core_space_revision,
            "method": self.method,
            "output_rank": self.output_rank,
            "proxy_size": self.proxy_size,
            "repair_config_hash": self.repair_config_hash,
            "repair_fraction": self.repair_fraction,
            "scale": self.scale,
            "schema_version": self.schema_version,
            "weighting": self.weighting,
        }
        return {**core, "content_hash": self.content_hash} if include_hash else core


@dataclass(frozen=True, slots=True)
class NodeArtifact:
    """Immutable semantic and tensor identity for one leaf or derived node."""

    run_hash: str
    software_manifest_hash: str
    git_commit: str
    creation_timestamp_utc: str
    level: int
    first_task: int
    last_task: int
    represented_task_ids: tuple[int, ...]
    represented_class_ids: tuple[int, ...]
    represented_train_image_count: int
    parent_hashes: tuple[str, ...]
    unrepaired_parent_hash: str | None
    consolidation_method: str
    consolidation_config_hash: str
    repair_config_hash: str
    lora_sha256: str
    classifier_sha256: str
    proxy_image_ids: tuple[str, ...]
    repair_image_ids: tuple[str, ...]
    source_priority_hash: str
    training_optimizer_steps: int
    lora_path: str = "adapter.safetensors"
    classifier_path: str = "classifier.safetensors"
    schema_version: str = NODE_SCHEMA

    def __post_init__(self) -> None:
        for label, value in (
            ("run", self.run_hash),
            ("software manifest", self.software_manifest_hash),
            ("consolidation config", self.consolidation_config_hash),
            ("repair config", self.repair_config_hash),
            ("LoRA tensors", self.lora_sha256),
            ("classifier tensors", self.classifier_sha256),
            ("source priorities", self.source_priority_hash),
        ):
            require_sha256(value, label)
        for parent in self.parent_hashes:
            require_sha256(parent, "parent node")
        if self.unrepaired_parent_hash is not None:
            require_sha256(self.unrepaired_parent_hash, "unrepaired parent")
        tasks = _sorted_unique(self.represented_task_ids, "represented tasks")
        classes = _sorted_unique(self.represented_class_ids, "represented classes")
        if (
            self.schema_version != NODE_SCHEMA
            or self.level < 0
            or self.first_task < 0
            or self.last_task < self.first_task
            or tasks != tuple(range(self.first_task, self.last_task + 1))
            or len(classes) != 4 * len(tasks)
            or self.represented_train_image_count < 1
            or self.training_optimizer_steps < 0
            or not self.git_commit
            or not self.creation_timestamp_utc.endswith("Z")
            or self.lora_path != "adapter.safetensors"
            or self.classifier_path != "classifier.safetensors"
            or len(set(self.proxy_image_ids)) != len(self.proxy_image_ids)
            or len(set(self.repair_image_ids)) != len(self.repair_image_ids)
        ):
            raise ValueError("invalid node artifact")
        expected_parents = 0 if self.consolidation_method in {"leaf", "baseline"} else 2
        if len(self.parent_hashes) != expected_parents:
            raise ValueError("node parent count does not match its method")
        if (len(self.repair_image_ids) > 0) != (self.unrepaired_parent_hash is not None):
            raise ValueError("only repaired nodes may reference an un-repaired parent")

    @property
    def content_hash(self) -> str:
        """Return the immutable semantic and tensor identity of the node."""
        return record_sha256(self._identity_record())

    def _identity_record(self) -> dict[str, object]:
        """Return material identity fields, excluding informational time/Git metadata."""
        return {
            key: value
            for key, value in self.as_record(include_hash=False).items()
            if key not in {"creation_timestamp_utc", "git_commit"}
        }

    def as_record(self, include_hash: bool = True) -> dict[str, object]:
        """Return the canonical node manifest record."""
        core: dict[str, object] = {
            "classifier_sha256": self.classifier_sha256,
            "classifier_path": self.classifier_path,
            "consolidation_config_hash": self.consolidation_config_hash,
            "consolidation_method": self.consolidation_method,
            "creation_timestamp_utc": self.creation_timestamp_utc,
            "first_task": self.first_task,
            "last_task": self.last_task,
            "level": self.level,
            "git_commit": self.git_commit,
            "lora_sha256": self.lora_sha256,
            "lora_path": self.lora_path,
            "parent_hashes": list(self.parent_hashes),
            "unrepaired_parent_hash": self.unrepaired_parent_hash,
            "proxy_image_ids": list(self.proxy_image_ids),
            "repair_config_hash": self.repair_config_hash,
            "repair_image_ids": list(self.repair_image_ids),
            "represented_class_ids": list(self.represented_class_ids),
            "represented_task_ids": list(self.represented_task_ids),
            "represented_train_image_count": self.represented_train_image_count,
            "run_hash": self.run_hash,
            "schema_version": self.schema_version,
            "source_priority_hash": self.source_priority_hash,
            "software_manifest_hash": self.software_manifest_hash,
            "training_optimizer_steps": self.training_optimizer_steps,
        }
        return {**core, "content_hash": self.content_hash} if include_hash else core


@dataclass(frozen=True, slots=True)
class JobSpec:
    """Immutable scheduler job with explicit dependencies and canonical payload."""

    run_hash: str
    kind: str
    dependencies: tuple[str, ...]
    payload: tuple[tuple[str, object], ...]
    schema_version: str = JOB_SCHEMA

    def __post_init__(self) -> None:
        require_sha256(self.run_hash, "job run")
        for dependency in self.dependencies:
            require_sha256(dependency, "job dependency")
        if (
            self.schema_version != JOB_SCHEMA
            or not self.kind
            or tuple(sorted(set(self.dependencies))) != self.dependencies
            or tuple(sorted(self.payload, key=lambda item: item[0])) != self.payload
            or len({key for key, _ in self.payload}) != len(self.payload)
        ):
            raise ValueError("invalid job specification")

    @classmethod
    def create(
        cls,
        run_hash: str,
        kind: str,
        dependencies: Sequence[str] = (),
        payload: Mapping[str, object] | None = None,
    ) -> JobSpec:
        """Create a job while canonicalizing dependency and payload order."""
        return cls(
            run_hash=run_hash,
            kind=kind,
            dependencies=tuple(sorted(set(dependencies))),
            payload=tuple(sorted((payload or {}).items())),
        )

    @property
    def content_hash(self) -> str:
        """Return the scheduler and artifact namespace identity for this job."""
        return record_sha256(self.as_record(include_hash=False))

    def as_record(self, include_hash: bool = True) -> dict[str, object]:
        """Return a canonical job record."""
        core: dict[str, object] = {
            "dependencies": list(self.dependencies),
            "kind": self.kind,
            "payload": dict(self.payload),
            "run_hash": self.run_hash,
            "schema_version": self.schema_version,
        }
        return {**core, "content_hash": self.content_hash} if include_hash else core


def material_tree_manifest(paths: Sequence[str | Path]) -> dict[str, object]:
    """Hash only selected material code/config files, excluding unrelated TRACE work."""
    from apm.continual.artifacts import file_sha256

    roots = tuple(Path(path).resolve() for path in paths)
    files = tuple(
        sorted(
            {
                file
                for root in roots
                for file in ((root,) if root.is_file() else root.rglob("*"))
                if file.is_file() and "__pycache__" not in file.parts
            }
        )
    )
    if not files:
        raise ValueError("material code manifest cannot be empty")
    common = Path(
        os.path.commonpath(
            [str(root if root.is_dir() else root.parent) for root in roots]
        )
    )
    rows = [
        {
            "path": str(file.relative_to(common)) if file.is_relative_to(common) else str(file),
            "sha256": file_sha256(file),
            "size_bytes": file.stat().st_size,
        }
        for file in files
    ]
    core: dict[str, object] = {
        "files": rows,
        "schema_version": "imagenetr50-material-code-v1",
    }
    return {**core, "content_hash": record_sha256(core)}


__all__ = [
    "JobSpec",
    "MergePolicy",
    "NodeArtifact",
    "ResolvedProtocol",
    "material_tree_manifest",
]

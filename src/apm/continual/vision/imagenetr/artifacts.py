"""Content-addressed storage and immutable publication for ImageNet-R VAMP."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import shutil
import tempfile

from apm.continual.artifacts import (
    file_sha256,
    fsync_directory,
    load_canonical_json,
    publish_immutable_json,
    record_sha256,
    require_sha256,
)
from apm.continual.vision.imagenetr.protocol import ResolvedProtocol
from apm.continual.vision.imagenetr.protocol import NodeArtifact
from apm.continual.vision.imagenetr.heads import (
    ClassifierRows,
    load_classifier,
    save_classifier,
)
from apm.continual.vision.imagenetr.lora import load_adapter, save_adapter
from apm.continual.vision.imagenetr.merging.common import LoRAFactors


ARTIFACT_MANIFEST = "artifact.json"


@dataclass(frozen=True, slots=True)
class VisionStore:
    """All stable paths belonging to one resolved ImageNet-R run."""

    root: Path
    run_hash: str

    def __post_init__(self) -> None:
        require_sha256(self.run_hash, "vision run")

    @classmethod
    def from_protocol(
        cls, protocol: ResolvedProtocol, artifact_root: str | Path
    ) -> VisionStore:
        """Construct a store without changing the filesystem."""
        return cls(Path(artifact_root).resolve(), protocol.content_hash)

    @property
    def run(self) -> Path:
        """Return the content-addressed run directory."""
        return self.root / "runs" / self.run_hash

    def prepare(self, protocol: ResolvedProtocol) -> None:
        """Create the fixed store layout and publish the run identity once."""
        if protocol.content_hash != self.run_hash:
            raise ValueError("resolved protocol differs from the store namespace")
        for relative in (
            "protocol",
            "leaves",
            "merge_cache",
            "trees",
            "baselines",
            "evaluations",
            "diagnostics",
            "reports",
            "logs",
            "checkpoints",
            "state",
            "work",
            "cache/proxy_activations",
            "cache/evaluation_logits",
            "cache/frozen_features",
        ):
            (self.run / relative).mkdir(parents=True, exist_ok=True)
        publish_immutable_json(
            self.run / "protocol" / "protocol_manifest.json", protocol.as_record()
        )

    def leaf_job(self, task_index: int, job_hash: str) -> Path:
        """Return one immutable leaf-training job directory."""
        require_sha256(job_hash, "leaf job")
        if not 0 <= task_index < 50:
            raise ValueError("leaf task index is outside 0..49")
        return self.run / "leaves" / f"task_{task_index:03d}" / job_hash

    def merge_pair(self, left_hash: str, right_hash: str, merge_hash: str) -> Path:
        """Return the replay-independent parameter merge cache directory."""
        for label, value in (
            ("left child", left_hash),
            ("right child", right_hash),
            ("merge policy", merge_hash),
        ):
            require_sha256(value, label)
        return self.run / "merge_cache" / f"{left_hash}__{right_hash}" / merge_hash

    def tree_node(self, policy_hash: str, logical_node_id: str) -> Path:
        """Return one final repaired-or-unrepaired policy node directory."""
        require_sha256(policy_hash, "tree policy")
        require_sha256(logical_node_id, "logical node")
        return self.run / "trees" / policy_hash / "nodes" / logical_node_id

    def baseline(self, name: str, job_hash: str) -> Path:
        """Return one controlled baseline artifact directory."""
        require_sha256(job_hash, "baseline job")
        if name not in {"frozen_reference", "seq_lora_r16", "joint_iid_lora_r16"}:
            raise ValueError("unknown controlled baseline")
        return self.run / "baselines" / name / job_hash


def artifact_manifest(source: str | Path) -> dict[str, object]:
    """Hash every regular file in a completed artifact directory."""
    root = Path(source)
    files = tuple(
        (path.relative_to(root).as_posix(), file_sha256(path), path.stat().st_size)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != ARTIFACT_MANIFEST
    )
    if not files:
        raise ValueError("immutable artifact directories cannot be empty")
    core: dict[str, object] = {
        "files": [
            {"path": path, "sha256": digest, "size_bytes": size}
            for path, digest, size in files
        ],
        "schema_version": "imagenetr50-artifact-directory-v1",
    }
    return {**core, "artifact_sha256": record_sha256(core)}


def validate_artifact_directory(path: str | Path) -> str:
    """Validate every byte of a published artifact directory."""
    root = Path(path)
    persisted = load_canonical_json(root / ARTIFACT_MANIFEST)
    computed = artifact_manifest(root)
    if persisted != computed:
        raise ValueError(f"immutable vision artifact changed: {root}")
    identity = str(computed["artifact_sha256"])
    require_sha256(identity, "vision artifact")
    return identity


def publish_artifact_directory(source: str | Path, target: str | Path) -> str:
    """Atomically publish one completed directory or reuse an identical prior one."""
    source_root, target_root = Path(source), Path(target)
    manifest = artifact_manifest(source_root)
    if target_root.is_dir():
        existing = validate_artifact_directory(target_root)
        if existing != manifest["artifact_sha256"]:
            raise ValueError(f"immutable vision artifact collision: {target_root}")
        return existing
    if target_root.exists():
        raise ValueError("artifact target exists but is not a directory")
    target_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target_root.name}.", dir=target_root.parent)
    )
    try:
        for path in sorted(source_root.rglob("*")):
            relative = path.relative_to(source_root)
            if path.is_symlink():
                raise ValueError("artifact directories cannot contain symbolic links")
            if path.is_dir():
                (temporary / relative).mkdir(parents=True, exist_ok=True)
            elif path.name != ARTIFACT_MANIFEST:
                destination = temporary / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, destination)
                with destination.open("rb") as copied:
                    os.fsync(copied.fileno())
        publish_immutable_json(temporary / ARTIFACT_MANIFEST, manifest)
        os.rename(temporary, target_root)
        fsync_directory(target_root.parent)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return validate_artifact_directory(target_root)


@dataclass(frozen=True, slots=True)
class NodeBundle:
    """Validated immutable node metadata, adapter factors, and classifier rows."""

    artifact: NodeArtifact
    adapter: dict[str, LoRAFactors]
    classifier: ClassifierRows
    directory: Path


def node_artifact_from_record(record: dict[str, object]) -> NodeArtifact:
    """Parse a canonical node manifest while restoring immutable tuple fields."""
    supplied = str(record.pop("content_hash", ""))
    for field in (
        "parent_hashes",
        "proxy_image_ids",
        "repair_image_ids",
        "represented_class_ids",
        "represented_task_ids",
    ):
        record[field] = tuple(record[field])
    artifact = NodeArtifact(**record)
    if artifact.content_hash != supplied:
        raise ValueError("node content hash changed")
    return artifact


def write_node_work_directory(
    directory: str | Path,
    adapter: dict[str, LoRAFactors],
    classifier: ClassifierRows,
    metadata: dict[str, object],
) -> NodeArtifact:
    """Write node tensors and its self-authenticating semantic manifest to work state."""
    work = Path(directory)
    work.mkdir(parents=True, exist_ok=False)
    lora_sha = save_adapter(work / "adapter.safetensors", adapter)
    classifier_sha = save_classifier(work / "classifier.safetensors", classifier)
    artifact = NodeArtifact(
        **metadata,
        lora_sha256=lora_sha,
        classifier_sha256=classifier_sha,
    )
    publish_immutable_json(work / "node.json", artifact.as_record())
    return artifact


def load_node_bundle(path: str | Path) -> NodeBundle:
    """Validate an immutable directory, semantic record, and both tensor files."""
    root = Path(path)
    validate_artifact_directory(root)
    artifact = node_artifact_from_record(load_canonical_json(root / "node.json"))
    adapter_path = root / artifact.lora_path
    classifier_path = root / artifact.classifier_path
    if (
        file_sha256(adapter_path) != artifact.lora_sha256
        or file_sha256(classifier_path) != artifact.classifier_sha256
    ):
        raise ValueError("node tensor file identity differs from its semantic manifest")
    return NodeBundle(
        artifact,
        load_adapter(adapter_path),
        load_classifier(classifier_path),
        root,
    )


__all__ = [
    "ARTIFACT_MANIFEST",
    "VisionStore",
    "NodeBundle",
    "artifact_manifest",
    "publish_artifact_directory",
    "load_node_bundle",
    "node_artifact_from_record",
    "validate_artifact_directory",
    "write_node_work_directory",
]

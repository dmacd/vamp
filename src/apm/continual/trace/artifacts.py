"""Content-addressed storage layout and immutable directory publication."""

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
from apm.continual.trace.protocol import RunContract, default_store_root


ARTIFACT_MANIFEST = "artifact.json"


@dataclass(frozen=True, slots=True)
class TraceStore:
    """Resolved persistent paths for one content-addressed TRACE run."""

    root: Path
    run_hash: str

    def __post_init__(self) -> None:
        require_sha256(self.run_hash, "TRACE run")

    @classmethod
    def from_contract(
        cls,
        contract: RunContract,
        root: str | Path | None = None,
    ) -> TraceStore:
        """Construct a store without mutating the filesystem."""
        return cls(
            root=Path(root) if root is not None else default_store_root(),
            run_hash=contract.run_contract_hash,
        )

    @property
    def run(self) -> Path:
        """Return this run's immutable namespace root."""
        return self.root / "runs" / self.run_hash

    @property
    def cache(self) -> Path:
        """Return the cross-policy cache root."""
        return self.root / "cache"

    def prepare(self, contract: RunContract) -> None:
        """Create the fixed directory skeleton and publish the run contract."""
        if contract.run_contract_hash != self.run_hash:
            raise ValueError("run contract does not match the TRACE store")
        for relative in (
            "manifests",
            "leaves",
            "merge_cache",
            "derived",
            "baselines",
            "evaluations",
            "reports",
            "logs",
            "checkpoints",
            "state",
            "work",
        ):
            (self.run / relative).mkdir(parents=True, exist_ok=True)
        (self.cache / "huggingface").mkdir(parents=True, exist_ok=True)
        (self.cache / "trace").mkdir(parents=True, exist_ok=True)
        publish_immutable_json(self.run / "manifests" / "run.json", contract.as_record())

    def leaf(self, leaf_id: str) -> Path:
        """Return the immutable artifact directory for one level-zero adapter."""
        require_sha256(leaf_id, "TRACE leaf")
        return self.run / "leaves" / leaf_id

    def merge_pair(self, left_hash: str, right_hash: str) -> Path:
        """Return the cache directory for one ordered child pair."""
        require_sha256(left_hash, "left child")
        require_sha256(right_hash, "right child")
        return self.run / "merge_cache" / f"{left_hash}__{right_hash}"

    def node(self, policy_hash: str, node_id: str) -> Path:
        """Return a derived node artifact directory."""
        require_sha256(policy_hash, "TRACE policy")
        require_sha256(node_id, "TRACE node")
        return self.run / "derived" / policy_hash / "nodes" / node_id

    def evaluation(self, policy_hash: str, router_hash: str) -> Path:
        """Return one policy/router evaluation namespace."""
        require_sha256(policy_hash, "TRACE policy")
        require_sha256(router_hash, "TRACE router")
        return self.run / "evaluations" / policy_hash / router_hash


def artifact_manifest(source: str | Path) -> dict[str, object]:
    """Build the canonical manifest for every regular artifact file."""
    root = Path(source)
    files = tuple(
        (str(path.relative_to(root)), file_sha256(path), path.stat().st_size)
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
        "format": "trace-artifact-directory-v1",
    }
    return {**core, "artifact_sha256": record_sha256(core)}


def validate_artifact_directory(path: str | Path) -> str:
    """Validate a published directory and return its content identity."""
    root = Path(path)
    persisted = load_canonical_json(root / ARTIFACT_MANIFEST)
    computed = artifact_manifest(root)
    if persisted != computed:
        raise ValueError(f"artifact directory changed: {root}")
    identity = str(computed["artifact_sha256"])
    require_sha256(identity, "TRACE artifact")
    return identity


def publish_artifact_directory(source: str | Path, target: str | Path) -> str:
    """Atomically publish a completed directory or require prior identity."""
    source_root, target_root = Path(source), Path(target)
    manifest = artifact_manifest(source_root)
    if target_root.is_dir():
        existing = validate_artifact_directory(target_root)
        if existing != manifest["artifact_sha256"]:
            raise ValueError(f"immutable artifact changed: {target_root}")
        return existing
    if target_root.exists():
        raise ValueError(f"artifact target is not a directory: {target_root}")
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
                (temporary / relative).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, temporary / relative)
                with (temporary / relative).open("rb") as copied:
                    os.fsync(copied.fileno())
        publish_immutable_json(temporary / ARTIFACT_MANIFEST, manifest)
        os.rename(temporary, target_root)
        fsync_directory(target_root.parent)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return validate_artifact_directory(target_root)


__all__ = [
    "ARTIFACT_MANIFEST",
    "TraceStore",
    "artifact_manifest",
    "publish_artifact_directory",
    "validate_artifact_directory",
]

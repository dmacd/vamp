"""Authenticated inputs and content-addressed storage for the integrator study."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from apm.continual.artifacts import (
    atomic_write,
    canonical_json_bytes,
    file_sha256,
    load_canonical_json,
    publish_immutable_bytes,
    publish_immutable_json,
    record_sha256,
    require_sha256,
)
from apm.continual.vision.imagenetr.config import ImageNetRConfig, load_config
from apm.continual.vision.imagenetr.data import DatasetManifest, image_transforms, load_dataset_manifest
from apm.continual.vision.imagenetr.integrator_config import (
    ImageNetRIntegratorConfig,
    load_integrator_config,
)
from apm.continual.vision.imagenetr.manifests import installed_environment_manifest
from apm.continual.vision.imagenetr.protocol import material_tree_manifest
from apm.continual.vision.imagenetr.router_artifacts import (
    SealedInferenceTree,
    load_sealed_tree,
    router_split_from_record,
)
from apm.continual.vision.imagenetr.router_protocol import RouterSplit


INTEGRATOR_PACKAGES = (
    "apm",
    "matplotlib",
    "numpy",
    "pandas",
    "Pillow",
    "pyarrow",
    "PyYAML",
    "safetensors",
    "timm",
    "torch",
    "torchvision",
    "tqdm",
)


@dataclass(frozen=True, slots=True)
class IntegratorProtocol:
    """Content identity binding sealed inputs, code, environment, and configuration."""

    sealed_run_hash: str
    sealed_u100_policy_hash: str
    sealed_final_node_hashes: tuple[str, ...]
    dataset_manifest_hash: str
    model_manifest_hash: str
    split_hash: str
    config_hash: str
    code_manifest_hash: str
    environment_manifest_hash: str
    reference_results_hash: str
    schema_version: str = "imagenetr50-logt-integrator-protocol-v1"

    def __post_init__(self) -> None:
        for label, identity in (
            ("sealed run", self.sealed_run_hash),
            ("sealed policy", self.sealed_u100_policy_hash),
            ("dataset", self.dataset_manifest_hash),
            ("model", self.model_manifest_hash),
            ("split", self.split_hash),
            ("config", self.config_hash),
            ("code", self.code_manifest_hash),
            ("environment", self.environment_manifest_hash),
            ("reference results", self.reference_results_hash),
        ):
            require_sha256(identity, label)
        for identity in self.sealed_final_node_hashes:
            require_sha256(identity, "sealed final node")
        if (
            self.schema_version != "imagenetr50-logt-integrator-protocol-v1"
            or not self.sealed_final_node_hashes
            or len(set(self.sealed_final_node_hashes)) != len(self.sealed_final_node_hashes)
        ):
            raise ValueError("invalid ImageNet-R integrator protocol")

    @property
    def content_hash(self) -> str:
        """Return the stable run namespace."""
        return record_sha256(self.as_record(False))

    def as_record(self, include_hash: bool = True) -> dict[str, object]:
        """Return the canonical JSON-compatible protocol record."""
        core: dict[str, object] = {
            "code_manifest_hash": self.code_manifest_hash,
            "config_hash": self.config_hash,
            "dataset_manifest_hash": self.dataset_manifest_hash,
            "environment_manifest_hash": self.environment_manifest_hash,
            "model_manifest_hash": self.model_manifest_hash,
            "reference_results_hash": self.reference_results_hash,
            "schema_version": self.schema_version,
            "sealed_final_node_hashes": list(self.sealed_final_node_hashes),
            "sealed_run_hash": self.sealed_run_hash,
            "sealed_u100_policy_hash": self.sealed_u100_policy_hash,
            "split_hash": self.split_hash,
        }
        return {**core, "content_hash": self.content_hash} if include_hash else core


@dataclass(frozen=True, slots=True)
class HierarchyPolicy:
    """Identity of one full-union or bounded-replay capacity-one hierarchy."""

    partition: str
    replay_mode: str
    reservoir_capacity: int
    training_config_hash: str
    seed: int
    schema_version: str = "imagenetr50-integrator-hierarchy-policy-v1"

    def __post_init__(self) -> None:
        require_sha256(self.training_config_hash, "hierarchy training config")
        if (
            self.partition not in {"fit", "all_train"}
            or self.replay_mode not in {"bounded", "full_union"}
            or self.reservoir_capacity < 1
            or self.seed < 0
            or self.schema_version != "imagenetr50-integrator-hierarchy-policy-v1"
        ):
            raise ValueError("invalid capacity-one hierarchy policy")

    @property
    def content_hash(self) -> str:
        """Return the canonical hierarchy namespace."""
        return record_sha256(self.as_record(False))

    def as_record(self, include_hash: bool = True) -> dict[str, object]:
        """Return a JSON-compatible policy record."""
        core = asdict(self)
        return {**core, "content_hash": self.content_hash} if include_hash else core


@dataclass(frozen=True, slots=True)
class IntegratorStageSnapshot:
    """Immutable mapping from a capacity-one logical frontier to trained nodes."""

    policy_hash: str
    stage: int
    logical_node_ids: tuple[str, ...]
    node_hashes: tuple[str, ...]
    levels: tuple[int, ...]
    schema_version: str = "imagenetr50-integrator-stage-snapshot-v1"

    def __post_init__(self) -> None:
        require_sha256(self.policy_hash, "hierarchy policy")
        for identity in (*self.logical_node_ids, *self.node_hashes):
            require_sha256(identity, "snapshot node")
        if (
            self.schema_version != "imagenetr50-integrator-stage-snapshot-v1"
            or not 1 <= self.stage <= 50
            or not self.node_hashes
            or len(self.logical_node_ids) != len(self.node_hashes)
            or len(self.levels) != len(self.node_hashes)
            or tuple(sorted(set(self.levels))) != self.levels
        ):
            raise ValueError("invalid integrator hierarchy snapshot")

    @property
    def content_hash(self) -> str:
        """Return the exact frontier identity."""
        return record_sha256(self.as_record(False))

    def as_record(self, include_hash: bool = True) -> dict[str, object]:
        """Return a canonical JSON-compatible snapshot record."""
        core: dict[str, object] = {
            "levels": list(self.levels),
            "logical_node_ids": list(self.logical_node_ids),
            "node_hashes": list(self.node_hashes),
            "policy_hash": self.policy_hash,
            "schema_version": self.schema_version,
            "stage": self.stage,
        }
        return {**core, "content_hash": self.content_hash} if include_hash else core


@dataclass(frozen=True, slots=True)
class IntegratorStore:
    """Stable paths owned exclusively by one prediction-integrator protocol."""

    root: Path
    run_hash: str

    def __post_init__(self) -> None:
        require_sha256(self.run_hash, "integrator run")

    @property
    def run(self) -> Path:
        """Return the content-addressed run directory."""
        return self.root / "runs" / self.run_hash

    def prepare(self, protocol: IntegratorProtocol) -> None:
        """Create the isolated layout and immutably publish the protocol."""
        if protocol.content_hash != self.run_hash:
            raise ValueError("integrator protocol differs from its store namespace")
        for relative in (
            "protocol",
            "cache/behaviors",
            "diagnostic",
            "hierarchies",
            "integrators",
            "evaluations",
            "checkpoints",
            "ledgers",
            "reports",
            "state",
            "work",
        ):
            (self.run / relative).mkdir(parents=True, exist_ok=True)
        publish_immutable_json(self.run / "protocol" / "protocol.json", protocol.as_record())

    def hierarchy(self, policy_hash: str) -> Path:
        """Return one hierarchy-policy root."""
        require_sha256(policy_hash, "hierarchy policy")
        return self.run / "hierarchies" / policy_hash

    def hierarchy_node(self, policy_hash: str, logical_node_id: str) -> Path:
        """Return one trained logical-node directory."""
        require_sha256(logical_node_id, "logical node")
        return self.hierarchy(policy_hash) / "nodes" / logical_node_id

    def snapshot(self, policy_hash: str, stage: int) -> Path:
        """Return one durable post-arrival frontier record."""
        if not 1 <= stage <= 50:
            raise ValueError("snapshot stage is outside 1..50")
        return self.hierarchy(policy_hash) / "snapshots" / f"stage_{stage:03d}.json"


@dataclass(frozen=True, slots=True)
class IntegratorBootstrap:
    """Authenticated local inputs and paths needed by every workflow phase."""

    project_root: Path
    config_path: Path
    config: ImageNetRIntegratorConfig
    primary_config: ImageNetRConfig
    manifest: DatasetManifest
    split: RouterSplit
    sealed_tree: SealedInferenceTree
    protocol: IntegratorProtocol
    store: IntegratorStore
    checkpoint: Path
    train_transform: object
    test_transform: object
    model_manifest: dict[str, object]
    code_manifest: dict[str, object]
    environment_manifest: dict[str, object]


def _material_paths(project_root: Path, config_path: Path) -> tuple[Path, ...]:
    package = project_root / "src/apm/continual/vision/imagenetr"
    shared = (
        project_root / "src/apm/continual/artifacts.py",
        *(package / name for name in (
            "artifacts.py",
            "bank.py",
            "calibration.py",
            "checkpoints.py",
            "config.py",
            "constants.py",
            "data.py",
            "heads.py",
            "lora.py",
            "manifests.py",
            "model.py",
            "protocol.py",
            "proxy_memory.py",
            "router_artifacts.py",
            "router_config.py",
            "router_descriptor.py",
            "router_features.py",
            "router_protocol.py",
            "training.py",
        )),
        package / "merging" / "common.py",
    )
    script = project_root / "scripts/vision/imagenetr/run_integrator_local.sh"
    protocol = project_root / "docs/imagenetr50_logt_prediction_integrator_protocol.md"
    optional = (script,) if script.is_file() else ()
    return (
        config_path,
        protocol,
        *shared,
        *sorted(package.glob("integrator_*.py")),
        *optional,
    )


def _checkpoint(data_root: Path, model_manifest: Mapping[str, object]) -> Path:
    identity = str(model_manifest["sha256"])
    candidates = tuple(
        path for path in (data_root / "model_cache").rglob(identity) if path.is_file()
    )
    if len(candidates) != 1 or file_sha256(candidates[0]) != identity:
        raise FileNotFoundError("the pinned ViT checkpoint is not locally authenticated")
    return candidates[0]


def bootstrap_integrator(config_path: str | Path) -> IntegratorBootstrap:
    """Authenticate sealed dependencies and freeze a separate integrator run."""
    source = Path(config_path).resolve()
    project_root = source.parents[3]
    config = load_integrator_config(source)
    primary_root = config.inference_artifact_root / "runs" / config.sealed_run_hash
    primary_protocol = load_canonical_json(primary_root / "protocol" / "protocol_manifest.json")
    if primary_protocol.get("content_hash") != config.sealed_run_hash:
        raise ValueError("configured primary run is not self-authenticating")
    primary_config = load_config(project_root / "configs/vision/imagenetr/primary.yaml")
    if primary_config.config_hash != primary_protocol.get("config_hash"):
        raise ValueError("current primary configuration differs from the sealed run")
    manifest = load_dataset_manifest(config.data_root / "imagenet-r" / "dataset_manifest.json")
    if manifest.content_hash != primary_protocol.get("dataset_manifest_hash"):
        raise ValueError("local ImageNet-R data differs from the sealed primary run")
    split_path = (
        config.router_artifact_root
        / "runs"
        / config.sealed_router_run_hash
        / "protocol"
        / "router_split.json"
    )
    split = router_split_from_record(load_canonical_json(split_path))
    if split.content_hash != config.sealed_router_split_hash:
        raise ValueError("configured clean fit/validation split changed")
    sealed_tree = load_sealed_tree(
        primary_root, "I-U100", config.sealed_u100_policy_hash
    )
    primary_summary = primary_root / "reports" / "summary.json"
    primary_preflight = primary_root / "protocol" / "preflight.json"
    router_stage_metrics = (
        config.router_artifact_root
        / "runs"
        / config.sealed_router_run_hash
        / "reports"
        / "stage_metrics.json"
    )
    if (
        not primary_summary.is_file()
        or not primary_preflight.is_file()
        or not router_stage_metrics.is_file()
    ):
        raise FileNotFoundError("sealed local comparison results are unavailable")
    reference_results = {
        "primary_preflight_sha256": file_sha256(primary_preflight),
        "primary_summary_sha256": file_sha256(primary_summary),
        "router_stage_metrics_sha256": file_sha256(router_stage_metrics),
        "schema_version": "imagenetr50-integrator-reference-results-v1",
    }
    reference_results = {
        **reference_results,
        "content_hash": record_sha256(reference_results),
    }
    model_manifest = load_canonical_json(primary_root / "protocol" / "model_manifest.json")
    environment = installed_environment_manifest(INTEGRATOR_PACKAGES)
    missing = tuple(
        str(row["name"])
        for row in environment["packages"]
        if row["version"] == "MISSING"
    )
    if missing:
        raise RuntimeError(f"isolated vision environment is incomplete: {missing}")
    code = material_tree_manifest(_material_paths(project_root, source))
    protocol = IntegratorProtocol(
        config.sealed_run_hash,
        config.sealed_u100_policy_hash,
        tuple(node.node_hash for node in sealed_tree.final.nodes),
        manifest.content_hash,
        str(primary_protocol["model_manifest_hash"]),
        split.content_hash,
        config.config_hash,
        str(code["content_hash"]),
        str(environment["content_hash"]),
        str(reference_results["content_hash"]),
    )
    store = IntegratorStore(config.artifact_root, protocol.content_hash)
    store.prepare(protocol)
    for filename, record in (
        ("code_manifest.json", code),
        ("environment_manifest.json", environment),
        ("model_manifest.json", model_manifest),
        ("router_split.json", split.as_record()),
        ("reference_results.json", reference_results),
    ):
        publish_immutable_json(store.run / "protocol" / filename, record)
    publish_immutable_bytes(
        store.run / "config_resolved.json", canonical_json_bytes(config.as_record())
    )
    atomic_write(
        config.artifact_root / "LATEST_RUN.json",
        canonical_json_bytes(
            {
                "run_hash": protocol.content_hash,
                "schema_version": "imagenetr50-integrator-latest-run-v1",
            }
        ),
    )
    train_transform, test_transform = image_transforms(primary_config.input_size)
    return IntegratorBootstrap(
        project_root,
        source,
        config,
        primary_config,
        manifest,
        split,
        sealed_tree,
        protocol,
        store,
        _checkpoint(config.data_root, model_manifest),
        train_transform,
        test_transform,
        model_manifest,
        code,
        environment,
    )


def latest_integrator_run(
    config_path: str | Path,
) -> tuple[ImageNetRIntegratorConfig, Path]:
    """Resolve the latest prepared run without mutating local state."""
    config = load_integrator_config(config_path)
    record = load_canonical_json(config.artifact_root / "LATEST_RUN.json")
    if record.get("schema_version") != "imagenetr50-integrator-latest-run-v1":
        raise ValueError("unknown integrator latest-run record")
    run_hash = str(record["run_hash"])
    require_sha256(run_hash, "latest integrator run")
    return config, config.artifact_root / "runs" / run_hash


def hierarchy_training_hash(config: ImageNetRIntegratorConfig) -> str:
    """Return the immutable consolidation optimizer identity."""
    return record_sha256(
        {
            **asdict(config.consolidation_training),
            "architecture": "vit_base_patch16_224.augreg_in21k+lora_r16",
            "schema_version": "imagenetr50-integrator-consolidation-training-v1",
        }
    )


def hierarchy_policy(
    config: ImageNetRIntegratorConfig,
    partition: str,
    replay_mode: str,
    reservoir_capacity: int,
    seed: int | None = None,
) -> HierarchyPolicy:
    """Construct one canonical hierarchy policy from the resolved protocol."""
    return HierarchyPolicy(
        partition,
        replay_mode,
        reservoir_capacity,
        hierarchy_training_hash(config),
        config.seed if seed is None else seed,
    )


__all__ = [
    "HierarchyPolicy",
    "IntegratorBootstrap",
    "IntegratorProtocol",
    "IntegratorStageSnapshot",
    "IntegratorStore",
    "bootstrap_integrator",
    "hierarchy_policy",
    "hierarchy_training_hash",
    "latest_integrator_run",
]

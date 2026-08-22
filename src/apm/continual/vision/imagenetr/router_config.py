"""Strict configuration for recursive learned ImageNet-R routing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from collections.abc import Mapping

from apm.continual.artifacts import record_sha256, require_sha256


@dataclass(frozen=True, slots=True)
class RouterTrainingConfig:
    """Deterministic cached-feature optimization settings."""

    lr: float
    weight_decay: float
    batch_size: int
    max_epochs: int
    patience: int
    margin: float
    negatives_per_live_node: int
    lse_weight: float

    def __post_init__(self) -> None:
        if (
            self.lr <= 0.0
            or self.weight_decay < 0.0
            or self.batch_size < 1
            or self.max_epochs < 1
            or self.patience < 1
            or self.patience > self.max_epochs
            or self.margin <= 0.0
            or self.negatives_per_live_node < 1
            or self.lse_weight < 0.0
        ):
            raise ValueError("invalid router training configuration")


@dataclass(frozen=True, slots=True)
class RouterConfig:
    """Complete resolved scientific identity of the router follow-up."""

    name: str
    seed: int
    router_seeds: tuple[int, ...]
    artifact_root: Path
    inference_artifact_root: Path
    data_root: Path
    sealed_run_hash: str
    handoff_commit: str
    inference_policies: tuple[tuple[str, str], ...]
    fit_fraction: float
    descriptor_dim: int
    descriptor_probe_dim: int
    descriptor_seed: int
    response_blocks: tuple[int, ...]
    response_targets: tuple[str, ...]
    response_dtype: str
    query_dim: int
    primary_rank: int
    fallback_rank: int
    mlp_hidden: int
    training: RouterTrainingConfig
    primary_repair_fraction: float
    secondary_repair_fraction: float
    feature_batch_size: int
    validation_batch_size: int
    num_workers: int
    smoke_tasks: int
    evaluation_checkpoints: tuple[int, ...]

    def __post_init__(self) -> None:
        require_sha256(self.sealed_run_hash, "sealed ImageNet-R run")
        for name, identity in self.inference_policies:
            if name not in {"I-U100", "I-SVD0", "I-SVD5"}:
                raise ValueError("unknown inference condition")
            require_sha256(identity, f"{name} policy")
        if (
            not self.name
            or self.seed < 0
            or not self.router_seeds
            or self.router_seeds[0] != self.seed
            or len(set(self.router_seeds)) != len(self.router_seeds)
            or any(value < 0 for value in self.router_seeds)
            or not self.handoff_commit
            or tuple(name for name, _ in self.inference_policies)
            != ("I-U100", "I-SVD0", "I-SVD5")
            or not 0.0 < self.fit_fraction < 1.0
            or self.descriptor_dim != 128
            or self.descriptor_probe_dim != 2
            or self.descriptor_seed < 0
            or self.response_blocks != (0, 4, 7, 11)
            or self.response_targets != ("attn.qkv", "mlp.fc1")
            or self.response_dtype != "bfloat16"
            or self.query_dim != 768
            or self.primary_rank != 8
            or self.fallback_rank != 16
            or self.mlp_hidden != 64
            or self.primary_repair_fraction != 0.05
            or self.secondary_repair_fraction != 0.01
            or self.feature_batch_size < 1
            or self.validation_batch_size < 1
            or self.num_workers < 0
            or not 1 <= self.smoke_tasks <= 50
            or self.evaluation_checkpoints != (8, 16, 32, 50)
        ):
            raise ValueError("invalid recursive-router protocol configuration")

    @property
    def policy_map(self) -> dict[str, str]:
        """Return the fixed condition-to-inference-policy mapping."""
        return dict(self.inference_policies)

    @property
    def config_hash(self) -> str:
        """Return the canonical resolved configuration identity."""
        return record_sha256(self.as_record())

    def as_record(self) -> dict[str, object]:
        """Return a canonical JSON-compatible record."""
        record = asdict(self)
        for name in ("artifact_root", "inference_artifact_root", "data_root"):
            record[name] = str(record[name])
        record["inference_policies"] = [list(value) for value in self.inference_policies]
        return record


def _mapping(value: object, label: str, keys: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{label} keys differ from the recursive-router protocol")
    return value


def _path(value: object, project_root: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def load_router_config(path: str | Path) -> RouterConfig:
    """Load the one strict YAML surface and reject silent scientific choices."""
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - environment gate
        raise RuntimeError("PyYAML is required by the vision environment") from error
    source = Path(path).resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    root = _mapping(
        raw,
        "configuration",
        {
            "experiment",
            "paths",
            "sealed_inference",
            "split",
            "descriptor",
            "response",
            "architecture",
            "training",
            "repair",
            "runtime",
        },
    )
    experiment = _mapping(root["experiment"], "experiment", {"name", "seed", "router_seeds"})
    paths = _mapping(
        root["paths"],
        "paths",
        {"artifact_root", "inference_artifact_root", "data_root"},
    )
    sealed = _mapping(
        root["sealed_inference"],
        "sealed_inference",
        {"run_hash", "handoff_commit", "policies"},
    )
    policies = _mapping(sealed["policies"], "sealed_inference.policies", {"I-U100", "I-SVD0", "I-SVD5"})
    split = _mapping(root["split"], "split", {"fit_fraction"})
    descriptor = _mapping(root["descriptor"], "descriptor", {"dim", "probe_dim", "seed"})
    response = _mapping(root["response"], "response", {"blocks", "targets", "dtype"})
    architecture = _mapping(
        root["architecture"],
        "architecture",
        {"query_dim", "primary_rank", "fallback_rank", "mlp_hidden"},
    )
    training = _mapping(
        root["training"],
        "training",
        {
            "optimizer",
            "lr",
            "weight_decay",
            "batch_size",
            "max_epochs",
            "patience",
            "margin",
            "negatives_per_live_node",
            "lse_weight",
        },
    )
    if str(training["optimizer"]).lower() != "adamw":
        raise ValueError("router training must use AdamW")
    repair = _mapping(root["repair"], "repair", {"primary_fraction", "secondary_fraction"})
    runtime = _mapping(
        root["runtime"],
        "runtime",
        {
            "feature_batch_size",
            "validation_batch_size",
            "num_workers",
            "smoke_tasks",
            "evaluation_checkpoints",
        },
    )
    project_root = source.parents[3]
    return RouterConfig(
        name=str(experiment["name"]),
        seed=int(experiment["seed"]),
        router_seeds=tuple(int(value) for value in experiment["router_seeds"]),
        artifact_root=_path(paths["artifact_root"], project_root),
        inference_artifact_root=_path(paths["inference_artifact_root"], project_root),
        data_root=_path(paths["data_root"], project_root),
        sealed_run_hash=str(sealed["run_hash"]),
        handoff_commit=str(sealed["handoff_commit"]),
        inference_policies=tuple((name, str(policies[name])) for name in ("I-U100", "I-SVD0", "I-SVD5")),
        fit_fraction=float(split["fit_fraction"]),
        descriptor_dim=int(descriptor["dim"]),
        descriptor_probe_dim=int(descriptor["probe_dim"]),
        descriptor_seed=int(descriptor["seed"]),
        response_blocks=tuple(int(value) for value in response["blocks"]),
        response_targets=tuple(str(value) for value in response["targets"]),
        response_dtype=str(response["dtype"]),
        query_dim=int(architecture["query_dim"]),
        primary_rank=int(architecture["primary_rank"]),
        fallback_rank=int(architecture["fallback_rank"]),
        mlp_hidden=int(architecture["mlp_hidden"]),
        training=RouterTrainingConfig(
            lr=float(training["lr"]),
            weight_decay=float(training["weight_decay"]),
            batch_size=int(training["batch_size"]),
            max_epochs=int(training["max_epochs"]),
            patience=int(training["patience"]),
            margin=float(training["margin"]),
            negatives_per_live_node=int(training["negatives_per_live_node"]),
            lse_weight=float(training["lse_weight"]),
        ),
        primary_repair_fraction=float(repair["primary_fraction"]),
        secondary_repair_fraction=float(repair["secondary_fraction"]),
        feature_batch_size=int(runtime["feature_batch_size"]),
        validation_batch_size=int(runtime["validation_batch_size"]),
        num_workers=int(runtime["num_workers"]),
        smoke_tasks=int(runtime["smoke_tasks"]),
        evaluation_checkpoints=tuple(int(value) for value in runtime["evaluation_checkpoints"]),
    )


__all__ = ["RouterConfig", "RouterTrainingConfig", "load_router_config"]

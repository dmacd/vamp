"""Clean ceiling selection and locked refit for the ImageNet-R macro-token head."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import math
import shutil
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

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
from apm.continual.vision.imagenetr.artifacts import validate_artifact_directory
from apm.continual.vision.imagenetr.data import ImageRecord, ManifestDataset
from apm.continual.vision.imagenetr.integrator_artifacts import IntegratorStore
from apm.continual.vision.imagenetr.integrator_hierarchy import HierarchyBuildResult
from apm.continual.vision.imagenetr.integrator_model import ImageNetResidualIntegrator
from apm.continual.vision.imagenetr.integrator_observations import BehaviorNode
from apm.continual.vision.imagenetr.macro_token_cache import (
    MacroTokenPopulation,
    clear_macro_population,
    clear_macro_population_identity,
    materialize_macro_population,
)
from apm.continual.vision.imagenetr.macro_token_config import (
    DEFAULT_MACRO_TOKEN_CONFIG,
    MacroTokenConfig,
    load_macro_token_config,
)
from apm.continual.vision.imagenetr.macro_token_model import (
    CLASS_COUNT,
    MAXIMUM_SLOTS,
    TOKEN_COUNT,
    TOKEN_DIMENSION,
    MacroTokenClassifier,
    MacroTokenInputs,
    behavior_meta_features,
    parameter_count,
)
from apm.continual.vision.imagenetr.macro_token_training import (
    CompactControlPopulation,
    MacroFitResult,
    MacroModelSpec,
    PopulationMetrics,
    compact_control_population,
    create_trainable_model,
    evaluate_compact_control,
    evaluate_frontier_controls,
    evaluate_frozen_owner_probe,
    evaluate_population,
    fit_compact_control,
    fit_frozen_owner_probe,
    fit_model,
    load_fitted_model,
    load_frozen_owner_probe,
    model_job_hash,
    publish_fitted_model,
    publish_frozen_owner_probe,
)
from apm.continual.vision.imagenetr.manifests import installed_environment_manifest
from apm.continual.vision.imagenetr.model import AdapterVisionModel, create_pinned_backbone
from apm.continual.vision.imagenetr.promoted_integrator_workflow import (
    PROMOTED_PACKAGES,
    PromotedBootstrap,
    _build,
    bootstrap_promoted_integrator,
)
from apm.continual.vision.imagenetr.behavior_replay_workflow import (
    _stored_integrator_protocol,
)
from apm.continual.vision.imagenetr.integrator_workflow import (
    _hierarchy_frontier,
    _partition_rows,
    _test_rows,
)
from apm.continual.vision.imagenetr.protocol import material_tree_manifest
from apm.continual.vision.imagenetr.router_features import test_transform_hash


@dataclass(frozen=True, slots=True)
class MacroTokenProtocol:
    """Content identity binding source hierarchy authority, code, and config."""

    source_run_hash: str
    source_protocol_sha256: str
    dataset_manifest_hash: str
    model_manifest_hash: str
    split_hash: str
    config_hash: str
    code_manifest_hash: str
    environment_manifest_hash: str
    schema_version: str = "imagenetr50-macro-token-protocol-v1"

    def __post_init__(self) -> None:
        for label, identity in (
            ("source run", self.source_run_hash),
            ("source protocol", self.source_protocol_sha256),
            ("dataset", self.dataset_manifest_hash),
            ("model", self.model_manifest_hash),
            ("split", self.split_hash),
            ("configuration", self.config_hash),
            ("code", self.code_manifest_hash),
            ("environment", self.environment_manifest_hash),
        ):
            require_sha256(identity, label)
        if self.schema_version != "imagenetr50-macro-token-protocol-v1":
            raise ValueError("invalid macro-token protocol")

    @property
    def content_hash(self) -> str:
        """Return the stable experiment namespace."""
        return record_sha256(self.as_record(False))

    def as_record(self, include_hash: bool = True) -> dict[str, object]:
        """Return a canonical JSON-compatible protocol record."""
        core = asdict(self)
        return {**core, "content_hash": self.content_hash} if include_hash else core


@dataclass(frozen=True, slots=True)
class MacroTokenBootstrap:
    """Authenticated source hierarchy plus isolated output namespace."""

    project_root: Path
    config: MacroTokenConfig
    source: PromotedBootstrap
    protocol: MacroTokenProtocol
    run: Path

    @property
    def scratch_root(self) -> Path:
        """Return the exact root of reproducible stage-local token caches."""
        return self.run / "scratch" / "macro_tokens"


def _material_paths(project_root: Path, config_path: Path) -> tuple[Path, ...]:
    package = project_root / "src/apm/continual/vision/imagenetr"
    return (
        config_path,
        project_root / "docs/ImageNetR_MacroToken_Integrator_Architecture_Reference.pdf",
        project_root / "docs/imagenetr50_macro_token_integrator_protocol.md",
        project_root / "scripts/vision/imagenetr/run_macro_token_integrator_local.sh",
        project_root / "src/apm/continual/artifacts.py",
        *(sorted(package.glob("macro_token_*.py"))),
        *(package / name for name in (
            "artifacts.py",
            "checkpoints.py",
            "data.py",
            "heads.py",
            "integrator_artifacts.py",
            "integrator_hierarchy.py",
            "integrator_model.py",
            "integrator_observations.py",
            "lora.py",
            "model.py",
            "promoted_integrator_config.py",
            "promoted_integrator_workflow.py",
            "protocol.py",
            "router_features.py",
            "training.py",
        )),
        package / "merging/common.py",
    )


def _prepare_run(run: Path, protocol: MacroTokenProtocol) -> None:
    for relative in (
        "protocol",
        "models",
        "evaluations/clean",
        "evaluations/locked_test",
        "checkpoints",
        "ledgers",
        "reports",
        "scratch/macro_tokens",
        "state",
        "work",
    ):
        (run / relative).mkdir(parents=True, exist_ok=True)
    publish_immutable_json(run / "protocol/protocol.json", protocol.as_record())


def bootstrap_macro_token(
    config_path: str | Path = DEFAULT_MACRO_TOKEN_CONFIG,
) -> MacroTokenBootstrap:
    """Authenticate the promoted hierarchy and prepare the isolated v8 run."""
    resolved = Path(config_path).resolve()
    project_root = resolved.parents[3]
    config = load_macro_token_config(resolved)
    current_source = bootstrap_promoted_integrator(config.source_config)
    source_store = IntegratorStore(config.source_artifact_root, config.source_run_hash)
    source_protocol_path = source_store.run / "protocol/protocol.json"
    stored_protocol = _stored_integrator_protocol(
        load_canonical_json(source_protocol_path)
    )
    source = replace(
        current_source,
        integrator=replace(
            current_source.integrator,
            protocol=stored_protocol,
            store=source_store,
            code_manifest=load_canonical_json(
                source_store.run / "protocol/code_manifest.json"
            ),
            environment_manifest=load_canonical_json(
                source_store.run / "protocol/environment_manifest.json"
            ),
        ),
    )
    source_paths = {
        "source_protocol": source_protocol_path,
        "fit_hierarchy_008": source_store.run
        / "hierarchies"
        / config.fit_hierarchy_policy_hash
        / "complete_008.json",
        "all_train_hierarchy_050": source_store.run
        / "hierarchies"
        / config.all_train_hierarchy_policy_hash
        / "complete_050.json",
        "stage_matched_joint": source_store.run
        / "evaluations/stage_matched_joint_iid.json",
        "source_locked_test": source_store.run / "evaluations/locked_test.json",
    }
    observed = tuple(file_sha256(path) for path in source_paths.values())
    expected = (
        config.source_protocol_sha256,
        config.fit_hierarchy_008_sha256,
        config.all_train_hierarchy_050_sha256,
        config.stage_matched_joint_sha256,
        config.source_locked_test_sha256,
    )
    if observed != expected or stored_protocol.content_hash != config.source_run_hash:
        raise ValueError("configured macro-token source artifacts changed")
    code = material_tree_manifest(_material_paths(project_root, resolved))
    environment = installed_environment_manifest(PROMOTED_PACKAGES)
    missing = tuple(
        str(row["name"])
        for row in environment["packages"]
        if row["version"] == "MISSING"
    )
    if missing:
        raise RuntimeError(f"isolated vision environment is incomplete: {missing}")
    protocol = MacroTokenProtocol(
        config.source_run_hash,
        config.source_protocol_sha256,
        source.integrator.manifest.content_hash,
        source.integrator.protocol.model_manifest_hash,
        source.integrator.split.content_hash,
        config.config_hash,
        str(code["content_hash"]),
        str(environment["content_hash"]),
    )
    run = config.artifact_root / "runs" / protocol.content_hash
    _prepare_run(run, protocol)
    publish_immutable_bytes(
        run / "config_resolved.json", canonical_json_bytes(config.as_record())
    )
    for filename, record in (
        ("code_manifest.json", code),
        ("environment_manifest.json", environment),
        ("source_protocol.json", load_canonical_json(source_protocol_path)),
        ("router_split.json", source.integrator.split.as_record()),
    ):
        publish_immutable_json(run / "protocol" / filename, record)
    for filename, source_path in (
        ("stage_matched_joint_iid.json", source_paths["stage_matched_joint"]),
        ("source_locked_test.json", source_paths["source_locked_test"]),
    ):
        publish_immutable_bytes(run / "evaluations" / filename, source_path.read_bytes())
    atomic_write(
        config.artifact_root / "LATEST_RUN.json",
        canonical_json_bytes(
            {
                "run_hash": protocol.content_hash,
                "schema_version": "imagenetr50-macro-token-latest-v1",
            }
        ),
    )
    atomic_write(
        current_source.promotion.artifact_root / "LATEST_RUN.json",
        canonical_json_bytes(
            {
                "run_hash": config.source_run_hash,
                "schema_version": "imagenetr50-promoted-integrator-latest-v1",
            }
        ),
    )
    return MacroTokenBootstrap(project_root, config, source, protocol, run)


def _write_state(
    bootstrap: MacroTokenBootstrap, phase: str, **values: object
) -> None:
    atomic_write(
        bootstrap.run / "state/workflow.json",
        canonical_json_bytes(
            {
                "phase": phase,
                "run_hash": bootstrap.protocol.content_hash,
                "schema_version": "imagenetr50-macro-token-workflow-state-v1",
                **values,
            }
        ),
    )


def _preflight(
    bootstrap: MacroTokenBootstrap, device: torch.device
) -> dict[str, object]:
    target = bootstrap.run / "protocol/preflight.json"
    if target.is_file():
        return load_canonical_json(target)
    config = bootstrap.config
    integrator = bootstrap.source.integrator
    fit_ids = frozenset(integrator.split.fit_image_ids)
    validation_ids = frozenset(integrator.split.validation_image_ids)
    test_ids = frozenset(
        row.image_id for row in integrator.manifest.images if row.split == "test"
    )
    training_ids = frozenset(
        row.image_id for row in integrator.manifest.images if row.split == "train"
    )
    sample_rows = tuple(
        row for row in integrator.manifest.images if row.split == "train"
    )[:2]
    loader = DataLoader(
        ManifestDataset(
            integrator.config.data_root / "imagenet-r",
            sample_rows,
            integrator.test_transform,
            0,
            0,
        ),
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )
    images, _labels, _ids = next(iter(loader))
    adapter_model = AdapterVisionModel(
        create_pinned_backbone(integrator.checkpoint), (0, 1, 2, 3), 16, 16, 0.0, 0
    ).to(device).eval()
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        tokens = adapter_model.token_sequence(images.to(device))
        features = adapter_model.backbone.forward_head(tokens, pre_logits=True)
        raw_local = adapter_model.classifier(features)
    token_cls_error = float((tokens[:, 0].float() - features.float()).abs().max())
    raw = torch.zeros(2, MAXIMUM_SLOTS, CLASS_COUNT)
    raw[:, 0, :4] = raw_local.float().cpu()
    ownership = torch.zeros(MAXIMUM_SLOTS, CLASS_COUNT, dtype=torch.bool)
    ownership[0, :4] = True
    active = ownership.any(dim=1)
    inputs = MacroTokenInputs(
        F.layer_norm(tokens.float(), (TOKEN_DIMENSION,), eps=1e-5).cpu()[:, None],
        torch.tensor((0,), dtype=torch.int64),
        behavior_meta_features(raw, ownership, active),
        raw,
        ownership,
        active,
        ownership.any(dim=0),
    )
    macro = MacroTokenClassifier(1, config.macro_optimization.dropout, config.seed).to(device)
    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        loss = F.cross_entropy(macro(inputs.to(device)), torch.tensor((0, 1), device=device))
    loss.backward()
    one = MacroTokenClassifier(1, config.macro_optimization.dropout, config.seed)
    two = MacroTokenClassifier(2, config.macro_optimization.dropout, config.seed)
    shared = set(one.state_dict()) & set(two.state_dict())
    shared_equal = all(
        torch.equal(one.state_dict()[name], two.state_dict()[name]) for name in shared
    )
    core: dict[str, object] = {
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(device),
        "effective_batch_size": config.macro_optimization.effective_batch_size,
        "fit_images": len(fit_ids),
        "macro_depth1_parameters": parameter_count(one),
        "macro_depth2_parameters": parameter_count(two),
        "one_step_loss": float(loss.detach()),
        "schema_version": "imagenetr50-macro-token-preflight-v1",
        "shared_initial_parameters_equal": shared_equal,
        "test_images": len(test_ids),
        "test_train_overlap": len(test_ids & training_ids),
        "token_cls_max_error": token_cls_error,
        "token_shape": list(tokens.shape),
        "validation_fit_overlap": len(validation_ids & fit_ids),
        "validation_images": len(validation_ids),
    }
    if (
        not core["bf16_supported"]
        or core["fit_images"] != 19_200
        or core["validation_images"] != 4_800
        or core["test_images"] != 6_000
        or core["test_train_overlap"] != 0
        or core["validation_fit_overlap"] != 0
        or core["token_shape"] != [2, TOKEN_COUNT, TOKEN_DIMENSION]
        or token_cls_error != 0.0
        or core["macro_depth1_parameters"] != 12_055_496
        or core["macro_depth2_parameters"] != 19_143_368
        or not shared_equal
        or not math.isfinite(float(core["one_step_loss"]))
    ):
        raise RuntimeError("macro-token preflight failed")
    record = {**core, "content_hash": record_sha256(core)}
    publish_immutable_json(target, record)
    del adapter_model, macro, one, two
    torch.cuda.empty_cache()
    return record


def _population(
    bootstrap: MacroTokenBootstrap,
    hierarchy: HierarchyBuildResult,
    partition: str,
    stage: int,
    rows: Sequence[ImageRecord],
    device: torch.device,
) -> MacroTokenPopulation:
    nodes, slots, frontier_hash = _hierarchy_frontier(hierarchy, stage)
    return materialize_macro_population(
        protocol_hash=bootstrap.protocol.content_hash,
        frontier_hash=frontier_hash,
        partition=partition,
        nodes=nodes,
        slot_indices=slots,
        rows=rows,
        prepared_root=bootstrap.source.integrator.config.data_root / "imagenet-r",
        transform=bootstrap.source.integrator.test_transform,
        transform_hash=test_transform_hash(
            bootstrap.source.integrator.primary_config.input_size
        ),
        model_hash=bootstrap.source.integrator.protocol.model_manifest_hash,
        backbone_factory=lambda: create_pinned_backbone(
            bootstrap.source.integrator.checkpoint
        ),
        scratch_root=bootstrap.scratch_root,
        request_ledger=ChainedJsonlLedger(
            bootstrap.run / "ledgers/macro_token_requests.jsonl",
            "imagenetr50-macro-token-request-v1",
        ),
        rank=bootstrap.source.integrator.primary_config.lora_rank,
        alpha=bootstrap.source.integrator.primary_config.lora_alpha,
        shard_size=bootstrap.config.cache_shard_size,
        batch_size=bootstrap.config.feature_batch_size,
        num_workers=bootstrap.config.num_workers,
        cache_limit_bytes=bootstrap.config.cache_limit_bytes,
        device=device,
    )


def _artifact_entry(
    bootstrap: MacroTokenBootstrap,
    path: Path,
    spec: MacroModelSpec,
    fit: MacroFitResult,
    metrics: Mapping[str, PopulationMetrics] | None = None,
) -> dict[str, object]:
    return {
        "artifact": str(path.relative_to(bootstrap.run)),
        "artifact_sha256": validate_artifact_directory(path),
        "fit": asdict(fit),
        "metrics": {
            name: value.as_record() for name, value in (metrics or {}).items()
        },
        "spec": spec.as_record(),
    }


def _fit_or_load(
    bootstrap: MacroTokenBootstrap,
    *,
    family: str,
    phase: str,
    stage: int,
    spec: MacroModelSpec,
    training: MacroTokenPopulation,
    validation: MacroTokenPopulation | None,
    fixed_epochs: int | None,
    device: torch.device,
) -> tuple[nn.Module, MacroFitResult, Path, bool]:
    job_hash = model_job_hash(
        protocol_hash=bootstrap.protocol.content_hash,
        phase=phase,
        stage=stage,
        spec=spec,
        fit_population_hash=training.identity,
        validation_population_hash=None if validation is None else validation.identity,
        fixed_epochs=fixed_epochs,
    )
    target = bootstrap.run / "models" / family / job_hash
    factory = lambda: create_trainable_model(spec, bootstrap.config, torch.device("cpu"))
    if target.is_dir():
        model, fit, _record = load_fitted_model(target, factory, spec)
        return model.to(device), fit, target, True
    model = create_trainable_model(spec, bootstrap.config, device)
    fit = fit_model(
        model=model,
        spec=spec,
        training=training,
        validation=validation,
        optimization=bootstrap.config.macro_optimization,
        fixed_epochs=fixed_epochs,
        checkpoint_path=bootstrap.run / "checkpoints" / f"{job_hash}.pt",
        checkpoint_key=job_hash,
        device=device,
    )
    target = publish_fitted_model(
        run_root=bootstrap.run,
        family=family,
        job_hash=job_hash,
        model=model,
        spec=spec,
        fit=fit,
        metadata={
            "fit_population_hash": training.identity,
            "fixed_epochs": fixed_epochs,
            "frontier_hash": training.frontier_hash,
            "phase": phase,
            "stage": stage,
            "validation_population_hash": None if validation is None else validation.identity,
        },
    )
    checkpoint = bootstrap.run / "checkpoints" / f"{job_hash}.pt"
    if checkpoint.is_file():
        checkpoint.unlink()
    return model, fit, target, False


def _fit_or_load_control(
    bootstrap: MacroTokenBootstrap,
    *,
    family: str,
    phase: str,
    stage: int,
    spec: MacroModelSpec,
    training: CompactControlPopulation,
    validation: CompactControlPopulation | None,
    fixed_epochs: int | None,
    device: torch.device,
) -> tuple[nn.Module, MacroFitResult, Path, bool]:
    """Fit or restore the v6 control from its compact data-matched projection."""
    if spec.kind != "v6_control":
        raise ValueError("compact control helper requires the v6 model kind")
    job_hash = model_job_hash(
        protocol_hash=bootstrap.protocol.content_hash,
        phase=phase,
        stage=stage,
        spec=spec,
        fit_population_hash=training.identity,
        validation_population_hash=None if validation is None else validation.identity,
        fixed_epochs=fixed_epochs,
    )
    target = bootstrap.run / "models" / family / job_hash
    factory = lambda: create_trainable_model(spec, bootstrap.config, torch.device("cpu"))
    if target.is_dir():
        model, fit, _record = load_fitted_model(target, factory, spec)
        return model.to(device), fit, target, True
    model = create_trainable_model(spec, bootstrap.config, device)
    if not isinstance(model, ImageNetResidualIntegrator):
        raise TypeError("v6 control construction returned the wrong model")
    fit = fit_compact_control(
        model=model,
        spec=spec,
        training=training,
        validation=validation,
        optimization=bootstrap.config.macro_optimization,
        batch_size=bootstrap.config.control_batch_size,
        fixed_epochs=fixed_epochs,
        checkpoint_path=bootstrap.run / "checkpoints" / f"{job_hash}.pt",
        checkpoint_key=job_hash,
        device=device,
    )
    target = publish_fitted_model(
        run_root=bootstrap.run,
        family=family,
        job_hash=job_hash,
        model=model,
        spec=spec,
        fit=fit,
        metadata={
            "fit_population_hash": training.identity,
            "fixed_epochs": fixed_epochs,
            "phase": phase,
            "stage": stage,
            "validation_population_hash": None if validation is None else validation.identity,
        },
    )
    checkpoint = bootstrap.run / "checkpoints" / f"{job_hash}.pt"
    if checkpoint.is_file():
        checkpoint.unlink()
    return model, fit, target, False


def _probe_job_hash(
    bootstrap: MacroTokenBootstrap,
    phase: str,
    stage: int,
    training: MacroTokenPopulation,
    validation: MacroTokenPopulation | None,
    fixed_epochs: int | None,
    source_sha256: str,
) -> str:
    return record_sha256(
        {
            "fit_population_hash": training.identity,
            "fixed_epochs": fixed_epochs,
            "phase": phase,
            "protocol_hash": bootstrap.protocol.content_hash,
            "schema_version": "imagenetr50-frozen-owner-probe-job-v1",
            "seed": bootstrap.config.owner_probe_seed,
            "source_artifact_sha256": source_sha256,
            "stage": stage,
            "validation_population_hash": None if validation is None else validation.identity,
        }
    )


def _fit_or_load_probe(
    bootstrap: MacroTokenBootstrap,
    *,
    phase: str,
    stage: int,
    source_model: MacroTokenClassifier,
    source_path: Path,
    training: MacroTokenPopulation,
    validation: MacroTokenPopulation | None,
    fixed_epochs: int | None,
    device: torch.device,
) -> tuple[nn.Linear, MacroFitResult, Path, bool]:
    source_sha256 = validate_artifact_directory(source_path)
    job_hash = _probe_job_hash(
        bootstrap,
        phase,
        stage,
        training,
        validation,
        fixed_epochs,
        source_sha256,
    )
    target = bootstrap.run / "models/frozen_owner_probe" / job_hash
    if target.is_dir():
        probe, fit, _record = load_frozen_owner_probe(target, device)
        return probe, fit, target, True
    config = bootstrap.config
    probe, fit = fit_frozen_owner_probe(
        source=source_model,
        training=training,
        validation=validation,
        learning_rate=config.control_learning_rate,
        weight_decay=config.control_weight_decay,
        batch_size=config.control_batch_size,
        minimum_epochs=config.macro_optimization.minimum_epochs,
        maximum_epochs=config.macro_optimization.maximum_epochs,
        patience=config.macro_optimization.patience,
        improvement_delta=config.macro_optimization.improvement_delta,
        fixed_epochs=fixed_epochs,
        seed=config.owner_probe_seed,
        checkpoint_path=bootstrap.run / "checkpoints" / f"probe_{job_hash}.pt",
        checkpoint_key=job_hash,
        device=device,
    )
    target = publish_frozen_owner_probe(
        run_root=bootstrap.run,
        job_hash=job_hash,
        probe=probe,
        fit=fit,
        metadata={
            "fit_population_hash": training.identity,
            "fixed_epochs": fixed_epochs,
            "phase": phase,
            "seed": config.owner_probe_seed,
            "source_artifact_sha256": source_sha256,
            "source_artifact": str(source_path.relative_to(bootstrap.run)),
            "stage": stage,
            "validation_population_hash": None if validation is None else validation.identity,
        },
    )
    checkpoint = bootstrap.run / "checkpoints" / f"probe_{job_hash}.pt"
    if checkpoint.is_file():
        checkpoint.unlink()
    return probe, fit, target, False


def _run_smoke(
    bootstrap: MacroTokenBootstrap,
    hierarchy: HierarchyBuildResult,
    device: torch.device,
) -> dict[str, object]:
    target = bootstrap.run / "evaluations/smoke.json"
    if target.is_file():
        return load_canonical_json(target)
    stage = 7
    fit_rows = _partition_rows(
        bootstrap.source.integrator, "fit", tuple(range(stage))
    )[:256]
    validation_rows = _partition_rows(
        bootstrap.source.integrator, "validation", tuple(range(stage))
    )[:128]
    fitting = _population(bootstrap, hierarchy, "fit", stage, fit_rows, device)
    validation = _population(
        bootstrap, hierarchy, "validation", stage, validation_rows, device
    )
    specs = (
        MacroModelSpec("macro_classifier", 1, 0.0003, bootstrap.config.seed),
        MacroModelSpec("macro_classifier", 2, 0.0003, bootstrap.config.seed),
        MacroModelSpec("owner_end_to_end", 1, 0.0003, bootstrap.config.seed),
        MacroModelSpec("v6_control", 1, bootstrap.config.control_learning_rate, bootstrap.config.seed),
    )
    rows: list[dict[str, object]] = []
    for index, spec in enumerate(specs):
        model = create_trainable_model(spec, bootstrap.config, device)
        checkpoint = bootstrap.run / "work" / f"smoke_{index}.pt"
        fit = fit_model(
            model=model,
            spec=spec,
            training=fitting,
            validation=None,
            optimization=bootstrap.config.macro_optimization,
            fixed_epochs=1,
            checkpoint_path=checkpoint,
            checkpoint_key=f"smoke-{index}",
            device=device,
            progress=False,
        )
        metrics = evaluate_population(model, validation, spec.kind, device)
        rows.append(
            {
                "fit": asdict(fit),
                "metrics": metrics.as_record(),
                "parameter_count": parameter_count(model),
                "spec": spec.as_record(),
            }
        )
        checkpoint.unlink(missing_ok=True)
        del model
        torch.cuda.empty_cache()
    replayed = _population(bootstrap, hierarchy, "fit", stage, fit_rows, device)
    request_ledger = ChainedJsonlLedger(
        bootstrap.run / "ledgers/macro_token_requests.jsonl",
        "imagenetr50-macro-token-request-v1",
    )
    checks = {
        "all_models_finite": all(
            math.isfinite(float(row["metrics"]["nll"])) for row in rows  # type: ignore[index]
        ),
        "cache_replay_zero_forwards": replayed.cache_misses == 0,
        "fit_hierarchy_reused": hierarchy.work.leaf_optimizer_steps == 0
        and hierarchy.work.parent_optimizer_steps == 0,
        "multi_node_frontier": len(hierarchy.frontier(stage)) == 3,
        "no_test_requests": not any(
            "test" in row["splits"] for row in request_ledger.rows
        ),
    }
    core: dict[str, object] = {
        "acceptance": checks,
        "integrity_passed": all(checks.values()),
        "models": rows,
        "schema_version": "imagenetr50-macro-token-smoke-v1",
        "stage": stage,
    }
    record = {**core, "content_hash": record_sha256(core)}
    publish_immutable_json(target, record)
    clear_macro_population(fitting, bootstrap.scratch_root)
    clear_macro_population(validation, bootstrap.scratch_root)
    return record


def _clean_candidate_rows(
    bootstrap: MacroTokenBootstrap,
    hierarchies: Mapping[int, tuple[MacroTokenPopulation, MacroTokenPopulation]],
    device: torch.device,
) -> tuple[dict[str, object], ...]:
    ledger = ChainedJsonlLedger(
        bootstrap.run / "evaluations/clean_candidates.jsonl",
        "imagenetr50-macro-token-clean-candidate-v1",
    )
    completed = {
        (int(row["stage"]), int(row["depth"]), float(row["learning_rate"]))
        for row in ledger.rows
    }
    for stage in bootstrap.config.stages:
        fitting, validation = hierarchies[stage]
        for depth, learning_rate in bootstrap.config.candidates:
            key = (stage, depth, learning_rate)
            if key in completed:
                continue
            spec = MacroModelSpec(
                "macro_classifier", depth, learning_rate, bootstrap.config.seed
            )
            model, fit, path, _reused = _fit_or_load(
                bootstrap,
                family="clean_candidates",
                phase="clean_selection",
                stage=stage,
                spec=spec,
                training=fitting,
                validation=validation,
                fixed_epochs=None,
                device=device,
            )
            if fit.validation_nll is None or fit.validation_accuracy is None:
                raise RuntimeError("clean macro candidate lacks validation metrics")
            ledger.append(
                {
                    "artifact": str(path.relative_to(bootstrap.run)),
                    "artifact_sha256": validate_artifact_directory(path),
                    "best_epoch": fit.best_epoch,
                    "depth": depth,
                    "learning_rate": learning_rate,
                    "parameter_count": parameter_count(model),
                    "seed": spec.seed,
                    "stage": stage,
                    "train_accuracy": fit.train_accuracy,
                    "train_nll": fit.train_nll,
                    "validation_accuracy": fit.validation_accuracy,
                    "validation_nll": fit.validation_nll,
                    "wall_seconds": fit.wall_seconds,
                }
            )
            del model
            torch.cuda.empty_cache()
    ledger.require_unique_keys(("stage", "depth", "learning_rate"))
    return tuple(dict(row) for row in ledger.rows)


def _select_architecture(
    bootstrap: MacroTokenBootstrap, candidates: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    target = bootstrap.run / "protocol/architecture_selection.json"
    means = tuple(
        {
            "depth": depth,
            "learning_rate": learning_rate,
            "mean_validation_nll": math.fsum(
                float(row["validation_nll"])
                for row in candidates
                if int(row["depth"]) == depth
                and float(row["learning_rate"]) == learning_rate
            )
            / len(bootstrap.config.stages),
        }
        for depth, learning_rate in bootstrap.config.candidates
    )
    winner = min(
        means,
        key=lambda row: (
            float(row["mean_validation_nll"]),
            int(row["depth"]),
            float(row["learning_rate"]),
        ),
    )
    core: dict[str, object] = {
        "aggregate_candidates": list(means),
        "criterion": "lowest_mean_validation_nll_then_depth_then_learning_rate",
        "schema_version": "imagenetr50-macro-token-architecture-selection-v1",
        "seed": bootstrap.config.seed,
        "stages": list(bootstrap.config.stages),
        "winner": winner,
    }
    record = {**core, "content_hash": record_sha256(core)}
    if target.is_file():
        if load_canonical_json(target) != record:
            raise ValueError("macro-token architecture selection changed")
    else:
        publish_immutable_json(target, record)
    return record


def _load_model_entry(
    bootstrap: MacroTokenBootstrap,
    entry: Mapping[str, object],
    device: torch.device,
) -> tuple[nn.Module, MacroFitResult]:
    spec = MacroModelSpec(**dict(entry["spec"]))
    path = bootstrap.run / str(entry["artifact"])
    model, fit, _record = load_fitted_model(
        path,
        lambda: create_trainable_model(spec, bootstrap.config, torch.device("cpu")),
        spec,
    )
    return model.to(device), fit


def _run_clean_stage(
    bootstrap: MacroTokenBootstrap,
    stage: int,
    fitting: MacroTokenPopulation,
    validation: MacroTokenPopulation,
    selection: Mapping[str, object],
    device: torch.device,
) -> dict[str, object]:
    target = bootstrap.run / "evaluations/clean" / f"stage_{stage:03d}.json"
    if target.is_file():
        return load_canonical_json(target)
    winner = dict(selection["winner"])
    depth, learning_rate = int(winner["depth"]), float(winner["learning_rate"])
    macro_entries: list[dict[str, object]] = []
    source_model: MacroTokenClassifier | None = None
    source_path: Path | None = None
    for seed in bootstrap.config.replication_seeds:
        spec = MacroModelSpec("macro_classifier", depth, learning_rate, seed)
        model, fit, path, _reused = _fit_or_load(
            bootstrap,
            family="clean_candidates" if seed == bootstrap.config.seed else "clean_replications",
            phase="clean_selection",
            stage=stage,
            spec=spec,
            training=fitting,
            validation=validation,
            fixed_epochs=None,
            device=device,
        )
        metrics = {
            "fit": evaluate_population(model, fitting, spec.kind, device),
            "validation": evaluate_population(model, validation, spec.kind, device),
        }
        macro_entries.append(_artifact_entry(bootstrap, path, spec, fit, metrics))
        if seed == bootstrap.config.owner_probe_seed:
            if not isinstance(model, MacroTokenClassifier):
                raise TypeError("selected macro source has the wrong type")
            source_model, source_path = model, path
        else:
            del model
            torch.cuda.empty_cache()
    control_entries: list[dict[str, object]] = []
    compact_fitting = compact_control_population(fitting)
    compact_validation = compact_control_population(validation)
    for seed in bootstrap.config.replication_seeds:
        spec = MacroModelSpec(
            "v6_control", 1, bootstrap.config.control_learning_rate, seed
        )
        model, fit, path, _reused = _fit_or_load_control(
            bootstrap,
            family="clean_v6_control",
            phase="clean_control",
            stage=stage,
            spec=spec,
            training=compact_fitting,
            validation=compact_validation,
            fixed_epochs=None,
            device=device,
        )
        if not isinstance(model, ImageNetResidualIntegrator):
            raise TypeError("loaded v6 control has the wrong type")
        control_entries.append(
            _artifact_entry(
                bootstrap,
                path,
                spec,
                fit,
                {
                    "fit": evaluate_compact_control(
                        model, compact_fitting, bootstrap.config.control_batch_size, device
                    ),
                    "validation": evaluate_compact_control(
                        model,
                        compact_validation,
                        bootstrap.config.control_batch_size,
                        device,
                    ),
                },
            )
        )
        del model
        torch.cuda.empty_cache()
    del compact_fitting, compact_validation
    owner_spec = MacroModelSpec(
        "owner_end_to_end",
        depth,
        learning_rate,
        bootstrap.config.owner_probe_seed,
    )
    owner_model, owner_fit, owner_path, _reused = _fit_or_load(
        bootstrap,
        family="clean_owner_end_to_end",
        phase="clean_owner",
        stage=stage,
        spec=owner_spec,
        training=fitting,
        validation=validation,
        fixed_epochs=None,
        device=device,
    )
    owner_entry = _artifact_entry(
        bootstrap,
        owner_path,
        owner_spec,
        owner_fit,
        {
            "fit": evaluate_population(owner_model, fitting, owner_spec.kind, device),
            "validation": evaluate_population(
                owner_model, validation, owner_spec.kind, device
            ),
        },
    )
    if source_model is None or source_path is None:
        raise RuntimeError("clean owner probe source model is unavailable")
    probe, probe_fit, probe_path, _reused = _fit_or_load_probe(
        bootstrap,
        phase="clean_owner_probe",
        stage=stage,
        source_model=source_model,
        source_path=source_path,
        training=fitting,
        validation=validation,
        fixed_epochs=None,
        device=device,
    )
    probe_entry = {
        "artifact": str(probe_path.relative_to(bootstrap.run)),
        "artifact_sha256": validate_artifact_directory(probe_path),
        "fit": asdict(probe_fit),
        "metrics": {
            "fit": evaluate_frozen_owner_probe(
                source_model, probe, fitting, device
            ).as_record(),
            "validation": evaluate_frozen_owner_probe(
                source_model, probe, validation, device
            ).as_record(),
        },
        "seed": bootstrap.config.owner_probe_seed,
        "source_artifact": str(source_path.relative_to(bootstrap.run)),
    }
    core: dict[str, object] = {
        "controls": {
            "fit": evaluate_frontier_controls(fitting),
            "validation": evaluate_frontier_controls(validation),
        },
        "fitting_population": fitting.identity,
        "frozen_owner_probe": probe_entry,
        "macro_models": macro_entries,
        "owner_end_to_end": owner_entry,
        "schema_version": "imagenetr50-macro-token-clean-stage-v1",
        "stage": stage,
        "v6_controls": control_entries,
        "validation_population": validation.identity,
        "winner": winner,
    }
    record = {**core, "content_hash": record_sha256(core)}
    publish_immutable_json(target, record)
    del source_model, owner_model, probe
    torch.cuda.empty_cache()
    return record


def _clean_experiment(
    bootstrap: MacroTokenBootstrap,
    hierarchy: HierarchyBuildResult,
    device: torch.device,
) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    selection_path = bootstrap.run / "protocol/architecture_selection.json"
    retained: dict[int, tuple[MacroTokenPopulation, MacroTokenPopulation]] = {}
    if selection_path.is_file():
        selection = load_canonical_json(selection_path)
    else:
        for stage in bootstrap.config.stages:
            retained[stage] = (
                _population(
                    bootstrap,
                    hierarchy,
                    "fit",
                    stage,
                    _partition_rows(
                        bootstrap.source.integrator, "fit", tuple(range(stage))
                    ),
                    device,
                ),
                _population(
                    bootstrap,
                    hierarchy,
                    "validation",
                    stage,
                    _partition_rows(
                        bootstrap.source.integrator,
                        "validation",
                        tuple(range(stage)),
                    ),
                    device,
                ),
            )
        selection = _select_architecture(
            bootstrap, _clean_candidate_rows(bootstrap, retained, device)
        )
    clean_records: list[dict[str, object]] = []
    for stage in bootstrap.config.stages:
        target = bootstrap.run / "evaluations/clean" / f"stage_{stage:03d}.json"
        if target.is_file():
            record = load_canonical_json(target)
            for identity in (
                str(record["fitting_population"]),
                str(record["validation_population"]),
            ):
                clear_macro_population_identity(identity, bootstrap.scratch_root)
        else:
            if stage not in retained:
                retained[stage] = (
                    _population(
                        bootstrap,
                        hierarchy,
                        "fit",
                        stage,
                        _partition_rows(
                            bootstrap.source.integrator, "fit", tuple(range(stage))
                        ),
                        device,
                    ),
                    _population(
                        bootstrap,
                        hierarchy,
                        "validation",
                        stage,
                        _partition_rows(
                            bootstrap.source.integrator,
                            "validation",
                            tuple(range(stage)),
                        ),
                        device,
                    ),
                )
            fitting, validation = retained[stage]
            record = _run_clean_stage(
                bootstrap, stage, fitting, validation, selection, device
            )
            clear_macro_population(fitting, bootstrap.scratch_root)
            clear_macro_population(validation, bootstrap.scratch_root)
        clean_records.append(record)
    return selection, tuple(clean_records)


def _best_epoch(entries: Sequence[Mapping[str, object]], seed: int) -> int:
    entry = next(row for row in entries if int(dict(row["spec"])["seed"]) == seed)
    return int(dict(entry["fit"])["best_epoch"])


def _run_refit_stage(
    bootstrap: MacroTokenBootstrap,
    hierarchy: HierarchyBuildResult,
    stage: int,
    clean: Mapping[str, object],
    selection: Mapping[str, object],
    device: torch.device,
) -> dict[str, object]:
    target = bootstrap.run / "protocol/refits" / f"stage_{stage:03d}.json"
    if target.is_file():
        record = load_canonical_json(target)
        clear_macro_population_identity(
            str(record["training_population"]), bootstrap.scratch_root
        )
        return record
    training = _population(
        bootstrap,
        hierarchy,
        "all_train",
        stage,
        _partition_rows(
            bootstrap.source.integrator, "all_train", tuple(range(stage))
        ),
        device,
    )
    winner = dict(selection["winner"])
    depth, learning_rate = int(winner["depth"]), float(winner["learning_rate"])
    macro_entries: list[dict[str, object]] = []
    source_model: MacroTokenClassifier | None = None
    source_path: Path | None = None
    clean_macro = tuple(dict(row) for row in clean["macro_models"])
    for seed in bootstrap.config.replication_seeds:
        spec = MacroModelSpec("macro_classifier", depth, learning_rate, seed)
        model, fit, path, _reused = _fit_or_load(
            bootstrap,
            family="all_train_macro_refit",
            phase="all_train_refit",
            stage=stage,
            spec=spec,
            training=training,
            validation=None,
            fixed_epochs=_best_epoch(clean_macro, seed),
            device=device,
        )
        macro_entries.append(_artifact_entry(bootstrap, path, spec, fit))
        if seed == bootstrap.config.owner_probe_seed:
            if not isinstance(model, MacroTokenClassifier):
                raise TypeError("all-training macro source has wrong type")
            source_model, source_path = model, path
        else:
            del model
            torch.cuda.empty_cache()
    control_entries: list[dict[str, object]] = []
    clean_controls = tuple(dict(row) for row in clean["v6_controls"])
    compact_training = compact_control_population(training)
    for seed in bootstrap.config.replication_seeds:
        spec = MacroModelSpec(
            "v6_control", 1, bootstrap.config.control_learning_rate, seed
        )
        model, fit, path, _reused = _fit_or_load_control(
            bootstrap,
            family="all_train_v6_control_refit",
            phase="all_train_control_refit",
            stage=stage,
            spec=spec,
            training=compact_training,
            validation=None,
            fixed_epochs=_best_epoch(clean_controls, seed),
            device=device,
        )
        control_entries.append(_artifact_entry(bootstrap, path, spec, fit))
        del model
        torch.cuda.empty_cache()
    del compact_training
    clean_owner = dict(clean["owner_end_to_end"])
    owner_spec = MacroModelSpec(
        "owner_end_to_end",
        depth,
        learning_rate,
        bootstrap.config.owner_probe_seed,
    )
    owner_model, owner_fit, owner_path, _reused = _fit_or_load(
        bootstrap,
        family="all_train_owner_refit",
        phase="all_train_owner_refit",
        stage=stage,
        spec=owner_spec,
        training=training,
        validation=None,
        fixed_epochs=int(dict(clean_owner["fit"])["best_epoch"]),
        device=device,
    )
    owner_entry = _artifact_entry(
        bootstrap, owner_path, owner_spec, owner_fit
    )
    if source_model is None or source_path is None:
        raise RuntimeError("all-training probe source model is unavailable")
    clean_probe = dict(clean["frozen_owner_probe"])
    probe, probe_fit, probe_path, _reused = _fit_or_load_probe(
        bootstrap,
        phase="all_train_owner_probe_refit",
        stage=stage,
        source_model=source_model,
        source_path=source_path,
        training=training,
        validation=None,
        fixed_epochs=int(dict(clean_probe["fit"])["best_epoch"]),
        device=device,
    )
    core: dict[str, object] = {
        "frozen_owner_probe": {
            "artifact": str(probe_path.relative_to(bootstrap.run)),
            "artifact_sha256": validate_artifact_directory(probe_path),
            "fit": asdict(probe_fit),
            "seed": bootstrap.config.owner_probe_seed,
            "source_artifact": str(source_path.relative_to(bootstrap.run)),
        },
        "macro_models": macro_entries,
        "owner_end_to_end": owner_entry,
        "schema_version": "imagenetr50-macro-token-all-train-refit-stage-v1",
        "stage": stage,
        "training_population": training.identity,
        "v6_controls": control_entries,
        "winner": winner,
    }
    record = {**core, "content_hash": record_sha256(core)}
    publish_immutable_json(target, record)
    clear_macro_population(training, bootstrap.scratch_root)
    del source_model, owner_model, probe
    torch.cuda.empty_cache()
    return record


def _training_seal(
    bootstrap: MacroTokenBootstrap,
    all_train_hierarchy: HierarchyBuildResult,
    refits: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    target = bootstrap.run / "protocol/training_seal.json"
    if target.is_file():
        return load_canonical_json(target)
    requests = ChainedJsonlLedger(
        bootstrap.run / "ledgers/macro_token_requests.jsonl",
        "imagenetr50-macro-token-request-v1",
    )
    test_requests = sum("test" in row["splits"] for row in requests.rows)
    model_hashes = {
        f"stage{record['stage']}:{family}:{index}": str(entry["artifact_sha256"])
        for record in refits
        for family in ("macro_models", "v6_controls")
        for index, entry in enumerate(record[family])
    }
    model_hashes.update(
        {
            f"stage{record['stage']}:{family}": str(
                dict(record[family])["artifact_sha256"]
            )
            for record in refits
            for family in ("owner_end_to_end", "frozen_owner_probe")
        }
    )
    core: dict[str, object] = {
        "all_train_hierarchy_policy_hash": all_train_hierarchy.policy.content_hash,
        "model_artifact_hashes": model_hashes,
        "request_ledger_rows": len(requests.rows),
        "request_ledger_tail_hash": requests.tail_hash,
        "schema_version": "imagenetr50-macro-token-training-seal-v1",
        "test_requests_before_seal": test_requests,
    }
    if test_requests:
        raise RuntimeError("test token sequences were opened before all refits sealed")
    record = {**core, "content_hash": record_sha256(core)}
    publish_immutable_json(target, record)
    return record


def _source_joint_row(bootstrap: MacroTokenBootstrap, stage: int) -> dict[str, object]:
    record = load_canonical_json(
        bootstrap.run / "evaluations/stage_matched_joint_iid.json"
    )
    return dict(next(row for row in record["rows"] if int(row["stage"]) == stage))


def _locked_stage(
    bootstrap: MacroTokenBootstrap,
    hierarchy: HierarchyBuildResult,
    refit: Mapping[str, object],
    device: torch.device,
) -> dict[str, object]:
    stage = int(refit["stage"])
    target = bootstrap.run / "evaluations/locked_test" / f"stage_{stage:03d}.json"
    if target.is_file():
        record = load_canonical_json(target)
        clear_macro_population_identity(str(record["test_population"]), bootstrap.scratch_root)
        return record
    test = _population(
        bootstrap,
        hierarchy,
        "test",
        stage,
        _test_rows(bootstrap.source.integrator, tuple(range(stage))),
        device,
    )
    compact_test = compact_control_population(test)
    macro_results: list[dict[str, object]] = []
    source_model: MacroTokenClassifier | None = None
    for entry in refit["macro_models"]:
        loaded_entry = dict(entry)
        model, fit = _load_model_entry(bootstrap, loaded_entry, device)
        metrics = evaluate_population(model, test, "macro_classifier", device)
        result = {**loaded_entry, "fit": asdict(fit), "test": metrics.as_record()}
        macro_results.append(result)
        if int(dict(loaded_entry["spec"])["seed"]) == bootstrap.config.owner_probe_seed:
            if not isinstance(model, MacroTokenClassifier):
                raise TypeError("locked probe source has wrong type")
            source_model = model
        else:
            del model
            torch.cuda.empty_cache()
    control_results: list[dict[str, object]] = []
    for entry in refit["v6_controls"]:
        loaded_entry = dict(entry)
        model, fit = _load_model_entry(bootstrap, loaded_entry, device)
        if not isinstance(model, ImageNetResidualIntegrator):
            raise TypeError("locked v6 control has the wrong type")
        control_results.append(
            {
                **loaded_entry,
                "fit": asdict(fit),
                "test": evaluate_compact_control(
                    model, compact_test, bootstrap.config.control_batch_size, device
                ).as_record(),
            }
        )
        del model
        torch.cuda.empty_cache()
    owner_entry = dict(refit["owner_end_to_end"])
    owner_model, owner_fit = _load_model_entry(bootstrap, owner_entry, device)
    owner_result = {
        **owner_entry,
        "fit": asdict(owner_fit),
        "test": evaluate_population(
            owner_model, test, "owner_end_to_end", device
        ).as_record(),
    }
    if source_model is None:
        raise RuntimeError("locked frozen probe source model is unavailable")
    probe_entry = dict(refit["frozen_owner_probe"])
    probe, probe_fit, _record = load_frozen_owner_probe(
        bootstrap.run / str(probe_entry["artifact"]), device
    )
    probe_result = {
        **probe_entry,
        "fit": asdict(probe_fit),
        "test": evaluate_frozen_owner_probe(
            source_model, probe, test, device
        ).as_record(),
    }
    core: dict[str, object] = {
        "controls": evaluate_frontier_controls(test),
        "frozen_owner_probe": probe_result,
        "macro_models": macro_results,
        "owner_end_to_end": owner_result,
        "schema_version": "imagenetr50-macro-token-locked-stage-v1",
        "stage": stage,
        "stage_matched_joint_iid": _source_joint_row(bootstrap, stage),
        "test_population": test.identity,
        "v6_controls": control_results,
    }
    record = {**core, "content_hash": record_sha256(core)}
    publish_immutable_json(target, record)
    clear_macro_population(test, bootstrap.scratch_root)
    del source_model, owner_model, probe
    torch.cuda.empty_cache()
    return record


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def _publish_result(
    bootstrap: MacroTokenBootstrap,
    selection: Mapping[str, object],
    clean: Sequence[Mapping[str, object]],
    locked: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    target = bootstrap.run / "evaluations/result.json"
    if target.is_file():
        return load_canonical_json(target)
    source_locked = load_canonical_json(
        bootstrap.run / "evaluations/source_locked_test.json"
    )
    local_references = dict(source_locked["local_references"])
    stage_summaries = []
    task_rows = []
    for record in locked:
        stage = int(record["stage"])
        macro = tuple(
            float(dict(entry["test"])["accuracy"])
            for entry in record["macro_models"]
        )
        control = tuple(
            float(dict(entry["test"])["accuracy"])
            for entry in record["v6_controls"]
        )
        owner = dict(dict(record["owner_end_to_end"])["test"])
        probe = dict(dict(record["frozen_owner_probe"])["test"])
        controls = dict(record["controls"])
        joint = float(dict(record["stage_matched_joint_iid"])["accuracy"])
        stage_summaries.append(
            {
                "macro_mean_accuracy": _mean(macro),
                "macro_seed_accuracies": list(macro),
                "macro_minus_joint_iid": _mean(macro) - joint,
                "macro_minus_true_node_oracle": _mean(macro)
                - float(controls["true_node_oracle_accuracy"]),
                "local_e2_lora_incremental_accuracy": (
                    float(local_references["local_e2_incremental"])
                    if stage == 50
                    else None
                ),
                "owner_end_to_end_accuracy": owner["accuracy"],
                "owner_end_to_end_routed_accuracy": owner["owner_routed_accuracy"],
                "owner_probe_accuracy": probe["accuracy"],
                "owner_probe_routed_accuracy": probe["owner_routed_accuracy"],
                "raw_union_accuracy": controls["raw_union_accuracy"],
                "stage": stage,
                "stage_matched_joint_iid_accuracy": joint,
                "published_e2_lora_incremental_accuracy": (
                    float(local_references["published_e2_incremental"])
                    if stage == 50
                    else None
                ),
                "true_node_oracle_accuracy": controls["true_node_oracle_accuracy"],
                "v6_control_mean_accuracy": _mean(control),
                "v6_control_seed_accuracies": list(control),
            }
        )
        for family, entries in (
            ("macro_token", record["macro_models"]),
            ("v6_final_cls_behavior_mlp", record["v6_controls"]),
        ):
            for entry in entries:
                seed = int(dict(entry["spec"])["seed"])
                task_rows.extend(
                    {
                        "accuracy": float(value),
                        "condition": family,
                        "seed": seed,
                        "stage": stage,
                        "task": int(task),
                    }
                    for task, value in dict(dict(entry["test"])["task_accuracies"]).items()
                )
    requests = ChainedJsonlLedger(
        bootstrap.run / "ledgers/macro_token_requests.jsonl",
        "imagenetr50-macro-token-request-v1",
    )
    core: dict[str, object] = {
        "architecture_selection": dict(selection),
        "clean_stages": list(clean),
        "local_references": local_references,
        "request_accounting": {
            "cache_bytes_sum": sum(int(row["cache_bytes"]) for row in requests.rows),
            "cache_hits": sum(int(row["cache_hits"]) for row in requests.rows),
            "cache_misses": sum(int(row["cache_misses"]) for row in requests.rows),
            "node_example_forwards": sum(
                int(row["node_example_forwards"]) for row in requests.rows
            ),
            "requests": len(requests.rows),
        },
        "role": "clean_macro_token_ceiling_study_with_locked_refit",
        "schema_version": "imagenetr50-macro-token-result-v1",
        "stage_summaries": stage_summaries,
        "task_accuracy_matrix": task_rows,
    }
    result = {**core, "content_hash": record_sha256(core)}
    publish_immutable_json(target, result)
    return result


def _artifact_fingerprint(run: Path) -> dict[str, object]:
    paths = tuple(sorted(path for path in (run / "models").rglob("artifact.json")))
    return {
        "artifact_count": len(paths),
        "identities_hash": record_sha256(
            [
                {
                    "path": str(path.relative_to(run)),
                    "sha256": file_sha256(path),
                }
                for path in paths
            ]
        ),
    }


def _reuse_proof(
    bootstrap: MacroTokenBootstrap,
    clean_hierarchy: HierarchyBuildResult,
    all_train_hierarchy: HierarchyBuildResult,
    device: torch.device,
) -> dict[str, object]:
    target = bootstrap.run / "protocol/reuse_proof.json"
    if target.is_file():
        return load_canonical_json(target)
    ledger = ChainedJsonlLedger(
        bootstrap.run / "ledgers/macro_token_requests.jsonl",
        "imagenetr50-macro-token-request-v1",
    )
    rows_before = len(ledger.rows)
    models_before = _artifact_fingerprint(bootstrap.run)
    clean_rebuilt = _build(bootstrap.source, "fit", 50, device, progress=False)
    all_train_rebuilt = _build(
        bootstrap.source, "all_train", 50, device, progress=False
    )
    for stage in bootstrap.config.stages:
        load_canonical_json(
            bootstrap.run / "evaluations/clean" / f"stage_{stage:03d}.json"
        )
        load_canonical_json(
            bootstrap.run / "protocol/refits" / f"stage_{stage:03d}.json"
        )
        load_canonical_json(
            bootstrap.run / "evaluations/locked_test" / f"stage_{stage:03d}.json"
        )
    rows_after = len(
        ChainedJsonlLedger(
            bootstrap.run / "ledgers/macro_token_requests.jsonl",
            "imagenetr50-macro-token-request-v1",
        ).rows
    )
    models_after = _artifact_fingerprint(bootstrap.run)
    scratch_files = tuple(
        path for path in bootstrap.scratch_root.rglob("*") if path.is_file()
    )
    checks = {
        "all_train_hierarchy_zero_optimizer_steps": all_train_rebuilt.work.leaf_optimizer_steps == 0
        and all_train_rebuilt.work.parent_optimizer_steps == 0,
        "clean_hierarchy_zero_optimizer_steps": clean_rebuilt.work.leaf_optimizer_steps == 0
        and clean_rebuilt.work.parent_optimizer_steps == 0,
        "model_artifacts_unchanged": models_before == models_after,
        "no_new_token_requests": rows_before == rows_after,
        "scratch_caches_cleared": not scratch_files,
        "source_frontiers_unchanged": clean_rebuilt.snapshots == clean_hierarchy.snapshots
        and all_train_rebuilt.snapshots == all_train_hierarchy.snapshots,
    }
    core: dict[str, object] = {
        "acceptance": checks,
        "all_train_hierarchy_work": asdict(all_train_rebuilt.work),
        "clean_hierarchy_work": asdict(clean_rebuilt.work),
        "integrity_passed": all(checks.values()),
        "model_fingerprint_after": models_after,
        "model_fingerprint_before": models_before,
        "request_rows_after": rows_after,
        "request_rows_before": rows_before,
        "schema_version": "imagenetr50-macro-token-reuse-proof-v1",
    }
    record = {**core, "content_hash": record_sha256(core)}
    publish_immutable_json(target, record)
    return record


def run_macro_token(
    config_path: str | Path = DEFAULT_MACRO_TOKEN_CONFIG,
) -> Path:
    """Run or exactly resume the clean ceiling, locked refit, and report."""
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("the macro-token ceiling study requires BF16 CUDA")
    started = time.monotonic()
    bootstrap = bootstrap_macro_token(config_path)
    device = torch.device("cuda:0")
    from tqdm.auto import tqdm

    overall = tqdm(total=9, desc="ImageNet-R macro-token ceiling", unit="phase")
    print(f"Temporary/resumable artifact directory: {bootstrap.run}", flush=True)
    print("[phase 1/9] Real ViT and macro-head preflight", flush=True)
    _write_state(bootstrap, "PREFLIGHT")
    _preflight(bootstrap, device)
    overall.update(1)

    print("[phase 2/9] Artifact-backed multi-node smoke", flush=True)
    _write_state(bootstrap, "SMOKE")
    smoke_hierarchy = _build(bootstrap.source, "fit", 8, device, progress=False)
    smoke = _run_smoke(bootstrap, smoke_hierarchy, device)
    if not bool(smoke["integrity_passed"]):
        raise RuntimeError("macro-token smoke failed")
    overall.update(1)

    print("[phase 3/9] Complete the end-to-end clean 50-task hierarchy", flush=True)
    _write_state(bootstrap, "CLEAN_HIERARCHY")
    clean_hierarchy = _build(bootstrap.source, "fit", 50, device)
    if clean_hierarchy.policy.content_hash != bootstrap.config.fit_hierarchy_policy_hash:
        raise RuntimeError("clean hierarchy policy changed")
    clean_completion = (
        bootstrap.source.integrator.store.run
        / "hierarchies"
        / clean_hierarchy.policy.content_hash
        / "complete_050.json"
    )
    publish_immutable_json(
        bootstrap.run / "protocol/clean_hierarchy.json",
        {
            "complete_sha256": file_sha256(clean_completion),
            "node_hashes": sorted(node.artifact.content_hash for node in clean_hierarchy.nodes),
            "policy_hash": clean_hierarchy.policy.content_hash,
            "schema_version": "imagenetr50-macro-token-clean-hierarchy-link-v1",
        },
    )
    overall.update(1)

    print("[phase 4/9] Six-cell clean architecture sweep at stages 31 and 50", flush=True)
    _write_state(bootstrap, "CLEAN_SELECTION")
    selection, clean_records = _clean_experiment(
        bootstrap, clean_hierarchy, device
    )
    overall.update(1)

    print("[phase 5/9] Replications, v6 controls, and owner diagnostics", flush=True)
    print(
        "Clean stage records include seeds 1993-1995 and both owner probes; no additional pass is needed.",
        flush=True,
    )
    overall.update(1)

    print("[phase 6/9] Authenticate all-training hierarchy and refit selected models", flush=True)
    _write_state(bootstrap, "ALL_TRAIN_REFITS")
    all_train_hierarchy = _build(
        bootstrap.source, "all_train", 50, device, progress=False
    )
    all_completion = (
        bootstrap.source.integrator.store.run
        / "hierarchies"
        / all_train_hierarchy.policy.content_hash
        / "complete_050.json"
    )
    if (
        all_train_hierarchy.policy.content_hash
        != bootstrap.config.all_train_hierarchy_policy_hash
        or file_sha256(all_completion)
        != bootstrap.config.all_train_hierarchy_050_sha256
        or all_train_hierarchy.work.leaf_optimizer_steps
        or all_train_hierarchy.work.parent_optimizer_steps
    ):
        raise RuntimeError("all-training hierarchy was not reused exactly")
    refits = tuple(
        _run_refit_stage(
            bootstrap,
            all_train_hierarchy,
            stage,
            next(row for row in clean_records if int(row["stage"]) == stage),
            selection,
            device,
        )
        for stage in bootstrap.config.stages
    )
    seal = _training_seal(bootstrap, all_train_hierarchy, refits)
    overall.update(1)

    print("[phase 7/9] Open locked test once and evaluate selected conditions", flush=True)
    _write_state(bootstrap, "LOCKED_TEST", training_seal=seal)
    locked = tuple(
        _locked_stage(bootstrap, all_train_hierarchy, refit, device)
        for refit in refits
    )
    result = _publish_result(bootstrap, selection, clean_records, locked)
    overall.update(1)

    print("[phase 8/9] Prove zero-work hierarchy/model/token-cache reuse", flush=True)
    proof = _reuse_proof(
        bootstrap, clean_hierarchy, all_train_hierarchy, device
    )
    if not bool(proof["integrity_passed"]):
        raise RuntimeError("macro-token reuse proof failed")
    overall.update(1)

    print("[phase 9/9] Generate publication-style reports and compact ledgers", flush=True)
    _write_state(bootstrap, "REPORTING", result=result, reuse_proof=proof)
    from apm.continual.vision.imagenetr.macro_token_reporting import (
        write_macro_token_report,
    )

    write_macro_token_report(bootstrap.run)
    elapsed = time.monotonic() - started
    atomic_write(
        bootstrap.run / "state/last_invocation.json",
        canonical_json_bytes(
            {
                "elapsed_seconds": elapsed,
                "schema_version": "imagenetr50-macro-token-invocation-v1",
            }
        ),
    )
    _write_state(bootstrap, "COMPLETE", result=result, reuse_proof=proof)
    overall.update(1)
    overall.close()
    print(f"Macro-token report complete in {elapsed / 60:.1f} minutes.", flush=True)
    return bootstrap.run


if __name__ == "__main__":
    print(run_macro_token())


__all__ = [
    "MacroTokenBootstrap",
    "MacroTokenProtocol",
    "bootstrap_macro_token",
    "run_macro_token",
]

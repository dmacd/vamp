"""Full-50 persistent single-affine frontier and matched joint-IID controls."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Protocol
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import tempfile
import time

import torch
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
from apm.continual.vision.imagenetr.artifacts import (
    publish_artifact_directory,
    validate_artifact_directory,
)
from apm.continual.vision.imagenetr.behavior_replay_workflow import (
    _stat_fingerprint,
    _stored_integrator_protocol,
)
from apm.continual.vision.imagenetr.data import ImageRecord, ManifestDataset
from apm.continual.vision.imagenetr.checkpoints import atomic_torch_save
from apm.continual.vision.imagenetr.heads import save_classifier
from apm.continual.vision.imagenetr.integrator_artifacts import IntegratorStore
from apm.continual.vision.imagenetr.integrator_hierarchy import HierarchyBuildResult
from apm.continual.vision.imagenetr.integrator_observations import BehaviorNode
from apm.continual.vision.imagenetr.integrator_workflow import _hierarchy_frontier
from apm.continual.vision.imagenetr.lora import adapter_factors, save_adapter
from apm.continual.vision.imagenetr.manifests import installed_environment_manifest
from apm.continual.vision.imagenetr.model import AdapterVisionModel, create_pinned_backbone
from apm.continual.vision.imagenetr.persistent_affine_config import (
    DEFAULT_PERSISTENT_AFFINE_CONFIG,
    PersistentAffineConfig,
    load_persistent_affine_config,
)
from apm.continual.vision.imagenetr.persistent_affine_model import (
    PersistentFrontierIntegrator,
    carry_model_state,
    export_model_state,
    load_exact_model_state,
    raw_union_logits,
)
from apm.continual.vision.imagenetr.persistent_affine_training import (
    EPOCH_LEDGER_SCHEMA,
    PrefixEvaluation,
    create_optimizer,
    evaluate_prefix,
    export_named_optimizer_state,
    fit_online_stage,
    restore_named_optimizer_state,
    rotating_replay_population,
)
from apm.continual.vision.imagenetr.promoted_integrator_workflow import (
    PROMOTED_PACKAGES,
    PromotedBootstrap,
    _build,
    bootstrap_promoted_integrator,
)
from apm.continual.vision.imagenetr.promoted_integrator_config import (
    load_promoted_integrator_config,
)
from apm.continual.vision.imagenetr.protocol import material_tree_manifest
from apm.continual.vision.imagenetr.stage_matched_joint import validate_stage_rows
from apm.continual.vision.imagenetr.training import os_cpu_workers, train_adapter_model


ARM_LEDGER_SCHEMA = "imagenetr50-persistent-affine-stage-v1"
RANK_LEDGER_SCHEMA = "imagenetr50-rank-matched-joint-stage-v1"
RESULT_PATH = Path("evaluations/result.json")
LEDGER_FIELDS = frozenset({"format", "previous_sha256", "result_sha256", "sequence"})


@dataclass(frozen=True, slots=True)
class PersistentAffineProtocol:
    """Content identity for the online arms and both joint-IID references."""

    source_run_hash: str
    source_protocol_sha256: str
    source_hierarchy_policy_hash: str
    source_hierarchy_complete_sha256: str
    stage_matched_joint_sha256: str
    stage_matched_joint_content_hash: str
    selection_result_sha256: str
    h4096_result_hash: str
    h8192_result_hash: str
    dataset_manifest_hash: str
    model_manifest_hash: str
    split_hash: str
    config_hash: str
    code_manifest_hash: str
    environment_manifest_hash: str
    schema_version: str = "imagenetr50-persistent-affine-protocol-v1"

    def __post_init__(self) -> None:
        for label, identity in (
            ("source run", self.source_run_hash),
            ("source protocol", self.source_protocol_sha256),
            ("source hierarchy policy", self.source_hierarchy_policy_hash),
            ("source hierarchy completion", self.source_hierarchy_complete_sha256),
            ("stage-matched joint file", self.stage_matched_joint_sha256),
            ("stage-matched joint result", self.stage_matched_joint_content_hash),
            ("selection result", self.selection_result_sha256),
            ("H=4096 selection", self.h4096_result_hash),
            ("H=8192 selection", self.h8192_result_hash),
            ("dataset", self.dataset_manifest_hash),
            ("model", self.model_manifest_hash),
            ("split", self.split_hash),
            ("configuration", self.config_hash),
            ("code", self.code_manifest_hash),
            ("environment", self.environment_manifest_hash),
        ):
            require_sha256(identity, label)
        if self.schema_version != "imagenetr50-persistent-affine-protocol-v1":
            raise ValueError("persistent affine protocol schema changed")

    @property
    def content_hash(self) -> str:
        """Return the immutable experiment namespace."""
        return record_sha256(self.as_record(False))

    def as_record(self, include_hash: bool = True) -> dict[str, object]:
        """Return canonical JSON-compatible protocol fields."""
        core = asdict(self)
        return {**core, "content_hash": self.content_hash} if include_hash else core


@dataclass(frozen=True, slots=True)
class PersistentAffineBootstrap:
    """Authenticated source hierarchy, comparisons, and isolated output root."""

    project_root: Path
    config: PersistentAffineConfig
    source: PromotedBootstrap
    stage_matched_joint: dict[str, object]
    selection_result: dict[str, object]
    protocol: RunProtocol
    run: Path
    hidden_dimension: int = 0
    reference_rows: tuple[Mapping[str, object], ...] = ()


class RunProtocol(Protocol):
    """Minimal immutable identity shared by affine and MLP experiment runners."""

    @property
    def content_hash(self) -> str: ...

    def as_record(self) -> dict[str, object]: ...


@dataclass(frozen=True, slots=True)
class InvocationWork:
    """New expensive work performed by one resumable invocation."""

    affine_optimizer_steps: int = 0
    affine_stages: int = 0
    rank_optimizer_steps: int = 0
    rank_stages: int = 0

    def __add__(self, other: "InvocationWork") -> "InvocationWork":
        return InvocationWork(
            self.affine_optimizer_steps + other.affine_optimizer_steps,
            self.affine_stages + other.affine_stages,
            self.rank_optimizer_steps + other.rank_optimizer_steps,
            self.rank_stages + other.rank_stages,
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _material_paths(project_root: Path, config_path: Path) -> tuple[Path, ...]:
    package = project_root / "src/apm/continual/vision/imagenetr"
    return (
        config_path,
        project_root / "configs/vision/imagenetr/primary.yaml",
        project_root / "docs/imagenetr50_logt_persistent_affine_protocol.md",
        project_root / "scripts/vision/imagenetr/run_persistent_affine_local.sh",
        project_root / "src/apm/continual/artifacts.py",
        *(package / name for name in (
            "artifacts.py",
            "behavior_replay_workflow.py",
            "checkpoints.py",
            "config.py",
            "data.py",
            "frontier_adaptation_training.py",
            "heads.py",
            "integrator_artifacts.py",
            "integrator_bank.py",
            "integrator_config.py",
            "integrator_hierarchy.py",
            "integrator_observations.py",
            "integrator_workflow.py",
            "lora.py",
            "manifests.py",
            "model.py",
            "parent_recipe_factorial.py",
            "persistent_affine_config.py",
            "persistent_affine_model.py",
            "persistent_integrator_head.py",
            "persistent_affine_training.py",
            "persistent_affine_workflow.py",
            "promoted_integrator_config.py",
            "promoted_integrator_workflow.py",
            "protocol.py",
            "router_artifacts.py",
            "stage_matched_joint.py",
            "training.py",
        )),
        package / "merging/common.py",
    )


def _validated_content_record(path: Path, schema: str) -> dict[str, object]:
    record = load_canonical_json(path)
    core = {key: value for key, value in record.items() if key != "content_hash"}
    if record.get("schema_version") != schema or record.get("content_hash") != record_sha256(core):
        raise ValueError(f"content-addressed result changed: {path}")
    return record


def _selection_cells(
    config: PersistentAffineConfig, aggregate: Mapping[str, object]
) -> tuple[dict[str, object], dict[str, object]]:
    selected: list[dict[str, object]] = []
    expected = {
        4_096: (config.h4096_result_sha256, config.h4096_result_content_hash),
        8_192: (config.h8192_result_sha256, config.h8192_result_content_hash),
    }
    for capacity in config.historical_capacities:
        entry = next(
            dict(row)
            for row in aggregate["cells"]
            if row["cell"]["family"] == "single_affine"
            and int(row["cell"]["historical_capacity"]) == capacity
        )
        path = config.selection_run / str(entry["result"])
        result = _validated_content_record(
            path, "imagenetr50-frontier-architecture-replay-cell-v1"
        )
        file_hash, content_hash = expected[capacity]
        if (
            file_sha256(path) != file_hash
            or result["content_hash"] != content_hash
            or entry["result_sha256"] != file_hash
            or entry["result_content_hash"] != content_hash
            or int(result["fit"]["best_nll_epoch"])
            != dict(config.epochs_per_stage)[capacity]
        ):
            raise ValueError("selected affine H result differs from the promoted checkpoint")
        selected.append(result)
    return selected[0], selected[1]


def _prepare_run(run: Path, protocol: RunProtocol) -> None:
    for relative in (
        "protocol",
        "arms",
        "controls/rank_matched_joint/stages",
        "controls/rank_matched_joint/checkpoints",
        "evaluations",
        "reports",
        "state",
        "work",
    ):
        (run / relative).mkdir(parents=True, exist_ok=True)
    publish_immutable_json(run / "protocol/protocol.json", protocol.as_record())


def load_persistent_inputs(
    config_path: str | Path = DEFAULT_PERSISTENT_AFFINE_CONFIG,
) -> tuple[PersistentAffineConfig, PromotedBootstrap, dict[str, object], dict[str, object]]:
    """Authenticate frozen source models and recipes without creating an affine run."""
    resolved = Path(config_path).resolve()
    project_root = resolved.parents[3]
    config = load_persistent_affine_config(resolved)
    if file_sha256(config.source_config) != config.source_config_sha256:
        raise ValueError("promoted source configuration bytes changed")
    promotion = load_promoted_integrator_config(config.source_config)
    source_latest = promotion.artifact_root / "LATEST_RUN.json"
    original_latest = source_latest.read_bytes() if source_latest.is_file() else None
    try:
        current_source = bootstrap_promoted_integrator(config.source_config)
    finally:
        if original_latest is not None:
            atomic_write(source_latest, original_latest)
    source_store = IntegratorStore(
        current_source.promotion.artifact_root, config.source_run_hash
    )
    stored_protocol_path = source_store.run / "protocol/protocol.json"
    stored_protocol = _stored_integrator_protocol(
        load_canonical_json(stored_protocol_path)
    )
    source = replace(
        current_source,
        integrator=replace(
            current_source.integrator,
            protocol=stored_protocol,
            store=source_store,
            code_manifest=load_canonical_json(source_store.run / "protocol/code_manifest.json"),
            environment_manifest=load_canonical_json(source_store.run / "protocol/environment_manifest.json"),
        ),
    )
    stored_resolved = load_canonical_json(source_store.run / "config_resolved.json")
    current_runtime = source.integrator.config.as_record()
    primary = source.integrator.primary_config
    joint = primary.joint_training
    expected_joint = (
        config.joint_epochs,
        config.joint_batch_size,
        config.joint_momentum,
        config.joint_weight_decay,
        config.joint_lora_learning_rate,
        config.joint_head_learning_rate,
    )
    observed_joint = (
        joint.epochs,
        joint.batch_size,
        joint.momentum,
        joint.weight_decay,
        joint.lora_lr,
        joint.head_lr,
    )
    if (
        record_sha256(current_runtime)
        != record_sha256(
            {key: stored_resolved.get(key) for key in current_runtime}
        )
        or primary.lora_rank != config.source_rank
        or primary.lora_alpha != config.source_alpha
        or primary.lora_dropout != 0.0
        or observed_joint != expected_joint
    ):
        raise ValueError("live source or joint-IID recipe differs from its frozen values")
    hierarchy_path = (
        source_store.run
        / "hierarchies"
        / config.source_hierarchy_policy_hash
        / "complete_050.json"
    )
    joint_path = source_store.run / "evaluations/stage_matched_joint_iid.json"
    selection_path = config.selection_run / "evaluations/frontier_architecture_replay_sweep.json"
    if (
        stored_protocol.content_hash != config.source_run_hash
        or file_sha256(stored_protocol_path) != config.source_protocol_sha256
        or file_sha256(hierarchy_path) != config.source_hierarchy_complete_sha256
        or file_sha256(joint_path) != config.stage_matched_joint_sha256
        or file_sha256(selection_path) != config.selection_result_sha256
    ):
        raise ValueError("one or more persistent affine source artifacts changed")
    stage_matched = load_canonical_json(joint_path)
    if (
        stage_matched.get("content_hash") != config.stage_matched_joint_content_hash
        or len(validate_stage_rows(stage_matched["rows"])) != config.tasks
    ):
        raise ValueError("stage-matched joint curve does not authenticate")
    selection = _validated_content_record(
        selection_path, "imagenetr50-frontier-architecture-replay-result-v1"
    )
    _selection_cells(config, selection)
    return config, source, stage_matched, selection


def bootstrap_persistent_affine(
    config_path: str | Path = DEFAULT_PERSISTENT_AFFINE_CONFIG,
) -> PersistentAffineBootstrap:
    """Authenticate all promoted inputs and create the isolated affine namespace."""
    resolved = Path(config_path).resolve()
    project_root = resolved.parents[3]
    config, source, stage_matched, selection = load_persistent_inputs(resolved)
    h4096, h8192 = _selection_cells(config, selection)
    code = material_tree_manifest(_material_paths(project_root, resolved))
    environment = installed_environment_manifest(PROMOTED_PACKAGES)
    missing = tuple(
        str(row["name"])
        for row in environment["packages"]
        if row["version"] == "MISSING"
    )
    if missing:
        raise RuntimeError(f"isolated vision environment is incomplete: {missing}")
    protocol = PersistentAffineProtocol(
        config.source_run_hash,
        config.source_protocol_sha256,
        config.source_hierarchy_policy_hash,
        config.source_hierarchy_complete_sha256,
        config.stage_matched_joint_sha256,
        config.stage_matched_joint_content_hash,
        config.selection_result_sha256,
        str(h4096["content_hash"]),
        str(h8192["content_hash"]),
        source.integrator.protocol.dataset_manifest_hash,
        source.integrator.protocol.model_manifest_hash,
        source.integrator.protocol.split_hash,
        config.config_hash,
        str(code["content_hash"]),
        str(environment["content_hash"]),
    )
    run = config.artifact_root / "runs" / protocol.content_hash
    _prepare_run(run, protocol)
    for name, record in (
        ("code_manifest.json", code),
        ("environment_manifest.json", environment),
        ("source_stage_matched_joint.json", stage_matched),
        ("stage31_selection.json", selection),
    ):
        publish_immutable_json(run / "protocol" / name, record)
    publish_immutable_bytes(
        run / "config_resolved.json", canonical_json_bytes(config.as_record())
    )
    atomic_write(
        config.artifact_root / "LATEST_RUN.json",
        canonical_json_bytes(
            {
                "run_hash": protocol.content_hash,
                "schema_version": "imagenetr50-persistent-affine-latest-v1",
            }
        ),
    )
    return PersistentAffineBootstrap(
        project_root, config, source, stage_matched, selection, protocol, run
    )


def _write_state(
    bootstrap: PersistentAffineBootstrap, phase: str, **values: object
) -> None:
    atomic_write(
        bootstrap.run / "state/workflow.json",
        canonical_json_bytes(
            {
                "phase": phase,
                "run_hash": bootstrap.protocol.content_hash,
                "schema_version": "imagenetr50-persistent-affine-state-v1",
                "updated_at_utc": _utc_now(),
                **values,
            }
        ),
    )


def _new_frontier_model(
    bootstrap: PersistentAffineBootstrap,
    nodes: Sequence[BehaviorNode],
    slots: Sequence[int],
    device: torch.device,
) -> PersistentFrontierIntegrator:
    return PersistentFrontierIntegrator(
        nodes,
        slots,
        lambda: create_pinned_backbone(bootstrap.source.integrator.checkpoint),
        bootstrap.config.source_rank,
        bootstrap.config.source_alpha,
        device,
        hidden_dimension=bootstrap.hidden_dimension,
        initialization_seed=bootstrap.config.seed + 10_000 * len({task for node in nodes for task in node.represented_task_ids}),
    )


def _preflight(
    bootstrap: PersistentAffineBootstrap,
    hierarchy: HierarchyBuildResult,
    device: torch.device,
) -> dict[str, object]:
    target = bootstrap.run / "evaluations/preflight.json"
    if target.is_file():
        return _validated_content_record(
            target, "imagenetr50-persistent-affine-preflight-v1"
        )
    integrator = bootstrap.source.integrator
    nodes2, slots2, _frontier2 = _hierarchy_frontier(hierarchy, 2)
    model2 = _new_frontier_model(bootstrap, nodes2, slots2, device)
    previous = export_model_state(model2)
    previous_node = model2.node_hashes[0]
    previous_adapter = previous["node_adapters"][previous_node]
    first_module = sorted(previous_adapter)[0]
    previous_adapter[first_module] = type(previous_adapter[first_module])(
        previous_adapter[first_module].a + 0.125,
        previous_adapter[first_module].b,
        previous_adapter[first_module].scale,
    )
    del model2
    torch.cuda.empty_cache()
    nodes3, slots3, frontier3 = _hierarchy_frontier(hierarchy, 3)
    model3 = _new_frontier_model(bootstrap, nodes3, slots3, device)
    loader = DataLoader(
        ManifestDataset(
            integrator.config.data_root / "imagenet-r",
            integrator.manifest.select("train", (0, 1, 2))[: bootstrap.config.microbatch_size],
            integrator.train_transform,
            bootstrap.config.seed,
            0,
        ),
        batch_size=bootstrap.config.microbatch_size,
        shuffle=False,
        num_workers=min(bootstrap.config.num_workers, os_cpu_workers()),
        pin_memory=True,
    )
    images, labels, _image_ids = next(iter(loader))
    images, labels = images.to(device), labels.to(device)
    model3.set_evaluation_mode()
    with torch.inference_mode():
        exact, _features, local = model3.forward_components(images, False, False)
        union = raw_union_logits(model3, local)
        union_error = float(
            torch.max(torch.abs(exact[:, model3.seen_class_mask] - union[:, model3.seen_class_mask]))
        )
    carry = carry_model_state(model3, previous)
    continued_index = model3.node_hashes.index(previous_node)
    continued = adapter_factors(model3.node_models[continued_index])[first_module]
    adapter_carry_error = float(
        torch.max(torch.abs(continued.a.cpu() - previous_adapter[first_module].a))
    )
    model3.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model3(images, True, True)
        loss = F.cross_entropy(logits, labels)
    loss.backward()
    trainables = model3.trainable_parameters
    optimizer = create_optimizer(model3, bootstrap.config)
    optimizer.step()
    model_snapshot = export_model_state(model3)
    optimizer_snapshot = export_named_optimizer_state(optimizer, model3)
    with torch.inference_mode():
        expected_resume = model3(images, False, False)[:, model3.seen_class_mask].clone()
    with tempfile.TemporaryDirectory(prefix="head-resume-preflight-", dir=bootstrap.run / "work") as temporary:
        checkpoint = Path(temporary) / "state.pt"
        atomic_torch_save(checkpoint, {"model": model_snapshot, "optimizer": optimizer_snapshot})
        restored = torch.load(checkpoint, map_location="cpu", weights_only=False)
    with torch.no_grad():
        for parameter in trainables:
            parameter.add_(0.25)
    load_exact_model_state(model3, restored["model"])
    optimizer.state.clear()
    restore_named_optimizer_state(optimizer, model3, restored["optimizer"])
    observed_optimizer = export_named_optimizer_state(optimizer, model3)
    optimizer_resume_exact = all(
        torch.equal(value, observed_optimizer["states"][name][key])
        for name, state in optimizer_snapshot["states"].items()
        for key, value in state.items()
    )
    with torch.inference_mode():
        observed_resume = model3(images, False, False)[:, model3.seen_class_mask]
        resume_error = float((expected_resume - observed_resume).abs().max())
    train_ids = {row.image_id for row in integrator.manifest.images if row.split == "train"}
    test_ids = {row.image_id for row in integrator.manifest.images if row.split == "test"}
    topology = [len(hierarchy.frontier(stage)) for stage in range(1, 9)]
    core: dict[str, object] = {
        "adapter_carry_max_error": adapter_carry_error,
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "carry": carry.as_record(),
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(device),
        "finite_gradient_tensors": sum(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            for parameter in trainables
        ),
        "frontier_hash": frontier3,
        "loss": float(loss.detach()),
        "hidden_dimension": bootstrap.hidden_dimension,
        "optimizer_resume_exact": optimizer_resume_exact,
        "resume_max_logit_error": resume_error,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(device),
        "output_shape": list(logits.shape),
        "schema_version": "imagenetr50-persistent-affine-preflight-v1",
        "stage_1_to_8_live_nodes": topology,
        "test_train_overlap": len(train_ids & test_ids),
        "trainable_parameters": sum(parameter.numel() for parameter in trainables),
        "trainable_tensors": len(trainables),
        "union_max_logit_error": union_error,
    }
    expected_continued = (previous_node,)
    if (
        not core["cuda_available"]
        or not core["bf16_supported"]
        or union_error > 1e-4
        or adapter_carry_error != 0.0
        or resume_error != 0.0
        or not optimizer_resume_exact
        or carry.continued_node_hashes != expected_continued
        or len(carry.reset_node_hashes) != 1
        or carry.retired_node_hashes
        or topology != [1, 1, 2, 1, 2, 2, 3, 1]
        or core["test_train_overlap"] != 0
        or core["finite_gradient_tensors"] != len(trainables)
        or core["output_shape"] != [bootstrap.config.microbatch_size, 200]
        or not math.isfinite(float(core["loss"]))
    ):
        raise RuntimeError(f"persistent affine preflight failed: {core}")
    model3.zero_grad(set_to_none=True)
    del optimizer, model3
    torch.cuda.empty_cache()
    record = {**core, "content_hash": record_sha256(core)}
    publish_immutable_json(target, record)
    return record


def _publish_affine_stage(
    bootstrap: PersistentAffineBootstrap,
    capacity: int,
    stage: int,
    model: PersistentFrontierIntegrator,
    result: Mapping[str, object],
) -> tuple[Path, str]:
    target = bootstrap.run / "arms" / f"h{capacity:05d}" / "stages" / f"stage_{stage:03d}"
    if target.is_dir():
        return target, validate_artifact_directory(target)
    work = Path(tempfile.mkdtemp(prefix="persistent-affine-", dir=bootstrap.run / "work"))
    try:
        from safetensors.torch import save_file

        save_file(
            {name: value.detach().cpu().contiguous() for name, value in model.integrator.state_dict().items()},
            work / "integrator.safetensors",
            metadata={"schema_version": "imagenetr50-persistent-dense-head-v1", "hidden_dimension": str(bootstrap.hidden_dimension)},
        )
        adapters = []
        for index, (node_hash, node_model) in enumerate(
            zip(model.node_hashes, model.node_models, strict=True)
        ):
            filename = f"node_{index:02d}_adapter.safetensors"
            adapters.append(
                {
                    "filename": filename,
                    "node_hash": node_hash,
                    "sha256": save_adapter(work / filename, adapter_factors(node_model)),
                }
            )
        core = {**dict(result), "adapters": adapters, "integrator_sha256": file_sha256(work / "integrator.safetensors")}
        publish_immutable_json(work / "result.json", {**core, "content_hash": record_sha256(core)})
        artifact_hash = publish_artifact_directory(work, target)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return target, artifact_hash


def _load_affine_stage(
    bootstrap: PersistentAffineBootstrap, capacity: int, stage: int
) -> dict[str, object] | None:
    target = bootstrap.run / "arms" / f"h{capacity:05d}" / "stages" / f"stage_{stage:03d}"
    if not target.is_dir():
        return None
    artifact_hash = validate_artifact_directory(target)
    result = _validated_content_record(target / "result.json", "imagenetr50-persistent-affine-stage-result-v1")
    if (
        result.get("protocol_hash") != bootstrap.protocol.content_hash
        or result.get("capacity") != capacity
        or result.get("stage") != stage
    ):
        raise ValueError("stored affine stage identity changed")
    return {**result, "stage_artifact_sha256": artifact_hash}


def _checkpoint_state(path: Path) -> tuple[dict[str, object], dict[str, object], int]:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    model, optimizer = saved.get("model_state"), saved.get("optimizer_state")
    if (
        saved.get("schema_version") != "imagenetr50-persistent-frontier-checkpoint-v1"
        or not isinstance(model, dict)
        or not isinstance(optimizer, dict)
    ):
        raise ValueError("continuing checkpoint is malformed")
    return model, optimizer, int(saved["optimizer_steps_total"])


def _run_affine_arm(
    bootstrap: PersistentAffineBootstrap,
    hierarchy: HierarchyBuildResult,
    capacity: int,
    device: torch.device,
    on_stage_complete: Callable[[int], None] | None = None,
    show_progress: bool = True,
) -> tuple[tuple[dict[str, object], ...], InvocationWork]:
    root = bootstrap.run / "arms" / f"h{capacity:05d}"
    for relative in ("checkpoints", "stages"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    stage_ledger = ChainedJsonlLedger(root / "stage_metrics.jsonl", ARM_LEDGER_SCHEMA)
    history = ChainedJsonlLedger(root / "epoch_history.jsonl", EPOCH_LEDGER_SCHEMA)
    stage_ledger.require_unique_keys(("stage",))
    completed = {int(row["stage"]): row for row in stage_ledger.rows}
    for stage, row in completed.items():
        stored = _load_affine_stage(bootstrap, capacity, stage)
        if (
            stored is None
            or stored["stage_artifact_sha256"] != row["stage_artifact_sha256"]
            or stored["content_hash"] != row.get("content_hash")
        ):
            raise ValueError("affine stage ledger refers to changed model evidence")
    # Repair a crash after immutable publication but before the ledger append.
    for stage in range(1, bootstrap.config.tasks + 1):
        if stage in completed:
            continue
        stored = _load_affine_stage(bootstrap, capacity, stage)
        if stored is None:
            break
        appended = stage_ledger.append(stored)
        completed[stage] = appended
        if on_stage_complete is not None:
            on_stage_complete(stage)
    first_missing = next(
        (stage for stage in range(1, bootstrap.config.tasks + 1) if stage not in completed),
        bootstrap.config.tasks + 1,
    )
    previous_model: Mapping[str, object] | None = None
    previous_optimizer: Mapping[str, object] | None = None
    optimizer_steps_total = 0
    if first_missing > 1 and first_missing <= bootstrap.config.tasks:
        prior_checkpoint = root / "checkpoints" / f"stage_{first_missing - 1:03d}.pt"
        if not prior_checkpoint.is_file():
            raise FileNotFoundError("latest completed affine stage lacks continuing optimizer state")
        previous_model, previous_optimizer, optimizer_steps_total = _checkpoint_state(prior_checkpoint)
    all_train = tuple(
        row for row in bootstrap.source.integrator.manifest.images if row.split == "train"
    )
    new_steps = new_stages = 0
    from tqdm.auto import tqdm

    stages = tqdm(
        range(first_missing, bootstrap.config.tasks + 1),
        total=bootstrap.config.tasks,
        initial=first_missing - 1,
        desc=f"persistent {'MLP' if bootstrap.hidden_dimension else 'affine'} H={capacity:,}",
        unit="task",
        disable=not show_progress,
    )
    for stage in stages:
        _write_state(bootstrap, "adaptive_mlp" if bootstrap.hidden_dimension else "adaptive_affine", capacity=capacity, stage=stage)
        nodes, slots, frontier_hash = _hierarchy_frontier(hierarchy, stage)
        model = _new_frontier_model(bootstrap, nodes, slots, device)
        carry = carry_model_state(model, previous_model)
        optimizer = create_optimizer(model, bootstrap.config)
        restored_optimizer_tensors = restore_named_optimizer_state(
            optimizer, model, previous_optimizer
        )
        population = rotating_replay_population(
            all_train, stage, capacity, bootstrap.config.seed
        )
        if bootstrap.reference_rows:
            reference = bootstrap.reference_rows[stage - 1]
            if (
                population.as_record() != reference["population"]
                or list(model.node_hashes) != reference["node_hashes"]
                or bootstrap.config.epoch_map[capacity] != reference["epochs_per_stage"]
            ):
                raise ValueError("new integrator condition differs from the affine replay/frontier budget")
        checkpoint = root / "checkpoints" / f"stage_{stage:03d}.pt"
        steps_before_invocation = (
            int(torch.load(checkpoint, map_location="cpu", weights_only=False)["optimizer_steps_total"])
            if checkpoint.is_file() else optimizer_steps_total
        )
        fit, model_state, optimizer_state = fit_online_stage(
            model=model,
            optimizer=optimizer,
            config=bootstrap.config,
            protocol_hash=bootstrap.protocol.content_hash,
            capacity=capacity,
            stage=stage,
            frontier_hash=frontier_hash,
            population=population,
            prepared_root=bootstrap.source.integrator.config.data_root / "imagenet-r",
            train_transform=bootstrap.source.integrator.train_transform,
            checkpoint_path=checkpoint,
            history=history,
            optimizer_steps_total=optimizer_steps_total,
            device=device,
        )
        evaluation_rows = bootstrap.source.integrator.manifest.select(
            "test", range(stage)
        )
        evaluation = evaluate_prefix(
            model=model,
            prepared_root=bootstrap.source.integrator.config.data_root / "imagenet-r",
            rows=evaluation_rows,
            transform=bootstrap.source.integrator.test_transform,
            batch_size=bootstrap.config.evaluation_batch_size,
            num_workers=bootstrap.config.num_workers,
            device=device,
        )
        core: dict[str, object] = {
            "capacity": capacity,
            "hidden_dimension": bootstrap.hidden_dimension,
            "integrator_parameters": sum(parameter.numel() for parameter in model.integrator.parameters()),
            "lora_parameters": sum(parameter.numel() for parameter in model.lora_parameters),
            "carry": carry.as_record(),
            "epochs_per_stage": bootstrap.config.epoch_map[capacity],
            "evaluation": evaluation.as_record(),
            "fit": fit.as_record(),
            "frontier_hash": frontier_hash,
            "live_nodes": len(nodes),
            "node_hashes": list(model.node_hashes),
            "optimizer_state_tensors_restored": restored_optimizer_tensors,
            "population": population.as_record(),
            "protocol_hash": bootstrap.protocol.content_hash,
            "schema_version": "imagenetr50-persistent-affine-stage-result-v1",
            "slots": list(model.slot_indices),
            "stage": stage,
        }
        _path, artifact_hash = _publish_affine_stage(
            bootstrap, capacity, stage, model, core
        )
        result = _load_affine_stage(bootstrap, capacity, stage)
        if result is None or result["stage_artifact_sha256"] != artifact_hash:
            raise RuntimeError("published affine stage could not be authenticated")
        stage_ledger.append(result)
        if on_stage_complete is not None:
            on_stage_complete(stage)
        new_steps += fit.optimizer_steps_total - steps_before_invocation
        new_stages += 1
        previous_model, previous_optimizer = model_state, optimizer_state
        optimizer_steps_total = fit.optimizer_steps_total
        previous_checkpoint = root / "checkpoints" / f"stage_{stage - 1:03d}.pt"
        if stage > 1:
            previous_checkpoint.unlink(missing_ok=True)
        stages.set_postfix(
            accuracy=f"{evaluation.accuracy:.2f}",
            live=len(nodes),
            nll=f"{evaluation.nll:.3f}",
        )
        del optimizer, model
        torch.cuda.empty_cache()
    stages.close()
    stage_ledger.require_unique_keys(("stage",))
    rows = tuple(sorted(stage_ledger.rows, key=lambda row: int(row["stage"])))
    if [int(row["stage"]) for row in rows] != list(range(1, bootstrap.config.tasks + 1)):
        raise RuntimeError("persistent affine arm is incomplete")
    return rows, InvocationWork(new_steps, new_stages)


def _evaluate_joint(
    model: AdapterVisionModel,
    prepared_root: Path,
    rows: Sequence[ImageRecord],
    transform: object,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> PrefixEvaluation:
    loader = DataLoader(
        ManifestDataset(prepared_root, rows, transform, 0, 0),
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(num_workers, os_cpu_workers()),
        pin_memory=True,
        persistent_workers=False,
    )
    model.eval()
    nll_sum = correct = examples = 0
    task_count = max(row.task_index for row in rows) + 1
    task_correct, task_examples = [0] * task_count, [0] * task_count
    started = time.monotonic()
    with torch.inference_mode():
        for images, labels, _image_ids in loader:
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(images.to(device, non_blocking=True))
                nll_sum += float(F.cross_entropy(logits, labels, reduction="sum"))
            predictions = logits.argmax(dim=1)
            matches = predictions == labels
            correct += int(matches.sum())
            examples += len(labels)
            tasks = labels // 4
            for task in torch.unique(tasks).tolist():
                selected = tasks == task
                task_examples[task] += int(selected.sum())
                task_correct[task] += int(matches[selected].sum())
    accuracy = 100.0 * correct / examples
    return PrefixEvaluation(
        accuracy,
        nll_sum / examples,
        examples,
        accuracy,
        accuracy,
        tuple(task_correct),
        tuple(task_examples),
        time.monotonic() - started,
    )


def _publish_rank_stage(
    root: Path,
    stage: int,
    model: AdapterVisionModel,
    result: Mapping[str, object],
) -> tuple[Path, str]:
    target = root / "stages" / f"stage_{stage:03d}"
    if target.is_dir():
        return target, validate_artifact_directory(target)
    work = Path(
        tempfile.mkdtemp(prefix="rank-matched-", dir=root.parent.parent / "work")
    )
    try:
        adapter_sha = save_adapter(work / "adapter.safetensors", adapter_factors(model))
        classifier_sha = save_classifier(work / "classifier.safetensors", model.classifier.rows())
        core = {**dict(result), "adapter_sha256": adapter_sha, "classifier_sha256": classifier_sha}
        publish_immutable_json(work / "result.json", {**core, "content_hash": record_sha256(core)})
        artifact_hash = publish_artifact_directory(work, target)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return target, artifact_hash


def _load_rank_stage(
    bootstrap: PersistentAffineBootstrap, stage: int
) -> dict[str, object] | None:
    target = bootstrap.run / "controls/rank_matched_joint/stages" / f"stage_{stage:03d}"
    if not target.is_dir():
        return None
    artifact_hash = validate_artifact_directory(target)
    result = _validated_content_record(
        target / "result.json", "imagenetr50-rank-matched-joint-stage-result-v1"
    )
    if result.get("protocol_hash") != bootstrap.protocol.content_hash or result.get("stage") != stage:
        raise ValueError("stored rank-matched stage identity changed")
    return {**result, "stage_artifact_sha256": artifact_hash}


def _run_rank_matched_joint(
    bootstrap: PersistentAffineBootstrap,
    device: torch.device,
    on_stage_complete: Callable[[int], None] | None = None,
    show_progress: bool = True,
) -> tuple[tuple[dict[str, object], ...], InvocationWork]:
    root = bootstrap.run / "controls/rank_matched_joint"
    ledger = ChainedJsonlLedger(root / "stage_metrics.jsonl", RANK_LEDGER_SCHEMA)
    ledger.require_unique_keys(("stage",))
    completed = {int(row["stage"]): row for row in ledger.rows}
    source_rows = {
        int(row["stage"]): dict(row) for row in bootstrap.stage_matched_joint["rows"]
    }
    for stage, row in completed.items():
        expected_rank = bootstrap.config.joint_rank_per_live_node * stage.bit_count()
        if int(row.get("rank", 0)) != expected_rank:
            raise ValueError("stored rank-matched ledger has the wrong rank")
        if expected_rank == bootstrap.config.source_rank:
            scientific_core = {
                key: value
                for key, value in row.items()
                if key not in LEDGER_FIELDS | {"content_hash", "stage_artifact_sha256"}
            }
            source = source_rows[stage]
            if (
                row.get("schema_version")
                != "imagenetr50-rank-matched-joint-import-v1"
                or row.get("stage_artifact_sha256") is not None
                or row.get("source_stage_result_hash") != source["stage_result_hash"]
                or float(row["accuracy"]) != float(source["accuracy"])
                or row.get("content_hash") != record_sha256(scientific_core)
            ):
                raise ValueError("imported rank-16 joint stage changed")
        else:
            stored = _load_rank_stage(bootstrap, stage)
            if (
                stored is None
                or stored["stage_artifact_sha256"]
                != row.get("stage_artifact_sha256")
                or stored["content_hash"] != row.get("content_hash")
            ):
                raise ValueError("rank-matched ledger refers to changed model evidence")
    new_steps = new_stages = 0
    from tqdm.auto import tqdm

    stages = tqdm(
        range(1, bootstrap.config.tasks + 1),
        total=bootstrap.config.tasks,
        initial=0,
        desc="aggregate-rank joint IID",
        unit="task",
        disable=not show_progress,
    )
    for stage in stages:
        if stage in completed:
            stages.update(0)
            continue
        _write_state(bootstrap, "rank_matched_joint", stage=stage)
        rank = bootstrap.config.joint_rank_per_live_node * stage.bit_count()
        if rank == bootstrap.config.source_rank:
            source = source_rows[stage]
            core: dict[str, object] = {
                "accuracy": float(source["accuracy"]),
                "alpha": rank,
                "class_count": 4 * stage,
                "evaluation_examples": int(source["test_examples"]),
                "image_presentations": int(source["image_presentations"]),
                "nll": None,
                "optimizer_steps": int(source["optimizer_steps"]),
                "protocol_hash": bootstrap.protocol.content_hash,
                "rank": rank,
                "reused_stage_matched_rank16": True,
                "schema_version": "imagenetr50-rank-matched-joint-import-v1",
                "source_stage_result_hash": source["stage_result_hash"],
                "stage": stage,
                "task_correct": source["task_correct"],
                "task_examples": source["task_examples"],
                "train_examples": int(source["train_examples"]),
                "training_seconds": float(source["training_seconds"]),
            }
            ledger.append({**core, "content_hash": record_sha256(core), "stage_artifact_sha256": None})
            if on_stage_complete is not None:
                on_stage_complete(stage)
            continue
        stored = _load_rank_stage(bootstrap, stage)
        if stored is not None:
            ledger.append(stored)
            if on_stage_complete is not None:
                on_stage_complete(stage)
            continue
        model = AdapterVisionModel(
            create_pinned_backbone(bootstrap.source.integrator.checkpoint),
            tuple(range(4 * stage)),
            rank,
            rank,
            0.0,
            bootstrap.config.seed,
        )
        train_rows = bootstrap.source.integrator.manifest.select("train", range(stage))
        checkpoint = root / "checkpoints" / f"stage_{stage:03d}.pt"
        training = train_adapter_model(
            model,
            bootstrap.source.integrator.config.data_root / "imagenet-r",
            train_rows,
            bootstrap.source.integrator.train_transform,
            bootstrap.source.integrator.primary_config.joint_training,
            bootstrap.config.seed + 50_000,
            device,
            checkpoint,
            num_workers=bootstrap.config.num_workers,
            checkpoint_steps=bootstrap.source.integrator.primary_config.checkpoint_steps,
            show_progress=True,
        )
        test_rows = bootstrap.source.integrator.manifest.select("test", range(stage))
        evaluation = _evaluate_joint(
            model,
            bootstrap.source.integrator.config.data_root / "imagenet-r",
            test_rows,
            bootstrap.source.integrator.test_transform,
            bootstrap.config.evaluation_batch_size,
            bootstrap.config.num_workers,
            device,
        )
        core = {
            "accuracy": evaluation.accuracy,
            "alpha": rank,
            "class_count": 4 * stage,
            "evaluation_examples": evaluation.examples,
            "image_presentations": training.image_presentations,
            "nll": evaluation.nll,
            "optimizer_steps": training.optimizer_steps,
            "protocol_hash": bootstrap.protocol.content_hash,
            "rank": rank,
            "reused_stage_matched_rank16": False,
            "schema_version": "imagenetr50-rank-matched-joint-stage-result-v1",
            "source_stage_result_hash": None,
            "stage": stage,
            "task_correct": list(evaluation.task_correct),
            "task_examples": list(evaluation.task_examples),
            "train_examples": len(train_rows),
            "training_final_loss": training.final_loss,
            "training_seconds": training.wall_seconds,
        }
        _path, artifact_hash = _publish_rank_stage(root, stage, model, core)
        result = _load_rank_stage(bootstrap, stage)
        if result is None or result["stage_artifact_sha256"] != artifact_hash:
            raise RuntimeError("published rank-matched stage could not be authenticated")
        ledger.append(result)
        if on_stage_complete is not None:
            on_stage_complete(stage)
        checkpoint.unlink(missing_ok=True)
        new_steps += training.optimizer_steps
        new_stages += 1
        stages.set_postfix(accuracy=f"{evaluation.accuracy:.2f}", rank=rank)
        del model
        torch.cuda.empty_cache()
    stages.close()
    ledger.require_unique_keys(("stage",))
    rows = tuple(sorted(ledger.rows, key=lambda row: int(row["stage"])))
    if [int(row["stage"]) for row in rows] != list(range(1, bootstrap.config.tasks + 1)):
        raise RuntimeError("aggregate-rank joint curve is incomplete")
    return rows, InvocationWork(rank_optimizer_steps=new_steps, rank_stages=new_stages)


def _affine_stage_work(
    bootstrap: PersistentAffineBootstrap, capacity: int, stage: int
) -> int:
    """Estimate one adaptive stage in node-image forward equivalents."""
    train_rows = tuple(
        row
        for row in bootstrap.source.integrator.manifest.images
        if row.split == "train" and row.task_index < stage
    )
    current = sum(row.task_index == stage - 1 for row in train_rows)
    population = current + min(capacity, len(train_rows) - current)
    test_examples = len(
        bootstrap.source.integrator.manifest.select("test", range(stage))
    )
    live_nodes = stage.bit_count()
    # Gradient checkpointing recomputes each adapted ViT once during backward.
    return live_nodes * (
        2 * bootstrap.config.epoch_map[capacity] * population + test_examples
    )


def _rank_stage_work(bootstrap: PersistentAffineBootstrap, stage: int) -> int:
    """Estimate one fresh aggregate-rank joint stage in image forwards."""
    if stage.bit_count() == 1:
        return 0
    train_examples = len(
        bootstrap.source.integrator.manifest.select("train", range(stage))
    )
    test_examples = len(
        bootstrap.source.integrator.manifest.select("test", range(stage))
    )
    return bootstrap.config.joint_epochs * train_examples + test_examples


def _completed_work(bootstrap: PersistentAffineBootstrap) -> tuple[int, int]:
    """Return completed and total estimated GPU work for the overall ETA bar."""
    total = completed = 0
    for capacity in bootstrap.config.historical_capacities:
        ledger = ChainedJsonlLedger(
            bootstrap.run / "arms" / f"h{capacity:05d}" / "stage_metrics.jsonl",
            ARM_LEDGER_SCHEMA,
        )
        completed_stages = {int(row["stage"]) for row in ledger.rows}
        for stage in range(1, bootstrap.config.tasks + 1):
            stage_work = _affine_stage_work(bootstrap, capacity, stage)
            total += stage_work
            if stage in completed_stages:
                completed += stage_work
    rank_ledger = ChainedJsonlLedger(
        bootstrap.run / "controls/rank_matched_joint/stage_metrics.jsonl",
        RANK_LEDGER_SCHEMA,
    )
    completed_rank_stages = {int(row["stage"]) for row in rank_ledger.rows}
    for stage in range(1, bootstrap.config.tasks + 1):
        stage_work = _rank_stage_work(bootstrap, stage)
        total += stage_work
        if stage in completed_rank_stages:
            completed += stage_work
    return completed, total


def _publish_result(
    bootstrap: PersistentAffineBootstrap,
    arm_rows: Mapping[int, Sequence[Mapping[str, object]]],
    rank_rows: Sequence[Mapping[str, object]],
    hierarchy_before: Mapping[str, object],
    hierarchy_after: Mapping[str, object],
    work: InvocationWork,
) -> Path:
    stage_rows = tuple(validate_stage_rows(bootstrap.stage_matched_joint["rows"]))
    arms = {
        str(capacity): [dict(row) for row in rows]
        for capacity, rows in sorted(arm_rows.items())
    }
    core: dict[str, object] = {
        "arms": arms,
        "completed_at_utc": _utc_now(),
        "hierarchy_source_unchanged": dict(hierarchy_before) == dict(hierarchy_after),
        "invocation_work": asdict(work),
        "protocol_hash": bootstrap.protocol.content_hash,
        "rank_matched_joint": [dict(row) for row in rank_rows],
        "schema_version": "imagenetr50-persistent-affine-result-v1",
        "stage_matched_joint": [dict(row) for row in stage_rows],
        "test_identity_role": "post-stage evaluation only; never optimization or selection",
    }
    if (
        not core["hierarchy_source_unchanged"]
        or any(len(rows) != bootstrap.config.tasks for rows in arm_rows.values())
        or len(rank_rows) != bootstrap.config.tasks
        or len(stage_rows) != bootstrap.config.tasks
    ):
        raise RuntimeError("persistent affine result is incomplete or changed its source")
    result = {**core, "content_hash": record_sha256(core)}
    target = bootstrap.run / RESULT_PATH
    if target.is_file():
        prior = load_canonical_json(target)
        # Wall-clock and invocation counters are provenance, not scientific identity.
        if prior.get("protocol_hash") == bootstrap.protocol.content_hash:
            return target
    publish_immutable_json(target, result)
    return target


def _prove_exact_reuse(
    bootstrap: PersistentAffineBootstrap,
    hierarchy: HierarchyBuildResult,
    expected_arms: Mapping[int, Sequence[Mapping[str, object]]],
    expected_rank: Sequence[Mapping[str, object]],
    device: torch.device,
) -> Path:
    """Invoke the completed workflows again and seal their zero-step reuse evidence."""
    result_path = bootstrap.run / RESULT_PATH
    result_sha256_before = file_sha256(result_path)
    hierarchy_root = (
        bootstrap.source.integrator.store.run
        / "hierarchies"
        / bootstrap.config.source_hierarchy_policy_hash
    )
    source_before = _stat_fingerprint(hierarchy_root)
    observed_arms: dict[int, tuple[dict[str, object], ...]] = {}
    reuse_work = InvocationWork()
    for capacity in bootstrap.config.historical_capacities:
        observed_arms[capacity], invocation = _run_affine_arm(
            bootstrap,
            hierarchy,
            capacity,
            device,
            show_progress=False,
        )
        reuse_work += invocation
    observed_rank, invocation = _run_rank_matched_joint(
        bootstrap, device, show_progress=False
    )
    reuse_work += invocation
    unchanged_rows = all(
        list(observed_arms[capacity]) == list(expected_arms[capacity])
        for capacity in bootstrap.config.historical_capacities
    ) and list(observed_rank) == list(expected_rank)
    core: dict[str, object] = {
        "hierarchy_source_unchanged": source_before
        == _stat_fingerprint(hierarchy_root),
        "protocol_hash": bootstrap.protocol.content_hash,
        "result_file_sha256": result_sha256_before,
        "result_file_unchanged": result_sha256_before == file_sha256(result_path),
        "schema_version": "imagenetr50-persistent-affine-reuse-proof-v1",
        "stage_rows_unchanged": unchanged_rows,
        "zero_new_optimizer_work": asdict(reuse_work),
    }
    if (
        reuse_work != InvocationWork()
        or not core["hierarchy_source_unchanged"]
        or not core["result_file_unchanged"]
        or not unchanged_rows
    ):
        raise RuntimeError(f"completed persistent-affine workflows did not reuse: {core}")
    target = bootstrap.run / "evaluations/reuse_proof.json"
    publish_immutable_json(target, {**core, "content_hash": record_sha256(core)})
    return target


def status_persistent_affine(
    config_path: str | Path = DEFAULT_PERSISTENT_AFFINE_CONFIG,
) -> dict[str, object]:
    """Return a read-only progress summary without constructing GPU models."""
    config = load_persistent_affine_config(config_path)
    latest = load_canonical_json(config.artifact_root / "LATEST_RUN.json")
    run = config.artifact_root / "runs" / str(latest["run_hash"])
    arms = {}
    for capacity in config.historical_capacities:
        ledger = ChainedJsonlLedger(
            run / "arms" / f"h{capacity:05d}" / "stage_metrics.jsonl",
            ARM_LEDGER_SCHEMA,
        )
        arms[str(capacity)] = len(ledger.rows)
    rank = ChainedJsonlLedger(
        run / "controls/rank_matched_joint/stage_metrics.jsonl", RANK_LEDGER_SCHEMA
    )
    state = load_canonical_json(run / "state/workflow.json") if (run / "state/workflow.json").is_file() else None
    return {
        "arms_completed": arms,
        "rank_stages_completed": len(rank.rows),
        "result_ready": (run / RESULT_PATH).is_file(),
        "run": str(run),
        "state": state,
    }


def run_persistent_affine(
    config_path: str | Path = DEFAULT_PERSISTENT_AFFINE_CONFIG,
) -> Path:
    """Run or exactly resume the complete v15 experiment and build its report."""
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("the persistent affine experiment requires BF16 CUDA")
    started = time.monotonic()
    print("[phase 1/7] Authenticate selected H cells, hierarchy, and joint-IID curve", flush=True)
    bootstrap = bootstrap_persistent_affine(config_path)
    print(f"Temporary/resumable artifact directory: {bootstrap.run}", flush=True)
    device = torch.device("cuda:0")
    hierarchy_root = (
        bootstrap.source.integrator.store.run
        / "hierarchies"
        / bootstrap.config.source_hierarchy_policy_hash
    )
    source_before = _stat_fingerprint(hierarchy_root)
    print("[phase 2/7] Reconstruct and verify the immutable all-train LogT hierarchy", flush=True)
    hierarchy = _build(bootstrap.source, "all_train", bootstrap.config.tasks, device, progress=False)
    if (
        hierarchy.policy.content_hash != bootstrap.config.source_hierarchy_policy_hash
        or hierarchy.work.leaf_optimizer_steps
        or hierarchy.work.parent_optimizer_steps
    ):
        raise RuntimeError("promoted source hierarchy was not reused exactly")
    print("[phase 3/7] Run the real-model transition and gradient preflight", flush=True)
    _preflight(bootstrap, hierarchy, device)
    completed_work, total_work = _completed_work(bootstrap)
    from tqdm.auto import tqdm

    overall = tqdm(
        total=total_work,
        initial=completed_work,
        desc="overall estimated GPU work",
        unit="node-image",
        unit_scale=True,
    )
    work = InvocationWork()
    arm_rows: dict[int, tuple[dict[str, object], ...]] = {}
    print("[phase 4/7] Run persistent single-affine H=4,096", flush=True)
    arm_rows[4_096], invocation = _run_affine_arm(
        bootstrap,
        hierarchy,
        4_096,
        device,
        lambda stage: overall.update(_affine_stage_work(bootstrap, 4_096, stage)),
    )
    work += invocation
    print("[phase 5/7] Run persistent single-affine H=8,192", flush=True)
    arm_rows[8_192], invocation = _run_affine_arm(
        bootstrap,
        hierarchy,
        8_192,
        device,
        lambda stage: overall.update(_affine_stage_work(bootstrap, 8_192, stage)),
    )
    work += invocation
    print("[phase 6/7] Train or reuse the aggregate-rank-matched joint-IID curve", flush=True)
    rank_rows, invocation = _run_rank_matched_joint(
        bootstrap,
        device,
        lambda stage: overall.update(_rank_stage_work(bootstrap, stage)),
    )
    work += invocation
    overall.close()
    source_after = _stat_fingerprint(hierarchy_root)
    result_path = _publish_result(
        bootstrap, arm_rows, rank_rows, source_before, source_after, work
    )
    print("[phase 7/7] Build the report and seal the exact-reuse projection", flush=True)
    reuse_proof = _prove_exact_reuse(
        bootstrap, hierarchy, arm_rows, rank_rows, device
    )
    from apm.continual.vision.imagenetr.persistent_affine_reporting import write_persistent_affine_report

    report = write_persistent_affine_report(bootstrap.run)
    _write_state(
        bootstrap,
        "complete",
        elapsed_seconds=time.monotonic() - started,
        report=str(report),
        result=str(result_path),
        reuse_proof=str(reuse_proof),
    )
    print(f"Result: {result_path}", flush=True)
    print(f"Report: {report}", flush=True)
    return report


def _main() -> None:
    import sys

    command = sys.argv[1] if len(sys.argv) > 1 else "run"
    config = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_PERSISTENT_AFFINE_CONFIG
    if command == "run":
        run_persistent_affine(config)
    elif command == "status":
        print(json.dumps(status_persistent_affine(config), indent=2, sort_keys=True))
    elif command == "report":
        loaded = load_persistent_affine_config(config)
        latest = load_canonical_json(loaded.artifact_root / "LATEST_RUN.json")
        from apm.continual.vision.imagenetr.persistent_affine_reporting import write_persistent_affine_report

        print(write_persistent_affine_report(loaded.artifact_root / "runs" / str(latest["run_hash"])))
    else:
        raise SystemExit("usage: persistent_affine_workflow [run|status|report] [config]")


if __name__ == "__main__":
    _main()


__all__ = [
    "InvocationWork",
    "PersistentAffineBootstrap",
    "PersistentAffineProtocol",
    "bootstrap_persistent_affine",
    "run_persistent_affine",
    "status_persistent_affine",
]

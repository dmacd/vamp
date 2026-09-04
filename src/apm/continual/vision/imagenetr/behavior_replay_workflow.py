"""Full-50 replay sweep over per-node LoRA-adapted ImageNet-R behaviors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import math
import os
import shutil
import tempfile
import time

import torch

from apm.continual.artifacts import (
    ChainedJsonlLedger,
    atomic_write,
    canonical_json_bytes,
    file_sha256,
    fsync_directory,
    load_canonical_json,
    publish_immutable_bytes,
    publish_immutable_json,
    record_sha256,
    require_sha256,
)
from apm.continual.vision.imagenetr.behavior_replay_config import (
    DEFAULT_BEHAVIOR_REPLAY_CONFIG,
    BehaviorReplayConfig,
    load_behavior_replay_config,
)
from apm.continual.vision.imagenetr.integrator_artifacts import (
    IntegratorBootstrap,
    IntegratorProtocol,
    IntegratorStore,
)
from apm.continual.vision.imagenetr.integrator_hierarchy import HierarchyBuildResult
from apm.continual.vision.imagenetr.integrator_model import (
    VARIANT_SLOT_DIMS,
    create_integrator_state,
    fit_integrator_epochs,
    parameter_count,
    predict_integrator,
)
from apm.continual.vision.imagenetr.integrator_observations import accuracy
from apm.continual.vision.imagenetr.integrator_persistence import (
    restore_integrator_checkpoint,
    save_integrator_checkpoint,
)
from apm.continual.vision.imagenetr.integrator_workflow import (
    _control_metrics,
    _frontier_tensors,
    _hierarchy_frontier,
    _partition_rows,
    _persistent_family_name,
    _persistent_training_rows,
    _supervision,
    _test_rows,
)
from apm.continual.vision.imagenetr.manifests import installed_environment_manifest
from apm.continual.vision.imagenetr.promoted_integrator_workflow import (
    PROMOTED_PACKAGES,
    PromotedBootstrap,
    _build,
    bootstrap_promoted_integrator,
)
from apm.continual.vision.imagenetr.protocol import material_tree_manifest


COMMON_STATE_NAME = "persistent-node-adapted-behavior-common-seed-v1"
FEATURE_LAYOUT = (
    ("lora_adapted_normalized_preclassifier_latent", 768),
    ("raw_affine_scores", 200),
    ("within_node_log_probabilities", 200),
    ("classifier_row_ownership", 200),
    ("active_slot", 1),
)


@dataclass(frozen=True, slots=True)
class BehaviorReplayProtocol:
    """Content identity binding the follow-up to one immutable source hierarchy."""

    source_run_hash: str
    source_protocol_sha256: str
    source_hierarchy_policy_hash: str
    source_hierarchy_complete_sha256: str
    source_locked_test_sha256: str
    dataset_manifest_hash: str
    model_manifest_hash: str
    split_hash: str
    config_hash: str
    code_manifest_hash: str
    environment_manifest_hash: str
    schema_version: str = "imagenetr50-logt-behavior-replay-protocol-v1"

    def __post_init__(self) -> None:
        for label, identity in (
            ("source run", self.source_run_hash),
            ("source protocol file", self.source_protocol_sha256),
            ("source hierarchy policy", self.source_hierarchy_policy_hash),
            ("source hierarchy completion", self.source_hierarchy_complete_sha256),
            ("source locked test", self.source_locked_test_sha256),
            ("dataset", self.dataset_manifest_hash),
            ("model", self.model_manifest_hash),
            ("split", self.split_hash),
            ("config", self.config_hash),
            ("code", self.code_manifest_hash),
            ("environment", self.environment_manifest_hash),
        ):
            require_sha256(identity, label)
        if self.schema_version != "imagenetr50-logt-behavior-replay-protocol-v1":
            raise ValueError("invalid adapted-latent replay protocol")

    @property
    def content_hash(self) -> str:
        """Return the stable follow-up run namespace."""
        return record_sha256(self.as_record(False))

    def as_record(self, include_hash: bool = True) -> dict[str, object]:
        """Return a canonical JSON-compatible protocol record."""
        core = asdict(self)
        return {**core, "content_hash": self.content_hash} if include_hash else core


@dataclass(frozen=True, slots=True)
class BehaviorReplayBootstrap:
    """Authenticated source, isolated target, and runtime view for the sweep."""

    project_root: Path
    config: BehaviorReplayConfig
    source: PromotedBootstrap
    protocol: BehaviorReplayProtocol
    store: IntegratorStore
    integrator: IntegratorBootstrap


@dataclass(frozen=True, slots=True)
class PersistentInvocation:
    """New work performed by one invocation of a persistent replay arm."""

    historical_capacity: int
    new_optimizer_steps: int
    new_training_stages: int
    new_evaluation_stages: int


def _material_paths(project_root: Path, config_path: Path) -> tuple[Path, ...]:
    package = project_root / "src/apm/continual/vision/imagenetr"
    return (
        config_path,
        project_root / "docs/imagenetr50_logt_behavior_replay_protocol.md",
        project_root / "scripts/vision/imagenetr/run_behavior_replay_local.sh",
        project_root / "src/apm/continual/artifacts.py",
        package,
    )


def _source_paths(
    source: PromotedBootstrap, config: BehaviorReplayConfig
) -> tuple[Path, Path, Path]:
    run = source.integrator.store.run
    hierarchy = run / "hierarchies" / config.source_hierarchy_policy_hash
    return (
        run / "protocol" / "protocol.json",
        hierarchy / "complete_050.json",
        run / "evaluations" / "locked_test.json",
    )


def _stored_integrator_protocol(record: Mapping[str, object]) -> IntegratorProtocol:
    """Restore and authenticate the source protocol independently of report code."""
    protocol = IntegratorProtocol(
        sealed_run_hash=str(record["sealed_run_hash"]),
        sealed_u100_policy_hash=str(record["sealed_u100_policy_hash"]),
        sealed_final_node_hashes=tuple(
            str(value) for value in record["sealed_final_node_hashes"]
        ),
        dataset_manifest_hash=str(record["dataset_manifest_hash"]),
        model_manifest_hash=str(record["model_manifest_hash"]),
        split_hash=str(record["split_hash"]),
        config_hash=str(record["config_hash"]),
        code_manifest_hash=str(record["code_manifest_hash"]),
        environment_manifest_hash=str(record["environment_manifest_hash"]),
        reference_results_hash=str(record["reference_results_hash"]),
        schema_version=str(record["schema_version"]),
    )
    if protocol.content_hash != record.get("content_hash"):
        raise ValueError("stored promoted protocol content changed")
    return protocol


def bootstrap_behavior_replay(
    config_path: str | Path = DEFAULT_BEHAVIOR_REPLAY_CONFIG,
) -> BehaviorReplayBootstrap:
    """Authenticate the promoted run and prepare an isolated follow-up namespace."""
    resolved = Path(config_path).resolve()
    project_root = resolved.parents[3]
    config = load_behavior_replay_config(resolved)
    current_source = bootstrap_promoted_integrator(config.source_config)
    original_store = IntegratorStore(
        current_source.promotion.artifact_root, config.source_run_hash
    )
    original_protocol_record = load_canonical_json(
        original_store.run / "protocol" / "protocol.json"
    )
    original_protocol = _stored_integrator_protocol(original_protocol_record)
    original_integrator = replace(
        current_source.integrator,
        protocol=original_protocol,
        store=original_store,
        code_manifest=load_canonical_json(
            original_store.run / "protocol" / "code_manifest.json"
        ),
        environment_manifest=load_canonical_json(
            original_store.run / "protocol" / "environment_manifest.json"
        ),
    )
    source = replace(current_source, integrator=original_integrator)
    atomic_write(
        current_source.promotion.artifact_root / "LATEST_RUN.json",
        canonical_json_bytes(
            {
                "run_hash": config.source_run_hash,
                "schema_version": "imagenetr50-promoted-integrator-latest-v1",
            }
        ),
    )
    protocol_path, hierarchy_path, locked_path = _source_paths(source, config)
    observed = (
        source.integrator.protocol.content_hash,
        file_sha256(protocol_path),
        source.integrator.protocol.dataset_manifest_hash,
        source.integrator.protocol.model_manifest_hash,
        source.integrator.protocol.split_hash,
        file_sha256(hierarchy_path),
        file_sha256(locked_path),
    )
    expected = (
        config.source_run_hash,
        config.source_protocol_sha256,
        source.integrator.manifest.content_hash,
        source.integrator.protocol.model_manifest_hash,
        source.integrator.split.content_hash,
        config.source_hierarchy_complete_sha256,
        config.source_locked_test_sha256,
    )
    if observed != expected:
        raise ValueError("the configured promoted source no longer authenticates")
    hierarchy_record = load_canonical_json(hierarchy_path)
    if (
        hierarchy_record.get("policy_hash") != config.source_hierarchy_policy_hash
        or hierarchy_record.get("task_count") != config.tasks
    ):
        raise ValueError("the source hierarchy completion record is not the requested tree")
    code = material_tree_manifest(_material_paths(project_root, resolved))
    environment = installed_environment_manifest(PROMOTED_PACKAGES)
    if any(
        row["version"] == "MISSING" for row in environment["packages"]
    ):
        raise RuntimeError("the isolated vision environment is incomplete")
    protocol = BehaviorReplayProtocol(
        config.source_run_hash,
        config.source_protocol_sha256,
        config.source_hierarchy_policy_hash,
        config.source_hierarchy_complete_sha256,
        config.source_locked_test_sha256,
        source.integrator.manifest.content_hash,
        source.integrator.protocol.model_manifest_hash,
        source.integrator.split.content_hash,
        config.config_hash,
        str(code["content_hash"]),
        str(environment["content_hash"]),
    )
    store = IntegratorStore(config.artifact_root, protocol.content_hash)
    store.prepare(protocol)  # type: ignore[arg-type]
    publish_immutable_bytes(
        store.run / "config_resolved.json", canonical_json_bytes(config.as_record())
    )
    for filename, record in (
        ("code_manifest.json", code),
        ("environment_manifest.json", environment),
        ("feature_layout.json", {
            "feature_variant": config.feature_variant,
            "fields": [
                {"dimensions": dimensions, "name": name}
                for name, dimensions in FEATURE_LAYOUT
            ],
            "maximum_slots": source.integrator.config.maximum_levels,
            "schema_version": "imagenetr50-node-adapted-feature-layout-v1",
            "slot_dimensions": VARIANT_SLOT_DIMS[config.feature_variant],
            "total_dimensions": (
                source.integrator.config.maximum_levels
                * VARIANT_SLOT_DIMS[config.feature_variant]
            ),
        }),
        ("source_hierarchy_complete.json", hierarchy_record),
        ("source_protocol.json", load_canonical_json(protocol_path)),
    ):
        publish_immutable_json(store.run / "protocol" / filename, record)
    atomic_write(
        config.artifact_root / "LATEST_RUN.json",
        canonical_json_bytes(
            {
                "run_hash": protocol.content_hash,
                "schema_version": "imagenetr50-logt-behavior-replay-latest-v1",
            }
        ),
    )
    return BehaviorReplayBootstrap(
        project_root,
        config,
        source,
        protocol,
        store,
        replace(source.integrator, store=store),
    )


def _write_state(
    bootstrap: BehaviorReplayBootstrap, phase: str, **values: object
) -> None:
    atomic_write(
        bootstrap.store.run / "state" / "workflow.json",
        canonical_json_bytes(
            {
                "phase": phase,
                "run_hash": bootstrap.protocol.content_hash,
                "schema_version": "imagenetr50-logt-behavior-replay-state-v1",
                **values,
            }
        ),
    )


def _stat_fingerprint(root: Path, pattern: str = "*") -> dict[str, object]:
    files = tuple(sorted(path for path in root.rglob(pattern) if path.is_file()))
    rows = tuple(
        {
            "mtime_ns": path.stat().st_mtime_ns,
            "path": str(path.relative_to(root)),
            "size_bytes": path.stat().st_size,
        }
        for path in files
    )
    return {
        "entries_hash": record_sha256(rows),
        "file_count": len(rows),
        "mtime_ns_sum": sum(int(row["mtime_ns"]) for row in rows),
        "size_bytes": sum(int(row["size_bytes"]) for row in rows),
    }


def seed_row_cache_shards(
    source_root: Path,
    target_root: Path,
    image_splits: Mapping[str, str],
    allowed_splits: frozenset[str],
) -> dict[str, object]:
    """Hard-link authenticated pure-split row shards into the isolated cache."""
    if not allowed_splits or not allowed_splits <= {"train", "test"}:
        raise ValueError("cache seeding requires declared dataset splits")
    source_rows = source_root / "row_shards"
    target_rows = target_root / "row_shards"
    eligible = linked = reused = skipped = fallback_copies = 0
    eligible_bytes = 0
    source_identities: list[str] = []
    for manifest_path in sorted(source_rows.glob("*/*/cache.json")):
        shard = manifest_path.parent
        manifest = load_canonical_json(manifest_path)
        image_ids = tuple(str(value) for value in manifest.get("image_ids", ()))
        if not image_ids or any(image_id not in image_splits for image_id in image_ids):
            raise ValueError("source row-cache shard contains unknown image identities")
        splits = frozenset(image_splits[image_id] for image_id in image_ids)
        if not splits <= allowed_splits:
            if splits & allowed_splits:
                raise ValueError("source row-cache shard mixes allowed and forbidden splits")
            skipped += 1
            continue
        tensor_path = shard / "tensors.safetensors"
        core = {
            "cache_key": manifest.get("cache_key"),
            "image_ids": manifest.get("image_ids"),
            "schema_version": "imagenetr50-row-tensor-cache-entry-v1",
            "semantic_values": manifest.get("semantic_values"),
            "tensor_sha256": file_sha256(tensor_path),
        }
        if manifest != {**core, "content_hash": record_sha256(core)}:
            raise ValueError("source row-cache shard does not authenticate")
        eligible += 1
        source_identities.append(str(manifest["content_hash"]))
        eligible_bytes += sum(
            (shard / name).stat().st_size
            for name in ("cache.json", "tensors.safetensors")
        )
        target = target_rows / shard.relative_to(source_rows)
        if target.is_dir():
            if any(
                file_sha256(target / name) != file_sha256(shard / name)
                for name in ("cache.json", "tensors.safetensors")
            ):
                raise ValueError("target cache shard differs from its source")
            reused += 1
            continue
        if target.exists():
            raise ValueError("target cache shard is a partial artifact")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
        try:
            for name in ("cache.json", "tensors.safetensors"):
                source_file = shard / name
                try:
                    os.link(source_file, temporary / name)
                except OSError:
                    shutil.copy2(source_file, temporary / name)
                    fallback_copies += 1
            os.rename(temporary, target)
            fsync_directory(target.parent)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        linked += 1
    return {
        "allowed_splits": sorted(allowed_splits),
        "eligible_bytes": eligible_bytes,
        "eligible_shards": eligible,
        "fallback_file_copies": fallback_copies,
        "linked_shards": linked,
        "reused_shards": reused,
        "schema_version": "imagenetr50-row-cache-seed-v1",
        "skipped_shards": skipped,
        "source_shards_hash": record_sha256(source_identities),
    }


def _stable_cache_seed_record(record: Mapping[str, object]) -> dict[str, object]:
    """Remove invocation-dependent link counts from immutable cache evidence."""
    return {
        key: value
        for key, value in record.items()
        if key not in {"fallback_file_copies", "linked_shards", "reused_shards"}
    }


def _preflight(
    bootstrap: BehaviorReplayBootstrap, device: torch.device
) -> dict[str, object]:
    target = bootstrap.store.run / "protocol" / "preflight.json"
    if target.is_file():
        return load_canonical_json(target)
    integrator = bootstrap.integrator
    train_ids = {row.image_id for row in integrator.manifest.images if row.split == "train"}
    test_ids = {row.image_id for row in integrator.manifest.images if row.split == "test"}
    state = create_integrator_state(
        COMMON_STATE_NAME,
        integrator.config.maximum_levels,
        bootstrap.config.feature_variant,
        integrator.config.optimization,
        bootstrap.config.seed,
        device,
    )
    features = torch.randn(3, state.model.input_dim, device=device)
    baseline = torch.randn(3, 200, device=device)
    mask = torch.ones(200, dtype=torch.bool, device=device)
    with torch.inference_mode():
        parity = float((state.model(features, baseline, mask) - baseline).abs().max())
    state.optimizer.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        loss = torch.nn.functional.cross_entropy(
            state.model(features, baseline, mask),
            torch.tensor((0, 1, 2), device=device),
        )
    loss.backward()
    state.optimizer.step()
    core: dict[str, object] = {
        "bf16_supported": bool(device.type == "cuda" and torch.cuda.is_bf16_supported()),
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(device),
        "feature_variant": bootstrap.config.feature_variant,
        "input_dimensions": state.model.input_dim,
        "parameter_count": parameter_count(state.model),
        "schema_version": "imagenetr50-logt-behavior-replay-preflight-v1",
        "slot_dimensions": state.model.slot_dim,
        "test_train_overlap": len(test_ids & train_ids),
        "train_images": len(train_ids),
        "test_images": len(test_ids),
        "one_step_loss": float(loss.detach()),
        "zero_residual_max_error": parity,
    }
    if (
        not core["bf16_supported"]
        or core["input_dimensions"] != 8214
        or core["slot_dimensions"] != 1369
        or core["test_train_overlap"] != 0
        or not math.isfinite(float(core["one_step_loss"]))
        or parity != 0.0
    ):
        raise RuntimeError("adapted-latent replay preflight failed")
    result = {**core, "content_hash": record_sha256(core)}
    publish_immutable_json(target, result)
    return result


def _persistent_behavior_run(
    bootstrap: IntegratorBootstrap,
    hierarchy: HierarchyBuildResult,
    historical_capacity: int,
    evaluation_stages: Sequence[int],
    device: torch.device,
) -> tuple[tuple[dict[str, object], ...], PersistentInvocation]:
    """Train or resume one arm with common initialization and adapted latents."""
    variant = "behavior"
    family = _persistent_family_name(
        hierarchy, variant, historical_capacity, "all_train", bootstrap.config.seed
    )
    state = create_integrator_state(
        COMMON_STATE_NAME,
        bootstrap.config.maximum_levels,
        variant,
        bootstrap.config.optimization,
        bootstrap.config.seed,
        device,
    )
    root = bootstrap.store.run / "integrators" / "persistent" / family
    training_ledger = ChainedJsonlLedger(
        root / "training_metrics.jsonl",
        "imagenetr50-integrator-stage-training-v1",
    )
    evaluation_ledger = ChainedJsonlLedger(
        root / "test_metrics.jsonl",
        "imagenetr50-integrator-stage-evaluation-v1",
    )
    completed_training = {int(row["stage"]): row for row in training_ledger.rows}
    completed_evaluations = {
        int(row["stage"]): row for row in evaluation_ledger.rows
    }
    requested_evaluations = frozenset(evaluation_stages)
    new_steps = new_training = new_evaluations = 0
    from tqdm.auto import tqdm

    stages = tqdm(
        range(1, bootstrap.config.tasks + 1),
        desc=f"adapted latent H={historical_capacity}",
        unit="task",
    )
    for stage in stages:
        nodes, slots, frontier_hash = _hierarchy_frontier(hierarchy, stage)
        training_rows = _persistent_training_rows(
            bootstrap, "all_train", stage, historical_capacity
        )
        current_rows = _partition_rows(bootstrap, "all_train", (stage - 1,))
        if (
            len(nodes) != stage.bit_count()
            or tuple(sorted(slots)) != slots
            or len(training_rows) > len(current_rows) + historical_capacity
        ):
            raise RuntimeError("adapted-latent arm violated its LogT/replay work bound")
        training_ids_hash = record_sha256([row.image_id for row in training_rows])
        checkpoint = root / "checkpoints" / f"stage_{stage:03d}.pt"
        tensors = fit_result = None
        if checkpoint.is_file():
            restore_integrator_checkpoint(
                checkpoint,
                state,
                stage,
                variant,
                frontier_hash,
                training_ids_hash,
            )
        else:
            tensors = _frontier_tensors(
                bootstrap,
                nodes,
                slots,
                bootstrap.config.maximum_levels,
                training_rows,
                device,
                f"hierarchy_{hierarchy.policy.content_hash}",
            )
            fit_result = fit_integrator_epochs(
                state,
                _supervision(tensors, variant),
                bootstrap.config.optimization.persistent_epochs,
                bootstrap.config.optimization,
                bootstrap.config.seed,
                stage,
                device,
            )
            save_integrator_checkpoint(
                checkpoint,
                state,
                stage,
                variant,
                frontier_hash,
                training_ids_hash,
            )
            new_steps += fit_result.optimizer_steps
            new_training += 1
        if stage not in completed_training:
            training_ledger.append(
                {
                    "base_example_forwards": (
                        0 if tensors is None else tensors.base_example_forwards
                    ),
                    "cache_hits": 0 if tensors is None else tensors.cache_hits,
                    "cache_misses": 0 if tensors is None else tensors.cache_misses,
                    "frontier_hash": frontier_hash,
                    "historical_capacity": historical_capacity,
                    "initialization_name": COMMON_STATE_NAME,
                    "integrator_optimizer_steps": state.optimizer_steps,
                    "live_nodes": len(nodes),
                    "node_example_forwards": (
                        0 if tensors is None else tensors.node_example_forwards
                    ),
                    "node_example_forwards_bound": len(training_rows) * len(nodes),
                    "parameter_count": parameter_count(state.model),
                    "stage": stage,
                    "training_examples": len(training_rows),
                    "training_fit": None if fit_result is None else asdict(fit_result),
                    "training_ids_hash": training_ids_hash,
                    "variant": variant,
                }
            )
        if stage in requested_evaluations and stage not in completed_evaluations:
            evaluation_rows = _test_rows(bootstrap, tuple(range(stage)))
            evaluation_tensors = _frontier_tensors(
                bootstrap,
                nodes,
                slots,
                bootstrap.config.maximum_levels,
                evaluation_rows,
                device,
                f"hierarchy_{hierarchy.policy.content_hash}",
            )
            predictions = predict_integrator(
                state.model,
                evaluation_tensors.observations(variant),
                bootstrap.config.optimization.batch_size,
                device,
            )
            evaluation_ledger.append(
                {
                    "accuracy": accuracy(predictions, evaluation_tensors.labels),
                    "controls": _control_metrics(evaluation_tensors),
                    "evaluation_examples": len(evaluation_rows),
                    "evaluation_partition": "test",
                    "frontier_hash": frontier_hash,
                    "live_nodes": len(nodes),
                    "stage": stage,
                    "task_accuracies": {
                        str(task + 1): accuracy(
                            predictions[evaluation_tensors.labels // 4 == task],
                            evaluation_tensors.labels[
                                evaluation_tensors.labels // 4 == task
                            ],
                        )
                        for task in range(stage)
                    },
                    "variant": variant,
                }
            )
            new_evaluations += 1
        stages.set_postfix(live=len(nodes), steps=state.optimizer_steps)
    training_ledger.require_unique_keys(("stage",))
    evaluation_ledger.require_unique_keys(("stage",))
    evaluations = {int(row["stage"]): row for row in evaluation_ledger.rows}
    combined = tuple(
        {**row, **evaluations.get(int(row["stage"]), {})}
        for row in training_ledger.rows
    )
    return combined, PersistentInvocation(
        historical_capacity, new_steps, new_training, new_evaluations
    )


def _training_seal(
    bootstrap: BehaviorReplayBootstrap,
    hierarchy: HierarchyBuildResult,
) -> dict[str, object]:
    target = bootstrap.store.run / "protocol" / "training_seal.json"
    if target.is_file():
        return load_canonical_json(target)
    behavior_ledger = ChainedJsonlLedger(
        bootstrap.store.run / "ledgers" / "behavior_requests.jsonl",
        "imagenetr50-integrator-behavior-request-v1",
    )
    test_requests = sum("test" in row["splits"] for row in behavior_ledger.rows)
    if test_requests:
        raise RuntimeError("test behavior was opened before all replay arms were trained")
    checkpoints = {
        str(capacity): file_sha256(
            bootstrap.store.run
            / "integrators"
            / "persistent"
            / _persistent_family_name(
                hierarchy,
                bootstrap.config.feature_variant,
                capacity,
                "all_train",
                bootstrap.config.seed,
            )
            / "checkpoints"
            / "stage_050.pt"
        )
        for capacity in bootstrap.config.historical_capacities
    }
    core: dict[str, object] = {
        "behavior_ledger_rows": len(behavior_ledger.rows),
        "behavior_ledger_tail_hash": behavior_ledger.tail_hash,
        "feature_variant": bootstrap.config.feature_variant,
        "final_frontier_hash": hierarchy.snapshots[-1].content_hash,
        "historical_capacities": list(bootstrap.config.historical_capacities),
        "persistent_checkpoint_sha256": checkpoints,
        "schema_version": "imagenetr50-logt-behavior-replay-training-seal-v1",
        "test_requests_before_seal": test_requests,
    }
    result = {**core, "content_hash": record_sha256(core)}
    publish_immutable_json(target, result)
    return result


def _publish_result(
    bootstrap: BehaviorReplayBootstrap,
    hierarchy: HierarchyBuildResult,
    condition_rows: Mapping[int, Sequence[Mapping[str, object]]],
) -> dict[str, object]:
    target = bootstrap.store.run / "evaluations" / "result.json"
    if target.is_file():
        return load_canonical_json(target)
    source_run = bootstrap.source.integrator.store.run
    source_locked_path = source_run / "evaluations" / "locked_test.json"
    source_joint_path = source_run / "evaluations" / "stage_matched_joint_iid.json"
    if (
        file_sha256(source_locked_path) != bootstrap.config.source_locked_test_sha256
        or not source_joint_path.is_file()
    ):
        raise ValueError("authenticated source comparisons are unavailable")
    publish_immutable_bytes(
        bootstrap.store.run / "evaluations" / "source_locked_test.json",
        source_locked_path.read_bytes(),
    )
    publish_immutable_bytes(
        bootstrap.store.run / "evaluations" / "stage_matched_joint_iid.json",
        source_joint_path.read_bytes(),
    )
    source_locked = load_canonical_json(source_locked_path)
    conditions = []
    task_matrix = []
    reference_controls: tuple[dict[str, object], ...] | None = None
    for capacity in bootstrap.config.historical_capacities:
        rows = tuple(dict(row) for row in condition_rows[capacity] if "accuracy" in row)
        if [int(row["stage"]) for row in rows] != list(range(1, 51)):
            raise ValueError("an adapted-latent replay arm lacks its complete test curve")
        controls = tuple(dict(row["controls"]) for row in rows)
        if reference_controls is not None and controls != reference_controls:
            raise ValueError("capacity arms were not evaluated against identical controls")
        reference_controls = controls
        training_rows = tuple(dict(row) for row in condition_rows[capacity])
        conditions.append(
            {
                "condition": f"behavior_h{capacity}",
                "feature_variant": bootstrap.config.feature_variant,
                "final_accuracy": float(rows[-1]["accuracy"]),
                "historical_capacity": capacity,
                "incremental_accuracy": math.fsum(
                    float(row["accuracy"]) for row in rows
                )
                / len(rows),
                "stage_metrics": list(rows),
                "work": {
                    "base_example_forwards": sum(
                        int(row.get("base_example_forwards", 0))
                        for row in training_rows
                    ),
                    "cache_hits": sum(
                        int(row.get("cache_hits", 0)) for row in training_rows
                    ),
                    "cache_misses": sum(
                        int(row.get("cache_misses", 0)) for row in training_rows
                    ),
                    "image_presentations": sum(
                        int(row["training_examples"])
                        * bootstrap.integrator.config.optimization.persistent_epochs
                        for row in training_rows
                    ),
                    "node_example_forwards": sum(
                        int(row.get("node_example_forwards", 0))
                        for row in training_rows
                    ),
                    "node_example_forwards_bound": sum(
                        int(row["node_example_forwards_bound"])
                        for row in training_rows
                    ),
                    "optimizer_steps": int(
                        training_rows[-1]["integrator_optimizer_steps"]
                    ),
                    "parameter_count": int(training_rows[-1]["parameter_count"]),
                    "training_wall_seconds": math.fsum(
                        float(dict(row.get("training_fit") or {}).get("wall_seconds", 0.0))
                        for row in training_rows
                    ),
                },
            }
        )
        task_matrix.extend(
            {
                "accuracy": float(value),
                "condition": f"behavior_h{capacity}",
                "historical_capacity": capacity,
                "stage": int(row["stage"]),
                "task": int(task),
            }
            for row in rows
            for task, value in dict(row["task_accuracies"]).items()
        )
    core: dict[str, object] = {
        "conditions": conditions,
        "feature_layout": {
            "fields": [
                {"dimensions": dimensions, "name": name}
                for name, dimensions in FEATURE_LAYOUT
            ],
            "maximum_slots": bootstrap.integrator.config.maximum_levels,
            "slot_dimensions": VARIANT_SLOT_DIMS[bootstrap.config.feature_variant],
            "total_dimensions": (
                bootstrap.integrator.config.maximum_levels
                * VARIANT_SLOT_DIMS[bootstrap.config.feature_variant]
            ),
        },
        "hierarchy_policy_hash": hierarchy.policy.content_hash,
        "local_references": dict(source_locked["local_references"]),
        "protocol_hash": bootstrap.protocol.content_hash,
        "role": "post_hoc_descriptive_replay_capacity_sweep",
        "schema_version": "imagenetr50-logt-behavior-replay-result-v1",
        "source_score_integrator": {
            "final_accuracy": float(source_locked["last_accuracy"]),
            "historical_capacity": 2048,
            "incremental_accuracy": float(source_locked["incremental_accuracy"]),
        },
        "task_accuracy_matrix": task_matrix,
    }
    result = {**core, "content_hash": record_sha256(core)}
    publish_immutable_json(target, result)
    return result


def _reuse_proof(
    bootstrap: BehaviorReplayBootstrap,
    hierarchy: HierarchyBuildResult,
    source_before: Mapping[str, object],
    device: torch.device,
) -> dict[str, object]:
    target = bootstrap.store.run / "protocol" / "reuse_proof.json"
    if target.is_file():
        return load_canonical_json(target)
    checkpoint_root = bootstrap.store.run / "integrators" / "persistent"
    before = _stat_fingerprint(checkpoint_root, "*.pt")
    invocations = tuple(
        _persistent_behavior_run(
            bootstrap.integrator,
            hierarchy,
            capacity,
            tuple(range(1, 51)),
            device,
        )[1]
        for capacity in bootstrap.config.historical_capacities
    )
    rebuilt = _build(bootstrap.source, "all_train", 50, device, progress=False)
    after = _stat_fingerprint(checkpoint_root, "*.pt")
    source_after = _stat_fingerprint(
        bootstrap.source.integrator.store.run
        / "hierarchies"
        / bootstrap.config.source_hierarchy_policy_hash
    )
    checks = {
        "all_integrator_checkpoints_unchanged": before == after,
        "all_replay_arms_zero_new_evaluations": all(
            invocation.new_evaluation_stages == 0 for invocation in invocations
        ),
        "all_replay_arms_zero_new_optimizer_steps": all(
            invocation.new_optimizer_steps == 0 for invocation in invocations
        ),
        "source_hierarchy_unchanged": dict(source_before) == source_after,
        "source_leaf_optimizer_steps_zero": rebuilt.work.leaf_optimizer_steps == 0,
        "source_parent_optimizer_steps_zero": rebuilt.work.parent_optimizer_steps == 0,
    }
    core: dict[str, object] = {
        "acceptance": checks,
        "checkpoint_fingerprint_after": after,
        "checkpoint_fingerprint_before": before,
        "integrity_passed": all(checks.values()),
        "replay_invocations": [asdict(invocation) for invocation in invocations],
        "schema_version": "imagenetr50-logt-behavior-replay-reuse-proof-v1",
        "source_hierarchy_after": source_after,
        "source_hierarchy_before": dict(source_before),
        "source_hierarchy_rebuild_work": asdict(rebuilt.work),
    }
    result = {**core, "content_hash": record_sha256(core)}
    publish_immutable_json(target, result)
    return result


def run_behavior_replay(
    config_path: str | Path = DEFAULT_BEHAVIOR_REPLAY_CONFIG,
) -> Path:
    """Run or exactly resume the complete node-adapted replay-capacity sweep."""
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("the full-50 adapted-latent experiment requires BF16 CUDA")
    started = time.monotonic()
    bootstrap = bootstrap_behavior_replay(config_path)
    device = torch.device("cuda:0")
    from tqdm.auto import tqdm

    overall = tqdm(total=7, desc="node-adapted replay sweep", unit="phase")
    print(f"Temporary/resumable artifact directory: {bootstrap.store.run}", flush=True)
    print("[phase 1/7] Adapted-latent MLP and split preflight", flush=True)
    _write_state(bootstrap, "PREFLIGHT")
    _preflight(bootstrap, device)
    overall.update(1)

    print("[phase 2/7] Authenticate and reuse the 50-task fresh-parent hierarchy", flush=True)
    source_hierarchy_root = (
        bootstrap.source.integrator.store.run
        / "hierarchies"
        / bootstrap.config.source_hierarchy_policy_hash
    )
    source_before = _stat_fingerprint(source_hierarchy_root)
    hierarchy = _build(bootstrap.source, "all_train", 50, device, progress=False)
    if (
        hierarchy.policy.content_hash != bootstrap.config.source_hierarchy_policy_hash
        or hierarchy.work.leaf_optimizer_steps
        or hierarchy.work.parent_optimizer_steps
    ):
        raise RuntimeError("the promoted hierarchy was not reused exactly")
    overall.update(1)

    image_splits = {
        row.image_id: row.split for row in bootstrap.integrator.manifest.images
    }
    print("[phase 3/7] Seed training-only adapted behavior cache", flush=True)
    train_cache_seed = seed_row_cache_shards(
        bootstrap.source.integrator.store.run / "cache" / "behaviors",
        bootstrap.store.run / "cache" / "behaviors",
        image_splits,
        frozenset({"train"}),
    )
    publish_immutable_json(
        bootstrap.store.run / "protocol" / "train_cache_seed.json",
        _stable_cache_seed_record(train_cache_seed),
    )
    overall.update(1)

    print("[phase 4/7] Train H=8192, 4096, and 2048 before test access", flush=True)
    _write_state(bootstrap, "TRAIN_ALL_CAPACITIES")
    training_invocations = []
    training_rows: dict[int, tuple[dict[str, object], ...]] = {}
    for capacity in reversed(bootstrap.config.historical_capacities):
        rows, invocation = _persistent_behavior_run(
            bootstrap.integrator, hierarchy, capacity, (), device
        )
        training_rows[capacity] = rows
        training_invocations.append(invocation)
        torch.cuda.empty_cache()
    seal = _training_seal(bootstrap, hierarchy)
    overall.update(1)

    print("[phase 5/7] Open locked test behaviors and evaluate all 50 stages", flush=True)
    _write_state(bootstrap, "LOCKED_TEST_EVALUATION", training_seal=seal)
    all_cache_seed = seed_row_cache_shards(
        bootstrap.source.integrator.store.run / "cache" / "behaviors",
        bootstrap.store.run / "cache" / "behaviors",
        image_splits,
        frozenset({"train", "test"}),
    )
    publish_immutable_json(
        bootstrap.store.run / "protocol" / "all_cache_seed.json",
        _stable_cache_seed_record(all_cache_seed),
    )
    evaluated_rows: dict[int, tuple[dict[str, object], ...]] = {}
    evaluation_invocations = []
    for capacity in bootstrap.config.historical_capacities:
        rows, invocation = _persistent_behavior_run(
            bootstrap.integrator,
            hierarchy,
            capacity,
            tuple(range(1, 51)),
            device,
        )
        evaluated_rows[capacity] = rows
        evaluation_invocations.append(invocation)
        torch.cuda.empty_cache()
    result = _publish_result(bootstrap, hierarchy, evaluated_rows)
    overall.update(1)

    print("[phase 6/7] Prove zero-work exact resume", flush=True)
    proof = _reuse_proof(bootstrap, hierarchy, source_before, device)
    if not bool(proof["integrity_passed"]):
        raise RuntimeError("adapted-latent replay reuse proof failed")
    overall.update(1)

    print("[phase 7/7] Generate Markdown, self-contained HTML, PDF, and tables", flush=True)
    elapsed = time.monotonic() - started
    invocation_core: dict[str, object] = {
        "elapsed_seconds": elapsed,
        "evaluation_invocations": [asdict(value) for value in evaluation_invocations],
        "schema_version": "imagenetr50-logt-behavior-replay-invocation-v1",
        "training_invocations": [asdict(value) for value in training_invocations],
    }
    atomic_write(
        bootstrap.store.run / "state" / "last_invocation.json",
        canonical_json_bytes(
            {**invocation_core, "content_hash": record_sha256(invocation_core)}
        ),
    )
    _write_state(bootstrap, "COMPLETE", result=result, reuse_proof=proof)
    from apm.continual.vision.imagenetr.behavior_replay_reporting import (
        write_behavior_replay_report,
    )

    write_behavior_replay_report(bootstrap.store.run)
    overall.update(1)
    overall.close()
    print(f"Adapted-latent replay report complete in {elapsed:.1f} seconds.", flush=True)
    return bootstrap.store.run


if __name__ == "__main__":
    print(run_behavior_replay())


__all__ = [
    "BehaviorReplayBootstrap",
    "BehaviorReplayProtocol",
    "PersistentInvocation",
    "bootstrap_behavior_replay",
    "run_behavior_replay",
    "seed_row_cache_shards",
]

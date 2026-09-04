"""Full-50 confirmation of a parent recipe selected on clean development data."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import math

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
)
from apm.continual.vision.imagenetr.integrator_artifacts import (
    HierarchyPolicy,
    IntegratorBootstrap,
    IntegratorProtocol,
    IntegratorStore,
)
from apm.continual.vision.imagenetr.integrator_hierarchy import (
    HierarchyParentRecipe,
    build_hierarchy,
)
from apm.continual.vision.imagenetr.integrator_workflow import (
    _hierarchy_controls,
    _local_reference_results,
    _persistent_family_name,
    _persistent_run,
    run_preflight,
)
from apm.continual.vision.imagenetr.manifests import installed_environment_manifest
from apm.continual.vision.imagenetr.model import create_pinned_backbone
from apm.continual.vision.imagenetr.parent_recipe_factorial import (
    ParentFactorialSource,
    load_parent_factorial_source,
    load_parent_recipe_config,
)
from apm.continual.vision.imagenetr.promoted_integrator_config import (
    DEFAULT_PROMOTED_CONFIG,
    PromotedIntegratorConfig,
    load_promoted_integrator_config,
)
from apm.continual.vision.imagenetr.protocol import material_tree_manifest
from apm.continual.vision.imagenetr.router_artifacts import load_sealed_tree


PROMOTED_PACKAGES = (
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
class PromotedBootstrap:
    """Prepared full-run context plus the exact selected parent recipe."""

    integrator: IntegratorBootstrap
    promotion: PromotedIntegratorConfig
    factorial_run: Path
    source: ParentFactorialSource
    factorial_summary: dict[str, object]
    parent_recipe: HierarchyParentRecipe


def _material_paths(project_root: Path, config_path: Path) -> tuple[Path, ...]:
    package = project_root / "src/apm/continual/vision/imagenetr"
    return (
        config_path,
        project_root / "scripts/vision/imagenetr/run_promoted_integrator_local.sh",
        project_root / "src/apm/continual/artifacts.py",
        *sorted(package.glob("integrator_*.py")),
        *(package / filename for filename in (
            "artifacts.py",
            "calibration.py",
            "config.py",
            "data.py",
            "heads.py",
            "lora.py",
            "manifests.py",
            "model.py",
            "parent_recipe_factorial.py",
            "promoted_integrator_config.py",
            "promoted_integrator_reporting.py",
            "promoted_integrator_workflow.py",
            "protocol.py",
            "proxy_memory.py",
            "router_artifacts.py",
            "router_features.py",
            "router_protocol.py",
            "training.py",
        )),
        package / "merging" / "common.py",
    )


def _selected_condition(
    config: PromotedIntegratorConfig, summary: Mapping[str, object]
) -> dict[str, object]:
    selection = dict(summary["selection"])
    if (
        not bool(selection.get("full50_triggered"))
        or selection.get("selected_condition") != config.selected_condition
    ):
        raise ValueError("promoted recipe was not selected by the frozen factorial")
    condition = next(
        dict(row)
        for row in summary["conditions"]
        if row["condition_key"] == config.selected_condition
    )
    if (
        condition["head_initialization"] != config.head_initialization
        or float(condition["weight_decay"]) != config.weight_decay
        or condition["seed_schedule"] != config.seed_schedule
    ):
        raise ValueError("promoted recipe fields differ from the selected factorial cell")
    return condition


def bootstrap_promoted_integrator(
    config_path: str | Path = DEFAULT_PROMOTED_CONFIG,
) -> PromotedBootstrap:
    """Bind the selected development result into a new full-50 run namespace."""
    resolved = Path(config_path).resolve()
    project_root = resolved.parents[3]
    promotion = load_promoted_integrator_config(resolved)
    factorial_config = load_parent_recipe_config(promotion.factorial_config)
    factorial_run = (
        factorial_config.artifact_root / "runs" / promotion.factorial_run_hash
    )
    factorial_protocol = load_canonical_json(
        factorial_run / "protocol" / "protocol.json"
    )
    if factorial_protocol.get("content_hash") != promotion.factorial_run_hash:
        raise ValueError("configured completed factorial run does not authenticate")
    summary_path = factorial_run / "summary.json"
    if file_sha256(summary_path) != promotion.factorial_summary_sha256:
        raise ValueError("selected factorial summary bytes changed")
    summary = load_canonical_json(summary_path)
    _selected_condition(promotion, summary)
    source = load_parent_factorial_source(factorial_config, project_root)
    runtime_config = replace(source.config, artifact_root=promotion.artifact_root)
    primary_root = (
        runtime_config.inference_artifact_root
        / "runs"
        / runtime_config.sealed_run_hash
    )
    sealed_tree = load_sealed_tree(
        primary_root, "I-U100", runtime_config.sealed_u100_policy_hash
    )
    code = material_tree_manifest(_material_paths(project_root, resolved))
    environment = installed_environment_manifest(PROMOTED_PACKAGES)
    missing = tuple(
        str(row["name"])
        for row in environment["packages"]
        if row["version"] == "MISSING"
    )
    if missing:
        raise RuntimeError(f"isolated vision environment is incomplete: {missing}")
    reference_results = load_canonical_json(
        source.run_root / "protocol" / "reference_results.json"
    )
    model_manifest = load_canonical_json(
        source.run_root / "protocol" / "model_manifest.json"
    )
    protocol = IntegratorProtocol(
        runtime_config.sealed_run_hash,
        runtime_config.sealed_u100_policy_hash,
        tuple(node.node_hash for node in sealed_tree.final.nodes),
        source.manifest.content_hash,
        source.model_manifest_hash,
        source.split.content_hash,
        promotion.config_hash,
        str(code["content_hash"]),
        str(environment["content_hash"]),
        str(reference_results["content_hash"]),
    )
    store = IntegratorStore(promotion.artifact_root, protocol.content_hash)
    store.prepare(protocol)
    for filename, record in (
        ("code_manifest.json", code),
        ("environment_manifest.json", environment),
        ("factorial_summary.json", summary),
        ("model_manifest.json", model_manifest),
        ("router_split.json", source.split.as_record()),
        ("reference_results.json", reference_results),
    ):
        publish_immutable_json(store.run / "protocol" / filename, record)
    resolved_record = {
        **runtime_config.as_record(),
        "promotion": promotion.as_record(),
        "promoted_config_hash": promotion.config_hash,
    }
    publish_immutable_bytes(
        store.run / "config_resolved.json", canonical_json_bytes(resolved_record)
    )
    atomic_write(
        promotion.artifact_root / "LATEST_RUN.json",
        canonical_json_bytes(
            {
                "run_hash": protocol.content_hash,
                "schema_version": "imagenetr50-promoted-integrator-latest-v1",
            }
        ),
    )
    parent_recipe = HierarchyParentRecipe(
        promotion.head_initialization,
        promotion.seed_schedule,
        replace(
            runtime_config.consolidation_training,
            weight_decay=promotion.weight_decay,
        ),
    )
    integrator = IntegratorBootstrap(
        project_root,
        resolved,
        runtime_config,
        source.primary_config,
        source.manifest,
        source.split,
        sealed_tree,
        protocol,
        store,
        source.checkpoint,
        source.train_transform,
        source.test_transform,
        model_manifest,
        code,
        environment,
    )
    return PromotedBootstrap(
        integrator, promotion, factorial_run, source, summary, parent_recipe
    )


def _policy(bootstrap: PromotedBootstrap, partition: str) -> HierarchyPolicy:
    training_hash = record_sha256(
        {
            "leaf_training": asdict(
                bootstrap.integrator.config.consolidation_training
            ),
            "parent_recipe": {
                **asdict(bootstrap.parent_recipe),
                "content_hash": bootstrap.parent_recipe.content_hash,
            },
            "schema_version": "imagenetr50-promoted-hierarchy-training-v1",
        }
    )
    return HierarchyPolicy(
        partition,
        "full_union",
        bootstrap.integrator.config.source_identity_capacity,
        training_hash,
        bootstrap.integrator.config.seed,
    )


def _write_state(bootstrap: IntegratorBootstrap, phase: str, **values: object) -> None:
    atomic_write(
        bootstrap.store.run / "state" / "workflow.json",
        canonical_json_bytes(
            {
                "phase": phase,
                "run_hash": bootstrap.protocol.content_hash,
                "schema_version": "imagenetr50-promoted-integrator-workflow-state-v1",
                **values,
            }
        ),
    )


def _build(
    bootstrap: PromotedBootstrap,
    partition: str,
    tasks: int,
    device: torch.device,
    progress: bool = True,
):
    return build_hierarchy(
        bootstrap.integrator,
        _policy(bootstrap, partition),
        tasks,
        lambda: create_pinned_backbone(bootstrap.integrator.checkpoint),
        device,
        progress=progress,
        parent_recipe=bootstrap.parent_recipe,
    )


def _run_smoke(
    bootstrap: PromotedBootstrap, device: torch.device
) -> dict[str, object]:
    target = bootstrap.integrator.store.run / "evaluations" / "smoke.json"
    if target.is_file():
        return load_canonical_json(target)
    hierarchy = _build(bootstrap, "fit", 8, device)
    replayed = _build(bootstrap, "fit", 8, device, progress=False)
    checks = {
        "artifact_reuse": (
            replayed.work.leaf_optimizer_steps == 0
            and replayed.work.parent_optimizer_steps == 0
            and tuple(node.artifact.content_hash for node in replayed.nodes)
            == tuple(node.artifact.content_hash for node in hierarchy.nodes)
        ),
        "full_source_membership": all(
            node.artifact.represented_train_image_count
            == len(node.artifact.proxy_image_ids)
            for node in hierarchy.nodes
        ),
        "one_live_root": len(hierarchy.frontier(8)) == 1,
        "seven_carries": len(hierarchy.nodes) - 8 == 7,
        "validation_excluded": not bool(
            set(bootstrap.integrator.split.validation_image_ids)
            & {
                image_id
                for node in hierarchy.nodes
                for image_id in node.artifact.proxy_image_ids
            }
        ),
    }
    core: dict[str, object] = {
        "acceptance": checks,
        "hierarchy_work": asdict(hierarchy.work),
        "integrity_passed": all(checks.values()),
        "parent_recipe_hash": bootstrap.parent_recipe.content_hash,
        "reuse_work": asdict(replayed.work),
        "schema_version": "imagenetr50-promoted-integrator-smoke-v1",
    }
    result = {**core, "content_hash": record_sha256(core)}
    publish_immutable_json(target, result)
    return result


def _run_locked(
    bootstrap: PromotedBootstrap, device: torch.device
) -> dict[str, object]:
    integrator = bootstrap.integrator
    target = integrator.store.run / "evaluations" / "locked_test.json"
    if target.is_file():
        return load_canonical_json(target)
    history = integrator.config.selection.historical_capacity
    variant = integrator.config.selection.feature_variant
    selection = {
        "selected_historical_capacity": history,
        "selected_parent_recipe": bootstrap.promotion.selected_condition,
        "selected_variant": variant,
    }
    matrix_core: dict[str, object] = {
        "historical_capacity": history,
        "parent_recipe": {
            **asdict(bootstrap.parent_recipe),
            "content_hash": bootstrap.parent_recipe.content_hash,
        },
        "protocol_hash": integrator.protocol.content_hash,
        "schema_version": "imagenetr50-promoted-integrator-locked-matrix-v1",
        "test_rows_opened_before_seal": 0,
        "variant": variant,
    }
    publish_immutable_json(
        integrator.store.run / "protocol" / "locked_matrix.json",
        {**matrix_core, "content_hash": record_sha256(matrix_core)},
    )
    hierarchy = _build(bootstrap, "all_train", 50, device)
    _state, _training_only = _persistent_run(
        integrator,
        hierarchy,
        variant,
        history,
        "all_train",
        "test",
        50,
        (),
        integrator.config.seed,
        device,
    )
    behavior_ledger = ChainedJsonlLedger(
        integrator.store.run / "ledgers" / "behavior_requests.jsonl",
        "imagenetr50-integrator-behavior-request-v1",
    )
    if any("test" in row["splits"] for row in behavior_ledger.rows):
        raise RuntimeError("test behavior was opened before promoted training completed")
    training_seal_core = {
        "final_frontier_hash": hierarchy.snapshots[-1].content_hash,
        "parent_recipe_hash": bootstrap.parent_recipe.content_hash,
        "persistent_checkpoint_sha256": file_sha256(
            integrator.store.run
            / "integrators"
            / "persistent"
            / _persistent_family_name(
                hierarchy,
                variant,
                history,
                "all_train",
                integrator.config.seed,
            )
            / "checkpoints"
            / "stage_050.pt"
        ),
        "schema_version": "imagenetr50-promoted-integrator-training-seal-v1",
        "test_requests_before_seal": 0,
    }
    publish_immutable_json(
        integrator.store.run / "protocol" / "locked_training_seal.json",
        {**training_seal_core, "content_hash": record_sha256(training_seal_core)},
    )
    _state, persistent = _persistent_run(
        integrator,
        hierarchy,
        variant,
        history,
        "all_train",
        "test",
        50,
        tuple(range(1, 51)),
        integrator.config.seed,
        device,
    )
    stage_rows = tuple(row for row in persistent if "accuracy" in row)
    if len(stage_rows) != 50:
        raise ValueError("promoted locked benchmark lacks a complete 50-stage curve")
    last = float(stage_rows[-1]["accuracy"])
    incremental = math.fsum(float(row["accuracy"]) for row in stage_rows) / 50
    final_controls = _hierarchy_controls(
        integrator, hierarchy, (50,), "test", device
    )[0]
    task_matrix = tuple(
        {"accuracy": value, "stage": int(row["stage"]), "task": int(task)}
        for row in stage_rows
        for task, value in row["task_accuracies"].items()
    )
    references = _local_reference_results(integrator)
    core = {
        "comparisons": {
            "incremental_minus_joint_iid": incremental
            - references["joint_iid_incremental"],
            "incremental_minus_local_e2": incremental
            - references["local_e2_incremental"],
            "last_minus_joint_iid": last - references["joint_iid_last"],
            "last_minus_local_e2": last - references["local_e2_last"],
        },
        "final_static_controls": final_controls,
        "incremental_accuracy": incremental,
        "last_accuracy": last,
        "local_e2_incremental": references["local_e2_incremental"],
        "local_e2_last": references["local_e2_last"],
        "local_references": references,
        "parent_recipe": matrix_core["parent_recipe"],
        "role": "development_promoted_descriptive_benchmark",
        "schema_version": "imagenetr50-promoted-integrator-locked-test-v1",
        "selection": selection,
        "stage_metrics": list(stage_rows),
        "task_accuracy_matrix": list(task_matrix),
    }
    result = {**core, "content_hash": record_sha256(core)}
    publish_immutable_json(target, result)
    return result


def _copy_stage_matched_curve(bootstrap: PromotedBootstrap) -> None:
    source = (
        bootstrap.source.run_root
        / "evaluations"
        / "stage_matched_joint_iid.json"
    )
    if not source.is_file():
        raise FileNotFoundError("the authenticated stage-matched joint curve is missing")
    atomic_write(
        bootstrap.integrator.store.run
        / "evaluations"
        / "stage_matched_joint_iid.json",
        source.read_bytes(),
    )


def run_promoted_integrator(
    config_path: str | Path = DEFAULT_PROMOTED_CONFIG,
) -> Path:
    """Run or resume the selected full-50 hierarchy and persistent integrator."""
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("the promoted full-50 experiment requires BF16 CUDA")
    bootstrap = bootstrap_promoted_integrator(config_path)
    integrator = bootstrap.integrator
    device = torch.device("cuda:0")
    from tqdm.auto import tqdm

    overall = tqdm(total=4, desc="promoted full-50 integrator", unit="phase")
    print(f"Temporary/resumable artifact directory: {integrator.store.run}", flush=True)
    _write_state(integrator, "PREFLIGHT")
    run_preflight(integrator, device)
    overall.update(1)
    _write_state(integrator, "SMOKE")
    smoke = _run_smoke(bootstrap, device)
    overall.update(1)
    if not bool(smoke["integrity_passed"]):
        _write_state(integrator, "BLOCKED_SMOKE_INTEGRITY", smoke=smoke)
        overall.update(2)
        overall.close()
        return integrator.store.run
    _write_state(integrator, "LOCKED_TRAIN_THEN_TEST")
    locked = _run_locked(bootstrap, device)
    overall.update(1)
    _copy_stage_matched_curve(bootstrap)
    _write_state(integrator, "COMPLETE", locked_test=locked)
    from apm.continual.vision.imagenetr.promoted_integrator_reporting import (
        write_promoted_integrator_report,
    )

    write_promoted_integrator_report(integrator.store.run)
    overall.update(1)
    overall.close()
    print("Promoted full-50 report complete.", flush=True)
    return integrator.store.run


if __name__ == "__main__":
    print(run_promoted_integrator())


__all__ = [
    "PromotedBootstrap",
    "bootstrap_promoted_integrator",
    "run_promoted_integrator",
]

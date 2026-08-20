"""One resumable local workflow from immutable protocol preparation through report."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from collections.abc import Callable, Mapping, Sequence
import json
import math
import time

import torch

from apm.continual.artifacts import (
    atomic_write,
    canonical_json_bytes,
    load_canonical_json,
    publish_immutable_bytes,
    publish_immutable_json,
    record_sha256,
)
from apm.continual.vision.imagenetr.activation_cache import FrozenActivationProvider
from apm.continual.vision.imagenetr.artifacts import NodeBundle, VisionStore
from apm.continual.vision.imagenetr.baselines import (
    BaselineExecution,
    baseline_job_spec,
    train_frozen_reference,
    train_joint_iid_lora,
    train_sequential_lora,
)
from apm.continual.vision.imagenetr.config import ImageNetRConfig, load_config
from apm.continual.vision.imagenetr.constants import (
    CORE_SPACE_REVISION,
    E2LORA_REVISION,
    TIMM_MODEL_NAME,
    TIMM_MODEL_REPOSITORY,
    TIMM_MODEL_REVISION,
)
from apm.continual.vision.imagenetr.controls import baseline_history_tree, leaf_bank_tree
from apm.continual.vision.imagenetr.data import (
    DatasetManifest,
    image_transforms,
    prepare_dataset,
)
from apm.continual.vision.imagenetr.evaluation import (
    EvaluationResult,
    StageAccuracy,
    TaskAccuracy,
    evaluate_tree,
)
from apm.continual.vision.imagenetr.exact_diagnostics import run_exact_rank_diagnostics
from apm.continual.vision.imagenetr.external import E2LoRAResult, run_official_e2lora
from apm.continual.vision.imagenetr.leaf_training import (
    LeafExecution,
    leaf_job_spec,
    train_leaf,
)
from apm.continual.vision.imagenetr.lineage import TreeBuildResult, build_tree
from apm.continual.vision.imagenetr.manifests import (
    git_commit_or_unknown,
    installed_environment_manifest,
    model_file_manifest,
)
from apm.continual.vision.imagenetr.model import (
    create_pinned_backbone,
    download_pinned_checkpoint,
)
from apm.continual.vision.imagenetr.preflight import PreflightResult, run_preflight
from apm.continual.vision.imagenetr.protocol import (
    JobSpec,
    MergePolicy,
    ResolvedProtocol,
    material_tree_manifest,
)
from apm.continual.vision.imagenetr.reporting import write_report
from apm.continual.vision.imagenetr.scheduler import LocalScheduler


@dataclass(frozen=True, slots=True)
class Bootstrap:
    """All resolved immutable inputs needed by scientific jobs."""

    config: ImageNetRConfig
    manifest: DatasetManifest
    prepared_root: Path
    checkpoint: Path
    protocol: ResolvedProtocol
    store: VisionStore
    software_manifest_hash: str
    git_commit: str
    train_transform: object
    test_transform: object


def _project_root(config_path: Path) -> Path:
    return config_path.resolve().parents[3]


def bootstrap_protocol(config_path: str | Path) -> Bootstrap:
    """Freeze dataset/model/code/environment/config manifests before any optimizer step."""
    source = Path(config_path).resolve()
    project_root = _project_root(source)
    config = load_config(source)
    print("Phase 1/12: authenticating and freezing ImageNet-R dataset membership.", flush=True)
    prepared_root, manifest = prepare_dataset(config.data_root, config.seed)
    print("Phase 2/12: downloading and authenticating the pinned timm checkpoint.", flush=True)
    checkpoint = download_pinned_checkpoint(config.data_root / "model_cache")
    model_manifest = model_file_manifest(
        checkpoint, TIMM_MODEL_REPOSITORY, TIMM_MODEL_REVISION, TIMM_MODEL_NAME
    )
    environment = installed_environment_manifest(
        (
            "apm",
            "huggingface_hub",
            "matplotlib",
            "numpy",
            "pandas",
            "Pillow",
            "pyarrow",
            "PyYAML",
            "safetensors",
            "scipy",
            "timm",
            "torch",
            "torchvision",
            "tqdm",
        )
    )
    missing = [
        row["name"]
        for row in environment["packages"]
        if row["version"] == "MISSING"
    ]
    if missing:
        raise RuntimeError(f"isolated vision environment is incomplete: {missing}")
    code_manifest = material_tree_manifest(
        (
            project_root / "src/apm/continual/vision",
            project_root / "src/apm/continual/artifacts.py",
            source,
            project_root / "scripts/vision/imagenetr",
        )
    )
    protocol = ResolvedProtocol(
        dataset_manifest_hash=manifest.content_hash,
        model_manifest_hash=str(model_manifest["content_hash"]),
        config_hash=config.config_hash,
        code_manifest_hash=str(code_manifest["content_hash"]),
        environment_manifest_hash=str(environment["content_hash"]),
        class_order=manifest.class_order,
    )
    store = VisionStore.from_protocol(protocol, config.artifact_root)
    store.prepare(protocol)
    protocol_root = store.run / "protocol"
    for filename, record in (
        ("dataset_manifest.json", manifest.as_record()),
        ("model_manifest.json", model_manifest),
        ("software_manifest.json", environment),
        ("code_manifest.json", code_manifest),
        (
            "class_order.json",
            {
                "class_order": list(manifest.class_order),
                "schema_version": "imagenetr50-class-order-v1",
                "seed": config.seed,
            },
        ),
        (
            "source_manifest.json",
            {
                "core_space_revision": CORE_SPACE_REVISION,
                "e2lora_revision": E2LORA_REVISION,
                "schema_version": "imagenetr50-source-revisions-v1",
            },
        ),
    ):
        publish_immutable_json(protocol_root / filename, record)
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - environment gate
        raise RuntimeError("PyYAML is required by the vision environment") from error
    publish_immutable_bytes(
        store.run / "config_resolved.yaml",
        yaml.safe_dump(config.as_record(), sort_keys=True).encode("utf-8"),
    )
    latest = {
        "run_hash": protocol.content_hash,
        "schema_version": "imagenetr50-latest-run-v1",
    }
    atomic_write(config.artifact_root / "LATEST_RUN.json", canonical_json_bytes(latest))
    train_transform, test_transform = image_transforms(config.input_size)
    return Bootstrap(
        config,
        manifest,
        prepared_root,
        checkpoint,
        protocol,
        store,
        str(environment["content_hash"]),
        git_commit_or_unknown(project_root),
        train_transform,
        test_transform,
    )


def _policy(config: ImageNetRConfig, method: str, repair_fraction: float) -> MergePolicy:
    repair_hash = record_sha256(
        {
            "fraction": repair_fraction,
            "reservoir": "deterministic_bottom_k_hash",
            "schema_version": "imagenetr50-repair-policy-v1",
            "training": asdict(config.repair_training),
        }
    )
    return MergePolicy(
        method=method,
        output_rank=config.output_rank,
        scale=config.merge_scale,
        weighting="source_image_count",
        repair_fraction=repair_fraction,
        repair_config_hash=repair_hash,
        proxy_size=config.proxy_images_per_node,
        core_space_revision=CORE_SPACE_REVISION if method == "core_tsv" else None,
    )


def _condition_name(policy: MergePolicy) -> str:
    """Return the stable report name, with repair encoded in percentage points."""
    if policy.method == "retrain_union":
        return f"logt_retrain_union_r{policy.output_rank}"
    method = "drift" if policy.method == "output_drift" else policy.method
    repair = int(round(100 * policy.repair_fraction))
    return f"logt_{method}_r{policy.output_rank}_repair{repair:03d}"


def _tree_job(
    run_hash: str,
    policy: MergePolicy,
    task_count: int,
    leaf_dependencies: Sequence[str],
) -> JobSpec:
    return JobSpec.create(
        run_hash,
        "build_tree",
        leaf_dependencies,
        {"policy_hash": policy.content_hash, "task_count": task_count},
    )


def _evaluation_record(result: EvaluationResult) -> dict[str, object]:
    return {
        "cache_hits": result.cache_hits,
        "cache_misses": result.cache_misses,
        "schema_version": "imagenetr50-evaluation-result-v1",
        "stages": [row.as_record() for row in result.stages],
        "tasks": [row.as_record() for row in result.tasks],
    }


def _load_evaluation(path: Path) -> EvaluationResult:
    record = load_canonical_json(path)
    if record["schema_version"] != "imagenetr50-evaluation-result-v1":
        raise ValueError("unknown persisted evaluation schema")
    return EvaluationResult(
        tuple(StageAccuracy(**row) for row in record["stages"]),
        tuple(TaskAccuracy(**row) for row in record["tasks"]),
        int(record["cache_hits"]),
        int(record["cache_misses"]),
    )


def _ensure_evaluation(
    scheduler: LocalScheduler,
    bootstrap: Bootstrap,
    tree: TreeBuildResult,
    condition: str,
    dependency: str,
    backbone_factory: Callable[[], torch.nn.Module],
) -> EvaluationResult:
    job = JobSpec.create(
        bootstrap.store.run_hash,
        "evaluate",
        (dependency,),
        {"condition": condition, "policy_hash": tree.policy.content_hash},
    )
    target = bootstrap.store.run / "evaluations" / condition / "evaluation.json"
    holder: dict[str, EvaluationResult] = {}

    def perform() -> Mapping[str, object]:
        result = evaluate_tree(
            bootstrap.store,
            tree,
            bootstrap.manifest,
            bootstrap.prepared_root,
            backbone_factory,
            bootstrap.test_transform,
            bootstrap.protocol.model_manifest_hash,
            record_sha256(
                {
                    "input_size": bootstrap.config.input_size,
                    "normalization": None,
                    "schema_version": "imagenetr50-test-transform-v1",
                }
            ),
            bootstrap.config.lora_rank,
            bootstrap.config.lora_alpha,
            bootstrap.config.cosine_scale,
            bootstrap.config.leaf_training.batch_size,
            bootstrap.config.num_workers,
            torch.device("cuda:0"),
            condition,
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        publish_immutable_json(target, _evaluation_record(result))
        holder["result"] = result
        return {"evaluation": str(target), "rows": len(result.stages)}

    scheduler.execute(job, perform)
    return holder.get("result") or _load_evaluation(target)


def _load_preflight(path: Path) -> PreflightResult:
    record = load_canonical_json(path)
    record.pop("schema_version")
    return PreflightResult(**record)


def _ensure_tree(
    scheduler: LocalScheduler,
    bootstrap: Bootstrap,
    leaves: Sequence[NodeBundle],
    policy: MergePolicy,
    task_count: int,
    leaf_dependencies: Sequence[str],
    backbone_factory: Callable[[], torch.nn.Module],
    activation_provider: FrozenActivationProvider,
) -> tuple[TreeBuildResult, str]:
    job = _tree_job(
        bootstrap.store.run_hash, policy, task_count, leaf_dependencies[:task_count]
    )
    holder: dict[str, TreeBuildResult] = {}

    def build() -> Mapping[str, object]:
        result = build_tree(
            bootstrap.store,
            leaves,
            bootstrap.manifest,
            bootstrap.prepared_root,
            bootstrap.config,
            policy,
            backbone_factory,
            bootstrap.train_transform,
            bootstrap.test_transform,
            torch.device("cuda:0"),
            bootstrap.software_manifest_hash,
            bootstrap.git_commit,
            activation_provider if policy.method == "output_drift" else None,
            task_count,
        )
        holder["result"] = result
        return {
            "leaf_optimizer_steps": result.leaf_optimizer_steps,
            "merge_events": result.merge_events,
            "policy_hash": policy.content_hash,
        }

    scheduler.execute(job, build)
    if "result" not in holder:
        holder["result"] = build_tree(
            bootstrap.store,
            leaves,
            bootstrap.manifest,
            bootstrap.prepared_root,
            bootstrap.config,
            policy,
            backbone_factory,
            bootstrap.train_transform,
            bootstrap.test_transform,
            torch.device("cuda:0"),
            bootstrap.software_manifest_hash,
            bootstrap.git_commit,
            activation_provider if policy.method == "output_drift" else None,
            task_count,
            show_progress=False,
        )
    return holder["result"], job.content_hash


def _preflight_acceptance(result: PreflightResult) -> None:
    if (
        not result.bf16_supported
        or result.batch_size != 64
        or not math.isfinite(result.one_step_loss)
        or result.dataset_images_per_second <= 0.0
        or result.peak_vram_bytes <= 0
    ):
        raise RuntimeError("real model/data/GPU preflight did not satisfy hard gates")


def _smoke_acceptance(evaluations: Sequence[EvaluationResult]) -> None:
    for result in evaluations:
        final_oracle = next(
            row.accuracy
            for row in result.stages
            if row.stage == 8 and row.score_mode == "true_node_oracle"
        )
        if not math.isfinite(final_oracle) or final_oracle <= 15.0:
            raise RuntimeError(
                f"8-task smoke true-node oracle is not healthy: {final_oracle:.3f}%"
            )


def run_workflow(config_path: str | Path) -> Path:
    """Run preflight, smoke, full primary DAG, reuse proofs, external control, and report."""
    bootstrap = bootstrap_protocol(config_path)
    config, store = bootstrap.config, bootstrap.store
    scheduler = LocalScheduler(
        store.run / "state" / "scheduler_state.json", store.run_hash
    )
    base_template = create_pinned_backbone(bootstrap.checkpoint)
    backbone_factory = lambda: deepcopy(base_template)

    print("Phase 3/12: real batch-64 BF16 model/data/GPU preflight.", flush=True)
    preflight_path = store.run / "protocol" / "preflight.json"
    preflight_job = JobSpec.create(
        store.run_hash,
        "preflight",
        payload={"batch_size": config.leaf_training.batch_size},
    )
    preflight_holder: dict[str, PreflightResult] = {}

    def preflight_handler() -> Mapping[str, object]:
        result = run_preflight(
            config,
            bootstrap.manifest,
            bootstrap.prepared_root,
            backbone_factory,
            bootstrap.train_transform,
        )
        publish_immutable_json(preflight_path, result.as_record())
        preflight_holder["result"] = result
        return result.as_record()

    scheduler.execute(preflight_job, preflight_handler)
    preflight = preflight_holder.get("result") or _load_preflight(preflight_path)
    _preflight_acceptance(preflight)

    leaf_jobs = tuple(
        leaf_job_spec(store.run_hash, task, bootstrap.manifest, config)
        for task in range(config.tasks)
    )
    leaves: list[NodeBundle] = []
    leaf_durations: list[float] = []

    def ensure_leaf(task: int) -> NodeBundle:
        holder: dict[str, LeafExecution] = {}
        started = time.monotonic()

        def handler() -> Mapping[str, object]:
            execution = train_leaf(
                store,
                bootstrap.manifest,
                bootstrap.prepared_root,
                config,
                task,
                backbone_factory,
                bootstrap.train_transform,
                torch.device("cuda:0"),
                bootstrap.software_manifest_hash,
                bootstrap.git_commit,
            )
            holder["execution"] = execution
            return {
                "artifact_hash": execution.bundle.artifact.content_hash,
                "optimizer_steps": execution.optimizer_steps_this_execution,
                "reused": execution.reused,
            }

        scheduler.execute(leaf_jobs[task], handler)
        execution = holder.get("execution")
        if execution is None:
            execution = train_leaf(
                store,
                bootstrap.manifest,
                bootstrap.prepared_root,
                config,
                task,
                backbone_factory,
                bootstrap.train_transform,
                torch.device("cuda:0"),
                bootstrap.software_manifest_hash,
                bootstrap.git_commit,
                show_progress=False,
            )
        leaf_durations.append(time.monotonic() - started)
        return execution.bundle

    print("Phase 4/12: training/reusing the first eight immutable leaves for smoke.", flush=True)
    leaves.extend(ensure_leaf(task) for task in range(config.smoke_tasks))
    activation_provider = FrozenActivationProvider(
        store.run / "cache" / "proxy_activations",
        bootstrap.manifest,
        bootstrap.prepared_root,
        backbone_factory,
        bootstrap.test_transform,
        bootstrap.protocol.model_manifest_hash,
        record_sha256(
            {"normalization": None, "schema_version": "imagenetr50-proxy-transform-v1"}
        ),
        config.lora_rank,
        config.lora_alpha,
        config.leaf_training.batch_size,
        config.num_workers,
        torch.device("cuda:0"),
    )
    smoke_policies = (
        _policy(config, "retrain_union", 0.0),
        _policy(config, "svd", 0.0),
        _policy(config, "core_tsv", 0.0),
        _policy(config, "output_drift", config.primary_repair_fraction),
    )
    smoke_evaluations = []
    for policy in smoke_policies:
        tree, tree_job_hash = _ensure_tree(
            scheduler,
            bootstrap,
            leaves,
            policy,
            config.smoke_tasks,
            tuple(job.content_hash for job in leaf_jobs),
            backbone_factory,
            activation_provider,
        )
        smoke_evaluations.append(
            _ensure_evaluation(
                scheduler,
                bootstrap,
                tree,
                f"smoke_{_condition_name(policy)}",
                tree_job_hash,
                backbone_factory,
            )
        )
        if not tree.leaf_hashes_unchanged or tree.leaf_optimizer_steps != 0:
            raise RuntimeError("smoke policy construction changed or retrained a leaf")
    _smoke_acceptance(smoke_evaluations)

    print("Phase 5/12: sealing all 50 immutable independent leaves.", flush=True)
    try:
        from tqdm.auto import tqdm
    except ImportError as error:  # pragma: no cover - environment gate
        raise RuntimeError("tqdm is required by the vision environment") from error
    progress = tqdm(total=50, initial=len(leaves), desc="immutable leaves", unit="leaf")
    for task in range(len(leaves), 50):
        leaves.append(ensure_leaf(task))
        progress.update(1)
        if len(leaves) >= 5:
            measured = sum(leaf_durations[-5:]) / min(5, len(leaf_durations))
            progress.set_postfix_str(f"measured ETA {(50-len(leaves))*measured/60:.1f}m")
    progress.close()

    print("Phase 6/12: controlled frozen, sequential, and joint-IID baselines.", flush=True)
    baseline_functions = {
        "frozen_reference": train_frozen_reference,
        "seq_lora_r16": train_sequential_lora,
        "joint_iid_lora_r16": train_joint_iid_lora,
    }
    baselines: dict[str, BaselineExecution] = {}
    for name, function in baseline_functions.items():
        job = baseline_job_spec(name, store, bootstrap.manifest, config)
        holder: dict[str, BaselineExecution] = {}

        def baseline_handler(
            selected: Callable[..., BaselineExecution] = function,
            label: str = name,
        ) -> Mapping[str, object]:
            execution = selected(
                store,
                bootstrap.manifest,
                bootstrap.prepared_root,
                config,
                backbone_factory,
                bootstrap.train_transform,
                torch.device("cuda:0"),
                bootstrap.software_manifest_hash,
                bootstrap.git_commit,
            )
            holder["execution"] = execution
            return {
                "artifact_hash": execution.bundle.artifact.content_hash,
                "optimizer_steps": execution.optimizer_steps_this_execution,
            }

        scheduler.execute(job, baseline_handler)
        baselines[name] = holder.get("execution") or function(
            store,
            bootstrap.manifest,
            bootstrap.prepared_root,
            config,
            backbone_factory,
            bootstrap.train_transform,
            torch.device("cuda:0"),
            bootstrap.software_manifest_hash,
            bootstrap.git_commit,
            show_progress=False,
        )

    print("Phase 7/12: full union-retrained logarithmic hierarchy.", flush=True)
    all_leaf_dependencies = tuple(job.content_hash for job in leaf_jobs)
    retrain_policy = _policy(config, "retrain_union", 0.0)
    retrain_tree, retrain_job = _ensure_tree(
        scheduler,
        bootstrap,
        leaves,
        retrain_policy,
        50,
        all_leaf_dependencies,
        backbone_factory,
        activation_provider,
    )

    print("Phase 8/12: SVD, Core+TSV, and output-drift trees at 0% and 5% repair.", flush=True)
    primary_policies = tuple(
        _policy(config, method, fraction)
        for method in ("svd", "core_tsv", "output_drift")
        for fraction in (0.0, config.primary_repair_fraction)
    )
    built_trees: dict[str, tuple[TreeBuildResult, str]] = {
        _condition_name(retrain_policy): (retrain_tree, retrain_job)
    }
    for policy in primary_policies:
        tree, job_hash = _ensure_tree(
            scheduler,
            bootstrap,
            leaves,
            policy,
            50,
            all_leaf_dependencies,
            backbone_factory,
            activation_provider,
        )
        built_trees[_condition_name(policy)] = (tree, job_hash)

    exact_tree_names = (
        "logt_retrain_union_r16",
        "logt_svd_r16_repair000",
        "logt_core_tsv_r16_repair000",
        "logt_drift_r16_repair000",
    )
    exact_job = JobSpec.create(
        store.run_hash,
        "exact_rank_diagnostics",
        tuple(built_trees[name][1] for name in exact_tree_names),
        {"events": 6, "stored_rank": 2 * config.lora_rank},
    )
    exact_path = store.run / "diagnostics" / "exact_rank_diagnostics.json"

    def exact_handler() -> Mapping[str, object]:
        rows = run_exact_rank_diagnostics(
            {name: built_trees[name][0] for name in exact_tree_names},
            bootstrap.manifest,
            bootstrap.prepared_root,
            backbone_factory,
            bootstrap.test_transform,
            config.lora_rank,
            config.leaf_training.batch_size,
            torch.device("cuda:0"),
        )
        publish_immutable_json(
            exact_path,
            {
                "rows": list(rows),
                "schema_version": "imagenetr50-exact-rank-diagnostics-v1",
            },
        )
        return {"events": len(rows), "path": str(exact_path)}

    scheduler.execute(exact_job, exact_handler)

    print("Phase 9/12: cached historical evaluation and addressing diagnostics.", flush=True)
    evaluations: dict[str, EvaluationResult] = {}
    report_trees: dict[str, TreeBuildResult] = {
        name: value[0] for name, value in built_trees.items()
    }
    evaluation_dependencies: dict[str, str] = {}
    for name, (tree, dependency) in built_trees.items():
        evaluations[name] = _ensure_evaluation(
            scheduler, bootstrap, tree, name, dependency, backbone_factory
        )
        evaluation_dependencies[name] = dependency
    leaf_control = leaf_bank_tree(leaves, config.lora_rank)
    leaf_job = JobSpec.create(
        store.run_hash,
        "materialize_leaf_bank_control",
        all_leaf_dependencies,
        {"leaves": 50},
    )
    scheduler.execute(leaf_job, lambda: {"leaf_hashes": [leaf.artifact.content_hash for leaf in leaves]})
    evaluations["leaf_bank_50"] = _ensure_evaluation(
        scheduler,
        bootstrap,
        leaf_control,
        "leaf_bank_50",
        leaf_job.content_hash,
        backbone_factory,
    )
    report_trees["leaf_bank_50"] = leaf_control
    evaluation_dependencies["leaf_bank_50"] = leaf_job.content_hash
    for name, execution in baselines.items():
        history = baseline_history_tree(execution, bootstrap.manifest, config.lora_rank)
        evaluations[name] = _ensure_evaluation(
            scheduler,
            bootstrap,
            history,
            name,
            baseline_job_spec(name, store, bootstrap.manifest, config).content_hash,
            backbone_factory,
        )
        report_trees[name] = history
        evaluation_dependencies[name] = baseline_job_spec(
            name, store, bootstrap.manifest, config
        ).content_hash

    zero_svd = evaluations["logt_svd_r16_repair000"]
    repaired_svd = evaluations[
        _condition_name(_policy(config, "svd", config.primary_repair_fraction))
    ]
    last = lambda result: next(
        row.accuracy
        for row in result.stages
        if row.stage == 50 and row.score_mode == "affine_calibrated"
    )
    material_help = last(repaired_svd) - last(zero_svd) >= config.repair_material_improvement_points
    one_percent_methods = (
        ("svd", "core_tsv", "output_drift") if material_help else ("svd",)
    )
    for method in one_percent_methods:
        policy = _policy(config, method, 0.01)
        tree, dependency = _ensure_tree(
            scheduler,
            bootstrap,
            leaves,
            policy,
            50,
            all_leaf_dependencies,
            backbone_factory,
            activation_provider,
        )
        name = _condition_name(policy)
        built_trees[name] = (tree, dependency)
        report_trees[name] = tree
        evaluation_dependencies[name] = dependency
        evaluations[name] = _ensure_evaluation(
            scheduler, bootstrap, tree, name, dependency, backbone_factory
        )

    print("Phase 10/12: unmodified official E2-LoRA common-split reproduction.", flush=True)
    external_job = JobSpec.create(
        store.run_hash,
        "external_e2lora",
        all_leaf_dependencies,
        {"revision": E2LORA_REVISION},
    )
    external_holder: dict[str, E2LoRAResult] = {}
    project_root = _project_root(Path(config_path))

    def external_handler() -> Mapping[str, object]:
        result = run_official_e2lora(
            project_root / "external/imagenetr50/E2-LoRA",
            bootstrap.prepared_root.parent,
            config.data_root / "model_cache",
            store.run / "baselines/e2lora",
        )
        external_holder["result"] = result
        return result.as_record()

    scheduler.execute(external_job, external_handler)
    external = external_holder.get("result") or run_official_e2lora(
        project_root / "external/imagenetr50/E2-LoRA",
        bootstrap.prepared_root.parent,
        config.data_root / "model_cache",
        store.run / "baselines/e2lora",
    )

    print("Phase 11/12: mandatory zero-leaf-step reuse proofs.", flush=True)
    one_percent_tree = built_trees["logt_svd_r16_repair001"][0]
    alternate_tree = built_trees["logt_drift_r16_repair000"][0]
    reuse_records = (
        {
            "leaf_hashes_unchanged": one_percent_tree.leaf_hashes_unchanged,
            "leaf_optimizer_steps": one_percent_tree.leaf_optimizer_steps,
            "new_gradient_work": "repair only",
            "rebuild": "svd 5% -> 1% repair",
        },
        {
            "leaf_hashes_unchanged": alternate_tree.leaf_hashes_unchanged,
            "leaf_optimizer_steps": alternate_tree.leaf_optimizer_steps,
            "new_gradient_work": "none",
            "rebuild": "svd -> output_drift",
        },
    )
    publish_immutable_json(
        store.run / "diagnostics" / "artifact_reuse.json",
        {
            "records": list(reuse_records),
            "schema_version": "imagenetr50-artifact-reuse-v1",
        },
    )

    print("Phase 12/12: ledgers, resource accounting, plots, lineage, and reports.", flush=True)
    report_job = JobSpec.create(
        store.run_hash,
        "report",
        tuple(
            sorted(
                {external_job.content_hash, exact_job.content_hash}
                | {
                    JobSpec.create(
                        store.run_hash,
                        "evaluate",
                        (evaluation_dependencies[name],),
                        {
                            "condition": name,
                            "policy_hash": report_trees[name].policy.content_hash,
                        },
                    ).content_hash
                    for name in evaluations
                }
            )
        ),
        {"conditions": sorted(evaluations)},
    )
    report_holder: dict[str, Path] = {}

    def report_handler() -> Mapping[str, object]:
        markdown, html = write_report(
            store,
            config,
            bootstrap.protocol,
            leaves,
            report_trees,
            evaluations,
            preflight,
            external,
            reuse_records,
            scheduler,
        )
        report_holder["markdown"] = markdown
        return {"html": str(html), "markdown": str(markdown)}

    # Evaluation and external jobs are complete; scheduler enforces the dependency proof.
    scheduler.execute(report_job, report_handler)
    report = report_holder.get("markdown") or store.run / "reports" / "REPORT.md"
    print(f"Complete ImageNet-R report: {report}", flush=True)
    return report


def latest_run_path(config_path: str | Path) -> tuple[ImageNetRConfig, Path]:
    """Resolve the latest run pointer without mutating experiment state."""
    config = load_config(config_path)
    latest = json.loads((config.artifact_root / "LATEST_RUN.json").read_text(encoding="utf-8"))
    return config, config.artifact_root / "runs" / latest["run_hash"]


__all__ = ["bootstrap_protocol", "latest_run_path", "run_workflow"]

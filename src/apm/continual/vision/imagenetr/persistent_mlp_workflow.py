"""Run one persistent MLP arm against immutable full-stream affine comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import torch

from apm.continual.artifacts import (
    ChainedJsonlLedger, atomic_write, canonical_json_bytes, file_sha256,
    load_canonical_json, publish_immutable_json, record_sha256, require_sha256,
)
from apm.continual.vision.imagenetr.behavior_replay_workflow import _stat_fingerprint
from apm.continual.vision.imagenetr.manifests import installed_environment_manifest
from apm.continual.vision.imagenetr.persistent_affine_workflow import (
    ARM_LEDGER_SCHEMA, InvocationWork, PersistentAffineBootstrap,
    _affine_stage_work, _material_paths, _preflight, _prepare_run,
    _run_affine_arm, _utc_now, _validated_content_record, _write_state,
    load_persistent_inputs,
)
from apm.continual.vision.imagenetr.persistent_mlp_config import (
    DEFAULT_PERSISTENT_MLP_CONFIG, PersistentMLPConfig, load_persistent_mlp_config,
)
from apm.continual.vision.imagenetr.promoted_integrator_workflow import PROMOTED_PACKAGES, _build
from apm.continual.vision.imagenetr.protocol import material_tree_manifest


@dataclass(frozen=True, slots=True)
class PersistentMLPProtocol:
    """Immutable identity for architecture, unchanged comparisons, and runtime."""

    config_hash: str
    reference_result_hash: str
    source_hierarchy_complete_sha256: str
    dataset_manifest_hash: str
    model_manifest_hash: str
    split_hash: str
    code_manifest_hash: str
    environment_manifest_hash: str
    schema_version: str = "imagenetr50-persistent-mlp-protocol-v1"

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if name != "schema_version":
                require_sha256(value, name)

    @property
    def content_hash(self) -> str:
        """Return the content-addressed run identity."""
        return record_sha256(asdict(self))

    def as_record(self) -> dict[str, object]:
        """Return the sealed protocol record."""
        return {**asdict(self), "content_hash": self.content_hash}


def bootstrap_persistent_mlp(
    config_path: str | Path = DEFAULT_PERSISTENT_MLP_CONFIG,
) -> tuple[PersistentMLPConfig, PersistentAffineBootstrap, dict[str, object]]:
    """Authenticate the completed comparison and create only the MLP namespace."""
    config = load_persistent_mlp_config(config_path)
    reference_path = config.reference_run / "evaluations/result.json"
    if (
        file_sha256(config.affine_config) != config.affine_config_sha256
        or file_sha256(reference_path) != config.reference_result_sha256
    ):
        raise ValueError("the pinned affine comparison or its recipe changed")
    reference = _validated_content_record(reference_path, "imagenetr50-persistent-affine-result-v1")
    if reference["content_hash"] != config.reference_result_hash:
        raise ValueError("the MLP comparison result identity changed")
    recipe, source, stage_matched, selection = load_persistent_inputs(config.affine_config)
    project_root = config.affine_config.parents[3]
    package = project_root / "src/apm/continual/vision/imagenetr"
    code = material_tree_manifest((
        *_material_paths(project_root, config.affine_config), Path(config_path).resolve(),
        package / "persistent_mlp_config.py", package / "persistent_mlp_workflow.py",
        project_root / "docs/imagenetr50_persistent_mlp_protocol.md",
        project_root / "scripts/vision/imagenetr/run_persistent_mlp_local.sh",
    ))
    environment = installed_environment_manifest(PROMOTED_PACKAGES)
    prior_protocol = load_canonical_json(config.reference_run / "protocol/protocol.json")
    for name in ("dataset_manifest_hash", "model_manifest_hash", "split_hash"):
        if prior_protocol[name] != getattr(source.integrator.protocol, name):
            raise ValueError(f"MLP comparison differs in {name}")
    if prior_protocol["environment_manifest_hash"] != environment["content_hash"]:
        raise ValueError("MLP environment differs from the completed affine comparison")
    protocol = PersistentMLPProtocol(
        config.config_hash, config.reference_result_hash, recipe.source_hierarchy_complete_sha256,
        source.integrator.protocol.dataset_manifest_hash, source.integrator.protocol.model_manifest_hash,
        source.integrator.protocol.split_hash, str(code["content_hash"]), str(environment["content_hash"]),
    )
    run = config.artifact_root / "runs" / protocol.content_hash
    _prepare_run(run, protocol)
    for name, record in (
        ("code_manifest.json", code), ("environment_manifest.json", environment),
        ("source_affine_result.json", reference),
    ):
        publish_immutable_json(run / "protocol" / name, record)
    publish_immutable_json(run / "config_resolved.json", {
        **config.as_record(), "training_recipe": recipe.as_record(),
        "activation_recomputation": recipe.activation_recomputation,
    })
    atomic_write(config.artifact_root / "LATEST_RUN.json", canonical_json_bytes({
        "schema_version": "imagenetr50-persistent-mlp-latest-v1", "run_hash": protocol.content_hash,
    }))
    bootstrap = PersistentAffineBootstrap(
        project_root, recipe, source, stage_matched, selection, protocol, run,
        config.hidden_dimension, tuple(reference["arms"]["4096"]),
    )
    return config, bootstrap, reference


def _register_report(config: PersistentMLPConfig, run: Path) -> Path:
    from apm.continual.vision.imagenetr.persistent_affine_reporting import write_persistent_affine_report

    result_path = run / "evaluations/result.json"
    result = _validated_content_record(result_path, "imagenetr50-persistent-mlp-result-v1")
    core = {
        "schema_version": "imagenetr50-persistent-mlp-report-extension-v1",
        "reference_result_hash": config.reference_result_hash,
        "run": str(run), "result_hash": result["content_hash"],
        "result_sha256": file_sha256(result_path),
    }
    publish_immutable_json(config.reference_run / "reports/mlp_extension.json", {
        **core, "content_hash": record_sha256(core),
    })
    return write_persistent_affine_report(config.reference_run)


def run_persistent_mlp(config_path: str | Path = DEFAULT_PERSISTENT_MLP_CONFIG) -> Path:
    """Train or exactly resume 50 MLP stages, prove reuse, and update the existing report."""
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("the persistent MLP experiment requires BF16 CUDA")
    started = time.monotonic()
    print("[phase 1/5] Authenticate the MLP recipe and immutable affine/joint comparisons", flush=True)
    config, bootstrap, reference = bootstrap_persistent_mlp(config_path)
    print(f"Temporary/resumable artifact directory: {bootstrap.run}", flush=True)
    device = torch.device("cuda:0")
    hierarchy_root = bootstrap.source.integrator.store.run / "hierarchies" / bootstrap.config.source_hierarchy_policy_hash
    before = _stat_fingerprint(hierarchy_root)
    print("[phase 2/5] Reuse all source nodes and check real MLP parity/carry/gradients", flush=True)
    hierarchy = _build(bootstrap.source, "all_train", 50, device, progress=False)
    if (
        hierarchy.policy.content_hash != bootstrap.config.source_hierarchy_policy_hash
        or hierarchy.work.leaf_optimizer_steps or hierarchy.work.parent_optimizer_steps
        or before != _stat_fingerprint(hierarchy_root)
    ):
        raise RuntimeError("MLP source hierarchy did not reuse exactly")
    _preflight(bootstrap, hierarchy, device)
    ledger = ChainedJsonlLedger(bootstrap.run / "arms/h04096/stage_metrics.jsonl", ARM_LEDGER_SCHEMA)
    completed = {int(row["stage"]) for row in ledger.rows}
    work_by_stage = {stage: _affine_stage_work(bootstrap, 4096, stage) for stage in range(1, 51)}
    from tqdm.auto import tqdm

    overall = tqdm(
        total=sum(work_by_stage.values()), initial=sum(work_by_stage[stage] for stage in completed),
        desc="overall MLP GPU work", unit="node-image", unit_scale=True,
    )
    print("[phase 3/5] Train the persistent two-layer MLP, H=4,096, across all 50 tasks", flush=True)
    rows, invocation = _run_affine_arm(
        bootstrap, hierarchy, 4096, device,
        lambda stage: overall.update(work_by_stage[stage]),
    )
    overall.close()
    if len(rows) != 50 or before != _stat_fingerprint(hierarchy_root):
        raise RuntimeError("MLP stages are incomplete or changed the source hierarchy")
    for row, prior in zip(rows, reference["arms"]["4096"], strict=True):
        if (
            row["population"] != prior["population"] or row["node_hashes"] != prior["node_hashes"]
            or row["fit"]["image_presentations"] != prior["fit"]["image_presentations"]
            or row["carry"] != prior["carry"]
        ):
            raise RuntimeError("MLP replay, LoRA lifetimes, or optimizer exposure differs from affine H=4,096")
    target = bootstrap.run / "evaluations/result.json"
    core = {
        "schema_version": "imagenetr50-persistent-mlp-result-v1",
        "protocol_hash": bootstrap.protocol.content_hash,
        "reference_result_hash": config.reference_result_hash,
        "hidden_dimension": config.hidden_dimension, "historical_capacity": 4096,
        "architecture": "linear_relu_linear", "rows": list(rows),
        "source_hierarchy_unchanged": True, "matched_replay_and_exposure": True,
        "new_leaf_optimizer_steps": 0, "new_parent_optimizer_steps": 0,
        "new_joint_optimizer_steps": 0, "invocation_work": asdict(invocation),
        "completed_at_utc": _utc_now(), "wall_seconds": time.monotonic() - started,
    }
    if not target.is_file():
        publish_immutable_json(target, {**core, "content_hash": record_sha256(core)})
    result = _validated_content_record(target, "imagenetr50-persistent-mlp-result-v1")
    if result["protocol_hash"] != bootstrap.protocol.content_hash or result["rows"] != list(rows):
        raise ValueError("stored MLP result differs from its authenticated stage ledger")
    print("[phase 4/5] Demonstrate zero-step reuse of all 50 MLP stages", flush=True)
    reused, work = _run_affine_arm(bootstrap, hierarchy, 4096, device, show_progress=False)
    if work != InvocationWork() or reused != rows or before != _stat_fingerprint(hierarchy_root):
        raise RuntimeError("completed MLP condition did not reuse exactly")
    proof = {
        "schema_version": "imagenetr50-persistent-mlp-reuse-v1",
        "protocol_hash": bootstrap.protocol.content_hash, "result_sha256": file_sha256(target),
        "zero_new_optimizer_work": asdict(work), "stage_rows_unchanged": True,
        "source_hierarchy_unchanged": True,
    }
    publish_immutable_json(bootstrap.run / "evaluations/reuse_proof.json", {**proof, "content_hash": record_sha256(proof)})
    print("[phase 5/5] Integrate the MLP condition into the existing report and figures", flush=True)
    report = _register_report(config, bootstrap.run)
    _write_state(bootstrap, "complete", elapsed_seconds=time.monotonic() - started, report=str(report), result=str(target))
    print(f"Result: {target}\nReport: {report}", flush=True)
    return report


def _main() -> None:
    import sys

    command = sys.argv[1] if len(sys.argv) > 1 else "run"
    if command == "run":
        run_persistent_mlp()
        return
    config = load_persistent_mlp_config()
    latest = load_canonical_json(config.artifact_root / "LATEST_RUN.json")
    run = config.artifact_root / "runs" / latest["run_hash"]
    if command == "report":
        print(_register_report(config, run))
    elif command == "status":
        state_path = run / "state/workflow.json"
        rows = ChainedJsonlLedger(run / "arms/h04096/stage_metrics.jsonl", ARM_LEDGER_SCHEMA).rows
        print(json.dumps({
            "run": str(run), "stages_completed": len(rows),
            "result_ready": (run / "evaluations/result.json").is_file(),
            "state": load_canonical_json(state_path) if state_path.is_file() else None,
        }, indent=2))
    else:
        raise SystemExit("usage: persistent_mlp_workflow [run|status|report]")


if __name__ == "__main__":
    _main()

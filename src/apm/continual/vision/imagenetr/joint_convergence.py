"""Run the task-50 rank-16 joint-IID validation-convergence study locally."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timezone
import fcntl
from functools import partial
import gc
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from apm.continual.artifacts import atomic_write, canonical_json_bytes, file_sha256, publish_immutable_json, record_sha256
from apm.continual.vision.imagenetr.constants import TIMM_MODEL_SHA256
from apm.continual.vision.imagenetr.data import DatasetManifest, ImageRecord, load_dataset_manifest, validate_prepared_dataset
from apm.continual.vision.imagenetr.joint_convergence_training import ConvergenceRule, JointPopulation, select_epochs, train_joint_job, validate_joint_job
from apm.continual.vision.imagenetr.manifests import installed_environment_manifest
from apm.continual.vision.imagenetr.protocol import material_tree_manifest
from apm.continual.vision.imagenetr.router_artifacts import build_router_split
from apm.continual.vision.imagenetr.srt_config import SRTConfig, load_srt_config
from apm.continual.vision.imagenetr.srt_evidence import read_sealed, sealed_record, write_parquet

if TYPE_CHECKING:
    import torch
    from apm.continual.vision.imagenetr.model import AdapterVisionModel

DEFAULT_CONFIG = Path("configs/vision/imagenetr/joint_convergence_r16.yaml")


@dataclass(frozen=True, slots=True)
class JointConvergenceConfig:
    """The single predeclared experiment, with no command-line science overrides."""

    artifact_root: str
    source_run: str
    source_result_hash: str
    development_seed: int
    refit_seeds: tuple[int, ...]
    fit_probe_images: int
    rule: ConvergenceRule


@dataclass(frozen=True, slots=True)
class JointInputs:
    """Authenticated experiment sources and explicit training-derived populations."""

    project: Path
    run: Path
    source: Path
    checkpoint: Path
    config: JointConvergenceConfig
    optimizer: SRTConfig
    manifest: DatasetManifest
    development: JointPopulation
    full: JointPopulation
    protocol: dict[str, object]


def load_config(path: Path = DEFAULT_CONFIG) -> JointConvergenceConfig:
    """Read the complete convergence protocol and validate its fixed task scope."""
    values = yaml.safe_load(path.read_text())
    rule_values = {field.name: values.pop(field.name) for field in fields(ConvergenceRule)}
    values["refit_seeds"] = tuple(values["refit_seeds"])
    config = JointConvergenceConfig(**values, rule=ConvergenceRule(**rule_values))
    if config.development_seed != 1993 or config.refit_seeds != (1993, 1994, 1995) or config.fit_probe_images != 2048:
        raise ValueError("joint convergence differs from the predeclared seed/probe matrix")
    return config


def bootstrap_joint(config_path: Path = DEFAULT_CONFIG) -> JointInputs:
    """Freeze source/data/model/software identities before requesting any image batch."""
    project = config_path.resolve().parents[3]
    config = load_config(config_path)
    source = project / config.source_run
    original = read_sealed(source / "result.json", "imagenetr50-srt-result-v1")
    if original["content_hash"] != config.source_result_hash:
        raise ValueError("joint study source result changed")
    optimizer = load_srt_config(project / "configs/vision/imagenetr/srt_r16_v1.yaml")
    source_protocol = read_sealed(source / "protocol.json")
    if optimizer.content_hash != source_protocol["config_hash"]:
        raise ValueError("the source optimizer configuration changed")
    old_code = read_sealed(source / "code_manifest.json")
    if any(file_sha256(project / row["path"]) != row["sha256"] for row in old_code["files"]):
        raise ValueError("the authenticated source training code changed")
    manifest = load_dataset_manifest(project / "data/imagenetr50/imagenet-r/dataset_manifest.json")
    split = build_router_split(manifest, .8, 1993)
    if manifest.content_hash != optimizer.dataset_hash or split.content_hash != optimizer.split_hash:
        raise ValueError("the frozen dataset or development split changed")
    fit = manifest.select("train", image_ids=split.fit_image_ids)
    validation = manifest.select("train", image_ids=split.validation_image_ids)
    full = manifest.select("train")
    probe = lambda rows: tuple(sorted(rows, key=lambda row: record_sha256(["joint-fit-probe-v1", row.image_id]))[:config.fit_probe_images])
    development, final = JointPopulation(fit, validation, probe(fit)), JointPopulation(full, (), probe(full))
    if (len(fit), len(validation), len(full)) != (19200, 4800, 24000):
        raise ValueError("joint study population counts changed")
    environment = installed_environment_manifest(tuple(row["name"] for row in read_sealed(source / "environment_manifest.json")["packages"]))
    if environment["content_hash"] != source_protocol["environment_hash"]:
        raise ValueError("joint study environment differs from its source")
    package = Path(__file__).parent
    material = material_tree_manifest(tuple(package / name for name in ("joint_convergence.py", "joint_convergence_training.py", "training.py"))
                                      + (config_path.resolve(), project / "docs/imagenetr50_joint_convergence_protocol.md"))
    protocol = sealed_record({
        "schema_version": "imagenetr50-joint-convergence-protocol-v1", "config": asdict(config),
        "source_result_hash": original["content_hash"], "source_protocol_hash": source_protocol["content_hash"],
        "source_code_hash": old_code["content_hash"], "code_hash": material["content_hash"],
        "environment_hash": environment["content_hash"], "dataset_hash": manifest.content_hash,
        "development_split_hash": split.content_hash, "model_sha256": TIMM_MODEL_SHA256,
        "model": "ViT-B/16; frozen base; rank/alpha 16 QKV+fc1 LoRAs; 200-way affine head",
        "primary_selection": "maximum validation accuracy, then minimum NLL, then earliest epoch",
        "final_refits": "three cold full-data seeds; replay frozen development schedule by epochs",
        "test_policy": "only predefined endpoints; after all full-data fits; no test-based selection",
    })
    run = project / config.artifact_root / "runs" / protocol["content_hash"]
    for name, record in (("protocol.json", protocol), ("code_manifest.json", material), ("environment_manifest.json", environment),
                         ("development_split.json", split.as_record())):
        publish_immutable_json(run / name, record)
    populations = sealed_record({"schema_version": "imagenetr50-joint-convergence-populations-v1",
                                 "fit": [row.image_id for row in fit], "validation": [row.image_id for row in validation],
                                 "full": [row.image_id for row in full], "development_probe": [row.image_id for row in development.probe],
                                 "full_probe": [row.image_id for row in final.probe], "test": [row.image_id for row in manifest.select("test")]})
    publish_immutable_json(run / "populations.json", populations)
    candidates = tuple((project / "data/imagenetr50/model_cache").rglob(TIMM_MODEL_SHA256))
    if len(candidates) != 1 or file_sha256(candidates[0]) != TIMM_MODEL_SHA256:
        raise ValueError("pinned backbone failed authentication")
    atomic_write(project / config.artifact_root / "LATEST_RUN.json", canonical_json_bytes({"run_hash": run.name}))
    return JointInputs(project, run, source, candidates[0], config, optimizer, manifest, development, final, protocol)


def _model(checkpoint: Path, seed: int) -> AdapterVisionModel:
    from apm.continual.vision.imagenetr.model import AdapterVisionModel, create_pinned_backbone, require_trainable_boundary
    model = AdapterVisionModel(create_pinned_backbone(checkpoint), tuple(range(200)), initialization_seed=seed)
    require_trainable_boundary(model)
    return model


def _progress(inputs: JointInputs, phase: str, current: dict[str, object] | None = None) -> None:
    current = current or {}
    history = tuple(current.get("history", ()))
    mean_epoch_seconds = math.fsum(row["training"]["wall_seconds"] + row["evaluation_wall_seconds"] for row in history) / len(history) if history else None
    remaining_epochs = int(current.get("epoch_limit", 0)) - int(current.get("epoch", 0)) + 1
    planned = inputs.config.rule.maximum_epochs * (19200 + 3 * 24000)
    if (inputs.run / "selection.json").is_file():
        selection = read_sealed(inputs.run / "selection.json")
        planned = len(selection["learning_rate_schedule"]) * (19200 + 3 * 24000)
    completed_roots = tuple(path for path in (inputs.run / "refits").glob("*/result.json") if path.parent.name != current.get("job"))
    completed = sum(read_sealed(path)["training_presentations"] for path in completed_roots) + int(current.get("presentations", 0))
    if phase != "development" and (inputs.run / "development/result.json").is_file():
        completed += read_sealed(inputs.run / "development/result.json")["training_presentations"]
    state = {"phase": phase, "updated_utc": datetime.now(timezone.utc).isoformat(), "run_hash": inputs.run.name,
             "completed_training_presentations": completed, "planned_training_presentations": planned,
             "phase_eta_seconds": remaining_epochs * mean_epoch_seconds if mean_epoch_seconds else None,
             "eta_is_upper_bound": phase == "development", **{name: value for name, value in current.items() if name != "history"}}
    if mean_epoch_seconds:
        state["overall_eta_seconds"] = max(0, planned - completed) * mean_epoch_seconds / (19200 if phase == "development" else 24000)
    atomic_write(inputs.run / "status.json", canonical_json_bytes(state))
    if not current or current.get("batch") == current.get("batches"):
        eta = state.get("overall_eta_seconds", state["phase_eta_seconds"])
        print(f"Phase {phase}: {completed:,} training presentations; "
              f"{'upper-bound ' if state['eta_is_upper_bound'] else ''}ETA {eta / 60:.1f} min" if eta else f"Phase {phase}: measuring throughput", flush=True)


def real_preflight(inputs: JointInputs, device: torch.device) -> dict[str, object]:
    """Check real BF16 training, population isolation, and exact interrupted resume."""
    import torch
    from safetensors.torch import load_file
    from apm.continual.vision.imagenetr.srt_data import SelectedImageLoader
    root = inputs.run / "preflight"
    if (root / "result.json").is_file():
        return read_sealed(root / "result.json")
    validate_prepared_dataset(inputs.project / "data/imagenetr50/imagenet-r", inputs.manifest)
    population = JointPopulation(inputs.development.fitting[:128], (), inputs.development.fitting[:64])
    optimizer = replace(inputs.optimizer, checkpoint_steps=1)
    with SelectedImageLoader(inputs.project / "data/imagenetr50/imagenet-r", 51993, inputs.optimizer.num_workers) as loader:
        arguments = dict(protocol_hash=inputs.run.name, seed=1993, population=population, optimizer_config=optimizer,
                         rule=inputs.config.rule, loader=loader, device=device, model_factory=partial(_model, inputs.checkpoint), fixed_schedule=(1.0, .2))
        uninterrupted = train_joint_job(root / "uninterrupted", **arguments)
        def interrupt(row: dict[str, object]) -> None:
            if row["steps"] == 1:
                raise InterruptedError("intentional preflight resume check")
        try:
            train_joint_job(root / "resumed", progress_callback=interrupt, **arguments)
        except InterruptedError:
            pass
        gc.collect()
        torch.cuda.empty_cache()
        resumed = train_joint_job(root / "resumed", **arguments)
        for epoch in (1, 2):
            first, second = (load_file(str(root / name / "epochs" / f"{epoch:03d}" / "model.safetensors")) for name in ("uninterrupted", "resumed"))
            if set(first) != set(second) or any(not torch.equal(first[name], second[name]) for name in first):
                raise ValueError("joint real-model interrupted resume is not exact")
        reuse = train_joint_job(root / "resumed", **arguments)
        if reuse["invocation_optimizer_steps"] != 0 or reuse["content_hash"] != resumed["content_hash"]:
            raise ValueError("joint preflight failed zero-step reuse")
    result = sealed_record({"schema_version": "imagenetr50-joint-convergence-preflight-v1", "dataset_images_authenticated": 30000,
                            "split_isolation": True, "interrupted_resume_exact": True, "zero_step_reuse": True,
                            "uninterrupted_hash": uninterrupted["content_hash"], "resumed_hash": resumed["content_hash"],
                            "peak_vram_bytes": max(row["peak_vram_bytes"] for row in resumed["epochs"])})
    publish_immutable_json(root / "result.json", result)
    return result


def evaluate_endpoints(inputs: JointInputs, selection: dict[str, object], device: torch.device) -> tuple[dict[str, object], ...]:
    """Evaluate only sealed, preselected task-50 checkpoints after every refit ends."""
    from safetensors.torch import load_file
    from apm.continual.vision.imagenetr.srt_data import SelectedImageLoader
    from apm.continual.vision.imagenetr.srt_training import evaluate_model, restore_trainables
    roots = tuple(inputs.run / "refits" / f"seed_{seed}" for seed in inputs.config.refit_seeds)
    if any(not (root / "result.json").is_file() for root in roots):
        raise ValueError("test evaluation requires every refit to be complete")
    results = []
    for seed, root in zip(inputs.config.refit_seeds, roots, strict=True):
        model = _model(inputs.checkpoint, seed).to(device)
        with SelectedImageLoader(inputs.project / "data/imagenetr50/imagenet-r", seed + 50_000, inputs.optimizer.num_workers) as loader:
            for epoch in sorted(set(selection["endpoints"].values())):
                checkpoint = root / "epochs" / f"{epoch:03d}"
                output = inputs.run / "evaluations" / f"seed_{seed}" / f"epoch_{epoch:03d}"
                if (output / "result.json").is_file():
                    record = read_sealed(output / "result.json")
                    if (record["selection_hash"] != selection["content_hash"] or record["model_sha256"] != file_sha256(checkpoint / "model.safetensors")
                            or record["predictions_sha256"] != file_sha256(output / "predictions.parquet")):
                        raise ValueError("joint endpoint identity changed")
                else:
                    restore_trainables(model, load_file(str(checkpoint / "model.safetensors")))
                    metrics, predictions = evaluate_model(model, inputs.manifest.select("test"), loader, device, inputs.optimizer.batch_size, 50)
                    prediction_hash = write_parquet(output / "predictions.parquet", predictions)
                    record = sealed_record({"schema_version": "imagenetr50-joint-convergence-evaluation-v1", "seed": seed, "epoch": epoch,
                                            "selection_hash": selection["content_hash"], "model_sha256": file_sha256(checkpoint / "model.safetensors"),
                                            "predictions_sha256": prediction_hash, "metrics": metrics,
                                            "endpoint_roles": [name for name, value in selection["endpoints"].items() if value == epoch]})
                    publish_immutable_json(output / "result.json", record)
                results.append(record)
                print(f"Test seed {seed}, epoch {epoch}: {record['metrics']['accuracy']:.3f}% / NLL {record['metrics']['nll']:.5f}; "
                      f"roles {record['endpoint_roles']}", flush=True)
        del model
        gc.collect()
    return tuple(results)


def run_joint_convergence() -> Path:
    """Complete the preflight, development, three full refits, audit, and current report."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from apm.continual.vision.imagenetr.srt_data import SelectedImageLoader
    torch.set_num_threads(4)
    torch.manual_seed(1993)
    torch.use_deterministic_algorithms(True)
    inputs = bootstrap_joint()
    print(f"Joint-convergence working artifacts: {inputs.run}", flush=True)
    lock = (inputs.source / "runner.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("run the joint study outside the sandbox on the local BF16 GPU")
        device = torch.device("cuda:0")
        _progress(inputs, "preflight")
        preflight = real_preflight(inputs, device)
        common = dict(protocol_hash=inputs.run.name, optimizer_config=inputs.optimizer, rule=inputs.config.rule,
                      device=device, model_factory=partial(_model, inputs.checkpoint))
        with SelectedImageLoader(inputs.project / "data/imagenetr50/imagenet-r", 51993, inputs.optimizer.num_workers) as loader:
            _progress(inputs, "development")
            train_joint_job(inputs.run / "development", seed=1993, population=inputs.development, loader=loader,
                            progress_callback=partial(_progress, inputs, "development"), **common)
        development = validate_joint_job(inputs.run / "development")
        selection = sealed_record({"schema_version": "imagenetr50-joint-convergence-selection-v1", "protocol_hash": inputs.run.name,
                                   "development_result_hash": development["content_hash"], "endpoints": select_epochs(tuple(development["epochs"])),
                                   "learning_rate_schedule": [row["learning_rate_scale"] for row in development["epochs"]],
                                   "stop_reason": development["stop_reason"], "validation_converged": development["validation_converged"], "test_used": False})
        publish_immutable_json(inputs.run / "selection.json", selection)
        print(f"Frozen schedule: {len(selection['learning_rate_schedule'])} epochs; endpoints {selection['endpoints']}; {selection['stop_reason']}", flush=True)
        jobs = {"development": development}
        for seed in inputs.config.refit_seeds:
            phase = f"full-data seed {seed}"
            _progress(inputs, phase)
            with SelectedImageLoader(inputs.project / "data/imagenetr50/imagenet-r", seed + 50_000, inputs.optimizer.num_workers) as loader:
                root = inputs.run / "refits" / f"seed_{seed}"
                train_joint_job(root, seed=seed, population=inputs.full, loader=loader, fixed_schedule=tuple(selection["learning_rate_schedule"]),
                                progress_callback=partial(_progress, inputs, phase), **common)
                jobs[f"seed_{seed}"] = validate_joint_job(root)
            gc.collect()
            torch.cuda.empty_cache()
        _progress(inputs, "evaluate frozen task-50 endpoints")
        evaluations = evaluate_endpoints(inputs, selection, device)
        reuse = {}
        for name, population in (("development", inputs.development), *((f"seed_{seed}", inputs.full) for seed in inputs.config.refit_seeds)):
            seed = 1993 if name == "development" else int(name.removeprefix("seed_"))
            root = inputs.run / name if name == "development" else inputs.run / "refits" / name
            with SelectedImageLoader(inputs.project / "data/imagenetr50/imagenet-r", seed + 50_000, 0) as loader:
                repeated = train_joint_job(root, seed=seed, population=population, loader=loader,
                                           fixed_schedule=None if name == "development" else tuple(selection["learning_rate_schedule"]), **common)
            reuse[name] = {"optimizer_steps": repeated["invocation_optimizer_steps"], "result_hash": repeated["content_hash"]}
        if any(row["optimizer_steps"] or row["result_hash"] != jobs[name]["content_hash"] for name, row in reuse.items()):
            raise ValueError("joint workflow failed zero-step reuse")
        if read_sealed(inputs.source / "result.json")["content_hash"] != inputs.config.source_result_hash:
            raise ValueError("source SRT results changed")
        result = sealed_record({"schema_version": "imagenetr50-joint-convergence-result-v1", "protocol_hash": inputs.run.name,
                                "source_result_hash": inputs.config.source_result_hash, "selection": selection,
                                "preflight_hash": preflight["content_hash"], "jobs": jobs, "evaluations": evaluations,
                                "reuse": reuse, "zero_step_reuse": True, "source_unchanged": True})
        publish_immutable_json(inputs.run / "result.json", result)
        pointer = sealed_record({"schema_version": "imagenetr50-joint-convergence-pointer-v1", "run": str(inputs.run.relative_to(inputs.project)),
                                 "result_hash": result["content_hash"], "source_result_hash": inputs.config.source_result_hash})
        publish_immutable_json(inputs.source / "reports/joint_convergence.json", pointer)
        _progress(inputs, "report")
        from apm.continual.vision.imagenetr.srt_reporting import write_srt_report
        report = write_srt_report(inputs.source)
        _progress(inputs, "complete")
        return report
    except BaseException as error:
        atomic_write(inputs.run / "failure.json", canonical_json_bytes({"exception": type(error).__name__, "message": str(error),
                                                                        "utc": datetime.now(timezone.utc).isoformat(), "resumable": True}))
        raise
    finally:
        lock.close()


def main() -> None:
    """Use one default workflow with read-only status and a report-only rebuild."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("run", "status", "report"), default="run")
    args = parser.parse_args()
    if args.command == "run":
        print(run_joint_convergence())
    elif args.command == "status":
        import json
        root = Path(load_config().artifact_root)
        latest = json.loads((root / "LATEST_RUN.json").read_text())
        print((root / "runs" / latest["run_hash"] / "status.json").read_text())
    else:
        from apm.continual.vision.imagenetr.srt_reporting import write_srt_report
        print(write_srt_report(Path(load_config().source_run)))


if __name__ == "__main__":
    main()

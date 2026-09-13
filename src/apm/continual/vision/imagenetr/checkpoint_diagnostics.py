"""Run fixed-checkpoint probability and replay-retention diagnostics, never training."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
from functools import partial
import gc
import os
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from safetensors.torch import load_file
import yaml

from apm.continual.artifacts import atomic_write, canonical_json_bytes, file_sha256, publish_immutable_json, record_sha256
from apm.continual.vision.imagenetr.checkpoint_diagnostic_inference import CollectionSpec, collect_logits, load_collection
from apm.continual.vision.imagenetr.checkpoint_diagnostic_math import cross_fit, fixed_folds, paired_training_cohorts, reliability_rows, summarize_scores, verify_saved_predictions
from apm.continual.vision.imagenetr.constants import TIMM_MODEL_SHA256
from apm.continual.vision.imagenetr.data import DatasetManifest, load_dataset_manifest, validate_prepared_dataset
from apm.continual.vision.imagenetr.joint_convergence import _model
from apm.continual.vision.imagenetr.manifests import installed_environment_manifest
from apm.continual.vision.imagenetr.protocol import material_tree_manifest
from apm.continual.vision.imagenetr.srt_data import SelectedImageLoader
from apm.continual.vision.imagenetr.srt_evidence import read_sealed, sealed_record, write_parquet
from apm.continual.vision.imagenetr.srt_training import restore_trainables


DEFAULT_CONFIG = Path("configs/vision/imagenetr/checkpoint_diagnostics.yaml")


@dataclass(frozen=True, slots=True)
class DiagnosticConfig:
    """The fixed diagnostic matrix and numerical settings, not CLI science flags."""

    artifact_root: str
    source_run: str
    source_result_hash: str
    offline_run: str
    offline_result_hash: str
    followup_hash: str
    followup_result_hash: str
    history_sha256: str
    folds: int
    temperature_bounds: tuple[float, float]
    batch_size: int
    chunk_images: int
    loader_workers: int


@dataclass(frozen=True, slots=True)
class Endpoint:
    """An exact task-50 checkpoint and its previously sealed test predictions."""

    name: str
    seed: int
    model: str
    model_sha256: str
    predictions: str
    predictions_sha256: str
    result: str
    result_hash: str
    collect_training: bool


@dataclass(frozen=True, slots=True)
class DiagnosticInputs:
    """Resolved, authenticated input populations and their content-addressed run."""

    project: Path
    run: Path
    source: Path
    backbone: Path
    config: DiagnosticConfig
    manifest: DatasetManifest
    endpoints: tuple[Endpoint, ...]
    protocol: dict[str, object]
    source_snapshot: dict[str, object]


def load_config(path: Path = DEFAULT_CONFIG) -> DiagnosticConfig:
    """Read the one resolved diagnostic configuration and validate its scope."""
    values = yaml.safe_load(path.read_text())
    values["temperature_bounds"] = tuple(values["temperature_bounds"])
    config = DiagnosticConfig(**values)
    if config.folds != 5 or config.batch_size != 64 or config.chunk_images % 64 or config.loader_workers < 0:
        raise ValueError("diagnostic fold/batch protocol differs")
    return config


def source_endpoints(project: Path, config: DiagnosticConfig) -> tuple[Endpoint, ...]:
    """Resolve only the declared offline seeds and the two matched replay pairs."""
    offline = project / config.offline_run
    result = read_sealed(offline / "result.json")
    if result["content_hash"] != config.offline_result_hash or not result["zero_step_reuse"]:
        raise ValueError("offline source result changed")
    endpoints = ()
    for evaluation in result["evaluations"]:
        seed = evaluation["seed"]
        model = offline / f"seeds/seed_{seed}/blocks/050/model.safetensors"
        prediction = offline / f"evaluations/seed_{seed}/predictions.parquet"
        record = prediction.with_name("result.json")
        if read_sealed(record) != evaluation:
            raise ValueError("offline endpoint differs from sealed result")
        endpoints += (Endpoint(f"offline_seed_{seed}", seed, str(model.relative_to(project)), evaluation["model_sha256"],
                               str(prediction.relative_to(project)), evaluation["predictions_sha256"],
                               str(record.relative_to(project)), evaluation["content_hash"], False),)
    if {row.seed for row in endpoints} != {1993, 1994, 1995} or len(endpoints) != 3:
        raise ValueError("all three offline seeds are required")
    followup = project / config.source_run / "followups" / config.followup_hash
    result = read_sealed(followup / "result.json")
    if result["content_hash"] != config.followup_result_hash or result["source_result_hash"] != config.source_result_hash:
        raise ValueError("replay follow-up result changed")
    for profile in ("standard", "strict"):
        for method in ("srt", "uniform"):
            name = f"{method}_h4096_{profile}_rho80_unit8"
            root = followup / "final" / name
            stage = read_sealed(root / "stages/050/result.json")
            if read_sealed(root / "result.json") != result["conditions"][name] or stage != result["conditions"][name]["rows"][-1]:
                raise ValueError("replay endpoint differs from sealed result")
            endpoints += (Endpoint(name, 1993, str((root / "stages/050/model.safetensors").relative_to(project)), stage["model_sha256"],
                                   str((root / "stages/050/predictions.parquet").relative_to(project)), stage["predictions_sha256"],
                                   str((root / "stages/050/result.json").relative_to(project)), stage["content_hash"], True),)
    for endpoint in endpoints:
        if file_sha256(project / endpoint.model) != endpoint.model_sha256 or file_sha256(project / endpoint.predictions) != endpoint.predictions_sha256:
            raise ValueError("source checkpoint or predictions failed authentication")
    return endpoints


def snapshot_sources(project: Path, paths: tuple[Path, ...]) -> dict[str, object]:
    """Record exact bytes and modification times of immutable scientific sources."""
    return {str(path.relative_to(project)): {"sha256": file_sha256(path), "bytes": path.stat().st_size,
                                            "mtime_ns": path.stat().st_mtime_ns} for path in sorted(set(paths))}


def bootstrap(config_path: Path = DEFAULT_CONFIG) -> DiagnosticInputs:
    """Freeze endpoint, data, fold, code, history, and environment identities before inference."""
    config, project = load_config(config_path), config_path.resolve().parents[3]
    source = project / config.source_run
    result, source_protocol = read_sealed(source / "result.json"), read_sealed(source / "protocol.json")
    if result["content_hash"] != config.source_result_hash:
        raise ValueError("original SRT result changed")
    original_code = read_sealed(source / "code_manifest.json")
    if any(file_sha256(project / row["path"]) != row["sha256"] for row in original_code["files"]):
        raise ValueError("frozen source training code changed")
    original_environment = read_sealed(source / "environment_manifest.json")
    packages = tuple(row["name"] for row in original_environment["packages"])
    if installed_environment_manifest(packages)["content_hash"] != original_environment["content_hash"]:
        raise ValueError("checkpoint inference environment differs from source")
    environment = installed_environment_manifest((*packages, "scipy", "pandas", "matplotlib", "reportlab"))
    manifest = load_dataset_manifest(project / "data/imagenetr50/imagenet-r/dataset_manifest.json")
    if manifest.content_hash != source_protocol["dataset_hash"]:
        raise ValueError("diagnostic dataset differs from source")
    endpoints = source_endpoints(project, config)
    history_path = source / "reports/sample_replay.parquet"
    if file_sha256(history_path) != config.history_sha256:
        raise ValueError("replay history summary changed")
    histories = tuple(row for row in pq.read_table(history_path).to_pylist() if row["condition"] in {item.name for item in endpoints})
    test, training = manifest.select("test"), manifest.select("train")
    if (len(test), len(training)) != (6000, 24000) or {row.image_id for row in test} & {row.image_id for row in training}:
        raise ValueError("diagnostic populations overlap or differ in size")
    if len(histories) != 96000 or any({row["image_id"] for row in histories if row["condition"] == item.name} != {row.image_id for row in training}
                                    for item in endpoints if item.collect_training):
        raise ValueError("replay histories lack the common training population")
    for endpoint in endpoints:
        predictions = pq.read_table(project / endpoint.predictions).to_pylist()
        if [(row["image_id"], row["label"]) for row in predictions] != [(row.image_id, row.remapped_class_index) for row in test]:
            raise ValueError("original test ordering or labels changed")
    folds = fixed_folds(tuple(row.image_id for row in test), tuple(row.remapped_class_index for row in test), config.folds)
    if tuple(folds.count(index) for index in range(5)) != (1200,) * 5:
        raise ValueError("diagnostic folds must contain 1,200 images each")
    fold_rows = tuple({"image_id": row.image_id, "label": row.remapped_class_index, "fold": fold} for row, fold in zip(test, folds, strict=True))
    package = Path(__file__).parent
    code = material_tree_manifest(tuple(package / name for name in ("checkpoint_diagnostics.py", "checkpoint_diagnostic_math.py",
                                    "checkpoint_diagnostic_inference.py")) + (config_path.resolve(), project / "docs/imagenetr50_checkpoint_diagnostics_protocol.md"))
    protocol = sealed_record({"schema_version": "imagenetr50-checkpoint-diagnostics-protocol-v1", "config": asdict(config),
                              "dataset_hash": manifest.content_hash, "model_sha256": TIMM_MODEL_SHA256,
                              "source_code_hash": original_code["content_hash"], "code_hash": code["content_hash"],
                              "environment_hash": environment["content_hash"], "endpoints": tuple(asdict(row) for row in endpoints),
                              "folds_hash": record_sha256(fold_rows), "planned_model_forwards": 138000,
                              "selection": "post-hoc cross-fitted diagnostic only; no model training or selection"})
    run = project / config.artifact_root / "runs" / protocol["content_hash"]
    for name, record in (("protocol.json", protocol), ("code_manifest.json", code), ("environment_manifest.json", environment)):
        publish_immutable_json(run / name, record)
    write_parquet(run / "folds.parquet", fold_rows)
    write_parquet(run / "replay_history.parquet", histories)
    candidates = tuple((project / "data/imagenetr50/model_cache").rglob(TIMM_MODEL_SHA256))
    if len(candidates) != 1 or file_sha256(candidates[0]) != TIMM_MODEL_SHA256:
        raise ValueError("base checkpoint failed authentication")
    paths = tuple(project / name for endpoint in endpoints for name in (endpoint.model, endpoint.predictions, endpoint.result))
    paths += (source / "result.json", source / "protocol.json", project / config.offline_run / "result.json",
              source / "followups" / config.followup_hash / "result.json", candidates[0])
    snapshot = sealed_record({"schema_version": "imagenetr50-checkpoint-source-snapshot-v1", "files": snapshot_sources(project, paths)})
    publish_immutable_json(run / "source_snapshot.json", snapshot)
    atomic_write(project / config.artifact_root / "LATEST_RUN.json", canonical_json_bytes({"run_hash": run.name}))
    return DiagnosticInputs(project, run, source, candidates[0], config, manifest, endpoints, protocol, snapshot)


def restored_model(inputs: DiagnosticInputs, endpoint: Endpoint) -> torch.nn.Module:
    """Load exactly one authenticated endpoint without constructing an optimizer."""
    model = _model(inputs.backbone, endpoint.seed)
    restore_trainables(model, load_file(str(inputs.project / endpoint.model)))
    return model


def analyze(inputs: DiagnosticInputs) -> dict[str, object]:
    """Compute cross-fitted test probabilities and paired clean-training cohorts from caches."""
    if (inputs.run / "result.json").is_file():
        result = read_sealed(inputs.run / "result.json")
        if any(file_sha256(inputs.run / "analysis" / f"{name}.parquet") != digest for name, digest in result["tables"].items()):
            raise ValueError("cached analysis table changed")
        return result
    folds = pq.read_table(inputs.run / "folds.parquet").to_pylist()
    summaries, fits, predictions, reliability, parity, resources = (), (), (), (), (), ()
    training_by_condition = {}
    for endpoint in inputs.endpoints:
        root = inputs.run / "collections" / endpoint.name
        rows, logits, collected = load_collection(root / "test")
        original = pq.read_table(inputs.project / endpoint.predictions).to_pylist()
        if any(row["image_id"] != fold["image_id"] for row, fold in zip(rows, folds, strict=True)):
            raise ValueError("independent logits disagree with fixed fold identities")
        parity += ({"condition": endpoint.name, **verify_saved_predictions(rows, original)},)
        print(f"Phase cross-fitted probabilities: {endpoint.name}", flush=True)
        calibrated = cross_fit(logits, np.array([row["label"] for row in rows]), tuple(row["image_id"] for row in rows),
                               tuple(row["fold"] for row in folds), inputs.config.temperature_bounds)
        fits += tuple({"condition": endpoint.name, **row} for row in calibrated.fits)
        predictions += tuple({"condition": endpoint.name, "task": original[index]["task"], **row} for index, row in enumerate(calibrated.rows))
        for prefix in ("raw", "calibrated"):
            scored = tuple({key[len(prefix) + 1:]: value for key, value in row.items() if key.startswith(prefix + "_")} for row in calibrated.rows)
            bins = reliability_rows(calibrated.rows, prefix)
            reliability += tuple({"condition": endpoint.name, "score_mode": prefix, **row} for row in bins)
            ece = sum(row["examples"] * abs(row["accuracy"] - row["confidence"]) for row in bins if row["examples"]) / len(rows)
            summaries += ({"condition": endpoint.name, "seed": endpoint.seed, "score_mode": prefix, "ece_15": ece,
                           "temperature_min": min(row["temperature"] for row in calibrated.fits),
                           "temperature_max": max(row["temperature"] for row in calibrated.fits), **summarize_scores(scored)},)
        resources += ({"condition": endpoint.name, "split": "test", "model_forward_images": collected["model_forward_images"],
                       "optimizer_steps": 0, "wall_seconds": collected["wall_seconds"]},)
        if endpoint.collect_training:
            training, _, collection = load_collection(root / "train")
            training_by_condition[endpoint.name] = training
            resources += ({"condition": endpoint.name, "split": "train", "model_forward_images": collection["model_forward_images"],
                           "optimizer_steps": 0, "wall_seconds": collection["wall_seconds"]},)
    histories = pq.read_table(inputs.run / "replay_history.parquet").to_pylist()
    cohorts, paired_images = (), ()
    for profile in ("standard", "strict"):
        names = tuple(f"{method}_h4096_{profile}_rho80_unit8" for method in ("srt", "uniform"))
        history = tuple(row for row in histories if row["condition"] == names[0])
        cohorts += tuple({"profile": profile, **row} for row in paired_training_cohorts(history, *(training_by_condition[name] for name in names)))
        lookup = {name: {row["image_id"]: row for row in training_by_condition[name]} for name in names}
        paired_images += tuple({"profile": profile, "image_id": item["image_id"], "label": lookup[names[0]][item["image_id"]]["label"],
                                "arrival_stage": item["arrival_stage"],
                                "srt_last_stage": item["last_stage"], "srt_last_augmented_probability": item["last_confidence"],
                                "srt_presentations": item["presentations"], "srt_last_quality": item["last_quality"],
                                **{f"{method}_{key}": value for method, name in zip(("srt", "uniform"), names, strict=True)
                                   for key, value in lookup[name][item["image_id"]].items() if key not in {"image_id", "task", "label"}}} for item in history)
    tables = {"checkpoint_probability_summary": summaries, "checkpoint_temperature_fits": fits,
              "checkpoint_test_predictions": predictions, "checkpoint_reliability": reliability,
              "checkpoint_prediction_parity": parity, "checkpoint_training_cohorts": cohorts,
              "checkpoint_training_predictions": paired_images, "checkpoint_resources": resources}
    digests = {name: write_parquet(inputs.run / "analysis" / f"{name}.parquet", rows) for name, rows in tables.items()}
    current = snapshot_sources(inputs.project, tuple(inputs.project / name for name in inputs.source_snapshot["files"]))
    if current != inputs.source_snapshot["files"]:
        raise ValueError("checkpoint diagnostics changed an original source")
    result = sealed_record({"schema_version": "imagenetr50-checkpoint-diagnostics-result-v1", "protocol_hash": inputs.run.name,
                            "source_result_hash": inputs.config.source_result_hash, "tables": digests,
                            "source_unchanged": True, "optimizer_steps": 0,
                            "model_forward_images": sum(row["model_forward_images"] for row in resources),
                            "raw_prediction_parity": parity})
    publish_immutable_json(inputs.run / "result.json", result)
    return result


def run() -> Path:
    """Run both diagnostics with source authentication, resumable inference, and the existing report."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(4)
    torch.manual_seed(1993)
    torch.use_deterministic_algorithms(True)
    inputs = bootstrap()
    print(f"Checkpoint diagnostic working artifacts: {inputs.run}", flush=True)
    lock = (inputs.source / "runner.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started, completed, new_forwards = time.monotonic(), 0, 0
    try:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("run checkpoint inference outside the sandbox on the BF16 GPU")
        print("Phase immutable dataset-byte audit", flush=True)
        validate_prepared_dataset(inputs.project / "data/imagenetr50/imagenet-r", inputs.manifest)
        publish_immutable_json(inputs.run / "data_audit.json", sealed_record({"dataset_hash": inputs.manifest.content_hash,
                                 "authenticated_images": 30000, "train_images": 24000, "test_images": 6000}))
        for endpoint in inputs.endpoints:
            for split in (("test", "train") if endpoint.collect_training else ("test",)):
                rows = inputs.manifest.select(split)
                def progress(phase: str, current: int, total: int, elapsed: float) -> None:
                    overall = completed + current
                    eta = (138000 - overall) * (time.monotonic() - started) / max(overall, 1)
                    atomic_write(inputs.run / "status.json", canonical_json_bytes({"phase": phase, "completed_forward_images": overall,
                        "planned_forward_images": 138000, "phase_images": current, "phase_total": total,
                        "overall_eta_seconds": eta, "updated_utc": datetime.now(timezone.utc).isoformat()}))
                    if current == total:
                        print(f"Phase {phase} complete; overall {overall:,}/138,000; measured ETA {eta / 60:.1f} min", flush=True)
                spec = CollectionSpec(inputs.run.name, endpoint.name, split, endpoint.model_sha256,
                                      record_sha256(tuple(row.image_id for row in rows)), inputs.config.batch_size, inputs.config.chunk_images)
                with SelectedImageLoader(inputs.project / "data/imagenetr50/imagenet-r", endpoint.seed, inputs.config.loader_workers) as loader:
                    _, fresh = collect_logits(inputs.run / "collections" / endpoint.name / split, spec, rows, loader,
                                              torch.device("cuda:0"), partial(restored_model, inputs, endpoint), progress)
                completed += len(rows)
                new_forwards += fresh
                if split == "test":
                    metadata, _, _ = load_collection(inputs.run / "collections" / endpoint.name / split)
                    parity = verify_saved_predictions(metadata, pq.read_table(inputs.project / endpoint.predictions).to_pylist())
                    print(f"Original prediction parity: zero class mismatches; maximum NLL difference {parity['maximum_nll_difference']:.3g}", flush=True)
                gc.collect()
                torch.cuda.empty_cache()
        result = analyze(inputs)
        if result["model_forward_images"] != 138000:
            raise ValueError("diagnostic matrix did not complete")
        pointer = sealed_record({"schema_version": "imagenetr50-checkpoint-diagnostics-pointer-v1", "run": str(inputs.run.relative_to(inputs.project)),
                                 "source_result_hash": inputs.config.source_result_hash, "result_hash": result["content_hash"]})
        publish_immutable_json(inputs.source / "reports/checkpoint_diagnostics.json", pointer)
        from apm.continual.vision.imagenetr.srt_reporting import write_srt_report
        report = write_srt_report(inputs.source)
        atomic_write(inputs.run / "status.json", canonical_json_bytes({"phase": "complete", "model_forward_images": 138000,
            "invocation_forward_images": new_forwards, "optimizer_steps": 0, "result_hash": result["content_hash"]}))
        print(f"Diagnostic invocation: {new_forwards:,} forward images, zero optimizer steps", flush=True)
        return report
    except Exception as error:
        atomic_write(inputs.run / "status.json", canonical_json_bytes({"phase": "failed", "error": str(error),
            "completed_forward_images": completed, "invocation_forward_images": new_forwards,
            "updated_utc": datetime.now(timezone.utc).isoformat(), "resumable": True}))
        raise
    finally:
        lock.close()


def main() -> None:
    """Expose the default workflow plus read-only status and cached report rebuilding."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "status", "report"), default="run", nargs="?")
    command = parser.parse_args().command
    if command == "run":
        print(run())
    elif command == "report":
        from apm.continual.vision.imagenetr.srt_reporting import write_srt_report
        print(write_srt_report(Path(load_config().source_run)))
    else:
        import json
        root = Path(load_config().artifact_root)
        print((root / "runs" / json.loads((root / "LATEST_RUN.json").read_text())["run_hash"] / "status.json").read_text())


if __name__ == "__main__":
    main()

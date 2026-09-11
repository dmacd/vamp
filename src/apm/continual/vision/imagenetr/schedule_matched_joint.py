"""Offline all-data rank-16 control with an authenticated replay update schedule."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import fcntl
from functools import partial
import gc
import json
import os
from pathlib import Path

import torch
import yaml

from apm.continual.artifacts import atomic_write, canonical_json_bytes, file_sha256, publish_immutable_json, record_sha256
from apm.continual.vision.imagenetr.constants import TIMM_MODEL_SHA256
from apm.continual.vision.imagenetr.data import DatasetManifest, load_dataset_manifest, validate_prepared_dataset
from apm.continual.vision.imagenetr.joint_convergence import _model
from apm.continual.vision.imagenetr.joint_convergence_training import JointPopulation
from apm.continual.vision.imagenetr.manifests import installed_environment_manifest
from apm.continual.vision.imagenetr.protocol import material_tree_manifest
from apm.continual.vision.imagenetr.schedule_matched_training import ScheduledBatch, audit_schedule_draws, require_schedule, train_schedule_job, validate_schedule_job
from apm.continual.vision.imagenetr.srt_config import SRTConfig, load_srt_config
from apm.continual.vision.imagenetr.srt_data import SelectedImageLoader
from apm.continual.vision.imagenetr.srt_evidence import read_sealed, sealed_record, trace_batches, write_parquet
from apm.continual.vision.imagenetr.srt_training import evaluate_model, restore_trainables


DEFAULT_CONFIG = Path("configs/vision/imagenetr/schedule_matched_joint_r16.yaml")


@dataclass(frozen=True, slots=True)
class ScheduleControlConfig:
    """One fixed source schedule and three offline seeds, with bounded loading."""

    artifact_root: str
    source_run: str
    source_result_hash: str
    followup_hash: str
    followup_result_hash: str
    source_condition: str
    seeds: tuple[int, ...]
    num_workers: int
    fit_probe_images: int


@dataclass(frozen=True, slots=True)
class ScheduleInputs:
    """Frozen control inputs, separated from mutable optimizer execution state."""

    project: Path
    source: Path
    run: Path
    checkpoint: Path
    config: ScheduleControlConfig
    training: SRTConfig
    manifest: DatasetManifest
    population: JointPopulation
    schedule: tuple[ScheduledBatch, ...]
    protocol: dict[str, object]
    source_blocks: tuple[dict[str, object], ...]


def load_config(path: Path = DEFAULT_CONFIG) -> ScheduleControlConfig:
    """Load the requested schedule-matched control without CLI science overrides."""
    values = yaml.safe_load(path.read_text())
    config = ScheduleControlConfig(**{**values, "seeds": tuple(values["seeds"])})
    if (config.seeds != (1993, 1994, 1995) or config.fit_probe_images != 2048 or not 1 <= config.num_workers <= 4
            or config.source_condition != "uniform_h4096_standard_rho80_unit8"):
        raise ValueError("configuration differs from the fixed schedule-matched control")
    return config


def source_schedule(source: Path, config: ScheduleControlConfig) -> tuple[tuple[ScheduledBatch, ...], dict[str, object]]:
    """Authenticate source batch traces and reduce them to work-only instructions."""
    followup = source / "followups" / config.followup_hash
    result = read_sealed(followup / "result.json", "imagenetr50-srt-followup-result-v1")
    if result["content_hash"] != config.followup_result_hash:
        raise ValueError("schedule source follow-up result changed")
    expected = result["conditions"][config.source_condition]
    root = followup / "final" / config.source_condition
    if read_sealed(root / "result.json") != expected:
        raise ValueError("schedule source condition changed")
    batches, blocks = (), ()
    for index, stage in enumerate(expected["rows"], 1):
        if stage["stage"] != index or read_sealed(root / f"stages/{index:03d}/result.json") != stage:
            raise ValueError("schedule source stage changed")
        if any(not (root / chunk["path"]).resolve().is_relative_to(root.resolve() / "trace") for chunk in stage["trace_chunks"]):
            raise ValueError("schedule source trace leaves its namespace")
        rows = trace_batches(root, stage["trace_chunks"])
        if (len(rows) != stage["fit"]["optimizer_steps"]
                or sum(row["presentations"] for row in rows) != stage["fit"]["image_presentations"]
                or any(row["stage"] != index or row["stage_step"] != offset for offset, row in enumerate(rows, 1))):
            raise ValueError("schedule source per-stage work disagrees")
        batches += tuple(ScheduledBatch(row["step"], index, row["presentations"]) for row in rows)
        blocks += ({"block": index, "source_stage_hash": stage["content_hash"], "steps": len(rows),
                    "presentations": sum(row["presentations"] for row in rows), "source_training_seconds": stage["fit"]["wall_seconds"],
                    "batch_evidence": [{"path": chunk["path"], "batches_sha256": chunk["batches_sha256"]} for chunk in stage["trace_chunks"]]},)
    schedule_hash = require_schedule(batches, 64)
    if (len(blocks), len(batches), sum(batch.size for batch in batches)) != (50, 56243, 844640):
        raise ValueError("schedule source has unexpected total work")
    return batches, {"schedule_hash": schedule_hash, "source_job_hash": expected["content_hash"], "blocks": blocks}


def bootstrap_control(config_path: Path = DEFAULT_CONFIG) -> ScheduleInputs:
    """Freeze every scientific input while preserving the earlier experiment trees."""
    project = config_path.resolve().parents[3]
    config = load_config(config_path)
    source = project / config.source_run
    original = read_sealed(source / "result.json", "imagenetr50-srt-result-v1")
    source_protocol = read_sealed(source / "protocol.json")
    if original["content_hash"] != config.source_result_hash:
        raise ValueError("original SRT source changed")
    original_training = load_srt_config(project / "configs/vision/imagenetr/srt_r16_v1.yaml")
    if original_training.content_hash != source_protocol["config_hash"]:
        raise ValueError("source optimization configuration changed")
    original_code = read_sealed(source / "code_manifest.json")
    if any(file_sha256(project / row["path"]) != row["sha256"] for row in original_code["files"]):
        raise ValueError("source training code changed")
    environment = installed_environment_manifest(tuple(row["name"] for row in read_sealed(source / "environment_manifest.json")["packages"]))
    if environment["content_hash"] != source_protocol["environment_hash"]:
        raise ValueError("offline control environment differs from replay")
    manifest = load_dataset_manifest(project / "data/imagenetr50/imagenet-r/dataset_manifest.json")
    if manifest.content_hash != source_protocol["dataset_hash"]:
        raise ValueError("offline control dataset differs from replay")
    fitting = manifest.select("train")
    probe = tuple(sorted(fitting, key=lambda row: record_sha256(["joint-fit-probe-v1", row.image_id]))[:config.fit_probe_images])
    population = JointPopulation(fitting, (), probe)
    if len(fitting) != 24000 or len(manifest.select("test")) != 6000:
        raise ValueError("offline control population sizes changed")
    schedule, evidence = source_schedule(source, config)
    package = Path(__file__).parent
    material = material_tree_manifest(tuple(package / name for name in (
        "schedule_matched_joint.py", "schedule_matched_training.py", "joint_convergence.py", "joint_convergence_training.py"))
        + (config_path.resolve(), project / "docs/imagenetr50_schedule_matched_joint_protocol.md"))
    training = replace(original_training, num_workers=config.num_workers)
    protocol = sealed_record({
        "schema_version": "imagenetr50-schedule-matched-protocol-v1", "config": asdict(config),
        "source_result_hash": original["content_hash"], "source_code_hash": original_code["content_hash"],
        "source_job_hash": evidence["source_job_hash"], "schedule_hash": evidence["schedule_hash"],
        "dataset_hash": manifest.content_hash, "model_sha256": TIMM_MODEL_SHA256,
        "code_hash": material["content_hash"], "environment_hash": environment["content_hash"],
        "optimizer_config_hash": training.content_hash,
        "optimizer": {key: getattr(training, key) for key in ("lora_learning_rate", "head_learning_rate", "momentum", "weight_decay")},
        "planned_steps_per_seed": len(schedule), "planned_presentations_per_seed": sum(batch.size for batch in schedule),
        "selection": "fixed final schedule endpoint; no new validation or test-based selection",
        "sampling": "uniform all training images; without replacement within batch, independent between batches",
        "model": "frozen ViT-B/16; rank/alpha 16 QKV+fc1; 200 affine rows active from the first update",
    })
    run = project / config.artifact_root / "runs" / protocol["content_hash"]
    schedule_file_hash = write_parquet(run / "schedule.parquet", tuple(asdict(batch) for batch in schedule))
    schedule_record = sealed_record({"schema_version": "imagenetr50-schedule-matched-schedule-v1", **evidence,
                                     "schedule_parquet_sha256": schedule_file_hash})
    populations = sealed_record({"schema_version": "imagenetr50-schedule-matched-populations-v1",
                                 "fitting": [row.image_id for row in fitting], "probe": [row.image_id for row in probe],
                                 "test": [row.image_id for row in manifest.select("test")]})
    for name, record in (("protocol.json", protocol), ("code_manifest.json", material), ("environment_manifest.json", environment),
                         ("schedule.json", schedule_record), ("populations.json", populations)):
        publish_immutable_json(run / name, record)
    candidates = tuple((project / "data/imagenetr50/model_cache").rglob(TIMM_MODEL_SHA256))
    if len(candidates) != 1 or file_sha256(candidates[0]) != TIMM_MODEL_SHA256:
        raise ValueError("pinned backbone failed authentication")
    atomic_write(project / config.artifact_root / "LATEST_RUN.json", canonical_json_bytes({"run_hash": run.name}))
    return ScheduleInputs(project, source, run, candidates[0], config, training, manifest, population, schedule, protocol, evidence["blocks"])


def _progress(inputs: ScheduleInputs, phase: str, current: dict[str, object] | None = None) -> None:
    """Publish live work and an ETA scaled by measured source-schedule block costs."""
    current = current or {}
    completed = tuple(read_sealed(path) for path in (inputs.run / "seeds").glob("*/result.json") if path.parent.name != f"seed_{current.get('seed')}")
    images = sum(row["training_presentations"] for row in completed) + int(current.get("presentations", 0))
    history = tuple(current.get("blocks", ()))
    measured = sum(row["training_wall_seconds"] + row["evaluation_and_artifact_seconds"] for row in completed)
    measured += sum(row["training_wall_seconds"] + row["evaluation_and_artifact_seconds"] for row in history)
    source_total = sum(row["source_training_seconds"] for row in inputs.source_blocks)
    source_done = len(completed) * source_total + sum(row["source_training_seconds"] for row in inputs.source_blocks[:len(history)])
    eta = (3 * source_total - source_done) * measured / source_done if source_done else None
    state = {"phase": phase, "updated_utc": datetime.now(timezone.utc).isoformat(), "run_hash": inputs.run.name,
             "completed_presentations": images, "planned_presentations": 3 * 844640, "overall_eta_seconds": eta,
             **{key: value for key, value in current.items() if key != "blocks"}}
    atomic_write(inputs.run / "status.json", canonical_json_bytes(state))
    if not current or current.get("boundary"):
        remaining = f"{eta / 60:.1f} min" if eta is not None else "measuring throughput"
        print(f"Phase {phase}: {images:,}/{3 * 844640:,} training presentations; overall ETA {remaining}", flush=True)


def real_preflight(inputs: ScheduleInputs, device: torch.device) -> dict[str, object]:
    """Authenticate images and prove exact resume with mixed real-GPU batch sizes."""
    from safetensors.torch import load_file
    root = inputs.run / "preflight"
    if (root / "result.json").is_file():
        return read_sealed(root / "result.json")
    validate_prepared_dataset(inputs.project / "data/imagenetr50/imagenet-r", inputs.manifest)
    population = JointPopulation(inputs.population.fitting[:128], (), inputs.population.fitting[:64])
    schedule = tuple(ScheduledBatch(index + 1, index // 2 + 1, size) for index, size in enumerate((64, 7, 1, 32)))
    with SelectedImageLoader(inputs.project / "data/imagenetr50/imagenet-r", 51993, inputs.config.num_workers) as loader:
        arguments = dict(protocol_hash=inputs.run.name, seed=1993, population=population, schedule=schedule,
                         optimizer_config=replace(inputs.training, checkpoint_steps=1), loader=loader, device=device,
                         model_factory=partial(_model, inputs.checkpoint))
        direct = train_schedule_job(root / "direct", **arguments)
        def interrupt(row: dict[str, object]) -> None:
            if row["steps"] == 1:
                raise InterruptedError("intentional mixed-batch preflight interruption")
        try:
            train_schedule_job(root / "resumed", progress_callback=interrupt, **arguments)
        except InterruptedError:
            pass
        gc.collect()
        torch.cuda.empty_cache()
        resumed = train_schedule_job(root / "resumed", **arguments)
        for block in (1, 2):
            first, second = (load_file(str(root / name / f"blocks/{block:03d}/model.safetensors")) for name in ("direct", "resumed"))
            if set(first) != set(second) or any(not torch.equal(first[name], second[name]) for name in first):
                raise ValueError("mixed-batch real-model resume differs")
        reused = train_schedule_job(root / "resumed", **arguments)
        if reused["invocation_optimizer_steps"] or reused["content_hash"] != resumed["content_hash"]:
            raise ValueError("mixed-batch preflight reuse differs")
    result = sealed_record({"schema_version": "imagenetr50-schedule-matched-preflight-v1", "authenticated_images": 30000,
                            "batch_sizes": [64, 7, 1, 32], "interrupted_resume_exact": True, "zero_step_reuse": True,
                            "direct_hash": direct["content_hash"], "resumed_hash": resumed["content_hash"],
                            "peak_vram_bytes": max(row["peak_vram_bytes"] for row in direct["blocks"])})
    publish_immutable_json(root / "result.json", result)
    return result


def evaluate_final(inputs: ScheduleInputs, device: torch.device) -> tuple[dict[str, object], ...]:
    """Test only completed final all-class models after all three seeds finish."""
    from safetensors.torch import load_file
    roots = tuple(inputs.run / "seeds" / f"seed_{seed}" for seed in inputs.config.seeds)
    if any(not (root / "result.json").is_file() for root in roots):
        raise ValueError("test evaluation requires all offline fits complete")
    evaluations = []
    for seed, root in zip(inputs.config.seeds, roots, strict=True):
        weights = root / "blocks/050/model.safetensors"
        output = inputs.run / "evaluations" / f"seed_{seed}"
        if (output / "result.json").is_file():
            record = read_sealed(output / "result.json")
            if record["model_sha256"] != file_sha256(weights) or record["predictions_sha256"] != file_sha256(output / "predictions.parquet"):
                raise ValueError("offline final prediction evidence changed")
        else:
            model = _model(inputs.checkpoint, seed).to(device)
            restore_trainables(model, load_file(str(weights)))
            with SelectedImageLoader(inputs.project / "data/imagenetr50/imagenet-r", seed + 50000, inputs.config.num_workers) as loader:
                metrics, predictions = evaluate_model(model, inputs.manifest.select("test"), loader, device, 64, 50)
            record = sealed_record({"schema_version": "imagenetr50-schedule-matched-evaluation-v1", "seed": seed,
                                    "protocol_hash": inputs.run.name, "model_sha256": file_sha256(weights), "metrics": metrics,
                                    "predictions_sha256": write_parquet(output / "predictions.parquet", predictions)})
            publish_immutable_json(output / "result.json", record)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        evaluations.append(record)
        print(f"Final test seed {seed}: {record['metrics']['accuracy']:.3f}% / NLL {record['metrics']['nll']:.5f}", flush=True)
    return tuple(evaluations)


def run_control() -> Path:
    """Run the preflight, three fixed fits, endpoint tests, reuse proof, and existing report."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(4)
    torch.manual_seed(1993)
    torch.use_deterministic_algorithms(True)
    inputs = bootstrap_control()
    print(f"Schedule-matched joint working artifacts: {inputs.run}", flush=True)
    lock = (inputs.source / "runner.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("run the control outside the sandbox on the local BF16 GPU")
        device = torch.device("cuda:0")
        _progress(inputs, "preflight")
        preflight = real_preflight(inputs, device)
        common = dict(protocol_hash=inputs.run.name, population=inputs.population, schedule=inputs.schedule,
                      optimizer_config=inputs.training, device=device, model_factory=partial(_model, inputs.checkpoint))
        jobs, reuse, draw_audits = {}, {}, {}
        for seed in inputs.config.seeds:
            phase, root = f"offline seed {seed}", inputs.run / "seeds" / f"seed_{seed}"
            _progress(inputs, phase)
            with SelectedImageLoader(inputs.project / "data/imagenetr50/imagenet-r", seed + 50000, inputs.config.num_workers) as loader:
                train_schedule_job(root, seed=seed, loader=loader, progress_callback=partial(_progress, inputs, phase), **common)
            jobs[f"seed_{seed}"] = validate_schedule_job(root)
            print(f"Phase sample-draw audit, seed {seed}", flush=True)
            draw_audits[f"seed_{seed}"] = audit_schedule_draws(root, inputs.population, inputs.schedule)
            gc.collect()
            torch.cuda.empty_cache()
        _progress(inputs, "final tests")
        evaluations = evaluate_final(inputs, device)
        for seed in inputs.config.seeds:
            with SelectedImageLoader(inputs.project / "data/imagenetr50/imagenet-r", seed + 50000, 0) as loader:
                repeated = train_schedule_job(inputs.run / "seeds" / f"seed_{seed}", seed=seed, loader=loader, **common)
            reuse[f"seed_{seed}"] = {"optimizer_steps": repeated["invocation_optimizer_steps"], "result_hash": repeated["content_hash"]}
        if any(row["optimizer_steps"] or row["result_hash"] != jobs[name]["content_hash"] for name, row in reuse.items()):
            raise ValueError("offline control failed zero-step reuse")
        if read_sealed(inputs.source / "result.json")["content_hash"] != inputs.config.source_result_hash:
            raise ValueError("original source result changed")
        _, source_evidence = source_schedule(inputs.source, inputs.config)
        if source_evidence["schedule_hash"] != inputs.protocol["schedule_hash"]:
            raise ValueError("source replay schedule changed during the control")
        result = sealed_record({"schema_version": "imagenetr50-schedule-matched-result-v1", "protocol_hash": inputs.run.name,
                                "source_result_hash": inputs.config.source_result_hash, "preflight_hash": preflight["content_hash"],
                                "schedule_hash": inputs.protocol["schedule_hash"], "jobs": jobs, "evaluations": evaluations,
                                "reuse": reuse, "draw_audits": draw_audits, "zero_step_reuse": True, "source_unchanged": True})
        publish_immutable_json(inputs.run / "result.json", result)
        pointer = sealed_record({"schema_version": "imagenetr50-schedule-matched-pointer-v1", "run": str(inputs.run.relative_to(inputs.project)),
                                 "result_hash": result["content_hash"], "source_result_hash": inputs.config.source_result_hash})
        publish_immutable_json(inputs.source / "reports/schedule_matched_joint.json", pointer)
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
    """Expose one default run, read-only status, and an existing-report rebuild."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("run", "status", "report"), default="run")
    command = parser.parse_args().command
    if command == "run":
        print(run_control())
    elif command == "status":
        root = Path(load_config().artifact_root)
        latest = json.loads((root / "LATEST_RUN.json").read_text())
        print((root / "runs" / latest["run_hash"] / "status.json").read_text())
    else:
        from apm.continual.vision.imagenetr.srt_reporting import write_srt_report
        print(write_srt_report(Path(load_config().source_run)))


if __name__ == "__main__":
    main()

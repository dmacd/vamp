"""Finite SRT calibration, four primary runs, and an isolated comparison report."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path

from apm.continual.artifacts import (
    atomic_write, canonical_json_bytes, file_sha256, load_canonical_json,
    publish_immutable_bytes, publish_immutable_json,
)
from apm.continual.vision.imagenetr.constants import TIMM_MODEL_SHA256
from apm.continual.vision.imagenetr.data import DatasetManifest, ImageRecord, load_dataset_manifest, validate_prepared_dataset
from apm.continual.vision.imagenetr.manifests import installed_environment_manifest
from apm.continual.vision.imagenetr.protocol import material_tree_manifest
from apm.continual.vision.imagenetr.router_artifacts import build_router_split
from apm.continual.vision.imagenetr.srt_config import DEFAULT_SRT_CONFIG, SRTConfig, load_srt_config, presentation_budget
from apm.continual.vision.imagenetr.srt_evidence import read_sealed, sealed_record, write_parquet
from apm.continual.vision.imagenetr.srt_scheduler import RecallPolicy


@dataclass(frozen=True, slots=True)
class SRTInputs:
    """Authenticated immutable populations and one isolated run directory."""

    project: Path
    run: Path
    config: SRTConfig
    checkpoint: Path
    manifest: DatasetManifest
    fitting: tuple[ImageRecord, ...]
    validation: tuple[ImageRecord, ...]
    protocol: dict[str, object]


def _reference_files(project: Path, config: SRTConfig) -> tuple[tuple[Path, str, str], ...]:
    reference = project / config.reference_run
    pointer = read_sealed(reference / "reports/mlp_extension.json")
    mlp = project / "artifacts/imagenetr50/persistent_mlp_v16/runs" / Path(pointer["run"]).name / "evaluations/result.json"
    report = read_sealed(reference / "reports/report_manifest.json")
    return (
        (reference / "evaluations/result.json", config.reference_result_sha256, "affine_result.json"),
        (mlp, config.mlp_result_sha256, "mlp_result.json"),
        (reference / "reports/stage_accuracy.png", config.reference_figure_sha256, "previous_stage_accuracy.png"),
        (reference / "reports/resource_metrics.json", report["resource_metrics_sha256"], "resource_metrics.json"),
        (project / "output/pdf/imagenetr50_persistent_affine_v15_report.pdf", config.reference_pdf_sha256, "previous_report.pdf"),
    )


def verify_references(project: Path, config: SRTConfig) -> tuple[tuple[Path, str, str], ...]:
    """Require byte-identical previous results, resource evidence, figure, and PDF."""
    files = _reference_files(project, config)
    for path, expected, _ in files:
        if file_sha256(path) != expected:
            raise ValueError(f"the existing comparison artifact changed: {path}")
    return files


def bootstrap_srt(config_path: Path = DEFAULT_SRT_CONFIG) -> SRTInputs:
    """Authenticate inputs and freeze identity without invoking older workflows."""
    project = config_path.resolve().parents[3]
    config = load_srt_config(config_path)
    references = verify_references(project, config)
    prepared = project / "data/imagenetr50/imagenet-r"
    manifest = load_dataset_manifest(prepared / "dataset_manifest.json")
    split = build_router_split(manifest, .8, config.seed)
    if manifest.content_hash != config.dataset_hash or split.content_hash != config.split_hash:
        raise ValueError("SRT data or calibration membership differs from the existing protocol")
    fitting_ids, validation_ids = frozenset(split.fit_image_ids), frozenset(split.validation_image_ids)
    fitting = tuple(row for row in manifest.images if row.image_id in fitting_ids)
    validation = tuple(row for row in manifest.images if row.image_id in validation_ids)
    package = project / "src/apm/continual/vision/imagenetr"
    material = material_tree_manifest(tuple(package / name for name in (
        "srt_config.py", "srt_scheduler.py", "srt_data.py", "srt_evidence.py", "srt_training.py", "srt_preflight.py", "srt_workflow.py",
        "data.py", "model.py", "heads.py", "lora.py", "checkpoints.py", "router_artifacts.py", "router_protocol.py",
        "constants.py", "manifests.py", "protocol.py", "merging/common.py",
    )) + (project / "src/apm/continual/artifacts.py", config_path.resolve(),
           project / "docs/imagenetr50_srt_protocol.md"))
    environment = installed_environment_manifest((
        "torch", "torchvision", "timm", "numpy", "Pillow", "pyrsistent", "safetensors", "pyarrow", "PyYAML", "tqdm",
    ))
    if any(row["version"] == "MISSING" for row in environment["packages"]):
        raise RuntimeError("the isolated vision training environment is incomplete")
    protocol = sealed_record({
        "schema_version": "imagenetr50-srt-protocol-v1", "config_hash": config.content_hash,
        "dataset_hash": manifest.content_hash, "calibration_split_hash": split.content_hash,
        "model_sha256": TIMM_MODEL_SHA256, "code_hash": material["content_hash"],
        "environment_hash": environment["content_hash"],
        "paper": "https://arxiv.org/html/2608.17530v1", "paper_code_available": False,
        "reference_hashes": {name: digest for _, digest, name in references},
        "training_model": "one persistent rank/alpha-16 QKV+fc1 LoRA and growing all-seen-class affine head",
        "quality": "pre-update correct-label probability from the augmented training forward",
        "scheduler_clock": "one tick per update; jump to next due time only when both due pools are empty",
        "calibration_selection": "mean_stage_accuracy_then_mean_nll_then_canonical_policy_order",
        "training_seed": config.seed,
    })
    run = project / config.artifact_root / "runs" / protocol["content_hash"]
    for name, record in (
        ("protocol.json", protocol), ("config_resolved.json", sealed_record(asdict(config))),
        ("code_manifest.json", material), ("environment_manifest.json", environment),
        ("calibration_split.json", split.as_record()),
    ):
        publish_immutable_json(run / name, record)
    for path, _, name in references:
        if name != "previous_report.pdf":
            publish_immutable_bytes(run / "references" / name, path.read_bytes())
    for phase, rows in (("calibration", fitting), ("final", manifest.select("train"))):
        index_path = run / f"{phase}/image_index.parquet"
        if not index_path.is_file():
            write_parquet(index_path, tuple({"image": index, "image_id": row.image_id,
                                            "arrival_stage": row.task_index + 1, "class_id": row.remapped_class_index}
                                           for index, row in enumerate(rows)))
    candidates = tuple((project / "data/imagenetr50/model_cache").rglob(TIMM_MODEL_SHA256))
    if len(candidates) != 1 or file_sha256(candidates[0]) != TIMM_MODEL_SHA256:
        raise ValueError("the pinned local ViT checkpoint failed authentication")
    atomic_write(project / config.artifact_root / "LATEST_RUN.json", canonical_json_bytes({"run_hash": protocol["content_hash"]}))
    return SRTInputs(project, run, config, candidates[0], manifest, fitting, validation, protocol)


def _job_summary(path: Path) -> dict[str, object]:
    result = read_sealed(path / "result.json", "imagenetr50-srt-job-result-v1")
    return {"job": path.name, "result_hash": result["content_hash"], "policy": result["policy"],
            "mean_accuracy": math.fsum(row["evaluation"]["accuracy"] for row in result["rows"]) / len(result["rows"]),
            "mean_nll": math.fsum(row["evaluation"]["nll"] for row in result["rows"]) / len(result["rows"]),
            "stages": len(result["rows"])}


def select_candidates(rows: tuple[dict[str, object], ...], count: int) -> tuple[dict[str, object], ...]:
    """Choose accuracy first, NLL second, then a deterministic canonical policy tie."""
    return tuple(sorted(rows, key=lambda row: (-row["mean_accuracy"], row["mean_nll"], canonical_json_bytes(row["policy"])))[:count])


def _total_budget(inputs: SRTInputs) -> int:
    def through(rows: tuple[ImageRecord, ...], capacity: int, stages: int) -> int:
        counts = Counter(row.task_index for row in rows)
        return sum(presentation_budget(counts[stage], sum(counts[index] for index in range(stage)), capacity)
                   for stage in range(stages))
    return sum(18 * through(inputs.fitting, capacity, 16)
               + 2 * (through(inputs.fitting, capacity, 50) - through(inputs.fitting, capacity, 16))
               + 2 * through(inputs.manifest.select("train"), capacity, 50)
               for capacity in inputs.config.budgets)


def _progress(inputs: SRTInputs, phase: str, current: dict[str, object] | None = None) -> None:
    current = current or {}
    jobs = tuple(path for phase_name in ("calibration", "final") for path in (inputs.run / phase_name).glob("*/result.json")
                 if path.parent.name != current.get("job"))
    prior = tuple(read_sealed(path) for path in jobs)
    done = sum(row["image_presentations"] for row in prior) + int(current.get("image_presentations", 0))
    measured_images = sum(row["image_presentations"] for row in prior)
    measured_seconds = math.fsum(stage["fit"]["wall_seconds"] for row in prior for stage in row["rows"])
    remaining = max(0, _total_budget(inputs) - done)
    eta = remaining * measured_seconds / measured_images if measured_images else None
    state = {"schema_version": "imagenetr50-srt-status-v1", "run_hash": inputs.protocol["content_hash"],
             "phase": phase, "updated_utc": datetime.now(timezone.utc).isoformat(),
             "completed_presentations": done, "planned_presentations": _total_budget(inputs),
             "overall_eta_seconds": eta, **current}
    atomic_write(inputs.run / "status.json", canonical_json_bytes(state))
    if not current:
        print(f"\nPhase: {phase}. Planned experiment work {done:,}/{_total_budget(inputs):,} images; "
              f"overall ETA {eta / 3600:.2f} h" if eta is not None else f"\nPhase: {phase}. Measuring initial throughput.", flush=True)


def run_srt(config_path: Path = DEFAULT_SRT_CONFIG) -> Path:
    """Execute preflight, fixed calibration, both paired budgets, and the separate report."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from apm.continual.vision.imagenetr.srt_data import SelectedImageLoader
    from apm.continual.vision.imagenetr.srt_preflight import real_srt_preflight
    from apm.continual.vision.imagenetr.srt_training import run_training_job

    torch.set_num_threads(4)
    torch.manual_seed(1993)
    torch.use_deterministic_algorithms(True)
    inputs = bootstrap_srt(config_path)
    config, run = inputs.config, inputs.run
    print(f"SRT working artifacts: {run}", flush=True)
    lock = (run / "runner.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock.close()
        raise RuntimeError("this SRT experiment already has an active worker") from error
    try:
        device = torch.device("cuda:0")
        if not torch.cuda.is_available():
            raise RuntimeError("run the SRT workflow outside the sandbox to access the GPU")
        if not (run / "preflight/result.json").is_file():
            _progress(inputs, "authenticate dataset and run real-model smoke")
            validate_prepared_dataset(inputs.project / "data/imagenetr50/imagenet-r", inputs.manifest)
        with SelectedImageLoader(inputs.project / "data/imagenetr50/imagenet-r", config.seed + 50_000, config.num_workers) as loader:
            real_srt_preflight(run / "preflight", inputs.protocol["content_hash"], inputs.checkpoint, config,
                               inputs.fitting, inputs.validation, loader, device)
            common = dict(protocol_hash=inputs.protocol["content_hash"], checkpoint_path=inputs.checkpoint,
                          config=config, loader=loader, device=device)
            screens = {}
            for capacity in config.budgets:
                for index, policy in enumerate(config.policies, 1):
                    phase = f"calibration H{capacity}, screen {index}/18 through task 16"
                    _progress(inputs, phase)
                    root = run / "calibration" / f"h{capacity}_{policy.name}"
                    run_training_job(root, policy=policy, capacity=capacity, method="srt", training_rows=inputs.fitting,
                                     evaluation_rows=inputs.validation, target_stage=16,
                                     progress_callback=lambda row, phase=phase: _progress(inputs, phase, row), **common)
                candidates = tuple(_job_summary(run / "calibration" / f"h{capacity}_{policy.name}") for policy in config.policies)
                screens[str(capacity)] = select_candidates(candidates, config.finalists)
            selection_path = run / "calibration/screen_selection.json"
            if selection_path.is_file():
                screens = read_sealed(selection_path)["selected"]
            else:
                publish_immutable_json(selection_path, sealed_record({"schema_version": "imagenetr50-srt-screen-selection-v1", "selected": screens}))
            selected = {}
            for capacity in config.budgets:
                for candidate in screens[str(capacity)]:
                    policy_values = dict(candidate["policy"])
                    policy_values["thresholds"] = tuple(policy_values["thresholds"])
                    policy = RecallPolicy(**policy_values)
                    phase = f"calibration finalist H{capacity} {policy.name}, through task 50"
                    _progress(inputs, phase)
                    run_training_job(run / "calibration" / candidate["job"], policy=policy, capacity=capacity, method="srt",
                                     training_rows=inputs.fitting, evaluation_rows=inputs.validation, target_stage=50,
                                     progress_callback=lambda row, phase=phase: _progress(inputs, phase, row), **common)
                finalists = tuple(_job_summary(run / "calibration" / candidate["job"]) for candidate in screens[str(capacity)])
                selected[str(capacity)] = select_candidates(finalists, 1)[0]
            selected_path = run / "calibration/selected.json"
            publish_immutable_json(selected_path, sealed_record({"schema_version": "imagenetr50-srt-selection-v1", "selected": selected,
                                                               "test_used": False, "criterion": "mean_accuracy_then_mean_nll_then_canonical_policy"}))
            for capacity in config.budgets:
                values = dict(selected[str(capacity)]["policy"])
                values["thresholds"] = tuple(values["thresholds"])
                for method in ("srt", "uniform"):
                    phase = f"final full-data {method} H{capacity}, tasks 1-50"
                    _progress(inputs, phase)
                    run_training_job(run / "final" / f"{method}_h{capacity}", policy=RecallPolicy(**values), capacity=capacity,
                                     method=method, training_rows=inputs.manifest.select("train"), evaluation_rows=inputs.manifest.select("test"),
                                     target_stage=50, paired_root=run / "final" / f"srt_h{capacity}" if method == "uniform" else None,
                                     progress_callback=lambda row, phase=phase: _progress(inputs, phase, row), **common)
            verify_references(inputs.project, config)
            results = {f"{method}_h{capacity}": read_sealed(run / "final" / f"{method}_h{capacity}" / "result.json")
                       for capacity in config.budgets for method in ("srt", "uniform")}
            expected_counts = {1024: 294368, 4096: 844640}
            if any(len(result["rows"]) != 50 or result["image_presentations"] != expected_counts[result["capacity"]]
                   for result in results.values()):
                raise ValueError("the final SRT matrix is incomplete or has incorrect exposure budgets")
            reuse = {}
            for name, result in results.items():
                values = dict(result["policy"])
                values["thresholds"] = tuple(values["thresholds"])
                repeated = run_training_job(run / "final" / name, policy=RecallPolicy(**values), capacity=result["capacity"],
                                            method=result["method"], training_rows=inputs.manifest.select("train"),
                                            evaluation_rows=inputs.manifest.select("test"), target_stage=50,
                                            paired_root=run / "final" / f"srt_h{result['capacity']}" if result["method"] == "uniform" else None, **common)
                reuse[name] = {"optimizer_steps": repeated["invocation_optimizer_steps"], "result_hash": repeated["content_hash"]}
            publish_immutable_json(run / "reuse.json", sealed_record({"schema_version": "imagenetr50-srt-reuse-v1", "conditions": reuse}))
            publish_immutable_json(run / "result.json", sealed_record({
                "schema_version": "imagenetr50-srt-result-v1", "protocol_hash": inputs.protocol["content_hash"],
                "selection_hash": read_sealed(selected_path)["content_hash"], "conditions": results,
                "references_unchanged": True, "zero_step_reuse": all(item["optimizer_steps"] == 0 for item in reuse.values()),
            }))
        _progress(inputs, "build and verify separate report")
        from apm.continual.vision.imagenetr.srt_reporting import write_srt_report
        report = write_srt_report(run)
        _progress(inputs, "complete")
        return report
    except BaseException as error:
        atomic_write(run / "failure.json", canonical_json_bytes({
            "exception": type(error).__name__, "message": str(error),
            "utc": datetime.now(timezone.utc).isoformat(), "resumable": True,
        }))
        raise
    finally:
        lock.close()


def main() -> None:
    """Expose one default run and lightweight status/report commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("run", "status", "report"), default="run")
    parser.add_argument("--config", type=Path, default=DEFAULT_SRT_CONFIG)
    args = parser.parse_args()
    if args.command == "run":
        print(run_srt(args.config))
        return
    config = load_srt_config(args.config)
    root = args.config.resolve().parents[3] / config.artifact_root
    latest = load_canonical_json(root / "LATEST_RUN.json")
    run = root / "runs" / latest["run_hash"]
    if args.command == "status":
        print(json.dumps(load_canonical_json(run / "status.json"), indent=2))
    else:
        from apm.continual.vision.imagenetr.srt_reporting import write_srt_report
        print(write_srt_report(run))


if __name__ == "__main__":
    main()

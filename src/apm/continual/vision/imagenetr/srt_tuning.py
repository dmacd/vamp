"""Resumable task-50 validation search for the existing H=128 SRT experiment."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
from functools import partial
import math
import os
from pathlib import Path
import time

from apm.continual.artifacts import atomic_write, canonical_json_bytes, publish_immutable_json, record_sha256
from apm.continual.vision.imagenetr.data import ImageRecord, validate_prepared_dataset
from apm.continual.vision.imagenetr.protocol import material_tree_manifest
from apm.continual.vision.imagenetr.router_artifacts import build_router_split
from apm.continual.vision.imagenetr.srt_evidence import read_sealed, sealed_record, write_parquet
from apm.continual.vision.imagenetr.srt_followup import FollowupInputs, bootstrap_followup, followup_presentations
from apm.continual.vision.imagenetr.srt_tuning_config import (
    DEFAULT_CONFIG, SRTCandidate, SRTTuningConfig, candidate_from_record, load_config,
    policy_candidates, rate_candidates, refinement_candidates, select_winner,
)


@dataclass(frozen=True, slots=True)
class TuningInputs:
    """Authenticated fixed-policy baseline and disjoint training-derived search populations."""

    base: FollowupInputs
    config: SRTTuningConfig
    run: Path
    fitting: tuple[ImageRecord, ...]
    validation: tuple[ImageRecord, ...]
    protocol: dict[str, object]


def bootstrap_tuning(config_path: Path = DEFAULT_CONFIG) -> TuningInputs:
    """Freeze the finite search after authenticating the completed H=128 baseline and source code."""
    project = config_path.resolve().parents[3]
    config = load_config(config_path)
    base = bootstrap_followup(project / config.baseline_config)
    baseline = read_sealed(base.run / "result.json", "imagenetr50-srt-followup-result-v1")
    if (base.config.capacity != config.capacity or base.config.profiles != ("standard",)
            or base.config.historical_fraction != .8 or base.config.interval_unit != 8
            or baseline["protocol_hash"] != base.run.name or not baseline["zero_step_reuse"]):
        raise ValueError("tuning requires the completed unchanged H=128 standard baseline")
    split = build_router_split(base.manifest, .8, base.training.seed)
    fitting = base.manifest.select("train", image_ids=split.fit_image_ids)
    validation = base.manifest.select("train", image_ids=split.validation_image_ids)
    if (split.content_hash != base.training.split_hash or (len(fitting), len(validation)) != (19200, 4800)
            or {row.image_id for row in fitting} & {row.image_id for row in validation}
            or any(row.split != "train" for row in (*fitting, *validation))):
        raise ValueError("tuning fit/validation membership differs from the frozen training-derived split")
    material = material_tree_manifest((Path(__file__), Path(__file__).with_name("srt_tuning_config.py"),
                                       config_path.resolve(), project / "docs/imagenetr50_srt_small_budget_tuning_protocol.md"))
    initial = policy_candidates(config, base.training)
    protocol = sealed_record({
        "schema_version": "imagenetr50-srt-tuning-protocol-v1", "config": asdict(config),
        "baseline_protocol_hash": base.run.name, "baseline_result_hash": baseline["content_hash"],
        "source_result_hash": base.config.source_result_hash, "source_code_hash": base.protocol["source_code_hash"],
        "dataset_hash": base.manifest.content_hash, "model_sha256": base.protocol["model_sha256"],
        "environment_hash": base.protocol["environment_hash"], "code_hash": material["content_hash"],
        "split_hash": split.content_hash, "fit_ids_hash": record_sha256([row.image_id for row in fitting]),
        "validation_ids_hash": record_sha256([row.image_id for row in validation]),
        "initial_candidates": [asdict(candidate) for candidate in initial],
        "maximum_candidates": config.maximum_candidates,
        "selection": "task-50 validation accuracy descending, NLL ascending, candidate content hash ascending",
        "test_used_for_selection": False, "early_task_elimination": False,
        "final_control": "uniform copies selected SRT batch schedule and optimizer settings; not independently tuned uniform",
    })
    run = base.source / "tuning" / protocol["content_hash"]
    for name, record in (("protocol.json", protocol), ("code_manifest.json", material), ("calibration_split.json", split.as_record())):
        publish_immutable_json(run / name, record)
    index = run / "calibration/image_index.parquet"
    if not index.is_file():
        write_parquet(index, tuple({"image": index, "image_id": row.image_id, "arrival_stage": row.task_index + 1,
                                    "class_id": row.remapped_class_index} for index, row in enumerate(fitting)))
    atomic_write(base.source / "tuning/LATEST_RUN.json", canonical_json_bytes({"run_hash": run.name}))
    return TuningInputs(base, config, run, fitting, validation, protocol)


def candidate_summary(inputs: TuningInputs, candidate: SRTCandidate) -> dict[str, object]:
    """Authenticate a complete validation job before exposing its endpoint to selection."""
    root = inputs.run / "calibration" / candidate.name
    result = read_sealed(root / "result.json", "imagenetr50-srt-job-result-v1")
    definition = read_sealed(root / "job.json", "imagenetr50-srt-job-v1")
    counts = Counter(row.task_index for row in inputs.fitting)
    expected_work = followup_presentations(inputs.config.capacity, tuple(counts[stage] for stage in range(50)))
    if (result["job_hash"] != definition["content_hash"] or definition["protocol_hash"] != inputs.run.name
            or definition["config_hash"] != candidate.optimizer_config(inputs.base.training).content_hash
            or definition["training_ids_hash"] != inputs.protocol["fit_ids_hash"]
            or definition["evaluation_ids_hash"] != inputs.protocol["validation_ids_hash"]
            or definition["policy"] != {**asdict(candidate.policy), "thresholds": list(candidate.policy.thresholds)}
            or definition["capacity"] != inputs.config.capacity or definition["method"] != "srt"
            or result["policy"] != definition["policy"]
            or result["method"] != "srt" or result["capacity"] != inputs.config.capacity
            or len(result["rows"]) != 50 or result["image_presentations"] != expected_work
            or result["rows"][-1]["evaluation"]["examples"] != len(inputs.validation)):
        raise ValueError("candidate differs from its validation-only scientific identity")
    endpoint = result["rows"][-1]["evaluation"]
    return sealed_record({
        "schema_version": "imagenetr50-srt-candidate-summary-v1", "candidate": asdict(candidate),
        "candidate_hash": candidate.content_hash, "job": candidate.name, "job_hash": definition["content_hash"],
        "result_hash": result["content_hash"], "status": "complete", "evaluation_role": "validation",
        "validation_ids_hash": inputs.protocol["validation_ids_hash"], "completed_stages": 50,
        "validation_accuracy": endpoint["accuracy"], "validation_nll": endpoint["nll"],
        "mean_stage_accuracy": math.fsum(row["evaluation"]["accuracy"] for row in result["rows"]) / 50,
        "image_presentations": result["image_presentations"], "optimizer_steps": result["optimizer_steps"],
        "training_wall_seconds": math.fsum(row["fit"]["wall_seconds"] for row in result["rows"]),
        "validation_forward_images": sum(row["evaluation"]["examples"] for row in result["rows"]),
        "validation_wall_seconds": math.fsum(row["evaluation"]["wall_seconds"] for row in result["rows"]),
    })


def _progress(inputs: TuningInputs, phase: str, current: dict[str, object] | None = None) -> None:
    current = current or {}
    paths = tuple(path for directory in ("calibration", "final") for path in (inputs.run / directory).glob("*/result.json"))
    results = tuple(read_sealed(path) for path in paths if path.parent.name != current.get("job"))
    completed = sum(row["image_presentations"] for row in results) + int(current.get("image_presentations", 0))
    stage_rows = tuple(row for result in results for row in result["rows"])
    if current.get("job"):
        stage_rows += tuple(read_sealed(path) for directory in ("calibration", "final")
                            for path in (inputs.run / directory / str(current["job"]) / "stages").glob("*/result.json"))
    fitting_counts = Counter(row.task_index for row in inputs.fitting)
    full_counts = Counter(row.task_index for row in inputs.base.manifest.select("train"))
    planned = (inputs.config.maximum_candidates * followup_presentations(inputs.config.capacity, tuple(fitting_counts[stage] for stage in range(50)))
               + 2 * followup_presentations(inputs.config.capacity, tuple(full_counts[stage] for stage in range(50))))
    measured_images = sum(row["fit"]["image_presentations"] for row in stage_rows)
    measured_seconds = math.fsum(row["fit"]["wall_seconds"] + row["evaluation"]["wall_seconds"] for row in stage_rows)
    eta = max(0, planned - completed) * measured_seconds / measured_images if measured_images else None
    record = {"schema_version": "imagenetr50-srt-tuning-status-v1", "run_hash": inputs.run.name,
              "phase": phase, "updated_utc": datetime.now(timezone.utc).isoformat(),
              "completed_presentations": completed, "planned_maximum_presentations": planned,
              "overall_eta_seconds": 0 if phase == "complete" else eta, "eta_is_upper_bound": True, **current}
    atomic_write(inputs.run / "status.json", canonical_json_bytes(record))
    if not current or current.get("stage_presentations") == current.get("stage_budget"):
        eta_text = f"{eta / 60:.1f} min (upper-bound search plan)" if eta is not None else "measuring"
        print(f"Tuning phase: {phase}; {completed:,}/{planned:,} maximum presentations; ETA {eta_text}", flush=True)


def run_search_phase(
    inputs: TuningInputs, phase: str, candidates: tuple[SRTCandidate, ...], known: tuple[dict[str, object], ...],
    run_job: Callable[..., dict[str, object]], common: dict[str, object],
) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    """Run complete unseen validation recipes and seal one deterministic phase decision."""
    rows = known
    for index, candidate in enumerate(candidates, 1):
        if candidate.content_hash in {row["candidate_hash"] for row in rows}:
            continue
        root = inputs.run / "calibration" / candidate.name
        publish_immutable_json(root / "candidate.json", sealed_record(asdict(candidate)))
        description = f"{phase} {index}/{len(candidates)}: {candidate.policy.name}; LoRA/head LR {candidate.lora_learning_rate:g}/{candidate.head_learning_rate:g}"
        _progress(inputs, description)
        summary_path = root / "candidate_result.json"
        previous = read_sealed(summary_path) if summary_path.is_file() else None
        if previous is not None and previous["status"] == "failed":
            summary = previous
        else:
            started = time.monotonic()
            try:
                run_job(root, config=candidate.optimizer_config(inputs.base.training), policy=candidate.policy,
                        method="srt", training_rows=inputs.fitting, evaluation_rows=inputs.validation,
                        progress_callback=partial(_progress, inputs, description), **common)
                summary = candidate_summary(inputs, candidate)
            except FloatingPointError as error:
                partial_result = read_sealed(root / "result.json") if (root / "result.json").is_file() else None
                summary = sealed_record({
                    "schema_version": "imagenetr50-srt-candidate-summary-v1", "candidate": asdict(candidate),
                    "candidate_hash": candidate.content_hash, "job": candidate.name, "status": "failed",
                    "evaluation_role": "validation", "validation_ids_hash": inputs.protocol["validation_ids_hash"],
                    "completed_stages": len(partial_result["rows"]) if partial_result else 0,
                    "committed_image_presentations": partial_result["image_presentations"] if partial_result else 0,
                    "committed_optimizer_steps": partial_result["optimizer_steps"] if partial_result else 0,
                    "failed_invocation_wall_seconds": time.monotonic() - started,
                    "exception": type(error).__name__, "message": str(error),
                    "work_note": "committed counts are a lower bound; a failed in-flight batch may not have a checkpoint",
                })
                print(f"Recorded numerical candidate failure: {candidate.name}: {error}", flush=True)
            publish_immutable_json(summary_path, summary)
        summary = read_sealed(summary_path)
        rows += (summary,)
        atomic_write(inputs.run / "calibration/candidates.json", canonical_json_bytes(sealed_record({
            "schema_version": "imagenetr50-srt-candidate-ledger-v1", "candidates": rows})))
    winner = select_winner(rows)
    decision = sealed_record({"schema_version": "imagenetr50-srt-tuning-phase-v1", "phase": phase,
                              "requested_candidates": [candidate.content_hash for candidate in candidates],
                              "completed_or_failed_candidates": [row["candidate_hash"] for row in rows],
                              "candidate_summary_hashes": [row["content_hash"] for row in rows],
                              "selected": winner, "test_used": False})
    publish_immutable_json(inputs.run / f"calibration/{phase}_selection.json", decision)
    return rows, winner


def run_tuning(config_path: Path = DEFAULT_CONFIG) -> Path:
    """Run the frozen full-horizon search, refit its selected pair and update the existing report."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from apm.continual.vision.imagenetr.srt_data import SelectedImageLoader
    from apm.continual.vision.imagenetr.srt_training import run_training_job

    torch.set_num_threads(4)
    torch.manual_seed(1993)
    torch.use_deterministic_algorithms(True)
    inputs = bootstrap_tuning(config_path)
    print(f"H=128 tuning working artifacts: {inputs.run}", flush=True)
    lock = (inputs.base.source / "runner.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("H=128 tuning requires the local BF16 GPU outside the sandbox")
        if not (inputs.run / "preflight.json").is_file():
            _progress(inputs, "authenticate data and validation-only membership")
            validate_prepared_dataset(inputs.base.project / "data/imagenetr50/imagenet-r", inputs.base.manifest)
            publish_immutable_json(inputs.run / "preflight.json", sealed_record({
                "schema_version": "imagenetr50-srt-tuning-preflight-v1", "dataset_hash": inputs.base.manifest.content_hash,
                "fit_examples": len(inputs.fitting), "validation_examples": len(inputs.validation), "test_overlap": 0,
                "real_model_evidence": inputs.protocol["baseline_result_hash"], "source_training_code_unchanged": True}))
        with SelectedImageLoader(inputs.base.project / "data/imagenetr50/imagenet-r", inputs.base.training.seed + 50_000,
                                 inputs.base.training.num_workers) as loader:
            common = dict(protocol_hash=inputs.run.name, checkpoint_path=inputs.base.checkpoint, capacity=inputs.config.capacity,
                          loader=loader, device=torch.device("cuda:0"), target_stage=50)
            rows, winner = run_search_phase(inputs, "policy", policy_candidates(inputs.config, inputs.base.training), (), run_training_job, common)
            candidates = rate_candidates(inputs.config, inputs.base.training, candidate_from_record(winner["candidate"]))
            rows, winner = run_search_phase(inputs, "rates", candidates, rows, run_training_job, common)
            candidates = refinement_candidates(inputs.config, candidate_from_record(winner["candidate"]))
            rows, winner = run_search_phase(inputs, "refinement", candidates, rows, run_training_job, common)
            selection = sealed_record({"schema_version": "imagenetr50-srt-tuning-selection-v1", "protocol_hash": inputs.run.name,
                                       "selected": winner, "candidates": rows, "test_used": False,
                                       "objective": inputs.protocol["selection"]})
            publish_immutable_json(inputs.run / "selection.json", selection)
            chosen = candidate_from_record(winner["candidate"])
            baseline_candidate = SRTCandidate(inputs.base.config.policies(inputs.base.training)[0],
                                              inputs.base.training.lora_learning_rate, inputs.base.training.head_learning_rate)
            baseline_reused = chosen == baseline_candidate
            roots = {f"{method}_h128_{'standard_rho80_unit8' if baseline_reused else 'tuned'}":
                     (inputs.base.run if baseline_reused else inputs.run) / "final" /
                     f"{method}_h128_{'standard_rho80_unit8' if baseline_reused else 'tuned'}" for method in ("srt", "uniform")}
            final_common = {**common, "protocol_hash": inputs.base.run.name if baseline_reused else inputs.run.name,
                            "config": chosen.optimizer_config(inputs.base.training), "policy": chosen.policy,
                            "training_rows": inputs.base.manifest.select("train"), "evaluation_rows": inputs.base.manifest.select("test")}
            for name, root in roots.items():
                method = name.split("_", 1)[0]
                phase = f"selected full-data {name}" + (" (reuse baseline)" if baseline_reused else "")
                _progress(inputs, phase)
                run_training_job(root, method=method,
                                 paired_root=roots[name.replace("uniform_", "srt_", 1)] if method == "uniform" else None,
                                 progress_callback=partial(_progress, inputs, phase), **final_common)
            reuse = {}
            for row in rows:
                if row["status"] != "complete":
                    continue
                candidate = candidate_from_record(row["candidate"])
                repeated = run_training_job(inputs.run / "calibration" / candidate.name, method="srt",
                                            policy=candidate.policy, config=candidate.optimizer_config(inputs.base.training),
                                            training_rows=inputs.fitting, evaluation_rows=inputs.validation, **common)
                reuse[candidate.name] = {"optimizer_steps": repeated["invocation_optimizer_steps"], "result_hash": repeated["content_hash"]}
                if repeated["content_hash"] != row["result_hash"]:
                    raise ValueError("candidate result changed during completed-job reuse")
            for name, root in roots.items():
                repeated = run_training_job(root, method=name.split("_", 1)[0],
                                            paired_root=roots[name.replace("uniform_", "srt_", 1)] if name.startswith("uniform_") else None,
                                            **final_common)
                reuse[name] = {"optimizer_steps": repeated["invocation_optimizer_steps"], "result_hash": repeated["content_hash"]}
            if any(row["optimizer_steps"] for row in reuse.values()):
                raise ValueError("a completed tuning job executed additional optimizer steps")
            if read_sealed(inputs.base.source / "result.json")["content_hash"] != inputs.base.config.source_result_hash:
                raise ValueError("the original SRT result changed during tuning")
            result = sealed_record({
                "schema_version": "imagenetr50-srt-tuning-result-v1", "protocol_hash": inputs.run.name,
                "source_result_hash": inputs.base.config.source_result_hash, "selection_hash": selection["content_hash"],
                "baseline_reused": baseline_reused, "source_unchanged": True, "zero_step_reuse": True, "reuse": reuse,
                "condition_roots": {name: root.relative_to(inputs.base.source).as_posix() for name, root in roots.items()},
                "conditions": {name: read_sealed(root / "result.json") for name, root in roots.items()},
            })
            publish_immutable_json(inputs.run / "result.json", result)
        publish_immutable_json(inputs.base.source / "reports/srt_h128_tuning.json", sealed_record({
            "schema_version": "imagenetr50-srt-tuning-pointer-v1", "run_hash": inputs.run.name,
            "result_hash": result["content_hash"], "source_result_hash": inputs.base.config.source_result_hash}))
        _progress(inputs, "audit and rebuild report")
        from apm.continual.vision.imagenetr.srt_reporting import write_srt_report
        report = write_srt_report(inputs.base.source)
        _progress(inputs, "complete")
        return report
    except BaseException as error:
        atomic_write(inputs.run / "failure.json", canonical_json_bytes({"exception": type(error).__name__, "message": str(error),
                                                                       "updated_utc": datetime.now(timezone.utc).isoformat(), "resumable": True}))
        raise
    finally:
        lock.close()


def main() -> None:
    """Run the frozen tuning workflow or inspect status/report without scientific CLI overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("run", "status", "report"), default="run")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    if args.command == "run":
        print(run_tuning(args.config))
        return
    from apm.continual.vision.imagenetr.srt_followup import load_followup_config
    project = args.config.resolve().parents[3]
    config = load_config(args.config)
    source = project / load_followup_config(project / config.baseline_config).source_run
    if args.command == "status":
        paths = tuple(path.parent / "status.json" for path in (source / "tuning").glob("*/protocol.json")
                      if canonical_json_bytes(read_sealed(path)["config"]) == canonical_json_bytes(asdict(config))
                      and (path.parent / "status.json").is_file())
        print(max(paths, key=lambda path: path.stat().st_mtime_ns).read_text() if paths else '{"phase":"not started"}')
    else:
        from apm.continual.vision.imagenetr.srt_reporting import write_srt_report
        print(write_srt_report(source))


if __name__ == "__main__":
    main()

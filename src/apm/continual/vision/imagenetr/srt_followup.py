"""Fixed-policy SRT reruns attached to the existing report without reselection."""

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

import yaml

from apm.continual.artifacts import atomic_write, canonical_json_bytes, file_sha256, publish_immutable_json
from apm.continual.vision.imagenetr.constants import TIMM_MODEL_SHA256
from apm.continual.vision.imagenetr.data import DatasetManifest, load_dataset_manifest
from apm.continual.vision.imagenetr.manifests import installed_environment_manifest
from apm.continual.vision.imagenetr.protocol import material_tree_manifest
from apm.continual.vision.imagenetr.srt_config import SRTConfig, load_srt_config, presentation_budget
from apm.continual.vision.imagenetr.srt_evidence import read_sealed, sealed_record
from apm.continual.vision.imagenetr.srt_scheduler import RecallPolicy


DEFAULT_FOLLOWUP_CONFIG = Path("configs/vision/imagenetr/srt_h4096_rho80_unit8.yaml")


@dataclass(frozen=True, slots=True)
class SRTFollowupConfig:
    """User-fixed policies, not candidates for another validation or test search."""

    source_run: str
    source_result_hash: str
    capacity: int
    historical_fraction: float
    interval_unit: int
    profiles: tuple[str, ...]

    def policies(self, training: SRTConfig) -> tuple[RecallPolicy, ...]:
        """Resolve named thresholds from the unchanged source training config."""
        profiles = dict(training.profiles)
        if (self.capacity != 4096 or self.historical_fraction != .8 or self.interval_unit != 8
                or self.profiles != ("standard", "strict")):
            raise ValueError("follow-up differs from the requested H4096/rho80/unit8 matrix")
        return tuple(RecallPolicy(name, profiles[name], self.historical_fraction, self.interval_unit)
                     for name in self.profiles)


@dataclass(frozen=True, slots=True)
class FollowupInputs:
    """Authenticated source experiment and isolated append-only follow-up state."""

    project: Path
    source: Path
    run: Path
    config: SRTFollowupConfig
    training: SRTConfig
    manifest: DatasetManifest
    checkpoint: Path
    protocol: dict[str, object]


def load_followup_config(path: Path = DEFAULT_FOLLOWUP_CONFIG) -> SRTFollowupConfig:
    """Load the single explicit follow-up matrix from YAML."""
    values = yaml.safe_load(path.read_text())
    return SRTFollowupConfig(**{**values, "profiles": tuple(values["profiles"])})


def followup_conditions(config: SRTFollowupConfig, training: SRTConfig) -> tuple[tuple[str, str, RecallPolicy], ...]:
    """Keep method, profile, mixture, and spacing explicit in each stable name."""
    return tuple((f"{method}_h{config.capacity}_{policy.name}", method, policy)
                 for policy in config.policies(training) for method in ("srt", "uniform"))


def bootstrap_followup(config_path: Path = DEFAULT_FOLLOWUP_CONFIG) -> FollowupInputs:
    """Freeze new provenance while requiring unchanged source code and populations."""
    project = config_path.resolve().parents[3]
    config = load_followup_config(config_path)
    source = project / config.source_run
    original = read_sealed(source / "result.json", "imagenetr50-srt-result-v1")
    if (original["content_hash"] != config.source_result_hash or not original["zero_step_reuse"]
            or not original["references_unchanged"]):
        raise ValueError("the requested completed SRT source changed")
    training = load_srt_config(project / "configs/vision/imagenetr/srt_r16_v1.yaml")
    source_protocol = read_sealed(source / "protocol.json", "imagenetr50-srt-protocol-v1")
    if training.content_hash != source_protocol["config_hash"]:
        raise ValueError("source optimization settings changed")
    original_code = read_sealed(source / "code_manifest.json")
    for row in original_code["files"]:
        if file_sha256(project / row["path"]) != row["sha256"]:
            raise ValueError(f"source training material changed: {row['path']}")
    manifest = load_dataset_manifest(project / "data/imagenetr50/imagenet-r/dataset_manifest.json")
    if manifest.content_hash != source_protocol["dataset_hash"]:
        raise ValueError("follow-up dataset differs from its source")
    environment = installed_environment_manifest(tuple(row["name"] for row in read_sealed(source / "environment_manifest.json")["packages"]))
    if environment["content_hash"] != source_protocol["environment_hash"]:
        raise ValueError("follow-up training environment differs from its source")
    material = material_tree_manifest((Path(__file__), config_path.resolve()))
    protocol = sealed_record({
        "schema_version": "imagenetr50-srt-followup-protocol-v1", "config": asdict(config),
        "source_result_hash": original["content_hash"], "source_protocol_hash": source_protocol["content_hash"],
        "source_code_hash": original_code["content_hash"], "environment_hash": environment["content_hash"],
        "code_hash": material["content_hash"], "dataset_hash": manifest.content_hash,
        "model_sha256": TIMM_MODEL_SHA256, "policies": [asdict(policy) for policy in config.policies(training)],
        "conditions": [name for name, _, _ in followup_conditions(config, training)],
        "selection": "user-fixed after examining the earlier test results; exploratory follow-up, no new search",
        "initialization": "each full-data stream restarts cold; carry adapter, head, and SGD state within the stream",
        "uniform_pairing": "copy the corresponding SRT stream's exact per-update old/current counts and batch sizes",
    })
    run = source / "followups" / protocol["content_hash"]
    for name, record in (("protocol.json", protocol), ("code_manifest.json", material), ("environment_manifest.json", environment)):
        publish_immutable_json(run / name, record)
    candidates = tuple((project / "data/imagenetr50/model_cache").rglob(TIMM_MODEL_SHA256))
    if len(candidates) != 1 or file_sha256(candidates[0]) != TIMM_MODEL_SHA256:
        raise ValueError("pinned backbone failed authentication")
    atomic_write(source / "followups/LATEST_RUN.json", canonical_json_bytes({"run_hash": protocol["content_hash"]}))
    return FollowupInputs(project, source, run, config, training, manifest, candidates[0], protocol)


def _progress(inputs: FollowupInputs, phase: str, current: dict[str, object] | None = None) -> None:
    current = current or {}
    results = tuple(read_sealed(path) for path in (inputs.run / "final").glob("*/result.json") if path.parent.name != current.get("job"))
    completed = sum(result["image_presentations"] for result in results) + int(current.get("image_presentations", 0))
    counts = Counter(row.task_index for row in inputs.manifest.select("train"))
    planned = 4 * sum(presentation_budget(counts[stage], sum(counts[old] for old in range(stage)), inputs.config.capacity)
                      for stage in range(50))
    stage_rows = tuple(read_sealed(path) for path in (inputs.run / "final").glob("*/stages/*/result.json"))
    measured_images = sum(row["fit"]["image_presentations"] for row in stage_rows)
    measured_seconds = math.fsum(row["fit"]["wall_seconds"] + row["evaluation"]["wall_seconds"] for row in stage_rows)
    eta = (planned - completed) * measured_seconds / measured_images if measured_images else None
    record = {"phase": phase, "updated_utc": datetime.now(timezone.utc).isoformat(), "run_hash": inputs.run.name,
              "completed_presentations": completed, "planned_presentations": planned, "overall_eta_seconds": eta, **current}
    atomic_write(inputs.run / "status.json", canonical_json_bytes(record))
    if not current or current.get("stage_presentations") == current.get("stage_budget"):
        eta_text = f"{eta / 60:.1f} min" if eta is not None else "measuring"
        print(f"Overall {completed:,}/{planned:,} presentations; ETA {eta_text}. {phase}", flush=True)


def run_followup(config_path: Path = DEFAULT_FOLLOWUP_CONFIG) -> Path:
    """Run the four resumable streams, prove reuse, then update the existing report."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from apm.continual.vision.imagenetr.srt_data import SelectedImageLoader
    from apm.continual.vision.imagenetr.srt_training import run_training_job

    torch.set_num_threads(4)
    torch.manual_seed(1993)
    torch.use_deterministic_algorithms(True)
    inputs = bootstrap_followup(config_path)
    print(f"Follow-up working artifacts: {inputs.run}", flush=True)
    lock = (inputs.source / "runner.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("the follow-up requires the local BF16 GPU outside the sandbox")
        with SelectedImageLoader(inputs.project / "data/imagenetr50/imagenet-r", inputs.training.seed + 50_000,
                                 inputs.training.num_workers) as loader:
            common = dict(protocol_hash=inputs.protocol["content_hash"], checkpoint_path=inputs.checkpoint,
                          config=inputs.training, loader=loader, device=torch.device("cuda:0"), capacity=inputs.config.capacity,
                          training_rows=inputs.manifest.select("train"), evaluation_rows=inputs.manifest.select("test"), target_stage=50)
            jobs = followup_conditions(inputs.config, inputs.training)
            for name, method, policy in jobs:
                phase = f"{name}, tasks 1-50"
                _progress(inputs, phase)
                run_training_job(inputs.run / "final" / name, method=method, policy=policy,
                                 paired_root=inputs.run / "final" / name.replace("uniform_", "srt_", 1) if method == "uniform" else None,
                                 progress_callback=lambda row, phase=phase: _progress(inputs, phase, row), **common)
            results = {name: read_sealed(inputs.run / "final" / name / "result.json") for name, _, _ in jobs}
            if any(len(result["rows"]) != 50 or result["image_presentations"] != 844640 for result in results.values()):
                raise ValueError("fixed-policy matrix has incomplete or unexpected training work")
            reuse = {}
            for name, method, policy in jobs:
                repeated = run_training_job(inputs.run / "final" / name, method=method, policy=policy,
                                            paired_root=inputs.run / "final" / name.replace("uniform_", "srt_", 1) if method == "uniform" else None,
                                            **common)
                reuse[name] = {"optimizer_steps": repeated["invocation_optimizer_steps"], "result_hash": repeated["content_hash"]}
            if any(row["optimizer_steps"] or row["result_hash"] != results[name]["content_hash"] for name, row in reuse.items()):
                raise ValueError("follow-up reuse changed completed training")
            if read_sealed(inputs.source / "result.json")["content_hash"] != inputs.config.source_result_hash:
                raise ValueError("the original result was modified during the follow-up")
            result = sealed_record({"schema_version": "imagenetr50-srt-followup-result-v1", "protocol_hash": inputs.run.name,
                                    "source_result_hash": inputs.config.source_result_hash, "conditions": results,
                                    "reuse": reuse, "source_unchanged": True, "zero_step_reuse": True})
            publish_immutable_json(inputs.run / "result.json", result)
        pointer = sealed_record({"schema_version": "imagenetr50-srt-followup-pointer-v1", "run_hash": inputs.run.name,
                                 "result_hash": result["content_hash"], "source_result_hash": inputs.config.source_result_hash})
        publish_immutable_json(inputs.source / "reports/fixed_policy_followup.json", pointer)
        _progress(inputs, "build and verify updated report")
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
    """Run the configured follow-up or inspect status without loading a model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("run", "status", "report"), default="run")
    parser.add_argument("--config", type=Path, default=DEFAULT_FOLLOWUP_CONFIG)
    args = parser.parse_args()
    if args.command == "run":
        print(run_followup(args.config))
        return
    source = args.config.resolve().parents[3] / load_followup_config(args.config).source_run
    if args.command == "status":
        latest = json.loads((source / "followups/LATEST_RUN.json").read_text())
        print((source / "followups" / latest["run_hash"] / "status.json").read_text())
    else:
        from apm.continual.vision.imagenetr.srt_reporting import write_srt_report
        print(write_srt_report(source))


if __name__ == "__main__":
    main()

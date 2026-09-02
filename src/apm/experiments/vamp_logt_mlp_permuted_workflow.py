"""Phase orchestration for the dense Permuted-MNIST LogT experiment."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from apm.continual.artifacts import (
    atomic_write,
    canonical_json_bytes,
    file_sha256,
    publish_immutable_json,
)
from apm.experiments.vamp_logt_mlp_permuted_calibration import run_calibration
from apm.experiments.vamp_logt_mlp_permuted_ceiling import (
    run_baseline_extension,
    run_ceiling,
)
from apm.experiments.vamp_logt_mlp_permuted_config import VampLogTDenseConfig
from apm.experiments.vamp_logt_mlp_permuted_data import resolved_device, source_manifest
from apm.experiments.vamp_logt_mlp_permuted_hierarchy import build_hierarchy_tape
from apm.experiments.vamp_logt_mlp_permuted_online import run_online
from apm.experiments.vamp_logt_mlp_permuted_reporting import write_results


MATERIAL_SOURCES = (
    "src/apm/continual/dense_mlp_adapter.py",
    "src/apm/continual/logt_behavioral_integrator.py",
    "src/apm/continual/logt_behavioral_router.py",
    "src/apm/continual/logt_evidence_bank.py",
    "src/apm/experiments/vamp_logt_mlp_permuted_calibration.py",
    "src/apm/experiments/vamp_logt_mlp_permuted_ceiling.py",
    "src/apm/experiments/vamp_logt_mlp_permuted_config.py",
    "src/apm/experiments/vamp_logt_mlp_permuted_data.py",
    "src/apm/experiments/vamp_logt_mlp_permuted_hierarchy.py",
    "src/apm/experiments/vamp_logt_mlp_permuted_online.py",
    "src/apm/experiments/vamp_logt_mlp_permuted_reporting.py",
    "src/apm/experiments/vamp_logt_mlp_permuted_workflow.py",
    "docs/logt_vamp_permuted_mnist_dense_mlp_protocol.md",
    "docs/logt_vamp_permuted_mnist_dense_mlp_ungated_successor.md",
)


def run_workflow(config: VampLogTDenseConfig, selected_phase: str = "all") -> Path:
    """Run or resume the requested phase and its data/model prerequisites."""
    if selected_phase not in {
        "calibration",
        "hierarchy",
        "online",
        "ceiling",
        "baselines",
        "all",
    }:
        raise ValueError(
            "phase must be calibration, hierarchy, online, ceiling, baselines, or all"
        )
    device = resolved_device(config.runtime.device)
    if config.runtime.deterministic_algorithms:
        torch.use_deterministic_algorithms(True)
    run_root = config.artifact_root / "runs" / config.config_hash
    run_root.mkdir(parents=True, exist_ok=True)
    if selected_phase == "baselines":
        print(f"Dense Permuted-MNIST run: {run_root}", flush=True)
        print("Post-hoc extension — seed-zero cumulative baselines", flush=True)
        run_baseline_extension(config, run_root, device)
        from apm.experiments.vamp_logt_mlp_permuted_amended_reporting import (
            write_amended_results,
        )

        write_amended_results(run_root, config)
        return run_root
    _write_protocol(config, run_root)
    print(f"Dense Permuted-MNIST run: {run_root}", flush=True)
    print("Phase 1/4 — architecture calibration", flush=True)
    calibration = run_calibration(config, run_root, device)
    if calibration["status"] != "complete":
        write_results(run_root, config)
        raise RuntimeError("no dense width satisfied the preregistered validation eligibility rule")
    if selected_phase == "calibration":
        write_results(run_root, config)
        return run_root
    print("Phase 2/4 — immutable LogT hierarchy tape", flush=True)
    build_hierarchy_tape(config, run_root, device)
    if selected_phase == "hierarchy":
        write_results(run_root, config)
        return run_root
    if selected_phase in {"online", "ceiling", "all"}:
        print("Phase 3/4 — matched online routers and integrators", flush=True)
        run_online(config, run_root, device)
    if selected_phase in {"ceiling", "all"}:
        print("Phase 4/4 — converged full-replay ceiling", flush=True)
        run_ceiling(config, run_root, device)
    write_results(run_root, config)
    atomic_write(
        config.artifact_root / "LATEST_RUN.json",
        canonical_json_bytes(
            {
                "config_hash": config.config_hash,
                "run_root": str(run_root),
                "schema_version": "vamp-logt-dense-latest-v1",
            }
        ),
    )
    return run_root


def _write_protocol(config: VampLogTDenseConfig, run_root: Path) -> None:
    project_root = Path(__file__).resolve().parents[3]
    missing = tuple(path for path in MATERIAL_SOURCES if not (project_root / path).is_file())
    if missing:
        raise FileNotFoundError(f"dense protocol material sources are missing: {missing}")
    publish_immutable_json(
        run_root / "protocol.json",
        {
            "config": config.as_record(),
            "config_hash": config.config_hash,
            "implementation_sha256": {
                path: file_sha256(project_root / path) for path in MATERIAL_SOURCES
            },
            "pytorch_version": torch.__version__,
            "schema_version": "vamp-logt-dense-protocol-v1",
            "source": source_manifest(config),
        },
    )


__all__ = ["MATERIAL_SOURCES", "run_workflow"]

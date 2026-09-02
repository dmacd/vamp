"""Training-work scaling study for dense LogT integration on 100 permutations."""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping, Sequence
import csv
from dataclasses import asdict, dataclass
from functools import lru_cache
from io import StringIO
from itertools import accumulate
import math
import os
from pathlib import Path
from statistics import fmean, stdev
from time import perf_counter

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from apm.continual.artifacts import (
    ChainedJsonlLedger,
    atomic_write,
    canonical_json_bytes,
    file_sha256,
    load_canonical_json,
    publish_immutable_json,
    record_sha256,
)
from apm.continual.dense_mlp_adapter import (
    DenseExamples,
    DenseMlpState,
    DenseMnistMLP,
    dense_metrics,
    fit_dense_model,
    load_dense_state,
)
from apm.continual.logt_behavioral_integrator import (
    IntegratorConditionState,
    IntegratorObservations,
    IntegratorSupervision,
    create_condition_state,
    named_seed as integrator_named_seed,
    prediction_logits,
    train_condition,
)
from apm.continual.logt_behavioral_router import sample_example_balanced
from apm.continual.vision.imagenetr.checkpoints import atomic_torch_save
from apm.experiments.vamp_logt_mlp_permuted_calibration import (
    initialize_dense_state,
    load_calibrated_base,
)
from apm.experiments.vamp_logt_mlp_permuted_config import (
    SAMPLE_CALIBRATION_REVISION,
    VampLogTDenseConfig,
    load_config,
)
from apm.experiments.vamp_logt_mlp_permuted_data import (
    ExampleBatch,
    PermutedMnistBenchmark,
    build_benchmark,
    concatenate_batches,
    named_seed,
    resolved_device,
    source_manifest,
    stratified_source_split,
)
from apm.experiments.vamp_logt_mlp_permuted_hierarchy import (
    DenseFrontier,
    build_dense_observations,
    build_hierarchy_tape,
    load_frontier,
)
from apm.experiments.vamp_logt_router_reporting import _html


DEFAULT_CONFIG = Path("configs/vamp_logt_mlp_permuted_100_scaling.yaml")
CAPACITY_CONFIG = Path("configs/vamp_logt_mlp_permuted_100_capacity.yaml")
CAPACITY_REVISION = "dense-full-model-v5-scaling-capacity"
UNIFORM_CONDITION = "integrator_uniform_replay"
FULL_REPLAY_CONDITION = "full_replay_integrator_20_epochs"
CALIBRATED_FULL_REPLAY_CONDITION = "full_replay_integrator_calibrated_epochs"
CONDITION_LABELS = {
    UNIFORM_CONDITION: "Persistent uniform replay",
    FULL_REPLAY_CONDITION: "Fresh full replay, 20 epochs",
    CALIBRATED_FULL_REPLAY_CONDITION: "Fresh full replay, calibrated epochs",
}
POLICY_NAMES = {
    1: "one_node_per_level",
    2: "two_nodes_per_level",
}
PLOT_STYLES = {
    (1, UNIFORM_CONDITION): ("#0072B2", "o", "-"),
    (1, FULL_REPLAY_CONDITION): ("#CC79A7", "s", "--"),
    (2, UNIFORM_CONDITION): ("#E69F00", "^", "-"),
    (2, FULL_REPLAY_CONDITION): ("#009E73", "D", "--"),
}
REFERENCE_ARM = "reference_model_standard_samples"
CAPACITY_ARM_NAMES = {
    1: "large_model_standard_samples",
    2: "large_model_double_samples",
}
CAPACITY_ARM_LABELS = {
    REFERENCE_ARM: "Reference model, standard samples",
    CAPACITY_ARM_NAMES[1]: "4× parameters, standard samples",
    CAPACITY_ARM_NAMES[2]: "4× parameters, doubled samples",
}
CAPACITY_PLOT_STYLES = {
    (REFERENCE_ARM, UNIFORM_CONDITION): ("#0072B2", "o", "-"),
    (REFERENCE_ARM, FULL_REPLAY_CONDITION): ("#CC79A7", "s", "--"),
    (CAPACITY_ARM_NAMES[1], UNIFORM_CONDITION): ("#E69F00", "^", "-"),
    (CAPACITY_ARM_NAMES[1], FULL_REPLAY_CONDITION): ("#009E73", "D", "--"),
    (CAPACITY_ARM_NAMES[2], UNIFORM_CONDITION): ("#D55E00", "v", "-"),
    (CAPACITY_ARM_NAMES[2], FULL_REPLAY_CONDITION): ("#56B4E9", "P", "--"),
}
REFERENCE_BASE_PARAMETER_COUNT = 2_383_370
LARGE_BASE_PARAMETER_COUNT = 9_541_274
REFERENCE_INTEGRATOR_PARAMETER_COUNT = 4_411_658
LARGE_INTEGRATOR_PARAMETER_COUNT = 17_650_160
FIT_MINIMUM_STEP = 4
MATERIAL_SOURCES = (
    "configs/vamp_logt_mlp_permuted_100_scaling.yaml",
    "docs/logt_vamp_permuted_mnist_100_scaling_protocol.md",
    "docs/logt_vamp_permuted_mnist_100_scaling_five_seed_amendment.md",
    "src/apm/continual/dense_mlp_adapter.py",
    "src/apm/continual/logt_behavioral_integrator.py",
    "src/apm/continual/logt_behavioral_router.py",
    "src/apm/continual/logt_evidence_bank.py",
    "src/apm/experiments/vamp_logt_mlp_permuted_calibration.py",
    "src/apm/experiments/vamp_logt_mlp_permuted_config.py",
    "src/apm/experiments/vamp_logt_mlp_permuted_data.py",
    "src/apm/experiments/vamp_logt_mlp_permuted_hierarchy.py",
    "src/apm/experiments/vamp_logt_mlp_permuted_scaling.py",
    "src/apm/experiments/vamp_logt_router_reporting.py",
)
CAPACITY_MATERIAL_SOURCES = (
    "configs/vamp_logt_mlp_permuted_100_capacity.yaml",
    "docs/logt_vamp_permuted_mnist_100_capacity_protocol.md",
    *MATERIAL_SOURCES[3:],
)
SAMPLE_CALIBRATION_MATERIAL_SOURCES = (
    "configs/vamp_logt_mlp_permuted_100_sample_calibrated.yaml",
    "docs/logt_vamp_permuted_mnist_100_sample_calibrated_protocol.md",
    *MATERIAL_SOURCES[3:],
)


@dataclass(frozen=True, slots=True)
class _TimedFeatures:
    observations: IntegratorObservations
    wall_seconds: float
    forward_example_passes: int
    forward_calls: int


@dataclass(frozen=True, slots=True)
class _TrainingWork:
    state_initialization_wall_seconds: float
    data_preparation_wall_seconds: float
    feature_wall_seconds: float
    optimizer_wall_seconds: float
    feature_forward_example_passes: int
    feature_forward_calls: int
    integrator_forward_example_passes: int
    integrator_backward_example_passes: int
    integrator_forward_calls: int
    integrator_backward_calls: int
    excluded_diagnostic_forward_example_passes: int
    excluded_diagnostic_forward_calls: int
    training_examples: int
    current_examples: int
    historical_examples: int
    epochs: int

    @property
    def total_training_wall_seconds(self) -> float:
        """Return condition setup, feature construction, and optimizer time."""
        return (
            self.state_initialization_wall_seconds
            + self.data_preparation_wall_seconds
            + self.feature_wall_seconds
            + self.optimizer_wall_seconds
        )

    @property
    def total_training_forward_example_passes(self) -> int:
        """Return frozen-node plus optimizing-integrator forward work."""
        return self.feature_forward_example_passes + self.integrator_forward_example_passes

    def as_record(self) -> dict[str, object]:
        """Return JSON-compatible counters with explicit computed totals."""
        return {
            **asdict(self),
            "total_training_forward_example_passes": self.total_training_forward_example_passes,
            "total_training_wall_seconds": self.total_training_wall_seconds,
        }


@dataclass(frozen=True, slots=True)
class _FullReplayFit:
    state: IntegratorConditionState
    work: _TrainingWork
    checkpoint_sha256: str
    feature_storage: str = "in_memory"
    peak_resident_feature_rows: int = 0
    temporary_feature_cache_bytes: int = 0

    def storage_record(self) -> dict[str, object]:
        """Return bounded-memory implementation details for evidence rows."""
        return {
            "feature_storage": self.feature_storage,
            "peak_resident_feature_rows": self.peak_resident_feature_rows,
            "temporary_feature_cache_bytes": self.temporary_feature_cache_bytes,
        }


@dataclass(frozen=True, slots=True)
class _DiskFeatureCache:
    path: Path
    rows: int
    input_dim: int
    record_columns: int
    data_preparation_wall_seconds: float
    feature_wall_seconds: float
    feature_forward_example_passes: int
    feature_forward_calls: int
    peak_resident_feature_rows: int
    bytes: int


def run_experiment(config_path: Path = DEFAULT_CONFIG) -> Path:
    """Run or resume one authenticated 100-domain scaling protocol."""
    config = load_config(config_path)
    if config.scaling is None:
        raise ValueError("the scaling entry point requires the scaling comparison config")
    device = resolved_device(config.runtime.device)
    if config.runtime.deterministic_algorithms:
        torch.use_deterministic_algorithms(True)
    run_root = config.artifact_root / "runs" / config.config_hash
    run_root.mkdir(parents=True, exist_ok=True)
    _write_protocol(config, run_root)
    experiment_started = perf_counter()
    if _is_sample_calibration_study(config):
        print(f"100-permutation sample-calibrated run: {run_root}", flush=True)
        print(f"Working artifact directory: {run_root}", flush=True)
        print("Overall ETA: pending the task-100 full-replay timing probe", flush=True)
        _run_sample_calibrated_experiment(
            config,
            run_root,
            device,
            experiment_started,
        )
    elif _is_capacity_study(config):
        print(f"100-permutation capacity and sample-count run: {run_root}", flush=True)
        print("Phase 1/4 — train or authenticate the fixed 4×-parameter base", flush=True)
        _fit_capacity_base(config, run_root, device)
        print("Phase 2/4 — build or resume the paired one-node hierarchies", flush=True)
        for multiplier in config.scaling.training_sample_multipliers:
            build_hierarchy_tape(
                config,
                run_root,
                device,
                max_nodes_per_level=1,
                hierarchy_root=_arm_root(config, run_root, multiplier) / "hierarchy",
                training_sample_multiplier=multiplier,
            )
        print("Phase 3/4 — train or resume both integrator conditions", flush=True)
        _run_scaling(config, run_root, device)
        print("Phase 4/4 — compare capacity, samples, and empirical growth", flush=True)
        _write_capacity_report(config, run_root)
    else:
        print(f"100-permutation five-seed scaling run: {run_root}", flush=True)
        print("Phase 1/4 — authenticate the calibrated dense base and predecessor", flush=True)
        _import_base_evidence(config, run_root)
        print("Phase 2/4 — build or resume both five-seed hierarchy policies", flush=True)
        for capacity in config.scaling.hierarchy_node_capacities:
            build_hierarchy_tape(
                config,
                run_root,
                device,
                max_nodes_per_level=capacity,
                hierarchy_root=_policy_root(config, run_root, capacity) / "hierarchy",
            )
        print("Phase 3/4 — train or resume both integrator conditions", flush=True)
        _run_scaling(config, run_root, device)
        print("Phase 4/4 — aggregate uncertainty and policy comparisons", flush=True)
        _write_aggregate_report(config, run_root)
    atomic_write(
        config.artifact_root / "LATEST_RUN.json",
        canonical_json_bytes(
            {
                "config_hash": config.config_hash,
                "run_root": str(run_root),
                "schema_version": (
                    "vamp-logt-dense-scaling-latest-v4"
                    if _is_sample_calibration_study(config)
                    else (
                        "vamp-logt-dense-scaling-latest-v3"
                        if _is_capacity_study(config)
                        else "vamp-logt-dense-scaling-latest-v2"
                    )
                ),
            }
        ),
    )
    return run_root


def _write_protocol(config: VampLogTDenseConfig, run_root: Path) -> None:
    project_root = Path(__file__).resolve().parents[3]
    material_sources = (
        SAMPLE_CALIBRATION_MATERIAL_SOURCES
        if _is_sample_calibration_study(config)
        else (
            CAPACITY_MATERIAL_SOURCES
            if _is_capacity_study(config)
            else MATERIAL_SOURCES
        )
    )
    missing = tuple(path for path in material_sources if not (project_root / path).is_file())
    if missing:
        raise FileNotFoundError(f"scaling protocol material sources are missing: {missing}")
    source_root = config.calibration_evidence_run
    if source_root is None or config.scaling is None:
        raise ValueError("scaling protocol requires an authenticated base-evidence run")
    source_base = source_root / "base" / "model.pt"
    source_summary = source_root / "calibration" / "summary.json"
    predecessor = config.scaling.predecessor_run
    predecessor_files = (
        predecessor / "protocol.json",
        predecessor / "summary.json",
        predecessor / "work_metrics.csv",
    )
    if any(not path.is_file() for path in predecessor_files):
        raise FileNotFoundError("the declared single-seed scaling predecessor is incomplete")
    protocol = {
        "base_evidence": {
            "base_checkpoint_sha256": file_sha256(source_base),
            "calibration_summary_sha256": file_sha256(source_summary),
            "source_run": str(source_root),
        },
        "config": config.as_record(),
        "config_hash": config.config_hash,
        "implementation_sha256": {
            path: file_sha256(project_root / path) for path in material_sources
        },
        "predecessor": {
            "run": str(predecessor),
            "sha256": {
                path.name: file_sha256(path) for path in predecessor_files
            },
        },
        "pytorch_version": torch.__version__,
        "schema_version": (
            "vamp-logt-dense-scaling-protocol-v4"
            if _is_sample_calibration_study(config)
            else (
                "vamp-logt-dense-scaling-protocol-v3"
                if _is_capacity_study(config)
                else "vamp-logt-dense-scaling-protocol-v2"
            )
        ),
        "source": source_manifest(config),
    }
    protocol_path = run_root / "protocol.json"
    if not protocol_path.is_file():
        publish_immutable_json(protocol_path, protocol)
        return
    original = load_canonical_json(protocol_path)
    if canonical_json_bytes(original) == canonical_json_bytes(protocol):
        return
    if not _is_sample_calibration_study(config):
        publish_immutable_json(protocol_path, protocol)
        return

    original_core = {
        key: value for key, value in original.items() if key != "implementation_sha256"
    }
    current_core = {
        key: value for key, value in protocol.items() if key != "implementation_sha256"
    }
    original_implementation = original.get("implementation_sha256")
    current_implementation = protocol["implementation_sha256"]
    if (
        canonical_json_bytes(original_core) != canonical_json_bytes(current_core)
        or not isinstance(original_implementation, dict)
    ):
        raise ValueError("OOM recovery cannot change the frozen scientific protocol")
    changed_sources = tuple(
        path
        for path in material_sources
        if original_implementation.get(path) != current_implementation[path]
    )
    permitted_changes = {
        "docs/logt_vamp_permuted_mnist_100_sample_calibrated_protocol.md",
        "src/apm/experiments/vamp_logt_mlp_permuted_scaling.py",
    }
    if not changed_sources or not set(changed_sources) <= permitted_changes:
        raise ValueError(
            "OOM recovery implementation drift exceeds its storage-only amendment"
        )
    oom_amendment_path = run_root / "protocol-amendment-oom-streaming.json"
    oom_amendment = {
        "changed_material_sources": list(changed_sources),
        "config_hash": config.config_hash,
        "implementation_sha256_after": {
            path: current_implementation[path] for path in changed_sources
        },
        "implementation_sha256_before": {
            path: original_implementation[path] for path in changed_sources
        },
        "original_protocol_sha256": file_sha256(protocol_path),
        "replacement_protocol_record_sha256": record_sha256(protocol),
        "retained_completed_phases": [
            "validation-only sample calibration",
            "selected 100-task frozen hierarchy",
        ],
        "schema_version": "vamp-logt-oom-streaming-amendment-v1",
        "scope": (
            "Storage-only correction after two unpublished task-100 OOMs: "
            "write frozen features once to a bounded memory-mapped cache, "
            "preserve seeded minibatch order and optimizer updates, and remove "
            "the temporary cache after checkpoint publication."
        ),
        "status": "active",
        "unpublished_oom_attempts": 2,
    }
    if not oom_amendment_path.is_file():
        publish_immutable_json(oom_amendment_path, oom_amendment)
        return

    stored_oom_amendment = load_canonical_json(oom_amendment_path)
    stored_after = stored_oom_amendment.get("implementation_sha256_after")
    if not isinstance(stored_after, dict):
        raise ValueError("stored OOM amendment lacks implementation hashes")
    implementation_after_oom = dict(original_implementation)
    implementation_after_oom.update(stored_after)
    protocol_after_oom = dict(protocol)
    protocol_after_oom["implementation_sha256"] = implementation_after_oom
    stored_oom_is_valid = (
        stored_oom_amendment.get("config_hash") == config.config_hash
        and stored_oom_amendment.get("schema_version")
        == "vamp-logt-oom-streaming-amendment-v1"
        and stored_oom_amendment.get("original_protocol_sha256")
        == file_sha256(protocol_path)
        and stored_oom_amendment.get("replacement_protocol_record_sha256")
        == record_sha256(protocol_after_oom)
    )
    if not stored_oom_is_valid:
        raise ValueError("stored OOM amendment does not authenticate its predecessor")

    drift_after_oom = tuple(
        path
        for path in material_sources
        if implementation_after_oom.get(path) != current_implementation[path]
    )
    if not drift_after_oom:
        return
    report_source = "src/apm/experiments/vamp_logt_mlp_permuted_scaling.py"
    if drift_after_oom != (report_source,):
        raise ValueError("post-OOM implementation drift is not report-only")
    projection_amendment_path = (
        run_root / "protocol-amendment-report-projection-clarification.json"
    )
    projection_amendment = {
        "changed_material_sources": [report_source],
        "config_hash": config.config_hash,
        "implementation_sha256_after": {
            report_source: current_implementation[report_source]
        },
        "implementation_sha256_before": {
            report_source: implementation_after_oom[report_source]
        },
        "prior_amendment_sha256": file_sha256(oom_amendment_path),
        "prior_protocol_record_sha256": record_sha256(protocol_after_oom),
        "replacement_protocol_record_sha256": record_sha256(protocol),
        "schema_version": "vamp-logt-report-clarification-amendment-v1",
        "scope": (
            "Report-only clarification separating elapsed setup time, "
            "remaining required work, reserve time, and the marginal cost of "
            "the four optional full-replay fits. No training result changed."
        ),
        "scientific_results_changed": False,
        "status": "active",
    }
    if not projection_amendment_path.is_file():
        publish_immutable_json(projection_amendment_path, projection_amendment)
        return

    stored_projection_amendment = load_canonical_json(projection_amendment_path)
    projection_after = stored_projection_amendment.get(
        "implementation_sha256_after"
    )
    if not isinstance(projection_after, dict):
        raise ValueError("stored projection amendment lacks implementation hashes")
    implementation_after_projection = dict(implementation_after_oom)
    implementation_after_projection.update(projection_after)
    protocol_after_projection = dict(protocol)
    protocol_after_projection["implementation_sha256"] = (
        implementation_after_projection
    )
    stored_projection_is_valid = (
        stored_projection_amendment.get("config_hash") == config.config_hash
        and stored_projection_amendment.get("schema_version")
        == "vamp-logt-report-clarification-amendment-v1"
        and stored_projection_amendment.get("prior_amendment_sha256")
        == file_sha256(oom_amendment_path)
        and stored_projection_amendment.get("prior_protocol_record_sha256")
        == record_sha256(protocol_after_oom)
        and stored_projection_amendment.get("replacement_protocol_record_sha256")
        == record_sha256(protocol_after_projection)
        and stored_projection_amendment.get("scientific_results_changed") is False
    )
    if not stored_projection_is_valid:
        raise ValueError(
            "stored projection amendment does not authenticate its predecessor"
        )

    drift_after_projection = tuple(
        path
        for path in material_sources
        if implementation_after_projection.get(path) != current_implementation[path]
    )
    if not drift_after_projection:
        return
    if drift_after_projection != (report_source,):
        raise ValueError("post-projection implementation drift is not report-only")
    publish_immutable_json(
        run_root / "protocol-amendment-report-scaling-format.json",
        {
            "changed_material_sources": [report_source],
            "config_hash": config.config_hash,
            "implementation_sha256_after": {
                report_source: current_implementation[report_source]
            },
            "implementation_sha256_before": {
                report_source: implementation_after_projection[report_source]
            },
            "prior_amendment_sha256": file_sha256(projection_amendment_path),
            "prior_protocol_record_sha256": record_sha256(
                protocol_after_projection
            ),
            "replacement_protocol_record_sha256": record_sha256(protocol),
            "schema_version": "vamp-logt-report-scaling-format-amendment-v1",
            "scope": (
                "Report-only restoration of absolute, cumulative empirical-fit, "
                "and theoretical-normalization views for persistent and fresh "
                "full replay. No training result changed."
            ),
            "scientific_results_changed": False,
            "status": "active",
        },
    )


def _fit_capacity_base(
    config: VampLogTDenseConfig,
    run_root: Path,
    device: torch.device,
) -> dict[str, object]:
    """Fit the predeclared large identity base with validation-only stopping."""
    if not _is_capacity_study(config) or config.scaling is None:
        raise ValueError("large-base fitting belongs only to the capacity study")
    summary_path = run_root / "calibration" / "summary.json"
    if summary_path.is_file():
        summary = load_canonical_json(summary_path)
        base = load_calibrated_base(config, run_root)
        if (
            summary.get("status") != "complete"
            or tuple(summary.get("selected_hidden_widths", ()))
            != config.scaling.base_hidden_widths
            or base.parameter_count != LARGE_BASE_PARAMETER_COUNT
        ):
            raise ValueError("stored large-base calibration changed")
        return summary

    benchmark = build_benchmark(config, 0)
    training_ids, validation_ids = stratified_source_split(
        benchmark.train_labels,
        config.calibration.validation_source_examples,
        config.calibration.split_seed,
    )
    if (
        len(training_ids) != config.calibration.training_source_examples
        or len(validation_ids) != config.calibration.validation_source_examples
        or set(training_ids.tolist()) & set(validation_ids.tolist())
    ):
        raise RuntimeError("large-base calibration split violates its frozen boundary")
    widths = config.scaling.base_hidden_widths
    if widths is None:
        raise ValueError("large-base widths are absent")
    training = DenseExamples(
        benchmark.train_images[training_ids],
        benchmark.train_labels[training_ids],
        benchmark.permutations[:1],
    )
    validation = DenseExamples(
        benchmark.train_images[validation_ids],
        benchmark.train_labels[validation_ids],
        benchmark.permutations[:1],
    )
    initialization_seed = named_seed(
        0,
        "calibration",
        widths,
        "identity-init",
    )
    result = fit_dense_model(
        training,
        initialize_dense_state(widths, config.calibration.dropout, initialization_seed),
        config.calibration.optimizer,
        named_seed(0, "calibration", widths, "identity-fit"),
        device,
        validation=validation,
        convergence=config.calibration.convergence,
        dropout=config.calibration.dropout,
        progress_label="large dense base",
        progress=config.runtime.progress,
    )
    if result.state.parameter_count != LARGE_BASE_PARAMETER_COUNT:
        raise RuntimeError("large base missed its frozen parameter target")
    base_path = run_root / "base" / "model.pt"
    atomic_torch_save(
        base_path,
        {
            "config_hash": config.config_hash,
            "hidden_widths": widths,
            "parameters": result.state.tensors,
            "schema_version": "vamp-logt-dense-base-v1",
            "selection_source": "fixed_architecture_validation_loss",
        },
    )
    model = DenseMnistMLP(widths, config.calibration.dropout).to(device)
    load_dense_state(model, result.state)
    identity_test_loss, identity_test_accuracy = dense_metrics(
        model,
        DenseExamples(
            benchmark.test_images,
            benchmark.test_labels,
            benchmark.permutations[:1],
        ),
        device,
        config.calibration.optimizer.batch_size,
    )
    best = result.history[result.best_epoch - 1]
    split_record = {
        "training_count": len(training_ids),
        "training_ids_sha256": record_sha256(training_ids.tolist()),
        "validation_count": len(validation_ids),
        "validation_ids_sha256": record_sha256(validation_ids.tolist()),
    }
    summary = {
        "base_checkpoint_sha256": file_sha256(base_path),
        "best_epoch": result.best_epoch,
        "config_hash": config.config_hash,
        "epochs_ran": result.epochs_ran,
        "history": [asdict(epoch) for epoch in result.history],
        "identity_test_accuracy": identity_test_accuracy,
        "identity_test_cross_entropy": identity_test_loss,
        "initialization_seed": initialization_seed,
        "parameter_count": result.state.parameter_count,
        "parameter_ratio_to_reference": (
            result.state.parameter_count / REFERENCE_BASE_PARAMETER_COUNT
        ),
        "schema_version": "vamp-logt-dense-capacity-calibration-v1",
        "selected_hidden_widths": list(widths),
        "selection_policy": "fixed_architecture_validation_loss",
        "source_split": split_record,
        "status": "complete",
        "stop_reason": result.stop_reason,
        "validation_accuracy_at_best_epoch": best.validation_accuracy,
        "validation_cross_entropy_at_best_epoch": best.validation_loss,
    }
    publish_immutable_json(
        run_root / "base" / "manifest.json",
        {
            "checkpoint_sha256": file_sha256(base_path),
            "parameter_count": result.state.parameter_count,
            "schema_version": "vamp-logt-dense-base-manifest-v1",
            "selected_hidden_widths": list(widths),
            "selection_uses_test": False,
        },
    )
    publish_immutable_json(summary_path, summary)
    load_calibrated_base(config, run_root)
    return summary


def _import_base_evidence(config: VampLogTDenseConfig, run_root: Path) -> None:
    source_root = config.calibration_evidence_run
    if source_root is None:
        raise ValueError("scaling protocol requires an authenticated base-evidence run")
    source_summary_path = source_root / "calibration" / "summary.json"
    source_base_path = source_root / "base" / "model.pt"
    source_summary = load_canonical_json(source_summary_path)
    source_base_sha256 = file_sha256(source_base_path)
    widths = tuple(int(value) for value in source_summary["selected_hidden_widths"])
    if (
        source_summary.get("status") != "complete"
        or source_summary.get("base_checkpoint_sha256") != source_base_sha256
        or widths != (1024, 1024, 512)
    ):
        raise ValueError("the declared dense-base evidence changed or is incomplete")
    source_payload = torch.load(source_base_path, map_location="cpu", weights_only=True)
    if tuple(source_payload.get("hidden_widths", ())) != widths:
        raise ValueError("the declared dense-base checkpoint architecture changed")
    base_path = run_root / "base" / "model.pt"
    summary_path = run_root / "calibration" / "summary.json"
    if not base_path.is_file():
        atomic_torch_save(
            base_path,
            {
                "config_hash": config.config_hash,
                "hidden_widths": widths,
                "parameters": tuple(source_payload["parameters"]),
                "schema_version": "vamp-logt-dense-base-v1",
                "source_checkpoint_sha256": source_base_sha256,
            },
        )
    local_payload = torch.load(base_path, map_location="cpu", weights_only=True)
    if (
        local_payload.get("config_hash") != config.config_hash
        or tuple(local_payload.get("hidden_widths", ())) != widths
        or local_payload.get("source_checkpoint_sha256") != source_base_sha256
        or len(local_payload.get("parameters", ())) != len(source_payload["parameters"])
        or any(
            not torch.equal(local, source)
            for local, source in zip(
                local_payload["parameters"], source_payload["parameters"], strict=True
            )
        )
    ):
        raise ValueError("the local imported dense base differs from its authenticated source")
    publish_immutable_json(
        summary_path,
        {
            "base_checkpoint_sha256": file_sha256(base_path),
            "config_hash": config.config_hash,
            "selected_hidden_widths": list(widths),
            "selection_policy": "authenticated_seed_zero_identity_import",
            "source_base_checkpoint_sha256": source_base_sha256,
            "source_calibration_summary_sha256": file_sha256(source_summary_path),
            "source_config_hash": source_summary["config_hash"],
            "status": "complete",
        },
    )
    load_calibrated_base(config, run_root)


def _run_sample_calibrated_experiment(
    config: VampLogTDenseConfig,
    run_root: Path,
    device: torch.device,
    experiment_started: float,
) -> None:
    """Run the validation-selected sample arm and its endpoint-first comparison."""
    calibration_config = config.sample_calibration
    if calibration_config is None:
        raise ValueError("sample-calibrated protocol lacks its selection settings")
    print("Phase 1/6 — authenticate the reference dense base", flush=True)
    _import_base_evidence(config, run_root)
    print("Phase 2/6 — calibrate samples and full-replay epochs on tasks 1–10", flush=True)
    selection = _calibrate_sample_budget(config, run_root, device)
    if selection.get("status") != "complete":
        print("Calibration stopped: no declared sample candidate reached 95%", flush=True)
        _write_sample_calibration_stop_report(config, run_root, selection)
        return

    sample_multiplier = int(selection["selected_sample_multiplier"])
    epoch_budget = int(selection["selected_full_replay_epochs"])
    arm_root = _arm_root(config, run_root, sample_multiplier)
    print(
        "Selected "
        f"{config.benchmark.observer_batch_size * sample_multiplier} samples per role "
        f"and {epoch_budget} full-replay epochs.",
        flush=True,
    )
    print("Phase 3/6 — extend the selected hierarchy through task 100", flush=True)
    build_hierarchy_tape(
        config,
        run_root,
        device,
        max_nodes_per_level=1,
        hierarchy_root=arm_root / "hierarchy",
        training_sample_multiplier=sample_multiplier,
    )
    print("Phase 4/6 — fit and evaluate the task-100 full-replay endpoint first", flush=True)
    endpoint = _run_full_replay_endpoint_probe(
        config,
        run_root,
        device,
        sample_multiplier,
        epoch_budget,
    )
    schedule = _resolve_sample_calibrated_schedule(
        config,
        run_root,
        endpoint,
        sample_multiplier,
        epoch_budget,
        perf_counter() - experiment_started,
    )
    print(
        f"Task-100 full replay: {100 * float(endpoint['accuracy']):.2f}% test "
        f"accuracy in {float(endpoint['total_training_wall_seconds']):.1f}s.",
        flush=True,
    )
    print(
        f"Projected total: {float(schedule['projected_total_seconds']):.0f}s; "
        f"intermediate full-replay checkpoints "
        f"{'included' if schedule['optional_checkpoints_included'] else 'omitted'}.",
        flush=True,
    )
    print("Phase 5/6 — train persistent replay and scheduled fresh full replays", flush=True)
    _run_scaling_seed(
        config,
        run_root,
        1,
        0,
        device,
        training_sample_multiplier=sample_multiplier,
        full_replay_checkpoints=tuple(int(value) for value in schedule["checkpoints"]),
        full_replay_condition=CALIBRATED_FULL_REPLAY_CONDITION,
        offline_epochs=epoch_budget,
    )
    print("Phase 6/6 — validate, plot, and write the standalone report", flush=True)
    _write_sample_calibrated_report(config, run_root)


def _calibrate_sample_budget(
    config: VampLogTDenseConfig,
    run_root: Path,
    device: torch.device,
) -> dict[str, object]:
    """Select the least data and earliest epoch satisfying all ten prefixes."""
    calibration_config = config.sample_calibration
    if calibration_config is None:
        raise ValueError("sample calibration settings are absent")
    selection_path = run_root / "calibration" / "sample_selection.json"
    if selection_path.is_file():
        selection = load_canonical_json(selection_path)
        if selection.get("config_hash") != config.config_hash:
            raise ValueError("stored sample selection belongs to another protocol")
        return selection

    calibration_started = perf_counter()
    candidate_summaries: tuple[dict[str, object], ...] = ()
    for multiplier in calibration_config.sample_multiplier_candidates:
        sample_count = config.benchmark.observer_batch_size * multiplier
        arm_root = _arm_root(config, run_root, multiplier)
        print(
            f"Calibration candidate: {sample_count} model + {sample_count} observer "
            "examples per task",
            flush=True,
        )
        build_hierarchy_tape(
            config,
            run_root,
            device,
            max_nodes_per_level=1,
            hierarchy_root=arm_root / "hierarchy",
            training_sample_multiplier=multiplier,
            stop_after_step=calibration_config.prefix_steps,
        )
        candidate = _calibrate_sample_candidate(
            config,
            run_root,
            device,
            multiplier,
        )
        candidate_summaries = (*candidate_summaries, candidate)
        if candidate["status"] == "passing":
            selection = {
                "calibration_wall_seconds": perf_counter() - calibration_started,
                "candidate_summaries": list(candidate_summaries),
                "config_hash": config.config_hash,
                "schema_version": "vamp-logt-sample-selection-v1",
                "selected_full_replay_epochs": candidate["selected_epoch"],
                "selected_model": "reference_1024_1024_512",
                "selected_sample_count_per_role": sample_count,
                "selected_sample_multiplier": multiplier,
                "selection_rule": calibration_config.selection_rule,
                "status": "complete",
                "target_accuracy": calibration_config.target_accuracy,
                "test_evaluations_during_selection": 0,
            }
            publish_immutable_json(selection_path, selection)
            return selection

    selection = {
        "calibration_wall_seconds": perf_counter() - calibration_started,
        "candidate_summaries": list(candidate_summaries),
        "config_hash": config.config_hash,
        "schema_version": "vamp-logt-sample-selection-v1",
        "selected_model": None,
        "selection_rule": calibration_config.selection_rule,
        "status": "failed_no_candidate",
        "target_accuracy": calibration_config.target_accuracy,
        "test_evaluations_during_selection": 0,
    }
    publish_immutable_json(selection_path, selection)
    return selection


def _calibrate_sample_candidate(
    config: VampLogTDenseConfig,
    run_root: Path,
    device: torch.device,
    sample_multiplier: int,
) -> dict[str, object]:
    """Measure all prefix-by-epoch validation accuracies for one sample count."""
    calibration_config = config.sample_calibration
    if calibration_config is None:
        raise ValueError("sample calibration settings are absent")
    directory = run_root / "calibration" / f"samples-{sample_multiplier:02d}x"
    summary_path = directory / "summary.json"
    ledger = ChainedJsonlLedger(
        directory / "metrics.jsonl",
        "vamp-logt-sample-calibration-metric-v1",
    )
    print(f"Calibration log: {ledger.path}", flush=True)
    if summary_path.is_file():
        summary = load_canonical_json(summary_path)
        if (
            summary.get("config_hash") != config.config_hash
            or int(summary.get("sample_multiplier", -1)) != sample_multiplier
            or int(summary.get("metric_rows", -1)) != ledger.next_sequence
        ):
            raise ValueError("stored sample candidate summary changed")
        return summary

    completed_steps = tuple(int(row["macro_step"]) for row in ledger.rows)
    if completed_steps != tuple(range(1, len(completed_steps) + 1)):
        raise ValueError("sample calibration ledger is not a complete prefix")
    benchmark = build_benchmark(
        config,
        0,
        training_sample_multiplier=sample_multiplier,
    )
    base = load_calibrated_base(config, run_root)
    slot_dim = base.embedding_dim + 11
    input_dim = config.observer.maximum_levels * slot_dim
    try:
        from tqdm.auto import tqdm
    except ImportError as error:  # pragma: no cover - vision environment gate
        raise RuntimeError("tqdm is required by the vision environment") from error
    macro_steps = tqdm(
        range(len(completed_steps) + 1, calibration_config.prefix_steps + 1),
        initial=len(completed_steps),
        total=calibration_config.prefix_steps,
        desc=f"full-replay calibration samples={sample_multiplier}x",
        disable=not config.runtime.progress,
        unit="prefix",
    )
    hierarchy_root = _arm_root(config, run_root, sample_multiplier) / "hierarchy"
    for macro_step in macro_steps:
        training_examples = concatenate_batches(tuple(
            benchmark.step(step).observer for step in range(1, macro_step + 1)
        ))
        validation_examples = concatenate_batches(tuple(
            benchmark.step(step).evaluation for step in range(1, macro_step + 1)
        ))
        frontier = load_frontier(
            config,
            run_root,
            0,
            macro_step,
            hierarchy_root=hierarchy_root,
            training_sample_multiplier=sample_multiplier,
        )
        training_features = _timed_features(
            config,
            frontier,
            training_examples,
            base,
            device,
        )
        validation_features = _timed_features(
            config,
            frontier,
            validation_examples,
            base,
            device,
        )
        state, initialization_seconds = _timed_state_creation(
            _full_replay_state_name(
                CALIBRATED_FULL_REPLAY_CONDITION,
                sample_multiplier,
                macro_step,
            ),
            input_dim,
            slot_dim,
            config,
            0,
            1,
            device,
        )
        epoch_metrics: tuple[dict[str, object], ...] = ()
        callback_wall_seconds = 0.0

        def capture_epoch(
            epoch: int,
            epoch_state: IntegratorConditionState,
        ) -> None:
            nonlocal epoch_metrics, callback_wall_seconds
            callback_started = perf_counter()
            metrics = _integrator_metrics(
                epoch_state,
                validation_features.observations,
                validation_examples.labels,
                config,
                device,
            )
            callback_wall_seconds += perf_counter() - callback_started
            epoch_metrics = (*epoch_metrics, {"epoch": epoch, **metrics})

        result = train_condition(
            state,
            IntegratorSupervision(
                training_features.observations,
                training_examples.labels,
            ),
            None,
            config.integrator.offline_epochs,
            config.integrator,
            0,
            macro_step,
            device,
            epoch_callback=capture_epoch,
        )
        ledger.append({
            "calibration_integrator_forward_example_passes": (
                config.integrator.offline_epochs * len(validation_examples.labels)
            ),
            "calibration_validation_wall_seconds": callback_wall_seconds,
            "config_hash": config.config_hash,
            "epoch_metrics": list(epoch_metrics),
            "feature_forward_example_passes": (
                training_features.forward_example_passes
            ),
            "feature_wall_seconds": training_features.wall_seconds,
            "integrator_backward_example_passes": (
                result.training_backward_example_passes
            ),
            "integrator_forward_example_passes": (
                result.training_forward_example_passes
            ),
            "macro_step": macro_step,
            "optimizer_wall_seconds": result.training_wall_seconds,
            "sample_count_per_role": (
                config.benchmark.observer_batch_size * sample_multiplier
            ),
            "sample_multiplier": sample_multiplier,
            "state_initialization_wall_seconds": initialization_seconds,
            "test_evaluations": 0,
            "training_examples": len(training_examples.labels),
            "validation_examples": len(validation_examples.labels),
            "validation_feature_forward_example_passes": (
                validation_features.forward_example_passes
            ),
            "validation_feature_wall_seconds": validation_features.wall_seconds,
        })
        macro_steps.set_postfix(
            best=f"{100 * max(float(row['accuracy']) for row in epoch_metrics):.2f}%"
        )

    selected_epoch, prefix_accuracies, best_epoch, best_minimum = (
        _select_sample_calibration_epoch(
            ledger.rows,
            config.integrator.offline_epochs,
            calibration_config.target_accuracy,
        )
    )
    summary = {
        "best_epoch_by_minimum_prefix_accuracy": best_epoch,
        "best_minimum_prefix_accuracy": best_minimum,
        "config_hash": config.config_hash,
        "metric_rows": ledger.next_sequence,
        "prefix_accuracies_at_selected_epoch": list(prefix_accuracies),
        "sample_count_per_role": (
            config.benchmark.observer_batch_size * sample_multiplier
        ),
        "sample_multiplier": sample_multiplier,
        "schema_version": "vamp-logt-sample-candidate-summary-v1",
        "selected_epoch": selected_epoch,
        "status": "passing" if selected_epoch is not None else "failed_threshold",
        "target_accuracy": calibration_config.target_accuracy,
        "test_evaluations": 0,
    }
    publish_immutable_json(summary_path, summary)
    return summary


def _select_sample_calibration_epoch(
    rows: Sequence[Mapping[str, object]],
    maximum_epochs: int,
    target_accuracy: float,
) -> tuple[int | None, tuple[float, ...], int, float]:
    """Apply the minimum-samples-then-epochs rule to complete prefix rows."""
    if not rows or any(
        len(tuple(row["epoch_metrics"])) != maximum_epochs for row in rows
    ):
        raise ValueError("sample calibration requires every epoch at every prefix")
    accuracy_by_epoch = tuple(
        tuple(
            float(tuple(row["epoch_metrics"])[epoch - 1]["accuracy"])
            for row in rows
        )
        for epoch in range(1, maximum_epochs + 1)
    )
    passing_epochs = tuple(
        epoch
        for epoch, accuracies in enumerate(accuracy_by_epoch, start=1)
        if min(accuracies) >= target_accuracy
    )
    best_epoch = max(
        range(1, maximum_epochs + 1),
        key=lambda epoch: (min(accuracy_by_epoch[epoch - 1]), -epoch),
    )
    selected_epoch = passing_epochs[0] if passing_epochs else None
    selected_accuracies = (
        accuracy_by_epoch[selected_epoch - 1]
        if selected_epoch is not None
        else ()
    )
    return (
        selected_epoch,
        selected_accuracies,
        best_epoch,
        min(accuracy_by_epoch[best_epoch - 1]),
    )


def _integrator_metrics(
    state: IntegratorConditionState,
    observations: IntegratorObservations,
    labels: Tensor,
    config: VampLogTDenseConfig,
    device: torch.device,
) -> dict[str, float]:
    """Measure one integrator on a fixed detached observation matrix."""
    logits = prediction_logits(
        state.integrator,
        observations,
        device,
        config.integrator.minibatch_size,
    )
    return {
        "accuracy": float((logits.argmax(dim=1) == labels).float().mean().item()),
        "cross_entropy": float(F.cross_entropy(logits, labels).item()),
    }


def _run_full_replay_endpoint_probe(
    config: VampLogTDenseConfig,
    run_root: Path,
    device: torch.device,
    sample_multiplier: int,
    epoch_budget: int,
) -> dict[str, object]:
    """Fit task 100 before every other final-condition training operation."""
    path = run_root / "endpoint_probe.json"
    if path.is_file():
        probe = load_canonical_json(path)
        if (
            probe.get("config_hash") != config.config_hash
            or int(probe.get("sample_multiplier", -1)) != sample_multiplier
            or int(probe.get("epochs", -1)) != epoch_budget
        ):
            raise ValueError("stored task-100 endpoint probe changed")
        return probe
    benchmark = build_benchmark(
        config,
        0,
        training_sample_multiplier=sample_multiplier,
    )
    base = load_calibrated_base(config, run_root)
    arm_root = _arm_root(config, run_root, sample_multiplier)
    frontier = load_frontier(
        config,
        run_root,
        0,
        config.benchmark.macro_steps,
        hierarchy_root=arm_root / "hierarchy",
        training_sample_multiplier=sample_multiplier,
    )
    slot_dim = base.embedding_dim + 11
    full_fit = _fit_full_replay(
        config,
        arm_root / "scaling" / "seed-0",
        config.benchmark.macro_steps,
        frontier,
        benchmark,
        base,
        config.observer.maximum_levels * slot_dim,
        slot_dim,
        0,
        1,
        device,
        sample_multiplier,
        condition=CALIBRATED_FULL_REPLAY_CONDITION,
        offline_epochs=epoch_budget,
    )
    evaluation, excluded = _evaluate_conditions(
        config,
        benchmark,
        frontier,
        base,
        {CALIBRATED_FULL_REPLAY_CONDITION: full_fit.state},
        config.benchmark.macro_steps,
        device,
    )
    probe = {
        **evaluation[CALIBRATED_FULL_REPLAY_CONDITION],
        **full_fit.work.as_record(),
        **full_fit.storage_record(),
        **excluded,
        "checkpoint_sha256": full_fit.checkpoint_sha256,
        "config_hash": config.config_hash,
        "epochs": epoch_budget,
        "macro_step": config.benchmark.macro_steps,
        "sample_count_per_role": (
            config.benchmark.observer_batch_size * sample_multiplier
        ),
        "sample_multiplier": sample_multiplier,
        "schema_version": "vamp-logt-full-replay-endpoint-probe-v2",
        "status": "complete",
        "trained_before_persistent_condition": True,
    }
    publish_immutable_json(path, probe)
    return probe


def _resolve_sample_calibrated_schedule(
    config: VampLogTDenseConfig,
    run_root: Path,
    endpoint: Mapping[str, object],
    sample_multiplier: int,
    epoch_budget: int,
    elapsed_seconds: float,
) -> dict[str, object]:
    """Freeze the optional checkpoint decision from the endpoint timing probe."""
    path = run_root / "full_replay_schedule.json"
    if path.is_file():
        schedule = load_canonical_json(path)
        if schedule.get("config_hash") != config.config_hash:
            raise ValueError("stored full-replay schedule belongs to another protocol")
        return schedule
    calibration_config = config.sample_calibration
    if calibration_config is None:
        raise ValueError("sample calibration settings are absent")
    selection_path = run_root / "calibration" / "sample_selection.json"
    hierarchy_summary_path = (
        _arm_root(config, run_root, sample_multiplier) / "hierarchy" / "summary.json"
    )
    selection = load_canonical_json(selection_path)
    calibration_started_at = selection_path.stat().st_mtime - float(
        selection["calibration_wall_seconds"]
    )
    durable_seconds_through_hierarchy = max(
        0.0,
        hierarchy_summary_path.stat().st_mtime - calibration_started_at,
    )
    resolved_elapsed_seconds = max(
        elapsed_seconds,
        durable_seconds_through_hierarchy
        + float(endpoint["total_training_wall_seconds"]),
    )
    mandatory = tuple(
        step
        for step in config.evaluation.full_checkpoints
        if step not in calibration_config.optional_full_checkpoints
    )
    endpoint_passes = int(endpoint["total_training_forward_example_passes"]) + int(
        endpoint["integrator_backward_example_passes"]
    )
    seconds_per_model_pass = (
        float(endpoint["total_training_wall_seconds"]) / endpoint_passes
    )
    remaining_mandatory_passes = sum(
        _full_replay_model_passes(config, step, sample_multiplier, epoch_budget)
        for step in mandatory
        if step != config.benchmark.macro_steps
    )
    optional_passes = sum(
        _full_replay_model_passes(config, step, sample_multiplier, epoch_budget)
        for step in calibration_config.optional_full_checkpoints
    )
    persistent_passes = _persistent_model_passes(config, sample_multiplier)
    safety_factor = 1.25
    reporting_reserve_seconds = 300.0
    projected_required_seconds = (
        resolved_elapsed_seconds
        + safety_factor
        * seconds_per_model_pass
        * (remaining_mandatory_passes + persistent_passes)
        + reporting_reserve_seconds
    )
    projected_with_optional_seconds = (
        projected_required_seconds
        + safety_factor * seconds_per_model_pass * optional_passes
    )
    include_optional = (
        projected_with_optional_seconds
        <= calibration_config.total_time_budget_seconds
    )
    checkpoints = tuple(sorted(
        (*mandatory, *calibration_config.optional_full_checkpoints)
        if include_optional
        else mandatory
    ))
    schedule = {
        "checkpoints": list(checkpoints),
        "config_hash": config.config_hash,
        "current_process_elapsed_seconds_through_endpoint": elapsed_seconds,
        "durable_seconds_through_hierarchy": durable_seconds_through_hierarchy,
        "elapsed_seconds_through_endpoint": resolved_elapsed_seconds,
        "endpoint_first_execution_order": [
            config.benchmark.macro_steps,
            *(step for step in checkpoints if step != config.benchmark.macro_steps),
        ],
        "optional_checkpoints": list(
            calibration_config.optional_full_checkpoints
        ),
        "optional_checkpoints_included": include_optional,
        "projected_required_seconds": projected_required_seconds,
        "projected_with_optional_seconds": projected_with_optional_seconds,
        "projected_total_seconds": (
            projected_with_optional_seconds
            if include_optional
            else projected_required_seconds
        ),
        "reporting_reserve_seconds": reporting_reserve_seconds,
        "safety_factor": safety_factor,
        "schema_version": "vamp-logt-full-replay-schedule-v2",
        "seconds_per_counted_model_pass": seconds_per_model_pass,
        "status": "complete",
        "time_budget_seconds": calibration_config.total_time_budget_seconds,
    }
    publish_immutable_json(path, schedule)
    return schedule


def _full_replay_model_passes(
    config: VampLogTDenseConfig,
    macro_step: int,
    sample_multiplier: int,
    epoch_budget: int,
) -> int:
    """Return frozen forward plus integrator forward/backward example-passes."""
    training_examples = (
        config.benchmark.observer_batch_size * sample_multiplier * macro_step
    )
    return training_examples * (
        macro_step.bit_count() + 2 * epoch_budget
    )


def _persistent_model_passes(
    config: VampLogTDenseConfig,
    sample_multiplier: int,
) -> int:
    """Return exact counted passes for all persistent uniform updates."""
    current_examples = config.benchmark.observer_batch_size * sample_multiplier
    return sum(
        (current_examples if step == 1 else 2 * current_examples)
        * (step.bit_count() + 2 * config.integrator.epochs_per_step)
        for step in range(1, config.benchmark.macro_steps + 1)
    )


def _run_scaling(
    config: VampLogTDenseConfig,
    run_root: Path,
    device: torch.device,
) -> dict[str, object]:
    """Run or resume every declared consolidation-policy and seed pair."""
    if _is_capacity_study(config):
        if config.scaling is None:
            raise ValueError("capacity study lacks scaling coordinates")
        summaries = tuple(
            _run_scaling_seed(
                config,
                run_root,
                1,
                seed,
                device,
                training_sample_multiplier=multiplier,
            )
            for multiplier in config.scaling.training_sample_multipliers
            for seed in config.online.seeds
        )
        return {
            "arms": list(summaries),
            "config_hash": config.config_hash,
            "status": "complete"
            if all(summary.get("status") == "complete" for summary in summaries)
            else "failed_acceptance",
        }
    capacities = (
        (1,)
        if config.scaling is None
        else config.scaling.hierarchy_node_capacities
    )
    summaries = tuple(
        _run_scaling_seed(config, run_root, capacity, seed, device)
        for capacity in capacities
        for seed in config.online.seeds
    )
    if config.scaling is None:
        return summaries[0]
    return _aggregate_summary(config, _all_metric_rows(config, run_root))


def _run_scaling_seed(
    config: VampLogTDenseConfig,
    run_root: Path,
    capacity: int,
    seed: int,
    device: torch.device,
    *,
    training_sample_multiplier: int = 1,
    full_replay_checkpoints: tuple[int, ...] | None = None,
    full_replay_condition: str = FULL_REPLAY_CONDITION,
    offline_epochs: int | None = None,
) -> dict[str, object]:
    """Run one crash-resumable seed against one frozen hierarchy policy."""
    policy_root = (
        _arm_root(config, run_root, training_sample_multiplier)
        if _uses_sample_multiplier_study(config)
        else _policy_root(config, run_root, capacity)
    )
    scheduled_full_replay = (
        config.evaluation.full_checkpoints
        if full_replay_checkpoints is None
        else full_replay_checkpoints
    )
    epoch_budget = (
        config.integrator.offline_epochs
        if offline_epochs is None
        else offline_epochs
    )
    directory = policy_root / "scaling" / f"seed-{seed}"
    summary_path = directory / "summary.json"
    checkpoint_path = directory / "state" / "uniform.pt"
    ledger = ChainedJsonlLedger(directory / "metrics.jsonl", "vamp-logt-scaling-metric-v2")
    base = load_calibrated_base(config, run_root)
    benchmark = (
        build_benchmark(config, seed)
        if training_sample_multiplier == 1
        else build_benchmark(
            config,
            seed,
            training_sample_multiplier=training_sample_multiplier,
        )
    )
    slot_dim = base.embedding_dim + 11
    input_dim = config.observer.maximum_levels * capacity * slot_dim
    uniform_state, initialization_seconds = _timed_state_creation(
        UNIFORM_CONDITION,
        input_dim,
        slot_dim,
        config,
        seed,
        capacity,
        device,
    )
    completed_step, checkpoint_rows = _load_uniform_checkpoint(
        checkpoint_path,
        config,
        uniform_state,
        seed,
        capacity,
        device,
        training_sample_multiplier=training_sample_multiplier,
    )
    if ledger.next_sequence < checkpoint_rows:
        raise ValueError("scaling checkpoint refers to absent metric rows")
    ledger.truncate(checkpoint_rows)
    if summary_path.is_file():
        summary = load_canonical_json(summary_path)
        if (
            completed_step != config.benchmark.macro_steps
            or int(summary.get("final_macro_step", -1)) != completed_step
            or int(summary.get("metric_rows", -1)) != ledger.next_sequence
            or int(summary.get("run_seed", -1)) != seed
            or int(summary.get("max_nodes_per_level", -1)) != capacity
            or (
                _uses_sample_multiplier_study(config)
                and int(summary.get("training_sample_multiplier", -1))
                != training_sample_multiplier
            )
            or (
                _is_sample_calibration_study(config)
                and (
                    tuple(int(value) for value in summary["full_replay_checkpoints"])
                    != scheduled_full_replay
                    or summary.get("full_replay_condition")
                    != full_replay_condition
                    or int(summary.get("full_replay_epochs", -1))
                    != epoch_budget
                )
            )
        ):
            raise ValueError("completed scaling summary is not covered by its checkpoint")
        return summary

    observer_batches = tuple(
        benchmark.step(step).observer for step in range(1, completed_step + 1)
    )
    try:
        from tqdm.auto import tqdm
    except ImportError as error:  # pragma: no cover - vision environment gate
        raise RuntimeError("tqdm is required by the vision environment") from error
    progress = tqdm(
        range(completed_step + 1, config.benchmark.macro_steps + 1),
        initial=completed_step,
        total=config.benchmark.macro_steps,
        desc=(
            f"integrators capacity={capacity} seed={seed} "
            f"samples={training_sample_multiplier}x"
        ),
        disable=not config.runtime.progress,
        unit="permutation",
    )
    for macro_step in progress:
        current_examples = benchmark.step(macro_step).observer
        frontier = load_frontier(
            config,
            run_root,
            seed,
            macro_step,
            max_nodes_per_level=capacity,
            hierarchy_root=policy_root / "hierarchy",
            training_sample_multiplier=training_sample_multiplier,
        )
        current_features = _timed_features(config, frontier, current_examples, base, device)
        replay_features, replay_examples, preparation_seconds = _prepare_uniform_replay(
            config,
            frontier,
            observer_batches,
            base,
            macro_step,
            seed,
            device,
            training_sample_multiplier,
        )
        uniform_result = train_condition(
            uniform_state,
            IntegratorSupervision(current_features.observations, current_examples.labels),
            (
                None
                if replay_features is None or replay_examples is None
                else IntegratorSupervision(replay_features.observations, replay_examples.labels)
            ),
            config.integrator.epochs_per_step,
            config.integrator,
            seed,
            macro_step,
            device,
        )
        uniform_work = _TrainingWork(
            initialization_seconds if macro_step == 1 and completed_step == 0 else 0.0,
            preparation_seconds,
            current_features.wall_seconds
            + (0.0 if replay_features is None else replay_features.wall_seconds),
            uniform_result.training_wall_seconds,
            current_features.forward_example_passes
            + (0 if replay_features is None else replay_features.forward_example_passes),
            current_features.forward_calls
            + (0 if replay_features is None else replay_features.forward_calls),
            uniform_result.training_forward_example_passes,
            uniform_result.training_backward_example_passes,
            uniform_result.training_forward_calls,
            uniform_result.training_backward_calls,
            2
            * (
                len(current_examples.labels)
                + (0 if replay_examples is None else len(replay_examples.labels))
            ),
            2
            * math.ceil(
                (
                    len(current_examples.labels)
                    + (0 if replay_examples is None else len(replay_examples.labels))
                )
                / config.integrator.minibatch_size
            ),
            len(current_examples.labels)
            + (0 if replay_examples is None else len(replay_examples.labels)),
            len(current_examples.labels),
            0 if replay_examples is None else len(replay_examples.labels),
            config.integrator.epochs_per_step,
        )
        cumulative_batches = (*observer_batches, current_examples)
        full_fit = (
            _fit_full_replay(
                config,
                directory,
                macro_step,
                frontier,
                cumulative_batches,
                base,
                input_dim,
                slot_dim,
                seed,
                capacity,
                device,
                training_sample_multiplier,
                condition=full_replay_condition,
                offline_epochs=epoch_budget,
            )
            if macro_step in scheduled_full_replay
            else None
        )
        evaluation, excluded_evaluation = _evaluate_conditions(
            config,
            benchmark,
            frontier,
            base,
            {
                UNIFORM_CONDITION: uniform_state,
                **(
                    {}
                    if full_fit is None
                    else {full_replay_condition: full_fit.state}
                ),
            },
            macro_step,
            device,
        )
        rows = [
            _shared_hierarchy_work(
                config,
                macro_step,
                capacity,
                seed,
                training_sample_multiplier=training_sample_multiplier,
            ),
            _condition_row(
                UNIFORM_CONDITION,
                macro_step,
                frontier,
                uniform_work,
                evaluation[UNIFORM_CONDITION],
                excluded_evaluation,
                None,
                seed,
                capacity,
                uniform_state,
                config,
                base,
                training_sample_multiplier,
            ),
        ]
        if full_fit is not None:
            rows.append(
                _condition_row(
                    full_replay_condition,
                    macro_step,
                    frontier,
                    full_fit.work,
                    evaluation[full_replay_condition],
                    excluded_evaluation,
                    full_fit.checkpoint_sha256,
                    seed,
                    capacity,
                    full_fit.state,
                    config,
                    base,
                    training_sample_multiplier,
                    training_storage=full_fit.storage_record(),
                )
            )
        ledger.append_many(rows)
        observer_batches = cumulative_batches
        _save_uniform_checkpoint(
            checkpoint_path,
            config,
            macro_step,
            uniform_state,
            ledger.next_sequence,
            seed,
            capacity,
            training_sample_multiplier=training_sample_multiplier,
        )
        progress.set_postfix(
            full="yes" if full_fit is not None else "no",
            nodes=len(frontier.nodes),
            uniform=f"{uniform_work.total_training_wall_seconds:.2f}s",
        )

    summary = _seed_summary(
        config,
        ledger.rows,
        seed,
        capacity,
        training_sample_multiplier=training_sample_multiplier,
        full_replay_checkpoints=scheduled_full_replay,
        full_replay_condition=full_replay_condition,
        offline_epochs=epoch_budget,
    )
    publish_immutable_json(summary_path, summary)
    return summary


def _timed_state_creation(
    name: str,
    input_dim: int,
    slot_dim: int,
    config: VampLogTDenseConfig,
    seed: int,
    capacity: int,
    device: torch.device,
) -> tuple[IntegratorConditionState, float]:
    """Create paired expansion-stable integrator initialization."""
    _synchronize(device)
    started = perf_counter()
    primary_input_dim = config.observer.maximum_levels * slot_dim
    primary = create_condition_state(
        name,
        primary_input_dim,
        slot_dim,
        config.integrator,
        seed,
        device,
    )
    if capacity == 1:
        if input_dim != primary_input_dim:
            raise ValueError("one-node policy input dimension changed")
        state = primary
    else:
        state = create_condition_state(
            name,
            input_dim,
            slot_dim,
            config.integrator,
            seed,
            device,
            maximum_slots=config.observer.maximum_levels * capacity,
        )
        with torch.no_grad():
            state.integrator.input_layer.weight.zero_()
            state.integrator.input_layer.weight[:, :primary_input_dim].copy_(
                primary.integrator.input_layer.weight
            )
            state.integrator.input_layer.bias.copy_(primary.integrator.input_layer.bias)
            state.integrator.middle.load_state_dict(primary.integrator.middle.state_dict())
            state.integrator.output_layer.load_state_dict(
                primary.integrator.output_layer.state_dict()
            )
    _synchronize(device)
    return state, perf_counter() - started


def _timed_features(
    config: VampLogTDenseConfig,
    frontier: DenseFrontier,
    examples: ExampleBatch,
    base: DenseMlpState,
    device: torch.device,
) -> _TimedFeatures:
    _synchronize(device)
    started = perf_counter()
    observations = build_dense_observations(
        frontier,
        examples,
        base,
        config.observer.maximum_levels,
        config.router.target_temperature,
        device,
        config.observer.inference_batch_size,
    ).integrator
    _synchronize(device)
    return _TimedFeatures(
        observations,
        perf_counter() - started,
        len(frontier.nodes) * len(examples.labels),
        len(frontier.nodes)
        * math.ceil(len(examples.labels) / config.observer.inference_batch_size),
    )


def _prepare_uniform_replay(
    config: VampLogTDenseConfig,
    frontier: DenseFrontier,
    observer_batches: tuple[ExampleBatch, ...],
    base: DenseMlpState,
    macro_step: int,
    seed: int,
    device: torch.device,
    training_sample_multiplier: int = 1,
) -> tuple[_TimedFeatures | None, ExampleBatch | None, float]:
    if not observer_batches:
        return None, None, 0.0
    started = perf_counter()
    replay_seed = named_seed(seed, UNIFORM_CONDITION, macro_step, "replay")
    if training_sample_multiplier == 1:
        historical_archive = concatenate_batches(observer_batches)
        replay_examples = sample_example_balanced(
            historical_archive,
            config.online.historical_budget,
            replay_seed,
            macro_step,
        ).batch
    elif _uses_sample_multiplier_study(config):
        group_size = config.benchmark.observer_batch_size
        group_archives = tuple(
            concatenate_batches(tuple(
                batch.select(torch.arange(group * group_size, (group + 1) * group_size))
                for batch in observer_batches
            ))
            for group in range(training_sample_multiplier)
        )
        group_seeds = tuple(
            replay_seed
            if group == 0
            else (
                named_seed(seed, UNIFORM_CONDITION, macro_step, "extra-replay")
                if group == 1
                else named_seed(
                    seed,
                    UNIFORM_CONDITION,
                    macro_step,
                    "extra-replay",
                    group,
                )
            )
            for group in range(training_sample_multiplier)
        )
        replay_examples = concatenate_batches(tuple(
            sample_example_balanced(
                archive,
                config.online.historical_budget,
                group_seed,
                macro_step,
            ).batch
            for archive, group_seed in zip(group_archives, group_seeds, strict=True)
        ))
    else:
        raise ValueError("unsupported paired replay sample multiplier")
    preparation_seconds = perf_counter() - started
    return (
        _timed_features(config, frontier, replay_examples, base, device),
        replay_examples,
        preparation_seconds,
    )


def _iter_full_replay_batches(
    source: Sequence[ExampleBatch] | PermutedMnistBenchmark,
    macro_step: int,
) -> Iterator[ExampleBatch]:
    """Yield full-replay observer batches without forcing transformed images resident."""
    if isinstance(source, PermutedMnistBenchmark):
        yield from (
            source.step(step).observer for step in range(1, macro_step + 1)
        )
        return
    if len(source) != macro_step:
        raise ValueError("full-replay source does not cover the requested prefix")
    yield from source


def _build_full_replay_feature_cache(
    config: VampLogTDenseConfig,
    directory: Path,
    macro_step: int,
    frontier: DenseFrontier,
    source: Sequence[ExampleBatch] | PermutedMnistBenchmark,
    base: DenseMlpState,
    input_dim: int,
    device: torch.device,
    training_sample_multiplier: int,
) -> _DiskFeatureCache:
    """Write one bounded-block frozen-feature pass to a temporary memory map."""
    cache_directory = directory / "full-replay"
    cache_directory.mkdir(parents=True, exist_ok=True)
    cache_path = cache_directory / f".step-{macro_step:03d}-features.f32"
    rows = (
        config.benchmark.observer_batch_size
        * training_sample_multiplier
        * macro_step
    )
    record_columns = input_dim + 10 + 1
    cache_bytes = rows * record_columns * np.dtype(np.float32).itemsize
    filesystem = os.statvfs(cache_directory)
    available_bytes = filesystem.f_bavail * filesystem.f_frsize
    reserve_bytes = 2 * 1024**3
    if available_bytes < cache_bytes + reserve_bytes:
        raise RuntimeError(
            "full-replay feature cache lacks disk headroom: "
            f"needs {(cache_bytes + reserve_bytes) / 1024**3:.1f} GiB, "
            f"has {available_bytes / 1024**3:.1f} GiB"
        )
    memory_available_bytes = next(
        (
            int(line.split()[1]) * 1024
            for line in Path("/proc/meminfo").read_text().splitlines()
            if line.startswith("MemAvailable:")
        ),
        0,
    )
    minimum_memory_headroom = 4 * 1024**3
    if memory_available_bytes < minimum_memory_headroom:
        raise RuntimeError(
            "full-replay feature cache requires at least 4 GiB available host "
            f"memory; found {memory_available_bytes / 1024**3:.1f} GiB"
        )
    if cache_path.exists():
        if not cache_path.is_file():
            raise ValueError("full-replay temporary cache path is not a file")
        cache_path.unlink()
    print(
        "Memory-bounded full replay: "
        f"{rows:,} rows, {cache_bytes / 1024**3:.1f} GiB temporary cache at "
        f"{cache_path}",
        flush=True,
    )
    try:
        from tqdm.auto import tqdm
    except ImportError as error:  # pragma: no cover - vision environment gate
        raise RuntimeError("tqdm is required by the vision environment") from error

    records = np.memmap(
        cache_path,
        dtype=np.float32,
        mode="w+",
        shape=(rows, record_columns),
    )
    feature_wall_seconds = 0.0
    feature_forward_example_passes = 0
    feature_forward_calls = 0
    peak_resident_feature_rows = 0
    offset = 0
    preparation_started = perf_counter()
    try:
        batches = _iter_full_replay_batches(source, macro_step)
        progress = tqdm(
            batches,
            total=macro_step,
            desc=f"cache frozen features t={macro_step}",
            disable=not config.runtime.progress,
            unit="task",
        )
        for batch_index, examples in enumerate(progress, start=1):
            stop = offset + len(examples.labels)
            if stop > rows:
                raise ValueError("full-replay feature cache exceeded its declared rows")
            timed = _timed_features(config, frontier, examples, base, device)
            observations = timed.observations
            records[offset:stop, :input_dim] = observations.features.numpy()
            records[offset:stop, input_dim : input_dim + 10] = (
                observations.baseline_log_probabilities.numpy()
            )
            records[offset:stop, -1] = examples.labels.numpy()
            feature_wall_seconds += timed.wall_seconds
            feature_forward_example_passes += timed.forward_example_passes
            feature_forward_calls += timed.forward_calls
            peak_resident_feature_rows = max(
                peak_resident_feature_rows,
                len(examples.labels),
            )
            offset = stop
            del observations, timed
            if batch_index % 8 == 0:
                records.flush()
        if offset != rows:
            raise ValueError("full-replay feature cache did not fill its declared rows")
        records.flush()
    except BaseException:
        del records
        cache_path.unlink(missing_ok=True)
        raise
    del records
    if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
        with cache_path.open("rb") as cache_file:
            os.posix_fadvise(
                cache_file.fileno(),
                0,
                0,
                os.POSIX_FADV_DONTNEED,
            )
    data_preparation_wall_seconds = (
        perf_counter() - preparation_started - feature_wall_seconds
    )
    return _DiskFeatureCache(
        cache_path,
        rows,
        input_dim,
        record_columns,
        data_preparation_wall_seconds,
        feature_wall_seconds,
        feature_forward_example_passes,
        feature_forward_calls,
        peak_resident_feature_rows,
        cache_bytes,
    )


def _train_full_replay_feature_cache(
    state: IntegratorConditionState,
    cache: _DiskFeatureCache,
    epochs: int,
    config: VampLogTDenseConfig,
    seed: int,
    macro_step: int,
    device: torch.device,
) -> tuple[float, float, int]:
    """Train in the original seeded minibatch order from a bounded disk cache."""
    if cache.path.stat().st_size != cache.bytes:
        raise ValueError("full-replay feature cache size changed")
    chunk_count = math.ceil(cache.rows / config.integrator.minibatch_size)
    records = np.memmap(
        cache.path,
        dtype=np.float32,
        mode="r",
        shape=(cache.rows, cache.record_columns),
    )
    try:
        from tqdm.auto import tqdm
    except ImportError as error:  # pragma: no cover - vision environment gate
        raise RuntimeError("tqdm is required by the vision environment") from error
    progress = tqdm(
        total=epochs * cache.rows,
        desc=f"train full replay t={macro_step}",
        disable=not config.runtime.progress,
        unit="example",
    )
    data_preparation_wall_seconds = 0.0
    optimizer_wall_seconds = 0.0
    device_indices = [device.index or 0] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=device_indices):
            torch.manual_seed(
                integrator_named_seed(seed, state.name, macro_step, "dropout")
            )
            for epoch in range(epochs):
                state.integrator.train()
                preparation_started = perf_counter()
                order = torch.randperm(
                    cache.rows,
                    generator=torch.Generator().manual_seed(
                        integrator_named_seed(
                            seed,
                            state.name,
                            macro_step,
                            "minibatches",
                            epoch,
                        )
                    ),
                )
                chunks = torch.tensor_split(order, chunk_count)
                data_preparation_wall_seconds += perf_counter() - preparation_started
                for indices in chunks:
                    preparation_started = perf_counter()
                    cached_rows = np.array(
                        records[indices.numpy()],
                        dtype=np.float32,
                        copy=True,
                        order="C",
                    )
                    feature_rows = torch.from_numpy(cached_rows[:, : cache.input_dim])
                    baseline_rows = torch.from_numpy(
                        cached_rows[:, cache.input_dim : cache.input_dim + 10]
                    )
                    labels = torch.from_numpy(
                        cached_rows[:, -1].astype(np.int64, copy=True)
                    )
                    data_preparation_wall_seconds += (
                        perf_counter() - preparation_started
                    )

                    _synchronize(device)
                    optimizer_started = perf_counter()
                    loss = (
                        len(indices)
                        * chunk_count
                        / cache.rows
                        * F.cross_entropy(
                            state.integrator(
                                feature_rows.to(device),
                                baseline_rows.to(device),
                            ),
                            labels.to(device),
                        )
                    )
                    state.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        state.integrator.parameters(),
                        config.integrator.gradient_clip_norm,
                    )
                    state.optimizer.step()
                    state.optimizer_steps += 1
                    _synchronize(device)
                    optimizer_wall_seconds += perf_counter() - optimizer_started
                    progress.update(len(indices))
    finally:
        progress.close()
        del records
    return data_preparation_wall_seconds, optimizer_wall_seconds, chunk_count


def _fit_full_replay(
    config: VampLogTDenseConfig,
    directory: Path,
    macro_step: int,
    frontier: DenseFrontier,
    observer_batches: Sequence[ExampleBatch] | PermutedMnistBenchmark,
    base: DenseMlpState,
    input_dim: int,
    slot_dim: int,
    seed: int,
    capacity: int,
    device: torch.device,
    training_sample_multiplier: int = 1,
    *,
    condition: str = FULL_REPLAY_CONDITION,
    offline_epochs: int | None = None,
) -> _FullReplayFit:
    path = directory / "full-replay" / f"step-{macro_step:03d}.pt"
    name = _full_replay_state_name(condition, training_sample_multiplier, macro_step)
    state, initialization_seconds = _timed_state_creation(
        name,
        input_dim,
        slot_dim,
        config,
        seed,
        capacity,
        device,
    )
    sample_multiplier_study = _uses_sample_multiplier_study(config)
    epoch_budget = (
        config.integrator.offline_epochs
        if offline_epochs is None
        else offline_epochs
    )
    if epoch_budget < 1 or epoch_budget > config.integrator.offline_epochs:
        raise ValueError("full-replay epoch budget is outside the calibrated limit")
    sample_calibration_study = _is_sample_calibration_study(config)
    schema_version = (
        "vamp-logt-scaling-full-replay-v4"
        if sample_calibration_study
        else (
            "vamp-logt-scaling-full-replay-v3"
            if sample_multiplier_study
            else "vamp-logt-scaling-full-replay-v2"
        )
    )
    if path.is_file():
        payload = torch.load(path, map_location=device, weights_only=True)
        if (
            payload.get("schema_version") != schema_version
            or payload.get("config_hash") != config.config_hash
            or int(payload.get("macro_step", -1)) != macro_step
            or int(payload.get("run_seed", -1)) != seed
            or int(payload.get("max_nodes_per_level", -1)) != capacity
            or (
                sample_multiplier_study
                and int(payload.get("training_sample_multiplier", -1))
                != training_sample_multiplier
            )
            or int(payload["work"].get("epochs", -1)) != epoch_budget
            or (
                sample_calibration_study
                and payload.get("condition") != condition
            )
        ):
            raise ValueError("stored scaling full-replay fit coordinates changed")
        state.integrator.load_state_dict(payload["model"], strict=True)
        state.optimizer_steps = int(payload["optimizer_steps"])
        return _FullReplayFit(
            state,
            _training_work_from_record(payload["work"]),
            file_sha256(path),
            str(payload.get("feature_storage", "in_memory")),
            int(payload.get("peak_resident_feature_rows", 0)),
            int(payload.get("temporary_feature_cache_bytes", 0)),
        )

    cache: _DiskFeatureCache | None = None
    try:
        if sample_calibration_study:
            cache = _build_full_replay_feature_cache(
                config,
                directory,
                macro_step,
                frontier,
                observer_batches,
                base,
                input_dim,
                device,
                training_sample_multiplier,
            )
            training_preparation_seconds, optimizer_seconds, chunk_count = (
                _train_full_replay_feature_cache(
                    state,
                    cache,
                    epoch_budget,
                    config,
                    seed,
                    macro_step,
                    device,
                )
            )
            work = _TrainingWork(
                initialization_seconds,
                cache.data_preparation_wall_seconds + training_preparation_seconds,
                cache.feature_wall_seconds,
                optimizer_seconds,
                cache.feature_forward_example_passes,
                cache.feature_forward_calls,
                cache.rows * epoch_budget,
                cache.rows * epoch_budget,
                chunk_count * epoch_budget,
                chunk_count * epoch_budget,
                0,
                0,
                cache.rows,
                cache.rows,
                0,
                epoch_budget,
            )
            feature_storage = "temporary_float32_memory_map"
            peak_resident_feature_rows = cache.peak_resident_feature_rows
            temporary_feature_cache_bytes = cache.bytes
        else:
            preparation_started = perf_counter()
            archive = concatenate_batches(
                tuple(_iter_full_replay_batches(observer_batches, macro_step))
            )
            preparation_seconds = perf_counter() - preparation_started
            features = _timed_features(config, frontier, archive, base, device)
            result = train_condition(
                state,
                IntegratorSupervision(features.observations, archive.labels),
                None,
                epoch_budget,
                config.integrator,
                seed,
                macro_step,
                device,
            )
            work = _TrainingWork(
                initialization_seconds,
                preparation_seconds,
                features.wall_seconds,
                result.training_wall_seconds,
                features.forward_example_passes,
                features.forward_calls,
                result.training_forward_example_passes,
                result.training_backward_example_passes,
                result.training_forward_calls,
                result.training_backward_calls,
                2 * len(archive.labels),
                2 * math.ceil(
                    len(archive.labels) / config.integrator.minibatch_size
                ),
                len(archive.labels),
                len(archive.labels),
                0,
                epoch_budget,
            )
            feature_storage = "in_memory"
            peak_resident_feature_rows = len(archive.labels)
            temporary_feature_cache_bytes = 0
        payload = {
            "config_hash": config.config_hash,
            "feature_storage": feature_storage,
            "macro_step": macro_step,
            "max_nodes_per_level": capacity,
            "model": _cpu_state_dict(state.integrator.state_dict()),
            "optimizer_steps": state.optimizer_steps,
            "peak_resident_feature_rows": peak_resident_feature_rows,
            "run_seed": seed,
            "schema_version": schema_version,
            "temporary_feature_cache_bytes": temporary_feature_cache_bytes,
            "work": work.as_record(),
        }
        if sample_multiplier_study:
            payload["training_sample_multiplier"] = training_sample_multiplier
        if sample_calibration_study:
            payload["condition"] = condition
        atomic_torch_save(path, payload)
    finally:
        if cache is not None:
            cache.path.unlink(missing_ok=True)
    return _FullReplayFit(
        state,
        work,
        file_sha256(path),
        feature_storage,
        peak_resident_feature_rows,
        temporary_feature_cache_bytes,
    )


def _training_work_from_record(value: object) -> _TrainingWork:
    if not isinstance(value, dict):
        raise ValueError("stored scaling work record is malformed")
    return _TrainingWork(
        *(float(value[name]) for name in (
            "state_initialization_wall_seconds",
            "data_preparation_wall_seconds",
            "feature_wall_seconds",
            "optimizer_wall_seconds",
        )),
        *(int(value[name]) for name in (
            "feature_forward_example_passes",
            "feature_forward_calls",
            "integrator_forward_example_passes",
            "integrator_backward_example_passes",
            "integrator_forward_calls",
            "integrator_backward_calls",
            "excluded_diagnostic_forward_example_passes",
            "excluded_diagnostic_forward_calls",
            "training_examples",
            "current_examples",
            "historical_examples",
            "epochs",
        )),
    )


def _evaluate_conditions(
    config: VampLogTDenseConfig,
    benchmark: PermutedMnistBenchmark,
    frontier: DenseFrontier,
    base: DenseMlpState,
    states: Mapping[str, IntegratorConditionState],
    macro_step: int,
    device: torch.device,
) -> tuple[dict[str, dict[str, float]], dict[str, object]]:
    started = perf_counter()
    learned_domains = tuple(
        allocation.domain_id for allocation in benchmark.allocations[:macro_step]
    )
    examples = concatenate_batches(
        tuple(benchmark.test_domain(domain, full=False) for domain in learned_domains)
    )
    features = _timed_features(config, frontier, examples, base, device)
    metrics = {}
    for name, state in states.items():
        logits = prediction_logits(
            state.integrator,
            features.observations,
            device,
            config.integrator.minibatch_size,
        )
        metrics[name] = {
            "accuracy": float(
                (logits.argmax(dim=1) == examples.labels).float().mean().item()
            ),
            "cross_entropy": float(F.cross_entropy(logits, examples.labels).item()),
        }
    _synchronize(device)
    return metrics, {
        "excluded_evaluation_examples": len(examples.labels),
        "excluded_evaluation_feature_forward_calls": features.forward_calls,
        "excluded_evaluation_feature_forward_example_passes": features.forward_example_passes,
        "excluded_evaluation_integrator_forward_calls_per_condition": math.ceil(
            len(examples.labels) / config.integrator.minibatch_size
        ),
        "excluded_evaluation_integrator_forward_example_passes_per_condition": len(
            examples.labels
        ),
        "excluded_evaluation_wall_seconds": perf_counter() - started,
    }


def _condition_row(
    condition: str,
    macro_step: int,
    frontier: DenseFrontier,
    work: _TrainingWork,
    metrics: Mapping[str, float],
    excluded_evaluation: Mapping[str, object],
    checkpoint_sha256: str | None,
    seed: int,
    capacity: int,
    state: IntegratorConditionState,
    config: VampLogTDenseConfig,
    base: DenseMlpState,
    training_sample_multiplier: int = 1,
    *,
    training_storage: Mapping[str, object] | None = None,
) -> dict[str, object]:
    arm = _arm_name(config, capacity, training_sample_multiplier)
    return {
        "accuracy": metrics["accuracy"],
        "active_node_count": len(frontier.nodes),
        "arm": arm,
        "arm_label": _arm_label(arm),
        "base_hidden_widths": list(base.hidden_widths),
        "base_parameter_count": base.parameter_count,
        "checkpoint_sha256": checkpoint_sha256,
        "condition": condition,
        "cross_entropy": metrics["cross_entropy"],
        "learned_permutation_count": macro_step,
        "macro_step": macro_step,
        "max_nodes_per_level": capacity,
        "input_slot_count": state.integrator.maximum_levels,
        "integrator_parameter_count": sum(
            parameter.numel() for parameter in state.integrator.parameters()
        ),
        "policy": POLICY_NAMES[capacity],
        "row_type": "condition",
        "run_seed": seed,
        "training_sample_multiplier": training_sample_multiplier,
        "temporal_ranges": [
            [node.first_block + 1, node.last_block + 1] for node in frontier.nodes
        ],
        **work.as_record(),
        **({} if training_storage is None else dict(training_storage)),
        **dict(excluded_evaluation),
    }


def _shared_hierarchy_work(
    config: VampLogTDenseConfig,
    macro_step: int,
    capacity: int = 1,
    seed: int = 0,
    *,
    training_sample_multiplier: int = 1,
) -> dict[str, object]:
    node_examples = tuple(
        config.benchmark.model_batch_size
        * training_sample_multiplier
        * 2**level
        for level in _created_node_levels(macro_step, capacity)
    )
    example_passes = config.node.epochs * sum(node_examples)
    batch_calls = config.node.epochs * sum(
        math.ceil(count / config.node.optimizer.batch_size) for count in node_examples
    )
    return {
        "arm": _arm_name(config, capacity, training_sample_multiplier),
        "created_node_count": len(node_examples),
        "learned_permutation_count": macro_step,
        "macro_step": macro_step,
        "max_nodes_per_level": capacity,
        "policy": POLICY_NAMES[capacity],
        "row_type": "shared_hierarchy_work",
        "run_seed": seed,
        "shared_backward_calls": batch_calls,
        "shared_backward_example_passes": example_passes,
        "shared_forward_calls": batch_calls,
        "shared_forward_example_passes": example_passes,
        "training_sample_multiplier": training_sample_multiplier,
    }


@lru_cache(maxsize=None)
def _created_node_levels(macro_step: int, capacity: int) -> tuple[int, ...]:
    """Return the leaf and carry-parent levels created by one insertion."""
    if macro_step < 1 or capacity < 1:
        raise ValueError("hierarchy work requires positive step and capacity")
    counts: dict[int, int] = {}
    created = ()
    for step in range(1, macro_step + 1):
        level = 0
        step_created = [0]
        while True:
            counts[level] = counts.get(level, 0) + 1
            if counts[level] <= capacity:
                break
            counts[level] -= 2
            level += 1
            step_created.append(level)
        if step == macro_step:
            created = tuple(step_created)
    return created


def _save_uniform_checkpoint(
    path: Path,
    config: VampLogTDenseConfig,
    macro_step: int,
    state: IntegratorConditionState,
    metric_rows: int,
    seed: int,
    capacity: int,
    *,
    training_sample_multiplier: int = 1,
) -> None:
    sample_multiplier_study = _uses_sample_multiplier_study(config)
    payload = {
            "config_hash": config.config_hash,
            "macro_step": macro_step,
            "max_nodes_per_level": capacity,
            "metric_rows": metric_rows,
            "model": _cpu_state_dict(state.integrator.state_dict()),
            "optimizer": state.optimizer.state_dict(),
            "optimizer_steps": state.optimizer_steps,
            "run_seed": seed,
            "schema_version": (
                "vamp-logt-scaling-uniform-checkpoint-v3"
                if sample_multiplier_study
                else "vamp-logt-scaling-uniform-checkpoint-v2"
            ),
        }
    if sample_multiplier_study:
        payload["training_sample_multiplier"] = training_sample_multiplier
    atomic_torch_save(path, payload)


def _load_uniform_checkpoint(
    path: Path,
    config: VampLogTDenseConfig,
    state: IntegratorConditionState,
    seed: int,
    capacity: int,
    device: torch.device,
    *,
    training_sample_multiplier: int = 1,
) -> tuple[int, int]:
    if not path.is_file():
        return 0, 0
    payload = torch.load(path, map_location=device, weights_only=False)
    if (
        payload.get("schema_version")
        != (
            "vamp-logt-scaling-uniform-checkpoint-v3"
            if _uses_sample_multiplier_study(config)
            else "vamp-logt-scaling-uniform-checkpoint-v2"
        )
        or payload.get("config_hash") != config.config_hash
        or int(payload.get("run_seed", -1)) != seed
        or int(payload.get("max_nodes_per_level", -1)) != capacity
        or (
            _uses_sample_multiplier_study(config)
            and int(payload.get("training_sample_multiplier", -1))
            != training_sample_multiplier
        )
    ):
        raise ValueError("stored scaling uniform checkpoint coordinates changed")
    state.integrator.load_state_dict(payload["model"], strict=True)
    state.optimizer.load_state_dict(payload["optimizer"])
    _optimizer_to_device(state.optimizer, device)
    state.optimizer_steps = int(payload["optimizer_steps"])
    return int(payload["macro_step"]), int(payload["metric_rows"])


def _seed_summary(
    config: VampLogTDenseConfig,
    rows: Sequence[Mapping[str, object]],
    seed: int,
    capacity: int,
    *,
    training_sample_multiplier: int = 1,
    full_replay_checkpoints: tuple[int, ...] | None = None,
    full_replay_condition: str = FULL_REPLAY_CONDITION,
    offline_epochs: int | None = None,
) -> dict[str, object]:
    """Summarize and validate one policy/seed ledger."""
    scheduled_full_replay = (
        config.evaluation.full_checkpoints
        if full_replay_checkpoints is None
        else full_replay_checkpoints
    )
    epoch_budget = (
        config.integrator.offline_epochs
        if offline_epochs is None
        else offline_epochs
    )
    conditions = tuple(row for row in rows if row.get("row_type") == "condition")
    uniform = tuple(row for row in conditions if row["condition"] == UNIFORM_CONDITION)
    full = tuple(
        row for row in conditions if row["condition"] == full_replay_condition
    )
    epoch_acceptance_name = (
        "full_replay_exactly_twenty_epochs"
        if full_replay_condition == FULL_REPLAY_CONDITION
        else "full_replay_selected_epoch_budget_exact"
    )
    hierarchy = tuple(
        row for row in rows if row.get("row_type") == "shared_hierarchy_work"
    )
    acceptance = {
        "all_metrics_finite": all(
            math.isfinite(float(row[name]))
            for row in conditions
            for name in ("accuracy", "cross_entropy", "total_training_wall_seconds")
        ),
        "ceiling_checkpoints_exact": tuple(int(row["macro_step"]) for row in full)
        == scheduled_full_replay,
        "evaluation_excluded_from_training_work": all(
            "excluded_evaluation_forward_example_passes"
            not in row
            and int(row["total_training_forward_example_passes"])
            == int(row["feature_forward_example_passes"])
            + int(row["integrator_forward_example_passes"])
            for row in conditions
        ),
        epoch_acceptance_name: all(
            int(row["epochs"]) == epoch_budget for row in full
        ),
        "coordinates_exact": all(
            int(row["run_seed"]) == seed
            and int(row["max_nodes_per_level"]) == capacity
            and row["policy"] == POLICY_NAMES[capacity]
            and int(row.get("training_sample_multiplier", 1))
            == training_sample_multiplier
            for row in rows
        ),
        "uniform_exact_replay_budget": all(
            int(row["historical_examples"])
            == (
                0
                if int(row["macro_step"]) == 1
                else config.online.historical_budget * training_sample_multiplier
            )
            for row in uniform
        ),
        "uniform_every_permutation": tuple(int(row["macro_step"]) for row in uniform)
        == tuple(range(1, config.benchmark.macro_steps + 1)),
    }
    return {
        "acceptance": acceptance,
        "condition_rows": len(conditions),
        "config_hash": config.config_hash,
        "final_macro_step": config.benchmark.macro_steps,
        "full_replay_checkpoints": list(scheduled_full_replay),
        "full_replay_condition": full_replay_condition,
        "full_replay_epochs": epoch_budget,
        "max_nodes_per_level": capacity,
        "metric_rows": len(rows),
        "policy": POLICY_NAMES[capacity],
        "run_seed": seed,
        "schema_version": "vamp-logt-scaling-seed-summary-v2",
        "shared_hierarchy_backward_example_passes": sum(
            int(row["shared_backward_example_passes"]) for row in hierarchy
        ),
        "shared_hierarchy_forward_example_passes": sum(
            int(row["shared_forward_example_passes"]) for row in hierarchy
        ),
        "status": "complete" if all(acceptance.values()) else "failed_acceptance",
        "training_sample_multiplier": training_sample_multiplier,
        "uniform_cumulative_training_backward_example_passes": sum(
            int(row["integrator_backward_example_passes"]) for row in uniform
        ),
        "uniform_cumulative_training_forward_example_passes": sum(
            int(row["total_training_forward_example_passes"]) for row in uniform
        ),
        "uniform_cumulative_training_wall_seconds": sum(
            float(row["total_training_wall_seconds"]) for row in uniform
        ),
    }


def _policy_root(
    config: VampLogTDenseConfig,
    run_root: Path,
    capacity: int,
) -> Path:
    """Return the artifact root for one consolidation policy."""
    if config.scaling is None:
        return run_root if capacity == 1 else run_root / "policies" / POLICY_NAMES[capacity]
    return run_root / "policies" / POLICY_NAMES[capacity]


def _is_capacity_study(config: VampLogTDenseConfig) -> bool:
    """Return whether the resolved protocol is the large-model comparison."""
    return config.protocol_revision == CAPACITY_REVISION


def _is_sample_calibration_study(config: VampLogTDenseConfig) -> bool:
    """Return whether sample count and full-replay epochs are calibrated."""
    return config.protocol_revision == SAMPLE_CALIBRATION_REVISION


def _uses_sample_multiplier_study(config: VampLogTDenseConfig) -> bool:
    """Return whether checkpoints authenticate a training-sample multiplier."""
    return _is_capacity_study(config) or _is_sample_calibration_study(config)


def _sample_calibration_arm_name(
    config: VampLogTDenseConfig,
    training_sample_multiplier: int,
) -> str:
    """Return the stable identifier for one reference-model sample candidate."""
    return (
        "reference_model_"
        f"{config.benchmark.observer_batch_size * training_sample_multiplier}_samples"
    )


def _full_replay_state_name(
    condition: str,
    training_sample_multiplier: int,
    macro_step: int,
) -> str:
    """Return a stable initialization name shared by calibration and final fits."""
    sample_coordinate = (
        f"-samples-{training_sample_multiplier}x"
        if condition == CALIBRATED_FULL_REPLAY_CONDITION
        else ""
    )
    return f"{condition}{sample_coordinate}-step-{macro_step}"


def _arm_name(
    config: VampLogTDenseConfig,
    capacity: int,
    training_sample_multiplier: int,
) -> str:
    """Return one stable report/artifact arm identifier."""
    if _is_capacity_study(config):
        if capacity != 1 or training_sample_multiplier not in CAPACITY_ARM_NAMES:
            raise ValueError("unknown capacity-study arm")
        return CAPACITY_ARM_NAMES[training_sample_multiplier]
    if _is_sample_calibration_study(config):
        if capacity != 1:
            raise ValueError("sample-calibrated study requires one node per level")
        return _sample_calibration_arm_name(config, training_sample_multiplier)
    return POLICY_NAMES[capacity]


def _arm_label(arm: str) -> str:
    """Return a literal user-facing arm label."""
    if arm in CAPACITY_ARM_LABELS:
        return CAPACITY_ARM_LABELS[arm]
    if arm.startswith("reference_model_") and arm.endswith("_samples"):
        sample_count = arm.removeprefix("reference_model_").removesuffix("_samples")
        return f"Reference model, {sample_count} samples per role and task"
    return arm.replace("_", " ")


def _arm_root(
    config: VampLogTDenseConfig,
    run_root: Path,
    training_sample_multiplier: int,
) -> Path:
    """Return the artifact root for one authenticated sample-count arm."""
    if not _uses_sample_multiplier_study(config):
        raise ValueError("sample arms require a sample-count protocol")
    return run_root / "arms" / _arm_name(config, 1, training_sample_multiplier)


def _all_metric_rows(
    config: VampLogTDenseConfig,
    run_root: Path,
) -> tuple[Mapping[str, object], ...]:
    """Load every completed policy/seed ledger in deterministic order."""
    if _is_capacity_study(config):
        if config.scaling is None:
            raise ValueError("capacity study lacks sample arms")
        coordinates = tuple(
            (1, seed, multiplier)
            for multiplier in config.scaling.training_sample_multipliers
            for seed in config.online.seeds
        )
    else:
        if config.scaling is None:
            capacities = (1,)
        else:
            capacities = config.scaling.hierarchy_node_capacities
        coordinates = tuple(
            (capacity, seed, 1)
            for capacity in capacities
            for seed in config.online.seeds
        )
    rows = []
    for capacity, seed, multiplier in coordinates:
        root = (
            _arm_root(config, run_root, multiplier)
            if _is_capacity_study(config)
            else _policy_root(config, run_root, capacity)
        )
        directory = root / "scaling" / f"seed-{seed}"
        summary = load_canonical_json(directory / "summary.json")
        ledger = ChainedJsonlLedger(
            directory / "metrics.jsonl",
            "vamp-logt-scaling-metric-v2",
        )
        if (
            summary.get("status") != "complete"
            or int(summary.get("metric_rows", -1)) != ledger.next_sequence
            or int(summary.get("training_sample_multiplier", 1)) != multiplier
        ):
            raise ValueError("aggregate requested before every scaling seed completed")
        rows.extend(ledger.rows)
    return tuple(rows)


def _aggregate_summary(
    config: VampLogTDenseConfig,
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Validate and summarize all policy/seed results."""
    if config.scaling is None:
        raise ValueError("aggregate scaling summary requires the comparison amendment")
    conditions = tuple(row for row in rows if row.get("row_type") == "condition")
    hierarchy = tuple(
        row for row in rows if row.get("row_type") == "shared_hierarchy_work"
    )
    capacities = config.scaling.hierarchy_node_capacities
    expected_uniform = {
        (capacity, seed, step)
        for capacity in capacities
        for seed in config.online.seeds
        for step in range(1, config.benchmark.macro_steps + 1)
    }
    expected_full = {
        (capacity, seed, step)
        for capacity in capacities
        for seed in config.online.seeds
        for step in config.evaluation.full_checkpoints
    }
    actual = lambda condition: {
        (
            int(row["max_nodes_per_level"]),
            int(row["run_seed"]),
            int(row["macro_step"]),
        )
        for row in conditions
        if row["condition"] == condition
    }
    acceptance = {
        "all_metrics_finite": all(
            math.isfinite(float(row[name]))
            for row in conditions
            for name in ("accuracy", "cross_entropy", "total_training_wall_seconds")
        ),
        "capacity_one_seed_zero_reproduces_predecessor": _predecessor_matches(
            config,
            conditions,
        ),
        "evaluation_excluded_from_training_work": all(
            int(row["total_training_forward_example_passes"])
            == int(row["feature_forward_example_passes"])
            + int(row["integrator_forward_example_passes"])
            for row in conditions
        ),
        "five_seeds_exact": config.online.seeds == (0, 1, 2, 3, 4),
        "full_replay_cells_exact": actual(FULL_REPLAY_CONDITION) == expected_full,
        "full_replay_exactly_twenty_epochs": all(
            int(row["epochs"]) == config.integrator.offline_epochs
            for row in conditions
            if row["condition"] == FULL_REPLAY_CONDITION
        ),
        "hierarchy_cells_exact": {
            (
                int(row["max_nodes_per_level"]),
                int(row["run_seed"]),
                int(row["macro_step"]),
            )
            for row in hierarchy
        }
        == expected_uniform,
        "policy_capacities_exact": capacities == (1, 2),
        "uniform_cells_exact": actual(UNIFORM_CONDITION) == expected_uniform,
        "uniform_exact_replay_budget": all(
            int(row["historical_examples"])
            == (0 if int(row["macro_step"]) == 1 else config.online.historical_budget)
            for row in conditions
            if row["condition"] == UNIFORM_CONDITION
        ),
    }
    policy_summaries = {}
    for capacity in capacities:
        policy_conditions = tuple(
            row for row in conditions if int(row["max_nodes_per_level"]) == capacity
        )
        policy_hierarchy = tuple(
            row for row in hierarchy if int(row["max_nodes_per_level"]) == capacity
        )
        seed_hierarchy_totals = tuple(
            sum(
                int(row["shared_forward_example_passes"])
                for row in policy_hierarchy
                if int(row["run_seed"]) == seed
            )
            for seed in config.online.seeds
        )
        policy_summaries[POLICY_NAMES[capacity]] = {
            "condition_rows": len(policy_conditions),
            "hierarchy_forward_example_passes_per_seed": seed_hierarchy_totals[0],
            "hierarchy_backward_example_passes_per_seed": seed_hierarchy_totals[0],
            "uniform_forward_example_passes_mean_per_seed": fmean(
                sum(
                    int(row["total_training_forward_example_passes"])
                    for row in policy_conditions
                    if row["condition"] == UNIFORM_CONDITION
                    and int(row["run_seed"]) == seed
                )
                for seed in config.online.seeds
            ),
            "uniform_backward_example_passes_mean_per_seed": fmean(
                sum(
                    int(row["integrator_backward_example_passes"])
                    for row in policy_conditions
                    if row["condition"] == UNIFORM_CONDITION
                    and int(row["run_seed"]) == seed
                )
                for seed in config.online.seeds
            ),
        }
    return {
        "acceptance": acceptance,
        "condition_rows": len(conditions),
        "config_hash": config.config_hash,
        "final_macro_step": config.benchmark.macro_steps,
        "full_replay_checkpoints": list(config.evaluation.full_checkpoints),
        "metric_rows": len(rows),
        "policies": policy_summaries,
        "run_seeds": list(config.online.seeds),
        "schema_version": "vamp-logt-scaling-aggregate-summary-v2",
        "status": "complete" if all(acceptance.values()) else "failed_acceptance",
    }


def _predecessor_matches(
    config: VampLogTDenseConfig,
    rows: Sequence[Mapping[str, object]],
) -> bool:
    """Check exact non-timing reproduction of the predecessor's seed-zero rows."""
    if config.scaling is None:
        return True
    path = config.scaling.predecessor_run / "work_metrics.csv"
    predecessor = {
        (row["condition"], int(row["learned_permutation_count"])): row
        for row in csv.DictReader(path.open(encoding="utf-8"))
    }
    reproduced = {
        (str(row["condition"]), int(row["learned_permutation_count"])): row
        for row in rows
        if int(row["max_nodes_per_level"]) == 1 and int(row["run_seed"]) == 0
    }
    numeric_fields = (
        "active_node_count",
        "accuracy",
        "cross_entropy",
        "training_examples",
        "current_examples",
        "historical_examples",
        "epochs",
        "total_training_forward_example_passes",
        "feature_forward_example_passes",
        "integrator_forward_example_passes",
        "integrator_backward_example_passes",
        "feature_forward_calls",
        "integrator_forward_calls",
        "integrator_backward_calls",
    )
    return set(predecessor) == set(reproduced) and all(
        all(float(old[field]) == float(reproduced[key][field]) for field in numeric_fields)
        for key, old in predecessor.items()
    )


def _reference_condition_rows(
    config: VampLogTDenseConfig,
) -> tuple[Mapping[str, object], ...]:
    """Load the authenticated seed-zero, one-node predecessor cells."""
    if not _is_capacity_study(config) or config.scaling is None:
        raise ValueError("reference rows belong only to the capacity study")
    path = config.scaling.predecessor_run / "work_metrics.csv"
    with path.open(encoding="utf-8", newline="") as source:
        rows = tuple(
            row
            for row in csv.DictReader(source)
            if int(row["max_nodes_per_level"]) == 1 and int(row["run_seed"]) == 0
        )
    decorated = []
    for source_row in rows:
        row: dict[str, object] = dict(source_row)
        row.update({
            "arm": REFERENCE_ARM,
            "arm_label": CAPACITY_ARM_LABELS[REFERENCE_ARM],
            "base_hidden_widths": [1024, 1024, 512],
            "base_parameter_count": REFERENCE_BASE_PARAMETER_COUNT,
            "macro_step": int(source_row["learned_permutation_count"]),
            "result_source": "authenticated_predecessor",
            "row_type": "condition",
            "training_sample_multiplier": 1,
        })
        decorated.append(row)
    return tuple(decorated)


def _capacity_condition_rows(
    config: VampLogTDenseConfig,
    run_root: Path,
) -> tuple[Mapping[str, object], ...]:
    """Combine authenticated reference cells with current large-model cells."""
    generated = tuple(
        {
            **row,
            "result_source": "current_run",
        }
        for row in _all_metric_rows(config, run_root)
        if row.get("row_type") == "condition"
    )
    return (*_reference_condition_rows(config), *generated)


def _capacity_cell(
    rows: Sequence[Mapping[str, object]],
    arm: str,
    condition: str,
    step: int,
    field: str,
) -> float:
    """Return one unique single-seed capacity-study metric cell."""
    values = tuple(
        float(row[field])
        for row in rows
        if row["arm"] == arm
        and row["condition"] == condition
        and int(row["macro_step"]) == step
    )
    if len(values) != 1:
        raise ValueError(f"capacity metric cell is not unique: {arm}, {condition}, {step}")
    return values[0]


def _condition_series(
    rows: Sequence[Mapping[str, object]],
    condition: str,
    field: str,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Return one ordered single-arm condition series."""
    selected = sorted(
        (
            (int(row["macro_step"]), float(row[field]))
            for row in rows
            if row["condition"] == condition
        ),
        key=lambda item: item[0],
    )
    if not selected or len({step for step, _ in selected}) != len(selected):
        raise ValueError("condition series is empty or contains duplicate steps")
    return tuple(step for step, _ in selected), tuple(value for _, value in selected)


def _condition_cumulative_series(
    rows: Sequence[Mapping[str, object]],
    condition: str,
    field: str,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Return cumulative values for one ordered single-arm condition series."""
    steps, values = _condition_series(rows, condition, field)
    return steps, tuple(accumulate(values))


def _capacity_series(
    rows: Sequence[Mapping[str, object]],
    arm: str,
    condition: str,
    field: str,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Return one ordered single-seed capacity-study series."""
    selected = sorted(
        (
            (int(row["macro_step"]), float(row[field]))
            for row in rows
            if row["arm"] == arm and row["condition"] == condition
        ),
        key=lambda item: item[0],
    )
    if not selected or len({step for step, _ in selected}) != len(selected):
        raise ValueError("capacity series is empty or contains duplicate steps")
    return tuple(step for step, _ in selected), tuple(value for _, value in selected)


def _capacity_cumulative_series(
    rows: Sequence[Mapping[str, object]],
    arm: str,
    condition: str,
    field: str,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Return cumulative values for one ordered capacity-study series."""
    steps, values = _capacity_series(rows, arm, condition, field)
    return steps, tuple(accumulate(values))


def _growth_scale(step: int | float) -> float:
    """Return the finite t log2(t + 1) comparison scale."""
    if step <= 0:
        raise ValueError("growth scale requires a positive step")
    return float(step) * math.log2(float(step) + 1.0)


def _r_squared(values: Sequence[float], predictions: Sequence[float]) -> float:
    """Return ordinary R-squared on the original measurement scale."""
    if len(values) != len(predictions) or not values:
        raise ValueError("R-squared requires aligned nonempty values")
    mean = fmean(values)
    residual = sum((value - prediction) ** 2 for value, prediction in zip(values, predictions, strict=True))
    total = sum((value - mean) ** 2 for value in values)
    if total == 0.0:
        return 1.0 if residual == 0.0 else 0.0
    return 1.0 - residual / total


def _growth_fits(
    steps: Sequence[int],
    values: Sequence[float],
) -> dict[str, object]:
    """Fit through-origin t-log-t and empirical power curves from step four."""
    selected = tuple(
        (float(step), float(value))
        for step, value in zip(steps, values, strict=True)
        if step >= FIT_MINIMUM_STEP and value > 0.0
    )
    if len(selected) < 3:
        raise ValueError("growth fitting requires at least three positive checkpoints")
    fit_steps = tuple(step for step, _ in selected)
    fit_values = tuple(value for _, value in selected)
    scales = tuple(_growth_scale(step) for step in fit_steps)
    scale_coefficient = sum(
        scale * value for scale, value in zip(scales, fit_values, strict=True)
    ) / sum(scale * scale for scale in scales)
    scale_predictions = tuple(scale_coefficient * scale for scale in scales)

    log_steps = tuple(math.log(step) for step in fit_steps)
    log_values = tuple(math.log(value) for value in fit_values)
    mean_log_step = fmean(log_steps)
    mean_log_value = fmean(log_values)
    exponent = sum(
        (step - mean_log_step) * (value - mean_log_value)
        for step, value in zip(log_steps, log_values, strict=True)
    ) / sum((step - mean_log_step) ** 2 for step in log_steps)
    power_coefficient = math.exp(mean_log_value - exponent * mean_log_step)
    power_predictions = tuple(
        power_coefficient * step**exponent for step in fit_steps
    )
    return {
        "fit_minimum_step": FIT_MINIMUM_STEP,
        "power_law": {
            "coefficient": power_coefficient,
            "exponent": exponent,
            "r_squared": _r_squared(fit_values, power_predictions),
        },
        "t_log2_t_plus_1": {
            "coefficient": scale_coefficient,
            "r_squared": _r_squared(fit_values, scale_predictions),
        },
    }


def _capacity_summary(
    config: VampLogTDenseConfig,
    run_root: Path,
    rows: Sequence[Mapping[str, object]],
    hierarchy: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Validate the two new arms and summarize their reference contrasts."""
    if config.scaling is None:
        raise ValueError("capacity summary requires scaling coordinates")
    new_arms = tuple(CAPACITY_ARM_NAMES[m] for m in config.scaling.training_sample_multipliers)
    all_arms = (REFERENCE_ARM, *new_arms)
    conditions = (UNIFORM_CONDITION, FULL_REPLAY_CONDITION)
    expected_uniform = {
        (arm, step)
        for arm in all_arms
        for step in range(1, config.benchmark.macro_steps + 1)
    }
    expected_full = {
        (arm, step)
        for arm in all_arms
        for step in config.evaluation.full_checkpoints
    }
    actual = lambda condition: {
        (str(row["arm"]), int(row["macro_step"]))
        for row in rows
        if row["condition"] == condition
    }
    generated = tuple(row for row in rows if row["result_source"] == "current_run")
    reference = tuple(
        row for row in rows if row["result_source"] == "authenticated_predecessor"
    )
    sample_by_arm = {
        CAPACITY_ARM_NAMES[multiplier]: multiplier
        for multiplier in config.scaling.training_sample_multipliers
    }
    acceptance = {
        "all_metrics_finite": all(
            math.isfinite(float(row[field]))
            for row in rows
            for field in ("accuracy", "cross_entropy", "total_training_wall_seconds")
        ),
        "base_parameter_target_exact": all(
            int(row["base_parameter_count"])
            == (
                REFERENCE_BASE_PARAMETER_COUNT
                if row["arm"] == REFERENCE_ARM
                else LARGE_BASE_PARAMETER_COUNT
            )
            for row in rows
        ),
        "full_replay_cells_exact": actual(FULL_REPLAY_CONDITION) == expected_full,
        "full_replay_exactly_twenty_epochs": all(
            int(row["epochs"]) == config.integrator.offline_epochs
            for row in rows
            if row["condition"] == FULL_REPLAY_CONDITION
        ),
        "generated_sample_counts_exact": all(
            (
                int(row["current_examples"])
                == config.benchmark.observer_batch_size * sample_by_arm[str(row["arm"])]
                and int(row["historical_examples"])
                == (
                    0
                    if int(row["macro_step"]) == 1
                    else config.online.historical_budget * sample_by_arm[str(row["arm"])]
                )
            )
            if row["condition"] == UNIFORM_CONDITION
            else int(row["training_examples"])
            == (
                config.benchmark.observer_batch_size
                * sample_by_arm[str(row["arm"])]
                * int(row["macro_step"])
            )
            for row in generated
        ),
        "hierarchy_cells_exact": {
            (str(row["arm"]), int(row["macro_step"])) for row in hierarchy
        }
        == {
            (arm, step)
            for arm in new_arms
            for step in range(1, config.benchmark.macro_steps + 1)
        },
        "integrator_parameter_target_exact": all(
            int(row["integrator_parameter_count"])
            == (
                REFERENCE_INTEGRATOR_PARAMETER_COUNT
                if row["arm"] == REFERENCE_ARM
                else LARGE_INTEGRATOR_PARAMETER_COUNT
            )
            for row in rows
        ),
        "one_node_per_level_only": all(
            int(row["max_nodes_per_level"]) == 1 for row in (*rows, *hierarchy)
        ),
        "reference_cells_exact": len(reference)
        == config.benchmark.macro_steps + len(config.evaluation.full_checkpoints),
        "single_seed_exact": all(int(row["run_seed"]) == 0 for row in (*rows, *hierarchy)),
        "uniform_cells_exact": actual(UNIFORM_CONDITION) == expected_uniform,
    }
    fits = {}
    for arm in all_arms:
        full_steps, full_wall = _capacity_series(
            rows,
            arm,
            FULL_REPLAY_CONDITION,
            "total_training_wall_seconds",
        )
        _, full_forward = _capacity_series(
            rows,
            arm,
            FULL_REPLAY_CONDITION,
            "total_training_forward_example_passes",
        )
        persistent_steps, persistent_wall = _capacity_cumulative_series(
            rows,
            arm,
            UNIFORM_CONDITION,
            "total_training_wall_seconds",
        )
        _, persistent_forward = _capacity_cumulative_series(
            rows,
            arm,
            UNIFORM_CONDITION,
            "total_training_forward_example_passes",
        )
        fits[arm] = {
            "persistent_uniform_cumulative_forward_example_passes": _growth_fits(
                persistent_steps,
                persistent_forward,
            ),
            "persistent_uniform_cumulative_training_wall_seconds": _growth_fits(
                persistent_steps,
                persistent_wall,
            ),
            "full_replay_forward_example_passes": _growth_fits(
                full_steps,
                full_forward,
            ),
            "full_replay_training_wall_seconds": _growth_fits(
                full_steps,
                full_wall,
            ),
        }
    final_step = config.benchmark.macro_steps
    final_accuracy = {
        arm: {
            condition: _capacity_cell(rows, arm, condition, final_step, "accuracy")
            for condition in conditions
        }
        for arm in all_arms
    }
    contrasts = {
        "capacity_effect_large_standard_minus_reference_percentage_points": {
            condition: 100.0
            * (
                final_accuracy[CAPACITY_ARM_NAMES[1]][condition]
                - final_accuracy[REFERENCE_ARM][condition]
            )
            for condition in conditions
        },
        "sample_effect_large_double_minus_large_standard_percentage_points": {
            condition: 100.0
            * (
                final_accuracy[CAPACITY_ARM_NAMES[2]][condition]
                - final_accuracy[CAPACITY_ARM_NAMES[1]][condition]
            )
            for condition in conditions
        },
    }
    calibration = load_canonical_json(run_root / "calibration" / "summary.json")
    return {
        "acceptance": acceptance,
        "calibration": {
            "best_epoch": calibration["best_epoch"],
            "epochs_ran": calibration["epochs_ran"],
            "identity_test_accuracy": calibration["identity_test_accuracy"],
            "parameter_count": calibration["parameter_count"],
            "stop_reason": calibration["stop_reason"],
            "validation_accuracy_at_best_epoch": calibration[
                "validation_accuracy_at_best_epoch"
            ],
        },
        "condition_rows": len(rows),
        "config_hash": config.config_hash,
        "contrasts_at_step_100": contrasts,
        "empirical_growth_fits": fits,
        "final_accuracy": final_accuracy,
        "full_replay_checkpoints": list(config.evaluation.full_checkpoints),
        "metric_rows_generated": len(generated),
        "metric_rows_reference": len(reference),
        "run_seeds": list(config.online.seeds),
        "report_revision": "both-condition-runtime-fits-v2",
        "schema_version": "vamp-logt-scaling-capacity-summary-v2",
        "status": "complete" if all(acceptance.values()) else "failed_acceptance",
    }


def _write_sample_calibration_stop_report(
    config: VampLogTDenseConfig,
    run_root: Path,
    selection: Mapping[str, object],
) -> None:
    """Write a visible terminal report when the fixed sample ladder fails."""
    candidate_lines = "\n".join(
        "| "
        f"{int(candidate['sample_count_per_role'])} | "
        f"{100 * float(candidate['best_minimum_prefix_accuracy']):.2f}% | "
        f"{int(candidate['best_epoch_by_minimum_prefix_accuracy'])} | "
        f"{candidate['status']} |"
        for candidate in selection["candidate_summaries"]
    )
    markdown = f"""# Sample-calibrated 100-permutation integrator

## Outcome

Calibration stopped before the 100-task experiment. None of the declared
reference-model candidates kept learned-domain held-out accuracy at or above
{100 * float(selection['target_accuracy']):.1f}% at every prefix from task 1
through task {config.sample_calibration.prefix_steps if config.sample_calibration else 10}.
No final test examples were evaluated and no larger model was selected
silently.

| Samples per model/observer role and task | Best minimum prefix accuracy | Epoch | Status |
|---:|---:|---:|---|
{candidate_lines}

## Interpretation boundary

This result rejects only the declared reference-sized model and sample ladder
through 32x samples with at most 20 full-replay epochs. It does not establish
that a larger network or different optimizer cannot reach the target.
"""
    atomic_write(run_root / "RESULTS.md", markdown.encode("utf-8"))
    atomic_write(
        run_root / "RESULTS.html",
        _html(markdown, run_root, "Sample-calibrated integrator stop").encode("utf-8"),
    )
    atomic_write(run_root / "summary.json", canonical_json_bytes(dict(selection)))


def _write_sample_calibrated_report(
    config: VampLogTDenseConfig,
    run_root: Path,
) -> None:
    """Write calibrated-condition evidence, figures, and standalone reports."""
    selection = load_canonical_json(run_root / "calibration" / "sample_selection.json")
    schedule = load_canonical_json(run_root / "full_replay_schedule.json")
    sample_multiplier = int(selection["selected_sample_multiplier"])
    arm_root = _arm_root(config, run_root, sample_multiplier)
    ledger = ChainedJsonlLedger(
        arm_root / "scaling" / "seed-0" / "metrics.jsonl",
        "vamp-logt-scaling-metric-v2",
    )
    condition_rows = tuple(
        row for row in ledger.rows if row.get("row_type") == "condition"
    )
    hierarchy_rows = tuple(
        row
        for row in ledger.rows
        if row.get("row_type") == "shared_hierarchy_work"
    )
    summary = _sample_calibrated_summary(
        config,
        run_root,
        condition_rows,
        hierarchy_rows,
    )
    _write_metrics_csv(run_root / "work_metrics.csv", condition_rows)
    _write_hierarchy_csv(run_root / "hierarchy_work.csv", hierarchy_rows)
    _write_sample_calibration_csv(config, run_root, selection)
    _write_sample_calibrated_plots(config, run_root, condition_rows, selection)
    markdown = _sample_calibrated_report_markdown(
        config,
        summary,
        condition_rows,
        selection,
        schedule,
    )
    atomic_write(run_root / "RESULTS.md", markdown.encode("utf-8"))
    atomic_write(
        run_root / "RESULTS.html",
        _html(
            markdown,
            run_root,
            "Sample-calibrated 100-permutation integrator comparison",
        ).encode("utf-8"),
    )
    atomic_write(run_root / "summary.json", canonical_json_bytes(summary))


def _sample_calibrated_summary(
    config: VampLogTDenseConfig,
    run_root: Path,
    rows: Sequence[Mapping[str, object]],
    hierarchy: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Validate the selected arm and summarize its accuracy and counted work."""
    selection = load_canonical_json(run_root / "calibration" / "sample_selection.json")
    schedule = load_canonical_json(run_root / "full_replay_schedule.json")
    endpoint = load_canonical_json(run_root / "endpoint_probe.json")
    protocol_amendment = load_canonical_json(
        run_root / "protocol-amendment-oom-streaming.json"
    )
    sample_multiplier = int(selection["selected_sample_multiplier"])
    sample_count = config.benchmark.observer_batch_size * sample_multiplier
    epoch_budget = int(selection["selected_full_replay_epochs"])
    scheduled_checkpoints = tuple(int(value) for value in schedule["checkpoints"])
    uniform = tuple(row for row in rows if row["condition"] == UNIFORM_CONDITION)
    full = tuple(
        row
        for row in rows
        if row["condition"] == CALIBRATED_FULL_REPLAY_CONDITION
    )
    selected_candidate = tuple(selection["candidate_summaries"])[-1]
    earlier_candidates = tuple(selection["candidate_summaries"][:-1])
    expected_uniform_steps = tuple(range(1, config.benchmark.macro_steps + 1))
    final_full = tuple(
        row for row in full if int(row["macro_step"]) == config.benchmark.macro_steps
    )
    acceptance = {
        "all_metrics_finite": all(
            math.isfinite(float(row[field]))
            for row in rows
            for field in ("accuracy", "cross_entropy", "total_training_wall_seconds")
        ),
        "bounded_memory_full_replay": all(
            row.get("feature_storage") == "temporary_float32_memory_map"
            and 0 < int(row.get("peak_resident_feature_rows", 0)) <= sample_count
            and int(row.get("temporary_feature_cache_bytes", 0)) > 0
            for row in full
        ),
        "calibration_passes_every_prefix": (
            selected_candidate["status"] == "passing"
            and min(
                float(value)
                for value in selected_candidate["prefix_accuracies_at_selected_epoch"]
            )
            >= float(selection["target_accuracy"])
        ),
        "earlier_sample_candidates_failed": all(
            candidate["status"] != "passing" for candidate in earlier_candidates
        ),
        "full_replay_cells_match_frozen_schedule": tuple(
            int(row["macro_step"]) for row in full
        )
        == scheduled_checkpoints,
        "full_replay_epoch_budget_exact": all(
            int(row["epochs"]) == epoch_budget for row in full
        ),
        "full_replay_work_exact": all(
            int(row["total_training_forward_example_passes"])
            + int(row["integrator_backward_example_passes"])
            == _full_replay_model_passes(
                config,
                int(row["macro_step"]),
                sample_multiplier,
                epoch_budget,
            )
            for row in full
        ),
        "hierarchy_cells_exact": tuple(int(row["macro_step"]) for row in hierarchy)
        == expected_uniform_steps,
        "optional_checkpoint_decision_matches_projection": bool(
            schedule["optional_checkpoints_included"]
        )
        == (
            float(schedule["projected_with_optional_seconds"])
            <= float(schedule["time_budget_seconds"])
        ),
        "reference_model_selected": selection["selected_model"]
        == "reference_1024_1024_512",
        "sample_counts_exact": all(
            (
                int(row["current_examples"]) == sample_count
                and int(row["historical_examples"])
                == (0 if int(row["macro_step"]) == 1 else sample_count)
            )
            if row["condition"] == UNIFORM_CONDITION
            else int(row["training_examples"])
            == sample_count * int(row["macro_step"])
            for row in rows
        ),
        "task_100_endpoint_matches_final_row": (
            len(final_full) == 1
            and float(final_full[0]["accuracy"]) == float(endpoint["accuracy"])
            and final_full[0]["checkpoint_sha256"] == endpoint["checkpoint_sha256"]
            and bool(endpoint["trained_before_persistent_condition"])
        ),
        "test_set_sealed_during_selection": int(
            selection["test_evaluations_during_selection"]
        )
        == 0,
        "uniform_every_permutation": tuple(
            int(row["macro_step"]) for row in uniform
        )
        == expected_uniform_steps,
        "uniform_work_exact": all(
            int(row["total_training_forward_example_passes"])
            + int(row["integrator_backward_example_passes"])
            == (
                (sample_count if int(row["macro_step"]) == 1 else 2 * sample_count)
                * (
                    int(row["macro_step"]).bit_count()
                    + 2 * config.integrator.epochs_per_step
                )
            )
            for row in uniform
        ),
    }
    persistent_steps, persistent_wall = _condition_cumulative_series(
        rows,
        UNIFORM_CONDITION,
        "total_training_wall_seconds",
    )
    _, persistent_forward = _condition_cumulative_series(
        rows,
        UNIFORM_CONDITION,
        "total_training_forward_example_passes",
    )
    full_steps, full_wall = _condition_series(
        rows,
        CALIBRATED_FULL_REPLAY_CONDITION,
        "total_training_wall_seconds",
    )
    _, full_forward = _condition_series(
        rows,
        CALIBRATED_FULL_REPLAY_CONDITION,
        "total_training_forward_example_passes",
    )
    return {
        "acceptance": acceptance,
        "calibration": selection,
        "condition_rows": len(rows),
        "config_hash": config.config_hash,
        "empirical_growth_fits": {
            "full_replay_forward_example_passes": _growth_fits(
                full_steps,
                full_forward,
            ),
            "full_replay_training_wall_seconds": _growth_fits(
                full_steps,
                full_wall,
            ),
            "persistent_uniform_cumulative_forward_example_passes": _growth_fits(
                persistent_steps,
                persistent_forward,
            ),
            "persistent_uniform_cumulative_training_wall_seconds": _growth_fits(
                persistent_steps,
                persistent_wall,
            ),
        },
        "endpoint_probe": endpoint,
        "final_accuracy": {
            CALIBRATED_FULL_REPLAY_CONDITION: float(final_full[0]["accuracy"]),
            UNIFORM_CONDITION: float(uniform[-1]["accuracy"]),
        },
        "full_replay_schedule": schedule,
        "protocol_amendment": protocol_amendment,
        "report_revision": "cumulative-t-log-comparisons-v3",
        "run_seed": 0,
        "schema_version": "vamp-logt-sample-calibrated-summary-v3",
        "status": "complete" if all(acceptance.values()) else "failed_acceptance",
    }


def _write_sample_calibration_csv(
    config: VampLogTDenseConfig,
    run_root: Path,
    selection: Mapping[str, object],
) -> None:
    """Flatten candidate prefix-by-epoch validation evidence to one compact CSV."""
    fields = (
        "sample_multiplier",
        "sample_count_per_role",
        "candidate_status",
        "macro_step",
        "epoch",
        "accuracy",
        "cross_entropy",
    )
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for candidate in selection["candidate_summaries"]:
        multiplier = int(candidate["sample_multiplier"])
        ledger = ChainedJsonlLedger(
            run_root / "calibration" / f"samples-{multiplier:02d}x" / "metrics.jsonl",
            "vamp-logt-sample-calibration-metric-v1",
        )
        for row in ledger.rows:
            for epoch in row["epoch_metrics"]:
                writer.writerow({
                    "sample_multiplier": multiplier,
                    "sample_count_per_role": (
                        config.benchmark.observer_batch_size * multiplier
                    ),
                    "candidate_status": candidate["status"],
                    "macro_step": row["macro_step"],
                    "epoch": epoch["epoch"],
                    "accuracy": epoch["accuracy"],
                    "cross_entropy": epoch["cross_entropy"],
                })
    atomic_write(
        run_root / "sample_calibration_metrics.csv",
        output.getvalue().encode("utf-8"),
    )


def _write_sample_calibrated_plots(
    config: VampLogTDenseConfig,
    run_root: Path,
    rows: Sequence[Mapping[str, object]],
    selection: Mapping[str, object],
) -> None:
    """Plot calibration, predecessor accuracy, work scaling, and final topology."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_root = run_root / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    candidates = tuple(selection["candidate_summaries"])
    selected = candidates[-1]
    figure, axes = plt.subplots(1, 2, figsize=(14.5, 5.4), constrained_layout=True)
    candidate_samples = tuple(int(row["sample_count_per_role"]) for row in candidates)
    candidate_minima = tuple(
        100 * float(row["best_minimum_prefix_accuracy"]) for row in candidates
    )
    axes[0].plot(
        candidate_samples,
        candidate_minima,
        color="#0072B2",
        marker="o",
        linewidth=2.4,
        label="Best minimum across prefixes",
    )
    axes[0].axhline(
        100 * float(selection["target_accuracy"]),
        color="#D55E00",
        linestyle="--",
        linewidth=2.0,
        label="95% requirement",
    )
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("Model and observer examples per task")
    axes[0].set_ylabel("Worst prefix held-out accuracy (%)")
    axes[0].set_title("Ascending sample calibration")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False)
    prefix_accuracies = tuple(
        100 * float(value) for value in selected["prefix_accuracies_at_selected_epoch"]
    )
    axes[1].plot(
        range(1, len(prefix_accuracies) + 1),
        prefix_accuracies,
        color="#009E73",
        marker="D",
        linewidth=2.4,
        label=f"Selected epoch {int(selected['selected_epoch'])}",
    )
    axes[1].axhline(
        100 * float(selection["target_accuracy"]),
        color="#D55E00",
        linestyle="--",
        linewidth=2.0,
        label="95% requirement",
    )
    axes[1].set_xlabel("Learned permutations")
    axes[1].set_ylabel("Learned-domain held-out accuracy (%)")
    axes[1].set_title("All ten selected-prefix checks")
    axes[1].set_xticks(range(1, len(prefix_accuracies) + 1))
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False)
    figure.savefig(plot_root / "01_sample_calibration.png", dpi=220)
    plt.close(figure)

    predecessor_path = config.scaling.predecessor_run / "work_metrics.csv"
    with predecessor_path.open(encoding="utf-8", newline="") as source:
        predecessor_rows = tuple(csv.DictReader(source))
    prior_arms = (
        REFERENCE_ARM,
        CAPACITY_ARM_NAMES[1],
        CAPACITY_ARM_NAMES[2],
    )
    arm_styles = {
        REFERENCE_ARM: ("#0072B2", "o"),
        CAPACITY_ARM_NAMES[1]: ("#E69F00", "s"),
        CAPACITY_ARM_NAMES[2]: ("#009E73", "^"),
        "selected": ("#CC79A7", "D"),
    }
    figure, axes = plt.subplots(1, 2, figsize=(16.5, 5.8), constrained_layout=True)
    for axis, condition, title in (
        (axes[0], UNIFORM_CONDITION, "Persistent uniform replay"),
        (axes[1], FULL_REPLAY_CONDITION, "Fresh full replay"),
    ):
        for arm in prior_arms:
            points = tuple(
                (int(row["learned_permutation_count"]), 100 * float(row["accuracy"]))
                for row in predecessor_rows
                if row["arm"] == arm and row["condition"] == condition
            )
            color, marker = arm_styles[arm]
            axis.plot(
                tuple(step for step, _ in points),
                tuple(value for _, value in points),
                color=color,
                marker=marker,
                markevery=10 if condition == UNIFORM_CONDITION else 1,
                linewidth=1.8,
                label=CAPACITY_ARM_LABELS[arm],
            )
        selected_condition = (
            UNIFORM_CONDITION
            if condition == UNIFORM_CONDITION
            else CALIBRATED_FULL_REPLAY_CONDITION
        )
        selected_points = tuple(
            (int(row["macro_step"]), 100 * float(row["accuracy"]))
            for row in rows
            if row["condition"] == selected_condition
        )
        color, marker = arm_styles["selected"]
        axis.plot(
            tuple(step for step, _ in selected_points),
            tuple(value for _, value in selected_points),
            color=color,
            marker=marker,
            markevery=10 if condition == UNIFORM_CONDITION else 1,
            linewidth=2.8,
            label=(
                f"Reference model, {selection['selected_sample_count_per_role']} samples"
            ),
        )
        axis.set_xscale("log", base=2)
        axis.set_xlabel("Learned permutations")
        axis.set_ylabel("Test accuracy over learned domains (%)")
        axis.set_title(title)
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    figure.savefig(plot_root / "02_accuracy_with_calibrated_arm.png", dpi=220)
    plt.close(figure)

    epoch_budget = int(selection["selected_full_replay_epochs"])
    condition_styles = {
        UNIFORM_CONDITION: (
            "#0072B2",
            "o",
            "-",
            "Persistent uniform replay",
        ),
        CALIBRATED_FULL_REPLAY_CONDITION: (
            "#CC79A7",
            "s",
            "--",
            f"Fresh full replay, {epoch_budget} epochs",
        ),
    }
    figure, axes = plt.subplots(1, 3, figsize=(17.5, 5.6), constrained_layout=True)
    quantities = (
        ("total_training_wall_seconds", "Training-only wall time per update (s)"),
        ("total_training_forward_example_passes", "Forward example-passes per update"),
        ("integrator_backward_example_passes", "Backward example-passes per update"),
    )
    for axis, (field, ylabel) in zip(axes, quantities, strict=True):
        for condition in (UNIFORM_CONDITION, CALIBRATED_FULL_REPLAY_CONDITION):
            color, marker, linestyle, label = condition_styles[condition]
            steps, values = _condition_series(rows, condition, field)
            axis.plot(
                steps,
                values,
                color=color,
                marker=marker,
                markersize=4.5 if condition == UNIFORM_CONDITION else 6.0,
                markevery=10 if condition == UNIFORM_CONDITION else 1,
                linestyle=linestyle,
                linewidth=2.2,
                label=label,
            )
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xlabel("Learned permutations")
        axis.set_ylabel(ylabel)
        axis.grid(True, which="both", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle(
        "Absolute condition-specific training work (evaluation excluded)",
        fontsize=14,
    )
    figure.savefig(plot_root / "03_calibrated_training_work.png", dpi=220)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(15.5, 5.8), constrained_layout=True)
    curve_steps = tuple(range(1, config.benchmark.macro_steps + 1))
    fit_specs = (
        (
            axes[0],
            UNIFORM_CONDITION,
            True,
            "Persistent replay cumulative training wall time through T (s)",
            "Observed cumulative runtime",
            "T",
        ),
        (
            axes[1],
            CALIBRATED_FULL_REPLAY_CONDITION,
            False,
            "Fresh full-replay wall time for one fit at t (s)",
            "Observed checkpoint runtime",
            "t",
        ),
    )
    for axis, condition, cumulative, ylabel, observed_label, symbol in fit_specs:
        series = _condition_cumulative_series if cumulative else _condition_series
        steps, values = series(rows, condition, "total_training_wall_seconds")
        fits = _growth_fits(steps, values)
        tlog = fits["t_log2_t_plus_1"]
        power = fits["power_law"]
        color, marker, _, _ = condition_styles[condition]
        axis.plot(
            steps,
            values,
            color=color,
            marker=marker,
            markersize=4.5 if cumulative else 6.5,
            markevery=10 if cumulative else 1,
            linewidth=1.8 if cumulative else 0,
            label=observed_label,
        )
        axis.plot(
            curve_steps,
            tuple(
                float(tlog["coefficient"]) * _growth_scale(step)
                for step in curve_steps
            ),
            color="#222222",
            linestyle="-",
            linewidth=2.0,
            label=(
                f"c·{symbol}·log₂({symbol}+1), "
                f"R²={float(tlog['r_squared']):.3f}"
            ),
        )
        axis.plot(
            curve_steps,
            tuple(
                float(power["coefficient"])
                * step ** float(power["exponent"])
                for step in curve_steps
            ),
            color="#E69F00",
            linestyle="--",
            linewidth=2.0,
            label=(
                f"c·{symbol}^{float(power['exponent']):.2f}, "
                f"R²={float(power['r_squared']):.3f}"
            ),
        )
        axis.axvspan(1, FIT_MINIMUM_STEP, color="#999999", alpha=0.08)
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xlabel("Learned permutations")
        axis.set_ylabel(ylabel)
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    figure.suptitle(
        "Empirical runtime fits (fit uses T or t ≥ 4)",
        fontsize=14,
    )
    figure.savefig(plot_root / "05_runtime_growth_fits.png", dpi=220)
    plt.close(figure)

    sample_count = int(selection["selected_sample_count_per_role"])
    persistent_steps, persistent_wall = _condition_cumulative_series(
        rows,
        UNIFORM_CONDITION,
        "total_training_wall_seconds",
    )
    _, persistent_features = _condition_cumulative_series(
        rows,
        UNIFORM_CONDITION,
        "feature_forward_example_passes",
    )
    _, persistent_backward = _condition_cumulative_series(
        rows,
        UNIFORM_CONDITION,
        "integrator_backward_example_passes",
    )
    full_steps, full_wall = _condition_series(
        rows,
        CALIBRATED_FULL_REPLAY_CONDITION,
        "total_training_wall_seconds",
    )
    _, full_features = _condition_series(
        rows,
        CALIBRATED_FULL_REPLAY_CONDITION,
        "feature_forward_example_passes",
    )
    _, full_backward = _condition_series(
        rows,
        CALIBRATED_FULL_REPLAY_CONDITION,
        "integrator_backward_example_passes",
    )
    figure, axes = plt.subplots(2, 3, figsize=(17.5, 9.0), constrained_layout=True)
    normalized_specs = (
        (
            axes[0, 0],
            persistent_steps,
            tuple(
                value / _growth_scale(step)
                for step, value in zip(
                    persistent_steps,
                    persistent_wall,
                    strict=True,
                )
            ),
            "Cumulative wall / [T log₂(T+1)]",
            "Persistent replay — wall time",
            "#0072B2",
            "o",
            None,
        ),
        (
            axes[0, 1],
            persistent_steps,
            tuple(
                value / (sample_count * _growth_scale(step))
                for step, value in zip(
                    persistent_steps,
                    persistent_features,
                    strict=True,
                )
            ),
            "Cumulative frozen forwards / [N T log₂(T+1)]",
            "Persistent replay — frozen features",
            "#0072B2",
            "o",
            None,
        ),
        (
            axes[0, 2],
            persistent_steps,
            tuple(
                value
                / (
                    config.integrator.epochs_per_step
                    * sample_count
                    * (2 * step - 1)
                )
                for step, value in zip(
                    persistent_steps,
                    persistent_backward,
                    strict=True,
                )
            ),
            "Cumulative backward / [4N(2T−1)]",
            "Persistent replay — integrator backward",
            "#0072B2",
            "o",
            1.0,
        ),
        (
            axes[1, 0],
            full_steps,
            tuple(
                value / _growth_scale(step)
                for step, value in zip(full_steps, full_wall, strict=True)
            ),
            "Fit wall / [t log₂(t+1)]",
            "Fresh full replay — wall time per fit",
            "#CC79A7",
            "s",
            None,
        ),
        (
            axes[1, 1],
            full_steps,
            tuple(
                value / (sample_count * _growth_scale(step))
                for step, value in zip(full_steps, full_features, strict=True)
            ),
            "Frozen forwards / [N t log₂(t+1)]",
            "Fresh full replay — frozen features per fit",
            "#CC79A7",
            "s",
            None,
        ),
        (
            axes[1, 2],
            full_steps,
            tuple(
                value / (epoch_budget * sample_count * step)
                for step, value in zip(full_steps, full_backward, strict=True)
            ),
            f"Backward / [{epoch_budget}N t]",
            "Fresh full replay — integrator backward per fit",
            "#CC79A7",
            "s",
            1.0,
        ),
    )
    for axis, steps, values, ylabel, title, color, marker, reference in normalized_specs:
        axis.plot(
            steps,
            values,
            color=color,
            marker=marker,
            markersize=4.5,
            markevery=10 if len(steps) > 10 else 1,
            linewidth=2.0,
        )
        if reference is not None:
            axis.axhline(reference, color="#222222", linestyle=":", linewidth=1.2)
        axis.set_xscale("log", base=2)
        axis.set_xlabel("Learned permutations")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(True, which="both", alpha=0.25)
    figure.suptitle(
        "Training work normalized by the relevant theoretical growth factor",
        fontsize=14,
        y=1.02,
    )
    figure.savefig(
        plot_root / "06_normalized_runtime_growth.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)

    sample_multiplier = int(selection["selected_sample_multiplier"])
    frontier = load_frontier(
        config,
        run_root,
        0,
        config.benchmark.macro_steps,
        hierarchy_root=_arm_root(config, run_root, sample_multiplier) / "hierarchy",
        training_sample_multiplier=sample_multiplier,
    )
    figure, axis = plt.subplots(figsize=(13.5, 4.8), constrained_layout=True)
    topology_colors = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")
    for index, node in enumerate(frontier.nodes):
        axis.barh(
            node.level,
            node.last_block - node.first_block + 1,
            left=node.first_block + 1,
            height=0.55,
            color=topology_colors[index % len(topology_colors)],
            edgecolor="#202124",
        )
        axis.text(
            (node.first_block + node.last_block + 2) / 2,
            node.level,
            f"{node.first_block + 1}–{node.last_block + 1}",
            ha="center",
            va="center",
            fontsize=9,
        )
    axis.set_xlabel("Permutation-task interval represented by frozen node")
    axis.set_ylabel("Temporal level / stable input slot")
    axis.set_title("Task-100 one-node-per-level frontier")
    axis.set_yticks(range(config.observer.maximum_levels))
    axis.grid(True, axis="x", alpha=0.25)
    figure.savefig(plot_root / "04_final_hierarchy.png", dpi=220)
    plt.close(figure)


def _sample_calibrated_report_markdown(
    config: VampLogTDenseConfig,
    summary: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    selection: Mapping[str, object],
    schedule: Mapping[str, object],
) -> str:
    """Render the sample-calibrated result in direct, self-contained language."""
    sample_count = int(selection["selected_sample_count_per_role"])
    epoch_budget = int(selection["selected_full_replay_epochs"])
    selected = tuple(selection["candidate_summaries"])[-1]
    candidate_lines = "\n".join(
        "| "
        f"{int(candidate['sample_count_per_role'])} | "
        f"{100 * float(candidate['best_minimum_prefix_accuracy']):.2f}% | "
        f"{int(candidate['best_epoch_by_minimum_prefix_accuracy'])} | "
        f"{candidate['status']} |"
        for candidate in selection["candidate_summaries"]
    )
    prefix_lines = "\n".join(
        f"| {step} | {100 * float(accuracy):.2f}% |"
        for step, accuracy in enumerate(
            selected["prefix_accuracies_at_selected_epoch"],
            start=1,
        )
    )
    full_by_step = {
        int(row["macro_step"]): row
        for row in rows
        if row["condition"] == CALIBRATED_FULL_REPLAY_CONDITION
    }
    uniform_by_step = {
        int(row["macro_step"]): row
        for row in rows
        if row["condition"] == UNIFORM_CONDITION
    }
    checkpoint_lines = "\n".join(
        f"| {step} | {100 * float(uniform_by_step[step]['accuracy']):.2f}% | "
        f"{100 * float(full_by_step[step]['accuracy']):.2f}% | "
        f"{float(full_by_step[step]['total_training_wall_seconds']):.2f} |"
        for step in schedule["checkpoints"]
    )
    acceptance_lines = "\n".join(
        f"| {name.replace('_', ' ')} | {value} |"
        for name, value in summary["acceptance"].items()
    )
    final_accuracy = summary["final_accuracy"]
    endpoint_seconds = float(summary["endpoint_probe"]["total_training_wall_seconds"])
    elapsed_through_endpoint = float(schedule["elapsed_seconds_through_endpoint"])
    durable_through_hierarchy = float(schedule["durable_seconds_through_hierarchy"])
    required_total = float(schedule["projected_required_seconds"])
    optional_total = float(schedule["projected_with_optional_seconds"])
    reporting_reserve = float(schedule["reporting_reserve_seconds"])
    remaining_required_training = (
        required_total - elapsed_through_endpoint - reporting_reserve
    )
    optional_training = optional_total - required_total
    fit_summary = summary["empirical_growth_fits"]
    persistent_wall_fits = fit_summary[
        "persistent_uniform_cumulative_training_wall_seconds"
    ]
    persistent_forward_fits = fit_summary[
        "persistent_uniform_cumulative_forward_example_passes"
    ]
    full_wall_fits = fit_summary["full_replay_training_wall_seconds"]
    full_forward_fits = fit_summary["full_replay_forward_example_passes"]
    persistent_steps, persistent_wall = _condition_cumulative_series(
        rows,
        UNIFORM_CONDITION,
        "total_training_wall_seconds",
    )
    _, persistent_features = _condition_cumulative_series(
        rows,
        UNIFORM_CONDITION,
        "feature_forward_example_passes",
    )
    _, persistent_forward = _condition_cumulative_series(
        rows,
        UNIFORM_CONDITION,
        "integrator_forward_example_passes",
    )
    _, persistent_backward = _condition_cumulative_series(
        rows,
        UNIFORM_CONDITION,
        "integrator_backward_example_passes",
    )
    full_steps, full_wall = _condition_series(
        rows,
        CALIBRATED_FULL_REPLAY_CONDITION,
        "total_training_wall_seconds",
    )
    _, full_features = _condition_series(
        rows,
        CALIBRATED_FULL_REPLAY_CONDITION,
        "feature_forward_example_passes",
    )
    _, full_forward = _condition_series(
        rows,
        CALIBRATED_FULL_REPLAY_CONDITION,
        "integrator_forward_example_passes",
    )
    _, full_backward = _condition_series(
        rows,
        CALIBRATED_FULL_REPLAY_CONDITION,
        "integrator_backward_example_passes",
    )
    final_step = config.benchmark.macro_steps
    if persistent_steps[-1] != final_step or full_steps[-1] != final_step:
        raise ValueError("training-work comparison does not reach task 100")
    growth_at_endpoint = _growth_scale(final_step)
    persistent_wall_normalized = persistent_wall[-1] / growth_at_endpoint
    persistent_feature_normalized = persistent_features[-1] / (
        sample_count * growth_at_endpoint
    )
    persistent_backward_normalized = persistent_backward[-1] / (
        config.integrator.epochs_per_step
        * sample_count
        * (2 * final_step - 1)
    )
    full_wall_normalized = full_wall[-1] / growth_at_endpoint
    full_feature_normalized = full_features[-1] / (
        sample_count * growth_at_endpoint
    )
    full_backward_normalized = full_backward[-1] / (
        epoch_budget * sample_count * final_step
    )
    persistent_fit_lines = (
        "| Cumulative wall seconds | "
        f"{float(persistent_wall_fits['t_log2_t_plus_1']['coefficient']):.5f} | "
        f"{float(persistent_wall_fits['t_log2_t_plus_1']['r_squared']):.3f} | "
        f"{float(persistent_wall_fits['power_law']['exponent']):.3f} | "
        f"{float(persistent_wall_fits['power_law']['r_squared']):.3f} |\n"
        "| Cumulative total forward example-passes | "
        f"{float(persistent_forward_fits['t_log2_t_plus_1']['coefficient']):.1f} | "
        f"{float(persistent_forward_fits['t_log2_t_plus_1']['r_squared']):.3f} | "
        f"{float(persistent_forward_fits['power_law']['exponent']):.3f} | "
        f"{float(persistent_forward_fits['power_law']['r_squared']):.3f} |"
    )
    full_fit_lines = (
        "| Wall seconds for one fresh fit | "
        f"{float(full_wall_fits['t_log2_t_plus_1']['coefficient']):.5f} | "
        f"{float(full_wall_fits['t_log2_t_plus_1']['r_squared']):.3f} | "
        f"{float(full_wall_fits['power_law']['exponent']):.3f} | "
        f"{float(full_wall_fits['power_law']['r_squared']):.3f} |\n"
        "| Total forward example-passes for one fresh fit | "
        f"{float(full_forward_fits['t_log2_t_plus_1']['coefficient']):.1f} | "
        f"{float(full_forward_fits['t_log2_t_plus_1']['r_squared']):.3f} | "
        f"{float(full_forward_fits['power_law']['exponent']):.3f} | "
        f"{float(full_forward_fits['power_law']['r_squared']):.3f} |"
    )
    endpoint_comparison_lines = "\n".join(
        (
            "| Wall time (s) | "
            f"{persistent_wall[-1]:.2f} | {full_wall[-1]:.2f} | "
            f"{full_wall[-1] / persistent_wall[-1]:.3f} |",
            "| Frozen-feature forward example-passes | "
            f"{persistent_features[-1]:,.0f} | {full_features[-1]:,.0f} | "
            f"{full_features[-1] / persistent_features[-1]:.3f} |",
            "| Integrator forward example-passes | "
            f"{persistent_forward[-1]:,.0f} | {full_forward[-1]:,.0f} | "
            f"{full_forward[-1] / persistent_forward[-1]:.3f} |",
            "| Integrator backward example-passes | "
            f"{persistent_backward[-1]:,.0f} | {full_backward[-1]:,.0f} | "
            f"{full_backward[-1] / persistent_backward[-1]:.3f} |",
        )
    )
    work_checkpoint_lines = []
    for step in schedule["checkpoints"]:
        for condition, label in (
            (UNIFORM_CONDITION, "Persistent uniform replay — one update"),
            (
                CALIBRATED_FULL_REPLAY_CONDITION,
                f"Fresh full replay — one {epoch_budget}-epoch fit",
            ),
        ):
            row = (
                uniform_by_step[int(step)]
                if condition == UNIFORM_CONDITION
                else full_by_step[int(step)]
            )
            work_checkpoint_lines.append(
                f"| {int(step)} | {label} | "
                f"{float(row['total_training_wall_seconds']):.3f} | "
                f"{int(row['total_training_forward_example_passes']):,} | "
                f"{int(row['integrator_backward_example_passes']):,} |"
            )
    return f"""# Sample-calibrated 100-permutation integrator comparison

## Outcome

The reference-sized model passed the calibration with {sample_count} model
examples and {sample_count} disjoint observer examples per task. Fresh full
replay first cleared 95% at all ten prefixes after {epoch_budget} epochs. At
task 100, persistent uniform replay reached
{100 * float(final_accuracy[UNIFORM_CONDITION]):.2f}% test accuracy and fresh
full replay reached
{100 * float(final_accuracy[CALIBRATED_FULL_REPLAY_CONDITION]):.2f}%.
This is one fixed-order seed, so the difference is not a variance estimate.

The task-100 full-replay fit ran before persistent training. It took
{endpoint_seconds:.2f} training
seconds. By then, calibration, hierarchy construction, and the endpoint fit had
already consumed {elapsed_through_endpoint:.2f}
seconds, which exceeded the {float(schedule['time_budget_seconds']):.0f}-second
limit. The four optional fits themselves were projected to add only
{optional_training:.2f} seconds after the 1.25 safety factor. They were
{'included' if schedule['optional_checkpoints_included'] else 'omitted'} because
the complete projected total with them was {optional_total:.2f} seconds.

| Projection component | Seconds |
|---|---:|
| Calibration and hierarchy already elapsed | {durable_through_hierarchy:.2f} |
| Task-100 full-replay endpoint already elapsed | {endpoint_seconds:.2f} |
| Remaining mandatory fits and persistent training, projected with 1.25 safety factor | {remaining_required_training:.2f} |
| Reporting reserve | {reporting_reserve:.2f} |
| Required projected total without optional fits | {required_total:.2f} |
| Four optional fits, projected with 1.25 safety factor | {optional_training:.2f} |
| Projected total with optional fits | {optional_total:.2f} |

Two earlier endpoint attempts were killed before producing a checkpoint because
the old implementation simultaneously held the complete image archive and a
roughly 12 GB dense feature matrix. The corrected implementation computed every
frozen feature exactly once into a temporary float32 memory map, retained at
most {int(summary['endpoint_probe']['peak_resident_feature_rows']):,} feature
rows in anonymous memory, preserved the original seeded minibatch order, and
deleted the cache after checkpoint publication. This storage-only correction is
authenticated by `protocol-amendment-oom-streaming.json`; the completed sample
calibration and frozen hierarchy were retained unchanged.

## Exact condition definitions

| Report name | What was trained |
|---|---|
| Persistent uniform replay | One integrator continued across all 100 tasks. At each task it trained four epochs on {sample_count} current observer examples and, after task 1, {sample_count} examples sampled uniformly from all earlier tasks. Current and historical losses each had weight 0.5. |
| Fresh full replay | A new integrator was initialized at every reported checkpoint and trained {epoch_budget} epochs on all {sample_count} observer examples from every task seen by that checkpoint. |

Both conditions used the same frozen reference MLP, permutation order,
one-node-per-level hierarchy, and test subsets. The smaller model was retained
because it met the calibration requirement; the 4x-parameter model was not
rerun.

## Sample and epoch calibration

| Samples per role and task | Best worst-prefix accuracy | Best epoch | Result |
|---:|---:|---:|---|
{candidate_lines}

The selection used 128 held-out training examples per learned domain. It
required every prefix accuracy below to reach 95% at the same epoch. It did
not evaluate final test examples.

| Prefix task | Held-out accuracy at selected epoch |
|---:|---:|
{prefix_lines}

![Sample and epoch calibration](plots/01_sample_calibration.png)

## Accuracy against the earlier experiment

The two panels separate persistent and fresh-full-replay training so four
model/sample arms remain visually distinct. Earlier full-replay arms used 20
epochs; the new purple arm uses the {epoch_budget}-epoch calibrated budget.

![Accuracy with calibrated arm](plots/02_accuracy_with_calibrated_arm.png)

| Learned tasks | Persistent uniform accuracy | Fresh full-replay accuracy | Fresh full-replay training seconds |
|---:|---:|---:|---:|
{checkpoint_lines}

## Training work

The initial calibrated report replaced the established scaling layout with four
unfitted absolute curves. That made the persistent condition's cumulative
`T log T` comparison invisible. This revision restores the absolute, fitted,
and normalized views used by the preceding capacity report. It changes no
training measurement.

The absolute plot restores the earlier report format: each panel shows both
conditions for the same quantity. A persistent point is the cost of that task's
one update. A fresh-full-replay point is the cost of one newly initialized fit
at that checkpoint. Forward counts include frozen-node feature forwards plus
integrator training forwards; backward counts include only integrator training
backwards.

Evaluation is excluded. Validation passes used to select samples and epochs are
retained separately in `sample_calibration_metrics.csv`. Full-replay wall time
includes temporary-cache writes and shuffled reads; the storage correction does
not change the examples, model passes, or optimizer updates.

![Calibrated training work](plots/03_calibrated_training_work.png)

### Persistent replay: cumulative scaling through task T

Persistent replay performs one update at every task, so its end-to-end cost is
the cumulative sum of all updates through `T`. The table fits observations from
`T >= {FIT_MINIMUM_STEP}`. The through-origin comparison is
`work = c × T × log2(T+1)`; the empirical alternative is `work = c × T^p`.
R-squared is calculated on the original measurement scale.

| Persistent cumulative series | T-log coefficient c | T-log R² | Power p | Power R² |
|---|---:|---:|---:|---:|
{persistent_fit_lines}

Measured cumulative wall time is slightly better described by `T log T`
(R²={float(persistent_wall_fits['t_log2_t_plus_1']['r_squared']):.3f}) than by
the fitted power curve
(R²={float(persistent_wall_fits['power_law']['r_squared']):.3f},
`p={float(persistent_wall_fits['power_law']['exponent']):.3f}`). Counted forward
passes follow the exact popcount schedule rather than a smooth curve; their
finite-range power fit is numerically tighter, but the schedule's asymptotic
bound remains `Theta(T log T)`.

At `T=100`, cumulative persistent wall time was {persistent_wall[-1]:.2f}
seconds. Dividing by `T log2(T+1)` gives
{persistent_wall_normalized:.5f} seconds. Cumulative frozen-node
forwards divided by `N T log2(T+1)` equal
{persistent_feature_normalized:.3f}. Cumulative integrator backwards divided by
their exact count, `4N(2T-1)`, equal {persistent_backward_normalized:.3f}.

### Fresh full replay: cost of one fit at task t

Fresh full replay was measured only at the six scheduled checkpoints. Its
series is therefore the cost of one independent fresh fit at `t`, not a
cumulative sum of fits at every earlier task. Fits again use
`t >= {FIT_MINIMUM_STEP}` and the same through-origin `t log2(t+1)` and power
curves.

| Fresh-fit series | t-log coefficient c | t-log R² | Power p | Power R² |
|---|---:|---:|---:|---:|
{full_fit_lines}

For wall time, the nearly linear power fit
(`p={float(full_wall_fits['power_law']['exponent']):.3f}`,
R²={float(full_wall_fits['power_law']['r_squared']):.3f}) is tighter than the
`t log t` comparison
(R²={float(full_wall_fits['t_log2_t_plus_1']['r_squared']):.3f}). The three
epochs of linear integrator work and memory-map I/O dominate the frozen-feature
term over these tasks. Only four sampled points—tasks 4, 8, 10, and 100—enter
these fits, so the fitted exponent is descriptive rather than a reliable
asymptotic estimate.

At `t=100`, wall time divided by `t log2(t+1)` is
{full_wall_normalized:.5f} seconds. Frozen-node forwards divided by
`N t log2(t+1)` equal {full_feature_normalized:.3f}. Integrator backwards
divided by their exact count, `{epoch_budget}Nt`, equal
{full_backward_normalized:.3f}.

![Empirical runtime fits for persistent and full replay](plots/05_runtime_growth_fits.png)

![Both conditions normalized by their theoretical factors](plots/06_normalized_runtime_growth.png)

The one-node frontier contains `popcount(t)` active frozen nodes. Persistent
replay evaluates a fixed `2N` examples after task 1, so cumulative frozen-node
work is `N + 2N × sum(popcount(k), k=2..T)`, which is `Theta(T log T)`.
Persistent integrator forward/backward work is exactly linear in `T`. One fresh
full-replay fit evaluates `Nt` examples against `popcount(t)` nodes, so its
frozen-node component is `Theta(t log t)` and its integrator component is
linear in `t`. If fresh full replay were actually rerun after every task, its
cumulative cost would instead be `Theta(T² log T)`; this protocol skipped
unsampled fits because they do not affect the independently initialized fits
that were measured.

### Direct endpoint work comparison

The persistent column below is all training accumulated from tasks 1 through
100. The fresh-full-replay column is only the independent task-100 fit. This is
the like-for-purpose endpoint comparison; it is not the cumulative cost of
running fresh full replay at every task.

| Quantity | Persistent cumulative through T=100 | Fresh full replay at t=100 | Fresh / persistent |
|---|---:|---:|---:|
{endpoint_comparison_lines}

### Scheduled checkpoint measurements

| Learned tasks | Condition and scope | Training seconds | Forward example-passes | Backward example-passes |
|---:|---|---:|---:|---:|
{chr(10).join(work_checkpoint_lines)}

## Frozen hierarchy at task 100

Each bar is one active frozen temporal node. Its vertical position is the
stable integrator input slot and its label is the inclusive task interval it
represents.

![Final hierarchy](plots/04_final_hierarchy.png)

## Acceptance checks

| Check | Passed |
|---|---|
{acceptance_lines}

The overall status is {summary['status']}. The resolved config identity is
{config.config_hash}.

## Limits

Calibration on one fixed validation split can overstate how reliably 95% will
hold on other sample draws or permutation orders. Passing tasks 1–10 does not
guarantee 95% at task 100. Fresh full replay is a high-information baseline at
a validation-selected fixed epoch count, not a mathematical best-possible
integrator or a proof of convergence.
"""


def _write_capacity_report(config: VampLogTDenseConfig, run_root: Path) -> None:
    """Write the capacity/sample comparison artifacts from authenticated rows."""
    all_rows = _all_metric_rows(config, run_root)
    hierarchy = tuple(
        row for row in all_rows if row.get("row_type") == "shared_hierarchy_work"
    )
    rows = _capacity_condition_rows(config, run_root)
    summary = _capacity_summary(config, run_root, rows, hierarchy)
    _write_metrics_csv(run_root / "work_metrics.csv", rows)
    _write_hierarchy_csv(run_root / "hierarchy_work.csv", hierarchy)
    _write_capacity_plots(config, run_root, rows)
    markdown = _capacity_report_markdown(config, summary, rows, hierarchy)
    atomic_write(run_root / "RESULTS.md", markdown.encode("utf-8"))
    atomic_write(
        run_root / "RESULTS.html",
        _html(
            markdown,
            run_root,
            "100-permutation model-capacity and sample-count comparison",
        ).encode("utf-8"),
    )
    atomic_write(run_root / "summary.json", canonical_json_bytes(summary))


def _write_aggregate_report(config: VampLogTDenseConfig, run_root: Path) -> None:
    """Write aggregate tables, figures, Markdown, HTML, CSV, and JSON."""
    rows = _all_metric_rows(config, run_root)
    summary = _aggregate_summary(config, rows)
    conditions = tuple(row for row in rows if row.get("row_type") == "condition")
    hierarchy = tuple(
        row for row in rows if row.get("row_type") == "shared_hierarchy_work"
    )
    _write_metrics_csv(run_root / "work_metrics.csv", conditions)
    _write_hierarchy_csv(run_root / "hierarchy_work.csv", hierarchy)
    _write_plots(run_root, conditions, hierarchy)
    markdown = _report_markdown(config, summary, conditions, hierarchy)
    atomic_write(run_root / "RESULTS.md", markdown.encode("utf-8"))
    atomic_write(
        run_root / "RESULTS.html",
        _html(
            markdown,
            run_root,
            "100-permutation five-seed consolidation comparison",
        ).encode("utf-8"),
    )
    atomic_write(run_root / "summary.json", canonical_json_bytes(summary))


def _write_metrics_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = (
        "arm",
        "arm_label",
        "result_source",
        "policy",
        "max_nodes_per_level",
        "run_seed",
        "training_sample_multiplier",
        "base_parameter_count",
        "condition",
        "learned_permutation_count",
        "active_node_count",
        "input_slot_count",
        "integrator_parameter_count",
        "accuracy",
        "cross_entropy",
        "training_examples",
        "current_examples",
        "historical_examples",
        "epochs",
        "total_training_wall_seconds",
        "state_initialization_wall_seconds",
        "data_preparation_wall_seconds",
        "feature_wall_seconds",
        "optimizer_wall_seconds",
        "feature_storage",
        "peak_resident_feature_rows",
        "temporary_feature_cache_bytes",
        "total_training_forward_example_passes",
        "feature_forward_example_passes",
        "integrator_forward_example_passes",
        "integrator_backward_example_passes",
        "feature_forward_calls",
        "integrator_forward_calls",
        "integrator_backward_calls",
        "excluded_diagnostic_forward_example_passes",
        "excluded_evaluation_examples",
        "excluded_evaluation_feature_forward_example_passes",
        "excluded_evaluation_integrator_forward_example_passes_per_condition",
    )
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(path, output.getvalue().encode("utf-8"))


def _write_hierarchy_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = (
        "arm",
        "policy",
        "max_nodes_per_level",
        "run_seed",
        "training_sample_multiplier",
        "learned_permutation_count",
        "created_node_count",
        "shared_forward_example_passes",
        "shared_backward_example_passes",
        "shared_forward_calls",
        "shared_backward_calls",
    )
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(path, output.getvalue().encode("utf-8"))


def _mean_sd(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("cannot summarize an empty metric cell")
    return fmean(values), stdev(values) if len(values) > 1 else 0.0


def _cell_values(
    rows: Sequence[Mapping[str, object]],
    capacity: int,
    condition: str,
    step: int,
    field: str,
) -> tuple[float, ...]:
    values = tuple(
        float(row[field])
        for row in rows
        if int(row["max_nodes_per_level"]) == capacity
        and row["condition"] == condition
        and int(row["macro_step"]) == step
    )
    return values


def _series(
    rows: Sequence[Mapping[str, object]],
    capacity: int,
    condition: str,
    field: str,
) -> tuple[tuple[int, ...], tuple[float, ...], tuple[float, ...]]:
    steps = tuple(sorted({
        int(row["macro_step"])
        for row in rows
        if int(row["max_nodes_per_level"]) == capacity
        and row["condition"] == condition
    }))
    cells = tuple(
        _mean_sd(_cell_values(rows, capacity, condition, step, field))
        for step in steps
    )
    return steps, tuple(cell[0] for cell in cells), tuple(cell[1] for cell in cells)


def _paired_policy_values(
    rows: Sequence[Mapping[str, object]],
    condition: str,
    step: int,
    field: str,
) -> tuple[float, ...]:
    by_coordinate = {
        (int(row["max_nodes_per_level"]), int(row["run_seed"])): float(row[field])
        for row in rows
        if row["condition"] == condition and int(row["macro_step"]) == step
    }
    seeds = tuple(sorted(seed for capacity, seed in by_coordinate if capacity == 1))
    if any((2, seed) not in by_coordinate for seed in seeds):
        raise ValueError("paired policy comparison lacks a seed cell")
    return tuple(by_coordinate[(2, seed)] - by_coordinate[(1, seed)] for seed in seeds)


def _plot_label(capacity: int, condition: str) -> str:
    policy = "1 node/level" if capacity == 1 else "2 nodes/level"
    return f"{policy} — {CONDITION_LABELS[condition]}"


def _capacity_plot_label(arm: str, condition: str) -> str:
    """Return an explicit capacity-study trace label."""
    return f"{CAPACITY_ARM_LABELS[arm]} — {CONDITION_LABELS[condition]}"


def _capacity_arm_multiplier(arm: str) -> int:
    """Return the observer-sample multiplier represented by an arm."""
    if arm in {REFERENCE_ARM, CAPACITY_ARM_NAMES[1]}:
        return 1
    if arm == CAPACITY_ARM_NAMES[2]:
        return 2
    raise ValueError(f"unknown capacity arm: {arm}")


def _write_capacity_plots(
    config: VampLogTDenseConfig,
    run_root: Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    """Plot accuracy, absolute work, empirical fits, and normalized growth."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_root = run_root / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    all_arms = (REFERENCE_ARM, CAPACITY_ARM_NAMES[1], CAPACITY_ARM_NAMES[2])
    conditions = (UNIFORM_CONDITION, FULL_REPLAY_CONDITION)

    figure, axis = plt.subplots(figsize=(11.5, 6.2), constrained_layout=True)
    for arm in all_arms:
        for condition in conditions:
            color, marker, linestyle = CAPACITY_PLOT_STYLES[(arm, condition)]
            steps, values = _capacity_series(rows, arm, condition, "accuracy")
            axis.plot(
                steps,
                tuple(100.0 * value for value in values),
                color=color,
                label=_capacity_plot_label(arm, condition),
                linewidth=2.2,
                marker=marker,
                markersize=4.5 if condition == UNIFORM_CONDITION else 6.0,
                markevery=10 if condition == UNIFORM_CONDITION else 1,
                linestyle=linestyle,
            )
    axis.set_xscale("log", base=2)
    axis.set_xlabel("Learned permutations")
    axis.set_ylabel("Accuracy over equal-size learned-domain test subsets (%)")
    axis.set_title("Capacity and training-sample comparison, seed 0")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(frameon=False, fontsize=8, ncol=2)
    figure.savefig(plot_root / "01_accuracy_capacity_and_samples.png", dpi=220)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(17.5, 5.6), constrained_layout=True)
    quantities = (
        ("total_training_wall_seconds", "Training-only wall time per update (s)"),
        ("total_training_forward_example_passes", "Forward example-passes per update"),
        ("integrator_backward_example_passes", "Backward example-passes per update"),
    )
    for axis, (field, ylabel) in zip(axes, quantities, strict=True):
        for arm in all_arms:
            for condition in conditions:
                color, marker, linestyle = CAPACITY_PLOT_STYLES[(arm, condition)]
                steps, values = _capacity_series(rows, arm, condition, field)
                axis.plot(
                    steps,
                    values,
                    color=color,
                    label=_capacity_plot_label(arm, condition),
                    linewidth=2.0,
                    marker=marker,
                    markersize=4.0 if condition == UNIFORM_CONDITION else 5.5,
                    markevery=10 if condition == UNIFORM_CONDITION else 1,
                    linestyle=linestyle,
                )
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xlabel("Learned permutations")
        axis.set_ylabel(ylabel)
        axis.grid(True, which="both", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=7.5)
    figure.suptitle(
        "Condition-specific training growth (evaluation and hierarchy excluded)",
        fontsize=14,
    )
    figure.savefig(plot_root / "02_training_work_absolute.png", dpi=220)
    plt.close(figure)

    figure, axes = plt.subplots(2, 3, figsize=(17.5, 10.0), constrained_layout=True)
    curve_steps = tuple(range(1, config.benchmark.macro_steps + 1))
    fit_specs = (
        (
            UNIFORM_CONDITION,
            True,
            "Persistent replay cumulative training wall time through t (s)",
            "Observed cumulative runtime",
        ),
        (
            FULL_REPLAY_CONDITION,
            False,
            "Fresh full-replay training wall time for one fit at t (s)",
            "Observed checkpoint runtime",
        ),
    )
    for row_index, (condition, cumulative, ylabel, observed_label) in enumerate(
        fit_specs
    ):
        for axis, arm in zip(axes[row_index], all_arms, strict=True):
            color, marker, _ = CAPACITY_PLOT_STYLES[(arm, condition)]
            series = _capacity_cumulative_series if cumulative else _capacity_series
            steps, values = series(
                rows,
                arm,
                condition,
                "total_training_wall_seconds",
            )
            fits = _growth_fits(steps, values)
            tlog = fits["t_log2_t_plus_1"]
            power = fits["power_law"]
            step_symbol = "T" if cumulative else "t"
            axis.plot(
                steps,
                values,
                color=color,
                marker=marker,
                linewidth=1.5 if cumulative else 0,
                markersize=4.5 if cumulative else 6.5,
                markevery=10 if cumulative else 1,
                label=observed_label,
            )
            axis.plot(
                curve_steps,
                tuple(
                    float(tlog["coefficient"]) * _growth_scale(step)
                    for step in curve_steps
                ),
                color="#222222",
                linewidth=2.0,
                linestyle="-",
                label=(
                    f"c·{step_symbol}·log₂({step_symbol}+1), "
                    f"R²={float(tlog['r_squared']):.3f}"
                ),
            )
            axis.plot(
                curve_steps,
                tuple(
                    float(power["coefficient"]) * step ** float(power["exponent"])
                    for step in curve_steps
                ),
                color="#7F3C8D",
                linewidth=2.0,
                linestyle="--",
                label=(
                    f"c·{step_symbol}^{float(power['exponent']):.2f}, "
                    f"R²={float(power['r_squared']):.3f}"
                ),
            )
            axis.axvspan(1, FIT_MINIMUM_STEP, color="#999999", alpha=0.08)
            axis.set_xscale("log", base=2)
            axis.set_yscale("log")
            axis.set_xlabel("Learned permutations")
            axis.set_ylabel(ylabel)
            if row_index == 0:
                axis.set_title(CAPACITY_ARM_LABELS[arm])
            axis.grid(True, which="both", alpha=0.25)
            axis.legend(frameon=False, fontsize=7.2)
    figure.suptitle(
        "Empirical runtime fits for both conditions (fit uses T or t ≥ 4)",
        fontsize=14,
    )
    figure.savefig(plot_root / "03_runtime_growth_fits.png", dpi=220)
    plt.close(figure)

    figure, axes = plt.subplots(2, 3, figsize=(17.5, 9.0), constrained_layout=True)
    for arm in all_arms:
        label = CAPACITY_ARM_LABELS[arm]
        multiplier = _capacity_arm_multiplier(arm)
        sample_count = config.benchmark.observer_batch_size * multiplier

        uniform_color, uniform_marker, _ = CAPACITY_PLOT_STYLES[
            (arm, UNIFORM_CONDITION)
        ]
        uniform_steps, uniform_wall = _capacity_cumulative_series(
            rows,
            arm,
            UNIFORM_CONDITION,
            "total_training_wall_seconds",
        )
        _, uniform_features = _capacity_cumulative_series(
            rows,
            arm,
            UNIFORM_CONDITION,
            "feature_forward_example_passes",
        )
        _, uniform_backward = _capacity_cumulative_series(
            rows,
            arm,
            UNIFORM_CONDITION,
            "integrator_backward_example_passes",
        )
        axes[0, 0].plot(
            uniform_steps,
            tuple(
                value / _growth_scale(step)
                for step, value in zip(uniform_steps, uniform_wall, strict=True)
            ),
            color=uniform_color,
            marker=uniform_marker,
            markevery=10,
            linewidth=2.0,
            label=label,
        )
        axes[0, 1].plot(
            uniform_steps,
            tuple(
                value / (sample_count * _growth_scale(step))
                for step, value in zip(
                    uniform_steps,
                    uniform_features,
                    strict=True,
                )
            ),
            color=uniform_color,
            marker=uniform_marker,
            markevery=10,
            linewidth=2.0,
            label=label,
        )
        axes[0, 2].plot(
            uniform_steps,
            tuple(
                value
                / (
                    config.integrator.epochs_per_step
                    * sample_count
                    * (2 * step - 1)
                )
                for step, value in zip(
                    uniform_steps,
                    uniform_backward,
                    strict=True,
                )
            ),
            color=uniform_color,
            marker=uniform_marker,
            markevery=10,
            linewidth=2.0,
            label=label,
        )

        full_color, full_marker, _ = CAPACITY_PLOT_STYLES[
            (arm, FULL_REPLAY_CONDITION)
        ]
        full_steps, full_wall = _capacity_series(
            rows,
            arm,
            FULL_REPLAY_CONDITION,
            "total_training_wall_seconds",
        )
        _, full_features = _capacity_series(
            rows,
            arm,
            FULL_REPLAY_CONDITION,
            "feature_forward_example_passes",
        )
        _, full_backward = _capacity_series(
            rows,
            arm,
            FULL_REPLAY_CONDITION,
            "integrator_backward_example_passes",
        )
        axes[1, 0].plot(
            full_steps,
            tuple(
                value / _growth_scale(step)
                for step, value in zip(full_steps, full_wall, strict=True)
            ),
            color=full_color,
            marker=full_marker,
            linewidth=2.0,
            label=label,
        )
        axes[1, 1].plot(
            full_steps,
            tuple(
                value / (sample_count * _growth_scale(step))
                for step, value in zip(full_steps, full_features, strict=True)
            ),
            color=full_color,
            marker=full_marker,
            linewidth=2.0,
            label=label,
        )
        axes[1, 2].plot(
            full_steps,
            tuple(
                value
                / (config.integrator.offline_epochs * sample_count * step)
                for step, value in zip(full_steps, full_backward, strict=True)
            ),
            color=full_color,
            marker=full_marker,
            linewidth=2.0,
            label=label,
        )
    normalized_specs = (
        (
            axes[0, 0],
            "Cumulative wall / [T log₂(T+1)]",
            None,
            "Persistent replay — wall time",
            False,
        ),
        (
            axes[0, 1],
            "Cumulative frozen forwards / [N T log₂(T+1)]",
            None,
            "Persistent replay — frozen features",
            True,
        ),
        (
            axes[0, 2],
            "Cumulative backward / [E N (2T−1)]",
            1.0,
            "Persistent replay — integrator backward",
            True,
        ),
        (
            axes[1, 0],
            "Fit wall / [t log₂(t+1)]",
            None,
            "Full replay — wall time",
            False,
        ),
        (
            axes[1, 1],
            "Frozen forwards / [N t log₂(t+1)]",
            None,
            "Full replay — frozen features",
            True,
        ),
        (
            axes[1, 2],
            "Backward / [20 N t]",
            1.0,
            "Full replay — integrator backward",
            True,
        ),
    )
    for axis, ylabel, reference, title, exact_overlap in normalized_specs:
        if reference is not None:
            axis.axhline(reference, color="#333333", linestyle=":", linewidth=1.2)
        axis.set_xscale("log", base=2)
        axis.set_xlabel("Learned permutations")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(frameon=False, fontsize=7.5)
        if exact_overlap:
            axis.text(
                0.03,
                0.05,
                "All three normalized arm traces overlap exactly",
                transform=axis.transAxes,
                fontsize=7.5,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
            )
    figure.suptitle(
        "Normalized training-work growth for both conditions",
        fontsize=14,
        y=1.02,
    )
    figure.savefig(
        plot_root / "04_normalized_runtime_growth.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(11.0, 6.0), constrained_layout=True)
    contrast_styles = (
        (UNIFORM_CONDITION, "capacity", "#6A3D9A", "o", "-"),
        (FULL_REPLAY_CONDITION, "capacity", "#009E73", "s", "--"),
        (UNIFORM_CONDITION, "samples", "#D55E00", "^", "-"),
        (FULL_REPLAY_CONDITION, "samples", "#0072B2", "D", "--"),
    )
    for condition, contrast, color, marker, linestyle in contrast_styles:
        steps = tuple(range(1, 101)) if condition == UNIFORM_CONDITION else tuple(
            sorted({
                int(row["macro_step"])
                for row in rows
                if row["condition"] == condition
            })
        )
        if contrast == "capacity":
            higher, lower = CAPACITY_ARM_NAMES[1], REFERENCE_ARM
            name = "4× params − reference"
        else:
            higher, lower = CAPACITY_ARM_NAMES[2], CAPACITY_ARM_NAMES[1]
            name = "double samples − standard"
        differences = tuple(
            100.0
            * (
                _capacity_cell(rows, higher, condition, step, "accuracy")
                - _capacity_cell(rows, lower, condition, step, "accuracy")
            )
            for step in steps
        )
        axis.plot(
            steps,
            differences,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2.1,
            markersize=4.5 if condition == UNIFORM_CONDITION else 6.0,
            markevery=10 if condition == UNIFORM_CONDITION else 1,
            label=f"{name} — {CONDITION_LABELS[condition]}",
        )
    axis.axhline(0.0, color="#333333", linewidth=1.2)
    axis.set_xscale("log", base=2)
    axis.set_xlabel("Learned permutations")
    axis.set_ylabel("Accuracy difference (percentage points)")
    axis.set_title("Single-factor accuracy contrasts, seed 0")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    figure.savefig(plot_root / "05_accuracy_contrasts.png", dpi=220)
    plt.close(figure)


def _write_plots(
    run_root: Path,
    rows: Sequence[Mapping[str, object]],
    hierarchy_rows: Sequence[Mapping[str, object]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(17, 5.4), constrained_layout=True)
    quantities = (
        ("total_training_wall_seconds", "Training-only wall time per update (s)"),
        ("total_training_forward_example_passes", "Forward example-passes per update"),
        ("integrator_backward_example_passes", "Backward example-passes per update"),
    )
    for axis, (field, ylabel) in zip(axes, quantities, strict=True):
        for (capacity, condition), (color, marker, linestyle) in PLOT_STYLES.items():
            steps, means, deviations = _series(rows, capacity, condition, field)
            axis.plot(
                steps,
                means,
                color=color,
                label=_plot_label(capacity, condition),
                linewidth=2.2,
                marker=marker,
                markersize=4.5 if condition == UNIFORM_CONDITION else 6.0,
                markevery=10 if condition == UNIFORM_CONDITION else 1,
                linestyle=linestyle,
            )
            if field == "total_training_wall_seconds":
                lower = tuple(max(mean - deviation, mean * 0.05) for mean, deviation in zip(means, deviations, strict=True))
                upper = tuple(mean + deviation for mean, deviation in zip(means, deviations, strict=True))
                axis.fill_between(steps, lower, upper, color=color, alpha=0.12)
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xlabel("Learned permutations")
        axis.set_ylabel(ylabel)
        axis.grid(True, which="both", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle(
        "Condition-specific training growth (evaluation and shared hierarchy excluded)",
        fontsize=14,
    )
    plot_root = run_root / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_root / "01_training_work_scaling.png", dpi=220)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    for (capacity, condition), (color, marker, linestyle) in PLOT_STYLES.items():
        steps, means, deviations = _series(rows, capacity, condition, "accuracy")
        means_percent = tuple(100.0 * value for value in means)
        deviations_percent = tuple(100.0 * value for value in deviations)
        axis.plot(
            steps,
            means_percent,
            color=color,
            label=_plot_label(capacity, condition),
            linewidth=2.2,
            marker=marker,
            markersize=4.5 if condition == UNIFORM_CONDITION else 6.0,
            markevery=10 if condition == UNIFORM_CONDITION else 1,
            linestyle=linestyle,
        )
        axis.fill_between(
            steps,
            tuple(mean - deviation for mean, deviation in zip(means_percent, deviations_percent, strict=True)),
            tuple(mean + deviation for mean, deviation in zip(means_percent, deviations_percent, strict=True)),
            color=color,
            alpha=0.12,
        )
    axis.set_xscale("log", base=2)
    axis.set_xlabel("Learned permutations")
    axis.set_ylabel("Accuracy over equal-size learned-permutation test subsets (%)")
    axis.set_title("Predictive performance while the permutation set grows")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(frameon=False)
    figure.savefig(plot_root / "02_accuracy_scaling.png", dpi=220)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    for condition, color, marker, linestyle in (
        (UNIFORM_CONDITION, "#6A3D9A", "o", "-"),
        (FULL_REPLAY_CONDITION, "#D55E00", "s", "--"),
    ):
        steps = tuple(sorted({
            int(row["macro_step"])
            for row in rows
            if row["condition"] == condition
        }))
        cells = tuple(
            _mean_sd(_paired_policy_values(rows, condition, step, "accuracy"))
            for step in steps
        )
        means = tuple(100.0 * cell[0] for cell in cells)
        deviations = tuple(100.0 * cell[1] for cell in cells)
        axis.plot(
            steps,
            means,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2.2,
            markersize=4.5 if condition == UNIFORM_CONDITION else 6.0,
            markevery=10 if condition == UNIFORM_CONDITION else 1,
            label=CONDITION_LABELS[condition],
        )
        axis.fill_between(
            steps,
            tuple(mean - deviation for mean, deviation in zip(means, deviations, strict=True)),
            tuple(mean + deviation for mean, deviation in zip(means, deviations, strict=True)),
            color=color,
            alpha=0.14,
        )
    axis.axhline(0.0, color="#333333", linewidth=1.2)
    axis.set_xscale("log", base=2)
    axis.set_xlabel("Learned permutations")
    axis.set_ylabel("Two-node minus one-node accuracy (percentage points)")
    axis.set_title("Paired consolidation-policy effect across five seeds")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(frameon=False)
    figure.savefig(plot_root / "03_policy_accuracy_difference.png", dpi=220)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)
    for capacity, color, marker in ((1, "#0072B2", "o"), (2, "#E69F00", "^")):
        selected = tuple(
            row for row in rows
            if row["condition"] == UNIFORM_CONDITION
            and int(row["max_nodes_per_level"]) == capacity
            and int(row["run_seed"]) == 0
        )
        steps = tuple(int(row["macro_step"]) for row in selected)
        axes[0].plot(
            steps,
            tuple(int(row["active_node_count"]) for row in selected),
            color=color,
            marker=marker,
            markevery=10,
            linewidth=2.0,
            label=f"{capacity} node{'s' if capacity > 1 else ''}/level",
        )
        work = tuple(
            row for row in hierarchy_rows
            if int(row["max_nodes_per_level"]) == capacity
            and int(row["run_seed"]) == 0
        )
        cumulative = []
        total = 0
        for row in work:
            total += int(row["shared_forward_example_passes"])
            cumulative.append(total)
        axes[1].plot(
            tuple(int(row["macro_step"]) for row in work),
            cumulative,
            color=color,
            marker=marker,
            markevery=10,
            linewidth=2.0,
            label=f"{capacity} node{'s' if capacity > 1 else ''}/level",
        )
    axes[0].set_ylabel("Active frozen nodes")
    axes[1].set_ylabel("Cumulative hierarchy forward example-passes")
    axes[1].set_yscale("log")
    for axis in axes:
        axis.set_xlabel("Learned permutations")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(frameon=False)
    figure.suptitle("Consolidation-policy frontier size and shared training work", fontsize=14)
    figure.savefig(plot_root / "04_hierarchy_policy_work.png", dpi=220)
    plt.close(figure)


def _report_markdown(
    config: VampLogTDenseConfig,
    summary: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    hierarchy_rows: Sequence[Mapping[str, object]],
) -> str:
    if config.scaling is None:
        raise ValueError("aggregate report requires the scaling comparison amendment")
    cell = lambda capacity, condition, step, field: _mean_sd(
        _cell_values(rows, capacity, condition, step, field)
    )
    paired = lambda condition, step: _mean_sd(
        _paired_policy_values(rows, condition, step, "accuracy")
    )
    final_step = config.benchmark.macro_steps
    final_cells = {
        (capacity, condition): cell(capacity, condition, final_step, "accuracy")
        for capacity in config.scaling.hierarchy_node_capacities
        for condition in (UNIFORM_CONDITION, FULL_REPLAY_CONDITION)
    }
    final_deltas = {
        condition: paired(condition, final_step)
        for condition in (UNIFORM_CONDITION, FULL_REPLAY_CONDITION)
    }
    table_rows = []
    for step in config.evaluation.full_checkpoints:
        for capacity in config.scaling.hierarchy_node_capacities:
            for condition in (UNIFORM_CONDITION, FULL_REPLAY_CONDITION):
                accuracy = cell(capacity, condition, step, "accuracy")
                seconds = cell(capacity, condition, step, "total_training_wall_seconds")
                forward = cell(capacity, condition, step, "total_training_forward_example_passes")[0]
                backward = cell(capacity, condition, step, "integrator_backward_example_passes")[0]
                table_rows.append(
                    f"| {step} | {capacity} | {CONDITION_LABELS[condition]} | "
                    f"{100 * accuracy[0]:.2f}% ± {100 * accuracy[1]:.2f} | "
                    f"{seconds[0]:.3f} ± {seconds[1]:.3f} | {forward:,.0f} | {backward:,.0f} |"
                )
    delta_rows = []
    for step in config.evaluation.full_checkpoints:
        for condition in (UNIFORM_CONDITION, FULL_REPLAY_CONDITION):
            difference = paired(condition, step)
            delta_rows.append(
                f"| {step} | {CONDITION_LABELS[condition]} | "
                f"{100 * difference[0]:+.2f} ± {100 * difference[1]:.2f} |"
            )
    seed_rows = []
    for seed in config.online.seeds:
        values = {
            (capacity, condition): next(
                float(row["accuracy"])
                for row in rows
                if int(row["run_seed"]) == seed
                and int(row["max_nodes_per_level"]) == capacity
                and row["condition"] == condition
                and int(row["macro_step"]) == final_step
            )
            for capacity in config.scaling.hierarchy_node_capacities
            for condition in (UNIFORM_CONDITION, FULL_REPLAY_CONDITION)
        }
        seed_rows.append(
            f"| {seed} | {100 * values[(1, UNIFORM_CONDITION)]:.2f}% | "
            f"{100 * values[(2, UNIFORM_CONDITION)]:.2f}% | "
            f"{100 * (values[(2, UNIFORM_CONDITION)] - values[(1, UNIFORM_CONDITION)]):+.2f} | "
            f"{100 * values[(1, FULL_REPLAY_CONDITION)]:.2f}% | "
            f"{100 * values[(2, FULL_REPLAY_CONDITION)]:.2f}% | "
            f"{100 * (values[(2, FULL_REPLAY_CONDITION)] - values[(1, FULL_REPLAY_CONDITION)]):+.2f} |"
        )
    work_means = {
        (capacity, condition, step): cell(
            capacity,
            condition,
            step,
            "total_training_wall_seconds",
        )[0]
        for capacity in config.scaling.hierarchy_node_capacities
        for condition in (UNIFORM_CONDITION, FULL_REPLAY_CONDITION)
        for step in (1, final_step)
    }
    hierarchy_totals = {
        capacity: sum(
            int(row["shared_forward_example_passes"])
            for row in hierarchy_rows
            if int(row["max_nodes_per_level"]) == capacity
            and int(row["run_seed"]) == 0
        )
        for capacity in config.scaling.hierarchy_node_capacities
    }
    final_nodes = {
        capacity: int(next(
            row["active_node_count"]
            for row in rows
            if int(row["max_nodes_per_level"]) == capacity
            and int(row["run_seed"]) == 0
            and row["condition"] == UNIFORM_CONDITION
            and int(row["macro_step"]) == final_step
        ))
        for capacity in config.scaling.hierarchy_node_capacities
    }
    parameter_counts = {
        capacity: int(next(
            row["integrator_parameter_count"]
            for row in rows
            if int(row["max_nodes_per_level"]) == capacity
            and row["condition"] == UNIFORM_CONDITION
        ))
        for capacity in config.scaling.hierarchy_node_capacities
    }
    criteria = "\n".join(
        f"| {name.replace('_', ' ')} | {'pass' if value else 'FAIL'} |"
        for name, value in summary["acceptance"].items()
    )
    return f"""# 100-permutation integrator scaling: five seeds and two consolidation capacities

## Result

The run completed under config hash `{config.config_hash}` with five paired run seeds. At permutation 100, one-node uniform replay reached {100 * final_cells[(1, UNIFORM_CONDITION)][0]:.2f}% ± {100 * final_cells[(1, UNIFORM_CONDITION)][1]:.2f}% accuracy; two-node uniform replay reached {100 * final_cells[(2, UNIFORM_CONDITION)][0]:.2f}% ± {100 * final_cells[(2, UNIFORM_CONDITION)][1]:.2f}%. The paired two-node-minus-one-node difference was {100 * final_deltas[UNIFORM_CONDITION][0]:+.2f} ± {100 * final_deltas[UNIFORM_CONDITION][1]:.2f} percentage points across seeds.

At the same checkpoint, the fresh 20-epoch full-replay fit reached {100 * final_cells[(1, FULL_REPLAY_CONDITION)][0]:.2f}% ± {100 * final_cells[(1, FULL_REPLAY_CONDITION)][1]:.2f}% with one node per level and {100 * final_cells[(2, FULL_REPLAY_CONDITION)][0]:.2f}% ± {100 * final_cells[(2, FULL_REPLAY_CONDITION)][1]:.2f}% with two. Its paired policy difference was {100 * final_deltas[FULL_REPLAY_CONDITION][0]:+.2f} ± {100 * final_deltas[FULL_REPLAY_CONDITION][1]:.2f} points. These are sample means and sample standard deviations, not confidence intervals.

The full-replay update's mean wall time grew from {work_means[(1, FULL_REPLAY_CONDITION, 1)]:.3f} to {work_means[(1, FULL_REPLAY_CONDITION, final_step)]:.3f} seconds under one-node consolidation and from {work_means[(2, FULL_REPLAY_CONDITION, 1)]:.3f} to {work_means[(2, FULL_REPLAY_CONDITION, final_step)]:.3f} seconds under two-node consolidation. Example-pass counts, rather than seconds, carry the hardware-independent scaling conclusion.

## Conditions, in literal terms

| Report label | Exactly what was trained |
|---|---|
| Persistent uniform replay | One persistent integrator per policy and seed. At each permutation it trains four epochs on 256 current observer examples plus, after step 1, 256 examples sampled uniformly from all earlier observer examples. Current and historical loss each receive weight 0.5. |
| Fresh full replay, 20 epochs | At each sampled checkpoint and for each policy/seed pair, discard prior integrator state, initialize a fresh integrator, build features for every observer example seen so far, and train exactly 20 epochs. There is no validation, early stopping, restart selection, or convergence claim. |

Both run under one-node-per-level and two-nodes-per-level consolidation. On a two-node overflow, the two older resident nodes merge and the newest remains. Primary slots preserve the predecessor's seven input positions; seven secondary slots are appended with exact-zero input weights. The one-node integrator has {parameter_counts[1]:,} parameters and the two-node integrator has {parameter_counts[2]:,}; an integrator example-pass is therefore more expensive under the two-node policy even when the pass count matches.

There are 100 domains total: identity plus 99 independently seeded fixed pixel permutations. Their order is fixed across seeds. Run seeds vary allocated examples, held-out subsets, node training, integrator training, and replay draws, but not the permutation order.

## Training-work boundary

The timed condition total includes state initialization, replay/archive preparation, frozen-node forwards used to construct training features, and integrator optimizer forwards/backwards. It excludes pre/post diagnostics, learned-domain test inference, report generation, artifact I/O, and shared temporal-node construction. Excluded inference counts remain in explicit audit columns.

Let `a_c(k)` be the active-node count at step `k` under capacity `c`. Uniform replay uses `512 a_c(k) + 2,048` forward example-passes and 2,048 backward example-passes after step 1: `O(log k)` forward and `O(1)` backward per update for either fixed capacity. Full replay uses `256k a_c(k) + 20 × 256k` forward and `20 × 256k` backward example-passes: `O(k log k)` forward and `O(k)` backward per sampled fit. If full replay ran every step, its cumulative bounds would be `O(T² log T)` forward and `O(T²)` backward.

At step 100 the one-node frontier has {final_nodes[1]} active nodes and the two-node frontier has {final_nodes[2]}. Shared hierarchy construction used {hierarchy_totals[1]:,} forward and the same number of backward example-passes per seed for one-node consolidation, versus {hierarchy_totals[2]:,} each for two-node consolidation. These shared costs are in `hierarchy_work.csv` and are not charged to both integrator conditions.

## Five-seed checkpoint means

| Learned permutations | Nodes per level | Condition | Accuracy mean ± SD | Training seconds mean ± SD | Forward example-passes | Backward example-passes |
|---:|---:|---|---:|---:|---:|---:|
{chr(10).join(table_rows)}

## Paired policy differences

Positive values favor two nodes per level.

| Learned permutations | Condition | Accuracy difference, percentage points mean ± SD |
|---:|---|---:|
{chr(10).join(delta_rows)}

## Seed-level results at permutation 100

| Seed | Uniform, one node | Uniform, two nodes | Uniform difference | Full, one node | Full, two nodes | Full difference |
|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(seed_rows)}

## Acceptance checks

| Check | Result |
|---|---|
{criteria}

## Figures

![Training-only time and model-pass growth](plots/01_training_work_scaling.png)

![Accuracy over learned permutations](plots/02_accuracy_scaling.png)

![Paired consolidation-policy accuracy difference](plots/03_policy_accuracy_difference.png)

![Hierarchy frontier size and shared work](plots/04_hierarchy_policy_work.png)

## Limits

Five seeds provide a first estimate of run-seed variance, but the shared permutation order means they do not estimate order sensitivity. Accuracy uses a fixed 256-example test subset per learned permutation and seed, equally weighted. GPU warm-up, scheduling, and host copies make seconds noisier than pass counts. The fresh full-replay condition is limited to 20 epochs and is not a converged or best-possible ceiling. The two-node policy changes both the frozen frontier and the integrator input parameter count, so its accuracy difference cannot be attributed to retention alone.
"""


def _capacity_report_markdown(
    config: VampLogTDenseConfig,
    summary: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    hierarchy_rows: Sequence[Mapping[str, object]],
) -> str:
    """Render the single-seed capacity/sample result in literal terms."""
    all_arms = (REFERENCE_ARM, CAPACITY_ARM_NAMES[1], CAPACITY_ARM_NAMES[2])
    conditions = (UNIFORM_CONDITION, FULL_REPLAY_CONDITION)
    final_step = config.benchmark.macro_steps
    capacity_effects = {
        condition: 100.0
        * (
            _capacity_cell(rows, CAPACITY_ARM_NAMES[1], condition, final_step, "accuracy")
            - _capacity_cell(rows, REFERENCE_ARM, condition, final_step, "accuracy")
        )
        for condition in conditions
    }
    sample_effects = {
        condition: 100.0
        * (
            _capacity_cell(rows, CAPACITY_ARM_NAMES[2], condition, final_step, "accuracy")
            - _capacity_cell(rows, CAPACITY_ARM_NAMES[1], condition, final_step, "accuracy")
        )
        for condition in conditions
    }
    final_accuracies = {
        (arm, condition): 100.0
        * _capacity_cell(rows, arm, condition, final_step, "accuracy")
        for arm in all_arms
        for condition in conditions
    }
    checkpoint_rows = []
    for step in config.evaluation.full_checkpoints:
        for arm in all_arms:
            for condition in conditions:
                checkpoint_rows.append(
                    f"| {step} | {CAPACITY_ARM_LABELS[arm]} | "
                    f"{CONDITION_LABELS[condition]} | "
                    f"{100 * _capacity_cell(rows, arm, condition, step, 'accuracy'):.2f}% | "
                    f"{_capacity_cell(rows, arm, condition, step, 'total_training_wall_seconds'):.3f} | "
                    f"{_capacity_cell(rows, arm, condition, step, 'total_training_forward_example_passes'):,.0f} | "
                    f"{_capacity_cell(rows, arm, condition, step, 'integrator_backward_example_passes'):,.0f} |"
                )

    persistent_fit_rows = []
    full_fit_rows = []
    persistent_normalized_rows = []
    full_normalized_rows = []
    fit_summary = summary["empirical_growth_fits"]
    for arm in all_arms:
        persistent_wall_fits = fit_summary[arm][
            "persistent_uniform_cumulative_training_wall_seconds"
        ]
        persistent_forward_fits = fit_summary[arm][
            "persistent_uniform_cumulative_forward_example_passes"
        ]
        persistent_wall_tlog = persistent_wall_fits["t_log2_t_plus_1"]
        persistent_wall_power = persistent_wall_fits["power_law"]
        persistent_forward_tlog = persistent_forward_fits["t_log2_t_plus_1"]
        persistent_forward_power = persistent_forward_fits["power_law"]
        persistent_fit_rows.append(
            f"| {CAPACITY_ARM_LABELS[arm]} | "
            f"{float(persistent_wall_tlog['coefficient']):.5f} | "
            f"{float(persistent_wall_tlog['r_squared']):.3f} | "
            f"{float(persistent_wall_power['exponent']):.3f} | "
            f"{float(persistent_wall_power['r_squared']):.3f} | "
            f"{float(persistent_forward_tlog['r_squared']):.3f} | "
            f"{float(persistent_forward_power['exponent']):.3f} | "
            f"{float(persistent_forward_power['r_squared']):.3f} |"
        )

        wall_fits = fit_summary[arm]["full_replay_training_wall_seconds"]
        forward_fits = fit_summary[arm]["full_replay_forward_example_passes"]
        wall_tlog = wall_fits["t_log2_t_plus_1"]
        wall_power = wall_fits["power_law"]
        forward_tlog = forward_fits["t_log2_t_plus_1"]
        forward_power = forward_fits["power_law"]
        full_fit_rows.append(
            f"| {CAPACITY_ARM_LABELS[arm]} | "
            f"{float(wall_tlog['coefficient']):.5f} | {float(wall_tlog['r_squared']):.3f} | "
            f"{float(wall_power['exponent']):.3f} | {float(wall_power['r_squared']):.3f} | "
            f"{float(forward_tlog['r_squared']):.3f} | "
            f"{float(forward_power['exponent']):.3f} | {float(forward_power['r_squared']):.3f} |"
        )
        multiplier = _capacity_arm_multiplier(arm)
        samples = config.benchmark.observer_batch_size * multiplier
        persistent_steps, persistent_wall = _capacity_cumulative_series(
            rows,
            arm,
            UNIFORM_CONDITION,
            "total_training_wall_seconds",
        )
        _, persistent_features = _capacity_cumulative_series(
            rows,
            arm,
            UNIFORM_CONDITION,
            "feature_forward_example_passes",
        )
        _, persistent_backward = _capacity_cumulative_series(
            rows,
            arm,
            UNIFORM_CONDITION,
            "integrator_backward_example_passes",
        )
        if persistent_steps[-1] != final_step:
            raise ValueError("persistent runtime series does not reach the final step")
        persistent_backward_denominator = (
            config.integrator.epochs_per_step * samples * (2 * final_step - 1)
        )
        persistent_normalized_rows.append(
            f"| {CAPACITY_ARM_LABELS[arm]} | "
            f"{persistent_wall[-1]:.3f} | "
            f"{persistent_wall[-1] / _growth_scale(final_step):.5f} | "
            f"{persistent_features[-1] / (samples * _growth_scale(final_step)):.3f} | "
            f"{persistent_backward[-1] / persistent_backward_denominator:.3f} |"
        )

        wall_100 = _capacity_cell(
            rows,
            arm,
            FULL_REPLAY_CONDITION,
            final_step,
            "total_training_wall_seconds",
        )
        feature_100 = _capacity_cell(
            rows,
            arm,
            FULL_REPLAY_CONDITION,
            final_step,
            "feature_forward_example_passes",
        )
        backward_100 = _capacity_cell(
            rows,
            arm,
            FULL_REPLAY_CONDITION,
            final_step,
            "integrator_backward_example_passes",
        )
        full_normalized_rows.append(
            f"| {CAPACITY_ARM_LABELS[arm]} | "
            f"{wall_100 / _growth_scale(final_step):.5f} | "
            f"{feature_100 / (samples * _growth_scale(final_step)):.3f} | "
            f"{backward_100 / (config.integrator.offline_epochs * samples * final_step):.3f} |"
        )

    hierarchy_rows_table = []
    for arm in (CAPACITY_ARM_NAMES[1], CAPACITY_ARM_NAMES[2]):
        selected = tuple(row for row in hierarchy_rows if row["arm"] == arm)
        hierarchy_rows_table.append(
            f"| {CAPACITY_ARM_LABELS[arm]} | "
            f"{sum(int(row['shared_forward_example_passes']) for row in selected):,} | "
            f"{sum(int(row['shared_backward_example_passes']) for row in selected):,} |"
        )

    criteria = "\n".join(
        f"| {name.replace('_', ' ')} | {'pass' if value else 'FAIL'} |"
        for name, value in summary["acceptance"].items()
    )
    calibration = summary["calibration"]
    capacity_readout = (
        "The capacity increase improved both measured training conditions at step 100."
        if all(value > 0.0 for value in capacity_effects.values())
        else "The capacity increase did not improve both measured training conditions at step 100."
    )
    sample_readout = (
        "Doubling task samples improved both large-model conditions at step 100."
        if all(value > 0.0 for value in sample_effects.values())
        else "Doubling task samples did not improve both large-model conditions at step 100."
    )
    return f"""# 100-permutation capacity and sample-count experiment

## Result

The run completed under config hash `{config.config_hash}` for seed 0. At
permutation 100, the persistent uniform-replay accuracies were
{final_accuracies[(REFERENCE_ARM, UNIFORM_CONDITION)]:.2f}% for the reference,
{final_accuracies[(CAPACITY_ARM_NAMES[1], UNIFORM_CONDITION)]:.2f}% for 4×
parameters with the same samples, and
{final_accuracies[(CAPACITY_ARM_NAMES[2], UNIFORM_CONDITION)]:.2f}% for 4×
parameters with doubled samples. The isolated capacity contrast was
{capacity_effects[UNIFORM_CONDITION]:+.2f} percentage points; the isolated
sample contrast was {sample_effects[UNIFORM_CONDITION]:+.2f} points.

For the fresh 20-epoch full-replay fits at permutation 100, the corresponding
accuracies were {final_accuracies[(REFERENCE_ARM, FULL_REPLAY_CONDITION)]:.2f}%,
{final_accuracies[(CAPACITY_ARM_NAMES[1], FULL_REPLAY_CONDITION)]:.2f}%, and
{final_accuracies[(CAPACITY_ARM_NAMES[2], FULL_REPLAY_CONDITION)]:.2f}%. The
capacity contrast was {capacity_effects[FULL_REPLAY_CONDITION]:+.2f} points and
the sample contrast was {sample_effects[FULL_REPLAY_CONDITION]:+.2f} points.
{capacity_readout} {sample_readout} These are observations for one seed and one
fixed domain order, not estimates of mean effects.

The new base stopped after {int(calibration['epochs_ran'])} epochs
(`{calibration['stop_reason']}`), restored epoch
{int(calibration['best_epoch'])}, reached
{100 * float(calibration['validation_accuracy_at_best_epoch']):.2f}% validation
accuracy at that epoch, and reached
{100 * float(calibration['identity_test_accuracy']):.2f}% on identity-MNIST
test examples. Test accuracy did not control training or architecture choice.

## What each arm literally changes

| Arm | Base MLP | Base parameters | Integrator MLP | Integrator parameters | Node examples/domain | Observer examples/domain | Historical replay after step 1 |
|---|---|---:|---|---:|---:|---:|---:|
| Reference model, standard samples | 1024/1024/512 | {REFERENCE_BASE_PARAMETER_COUNT:,} | 1024/512/256 | {REFERENCE_INTEGRATOR_PARAMETER_COUNT:,} | 256 | 256 | 256 |
| 4× parameters, standard samples | 2272/2272/1136 | {LARGE_BASE_PARAMETER_COUNT:,} | 1912/956/478 | {LARGE_INTEGRATOR_PARAMETER_COUNT:,} | 256 | 256 | 256 |
| 4× parameters, doubled samples | 2272/2272/1136 | {LARGE_BASE_PARAMETER_COUNT:,} | 1912/956/478 | {LARGE_INTEGRATOR_PARAMETER_COUNT:,} | 512 | 512 | 512 |

The base parameter ratio is
{LARGE_BASE_PARAMETER_COUNT / REFERENCE_BASE_PARAMETER_COUNT:.4f}× and the
integrator ratio is
{LARGE_INTEGRATOR_PARAMETER_COUNT / REFERENCE_INTEGRATOR_PARAMETER_COUNT:.4f}×.
Reference versus large/standard isolates capacity. Large/standard versus
large/doubled isolates samples. All original training-role rows are retained in
the doubled arm, evaluation rows are identical, and the two large integrators
start from the same weights. Only one node per temporal level is allowed.

`Persistent uniform replay` is one continuing integrator trained four epochs
per domain on the current observer batch and, after step 1, an equally weighted
uniform historical batch. `Fresh full replay, 20 epochs` discards prior
integrator state at each reported checkpoint and trains a new model for exactly
20 epochs on all observer examples seen so far. It has no early stopping and is
not claimed to be converged.

## Reporting correction

The first report version plotted empirical fits only for full replay. That
omitted the requested fit for persistent replay. This revision adds cumulative
persistent wall-time and forward-pass fits over all 100 updates, gives both
conditions equal space in the normalized diagnostics, and leaves every
training measurement unchanged.

## Persistent-replay runtime scaling

Persistent replay runs once at every task, so its end-to-end scaling quantity
is cumulative training work through learned-task count `T`. The table fits all
cumulative observations with `T >= {FIT_MINIMUM_STEP}`. The `T log T` curve is
`work = c × T × log2(T+1)` through the origin. The power curve is
`work = c × T^p`. R-squared is calculated on the original measurement scale,
including for the power fit.

| Arm | Cumulative wall T-log coefficient | Wall T-log R² | Wall power p | Wall power R² | Cumulative forward-pass T-log R² | Forward-pass power p | Forward-pass power R² |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(persistent_fit_rows)}

| Arm | Cumulative wall at T=100 (s) | Wall / T-log | Cumulative frozen features / (N T-log) | Cumulative backward / [E N (2T−1)] |
|---|---:|---:|---:|---:|
{chr(10).join(persistent_normalized_rows)}

Here `E=4` persistent-training epochs and `N` is the current-task sample count
for the arm. The backward denominator is exact: step 1 trains on `N` examples,
and each later step trains on `N` current plus `N` replay examples. The
frozen-feature numerator is measured separately from integrator work.

## Full-replay runtime scaling

Full replay was sampled at ten checkpoints. Its plotted quantity is the cost
of one fresh 20-epoch fit at task count `t`, not cumulative full replay through
all preceding task counts. Fits use the sampled checkpoints with
`t >= {FIT_MINIMUM_STEP}` and the same through-origin `t log2(t+1)` and power
curves.

| Arm | Fit wall t-log coefficient | Wall t-log R² | Wall power p | Wall power R² | Fit forward-pass t-log R² | Forward-pass power p | Forward-pass power R² |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(full_fit_rows)}

| Arm | Fit wall / t-log at 100 | Frozen features / (N t-log) | Backward / (20 N t) |
|---|---:|---:|---:|
{chr(10).join(full_normalized_rows)}

The one-node frontier has `popcount(t)` active nodes. After step 1, one
persistent update performs constant integrator work on `2N` examples and
`2N × popcount(t)` frozen-feature forwards. Its integrator-only cumulative
work is therefore linear, while its cumulative frozen-feature work is
`N + 2N × sum(popcount(k), k=2..T)`, which is `Theta(T log T)`. One full-replay
fit performs `20Nt` integrator forwards and backwards plus
`Nt × popcount(t)` frozen-feature forwards. Its integrator component is
exactly linear in `t`; only its frozen-feature term has the logarithmic upper
bound.

Reference seconds came from the preceding GPU process, whereas the two new
arms were measured in this run. Pass-count comparisons are stronger than
cross-session wall-time comparisons.

## Checkpoint measurements

| Learned permutations | Arm | Condition | Accuracy | Training seconds | Forward example-passes | Backward example-passes |
|---:|---|---|---:|---:|---:|---:|
{chr(10).join(checkpoint_rows)}

## Shared hierarchy work

Temporal-node construction is required by both integrator conditions, so it is
recorded once and excluded from both condition curves.

| New arm | Hierarchy forward example-passes | Hierarchy backward example-passes |
|---|---:|---:|
{chr(10).join(hierarchy_rows_table)}

## Acceptance checks

| Check | Result |
|---|---|
{criteria}

## Figures

![Accuracy across capacity and sample arms](plots/01_accuracy_capacity_and_samples.png)

![Absolute training-only time and model-pass growth](plots/02_training_work_absolute.png)

![Empirical runtime fits for persistent and full replay](plots/03_runtime_growth_fits.png)

![Both conditions normalized by their theoretical factors](plots/04_normalized_runtime_growth.png)

![Capacity-only and sample-only accuracy contrasts](plots/05_accuracy_contrasts.png)

## Limits

This is one seed with one fixed permutation order, so there is no variance
estimate and no order-sensitivity estimate. The reference is authenticated
from the previous run rather than timed in the same process. Accuracy is the
mean over equal 256-example test subsets for learned domains. The 20-epoch
full-replay condition is a fixed-budget comparator, not a converged upper
bound. Doubling samples intentionally doubles node-training and replay work;
the resulting accuracy difference is the effect of that entire data-budget
change under the large architecture.
"""


def _cpu_state_dict(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in state.items()}


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for values in optimizer.state.values():
        for name, value in values.items():
            if isinstance(value, Tensor):
                values[name] = value.to(device)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    """Run one frozen scaling configuration through the shared entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="frozen scaling YAML (default: five-seed consolidation study)",
    )
    arguments = parser.parse_args()
    result = run_experiment(arguments.config)
    print(f"Scaling artifacts: {result}", flush=True)


if __name__ == "__main__":
    main()


__all__ = ["CAPACITY_CONFIG", "DEFAULT_CONFIG", "run_experiment"]

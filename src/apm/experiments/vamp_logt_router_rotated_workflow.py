"""Resumable behavioral-router workflow for VAMP-AF Rotated-MNIST contexts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from apm.continual.artifacts import (
    ChainedJsonlLedger,
    atomic_write,
    canonical_json_bytes,
    file_sha256,
    publish_immutable_json,
)
from apm.continual.logt_behavioral_router import (
    RouterConditionState,
    create_condition_state,
    sample_example_balanced,
    sample_range_balanced,
    train_condition,
)
from apm.experiments.vamp_logt_router_config import PhaseConfig
from apm.experiments.vamp_logt_router_data import (
    ExampleBatch,
    FrozenClassifierDependency,
    concatenate_batches,
    load_frozen_classifier,
    named_seed,
    resolved_device,
)
from apm.experiments.vamp_logt_router_rotated_config import (
    VampLogTRotatedRouterConfig,
)
from apm.experiments.vamp_logt_router_rotated_data import build_benchmark
from apm.experiments.vamp_logt_router_rotated_reporting import (
    write_phase_report,
    write_results,
)
from apm.experiments.vamp_logt_router_state import (
    advance_adapter_bank,
    retire_inactive_nodes,
)
from apm.experiments.vamp_logt_router_workflow import (
    RouterWorkCounters,
    SeedResult,
    _evaluate_step,
    _load_seed_checkpoint,
    _require_smoke_gate,
    _save_seed_checkpoint,
    _supervision,
)


def run_workflow(
    config: VampLogTRotatedRouterConfig,
    selected_phase: str = "all",
) -> Path:
    """Run or resume smoke and primary under the new sealed task protocol."""
    if selected_phase not in {"smoke", "primary", "all"}:
        raise ValueError("phase must be smoke, primary, or all")
    device = resolved_device(config.runtime.device)
    if config.runtime.deterministic_algorithms:
        torch.use_deterministic_algorithms(True)
    run_root = config.artifact_root / "runs" / config.config_hash
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"Durable Rotated-MNIST behavioral-router run: {run_root}", flush=True)
    dependency = load_frozen_classifier(config)
    _write_protocol(run_root, config, dependency)
    phases = (
        (("smoke", config.smoke),)
        if selected_phase == "smoke"
        else (("primary", config.primary),)
        if selected_phase == "primary"
        else (("smoke", config.smoke), ("primary", config.primary))
    )
    completed = []
    for phase_name, phase in phases:
        if phase_name == "primary":
            _require_smoke_gate(run_root, config)
        print(
            f"Phase {phase_name}: {len(phase.seeds)} seed(s), "
            f"{phase.macro_steps} macro-steps",
            flush=True,
        )
        for seed in phase.seeds:
            completed.append(
                run_phase_seed(
                    config,
                    phase_name,
                    phase,
                    seed,
                    dependency,
                    run_root,
                    device,
                )
            )
    write_results(run_root, config, tuple(completed))
    atomic_write(
        config.artifact_root / "LATEST_RUN.json",
        canonical_json_bytes(
            {
                "config_hash": config.config_hash,
                "run_root": str(run_root),
                "schema_version": "vamp-logt-rotated-router-latest-v1",
            }
        ),
    )
    return run_root


def run_phase_seed(
    config: VampLogTRotatedRouterConfig,
    phase_name: str,
    phase: PhaseConfig,
    seed: int,
    dependency: FrozenClassifierDependency,
    run_root: Path,
    device: torch.device,
) -> SeedResult:
    """Run or resume one hierarchy and its independent router conditions."""
    directory = run_root / phase_name / f"seed-{seed}"
    checkpoint_path = directory / "state" / "checkpoint.pt"
    nodes_root = directory / "nodes"
    ledger = ChainedJsonlLedger(directory / "metrics.jsonl", "vamp-logt-router-metric-v1")
    benchmark = build_benchmark(config, phase_name, seed)
    input_dim = config.router.maximum_levels * (dependency.base.hidden_dim + 10 + 1)
    conditions = {
        name: create_condition_state(name, input_dim, config.router, seed, device)
        for name in phase.conditions
    }
    bank, work, completed_step, checkpoint_rows = _load_seed_checkpoint(
        checkpoint_path,
        config,
        phase_name,
        seed,
        phase,
        conditions,
        device,
    )
    if ledger.next_sequence < checkpoint_rows:
        raise ValueError("checkpoint refers to metric rows absent from the ledger")
    ledger.truncate(checkpoint_rows)
    model_batches = tuple(
        benchmark.step(step).model for step in range(1, completed_step + 1)
    )
    router_batches = tuple(
        benchmark.step(step).router for step in range(1, completed_step + 1)
    )
    evaluation_batches = tuple(
        benchmark.step(step).evaluation for step in range(1, completed_step + 1)
    )
    phase_start = perf_counter()
    try:
        from tqdm.auto import tqdm
    except ImportError as error:  # pragma: no cover - vision environment gate
        raise RuntimeError("tqdm is required by the vision environment") from error
    progress = tqdm(
        range(completed_step + 1, phase.macro_steps + 1),
        initial=completed_step,
        total=phase.macro_steps,
        desc=f"rotated {phase_name} seed {seed} overall",
        disable=not config.runtime.progress,
        unit="step",
    )
    for macro_step in progress:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        batches = benchmark.step(macro_step)
        model_batches = (*model_batches, batches.model)
        model_archive = concatenate_batches(model_batches)
        hierarchy_start = perf_counter()
        bank = advance_adapter_bank(
            config,
            bank,
            model_archive,
            dependency,
            seed,
            nodes_root,
            device,
        )
        hierarchy_seconds = perf_counter() - hierarchy_start
        current_supervision = _supervision(
            config,
            dependency,
            bank,
            batches.router,
            device,
        )
        router_start = perf_counter()
        historical_counts: dict[str, int] = {}
        training_rows = []
        router_archive = (
            None if not router_batches else concatenate_batches(router_batches)
        )
        for condition_name, state in conditions.items():
            historical_supervision = None
            duplicate_draws = 0
            range_counts: tuple[tuple[int, int, int], ...] = ()
            if condition_name != "no_replay_hard" and router_archive is not None:
                sampler_seed = named_seed(seed, condition_name, macro_step, "replay")
                draw = (
                    sample_example_balanced(
                        router_archive,
                        phase.historical_budget,
                        sampler_seed,
                        macro_step,
                    )
                    if condition_name.startswith("example_")
                    else sample_range_balanced(
                        router_archive,
                        bank.topology.active_nodes,
                        phase.historical_budget,
                        sampler_seed,
                        macro_step,
                    )
                )
                if len(draw.batch.labels) != phase.historical_budget:
                    raise RuntimeError("replay condition did not receive its exact fixed budget")
                historical_supervision = _supervision(
                    config,
                    dependency,
                    bank,
                    draw.batch,
                    device,
                )
                historical_counts[condition_name] = len(draw.batch.labels)
                duplicate_draws = draw.duplicate_draws
                range_counts = draw.range_draw_counts
            result = train_condition(
                state,
                current_supervision,
                historical_supervision,
                "soft" if condition_name.endswith("_soft") else "hard",
                phase.router_epochs_per_step,
                config.router,
                seed,
                macro_step,
                device,
            )
            training_rows.append(
                {
                    "active_node_count": len(bank.topology.active_nodes),
                    "condition": condition_name,
                    "duplicate_replay_draws": duplicate_draws,
                    "historical_examples": historical_counts.get(condition_name, 0),
                    "macro_step": macro_step,
                    "mean_first_epoch_loss": result.mean_first_epoch_loss,
                    "mean_last_epoch_loss": result.mean_last_epoch_loss,
                    "optimizer_steps": result.optimizer_steps,
                    "range_draw_counts": [list(value) for value in range_counts],
                    "row_type": "training",
                    "run_seed": seed,
                }
            )
        router_seconds = perf_counter() - router_start
        router_batches = (*router_batches, batches.router)
        evaluation_batches = (*evaluation_batches, batches.evaluation)
        evaluation_start = perf_counter()
        evaluation_rows, evaluation_examples, joint_updates = _evaluate_step(
            config,
            phase_name,
            phase,
            seed,
            macro_step,
            benchmark,
            dependency,
            bank,
            conditions,
            concatenate_batches(evaluation_batches),
            batches.evaluation,
            model_archive,
            directory,
            device,
        )
        evaluation_seconds = perf_counter() - evaluation_start
        work = work.advanced(
            active_nodes=len(bank.topology.active_nodes),
            current_examples=len(batches.router.labels),
            historical_examples=historical_counts,
            evaluation_examples=evaluation_examples,
            joint_updates=joint_updates,
        )
        peak_memory = (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        )
        accounting_row = {
            "active_node_count": len(bank.topology.active_nodes),
            "adapter_example_updates": bank.adapter_example_updates,
            "evaluation_seconds": evaluation_seconds,
            "hierarchy_seconds": hierarchy_seconds,
            "macro_step": macro_step,
            "peak_gpu_memory_bytes": peak_memory,
            "router_seconds": router_seconds,
            "row_type": "accounting",
            "run_seed": seed,
            "temporal_ranges": [
                [node.first_block + 1, node.last_block + 1]
                for node in bank.topology.active_nodes
            ],
            "work": asdict(work),
        }
        ledger.append_many((*training_rows, *evaluation_rows, accounting_row))
        _save_seed_checkpoint(
            checkpoint_path,
            config,
            phase_name,
            seed,
            macro_step,
            bank,
            conditions,
            work,
            ledger.next_sequence,
        )
        retire_inactive_nodes(
            nodes_root,
            {node.node_id for node in bank.topology.active_nodes},
        )
        progress.set_postfix(
            nodes=len(bank.topology.active_nodes),
            rows=ledger.next_sequence,
        )
    summary = write_phase_report(
        directory,
        config,
        phase_name,
        seed,
        bank,
        conditions,
        work,
        ledger.rows,
        perf_counter() - phase_start,
    )
    return SeedResult(phase_name, seed, directory, summary)


def _write_protocol(
    run_root: Path,
    config: VampLogTRotatedRouterConfig,
    dependency: FrozenClassifierDependency,
) -> None:
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - vision environment gate
        raise RuntimeError("PyYAML is required by the vision environment") from error
    resolved = yaml.safe_dump(config.as_record(), sort_keys=False).encode("utf-8")
    config_path = run_root / "config_resolved.yaml"
    if config_path.is_file() and config_path.read_bytes() != resolved:
        raise ValueError("resolved protocol changed inside one run identity")
    atomic_write(config_path, resolved)
    project_root = Path(__file__).resolve().parents[3]
    material_paths = (
        "configs/vamp_af_mnist/poc.yaml",
        "configs/vamp_logt_router_rotated_mnist/primary.yaml",
        "docs/logt_vamp_mnist_integrated_router_plan.md",
        "docs/logt_vamp_rotated_mnist_integrated_router_protocol.md",
        "src/apm/continual/logt_behavioral_router.py",
        "src/apm/continual/logt_evidence_bank.py",
        "src/apm/continual/top_two_adapter.py",
        "src/apm/experiments/vamp_af_config.py",
        "src/apm/experiments/vamp_af_data.py",
        "src/apm/experiments/vamp_logt_evidence_training.py",
        "src/apm/experiments/vamp_logt_router_config.py",
        "src/apm/experiments/vamp_logt_router_data.py",
        "src/apm/experiments/vamp_logt_router_metrics.py",
        "src/apm/experiments/vamp_logt_router_reporting.py",
        "src/apm/experiments/vamp_logt_router_state.py",
        "src/apm/experiments/vamp_logt_router_workflow.py",
        "src/apm/experiments/vamp_logt_router_rotated_config.py",
        "src/apm/experiments/vamp_logt_router_rotated_data.py",
        "src/apm/experiments/vamp_logt_router_rotated_mnist.py",
        "src/apm/experiments/vamp_logt_router_rotated_reporting.py",
        "src/apm/experiments/vamp_logt_router_rotated_workflow.py",
    )
    publish_immutable_json(
        run_root / "protocol.json",
        {
            "baseline_checkpoint_sha256": config.baseline.checkpoint_sha256,
            "baseline_protocol_sha256": dependency.protocol_sha256,
            "config": config.as_record(),
            "config_hash": config.config_hash,
            "data_sha256": dict(dependency.data_sha256),
            "material_source_sha256": {
                path: file_sha256(project_root / path) for path in material_paths
            },
            "schema_version": "vamp-logt-rotated-router-protocol-v1",
            "torch_version": torch.__version__,
        },
    )


__all__ = ["run_phase_seed", "run_workflow"]

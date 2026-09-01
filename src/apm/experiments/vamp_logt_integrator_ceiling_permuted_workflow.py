"""Resumable converged full-replay integrator ceiling on Permuted-MNIST."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from apm.continual.artifacts import (
    ChainedJsonlLedger,
    atomic_write,
    canonical_json_bytes,
    file_sha256,
    load_canonical_json,
    publish_immutable_json,
)
from apm.continual.logt_behavioral_integrator import (
    ConvergedFullReplayResult,
    IntegratorConditionState,
    IntegratorSupervision,
    LevelSlotIntegrator,
    create_condition_state,
    prediction_logits,
    train_converged_full_replay,
)
from apm.continual.vision.imagenetr.checkpoints import atomic_torch_save
from apm.experiments.vamp_logt_integrator_ceiling_permuted_config import (
    IntegratorCeilingPhaseConfig,
    VampLogTIntegratorCeilingConfig,
)
from apm.experiments.vamp_logt_integrator_ceiling_permuted_reporting import (
    write_phase_report,
    write_results,
)
from apm.experiments.vamp_logt_integrator_features import integrator_supervision
from apm.experiments.vamp_logt_integrator_metrics import (
    fixed_control_logits,
    prediction_metric_rows,
)
from apm.experiments.vamp_logt_router_data import (
    ExampleBatch,
    FrozenClassifierDependency,
    PermutedMnistBenchmark,
    build_benchmark,
    concatenate_batches,
    load_frozen_classifier,
    named_seed,
    resolved_device,
)
from apm.experiments.vamp_logt_router_state import (
    ActiveAdapterBank,
    advance_adapter_bank,
    bank_from_record,
    bank_record,
    empty_adapter_bank,
    retire_inactive_nodes,
)
from apm.experiments.vamp_logt_router_workflow import SeedResult


CEILING_CONDITION = "converged_full_replay_integrator"
CEILING_CONTROLS = ("mean_ensemble", "best_single_node")


@dataclass(frozen=True, slots=True)
class CeilingWorkCounters:
    """Exact feature, optimization, validation, and test work."""

    training_node_feature_evals: int = 0
    validation_node_feature_evals: int = 0
    test_node_feature_evals: int = 0
    training_example_presentations: int = 0
    validation_example_presentations: int = 0
    optimizer_steps: int = 0
    restart_fits: int = 0

    def __post_init__(self) -> None:
        if any(value < 0 for value in asdict(self).values()):
            raise ValueError("ceiling work counters must be nonnegative")

    def advanced(
        self,
        *,
        active_nodes: int,
        training_examples: int,
        validation_examples: int,
        test_examples: int,
        fits: Sequence["RestartFit"],
    ) -> "CeilingWorkCounters":
        """Return counters after one complete fresh-restart ceiling step."""
        return CeilingWorkCounters(
            self.training_node_feature_evals + active_nodes * training_examples,
            self.validation_node_feature_evals + active_nodes * validation_examples,
            self.test_node_feature_evals + active_nodes * test_examples,
            self.training_example_presentations
            + sum(fit.result.training_example_presentations for fit in fits),
            self.validation_example_presentations
            + sum(fit.result.validation_example_presentations for fit in fits),
            self.optimizer_steps + sum(fit.result.optimizer_steps for fit in fits),
            self.restart_fits + len(fits),
        )


@dataclass(frozen=True, slots=True)
class RestartFit:
    """One independently initialized step-local convergence fit."""

    restart: int
    fit_seed: int
    state: IntegratorConditionState
    result: ConvergedFullReplayResult
    parameter_path: Path
    parameter_sha256: str


def run_workflow(
    config: VampLogTIntegratorCeilingConfig,
    selected_phase: str = "all",
) -> Path:
    """Run or resume the selected empirical-ceiling phases."""
    if selected_phase not in {"smoke", "primary", "all"}:
        raise ValueError("phase must be smoke, primary, or all")
    device = resolved_device(config.runtime.device)
    if config.runtime.deterministic_algorithms:
        torch.use_deterministic_algorithms(True)
    run_root = config.artifact_root / "runs" / config.config_hash
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"Durable Permuted-MNIST integrator ceiling: {run_root}", flush=True)
    dependency = load_frozen_classifier(config)
    parent_summary = _authenticate_parent_integrator(config)
    _write_protocol(run_root, config, dependency)
    phases = (
        (("smoke", config.smoke),)
        if selected_phase == "smoke"
        else (("primary", config.primary),)
        if selected_phase == "primary"
        else (("smoke", config.smoke), ("primary", config.primary))
    )
    for phase_name, phase in phases:
        if phase_name == "primary":
            _require_smoke_gate(run_root, config)
        print(
            f"Phase {phase_name}: {len(phase.seeds)} seed(s), "
            f"{phase.macro_steps} steps, {phase.restarts_per_step} restart(s) per step",
            flush=True,
        )
        for seed in phase.seeds:
            run_phase_seed(
                config,
                phase_name,
                phase,
                seed,
                dependency,
                run_root,
                device,
            )
    write_results(run_root, config, parent_summary)
    atomic_write(
        config.artifact_root / "LATEST_RUN.json",
        canonical_json_bytes(
            {
                "config_hash": config.config_hash,
                "run_root": str(run_root),
                "schema_version": "vamp-logt-integrator-ceiling-latest-v1",
            }
        ),
    )
    return run_root


def run_phase_seed(
    config: VampLogTIntegratorCeilingConfig,
    phase_name: str,
    phase: IntegratorCeilingPhaseConfig,
    seed: int,
    dependency: FrozenClassifierDependency,
    run_root: Path,
    device: torch.device,
) -> SeedResult:
    """Run or resume one hierarchy and its fresh full-replay fits."""
    directory = run_root / phase_name / f"seed-{seed}"
    checkpoint_path = directory / "state" / "checkpoint.pt"
    nodes_root = directory / "nodes"
    ledger = ChainedJsonlLedger(
        directory / "metrics.jsonl", "vamp-logt-integrator-ceiling-metric-v1"
    )
    benchmark = build_benchmark(config, seed)
    bank, work, completed_step, checkpoint_rows = _load_checkpoint(
        checkpoint_path, config, phase_name, seed
    )
    if ledger.next_sequence < checkpoint_rows:
        raise ValueError("ceiling checkpoint refers to absent metric rows")
    ledger.truncate(checkpoint_rows)
    _retire_completed_restart_parameters(
        directory / "integrators",
        completed_step,
        set(config.evaluation.full_checkpoints) if phase_name == "primary" else set(),
    )
    model_batches = tuple(
        benchmark.step(step).model for step in range(1, completed_step + 1)
    )
    training_batches = tuple(
        benchmark.step(step).router for step in range(1, completed_step + 1)
    )
    validation_batches = tuple(
        benchmark.step(step).evaluation for step in range(1, completed_step + 1)
    )
    parent_micro = (
        _load_parent_micro_rows(config, seed) if phase_name == "primary" else {}
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
        desc=f"ceiling {phase_name} seed {seed} overall",
        disable=not config.runtime.progress,
        unit="step",
    )
    for macro_step in progress:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        batches = benchmark.step(macro_step)
        model_batches = (*model_batches, batches.model)
        training_batches = (*training_batches, batches.router)
        validation_batches = (*validation_batches, batches.evaluation)
        model_archive = concatenate_batches(model_batches)
        training_archive = concatenate_batches(training_batches)
        validation_archive = concatenate_batches(validation_batches)
        if _training_validation_overlap(training_archive, validation_archive):
            raise RuntimeError("ceiling training and validation allocations overlap")

        hierarchy_start = perf_counter()
        bank = advance_adapter_bank(
            config, bank, model_archive, dependency, seed, nodes_root, device
        )
        hierarchy_seconds = perf_counter() - hierarchy_start
        feature_start = perf_counter()
        training_supervision, validation_supervision = (
            integrator_supervision(
                config, dependency, bank, archive, device, base_only=False
            )
            for archive in (training_archive, validation_archive)
        )
        feature_seconds = perf_counter() - feature_start
        fit_start = perf_counter()
        fits = _fit_or_load_restarts(
            config,
            phase,
            seed,
            macro_step,
            dependency,
            training_supervision,
            validation_supervision,
            directory / "integrators" / f"step-{macro_step}",
            device,
        )
        selected = _select_restart(fits)
        fit_seconds = perf_counter() - fit_start
        selection_record = {
            "config_hash": config.config_hash,
            "macro_step": macro_step,
            "schema_version": "vamp-logt-integrator-ceiling-selection-v1",
            "selected_parameter_sha256": selected.parameter_sha256,
            "selected_restart": selected.restart,
            "selected_validation_accuracy": selected.result.best_validation_accuracy,
            "selected_validation_cross_entropy": selected.result.best_validation_loss,
        }
        publish_immutable_json(
            directory / "integrators" / f"step-{macro_step}" / "selection.json",
            selection_record,
        )
        training_rows = tuple(
            _training_row(
                fit,
                fit.restart == selected.restart,
                seed,
                macro_step,
                len(bank.topology.active_nodes),
                len(training_archive.labels),
                len(validation_archive.labels),
            )
            for fit in fits
        )
        evaluation_start = perf_counter()
        evaluation_rows, test_examples = _evaluate_step(
            config,
            phase_name,
            seed,
            macro_step,
            benchmark,
            dependency,
            bank,
            selected.state.integrator,
            parent_micro,
            device,
        )
        evaluation_seconds = perf_counter() - evaluation_start
        work = work.advanced(
            active_nodes=len(bank.topology.active_nodes),
            training_examples=len(training_archive.labels),
            validation_examples=len(validation_archive.labels),
            test_examples=test_examples,
            fits=fits,
        )
        accounting_row = {
            "active_node_count": len(bank.topology.active_nodes),
            "adapter_example_updates": bank.adapter_example_updates,
            "evaluation_seconds": evaluation_seconds,
            "feature_seconds": feature_seconds,
            "fit_seconds": fit_seconds,
            "hierarchy_seconds": hierarchy_seconds,
            "macro_step": macro_step,
            "peak_gpu_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
            "row_type": "accounting",
            "run_seed": seed,
            "temporal_ranges": [
                [node.first_block + 1, node.last_block + 1]
                for node in bank.topology.active_nodes
            ],
            "work": asdict(work),
        }
        ledger.append_many((*training_rows, *evaluation_rows, accounting_row))
        _save_checkpoint(
            checkpoint_path,
            config,
            phase_name,
            seed,
            macro_step,
            bank,
            work,
            ledger.next_sequence,
        )
        retire_inactive_nodes(
            nodes_root, {node.node_id for node in bank.topology.active_nodes}
        )
        _retire_completed_restart_parameters(
            directory / "integrators",
            macro_step,
            set(config.evaluation.full_checkpoints) if phase_name == "primary" else set(),
        )
        progress.set_postfix(
            best_epoch=selected.result.best_epoch,
            nodes=len(bank.topology.active_nodes),
            validation=f"{selected.result.best_validation_loss:.4f}",
        )
    summary = write_phase_report(
        directory,
        config,
        phase_name,
        seed,
        bank,
        work,
        ledger.rows,
        perf_counter() - phase_start,
    )
    return SeedResult(phase_name, seed, directory, summary)


def _fit_or_load_restarts(
    config: VampLogTIntegratorCeilingConfig,
    phase: IntegratorCeilingPhaseConfig,
    run_seed: int,
    macro_step: int,
    dependency: FrozenClassifierDependency,
    training: IntegratorSupervision,
    validation: IntegratorSupervision,
    root: Path,
    device: torch.device,
) -> tuple[RestartFit, ...]:
    slot_dim = dependency.base.hidden_dim + 10 + 1
    input_dim = config.integrator.maximum_levels * slot_dim
    fits = []
    for restart in range(phase.restarts_per_step):
        fit_seed = named_seed(
            run_seed, "converged-full-replay-integrator", macro_step, restart
        )
        state = create_condition_state(
            f"{CEILING_CONDITION}_step_{macro_step}_restart_{restart}",
            input_dim,
            slot_dim,
            config.integrator,
            fit_seed,
            device,
        )
        parameter_path = root / f"restart-{restart}.pt"
        history_path = root / f"restart-{restart}.json"
        if parameter_path.is_file() != history_path.is_file():
            raise ValueError("partial ceiling restart artifact cannot be resumed")
        if parameter_path.is_file():
            payload = torch.load(parameter_path, map_location="cpu", weights_only=True)
            history_payload = load_canonical_json(history_path)
            if (
                payload.get("schema_version")
                != "vamp-logt-integrator-ceiling-restart-v1"
                or payload.get("config_hash") != config.config_hash
                or payload.get("macro_step") != macro_step
                or payload.get("restart") != restart
                or payload.get("fit_seed") != fit_seed
                or history_payload.get("coordinates")
                != {
                    "config_hash": config.config_hash,
                    "fit_seed": fit_seed,
                    "macro_step": macro_step,
                    "restart": restart,
                }
            ):
                raise ValueError("stored ceiling restart coordinates changed")
            state.integrator.load_state_dict(payload["parameters"], strict=True)
            result = _result_from_record(history_payload["result"])
            state.optimizer_steps = result.optimizer_steps
        else:
            print(
                f"Ceiling seed {run_seed} step {macro_step}: restart "
                f"{restart + 1}/{phase.restarts_per_step}",
                flush=True,
            )
            result = train_converged_full_replay(
                state,
                training,
                validation,
                config.integrator,
                config.convergence,
                fit_seed,
                macro_step,
                device,
                progress=config.runtime.progress,
            )
            parameter_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_torch_save(
                parameter_path,
                {
                    "config_hash": config.config_hash,
                    "fit_seed": fit_seed,
                    "macro_step": macro_step,
                    "parameters": state.integrator.state_dict(),
                    "restart": restart,
                    "schema_version": "vamp-logt-integrator-ceiling-restart-v1",
                },
            )
            publish_immutable_json(
                history_path,
                {
                    "coordinates": {
                        "config_hash": config.config_hash,
                        "fit_seed": fit_seed,
                        "macro_step": macro_step,
                        "restart": restart,
                    },
                    "result": result.as_record(),
                    "schema_version": "vamp-logt-integrator-ceiling-history-v1",
                },
            )
        fits.append(
            RestartFit(
                restart,
                fit_seed,
                state,
                result,
                parameter_path,
                file_sha256(parameter_path),
            )
        )
    return tuple(fits)


def _result_from_record(value: object) -> ConvergedFullReplayResult:
    if not isinstance(value, Mapping):
        raise ValueError("ceiling convergence history is malformed")
    from apm.continual.logt_behavioral_integrator import ConvergenceEpochResult

    history = value.get("history")
    if not isinstance(history, list):
        raise ValueError("ceiling convergence epochs are missing")
    scalar = {key: item for key, item in value.items() if key != "history"}
    return ConvergedFullReplayResult(
        converged=bool(scalar["converged"]),
        stop_reason=str(scalar["stop_reason"]),
        best_epoch=int(scalar["best_epoch"]),
        epochs_ran=int(scalar["epochs_ran"]),
        best_training_loss=float(scalar["best_training_loss"]),
        best_training_accuracy=float(scalar["best_training_accuracy"]),
        best_validation_loss=float(scalar["best_validation_loss"]),
        best_validation_accuracy=float(scalar["best_validation_accuracy"]),
        final_learning_rate=float(scalar["final_learning_rate"]),
        optimizer_steps=int(scalar["optimizer_steps"]),
        training_example_presentations=int(scalar["training_example_presentations"]),
        validation_example_presentations=int(scalar["validation_example_presentations"]),
        history=tuple(
            ConvergenceEpochResult(
                epoch=int(row["epoch"]),
                learning_rate=float(row["learning_rate"]),
                next_learning_rate=float(row["next_learning_rate"]),
                training_loss=float(row["training_loss"]),
                training_accuracy=float(row["training_accuracy"]),
                validation_loss=float(row["validation_loss"]),
                validation_accuracy=float(row["validation_accuracy"]),
                best_validation_loss=float(row["best_validation_loss"]),
                improved_best=bool(row["improved_best"]),
                significant_improvement=bool(row["significant_improvement"]),
                optimizer_steps=int(row["optimizer_steps"]),
            )
            for row in history
        ),
    )


def _select_restart(fits: Sequence[RestartFit]) -> RestartFit:
    converged = tuple(fit for fit in fits if fit.result.converged)
    if not converged:
        raise RuntimeError("no full-replay restart reached the frozen convergence rule")
    return min(
        converged,
        key=lambda fit: (
            fit.result.best_validation_loss,
            -fit.result.best_validation_accuracy,
            fit.restart,
        ),
    )


def _training_row(
    fit: RestartFit,
    selected: bool,
    run_seed: int,
    macro_step: int,
    active_nodes: int,
    training_examples: int,
    validation_examples: int,
) -> dict[str, object]:
    return {
        **fit.result.as_record(include_history=False),
        "active_node_count": active_nodes,
        "all_cumulative_training_examples": True,
        "condition": CEILING_CONDITION,
        "fit_seed": fit.fit_seed,
        "fresh_initialization": True,
        "macro_step": macro_step,
        "parameter_sha256": fit.parameter_sha256,
        "restart": fit.restart,
        "row_type": "training",
        "run_seed": run_seed,
        "selected": selected,
        "test_labels_used_for_selection": False,
        "training_examples": training_examples,
        "validation_examples": validation_examples,
        "validation_updates": 0,
    }


def _evaluate_step(
    config: VampLogTIntegratorCeilingConfig,
    phase_name: str,
    seed: int,
    macro_step: int,
    benchmark: PermutedMnistBenchmark,
    dependency: FrozenClassifierDependency,
    bank: ActiveAdapterBank,
    integrator: LevelSlotIntegrator,
    parent_micro: Mapping[tuple[int, str, str], Mapping[str, object]],
    device: torch.device,
) -> tuple[tuple[dict[str, object], ...], int]:
    observed_domains = tuple(
        sorted({allocation.domain_id for allocation in benchmark.allocations[:macro_step]})
    )
    datasets = [
        (
            "test_subset",
            concatenate_batches(
                tuple(
                    benchmark.test_domain(domain, full=False)
                    for domain in observed_domains
                )
            ),
        )
    ]
    if phase_name == "primary" and macro_step in config.evaluation.full_checkpoints:
        datasets.append(
            (
                "full_test",
                concatenate_batches(
                    tuple(
                        benchmark.test_domain(domain, full=True)
                        for domain in observed_domains
                    )
                ),
            )
        )
    rows = []
    total_examples = 0
    for scope, examples in datasets:
        supervision = integrator_supervision(
            config, dependency, bank, examples, device, base_only=False
        )
        condition_logits = {
            CEILING_CONDITION: prediction_logits(
                integrator,
                supervision.observations,
                device,
                config.evaluation.inference_batch_size,
            ),
            **{
                control: fixed_control_logits(
                    control,
                    bank.topology.active_nodes,
                    supervision.observations,
                    examples.labels,
                    named_seed(seed, macro_step, scope, control),
                )
                for control in CEILING_CONTROLS
            },
        }
        metric_rows = {
            condition: prediction_metric_rows(
                condition=condition,
                logits=logits,
                examples=examples,
                node_observations=supervision.observations,
                nodes=bank.topology.active_nodes,
                run_seed=seed,
                macro_step=macro_step,
                evaluation_scope=scope,
            )
            for condition, logits in condition_logits.items()
        }
        parent_match = None
        if phase_name == "primary":
            parent = parent_micro.get((macro_step, scope, "mean_ensemble"))
            new_mean = next(
                row for row in metric_rows["mean_ensemble"] if row["group"] == "micro"
            )
            if parent is None:
                raise ValueError("authenticated parent lacks mean-ensemble coordinates")
            parent_match = all(
                math.isclose(
                    float(parent[field]),
                    float(new_mean[field]),
                    rel_tol=0.0,
                    abs_tol=1.0e-8,
                )
                for field in ("accuracy", "mean_cross_entropy")
            )
            if not parent_match:
                raise RuntimeError("rebuilt frontier differs from the completed parent")
        rows.extend(
            {
                **row,
                "parent_mean_ensemble_match": parent_match,
                "row_type": "evaluation",
                "selection_archive": False,
            }
            for condition_rows in metric_rows.values()
            for row in condition_rows
        )
        total_examples += len(examples.labels)
    return tuple(rows), total_examples


def _training_validation_overlap(
    training: ExampleBatch, validation: ExampleBatch
) -> bool:
    training_ids = set(
        zip(training.domain_ids.tolist(), training.source_indices.tolist(), strict=True)
    )
    validation_ids = set(
        zip(validation.domain_ids.tolist(), validation.source_indices.tolist(), strict=True)
    )
    return bool(training_ids & validation_ids)


def _load_parent_micro_rows(
    config: VampLogTIntegratorCeilingConfig, seed: int
) -> dict[tuple[int, str, str], Mapping[str, object]]:
    path = (
        config.parent_integrator_run_root
        / "primary"
        / f"seed-{seed}"
        / "metrics.jsonl"
    )
    rows = tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    return {
        (int(row["macro_step"]), str(row["evaluation_scope"]), str(row["condition"])): row
        for row in rows
        if row.get("row_type") == "evaluation" and row.get("group") == "micro"
    }


def _save_checkpoint(
    path: Path,
    config: VampLogTIntegratorCeilingConfig,
    phase_name: str,
    seed: int,
    macro_step: int,
    bank: ActiveAdapterBank,
    work: CeilingWorkCounters,
    metric_rows: int,
) -> None:
    atomic_torch_save(
        path,
        {
            "bank": bank_record(bank),
            "completed_macro_step": macro_step,
            "config_hash": config.config_hash,
            "metric_rows": metric_rows,
            "phase": phase_name,
            "run_seed": seed,
            "schema_version": "vamp-logt-integrator-ceiling-checkpoint-v1",
            "work": asdict(work),
        },
    )


def _load_checkpoint(
    path: Path,
    config: VampLogTIntegratorCeilingConfig,
    phase_name: str,
    seed: int,
) -> tuple[ActiveAdapterBank, CeilingWorkCounters, int, int]:
    if not path.is_file():
        return (
            empty_adapter_bank(config.benchmark.model_batch_size),
            CeilingWorkCounters(),
            0,
            0,
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("schema_version")
        != "vamp-logt-integrator-ceiling-checkpoint-v1"
        or payload.get("config_hash") != config.config_hash
        or payload.get("phase") != phase_name
        or payload.get("run_seed") != seed
    ):
        raise ValueError("integrator-ceiling checkpoint coordinates changed")
    return (
        bank_from_record(payload["bank"]),
        CeilingWorkCounters(**payload["work"]),
        int(payload["completed_macro_step"]),
        int(payload["metric_rows"]),
    )


def _retire_completed_restart_parameters(
    root: Path, completed_step: int, retained_steps: set[int]
) -> None:
    if not root.is_dir():
        return
    for step in range(1, completed_step + 1):
        directory = root / f"step-{step}"
        selection_path = directory / "selection.json"
        if not selection_path.is_file():
            continue
        selected = int(load_canonical_json(selection_path)["selected_restart"])
        for path in directory.glob("restart-*.pt"):
            if step not in retained_steps or path.name != f"restart-{selected}.pt":
                path.unlink()


def _require_smoke_gate(
    run_root: Path, config: VampLogTIntegratorCeilingConfig
) -> None:
    path = run_root / "smoke" / f"seed-{config.smoke.seeds[0]}" / "summary.json"
    if not path.is_file():
        raise RuntimeError("the exact integrator-ceiling smoke must complete before primary")
    acceptance = load_canonical_json(path)["acceptance"]
    if not all(bool(value) for value in acceptance.values()):
        raise RuntimeError("the exact integrator-ceiling smoke did not pass its gates")


def _authenticate_parent_integrator(
    config: VampLogTIntegratorCeilingConfig,
) -> dict[str, object]:
    protocol_path = config.parent_integrator_run_root / "protocol.json"
    summary_path = config.parent_integrator_run_root / "summary.json"
    expected_ledgers = dict(config.parent_integrator.primary_metric_ledger_sha256)
    ledger_paths = {
        seed: config.parent_integrator_run_root
        / "primary"
        / f"seed-{seed}"
        / "metrics.jsonl"
        for seed in expected_ledgers
    }
    if (
        not protocol_path.is_file()
        or not summary_path.is_file()
        or file_sha256(protocol_path) != config.parent_integrator.protocol_sha256
        or file_sha256(summary_path) != config.parent_integrator.summary_sha256
        or any(
            not path.is_file() or file_sha256(path) != expected_ledgers[seed]
            for seed, path in ledger_paths.items()
        )
    ):
        raise ValueError("completed parent integrator artifacts changed or are missing")
    protocol = load_canonical_json(protocol_path)
    summary = load_canonical_json(summary_path)
    if (
        protocol.get("config_hash") != config.parent_integrator.run_id
        or summary.get("status") != "complete"
        or summary.get("completed_primary_seeds") != len(config.primary.seeds)
    ):
        raise ValueError("parent integrator is not the frozen completed run")
    return summary


def _write_protocol(
    run_root: Path,
    config: VampLogTIntegratorCeilingConfig,
    dependency: FrozenClassifierDependency,
) -> None:
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - vision environment gate
        raise RuntimeError("PyYAML is required by the vision environment") from error
    resolved = yaml.safe_dump(config.as_record(), sort_keys=False).encode("utf-8")
    config_path = run_root / "config_resolved.yaml"
    if config_path.is_file() and config_path.read_bytes() != resolved:
        raise ValueError("resolved ceiling protocol changed inside one run identity")
    atomic_write(config_path, resolved)
    project_root = Path(__file__).resolve().parents[3]
    material_paths = (
        "configs/vamp_af_mnist/poc.yaml",
        "configs/vamp_logt_router_mnist/primary.yaml",
        "configs/vamp_logt_integrator_permuted_mnist/primary.yaml",
        "configs/vamp_logt_integrator_ceiling_permuted_mnist/primary.yaml",
        "docs/logt_vamp_permuted_mnist_integrator_protocol.md",
        "src/apm/continual/artifacts.py",
        "src/apm/continual/logt_behavioral_integrator.py",
        "src/apm/continual/logt_evidence_bank.py",
        "src/apm/continual/top_two_adapter.py",
        "src/apm/continual/vision/imagenetr/checkpoints.py",
        "src/apm/data/mnist/loader.py",
        "src/apm/data/mnist/permutations.py",
        "src/apm/experiments/vamp_af_config.py",
        "src/apm/experiments/vamp_af_data.py",
        "src/apm/experiments/vamp_logt_evidence_training.py",
        "src/apm/experiments/vamp_logt_integrator_ceiling_permuted_config.py",
        "src/apm/experiments/vamp_logt_integrator_ceiling_permuted_mnist.py",
        "src/apm/experiments/vamp_logt_integrator_ceiling_permuted_reporting.py",
        "src/apm/experiments/vamp_logt_integrator_ceiling_permuted_workflow.py",
        "src/apm/experiments/vamp_logt_integrator_features.py",
        "src/apm/experiments/vamp_logt_integrator_metrics.py",
        "src/apm/experiments/vamp_logt_integrator_permuted_config.py",
        "src/apm/experiments/vamp_logt_router_config.py",
        "src/apm/experiments/vamp_logt_router_data.py",
        "src/apm/experiments/vamp_logt_router_state.py",
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
            "parent_integrator_metric_ledger_sha256": dict(
                config.parent_integrator.primary_metric_ledger_sha256
            ),
            "parent_integrator_protocol_sha256": config.parent_integrator.protocol_sha256,
            "parent_integrator_summary_sha256": config.parent_integrator.summary_sha256,
            "schema_version": "vamp-logt-integrator-ceiling-protocol-v1",
            "torch_version": torch.__version__,
        },
    )


__all__ = [
    "CEILING_CONDITION",
    "CeilingWorkCounters",
    "run_phase_seed",
    "run_workflow",
]

"""Converged full-replay integrator ceiling over the dense hierarchy tape."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import torch

from apm.continual.artifacts import (
    ChainedJsonlLedger,
    file_sha256,
    load_canonical_json,
    publish_immutable_json,
)
from apm.continual.dense_mlp_adapter import (
    DenseExamples,
    DenseMlpState,
    dense_hidden_logits,
    fit_dense_model,
    zero_dense_delta,
)
from apm.continual.logt_behavioral_integrator import (
    ConvergedFullReplayResult,
    FullReplayConvergenceConfig,
    IntegratorConditionState,
    IntegratorSupervision,
    create_condition_state,
    prediction_logits,
    train_converged_full_replay,
)
from apm.continual.vision.imagenetr.checkpoints import atomic_torch_save
from apm.experiments.vamp_logt_integrator_metrics import prediction_metric_rows
from apm.experiments.vamp_logt_mlp_permuted_calibration import load_calibrated_base
from apm.experiments.vamp_logt_mlp_permuted_config import VampLogTDenseConfig
from apm.experiments.vamp_logt_mlp_permuted_data import (
    ExampleBatch,
    build_benchmark,
    concatenate_batches,
    named_seed,
)
from apm.experiments.vamp_logt_mlp_permuted_hierarchy import (
    DenseFrontier,
    build_base_observations,
    build_dense_observations,
    load_frontier,
)


CEILING_CONDITION = "converged_full_replay_integrator"
FROZEN_BASE_CONDITION = "frozen_base_mlp"
CONVERGED_MLP_CONDITION = "converged_cumulative_mlp"
CONVERGED_BASE_INTEGRATOR_CONDITION = "converged_base_only_integrator"
BASELINE_SEED = 0


def run_baseline_extension(
    config: VampLogTDenseConfig,
    run_root: Path,
    device: torch.device,
) -> dict[str, object]:
    """Run the post-hoc seed-zero cumulative baselines at full checkpoints."""
    protocol_path = run_root / "protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError("baseline extension requires the immutable source protocol")
    protocol = load_canonical_json(protocol_path)
    if protocol.get("config_hash") != config.config_hash:
        raise ValueError("baseline extension config differs from the source run")
    hierarchy_summary = run_root / "hierarchy" / f"seed-{BASELINE_SEED}" / "summary.json"
    if not hierarchy_summary.is_file() or load_canonical_json(hierarchy_summary).get("status") != "complete":
        raise RuntimeError("baseline extension requires the complete seed-zero hierarchy tape")
    directory = run_root / "baselines"
    publish_immutable_json(
        directory / "protocol.json",
        {
            "conditions": {
                FROZEN_BASE_CONDITION: {
                    "training": "none; calibrated identity-domain base is frozen",
                },
                CONVERGED_MLP_CONDITION: {
                    "training": "fresh MLP on all cumulative node-training rows",
                    "validation": "all cumulative held-out evaluation rows",
                },
                CONVERGED_BASE_INTEGRATOR_CONDITION: {
                    "features": "frozen calibrated base hidden state and class log probabilities",
                    "training": "fresh integrator on all cumulative observer rows",
                    "validation": "all cumulative held-out evaluation rows",
                },
            },
            "config_hash": config.config_hash,
            "evaluation_checkpoints": list(config.evaluation.full_checkpoints),
            "restarts_per_fit": config.ceiling.restarts_per_step,
            "run_seeds": [BASELINE_SEED],
            "schema_version": "vamp-logt-dense-baseline-extension-protocol-v1",
            "source_protocol_sha256": file_sha256(protocol_path),
            "status": "post-hoc-diagnostic",
            "test_used_for_selection": False,
        },
    )
    base = load_calibrated_base(config, run_root)
    seed_summary = _run_baseline_seed(config, run_root, BASELINE_SEED, base, device)
    summary = {
        "config_hash": config.config_hash,
        "run_seeds": [BASELINE_SEED],
        "schema_version": "vamp-logt-dense-baseline-extension-summary-v1",
        "seed_summaries": [seed_summary],
        "status": "complete",
    }
    publish_immutable_json(directory / "summary.json", summary)
    return summary


def run_ceiling(
    config: VampLogTDenseConfig,
    run_root: Path,
    device: torch.device,
) -> tuple[dict[str, object], ...]:
    """Fit three fresh cumulative integrators at every step for every seed."""
    base = load_calibrated_base(config, run_root)
    if load_canonical_json(run_root / "hierarchy" / "summary.json").get("status") != "complete":
        raise RuntimeError("ceiling requires the complete dense hierarchy tape")
    summaries = tuple(
        _run_ceiling_seed(config, run_root, seed, base, device)
        for seed in config.online.seeds
    )
    publish_immutable_json(
        run_root / "ceiling" / "summary.json",
        {
            "config_hash": config.config_hash,
            "schema_version": "vamp-logt-dense-ceiling-summary-v1",
            "seeds": list(summaries),
            "status": "complete",
        },
    )
    return summaries


def _run_ceiling_seed(
    config: VampLogTDenseConfig,
    run_root: Path,
    seed: int,
    base: DenseMlpState,
    device: torch.device,
) -> dict[str, object]:
    directory = run_root / "ceiling" / f"seed-{seed}"
    summary_path = directory / "summary.json"
    ledger = ChainedJsonlLedger(directory / "metrics.jsonl", "vamp-logt-dense-ceiling-metric-v1")
    checkpoint_path = directory / "state" / "checkpoint.pt"
    completed_step, checkpoint_rows = _load_checkpoint(checkpoint_path, config, seed)
    if ledger.next_sequence < checkpoint_rows:
        raise ValueError("ceiling checkpoint refers to absent metric rows")
    ledger.truncate(checkpoint_rows)
    if summary_path.is_file():
        summary = load_canonical_json(summary_path)
        if (
            completed_step != config.benchmark.macro_steps
            or checkpoint_rows != ledger.next_sequence
            or int(summary.get("final_macro_step", -1)) != completed_step
            or int(summary.get("metric_rows", -1)) != ledger.next_sequence
        ):
            raise ValueError("completed ceiling summary is not covered by its checkpoint and ledger")
        return summary
    benchmark = build_benchmark(config, seed)
    training_batches = tuple(
        benchmark.step(step).observer for step in range(1, completed_step + 1)
    )
    validation_batches = tuple(
        benchmark.step(step).evaluation for step in range(1, completed_step + 1)
    )
    started = perf_counter()
    try:
        from tqdm.auto import tqdm
    except ImportError as error:  # pragma: no cover - environment dependency
        raise RuntimeError("tqdm is required by the vision environment") from error
    progress = tqdm(
        range(completed_step + 1, config.benchmark.macro_steps + 1),
        initial=completed_step,
        total=config.benchmark.macro_steps,
        desc=f"dense ceiling seed {seed}",
        disable=not config.runtime.progress,
        unit="step",
    )
    for macro_step in progress:
        batches = benchmark.step(macro_step)
        training_batches = (*training_batches, batches.observer)
        validation_batches = (*validation_batches, batches.evaluation)
        training_examples = concatenate_batches(training_batches)
        validation_examples = concatenate_batches(validation_batches)
        if set(zip(training_examples.domain_ids.tolist(), training_examples.source_indices.tolist())) & set(
            zip(validation_examples.domain_ids.tolist(), validation_examples.source_indices.tolist())
        ):
            raise RuntimeError("ceiling training and validation allocations overlap")
        frontier = load_frontier(config, run_root, seed, macro_step)
        training_observations = build_dense_observations(
            frontier,
            training_examples,
            base,
            config.observer.maximum_levels,
            config.router.target_temperature,
            device,
            config.observer.inference_batch_size,
        ).integrator
        validation_observations = build_dense_observations(
            frontier,
            validation_examples,
            base,
            config.observer.maximum_levels,
            config.router.target_temperature,
            device,
            config.observer.inference_batch_size,
        ).integrator
        training = IntegratorSupervision(training_observations, training_examples.labels)
        validation = IntegratorSupervision(validation_observations, validation_examples.labels)
        restarts = tuple(
            _fit_or_load_restart(
                config,
                directory,
                seed,
                macro_step,
                restart,
                training,
                validation,
                base,
                device,
            )
            for restart in range(config.ceiling.restarts_per_step)
        )
        converged = tuple(item for item in restarts if item[1].converged)
        if len(converged) != config.ceiling.restarts_per_step:
            raise RuntimeError(
                f"ceiling step {macro_step} seed {seed} reached the epoch cap before convergence"
            )
        selected_restart, selected_result, selected_state = min(
            converged,
            key=lambda item: (
                item[1].best_validation_loss,
                -item[1].best_validation_accuracy,
                item[0],
            ),
        )
        publish_immutable_json(
            directory / "steps" / f"step-{macro_step:03d}" / "selection.json",
            {
                "config_hash": config.config_hash,
                "macro_step": macro_step,
                "schema_version": "vamp-logt-dense-ceiling-selection-v1",
                "selected_restart": selected_restart,
                "selection_metric": "lowest_restored_validation_cross_entropy",
                "test_used_for_selection": False,
                "validation_cross_entropy": selected_result.best_validation_loss,
            },
        )
        evaluation_rows = _evaluate_ceiling_step(
            config,
            seed,
            macro_step,
            benchmark,
            frontier,
            base,
            selected_state,
            validation_examples,
            device,
        )
        convergence_rows = tuple(
            {
                **result.as_record(include_history=False),
                "condition": CEILING_CONDITION,
                "macro_step": macro_step,
                "restart": restart,
                "row_type": "ceiling_convergence",
                "run_seed": seed,
                "selected": restart == selected_restart,
                "training_examples": len(training_examples.labels),
                "validation_examples": len(validation_examples.labels),
            }
            for restart, result, _state in restarts
        )
        accounting_row = {
            "active_node_count": len(frontier.nodes),
            "macro_step": macro_step,
            "row_type": "ceiling_accounting",
            "run_seed": seed,
            "training_feature_node_evals": len(frontier.nodes) * len(training_examples.labels),
            "validation_feature_node_evals": len(frontier.nodes) * len(validation_examples.labels),
        }
        ledger.append_many((*convergence_rows, *evaluation_rows, accounting_row))
        atomic_torch_save(
            checkpoint_path,
            {
                "config_hash": config.config_hash,
                "macro_step": macro_step,
                "metric_rows": ledger.next_sequence,
                "run_seed": seed,
                "schema_version": "vamp-logt-dense-ceiling-checkpoint-v1",
            },
        )
    convergence_rows = tuple(row for row in ledger.rows if row.get("row_type") == "ceiling_convergence")
    accounting_rows = tuple(row for row in ledger.rows if row.get("row_type") == "ceiling_accounting")
    mean_parity = _online_mean_parity(run_root, seed, ledger.rows)
    summary = {
        "acceptance": {
            "all_restarts_converged": all(bool(row["converged"]) for row in convergence_rows),
            "cumulative_training_counts": all(
                int(row["training_examples"]) == int(row["macro_step"]) * config.benchmark.observer_batch_size
                for row in convergence_rows
            ),
            "cumulative_validation_counts": all(
                int(row["validation_examples"]) == int(row["macro_step"]) * config.benchmark.evaluation_batch_size
                for row in convergence_rows
            ),
            "independent_restarts": all(
                {int(row["restart"]) for row in convergence_rows if int(row["macro_step"]) == step}
                == set(range(config.ceiling.restarts_per_step))
                for step in range(1, config.benchmark.macro_steps + 1)
            ),
            "exact_feature_work": all(
                int(row["training_feature_node_evals"])
                == int(row["active_node_count"]) * int(row["macro_step"]) * config.benchmark.observer_batch_size
                and int(row["validation_feature_node_evals"])
                == int(row["active_node_count"]) * int(row["macro_step"]) * config.benchmark.evaluation_batch_size
                for row in accounting_rows
            ),
            "exact_example_presentations": all(
                int(row["training_example_presentations"])
                == int(row["epochs_ran"]) * int(row["training_examples"])
                and int(row["validation_example_presentations"])
                == (int(row["epochs_ran"]) + 2) * int(row["validation_examples"])
                for row in convergence_rows
            ),
            "online_mean_ensemble_parity": mean_parity,
            "validation_only_selection": True,
        },
        "config_hash": config.config_hash,
        "final_macro_step": config.benchmark.macro_steps,
        "metric_rows": ledger.next_sequence,
        "run_seed": seed,
        "schema_version": "vamp-logt-dense-ceiling-seed-v1",
        "status": "complete",
        "wall_seconds": perf_counter() - started,
    }
    publish_immutable_json(summary_path, summary)
    return summary


def _run_baseline_seed(
    config: VampLogTDenseConfig,
    run_root: Path,
    seed: int,
    base: DenseMlpState,
    device: torch.device,
) -> dict[str, object]:
    directory = run_root / "baselines" / f"seed-{seed}"
    summary_path = directory / "summary.json"
    ledger = ChainedJsonlLedger(
        directory / "metrics.jsonl", "vamp-logt-dense-baseline-metric-v1"
    )
    checkpoint_path = directory / "state" / "checkpoint.pt"
    checkpoints = config.evaluation.full_checkpoints
    completed_count, checkpoint_rows = _load_baseline_checkpoint(
        checkpoint_path, config, seed, checkpoints
    )
    if ledger.next_sequence < checkpoint_rows:
        raise ValueError("baseline checkpoint refers to absent metric rows")
    ledger.truncate(checkpoint_rows)
    if summary_path.is_file():
        summary = load_canonical_json(summary_path)
        if (
            completed_count != len(checkpoints)
            or checkpoint_rows != ledger.next_sequence
            or int(summary.get("metric_rows", -1)) != ledger.next_sequence
            or tuple(summary.get("evaluation_checkpoints", ())) != checkpoints
        ):
            raise ValueError("completed baseline summary is not covered by its checkpoint")
        return summary

    benchmark = build_benchmark(config, seed)
    model_batches: tuple[ExampleBatch, ...] = ()
    observer_batches: tuple[ExampleBatch, ...] = ()
    validation_batches: tuple[ExampleBatch, ...] = ()
    previous_step = 0
    if completed_count:
        previous_step = checkpoints[completed_count - 1]
        for macro_step in range(1, previous_step + 1):
            batches = benchmark.step(macro_step)
            model_batches = (*model_batches, batches.model)
            observer_batches = (*observer_batches, batches.observer)
            validation_batches = (*validation_batches, batches.evaluation)

    started = perf_counter()
    try:
        from tqdm.auto import tqdm
    except ImportError as error:  # pragma: no cover - environment dependency
        raise RuntimeError("tqdm is required by the vision environment") from error
    progress = tqdm(
        checkpoints[completed_count:],
        initial=completed_count,
        total=len(checkpoints),
        desc=f"dense cumulative baselines seed {seed}",
        disable=not config.runtime.progress,
        unit="checkpoint",
    )
    for macro_step in progress:
        for step in range(previous_step + 1, macro_step + 1):
            batches = benchmark.step(step)
            model_batches = (*model_batches, batches.model)
            observer_batches = (*observer_batches, batches.observer)
            validation_batches = (*validation_batches, batches.evaluation)
        previous_step = macro_step
        model_archive = concatenate_batches(model_batches)
        observer_archive = concatenate_batches(observer_batches)
        validation_archive = concatenate_batches(validation_batches)
        if _example_coordinates(model_archive) & _example_coordinates(validation_archive):
            raise RuntimeError("cumulative MLP training and validation allocations overlap")
        if _example_coordinates(observer_archive) & _example_coordinates(validation_archive):
            raise RuntimeError("base-only integrator training and validation allocations overlap")

        identity = (torch.arange(784, dtype=torch.int64),)
        mlp_training = DenseExamples(
            model_archive.images, model_archive.labels, identity
        )
        mlp_validation = DenseExamples(
            validation_archive.images, validation_archive.labels, identity
        )
        mlp_restarts = tuple(
            _fit_or_load_cumulative_mlp_restart(
                config,
                directory,
                seed,
                macro_step,
                restart,
                mlp_training,
                mlp_validation,
                base,
                device,
            )
            for restart in range(config.ceiling.restarts_per_step)
        )
        if not all(bool(record["converged"]) for _restart, record, _state in mlp_restarts):
            raise RuntimeError(
                f"cumulative MLP step {macro_step} seed {seed} reached the epoch cap"
            )
        selected_mlp_restart, selected_mlp_record, selected_mlp = min(
            mlp_restarts,
            key=lambda item: (
                float(item[1]["best_validation_loss"]),
                -float(item[1]["best_validation_accuracy"]),
                item[0],
            ),
        )

        integrator_training_observations = build_base_observations(
            observer_archive,
            base,
            config.observer.maximum_levels,
            device,
            config.observer.inference_batch_size,
        )
        integrator_validation_observations = build_base_observations(
            validation_archive,
            base,
            config.observer.maximum_levels,
            device,
            config.observer.inference_batch_size,
        )
        integrator_training = IntegratorSupervision(
            integrator_training_observations, observer_archive.labels
        )
        integrator_validation = IntegratorSupervision(
            integrator_validation_observations, validation_archive.labels
        )
        integrator_restarts = tuple(
            _fit_or_load_base_integrator_restart(
                config,
                directory,
                seed,
                macro_step,
                restart,
                integrator_training,
                integrator_validation,
                base,
                device,
            )
            for restart in range(config.ceiling.restarts_per_step)
        )
        if not all(result.converged for _restart, result, _state in integrator_restarts):
            raise RuntimeError(
                f"base-only integrator step {macro_step} seed {seed} reached the epoch cap"
            )
        selected_integrator_restart, selected_integrator_result, selected_integrator = min(
            integrator_restarts,
            key=lambda item: (
                item[1].best_validation_loss,
                -item[1].best_validation_accuracy,
                item[0],
            ),
        )
        publish_immutable_json(
            directory / "steps" / f"step-{macro_step:03d}" / "selection.json",
            {
                "conditions": {
                    CONVERGED_MLP_CONDITION: {
                        "selected_restart": selected_mlp_restart,
                        "validation_accuracy": selected_mlp_record[
                            "best_validation_accuracy"
                        ],
                        "validation_cross_entropy": selected_mlp_record[
                            "best_validation_loss"
                        ],
                    },
                    CONVERGED_BASE_INTEGRATOR_CONDITION: {
                        "selected_restart": selected_integrator_restart,
                        "validation_accuracy": selected_integrator_result.best_validation_accuracy,
                        "validation_cross_entropy": selected_integrator_result.best_validation_loss,
                    },
                },
                "config_hash": config.config_hash,
                "macro_step": macro_step,
                "schema_version": "vamp-logt-dense-baseline-selection-v1",
                "selection_metric": "lowest_restored_validation_cross_entropy",
                "test_used_for_selection": False,
            },
        )
        frontier = load_frontier(config, run_root, seed, macro_step)
        evaluation_rows = _evaluate_baseline_step(
            config,
            seed,
            macro_step,
            benchmark,
            frontier,
            base,
            selected_mlp,
            selected_integrator,
            validation_archive,
            device,
        )
        mlp_convergence_rows = tuple(
            {
                **{key: value for key, value in record.items() if key != "history"},
                "condition": CONVERGED_MLP_CONDITION,
                "macro_step": macro_step,
                "restart": restart,
                "row_type": "baseline_convergence",
                "run_seed": seed,
                "selected": restart == selected_mlp_restart,
                "training_examples": len(mlp_training),
                "validation_examples": len(mlp_validation),
            }
            for restart, record, _state in mlp_restarts
        )
        integrator_convergence_rows = tuple(
            {
                **result.as_record(include_history=False),
                "condition": CONVERGED_BASE_INTEGRATOR_CONDITION,
                "macro_step": macro_step,
                "restart": restart,
                "row_type": "baseline_convergence",
                "run_seed": seed,
                "selected": restart == selected_integrator_restart,
                "training_examples": len(integrator_training.labels),
                "validation_examples": len(integrator_validation.labels),
            }
            for restart, result, _state in integrator_restarts
        )
        accounting_row = {
            "base_integrator_training_examples": len(integrator_training.labels),
            "base_integrator_validation_examples": len(integrator_validation.labels),
            "cumulative_mlp_training_examples": len(mlp_training),
            "cumulative_mlp_validation_examples": len(mlp_validation),
            "macro_step": macro_step,
            "restarts_per_fit": config.ceiling.restarts_per_step,
            "row_type": "baseline_accounting",
            "run_seed": seed,
        }
        ledger.append_many(
            (
                *mlp_convergence_rows,
                *integrator_convergence_rows,
                *evaluation_rows,
                accounting_row,
            )
        )
        completed_count += 1
        atomic_torch_save(
            checkpoint_path,
            {
                "completed_checkpoint_count": completed_count,
                "config_hash": config.config_hash,
                "last_macro_step": macro_step,
                "metric_rows": ledger.next_sequence,
                "run_seed": seed,
                "schema_version": "vamp-logt-dense-baseline-checkpoint-v1",
            },
        )
        del integrator_training_observations, integrator_validation_observations
        del integrator_training, integrator_validation, integrator_restarts
        if device.type == "cuda":
            torch.cuda.empty_cache()

    convergence_rows = tuple(
        row for row in ledger.rows if row.get("row_type") == "baseline_convergence"
    )
    evaluation_rows = tuple(
        row for row in ledger.rows if row.get("row_type") == "baseline_evaluation"
    )
    summary = {
        "acceptance": {
            "all_restarts_converged": all(bool(row["converged"]) for row in convergence_rows),
            "all_test_rows_excluded_from_selection": all(
                not bool(row["test_used_for_selection"]) for row in evaluation_rows
            ),
            "cumulative_training_counts": all(
                int(row["training_examples"])
                == int(row["macro_step"])
                * (
                    config.benchmark.model_batch_size
                    if row["condition"] == CONVERGED_MLP_CONDITION
                    else config.benchmark.observer_batch_size
                )
                for row in convergence_rows
            ),
            "cumulative_validation_counts": all(
                int(row["validation_examples"])
                == int(row["macro_step"]) * config.benchmark.evaluation_batch_size
                for row in convergence_rows
            ),
            "exact_checkpoint_set": sorted(
                {int(row["macro_step"]) for row in convergence_rows}
            )
            == list(checkpoints),
            "exact_example_presentations": all(
                int(row["training_example_presentations"])
                == int(row["epochs_ran"]) * int(row["training_examples"])
                and int(row["validation_example_presentations"])
                == (
                    int(row["epochs_ran"])
                    + (2 if row["condition"] == CONVERGED_BASE_INTEGRATOR_CONDITION else 0)
                )
                * int(row["validation_examples"])
                for row in convergence_rows
            ),
            "independent_restarts": all(
                {
                    int(row["restart"])
                    for row in convergence_rows
                    if int(row["macro_step"]) == macro_step
                    and row["condition"] == condition
                }
                == set(range(config.ceiling.restarts_per_step))
                for macro_step in checkpoints
                for condition in (
                    CONVERGED_MLP_CONDITION,
                    CONVERGED_BASE_INTEGRATOR_CONDITION,
                )
            ),
            "one_selected_restart_per_fit": all(
                sum(
                    bool(row["selected"])
                    for row in convergence_rows
                    if int(row["macro_step"]) == macro_step
                    and row["condition"] == condition
                )
                == 1
                for macro_step in checkpoints
                for condition in (
                    CONVERGED_MLP_CONDITION,
                    CONVERGED_BASE_INTEGRATOR_CONDITION,
                )
            ),
            "single_seed_only": seed == BASELINE_SEED,
            "validation_only_selection": True,
        },
        "config_hash": config.config_hash,
        "evaluation_checkpoints": list(checkpoints),
        "metric_rows": ledger.next_sequence,
        "run_seed": seed,
        "schema_version": "vamp-logt-dense-baseline-seed-v1",
        "status": "complete",
        "wall_seconds": perf_counter() - started,
    }
    publish_immutable_json(summary_path, summary)
    return summary


def _fit_or_load_cumulative_mlp_restart(
    config: VampLogTDenseConfig,
    directory: Path,
    seed: int,
    macro_step: int,
    restart: int,
    training: DenseExamples,
    validation: DenseExamples,
    base: DenseMlpState,
    device: torch.device,
) -> tuple[int, dict[str, object], DenseMlpState]:
    root = directory / "steps" / f"step-{macro_step:03d}" / "cumulative-mlp"
    model_path = root / f"restart-{restart}.pt"
    result_path = root / f"restart-{restart}.json"
    if model_path.is_file() and result_path.is_file():
        record = load_canonical_json(result_path)
        if (
            record.get("config_hash") != config.config_hash
            or record.get("condition") != CONVERGED_MLP_CONDITION
            or int(record.get("macro_step", -1)) != macro_step
            or int(record.get("restart", -1)) != restart
            or int(record.get("run_seed", -1)) != seed
            or file_sha256(model_path) != record.get("checkpoint_sha256")
        ):
            raise ValueError("stored cumulative MLP restart coordinates changed")
        payload = torch.load(model_path, map_location="cpu", weights_only=True)
        return restart, record, DenseMlpState(tuple(payload["parameters"]))

    fit_seed = named_seed(seed, "baseline", CONVERGED_MLP_CONDITION, macro_step, restart)
    result = fit_dense_model(
        training,
        base,
        config.node.optimizer,
        fit_seed,
        device,
        validation=validation,
        convergence=config.ceiling.convergence,
        dropout=config.node.dropout,
        progress_label=f"cumulative MLP seed {seed} step {macro_step} restart {restart}",
        progress=config.runtime.progress,
    )
    best = result.history[result.best_epoch - 1]
    if best.validation_loss is None or best.validation_accuracy is None:
        raise RuntimeError("converged cumulative MLP lacks validation evidence")
    atomic_torch_save(
        model_path,
        {
            "condition": CONVERGED_MLP_CONDITION,
            "config_hash": config.config_hash,
            "macro_step": macro_step,
            "parameters": result.state.tensors,
            "restart": restart,
            "run_seed": seed,
            "schema_version": "vamp-logt-dense-cumulative-mlp-restart-v1",
        },
    )
    record = {
        "best_epoch": result.best_epoch,
        "best_training_accuracy": best.training_accuracy,
        "best_training_loss": best.training_loss,
        "best_validation_accuracy": best.validation_accuracy,
        "best_validation_loss": best.validation_loss,
        "checkpoint_sha256": file_sha256(model_path),
        "condition": CONVERGED_MLP_CONDITION,
        "config_hash": config.config_hash,
        "converged": result.stop_reason == "minimum_learning_rate_plateau",
        "epochs_ran": result.epochs_ran,
        "final_learning_rate": result.history[-1].learning_rate,
        "history": [asdict(row) for row in result.history],
        "macro_step": macro_step,
        "optimizer_steps": result.optimizer_steps,
        "restart": restart,
        "run_seed": seed,
        "schema_version": "vamp-logt-dense-cumulative-mlp-restart-result-v1",
        "stop_reason": result.stop_reason,
        "training_example_presentations": result.training_example_presentations,
        "validation_example_presentations": result.validation_example_presentations,
    }
    publish_immutable_json(result_path, record)
    return restart, record, result.state


def _fit_or_load_base_integrator_restart(
    config: VampLogTDenseConfig,
    directory: Path,
    seed: int,
    macro_step: int,
    restart: int,
    training: IntegratorSupervision,
    validation: IntegratorSupervision,
    base: DenseMlpState,
    device: torch.device,
) -> tuple[int, ConvergedFullReplayResult, IntegratorConditionState]:
    root = directory / "steps" / f"step-{macro_step:03d}" / "base-only-integrator"
    model_path = root / f"restart-{restart}.pt"
    result_path = root / f"restart-{restart}.json"
    slot_dim = base.embedding_dim + 11
    input_dim = config.observer.maximum_levels * slot_dim
    name = f"baseline-base-only-step-{macro_step}-restart-{restart}"
    state = create_condition_state(name, input_dim, slot_dim, config.integrator, seed, device)
    if model_path.is_file() and result_path.is_file():
        record = load_canonical_json(result_path)
        if (
            record.get("config_hash") != config.config_hash
            or record.get("condition") != CONVERGED_BASE_INTEGRATOR_CONDITION
            or int(record.get("macro_step", -1)) != macro_step
            or int(record.get("restart", -1)) != restart
            or int(record.get("run_seed", -1)) != seed
            or file_sha256(model_path) != record.get("checkpoint_sha256")
        ):
            raise ValueError("stored base-only integrator restart coordinates changed")
        payload = torch.load(model_path, map_location=device, weights_only=True)
        state.integrator.load_state_dict(payload["model"], strict=True)
        state.optimizer_steps = int(record["optimizer_steps"])
        result = ConvergedFullReplayResult(
            bool(record["converged"]),
            str(record["stop_reason"]),
            int(record["best_epoch"]),
            int(record["epochs_ran"]),
            float(record["best_training_loss"]),
            float(record["best_training_accuracy"]),
            float(record["best_validation_loss"]),
            float(record["best_validation_accuracy"]),
            float(record["final_learning_rate"]),
            int(record["optimizer_steps"]),
            int(record["training_example_presentations"]),
            int(record["validation_example_presentations"]),
            tuple(_convergence_epoch(row) for row in record["history"]),
        )
        return restart, result, state

    convergence = FullReplayConvergenceConfig(
        config.ceiling.convergence.minimum_epochs,
        config.ceiling.convergence.maximum_epochs,
        config.ceiling.convergence.improvement_delta,
        config.ceiling.convergence.learning_rate_patience,
        config.ceiling.convergence.learning_rate_factor,
        config.ceiling.convergence.minimum_learning_rate,
        config.ceiling.convergence.convergence_patience,
    )
    result = train_converged_full_replay(
        state,
        training,
        validation,
        config.integrator,
        convergence,
        seed,
        macro_step,
        device,
        progress=config.runtime.progress,
    )
    atomic_torch_save(
        model_path,
        {
            "condition": CONVERGED_BASE_INTEGRATOR_CONDITION,
            "config_hash": config.config_hash,
            "macro_step": macro_step,
            "model": {
                name: tensor.detach().cpu().clone()
                for name, tensor in state.integrator.state_dict().items()
            },
            "restart": restart,
            "run_seed": seed,
            "schema_version": "vamp-logt-dense-base-integrator-restart-v1",
        },
    )
    publish_immutable_json(
        result_path,
        {
            **result.as_record(include_history=True),
            "checkpoint_sha256": file_sha256(model_path),
            "condition": CONVERGED_BASE_INTEGRATOR_CONDITION,
            "config_hash": config.config_hash,
            "macro_step": macro_step,
            "restart": restart,
            "run_seed": seed,
            "schema_version": "vamp-logt-dense-base-integrator-restart-result-v1",
        },
    )
    return restart, result, state


def _evaluate_baseline_step(
    config: VampLogTDenseConfig,
    seed: int,
    macro_step: int,
    benchmark,
    frontier: DenseFrontier,
    base: DenseMlpState,
    cumulative_mlp: DenseMlpState,
    base_integrator: IntegratorConditionState,
    validation_archive: ExampleBatch,
    device: torch.device,
) -> tuple[dict[str, object], ...]:
    seen_domains = tuple(
        sorted({allocation.domain_id for allocation in benchmark.allocations[:macro_step]})
    )
    scopes = [
        ("validation_archive", validation_archive),
        (
            "test_subset",
            concatenate_batches(
                tuple(benchmark.test_domain(domain, full=False) for domain in seen_domains)
            ),
        ),
        *(
            ("full_test", benchmark.test_domain(domain, full=True))
            for domain in seen_domains
        ),
    ]
    rows = []
    for scope, examples in scopes:
        node_observations = build_dense_observations(
            frontier,
            examples,
            base,
            config.observer.maximum_levels,
            config.router.target_temperature,
            device,
            config.observer.inference_batch_size,
        ).integrator
        base_observations = build_base_observations(
            examples,
            base,
            config.observer.maximum_levels,
            device,
            config.observer.inference_batch_size,
        )
        mlp_logits = _dense_state_logits(
            examples.images,
            cumulative_mlp,
            device,
            config.observer.inference_batch_size,
        )
        condition_logits = (
            (FROZEN_BASE_CONDITION, base_observations.baseline_log_probabilities),
            (CONVERGED_MLP_CONDITION, mlp_logits),
            (
                CONVERGED_BASE_INTEGRATOR_CONDITION,
                prediction_logits(
                    base_integrator.integrator,
                    base_observations,
                    device,
                    config.observer.inference_batch_size,
                ),
            ),
        )
        for condition, logits in condition_logits:
            rows.extend(
                {
                    **row,
                    "post_hoc_single_seed": True,
                    "row_type": "baseline_evaluation",
                    "test_used_for_selection": False,
                }
                for row in prediction_metric_rows(
                    condition=condition,
                    logits=logits,
                    examples=examples,
                    node_observations=node_observations,
                    nodes=frontier.nodes,
                    run_seed=seed,
                    macro_step=macro_step,
                    evaluation_scope=scope,
                    joint_logits=mlp_logits,
                )
            )
    return tuple(rows)


def _dense_state_logits(
    images: torch.Tensor,
    state: DenseMlpState,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    target = DenseMlpState(tuple(tensor.to(device) for tensor in state.tensors))
    zero = zero_dense_delta(target)
    rows = []
    with torch.inference_mode():
        for offset in range(0, len(images), batch_size):
            rows.append(
                dense_hidden_logits(
                    images[offset : offset + batch_size].flatten(1).to(device),
                    target,
                    zero,
                )[1].cpu()
            )
    return torch.cat(rows)


def _example_coordinates(examples: ExampleBatch) -> set[tuple[int, int]]:
    return set(zip(examples.domain_ids.tolist(), examples.source_indices.tolist()))


def _load_baseline_checkpoint(
    path: Path,
    config: VampLogTDenseConfig,
    seed: int,
    checkpoints: tuple[int, ...],
) -> tuple[int, int]:
    if not path.is_file():
        return 0, 0
    payload = torch.load(path, map_location="cpu", weights_only=True)
    completed = int(payload.get("completed_checkpoint_count", -1))
    expected_last = (
        0
        if completed == 0
        else checkpoints[completed - 1]
        if 0 < completed <= len(checkpoints)
        else -1
    )
    if (
        payload.get("schema_version") != "vamp-logt-dense-baseline-checkpoint-v1"
        or payload.get("config_hash") != config.config_hash
        or int(payload.get("run_seed", -1)) != seed
        or not 0 <= completed <= len(checkpoints)
        or int(payload.get("last_macro_step", -1)) != expected_last
    ):
        raise ValueError("dense baseline checkpoint coordinates changed")
    return completed, int(payload["metric_rows"])


def _fit_or_load_restart(
    config: VampLogTDenseConfig,
    directory: Path,
    seed: int,
    macro_step: int,
    restart: int,
    training: IntegratorSupervision,
    validation: IntegratorSupervision,
    base: DenseMlpState,
    device: torch.device,
) -> tuple[int, ConvergedFullReplayResult, IntegratorConditionState]:
    root = directory / "steps" / f"step-{macro_step:03d}"
    model_path = root / f"restart-{restart}.pt"
    result_path = root / f"restart-{restart}.json"
    slot_dim = base.embedding_dim + 11
    input_dim = config.observer.maximum_levels * slot_dim
    name = f"ceiling-step-{macro_step}-restart-{restart}"
    state = create_condition_state(name, input_dim, slot_dim, config.integrator, seed, device)
    if model_path.is_file() and result_path.is_file():
        record = load_canonical_json(result_path)
        if (
            record.get("config_hash") != config.config_hash
            or int(record.get("macro_step", -1)) != macro_step
            or int(record.get("restart", -1)) != restart
            or file_sha256(model_path) != record.get("checkpoint_sha256")
        ):
            raise ValueError("stored ceiling restart coordinates changed")
        payload = torch.load(model_path, map_location=device, weights_only=True)
        state.integrator.load_state_dict(payload["model"], strict=True)
        state.optimizer_steps = int(record["optimizer_steps"])
        result = ConvergedFullReplayResult(
            bool(record["converged"]),
            str(record["stop_reason"]),
            int(record["best_epoch"]),
            int(record["epochs_ran"]),
            float(record["best_training_loss"]),
            float(record["best_training_accuracy"]),
            float(record["best_validation_loss"]),
            float(record["best_validation_accuracy"]),
            float(record["final_learning_rate"]),
            int(record["optimizer_steps"]),
            int(record["training_example_presentations"]),
            int(record["validation_example_presentations"]),
            tuple(_convergence_epoch(row) for row in record["history"]),
        )
        return restart, result, state
    convergence = FullReplayConvergenceConfig(
        config.ceiling.convergence.minimum_epochs,
        config.ceiling.convergence.maximum_epochs,
        config.ceiling.convergence.improvement_delta,
        config.ceiling.convergence.learning_rate_patience,
        config.ceiling.convergence.learning_rate_factor,
        config.ceiling.convergence.minimum_learning_rate,
        config.ceiling.convergence.convergence_patience,
    )
    result = train_converged_full_replay(
        state,
        training,
        validation,
        config.integrator,
        convergence,
        seed,
        macro_step,
        device,
        progress=config.runtime.progress,
    )
    atomic_torch_save(
        model_path,
        {
            "config_hash": config.config_hash,
            "macro_step": macro_step,
            "model": {
                name: tensor.detach().cpu().clone()
                for name, tensor in state.integrator.state_dict().items()
            },
            "restart": restart,
            "schema_version": "vamp-logt-dense-ceiling-restart-v1",
        },
    )
    publish_immutable_json(
        result_path,
        {
            **result.as_record(include_history=True),
            "checkpoint_sha256": file_sha256(model_path),
            "config_hash": config.config_hash,
            "macro_step": macro_step,
            "restart": restart,
            "schema_version": "vamp-logt-dense-ceiling-restart-result-v1",
        },
    )
    return restart, result, state


def _evaluate_ceiling_step(
    config: VampLogTDenseConfig,
    seed: int,
    macro_step: int,
    benchmark,
    frontier: DenseFrontier,
    base: DenseMlpState,
    state: IntegratorConditionState,
    validation_archive: ExampleBatch,
    device: torch.device,
) -> tuple[dict[str, object], ...]:
    seen_domains = tuple(
        sorted({allocation.domain_id for allocation in benchmark.allocations[:macro_step]})
    )
    scopes = [
        ("validation_archive", validation_archive),
        (
            "test_subset",
            concatenate_batches(tuple(benchmark.test_domain(domain, full=False) for domain in seen_domains)),
        ),
    ]
    if macro_step in config.evaluation.full_checkpoints:
        scopes.extend(
            ("full_test", benchmark.test_domain(domain, full=True))
            for domain in seen_domains
        )
    rows = []
    for scope, examples in scopes:
        observations = build_dense_observations(
            frontier,
            examples,
            base,
            config.observer.maximum_levels,
            config.router.target_temperature,
            device,
            config.observer.inference_batch_size,
        ).integrator
        rows.extend(
            {
                **row,
                "row_type": "ceiling_evaluation",
                "test_used_for_selection": False,
            }
            for row in prediction_metric_rows(
                condition=CEILING_CONDITION,
                logits=prediction_logits(
                    state.integrator,
                    observations,
                    device,
                    config.observer.inference_batch_size,
                ),
                examples=examples,
                node_observations=observations,
                nodes=frontier.nodes,
                run_seed=seed,
                macro_step=macro_step,
                evaluation_scope=scope,
                joint_logits=None,
            )
        )
        rows.extend(
            {
                **row,
                "row_type": "ceiling_evaluation",
                "test_used_for_selection": False,
            }
            for row in prediction_metric_rows(
                condition="reconstructed_mean_ensemble",
                logits=observations.baseline_log_probabilities,
                examples=examples,
                node_observations=observations,
                nodes=frontier.nodes,
                run_seed=seed,
                macro_step=macro_step,
                evaluation_scope=scope,
                joint_logits=None,
            )
        )
    return tuple(rows)


def _online_mean_parity(
    run_root: Path,
    seed: int,
    ceiling_rows,
) -> bool:
    online_path = run_root / "online" / f"seed-{seed}" / "metrics.jsonl"
    if not online_path.is_file():
        return False
    online = ChainedJsonlLedger(
        online_path, "vamp-logt-dense-online-metric-v1"
    ).rows
    expected = {
        int(row["macro_step"]): (float(row["accuracy"]), float(row["mean_cross_entropy"]))
        for row in online
        if row.get("row_type") == "integrator_evaluation"
        and row.get("condition") == "mean_ensemble"
        and row.get("evaluation_scope") == "test_subset"
        and row.get("group") == "micro"
    }
    reconstructed = {
        int(row["macro_step"]): (float(row["accuracy"]), float(row["mean_cross_entropy"]))
        for row in ceiling_rows
        if row.get("row_type") == "ceiling_evaluation"
        and row.get("condition") == "reconstructed_mean_ensemble"
        and row.get("evaluation_scope") == "test_subset"
        and row.get("group") == "micro"
    }
    return set(expected) == set(range(1, max(expected, default=0) + 1)) and expected == reconstructed


def _load_checkpoint(
    path: Path,
    config: VampLogTDenseConfig,
    seed: int,
) -> tuple[int, int]:
    if not path.is_file():
        return 0, 0
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("schema_version") != "vamp-logt-dense-ceiling-checkpoint-v1"
        or payload.get("config_hash") != config.config_hash
        or int(payload.get("run_seed", -1)) != seed
    ):
        raise ValueError("dense ceiling checkpoint coordinates changed")
    return int(payload["macro_step"]), int(payload["metric_rows"])


def _convergence_epoch(row: object):
    from apm.continual.logt_behavioral_integrator import ConvergenceEpochResult

    if not isinstance(row, dict):
        raise ValueError("ceiling convergence history is malformed")
    return ConvergenceEpochResult(
        int(row["epoch"]),
        float(row["learning_rate"]),
        float(row["next_learning_rate"]),
        float(row["training_loss"]),
        float(row["training_accuracy"]),
        float(row["validation_loss"]),
        float(row["validation_accuracy"]),
        float(row["best_validation_loss"]),
        bool(row["improved_best"]),
        bool(row["significant_improvement"]),
        int(row["optimizer_steps"]),
    )


__all__ = [
    "BASELINE_SEED",
    "CEILING_CONDITION",
    "CONVERGED_BASE_INTEGRATOR_CONDITION",
    "CONVERGED_MLP_CONDITION",
    "FROZEN_BASE_CONDITION",
    "run_baseline_extension",
    "run_ceiling",
]

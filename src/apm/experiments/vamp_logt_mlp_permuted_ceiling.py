"""Converged full-replay integrator ceiling over the dense hierarchy tape."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import torch

from apm.continual.artifacts import (
    ChainedJsonlLedger,
    file_sha256,
    load_canonical_json,
    publish_immutable_json,
)
from apm.continual.dense_mlp_adapter import DenseMlpState
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
)
from apm.experiments.vamp_logt_mlp_permuted_hierarchy import (
    DenseFrontier,
    build_dense_observations,
    load_frontier,
)


CEILING_CONDITION = "converged_full_replay_integrator"


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


__all__ = ["CEILING_CONDITION", "run_ceiling"]

"""Paired router and prediction-integrator training on one dense hierarchy tape."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch
from torch import Tensor

from apm.continual.artifacts import ChainedJsonlLedger, load_canonical_json, publish_immutable_json
from apm.continual.dense_mlp_adapter import (
    DenseExamples,
    DenseMlpState,
    dense_delta,
    dense_hidden_logits,
    fit_dense_model,
)
from apm.continual.logt_behavioral_integrator import (
    IntegratorConditionState,
    IntegratorObservations,
    IntegratorSupervision,
    create_condition_state as create_integrator_state,
    inactive_slots_are_zero,
    prediction_logits,
    train_condition as train_integrator,
)
from apm.continual.logt_behavioral_router import (
    RouterConditionState,
    RouterSupervision,
    create_condition_state as create_router_state,
    router_selections,
    sample_example_balanced,
    sample_range_balanced,
    train_condition as train_router,
)
from apm.continual.vision.imagenetr.checkpoints import atomic_torch_save
from apm.experiments.vamp_logt_integrator_metrics import (
    FIXED_CONTROLS,
    fixed_control_logits,
    prediction_metric_rows,
)
from apm.experiments.vamp_logt_mlp_permuted_calibration import load_calibrated_base
from apm.experiments.vamp_logt_mlp_permuted_config import VampLogTDenseConfig
from apm.experiments.vamp_logt_mlp_permuted_data import (
    ExampleBatch,
    PermutedMnistBenchmark,
    build_benchmark,
    concatenate_batches,
    named_seed,
)
from apm.experiments.vamp_logt_mlp_permuted_hierarchy import (
    DenseFrontier,
    DenseObservations,
    build_base_observations,
    build_dense_observations,
    load_frontier,
)
from apm.experiments.vamp_logt_router_metrics import (
    fixed_policy_selections,
    routing_metric_rows,
)


ROUTER_FIXED_CONTROLS = ("most_recent_range", "largest_range", "uniform_active", "oracle")


@dataclass(frozen=True, slots=True)
class OnlineSeedResult:
    """One completed/resumed seed directory and immutable summary."""

    directory: Path
    summary: dict[str, object]


def run_online(
    config: VampLogTDenseConfig,
    run_root: Path,
    device: torch.device,
) -> tuple[OnlineSeedResult, ...]:
    """Train every paired online condition against each shared hierarchy tape."""
    base = load_calibrated_base(config, run_root)
    hierarchy_summary = load_canonical_json(run_root / "hierarchy" / "summary.json")
    if hierarchy_summary.get("status") != "complete":
        raise RuntimeError("online phase requires the complete dense hierarchy tape")
    results = tuple(
        run_online_seed(config, run_root, seed, base, device)
        for seed in config.online.seeds
    )
    publish_immutable_json(
        run_root / "online" / "summary.json",
        {
            "config_hash": config.config_hash,
            "schema_version": "vamp-logt-dense-online-summary-v1",
            "seeds": [result.summary for result in results],
            "status": "complete",
        },
    )
    return results


def run_online_seed(
    config: VampLogTDenseConfig,
    run_root: Path,
    seed: int,
    base: DenseMlpState,
    device: torch.device,
) -> OnlineSeedResult:
    """Run or exactly resume one seed's persistent routers and integrators."""
    directory = run_root / "online" / f"seed-{seed}"
    summary_path = directory / "summary.json"
    checkpoint_path = directory / "state" / "checkpoint.pt"
    ledger = ChainedJsonlLedger(directory / "metrics.jsonl", "vamp-logt-dense-online-metric-v1")
    benchmark = build_benchmark(config, seed)
    slot_dim = base.embedding_dim + 11
    input_dim = config.observer.maximum_levels * slot_dim
    routers = {
        name: create_router_state(name, input_dim, config.router, seed, device)
        for name in config.router.conditions
    }
    integrators = {
        name: create_integrator_state(name, input_dim, slot_dim, config.integrator, seed, device)
        for name in config.integrator.conditions
    }
    completed_step, checkpoint_rows = _load_online_checkpoint(
        checkpoint_path, config, seed, routers, integrators, device
    )
    if ledger.next_sequence < checkpoint_rows:
        raise ValueError("dense online checkpoint refers to absent metric rows")
    ledger.truncate(checkpoint_rows)
    if summary_path.is_file():
        summary = load_canonical_json(summary_path)
        if (
            completed_step != config.benchmark.macro_steps
            or checkpoint_rows != ledger.next_sequence
            or int(summary.get("final_macro_step", -1)) != completed_step
            or int(summary.get("metric_rows", -1)) != ledger.next_sequence
        ):
            raise ValueError("completed dense online summary is not covered by its checkpoint and ledger")
        return OnlineSeedResult(directory, summary)
    observer_batches = tuple(
        benchmark.step(step).observer for step in range(1, completed_step + 1)
    )
    evaluation_batches = tuple(
        benchmark.step(step).evaluation for step in range(1, completed_step + 1)
    )
    model_batches = tuple(
        benchmark.step(step).model for step in range(1, completed_step + 1)
    )
    start = perf_counter()
    try:
        from tqdm.auto import tqdm
    except ImportError as error:  # pragma: no cover - environment dependency
        raise RuntimeError("tqdm is required by the vision environment") from error
    progress = tqdm(
        range(completed_step + 1, config.benchmark.macro_steps + 1),
        initial=completed_step,
        total=config.benchmark.macro_steps,
        desc=f"dense online seed {seed}",
        disable=not config.runtime.progress,
        unit="step",
    )
    for macro_step in progress:
        batches = benchmark.step(macro_step)
        frontier = load_frontier(config, run_root, seed, macro_step)
        current = build_dense_observations(
            frontier,
            batches.observer,
            base,
            config.observer.maximum_levels,
            config.router.target_temperature,
            device,
            config.observer.inference_batch_size,
        )
        base_current = build_base_observations(
            batches.observer,
            base,
            config.observer.maximum_levels,
            device,
            config.observer.inference_batch_size,
        )
        historical_archive = None if not observer_batches else concatenate_batches(observer_batches)
        training_rows = list(
            _train_routers(
                config,
                seed,
                macro_step,
                frontier,
                current,
                historical_archive,
                routers,
                base,
                device,
            )
        )
        training_rows.extend(
            _train_integrators(
                config,
                seed,
                macro_step,
                frontier,
                current,
                base_current,
                batches.observer,
                historical_archive,
                integrators,
                base,
                device,
            )
        )
        observer_batches = (*observer_batches, batches.observer)
        evaluation_batches = (*evaluation_batches, batches.evaluation)
        model_batches = (*model_batches, batches.model)
        evaluation_rows = _evaluate_step(
            config,
            directory,
            seed,
            macro_step,
            benchmark,
            frontier,
            base,
            routers,
            integrators,
            concatenate_batches(observer_batches),
            concatenate_batches(evaluation_batches),
            concatenate_batches(model_batches),
            device,
        )
        accounting = {
            "active_node_count": len(frontier.nodes),
            "current_only_optimizer_updates_per_step": next(
                int(row["optimizer_updates_this_step"])
                for row in training_rows
                if row["condition"] == "integrator_current_only"
            ),
            "macro_step": macro_step,
            "replay_optimizer_updates_per_step": next(
                int(row["optimizer_updates_this_step"])
                for row in training_rows
                if row["condition"] == "integrator_uniform_replay"
            ),
            "row_type": "accounting",
            "run_seed": seed,
            "temporal_ranges": [
                [node.first_block + 1, node.last_block + 1] for node in frontier.nodes
            ],
        }
        ledger.append_many((*training_rows, *evaluation_rows, accounting))
        _save_online_checkpoint(
            checkpoint_path,
            config,
            seed,
            macro_step,
            routers,
            integrators,
            ledger.next_sequence,
        )
    summary = _online_summary(config, seed, ledger, perf_counter() - start, routers, integrators)
    publish_immutable_json(summary_path, summary)
    return OnlineSeedResult(directory, summary)


def _train_routers(
    config: VampLogTDenseConfig,
    seed: int,
    macro_step: int,
    frontier: DenseFrontier,
    current: DenseObservations,
    historical_archive: ExampleBatch | None,
    routers: Mapping[str, RouterConditionState],
    base: DenseMlpState,
    device: torch.device,
) -> tuple[dict[str, object], ...]:
    rows = []
    for condition, state in routers.items():
        historical = None
        duplicate_draws = 0
        range_counts: tuple[tuple[int, int, int], ...] = ()
        if condition != "router_current_hard" and historical_archive is not None:
            sampler = sample_range_balanced if "range" in condition else sample_example_balanced
            arguments = (
                (historical_archive, frontier.nodes, config.online.historical_budget)
                if sampler is sample_range_balanced
                else (historical_archive, config.online.historical_budget)
            )
            draw = sampler(
                *arguments,
                named_seed(seed, condition, macro_step, "replay"),
                macro_step,
            )
            historical = build_dense_observations(
                frontier,
                draw.batch,
                base,
                config.observer.maximum_levels,
                config.router.target_temperature,
                device,
                config.observer.inference_batch_size,
            ).router
            duplicate_draws = draw.duplicate_draws
            range_counts = draw.range_draw_counts
        before_steps = state.optimizer_steps
        result = train_router(
            state,
            current.router,
            historical,
            "soft" if condition.endswith("soft") else "hard",
            config.router.epochs_per_step,
            config.router,
            seed,
            macro_step,
            device,
        )
        rows.append(
            {
                "active_node_count": len(frontier.nodes),
                "condition": condition,
                "duplicate_replay_draws": duplicate_draws,
                "historical_examples": 0 if historical is None else len(historical.features),
                "macro_step": macro_step,
                "mean_first_epoch_loss": result.mean_first_epoch_loss,
                "mean_last_epoch_loss": result.mean_last_epoch_loss,
                "optimizer_updates_this_step": result.optimizer_steps - before_steps,
                "optimizer_updates_total": result.optimizer_steps,
                "range_draw_counts": [list(value) for value in range_counts],
                "row_type": "router_training",
                "run_seed": seed,
            }
        )
    return tuple(rows)


def _train_integrators(
    config: VampLogTDenseConfig,
    seed: int,
    macro_step: int,
    frontier: DenseFrontier,
    current: DenseObservations,
    base_current: IntegratorObservations,
    current_examples: ExampleBatch,
    historical_archive: ExampleBatch | None,
    integrators: Mapping[str, IntegratorConditionState],
    base: DenseMlpState,
    device: torch.device,
) -> tuple[dict[str, object], ...]:
    uniform_draw = (
        None
        if historical_archive is None
        else sample_example_balanced(
            historical_archive,
            config.online.historical_budget,
            named_seed(seed, "integrator-uniform", macro_step),
            macro_step,
        )
    )
    range_draw = (
        None
        if historical_archive is None
        else sample_range_balanced(
            historical_archive,
            frontier.nodes,
            config.online.historical_budget,
            named_seed(seed, "integrator-range", macro_step),
            macro_step,
        )
    )
    node_uniform = (
        None
        if uniform_draw is None
        else build_dense_observations(
            frontier,
            uniform_draw.batch,
            base,
            config.observer.maximum_levels,
            config.router.target_temperature,
            device,
            config.observer.inference_batch_size,
        ).integrator
    )
    base_uniform = (
        None
        if uniform_draw is None
        else build_base_observations(
            uniform_draw.batch,
            base,
            config.observer.maximum_levels,
            device,
            config.observer.inference_batch_size,
        )
    )
    node_range = (
        None
        if range_draw is None
        else build_dense_observations(
            frontier,
            range_draw.batch,
            base,
            config.observer.maximum_levels,
            config.router.target_temperature,
            device,
            config.observer.inference_batch_size,
        ).integrator
    )
    rows = []
    for condition, state in integrators.items():
        current_observations = (
            base_current if condition == "integrator_base_uniform_replay" else current.integrator
        )
        historical_observations, draw = {
            "integrator_current_only": (None, None),
            "integrator_uniform_replay": (node_uniform, uniform_draw),
            "integrator_range_replay": (node_range, range_draw),
            "integrator_base_uniform_replay": (base_uniform, uniform_draw),
        }[condition]
        current_supervision = IntegratorSupervision(
            current_observations, current_examples.labels
        )
        historical_supervision = (
            None
            if historical_observations is None or draw is None
            else IntegratorSupervision(historical_observations, draw.batch.labels)
        )
        before_steps = state.optimizer_steps
        result = train_integrator(
            state,
            current_supervision,
            historical_supervision,
            config.integrator.epochs_per_step,
            config.integrator,
            seed,
            macro_step,
            device,
        )
        rows.append(
            {
                "active_node_count": len(frontier.nodes),
                "baseline_current_accuracy": result.baseline_current_accuracy,
                "condition": condition,
                "current_accuracy_after": result.current_accuracy_after,
                "current_accuracy_before": result.current_accuracy_before,
                "current_loss_after": result.current_loss_after,
                "current_loss_before": result.current_loss_before,
                "duplicate_replay_draws": 0 if draw is None else draw.duplicate_draws,
                "historical_examples": 0 if historical_supervision is None else len(historical_supervision.labels),
                "historical_loss_after": result.historical_loss_after,
                "historical_loss_before": result.historical_loss_before,
                "inactive_slots_zero": inactive_slots_are_zero(current_observations, base.embedding_dim + 11)
                and (
                    historical_observations is None
                    or inactive_slots_are_zero(historical_observations, base.embedding_dim + 11)
                ),
                "macro_step": macro_step,
                "node_parameters_unchanged": True,
                "objective_after": result.objective_after,
                "objective_before": result.objective_before,
                "optimizer_updates_this_step": result.optimizer_steps - before_steps,
                "optimizer_updates_total": result.optimizer_steps,
                "range_draw_counts": [] if draw is None else [list(value) for value in draw.range_draw_counts],
                "row_type": "integrator_training",
                "run_seed": seed,
            }
        )
    return tuple(rows)


def _evaluate_step(
    config: VampLogTDenseConfig,
    directory: Path,
    seed: int,
    macro_step: int,
    benchmark: PermutedMnistBenchmark,
    frontier: DenseFrontier,
    base: DenseMlpState,
    routers: Mapping[str, RouterConditionState],
    integrators: Mapping[str, IntegratorConditionState],
    observer_archive: ExampleBatch,
    evaluation_archive: ExampleBatch,
    model_archive: ExampleBatch,
    device: torch.device,
) -> tuple[dict[str, object], ...]:
    full_checkpoint = macro_step in config.evaluation.full_checkpoints
    joint_delta = (
        _fit_or_load_joint_reference(
            config, directory, seed, macro_step, model_archive, base, device
        )
        if full_checkpoint
        else None
    )
    offline = (
        _fit_or_load_offline_integrator(
            config,
            directory,
            seed,
            macro_step,
            frontier,
            observer_archive,
            base,
            device,
        )
        if full_checkpoint
        else None
    )
    seen_domains = tuple(
        sorted({allocation.domain_id for allocation in benchmark.allocations[:macro_step]})
    )
    scopes = [
        ("evaluation_archive", evaluation_archive),
        (
            "test_subset",
            concatenate_batches(tuple(benchmark.test_domain(domain, full=False) for domain in seen_domains)),
        ),
    ]
    if full_checkpoint:
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
        )
        base_observations = build_base_observations(
            examples,
            base,
            config.observer.maximum_levels,
            device,
            config.observer.inference_batch_size,
        )
        joint_logits = (
            None
            if joint_delta is None
            else _dense_logits(examples.images, base, joint_delta, device, config.observer.inference_batch_size)
        )
        for condition, state in routers.items():
            selections, probabilities, inactive = router_selections(
                state.router,
                observations.router,
                device,
                config.observer.inference_batch_size,
            )
            rows.extend(
                {
                    **row,
                    "row_type": "router_evaluation",
                }
                for row in routing_metric_rows(
                    condition=condition,
                    selections=selections,
                    probabilities=probabilities,
                    inactive_attempts=inactive,
                    examples=examples,
                    supervision=observations.router,
                    nodes=frontier.nodes,
                    run_seed=seed,
                    macro_step=macro_step,
                    evaluation_scope=scope,
                    near_oracle_thresholds=(0.05, 0.10, 0.25),
                    joint_logits=joint_logits,
                )
            )
        for condition in ROUTER_FIXED_CONTROLS:
            selections = fixed_policy_selections(
                condition,
                frontier.nodes,
                observations.router,
                named_seed(seed, "router-control", condition, macro_step, scope),
            )
            rows.extend(
                {
                    **row,
                    "row_type": "router_evaluation",
                }
                for row in routing_metric_rows(
                    condition=condition,
                    selections=selections,
                    probabilities=None,
                    inactive_attempts=None,
                    examples=examples,
                    supervision=observations.router,
                    nodes=frontier.nodes,
                    run_seed=seed,
                    macro_step=macro_step,
                    evaluation_scope=scope,
                    near_oracle_thresholds=(0.05, 0.10, 0.25),
                    joint_logits=joint_logits,
                )
            )
        for condition, state in integrators.items():
            condition_observations = (
                base_observations
                if condition == "integrator_base_uniform_replay"
                else observations.integrator
            )
            logits = prediction_logits(
                state.integrator,
                condition_observations,
                device,
                config.observer.inference_batch_size,
            )
            rows.extend(
                {
                    **row,
                    "row_type": "integrator_evaluation",
                }
                for row in prediction_metric_rows(
                    condition=condition,
                    logits=logits,
                    examples=examples,
                    node_observations=observations.integrator,
                    nodes=frontier.nodes,
                    run_seed=seed,
                    macro_step=macro_step,
                    evaluation_scope=scope,
                    joint_logits=joint_logits,
                )
            )
        for condition in FIXED_CONTROLS:
            logits = fixed_control_logits(
                condition,
                frontier.nodes,
                observations.integrator,
                examples.labels,
                named_seed(seed, "integrator-control", condition, macro_step, scope),
            )
            rows.extend(
                {
                    **row,
                    "row_type": "integrator_evaluation",
                }
                for row in prediction_metric_rows(
                    condition=condition,
                    logits=logits,
                    examples=examples,
                    node_observations=observations.integrator,
                    nodes=frontier.nodes,
                    run_seed=seed,
                    macro_step=macro_step,
                    evaluation_scope=scope,
                    joint_logits=joint_logits,
                )
            )
        if joint_logits is not None:
            rows.extend(
                {
                    **row,
                    "row_type": "integrator_evaluation",
                }
                for row in prediction_metric_rows(
                    condition="pooled_single_mlp_reference",
                    logits=joint_logits,
                    examples=examples,
                    node_observations=observations.integrator,
                    nodes=frontier.nodes,
                    run_seed=seed,
                    macro_step=macro_step,
                    evaluation_scope=scope,
                    joint_logits=joint_logits,
                )
            )
        if offline is not None:
            rows.extend(
                {
                    **row,
                    "row_type": "integrator_evaluation",
                }
                for row in prediction_metric_rows(
                    condition="fresh_cumulative_four_epoch_integrator",
                    logits=prediction_logits(
                        offline.integrator,
                        observations.integrator,
                        device,
                        config.observer.inference_batch_size,
                    ),
                    examples=examples,
                    node_observations=observations.integrator,
                    nodes=frontier.nodes,
                    run_seed=seed,
                    macro_step=macro_step,
                    evaluation_scope=scope,
                    joint_logits=joint_logits,
                )
            )
    return tuple(rows)


def _fit_or_load_joint_reference(
    config: VampLogTDenseConfig,
    directory: Path,
    seed: int,
    macro_step: int,
    model_archive: ExampleBatch,
    base: DenseMlpState,
    device: torch.device,
) -> DenseMlpState:
    path = directory / "references" / f"step-{macro_step:03d}" / "pooled-model-delta.pt"
    if path.is_file():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("config_hash") != config.config_hash or int(payload.get("macro_step", -1)) != macro_step:
            raise ValueError("stored pooled MLP reference coordinates changed")
        return DenseMlpState(tuple(payload["delta_parameters"]))
    result = fit_dense_model(
        DenseExamples(
            model_archive.images,
            model_archive.labels,
            (torch.arange(784, dtype=torch.int64),),
        ),
        base,
        config.node.optimizer,
        named_seed(seed, "pooled-reference", macro_step),
        device,
        fixed_epochs=config.node.epochs,
        dropout=config.node.dropout,
        progress_label=f"pooled MLP reference seed {seed} step {macro_step}",
        progress=config.runtime.progress,
    )
    delta = dense_delta(base, result.state)
    atomic_torch_save(
        path,
        {
            "config_hash": config.config_hash,
            "delta_parameters": delta.tensors,
            "example_updates": result.training_example_presentations,
            "macro_step": macro_step,
            "schema_version": "vamp-logt-dense-pooled-reference-v1",
            "seed": seed,
        },
    )
    return delta


def _fit_or_load_offline_integrator(
    config: VampLogTDenseConfig,
    directory: Path,
    seed: int,
    macro_step: int,
    frontier: DenseFrontier,
    observer_archive: ExampleBatch,
    base: DenseMlpState,
    device: torch.device,
) -> IntegratorConditionState:
    path = directory / "references" / f"step-{macro_step:03d}" / "four-epoch-integrator.pt"
    slot_dim = base.embedding_dim + 11
    input_dim = config.observer.maximum_levels * slot_dim
    name = f"fresh-cumulative-four-epoch-step-{macro_step}"
    state = create_integrator_state(name, input_dim, slot_dim, config.integrator, seed, device)
    if path.is_file():
        payload = torch.load(path, map_location=device, weights_only=True)
        if payload.get("config_hash") != config.config_hash or int(payload.get("macro_step", -1)) != macro_step:
            raise ValueError("stored fresh cumulative integrator coordinates changed")
        state.integrator.load_state_dict(payload["model"], strict=True)
        state.optimizer_steps = int(payload["optimizer_steps"])
        return state
    observations = build_dense_observations(
        frontier,
        observer_archive,
        base,
        config.observer.maximum_levels,
        config.router.target_temperature,
        device,
        config.observer.inference_batch_size,
    ).integrator
    train_integrator(
        state,
        IntegratorSupervision(observations, observer_archive.labels),
        None,
        config.integrator.offline_epochs,
        config.integrator,
        seed,
        macro_step,
        device,
    )
    atomic_torch_save(
        path,
        {
            "config_hash": config.config_hash,
            "macro_step": macro_step,
            "model": _cpu_state_dict(state.integrator.state_dict()),
            "optimizer_steps": state.optimizer_steps,
            "schema_version": "vamp-logt-dense-four-epoch-integrator-v1",
            "seed": seed,
        },
    )
    return state


def _dense_logits(
    images: Tensor,
    base: DenseMlpState,
    delta: DenseMlpState,
    device: torch.device,
    batch_size: int,
) -> Tensor:
    target_base = DenseMlpState(tuple(tensor.to(device) for tensor in base.tensors))
    target_delta = DenseMlpState(tuple(tensor.to(device) for tensor in delta.tensors))
    rows = []
    with torch.inference_mode():
        for offset in range(0, len(images), batch_size):
            rows.append(
                dense_hidden_logits(
                    images[offset : offset + batch_size].flatten(1).to(device),
                    target_base,
                    target_delta,
                )[1].cpu()
            )
    return torch.cat(rows)


def _save_online_checkpoint(
    path: Path,
    config: VampLogTDenseConfig,
    seed: int,
    macro_step: int,
    routers: Mapping[str, RouterConditionState],
    integrators: Mapping[str, IntegratorConditionState],
    metric_rows: int,
) -> None:
    atomic_torch_save(
        path,
        {
            "config_hash": config.config_hash,
            "integrators": {
                name: {
                    "model": _cpu_state_dict(state.integrator.state_dict()),
                    "optimizer": state.optimizer.state_dict(),
                    "optimizer_steps": state.optimizer_steps,
                }
                for name, state in integrators.items()
            },
            "macro_step": macro_step,
            "metric_rows": metric_rows,
            "routers": {
                name: {
                    "model": _cpu_state_dict(state.router.state_dict()),
                    "optimizer": state.optimizer.state_dict(),
                    "optimizer_steps": state.optimizer_steps,
                }
                for name, state in routers.items()
            },
            "run_seed": seed,
            "schema_version": "vamp-logt-dense-online-checkpoint-v1",
        },
    )


def _load_online_checkpoint(
    path: Path,
    config: VampLogTDenseConfig,
    seed: int,
    routers: Mapping[str, RouterConditionState],
    integrators: Mapping[str, IntegratorConditionState],
    device: torch.device,
) -> tuple[int, int]:
    if not path.is_file():
        return 0, 0
    payload = torch.load(path, map_location=device, weights_only=False)
    if (
        payload.get("schema_version") != "vamp-logt-dense-online-checkpoint-v1"
        or payload.get("config_hash") != config.config_hash
        or int(payload.get("run_seed", -1)) != seed
        or set(payload.get("routers", {})) != set(routers)
        or set(payload.get("integrators", {})) != set(integrators)
    ):
        raise ValueError("dense online checkpoint coordinates changed")
    for collection, states, model_attribute in (
        (payload["routers"], routers, "router"),
        (payload["integrators"], integrators, "integrator"),
    ):
        for name, state in states.items():
            record = collection[name]
            getattr(state, model_attribute).load_state_dict(record["model"], strict=True)
            state.optimizer.load_state_dict(record["optimizer"])
            _optimizer_to_device(state.optimizer, device)
            state.optimizer_steps = int(record["optimizer_steps"])
    return int(payload["macro_step"]), int(payload["metric_rows"])


def _online_summary(
    config: VampLogTDenseConfig,
    seed: int,
    ledger: ChainedJsonlLedger,
    wall_seconds: float,
    routers: Mapping[str, RouterConditionState],
    integrators: Mapping[str, IntegratorConditionState],
) -> dict[str, object]:
    rows = ledger.rows
    training = tuple(row for row in rows if str(row.get("row_type", "")).endswith("training"))
    evaluations = tuple(row for row in rows if str(row.get("row_type", "")).endswith("evaluation"))
    replay = tuple(row for row in training if int(row["macro_step"]) > 1 and row["condition"] not in {
        "router_current_hard", "integrator_current_only",
    })
    acceptance = {
        "all_metrics_finite": all(
            torch.isfinite(torch.tensor(float(row.get("mean_cross_entropy", row.get("mean_regret", 0.0)))))
            for row in evaluations
        ),
        "exact_replay_budget": all(int(row["historical_examples"]) == config.online.historical_budget for row in replay),
        "inactive_slots_zero": all(bool(row.get("inactive_slots_zero", True)) for row in training),
        "node_parameters_unchanged": all(bool(row.get("node_parameters_unchanged", True)) for row in training),
        "training_loss_decreased": all(
            float(row.get("objective_after", row.get("mean_last_epoch_loss", 0.0)))
            <= float(row.get("objective_before", row.get("mean_first_epoch_loss", 0.0))) + 1.0e-6
            for row in training
        ),
    }
    return {
        "acceptance": acceptance,
        "config_hash": config.config_hash,
        "final_macro_step": config.benchmark.macro_steps,
        "integrator_optimizer_steps": {name: state.optimizer_steps for name, state in integrators.items()},
        "metric_rows": len(rows),
        "router_optimizer_steps": {name: state.optimizer_steps for name, state in routers.items()},
        "run_seed": seed,
        "schema_version": "vamp-logt-dense-online-seed-v1",
        "status": "complete",
        "wall_seconds": wall_seconds,
    }


def _cpu_state_dict(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in state.items()}


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for name, value in tuple(state.items()):
            if isinstance(value, Tensor):
                state[name] = value.to(device)


__all__ = ["OnlineSeedResult", "ROUTER_FIXED_CONTROLS", "run_online", "run_online_seed"]

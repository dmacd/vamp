"""Resumable direct-prediction integration on VAMP-AF Rotated-MNIST."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import Tensor

from apm.continual.artifacts import (
    ChainedJsonlLedger,
    atomic_write,
    canonical_json_bytes,
    file_sha256,
    load_canonical_json,
    publish_immutable_json,
)
from apm.continual.logt_behavioral_integrator import (
    IntegratorConditionState,
    IntegratorObservations,
    IntegratorSupervision,
    LevelSlotIntegrator,
    create_condition_state,
    inactive_slots_are_zero,
    prediction_logits,
    train_condition,
)
from apm.continual.logt_behavioral_router import (
    ReplayDraw,
    sample_example_balanced,
    sample_range_balanced,
)
from apm.continual.top_two_adapter import TopTwoAdapterState
from apm.continual.vision.imagenetr.checkpoints import atomic_torch_save
from apm.experiments.vamp_logt_integrator_metrics import (
    FIXED_CONTROLS,
    fixed_control_logits,
    prediction_metric_rows,
)
from apm.experiments.vamp_logt_integrator_features import (
    frozen_integrator_trunk_features as _trunk_features,
    integrator_supervision as _supervision,
    integrator_supervision_from_trunk as _supervision_from_trunk,
)
from apm.experiments.vamp_logt_integrator_rotated_config import (
    IntegratorPhaseConfig,
    VampLogTIntegratorConfig,
)
from apm.experiments.vamp_logt_integrator_rotated_reporting import (
    write_phase_report,
    write_results,
)
from apm.experiments.vamp_logt_router_data import (
    ExampleBatch,
    FrozenClassifierDependency,
    concatenate_batches,
    load_frozen_classifier,
    named_seed,
    resolved_device,
)
from apm.experiments.vamp_logt_router_rotated_data import (
    RotatedMnistBenchmark,
    build_benchmark,
)
from apm.experiments.vamp_logt_router_state import (
    ActiveAdapterBank,
    advance_adapter_bank,
    bank_from_record,
    bank_record,
    empty_adapter_bank,
    retire_inactive_nodes,
)
from apm.experiments.vamp_logt_router_workflow import (
    SeedResult,
    adapter_logits,
)


@dataclass(frozen=True, slots=True)
class IntegratorWorkCounters:
    """Exact feature, external-reference, and evaluation work."""

    node_current_evals: int = 0
    node_historical_evals: int = 0
    base_current_evals: int = 0
    base_historical_evals: int = 0
    evaluation_node_evals: int = 0
    offline_node_evals: int = 0
    offline_example_updates: int = 0
    parent_joint_reference_example_updates: int = 0

    def __post_init__(self) -> None:
        if any(value < 0 for value in asdict(self).values()):
            raise ValueError("integrator work counters must be nonnegative")

    def advanced(
        self,
        *,
        active_nodes: int,
        current_examples: int,
        full_node_historical_examples: int,
        base_historical_examples: int,
        evaluation_examples: int,
        offline_node_evals: int,
        offline_example_updates: int,
        parent_joint_reference_updates: int,
    ) -> "IntegratorWorkCounters":
        """Return counters after one complete durable macro-step."""
        return IntegratorWorkCounters(
            self.node_current_evals + active_nodes * current_examples,
            self.node_historical_evals
            + active_nodes * full_node_historical_examples,
            self.base_current_evals + current_examples,
            self.base_historical_evals + base_historical_examples,
            self.evaluation_node_evals + active_nodes * evaluation_examples,
            self.offline_node_evals + offline_node_evals,
            self.offline_example_updates + offline_example_updates,
            self.parent_joint_reference_example_updates
            + parent_joint_reference_updates,
        )


def run_workflow(
    config: VampLogTIntegratorConfig,
    selected_phase: str = "all",
) -> Path:
    """Run or resume the selected sealed integrator phases."""
    if selected_phase not in {"smoke", "primary", "all"}:
        raise ValueError("phase must be smoke, primary, or all")
    device = resolved_device(config.runtime.device)
    if config.runtime.deterministic_algorithms:
        torch.use_deterministic_algorithms(True)
    run_root = config.artifact_root / "runs" / config.config_hash
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"Durable Rotated-MNIST prediction-integrator run: {run_root}", flush=True)
    dependency = load_frozen_classifier(config)
    parent_summary = _authenticate_parent_router(config)
    _write_protocol(run_root, config, dependency, parent_summary)
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
            f"{phase.macro_steps} macro-steps",
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
                "schema_version": "vamp-logt-integrator-latest-v1",
            }
        ),
    )
    return run_root


def run_phase_seed(
    config: VampLogTIntegratorConfig,
    phase_name: str,
    phase: IntegratorPhaseConfig,
    seed: int,
    dependency: FrozenClassifierDependency,
    run_root: Path,
    device: torch.device,
) -> SeedResult:
    """Run or resume one hierarchy and its independent integrators."""
    directory = run_root / phase_name / f"seed-{seed}"
    checkpoint_path = directory / "state" / "checkpoint.pt"
    nodes_root = directory / "nodes"
    ledger = ChainedJsonlLedger(
        directory / "metrics.jsonl", "vamp-logt-integrator-metric-v1"
    )
    benchmark = build_benchmark(config, phase_name, seed)
    slot_dim = dependency.base.hidden_dim + 10 + 1
    input_dim = config.integrator.maximum_levels * slot_dim
    conditions = {
        name: create_condition_state(
            name, input_dim, slot_dim, config.integrator, seed, device
        )
        for name in phase.conditions
    }
    initial_invariants = _initial_invariants(conditions, slot_dim)
    bank, work, completed_step, checkpoint_rows = _load_checkpoint(
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
    integrator_batches = tuple(
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
        desc=f"integrator {phase_name} seed {seed} overall",
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
            config, bank, model_archive, dependency, seed, nodes_root, device
        )
        hierarchy_seconds = perf_counter() - hierarchy_start
        current_trunk = _trunk_features(
            config, dependency, batches.router.images, device
        )
        node_current, base_current = (
            _supervision_from_trunk(
                config,
                dependency,
                bank,
                batches.router,
                current_trunk,
                device,
                base_only=base_only,
            )
            for base_only in (False, True)
        )
        integrator_archive = (
            None
            if not integrator_batches
            else concatenate_batches(integrator_batches)
        )
        example_draw = (
            None
            if integrator_archive is None
            else sample_example_balanced(
                integrator_archive,
                phase.historical_budget,
                named_seed(seed, "integrator-example-replay", macro_step),
                macro_step,
            )
        )
        range_draw = (
            None
            if integrator_archive is None
            else sample_range_balanced(
                integrator_archive,
                bank.topology.active_nodes,
                phase.historical_budget,
                named_seed(seed, "integrator-range-replay", macro_step),
                macro_step,
            )
        )
        if example_draw is None:
            node_example, base_example = None, None
        else:
            example_trunk = _trunk_features(
                config, dependency, example_draw.batch.images, device
            )
            node_example, base_example = (
                _supervision_from_trunk(
                    config,
                    dependency,
                    bank,
                    example_draw.batch,
                    example_trunk,
                    device,
                    base_only=base_only,
                )
                for base_only in (False, True)
            )
        node_range = (
            None
            if range_draw is None
            else _supervision(
                config, dependency, bank, range_draw.batch, device, base_only=False
            )
        )
        before_nodes = {
            node_id: tuple(tensor.clone() for tensor in bank.adapters[node_id].tensors)
            for node_id in sorted(bank.adapters)
        }
        integration_start = perf_counter()
        training_rows = []
        historical_counts: dict[str, int] = {}
        for condition_name, state in conditions.items():
            current = base_current if condition_name == "base_example_replay" else node_current
            historical, draw = _historical_condition(
                condition_name,
                node_example,
                node_range,
                base_example,
                example_draw,
                range_draw,
            )
            if historical is not None:
                historical_counts[condition_name] = len(historical.labels)
                if len(historical.labels) != phase.historical_budget:
                    raise RuntimeError("integrator replay did not receive its exact budget")
            result = train_condition(
                state,
                current,
                historical,
                phase.integrator_epochs_per_step,
                config.integrator,
                seed,
                macro_step,
                device,
            )
            training_rows.append(
                {
                    "active_node_count": len(bank.topology.active_nodes),
                    "baseline_current_accuracy": result.baseline_current_accuracy,
                    "carry_accuracy_change_from_baseline": (
                        result.current_accuracy_after
                        - result.baseline_current_accuracy
                    ),
                    "condition": condition_name,
                    "current_accuracy_after": result.current_accuracy_after,
                    "current_accuracy_before": result.current_accuracy_before,
                    "current_loss_after": result.current_loss_after,
                    "current_loss_before": result.current_loss_before,
                    "duplicate_replay_draws": 0 if draw is None else draw.duplicate_draws,
                    "historical_examples": historical_counts.get(condition_name, 0),
                    "historical_loss_after": result.historical_loss_after,
                    "historical_loss_before": result.historical_loss_before,
                    "inactive_slots_zero": inactive_slots_are_zero(
                        current.observations, slot_dim
                    )
                    and (
                        historical is None
                        or inactive_slots_are_zero(historical.observations, slot_dim)
                    ),
                    "is_carry": macro_step % 2 == 0,
                    "macro_step": macro_step,
                    "node_parameters_unchanged": True,
                    "objective_after": result.objective_after,
                    "objective_before": result.objective_before,
                    "optimizer_steps": result.optimizer_steps,
                    "range_draw_counts": (
                        [] if draw is None else [list(value) for value in draw.range_draw_counts]
                    ),
                    "row_type": "training",
                    "run_seed": seed,
                }
            )
        integration_seconds = perf_counter() - integration_start
        nodes_unchanged = not before_nodes or all(
            all(
                torch.equal(before, after)
                for before, after in zip(
                    before_nodes[node_id],
                    bank.adapters[node_id].tensors,
                    strict=True,
                )
            )
            for node_id in sorted(before_nodes)
        )
        if not nodes_unchanged:
            raise RuntimeError("integrator optimization modified frozen node tensors")
        training_rows = [
            {**row, "node_parameters_unchanged": nodes_unchanged}
            for row in training_rows
        ]
        integrator_batches = (*integrator_batches, batches.router)
        evaluation_batches = (*evaluation_batches, batches.evaluation)
        evaluation_start = perf_counter()
        (
            evaluation_rows,
            evaluation_examples,
            offline_node_evals,
            offline_example_updates,
            parent_joint_reference_updates,
        ) = _evaluate_step(
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
            concatenate_batches(integrator_batches),
            directory,
            device,
        )
        evaluation_seconds = perf_counter() - evaluation_start
        full_history = sum(
            historical_counts.get(name, 0)
            for name in ("integrator_example_replay", "integrator_range_replay")
        )
        work = work.advanced(
            active_nodes=len(bank.topology.active_nodes),
            current_examples=len(batches.router.labels),
            full_node_historical_examples=full_history,
            base_historical_examples=historical_counts.get("base_example_replay", 0),
            evaluation_examples=evaluation_examples,
            offline_node_evals=offline_node_evals,
            offline_example_updates=offline_example_updates,
            parent_joint_reference_updates=parent_joint_reference_updates,
        )
        accounting_row = {
            "active_node_count": len(bank.topology.active_nodes),
            "adapter_example_updates": bank.adapter_example_updates,
            "evaluation_seconds": evaluation_seconds,
            "hierarchy_seconds": hierarchy_seconds,
            "integration_seconds": integration_seconds,
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
            conditions,
            work,
            ledger.next_sequence,
        )
        retire_inactive_nodes(
            nodes_root, {node.node_id for node in bank.topology.active_nodes}
        )
        progress.set_postfix(nodes=len(bank.topology.active_nodes), rows=ledger.next_sequence)
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
        initial_invariants,
    )
    return SeedResult(phase_name, seed, directory, summary)


def _historical_condition(
    condition: str,
    node_example: IntegratorSupervision | None,
    node_range: IntegratorSupervision | None,
    base_example: IntegratorSupervision | None,
    example_draw: ReplayDraw | None,
    range_draw: ReplayDraw | None,
) -> tuple[IntegratorSupervision | None, ReplayDraw | None]:
    choices = {
        "integrator_no_replay": (None, None),
        "integrator_example_replay": (node_example, example_draw),
        "integrator_range_replay": (node_range, range_draw),
        "base_example_replay": (base_example, example_draw),
    }
    if condition not in choices:
        raise ValueError(f"unknown integrator condition: {condition}")
    return choices[condition]


def _evaluate_step(
    config: VampLogTIntegratorConfig,
    phase_name: str,
    phase: IntegratorPhaseConfig,
    seed: int,
    macro_step: int,
    benchmark: RotatedMnistBenchmark,
    dependency: FrozenClassifierDependency,
    bank: ActiveAdapterBank,
    conditions: Mapping[str, IntegratorConditionState],
    evaluation_archive: ExampleBatch,
    current_evaluation: ExampleBatch,
    integrator_archive: ExampleBatch,
    directory: Path,
    device: torch.device,
) -> tuple[tuple[dict[str, object], ...], int, int, int, int]:
    observed_domains = tuple(
        sorted({allocation.domain_id for allocation in benchmark.allocations[:macro_step]})
    )
    test_subset = concatenate_batches(
        tuple(benchmark.test_domain(domain, full=False) for domain in observed_domains)
    )
    datasets: list[tuple[str, ExampleBatch, TopTwoAdapterState | None]] = [
        ("current_eval", current_evaluation, None),
        ("evaluation_archive", evaluation_archive, None),
        ("test_subset", test_subset, None),
    ]
    joint_adapter = None
    parent_joint_reference_updates = 0
    offline_integrator = None
    offline_node_evals = 0
    offline_example_updates = 0
    if phase_name == "primary" and macro_step in config.evaluation.full_checkpoints:
        full_test = concatenate_batches(
            tuple(benchmark.test_domain(domain, full=True) for domain in observed_domains)
        )
        joint_adapter, parent_joint_reference_updates = _load_parent_joint_reference(
            config,
            seed,
            macro_step,
        )
        (
            offline_integrator,
            offline_node_evals,
            offline_example_updates,
        ) = _fit_or_load_offline_integrator(
            config,
            dependency,
            bank,
            integrator_archive,
            seed,
            macro_step,
            directory / "offline_integrator",
            device,
        )
        datasets.append(("full_test", full_test, joint_adapter))
    result_rows = []
    total_examples = 0
    for scope, examples, reference in datasets:
        trunk = _trunk_features(config, dependency, examples.images, device)
        node_supervision, base_supervision = (
            _supervision_from_trunk(
                config,
                dependency,
                bank,
                examples,
                trunk,
                device,
                base_only=base_only,
            )
            for base_only in (False, True)
        )
        joint_logits = (
            None
            if reference is None
            else adapter_logits(
                reference,
                dependency,
                trunk,
                device,
                config.evaluation.inference_batch_size,
            )
        )
        for control in FIXED_CONTROLS:
            logits = fixed_control_logits(
                control,
                bank.topology.active_nodes,
                node_supervision.observations,
                examples.labels,
                named_seed(seed, macro_step, scope, control),
            )
            result_rows.extend(
                prediction_metric_rows(
                    condition=control,
                    logits=logits,
                    examples=examples,
                    node_observations=node_supervision.observations,
                    nodes=bank.topology.active_nodes,
                    run_seed=seed,
                    macro_step=macro_step,
                    evaluation_scope=scope,
                    joint_logits=joint_logits,
                )
            )
        for condition_name, state in conditions.items():
            observations = (
                base_supervision.observations
                if condition_name == "base_example_replay"
                else node_supervision.observations
            )
            logits = prediction_logits(
                state.integrator,
                observations,
                device,
                config.evaluation.inference_batch_size,
            )
            result_rows.extend(
                prediction_metric_rows(
                    condition=condition_name,
                    logits=logits,
                    examples=examples,
                    node_observations=node_supervision.observations,
                    nodes=bank.topology.active_nodes,
                    run_seed=seed,
                    macro_step=macro_step,
                    evaluation_scope=scope,
                    joint_logits=joint_logits,
                )
            )
        if offline_integrator is not None:
            logits = prediction_logits(
                offline_integrator,
                node_supervision.observations,
                device,
                config.evaluation.inference_batch_size,
            )
            result_rows.extend(
                prediction_metric_rows(
                    condition="offline_cumulative_integrator",
                    logits=logits,
                    examples=examples,
                    node_observations=node_supervision.observations,
                    nodes=bank.topology.active_nodes,
                    run_seed=seed,
                    macro_step=macro_step,
                    evaluation_scope=scope,
                    joint_logits=joint_logits,
                )
            )
        total_examples += len(examples.labels)
    return (
        tuple({**row, "row_type": "evaluation"} for row in result_rows),
        total_examples,
        offline_node_evals,
        offline_example_updates,
        parent_joint_reference_updates,
    )


def _fit_or_load_offline_integrator(
    config: VampLogTIntegratorConfig,
    dependency: FrozenClassifierDependency,
    bank: ActiveAdapterBank,
    archive: ExampleBatch,
    seed: int,
    macro_step: int,
    root: Path,
    device: torch.device,
) -> tuple[LevelSlotIntegrator, int, int]:
    path = root / f"step-{macro_step}.pt"
    slot_dim = dependency.base.hidden_dim + 10 + 1
    input_dim = config.integrator.maximum_levels * slot_dim
    reference_seed = named_seed(seed, "offline-cumulative-integrator", macro_step)
    state = create_condition_state(
        f"offline_cumulative_integrator_{macro_step}",
        input_dim,
        slot_dim,
        config.integrator,
        reference_seed,
        device,
    )
    if path.is_file():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (
            payload.get("schema_version") != "vamp-logt-offline-integrator-v1"
            or payload.get("macro_step") != macro_step
            or payload.get("seed") != reference_seed
        ):
            raise ValueError("offline integrator coordinates changed")
        state.integrator.load_state_dict(payload["parameters"], strict=True)
        return (
            state.integrator,
            len(bank.topology.active_nodes) * len(archive.labels),
            int(payload["example_updates"]),
        )
    supervision = _supervision(
        config, dependency, bank, archive, device, base_only=False
    )
    train_condition(
        state,
        supervision,
        None,
        config.integrator.offline_epochs,
        config.integrator,
        reference_seed,
        macro_step,
        device,
    )
    example_updates = len(archive.labels) * config.integrator.offline_epochs
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(
        path,
        {
            "example_updates": example_updates,
            "macro_step": macro_step,
            "parameters": state.integrator.state_dict(),
            "schema_version": "vamp-logt-offline-integrator-v1",
            "seed": reference_seed,
        },
    )
    return (
        state.integrator,
        len(bank.topology.active_nodes) * len(archive.labels),
        example_updates,
    )


def _load_parent_joint_reference(
    config: VampLogTIntegratorConfig,
    seed: int,
    macro_step: int,
) -> tuple[TopTwoAdapterState, int]:
    """Load the exact read-only joint-IID adapter from the sealed router run."""
    path = (
        config.parent_router_run_root
        / "primary"
        / f"seed-{seed}"
        / "joint_reference"
        / f"step-{macro_step}.pt"
    )
    if not path.is_file():
        raise ValueError("sealed parent joint-IID reference is missing")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("schema_version") != "vamp-logt-router-joint-reference-v1"
        or payload.get("macro_step") != macro_step
        or payload.get("seed") != named_seed(seed, "joint-reference", macro_step)
    ):
        raise ValueError("sealed parent joint-IID reference coordinates changed")
    return (
        TopTwoAdapterState(*payload["parameters"]),
        int(payload["example_updates"]),
    )


def _initial_invariants(
    conditions: Mapping[str, IntegratorConditionState], slot_dim: int
) -> dict[str, bool]:
    if not conditions:
        raise ValueError("initial integrator invariants require a condition")
    future_zero = all(
        bool(
            torch.equal(
                state.integrator.input_layer.weight[:, slot_dim:],
                torch.zeros_like(state.integrator.input_layer.weight[:, slot_dim:]),
            )
        )
        for state in conditions.values()
    )
    residual_zero = all(
        bool(
            torch.equal(
                state.integrator.output_layer.weight,
                torch.zeros_like(state.integrator.output_layer.weight),
            )
            and torch.equal(
                state.integrator.output_layer.bias,
                torch.zeros_like(state.integrator.output_layer.bias),
            )
        )
        for state in conditions.values()
    )
    input_dims = {state.integrator.input_dim for state in conditions.values()}
    if len(input_dims) != 1:
        raise ValueError("integrator conditions do not share one fixed input")
    features = torch.zeros((3, input_dims.pop()))
    one_node = torch.log_softmax(
        torch.arange(30, dtype=torch.float32).reshape(3, 10), dim=1
    )
    ensemble = torch.log_softmax(
        torch.arange(30, 0, -1, dtype=torch.float32).reshape(3, 10), dim=1
    )
    parity = {
        name: all(
            torch.equal(
                state.integrator(
                    features.to(next(state.integrator.parameters()).device),
                    baseline.to(next(state.integrator.parameters()).device),
                ).cpu(),
                baseline,
            )
            for state in conditions.values()
        )
        for name, baseline in (
            ("one_node_initial_parity", one_node),
            ("mean_ensemble_initial_parity", ensemble),
        )
    }
    return {
        "future_slot_columns_zero": future_zero,
        "zero_residual_output": residual_zero,
        **parity,
    }


def _save_checkpoint(
    path: Path,
    config: VampLogTIntegratorConfig,
    phase_name: str,
    seed: int,
    macro_step: int,
    bank: ActiveAdapterBank,
    conditions: Mapping[str, IntegratorConditionState],
    work: IntegratorWorkCounters,
    metric_rows: int,
) -> None:
    atomic_torch_save(
        path,
        {
            "bank": bank_record(bank),
            "completed_macro_step": macro_step,
            "conditions": {
                name: {
                    "integrator": state.integrator.state_dict(),
                    "optimizer": state.optimizer.state_dict(),
                    "optimizer_steps": state.optimizer_steps,
                }
                for name, state in conditions.items()
            },
            "config_hash": config.config_hash,
            "metric_rows": metric_rows,
            "phase": phase_name,
            "run_seed": seed,
            "schema_version": "vamp-logt-integrator-checkpoint-v1",
            "work": asdict(work),
        },
    )


def _load_checkpoint(
    path: Path,
    config: VampLogTIntegratorConfig,
    phase_name: str,
    seed: int,
    phase: IntegratorPhaseConfig,
    conditions: Mapping[str, IntegratorConditionState],
    device: torch.device,
) -> tuple[ActiveAdapterBank, IntegratorWorkCounters, int, int]:
    if not path.is_file():
        return (
            empty_adapter_bank(config.benchmark.model_batch_size),
            IntegratorWorkCounters(),
            0,
            0,
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("schema_version") != "vamp-logt-integrator-checkpoint-v1"
        or payload.get("config_hash") != config.config_hash
        or payload.get("phase") != phase_name
        or payload.get("run_seed") != seed
        or set(payload.get("conditions", {})) != set(conditions)
        or phase.conditions != tuple(conditions)
    ):
        raise ValueError("prediction-integrator checkpoint coordinates changed")
    for name, state in conditions.items():
        record = payload["conditions"][name]
        state.integrator.load_state_dict(record["integrator"], strict=True)
        state.optimizer.load_state_dict(record["optimizer"])
        _optimizer_to(state.optimizer, device)
        state.optimizer_steps = int(record["optimizer_steps"])
    return (
        bank_from_record(payload["bank"]),
        IntegratorWorkCounters(**payload["work"]),
        int(payload["completed_macro_step"]),
        int(payload["metric_rows"]),
    )


def _optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for values in optimizer.state.values():
        for key, value in values.items():
            if isinstance(value, Tensor):
                values[key] = value.to(device)


def _require_smoke_gate(
    run_root: Path, config: VampLogTIntegratorConfig
) -> None:
    path = run_root / "smoke" / f"seed-{config.smoke.seeds[0]}" / "summary.json"
    if not path.is_file():
        raise RuntimeError("the exact integrator smoke must complete before primary")
    acceptance = load_canonical_json(path)["acceptance"]
    required = (
        "all_primary_metrics_finite",
        "exact_historical_budget",
        "future_slot_columns_zero",
        "fixed_budget_training_work",
        "inactive_slots_zero",
        "mean_ensemble_initial_parity",
        "node_parameters_unchanged",
        "one_node_initial_parity",
        "zero_residual_output",
    )
    if not all(bool(acceptance[name]) for name in required) or float(
        acceptance["loss_decrease_fraction"]
    ) <= 0.5:
        raise RuntimeError("the exact integrator smoke did not pass its gates")


def _authenticate_parent_router(
    config: VampLogTIntegratorConfig,
) -> dict[str, object]:
    protocol_path = config.parent_router_run_root / "protocol.json"
    summary_path = config.parent_router_run_root / "summary.json"
    if (
        not protocol_path.is_file()
        or not summary_path.is_file()
        or file_sha256(protocol_path) != config.parent_router.protocol_sha256
        or file_sha256(summary_path) != config.parent_router.summary_sha256
    ):
        raise ValueError("completed parent router artifacts changed or are missing")
    protocol = load_canonical_json(protocol_path)
    summary = load_canonical_json(summary_path)
    if (
        protocol.get("config_hash") != config.parent_router.run_id
        or summary.get("status") != "complete"
        or summary.get("completed_primary_seeds") != 5
    ):
        raise ValueError("parent router result is not the frozen completed run")
    return summary


def _write_protocol(
    run_root: Path,
    config: VampLogTIntegratorConfig,
    dependency: FrozenClassifierDependency,
    parent_summary: Mapping[str, object],
) -> None:
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - vision environment gate
        raise RuntimeError("PyYAML is required by the vision environment") from error
    resolved = yaml.safe_dump(config.as_record(), sort_keys=False).encode("utf-8")
    config_path = run_root / "config_resolved.yaml"
    if config_path.is_file() and config_path.read_bytes() != resolved:
        raise ValueError("resolved integrator protocol changed inside one run identity")
    atomic_write(config_path, resolved)
    project_root = Path(__file__).resolve().parents[3]
    material_paths = (
        "configs/vamp_af_mnist/poc.yaml",
        "configs/vamp_logt_integrator_rotated_mnist/primary.yaml",
        "docs/logt_vamp_rotated_mnist_integrator_plan.md",
        "src/apm/continual/artifacts.py",
        "src/apm/continual/logt_behavioral_integrator.py",
        "src/apm/continual/logt_behavioral_router.py",
        "src/apm/continual/logt_evidence_bank.py",
        "src/apm/continual/top_two_adapter.py",
        "src/apm/continual/vision/imagenetr/checkpoints.py",
        "src/apm/data/mnist/loader.py",
        "src/apm/experiments/vamp_af_config.py",
        "src/apm/experiments/vamp_af_data.py",
        "src/apm/experiments/vamp_logt_evidence_training.py",
        "src/apm/experiments/vamp_logt_integrator_metrics.py",
        "src/apm/experiments/vamp_logt_integrator_rotated_config.py",
        "src/apm/experiments/vamp_logt_integrator_rotated_mnist.py",
        "src/apm/experiments/vamp_logt_integrator_rotated_reporting.py",
        "src/apm/experiments/vamp_logt_integrator_rotated_workflow.py",
        "src/apm/experiments/vamp_logt_router_data.py",
        "src/apm/experiments/vamp_logt_router_config.py",
        "src/apm/experiments/vamp_logt_router_reporting.py",
        "src/apm/experiments/vamp_logt_router_rotated_config.py",
        "src/apm/experiments/vamp_logt_router_rotated_data.py",
        "src/apm/experiments/vamp_logt_router_state.py",
        "src/apm/experiments/vamp_logt_router_workflow.py",
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
            "parent_router_high_checkpoint_means": parent_summary.get(
                "condition_high_checkpoint_means"
            ),
            "parent_router_protocol_sha256": config.parent_router.protocol_sha256,
            "parent_joint_reference_sha256": {
                str(path.relative_to(config.parent_router_run_root)): file_sha256(path)
                for path in _parent_joint_reference_paths(config)
            },
            "parent_router_summary_sha256": config.parent_router.summary_sha256,
            "schema_version": "vamp-logt-integrator-protocol-v1",
            "torch_version": torch.__version__,
        },
    )


def _parent_joint_reference_paths(
    config: VampLogTIntegratorConfig,
) -> tuple[Path, ...]:
    """Return every sealed joint-IID checkpoint consumed by primary evaluation."""
    paths = tuple(
        config.parent_router_run_root
        / "primary"
        / f"seed-{seed}"
        / "joint_reference"
        / f"step-{macro_step}.pt"
        for seed in config.primary.seeds
        for macro_step in config.evaluation.full_checkpoints
    )
    if any(not path.is_file() for path in paths):
        raise ValueError("sealed parent joint-IID reference set is incomplete")
    return paths


__all__ = ["IntegratorWorkCounters", "run_phase_seed", "run_workflow"]

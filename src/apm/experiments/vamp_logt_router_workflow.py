"""Resumable LogT hierarchy and integrated behavioral-router workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from collections.abc import Mapping
import math
import os

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import Tensor

from apm.continual.artifacts import (
    ChainedJsonlLedger,
    atomic_write,
    canonical_json_bytes,
    file_sha256,
    publish_immutable_json,
)
from apm.continual.logt_behavioral_router import (
    RouterConditionState,
    RouterSupervision,
    build_router_supervision,
    create_condition_state,
    frozen_trunk_features,
    router_selections,
    sample_example_balanced,
    sample_range_balanced,
    train_condition,
)
from apm.continual.top_two_adapter import (
    TopTwoAdapterState,
    top_two_base_state,
    top_two_logits,
)
from apm.continual.vision.imagenetr.checkpoints import atomic_torch_save
from apm.experiments.vamp_logt_evidence_training import train_node_adapter
from apm.experiments.vamp_logt_router_config import (
    PhaseConfig,
    VampLogTRouterConfig,
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
from apm.experiments.vamp_logt_router_metrics import (
    fixed_policy_selections,
    routing_metric_rows,
)
from apm.experiments.vamp_logt_router_reporting import write_phase_report, write_results
from apm.experiments.vamp_logt_router_state import (
    ActiveAdapterBank,
    advance_adapter_bank,
    bank_from_record,
    bank_record,
    empty_adapter_bank,
    retire_inactive_nodes,
)


FIXED_POLICIES = (
    "oracle",
    "most_recent_range",
    "largest_range",
    "uniform_active",
)


@dataclass(frozen=True, slots=True)
class RouterWorkCounters:
    """Exact logical and physical behavioral-router work."""

    logical_current_node_evals: tuple[tuple[str, int], ...]
    logical_historical_node_evals: tuple[tuple[str, int], ...]
    physical_current_node_evals: int = 0
    physical_historical_node_evals: int = 0
    evaluation_node_evals: int = 0
    joint_reference_example_updates: int = 0

    def __post_init__(self) -> None:
        values = (
            *(value for _name, value in self.logical_current_node_evals),
            *(value for _name, value in self.logical_historical_node_evals),
            self.physical_current_node_evals,
            self.physical_historical_node_evals,
            self.evaluation_node_evals,
            self.joint_reference_example_updates,
        )
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("router work counters must be nonnegative integers")

    @classmethod
    def empty(cls, conditions: tuple[str, ...]) -> "RouterWorkCounters":
        """Return zero work for the declared condition order."""
        zero = tuple((condition, 0) for condition in conditions)
        return cls(zero, zero)

    def advanced(
        self,
        *,
        active_nodes: int,
        current_examples: int,
        historical_examples: Mapping[str, int],
        evaluation_examples: int,
        joint_updates: int,
    ) -> "RouterWorkCounters":
        """Return counters after one complete macro-step."""
        current = dict(self.logical_current_node_evals)
        historical = dict(self.logical_historical_node_evals)
        for condition in current:
            current[condition] += active_nodes * current_examples
            historical[condition] += active_nodes * historical_examples.get(condition, 0)
        return RouterWorkCounters(
            tuple((name, current[name]) for name, _value in self.logical_current_node_evals),
            tuple((name, historical[name]) for name, _value in self.logical_historical_node_evals),
            self.physical_current_node_evals + active_nodes * current_examples,
            self.physical_historical_node_evals
            + active_nodes * sum(historical_examples.values()),
            self.evaluation_node_evals + active_nodes * evaluation_examples,
            self.joint_reference_example_updates + joint_updates,
        )


@dataclass(frozen=True, slots=True)
class SeedResult:
    """One completed phase/seed artifact result."""

    phase: str
    seed: int
    directory: Path
    summary: dict[str, object]


def run_workflow(
    config: VampLogTRouterConfig,
    selected_phase: str = "all",
) -> Path:
    """Run or resume smoke and primary phases in their fixed order."""
    if selected_phase not in {"smoke", "primary", "all"}:
        raise ValueError("phase must be smoke, primary, or all")
    device = resolved_device(config.runtime.device)
    if config.runtime.deterministic_algorithms:
        torch.use_deterministic_algorithms(True)
    run_root = config.artifact_root / "runs" / config.config_hash
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"Durable behavioral-router run: {run_root}", flush=True)
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
        canonical_json_bytes({
            "config_hash": config.config_hash,
            "run_root": str(run_root),
            "schema_version": "vamp-logt-router-latest-v1",
        }),
    )
    return run_root


def run_phase_seed(
    config: VampLogTRouterConfig,
    phase_name: str,
    phase: PhaseConfig,
    seed: int,
    dependency: FrozenClassifierDependency,
    run_root: Path,
    device: torch.device,
) -> SeedResult:
    """Run or resume one shared hierarchy and its independent routers."""
    directory = run_root / phase_name / f"seed-{seed}"
    checkpoint_path = directory / "state" / "checkpoint.pt"
    nodes_root = directory / "nodes"
    ledger = ChainedJsonlLedger(directory / "metrics.jsonl", "vamp-logt-router-metric-v1")
    benchmark = build_benchmark(config, seed)
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
    model_batches = tuple(benchmark.step(step).model for step in range(1, completed_step + 1))
    router_batches = tuple(benchmark.step(step).router for step in range(1, completed_step + 1))
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
        desc=f"{phase_name} seed {seed} overall",
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
            config, dependency, bank, batches.router, device
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
                    config, dependency, bank, draw.batch, device
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
    )
    return SeedResult(phase_name, seed, directory, summary)


def _supervision(
    config: VampLogTRouterConfig,
    dependency: FrozenClassifierDependency,
    bank: ActiveAdapterBank,
    examples: ExampleBatch,
    device: torch.device,
) -> RouterSupervision:
    trunk = frozen_trunk_features(
        dependency.model,
        examples.images,
        device,
        config.evaluation.inference_batch_size,
    )
    return build_router_supervision(
        bank.topology.active_nodes,
        bank.adapters,
        trunk,
        examples.labels,
        dependency.base,
        config.router.maximum_levels,
        config.router.temperature,
        device,
        config.evaluation.inference_batch_size,
    )


def _evaluate_step(
    config: VampLogTRouterConfig,
    phase_name: str,
    phase: PhaseConfig,
    seed: int,
    macro_step: int,
    benchmark: PermutedMnistBenchmark,
    dependency: FrozenClassifierDependency,
    bank: ActiveAdapterBank,
    conditions: Mapping[str, RouterConditionState],
    evaluation_archive: ExampleBatch,
    current_evaluation: ExampleBatch,
    model_archive: ExampleBatch,
    directory: Path,
    device: torch.device,
) -> tuple[tuple[dict[str, object], ...], int, int]:
    observed_domains = tuple(
        sorted({allocation.domain_id for allocation in benchmark.allocations[:macro_step]})
    )
    test_subset = concatenate_batches(
        tuple(benchmark.test_domain(domain, full=False) for domain in observed_domains)
    )
    datasets = [
        ("current_eval", current_evaluation, None),
        ("evaluation_archive", evaluation_archive, None),
        ("test_subset", test_subset, None),
    ]
    joint_adapter = None
    joint_updates = 0
    if phase_name == "primary" and macro_step in config.evaluation.full_checkpoints:
        full_test = concatenate_batches(
            tuple(benchmark.test_domain(domain, full=True) for domain in observed_domains)
        )
        joint_adapter, joint_updates = _fit_or_load_joint_reference(
            config,
            dependency,
            model_archive,
            seed,
            macro_step,
            directory / "joint_reference",
            device,
        )
        datasets.append(("full_test", full_test, joint_adapter))
    result_rows = []
    total_examples = 0
    for scope, examples, reference in datasets:
        trunk = frozen_trunk_features(
            dependency.model,
            examples.images,
            device,
            config.evaluation.inference_batch_size,
        )
        supervision = build_router_supervision(
            bank.topology.active_nodes,
            bank.adapters,
            trunk,
            examples.labels,
            dependency.base,
            config.router.maximum_levels,
            config.router.temperature,
            device,
            config.evaluation.inference_batch_size,
        )
        joint_logits = (
            None
            if reference is None
            else _adapter_logits(
                reference,
                dependency,
                trunk,
                device,
                config.evaluation.inference_batch_size,
            )
        )
        for policy in FIXED_POLICIES:
            selections = fixed_policy_selections(
                policy,
                bank.topology.active_nodes,
                supervision,
                named_seed(seed, macro_step, scope, policy),
            )
            result_rows.extend(
                routing_metric_rows(
                    condition=policy,
                    selections=selections,
                    probabilities=None,
                    inactive_attempts=None,
                    examples=examples,
                    supervision=supervision,
                    nodes=bank.topology.active_nodes,
                    run_seed=seed,
                    macro_step=macro_step,
                    evaluation_scope=scope,
                    near_oracle_thresholds=config.evaluation.near_oracle_thresholds,
                    joint_logits=joint_logits,
                )
            )
        for condition_name, state in conditions.items():
            selections, probabilities, inactive = router_selections(
                state.router,
                supervision,
                device,
                config.evaluation.inference_batch_size,
            )
            result_rows.extend(
                routing_metric_rows(
                    condition=condition_name,
                    selections=selections,
                    probabilities=probabilities,
                    inactive_attempts=inactive,
                    examples=examples,
                    supervision=supervision,
                    nodes=bank.topology.active_nodes,
                    run_seed=seed,
                    macro_step=macro_step,
                    evaluation_scope=scope,
                    near_oracle_thresholds=config.evaluation.near_oracle_thresholds,
                    joint_logits=joint_logits,
                )
            )
        total_examples += len(examples.labels)
    return (
        tuple({**row, "row_type": "evaluation"} for row in result_rows),
        total_examples,
        joint_updates,
    )


def _fit_or_load_joint_reference(
    config: VampLogTRouterConfig,
    dependency: FrozenClassifierDependency,
    model_archive: ExampleBatch,
    seed: int,
    macro_step: int,
    root: Path,
    device: torch.device,
) -> tuple[TopTwoAdapterState, int]:
    path = root / f"step-{macro_step}.pt"
    reference_seed = named_seed(seed, "joint-reference", macro_step)
    if path.is_file():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (
            payload.get("schema_version") != "vamp-logt-router-joint-reference-v1"
            or payload.get("macro_step") != macro_step
            or payload.get("seed") != reference_seed
        ):
            raise ValueError("joint-IID reference coordinates changed")
        return TopTwoAdapterState(*payload["parameters"]), int(payload["example_updates"])
    trunk = frozen_trunk_features(
        dependency.model,
        model_archive.images,
        device,
        config.evaluation.inference_batch_size,
    )
    result = train_node_adapter(
        trunk,
        model_archive.labels,
        dependency.base,
        config.adapter,
        reference_seed,
        device,
        f"joint-IID step {macro_step}",
        config.runtime.progress,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(
        path,
        {
            "example_updates": result.example_updates,
            "macro_step": macro_step,
            "parameters": tuple(tensor.detach().cpu() for tensor in result.adapter.tensors),
            "schema_version": "vamp-logt-router-joint-reference-v1",
            "seed": reference_seed,
        },
    )
    return result.adapter, result.example_updates


def _adapter_logits(
    adapter: TopTwoAdapterState,
    dependency: FrozenClassifierDependency,
    trunk: Tensor,
    device: torch.device,
    batch_size: int,
) -> Tensor:
    base = top_two_base_state(*dependency.base.tensors, device=device)
    target = TopTwoAdapterState(
        *(tensor.detach().to(device).clone() for tensor in adapter.tensors)
    )
    rows = []
    with torch.inference_mode():
        for offset in range(0, len(trunk), batch_size):
            rows.append(top_two_logits(trunk[offset : offset + batch_size].to(device), base, target).cpu())
    return torch.cat(rows)


def _save_seed_checkpoint(
    path: Path,
    config: VampLogTRouterConfig,
    phase_name: str,
    seed: int,
    macro_step: int,
    bank: ActiveAdapterBank,
    conditions: Mapping[str, RouterConditionState],
    work: RouterWorkCounters,
    metric_rows: int,
) -> None:
    payload = {
        "bank": bank_record(bank),
        "completed_macro_step": macro_step,
        "conditions": {
            name: {
                "optimizer": state.optimizer.state_dict(),
                "optimizer_steps": state.optimizer_steps,
                "router": state.router.state_dict(),
            }
            for name, state in conditions.items()
        },
        "config_hash": config.config_hash,
        "metric_rows": metric_rows,
        "phase": phase_name,
        "run_seed": seed,
        "schema_version": "vamp-logt-router-checkpoint-v1",
        "work": asdict(work),
    }
    atomic_torch_save(path, payload)


def _load_seed_checkpoint(
    path: Path,
    config: VampLogTRouterConfig,
    phase_name: str,
    seed: int,
    phase: PhaseConfig,
    conditions: Mapping[str, RouterConditionState],
    device: torch.device,
) -> tuple[ActiveAdapterBank, RouterWorkCounters, int, int]:
    if not path.is_file():
        return (
            empty_adapter_bank(config.benchmark.model_batch_size),
            RouterWorkCounters.empty(phase.conditions),
            0,
            0,
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("schema_version") != "vamp-logt-router-checkpoint-v1"
        or payload.get("config_hash") != config.config_hash
        or payload.get("phase") != phase_name
        or payload.get("run_seed") != seed
        or set(payload.get("conditions", {})) != set(conditions)
    ):
        raise ValueError("behavioral-router checkpoint coordinates changed")
    for name, state in conditions.items():
        record = payload["conditions"][name]
        state.router.load_state_dict(record["router"], strict=True)
        state.optimizer.load_state_dict(record["optimizer"])
        _optimizer_to(state.optimizer, device)
        state.optimizer_steps = int(record["optimizer_steps"])
    work = payload["work"]
    return (
        bank_from_record(payload["bank"]),
        RouterWorkCounters(
            tuple((str(name), int(value)) for name, value in work["logical_current_node_evals"]),
            tuple((str(name), int(value)) for name, value in work["logical_historical_node_evals"]),
            int(work["physical_current_node_evals"]),
            int(work["physical_historical_node_evals"]),
            int(work["evaluation_node_evals"]),
            int(work["joint_reference_example_updates"]),
        ),
        int(payload["completed_macro_step"]),
        int(payload["metric_rows"]),
    )


def _optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for values in optimizer.state.values():
        for key, value in values.items():
            if isinstance(value, Tensor):
                values[key] = value.to(device)


def _require_smoke_gate(run_root: Path, config: VampLogTRouterConfig) -> None:
    summary_path = run_root / "smoke" / f"seed-{config.smoke.seeds[0]}" / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError("the exact smoke must complete before the primary phase")
    from apm.continual.artifacts import load_canonical_json

    acceptance = load_canonical_json(summary_path)["acceptance"]
    required = (
        "all_primary_metrics_finite",
        "exact_historical_budget",
        "inactive_levels_never_selected",
        "nonnegative_routing_regret",
        "single_candidate_parity",
    )
    if not all(bool(acceptance[name]) for name in required) or float(
        acceptance["router_loss_decrease_fraction"]
    ) <= 0.5:
        raise RuntimeError("the exact smoke did not pass its structural integration gate")


def _write_protocol(
    run_root: Path,
    config: VampLogTRouterConfig,
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
        "configs/vamp_logt_router_mnist/primary.yaml",
        "docs/logt_vamp_mnist_integrated_router_plan.md",
        "src/apm/continual/logt_behavioral_router.py",
        "src/apm/continual/logt_evidence_bank.py",
        "src/apm/continual/top_two_adapter.py",
        "src/apm/experiments/vamp_logt_router_config.py",
        "src/apm/experiments/vamp_logt_router_data.py",
        "src/apm/experiments/vamp_logt_router_metrics.py",
        "src/apm/experiments/vamp_logt_router_mnist.py",
        "src/apm/experiments/vamp_logt_router_reporting.py",
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
            "schema_version": "vamp-logt-router-protocol-v1",
            "torch_version": torch.__version__,
        },
    )


__all__ = [
    "RouterWorkCounters",
    "SeedResult",
    "run_phase_seed",
    "run_workflow",
]

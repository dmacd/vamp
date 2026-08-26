"""Resumable VAMP-AF MNIST preflight, comparison matrix, and pass workflow."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from collections.abc import Mapping, Sequence
import math
import os
import time

import numpy as np
from pyrsistent import pvector
from pyrsistent.typing import PVector
import torch
from torch import Tensor
from torch.nn import functional as F

from apm.continual.addressing_first import (
    AFHyperparameters,
    AFState,
    StoredExampleTable,
    collapse_leaf_pair,
    current_depth_cap,
    effective_adapter,
    init_af_state,
    initialize_split_children,
    install_split,
    predict_for_node,
    structural_action,
    update_microbatch,
    validate_af_state,
)
from apm.continual.artifacts import (
    ChainedJsonlLedger,
    atomic_write,
    canonical_json_bytes,
    file_sha256,
    load_canonical_json,
    publish_immutable_json,
)
from apm.continual.top_two_adapter import (
    TopTwoAdapterState,
    TopTwoAdamWState,
    TopTwoBaseState,
    TopTwoOptimizerConfig,
    top_two_base_state,
    top_two_logits,
    train_top_two_adapter_step,
    zero_top_two_adapter,
    zero_top_two_adamw,
)
from apm.continual.vision.imagenetr.checkpoints import atomic_torch_save
from apm.experiments.vamp_af_config import PassConfig, VampAFConfig
from apm.experiments.vamp_af_data import (
    FeatureTables,
    build_feature_tables,
    load_arrays,
    pass_training_table,
    resolved_device,
    train_or_load_base,
)


CONDITIONS = ("frozen_base", "global_replay", "af", "oracle_context", "joint_iid")


@dataclass(frozen=True, slots=True)
class BaselineState:
    """One online top-two adapter, replay membership, and presentation total."""

    adapter: TopTwoAdapterState
    optimizer: TopTwoAdamWState
    buffer: PVector[int]
    presentations: int


@dataclass(frozen=True, slots=True)
class OracleState:
    """Five independent context adapters and their isolated replay buffers."""

    adapters: tuple[TopTwoAdapterState, ...]
    optimizers: tuple[TopTwoAdamWState, ...]
    buffers: tuple[PVector[int], ...]
    presentations: int


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Long-form metric rows plus tree/routing diagnostic rows."""

    metrics: tuple[dict[str, object], ...]
    routing: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class PassSeedResult:
    """Completed one pass/seed directory and its summary payload."""

    directory: Path
    summary: dict[str, object]


@dataclass(frozen=True, slots=True)
class RunCheckpoint:
    """Exact resumable online boundary for one pass and seed."""

    next_offset: int
    af_state: AFState
    global_state: BaselineState | None
    oracle_state: OracleState | None
    metric_rows: int
    routing_rows: tuple[dict[str, object], ...]
    consolidation_rows: tuple[dict[str, object], ...]
    split_events: int
    consolidation_events: int


def run_workflow(
    config: VampAFConfig,
    *,
    preflight_only: bool = False,
    stop_after_pass: str | None = None,
) -> Path:
    """Run the shared-base gate and all three fixed experiment passes."""
    pass_names = tuple(pass_config.name for pass_config in config.passes)
    if stop_after_pass is not None and stop_after_pass not in pass_names:
        raise ValueError(f"unknown VAMP-AF stop pass: {stop_after_pass}")
    if config.runtime.deterministic_algorithms:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    torch.manual_seed(config.base.seed)
    np.random.seed(config.base.seed)
    device = resolved_device(config.runtime.device)
    run_root = config.artifact_root / "runs" / config.config_hash
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"VAMP-AF durable working directory: {run_root}", flush=True)
    _write_resolved_yaml(run_root / "config_resolved.yaml", config.as_record())
    print("Phase 1/5: shared CNN", flush=True)
    arrays = load_arrays(config)
    base = train_or_load_base(config, arrays, run_root, device)
    publish_immutable_json(
        run_root / "protocol.json",
        {
            "base_checkpoint_sha256": base.sha256,
            "config": config.as_record(),
            "config_hash": config.config_hash,
            "data_sha256": {
                path.name: file_sha256(path)
                for path in sorted(config.data_root.glob("*-ubyte"))
            },
            "material_source_sha256": _material_source_hashes(),
            "schema_version": "vamp-af-protocol-v2",
            "torch_version": torch.__version__,
        },
    )
    print(
        f"Shared CNN: epochs={base.selected_epochs}, test accuracy={base.test_accuracy:.4f}",
        flush=True,
    )
    print("Phase 2/5: frozen feature cache", flush=True)
    tables = build_feature_tables(config, arrays, base, run_root, device)
    suffix_base = top_two_base_state(
        base.model.embedding.weight,
        base.model.embedding.bias,
        base.model.classifier.weight,
        base.model.classifier.bias,
        device,
    )
    print("Phase 3/5: representation preflight", flush=True)
    preflight = run_preflight(config, tables, suffix_base, run_root, device)
    if not bool(preflight["passed"]):
        _write_blocked_handoff(run_root, preflight)
        if preflight_only:
            return run_root
        raise RuntimeError("VAMP-AF representation preflight failed; AF runs were not started")
    if preflight_only:
        return run_root

    completed = []
    total_runs = sum(len(pass_config.seeds) for pass_config in config.passes)
    completed_runs = 0
    for pass_config in config.passes:
        for seed in pass_config.seeds:
            completed_runs += 1
            print(
                f"Phase 4/5: {pass_config.name} seed {seed} "
                f"({completed_runs}/{total_runs})",
                flush=True,
            )
            completed.append(
                run_pass_seed(config, pass_config, seed, tables, suffix_base, run_root, device)
            )
        if pass_config.name == stop_after_pass:
            return run_root
    print("Phase 5/5: aggregate reports and acceptance", flush=True)
    from apm.experiments.vamp_af_reporting import write_aggregate_report

    write_aggregate_report(run_root, config, preflight, tuple(completed))
    return run_root


def run_preflight(
    config: VampAFConfig,
    tables: FeatureTables,
    top_two_base: TopTwoBaseState,
    run_root: Path,
    device: torch.device,
) -> dict[str, object]:
    """Measure frozen-address context information and top-two adapter capacity."""
    path = run_root / "preflight" / "summary.json"
    if path.is_file():
        cached = load_canonical_json(path)
        if cached.get("schema_version") != "vamp-af-preflight-v2":
            raise ValueError("VAMP-AF preflight schema changed inside one run identity")
        return cached
    training = tables.train
    test = tables.test
    context_accuracy = _context_probe_accuracy(
        training.embeddings,
        training.context_ids,
        test.embeddings,
        test.context_ids,
        _top_two_optimizer_config(config),
        config.preflight.batch_size,
        config.preflight.epochs,
        config.base.seed,
        device,
        config.runtime.progress,
    )
    optimizer_config = _top_two_optimizer_config(config)
    oracle_accuracies: list[float] = []
    for context_id in range(5):
        train_ids = torch.nonzero(training.context_ids == context_id, as_tuple=True)[0]
        test_ids = torch.nonzero(test.context_ids == context_id, as_tuple=True)[0]
        oracle = _train_top_two_epochs(
            training.trunk_features[train_ids],
            training.labels[train_ids],
            top_two_base,
            optimizer_config,
            config.preflight.batch_size,
            config.preflight.epochs,
            config.base.seed + 100 + context_id,
            device,
            f"preflight top-two oracle context {context_id}",
            config.runtime.progress,
        )
        oracle_accuracies.append(
            _top_two_accuracy(
                oracle,
                top_two_base,
                test.trunk_features[test_ids],
                test.labels[test_ids],
                device,
            )
        )
    oracle_mean = float(np.mean(oracle_accuracies))
    oracle_gate_passed = oracle_mean >= config.preflight.oracle_accuracy_minimum
    joint_accuracy: float | None = None
    if oracle_gate_passed:
        joint_adapter = _train_top_two_epochs(
            training.trunk_features,
            training.labels,
            top_two_base,
            optimizer_config,
            config.preflight.batch_size,
            config.preflight.epochs,
            config.base.seed + 1_000,
            device,
            "preflight joint top-two adapter",
            config.runtime.progress,
        )
        joint_accuracy = _top_two_accuracy(
            joint_adapter,
            top_two_base,
            test.trunk_features,
            test.labels,
            device,
        )
    result: dict[str, object] = {
        "adapter_class": "full_top_two_delta",
        "context_accuracy": context_accuracy,
        "context_accuracy_reference": config.preflight.context_accuracy_reference,
        "context_reference_met": context_accuracy >= config.preflight.context_accuracy_reference,
        "joint_accuracy": joint_accuracy,
        "joint_near_oracle": (
            oracle_mean - joint_accuracy <= config.preflight.joint_oracle_tolerance
            if joint_accuracy is not None
            else None
        ),
        "oracle_context_accuracies": oracle_accuracies,
        "oracle_gate_passed": oracle_gate_passed,
        "oracle_context_mean_accuracy": oracle_mean,
        "oracle_context_minimum": config.preflight.oracle_accuracy_minimum,
        "passed": oracle_gate_passed,
        "routing_address": "frozen_normalized_base_embedding",
        "schema_version": "vamp-af-preflight-v2",
    }
    publish_immutable_json(path, result)
    print(
        f"Preflight: context={context_accuracy:.4f}, top-two oracle={oracle_mean:.4f}, "
        f"joint={joint_accuracy if joint_accuracy is not None else 'not-run'}",
        flush=True,
    )
    return result


def run_pass_seed(
    config: VampAFConfig,
    pass_config: PassConfig,
    seed: int,
    tables: FeatureTables,
    base: TopTwoBaseState,
    run_root: Path,
    device: torch.device,
) -> PassSeedResult:
    """Run or reuse one exact pass/seed comparison directory."""
    directory = run_root / "passes" / pass_config.name / f"seed-{seed}"
    summary_path = directory / "summary.json"
    if summary_path.is_file():
        from apm.experiments.vamp_af_reporting import REQUIRED_ARTIFACTS

        missing = tuple(name for name in REQUIRED_ARTIFACTS if not (directory / name).is_file())
        if missing:
            raise ValueError(f"completed VAMP-AF pass is missing artifacts: {missing}")
        summary = load_canonical_json(summary_path)
        if pass_config.require_consolidation and int(summary["consolidation_events"]) < 1:
            raise RuntimeError("forced consolidation pass completed without a collapse")
        return PassSeedResult(directory, summary)
    directory.mkdir(parents=True, exist_ok=True)
    metrics = ChainedJsonlLedger(directory / "metrics.jsonl", "vamp-af-metric-v1")
    table = pass_training_table(tables, pass_config.examples_per_context, seed)
    hyperparameters = _af_hyperparameters(config, pass_config)
    checkpoint_path = directory / "state" / "checkpoint.pt"
    if checkpoint_path.is_file():
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if payload.get("schema_version") != "vamp-af-run-checkpoint-v2":
            raise ValueError("VAMP-AF checkpoint schema changed inside one run identity")
        checkpoint = payload["checkpoint"]
        if not isinstance(checkpoint, RunCheckpoint):
            raise ValueError("VAMP-AF run checkpoint is malformed")
        metrics.truncate(checkpoint.metric_rows)
    else:
        af_state = init_af_state(base, device)
        global_state = (
            _initial_baseline(base, device) if "global_replay" in pass_config.conditions else None
        )
        oracle_state = (
            _initial_oracle(base, device) if "oracle_context" in pass_config.conditions else None
        )
        initial = _evaluate_online_conditions(
            pass_config,
            seed,
            0,
            "before_stream",
            af_state,
            global_state,
            oracle_state,
            tables.test,
            device,
        )
        metrics.append_many(initial.metrics)
        checkpoint = RunCheckpoint(
            0,
            af_state,
            global_state,
            oracle_state,
            len(metrics.rows),
            initial.routing,
            (),
            0,
            0,
        )
        _save_run_checkpoint(checkpoint_path, checkpoint)

    try:
        from tqdm.auto import tqdm
    except ImportError as error:  # pragma: no cover - environment gate
        raise RuntimeError("tqdm is required by the vision environment") from error
    started = time.monotonic()
    for offset in tqdm(
        range(checkpoint.next_offset, table.embeddings.shape[0], config.adapter.batch_size),
        initial=checkpoint.next_offset // config.adapter.batch_size,
        total=math.ceil(table.embeddings.shape[0] / config.adapter.batch_size),
        desc=f"{pass_config.name} seed {seed}",
        disable=not config.runtime.progress,
        unit="batch",
    ):
        previous_split_events = checkpoint.split_events
        previous_consolidation_events = checkpoint.consolidation_events
        batch_ids = tuple(range(offset, min(offset + config.adapter.batch_size, table.embeddings.shape[0])))
        global_state = checkpoint.global_state
        oracle_state = checkpoint.oracle_state
        if global_state is not None:
            global_state = _update_baseline(
                global_state,
                checkpoint.af_state.base,
                table,
                batch_ids,
                hyperparameters,
                _seed(seed, "global", offset),
                device,
            )
        if oracle_state is not None:
            oracle_state = _update_oracle(
                oracle_state,
                checkpoint.af_state.base,
                table,
                batch_ids,
                hyperparameters,
                _seed(seed, "oracle", offset),
                device,
            )
        microbatch = update_microbatch(
            checkpoint.af_state, table, batch_ids, hyperparameters, _seed(seed, "af", offset)
        )
        af_state = microbatch.state
        routing_rows = list(checkpoint.routing_rows)
        consolidation_rows = list(checkpoint.consolidation_rows)
        split_events = checkpoint.split_events
        consolidation_events = checkpoint.consolidation_events
        stream_examples = batch_ids[-1] + 1
        for touched_leaf in microbatch.touched_leaf_ids:
            action = structural_action(af_state, touched_leaf, stream_examples, hyperparameters)
            if action == "split":
                before = _evaluate_af(
                    af_state,
                    tables.test,
                    pass_config,
                    seed,
                    stream_examples,
                    "pre_split",
                    device,
                )
                metrics.append_many(before.metrics)
                routing_rows.extend(before.routing)
                old_state = af_state
                af_state, event = install_split(
                    af_state,
                    table,
                    touched_leaf,
                    stream_examples,
                    hyperparameters,
                    _seed(seed, "split", stream_examples, touched_leaf),
                )
                _require_zero_split_parity(old_state, af_state, table, event.parent_id)
                zero_child = _evaluate_af(
                    af_state,
                    tables.test,
                    pass_config,
                    seed,
                    stream_examples,
                    "post_split_zero",
                    device,
                )
                metrics.append_many(zero_child.metrics)
                routing_rows.extend(zero_child.routing)
                af_state = initialize_split_children(
                    af_state,
                    table,
                    event,
                    hyperparameters,
                    _seed(seed, "split-init", stream_examples, touched_leaf),
                )
                initialized = _evaluate_af(
                    af_state,
                    tables.test,
                    pass_config,
                    seed,
                    stream_examples,
                    "post_split_init",
                    device,
                )
                metrics.append_many(initialized.metrics)
                routing_rows.extend(initialized.routing)
                split_events += 1
            elif action == "collapse":
                before = _evaluate_af(
                    af_state,
                    tables.test,
                    pass_config,
                    seed,
                    stream_examples,
                    "pre_consolidation",
                    device,
                )
                metrics.append_many(before.metrics)
                routing_rows.extend(before.routing)
                before_average = _average_accuracy(before.metrics)
                af_state, event = collapse_leaf_pair(
                    af_state,
                    table,
                    touched_leaf,
                    stream_examples,
                    hyperparameters,
                    _seed(seed, "collapse", stream_examples, touched_leaf),
                )
                after = _evaluate_af(
                    af_state,
                    tables.test,
                    pass_config,
                    seed,
                    stream_examples,
                    "post_consolidation",
                    device,
                )
                metrics.append_many(after.metrics)
                routing_rows.extend(after.routing)
                after_average = _average_accuracy(after.metrics)
                consolidation_rows.append(
                    {
                        "accuracy_after": after_average,
                        "accuracy_before": before_average,
                        "accuracy_change": after_average - before_average,
                        "example_count": len(event.example_ids),
                        "parent_id": event.parent_id,
                        "removed_child_ids": list(event.removed_child_ids),
                        "step": stream_examples,
                    }
                )
                consolidation_events += 1
        validate_af_state(af_state, range(stream_examples))
        next_offset = batch_ids[-1] + 1
        boundary_reasons = _boundary_reasons(
            next_offset,
            pass_config.examples_per_context,
            table.embeddings.shape[0],
            config.runtime.evaluation_interval,
        )
        if boundary_reasons:
            evaluation = _evaluate_online_conditions(
                pass_config,
                seed,
                next_offset,
                "+".join(boundary_reasons),
                af_state,
                global_state,
                oracle_state,
                tables.test,
                device,
            )
            metrics.append_many(evaluation.metrics)
            routing_rows.extend(evaluation.routing)
        checkpoint = RunCheckpoint(
            next_offset,
            af_state,
            global_state,
            oracle_state,
            len(metrics.rows),
            tuple(routing_rows),
            tuple(consolidation_rows),
            split_events,
            consolidation_events,
        )
        if (
            boundary_reasons
            or split_events != previous_split_events
            or consolidation_events != previous_consolidation_events
        ):
            _save_run_checkpoint(checkpoint_path, checkpoint)
    _save_run_checkpoint(checkpoint_path, checkpoint)

    if "joint_iid" in pass_config.conditions:
        if checkpoint.global_state is None:
            raise RuntimeError("joint-IID presentation matching requires global replay")
        joint = _train_adapter_presentations(
            table,
            checkpoint.af_state.base,
            checkpoint.global_state.presentations,
            hyperparameters,
            _seed(seed, "joint"),
            device,
            config.runtime.progress,
        )
        joint_result = _evaluate_adapter(
            "joint_iid",
            joint,
            checkpoint.af_state.base,
            tables.test,
            pass_config,
            seed,
            table.embeddings.shape[0],
            "final",
            device,
        )
        metrics.append_many(joint_result)
    _ensure_final_aliases(metrics, pass_config)
    from apm.experiments.vamp_af_reporting import write_pass_artifacts

    if pass_config.require_consolidation and checkpoint.consolidation_events < 1:
        raise RuntimeError("forced consolidation pass completed without a collapse")
    summary = write_pass_artifacts(
        directory,
        config,
        pass_config,
        seed,
        checkpoint.af_state,
        table,
        tuple(metrics.rows),
        checkpoint.routing_rows,
        checkpoint.consolidation_rows,
        time.monotonic() - started,
    )
    return PassSeedResult(directory, summary)


def _initial_baseline(base: TopTwoBaseState, device: torch.device) -> BaselineState:
    adapter = zero_top_two_adapter(base, device)
    return BaselineState(adapter, zero_top_two_adamw(adapter), pvector(), 0)


def _initial_oracle(base: TopTwoBaseState, device: torch.device) -> OracleState:
    adapters = tuple(zero_top_two_adapter(base, device) for _ in range(5))
    return OracleState(
        adapters,
        tuple(zero_top_two_adamw(adapter) for adapter in adapters),
        tuple(pvector() for _ in adapters),
        0,
    )


def _update_baseline(
    state: BaselineState,
    base: TopTwoBaseState,
    table: StoredExampleTable,
    new_ids: Sequence[int],
    hyperparameters: AFHyperparameters,
    seed: int,
    device: torch.device,
) -> BaselineState:
    replay = _replay(tuple(state.buffer), len(new_ids), seed)
    ids = torch.as_tensor(tuple(new_ids) + replay, dtype=torch.int64)
    adapter, optimizer, _loss = train_top_two_adapter_step(
        table.trunk_features[ids].to(device),
        table.labels[ids].to(device),
        base,
        zero_top_two_adapter(base, device),
        state.adapter,
        state.optimizer,
        _top_two_optimizer_from_hyperparameters(hyperparameters),
    )
    return BaselineState(
        adapter,
        optimizer,
        state.buffer.extend(int(value) for value in new_ids),
        state.presentations + len(ids),
    )


def _update_oracle(
    state: OracleState,
    base: TopTwoBaseState,
    table: StoredExampleTable,
    new_ids: Sequence[int],
    hyperparameters: AFHyperparameters,
    seed: int,
    device: torch.device,
) -> OracleState:
    context_ids = tuple(sorted({int(table.context_ids[example_id]) for example_id in new_ids}))
    adapters, optimizers, buffers = list(state.adapters), list(state.optimizers), list(state.buffers)
    presentations = state.presentations
    for context_id in context_ids:
        arrivals = tuple(example_id for example_id in new_ids if int(table.context_ids[example_id]) == context_id)
        replay = _replay(tuple(buffers[context_id]), len(arrivals), _seed(seed, context_id))
        ids = torch.as_tensor(arrivals + replay, dtype=torch.int64)
        adapters[context_id], optimizers[context_id], _loss = train_top_two_adapter_step(
            table.trunk_features[ids].to(device),
            table.labels[ids].to(device),
            base,
            zero_top_two_adapter(base, device),
            adapters[context_id],
            optimizers[context_id],
            _top_two_optimizer_from_hyperparameters(hyperparameters),
        )
        buffers[context_id] = buffers[context_id].extend(arrivals)
        presentations += len(ids)
    return OracleState(tuple(adapters), tuple(optimizers), tuple(buffers), presentations)


def _evaluate_online_conditions(
    pass_config: PassConfig,
    seed: int,
    step: int,
    event: str,
    af_state: AFState,
    global_state: BaselineState | None,
    oracle_state: OracleState | None,
    test: StoredExampleTable,
    device: torch.device,
) -> EvaluationResult:
    metrics = []
    routing = []
    if "frozen_base" in pass_config.conditions:
        metrics.extend(
            _evaluate_adapter(
                "frozen_base",
                zero_top_two_adapter(af_state.base, device),
                af_state.base,
                test,
                pass_config,
                seed,
                step,
                event,
                device,
            )
        )
    if global_state is not None:
        metrics.extend(
            _evaluate_adapter(
                "global_replay",
                global_state.adapter,
                af_state.base,
                test,
                pass_config,
                seed,
                step,
                event,
                device,
            )
        )
    if oracle_state is not None:
        metrics.extend(
            _evaluate_oracle_context(
                oracle_state, af_state.base, test, pass_config, seed, step, event, device
            )
        )
    af = _evaluate_af(af_state, test, pass_config, seed, step, event, device)
    metrics.extend(af.metrics)
    routing.extend(af.routing)
    return EvaluationResult(tuple(metrics), tuple(routing))


def _evaluate_adapter(
    condition: str,
    adapter: TopTwoAdapterState,
    base: TopTwoBaseState,
    test: StoredExampleTable,
    pass_config: PassConfig,
    seed: int,
    step: int,
    event: str,
    device: torch.device,
) -> tuple[dict[str, object], ...]:
    logits = _adapter_logits(adapter, base, test.trunk_features, device)
    return _metric_rows(condition, logits, test, pass_config, seed, step, event)


def _evaluate_oracle_context(
    state: OracleState,
    base: TopTwoBaseState,
    test: StoredExampleTable,
    pass_config: PassConfig,
    seed: int,
    step: int,
    event: str,
    device: torch.device,
) -> tuple[dict[str, object], ...]:
    logits = torch.empty_like(test.base_logits)
    for context_id, adapter in enumerate(state.adapters):
        ids = torch.nonzero(test.context_ids == context_id, as_tuple=True)[0]
        logits[ids] = _adapter_logits(adapter, base, test.trunk_features[ids], device)
    return _metric_rows("oracle_context", logits, test, pass_config, seed, step, event)


def _evaluate_af(
    state: AFState,
    test: StoredExampleTable,
    pass_config: PassConfig,
    seed: int,
    step: int,
    event: str,
    device: torch.device,
) -> EvaluationResult:
    routed_leaf_ids, route_depths, minimum_margins = _route_many(state, test.embeddings)
    leaf_ids = tuple(sorted(state.leaf_buffers))
    leaf_index = {leaf_id: index for index, leaf_id in enumerate(leaf_ids)}
    adapters = tuple(effective_adapter(state, leaf_id) for leaf_id in leaf_ids)
    routed_logits = torch.empty_like(test.base_logits)
    oracle_losses = torch.empty(test.embeddings.shape[0], dtype=torch.float32)
    oracle_leaf_ids = torch.empty(test.embeddings.shape[0], dtype=torch.int64)
    oracle_predictions = torch.empty(test.embeddings.shape[0], dtype=torch.int64)
    batch_size = 2_048
    for offset in range(0, test.embeddings.shape[0], batch_size):
        stop = min(offset + batch_size, test.embeddings.shape[0])
        trunk_features = test.trunk_features[offset:stop].to(device)
        labels = test.labels[offset:stop].to(device)
        candidate_logits = torch.stack(
            tuple(top_two_logits(trunk_features, state.base, adapter) for adapter in adapters),
            dim=1,
        )
        candidate_losses = -F.log_softmax(candidate_logits, dim=2).gather(
            2,
            labels[:, None, None].expand(-1, len(leaf_ids), 1),
        ).squeeze(2)
        winner = candidate_losses.argmin(dim=1)
        routed_indices = torch.as_tensor(
            tuple(leaf_index[int(value)] for value in routed_leaf_ids[offset:stop]),
            dtype=torch.int64,
            device=device,
        )
        routed_logits[offset:stop] = candidate_logits[
            torch.arange(stop - offset, device=device), routed_indices
        ].cpu()
        oracle_losses[offset:stop] = candidate_losses[
            torch.arange(stop - offset, device=device), winner
        ].cpu()
        oracle_leaf_ids[offset:stop] = torch.as_tensor(
            tuple(leaf_ids[int(value)] for value in winner.cpu()), dtype=torch.int64
        )
        oracle_predictions[offset:stop] = candidate_logits[
            torch.arange(stop - offset, device=device), winner
        ].argmax(dim=1).cpu()
    rows = list(_metric_rows("af", routed_logits, test, pass_config, seed, step, event))
    routed_losses = F.cross_entropy(routed_logits, test.labels, reduction="none")
    routed_ids = torch.as_tensor(routed_leaf_ids, dtype=torch.int64)
    for row in rows:
        context_id = int(row["context_id"])
        mask = torch.ones(len(routed_ids), dtype=torch.bool) if context_id == -1 else test.context_ids == context_id
        oracle_accuracy = float((oracle_predictions[mask] == test.labels[mask]).float().mean().item())
        row.update(
            {
                "oracle_leaf_accuracy": oracle_accuracy,
                "oracle_leaf_loss": float(oracle_losses[mask].mean().item()),
                "route_oracle_agreement": float((routed_ids[mask] == oracle_leaf_ids[mask]).float().mean().item()),
                "routing_loss_regret": float((routed_losses[mask] - oracle_losses[mask]).mean().item()),
            }
        )
    routing = _routing_rows(
        state,
        test,
        pass_config,
        seed,
        step,
        event,
        routed_ids,
        route_depths,
        minimum_margins,
        rows[-1],
    )
    return EvaluationResult(tuple(rows), routing)


def _route_many(state: AFState, embeddings: Tensor) -> tuple[tuple[int, ...], Tensor, Tensor]:
    count = embeddings.shape[0]
    leaf_ids = torch.full((count,), -1, dtype=torch.int64)
    depths = torch.zeros(count, dtype=torch.int64)
    margins = torch.full((count,), math.inf, dtype=torch.float32)
    pending = [(state.root_id, torch.arange(count, dtype=torch.int64))]
    while pending:
        node_id, ids = pending.pop()
        node = state.nodes[node_id]
        if node.is_leaf:
            leaf_ids[ids] = node_id
            continue
        if node.split_direction is None or node.left_id is None or node.right_id is None:
            raise RuntimeError("internal AF node is incomplete")
        scores = embeddings[ids] @ node.split_direction
        local_margins = torch.abs(scores - float(node.split_threshold))
        margins[ids] = torch.minimum(margins[ids], local_margins)
        depths[ids] += 1
        left_mask = scores <= float(node.split_threshold)
        if torch.any(left_mask):
            pending.append((node.left_id, ids[left_mask]))
        if torch.any(~left_mask):
            pending.append((node.right_id, ids[~left_mask]))
    if torch.any(leaf_ids < 0):
        raise RuntimeError("vectorized routing left examples unassigned")
    margins[torch.isinf(margins)] = math.nan
    return tuple(int(value) for value in leaf_ids.tolist()), depths, margins


def _routing_rows(
    state: AFState,
    test: StoredExampleTable,
    pass_config: PassConfig,
    seed: int,
    step: int,
    event: str,
    routed_ids: Tensor,
    depths: Tensor,
    margins: Tensor,
    average_metric: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    finite_margins = margins[torch.isfinite(margins)]
    aggregate: dict[str, object] = {
        "adapter_norm": None,
        "context_entropy_bits": None,
        "counted_work": state.counters.counted_work,
        "dominant_context": None,
        "event": event,
        "leaf_id": None,
        "leaf_traffic_fraction": None,
        "leaves": len(state.leaf_buffers),
        "margin_p10": float(torch.quantile(finite_margins, 0.1).item()) if len(finite_margins) else None,
        "margin_p50": float(torch.quantile(finite_margins, 0.5).item()) if len(finite_margins) else None,
        "max_depth": int(depths.max().item()),
        "mean_depth": float(depths.float().mean().item()),
        "depth_p50": float(torch.quantile(depths.float(), 0.50).item()),
        "depth_p90": float(torch.quantile(depths.float(), 0.90).item()),
        "depth_p95": float(torch.quantile(depths.float(), 0.95).item()),
        "depth_p99": float(torch.quantile(depths.float(), 0.99).item()),
        "nodes": len(state.nodes),
        "oracle_leaf_accuracy": average_metric["oracle_leaf_accuracy"],
        "pass": pass_config.name,
        "route_oracle_agreement": average_metric["route_oracle_agreement"],
        "routed_accuracy": average_metric["accuracy"],
        "row_type": "aggregate",
        "sample_count": int(test.embeddings.shape[0]),
        "seed": seed,
        "step": step,
        "t_log2_t": step * math.log2(step + 1) if step else 0.0,
        "work_ratio": _work_ratio(state, step),
    }
    rows = [aggregate]
    total_test = len(routed_ids)
    for node_id in sorted(state.nodes):
        is_leaf = node_id in state.leaf_buffers
        members = tuple(state.leaf_buffers[node_id]) if is_leaf else ()
        traffic_mask = routed_ids == node_id if is_leaf else torch.zeros_like(routed_ids, dtype=torch.bool)
        traffic = int(traffic_mask.sum().item())
        context_counts = (
            torch.bincount(test.context_ids[traffic_mask], minlength=5).tolist()
            if traffic
            else [0] * 5
        )
        probabilities = np.asarray(context_counts, dtype=np.float64)
        probabilities = probabilities[probabilities > 0] / probabilities.sum() if probabilities.sum() else np.asarray([])
        entropy = float(-(probabilities * np.log2(probabilities)).sum()) if probabilities.size else None
        rows.append(
            {
                **aggregate,
                "adapter_norm": float(
                    torch.linalg.vector_norm(
                        torch.cat(
                            tuple(
                                tensor.flatten()
                                for tensor in state.nodes[node_id].adapter.tensors
                            )
                        )
                    ).item()
                ),
                "context_counts": context_counts,
                "context_entropy_bits": entropy,
                "dominant_context": int(np.argmax(context_counts)) if traffic else None,
                "leaf_id": node_id if is_leaf else None,
                "leaf_traffic_fraction": traffic / total_test if is_leaf else None,
                "node_id": node_id,
                "row_type": "leaf" if is_leaf else "internal_node",
                "sample_count": _represented_count(state, node_id),
            }
        )
    return tuple(rows)


def _metric_rows(
    condition: str,
    logits: Tensor,
    table: StoredExampleTable,
    pass_config: PassConfig,
    seed: int,
    step: int,
    event: str,
) -> tuple[dict[str, object], ...]:
    losses = F.cross_entropy(logits, table.labels, reduction="none")
    predictions = logits.argmax(dim=1)
    return tuple(
        {
            "accuracy": float((predictions[mask] == table.labels[mask]).float().mean().item()),
            "condition": condition,
            "context_id": context_id,
            "event": event,
            "loss": float(losses[mask].mean().item()),
            "pass": pass_config.name,
            "seed": seed,
            "step": step,
        }
        for context_id, mask in (
            *((context_id, table.context_ids == context_id) for context_id in range(5)),
            (-1, torch.ones(table.labels.shape[0], dtype=torch.bool)),
        )
    )


def _represented_count(state: AFState, node_id: int) -> int:
    node = state.nodes[node_id]
    if node.is_leaf:
        return len(state.leaf_buffers[node_id])
    return _represented_count(state, int(node.left_id)) + _represented_count(state, int(node.right_id))


def _top_two_optimizer_config(config: VampAFConfig) -> TopTwoOptimizerConfig:
    return TopTwoOptimizerConfig(
        config.adapter.learning_rate,
        config.adapter.weight_decay,
        config.adapter.beta1,
        config.adapter.beta2,
        config.adapter.epsilon,
    )


def _train_top_two_epochs(
    trunk_features: Tensor,
    labels: Tensor,
    base: TopTwoBaseState,
    optimizer_config: TopTwoOptimizerConfig,
    batch_size: int,
    epochs: int,
    seed: int,
    device: torch.device,
    description: str,
    show_progress: bool,
) -> TopTwoAdapterState:
    try:
        from tqdm.auto import tqdm
    except ImportError as error:  # pragma: no cover - environment gate
        raise RuntimeError("tqdm is required by the vision environment") from error
    adapter = zero_top_two_adapter(base, device)
    optimizer = zero_top_two_adamw(adapter)
    fixed = zero_top_two_adapter(base, device)
    for epoch in tqdm(range(epochs), desc=description, disable=not show_progress, leave=False):
        order = np.random.default_rng(_seed(seed, epoch)).permutation(len(labels))
        for offset in range(0, len(order), batch_size):
            ids = torch.from_numpy(order[offset : offset + batch_size].astype(np.int64))
            adapter, optimizer, _loss = train_top_two_adapter_step(
                trunk_features[ids].to(device),
                labels[ids].to(device),
                base,
                fixed,
                adapter,
                optimizer,
                optimizer_config,
            )
    return adapter


def _top_two_accuracy(
    adapter: TopTwoAdapterState,
    base: TopTwoBaseState,
    trunk_features: Tensor,
    labels: Tensor,
    device: torch.device,
) -> float:
    correct = 0
    with torch.inference_mode():
        for offset in range(0, trunk_features.shape[0], 2_048):
            features = trunk_features[offset : offset + 2_048].to(device)
            predictions = top_two_logits(features, base, adapter).argmax(dim=1).cpu()
            correct += int((predictions == labels[offset : offset + len(predictions)]).sum().item())
    return correct / len(labels)


def _adapter_logits(
    adapter: TopTwoAdapterState,
    base: TopTwoBaseState,
    trunk_features: Tensor,
    device: torch.device,
) -> Tensor:
    rows = []
    with torch.inference_mode():
        for offset in range(0, trunk_features.shape[0], 2_048):
            batch = trunk_features[offset : offset + 2_048].to(device)
            rows.append(top_two_logits(batch, base, adapter).cpu())
    return torch.cat(rows)


def _context_probe_accuracy(
    train_embeddings: Tensor,
    train_context_ids: Tensor,
    test_embeddings: Tensor,
    test_context_ids: Tensor,
    optimizer_config: TopTwoOptimizerConfig,
    batch_size: int,
    epochs: int,
    seed: int,
    device: torch.device,
    show_progress: bool,
) -> float:
    try:
        from tqdm.auto import tqdm
    except ImportError as error:  # pragma: no cover - environment gate
        raise RuntimeError("tqdm is required by the vision environment") from error
    probe = torch.nn.Linear(train_embeddings.shape[1], 5).to(device)
    torch.nn.init.zeros_(probe.weight)
    torch.nn.init.zeros_(probe.bias)
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=optimizer_config.learning_rate,
        weight_decay=optimizer_config.weight_decay,
        betas=(optimizer_config.beta1, optimizer_config.beta2),
        eps=optimizer_config.epsilon,
    )
    for epoch in tqdm(
        range(epochs), desc="preflight context probe", disable=not show_progress, leave=False
    ):
        order = np.random.default_rng(_seed(seed, epoch)).permutation(len(train_context_ids))
        for offset in range(0, len(order), batch_size):
            ids = torch.from_numpy(order[offset : offset + batch_size].astype(np.int64))
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(
                probe(train_embeddings[ids].to(device)),
                train_context_ids[ids].to(device),
            )
            loss.backward()
            optimizer.step()
    correct = 0
    with torch.inference_mode():
        for offset in range(0, len(test_context_ids), 8_192):
            predictions = probe(test_embeddings[offset : offset + 8_192].to(device)).argmax(dim=1).cpu()
            correct += int((predictions == test_context_ids[offset : offset + len(predictions)]).sum().item())
    return correct / len(test_context_ids)


def _train_adapter_presentations(
    table: StoredExampleTable,
    base: TopTwoBaseState,
    presentations: int,
    hyperparameters: AFHyperparameters,
    seed: int,
    device: torch.device,
    show_progress: bool,
) -> TopTwoAdapterState:
    try:
        from tqdm.auto import tqdm
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("tqdm is required by the vision environment") from error
    adapter = zero_top_two_adapter(base, device)
    optimizer = zero_top_two_adamw(adapter)
    fixed = zero_top_two_adapter(base, device)
    consumed, epoch = 0, 0
    progress = tqdm(total=presentations, desc="joint IID presentations", disable=not show_progress, leave=False)
    while consumed < presentations:
        order = np.random.default_rng(_seed(seed, epoch)).permutation(table.embeddings.shape[0])
        for offset in range(0, len(order), hyperparameters.batch_size):
            remaining = presentations - consumed
            ids = torch.from_numpy(order[offset : offset + min(hyperparameters.batch_size, remaining)].astype(np.int64))
            adapter, optimizer, _loss = train_top_two_adapter_step(
                table.trunk_features[ids].to(device),
                table.labels[ids].to(device),
                base,
                fixed,
                adapter,
                optimizer,
                _top_two_optimizer_from_hyperparameters(hyperparameters),
            )
            consumed += len(ids)
            progress.update(len(ids))
            if consumed == presentations:
                break
        epoch += 1
    progress.close()
    return adapter


def _top_two_optimizer_from_hyperparameters(
    hyperparameters: AFHyperparameters,
) -> TopTwoOptimizerConfig:
    return TopTwoOptimizerConfig(
        hyperparameters.adapter_lr,
        hyperparameters.weight_decay,
        hyperparameters.beta1,
        hyperparameters.beta2,
        hyperparameters.epsilon,
    )


def _adapter_hyperparameters(config: VampAFConfig, batch_size: int | None = None) -> AFHyperparameters:
    return AFHyperparameters(
        leaf_capacity=config.structure.leaf_capacity,
        split_fit_samples=config.structure.split_fit_samples,
        batch_size=config.adapter.batch_size if batch_size is None else batch_size,
        adapter_lr=config.adapter.learning_rate,
        weight_decay=config.adapter.weight_decay,
        beta1=config.adapter.beta1,
        beta2=config.adapter.beta2,
        epsilon=config.adapter.epsilon,
        split_epochs=config.structure.split_epochs,
        consolidation_epochs=config.structure.consolidation_epochs,
    )


def _af_hyperparameters(config: VampAFConfig, pass_config: PassConfig) -> AFHyperparameters:
    return replace(
        _adapter_hyperparameters(config),
        leaf_capacity=pass_config.leaf_capacity,
        depth_cap_override=pass_config.depth_cap_override,
    )


def _replay(existing: Sequence[int], count: int, seed: int) -> tuple[int, ...]:
    if not existing:
        return ()
    indices = np.random.default_rng(seed).choice(len(existing), size=count, replace=len(existing) < count)
    return tuple(int(existing[int(index)]) for index in indices)


def _seed(seed: int, *parts: object) -> int:
    payload = "\0".join((str(seed), *(str(part) for part in parts))).encode("utf-8")
    return int.from_bytes(sha256(payload).digest()[:8], "big")


def _boundary_reasons(
    step: int,
    examples_per_context: int,
    total_examples: int,
    interval: int,
) -> tuple[str, ...]:
    return tuple(
        reason
        for reason, active in (
            ("interval", step % interval == 0),
            ("context_end", step % examples_per_context == 0),
            ("final", step == total_examples),
        )
        if active
    )


def _require_zero_split_parity(
    before: AFState,
    after: AFState,
    table: StoredExampleTable,
    parent_id: int,
) -> None:
    parent = after.nodes[parent_id]
    if parent.left_id is None or parent.right_id is None:
        raise RuntimeError("installed split lacks children")
    members = tuple(after.leaf_buffers[parent.left_id]) + tuple(after.leaf_buffers[parent.right_id])
    old = torch.stack(
        tuple(
            predict_for_node(before, table, parent_id, (example_id,))[0]
            for example_id in members
        )
    )
    new = torch.stack(
        tuple(
            predict_for_node(
                after,
                table,
                parent.left_id
                if float(torch.dot(parent.split_direction, table.embeddings[example_id])) <= float(parent.split_threshold)
                else parent.right_id,
                (example_id,),
            )[0]
            for example_id in members
        )
    )
    torch.testing.assert_close(new, old, rtol=0.0, atol=0.0)


def _average_accuracy(rows: Sequence[Mapping[str, object]]) -> float:
    matches = [float(row["accuracy"]) for row in rows if int(row["context_id"]) == -1]
    if len(matches) != 1:
        raise ValueError("evaluation lacks one average accuracy row")
    return matches[0]


def _work_ratio(state: AFState, step: int) -> float | None:
    denominator = step * math.log2(step + 1) if step > 0 else 0.0
    return state.counters.counted_work / denominator if denominator else None


def _ensure_final_aliases(metrics: ChainedJsonlLedger, pass_config: PassConfig) -> None:
    rows = metrics.rows
    for condition in pass_config.conditions:
        if condition == "joint_iid":
            continue
        candidates = [
            row
            for row in rows
            if row.get("condition") == condition and row.get("context_id") == -1
        ]
        if not candidates:
            raise ValueError(f"condition {condition} has no evaluation rows")
        final = max(candidates, key=lambda row: (int(row["step"]), int(row["sequence"])))
        if final["event"] == "final":
            continue
        aliases = [
            {
                key: value
                for key, value in row.items()
                if key not in {"format", "previous_sha256", "result_sha256", "sequence"}
            }
            for row in rows
            if row.get("condition") == condition
            and row.get("step") == final["step"]
            and row.get("event") == final["event"]
        ]
        metrics.append_many(tuple({**row, "event": "final"} for row in aliases))


def _save_run_checkpoint(path: Path, checkpoint: RunCheckpoint) -> None:
    atomic_torch_save(path, {"checkpoint": checkpoint, "schema_version": "vamp-af-run-checkpoint-v2"})


def _write_resolved_yaml(path: Path, record: Mapping[str, object]) -> None:
    try:
        import yaml
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("PyYAML is required by the vision environment") from error
    atomic_write(path, yaml.safe_dump(dict(record), sort_keys=True).encode("utf-8"))


def _write_blocked_handoff(run_root: Path, preflight: Mapping[str, object]) -> None:
    atomic_write(
        run_root / "HANDOFF.md",
        (
            "# VAMP-AF blocked at representation preflight\n\n"
            "AF was not run because a required frozen-feature gate failed. No AF heuristic was changed.\n\n"
            f"```json\n{canonical_json_bytes(dict(preflight)).decode('utf-8')}```\n"
        ).encode("utf-8"),
    )


def _material_source_hashes() -> dict[str, str]:
    project_root = Path(__file__).resolve().parents[3]
    paths = (
        project_root / "src/apm/continual/addressing_first.py",
        project_root / "src/apm/continual/top_two_adapter.py",
        project_root / "src/apm/experiments/vamp_af_config.py",
        project_root / "src/apm/experiments/vamp_af_data.py",
        project_root / "src/apm/experiments/vamp_af_mnist.py",
        project_root / "src/apm/experiments/vamp_af_reporting.py",
        project_root / "src/apm/experiments/vamp_af_workflow.py",
        project_root / "configs/vamp_af_mnist/poc.yaml",
    )
    return {path.relative_to(project_root).as_posix(): file_sha256(path) for path in paths}


__all__ = [
    "BaselineState",
    "EvaluationResult",
    "OracleState",
    "PassSeedResult",
    "RunCheckpoint",
    "run_pass_seed",
    "run_preflight",
    "run_workflow",
]

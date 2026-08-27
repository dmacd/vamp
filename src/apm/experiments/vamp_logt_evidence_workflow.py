"""Resumable calibration, static, consolidation, and online NCE/TRE workflow."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import os
import time
from collections.abc import Mapping

import numpy as np
import torch
from pyrsistent import PMap, pmap

from apm.continual.artifacts import (
    ChainedJsonlLedger,
    atomic_write,
    file_sha256,
    load_canonical_json,
    publish_immutable_json,
)
from apm.continual.logt_evidence_bank import (
    EvidenceWorkCounters,
    TemporalNode,
    insert_block,
    require_evidence_work_bound,
)
from apm.continual.nce_tre_evidence import (
    FrozenEvidenceState,
    freeze_evidence_model,
    materialize_evidence_model,
)
from apm.continual.top_two_adapter import TopTwoAdapterState
from apm.experiments.vamp_logt_evidence_config import VampLogTEvidenceConfig
from apm.experiments.vamp_logt_evidence_data import (
    AuthenticatedBaseline,
    NodeHoldout,
    RawFeatureTable,
    authenticate_and_load_baseline,
    build_node_holdout,
    node_context_key,
    resolved_device,
    stream_training_table,
)
from apm.experiments.vamp_logt_evidence_state import (
    ActiveEvidenceBank,
    StoredAdapterResult,
    StoredEvidenceResult,
    adapter_artifact_path,
    empty_active_bank,
    evidence_artifact_path,
    load_adapter_result,
    load_bank_checkpoint,
    load_evidence_result,
    publish_adapter_result,
    publish_evidence_result,
    retire_inactive_node_artifacts,
    save_bank_checkpoint,
)
from apm.experiments.vamp_logt_evidence_training import (
    bridge_diagnostics,
    evaluate_routing,
    evidence_training_config,
    protocol_seed,
    score_evidence_bank,
    train_node_adapter,
    train_node_evidence,
)
from apm.experiments.vamp_logt_ratio_calibration import run_ratio_calibration


STATIC_CONDITION_DEFINITION = {
    "direct_nce": (
        "Direct NCE gives each active temporal node one full-capacity raw-image "
        "classifier between its lightly corrupted replay distribution and the shared "
        "configured raw-image reference, so K equals one."
    ),
    "tre": (
        "TRE gives each active temporal node one full-capacity bridge-conditioned raw-image "
        "network and sums all adjacent-waymark logits before choosing the largest node score."
    ),
    "oracle_node": (
        "The oracle node is a diagnostic that uses the true digit target to choose the active "
        "adapter with the smallest per-example classification loss; it is not deployable."
    ),
}


def run_workflow(config: VampLogTEvidenceConfig) -> Path:
    """Run or resume the one phase-gated NCE/TRE LogT MNIST experiment."""
    if config.runtime.deterministic_algorithms:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    device = resolved_device(config.runtime.device)
    run_root = config.artifact_root / "runs" / config.config_hash
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"NCE/TRE durable working directory: {run_root}", flush=True)
    _write_resolved_yaml(run_root / "config_resolved.yaml", config.as_record())

    print("Phase 1/6: authenticate sealed VAMP-AF data and base", flush=True)
    baseline = authenticate_and_load_baseline(config, run_root)
    publish_immutable_json(
        run_root / "protocol.json",
        {
            "baseline_protocol_sha256": baseline.protocol_sha256,
            "config": config.as_record(),
            "config_hash": config.config_hash,
            "evidence_reference": {
                "content_sha256": baseline.reference_sha256,
                "examples": (
                    None
                    if baseline.reference_raw_images is None
                    else len(baseline.reference_raw_images)
                ),
                "kind": config.evidence.reference,
            },
            "material_source_sha256": _material_source_hashes(),
            "schema_version": "vamp-logt-nce-tre-protocol-v2",
            "torch_version": torch.__version__,
        },
    )

    print("Phase 2/6: normalized multimodal ratio calibration", flush=True)
    calibration = run_ratio_calibration(
        config.calibration,
        run_root / "calibration",
        device,
        config.runtime.progress,
    )
    if not bool(calibration["passed"]):
        return _finish_blocked_report(run_root, config, baseline, {"calibration": calibration})

    print("Phase 3/6: fixed 63-block static LogT routing test", flush=True)
    static = run_static_phase(config, baseline, run_root, device)
    if not bool(static["passed"]):
        return _finish_blocked_report(
            run_root,
            config,
            baseline,
            {"calibration": calibration, "static": static},
        )

    selected_bridges = int(static["selected_tre_bridges"])
    print(
        f"Phase 4/6: block-64 consolidation stability with frozen K={selected_bridges}",
        flush=True,
    )
    consolidation = run_consolidation_phase(
        config,
        baseline,
        run_root,
        device,
        selected_bridges,
    )
    if not bool(consolidation["passed"]):
        return _finish_blocked_report(
            run_root,
            config,
            baseline,
            {
                "calibration": calibration,
                "static": static,
                "consolidation": consolidation,
            },
        )

    print("Phase 5/6: complete 100-block online direct/TRE/oracle comparison", flush=True)
    online = run_online_phase(
        config,
        baseline,
        run_root,
        device,
        selected_bridges,
    )
    print("Phase 6/6: plain-language Markdown/HTML result report", flush=True)
    from apm.experiments.vamp_logt_evidence_reporting import write_result_report

    write_result_report(
        run_root,
        config,
        baseline.summary,
        {
            "calibration": calibration,
            "static": static,
            "consolidation": consolidation,
            "online": online,
        },
    )
    return run_root


def run_static_phase(
    config: VampLogTEvidenceConfig,
    baseline: AuthenticatedBaseline,
    run_root: Path,
    device: torch.device,
) -> dict[str, object]:
    """Build three genuine LogT snapshots, fit all K candidates, and select once."""
    phase_root = run_root / "static"
    summary_path = phase_root / "summary.json"
    if summary_path.is_file():
        summary = load_canonical_json(summary_path)
        if summary.get("schema_version") != "vamp-logt-static-routing-v1":
            raise ValueError("static routing schema changed inside one run identity")
        return summary
    phase_root.mkdir(parents=True, exist_ok=True)
    ledger = ChainedJsonlLedger(phase_root / "metrics.jsonl", "vamp-logt-static-metric-v1")
    existing_keys = {
        (int(row["stream_seed"]), str(row["condition"]), int(row["replica"]))
        for row in ledger.rows
    }
    rows: list[dict[str, object]] = []
    route_outputs: dict[tuple[int, int, int], tuple[str, ...]] = {}
    score_outputs: dict[tuple[int, int, int], torch.Tensor] = {}
    for stream_seed in config.static.stream_seeds:
        stream = stream_training_table(baseline, stream_seed)
        seed_root = phase_root / f"seed-{stream_seed}"
        bank = _build_bank_to_blocks(
            config,
            stream,
            baseline,
            stream_seed,
            seed_root,
            device,
            config.stream.static_snapshot_blocks,
            {},
            "static",
        )
        specifications = _static_condition_specifications(config)
        bank = _attach_evidence_to_active_nodes(
            config,
            bank,
            stream,
            baseline,
            stream_seed,
            seed_root,
            device,
            specifications,
            "static",
        )
        holdout = build_node_holdout(
            bank.topology.active_nodes,
            stream,
            baseline.test,
            config.static.heldout_examples_per_node,
            protocol_seed(stream_seed, "static", "holdout"),
        )
        equivalence = {
            node.node_id: node_context_key(node, stream)
            for node in bank.topology.active_nodes
        }
        try:
            from tqdm.auto import tqdm
        except ImportError as error:  # pragma: no cover - vision environment gate
            raise RuntimeError("tqdm is required by the vision environment") from error
        for condition, bridges in tqdm(
            tuple(specifications.items()),
            desc=f"static seed {stream_seed} route evaluations",
            disable=not config.runtime.progress,
            unit="condition",
        ):
            replica = int(condition.rsplit("r", 1)[1])
            models = {
                node_id: materialize_evidence_model(state)
                for node_id, state in bank.evidence_by_condition[condition].items()
            }
            scores = score_evidence_bank(
                bank.topology.active_nodes,
                models,
                holdout.table.raw_images,
                device,
                config.evidence.score_batch_size,
            )
            evaluation = evaluate_routing(
                bank.topology.active_nodes,
                bank.adapters,
                scores,
                holdout.table,
                baseline.top_two_base,
                device,
                config.online.evaluation_batch_size,
                holdout,
                equivalence,
            )
            maximum_adjacent_accuracy, mean_adjacent_loss = _static_bridge_metrics(
                config,
                baseline,
                bank.topology.active_nodes,
                models,
                holdout,
                bridges,
                stream_seed,
                device,
            )
            family = "direct_nce" if bridges == 1 else "tre"
            row = {
                "bridges": bridges,
                "condition": condition,
                "condition_family": family,
                "maximum_adjacent_balanced_accuracy": maximum_adjacent_accuracy,
                "mean_adjacent_balanced_loss": mean_adjacent_loss,
                "replica": replica,
                "stream_seed": stream_seed,
                **evaluation.as_record(),
            }
            rows.append(row)
            route_outputs[(stream_seed, bridges, replica)] = evaluation.routed_node_ids
            score_outputs[(stream_seed, bridges, replica)] = scores.scores
            key = (stream_seed, condition, replica)
            if key not in existing_keys:
                ledger.append(row)
                existing_keys.add(key)
        expected_route_evaluations = (
            len(holdout.table.labels)
            * len(bank.topology.active_nodes)
            * len(specifications)
        )
        if bank.counters.evidence_route_model_evals > expected_route_evaluations:
            raise ValueError("static checkpoint contains excess routing work")
        if bank.counters.evidence_route_model_evals < expected_route_evaluations:
            bank = ActiveEvidenceBank(
                bank.topology,
                bank.adapters,
                bank.evidence_by_condition,
                bank.counters.with_routing(
                    expected_route_evaluations
                    - bank.counters.evidence_route_model_evals,
                    len(bank.topology.active_nodes) * len(specifications),
                ),
                bank.adapter_example_updates,
            )
            save_bank_checkpoint(seed_root / "state" / "checkpoint.pt", bank)
        publish_immutable_json(
            seed_root / "snapshot.json",
            {
                "active_intervals": [
                    {
                        "example_count": len(node.example_ids),
                        "first_block": node.first_block,
                        "last_block": node.last_block,
                        "level": node.level,
                        "node_id": node.node_id,
                    }
                    for node in bank.topology.active_nodes
                ],
                "schema_version": "vamp-logt-static-snapshot-v1",
                "stream_seed": stream_seed,
                "work_counters": asdict(bank.counters),
            },
        )
    candidate_rows = tuple(
        _static_candidate_result(
            bridges,
            config,
            rows,
            route_outputs,
            score_outputs,
        )
        for bridges in config.evidence.candidate_tre_bridges
    )
    passing = tuple(int(row["bridges"]) for row in candidate_rows if row["passed"])
    selected = min(passing) if passing else None
    summary: dict[str, object] = {
        "candidate_results": list(candidate_rows),
        "condition_definitions": STATIC_CONDITION_DEFINITION,
        "direct_nce_results": [row for row in rows if row["condition_family"] == "direct_nce"],
        "evidence_reference": config.evidence.reference,
        "passed": selected is not None,
        "schema_version": "vamp-logt-static-routing-v1",
        "selected_tre_bridges": selected,
        "snapshot_blocks": config.stream.static_snapshot_blocks,
        "tre_results": [row for row in rows if row["condition_family"] == "tre"],
    }
    publish_immutable_json(summary_path, summary)
    return summary


def run_consolidation_phase(
    config: VampLogTEvidenceConfig,
    baseline: AuthenticatedBaseline,
    run_root: Path,
    device: torch.device,
    selected_bridges: int,
) -> dict[str, object]:
    """Execute the block-64 carry and compare every fresh parent with a de-novo twin."""
    phase_root = run_root / "consolidation"
    summary_path = phase_root / "summary.json"
    if summary_path.is_file():
        summary = load_canonical_json(summary_path)
        if summary.get("schema_version") != "vamp-logt-consolidation-v1":
            raise ValueError("consolidation schema changed inside one run identity")
        return summary
    phase_root.mkdir(parents=True, exist_ok=True)
    seed_summaries = tuple(
        _run_consolidation_seed(
            config,
            baseline,
            run_root,
            phase_root,
            device,
            selected_bridges,
            stream_seed,
        )
        for stream_seed in config.static.stream_seeds
    )
    gates = {
        "classifier_accuracy_is_reproduced": all(
            bool(summary["gates"]["classifier_accuracy_is_reproduced"])
            for summary in seed_summaries
        ),
        "evidence_loss_is_reproduced": all(
            bool(summary["gates"]["evidence_loss_is_reproduced"])
            for summary in seed_summaries
        ),
        "raw_scores_are_stable": all(
            bool(summary["gates"]["raw_scores_are_stable"])
            for summary in seed_summaries
        ),
        "route_decisions_are_stable": all(
            bool(summary["gates"]["route_decisions_are_stable"])
            for summary in seed_summaries
        ),
        "score_offset_has_no_level_slope": all(
            bool(summary["gates"]["score_offset_has_no_level_slope"])
            for summary in seed_summaries
        ),
    }
    summary: dict[str, object] = {
        "condition_definitions": {
            "normal_consolidation": (
                "Normal consolidation trains a fresh full-capacity adapter and a fresh "
                "full-capacity TRE model on each exact child-union replay during the carry."
            ),
            "independent_de_novo_control": (
                "The independent de-novo control trains a second fresh TRE model with a "
                "different initialization on exactly the same union replay and fixed schedule."
            ),
        },
        "gates": gates,
        "passed": all(gates.values()),
        "schema_version": "vamp-logt-consolidation-v1",
        "seed_results": list(seed_summaries),
        "selected_tre_bridges": selected_bridges,
    }
    publish_immutable_json(summary_path, summary)
    return summary


def run_online_phase(
    config: VampLogTEvidenceConfig,
    baseline: AuthenticatedBaseline,
    run_root: Path,
    device: torch.device,
    selected_bridges: int,
) -> dict[str, object]:
    """Run direct NCE and frozen-schedule TRE on one shared adapter bank for 100 blocks."""
    phase_root = run_root / "online"
    summary_path = phase_root / "summary.json"
    if summary_path.is_file():
        summary = load_canonical_json(summary_path)
        if summary.get("schema_version") != "vamp-logt-online-v1":
            raise ValueError("online comparison schema changed inside one run identity")
        return summary
    phase_root.mkdir(parents=True, exist_ok=True)
    seed_results = tuple(
        _run_online_seed(
            config,
            baseline,
            phase_root,
            device,
            selected_bridges,
            stream_seed,
        )
        for stream_seed in config.online.stream_seeds
    )
    final_rows = [
        row
        for result in seed_results
        for row in result["evaluations"]
        if int(row["processed_blocks"]) == config.stream.total_blocks
    ]
    condition_means = {
        condition: float(
            np.mean(
                [
                    float(row["accuracy"])
                    for row in final_rows
                    if row["condition"] == condition
                ]
            )
        )
        for condition in ("direct_nce", "tre", "oracle_node")
    }
    summary: dict[str, object] = {
        "condition_definitions": STATIC_CONDITION_DEFINITION,
        "final_mean_accuracy": condition_means,
        "recorded_vamp_af_baseline": baseline.summary,
        "schema_version": "vamp-logt-online-v1",
        "seed_results": list(seed_results),
        "selected_tre_bridges": selected_bridges,
    }
    publish_immutable_json(summary_path, summary)
    return summary


def _build_bank_to_blocks(
    config: VampLogTEvidenceConfig,
    stream: RawFeatureTable,
    baseline: AuthenticatedBaseline,
    stream_seed: int,
    directory: Path,
    device: torch.device,
    target_blocks: int,
    evidence_specifications: Mapping[str, int],
    purpose: str,
) -> ActiveEvidenceBank:
    checkpoint_path = directory / "state" / "checkpoint.pt"
    nodes_root = directory / "nodes"
    bank = (
        load_bank_checkpoint(checkpoint_path)
        if checkpoint_path.is_file()
        else empty_active_bank(config.stream.block_size)
    )
    if (
        set(bank.evidence_by_condition) != set(evidence_specifications)
        and not (
            bank.topology.processed_blocks == target_blocks
            and not evidence_specifications
        )
        and bank.topology.processed_blocks
    ):
        raise ValueError("resumed bank evidence families differ from the requested build")
    try:
        from tqdm.auto import tqdm
    except ImportError as error:  # pragma: no cover - vision environment gate
        raise RuntimeError("tqdm is required by the vision environment") from error
    for _block in tqdm(
        range(bank.topology.processed_blocks, target_blocks),
        initial=bank.topology.processed_blocks,
        total=target_blocks,
        desc=f"{purpose} seed {stream_seed} LogT blocks",
        disable=not config.runtime.progress,
        unit="block",
    ):
        bank = _advance_one_block(
            config,
            bank,
            stream,
            baseline,
            stream_seed,
            directory,
            device,
            evidence_specifications,
            purpose,
        )
    retire_inactive_node_artifacts(
        nodes_root,
        {node.node_id for node in bank.topology.active_nodes},
    )
    return bank


def _advance_one_block(
    config: VampLogTEvidenceConfig,
    bank: ActiveEvidenceBank,
    stream: RawFeatureTable,
    baseline: AuthenticatedBaseline,
    stream_seed: int,
    directory: Path,
    device: torch.device,
    evidence_specifications: Mapping[str, int],
    purpose: str,
) -> ActiveEvidenceBank:
    first_example = bank.topology.processed_blocks * config.stream.block_size
    block_ids = tuple(range(first_example, first_example + config.stream.block_size))
    topology, leaf, merges = insert_block(bank.topology, block_ids)
    nodes_root = directory / "nodes"
    adapters = bank.adapters
    evidence_maps = bank.evidence_by_condition
    counters = bank.counters
    adapter_updates = bank.adapter_example_updates
    created = ((leaf, False), *((merge.parent, True) for merge in merges))
    for node, is_merge in created:
        adapter_result = _fit_or_load_adapter(
            config,
            node,
            stream,
            baseline,
            stream_seed,
            nodes_root,
            device,
            purpose,
        )
        if is_merge:
            parent_ids = node.parent_node_ids
            adapters = adapters.remove(parent_ids[0]).remove(parent_ids[1])
        adapters = adapters.set(node.node_id, adapter_result.adapter)
        adapter_updates += adapter_result.example_updates
        for condition, bridges in evidence_specifications.items():
            evidence_result = _fit_or_load_evidence(
                config,
                node,
                stream,
                baseline,
                stream_seed,
                nodes_root,
                device,
                condition,
                bridges,
                purpose,
            )
            models = evidence_maps.get(condition, pmap())
            if is_merge:
                models = models.remove(node.parent_node_ids[0]).remove(node.parent_node_ids[1])
            evidence_maps = evidence_maps.set(condition, models.set(node.node_id, evidence_result.state))
            counters = counters.with_training(evidence_result.example_updates, merge=is_merge)
    counters = counters.with_routing(
        0,
        len(topology.active_nodes) * len(evidence_specifications),
    )
    committed = ActiveEvidenceBank(
        topology,
        adapters,
        evidence_maps,
        counters,
        adapter_updates,
    )
    if evidence_specifications:
        require_evidence_work_bound(
            committed.counters,
            topology.processed_blocks,
            config.stream.block_size,
            config.evidence.epochs,
            len(evidence_specifications),
        )
    save_bank_checkpoint(directory / "state" / "checkpoint.pt", committed)
    retire_inactive_node_artifacts(
        nodes_root,
        {node.node_id for node in topology.active_nodes},
    )
    return committed


def _attach_evidence_to_active_nodes(
    config: VampLogTEvidenceConfig,
    bank: ActiveEvidenceBank,
    stream: RawFeatureTable,
    baseline: AuthenticatedBaseline,
    stream_seed: int,
    directory: Path,
    device: torch.device,
    specifications: Mapping[str, int],
    purpose: str,
) -> ActiveEvidenceBank:
    unknown = set(bank.evidence_by_condition) - set(specifications)
    if unknown:
        raise ValueError(f"static checkpoint contains undeclared evidence conditions: {unknown}")
    try:
        from tqdm.auto import tqdm
    except ImportError as error:  # pragma: no cover - vision environment gate
        raise RuntimeError("tqdm is required by the vision environment") from error
    completed_conditions = len(bank.evidence_by_condition)
    pending = tuple(
        (condition, bridges)
        for condition, bridges in specifications.items()
        if condition not in bank.evidence_by_condition
    )
    for condition, bridges in tqdm(
        pending,
        initial=completed_conditions,
        total=len(specifications),
        desc=f"{purpose} seed {stream_seed} evidence families",
        disable=not config.runtime.progress,
        unit="family",
    ):
        models: PMap[str, FrozenEvidenceState] = pmap()
        counters = bank.counters
        for node in bank.topology.active_nodes:
            result = _fit_or_load_evidence(
                config,
                node,
                stream,
                baseline,
                stream_seed,
                directory / "nodes",
                device,
                condition,
                bridges,
                purpose,
            )
            models = models.set(node.node_id, result.state)
            counters = counters.with_training(result.example_updates, merge=node.level > 0)
        evidence_maps = bank.evidence_by_condition.set(condition, models)
        counters = counters.with_routing(
            0,
            len(bank.topology.active_nodes) * len(evidence_maps),
        )
        bank = ActiveEvidenceBank(
            bank.topology,
            bank.adapters,
            evidence_maps,
            counters,
            bank.adapter_example_updates,
        )
        require_evidence_work_bound(
            bank.counters,
            bank.topology.processed_blocks,
            config.stream.block_size,
            config.evidence.epochs,
            len(specifications),
        )
        save_bank_checkpoint(directory / "state" / "checkpoint.pt", bank)
    return bank


def _fit_or_load_adapter(
    config: VampLogTEvidenceConfig,
    node: TemporalNode,
    stream: RawFeatureTable,
    baseline: AuthenticatedBaseline,
    stream_seed: int,
    nodes_root: Path,
    device: torch.device,
    purpose: str,
) -> StoredAdapterResult:
    seed = protocol_seed(stream_seed, purpose, "adapter", node.node_id)
    path = adapter_artifact_path(nodes_root, node.node_id)
    if path.is_file():
        return load_adapter_result(path, node.node_id, seed)
    ids = torch.tensor(node.example_ids, dtype=torch.int64)
    trained = train_node_adapter(
        stream.trunk_features[ids],
        stream.labels[ids],
        baseline.top_two_base,
        config.adapter,
        seed,
        device,
        f"adapter L{node.level} n={len(ids)}",
        config.runtime.progress,
    )
    return publish_adapter_result(
        path,
        node.node_id,
        seed,
        trained.adapter,
        trained.final_loss,
        trained.example_updates,
    )


def _fit_or_load_evidence(
    config: VampLogTEvidenceConfig,
    node: TemporalNode,
    stream: RawFeatureTable,
    baseline: AuthenticatedBaseline,
    stream_seed: int,
    nodes_root: Path,
    device: torch.device,
    condition: str,
    bridges: int,
    purpose: str,
) -> StoredEvidenceResult:
    seed = protocol_seed(stream_seed, purpose, "evidence", condition, node.node_id)
    path = evidence_artifact_path(nodes_root, node.node_id, condition)
    if path.is_file():
        loaded = load_evidence_result(path, node.node_id, condition, seed)
        if loaded.state.bridges != bridges:
            raise ValueError("stored evidence bridge count differs from the frozen condition")
        return loaded
    ids = torch.tensor(node.example_ids, dtype=torch.int64)
    trained = train_node_evidence(
        stream.raw_images[ids],
        baseline.reference_raw_images,
        config.evidence,
        bridges,
        seed,
        device,
        config.runtime.progress,
    )
    return publish_evidence_result(
        path,
        node.node_id,
        condition,
        seed,
        freeze_evidence_model(trained.model),
        trained.final_loss,
        trained.source_example_updates,
        trained.reference_examples,
    )


def _static_condition_specifications(config: VampLogTEvidenceConfig) -> dict[str, int]:
    return {
        **{
            f"direct-r{replica}": config.evidence.direct_bridges
            for replica in range(config.evidence.independent_replicas)
        },
        **{
            f"tre-k{bridges}-r{replica}": bridges
            for bridges in config.evidence.candidate_tre_bridges
            for replica in range(config.evidence.independent_replicas)
        },
    }


def _static_bridge_metrics(
    config: VampLogTEvidenceConfig,
    baseline: AuthenticatedBaseline,
    nodes: tuple[TemporalNode, ...],
    models: Mapping[str, torch.nn.Module],
    holdout: NodeHoldout,
    bridges: int,
    stream_seed: int,
    device: torch.device,
) -> tuple[float, float]:
    diagnostics = tuple(
        diagnostic
        for node_index, node in enumerate(nodes)
        for diagnostic in bridge_diagnostics(
            models[node.node_id],
            holdout.table.raw_images[
                node_index
                * config.static.heldout_examples_per_node : (node_index + 1)
                * config.static.heldout_examples_per_node
            ],
            baseline.reference_raw_images,
            evidence_training_config(config.evidence, bridges),
            protocol_seed(stream_seed, "static", "diagnostic", node.node_id, bridges),
            device,
        )
    )
    return (
        max(row.balanced_accuracy for row in diagnostics),
        float(np.mean([row.balanced_loss for row in diagnostics])),
    )


def _static_candidate_result(
    bridges: int,
    config: VampLogTEvidenceConfig,
    rows: list[dict[str, object]],
    routes: Mapping[tuple[int, int, int], tuple[str, ...]],
    scores: Mapping[tuple[int, int, int], torch.Tensor],
) -> dict[str, object]:
    selected_rows = [row for row in rows if int(row["bridges"]) == bridges]
    agreements = [
        float(
            np.mean(
                [left == right for left, right in zip(routes[(seed, bridges, first)], routes[(seed, bridges, second)])]
            )
        )
        for seed in config.static.stream_seeds
        for first in range(config.evidence.independent_replicas)
        for second in range(first + 1, config.evidence.independent_replicas)
    ]
    score_rmse = [
        float(torch.sqrt(torch.mean((scores[(seed, bridges, first)] - scores[(seed, bridges, second)]).square())).item())
        for seed in config.static.stream_seeds
        for first in range(config.evidence.independent_replicas)
        for second in range(first + 1, config.evidence.independent_replicas)
    ]
    maximum_adjacent = max(float(row["maximum_adjacent_balanced_accuracy"]) for row in selected_rows)
    minimum_agreement = min(agreements)
    maximum_oracle_gap = max(
        float(row["oracle_accuracy"]) - float(row["routed_accuracy"])
        for row in selected_rows
    )
    gates = {
        "adjacent_waymarks_retain_overlap": maximum_adjacent
        <= config.static.adjacent_balanced_accuracy_max,
        "independent_routes_agree": minimum_agreement
        >= config.static.independent_route_agreement_min,
        "routed_classifier_is_within_ten_points_of_oracle_on_every_seed_and_replica": maximum_oracle_gap
        <= config.static.classifier_oracle_gap_max,
    }
    return {
        "bridges": bridges,
        "gates": gates,
        "maximum_adjacent_balanced_accuracy": maximum_adjacent,
        "maximum_classifier_oracle_gap": maximum_oracle_gap,
        "maximum_interreplica_score_rmse_nats": max(score_rmse),
        "minimum_independent_route_agreement": minimum_agreement,
        "passed": all(gates.values()),
    }


def _run_consolidation_seed(
    config: VampLogTEvidenceConfig,
    baseline: AuthenticatedBaseline,
    run_root: Path,
    phase_root: Path,
    device: torch.device,
    selected_bridges: int,
    stream_seed: int,
) -> dict[str, object]:
    seed_root = phase_root / f"seed-{stream_seed}"
    summary_path = seed_root / "summary.json"
    if summary_path.is_file():
        summary = load_canonical_json(summary_path)
        checkpoint_path = seed_root / "state" / "checkpoint.pt"
        if checkpoint_path.is_file():
            completed_bank = load_bank_checkpoint(checkpoint_path)
            retire_inactive_node_artifacts(
                seed_root / "nodes",
                {node.node_id for node in completed_bank.topology.active_nodes},
            )
        return summary
    stream = stream_training_table(baseline, stream_seed)
    static_bank = load_bank_checkpoint(
        run_root / "static" / f"seed-{stream_seed}" / "state" / "checkpoint.pt"
    )
    selected_condition = f"tre-k{selected_bridges}-r0"
    selected_states = static_bank.evidence_by_condition[selected_condition]
    initial_updates = sum(config.evidence.epochs * len(node.example_ids) for node in static_bank.topology.active_nodes)
    initial_leaf_updates = sum(
        config.evidence.epochs * len(node.example_ids)
        for node in static_bank.topology.active_nodes
        if node.level == 0
    )
    bank = ActiveEvidenceBank(
        static_bank.topology,
        static_bank.adapters,
        pmap({"tre": selected_states}),
        EvidenceWorkCounters(
            initial_leaf_updates,
            initial_updates - initial_leaf_updates,
            initial_updates,
            0,
            len(selected_states),
        ),
        static_bank.adapter_example_updates,
    )
    next_ids = tuple(
        range(
            bank.topology.processed_blocks * config.stream.block_size,
            (bank.topology.processed_blocks + 1) * config.stream.block_size,
        )
    )
    topology, leaf, merges = insert_block(bank.topology, next_ids)
    nodes_root = seed_root / "nodes"
    adapters = bank.adapters
    evidence = selected_states
    counters = bank.counters
    adapter_updates = bank.adapter_example_updates
    leaf_adapter = _fit_or_load_adapter(
        config, leaf, stream, baseline, stream_seed, nodes_root, device, "consolidation"
    )
    leaf_evidence = _fit_or_load_evidence(
        config,
        leaf,
        stream,
        baseline,
        stream_seed,
        nodes_root,
        device,
        "tre",
        selected_bridges,
        "consolidation",
    )
    adapters = adapters.set(leaf.node_id, leaf_adapter.adapter)
    evidence = evidence.set(leaf.node_id, leaf_evidence.state)
    adapter_updates += leaf_adapter.example_updates
    counters = counters.with_training(leaf_evidence.example_updates, merge=False)
    remaining = bank.topology.active_by_level
    current = leaf
    metric_rows = []
    route_model_evaluations = 0
    ledger = ChainedJsonlLedger(seed_root / "metrics.jsonl", "vamp-logt-consolidation-metric-v1")
    existing_levels = {int(row["level"]) for row in ledger.rows}
    for merge in merges:
        normal_adapter = _fit_or_load_adapter(
            config,
            merge.parent,
            stream,
            baseline,
            stream_seed,
            nodes_root,
            device,
            "consolidation",
        )
        normal_evidence = _fit_or_load_evidence(
            config,
            merge.parent,
            stream,
            baseline,
            stream_seed,
            nodes_root,
            device,
            "tre",
            selected_bridges,
            "consolidation",
        )
        control_evidence = _fit_or_load_evidence(
            config,
            merge.parent,
            stream,
            baseline,
            stream_seed,
            seed_root / "controls",
            device,
            "independent-control",
            selected_bridges,
            "consolidation-control",
        )
        counters = counters.with_training(control_evidence.example_updates, merge=True)
        adapters = adapters.remove(merge.left.node_id).remove(merge.right.node_id).set(
            merge.parent.node_id, normal_adapter.adapter
        )
        evidence = evidence.remove(merge.left.node_id).remove(merge.right.node_id).set(
            merge.parent.node_id, normal_evidence.state
        )
        adapter_updates += normal_adapter.example_updates
        counters = counters.with_training(normal_evidence.example_updates, merge=True)
        remaining = remaining.remove(merge.left.level)
        current = merge.parent
        frontier = tuple(
            sorted(
                (*remaining.values(), current),
                key=lambda node: node.first_block,
            )
        )
        frontier_adapters = pmap(
            {node.node_id: adapters[node.node_id] for node in frontier}
        )
        normal_states = pmap({node.node_id: evidence[node.node_id] for node in frontier})
        control_states = normal_states.set(merge.parent.node_id, control_evidence.state)
        holdout = build_node_holdout(
            frontier,
            stream,
            baseline.test,
            config.consolidation.heldout_examples_per_merge,
            protocol_seed(stream_seed, "consolidation", "holdout", merge.parent.level),
        )
        equivalence = {node.node_id: node_context_key(node, stream) for node in frontier}
        normal_scores, normal_eval = _score_and_evaluate_states(
            frontier,
            frontier_adapters,
            normal_states,
            holdout,
            equivalence,
            baseline,
            config,
            device,
        )
        control_scores, control_eval = _score_and_evaluate_states(
            frontier,
            frontier_adapters,
            control_states,
            holdout,
            equivalence,
            baseline,
            config,
            device,
        )
        route_model_evaluations += 2 * len(holdout.table.labels) * len(frontier)
        parent_column = normal_scores.node_ids.index(merge.parent.node_id)
        normal_model = materialize_evidence_model(normal_evidence.state)
        control_model = materialize_evidence_model(control_evidence.state)
        parent_index = frontier.index(merge.parent)
        parent_raw = holdout.table.raw_images[
            parent_index
            * config.consolidation.heldout_examples_per_merge : (parent_index + 1)
            * config.consolidation.heldout_examples_per_merge
        ]
        normal_diagnostics = bridge_diagnostics(
            normal_model,
            parent_raw,
            baseline.reference_raw_images,
            evidence_training_config(config.evidence, selected_bridges),
            protocol_seed(stream_seed, "consolidation", "paired-diagnostic", merge.parent.level),
            device,
        )
        control_diagnostics = bridge_diagnostics(
            control_model,
            parent_raw,
            baseline.reference_raw_images,
            evidence_training_config(config.evidence, selected_bridges),
            protocol_seed(stream_seed, "consolidation", "paired-diagnostic", merge.parent.level),
            device,
        )
        normal_loss = float(np.mean([row.balanced_loss for row in normal_diagnostics]))
        control_loss = float(np.mean([row.balanced_loss for row in control_diagnostics]))
        differences = normal_scores.scores[:, parent_column] - control_scores.scores[:, parent_column]
        row = {
            "classifier_accuracy_gap": abs(normal_eval.routed_accuracy - control_eval.routed_accuracy),
            "independent_control_accuracy": control_eval.routed_accuracy,
            "level": merge.parent.level,
            "mean_absolute_raw_score_difference_nats": float(differences.abs().mean().item()),
            "mean_signed_raw_score_difference_nats": float(differences.mean().item()),
            "nce_loss_relative_difference": abs(normal_loss - control_loss)
            / max((normal_loss + control_loss) / 2.0, 1.0e-12),
            "normal_accuracy": normal_eval.routed_accuracy,
            "route_agreement": float(
                np.mean(
                    [
                        left == right
                        for left, right in zip(
                            normal_eval.routed_node_ids, control_eval.routed_node_ids
                        )
                    ]
                )
            ),
            "stream_seed": stream_seed,
        }
        metric_rows.append(row)
        if merge.parent.level not in existing_levels:
            ledger.append(row)
            existing_levels.add(merge.parent.level)
    remaining = remaining.set(current.level, current)
    counters = counters.with_routing(route_model_evaluations, len(topology.active_nodes))
    final_bank = ActiveEvidenceBank(
        topology,
        adapters,
        pmap({"tre": evidence}),
        counters,
        adapter_updates,
    )
    require_evidence_work_bound(
        final_bank.counters,
        topology.processed_blocks,
        config.stream.block_size,
        config.evidence.epochs,
        2,
    )
    save_bank_checkpoint(seed_root / "state" / "checkpoint.pt", final_bank)
    levels = np.asarray([int(row["level"]) for row in metric_rows], dtype=np.float64)
    offsets = np.asarray(
        [float(row["mean_signed_raw_score_difference_nats"]) for row in metric_rows],
        dtype=np.float64,
    )
    slope = float(np.polyfit(levels, offsets, 1)[0]) if len(levels) > 1 else 0.0
    gates = {
        "classifier_accuracy_is_reproduced": max(
            float(row["classifier_accuracy_gap"]) for row in metric_rows
        )
        <= config.consolidation.classifier_accuracy_gap_max,
        "evidence_loss_is_reproduced": max(
            float(row["nce_loss_relative_difference"]) for row in metric_rows
        )
        <= config.consolidation.nce_loss_relative_difference_max,
        "raw_scores_are_stable": max(
            float(row["mean_absolute_raw_score_difference_nats"]) for row in metric_rows
        )
        <= config.consolidation.raw_score_difference_max_nats,
        "route_decisions_are_stable": min(float(row["route_agreement"]) for row in metric_rows)
        >= config.consolidation.route_agreement_min,
        "score_offset_has_no_level_slope": abs(slope)
        <= config.consolidation.level_offset_slope_max_nats,
    }
    summary: dict[str, object] = {
        "final_active_levels": [node.level for node in final_bank.topology.active_nodes],
        "gates": gates,
        "level_offset_slope_nats": slope,
        "merge_results": metric_rows,
        "passed": all(gates.values()),
        "schema_version": "vamp-logt-consolidation-seed-v1",
        "stream_seed": stream_seed,
        "work_counters": asdict(final_bank.counters),
    }
    publish_immutable_json(summary_path, summary)
    retire_inactive_node_artifacts(nodes_root, {node.node_id for node in topology.active_nodes})
    return summary


def _score_and_evaluate_states(
    nodes: tuple[TemporalNode, ...],
    adapters: Mapping[str, TopTwoAdapterState],
    states: Mapping[str, FrozenEvidenceState],
    holdout: NodeHoldout,
    equivalence: Mapping[str, tuple[int, ...]],
    baseline: AuthenticatedBaseline,
    config: VampLogTEvidenceConfig,
    device: torch.device,
):
    models = {node_id: materialize_evidence_model(state) for node_id, state in states.items()}
    scores = score_evidence_bank(
        nodes,
        models,
        holdout.table.raw_images,
        device,
        config.evidence.score_batch_size,
    )
    evaluation = evaluate_routing(
        nodes,
        adapters,
        scores,
        holdout.table,
        baseline.top_two_base,
        device,
        config.online.evaluation_batch_size,
        holdout,
        equivalence,
    )
    return scores, evaluation


def _run_online_seed(
    config: VampLogTEvidenceConfig,
    baseline: AuthenticatedBaseline,
    phase_root: Path,
    device: torch.device,
    selected_bridges: int,
    stream_seed: int,
) -> dict[str, object]:
    seed_root = phase_root / f"seed-{stream_seed}"
    summary_path = seed_root / "summary.json"
    if summary_path.is_file():
        return load_canonical_json(summary_path)
    seed_root.mkdir(parents=True, exist_ok=True)
    stream = stream_training_table(baseline, stream_seed)
    specifications = {"direct": 1, "tre": selected_bridges}
    checkpoint = seed_root / "state" / "checkpoint.pt"
    bank = load_bank_checkpoint(checkpoint) if checkpoint.is_file() else empty_active_bank(config.stream.block_size)
    ledger = ChainedJsonlLedger(seed_root / "metrics.jsonl", "vamp-logt-online-metric-v1")
    bank = _reconcile_online_route_work(config, bank, ledger)
    if checkpoint.is_file():
        save_bank_checkpoint(checkpoint, bank)
    evaluated = {int(row["processed_blocks"]) for row in ledger.rows}
    started = time.monotonic()
    try:
        from tqdm.auto import tqdm
    except ImportError as error:  # pragma: no cover - vision environment gate
        raise RuntimeError("tqdm is required by the vision environment") from error
    progress = tqdm(
        total=config.stream.total_blocks,
        initial=bank.topology.processed_blocks,
        desc=f"online seed {stream_seed} total",
        disable=not config.runtime.progress,
        unit="block",
    )
    context_blocks = config.stream.examples_per_context // config.stream.block_size
    while True:
        blocks = bank.topology.processed_blocks
        if blocks > 0 and blocks % context_blocks == 0 and blocks not in evaluated:
            rows, route_model_evals = _online_evaluation_rows(
                config,
                bank,
                baseline,
                blocks,
                stream_seed,
                device,
            )
            ledger.append_many(rows)
            evaluated.add(blocks)
            bank = ActiveEvidenceBank(
                bank.topology,
                bank.adapters,
                bank.evidence_by_condition,
                bank.counters.with_routing(
                    route_model_evals,
                    len(bank.topology.active_nodes) * len(specifications),
                ),
                bank.adapter_example_updates,
            )
            save_bank_checkpoint(checkpoint, bank)
        if blocks == config.stream.total_blocks:
            break
        bank = _advance_one_block(
            config,
            bank,
            stream,
            baseline,
            stream_seed,
            seed_root,
            device,
            specifications,
            "online",
        )
        progress.update(1)
    progress.close()
    metric_rows = tuple(_without_chain(row) for row in ledger.rows)
    final = [row for row in metric_rows if int(row["processed_blocks"]) == config.stream.total_blocks]
    summary: dict[str, object] = {
        "active_intervals": [
            {
                "first_block": node.first_block,
                "last_block": node.last_block,
                "level": node.level,
                "node_id": node.node_id,
            }
            for node in bank.topology.active_nodes
        ],
        "evaluations": list(metric_rows),
        "final_accuracy": {
            str(row["condition"]): float(row["accuracy"]) for row in final
        },
        "schema_version": "vamp-logt-online-seed-v1",
        "stream_seed": stream_seed,
        "wall_seconds": time.monotonic() - started,
        "work_counters": asdict(bank.counters),
    }
    publish_immutable_json(summary_path, summary)
    return summary


def _online_evaluation_rows(
    config: VampLogTEvidenceConfig,
    bank: ActiveEvidenceBank,
    baseline: AuthenticatedBaseline,
    processed_blocks: int,
    stream_seed: int,
    device: torch.device,
) -> tuple[tuple[dict[str, object], ...], int]:
    seen_contexts = processed_blocks * config.stream.block_size // config.stream.examples_per_context
    ids = torch.nonzero(baseline.test.context_ids < seen_contexts, as_tuple=True)[0]
    evaluation_table = baseline.test.select(ids)
    rows = []
    oracle_accuracy = None
    oracle_regret_reference = None
    for condition in ("direct", "tre"):
        models = {
            node_id: materialize_evidence_model(state)
            for node_id, state in bank.evidence_by_condition[condition].items()
        }
        scores = score_evidence_bank(
            bank.topology.active_nodes,
            models,
            evaluation_table.raw_images,
            device,
            config.evidence.score_batch_size,
        )
        evaluation = evaluate_routing(
            bank.topology.active_nodes,
            bank.adapters,
            scores,
            evaluation_table,
            baseline.top_two_base,
            device,
            config.online.evaluation_batch_size,
        )
        rows.append(
            {
                "accuracy": evaluation.routed_accuracy,
                "active_evidence_models": bank.counters.active_evidence_models,
                "active_nodes": len(bank.topology.active_nodes),
                "condition": "direct_nce" if condition == "direct" else "tre",
                "level_rows": list(evaluation.level_rows),
                "processed_blocks": processed_blocks,
                "route_oracle_agreement": evaluation.route_oracle_agreement,
                "routing_regret_nats": evaluation.routing_regret_nats,
                "seen_contexts": seen_contexts,
                "stream_seed": stream_seed,
            }
        )
        oracle_accuracy = evaluation.oracle_accuracy
        oracle_regret_reference = evaluation
    if oracle_accuracy is None or oracle_regret_reference is None:
        raise RuntimeError("online evaluation did not run either evidence condition")
    rows.append(
        {
            "accuracy": oracle_accuracy,
            "active_evidence_models": bank.counters.active_evidence_models,
            "active_nodes": len(bank.topology.active_nodes),
            "condition": "oracle_node",
            "level_rows": [],
            "processed_blocks": processed_blocks,
            "route_oracle_agreement": 1.0,
            "routing_regret_nats": 0.0,
            "seen_contexts": seen_contexts,
            "stream_seed": stream_seed,
        }
    )
    model_evals = len(evaluation_table.labels) * len(bank.topology.active_nodes) * 2
    return tuple(rows), model_evals


def _reconcile_online_route_work(
    config: VampLogTEvidenceConfig,
    bank: ActiveEvidenceBank,
    ledger: ChainedJsonlLedger,
) -> ActiveEvidenceBank:
    expected = sum(
        int(row["seen_contexts"])
        * config.stream.examples_per_context
        * int(row["active_nodes"])
        * 2
        for row in ledger.rows
        if row.get("condition") == "direct_nce"
    )
    actual = bank.counters.evidence_route_model_evals
    if actual > expected:
        raise ValueError("online checkpoint contains routing work absent from the ledger")
    if actual == expected:
        return bank
    return ActiveEvidenceBank(
        bank.topology,
        bank.adapters,
        bank.evidence_by_condition,
        bank.counters.with_routing(
            expected - actual,
            len(bank.topology.active_nodes) * len(bank.evidence_by_condition),
        ),
        bank.adapter_example_updates,
    )


def _finish_blocked_report(
    run_root: Path,
    config: VampLogTEvidenceConfig,
    baseline: AuthenticatedBaseline,
    phases: Mapping[str, Mapping[str, object]],
) -> Path:
    from apm.experiments.vamp_logt_evidence_reporting import write_result_report

    write_result_report(run_root, config, baseline.summary, phases)
    return run_root


def _without_chain(row: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in row.items()
        if key not in {"format", "previous_sha256", "result_sha256", "sequence"}
    }


def _write_resolved_yaml(path: Path, record: Mapping[str, object]) -> None:
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - vision environment gate
        raise RuntimeError("PyYAML is required by the vision environment") from error
    atomic_write(path, yaml.safe_dump(dict(record), sort_keys=True).encode("utf-8"))


def _material_source_hashes() -> dict[str, str]:
    project_root = Path(__file__).resolve().parents[3]
    relative_paths = (
        "src/apm/continual/logt_evidence_bank.py",
        "src/apm/continual/nce_tre_evidence.py",
        "src/apm/continual/top_two_adapter.py",
        "src/apm/experiments/vamp_logt_evidence_config.py",
        "src/apm/experiments/vamp_logt_evidence_data.py",
        "src/apm/experiments/vamp_logt_evidence_mnist.py",
        "src/apm/experiments/vamp_logt_evidence_reporting.py",
        "src/apm/experiments/vamp_logt_evidence_state.py",
        "src/apm/experiments/vamp_logt_evidence_training.py",
        "src/apm/experiments/vamp_logt_evidence_workflow.py",
        "src/apm/experiments/vamp_logt_ratio_calibration.py",
        "configs/vamp_logt_evidence_mnist/nce_tre.yaml",
        "configs/vamp_logt_evidence_mnist/nce_tre_base_reference.yaml",
        "docs/Codex Handoff_ NCE-TRE Evidence Routing for LogT-VAMP on MNIST.md",
        "docs/NCE_TRE_BASE_REFERENCE.md",
    )
    return {path: file_sha256(project_root / path) for path in relative_paths}


__all__ = [
    "run_consolidation_phase",
    "run_online_phase",
    "run_static_phase",
    "run_workflow",
]

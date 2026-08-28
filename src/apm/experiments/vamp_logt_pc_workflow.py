"""Resumable phase-gated workflow for normalized generative-PC evidence."""

from __future__ import annotations

from dataclasses import asdict, replace
from importlib.metadata import version
import gc
import io
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.65")

import jax
import jax.numpy as jnp
import numpy as np
from pyrsistent import pmap

from apm.continual.artifacts import (
    ChainedJsonlLedger,
    atomic_write,
    file_sha256,
    load_canonical_json,
    publish_immutable_bytes,
    publish_immutable_json,
    record_sha256,
)
from apm.continual.logt_evidence_bank import empty_logt_state, insert_block
from apm.experiments.vamp_logt_pc_config import VampLogTPcConfig
from apm.experiments.vamp_logt_pc_data import (
    CONDITIONS,
    AuthenticatedPcData,
    PcNodeHoldout,
    PcRawTable,
    authenticate_and_load_pc_data,
    build_condition_stream,
    build_node_holdout,
    context_holdout,
    preflight_tables,
)
from apm.experiments.vamp_logt_pc_state import (
    ActivePcBank,
    PcWorkCounters,
    load_bank_checkpoint,
    require_pc_work_bound,
    retire_inactive_models,
    save_bank_checkpoint,
)
from apm.experiments.vamp_logt_pc_training import (
    SCORE_NAMES,
    SelectedPcProtocol,
    classifier_config,
    evaluate_node_replica,
    fit_or_load_node_replica,
    make_backend,
    score_array,
)
from apm.models.fabricpc_density_backend import (
    PcDensityConfig,
    PcDensityTrainConfig,
    PcFitResult,
    classifier_logits,
    fit_classifier,
    load_pc_model,
    publish_pc_model,
)


FABRICPC_COMMIT = "138941ef5763ab202c7df07879d3f21678e6cc0a"


def _release_jax_caches() -> None:
    """Drop compilation caches between independent experiment units."""
    jax.clear_caches()
    gc.collect()


def run_workflow(config: VampLogTPcConfig) -> Path:
    """Run or resume every preregistered PC phase in de-risking order."""
    if config.protocol_revision in {"generative-pc-gn-v1", "generative-pc-gn-v2"}:
        from apm.experiments.vamp_logt_pc_gn_workflow import run_gn_workflow

        return run_gn_workflow(config)
    _require_runtime(config)
    run_root = config.artifact_root / "runs" / config.config_hash
    run_root.mkdir(parents=True, exist_ok=True)
    _write_resolved_yaml(run_root / "config_resolved.yaml", config.as_record())
    print(f"Generative-PC durable working directory: {run_root}", flush=True)

    print("Phase 0/5: authenticate raw-only source artifacts", flush=True)
    data = authenticate_and_load_pc_data(config)
    publish_immutable_json(
        run_root / "protocol.json",
        {
            "config": config.as_record(),
            "config_hash": config.config_hash,
            "fabricpc_commit": FABRICPC_COMMIT,
            "fabricpc_version": version("fabricpc"),
            "jax_version": version("jax"),
            "jaxlib_version": version("jaxlib"),
            "material_source_sha256": _material_source_hashes(),
            "raw_cache_sha256": data.raw_cache_sha256,
            "schema_version": "vamp-logt-generative-pc-map-protocol-v1",
            "source_protocol_sha256": data.source_protocol_sha256,
        },
    )

    print("Phase 1/5: exact analytic MAP-score validation", flush=True)
    analytic = run_analytic_phase(config, run_root / "calibration")
    if not bool(analytic["passed"]):
        return _finish_report(run_root, config, {"analytic": analytic})

    print("Phase 2/5: one-node image-model preflight", flush=True)
    preflight = run_preflight_phase(config, data, run_root / "preflight")
    if not bool(preflight["passed"]):
        return _finish_report(run_root, config, {"analytic": analytic, "preflight": preflight})
    selected = SelectedPcProtocol(**preflight["selected_protocol"])

    print("Phase 3/5: three controlled 31-block static LogT conditions", flush=True)
    minimal = run_static_phase(
        config,
        data,
        selected,
        run_root / "static" / "minimal",
        config.stream.minimal_stream_seeds,
        "minimal",
    )
    phases: dict[str, dict[str, object]] = {
        "analytic": analytic,
        "preflight": preflight,
        "static_minimal": minimal,
    }
    if not bool(minimal["passed"]):
        return _finish_report(run_root, config, phases)

    print("Phase 3b/5: confirmation on two additional stream seeds", flush=True)
    confirmation = run_static_phase(
        config,
        data,
        selected,
        run_root / "static" / "confirmation",
        config.stream.confirmation_stream_seeds,
        "confirmation",
        required_scores=tuple(str(value) for value in minimal["passing_scores"]),
    )
    phases["static_confirmation"] = confirmation
    if not bool(confirmation["passed"]):
        return _finish_report(run_root, config, phases)

    print("Phase 4/5: recurrent-schedule block-27 to block-28 partial carry", flush=True)
    consolidation = run_partial_carry_phase(
        config,
        data,
        selected,
        run_root / "consolidation",
        tuple(str(value) for value in confirmation["passing_scores"]),
    )
    phases["consolidation"] = consolidation
    return _finish_report(run_root, config, phases)


def run_analytic_phase(config: VampLogTPcConfig, phase_root: Path) -> dict[str, object]:
    """Validate the complete normalized MAP score without using curvature."""
    summary_path = phase_root / "summary.json"
    if summary_path.is_file():
        return load_canonical_json(summary_path)
    phase_root.mkdir(parents=True, exist_ok=True)
    with jax.experimental.enable_x64():
        weight = jnp.asarray([[0.7, -0.2], [0.1, 0.5], [-0.3, 0.4]], dtype=jnp.float64)
        bias = jnp.asarray([0.2, -0.1, 0.3], dtype=jnp.float64)
        variance = 0.4
        precision = 1.0 / variance
        errors: list[float] = []
        missing_prior_changes: list[float] = []
        for image in (
            jnp.asarray([0.1, 0.4, -0.2], dtype=jnp.float64),
            jnp.asarray([1.2, -0.7, 0.3], dtype=jnp.float64),
            jnp.asarray([-0.8, 0.2, 1.1], dtype=jnp.float64),
        ):
            normal_matrix = jnp.eye(2, dtype=jnp.float64) + precision * weight.T @ weight
            mode = jnp.linalg.solve(normal_matrix, precision * weight.T @ (image - bias))
            residual = image - (weight @ mode + bias)
            implemented = -(
                0.5 * jnp.sum(jnp.square(mode))
                + mode.size / 2.0 * jnp.log(2.0 * jnp.pi)
                + 0.5 * precision * jnp.sum(jnp.square(residual))
                + image.size / 2.0 * jnp.log(2.0 * jnp.pi * variance)
            )
            reference = jax.scipy.stats.multivariate_normal.logpdf(
                mode,
                jnp.zeros_like(mode),
                jnp.eye(mode.size, dtype=mode.dtype),
            ) + jax.scipy.stats.multivariate_normal.logpdf(
                image,
                weight @ mode + bias,
                variance * jnp.eye(image.size, dtype=image.dtype),
            )
            likelihood_only = jax.scipy.stats.multivariate_normal.logpdf(
                image,
                weight @ mode + bias,
                variance * jnp.eye(image.size, dtype=image.dtype),
            )
            errors.append(abs(float(implemented - reference)))
            missing_prior_changes.append(abs(float(likelihood_only - implemented)))
    curved = _curved_manifold_map_diagnostics()
    maximum_error = max(errors)
    summary = {
        "curved_manifold": curved,
        "estimator": config.evidence.estimator,
        "linear_gaussian_maximum_map_formula_error_nats": maximum_error,
        "missing_prior_minimum_score_change_nats": min(missing_prior_changes),
        "passed": maximum_error < 1.0e-10 and min(missing_prior_changes) > 1.0e-2,
        "schema_version": "vamp-logt-pc-map-analytic-v1",
    }
    publish_immutable_json(summary_path, summary)
    return summary


def run_preflight_phase(
    config: VampLogTPcConfig,
    data: AuthenticatedPcData,
    phase_root: Path,
) -> dict[str, object]:
    """Choose globally shared training settings without inspecting routing."""
    summary_path = phase_root / "summary.json"
    if summary_path.is_file():
        return load_canonical_json(summary_path)
    phase_root.mkdir(parents=True, exist_ok=True)
    train, heldout = preflight_tables(
        data,
        config.preflight.train_examples,
        config.preflight.heldout_examples,
    )
    rows: list[dict[str, object]] = []
    candidates: list[tuple[tuple[float, ...], dict[str, object], str]] = []
    for image_precision in config.preflight.image_precisions:
        for hidden_precision in config.preflight.hidden_precisions:
            for eta in config.preflight.inference_step_sizes:
                selected = SelectedPcProtocol(image_precision, hidden_precision, eta)
                backend = make_backend(config, selected, 0)
                identity = record_sha256(
                    {
                        "hidden_precision": hidden_precision,
                        "image_precision": image_precision,
                        "inference_step_size": eta,
                        "schema_version": "vamp-logt-pc-map-preflight-model-v1",
                    }
                )
                directory = phase_root / "models" / identity
                density = None
                settled_train = None
                classifier = None
                if (directory / "model.npz").is_file() and (directory / "manifest.json").is_file():
                    stored = load_pc_model(directory, backend, identity, 0)
                else:
                    density = backend.fit(train.images_float32, 0)
                    settled_train = backend.settle_images(density.params, train.images_float32)
                    classifier = fit_classifier(
                        settled_train.hidden,
                        train.labels,
                        0,
                        classifier_config(config),
                    )
                    stored = publish_pc_model(directory, backend, identity, 0, density, classifier)
                settled = backend.settle_images(stored.params, heldout.images_float32)
                trained_joint = backend.map_joint_scores_from_settled(
                    stored.params,
                    heldout.images_float32,
                    settled,
                )
                untrained_params = backend.init_params(0)
                untrained_joint = backend.map_joint_scores(untrained_params, heldout.images_float32)
                logits = classifier_logits(stored.classifier, settled.hidden)
                accuracy = float(np.mean(np.argmax(logits, axis=-1) == heldout.labels))
                reconstruction = backend.reconstruct_images(stored.params, heldout.images_float32)
                reconstruction_mse = float(np.mean(np.square(reconstruction - heldout.images_float32)))
                initial_median = float(np.median(settled.initial_gradient_norm))
                final_median = float(np.median(settled.final_gradient_norm))
                reduction = initial_median / max(final_median, 1.0e-12)
                improvement = float(np.mean(trained_joint) - np.mean(untrained_joint))
                finite = bool(
                    np.all(np.isfinite(trained_joint))
                    and np.all(np.isfinite(settled.final_gradient_norm))
                    and np.all(np.isfinite(reconstruction))
                )
                passed_training = bool(
                    finite
                    and accuracy >= config.preflight.classifier_accuracy_min
                    and improvement > 0.01
                    and reduction >= config.preflight.gradient_reduction_min
                )
                row: dict[str, object] = {
                    "classifier_accuracy": accuracy,
                    "gradient_reduction": reduction,
                    "hidden_precision": hidden_precision,
                    "image_precision": image_precision,
                    "inference_step_size": eta,
                    "joint_score_improvement_nats": improvement,
                    "mean_complete_joint_score": float(np.mean(trained_joint)),
                    "passed_training": passed_training,
                    "reconstruction_mse": reconstruction_mse,
                }
                rows.append(row)
                if passed_training:
                    ranking = (
                        float(np.mean(trained_joint)),
                        -reconstruction_mse,
                        reduction,
                        accuracy,
                    )
                    candidates.append((ranking, row, identity))
                del (
                    backend,
                    stored,
                    density,
                    settled_train,
                    classifier,
                    settled,
                    trained_joint,
                    untrained_params,
                    untrained_joint,
                    logits,
                    reconstruction,
                )
                _release_jax_caches()
    if not candidates:
        summary = {
            "candidates": rows,
            "passed": False,
            "reason": "No one-node model passed the preregistered learning and convergence gates.",
            "schema_version": "vamp-logt-pc-map-preflight-v1",
            "selected_protocol": None,
        }
        publish_immutable_json(summary_path, summary)
        return summary
    _ranking, chosen, _chosen_identity = max(candidates, key=lambda value: value[0])
    selected_protocol = SelectedPcProtocol(
        float(chosen["image_precision"]),
        float(chosen["hidden_precision"]),
        float(chosen["inference_step_size"]),
    )
    summary = {
        "candidates": rows,
        "passed": True,
        "reason": "The selected one-node model passed every MAP-only learning gate.",
        "schema_version": "vamp-logt-pc-map-preflight-v1",
        "selected_model_metrics": chosen,
        "selected_protocol": asdict(selected_protocol),
    }
    publish_immutable_json(summary_path, summary)
    return summary


def run_static_phase(
    config: VampLogTPcConfig,
    data: AuthenticatedPcData,
    selected: SelectedPcProtocol,
    phase_root: Path,
    stream_seeds: tuple[int, ...],
    phase_name: str,
    required_scores: tuple[str, ...] = SCORE_NAMES,
) -> dict[str, object]:
    """Run every controlled condition for the supplied stream seeds."""
    summary_path = phase_root / "summary.json"
    if summary_path.is_file():
        return load_canonical_json(summary_path)
    phase_root.mkdir(parents=True, exist_ok=True)
    condition_results: list[dict[str, object]] = []
    for stream_seed in stream_seeds:
        for condition in CONDITIONS:
            print(
                f"  {phase_name}: stream seed {stream_seed}, condition {condition}",
                flush=True,
            )
            result = _run_static_condition(
                config,
                data,
                selected,
                phase_root / f"seed-{stream_seed}" / condition,
                condition,
                stream_seed,
            )
            condition_results.append(result)
            del result
            _release_jax_caches()
    passing_scores = tuple(
        score
        for score in required_scores
        if all(bool(result["passed_by_score"][score]) for result in condition_results)
    )
    summary = {
        "conditions": condition_results,
        "passed": bool(passing_scores),
        "passing_scores": list(passing_scores),
        "phase": phase_name,
        "required_scores": list(required_scores),
        "schema_version": "vamp-logt-pc-map-static-phase-v1",
        "stream_seeds": list(stream_seeds),
    }
    publish_immutable_json(summary_path, summary)
    return summary


def _run_static_condition(
    config: VampLogTPcConfig,
    data: AuthenticatedPcData,
    selected: SelectedPcProtocol,
    condition_root: Path,
    condition: str,
    stream_seed: int,
) -> dict[str, object]:
    summary_path = condition_root / "summary.json"
    if summary_path.is_file():
        return load_canonical_json(summary_path)
    condition_root.mkdir(parents=True, exist_ok=True)
    stream = build_condition_stream(data.train, condition, stream_seed, config.stream.block_size)
    bank = _build_bank(
        config,
        selected,
        stream,
        condition_root,
        config.stream.static_blocks,
    )
    nodes = bank.topology.active_nodes
    if tuple(node.level for node in nodes) != (4, 3, 2, 1, 0):
        raise RuntimeError("31-block PC snapshot does not have the expected five-node frontier")
    general = build_node_holdout(
        nodes,
        stream,
        data.test,
        config.evaluation.heldout_per_node,
        stream_seed * 1_000 + 17,
    )
    focused = context_holdout(
        data.test,
        4,
        config.evaluation.focused_examples,
        stream_seed * 1_000 + 29,
    )
    models_root = condition_root / "models"
    evaluations: dict[int, dict[str, Any]] = {}
    focused_evaluations: dict[int, dict[str, Any]] = {}
    for replica_seed in config.stream.model_seeds:
        evaluations[replica_seed] = {}
        focused_evaluations[replica_seed] = {}
        backend = make_backend(config, selected, replica_seed)
        for node in nodes:
            model = load_pc_model(
                models_root / node.node_id / f"replica-{replica_seed}",
                backend,
                node.node_id,
                replica_seed,
            )
            evaluations[replica_seed][node.node_id] = evaluate_node_replica(backend, model, general.table)
            if node.level in {0, 4}:
                focused_evaluations[replica_seed][node.node_id] = evaluate_node_replica(
                    backend,
                    model,
                    focused,
                )
    model_count = len(nodes) * len(config.stream.model_seeds)
    counters = bank.counters.with_scoring(
        len(general.table.labels),
        model_count,
        config.training.infer_steps,
        hessians=False,
        active_models=model_count,
        settle_passes=1,
    ).with_scoring(
        len(focused.labels),
        2 * len(config.stream.model_seeds),
        config.training.infer_steps,
        hessians=False,
        settle_passes=1,
    )
    result, raw_arrays = _static_metrics(
        config,
        condition,
        stream_seed,
        stream,
        nodes,
        general,
        focused,
        evaluations,
        focused_evaluations,
    )
    require_pc_work_bound(
        counters,
        config.stream.static_blocks,
        config.stream.block_size,
        config.training.epochs,
        config.training.classifier_epochs,
        len(config.stream.model_seeds),
    )
    if counters.pc_laplace_hessian_evals or counters.pc_importance_audit_samples:
        raise RuntimeError("the MAP-only protocol performed curvature-dependent work")
    result.update(
        {
            "schema_version": "vamp-logt-pc-map-static-condition-v1",
            "work_counters": asdict(counters),
        }
    )
    _publish_npz(condition_root / "raw_scores.npz", raw_arrays)
    publish_immutable_json(summary_path, result)
    return result


def _build_bank(
    config: VampLogTPcConfig,
    selected: SelectedPcProtocol,
    stream: PcRawTable,
    root: Path,
    target_blocks: int,
) -> ActivePcBank:
    checkpoint = root / "bank.json"
    models_root = root / "models"
    if checkpoint.is_file():
        bank = load_bank_checkpoint(checkpoint)
    else:
        bank = ActivePcBank(empty_logt_state(config.stream.block_size), pmap(), PcWorkCounters())
    if bank.topology.processed_blocks > target_blocks:
        raise ValueError("PC bank checkpoint is beyond the requested boundary")
    backends = {
        replica_seed: make_backend(config, selected, replica_seed)
        for replica_seed in config.stream.model_seeds
    }
    for block in range(bank.topology.processed_blocks, target_blocks):
        start = block * config.stream.block_size
        ids = tuple(range(start, start + config.stream.block_size))
        topology, leaf, merges = insert_block(bank.topology, ids)
        counters = bank.counters
        for node in (leaf, *(merge.parent for merge in merges)):
            for replica_seed in config.stream.model_seeds:
                stored = fit_or_load_node_replica(
                    config,
                    stream,
                    node,
                    replica_seed,
                    models_root,
                    backends[replica_seed],
                )
                counters = counters.with_fit(
                    stored.density_example_presentations,
                    stored.classifier_example_presentations,
                    len(node.example_ids),
                    merge=node.level > 0,
                    infer_steps=config.training.infer_steps,
                )
        replicas = pmap(
            {node.node_id: config.stream.model_seeds for node in topology.active_nodes}
        )
        counters = replace(
            counters,
            active_pc_models=len(topology.active_nodes) * len(config.stream.model_seeds),
        )
        bank = ActivePcBank(topology, replicas, counters)
        require_pc_work_bound(
            counters,
            topology.processed_blocks,
            config.stream.block_size,
            config.training.epochs,
            config.training.classifier_epochs,
            len(config.stream.model_seeds),
        )
        save_bank_checkpoint(checkpoint, bank)
        retire_inactive_models(models_root, {node.node_id for node in topology.active_nodes})
    return bank


def _static_metrics(
    config: VampLogTPcConfig,
    condition: str,
    stream_seed: int,
    stream: PcRawTable,
    nodes: tuple[Any, ...],
    holdout: PcNodeHoldout,
    focused: PcRawTable,
    evaluations: dict[int, dict[str, Any]],
    focused_evaluations: dict[int, dict[str, Any]],
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    node_ids = tuple(node.node_id for node in nodes)
    node_index = {node_id: index for index, node_id in enumerate(node_ids)}
    source_indices = np.asarray(
        [node_index[node_id] for node_id in holdout.source_node_ids],
        dtype=np.int64,
    )
    mixture_keys = tuple(_node_context_key(node, stream) for node in nodes)
    source_mixture_keys = tuple(mixture_keys[index] for index in source_indices)
    routes_by_score: dict[str, dict[int, np.ndarray]] = {name: {} for name in SCORE_NAMES}
    replica_rows: list[dict[str, object]] = []
    raw: dict[str, np.ndarray] = {
        "general_labels": holdout.table.labels,
        "general_source_node_indices": source_indices,
        "focused_labels": focused.labels,
    }
    oracle_accuracies: dict[int, float] = {}
    for replica_seed in config.stream.model_seeds:
        candidate = evaluations[replica_seed]
        logits = np.stack([candidate[node_id].logits for node_id in node_ids])
        cross_entropy = np.stack([candidate[node_id].cross_entropy for node_id in node_ids])
        oracle_routes = np.argmin(cross_entropy, axis=0)
        examples = np.arange(len(holdout.table.labels))
        oracle_predictions = np.argmax(logits[oracle_routes, examples], axis=-1)
        oracle_accuracy = float(np.mean(oracle_predictions == holdout.table.labels))
        oracle_accuracies[replica_seed] = oracle_accuracy
        raw[f"general_{replica_seed}_logits"] = logits
        raw[f"general_{replica_seed}_oracle_routes"] = oracle_routes
        for score_name in SCORE_NAMES:
            matrix = np.stack(
                [score_array(candidate[node_id], score_name) for node_id in node_ids]
            )
            routes = np.argmax(matrix, axis=0)
            routes_by_score[score_name][replica_seed] = routes
            routed_predictions = np.argmax(logits[routes, examples], axis=-1)
            routed_accuracy = float(np.mean(routed_predictions == holdout.table.labels))
            exact_source_accuracy = float(np.mean(routes == source_indices))
            equivalent_accuracy = float(
                np.mean(
                    [mixture_keys[int(route)] == source for route, source in zip(routes, source_mixture_keys, strict=True)]
                )
            )
            conditional = {
                str(nodes[index].level): (
                    None
                    if not np.any(routes == index)
                    else float(
                        np.mean(
                            routed_predictions[routes == index]
                            == holdout.table.labels[routes == index]
                        )
                    )
                )
                for index in range(len(nodes))
            }
            replica_rows.append(
                {
                    "classifier_accuracy_by_routed_level": conditional,
                    "context_mixture_equivalent_source_accuracy": equivalent_accuracy,
                    "exact_temporal_source_accuracy": exact_source_accuracy,
                    "oracle_accuracy": oracle_accuracy,
                    "oracle_gap": oracle_accuracy - routed_accuracy,
                    "replica_seed": replica_seed,
                    "routed_accuracy": routed_accuracy,
                    "routing_regret_cross_entropy_nats": float(
                        np.mean(cross_entropy[routes, examples] - cross_entropy[oracle_routes, examples])
                    ),
                    "score": score_name,
                }
            )
            raw[f"general_{replica_seed}_{score_name}_scores"] = matrix
            raw[f"general_{replica_seed}_{score_name}_routes"] = routes

    leaf = next(node for node in nodes if node.level == 0)
    history = next(node for node in nodes if node.level == 4)
    focused_rows: list[dict[str, object]] = []
    focused_scores: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {
        score: {} for score in SCORE_NAMES
    }
    for replica_seed in config.stream.model_seeds:
        candidate = focused_evaluations[replica_seed]
        for score_name in SCORE_NAMES:
            leaf_scores = score_array(candidate[leaf.node_id], score_name)
            history_scores = score_array(candidate[history.node_id], score_name)
            margin = leaf_scores - history_scores
            wins = margin > 0.0
            lower = _bootstrap_win_lower(
                wins,
                config.evaluation.bootstrap_resamples,
                stream_seed * 100_003 + replica_seed * 101 + SCORE_NAMES.index(score_name),
            )
            focused_scores[score_name][replica_seed] = (leaf_scores, history_scores)
            focused_rows.append(
                {
                    "bootstrap_lower_95": lower,
                    "median_leaf_minus_history_nats": float(np.median(margin)),
                    "replica_seed": replica_seed,
                    "score": score_name,
                    "win_rate": float(np.mean(wins)),
                }
            )
            raw[f"focused_{replica_seed}_{score_name}_leaf"] = leaf_scores
            raw[f"focused_{replica_seed}_{score_name}_history"] = history_scores

    agreement_by_score = {
        score: min(
            float(np.mean(routes_by_score[score][left] == routes_by_score[score][right]))
            for index, left in enumerate(config.stream.model_seeds)
            for right in config.stream.model_seeds[index + 1 :]
        )
        for score in SCORE_NAMES
    }
    score_summaries: dict[str, dict[str, object]] = {}
    passed_by_score: dict[str, bool] = {}
    for score_name in SCORE_NAMES:
        score_replica_rows = [row for row in replica_rows if row["score"] == score_name]
        score_focused_rows = [row for row in focused_rows if row["score"] == score_name]
        common_gates = {
            "independent_replica_route_agreement": agreement_by_score[score_name]
            >= config.evaluation.route_agreement_min,
            "oracle_accuracy": min(oracle_accuracies.values()) >= config.evaluation.oracle_accuracy_min,
            "routed_within_oracle_gap": max(float(row["oracle_gap"]) for row in score_replica_rows)
            <= config.evaluation.classifier_oracle_gap_max,
        }
        condition_gates: dict[str, bool]
        offset_record: dict[str, float] | None = None
        if condition == "novel_leaf":
            condition_gates = {
                "novel_leaf_focused_win_rate": min(
                    float(row["bootstrap_lower_95"]) for row in score_focused_rows
                )
                > config.evaluation.novel_win_lower_min
            }
        elif condition == "recurrent_leaf_1_8":
            condition_gates = {
                "recurrent_leaf_focused_win_rate": min(
                    float(row["bootstrap_lower_95"]) for row in score_focused_rows
                )
                > config.evaluation.recurrent_win_lower_min,
                "recurrent_leaf_positive_margin": min(
                    float(row["median_leaf_minus_history_nats"]) for row in score_focused_rows
                )
                > 0.0,
            }
        elif condition == "identical_regime":
            cross_offsets = [
                float(np.median(focused_scores[score_name][left][0] - focused_scores[score_name][right][1]))
                for left in config.stream.model_seeds
                for right in config.stream.model_seeds
            ]
            same_offsets = []
            for position in (0, 1):
                for index, left in enumerate(config.stream.model_seeds):
                    for right in config.stream.model_seeds[index + 1 :]:
                        same_offsets.append(
                            abs(
                                float(
                                    np.median(
                                        focused_scores[score_name][left][position]
                                        - focused_scores[score_name][right][position]
                                    )
                                )
                            )
                        )
            cross_level = abs(float(np.median(cross_offsets)))
            typical_same = float(np.median(same_offsets))
            allowance = 2.0 * typical_same + config.evaluation.identical_offset_allowance_nats
            offset_record = {
                "absolute_cross_level_median_offset_nats": cross_level,
                "allowed_offset_nats": allowance,
                "typical_same_interval_replica_offset_nats": typical_same,
            }
            condition_gates = {"identical_regime_level_offset": cross_level <= allowance}
        else:  # pragma: no cover - condition validated by the data module
            raise ValueError(condition)
        gates = {**common_gates, **condition_gates}
        passed = all(gates.values())
        passed_by_score[score_name] = passed
        score_summaries[score_name] = {
            "gates": gates,
            "identical_offset": offset_record,
            "minimum_route_agreement": agreement_by_score[score_name],
            "passed": passed,
            "worst_oracle_gap": max(float(row["oracle_gap"]) for row in score_replica_rows),
        }

    level_statistics = []
    for replica_seed in config.stream.model_seeds:
        for node in nodes:
            evaluated = evaluations[replica_seed][node.node_id]
            for score_name in SCORE_NAMES:
                values = score_array(evaluated, score_name)
                level_statistics.append(
                    {
                        "level": node.level,
                        "mean": float(np.mean(values)),
                        "replica_seed": replica_seed,
                        "score": score_name,
                        "variance": float(np.var(values)),
                    }
                )
    return (
        {
            "condition": condition,
            "focused_metrics": focused_rows,
            "level_score_statistics": level_statistics,
            "passed": any(passed_by_score.values()),
            "passed_by_score": passed_by_score,
            "replica_metrics": replica_rows,
            "score_summaries": score_summaries,
            "stream_seed": stream_seed,
        },
        raw,
    )


def _node_context_key(node: Any, stream: PcRawTable) -> tuple[int, ...]:
    counts = np.bincount(stream.context_ids[np.asarray(node.example_ids)], minlength=5)
    divisor = int(np.gcd.reduce(counts))
    return tuple(int(value // divisor) for value in counts) if divisor else tuple(int(value) for value in counts)


def _bootstrap_win_lower(wins: np.ndarray, resamples: int, seed: int) -> float:
    probability = float(np.mean(np.asarray(wins, dtype=np.float64)))
    counts = np.random.default_rng(seed).binomial(len(wins), probability, size=resamples)
    return float(np.quantile(counts / len(wins), 0.025))


def run_partial_carry_phase(
    config: VampLogTPcConfig,
    data: AuthenticatedPcData,
    selected: SelectedPcProtocol,
    phase_root: Path,
    passing_scores: tuple[str, ...],
) -> dict[str, object]:
    """Compare the block-28 level-two parent with an independent de-novo twin."""
    summary_path = phase_root / "summary.json"
    if summary_path.is_file():
        return load_canonical_json(summary_path)
    phase_root.mkdir(parents=True, exist_ok=True)
    stream = build_condition_stream(data.train, "recurrent_leaf_1_8", 0, config.stream.block_size)
    bank = _build_bank(config, selected, stream, phase_root / "normal", 28)
    nodes = bank.topology.active_nodes
    if tuple(node.level for node in nodes) != (4, 3, 2):
        raise RuntimeError("block-28 partial carry did not produce levels four, three, and two")
    parent = next(node for node in nodes if node.level == 2)
    if len(parent.parent_node_ids) != 2:
        raise RuntimeError("partial-carry parent has no two-child lineage")
    twin_seed = 10_000
    twin_backend = make_backend(config, selected, twin_seed)
    twin = fit_or_load_node_replica(
        config,
        stream,
        parent,
        twin_seed,
        phase_root / "twin_models",
        twin_backend,
    )
    holdout = build_node_holdout(
        nodes,
        stream,
        data.test,
        config.evaluation.heldout_per_node,
        91,
    )
    model_seed = config.stream.model_seeds[0]
    normal_backend = make_backend(config, selected, model_seed)
    evaluations = {}
    for node in nodes:
        model = load_pc_model(
            phase_root / "normal" / "models" / node.node_id / f"replica-{model_seed}",
            normal_backend,
            node.node_id,
            model_seed,
        )
        evaluations[node.node_id] = evaluate_node_replica(normal_backend, model, holdout.table)
    twin_evaluation = evaluate_node_replica(twin_backend, twin, holdout.table)
    examples = np.arange(len(holdout.table.labels))
    normal_logits = np.stack([evaluations[node.node_id].logits for node in nodes])
    twin_logits = normal_logits.copy()
    parent_index = nodes.index(parent)
    twin_logits[parent_index] = twin_evaluation.logits
    score_results: dict[str, object] = {}
    for score_name in passing_scores:
        normal_scores = np.stack(
            [score_array(evaluations[node.node_id], score_name) for node in nodes]
        )
        twin_scores = normal_scores.copy()
        twin_scores[parent_index] = score_array(twin_evaluation, score_name)
        normal_routes = np.argmax(normal_scores, axis=0)
        twin_routes = np.argmax(twin_scores, axis=0)
        normal_predictions = np.argmax(normal_logits[normal_routes, examples], axis=-1)
        twin_predictions = np.argmax(twin_logits[twin_routes, examples], axis=-1)
        normal_accuracy = float(np.mean(normal_predictions == holdout.table.labels))
        twin_accuracy = float(np.mean(twin_predictions == holdout.table.labels))
        normal_parent_replicas = []
        for replica_seed in config.stream.model_seeds:
            backend = make_backend(config, selected, replica_seed)
            model = load_pc_model(
                phase_root / "normal" / "models" / parent.node_id / f"replica-{replica_seed}",
                backend,
                parent.node_id,
                replica_seed,
            )
            normal_parent_replicas.append(
                score_array(evaluate_node_replica(backend, model, holdout.table), score_name)
            )
        same_data_offsets = [
            abs(float(np.median(normal_parent_replicas[left] - normal_parent_replicas[right])))
            for left in range(len(normal_parent_replicas))
            for right in range(left + 1, len(normal_parent_replicas))
        ]
        typical_offset = float(np.median(same_data_offsets))
        parent_twin_offset = abs(
            float(
                np.median(
                    normal_parent_replicas[0]
                    - score_array(twin_evaluation, score_name)
                )
            )
        )
        gates = {
            "classifier_accuracy_difference": abs(normal_accuracy - twin_accuracy)
            <= config.evaluation.parent_accuracy_gap_max,
            "parent_twin_route_agreement": float(np.mean(normal_routes == twin_routes))
            >= config.evaluation.route_agreement_min,
            "raw_score_offset": parent_twin_offset
            <= 2.0 * typical_offset + config.evaluation.identical_offset_allowance_nats,
        }
        score_results[score_name] = {
            "gates": gates,
            "normal_accuracy": normal_accuracy,
            "parent_twin_offset_nats": parent_twin_offset,
            "passed": all(gates.values()),
            "route_agreement": float(np.mean(normal_routes == twin_routes)),
            "same_data_replica_offset_nats": typical_offset,
            "twin_accuracy": twin_accuracy,
        }
    checkpoint_exists = (phase_root / "normal" / "bank.json").is_file()
    children_absent = all(
        not (phase_root / "normal" / "models" / child).exists()
        for child in parent.parent_node_ids
    )
    lifecycle_gates = {
        "children_absent_after_commit": children_absent,
        "parent_checkpoint_committed": checkpoint_exists,
        "work_bound": True,
    }
    require_pc_work_bound(
        bank.counters,
        28,
        config.stream.block_size,
        config.training.epochs,
        config.training.classifier_epochs,
        len(config.stream.model_seeds),
    )
    summary = {
        "lifecycle_gates": lifecycle_gates,
        "passed": all(lifecycle_gates.values())
        and all(bool(value["passed"]) for value in score_results.values()),
        "parent_node_id": parent.node_id,
        "schema_version": "vamp-logt-pc-map-partial-carry-v1",
        "score_results": score_results,
        "twin_seed": twin_seed,
        "work_counters": asdict(bank.counters),
    }
    publish_immutable_json(summary_path, summary)
    return summary


def _finish_report(
    run_root: Path,
    config: VampLogTPcConfig,
    phases: dict[str, dict[str, object]],
) -> Path:
    from apm.experiments.vamp_logt_pc_reporting import write_result_report

    write_result_report(run_root, config, phases)
    return run_root


def _curved_manifold_map_diagnostics() -> dict[str, object]:
    """Compare the MAP joint score with dense one-dimensional marginalization."""
    from scipy.special import logsumexp

    latent = np.linspace(-8.0, 8.0, 40_001, dtype=np.float64)
    spacing = latent[1] - latent[0]
    variance = 0.09
    points = np.asarray([[0.0, 0.0], [1.0, 0.5], [-1.0, 0.5], [0.0, 1.5]])
    rows = []
    for point in points:
        mean = np.stack((latent, 0.5 * latent**2), axis=-1)
        joint = (
            0.5 * latent**2
            + 0.5 * math.log(2.0 * math.pi)
            + 0.5 / variance * np.sum(np.square(point - mean), axis=-1)
            + math.log(2.0 * math.pi * variance)
        )
        exact = float(logsumexp(-joint) + math.log(spacing))
        index = int(np.argmin(joint))
        rows.append(
            {
                "exact_log_evidence": exact,
                "map_error_nats": -float(joint[index]) - exact,
                "map_joint_score": -float(joint[index]),
                "point": point.tolist(),
            }
        )
    return {
        "interpretation": (
            "This diagnostic states the MAP limitation directly: the best complete joint "
            "score is not a marginal likelihood because it omits all latent volume away from "
            "the selected state. It is descriptive and does not gate the MAP-only run."
        ),
        "rows": rows,
    }


def _require_runtime(config: VampLogTPcConfig) -> None:
    if version("fabricpc") != "0.4.0" or version("jax") != "0.7.0" or version("jaxlib") != "0.7.0":
        raise RuntimeError("the PC protocol requires FabricPC 0.4.0 and JAX/JAXlib 0.7.0")
    if config.runtime.device == "gpu" and not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("the PC protocol requested a GPU but JAX cannot see one")
    if config.runtime.device == "cpu" and any(device.platform != "cpu" for device in jax.devices()):
        jax.config.update("jax_platform_name", "cpu")


def _write_resolved_yaml(path: Path, record: dict[str, object]) -> None:
    try:
        import yaml
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("PyYAML is required for the PC experiment") from error
    atomic_write(path, yaml.safe_dump(record, sort_keys=True).encode("utf-8"))


def _publish_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    publish_immutable_bytes(path, buffer.getvalue())


def _material_source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[3]
    relative_paths = (
        "configs/vamp_logt_pc_mnist/minimal.yaml",
        "docs/CODEX_HANDOFF_LOGT_GENERATIVE_PC_EVIDENCE.md",
        "src/apm/continual/artifacts.py",
        "src/apm/continual/logt_evidence_bank.py",
        "src/apm/models/fabricpc_density_backend.py",
        "src/apm/experiments/vamp_logt_pc_config.py",
        "src/apm/experiments/vamp_logt_pc_data.py",
        "src/apm/experiments/vamp_logt_pc_state.py",
        "src/apm/experiments/vamp_logt_pc_training.py",
        "src/apm/experiments/vamp_logt_pc_workflow.py",
        "src/apm/experiments/vamp_logt_pc_reporting.py",
        "src/apm/experiments/vamp_logt_pc_mnist.py",
    )
    missing = tuple(path for path in relative_paths if not (root / path).is_file())
    if missing:
        raise FileNotFoundError(f"PC protocol source manifest is incomplete: {missing}")
    return {path: file_sha256(root / path) for path in relative_paths}


__all__ = [
    "run_analytic_phase",
    "run_partial_carry_phase",
    "run_preflight_phase",
    "run_static_phase",
    "run_workflow",
]

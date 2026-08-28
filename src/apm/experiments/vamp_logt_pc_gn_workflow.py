"""Exact generalized Gauss-Newton evidence experiment for generative PC models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from importlib.metadata import version
import io
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from apm.continual.artifacts import (
    file_sha256,
    load_canonical_json,
    publish_immutable_bytes,
    publish_immutable_json,
    record_sha256,
)
from apm.continual.logt_evidence_bank import TemporalNode
from apm.experiments.vamp_logt_pc_config import (
    PcGaussNewtonContinuationEvidenceConfig,
    PcGaussNewtonEvidenceConfig,
    VampLogTPcConfig,
)
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
)
from apm.experiments.vamp_logt_pc_training import (
    GN_ALL_SCORE_NAMES,
    GN_ROUTING_SCORE_NAMES,
    GnNodeReplicaEvaluation,
    SelectedPcProtocol,
    evaluate_gn_node_replica,
    fit_or_load_node_replica,
    gn_score_array,
    make_backend,
)
from apm.experiments.vamp_logt_pc_workflow import (
    FABRICPC_COMMIT,
    _bootstrap_win_lower,
    _build_bank,
    _material_source_hashes,
    _node_context_key,
    _release_jax_caches,
    _require_runtime,
    _write_resolved_yaml,
)
from apm.models.fabricpc_density_backend import (
    PcGaussNewtonScores,
    PcSettledState,
    StoredPcModel,
    gauss_newton_log_evidence_at_state,
    load_pc_model,
)


GN_CANDIDATE_SCORES = ("gn0", "gn1")


@dataclass(frozen=True, slots=True)
class ImportedMapSource:
    """Authenticated self-contained copy of the sealed MAP inputs."""

    root: Path
    selected_protocol: SelectedPcProtocol
    selected_model_id: str
    source_tree_sha256: str


def run_gn_workflow(config: VampLogTPcConfig) -> Path:
    """Run or resume the separately versioned exact-GGN protocol."""
    _gn_evidence(config)
    _require_runtime(config)
    run_root = config.artifact_root / "runs" / config.config_hash
    run_root.mkdir(parents=True, exist_ok=True)
    _write_resolved_yaml(run_root / "config_resolved.yaml", config.as_record())
    print(f"Generative-PC GN durable working directory: {run_root}", flush=True)

    print("Phase 0/5: authenticate the raw MNIST boundary", flush=True)
    data = authenticate_and_load_pc_data(config)
    config_relative_path = (
        "configs/vamp_logt_pc_mnist/gauss_newton_v2.yaml"
        if config.protocol_revision == "generative-pc-gn-v2"
        else "configs/vamp_logt_pc_mnist/gauss_newton.yaml"
    )
    publish_immutable_json(
        run_root / "protocol.json",
        {
            "config": config.as_record(),
            "config_hash": config.config_hash,
            "fabricpc_commit": FABRICPC_COMMIT,
            "fabricpc_version": version("fabricpc"),
            "jax_version": version("jax"),
            "jaxlib_version": version("jaxlib"),
            "material_source_sha256": {
                **_material_source_hashes(),
                config_relative_path: file_sha256(
                    Path(__file__).resolve().parents[3] / config_relative_path
                ),
                "src/apm/experiments/vamp_logt_pc_gn_workflow.py": file_sha256(
                    Path(__file__)
                ),
                "src/apm/experiments/vamp_logt_pc_gn_reporting.py": file_sha256(
                    Path(__file__).with_name("vamp_logt_pc_gn_reporting.py")
                ),
            },
            "raw_cache_sha256": data.raw_cache_sha256,
            "resource_contract": {
                "gpu_processes": 1,
                "host_memory_hard_cap_gib": 8,
                "score_batch_size": config.training.score_batch_size,
                "store_dense_matrices": False,
            },
            "schema_version": "vamp-logt-generative-pc-gn-protocol-v1",
            "source_protocol_sha256": data.source_protocol_sha256,
        },
    )

    print("Phase 1/5: analytic GN formula and nonlinear-curvature checks", flush=True)
    analytic = run_gn_analytic_phase(config, run_root / "analytic")
    phases: dict[str, dict[str, object]] = {"analytic": analytic}
    if not bool(analytic["passed"]):
        return _finish_gn_report(run_root, config, phases)

    print("Phase 2/5: authenticate/import MAP models and audit 64 images", flush=True)
    try:
        imported = authenticate_and_import_map_source(config, run_root / "imported_map_source")
    except (FileNotFoundError, ValueError) as error:
        source_preflight = {
            "passed": False,
            "reason": str(error),
            "schema_version": "vamp-logt-pc-gn-source-preflight-v1",
        }
        publish_immutable_json(run_root / "source_preflight" / "summary.json", source_preflight)
        phases["source_preflight"] = source_preflight
        return _finish_gn_report(run_root, config, phases)
    source_preflight = run_gn_source_preflight(
        config,
        data,
        imported,
        run_root / "source_preflight",
    )
    phases["source_preflight"] = source_preflight
    if not bool(source_preflight["passed"]):
        return _finish_gn_report(run_root, config, phases)

    print("Phase 3/5: paired minimal rescore of the sealed MAP models", flush=True)
    minimal = run_gn_static_phase(
        config,
        data,
        imported.selected_protocol,
        run_root / "static" / "minimal",
        config.stream.minimal_stream_seeds,
        "minimal",
        imported_source_root=imported.root,
        required_scores=GN_CANDIDATE_SCORES,
    )
    phases["static_minimal"] = minimal
    if not bool(minimal["passed"]):
        return _finish_gn_report(run_root, config, phases)

    print("Phase 4/5: fresh confirmation models for seeds 1 and 2", flush=True)
    confirmation = run_gn_static_phase(
        config,
        data,
        imported.selected_protocol,
        run_root / "static" / "confirmation",
        config.stream.confirmation_stream_seeds,
        "confirmation",
        required_scores=tuple(str(value) for value in minimal["passing_scores"]),
    )
    phases["static_confirmation"] = confirmation
    if not bool(confirmation["passed"]):
        return _finish_gn_report(run_root, config, phases)

    print("Phase 5/5: block-27 to block-28 partial carry", flush=True)
    consolidation = run_gn_partial_carry_phase(
        config,
        data,
        imported.selected_protocol,
        run_root / "consolidation",
        tuple(str(value) for value in confirmation["passing_scores"]),
    )
    phases["consolidation"] = consolidation
    return _finish_gn_report(run_root, config, phases)


def run_gn_analytic_phase(
    config: VampLogTPcConfig,
    phase_root: Path,
) -> dict[str, object]:
    """Check GN0/GN1 against exact Gaussian evidence and nonlinear curvature."""
    summary_path = phase_root / "summary.json"
    if summary_path.is_file():
        return load_canonical_json(summary_path)
    phase_root.mkdir(parents=True, exist_ok=True)
    with jax.experimental.enable_x64():
        weight = jnp.asarray([[0.7, -0.2], [0.1, 0.5], [-0.3, 0.4]], dtype=jnp.float64)
        bias = jnp.asarray([0.2, -0.1, 0.3], dtype=jnp.float64)
        variance = 0.4
        precision = 1.0 / variance
        covariance = weight @ weight.T + variance * jnp.eye(3, dtype=jnp.float64)
        covariance_inverse = jnp.linalg.inv(covariance)
        _sign, covariance_logdet = jnp.linalg.slogdet(covariance)
        matrix_errors: list[float] = []
        gn0_errors: list[float] = []
        gn1_errors: list[float] = []
        for row, image in enumerate(
            (
                jnp.asarray([0.1, 0.4, -0.2], dtype=jnp.float64),
                jnp.asarray([1.2, -0.7, 0.3], dtype=jnp.float64),
                jnp.asarray([-0.8, 0.2, 1.1], dtype=jnp.float64),
            )
        ):
            def residual(state: jax.Array) -> jax.Array:
                return jnp.concatenate((state, jnp.sqrt(precision) * (image - weight @ state - bias)))

            def negative_log_joint(state: jax.Array) -> jax.Array:
                return (
                    0.5 * jnp.sum(jnp.square(residual(state)))
                    + state.size / 2.0 * jnp.log(2.0 * jnp.pi)
                    + image.size / 2.0 * jnp.log(2.0 * jnp.pi * variance)
                )

            normal_matrix = jnp.eye(2, dtype=jnp.float64) + precision * weight.T @ weight
            mode = jnp.linalg.solve(normal_matrix, precision * weight.T @ (image - bias))
            arbitrary = jnp.asarray([0.35 * (row + 1), -0.27 * (row + 1)], dtype=jnp.float64)
            at_mode = gauss_newton_log_evidence_at_state(residual, negative_log_joint, mode)
            away = gauss_newton_log_evidence_at_state(residual, negative_log_joint, arbitrary)
            delta = image - bias
            exact = -0.5 * (
                image.size * jnp.log(2.0 * jnp.pi)
                + covariance_logdet
                + delta @ covariance_inverse @ delta
            )
            matrix_errors.append(
                float(jnp.max(jnp.abs(at_mode.gauss_newton_matrix - at_mode.exact_hessian)))
            )
            gn0_errors.append(abs(float(at_mode.gn0_log_evidence - exact)))
            gn1_errors.append(abs(float(away.gn1_log_evidence - exact)))

        nonlinear_residual = lambda state: jnp.asarray([state[0], state[0] ** 2 - 1.0])
        nonlinear_joint = lambda state: 0.5 * jnp.sum(jnp.square(nonlinear_residual(state)))
        nonlinear = gauss_newton_log_evidence_at_state(
            nonlinear_residual,
            nonlinear_joint,
            jnp.asarray([0.0], dtype=jnp.float64),
        )
        nonlinear_hessian_min = float(jnp.linalg.eigvalsh(nonlinear.exact_hessian)[0])
        nonlinear_gn_min = float(jnp.linalg.eigvalsh(nonlinear.gauss_newton_matrix)[0])
        base = float(nonlinear_joint(jnp.asarray([0.0], dtype=jnp.float64)))
        direction_changes = {
            str(epsilon): [
                float(nonlinear_joint(jnp.asarray([epsilon], dtype=jnp.float64))) - base,
                float(nonlinear_joint(jnp.asarray([-epsilon], dtype=jnp.float64))) - base,
            ]
            for epsilon in _gn_evidence(config).negative_direction_epsilons
        }
    maximum_matrix_error = max(matrix_errors)
    maximum_gn0_error = max(gn0_errors)
    maximum_gn1_error = max(gn1_errors)
    passed = bool(
        maximum_matrix_error < 1.0e-10
        and maximum_gn0_error < 1.0e-10
        and maximum_gn1_error < 1.0e-10
        and nonlinear_hessian_min < 0.0
        and nonlinear_gn_min > 0.0
        and all(min(changes) < 0.0 for changes in direction_changes.values())
    )
    summary = {
        "linear_gaussian_g_equals_h_maximum_error": maximum_matrix_error,
        "linear_gaussian_gn0_at_mode_maximum_error_nats": maximum_gn0_error,
        "linear_gaussian_gn1_away_from_mode_maximum_error_nats": maximum_gn1_error,
        "nonlinear_direction_delta_nll": direction_changes,
        "nonlinear_minimum_gauss_newton_eigenvalue": nonlinear_gn_min,
        "nonlinear_minimum_hessian_eigenvalue": nonlinear_hessian_min,
        "passed": passed,
        "schema_version": "vamp-logt-pc-gn-analytic-v1",
    }
    publish_immutable_json(summary_path, summary)
    return summary


def required_map_source_files(source_root: Path) -> tuple[Path, ...]:
    """Resolve the exact sealed MAP files consumed by the paired rescore."""
    protocol = load_canonical_json(source_root / "protocol.json")
    preflight = load_canonical_json(source_root / "preflight" / "summary.json")
    selected_record = preflight.get("selected_protocol")
    if not isinstance(selected_record, dict):
        raise ValueError("the MAP source has no selected preflight protocol")
    selected = SelectedPcProtocol(
        float(selected_record["image_precision"]),
        float(selected_record["hidden_precision"]),
        float(selected_record["inference_step_size"]),
    )
    selected_model_id = record_sha256(
        {
            "hidden_precision": selected.hidden_precision,
            "image_precision": selected.image_precision,
            "inference_step_size": selected.inference_step_size,
            "schema_version": "vamp-logt-pc-map-preflight-model-v1",
        }
    )
    files = [
        source_root / "config_resolved.yaml",
        source_root / "protocol.json",
        source_root / "summary.json",
        source_root / "preflight" / "summary.json",
        source_root / "preflight" / "models" / selected_model_id / "manifest.json",
        source_root / "preflight" / "models" / selected_model_id / "model.npz",
        source_root / "static" / "minimal" / "summary.json",
    ]
    for condition in CONDITIONS:
        condition_root = source_root / "static" / "minimal" / "seed-0" / condition
        bank = load_bank_checkpoint(condition_root / "bank.json")
        if tuple(node.level for node in bank.topology.active_nodes) != (4, 3, 2, 1, 0):
            raise ValueError(f"MAP source condition {condition} has the wrong active topology")
        files.extend(
            (
                condition_root / "bank.json",
                condition_root / "summary.json",
                condition_root / "raw_scores.npz",
            )
        )
        for node in bank.topology.active_nodes:
            for replica_seed in (0, 1, 2):
                model_root = condition_root / "models" / node.node_id / f"replica-{replica_seed}"
                files.extend((model_root / "manifest.json", model_root / "model.npz"))
    missing = tuple(path for path in files if not path.is_file())
    if missing:
        raise FileNotFoundError(f"MAP model-source tree is incomplete: {missing[:3]}")
    if protocol.get("schema_version") != "vamp-logt-generative-pc-map-protocol-v1":
        raise ValueError("the model source is not a sealed MAP protocol run")
    return tuple(sorted(set(files)))


def map_source_tree_sha256(source_root: Path) -> tuple[str, tuple[dict[str, str], ...]]:
    """Hash every consumed source file with its source-relative path."""
    records = tuple(
        {
            "path": str(path.relative_to(source_root)),
            "sha256": file_sha256(path),
        }
        for path in required_map_source_files(source_root)
    )
    return (
        record_sha256(
            {
                "files": list(records),
                "schema_version": "vamp-logt-pc-map-model-source-tree-v1",
            }
        ),
        records,
    )


def authenticate_and_import_map_source(
    config: VampLogTPcConfig,
    destination: Path,
) -> ImportedMapSource:
    """Authenticate the sealed MAP tree and copy only consumed files."""
    if config.model_source is None or config.model_source_run_root is None:
        raise ValueError("GN protocol has no MAP model source")
    source_root = config.model_source_run_root
    if source_root.name != config.model_source.map_run_id:
        raise ValueError("MAP model-source run coordinate changed")
    protocol = load_canonical_json(source_root / "protocol.json")
    if (
        protocol.get("config_hash") != config.model_source.map_run_id
        or not isinstance(protocol.get("config"), dict)
        or protocol["config"].get("protocol_revision") != "generative-pc-map-v1"
    ):
        raise ValueError("MAP model-source protocol identity changed")
    tree_sha256, records = map_source_tree_sha256(source_root)
    if tree_sha256 != config.model_source.required_tree_sha256:
        raise ValueError(
            f"MAP model-source tree changed: expected {config.model_source.required_tree_sha256}, "
            f"found {tree_sha256}"
        )
    for record in records:
        relative = Path(record["path"])
        publish_immutable_bytes(destination / relative, (source_root / relative).read_bytes())
    publish_immutable_json(
        destination / "IMPORT_MANIFEST.json",
        {
            "files": list(records),
            "schema_version": "vamp-logt-pc-map-model-source-import-v1",
            "source_run_id": config.model_source.map_run_id,
            "source_tree_sha256": tree_sha256,
        },
    )
    preflight = load_canonical_json(destination / "preflight" / "summary.json")
    selected_record = preflight["selected_protocol"]
    selected = SelectedPcProtocol(
        float(selected_record["image_precision"]),
        float(selected_record["hidden_precision"]),
        float(selected_record["inference_step_size"]),
    )
    if selected != SelectedPcProtocol(100.0, 1.0, 0.01):
        raise ValueError("MAP source selected settings differ from the paired protocol")
    selected_model_id = record_sha256(
        {
            "hidden_precision": selected.hidden_precision,
            "image_precision": selected.image_precision,
            "inference_step_size": selected.inference_step_size,
            "schema_version": "vamp-logt-pc-map-preflight-model-v1",
        }
    )
    return ImportedMapSource(destination, selected, selected_model_id, tree_sha256)


def run_gn_source_preflight(
    config: VampLogTPcConfig,
    data: AuthenticatedPcData,
    imported: ImportedMapSource,
    phase_root: Path,
) -> dict[str, object]:
    """Verify source parity, G positive-definiteness, and float precision."""
    summary_path = phase_root / "summary.json"
    if summary_path.is_file():
        return load_canonical_json(summary_path)
    evidence_config = _gn_evidence(config)
    phase_root.mkdir(parents=True, exist_ok=True)
    _train, heldout = preflight_tables(
        data,
        config.preflight.train_examples,
        config.preflight.heldout_examples,
    )
    backend = make_backend(config, imported.selected_protocol, 0)
    model = load_pc_model(
        imported.root / "preflight" / "models" / imported.selected_model_id,
        backend,
        imported.selected_model_id,
        0,
    )
    all_settled = backend.settle_images(model.params, heldout.images_float32)
    map_scores = backend.map_joint_scores_from_settled(
        model.params,
        heldout.images_float32,
        all_settled,
    )
    source_preflight = load_canonical_json(imported.root / "preflight" / "summary.json")
    source_mean = float(source_preflight["selected_model_metrics"]["mean_complete_joint_score"])
    map_mean_error = abs(float(np.mean(map_scores)) - source_mean)
    audit_table = heldout.select(np.arange(evidence_config.curvature_audit_examples))
    settled, scores = backend.settle_and_score_gauss_newton(
        model.params,
        audit_table.images_float32,
        evidence_config.negative_direction_epsilons,
    )
    audit_rows = evidence_config.float64_audit_examples
    float32_gn1 = scores.gn1_log_evidence[:audit_rows].astype(np.float64)
    audit_settled = PcSettledState(
        settled.latent[:audit_rows],
        settled.hidden[:audit_rows],
        settled.initial_gradient_norm[:audit_rows],
        settled.final_gradient_norm[:audit_rows],
    )
    with jax.experimental.enable_x64():
        float64_scores = backend.gauss_newton_scores_from_settled(
            model.params,
            audit_table.images_float32[:audit_rows],
            audit_settled,
            evidence_config.negative_direction_epsilons,
            use_float64=True,
        )
    precision_errors = np.abs(float32_gn1 - float64_scores.gn1_log_evidence)
    g_successes = int(np.count_nonzero(scores.gauss_newton_cholesky_succeeded))
    h_successes = int(np.count_nonzero(scores.hessian_cholesky_succeeded))
    finite_scores = bool(
        np.all(np.isfinite(scores.gn0_log_evidence))
        and np.all(np.isfinite(scores.gn1_log_evidence))
    )
    map_parity_passed = bool(
        map_mean_error <= config.model_source.map_score_tolerance_nats
        if config.model_source is not None
        else False
    )
    precision_passed = bool(
        np.max(precision_errors) <= evidence_config.float64_tolerance_nats
    )
    gates = {
        "finite_gn_scores": finite_scores,
        "map_source_parity": map_parity_passed,
        "raw_gauss_newton_cholesky_every_image": g_successes
        == evidence_config.curvature_audit_examples,
    }
    precision_role = (
        evidence_config.float64_agreement_role
        if isinstance(evidence_config, PcGaussNewtonContinuationEvidenceConfig)
        else "required_gate"
    )
    if precision_role == "required_gate":
        gates["float32_float64_agreement"] = precision_passed
    passed = bool(
        all(gates.values())
    )
    arrays = {field.name: getattr(scores, field.name) for field in fields(PcGaussNewtonScores)}
    arrays["float64_gn1"] = float64_scores.gn1_log_evidence
    arrays["float32_gn1"] = float32_gn1
    _publish_npz(phase_root / "raw_scores.npz", arrays)
    summary = {
        "audit_examples": evidence_config.curvature_audit_examples,
        "exact_hessian_cholesky_successes": h_successes,
        "finite_gn_scores": finite_scores,
        "float32_float64_gn1_maximum_error_nats": float(np.max(precision_errors)),
        "float32_float64_passed": precision_passed,
        "float32_float64_role": precision_role,
        "gauss_newton_cholesky_successes": g_successes,
        "gates": gates,
        "map_mean_parity_error_nats": map_mean_error,
        "map_source_parity_passed": map_parity_passed,
        "negative_hessian_states": int(np.count_nonzero(scores.minimum_hessian_eigenvalue < 0.0)),
        "passed": passed,
        "reason": (
            "Every required source and numerical prerequisite passed. The float32-versus-"
            "float64 comparison remains recorded as a diagnostic."
            if passed
            else "The failed prerequisites were: "
            + ", ".join(name for name, value in gates.items() if not value)
            + "."
        ),
        "schema_version": "vamp-logt-pc-gn-source-preflight-v1",
        "selected_model_id": imported.selected_model_id,
        "selected_protocol": asdict(imported.selected_protocol),
        "source_tree_sha256": imported.source_tree_sha256,
    }
    publish_immutable_json(summary_path, summary)
    return summary


def run_gn_static_phase(
    config: VampLogTPcConfig,
    data: AuthenticatedPcData,
    selected: SelectedPcProtocol,
    phase_root: Path,
    stream_seeds: tuple[int, ...],
    phase_name: str,
    *,
    imported_source_root: Path | None = None,
    required_scores: tuple[str, ...] = GN_CANDIDATE_SCORES,
) -> dict[str, object]:
    """Run paired GN scoring for every controlled static condition."""
    if any(score not in GN_CANDIDATE_SCORES for score in required_scores):
        raise ValueError("only GN0 and GN1 may advance the GN protocol")
    if imported_source_root is not None and stream_seeds != (0,):
        raise ValueError("the sealed MAP import contains only minimal stream seed 0")
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
            source_condition_root = (
                imported_source_root / "static" / "minimal" / "seed-0" / condition
                if imported_source_root is not None
                else None
            )
            result = _run_gn_static_condition(
                config,
                data,
                selected,
                phase_root / f"seed-{stream_seed}" / condition,
                condition,
                stream_seed,
                source_condition_root=source_condition_root,
            )
            condition_results.append(result)
            _release_jax_caches()
    numerically_valid = all(bool(result["gn_numerically_valid"]) for result in condition_results)
    passing_scores = tuple(
        score
        for score in required_scores
        if numerically_valid
        and all(bool(result["passed_by_score"][score]) for result in condition_results)
    )
    summary = {
        "conditions": condition_results,
        "gn_numerically_valid": numerically_valid,
        "passed": bool(passing_scores),
        "passing_scores": list(passing_scores),
        "phase": phase_name,
        "required_scores": list(required_scores),
        "schema_version": "vamp-logt-pc-gn-static-phase-v1",
        "stream_seeds": list(stream_seeds),
    }
    publish_immutable_json(summary_path, summary)
    return summary


def _run_gn_static_condition(
    config: VampLogTPcConfig,
    data: AuthenticatedPcData,
    selected: SelectedPcProtocol,
    condition_root: Path,
    condition: str,
    stream_seed: int,
    *,
    source_condition_root: Path | None,
) -> dict[str, object]:
    summary_path = condition_root / "summary.json"
    if summary_path.is_file():
        return load_canonical_json(summary_path)
    condition_root.mkdir(parents=True, exist_ok=True)
    stream = build_condition_stream(data.train, condition, stream_seed, config.stream.block_size)
    if source_condition_root is None:
        bank = _build_bank(
            config,
            selected,
            stream,
            condition_root,
            config.stream.static_blocks,
        )
        models_root = condition_root / "models"
    else:
        bank = load_bank_checkpoint(source_condition_root / "bank.json")
        models_root = source_condition_root / "models"
    nodes = bank.topology.active_nodes
    if tuple(node.level for node in nodes) != (4, 3, 2, 1, 0):
        raise RuntimeError("31-block GN snapshot does not have the expected five-node frontier")
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
    evaluations, focused_evaluations = _evaluate_gn_condition(
        config,
        selected,
        nodes,
        models_root,
        general.table,
        focused,
        condition_root / "score_checkpoints",
    )
    result, raw_arrays = _gn_static_metrics(
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
    map_parity_error = None
    map_parity_passed = True
    if source_condition_root is not None:
        map_parity_error = _map_source_parity_error(
            source_condition_root / "raw_scores.npz",
            nodes,
            evaluations,
            focused_evaluations,
        )
        if config.model_source is None:
            raise ValueError("source parity has no declared tolerance")
        map_parity_passed = map_parity_error <= config.model_source.map_score_tolerance_nats
    negative_hessian_states = sum(
        int(np.count_nonzero(evaluation.evidence.minimum_hessian_eigenvalue < 0.0))
        for candidates in (*evaluations.values(), *focused_evaluations.values())
        for evaluation in candidates.values()
    )
    model_count = len(nodes) * len(config.stream.model_seeds)
    counters = bank.counters.with_gauss_newton_scoring(
        len(general.table.labels),
        model_count,
        config.training.infer_steps,
        negative_hessian_states=sum(
            int(np.count_nonzero(evaluation.evidence.minimum_hessian_eigenvalue < 0.0))
            for candidates in evaluations.values()
            for evaluation in candidates.values()
        ),
        direction_epsilon_count=len(_gn_evidence(config).negative_direction_epsilons),
        active_models=model_count,
    ).with_gauss_newton_scoring(
        len(focused.labels),
        2 * len(config.stream.model_seeds),
        config.training.infer_steps,
        negative_hessian_states=sum(
            int(np.count_nonzero(evaluation.evidence.minimum_hessian_eigenvalue < 0.0))
            for candidates in focused_evaluations.values()
            for evaluation in candidates.values()
        ),
        direction_epsilon_count=len(_gn_evidence(config).negative_direction_epsilons),
        active_models=model_count,
    )
    require_pc_work_bound(
        counters,
        config.stream.static_blocks,
        config.stream.block_size,
        config.training.epochs,
        config.training.classifier_epochs,
        len(config.stream.model_seeds),
    )
    expected_evaluations = len(general.table.labels) * model_count + len(focused.labels) * 2 * len(
        config.stream.model_seeds
    )
    gn_numerically_valid = bool(
        result["gauss_newton_cholesky_successes"] == expected_evaluations
        and result["finite_gn_score_count"] == 2 * expected_evaluations
        and map_parity_passed
    )
    result.update(
        {
            "gn_numerically_valid": gn_numerically_valid,
            "map_source_parity_error_nats": map_parity_error,
            "map_source_parity_passed": map_parity_passed,
            "negative_hessian_states": negative_hessian_states,
            "schema_version": "vamp-logt-pc-gn-static-condition-v1",
            "work_counters": asdict(counters),
        }
    )
    _publish_npz(condition_root / "raw_scores.npz", raw_arrays)
    publish_immutable_json(summary_path, result)
    return result


def _evaluate_gn_condition(
    config: VampLogTPcConfig,
    selected: SelectedPcProtocol,
    nodes: tuple[TemporalNode, ...],
    models_root: Path,
    general: PcRawTable,
    focused: PcRawTable,
    checkpoint_root: Path,
) -> tuple[
    dict[int, dict[str, GnNodeReplicaEvaluation]],
    dict[int, dict[str, GnNodeReplicaEvaluation]],
]:
    evaluations: dict[int, dict[str, GnNodeReplicaEvaluation]] = {}
    focused_evaluations: dict[int, dict[str, GnNodeReplicaEvaluation]] = {}
    for replica_seed in config.stream.model_seeds:
        archive_path = checkpoint_root / f"replica-{replica_seed}.npz"
        manifest_path = checkpoint_root / f"replica-{replica_seed}.json"
        if archive_path.is_file() and manifest_path.is_file():
            evaluations[replica_seed], focused_evaluations[replica_seed] = _load_replica_checkpoint(
                archive_path,
                manifest_path,
                nodes,
                replica_seed,
            )
            print(f"    replica {replica_seed}: loaded completed score checkpoint", flush=True)
            continue
        backend = make_backend(config, selected, replica_seed)
        general_candidates: dict[str, GnNodeReplicaEvaluation] = {}
        focused_candidates: dict[str, GnNodeReplicaEvaluation] = {}
        for node_index, node in enumerate(nodes, start=1):
            print(
                f"    replica {replica_seed}: scoring node {node_index}/{len(nodes)} at level {node.level}",
                flush=True,
            )
            model = load_pc_model(
                models_root / node.node_id / f"replica-{replica_seed}",
                backend,
                node.node_id,
                replica_seed,
            )
            general_candidates[node.node_id] = evaluate_gn_node_replica(
                backend,
                model,
                general,
                _gn_evidence(config).negative_direction_epsilons,
            )
            if node.level in {0, 4}:
                focused_candidates[node.node_id] = evaluate_gn_node_replica(
                    backend,
                    model,
                    focused,
                    _gn_evidence(config).negative_direction_epsilons,
                )
        _publish_replica_checkpoint(
            archive_path,
            manifest_path,
            nodes,
            replica_seed,
            general_candidates,
            focused_candidates,
        )
        evaluations[replica_seed] = general_candidates
        focused_evaluations[replica_seed] = focused_candidates
        del backend
        _release_jax_caches()
    return evaluations, focused_evaluations


def _evaluation_arrays(prefix: str, evaluation: GnNodeReplicaEvaluation) -> dict[str, np.ndarray]:
    arrays = {
        f"{prefix}_evidence_{field.name}": np.asarray(getattr(evaluation.evidence, field.name))
        for field in fields(PcGaussNewtonScores)
    }
    arrays[f"{prefix}_logits"] = evaluation.logits
    arrays[f"{prefix}_cross_entropy"] = evaluation.cross_entropy
    arrays[f"{prefix}_predictions"] = evaluation.predictions
    return arrays


def _evaluation_from_archive(
    archive: np.lib.npyio.NpzFile,
    prefix: str,
) -> GnNodeReplicaEvaluation:
    evidence = PcGaussNewtonScores(
        *(archive[f"{prefix}_evidence_{field.name}"] for field in fields(PcGaussNewtonScores))
    )
    return GnNodeReplicaEvaluation(
        evidence,
        archive[f"{prefix}_logits"],
        archive[f"{prefix}_cross_entropy"],
        archive[f"{prefix}_predictions"],
    )


def _publish_replica_checkpoint(
    archive_path: Path,
    manifest_path: Path,
    nodes: tuple[TemporalNode, ...],
    replica_seed: int,
    evaluations: dict[str, GnNodeReplicaEvaluation],
    focused_evaluations: dict[str, GnNodeReplicaEvaluation],
) -> None:
    arrays: dict[str, np.ndarray] = {}
    for node in nodes:
        arrays.update(_evaluation_arrays(f"general_{node.node_id}", evaluations[node.node_id]))
        if node.node_id in focused_evaluations:
            arrays.update(
                _evaluation_arrays(f"focused_{node.node_id}", focused_evaluations[node.node_id])
            )
    _publish_npz(archive_path, arrays)
    publish_immutable_json(
        manifest_path,
        {
            "archive_sha256": file_sha256(archive_path),
            "focused_node_ids": sorted(focused_evaluations),
            "node_ids": [node.node_id for node in nodes],
            "replica_seed": replica_seed,
            "schema_version": "vamp-logt-pc-gn-score-checkpoint-v1",
        },
    )


def _load_replica_checkpoint(
    archive_path: Path,
    manifest_path: Path,
    nodes: tuple[TemporalNode, ...],
    replica_seed: int,
) -> tuple[dict[str, GnNodeReplicaEvaluation], dict[str, GnNodeReplicaEvaluation]]:
    manifest = load_canonical_json(manifest_path)
    node_ids = [node.node_id for node in nodes]
    if (
        manifest.get("schema_version") != "vamp-logt-pc-gn-score-checkpoint-v1"
        or manifest.get("archive_sha256") != file_sha256(archive_path)
        or manifest.get("replica_seed") != replica_seed
        or manifest.get("node_ids") != node_ids
    ):
        raise ValueError("GN score checkpoint coordinates or content changed")
    focused_ids = manifest.get("focused_node_ids")
    if not isinstance(focused_ids, list):
        raise ValueError("GN score checkpoint has no focused node coordinates")
    with np.load(archive_path, allow_pickle=False) as archive:
        evaluations = {
            node_id: _evaluation_from_archive(archive, f"general_{node_id}")
            for node_id in node_ids
        }
        focused = {
            str(node_id): _evaluation_from_archive(archive, f"focused_{node_id}")
            for node_id in focused_ids
        }
    return evaluations, focused


def _map_source_parity_error(
    source_path: Path,
    nodes: tuple[TemporalNode, ...],
    evaluations: dict[int, dict[str, GnNodeReplicaEvaluation]],
    focused_evaluations: dict[int, dict[str, GnNodeReplicaEvaluation]],
) -> float:
    leaf = next(node for node in nodes if node.level == 0)
    history = next(node for node in nodes if node.level == 4)
    errors: list[float] = []
    with np.load(source_path, allow_pickle=False) as source:
        for replica_seed, candidates in evaluations.items():
            general = np.stack(
                [candidates[node.node_id].evidence.map_log_evidence for node in nodes]
            )
            errors.append(
                float(np.max(np.abs(general - source[f"general_{replica_seed}_map_scores"])))
            )
            focused = focused_evaluations[replica_seed]
            errors.append(
                float(
                    np.max(
                        np.abs(
                            focused[leaf.node_id].evidence.map_log_evidence
                            - source[f"focused_{replica_seed}_map_leaf"]
                        )
                    )
                )
            )
            errors.append(
                float(
                    np.max(
                        np.abs(
                            focused[history.node_id].evidence.map_log_evidence
                            - source[f"focused_{replica_seed}_map_history"]
                        )
                    )
                )
            )
    return max(errors)


def _gn_static_metrics(
    config: VampLogTPcConfig,
    condition: str,
    stream_seed: int,
    stream: PcRawTable,
    nodes: tuple[TemporalNode, ...],
    holdout: PcNodeHoldout,
    focused: PcRawTable,
    evaluations: dict[int, dict[str, GnNodeReplicaEvaluation]],
    focused_evaluations: dict[int, dict[str, GnNodeReplicaEvaluation]],
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Compute unchanged static gates plus conditional raw-H diagnostics."""
    node_ids = tuple(node.node_id for node in nodes)
    node_index = {node_id: index for index, node_id in enumerate(node_ids)}
    source_indices = np.asarray(
        [node_index[node_id] for node_id in holdout.source_node_ids],
        dtype=np.int64,
    )
    mixture_keys = tuple(_node_context_key(node, stream) for node in nodes)
    source_mixture_keys = tuple(mixture_keys[index] for index in source_indices)
    routes_by_score: dict[str, dict[int, np.ndarray]] = {
        name: {} for name in GN_ROUTING_SCORE_NAMES
    }
    replica_rows: list[dict[str, object]] = []
    hessian_rows: list[dict[str, object]] = []
    raw: dict[str, np.ndarray] = {
        "general_labels": holdout.table.labels,
        "general_source_node_indices": source_indices,
        "focused_labels": focused.labels,
        "node_levels": np.asarray([node.level for node in nodes], dtype=np.int64),
    }
    oracle_accuracies: dict[int, float] = {}
    g_successes = 0
    finite_gn_score_count = 0
    h_successes = 0
    total_evaluations = 0
    examples = np.arange(len(holdout.table.labels))
    for replica_seed in config.stream.model_seeds:
        candidates = evaluations[replica_seed]
        logits = np.stack([candidates[node_id].logits for node_id in node_ids])
        cross_entropy = np.stack([candidates[node_id].cross_entropy for node_id in node_ids])
        oracle_routes = np.argmin(cross_entropy, axis=0)
        oracle_predictions = np.argmax(logits[oracle_routes, examples], axis=-1)
        oracle_accuracy = float(np.mean(oracle_predictions == holdout.table.labels))
        oracle_accuracies[replica_seed] = oracle_accuracy
        raw[f"general_{replica_seed}_logits"] = logits
        raw[f"general_{replica_seed}_oracle_routes"] = oracle_routes
        for field in fields(PcGaussNewtonScores):
            raw[f"general_{replica_seed}_{field.name}"] = np.stack(
                [np.asarray(getattr(candidates[node_id].evidence, field.name)) for node_id in node_ids]
            )
        for candidate in candidates.values():
            score = candidate.evidence
            total_evaluations += len(score.map_log_evidence)
            g_successes += int(np.count_nonzero(score.gauss_newton_cholesky_succeeded))
            h_successes += int(np.count_nonzero(score.hessian_cholesky_succeeded))
            finite_gn_score_count += int(np.count_nonzero(np.isfinite(score.gn0_log_evidence)))
            finite_gn_score_count += int(np.count_nonzero(np.isfinite(score.gn1_log_evidence)))
        for score_name in GN_ROUTING_SCORE_NAMES:
            matrix = np.stack(
                [gn_score_array(candidates[node_id], score_name) for node_id in node_ids]
            )
            routes = np.argmax(matrix, axis=0)
            partitioned = np.partition(matrix, matrix.shape[0] - 2, axis=0)
            top_two_margin = partitioned[-1] - partitioned[-2]
            routes_by_score[score_name][replica_seed] = routes
            routed_predictions = np.argmax(logits[routes, examples], axis=-1)
            routed_accuracy = float(np.mean(routed_predictions == holdout.table.labels))
            exact_source_accuracy = float(np.mean(routes == source_indices))
            equivalent_accuracy = float(
                np.mean(
                    [
                        mixture_keys[int(route)] == source
                        for route, source in zip(routes, source_mixture_keys, strict=True)
                    ]
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
                    "minimum_top_two_score_margin_nats": float(np.min(top_two_margin)),
                    "median_top_two_score_margin_nats": float(np.median(top_two_margin)),
                    "routing_regret_cross_entropy_nats": float(
                        np.mean(
                            cross_entropy[routes, examples]
                            - cross_entropy[oracle_routes, examples]
                        )
                    ),
                    "score": score_name,
                }
            )
            raw[f"general_{replica_seed}_{score_name}_scores"] = matrix
            raw[f"general_{replica_seed}_{score_name}_routes"] = routes
            raw[f"general_{replica_seed}_{score_name}_top_two_margin_nats"] = top_two_margin

        hessian_matrix = np.stack(
            [gn_score_array(candidates[node_id], "h_laplace") for node_id in node_ids]
        )
        hessian_covered = np.all(np.isfinite(hessian_matrix), axis=0)
        hessian_routes = np.full(len(examples), -1, dtype=np.int64)
        hessian_routes[hessian_covered] = np.argmax(hessian_matrix[:, hessian_covered], axis=0)
        raw[f"general_{replica_seed}_h_laplace_scores"] = hessian_matrix
        raw[f"general_{replica_seed}_h_laplace_covered"] = hessian_covered
        raw[f"general_{replica_seed}_h_laplace_routes"] = hessian_routes
        if np.any(hessian_covered):
            covered_examples = examples[hessian_covered]
            routes = hessian_routes[hessian_covered]
            predictions = np.argmax(logits[routes, covered_examples], axis=-1)
            routed_accuracy: float | None = float(
                np.mean(predictions == holdout.table.labels[hessian_covered])
            )
            oracle_gap: float | None = float(
                np.mean(
                    np.argmax(logits[oracle_routes[hessian_covered], covered_examples], axis=-1)
                    == holdout.table.labels[hessian_covered]
                )
                - routed_accuracy
            )
        else:
            routed_accuracy = None
            oracle_gap = None
        hessian_rows.append(
            {
                "covered_examples": int(np.count_nonzero(hessian_covered)),
                "coverage_fraction": float(np.mean(hessian_covered)),
                "oracle_gap_on_covered_examples": oracle_gap,
                "replica_seed": replica_seed,
                "routed_accuracy_on_covered_examples": routed_accuracy,
                "total_examples": len(hessian_covered),
            }
        )

    leaf = next(node for node in nodes if node.level == 0)
    history = next(node for node in nodes if node.level == 4)
    focused_rows: list[dict[str, object]] = []
    focused_hessian_rows: list[dict[str, object]] = []
    focused_scores: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {
        score: {} for score in GN_ROUTING_SCORE_NAMES
    }
    for replica_seed in config.stream.model_seeds:
        candidates = focused_evaluations[replica_seed]
        for field in fields(PcGaussNewtonScores):
            raw[f"focused_{replica_seed}_leaf_{field.name}"] = np.asarray(
                getattr(candidates[leaf.node_id].evidence, field.name)
            )
            raw[f"focused_{replica_seed}_history_{field.name}"] = np.asarray(
                getattr(candidates[history.node_id].evidence, field.name)
            )
        for candidate in candidates.values():
            score = candidate.evidence
            total_evaluations += len(score.map_log_evidence)
            g_successes += int(np.count_nonzero(score.gauss_newton_cholesky_succeeded))
            h_successes += int(np.count_nonzero(score.hessian_cholesky_succeeded))
            finite_gn_score_count += int(np.count_nonzero(np.isfinite(score.gn0_log_evidence)))
            finite_gn_score_count += int(np.count_nonzero(np.isfinite(score.gn1_log_evidence)))
        for score_name in GN_ROUTING_SCORE_NAMES:
            leaf_scores = gn_score_array(candidates[leaf.node_id], score_name)
            history_scores = gn_score_array(candidates[history.node_id], score_name)
            margin = leaf_scores - history_scores
            wins = margin > 0.0
            lower = _bootstrap_win_lower(
                wins,
                config.evaluation.bootstrap_resamples,
                stream_seed * 100_003
                + replica_seed * 101
                + GN_ROUTING_SCORE_NAMES.index(score_name),
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
        h_leaf = gn_score_array(candidates[leaf.node_id], "h_laplace")
        h_history = gn_score_array(candidates[history.node_id], "h_laplace")
        covered = np.isfinite(h_leaf) & np.isfinite(h_history)
        focused_hessian_rows.append(
            {
                "covered_examples": int(np.count_nonzero(covered)),
                "coverage_fraction": float(np.mean(covered)),
                "median_leaf_minus_history_nats": (
                    None if not np.any(covered) else float(np.median(h_leaf[covered] - h_history[covered]))
                ),
                "replica_seed": replica_seed,
                "total_examples": len(covered),
                "win_rate": (
                    None if not np.any(covered) else float(np.mean(h_leaf[covered] > h_history[covered]))
                ),
            }
        )

    agreement_by_score = {
        score: min(
            float(np.mean(routes_by_score[score][left] == routes_by_score[score][right]))
            for index, left in enumerate(config.stream.model_seeds)
            for right in config.stream.model_seeds[index + 1 :]
        )
        for score in GN_ROUTING_SCORE_NAMES
    }
    score_summaries: dict[str, dict[str, object]] = {}
    passed_by_score: dict[str, bool] = {}
    for score_name in GN_ROUTING_SCORE_NAMES:
        score_replica_rows = [row for row in replica_rows if row["score"] == score_name]
        score_focused_rows = [row for row in focused_rows if row["score"] == score_name]
        common_gates = {
            "independent_replica_route_agreement": agreement_by_score[score_name]
            >= config.evaluation.route_agreement_min,
            "oracle_accuracy": min(oracle_accuracies.values())
            >= config.evaluation.oracle_accuracy_min,
            "routed_within_oracle_gap": max(
                float(row["oracle_gap"]) for row in score_replica_rows
            )
            <= config.evaluation.classifier_oracle_gap_max,
        }
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
                    float(row["median_leaf_minus_history_nats"])
                    for row in score_focused_rows
                )
                > 0.0,
            }
        elif condition == "identical_regime":
            cross_offsets = [
                float(
                    np.median(
                        focused_scores[score_name][left][0]
                        - focused_scores[score_name][right][1]
                    )
                )
                for left in config.stream.model_seeds
                for right in config.stream.model_seeds
            ]
            same_offsets = [
                abs(
                    float(
                        np.median(
                            focused_scores[score_name][left][position]
                            - focused_scores[score_name][right][position]
                        )
                    )
                )
                for position in (0, 1)
                for index, left in enumerate(config.stream.model_seeds)
                for right in config.stream.model_seeds[index + 1 :]
            ]
            cross_level = abs(float(np.median(cross_offsets)))
            typical_same = float(np.median(same_offsets))
            allowance = 2.0 * typical_same + config.evaluation.identical_offset_allowance_nats
            offset_record = {
                "absolute_cross_level_median_offset_nats": cross_level,
                "allowed_offset_nats": allowance,
                "typical_same_interval_replica_offset_nats": typical_same,
            }
            condition_gates = {"identical_regime_level_offset": cross_level <= allowance}
        else:
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

    level_statistics = [
        {
            "level": node.level,
            "mean": float(np.mean(values[np.isfinite(values)])),
            "replica_seed": replica_seed,
            "score": score_name,
            "variance": float(np.var(values[np.isfinite(values)])),
        }
        for replica_seed in config.stream.model_seeds
        for node in nodes
        for score_name in GN_ALL_SCORE_NAMES
        for values in (gn_score_array(evaluations[replica_seed][node.node_id], score_name),)
        if np.any(np.isfinite(values))
    ]
    return (
        {
            "condition": condition,
            "exact_hessian_cholesky_successes": h_successes,
            "finite_gn_score_count": finite_gn_score_count,
            "focused_hessian_diagnostics": focused_hessian_rows,
            "focused_metrics": focused_rows,
            "gauss_newton_cholesky_successes": g_successes,
            "hessian_route_diagnostics": hessian_rows,
            "level_score_statistics": level_statistics,
            "passed": any(passed_by_score[score] for score in GN_CANDIDATE_SCORES),
            "passed_by_score": passed_by_score,
            "replica_metrics": replica_rows,
            "score_summaries": score_summaries,
            "stream_seed": stream_seed,
            "total_scored_states": total_evaluations,
        },
        raw,
    )


def run_gn_partial_carry_phase(
    config: VampLogTPcConfig,
    data: AuthenticatedPcData,
    selected: SelectedPcProtocol,
    phase_root: Path,
    passing_scores: tuple[str, ...],
) -> dict[str, object]:
    """Compare the block-28 parent with a de-novo twin using surviving GN scores."""
    summary_path = phase_root / "summary.json"
    if summary_path.is_file():
        return load_canonical_json(summary_path)
    if not passing_scores or any(score not in GN_CANDIDATE_SCORES for score in passing_scores):
        raise ValueError("partial carry requires at least one confirmed GN score")
    phase_root.mkdir(parents=True, exist_ok=True)
    stream = build_condition_stream(data.train, "recurrent_leaf_1_8", 0, config.stream.block_size)
    bank = _build_bank(config, selected, stream, phase_root / "normal", 28)
    nodes = bank.topology.active_nodes
    if tuple(node.level for node in nodes) != (4, 3, 2):
        raise RuntimeError("block-28 GN partial carry did not produce levels four, three, and two")
    parent = next(node for node in nodes if node.level == 2)
    if len(parent.parent_node_ids) != 2:
        raise RuntimeError("GN partial-carry parent has no two-child lineage")
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
    evaluations: dict[str, GnNodeReplicaEvaluation] = {}
    for node in nodes:
        model = load_pc_model(
            phase_root / "normal" / "models" / node.node_id / f"replica-{model_seed}",
            normal_backend,
            node.node_id,
            model_seed,
        )
        evaluations[node.node_id] = evaluate_gn_node_replica(
            normal_backend,
            model,
            holdout.table,
            _gn_evidence(config).negative_direction_epsilons,
        )
    twin_evaluation = evaluate_gn_node_replica(
        twin_backend,
        twin,
        holdout.table,
        _gn_evidence(config).negative_direction_epsilons,
    )
    parent_replicas = [evaluations[parent.node_id]]
    for replica_seed in config.stream.model_seeds[1:]:
        backend = make_backend(config, selected, replica_seed)
        model = load_pc_model(
            phase_root / "normal" / "models" / parent.node_id / f"replica-{replica_seed}",
            backend,
            parent.node_id,
            replica_seed,
        )
        parent_replicas.append(
            evaluate_gn_node_replica(
                backend,
                model,
                holdout.table,
                _gn_evidence(config).negative_direction_epsilons,
            )
        )
        del backend
        _release_jax_caches()
    examples = np.arange(len(holdout.table.labels))
    normal_logits = np.stack([evaluations[node.node_id].logits for node in nodes])
    twin_logits = normal_logits.copy()
    parent_index = nodes.index(parent)
    twin_logits[parent_index] = twin_evaluation.logits
    score_results: dict[str, object] = {}
    raw: dict[str, np.ndarray] = {
        "labels": holdout.table.labels,
        "node_levels": np.asarray([node.level for node in nodes], dtype=np.int64),
    }
    for score_name in passing_scores:
        normal_scores = np.stack(
            [gn_score_array(evaluations[node.node_id], score_name) for node in nodes]
        )
        twin_scores = normal_scores.copy()
        twin_scores[parent_index] = gn_score_array(twin_evaluation, score_name)
        normal_routes = np.argmax(normal_scores, axis=0)
        twin_routes = np.argmax(twin_scores, axis=0)
        normal_predictions = np.argmax(normal_logits[normal_routes, examples], axis=-1)
        twin_predictions = np.argmax(twin_logits[twin_routes, examples], axis=-1)
        normal_accuracy = float(np.mean(normal_predictions == holdout.table.labels))
        twin_accuracy = float(np.mean(twin_predictions == holdout.table.labels))
        normal_parent_scores = [gn_score_array(value, score_name) for value in parent_replicas]
        same_data_offsets = [
            abs(float(np.median(normal_parent_scores[left] - normal_parent_scores[right])))
            for left in range(len(normal_parent_scores))
            for right in range(left + 1, len(normal_parent_scores))
        ]
        typical_offset = float(np.median(same_data_offsets))
        parent_twin_offset = abs(
            float(
                np.median(
                    normal_parent_scores[0] - gn_score_array(twin_evaluation, score_name)
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
        raw[f"{score_name}_normal_scores"] = normal_scores
        raw[f"{score_name}_twin_scores"] = twin_scores
        raw[f"{score_name}_normal_routes"] = normal_routes
        raw[f"{score_name}_twin_routes"] = twin_routes
    scored_evaluations = [*evaluations.values(), twin_evaluation, *parent_replicas[1:]]
    total_scored_states = sum(len(value.evidence.map_log_evidence) for value in scored_evaluations)
    g_successes = sum(
        int(np.count_nonzero(value.evidence.gauss_newton_cholesky_succeeded))
        for value in scored_evaluations
    )
    negative_hessian_states = sum(
        int(np.count_nonzero(value.evidence.minimum_hessian_eigenvalue < 0.0))
        for value in scored_evaluations
    )
    finite_gn = all(
        np.all(np.isfinite(value.evidence.gn0_log_evidence))
        and np.all(np.isfinite(value.evidence.gn1_log_evidence))
        for value in scored_evaluations
    )
    counters = bank.counters.with_gauss_newton_scoring(
        len(holdout.table.labels),
        len(scored_evaluations),
        config.training.infer_steps,
        negative_hessian_states=negative_hessian_states,
        direction_epsilon_count=len(_gn_evidence(config).negative_direction_epsilons),
        active_models=bank.counters.active_pc_models,
    )
    require_pc_work_bound(
        bank.counters,
        28,
        config.stream.block_size,
        config.training.epochs,
        config.training.classifier_epochs,
        len(config.stream.model_seeds),
    )
    lifecycle_gates = {
        "children_absent_after_commit": all(
            not (phase_root / "normal" / "models" / child).exists()
            for child in parent.parent_node_ids
        ),
        "parent_checkpoint_committed": (phase_root / "normal" / "bank.json").is_file(),
        "work_bound": True,
    }
    numerical_gates = {
        "finite_gn_scores": finite_gn,
        "raw_gauss_newton_cholesky_every_state": g_successes == total_scored_states,
    }
    passed = bool(
        all(lifecycle_gates.values())
        and all(numerical_gates.values())
        and all(bool(value["passed"]) for value in score_results.values())
    )
    summary = {
        "gauss_newton_cholesky_successes": g_successes,
        "lifecycle_gates": lifecycle_gates,
        "negative_hessian_states": negative_hessian_states,
        "numerical_gates": numerical_gates,
        "parent_node_id": parent.node_id,
        "passed": passed,
        "schema_version": "vamp-logt-pc-gn-partial-carry-v1",
        "score_results": score_results,
        "total_scored_states": total_scored_states,
        "twin_seed": twin_seed,
        "twin_training": {
            "classifier_example_presentations": twin.classifier_example_presentations,
            "density_example_presentations": twin.density_example_presentations,
        },
        "work_counters": asdict(counters),
    }
    _publish_npz(phase_root / "raw_scores.npz", raw)
    publish_immutable_json(summary_path, summary)
    return summary


def _finish_gn_report(
    run_root: Path,
    config: VampLogTPcConfig,
    phases: dict[str, dict[str, object]],
) -> Path:
    from apm.experiments.vamp_logt_pc_gn_reporting import write_gn_result_report

    write_gn_result_report(run_root, config, phases)
    return run_root


def _publish_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    publish_immutable_bytes(path, buffer.getvalue())


def _gn_evidence(
    config: VampLogTPcConfig,
) -> PcGaussNewtonEvidenceConfig | PcGaussNewtonContinuationEvidenceConfig:
    if not isinstance(config.evidence, PcGaussNewtonEvidenceConfig):
        raise ValueError("the GN workflow requires the generalized-Gauss-Newton evidence config")
    return config.evidence


__all__ = [
    "authenticate_and_import_map_source",
    "map_source_tree_sha256",
    "required_map_source_files",
    "run_gn_analytic_phase",
    "run_gn_partial_carry_phase",
    "run_gn_source_preflight",
    "run_gn_static_phase",
    "run_gn_workflow",
]

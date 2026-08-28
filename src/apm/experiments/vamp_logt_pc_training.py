"""Fixed-budget node fitting and classifier evaluation for PC evidence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from apm.continual.logt_evidence_bank import TemporalNode
from apm.experiments.vamp_logt_pc_config import VampLogTPcConfig
from apm.experiments.vamp_logt_pc_data import PcRawTable
from apm.models.fabricpc_density_backend import (
    FabricPcDensityBackend,
    PcClassifierConfig,
    PcDensityConfig,
    PcDensityTrainConfig,
    PcGaussNewtonScores,
    StoredPcModel,
    classifier_cross_entropy,
    classifier_logits,
    fit_classifier,
    load_pc_model,
    publish_pc_model,
)


SCORE_NAMES = ("map",)
GN_ROUTING_SCORE_NAMES = ("map", "gn0", "gn1")
GN_ALL_SCORE_NAMES = ("map", "h_laplace", "gn0", "gn1")


@dataclass(frozen=True, slots=True)
class SelectedPcProtocol:
    """Preflight-selected global choices frozen before static routing."""

    image_precision: float
    hidden_precision: float
    inference_step_size: float


@dataclass(frozen=True, slots=True)
class NodeReplicaEvaluation:
    """Raw evidence and node-local task outputs for one candidate model."""

    map_joint_scores: np.ndarray
    logits: np.ndarray
    cross_entropy: np.ndarray
    predictions: np.ndarray


@dataclass(frozen=True, slots=True)
class GnNodeReplicaEvaluation:
    """Paired evidence scores, curvature diagnostics, and task outputs."""

    evidence: PcGaussNewtonScores
    logits: np.ndarray
    cross_entropy: np.ndarray
    predictions: np.ndarray


def make_backend(
    config: VampLogTPcConfig,
    selected: SelectedPcProtocol,
    seed: int,
) -> FabricPcDensityBackend:
    """Construct the exact fixed graph for one independent replica."""
    return FabricPcDensityBackend(
        PcDensityConfig(
            config.model.latent_dim,
            config.model.hidden_dim,
            config.model.image_dim,
            selected.hidden_precision,
            selected.image_precision,
            config.model.weight_init_std,
        ),
        PcDensityTrainConfig(
            seed,
            config.training.epochs,
            config.training.batch_size,
            config.training.learning_rate,
            config.training.weight_decay,
            config.training.infer_steps,
            selected.inference_step_size,
            config.training.score_batch_size,
            0.0,
            config.runtime.progress,
        ),
    )


def classifier_config(config: VampLogTPcConfig) -> PcClassifierConfig:
    """Return the fixed stopped-gradient head schedule."""
    return PcClassifierConfig(
        config.training.classifier_epochs,
        config.training.classifier_batch_size,
        config.training.classifier_learning_rate,
        config.training.classifier_weight_decay,
        10,
    )


def fit_or_load_node_replica(
    config: VampLogTPcConfig,
    stream: PcRawTable,
    node: TemporalNode,
    replica_seed: int,
    models_root: Path,
    backend: FabricPcDensityBackend,
) -> StoredPcModel:
    """Fit one node de novo from its exact interval, or authenticate its artifact."""
    if backend.train_config.seed != replica_seed:
        raise ValueError("PC backend seed does not match the requested replica")
    directory = models_root / node.node_id / f"replica-{replica_seed}"
    if (directory / "model.npz").is_file() and (directory / "manifest.json").is_file():
        return load_pc_model(directory, backend, node.node_id, replica_seed)
    rows = np.asarray(node.example_ids, dtype=np.int64)
    table = stream.select(rows)
    density = backend.fit(table.images_float32, replica_seed)
    settled = backend.settle_images(density.params, table.images_float32)
    classifier = fit_classifier(
        settled.hidden,
        table.labels,
        replica_seed,
        classifier_config(config),
    )
    return publish_pc_model(
        directory,
        backend,
        node.node_id,
        replica_seed,
        density,
        classifier,
    )


def evaluate_node_replica(
    backend: FabricPcDensityBackend,
    model: StoredPcModel,
    table: PcRawTable,
) -> NodeReplicaEvaluation:
    """Evaluate one candidate using images for evidence and labels only afterward."""
    images = table.images_float32
    settled = backend.settle_images(model.params, images)
    map_joint_scores = backend.map_joint_scores_from_settled(model.params, images, settled)
    logits = classifier_logits(model.classifier, settled.hidden)
    cross_entropy = classifier_cross_entropy(model.classifier, settled.hidden, table.labels)
    return NodeReplicaEvaluation(
        map_joint_scores,
        logits,
        cross_entropy,
        np.argmax(logits, axis=-1).astype(np.int64),
    )


def evaluate_gn_node_replica(
    backend: FabricPcDensityBackend,
    model: StoredPcModel,
    table: PcRawTable,
    negative_direction_epsilons: tuple[float, ...],
) -> GnNodeReplicaEvaluation:
    """Settle once and evaluate every paired score plus the classifier."""
    settled, evidence = backend.settle_and_score_gauss_newton(
        model.params,
        table.images_float32,
        negative_direction_epsilons,
    )
    logits = classifier_logits(model.classifier, settled.hidden)
    cross_entropy = classifier_cross_entropy(model.classifier, settled.hidden, table.labels)
    return GnNodeReplicaEvaluation(
        evidence,
        logits,
        cross_entropy,
        np.argmax(logits, axis=-1).astype(np.int64),
    )


def score_array(evaluation: NodeReplicaEvaluation, name: str) -> np.ndarray:
    """Return the sole active MAP score vector by its protocol name."""
    if name == "map":
        return evaluation.map_joint_scores
    raise ValueError(f"unknown PC evidence score: {name}")


def gn_score_array(evaluation: GnNodeReplicaEvaluation, name: str) -> np.ndarray:
    """Return one paired score vector by its protocol name."""
    if name == "map":
        return evaluation.evidence.map_log_evidence
    if name == "h_laplace":
        return evaluation.evidence.hessian_laplace_log_evidence
    if name == "gn0":
        return evaluation.evidence.gn0_log_evidence
    if name == "gn1":
        return evaluation.evidence.gn1_log_evidence
    raise ValueError(f"unknown Gauss-Newton evidence score: {name}")


__all__ = [
    "NodeReplicaEvaluation",
    "GnNodeReplicaEvaluation",
    "GN_ALL_SCORE_NAMES",
    "GN_ROUTING_SCORE_NAMES",
    "SCORE_NAMES",
    "SelectedPcProtocol",
    "classifier_config",
    "evaluate_node_replica",
    "evaluate_gn_node_replica",
    "fit_or_load_node_replica",
    "make_backend",
    "gn_score_array",
    "score_array",
]

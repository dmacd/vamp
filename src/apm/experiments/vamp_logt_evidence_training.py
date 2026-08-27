"""De-novo node training and label-isolated evidence-routing evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
from collections.abc import Mapping

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from apm.continual.logt_evidence_bank import TemporalNode
from apm.continual.nce_tre_evidence import (
    ConditionalEvidenceCNN,
    EvidenceTrainingConfig,
    EvidenceTrainingResult,
    balanced_nce_loss,
    evidence_scores,
    sample_reference_images,
    sample_discrete_waymark,
    train_evidence_cnn,
)
from apm.continual.top_two_adapter import (
    TopTwoAdapterState,
    TopTwoBaseState,
    TopTwoOptimizerConfig,
    top_two_base_state,
    top_two_logits,
    train_top_two_adapter_step,
    zero_top_two_adapter,
    zero_top_two_adamw,
)
from apm.experiments.vamp_logt_evidence_config import (
    AdapterConfig,
    EvidenceConfig,
)
from apm.experiments.vamp_logt_evidence_data import NodeHoldout, RawFeatureTable


@dataclass(frozen=True, slots=True)
class AdapterTrainingResult:
    """One fresh frozen node adapter and its exact replay accounting."""

    adapter: TopTwoAdapterState
    final_loss: float
    example_updates: int

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.final_loss)
            or self.example_updates < 1
            or any(parameter.requires_grad for parameter in self.adapter.tensors)
        ):
            raise ValueError("invalid completed node-adapter training result")


@dataclass(frozen=True, slots=True)
class BridgeDiagnostic:
    """Held-out overlap and balanced NCE loss for one adjacent waymark pair."""

    bridge_index: int
    balanced_accuracy: float
    balanced_loss: float


@dataclass(frozen=True, slots=True)
class EvidenceScoreMatrix:
    """Raw comparable node evidence with an explicit deterministic column order."""

    node_ids: tuple[str, ...]
    scores: Tensor

    def __post_init__(self) -> None:
        if (
            not self.node_ids
            or len(set(self.node_ids)) != len(self.node_ids)
            or self.scores.ndim != 2
            or self.scores.shape[1] != len(self.node_ids)
            or not torch.isfinite(self.scores).all()
        ):
            raise ValueError("invalid active-node evidence score matrix")


@dataclass(frozen=True, slots=True)
class RoutingEvaluation:
    """Classifier, oracle, latent-source, regret, and per-level routing metrics."""

    routed_accuracy: float
    oracle_accuracy: float
    route_oracle_agreement: float
    routing_regret_nats: float
    exact_source_accuracy: float | None
    equivalent_source_accuracy: float | None
    routed_node_ids: tuple[str, ...]
    oracle_node_ids: tuple[str, ...]
    route_counts: tuple[tuple[str, int], ...]
    node_mean_scores: tuple[tuple[str, float], ...]
    level_rows: tuple[dict[str, object], ...]

    def as_record(self) -> dict[str, object]:
        """Return JSON-compatible metrics without dropping condition definitions."""
        return {
            "equivalent_source_accuracy": self.equivalent_source_accuracy,
            "exact_source_accuracy": self.exact_source_accuracy,
            "level_rows": list(self.level_rows),
            "node_mean_scores": dict(self.node_mean_scores),
            "oracle_accuracy": self.oracle_accuracy,
            "route_counts": dict(self.route_counts),
            "route_oracle_agreement": self.route_oracle_agreement,
            "routed_accuracy": self.routed_accuracy,
            "routing_regret_nats": self.routing_regret_nats,
        }


def protocol_seed(seed: int, *parts: object) -> int:
    """Derive a stable unsigned 63-bit seed from semantic protocol coordinates."""
    if seed < 0:
        raise ValueError("protocol seeds must be nonnegative")
    payload = "\0".join((str(seed), *(str(part) for part in parts))).encode("utf-8")
    return int.from_bytes(sha256(payload).digest()[:8], "big") % (2**63 - 1)


def evidence_training_config(config: EvidenceConfig, bridges: int) -> EvidenceTrainingConfig:
    """Project the resolved experiment configuration onto one evidence model fit."""
    return EvidenceTrainingConfig(
        bridges=bridges,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        beta1=config.beta1,
        beta2=config.beta2,
        epsilon=config.epsilon,
        initial_replacement_probability=config.initial_replacement_probability,
    )


def train_node_adapter(
    trunk_features: Tensor,
    labels: Tensor,
    base: TopTwoBaseState,
    config: AdapterConfig,
    seed: int,
    device: torch.device,
    description: str,
    show_progress: bool,
) -> AdapterTrainingResult:
    """Train a full-rank top-two adapter de novo on exactly one node replay union."""
    if trunk_features.shape[0] < 2 or labels.shape != (trunk_features.shape[0],):
        raise ValueError("node adapter replay arrays are incompatible")
    try:
        from tqdm.auto import tqdm
    except ImportError as error:  # pragma: no cover - vision environment gate
        raise RuntimeError("tqdm is required by the vision environment") from error
    target_base = _base_to(base, device)
    adapter = zero_top_two_adapter(target_base)
    optimizer = zero_top_two_adamw(adapter)
    fixed = zero_top_two_adapter(target_base)
    optimizer_config = TopTwoOptimizerConfig(
        config.learning_rate,
        config.weight_decay,
        config.beta1,
        config.beta2,
        config.epsilon,
    )
    final_loss = math.nan
    for epoch in tqdm(
        range(config.epochs),
        desc=description,
        disable=not show_progress,
        leave=False,
    ):
        order = np.random.default_rng(protocol_seed(seed, "adapter", epoch)).permutation(len(labels))
        for offset in range(0, len(order), config.batch_size):
            ids = torch.from_numpy(order[offset : offset + config.batch_size].astype(np.int64))
            adapter, optimizer, final_loss = train_top_two_adapter_step(
                trunk_features[ids].to(device),
                labels[ids].to(device),
                target_base,
                fixed,
                adapter,
                optimizer,
                optimizer_config,
            )
    return AdapterTrainingResult(
        _adapter_to(adapter, torch.device("cpu")),
        final_loss,
        config.epochs * len(labels),
    )


def train_node_evidence(
    raw_images: Tensor,
    reference_raw_images: Tensor | None,
    config: EvidenceConfig,
    bridges: int,
    seed: int,
    device: torch.device,
    show_progress: bool,
) -> EvidenceTrainingResult:
    """Train a fresh full-capacity conditional CNN without accepting labels or features."""
    return train_evidence_cnn(
        raw_images,
        reference_raw_images,
        evidence_training_config(config, bridges),
        seed,
        device,
        show_progress,
    )


def bridge_diagnostics(
    model: ConditionalEvidenceCNN,
    raw_images: Tensor,
    reference_raw_images: Tensor | None,
    config: EvidenceTrainingConfig,
    seed: int,
    device: torch.device,
) -> tuple[BridgeDiagnostic, ...]:
    """Measure held-out adjacent-waymark accuracy and NCE loss for every bridge."""
    if raw_images.dtype != torch.uint8 or len(raw_images) < 2:
        raise ValueError("bridge diagnostics require held-out uint8 source images")
    if reference_raw_images is not None and (
        reference_raw_images.dtype != torch.uint8
        or reference_raw_images.ndim != 4
        or reference_raw_images.shape[1:] != (1, 28, 28)
        or len(reference_raw_images) < 2
    ):
        raise ValueError("bridge diagnostics require an empirical uint8 reference bank")
    sources = raw_images.to(device)
    reference = (
        None
        if reference_raw_images is None
        else reference_raw_images.cpu()
    )
    target = model.to(device)
    target.eval()
    rows = []
    with torch.inference_mode():
        for bridge_index in range(config.bridges):
            generator = torch.Generator(device=device)
            generator.manual_seed(protocol_seed(seed, "bridge", bridge_index))
            indices = torch.full(
                (len(sources),), bridge_index, dtype=torch.int64, device=device
            )
            negatives = sources[
                torch.randperm(len(sources), device=device, generator=generator)
            ]
            reference_images = sample_reference_images(
                reference,
                len(sources),
                device,
                generator,
            )
            positives = sample_discrete_waymark(
                sources,
                reference_images,
                indices,
                config.bridges,
                config.initial_replacement_probability,
                generator,
            )
            negative_waymarks = sample_discrete_waymark(
                negatives,
                reference_images,
                indices + 1,
                config.bridges,
                config.initial_replacement_probability,
                generator,
            )
            positive_logits = target(positives, indices)
            negative_logits = target(negative_waymarks, indices)
            accuracy = 0.5 * (
                float((positive_logits >= 0.0).float().mean().item())
                + float((negative_logits < 0.0).float().mean().item())
            )
            rows.append(
                BridgeDiagnostic(
                    bridge_index,
                    accuracy,
                    float(balanced_nce_loss(positive_logits, negative_logits).item()),
                )
            )
    model.cpu()
    return tuple(rows)


def score_evidence_bank(
    nodes: tuple[TemporalNode, ...],
    models: Mapping[str, ConditionalEvidenceCNN],
    raw_images: Tensor,
    device: torch.device,
    batch_size: int,
) -> EvidenceScoreMatrix:
    """Evaluate each active node once per query and retain chronological columns."""
    node_ids = tuple(node.node_id for node in nodes)
    if set(models) != set(node_ids):
        raise ValueError("evidence models do not exactly cover the active LogT nodes")
    columns = tuple(
        evidence_scores(models[node_id], raw_images, device, batch_size)
        for node_id in node_ids
    )
    return EvidenceScoreMatrix(node_ids, torch.stack(columns, dim=1))


def evaluate_routing(
    nodes: tuple[TemporalNode, ...],
    adapters: Mapping[str, TopTwoAdapterState],
    evidence: EvidenceScoreMatrix,
    table: RawFeatureTable,
    base: TopTwoBaseState,
    device: torch.device,
    batch_size: int,
    holdout: NodeHoldout | None = None,
    node_equivalence_keys: Mapping[str, tuple[int, ...]] | None = None,
) -> RoutingEvaluation:
    """Compare evidence routing with a label-aware minimum-loss node oracle."""
    node_ids = tuple(node.node_id for node in nodes)
    if (
        evidence.node_ids != node_ids
        or evidence.scores.shape[0] != len(table.labels)
        or set(adapters) != set(node_ids)
        or (holdout is not None and holdout.table is not table)
    ):
        raise ValueError("routing inputs do not describe one aligned active bank")
    logits = torch.stack(
        tuple(
            _adapter_logits(adapters[node_id], base, table.trunk_features, device, batch_size)
            for node_id in node_ids
        ),
        dim=1,
    )
    labels = table.labels
    loss_matrix = -F.log_softmax(logits, dim=2).gather(
        2,
        labels[:, None, None].expand(-1, len(node_ids), 1),
    ).squeeze(2)
    routed = evidence.scores.argmax(dim=1)
    oracle = loss_matrix.argmin(dim=1)
    rows = torch.arange(len(labels))
    routed_predictions = logits[rows, routed].argmax(dim=1)
    oracle_predictions = logits[rows, oracle].argmax(dim=1)
    routed_ids = tuple(node_ids[int(index)] for index in routed.tolist())
    oracle_ids = tuple(node_ids[int(index)] for index in oracle.tolist())
    exact_source_accuracy: float | None = None
    equivalent_source_accuracy: float | None = None
    if holdout is not None:
        exact_source_accuracy = float(
            np.mean([actual == expected for actual, expected in zip(routed_ids, holdout.source_node_ids)])
        )
        if node_equivalence_keys is None or set(node_equivalence_keys) != set(node_ids):
            raise ValueError("source-equivalent routing requires every active node key")
        equivalent_source_accuracy = float(
            np.mean(
                [
                    node_equivalence_keys[actual] == expected
                    for actual, expected in zip(routed_ids, holdout.equivalent_source_keys)
                ]
            )
        )
    node_levels = {node.node_id: node.level for node in nodes}
    level_rows = tuple(
        _level_metrics(
            level,
            node_ids,
            node_levels,
            routed_ids,
            oracle_ids,
            holdout.source_node_ids if holdout is not None else None,
            routed_predictions,
            labels,
            evidence.scores,
        )
        for level in sorted(set(node_levels.values()))
    )
    return RoutingEvaluation(
        float((routed_predictions == labels).float().mean().item()),
        float((oracle_predictions == labels).float().mean().item()),
        float((routed == oracle).float().mean().item()),
        float((loss_matrix[rows, routed] - loss_matrix[rows, oracle]).mean().item()),
        exact_source_accuracy,
        equivalent_source_accuracy,
        routed_ids,
        oracle_ids,
        tuple((node_id, routed_ids.count(node_id)) for node_id in node_ids),
        tuple(
            (node_id, float(evidence.scores[:, column].mean().item()))
            for column, node_id in enumerate(node_ids)
        ),
        level_rows,
    )


def _level_metrics(
    level: int,
    node_ids: tuple[str, ...],
    node_levels: Mapping[str, int],
    routed_ids: tuple[str, ...],
    oracle_ids: tuple[str, ...],
    source_ids: tuple[str, ...] | None,
    predictions: Tensor,
    labels: Tensor,
    scores: Tensor,
) -> dict[str, object]:
    source_mask = (
        torch.tensor([node_levels[node_id] == level for node_id in source_ids])
        if source_ids is not None
        else torch.ones(len(labels), dtype=torch.bool)
    )
    level_columns = [index for index, node_id in enumerate(node_ids) if node_levels[node_id] == level]
    source_examples = int(source_mask.sum().item())
    if source_examples == 0:
        return {
            "classifier_accuracy": None,
            "level": level,
            "mean_evidence_score": None,
            "oracle_agreement": None,
            "source_examples": 0,
        }
    return {
        "classifier_accuracy": float((predictions[source_mask] == labels[source_mask]).float().mean().item()),
        "level": level,
        "mean_evidence_score": float(scores[source_mask][:, level_columns].mean().item()),
        "oracle_agreement": float(
            np.mean(
                [
                    routed == oracle
                    for routed, oracle, included in zip(
                        routed_ids, oracle_ids, source_mask.tolist()
                    )
                    if included
                ]
            )
        ),
        "source_examples": source_examples,
    }


def _adapter_logits(
    adapter: TopTwoAdapterState,
    base: TopTwoBaseState,
    trunk_features: Tensor,
    device: torch.device,
    batch_size: int,
) -> Tensor:
    target_base = _base_to(base, device)
    target_adapter = _adapter_to(adapter, device)
    rows = []
    with torch.inference_mode():
        for offset in range(0, len(trunk_features), batch_size):
            rows.append(
                top_two_logits(
                    trunk_features[offset : offset + batch_size].to(device),
                    target_base,
                    target_adapter,
                ).cpu()
            )
    return torch.cat(rows)


def _base_to(base: TopTwoBaseState, device: torch.device) -> TopTwoBaseState:
    return top_two_base_state(*base.tensors, device=device)


def _adapter_to(adapter: TopTwoAdapterState, device: torch.device) -> TopTwoAdapterState:
    return TopTwoAdapterState(*(tensor.detach().to(device).clone() for tensor in adapter.tensors))


__all__ = [
    "AdapterTrainingResult",
    "BridgeDiagnostic",
    "EvidenceScoreMatrix",
    "RoutingEvaluation",
    "bridge_diagnostics",
    "evaluate_routing",
    "evidence_training_config",
    "protocol_seed",
    "score_evidence_bank",
    "train_node_adapter",
    "train_node_evidence",
]

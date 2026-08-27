"""Normalized multimodal calibration for additive NCE/TRE evidence offsets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np
import torch
from torch import Tensor

from apm.continual.artifacts import ChainedJsonlLedger, load_canonical_json, publish_immutable_json
from apm.continual.nce_tre_evidence import ConditionalVectorEvidence, balanced_nce_loss
from apm.continual.vision.imagenetr.checkpoints import atomic_torch_save
from apm.experiments.vamp_logt_evidence_config import CalibrationConfig
from apm.experiments.vamp_logt_evidence_training import protocol_seed


@dataclass(frozen=True, slots=True)
class CalibrationModelResult:
    """One trained conditional ratio model and its held-out ratio diagnostics."""

    condition: str
    replica: int
    model: ConditionalVectorEvidence
    predictions: Tensor
    adjacent_predictions: Tensor
    adjacent_truth: Tensor
    signed_bias_nats: float
    rmse_nats: float
    maximum_adjacent_accuracy: float


def run_ratio_calibration(
    config: CalibrationConfig,
    directory: Path,
    device: torch.device,
    show_progress: bool,
) -> dict[str, object]:
    """Train/reuse direct and TRE estimators and apply all normalized ratio gates."""
    summary_path = directory / "summary.json"
    if summary_path.is_file():
        summary = load_canonical_json(summary_path)
        if summary.get("schema_version") != "vamp-logt-ratio-calibration-v1":
            raise ValueError("ratio-calibration schema changed inside one run identity")
        return summary
    directory.mkdir(parents=True, exist_ok=True)
    evaluation, truth = _evaluation_sample(config, device)
    ledger = ChainedJsonlLedger(directory / "metrics.jsonl", "vamp-logt-calibration-metric-v1")
    results = tuple(
        _fit_and_evaluate(
            condition,
            1 if condition == "direct_nce" else config.tre_bridges,
            replica,
            config,
            directory,
            evaluation,
            truth,
            device,
            show_progress,
        )
        for condition in ("direct_nce", "tre")
        for replica in range(config.replicas)
    )
    ledger.append_many(
        {
            "condition": result.condition,
            "maximum_adjacent_balanced_accuracy": result.maximum_adjacent_accuracy,
            "replica": result.replica,
            "rmse_nats": result.rmse_nats,
            "signed_bias_nats": result.signed_bias_nats,
        }
        for result in results
    )
    direct = tuple(result for result in results if result.condition == "direct_nce")
    tre = tuple(result for result in results if result.condition == "tre")
    direct_rmse = float(np.mean([result.rmse_nats for result in direct]))
    tre_rmse = float(np.mean([result.rmse_nats for result in tre]))
    tre_bias = max(abs(result.signed_bias_nats) for result in tre)
    interseed_rmse = max(
        _rmse(left.predictions, right.predictions)
        for index, left in enumerate(tre)
        for right in tre[index + 1 :]
    )
    triangle_slack = min(
        float(
            (
                result.adjacent_predictions.sub(result.adjacent_truth).abs().sum(dim=1)
                - result.predictions.sub(truth.cpu()).abs()
            ).min().item()
        )
        for result in tre
    )
    gates = {
        "additive_offset_recovered": tre_bias <= config.signed_bias_max_nats,
        "independent_models_agree": interseed_rmse <= config.interseed_rmse_max_nats,
        "summed_ratio_error_obeys_triangle_bound": triangle_slack >= -1.0e-5,
        "tre_ratio_rmse_is_acceptable": tre_rmse <= config.tre_rmse_max_nats,
        "tre_succeeds_where_direct_nce_saturates": direct_rmse
        >= config.direct_to_tre_rmse_ratio_min * tre_rmse,
    }
    summary: dict[str, object] = {
        "condition_definitions": {
            "direct_nce": (
                "Direct NCE trains one balanced classifier to distinguish the normalized "
                "bimodal data distribution from the normalized Bernoulli reference."
            ),
            "tre": (
                "TRE trains balanced classifiers between consecutive normalized waymarks "
                "and sums their logits to estimate the same data-to-reference log ratio."
            ),
        },
        "direct_mean_rmse_nats": direct_rmse,
        "gates": gates,
        "interseed_tre_rmse_nats": interseed_rmse,
        "passed": all(gates.values()),
        "schema_version": "vamp-logt-ratio-calibration-v1",
        "tre_maximum_absolute_signed_bias_nats": tre_bias,
        "tre_mean_rmse_nats": tre_rmse,
        "triangle_minimum_slack_nats": triangle_slack,
    }
    publish_immutable_json(summary_path, summary)
    return summary


def _fit_and_evaluate(
    condition: str,
    bridges: int,
    replica: int,
    config: CalibrationConfig,
    directory: Path,
    evaluation: Tensor,
    truth: Tensor,
    device: torch.device,
    show_progress: bool,
) -> CalibrationModelResult:
    model_path = directory / "models" / condition / f"replica-{replica}.pt"
    model = ConditionalVectorEvidence(config.dimensions, bridges).to(device)
    if model_path.is_file():
        payload = torch.load(model_path, map_location=device, weights_only=True)
        if payload.get("bridges") != bridges or payload.get("schema_version") != "vamp-logt-calibration-model-v1":
            raise ValueError("calibration model metadata changed")
        model.load_state_dict(payload["state_dict"], strict=True)
    else:
        model = _train_model(
            model,
            bridges,
            config,
            protocol_seed(0, "calibration", condition, replica),
            device,
            show_progress,
            f"calibration {condition} replica {replica + 1}/{config.replicas}",
        )
        model_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_torch_save(
            model_path,
            {
                "bridges": bridges,
                "schema_version": "vamp-logt-calibration-model-v1",
                "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
            },
        )
    model.eval()
    with torch.inference_mode():
        adjacent = torch.stack(
            tuple(
                model(
                    evaluation,
                    torch.full(
                        (len(evaluation),), bridge, dtype=torch.int64, device=device
                    ),
                )
                for bridge in range(bridges)
            ),
            dim=1,
        )
    adjacent_truth = torch.stack(
        tuple(
            _exact_log_density(evaluation, (bridge / bridges), config)
            - _exact_log_density(evaluation, ((bridge + 1) / bridges), config)
            for bridge in range(bridges)
        ),
        dim=1,
    )
    predictions = adjacent.sum(dim=1)
    signed_bias = float((predictions - truth).mean().item())
    rmse = _rmse(predictions, truth)
    adjacent_accuracies = _adjacent_accuracies(model, bridges, config, replica, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return CalibrationModelResult(
        condition,
        replica,
        model.cpu(),
        predictions.cpu(),
        adjacent.cpu(),
        adjacent_truth.cpu(),
        signed_bias,
        rmse,
        max(adjacent_accuracies),
    )


def _train_model(
    model: ConditionalVectorEvidence,
    bridges: int,
    config: CalibrationConfig,
    seed: int,
    device: torch.device,
    show_progress: bool,
    description: str,
) -> ConditionalVectorEvidence:
    try:
        from tqdm.auto import tqdm
    except ImportError as error:  # pragma: no cover - vision environment gate
        raise RuntimeError("tqdm is required by the vision environment") from error
    torch.manual_seed(seed)
    model.apply(_reset_parameters)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 1)
    model.train()
    for _step in tqdm(
        range(config.training_steps),
        desc=description,
        disable=not show_progress,
        leave=False,
    ):
        bridge_indices = torch.arange(
            bridges,
            dtype=torch.int64,
            device=device,
        ).repeat_interleave(config.batch_size)
        positives = _sample_waymarks(
            bridge_indices.to(torch.float32) / bridges,
            config,
            generator,
            device,
        )
        negatives = _sample_waymarks(
            (bridge_indices.to(torch.float32) + 1.0) / bridges,
            config,
            generator,
            device,
        )
        optimizer.zero_grad(set_to_none=True)
        loss = balanced_nce_loss(
            model(positives, bridge_indices),
            model(negatives, bridge_indices),
        )
        loss.backward()
        optimizer.step()
    return model


def _reset_parameters(module: torch.nn.Module) -> None:
    reset = getattr(module, "reset_parameters", None)
    if callable(reset):
        reset()


def _evaluation_sample(
    config: CalibrationConfig,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(protocol_seed(0, "calibration", "evaluation"))
    half = config.evaluation_examples // 2
    data = _sample_waymarks(
        torch.zeros(half, device=device), config, generator, device
    )
    reference = _sample_waymarks(
        torch.ones(config.evaluation_examples - half, device=device),
        config,
        generator,
        device,
    )
    values = torch.cat((data, reference))
    truth = _exact_log_density(values, 0.0, config) - _exact_log_density(values, 1.0, config)
    return values, truth


def _sample_waymarks(
    alphas: Tensor,
    config: CalibrationConfig,
    generator: torch.Generator,
    device: torch.device,
) -> Tensor:
    if alphas.ndim != 1 or bool((alphas < 0.0).any()) or bool((alphas > 1.0).any()):
        raise ValueError("calibration waymark alphas must be a probability vector")
    low, high = config.component_probabilities
    components = torch.randint(
        0, 2, (len(alphas),), generator=generator, device=device
    ).to(torch.float32)
    data_probabilities = low + (high - low) * components
    probabilities = (
        (1.0 - alphas) * data_probabilities + alphas * config.reference_probability
    )[:, None]
    values = (
        torch.rand(
            (len(alphas), config.dimensions), generator=generator, device=device
        )
        < probabilities
    ).to(torch.float32)
    # The count is sufficient for every member of this exchangeable family.  Sorting
    # removes irrelevant coordinate-order variance; the binomial multiplicity cancels
    # from each adjacent density ratio.
    return torch.sort(values, dim=1).values


def _exact_log_density(
    values: Tensor,
    alpha: float,
    config: CalibrationConfig,
) -> Tensor:
    low, high = config.component_probabilities
    probabilities = tuple(
        (1.0 - alpha) * probability + alpha * config.reference_probability
        for probability in (low, high)
    )
    component_rows = tuple(
        (
            values * math.log(probability)
            + (1.0 - values) * math.log1p(-probability)
        ).sum(dim=1)
        + math.log(0.5)
        for probability in probabilities
    )
    return torch.logsumexp(torch.stack(component_rows, dim=1), dim=1)


def _adjacent_accuracies(
    model: ConditionalVectorEvidence,
    bridges: int,
    config: CalibrationConfig,
    replica: int,
    device: torch.device,
) -> tuple[float, ...]:
    generator = torch.Generator(device=device)
    generator.manual_seed(protocol_seed(0, "calibration", "diagnostic", replica, bridges))
    rows = []
    with torch.inference_mode():
        for bridge in range(bridges):
            indices = torch.full(
                (config.batch_size,), bridge, dtype=torch.int64, device=device
            )
            positives = _sample_waymarks(
                indices.to(torch.float32) / bridges, config, generator, device
            )
            negatives = _sample_waymarks(
                (indices.to(torch.float32) + 1.0) / bridges,
                config,
                generator,
                device,
            )
            positive_logits = model(positives, indices)
            negative_logits = model(negatives, indices)
            rows.append(
                0.5
                * (
                    float((positive_logits >= 0.0).float().mean().item())
                    + float((negative_logits < 0.0).float().mean().item())
                )
            )
    return tuple(rows)


def _rmse(left: Tensor, right: Tensor) -> float:
    return float(torch.sqrt(torch.mean((left - right).square())).item())


__all__ = ["CalibrationModelResult", "run_ratio_calibration"]

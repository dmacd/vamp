"""Independent float64 probability and fixed-cohort checkpoint diagnostics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp

from apm.continual.artifacts import record_sha256


def fixed_folds(image_ids: Sequence[str], labels: Sequence[int], count: int = 5) -> tuple[int, ...]:
    """Hash-stratify each class, balancing fold sizes without using predictions."""
    if len(image_ids) != len(labels) or len(set(image_ids)) != len(image_ids) or count < 2:
        raise ValueError("folds require unique aligned image identities")
    classes = tuple(sorted(set(labels)))
    groups = tuple(tuple(sorted((name for name, label in zip(image_ids, labels, strict=True) if label == target),
                                key=lambda name: record_sha256(["checkpoint-diagnostic-fold-v1", name])))
                   for target in classes)
    if any(len(group) < count for group in groups):
        raise ValueError("each class must represent every diagnostic fold")
    assignments = {name: (position + sum(map(len, groups[:index]))) % count
                   for index, group in enumerate(groups) for position, name in enumerate(group)}
    return tuple(assignments[name] for name in image_ids)


def require_logits(logits: np.ndarray, labels: np.ndarray) -> None:
    """Reject malformed, nonfinite, or misaligned probability inputs."""
    if (logits.ndim != 2 or logits.shape[1] < 2 or labels.shape != (len(logits),) or not len(logits)
            or not np.isfinite(logits).all() or not np.issubdtype(labels.dtype, np.integer)
            or np.any(labels < 0) or np.any(labels >= logits.shape[1])):
        raise ValueError("invalid checkpoint logits or labels")


def score_rows(logits: np.ndarray, labels: np.ndarray, temperature: float | np.ndarray = 1.) -> tuple[dict[str, object], ...]:
    """Compute natural-log loss, probabilities, Brier score, and margins in float64."""
    logits, labels = np.asarray(logits, dtype=np.float64), np.asarray(labels)
    require_logits(logits, labels)
    temperatures = np.broadcast_to(np.asarray(temperature, dtype=np.float64), (len(logits),))
    if not np.isfinite(temperatures).all() or np.any(temperatures <= 0):
        raise ValueError("temperature must be finite and positive")
    scaled = logits / temperatures[:, None]
    log_probabilities = scaled - logsumexp(scaled, axis=1, keepdims=True)
    probabilities = np.exp(log_probabilities)
    predicted = np.argmax(logits, axis=1)
    if not np.array_equal(predicted, np.argmax(scaled, axis=1)):
        raise ValueError("temperature changed the winning class")
    indices = np.arange(len(labels))
    true_probability = probabilities[indices, labels]
    other_max = np.max(np.where(np.arange(logits.shape[1])[None, :] == labels[:, None], -np.inf, logits), axis=1)
    return tuple({"prediction": int(predicted[index]), "correct": bool(predicted[index] == label),
                  "nll": float(-log_probabilities[index, label]), "true_probability": float(true_probability[index]),
                  "top1_probability": float(probabilities[index, predicted[index]]),
                  "entropy": float(-np.sum(probabilities[index] * log_probabilities[index])),
                  "brier": float(np.sum(probabilities[index] ** 2) - 2 * true_probability[index] + 1),
                  "true_logit_margin": float(logits[index, label] - other_max[index])}
                 for index, label in enumerate(labels))


def summarize_scores(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    """Summarize one fixed population, preserving the denominator for empty cohorts."""
    return {"examples": len(rows), **{name: (math.fsum(float(row[field]) for row in rows) / len(rows) * scale if rows else None)
            for name, field, scale in (("accuracy", "correct", 100), ("nll", "nll", 1),
                ("mean_true_probability", "true_probability", 1), ("mean_top1_probability", "top1_probability", 1),
                ("entropy", "entropy", 1), ("brier", "brier", 1))}}


def verify_saved_predictions(rows: Sequence[dict[str, object]], original: Sequence[dict[str, object]]) -> dict[str, object]:
    """Check exact identities/classes and measure independent float64 versus saved FP32 NLL."""
    if (not rows or len(rows) != len(original) or [(row["image_id"], row["label"], row["prediction"]) for row in rows]
            != [(row["image_id"], row["label"], row["prediction"]) for row in original]):
        raise ValueError("independent logits disagree with original identities or predictions")
    differences = tuple(abs(row["nll"] - old["nll"]) for row, old in zip(rows, original, strict=True))
    if max(differences) >= .00003:
        raise ValueError("independent logits disagree with original NLL")
    return {"examples": len(rows), "prediction_mismatches": 0, "maximum_nll_difference": max(differences),
            "mean_nll_difference": math.fsum(differences) / len(rows)}


def fit_temperature(logits: np.ndarray, labels: np.ndarray, bounds: tuple[float, float]) -> tuple[float, bool]:
    """Find the convex inverse-temperature NLL optimum on calibration rows only."""
    logits, labels = np.asarray(logits, dtype=np.float64), np.asarray(labels)
    require_logits(logits, labels)
    if not 0 < bounds[0] < bounds[1]:
        raise ValueError("invalid numerical temperature bounds")
    centered = logits - np.max(logits, axis=1, keepdims=True)
    target = centered[np.arange(len(labels)), labels]

    def derivative(inverse_temperature: float) -> float:
        scaled = centered * inverse_temperature
        probabilities = np.exp(scaled - logsumexp(scaled, axis=1, keepdims=True))
        return float(np.mean(np.sum(probabilities * centered, axis=1) - target))

    lower, upper = 1 / bounds[1], 1 / bounds[0]
    if derivative(lower) >= 0:
        return bounds[1], True
    if derivative(upper) <= 0:
        return bounds[0], True
    return 1 / brentq(derivative, lower, upper, xtol=1e-12, rtol=1e-12), False


@dataclass(frozen=True, slots=True)
class CrossFitResult:
    """Out-of-fold scores and the exact disjoint populations used for each fit."""

    fits: tuple[dict[str, object], ...]
    rows: tuple[dict[str, object], ...]


def cross_fit(
    logits: np.ndarray, labels: np.ndarray, image_ids: Sequence[str], folds: Sequence[int],
    bounds: tuple[float, float] = (.01, 100.),
) -> CrossFitResult:
    """Fit each temperature on all other folds, never its scored image labels."""
    require_logits(logits, labels)
    if len(image_ids) != len(labels) or len(set(image_ids)) != len(image_ids) or len(folds) != len(labels):
        raise ValueError("cross-fitting requires unique aligned identities")
    fold_array = np.asarray(folds)
    if set(folds) != set(range(len(set(folds)))) or len(set(folds)) < 2:
        raise ValueError("cross-fitting requires consecutive nonempty folds")
    fits = ()
    for fold in sorted(set(folds)):
        calibration, evaluation = fold_array != fold, fold_array == fold
        temperature, at_bound = fit_temperature(logits[calibration], labels[calibration], bounds)
        calibration_ids = tuple(name for name, keep in zip(image_ids, calibration, strict=True) if keep)
        evaluation_ids = tuple(name for name, keep in zip(image_ids, evaluation, strict=True) if keep)
        if set(calibration_ids) & set(evaluation_ids):
            raise ValueError("calibration/evaluation overlap")
        fits += ({"fold": int(fold), "temperature": temperature, "at_numerical_bound": at_bound,
                  "calibration_examples": len(calibration_ids), "evaluation_examples": len(evaluation_ids),
                  "calibration_ids_hash": record_sha256(calibration_ids), "evaluation_ids_hash": record_sha256(evaluation_ids)},)
    temperatures = np.array([fits[fold]["temperature"] for fold in folds])
    raw, calibrated = score_rows(logits, labels), score_rows(logits, labels, temperatures)
    rows = tuple({"image_id": name, "label": int(label), "fold": int(fold), "temperature": float(temperature),
                  **{f"{prefix}_{key}": value for prefix, values in (("raw", before), ("calibrated", after)) for key, value in values.items()}}
                 for name, label, fold, temperature, before, after in zip(image_ids, labels, folds, temperatures, raw, calibrated, strict=True))
    return CrossFitResult(fits, rows)


def reliability_rows(rows: Sequence[dict[str, object]], prefix: str, bins: int = 15) -> tuple[dict[str, object], ...]:
    """Report fixed equal-width top-label reliability bins, including empty bins."""
    return tuple({"bin": index, "lower": index / bins, "upper": (index + 1) / bins,
                  "examples": len(selected),
                  "accuracy": math.fsum(float(row[f"{prefix}_correct"]) for row in selected) / len(selected) if selected else None,
                  "confidence": math.fsum(float(row[f"{prefix}_top1_probability"]) for row in selected) / len(selected) if selected else None}
                 for index in range(bins)
                 for selected in (tuple(row for row in rows if min(bins - 1, int(float(row[f"{prefix}_top1_probability"]) * bins)) == index),))


def paired_training_cohorts(
    history: Sequence[dict[str, object]], srt_scores: Sequence[dict[str, object]], uniform_scores: Sequence[dict[str, object]],
) -> tuple[dict[str, object], ...]:
    """Apply SRT-defined cohorts to the same image identities under both checkpoints."""
    histories = {row["image_id"]: row for row in history}
    by_method = {method: {row["image_id"]: row for row in scores} for method, scores in (("srt", srt_scores), ("uniform", uniform_scores))}
    if (not histories or len(histories) != len(history) or len(srt_scores) != len(history) or len(uniform_scores) != len(history)
            or any(set(values) != set(histories) for values in by_method.values())):
        raise ValueError("paired training cohorts require identical image populations")
    if any(by_method["srt"][key]["label"] != by_method["uniform"][key]["label"] for key in histories):
        raise ValueError("paired training labels differ")
    heavy = frozenset(row["image_id"] for row in sorted(history, key=lambda row: (-row["presentations"], row["image_id"]))[:max(1, len(history) // 100)])
    groups = (("all", lambda row: True), ("unrevisited_tasks40_50", lambda row: row["last_stage"] < 40),
              ("unrevisited_and_last_probability_ge90", lambda row: row["last_stage"] < 40 and row["last_confidence"] >= .9),
              ("revisited_tasks40_50", lambda row: row["last_stage"] >= 40),
              ("top1pct_presentations", lambda row: row["image_id"] in heavy))
    groups += tuple((f"last_review_{lower:02d}_{upper:02d}", lambda row, lo=lower, hi=upper: lo <= row["last_stage"] <= hi)
                    for lower, upper in ((1, 10), (11, 20), (21, 30), (31, 39), (40, 49), (50, 50)))
    return tuple({"cohort": name, "method": method, "cohort_ids_hash": record_sha256(ids),
                  "last_augmented_probability_mean": math.fsum(histories[key]["last_confidence"] for key in ids) / len(ids) if ids else None,
                  "now_wrong": sum(not bool(values[key]["correct"]) for key in ids),
                  "now_probability_below_half": sum(float(values[key]["true_probability"]) < .5 for key in ids),
                  **summarize_scores(tuple(values[key] for key in ids))}
                 for name, predicate in groups for ids in (tuple(sorted(key for key, row in histories.items() if predicate(row))),)
                 for method, values in by_method.items())

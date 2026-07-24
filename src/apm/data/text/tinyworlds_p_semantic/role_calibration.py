"""Cross-fitted role calibration for TinyWorlds-P semantic-v2."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping
from hashlib import sha256
import math

from apm.data.text.tinyworlds_p_semantic.contracts import Role
from apm.data.text.tinyworlds_p_semantic.v2_contracts import (
    CalibratedRoleScore,
    RoleCalibrationReference,
    V2_BENCHMARK_ID,
    V2SemanticConstructionConfig,
)


class RoleCalibrationError(ValueError):
    """The frozen cross-conformal role calibration cannot be constructed."""


def role_calibration_fold(
    role: Role,
    word: str,
    config: V2SemanticConstructionConfig,
) -> int:
    """Assign one word to its deterministic, benchmark-namespaced fold."""
    if role not in ("noun", "verb") or type(word) is not str or not word:
        raise ValueError("role calibration fold requires a role and word")
    digest = sha256(
        f"{V2_BENCHMARK_ID}\0{config.role_calibration_namespace}\0"
        f"{role}\0{word}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest, "big") % config.role_calibration_fold_count


def calibrate_role_margins(
    raw_margins: Mapping[tuple[Role, str], float],
    config: V2SemanticConstructionConfig,
) -> tuple[
    dict[tuple[Role, str], CalibratedRoleScore],
    tuple[RoleCalibrationReference, ...],
]:
    """Compute held-out-fold lower-tail conformal p-values for every word."""
    canonical: dict[tuple[Role, str], float] = {}
    for key, value in raw_margins.items():
        role, word = key
        if role not in ("noun", "verb") or type(word) is not str or not word:
            raise ValueError("role-margin keys must contain a role and word")
        if type(value) not in (int, float) or not math.isfinite(float(value)):
            raise ValueError("role margins must be finite numbers")
        canonical[(role, word)] = float(value)
    if len(canonical) != len(raw_margins):
        raise ValueError("role margins contain duplicate normalized identities")

    by_role = {
        role: tuple(
            sorted(
                (word, margin)
                for (item_role, word), margin in canonical.items()
                if item_role == role
            )
        )
        for role in ("noun", "verb")
    }
    if any(not values for values in by_role.values()):
        raise RoleCalibrationError("cross-conformal calibration requires both roles")

    references: dict[tuple[Role, int], tuple[float, ...]] = {}
    summaries = []
    for role in ("noun", "verb"):
        for fold in range(config.role_calibration_fold_count):
            values = tuple(
                sorted(
                    margin
                    for word, margin in by_role[role]
                    if role_calibration_fold(role, word, config) != fold
                )
            )
            if len(values) < config.minimum_calibration_reference_words:
                raise RoleCalibrationError(
                    f"{role} fold {fold} has fewer than "
                    f"{config.minimum_calibration_reference_words} calibration references"
                )
            references[(role, fold)] = values
            summaries.append(
                RoleCalibrationReference(
                    role=role,
                    fold=fold,
                    reference_count=len(values),
                    rejection_cutoff=_rejection_cutoff(
                        values,
                        config.role_calibration_alpha,
                    ),
                )
            )

    scores = {}
    summary_by_key = {(item.role, item.fold): item for item in summaries}
    for (role, word), margin in sorted(canonical.items()):
        fold = role_calibration_fold(role, word, config)
        reference = references[(role, fold)]
        conformal_p = (1 + bisect_right(reference, margin)) / (len(reference) + 1)
        summary = summary_by_key[(role, fold)]
        scores[(role, word)] = CalibratedRoleScore(
            role=role,
            word=word,
            fold=fold,
            raw_margin=margin,
            reference_count=len(reference),
            conformal_p=conformal_p,
            rejection_cutoff=summary.rejection_cutoff,
        )
    return scores, tuple(summaries)


def _rejection_cutoff(
    sorted_reference: tuple[float, ...],
    alpha: float,
) -> float | None:
    """Return the largest observed score whose conformal p-value can reject."""
    allowed_count = math.floor(alpha * (len(sorted_reference) + 1) - 1 + 1e-12)
    if allowed_count <= 0:
        return None
    index = min(allowed_count, len(sorted_reference)) - 1
    while index >= 0:
        value = sorted_reference[index]
        if bisect_right(sorted_reference, value) <= allowed_count:
            return value
        index -= 1
    return None


__all__ = [
    "RoleCalibrationError",
    "calibrate_role_margins",
    "role_calibration_fold",
]

"""Finite, validation-only candidate construction for H=128 SRT tuning."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from itertools import product
import math
from pathlib import Path

import yaml

from apm.continual.artifacts import record_sha256
from apm.continual.vision.imagenetr.srt_config import SRTConfig
from apm.continual.vision.imagenetr.srt_scheduler import RecallPolicy


DEFAULT_CONFIG = Path("configs/vision/imagenetr/srt_h128_tuning.yaml")


@dataclass(frozen=True, slots=True)
class SRTTuningConfig:
    """A frozen policy grid, learning-rate grid and one local refinement phase."""

    baseline_config: str
    capacity: int
    profiles: tuple[str, ...]
    historical_fractions: tuple[float, ...]
    interval_units: tuple[int, ...]
    lora_rate_multipliers: tuple[float, ...]
    head_rate_multipliers: tuple[float, ...]
    threshold_powers: tuple[float, ...]
    historical_fraction_offsets: tuple[float, ...]
    interval_multipliers: tuple[float, ...]

    def __post_init__(self) -> None:
        positive = (*self.lora_rate_multipliers, *self.head_rate_multipliers,
                    *self.threshold_powers, *self.interval_multipliers)
        if (self.capacity != 128 or not self.profiles or len(set(self.profiles)) != len(self.profiles)
                or any(not values for values in (self.lora_rate_multipliers, self.head_rate_multipliers,
                                                 self.threshold_powers, self.interval_multipliers, self.historical_fraction_offsets))
                or 1. not in self.lora_rate_multipliers or 1. not in self.head_rate_multipliers
                or not self.historical_fractions or any(not 0 <= value <= 1 for value in self.historical_fractions)
                or not self.interval_units or any(type(value) is not int or value < 1 for value in self.interval_units)
                or not positive or any(not math.isfinite(value) or value <= 0 for value in positive)
                or any(not math.isfinite(value) for value in self.historical_fraction_offsets)):
            raise ValueError("invalid finite H=128 tuning configuration")

    @property
    def maximum_candidates(self) -> int:
        return (len(self.profiles) * len(self.historical_fractions) * len(self.interval_units)
                + len(self.lora_rate_multipliers) * len(self.head_rate_multipliers) - 1
                + len(self.threshold_powers) + len(self.historical_fraction_offsets) + len(self.interval_multipliers))


@dataclass(frozen=True, slots=True)
class SRTCandidate:
    """One replay policy and two constant learning rates, independent of evaluation scores."""

    policy: RecallPolicy
    lora_learning_rate: float
    head_learning_rate: float

    def __post_init__(self) -> None:
        if any(not math.isfinite(value) or value <= 0 for value in (self.lora_learning_rate, self.head_learning_rate)):
            raise ValueError("candidate learning rates must be finite and positive")

    @property
    def content_hash(self) -> str:
        return record_sha256(asdict(self))

    @property
    def name(self) -> str:
        return "candidate_" + self.content_hash[:20]

    def optimizer_config(self, training: SRTConfig) -> SRTConfig:
        """Change only the two declared optimizer rates in the authenticated source recipe."""
        return replace(training, lora_learning_rate=self.lora_learning_rate, head_learning_rate=self.head_learning_rate)


def load_config(path: Path = DEFAULT_CONFIG) -> SRTTuningConfig:
    """Load the finite search without exposing command-line scientific overrides."""
    values = yaml.safe_load(path.read_text())
    return SRTTuningConfig(**{name: tuple(value) if isinstance(value, list) else value for name, value in values.items()})


def candidate_from_record(record: dict[str, object]) -> SRTCandidate:
    """Reconstruct a candidate while revalidating ordered thresholds and positive rates."""
    policy = record["policy"]
    return SRTCandidate(RecallPolicy(**{**policy, "thresholds": tuple(policy["thresholds"])}),
                        record["lora_learning_rate"], record["head_learning_rate"])


def unique_candidates(candidates: tuple[SRTCandidate, ...]) -> tuple[SRTCandidate, ...]:
    """Deduplicate exact scientific recipes in stable first-occurrence order."""
    return tuple({candidate.content_hash: candidate for candidate in candidates}.values())


def policy_candidates(config: SRTTuningConfig, training: SRTConfig) -> tuple[SRTCandidate, ...]:
    """Cross declared profiles, old fractions and intervals at the original learning rates."""
    profiles = dict(training.profiles)
    if any(name not in profiles for name in config.profiles):
        raise ValueError("unknown threshold profile in the tuning grid")
    return unique_candidates(tuple(SRTCandidate(RecallPolicy(name, profiles[name], fraction, unit),
                                                training.lora_learning_rate, training.head_learning_rate)
                                   for name, fraction, unit in product(config.profiles, config.historical_fractions, config.interval_units)))


def rate_candidates(config: SRTTuningConfig, training: SRTConfig, winner: SRTCandidate) -> tuple[SRTCandidate, ...]:
    """Cross absolute baseline-relative learning rates around the selected replay policy."""
    return unique_candidates(tuple(SRTCandidate(winner.policy, training.lora_learning_rate * lora,
                                                training.head_learning_rate * head)
                                   for lora, head in product(config.lora_rate_multipliers, config.head_rate_multipliers)))


def refinement_candidates(config: SRTTuningConfig, winner: SRTCandidate) -> tuple[SRTCandidate, ...]:
    """Change one policy coordinate at a time while keeping the selected learning rates."""
    policy = winner.policy
    neighbors = tuple(replace(policy, profile=f"{policy.profile}_power{power:g}",
                              thresholds=tuple(round(value ** power, 8) for value in policy.thresholds))
                      for power in config.threshold_powers)
    neighbors += tuple(replace(policy, historical_fraction=round(max(0., min(1., policy.historical_fraction + offset)), 8))
                       for offset in config.historical_fraction_offsets)
    neighbors += tuple(replace(policy, interval_unit=max(1, round(policy.interval_unit * multiplier)))
                       for multiplier in config.interval_multipliers)
    return unique_candidates(tuple(replace(winner, policy=neighbor) for neighbor in neighbors))


def select_winner(rows: tuple[dict[str, object], ...]) -> dict[str, object]:
    """Choose task-50 validation accuracy, then NLL, then content identity; never a test score."""
    complete = tuple(row for row in rows if row["status"] == "complete")
    if not complete:
        raise ValueError("no numerically valid full-stream tuning candidate")
    if any(row["evaluation_role"] != "validation" or row["completed_stages"] != 50
           or not math.isfinite(row["validation_accuracy"]) or not math.isfinite(row["validation_nll"])
           or not 0 <= row["validation_accuracy"] <= 100 for row in complete):
        raise ValueError("selection accepts only finite task-50 validation results")
    return min(complete, key=lambda row: (-row["validation_accuracy"], row["validation_nll"], row["candidate_hash"]))

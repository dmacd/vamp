"""Shared fixed-epoch and energy-convergence training schedules."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import Literal, TypeAlias


MetricValue: TypeAlias = int | float | str | bool
MetricsRow: TypeAlias = dict[str, MetricValue]


@dataclass(frozen=True)
class FixedEpochSchedule:
    """Train for exactly ``epochs`` epochs."""

    epochs: int
    mode: Literal["fixed"] = field(init=False, default="fixed")

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")

    @property
    def epoch_limit(self) -> int:
        return self.epochs


@dataclass(frozen=True)
class EnergyConvergenceSchedule:
    """Stop after observed-digit energy stops improving materially."""

    min_epochs: int = 10
    max_epochs: int = 100
    relative_delta: float = 1e-3
    patience: int = 5
    probe_count: int = 1_024
    mode: Literal["energy_convergence"] = field(init=False, default="energy_convergence")

    def __post_init__(self) -> None:
        if self.min_epochs < 1:
            raise ValueError("min_epochs must be at least 1")
        if self.max_epochs < self.min_epochs:
            raise ValueError("max_epochs must be greater than or equal to min_epochs")
        if self.relative_delta <= 0.0:
            raise ValueError("relative_delta must be greater than zero")
        if self.patience < 1:
            raise ValueError("patience must be at least 1")
        if self.probe_count < 1:
            raise ValueError("probe_count must be at least 1")

    @property
    def epoch_limit(self) -> int:
        return self.max_epochs


TrainingSchedule: TypeAlias = FixedEpochSchedule | EnergyConvergenceSchedule


@dataclass(frozen=True)
class ConvergenceObservation:
    """One comparable energy observation and the resulting tracker state."""

    epoch: int
    energy: float
    best_energy: float
    reference_energy: float
    relative_improvement: float
    stale_epochs: int
    is_best: bool
    converged: bool

    def as_metrics(self) -> MetricsRow:
        return {
            "monitor_energy": self.energy,
            "best_energy": self.best_energy,
            "reference_energy": self.reference_energy,
            "relative_improvement": self.relative_improvement,
            "stale_epochs": self.stale_epochs,
            "is_best": self.is_best,
            "converged": self.converged,
        }


@dataclass(frozen=True)
class TrainingTrace:
    """Per-epoch diagnostics and the state-selection result for one task."""

    rows: tuple[MetricsRow, ...]
    stop_reason: Literal["fixed_epochs", "converged", "max_epochs"]
    epochs_run: int
    selected_epoch: int
    selected_energy: float | None
    converged: bool


class EnergyConvergenceTracker:
    """Track cumulative meaningful improvement and a separate absolute best."""

    def __init__(self, schedule: EnergyConvergenceSchedule) -> None:
        self.schedule = schedule
        self.best_energy: float | None = None
        self.best_epoch = 0
        self.reference_energy: float | None = None
        self.stale_epochs = 0

    def observe(self, epoch: int, energy: float) -> ConvergenceObservation:
        if epoch < 1:
            raise ValueError("epoch must be at least 1")
        if not isfinite(energy):
            raise FloatingPointError(f"non-finite convergence energy at epoch {epoch}: {energy}")

        is_first = self.reference_energy is None
        is_best = self.best_energy is None or energy < self.best_energy
        if is_best:
            self.best_energy = energy
            self.best_epoch = epoch

        if is_first:
            self.reference_energy = energy
            relative_improvement = 0.0
            self.stale_epochs = 0
        else:
            assert self.reference_energy is not None
            denominator = max(abs(self.reference_energy), 1e-12)
            relative_improvement = (self.reference_energy - energy) / denominator
            if relative_improvement >= self.schedule.relative_delta:
                self.reference_energy = energy
                self.stale_epochs = 0
            else:
                self.stale_epochs += 1

        assert self.best_energy is not None
        assert self.reference_energy is not None
        converged = epoch >= self.schedule.min_epochs and self.stale_epochs >= self.schedule.patience
        return ConvergenceObservation(
            epoch=epoch,
            energy=energy,
            best_energy=self.best_energy,
            reference_energy=self.reference_energy,
            relative_improvement=relative_improvement,
            stale_epochs=self.stale_epochs,
            is_best=is_best,
            converged=converged,
        )


def schedule_payload(schedule: TrainingSchedule) -> dict[str, int | float | str]:
    """Return a JSON-serializable training schedule."""
    return asdict(schedule)

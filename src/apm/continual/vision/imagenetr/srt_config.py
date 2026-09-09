"""Resolved scientific choices for the separate single-adapter SRT experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import yaml

from apm.continual.artifacts import record_sha256
from apm.continual.vision.imagenetr.srt_scheduler import RecallPolicy


DEFAULT_SRT_CONFIG = Path("configs/vision/imagenetr/srt_r16_v1.yaml")


@dataclass(frozen=True, slots=True)
class SRTConfig:
    """One finite calibration matrix and four final full-data runs."""

    seed: int
    tasks: int
    budgets: tuple[int, ...]
    batch_size: int
    lora_learning_rate: float
    head_learning_rate: float
    momentum: float
    weight_decay: float
    num_workers: int
    checkpoint_steps: int
    screen_tasks: int
    finalists: int
    profiles: tuple[tuple[str, tuple[float, ...]], ...]
    historical_fractions: tuple[float, ...]
    interval_units: tuple[int, ...]
    artifact_root: str
    reference_run: str
    reference_result_sha256: str
    reference_pdf_sha256: str
    reference_figure_sha256: str
    mlp_result_sha256: str
    dataset_hash: str
    split_hash: str

    def __post_init__(self) -> None:
        if (self.seed != 1993 or self.tasks != 50 or self.budgets != (1024, 4096)
                or self.batch_size != 64 or self.screen_tasks != 16 or self.finalists != 2
                or self.checkpoint_steps < 1 or not 0 <= self.num_workers <= 4):
            raise ValueError("SRT config differs from the approved finite protocol")
        if len(self.policies) != 18:
            raise ValueError("SRT calibration requires eighteen settings per budget")

    @property
    def policies(self) -> tuple[RecallPolicy, ...]:
        return tuple(RecallPolicy(name, thresholds, fraction, unit)
                     for (name, thresholds), fraction, unit in product(
                         self.profiles, self.historical_fractions, self.interval_units))

    @property
    def content_hash(self) -> str:
        return record_sha256(asdict(self))


def load_srt_config(path: str | Path = DEFAULT_SRT_CONFIG) -> SRTConfig:
    """Read one explicit configuration without command-line science overrides."""
    values = yaml.safe_load(Path(path).read_text())
    values["profiles"] = tuple((name, tuple(thresholds)) for name, thresholds in values["profiles"].items())
    for name in ("budgets", "historical_fractions", "interval_units"):
        values[name] = tuple(values[name])
    return SRTConfig(**values)


def presentation_budget(current: int, historical: int, capacity: int) -> int:
    """Match four passes over current data plus the bounded historical count."""
    if current < 1 or historical < 0 or capacity < 1:
        raise ValueError("invalid SRT stage population")
    return 4 * (current + min(capacity, historical))

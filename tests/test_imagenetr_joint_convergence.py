"""Small CPU tests for validation selection, isolation, and exact joint resume."""

from dataclasses import replace
from hashlib import sha256

import pytest
from safetensors.torch import load_file
import torch
from torch import nn

from apm.continual.vision.imagenetr.data import ImageRecord
from apm.continual.vision.imagenetr.heads import AffineClassifier
from apm.continual.vision.imagenetr.joint_convergence import load_config
from apm.continual.vision.imagenetr.joint_convergence_training import (
    ConvergenceRule, JointPopulation, PlateauState, observe_validation, select_epochs, train_joint_job,
)
from apm.continual.vision.imagenetr.lora import LoRALinear
from apm.continual.vision.imagenetr.srt_config import load_srt_config


def test_plateau_requires_reductions_then_terminal_patience() -> None:
    rule = ConvergenceRule(minimum_epochs=2, maximum_epochs=30, plateau_patience=2,
                           terminal_patience=3, learning_rate_reductions=2)
    state = PlateauState()
    states = []
    for epoch in range(1, 9):
        state = observe_validation(state, 70, 1., epoch, rule)
        states.append(state)
    assert states[2].next_scale == .2 and states[4].next_scale == pytest.approx(.04)
    assert all(row.stop_reason is None for row in states[:-1])
    assert states[-1].stop_reason == "validation_plateau"
    with pytest.raises(ValueError):
        observe_validation(state, 70, 1., 9, rule)


def test_either_metric_and_accumulated_small_improvements_reset_plateau() -> None:
    rule = ConvergenceRule()
    first = observe_validation(PlateauState(), 70, 1., 1, rule)
    second = observe_validation(first, 70.05, 1., 2, rule)
    third = observe_validation(second, 70.1, 1., 3, rule)
    fourth = observe_validation(third, 69, .997, 4, rule)
    assert second.stale_epochs == 1 and third.stale_epochs == fourth.stale_epochs == 0
    assert fourth.accuracy_anchor == 70.1 and fourth.nll_anchor == .997


def test_epoch_cap_is_not_convergence_and_selection_uses_validation() -> None:
    rule = ConvergenceRule(minimum_epochs=2, maximum_epochs=2)
    state = observe_validation(observe_validation(PlateauState(), 70, 1., 1, rule), 72, .9, 2, rule)
    assert state.stop_reason == "maximum_epochs_without_plateau"
    rows = tuple({"epoch": epoch, "validation": {"accuracy": accuracy, "nll": nll}}
                 for epoch, accuracy, nll in ((1, 70, .8), (2, 72, .9), (3, 72, .85), (4, 72, .85), (5, 71, .7)))
    assert select_epochs(rows) == {"accuracy_selected": 3, "nll_selected": 5, "terminal": 5, "five_epoch": 5}
    with pytest.raises(ValueError, match="validation-only"):
        select_epochs(({"epoch": 1, "test": {"accuracy": 90}},))


@pytest.fixture
def joint_rows():
    def row(index):
        digest = sha256(str(index).encode()).hexdigest()
        return ImageRecord(digest, f"source/{index}.jpg", f"train/{index}.jpg", digest,
                           f"class{index % 4}", index % 4, index % 4, 0, "train", digest, 1)
    return tuple(row(index) for index in range(160))


def test_population_rejects_test_images_overlap_and_foreign_probes(joint_rows) -> None:
    fitting, validation = joint_rows[:128], joint_rows[128:]
    JointPopulation(fitting, validation, fitting[:16])
    for fit, val, probe in ((fitting, fitting[:2], fitting[:16]), (fitting, validation, validation[:2]),
                            ((replace(fitting[0], split="test"),), (), (fitting[0],))):
        with pytest.raises(ValueError, match="isolated training"):
            JointPopulation(fit, val, probe)


class TinyJoint(nn.Module):
    def __init__(self, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        self.adapter = LoRALinear(nn.Linear(4, 4), rank=2, alpha=2, initialization_seed=seed)
        self.classifier = AffineClassifier(tuple(range(4)), 4, seed + 10000)

    def forward(self, images):
        return self.classifier(self.adapter(images))


class TinyLoader:
    def load(self, items, training):
        return (torch.stack(tuple(torch.randn(4, generator=torch.Generator().manual_seed(int(item.row.image_id[:8], 16) + item.ordinal)) for item in items)),
                torch.tensor(tuple(item.row.remapped_class_index for item in items)))


def test_joint_step_resume_and_completed_reuse_are_exact(tmp_path, joint_rows) -> None:
    population = JointPopulation(joint_rows[:128], (), joint_rows[:16])
    arguments = dict(protocol_hash="a" * 64, seed=1993, population=population,
                     optimizer_config=replace(load_srt_config(), checkpoint_steps=1, num_workers=0),
                     rule=ConvergenceRule(), loader=TinyLoader(), device=torch.device("cpu"),
                     model_factory=TinyJoint, fixed_schedule=(1., .2))
    direct = train_joint_job(tmp_path / "direct", **arguments)
    def interrupt(row):
        if row["steps"] == 1:
            raise InterruptedError("test interruption")
    with pytest.raises(InterruptedError):
        train_joint_job(tmp_path / "resumed", progress_callback=interrupt, **arguments)
    resumed = train_joint_job(tmp_path / "resumed", **arguments)
    assert direct["optimizer_steps"] == resumed["optimizer_steps"] == 4
    assert resumed["training_presentations"] == 256 and resumed["invocation_optimizer_steps"] == 3
    assert direct["epochs"][1]["training"]["nll"] == resumed["epochs"][1]["training"]["nll"]
    for epoch in (1, 2):
        first, second = (load_file(str(tmp_path / name / "epochs" / f"{epoch:03d}" / "model.safetensors")) for name in ("direct", "resumed"))
        assert all(torch.equal(first[name], second[name]) for name in first)
    repeated = train_joint_job(tmp_path / "resumed", **arguments)
    assert repeated["invocation_optimizer_steps"] == 0 and repeated["content_hash"] == resumed["content_hash"]


def test_joint_development_cannot_accept_a_fixed_schedule(tmp_path, joint_rows) -> None:
    population = JointPopulation(joint_rows[:128], joint_rows[128:], joint_rows[:16])
    with pytest.raises(ValueError, match="only development"):
        train_joint_job(tmp_path, "a" * 64, 1993, population, load_srt_config(), ConvergenceRule(), TinyLoader(),
                        torch.device("cpu"), TinyJoint, fixed_schedule=(1.,))


def test_requested_convergence_configuration_is_explicit() -> None:
    config = load_config()
    assert config.refit_seeds == (1993, 1994, 1995)
    assert config.rule.learning_rate_factor == .2 and config.rule.learning_rate_reductions == 3
    assert config.rule.minimum_epochs == 30 and config.rule.maximum_epochs == 160

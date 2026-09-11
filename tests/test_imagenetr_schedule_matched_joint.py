"""Short CPU checks for replay-work matching, full-data draws, and exact resume."""

from dataclasses import replace
from hashlib import sha256

import pyarrow.parquet as pq
from pyrsistent import pvector
import pytest
from safetensors.torch import load_file
import torch
from torch import nn

from apm.continual.artifacts import atomic_write, canonical_json_bytes
from apm.continual.vision.imagenetr.data import ImageRecord
from apm.continual.vision.imagenetr.heads import AffineClassifier
from apm.continual.vision.imagenetr.joint_convergence_training import JointPopulation
from apm.continual.vision.imagenetr.lora import LoRALinear
from apm.continual.vision.imagenetr.schedule_matched_joint import load_config, source_schedule
from apm.continual.vision.imagenetr.schedule_matched_training import (
    ScheduledBatch, audit_schedule_draws, draw_presentations, require_schedule, train_schedule_job, validate_schedule_job,
)
from apm.continual.vision.imagenetr.srt_config import load_srt_config
from apm.continual.vision.imagenetr.srt_evidence import read_sealed


@pytest.fixture
def population():
    rows = tuple(ImageRecord(sha256(str(index).encode()).hexdigest(), f"source/{index}.jpg", f"train/{index}.jpg",
                             sha256(str(index).encode()).hexdigest(), f"class{index}", index, index,
                             index // 4, "train", sha256(str(index).encode()).hexdigest(), 1) for index in range(200))
    return JointPopulation(rows, (), rows[:16])


class TinyModel(nn.Module):
    def __init__(self, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.adapter = LoRALinear(nn.Linear(4, 4), rank=2, alpha=2, initialization_seed=seed)
        self.classifier = AffineClassifier(tuple(range(200)), 4, seed + 10000)

    def forward(self, images):
        return self.classifier(self.adapter(images))


class TinyLoader:
    def load(self, items, training):
        images = tuple(torch.randn(4, generator=torch.Generator().manual_seed(int(item.row.image_id[:8], 16) + item.ordinal)) for item in items)
        return torch.stack(images), torch.tensor(tuple(item.row.remapped_class_index for item in items))


@pytest.fixture
def arguments(population):
    return dict(protocol_hash="a" * 64, seed=1993, population=population,
                schedule=tuple(ScheduledBatch(index + 1, index // 2 + 1, size) for index, size in enumerate((64, 7, 1, 32))),
                optimizer_config=replace(load_srt_config(), checkpoint_steps=1, num_workers=0),
                loader=TinyLoader(), device=torch.device("cpu"), model_factory=TinyModel)


@pytest.mark.parametrize("schedule", [(), (ScheduledBatch(2, 1, 4),), (ScheduledBatch(1, 2, 4),),
                                      (ScheduledBatch(1, 1, 0),), (ScheduledBatch(1, 1, 65),),
                                      (ScheduledBatch(1, 1, 4), ScheduledBatch(2, 3, 4))])
def test_schedule_rejects_missing_steps_blocks_and_invalid_sizes(schedule):
    with pytest.raises(ValueError):
        require_schedule(schedule, 64)


def test_all_training_images_are_available_independent_of_source_block(population):
    history = pvector([0] * len(population.fitting))
    first, following = draw_presentations(population, history, 1993, ScheduledBatch(1, 1, 64))
    same, _ = draw_presentations(population, history, 1993, ScheduledBatch(1, 50, 64))
    different, _ = draw_presentations(population, history, 1994, ScheduledBatch(1, 1, 64))
    assert first == same and first != different
    assert len({item.row.image_id for item in first}) == 64
    assert any(item.row.task_index >= 40 for item in first)
    assert all(item.ordinal == 1 for item in first) and sum(following) == 64 and sum(history) == 0
    next_items, next_history = draw_presentations(population, following, 1993, ScheduledBatch(2, 1, 64))
    assert sum(next_history) == 128 and any(item.ordinal == 2 for item in next_items)


@pytest.mark.parametrize("interruption_step", [1, 2, 3, 4])
def test_mixed_batch_resume_reuse_and_draw_audit(tmp_path, arguments, interruption_step):
    direct = train_schedule_job(tmp_path / "direct", **arguments)
    def interrupt(row):
        if row["steps"] == interruption_step:
            raise InterruptedError("test interruption")
    with pytest.raises(InterruptedError):
        train_schedule_job(tmp_path / "resumed", progress_callback=interrupt, **arguments)
    resumed = train_schedule_job(tmp_path / "resumed", **arguments)
    assert resumed["invocation_optimizer_steps"] == 4 - interruption_step
    assert direct["optimizer_steps"] == resumed["optimizer_steps"] == 4
    assert direct["training_presentations"] == resumed["training_presentations"] == 104
    for block in (1, 2):
        first, second = (load_file(str(tmp_path / name / f"blocks/{block:03d}/model.safetensors")) for name in ("direct", "resumed"))
        assert all(torch.equal(first[key], second[key]) for key in first)
    audit = audit_schedule_draws(tmp_path / "resumed", arguments["population"], arguments["schedule"])
    assert audit["verified_updates"] == 4 and audit["verified_presentations"] == 104
    reused = train_schedule_job(tmp_path / "resumed", **arguments)
    assert reused["invocation_optimizer_steps"] == 0 and reused["content_hash"] == resumed["content_hash"]
    for chunk in resumed["chunks"]:
        for row in pq.read_table(tmp_path / "resumed" / chunk["path"]).to_pylist():
            assert row["lora_learning_rate"] == .0005 and row["head_learning_rate"] == .01


def test_validation_population_and_changed_schedule_are_rejected(tmp_path, arguments, population):
    with pytest.raises(ValueError, match="training-only"):
        train_schedule_job(tmp_path, **{**arguments, "population": JointPopulation(population.fitting[:128], population.fitting[128:], population.probe)})
    train_schedule_job(tmp_path, **arguments)
    with pytest.raises(ValueError, match="immutable"):
        train_schedule_job(tmp_path, **{**arguments, "schedule": (ScheduledBatch(1, 1, 64),)})


def test_modified_committed_chunk_is_not_reused(tmp_path, arguments):
    result = train_schedule_job(tmp_path, **arguments)
    atomic_write(tmp_path / result["chunks"][0]["path"], b"corrupt fixture")
    with pytest.raises(ValueError, match="chunk changed"):
        validate_schedule_job(tmp_path)


def test_partial_resume_does_not_adopt_orphan_block(tmp_path, arguments):
    def interrupt(row):
        if row["steps"] == 1:
            raise InterruptedError("test interruption")
    with pytest.raises(InterruptedError):
        train_schedule_job(tmp_path, progress_callback=interrupt, **arguments)
    atomic_write(tmp_path / "blocks/001/result.json", canonical_json_bytes({"uncommitted": True}))
    resumed = train_schedule_job(tmp_path, **arguments)
    assert resumed["invocation_optimizer_steps"] == 3
    assert read_sealed(tmp_path / "blocks/001/result.json") == resumed["blocks"][0]


def test_control_configuration_is_explicit():
    config = load_config()
    assert config.seeds == (1993, 1994, 1995)
    assert config.source_condition == "uniform_h4096_standard_rho80_unit8"
    assert config.fit_probe_images == 2048 and config.num_workers == 2


@pytest.mark.integration
def test_real_source_schedule_has_exact_work():
    from pathlib import Path
    config = load_config()
    schedule, evidence = source_schedule(Path(config.source_run), config)
    assert len(schedule) == 56243 and sum(batch.size for batch in schedule) == 844640
    assert len(evidence["blocks"]) == 50 and min(batch.size for batch in schedule) == 1

"""Fast numerical, population, and interrupted-inference diagnostic checks."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from apm.continual.artifacts import record_sha256
from apm.continual.vision.imagenetr.checkpoint_diagnostic_inference import CollectionSpec, collect_logits, load_collection
from apm.continual.vision.imagenetr.checkpoint_diagnostic_math import cross_fit, fit_temperature, fixed_folds, paired_training_cohorts, reliability_rows, score_rows
from apm.continual.vision.imagenetr.data import ImageRecord


def test_folds_are_identity_stable_class_stratified_and_balanced():
    ids = tuple(f"image-{index}" for index in range(45))
    labels = (0,) * 13 + (1,) * 15 + (2,) * 17
    folds = fixed_folds(ids, labels)
    assert tuple(folds.count(index) for index in range(5)) == (9,) * 5
    assert dict(zip(ids, folds)) == dict(zip(reversed(ids), fixed_folds(tuple(reversed(ids)), tuple(reversed(labels)))))
    for label in set(labels):
        counts = tuple(sum(target == label and fold == index for target, fold in zip(labels, folds)) for index in range(5))
        assert min(counts) > 0 and max(counts) - min(counts) <= 1
    with pytest.raises(ValueError, match="unique"):
        fixed_folds(("duplicate",) * 10, (0,) * 10)


def test_independent_nll_matches_torch_and_temperature_preserves_predictions():
    logits = np.array([[1000., 999., -1000.], [-1000., 500., 1000.], [1., 1., 1.]])
    labels = np.array([0, 1, 2])
    actual = score_rows(logits, labels)
    expected = torch.nn.functional.cross_entropy(torch.from_numpy(logits), torch.from_numpy(labels), reduction="none")
    assert np.allclose([row["nll"] for row in actual], expected.numpy(), atol=1e-12)
    assert all(0 <= row["brier"] <= 2 and row["entropy"] >= 0 for row in actual)
    for temperature in (.01, 2., 100.):
        assert [row["prediction"] for row in actual] == [row["prediction"] for row in score_rows(logits, labels, temperature)]
    with pytest.raises(ValueError, match="positive"):
        score_rows(logits, labels, 0.)


def test_temperature_known_optimum_and_bound_flags():
    logits = np.tile([2., 0.], (100, 1))
    labels = np.array([0] * 75 + [1] * 25)
    temperature, at_bound = fit_temperature(logits, labels, (.01, 100.))
    assert temperature == pytest.approx(2 / np.log(3), rel=1e-10)
    assert not at_bound
    assert fit_temperature(logits, np.zeros(100, dtype=int), (.01, 100.)) == (.01, True)
    assert fit_temperature(logits, np.ones(100, dtype=int), (.01, 100.)) == (100., True)


def test_cross_fit_never_uses_held_out_labels_for_its_temperature():
    logits = np.tile([2., 0.], (40, 1))
    labels = np.array([0] * 30 + [1] * 10)
    ids = tuple(f"id-{index}" for index in range(40))
    folds = tuple(index % 5 for index in range(40))
    first = cross_fit(logits, labels, ids, folds)
    changed = np.where(np.asarray(folds) == 0, 1 - labels, labels)
    second = cross_fit(logits, changed, ids, folds)
    assert first.fits[0] == second.fits[0]
    for fit in first.fits:
        calibration = tuple(name for name, fold in zip(ids, folds) if fold != fit["fold"])
        evaluation = tuple(name for name, fold in zip(ids, folds) if fold == fit["fold"])
        assert not set(calibration) & set(evaluation)
        assert fit["calibration_ids_hash"] == record_sha256(calibration)
        assert fit["evaluation_ids_hash"] == record_sha256(evaluation)
    assert all(row["raw_prediction"] == row["calibrated_prediction"] for row in first.rows)
    assert sum(row["examples"] for row in reliability_rows(first.rows, "raw")) == 40


def test_common_training_cohorts_use_srt_history_not_separate_model_populations():
    history = tuple({"image_id": f"id-{index}", "last_stage": stage, "last_confidence": probability, "presentations": count}
                    for index, (stage, probability, count) in enumerate(((5, .95, 4), (38, .6, 10), (50, .98, 40))))
    labels = np.array([0, 0, 0])
    scores = score_rows(np.array([[0., 2.], [3., 0.], [2., 0.]]), labels)
    srt = tuple({"image_id": row["image_id"], "label": 0, **score} for row, score in zip(history, scores))
    uniform = tuple({**row, "correct": True} for row in reversed(srt))
    cohorts = paired_training_cohorts(history, srt, uniform)
    stale = tuple(row for row in cohorts if row["cohort"] == "unrevisited_and_last_probability_ge90")
    assert {row["examples"] for row in stale} == {1}
    assert len({row["cohort_ids_hash"] for row in stale}) == 1
    assert tuple(row["accuracy"] for row in stale) == (0., 100.)
    assert any(row["examples"] == 0 and row["nll"] is None for row in cohorts)
    with pytest.raises(ValueError, match="identical"):
        paired_training_cohorts(history, srt, uniform + (uniform[0],))
    with pytest.raises(ValueError, match="labels"):
        paired_training_cohorts(history, srt, ({**uniform[0], "label": 1}, *uniform[1:]))


def image_rows():
    """Return five deterministic, valid manifest rows for a tiny CPU collection."""
    return tuple(ImageRecord(image_id=record_sha256(index), source_relative_path=f"n0/{index}.png",
        prepared_relative_path=f"test/n0/{index}.png", image_sha256=record_sha256(index), original_class_name="n0",
        original_class_index=index % 3, remapped_class_index=index % 3, task_index=0, split="test",
        priority=record_sha256(index), size_bytes=1) for index in range(5))


class FixtureLoader:
    def load(self, presentations, training):
        assert training is False
        labels = torch.tensor([row.row.remapped_class_index for row in presentations])
        return torch.nn.functional.one_hot(labels, 3).float(), labels


def fixture_model():
    """Construct an identical deterministic affine model on every resume."""
    layer = torch.nn.Linear(3, 3)
    with torch.no_grad():
        layer.weight.copy_(torch.eye(3) * 2)
        layer.bias.zero_()
    return layer


def test_chunk_resume_matches_direct_and_completed_reuse_never_loads_model(tmp_path: Path):
    rows = image_rows()
    spec = CollectionSpec("protocol", "fixture", "test", "checkpoint", record_sha256(tuple(row.image_id for row in rows)), 2, 2, 3)
    root = tmp_path / "resumed"
    with pytest.raises(InterruptedError):
        collect_logits(root, spec, rows, FixtureLoader(), torch.device("cpu"), fixture_model, stop_after_chunks=1)
    result, forwarded = collect_logits(root, spec, rows, FixtureLoader(), torch.device("cpu"), fixture_model)
    assert forwarded == 3 and result["model_forward_images"] == 5 and result["optimizer_steps"] == 0
    collect_logits(tmp_path / "direct", spec, rows, FixtureLoader(), torch.device("cpu"), fixture_model)
    resumed, logits, _ = load_collection(root)
    direct, direct_logits, _ = load_collection(tmp_path / "direct")
    assert resumed == direct and np.array_equal(logits, direct_logits)
    def unavailable_model():
        raise AssertionError("completed collection constructed a model")
    assert collect_logits(root, spec, rows, FixtureLoader(), torch.device("cpu"), unavailable_model) == (result, 0)
    (root / "chunks/000000/logits.parquet").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="chunk changed"):
        load_collection(root)


def test_collection_rejects_changed_population_before_loading_model(tmp_path: Path):
    rows = image_rows()
    spec = CollectionSpec("protocol", "fixture", "test", "checkpoint", record_sha256(tuple(row.image_id for row in rows)), 2, 2, 3)
    with pytest.raises(ValueError, match="population"):
        collect_logits(tmp_path, spec, (replace(rows[0], split="train"), *rows[1:]), FixtureLoader(), torch.device("cpu"), fixture_model)

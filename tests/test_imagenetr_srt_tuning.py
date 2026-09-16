"""Fast pure selection and candidate-grid checks; no model or accelerator import."""

from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from apm.continual.vision.imagenetr.srt_config import load_srt_config
from apm.continual.vision.imagenetr.srt_tuning_config import (
    SRTCandidate, candidate_from_record, load_config, policy_candidates, rate_candidates,
    refinement_candidates, select_winner, unique_candidates,
)


def test_frozen_search_is_bounded_includes_baseline_and_deduplicates():
    config, training = load_config(), load_srt_config()
    initial = policy_candidates(config, training)
    assert config.capacity == 128 and config.maximum_candidates == 32
    assert len(initial) == len({row.content_hash for row in initial}) == 18
    baseline = next(row for row in initial if row.policy.name == "standard_rho80_unit8")
    rates = rate_candidates(config, training, baseline)
    assert len(rates) == 9 and baseline in rates
    assert len(unique_candidates(initial + rates)) == 26
    refinements = refinement_candidates(config, baseline)
    assert len(refinements) == 6
    assert len(unique_candidates(initial + rates + refinements)) <= 32
    assert {candidate.policy.historical_fraction for candidate in initial} == {.5, .8, .95}


def test_candidate_round_trip_hash_and_optimizer_boundary():
    config, training = load_config(), load_srt_config()
    candidate = policy_candidates(config, training)[0]
    assert candidate_from_record(asdict(candidate)) == candidate
    changed = replace(candidate, lora_learning_rate=candidate.lora_learning_rate * 2)
    assert changed.content_hash != candidate.content_hash and changed.name != candidate.name
    assert changed.optimizer_config(training) == replace(training, lora_learning_rate=changed.lora_learning_rate)
    assert unique_candidates((candidate, changed, candidate)) == (candidate, changed)


def test_refinement_changes_one_policy_coordinate_and_never_rates():
    config, training = load_config(), load_srt_config()
    candidate = next(row for row in policy_candidates(config, training) if row.policy.name == "standard_rho95_unit1")
    neighbors = refinement_candidates(config, candidate)
    for neighbor in neighbors:
        assert (neighbor.lora_learning_rate, neighbor.head_learning_rate) == (candidate.lora_learning_rate, candidate.head_learning_rate)
        changes = sum(getattr(neighbor.policy, field) != getattr(candidate.policy, field)
                      for field in ("thresholds", "historical_fraction", "interval_unit"))
        assert changes <= 1
        assert 0 <= neighbor.policy.historical_fraction <= 1
        assert neighbor.policy.interval_unit >= 1
        assert tuple(sorted(set(neighbor.policy.thresholds))) == neighbor.policy.thresholds
    assert candidate in neighbors  # halving unit 1 reuses unit 1, not another training job.


def _score(identity, accuracy, nll, **updates):
    return {"candidate_hash": identity, "status": "complete", "evaluation_role": "validation", "completed_stages": 50,
            "validation_accuracy": accuracy, "validation_nll": nll, "mean_stage_accuracy": 100 - accuracy, **updates}


def test_selection_uses_final_validation_accuracy_not_mean_stage_nll_or_test():
    high_mean = _score("a", 70., .5)
    high_final = _score("b", 80., 1.0)
    assert select_winner((high_mean, high_final)) == high_final
    tied = _score("c", 80., .9)
    assert select_winner((high_mean, high_final, tied)) == tied
    canonical = _score("a", 80., .9)
    assert select_winner((tied, canonical)) == canonical
    failed = {"status": "failed", "candidate_hash": "bad"}
    assert select_winner((failed, high_final)) == high_final
    for bad in (_score("x", 90., .2, evaluation_role="test"), _score("x", 90., .2, completed_stages=16),
                _score("x", float("nan"), .2)):
        with pytest.raises(ValueError, match="validation"):
            select_winner((high_final, bad))
    with pytest.raises(ValueError, match="no numerically valid"):
        select_winner((failed,))


@pytest.mark.parametrize("updates", ({"capacity": 256}, {"interval_units": (0,)}, {"historical_fractions": (1.1,)},
                                    {"threshold_powers": ()}, {"head_rate_multipliers": (.5, 2.)},
                                    {"lora_rate_multipliers": (1., float("inf"))}))
def test_invalid_search_is_rejected_before_training(updates):
    with pytest.raises(ValueError, match="configuration"):
        replace(load_config(), **updates)


@pytest.fixture
def synthetic_search(tmp_path, monkeypatch):
    """Supply miniature sealed validation jobs, explicitly without model measurements."""
    from apm.continual.artifacts import publish_immutable_json, record_sha256
    from apm.continual.vision.imagenetr import srt_tuning as tuning
    from apm.continual.vision.imagenetr.srt_evidence import read_sealed, sealed_record

    fitting = tuple(SimpleNamespace(task_index=stage, image_id=f"fit-{stage}") for stage in range(50))
    validation = tuple(SimpleNamespace(task_index=stage, image_id=f"validation-{stage}") for stage in range(50))
    inputs = tuning.TuningInputs(SimpleNamespace(training=load_srt_config()), load_config(), tmp_path, fitting, validation,
                                 {"fit_ids_hash": record_sha256([row.image_id for row in fitting]),
                                  "validation_ids_hash": record_sha256([row.image_id for row in validation])})
    monkeypatch.setattr(tuning, "_progress", lambda *_args, **_kwargs: None)
    calls = []

    def fake_job(root, config, policy, method, training_rows, evaluation_rows, **_kwargs):
        if (root / "result.json").is_file():
            return {**read_sealed(root / "result.json"), "invocation_optimizer_steps": 0}
        calls.append(root.name)
        assert training_rows == fitting and evaluation_rows == validation and method == "srt"
        job = sealed_record({"schema_version": "imagenetr50-srt-job-v1", "protocol_hash": inputs.run.name,
                             "config_hash": config.content_hash, "policy": asdict(policy), "capacity": 128, "method": method,
                             "training_ids_hash": inputs.protocol["fit_ids_hash"],
                             "evaluation_ids_hash": inputs.protocol["validation_ids_hash"]})
        rows = tuple({"fit": {"wall_seconds": .01},
                      "evaluation": {"examples": stage, "accuracy": 70. + stage / 10, "nll": .8, "wall_seconds": .001}}
                     for stage in range(1, 51))
        result = sealed_record({"schema_version": "imagenetr50-srt-job-result-v1", "job_hash": job["content_hash"],
                                "rows": rows, "method": method, "capacity": 128, "policy": asdict(policy),
                                "image_presentations": 5100, "optimizer_steps": 100})
        publish_immutable_json(root / "job.json", job)
        publish_immutable_json(root / "result.json", result)
        return {**result, "invocation_optimizer_steps": 100}

    return inputs, fake_job, calls


def test_phase_resume_and_duplicate_recipes_do_not_repeat_training(synthetic_search):
    from apm.continual.vision.imagenetr.srt_tuning import run_search_phase
    inputs, fake_job, calls = synthetic_search
    candidates = policy_candidates(inputs.config, inputs.base.training)[:2]
    rows, winner = run_search_phase(inputs, "policy", (*candidates, candidates[0]), (), fake_job, {})
    assert len(rows) == len(calls) == 2
    resumed, selected = run_search_phase(inputs, "policy", (*candidates, candidates[0]), (), fake_job, {})
    assert resumed == rows and selected == winner and len(calls) == 2


def test_phase_recovers_interruption_and_records_numerical_failure(synthetic_search):
    from apm.continual.vision.imagenetr.srt_tuning import run_search_phase
    inputs, fake_job, calls = synthetic_search
    candidates = policy_candidates(inputs.config, inputs.base.training)[:3]

    def interrupted(root, **kwargs):
        if root.name == candidates[1].name:
            raise RuntimeError("synthetic interruption")
        return fake_job(root, **kwargs)

    with pytest.raises(RuntimeError, match="interruption"):
        run_search_phase(inputs, "policy", candidates, (), interrupted, {})
    assert calls == [candidates[0].name]

    def diverged(root, **kwargs):
        if root.name == candidates[1].name:
            raise FloatingPointError("synthetic divergence")
        return fake_job(root, **kwargs)

    rows, selected = run_search_phase(inputs, "policy", candidates, (), diverged, {})
    assert len(rows) == 3 and len(calls) == 2
    assert rows[1]["status"] == "failed" and selected["candidate_hash"] != candidates[1].content_hash
    resumed, _ = run_search_phase(inputs, "policy", candidates, (), fake_job, {})
    assert resumed == rows and len(calls) == 2  # The failed recipe is not silently retried.


def test_candidate_summary_rejects_changed_evaluation_population(synthetic_search):
    from apm.continual.artifacts import atomic_write, canonical_json_bytes
    from apm.continual.vision.imagenetr.srt_evidence import read_sealed, sealed_record
    from apm.continual.vision.imagenetr.srt_tuning import candidate_summary, run_search_phase
    inputs, fake_job, _ = synthetic_search
    candidate = policy_candidates(inputs.config, inputs.base.training)[0]
    run_search_phase(inputs, "policy", (candidate,), (), fake_job, {})
    path = inputs.run / "calibration" / candidate.name / "job.json"
    row = read_sealed(path)
    changed = sealed_record({**{key: value for key, value in row.items() if key != "content_hash"},
                             "evaluation_ids_hash": "test-not-validation"})
    atomic_write(path, canonical_json_bytes(changed))
    with pytest.raises(ValueError, match="validation-only"):
        candidate_summary(inputs, candidate)

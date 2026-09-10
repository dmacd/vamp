"""Bounded checks for explicitly fixed policies and interpretable quality scores."""

from dataclasses import replace

import pytest

from apm.continual.vision.imagenetr.srt_config import load_srt_config, presentation_budget
from apm.continual.vision.imagenetr.srt_followup import followup_conditions, load_followup_config


def test_requested_followup_has_both_profiles_and_their_uniform_controls() -> None:
    config, training = load_followup_config(), load_srt_config()
    jobs = followup_conditions(config, training)
    assert len(jobs) == 4
    assert {name for name, _, _ in jobs} == {
        f"{method}_h4096_{profile}_rho80_unit8"
        for method in ("srt", "uniform") for profile in ("standard", "strict")
    }
    assert all(policy.historical_fraction == .8 and policy.interval_unit == 8 for _, _, policy in jobs)
    assert presentation_budget(444, 23556, config.capacity) == 18160
    with pytest.raises(ValueError, match="requested"):
        replace(config, historical_fraction=.5).policies(training)


def test_probability_half_is_success_under_standard_but_failure_under_strict() -> None:
    standard, strict = load_followup_config().policies(load_srt_config())
    assert standard.quality(.5) == 3
    assert strict.quality(.5) == 2
    # These distributions have very different runner-up margins but the same
    # correct-class probability, so either profile assigns them the same quality.
    safe, narrow = (.5, .01, *([.49 / 198] * 198)), (.5, .49, *([.01 / 198] * 198))
    assert safe[0] - max(safe[1:]) > narrow[0] - max(narrow[1:])
    assert all(policy.quality(safe[0]) == policy.quality(narrow[0]) for policy in (standard, strict))

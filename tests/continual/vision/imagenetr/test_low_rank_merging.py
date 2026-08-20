import pytest
import torch

from apm.continual.vision.imagenetr.merging.common import (
    LoRAFactors,
    dense_truncated_reference,
    exact_weighted_factors,
)
from apm.continual.vision.imagenetr.merging.core_space import (
    build_core_space,
    dense_core_reference,
)
from apm.continual.vision.imagenetr.merging.core_tsv import core_tsv_merge
from apm.continual.vision.imagenetr.merging.output_drift import output_drift_merge
from apm.continual.vision.imagenetr.merging.svd import weighted_svd_merge
from apm.continual.vision.imagenetr.exact_diagnostics import selected_diagnostic_events


def _children() -> tuple[LoRAFactors, LoRAFactors]:
    generator = torch.Generator().manual_seed(17)
    return tuple(
        LoRAFactors(
            torch.randn(3, 9, generator=generator),
            torch.randn(7, 3, generator=generator),
            scale,
        )
        for scale in (0.75, 1.25)
    )


def test_exact_stacked_factors_match_arbitrary_weighted_dense_sum() -> None:
    children = _children()
    exact = exact_weighted_factors(children, (0.3, 0.7), 0.75)
    reference = 0.75 * (0.3 * children[0].dense() + 0.7 * children[1].dense())
    torch.testing.assert_close(exact.dense(), reference, rtol=2e-6, atol=2e-6)
    assert exact.rank == 6


def test_compact_svd_matches_dense_optimal_truncation() -> None:
    children = _children()
    merged, diagnostic = weighted_svd_merge(
        children, (0.3, 0.7), output_rank=4, parent_scale=1.0, merge_scale=0.75
    )
    reference = dense_truncated_reference(children, (0.3, 0.7), 4, 0.75)
    torch.testing.assert_close(merged.dense(), reference, rtol=2e-5, atol=2e-5)
    assert merged.rank == 4
    assert 0.0 <= diagnostic.retained_parameter_energy <= 1.0


def test_core_space_aligned_cores_match_dense_projection() -> None:
    children = _children()
    core = build_core_space(children)
    for child, aligned in zip(children, core.aligned_cores):
        torch.testing.assert_close(
            aligned,
            dense_core_reference(child, core.left_basis, core.right_basis),
            rtol=2e-5,
            atol=2e-5,
        )


def test_core_tsv_is_finite_and_rank_bounded() -> None:
    result = core_tsv_merge(_children(), (0.5, 0.5), output_rank=3)
    assert result.factors.rank == 3
    assert torch.isfinite(result.factors.dense()).all()
    assert result.merged_core.shape == (6, 6)


def test_output_drift_factors_reproduce_dense_projected_update() -> None:
    children = _children()
    inputs = torch.randn(40, 9, generator=torch.Generator().manual_seed(23))
    result = output_drift_merge(children, (0.5, 0.5), inputs, output_rank=3)
    raw = 0.5 * children[0].dense() + 0.5 * children[1].dense()
    output = inputs @ raw.T
    _left, _singular, right = torch.linalg.svd(output, full_matrices=False)
    directions = right[:3].T
    reference = directions @ directions.T @ raw
    torch.testing.assert_close(result.factors.dense(), reference, rtol=3e-5, atol=3e-5)
    assert result.factors.rank == 3
    assert 0.0 <= result.retained_output_energy <= 1.0


def test_exact_rank_diagnostic_selection_spans_low_mid_and_high_merges() -> None:
    events = selected_diagnostic_events()
    assert len(events) == len({event.merge_id for event in events}) == 6
    levels = tuple(event.parent.level for event in events)
    assert levels[:4] == (1, 1, 2, 2)
    assert all(level >= 3 for level in levels[4:])

import torch

from apm.continual.vision.imagenetr.router_descriptor import NodeRouterFeatures
from apm.continual.vision.imagenetr.router_merge import (
    functional_merge_diagnostics,
    svd_merge_scorers,
)
from apm.continual.vision.imagenetr.router_scores import (
    ExactLSEScorer,
    R0Scorer,
    R1Scorer,
    R3Scorer,
    RouterQuery,
    ScoringNode,
    query_public_fields,
)


def _features(response: bool = False) -> NodeRouterFeatures:
    generator = torch.Generator().manual_seed(1)
    kernels = {
        f"m{index}": torch.randn(3, 768, generator=generator)
        for index in range(8)
    } if response else {}
    return NodeRouterFeatures(torch.randn(128, generator=generator), kernels, "a" * 64, "b" * 64 if response else None)


def test_zero_response_branch_makes_r3_exactly_r1() -> None:
    generator = torch.Generator().manual_seed(2)
    r1, r3 = R1Scorer(8, seed=9), R3Scorer(8, seed=10)
    r3.load_state_dict({**r1.state_dict(), "response_weight": torch.zeros(8)})
    activations = {
        f"m{index}": torch.randn(5, 768, generator=generator)
        for index in range(8)
    }
    query = RouterQuery(tuple(f"x{index}" for index in range(5)), torch.randn(5, 768, generator=generator), activations)
    features = _features(True)
    torch.testing.assert_close(
        r1.score(query, features), r3.score(query, features), rtol=0.0, atol=0.0
    )
    assert query_public_fields() == {"image_ids", "prelogits", "cls_activations"}


def test_compact_router_svd_matches_dense_rank_16_sum() -> None:
    left, right = R1Scorer(8, seed=3), R1Scorer(8, seed=4)
    parent, diagnostics = svd_merge_scorers(left, right, (40, 60), 16, seed=5)
    assert isinstance(parent, R1Scorer)
    expected = 0.4 * (left.interaction_left @ left.interaction_right) + 0.6 * (
        right.interaction_left @ right.interaction_right
    )
    actual = parent.interaction_left @ parent.interaction_right
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    assert diagnostics.exact_rank <= 16
    assert diagnostics.retained_parameter_energy == 1.0


def test_exact_lse_preserves_collapsed_frontier_probability_mass() -> None:
    feature = _features()
    left = ScoringNode("left", R0Scorer(seed=1), feature, (0,), (0, 1, 2, 3), 10)
    right = ScoringNode("right", R0Scorer(seed=2), feature, (1,), (4, 5, 6, 7), 10)
    parent = ScoringNode(
        "parent",
        ExactLSEScorer(left, right),
        feature,
        (0, 1),
        tuple(range(8)),
        20,
    )
    query = RouterQuery(("x", "y"), torch.randn(2, 768), {})
    result = functional_merge_diagnostics(query, (left, right), 0, 1, parent)
    assert result.mean_mass_error is not None and result.mean_mass_error < 1e-12
    assert result.collapsed_kl is not None and abs(result.collapsed_kl) < 1e-12
    assert result.lse_mse is not None and result.lse_mse < 1e-12

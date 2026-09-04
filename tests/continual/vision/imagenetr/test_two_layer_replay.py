from __future__ import annotations

import torch

from apm.continual.vision.imagenetr.integrator_config import load_integrator_config
from apm.continual.vision.imagenetr.integrator_model import (
    VARIANT_SLOT_DIMS,
    create_integrator_state,
    parameter_count,
)
from apm.continual.vision.imagenetr.replay_adaptation_training import (
    create_replay_integrator_state,
)
from apm.continual.vision.imagenetr.two_layer_replay_config import (
    load_two_layer_replay_config,
)


INTEGRATOR_CONFIG = (
    "configs/vision/imagenetr/logt_prediction_integrator_full_union_ungated_v3.yaml"
)


def test_two_layer_config_freezes_the_nested_representation_comparison() -> None:
    config = load_two_layer_replay_config()
    assert config.feature_variant == "behavior_two_layer"
    assert config.diagnostic_stages == (31, 50)
    assert config.historical_capacity == 8192
    assert len(config.conditions) == 8
    assert config.added_latent.endswith("before_final_block_and_norm")
    assert config.comparison_run_hash == (
        "f32b127b633ade1345927abd10dd4ea46d3ab0259638b699c02730b85074cf63"
    )


def test_two_layer_initialization_nests_every_single_layer_parameter() -> None:
    optimization = load_integrator_config(INTEGRATOR_CONFIG).optimization
    device = torch.device("cpu")
    name = "persistent-node-adapted-behavior-common-seed-v1"
    baseline = create_integrator_state(
        name, 6, "behavior", optimization, 1993, device
    )
    expanded = create_replay_integrator_state(
        name, 6, "behavior_two_layer", optimization, 1993, device
    )
    old_dim = VARIANT_SLOT_DIMS["behavior"]
    new_dim = VARIANT_SLOT_DIMS["behavior_two_layer"]
    assert parameter_count(expanded.model) - parameter_count(baseline.model) == (
        6 * 768 * optimization.hidden_widths[0]
    )
    assert expanded.model.middle.state_dict().keys() == baseline.model.middle.state_dict().keys()
    for key, value in baseline.model.middle.state_dict().items():
        assert torch.equal(expanded.model.middle.state_dict()[key], value)
    assert torch.equal(expanded.model.output_layer.weight, baseline.model.output_layer.weight)
    assert torch.equal(expanded.model.output_layer.bias, baseline.model.output_layer.bias)
    assert torch.equal(expanded.model.input_layer.bias, baseline.model.input_layer.bias)
    for slot in range(6):
        old = baseline.model.input_layer.weight[
            :, slot * old_dim : (slot + 1) * old_dim
        ]
        new = expanded.model.input_layer.weight[
            :, slot * new_dim : (slot + 1) * new_dim
        ]
        assert torch.equal(new[:, :old_dim], old)
        assert torch.count_nonzero(new[:, old_dim:]).item() == 0


def test_arbitrary_added_latents_have_exact_zero_column_parity_at_initialization() -> None:
    optimization = load_integrator_config(INTEGRATOR_CONFIG).optimization
    device = torch.device("cpu")
    name = "nested-parity"
    baseline = create_integrator_state(
        name, 2, "behavior", optimization, 1993, device
    )
    expanded = create_replay_integrator_state(
        name, 2, "behavior_two_layer", optimization, 1993, device
    )
    baseline.model.eval()
    expanded.model.eval()
    generator = torch.Generator().manual_seed(17)
    with torch.no_grad():
        weight = torch.randn(
            baseline.model.output_layer.weight.shape, generator=generator
        )
        bias = torch.randn(
            baseline.model.output_layer.bias.shape, generator=generator
        )
        baseline.model.output_layer.weight.copy_(weight)
        baseline.model.output_layer.bias.copy_(bias)
        expanded.model.output_layer.weight.copy_(weight)
        expanded.model.output_layer.bias.copy_(bias)
    old_dim = VARIANT_SLOT_DIMS["behavior"]
    old_features = torch.randn(5, 2, old_dim, generator=generator)
    extra = torch.randn(5, 2, 768, generator=generator)
    expanded_features = torch.cat((old_features, extra), dim=2).flatten(1)
    baseline_logits = torch.randn(5, 200, generator=generator)
    mask = torch.ones(200, dtype=torch.bool)
    assert torch.equal(
        baseline.model(old_features.flatten(1), baseline_logits, mask),
        expanded.model(expanded_features, baseline_logits, mask),
    )

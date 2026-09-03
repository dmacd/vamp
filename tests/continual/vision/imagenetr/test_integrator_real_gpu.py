from __future__ import annotations

from pathlib import Path

import pytest
import torch

from apm.continual.vision.imagenetr.integrator_artifacts import bootstrap_integrator
from apm.continual.vision.imagenetr.integrator_model import create_integrator_state
from apm.continual.vision.imagenetr.integrator_observations import (
    BehaviorNode,
    build_frontier_tensors,
)
from apm.continual.vision.imagenetr.model import create_pinned_backbone
from apm.continual.vision.imagenetr.router_features import (
    test_transform_hash as _test_transform_hash,
)


@pytest.mark.integration
@pytest.mark.benchmark
def test_real_integrator_bootstrap_and_bf16_step() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    bootstrap = bootstrap_integrator(
        "configs/vision/imagenetr/logt_prediction_integrator_full_union_v2.yaml"
    )
    device = torch.device("cuda:0")
    state = create_integrator_state(
        "real-smoke",
        bootstrap.config.maximum_levels,
        "scores",
        bootstrap.config.optimization,
        bootstrap.config.seed,
        device,
    )
    features = torch.randn(2, state.model.input_dim, device=device)
    baseline = torch.randn(2, 200, device=device)
    mask = torch.ones(200, dtype=torch.bool, device=device)
    labels = torch.tensor((0, 1), device=device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = torch.nn.functional.cross_entropy(
            state.model(features, baseline, mask), labels
        )
    loss.backward()
    state.optimizer.step()
    assert torch.isfinite(loss)


@pytest.mark.integration
@pytest.mark.benchmark
def test_real_sealed_node_behavior_cache_matches_direct_forward(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    bootstrap = bootstrap_integrator(
        "configs/vision/imagenetr/logt_prediction_integrator_full_union_v2.yaml"
    )
    device = torch.device("cuda:0")
    node = BehaviorNode.from_sealed(bootstrap.sealed_tree.final.nodes[0])
    rows = bootstrap.manifest.select("train")[:2]
    arguments = (
        (node,),
        (0,),
        1,
        bootstrap.config.data_root / "imagenet-r",
        rows,
        bootstrap.test_transform,
        _test_transform_hash(bootstrap.primary_config.input_size),
        bootstrap.protocol.model_manifest_hash,
        lambda: create_pinned_backbone(bootstrap.checkpoint),
        tmp_path / "behavior_cache",
        bootstrap.primary_config.lora_rank,
        bootstrap.primary_config.lora_alpha,
        bootstrap.primary_config.cosine_scale,
        2,
        0,
        device,
    )
    direct = build_frontier_tensors(*arguments)
    cached = build_frontier_tensors(*arguments)
    assert direct.base_example_forwards == 2
    assert direct.node_example_forwards == 2
    assert cached.base_example_forwards == 0
    assert cached.node_example_forwards == 0
    assert cached.cache_hits == 4
    assert cached.cache_misses == 0
    assert torch.equal(
        direct.observations("behavior_base").features,
        cached.observations("behavior_base").features,
    )

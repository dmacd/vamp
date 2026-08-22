from pathlib import Path
from types import SimpleNamespace

import torch

from apm.continual.vision.imagenetr.router_evaluation import CentroidNodeScorer
from apm.continual.vision.imagenetr.router_scores import move_scorer
from apm.continual.vision.imagenetr.router_workflow import _phase0_preflight


def test_phase0_preflight_reuses_original_measured_timing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cache = tmp_path / "features" / "cls_activations" / "fixture" / "tensors.safetensors"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"cache")
    fit = tuple(range(19_200))
    validation = tuple(range(19_200, 24_000))
    bootstrap = SimpleNamespace(
        store=SimpleNamespace(run=tmp_path),
        split=SimpleNamespace(fit_image_ids=fit, validation_image_ids=validation),
        base=SimpleNamespace(inventory_hash="a" * 64),
    )
    train = SimpleNamespace(
        image_ids=tuple(range(24_000)),
        cls_activations={f"module-{index}": object() for index in range(8)},
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index: "fixture-gpu")

    original = _phase0_preflight(bootstrap, train, 10.0)
    resumed = _phase0_preflight(bootstrap, train, 0.01)

    assert resumed == original
    assert resumed["activation_cache_seconds"] == 10.0
    assert resumed["activation_rows_per_second"] == 2_400.0


def test_centroid_baseline_supports_standard_scorer_device_traversal() -> None:
    scorer = CentroidNodeScorer((0, 1), torch.eye(2))

    move_scorer(scorer, torch.device("cpu"))

    assert scorer.centroids.device.type == "cpu"

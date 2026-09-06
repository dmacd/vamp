from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from apm.continual.artifacts import canonical_json_bytes, file_sha256, record_sha256
from apm.continual.vision.imagenetr.frontier_adaptation_reporting import (
    LINEAR_PRECLASSIFIER_LABEL,
    _linear_preclassifier_history,
    _linear_preclassifier_summary,
    _validated_linear_preclassifier_control,
)
from apm.continual.vision.imagenetr.frontier_linear_preclassifier_config import (
    load_frontier_linear_preclassifier_config,
)
from apm.continual.vision.imagenetr.frontier_linear_preclassifier_model import (
    SingleLayerPreclassifierIntegrator,
)
from apm.continual.vision.imagenetr.frontier_linear_preclassifier_workflow import (
    FrontierLinearPreclassifierProtocol,
    _material_paths,
)
from apm.continual.vision.imagenetr.heads import ClassifierRows


CONFIG = Path(
    "configs/vision/imagenetr/logt_frontier_linear_preclassifier_v13.yaml"
)


def _classifier_rows() -> tuple[ClassifierRows, ...]:
    boundaries = (0, 4, 12, 28, 60, 124)
    generator = torch.Generator().manual_seed(1993)
    return tuple(
        ClassifierRows(
            tuple(range(first, last)),
            torch.randn(last - first, 768, generator=generator),
            torch.randn(last - first, generator=generator),
        )
        for first, last in zip(boundaries[:-1], boundaries[1:], strict=True)
    )


def test_linear_config_freezes_architecture_data_and_training() -> None:
    config = load_frontier_linear_preclassifier_config(CONFIG)
    assert config.input_dimension == 3_840
    assert config.integrator_parameters == 768_200
    assert config.frontier_lora_parameters == 6_635_520
    assert config.active_parameters == 7_403_720
    assert config.full_fit_examples == 12_194
    assert config.validation_examples == 3_049
    assert config.macro_peak_learning_rate == config.integrator_peak_learning_rate
    with pytest.raises(ValueError, match="linear pre-classifier"):
        replace(config, feature_normalization="layer_norm")


def test_single_layer_starts_as_exact_local_classifier_union() -> None:
    classifiers = _classifier_rows()
    model = SingleLayerPreclassifierIntegrator(classifiers)
    features = tuple(torch.randn(7, 768) for _classifier in classifiers)
    seen = torch.zeros(200, dtype=torch.bool)
    seen[:124] = True
    actual = model(torch.cat(features, dim=1), seen)
    expected = torch.full((7, 200), -torch.inf)
    for classifier, node_features in zip(classifiers, features, strict=True):
        expected[:, torch.tensor(classifier.class_ids)] = F.linear(
            node_features, classifier.weight, classifier.bias
        )
    torch.testing.assert_close(actual[:, seen], expected[:, seen], atol=5e-5, rtol=1e-5)
    assert bool(torch.isneginf(actual[:, ~seen]).all())
    assert sum(parameter.numel() for parameter in model.parameters()) == 768_200
    for node_index, classifier in enumerate(classifiers):
        owned = torch.tensor(classifier.class_ids)
        for other_index in range(5):
            if node_index == other_index:
                continue
            first = other_index * 768
            assert int(torch.count_nonzero(model.affine.weight[owned, first : first + 768])) == 0


def test_linear_protocol_and_material_surface_are_content_addressed() -> None:
    identities = tuple(f"{index:064x}" for index in range(1, 14))
    protocol = FrontierLinearPreclassifierProtocol(*identities)
    assert len(protocol.content_hash) == 64
    assert replace(protocol, config_hash="f" * 64).content_hash != protocol.content_hash
    config = load_frontier_linear_preclassifier_config(CONFIG)
    material = _material_paths(Path.cwd(), CONFIG.resolve(), config.parent_config)
    names = {path.name for path in material}
    assert {
        "frontier_linear_preclassifier_config.py",
        "frontier_linear_preclassifier_model.py",
        "frontier_linear_preclassifier_workflow.py",
        "imagenetr50_frontier_linear_preclassifier_protocol.md",
        "run_frontier_linear_preclassifier_local.sh",
    } <= names
    assert "frontier_adaptation_reporting.py" not in names
    assert all(path.is_file() for path in material)


def test_report_authenticates_linear_result_and_history(tmp_path: Path) -> None:
    history = tmp_path / "controls/linear/history.jsonl"
    history.parent.mkdir(parents=True)
    history.write_text(
        "".join(
            canonical_json_bytes(
                {
                    "epoch": epoch,
                    "validation_accuracy": 70.0 + epoch / 10,
                    "validation_nll": 1.5 - epoch / 100,
                }
            ).decode("utf-8")
            + "\n"
            for epoch in range(1, 51)
        ),
        encoding="utf-8",
    )
    architecture = {
        "active_nodes": 5,
        "feature_dimension": 768,
        "feature_normalization": "none_after_pinned_vit_pre_logits",
        "feature_source": "node_preclassifier",
        "frontier_lora_parameters": 6_635_520,
        "initialization": "exact_local_classifier_union",
        "input_dimension": 3_840,
        "input_order": "ascending_frontier_level",
        "integrator_kind": "single_affine",
        "integrator_parameters": 768_200,
        "output_classes": 200,
        "source_rank": 16,
        "train_base_vit": False,
        "train_frontier_loras": True,
        "train_integrator": True,
        "train_node_classifiers": False,
        "trainable_parameters": 7_403_720,
    }
    fit = {
        "best_nll_epoch": 40,
        "best_validation_nll": 0.6,
        "epochs": 50,
        "fixed_epoch": 5,
        "fixed_image_presentations": 60_970,
        "fixed_validation_accuracy": 78.0,
        "fixed_validation_nll": 0.9,
        "image_presentations": 609_700,
        "max_accuracy_epoch": 42,
        "max_validation_accuracy": 82.0,
        "peak_vram_bytes": 10,
        "train_accuracy_at_best": 95.0,
        "train_nll_at_best": 0.2,
        "trainable_parameters": 7_403_720,
        "validation_accuracy_at_best_nll": 81.0,
        "validation_nll_at_max_accuracy": 0.7,
        "wall_seconds": 1.0,
    }
    parent = {"content_hash": "a" * 64}
    core = {
        "architecture": architecture,
        "fit": fit,
        "history": str(history.relative_to(tmp_path)),
        "history_sha256": file_sha256(history),
        "parent_result_hash": parent["content_hash"],
        "schema_version": "imagenetr50-frontier-linear-preclassifier-control-v1",
        "test_evaluations": 0,
    }
    result_path = tmp_path / "evaluations/frontier_linear_preclassifier.json"
    result_path.parent.mkdir()
    result_path.write_bytes(
        canonical_json_bytes({**core, "content_hash": record_sha256(core)})
    )
    control = _validated_linear_preclassifier_control(tmp_path, parent)
    assert control is not None
    summary = _linear_preclassifier_summary(control)
    assert summary["condition"] == LINEAR_PRECLASSIFIER_LABEL
    assert summary["fixed_validation_accuracy"] == 78.0
    assert len(_linear_preclassifier_history(tmp_path, control)) == 50

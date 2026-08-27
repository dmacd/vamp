from pathlib import Path

import torch

from apm.continual.nce_tre_evidence import (
    ConditionalEvidenceCNN,
    EvidenceTrainingConfig,
    freeze_evidence_model,
    materialize_evidence_model,
    quantize_raw_images,
    replacement_probability,
    sample_discrete_waymark,
    sample_reference_images,
    train_evidence_cnn,
)
from apm.experiments.vamp_logt_evidence_config import CalibrationConfig
from apm.experiments.vamp_logt_ratio_calibration import run_ratio_calibration


def test_evidence_cnn_keeps_address_cnn_backbone_capacity_and_scalar_output() -> None:
    model = ConditionalEvidenceCNN(4)
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert model.conv1.in_channels == 1 and model.conv1.out_channels == 32
    assert model.conv2.in_channels == 32 and model.conv2.out_channels == 64
    assert model.embedding.in_features == 64 * 7 * 7
    assert model.embedding.out_features == 128
    assert model.scalar.in_features == 128 and model.scalar.out_features == 1
    images = torch.rand(3, 1, 28, 28)
    assert model(images, torch.tensor([0, 1, 3])).shape == (3,)
    all_logits = model.all_bridge_logits(images)
    assert all_logits.shape == (3, 4)
    expected = torch.stack(
        tuple(
            model(images, torch.full((3,), bridge, dtype=torch.int64))
            for bridge in range(4)
        ),
        dim=1,
    )
    torch.testing.assert_close(all_logits, expected)


def test_uint8_raw_images_are_not_requantized_or_clipped() -> None:
    images = torch.arange(256, dtype=torch.uint8).repeat(7)[: 2 * 28 * 28].reshape(2, 1, 28, 28)
    quantized = quantize_raw_images(images)
    assert torch.equal(quantized, images)
    assert quantized.data_ptr() != images.data_ptr()
    assert replacement_probability(0, 8) == 1.0 / 784.0
    assert replacement_probability(8, 8) == 1.0


def test_evidence_training_is_deterministic_and_commits_frozen_state() -> None:
    raw = torch.randint(0, 256, (8, 1, 28, 28), dtype=torch.uint8)
    reference = torch.randint(0, 256, (12, 1, 28, 28), dtype=torch.uint8)
    config = EvidenceTrainingConfig(bridges=2, epochs=1, batch_size=4)
    left = train_evidence_cnn(raw, reference, config, 17, torch.device("cpu"))
    right = train_evidence_cnn(raw, reference, config, 17, torch.device("cpu"))
    left_state = freeze_evidence_model(left.model)
    right_state = freeze_evidence_model(right.model)
    assert left.source_example_updates == 8
    assert left.reference_examples == 8
    assert tuple(name for name, _tensor in left_state.parameters) == tuple(
        name for name, _tensor in right_state.parameters
    )
    for (_name, left_value), (_other_name, right_value) in zip(
        left_state.parameters, right_state.parameters
    ):
        torch.testing.assert_close(left_value, right_value, rtol=0.0, atol=0.0)
    restored = materialize_evidence_model(left_state)
    assert not any(parameter.requires_grad for parameter in restored.parameters())


def test_empirical_reference_endpoint_is_one_complete_sampled_base_image() -> None:
    sources = torch.zeros((6, 1, 28, 28), dtype=torch.uint8)
    reference = torch.stack(
        tuple(torch.full((1, 28, 28), value, dtype=torch.uint8) for value in (17, 83, 241))
    )
    generator = torch.Generator().manual_seed(31)
    donors = sample_reference_images(reference, len(sources), torch.device("cpu"), generator)
    endpoint = sample_discrete_waymark(
        sources,
        donors,
        torch.full((len(sources),), 4, dtype=torch.int64),
        4,
        1.0 / 784.0,
        generator,
    )
    torch.testing.assert_close(endpoint, donors.to(torch.float32) / 255.0)
    assert set(donors[:, 0, 0, 0].tolist()) <= {17, 83, 241}


def test_small_normalized_multimodal_problem_recovers_offsets_and_exposes_direct_saturation(
    tmp_path: Path,
) -> None:
    config = CalibrationConfig(
        dimensions=12,
        component_probabilities=(0.1, 0.9),
        reference_probability=0.5,
        tre_bridges=4,
        training_steps=300,
        batch_size=128,
        evaluation_examples=2_048,
        replicas=3,
        learning_rate=0.001,
        weight_decay=0.0001,
        signed_bias_max_nats=0.3,
        tre_rmse_max_nats=0.6,
        interseed_rmse_max_nats=0.5,
        direct_to_tre_rmse_ratio_min=1.3,
    )
    summary = run_ratio_calibration(
        config, tmp_path / "calibration", torch.device("cpu"), False
    )
    assert summary["passed"] is True
    assert all(summary["gates"].values())
    assert summary["direct_mean_rmse_nats"] > summary["tre_mean_rmse_nats"]
    assert summary["triangle_minimum_slack_nats"] >= -1.0e-5

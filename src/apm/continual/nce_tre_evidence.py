"""Balanced NCE/TRE density-ratio estimation from raw quantized images."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class EvidenceTrainingConfig:
    """Fixed optimizer, waymark, and replay settings for one evidence model."""

    bridges: int
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1.0e-8
    initial_replacement_probability: float = 1.0 / 784.0

    def __post_init__(self) -> None:
        if (
            self.bridges < 1
            or self.epochs < 1
            or self.batch_size < 1
            or self.learning_rate <= 0.0
            or self.weight_decay < 0.0
            or not 0.0 <= self.beta1 < 1.0
            or not 0.0 <= self.beta2 < 1.0
            or self.epsilon <= 0.0
            or not 0.0 < self.initial_replacement_probability < 1.0
        ):
            raise ValueError("invalid evidence-training configuration")


@dataclass(frozen=True, slots=True)
class EvidenceTrainingResult:
    """One frozen evidence CNN and its exact example accounting."""

    model: "ConditionalEvidenceCNN"
    final_loss: float
    source_example_updates: int
    reference_examples: int

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.final_loss)
            or self.source_example_updates < 1
            or self.reference_examples != self.source_example_updates
            or any(parameter.requires_grad for parameter in self.model.parameters())
        ):
            raise ValueError("invalid completed evidence training result")


@dataclass(frozen=True, slots=True)
class FrozenEvidenceState:
    """Immutable CPU parameters for one committed conditional evidence CNN."""

    bridges: int
    parameters: tuple[tuple[str, Tensor], ...]

    def __post_init__(self) -> None:
        names = tuple(name for name, _tensor in self.parameters)
        if (
            self.bridges < 1
            or not self.parameters
            or len(set(names)) != len(names)
            or any(
                tensor.device.type != "cpu"
                or tensor.requires_grad
                or not torch.isfinite(tensor).all()
                for _name, tensor in self.parameters
            )
        ):
            raise ValueError("invalid frozen evidence-model parameters")


def freeze_evidence_model(model: "ConditionalEvidenceCNN") -> FrozenEvidenceState:
    """Copy a trained evidence CNN into immutable, device-neutral state."""
    return FrozenEvidenceState(
        model.bridges,
        tuple(
            (name, value.detach().cpu().clone())
            for name, value in model.state_dict().items()
        ),
    )


def materialize_evidence_model(state: FrozenEvidenceState) -> "ConditionalEvidenceCNN":
    """Materialize a frozen inference module from committed evidence state."""
    model = ConditionalEvidenceCNN(state.bridges)
    model.load_state_dict(dict(state.parameters), strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


class ConditionalEvidenceCNN(nn.Module):
    """AddressCNN-width scalar discriminator with bridge-conditioned FiLM."""

    def __init__(self, bridges: int) -> None:
        super().__init__()
        if bridges < 1:
            raise ValueError("an evidence CNN requires at least one bridge")
        self.bridges = bridges
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.embedding = nn.Linear(64 * 7 * 7, 128)
        self.bridge_film = nn.Embedding(bridges, 2 * 128)
        self.scalar = nn.Linear(128, 1)

    def forward(self, images: Tensor, bridge_indices: Tensor) -> Tensor:
        """Return one adjacent log-density-ratio estimate per image."""
        if (
            images.ndim != 4
            or images.shape[1:] != (1, 28, 28)
            or bridge_indices.shape != (images.shape[0],)
            or bridge_indices.dtype != torch.int64
            or bool((bridge_indices < 0).any())
            or bool((bridge_indices >= self.bridges).any())
        ):
            raise ValueError("evidence CNN input shapes or bridge indices are invalid")
        hidden = self._backbone(images)
        return self._conditioned_scalar(hidden, bridge_indices)

    def _backbone(self, images: Tensor) -> Tensor:
        hidden = F.max_pool2d(F.relu(self.conv1(images)), 2)
        hidden = F.max_pool2d(F.relu(self.conv2(hidden)), 2).flatten(1)
        return self.embedding(hidden)

    def _conditioned_scalar(self, hidden: Tensor, bridge_indices: Tensor) -> Tensor:
        scale, shift = self.bridge_film(bridge_indices).chunk(2, dim=1)
        conditioned = F.relu((1.0 + torch.tanh(scale)) * hidden + shift)
        return self.scalar(conditioned).squeeze(1)

    def all_bridge_logits(self, images: Tensor) -> Tensor:
        """Evaluate the shared backbone once, then every fixed conditional scalar."""
        if images.ndim != 4 or images.shape[1:] != (1, 28, 28):
            raise ValueError("all-bridge evidence inputs must be NCHW MNIST images")
        batch_size = images.shape[0]
        hidden = self._backbone(images)
        bridge_indices = torch.arange(self.bridges, device=images.device)
        scale, shift = self.bridge_film(bridge_indices).chunk(2, dim=1)
        conditioned = F.relu(
            (1.0 + torch.tanh(scale))[None, :, :] * hidden[:, None, :]
            + shift[None, :, :]
        )
        return self.scalar(conditioned).reshape(batch_size, self.bridges)


class ConditionalVectorEvidence(nn.Module):
    """Small conditional scorer used only by the normalized calibration problem."""

    def __init__(self, dimensions: int, bridges: int, hidden_width: int = 128) -> None:
        super().__init__()
        if dimensions < 1 or bridges < 1 or hidden_width < 1:
            raise ValueError("invalid conditional vector evidence dimensions")
        self.bridges = bridges
        self.network = nn.Sequential(
            nn.Linear(dimensions, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, hidden_width),
            nn.SiLU(),
        )
        self.bridge_film = nn.Embedding(bridges, 2 * hidden_width)
        self.scalar = nn.Linear(hidden_width, 1)

    def forward(self, values: Tensor, bridge_indices: Tensor) -> Tensor:
        """Return conditioned scalar logits for a generic vector batch."""
        hidden = self.network(values)
        if bridge_indices.shape != (values.shape[0],):
            raise ValueError("vector evidence bridge indices do not match the batch")
        scale, shift = self.bridge_film(bridge_indices).chunk(2, dim=1)
        return self.scalar(F.silu((1.0 + torch.tanh(scale)) * hidden + shift)).squeeze(1)


def quantize_raw_images(images: Tensor) -> Tensor:
    """Project normalized raw pixels onto the shared 8-bit evidence alphabet."""
    if images.ndim not in (3, 4) or images.shape[-2:] != (28, 28):
        raise ValueError("raw evidence images must end in 28x28 pixels")
    with_channel = images[:, None] if images.ndim == 3 else images
    if with_channel.shape[1] != 1:
        raise ValueError("raw evidence images must be single-channel")
    if with_channel.dtype == torch.uint8:
        return with_channel.detach().cpu().clone()
    if not torch.is_floating_point(with_channel) or not torch.isfinite(with_channel).all():
        raise ValueError("raw evidence images must be finite and single-channel")
    return torch.round(with_channel.clamp(0.0, 1.0) * 255.0).to(torch.uint8)


def replacement_probability(
    bridge_index: int | Tensor,
    bridges: int,
    initial_probability: float = 1.0 / 784.0,
) -> float | Tensor:
    """Return the linear coordinate-replacement schedule including both endpoints."""
    if bridges < 1 or not 0.0 < initial_probability < 1.0:
        raise ValueError("invalid replacement schedule")
    if isinstance(bridge_index, Tensor):
        if bool((bridge_index < 0).any()) or bool((bridge_index > bridges).any()):
            raise ValueError("bridge index lies outside the replacement schedule")
        return initial_probability + (1.0 - initial_probability) * bridge_index / bridges
    if not 0 <= bridge_index <= bridges:
        raise ValueError("bridge index lies outside the replacement schedule")
    return initial_probability + (1.0 - initial_probability) * bridge_index / bridges


def sample_discrete_waymark(
    raw_images: Tensor,
    reference_images: Tensor,
    bridge_indices: Tensor,
    bridges: int,
    initial_probability: float,
    generator: torch.Generator,
) -> Tensor:
    """Sample normalized waymarks by replacing pixels with paired reference images."""
    if (
        raw_images.dtype != torch.uint8
        or raw_images.ndim != 4
        or raw_images.shape[1:] != (1, 28, 28)
        or reference_images.dtype != torch.uint8
        or reference_images.shape != raw_images.shape
        or reference_images.device != raw_images.device
    ):
        raise ValueError("waymark sources must be quantized NCHW images")
    if bridge_indices.shape != (raw_images.shape[0],) or bridge_indices.dtype != torch.int64:
        raise ValueError("waymark bridge indices do not match the image batch")
    probabilities = replacement_probability(
        bridge_indices.to(torch.float32), bridges, initial_probability
    ).reshape(-1, 1, 1, 1)
    mask = torch.rand(
        raw_images.shape,
        device=raw_images.device,
        generator=generator,
    ) < probabilities
    return torch.where(mask, reference_images, raw_images).to(torch.float32) / 255.0


def sample_reference_images(
    reference_raw_images: Tensor | None,
    examples: int,
    device: torch.device,
    generator: torch.Generator,
) -> Tensor:
    """Draw complete uint8 reference images from uniform pixels or an empirical bank."""
    if examples < 1:
        raise ValueError("reference sampling requires at least one example")
    if reference_raw_images is None:
        return torch.randint(
            0,
            256,
            (examples, 1, 28, 28),
            dtype=torch.int16,
            device=device,
            generator=generator,
        ).to(torch.uint8)
    if (
        reference_raw_images.dtype != torch.uint8
        or reference_raw_images.ndim != 4
        or reference_raw_images.shape[1:] != (1, 28, 28)
        or len(reference_raw_images) < 2
    ):
        raise ValueError("empirical reference images must be a nontrivial uint8 NCHW bank")
    indices = torch.randint(
        0,
        len(reference_raw_images),
        (examples,),
        device=device,
        generator=generator,
    ).cpu()
    return reference_raw_images.cpu()[indices].to(device)


def balanced_nce_loss(positive_logits: Tensor, negative_logits: Tensor) -> Tensor:
    """Return equal-prior binary NCE loss with no unknown offset correction."""
    if positive_logits.shape != negative_logits.shape or positive_logits.ndim != 1:
        raise ValueError("balanced NCE logits must be matching vectors")
    return 0.5 * (
        F.binary_cross_entropy_with_logits(positive_logits, torch.ones_like(positive_logits))
        + F.binary_cross_entropy_with_logits(negative_logits, torch.zeros_like(negative_logits))
    )


def train_evidence_cnn(
    raw_images: Tensor,
    reference_raw_images: Tensor | None,
    config: EvidenceTrainingConfig,
    seed: int,
    device: torch.device,
    show_progress: bool = False,
) -> EvidenceTrainingResult:
    """Train one fresh full-capacity evidence model for a node replay distribution."""
    if seed < 0 or raw_images.shape[0] < 2:
        raise ValueError("evidence training requires a nonnegative seed and at least two images")
    quantized = quantize_raw_images(raw_images).cpu()
    reference = (
        None
        if reference_raw_images is None
        else quantize_raw_images(reference_raw_images).cpu()
    )
    torch.manual_seed(seed)
    model = ConditionalEvidenceCNN(config.bridges).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2),
        eps=config.epsilon,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 1)
    try:
        from tqdm.auto import tqdm
    except ImportError as error:  # pragma: no cover - vision environment gate
        raise RuntimeError("tqdm is required by the vision environment") from error
    final_loss = math.nan
    model.train()
    for epoch in tqdm(
        range(config.epochs),
        desc=f"evidence K={config.bridges}",
        disable=not show_progress,
        leave=False,
    ):
        order = np.random.default_rng(seed + epoch).permutation(len(quantized))
        for offset in range(0, len(order), config.batch_size):
            ids = torch.from_numpy(order[offset : offset + config.batch_size].astype(np.int64))
            sources = quantized[ids].to(device)
            bridge_indices = torch.randint(
                0,
                config.bridges,
                (len(ids),),
                dtype=torch.int64,
                device=device,
                generator=generator,
            )
            negative_sources = sources[torch.randperm(len(ids), device=device, generator=generator)]
            reference_images = sample_reference_images(
                reference,
                len(ids),
                device,
                generator,
            )
            positives = sample_discrete_waymark(
                sources,
                reference_images,
                bridge_indices,
                config.bridges,
                config.initial_replacement_probability,
                generator,
            )
            negatives = sample_discrete_waymark(
                negative_sources,
                reference_images,
                bridge_indices + 1,
                config.bridges,
                config.initial_replacement_probability,
                generator,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = balanced_nce_loss(
                model(positives, bridge_indices),
                model(negatives, bridge_indices),
            )
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().item())
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    updates = config.epochs * len(quantized)
    return EvidenceTrainingResult(model.cpu(), final_loss, updates, updates)


def evidence_scores(
    model: ConditionalEvidenceCNN,
    raw_images: Tensor,
    device: torch.device,
    batch_size: int = 2_048,
) -> Tensor:
    """Return summed log-ratio evidence for each uncorrupted raw image."""
    if batch_size < 1:
        raise ValueError("evidence scoring batch size must be positive")
    quantized = quantize_raw_images(raw_images)
    target = model.to(device)
    rows = []
    with torch.inference_mode():
        for offset in range(0, len(quantized), batch_size):
            values = quantized[offset : offset + batch_size].to(device).to(torch.float32) / 255.0
            rows.append(target.all_bridge_logits(values).sum(dim=1).cpu())
    model.cpu()
    return torch.cat(rows)


__all__ = [
    "ConditionalEvidenceCNN",
    "ConditionalVectorEvidence",
    "EvidenceTrainingConfig",
    "EvidenceTrainingResult",
    "FrozenEvidenceState",
    "balanced_nce_loss",
    "evidence_scores",
    "freeze_evidence_model",
    "materialize_evidence_model",
    "quantize_raw_images",
    "replacement_probability",
    "sample_discrete_waymark",
    "sample_reference_images",
    "train_evidence_cnn",
]

"""Frozen CNN and deterministic Addressable Rotated MNIST feature tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from apm.continual.addressing_first import StoredExampleTable
from apm.continual.artifacts import file_sha256, load_canonical_json, publish_immutable_json
from apm.continual.vision.imagenetr.checkpoints import atomic_torch_save
from apm.data.mnist.loader import MnistArrays, load_mnist
from apm.experiments.vamp_af_config import VampAFConfig


class AddressCNN(nn.Module):
    """The exact small CNN whose normalized penultimate activations form addresses."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.embedding = nn.Linear(64 * 7 * 7, 128)
        self.classifier = nn.Linear(128, 10)

    def trunk_features(self, images: Tensor) -> Tensor:
        """Return the frozen 3,136-dimensional input to the adaptable suffix."""
        hidden = F.max_pool2d(F.relu(self.conv1(images)), 2)
        hidden = F.max_pool2d(F.relu(self.conv2(hidden)), 2)
        return hidden.flatten(1)

    def features_from_trunk(self, trunk_features: Tensor) -> Tensor:
        """Return the unnormalized 128-dimensional penultimate activation."""
        return F.relu(self.embedding(trunk_features))

    def features(self, images: Tensor) -> Tensor:
        """Return the unnormalized 128-dimensional penultimate activation."""
        return self.features_from_trunk(self.trunk_features(images))

    def forward(self, images: Tensor) -> Tensor:
        """Return ordinary ten-class MNIST logits."""
        return self.classifier(self.features(images))


@dataclass(frozen=True, slots=True)
class BaseCheckpoint:
    """Loaded frozen base and authenticated training-selection metadata."""

    model: AddressCNN
    path: Path
    sha256: str
    selected_epochs: int
    test_accuracy: float


@dataclass(frozen=True, slots=True)
class FeatureTables:
    """Canonical main-train and complete transformed-test frozen tables."""

    train: StoredExampleTable
    test: StoredExampleTable
    source_train_indices: tuple[int, ...]


def resolved_device(name: str) -> torch.device:
    """Resolve the configured device and fail explicit unavailable CUDA requests."""
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not visible")
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else name if name != "auto" else "cpu")


def load_arrays(config: VampAFConfig) -> MnistArrays:
    """Load the existing IDX cache without permitting implicit network access."""
    return load_mnist(root=config.data_root, allow_download=False, npz_cache_path=None)


def train_or_load_base(
    config: VampAFConfig,
    arrays: MnistArrays,
    run_root: Path,
    device: torch.device,
) -> BaseCheckpoint:
    """Select convergence on 50k/10k, retrain on all MNIST, and freeze one checkpoint."""
    checkpoint = run_root / "base" / "model.pt"
    manifest = run_root / "base" / "manifest.json"
    if checkpoint.is_file() and manifest.is_file():
        record = load_canonical_json(manifest)
        if file_sha256(checkpoint) != record["checkpoint_sha256"]:
            raise ValueError("shared CNN checkpoint bytes changed")
        model = AddressCNN()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(payload, strict=True)
        _freeze(model)
        return BaseCheckpoint(
            model,
            checkpoint,
            str(record["checkpoint_sha256"]),
            int(record["selected_epochs"]),
            float(record["test_accuracy"]),
        )

    torch.manual_seed(config.base.seed)
    np.random.seed(config.base.seed)
    train_indices, validation_indices = _stratified_validation_split(
        arrays.train_labels, config.base.validation_examples, config.base.seed
    )
    selection_model = AddressCNN().to(device)
    optimizer = torch.optim.AdamW(
        selection_model.parameters(),
        lr=config.base.learning_rate,
        weight_decay=config.base.weight_decay,
    )
    best_loss, best_epoch, stale_epochs = math.inf, 1, 0
    for epoch in range(1, config.base.maximum_epochs + 1):
        _train_cnn_epoch(
            selection_model,
            optimizer,
            arrays.train_images[train_indices],
            arrays.train_labels[train_indices],
            config.base.batch_size,
            config.base.seed + epoch,
            device,
            f"base selection {epoch}/{config.base.maximum_epochs}",
            config.runtime.progress,
        )
        validation_loss, _accuracy = _evaluate_cnn(
            selection_model,
            arrays.train_images[validation_indices],
            arrays.train_labels[validation_indices],
            config.base.batch_size * 4,
            device,
        )
        print(f"Base selection epoch {epoch}: validation loss={validation_loss:.6f}", flush=True)
        if validation_loss < best_loss - config.base.minimum_improvement:
            best_loss, best_epoch, stale_epochs = validation_loss, epoch, 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.base.patience:
                break

    torch.manual_seed(config.base.seed)
    model = AddressCNN().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.base.learning_rate,
        weight_decay=config.base.weight_decay,
    )
    for epoch in range(best_epoch):
        _train_cnn_epoch(
            model,
            optimizer,
            arrays.train_images,
            arrays.train_labels,
            config.base.batch_size,
            config.base.seed + 10_000 + epoch,
            device,
            f"base final {epoch + 1}/{best_epoch}",
            config.runtime.progress,
        )
    _test_loss, test_accuracy = _evaluate_cnn(
        model, arrays.test_images, arrays.test_labels, config.base.batch_size * 4, device
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(
        checkpoint,
        {key: value.detach().cpu() for key, value in model.state_dict().items()},
    )
    checkpoint_hash = file_sha256(checkpoint)
    publish_immutable_json(
        manifest,
        {
            "checkpoint_sha256": checkpoint_hash,
            "schema_version": "vamp-af-base-v1",
            "seed": config.base.seed,
            "selected_epochs": best_epoch,
            "selection_validation_loss": best_loss,
            "test_accuracy": test_accuracy,
        },
    )
    _freeze(model)
    return BaseCheckpoint(model.cpu(), checkpoint, checkpoint_hash, best_epoch, test_accuracy)


def build_feature_tables(
    config: VampAFConfig,
    arrays: MnistArrays,
    checkpoint: BaseCheckpoint,
    run_root: Path,
    device: torch.device,
) -> FeatureTables:
    """Build or load the shared 10k-per-context train and complete test feature cache."""
    cache = run_root / "cache" / "features.pt"
    manifest = run_root / "cache" / "features.json"
    if cache.is_file() and manifest.is_file():
        record = load_canonical_json(manifest)
        if (
            record.get("schema_version") != "vamp-af-feature-cache-v2"
            or record["base_checkpoint_sha256"] != checkpoint.sha256
            or file_sha256(cache) != record["cache_sha256"]
        ):
            raise ValueError("frozen feature cache identity changed")
        payload = torch.load(cache, map_location="cpu", weights_only=True)
        return FeatureTables(
            _table_from_payload(payload, "train"),
            _table_from_payload(payload, "test"),
            tuple(int(value) for value in payload["source_train_indices"].tolist()),
        )

    source_indices = _balanced_indices(
        arrays.train_labels, config.data.main_examples_per_context, config.base.seed
    )
    model = checkpoint.model.to(device)
    train = _context_feature_table(
        model,
        arrays.train_images[np.asarray(source_indices)],
        arrays.train_labels[np.asarray(source_indices)],
        config,
        device,
        "train features",
    )
    test = _context_feature_table(
        model,
        arrays.test_images,
        arrays.test_labels,
        config,
        device,
        "test features",
    )
    payload = {
        **_table_payload(train, "train"),
        **_table_payload(test, "test"),
        "source_train_indices": torch.as_tensor(source_indices, dtype=torch.int64),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(cache, payload)
    cache_hash = file_sha256(cache)
    publish_immutable_json(
        manifest,
        {
            "base_checkpoint_sha256": checkpoint.sha256,
            "cache_sha256": cache_hash,
            "schema_version": "vamp-af-feature-cache-v2",
            "test_examples": int(test.embeddings.shape[0]),
            "train_examples": int(train.embeddings.shape[0]),
            "trunk_feature_dim": int(train.trunk_features.shape[1]),
        },
    )
    checkpoint.model.cpu()
    return FeatureTables(train, test, source_indices)


def pass_training_table(
    tables: FeatureTables,
    examples_per_context: int,
    seed: int,
) -> StoredExampleTable:
    """Select paired identities and permute examples within each blocked context."""
    context_ids = tuple(int(value) for value in torch.unique(tables.train.context_ids).tolist())
    selected = []
    for context_id in context_ids:
        candidates = torch.nonzero(tables.train.context_ids == context_id, as_tuple=True)[0].numpy()
        labels = tables.train.labels[candidates].numpy()
        context_selection = np.asarray(
            [candidates[index] for index in _balanced_indices(labels, examples_per_context, 0)],
            dtype=np.int64,
        )
        order = np.random.default_rng(seed * 10_000 + context_id).permutation(context_selection)
        selected.extend(int(value) for value in order)
    ids = torch.as_tensor(selected, dtype=torch.int64)
    return StoredExampleTable(
        tables.train.embeddings[ids],
        tables.train.trunk_features[ids],
        tables.train.base_logits[ids],
        tables.train.labels[ids],
        tables.train.context_ids[ids],
        torch.arange(len(selected), dtype=torch.int64),
    )


def _stratified_validation_split(
    labels: np.ndarray, validation_count: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    validation = np.asarray(_balanced_indices(labels, validation_count, seed), dtype=np.int64)
    mask = np.ones(labels.shape[0], dtype=bool)
    mask[validation] = False
    return np.flatnonzero(mask), validation


def _balanced_indices(labels: np.ndarray, count: int, seed: int) -> tuple[int, ...]:
    values = np.asarray(labels, dtype=np.int64)
    unique = tuple(int(value) for value in np.unique(values))
    target = min(int(count), values.shape[0])
    base, remainder = divmod(target, len(unique))
    rng = np.random.default_rng(seed)
    selected = tuple(
        int(index)
        for label_index, label in enumerate(unique)
        for index in rng.choice(
            np.flatnonzero(values == label),
            size=base + int(label_index < remainder),
            replace=False,
        )
    )
    return tuple(sorted(selected))


def _train_cnn_epoch(
    model: AddressCNN,
    optimizer: torch.optim.Optimizer,
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    seed: int,
    device: torch.device,
    description: str,
    show_progress: bool,
) -> None:
    order = torch.from_numpy(np.random.default_rng(seed).permutation(len(labels)).astype(np.int64))
    dataset = TensorDataset(
        torch.from_numpy(images).unsqueeze(1)[order],
        torch.from_numpy(labels)[order],
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    try:
        from tqdm.auto import tqdm
    except ImportError as error:  # pragma: no cover - environment gate
        raise RuntimeError("tqdm is required by the vision environment") from error
    model.train()
    for batch_images, batch_labels in tqdm(loader, desc=description, disable=not show_progress, leave=False):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(batch_images.to(device)), batch_labels.to(device))
        loss.backward()
        optimizer.step()


def _evaluate_cnn(
    model: AddressCNN,
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    loss_sum, correct = 0.0, 0
    with torch.inference_mode():
        for offset in range(0, len(labels), batch_size):
            batch_images = torch.from_numpy(images[offset : offset + batch_size]).unsqueeze(1).to(device)
            batch_labels = torch.from_numpy(labels[offset : offset + batch_size]).to(device)
            logits = model(batch_images)
            loss_sum += float(F.cross_entropy(logits, batch_labels, reduction="sum").item())
            correct += int((logits.argmax(dim=1) == batch_labels).sum().item())
    return loss_sum / len(labels), correct / len(labels)


def _context_feature_table(
    model: AddressCNN,
    images: np.ndarray,
    labels: np.ndarray,
    config: VampAFConfig,
    device: torch.device,
    description: str,
) -> StoredExampleTable:
    try:
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.functional import rotate
        from tqdm.auto import tqdm
    except ImportError as error:  # pragma: no cover - environment gate
        raise RuntimeError("torchvision and tqdm are required by the vision environment") from error
    image_tensor = torch.from_numpy(images).unsqueeze(1)
    embeddings, logits, trunk_rows, shifted_labels, contexts = [], [], [], [], []
    model.eval()
    batch_size = config.base.batch_size * 4
    with torch.inference_mode():
        for context_id, (angle, shift) in enumerate(zip(config.data.rotations_deg, config.data.label_shifts)):
            for offset in tqdm(
                range(0, len(labels), batch_size),
                desc=f"{description} context {context_id}",
                disable=not config.runtime.progress,
                leave=False,
            ):
                transformed = rotate(
                    image_tensor[offset : offset + batch_size],
                    float(angle),
                    interpolation=InterpolationMode.BILINEAR,
                    expand=False,
                    fill=0.0,
                ).to(device)
                trunk_features = model.trunk_features(transformed)
                raw_embedding = model.features_from_trunk(trunk_features)
                trunk_rows.append(trunk_features.float().cpu())
                embeddings.append(F.normalize(raw_embedding, p=2, dim=1, eps=1.0e-12).cpu())
                logits.append(model.classifier(raw_embedding).float().cpu())
                count = transformed.shape[0]
                shifted_labels.append(
                    (torch.from_numpy(labels[offset : offset + count]).to(torch.int64) + shift) % 10
                )
                contexts.append(torch.full((count,), context_id, dtype=torch.int64))
    rows = sum(tensor.shape[0] for tensor in embeddings)
    return StoredExampleTable(
        torch.cat(embeddings).to(torch.float32),
        torch.cat(trunk_rows).to(torch.float32),
        torch.cat(logits).to(torch.float32),
        torch.cat(shifted_labels),
        torch.cat(contexts),
        torch.arange(rows, dtype=torch.int64),
    )


def _table_payload(table: StoredExampleTable, prefix: str) -> dict[str, Tensor]:
    return {
        f"{prefix}_embeddings": table.embeddings,
        f"{prefix}_trunk_features": table.trunk_features,
        f"{prefix}_base_logits": table.base_logits,
        f"{prefix}_labels": table.labels,
        f"{prefix}_context_ids": table.context_ids,
        f"{prefix}_stream_steps": table.stream_steps,
    }


def _table_from_payload(payload: dict[str, Tensor], prefix: str) -> StoredExampleTable:
    return StoredExampleTable(
        payload[f"{prefix}_embeddings"],
        payload[f"{prefix}_trunk_features"],
        payload[f"{prefix}_base_logits"],
        payload[f"{prefix}_labels"],
        payload[f"{prefix}_context_ids"],
        payload[f"{prefix}_stream_steps"],
    )


def _freeze(model: AddressCNN) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


__all__ = [
    "AddressCNN",
    "BaseCheckpoint",
    "FeatureTables",
    "build_feature_tables",
    "load_arrays",
    "pass_training_table",
    "resolved_device",
    "train_or_load_base",
]

"""Frozen-base query and compact R3 activation caches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable, Mapping, Sequence

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from apm.continual.artifacts import record_sha256
from apm.continual.vision.imagenetr.data import (
    DatasetManifest,
    ImageRecord,
    ManifestDataset,
)
from apm.continual.vision.imagenetr.lora import LoRALinear
from apm.continual.vision.imagenetr.model import AdapterVisionModel
from apm.continual.vision.imagenetr.proxy_memory import TensorCache
from apm.continual.vision.imagenetr.router_artifacts import RouterStore
from apm.continual.vision.imagenetr.router_config import RouterConfig
from apm.continual.vision.imagenetr.router_descriptor import selected_response_modules


@dataclass(frozen=True, slots=True)
class RouterFeatureUniverse:
    """Stable image sequence with task-free frozen-base router inputs."""

    image_ids: tuple[str, ...]
    labels: Tensor
    task_ids: Tensor
    prelogits: Tensor
    cls_activations: dict[str, Tensor]

    def __post_init__(self) -> None:
        rows = len(self.image_ids)
        if (
            rows < 1
            or len(set(self.image_ids)) != rows
            or tuple(self.labels.shape) != (rows,)
            or tuple(self.task_ids.shape) != (rows,)
            or tuple(self.prelogits.shape) != (rows, 768)
            or any(tuple(value.shape) != (rows, 768) for value in self.cls_activations.values())
            or not torch.isfinite(self.prelogits).all()
            or any(not torch.isfinite(value).all() for value in self.cls_activations.values())
        ):
            raise ValueError("invalid frozen router feature universe")

    @property
    def index(self) -> dict[str, int]:
        return {image_id: index for index, image_id in enumerate(self.image_ids)}

    def indices(self, image_ids: Sequence[str]) -> Tensor:
        """Resolve an exact requested identity sequence or fail closed."""
        index = self.index
        if len(set(image_ids)) != len(image_ids) or any(value not in index for value in image_ids):
            raise ValueError("router feature view contains unknown or duplicate image IDs")
        return torch.tensor([index[value] for value in image_ids], dtype=torch.long)


def test_transform_hash(input_size: int = 224) -> str:
    """Return the exact sealed deterministic transform identity."""
    return record_sha256(
        {
            "input_size": input_size,
            "normalization": None,
            "schema_version": "imagenetr50-test-transform-v1",
        }
    )


def _existing_prelogits(
    inference_run: Path,
    model_hash: str,
    rows: Sequence[ImageRecord],
    split: str,
) -> Tensor:
    image_ids = tuple(row.image_id for row in rows)
    values = {
        "image_ids_hash": record_sha256(list(image_ids)),
        "model_hash": model_hash,
        "split": split,
        "transform_hash": f"{test_transform_hash()}:frozen-router-{split if split == 'test' else 'training'}",
    }
    cache = TensorCache(
        inference_run / "cache" / "frozen_features",
        "imagenetr50-frozen-router-features-v1",
    )

    def missing() -> Mapping[str, Tensor]:
        raise FileNotFoundError("sealed frozen prelogit cache is missing")

    tensors, reused = cache.get_or_compute(values, missing)
    if not reused or tuple(tensors["features"].shape) != (len(rows), 768):
        raise ValueError("sealed frozen prelogit cache does not match its image universe")
    return tensors["features"].to(torch.float32)


def _capture_selected_cls(
    backbone_factory: Callable[[], torch.nn.Module],
    prepared_root: Path,
    rows: Sequence[ImageRecord],
    transform: object,
    config: RouterConfig,
    device: torch.device,
) -> dict[str, Tensor]:
    model = AdapterVisionModel(backbone_factory(), (0,), 16, 16, 0.0, config.seed)
    model.to(device).eval()
    modules = dict(model.named_modules())
    selected = selected_response_modules(config)
    if any(not isinstance(modules.get(name), LoRALinear) for name in selected):
        raise ValueError("pinned model does not expose the configured R3 modules")
    dataset = ManifestDataset(prepared_root, rows, transform, 0, 0)
    loader = DataLoader(
        dataset,
        batch_size=config.feature_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    batches: dict[str, list[Tensor]] = {name: [] for name in selected}
    from tqdm.auto import tqdm

    for images, _labels, _ids in tqdm(
        loader,
        desc=f"R3 frozen CLS cache ({len(rows):,} {getattr(rows[0], 'split', 'images')})",
        unit="batch",
    ):
        captured: dict[str, Tensor] = {}
        handles = tuple(
            modules[name].register_forward_pre_hook(
                lambda _module, inputs, name=name: captured.__setitem__(
                    name, inputs[0][:, 0, :].detach()
                )
            )
            for name in selected
        )
        try:
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                model.features(images.to(device, non_blocking=True))
        finally:
            for handle in handles:
                handle.remove()
        if tuple(captured) != selected:
            raise ValueError("R3 feature capture missed or reordered selected modules")
        for name in selected:
            batches[name].append(captured[name].to(device="cpu", dtype=torch.bfloat16))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {name: torch.cat(tuple(batches[name])) for name in selected}


def _cached_cls(
    store: RouterStore,
    model_hash: str,
    split: str,
    rows: Sequence[ImageRecord],
    backbone_factory: Callable[[], torch.nn.Module] | None,
    prepared_root: Path,
    transform: object,
    config: RouterConfig,
    device: torch.device,
) -> dict[str, Tensor]:
    selected = selected_response_modules(config)
    image_ids = tuple(row.image_id for row in rows)
    values = {
        "dtype": config.response_dtype,
        "image_ids_hash": record_sha256(list(image_ids)),
        "model_hash": model_hash,
        "modules": list(selected),
        "pooling": "cls",
        "split": split,
        "transform_hash": test_transform_hash(),
    }
    cache = TensorCache(
        store.run / "features" / "cls_activations",
        "imagenetr50-router-cls-activation-cache-v1",
    )

    def compute() -> Mapping[str, Tensor]:
        if backbone_factory is None:
            raise FileNotFoundError("R3 CLS cache is missing and no backbone factory was supplied")
        return {
            f"activation_{index:02d}": value
            for index, value in enumerate(
                _capture_selected_cls(
                    backbone_factory,
                    prepared_root,
                    rows,
                    transform,
                    config,
                    device,
                ).values()
            )
        }

    tensors, _ = cache.get_or_compute(values, compute)
    activations = dict(zip(selected, tensors.values()))
    if any(tuple(value.shape) != (len(rows), 768) for value in activations.values()):
        raise ValueError("R3 CLS cache has the wrong image or feature dimension")
    return activations


def load_router_feature_universe(
    store: RouterStore,
    inference_run: Path,
    manifest: DatasetManifest,
    model_hash: str,
    split: str,
    include_response: bool,
    config: RouterConfig,
    backbone_factory: Callable[[], torch.nn.Module] | None = None,
    prepared_root: Path | None = None,
    transform: object | None = None,
    device: torch.device = torch.device("cpu"),
) -> RouterFeatureUniverse:
    """Load exact prelogits and optionally compute/reuse compact R3 inputs."""
    if split not in {"train", "test"}:
        raise ValueError("router feature split must be train or test")
    rows = manifest.select(split)
    activations = (
        _cached_cls(
            store,
            model_hash,
            split,
            rows,
            backbone_factory,
            prepared_root or Path("."),
            transform,
            config,
            device,
        )
        if include_response
        else {}
    )
    return RouterFeatureUniverse(
        tuple(row.image_id for row in rows),
        torch.tensor([row.remapped_class_index for row in rows], dtype=torch.long),
        torch.tensor([row.task_index for row in rows], dtype=torch.long),
        _existing_prelogits(inference_run, model_hash, rows, split),
        activations,
    )


__all__ = [
    "RouterFeatureUniverse",
    "load_router_feature_universe",
    "test_transform_hash",
]

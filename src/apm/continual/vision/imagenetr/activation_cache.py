"""Frozen-base adapted-layer input activation caching for output-drift merges."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Callable, Mapping

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from apm.continual.artifacts import record_sha256
from apm.continual.vision.imagenetr.data import DatasetManifest, ImageRecord, ManifestDataset
from apm.continual.vision.imagenetr.model import AdapterVisionModel, capture_adapted_inputs
from apm.continual.vision.imagenetr.proxy_memory import TensorCache, require_training_only


class FrozenActivationProvider:
    """Callable training-proxy-only cache over all 24 frozen-base layer inputs."""

    def __init__(
        self,
        cache_root: str | Path,
        manifest: DatasetManifest,
        prepared_root: str | Path,
        backbone_factory: Callable[[], torch.nn.Module],
        transform: object,
        model_hash: str,
        transform_hash: str,
        rank: int,
        alpha: int,
        batch_size: int,
        num_workers: int,
        device: torch.device,
    ) -> None:
        self.cache = TensorCache(cache_root, "imagenetr50-frozen-proxy-activations-v1")
        self.manifest = manifest
        self.prepared_root = Path(prepared_root)
        self.backbone_factory = backbone_factory
        self.transform = transform
        self.model_hash = model_hash
        self.transform_hash = transform_hash
        self.rank = rank
        self.alpha = alpha
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device

    def __call__(self, image_ids: tuple[str, ...]) -> Mapping[str, Tensor]:
        """Load or compute exact frozen-base inputs for one child-proxy union."""
        require_training_only(image_ids, self.manifest, "output-drift proxy")
        rows = self.manifest.select("train", image_ids=image_ids)
        if {row.image_id for row in rows} != set(image_ids):
            raise ValueError("output-drift proxy identities cannot be resolved")
        tensors, _reused = self.cache.get_or_compute(
            {
                "image_ids_hash": record_sha256(list(image_ids)),
                "model_hash": self.model_hash,
                "transform_hash": self.transform_hash,
            },
            lambda: self._compute(rows),
        )
        return tensors

    def _compute(self, rows: tuple[ImageRecord, ...]) -> dict[str, Tensor]:
        model = AdapterVisionModel(
            self.backbone_factory(),
            (0,),
            self.rank,
            self.alpha,
            0.0,
            0,
        )
        model.to(self.device).eval()
        dataset = ManifestDataset(self.prepared_root, rows, self.transform, 0, 0)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
        )
        batches: dict[str, list[Tensor]] = {}
        for images, _labels, _image_ids in loader:
            captured = capture_adapted_inputs(model, images.to(self.device, non_blocking=True))
            for module, values in captured.items():
                batches.setdefault(module, []).append(values.to(torch.bfloat16))
        del model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return {
            module: torch.cat(tuple(values), dim=0)
            for module, values in sorted(batches.items())
        }


__all__ = ["FrozenActivationProvider"]

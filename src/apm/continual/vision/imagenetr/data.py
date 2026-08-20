"""Immutable ImageNet-R download, split, ImageFolder, and dataset surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from hashlib import md5, sha256
from pathlib import Path, PurePosixPath
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
import json
import os
import shutil
import tarfile
import tempfile
import time
import urllib.request

import numpy as np

from apm.continual.artifacts import (
    atomic_write,
    canonical_json_bytes,
    file_sha256,
    publish_immutable_json,
    record_sha256,
)
from apm.continual.vision.imagenetr.constants import (
    IMAGENET_R_MD5,
    IMAGENET_R_SHA256,
    IMAGENET_R_URL,
)
from apm.continual.vision.imagenetr.manifests import require_sealed_manifest, sealed_manifest


IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})


@dataclass(frozen=True, slots=True)
class ImageRecord:
    """One permanent image identity and its frozen protocol assignments."""

    image_id: str
    source_relative_path: str
    prepared_relative_path: str
    image_sha256: str
    original_class_name: str
    original_class_index: int
    remapped_class_index: int
    task_index: int
    split: str
    priority: str
    size_bytes: int

    def __post_init__(self) -> None:
        if (
            len(self.image_id) != 64
            or len(self.image_sha256) != 64
            or len(self.priority) != 64
            or self.split not in {"train", "test"}
            or not 0 <= self.original_class_index < 200
            or not 0 <= self.remapped_class_index < 200
            or self.task_index != self.remapped_class_index // 4
            or self.size_bytes < 1
        ):
            raise ValueError("invalid ImageNet-R image record")

    def as_record(self) -> dict[str, object]:
        """Return a canonical JSON-compatible image row."""
        return {
            "image_id": self.image_id,
            "image_sha256": self.image_sha256,
            "original_class_index": self.original_class_index,
            "original_class_name": self.original_class_name,
            "prepared_relative_path": self.prepared_relative_path,
            "priority": self.priority,
            "remapped_class_index": self.remapped_class_index,
            "size_bytes": self.size_bytes,
            "source_relative_path": self.source_relative_path,
            "split": self.split,
            "task_index": self.task_index,
        }


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """The complete frozen 24,000/6,000 ImageNet-R experiment split."""

    archive_md5: str
    archive_sha256: str
    class_order: tuple[int, ...]
    images: tuple[ImageRecord, ...]
    seed: int

    def __post_init__(self) -> None:
        if (
            self.archive_md5 != IMAGENET_R_MD5
            or self.archive_sha256 != IMAGENET_R_SHA256
            or tuple(sorted(self.class_order)) != tuple(range(200))
            or len(self.images) != 30_000
            or sum(row.split == "train" for row in self.images) != 24_000
            or sum(row.split == "test" for row in self.images) != 6_000
            or len({row.image_id for row in self.images}) != len(self.images)
            or len({row.prepared_relative_path for row in self.images}) != len(self.images)
        ):
            raise ValueError("dataset manifest violates the frozen 80/20 protocol")
        by_class = defaultdict(set)
        for image in self.images:
            by_class[image.original_class_index].add(image.remapped_class_index)
        if len(by_class) != 200 or any(len(values) != 1 for values in by_class.values()):
            raise ValueError("class remapping is incomplete or inconsistent")

    @property
    def content_hash(self) -> str:
        """Return the frozen split and image-byte identity."""
        return record_sha256(self.as_record(include_hash=False))

    def as_record(self, include_hash: bool = True) -> dict[str, object]:
        """Return the complete canonical dataset manifest."""
        core: dict[str, object] = {
            "archive_md5": self.archive_md5,
            "archive_sha256": self.archive_sha256,
            "class_order": list(self.class_order),
            "images": [image.as_record() for image in self.images],
            "schema_version": "imagenetr50-dataset-v1",
            "seed": self.seed,
            "test_images": 6_000,
            "total_images": 30_000,
            "train_images": 24_000,
        }
        return {**core, "content_hash": self.content_hash} if include_hash else core

    def select(
        self,
        split: str,
        tasks: Iterable[int] | None = None,
        image_ids: Iterable[str] | None = None,
    ) -> tuple[ImageRecord, ...]:
        """Return stable-path-ordered rows matching one frozen subset."""
        if split not in {"train", "test"}:
            raise ValueError("dataset split must be train or test")
        task_set = None if tasks is None else frozenset(tasks)
        id_set = None if image_ids is None else frozenset(image_ids)
        return tuple(
            row
            for row in self.images
            if row.split == split
            and (task_set is None or row.task_index in task_set)
            and (id_set is None or row.image_id in id_set)
        )


def pycil_class_order(classes: int = 200, seed: int = 1993) -> tuple[int, ...]:
    """Reproduce PyCIL's legacy global-NumPy seeded permutation exactly."""
    if classes < 1 or seed < 0:
        raise ValueError("class count and seed must be positive")
    return tuple(int(value) for value in np.random.RandomState(seed).permutation(classes))


def task_classes(class_order: Sequence[int], classes_per_task: int = 4) -> tuple[tuple[int, ...], ...]:
    """Return original class IDs grouped in their remapped task order."""
    order = tuple(class_order)
    if classes_per_task < 1 or len(order) % classes_per_task:
        raise ValueError("class order does not divide into equal tasks")
    if tuple(sorted(order)) != tuple(range(len(order))):
        raise ValueError("class order is not a permutation")
    return tuple(
        order[offset : offset + classes_per_task]
        for offset in range(0, len(order), classes_per_task)
    )


def largest_remainder_counts(
    class_counts: Mapping[str, int],
    selected_total: int,
) -> dict[str, int]:
    """Allocate an exact global count proportionally with stable largest remainders."""
    if not class_counts or any(count < 1 for count in class_counts.values()):
        raise ValueError("class counts must be positive")
    total = sum(class_counts.values())
    if not 0 <= selected_total <= total:
        raise ValueError("selected total is outside dataset size")
    ideals = {
        name: Fraction(count * selected_total, total)
        for name, count in class_counts.items()
    }
    allocation = {name: int(value) for name, value in ideals.items()}
    remaining = selected_total - sum(allocation.values())
    priority = sorted(
        ideals,
        key=lambda name: (-(ideals[name] - allocation[name]), name),
    )
    return {
        name: allocation[name] + int(name in frozenset(priority[:remaining]))
        for name in sorted(allocation)
    }


def _stream_hash(path: Path, algorithm: str) -> str:
    digest = md5() if algorithm == "md5" else sha256()
    with path.open("rb") as source:
        while chunk := source.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive(target: str | Path = Path("data/imagenetr50/downloads/imagenet-r.tar")) -> Path:
    """Download the public archive with resumable progress and dual hash gates."""
    destination = Path(target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        _validate_archive(destination)
        return destination
    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.is_file() else 0
    request = urllib.request.Request(IMAGENET_R_URL)
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    with urllib.request.urlopen(request, timeout=60) as response:
        resumed = offset > 0 and response.status == 206
        mode, completed = ("ab", offset) if resumed else ("wb", 0)
        total = int(response.headers.get("Content-Length", "0")) + completed
        started = time.monotonic()
        with partial.open(mode) as output:
            while chunk := response.read(4 * 1024 * 1024):
                output.write(chunk)
                completed += len(chunk)
                elapsed = max(time.monotonic() - started, 1.0e-6)
                rate = (completed - offset) / elapsed
                eta = (total - completed) / rate if rate and total else 0.0
                print(
                    f"\rDataset download: {completed / 2**30:.2f}/{total / 2**30:.2f} GiB "
                    f"({rate / 2**20:.1f} MiB/s, ETA {eta / 60:.1f}m)",
                    end="",
                    flush=True,
                )
            output.flush()
            os.fsync(output.fileno())
    print(flush=True)
    os.replace(partial, destination)
    _validate_archive(destination)
    return destination


def _validate_archive(path: Path) -> None:
    if _stream_hash(path, "md5") != IMAGENET_R_MD5:
        raise ValueError("ImageNet-R archive MD5 differs from the public release")
    if _stream_hash(path, "sha256") != IMAGENET_R_SHA256:
        raise ValueError("ImageNet-R archive SHA-256 differs from the frozen protocol")


def _safe_member(member: tarfile.TarInfo) -> bool:
    path = PurePosixPath(member.name)
    return (
        not path.is_absolute()
        and ".." not in path.parts
        and not member.issym()
        and not member.islnk()
        and not member.isdev()
    )


def extract_archive(archive: str | Path, target_parent: str | Path) -> Path:
    """Safely extract the authenticated source tree once and publish atomically."""
    source, parent = Path(archive), Path(target_parent)
    _validate_archive(source)
    target = parent / "imagenet-r"
    if target.is_dir():
        return target
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".imagenet-r-source.", dir=parent))
    try:
        with tarfile.open(source, "r:*") as bundle:
            members = tuple(bundle.getmembers())
            if not members or not all(_safe_member(member) for member in members):
                raise ValueError("ImageNet-R archive contains an unsafe member")
            bundle.extractall(temporary, members=members)
        extracted = temporary / "imagenet-r"
        if not extracted.is_dir():
            raise ValueError("ImageNet-R archive lacks its canonical root")
        os.rename(extracted, target)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return target


def _source_images(source_root: Path) -> dict[str, tuple[Path, ...]]:
    classes = tuple(sorted(path for path in source_root.iterdir() if path.is_dir()))
    if len(classes) != 200:
        raise ValueError(f"ImageNet-R source has {len(classes)} rather than 200 classes")
    result = {
        directory.name: tuple(
            path
            for path in sorted(directory.rglob("*"))
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        for directory in classes
    }
    if sum(map(len, result.values())) != 30_000 or any(not paths for paths in result.values()):
        raise ValueError("ImageNet-R source does not contain the canonical 30,000 images")
    return result


def build_split_manifest(source_root: str | Path, seed: int = 1993) -> DatasetManifest:
    """Hash all images and deterministically freeze the exact 24k/6k membership."""
    root = Path(source_root).resolve()
    by_class = _source_images(root)
    order = pycil_class_order(200, seed)
    remapped = {original: position for position, original in enumerate(order)}
    class_names = tuple(sorted(by_class))
    test_counts = largest_remainder_counts(
        {name: len(paths) for name, paths in by_class.items()}, 6_000
    )
    rows: list[ImageRecord] = []
    for original_index, class_name in enumerate(class_names):
        candidates = []
        for path in by_class[class_name]:
            relative = path.relative_to(root).as_posix()
            digest = file_sha256(path)
            priority = sha256(
                f"imagenetr50-split-v1\0{seed}\0{relative}\0{digest}".encode()
            ).hexdigest()
            image_id = sha256(f"imagenet-r\0{relative}\0{digest}".encode()).hexdigest()
            candidates.append((priority, relative, digest, image_id, path.stat().st_size))
        selected_test = frozenset(
            item[3] for item in sorted(candidates)[: test_counts[class_name]]
        )
        new_class = remapped[original_index]
        rows.extend(
            ImageRecord(
                image_id=image_id,
                source_relative_path=relative,
                prepared_relative_path=(
                    f"{'test' if image_id in selected_test else 'train'}/"
                    f"{class_name}/{Path(relative).relative_to(class_name).as_posix()}"
                ),
                image_sha256=digest,
                original_class_name=class_name,
                original_class_index=original_index,
                remapped_class_index=new_class,
                task_index=new_class // 4,
                split="test" if image_id in selected_test else "train",
                priority=priority,
                size_bytes=size,
            )
            for priority, relative, digest, image_id, size in candidates
        )
    return DatasetManifest(
        archive_md5=IMAGENET_R_MD5,
        archive_sha256=IMAGENET_R_SHA256,
        class_order=order,
        images=tuple(sorted(rows, key=lambda row: row.image_id)),
        seed=seed,
    )


def materialize_imagefolder(
    source_root: str | Path,
    target_root: str | Path,
    manifest: DatasetManifest,
) -> Path:
    """Atomically hard-link the frozen split into the E2-LoRA ImageFolder layout."""
    source, target = Path(source_root).resolve(), Path(target_root).resolve()
    manifest_path = target / "dataset_manifest.json"
    if manifest_path.is_file():
        persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
        if persisted != manifest.as_record():
            raise ValueError("prepared ImageFolder manifest changed")
        validate_prepared_dataset(target, manifest)
        return target
    if target.exists():
        raise ValueError("unsealed ImageNet-R prepared directory already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".imagenet-r-prepared.", dir=target.parent))
    try:
        for row in manifest.images:
            origin = source / row.source_relative_path
            destination = temporary / row.prepared_relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(origin, destination)
        publish_immutable_json(temporary / "dataset_manifest.json", manifest.as_record())
        os.rename(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    validate_prepared_dataset(target, manifest)
    return target


def validate_prepared_dataset(root: str | Path, manifest: DatasetManifest) -> None:
    """Fail closed on membership, path, image bytes, or train/test leakage changes."""
    prepared = Path(root)
    train_ids = {row.image_id for row in manifest.images if row.split == "train"}
    test_ids = {row.image_id for row in manifest.images if row.split == "test"}
    if train_ids & test_ids:
        raise ValueError("train and test image identities overlap")
    missing = tuple(
        row.prepared_relative_path
        for row in manifest.images
        if not (prepared / row.prepared_relative_path).is_file()
    )
    if missing:
        raise ValueError(f"prepared dataset is missing {len(missing)} files")
    actual = {
        path.relative_to(prepared).as_posix()
        for split in ("train", "test")
        for path in (prepared / split).rglob("*")
        if path.is_file()
    }
    expected = {row.prepared_relative_path for row in manifest.images}
    if actual != expected:
        raise ValueError("prepared ImageFolder membership differs from its manifest")
    changed = tuple(
        row.prepared_relative_path
        for row in manifest.images
        if (prepared / row.prepared_relative_path).stat().st_size != row.size_bytes
        or file_sha256(prepared / row.prepared_relative_path) != row.image_sha256
    )
    if changed:
        raise ValueError(f"prepared ImageFolder has {len(changed)} changed image files")


def prepare_dataset(data_root: str | Path, seed: int = 1993) -> tuple[Path, DatasetManifest]:
    """Run download, extraction, hashing, split, and hard-link publication gates."""
    root = Path(data_root).resolve()
    archive = download_archive(root / "downloads" / "imagenet-r.tar")
    source = extract_archive(archive, root / "source")
    prepared = root / "imagenet-r"
    persisted = prepared / "dataset_manifest.json"
    if persisted.is_file():
        manifest = load_dataset_manifest(persisted)
    else:
        manifest = build_split_manifest(source, seed)
    materialize_imagefolder(source, prepared, manifest)
    return prepared, manifest


def load_dataset_manifest(path: str | Path) -> DatasetManifest:
    """Load and fully validate one canonical frozen dataset manifest."""
    payload = Path(path).read_bytes()
    record = json.loads(payload)
    if payload != canonical_json_bytes(record):
        raise ValueError("dataset manifest is not canonical JSON")
    supplied = record.pop("content_hash", None)
    images = tuple(ImageRecord(**row) for row in record.pop("images"))
    schema = record.pop("schema_version", None)
    train_images = record.pop("train_images", None)
    test_images = record.pop("test_images", None)
    total_images = record.pop("total_images", None)
    manifest = DatasetManifest(
        archive_md5=str(record["archive_md5"]),
        archive_sha256=str(record["archive_sha256"]),
        class_order=tuple(int(value) for value in record["class_order"]),
        images=images,
        seed=int(record["seed"]),
    )
    if (
        schema != "imagenetr50-dataset-v1"
        or (train_images, test_images, total_images) != (24_000, 6_000, 30_000)
        or supplied != manifest.content_hash
    ):
        raise ValueError("dataset manifest identity or summary changed")
    return manifest


def deterministic_bottom_k(
    rows: Sequence[ImageRecord],
    count: int,
    namespace: str,
) -> tuple[str, ...]:
    """Select an order-independent permanent hash reservoir from training rows only."""
    if count < 0 or not namespace or any(row.split != "train" for row in rows):
        raise ValueError("reservoir selection requires training rows and a valid count")
    ranked = sorted(
        rows,
        key=lambda row: (
            sha256(f"{namespace}\0{row.image_id}\0{row.priority}".encode()).hexdigest(),
            row.image_id,
        ),
    )
    return tuple(row.image_id for row in ranked[:count])


class ManifestDataset:
    """Stateless-augmentation PyTorch dataset over explicit manifest rows."""

    def __init__(
        self,
        root: str | Path,
        rows: Sequence[ImageRecord],
        transform: Callable[[object], object],
        augmentation_seed: int,
        epoch: int,
    ) -> None:
        self.root = Path(root)
        self.rows = tuple(rows)
        self.transform = transform
        self.augmentation_seed = augmentation_seed
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[object, int, str]:
        from PIL import Image
        import torch

        row = self.rows[index]
        seed = int(
            sha256(
                f"augmentation-v1\0{self.augmentation_seed}\0{self.epoch}\0{row.image_id}".encode()
            ).hexdigest()[:16],
            16,
        )
        with (self.root / row.prepared_relative_path).open("rb") as source:
            image = Image.open(source).convert("RGB")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            transformed = self.transform(image)
        return transformed, row.remapped_class_index, row.image_id


def image_transforms(input_size: int = 224) -> tuple[object, object]:
    """Return the exact unnormalized E2-LoRA train and test preprocessing."""
    try:
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode
    except ImportError as error:  # pragma: no cover - vision environment gate
        raise RuntimeError("torchvision is required by the vision environment") from error
    train = transforms.Compose(
        (
            transforms.RandomResizedCrop(
                input_size, scale=(0.05, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        )
    )
    test = transforms.Compose(
        (
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
        )
    )
    return train, test


__all__ = [
    "DatasetManifest",
    "ImageRecord",
    "ManifestDataset",
    "build_split_manifest",
    "deterministic_bottom_k",
    "download_archive",
    "image_transforms",
    "largest_remainder_counts",
    "load_dataset_manifest",
    "materialize_imagefolder",
    "prepare_dataset",
    "pycil_class_order",
    "task_classes",
    "validate_prepared_dataset",
]

"""Preparation and validation of the pinned TreeLoRA TRACE dataset."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
from urllib.request import urlopen

from apm.continual.artifacts import (
    canonical_json_bytes,
    file_sha256,
    fsync_directory,
    publish_immutable_bytes,
    publish_immutable_json,
    record_sha256,
    require_sha256,
)
from apm.continual.trace.protocol import (
    ARRIVAL_COUNT,
    ARRIVALS_PER_TASK,
    DATASET_ARCHIVE_SHA256,
    DATASET_FORMAT,
    EXAMPLES_PER_ARRIVAL,
    SEED,
    TASK_NAMES,
    TREE_LORA_REVISION,
)


TREE_LORA_ARCHIVE_URL = (
    "https://raw.githubusercontent.com/ZinYY/TreeLoRA/"
    f"{TREE_LORA_REVISION}/data/LLM-CL-Benchmark/LLM-CL-Benchmark_500.tar.xz"
)
_SPLIT_ALIASES = {
    "train": ("train",),
    "validation": ("validation", "valid", "val", "dev", "eval"),
    "test": ("test",),
}


@dataclass(frozen=True, slots=True)
class TraceExample:
    """One immutable source example from a named TRACE split."""

    example_id: str
    task: str
    split: str
    source_index: int
    prompt: str
    answer: str
    arrival: int | None = None

    def __post_init__(self) -> None:
        require_sha256(self.example_id, "TRACE example")
        if self.task not in TASK_NAMES or self.split not in _SPLIT_ALIASES:
            raise ValueError("TRACE example task or split is not canonical")
        if self.source_index < 0 or not self.prompt or not self.answer:
            raise ValueError("TRACE examples require nonempty text and a source index")
        if self.split == "train":
            if self.arrival is None or not 1 <= self.arrival <= ARRIVAL_COUNT:
                raise ValueError("training examples require a valid arrival")
        elif self.arrival is not None:
            raise ValueError("validation and test examples cannot have arrivals")

    def as_record(self) -> dict[str, object]:
        """Return the complete persisted example record."""
        return {
            "answer": self.answer,
            "arrival": self.arrival,
            "example_id": self.example_id,
            "prompt": self.prompt,
            "source_index": self.source_index,
            "split": self.split,
            "task": self.task,
        }


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """Permanent identity and split accounting for prepared TRACE examples."""

    archive_sha256: str
    source_files: tuple[tuple[str, str], ...]
    examples: tuple[TraceExample, ...]
    format: str = DATASET_FORMAT
    seed: int = SEED

    def __post_init__(self) -> None:
        require_sha256(self.archive_sha256, "TRACE source archive")
        if self.archive_sha256 != DATASET_ARCHIVE_SHA256:
            raise ValueError("TRACE source archive differs from the pinned contract")
        if self.format != DATASET_FORMAT or self.seed != SEED:
            raise ValueError("TRACE dataset manifest protocol differs")
        if len({example.example_id for example in self.examples}) != len(self.examples):
            raise ValueError("TRACE example identities are not unique")
        training = tuple(example for example in self.examples if example.split == "train")
        if len(training) != len(TASK_NAMES) * ARRIVALS_PER_TASK * EXAMPLES_PER_ARRIVAL:
            raise ValueError("TRACE manifest does not contain 4,000 training examples")
        arrival_counts = {
            arrival: sum(example.arrival == arrival for example in training)
            for arrival in range(1, ARRIVAL_COUNT + 1)
        }
        if set(arrival_counts.values()) != {EXAMPLES_PER_ARRIVAL}:
            raise ValueError("TRACE arrivals must contain exactly 100 examples")
        for path, digest in self.source_files:
            if not path:
                raise ValueError("TRACE source paths must be nonempty")
            require_sha256(digest, "TRACE source file")

    @property
    def manifest_sha256(self) -> str:
        """Return the content identity excluding prompt and answer payloads."""
        return record_sha256(self.as_record())

    def as_record(self) -> dict[str, object]:
        """Return the permanent dataset assignment manifest."""
        return {
            "archive_sha256": self.archive_sha256,
            "arrivals": [
                {
                    "arrival": arrival,
                    "example_ids": [
                        example.example_id
                        for example in self.examples
                        if example.arrival == arrival
                    ],
                    "task": TASK_NAMES[(arrival - 1) // ARRIVALS_PER_TASK],
                }
                for arrival in range(1, ARRIVAL_COUNT + 1)
            ],
            "examples": [
                {
                    "arrival": example.arrival,
                    "example_id": example.example_id,
                    "source_index": example.source_index,
                    "split": example.split,
                    "task": example.task,
                }
                for example in self.examples
            ],
            "format": self.format,
            "seed": self.seed,
            "source_files": [list(item) for item in self.source_files],
            "tree_lora_revision": TREE_LORA_REVISION,
        }


def download_pinned_archive(destination: str | Path) -> Path:
    """Download the pinned TreeLoRA archive once and verify its digest."""
    target = Path(destination)
    if target.is_file():
        if file_sha256(target) != DATASET_ARCHIVE_SHA256:
            raise ValueError("existing TreeLoRA archive has the wrong SHA-256")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.download")
    try:
        with urlopen(TREE_LORA_ARCHIVE_URL, timeout=120) as response, temporary.open(
            "wb"
        ) as output:
            shutil.copyfileobj(response, output)
        if file_sha256(temporary) != DATASET_ARCHIVE_SHA256:
            raise ValueError("downloaded TreeLoRA archive has the wrong SHA-256")
        publish_immutable_bytes(target, temporary.read_bytes())
    finally:
        temporary.unlink(missing_ok=True)
    return target


def extract_pinned_archive(archive: str | Path, destination: str | Path) -> Path:
    """Safely extract the verified TreeLoRA archive into an empty cache directory."""
    source, target = Path(archive), Path(destination)
    if file_sha256(source) != DATASET_ARCHIVE_SHA256:
        raise ValueError("TreeLoRA archive has the wrong SHA-256")
    marker = target / ".trace-extracted.json"
    if marker.is_file():
        record = json.loads(marker.read_text(encoding="utf-8"))
        if record.get("archive_sha256") != DATASET_ARCHIVE_SHA256:
            raise ValueError("dataset extraction marker differs")
        return target
    if target.exists():
        raise ValueError("unmarked dataset extraction target already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{target.name}.extracting-",
        dir=target.parent,
    ) as temporary:
        building = Path(temporary)
        resolved_building = building.resolve()
        with tarfile.open(source, "r:xz") as bundle:
            members = tuple(bundle.getmembers())
            for member in members:
                resolved_member = (building / member.name).resolve()
                if (
                    resolved_building not in resolved_member.parents
                    and resolved_member != resolved_building
                ):
                    raise ValueError("TreeLoRA archive contains an unsafe path")
                if member.issym() or member.islnk() or member.isdev():
                    raise ValueError("TreeLoRA archive contains a link or device")
            bundle.extractall(building, members=members)
        publish_immutable_json(
            building / ".trace-extracted.json",
            {
                "archive_sha256": DATASET_ARCHIVE_SHA256,
                "format": "trace-tree-lora-extraction-v1",
            },
        )
        os.rename(building, target)
        fsync_directory(target.parent)
    return target


def prepare_dataset(
    source_root: str | Path,
    manifest_path: str | Path,
    examples_path: str | Path,
    archive_sha256: str = DATASET_ARCHIVE_SHA256,
) -> DatasetManifest:
    """Load, partition, validate, and immutably publish the TRACE dataset."""
    root = _find_dataset_root(Path(source_root))
    source_files = tuple(
        sorted(
            (
                str(path.relative_to(root)),
                file_sha256(path),
            )
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".json", ".jsonl"}
        )
    )
    raw_examples = tuple(
        raw
        for task in TASK_NAMES
        for split in _SPLIT_ALIASES
        for raw in _load_split(root, task, split)
    )
    examples = _assign_arrivals(raw_examples, archive_sha256)
    manifest = DatasetManifest(
        archive_sha256=archive_sha256,
        source_files=source_files,
        examples=examples,
    )
    publish_immutable_json(manifest_path, manifest.as_record())
    publish_immutable_bytes(
        examples_path,
        b"".join(canonical_json_bytes(example.as_record()) for example in examples),
    )
    return manifest


def load_examples(path: str | Path) -> tuple[TraceExample, ...]:
    """Load and validate the canonical prepared example JSONL cache."""
    rows = []
    for line in Path(path).read_bytes().splitlines(keepends=True):
        value = json.loads(line)
        if type(value) is not dict or line != canonical_json_bytes(value):
            raise ValueError("TRACE example cache is not canonical JSONL")
        rows.append(
            TraceExample(
                example_id=str(value["example_id"]),
                task=str(value["task"]),
                split=str(value["split"]),
                source_index=int(value["source_index"]),
                prompt=str(value["prompt"]),
                answer=str(value["answer"]),
                arrival=int(value["arrival"]) if value["arrival"] is not None else None,
            )
        )
    return tuple(rows)


def arrival_identities(examples: Sequence[TraceExample]) -> tuple[str, ...]:
    """Return the permanent ordered identities of all 40 prepared arrivals."""
    return tuple(
        record_sha256(
            {
                "arrival": arrival,
                "example_ids": sorted(
                    example.example_id for example in examples if example.arrival == arrival
                ),
                "format": "trace-arrival-v1",
            }
        )
        for arrival in range(1, ARRIVAL_COUNT + 1)
    )


def _find_dataset_root(root: Path) -> Path:
    candidates = tuple(
        path
        for path in (root, *root.rglob("LLM-CL-Benchmark_500"))
        if path.is_dir()
    )
    for candidate in candidates:
        if all(any((candidate / task).glob("*.json*")) for task in TASK_NAMES):
            return candidate
    raise FileNotFoundError("could not find all TRACE task directories")


def _load_split(root: Path, task: str, split: str) -> tuple[dict[str, object], ...]:
    task_root = root / task
    matches = tuple(
        sorted(
            path
            for alias in _SPLIT_ALIASES[split]
            for path in (task_root / f"{alias}.json", task_root / f"{alias}.jsonl")
            if path.is_file()
        )
    )
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {task}/{split} source, found {matches}")
    path = matches[0]
    if path.suffix == ".jsonl":
        values = tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
        if type(value) is dict and type(value.get("data")) is list:
            value = value["data"]
        if type(value) is not list:
            raise ValueError(f"TRACE split is not a JSON list: {path}")
        values = tuple(value)
    if not all(type(item) is dict for item in values):
        raise ValueError(f"TRACE split contains a non-object: {path}")
    expected = 500 if split == "train" else None
    if expected is not None and len(values) != expected:
        raise ValueError(f"TRACE task {task} requires exactly 500 training examples")
    return tuple(
        {
            "answer": _required_text(item, ("answer", "output", "response"), path),
            "prompt": _required_text(item, ("prompt", "input", "instruction"), path),
            "source_index": index,
            "split": split,
            "task": task,
        }
        for index, item in enumerate(values)
    )


def _required_text(
    item: Mapping[str, object],
    names: Sequence[str],
    source: Path,
) -> str:
    values = tuple(item.get(name) for name in names if type(item.get(name)) is str)
    if len(values) != 1 or not str(values[0]).strip():
        raise ValueError(f"TRACE row has no unique {names} text in {source}")
    return str(values[0])


def _assign_arrivals(
    raw_examples: Iterable[Mapping[str, object]],
    archive_sha256: str,
) -> tuple[TraceExample, ...]:
    require_sha256(archive_sha256, "TRACE source archive")
    staged = tuple(
        (
            raw,
            sha256(
                canonical_json_bytes(
                    {
                        "answer": raw["answer"],
                        "archive_sha256": archive_sha256,
                        "prompt": raw["prompt"],
                        "source_index": raw["source_index"],
                        "split": raw["split"],
                        "task": raw["task"],
                    }
                )
            ).hexdigest(),
        )
        for raw in raw_examples
    )
    arrivals: dict[str, int] = {}
    for task_index, task in enumerate(TASK_NAMES):
        task_train = tuple(
            sorted(
                (item for item in staged if item[0]["task"] == task and item[0]["split"] == "train"),
                key=lambda item: (record_sha256({"example_id": item[1], "seed": SEED}), item[1]),
            )
        )
        if len(task_train) != ARRIVALS_PER_TASK * EXAMPLES_PER_ARRIVAL:
            raise ValueError(f"TRACE task {task} does not have 500 training rows")
        arrivals.update(
            {
                identity: task_index * ARRIVALS_PER_TASK + offset // EXAMPLES_PER_ARRIVAL + 1
                for offset, (_, identity) in enumerate(task_train)
            }
        )
    return tuple(
        TraceExample(
            example_id=identity,
            task=str(raw["task"]),
            split=str(raw["split"]),
            source_index=int(raw["source_index"]),
            prompt=str(raw["prompt"]),
            answer=str(raw["answer"]),
            arrival=arrivals.get(identity),
        )
        for raw, identity in staged
    )


__all__ = [
    "DatasetManifest",
    "TREE_LORA_ARCHIVE_URL",
    "TraceExample",
    "arrival_identities",
    "download_pinned_archive",
    "extract_pinned_archive",
    "load_examples",
    "prepare_dataset",
]

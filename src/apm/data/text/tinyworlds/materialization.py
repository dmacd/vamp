"""Fixed, content-addressed materialization of rendered TinyWorlds bundles."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Literal, TypeAlias

from apm.data.text.tinyworlds.persistence import (
    load_tinyworlds_bundle,
    load_tinyworlds_manifest,
    tinyworlds_bundle_sha256,
)
from apm.data.text.tinyworlds.rendered_persistence import (
    RenderedTinyWorldsBundleError,
    load_rendered_tinyworlds_bundle,
    load_rendered_tinyworlds_manifest,
    write_rendered_tinyworlds_bundle,
)
from apm.data.text.tinyworlds.rendering import (
    TinyWorldsRenderPreset,
    render_tinyworlds_bundle,
)
from apm.lm.text import TextTokenizer


CANONICAL_RENDERED_WORLD_NAMES = ("calibration", "pilot")
_FORMAT = "apm.tinyworlds.rendered-materialization"
_SCHEMA_VERSION = 1
_SHA256_LENGTH = 64
MaterializationAction: TypeAlias = Literal["materialized", "verified"]


@dataclass(frozen=True, slots=True)
class RenderedWorldArtifact:
    """Content identities and record counts for one verified rendered world."""

    world_name: str
    symbolic_bundle_id: str
    symbolic_bundle_sha256: str
    rendered_bundle_id: str
    rendered_bundle_sha256: str
    story_count: int
    query_group_count: int

    def __post_init__(self) -> None:
        if self.world_name not in CANONICAL_RENDERED_WORLD_NAMES:
            raise ValueError(f"unknown canonical rendered world: {self.world_name!r}")
        if any(
            type(value) is not str or not value
            for value in (self.symbolic_bundle_id, self.rendered_bundle_id)
        ):
            raise ValueError("bundle IDs must be nonempty strings")
        _require_sha256(self.symbolic_bundle_sha256, "symbolic bundle")
        _require_sha256(self.rendered_bundle_sha256, "rendered bundle")
        if any(
            type(value) is not int or value <= 0
            for value in (self.story_count, self.query_group_count)
        ):
            raise ValueError("rendered record counts must be positive integers")

    def as_dict(self) -> dict[str, object]:
        """Return the canonical JSON object for this artifact."""
        return {
            "query_group_count": self.query_group_count,
            "rendered_bundle_id": self.rendered_bundle_id,
            "rendered_bundle_sha256": self.rendered_bundle_sha256,
            "story_count": self.story_count,
            "symbolic_bundle_id": self.symbolic_bundle_id,
            "symbolic_bundle_sha256": self.symbolic_bundle_sha256,
            "world_name": self.world_name,
        }


@dataclass(frozen=True, slots=True)
class RenderedWorldMaterialization:
    """One materialize-or-load outcome with a run-local action."""

    artifact: RenderedWorldArtifact
    action: MaterializationAction

    def __post_init__(self) -> None:
        if type(self.artifact) is not RenderedWorldArtifact:
            raise TypeError("artifact must be a RenderedWorldArtifact")
        if self.action not in ("materialized", "verified"):
            raise ValueError(f"unknown materialization action: {self.action!r}")


@dataclass(frozen=True, slots=True)
class TinyWorldsRenderedMaterializationResult:
    """Canonical final identity of both rendered worlds and their tokenizer."""

    tokenizer_file_sha256: str
    artifacts: tuple[RenderedWorldArtifact, ...]

    def __post_init__(self) -> None:
        _require_sha256(self.tokenizer_file_sha256, "tokenizer file")
        if type(self.artifacts) is not tuple or any(
            type(item) is not RenderedWorldArtifact for item in self.artifacts
        ):
            raise TypeError("artifacts must be immutable rendered-world records")
        if (
            tuple(item.world_name for item in self.artifacts)
            != CANONICAL_RENDERED_WORLD_NAMES
        ):
            raise ValueError("materialization result requires calibration then pilot")

    def as_dict(self) -> dict[str, object]:
        """Return the canonical result record independent of run-local actions."""
        return {
            "artifacts": [item.as_dict() for item in self.artifacts],
            "format": _FORMAT,
            "schema_version": _SCHEMA_VERSION,
            "tokenizer_file_sha256": self.tokenizer_file_sha256,
        }

    @property
    def canonical_json(self) -> str:
        """Return one compact, sorted, deterministic JSON result."""
        return _canonical_json(self.as_dict()).decode("utf-8").rstrip("\n")

    @property
    def result_sha256(self) -> str:
        """Return the SHA-256 of the newline-terminated canonical result."""
        return sha256(_canonical_json(self.as_dict())).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Stream one file and return its lowercase SHA-256 identity."""
    digest = sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def materialize_or_verify_rendered_world(
    world_name: str,
    symbolic_directory: str | Path,
    rendered_directory: str | Path,
    tokenizer: TextTokenizer,
    preset: TinyWorldsRenderPreset,
) -> RenderedWorldMaterialization:
    """Render one missing immutable world, or strictly load and verify it."""
    if world_name not in CANONICAL_RENDERED_WORLD_NAMES:
        raise ValueError(f"unknown canonical rendered world: {world_name!r}")
    if not isinstance(tokenizer, TextTokenizer):
        raise TypeError("tokenizer must satisfy TextTokenizer")
    if type(preset) is not TinyWorldsRenderPreset:
        raise TypeError("preset must be a TinyWorldsRenderPreset")
    symbolic_path = Path(symbolic_directory)
    target = Path(rendered_directory)
    if symbolic_path.is_symlink() or not symbolic_path.is_dir():
        raise FileNotFoundError(f"canonical symbolic bundle is absent: {symbolic_path}")
    if target.is_symlink() or (target.exists() and not target.is_dir()):
        raise RenderedTinyWorldsBundleError(
            f"rendered bundle target must be a regular directory: {target}"
        )
    symbolic_manifest = load_tinyworlds_manifest(symbolic_path)
    symbolic = load_tinyworlds_bundle(symbolic_path)
    if (
        symbolic_manifest.bundle_sha256 != tinyworlds_bundle_sha256(symbolic)
        or symbolic.bundle_id != f"tinyworlds-v1:{world_name}"
        or symbolic.world.world_id != world_name
    ):
        raise RenderedTinyWorldsBundleError(
            "canonical symbolic bundle identity does not match its world slot"
        )

    if not target.exists():
        rendered = render_tinyworlds_bundle(symbolic, tokenizer, preset)
        try:
            rendered_manifest = write_rendered_tinyworlds_bundle(
                rendered,
                symbolic,
                tokenizer,
                target,
            )
            if load_rendered_tinyworlds_manifest(target) != rendered_manifest:
                raise RenderedTinyWorldsBundleError(
                    "new rendered manifest changed after atomic publication"
                )
            return RenderedWorldMaterialization(
                artifact=RenderedWorldArtifact(
                    world_name=world_name,
                    symbolic_bundle_id=symbolic.bundle_id,
                    symbolic_bundle_sha256=symbolic_manifest.bundle_sha256,
                    rendered_bundle_id=rendered.bundle_id,
                    rendered_bundle_sha256=rendered_manifest.bundle_sha256,
                    story_count=len(rendered.stories),
                    query_group_count=len(rendered.query_groups),
                ),
                action="materialized",
            )
        except FileExistsError:
            pass

    loaded = load_rendered_tinyworlds_bundle(target, symbolic, tokenizer)
    rendered_manifest = load_rendered_tinyworlds_manifest(target)
    if loaded.preset != preset:
        raise RenderedTinyWorldsBundleError(
            "existing rendered bundle uses a different fixed render preset"
        )
    return RenderedWorldMaterialization(
        artifact=RenderedWorldArtifact(
            world_name=world_name,
            symbolic_bundle_id=symbolic.bundle_id,
            symbolic_bundle_sha256=symbolic_manifest.bundle_sha256,
            rendered_bundle_id=loaded.bundle_id,
            rendered_bundle_sha256=rendered_manifest.bundle_sha256,
            story_count=len(loaded.stories),
            query_group_count=len(loaded.query_groups),
        ),
        action="verified",
    )


def build_rendered_materialization_result(
    tokenizer_file_sha256: str,
    outcomes: tuple[RenderedWorldMaterialization, ...],
) -> TinyWorldsRenderedMaterializationResult:
    """Project run-local outcomes into one stable content-only result."""
    return TinyWorldsRenderedMaterializationResult(
        tokenizer_file_sha256=tokenizer_file_sha256,
        artifacts=tuple(item.artifact for item in outcomes),
    )


def write_rendered_materialization_result(
    result: TinyWorldsRenderedMaterializationResult,
    directory: str | Path,
) -> Path:
    """Atomically write a new canonical materialization result file."""
    if type(result) is not TinyWorldsRenderedMaterializationResult:
        raise TypeError("result must be a TinyWorldsRenderedMaterializationResult")
    target = Path(directory) / "materialization_result.json"
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"materialization result already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".materialization-result-",
        suffix=".json",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(_canonical_json(result.as_dict()))
            output.flush()
            os.fsync(output.fileno())
        os.rename(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target


def _canonical_json(record: object) -> bytes:
    return (
        json.dumps(
            record,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _require_sha256(value: str, label: str) -> None:
    if (
        type(value) is not str
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} SHA-256 must be 64 lowercase hexadecimal digits")


__all__ = [
    "CANONICAL_RENDERED_WORLD_NAMES",
    "RenderedWorldArtifact",
    "RenderedWorldMaterialization",
    "TinyWorldsRenderedMaterializationResult",
    "build_rendered_materialization_result",
    "file_sha256",
    "materialize_or_verify_rendered_world",
    "write_rendered_materialization_result",
]

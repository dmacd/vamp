"""Canonical immutable persistence for TinyWorlds-v2 Phase 1 artifacts.

The builder writes into a caller-created staging directory.  Finalization
records every nested file in a self-authenticating manifest, and promotion is
an atomic directory rename that never intentionally replaces an existing
destination.  JSON and JSONL files are canonical derived artifacts; all other
suffixes are preserved and authenticated as opaque bytes, including cached
provider response bodies.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import errno
from hashlib import sha256
import os
from pathlib import Path, PurePosixPath
import re
from typing import Iterable

from apm.data.text.tinyworlds_v2.json_contracts import (
    CanonicalJsonError,
    JsonObject,
    JsonValue,
    canonical_json_bytes,
    canonical_json_line_bytes,
    canonical_json_loads,
    require_exact_fields,
    require_json_object,
)


PHASE1_ARTIFACT_FORMAT = "apm.tinyworlds-v2.phase1-reference"
PHASE1_ARTIFACT_SCHEMA_VERSION = 1
DEFAULT_PHASE1_ARTIFACT_VERSION = "tinyworlds-v2-phase1"

_MANIFEST_FILENAME = "manifest.json"
_CANONICAL_JSON = "canonical-json"
_CANONICAL_JSONL = "canonical-jsonl"
_BINARY = "binary"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class Phase1ArtifactError(ValueError):
    """A Phase 1 tree is malformed or fails an integrity check."""


@dataclass(frozen=True, slots=True)
class Phase1ArtifactFile:
    """Integrity and serialization metadata for one manifested file."""

    path: str
    content_format: str
    sha256: str
    size_bytes: int
    record_count: int

    def __post_init__(self) -> None:
        _validate_relative_path(self.path)
        expected_format = _content_format_for_path(self.path)
        if self.content_format != expected_format:
            raise Phase1ArtifactError(
                f"artifact {self.path!r} must use content format {expected_format!r}"
            )
        if type(self.sha256) is not str or _SHA256.fullmatch(self.sha256) is None:
            raise Phase1ArtifactError("artifact sha256 must be a lowercase SHA-256")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise Phase1ArtifactError("artifact size_bytes must be nonnegative")
        if type(self.record_count) is not int or self.record_count < 0:
            raise Phase1ArtifactError("artifact record_count must be nonnegative")
        if self.content_format == _CANONICAL_JSON and self.record_count != 1:
            raise Phase1ArtifactError("canonical JSON artifacts contain one record")
        if self.content_format == _BINARY and self.record_count != 0:
            raise Phase1ArtifactError("binary artifacts must have record_count zero")

    def as_record(self) -> JsonObject:
        """Return the canonical manifest projection of this descriptor."""
        return {
            "content_format": self.content_format,
            "path": self.path,
            "record_count": self.record_count,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class Phase1ArtifactManifest:
    """Self-authenticating manifest for one complete immutable Phase 1 tree."""

    format: str
    schema_version: int
    version: str
    artifacts: tuple[Phase1ArtifactFile, ...]
    manifest_sha256: str

    def __post_init__(self) -> None:
        if self.format != PHASE1_ARTIFACT_FORMAT:
            raise Phase1ArtifactError(f"unsupported Phase 1 format: {self.format!r}")
        if (
            type(self.schema_version) is not int
            or self.schema_version != PHASE1_ARTIFACT_SCHEMA_VERSION
        ):
            raise Phase1ArtifactError(
                f"unsupported Phase 1 schema version: {self.schema_version!r}"
            )
        if type(self.version) is not str or _VERSION.fullmatch(self.version) is None:
            raise Phase1ArtifactError("artifact version is not a canonical identifier")
        if type(self.artifacts) is not tuple or any(
            type(artifact) is not Phase1ArtifactFile for artifact in self.artifacts
        ):
            raise Phase1ArtifactError("manifest artifacts must be file descriptors")
        paths = tuple(artifact.path for artifact in self.artifacts)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise Phase1ArtifactError(
                "manifest artifact paths must be unique and sorted"
            )
        if type(self.manifest_sha256) is not str or _SHA256.fullmatch(
            self.manifest_sha256
        ) is None:
            raise Phase1ArtifactError(
                "manifest_sha256 must be a lowercase SHA-256"
            )
        expected_digest = sha256(
            canonical_json_bytes(_manifest_core_record(self))
        ).hexdigest()
        if self.manifest_sha256 != expected_digest:
            raise Phase1ArtifactError("manifest self-digest mismatch")

    def as_record(self) -> JsonObject:
        """Return the complete canonical JSON manifest record."""
        return {
            **_manifest_core_record(self),
            "manifest_sha256": self.manifest_sha256,
        }


@dataclass(frozen=True, slots=True)
class Phase1ArtifactBuilder:
    """Write-once builder rooted at an already-created staging directory."""

    root: Path
    version: str = DEFAULT_PHASE1_ARTIFACT_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.root, Path):
            raise TypeError("builder root must be a Path")
        _require_regular_directory(self.root, "Phase 1 staging directory")
        if type(self.version) is not str or _VERSION.fullmatch(self.version) is None:
            raise Phase1ArtifactError("artifact version is not a canonical identifier")
        if (self.root / _MANIFEST_FILENAME).exists() or (
            self.root / _MANIFEST_FILENAME
        ).is_symlink():
            raise Phase1ArtifactError("staging tree is already finalized")

    def write_bytes(self, relative_path: str, payload: bytes) -> Path:
        """Write one new artifact exactly once without replacing existing data."""
        if type(payload) is not bytes:
            raise TypeError("artifact payload must be bytes")
        path = self._new_artifact_path(relative_path)
        with path.open("xb") as stream:
            stream.write(payload)
        return path

    def write_json(self, relative_path: str, value: JsonValue) -> Path:
        """Write one new canonical JSON artifact without a trailing newline."""
        if not relative_path.endswith(".json"):
            raise Phase1ArtifactError("canonical JSON artifacts must end in .json")
        return self.write_bytes(relative_path, canonical_json_bytes(value))

    def append_jsonl(self, relative_path: str, value: JsonValue) -> Path:
        """Append one canonical record to a new or existing JSONL artifact."""
        if not relative_path.endswith(".jsonl"):
            raise Phase1ArtifactError("canonical JSONL artifacts must end in .jsonl")
        self._require_unfinalized()
        normalized = _validate_relative_path(relative_path)
        path = self.root.joinpath(*PurePosixPath(normalized).parts)
        _prepare_parent_directories(self.root, PurePosixPath(normalized).parent.parts)
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise Phase1ArtifactError(
                f"artifact path is not a regular file: {normalized}"
            )
        if path.exists():
            _canonical_jsonl_record_count(path.read_bytes(), normalized)
            mode = "ab"
        else:
            mode = "xb"
        with path.open(mode) as stream:
            stream.write(canonical_json_line_bytes(value))
        return path

    def finalize(self) -> Phase1ArtifactManifest:
        """Validate every staged byte and write the canonical manifest once."""
        self._require_unfinalized()
        files, directories = _scan_tree(self.root)
        artifact_paths = tuple(
            path for path in files if path != _MANIFEST_FILENAME
        )
        if not artifact_paths:
            raise Phase1ArtifactError("a Phase 1 tree must contain an artifact")
        artifacts = tuple(
            _describe_artifact(self.root, relative_path)
            for relative_path in artifact_paths
        )
        if directories != _expected_directories(artifacts):
            raise Phase1ArtifactError(
                "staging tree contains empty or unlisted directories"
            )
        core: JsonObject = {
            "artifacts": [artifact.as_record() for artifact in artifacts],
            "format": PHASE1_ARTIFACT_FORMAT,
            "schema_version": PHASE1_ARTIFACT_SCHEMA_VERSION,
            "version": self.version,
        }
        manifest = Phase1ArtifactManifest(
            format=PHASE1_ARTIFACT_FORMAT,
            schema_version=PHASE1_ARTIFACT_SCHEMA_VERSION,
            version=self.version,
            artifacts=artifacts,
            manifest_sha256=sha256(canonical_json_bytes(core)).hexdigest(),
        )
        with (self.root / _MANIFEST_FILENAME).open("xb") as stream:
            stream.write(canonical_json_bytes(manifest.as_record()))
        # ``_scan_tree`` and ``_describe_artifact`` already authenticated every
        # staged byte while constructing this self-validating manifest. Callers
        # that cross a trust boundary still use ``load_phase1_artifact_tree``;
        # rereading the complete tree here would only duplicate that work.
        return manifest

    def promote(
        self,
        destination: str | Path,
        *,
        expected_manifest_sha256: str | None = None,
    ) -> Path:
        """Finalize and atomically rename this staging tree to a fresh destination."""
        manifest = self.root / _MANIFEST_FILENAME
        if not (manifest.exists() or manifest.is_symlink()):
            self.finalize()
        return promote_phase1_artifact_tree(
            self.root,
            destination,
            expected_manifest_sha256=expected_manifest_sha256,
        )

    def _require_unfinalized(self) -> None:
        manifest = self.root / _MANIFEST_FILENAME
        if manifest.exists() or manifest.is_symlink():
            raise Phase1ArtifactError("staging tree is already finalized")

    def _new_artifact_path(self, relative_path: str) -> Path:
        self._require_unfinalized()
        normalized = _validate_relative_path(relative_path)
        pure_path = PurePosixPath(normalized)
        _prepare_parent_directories(self.root, pure_path.parent.parts)
        path = self.root.joinpath(*pure_path.parts)
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"artifact already exists: {normalized}")
        return path


def canonical_jsonl_bytes(records: Iterable[JsonValue]) -> bytes:
    """Encode records as canonical newline-terminated JSONL bytes."""
    return b"".join(canonical_json_line_bytes(record) for record in records)


def load_phase1_artifact_tree(
    directory: str | Path,
) -> Phase1ArtifactManifest:
    """Strictly load and validate a complete Phase 1 artifact tree."""
    root = Path(directory)
    _require_regular_directory(root, "Phase 1 artifact directory")
    manifest_path = root / _MANIFEST_FILENAME
    _require_regular_file(manifest_path, "Phase 1 manifest")
    try:
        manifest_value = canonical_json_loads(
            manifest_path.read_bytes(), label="Phase 1 manifest"
        )
        manifest = _decode_manifest(
            require_json_object(manifest_value, label="Phase 1 manifest")
        )
    except CanonicalJsonError as error:
        raise Phase1ArtifactError(str(error)) from error

    actual_files, actual_directories = _scan_tree(root)
    manifested_paths = tuple(artifact.path for artifact in manifest.artifacts)
    expected_files = tuple(sorted((_MANIFEST_FILENAME, *manifested_paths)))
    if actual_files != expected_files:
        raise Phase1ArtifactError(
            "artifact tree contains missing or unlisted files"
        )
    expected_directories = _expected_directories(manifest.artifacts)
    if actual_directories != expected_directories:
        raise Phase1ArtifactError(
            "artifact tree contains missing or unlisted directories"
        )

    for artifact in manifest.artifacts:
        path = root.joinpath(*PurePosixPath(artifact.path).parts)
        _require_regular_file(path, f"artifact {artifact.path}")
        payload = path.read_bytes()
        if len(payload) != artifact.size_bytes:
            raise Phase1ArtifactError(f"artifact size mismatch: {artifact.path}")
        if sha256(payload).hexdigest() != artifact.sha256:
            raise Phase1ArtifactError(f"artifact digest mismatch: {artifact.path}")
        actual_count = _validate_payload(
            artifact.path, artifact.content_format, payload
        )
        if actual_count != artifact.record_count:
            raise Phase1ArtifactError(
                f"artifact record count mismatch: {artifact.path}"
            )
    return manifest


def promote_phase1_artifact_tree(
    staging_directory: str | Path,
    destination: str | Path,
    *,
    expected_manifest_sha256: str | None = None,
) -> Path:
    """Validate and atomically rename a staging tree without overwriting a target."""
    staging = Path(staging_directory)
    target = Path(destination)
    staging_manifest = load_phase1_artifact_tree(staging)
    if (
        expected_manifest_sha256 is not None
        and staging_manifest.manifest_sha256 != expected_manifest_sha256
    ):
        raise Phase1ArtifactError(
            "staging artifact identity differs from its semantic validation"
        )
    if staging.resolve() == target.resolve():
        raise Phase1ArtifactError("staging and destination must differ")
    if target.resolve().is_relative_to(staging.resolve()):
        raise Phase1ArtifactError("destination cannot be nested inside staging")
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Phase 1 artifact target already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    _require_regular_directory(target.parent, "artifact destination parent")
    _rename_directory_noreplace(staging, target)
    promoted_manifest = load_phase1_artifact_tree(target)
    if promoted_manifest.manifest_sha256 != staging_manifest.manifest_sha256:
        raise Phase1ArtifactError(
            "promoted artifact identity differs from its staging identity"
        )
    return target


def _rename_directory_noreplace(source: Path, target: Path) -> None:
    """Atomically rename with Linux ``RENAME_NOREPLACE`` semantics."""
    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as error:
        raise Phase1ArtifactError(
            "atomic no-replace promotion requires Linux renameat2"
        ) from error
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace = 1
    result = renameat2(
        at_fdcwd,
        os.fsencode(source),
        at_fdcwd,
        os.fsencode(target),
        rename_noreplace,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(
            error_number,
            f"Phase 1 artifact target already exists: {target}",
            target,
        )
    if error_number in (errno.ENOSYS, errno.EINVAL):
        raise Phase1ArtifactError(
            "atomic no-replace promotion is unavailable on this filesystem"
        )
    raise OSError(error_number, os.strerror(error_number), target)


def _manifest_core_record(manifest: Phase1ArtifactManifest) -> JsonObject:
    return {
        "artifacts": [artifact.as_record() for artifact in manifest.artifacts],
        "format": manifest.format,
        "schema_version": manifest.schema_version,
        "version": manifest.version,
    }


def _decode_manifest(record: JsonObject) -> Phase1ArtifactManifest:
    require_exact_fields(
        record,
        (
            "artifacts",
            "format",
            "manifest_sha256",
            "schema_version",
            "version",
        ),
        label="Phase 1 manifest",
    )
    artifact_values = record["artifacts"]
    if type(artifact_values) is not list:
        raise Phase1ArtifactError("manifest artifacts must be a list")
    artifacts = tuple(
        _decode_artifact(
            require_json_object(value, label=f"artifact descriptor {index}")
        )
        for index, value in enumerate(artifact_values)
    )
    return Phase1ArtifactManifest(
        format=_require_string(record["format"], "manifest format"),
        schema_version=_require_integer(
            record["schema_version"], "manifest schema_version"
        ),
        version=_require_string(record["version"], "manifest version"),
        artifacts=artifacts,
        manifest_sha256=_require_string(
            record["manifest_sha256"], "manifest_sha256"
        ),
    )


def _decode_artifact(record: JsonObject) -> Phase1ArtifactFile:
    require_exact_fields(
        record,
        ("content_format", "path", "record_count", "sha256", "size_bytes"),
        label="artifact descriptor",
    )
    return Phase1ArtifactFile(
        path=_require_string(record["path"], "artifact path"),
        content_format=_require_string(
            record["content_format"], "artifact content_format"
        ),
        sha256=_require_string(record["sha256"], "artifact sha256"),
        size_bytes=_require_integer(record["size_bytes"], "artifact size_bytes"),
        record_count=_require_integer(
            record["record_count"], "artifact record_count"
        ),
    )


def _describe_artifact(root: Path, relative_path: str) -> Phase1ArtifactFile:
    path = root.joinpath(*PurePosixPath(relative_path).parts)
    _require_regular_file(path, f"artifact {relative_path}")
    payload = path.read_bytes()
    content_format = _content_format_for_path(relative_path)
    return Phase1ArtifactFile(
        path=relative_path,
        content_format=content_format,
        sha256=sha256(payload).hexdigest(),
        size_bytes=len(payload),
        record_count=_validate_payload(relative_path, content_format, payload),
    )


def _validate_payload(path: str, content_format: str, payload: bytes) -> int:
    try:
        if content_format == _CANONICAL_JSON:
            canonical_json_loads(payload, label=path)
            return 1
        if content_format == _CANONICAL_JSONL:
            return _canonical_jsonl_record_count(payload, path)
        if content_format == _BINARY:
            return 0
    except CanonicalJsonError as error:
        raise Phase1ArtifactError(str(error)) from error
    raise Phase1ArtifactError(f"unsupported content format: {content_format!r}")


def _canonical_jsonl_record_count(payload: bytes, label: str) -> int:
    if not payload:
        return 0
    lines = payload.splitlines(keepends=True)
    if any(not line.endswith(b"\n") for line in lines):
        raise Phase1ArtifactError(f"non-canonical JSONL framing: {label}")
    try:
        for index, line in enumerate(lines, start=1):
            canonical_json_loads(line[:-1], label=f"{label} line {index}")
    except CanonicalJsonError as error:
        raise Phase1ArtifactError(str(error)) from error
    return len(lines)


def _content_format_for_path(path: str) -> str:
    if path.endswith(".jsonl"):
        return _CANONICAL_JSONL
    if path.endswith(".json"):
        return _CANONICAL_JSON
    return _BINARY


def _validate_relative_path(path: str) -> str:
    if type(path) is not str or not path or "\\" in path or "\x00" in path:
        raise Phase1ArtifactError("artifact path must be canonical relative POSIX text")
    pure_path = PurePosixPath(path)
    if (
        pure_path.is_absolute()
        or not pure_path.parts
        or path != pure_path.as_posix()
        or any(part in ("", ".", "..") for part in pure_path.parts)
        or path == _MANIFEST_FILENAME
    ):
        raise Phase1ArtifactError(f"unsafe or reserved artifact path: {path!r}")
    return path


def _prepare_parent_directories(root: Path, parts: tuple[str, ...]) -> None:
    current = root
    for part in parts:
        current /= part
        if current.is_symlink() or (current.exists() and not current.is_dir()):
            raise Phase1ArtifactError(
                f"artifact parent is not a regular directory: {current}"
            )
        current.mkdir(exist_ok=True)


def _scan_tree(root: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    files: list[str] = []
    directories: list[str] = []

    def visit(directory: Path, prefix: PurePosixPath) -> None:
        for entry in sorted(os.scandir(directory), key=lambda item: item.name):
            relative = (prefix / entry.name).as_posix()
            if entry.is_symlink():
                raise Phase1ArtifactError(f"artifact tree contains symlink: {relative}")
            if entry.is_file(follow_symlinks=False):
                if relative != _MANIFEST_FILENAME:
                    _validate_relative_path(relative)
                files.append(relative)
            elif entry.is_dir(follow_symlinks=False):
                _validate_relative_path(relative)
                directories.append(relative)
                visit(Path(entry.path), prefix / entry.name)
            else:
                raise Phase1ArtifactError(
                    f"artifact tree contains a non-regular entry: {relative}"
                )

    visit(root, PurePosixPath())
    return tuple(sorted(files)), tuple(sorted(directories))


def _expected_directories(
    artifacts: tuple[Phase1ArtifactFile, ...],
) -> tuple[str, ...]:
    directories = {
        PurePosixPath(*PurePosixPath(artifact.path).parts[:depth]).as_posix()
        for artifact in artifacts
        for depth in range(1, len(PurePosixPath(artifact.path).parts))
    }
    return tuple(sorted(directories))


def _require_regular_directory(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise Phase1ArtifactError(f"{label} must be a regular directory: {path}")


def _require_regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise Phase1ArtifactError(f"{label} must be a regular file: {path}")


def _require_string(value: JsonValue, label: str) -> str:
    if type(value) is not str:
        raise Phase1ArtifactError(f"{label} must be a string")
    return value


def _require_integer(value: JsonValue, label: str) -> int:
    if type(value) is not int:
        raise Phase1ArtifactError(f"{label} must be an integer")
    return value


__all__ = [
    "DEFAULT_PHASE1_ARTIFACT_VERSION",
    "PHASE1_ARTIFACT_FORMAT",
    "PHASE1_ARTIFACT_SCHEMA_VERSION",
    "Phase1ArtifactBuilder",
    "Phase1ArtifactError",
    "Phase1ArtifactFile",
    "Phase1ArtifactManifest",
    "canonical_json_bytes",
    "canonical_jsonl_bytes",
    "load_phase1_artifact_tree",
    "promote_phase1_artifact_tree",
]

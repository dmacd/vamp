"""Dataset, model, software, and environment manifest construction."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path
from collections.abc import Mapping, Sequence
import platform
import subprocess
import sys

from apm.continual.artifacts import file_sha256, record_sha256


def sealed_manifest(schema_version: str, values: Mapping[str, object]) -> dict[str, object]:
    """Add a content hash to a canonical manifest core."""
    core = {**dict(values), "schema_version": schema_version}
    return {**core, "content_hash": record_sha256(core)}


def require_sealed_manifest(record: Mapping[str, object], schema_version: str) -> str:
    """Validate a sealed manifest and return its content identity."""
    if record.get("schema_version") != schema_version:
        raise ValueError("manifest schema differs from the required protocol")
    supplied = record.get("content_hash")
    core = {key: value for key, value in record.items() if key != "content_hash"}
    if supplied != record_sha256(core):
        raise ValueError("manifest content hash changed")
    return str(supplied)


def installed_environment_manifest(packages: Sequence[str]) -> dict[str, object]:
    """Bind material Python, package, PyTorch, and CUDA runtime versions."""
    versions = []
    for package in sorted(set(packages)):
        try:
            version = metadata.version(package)
        except metadata.PackageNotFoundError:
            version = "MISSING"
        versions.append({"name": package, "version": version})
    torch_record: dict[str, object]
    try:
        import torch

        torch_record = {
            "bf16_supported": bool(
                torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            ),
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "torch": torch.__version__,
        }
    except ImportError:
        torch_record = {"torch": "MISSING"}
    return sealed_manifest(
        "imagenetr50-environment-v1",
        {
            "executable": sys.executable,
            "packages": versions,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch_runtime": torch_record,
        },
    )


def model_file_manifest(
    path: str | Path,
    repository: str,
    revision: str,
    model_name: str,
) -> dict[str, object]:
    """Create the immutable local model-checkpoint manifest."""
    source = Path(path).resolve()
    return sealed_manifest(
        "imagenetr50-model-v1",
        {
            "filename": source.name,
            "model_name": model_name,
            "repository": repository,
            "revision": revision,
            "sha256": file_sha256(source),
            "size_bytes": source.stat().st_size,
        },
    )


def git_commit_or_unknown(project_root: str | Path) -> str:
    """Return the enclosing Git revision without making it run identity authority."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(project_root),
        check=False,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and len(value) == 40 else "UNKNOWN"


__all__ = [
    "git_commit_or_unknown",
    "installed_environment_manifest",
    "model_file_manifest",
    "require_sealed_manifest",
    "sealed_manifest",
]

"""Unmodified pinned official E2-LoRA common-split reproduction wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import ast
import os
import re
import subprocess
import sys
import time
import platform

from apm.continual.artifacts import (
    atomic_write,
    file_sha256,
    publish_immutable_json,
)
from apm.continual.vision.imagenetr.constants import (
    E2LORA_CONFIG,
    E2LORA_REPOSITORY,
    E2LORA_REVISION,
    PUBLISHED_E2LORA_INCREMENTAL_ACCURACY,
    PUBLISHED_E2LORA_LAST_ACCURACY,
    TIMM_MODEL_REPOSITORY,
    TIMM_MODEL_REVISION,
    TIMM_MODEL_SHA256,
)
from apm.continual.vision.imagenetr.manifests import installed_environment_manifest


@dataclass(frozen=True, slots=True)
class E2LoRAResult:
    """Local unmodified reproduction outcome or explicit failure record."""

    succeeded: bool
    final_accuracy: float | None
    incremental_average_accuracy: float | None
    return_code: int
    wall_seconds: float
    log_sha256: str
    failure: str | None
    config_sha256: str | None
    environment_manifest_hash: str
    dependency_versions: tuple[tuple[str, str], ...]
    device_name: str

    def as_record(self) -> dict[str, object]:
        """Return local and separately labeled published external context."""
        return {
            "config": E2LORA_CONFIG,
            "config_sha256": self.config_sha256,
            "dependency_versions": [
                {"name": name, "version": version}
                for name, version in self.dependency_versions
            ],
            "device_name": self.device_name,
            "environment_manifest_hash": self.environment_manifest_hash,
            "failure": self.failure,
            "local_final_accuracy": self.final_accuracy,
            "local_incremental_average_accuracy": self.incremental_average_accuracy,
            "log_sha256": self.log_sha256,
            "published_final_accuracy": PUBLISHED_E2LORA_LAST_ACCURACY,
            "published_incremental_average_accuracy": PUBLISHED_E2LORA_INCREMENTAL_ACCURACY,
            "repository": E2LORA_REPOSITORY,
            "return_code": self.return_code,
            "revision": E2LORA_REVISION,
            "schema_version": "imagenetr50-e2lora-reproduction-v1",
            "succeeded": self.succeeded,
            "wall_seconds": self.wall_seconds,
        }


def _prepare_checkout(root: Path) -> Path:
    if not (root / ".git").is_dir():
        root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--no-checkout", E2LORA_REPOSITORY, str(root)],
            check=True,
        )
    subprocess.run(
        ["git", "checkout", "--detach", E2LORA_REVISION], cwd=root, check=True
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    if head != E2LORA_REVISION:
        raise ValueError("E2-LoRA checkout does not match the pinned revision")
    return root


def _parse_curve(log: str) -> tuple[float, float]:
    matches = re.findall(r"CNN top1 curve:\s*(\[[^\n]+\])", log)
    if not matches:
        raise ValueError("official E2-LoRA log contains no top-one curve")
    curve = tuple(float(value) for value in ast.literal_eval(matches[-1]))
    if len(curve) != 50:
        raise ValueError("official E2-LoRA did not complete all 50 tasks")
    return curve[-1], sum(curve) / len(curve)


def _pin_official_model_cache(cache_root: Path) -> None:
    """Make the moving official alias resolve offline to the required checkpoint bytes."""
    repository_cache = cache_root / f"models--{TIMM_MODEL_REPOSITORY.replace('/', '--')}"
    checkpoint = (
        repository_cache / "snapshots" / TIMM_MODEL_REVISION / "model.safetensors"
    )
    if not checkpoint.is_file() or file_sha256(checkpoint) != TIMM_MODEL_SHA256:
        raise ValueError("official reproduction cache lacks the pinned timm checkpoint")
    atomic_write(repository_cache / "refs" / "main", TIMM_MODEL_REVISION.encode("ascii"))


def run_official_e2lora(
    checkout_root: str | Path,
    prepared_data_parent: str | Path,
    cache_root: str | Path,
    output_directory: str | Path,
) -> E2LoRAResult:
    """Run the pinned repository unchanged and persist success or reproduction failure."""
    output = Path(output_directory)
    result_path, log_path = output / "result.json", output / "official.log"
    if result_path.is_file() and log_path.is_file():
        import json

        record = json.loads(result_path.read_text(encoding="utf-8"))
        return E2LoRAResult(
            bool(record["succeeded"]),
            record["local_final_accuracy"],
            record["local_incremental_average_accuracy"],
            int(record["return_code"]),
            float(record["wall_seconds"]),
            str(record["log_sha256"]),
            record["failure"],
            record["config_sha256"],
            str(record["environment_manifest_hash"]),
            tuple(
                (str(row["name"]), str(row["version"]))
                for row in record["dependency_versions"]
            ),
            str(record["device_name"]),
        )
    output.mkdir(parents=True, exist_ok=True)
    dependency_environment = installed_environment_manifest(
        ("numpy", "Pillow", "scipy", "timm", "torch", "torchvision", "tqdm")
    )
    dependency_versions = tuple(
        (str(row["name"]), str(row["version"]))
        for row in dependency_environment["packages"]
    )
    try:
        import torch

        device_name = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA_UNAVAILABLE"
        )
    except ImportError:
        device_name = "TORCH_MISSING"
    config_sha: str | None = None
    environment = {
        **os.environ,
        "CIL_DATA_ROOT": str(Path(prepared_data_parent).resolve()),
        "HF_HOME": str(Path(cache_root).resolve()),
        "HF_HUB_CACHE": str(Path(cache_root).resolve()),
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_OFFLINE": "1",
        "PYTHONHASHSEED": "1993",
    }
    command = [
        sys.executable,
        "main.py",
        "--config",
        "./exps/e2lora_inr_lora_50.json",
    ]
    started = time.monotonic()
    failure = None
    final = average = None
    try:
        checkout = _prepare_checkout(Path(checkout_root))
        config_path = checkout / E2LORA_CONFIG
        config_sha = file_sha256(config_path)
        _pin_official_model_cache(Path(cache_root))
        completed = subprocess.run(
            command,
            cwd=checkout / "class_incremental_learning",
            env=environment,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        atomic_write(log_path, completed.stdout.encode("utf-8", errors="replace"))
        if completed.returncode != 0:
            raise RuntimeError(f"official process exited {completed.returncode}")
        final, average = _parse_curve(completed.stdout)
        tracked_diff = subprocess.run(
            ["git", "diff", "--exit-code"], cwd=checkout, check=False
        )
        if tracked_diff.returncode != 0 or file_sha256(config_path) != config_sha:
            raise RuntimeError("official tracked source or config changed during reproduction")
        return_code = completed.returncode
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
        return_code = completed.returncode if "completed" in locals() else -1
        if not log_path.is_file():
            atomic_write(
                log_path,
                (failure + "\n" + f"python={platform.python_version()}\n").encode("utf-8"),
            )
    wall = time.monotonic() - started
    result = E2LoRAResult(
        failure is None,
        final,
        average,
        return_code,
        wall,
        file_sha256(log_path),
        failure,
        config_sha,
        str(dependency_environment["content_hash"]),
        dependency_versions,
        device_name,
    )
    publish_immutable_json(result_path, result.as_record())
    return result


__all__ = ["E2LoRAResult", "run_official_e2lora"]

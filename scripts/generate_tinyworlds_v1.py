"""Generate or verify the fixed symbolic TinyWorlds v1 benchmark bundles."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from apm.data.text.tinyworlds import (
    TINYWORLDS_VERSION,
    TinyWorldsBundle,
    derive_master_seed,
    generate_calibration_bundle,
    generate_pilot_bundle,
    load_tinyworlds_bundle,
    load_tinyworlds_manifest,
    write_tinyworlds_bundle,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BASE_MANIFEST = (
    REPOSITORY_ROOT
    / "checkpoints"
    / "tinystories-8m"
    / "checkpoint"
    / "manifest.json"
)
OUTPUT_ROOT = REPOSITORY_ROOT / "data" / "tinyworlds" / "v1"
PUBLIC_SEED = 0


def main() -> None:
    """Build the fixed calibration and pilot presets without research switches."""
    manifest_bytes = BASE_MANIFEST.read_bytes()
    base_manifest = json.loads(manifest_bytes)
    base_parameter_checksum = base_manifest["parameter_checksum"]
    master_seed = derive_master_seed(
        TINYWORLDS_VERSION,
        PUBLIC_SEED,
        sha256(manifest_bytes).hexdigest(),
        base_parameter_checksum,
    )
    bundles = (
        ("calibration", generate_calibration_bundle(master_seed)),
        ("pilot", generate_pilot_bundle(master_seed)),
    )
    for name, bundle in bundles:
        target = OUTPUT_ROOT / name
        _write_or_verify(bundle, target)
        manifest = load_tinyworlds_manifest(target)
        print(f"{name}: {target} {manifest.bundle_sha256}", flush=True)


def _write_or_verify(bundle: TinyWorldsBundle, target: Path) -> None:
    if target.exists():
        if load_tinyworlds_bundle(target) != bundle:
            raise RuntimeError(
                f"existing immutable TinyWorlds bundle differs: {target}"
            )
        return
    write_tinyworlds_bundle(bundle, target)


if __name__ == "__main__":
    main()

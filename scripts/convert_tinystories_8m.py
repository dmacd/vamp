#!/usr/bin/env python3
"""Prepare and convert the exact supported TinyStories-8M snapshot."""

from __future__ import annotations

from pathlib import Path
import platform


def main() -> int:
    """Download pinned inputs, validate them, and publish one local artifact."""
    from huggingface_hub import __version__ as huggingface_hub_version
    from huggingface_hub import snapshot_download
    import jax
    import numpy as np
    import torch
    import transformers
    from transformers import GPTNeoForCausalLM

    from apm.lm.tinystories_conversion import (
        TINYSTORIES_SOURCE,
        convert_tinystories_state_dict,
        save_tinystories_artifact,
        tinystories_conversion_provenance,
        validate_pinned_artifact_file,
    )

    if transformers.__version__ != TINYSTORIES_SOURCE.transformers_version:
        raise RuntimeError(
            "conversion requires transformers=="
            f"{TINYSTORIES_SOURCE.transformers_version}; found {transformers.__version__}"
        )
    required_files = (
        TINYSTORIES_SOURCE.config_file,
        TINYSTORIES_SOURCE.model_file,
        *TINYSTORIES_SOURCE.tokenizer_files,
    )
    snapshot = Path(
        snapshot_download(
            repo_id=TINYSTORIES_SOURCE.model_id,
            revision=TINYSTORIES_SOURCE.revision,
            allow_patterns=[artifact.name for artifact in required_files],
        )
    )
    for artifact in required_files:
        validate_pinned_artifact_file(snapshot / artifact.name, artifact)

    model = GPTNeoForCausalLM.from_pretrained(snapshot, local_files_only=True)
    model.eval()
    state_dict = {
        name: np.asarray(tensor.detach().cpu().numpy())
        for name, tensor in model.state_dict().items()
    }
    config_contents = (snapshot / TINYSTORIES_SOURCE.config_file.name).read_bytes()
    converted = convert_tinystories_state_dict(state_dict, config_contents)
    provenance = tinystories_conversion_provenance(
        library_versions=(
            ("huggingface_hub", str(huggingface_hub_version)),
            ("jax", str(jax.__version__)),
            ("numpy", str(np.__version__)),
            ("torch", str(torch.__version__)),
            ("transformers", str(transformers.__version__)),
        ),
        environment=(
            ("platform", platform.platform()),
            ("python", platform.python_version()),
        ),
    )
    artifact = save_tinystories_artifact(
        Path(__file__).resolve().parents[1] / "checkpoints" / "tinystories-8m",
        converted,
        tokenizer_files={
            expected.name: (snapshot / expected.name).read_bytes()
            for expected in TINYSTORIES_SOURCE.tokenizer_files
        },
        provenance=provenance,
    )
    print(artifact.directory)
    print(artifact.checkpoint.reference.parameter_checksum)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

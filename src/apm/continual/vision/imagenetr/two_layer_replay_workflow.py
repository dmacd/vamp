"""Nested two-layer node-latent ablation on the ImageNet-R replay matrix."""

from __future__ import annotations

from pathlib import Path

from apm.continual.artifacts import (
    file_sha256,
    load_canonical_json,
    publish_immutable_json,
    record_sha256,
)
from apm.continual.vision.imagenetr.replay_adaptation_workflow import (
    ReplayAdaptationBootstrap,
    _bootstrap_loaded_replay_adaptation,
    _run_loaded_replay_adaptation,
)
from apm.continual.vision.imagenetr.two_layer_replay_config import (
    DEFAULT_TWO_LAYER_REPLAY_CONFIG,
    TwoLayerReplayConfig,
    load_two_layer_replay_config,
)


def _material_paths(
    project_root: Path, config_path: Path
) -> tuple[Path, ...]:
    package = project_root / "src/apm/continual/vision/imagenetr"
    return (
        config_path,
        project_root / "docs/imagenetr50_logt_two_layer_replay_protocol.md",
        project_root / "scripts/vision/imagenetr/run_two_layer_replay_local.sh",
        project_root / "src/apm/continual/artifacts.py",
        project_root / "src/apm/continual/logt_behavioral_integrator.py",
        project_root / "src/apm/continual/logt_behavioral_router.py",
        project_root / "src/apm/experiments/vamp_logt_mlp_permuted_online.py",
        project_root / "configs/vamp_logt_mlp_permuted_mnist/primary.yaml",
        package,
    )


def _authenticate_single_layer_comparison(
    config: TwoLayerReplayConfig,
) -> tuple[dict[str, object], dict[str, object]]:
    run = config.comparison_artifact_root / "runs" / config.comparison_run_hash
    result_path = run / "evaluations" / "result.json"
    seal_path = run / "protocol" / "training_seal.json"
    if (
        file_sha256(result_path) != config.comparison_result_sha256
        or file_sha256(seal_path) != config.comparison_training_seal_sha256
    ):
        raise ValueError("single-layer comparison artifacts changed")
    result = load_canonical_json(result_path)
    result_core = {
        key: value for key, value in result.items() if key != "content_hash"
    }
    seal = load_canonical_json(seal_path)
    if (
        result.get("schema_version")
        != "imagenetr50-replay-adaptation-result-v1"
        or result.get("protocol_hash") != config.comparison_run_hash
        or result.get("content_hash") != record_sha256(result_core)
        or int(seal.get("test_requests_before_seal", -1)) != 0
    ):
        raise ValueError("single-layer comparison run does not authenticate")
    return result, seal


def bootstrap_two_layer_replay(
    config_path: str | Path = DEFAULT_TWO_LAYER_REPLAY_CONFIG,
) -> ReplayAdaptationBootstrap:
    """Authenticate the v6 comparison and prepare an isolated v7 run."""
    resolved = Path(config_path).resolve()
    project_root = resolved.parents[3]
    config = load_two_layer_replay_config(resolved)
    comparison, comparison_seal = _authenticate_single_layer_comparison(config)
    bootstrap = _bootstrap_loaded_replay_adaptation(  # type: ignore[arg-type]
        resolved, config, _material_paths(project_root, resolved)
    )
    publish_immutable_json(
        bootstrap.store.run / "protocol" / "source_single_layer_training_seal.json",
        comparison_seal,
    )
    publish_immutable_json(
        bootstrap.store.run / "protocol" / "source_single_layer_identity.json",
        {
            "result_content_hash": comparison["content_hash"],
            "result_sha256": config.comparison_result_sha256,
            "run_hash": config.comparison_run_hash,
            "schema_version": "imagenetr50-two-layer-comparison-identity-v1",
            "training_seal_sha256": config.comparison_training_seal_sha256,
        },
    )
    return bootstrap


def run_two_layer_replay(
    config_path: str | Path = DEFAULT_TWO_LAYER_REPLAY_CONFIG,
) -> Path:
    """Run or exactly resume the complete two-layer representation ablation."""
    bootstrap = bootstrap_two_layer_replay(config_path)
    from apm.continual.vision.imagenetr.two_layer_replay_reporting import (
        write_two_layer_replay_report,
    )

    return _run_loaded_replay_adaptation(
        bootstrap, write_two_layer_replay_report
    )


if __name__ == "__main__":
    print(run_two_layer_replay())


__all__ = ["bootstrap_two_layer_replay", "run_two_layer_replay"]

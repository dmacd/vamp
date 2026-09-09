"""One frozen MLP extension to the completed persistent-affine experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from apm.continual.artifacts import record_sha256, require_sha256
from apm.continual.vision.imagenetr.persistent_affine_config import _mapping, _path


DEFAULT_PERSISTENT_MLP_CONFIG = Path("configs/vision/imagenetr/logt_persistent_mlp_v16.yaml")


@dataclass(frozen=True, slots=True)
class PersistentMLPConfig:
    """Declare the architecture change and pin every unchanged comparison input."""

    name: str
    historical_capacity: int
    hidden_dimension: int
    activation: str
    initialization: str
    artifact_root: Path
    affine_config: Path
    reference_run: Path
    affine_config_sha256: str
    reference_result_sha256: str
    reference_result_hash: str

    def __post_init__(self) -> None:
        for value in (self.affine_config_sha256, self.reference_result_sha256, self.reference_result_hash):
            require_sha256(value, "MLP comparison source")
        if (
            self.name != "imagenetr50_logt_persistent_mlp_v16"
            or self.historical_capacity != 4096 or self.hidden_dimension != 1024
            or self.activation != "relu"
            or self.initialization != "signed_union_with_zero_output_random_units"
        ):
            raise ValueError("configuration differs from the single H=4,096 two-layer MLP condition")

    def as_record(self) -> dict[str, object]:
        """Return the path-resolved architecture and source contract."""
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(self).items()
        }

    @property
    def config_hash(self) -> str:
        """Return the architecture identity, including the frozen training-recipe source."""
        return record_sha256(self.as_record())


def load_persistent_mlp_config(path: str | Path = DEFAULT_PERSISTENT_MLP_CONFIG) -> PersistentMLPConfig:
    """Load the sole MLP condition without copying or tuning the affine recipe."""
    import yaml

    source = Path(path).resolve()
    root = _mapping(yaml.safe_load(source.read_text()), "MLP configuration", {"experiment", "paths", "sources"})
    experiment = _mapping(root["experiment"], "MLP experiment", {"name", "historical_capacity", "hidden_dimension", "activation", "initialization"})
    paths = _mapping(root["paths"], "MLP paths", {"artifact_root", "affine_config", "reference_run"})
    sources = _mapping(root["sources"], "MLP sources", {"affine_config_sha256", "reference_result_sha256", "reference_result_hash"})
    return PersistentMLPConfig(
        **dict(experiment),
        **{name: _path(value, source.parents[3]) for name, value in paths.items()},
        **dict(sources),
    )

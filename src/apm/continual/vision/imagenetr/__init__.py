"""ImageNet-R-50 logarithmic VAMP experiment."""

from apm.continual.vision.imagenetr.config import ImageNetRConfig, load_config
from apm.continual.vision.imagenetr.protocol import ResolvedProtocol

__all__ = ["ImageNetRConfig", "ResolvedProtocol", "load_config"]

"""Static configuration values for the plain-JAX GPT-Neo implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AttentionType = Literal["global", "local"]


@dataclass(frozen=True)
class GptNeoConfig:
    """Immutable architecture and dropout configuration for GPT-Neo."""

    vocab_size: int
    max_position_embeddings: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    attention_types: tuple[AttentionType, ...]
    local_window_size: int
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    activation: Literal["gelu_new"] = "gelu_new"
    embedding_dropout: float = 0.0
    attention_dropout: float = 0.0
    residual_dropout: float = 0.0

    def __post_init__(self) -> None:
        """Reject configurations that cannot produce a valid static model."""
        dimensions = {
            "vocab_size": self.vocab_size,
            "max_position_embeddings": self.max_position_embeddings,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "local_window_size": self.local_window_size,
        }
        invalid_dimensions = tuple(name for name, value in dimensions.items() if value <= 0)
        if invalid_dimensions:
            raise ValueError(f"GPT-Neo dimensions must be positive: {invalid_dimensions}")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if len(self.attention_types) != self.num_layers:
            raise ValueError("attention_types must contain one entry per layer")
        if any(attention_type not in ("global", "local") for attention_type in self.attention_types):
            raise ValueError("attention_types may contain only 'global' or 'local'")
        if self.local_window_size > self.max_position_embeddings:
            raise ValueError("local_window_size cannot exceed max_position_embeddings")
        if self.layer_norm_epsilon <= 0.0:
            raise ValueError("layer_norm_epsilon must be positive")
        if self.initializer_range <= 0.0:
            raise ValueError("initializer_range must be positive")
        if self.activation != "gelu_new":
            raise ValueError("the initial GPT-Neo implementation supports only gelu_new")
        dropouts = {
            "embedding_dropout": self.embedding_dropout,
            "attention_dropout": self.attention_dropout,
            "residual_dropout": self.residual_dropout,
        }
        invalid_dropouts = tuple(name for name, value in dropouts.items() if not 0.0 <= value < 1.0)
        if invalid_dropouts:
            raise ValueError(f"dropout probabilities must be in [0, 1): {invalid_dropouts}")

    @property
    def head_size(self) -> int:
        """Return the hidden width assigned to one attention head."""
        return self.hidden_size // self.num_heads

    @property
    def uses_dropout(self) -> bool:
        """Return whether training requires a dropout random key."""
        return any(
            rate > 0.0
            for rate in (
                self.embedding_dropout,
                self.attention_dropout,
                self.residual_dropout,
            )
        )

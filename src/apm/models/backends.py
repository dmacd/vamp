"""Model backend adapters for addressed-memory experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple, Protocol

import jax
import numpy as np

from apm.models.mlp_vae import VaeConfig, VaeParams
from apm.models.vae_losses import flatten_canvases
from apm.training.vae import (
    TrainConfig,
    TrainState,
    config_to_dict,
    continue_train_epochs,
    evaluate_vae,
    init_train_state,
    init_train_state_from_params,
    per_example_observed_energy,
    reconstruct,
)


class BackendTrainState(NamedTuple):
    """Generic train state shape used by benchmark code."""

    params: Any
    opt_state: Any
    rng_key: jax.Array


class ModelBackend(Protocol):
    """Common operations required by Stage 1 memory experiments."""

    kind: str
    accuracy_key: str
    train_config: Any

    def config_payload(self) -> dict[str, object]:
        """Return JSON-serializable backend config."""

    def init_state(self, rng_key: jax.Array) -> BackendTrainState | TrainState:
        """Initialize a fresh model train state."""

    def init_state_from_params(self, params: Any, rng_key: jax.Array) -> BackendTrainState | TrainState:
        """Initialize optimizer state around existing params."""

    def continue_train(
        self,
        state: BackendTrainState | TrainState,
        train_canvases: np.ndarray,
        test_canvases: np.ndarray,
        train_labels: np.ndarray,
        test_labels: np.ndarray,
        collect_epoch_metrics: bool = True,
    ) -> tuple[BackendTrainState | TrainState, list[dict[str, int | float]]]:
        """Continue training on one task."""

    def evaluate(
        self,
        params: Any,
        canvases: np.ndarray,
        labels: np.ndarray,
        rng_key: jax.Array,
    ) -> dict[str, float]:
        """Evaluate params on a canvas batch."""

    def reconstruct(
        self,
        params: Any,
        canvases: np.ndarray,
        rng_key: jax.Array,
        mask_label: bool = False,
    ) -> np.ndarray:
        """Return flat 32x32 reconstructions."""

    def per_example_observed_energy(
        self,
        params: Any,
        canvases: np.ndarray,
        rng_key: jax.Array,
    ) -> np.ndarray:
        """Return raw per-example energy using only observed digit pixels."""


@dataclass(frozen=True)
class VaeBackend:
    """Backend adapter for the existing VAE implementation."""

    vae_config: VaeConfig
    train_config: TrainConfig
    kind: str = "vae"
    accuracy_key: str = "energy_classifier_accuracy"

    def config_payload(self) -> dict[str, object]:
        return {
            "model": {"kind": self.kind},
            "vae": config_to_dict(self.vae_config),
            "train": config_to_dict(self.train_config),
        }

    def init_state(self, rng_key: jax.Array) -> TrainState:
        return init_train_state(rng_key, self.vae_config, self.train_config)

    def init_state_from_params(self, params: VaeParams, rng_key: jax.Array) -> TrainState:
        return init_train_state_from_params(params, rng_key, self.train_config)

    def continue_train(
        self,
        state: TrainState,
        train_canvases: np.ndarray,
        test_canvases: np.ndarray,
        train_labels: np.ndarray,
        test_labels: np.ndarray,
        collect_epoch_metrics: bool = True,
    ) -> tuple[TrainState, list[dict[str, int | float]]]:
        return continue_train_epochs(
            state,
            train_canvases,
            test_canvases,
            train_labels,
            test_labels,
            self.train_config,
            collect_epoch_metrics=collect_epoch_metrics,
        )

    def evaluate(
        self,
        params: VaeParams,
        canvases: np.ndarray,
        labels: np.ndarray,
        rng_key: jax.Array,
    ) -> dict[str, float]:
        return evaluate_vae(params, flatten_canvases(canvases), labels, rng_key, self.train_config)

    def reconstruct(
        self,
        params: VaeParams,
        canvases: np.ndarray,
        rng_key: jax.Array,
        mask_label: bool = False,
    ) -> np.ndarray:
        return reconstruct(params, canvases, rng_key, mask_label=mask_label)

    def per_example_observed_energy(
        self,
        params: VaeParams,
        canvases: np.ndarray,
        rng_key: jax.Array,
    ) -> np.ndarray:
        return np.asarray(
            per_example_observed_energy(params, canvases, rng_key, self.train_config.beta),
            dtype=np.float32,
        )


def make_model_backend(model_kind: str, task_epochs: int, show_progress: bool = False) -> ModelBackend:
    """Create the configured model backend for a benchmark run."""
    if model_kind == "vae":
        return VaeBackend(
            vae_config=VaeConfig(architecture="conv"),
            train_config=TrainConfig(epochs=task_epochs, beta=0.01),
        )
    if model_kind == "fabricpc":
        from apm.models.fabricpc_backend import FabricPcBackend, FabricPcTrainConfig

        return FabricPcBackend(train_config=FabricPcTrainConfig(epochs=task_epochs, show_progress=show_progress))
    raise ValueError(f"unknown model_kind: {model_kind}")

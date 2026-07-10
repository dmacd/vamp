"""Optional FabricPC backend for continuous generative label-canvas models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from apm.data.mnist.label_canvas import (
    CANVAS_SIZE,
    DIGIT_SIZE,
    LABEL_CELL_WIDTH,
    LABEL_CLASSES,
    LABEL_PATCH_COLS,
    LABEL_PATCH_ROWS,
)
from apm.models.backends import BackendTrainState

ProgressCallback = Callable[[], None]


@dataclass(frozen=True)
class FabricPcConfig:
    """Architecture config for the FabricPC generative MNIST backend."""

    latent_dim: int = 64
    hidden_widths: tuple[int, ...] = (256, 128)
    latent_init_std: float = 0.25
    weight_init_std: float = 0.05
    hidden_precision: float = 1.0
    output_precision: float = 1.0


@dataclass(frozen=True)
class FabricPcTrainConfig:
    """Training config for FabricPC PC inference and local learning."""

    seed: int = 0
    batch_size: int = 256
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    infer_steps: int = 40
    eta_infer: float = 0.05
    eval_batch_size: int = 256
    eval_train_count: int | None = 2_000
    eval_test_count: int | None = 2_000
    show_progress: bool = False


class FabricPcBackend:
    """Backend adapter for a top-down generative FabricPC model."""

    kind = "fabricpc"
    accuracy_key = "energy_classifier_accuracy"

    def __init__(
        self,
        model_config: FabricPcConfig | None = None,
        train_config: FabricPcTrainConfig | None = None,
    ) -> None:
        self.model_config = FabricPcConfig() if model_config is None else model_config
        self.train_config = FabricPcTrainConfig() if train_config is None else train_config
        self._fabricpc = _import_fabricpc()
        self.structure = self._build_structure()
        self.optimizer = optax.adamw(
            learning_rate=self.train_config.learning_rate,
            weight_decay=self.train_config.weight_decay,
        )
        self._train_step = jax.jit(
            lambda params, opt_state, batch, rng_key: self._fabricpc["train_step"](
                params,
                opt_state,
                batch,
                self.structure,
                self.optimizer,
                rng_key,
            )
        )
        self._eval_batch = jax.jit(self._evaluate_batch_jax)
        self._observed_digit_energy = jax.jit(
            lambda params, digit, rng_key: self._state_energy(
                self._run_inference_jax(params, {"digit": digit}, digit.shape[0], rng_key)
            )
        )
        self._reconstruct_digit_only = jax.jit(
            lambda params, digit, rng_key: self._digit_label_mu(
                self._run_inference_jax(params, {"digit": digit}, digit.shape[0], rng_key)
            )
        )
        self._reconstruct_digit_label = jax.jit(
            lambda params, digit, label, rng_key: self._digit_label_mu(
                self._run_inference_jax(params, {"digit": digit, "label": label}, digit.shape[0], rng_key)
            )
        )
        self._label_patch_predictions = jax.jit(
            lambda params, digit, rng_key: _decode_label_patch_jax(
                self._run_inference_jax(params, {"digit": digit}, digit.shape[0], rng_key).nodes["label"].z_mu
            )
        )
        self._energy_classifier_predictions = jax.jit(
            lambda params, digit, candidate_patches, rng_key: self._energy_classifier_predictions_jax(
                params,
                digit,
                candidate_patches,
                rng_key,
            )
        )

    def config_payload(self) -> dict[str, object]:
        return {
            "model": {"kind": self.kind},
            "fabricpc": asdict(self.model_config),
            "train": asdict(self.train_config),
        }

    def init_state(self, rng_key: jax.Array) -> BackendTrainState:
        params = self._fabricpc["initialize_params"](self.structure, rng_key)
        return self.init_state_from_params(params, rng_key)

    def init_state_from_params(self, params: Any, rng_key: jax.Array) -> BackendTrainState:
        return BackendTrainState(params=params, opt_state=self.optimizer.init(params), rng_key=rng_key)

    def continue_train(
        self,
        state: BackendTrainState,
        train_canvases: np.ndarray,
        test_canvases: np.ndarray,
        train_labels: np.ndarray,
        test_labels: np.ndarray,
        collect_epoch_metrics: bool = True,
        progress_callback: ProgressCallback | None = None,
    ) -> tuple[BackendTrainState, list[dict[str, int | float]]]:
        train_parts = _canvas_parts(train_canvases)
        params, opt_state, rng_key = state
        metrics_rows: list[dict[str, int | float]] = []
        epoch_iterable = _progress(
            range(self.train_config.epochs),
            enabled=self.train_config.show_progress,
            desc="FabricPC epochs",
            total=self.train_config.epochs,
        )
        for epoch in epoch_iterable:
            rng_key, epoch_key = jax.random.split(rng_key)
            params, opt_state, train_metrics = self._train_epoch(
                params,
                opt_state,
                train_parts,
                epoch_key,
                epoch + 1,
                progress_callback,
            )
            if collect_epoch_metrics:
                train_eval_key, test_eval_key, rng_key = jax.random.split(rng_key, 3)
                train_eval_canvases, train_eval_labels = _evaluation_subset(
                    train_canvases,
                    train_labels,
                    self.train_config.eval_train_count,
                )
                test_eval_canvases, test_eval_labels = _evaluation_subset(
                    test_canvases,
                    test_labels,
                    self.train_config.eval_test_count,
                )
                train_eval = self.evaluate(
                    params,
                    train_eval_canvases,
                    train_eval_labels,
                    train_eval_key,
                    progress_desc=f"epoch {epoch + 1} train eval",
                )
                test_eval = self.evaluate(
                    params,
                    test_eval_canvases,
                    test_eval_labels,
                    test_eval_key,
                    progress_desc=f"epoch {epoch + 1} test eval",
                )
                metrics_rows.append(
                    {
                        "epoch": epoch + 1,
                        **{f"train_{name}": value for name, value in train_metrics.items()},
                        **{f"train_eval_{name}": value for name, value in train_eval.items()},
                        **{f"test_{name}": value for name, value in test_eval.items()},
                    }
                )
                _set_progress_postfix(
                    epoch_iterable,
                    train_loss=train_metrics["loss"],
                    test_loss=test_eval["loss"],
                    test_acc=test_eval["energy_classifier_accuracy"],
                )
            else:
                _set_progress_postfix(epoch_iterable, train_loss=train_metrics["loss"])
        return BackendTrainState(params=params, opt_state=opt_state, rng_key=rng_key), metrics_rows

    def evaluate(
        self,
        params: Any,
        canvases: np.ndarray,
        labels: np.ndarray,
        rng_key: jax.Array,
        progress_callback: ProgressCallback | None = None,
        progress_desc: str | None = None,
    ) -> dict[str, float]:
        if canvases.shape[0] > self.train_config.eval_batch_size:
            return self._evaluate_batched(params, canvases, labels, rng_key, progress_desc, progress_callback)
        metrics = self._evaluate_unbatched(params, canvases, labels, rng_key)
        if progress_callback is not None:
            progress_callback()
        return metrics

    def _evaluate_batched(
        self,
        params: Any,
        canvases: np.ndarray,
        labels: np.ndarray,
        rng_key: jax.Array,
        progress_desc: str | None,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, float]:
        batch_starts = range(0, canvases.shape[0], self.train_config.eval_batch_size)
        batch_iterable = _progress(
            batch_starts,
            enabled=self.train_config.show_progress,
            desc=progress_desc or "FabricPC eval",
            total=len(batch_starts),
            leave=False,
        )
        weighted_rows: list[tuple[int, dict[str, float]]] = []
        for batch_index, start in enumerate(batch_iterable):
            end = min(start + self.train_config.eval_batch_size, canvases.shape[0])
            metrics = self._evaluate_unbatched(
                params,
                canvases[start:end],
                labels[start:end],
                jax.random.fold_in(rng_key, batch_index),
            )
            weighted_rows.append((end - start, metrics))
            _set_progress_postfix(
                batch_iterable,
                loss=metrics["loss"],
                acc=metrics["energy_classifier_accuracy"],
            )
            if progress_callback is not None:
                progress_callback()
        return _weighted_metric_average(weighted_rows)

    def _evaluate_unbatched(
        self,
        params: Any,
        canvases: np.ndarray,
        labels: np.ndarray,
        rng_key: jax.Array,
    ) -> dict[str, float]:
        parts = _canvas_parts(canvases)
        candidate_patches = _candidate_label_patches(parts["digit"].shape[0])
        metrics = self._eval_batch(
            params,
            jnp.asarray(parts["digit"], dtype=jnp.float32),
            jnp.asarray(parts["label"], dtype=jnp.float32),
            jnp.asarray(labels, dtype=jnp.int32),
            jnp.asarray(candidate_patches, dtype=jnp.float32),
            rng_key,
        )
        return {name: float(value) for name, value in metrics.items()}

    def reconstruct(
        self,
        params: Any,
        canvases: np.ndarray,
        rng_key: jax.Array,
        mask_label: bool = False,
    ) -> np.ndarray:
        parts = _canvas_parts(canvases)
        digit_array = jnp.asarray(parts["digit"], dtype=jnp.float32)
        if mask_label:
            digit, label = self._reconstruct_digit_only(params, digit_array, rng_key)
        else:
            digit, label = self._reconstruct_digit_label(
                params,
                digit_array,
                jnp.asarray(parts["label"], dtype=jnp.float32),
                rng_key,
            )
        return _compose_canvas(np.clip(digit, 0.0, 1.0), np.clip(label, 0.0, 1.0)).reshape((canvases.shape[0], -1))

    def per_example_observed_energy(
        self,
        params: Any,
        canvases: np.ndarray,
        rng_key: jax.Array,
        progress_callback: ProgressCallback | None = None,
    ) -> np.ndarray:
        parts = _canvas_parts(canvases)
        if parts["digit"].shape[0] > self.train_config.eval_batch_size:
            return self._observed_digit_energy_batched(params, parts["digit"], rng_key, progress_callback)
        energies = self._observed_digit_energy_unbatched(params, parts["digit"], rng_key)
        if progress_callback is not None:
            progress_callback()
        return energies

    def _observed_digit_energy_batched(
        self,
        params: Any,
        digits: np.ndarray,
        rng_key: jax.Array,
        progress_callback: ProgressCallback | None = None,
    ) -> np.ndarray:
        batch_starts = range(0, digits.shape[0], self.train_config.eval_batch_size)
        batch_iterable = _progress(
            batch_starts,
            enabled=self.train_config.show_progress,
            desc="FabricPC observed energy",
            total=len(batch_starts),
            leave=False,
        )
        batches: list[np.ndarray] = []
        for batch_index, start in enumerate(batch_iterable):
            end = min(start + self.train_config.eval_batch_size, digits.shape[0])
            energies = self._observed_digit_energy_unbatched(
                params,
                digits[start:end],
                jax.random.fold_in(rng_key, batch_index),
            )
            batches.append(energies)
            _set_progress_postfix(batch_iterable, energy=float(np.mean(energies)))
            if progress_callback is not None:
                progress_callback()
        return np.concatenate(batches, axis=0).astype(np.float32)

    def _observed_digit_energy_unbatched(
        self,
        params: Any,
        digits: np.ndarray,
        rng_key: jax.Array,
    ) -> np.ndarray:
        return np.asarray(
            self._observed_digit_energy(params, jnp.asarray(digits, dtype=jnp.float32), rng_key),
            dtype=np.float32,
        )

    def energy_classifier_predictions(
        self,
        params: Any,
        canvases: np.ndarray,
        rng_key: jax.Array,
    ) -> np.ndarray:
        parts = _canvas_parts(canvases)
        candidate_patches = _candidate_label_patches(parts["digit"].shape[0])
        return np.asarray(
            self._energy_classifier_predictions(
                params,
                jnp.asarray(parts["digit"], dtype=jnp.float32),
                jnp.asarray(candidate_patches, dtype=jnp.float32),
                rng_key,
            ),
            dtype=np.int64,
        )

    def label_patch_predictions(
        self,
        params: Any,
        canvases: np.ndarray,
        rng_key: jax.Array,
    ) -> np.ndarray:
        parts = _canvas_parts(canvases)
        return np.asarray(
            self._label_patch_predictions(params, jnp.asarray(parts["digit"], dtype=jnp.float32), rng_key),
            dtype=np.int64,
        )

    def _build_structure(self) -> Any:
        fabricpc = self._fabricpc
        hidden_nodes = [
            fabricpc["Linear"](
                shape=(width,),
                activation=fabricpc["TanhActivation"](),
                energy=fabricpc["GaussianEnergy"](precision=self.model_config.hidden_precision),
                weight_init=fabricpc["NormalInitializer"](std=self.model_config.weight_init_std),
                name=f"hidden_{index}",
            )
            for index, width in enumerate(self.model_config.hidden_widths)
        ]
        latent = fabricpc["Linear"](
            shape=(self.model_config.latent_dim,),
            activation=fabricpc["IdentityActivation"](),
            energy=fabricpc["GaussianEnergy"](precision=self.model_config.hidden_precision),
            latent_init=fabricpc["NormalInitializer"](std=self.model_config.latent_init_std),
            name="latent",
        )
        digit = fabricpc["Linear"](
            shape=(DIGIT_SIZE * DIGIT_SIZE,),
            activation=fabricpc["IdentityActivation"](),
            energy=fabricpc["GaussianEnergy"](precision=self.model_config.output_precision),
            weight_init=fabricpc["NormalInitializer"](std=self.model_config.weight_init_std),
            name="digit",
        )
        label = fabricpc["Linear"](
            shape=(_label_patch_size(),),
            activation=fabricpc["IdentityActivation"](),
            energy=fabricpc["GaussianEnergy"](precision=self.model_config.output_precision),
            weight_init=fabricpc["NormalInitializer"](std=self.model_config.weight_init_std),
            name="label",
        )
        decoder_nodes = [latent, *hidden_nodes]
        edges = [
            fabricpc["Edge"](source=decoder_nodes[index], target=decoder_nodes[index + 1].slot("in"))
            for index in range(len(decoder_nodes) - 1)
        ]
        final_hidden = decoder_nodes[-1]
        edges.extend(
            (
                fabricpc["Edge"](source=final_hidden, target=digit.slot("in")),
                fabricpc["Edge"](source=final_hidden, target=label.slot("in")),
            )
        )
        return fabricpc["graph"](
            nodes=[*decoder_nodes, digit, label],
            edges=edges,
            task_map=fabricpc["TaskMap"](digit=digit, label=label),
            inference=fabricpc["InferenceSGD"](
                eta_infer=self.train_config.eta_infer,
                infer_steps=self.train_config.infer_steps,
            ),
            graph_state_initializer=fabricpc["NodeDistributionStateInit"](),
        )

    def _train_epoch(
        self,
        params: Any,
        opt_state: Any,
        parts: dict[str, np.ndarray],
        rng_key: jax.Array,
        epoch: int,
        progress_callback: ProgressCallback | None = None,
    ) -> tuple[Any, Any, dict[str, float]]:
        permutation = np.asarray(jax.random.permutation(rng_key, parts["digit"].shape[0]))
        energies: list[float] = []
        batch_starts = range(0, permutation.shape[0], self.train_config.batch_size)
        batch_iterable = _progress(
            batch_starts,
            enabled=self.train_config.show_progress,
            desc=f"epoch {epoch} batches",
            total=len(batch_starts),
            leave=False,
        )
        for batch_index, start in enumerate(batch_iterable):
            indices = permutation[start : start + self.train_config.batch_size]
            batch = {
                "digit": jnp.asarray(parts["digit"][indices], dtype=jnp.float32),
                "label": jnp.asarray(parts["label"][indices], dtype=jnp.float32),
            }
            params, opt_state, energy, _ = self._train_step(
                params,
                opt_state,
                batch,
                jax.random.fold_in(rng_key, batch_index),
            )
            energies.append(float(energy) / float(indices.shape[0]))
            _set_progress_postfix(batch_iterable, train_loss=energies[-1])
            if progress_callback is not None:
                progress_callback()
        return params, opt_state, {"loss": float(np.mean(energies)), "normal_energy": float(np.mean(energies))}

    def _evaluate_batch_jax(
        self,
        params: Any,
        digit: jax.Array,
        label: jax.Array,
        labels: jax.Array,
        candidate_patches: jax.Array,
        rng_key: jax.Array,
    ) -> dict[str, jax.Array]:
        recon_key, loss_key, energy_key, label_key = jax.random.split(rng_key, 4)
        recon_digit, recon_label = self._digit_label_mu(
            self._run_inference_jax(params, {"digit": digit}, digit.shape[0], recon_key)
        )
        digit_error = jnp.square(recon_digit - digit)
        label_error = jnp.square(recon_label - label)
        loss = jnp.mean(
            self._state_energy(
                self._run_inference_jax(params, {"digit": digit, "label": label}, digit.shape[0], loss_key)
            )
        )
        energy_predictions = self._energy_classifier_predictions_jax(params, digit, candidate_patches, energy_key)
        label_predictions = _decode_label_patch_jax(
            self._run_inference_jax(params, {"digit": digit}, digit.shape[0], label_key).nodes["label"].z_mu
        )
        return {
            "loss": loss,
            "normal_energy": loss,
            "reconstruction_mse": (jnp.sum(digit_error) + jnp.sum(label_error)) / (digit.shape[0] * CANVAS_SIZE * CANVAS_SIZE),
            "digit_mse": jnp.mean(digit_error),
            "label_patch_mse": jnp.mean(label_error),
            "energy_classifier_accuracy": jnp.mean((energy_predictions == labels).astype(jnp.float32)),
            "label_patch_accuracy": jnp.mean((label_predictions == labels).astype(jnp.float32)),
        }

    def _run_inference_jax(
        self,
        params: Any,
        clamps: dict[str, jax.Array],
        batch_size: int,
        rng_key: jax.Array,
    ) -> Any:
        state = self._fabricpc["initialize_graph_state"](
            self.structure,
            batch_size,
            rng_key,
            clamps=clamps,
            params=params,
        )
        return self._fabricpc["run_inference"](
            params,
            state,
            clamps,
            self.structure,
        )

    def _energy_classifier_predictions_jax(
        self,
        params: Any,
        digit: jax.Array,
        candidate_patches: jax.Array,
        rng_key: jax.Array,
    ) -> jax.Array:
        candidate_labels = jnp.transpose(candidate_patches, (1, 0, 2))
        candidate_keys = jax.random.split(rng_key, LABEL_CLASSES)

        def candidate_energy(args: tuple[jax.Array, jax.Array]) -> jax.Array:
            candidate_label, candidate_key = args
            return self._state_energy(
                self._run_inference_jax(
                    params,
                    {"digit": digit, "label": candidate_label},
                    digit.shape[0],
                    candidate_key,
                )
            )

        energies = jax.lax.map(candidate_energy, (candidate_labels, candidate_keys))
        return jnp.argmin(jnp.transpose(energies, (1, 0)), axis=1).astype(jnp.int32)

    def _state_energy(self, state: Any) -> jax.Array:
        return sum(
            state.nodes[node_name].energy
            for node_name, node in self.structure.nodes.items()
            if node.node_info.in_degree > 0
        )

    def _digit_label_mu(self, state: Any) -> tuple[jax.Array, jax.Array]:
        return state.nodes["digit"].z_mu, state.nodes["label"].z_mu


def _import_fabricpc() -> dict[str, Any]:
    try:
        from fabricpc.core.inference import InferenceSGD, run_inference
        from fabricpc.core.topology import Edge
        from fabricpc.graph_assembly import TaskMap, graph
        from fabricpc.core.activations import IdentityActivation, TanhActivation
        from fabricpc.core.energy import GaussianEnergy
        from fabricpc.core.initializers import NormalInitializer
        from fabricpc.graph_initialization import NodeDistributionStateInit, initialize_graph_state, initialize_params
        from fabricpc.nodes import Linear
        from fabricpc.training import train_step
    except ImportError as exc:
        raise ImportError(
            "FabricPC backend requires the optional pc dependency. "
            "Install it with: ve/bin/python -m pip install -e '.[dev,pc]'"
        ) from exc
    return {
        "Edge": Edge,
        "GaussianEnergy": GaussianEnergy,
        "IdentityActivation": IdentityActivation,
        "InferenceSGD": InferenceSGD,
        "Linear": Linear,
        "NodeDistributionStateInit": NodeDistributionStateInit,
        "NormalInitializer": NormalInitializer,
        "TanhActivation": TanhActivation,
        "TaskMap": TaskMap,
        "graph": graph,
        "initialize_graph_state": initialize_graph_state,
        "initialize_params": initialize_params,
        "run_inference": run_inference,
        "train_step": train_step,
    }


def _canvas_parts(canvases: np.ndarray) -> dict[str, np.ndarray]:
    canvas_array = np.asarray(canvases, dtype=np.float32).reshape((-1, CANVAS_SIZE, CANVAS_SIZE))
    return {
        "digit": canvas_array[:, :DIGIT_SIZE, :DIGIT_SIZE].reshape((canvas_array.shape[0], -1)),
        "label": canvas_array[:, LABEL_PATCH_ROWS, LABEL_PATCH_COLS].reshape((canvas_array.shape[0], -1)),
    }


def _compose_canvas(digit: np.ndarray, label: np.ndarray) -> np.ndarray:
    digit_array = np.asarray(digit, dtype=np.float32)
    label_array = np.asarray(label, dtype=np.float32)
    canvases = np.zeros((digit_array.shape[0], CANVAS_SIZE, CANVAS_SIZE), dtype=np.float32)
    canvases[:, :DIGIT_SIZE, :DIGIT_SIZE] = digit_array.reshape((-1, DIGIT_SIZE, DIGIT_SIZE))
    canvases[:, LABEL_PATCH_ROWS, LABEL_PATCH_COLS] = label_array.reshape(
        (-1, LABEL_PATCH_ROWS.stop - LABEL_PATCH_ROWS.start, LABEL_PATCH_COLS.stop - LABEL_PATCH_COLS.start)
    )
    return canvases


def _candidate_label_patches(batch_size: int) -> np.ndarray:
    patches = np.zeros((batch_size, LABEL_CLASSES, _label_patch_size()), dtype=np.float32)
    for label in range(LABEL_CLASSES):
        patch_canvas = patches[:, label, :].reshape(
            (batch_size, LABEL_PATCH_ROWS.stop - LABEL_PATCH_ROWS.start, LABEL_PATCH_COLS.stop - LABEL_PATCH_COLS.start)
        )
        cell_start = label * LABEL_CELL_WIDTH
        patch_canvas[:, :, cell_start : cell_start + LABEL_CELL_WIDTH] = 1.0
    return patches


def _decode_label_patch(label_patch: np.ndarray) -> np.ndarray:
    patch = np.asarray(label_patch, dtype=np.float32).reshape((-1, 2, LABEL_CLASSES, LABEL_CELL_WIDTH))
    return np.argmax(np.mean(patch, axis=(1, 3)), axis=-1).astype(np.int64)


def _decode_label_patch_jax(label_patch: jax.Array) -> jax.Array:
    patch = jnp.reshape(label_patch, (-1, 2, LABEL_CLASSES, LABEL_CELL_WIDTH))
    return jnp.argmax(jnp.mean(patch, axis=(1, 3)), axis=-1).astype(jnp.int32)


def _evaluation_subset(
    canvases: np.ndarray,
    labels: np.ndarray,
    requested_count: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if requested_count is None or requested_count <= 0 or requested_count >= labels.shape[0]:
        return canvases, labels
    indices = _balanced_prefix_indices(labels, requested_count)
    return canvases[indices], labels[indices]


def _balanced_prefix_indices(labels: np.ndarray, requested_count: int) -> np.ndarray:
    label_array = np.asarray(labels, dtype=np.int64)
    labels_in_order = tuple(int(label) for label in np.unique(label_array))
    target_count = min(requested_count, label_array.shape[0])
    base_count, remainder = divmod(target_count, len(labels_in_order))
    selected = tuple(
        np.flatnonzero(label_array == label)[: base_count + (1 if label_index < remainder else 0)]
        for label_index, label in enumerate(labels_in_order)
    )
    return np.sort(np.concatenate(selected)).astype(np.int64)


def _weighted_metric_average(weighted_rows: list[tuple[int, dict[str, float]]]) -> dict[str, float]:
    total = sum(count for count, _ in weighted_rows)
    return {
        metric_name: float(sum(metrics[metric_name] * count for count, metrics in weighted_rows) / total)
        for metric_name in weighted_rows[0][1]
    }


def _label_patch_size() -> int:
    return (LABEL_PATCH_ROWS.stop - LABEL_PATCH_ROWS.start) * (LABEL_PATCH_COLS.stop - LABEL_PATCH_COLS.start)


def _progress(iterable: Any, enabled: bool, **kwargs: Any) -> Any:
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, dynamic_ncols=True, **kwargs)


def _set_progress_postfix(progress: Any, **values: float) -> None:
    set_postfix = getattr(progress, "set_postfix", None)
    if set_postfix is not None:
        set_postfix({key: f"{value:.4g}" for key, value in values.items()})

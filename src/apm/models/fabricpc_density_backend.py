"""Normalized generative FabricPC density model for image-only evidence.

The legacy :mod:`apm.models.fabricpc_backend` exposes settled prediction error
for a label-canvas model.  That quantity is useful for predictive-coding
inference, but it is not a normalized cross-model density.  This module keeps
the FabricPC graph and local-learning machinery while making the complete
probabilistic model, latent prior, normalization constants, and posterior
volume correction explicit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import io
import math
from pathlib import Path
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from apm.continual.artifacts import (
    file_sha256,
    load_canonical_json,
    publish_immutable_bytes,
    publish_immutable_json,
    require_sha256,
)


@dataclass(frozen=True, slots=True)
class PcDensityConfig:
    """Architecture and globally shared Gaussian precisions."""

    latent_dim: int = 32
    hidden_dim: int = 128
    image_dim: int = 784
    hidden_precision: float = 1.0
    image_precision: float = 25.0
    weight_init_std: float = 0.05

    def __post_init__(self) -> None:
        if (
            self.latent_dim < 1
            or self.hidden_dim < 1
            or self.image_dim < 1
            or self.hidden_precision <= 0.0
            or self.image_precision <= 0.0
            or self.weight_init_std <= 0.0
        ):
            raise ValueError("invalid generative-PC density configuration")

    @property
    def free_dim(self) -> int:
        """Return the number of inferred latent values per example."""
        return self.latent_dim + self.hidden_dim


@dataclass(frozen=True, slots=True)
class PcDensityTrainConfig:
    """Fixed-cost image-model training and latent-settling schedule."""

    seed: int = 0
    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    infer_steps: int = 40
    eta_infer: float = 0.01
    score_batch_size: int = 16
    laplace_floor: float = 1.0e-6
    show_progress: bool = False

    def __post_init__(self) -> None:
        if (
            self.seed < 0
            or self.epochs < 1
            or self.batch_size < 1
            or self.learning_rate <= 0.0
            or self.weight_decay < 0.0
            or self.infer_steps < 1
            or self.eta_infer <= 0.0
            or self.score_batch_size < 1
            or self.laplace_floor < 0.0
        ):
            raise ValueError("invalid fixed generative-PC training configuration")


@dataclass(frozen=True, slots=True)
class PcClassifierConfig:
    """Stopped-gradient linear classifier schedule."""

    epochs: int = 50
    batch_size: int = 128
    learning_rate: float = 1.0e-2
    weight_decay: float = 1.0e-4
    classes: int = 10

    def __post_init__(self) -> None:
        if (
            self.epochs < 1
            or self.batch_size < 1
            or self.learning_rate <= 0.0
            or self.weight_decay < 0.0
            or self.classes < 2
        ):
            raise ValueError("invalid PC classifier configuration")


@dataclass(frozen=True, slots=True)
class PcClassifierState:
    """Node-local linear softmax head, separate from density parameters."""

    weights: jax.Array
    bias: jax.Array


jax.tree_util.register_dataclass(
    PcClassifierState,
    data_fields=("weights", "bias"),
    meta_fields=(),
)


@dataclass(frozen=True, slots=True)
class PcFitResult:
    """Completed fixed-budget FabricPC fit."""

    params: Any
    final_loss: float
    epoch_losses: tuple[float, ...]
    example_presentations: int


@dataclass(frozen=True, slots=True)
class PcClassifierFitResult:
    """Completed stopped-gradient classifier fit."""

    state: PcClassifierState
    final_loss: float
    example_presentations: int


@dataclass(frozen=True, slots=True)
class PcSettledState:
    """Settled free states and convergence diagnostics."""

    latent: np.ndarray
    hidden: np.ndarray
    initial_gradient_norm: np.ndarray
    final_gradient_norm: np.ndarray


@dataclass(frozen=True, slots=True)
class PcEvidenceScores:
    """Uncalibrated image-only evidence and curvature diagnostics."""

    residual: np.ndarray
    map_log_evidence: np.ndarray
    laplace_log_evidence: np.ndarray
    final_gradient_norm: np.ndarray
    minimum_hessian_eigenvalue: np.ndarray
    hessian_was_regularized: np.ndarray
    raw_cholesky_succeeded: np.ndarray

    def __post_init__(self) -> None:
        rows = self.residual.shape
        arrays = (
            self.map_log_evidence,
            self.laplace_log_evidence,
            self.final_gradient_norm,
            self.minimum_hessian_eigenvalue,
            self.hessian_was_regularized,
            self.raw_cholesky_succeeded,
        )
        if len(rows) != 1 or any(value.shape != rows for value in arrays):
            raise ValueError("PC evidence diagnostics are not row aligned")


@dataclass(frozen=True, slots=True)
class PcGaussNewtonScores:
    """Four paired scores and exact curvature diagnostics at one settled state."""

    residual: np.ndarray
    map_log_evidence: np.ndarray
    hessian_laplace_log_evidence: np.ndarray
    gn0_log_evidence: np.ndarray
    gn1_log_evidence: np.ndarray
    final_gradient_norm: np.ndarray
    gn_decrement: np.ndarray
    minimum_hessian_eigenvalue: np.ndarray
    minimum_gauss_newton_eigenvalue: np.ndarray
    hessian_cholesky_succeeded: np.ndarray
    gauss_newton_cholesky_succeeded: np.ndarray
    negative_direction_plus_delta_nll: np.ndarray
    negative_direction_minus_delta_nll: np.ndarray

    def __post_init__(self) -> None:
        rows = self.residual.shape
        vectors = (
            self.map_log_evidence,
            self.hessian_laplace_log_evidence,
            self.gn0_log_evidence,
            self.gn1_log_evidence,
            self.final_gradient_norm,
            self.gn_decrement,
            self.minimum_hessian_eigenvalue,
            self.minimum_gauss_newton_eigenvalue,
            self.hessian_cholesky_succeeded,
            self.gauss_newton_cholesky_succeeded,
        )
        direction_shape = (
            rows[0],
            self.negative_direction_plus_delta_nll.shape[1]
            if self.negative_direction_plus_delta_nll.ndim == 2
            else -1,
        )
        if (
            len(rows) != 1
            or any(value.shape != rows for value in vectors)
            or self.negative_direction_plus_delta_nll.shape != direction_shape
            or self.negative_direction_minus_delta_nll.shape != direction_shape
        ):
            raise ValueError("Gauss-Newton evidence diagnostics are not row aligned")


@dataclass(frozen=True, slots=True)
class GaussNewtonReference:
    """Dense one-state reference result used by analytic protocol tests."""

    gn0_log_evidence: jax.Array
    gn1_log_evidence: jax.Array
    gauss_newton_matrix: jax.Array
    exact_hessian: jax.Array
    gradient: jax.Array


@dataclass(frozen=True, slots=True)
class PcImportanceAudit:
    """Laplace-proposal importance estimates for a fixed diagnostic subset."""

    importance_log_evidence: np.ndarray
    laplace_difference: np.ndarray
    effective_sample_size: np.ndarray
    multistart_map_scores: np.ndarray


@dataclass(frozen=True, slots=True)
class StoredPcModel:
    """Authenticated density parameters and classifier loaded from one node."""

    params: Any
    classifier: PcClassifierState
    density_final_loss: float
    classifier_final_loss: float
    density_example_presentations: int
    classifier_example_presentations: int


class FabricPcDensityBackend:
    """One normalized nonlinear generative image model built with FabricPC."""

    def __init__(
        self,
        model_config: PcDensityConfig | None = None,
        train_config: PcDensityTrainConfig | None = None,
    ) -> None:
        self.model_config = PcDensityConfig() if model_config is None else model_config
        self.train_config = PcDensityTrainConfig() if train_config is None else train_config
        self._fabricpc = _import_fabricpc()
        self.structure = self._build_structure()
        self.optimizer = optax.adamw(
            self.train_config.learning_rate,
            weight_decay=self.train_config.weight_decay,
        )
        self._train_step = jax.jit(
            lambda params, opt_state, batch, key: self._fabricpc["train_step"](
                params,
                opt_state,
                batch,
                self.structure,
                self.optimizer,
                key,
            )
        )
        self._settle_batch = jax.jit(self._settle_batch_jax)
        self._map_scores_from_free = jax.jit(self._map_scores_from_free_jax)
        self._score_batch = jax.jit(self._score_batch_jax)
        self._gauss_newton_from_free = jax.jit(self._gauss_newton_from_free_jax)

    def config_payload(self) -> dict[str, object]:
        """Return all choices that affect a density fit or score."""
        return {
            "model": asdict(self.model_config),
            "training": asdict(self.train_config),
            "fabricpc_version": self._fabricpc["version"],
            "fabricpc_commit": "138941ef5763ab202c7df07879d3f21678e6cc0a",
        }

    def init_params(self, seed: int | None = None) -> Any:
        """Initialize one independent model replica."""
        resolved_seed = self.train_config.seed if seed is None else seed
        if resolved_seed < 0:
            raise ValueError("model seeds must be nonnegative")
        return self._fabricpc["initialize_params"](
            self.structure,
            jax.random.PRNGKey(resolved_seed),
        )

    def fit(self, images: np.ndarray, seed: int | None = None) -> PcFitResult:
        """Train one image density with the declared fixed presentation budget."""
        image_array = _images(images, self.model_config.image_dim)
        resolved_seed = self.train_config.seed if seed is None else seed
        params = self.init_params(resolved_seed)
        opt_state = self.optimizer.init(params)
        epoch_losses: list[float] = []
        iterable = _progress(
            range(self.train_config.epochs),
            self.train_config.show_progress,
            desc="FabricPC density epochs",
        )
        for epoch in iterable:
            generator = np.random.default_rng(resolved_seed * 1_000_003 + epoch)
            permutation = generator.permutation(len(image_array))
            weighted_loss = 0.0
            seen = 0
            for batch_index, start in enumerate(range(0, len(permutation), self.train_config.batch_size)):
                indices = permutation[start : start + self.train_config.batch_size]
                batch = jnp.asarray(image_array[indices])
                params, opt_state, loss, _state = self._train_step(
                    params,
                    opt_state,
                    {
                        "image": batch,
                        "prior": jnp.zeros((len(indices), self.model_config.latent_dim), dtype=batch.dtype),
                    },
                    jax.random.PRNGKey(resolved_seed * 10_000_019 + epoch * 10_007 + batch_index),
                )
                weighted_loss += float(loss) * len(indices)
                seen += len(indices)
            epoch_losses.append(weighted_loss / seen)
            _set_progress(iterable, loss=epoch_losses[-1])
        return PcFitResult(
            params,
            epoch_losses[-1],
            tuple(epoch_losses),
            len(image_array) * self.train_config.epochs,
        )

    def settle_images(self, params: Any, images: np.ndarray) -> PcSettledState:
        """Settle free states from the same zero initialization for every node."""
        image_array = _images(images, self.model_config.image_dim)
        batches = self._padded_batches(image_array, self.train_config.score_batch_size)
        latent_rows: list[np.ndarray] = []
        hidden_rows: list[np.ndarray] = []
        initial_norms: list[np.ndarray] = []
        final_norms: list[np.ndarray] = []
        for batch, valid in batches:
            latent, hidden, initial_norm, final_norm = self._settle_batch(params, jnp.asarray(batch))
            latent_rows.append(np.asarray(latent[:valid]))
            hidden_rows.append(np.asarray(hidden[:valid]))
            initial_norms.append(np.asarray(initial_norm[:valid]))
            final_norms.append(np.asarray(final_norm[:valid]))
        return PcSettledState(
            np.concatenate(latent_rows),
            np.concatenate(hidden_rows),
            np.concatenate(initial_norms),
            np.concatenate(final_norms),
        )

    def score_images(self, params: Any, images: np.ndarray) -> PcEvidenceScores:
        """Return residual, complete MAP, and full-Hessian Laplace scores."""
        image_array = _images(images, self.model_config.image_dim)
        columns: list[list[np.ndarray]] = [[] for _ in range(7)]
        for batch, valid in self._padded_batches(image_array, self.train_config.score_batch_size):
            values = self._score_batch(params, jnp.asarray(batch))
            for destination, value in zip(columns, values, strict=True):
                destination.append(np.asarray(value[:valid]))
        arrays = tuple(np.concatenate(column) for column in columns)
        return PcEvidenceScores(*arrays)

    def map_joint_scores(self, params: Any, images: np.ndarray) -> np.ndarray:
        """Return negative complete joint scores without constructing Hessians."""
        image_array = _images(images, self.model_config.image_dim)
        settled = self.settle_images(params, image_array)
        return self.map_joint_scores_from_settled(params, image_array, settled)

    def map_joint_scores_from_settled(
        self,
        params: Any,
        images: np.ndarray,
        settled: PcSettledState,
    ) -> np.ndarray:
        """Score already-settled states without repeating inference or constructing Hessians."""
        image_array = _images(images, self.model_config.image_dim)
        expected_rows = len(image_array)
        if settled.latent.shape != (expected_rows, self.model_config.latent_dim) or settled.hidden.shape != (
            expected_rows,
            self.model_config.hidden_dim,
        ):
            raise ValueError("settled PC states do not match the supplied images")
        free = jnp.asarray(np.concatenate((settled.latent, settled.hidden), axis=-1))
        return np.asarray(self._map_scores_from_free(params, jnp.asarray(image_array), free))

    def settle_and_score_gauss_newton(
        self,
        params: Any,
        images: np.ndarray,
        negative_direction_epsilons: tuple[float, ...] = (0.01, 0.05, 0.10),
    ) -> tuple[PcSettledState, PcGaussNewtonScores]:
        """Settle once, then compute MAP, raw-Hessian, GN0, and GN1 scores.

        The generalized Gauss-Newton matrix is used exactly as ``A.T @ A``.
        This path never clips eigenvalues, adds damping, or takes absolute
        determinant values. The full-Hessian score is NaN unless its raw
        Cholesky factorization succeeds; that diagnostic never gates GN scores.
        """
        image_array = _images(images, self.model_config.image_dim)
        epsilons = _negative_direction_epsilons(negative_direction_epsilons)
        latent_rows: list[np.ndarray] = []
        hidden_rows: list[np.ndarray] = []
        initial_norm_rows: list[np.ndarray] = []
        final_norm_rows: list[np.ndarray] = []
        columns: list[list[np.ndarray]] = [[] for _ in range(13)]
        for batch, valid in self._padded_batches(image_array, self.train_config.score_batch_size):
            batch_jax = jnp.asarray(batch)
            latent, hidden, initial_norm, final_norm = self._settle_batch(params, batch_jax)
            free = jnp.concatenate((latent, hidden), axis=-1)
            values = self._gauss_newton_from_free(
                params,
                batch_jax,
                free,
                jnp.asarray(epsilons, dtype=batch_jax.dtype),
            )
            latent_rows.append(np.asarray(latent[:valid]))
            hidden_rows.append(np.asarray(hidden[:valid]))
            initial_norm_rows.append(np.asarray(initial_norm[:valid]))
            final_norm_rows.append(np.asarray(final_norm[:valid]))
            for destination, value in zip(columns, values, strict=True):
                destination.append(np.asarray(value[:valid]))
        settled = PcSettledState(
            np.concatenate(latent_rows),
            np.concatenate(hidden_rows),
            np.concatenate(initial_norm_rows),
            np.concatenate(final_norm_rows),
        )
        arrays = tuple(np.concatenate(column) for column in columns)
        scores = PcGaussNewtonScores(*arrays)
        return settled, scores

    def gauss_newton_scores_from_settled(
        self,
        params: Any,
        images: np.ndarray,
        settled: PcSettledState,
        negative_direction_epsilons: tuple[float, ...] = (0.01, 0.05, 0.10),
        *,
        use_float64: bool = False,
    ) -> PcGaussNewtonScores:
        """Score supplied states without repeating inference.

        ``use_float64`` exists only for the fixed numerical audit. Normal
        experiment scoring uses the model's native float32 arrays.
        """
        image_array = _images(images, self.model_config.image_dim)
        expected_rows = len(image_array)
        if settled.latent.shape != (expected_rows, self.model_config.latent_dim) or settled.hidden.shape != (
            expected_rows,
            self.model_config.hidden_dim,
        ):
            raise ValueError("settled PC states do not match the supplied images")
        epsilons = _negative_direction_epsilons(negative_direction_epsilons)
        dtype = jnp.float64 if use_float64 else jnp.float32
        if use_float64 and not jax.config.x64_enabled:
            raise RuntimeError("the float64 GN audit requires jax.experimental.enable_x64")
        score_params = jax.tree_util.tree_map(lambda value: jnp.asarray(value, dtype=dtype), params)
        images_jax = jnp.asarray(image_array, dtype=dtype)
        free = jnp.asarray(np.concatenate((settled.latent, settled.hidden), axis=-1), dtype=dtype)
        columns: list[list[np.ndarray]] = [[] for _ in range(13)]
        batch_size = self.train_config.score_batch_size
        for start in range(0, expected_rows, batch_size):
            stop = min(start + batch_size, expected_rows)
            values = self._gauss_newton_from_free(
                score_params,
                images_jax[start:stop],
                free[start:stop],
                jnp.asarray(epsilons, dtype=dtype),
            )
            for destination, value in zip(columns, values, strict=True):
                destination.append(np.asarray(value))
        return PcGaussNewtonScores(*(np.concatenate(column) for column in columns))

    def reconstruct_images(self, params: Any, images: np.ndarray) -> np.ndarray:
        """Return image means decoded from settled hidden states."""
        settled = self.settle_images(params, images)
        _wh, _bh, wx, bx = self._parameter_arrays(params)
        return np.asarray(jax.nn.sigmoid(jnp.asarray(settled.hidden) @ wx + bx))

    def importance_audit(
        self,
        params: Any,
        images: np.ndarray,
        samples: int = 64,
        starts: int = 4,
        seed: int = 0,
    ) -> PcImportanceAudit:
        """Estimate marginal evidence from the Laplace proposal on a small subset."""
        if samples < 2 or starts < 1 or seed < 0:
            raise ValueError("importance audits require samples, starts, and a nonnegative seed")
        image_array = _images(images, self.model_config.image_dim)
        settled = self.settle_images(params, image_array)
        free = np.concatenate((settled.latent, settled.hidden), axis=-1)
        laplace = self.score_images(params, image_array).laplace_log_evidence
        estimates: list[float] = []
        effective_sizes: list[float] = []
        for row, (image, mode) in enumerate(zip(image_array, free, strict=True)):
            image_jax = jnp.asarray(image)
            mode_jax = jnp.asarray(mode)
            joint = lambda value: self.image_joint_nll(params, image_jax, value)
            hessian = jax.hessian(joint)(mode_jax)
            regularized = hessian + self.train_config.laplace_floor * jnp.eye(
                self.model_config.free_dim,
                dtype=mode_jax.dtype,
            )
            chol = jnp.linalg.cholesky(regularized)
            normal = jax.random.normal(
                jax.random.PRNGKey(seed * 1_000_003 + row),
                (samples, self.model_config.free_dim),
            )
            offsets = jax.scipy.linalg.solve_triangular(chol.T, normal.T, lower=False).T
            proposals = mode_jax + offsets
            log_joint = -jax.vmap(joint)(proposals)
            logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))
            log_proposal = (
                -0.5 * self.model_config.free_dim * jnp.log(2.0 * jnp.pi)
                + 0.5 * logdet
                - 0.5 * jnp.sum(jnp.square(normal), axis=-1)
            )
            log_weights = log_joint - log_proposal
            estimate = jax.scipy.special.logsumexp(log_weights) - jnp.log(samples)
            normalized = jax.nn.softmax(log_weights)
            effective = 1.0 / jnp.sum(jnp.square(normalized))
            estimates.append(float(estimate))
            effective_sizes.append(float(effective))
        multistart = self._multistart_scores(params, image_array, starts, seed)
        importance = np.asarray(estimates, dtype=np.float32)
        return PcImportanceAudit(
            importance,
            importance - laplace,
            np.asarray(effective_sizes, dtype=np.float32),
            multistart,
        )

    def image_joint_nll(self, params: Any, image: jax.Array, free_state: jax.Array) -> jax.Array:
        """Return the complete normalized negative log joint for one example."""
        model = self.model_config
        z = free_state[: model.latent_dim]
        hidden = free_state[model.latent_dim :]
        wh, bh, wx, bx = self._parameter_arrays(params)
        hidden_mean = jnp.tanh(z @ wh + bh)
        image_mean = jax.nn.sigmoid(hidden @ wx + bx)
        dtype = image.dtype
        two_pi = jnp.asarray(2.0 * math.pi, dtype=dtype)
        return (
            0.5 * jnp.sum(jnp.square(z))
            + 0.5 * model.latent_dim * jnp.log(two_pi)
            + 0.5 * model.hidden_precision * jnp.sum(jnp.square(hidden - hidden_mean))
            + 0.5 * model.hidden_dim * jnp.log(two_pi / model.hidden_precision)
            + 0.5 * model.image_precision * jnp.sum(jnp.square(image - image_mean))
            + 0.5 * model.image_dim * jnp.log(two_pi / model.image_precision)
        )

    def residual_energy(self, params: Any, image: jax.Array, free_state: jax.Array) -> jax.Array:
        """Return the legacy hidden-plus-image prediction error without constants."""
        model = self.model_config
        z = free_state[: model.latent_dim]
        hidden = free_state[model.latent_dim :]
        wh, bh, wx, bx = self._parameter_arrays(params)
        hidden_mean = jnp.tanh(z @ wh + bh)
        image_mean = jax.nn.sigmoid(hidden @ wx + bx)
        return 0.5 * model.hidden_precision * jnp.sum(jnp.square(hidden - hidden_mean)) + 0.5 * model.image_precision * jnp.sum(jnp.square(image - image_mean))

    def whitened_residual(
        self,
        params: Any,
        image: jax.Array,
        free_state: jax.Array,
    ) -> jax.Array:
        """Return residuals whose squared norm is the nonconstant joint NLL."""
        model = self.model_config
        z = free_state[: model.latent_dim]
        hidden = free_state[model.latent_dim :]
        wh, bh, wx, bx = self._parameter_arrays(params)
        hidden_error = hidden - jnp.tanh(z @ wh + bh)
        image_error = image - jax.nn.sigmoid(hidden @ wx + bx)
        return jnp.concatenate(
            (
                z,
                jnp.sqrt(jnp.asarray(model.hidden_precision, dtype=image.dtype)) * hidden_error,
                jnp.sqrt(jnp.asarray(model.image_precision, dtype=image.dtype)) * image_error,
            )
        )

    def _build_structure(self) -> Any:
        fabricpc = self._fabricpc
        model = self.model_config
        zeros = fabricpc["ZerosInitializer"]()
        prior = fabricpc["IdentityNode"](
            shape=(model.latent_dim,),
            name="prior",
            latent_init=zeros,
        )
        latent = fabricpc["IdentityNode"](
            shape=(model.latent_dim,),
            name="latent",
            energy=fabricpc["GaussianEnergy"](precision=1.0),
            latent_init=zeros,
        )
        hidden = fabricpc["Linear"](
            shape=(model.hidden_dim,),
            name="hidden",
            activation=fabricpc["TanhActivation"](),
            energy=fabricpc["GaussianEnergy"](precision=model.hidden_precision),
            weight_init=fabricpc["NormalInitializer"](std=model.weight_init_std),
            latent_init=zeros,
        )
        image = fabricpc["Linear"](
            shape=(model.image_dim,),
            name="image",
            activation=fabricpc["SigmoidActivation"](),
            energy=fabricpc["GaussianEnergy"](precision=model.image_precision),
            weight_init=fabricpc["NormalInitializer"](std=model.weight_init_std),
            latent_init=zeros,
        )
        return fabricpc["graph"](
            nodes=[prior, latent, hidden, image],
            edges=[
                fabricpc["Edge"](prior, latent.slot("in")),
                fabricpc["Edge"](latent, hidden.slot("in")),
                fabricpc["Edge"](hidden, image.slot("in")),
            ],
            task_map=fabricpc["TaskMap"](prior=prior, image=image),
            inference=fabricpc["InferenceSGD"](
                eta_infer=self.train_config.eta_infer,
                infer_steps=self.train_config.infer_steps,
            ),
            graph_state_initializer=fabricpc["NodeDistributionStateInit"](),
        )

    def _parameter_arrays(self, params: Any) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        hidden = params.nodes["hidden"]
        image = params.nodes["image"]
        return (
            hidden.weights["latent->hidden:in"],
            hidden.biases["b"][0],
            image.weights["hidden->image:in"],
            image.biases["b"][0],
        )

    def _settle_batch_jax(
        self,
        params: Any,
        images: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        zeros = jnp.zeros((images.shape[0], self.model_config.latent_dim), dtype=images.dtype)
        clamps = {"image": images, "prior": zeros}
        state = self._fabricpc["initialize_graph_state"](
            self.structure,
            images.shape[0],
            jax.random.PRNGKey(0),
            clamps=clamps,
            params=params,
        )
        final_state = self._fabricpc["run_inference"](params, state, clamps, self.structure)
        latent = final_state.nodes["latent"].z_latent
        hidden = final_state.nodes["hidden"].z_latent
        initial_free = jnp.zeros((images.shape[0], self.model_config.free_dim), dtype=images.dtype)
        final_free = jnp.concatenate((latent, hidden), axis=-1)
        gradient = jax.grad(lambda free, image: self.image_joint_nll(params, image, free))
        initial_gradient = jax.vmap(gradient)(initial_free, images)
        final_gradient = jax.vmap(gradient)(final_free, images)
        return (
            latent,
            hidden,
            jnp.linalg.norm(initial_gradient, axis=-1),
            jnp.linalg.norm(final_gradient, axis=-1),
        )

    def _score_batch_jax(
        self,
        params: Any,
        images: jax.Array,
    ) -> tuple[jax.Array, ...]:
        latent, hidden, _initial_norm, final_norm = self._settle_batch_jax(params, images)
        free = jnp.concatenate((latent, hidden), axis=-1)
        joint = lambda image, state: self.image_joint_nll(params, image, state)
        residual = -jax.vmap(lambda image, state: self.residual_energy(params, image, state))(images, free)
        map_score = -jax.vmap(joint)(images, free)
        hessians = jax.vmap(jax.hessian(joint, argnums=1))(images, free)
        eigenvalues = jnp.linalg.eigvalsh(hessians)
        minimum_eigenvalue = eigenvalues[:, 0]
        raw_cholesky = jnp.linalg.cholesky(hessians)
        raw_ok = jnp.all(jnp.isfinite(raw_cholesky), axis=(1, 2))
        identity = jnp.eye(self.model_config.free_dim, dtype=images.dtype)
        regularized = hessians + self.train_config.laplace_floor * identity
        chol = jnp.linalg.cholesky(regularized)
        log_determinant = 2.0 * jnp.sum(jnp.log(jnp.diagonal(chol, axis1=-2, axis2=-1)), axis=-1)
        laplace = map_score + 0.5 * self.model_config.free_dim * jnp.log(
            jnp.asarray(2.0 * math.pi, dtype=images.dtype)
        ) - 0.5 * log_determinant
        return (
            residual,
            map_score,
            laplace,
            final_norm,
            minimum_eigenvalue,
            jnp.logical_not(raw_ok),
            raw_ok,
        )

    def _gauss_newton_from_free_jax(
        self,
        params: Any,
        images: jax.Array,
        free: jax.Array,
        negative_direction_epsilons: jax.Array,
    ) -> tuple[jax.Array, ...]:
        """Compute exact dense GGN and Hessian diagnostics for one score batch."""
        joint = lambda image, state: self.image_joint_nll(params, image, state)
        residual_function = lambda image, state: self.whitened_residual(params, image, state)
        residual_vectors = jax.vmap(residual_function)(images, free)
        residual_score = -jax.vmap(
            lambda image, state: self.residual_energy(params, image, state)
        )(images, free)
        joint_nll = jax.vmap(joint)(images, free)
        map_score = -joint_nll
        jacobians = jax.vmap(jax.jacfwd(residual_function, argnums=1))(images, free)
        gauss_newton = jnp.einsum("bri,brj->bij", jacobians, jacobians)
        gradients = jnp.einsum("bri,br->bi", jacobians, residual_vectors)
        gauss_newton_eigenvalues = jnp.linalg.eigvalsh(gauss_newton)
        minimum_gauss_newton_eigenvalue = gauss_newton_eigenvalues[:, 0]
        gauss_newton_cholesky = jnp.linalg.cholesky(gauss_newton)
        gauss_newton_ok = jnp.all(jnp.isfinite(gauss_newton_cholesky), axis=(1, 2))
        gauss_newton_logdet = 2.0 * jnp.sum(
            jnp.log(jnp.diagonal(gauss_newton_cholesky, axis1=-2, axis2=-1)),
            axis=-1,
        )
        solved_gradient = jax.vmap(
            lambda cholesky, gradient: jax.scipy.linalg.cho_solve(
                (cholesky, True), gradient
            )
        )(gauss_newton_cholesky, gradients)
        decrement = jnp.einsum("bi,bi->b", gradients, solved_gradient)
        volume_constant = 0.5 * self.model_config.free_dim * jnp.log(
            jnp.asarray(2.0 * math.pi, dtype=images.dtype)
        )
        gn0 = map_score + volume_constant - 0.5 * gauss_newton_logdet
        gn1 = gn0 + 0.5 * decrement

        hessians = jax.vmap(jax.hessian(joint, argnums=1))(images, free)
        hessian_eigenvalues, hessian_eigenvectors = jnp.linalg.eigh(hessians)
        minimum_hessian_eigenvalue = hessian_eigenvalues[:, 0]
        hessian_cholesky = jnp.linalg.cholesky(hessians)
        hessian_ok = jnp.all(jnp.isfinite(hessian_cholesky), axis=(1, 2))
        hessian_logdet = 2.0 * jnp.sum(
            jnp.log(jnp.diagonal(hessian_cholesky, axis1=-2, axis2=-1)),
            axis=-1,
        )
        hessian_laplace = jnp.where(
            hessian_ok,
            map_score + volume_constant - 0.5 * hessian_logdet,
            jnp.nan,
        )
        minimum_directions = hessian_eigenvectors[:, :, 0]

        def direction_deltas(
            image: jax.Array,
            state: jax.Array,
            direction: jax.Array,
            base_nll: jax.Array,
            minimum_eigenvalue: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            plus = jax.vmap(lambda epsilon: joint(image, state + epsilon * direction))(
                negative_direction_epsilons
            ) - base_nll
            minus = jax.vmap(lambda epsilon: joint(image, state - epsilon * direction))(
                negative_direction_epsilons
            ) - base_nll
            missing = jnp.full_like(plus, jnp.nan)
            return (
                jnp.where(minimum_eigenvalue < 0.0, plus, missing),
                jnp.where(minimum_eigenvalue < 0.0, minus, missing),
            )

        plus_delta, minus_delta = jax.vmap(direction_deltas)(
            images,
            free,
            minimum_directions,
            joint_nll,
            minimum_hessian_eigenvalue,
        )
        gradient_norm = jnp.linalg.norm(gradients, axis=-1)
        return (
            residual_score,
            map_score,
            hessian_laplace,
            gn0,
            gn1,
            gradient_norm,
            decrement,
            minimum_hessian_eigenvalue,
            minimum_gauss_newton_eigenvalue,
            hessian_ok,
            gauss_newton_ok,
            plus_delta,
            minus_delta,
        )

    def _map_scores_from_free_jax(
        self,
        params: Any,
        images: jax.Array,
        free: jax.Array,
    ) -> jax.Array:
        return -jax.vmap(lambda image, state: self.image_joint_nll(params, image, state))(
            images,
            free,
        )

    def _multistart_scores(
        self,
        params: Any,
        images: np.ndarray,
        starts: int,
        seed: int,
    ) -> np.ndarray:
        gradient = jax.grad(lambda state, image: self.image_joint_nll(params, image, state))

        @jax.jit
        def settle(initial: jax.Array, image: jax.Array) -> jax.Array:
            return jax.lax.fori_loop(
                0,
                self.train_config.infer_steps,
                lambda _step, state: state - self.train_config.eta_infer * gradient(state, image),
                initial,
            )

        rows: list[np.ndarray] = []
        for start in range(starts):
            if start == 0:
                initial = jnp.zeros((len(images), self.model_config.free_dim), dtype=jnp.float32)
            else:
                initial = 0.1 * jax.random.normal(
                    jax.random.PRNGKey(seed * 100_003 + start),
                    (len(images), self.model_config.free_dim),
                )
            modes = jax.vmap(settle)(initial, jnp.asarray(images))
            scores = -jax.vmap(lambda image, state: self.image_joint_nll(params, image, state))(
                jnp.asarray(images),
                modes,
            )
            rows.append(np.asarray(scores))
        return np.stack(rows)

    @staticmethod
    def _padded_batches(images: np.ndarray, batch_size: int) -> tuple[tuple[np.ndarray, int], ...]:
        batches: list[tuple[np.ndarray, int]] = []
        for start in range(0, len(images), batch_size):
            value = images[start : start + batch_size]
            valid = len(value)
            if valid < batch_size:
                value = np.concatenate((value, np.repeat(value[-1:], batch_size - valid, axis=0)))
            batches.append((value, valid))
        return tuple(batches)


def fit_classifier(
    hidden_states: np.ndarray,
    labels: np.ndarray,
    seed: int,
    config: PcClassifierConfig | None = None,
) -> PcClassifierFitResult:
    """Fit a linear head without providing any path back to density parameters."""
    resolved = PcClassifierConfig() if config is None else config
    hidden = np.asarray(hidden_states, dtype=np.float32)
    targets = np.asarray(labels, dtype=np.int32)
    if hidden.ndim != 2 or targets.shape != (len(hidden),) or len(hidden) < 1:
        raise ValueError("classifier inputs are not aligned")
    if np.any(targets < 0) or np.any(targets >= resolved.classes):
        raise ValueError("classifier labels are outside the declared class set")
    key = jax.random.PRNGKey(seed)
    weights = jax.random.normal(key, (hidden.shape[1], resolved.classes)) * 0.01
    state = PcClassifierState(weights, jnp.zeros((resolved.classes,), dtype=weights.dtype))
    optimizer = optax.adamw(resolved.learning_rate, weight_decay=resolved.weight_decay)
    opt_state = optimizer.init(state)

    def loss_fn(candidate: PcClassifierState, x: jax.Array, y: jax.Array) -> jax.Array:
        logits = jax.lax.stop_gradient(x) @ candidate.weights + candidate.bias
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y))

    @jax.jit
    def step(
        candidate: PcClassifierState,
        optimizer_state: optax.OptState,
        x: jax.Array,
        y: jax.Array,
    ) -> tuple[PcClassifierState, optax.OptState, jax.Array]:
        loss, gradients = jax.value_and_grad(loss_fn)(candidate, x, y)
        updates, optimizer_state = optimizer.update(gradients, optimizer_state, candidate)
        return optax.apply_updates(candidate, updates), optimizer_state, loss

    final_loss = math.nan
    for epoch in range(resolved.epochs):
        permutation = np.random.default_rng(seed * 1_000_003 + epoch).permutation(len(hidden))
        for start in range(0, len(hidden), resolved.batch_size):
            indices = permutation[start : start + resolved.batch_size]
            state, opt_state, loss = step(
                state,
                opt_state,
                jnp.asarray(hidden[indices]),
                jnp.asarray(targets[indices]),
            )
            final_loss = float(loss)
    return PcClassifierFitResult(state, final_loss, len(hidden) * resolved.epochs)


def classifier_logits(state: PcClassifierState, hidden_states: np.ndarray) -> np.ndarray:
    """Return node-local logits for already-settled hidden states."""
    hidden = np.asarray(hidden_states, dtype=np.float32)
    if hidden.ndim != 2 or hidden.shape[1] != state.weights.shape[0]:
        raise ValueError("classifier hidden-state width changed")
    return np.asarray(jnp.asarray(hidden) @ state.weights + state.bias)


def classifier_cross_entropy(
    state: PcClassifierState,
    hidden_states: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Return per-example true-label cross entropy for oracle diagnostics."""
    logits = jnp.asarray(classifier_logits(state, hidden_states))
    targets = jnp.asarray(labels, dtype=jnp.int32)
    if targets.shape != (logits.shape[0],):
        raise ValueError("classifier labels are not row aligned")
    return np.asarray(optax.softmax_cross_entropy_with_integer_labels(logits, targets))


def laplace_log_evidence_at_state(
    negative_log_joint: Any,
    free_state: jax.Array,
    regularizer: float = 0.0,
) -> tuple[jax.Array, jax.Array]:
    """Evaluate the full-Hessian Laplace formula for an arbitrary latent joint.

    This small reference path is intentionally separate from batched FabricPC
    execution so float64 analytic tests can validate the prior, constants, and
    determinant sign against closed-form densities.
    """
    if regularizer < 0.0:
        raise ValueError("Laplace regularization cannot be negative")
    state = jnp.asarray(free_state)
    if state.ndim != 1:
        raise ValueError("Laplace reference scoring requires one flat latent state")
    hessian = jax.hessian(negative_log_joint)(state)
    regularized = hessian + regularizer * jnp.eye(len(state), dtype=state.dtype)
    sign, log_determinant = jnp.linalg.slogdet(regularized)
    score = (
        -negative_log_joint(state)
        + 0.5 * len(state) * jnp.log(jnp.asarray(2.0 * math.pi, dtype=state.dtype))
        - 0.5 * log_determinant
    )
    return score, sign


def gauss_newton_log_evidence_at_state(
    residual_function: Callable[[jax.Array], jax.Array],
    negative_log_joint: Callable[[jax.Array], jax.Array],
    free_state: jax.Array,
) -> GaussNewtonReference:
    """Return dense GN0/GN1 reference scores without damping or clipping."""
    state = jnp.asarray(free_state)
    if state.ndim != 1:
        raise ValueError("Gauss-Newton reference scoring requires one flat latent state")
    residual = residual_function(state)
    if residual.ndim != 1:
        raise ValueError("the whitened residual function must return one flat vector")
    jacobian = jax.jacfwd(residual_function)(state)
    matrix = jacobian.T @ jacobian
    cholesky = jnp.linalg.cholesky(matrix)
    log_determinant = 2.0 * jnp.sum(jnp.log(jnp.diag(cholesky)))
    gradient = jax.grad(negative_log_joint)(state)
    solved = jax.scipy.linalg.cho_solve((cholesky, True), gradient)
    decrement = gradient @ solved
    volume_constant = 0.5 * len(state) * jnp.log(
        jnp.asarray(2.0 * math.pi, dtype=state.dtype)
    )
    gn0 = -negative_log_joint(state) + volume_constant - 0.5 * log_determinant
    return GaussNewtonReference(
        gn0,
        gn0 + 0.5 * decrement,
        matrix,
        jax.hessian(negative_log_joint)(state),
        gradient,
    )


def publish_pc_model(
    directory: Path,
    backend: FabricPcDensityBackend,
    node_id: str,
    replica_seed: int,
    density: PcFitResult,
    classifier: PcClassifierFitResult,
) -> StoredPcModel:
    """Publish a completed node model once using a non-pickle array archive."""
    require_sha256(node_id, "PC node ID")
    archive_path = directory / "model.npz"
    manifest_path = directory / "manifest.json"
    if archive_path.is_file() and manifest_path.is_file():
        return load_pc_model(directory, backend, node_id, replica_seed)
    leaves, _tree = jax.tree_util.tree_flatten(density.params)
    arrays = {f"density_{index}": np.asarray(value) for index, value in enumerate(leaves)}
    arrays["classifier_weights"] = np.asarray(classifier.state.weights)
    arrays["classifier_bias"] = np.asarray(classifier.state.bias)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    directory.mkdir(parents=True, exist_ok=True)
    publish_immutable_bytes(archive_path, buffer.getvalue())
    publish_immutable_json(
        manifest_path,
        {
            "archive_sha256": file_sha256(archive_path),
            "classifier_example_presentations": classifier.example_presentations,
            "classifier_final_loss": classifier.final_loss,
            "config": backend.config_payload(),
            "density_example_presentations": density.example_presentations,
            "density_final_loss": density.final_loss,
            "density_leaves": [
                {"dtype": str(np.asarray(value).dtype), "shape": list(np.asarray(value).shape)}
                for value in leaves
            ],
            "node_id": node_id,
            "replica_seed": replica_seed,
            "schema_version": "vamp-logt-pc-node-v1",
        },
    )
    return load_pc_model(directory, backend, node_id, replica_seed)


def load_pc_model(
    directory: Path,
    backend: FabricPcDensityBackend,
    node_id: str,
    replica_seed: int,
) -> StoredPcModel:
    """Authenticate and load one fixed-graph node model."""
    require_sha256(node_id, "PC node ID")
    archive_path = directory / "model.npz"
    record = load_canonical_json(directory / "manifest.json")
    if (
        record.get("schema_version") != "vamp-logt-pc-node-v1"
        or record.get("node_id") != node_id
        or record.get("replica_seed") != replica_seed
        or record.get("config") != backend.config_payload()
        or record.get("archive_sha256") != file_sha256(archive_path)
    ):
        raise ValueError("PC model artifact coordinates or content changed")
    template = backend.init_params(replica_seed)
    template_leaves, tree = jax.tree_util.tree_flatten(template)
    specs = record.get("density_leaves")
    if not isinstance(specs, list) or len(specs) != len(template_leaves):
        raise ValueError("PC density parameter structure changed")
    with np.load(archive_path, allow_pickle=False) as archive:
        leaves = []
        for index, (template_value, spec) in enumerate(zip(template_leaves, specs, strict=True)):
            value = archive[f"density_{index}"]
            if (
                not isinstance(spec, dict)
                or spec.get("shape") != list(template_value.shape)
                or spec.get("dtype") != str(template_value.dtype)
                or value.shape != template_value.shape
                or str(value.dtype) != str(template_value.dtype)
            ):
                raise ValueError("PC density parameter metadata changed")
            leaves.append(jnp.asarray(value))
        classifier = PcClassifierState(
            jnp.asarray(archive["classifier_weights"]),
            jnp.asarray(archive["classifier_bias"]),
        )
    return StoredPcModel(
        jax.tree_util.tree_unflatten(tree, leaves),
        classifier,
        float(record["density_final_loss"]),
        float(record["classifier_final_loss"]),
        int(record["density_example_presentations"]),
        int(record["classifier_example_presentations"]),
    )


def _images(value: np.ndarray, image_dim: int) -> np.ndarray:
    images = np.asarray(value)
    if images.ndim != 2 or images.shape[1] != image_dim or len(images) < 1:
        raise ValueError(f"PC evidence requires a nonempty [rows, {image_dim}] image matrix")
    if not np.issubdtype(images.dtype, np.floating):
        raise TypeError("PC evidence requires floating image values")
    images = images.astype(np.float32, copy=False)
    if not np.all(np.isfinite(images)) or np.any(images < 0.0) or np.any(images > 1.0):
        raise ValueError("PC image values must be finite and in [0, 1]")
    return images


def _negative_direction_epsilons(value: tuple[float, ...]) -> tuple[float, ...]:
    if not value or any(not math.isfinite(item) or item <= 0.0 for item in value):
        raise ValueError("negative-direction probe sizes must be finite and positive")
    if tuple(sorted(set(value))) != value:
        raise ValueError("negative-direction probe sizes must be unique and increasing")
    return value


def _import_fabricpc() -> dict[str, Any]:
    try:
        from importlib.metadata import version

        from fabricpc.core.activations import SigmoidActivation, TanhActivation
        from fabricpc.core.energy import GaussianEnergy
        from fabricpc.core.inference import InferenceSGD, run_inference
        from fabricpc.core.initializers import NormalInitializer, ZerosInitializer
        from fabricpc.core.topology import Edge
        from fabricpc.graph_assembly import TaskMap, graph
        from fabricpc.graph_initialization import (
            NodeDistributionStateInit,
            initialize_graph_state,
            initialize_params,
        )
        from fabricpc.nodes import IdentityNode, Linear
        from fabricpc.training import train_step
    except ImportError as error:
        raise ImportError(
            "Generative PC evidence requires FabricPC 0.4.0. "
            "Install it with: uv sync --python 3.11 --extra pc"
        ) from error
    resolved_version = version("fabricpc")
    if resolved_version != "0.4.0":
        raise RuntimeError(f"generative PC evidence requires FabricPC 0.4.0, found {resolved_version}")
    return {
        "Edge": Edge,
        "GaussianEnergy": GaussianEnergy,
        "IdentityNode": IdentityNode,
        "InferenceSGD": InferenceSGD,
        "Linear": Linear,
        "NodeDistributionStateInit": NodeDistributionStateInit,
        "NormalInitializer": NormalInitializer,
        "SigmoidActivation": SigmoidActivation,
        "TanhActivation": TanhActivation,
        "TaskMap": TaskMap,
        "ZerosInitializer": ZerosInitializer,
        "graph": graph,
        "initialize_graph_state": initialize_graph_state,
        "initialize_params": initialize_params,
        "run_inference": run_inference,
        "train_step": train_step,
        "version": resolved_version,
    }


def _progress(iterable: Any, enabled: bool, **kwargs: Any) -> Any:
    if not enabled:
        return iterable
    from tqdm.auto import tqdm

    return tqdm(iterable, dynamic_ncols=True, **kwargs)


def _set_progress(progress: Any, **values: float) -> None:
    method = getattr(progress, "set_postfix", None)
    if method is not None:
        method(values)


__all__ = [
    "FabricPcDensityBackend",
    "PcClassifierConfig",
    "PcClassifierFitResult",
    "PcClassifierState",
    "PcDensityConfig",
    "PcDensityTrainConfig",
    "PcEvidenceScores",
    "PcGaussNewtonScores",
    "PcFitResult",
    "PcImportanceAudit",
    "PcSettledState",
    "StoredPcModel",
    "classifier_cross_entropy",
    "classifier_logits",
    "fit_classifier",
    "gauss_newton_log_evidence_at_state",
    "GaussNewtonReference",
    "laplace_log_evidence_at_state",
    "load_pc_model",
    "publish_pc_model",
]

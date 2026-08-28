from __future__ import annotations

import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from apm.models.fabricpc_density_backend import (
    FabricPcDensityBackend,
    PcClassifierConfig,
    PcClassifierState,
    PcDensityConfig,
    PcDensityTrainConfig,
    classifier_logits,
    fit_classifier,
    gauss_newton_log_evidence_at_state,
    laplace_log_evidence_at_state,
    load_pc_model,
    publish_pc_model,
)


def test_linear_gaussian_laplace_equals_exact_marginal() -> None:
    with jax.experimental.enable_x64():
        weight = jnp.asarray([[0.7, -0.2], [0.1, 0.5], [-0.3, 0.4]], dtype=jnp.float64)
        bias = jnp.asarray([0.2, -0.1, 0.3], dtype=jnp.float64)
        variance = 0.4
        precision = 1.0 / variance
        covariance = weight @ weight.T + variance * jnp.eye(3, dtype=jnp.float64)
        inverse = jnp.linalg.inv(covariance)
        sign, logdet = jnp.linalg.slogdet(covariance)
        assert sign == 1

        errors = []
        missing_prior_errors = []
        for image in (
            jnp.asarray([0.1, 0.4, -0.2], dtype=jnp.float64),
            jnp.asarray([1.2, -0.7, 0.3], dtype=jnp.float64),
            jnp.asarray([-0.8, 0.2, 1.1], dtype=jnp.float64),
        ):
            def joint(latent: jax.Array) -> jax.Array:
                residual = image - (weight @ latent + bias)
                return (
                    0.5 * jnp.sum(jnp.square(latent))
                    + latent.size / 2.0 * jnp.log(2.0 * jnp.pi)
                    + 0.5 * precision * jnp.sum(jnp.square(residual))
                    + image.size / 2.0 * jnp.log(2.0 * jnp.pi * variance)
                )

            hessian = jnp.eye(2, dtype=jnp.float64) + precision * weight.T @ weight
            mode = jnp.linalg.solve(hessian, precision * weight.T @ (image - bias))
            laplace, laplace_sign = laplace_log_evidence_at_state(joint, mode)
            delta = image - bias
            exact = -0.5 * (
                image.size * jnp.log(2.0 * jnp.pi)
                + logdet
                + delta @ inverse @ delta
            )
            errors.append(abs(float(laplace - exact)))
            assert laplace_sign == 1

            def missing_prior(latent: jax.Array) -> jax.Array:
                residual = image - (weight @ latent + bias)
                return (
                    0.5 * precision * jnp.sum(jnp.square(residual))
                    + image.size / 2.0 * jnp.log(2.0 * jnp.pi * variance)
                )

            bad_hessian = precision * weight.T @ weight
            bad_mode = jnp.linalg.solve(bad_hessian, precision * weight.T @ (image - bias))
            bad_score, _ = laplace_log_evidence_at_state(missing_prior, bad_mode)
            missing_prior_errors.append(abs(float(bad_score - exact)))

    assert max(errors) < 1.0e-4
    assert min(missing_prior_errors) > 1.0e-2


def test_linear_gaussian_gn0_at_mode_and_gn1_away_are_exact() -> None:
    with jax.experimental.enable_x64():
        weight = jnp.asarray([[0.7, -0.2], [0.1, 0.5], [-0.3, 0.4]], dtype=jnp.float64)
        bias = jnp.asarray([0.2, -0.1, 0.3], dtype=jnp.float64)
        image = jnp.asarray([1.2, -0.7, 0.3], dtype=jnp.float64)
        variance = 0.4
        precision = 1.0 / variance

        def residual(state: jax.Array) -> jax.Array:
            return jnp.concatenate((state, jnp.sqrt(precision) * (image - weight @ state - bias)))

        def joint(state: jax.Array) -> jax.Array:
            return (
                0.5 * jnp.sum(residual(state) ** 2)
                + state.size / 2.0 * jnp.log(2.0 * jnp.pi)
                + image.size / 2.0 * jnp.log(2.0 * jnp.pi * variance)
            )

        normal_matrix = jnp.eye(2, dtype=jnp.float64) + precision * weight.T @ weight
        mode = jnp.linalg.solve(normal_matrix, precision * weight.T @ (image - bias))
        covariance = weight @ weight.T + variance * jnp.eye(3, dtype=jnp.float64)
        exact = jax.scipy.stats.multivariate_normal.logpdf(image, bias, covariance)
        at_mode = gauss_newton_log_evidence_at_state(residual, joint, mode)
        away = gauss_newton_log_evidence_at_state(
            residual,
            joint,
            jnp.asarray([0.8, -0.6], dtype=jnp.float64),
        )

    np.testing.assert_allclose(at_mode.gauss_newton_matrix, at_mode.exact_hessian, atol=1e-12)
    np.testing.assert_allclose(at_mode.gn0_log_evidence, exact, atol=1e-10)
    np.testing.assert_allclose(away.gn1_log_evidence, exact, atol=1e-10)


def test_nonlinear_exact_hessian_can_be_indefinite_while_g_is_positive() -> None:
    with jax.experimental.enable_x64():
        residual = lambda state: jnp.asarray([state[0], state[0] ** 2 - 1.0])
        joint = lambda state: 0.5 * jnp.sum(residual(state) ** 2)
        state = jnp.asarray([0.0], dtype=jnp.float64)
        result = gauss_newton_log_evidence_at_state(residual, joint, state)

    assert float(jnp.linalg.eigvalsh(result.exact_hessian)[0]) < 0.0
    assert float(jnp.linalg.eigvalsh(result.gauss_newton_matrix)[0]) > 0.0
    assert float(joint(jnp.asarray([0.1]))) < float(joint(state))
    assert float(joint(jnp.asarray([-0.1]))) < float(joint(state))


def test_complete_joint_contains_prior_and_precision_constants() -> None:
    backend = _backend(hidden_precision=4.0, image_precision=9.0)
    params = backend.init_params(0)
    image = jnp.asarray([0.2, 0.7, 0.4], dtype=jnp.float32)
    free = jnp.asarray([0.1, -0.2, 0.3], dtype=jnp.float32)
    z = free[:1]
    hidden = free[1:]
    wh, bh, wx, bx = backend._parameter_arrays(params)
    hidden_mean = jnp.tanh(z @ wh + bh)
    image_mean = jax.nn.sigmoid(hidden @ wx + bx)
    expected = (
        0.5 * jnp.sum(z**2)
        + 0.5 * math.log(2.0 * math.pi)
        + 2.0 * jnp.sum((hidden - hidden_mean) ** 2)
        + math.log(2.0 * math.pi / 4.0)
        + 4.5 * jnp.sum((image - image_mean) ** 2)
        + 1.5 * math.log(2.0 * math.pi / 9.0)
    )
    np.testing.assert_allclose(backend.image_joint_nll(params, image, free), expected, rtol=1e-6)
    residual_nll = 0.5 * jnp.sum(backend.whitened_residual(params, image, free) ** 2)
    constants = (
        0.5 * math.log(2.0 * math.pi)
        + math.log(2.0 * math.pi / 4.0)
        + 1.5 * math.log(2.0 * math.pi / 9.0)
    )
    np.testing.assert_allclose(residual_nll + constants, expected, rtol=1e-6)


def test_scoring_is_deterministic_and_image_only() -> None:
    backend = _backend()
    images = _images()
    fit = backend.fit(images, seed=3)
    first = backend.score_images(fit.params, images)
    second = backend.score_images(fit.params, images)
    for name in first.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
    with pytest.raises(TypeError):
        backend.score_images(fit.params, images, context_ids=np.zeros(len(images)))  # type: ignore[call-arg]


def test_gauss_newton_scoring_is_deterministic_positive_and_paired() -> None:
    backend = _backend()
    images = _images()
    fit = backend.fit(images, seed=3)
    settled, first = backend.settle_and_score_gauss_newton(fit.params, images)
    second = backend.gauss_newton_scores_from_settled(fit.params, images, settled)

    for name in first.__dataclass_fields__:
        np.testing.assert_allclose(
            getattr(first, name),
            getattr(second, name),
            equal_nan=True,
            rtol=1e-6,
            atol=1e-6,
        )
    np.testing.assert_allclose(
        first.map_log_evidence,
        backend.map_joint_scores_from_settled(fit.params, images, settled),
        rtol=1e-6,
    )
    assert np.all(first.gauss_newton_cholesky_succeeded)
    assert np.all(first.minimum_gauss_newton_eigenvalue > 0.0)
    assert np.all(first.gn1_log_evidence >= first.gn0_log_evidence)
    np.testing.assert_allclose(
        first.gn1_log_evidence - first.gn0_log_evidence,
        0.5 * first.gn_decrement,
        rtol=1e-6,
        atol=5.0e-8,
    )


def test_classifier_parameters_and_labels_cannot_change_evidence() -> None:
    backend = _backend()
    images = _images()
    fit = backend.fit(images, seed=4)
    settled = backend.settle_images(fit.params, images)
    before = backend.map_joint_scores(fit.params, images)
    first = fit_classifier(
        settled.hidden,
        np.asarray([0, 1, 0, 1], dtype=np.int64),
        2,
        PcClassifierConfig(epochs=2, batch_size=2, classes=2),
    )
    second = fit_classifier(
        settled.hidden,
        np.asarray([1, 0, 1, 0], dtype=np.int64),
        2,
        PcClassifierConfig(epochs=2, batch_size=2, classes=2),
    )
    after = backend.map_joint_scores(fit.params, images)
    np.testing.assert_array_equal(before, after)
    assert not np.array_equal(
        classifier_logits(first.state, settled.hidden),
        classifier_logits(second.state, settled.hidden),
    )


def test_map_scoring_never_constructs_a_hessian(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _backend()
    images = _images()
    fit = backend.fit(images, seed=4)
    monkeypatch.setattr(
        jax,
        "hessian",
        lambda *_args, **_kwargs: pytest.fail("MAP scoring attempted to construct a Hessian"),
    )

    first = backend.map_joint_scores(fit.params, images)
    settled = backend.settle_images(fit.params, images)
    second = backend.map_joint_scores_from_settled(fit.params, images, settled)

    np.testing.assert_array_equal(first, second)


def test_autodiff_hessian_matches_finite_differences() -> None:
    backend = _backend()
    params = backend.init_params(5)
    with jax.experimental.enable_x64():
        image = jnp.asarray(_images()[0], dtype=jnp.float64)
        free = jnp.asarray([0.1, -0.2, 0.3], dtype=jnp.float64)
        gradient = jax.grad(lambda value: backend.image_joint_nll(params, image, value))
        autodiff = np.asarray(jax.hessian(lambda value: backend.image_joint_nll(params, image, value))(free))
        epsilon = 1.0e-4
        columns = []
        for index in range(len(free)):
            direction = np.zeros(len(free), dtype=np.float64)
            direction[index] = epsilon
            columns.append(
                (
                    np.asarray(gradient(free + direction))
                    - np.asarray(gradient(free - direction))
                )
                / (2.0 * epsilon)
            )
    finite_difference = np.stack(columns, axis=1)
    np.testing.assert_allclose(autodiff, finite_difference, atol=2.0e-4, rtol=2.0e-4)


def test_nonpickle_model_artifact_round_trip_and_tamper_rejection(tmp_path: Path) -> None:
    backend = _backend()
    images = _images()
    density = backend.fit(images, seed=7)
    settled = backend.settle_images(density.params, images)
    classifier = fit_classifier(
        settled.hidden,
        np.asarray([0, 1, 0, 1]),
        7,
        PcClassifierConfig(epochs=1, batch_size=2, classes=2),
    )
    node_id = "a" * 64
    stored = publish_pc_model(tmp_path, backend, node_id, 7, density, classifier)
    loaded = load_pc_model(tmp_path, backend, node_id, 7)
    expected = backend.map_joint_scores(stored.params, images)
    actual = backend.map_joint_scores(loaded.params, images)
    np.testing.assert_array_equal(expected, actual)
    assert loaded.classifier.weights.shape == (2, 2)

    payload = bytearray((tmp_path / "model.npz").read_bytes())
    payload[-1] ^= 1
    (tmp_path / "model.npz").write_bytes(payload)
    with pytest.raises(ValueError, match="content changed"):
        load_pc_model(tmp_path, backend, node_id, 7)


def _backend(hidden_precision: float = 2.0, image_precision: float = 3.0) -> FabricPcDensityBackend:
    return FabricPcDensityBackend(
        PcDensityConfig(
            latent_dim=1,
            hidden_dim=2,
            image_dim=3,
            hidden_precision=hidden_precision,
            image_precision=image_precision,
            weight_init_std=0.05,
        ),
        PcDensityTrainConfig(
            seed=0,
            epochs=1,
            batch_size=2,
            infer_steps=3,
            eta_infer=0.01,
            score_batch_size=2,
            laplace_floor=1.0e-6,
        ),
    )


def _images() -> np.ndarray:
    return np.asarray(
        [
            [0.0, 0.2, 1.0],
            [1.0, 0.7, 0.0],
            [0.3, 0.8, 0.4],
            [0.9, 0.1, 0.6],
        ],
        dtype=np.float32,
    )

from __future__ import annotations

import numpy as np

from sciona.atoms.state_estimation.kalman_filters import filter_rs
from sciona.atoms.state_estimation.kalman_filters.filter_rs import (
    evaluate_measurement_oracle,
    initialize_kalman_state_model,
    predict_latent_state_and_covariance,
    predict_latent_state_steady_state,
    update_posterior_state_and_covariance,
    update_posterior_state_steady_state,
)


def test_initialize_kalman_state_model_returns_state_dict() -> None:
    state = initialize_kalman_state_model({"initial_state": [0.0, 1.0]})
    assert set(state) == {"x", "P"}
    assert state["x"].shape == (2,)
    assert state["P"].shape == (2, 2)


def test_evaluate_measurement_oracle_returns_prediction_and_innovation() -> None:
    z_pred, innovation = evaluate_measurement_oracle(
        np.array([1.0, 2.0]),
        np.array([1.5]),
        np.array([[1.0, 0.0]]),
    )
    assert np.allclose(z_pred, np.array([1.0]))
    assert np.allclose(innovation, np.array([0.5]))


def test_predict_and_update_covariance_paths_return_finite_state() -> None:
    state = initialize_kalman_state_model({"x": [0.0, 0.0], "P": np.eye(2)})
    predicted = predict_latent_state_and_covariance(
        state,
        np.array([1.0]),
        np.array([[1.0], [0.0]]),
        np.eye(2),
        np.eye(2) * 0.1,
    )
    _z_pred, innovation = evaluate_measurement_oracle(
        predicted["x"],
        np.array([1.2]),
        np.array([[1.0, 0.0]]),
    )
    posterior = update_posterior_state_and_covariance(
        predicted,
        np.array([1.2]),
        np.array([[0.2]]),
        np.array([[1.0, 0.0]]),
        innovation,
    )
    assert np.isfinite(posterior["x"]).all()
    assert np.isfinite(posterior["P"]).all()


def test_steady_state_predict_and_update_preserve_covariance() -> None:
    state = initialize_kalman_state_model({"x": [2.0, -1.0], "P": np.eye(2) * 3.0})
    predicted = predict_latent_state_steady_state(
        state,
        np.array([0.5]),
        np.array([[2.0], [1.0]]),
    )
    posterior = update_posterior_state_steady_state(
        {**predicted, "K": np.array([[0.1], [0.3]])},
        np.array([0.0]),
        np.array([1.0]),
    )
    assert np.allclose(predicted["P"], state["P"])
    assert np.allclose(posterior["x"], np.array([3.1, -0.2]))
    assert np.allclose(posterior["P"], state["P"])


def test_legacy_aliases_resolve_to_snake_case_exports() -> None:
    assert filter_rs.initializekalmanstatemodel is initialize_kalman_state_model
    assert filter_rs.predictlatentstateandcovariance is predict_latent_state_and_covariance
    assert filter_rs.predictlatentstatesteadystate is predict_latent_state_steady_state
    assert filter_rs.evaluatemeasurementoracle is evaluate_measurement_oracle
    assert filter_rs.updateposteriorstateandcovariance is update_posterior_state_and_covariance
    assert filter_rs.updateposteriorstatesteadystate is update_posterior_state_steady_state

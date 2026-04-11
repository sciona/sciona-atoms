from __future__ import annotations

import numpy as np

from sciona.atoms.state_estimation.kalman_filters.filter_rs import (
    evaluatemeasurementoracle,
    initializekalmanstatemodel,
    predictlatentstateandcovariance,
    updateposteriorstateandcovariance,
)


def test_initializekalmanstatemodel_returns_state_dict() -> None:
    state = initializekalmanstatemodel({"initial_state": [0.0, 1.0]})
    assert set(state) == {"x", "P"}
    assert state["x"].shape == (2,)
    assert state["P"].shape == (2, 2)


def test_evaluatemeasurementoracle_returns_prediction_and_innovation() -> None:
    z_pred, innovation = evaluatemeasurementoracle(
        np.array([1.0, 2.0]),
        np.array([1.5]),
        np.array([[1.0, 0.0]]),
    )
    assert np.allclose(z_pred, np.array([1.0]))
    assert np.allclose(innovation, np.array([0.5]))


def test_predict_and_update_covariance_paths_return_finite_state() -> None:
    state = initializekalmanstatemodel({"x": [0.0, 0.0], "P": np.eye(2)})
    predicted = predictlatentstateandcovariance(
        state,
        np.array([1.0]),
        np.array([[1.0], [0.0]]),
        np.eye(2),
        np.eye(2) * 0.1,
    )
    _z_pred, innovation = evaluatemeasurementoracle(
        predicted["x"],
        np.array([1.2]),
        np.array([[1.0, 0.0]]),
    )
    posterior = updateposteriorstateandcovariance(
        predicted,
        np.array([1.2]),
        np.array([[0.2]]),
        np.array([[1.0, 0.0]]),
        innovation,
    )
    assert np.isfinite(posterior["x"]).all()
    assert np.isfinite(posterior["P"]).all()

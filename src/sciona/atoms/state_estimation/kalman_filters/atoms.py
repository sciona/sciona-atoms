"""Contract-level Kalman tracking atoms."""

from __future__ import annotations

import icontract
import numpy as np
from sciona.ghost.abstract import AbstractArray, AbstractScalar
from sciona.ghost.registry import register_atom

from .filter_rs import (
    evaluate_measurement_oracle,
    initialize_kalman_state_model,
    predict_latent_state_and_covariance,
    update_posterior_state_and_covariance,
)


def witness_track_linear_gaussian_state(
    observations: AbstractArray,
    process_noise: AbstractScalar,
    observation_noise: AbstractScalar,
) -> tuple[AbstractArray, AbstractArray]:
    """Describe posterior means and variances for a linear-Gaussian tracking run."""
    length = observations.shape[0] if observations.shape else 0
    posterior_means = AbstractArray(shape=(length,), dtype="float64")
    posterior_covariances = AbstractArray(shape=(length,), dtype="float64")
    return posterior_means, posterior_covariances


@register_atom(witness_track_linear_gaussian_state)
@icontract.require(lambda observations: isinstance(observations, np.ndarray), "observations must be an ndarray")
@icontract.require(
    lambda process_noise: isinstance(process_noise, (float, int, np.number)) and float(process_noise) > 0.0,
    "process_noise must be positive",
)
@icontract.require(
    lambda observation_noise: isinstance(observation_noise, (float, int, np.number)) and float(observation_noise) > 0.0,
    "observation_noise must be positive",
)
@icontract.ensure(lambda result: result is not None, "track_linear_gaussian_state output must not be None")
def track_linear_gaussian_state(
    observations: np.ndarray,
    *,
    process_noise: float = 0.05,
    observation_noise: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Track a scalar latent state under a linear-Gaussian observation model."""
    obs = np.asarray(observations, dtype=float).reshape(-1)
    if obs.size == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    state = initialize_kalman_state_model({"initial_state": [0.0], "initial_covariance": [[1.0]]})
    means: list[float] = []
    covariances: list[float] = []
    for measurement in obs:
        predicted = predict_latent_state_and_covariance(
            state,
            u=np.zeros(1, dtype=float),
            B=np.zeros((1, 1), dtype=float),
            F=np.eye(1, dtype=float),
            Q=np.eye(1, dtype=float) * float(process_noise),
        )
        innovation = evaluate_measurement_oracle(
            predicted["x"],
            np.array([measurement], dtype=float),
            np.eye(1, dtype=float),
        )[1]
        state = update_posterior_state_and_covariance(
            predicted,
            z=np.array([measurement], dtype=float),
            R=np.eye(1, dtype=float) * float(observation_noise),
            H=np.eye(1, dtype=float),
            innovation=innovation,
        )
        means.append(float(state["x"][0]))
        covariances.append(float(state["P"][0, 0]))
    return np.asarray(means, dtype=np.float64), np.asarray(covariances, dtype=np.float64)

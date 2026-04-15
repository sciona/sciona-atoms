from __future__ import annotations
"""Auto-generated atom wrappers following the sciona pattern."""


import numpy as np

import icontract
from sciona.ghost.registry import register_atom
from .witnesses import witness_exposecovariance, witness_exposelatentmean, witness_initializelineargaussianstatemodel, witness_predictlatentstate, witness_updatewithmeasurement

import ctypes
import ctypes.util
from pathlib import Path


@register_atom(witness_initializelineargaussianstatemodel)
@icontract.require(lambda initial_state: isinstance(initial_state, (float, int, np.number)), "initial_state must be numeric")
@icontract.require(lambda initial_covariance: isinstance(initial_covariance, (float, int, np.number)), "initial_covariance must be numeric")
@icontract.require(lambda transition_matrix: isinstance(transition_matrix, (float, int, np.number)), "transition_matrix must be numeric")
@icontract.require(lambda process_noise: isinstance(process_noise, (float, int, np.number)), "process_noise must be numeric")
@icontract.require(lambda observation_matrix: isinstance(observation_matrix, (float, int, np.number)), "observation_matrix must be numeric")
@icontract.require(lambda measurement_noise: isinstance(measurement_noise, (float, int, np.number)), "measurement_noise must be numeric")
@icontract.ensure(lambda result: result is not None, "InitializeLinearGaussianStateModel output must not be None")
def initializelineargaussianstatemodel(initial_state: np.ndarray | float, initial_covariance: np.ndarray | float, transition_matrix: np.ndarray | float, process_noise: np.ndarray | float, observation_matrix: np.ndarray | float, measurement_noise: np.ndarray | float) -> dict[str, np.ndarray]:
    """Create the immutable Kalman state-space model with latent mean and covariance plus fixed system/noise matrices.

    Args:
        initial_state: Dimension n
        initial_covariance: Shape n x n; symmetric positive semi-definite
        transition_matrix: Shape n x n
        process_noise: Shape n x n; symmetric positive semi-definite
        observation_matrix: Shape m x n
        measurement_noise: Shape m x m; symmetric positive semi-definite

    Returns:
        Immutable object; no hidden mutation
    """
    x = np.atleast_1d(np.asarray(initial_state, dtype=float))
    P = np.atleast_2d(np.asarray(initial_covariance, dtype=float))
    F = np.atleast_2d(np.asarray(transition_matrix, dtype=float))
    Q = np.atleast_2d(np.asarray(process_noise, dtype=float))
    H = np.atleast_2d(np.asarray(observation_matrix, dtype=float))
    R = np.atleast_2d(np.asarray(measurement_noise, dtype=float))
    return {"x": x, "P": P, "F": F, "Q": Q, "H": H, "R": R}


@register_atom(witness_predictlatentstate)
@icontract.require(lambda state_model: state_model is not None, "state_model cannot be None")
@icontract.ensure(lambda result: result is not None, "PredictLatentState output must not be None")
def predictlatentstate(state_model: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Apply the Kalman predict transition kernel to propagate latent mean/covariance forward in time.

    Args:
        state_model: Immutable prior/posterior from previous step

    Returns:
        New object with updated x and P only
    """
    sm = dict(state_model)
    F = sm["F"]
    Q = sm["Q"]
    x = sm["x"]
    P = sm["P"]
    x_pred = F @ x
    P_pred = F @ P @ F.T + Q
    return {**sm, "x": x_pred, "P": P_pred}


@register_atom(witness_updatewithmeasurement)
@icontract.require(lambda measurement: isinstance(measurement, (float, int, np.number)), "measurement must be numeric")
@icontract.ensure(lambda result: result is not None, "UpdateWithMeasurement output must not be None")
def updatewithmeasurement(predicted_state_model: dict[str, np.ndarray], measurement: np.ndarray | float) -> dict[str, np.ndarray]:
    """Apply the Kalman update kernel to incorporate a measurement and produce posterior latent mean/covariance.

    Args:
        predicted_state_model: Output of predict kernel
        measurement: Dimension m; compatible with H and R

    Returns:
        New object; analytical Bayesian posterior update
    """
    sm = dict(predicted_state_model)
    x = sm["x"]
    P = sm["P"]
    H = sm["H"]
    R = sm["R"]
    z = np.atleast_1d(np.asarray(measurement, dtype=float))
    innovation = z - H @ x
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    x_post = x + K @ innovation
    n = len(x)
    P_post = (np.eye(n) - K @ H) @ P
    return {**sm, "x": x_post, "P": P_post}


@register_atom(witness_exposelatentmean)
@icontract.require(lambda current_state_model: current_state_model is not None, "current_state_model cannot be None")
@icontract.ensure(lambda result: result is not None, "ExposeLatentMean output must not be None")
def exposelatentmean(current_state_model: dict[str, np.ndarray]) -> np.ndarray:
    """Read out the current latent state mean estimate from immutable filter state.

    Args:
        current_state_model: Can be initialized, predicted, or updated state

    Returns:
        Dimension n
    """
    return current_state_model["x"]


@register_atom(witness_exposecovariance)
@icontract.require(lambda current_state_model: current_state_model is not None, "current_state_model cannot be None")
@icontract.ensure(lambda result: result is not None, "ExposeCovariance output must not be None")
def exposecovariance(current_state_model: dict[str, np.ndarray]) -> np.ndarray:
    """Read out the current latent covariance estimate from immutable filter state.

    Args:
        current_state_model: Can be initialized, predicted, or updated state

    Returns:
        Shape n x n; symmetric positive semi-definite
    """
    return current_state_model["P"]


def _initializelineargaussianstatemodel_ffi(initial_state: object, initial_covariance: object, transition_matrix: object, process_noise: object, observation_matrix: object, measurement_noise: object) -> object:
    raise NotImplementedError("FFI bridge not wired")


def _predictlatentstate_ffi(state_model: object) -> object:
    raise NotImplementedError("FFI bridge not wired")


def _updatewithmeasurement_ffi(predicted_state_model: object, measurement: object) -> object:
    raise NotImplementedError("FFI bridge not wired")


def _exposelatentmean_ffi(current_state_model: object) -> object:
    raise NotImplementedError("FFI bridge not wired")


def _exposecovariance_ffi(current_state_model: object) -> object:
    raise NotImplementedError("FFI bridge not wired")


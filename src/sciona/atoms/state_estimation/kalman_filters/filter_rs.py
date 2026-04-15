"""Kalman filter reference wrappers for the sequential-filter namespace slice."""

from __future__ import annotations

import icontract
import numpy as np


StateDict = dict[str, np.ndarray]


@icontract.require(lambda init_config: init_config is not None, "init_config cannot be None")
@icontract.ensure(lambda result: result is not None, "initialize_kalman_state_model output must not be None")
def initialize_kalman_state_model(
    init_config: dict[str, np.ndarray | list[float]],
) -> StateDict:
    """Create the initial read-only latent state and covariance."""
    x = np.asarray(
        init_config.get("x", init_config.get("initial_state", [0.0])),
        dtype=float,
    )
    P = np.asarray(
        init_config.get("P", init_config.get("initial_covariance", np.eye(len(x)))),
        dtype=float,
    )
    return {"x": x, "P": P}


@icontract.require(lambda state_in: state_in is not None, "state_in cannot be None")
@icontract.require(lambda u: u is not None, "u cannot be None")
@icontract.require(lambda B: B is not None, "B cannot be None")
@icontract.require(lambda F: F is not None, "F cannot be None")
@icontract.require(lambda Q: Q is not None, "Q cannot be None")
@icontract.ensure(lambda result: result is not None, "predict_latent_state_and_covariance output must not be None")
def predict_latent_state_and_covariance(
    state_in: StateDict,
    u: np.ndarray,
    B: np.ndarray,
    F: np.ndarray,
    Q: np.ndarray,
) -> StateDict:
    """Propagate latent mean and covariance through a linear dynamics step."""
    x = np.asarray(state_in["x"], dtype=float)
    P = np.asarray(state_in["P"], dtype=float)
    F_mat = np.asarray(F, dtype=float)
    B_mat = np.asarray(B, dtype=float)
    Q_mat = np.asarray(Q, dtype=float)
    u_arr = np.asarray(u, dtype=float)
    x_pred = F_mat @ x + B_mat @ u_arr
    P_pred = F_mat @ P @ F_mat.T + Q_mat
    return {"x": x_pred, "P": P_pred}


@icontract.require(lambda state_in: state_in is not None, "state_in cannot be None")
@icontract.require(lambda u: u is not None, "u cannot be None")
@icontract.require(lambda B: B is not None, "B cannot be None")
@icontract.ensure(lambda result: result is not None, "predict_latent_state_steady_state output must not be None")
def predict_latent_state_steady_state(
    state_in: StateDict,
    u: np.ndarray,
    B: np.ndarray,
) -> StateDict:
    """Propagate latent state when covariance is assumed fixed."""
    x = np.asarray(state_in["x"], dtype=float)
    P = np.asarray(state_in["P"], dtype=float)
    B_mat = np.asarray(B, dtype=float)
    u_arr = np.asarray(u, dtype=float)
    x_pred = x + B_mat @ u_arr
    return {"x": x_pred, "P": P}


@icontract.require(lambda x: x is not None, "x cannot be None")
@icontract.require(lambda z: z is not None, "z cannot be None")
@icontract.require(lambda H: H is not None, "H cannot be None")
@icontract.ensure(lambda result: all(item is not None for item in result), "evaluate_measurement_oracle all outputs must not be None")
def evaluate_measurement_oracle(
    x: np.ndarray,
    z: np.ndarray,
    H: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project latent state into measurement space and compute innovation."""
    x_arr = np.asarray(x, dtype=float)
    z_arr = np.asarray(z, dtype=float)
    H_mat = np.asarray(H, dtype=float)
    z_pred = H_mat @ x_arr
    innovation = z_arr - z_pred
    return z_pred, innovation


@icontract.require(lambda predicted_state: predicted_state is not None, "predicted_state cannot be None")
@icontract.require(lambda z: z is not None, "z cannot be None")
@icontract.require(lambda R: R is not None, "R cannot be None")
@icontract.require(lambda H: H is not None, "H cannot be None")
@icontract.require(lambda innovation: innovation is not None, "innovation cannot be None")
@icontract.ensure(lambda result: result is not None, "update_posterior_state_and_covariance output must not be None")
def update_posterior_state_and_covariance(
    predicted_state: StateDict,
    z: np.ndarray,
    R: np.ndarray,
    H: np.ndarray,
    innovation: np.ndarray,
) -> StateDict:
    """Fuse measurement innovation into the predicted state."""
    x = np.asarray(predicted_state["x"], dtype=float)
    P = np.asarray(predicted_state["P"], dtype=float)
    H_mat = np.asarray(H, dtype=float)
    R_mat = np.asarray(R, dtype=float)
    inn = np.asarray(innovation, dtype=float)
    S = H_mat @ P @ H_mat.T + R_mat
    K = P @ H_mat.T @ np.linalg.inv(S)
    x_post = x + K @ inn
    P_post = (np.eye(len(x)) - K @ H_mat) @ P
    return {"x": x_post, "P": P_post}


@icontract.require(lambda predicted_state_steady: predicted_state_steady is not None, "predicted_state_steady cannot be None")
@icontract.require(lambda z: z is not None, "z cannot be None")
@icontract.require(lambda innovation: innovation is not None, "innovation cannot be None")
@icontract.ensure(lambda result: result is not None, "update_posterior_state_steady_state output must not be None")
def update_posterior_state_steady_state(
    predicted_state_steady: dict[str, np.ndarray | np.ndarray],
    z: np.ndarray,
    innovation: np.ndarray,
) -> StateDict:
    """Fuse innovation under a steady-state gain assumption."""
    x = np.asarray(predicted_state_steady["x"], dtype=float)
    P = np.asarray(predicted_state_steady["P"], dtype=float)
    inn = np.asarray(innovation, dtype=float)
    K = predicted_state_steady.get("K")
    if K is not None:
        x_post = x + np.asarray(K, dtype=float) @ inn
    else:
        x_post = x + inn
    return {"x": x_post, "P": P}


# Backward-compatible aliases while matcher metadata and legacy assets still reference
# the collapsed wrapper symbols from the original package layout.
initializekalmanstatemodel = initialize_kalman_state_model
predictlatentstateandcovariance = predict_latent_state_and_covariance
predictlatentstatesteadystate = predict_latent_state_steady_state
evaluatemeasurementoracle = evaluate_measurement_oracle
updateposteriorstateandcovariance = update_posterior_state_and_covariance
updateposteriorstatesteadystate = update_posterior_state_steady_state


__all__ = [
    "StateDict",
    "initialize_kalman_state_model",
    "predict_latent_state_and_covariance",
    "predict_latent_state_steady_state",
    "evaluate_measurement_oracle",
    "update_posterior_state_and_covariance",
    "update_posterior_state_steady_state",
    "initializekalmanstatemodel",
    "predictlatentstateandcovariance",
    "predictlatentstatesteadystate",
    "evaluatemeasurementoracle",
    "updateposteriorstateandcovariance",
    "updateposteriorstatesteadystate",
]

from __future__ import annotations

from typing import Callable, TypeAlias

import icontract
import numpy as np

from sciona.ghost.registry import register_atom

from .core_witnesses import (
    witness_evaluate_log_probability_density,
    witness_gradient_oracle_evaluation,
    witness_optimizationlooporchestration,
)

ObjectiveFn: TypeAlias = Callable[[np.ndarray], float]
RestructureFn: TypeAlias = Callable[[np.ndarray], np.ndarray]
RngState: TypeAlias = int | np.ndarray
OptimizationStep: TypeAlias = Callable[[np.ndarray, ObjectiveFn, RngState], tuple[np.ndarray, RngState]]


@register_atom(witness_evaluate_log_probability_density)
@icontract.require(lambda q: q is not None, "q cannot be None")
@icontract.require(lambda z: z is not None, "z cannot be None")
@icontract.ensure(lambda result: result is not None, "evaluate_log_probability_density output must not be None")
def evaluate_log_probability_density(q: np.ndarray, z: np.ndarray) -> float:
    """Compute a Gaussian log density from location-scale parameters.

    Args:
        q: Concatenated parameter vector `[mu, log_sigma]`.
        z: Sample vector to score under the diagonal Gaussian.

    Returns:
        Scalar log-density value.
    """
    q_arr = np.asarray(q, dtype=np.float64).ravel()
    z_arr = np.asarray(z, dtype=np.float64).ravel()
    if q_arr.size == 0 or q_arr.size % 2 != 0:
        raise ValueError("q must contain [mu, log_sigma] with equal non-empty halves")
    d = q_arr.size // 2
    if z_arr.size != d:
        raise ValueError("z dimensionality must match the variational mean vector")
    mu = q_arr[:d]
    log_sigma = q_arr[d:]
    sigma = np.exp(log_sigma)
    return float(-0.5 * d * np.log(2 * np.pi) - np.sum(log_sigma) - 0.5 * np.sum(((z_arr - mu) / sigma) ** 2))


@register_atom(witness_optimizationlooporchestration)
@icontract.require(lambda max_iter: max_iter > 0, "max_iter must be positive")
@icontract.require(lambda prob: callable(prob), "prob must be callable")
@icontract.require(lambda q_init: q_init is not None, "q_init cannot be None")
@icontract.require(lambda rng_state_in: rng_state_in is not None, "rng_state_in cannot be None")
@icontract.ensure(lambda result: all(r is not None for r in result), "OptimizationLoopOrchestration all outputs must not be None")
def optimizationlooporchestration(
    algorithm: OptimizationStep | None,
    max_iter: int,
    prob: ObjectiveFn,
    q_init: np.ndarray,
    rng_state_in: RngState,
) -> tuple[np.ndarray, RngState, np.ndarray]:
    """Run a bounded variational optimization loop.

    Args:
        algorithm: Optional step function implementing one optimizer update.
        max_iter: Positive iteration budget.
        prob: Objective function over the variational parameter vector.
        q_init: Initial parameter vector.
        rng_state_in: Explicit random state token threaded through the loop.

    Returns:
        Optimized parameters, updated RNG state, and the final parameter vector.
    """
    from scipy.optimize import minimize as scipy_minimize

    q = np.asarray(q_init, dtype=np.float64).copy()
    rng_state = rng_state_in

    if algorithm is not None:
        for _ in range(max_iter):
            q, rng_state = algorithm(q, prob, rng_state)
    else:
        result = scipy_minimize(prob, q, method="L-BFGS-B", options={"maxiter": max_iter})
        q = np.asarray(result.x, dtype=np.float64)

    return (q, rng_state, q)


@register_atom(witness_gradient_oracle_evaluation)
@icontract.require(lambda obj: callable(obj), "obj must be callable")
@icontract.require(lambda params: params is not None, "params cannot be None")
@icontract.require(lambda out_in: out_in is not None, "out_in cannot be None")
@icontract.require(lambda state_in: state_in is not None, "state_in cannot be None")
@icontract.ensure(lambda result: all(r is not None for r in result), "Gradient Oracle Evaluation all outputs must not be None")
def gradient_oracle_evaluation(
    rng_in: RngState,
    obj: ObjectiveFn,
    adtype: str,
    out_in: np.ndarray,
    state_in: np.ndarray,
    params: np.ndarray,
    restructure: RestructureFn | None,
) -> tuple[np.ndarray, float, np.ndarray, RngState]:
    """Estimate an objective value and gradient with explicit state threading.

    Args:
        rng_in: Explicit stochastic state token.
        obj: Objective function to evaluate.
        adtype: Name of the differentiation backend or strategy.
        out_in: Gradient buffer prototype.
        state_in: Explicit algorithm state threaded through the call.
        params: Parameter vector to evaluate.
        restructure: Optional function to reshape the raw gradient.

    Returns:
        Gradient output, scalar objective value, unchanged state token, and RNG.
    """
    del adtype
    del out_in

    params_arr = np.asarray(params, dtype=np.float64)
    eps = 1e-5
    grad = np.zeros_like(params_arr)
    value = float(obj(params_arr))

    for i in np.ndindex(params_arr.shape):
        params_plus = params_arr.copy()
        params_minus = params_arr.copy()
        params_plus[i] += eps
        params_minus[i] -= eps
        grad[i] = (obj(params_plus) - obj(params_minus)) / (2 * eps)

    out_out = restructure(grad) if restructure is not None else grad
    return (np.asarray(out_out, dtype=np.float64), value, np.asarray(state_in, dtype=np.float64), rng_in)

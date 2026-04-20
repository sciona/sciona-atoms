from __future__ import annotations

from typing import Any, Callable

import icontract
import numpy as np
from sciona.ghost.registry import register_atom

from .rmhmc_witnesses import witness_buildrmhmctransitionkernel


_DEFAULT_FINITE_DIFF_EPS = 1e-5


def _seed_from_key(rng_key: np.ndarray) -> int:
    key = np.asarray(rng_key, dtype=np.int64).ravel()
    if key.size == 0:
        raise ValueError("rng must contain at least one integer seed value")
    return int(np.abs(key).sum()) % (2**31 - 1)


def _logp_value(value: Any) -> float:
    if isinstance(value, tuple) and len(value) == 2:
        value = value[0]
    return float(value)


def _target_value_and_grad(
    target_log_kernel: Callable[[np.ndarray], Any],
    position: np.ndarray,
    finite_diff_eps: float,
) -> tuple[float, np.ndarray]:
    evaluated = target_log_kernel(np.array(position, copy=True))
    if isinstance(evaluated, tuple) and len(evaluated) == 2:
        logp, grad = evaluated
        grad_arr = np.asarray(grad, dtype=np.float64).reshape(-1)
        if grad_arr.shape != position.shape:
            raise ValueError("target_log_kernel gradient must have shape (n_dimensions,)")
        return float(logp), grad_arr

    logp = float(evaluated)
    grad = np.zeros_like(position, dtype=np.float64)
    for index in range(position.size):
        plus = position.copy()
        minus = position.copy()
        plus[index] += finite_diff_eps
        minus[index] -= finite_diff_eps
        grad[index] = (_logp_value(target_log_kernel(plus)) - _logp_value(target_log_kernel(minus))) / (
            2.0 * finite_diff_eps
        )
    return logp, grad


def _validate_metric(metric: np.ndarray, dim: int) -> np.ndarray:
    metric_arr = np.asarray(metric, dtype=np.float64)
    if metric_arr.shape != (dim, dim):
        raise ValueError("tensor_fn metric must have shape (n_dimensions, n_dimensions)")
    if not np.all(np.isfinite(metric_arr)):
        raise ValueError("tensor_fn metric must be finite")
    return metric_arr


def _metric_value(tensor_fn: Callable[[np.ndarray], Any], position: np.ndarray) -> np.ndarray:
    evaluated = tensor_fn(np.array(position, copy=True))
    if isinstance(evaluated, tuple) and len(evaluated) == 2:
        evaluated = evaluated[0]
    return _validate_metric(evaluated, position.size)


def _metric_and_derivative(
    tensor_fn: Callable[[np.ndarray], Any],
    position: np.ndarray,
    finite_diff_eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    evaluated = tensor_fn(np.array(position, copy=True))
    if isinstance(evaluated, tuple) and len(evaluated) == 2:
        metric, derivative = evaluated
        metric_arr = _validate_metric(metric, position.size)
        derivative_arr = np.asarray(derivative, dtype=np.float64)
        expected_shape = (position.size, position.size, position.size)
        if derivative_arr.shape != expected_shape:
            raise ValueError("tensor_fn derivative must have shape (n_dimensions, n_dimensions, n_dimensions)")
        if not np.all(np.isfinite(derivative_arr)):
            raise ValueError("tensor_fn derivative must be finite")
        return metric_arr, derivative_arr

    metric_arr = _validate_metric(evaluated, position.size)
    derivative_arr = np.zeros((position.size, position.size, position.size), dtype=np.float64)
    for index in range(position.size):
        plus = position.copy()
        minus = position.copy()
        plus[index] += finite_diff_eps
        minus[index] -= finite_diff_eps
        derivative_arr[index] = (_metric_value(tensor_fn, plus) - _metric_value(tensor_fn, minus)) / (
            2.0 * finite_diff_eps
        )
    return metric_arr, derivative_arr


def _metric_inverse_and_logdet(metric: np.ndarray) -> tuple[np.ndarray, float]:
    sign, logdet = np.linalg.slogdet(metric)
    if sign <= 0 or not np.isfinite(logdet):
        raise ValueError("tensor_fn metric must be positive definite")
    return np.linalg.inv(metric), float(logdet)


def _hamiltonian(
    position: np.ndarray,
    momentum: np.ndarray,
    target_log_kernel: Callable[[np.ndarray], Any],
    metric: np.ndarray,
) -> float:
    logp = _logp_value(target_log_kernel(np.array(position, copy=True)))
    inv_metric, logdet = _metric_inverse_and_logdet(metric)
    kinetic = 0.5 * float(momentum @ inv_metric @ momentum)
    return -logp + 0.5 * logdet + kinetic


def _position_hamiltonian_grad(
    position: np.ndarray,
    momentum: np.ndarray,
    target_log_kernel: Callable[[np.ndarray], Any],
    inv_metric: np.ndarray,
    metric_derivative: np.ndarray,
    finite_diff_eps: float,
) -> np.ndarray:
    _, logp_grad = _target_value_and_grad(target_log_kernel, position, finite_diff_eps)
    inv_metric_momentum = inv_metric @ momentum
    grad = -logp_grad.copy()

    for index in range(position.size):
        derivative_slice = metric_derivative[index]
        trace_term = float(np.trace(inv_metric @ derivative_slice))
        quadratic_term = float(inv_metric_momentum @ derivative_slice @ inv_metric_momentum)
        grad[index] += 0.5 * (trace_term - quadratic_term)

    return grad


@register_atom(witness_buildrmhmctransitionkernel)  # type: ignore[untyped-decorator,name-defined]
@icontract.require(lambda target_log_kernel: callable(target_log_kernel), "target_log_kernel must be callable")
@icontract.require(lambda tensor_fn: callable(tensor_fn), "tensor_fn must be callable")
@icontract.require(lambda step_size: float(step_size) > 0.0, "step_size must be positive")
@icontract.require(lambda n_leapfrog: int(n_leapfrog) >= 1, "n_leapfrog must be at least one")
@icontract.require(lambda n_fixed_point: int(n_fixed_point) >= 1, "n_fixed_point must be at least one")
@icontract.require(lambda finite_diff_eps: float(finite_diff_eps) > 0.0, "finite_diff_eps must be positive")
@icontract.ensure(lambda result: callable(result), "BuildRMHMCTransitionKernel output must be a transition kernel")
def buildrmhmctransitionkernel(
    target_log_kernel: Callable[[np.ndarray], Any],
    tensor_fn: Callable[[np.ndarray], Any],
    step_size: float = 0.1,
    n_leapfrog: int = 10,
    n_fixed_point: int = 5,
    finite_diff_eps: float = _DEFAULT_FINITE_DIFF_EPS,
) -> Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Build a NumPy RMHMC transition kernel using generalized leapfrog steps.

    This is a source-shaped educational adaptation of the unbounded
    ``mcmc::rmhmc`` transition: the returned callable performs one
    Metropolis-corrected RMHMC move with explicit ``state`` and ``rng`` inputs.
    The upstream C++ API requests gradients and metric derivatives through
    output pointers; this Python version accepts either scalar/log-density and
    metric-only callbacks, or richer callbacks returning ``(logp, grad_logp)``
    and ``(metric, metric_derivative)``. Metric derivatives are ordered so that
    ``metric_derivative[i]`` is ``d metric / d state[i]``. Finite differences
    are used only when a callback does not provide the derivative explicitly.
    """
    step = float(step_size)
    leapfrog_steps = int(n_leapfrog)
    fixed_point_steps = int(n_fixed_point)
    diff_eps = float(finite_diff_eps)

    def _rmhmc_kernel(state: np.ndarray, rng: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        state_arr = np.asarray(state, dtype=np.float64)
        state_shape = state_arr.shape
        current = state_arr.reshape(-1)
        if current.size == 0:
            raise ValueError("state must contain at least one dimension")

        rng_arr = np.asarray(rng)
        local_rng = np.random.RandomState(_seed_from_key(rng_arr))

        current_metric, current_metric_derivative = _metric_and_derivative(tensor_fn, current, diff_eps)
        current_inv_metric, _ = _metric_inverse_and_logdet(current_metric)
        initial_momentum = np.linalg.cholesky(current_metric) @ local_rng.normal(size=current.size)

        position = current.copy()
        momentum = initial_momentum.copy()
        metric = current_metric
        metric_derivative = current_metric_derivative
        inv_metric = current_inv_metric

        for _ in range(leapfrog_steps):
            fixed_momentum = momentum.copy()
            for _ in range(fixed_point_steps):
                grad_h = _position_hamiltonian_grad(
                    position,
                    fixed_momentum,
                    target_log_kernel,
                    inv_metric,
                    metric_derivative,
                    diff_eps,
                )
                fixed_momentum = momentum - 0.5 * step * grad_h
            momentum = fixed_momentum

            fixed_position = position.copy()
            old_velocity = inv_metric @ momentum
            for _ in range(fixed_point_steps):
                proposal_metric = _metric_value(tensor_fn, fixed_position)
                proposal_inv_metric, _ = _metric_inverse_and_logdet(proposal_metric)
                fixed_position = position + 0.5 * step * (old_velocity + proposal_inv_metric @ momentum)
            position = fixed_position

            metric, metric_derivative = _metric_and_derivative(tensor_fn, position, diff_eps)
            inv_metric, _ = _metric_inverse_and_logdet(metric)
            grad_h = _position_hamiltonian_grad(
                position,
                momentum,
                target_log_kernel,
                inv_metric,
                metric_derivative,
                diff_eps,
            )
            momentum = momentum - 0.5 * step * grad_h

        proposed_momentum = -momentum
        current_h = _hamiltonian(current, initial_momentum, target_log_kernel, current_metric)
        proposed_h = _hamiltonian(position, proposed_momentum, target_log_kernel, metric)
        if not np.isfinite(proposed_h):
            proposed_h = np.inf

        log_accept = min(0.0, current_h - proposed_h)
        if np.log(local_rng.uniform()) < log_accept:
            state_out = position.reshape(state_shape)
        else:
            state_out = current.reshape(state_shape).copy()

        rng_dtype = rng_arr.dtype if np.issubdtype(rng_arr.dtype, np.integer) else np.int64
        rng_out = np.asarray(local_rng.randint(0, 2**31 - 1, size=rng_arr.shape), dtype=rng_dtype)
        return state_out, rng_out

    return _rmhmc_kernel

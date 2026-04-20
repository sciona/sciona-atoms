from __future__ import annotations

from typing import Callable

import icontract
import numpy as np
from sciona.ghost.registry import register_atom

from .hmc_witnesses import witness_buildhmckernelfromlogdensityoracle


def _seed_from_key(rng_key: np.ndarray) -> int:
    key = np.asarray(rng_key, dtype=np.int64).ravel()
    if key.size == 0:
        raise ValueError("rng must contain at least one integer seed value")
    return int(np.abs(key).sum()) % (2**31 - 1)


def _finite_difference_grad(target_log_kernel: Callable[[np.ndarray], float], position: np.ndarray) -> np.ndarray:
    eps = 1e-5
    grad = np.zeros_like(position, dtype=np.float64)
    for index in np.ndindex(position.shape):
        pos_plus = position.copy()
        pos_minus = position.copy()
        pos_plus[index] += eps
        pos_minus[index] -= eps
        grad[index] = (float(target_log_kernel(pos_plus)) - float(target_log_kernel(pos_minus))) / (2.0 * eps)
    return grad


@register_atom(witness_buildhmckernelfromlogdensityoracle)
@icontract.require(lambda target_log_kernel: callable(target_log_kernel), "target_log_kernel must be callable")
@icontract.require(lambda step_size: float(step_size) > 0.0, "step_size must be positive")
@icontract.require(lambda n_leapfrog: int(n_leapfrog) >= 1, "n_leapfrog must be at least one")
@icontract.ensure(lambda result: callable(result), "result must be a transition kernel")
def buildhmckernelfromlogdensityoracle(
    target_log_kernel: Callable[[np.ndarray], float],
    step_size: float = 0.1,
    n_leapfrog: int = 10,
) -> Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Build a small identity-mass Hamiltonian Monte Carlo transition kernel.

    This educational NumPy kernel mirrors the core HMC move from the source
    library but uses finite-difference gradients from a scalar log-density
    oracle instead of binding to the upstream C++ gradient callback interface.
    """

    def _hmc_kernel(state: np.ndarray, rng: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        current = np.asarray(state, dtype=np.float64)
        local_rng = np.random.RandomState(_seed_from_key(rng))
        initial_momentum = local_rng.normal(size=current.shape)
        position = current.copy()
        momentum = initial_momentum.copy()

        momentum += 0.5 * float(step_size) * _finite_difference_grad(target_log_kernel, position)
        for _ in range(int(n_leapfrog) - 1):
            position += float(step_size) * momentum
            momentum += float(step_size) * _finite_difference_grad(target_log_kernel, position)
        position += float(step_size) * momentum
        momentum += 0.5 * float(step_size) * _finite_difference_grad(target_log_kernel, position)
        momentum = -momentum

        current_h = -float(target_log_kernel(current)) + 0.5 * float(np.sum(initial_momentum**2))
        proposed_h = -float(target_log_kernel(position)) + 0.5 * float(np.sum(momentum**2))
        log_accept = min(0.0, -(proposed_h - current_h))
        if np.log(local_rng.uniform()) < log_accept:
            state_out = position
        else:
            state_out = current.copy()
        rng_out = np.asarray(local_rng.randint(0, 2**31 - 1, size=np.asarray(rng).shape), dtype=np.int64)
        return state_out, rng_out

    return _hmc_kernel

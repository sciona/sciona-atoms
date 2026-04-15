from __future__ import annotations

from typing import Callable
"""Auto-generated atom wrappers following the sciona pattern."""


import numpy as np

import icontract
from sciona.ghost.registry import register_atom
from .hmc_witnesses import witness_buildhmckernelfromlogdensityoracle

import ctypes
import ctypes.util
from pathlib import Path


@register_atom(witness_buildhmckernelfromlogdensityoracle)
@icontract.require(lambda target_log_kernel: callable(target_log_kernel), "target_log_kernel must be callable")
@icontract.ensure(lambda result: result is not None, "BuildHMCKernelFromLogDensityOracle output must not be None")
def buildhmckernelfromlogdensityoracle(target_log_kernel: Callable[[np.ndarray], float]) -> Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Creates a pure Hamiltonian Monte Carlo transition kernel from a provided target log-density oracle, with stochasticity and chain state threaded explicitly.

    Args:
        target_log_kernel: Stateless oracle; no persistent writes; deterministic for fixed input.

    Returns:
        Pure transition; consumes and returns new PRNGKey; HMCState is immutable state_in->state_out and may include position, momentum, mass_matrix, and trace diagnostics.
    """
    def _hmc_kernel(state: np.ndarray, rng: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rng_int = int(np.sum(np.abs(rng))) % (2**31)
        local_rng = np.random.RandomState(rng_int)
        dim = state.shape[0]
        step_size = 0.1
        n_leapfrog = 10

        momentum = local_rng.randn(dim)
        pos = state.copy()
        mom = momentum.copy()
        eps = 1e-5

        def _grad(x):
            g = np.zeros(dim)
            for i in range(dim):
                xp = x.copy(); xp[i] += eps
                xm = x.copy(); xm[i] -= eps
                g[i] = (target_log_kernel(xp) - target_log_kernel(xm)) / (2.0 * eps)
            return g

        mom = mom + 0.5 * step_size * _grad(pos)
        for _ in range(n_leapfrog - 1):
            pos = pos + step_size * mom
            mom = mom + step_size * _grad(pos)
        pos = pos + step_size * mom
        mom = mom + 0.5 * step_size * _grad(pos)
        mom = -mom

        current_H = -target_log_kernel(state) + 0.5 * np.dot(momentum, momentum)
        proposed_H = -target_log_kernel(pos) + 0.5 * np.dot(mom, mom)
        log_accept = -(proposed_H - current_H)

        if np.log(local_rng.rand()) < log_accept:
            new_state = pos
        else:
            new_state = state.copy()

        new_rng = np.array(local_rng.randint(0, 2**31, size=rng.shape), dtype=rng.dtype)
        return (new_state, new_rng)

    return _hmc_kernel


"""Auto-generated FFI bindings for cpp implementations."""

"""Auto-generated FFI bindings for cpp implementations."""

def _buildhmckernelfromlogdensityoracle_ffi(target_log_kernel: ctypes.c_void_p) -> ctypes.c_void_p:
    return target_log_kernel

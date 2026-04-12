from __future__ import annotations

from typing import Callable
"""Auto-generated atom wrappers following the ageoa pattern."""


import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from .rmhmc_witnesses import witness_buildrmhmctransitionkernel

import ctypes
import ctypes.util
from pathlib import Path


# Witness functions should be imported from the generated witnesses module

@register_atom(witness_buildrmhmctransitionkernel)  # type: ignore[untyped-decorator,name-defined]
@icontract.require(lambda target_log_kernel: target_log_kernel is not None, "target_log_kernel cannot be None")
@icontract.require(lambda tensor_fn: tensor_fn is not None, "tensor_fn cannot be None")
@icontract.ensure(lambda result: result is not None, "BuildRMHMCTransitionKernel output must not be None")
def buildrmhmctransitionkernel(target_log_kernel: Callable[[np.ndarray], float], tensor_fn: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Constructs a pure Riemannian Manifold Hamiltonian Monte Carlo (HMC) transition kernel from a target log-density oracle and metric/tensor oracle. The produced kernel is expected to thread immutable sampler state explicitly (e.g., position, momentum, mass/metric tensor, and PRNGKey) as state_in -> state_out.

Args:
    target_log_kernel: Pure oracle-style log-density/log-kernel evaluator; no persistent state mutation.
    tensor_fn: Pure oracle-style metric/tensor evaluator compatible with RMHMC geometry.

Returns:
    Pure Markov Chain Monte Carlo (MCMC) transition function that consumes explicit state (including PRNGKey) and returns a new state object."""
    def _rmhmc_kernel(state: np.ndarray, rng: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rng_int = int(np.sum(np.abs(rng))) % (2**31)
        local_rng = np.random.RandomState(rng_int)
        dim = state.shape[0]
        step_size = 0.1
        n_leapfrog = 10
        eps = 1e-5

        # Get metric tensor at current position
        G = tensor_fn(state)
        G_inv = np.linalg.inv(G)

        # Sample momentum from N(0, G)
        L = np.linalg.cholesky(G)
        momentum = L @ local_rng.randn(dim)

        pos = state.copy()
        mom = momentum.copy()

        def _grad(x):
            g = np.zeros(dim)
            for i in range(dim):
                xp = x.copy(); xp[i] += eps
                xm = x.copy(); xm[i] -= eps
                g[i] = (target_log_kernel(xp) - target_log_kernel(xm)) / (2.0 * eps)
            return g

        # Leapfrog with position-dependent mass matrix
        for _ in range(n_leapfrog):
            mom = mom + 0.5 * step_size * _grad(pos)
            G_curr = tensor_fn(pos)
            G_inv_curr = np.linalg.inv(G_curr)
            pos = pos + step_size * G_inv_curr @ mom
            mom = mom + 0.5 * step_size * _grad(pos)

        mom = -mom

        # MH acceptance
        G_prop = tensor_fn(pos)
        current_H = -target_log_kernel(state) + 0.5 * momentum @ np.linalg.inv(G) @ momentum + 0.5 * np.log(np.linalg.det(G))
        proposed_H = -target_log_kernel(pos) + 0.5 * mom @ np.linalg.inv(G_prop) @ mom + 0.5 * np.log(np.linalg.det(G_prop))
        log_accept = -(proposed_H - current_H)

        if np.log(local_rng.rand()) < log_accept:
            new_state = pos
        else:
            new_state = state.copy()

        new_rng = np.array(local_rng.randint(0, 2**31, size=rng.shape), dtype=rng.dtype)
        return (new_state, new_rng)

    return _rmhmc_kernel


"""Auto-generated FFI bindings for cpp implementations."""


import ctypes
import ctypes.util
from pathlib import Path


def _buildrmhmctransitionkernel_ffi(target_log_kernel: Callable[[np.ndarray], float], tensor_fn: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Wrapper that calls the C++ version of build rmhmc transition kernel. Passes arguments through and returns the result."""
    _lib = ctypes.CDLL("./buildrmhmctransitionkernel.so")
    _func_name = "buildrmhmctransitionkernel"
    _func = getattr(_lib, _func_name)
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(target_log_kernel, tensor_fn)

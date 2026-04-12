from __future__ import annotations

from typing import Callable, cast
"""Auto-generated atom wrappers following the ageoa pattern."""


import numpy as np

import icontract
from ageoa.ghost.registry import register_atom

import ctypes
import ctypes.util
from pathlib import Path

from .rwmh_witnesses import witness_constructrandomwalkmetropoliskernel

@register_atom(witness_constructrandomwalkmetropoliskernel)
@icontract.require(lambda target_log_kernel: target_log_kernel is not None, "target_log_kernel must not be None")
@icontract.ensure(lambda result: result is not None, "ConstructRandomWalkMetropolisKernel output must not be None")
def constructrandomwalkmetropoliskernel(target_log_kernel: Callable[[np.ndarray], float]) -> Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Builds a pure Random-Walk Metropolis-Hastings transition kernel from a target log-density oracle, with explicit immutable state and random number generator (RNG) threading.

Args:
    target_log_kernel: Stateless/pure log-density oracle; no persistent state mutation.

Returns:
    Pure Markov Chain Monte Carlo (MCMC) kernel; chain_state is immutable (e.g., latent sample and cached log_prob), PRNGKey must be explicitly split/threaded."""
    def _rwmh_kernel(state: np.ndarray, rng: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rng_int = int(np.sum(np.abs(rng))) % (2**31)
        local_rng = np.random.RandomState(rng_int)
        proposal = state + local_rng.randn(*state.shape)
        log_ratio = target_log_kernel(proposal) - target_log_kernel(state)
        if np.log(local_rng.rand()) < log_ratio:
            new_state = proposal
        else:
            new_state = state.copy()
        new_rng = np.array(local_rng.randint(0, 2**31, size=rng.shape), dtype=rng.dtype)
        return (new_state, new_rng)

    return _rwmh_kernel


"""Auto-generated FFI bindings for cpp implementations."""


import ctypes
import ctypes.util
from pathlib import Path


def _constructrandomwalkmetropoliskernel_ffi(target_log_kernel: ctypes.c_void_p) -> ctypes.c_void_p:
    """Wrapper that calls the C++ version of construct random walk metropolis kernel. Passes arguments through and returns the result."""
    _lib = ctypes.CDLL("./constructrandomwalkmetropoliskernel.so")
    _func_name = 'constructrandomwalkmetropoliskernel'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p]
    return cast(ctypes.c_void_p, _func(target_log_kernel))

from __future__ import annotations

from typing import Callable
"""Auto-generated atom wrappers following the ageoa pattern."""


import numpy as np

import icontract
from ageoa.ghost.registry import register_atom  # type: ignore[import-untyped]

import ctypes
import ctypes.util
from pathlib import Path


from .de_witnesses import witness_build_de_transition_kernel

@register_atom(witness_build_de_transition_kernel)
@icontract.require(lambda target_log_kernel: target_log_kernel is not None, "target_log_kernel cannot be None")
@icontract.ensure(lambda result: result is not None, "build_de_transition_kernel output must not be None")
def build_de_transition_kernel(target_log_kernel: Callable[[np.ndarray], float]) -> Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Creates a pure Differential Evolution transition kernel from the provided target log-density oracle.

Args:
    target_log_kernel: Stateless log-density oracle; no persistent state mutation.

Returns:
    Pure transition function; any stochastic state (e.g., random number generator (RNG)/PRNGKey) must be explicit input/output."""
    def _de_kernel(state: np.ndarray, rng: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # state is the population matrix: (n_members, dim)
        rng_int = int(np.sum(np.abs(rng))) % (2**31)
        local_rng = np.random.RandomState(rng_int)

        if state.ndim == 1:
            # Single member: just do RW-MH
            proposal = state + local_rng.randn(*state.shape)
            log_ratio = target_log_kernel(proposal) - target_log_kernel(state)
            if np.log(local_rng.rand()) < log_ratio:
                new_state = proposal
            else:
                new_state = state.copy()
        else:
            n_members = state.shape[0]
            new_state = state.copy()
            F = 2.38 / np.sqrt(2.0 * state.shape[1]) if state.shape[1] > 0 else 1.0
            for k in range(n_members):
                idxs = [j for j in range(n_members) if j != k]
                a, b = local_rng.choice(idxs, size=2, replace=False)
                proposal = state[k] + F * (state[a] - state[b])
                log_ratio = target_log_kernel(proposal) - target_log_kernel(state[k])
                if np.log(local_rng.rand()) < log_ratio:
                    new_state[k] = proposal

        new_rng = np.array(local_rng.randint(0, 2**31, size=rng.shape), dtype=rng.dtype)
        return (new_state, new_rng)

    return _de_kernel


"""Auto-generated FFI bindings for cpp implementations."""


import ctypes
import ctypes.util
from pathlib import Path

def _build_de_transition_kernel_ffi(target_log_kernel: Callable[[np.ndarray], float]) -> Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Wrapper that calls the C++ version of build de transition kernel. Passes arguments through and returns the result."""
    _func_name = 'build_de_transition_kernel'
    _func_name = 'build_de_transition_kernel'
    _func = ctypes.CDLL(None)[_func_name]
    _func.argtypes = [ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(target_log_kernel)

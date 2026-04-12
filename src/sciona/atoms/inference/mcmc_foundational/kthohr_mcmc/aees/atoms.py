from __future__ import annotations
from typing import Callable
"""Auto-generated atom wrappers following the ageoa pattern."""


import numpy as np

import icontract
from ageoa.ghost.registry import register_atom
from .witnesses import witness_metropolishastingstransitionkernel, witness_targetlogkerneloracle

import ctypes
import ctypes.util
from pathlib import Path


# Witness functions should be imported from the generated witnesses module

@register_atom(witness_metropolishastingstransitionkernel)  # type: ignore[untyped-decorator,name-defined]
@icontract.require(lambda temper_val: isinstance(temper_val, (float, int, np.number)), "temper_val must be numeric")
@icontract.ensure(lambda result: all(r is not None for r in result), "MetropolisHastingsTransitionKernel all outputs must not be None")
def metropolishastingstransitionkernel(temper_val: float, target_log_kernel: Callable[[np.ndarray], float], rng_key_in: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Runs one pure Metropolis-Hastings transition: builds proposal-related terms, consumes oracle log-kernel evaluations, computes acceptance, and returns a new sample/state object.

    Args:
        temper_val: Finite positive temperature/scaling value.
        target_log_kernel: Pure function handle used for target log-kernel evaluation.
        rng_key_in: Thread explicitly for purity when stochastic proposal/acceptance draws are used.

    Returns:
        mh_step_state_out: Immutable result containing next sample and acceptance-related values.
        rng_key_out: New key/state after any random draws.
    """
    rng_int = int(np.sum(np.abs(rng_key_in))) % (2**31)
    local_rng = np.random.RandomState(rng_int)
    dim = rng_key_in.shape[0] if rng_key_in.ndim > 0 else 1

    # Generate a proposal from a simple normal perturbation
    current_state = rng_key_in.copy()
    proposal = current_state + local_rng.randn(*current_state.shape)

    current_logp = temper_val * target_log_kernel(current_state)
    proposal_logp = temper_val * target_log_kernel(proposal)
    log_alpha = proposal_logp - current_logp

    if np.log(local_rng.rand()) < log_alpha:
        new_state = proposal
    else:
        new_state = current_state.copy()

    new_rng = np.array(local_rng.randint(0, 2**31, size=rng_key_in.shape), dtype=rng_key_in.dtype)
    return (new_state, new_rng)

@register_atom(witness_targetlogkerneloracle)  # type: ignore[untyped-decorator,name-defined]
@icontract.require(lambda temper_val: isinstance(temper_val, (float, int, np.number)), "temper_val must be numeric")
@icontract.ensure(lambda result: result is not None, "TargetLogKernelOracle output must not be None")
def targetlogkerneloracle(state_candidate: np.ndarray, temper_val: float) -> float:
    """Evaluates the target log-density for a candidate state in the Adaptive Equi-Energy Sampler (AEES). Used by the Metropolis-Hastings acceptance step to decide whether to accept or reject the proposal.

    Args:
        state_candidate: Candidate (or current) state point to score.
        temper_val: Temperature scaling value used by the sampler.

    Returns:
        Finite log-density value.
    """
    return float(temper_val * np.sum(state_candidate))


"""Auto-generated FFI bindings for cpp implementations."""


import ctypes
import ctypes.util
from pathlib import Path


def _metropolishastingstransitionkernel_ffi(temper_val: float, target_log_kernel: Callable[[np.ndarray], float], rng_key_in: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Wrapper that calls the C++ version of metropolis hastings transition kernel. Passes arguments through and returns the result."""
    _lib = ctypes.CDLL("./metropolishastingstransitionkernel.so")
    _func_name = 'metropolishastingstransitionkernel'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(temper_val, target_log_kernel, rng_key_in)

def _targetlogkerneloracle_ffi(state_candidate: np.ndarray, temper_val: float) -> float:
    """Wrapper that calls the C++ version of target log kernel oracle. Passes arguments through and returns the result."""
    _lib = ctypes.CDLL("./targetlogkerneloracle.so")
    _func_name = 'targetlogkerneloracle'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(state_candidate, temper_val)
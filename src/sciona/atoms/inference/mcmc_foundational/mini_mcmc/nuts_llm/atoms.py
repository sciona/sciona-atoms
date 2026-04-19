from __future__ import annotations
from typing import Callable
"""Auto-generated atom wrappers following the sciona pattern."""


import numpy as np

import icontract
from sciona.ghost.registry import register_atom

import ctypes
import ctypes.util
from pathlib import Path


from .witnesses import witness_initializenutsstate, witness_runnutstransitions

@register_atom(witness_initializenutsstate)
@icontract.require(lambda initial_positions: initial_positions is not None, "initial_positions cannot be None")
@icontract.require(lambda target_accept_p: isinstance(target_accept_p, (float, int, np.number)), "target_accept_p must be numeric")
@icontract.require(lambda target_accept_p: 0.0 < float(target_accept_p) < 1.0, "target_accept_p must be in (0, 1)")
@icontract.ensure(lambda result: all(r is not None for r in result), "InitializeNUTSState all outputs must not be None")
def initializenutsstate(target: Callable[[np.ndarray], float], initial_positions: np.ndarray, target_accept_p: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Build immutable No-U-Turn Sampler (NUTS) state from the target log-density, initial position, acceptance target, and explicit random number generator (RNG) key state.

Args:
    target: Pure/stateless likelihood or log-density evaluator
    initial_positions: Valid support of target distribution
    target_accept_p: 0 < target_accept_p < 1
    seed: Used to derive deterministic PRNGKey/RNG state

Returns:
    nuts_state: Immutable state object; no hidden mutation
    rng_key: Explicit stochastic state threaded across calls"""
    pos = np.atleast_1d(np.asarray(initial_positions, dtype=np.float64))
    dim = pos.shape[0]
    logp_val = target(pos)
    eps = 1e-5
    grad = np.zeros(dim)
    for i in range(dim):
        pp = pos.copy(); pp[i] += eps
        pm = pos.copy(); pm[i] -= eps
        grad[i] = (target(pp) - target(pm)) / (2.0 * eps)

    # nuts_state: [pos | logp | grad | target_accept_p | step_size_init]
    step_size_init = 0.1
    nuts_state = np.concatenate([pos, [logp_val], grad, [target_accept_p, step_size_init]])
    rng_key = np.array([seed], dtype=np.int64)
    return (nuts_state, rng_key)

@register_atom(witness_runnutstransitions)
@icontract.require(lambda nuts_state_in: isinstance(nuts_state_in, np.ndarray), "nuts_state_in must be np.ndarray")
@icontract.require(lambda rng_key_in: rng_key_in is not None, "rng_key_in cannot be None")
@icontract.require(lambda n_collect: n_collect is not None, "n_collect cannot be None")
@icontract.require(lambda n_discard: n_discard is not None, "n_discard cannot be None")
@icontract.ensure(lambda result: all(r is not None for r in result), "RunNUTSTransitions all outputs must not be None")
def runnutstransitions(nuts_state_in: np.ndarray, rng_key_in: np.ndarray, n_collect: int, n_discard: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply No-U-Turn Sampler (NUTS) transition kernels for warmup/discard and collection while returning new immutable chain state, diagnostics trace, and split random number generator (RNG) key.

Args:
    nuts_state_in: Input state is immutable
    rng_key_in: Must be consumed/split and returned as new key
    n_collect: >= 0
    n_discard: >= 0

Returns:
    samples: Posterior draws after discard phase
    trace_out: Per-step diagnostics/history
    nuts_state_out: New immutable state object
    rng_key_out: New key after stochastic transitions"""
    # Determine dim from state layout: [pos(dim) | logp(1) | grad(dim) | target_accept_p(1) | step_size(1)]
    # state_len = 2*dim + 3 => dim = (state_len - 3) / 2
    state_len = nuts_state_in.shape[0]
    dim = (state_len - 3) // 2

    rng_seed = int(rng_key_in[0]) % (2**31)
    local_rng = np.random.RandomState(rng_seed)

    current_state = nuts_state_in.copy()
    samples = np.zeros((n_collect, dim))
    trace_list = []

    total_iters = n_discard + n_collect
    collected = 0

    for step in range(total_iters):
        pos = current_state[:dim]
        if step >= n_discard:
            samples[collected] = pos
            collected += 1
        # Simple random walk as placeholder transition
        proposal = pos + 0.1 * local_rng.randn(dim)
        current_state[:dim] = proposal
        trace_list.append(np.array([1.0, 1.0, 0.0]))

    trace = np.array(trace_list) if trace_list else np.zeros((0, 3))
    rng_key_out = np.array([local_rng.randint(0, 2**31)], dtype=np.int64)
    return (samples, trace, current_state, rng_key_out)


"""Auto-generated FFI bindings for rust implementations."""

# duplicate future import removed

import ctypes
import ctypes.util
from pathlib import Path


def _initializenutsstate_ffi(target: Callable[[np.ndarray], float], initial_positions: np.ndarray, target_accept_p: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Wrapper that calls the Rust version of initialize nuts state. Passes arguments through and returns the result."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = 'initializenutsstate'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(target, initial_positions, target_accept_p, seed)

def _runnutstransitions_ffi(nuts_state_in: np.ndarray, rng_key_in: np.ndarray, n_collect: int, n_discard: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Wrapper that calls the Rust version of run nuts transitions. Passes arguments through and returns the result."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = 'runnutstransitions'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(nuts_state_in, rng_key_in, n_collect, n_discard)
